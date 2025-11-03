#!/usr/bin/env python3

import argparse
import json
import os
import sys
import numpy as np
import multiprocessing as mp
import signal
import time, random
from psycopg2 import errors
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool
from pathlib import Path
from datetime import datetime, UTC
import grpc
from gen import kmeans_pb2 as pb2
from gen import kmeans_pb2_grpc as pb2_grpc
from google.protobuf import empty_pb2
from concurrent.futures import ThreadPoolExecutor




def worker_cluster_assigner(parent_conn, url, table, column, verbose=False):
    concurrency = len(os.sched_getaffinity(0))
    
    pool = ThreadedConnectionPool(concurrency, 3*concurrency, dsn=url)
    stream = None

    if verbose:
        print(f"[INFO] Starting cluster assigner worker")

    # Close over 'parent_conn' so handlers can safely unblock recv()
    def _shutdown_handler(signum, frame):
        if verbose:
            print(f"[INFO] Cluster assigner worker draining...")

        try:
            stream is None or stream.cancel()
        except Exception:
            pass  # already closed or race


    # Trap SIGTERM (prod) and SIGINT (local Ctrl-C) for a quiet exit
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT,  _shutdown_handler)

    try:
        with grpc.insecure_channel("localhost:50051") as channel:
            stub = pb2_grpc.KmeansStub(channel)
            stream = stub.GetPkCentroids(empty_pb2.Empty(), wait_for_ready=True, timeout=None)
            for batch in stream:
                pks = list(batch.pks)
                labels = list(batch.labels)
                centroids = [list(c.feature) for c in batch.centroids]
                if verbose:
                    print(f"[INFO] PKs: {pks}")
                    print(f"[INFO] Labels: {labels}")
                    print(f"[INFO] Centroids: {centroids}")

                # persist in a single retriable txn
                conn = pool.getconn()
                epoch = persist_batch(
                            conn,
                            table, column,
                            pks, labels, centroids,
                            verbose=verbose
                        )
                if verbose:
                    print(f"[INFO] Completed iteration (epoch={epoch})")
                pool.putconn(conn)


            if verbose:
                if stream.code() == grpc.StatusCode.OK:
                    print(f"[INFO] gRPC closed stream normally: {stream.code()}")
                else:
                    print(f"[INFO] gRPC closed stream with error: {stream.details()}")


    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.CANCELLED:
            if verbose:
                print(f'[INFO] Cluster assigner worker exiting...')

            # TODO: We're handling SIGTERM here. By default exists with SIGTERM (-15)
            #       If a different exit code is desired, un-comment and exist with a
            #       desired code.
            sys.exit(130)

    finally:
        # Clean up before exiting...
        pool.closeall()




def get_latest_epoch(conn, table, verbose=False) -> int:
    """Return the latest epoch from the centroid table (0 if none)."""
    sql = f"SELECT COALESCE(MAX(epoch), 0) FROM {table}"
    if verbose:
        print(sql)
    with conn.cursor() as cur:
        cur.execute(sql)
        (epoch,) = cur.fetchone()
    return int(epoch or 0)


def bucket_vector_pks(conn, concurrency, table, pk, batch_size, verbose=False):

    sql = f"""
        WITH b AS (                                                                   
            SELECT {pk}, NTILE({concurrency}) OVER (ORDER BY {pk}) AS bucket                         
            FROM {table}                                                                
        )                                                                             
        SELECT
            bucket,
            MIN({pk}) AS start_pk,
            MAX({pk}) AS end_pk,
            COUNT(*) AS rows_in_bucket
            FROM b
            GROUP BY bucket
            ORDER BY bucket
    """

    if verbose:
        print(f"[INFO] {sql}")

    buckets = None
    with conn.cursor() as cur:
        cur.execute(sql)
        buckets = cur.fetchall()

    # print(json.dumps(rows, indent=2))
    return buckets



def fetch_vectors(
            conn, table, pk, column,
            pk_start, pk_end,
            batch_size, verbose=False):

    centroids_table = f"{table}_{column}_centroid"
    clusters_table  = f"{table}_{column}_clusters"
    # batch_size_multiplier = 10


    sql = f"""
            SELECT s.{pk}, s.{column}
            FROM {table} AS s
            WHERE s.{column} IS NOT NULL
            AND s.{pk} BETWEEN %s AND %s
            AND NOT EXISTS (
                SELECT 1 FROM {clusters_table} c
                WHERE c.{pk} = s.{pk}                  
            )
            ORDER BY s.{pk}                                                
            LIMIT {batch_size}
        """

    if verbose:
        print("[INFO] ", sql % (f"'{pk_start}'", f"'{pk_end}"))

    rows = None
    with conn.cursor() as cur:
        cur.execute(sql, (pk_start,pk_end))
        rows = cur.fetchall()

    return rows






# --- transactional retry + batch persistence helpers ---

def run_txn_with_retry(conn, fn, max_retries=8, base_sleep=0.05, jitter=0.05):
    """
    Runs `fn(cur)` inside a transaction and retries on CockroachDB serializable conflicts.
    `fn` receives a cursor and must do only transactional work (no external side effects).
    """
    attempt = 0
    while True:
        try:
            with conn:  # opens a transaction, commits/rolls back automatically
                with conn.cursor() as cur:
                    return fn(cur)
        except errors.SerializationFailure:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, jitter)
            if attempt == 1:
                print(f"[WARN] txn conflict; retrying (attempt {attempt})")
            time.sleep(sleep)
        except Exception:
            raise


def persist_batch(conn, table, column, pks, labels, centroids, verbose=False):
    """
    Atomically:
      - allocate fresh epoch
      - upsert all centroids for that epoch
      - upsert cluster assignments for all pks with that epoch
    All inside a single retriable transaction.
    Returns the epoch used.
    """
    centroids_table = f"{table}_{column}_centroid"
    clusters_table  = f"{table}_{column}_clusters"
    seq_name        = f"{table}_{column}_centroid_seq"

    def _txn(cur):
        # 1) epoch
        cur.execute("SELECT nextval(%s)", (seq_name,))
        (epoch,) = cur.fetchone()
        if verbose:
            print(f"[INFO] Using epoch {epoch}")

        # 2) centroids
        centroid_rows = []
        for i, c in enumerate(centroids):
            vec = c.tolist() if isinstance(c, np.ndarray) else (c if isinstance(c, list) else list(c))
            centroid_rows.append((epoch, i, vec))
        if centroid_rows:
            sql_cent = f"UPSERT INTO {centroids_table} (epoch, id, centroid) VALUES %s"
            execute_values(cur, sql_cent, centroid_rows)
            if verbose:
                print(f"[INFO] Upserted {len(centroid_rows)} centroids")

        # 3) assignments
        assign_rows = [(pk, epoch, int(label)) for pk, label in zip(pks, labels)]
        if assign_rows:
            sql_assign = f"UPSERT INTO {clusters_table} (pid, epoch, cluster_id) VALUES %s"
            execute_values(cur, sql_assign, assign_rows)
            if verbose:
                print(f"[INFO] Upserted {len(assign_rows)} assignments")

        return epoch

    return run_txn_with_retry(conn, _txn)





def run_kmeans_iteration(
                conn,
                table, pk, column,
                bucket,
                batch_size, verbose, k
            ):
    """One full KMeans iteration: fetch → partial_fit → predict → save epoch + assignments."""
    
    rows = fetch_vectors (
                conn, table, pk, column, bucket[1], bucket[2],
                batch_size=batch_size, verbose=verbose
            )

    if len(rows) == 0:
        if verbose:
            print("[INFO] No more vectors to process.")
        return None

    # Shuffle client-side to remove pid-order bias
    # Use numpy RNG for reproducibility
    rng = np.random.default_rng()
    rng.shuffle(rows)  # in-place

    pks, vectors = [], []
    for pk_value, v in rows:
        try:
            if isinstance(v, list):
                vec = v
            elif isinstance(v, np.ndarray):
                vec = v.tolist()
            elif isinstance(v, str):
                vec = json.loads(v)
                if not isinstance(vec, list):
                    raise ValueError("parsed JSON is not a list")
            else:
                vec = list(v)

            pks.append(pk_value)
            vectors.append(vec)

        except Exception as e:
            if verbose:
                print(f"[WARN] Skipping row {pk_value}: {e}")

    if verbose:
        print(f"[INFO] Fetched {len(rows)} rows")

    with grpc.insecure_channel("localhost:50051") as channel:
        def request_iter():
            for pk, vector in zip(pks, vectors):
                yield pb2.PkVector(pk=pk, vector=vector)

        stub = pb2_grpc.KmeansStub(channel)

        resp = None
        try:
            resp = stub.PutPkVector(request_iter(), timeout=60)
            print(resp)

        except grpc.RpcError:
            if verbose:
                print(f"[INFO] gRPC server closed connection.")
            pass

    return False if resp is None else True




# --- main ---

def main():
    parser = argparse.ArgumentParser(description="Cluster vector column from CockroachDB using KMeans")
    parser.add_argument("-u", "--url", required=True, help="CockroachDB connection URL")
    parser.add_argument("-t", "--table", required=True, help="Table name containing vectors")
    parser.add_argument("-p", "--primary-key", required=True, help="Primary key of the table name containing vectors")
    parser.add_argument("-i", "--input", required=True, help="Column containing vector embeddings")
    parser.add_argument("-c", "--clusters", type=int, default=8, help="Number of clusters (only for KMeans)")

    # TODO: this option isn't being used at this time.
    #       In the future, may be used a the CPU multiplier. Right now, the number of threads is based on the number of
    #       available CPU's.
    parser.add_argument("-w", "--workers", type=int, default=mp.cpu_count(), help="Number of parallel workers (not yet used)")
    
    parser.add_argument("-b", "--batch-size", type=int, default=1000, help="Batch size for processing rows")
    parser.add_argument("-n", "--num-batches", type=int, default=None, help="Limit number of batches to process (default: all)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output (used for debugging)")
    parser.add_argument("-m", "--model", default="model", help="Path to directory to persist KMeans model")

    args = parser.parse_args()

    if args.batch_size < args.clusters:
        parser.error(f"--batch-size ({args.batch_size}) must be >= --clusters ({args.clusters}).")

    concurrency = len(os.sched_getaffinity(0))

    # Fork a separate child process to receive the Kmeans results
    # and update the vector-to-centroid mapping.
    # This process will create its own DB connection pool.
    mp.set_start_method("forkserver")
    parent_conn, child_conn = mp.Pipe(duplex=True)
    p = mp.Process(target=worker_cluster_assigner, args=(child_conn, args.url, args.table, args.input, args.verbose))
    p.start()
    # close child's end in parent
    child_conn.close()              


    pool = ThreadedConnectionPool(concurrency, 3*concurrency, dsn=args.url)
    remaining = args.num_batches
    shutting_down = False

    # BEGIN graceful shutdown setup
    # 1. Stop fetching vectors and feeding to the gRPC server.
    # 2. Tell the worker_cluster_assigner() to procecc until nothing else is coming
    #    from the server. Then exit.
    # 3. Wait for the child process to exit for X secs. If the child hasn't existed by then,
    #.   kill it.

    def _shutdown_handler(signum, frame):
        nonlocal shutting_down, remaining

        if args.verbose:
            print(f"[INFO] Shutting down {__file__}")

        try:
            # Flag to run_kmeans_iteration() to fold
            remaining = None

            # If we're still in the process of shutting down gracefully
            if not shutting_down:
                shutting_down = True
                # Tell the child process to fold and wait for it to exit
                # p.terminate()
                # while p.is_alive(): p.join(timeout=0.5)
            else:
                p.kill()
                sys.exit(130)
 
        except Exception:
            pass  # already closed or race

    # Trap SIGTERM (prod) and SIGINT (local Ctrl-C) for a quiet exit
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT,  _shutdown_handler)

    # END OF graceful shutdown setup



    centroids_table = f"{args.table}_{args.input}_centroid"
    conn = pool.getconn()
    epoch = get_latest_epoch(conn, centroids_table, verbose=args.verbose)
    pool.putconn(conn)

    # In order to build vector batched in parallel, we bucket the entire
    # {table}.{primary_key} value space into the number of buckets, where
    # number of buckets is a multiple of the number of CPU's available to
    # this process.


    # remaining / shutting_down logic:
    # remaining = None | args.num_batches
    # None.           ==> Keep going forever until killed.
    #                     The signal handler will set shutting_down to True.
    # args.num_baches ==> fetch the specified number of batches and stop.
    #                     Start the max number of threads possible, either 'concurrency' or remaining, whichever is less.
    #                     Once completed, subtract from remaining and continue.
    #                     When remaining == 0, set shutting_down to True.

    while not shutting_down:
        if remaining is not None and remaining < 1:
            if args.verbose:
                print(f"[INFO] Reached requested iterations: {args.num_batches}")
            shutting_down = True
            break


        if args.verbose:
            print(f"[INFO] Splitting all un-associated vectors into {concurrency} buckets")

        conn = pool.getconn()
        pk_buckets = bucket_vector_pks(conn, concurrency, args.table, args.primary_key, args.batch_size, args.verbose)
        pool.putconn(conn)

        for bucket in pk_buckets:
            conn = pool.getconn()
            success = run_kmeans_iteration(
                conn,
                args.table,
                args.primary_key,
                args.input,
                bucket,
                batch_size=args.batch_size,
                verbose=args.verbose,
                k=args.clusters
            )
            pool.putconn(conn)
            
            if not success:
                shutting_down = True
                break

        if remaining is not None:
            remaining -= 1


    # At this point, we're not feeding any new vectors to the gRPC server.
    # Send a termination to the child process and wait for it exit.
    p.terminate()
    while p.is_alive(): p.join(timeout=0.5)
    # p.join()
    # TODO: The above line waits for the child process running worker_cluster_assigner() to exit.
    #       If any logic around the child exit code is required, un-comment below and implement.
    # p.exitcode

    pool.closeall()

if __name__ == "__main__":
    main()
