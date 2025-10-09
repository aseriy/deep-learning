import argparse
import os
import signal
import threading
import asyncio
from asyncio.queues import QueueShutDown
import grpc
from gen import kmeans_pb2 as pb2
from gen import kmeans_pb2_grpc as pb2_grpc
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
import atexit
import joblib
from pathlib import Path
from datetime import datetime, UTC
import psycopg2
import time



def cluster_kmeans(model, vectors, verbose=False):
    model.partial_fit(vectors)
    labels = model.predict(vectors)
    centroids = model.cluster_centers_

    return labels, centroids


#     # TODO: This below functionality has been moved to the Kmeans Server
#     #
#     #
#     # # 3) single-batch update + labels
#     labels = cluster_kmeans(model, all_vectors, verbose=verbose)
#     centroids = model.cluster_centers_

#     if verbose:
#         print(f"[INFO] Labels: {labels}")
#         print(f"[INFO] Centroids shape: {centroids.shape}")

#     # # 4) persist in a single retriable txn
#     # epoch = persist_batch(conn, table, column, all_pks, labels, centroids, verbose=verbose, dry_run=dry_run)

#     # if verbose:
#     #     print(f"[INFO] Completed iteration (epoch={epoch})")





async def worker(name, queue_in, queue_out, batch_size, model, verbose=False):
    try:
        # Initialize the message buffer
        pks, vectors = [], []

        while True:
            try:
                pk, vector = await queue_in.get()
                if verbose:
                    print(f"[INFO] Retrieved {pk} from the queue...")

            except QueueShutDown:
                if verbose:
                    print(f"[INFO] queue shutdown -> exiting worker loop")

                break

            else:
                # Notify the queue that the "work item" has been processed.
                queue_in.task_done()
                pks.append(pk)
                vectors.append(vector)
                if verbose:
                    print(f"[INFO] Received {len(pks)} vectors, {queue_in.qsize()} messages still in the queue...")

                if len(pks) == batch_size:
                    if verbose:
                        print(f"[INFO] Calling Kmeans partial fit")


                    # TODO: cluster_kmeans() may raise exceptions like ValueError, LinAlgError, MemoryError, etc.
                    #       that should be handled here.

                    vector_np = np.array(vectors, dtype=np.float32)
                    labels, centroids = await asyncio.to_thread(
                        cluster_kmeans,
                        model,
                        vector_np,
                        verbose
                    )
                    if verbose:
                        print(f"[INFO] Updated KMeans on batch of {vector_np.shape[0]} rows")
                        print(f"[INFO] Labels: {labels}")
                        print(f"[INFO] Centroids shape: {centroids.shape}")
                        print(f"[INFO] Centroids: {centroids}")

                    # Add the results to the output queue
                    await queue_out.put((labels, centroids))


                    # Empty the buffers
                    pks, vectors = [], []



    except asyncio.CancelledError:
        if verbose:
            print("[INFO] Worker cancelled successfully")
        raise
    
    finally:
        print(f"[INFO] Worker exiting...")



class KmeansService(pb2_grpc.KmeansServicer):
    def __init__(self, model, batch_size, verbose=False):
        self.vectors_in_queue = asyncio.Queue()
        self.labels_out_queue = asyncio.Queue()
        
        # There is a single worker that runs kmeans partial fit
        self.task = asyncio.create_task(worker(
            f"kmeans",
            self.vectors_in_queue,
            self.labels_out_queue,
            batch_size,
            model,
            True)
        )


    async def cleanup(self):
        print(f"[INFO] Getting killed... cleaning up....")
        self.vectors_in_queue.shutdown()
        await self.vectors_in_queue.join()
        self.task.cancel()
        try:
            await self.task                      # <- this waits until the worker finishes
        except asyncio.CancelledError:
            print("[INFO] Worker cancelled cleanly")
        except Exception as e:
            print(f"[ERROR] Worker crashed: {e!r}")
        else:
            print("[INFO] Worker returned normally")

        # Now deal with the output queue
        # self.labels_out_queue.shutdown()
        # await self.labels_out_queue.join()




    async def PutPkVector(self, request_iterator, context):
        # do non-blocking work here (or offload CPU work; see note below)

        async for req in request_iterator:
            # print(f"[DEBUG] {req.pk}")
            # print(f"[DEBUG] {req.vector}")
            # pks, vectors = pickle.loads(request.data)
            # vectors = np.array(vectors, dtype=np.float32)
            await self.vectors_in_queue.put((req.pk, req.vector))

        return pb2.PkVectorAck(ok=True)






def get_last_saved_model(dir_path, table, column):
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)

    files = sorted(p.glob(f"{table}_{column}.*.joblib"), key=lambda f: f.name, reverse=True)

    return str(files[0]) if files else None


def save_model(dir_path, table, column, model, compress=3):
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    filename = f"{table}_{column}.{ts}.joblib"
    model_path = p.joinpath(filename)

    if compress:
        joblib.dump(model, model_path, compress=compress)
    else:
        joblib.dump(model, model_path)

    return str(model_path)


def load_model(model_path, verbose=False):
    """Load MiniBatchKMeans from file if present; else return None."""

    model = None

    if model_path and os.path.exists(model_path):
        if verbose:
            print(f"[INFO] Loading model from {model_path}")

        model = joblib.load(model_path)

    return model



def get_latest_epoch(conn, table, verbose=False) -> int:
    """Return the latest epoch from the centroid table (0 if none)."""
    sql = f"SELECT COALESCE(MAX(epoch), 0) FROM {table}"

    if verbose:
        print(sql)

    with conn.cursor() as cur:
        cur.execute(sql)
        (epoch,) = cur.fetchone()

    return int(epoch or 0)



def load_existing_centroids(conn, table, column, epoch, verbose=False):
    centroids_table = f"{table}_{column}_centroid"
    result = None
    if verbose:
        print(f"[INFO] Centroid epoch (passed): {epoch}")
    if epoch:
        sql_rows = f"SELECT id, centroid FROM {centroids_table} WHERE epoch = %s ORDER BY id"
        if verbose:
            print(sql_rows % (epoch,))
        with conn.cursor() as cur:
            cur.execute(sql_rows, (epoch,))
            rows = cur.fetchall()
        centroids = []
        for _cid, c in rows:
            if isinstance(c, list):
                vec = c
            elif isinstance(c, np.ndarray):
                vec = c.tolist()
            elif isinstance(c, str):
                vec = json.loads(c)
                if not isinstance(vec, list):
                    raise ValueError("parsed centroid JSON is not a list")
            else:
                vec = list(c)
            centroids.append(vec)
        arr = np.array(centroids, dtype=np.float32)
        if arr.size > 0:
            result = arr
    return result



# --- gRPC server ----

async def serve(stop_evt, model, batch_size, verbose=False):
    server = grpc.aio.server()
    kmeans_service = KmeansService(model, batch_size, verbose)
    pb2_grpc.add_KmeansServicer_to_server(kmeans_service, server)
    server.add_insecure_port("[::]:50051")
    
    await server.start()
    if verbose:
        print("[INFO] Async gRPC server on 50051")
    
    try:
        await stop_evt.wait()

    finally:
        # Stop servicing all RCP calls and allow 30 secs
        # for the current connections to finish.
        await server.stop(grace=30)

        # Call the server's method to stop cleanly
        await kmeans_service.cleanup()



# --- End of gRPC server ----



# --- main() ---

def main():
    parser = argparse.ArgumentParser(description="Kmeans gRPC server")
    parser.add_argument("-u", "--url", required=True, help="CockroachDB connection URL")
    parser.add_argument("-m", "--model", default="model", help="Path to directory to persist KMeans model")
    parser.add_argument("-t", "--table", required=True, help="Table name containing vectors")
    # parser.add_argument("-p", "--primary-key", required=True, help="Primary key of the table name containing vectors")
    parser.add_argument("-i", "--input", required=True, help="Column containing vector embeddings")
    parser.add_argument("-c", "--clusters", type=int, default=8, help="Number of clusters (only for KMeans)")
    parser.add_argument("-b", "--batch-size", type=int, default=1000, help="Batch size for processing rows")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output (used for debugging)")

    args = parser.parse_args()

    if args.batch_size < args.clusters:
        parser.error(f"--batch-size ({args.batch_size}) must be >= --clusters ({args.clusters}).")

    conn = psycopg2.connect(args.url)
    centroids_table = f"{args.table}_{args.input}_centroid"
    epoch = get_latest_epoch(conn, centroids_table, verbose=args.verbose)


    os.environ["OMP_NUM_THREADS"] = str(len(os.sched_getaffinity(0)))   # OpenMP (MiniBatchKMeans)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"                            # keep BLAS single-threaded
    os.environ["MKL_NUM_THREADS"] = "1"

    if args.verbose:
        print(f"[INFO] Utilizing {os.environ['OMP_NUM_THREADS']} CPU cores")

    # Prefer file resume; else warm-start from latest DB centroids; else fresh
    model_path = get_last_saved_model(args.model, args.table, args.input)
    if args.verbose:
        print(f"[INFO] Loading last saved mode from {model_path}")

    model = None
    model = load_model(model_path, verbose=args.verbose)
    if args.verbose:
        print(f"[INFO] Model: {model}")

    # Save once on process exit
    # register_save_on_exit(model_path, get_model=lambda: model, verbose=args.verbose)


    if model is None:
        initial = load_existing_centroids(
                                conn,
                                args.table, args.input,
                                epoch=epoch,
                                verbose=args.verbose
                    )
        model = MiniBatchKMeans(
            n_clusters=args.clusters,
            init=initial if initial is not None else 'k-means++',
            batch_size=args.batch_size,
            random_state=42,
            n_init=1 if initial is not None else 10
        )


    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop_evt = asyncio.Event()
    loop.add_signal_handler(signal.SIGTERM, stop_evt.set)
    loop.add_signal_handler(signal.SIGINT,  stop_evt.set)
    
    try:
        loop.run_until_complete(serve(stop_evt, model, args.batch_size, args.verbose))

        if args.verbose:
            print(f"[INFO] Saving mode to {model_path}")

        model_path = save_model(args.model, args.table, args.input, model)

    finally:
        loop.close()


    conn.close()

# --- End of main() ---




if __name__ == "__main__":
    main()