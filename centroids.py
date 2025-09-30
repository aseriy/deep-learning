#!/usr/bin/env python3

import argparse
import json
import os
import sys
import psycopg2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, DBSCAN
import multiprocessing
import ast
import time, random
from psycopg2 import errors
from psycopg2.extras import execute_values
import atexit
import joblib
from pathlib import Path
from datetime import datetime, UTC


def get_latest_epoch(conn, table, verbose=False) -> int:
    """Return the latest epoch from the centroid table (0 if none)."""
    sql = f"SELECT COALESCE(MAX(epoch), 0) FROM {table}"
    if verbose:
        print(sql)
    with conn.cursor() as cur:
        cur.execute(sql)
        (epoch,) = cur.fetchone()
    return int(epoch or 0)


# --- data access helpers ---

def fetch_vectors(conn, table, pk, column, batch_size, verbose=False):
    centroids_table = f"{table}_{column}_centroid"
    clusters_table  = f"{table}_{column}_clusters"
    batch_size_multiplier = 10

    sql_0 = f"SELECT MAX({pk}) FROM {clusters_table}"

    rows = []
    with conn.cursor() as cur:
        cur.execute(sql_0)
        (max_pk,) = cur.fetchone()

        if verbose:
            print(f"[INFO] Search for candidate vectors from {max_pk} on...")

        sql_max_pk = f"AND s.{pk} > %s" if max_pk is not None else ""

        sql = f"""
            SELECT s.{pk}, s.{column}                                                
            FROM {table} AS s                                                             
            WHERE s.{column} IS NOT NULL  
            {sql_max_pk}                                    
            AND NOT EXISTS (                                                            
            SELECT 1 FROM {clusters_table} c                           
            WHERE c.{pk} = s.{pk}                                                       
            )                                                                           
            ORDER BY s.{pk}                                                             
            LIMIT {batch_size * batch_size_multiplier}
        """

        if verbose:
            print(sql % (max_pk,))

        cur.execute(sql, (max_pk,))
        rows = cur.fetchall()

    # Shuffle client-side to remove pid-order bias
    # Use numpy RNG for reproducibility
    rng = np.random.default_rng()
    rng.shuffle(rows)  # in-place
    rows = rows[:batch_size]

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

    return pks, np.array(vectors, dtype=np.float32)


def save_centroids(conn, table, column, centroids, increment, verbose, dry_run):
    centroids_table = f"{table}_{column}_centroid"
    seq_name = f"{table}_{column}_centroid_seq"
    with conn.cursor() as cur:
        cur.execute(f"SELECT nextval('{seq_name}')")
        (target_epoch,) = cur.fetchone()
    if verbose:
        print(f"[INFO] Saving {len(centroids)} centroids into epoch {target_epoch}")
    sql = f"UPSERT INTO {centroids_table} (epoch, id, centroid) VALUES (%s, %s, %s)"
    with conn.cursor() as cur:
        for i, c in enumerate(centroids):
            vec = c.tolist() if isinstance(c, np.ndarray) else (c if isinstance(c, list) else list(c))
            if verbose:
                print(sql % (target_epoch, i, vec))
            if not dry_run:
                cur.execute(sql, (target_epoch, i, vec))
    if not dry_run:
        conn.commit()
    return target_epoch


def save_cluster_assignments(conn, table, column, epoch, pks, labels, verbose=False, dry_run=False):
    clusters_table = f"{table}_{column}_clusters"
    sql = f"UPSERT INTO {clusters_table} (pid, epoch, cluster_id) VALUES (%s, %s, %s)"
    with conn.cursor() as cur:
        for pk, label in zip(pks, labels):
            cluster_id = int(label)
            if verbose:
                print(sql % (pk, epoch, cluster_id))
            if not dry_run:
                cur.execute(sql, (pk, epoch, cluster_id))
    if not dry_run:
        conn.commit()


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


# --- file-based model persistence (minimal) ---

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


def persist_batch(conn, table, column, pks, labels, centroids, verbose=False, dry_run=False):
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

    if dry_run:
        simulated_epoch = -1
        if verbose:
            print(f"[DRY RUN] Would select nextval('{seq_name}') → epoch X")
            print(f"[DRY RUN] Would UPSERT {len(centroids)} centroids into {centroids_table}")
            print(f"[DRY RUN] Would UPSERT {len(pks)} assignments into {clusters_table}")
        return simulated_epoch

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


# --- clustering helpers ---

def build_kmeans_model(k, batch_size, initial_centroids=None, restarts=10, seed=42):
    return MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        random_state=seed,
        init=initial_centroids if initial_centroids is not None else 'k-means++',
        n_init=1 if initial_centroids is not None else restarts
    )


def cluster_kmeans(model, vectors, verbose=False):
    model.partial_fit(vectors)
    labels = model.predict(vectors)

    if verbose:
        print(f"[INFO] Updated KMeans on batch of {vectors.shape[0]} rows")
    
    return labels


def run_kmeans_iteration(conn, model, table, pk, column, batch_size, verbose, dry_run, k):
    """One full KMeans iteration: fetch → partial_fit → predict → save epoch + assignments."""
    
    # 1) fetch one DB batch
    pks, vectors = fetch_vectors(conn, table, pk, column, batch_size=batch_size, verbose=verbose)
    if vectors.size == 0:
        if verbose:
            print("[INFO] No more vectors to process.")
        return False

    if verbose:
        print(f"[INFO] Fetched {vectors.shape[0]} rows")

    # 3) single-batch update + labels
    labels = cluster_kmeans(model, vectors, verbose=verbose)
    centroids = model.cluster_centers_

    if verbose:
        print(f"[INFO] Labels: {labels}")
        print(f"[INFO] Centroids shape: {centroids.shape}")

    # 4) persist in a single retriable txn
    epoch = persist_batch(conn, table, column, pks, labels, centroids, verbose=verbose, dry_run=dry_run)

    if verbose:
        print(f"[INFO] Completed iteration (epoch={epoch})")

    return True


def cluster_dbscan(vectors):
    model = DBSCAN()
    labels = model.fit_predict(vectors)
    return labels, None


# --- main ---

def main():
    parser = argparse.ArgumentParser(description="Cluster vector column from CockroachDB using KMeans or DBSCAN.")
    parser.add_argument("-u", "--url", required=True, help="CockroachDB connection URL")
    parser.add_argument("-t", "--table", required=True, help="Table name containing vectors")
    parser.add_argument("-p", "--primary-key", required=True, help="Primary key of the table name containing vectors")
    parser.add_argument("-i", "--input", required=True, help="Column containing vector embeddings")
    parser.add_argument("-a", "--algorithm", choices=["kmeans", "dbscan"], default="kmeans", help="Clustering algorithm to use")
    parser.add_argument("-c", "--clusters", type=int, default=8, help="Number of clusters (only for KMeans)")
    parser.add_argument("-w", "--workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers (not yet used)")
    parser.add_argument("-b", "--batch-size", type=int, default=1000, help="Batch size for processing rows")
    parser.add_argument("-n", "--num-batches", type=int, default=None, help="Limit number of batches to process (default: all)")

    group = parser.add_argument_group()
    group.add_argument("-v", "--verbose", action="store_true", help="Verbose output (used for debugging)")
    group.add_argument("--progress", action="store_true", help="Show progress bar")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("-m", "--model", default="model", help="Path to directory to persist KMeans model")

    args = parser.parse_args()

    if args.batch_size < args.clusters:
        parser.error(f"--batch-size ({args.batch_size}) must be >= --clusters ({args.clusters}).")

    conn = psycopg2.connect(args.url)

    centroids_table = f"{args.table}_{args.input}_centroid"
    epoch = get_latest_epoch(conn, centroids_table, verbose=args.verbose)

    if args.algorithm == "kmeans":

        # Prefer file resume; else warm-start from latest DB centroids; else fresh
        model_path = get_last_saved_model(args.model, args.table, args.input)
        print(model_path)
        # exit(0)

        model = None
        model = load_model(model_path, verbose=args.verbose)
        print("Model: ", model)

        # Save once on process exit
        # register_save_on_exit(model_path, get_model=lambda: model, verbose=args.verbose)


        if model is None:
            initial = load_existing_centroids(
                                    conn,
                                    args.table, args.input,
                                    epoch=epoch,
                                    verbose=args.verbose
                        )
            model = build_kmeans_model(args.clusters, args.batch_size, initial)

        remaining = args.num_batches
        while True:
            if remaining is not None and remaining <= 0:
                if args.verbose:
                    print(f"[INFO] Reached requested iterations: {args.num_batches}")
                break
    
            processed = run_kmeans_iteration(
                conn, 
                model,
                args.table, args.primary_key, args.input,
                batch_size=args.batch_size,
                verbose=args.verbose,
                dry_run=args.dry_run,
                k=args.clusters
            )
            if not processed:
                break
    
            if remaining is not None:
                remaining -= 1

        model_path = save_model(args.model, args.table, args.input, model)
        print(model_path)

    else:
        print("DBSCAN is not implemented yet...")

    conn.close()


if __name__ == "__main__":
    main()
