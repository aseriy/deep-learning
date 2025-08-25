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


def get_current_epoch(conn, table, verbose=False) -> int:
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
    latest_epoch = get_current_epoch(conn, centroids_table, verbose=verbose)
    if verbose:
        print(f"[INFO] Latest epoch: {latest_epoch}")

    sql = f"""
        SELECT s.{pk} AS pk, s.{column} AS vec
        FROM {table} AS s
        LEFT JOIN {clusters_table} AS c
          ON c.pid = s.{pk}
        WHERE s.{column} IS NOT NULL
          AND c.pid IS NULL
        ORDER BY s.{pk} ASC
        LIMIT {batch_size}
    """
    if verbose:
        print(sql)

    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

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


def load_existing_centroids(conn, table, column, verbose=False):
    centroids_table = f"{table}_{column}_centroid"
    result = None
    latest_epoch = get_current_epoch(conn, centroids_table, verbose=verbose)
    if verbose:
        print(f"[INFO] Latest centroid epoch: {latest_epoch}")
    if latest_epoch:
        sql_rows = f"SELECT id, centroid FROM {centroids_table} WHERE epoch = %s ORDER BY id"
        if verbose:
            print(sql_rows % (latest_epoch,))
        with conn.cursor() as cur:
            cur.execute(sql_rows, (latest_epoch,))
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


# --- clustering helpers ---

def build_kmeans_model(k, batch_size, initial_centroids=None):
    return MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        random_state=42,
        init=initial_centroids if initial_centroids is not None else 'k-means++',
        n_init='auto' if initial_centroids is None else 1,
    )


def cluster_kmeans(model, vectors, verbose=False):
    model.partial_fit(vectors)
    labels = model.predict(vectors)
    if verbose:
        print(f"[INFO] Updated KMeans on batch of {vectors.shape[0]} rows")
    return labels


def run_kmeans_iteration(conn, table, pk, column, batch_size, verbose, dry_run, k, use_increment):
    """One full KMeans iteration: fetch → init model (fresh) → partial_fit → predict → save epoch + assignments."""
    # 1) fetch one DB batch
    pks, vectors = fetch_vectors(conn, table, pk, column, batch_size=batch_size, verbose=verbose)
    if vectors.size == 0:
        if verbose:
            print("[INFO] No more vectors to process.")
        return False

    if verbose:
        print(f"[INFO] Fetched {vectors.shape[0]} rows")

    # 2) fresh model INIT each batch, optionally seeded from latest centroids
    initial = load_existing_centroids(conn, table, column, verbose=verbose) if use_increment else None
    model = build_kmeans_model(k, batch_size, initial)

    # 3) single-batch update + labels
    labels = cluster_kmeans(model, vectors, verbose=verbose)
    centroids = model.cluster_centers_

    if verbose:
        print(f"[INFO] Labels: {labels}")
        print(f"[INFO] Centroids shape: {centroids.shape}")

    # 4) new epoch + assignments
    epoch = save_centroids(conn, table, column, centroids, increment=True, verbose=verbose, dry_run=dry_run)
    save_cluster_assignments(conn, table, column, epoch=epoch, pks=pks, labels=labels, verbose=verbose, dry_run=dry_run)

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
    parser.add_argument("--increment", action="store_true", help="Update latest centroid version incrementally instead of creating a new one")
    args = parser.parse_args()

    if args.batch_size < args.clusters:
        parser.error(f"--batch-size ({args.batch_size}) must be >= --clusters ({args.clusters}).")

    conn = psycopg2.connect(args.url)

    if args.algorithm == "kmeans":
        remaining = args.num_batches
        while True:
            if remaining is not None and remaining <= 0:
                if args.verbose:
                    print(f"[INFO] Reached requested iterations: {args.num_batches}")
                break
    
            processed = run_kmeans_iteration(
                conn, args.table, args.primary_key, args.input,
                batch_size=args.batch_size,
                verbose=args.verbose,
                dry_run=args.dry_run,
                k=args.clusters,
                use_increment=args.increment,
            )
            if not processed:
                break
    
            if remaining is not None:
                remaining -= 1

    else:
        pks, vectors = fetch_vectors(conn, args.table, args.primary_key, args.input, batch_size=args.batch_size, verbose=args.verbose)
        if vectors.size == 0:
            if args.verbose:
                print("[INFO] No vectors to cluster; exiting.")
            conn.close()
            sys.exit(0)
        labels, _ = cluster_dbscan(vectors)
        if args.verbose:
            print("DBSCAN does not generate explicit centroids.")
    conn.close()


if __name__ == "__main__":
    main()
