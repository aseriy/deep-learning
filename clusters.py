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


def get_primary_key_column(conn, table: str, verbose: bool = False) -> str:
    """Return the primary key column name for `table` (first column if composite)."""
    
    sql = f"""
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
         AND tc.table_name   = kcu.table_name
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_name = '%s'
        ORDER BY kcu.ordinal_position
        LIMIT 1
    """
    if verbose:
        print(sql % (repr(table),))

    with conn.cursor() as cur:
        cur.execute(sql, (table,))
        row = cur.fetchone()

    if not row:
        raise RuntimeError(f"No primary key found for table '{table}'")

    return row[0]


def get_current_epoch(conn, table, verbose=False) -> int:
    """Return the latest epoch from the centroid table (0 if none)."""
    
    sql = f"SELECT COALESCE(MAX(epoch), 0) FROM {table}"
    
    if verbose:
        print(sql
    
    with conn.cursor() as cur:
        cur.execute(sql, (table,))
        (epoch,) = cur.fetchone()
    
    return int(epoch or 0)


def fetch_vectors(conn, table, column, batch_size=None, num_batches=None, verbose=False):
    """
    Fetch (pid, vector) for rows in `table` whose vector column is NOT NULL
    and that do NOT yet have a mapping in {table}_{column}_clusters for the
    latest epoch. Returns (pids: List[str], vectors: np.ndarray[float32, N×D]).
    """

    centroids_table = f"{table}_{column}_centroids"
  
    pk = get_primary_key_column(conn, table, verbose=verbose)
    latest_epoch = get_current_epoch(conn, centroids_table, verbose=verbose)

    return np.array([], dtype=np.float32)


def save_centroids(conn, table, column, centroids, increment, verbose, dry_run):
    cur = conn.cursor()
    cur.execute(f"SELECT nextval('%s_%s_centroid_seq')" % (table, column))
    cluster_version = cur.fetchone()[0]

    if verbose:
        print("\n[RESULT] Cluster centroids:")
    
    for i, centroid in enumerate(centroids):
        version_to_use = cluster_version if not increment else f"(SELECT MAX(version) FROM {table}_{column}_centroid)"

        sql = f"UPSERT INTO {table}_{column}_centroid (version, id, centroid) VALUES ({version_to_use}, %s, %s)"

        if verbose:
            print(sql % (i, centroid.tolist()))

        if dry_run:
            print(f"[INFO] Upserting centroid {i} to version {version_to_use}: {centroid.tolist()}")

        cur.execute(sql, (i, centroid.tolist()))

    conn.commit()
    cur.close()


def load_existing_centroids(conn, table, column, verbose=False):
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(version) FROM {table}_{column}_centroid")
        latest_version = cur.fetchone()[0]
        cur.execute(f"SELECT centroid FROM {table}_{column}_centroid WHERE version = %s ORDER BY id", (latest_version,))
        rows = cur.fetchall()
    if verbose:
        for i, row in enumerate(rows):
            print(f"[INFO] Loaded centroid {i}: {row[0]}")
    return np.array([ast.literal_eval(r[0]) if isinstance(r[0], str) else r[0] for r in rows], dtype=np.float32)


def cluster_kmeans(vectors, num_clusters, num_batches=None, initial_centroids=None, batch_size=1000):
    model = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=batch_size,
        random_state=42,
        init=initial_centroids if initial_centroids is not None else 'k-means++',
        n_init='auto' if initial_centroids is None else 1
    )
    batch_count = 0
    for i in range(0, len(vectors), model.batch_size):
        if num_batches is not None and batch_count >= num_batches:
            break
        batch = vectors[i:i+model.batch_size]
        model.partial_fit(batch)
        print(f"[INFO] Updated with batch {batch_count + 1} ({len(batch)} vectors)")
        batch_count += 1
    labels = model.predict(vectors)
    return labels, model


def cluster_dbscan(vectors):
    model = DBSCAN()
    labels = model.fit_predict(vectors)
    return labels, None


def main():
    parser = argparse.ArgumentParser(description="Cluster vector column from CockroachDB using KMeans or DBSCAN.")
    parser.add_argument("-u", "--url", required=True, help="CockroachDB connection URL")
    parser.add_argument("-t", "--table", required=True, help="Table name containing vectors")
    parser.add_argument("-i", "--input", required=True, help="Column containing vector embeddings")
    parser.add_argument("-a", "--algorithm", choices=["kmeans", "dbscan"], default="kmeans", help="Clustering algorithm to use")
    parser.add_argument("-c", "--clusters", type=int, default=8, help="Number of clusters (only for KMeans)")
    parser.add_argument("-w", "--workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers (not yet used)")
    parser.add_argument("-b", "--batch-size", type=int, default=1000, help="Batch size for processing rows")
    parser.add_argument("-n", "--num-batches", type=int, default=None, help="Limit number of batches to process (default: all)")

    group = parser.add_argument_group()
    group.add_argument("-v", "--verbose", action="store_true", help="Verbose output (used for debugging)")
    group.add_argument("-p", "--progress", action="store_true", help="Show progress bar")

    parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--increment", action="store_true", help="Update latest centroid version incrementally instead of creating a new one")
    args = parser.parse_args()

    conn = psycopg2.connect(args.url)
    vectors = fetch_vectors(conn, args.table, args.input, batch_size=args.batch_size, num_batches=args.num_batches, verbose=args.verbose)
    exit(0)
  
    if args.progress:
        vectors = tqdm(vectors, desc="Clustering", unit="vec")  # Just visual aid, does not affect clustering

    if args.algorithm == "kmeans":
        initial_centroids = load_existing_centroids(conn, args.table, args.input, verbose=args.verbose) if args.increment else None
        labels, model = cluster_kmeans(np.array(vectors), args.clusters, args.num_batches, initial_centroids, batch_size=args.batch_size)
        centroids = model.cluster_centers_
    else:
        labels, centroids = cluster_dbscan(np.array(vectors))

    if centroids is not None:
        save_centroids(conn, args.table, args.input, centroids, args.increment, args.verbose, args.dry_run)
    else:
        print("DBSCAN does not generate explicit centroids.")

    conn.close()


if __name__ == "__main__":
    main()
