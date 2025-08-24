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


def fetch_vectors(conn,
                  table, pk, column,
                  batch_size=None, num_batches=None,
                  verbose=False):
    """
    Fetch (pk, vector) for rows in `table` whose vector column is NOT NULL
    and that do NOT yet have a mapping in {table}_{column}_clusters for the
    latest epoch. Returns np.ndarray[float32, N×D].
    """

    centroids_table = f"{table}_{column}_centroid"
    clusters_table = f"{table}_{column}_clusters"
  
    latest_epoch = get_current_epoch(conn, centroids_table, verbose=verbose)
    if verbose:
        print(f"[INFO] Latest epoch: {latest_epoch}")

    total_limit = None
    if batch_size is not None and num_batches is not None:
        total_limit = batch_size * num_batches
    elif batch_size is not None:
        total_limit = batch_size

    limit_clause = f"LIMIT {int(total_limit)}" if total_limit is not None else ""
    
    sql = f"""
        SELECT s.{pk} AS pk, s.{column} AS vec
        FROM {table} AS s
        LEFT JOIN {clusters_table} AS c
          ON c.pid = s.{pk}
        WHERE s.{column} IS NOT NULL
          AND c.pid IS NULL
        ORDER BY s.{pk} ASC
        {limit_clause}
    """
  
    if verbose:
        print(sql)

    with conn.cursor() as cur:
        cur.execute(sql, (latest_epoch,))
        rows = cur.fetchall()

    pks = []
    vectors = []

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
                vec = list(v)  # fallback, e.g. psycopg2 VECTOR adapts to Python list
            pks.append(pk_value)
            vectors.append(vec)
        except Exception as e:
            if verbose:
                print(f"[WARN] Skipping row {pk_value}: {e}")

    return pks, np.array(vectors, dtype=np.float32)


def save_centroids(conn, table, column, centroids, increment, verbose, dry_run):
    """
    Save centroids into {table}_{column}_centroid with epoch policy:
      - First ever save → epoch = 0
      - Every subsequent save → epoch = nextval({table}_{column}_centroid_seq)

    Note: `increment` only affects whether caller used prior centroids as init.
          We *always* create a new epoch when we save (centroids shifted).
    """
    centroids_table = f"{table}_{column}_centroid"
    seq_name = f"{table}_{column}_centroid_seq"

    # Allocate a fresh epoch from the sequence (first call yields whatever you set via RESTART WITH)
    target_epoch = 0
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



def save_cluster_assignments(
        conn,
        table, column,
        epoch, pks, labels,
        verbose=False, dry_run=False):
    """
    Save row→centroid assignments into {table}_{column}_clusters for a given epoch.
    Expects:
      - epoch: int   (the epoch you just used for centroids)
      - pks:   list  (primary-key values aligned with `labels`)
      - labels: array-like of ints (cluster ids aligned with `pks`)
    """
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
    """
    Load centroids for the latest epoch from {table}_{column}_centroid.
    Returns: np.ndarray[float32, K×D] or None if none exist.
    """
    centroids_table = f"{table}_{column}_centroid"
    result = None

    # 1) Get latest epoch
    latest_epoch = get_current_epoch(conn, centroids_table, verbose=verbose)

    if verbose:
        print(f"[INFO] Latest centroid epoch: {latest_epoch}")

    if latest_epoch:
        # 2) Fetch centroids for that epoch
        sql_rows = f"SELECT id, centroid FROM {centroids_table} WHERE epoch = %s ORDER BY id"
        if verbose:
            print(sql_rows % (latest_epoch,))

        with conn.cursor() as cur:
            cur.execute(sql_rows, (latest_epoch,))
            rows = cur.fetchall()

        # 3) Normalize to 2-D float32 array
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
                vec = list(c)  # fallback for VECTOR adapter
            centroids.append(vec)

        arr = np.array(centroids, dtype=np.float32)
        if arr.size > 0:
            result = arr  # else stays None

    return result


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

    # MiniBatchKMeans requires the **first** partial_fit batch to have >= n_clusters
    if args.batch_size < args.clusters:
        parser.error(f"--batch-size ({args.batch_size}) must be >= --clusters ({args.clusters}).")

  
    conn = psycopg2.connect(args.url)
    pks, vectors = fetch_vectors(
        conn,
        args.table,
        args.primary_key,
        args.input,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        verbose=args.verbose,
    )

    if args.verbose:
      print(f"[INFO] PKs: {json.dumps(pks, indent=2)}")
      print(f"[INFO] Vectors: {vectors}")
  
  
    if args.progress:
        vectors = tqdm(vectors, desc="Clustering", unit="vec")  # Just visual aid, does not affect clustering

    if args.algorithm == "kmeans":
        initial_centroids = load_existing_centroids(
              conn,
              args.table, args.input,
              verbose=args.verbose
        ) if args.increment else None

        if args.verbose:
          print(f"[INFO] Initial centroids: {initial_centroids}")
      
        labels, model = cluster_kmeans(
                  np.array(vectors), args.clusters, args.num_batches,
                  initial_centroids,
                  batch_size=args.batch_size
        )
        centroids = model.cluster_centers_
    else:
        labels, centroids = cluster_dbscan(np.array(vectors))

    if args.verbose:
      print(f"[INFO] Labels: {labels}")
      print(f"[INFO] Model: {model}")
      print(f"[INFO] Centroids: {centroids}")
      

    if centroids is not None:
        current_epoch = save_centroids(
                conn, args.table, args.input,
                centroids, args.increment,
                args.verbose, args.dry_run
        )
        save_cluster_assignments(
            conn, args.table, args.input,
            epoch = current_epoch,       # same epoch used in save_centroids
            pks=pks, labels=labels,
            verbose=args.verbose, dry_run=args.dry_run,
        )
    else:
        print("DBSCAN does not generate explicit centroids.")

    conn.close()


if __name__ == "__main__":
    main()
