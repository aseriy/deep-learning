#!/usr/bin/env python3

import argparse
import json
import os
import sys
import psycopg2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
import multiprocessing

def fetch_vectors(conn, table, column, verbose=False):
    with conn.cursor() as cur:
        cur.execute(f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL")
        rows = cur.fetchall()
    if verbose:
        print(f"[INFO] Retrieved {len(rows)} vectors from '{table}.{column}'")
    return np.array([r[0] for r in rows], dtype=np.float32)

def cluster_kmeans(vectors, num_clusters):
    model = KMeans(n_clusters=num_clusters, random_state=42)
    labels = model.fit_predict(vectors)
    return labels, model.cluster_centers_

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

    group = parser.add_argument_group()
    group.add_argument("-v", "--verbose", action="store_true", help="Verbose output (used for debugging)")
    group.add_argument("-p", "--progress", action="store_true", help="Show progress bar")

    parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run mode")
    args = parser.parse_args()

    conn = psycopg2.connect(args.url)
    vectors = fetch_vectors(conn, args.table, args.input, verbose=args.verbose)

    if args.progress:
        vectors = tqdm(vectors, desc="Clustering", unit="vec")  # Just visual aid, does not affect clustering

    if args.algorithm == "kmeans":
        labels, centroids = cluster_kmeans(np.array(vectors), args.clusters)
    else:
        labels, centroids = cluster_dbscan(np.array(vectors))

    print("\n[RESULT] Cluster centroids:")
    if centroids is not None:
        for i, centroid in enumerate(centroids):
            print(f"Cluster {i}: {centroid.tolist()}")
    else:
        print("DBSCAN does not generate explicit centroids.")

    conn.close()

if __name__ == "__main__":
    main()
