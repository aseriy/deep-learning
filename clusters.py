#!/usr/bin/env python3

import argparse
import psycopg2
import multiprocessing
import json

def get_primary_key(conn, table):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = %s::regclass AND i.indisprimary;
        """, (table,))
        result = cur.fetchone()
        if not result:
            raise ValueError(f"No primary key found for table: {table}")
        return result[0]

def get_latest_centroid_version(conn, centroid_table):
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(version) FROM {centroid_table}")
        result = cur.fetchone()
        return result[0]

def fetch_unassigned_vector_ids(conn,
                                table, column, cluster_table, primary_key,
                                latest_version, limit, verbose):
    sql = f"""
        SELECT s.{primary_key} FROM {table} AS s
        LEFT JOIN {cluster_table} AS cl ON cl.{primary_key} = s.{primary_key} AND cl.cluster_version = %s
        WHERE s.{column} IS NOT NULL AND cl.{primary_key} IS NULL
        ORDER BY s.{primary_key}
        """

    if limit is not None:
        sql += "LIMIT %s"

    cur = conn.cursor()

    if limit:
        if verbose:
            print(sql % (latest_version, limit))
        cur.execute(sql, (latest_version, limit))
    else:
        if verbose:
            print(sql % (latest_version,))
        cur.execute(sql, (latest_version,))

    
    return [row[0] for row in cur.fetchall()]




def assign_clusters(conn, table, column, primary_key, latest_version, verbose, dry_run, batch_index, ids):
    centroid_table = f"{table}_{column}_centroid"
    cluster_table = f"{table}_{column}_clusters"

    sql = f"""
        WITH distances AS (
            SELECT
                s.{primary_key},
                c.version AS cluster_version,
                c.id AS cluster_id,
                s.{column} <-> c.centroid AS distance,
                ROW_NUMBER() OVER (
                    PARTITION BY s.{primary_key}
                    ORDER BY s.{column} <-> c.centroid
                ) AS rn
            FROM {table} AS s,
                {centroid_table} AS c
            WHERE s.{primary_key} = ANY(%s)
                AND s.{column} IS NOT NULL
                AND c.version = %s
        )
        INSERT INTO {cluster_table} ({primary_key}, cluster_version, cluster_id)
            SELECT {primary_key}, cluster_version, cluster_id
                FROM distances WHERE rn = 1
        """

    ids_list = ', '.join(f"'{id}'" for id in ids)

    if verbose:
        print("[SQL]", sql % (ids_list, latest_version))

    rowcount = 0

    if dry_run:
        print("[DRY-RUN] Assignment skipped.")
 
    else:
        with conn.cursor() as cur:
            cur.execute(sql, (ids, latest_version))
            rowcount = cur.rowcount
            conn.commit()

        if verbose:
            print(f"[INFO] Assigned {rowcount} vectors")

    return rowcount


def main():
    parser = argparse.ArgumentParser(description="Assign each vector to the nearest centroid using CRDB's vector distance operator.")
    parser.add_argument("-u", "--url", required=True, help="CockroachDB connection URL")
    parser.add_argument("-t", "--table", required=True, help="Name of the table containing vectors")
    parser.add_argument("-i", "--input", required=True, help="Name of the vector column")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Do not execute, just print SQL")
    parser.add_argument("-b", "--batch-size", type=int, default=10000, help="Number of rows to process per batch")
    parser.add_argument("-n", "--num-batches", type=int, default=None, help="Maximum number of batches to run")
    parser.add_argument("-w", "--workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workders to use (default: 1)")

    args = parser.parse_args()

    conn = psycopg2.connect(args.url)
    primary_key = get_primary_key(conn, args.table)
    centroid_table = f"{args.table}_{args.input}_centroid"
    cluster_table = f"{args.table}_{args.input}_clusters"
    latest_version = get_latest_centroid_version(conn, centroid_table)

    total_limit = None
    if args.num_batches:
        total_limit = args.batch_size * args.num_batches


    all_ids = fetch_unassigned_vector_ids(
        conn,
        args.table, args.input, cluster_table, primary_key,
        latest_version, total_limit,
        args.verbose
    )
    chunks = [all_ids[i:i + args.batch_size] for i in range(0, len(all_ids), args.batch_size)]
    print("chunks: ", json.dumps(chunks, indent=2))

    if args.verbose:
        print(f"[INFO] Prefetched {len(all_ids)} IDs and split into {len(chunks)} chunks")

    for batch_index, id_chunk in enumerate(chunks):
        print (batch_index, json.dumps(id_chunk, indent=2))
        rowcount = assign_clusters(
            conn,
            args.table, args.input, primary_key, latest_version,
            args.verbose, args.dry_run, batch_index, id_chunk
        )


    conn.close()


if __name__ == "__main__":
    main()
