#!/usr/bin/env python3

import argparse
import psycopg2

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

def assign_clusters(conn, table, column, primary_key, latest_version, verbose=False, dry_run=False, batch_size=None):
    centroid_table = f"{table}_{column}_centroid"
    cluster_table = f"{table}_{column}_clusters"

    sql = f"""
      WITH candidates AS (
          SELECT s.{primary_key}
          FROM {table} s
          WHERE s.{column} IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM {cluster_table} cl
                WHERE cl.{primary_key} = s.{primary_key} AND cl.cluster_version = %s
            )
          ORDER BY s.{primary_key}
          LIMIT %s
      ),
      distances AS (
          SELECT
              s.{primary_key},
              c.version AS cluster_version,
              c.id AS cluster_id,
              s.{column} <-> c.centroid AS distance,
              ROW_NUMBER() OVER (
                  PARTITION BY s.{primary_key}
                  ORDER BY s.{column} <-> c.centroid
              ) AS rn
          FROM {table} s
          JOIN candidates cand ON s.{primary_key} = cand.{primary_key},
               {centroid_table} c
          WHERE c.version = %s
      )
      INSERT INTO {cluster_table} ({primary_key}, cluster_version, cluster_id)
      SELECT {primary_key}, cluster_version, cluster_id FROM distances WHERE rn = 1;
    """

    if verbose:
        print("[SQL]", sql)

    if dry_run:
        print("[DRY-RUN] Assignment skipped.")
        return

    with conn.cursor() as cur:
        cur.execute(sql, (latest_version, batch_size, latest_version))
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
    parser.add_argument("-b", "--batch-size", type=int, help="Number of rows to process per batch")
    parser.add_argument("-n", "--num-batches", type=int, help="Maximum number of batches to run")
    parser.add_argument("-w", "--workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workders to use (default: 1)")

    args = parser.parse_args()

    conn = psycopg2.connect(args.url)
    primary_key = get_primary_key(conn, args.table)
    centroid_table = f"{args.table}_{args.input}_centroid"
    latest_version = get_latest_centroid_version(conn, centroid_table)

    batches_run = 0
    while True:
        if args.num_batches is not None and batches_run >= args.num_batches:
            if args.verbose:
                print(f"[INFO] Reached maximum number of batches: {args.num_batches}")
            break

        rowcount = assign_clusters(
            conn,
            args.table,
            args.input,
            primary_key,
            latest_version,
            verbose=args.verbose,
            dry_run=args.dry_run,
            batch_size=args.batch_size
        )

        if rowcount == 0:
            if args.verbose:
                print("[INFO] No more unassigned vectors remaining.")
            break

        batches_run += 1

    conn.close()

if __name__ == "__main__":
    main()
