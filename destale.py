#!/usr/bin/env python3

import argparse
import psycopg2

def destale(
        conn,
        table: str, column: str,
        stale_epoch: int,
        batch_size: int, *,
        metric: str = "l2_distance",
        delete_stale: bool = True,
        verbose: bool = False, dry_run: bool = False) -> int:
    """Re-map PIDs from `stale_epoch` in `{table}_{column}_clusters` to the nearest centroid in the current epoch, in batches. Returns number of PIDs re-mapped."""

    clusters_table = f"{table}_{column}_clusters"
    centroids_table = f"{table}_{column}_centroid"
    epoch_table = f"{table}_{column}_epoch"

    sql = f"""
        WITH stale AS (
          SELECT pid
          FROM {clusters_table}
          WHERE epoch = {stale_epoch}
          ORDER BY pid
          LIMIT {batch_size}
        ),
        latest AS (
          SELECT max(epoch) AS e
          FROM {epoch_table}
        )
        UPDATE {clusters_table} AS cl
        SET epoch = l.e,
            cluster_id = c.id
        FROM stale AS s
        JOIN {table} AS p
          ON p.pid = s.pid
        CROSS JOIN latest AS l
        JOIN LATERAL (
          SELECT id
          FROM {centroids_table}
          WHERE epoch = l.e
          ORDER BY {metric}(p.{column}, centroid)
          LIMIT 1
        ) AS c ON true
        WHERE cl.pid = s.pid
          AND cl.epoch = {stale_epoch}
        RETURNING cl.pid;
    """

    if verbose:
        print(sql)

    processed = 0
    if not dry_run:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            processed = len(rows)
        conn.commit()
    return processed


def get_oldest_epoch(conn, table: str, column: str, verbose: bool = False) -> int | None:
    """
    Return the oldest epoch (MIN(epoch)) from `{table}_{column}_centroid`,
    but only if it differs from the latest (MAX(epoch)).
    Returns None if no rows exist or oldest == latest.
    """
    centroids_table = f"{table}_{column}_centroid"
    sql = f"SELECT MIN(epoch), MAX(epoch) FROM {centroids_table}"
    if verbose:
        print(sql)
    with conn.cursor() as cur:
        cur.execute(sql)
        min_epoch, max_epoch = cur.fetchone()

    result = None
    if min_epoch is not None and min_epoch != max_epoch:
        result = min_epoch
    return result


def prune_orphan_epoch(conn, table: str, column: str, epoch: int, *, verbose: bool = False, dry_run: bool = False) -> bool:
    """
    Delete `{table}_{column}_centroid` rows for `epoch` iff `{table}_{column}_clusters` has no rows for that epoch.
    Returns True if a delete occurred, False otherwise.
    """
    centroids_table = f"{table}_{column}_centroid"
    clusters_table  = f"{table}_{column}_clusters"

    sql = f"""
        DELETE FROM {centroids_table} AS c
        WHERE c.epoch = %s
          AND NOT EXISTS (
                SELECT 1
                FROM {clusters_table} AS cl
                WHERE cl.epoch = c.epoch
          )
    """

    if verbose:
        print(sql % (epoch,))

    deleted = False
    with conn.cursor() as cur:
        if not dry_run:
            cur.execute(sql, (epoch,))
            deleted = cur.rowcount > 0
    if not dry_run:
        conn.commit()

    return deleted


def main():
    parser = argparse.ArgumentParser(description="De-stale vector cluster mappings in CockroachDB.")
    parser.add_argument("-u", "--url", required=True, help="CockroachDB connection URL")
    parser.add_argument("-t", "--table", required=True, help="Table name containing vectors")
    parser.add_argument("-i", "--input", required=True, help="Column containing vector embeddings")
    parser.add_argument("-b", "--batch-size", type=int, default=1000, help="Batch size for processing rows")
    parser.add_argument("-e", "--num-epochs", type=int, default=None, help="Number of stale epochs to process (default: all)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output (used for debugging)")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run mode")
    args = parser.parse_args()

    conn = psycopg2.connect(args.url)

    epochs_to_process = args.num_epochs
    count = 0

    while epochs_to_process is None or count < epochs_to_process:
        stale_epoch = get_oldest_epoch(conn, args.table, args.input, verbose=args.verbose)
        if stale_epoch is None:
            if args.verbose:
                print("[INFO] No stale epochs found.")
            break

        # Drain this epoch in batches
        while True:
            processed = destale(
                conn,
                args.table,
                args.input,
                stale_epoch,
                args.batch_size,
                verbose=args.verbose,
                dry_run=args.dry_run,
            )
            if args.verbose:
                print(f"[INFO] destale() remapped {processed} vectors from epoch {stale_epoch}")
            if processed < args.batch_size:
                break

        pruned = prune_orphan_epoch(conn, args.table, args.input, stale_epoch, verbose=args.verbose, dry_run=args.dry_run)
        if args.verbose:
            print(f"[INFO] prune_orphan_epoch({stale_epoch}) => {pruned}")

        count += 1

    conn.close()


if __name__ == "__main__":
    main()
