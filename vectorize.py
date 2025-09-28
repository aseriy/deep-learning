import os
import sys
import gzip
import json
import time
import random
import argparse
import logging
import contextlib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import psycopg2
from psycopg2.extras import execute_values
from urllib.parse import urlparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


_MODEL_CACHE = {}

def get_model(model_path: str) -> SentenceTransformer:
    m = _MODEL_CACHE.get(model_path)
    if m is None:
        m = SentenceTransformer(model_path)   # loads once per process
        _MODEL_CACHE[model_path] = m
    return m


@contextlib.contextmanager
def silence_everything():
    with open(os.devnull, "w") as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def get_connection(db_url):
    parsed_url = urlparse(db_url)
    return psycopg2.connect(
        dbname=parsed_url.path[1:],
        user=parsed_url.username,
        password=parsed_url.password,
        host=parsed_url.hostname,
        port=parsed_url.port or 26257,
        sslmode=parsed_url.query.split('sslmode=')[1] if parsed_url.query and 'sslmode=' in parsed_url.query else 'require'
    )

def estimate_total_lines(path):
    count = 0
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for _ in f:
                count += 1
    except Exception:
        return None
    return count

def ensure_vector_column(conn, model_path, table_name, output_column, dry_run, show_info=True):
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT a.attname, t.typname
            FROM pg_attribute a
            JOIN pg_type t ON a.atttypid = t.oid
            WHERE a.attrelid = %s::regclass
              AND a.attname = %s
              AND a.attnum > 0
              AND NOT a.attisdropped
        """, (table_name, output_column))
        existing = cur.fetchone()

        if existing:
            if 'vector' not in existing[1]:
                raise RuntimeError(f"Column {output_column} exists but is not of VECTOR type.")
            if show_info:
                print(f"[INFO] Column {output_column} already exists")
            return

        vector_dim = SentenceTransformer(model_path).get_sentence_embedding_dimension()
        sql = f'ALTER TABLE "{table_name}" ADD COLUMN "{output_column}" VECTOR({vector_dim})'
        if dry_run:
            print(f"[DRY RUN] Would execute: {sql}")
        else:
            cur.execute(sql)

def get_primary_key_column(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT a.attname
            FROM   pg_index i
            JOIN   pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE  i.indrelid = %s::regclass AND i.indisprimary
        """, (table_name,))
        pk_result = cur.fetchone()
        if not pk_result:
            raise RuntimeError(f"No primary key found for table '{table_name}'")
        return pk_result[0]

def get_null_vector_row_count(conn, table_name, output_column, primary_key):
    with conn.cursor() as cur:
        cur.execute(f'SELECT COUNT("{primary_key}") FROM "{table_name}" WHERE "{output_column}" IS NULL')
        return cur.fetchone()[0]

def fetch_null_vector_ids(conn, table_name, output_column, primary_key, limit, verbose=False):
    max_retries = 10
    for attempt in range(1, max_retries + 1):
        try:
            with conn.cursor() as cur:
                cur.execute(f'SELECT "{primary_key}" FROM "{table_name}" WHERE "{output_column}" IS NULL LIMIT %s', (limit,))
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            if attempt < max_retries:
                print(f"[WARN] Retry {attempt}/{max_retries} on fetch_null_vector_ids: {e}", flush=True)
                time.sleep(0.5 * attempt + random.uniform(0, 0.3))
            else:
                raise


# Vectorizes a batch of rows by primary key.
# - Establishes a new DB connection per batch.
# - Encodes using SentenceTransformer in parallel.
# - Retries on failure with exponential backoff.
# - Reports status to tqdm or stdout depending on mode.

def vectorize_batch(
                    db_url, model_path, table_name,
                    input_column, output_column, primary_key, ids,
                    dry_run, verbose, pbar=None, batch_index=0, warnings=None
                    ):
    
    if not ids:
        return 0

    conn = get_connection(db_url)
    conn.autocommit = False

    with conn.cursor() as cur:
        placeholders = ','.join(['%s'] * len(ids))
        cur.execute(
            f'''
                SELECT "{input_column}", "{primary_key}"
                FROM "{table_name}"
                WHERE "{primary_key}" IN ({placeholders})
            ''', ids)
        batch = cur.fetchall()

    if not batch:
        conn.close()
        return 0

    texts = [row_text for row_text, _ in batch]
    row_ids = [row_id for _, row_id in batch]
    model = get_model(model_path)
    embeddings = model.encode(texts, batch_size=128, show_progress_bar=False)

    if verbose:
        for i, (row_id, row_text) in enumerate(zip(row_ids, texts), 1):
            input_column_text = row_text[:40].replace('\n', '').replace('\r', '')
            print(f"[INFO] (batch {batch_index}, {i}/{len(batch)}) Updating vector for row id {row_id}: '{input_column_text}'")

    if not dry_run:
        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                with conn.cursor() as cur:
                    values = [(row_id, embedding.tolist()) for row_id, embedding in zip(row_ids, embeddings)]
                    sql = f'''
                        UPDATE "{table_name}" AS t
                        SET "{output_column}" = v.embedding
                        FROM (VALUES %s) AS v("{primary_key}", embedding)
                        WHERE t."{primary_key}" = v."{primary_key}"
                    '''
                    execute_values(cur, sql, values, template="(%s, %s)")
                conn.commit()
                break
            except Exception as e:
                conn.rollback()
                if attempt < max_retries:
                    if warnings is not None:
                        from datetime import datetime
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        warnings.append(f"[{timestamp}] [WARN] (batch {batch_index}) Retry {attempt}/{max_retries} after failure: {e}")
                    elif pbar is None:
                        print(f"[WARN] Retry {attempt}/{max_retries} after failure: {e}", flush=True)
                    time.sleep(0.5 * attempt + random.uniform(0, 0.3))
                else:
                    print(f"[ERROR] Failed after {max_retries} retries: {e}")

    if pbar is not None:
        pbar.update(len(batch))

    conn.close()
    return len(batch)
    

def main():
    warnings = []
    parser = argparse.ArgumentParser(description="Vectorize rows in CockroachDB using SentenceTransformers.")
    parser.add_argument("-u", "--url", required=True, help="CockroachDB connection URL")
    parser.add_argument("-t", "--table", required=True, help="Target table name")
    parser.add_argument("-i", "--input", required=True, help="Column containing input text")
    parser.add_argument("-o", "--output", required=True, help="Column to store the vector")
    parser.add_argument("-b", "--batch-size", type=int, default=1000, help="Rows to process per batch")
    parser.add_argument("-n", "--num-batches", type=int, default=1,
                        help="Number of batches to process before exiting (default: 1)")
    parser.add_argument("-F", "--follow", action="store_true",
                        help="Keep running: keep vectorizing new NULL rows indefinitely")
    parser.add_argument("--max-idle", type=float, default=60.0,
                        help="Max idle time before exit, in MINUTES (0 = no idle limit)")
    parser.add_argument("--min-idle", type=float, default=15.0,
                        help="Initial idle backoff between empty scans, in SECONDS")
    parser.add_argument("-w", "--workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workders to use (default: 1)")

    # Disallow verbose and progress together
    group = parser.add_argument_group()
    group.add_argument("-v", "--verbose", action="store_true", help="Verbose output (used for debugging)")
    group.add_argument("-p", "--progress", action="store_true", help="Show progress bar")

    parser.add_argument("-d", "--dry-run", action="store_true", help="Print SQL statements without executing (only valid with --verbose)")
    args = parser.parse_args()

    if args.dry_run:
        if not args.verbose:
            parser.error("--dry-run must be used with --verbose")

    # Suppress huggingface_hub logger
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    with silence_everything():
        huggingface_path = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")


    batch_counter = 0
    conn = get_connection(args.url)
    conn.autocommit = True

    primary_key = get_primary_key_column(conn, args.table)
    ensure_vector_column(conn, huggingface_path, args.table, args.output, args.dry_run, show_info=not args.progress)

    total_rows = get_null_vector_row_count(conn, args.table, args.output, primary_key)

    max_workers = os.cpu_count() or 4

    pbar = None
    if args.progress:
        pbar = tqdm(
                    total=total_rows if args.num_batches is None else min(total_rows,
                    args.num_batches * args.batch_size),
                    desc="Vectorizing",
                    unit="rows",
                    smoothing=0.01
                )

    executor = ProcessPoolExecutor(max_workers=args.workers)
    futures = []
    warnings = []

    # Backoff state
    idle_wait = max(0.001, float(args.min_idle))   # seconds
    idle_spent = 0.0                               # seconds
    idle_budget = max(0.0, float(args.max_idle) * 60.0)  # seconds (0 = unlimited)

    start = time.time() if args.verbose else None

    # Per-run counters (1-based for human-friendly logs)
    run_counter = 1
    batch_in_run = 1

    while True:
        # Stop after N batches per run (default 1) unless following
        if (not args.follow) and batch_in_run > args.num_batches:
            break

        # Fetch one page of IDs (no wait on start or after successful work)
        ids = fetch_null_vector_ids(conn, args.table, args.output, primary_key, args.batch_size)

        if ids:
            # Got work → reset backoff
            idle_wait = max(0.001, float(args.min_idle))
            idle_spent = 0.0

            # Run one batch (via pool for per-process model reuse)
            if args.verbose:
                print(f"[INFO] Run {run_counter}, Batch {batch_in_run} starting ({len(ids)} rows)")
            fut = executor.submit(
                vectorize_batch,
                args.url, huggingface_path, args.table,
                args.input, args.output, primary_key, ids,
                args.dry_run, args.verbose, pbar, batch_in_run, warnings
            )
            fut.result()
            batch_counter += 1
            batch_in_run += 1
            if args.follow and batch_in_run > args.num_batches:
                if args.verbose:
                    print(f"[INFO] Run {run_counter} complete ({args.num_batches} batches).")
                run_counter += 1
                batch_in_run = 1
            continue

        # No work returned → back off or exit if max idle reached
        if idle_budget > 0.0 and idle_spent >= idle_budget:
            if args.verbose:
                print(f"[INFO] Max idle reached ({args.max_idle} min). Exiting.")
            break

        # Sleep current backoff and then double it (exponential), cap by remaining budget if any
        to_sleep = idle_wait
        if idle_budget > 0.0:
            remaining = max(0.0, idle_budget - idle_spent)
            to_sleep = min(to_sleep, remaining)
        time.sleep(to_sleep)
        idle_spent += to_sleep
        idle_wait = idle_wait * 2.0

    if args.verbose:
        print("Done in", time.time() - start, "seconds")



    if args.verbose:
        print("[INFO] Vectorization complete.")

    if args.progress and warnings:
        from datetime import datetime
        print("\n[WARNINGS SUMMARY]", flush=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"warnings_{timestamp}.log"
        with open(log_filename, "w") as f:
            for w in warnings:
                print(w)
                f.write(w + "\n")
        print(f"Total warnings: {len(warnings)}")


if __name__ == "__main__":
    main()
