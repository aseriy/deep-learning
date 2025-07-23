import os
import gzip
import json
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_values
from urllib.parse import urlparse
import sys
import time
import random

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

def ensure_vector_column(conn, table_name, output_column, vector_dim, dry_run, show_info=True):
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

def get_null_vector_row_count(conn, table_name, output_column):
    with conn.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM "{table_name}" WHERE "{output_column}" IS NULL')
        return cur.fetchone()[0]

def vectorize_batch(db_url, table_name, input_column, output_column, primary_key, model, batch_size, dry_run, verbose, pbar=None, batch_index=0, warnings=None):
    conn = get_connection(db_url)
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute(f'''
            SELECT "{input_column}", "{primary_key}"
            FROM "{table_name}"
            WHERE "{output_column}" IS NULL
            LIMIT %s
        ''', (batch_size,))
        batch = cur.fetchall()

    if not batch:
        conn.close()
        return 0

    texts = [row_text for row_text, _ in batch]
    ids = [row_id for _, row_id in batch]
    embeddings = model.encode(texts, show_progress_bar=False)

    if verbose:
        for i, (row_id, row_text) in enumerate(zip(ids, texts), 1):
            print(f"[INFO] (batch {batch_index}, {i}/{len(batch)}) Updating vector for row id {row_id}: '{row_text[:40]}'")

    if not dry_run:
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                with conn.cursor() as cur:
                    values = [(row_id, embedding.tolist()) for row_id, embedding in zip(ids, embeddings)]
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
    parser.add_argument("-n", "--num-batches", type=int, default=None, help="Limit number of batches to process (default: all)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    group.add_argument("-p", "--progress", action="store_true", help="Show progress bar")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Print SQL statements without executing (only valid with --verbose)")
    args = parser.parse_args()

    if args.dry_run:
        if not args.verbose:
            parser.error("--dry-run must be used with --verbose")
        if args.progress:
            parser.error("--dry-run cannot be used with --progress")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    vector_dim = model.get_sentence_embedding_dimension()

    conn = get_connection(args.url)
    conn.autocommit = False

    primary_key = get_primary_key_column(conn, args.table)
    ensure_vector_column(conn, args.table, args.output, vector_dim, args.dry_run, show_info=not args.progress)

    if args.progress or args.verbose:
        total_rows = get_null_vector_row_count(conn, args.table, args.output)
        if args.verbose:
            print(f"[INFO] Found {total_rows} rows with NULL vectors to process")

    conn.close()

    total_batches = args.num_batches or float('inf')
    batch_counter = 0

    if args.progress:
        with tqdm(total=total_rows if args.num_batches is None else min(total_rows, args.num_batches * args.batch_size), desc="Vectorizing", unit="rows", smoothing=0.01) as pbar:
            while batch_counter < total_batches:
                processed = vectorize_batch(args.url, args.table, args.input, args.output, primary_key, model, args.batch_size, args.dry_run, False, pbar=pbar, batch_index=batch_counter + 1, warnings=warnings)
                if processed == 0:
                    break
                batch_counter += 1
    else:
        while batch_counter < total_batches:
            processed = vectorize_batch(args.url, args.table, args.input, args.output, primary_key, model, args.batch_size, args.dry_run, args.verbose, batch_index=batch_counter + 1)
            if processed == 0:
                break
            batch_counter += 1

    if args.verbose:
        print("[INFO] Vectorization complete.")
    if args.progress and warnings:
        from datetime import datetime
        print("[WARNINGS SUMMARY]", flush=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"warnings_{timestamp}.log"
        with open(log_filename, "w") as f:
            for w in warnings:
                print(w)
                f.write(w + "\n")
        print(f"Total warnings: {len(warnings)}")

if __name__ == "__main__":
    main()
