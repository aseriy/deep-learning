import os
import gzip
import json
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import psycopg2
from urllib.parse import urlparse
import sys

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

def ensure_vector_column(cur, table_name, output_column, vector_dim, dry_run):
    sql = f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {output_column} VECTOR({vector_dim})"
    if dry_run:
        print(f"[DRY RUN] Would execute: {sql}")
    else:
        cur.execute(sql)

def vectorize_batch(conn, table_name, input_column, output_column, primary_key, model, batch_size, dry_run, verbose, pbar=None):
    with conn.cursor() as cur:
        cur.execute(f"SELECT {input_column}, {primary_key} FROM {table_name} WHERE {output_column} IS NULL LIMIT {batch_size}")
        batch = cur.fetchall()
    if verbose:
        print(f"[DEBUG] Retrieved batch of {len(batch)} rows")

    if not batch:
        return 0

    with conn.cursor() as cur:
        texts = [row_text for row_text, _ in batch]
        ids = [row_id for _, row_id in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        if verbose:
            for row_id, row_text in zip(ids, texts):
                print(f"[INFO] Updating vector for row id {row_id}: '{row_text[:40]}'")

        if not dry_run:
            mogrified = [
                cur.mogrify('(%s, %s)', (row_id, embedding.tolist())).decode('utf-8')
                for row_id, embedding in zip(ids, embeddings)
            ]
            values_clause = ', '.join(mogrified)
            sql = f"""
                  UPDATE {table_name} AS t
                  SET {output_column} = v.embedding
                  FROM (VALUES {values_clause}) AS v({primary_key}, embedding)
                  WHERE t.{primary_key} = v.{primary_key}
                  """.strip()
            cur.execute(sql)

        if pbar:
            pbar.update(len(batch))

    return len(batch)

def main():
    parser = argparse.ArgumentParser(description="Vectorize rows in CockroachDB using SentenceTransformers.")
    parser.add_argument("-u", "--url", required=True, help="CockroachDB connection URL")
    parser.add_argument("-t", "--table", required=True, help="Target table name")
    parser.add_argument("-i", "--input", required=True, help="Column containing input text")
    parser.add_argument("-o", "--output", required=True, help="Column to store the vector")
    parser.add_argument("-b", "--batch-size", type=int, default=1000, help="Rows to process per batch")
    parser.add_argument("-n", "--num-batches", type=int, help="Limit number of batches to process")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    group.add_argument("-p", "--progress", action="store_true", help="Show progress bar")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Print SQL statements without executing (implies -v)")
    args = parser.parse_args()

    if args.dry_run:
        args.verbose = True

    model = SentenceTransformer('all-MiniLM-L6-v2')
    vector_dim = model.get_sentence_embedding_dimension()

    conn = get_connection(args.url)
    conn.autocommit = True
    cur = conn.cursor()

    # Determine primary key column
    cur.execute(f"""
        SELECT a.attname
        FROM   pg_index i
        JOIN   pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE  i.indrelid = %s::regclass AND i.indisprimary
    """, (args.table,))
    pk_result = cur.fetchone()
    if not pk_result:
        raise RuntimeError(f"No primary key found for table '{args.table}'")
    primary_key = pk_result[0]

    ensure_vector_column(cur, args.table, args.output, vector_dim, args.dry_run)

    total_batches = args.num_batches or float('inf')
    batch_counter = 0

    if args.progress:
        if args.num_batches:
            total_rows = args.num_batches * args.batch_size
        else:
            cur.execute(f"SELECT COUNT(*) FROM {args.table} WHERE {args.output} IS NULL")
            total_rows = cur.fetchone()[0]
        if args.verbose:
            print(f"[INFO] Found {total_rows} rows with NULL vectors to process")

        with tqdm(total=total_rows, desc="Vectorizing", unit="rows", smoothing=0.01) as pbar:
            while batch_counter < total_batches:
                processed = vectorize_batch(conn, args.table, args.input, args.output, primary_key, model, args.batch_size, args.dry_run, args.verbose, pbar=pbar)
                if processed == 0:
                    if args.progress and not args.verbose:
                        print("[INFO] No rows returned in batch. Exiting.")
                    break
                # pbar.update(processed)  # already updated per row inside vectorize_batch
                batch_counter += 1

    else:
        if args.verbose:
            cur.execute(f"SELECT COUNT(*) FROM {args.table} WHERE {args.output} IS NULL")
            total_rows = cur.fetchone()[0]
            print(f"[INFO] Found {total_rows} rows with NULL vectors to process")

        while batch_counter < total_batches:
            processed = vectorize_batch(conn, args.table, args.input, args.output, primary_key, model, args.batch_size, args.dry_run, args.verbose)
            if processed == 0:
                break
            batch_counter += 1

    cur.close()
    conn.close()
    if args.verbose:
        print("[INFO] Vectorization complete.")

if __name__ == "__main__":
    main()
