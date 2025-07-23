#!/usr/bin/env python3

import json
import gzip
import argparse
import os
from tqdm import tqdm

def sql_escape(value):
    if value is None:
        return 'NULL'
    if isinstance(value, (int, float)):
        return str(value)
    val = str(value).replace("'", "''")
    return f"'{val}'"

def write_insert(out, table, columns, batch, primary_key, verbose=False):
    column_list = ', '.join(columns)
    values = ',\n  '.join(batch)
    conflict_clause = f" ON CONFLICT ({primary_key}) DO NOTHING" if primary_key else ""
    insert_stmt = f"INSERT INTO {table} ({column_list}) VALUES\n  {values}{conflict_clause};\n\n"
    out.write(insert_stmt)
    if verbose:
        print(f"[INSERT] Wrote batch of {len(batch)} rows to table '{table}'")
        print(f"         Columns: {column_list}")
        print(f"         First row: {batch[0] if batch else 'N/A'}")

def generate_sql(input_path, output_path, table_name, batch_size, primary_key=None, verbose=False, progress=False):
    inserts = []
    columns = None
    batch = []

    if verbose:
        print(f"[INFO] Processing input file: {input_path}")
        print(f"[INFO] Output SQL file will be written to: {output_path}")

    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        lines = f.readlines()

    iterator = tqdm(lines, desc=os.path.basename(input_path), unit="lines", leave=False) if progress else lines

    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(f"-- Generated from {os.path.basename(input_path)}\n\n")

        for line_num, line in enumerate(iterator, start=1):
            record = json.loads(line)

            if columns is None:
                columns = list(record.keys())
                if verbose and not progress:
                    print(f"[INFO] Detected columns: {columns}")

            values = [sql_escape(record.get(col)) for col in columns]
            batch.append(f"({', '.join(values)})")

            if batch_size and len(batch) >= batch_size:
                write_insert(out, table_name, columns, batch, primary_key, verbose and not progress)
                batch.clear()

        if batch:
            write_insert(out, table_name, columns, batch, primary_key, verbose and not progress)

        if verbose:
            print(f"[INFO] Finished processing {line_num} lines")
            print(f"[INFO] Final SQL written to: {output_path}")
        else:
            print(f"[DONE] {line_num} lines processed, output at: {output_path}")

# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert gzipped JSONL to SQL insert statements.")
    parser.add_argument("input", nargs='+', help="Path(s) to input .gz file(s)")
    parser.add_argument("-o", "--output", help="Output directory (default: current directory)")
    parser.add_argument("-t", "--table", required=True, help="Target SQL table name")
    parser.add_argument("-k", "--primary", metavar="COL", help="Primary key column (adds ON CONFLICT DO NOTHING)")
    parser.add_argument("-b", "--batch-size", type=int, default=1000, help="Rows per INSERT batch (default: 1000; 0 = all rows in one INSERT)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-p", "--progress", action="store_true", help="Show progress bar")
    args = parser.parse_args()
    output_dir = args.output or os.getcwd()
    total_files = len(args.input)
    if args.progress and total_files > 1:
        outer = tqdm(args.input, desc="Overall", unit="file", bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] - {desc}")
    else:
        outer = args.input

    for i, input_path in enumerate(outer, start=1):
        if isinstance(outer, tqdm):
            outer.set_description(f"[{i}/{total_files}] {os.path.basename(input_path)}")
        input_base = os.path.basename(input_path)
        if input_base.endswith('.gz'):
            output_name = input_base[:-3] + '.sql'
        else:
            output_name = os.path.splitext(input_base)[0] + '.sql'

        output_file = os.path.join(output_dir, output_name)
        import time
        start_time = time.time()
        row_count = [0]
        def counting_generate_sql(*args_, **kwargs_):
            def wrapper(*args__, **kwargs__):
                return generate_sql(*args__, **kwargs__)
            return wrapper(*args_, **kwargs_)

        def wrapped_generate_sql(input_path, output_path, table_name, batch_size, primary_key, verbose, progress):
            inserts = []
            columns = None
            batch = []

            if verbose:
                print(f"[INFO] Processing input file: {input_path}")
                print(f"[INFO] Output SQL file will be written to: {output_path}")

            with gzip.open(input_path, 'rt', encoding='utf-8') as f:
                lines = f.readlines()

            iterator = tqdm(lines, desc=os.path.basename(input_path), unit="lines", leave=False) if progress else lines

            with open(output_path, 'w', encoding='utf-8') as out:
                out.write(f"-- Generated from {os.path.basename(input_path)}\n\n")

                for line_num, line in enumerate(iterator, start=1):
                    record = json.loads(line)
                    row_count[0] += 1

                    if columns is None:
                        columns = list(record.keys())
                        if verbose and not progress:
                            print(f"[INFO] Detected columns: {columns}")

                    values = [sql_escape(record.get(col)) for col in columns]
                    batch.append(f"({', '.join(values)})")

                    if batch_size and len(batch) >= batch_size:
                        write_insert(out, table_name, columns, batch, primary_key, verbose and not progress)
                        batch.clear()

                if batch:
                    write_insert(out, table_name, columns, batch, primary_key, verbose and not progress)

                if verbose:
                    print(f"[INFO] Finished processing {line_num} lines")
                    print(f"[INFO] Final SQL written to: {output_path}")
                else:
                    print(f"[DONE] {line_num} lines processed, output at: {output_path}")

        wrapped_generate_sql(input_path, output_file, args.table, args.batch_size, args.primary, args.verbose, args.progress)
        duration = time.time() - start_time

        duration = time.time() - start_time
        input_size_bytes = os.path.getsize(input_path)
        compression_ratio = input_size_bytes / (row_count[0] if row_count[0] else 1)
        uncompressed_size_bytes = sum(len(line.encode('utf-8')) for line in lines) if 'lines' in locals() else 0
        estimated_compression = uncompressed_size_bytes / input_size_bytes if input_size_bytes else 0
        with open(os.path.join(output_dir, "summary.log"), "a", encoding="utf-8") as log:
            log.write(f"[SUMMARY] {os.path.basename(input_path)}: {row_count[0]} rows in {duration:.2f} seconds across {row_count[0] // args.batch_size + (1 if row_count[0] % args.batch_size != 0 else 0)} batches, avg {row_count[0] / duration:.2f} rows/sec, {input_size_bytes / 1024 ** 2:.2f} MB read, ~{compression_ratio:.1f} bytes/row, ~{estimated_compression:.2f}x compression\n")
        print(f"[SUMMARY] {os.path.basename(input_path)}: {row_count[0]} rows in {duration:.2f} seconds across {row_count[0] // args.batch_size + (1 if row_count[0] % args.batch_size != 0 else 0)} batches, avg {row_count[0] / duration:.2f} rows/sec, {input_size_bytes / 1024 ** 2:.2f} MB read, ~{compression_ratio:.1f} bytes/row, ~{estimated_compression:.2f}x compression")
