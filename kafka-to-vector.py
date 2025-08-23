import argparse
import json
import sys
import time
from kafka import KafkaConsumer, TopicPartition, OffsetAndMetadata
from typing import List, Dict, Any, Tuple
import logging
from huggingface_hub import snapshot_download
from vectorize import silence_everything
from vectorize import vectorize_batch


def validate_kafka(kafka_url, kafka_topic, consumer_group):
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=kafka_url,
            group_id=consumer_group,
            enable_auto_commit=False,
        )
    except Exception as e:
        print(f"ERROR: Unable to connect to Kafka broker at {kafka_url} - {e}")
        return False

    try:
        topics = consumer.topics()
        if kafka_topic not in topics:
            print(f"ERROR: Kafka topic '{kafka_topic}' does not exist. Available topics: {topics}")
            return False
    except Exception as e:
        print(f"ERROR: Unable to retrieve topics from Kafka - {e}")
        return False
    finally:
        consumer.close()

    return True

def fetch_kafka_batch(kafka_url, kafka_topic, consumer_group, batch_size, timeout_ms=5000):
    consumer = KafkaConsumer(
        kafka_topic,
        bootstrap_servers=kafka_url,
        group_id=consumer_group,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        max_poll_records=batch_size,
    )
    try:
        for _ in range(50):
            consumer.poll(timeout_ms=100)
            if consumer.assignment():
                break

        polled = consumer.poll(timeout_ms=timeout_ms)
        batch = []
        for tp, records in polled.items():
            for r in records:
                batch.append({
                    "topic": r.topic,
                    "partition": r.partition,
                    "offset": r.offset,
                    "timestamp": r.timestamp,
                    "key": r.key,
                    "value": r.value,
                    "headers": dict(r.headers) if r.headers else {},
                })
                if len(batch) >= batch_size:
                    break
            if len(batch) >= batch_size:
                break
        return batch
    finally:
        consumer.close()

def commit_processed_messages(kafka_url: str, kafka_topic: str, consumer_group: str, processed_messages: List[Tuple[int, int]]) -> int:
    if not processed_messages:
        return 0

    highest: Dict[int, int] = {}
    for partition, offset in processed_messages:
        prev = highest.get(partition)
        if prev is None or offset > prev:
            highest[partition] = offset

    consumer = KafkaConsumer(
        kafka_topic,
        bootstrap_servers=kafka_url,
        group_id=consumer_group,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )
    try:
        for _ in range(50):
            consumer.poll(timeout_ms=100)
            if consumer.assignment():
                break
        else:
            raise RuntimeError("Timed out joining consumer group before commit")

        commit_map = {
            TopicPartition(kafka_topic, p): OffsetAndMetadata(o + 1, None, -1)
            for p, o in highest.items()
        }
        if commit_map:
            consumer.commit(offsets=commit_map)
            return len(commit_map)
        return 0
    finally:
        consumer.close()


def vectorize_from_kafka_batch(
    messages: List[Dict[str, Any]],
    *,
    db_url: str,
    table: str,
    input_column: str,
    output_column: str,
    primary_key: str,
    model_path: str,
    dry_run: bool,
    verbose: bool,
    batch_index: int,
) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Process a batch of Kafka CDC messages and perform vectorization for INSERT events only.

    Rules:
      - Skip resolved watermarks (payload contains top-level "resolved").
      - If envelope provides `op`, only process inserts (`op == "c"`).
      - If no `op`, treat as insert/update without op; continue.
      - Require `after[output_column] is None` before vectorizing.
      - Extract primary key from `after[primary_key]`, else from Kafka key.
      - De-duplicate IDs (keep first occurrence) and call vectorize_batch(...).
      - Return (rows_vectorized, processed_offsets) for messages that contributed.
    """

    def _loads_json(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, (bytes, bytearray)):
            try:
                return json.loads(x.decode("utf-8", errors="replace"))
            except Exception:
                return None
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return None
        return None

    ids: List[Any] = []
    seen: set = set()
    used_msgs: List[Dict[str, Any]] = []

    for m in messages:
        payload = _loads_json(m.get("value"))
        if not isinstance(payload, dict):
            continue

        # Skip resolved watermarks
        if "resolved" in payload:
            continue

        # Only handle inserts when op is provided; otherwise treat as insert-like
        op = payload.get("op")
        if op is not None and op != "c":
            continue

        after = payload.get("after")
        if not isinstance(after, dict):
            continue

        # Only vectorize rows that still need it
        if after.get(output_column) is not None:
            continue

        # Extract primary key
        pk = after.get(primary_key)
        if pk is None:
            key_obj = _loads_json(m.get("key"))
            if isinstance(key_obj, list) and key_obj:
                pk = key_obj[0]
            elif isinstance(key_obj, dict) and primary_key in key_obj:
                pk = key_obj[primary_key]
            elif isinstance(key_obj, (str, int)):
                pk = key_obj

        if pk is None:
            continue

        if pk not in seen:
            seen.add(pk)
            ids.append(pk)
            used_msgs.append(m)

    if not ids:
        return 0, []

    # Perform vectorization by primary keys
    processed_count = vectorize_batch(
        db_url=db_url,
        model_path=model_path,
        table_name=table,
        input_column=input_column,
        output_column=output_column,
        primary_key=primary_key,
        ids=ids,
        dry_run=dry_run,
        verbose=verbose,
        pbar=None,
        batch_index=batch_index,
        warnings=None,
    )

    # Commit offsets only on full success (conservative)
    if processed_count != len(ids):
        return processed_count, []

    # Collect offsets for messages that contributed to vectorization
    processed_offsets: List[Tuple[int, int]] = []
    for m in used_msgs:
        p = m.get("partition")
        o = m.get("offset")
        if p is not None and o is not None:
            processed_offsets.append((p, o))

    return processed_count, processed_offsets


def main():
    parser = argparse.ArgumentParser(description="Consume CDC messages from Kafka and update CockroachDB.")
    parser.add_argument("-u", "--url", required=True, help="CockroachDB connection URL (e.g., postgresql://user:pass@host:26257/db?sslmode=require)")
    parser.add_argument("-T", "--table", required=True, help="Target table name (e.g., passages)")
    parser.add_argument("-k", "--kafka-url", required=True)
    parser.add_argument("-t", "--kafka-topic", required=True)
    parser.add_argument("-g", "--kafka-consumer-group", required=True)
    parser.add_argument("-b", "--batch-size", type=int, default=100)
    parser.add_argument("-n", "--num-batches", type=int, default=None)
    parser.add_argument("--batch-interval-ms", type=int, default=0)
    parser.add_argument("-i", "--input", required=True, help="Text input column to encode (e.g., 'passage')")
    parser.add_argument("-o", "--output", required=True, help="Vector output column to update (e.g., 'passage_vector')")
    parser.add_argument("-p", "--primary-key", dest="primary_key", required=True, help="Primary key column name (e.g., 'pid')")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Compute but do not update the database")

    args = parser.parse_args()

    # Suppress huggingface_hub logger and ensure model snapshot is available (as in vectorize.py)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    with silence_everything():
        huggingface_path = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")

    if not validate_kafka(args.kafka_url, args.kafka_topic, args.kafka_consumer_group):
        sys.exit(-1)

    total_processed = 0
    batch_num = 0

    while True:
        if args.num_batches is not None and batch_num >= args.num_batches:
            break

        batch = fetch_kafka_batch(args.kafka_url, args.kafka_topic, args.kafka_consumer_group, args.batch_size)
        if args.verbose:
            print(f"[batch {batch_num+1}] fetched messages: {json.dumps(batch, indent=2, default=str)}")

        if not batch:
            print(f"[batch {batch_num+1}] No more messages to process.")
            break

        processed_count, processed_offsets = vectorize_from_kafka_batch(
            messages=batch,
            db_url=args.url,
            table=args.table,
            input_column=args.input,
            output_column=args.output,
            primary_key=args.primary_key,
            model_path=huggingface_path,
            dry_run=args.dry_run,
            verbose=args.verbose,
            batch_index=batch_num + 1,
        )
        print(f"[batch {batch_num+1}] processed rows: {processed_count}")
        print(f"[batch {batch_num+1}] processed offsets: {len(processed_offsets)}")

        committed_partitions = commit_processed_messages(
            kafka_url=args.kafka_url,
            kafka_topic=args.kafka_topic,
            consumer_group=args.kafka_consumer_group,
            processed_messages=processed_offsets,
        )
        print(f"[batch {batch_num+1}] committed offsets for {committed_partitions} partitions (messages: {len(processed_offsets)})")

        total_processed += processed_count
        batch_num += 1

        if args.batch_interval_ms > 0 and (args.num_batches is None or batch_num < args.num_batches):
            time.sleep(args.batch_interval_ms / 1000.0)

    print(f"Total messages indexed: {total_processed}")

if __name__ == "__main__":
    main()
