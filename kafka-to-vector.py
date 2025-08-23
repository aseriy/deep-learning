import argparse
import json
import sys
import time
from kafka import KafkaConsumer, TopicPartition, OffsetAndMetadata
from typing import List, Dict, Any, Tuple

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

def main():
    parser = argparse.ArgumentParser(description="Consume CDC messages from Kafka and populate Solr.")
    parser.add_argument("-k", "--kafka-url", required=True)
    parser.add_argument("-t", "--kafka-topic", required=True)
    parser.add_argument("-g", "--kafka-consumer-group", required=True)
    parser.add_argument("-b", "--batch-size", type=int, default=100)
    parser.add_argument("-n", "--num-batches", type=int, default=None)
    parser.add_argument("-i", "--batch-interval-ms", type=int, default=0)
    parser.add_argument("-f", "--fields", nargs="+", required=True)
    args = parser.parse_args()

    if not validate_kafka(args.kafka_url, args.kafka_topic, args.kafka_consumer_group):
        sys.exit(-1)

    total_processed = 0
    batch_num = 0

    while True:
        if args.num_batches is not None and batch_num >= args.num_batches:
            break

        batch = fetch_kafka_batch(args.kafka_url, args.kafka_topic, args.kafka_consumer_group, args.batch_size)

        if not batch:
            print(f"[batch {batch_num+1}] No more messages to process.")
            break

        print(f"[batch {batch_num+1}] processed: []")

        committed_partitions = commit_processed_messages(
            kafka_url=args.kafka_url,
            kafka_topic=args.kafka_topic,
            consumer_group=args.kafka_consumer_group,
            processed_messages=[],
        )
        print(f"[batch {batch_num+1}] committed offsets for {committed_partitions} partitions")

        total_processed += 0
        batch_num += 1

        if args.batch_interval_ms > 0 and (args.num_batches is None or batch_num < args.num_batches):
            time.sleep(args.batch_interval_ms / 1000.0)

    print(f"Total messages indexed: {total_processed}")

if __name__ == "__main__":
    main()
