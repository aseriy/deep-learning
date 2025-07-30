## Problem

Large-scale passage datasets — consisting of millions of short, structured text segments — are increasingly common in deep learning workflows. These datasets are typically stored in compressed formats and organized for downstream tasks like retrieval, ranking, or clustering.

The core challenge is enabling clustering over such datasets at scale: grouping semantically similar passages in a way that supports modern machine learning practices. This requires infrastructure that can efficiently ingest, index, and search vector representations — without introducing bottlenecks or complexity that hinder experimentation.

The MS MARCO TREC Deep Learning dataset serves as a representative example of this problem space.

### Example Data Structure

Each data point in the dataset is a JSON object with the following fields:

* `pid`: A unique passage ID
* `passage`: The text of the passage
* `spans`: Character-level span annotations (optional, format varies)
* `docid`: The document ID the passage is derived from

Example:

```json
{
  "pid": "msmarco_passage_00_997",
  "passage": "Those are all important, but the most asked question is, \u201cWhat\u2019s the 0-60 time?\u201d...",
  "spans": "(192,278),(279,376),(377,515)",
  "docid": "msmarco_doc_00_0"
}
```

Millions of such records are distributed across `.gz` files and must be parsed, processed, and clustered efficiently.

## Rationale for CockroachDB

CockroachDB enables deep learning workflows to operate directly on live data within the SQL database — the system of record — without the need to pipeline embeddings or clustering steps into external systems. Its native support for vector operations allows semantic processing to happen in place, not in delayed overnight batches. This simplifies architecture, reduces lag between data arrival and insight, and ensures consistency across both metadata and model-derived features.

## Solution Overview

This is a **proof-of-concept implementation**, not yet production-ready. The goal is to explore how CockroachDB's vector capabilities can support end-to-end clustering of unstructured text data — from raw JSON ingestion to vector-based semantic grouping — all within a single SQL-native platform. The code is written to be flexible and adaptable to datasets with similar JSON formats.

### Assumptions

* The dataset is a collection of `.gz` files, each containing newline-delimited JSON records.
* Each record has at least one primary identifier and a passage-like field to vectorize.
* CockroachDB is pre-configured with tables for storing raw passages, vectors, centroids, and clustering results.

### Pipeline Steps

1. **Ingest** – `json-to-sql.py`

   * Reads `.gz` files and converts JSON lines into SQL `INSERT` statements.

2. **Import** – Cockroach SQL shell

   * Load generated `.sql` files into the database.

3. **Vectorize** – `vectorize.py`

   * Uses a sentence encoder (e.g., from `transformers`) to generate vector embeddings for a specified column in the target table.

4. **Compute Centroids** – `centroid.py`

   * Computes centroids incrementally for all vectorized rows.
   * Supports versioning so clusters can evolve over time.

5. **Assign Clusters** – `clusters.py`

   * Batches over vectorized rows not yet assigned to a cluster.
   * Uses CockroachDB’s vector similarity functions to find and assign the nearest centroid.

Each step is modular, designed to allow experimentation and future enhancements such as parallelization, smarter cluster initialization, or dynamic re-clustering.

# Appendix

```sql
CREATE TABLE IF NOT EXISTS <table>_<input>_clusters (
  <primary> STRING NOT NULL,
  cluster_version INT NOT NULL,
  cluster_id INT NOT NULL,
  PRIMARY KEY (<primary>, cluster_version, cluster_id),
  FOREIGN KEY (<primary>) REFERENCES <table> (<primary>),
  FOREIGN KEY (cluster_version, cluster_id)
    REFERENCES <table>_<input>_centroid (version, id)
);
```

```sql
CREATE TABLE IF NOT EXISTS <table>_<input>_centroid (
  version INT NOT NULL,
  id INT NOT NULL,
  centroid VECTOR(<dimension>),
  PRIMARY KEY (version, id)
);
```

```sql
CREATE SEQUENCE IF NOT EXISTS <table>_<input>_centroid_seq;
```

```sql
CREATE INDEX <table>_<input>_not_null_idx
ON passage (pid)
WHERE passage_vector IS NOT NULL;
```

```sql
CREATE INDEX <table>_clusters_<primary_key>_version_idx
ON passage_passage_vector_clusters (pid, cluster_version);
```
