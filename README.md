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
   Reads `.gz` files containing newline-delimited JSON records and generates **batched SQL `INSERT` statements**.

   * Automatically infers columns from the first record
   * Supports `ON CONFLICT DO NOTHING` for safe re-ingestion
   * Accepts CLI flags for batch size (`--batch-size`), verbosity, and output location

   Example usage:

   ````bash
   python json-to-sql.py \
     data/msmarco_passage_00.gz \
     -t passage \
     -k pid \
     -b 1000 \
     -o ./sql \
     --progress
   ```\
   Reads `.gz` files containing newline-delimited JSON records and generates **batched SQL ****`INSERT`**** statements**.

   - Automatically infers columns from the first record
   - Supports `ON CONFLICT DO NOTHING` for safe re-ingestion
   - Accepts CLI flags for batch size (`--batch-size`), verbosity, and output location

   ````

2. **Import** – Cockroach SQL shell
   Load the `.sql` files produced in Step 1 into CockroachDB using the built-in SQL client. For example:

   ```bash
   cockroach sql --database=<your_database> --host=<your_host> --insecure < your_file.sql
   ```

   Or to batch-import all SQL files from a directory:

   ```bash
   for file in ./sql/*.sql; do
     cockroach sql --database=mydb --host=localhost --insecure < "$file"
   done
   ```

   Adjust `--insecure` and connection parameters as needed for your cluster.`.sql` files into the database using `cockroach sql` or a compatible client.

3. **Vectorize** – `vectorize.py`
   Generates vector embeddings for rows in a target table using the [SentenceTransformers](https://www.sbert.net/) library. Operates in parallel across batches of rows with missing vectors.

   * Connects to CockroachDB and identifies rows where the target vector column is `NULL`
   * Downloads a Hugging Face model snapshot and uses it to encode input text
   * Uses multiprocessing to process batches in parallel via `ProcessPoolExecutor`
   * Embeddings are written back into the table using `UPDATE ... FROM (VALUES ...)`
   * Allows full control over the output column name via the `--output` CLI flag
   * Automatically creates the vector column if it doesn’t exist (based on the model’s embedding dimension)

   Example usage:

   ````bash
   python vectorize.py \
     -u postgresql://user:pass@localhost:26257/mydb \
     -t passage \
     -i passage \
     -o passage_vector \
     -b 1000 \
     -w 4 \
     --progress
   ```\
   Generates vector embeddings for rows in a target table using the [SentenceTransformers](https://www.sbert.net/) library. Operates in parallel across batches of rows with missing vectors.

   - Connects to CockroachDB and identifies rows where the target vector column is `NULL`
   - Downloads a Hugging Face model snapshot and uses it to encode input text
   - Uses multiprocessing to process batches in parallel via `ProcessPoolExecutor`
   - Embeddings are written back into the table using `UPDATE ... FROM (VALUES ...)`
   - Allows full control over the output column name via the `--output` CLI flag
   - Automatically creates the vector column if it doesn’t exist (based on the model’s embedding dimension)

   ````

4. **Compute Centroids** – `centroids.py`
   Clusters vectorized rows using KMeans (or optionally DBSCAN) and saves the resulting centroids to the database.

   * Fetches all non-null vectors from the specified table and column
   * Uses `MiniBatchKMeans` for efficient clustering with partial fitting
   * Supports **incremental updates** to centroids by loading the latest version and refining it (`--increment`)
   * Assigns and saves a versioned cluster ID using an auto-incrementing sequence

   **Note:** The `<table>_<input>_centroid` table and associated sequence must be created in advance.

   Generic template:

   ```sql
   CREATE TABLE IF NOT EXISTS <table>_<input>_centroid (
     version INT NOT NULL,
     id INT NOT NULL,
     centroid VECTOR(<dimension>),
     PRIMARY KEY (version, id)
   );

   CREATE SEQUENCE IF NOT EXISTS <table>_<input>_centroid_seq;
   ```

   Example (for table `passage` and column `passage_vector`):

   ```sql
   CREATE TABLE IF NOT EXISTS passage_passage_vector_centroid (
     version INT NOT NULL,
     id INT NOT NULL,
     centroid VECTOR(384),
     PRIMARY KEY (version, id)
   );

   CREATE SEQUENCE IF NOT EXISTS passage_passage_vector_centroid_seq;
   ```

   Example usage:

   ````bash
   python centroids.py \
     -u postgresql://user:pass@localhost:26257/mydb \
     -t passage_table \
     -i passage_vector \
     -c 16 \
     --algorithm kmeans \
     --progress
   ``` for all vectorized rows.  
   - Supports versioning so clusters can evolve over time.

   ````

5. **Assign Clusters** – `clusters.py`
   Assigns each vectorized row to its nearest centroid using CockroachDB’s vector distance operator.

   * Retrieves all vectors not yet associated with a cluster (based on the latest centroid version)
   * Uses a SQL `WITH` clause and `<->` operator to compute distances and select the nearest centroid
   * Inserts cluster assignments into a `<table>_<input>_clusters` table
   * Executes in parallel batches using `ProcessPoolExecutor`

   **Note:** The `<table>_<input>_clusters` table must be created in advance. For example:

   ```sql
   CREATE TABLE IF NOT EXISTS passage_passage_vector_clusters (
     pid STRING NOT NULL,
     cluster_version INT NOT NULL,
     cluster_id INT NOT NULL,
     PRIMARY KEY (pid, cluster_version, cluster_id),
     FOREIGN KEY (pid) REFERENCES passage (pid),
     FOREIGN KEY (cluster_version, cluster_id)
       REFERENCES passage_passage_vector_centroid (version, id)
   );
   ```

   Example usage:

   ````bash
   python clusters.py \
     -u postgresql://user:pass@localhost:26257/mydb \
     -t passage \
     -i passage_vector \
     -b 5000 \
     --workers 4 \
     --progress
   ```\
   Batches over vectorized rows not yet assigned to a cluster.

   - Uses CockroachDB’s vector similarity functions to find and assign the nearest centroid.
   ````

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
