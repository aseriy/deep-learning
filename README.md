# Deep Learning

```sql
CREATE TABLE IF NOT EXISTS <table>_<input>_clusters (
  <primary> STRING NOT NULL,
  cluster_version INT NOT NULL,
  cluster_id INT NOT NULL,
  PRIMARY KEY (<primary>, cluster_version, cluster_id),
  FOREIGN KEY (<primary>)
    REFERENCES <table> (<primary>),
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