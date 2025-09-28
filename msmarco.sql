create_statement
"CREATE TABLE public.passage (
	pid STRING NOT NULL,
	passage STRING NULL,
	spans STRING NULL,
	docid STRING NULL,
	passage_vector VECTOR(384) NULL,
	CONSTRAINT passage_pkey PRIMARY KEY (pid ASC),
	VECTOR INDEX passage_passage_vector_idx (passage_vector vector_l2_ops),
	INDEX passage_vector_not_null_idx (pid ASC) WHERE passage_vector IS NOT NULL
);"
CREATE SEQUENCE public.passage_passage_vector_centroid_seq MINVALUE 0 MAXVALUE 9223372036854775807 INCREMENT 1 START 1;
"CREATE TABLE public.passage_passage_vector_centroid (
	epoch INT8 NOT NULL DEFAULT nextval('public.passage_passage_vector_centroid_seq'::REGCLASS),
	updated_at TIMESTAMPTZ NOT NULL DEFAULT now():::TIMESTAMPTZ,
	id INT8 NOT NULL,
	centroid VECTOR(384) NULL,
	CONSTRAINT passage_passage_vector_centroid_pkey PRIMARY KEY (epoch ASC, id ASC),
	INDEX passage_passage_vector_centroid_id_storing_rec_idx (id ASC) STORING (centroid)
);"
"CREATE TABLE public.passage_passage_vector_clusters (
	pid STRING NOT NULL,
	epoch INT8 NOT NULL,
	cluster_id INT8 NOT NULL,
	CONSTRAINT passage_passage_vector_clusters_pkey PRIMARY KEY (pid ASC),
	INDEX cluster_membership_idx (epoch ASC, cluster_id ASC, pid ASC),
	UNIQUE INDEX passage_passage_vector_clusters_pid_epoch_key (pid ASC, epoch ASC),
	INDEX passage_passage_vector_clusters_epoch_rec_idx (epoch ASC)
);"
ALTER TABLE public.passage_passage_vector_clusters ADD CONSTRAINT passage_passage_vector_clusters_pid_fkey FOREIGN KEY (pid) REFERENCES public.passage(pid);
ALTER TABLE public.passage_passage_vector_clusters ADD CONSTRAINT passage_passage_vector_clusters_epoch_cluster_id_fkey FOREIGN KEY (epoch, cluster_id) REFERENCES public.passage_passage_vector_centroid(epoch, id);
-- Validate foreign key constraints. These can fail if there was unvalidated data during the SHOW CREATE ALL TABLES
ALTER TABLE public.passage_passage_vector_clusters VALIDATE CONSTRAINT passage_passage_vector_clusters_pid_fkey;
ALTER TABLE public.passage_passage_vector_clusters VALIDATE CONSTRAINT passage_passage_vector_clusters_epoch_cluster_id_fkey;
