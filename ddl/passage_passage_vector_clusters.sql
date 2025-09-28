CREATE TABLE passage_passage_vector_clusters (
      pid STRING NOT NULL,
      epoch INT8 NOT NULL,
      cluster_id INT8 NOT NULL,
      CONSTRAINT passage_passage_vector_clusters_pkey PRIMARY KEY (pid ASC),
      CONSTRAINT passage_passage_vector_clusters_pid_fkey FOREIGN KEY (pid) REFERENCES public.passage(pid),
      CONSTRAINT passage_passage_vector_clusters_epoch_cluster_id_fkey FOREIGN KEY (epoch, cluster_id) REFERENCES public.passage_passage_vector_centroid(epoch, id),
      INDEX cluster_membership_idx (epoch ASC, cluster_id ASC, pid ASC),
      INDEX passage_passage_vector_clusters_epoch_rec_idx (epoch ASC)
  );
