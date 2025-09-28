CREATE TABLE passage_passage_vector_centroid (
      epoch INT8 NOT NULL DEFAULT nextval('public.passage_passage_vector_centroid_seq'::REGCLASS),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now():::TIMESTAMPTZ,
      id INT8 NOT NULL,
      centroid VECTOR(384) NULL,
      CONSTRAINT passage_passage_vector_centroid_pkey PRIMARY KEY (epoch ASC, id ASC),
      INDEX passage_passage_vector_centroid_id_storing_rec_idx (id ASC) STORING (centroid)
  );
  