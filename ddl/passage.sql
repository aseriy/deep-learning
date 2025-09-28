CREATE TABLE passage (
      pid STRING NOT NULL,
      passage STRING NULL,
      spans STRING NULL,
      docid STRING NULL,
      passage_vector VECTOR(384) NULL,
      CONSTRAINT passage_pkey PRIMARY KEY (pid ASC),
      VECTOR INDEX passage_passage_vector_idx (passage_vector vector_l2_ops),
      INDEX passage_vector_not_null_idx (pid ASC) WHERE passage_vector IS NOT NULL
  );
