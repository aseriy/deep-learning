CREATE TABLE passage_passage_vector_epoch (
    epoch INT8 PRIMARY KEY,
    current_at TIMESTAMPTZ NOT NULL DEFAULT now():::TIMESTAMPTZ
);
