import argparse
import psycopg2
from sentence_transformers import SentenceTransformer
import logging



# --- helpers ---

def get_latest_epoch(conn, table: str, embedding: str, verbose: bool = False) -> int | None:
    """
    Return the latest epoch (MAX(epoch)) from `{table}_{column}_centroid`.
    Returns None if no rows exist.
    """

    centroid_table = f"{table}_{embedding}_centroid"
    sql = f"SELECT max(epoch) FROM {centroid_table};"
    if verbose:
        print(sql)

    epoch = None
    with conn.cursor() as cur:
        cur.execute(sql)
        epoch = cur.fetchone()[0]
      
    if verbose:
        print(f"[DEBUG] get_latest_epoch: {epoch}")
        
    return epoch


def nearest_centroid(conn, sentence: str, table: str, embedding: str, verbose: bool = False):
    """
    Given an embedding vector, find the nearest centroid id in `{table}_{column}_centroid`
    for the specified epoch.
    """

    model = SentenceTransformer("all-MiniLM-L6-v2") 
    sentence_vector =  model.encode(sentence).tolist()
    sentence_vector = "[" + ",".join(f"{float(x):.17g}" for x in sentence_vector) + "]"


    epoch = get_latest_epoch(conn, table, embedding, verbose) 


    sql = f"""
            SELECT id
            FROM {table}_{embedding}_centroid
            WHERE epoch = %s
            ORDER BY l2_distance (centroid, %s::VECTOR(384))
            LIMIT 1
        """

    if verbose:
        print(sql % (epoch, "'" + sentence_vector + "'"))

    centroid = None
    with conn.cursor() as cur:
        cur.execute(sql, (epoch, sentence_vector))
        centroid = cur.fetchone()[0]

    if verbose:
        print(f"[DEBUG] nearest_centroid: {centroid}")

    return epoch, centroid



def find_similar_vectors(conn, table: str, pk: str, sentence: str, epoch: int, centroid: int, limit: int = 10):
    """
    Return up to {limit} vectors mapped to the specified epoch-centroid
    """
    similar_vectors = None





    return similar_vectors



# --- main ---

def main():
    parser = argparse.ArgumentParser(description="Find similar sentences from CockroachDB.")
    parser.add_argument("-u", "--url", required=True, help="CockroachDB connection URL")
    parser.add_argument("-t", "--table", required=True, help="Table name containing sentences")
    parser.add_argument("-p", "--primary-key", required=True, help="Primary key of the table name containing vectors")
    parser.add_argument("-s", "--sentence", required=True, help="Column containing text to search for similarities")
    parser.add_argument("-e", "--embedding", required=True, help="Column containing vector embeddings")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output (used for debugging)")
    parser.add_argument("sentence", help="Input sentence to search for similars")

    args = parser.parse_args()

    conn = psycopg2.connect(args.url)

    epoch, centroid = nearest_centroid(conn, args.sentence, args.table, args.embedding, args.verbose)
    similar_vectors = find_similar_vectors(conn, args.table, args.primary_key, args.sentence, epoch, centroid)


    conn.close()


if __name__ == "__main__":
    main()
