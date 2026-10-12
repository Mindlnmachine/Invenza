from __future__ import annotations

import argparse
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


def search(query: str, database_url: str, model_name: str, limit: int) -> None:
    model = SentenceTransformer(model_name)
    embedding = model.encode(query, normalize_embeddings=True)

    sql = """
    SELECT patent_id, source_file, title, assignee, result_link,
           representative_figure_link, 1 - (embedding <=> %s) AS similarity
    FROM patents
    ORDER BY embedding <=> %s
    LIMIT %s
    """
    with psycopg.connect(database_url) as connection:
        register_vector(connection)
        rows = connection.execute(sql, (embedding, embedding, limit)).fetchall()

    for patent_id, source_file, title, assignee, result_link, image_link, similarity in rows:
        print(f"{similarity:.3f} | {patent_id} | {title}")
        print(f"  Source: {source_file} | Assignee: {assignee or '-'}")
        print(f"  Patent: {result_link or '-'}")
        print(f"  Figure: {image_link or '-'}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Search patent embeddings in PostgreSQL")
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required; set it in Backend/.env")
    search(args.query, database_url, os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"), args.limit)


if __name__ == "__main__":
    main()
