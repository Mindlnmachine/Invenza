from __future__ import annotations

import argparse
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from .config import settings


def _patent_columns(connection: psycopg.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'patents'"
    ).fetchall()
    return {row[0] for row in rows}


def search_patents(query: str, limit: int = 5, database_url: str | None = None, model_name: str | None = None) -> list[dict[str, Any]]:
    database_url = database_url or settings.database_url
    model_name = model_name or settings.embedding_model

    model = SentenceTransformer(model_name)
    embedding = model.encode(query, normalize_embeddings=True)

    with psycopg.connect(database_url) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(connection)
        columns = _patent_columns(connection)

        selected_columns = [
            "patent_id",
            "title",
            "assignee",
            "filing_date",
        ]
        if "cpc_codes" in columns:
            selected_columns.append("cpc_codes")
        else:
            selected_columns.append("NULL::text[] AS cpc_codes")
        if "ipc_codes" in columns:
            selected_columns.append("ipc_codes")
        else:
            selected_columns.append("NULL::text[] AS ipc_codes")
        if "result_link" in columns:
            selected_columns.append("result_link")
        else:
            selected_columns.append("NULL::text AS result_link")
        if "representative_figure_link" in columns:
            selected_columns.append("representative_figure_link")
        else:
            selected_columns.append("NULL::text AS representative_figure_link")
        if "abstract" in columns:
            selected_columns.append("abstract")
        else:
            selected_columns.append("title AS abstract")

        sql = f"""
        SELECT {', '.join(selected_columns)}, 1 - (embedding <=> %s) AS cosine_similarity,
               embedding <-> %s AS l2_distance
        FROM patents
        ORDER BY embedding <=> %s
        LIMIT %s
        """

        rows = connection.execute(sql, (embedding, embedding, embedding, limit)).fetchall()

    results: list[dict[str, Any]] = []
    for item in rows:
        row = dict(zip(["patent_id", "title", "assignee", "filing_date", "cpc_codes", "ipc_codes", "result_link", "representative_figure_link", "abstract", "cosine_similarity", "l2_distance"], item))
        results.append(
            {
                "patent_id": row.get("patent_id"),
                "title": row.get("title"),
                "assignee": row.get("assignee"),
                "filing_date": row.get("filing_date"),
                "cpc_codes": row.get("cpc_codes") or [],
                "ipc_codes": row.get("ipc_codes") or [],
                "result_link": row.get("result_link"),
                "representative_figure_link": row.get("representative_figure_link"),
                "abstract": row.get("abstract") or row.get("title"),
                "cosine_similarity": float(row.get("cosine_similarity", 0.0)),
                "l2_distance": float(row.get("l2_distance", 0.0)),
            }
        )
    return results


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Search patents in PostgreSQL using pgvector.")
    parser.add_argument("query", nargs="+", help="Query text")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    query = " ".join(args.query)
    for i, result in enumerate(search_patents(query, limit=args.limit), start=1):
        print(f"{i}. {result['patent_id']} | {result['title']}")
        print(f"   similarity={result['cosine_similarity']:.3f} | l2={result['l2_distance']:.3f}")
        print(f"   assignee={result['assignee'] or '-'} | filing={result['filing_date'] or '-'}")


if __name__ == "__main__":
    main()
