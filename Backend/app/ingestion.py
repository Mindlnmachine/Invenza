from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from .config import settings
from .preprocessing import as_patent_record

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS patents (
    source_file TEXT NOT NULL,
    patent_id TEXT NOT NULL,
    title TEXT,
    abstract TEXT,
    claims TEXT,
    assignee TEXT,
    inventors TEXT,
    filing_date TEXT,
    publication_date TEXT,
    grant_date TEXT,
    cpc_codes TEXT[],
    ipc_codes TEXT[],
    citations TEXT[],
    result_link TEXT,
    representative_figure_link TEXT,
    embedding_text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    PRIMARY KEY (source_file, patent_id)
)
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS patents_embedding_hnsw_idx
ON patents USING hnsw (embedding vector_cosine_ops)
"""

UPSERT_SQL = """
INSERT INTO patents (
    source_file, patent_id, title, abstract, claims, assignee, inventors,
    filing_date, publication_date, grant_date, cpc_codes, ipc_codes, citations,
    result_link, representative_figure_link, embedding_text, embedding
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (source_file, patent_id) DO UPDATE SET
    title = EXCLUDED.title,
    abstract = EXCLUDED.abstract,
    claims = EXCLUDED.claims,
    assignee = EXCLUDED.assignee,
    inventors = EXCLUDED.inventors,
    filing_date = EXCLUDED.filing_date,
    publication_date = EXCLUDED.publication_date,
    grant_date = EXCLUDED.grant_date,
    cpc_codes = EXCLUDED.cpc_codes,
    ipc_codes = EXCLUDED.ipc_codes,
    citations = EXCLUDED.citations,
    result_link = EXCLUDED.result_link,
    representative_figure_link = EXCLUDED.representative_figure_link,
    embedding_text = EXCLUDED.embedding_text,
    embedding = EXCLUDED.embedding
"""


def read_csv_rows(csv_path: Path) -> Iterator[dict[str, object]]:
    df = pd.read_csv(csv_path, skiprows=1, dtype=str, keep_default_na=False)
    df.columns = [str(col).strip().lower() for col in df.columns]
    for _, row in df.iterrows():
        yield row.to_dict()


def iter_patent_records(csv_path: Path) -> Iterator[dict[str, object]]:
    for row in read_csv_rows(csv_path):
        if not row:
            continue
        try:
            record = as_patent_record(row, csv_path.name)
        except ValueError:
            continue
        yield {
            "source_file": record.source_file,
            "patent_id": record.patent_id,
            "title": record.title,
            "abstract": record.abstract,
            "claims": record.claims,
            "assignee": record.assignee,
            "inventors": record.inventors,
            "filing_date": record.filing_date,
            "publication_date": record.publication_date,
            "grant_date": record.grant_date,
            "cpc_codes": record.cpc_codes,
            "ipc_codes": record.ipc_codes,
            "citations": record.citations,
            "result_link": record.result_link,
            "representative_figure_link": record.representative_figure_link,
            "embedding_text": record.embedding_text,
        }


def load_csv_files(csv_dir: Path | None = None) -> list[Path]:
    target_dir = csv_dir or settings.csv_dir
    return sorted(target_dir.glob("gp-search-*.csv"))


def batch_records(items: Iterable[dict[str, object]], size: int) -> Iterator[list[dict[str, object]]]:
    batch: list[dict[str, object]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def ingest_patents(csv_paths: list[Path] | None = None, database_url: str | None = None, model_name: str | None = None, batch_size: int | None = None) -> int:
    csv_paths = csv_paths or load_csv_files()
    database_url = database_url or settings.database_url
    model_name = model_name or settings.embedding_model
    batch_size = batch_size or settings.batch_size

    model = SentenceTransformer(model_name)
    dimensions = model.get_embedding_dimension()
    if dimensions != 384:
        raise ValueError(f"Embedding model dimension mismatch: expected 384, got {dimensions}")

    total = 0
    with psycopg.connect(database_url) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(connection)
        connection.execute(TABLE_SQL)
        connection.execute(INDEX_SQL)

        for csv_path in csv_paths:
            for batch in batch_records(iter_patent_records(csv_path), batch_size):
                texts = [str(item["embedding_text"] or "") for item in batch]
                embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                values = [
                    (
                        item["source_file"],
                        item["patent_id"],
                        item["title"],
                        item["abstract"],
                        item["claims"],
                        item["assignee"],
                        item["inventors"],
                        item["filing_date"],
                        item["publication_date"],
                        item["grant_date"],
                        item["cpc_codes"],
                        item["ipc_codes"],
                        item["citations"],
                        item["result_link"],
                        item["representative_figure_link"],
                        item["embedding_text"],
                        embedding,
                    )
                    for item, embedding in zip(batch, embeddings, strict=True)
                ]

                with connection.cursor() as cursor:
                    cursor.executemany(UPSERT_SQL, values)
                connection.commit()
                total += len(batch)
                print(f"{csv_path.name}: {len(batch)} records ingested")

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest patent CSV files into PostgreSQL + pgvector")
    parser.add_argument("--csv-dir", type=Path, default=settings.csv_dir)
    parser.add_argument("--batch-size", type=int, default=settings.batch_size)
    args = parser.parse_args()

    csv_paths = load_csv_files(args.csv_dir)
    if not csv_paths:
        raise SystemExit(f"No gp-search-*.csv files found in {args.csv_dir.resolve()}")

    ingest_patents(csv_paths=csv_paths, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
