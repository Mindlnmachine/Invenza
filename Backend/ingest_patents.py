from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterator

import pandas as pd
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS patents (
    source_file TEXT NOT NULL,
    patent_id TEXT NOT NULL,
    title TEXT,
    assignee TEXT,
    inventor TEXT,
    priority_date TEXT,
    filing_date TEXT,
    publication_date TEXT,
    grant_date TEXT,
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
    source_file, patent_id, title, assignee, inventor, priority_date,
    filing_date, publication_date, grant_date, result_link,
    representative_figure_link, embedding_text, embedding
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (source_file, patent_id) DO UPDATE SET
    title = EXCLUDED.title,
    assignee = EXCLUDED.assignee,
    inventor = EXCLUDED.inventor,
    priority_date = EXCLUDED.priority_date,
    filing_date = EXCLUDED.filing_date,
    publication_date = EXCLUDED.publication_date,
    grant_date = EXCLUDED.grant_date,
    result_link = EXCLUDED.result_link,
    representative_figure_link = EXCLUDED.representative_figure_link,
    embedding_text = EXCLUDED.embedding_text,
    embedding = EXCLUDED.embedding
"""


def clean(value: object) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def build_embedding_text(row: pd.Series) -> str:
    fields = {
        "Title": row.get("title"),
        "Assignee": row.get("assignee"),
        "Inventor": row.get("inventor/author"),
        "Priority date": row.get("priority date"),
        "Filing date": row.get("filing/creation date"),
        "Publication date": row.get("publication date"),
        "Grant date": row.get("grant date"),
    }
    return "\n".join(f"{label}: {clean(value) or ''}" for label, value in fields.items())


def read_csv(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, skiprows=1, dtype=str, keep_default_na=False)
    data.columns = data.columns.str.strip().str.lower()
    return data


def rows_for_file(path: Path) -> Iterator[dict[str, str | None]]:
    data = read_csv(path)
    required = {"id", "title", "representative figure link"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    for _, row in data.iterrows():
        patent_id = clean(row.get("id"))
        if not patent_id:
            continue
        yield {
            "source_file": path.name,
            "patent_id": patent_id,
            "title": clean(row.get("title")),
            "assignee": clean(row.get("assignee")),
            "inventor": clean(row.get("inventor/author")),
            "priority_date": clean(row.get("priority date")),
            "filing_date": clean(row.get("filing/creation date")),
            "publication_date": clean(row.get("publication date")),
            "grant_date": clean(row.get("grant date")),
            "result_link": clean(row.get("result link")),
            "representative_figure_link": clean(row.get("representative figure link")),
            "embedding_text": build_embedding_text(row),
        }


def batched(items: Iterator[dict[str, str | None]], size: int) -> Iterator[list[dict[str, str | None]]]:
    batch: list[dict[str, str | None]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def ingest(csv_paths: list[Path], database_url: str, model_name: str, batch_size: int) -> None:
    model = SentenceTransformer(model_name)
    dimensions = model.get_embedding_dimension()
    if dimensions != 384:
        raise ValueError(f"{model_name} produces {dimensions} dimensions; table expects 384")

    with psycopg.connect(database_url) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(connection)
        connection.execute(TABLE_SQL)
        connection.execute(INDEX_SQL)

        total = 0
        for csv_path in csv_paths:
            file_count = 0
            for batch in batched(rows_for_file(csv_path), batch_size):
                texts = [item["embedding_text"] or "" for item in batch]
                embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                values = [
                    (
                        item["source_file"], item["patent_id"], item["title"], item["assignee"],
                        item["inventor"], item["priority_date"], item["filing_date"],
                        item["publication_date"], item["grant_date"], item["result_link"],
                        item["representative_figure_link"], item["embedding_text"], embedding,
                    )
                    for item, embedding in zip(batch, embeddings, strict=True)
                ]

                with connection.cursor() as cursor:
                    cursor.executemany(UPSERT_SQL, values)
                connection.commit()
                file_count += len(batch)
                total += len(batch)
            print(f"{csv_path.name}: {file_count} rows upserted")
        print(f"Finished: {total} rows upserted into patents")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, action="append")
    parser.add_argument("--csv-dir", type=Path, default=Path(os.getenv("CSV_DIR", "..")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "128")))
    args = parser.parse_args()

    csv_paths = args.csv or sorted(args.csv_dir.glob("gp-search-*.csv"))
    if not csv_paths:
        raise SystemExit(f"No gp-search-*.csv files found in {args.csv_dir.resolve()}")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required; set it in Backend/.env")
    ingest(csv_paths, database_url, os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"), args.batch_size)


if __name__ == "__main__":
    main()
