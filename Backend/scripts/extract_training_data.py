"""
Extract patent data from PGVector for fine-tuning.

This script retrieves all patents from PostgreSQL with their embeddings
and saves them in JSONL format for domain pre-training of qwen3:4b.

Usage:
    python scripts/extract_training_data.py --output data/patent_domain_data.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

# Prefer this project's Backend/app package over same-named packages in the environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings


def extract_patents(output_file: Path, database_url: str | None = None, batch_size: int = 500):
    """
    Extract all patents from PostgreSQL and save as JSONL.
    
    Args:
        output_file: Path to save JSONL file
        database_url: PostgreSQL connection URL
        batch_size: Number of records to fetch at a time
    """
    database_url = database_url or settings.database_url
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    sql = """
    SELECT 
        source_file, patent_id, title, assignee, inventor,
        priority_date, filing_date, publication_date, grant_date,
        result_link, representative_figure_link, embedding_text
    FROM patents
    ORDER BY source_file, patent_id
    """
    
    total_records = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        with psycopg.connect(database_url) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(connection)
            
            # Get total count
            count_result = connection.execute("SELECT COUNT(*) FROM patents").fetchone()
            total_count = count_result[0] if count_result else 0
            print(f"Total patents in database: {total_count}")
            
            # Stream results with cursor
            with connection.cursor(name="patent_cursor") as cursor:
                cursor.itersize = batch_size
                cursor.execute(sql)
                
                for row in cursor:
                    (source_file, patent_id, title, assignee, inventor,
                     priority_date, filing_date, publication_date, grant_date,
                     result_link, representative_figure_link, embedding_text) = row
                    
                    # Build training text: combine all available fields
                    text_parts = []
                    if title:
                        text_parts.append(f"Title: {title}")
                    if assignee:
                        text_parts.append(f"Assignee: {assignee}")
                    if inventor:
                        text_parts.append(f"Inventor: {inventor}")
                    if embedding_text:
                        text_parts.append(embedding_text)
                    
                    training_text = "\n".join(text_parts)
                    
                    record = {
                        "text": training_text,
                        "patent_id": patent_id,
                        "source_file": source_file,
                        "metadata": {
                            "title": title,
                            "assignee": assignee,
                            "filing_date": filing_date,
                            "publication_date": publication_date,
                        }
                    }
                    
                    # Write as JSONL
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_records += 1
                    
                    if total_records % 5000 == 0:
                        print(f"Extracted {total_records}/{total_count} patents...")
    
    print(f"\n✓ Extracted {total_records} patents to {output_file}")
    print(f"  File size: {output_file.stat().st_size / (1024*1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Extract patents from PGVector for fine-tuning")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "patent_domain_data.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--database-url",
        help="PostgreSQL connection URL (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for cursor iteration"
    )
    
    args = parser.parse_args()
    extract_patents(args.output, args.database_url, args.batch_size)


if __name__ == "__main__":
    main()
