"""
Create embedding fine-tuning dataset with hard negatives.

This script creates a dataset for fine-tuning sentence-transformers on patent data.
It pairs patents with hard negatives (non-similar patents) for contrastive learning.

Usage:
    python scripts/prepare_embedding_dataset.py --output data/embedding_training_data
"""

import argparse
import json
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

# Prefer this project's Backend/app package over same-named packages in the environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings


def prepare_embedding_dataset(
    output_dir: Path,
    database_url: str | None = None,
    embedding_model: str | None = None,
    sample_size: int | None = None,
    hard_negative_count: int = 3,
):
    """
    Prepare embedding fine-tuning dataset with triplets (anchor, positive, negative).
    
    Args:
        output_dir: Directory to save dataset
        database_url: PostgreSQL connection URL
        embedding_model: Embedding model to use for initial similarity
        sample_size: Limit number of patents (None = all)
        hard_negative_count: Number of hard negatives per anchor
    """
    database_url = database_url or settings.database_url
    embedding_model = embedding_model or settings.embedding_model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the existing vectors and HNSW index to mine neighbors. Building an
    # all-pairs matrix for 87K patents would require tens of GiB of memory.
    sql = "SELECT source_file, patent_id, title, embedding_text, embedding FROM patents ORDER BY patent_id"
    if sample_size:
        sql += f" LIMIT {sample_size}"

    with psycopg.connect(database_url) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(connection)
        rows = connection.execute(sql).fetchall()

        print(f"Fetched {len(rows)} patents from database")
        if len(rows) < 2:
            raise ValueError("At least two patents with embeddings are required")

        neighbor_sql = """
        SELECT source_file, patent_id, title, embedding_text,
               1 - (embedding <=> %s) AS cosine_similarity
        FROM patents
        WHERE NOT (source_file = %s AND patent_id = %s)
        ORDER BY embedding <=> %s
        LIMIT %s
        """

        output_file = output_dir / "triplets.jsonl"
        triplet_count = 0
        print("Creating triplets using the PGVector HNSW index...")

        with open(output_file, "w", encoding="utf-8") as output:
            for i, anchor_row in enumerate(rows):
                source_file, patent_id, title, text, embedding = anchor_row
                neighbor_rows = connection.execute(
                    neighbor_sql,
                    (embedding, source_file, patent_id, embedding, hard_negative_count + 1),
                ).fetchall()

                # Nearest neighbor is the pseudo-positive; following nearest
                # neighbors are hard negatives without a large in-memory matrix.
                if not neighbor_rows:
                    continue
                positive = neighbor_rows[0]
                negatives = neighbor_rows[1:hard_negative_count + 1]
                for negative in negatives:
                    triplet = {
                        "anchor": text,
                        "positive": positive[3],
                        "negative": negative[3],
                        "anchor_id": patent_id,
                        "positive_id": positive[1],
                        "negative_id": negative[1],
                        "anchor_source_file": source_file,
                        "positive_source_file": positive[0],
                        "negative_source_file": negative[0],
                    }
                    output.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                    triplet_count += 1

                if (i + 1) % 5000 == 0:
                    print(f"  Created triplets for {i + 1}/{len(rows)} patents...")

    print(f"\n✓ Created {triplet_count} triplets")
    print(f"  Saved to {output_file} ({output_file.stat().st_size / (1024*1024):.2f} MB)")

    # Save metadata
    metadata = {
        "total_patents": len(rows),
        "total_triplets": triplet_count,
        "embedding_model": embedding_model,
        "hard_negative_count": hard_negative_count,
        "neighbor_source": "existing_pgvector_embeddings",
        "triplets_per_anchor": triplet_count / len(rows) if rows else 0,
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare embedding fine-tuning dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "embedding_training_data",
        help="Output directory"
    )
    parser.add_argument(
        "--database-url",
        help="PostgreSQL connection URL (overrides config)"
    )
    parser.add_argument(
        "--embedding-model",
        help="Embedding model (overrides config)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Limit number of patents for testing (None = all)"
    )
    parser.add_argument(
        "--hard-negative-count",
        type=int,
        default=3,
        help="Number of hard negatives per anchor"
    )
    
    args = parser.parse_args()
    prepare_embedding_dataset(
        args.output,
        args.database_url,
        args.embedding_model,
        args.sample_size,
        args.hard_negative_count,
    )


if __name__ == "__main__":
    main()
