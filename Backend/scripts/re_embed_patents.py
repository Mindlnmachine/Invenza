"""
Re-embed all patents in PostgreSQL using fine-tuned embedding model.

This script loads the fine-tuned embedding model and re-encodes all patents,
updating the embeddings in PostgreSQL.

Usage:
    python scripts/re_embed_patents.py --model models/patent_embeddings_ft
"""

import argparse
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

# Prefer this project's Backend/app package over same-named packages in the environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings


def re_embed_patents(
    model_path: Path,
    database_url: str | None = None,
    batch_size: int = 128,
):
    """
    Re-embed all patents using fine-tuned embedding model.
    
    Args:
        model_path: Path to fine-tuned embedding model
        database_url: PostgreSQL connection URL
        batch_size: Batch size for encoding
    """
    database_url = database_url or settings.database_url
    
    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        return

    required_files = [model_path / "modules.json", model_path / "config_sentence_transformers.json"]
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        print(
            "Error: The embedding model directory is incomplete. "
            "Fine-tuning must finish successfully before re-embedding."
        )
        print(f"Missing files: {', '.join(missing_files)}")
        print("Run finetune_embeddings.py first, then run this script again.")
        return
    
    print(f"Loading fine-tuned embedding model from {model_path}")
    model = SentenceTransformer(str(model_path))
    print(f"  Model dimensions: {model.get_sentence_embedding_dimension()}")
    
    # Fetch all patent texts
    with psycopg.connect(database_url) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(connection)
        
        # Get total count
        count_result = connection.execute("SELECT COUNT(*) FROM patents").fetchone()
        total_count = count_result[0] if count_result else 0
        print(f"\nTotal patents to re-embed: {total_count}")
        
        # Fetch all patents in batches
        sql_fetch = "SELECT source_file, patent_id, embedding_text FROM patents ORDER BY source_file, patent_id"
        rows = connection.execute(sql_fetch).fetchall()
        
        print(f"Fetched {len(rows)} patents")
        
        # Re-encode in batches
        patents_data = [(source_file, patent_id, embedding_text) for source_file, patent_id, embedding_text in rows]
        texts = [embedding_text for _, _, embedding_text in patents_data]
        
        print(f"\nEncoding {len(texts)} embeddings (batch size: {batch_size})...")
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        
        print(f"Encoded {len(embeddings)} embeddings")
        
        # Update database in batches
        sql_update = """
        UPDATE patents
        SET embedding = %s
        WHERE source_file = %s AND patent_id = %s
        """
        
        print(f"\nUpdating database...")
        with connection.cursor() as cursor:
            for i, (source_file, patent_id, embedding_text) in enumerate(patents_data):
                embedding = embeddings[i].tolist()
                cursor.execute(sql_update, (embedding, source_file, patent_id))
                
                if (i + 1) % 5000 == 0:
                    connection.commit()
                    print(f"  Updated {i + 1}/{len(patents_data)} patents...")
            
            connection.commit()
            print(f"  Updated {len(patents_data)}/{len(patents_data)} patents (committed)")
    
    print(f"\n✓ Successfully re-embedded all {len(patents_data)} patents")
    print(f"  Model: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Re-embed patents using fine-tuned model")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent.parent / "models" / "patent_embeddings_ft",
        help="Path to fine-tuned embedding model"
    )
    parser.add_argument(
        "--database-url",
        help="PostgreSQL connection URL (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for encoding"
    )
    
    args = parser.parse_args()
    re_embed_patents(args.model, args.database_url, args.batch_size)


if __name__ == "__main__":
    main()
