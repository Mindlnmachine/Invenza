"""
Fine-tune sentence-transformers on patent domain data.

This script fine-tunes a sentence-transformers model using contrastive learning
on patent data with triplet loss (anchor, positive, hard negative).

Usage:
    python scripts/finetune_embeddings.py --data data/embedding_training_data/triplets.jsonl
"""

import argparse
import json
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from torch.utils.data import DataLoader


def load_triplets_dataset(jsonl_file: Path, max_examples: int | None = None):
    """Load triplet dataset from JSONL file."""
    examples = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            triplet = json.loads(line)
            example = InputExample(
                texts=[
                    triplet["anchor"],
                    triplet["positive"],
                    triplet["negative"],
                ]
            )
            examples.append(example)
    return examples


def finetune_embeddings(
    data_file: Path,
    output_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 3,
    batch_size: int = 16,
    warmup_steps: int = 500,
    max_examples: int | None = None,
):
    """
    Fine-tune sentence-transformers on patent data.
    
    Args:
        data_file: Path to JSONL triplet file
        output_dir: Directory to save fine-tuned model
        model_name: Base model name
        epochs: Number of training epochs
        batch_size: Batch size for training
        warmup_steps: Warmup steps for learning rate scheduler
        max_examples: Limit examples (for testing)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The installed sentence-transformers version requires the "
            "'datasets' package for model.fit(). Run: python -m pip install datasets"
        ) from exc
    
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Loading triplet dataset from {data_file}")
    train_examples = load_triplets_dataset(data_file, max_examples)
    print(f"Loaded {len(train_examples)} triplet examples")
    
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Create loss function (Triplet loss)
    train_loss = losses.TripletLoss(model=model)
    
    # Optional: Create evaluator (for later use)
    # For now, we'll skip evaluation as we're domain pre-training
    
    print(f"Starting fine-tuning for {epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Output directory: {output_dir}")
    
    # Fine-tune
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )
    
    # Save model
    model.save(str(output_dir))
    print(f"\n✓ Fine-tuned model saved to {output_dir}")
    
    # Save configuration
    config = {
        "base_model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "total_examples": len(train_examples),
        "loss_function": "TripletLoss",
    }
    
    config_file = output_dir / "finetuning_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Configuration saved to {config_file}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune sentence-transformers on patent data")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "embedding_training_data" / "triplets.jsonl",
        help="Path to JSONL triplet file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "models" / "patent_embeddings_ft",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model name"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Warmup steps for learning rate scheduler"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Limit examples for testing (None = all)"
    )
    
    args = parser.parse_args()
    
    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}")
        print(f"Run: python scripts/prepare_embedding_dataset.py --output {args.data.parent}")
        return
    
    finetune_embeddings(
        args.data,
        args.output,
        args.model,
        args.epochs,
        args.batch_size,
        args.warmup_steps,
        args.max_examples,
    )


if __name__ == "__main__":
    main()
