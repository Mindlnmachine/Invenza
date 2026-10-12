"""
Fine-tune qwen2.5-3b using QLoRA on patent domain data.

This script performs domain pre-training of qwen2.5-3b on patent text using QLoRA
(Quantized LoRA) for memory efficiency.

Usage:
    python scripts/finetune_qwen.py --data data/patent_domain_data.jsonl --config config/qwen_qlora_config.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def load_config(config_file: Path) -> dict:
    """Load QLoRA configuration."""
    with open(config_file, "r") as f:
        return json.load(f)


def load_dataset_from_jsonl(jsonl_file: Path, max_samples: Optional[int] = None) -> Dataset:
    """Load dataset from JSONL file."""
    texts = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            record = json.loads(line)
            texts.append(record["text"])
    
    dataset = Dataset.from_dict({"text": texts})
    return dataset


def preprocess_function(examples, tokenizer, max_length=512):
    """Preprocess examples for causal language modeling."""
    return tokenizer(
        examples["text"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


def finetune_qwen(
    data_file: Path,
    config_file: Path,
    output_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
    resume_from_checkpoint: Optional[Path] = None,
):
    """
    Fine-tune qwen2.5-3b using QLoRA on patent domain data.
    
    Args:
        data_file: Path to JSONL training data
        config_file: Path to QLoRA configuration file
        output_dir: Output directory for fine-tuned model (overrides config)
        max_samples: Limit examples (for testing)
        resume_from_checkpoint: Path to resume training from
    """
    # Load configuration
    config = load_config(config_file)
    training_config = config["training"]
    lora_config_dict = config["lora"]
    quant_config_dict = config["quantization"]
    
    output_dir = output_dir or Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("QLoRA Fine-tuning Configuration")
    print("=" * 80)
    print(f"Base model: {config['model_name']}")
    print(f"Training data: {data_file}")
    print(f"Output directory: {output_dir}")
    print(f"Max samples: {max_samples or 'all'}")
    print(f"Epochs: {training_config['num_train_epochs']}")
    print(f"Batch size (per device): {training_config['per_device_train_batch_size']}")
    print(f"Gradient accumulation steps: {training_config['gradient_accumulation_steps']}")
    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"LoRA rank: {lora_config_dict['rank']}")
    print("=" * 80)
    
    # Load tokenizer and model
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading quantized model...")
    
    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_config_dict.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=quant_config_dict.get("bnb_4bit_use_double_quant", True),
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Optional: faster attention
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=lora_config_dict["rank"],
        lora_alpha=lora_config_dict["alpha"],
        lora_dropout=lora_config_dict["dropout"],
        bias=lora_config_dict.get("bias", "none"),
        target_modules=lora_config_dict["target_modules"],
        modules_to_save=lora_config_dict.get("modules_to_save", []),
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print(f"\nLoading dataset from {data_file}...")
    dataset = load_dataset_from_jsonl(data_file, max_samples)
    print(f"Loaded {len(dataset)} examples")
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length=512),
        batched=True,
        remove_columns=["text"],
    )
    
    # Split into train and validation
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Train set: {len(train_dataset)}")
    print(f"Eval set: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        warmup_ratio=training_config["warmup_ratio"],
        weight_decay=training_config["weight_decay"],
        max_grad_norm=training_config["max_grad_norm"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        eval_steps=training_config.get("eval_steps", 500),
        save_total_limit=training_config["save_total_limit"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=training_config["seed"],
        optim=training_config.get("optim", "paged_adamw_8bit"),
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "logs"),
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Start training
    print("\nStarting training...")
    print("=" * 80)
    
    resume_from_checkpoint_path = None
    if resume_from_checkpoint:
        resume_from_checkpoint_path = str(resume_from_checkpoint)
        print(f"Resuming from checkpoint: {resume_from_checkpoint_path}")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint_path)
    
    # Save final model and adapter
    print("\nSaving model...")
    model.save_pretrained(str(output_dir / "adapter_model"))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save training metadata
    metadata = {
        "base_model": config["model_name"],
        "training_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "total_epochs": training_config["num_train_epochs"],
        "final_checkpoint": str(output_dir),
        "config": config,
    }
    
    metadata_file = output_dir / "training_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"  Model saved to: {output_dir}")
    print(f"  LoRA adapter: {output_dir / 'adapter_model'}")
    print(f"  Tokenizer: {output_dir}")
    print(f"  Metadata: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune qwen2.5-3b using QLoRA")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "patent_domain_data.jsonl",
        help="Path to JSONL training data"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "qwen_qlora_config.json",
        help="Path to QLoRA configuration file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit examples for testing (None = all)"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume training from checkpoint"
    )
    
    args = parser.parse_args()
    
    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}")
        print(f"Run: python scripts/extract_training_data.py --output {args.data}")
        return
    
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return
    
    finetune_qwen(
        args.data,
        args.config,
        args.output,
        args.max_samples,
        args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
