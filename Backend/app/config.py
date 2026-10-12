from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/postgres")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    csv_dir: Path = Path(os.getenv("CSV_DIR", Path(__file__).resolve().parents[2]))
    batch_size: int = int(os.getenv("BATCH_SIZE", "128"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    qwen_ollama_model: str = os.getenv("QWEN_OLLAMA_MODEL", "qwen3:4b")
    default_topic: str = os.getenv("DEFAULT_TOPIC", "precision agriculture")
    metric_target_min: float = 75.0
    metric_target_max: float = 80.0


settings = Settings()
