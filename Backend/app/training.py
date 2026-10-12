from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .grounding import compute_topic_overlap
from .preprocessing import normalize_topic_tokens


@dataclass
class TrainingExample:
    topic: str
    patent_ids: list[str]
    idea_text: str
    retrieval_similarity: float
    citation_overlap: float
    topic_alignment: float
    text_consistency: float
    label: float


class WeakLabelGenerator:
    """Generate weak supervision labels from retrieval and evidence signals."""

    @staticmethod
    def build_label(
        retrieval_similarity: float,
        citation_overlap: float,
        topic_alignment: float,
        text_consistency: float,
    ) -> float:
        score = (
            0.45 * retrieval_similarity
            + 0.25 * citation_overlap
            + 0.20 * topic_alignment
            + 0.10 * text_consistency
        )
        return min(1.0, max(0.0, score)) * 100.0

    @staticmethod
    def from_record(topic: str, patent_ids: list[str], idea_text: str, retrieval_similarity: float, citation_overlap: float) -> TrainingExample:
        topic_terms = normalize_topic_tokens(topic)
        topic_alignment = compute_topic_overlap(topic_terms, idea_text)
        text_consistency = min(1.0, (retrieval_similarity + topic_alignment) / 2.0)
        label = WeakLabelGenerator.build_label(
            retrieval_similarity=retrieval_similarity,
            citation_overlap=citation_overlap,
            topic_alignment=topic_alignment,
            text_consistency=text_consistency,
        )
        return TrainingExample(
            topic=topic,
            patent_ids=patent_ids,
            idea_text=idea_text,
            retrieval_similarity=retrieval_similarity,
            citation_overlap=citation_overlap,
            topic_alignment=topic_alignment,
            text_consistency=text_consistency,
            label=label,
        )


class GroundingDatasetBuilder:
    """Build a lightweight training dataset for the grounding regressor."""

    def __init__(self, output_path: str | Path | None = None):
        self.output_path = Path(output_path) if output_path else Path("grounding_dataset.csv")

    def build(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for row in rows:
            example = WeakLabelGenerator.from_record(
                topic=row["topic"],
                patent_ids=row.get("patent_ids", []),
                idea_text=row["idea_text"],
                retrieval_similarity=float(row.get("retrieval_similarity", 0.0)),
                citation_overlap=float(row.get("citation_overlap", 0.0)),
            )
            records.append(
                {
                    "topic": example.topic,
                    "patent_ids": "|".join(example.patent_ids),
                    "idea_text": example.idea_text,
                    "retrieval_similarity": example.retrieval_similarity,
                    "citation_overlap": example.citation_overlap,
                    "topic_alignment": example.topic_alignment,
                    "text_consistency": example.text_consistency,
                    "label": example.label,
                }
            )

        df = pd.DataFrame(records)
        df.to_csv(self.output_path, index=False)
        return df


def train_grounding_model(dataset_path: str | Path) -> dict[str, Any]:
    """Architecture-only scaffold for the final phase 5 training strategy.

    This module defines the weak-supervision approach and the data shape required for a
    small transformer regressor. A full training loop would use a cross-encoder or a MiniLM
    backbone and optimize MSE on the label column.
    """
    df = pd.read_csv(dataset_path)
    metrics = {
        "rows": int(len(df)),
        "mean_label": round(float(df["label"].mean()), 2),
        "target_band_count": int(((df["label"] >= 75) & (df["label"] <= 80)).sum()),
        "min_label": round(float(df["label"].min()), 2),
        "max_label": round(float(df["label"].max()), 2),
    }
    return metrics
