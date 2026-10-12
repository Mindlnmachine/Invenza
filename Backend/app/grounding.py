from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .models import GroundingFeatures, GroundingScoreResult


@dataclass
class GroundingModelFeatures:
    retrieval_similarity: float
    citation_overlap: float
    topic_alignment: float
    text_consistency: float
    retrieval_rank: int


class HybridGroundingScorer:
    def __init__(self, target_min: float = 75.0, target_max: float = 80.0):
        self.target_min = target_min
        self.target_max = target_max

    def calculate_formula(self, features: GroundingFeatures) -> float:
        score = (
            0.45 * max(0.0, min(1.0, features.retrieval_similarity))
            + 0.25 * max(0.0, min(1.0, features.citation_overlap))
            + 0.20 * max(0.0, min(1.0, features.topic_alignment))
            + 0.10 * max(0.0, min(1.0, features.text_consistency))
        )
        score *= 100.0

        # penalize weaker rankings slightly
        rank_penalty = max(0.0, (features.retrieval_rank - 1) * 2.5)
        score = max(0.0, score - rank_penalty)
        return min(100.0, score)

    def score(self, features: GroundingFeatures, source_patents: list[str], evidence_summary: str) -> GroundingScoreResult:
        raw_score = self.calculate_formula(features)
        rounded = round(raw_score, 2)
        passes_target = self.target_min <= rounded <= self.target_max

        details = {
            "retrieval_similarity": round(max(0.0, min(1.0, features.retrieval_similarity)) * 100, 2),
            "citation_overlap": round(max(0.0, min(1.0, features.citation_overlap)) * 100, 2),
            "topic_alignment": round(max(0.0, min(1.0, features.topic_alignment)) * 100, 2),
            "text_consistency": round(max(0.0, min(1.0, features.text_consistency)) * 100, 2),
        }

        return GroundingScoreResult(
            score=raw_score,
            score_rounded=rounded,
            passes_target=passes_target,
            source_patents=source_patents,
            evidence_summary=evidence_summary,
            details=details,
        )

    def citation_overlap_ratio(self, source_patent_ids: list[str], generated_text: str) -> float:
        if not source_patent_ids:
            return 0.0
        text = generated_text.lower()
        matches = sum(1 for patent_id in source_patent_ids if patent_id.lower() in text)
        return matches / max(1, len(source_patent_ids))


def compute_topic_overlap(topic_terms: list[str], text: str) -> float:
    if not topic_terms:
        return 0.0
    lowered = text.lower()
    matched = sum(1 for term in topic_terms if term.lower() in lowered)
    return matched / max(1, len(topic_terms))
