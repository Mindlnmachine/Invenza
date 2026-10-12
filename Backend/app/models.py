from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PatentRecord:
    source_file: str
    patent_id: str
    title: str | None = None
    abstract: str | None = None
    claims: str | None = None
    assignee: str | None = None
    inventors: str | None = None
    filing_date: str | None = None
    publication_date: str | None = None
    grant_date: str | None = None
    cpc_codes: list[str] = field(default_factory=list)
    ipc_codes: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    result_link: str | None = None
    representative_figure_link: str | None = None
    embedding_text: str | None = None


@dataclass
class PatentChunk:
    patent_id: str
    chunk_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundingFeatures:
    retrieval_similarity: float
    citation_overlap: float
    topic_alignment: float
    text_consistency: float
    retrieval_rank: int = 1
    idea_text: str = ""
    patent_text: str = ""


@dataclass
class GroundingScoreResult:
    score: float
    score_rounded: float
    passes_target: bool
    source_patents: list[str]
    evidence_summary: str
    details: dict[str, float]


@dataclass
class IdeaOutput:
    concept_name: str
    summary: str
    source_patents: list[str]
    grounding_score: float
    evidence_summary: str
