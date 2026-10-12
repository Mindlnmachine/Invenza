from __future__ import annotations

import re
from typing import Iterable

from .models import PatentRecord


def clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    text = re.sub(r"\s+", " ", text)
    return text or None


def normalize_cpc_ipc(value: object) -> list[str]:
    if value is None:
        return []
    cleaned = str(value).upper().replace(";", " ").replace(",", " ")
    cleaned = re.sub(r"[^A-Z0-9/\- ]", " ", cleaned)
    parts = [part.strip() for part in cleaned.split() if part.strip()]
    normalized: list[str] = []
    for part in parts:
        part = part.strip().replace(" ", "")
        if part:
            normalized.append(part)
    return normalized


def build_embedding_text(record: PatentRecord) -> str:
    fields = {
        "Title": record.title,
        "Abstract": record.abstract,
        "Claims": record.claims,
        "Assignee": record.assignee,
        "Inventors": record.inventors,
        "Filing date": record.filing_date,
        "Publication date": record.publication_date,
        "Grant date": record.grant_date,
        "CPC codes": ", ".join(record.cpc_codes),
        "IPC codes": ", ".join(record.ipc_codes),
    }
    return "\n".join(f"{label}: {clean_text(value) or ''}" for label, value in fields.items())


def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current_len and current_len + len(sentence) > max_chars:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current).strip())

    if not chunks:
        return [cleaned[:max_chars]]

    # soft overlap for improved retrieval quality
    if len(chunks) > 1:
        overlapped: list[str] = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                overlapped.append(chunk)
                continue
            prev = overlapped[-1]
            if len(chunk) < overlap:
                overlapped.append(chunk)
                continue
            overlapped.append(f"{prev[-overlap:]} {chunk}".strip())
        return overlapped

    return chunks


def normalize_topic_tokens(topic: str) -> list[str]:
    if not topic:
        return []
    topic = topic.lower()
    topic = re.sub(r"[^a-z0-9\s-]", " ", topic)
    return [token for token in re.split(r"\s+", topic) if token.strip()]


def as_patent_record(row: dict[str, object], source_file: str) -> PatentRecord:
    patent_id = clean_text(row.get("id")) or clean_text(row.get("patent_id"))
    if not patent_id:
        raise ValueError("Patent row missing an ID")

    record = PatentRecord(
        source_file=source_file,
        patent_id=patent_id,
        title=clean_text(row.get("title")),
        abstract=clean_text(row.get("abstract")),
        claims=clean_text(row.get("claims")),
        assignee=clean_text(row.get("assignee")),
        inventors=clean_text(row.get("inventor/author")) or clean_text(row.get("inventors")),
        filing_date=clean_text(row.get("filing/creation date")) or clean_text(row.get("filing_date")),
        publication_date=clean_text(row.get("publication date")) or clean_text(row.get("publication_date")),
        grant_date=clean_text(row.get("grant date")) or clean_text(row.get("grant_date")),
        cpc_codes=normalize_cpc_ipc(row.get("cpc")) or normalize_cpc_ipc(row.get("cpc codes")),
        ipc_codes=normalize_cpc_ipc(row.get("ipc")) or normalize_cpc_ipc(row.get("ipc codes")),
        citations=[],
        result_link=clean_text(row.get("result link")) or clean_text(row.get("result_link")),
        representative_figure_link=clean_text(row.get("representative figure link")) or clean_text(row.get("representative_figure_link")),
    )
    record.embedding_text = build_embedding_text(record)
    return record
