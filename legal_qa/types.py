from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable section of a source document."""

    id: str
    text: str
    source: str
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved chunk and its similarity score."""

    chunk: DocumentChunk
    score: float


@dataclass(frozen=True)
class Answer:
    """Final answer returned by the assistant."""

    question: str
    answer: str
    confidence: float
    evidence: str
    source: str
    chunk_id: str
    retrieval_score: float
    model: str
    found: bool
