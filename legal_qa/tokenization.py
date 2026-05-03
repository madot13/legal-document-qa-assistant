from __future__ import annotations

import re


LEGAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "shall",
    "should",
    "that",
    "the",
    "there",
    "this",
    "to",
    "under",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
}

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]*|\d+(?:\.\d+)?")


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Tokenize legal prose into normalized terms."""

    raw_tokens = [match.group(0).lower().strip("'") for match in TOKEN_PATTERN.finditer(text)]
    tokens = [_normalize_token(token) for token in raw_tokens if token]
    if keep_stopwords:
        return [token for token in tokens if token]
    return [token for token in tokens if token and token not in LEGAL_STOPWORDS]


def sentence_split(text: str) -> list[str]:
    """Split text into sentences without losing legal list fragments."""

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
    return [part.strip() for part in parts if part.strip()]


def _normalize_token(token: str) -> str:
    if token in LEGAL_STOPWORDS:
        return token
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token
