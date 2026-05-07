from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Sequence

from legal_qa.tokenization import tokenize
from legal_qa.types import DocumentChunk, RetrievalResult


class BM25Retriever:
    """Small BM25 retriever for legal document chunks."""

    def __init__(self, chunks: Sequence[DocumentChunk], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self._term_freqs = [Counter(tokens) for tokens in self._tokens]
        self._doc_freqs = self._build_doc_freqs()
        self._avg_doc_len = self._average_doc_len()

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        query_terms = tokenize(query)
        if not query_terms or not self.chunks:
            return []

        scored = [
            RetrievalResult(chunk=chunk, score=self._score(query_terms, index))
            for index, chunk in enumerate(self.chunks)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return [item for item in scored[:top_k] if item.score > 0]

    def _build_doc_freqs(self) -> Counter[str]:
        doc_freqs: Counter[str] = Counter()
        for tokens in self._tokens:
            doc_freqs.update(set(tokens))
        return doc_freqs

    def _average_doc_len(self) -> float:
        if not self._tokens:
            return 0.0
        return sum(len(tokens) for tokens in self._tokens) / len(self._tokens)

    def _idf(self, term: str) -> float:
        doc_count = len(self.chunks)
        freq = self._doc_freqs.get(term, 0)
        return math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))

    def _score(self, query_terms: list[str], index: int) -> float:
        tokens = self._tokens[index]
        if not tokens:
            return 0.0

        score = 0.0
        term_freqs = self._term_freqs[index]
        doc_len = len(tokens)
        avg_len = self._avg_doc_len or 1.0

        for term in query_terms:
            term_freq = term_freqs.get(term, 0)
            if term_freq == 0:
                continue
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
            score += self._idf(term) * numerator / denominator
        return score


class DenseRetriever:
    """Optional embedding retriever backed by sentence-transformers."""

    def __init__(self, chunks: Sequence[DocumentChunk], *, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval requires sentence-transformers and numpy. Install requirements.txt."
            ) from exc

        self._np = np
        self.chunks = list(chunks)
        self.model_name = model_name
        allow_downloads = os.environ.get("LEGAL_QA_ALLOW_MODEL_DOWNLOADS") == "1"
        self.model = SentenceTransformer(model_name, local_files_only=not allow_downloads)
        self.embeddings = self.model.encode(
            [chunk.text for chunk in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        if not query.strip() or not self.chunks:
            return []

        query_embedding = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.embeddings @ query_embedding
        indexes = self._np.argsort(scores)[::-1][:top_k]
        return [
            RetrievalResult(chunk=self.chunks[int(index)], score=float(scores[int(index)]))
            for index in indexes
            if float(scores[int(index)]) > 0
        ]
