from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from legal_qa.chunking import chunk_many, chunk_text
from legal_qa.document_loader import load_text
from legal_qa.reader import LexicalReader, TransformersReader
from legal_qa.retriever import BM25Retriever, DenseRetriever
from legal_qa.types import Answer, DocumentChunk


class LegalQAAssistant:
    """High-level legal document QA pipeline."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        retriever: str = "bm25",
        reader: str = "lexical",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        qa_model: str = "deepset/roberta-base-squad2",
    ):
        self.chunks = list(chunks)
        self.retriever_name = retriever
        self.reader_name = reader

        if retriever == "bm25":
            self.retriever = BM25Retriever(self.chunks)
        elif retriever == "dense":
            self.retriever = DenseRetriever(self.chunks, model_name=embedding_model)
        else:
            raise ValueError(f"Unsupported retriever: {retriever}")

        if reader == "lexical":
            self.reader = LexicalReader()
        elif reader == "transformers":
            self.reader = TransformersReader(model_name=qa_model)
        else:
            raise ValueError(f"Unsupported reader: {reader}")

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[str | Path],
        *,
        max_tokens: int = 220,
        overlap_tokens: int = 40,
        **kwargs: object,
    ) -> "LegalQAAssistant":
        documents = []
        for path in paths:
            file_path = Path(path)
            documents.append((str(file_path), load_text(file_path)))
        chunks = chunk_many(documents, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        return cls(chunks, **kwargs)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        source: str = "uploaded-document",
        max_tokens: int = 220,
        overlap_tokens: int = 40,
        **kwargs: object,
    ) -> "LegalQAAssistant":
        chunks = chunk_text(text, source=source, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        return cls(chunks, **kwargs)

    def ask(self, question: str, *, top_k: int = 5) -> Answer:
        retrievals = self.retriever.search(question, top_k=top_k)
        reader_output = self.reader.answer(question, retrievals)

        best = retrievals[0] if retrievals else None
        return Answer(
            question=question,
            answer=reader_output.answer,
            confidence=reader_output.confidence,
            evidence=reader_output.evidence,
            source=best.chunk.source if best else "",
            chunk_id=best.chunk.id if best else "",
            retrieval_score=best.score if best else 0.0,
            model=reader_output.model,
            found=reader_output.found,
        )
