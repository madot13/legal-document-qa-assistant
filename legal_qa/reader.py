from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from legal_qa.tokenization import sentence_split, tokenize
from legal_qa.types import RetrievalResult


NOT_FOUND_ANSWER = "Not found in the provided document."


@dataclass(frozen=True)
class ReaderOutput:
    answer: str
    confidence: float
    evidence: str
    model: str
    found: bool


class LexicalReader:
    """Offline answer extractor built for transparent baseline behavior."""

    model_name = "lexical-baseline"

    def answer(self, question: str, retrievals: Sequence[RetrievalResult]) -> ReaderOutput:
        if not retrievals:
            return ReaderOutput(NOT_FOUND_ANSWER, 0.0, "", self.model_name, False)

        best_retrieval = retrievals[0]
        if best_retrieval.score < 0.1:
            return ReaderOutput(NOT_FOUND_ANSWER, 0.0, "", self.model_name, False)

        evidence = self._best_sentence(question, best_retrieval.chunk.text) or best_retrieval.chunk.text
        answer = self._compose_answer(question, evidence)
        confidence = min(0.94, 0.35 + (best_retrieval.score / (best_retrieval.score + 6.0)) * 0.55)
        return ReaderOutput(answer, confidence, evidence, self.model_name, True)

    def _best_sentence(self, question: str, context: str) -> str:
        question_terms = set(tokenize(question))
        best_sentence = ""
        best_score = 0.0

        candidates = sentence_split(context) or [context]
        for sentence in candidates:
            sentence_terms = tokenize(sentence)
            if not sentence_terms:
                continue
            overlap = len(question_terms.intersection(sentence_terms))
            coverage = overlap / max(1, len(question_terms))
            density = overlap / max(1, len(set(sentence_terms)))
            score = (2 * coverage) + density
            if len(set(sentence_terms)) < 3:
                score *= 0.2
            if score > best_score:
                best_score = score
                best_sentence = sentence

        return best_sentence

    def _compose_answer(self, question: str, evidence: str) -> str:
        if _is_yes_no_question(question):
            normalized = evidence.lower()
            conditional_consent = re.search(r"\bwithout\b.{0,80}\b(consent|approval|permission)\b", normalized)
            if conditional_consent:
                return f"Only with required consent or approval. {evidence}"
            if _contains_negative_legal_signal(normalized):
                return f"No. {evidence}"
            if _contains_positive_legal_signal(normalized):
                return f"Yes. {evidence}"
        return evidence


class TransformersReader:
    """Optional extractive QA reader using Hugging Face transformers."""

    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Transformer QA requires transformers and torch. Install requirements.txt.") from exc

        self.model_name = model_name
        self._pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

    def answer(self, question: str, retrievals: Sequence[RetrievalResult]) -> ReaderOutput:
        if not retrievals:
            return ReaderOutput(NOT_FOUND_ANSWER, 0.0, "", self.model_name, False)

        context = "\n\n".join(item.chunk.text for item in retrievals)
        result = self._pipeline(question=question, context=context)
        answer = str(result.get("answer", "")).strip()
        confidence = float(result.get("score", 0.0))

        if not answer or confidence < 0.05:
            return ReaderOutput(NOT_FOUND_ANSWER, confidence, "", self.model_name, False)

        evidence = _sentence_containing_answer(context, answer) or retrievals[0].chunk.text
        return ReaderOutput(answer, confidence, evidence, self.model_name, True)


def _is_yes_no_question(question: str) -> bool:
    return bool(re.match(r"^\s*(can|could|may|must|shall|should|is|are|do|does|did|will|would|has|have)\b", question, re.I))


def _contains_negative_legal_signal(text: str) -> bool:
    patterns = [
        r"\bmay not\b",
        r"\bshall not\b",
        r"\bmust not\b",
        r"\bcannot\b",
        r"\bno party may\b",
        r"\bneither party may\b",
        r"\bnot permitted\b",
        r"\bprohibited\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _contains_positive_legal_signal(text: str) -> bool:
    patterns = [
        r"\bmay\b",
        r"\bshall\b",
        r"\bmust\b",
        r"\bcan\b",
        r"\bis permitted\b",
        r"\bis allowed\b",
        r"\bsurvive\b",
        r"\bcontinues?\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _sentence_containing_answer(context: str, answer: str) -> str:
    answer_lower = answer.lower()
    for sentence in sentence_split(context):
        if answer_lower in sentence.lower():
            return sentence
    return ""
