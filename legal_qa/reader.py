from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

from legal_qa.tokenization import sentence_split, tokenize
from legal_qa.types import RetrievalResult


NOT_FOUND_ANSWER = "Not found in the provided document."
QUESTION_GENERIC_TERMS = {
    "about",
    "action",
    "agreement",
    "allow",
    "allowed",
    "answer",
    "any",
    "apply",
    "ask",
    "before",
    "begin",
    "between",
    "cap",
    "clause",
    "contract",
    "could",
    "deliverable",
    "document",
    "does",
    "during",
    "event",
    "exclud",
    "free",
    "freely",
    "govern",
    "happen",
    "include",
    "includ",
    "large",
    "last",
    "law",
    "leas",
    "legislation",
    "long",
    "many",
    "may",
    "miss",
    "much",
    "must",
    "non-payment",
    "often",
    "other",
    "own",
    "party",
    "period",
    "promise",
    "prohibit",
    "provide",
    "purpose",
    "question",
    "quickly",
    "require",
    "required",
    "requir",
    "responsible",
    "section",
    "service",
    "set",
    "specific",
    "term",
    "third-party",
    "time",
    "type",
    "use",
    "used",
    "using",
    "weekly",
    "work",
}


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

        if not _is_answerable(question, retrievals):
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
            import torch
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Transformer QA requires transformers and torch. Install requirements.txt.") from exc

        self.model_name = model_name
        self._torch = torch
        allow_downloads = os.environ.get("LEGAL_QA_ALLOW_MODEL_DOWNLOADS") == "1"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=not allow_downloads)
        self._model = AutoModelForQuestionAnswering.from_pretrained(model_name, local_files_only=not allow_downloads)
        self._model.eval()
        self.fallback_threshold = float(os.environ.get("LEGAL_QA_TRANSFORMER_FALLBACK_THRESHOLD", "0.15"))

    def answer(self, question: str, retrievals: Sequence[RetrievalResult]) -> ReaderOutput:
        if not retrievals:
            return ReaderOutput(NOT_FOUND_ANSWER, 0.0, "", self.model_name, False)

        context = "\n\n".join(item.chunk.text for item in retrievals)
        inputs = self._tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=384,
        )

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        start_scores = self._torch.softmax(outputs.start_logits[0], dim=0)
        end_scores = self._torch.softmax(outputs.end_logits[0], dim=0)
        start_index = int(self._torch.argmax(start_scores))
        end_index = int(self._torch.argmax(end_scores))

        if end_index < start_index:
            start_index, end_index = end_index, start_index

        answer_ids = inputs["input_ids"][0][start_index : end_index + 1]
        answer = _post_process_answer_span(self._tokenizer.decode(answer_ids, skip_special_tokens=True))
        confidence = float(start_scores[start_index] * end_scores[end_index])

        # Low confidence fallback
        if not answer or confidence < self.fallback_threshold:
            best_retrieval = retrievals[0]
            fallback_evidence = best_retrieval.chunk.text
            fallback_answer = _best_fallback_answer(question, fallback_evidence)
            fallback_confidence = min(0.94, 0.35 + (best_retrieval.score / (best_retrieval.score + 6.0)) * 0.55)
            return ReaderOutput(
                answer=fallback_answer,
                confidence=fallback_confidence,
                evidence=fallback_evidence,
                model=f"{self.model_name} (fallback)",
                found=True
            )

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


def _best_fallback_answer(question: str, evidence: str) -> str:
    question_terms = set(tokenize(question))
    best_sentence = ""
    best_score = -1.0

    for sentence in sentence_split(evidence) or [evidence]:
        sentence_terms = set(tokenize(sentence))
        if not sentence_terms:
            continue
        overlap = len(question_terms & sentence_terms)
        score = (2 * overlap / max(1, len(question_terms))) + (overlap / max(1, len(sentence_terms)))
        if score > best_score:
            best_score = score
            best_sentence = sentence

    return _strip_section_prefix(best_sentence or evidence)


def _strip_section_prefix(text: str) -> str:
    text = re.sub(r"^=+\s*Section\s+\d+:\s*[^=]+===\s*", "", text).strip()
    text = re.sub(r"^[A-Z][A-Za-z ]{2,40}:\s+", "", text).strip()
    return text


def _is_answerable(question: str, retrievals: Sequence[RetrievalResult]) -> bool:
    question_terms = _specific_question_terms(question)
    if not question_terms:
        return True

    evidence_terms = set(tokenize(" ".join(item.chunk.text for item in retrievals)))
    supported_terms = {term for term in question_terms if _term_supported(term, evidence_terms)}
    missing_terms = question_terms - supported_terms
    coverage = len(supported_terms) / len(question_terms)

    if len(question_terms) >= 2 and coverage < 0.5:
        return False
    if len(missing_terms) >= 2 and coverage <= 0.6:
        return False
    if _is_yes_no_question(question) and missing_terms and coverage < 0.8:
        return False
    if any(len(term) >= 8 for term in missing_terms) and coverage < 0.75:
        return False
    return True


def _specific_question_terms(question: str) -> set[str]:
    return {
        term
        for term in tokenize(question)
        if len(term) > 2 and term not in QUESTION_GENERIC_TERMS
    }


def _term_supported(term: str, evidence_terms: set[str]) -> bool:
    if term in evidence_terms or f"{term}e" in evidence_terms:
        return True
    if len(term) <= 5:
        return False
    return any(
        abs(len(evidence_term) - len(term)) <= 2
        and (evidence_term.startswith(term) or term.startswith(evidence_term))
        for evidence_term in evidence_terms
    )


def _post_process_answer_span(answer: str) -> str:
    answer = re.sub(r"\s+", " ", answer).strip()
    answer = answer.strip(" \t\r\n\"'`.,;:()[]{}")
    answer = re.sub(r"\s+([,.;:!?])", r"\1", answer)
    return answer
