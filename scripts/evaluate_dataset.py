from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legal_qa.assistant import LegalQAAssistant
from legal_qa.tokenization import tokenize


DATASET_PATH = ROOT / "data" / "leqal_qa_dataset.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "evaluation_results.csv"
SUMMARY_JSON = RESULTS_DIR / "evaluation_summary.json"
CHART_SVG = RESULTS_DIR / "category_accuracy.svg"
ERROR_ANALYSIS_CSV = RESULTS_DIR / "error_analysis_by_category.csv"
METHOD_COMPARISON_CSV = RESULTS_DIR / "method_comparison.csv"
METHOD_COMPARISON_JSON = RESULTS_DIR / "method_comparison.json"

NO_ANSWER = "Not found in the provided document."


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def token_f1(expected: str, predicted: str) -> float:
    if normalize(expected) == normalize(NO_ANSWER):
        return 1.0 if normalize(NO_ANSWER) in normalize(predicted) else 0.0

    expected_tokens = tokenize(expected, keep_stopwords=True)
    predicted_tokens = tokenize(predicted, keep_stopwords=True)
    if not expected_tokens or not predicted_tokens:
        return 0.0

    remaining = predicted_tokens.copy()
    overlap = 0
    for token in expected_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)

    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return (2 * precision * recall) / (precision + recall)


def load_rows() -> list[dict[str, str]]:
    with DATASET_PATH.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_document(rows: list[dict[str, str]]) -> str:
    seen: set[tuple[str, str, str]] = set()
    clauses: list[str] = []
    for row in rows:
        key = (row["contract_id"], row["section"], row["clause_text"])
        if key in seen:
            continue
        seen.add(key)
        if row["section"] == "No-Answer":
            continue
        clauses.append(f"{row['section']}: {row['clause_text']}")
    return "\n\n".join(clauses)


def score_prediction(row: dict[str, str], result: object) -> tuple[bool, bool, bool, bool]:
    prediction_text = normalize(f"{result.answer} {result.evidence}")
    expected_answer = normalize(row["answer"])
    expected_evidence = normalize(row["evidence"])
    is_no_answer = expected_answer == normalize(NO_ANSWER)

    if is_no_answer:
        no_answer_match = (not result.found) or normalize(NO_ANSWER) in prediction_text
        return False, False, no_answer_match, no_answer_match

    answer_match = bool(expected_answer and expected_answer in prediction_text)
    evidence_match = bool(expected_evidence and expected_evidence in prediction_text)
    return answer_match, evidence_match, False, answer_match or evidence_match


def evaluate_method(
    rows: list[dict[str, str]],
    document_text: str,
    *,
    retriever: str,
    reader: str,
    qa_model: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    assistant_kwargs = {
        "retriever": retriever,
        "reader": reader,
    }
    if qa_model:
        assistant_kwargs["qa_model"] = qa_model

    assistant = LegalQAAssistant.from_text(
        document_text,
        source=str(DATASET_PATH),
        **assistant_kwargs,
    )

    evaluated: list[dict[str, object]] = []
    by_category: dict[str, list[int]] = defaultdict(list)
    f1_scores: list[float] = []

    for row in rows:
        result = assistant.ask(row["question"], top_k=3)
        answer_match, evidence_match, no_answer_match, correct = score_prediction(row, result)
        f1 = token_f1(row["answer"], result.answer)
        f1_scores.append(f1)
        by_category[row["section"]].append(int(correct))

        evaluated.append(
            {
                "id": row["id"],
                "section": row["section"],
                "question": row["question"],
                "expected_answer": row["answer"],
                "predicted_answer": result.answer,
                "expected_evidence": row["evidence"],
                "retrieved_evidence": result.evidence,
                "confidence": f"{result.confidence:.4f}",
                "retrieval_score": f"{result.retrieval_score:.4f}",
                "answer_match": int(answer_match),
                "evidence_match": int(evidence_match),
                "no_answer_match": int(no_answer_match),
                "correct": int(correct),
                "token_f1": f"{f1:.4f}",
            }
        )

    total = len(evaluated)
    correct_total = sum(int(row["correct"]) for row in evaluated)
    exact_answer_total = sum(int(row["answer_match"]) for row in evaluated)
    evidence_total = sum(int(row["evidence_match"]) for row in evaluated)
    no_answer_total = sum(int(row["no_answer_match"]) for row in evaluated)
    category_accuracy = {
        category: {
            "correct": sum(values),
            "total": len(values),
            "accuracy": sum(values) / len(values) if values else 0.0,
        }
        for category, values in sorted(by_category.items())
    }

    summary: dict[str, object] = {
        "dataset": str(DATASET_PATH.relative_to(ROOT)),
        "total_questions": total,
        "correct": correct_total,
        "accuracy": correct_total / total if total else 0.0,
        "answer_match_rate": exact_answer_total / total if total else 0.0,
        "evidence_match_rate": evidence_total / total if total else 0.0,
        "no_answer_match_rate": no_answer_total / total if total else 0.0,
        "average_token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "retriever": retriever,
        "reader": reader,
        "top_k": 3,
        "category_accuracy": category_accuracy,
    }
    return summary, evaluated


def write_evaluation_results(evaluated: list[dict[str, object]]) -> None:
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(evaluated[0].keys()))
        writer.writeheader()
        writer.writerows(evaluated)


def write_error_analysis(summary: dict[str, object]) -> None:
    rows = []
    for category, stats in summary["category_accuracy"].items():
        total = int(stats["total"])
        correct = int(stats["correct"])
        rows.append(
            {
                "section": category,
                "total": total,
                "correct": correct,
                "errors": total - correct,
                "accuracy": f"{float(stats['accuracy']):.4f}",
            }
        )

    with ERROR_ANALYSIS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["section", "total", "correct", "errors", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def compare_methods(rows: list[dict[str, str]], document_text: str) -> list[dict[str, object]]:
    methods = [
        ("bm25+lexical", "bm25", "lexical", None),
        ("dense+lexical", "dense", "lexical", None),
        ("bm25+transformers", "bm25", "transformers", None),
        ("dense+transformers", "dense", "transformers", None),
    ]
    legal_bert_path = ROOT / "models" / "legal-bert-qa"
    if legal_bert_path.exists():
        methods.append(("bm25+legal-bert", "bm25", "transformers", str(legal_bert_path)))
    comparison: list[dict[str, object]] = []

    for method_name, retriever, reader, qa_model in methods:
        try:
            summary, _ = evaluate_method(
                rows,
                document_text,
                retriever=retriever,
                reader=reader,
                qa_model=qa_model,
            )
            comparison.append(
                {
                    "method": method_name,
                    "status": "ok",
                    "correct": summary["correct"],
                    "total_questions": summary["total_questions"],
                    "accuracy": round(float(summary["accuracy"]), 4),
                    "answer_match_rate": round(float(summary["answer_match_rate"]), 4),
                    "evidence_match_rate": round(float(summary["evidence_match_rate"]), 4),
                    "no_answer_match_rate": round(float(summary["no_answer_match_rate"]), 4),
                    "average_token_f1": round(float(summary["average_token_f1"]), 4),
                    "note": "",
                }
            )
        except Exception as exc:
            comparison.append(
                {
                    "method": method_name,
                    "status": "unavailable",
                    "correct": "",
                    "total_questions": len(rows),
                    "accuracy": "",
                    "answer_match_rate": "",
                    "evidence_match_rate": "",
                    "no_answer_match_rate": "",
                    "average_token_f1": "",
                    "note": str(exc).splitlines()[0][:180],
                }
            )

    with METHOD_COMPARISON_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)
    METHOD_COMPARISON_JSON.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def evaluate() -> dict[str, object]:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = load_rows()
    document_text = build_document(rows)

    summary, evaluated = evaluate_method(rows, document_text, retriever="bm25", reader="lexical")
    write_evaluation_results(evaluated)
    write_error_analysis(summary)
    method_comparison = compare_methods(rows, document_text)
    summary["method_comparison"] = method_comparison

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    CHART_SVG.write_text(build_category_chart(summary["category_accuracy"]), encoding="utf-8")
    return summary


def build_category_chart(category_accuracy: dict[str, dict[str, float]]) -> str:
    width = 860
    row_height = 34
    left = 180
    top = 46
    bar_width = 560
    height = top + row_height * len(category_accuracy) + 48
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="28" font-family="Arial" font-size="18" font-weight="700" fill="#1f2937">Accuracy by legal clause category</text>',
    ]
    for index, (category, stats) in enumerate(category_accuracy.items()):
        y = top + index * row_height
        accuracy = float(stats["accuracy"])
        fill_width = int(bar_width * accuracy)
        label = f"{accuracy:.0%} ({int(stats['correct'])}/{int(stats['total'])})"
        lines.extend(
            [
                f'<text x="24" y="{y + 20}" font-family="Arial" font-size="12" fill="#374151">{escape_xml(category)}</text>',
                f'<rect x="{left}" y="{y + 6}" width="{bar_width}" height="18" rx="4" fill="#e5e7eb"/>',
                f'<rect x="{left}" y="{y + 6}" width="{fill_width}" height="18" rx="4" fill="#2563eb"/>',
                f'<text x="{left + bar_width + 14}" y="{y + 20}" font-family="Arial" font-size="12" fill="#111827">{label}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines)


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main() -> None:
    summary = evaluate()
    print(f"Dataset accuracy: {summary['correct']}/{summary['total_questions']} = {summary['accuracy']:.2%}")
    print(f"Average token F1: {summary['average_token_f1']:.2%}")
    print(f"Wrote {RESULTS_CSV.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_JSON.relative_to(ROOT)}")
    print(f"Wrote {CHART_SVG.relative_to(ROOT)}")
    print(f"Wrote {ERROR_ANALYSIS_CSV.relative_to(ROOT)}")
    print(f"Wrote {METHOD_COMPARISON_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
