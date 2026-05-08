from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legal_qa.assistant import LegalQAAssistant
from legal_qa.cuad_adapter import CUADAdapter
from legal_qa.tokenization import tokenize


RESULTS_DIR = ROOT / "results"
DETAILS_CSV = RESULTS_DIR / "cuad_evaluation_results.csv"
SUMMARY_JSON = RESULTS_DIR / "cuad_evaluation_summary.json"
METHODS_CSV = RESULTS_DIR / "cuad_method_comparison.csv"
NO_ANSWER = "Not found in the provided document."


def normalize(text: object) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    return re.sub(r"[^a-z0-9\s]", "", text)


def token_f1(expected: object, predicted: object) -> float:
    expected_tokens = tokenize(str(expected or ""), keep_stopwords=True)
    predicted_tokens = tokenize(str(predicted or ""), keep_stopwords=True)
    if not expected_tokens and not predicted_tokens:
        return 1.0
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


def answer_contains_expected(expected: object, answer: object, evidence: object) -> bool:
    expected_text = normalize(expected)
    predicted_text = normalize(f"{answer} {evidence}")
    return bool(expected_text and expected_text in predicted_text)


def group_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("title") or ""),
        str(row.get("category") or ""),
        str(row.get("question") or ""),
    )


def grouped_gold_answers(rows: list[dict[str, object]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(
            {
                "answer": str(row.get("answer") or ""),
                "evidence": str(row.get("evidence") or row.get("answer") or ""),
            }
        )
    return groups


def any_group_match(gold_items: list[dict[str, str]], answer: object, evidence: object) -> bool:
    return any(
        answer_contains_expected(item["answer"], answer, evidence)
        or answer_contains_expected(item["evidence"], answer, evidence)
        for item in gold_items
    )


def load_cuad_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    adapter = CUADAdapter()
    if args.demo:
        if not adapter._load_demo_data():
            raise SystemExit("Could not load demo CUAD examples.")
    elif args.local_json:
        if not adapter.load_from_json(args.local_json):
            raise SystemExit(f"Could not load CUAD JSON: {args.local_json}")
    else:
        load_limit = None if args.category else args.limit
        if not adapter.load_from_huggingface(limit=load_limit, use_demo_on_fail=args.demo_on_fail):
            raise SystemExit(f"Could not load CUAD from Hugging Face: {adapter.last_error}")

    rows = adapter.qa_pairs
    if args.category:
        rows = [row for row in rows if row.get("category") == args.category]
    if args.limit:
        rows = rows[: args.limit]
    return rows


def make_assistant_context(row: dict[str, object], rows: list[dict[str, object]], mode: str) -> str:
    if mode == "evidence-context":
        return str(row.get("evidence") or row.get("answer") or "")

    if mode == "gold-context":
        return str(row.get("context") or "")

    contexts = []
    seen = set()
    for item in rows:
        context = str(item.get("context") or "")
        if context and context not in seen:
            contexts.append(context)
            seen.add(context)
    return "\n\n".join(contexts)


def retrieval_hit(expected_evidence: object, assistant: LegalQAAssistant, question: str, top_k: int) -> bool:
    expected_text = normalize(expected_evidence)
    if not expected_text:
        return False
    retrievals = assistant.retriever.search(question, top_k=top_k)
    retrieved_text = normalize(" ".join(item.chunk.text for item in retrievals))
    return expected_text in retrieved_text


def evaluate_method(
    rows: list[dict[str, object]],
    *,
    method_name: str,
    retriever: str,
    reader: str,
    qa_model: str | None,
    context_mode: str,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    evaluated = []
    exact_total = 0
    evidence_total = 0
    grouped_total = 0
    retrieval_total = 0
    f1_scores = []
    gold_groups = grouped_gold_answers(rows)

    for index, row in enumerate(rows):
        question = str(row.get("question") or "")
        expected_answer = str(row.get("answer") or "")
        expected_evidence = str(row.get("evidence") or expected_answer)
        context = make_assistant_context(row, rows, context_mode)
        source = f"CUAD {context_mode}: {row.get('title') or index}"

        assistant_kwargs = {"retriever": retriever, "reader": reader}
        if qa_model:
            assistant_kwargs["qa_model"] = qa_model

        assistant = LegalQAAssistant.from_text(context, source=source, **assistant_kwargs)
        result = assistant.ask(question, top_k=top_k)

        answer_match = answer_contains_expected(expected_answer, result.answer, result.evidence)
        evidence_match = answer_contains_expected(expected_evidence, result.answer, result.evidence)
        group_match = any_group_match(gold_groups[group_key(row)], result.answer, result.evidence)
        hit = retrieval_hit(expected_evidence, assistant, question, top_k)
        f1 = token_f1(expected_answer, result.answer)

        exact_total += int(answer_match)
        evidence_total += int(evidence_match)
        grouped_total += int(group_match)
        retrieval_total += int(hit)
        f1_scores.append(f1)

        evaluated.append(
            {
                "method": method_name,
                "context_mode": context_mode,
                "id": row.get("id", ""),
                "title": row.get("title", ""),
                "category": row.get("category", ""),
                "question": question,
                "expected_answer": expected_answer,
                "predicted_answer": result.answer,
                "expected_evidence": expected_evidence,
                "predicted_evidence": result.evidence,
                "retrieval_hit": int(hit),
                "answer_match": int(answer_match),
                "evidence_match": int(evidence_match),
                "group_match": int(group_match),
                "token_f1": f"{f1:.4f}",
                "confidence": f"{result.confidence:.4f}",
                "retrieval_score": f"{result.retrieval_score:.4f}",
                "model": result.model,
            }
        )

    total = len(rows)
    summary = {
        "method": method_name,
        "context_mode": context_mode,
        "total": total,
        "retrieval_hit_rate": retrieval_total / total if total else 0.0,
        "answer_match_rate": exact_total / total if total else 0.0,
        "evidence_match_rate": evidence_total / total if total else 0.0,
        "group_match_rate": grouped_total / total if total else 0.0,
        "average_token_f1": sum(f1_scores) / total if total else 0.0,
    }
    return summary, evaluated


def available_methods(args: argparse.Namespace) -> list[tuple[str, str, str, str | None]]:
    methods = [
        ("bm25+lexical", "bm25", "lexical", None),
        ("dense+lexical", "dense", "lexical", None),
        ("bm25+transformers", "bm25", "transformers", None),
        ("dense+transformers", "dense", "transformers", None),
    ]
    if args.qa_model:
        model_label = args.qa_model.split("/")[-1]
        methods.append((f"bm25+{model_label}", "bm25", "transformers", args.qa_model))
        methods.append((f"dense+{model_label}", "dense", "transformers", args.qa_model))

    legal_bert = ROOT / "models" / "legal-bert-qa"
    if legal_bert.exists():
        methods.append(("bm25+legal-bert", "bm25", "transformers", str(legal_bert)))
        methods.append(("dense+legal-bert", "dense", "transformers", str(legal_bert)))

    if args.methods == "all":
        return methods

    requested = set(args.methods.split(","))
    return [method for method in methods if method[0] in requested]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Legal QA pipeline on CUAD examples.")
    parser.add_argument("--limit", type=int, default=25, help="Maximum CUAD examples to evaluate.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category", help="Optional CUAD category filter, for example 'Warranty Duration'.")
    parser.add_argument("--local-json", type=Path, help="Optional local CUAD_v1.json path.")
    parser.add_argument("--demo", action="store_true", help="Use built-in demo CUAD-style examples.")
    parser.add_argument("--demo-on-fail", action="store_true", help="Use demo examples if Hugging Face CUAD loading fails.")
    parser.add_argument("--qa-model", help="Optional Hugging Face/local QA model to include in comparison.")
    parser.add_argument(
        "--context-mode",
        choices=["evidence-context", "gold-context", "full-document", "all"],
        default="all",
        help="evidence-context tests only the reader; gold-context tests chunk retrieval in the selected contract; full-document tests retrieval across loaded CUAD rows.",
    )
    parser.add_argument(
        "--methods",
        default="bm25+lexical,bm25+legal-bert",
        help="Comma-separated methods or 'all'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = load_cuad_rows(args)
    if not rows:
        raise SystemExit("No CUAD rows available for evaluation.")

    modes = (
        ["evidence-context", "gold-context", "full-document"]
        if args.context_mode == "all"
        else [args.context_mode]
    )
    details = []
    summaries = []

    for context_mode in modes:
        for method_name, retriever, reader, qa_model in available_methods(args):
            try:
                summary, evaluated = evaluate_method(
                    rows,
                    method_name=method_name,
                    retriever=retriever,
                    reader=reader,
                    qa_model=qa_model,
                    context_mode=context_mode,
                    top_k=args.top_k,
                )
                summaries.append({**summary, "status": "ok", "note": ""})
                details.extend(evaluated)
            except Exception as exc:
                summaries.append(
                    {
                        "method": method_name,
                        "context_mode": context_mode,
                        "total": len(rows),
                        "retrieval_hit_rate": "",
                        "answer_match_rate": "",
                        "evidence_match_rate": "",
                        "group_match_rate": "",
                        "average_token_f1": "",
                        "status": "unavailable",
                        "note": str(exc).splitlines()[0][:180],
                    }
                )

    summary_doc = {
        "total_rows": len(rows),
        "limit": args.limit,
        "category": args.category,
        "top_k": args.top_k,
        "summaries": summaries,
    }
    SUMMARY_JSON.write_text(json.dumps(summary_doc, indent=2), encoding="utf-8")
    write_rows(DETAILS_CSV, details)
    write_rows(METHODS_CSV, summaries)

    for item in summaries:
        if item["status"] != "ok":
            print(f"{item['method']} [{item['context_mode']}]: unavailable - {item['note']}")
            continue
        print(
            f"{item['method']} [{item['context_mode']}]: "
            f"retrieval_hit={float(item['retrieval_hit_rate']):.1%}, "
            f"answer_match={float(item['answer_match_rate']):.1%}, "
            f"evidence_match={float(item['evidence_match_rate']):.1%}, "
            f"group_match={float(item['group_match_rate']):.1%}, "
            f"token_f1={float(item['average_token_f1']):.1%}"
        )
    print(f"Wrote {METHODS_CSV.relative_to(ROOT)}")
    print(f"Wrote {DETAILS_CSV.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
