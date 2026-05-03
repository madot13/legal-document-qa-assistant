from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_qa.assistant import LegalQAAssistant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask questions over legal documents.")
    parser.add_argument("--document", "-d", action="append", required=True, help="Path to a legal document.")
    parser.add_argument("--question", "-q", required=True, help="Question to answer.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    parser.add_argument("--reader", choices=["lexical", "transformers"], default="lexical")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model", default="deepset/roberta-base-squad2", help="Hugging Face QA model.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assistant = LegalQAAssistant.from_paths(
        [Path(path) for path in args.document],
        retriever=args.retriever,
        reader=args.reader,
        embedding_model=args.embedding_model,
        qa_model=args.model,
    )
    result = assistant.ask(args.question, top_k=args.top_k)

    if args.json:
        print(json.dumps(result.__dict__, indent=2))
        return

    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Source: {result.source}")
    print(f"Chunk: {result.chunk_id}")
    print(f"Retrieval score: {result.retrieval_score:.3f}")
    if result.evidence:
        print("\nEvidence:")
        print(result.evidence)


if __name__ == "__main__":
    main()
