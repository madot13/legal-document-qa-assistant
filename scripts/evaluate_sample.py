from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legal_qa.assistant import LegalQAAssistant

SAMPLE_DOCUMENT = ROOT / "data" / "sample_contract.txt"
SAMPLE_QUESTIONS = ROOT / "data" / "sample_questions.jsonl"


def main() -> None:
    assistant = LegalQAAssistant.from_paths([SAMPLE_DOCUMENT])
    examples = [
        json.loads(line)
        for line in SAMPLE_QUESTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    passed = 0
    for example in examples:
        result = assistant.ask(example["question"], top_k=3)
        expected_terms = [term.lower() for term in example["expected_terms"]]
        evidence_text = f"{result.answer} {result.evidence}".lower()
        matched = all(term in evidence_text for term in expected_terms)
        passed += int(matched)
        status = "PASS" if matched else "FAIL"
        print(f"{status} | {example['question']}")
        print(f"  answer: {result.answer}")
        print(f"  expected terms: {', '.join(expected_terms)}")

    total = len(examples)
    print(f"\nSample evidence accuracy: {passed}/{total} = {passed / total:.2%}")


if __name__ == "__main__":
    main()
