from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_dataset import NO_ANSWER, build_document, normalize


DEFAULT_MODEL = "nlpaueb/legal-bert-base-uncased"
DATASET_PATH = ROOT / "data" / "leqal_qa_dataset.csv"
OUTPUT_DIR = ROOT / "models" / "legal-bert-qa"


@dataclass(frozen=True)
class QAExample:
    question: str
    context: str
    answer_text: str
    answer_start: int


class QADataset:
    def __init__(self, features: list[dict[str, object]]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in self.features[index].items()
        }


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_examples(rows: list[dict[str, str]]) -> list[QAExample]:
    document_text = build_document(rows)
    examples: list[QAExample] = []

    for row in rows:
        answer = row["answer"]
        if normalize(answer) == normalize(NO_ANSWER):
            examples.append(
                QAExample(
                    question=row["question"],
                    context=document_text,
                    answer_text="",
                    answer_start=0,
                )
            )
            continue

        context = f"{row['section']}: {row['clause_text']}"
        answer_text = row["evidence"] or answer
        answer_start = context.lower().find(answer_text.lower())
        if answer_start < 0:
            answer_text = answer
            answer_start = context.lower().find(answer_text.lower())
        if answer_start < 0:
            continue

        examples.append(
            QAExample(
                question=row["question"],
                context=context,
                answer_text=answer_text,
                answer_start=answer_start,
            )
        )

    return examples


def encode_examples(
    examples: list[QAExample],
    tokenizer: object,
    *,
    max_length: int,
) -> list[dict[str, object]]:
    features: list[dict[str, object]] = []

    for example in examples:
        encoded = tokenizer(
            example.question,
            example.context,
            truncation="only_second",
            max_length=max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        sequence_ids = encoded.sequence_ids()

        if not example.answer_text:
            encoded["start_positions"] = 0
            encoded["end_positions"] = 0
            features.append(dict(encoded))
            continue

        answer_end = example.answer_start + len(example.answer_text)
        start_position = None
        end_position = None

        for index, (offset_start, offset_end) in enumerate(offsets):
            if sequence_ids[index] != 1:
                continue
            if start_position is None and offset_start <= example.answer_start < offset_end:
                start_position = index
            if offset_start < answer_end <= offset_end:
                end_position = index
                break

        if start_position is None or end_position is None:
            continue

        encoded["start_positions"] = start_position
        encoded["end_positions"] = end_position
        features.append(dict(encoded))

    return features


def split_features(features: list[dict[str, object]], *, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    shuffled = features[:]
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, round(len(shuffled) * 0.2))
    return shuffled[validation_size:], shuffled[:validation_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Legal-BERT for extractive legal QA.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Hugging Face model downloads. By default, only cached local model files are used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install the ML dependencies from requirements.txt before training.") from exc

    allow_downloads = args.allow_downloads or os.environ.get("LEGAL_QA_ALLOW_MODEL_DOWNLOADS") == "1"
    rows = load_rows(args.dataset)
    examples = build_examples(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not allow_downloads)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model, local_files_only=not allow_downloads)

    features = encode_examples(examples, tokenizer, max_length=args.max_length)
    if len(features) < 2:
        raise SystemExit("Not enough trainable QA examples were encoded.")

    train_features, validation_features = split_features(features, seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    train_loader = DataLoader(QADataset(train_features), batch_size=args.batch_size, shuffle=True)
    epoch_count = max(1, round(args.epochs))

    for epoch in range(epoch_count):
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.detach().cpu())
        average_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1}/{epoch_count} - train loss: {average_loss:.4f}")

    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print(f"Fine-tuned {args.model} on {len(train_features)} examples.")
    print(f"Validation examples: {len(validation_features)}")
    print(f"Saved model to {args.output_dir.resolve().relative_to(ROOT)}")


if __name__ == "__main__":
    main()
