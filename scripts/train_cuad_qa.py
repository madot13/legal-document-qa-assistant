from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legal_qa.cuad_adapter import CUADAdapter
from scripts.train_legal_bert_qa import (
    DEFAULT_MODEL,
    OUTPUT_DIR,
    QADataset,
    QAExample,
    split_features,
)


def load_cuad_examples(args: argparse.Namespace) -> list[QAExample]:
    adapter = CUADAdapter()
    if args.demo:
        adapter._load_demo_data()
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

    examples: list[QAExample] = []
    for row in rows:
        answer_text = str(row.get("evidence") or row.get("answer") or "").strip()
        context = str(row.get("context") or "").strip()
        question = str(row.get("question") or "").strip()
        if not answer_text or not context or not question:
            continue

        answer_start = context.lower().find(answer_text.lower())
        if answer_start < 0:
            answer_text = str(row.get("answer") or "").strip()
            answer_start = context.lower().find(answer_text.lower()) if answer_text else -1
        if answer_start < 0:
            continue

        examples.append(
            QAExample(
                question=question,
                context=context,
                answer_text=answer_text,
                answer_start=answer_start,
            )
        )

    return examples


def encode_cuad_examples(
    examples: list[QAExample],
    tokenizer: object,
    *,
    max_length: int,
    doc_stride: int,
) -> list[dict[str, object]]:
    features: list[dict[str, object]] = []

    for example in examples:
        encoded = tokenizer(
            example.question,
            example.context,
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            padding="max_length",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        )

        answer_end = example.answer_start + len(example.answer_text)

        for feature_index in range(len(encoded["input_ids"])):
            offsets = encoded["offset_mapping"][feature_index]
            sequence_ids = encoded.sequence_ids(feature_index)

            token_start = None
            token_end = None

            for token_index, (offset_start, offset_end) in enumerate(offsets):
                if sequence_ids[token_index] != 1:
                    continue
                if token_start is None and offset_start <= example.answer_start < offset_end:
                    token_start = token_index
                if offset_start < answer_end <= offset_end:
                    token_end = token_index
                    break

            if token_start is None or token_end is None:
                continue

            feature = {
                key: values[feature_index]
                for key, values in encoded.items()
                if key not in {"offset_mapping", "overflow_to_sample_mapping"}
            }
            feature["start_positions"] = token_start
            feature["end_positions"] = token_end
            features.append(feature)
            break

    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the QA reader on CUAD examples.")
    parser.add_argument("--local-json", type=Path, help="Optional local CUAD_v1.json path.")
    parser.add_argument("--demo", action="store_true", help="Train on built-in demo examples.")
    parser.add_argument("--demo-on-fail", action="store_true", help="Use demo examples if Hugging Face CUAD loading fails.")
    parser.add_argument("--category", help="Optional category filter, for example 'Warranty Duration'.")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--doc-stride", type=int, default=128)
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
    examples = load_cuad_examples(args)
    if len(examples) < 2:
        raise SystemExit("Not enough trainable CUAD examples were found.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not allow_downloads)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model, local_files_only=not allow_downloads)

    features = encode_cuad_examples(
        examples,
        tokenizer,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
    )
    if len(features) < 2:
        raise SystemExit("Not enough CUAD features were encoded.")

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print(f"Fine-tuned {args.model} on {len(train_features)} CUAD examples.")
    print(f"Validation examples: {len(validation_features)}")
    try:
        saved_path = args.output_dir.resolve().relative_to(ROOT)
    except ValueError:
        saved_path = args.output_dir.resolve()
    print(f"Saved model to {saved_path}")


if __name__ == "__main__":
    main()
