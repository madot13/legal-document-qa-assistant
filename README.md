# Legal Document QA Assistant

Evidence-grounded question answering over legal documents and contracts.

This project starts with a dependency-light retrieval QA baseline and leaves clear extension points for transformer readers, dense embeddings, and dataset fine-tuning.

## What it does

- Loads legal text from `.txt`, `.md`, `.pdf`, or `.docx` files.
- Splits documents into traceable chunks with source metadata.
- Retrieves relevant clauses with BM25, or optional dense embeddings.
- Answers questions with either a lexical baseline or an optional Hugging Face extractive QA model.
- Returns an answer, supporting evidence, source chunk, confidence, and retrieval score.

This is a document assistance tool, not legal advice.

## Project layout

```text
legal_qa/
  assistant.py          High-level pipeline
  chunking.py           Document chunking
  document_loader.py    Text extraction
  reader.py             Lexical and transformer answer readers
  retriever.py          BM25 and optional dense retrievers
  tokenization.py       Legal-aware tokenization helpers
  types.py              Shared dataclasses
app.py                  Optional Streamlit UI
scripts/evaluate_sample.py
data/sample_contract.txt
data/sample_questions.jsonl
tests/
```

## Quick start

Run the offline baseline:

```bash
python -m legal_qa.cli \
  --document data/leqal_qa_dataset.csv \
  --question "Can either party assign the agreement?"
```

Run sample evaluation:

```bash
python scripts/evaluate_sample.py
```

Run tests:

```bash
python -m unittest discover -s tests
```

## Optional transformer QA

Install optional dependencies:

```bash
python -m pip install -r requirements.txt
```

Then run:

```bash
python -m legal_qa.cli \
  --document data/sample_contract.txt \
  --question "How long do confidentiality obligations survive?" \
  --reader transformers \
  --model deepset/roberta-base-squad2
```

The default pipeline is intentionally offline-friendly. The transformer reader downloads a model on first use.

## Optional Streamlit demo

```bash
streamlit run app.py
```

Upload a text, PDF, or DOCX contract, ask a question, and inspect the evidence chunk returned by the pipeline.

## Dataset path

Recommended datasets for the next phase:

- ContractNLI: document-level contract inference with evidence spans. https://stanfordnlp.github.io/contract-nli/
- CUAD: clause-level commercial contract annotations. https://www.atticusprojectai.org/cuad
- LegalQA: Chinese legal advice QA. https://github.com/siatnlp/LegalQA
- LexGLUE: broader legal NLP benchmark. https://github.com/coastalcph/lex-glue

## Suggested next milestones

1. Add dataset adapters for ContractNLI and CUAD.
2. Fine-tune a transformer QA or NLI model.
3. Add a persistent vector index.
4. Evaluate retrieval recall and evidence overlap on a held-out split.
5. Add page/section tracking for PDFs with OCR fallback.
