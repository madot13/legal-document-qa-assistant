---
title: Legal Document QA Assistant
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
suggested_hardware: cpu-basic
models:
  - deepset/roberta-base-squad2
  - sentence-transformers/all-MiniLM-L6-v2
datasets:
  - theatticusproject/cuad
---

# Demo link

https://madot12-legal-doc-qa-assistant.hf.space

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

## Installation and Deployment

### Prerequisites

- Python 3.9+
- Git

### Hugging Face Spaces Deployment

This repository is ready for a Docker-based Hugging Face Space. Create a new Space with `sdk: docker`, then push this repository to the Space repo. The included `Dockerfile` runs Streamlit on port `7860`, matching the `app_port: 7860` setting in this README metadata.

For the fine-tuned Legal-BERT checkpoint, use a separate Hugging Face model repository and set this Space variable:

```text
LEGAL_QA_MODEL=<your-username-or-org>/legal-bert-qa
```

The Docker image sets `LEGAL_QA_ALLOW_MODEL_DOWNLOADS=1`, so the Space can download the configured model at runtime. The local `models/` and `.hf_cache/` folders are intentionally ignored by git because they are large generated artifacts.

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ai_project_nlp
```

2. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Streamlit UI (Recommended)
```bash
streamlit run app.py
```

#### Option 2: Command Line Interface
```bash
# Offline baseline
python -m legal_qa.cli \
  --document data/sample_contract.txt \
  --question "Can either party assign the agreement?"
```

#### Option 3: Load CUAD Dataset
```bash
streamlit run app.py
# Select "Load CUAD Dataset" mode
# Click "Load CUAD Dataset from Hugging Face"
# Choose questions from dropdown
```

### Development

#### Running Tests
```bash
python -m unittest discover -s tests
```

#### Sample Evaluation
```bash
python scripts/evaluate_sample.py
```

## Quick start

Run the offline baseline:

```bash
python -m legal_qa.cli \
  --document data/sample_contract.txt \
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

### Supported Input Modes

1. **Upload Document** - Load individual legal documents (.txt, .md, .pdf, .docx)
2. **Upload QA Dataset (CSV)** - Load custom QA datasets with question-answer pairs
3. **Load CUAD Dataset** - Access the Contract Understanding Atticus Dataset (CUAD) from Hugging Face

The CUAD dataset provides:
- 13,000+ expert-annotated labels across 510 commercial contracts
- 41 categories of important legal clauses
- Real-world contract review scenarios
- Benchmark for legal NLP systems
