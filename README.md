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
  - madot12/legal-bert-qa
  - sentence-transformers/all-MiniLM-L6-v2
datasets:
  - theatticusproject/cuad
---

# Legal Document QA Assistant

Evidence-grounded question answering over legal documents and contracts.

Live demo: https://madot12-legal-doc-qa-assistant.hf.space  
Fine-tuned model: https://huggingface.co/madot12/legal-bert-qa

## Project Overview

This project implements a retrieval-augmented legal QA pipeline. A user uploads a legal document or selects a QA dataset question, the system retrieves the most relevant contract clauses, and then extracts an answer with supporting evidence.

The application supports:

- TXT, Markdown, PDF, and DOCX document loading.
- Section-aware legal document chunking.
- BM25 keyword retrieval and dense semantic retrieval.
- Lexical baseline QA and transformer-based extractive QA.
- Fine-tuned Legal-BERT checkpoint for legal-domain question answering.
- Answerability detection for unsupported/no-answer questions.
- Dataset-level evaluation with CSV and JSON result outputs.

This is a document assistance tool, not legal advice.

## Project Structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── legal_qa/
│   ├── __init__.py
│   ├── assistant.py
│   ├── chunking.py
│   ├── cli.py
│   ├── cuad_adapter.py
│   ├── document_loader.py
│   ├── reader.py
│   ├── retriever.py
│   ├── tokenization.py
│   └── types.py
├── scripts/
│   ├── evaluate_dataset.py
│   ├── evaluate_sample.py
│   └── train_legal_bert_qa.py
├── data/
│   ├── leqal_qa_dataset.csv
│   ├── sample_contract.txt
│   └── sample_questions.jsonl
├── docs/
│   └── Legal_Document_QA_Final_Report.pdf
└── poster/
```

## Main Files

- `app.py` - Streamlit web interface for document upload, dataset QA, CUAD demo, method selection, and result display.
- `legal_qa/assistant.py` - High-level pipeline that connects chunking, retrieval, reading, and final answer formatting.
- `legal_qa/chunking.py` - Splits legal text into searchable chunks while preserving section context and source offsets.
- `legal_qa/retriever.py` - Implements BM25 keyword retrieval and optional dense embedding retrieval.
- `legal_qa/reader.py` - Implements lexical QA, transformer QA, answer post-processing, and no-answer detection.
- `legal_qa/document_loader.py` - Loads text from TXT, Markdown, PDF, and DOCX files.
- `legal_qa/cuad_adapter.py` - Loads CUAD data from Hugging Face or fallback demo examples.
- `scripts/train_legal_bert_qa.py` - Fine-tunes `nlpaueb/legal-bert-base-uncased` for extractive legal QA.
- `scripts/evaluate_dataset.py` - Runs dataset evaluation and method comparison.
- `data/leqal_qa_dataset.csv` - Self-created manually labeled legal QA dataset.
- `docs/Legal_Document_QA_Final_Report.pdf` - Final 3-5 page project documentation.

## Dataset

The main dataset is:

```text
data/leqal_qa_dataset.csv
```

It contains manually created legal QA examples with these columns:

```text
id, contract_id, section, clause_text, question, answer, evidence
```

The dataset includes multiple legal clause categories and harder no-answer questions. The application also supports CUAD-style examples through the Hugging Face dataset loader.

## Method

The system follows this workflow:

```text
document or dataset
  -> text loading
  -> section-aware chunking
  -> BM25 or dense retrieval
  -> lexical or transformer QA reader
  -> answer + evidence + confidence + retrieval score
```

The fine-tuned transformer model is based on `nlpaueb/legal-bert-base-uncased`. It was trained as an extractive QA model to predict the start and end token positions of the answer span inside the legal context.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run app.py
```

Then choose one of the input modes:

1. `Upload Document`
2. `Upload QA Dataset (CSV)`
3. `Load CUAD Dataset`

## Run from CLI

```bash
python -m legal_qa.cli \
  --document data/sample_contract.txt \
  --question "Can either party assign the agreement?"
```

Transformer mode:

```bash
LEGAL_QA_ALLOW_MODEL_DOWNLOADS=1 python -m legal_qa.cli \
  --document data/sample_contract.txt \
  --question "How long must confidential information be kept secret?" \
  --reader transformers \
  --model madot12/legal-bert-qa
```

## Fine-Tuning

The training script converts the CSV dataset into extractive QA examples:

```text
question + context + answer_start + answer_end
```

Run training:

```bash
LEGAL_QA_ALLOW_MODEL_DOWNLOADS=1 python scripts/train_legal_bert_qa.py \
  --allow-downloads \
  --epochs 1 \
  --batch-size 4 \
  --output-dir models/legal-bert-qa
```

The local `models/` directory is not included in the submission ZIP because the checkpoint is large. The deployed checkpoint is available here:

```text
https://huggingface.co/madot12/legal-bert-qa
```

## Evaluation

Run full dataset evaluation:

```bash
python scripts/evaluate_dataset.py
```

Generated outputs:

```text
results/evaluation_results.csv
results/evaluation_summary.json
results/method_comparison.csv
results/method_comparison.json
results/error_analysis_by_category.csv
results/category_accuracy.svg
```

Run the small sample evaluation:

```bash
python scripts/evaluate_sample.py
```

## Deployment

The project is deployed on Hugging Face Spaces with Docker.

Space URL:

```text
https://madot12-legal-doc-qa-assistant.hf.space
```

The Space uses this model variable:

```text
LEGAL_QA_MODEL=madot12/legal-bert-qa
```

The `Dockerfile` installs `requirements.txt` and starts Streamlit on port `7860`.