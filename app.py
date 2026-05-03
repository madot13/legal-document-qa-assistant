from __future__ import annotations

import tempfile
from pathlib import Path
import pandas as pd

import streamlit as st

from legal_qa.assistant import LegalQAAssistant
from legal_qa.document_loader import DocumentLoadError, load_text


def load_qa_dataset(csv_file):
    """Load QA dataset from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        # Ensure required columns exist
        required_cols = ['question', 'answer', 'clause_text', 'evidence']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {', '.join(required_cols)}")
            return None
        
        # Convert clause_text to document format
        all_clauses = df['clause_text'].tolist()
        document_text = "\n\n".join([f"Clause {i+1}: {clause}" for i, clause in enumerate(all_clauses)])
        
        return df, document_text
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None, None


st.set_page_config(page_title="Legal Document QA", layout="wide")

st.title("Legal Document QA Assistant")

# Choose input mode
input_mode = st.radio("Choose input mode:", ["Upload Document", "Upload QA Dataset (CSV)"])

# Initialize variables
uploaded_file = None
uploaded_csv = None
qa_dataset = None
document_text = ""
question = ""

if input_mode == "Upload Document":
    uploaded_file = st.file_uploader("Document", type=["txt", "md", "pdf", "docx"])
    question = st.text_input("Question", placeholder="Can either party assign the agreement?")
else:
    uploaded_csv = st.file_uploader("QA Dataset", type=["csv"])
    if uploaded_csv:
        qa_dataset, document_text = load_qa_dataset(uploaded_csv)
        if qa_dataset is not None:
            st.success(f"Loaded {len(qa_dataset)} QA pairs from dataset")
            
            # Show available questions
            st.subheader("Available Questions:")
            question_options = qa_dataset['question'].tolist()
            selected_question_idx = st.selectbox("Select a question:", range(len(question_options)), format_func=lambda i: f"{i+1}. {question_options[i]}")
            question = question_options[selected_question_idx]
            
            # Show expected answer for comparison
            with st.expander("Expected Answer"):
                st.write("**Answer:**", qa_dataset.iloc[selected_question_idx]['answer'])
                st.write("**Evidence:**", qa_dataset.iloc[selected_question_idx]['evidence'])
        else:
            st.stop()

with st.sidebar:
    st.header("Pipeline")
    retriever = st.selectbox("Retriever", ["bm25", "dense"])
    reader = st.selectbox("Reader", ["lexical", "transformers"])
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=5)
    qa_model = st.text_input("QA model", value="deepset/roberta-base-squad2")
    embedding_model = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")

if (uploaded_file and question) or (qa_dataset is not None and question):
    try:
        if input_mode == "Upload Document":
            # Handle document upload
            suffix = Path(uploaded_file.name).suffix or ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                temporary_file.write(uploaded_file.getbuffer())
                temporary_path = Path(temporary_file.name)

            text = load_text(temporary_path)
            source_name = uploaded_file.name
            
            # Clean up temp file
            temporary_path.unlink(missing_ok=True)
        else:
            # Handle CSV dataset
            text = document_text
            source_name = uploaded_csv.name if uploaded_csv else "QA Dataset"

        assistant = LegalQAAssistant.from_text(
            text,
            source=source_name,
            retriever=retriever,
            reader=reader,
            qa_model=qa_model,
            embedding_model=embedding_model,
        )
        result = assistant.ask(question, top_k=top_k)
        
    except DocumentLoadError as exc:
        st.error(str(exc))
        st.stop()
    except (RuntimeError, OSError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    metric_cols = st.columns(3)
    metric_cols[0].metric("Confidence", f"{result.confidence:.2f}")
    metric_cols[1].metric("Retrieval score", f"{result.retrieval_score:.3f}")
    metric_cols[2].metric("Chunks indexed", len(assistant.chunks))

    st.subheader("Answer")
    st.write(result.answer)

    st.subheader("Evidence")
    st.write(result.evidence or "No supporting evidence found.")

    st.caption(f"Source: {result.source} | Chunk: {result.chunk_id} | Model: {result.model}")

# If using CSV dataset, show comparison with expected answer
if qa_dataset is not None and question:
    expected_row = qa_dataset[qa_dataset['question'] == question]
    if not expected_row.empty:
        expected_answer = expected_row.iloc[0]['answer']
        expected_evidence = expected_row.iloc[0]['evidence']
        
        st.subheader("Expected Answer (from dataset)")
        st.write(f"**Answer:** {expected_answer}")
        st.write(f"**Evidence:** {expected_evidence}")
        
        # Simple comparison
        if expected_answer.lower().strip() in result.answer.lower().strip() or result.answer.lower().strip() in expected_answer.lower().strip():
            st.success("✓ Answer matches expected result!")
        else:
            st.warning("⚠ Answer differs from expected result")
else:
    if input_mode == "Upload Document":
        st.info("Upload a legal document and ask a question.")
    else:
        st.info("Upload a CSV dataset with QA pairs to get started.")
