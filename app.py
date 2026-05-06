from __future__ import annotations

import tempfile
from pathlib import Path
import pandas as pd

import streamlit as st

from legal_qa.assistant import LegalQAAssistant
from legal_qa.document_loader import DocumentLoadError, load_text
from legal_qa.cuad_adapter import CUADAdapter


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


def load_cuad_dataset():
    """Load CUAD dataset from Hugging Face"""
    try:
        cuad_adapter = CUADAdapter()
        
        with st.spinner("Loading CUAD dataset..."):
            if cuad_adapter.load_from_huggingface():
                # Check if we're using demo data
                if len(cuad_adapter.qa_pairs) > 0 and len(cuad_adapter.qa_pairs) <= 10:
                    st.info("📋 Using CUAD-style demo data for demonstration. Full CUAD dataset requires additional setup.")
                else:
                    st.success("CUAD dataset loaded successfully!")
                
                # Get sample for display
                df = cuad_adapter.get_dataframe(limit=100)  # Limit for performance
                document_text = cuad_adapter.create_document_from_contexts(limit=50)
                
                return cuad_adapter, df, document_text
            else:
                st.error("Failed to load CUAD dataset. Please check your internet connection.")
                return None, None, None
    except Exception as e:
        st.error(f"Error loading CUAD dataset: {str(e)}")
        return None, None, None


st.set_page_config(page_title="Legal Document QA", layout="wide")

st.title("Legal Document QA Assistant")

# Initialize session state
if 'cuad_adapter' not in st.session_state:
    st.session_state.cuad_adapter = None
if 'qa_dataset' not in st.session_state:
    st.session_state.qa_dataset = None
if 'document_text' not in st.session_state:
    st.session_state.document_text = ""

# Choose input mode
input_mode = st.radio("Choose input mode:", ["Upload Document", "Upload QA Dataset (CSV)", "Load CUAD Dataset"])

# Initialize variables
uploaded_file = None
uploaded_csv = None
qa_dataset = st.session_state.qa_dataset
cuad_adapter = st.session_state.cuad_adapter
document_text = st.session_state.document_text
question = ""
assistant = None
result = None

if input_mode == "Upload Document":
    uploaded_file = st.file_uploader("Document", type=["txt", "md", "pdf", "docx"])
    question = st.text_input("Question", placeholder="Can either party assign the agreement?")
    
elif input_mode == "Upload QA Dataset (CSV)":
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
            
elif input_mode == "Load CUAD Dataset":
    # Load button - only show if not loaded yet
    if st.session_state.cuad_adapter is None:
        if st.button("Load CUAD Dataset from Hugging Face"):
            cuad_adapter, qa_dataset, document_text = load_cuad_dataset()
            if cuad_adapter is not None:
                st.session_state.cuad_adapter = cuad_adapter
                st.session_state.qa_dataset = qa_dataset
                st.session_state.document_text = document_text
                st.rerun()  # Rerun to update the UI
    
    # Show interface if data is loaded
    if st.session_state.cuad_adapter is not None and st.session_state.qa_dataset is not None:
        cuad_adapter = st.session_state.cuad_adapter
        qa_dataset = st.session_state.qa_dataset
        document_text = st.session_state.document_text
        
        st.success(f"Loaded {len(qa_dataset)} QA pairs from CUAD dataset")
        
        # Show categories filter
        categories = cuad_adapter.get_categories()
        selected_category = st.selectbox("Filter by category:", ["All Categories"] + categories)
        
        # Filter questions by category
        if selected_category == "All Categories":
            filtered_df = qa_dataset
        else:
            filtered_df = qa_dataset[qa_dataset['category'] == selected_category]
        
        # Show available questions
        st.subheader("Available Questions:")
        question_options = filtered_df['question'].tolist()
        if question_options:
            selected_question_idx = st.selectbox("Select a question:", range(len(question_options)), format_func=lambda i: f"{i+1}. {question_options[i][:100]}...")
            question = question_options[selected_question_idx]
            
            # Get the specific QA pair for comparison
            selected_row = filtered_df.iloc[selected_question_idx]
            
            # Show expected answer for comparison
            with st.expander("Expected Answer"):
                st.write("**Category:**", selected_row['category'])
                st.write("**Answer:**", selected_row['answer'])
                st.write("**Evidence:**", selected_row['evidence'])
                st.write("**Document:**", selected_row['title'][:50] + "..." if len(selected_row['title']) > 50 else selected_row['title'])
                
                # Show original CUAD question for reference
                if 'original_question' in selected_row:
                    with st.expander("Original CUAD Question"):
                        st.write(selected_row['original_question'])
        
        # Add reset button
        if st.button("Reset CUAD Dataset"):
            st.session_state.cuad_adapter = None
            st.session_state.qa_dataset = None
            st.session_state.document_text = ""
            st.rerun()

with st.sidebar:
    st.header("Pipeline")
    
    # Retriever options with clearer labels
    retriever_options = {
        "BM25 keyword search": "bm25",
        "Dense semantic search": "dense"
    }
    retriever_label = st.selectbox("Retriever", list(retriever_options.keys()), index=1)  # Default to Dense
    retriever = retriever_options[retriever_label]
    
    # Reader options with clearer labels
    reader_options = {
        "Lexical baseline": "lexical",
        "Transformer QA": "transformers"
    }
    reader_label = st.selectbox("Reader", list(reader_options.keys()), index=1)  # Default to Transformer
    reader = reader_options[reader_label]
    
    # Dynamic UI based on selections
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=5)
    
    # Show embedding model only for dense retriever
    if retriever == "dense":
        embedding_model = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
    else:
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Keep default for consistency
        st.info("Embedding model is not used for BM25 retrieval.")
    
    # Show QA model only for transformer reader
    if reader == "transformers":
        qa_model = st.text_input("QA model", value="deepset/roberta-base-squad2")
    else:
        qa_model = "deepset/roberta-base-squad2"  # Keep default for consistency
        st.info("QA model is not used for lexical reader.")
    
    # Comparison mode
    compare_mode = st.checkbox("Compare with baseline", help="Run both baseline and transformer models side by side")

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
            # Handle CSV or CUAD dataset
            text = document_text
            if input_mode == "Upload QA Dataset (CSV)":
                source_name = uploaded_csv.name if uploaded_csv else "QA Dataset"
            else:
                source_name = "CUAD Dataset"

        if compare_mode and reader == "transformers":
            # Run both baseline and transformer for comparison
            st.subheader("🔄 Comparison Mode")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Baseline (BM25 + Lexical)**")
                baseline_assistant = LegalQAAssistant.from_text(
                    text,
                    source=source_name,
                    retriever="bm25",
                    reader="lexical",
                    qa_model=qa_model,
                    embedding_model=embedding_model,
                )
                baseline_result = baseline_assistant.ask(question, top_k=top_k)
                
                # Display baseline results
                st.metric("Confidence", f"{baseline_result.confidence:.2f}")
                st.metric("Retrieval Score", f"{baseline_result.retrieval_score:.3f}")
                st.write("**Answer:**", baseline_result.answer)
                st.write("**Evidence:**", baseline_result.evidence or "No supporting evidence found.")
                st.caption(f"Model: {baseline_result.model}")
            
            with col2:
                st.write("**Advanced (Dense + Transformer)**")
                advanced_assistant = LegalQAAssistant.from_text(
                    text,
                    source=source_name,
                    retriever="dense",
                    reader="transformers",
                    qa_model=qa_model,
                    embedding_model=embedding_model,
                )
                advanced_result = advanced_assistant.ask(question, top_k=top_k)
                
                # Display advanced results
                st.metric("Confidence", f"{advanced_result.confidence:.2f}")
                st.metric("Retrieval Score", f"{advanced_result.retrieval_score:.3f}")
                st.write("**Answer:**", advanced_result.answer)
                st.write("**Evidence:**", advanced_result.evidence or "No supporting evidence found.")
                st.caption(f"Model: {advanced_result.model}")
            
            # Use advanced result as main result
            result = advanced_result
        else:
            # Single mode
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

    # Enhanced result display (only show if not in comparison mode)
if result is not None and not (compare_mode and reader == "transformers"):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Confidence", f"{result.confidence:.2f}")
    col2.metric("Retrieval Score", f"{result.retrieval_score:.3f}")
    col3.metric("Chunks Used", top_k)
    col4.metric("Chunks Indexed", len(assistant.chunks))

    st.subheader("Answer")
    st.write(result.answer)

    st.subheader("Evidence")
    st.write(result.evidence or "No supporting evidence found.")

    # Method information
    method_info = f"{retriever_label} + {reader_label}"
    if "fallback" in result.model:
        method_info += " (with fallback)"
    st.info(f"**Method:** {method_info}")

    # Source information
    st.caption(f"Source: {result.source} | Chunk: {result.chunk_id} | Model: {result.model}")

# If using CSV or CUAD dataset, show comparison with expected answer
if result is not None and qa_dataset is not None and question:
    if input_mode == "Load CUAD Dataset":
        # For CUAD, we need to find the exact match
        expected_row = qa_dataset[qa_dataset['question'] == question]
        if not expected_row.empty:
            expected_answer = expected_row.iloc[0]['answer']
            expected_evidence = expected_row.iloc[0]['evidence']
            category = expected_row.iloc[0]['category']
            
            st.subheader("Expected Answer (from CUAD)")
            st.write(f"**Category:** {category}")
            st.write(f"**Answer:** {expected_answer}")
            st.write(f"**Evidence:** {expected_evidence}")
            
            # Simple comparison
            if expected_answer.lower().strip() in result.answer.lower().strip() or result.answer.lower().strip() in expected_answer.lower().strip():
                st.success("✓ Answer matches expected result!")
            else:
                st.warning("⚠ Answer differs from expected result")
    else:
        # For CSV dataset
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
    elif input_mode == "Upload QA Dataset (CSV)":
        st.info("Upload a CSV dataset with QA pairs to get started.")
    else:
        st.info("Click 'Load CUAD Dataset from Hugging Face' to load the CUAD dataset.")
