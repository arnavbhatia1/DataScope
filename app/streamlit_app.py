"""
MarketPulse Dashboard -- Main Entry Point

Run with:
    streamlit run app/streamlit_app.py

This is the home page of the multi-page Streamlit app.
Individual pages live in app/pages/ and are auto-discovered by Streamlit.
"""

import streamlit as st
import sys
import os

# Project root must be on sys.path before any local imports.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.components.styles import apply_theme

st.set_page_config(
    page_title="MarketPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📊 MarketPulse")
st.sidebar.markdown("**Sentiment Intelligence for Financial Markets**")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Navigate to a page using the links above, "
    "or follow the quick-start below."
)

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
st.title("MarketPulse")
st.markdown("### Financial Sentiment Intelligence Dashboard")
st.markdown(
    "MarketPulse is a **data-centric ML pipeline** that ingests financial social "
    "media, programmatically labels sentiment, trains a classifier, and surfaces "
    "everything through this interactive dashboard."
)

st.info(
    "Select a page from the **sidebar** to get started. "
    "Start with **Data Ingestion** to load posts, then work through "
    "Labeling Studio, Label Quality, and Model Training."
)

# ---------------------------------------------------------------------------
# Pipeline thesis callout
# ---------------------------------------------------------------------------
st.markdown("---")
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("#### Core Thesis")
    st.markdown(
        "> *A logistic regression trained on high-quality, programmatically "
        "labeled data is a production-ready system. **The data is the model.***"
    )
    st.markdown(
        "The **Label Quality** page runs a controlled experiment comparing "
        "model F1 across four label sources:"
    )
    st.markdown(
        "- **Gold** — hand-labeled ground truth  \n"
        "- **Programmatic** — our labeling pipeline output  \n"
        "- **Noisy** — gold labels with 30% random noise  \n"
        "- **Random** — completely random labels  \n\n"
        "Result: the same model with better labels dramatically outperforms."
    )

with col_right:
    st.markdown("#### Quick Start")
    st.markdown(
        "1. **Data Ingestion** — ingest posts  \n"
        "2. **Labeling Studio** — apply labeling functions  \n"
        "3. **Label Quality** — evaluate label quality  \n"
        "4. **Model Training** — train the classifier  \n"
        "5. **Entity Extraction** — extract ticker mentions  \n"
        "6. **Evaluation** — full error analysis  \n"
        "7. **Live Inference** — classify new posts  \n"
    )

# ---------------------------------------------------------------------------
# Status overview (degrade gracefully if pipeline hasn't run yet)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("#### Pipeline Status")

try:
    from app.pipeline_runner import run_pipeline, load_model

    with st.spinner("Loading pipeline data..."):
        data = run_pipeline()

    df = data['df']
    source_summary = data['source_summary']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Posts Ingested", f"{len(df):,}")
    col2.metric(
        "Labeled",
        f"{df['programmatic_label'].notna().sum():,}"
        if 'programmatic_label' in df.columns else "—",
    )
    col3.metric(
        "Sources",
        ', '.join(source_summary.get('sources_used', [])) or '—',
    )

    model = load_model()
    if model and getattr(model, 'is_trained', False):
        f1 = model.metadata.get('weighted_f1', None)
        col4.metric("Model F1", f"{f1:.3f}" if f1 is not None else "trained")
    else:
        col4.metric("Model F1", "not trained")

except Exception as e:
    st.warning(
        "Pipeline has not run yet. "
        "Go to **Data Ingestion** to load your first batch of posts, "
        "or run `make pipeline` from the terminal."
    )
    st.caption(f"(Technical detail: {e})")
