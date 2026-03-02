"""
MarketPulse — Live Inference

Paste any financial social media post and get instant sentiment classification,
confidence scores, and extracted ticker entities.
"""

import streamlit as st
import pandas as pd
import io
import sys
import os

# Ensure project root is on the path so imports work when run via streamlit.
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.pipeline_runner import load_model
from app.components.charts import probability_bar
from app.components.styles import COLORS, SENTIMENT_COLORS, apply_theme
from src.extraction.ticker_extractor import TickerExtractor

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Live Inference | MarketPulse",
    page_icon="🎯",
    layout="wide",
)
apply_theme()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Live Inference")
st.markdown("Classify any financial social media post in real time — see sentiment, confidence, and extracted tickers instantly.")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
model = load_model()

if model is None:
    st.warning(
        "No trained model found. Run the pipeline first to train a model.\n\n"
        "From the project root: `make train` or `python scripts/train.py`"
    )
    st.stop()

# Shared extractor instance
_extractor = TickerExtractor()

# ---------------------------------------------------------------------------
# Helper: render sentiment badge
# ---------------------------------------------------------------------------
SENTIMENT_LABELS = {
    "bullish": "BULLISH",
    "bearish": "BEARISH",
    "neutral": "NEUTRAL",
    "meme": "MEME",
}


def _sentiment_badge(label: str) -> str:
    """Return an HTML badge for the given sentiment label."""
    color = SENTIMENT_COLORS.get(label, COLORS["secondary"])
    display = SENTIMENT_LABELS.get(label, label.upper())
    return (
        f'<span style="background:{color}22; border:2px solid {color}; '
        f'border-radius:8px; padding:6px 18px; font-size:1.25rem; '
        f'font-weight:bold; color:{color}; letter-spacing:0.05em;">'
        f'{display}</span>'
    )


def _ticker_pills(tickers: list) -> str:
    """Return HTML ticker pills for a list of company names."""
    if not tickers:
        return "*No tickers detected*"
    pills = " ".join(
        f'<span style="background:#21262D; border:1px solid #30363D; '
        f'border-radius:16px; padding:4px 12px; margin:2px; '
        f'display:inline-block; font-size:0.85rem;">{t}</span>'
        for t in tickers
    )
    return f"**Tickers mentioned:** {pills}"


# ---------------------------------------------------------------------------
# Section 1 — Single post inference
# ---------------------------------------------------------------------------
st.header("Single Post")

DEFAULT_POST = (
    "Just loaded up on $NVDA calls before earnings. "
    "AI buildout is nowhere near done — this thing is going to $1000. 🚀"
)

text_input = st.text_area(
    "Paste a financial social media post:",
    value=DEFAULT_POST,
    height=120,
    placeholder="e.g. Loaded 500 shares of TSLA, this dip won't last long...",
)

analyze_btn = st.button("Analyze", type="primary", use_container_width=False)

if analyze_btn and text_input.strip():
    with st.spinner("Classifying..."):
        result = model.predict_single(text_input)
        tickers = _extractor.extract(text_input)

    label = result["label"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]

    # ---- Top summary row ----
    col_badge, col_conf, col_spacer = st.columns([2, 2, 4])

    with col_badge:
        st.markdown("**Predicted Sentiment**")
        st.markdown(_sentiment_badge(label), unsafe_allow_html=True)

    with col_conf:
        st.markdown("**Confidence**")
        color = SENTIMENT_COLORS.get(label, COLORS["secondary"])
        st.markdown(
            f'<p style="font-size:2rem; font-weight:bold; color:{color}; '
            f'margin:0; padding:6px 0;">{confidence:.1%}</p>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ---- Probability chart ----
    col_chart, col_tickers = st.columns([3, 2])

    with col_chart:
        fig = probability_bar(probabilities)
        st.plotly_chart(fig, use_container_width=True)

    with col_tickers:
        st.markdown("**Extracted Entities**")
        ticker_html = _ticker_pills(tickers)
        st.markdown(ticker_html, unsafe_allow_html=True)

        if tickers:
            st.markdown("")
            st.caption(f"{len(tickers)} entit{'y' if len(tickers) == 1 else 'ies'} detected")

elif analyze_btn and not text_input.strip():
    st.warning("Please paste some text before clicking Analyze.")

# ---------------------------------------------------------------------------
# Section 2 — Recent predictions log (session state)
# ---------------------------------------------------------------------------
if "inference_log" not in st.session_state:
    st.session_state["inference_log"] = []

if analyze_btn and text_input.strip():
    # Append to log (keep last 20)
    st.session_state["inference_log"].insert(0, {
        "text": text_input[:120] + ("..." if len(text_input) > 120 else ""),
        "sentiment": result["label"],
        "confidence": f"{result['confidence']:.1%}",
        "tickers": ", ".join(tickers) if tickers else "—",
    })
    st.session_state["inference_log"] = st.session_state["inference_log"][:20]

if st.session_state["inference_log"]:
    with st.expander("Recent predictions (this session)", expanded=False):
        log_df = pd.DataFrame(st.session_state["inference_log"])
        st.dataframe(log_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Section 3 — Batch inference
# ---------------------------------------------------------------------------
st.divider()
st.header("Batch Inference")
st.markdown("Upload a CSV with a `text` column to classify many posts at once.")

uploaded = st.file_uploader(
    "Upload CSV with 'text' column",
    type=["csv"],
    help="The CSV must contain a column named 'text'. All other columns are preserved.",
)

if uploaded is not None:
    try:
        batch_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()

    if "text" not in batch_df.columns:
        st.error("The uploaded CSV must have a column named **text**.")
    else:
        st.info(f"Loaded {len(batch_df):,} rows. Running inference...")

        with st.spinner(f"Classifying {len(batch_df):,} posts..."):
            texts = batch_df["text"].fillna("").tolist()
            predictions = model.predict(texts)
            batch_df["sentiment"] = [p["label"] for p in predictions]
            batch_df["confidence"] = [round(p["confidence"], 4) for p in predictions]
            batch_df["tickers"] = [
                ", ".join(_extractor.extract(t)) for t in texts
            ]

        # Summary metrics
        total = len(batch_df)
        sentiment_counts = batch_df["sentiment"].value_counts()

        met_cols = st.columns(4)
        for i, label in enumerate(["bullish", "bearish", "neutral", "meme"]):
            count = int(sentiment_counts.get(label, 0))
            pct = count / total if total > 0 else 0.0
            met_cols[i].metric(label.upper(), f"{count}", f"{pct:.1%} of posts")

        st.dataframe(batch_df, use_container_width=True, height=350)

        # Download button
        csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results CSV",
            data=csv_bytes,
            file_name="marketpulse_batch_predictions.csv",
            mime="text/csv",
            type="primary",
        )
