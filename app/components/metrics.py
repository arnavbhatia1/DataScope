"""Metric display components for MarketPulse dashboard."""

import streamlit as st
from .styles import COLORS, SENTIMENT_COLORS


def source_status_indicator(sources_used, sources_unavailable):
    """
    Show data source connection status in sidebar.

    Args:
        sources_used: list of source keys that are active, e.g. ['reddit', 'synthetic']
        sources_unavailable: list of source keys with missing API keys
    """
    st.sidebar.markdown("### Data Sources")
    all_sources = {
        'reddit': 'Reddit',
        'stocktwits': 'Stocktwits',
        'news': 'News (RSS)',
    }
    for key, name in all_sources.items():
        if key in sources_used:
            st.sidebar.markdown(f"🟢 **{name}** — connected")
        elif key in sources_unavailable:
            st.sidebar.markdown(f"🔴 **{name}** — no API key")
        else:
            st.sidebar.markdown(f"⚫ **{name}** — not configured")


def pipeline_status_card(source_summary):
    """
    Display a compact pipeline status row with key ingestion stats.

    Args:
        source_summary: dict returned by IngestionManager.get_source_summary()
    """
    total = source_summary.get('total_posts', 0)
    sources = ', '.join(source_summary.get('sources_used', [])) or 'none'
    mode = source_summary.get('mode', 'auto')
    used_fallback = source_summary.get('used_fallback', False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Posts", f"{total:,}")
    col2.metric("Active Sources", sources)
    col3.metric("Mode", mode.upper() + (" (fallback)" if used_fallback else ""))


def sentiment_metrics_row(sentiment_counts):
    """
    Display one st.metric per sentiment class in a single row.

    Args:
        sentiment_counts: dict mapping sentiment label -> count
    """
    labels = ['bullish', 'bearish', 'neutral', 'meme']
    cols = st.columns(len(labels))
    total = sum(sentiment_counts.values()) or 1

    for col, label in zip(cols, labels):
        count = sentiment_counts.get(label, 0)
        pct = count / total * 100
        col.metric(
            label=label.upper(),
            value=f"{count:,}",
            delta=f"{pct:.1f}%",
        )


def model_status_card(model_pipeline):
    """
    Show model training status; warns if model not yet trained.

    Args:
        model_pipeline: SentimentPipeline instance or None

    Returns:
        True if model is ready, False otherwise
    """
    if model_pipeline is None or not getattr(model_pipeline, 'is_trained', False):
        st.warning(
            "No trained model found. "
            "Go to **Model Training** and click Train, "
            "or run `make train` from the terminal."
        )
        return False

    meta = getattr(model_pipeline, 'metadata', {})
    f1 = meta.get('weighted_f1', None)
    trained_on = meta.get('training_date', 'unknown date')
    label_source = meta.get('label_source', 'unknown')

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Status", "Ready")
    col2.metric("Weighted F1", f"{f1:.3f}" if f1 is not None else "N/A")
    col3.metric("Trained On", f"{label_source} labels")
    st.caption(f"Last trained: {trained_on}")
    return True


