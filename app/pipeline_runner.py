"""
Shared pipeline runner for Streamlit pages.

Runs ingestion -> labeling -> ticker extraction -> analysis and caches results.
All dashboard pages import from here rather than running the pipeline themselves.

Usage in a page:
    from app.pipeline_runner import run_pipeline, load_model
    data = run_pipeline()
    model = load_model()
"""

import streamlit as st
import sys
import os

# Make sure the project root is on the path regardless of working directory.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.config import load_config
from src.ingestion.manager import IngestionManager
from src.labeling.aggregator import LabelAggregator
from src.extraction.ticker_extractor import TickerExtractor
from src.models.pipeline import SentimentPipeline


@st.cache_data(ttl=300)
def run_pipeline():
    """
    Run the full MarketPulse pipeline and return all results.

    Cached for 5 minutes (ttl=300). Re-runs automatically after expiry
    or when the user manually clears the cache via st.cache_data.clear().

    Returns:
        dict with keys:
            'df'             -- pandas DataFrame with ingested + labeled posts
            'ticker_results' -- dict mapping symbol -> ticker analysis dict
            'market_summary' -- high-level market sentiment summary dict
            'source_summary' -- ingestion source stats dict
            'config'         -- loaded config dict
    """
    config = load_config()

    # --- Ingestion ---
    mgr = IngestionManager(config)
    df = mgr.ingest()
    source_summary = mgr.get_source_summary()

    # --- Labeling ---
    agg = LabelAggregator(config=config)
    df = agg.aggregate_batch(df)

    # --- Ticker extraction ---
    te = TickerExtractor()
    df['tickers'] = df['text'].apply(te.extract)

    # --- Ticker-level sentiment aggregation ---
    # Import here so pages that don't need analysis don't pay the cost
    try:
        from src.analysis.ticker_sentiment import TickerSentimentAnalyzer
        analyzer = TickerSentimentAnalyzer()
        ticker_results = analyzer.analyze(df)
        market_summary = analyzer.get_market_summary(ticker_results)
    except (ImportError, AttributeError):
        # src.analysis module may not exist yet; degrade gracefully
        ticker_results = {}
        market_summary = {}

    return {
        'df': df,
        'ticker_results': ticker_results,
        'market_summary': market_summary,
        'source_summary': source_summary,
        'config': config,
    }


@st.cache_resource
def load_model():
    """
    Load the trained sentiment model from disk.

    Cached as a resource (shared across all sessions, not serialized).
    Returns None if no trained model exists — pages must handle this case.

    Returns:
        SentimentPipeline instance with is_trained=True, or None
    """
    model_dir = os.path.join(_project_root, "data", "models")
    model_path = os.path.join(model_dir, "sentiment_model.pkl")

    if not os.path.exists(model_path):
        return None

    pipeline = SentimentPipeline()
    try:
        pipeline.load(model_dir)
        return pipeline
    except Exception:
        return None
