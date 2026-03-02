"""
MarketPulse — Market Overview

The main product page. Shows the full market-wide sentiment picture:
top KPI metrics, per-ticker sentiment cards, and a ranked mentions chart.
"""

import streamlit as st
import sys
import os

# Ensure project root is importable regardless of launch directory.
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.pipeline_runner import run_pipeline
from app.components.styles import apply_theme, COLORS, SENTIMENT_COLORS
from app.components.charts import ticker_mentions_bar
from app.components.metrics import source_status_indicator

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Market Overview | MarketPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# ---------------------------------------------------------------------------
# Sidebar: refresh control + source status
# ---------------------------------------------------------------------------
st.sidebar.title("📊 MarketPulse")
st.sidebar.markdown("**Market Overview**")
st.sidebar.markdown("---")

if st.sidebar.button("Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Load pipeline data
# ---------------------------------------------------------------------------
with st.spinner("Loading pipeline data..."):
    try:
        data = run_pipeline()
    except Exception as e:
        st.error(
            "Pipeline failed to run. Make sure dependencies are installed "
            "and at least synthetic data is available."
        )
        st.caption(f"Technical detail: {e}")
        st.stop()

df = data["df"]
ticker_results = data["ticker_results"]
market_summary = data["market_summary"]
source_summary = data["source_summary"]

# Sidebar source status
source_status_indicator(
    sources_used=source_summary.get("sources_used", []),
    sources_unavailable=source_summary.get("sources_unavailable", []),
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Market Overview")
st.markdown("Real-time sentiment intelligence across financial social media.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Top KPI metrics row
# ---------------------------------------------------------------------------
total_posts = len(df)
tickers_tracked = len(ticker_results)

# Labeling coverage: fraction of posts that received a programmatic label
if "programmatic_label" in df.columns:
    labeled_count = df["programmatic_label"].notna().sum()
    coverage_pct = labeled_count / total_posts if total_posts > 0 else 0.0
else:
    coverage_pct = 0.0

sources_used = source_summary.get("sources_used", [])
data_source_display = ", ".join(sources_used).upper() if sources_used else "NONE"

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Posts Analyzed", f"{total_posts:,}")
col_m2.metric("Tickers Tracked", f"{tickers_tracked:,}")
col_m3.metric("Labeling Coverage", f"{coverage_pct:.1%}")
col_m4.metric("Data Source", data_source_display)

st.markdown("---")

# ---------------------------------------------------------------------------
# Ticker card grid
# ---------------------------------------------------------------------------
if not ticker_results:
    st.info(
        "No ticker data yet. The pipeline may still be warming up, "
        "or no ticker mentions were found in the current dataset."
    )
else:
    st.markdown("### Ticker Sentiment")
    st.caption(
        "Each card shows the dominant sentiment direction for that ticker based "
        "on programmatic labeling of all posts mentioning it."
    )

    cols = st.columns(3)
    for i, (company, ticker_data) in enumerate(ticker_results.items()):
        col = cols[i % 3]
        with col:
            sentiment = ticker_data.get("dominant_sentiment", "neutral")
            color = SENTIMENT_COLORS.get(sentiment, COLORS["secondary"])
            symbol = ticker_data.get("symbol", company.upper())
            mention_count = ticker_data.get("mention_count", 0)
            avg_conf = ticker_data.get("avg_confidence", 0.0)

            st.markdown(
                f"""
                <div class="ticker-card">
                    <div style="font-size:1.3em; font-weight:bold;">{symbol}</div>
                    <div style="color: #8B949E;">{company}</div>
                    <div class="sentiment-{sentiment}" style="font-size:1.1em; margin:8px 0;">
                        {sentiment.upper()}
                    </div>
                    <div style="color: #8B949E; font-size:0.9em;">
                        {mention_count} mentions &middot; {avg_conf:.0%} confidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(f"View {symbol}", key=f"btn_{company}"):
                st.session_state["selected_ticker"] = company
                st.switch_page("pages/2_Ticker_Detail.py")

    # ---------------------------------------------------------------------------
    # Ticker mentions bar chart
    # ---------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Most Mentioned Tickers")
    st.caption("Bar length shows total mentions. Color reflects dominant sentiment.")

    fig = ticker_mentions_bar(ticker_results, top_n=15)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Market-level summary (if available from analyzer)
# ---------------------------------------------------------------------------
if market_summary:
    st.markdown("---")
    st.markdown("### Market Sentiment Summary")

    overall = market_summary.get("overall_sentiment", "")
    bullish_pct = market_summary.get("bullish_pct", 0.0)
    bearish_pct = market_summary.get("bearish_pct", 0.0)

    ms_col1, ms_col2, ms_col3 = st.columns(3)
    ms_col1.metric("Overall Market Bias", overall.upper() if overall else "MIXED")
    ms_col2.metric("Bullish Tickers", f"{bullish_pct:.0%}")
    ms_col3.metric("Bearish Tickers", f"{bearish_pct:.0%}")

# ---------------------------------------------------------------------------
# Footer note
# ---------------------------------------------------------------------------
st.markdown("---")
used_fallback = source_summary.get("used_fallback", False)
if used_fallback:
    st.info(
        "No live API keys detected. Data shown is from the **synthetic dataset**. "
        "Add credentials to `.env` to enable live Reddit, Stocktwits, or News ingestion."
    )
else:
    mode = source_summary.get("mode", "auto").upper()
    st.caption(f"Data mode: {mode} | Sources: {data_source_display}")
