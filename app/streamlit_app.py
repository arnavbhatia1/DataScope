"""
MarketPulse Dashboard -- Main Entry Point

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.components.styles import apply_theme, SENTIMENT_COLORS

st.set_page_config(
    page_title="MarketPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# -- Sidebar --
st.sidebar.title("📊 MarketPulse")
st.sidebar.markdown("**Sentiment Intelligence for Financial Markets**")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pages**\n\n"
    "- **Market Overview** — ticker sentiment at a glance\n"
    "- **Ticker Detail** — drill into any ticker\n"
    "- **Live Inference** — classify new posts\n"
    "- **Under the Hood** — model & labeling diagnostics"
)

# -- Hero --
st.title("MarketPulse")
st.markdown(
    "Real-time sentiment intelligence for financial markets. "
    "See which tickers are **bullish**, **bearish**, or drowning in **memes** "
    "— backed by ML analysis of social media posts."
)

# -- Load pipeline data --
try:
    from app.pipeline_runner import run_pipeline, load_model

    with st.spinner("Analyzing market sentiment..."):
        data = run_pipeline()

    df = data['df']
    ticker_results = data['ticker_results']
    market_summary = data['market_summary']
    source_summary = data['source_summary']

    # -- Top metrics --
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts Analyzed", f"{len(df):,}")
    c2.metric("Tickers Tracked", f"{market_summary['total_tickers']}")

    labeled_count = df['programmatic_label'].notna().sum() if 'programmatic_label' in df.columns else 0
    c3.metric("Coverage", f"{labeled_count / len(df):.0%}" if len(df) > 0 else "—")

    sources = ', '.join(source_summary.get('sources_used', []))
    c4.metric("Data Source", sources or "—")

    # -- Market snapshot --
    st.markdown("---")
    st.markdown("### Market Snapshot")

    dist = market_summary.get('ticker_sentiment_distribution', {})
    scols = st.columns(4)
    for i, sentiment in enumerate(['bullish', 'bearish', 'neutral', 'meme']):
        count = dist.get(sentiment, 0)
        color = SENTIMENT_COLORS.get(sentiment, '#78909C')
        scols[i].markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-size:2em; font-weight:bold; color:{color};'>{count}</div>"
            f"<div style='color:#8B949E;'>{sentiment.upper()} tickers</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # -- Top movers --
    st.markdown("---")
    col_bull, col_bear = st.columns(2)

    with col_bull:
        st.markdown("#### Top Bullish")
        bullish = [t for t in ticker_results.values() if t['dominant_sentiment'] == 'bullish'][:5]
        for t in bullish:
            st.markdown(
                f"**{t['symbol']}** ({t['company']}) — "
                f"{t['mention_count']} mentions, {t['avg_confidence']:.0%} confidence"
            )
        if not bullish:
            st.caption("No bullish tickers detected.")

    with col_bear:
        st.markdown("#### Top Bearish")
        bearish = [t for t in ticker_results.values() if t['dominant_sentiment'] == 'bearish'][:5]
        for t in bearish:
            st.markdown(
                f"**{t['symbol']}** ({t['company']}) — "
                f"{t['mention_count']} mentions, {t['avg_confidence']:.0%} confidence"
            )
        if not bearish:
            st.caption("No bearish tickers detected.")

    # -- CTA --
    st.markdown("---")
    st.markdown(
        "Go to **Market Overview** for the full ticker grid, "
        "or **Live Inference** to classify your own posts."
    )

except Exception as e:
    st.warning(
        "Run the pipeline first to load data:\n\n"
        "```\npython3 scripts/run_pipeline.py\n```\n\n"
        "Then refresh this page."
    )
    st.caption(f"({e})")
