# MarketPulse Product Pivot — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform MarketPulse from a thesis experiment into a ticker-centric sentiment intelligence product with a Streamlit dashboard, real API ingestion, and per-ticker sentiment breakdowns with evidence posts.

**Architecture:** The existing ML pipeline (ingestion → labeling → model → extraction) becomes the engine. A new analysis module aggregates per-ticker sentiment. A 4-page Streamlit dashboard consumes the analysis output. Reddit/Stocktwits APIs are wired up with synthetic fallback.

**Tech Stack:** Streamlit, Plotly (plotly_dark), PRAW (Reddit), requests (Stocktwits), existing sklearn pipeline

---

### Task 1: Ticker Sentiment Analysis Module

**Files:**
- Create: `src/analysis/__init__.py`
- Create: `src/analysis/ticker_sentiment.py`

**Step 1: Create the analysis package**

Create `src/analysis/__init__.py` (empty file).

**Step 2: Write the ticker sentiment aggregator**

Create `src/analysis/ticker_sentiment.py`:

```python
"""
Per-Ticker Sentiment Aggregation

Takes labeled + extracted DataFrame and produces per-ticker
sentiment summaries for the dashboard.
"""

import pandas as pd
from collections import Counter
from src.extraction.ticker_extractor import TickerExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)

SENTIMENT_COLORS = {
    'bullish': '#00C853',
    'bearish': '#FF1744',
    'neutral': '#78909C',
    'meme': '#FFD600',
}


class TickerSentimentAnalyzer:

    def __init__(self):
        self.extractor = TickerExtractor()
        self._ticker_to_symbol = {}
        for symbol, company in self.extractor.ticker_map.items():
            if company not in self._ticker_to_symbol:
                self._ticker_to_symbol[company] = symbol

    def analyze(self, df):
        """
        Analyze per-ticker sentiment from labeled DataFrame.

        Args:
            df: DataFrame with 'text', 'programmatic_label',
                'label_confidence' columns. 'tickers' column optional
                (will extract if missing).

        Returns:
            dict mapping company name to ticker summary dict.
        """
        df = df.copy()

        # Extract tickers if not already present
        if 'tickers' not in df.columns:
            df['tickers'] = df['text'].apply(self.extractor.extract)

        # Only use labeled posts
        labeled = df[df['programmatic_label'].notna()].copy()

        ticker_data = {}

        for _, row in labeled.iterrows():
            tickers = row['tickers']
            if not tickers or not isinstance(tickers, list):
                continue

            for company in tickers:
                if company not in ticker_data:
                    ticker_data[company] = {
                        'company': company,
                        'symbol': self._ticker_to_symbol.get(company, ''),
                        'posts': [],
                        'sentiments': [],
                        'confidences': [],
                    }

                ticker_data[company]['posts'].append({
                    'post_id': row.get('post_id', ''),
                    'text': row['text'],
                    'sentiment': row['programmatic_label'],
                    'confidence': float(row.get('label_confidence', 0)),
                    'source': row.get('source', 'unknown'),
                    'timestamp': str(row.get('timestamp', '')),
                    'author': row.get('author', 'unknown'),
                })
                ticker_data[company]['sentiments'].append(row['programmatic_label'])
                ticker_data[company]['confidences'].append(
                    float(row.get('label_confidence', 0))
                )

        # Build summaries
        results = {}
        for company, data in ticker_data.items():
            sentiment_counts = Counter(data['sentiments'])
            total = len(data['sentiments'])
            dominant = sentiment_counts.most_common(1)[0][0] if total > 0 else 'neutral'

            results[company] = {
                'company': company,
                'symbol': data['symbol'],
                'mention_count': total,
                'sentiment': dict(sentiment_counts),
                'dominant_sentiment': dominant,
                'dominant_color': SENTIMENT_COLORS.get(dominant, '#78909C'),
                'bullish_ratio': sentiment_counts.get('bullish', 0) / total if total > 0 else 0,
                'bearish_ratio': sentiment_counts.get('bearish', 0) / total if total > 0 else 0,
                'avg_confidence': sum(data['confidences']) / len(data['confidences']) if data['confidences'] else 0,
                'posts': sorted(data['posts'], key=lambda p: p['confidence'], reverse=True),
            }

        # Sort by mention count
        results = dict(sorted(results.items(), key=lambda x: x[1]['mention_count'], reverse=True))

        logger.info(f"Ticker analysis: {len(results)} tickers found across {len(labeled)} labeled posts")
        return results

    def get_market_summary(self, ticker_results):
        """
        Aggregate market-level summary from ticker results.

        Returns dict with overall sentiment distribution across tickers.
        """
        sentiment_counts = Counter()
        total_mentions = 0
        total_tickers = len(ticker_results)

        for company, data in ticker_results.items():
            sentiment_counts[data['dominant_sentiment']] += 1
            total_mentions += data['mention_count']

        return {
            'total_tickers': total_tickers,
            'total_mentions': total_mentions,
            'ticker_sentiment_distribution': dict(sentiment_counts),
            'top_bullish': [
                t for t in ticker_results.values()
                if t['dominant_sentiment'] == 'bullish'
            ][:5],
            'top_bearish': [
                t for t in ticker_results.values()
                if t['dominant_sentiment'] == 'bearish'
            ][:5],
        }
```

**Step 3: Verify it works**

Run: `python3 -c "
import sys; sys.path.insert(0, '.')
from src.analysis.ticker_sentiment import TickerSentimentAnalyzer
import pandas as pd
df = pd.read_csv('data/labeled/labeled_data.csv')
analyzer = TickerSentimentAnalyzer()
results = analyzer.analyze(df)
print(f'Tickers found: {len(results)}')
for name, data in list(results.items())[:3]:
    print(f'  {name}: {data[\"mention_count\"]} mentions, {data[\"dominant_sentiment\"]}')
"`

Expected: tickers found with mention counts and dominant sentiments.

**Step 4: Commit**

```bash
git add src/analysis/
git commit -m "feat: ticker sentiment analysis module"
```

---

### Task 2: Wire Extraction Into Pipeline

**Files:**
- Modify: `scripts/run_pipeline.py`

**Step 1: Update run_pipeline.py**

Replace the current pipeline script to:
1. Run extraction right after labeling (step 2.5)
2. Replace thesis experiment (step 5) with ticker analysis
3. Update the summary output to be product-focused

The key changes to `scripts/run_pipeline.py`:

After step 2 (labeling), add extraction:
```python
# -- Step 3: Extract Entities --
print("\n[3/7] EXTRACTING ENTITIES...")
te = TickerExtractor()
df['tickers'] = df['text'].apply(te.extract)
posts_with_tickers = sum(1 for t in df['tickers'] if t)
print(f"  -> Entities in {posts_with_tickers}/{len(df)} posts")
```

Replace thesis experiment step with ticker analysis:
```python
# -- Step 5: Ticker Sentiment Analysis --
print("\n[5/7] ANALYZING TICKER SENTIMENT...")
from src.analysis.ticker_sentiment import TickerSentimentAnalyzer
analyzer = TickerSentimentAnalyzer()
ticker_results = analyzer.analyze(df)
market = analyzer.get_market_summary(ticker_results)
print(f"  -> {market['total_tickers']} tickers, {market['total_mentions']} total mentions")
print(f"  -> Sentiment: {market['ticker_sentiment_distribution']}")
for company, data in list(ticker_results.items())[:5]:
    print(f"     {data['symbol']:>5} ({company}): {data['dominant_sentiment']} "
          f"({data['mention_count']} mentions, {data['avg_confidence']:.0%} conf)")
```

Update step numbering: Ingest → Label → Extract → Quality → Train → Analyze → Summary.

Update the summary to be product-focused (remove thesis language).

Save `df` with tickers column to labeled_data.csv.

**Step 2: Verify pipeline runs end-to-end**

Run: `python3 scripts/run_pipeline.py`

Expected: pipeline completes with ticker sentiment analysis showing top tickers.

**Step 3: Commit**

```bash
git add scripts/run_pipeline.py
git commit -m "feat: integrate extraction and ticker analysis into pipeline"
```

---

### Task 3: Streamlit App Shell + Theme

**Files:**
- Create: `app/streamlit_app.py`
- Create: `app/components/styles.py`
- Create: `app/components/charts.py`
- Create: `app/components/metrics.py`
- Create: `app/components/__init__.py`
- Create: `app/pages/__init__.py` (empty, for package)

**Step 1: Create the styles module**

Create `app/components/__init__.py` (empty).

Create `app/components/styles.py` with the dark financial theme CSS and color constants from CLAUDE.md:

```python
"""Dark financial terminal theme for MarketPulse dashboard."""

COLORS = {
    'bullish': '#00C853',
    'bearish': '#FF1744',
    'neutral': '#78909C',
    'meme': '#FFD600',
    'primary': '#58A6FF',
    'secondary': '#8B949E',
    'bg_primary': '#0D1117',
    'bg_secondary': '#161B22',
    'bg_tertiary': '#21262D',
    'text_primary': '#E6EDF3',
    'text_secondary': '#8B949E',
    'border': '#30363D',
}

SENTIMENT_COLORS = {
    'bullish': COLORS['bullish'],
    'bearish': COLORS['bearish'],
    'neutral': COLORS['neutral'],
    'meme': COLORS['meme'],
}

def apply_theme():
    """Inject custom CSS into Streamlit app."""
    import streamlit as st
    st.markdown("""
    <style>
    .stApp { background-color: #0D1117; }
    .ticker-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .ticker-card:hover {
        border-color: #58A6FF;
        cursor: pointer;
    }
    .sentiment-bullish { color: #00C853; }
    .sentiment-bearish { color: #FF1744; }
    .sentiment-neutral { color: #78909C; }
    .sentiment-meme { color: #FFD600; }
    .metric-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
```

**Step 2: Create the charts module**

Create `app/components/charts.py` with reusable Plotly chart functions:
- `sentiment_pie(sentiment_dict)` — pie chart of bullish/bearish/neutral/meme
- `sentiment_bar(sentiment_dict)` — horizontal bar chart
- `ticker_mentions_bar(ticker_results, top_n=15)` — bar chart of top mentioned tickers

All charts use `template='plotly_dark'` and the COLORS palette.

**Step 3: Create the metrics module**

Create `app/components/metrics.py` with:
- `metric_card(label, value, color=None)` — renders a styled metric card via st.markdown
- `source_status_indicator(sources_used, sources_unavailable)` — green/red dot indicators

**Step 4: Create main app entry point**

Create `app/streamlit_app.py`:

```python
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.components.styles import apply_theme

st.set_page_config(
    page_title="MarketPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()

st.sidebar.title("📊 MarketPulse")
st.sidebar.markdown("**Sentiment Intelligence for Financial Markets**")

st.title("MarketPulse")
st.markdown("### Loading... Run the pipeline first, then navigate to pages.")
```

Create `app/pages/__init__.py` (empty).

**Step 5: Verify Streamlit launches**

Run: `streamlit run app/streamlit_app.py --server.headless true &` then kill it.

Expected: Streamlit starts on port 8501 without errors.

**Step 6: Commit**

```bash
git add app/
git commit -m "feat: Streamlit app shell with dark financial theme"
```

---

### Task 4: Market Overview Page

**Files:**
- Create: `app/pages/1_Market_Overview.py`

**Step 1: Build the overview page**

This is the main product page. It:
1. Loads the pipeline data (run ingestion → labeling → extraction → analysis on app startup, cache with `@st.cache_data`)
2. Shows a top metrics bar (total posts, tickers tracked, data source, coverage)
3. Renders a grid of ticker cards — each showing: symbol, company name, dominant sentiment color, mention count, avg confidence
4. Shows a market-level sentiment summary bar chart
5. Each ticker card is clickable (uses `st.session_state` to navigate to detail page)

Key implementation details:
- Use `@st.cache_data(ttl=300)` for pipeline execution (cache 5 min)
- The pipeline function runs: IngestionManager.ingest() → LabelAggregator.aggregate_batch() → TickerSentimentAnalyzer.analyze()
- Use `st.columns()` to lay out ticker cards in a 3-column grid
- Each card rendered via `st.markdown()` with HTML/CSS from styles
- Add a sidebar "Refresh Data" button that clears cache and reruns
- Show source status in sidebar (which APIs are connected)

The cached pipeline function:
```python
@st.cache_data(ttl=300)
def run_pipeline():
    config = load_config()
    mgr = IngestionManager(config)
    df = mgr.ingest()
    summary = mgr.get_source_summary()

    agg = LabelAggregator(config=config)
    df = agg.aggregate_batch(df)

    te = TickerExtractor()
    df['tickers'] = df['text'].apply(te.extract)

    analyzer = TickerSentimentAnalyzer()
    ticker_results = analyzer.analyze(df)
    market_summary = analyzer.get_market_summary(ticker_results)

    return df, ticker_results, market_summary, summary
```

For the ticker card grid, render each card with:
```python
for i, (company, data) in enumerate(ticker_results.items()):
    col = cols[i % 3]
    with col:
        if st.button(f"{data['symbol']} — {company}", key=f"ticker_{company}"):
            st.session_state['selected_ticker'] = company
            st.switch_page("pages/2_Ticker_Detail.py")
        st.markdown(f"**{data['dominant_sentiment'].upper()}** · "
                    f"{data['mention_count']} mentions · "
                    f"{data['avg_confidence']:.0%} conf")
```

**Step 2: Verify the page loads**

Run: `streamlit run app/streamlit_app.py`

Expected: Market Overview page shows ticker cards with sentiment.

**Step 3: Commit**

```bash
git add app/pages/1_Market_Overview.py
git commit -m "feat: Market Overview page with ticker cards"
```

---

### Task 5: Ticker Detail Page

**Files:**
- Create: `app/pages/2_Ticker_Detail.py`

**Step 1: Build the detail page**

This page shows when a user selects a ticker from the overview. It:
1. Gets the selected ticker from `st.session_state['selected_ticker']`
2. Also provides a dropdown to select any ticker (for direct access)
3. Shows: company name + symbol header, dominant sentiment (large, color-coded)
4. Sentiment distribution: horizontal stacked bar or pie chart via Plotly
5. Key metrics: mention count, avg confidence, bullish/bearish ratio
6. Evidence posts table: scrollable `st.dataframe` with columns — text, sentiment (color-coded), confidence, source, timestamp
7. A "Back to Overview" button

Key implementation:
- Reuse the same `@st.cache_data` pipeline function from Task 4 (import it or define a shared module)
- For the evidence posts table, use `st.dataframe()` with a styled DataFrame
- Sentiment pie uses `charts.sentiment_pie(data['sentiment'])`
- The post table is sorted by confidence (highest first)

**Step 2: Verify navigation works**

Run: `streamlit run app/streamlit_app.py`

Expected: Click a ticker on overview → see detail page with sentiment breakdown and posts.

**Step 3: Commit**

```bash
git add app/pages/2_Ticker_Detail.py
git commit -m "feat: Ticker Detail page with sentiment breakdown and evidence posts"
```

---

### Task 6: Live Inference Page

**Files:**
- Create: `app/pages/3_Live_Inference.py`

**Step 1: Build the inference page**

This page lets users paste text and get instant predictions. It:
1. Loads the trained model from `data/models/` (cached)
2. Shows a text area: "Paste a financial social media post"
3. On submit: runs model prediction + ticker extraction
4. Displays:
   - Predicted sentiment (large, color-coded) with confidence %
   - Probability bar chart (all 4 classes, Plotly horizontal bar)
   - Extracted tickers (as pills/tags)
5. Batch mode: file uploader for CSV, runs predictions on all rows, download button for results CSV

Key implementation:
```python
pipeline = SentimentPipeline()
pipeline.load("data/models")
te = TickerExtractor()

text = st.text_area("Paste a post:", height=100)
if st.button("Analyze") and text:
    result = pipeline.predict_single(text)
    tickers = te.extract(text)
    # Display result...
```

For batch mode:
```python
uploaded = st.file_uploader("Upload CSV with 'text' column", type=['csv'])
if uploaded:
    batch_df = pd.read_csv(uploaded)
    predictions = pipeline.predict(batch_df['text'].tolist())
    batch_df['sentiment'] = [p['label'] for p in predictions]
    batch_df['confidence'] = [p['confidence'] for p in predictions]
    batch_df['tickers'] = batch_df['text'].apply(te.extract)
    st.dataframe(batch_df)
    st.download_button("Download Results", batch_df.to_csv(index=False), "predictions.csv")
```

**Step 2: Verify inference works**

Run: `streamlit run app/streamlit_app.py`

Expected: Paste text → get sentiment prediction + tickers.

**Step 3: Commit**

```bash
git add app/pages/3_Live_Inference.py
git commit -m "feat: Live Inference page with single and batch prediction"
```

---

### Task 7: Under the Hood Page

**Files:**
- Create: `app/pages/4_Under_the_Hood.py`

**Step 1: Build the ML internals page**

This optional page shows model/labeling diagnostics:
1. **Model Metrics** — loads model metadata from `data/models/model_metadata.json`, shows validation F1, accuracy, confusion matrix heatmap (Plotly)
2. **Feature Importance** — top 10 features per class as bar chart (from `pipeline.get_feature_importance()`)
3. **Labeling Function Performance** — table with each function's name, coverage %, label distribution (from `LabelQualityAnalyzer.per_function_report()`)
4. **Data Quality Stats** — total posts, coverage %, conflict rate, confidence distribution histogram

All cached. Read-only diagnostic view.

**Step 2: Verify page loads**

Run: `streamlit run app/streamlit_app.py`

Expected: Under the Hood page shows model metrics and labeling stats.

**Step 3: Commit**

```bash
git add app/pages/4_Under_the_Hood.py
git commit -m "feat: Under the Hood page with model diagnostics"
```

---

### Task 8: Reddit API Integration

**Files:**
- Modify: `src/ingestion/reddit.py`

**Step 1: Implement RedditIngester.ingest()**

Replace the stub with a real PRAW implementation:

```python
import praw
import os
from datetime import datetime
import pandas as pd
from .base import BaseIngester
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RedditIngester(BaseIngester):
    def __init__(self, config):
        self.config = config
        reddit_cfg = config.get('ingestion', {}).get('reddit', {})
        self.subreddits = reddit_cfg.get('subreddits', ['wallstreetbets', 'stocks', 'investing'])
        self.post_limit = reddit_cfg.get('post_limit_per_sub', 200)
        self.min_score = reddit_cfg.get('min_score', 5)

    def is_available(self) -> bool:
        return bool(os.getenv('REDDIT_CLIENT_ID') and os.getenv('REDDIT_CLIENT_SECRET'))

    def ingest(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        if not self.is_available():
            return self._empty_dataframe()

        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'MarketPulse/1.0'),
        )

        rows = []
        for sub_name in self.subreddits:
            logger.info(f"Fetching from r/{sub_name}...")
            try:
                subreddit = reddit.subreddit(sub_name)
                for post in subreddit.new(limit=self.post_limit):
                    created = datetime.utcfromtimestamp(post.created_utc)
                    if created < start_date or created > end_date:
                        continue
                    if post.score < self.min_score:
                        continue

                    text = post.title
                    if post.selftext:
                        text += " " + post.selftext

                    rows.append({
                        'post_id': f"reddit_{sub_name}_{post.id}",
                        'text': text[:500],  # cap length
                        'source': 'reddit',
                        'timestamp': created.isoformat(),
                        'author': str(post.author) if post.author else 'unknown',
                        'score': int(post.score),
                        'url': f"https://reddit.com{post.permalink}",
                        'metadata': str({
                            'subreddit': sub_name,
                            'num_comments': post.num_comments,
                            'flair': post.link_flair_text or '',
                            'is_self': post.is_self,
                        }),
                    })
            except Exception as e:
                logger.warning(f"Error fetching r/{sub_name}: {e}")

        logger.info(f"Reddit: {len(rows)} posts fetched")
        return pd.DataFrame(rows) if rows else self._empty_dataframe()
```

**Step 2: Verify (requires API keys in .env to fully test)**

Without API keys: `is_available()` returns False, fallback to synthetic.
With API keys: test manually.

**Step 3: Commit**

```bash
git add src/ingestion/reddit.py
git commit -m "feat: Reddit ingestion via PRAW"
```

---

### Task 9: Stocktwits API Integration

**Files:**
- Modify: `src/ingestion/stocktwits.py`

**Step 1: Implement StocktwitsIngester.ingest()**

Replace the stub with real API calls:

```python
import os
import requests
from datetime import datetime
import pandas as pd
from .base import BaseIngester
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StocktwitsIngester(BaseIngester):
    BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

    def __init__(self, config):
        self.config = config
        st_cfg = config.get('ingestion', {}).get('stocktwits', {})
        self.symbols = st_cfg.get('symbols', ['AAPL', 'TSLA', 'NVDA', 'GME', 'SPY'])
        self.limit_per_symbol = st_cfg.get('limit_per_symbol', 50)

    def is_available(self) -> bool:
        # Stocktwits public API doesn't always require auth
        # but we check for token to indicate intent to use it
        return bool(os.getenv('STOCKTWITS_ACCESS_TOKEN'))

    def ingest(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        if not self.is_available():
            return self._empty_dataframe()

        token = os.getenv('STOCKTWITS_ACCESS_TOKEN')
        rows = []

        for symbol in self.symbols:
            logger.info(f"Fetching Stocktwits for {symbol}...")
            try:
                url = self.BASE_URL.format(symbol=symbol)
                params = {'access_token': token} if token else {}
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                messages = data.get('messages', [])
                for msg in messages[:self.limit_per_symbol]:
                    created = datetime.strptime(
                        msg['created_at'], "%Y-%m-%dT%H:%M:%SZ"
                    )
                    if created < start_date or created > end_date:
                        continue

                    user_sentiment = None
                    if msg.get('entities', {}).get('sentiment'):
                        user_sentiment = msg['entities']['sentiment'].get('basic')

                    rows.append({
                        'post_id': f"stocktwits_{msg['id']}",
                        'text': msg['body'][:500],
                        'source': 'stocktwits',
                        'timestamp': created.isoformat(),
                        'author': msg.get('user', {}).get('username', 'unknown'),
                        'score': msg.get('likes', {}).get('total', 0),
                        'url': f"https://stocktwits.com/message/{msg['id']}",
                        'metadata': str({
                            'symbols': [s['symbol'] for s in msg.get('symbols', [])],
                            'user_sentiment': user_sentiment,
                            'reshares': msg.get('reshares', {}).get('reshared_count', 0),
                            'likes': msg.get('likes', {}).get('total', 0),
                        }),
                    })
            except Exception as e:
                logger.warning(f"Error fetching Stocktwits {symbol}: {e}")

        logger.info(f"Stocktwits: {len(rows)} messages fetched")
        return pd.DataFrame(rows) if rows else self._empty_dataframe()
```

**Step 2: Verify (requires API key)**

Without API key: `is_available()` returns False, synthetic fallback works.

**Step 3: Commit**

```bash
git add src/ingestion/stocktwits.py
git commit -m "feat: Stocktwits ingestion via public API"
```

---

### Task 10: Shared Pipeline Runner for Dashboard

**Files:**
- Create: `app/pipeline_runner.py`

**Step 1: Create the shared pipeline runner**

This module provides a cached pipeline function used by all dashboard pages, so the pipeline only runs once per session (or on refresh).

```python
"""
Shared pipeline runner for Streamlit pages.
Runs ingestion → labeling → extraction → analysis, cached.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.ingestion.manager import IngestionManager
from src.labeling.aggregator import LabelAggregator
from src.extraction.ticker_extractor import TickerExtractor
from src.analysis.ticker_sentiment import TickerSentimentAnalyzer
from src.models.pipeline import SentimentPipeline


@st.cache_data(ttl=300)
def run_pipeline():
    """Run full pipeline and return results. Cached for 5 minutes."""
    config = load_config()
    mgr = IngestionManager(config)
    df = mgr.ingest()
    source_summary = mgr.get_source_summary()

    agg = LabelAggregator(config=config)
    df = agg.aggregate_batch(df)

    te = TickerExtractor()
    df['tickers'] = df['text'].apply(te.extract)

    analyzer = TickerSentimentAnalyzer()
    ticker_results = analyzer.analyze(df)
    market_summary = analyzer.get_market_summary(ticker_results)

    return {
        'df': df,
        'ticker_results': ticker_results,
        'market_summary': market_summary,
        'source_summary': source_summary,
        'config': config,
    }


@st.cache_resource
def load_model():
    """Load trained model. Cached across sessions."""
    model_dir = "data/models"
    if not os.path.exists(os.path.join(model_dir, 'sentiment_model.pkl')):
        return None
    pipeline = SentimentPipeline()
    pipeline.load(model_dir)
    return pipeline
```

**Step 2: Commit**

```bash
git add app/pipeline_runner.py
git commit -m "feat: shared cached pipeline runner for dashboard"
```

---

### Task 11: Final Integration and Polish

**Files:**
- Modify: `app/streamlit_app.py` (update main page)
- Modify: `Makefile` (ensure `make run` works)

**Step 1: Update main app to redirect to overview**

Update `app/streamlit_app.py` to show a brief landing or auto-redirect to the Market Overview page. Ensure sidebar shows source status.

**Step 2: Update Makefile**

Ensure `make run` launches the dashboard:
```makefile
run:
	streamlit run app/streamlit_app.py
```

Also ensure `make pipeline` still runs the CLI version.

**Step 3: Run full end-to-end verification**

1. `python3 scripts/run_pipeline.py` — CLI pipeline runs cleanly with ticker analysis
2. `streamlit run app/streamlit_app.py` — dashboard launches, Market Overview shows tickers
3. Click a ticker → Ticker Detail page works
4. Live Inference → paste text, get prediction
5. Under the Hood → shows model metrics

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: MarketPulse product pivot complete — ticker sentiment dashboard"
```
