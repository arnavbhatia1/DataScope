# Sentiment Hub + Research Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Pivot MarketPulse into a financial sentiment hub with a search-first home page and AI-powered ticker briefing cards backed by SQLite persistence and Claude synthesis.

**Architecture:** Ingest from Reddit/Stocktwits/NewsAPI → keyword/weighted vote labeling → TF-IDF + LogReg classification → SQLite storage → home page grid reads from cache → user searches ticker → Claude writes 2-3 sentence verdict from pre-aggregated data.

**Tech Stack:** Python, Streamlit, SQLite (stdlib), scikit-learn, Plotly, Anthropic SDK (`anthropic` package), pandas

---

## Task 1: Add Anthropic SDK + SQLite Storage Layer

**Files:**
- Modify: `requirements.txt`
- Create: `src/storage/__init__.py`
- Create: `src/storage/db.py`
- Create: `tests/test_db.py`

**Step 1: Add anthropic to requirements.txt**

```
anthropic>=0.40.0
```

Run: `pip install anthropic`

**Step 2: Write failing tests for the storage layer**

Create `tests/test_db.py`:

```python
"""Tests for SQLite storage layer."""
import os
import json
import pytest
import pandas as pd
import tempfile
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Override DB_PATH to a temp file for each test."""
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr("src.storage.db.DB_PATH", db_file)
    from src.storage import db
    db.init_db()
    return db


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            'post_id': 'reddit_wsb_001',
            'text': 'Loading NVDA calls, bullish af 🚀',
            'source': 'reddit',
            'timestamp': '2026-03-07 10:00:00',
            'author': 'user1',
            'score': 42,
            'tickers': ['NVIDIA'],
            'sentiment': 'bullish',
            'confidence': 0.82,
            'url': '',
        },
        {
            'post_id': 'news_001',
            'text': 'Tesla reports Q3 deliveries according to SEC filing.',
            'source': 'news',
            'timestamp': '2026-03-06 09:00:00',
            'author': 'Reuters',
            'score': 0,
            'tickers': ['Tesla'],
            'sentiment': 'neutral',
            'confidence': 0.71,
            'url': 'https://example.com',
        },
    ])


def test_init_db_creates_tables(tmp_db):
    import sqlite3
    from src.storage.db import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    assert 'posts' in tables
    assert 'ticker_cache' in tables
    assert 'model_training_log' in tables


def test_save_and_load_posts(tmp_db, sample_df):
    from src.storage.db import save_posts, load_posts
    save_posts(sample_df)
    result = load_posts()
    assert len(result) == 2
    assert set(result['post_id']) == {'reddit_wsb_001', 'news_001'}


def test_load_posts_filters_by_date(tmp_db, sample_df):
    from src.storage.db import save_posts, load_posts
    save_posts(sample_df)
    result = load_posts(start_date='2026-03-07', end_date='2026-03-07')
    assert len(result) == 1
    assert result.iloc[0]['post_id'] == 'reddit_wsb_001'


def test_save_posts_upserts(tmp_db, sample_df):
    """Saving same post twice should not duplicate it."""
    from src.storage.db import save_posts, load_posts
    save_posts(sample_df)
    save_posts(sample_df)
    result = load_posts()
    assert len(result) == 2


def test_save_and_load_ticker_cache(tmp_db):
    from src.storage.db import save_ticker_cache, load_ticker_cache
    ticker_results = {
        'Tesla': {
            'symbol': 'TSLA',
            'dominant_sentiment': 'bearish',
            'mention_count': 45,
            'reddit_sentiment': 'bearish',
            'news_sentiment': 'neutral',
            'stocktwits_sentiment': 'bearish',
            'sentiment_by_day': {'2026-03-07': 'bearish'},
            'top_posts': {'reddit': [{'text': 'TSLA puts loaded', 'sentiment': 'bearish'}]},
        }
    }
    save_ticker_cache(ticker_results)
    result = load_ticker_cache()
    assert 'Tesla' in result
    assert result['Tesla']['dominant_sentiment'] == 'bearish'
    assert result['Tesla']['sentiment_by_day'] == {'2026-03-07': 'bearish'}


def test_log_training_run(tmp_db):
    from src.storage.db import log_training_run, get_training_history
    log_training_run('run_001', num_samples=400, weighted_f1=0.74)
    history = get_training_history()
    assert len(history) == 1
    assert history[0]['run_id'] == 'run_001'
    assert history[0]['weighted_f1'] == pytest.approx(0.74)
```

**Step 3: Run tests — expect FAIL**

```bash
pytest tests/test_db.py -v
```
Expected: `ImportError: cannot import name 'save_posts' from 'src.storage.db'`

**Step 4: Implement `src/storage/__init__.py`**

```python
```
(empty file)

**Step 5: Implement `src/storage/db.py`**

```python
"""
SQLite storage layer for MarketPulse.

Single .db file at data/marketpulse.db — free, persistent, no cloud required.
All reads/writes go through this module. No direct sqlite3 calls elsewhere.
"""

import sqlite3
import json
import os
from datetime import datetime
import pandas as pd

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "marketpulse.db"
)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call repeatedly."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS posts (
            post_id    TEXT PRIMARY KEY,
            text       TEXT NOT NULL,
            source     TEXT NOT NULL,
            timestamp  TEXT,
            author     TEXT,
            score      INTEGER DEFAULT 0,
            tickers    TEXT DEFAULT '[]',
            sentiment  TEXT,
            confidence REAL DEFAULT 0.0,
            url        TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS ticker_cache (
            ticker                TEXT PRIMARY KEY,
            symbol                TEXT,
            last_updated          TEXT,
            dominant_sentiment    TEXT,
            mention_count         INTEGER DEFAULT 0,
            reddit_sentiment      TEXT,
            news_sentiment        TEXT,
            stocktwits_sentiment  TEXT,
            sentiment_by_day      TEXT DEFAULT '{}',
            top_posts             TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS model_training_log (
            run_id       TEXT PRIMARY KEY,
            trained_at   TEXT,
            num_samples  INTEGER,
            weighted_f1  REAL,
            label_source TEXT DEFAULT 'keyword_majority'
        );
    """)
    conn.commit()
    conn.close()


def save_posts(df: pd.DataFrame):
    """Upsert posts DataFrame into SQLite. Safe to call multiple times."""
    conn = get_connection()
    for _, row in df.iterrows():
        tickers = row.get('tickers', [])
        if not isinstance(tickers, list):
            tickers = []
        conn.execute(
            """INSERT OR REPLACE INTO posts
               (post_id, text, source, timestamp, author, score,
                tickers, sentiment, confidence, url)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(row['post_id']),
                str(row['text']),
                str(row.get('source', 'unknown')),
                str(row.get('timestamp', '')),
                str(row.get('author', 'unknown')),
                int(row.get('score', 0)),
                json.dumps(tickers),
                row.get('sentiment'),
                float(row.get('confidence', 0.0)),
                str(row.get('url', '')),
            )
        )
    conn.commit()
    conn.close()


def load_posts(start_date=None, end_date=None) -> pd.DataFrame:
    """Load posts, optionally filtered by date range (YYYY-MM-DD strings)."""
    conn = get_connection()
    if start_date and end_date:
        rows = conn.execute(
            "SELECT * FROM posts WHERE timestamp >= ? AND timestamp <= ?",
            (str(start_date), str(end_date) + " 23:59:59")
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM posts").fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame(columns=[
            'post_id', 'text', 'source', 'timestamp', 'author',
            'score', 'tickers', 'sentiment', 'confidence', 'url'
        ])

    df = pd.DataFrame([dict(r) for r in rows])
    df['tickers'] = df['tickers'].apply(
        lambda x: json.loads(x) if x else []
    )
    return df


def save_ticker_cache(ticker_results: dict):
    """Upsert ticker_results dict into ticker_cache table."""
    conn = get_connection()
    now = datetime.utcnow().isoformat()
    for company, data in ticker_results.items():
        conn.execute(
            """INSERT OR REPLACE INTO ticker_cache
               (ticker, symbol, last_updated, dominant_sentiment, mention_count,
                reddit_sentiment, news_sentiment, stocktwits_sentiment,
                sentiment_by_day, top_posts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                company,
                data.get('symbol', ''),
                now,
                data.get('dominant_sentiment', 'neutral'),
                int(data.get('mention_count', 0)),
                data.get('reddit_sentiment', 'neutral'),
                data.get('news_sentiment', 'neutral'),
                data.get('stocktwits_sentiment', 'neutral'),
                json.dumps(data.get('sentiment_by_day', {})),
                json.dumps(data.get('top_posts', {})),
            )
        )
    conn.commit()
    conn.close()


def load_ticker_cache() -> dict:
    """Return all rows from ticker_cache as dict keyed by company name."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM ticker_cache").fetchall()
    conn.close()
    result = {}
    for row in rows:
        d = dict(row)
        d['sentiment_by_day'] = json.loads(d.get('sentiment_by_day') or '{}')
        d['top_posts'] = json.loads(d.get('top_posts') or '{}')
        result[d['ticker']] = d
    return result


def log_training_run(run_id: str, num_samples: int, weighted_f1: float,
                     label_source: str = 'keyword_majority'):
    """Record a model training run."""
    conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO model_training_log
           (run_id, trained_at, num_samples, weighted_f1, label_source)
           VALUES (?, ?, ?, ?, ?)""",
        (run_id, datetime.utcnow().isoformat(), num_samples, weighted_f1, label_source)
    )
    conn.commit()
    conn.close()


def get_training_history() -> list:
    """Return all training runs as list of dicts, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM model_training_log ORDER BY trained_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
```

**Step 6: Run tests — expect PASS**

```bash
pytest tests/test_db.py -v
```
Expected: all 6 tests PASS

**Step 7: Commit**

```bash
git add src/storage/__init__.py src/storage/db.py tests/test_db.py requirements.txt
git commit -m "feat: add SQLite storage layer and anthropic dependency"
```

---

## Task 2: Extend TickerSentimentAnalyzer with Per-Source + Time-Series Data

The current `analyze()` method doesn't produce `reddit_sentiment`, `news_sentiment`, `stocktwits_sentiment`, `sentiment_by_day`, or `top_posts`. SQLite needs all of these.

**Files:**
- Modify: `src/analysis/ticker_sentiment.py`
- Modify: `tests/test_extraction.py` (add new assertions — don't break existing)

**Step 1: Write failing tests**

Add to `tests/test_extraction.py` (or create `tests/test_ticker_sentiment.py`):

```python
"""Tests for extended TickerSentimentAnalyzer fields."""
import os, sys, pytest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.ticker_sentiment import TickerSentimentAnalyzer


@pytest.fixture
def multi_source_df():
    return pd.DataFrame([
        {'text': 'Loading TSLA calls 🚀', 'source': 'reddit',
         'programmatic_label': 'bullish', 'label_confidence': 0.8,
         'tickers': ['Tesla'], 'timestamp': '2026-03-07 10:00:00'},
        {'text': 'Shorting TSLA, overvalued', 'source': 'stocktwits',
         'programmatic_label': 'bearish', 'label_confidence': 0.75,
         'tickers': ['Tesla'], 'timestamp': '2026-03-06 11:00:00'},
        {'text': 'Tesla reports Q3 deliveries per SEC filing', 'source': 'news',
         'programmatic_label': 'neutral', 'label_confidence': 0.9,
         'tickers': ['Tesla'], 'timestamp': '2026-03-05 09:00:00'},
        {'text': 'TSLA puts loaded, crash incoming 📉', 'source': 'reddit',
         'programmatic_label': 'bearish', 'label_confidence': 0.7,
         'tickers': ['Tesla'], 'timestamp': '2026-03-07 14:00:00'},
    ])


def test_per_source_sentiment(multi_source_df):
    analyzer = TickerSentimentAnalyzer()
    results = analyzer.analyze(multi_source_df)
    tesla = results['Tesla']
    assert 'reddit_sentiment' in tesla
    assert 'news_sentiment' in tesla
    assert 'stocktwits_sentiment' in tesla
    assert tesla['news_sentiment'] == 'neutral'
    assert tesla['stocktwits_sentiment'] == 'bearish'


def test_sentiment_by_day(multi_source_df):
    analyzer = TickerSentimentAnalyzer()
    results = analyzer.analyze(multi_source_df)
    tesla = results['Tesla']
    assert 'sentiment_by_day' in tesla
    assert isinstance(tesla['sentiment_by_day'], dict)
    # Should have entries for 3 different days
    assert len(tesla['sentiment_by_day']) == 3


def test_top_posts_per_source(multi_source_df):
    analyzer = TickerSentimentAnalyzer()
    results = analyzer.analyze(multi_source_df)
    tesla = results['Tesla']
    assert 'top_posts' in tesla
    assert isinstance(tesla['top_posts'], dict)
    # Should have reddit and stocktwits and news keys
    assert 'reddit' in tesla['top_posts']
    assert isinstance(tesla['top_posts']['reddit'], list)
    # Top posts capped at 3 per source
    assert len(tesla['top_posts']['reddit']) <= 3
```

**Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_ticker_sentiment.py -v
```
Expected: `KeyError: 'reddit_sentiment'`

**Step 3: Update `src/analysis/ticker_sentiment.py`**

Replace the `analyze()` method and add a helper. Keep `get_market_summary()` unchanged:

```python
def analyze(self, df):
    df = df.copy()

    if 'tickers' not in df.columns:
        df['tickers'] = df['text'].apply(self.extractor.extract)

    labeled = df[df['programmatic_label'].notna()].copy()

    # Normalize timestamp column to date string
    labeled['_date'] = pd.to_datetime(
        labeled['timestamp'], errors='coerce'
    ).dt.strftime('%Y-%m-%d').fillna('unknown')

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
                    'all_posts': [],
                }

            ticker_data[company]['all_posts'].append({
                'post_id': str(row.get('post_id', '')),
                'text': str(row['text']),
                'sentiment': row['programmatic_label'],
                'confidence': float(row.get('label_confidence', 0)),
                'source': str(row.get('source', 'unknown')),
                'timestamp': str(row.get('timestamp', '')),
                'date': str(row['_date']),
                'author': str(row.get('author', 'unknown')),
                'url': str(row.get('url', '')),
            })

    results = {}
    for company, data in ticker_data.items():
        posts = data['all_posts']
        total = len(posts)
        if total == 0:
            continue

        all_sentiments = [p['sentiment'] for p in posts]
        sentiment_counts = Counter(all_sentiments)
        dominant = sentiment_counts.most_common(1)[0][0]

        # Per-source dominant sentiment
        source_sentiments = {}
        for src in ('reddit', 'stocktwits', 'news'):
            src_posts = [p for p in posts if p['source'] == src]
            if src_posts:
                src_counts = Counter(p['sentiment'] for p in src_posts)
                source_sentiments[src] = src_counts.most_common(1)[0][0]
            else:
                source_sentiments[src] = None

        # Sentiment by day (dominant per day)
        by_day = {}
        from itertools import groupby
        day_groups = {}
        for p in posts:
            day_groups.setdefault(p['date'], []).append(p['sentiment'])
        for day, sentiments in day_groups.items():
            by_day[day] = Counter(sentiments).most_common(1)[0][0]

        # Top 3 posts per source (sorted by confidence desc)
        top_posts = {}
        for src in ('reddit', 'stocktwits', 'news'):
            src_posts = sorted(
                [p for p in posts if p['source'] == src],
                key=lambda p: p['confidence'], reverse=True
            )[:3]
            if src_posts:
                top_posts[src] = src_posts

        confidences = [p['confidence'] for p in posts]
        results[company] = {
            'company': company,
            'symbol': data['symbol'],
            'mention_count': total,
            'sentiment': dict(sentiment_counts),
            'dominant_sentiment': dominant,
            'dominant_color': SENTIMENT_COLORS.get(dominant, '#78909C'),
            'bullish_ratio': sentiment_counts.get('bullish', 0) / total,
            'bearish_ratio': sentiment_counts.get('bearish', 0) / total,
            'avg_confidence': sum(confidences) / len(confidences),
            'reddit_sentiment': source_sentiments['reddit'],
            'news_sentiment': source_sentiments['news'],
            'stocktwits_sentiment': source_sentiments['stocktwits'],
            'sentiment_by_day': by_day,
            'top_posts': top_posts,
            'posts': sorted(posts, key=lambda p: p['confidence'], reverse=True),
        }

    results = dict(sorted(
        results.items(), key=lambda x: x[1]['mention_count'], reverse=True
    ))
    logger.info(f"Ticker analysis: {len(results)} tickers, {len(labeled)} labeled posts")
    return results
```

**Step 4: Run tests — expect PASS**

```bash
pytest tests/test_ticker_sentiment.py tests/test_extraction.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/analysis/ticker_sentiment.py tests/test_ticker_sentiment.py
git commit -m "feat: add per-source sentiment, sentiment_by_day, and top_posts to ticker analysis"
```

---

## Task 3: Update Pipeline Runner to Use SQLite

**Files:**
- Modify: `app/pipeline_runner.py`

The pipeline runner currently holds everything in memory and reruns on every cache miss. Refactor it to:
1. `refresh_pipeline()` — ingest + label + analyze + write to SQLite (called on startup/refresh)
2. `get_ticker_cache()` — read from SQLite (fast, called by home page)
3. Keep `load_model()` as-is

**Step 1: Replace `app/pipeline_runner.py`**

```python
"""
Pipeline runner for MarketPulse.

Two distinct operations:
  refresh_pipeline() — slow: ingest → label → analyze → write SQLite
  get_ticker_cache() — fast: read SQLite → return to dashboard

Pages import get_ticker_cache() for display and call refresh_pipeline()
only on startup or when user clicks Refresh.
"""

import streamlit as st
import sys
import os
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.config import load_config
from src.ingestion.manager import IngestionManager
from src.labeling.aggregator import LabelAggregator
from src.extraction.ticker_extractor import TickerExtractor
from src.analysis.ticker_sentiment import TickerSentimentAnalyzer
from src.storage.db import (
    init_db, save_posts, load_posts,
    save_ticker_cache, load_ticker_cache
)


def refresh_pipeline(start_date_str=None, end_date_str=None) -> dict:
    """
    Run full pipeline: ingest → label → extract → analyze → write SQLite.

    Expensive. Call on startup or manual Refresh only.
    Returns source_summary for status display.
    """
    init_db()
    config = load_config()

    start_date = datetime.fromisoformat(start_date_str) if start_date_str else None
    end_date = datetime.fromisoformat(end_date_str) if end_date_str else None

    # Ingest
    mgr = IngestionManager(config)
    df = mgr.ingest(start_date=start_date, end_date=end_date)
    source_summary = mgr.get_source_summary()

    # Label
    agg = LabelAggregator(config=config)
    df = agg.aggregate_batch(df)

    # Extract tickers
    te = TickerExtractor()
    df['tickers'] = df['text'].apply(te.extract)

    # Map programmatic_label → sentiment column for storage
    df['sentiment'] = df['programmatic_label']
    df['confidence'] = df['label_confidence'].fillna(0.0)

    # Analyze per-ticker
    analyzer = TickerSentimentAnalyzer()
    ticker_results = analyzer.analyze(df)

    # Write to SQLite
    save_posts(df)
    save_ticker_cache(ticker_results)

    # Optionally train/retrain model if enough labeled data
    _maybe_train_model(df, config)

    return source_summary


@st.cache_data(ttl=60)
def get_ticker_cache() -> dict:
    """
    Read ticker_cache from SQLite. Cached for 60 seconds.
    Fast — no ingestion, no compute.
    """
    init_db()
    return load_ticker_cache()


@st.cache_data(ttl=60)
def get_market_summary(ticker_results: dict) -> dict:
    """Compute market-level summary from ticker_results."""
    from src.analysis.ticker_sentiment import TickerSentimentAnalyzer
    analyzer = TickerSentimentAnalyzer()
    return analyzer.get_market_summary(ticker_results)


@st.cache_resource
def load_model():
    """Load trained sentiment model from disk. None if not yet trained."""
    from src.models.pipeline import SentimentPipeline
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


def _maybe_train_model(df, config, min_samples=200):
    """
    Train model if we have enough labeled data and no model exists yet.
    Silently skips if not enough data.
    """
    from src.models.pipeline import SentimentPipeline
    from src.storage.db import log_training_run
    import uuid

    labeled = df[df['programmatic_label'].notna()]
    if len(labeled) < min_samples:
        return

    model_path = os.path.join(_project_root, "data", "models", "sentiment_model.pkl")
    if os.path.exists(model_path):
        return  # Already trained — don't auto-retrain

    pipeline = SentimentPipeline(config=config)
    report = pipeline.train(
        labeled['text'].tolist(),
        labeled['programmatic_label'].tolist()
    )
    pipeline.save(os.path.join(_project_root, "data", "models"))

    f1 = report.get('weighted_f1', 0.0) if report else 0.0
    log_training_run(
        run_id=str(uuid.uuid4()),
        num_samples=len(labeled),
        weighted_f1=f1,
    )
```

**Step 2: Verify existing pages still import cleanly**

```bash
python -c "from app.pipeline_runner import get_ticker_cache, load_model; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add app/pipeline_runner.py
git commit -m "feat: refactor pipeline_runner to use SQLite (refresh vs read separation)"
```

---

## Task 4: Claude Briefing Agent

**Files:**
- Create: `src/agent/__init__.py`
- Create: `src/agent/briefing.py`
- Create: `tests/test_briefing.py`

**Step 1: Write failing test (mock Claude)**

Create `tests/test_briefing.py`:

```python
"""Tests for the Claude briefing agent."""
import os, sys, pytest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


SAMPLE_TICKER_DATA = {
    'symbol': 'TSLA',
    'dominant_sentiment': 'bearish',
    'mention_count': 120,
    'reddit_sentiment': 'bearish',
    'news_sentiment': 'neutral',
    'stocktwits_sentiment': 'bearish',
    'sentiment_by_day': {
        '2026-03-05': 'neutral',
        '2026-03-06': 'bearish',
        '2026-03-07': 'bearish',
    },
    'top_posts': {
        'reddit': [
            {'text': 'TSLA puts loaded, P/E is insane', 'sentiment': 'bearish'},
            {'text': 'Shorting TSLA at these levels', 'sentiment': 'bearish'},
        ],
        'news': [
            {'text': 'Tesla reports Q3 deliveries at 435K units', 'sentiment': 'neutral'},
        ],
    },
}


def test_generate_briefing_returns_string():
    """generate_briefing returns a non-empty string."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Tesla sentiment has turned bearish this week.")]

    with patch('src.agent.briefing.anthropic.Anthropic') as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        from src.agent.briefing import generate_briefing
        result = generate_briefing('Tesla', 'TSLA', SAMPLE_TICKER_DATA)

    assert isinstance(result, str)
    assert len(result) > 10


def test_generate_briefing_calls_claude_once():
    """Only one API call per briefing."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Some verdict.")]

    with patch('src.agent.briefing.anthropic.Anthropic') as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        from src.agent import briefing
        # Reload to pick up fresh mock
        import importlib
        importlib.reload(briefing)
        briefing.generate_briefing('Tesla', 'TSLA', SAMPLE_TICKER_DATA)

    assert mock_client.messages.create.call_count == 1


def test_generate_briefing_no_api_key(monkeypatch):
    """Returns fallback string when ANTHROPIC_API_KEY is missing."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

    with patch('src.agent.briefing.anthropic.Anthropic') as mock_client_cls:
        mock_client_cls.side_effect = Exception("No API key")

        from src.agent import briefing
        import importlib
        importlib.reload(briefing)
        result = briefing.generate_briefing('Tesla', 'TSLA', SAMPLE_TICKER_DATA)

    assert isinstance(result, str)
    assert len(result) > 0  # Returns fallback, doesn't crash
```

**Step 2: Run — expect FAIL**

```bash
pytest tests/test_briefing.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.agent'`

**Step 3: Implement `src/agent/__init__.py`**

```python
```
(empty)

**Step 4: Implement `src/agent/briefing.py`**

```python
"""
Claude briefing agent for MarketPulse.

One API call per ticker search. Claude writes the AI Verdict paragraph only.
All charts, stats, and post samples come from SQLite — Claude is narrative only.
"""

import json
import os

import anthropic

_FALLBACK = (
    "Sentiment data has been aggregated from Reddit, Stocktwits, and financial news. "
    "See the source breakdown below for details."
)


def generate_briefing(company: str, ticker: str, ticker_data: dict) -> str:
    """
    Generate a 2-3 sentence AI verdict for a ticker briefing card.

    Args:
        company: canonical company name e.g. "Tesla"
        ticker:  stock symbol e.g. "TSLA"
        ticker_data: dict from ticker_cache (SQLite row with all fields)

    Returns:
        2-3 sentence verdict string. Returns a safe fallback if Claude unavailable.
    """
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        prompt = _build_prompt(company, ticker, ticker_data)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception:
        return _FALLBACK


def _build_prompt(company: str, ticker: str, ticker_data: dict) -> str:
    """Build the structured prompt sent to Claude."""
    top_posts = ticker_data.get('top_posts', {})

    samples = []
    for source in ('reddit', 'stocktwits', 'news'):
        for post in top_posts.get(source, [])[:2]:
            text = str(post.get('text', ''))[:140]
            samples.append(f"[{source}] {text}")

    trend = json.dumps(ticker_data.get('sentiment_by_day', {}))

    return f"""You are a financial sentiment analyst writing for a market intelligence dashboard.

Ticker: {ticker} ({company})
Overall sentiment: {ticker_data.get('dominant_sentiment', 'unknown')}
Mentions: {ticker_data.get('mention_count', 0)} posts
Reddit: {ticker_data.get('reddit_sentiment', 'N/A')}
Stocktwits: {ticker_data.get('stocktwits_sentiment', 'N/A')}
News: {ticker_data.get('news_sentiment', 'N/A')}
7-day trend: {trend}

Sample posts:
{chr(10).join(samples) if samples else 'No sample posts available.'}

Write a 2-3 sentence verdict summarizing the current sentiment. Be specific about \
which sources are most bearish/bullish and any trend direction. Plain prose, no \
bullets, no headers. Do not start with "Based on" or "According to"."""
```

**Step 5: Run tests — expect PASS**

```bash
pytest tests/test_briefing.py -v
```
Expected: all 3 PASS

**Step 6: Commit**

```bash
git add src/agent/__init__.py src/agent/briefing.py tests/test_briefing.py
git commit -m "feat: add Claude briefing agent with fallback"
```

---

## Task 5: Redesign Home Page

**Files:**
- Modify: `app/MarketPulse.py`

Replace the current pipeline-on-load home page with:
- Search bar as primary action (top of page)
- Briefing card renders inline when ticker is searched
- Market overview grid below (reads from SQLite via `get_ticker_cache`)
- Refresh button triggers `refresh_pipeline()`

**Step 1: Replace `app/MarketPulse.py`**

```python
"""
MarketPulse — Financial Sentiment Hub

Run: streamlit run app/MarketPulse.py
"""

import streamlit as st
import sys, os, json
from datetime import date, timedelta

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.components.styles import apply_theme, COLORS, SENTIMENT_COLORS
from app.components.charts import ticker_mentions_bar

st.set_page_config(
    page_title="MarketPulse",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("MarketPulse")
st.sidebar.markdown("**Sentiment intelligence for financial markets**")
st.sidebar.markdown("---")

_today = date.today()
start_date = st.sidebar.date_input(
    "Start date", value=_today - timedelta(days=7),
    min_value=_today - timedelta(days=30), max_value=_today,
)
end_date = st.sidebar.date_input(
    "End date", value=_today,
    min_value=_today - timedelta(days=30), max_value=_today,
)
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")

st.session_state["start_date"] = start_date.isoformat()
st.session_state["end_date"] = end_date.isoformat()

if st.sidebar.button("Refresh Data", use_container_width=True):
    with st.spinner("Ingesting and analyzing market data..."):
        from app.pipeline_runner import refresh_pipeline
        source_summary = refresh_pipeline(
            start_date_str=start_date.isoformat(),
            end_date_str=end_date.isoformat(),
        )
        st.cache_data.clear()
    st.rerun()

# Model status
from app.pipeline_runner import load_model, get_ticker_cache
from src.storage.db import init_db, get_training_history

init_db()
model = load_model()
if model and model.is_trained:
    history = get_training_history()
    f1 = history[0]['weighted_f1'] if history else 0.0
    st.sidebar.success(f"Model trained (F1: {f1:.2f})")
else:
    st.sidebar.info("Keyword fallback active (model not yet trained)")

st.sidebar.markdown("---")

# ── Load ticker cache ─────────────────────────────────────────────────────────
ticker_results = get_ticker_cache()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("MarketPulse")
st.markdown("Sentiment intelligence for financial markets.")
st.markdown("---")

# ── Search bar (PRIMARY) ──────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        label="Research a ticker",
        placeholder="TSLA, NVDA, AAPL...",
        label_visibility="collapsed",
    )
with col_btn:
    search_clicked = st.button("Research", use_container_width=True)

# ── Briefing card (inline, below search) ─────────────────────────────────────
if search_clicked and query.strip():
    from src.extraction.ticker_extractor import TickerExtractor
    from src.extraction.normalizer import EntityNormalizer
    from src.agent.briefing import generate_briefing

    te = TickerExtractor()
    normalizer = EntityNormalizer()

    # Resolve query to canonical company name
    resolved = normalizer.normalize(query.strip())
    ticker_data = ticker_results.get(resolved)

    # Fallback: try symbol lookup
    if not ticker_data:
        symbol_upper = query.strip().upper()
        for company, data in ticker_results.items():
            if data.get('symbol', '').upper() == symbol_upper:
                ticker_data = data
                resolved = company
                break

    if not ticker_data:
        st.warning(f"No data found for **{query.strip()}**. Try refreshing data or check the ticker symbol.")
    else:
        symbol = ticker_data.get('symbol', resolved.upper())
        dominant = ticker_data.get('dominant_sentiment', 'neutral')
        color = SENTIMENT_COLORS.get(dominant, COLORS['secondary'])
        mention_count = ticker_data.get('mention_count', 0)
        last_updated = ticker_data.get('last_updated', 'unknown')

        # Header card
        st.markdown(f"""
        <div class="ticker-card" style="border-left: 4px solid {color}; padding: 16px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:1.6em; font-weight:bold;">{symbol}</span>
                    <span style="color:#8B949E; margin-left:10px;">{resolved}</span>
                </div>
                <div class="sentiment-{dominant}" style="font-size:1.2em; font-weight:bold;">
                    {dominant.upper()}
                </div>
            </div>
            <div style="color:#8B949E; font-size:0.85em; margin-top:4px;">
                {mention_count} mentions · updated {last_updated[:16] if last_updated != 'unknown' else 'unknown'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # AI Verdict
        with st.container():
            st.markdown("#### AI Verdict")
            with st.spinner("Generating verdict..."):
                verdict = generate_briefing(resolved, symbol, ticker_data)
            st.info(f'"{verdict}"\n\n— MarketPulse AI')

        # Sentiment trend chart
        by_day = ticker_data.get('sentiment_by_day', {})
        if by_day:
            st.markdown("#### Sentiment Trend (7 days)")
            import plotly.graph_objects as go
            import pandas as pd

            days = sorted(by_day.keys())
            sentiment_order = ['bullish', 'bearish', 'neutral', 'meme']

            # Build daily counts from posts (recompute from ticker top_posts is limited,
            # so we use the by_day dominant as the label)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=days,
                y=[1 if by_day[d] == 'bullish' else 0 for d in days],
                name='Bullish', mode='lines+markers',
                line=dict(color=SENTIMENT_COLORS['bullish'])
            ))
            fig.add_trace(go.Scatter(
                x=days,
                y=[1 if by_day[d] == 'bearish' else 0 for d in days],
                name='Bearish', mode='lines+markers',
                line=dict(color=SENTIMENT_COLORS['bearish'])
            ))
            fig.update_layout(
                template='plotly_dark', height=200,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        # By Source breakdown
        st.markdown("#### By Source")
        top_posts = ticker_data.get('top_posts', {})
        src_cols = st.columns(3)
        for i, source in enumerate(('reddit', 'stocktwits', 'news')):
            src_sentiment = ticker_data.get(f'{source}_sentiment') or 'N/A'
            src_posts = top_posts.get(source, [])
            src_color = SENTIMENT_COLORS.get(src_sentiment, COLORS['secondary'])
            with src_cols[i]:
                st.markdown(f"**{source.upper()}**")
                st.markdown(
                    f"<span style='color:{src_color}; font-weight:bold;'>"
                    f"{src_sentiment.upper() if src_sentiment else 'N/A'}</span>",
                    unsafe_allow_html=True
                )
                for post in src_posts[:3]:
                    st.caption(f"> {post['text'][:100]}...")

    st.markdown("---")

# ── Market Overview grid (SECONDARY) ─────────────────────────────────────────
if not ticker_results:
    st.info(
        "No market data yet. Click **Refresh Data** in the sidebar to ingest and analyze."
    )
else:
    st.markdown("### Market Overview")

    # KPI row
    from collections import Counter
    sentiment_dist = Counter(
        v['dominant_sentiment'] for v in ticker_results.values()
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tickers Tracked", len(ticker_results))
    k2.metric("Bullish", sentiment_dist.get('bullish', 0))
    k3.metric("Bearish", sentiment_dist.get('bearish', 0))
    k4.metric("Neutral", sentiment_dist.get('neutral', 0))

    st.markdown("---")

    # Ticker card grid
    cols = st.columns(3)
    for i, (company, data) in enumerate(ticker_results.items()):
        sentiment = data.get('dominant_sentiment', 'neutral')
        color = SENTIMENT_COLORS.get(sentiment, COLORS['secondary'])
        symbol = data.get('symbol', company.upper())
        mentions = data.get('mention_count', 0)
        conf = data.get('avg_confidence', 0.0)

        with cols[i % 3]:
            st.markdown(f"""
            <div class="ticker-card">
                <div style="font-size:1.2em; font-weight:bold;">{symbol}</div>
                <div style="color:#8B949E; font-size:0.85em;">{company}</div>
                <div class="sentiment-{sentiment}" style="margin:6px 0;">
                    {sentiment.upper()}
                </div>
                <div style="color:#8B949E; font-size:0.8em;">
                    {mentions} mentions · {conf:.0%} confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Most mentioned bar chart
    st.markdown("---")
    st.markdown("### Most Mentioned Tickers")
    fig = ticker_mentions_bar(ticker_results, top_n=15)
    st.plotly_chart(fig, use_container_width=True)
```

**Step 2: Verify app starts without error**

```bash
cd /Users/arnavbhatia/Desktop/MarketPulse
python -c "import ast; ast.parse(open('app/MarketPulse.py').read()); print('syntax OK')"
```
Expected: `syntax OK`

**Step 3: Commit**

```bash
git add app/MarketPulse.py
git commit -m "feat: redesign home page — search-first with inline briefing card and market grid"
```

---

## Task 6: Cleanup — Remove Unused Code and Pages

**Files:**
- Delete: `app/pages/2_Live_Inference.py`
- Delete: `app/pages/3_Under_the_Hood.py`
- Delete: `src/evaluation/` directory
- Delete: `src/models/versioning.py`
- Delete: `tests/test_evaluation.py`
- Modify: `CLAUDE.md` — update to reflect new architecture

**Step 1: Remove files**

```bash
cd /Users/arnavbhatia/Desktop/MarketPulse
rm app/pages/2_Live_Inference.py
rm app/pages/3_Under_the_Hood.py
rm -rf src/evaluation/
rm src/models/versioning.py
rm tests/test_evaluation.py
```

**Step 2: Run remaining tests to confirm nothing broke**

```bash
pytest tests/ -v --ignore=tests/test_evaluation.py
```
Expected: all pass (test_evaluation.py is gone so no need to ignore, but safe to specify)

**Step 3: Update `CLAUDE.md` architecture section**

Update the ARCHITECTURE section file tree to remove deleted paths and add new ones:
- Add `src/storage/db.py`
- Add `src/agent/briefing.py`
- Remove `src/evaluation/`
- Remove `src/models/versioning.py`
- Remove deleted page files

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove unused evaluation, versioning, and old dashboard pages"
```

---

## Task 7: End-to-End Smoke Test

Verify the full happy path works before calling it done.

**Step 1: Initialize DB**

```bash
python -c "from src.storage.db import init_db; init_db(); print('DB initialized')"
```
Expected: `DB initialized`, `data/marketpulse.db` exists

**Step 2: Run pipeline in synthetic mode**

```bash
python scripts/run_pipeline.py --synthetic
```
Expected: posts ingested, labeled, written to SQLite, model training attempted

**Step 3: Verify SQLite has data**

```bash
python -c "
from src.storage.db import load_posts, load_ticker_cache
posts = load_posts()
cache = load_ticker_cache()
print(f'Posts: {len(posts)}')
print(f'Tickers: {len(cache)}')
print('Sample ticker:', list(cache.keys())[:3])
"
```
Expected: `Posts: 500+`, `Tickers: 10+`

**Step 4: Start dashboard**

```bash
streamlit run app/MarketPulse.py
```
Expected: home page loads, market grid shows ticker cards, search works, briefing card renders.

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete sentiment hub + research agent pivot"
```

---

## Summary

| Task | Key Output |
|------|-----------|
| 1 | `src/storage/db.py` — SQLite layer with init, save, load, cache |
| 2 | `ticker_sentiment.py` extended with per-source + time-series fields |
| 3 | `pipeline_runner.py` refactored — refresh vs read separation |
| 4 | `src/agent/briefing.py` — Claude verdict with fallback |
| 5 | `app/MarketPulse.py` — search-first home page with inline briefing |
| 6 | Dead code removed (evaluation, versioning, old pages) |
| 7 | End-to-end smoke test |
