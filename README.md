# MarketPulse

Financial sentiment hub that tracks market mood across Reddit, Stocktwits, and financial news. Search any ticker to get a structured briefing card — per-source sentiment breakdown, 7-day trend, and a 2-3 sentence AI verdict synthesized by Claude.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env          # add your API keys (see below)
streamlit run app/MarketPulse.py
```

Opens at **http://localhost:8501**. Click **Refresh Data** in the sidebar to ingest and analyze posts (~30 seconds).

## API Keys

| Key | What it enables | Where to get it |
|-----|----------------|--------------------|
| `ANTHROPIC_API_KEY` | AI verdict on briefing cards | [console.anthropic.com](https://console.anthropic.com) → API Keys |
| `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` | Reddit/WSB posts | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) → create script app |
| `STOCKTWITS_ACCESS_TOKEN` | Stocktwits messages | [api.stocktwits.com/developers](https://api.stocktwits.com/developers) |

**News is always free** — scraped from Google News RSS and Yahoo Finance RSS, no key needed. The app works without any keys (AI verdict falls back to a static summary).

## How It Works

```
Ingest (Reddit + Stocktwits + free News RSS)
        ↓
16 keyword/emoji labeling functions → confidence-weighted vote
        ↓
TF-IDF + LogReg classifies all posts (auto-trains when enough data)
        ↓
Posts + per-ticker summaries saved to SQLite
        ↓
Home grid reads from SQLite (instant load)
User searches ticker → Claude writes 2-3 sentence verdict → briefing card
```

## Using the App

**Search a ticker:** Enter a symbol (`TSLA`) or company name (`Tesla`) and click **Research**. The briefing card renders inline with sentiment by source, a 7-day trend chart, and top post snippets.

**Refresh data:** Click **Refresh Data** in the sidebar. Use the date range picker to control the lookback window (up to 30 days).

**Market grid:** Shows all tracked tickers color-coded by dominant sentiment. Updates after each refresh.

## Project Structure

```
MarketPulse/
├── app/
│   ├── MarketPulse.py          # Home page: search bar + market grid
│   ├── pipeline_runner.py      # refresh_pipeline(), get_ticker_cache(), load_model()
│   └── pages/
│       └── 1_Ticker_Detail.py  # Deep-dive page for a single ticker
├── src/
│   ├── ingestion/              # Reddit, Stocktwits, News RSS ingesters
│   ├── labeling/               # 16 labeling functions + confidence-weighted aggregator
│   ├── models/                 # TF-IDF + LogReg training pipeline
│   ├── extraction/             # Ticker entity extraction + normalization
│   ├── analysis/               # Per-ticker sentiment aggregation
│   ├── storage/db.py           # SQLite read/write (data/marketpulse.db)
│   └── agent/briefing.py       # Claude synthesis — one API call per search
├── scripts/run_pipeline.py     # CLI: full pipeline end-to-end
├── config/default.yaml         # Data sources, model hyperparameters
└── tests/                      # 158 tests
```

## CLI

```bash
# Full pipeline (ingest → label → analyze → store → train)
python scripts/run_pipeline.py

# With custom lookback
python scripts/run_pipeline.py --days 14

# Tests
pytest tests/ -v
```

## Tech Stack

Python 3.9+ · Streamlit · scikit-learn · Plotly · Anthropic SDK · SQLite · feedparser · PRAW · pandas
