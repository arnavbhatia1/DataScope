# MarketPulse

Sentiment intelligence for financial markets. MarketPulse ingests financial social media posts, applies programmatic labeling through weak supervision, trains an ML classifier, extracts ticker entities, and surfaces per-ticker sentiment through an interactive Streamlit dashboard.

## Quick Start

```bash
# Install dependencies
make setup

# Run the full pipeline (ingest, label, train, analyze)
make pipeline

# Launch the dashboard
make run
```

The dashboard opens at **http://localhost:8501**. No API keys needed — synthetic data is used by default so the full pipeline runs out of the box.

## Dashboard Pages

| Page | What it shows |
|------|--------------|
| **Home** | Market snapshot — top bullish/bearish tickers, sentiment distribution |
| **Market Overview** | Full ticker grid with sentiment cards, mentions chart |
| **Ticker Detail** | Drill into any ticker — sentiment breakdown, evidence posts, CSV export |
| **Live Inference** | Classify any text in real time — single post or batch CSV |
| **Under the Hood** | ML diagnostics — model metrics, feature importance, labeling function performance |

## How It Works

1. **Ingestion** — Pulls posts from Reddit (WSB), Stocktwits, and financial news. Falls back to a synthetic dataset when no API keys are configured.
2. **Programmatic Labeling** — 16 labeling functions encode financial domain heuristics (keyword patterns, emoji signals, options language, sarcasm detection). A confidence-weighted aggregator combines their votes into a single label per post.
3. **Training** — TF-IDF + Logistic Regression trained on programmatically labeled data. Minimal preprocessing preserves emojis, tickers, and punctuation as features.
4. **Ticker Extraction** — Rule-based extractor identifies cashtags, bare tickers, company names, and informal aliases (e.g., "Elon" maps to Tesla).
5. **Dashboard** — Streamlit app with per-ticker sentiment cards, evidence posts, live inference, and model diagnostics.

## Live Data Sources

Add credentials to `.env` to enable real data ingestion:

```bash
cp .env.example .env
# Fill in your API keys
```

| Source | What you need |
|--------|--------------|
| Reddit | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` — [create app](https://www.reddit.com/prefs/apps) |
| Stocktwits | `STOCKTWITS_ACCESS_TOKEN` — [developer portal](https://api.stocktwits.com/developers) |
| News | `NEWS_API_KEY` — [newsapi.org](https://newsapi.org/) |

When running in `auto` mode (default), the pipeline uses all available sources and falls back to synthetic data if none are configured.

## CLI Commands

```bash
make setup      # Install dependencies
make ingest     # Ingest data (live or synthetic)
make label      # Run labeling functions
make train      # Train sentiment model
make evaluate   # Evaluate model performance
make pipeline   # Run full pipeline end-to-end
make run        # Launch Streamlit dashboard
make test       # Run test suite
make clean      # Remove generated data and models
```

## Project Structure

```
MarketPulse/
├── app/                        # Streamlit dashboard
│   ├── streamlit_app.py        # Home page
│   ├── pipeline_runner.py      # Cached pipeline for all pages
│   ├── pages/                  # Dashboard pages
│   └── components/             # Reusable charts, metrics, styles
├── src/
│   ├── ingestion/              # Data sources (Reddit, Stocktwits, News, Synthetic)
│   ├── labeling/               # Programmatic labeling functions + aggregator
│   ├── models/                 # TF-IDF + LogReg training pipeline
│   ├── extraction/             # Ticker entity extraction + normalization
│   ├── analysis/               # Per-ticker sentiment aggregation
│   └── evaluation/             # Classification + extraction metrics
├── scripts/                    # CLI entry points
├── config/                     # YAML configuration
├── data/                       # Raw, labeled, gold, synthetic, models
└── tests/                      # Test suite
```

## Tech Stack

Python 3.9+ | Streamlit | scikit-learn | Plotly | PRAW | pandas
