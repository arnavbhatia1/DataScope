# MarketPulse Product Pivot — Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

## Goal

Transform MarketPulse from a thesis experiment (gold vs programmatic vs noisy labels) into a working sentiment intelligence product that shows per-ticker sentiment breakdowns with evidence posts, powered by the existing ML pipeline.

## Context

The core ML pipeline is complete: ingestion (541 synthetic posts), programmatic labeling (18 functions, 74.5% coverage), TF-IDF + LogReg model (validation F1=0.747), and rule-based ticker extraction (F1=0.980). What's missing is: entity extraction wired into the pipeline, per-ticker aggregation, a Streamlit dashboard, and real API integration.

## Architecture

The existing pipeline (ingest → label → train → extract → evaluate) stays. Two additions connect it to the product:

1. **Entity extraction integrated into pipeline** — every labeled post gets tickers extracted
2. **New analysis module** — aggregates labeled + extracted data into per-ticker sentiment summaries

The Streamlit dashboard consumes the analysis output.

```
ingestion → labeling → extraction → analysis → dashboard
                ↓
            model training (background, for prediction)
```

## Pipeline Changes

### Integrate extraction into pipeline

After `aggregate_batch()`, run `TickerExtractor.extract()` on every post. Store as `tickers` column (list of canonical company names) in `labeled_data.csv`.

### New module: `src/analysis/ticker_sentiment.py`

Takes the labeled + extracted DataFrame and produces per-ticker summaries:

```python
{
    "TSLA": {
        "ticker": "TSLA",
        "company": "Tesla",
        "mention_count": 43,
        "sentiment": {"bullish": 18, "bearish": 12, "neutral": 8, "meme": 5},
        "dominant_sentiment": "bullish",
        "bullish_ratio": 0.42,
        "avg_confidence": 0.72,
        "posts": [
            {"post_id": "...", "text": "...", "sentiment": "bullish",
             "confidence": 0.85, "source": "reddit", "timestamp": "..."}
        ]
    }
}
```

## Dashboard Design

### Page 1 — Market Overview (main page)

- **Top bar:** data source indicator (Synthetic / Live), last refresh time, total posts analyzed
- **Ticker cards grid:** each card shows ticker name, dominant sentiment (color-coded green/red/gray/gold), mention count, confidence bar. Sorted by mention count.
- **Sentiment summary:** aggregate bar chart showing how many tickers are bullish vs bearish vs neutral vs meme
- Click any ticker card → navigates to Ticker Detail

### Page 2 — Ticker Detail

- Selected ticker header (e.g., "TSLA — Tesla")
- Sentiment distribution: horizontal bar or pie showing bullish/bearish/neutral/meme split
- Confidence score and mention count
- **Evidence posts:** scrollable table of every post mentioning this ticker — text snippet, sentiment label (color-coded), confidence, source, timestamp
- Back button to overview

### Page 3 — Live Inference

- Text input: "Paste a post to analyze"
- Output: predicted sentiment + confidence, extracted tickers, probability distribution bar chart
- Batch mode: upload CSV, download with predictions + tickers

### Page 4 — Under the Hood (optional)

- Model metrics (F1, confusion matrix)
- Labeling function performance (coverage, precision per function)
- Feature importance per class
- Data quality stats

### Visual Theme

Dark financial terminal aesthetic using CLAUDE.md color palette:
- Bullish: #00C853 (green)
- Bearish: #FF1744 (red)
- Neutral: #78909C (blue-gray)
- Meme: #FFD600 (gold)
- Background: #0D1117 / #161B22
- Plotly with template='plotly_dark'

## Data Source Toggle

The ingestion manager's existing `mode` config drives this:

- **No API keys (default):** Synthetic mode. Works out of the box for demo.
- **With API keys in `.env`:** Auto mode. Reddit (PRAW) and Stocktwits APIs tried first, synthetic fallback.
- **Sidebar:** shows source status (green/red dots), "Refresh Data" button re-runs pipeline.
- **Reddit:** Real PRAW implementation for r/wallstreetbets, r/stocks, r/investing.
- **Stocktwits:** Real API implementation for configured symbols.
- **News:** Leave as stub (NewsAPI free tier too limited).

## What Gets Removed/Changed

- `src/evaluation/label_quality.py` — thesis experiment stays in code but is no longer the pipeline's focal point. Moved to "Under the Hood" page only.
- `scripts/run_pipeline.py` — updated to include extraction + analysis steps, remove thesis as primary output
- Gold standard comparison becomes optional diagnostic, not the main event

## Success Criteria

1. `streamlit run app/streamlit_app.py` launches and shows ticker sentiment cards
2. Clicking a ticker shows sentiment breakdown + evidence posts
3. Live inference works (paste text → get prediction + tickers)
4. Works out of the box with synthetic data (no API keys required)
5. Reddit/Stocktwits ingestion works when API keys provided
