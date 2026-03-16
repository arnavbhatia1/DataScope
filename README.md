# MarketPulse

Financial sentiment dashboard + autonomous paper-trading bot. Two pages:

1. **MarketPulse** — scours free RSS news feeds, analyzes sentiment across 60 tickers, shows top 50 by mentions with AI-powered verdicts
2. **Trading Bot** — autonomous scalp bot powered by [financial-mcp-server](https://github.com/arnavbhat1/financial-mcp), makes buy/sell decisions using probability math (Expected Value, Kelly Criterion, Risk of Ruin)

No API keys required. Works out of the box.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env       # optionally add ANTHROPIC_API_KEY for AI verdicts
start.bat                  # starts MCP server + Streamlit on localhost:8501
```

## What You Get

**MarketPulse page:**
- Top 50 tickers by news mention count, auto-discovered from RSS
- Click any ticker → dialog popup with sentiment breakdown, AI verdict, 7-day trend, news headlines
- Search bar for any ticker → inline briefing card

**Trading Bot page:**
- Market regime detection (BULL/BEAR/SIDEWAYS) + VIX analysis
- Ticker analysis with candlestick charts, fundamentals, momentum, smart money signals
- Autonomous bot: click Start → it continuously scans, buys, sells, rotates positions
- Live dashboard updating every second: portfolio value, P&L, positions, activity log
- Edge Statistics panel: EV/trade, win rate, R:R ratio, Kelly %, risk of ruin

## Architecture

```
Free News RSS (Google News + Yahoo Finance + CNBC + MarketWatch)
        ↓
16 labeling functions → sentiment classification → SQLite
        ↓
MarketPulse: top 50 grid + search + AI verdict (Claude)
        ↓
Trading Bot: MCP client → financial-mcp-server (market data + paper trading)
        ↓
Bot Engine: Kelly Criterion sizing, EV tracking, smart exits, position rotation
```

**Data separation:** MarketPulse uses only RSS/SQLite. Trading Bot uses only the MCP server. They don't share data sources.

## Trading Bot Math

The bot thinks like a casino, not a gambler:

- **Expected Value:** `EV = (WinRate × AvgWin) - (LossRate × AvgLoss)` — only sizes up when EV is positive
- **Kelly Criterion:** half-Kelly position sizing from actual trade statistics
- **Risk of Ruin:** auto-halves positions if account destruction probability exceeds 5%
- **2% hard cap:** never risks more than 2% of portfolio per trade
- **Conservative bootstrap:** 1% per trade until 10+ closed trades prove the edge
- **4 exit triggers:** signal reversal, profit taking, momentum stall, outlier loss (>2σ)
- **Position rotation:** sells weakest holding when a stronger candidate appears

## API Keys

| Key | Required? | What it does |
|-----|-----------|-------------|
| `ANTHROPIC_API_KEY` | Optional | AI verdict on ticker briefing cards. Falls back to static text without it. |

News RSS is free — no keys needed. The MCP server uses yfinance (free) for market data.

## Tech Stack

Python 3.9+ · Streamlit · scikit-learn · Plotly · Anthropic SDK · SQLite · feedparser · yfinance · financial-mcp-server · pandas
