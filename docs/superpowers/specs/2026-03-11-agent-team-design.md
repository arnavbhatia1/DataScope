# MarketPulse Agent Team — Design Spec

**Date**: 2026-03-11
**Status**: Approved

---

## Overview

A team of 4 AI agents specialized for the MarketPulse project, covering all aspects of development: data acquisition, sentiment/ML, UI/UX, and quality assurance. Agents are collaborative (propose before executing), share a single persistent memory space, and are scoped exclusively to MarketPulse.

## Agent Team

### 1. `mp-ingestion` — Data Acquisition Specialist

**Triggers**: Adding/modifying data sources, fixing ingestion failures, changing ticker extraction or normalization, working with SQLite schema, adding new RSS feeds or API integrations, debugging data quality issues.

**Scope**: `src/ingestion/`, `src/extraction/`, `src/storage/`, `src/utils/`, `config/default.yaml` (ingestion section), `scripts/ingest.py`, `scripts/run_pipeline.py`

**Domain expertise**:
- `BaseIngester` contract and `REQUIRED_COLUMNS` schema
- `IngestionManager` orchestration, deduplication, schema validation
- `TickerExtractor` multi-method pipeline (cashtags, bare tickers, company names, informal refs, emoji)
- `EntityNormalizer` canonical mapping (188+ variations, 40+ companies)
- SQLite schema (posts, ticker_cache, model_training_log)
- RSS via feedparser, PRAW for Reddit, requests for Stocktwits

### 2. `mp-sentiment` — ML & NLP Specialist

**Triggers**: Tuning labeling functions, adjusting confidence thresholds/weights, improving model accuracy, adding new sentiment categories, debugging misclassifications, retraining the model, working on the analysis/aggregation layer.

**Scope**: `src/labeling/`, `src/models/`, `src/analysis/`, `config/default.yaml` (labeling + model sections)

**Domain expertise**:
- All 16 labeling functions and their weights (options_directional at 3.0x, etc.)
- Confidence-weighted voting in `LabelAggregator`
- TF-IDF + LogReg pipeline (500 features, 1-2 ngrams, balanced classes)
- Auto-train trigger (>=200 labeled posts)
- `TickerSentimentAnalyzer` aggregation (per-source, 7-day trend, top posts)

### 3. `mp-dashboard` — UI/UX Specialist

**Triggers**: Modifying Streamlit pages, adding charts/components, changing layout or styling, working on the search flow, modifying `pipeline_runner.py`, improving user experience, adding new pages.

**Scope**: `app/` (all files), `src/agent/briefing.py`

**Domain expertise**:
- Streamlit page structure (MarketPulse.py home, 1_Ticker_Detail.py detail)
- `pipeline_runner.py` caching strategy (`@st.cache_data` 60s, `@st.cache_resource`)
- Plotly chart components (pie, bar, mentions, probability)
- Dark theme color system (bullish=#00C853, bearish=#FF1744, neutral=#78909C, meme=#FFD600)
- Claude briefing integration (claude-sonnet-4-6, max 150 tokens, static fallback)

### 4. `mp-qa` — Quality Gatekeeper

**Triggers**: Running tests, writing new tests, reviewing code from other agents, validating changes before commit, checking for regressions, auditing test coverage.

**Scope**: `tests/` (owns), all source files (read-only for review)

**Domain expertise**:
- 158-test suite structure and fixture system
- Test patterns per module (mocking for API calls, tmp_path for SQLite, synthetic data for model)
- Validation strategies for labeling functions, ingestion schema, model predictions
- Runs `pytest tests/ -v` to verify; never marks work done without green tests

## Shared Memory

All agents share a single memory space at `.claude/agent-memory/shared/`.

```
shared/
├── MEMORY.md              # Index (loaded into every agent's context)
├── known_issues.md        # Recurring bugs, edge cases, gotchas
├── decisions.md           # Architectural decisions and rationale
├── patterns.md            # Confirmed project patterns and conventions
└── ticker_notes.md        # Ticker-specific quirks
```

**Rules**:
- Any agent can read and write
- Check MEMORY.md before starting work
- Record non-obvious discoveries (edge cases, gotchas)
- No session-specific or temporary state
- Keep MEMORY.md under 200 lines

## Behavioral Rules (All Agents)

1. **Propose before executing** — Present approach, wait for approval before writing code
2. **Read before modifying** — Understand existing code before changing it
3. **Check shared memory** — Consult MEMORY.md at task start
4. **Update shared memory** — Record non-obvious patterns, edge cases, gotchas
5. **Stay in your lane** — Only modify files in your scope; flag cross-boundary work
6. **Verify your work** — Run tests or demonstrate correctness; never claim "done" without evidence
7. **Follow CLAUDE.md** — All project conventions and global rules apply

## Cross-Agent Coordination

When work spans multiple agents, the relevant agent proposes its piece and flags what other agents need to do. The user orchestrates handoffs — agents don't call each other directly.

## File Locations

- Agent definitions: `.claude/agents/mp-{ingestion,sentiment,dashboard,qa}.md`
- Shared memory: `.claude/agent-memory/shared/`
- Color coding: ingestion=blue, sentiment=magenta, dashboard=green, qa=yellow
- Model: sonnet (overridable to opus per invocation)
