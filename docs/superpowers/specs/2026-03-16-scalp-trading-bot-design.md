# Scalp Trading Bot — Design Spec
**Date:** 2026-03-16
**Status:** Approved

---

## Overview

An autonomous paper-trading bot that runs on a 5-minute timer, scans for opportunities using the `financial-mcp-server` tools, enters and exits positions based on score signals, and displays a live dashboard in the Trading Bot Streamlit page.

---

## Architecture

```
BotEngine (src/investor/bot_engine.py)
    ↕ threading.Lock
BotState singleton (module-level, survives Streamlit page navigation)
    ↑ reads/writes
Background thread (5-min cycle loop)
    ↓ MCP tool calls
financial-mcp-server (localhost:8520)
    ↕ portfolio state
financial_mcp.db

Streamlit page (app/pages/2_Trading_Bot.py)
    reads BotState every 10s via session_state timestamp + st.rerun()
    Start/Stop button → BotEngine.start() / .stop()
```

**New files:**
- `src/investor/bot_engine.py` — all trading logic, state, background thread
- `app/pages/2_Trading_Bot.py` — new "Bot Control" section appended to existing page

---

## BotState Singleton

A single module-level instance, created once at import time. Never re-instantiated. Protected by a `threading.Lock` for all reads/writes from the background thread and the Streamlit render thread.

```python
@dataclass
class BotState:
    is_running: bool = False
    portfolio_id: str | None = None          # "ScalpBot" portfolio ID from MCP
    portfolio_cash: float = 10_000.0         # reconciled from analyze_portfolio each cycle
    portfolio_value: float = 10_000.0        # total portfolio value including positions
    total_pnl: float = 0.0
    open_positions: dict = field(default_factory=dict)
    # { ticker: { entry_price, shares, entry_score, entry_time, current_price, current_score } }
    pending_sells: set = field(default_factory=set)  # tickers where execute_sell failed last cycle
    trade_log: list = field(default_factory=list)
    # [ { time, action, ticker, price, shares, score, reason, pnl } ]
    cycle_count: int = 0
    last_cycle_time: datetime | None = None
    next_cycle_time: datetime | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

# Module-level singleton — only instance ever created
_bot_state = BotState()

def get_state() -> BotState:
    return _bot_state
```

---

## Default Scan Universe

`scan_universe(symbols)` requires a symbols list. The bot uses a hardcoded default universe of 50 large-cap, high-liquidity tickers:

```python
DEFAULT_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","JPM","V",
    "UNH","XOM","LLY","MA","HD","PG","COST","JNJ","MRK","ABBV",
    "CVX","BAC","WMT","KO","NFLX","PEP","DIS","ADBE","AMD","INTC",
    "CRM","PYPL","QCOM","TXN","CSCO","NEE","RTX","CAT","GS","AMGN",
    "T","VZ","PFE","BMY","MO","SBUX","DE","MMM","GE","F"
]
```

This list is also used for `scan_anomalies()` and `scan_volume_leaders()` where applicable.

---

## Composite Score Definition

The "composite score" used for all entry/exit decisions is the `score` field nested inside `analyze_ticker(ticker)` response:

```python
analysis = analyze_ticker(ticker)
composite_score = analysis.get("score", {}).get("score", 0)
```

This is the same field used by the existing Ticker Analysis section of the Trading Bot page. A value of 0 is treated as "no data" and the ticker is skipped.

---

## Trading Cycle (every 5 minutes from cycle END)

Steps run in strict order each cycle. Timer starts after the previous cycle finishes (not from cycle start), preventing overlap.

### 1. Market Context
- `detect_market_regime()` → log regime label
- `get_vix_analysis()` → read `vix_data.get("vix", None)`:
  - If value is numeric and > 30: set `high_vix = True` (halve all allocation tiers this cycle)
  - If value is non-numeric ("N/A", error, or missing): set `high_vix = False`, continue normally

### 2. Retry Pending Sells
For each ticker in `pending_sells`:
- Call `execute_sell(portfolio_id, ticker, shares)` where shares is read from `open_positions`
- If success: remove from `pending_sells`, remove from `open_positions`, append to `trade_log`
- If failure again: log warning, leave in `pending_sells` for next cycle

### 3. Exit Check
For each open position:
- Call `analyze_ticker(ticker)` → extract composite score as `current_score`
- Store `current_score` and current price (`analysis.get("price", entry_price)`) back into `open_positions[ticker]`
- **Sell if:** `current_score < entry_score * 0.70` OR `current_score < 40`
- Call `execute_sell(portfolio_id, ticker, shares)` using actual `shares` from `open_positions`
- If success: remove from `open_positions`, append to `trade_log` with P&L = `(current_price - entry_price) * shares`
- If failure: add ticker to `pending_sells`, log warning

### 4. Candidate Scan
- Call `scan_universe(DEFAULT_UNIVERSE)`, `scan_anomalies()`, `scan_volume_leaders()`
- If all three fail: skip entry phase this cycle, log "scan failed"
- Merge results, deduplicate by ticker symbol
- Filter out tickers already in `open_positions` or `pending_sells`

### 5. Score Candidates
- Take top 10 candidates by raw scan ranking
- For each: call `analyze_ticker(ticker)` → extract composite score
- Skip tickers where composite score == 0 (no data)
- Keep only those with composite score ≥ 60
- Sort descending by score

### 6. Position Sizing & Entry
Max 5 concurrent open positions. Process candidates highest score first until positions full or cash exhausted.

**Allocation tiers (applied to `portfolio_cash`):**

| Score Range | Allocation | With high VIX (>30) |
|-------------|-----------|----------------------|
| 90–100      | 12%       | 6%                   |
| 70–89       | 8%        | 4%                   |
| 60–69       | 5%        | 2.5%                 |

**Share count calculation:**
```python
dollar_amount = portfolio_cash * allocation_pct
price_result = get_price(ticker)
price = price_result.get("price", 0)
if price <= 0:
    # skip this ticker, log "could not get price"
    continue
shares = int(dollar_amount / price)   # floor division, whole shares only
if shares < 1:
    continue   # not enough cash for even 1 share
execute_buy(portfolio_id=portfolio_id, symbol=ticker, shares=shares)
```

`execute_buy` signature: `execute_buy(portfolio_id: str, symbol: str, shares: int) -> dict`

If `execute_buy` fails: log failure, do NOT add to `open_positions`.
If success: add to `open_positions` with `entry_price=price`, `shares=shares`, `entry_score=composite_score`, `entry_time=now`.

### 7. Portfolio Snapshot
- Call `analyze_portfolio(portfolio_id)` → update `portfolio_cash`, `portfolio_value`, `total_pnl` from response
- Increment `cycle_count`, set `last_cycle_time = now`
- Sleep 5 minutes, then start next cycle

---

## Portfolio Bootstrap

On first `Start`:
1. `create_portfolio(starting_capital=10000, risk_profile="aggressive", investment_horizon="short", name="ScalpBot")`
   - Actual signature: `create_portfolio(starting_capital, risk_profile, investment_horizon, name="Default")`
2. Store returned `portfolio_id` in `BotState.portfolio_id`
3. On subsequent starts, reuse stored `portfolio_id` (preserved across stop/start within the same Streamlit session)

**Coexistence with manual portfolio (Zone 3):** The bot creates its own separate "ScalpBot" portfolio. Zone 3's manual portfolio management uses a different portfolio ID stored in `data/portfolio_id.txt`. Both portfolios coexist in `financial_mcp.db`. Zone 3 continues to display the manual portfolio; the Bot Control section displays the ScalpBot portfolio.

---

## Stop Behavior

`BotEngine.stop()` sets an `_stop_event` (threading.Event). The background thread checks this event:
- Between MCP calls (after each tool call in the cycle)
- At the start of each new cycle

This is a **graceful shutdown**: the current cycle completes before the thread exits. Partial-cycle state (e.g., a buy was executed but portfolio snapshot not yet taken) is preserved in `BotState` and reconciled at the start of the next cycle when the bot is restarted.

---

## Time-Based Exit

**By design, there is no time-based exit.** The bot holds positions until the signal reversal condition is met. This is intentional per user specification.

---

## Error Handling

| Failure | Behavior |
|---------|----------|
| Single ticker scan error | Skip that ticker, continue |
| All scan tools fail | Skip entry phase; still run exits |
| `get_price()` returns 0 or error | Skip that ticker entry, log |
| `execute_buy` fails | Log, do not add to `open_positions` |
| `execute_sell` fails | Add to `pending_sells`; retry next cycle |
| `analyze_ticker` returns score=0 | Treat as "no data", skip exit check |
| VIX returns non-numeric | Treat as normal VIX, no adjustment |
| MCP disconnected | Log; pause entries; keep retrying pending sells |
| `analyze_portfolio` fails | Keep previous `portfolio_cash` value, log warning |

---

## Dashboard UI (Bot Control Section)

Added to `app/pages/2_Trading_Bot.py` after existing sections:

```
┌─────────────────────────────────────────────────────┐
│  BOT CONTROL                                        │
│  [▶ Start Bot]  Status: RUNNING  Next cycle: 3m 12s │
├──────────────┬──────────────┬───────────────────────┤
│ Portfolio    │ Total P&L    │ Open Positions         │
│ $10,000.00   │ +$142.30     │ 3 / 5 max              │
├──────────────┴──────────────┴───────────────────────┤
│ Open Positions Table                                │
│  Ticker │ Entry │ Current │ P&L% │ Score │ Since    │
│  (current_price from last analyze_ticker call)      │
├─────────────────────────────────────────────────────┤
│ Activity Log (latest 50 entries, newest first)      │
│  [14:32] Cycle 7 — Bought NVDA $820 (score 82, 8%) │
│  [14:27] Cycle 6 — Sold TSLA (score dropped 58→34) │
├─────────────────────────────────────────────────────┤
│ Trade History (all closed trades)                   │
│  Ticker │ Entry │ Exit │ P&L │ Hold time │ Reason   │
└─────────────────────────────────────────────────────┘
```

**Auto-refresh mechanism:** Uses `st.session_state` timestamp to avoid blocking the UI thread:

```python
# At top of Bot Control section render
now = time.time()
last_refresh = st.session_state.get("bot_last_refresh", 0)
if state.is_running and (now - last_refresh) >= 10:
    st.session_state["bot_last_refresh"] = now
    st.rerun()
```

This ensures `st.rerun()` is called at most every 10 seconds without `time.sleep()` blocking the main thread, preserving button interactivity.

---

## Bug Fixes (same PR)

1. **VIX N/A / Score: 0** — diagnose root cause in FinancialMCP server response parsing; add defensive `get()` with numeric type checks throughout the UI
2. **"Could not analyze LUCID"** — add UI hint below ticker input: "Enter a ticker symbol (e.g. LCID, not Lucid Motors)"

---

## Constraints

- **Paper trading only** — all trades go through MCP `execute_buy`/`execute_sell` operating on `financial_mcp.db`
- **No circuit breaker** — bot keeps trading until manually stopped
- **Max 5 concurrent positions**
- **Bot state is in-memory** — restarting the Streamlit server resets `BotState` (positions in `financial_mcp.db` persist, but in-memory log resets)
- **Whole shares only** — fractional shares are not supported; `int(dollar_amount / price)` floors to whole shares
