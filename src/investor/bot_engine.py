"""Autonomous scalp trading bot engine — probability-based, not prediction-based.

Think like a casino, not a gambler. The edge comes from:
- Expected Value (EV): only take trades where math favors us
- Kelly Criterion: size positions based on actual win/loss statistics
- Risk of Ruin: never let position sizing threaten account survival
- Law of Large Numbers: many small trades > few big bets
- Variance awareness: drawdowns are normal, don't abandon edge during them

Module-level BotState singleton. Background daemon thread runs continuous
cycles. All trade execution goes through MCP client wrappers.
"""
import logging
import math
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.investor.mcp_client import (
    create_portfolio,
    analyze_portfolio,
    execute_buy,
    execute_sell,
    detect_market_regime,
    get_vix_analysis,
    scan_universe,
    scan_anomalies,
    scan_volume_leaders,
    analyze_ticker,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CYCLE_INTERVAL = 5            # seconds between cycles
MAX_POSITIONS = 20
MIN_SCORE = 60
EXIT_SCORE_DROP_THRESHOLD = 0.30
EXIT_ABSOLUTE_THRESHOLD = 40
STARTING_CAPITAL = 10_000.0

# Risk management — casino rules
MAX_RISK_PER_TRADE = 0.02     # never risk more than 2% of portfolio per trade
MIN_TRADES_FOR_STATS = 10     # minimum closed trades before using Kelly sizing
DEFAULT_RISK_PER_TRADE = 0.01 # conservative 1% until we have enough data
MAX_KELLY_FRACTION = 0.5      # use half-Kelly (full Kelly is too aggressive)
MAX_ACCEPTABLE_RUIN = 0.05    # 5% risk of ruin = reduce sizing
VARIANCE_LOOKBACK = 50        # last N trades for rolling stats

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
    "UNH", "XOM", "LLY", "MA", "HD", "PG", "COST", "JNJ", "MRK", "ABBV",
    "CVX", "BAC", "WMT", "KO", "NFLX", "PEP", "DIS", "ADBE", "AMD", "INTC",
    "CRM", "PYPL", "QCOM", "TXN", "CSCO", "NEE", "RTX", "CAT", "GS", "AMGN",
    "T", "VZ", "PFE", "BMY", "MO", "SBUX", "DE", "MMM", "GE", "F",
]


# ---------------------------------------------------------------------------
# Trade Statistics — the math that matters
# ---------------------------------------------------------------------------

@dataclass
class TradeStats:
    """Rolling statistics computed from closed trades. Updated each cycle."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expected_value: float = 0.0     # EV per trade in dollars
    kelly_fraction: float = 0.0     # optimal fraction of capital to risk
    risk_of_ruin: float = 0.0       # probability of total account loss
    std_dev: float = 0.0            # standard deviation of trade P&L
    reward_risk_ratio: float = 0.0  # avg_win / avg_loss
    largest_win: float = 0.0
    largest_loss: float = 0.0
    current_streak: int = 0         # positive = win streak, negative = loss streak
    has_edge: bool = False          # True if EV > 0 with enough sample size


def _compute_trade_stats(trade_log: list) -> TradeStats:
    """Compute all trading statistics from closed trades (SELL entries in log).

    This is the foundation of the entire system. Every sizing and risk
    decision flows from these numbers.
    """
    stats = TradeStats()
    closed = [t for t in trade_log if t["action"] == "SELL"]
    if not closed:
        return stats

    pnls = [t["pnl"] for t in closed]
    stats.total_trades = len(pnls)
    stats.wins = sum(1 for p in pnls if p > 0)
    stats.losses = sum(1 for p in pnls if p <= 0)

    # Win rate
    stats.win_rate = stats.wins / stats.total_trades if stats.total_trades > 0 else 0

    # Average win and average loss (absolute value for loss)
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [abs(p) for p in pnls if p <= 0]
    stats.avg_win = statistics.mean(win_pnls) if win_pnls else 0
    stats.avg_loss = statistics.mean(loss_pnls) if loss_pnls else 0

    # Expected Value: EV = (WinRate × AvgWin) - (LossRate × AvgLoss)
    loss_rate = 1 - stats.win_rate
    stats.expected_value = (stats.win_rate * stats.avg_win) - (loss_rate * stats.avg_loss)

    # Reward/Risk ratio
    stats.reward_risk_ratio = stats.avg_win / stats.avg_loss if stats.avg_loss > 0 else 0

    # Standard deviation of all trade P&Ls
    if len(pnls) >= 2:
        stats.std_dev = statistics.stdev(pnls)

    # Extremes
    stats.largest_win = max(pnls) if pnls else 0
    stats.largest_loss = min(pnls) if pnls else 0

    # Current streak
    streak = 0
    for t in closed:  # most recent first (list is insert(0, ...))
        if streak == 0:
            streak = 1 if t["pnl"] > 0 else -1
        elif (streak > 0 and t["pnl"] > 0) or (streak < 0 and t["pnl"] <= 0):
            streak += 1 if streak > 0 else -1
        else:
            break
    stats.current_streak = streak

    # Kelly Criterion: Kelly% = W - (1-W)/R
    # where W = win rate, R = reward/risk ratio
    if stats.reward_risk_ratio > 0 and stats.total_trades >= MIN_TRADES_FOR_STATS:
        kelly = stats.win_rate - (1 - stats.win_rate) / stats.reward_risk_ratio
        stats.kelly_fraction = max(0, kelly * MAX_KELLY_FRACTION)  # half-Kelly, floor at 0

    # Risk of Ruin: RoR = ((1-W)/W × 1/R) ^ (Account/Risk)
    # Only meaningful with enough trades
    if (stats.win_rate > 0 and stats.reward_risk_ratio > 0
            and stats.total_trades >= MIN_TRADES_FOR_STATS):
        base = ((1 - stats.win_rate) / stats.win_rate) * (1 / stats.reward_risk_ratio)
        if 0 < base < 1:
            # Use current position sizing to estimate units at risk
            risk_per_trade = stats.kelly_fraction if stats.kelly_fraction > 0 else DEFAULT_RISK_PER_TRADE
            units = 1 / risk_per_trade if risk_per_trade > 0 else 100
            stats.risk_of_ruin = base ** units
        elif base >= 1:
            stats.risk_of_ruin = 1.0  # no edge, guaranteed ruin

    # Do we have a statistically meaningful edge?
    stats.has_edge = (
        stats.expected_value > 0
        and stats.total_trades >= MIN_TRADES_FOR_STATS
    )

    return stats


# ---------------------------------------------------------------------------
# Position Sizing — Kelly-based with safety rails
# ---------------------------------------------------------------------------

def _compute_position_size(score: float, high_vix: bool, stats: TradeStats,
                           portfolio_value: float) -> float:
    """Calculate dollar amount to risk on this trade.

    Sizing hierarchy:
    1. If enough trade history: use half-Kelly from actual statistics
    2. If not enough history: use conservative 1% of portfolio
    3. Apply VIX discount if markets are volatile
    4. Cap at 2% of portfolio (professional standard)
    5. Scale by score conviction (higher score = closer to full size)

    Returns dollar amount to allocate (not percentage).
    """
    if portfolio_value <= 0:
        return 0

    # Base risk fraction
    if stats.has_edge and stats.kelly_fraction > 0:
        # Kelly-based: we have enough data to size intelligently
        base_risk = stats.kelly_fraction
        logger.debug("Using Kelly sizing: %.4f", base_risk)
    else:
        # Not enough data yet — be conservative, many small bets
        base_risk = DEFAULT_RISK_PER_TRADE
        logger.debug("Using default conservative sizing: %.4f", base_risk)

    # Risk of ruin check — if we're in danger zone, reduce sizing
    if stats.risk_of_ruin > MAX_ACCEPTABLE_RUIN:
        base_risk *= 0.5
        logger.warning("Risk of ruin %.2f%% > %.0f%% — halving position size",
                        stats.risk_of_ruin * 100, MAX_ACCEPTABLE_RUIN * 100)

    # VIX discount — high volatility means smaller bets
    if high_vix:
        base_risk *= 0.5

    # Score conviction scaling: score 60 = 60% of base size, score 100 = 100%
    conviction = score / 100.0
    adjusted_risk = base_risk * conviction

    # Hard cap at 2% per trade — never more, regardless of Kelly
    capped_risk = min(adjusted_risk, MAX_RISK_PER_TRADE)

    return portfolio_value * capped_risk


# ---------------------------------------------------------------------------
# BotState
# ---------------------------------------------------------------------------

@dataclass
class BotState:
    is_running: bool = False
    portfolio_id: Optional[str] = None
    portfolio_cash: float = STARTING_CAPITAL
    portfolio_value: float = STARTING_CAPITAL
    total_pnl: float = 0.0
    open_positions: dict = field(default_factory=dict)
    pending_sells: set = field(default_factory=set)
    trade_log: list = field(default_factory=list)
    cycle_count: int = 0
    last_cycle_time: Optional[datetime] = None
    # Quant stats — updated every cycle
    stats: TradeStats = field(default_factory=TradeStats)


_state = BotState()
_lock = threading.Lock()


def get_state() -> BotState:
    """Return the global BotState singleton."""
    return _state


def _get_composite_score(analysis: dict) -> float:
    """Extract composite score from analyze_ticker response. Returns 0.0 on any error."""
    if "error" in analysis:
        return 0.0
    raw = analysis.get("score", {}).get("score")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Cycle step functions
# ---------------------------------------------------------------------------

def _check_vix() -> bool:
    """Returns True if VIX is numeric and > 30."""
    try:
        data = get_vix_analysis()
        vix = data.get("vix")
        return isinstance(vix, (int, float)) and not isinstance(vix, bool) and vix > 30
    except Exception as e:
        logger.warning("VIX check failed: %s", e)
        return False


def _sell_position(portfolio_id: str, ticker: str, pos: dict, reason: str) -> bool:
    """Execute a sell and update state. Returns True if successful."""
    current_price = pos.get("current_price", pos["entry_price"])
    result = execute_sell(portfolio_id, ticker, pos["shares"])
    pnl = round((current_price - pos["entry_price"]) * pos["shares"], 2)

    if "error" not in result:
        with _lock:
            _state.open_positions.pop(ticker, None)
            _state.trade_log.insert(0, {
                "time": datetime.now().strftime("%H:%M"),
                "action": "SELL",
                "ticker": ticker,
                "price": current_price,
                "shares": pos["shares"],
                "score": pos.get("current_score", 0),
                "reason": reason,
                "pnl": pnl,
            })
        logger.info("Sold %s: %s (P&L: $%.2f)", ticker, reason, pnl)
        return True
    else:
        with _lock:
            _state.pending_sells.add(ticker)
        logger.warning("Sell failed for %s → pending: %s", ticker, result["error"])
        return False


def _check_exits(portfolio_id: str, stop_event: threading.Event) -> None:
    """Smart exit engine — probability-aware selling.

    Exit triggers:
    1. SIGNAL REVERSAL: score dropped 30%+ or below absolute threshold
    2. PROFIT TAKING: score fading 15%+ while position is green
    3. MOMENTUM STALL: held 10+ cycles with no score improvement
    4. OUTLIER LOSS: loss exceeds 2 standard deviations (cut the outlier)
    """
    with _lock:
        positions = dict(_state.open_positions)
        current_stats = _state.stats

    for ticker, pos in positions.items():
        if stop_event.is_set():
            return
        analysis = analyze_ticker(ticker)
        current_score = _get_composite_score(analysis)
        current_price = (
            analysis.get("price") or pos["entry_price"]
            if "error" not in analysis
            else pos["entry_price"]
        )

        with _lock:
            if ticker in _state.open_positions:
                _state.open_positions[ticker]["current_price"] = current_price
                _state.open_positions[ticker]["current_score"] = current_score

        if current_score == 0:
            continue

        entry_score = pos["entry_score"]
        entry_price = pos["entry_price"]
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
        pnl_dollar = (current_price - entry_price) * pos["shares"]
        score_change_pct = (current_score - entry_score) / entry_score if entry_score > 0 else 0

        # Exit 1: Signal reversal
        if (current_score < entry_score * (1 - EXIT_SCORE_DROP_THRESHOLD)
                or current_score < EXIT_ABSOLUTE_THRESHOLD):
            reason = (
                f"signal reversal {entry_score:.0f}→{current_score:.0f}"
                if current_score >= EXIT_ABSOLUTE_THRESHOLD
                else f"below threshold ({current_score:.0f})"
            )
            _sell_position(portfolio_id, ticker, pos, reason)
            continue

        # Exit 2: Profit taking — lock gains before they evaporate
        if pnl_pct > 0.005 and score_change_pct < -0.15:
            reason = f"take profit +{pnl_pct*100:.1f}% (score fading {entry_score:.0f}→{current_score:.0f})"
            _sell_position(portfolio_id, ticker, pos, reason)
            continue

        # Exit 3: Momentum stall
        cycles_held = _state.cycle_count - pos.get("entry_cycle", _state.cycle_count)
        if cycles_held >= 10 and score_change_pct <= 0:
            reason = f"stale {cycles_held} cycles (score {entry_score:.0f}→{current_score:.0f})"
            _sell_position(portfolio_id, ticker, pos, reason)
            continue

        # Exit 4: Outlier loss — if unrealized loss > 2 std deviations, cut it
        # This is variance management: don't let one bad trade destroy the account
        if current_stats.std_dev > 0 and current_stats.total_trades >= MIN_TRADES_FOR_STATS:
            mean_pnl = current_stats.expected_value
            if pnl_dollar < mean_pnl - 2 * current_stats.std_dev:
                reason = f"outlier loss ${pnl_dollar:.2f} > 2σ (cut variance)"
                _sell_position(portfolio_id, ticker, pos, reason)


def _retry_pending_sells(portfolio_id: str, stop_event: threading.Event) -> None:
    """Retry execute_sell for any tickers in pending_sells."""
    with _lock:
        pending = set(_state.pending_sells)
    for ticker in pending:
        if stop_event.is_set():
            return
        with _lock:
            pos = _state.open_positions.get(ticker)
        if pos is None:
            with _lock:
                _state.pending_sells.discard(ticker)
            continue
        result = execute_sell(portfolio_id, ticker, pos["shares"])
        if "error" not in result:
            current_price = pos.get("current_price", pos["entry_price"])
            pnl = round((current_price - pos["entry_price"]) * pos["shares"], 2)
            with _lock:
                _state.pending_sells.discard(ticker)
                _state.open_positions.pop(ticker, None)
                _state.trade_log.insert(0, {
                    "time": datetime.now().strftime("%H:%M"),
                    "action": "SELL",
                    "ticker": ticker,
                    "price": current_price,
                    "shares": pos["shares"],
                    "score": pos.get("current_score", 0),
                    "reason": "pending retry",
                    "pnl": pnl,
                })


def _scan_candidates(stop_event: threading.Event) -> list:
    """Scan universe + anomalies + volume leaders. Returns deduped candidate symbols."""
    raw: list = []

    universe_result = scan_universe(DEFAULT_UNIVERSE)
    if "error" not in universe_result:
        for item in universe_result.get("scores", []):
            sym = item.get("symbol", "")
            if sym:
                raw.append(sym)

    if stop_event.is_set():
        return []

    anomaly_result = scan_anomalies(DEFAULT_UNIVERSE)
    if "error" not in anomaly_result:
        for item in anomaly_result.get("anomalies", []):
            sym = item.get("symbol", "")
            if sym:
                raw.append(sym)

    if stop_event.is_set():
        return []

    volume_result = scan_volume_leaders(DEFAULT_UNIVERSE)
    if "error" not in volume_result:
        for item in volume_result.get("leaders", []):
            sym = item.get("symbol", "")
            if sym:
                raw.append(sym)

    with _lock:
        held = set(_state.open_positions.keys()) | _state.pending_sells

    seen: set = set()
    candidates = []
    for sym in raw:
        if sym not in held and sym not in seen:
            seen.add(sym)
            candidates.append(sym)
    return candidates


def _score_candidates(candidates: list, stop_event: threading.Event) -> list:
    """Score top-10 candidates. Returns [{ticker, score, price}] sorted score desc."""
    scored = []
    for sym in candidates[:10]:
        if stop_event.is_set():
            return scored
        analysis = analyze_ticker(sym)
        score = _get_composite_score(analysis)
        if score < MIN_SCORE:
            continue
        price = analysis.get("price") if "error" not in analysis else None
        if price and float(price) > 0:
            scored.append({"ticker": sym, "score": score, "price": float(price)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _enter_positions(
    portfolio_id: str,
    scored: list,
    high_vix: bool,
    stop_event: threading.Event,
) -> None:
    """Enter positions using Kelly-based sizing with rotation.

    Casino mindset: many small bets sized by mathematical edge.
    Position size comes from _compute_position_size (Kelly + safety rails),
    not fixed percentage tiers.
    """
    with _lock:
        remaining_cash = _state.portfolio_cash
        portfolio_value = _state.portfolio_value
        current_stats = _state.stats

    for candidate in scored:
        if stop_event.is_set():
            return

        with _lock:
            at_max = len(_state.open_positions) >= MAX_POSITIONS
            weakest_ticker = None
            weakest_score = float("inf")
            if at_max and _state.open_positions:
                for t, p in _state.open_positions.items():
                    s = p.get("current_score", p["entry_score"])
                    if s < weakest_score:
                        weakest_score = s
                        weakest_ticker = t

        # Rotation: only swap if candidate is meaningfully better
        if at_max:
            if weakest_ticker is None:
                break
            if candidate["score"] <= weakest_score + 10:
                continue
            with _lock:
                weak_pos = _state.open_positions.get(weakest_ticker)
            if weak_pos:
                reason = f"rotated for {candidate['ticker']} ({weakest_score:.0f}→{candidate['score']:.0f})"
                sold = _sell_position(portfolio_id, weakest_ticker, weak_pos, reason)
                if not sold:
                    continue
                with _lock:
                    remaining_cash = _state.portfolio_cash

        # Kelly-based position sizing
        dollar_amount = _compute_position_size(
            candidate["score"], high_vix, current_stats, portfolio_value
        )
        if dollar_amount <= 0:
            continue

        shares = int(dollar_amount / candidate["price"])
        if shares < 1:
            continue
        # Don't spend more than available cash
        cost = shares * candidate["price"]
        if cost > remaining_cash:
            shares = int(remaining_cash / candidate["price"])
            if shares < 1:
                continue

        result = execute_buy(portfolio_id, candidate["ticker"], shares)
        if "error" not in result:
            actual_cost = shares * candidate["price"]
            risk_pct = actual_cost / portfolio_value * 100 if portfolio_value > 0 else 0
            with _lock:
                _state.open_positions[candidate["ticker"]] = {
                    "entry_price": candidate["price"],
                    "shares": shares,
                    "entry_score": candidate["score"],
                    "entry_time": datetime.now(),
                    "entry_cycle": _state.cycle_count,
                    "current_price": candidate["price"],
                    "current_score": candidate["score"],
                }
                _state.trade_log.insert(0, {
                    "time": datetime.now().strftime("%H:%M"),
                    "action": "BUY",
                    "ticker": candidate["ticker"],
                    "price": candidate["price"],
                    "shares": shares,
                    "score": candidate["score"],
                    "reason": f"score {candidate['score']:.0f} · {risk_pct:.1f}% risk",
                    "pnl": 0.0,
                })
            remaining_cash -= actual_cost
            logger.info(
                "Bought %s: %d shares @ $%.2f (score %.0f, %.1f%% of portfolio)",
                candidate["ticker"], shares, candidate["price"],
                candidate["score"], risk_pct,
            )
        else:
            logger.warning("Buy failed for %s: %s", candidate["ticker"], result["error"])


def _snapshot_portfolio(portfolio_id: str) -> None:
    """Reconcile BotState cash/value from MCP server and recompute trade stats."""
    result = analyze_portfolio(portfolio_id)
    if "error" not in result:
        portfolio_info = result.get("portfolio", {})
        with _lock:
            _state.portfolio_cash = portfolio_info.get("current_cash", _state.portfolio_cash)
            _state.portfolio_value = result.get("total_value", _state.portfolio_value)
            _state.total_pnl = _state.portfolio_value - STARTING_CAPITAL
    else:
        logger.warning("Portfolio snapshot failed: %s", result["error"])

    # Recompute trade statistics every cycle
    with _lock:
        _state.stats = _compute_trade_stats(_state.trade_log)
        s = _state.stats
    if s.total_trades > 0:
        logger.info(
            "Stats: %d trades, %.0f%% win rate, EV=$%.2f, Kelly=%.2f%%, RoR=%.4f%%, R:R=%.2f",
            s.total_trades, s.win_rate * 100, s.expected_value,
            s.kelly_fraction * 100, s.risk_of_ruin * 100, s.reward_risk_ratio,
        )


def _run_cycle(portfolio_id: str, stop_event: threading.Event) -> None:
    """Execute one full trading cycle."""
    logger.info("=== Cycle %d start ===", _state.cycle_count)

    regime = detect_market_regime()
    if "error" not in regime:
        logger.info("Market regime: %s (score %s)", regime.get("regime"), regime.get("score"))

    high_vix = _check_vix()
    _retry_pending_sells(portfolio_id, stop_event)
    if stop_event.is_set():
        return
    _check_exits(portfolio_id, stop_event)
    if stop_event.is_set():
        return
    candidates = _scan_candidates(stop_event)
    if candidates:
        scored = _score_candidates(candidates, stop_event)
        if not stop_event.is_set():
            _enter_positions(portfolio_id, scored, high_vix, stop_event)
    else:
        logger.warning("No candidates — skipping entry phase")
    _snapshot_portfolio(portfolio_id)


# ---------------------------------------------------------------------------
# BotEngine — controls the background loop
# ---------------------------------------------------------------------------

class BotEngine:
    """Start/stop the background trading thread."""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        with _lock:
            if _state.is_running:
                return

        if _state.portfolio_id is None:
            result = create_portfolio(
                starting_capital=STARTING_CAPITAL,
                risk_profile="aggressive",
                investment_horizon="short",
                name="ScalpBot",
            )
            if "error" not in result:
                with _lock:
                    _state.portfolio_id = result["portfolio_id"]
                logger.info("Created ScalpBot portfolio: %s", _state.portfolio_id)
            else:
                logger.error("Portfolio creation failed: %s", result["error"])
                return

        self._stop_event.clear()
        with _lock:
            _state.is_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Bot started")

    def stop(self) -> None:
        self._stop_event.set()
        with _lock:
            _state.is_running = False
        logger.info("Bot stop signaled")

    def is_running(self) -> bool:
        return (
            _state.is_running
            and self._thread is not None
            and self._thread.is_alive()
        )

    def _loop(self) -> None:
        with _lock:
            portfolio_id = _state.portfolio_id

        while not self._stop_event.is_set():
            with _lock:
                _state.cycle_count += 1
                _state.last_cycle_time = datetime.now()
            try:
                _run_cycle(portfolio_id, self._stop_event)
            except Exception as e:
                logger.error("Cycle error: %s", e, exc_info=True)
            for _ in range(CYCLE_INTERVAL):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

        with _lock:
            _state.is_running = False
        logger.info("Bot loop exited")


_engine = BotEngine()


def get_engine() -> BotEngine:
    """Return the global BotEngine singleton."""
    return _engine
