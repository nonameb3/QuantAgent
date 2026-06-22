"""
QuantAgent signal bot scheduler.

Runs a TradingGraph analysis for each configured (symbol, interval) pair
on a recurring schedule and routes results through the filter → store →
Telegram pipeline.

Usage:
    python scheduler.py

Required env vars (set in .env):
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
    OPENAI_API_KEY          (or ANTHROPIC_API_KEY / DASHSCOPE_API_KEY)

Optional env vars:
    SCHEDULER_SYMBOLS       comma-separated, e.g. "BTC,ETH,AAPL"  (default: BTC)
    SCHEDULER_INTERVAL      candle timeframe, e.g. "15m"           (default: 15m)
    SCHEDULER_CRON          cron expression for when to run        (default: every 15min)
    SIGNAL_COOLDOWN_MINUTES minimum minutes between signals        (default: 60)
    HEALTH_CHECK_MINUTES    silence threshold before alert fires   (default: 120)
"""

from __future__ import annotations

import json
import logging
import os
import signal as _signal
import sys
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv

from data_feed import YFinanceFeed
from decision_agent import _parse_llm_json, _validate_decision
from health import HealthMonitor
from signal_filter import SignalFilter
from signal_store import Signal, SignalStore
from telegram_notifier import TelegramNotifier
from trading_graph import TradingGraph
import static_util

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("quantagent.scheduler")

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

SYMBOLS          = [s.strip() for s in os.environ.get("SCHEDULER_SYMBOLS", "BTC").split(",")]
INTERVAL         = os.environ.get("SCHEDULER_INTERVAL", "15m")
COOLDOWN_MINUTES = int(os.environ.get("SIGNAL_COOLDOWN_MINUTES", "60"))
HEALTH_MINUTES   = int(os.environ.get("HEALTH_CHECK_MINUTES", "120"))

# Map interval string to APScheduler minutes
_INTERVAL_TO_MINUTES: dict[str, int] = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440, "1w": 10080,
}


# ---------------------------------------------------------------------------
# Core analysis + dispatch
# ---------------------------------------------------------------------------

def run_analysis(
    symbol: str,
    interval: str,
    graph: TradingGraph,
    feed: YFinanceFeed,
    store: SignalStore,
    filt: SignalFilter,
    notifier: TelegramNotifier,
    health: HealthMonitor,
):
    log.info("Running analysis — symbol=%s interval=%s", symbol, interval)

    # 1. Fetch candles
    df = feed.get_klines(symbol, interval, limit=45)
    if df.empty:
        log.warning("No data returned for %s/%s — skipping", symbol, interval)
        return

    # 2. Build initial state
    df_slice = df.tail(45).reset_index(drop=True)
    df_dict: dict = {}
    for col in ["Datetime", "Open", "High", "Low", "Close"]:
        if col == "Datetime":
            df_dict[col] = df_slice[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        else:
            df_dict[col] = df_slice[col].tolist()

    p_image = static_util.generate_kline_image(df_dict)
    t_image = static_util.generate_trend_image(df_dict)

    initial_state = {
        "kline_data":       df_dict,
        "analysis_results": None,
        "messages":         [],
        "time_frame":       interval,
        "stock_name":       symbol,
        "pattern_image":    p_image["pattern_image"],
        "trend_image":      t_image["trend_image"],
        "agent_errors":     {},
        "confidence_scores": {},
        "signal_valid":     True,
    }

    # 3. Run the graph
    try:
        final_state = graph.graph.invoke(initial_state)
    except Exception as exc:
        log.error("Graph invocation failed for %s/%s: %s", symbol, interval, exc)
        notifier.send_text(f"⚠️ *QuantAgent error*\n`{symbol}/{interval}` graph failed:\n`{exc}`")
        return

    # 4. Parse the decision
    raw = final_state.get("final_trade_decision", "")
    try:
        parsed    = _parse_llm_json(raw)
        validated = _validate_decision(parsed)
    except (ValueError, KeyError) as exc:
        log.error("Decision parse failed for %s/%s: %s | raw=%r", symbol, interval, exc, raw[:200])
        return

    decision = validated["decision"]
    log.info("Decision for %s/%s: %s  R:R=%.2f", symbol, interval, decision, validated["risk_reward_ratio"])

    # 5. Filter (cooldown / dedup)
    allow, reason = filt.should_send(symbol, decision)
    if not allow:
        log.info("Signal filtered for %s: %s", symbol, reason)
        return

    log.info("Signal passed filter for %s: %s", symbol, reason)

    # 6. Persist
    sig = Signal(
        symbol           = symbol,
        interval         = interval,
        decision         = decision,
        justification    = validated["justification"],
        risk_reward_ratio= validated["risk_reward_ratio"],
        forecast_horizon = validated["forecast_horizon"],
    )
    sig = store.save(sig)

    # 7. Send to Telegram
    sent = notifier.send(sig)
    if sent and sig.id:
        store.mark_sent(sig.id)
        log.info("Signal sent to Telegram — id=%s symbol=%s decision=%s", sig.id, symbol, decision)
    else:
        log.warning("Telegram send failed for signal id=%s", sig.id)

    # 8. Update heartbeat
    health.heartbeat()


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def build_scheduler() -> BlockingScheduler:
    graph    = TradingGraph()
    feed     = YFinanceFeed()
    store    = SignalStore()
    filt     = SignalFilter(store, cooldown_minutes=COOLDOWN_MINUTES)
    notifier = TelegramNotifier()
    health   = HealthMonitor(notifier, silence_threshold_minutes=HEALTH_MINUTES)

    interval_minutes = _INTERVAL_TO_MINUTES.get(INTERVAL, 15)
    scheduler = BlockingScheduler(timezone="UTC")

    for sym in SYMBOLS:
        scheduler.add_job(
            run_analysis,
            trigger=IntervalTrigger(minutes=interval_minutes),
            kwargs=dict(
                symbol=sym, interval=INTERVAL,
                graph=graph, feed=feed, store=store,
                filt=filt, notifier=notifier, health=health,
            ),
            id=f"analysis_{sym}_{INTERVAL}",
            name=f"QuantAgent {sym}/{INTERVAL}",
            max_instances=1,
            misfire_grace_time=30,
        )
        log.info("Scheduled %s/%s every %d minutes", sym, INTERVAL, interval_minutes)

    # Dead man's switch — fires every 30 min regardless of analysis cadence
    scheduler.add_job(
        health.check,
        trigger=IntervalTrigger(minutes=30),
        id="health_check",
        name="Health check",
    )

    return scheduler


def main():
    log.info("Starting QuantAgent scheduler — symbols=%s interval=%s", SYMBOLS, INTERVAL)

    scheduler = build_scheduler()

    # Graceful shutdown on SIGINT / SIGTERM
    def _shutdown(signum, frame):
        log.info("Shutdown signal received — stopping scheduler")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    _signal.signal(_signal.SIGINT,  _shutdown)
    _signal.signal(_signal.SIGTERM, _shutdown)

    scheduler.start()


if __name__ == "__main__":
    main()
