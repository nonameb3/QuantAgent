"""
SQLite-backed signal history store.

Records every signal that passes the filter so we can:
- drive cooldown logic in SignalFilter
- track accuracy over time
- audit the bot's output
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


DB_PATH = Path("data/signals.db")


@dataclass
class Signal:
    symbol: str
    interval: str
    decision: str          # "LONG" | "SHORT"
    justification: str
    risk_reward_ratio: float
    forecast_horizon: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    sent_at: Optional[str] = None
    id: Optional[int] = None


class SignalStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol           TEXT    NOT NULL,
                    interval         TEXT    NOT NULL,
                    decision         TEXT    NOT NULL,
                    justification    TEXT    NOT NULL,
                    risk_reward_ratio REAL   NOT NULL,
                    forecast_horizon TEXT    NOT NULL,
                    created_at       TEXT    NOT NULL,
                    sent_at          TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_created ON signals(symbol, created_at)")

    def save(self, signal: Signal) -> Signal:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO signals
                   (symbol, interval, decision, justification, risk_reward_ratio,
                    forecast_horizon, created_at, sent_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    signal.symbol, signal.interval, signal.decision,
                    signal.justification, signal.risk_reward_ratio,
                    signal.forecast_horizon, signal.created_at, signal.sent_at,
                ),
            )
            signal.id = cur.lastrowid
        return signal

    def mark_sent(self, signal_id: int):
        with self._conn() as conn:
            conn.execute(
                "UPDATE signals SET sent_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), signal_id),
            )

    def get_last(self, symbol: str) -> Optional[Signal]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM signals WHERE symbol = ? ORDER BY created_at DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        return Signal(**dict(row)) if row else None

    def get_recent(self, symbol: str, limit: int = 10) -> List[Signal]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM signals WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        return [Signal(**dict(r)) for r in rows]
