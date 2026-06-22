"""
Signal deduplication and cooldown filter.

Prevents the bot from spamming the same signal repeatedly.
Rules (all must pass):
  1. Cooldown  — at least `cooldown_minutes` must have passed since the last
                 signal for this symbol, regardless of direction.
  2. Dedup     — the new decision must differ from the last sent decision, OR
                 the cooldown has fully elapsed (same direction is fine after cooldown).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from signal_store import Signal, SignalStore


class SignalFilter:
    def __init__(
        self,
        store: SignalStore,
        cooldown_minutes: int = 60,
    ):
        self.store = store
        self.cooldown = timedelta(minutes=cooldown_minutes)

    def should_send(self, symbol: str, decision: str) -> tuple[bool, str]:
        """
        Returns (allow: bool, reason: str).

        `reason` explains why the signal was blocked (useful for logging).
        """
        last: Optional[Signal] = self.store.get_last(symbol)

        if last is None:
            return True, "first signal for this symbol"

        last_time = datetime.fromisoformat(last.created_at)
        elapsed = datetime.utcnow() - last_time

        if elapsed < self.cooldown:
            remaining = int((self.cooldown - elapsed).total_seconds() / 60)
            return False, (
                f"cooldown active — {remaining}min remaining "
                f"(last: {last.decision} @ {last_time.strftime('%H:%M UTC')})"
            )

        return True, f"cooldown elapsed ({int(elapsed.total_seconds() / 60)}min since last signal)"
