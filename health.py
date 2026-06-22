"""
Bot heartbeat / dead-man's switch.

Sends a Telegram alert if the scheduler goes silent for longer than
`silence_threshold_minutes`.  Call heartbeat() after every successful
analysis cycle and check() on a separate slow-tick timer (e.g. every 30 min).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from telegram_notifier import TelegramNotifier


class HealthMonitor:
    def __init__(
        self,
        notifier: TelegramNotifier,
        silence_threshold_minutes: int = 120,
    ):
        self.notifier  = notifier
        self.threshold = timedelta(minutes=silence_threshold_minutes)
        self._last_beat: Optional[datetime] = None
        self._alerted   = False

    def heartbeat(self):
        """Call this after every successful analysis run."""
        self._last_beat = datetime.utcnow()
        self._alerted   = False  # reset alert so it fires again if bot goes silent again

    def check(self):
        """
        Call this on a slow timer (e.g. every 30 min).
        Sends a Telegram alert once if the bot has been silent too long.
        """
        if self._last_beat is None:
            return  # bot hasn't run yet — nothing to alert on

        silent_for = datetime.utcnow() - self._last_beat
        if silent_for >= self.threshold and not self._alerted:
            minutes = int(silent_for.total_seconds() / 60)
            self.notifier.send_text(
                f"⚠️ *QuantAgent heartbeat alert*\n\n"
                f"No analysis has run in the last *{minutes} minutes*.\n"
                f"Last heartbeat: `{self._last_beat.strftime('%Y-%m-%d %H:%M UTC')}`\n\n"
                f"Check the scheduler logs."
            )
            self._alerted = True

    def status(self) -> dict:
        if self._last_beat is None:
            return {"ok": False, "last_beat": None, "silent_minutes": None}
        silent = datetime.utcnow() - self._last_beat
        return {
            "ok":             silent < self.threshold,
            "last_beat":      self._last_beat.isoformat(),
            "silent_minutes": int(silent.total_seconds() / 60),
        }
