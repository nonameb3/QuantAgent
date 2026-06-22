"""
Telegram signal notifier.

Formats a Signal into a clean Telegram message and sends it via the Bot API.
Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment or .env.
"""

from __future__ import annotations

import os
from typing import Optional

import requests

from signal_store import Signal


TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

_LONG_EMOJI  = "🟢"
_SHORT_EMOJI = "🔴"
_DIVIDER     = "━━━━━━━━━━━━━━━━"


def _format_message(signal: Signal) -> str:
    direction_emoji = _LONG_EMOJI if signal.decision == "LONG" else _SHORT_EMOJI
    interval_label = signal.interval.upper()

    return (
        f"📊 *{signal.symbol}* | `{interval_label}`\n"
        f"{_DIVIDER}\n"
        f"{direction_emoji} *{signal.decision}*  |  R:R `{signal.risk_reward_ratio:.2f}`\n"
        f"{_DIVIDER}\n"
        f"📅 _{signal.forecast_horizon}_\n\n"
        f"📝 {signal.justification}\n"
        f"{_DIVIDER}\n"
        f"⚙️ _QuantAgent_"
    )


class TelegramNotifier:
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id   = chat_id   or os.environ.get("TELEGRAM_CHAT_ID", "")

        if not self.bot_token:
            raise ValueError(
                "Telegram bot token not found. "
                "Set TELEGRAM_BOT_TOKEN in your .env file."
            )
        if not self.chat_id:
            raise ValueError(
                "Telegram chat ID not found. "
                "Set TELEGRAM_CHAT_ID in your .env file."
            )

    def send(self, signal: Signal) -> bool:
        """
        Send a formatted signal message.
        Returns True on success, False on failure (logs the error).
        """
        text = _format_message(signal)
        url  = TELEGRAM_API.format(token=self.bot_token)

        try:
            resp = requests.post(
                url,
                json={
                    "chat_id":    self.chat_id,
                    "text":       text,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            print(f"[TelegramNotifier] Failed to send signal: {exc}")
            return False

    def send_text(self, text: str) -> bool:
        """Send a plain text message (used by health checks and error alerts)."""
        url = TELEGRAM_API.format(token=self.bot_token)
        try:
            resp = requests.post(
                url,
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            print(f"[TelegramNotifier] Failed to send text: {exc}")
            return False
