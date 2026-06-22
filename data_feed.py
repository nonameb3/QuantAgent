"""
Abstract data feed interface and implementations.

Swap the data source by passing a different DataFeed subclass to DataAnalyzer
without touching any agent code.  The contract: every implementation returns a
DataFrame with exactly these columns:

    Datetime  (datetime64[ns], tz-naive)
    Open      (float64)
    High      (float64)
    Low       (float64)
    Close     (float64)

Rows are sorted oldest-first.  The caller decides how many rows to keep.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DataFeed(ABC):
    """Minimal interface every data source must implement."""

    REQUIRED_COLUMNS = ["Datetime", "Open", "High", "Low", "Close"]

    @abstractmethod
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 45,
    ) -> pd.DataFrame:
        """
        Return the most-recent `limit` closed candles for `symbol` at `interval`.

        Args:
            symbol:   Exchange-agnostic symbol string (e.g. "BTC", "AAPL").
            interval: Timeframe string (e.g. "1m", "15m", "1h", "1d").
            limit:    Number of candles to return.

        Returns:
            DataFrame with REQUIRED_COLUMNS, sorted oldest-first.
            Returns an empty DataFrame on any error.
        """

    @abstractmethod
    def get_klines_by_range(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Return candles for `symbol` between `start` and `end` (inclusive).

        Same column contract as get_klines().
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure column names and types match the required contract."""
        if df.empty:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        # Flatten MultiIndex columns produced by some libraries
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Rename common variants
        df = df.rename(columns={"Date": "Datetime", "Timestamp": "Datetime"})

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        df = df[self.REQUIRED_COLUMNS].copy()
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.sort_values("Datetime").reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# Yahoo Finance implementation
# ---------------------------------------------------------------------------

class YFinanceFeed(DataFeed):
    """
    Data feed backed by yfinance (yahoo finance scraper).

    Suitable for development and paper-trading.  Not recommended for
    production use — it is an unofficial scraper with no SLA.
    """

    # Canonical symbol → yfinance ticker
    SYMBOL_MAP: dict[str, str] = {
        "SPX": "^GSPC",
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "GC":  "GC=F",
        "NQ":  "NQ=F",
        "CL":  "CL=F",
        "ES":  "ES=F",
        "DJI": "^DJI",
        "QQQ": "QQQ",
        "VIX": "^VIX",
        "DXY": "DX-Y.NYB",
        "AAPL": "AAPL",
        "TSLA": "TSLA",
    }

    # Canonical interval → yfinance interval
    INTERVAL_MAP: dict[str, str] = {
        "1m":  "1m",
        "5m":  "5m",
        "15m": "15m",
        "30m": "30m",
        "1h":  "1h",
        "4h":  "4h",
        "1d":  "1d",
        "1w":  "1wk",
        "1mo": "1mo",
    }

    # How far back to look per interval to guarantee `limit` candles
    _LOOKBACK: dict[str, timedelta] = {
        "1m":  timedelta(days=1),
        "5m":  timedelta(days=5),
        "15m": timedelta(days=10),
        "30m": timedelta(days=20),
        "1h":  timedelta(days=30),
        "4h":  timedelta(days=60),
        "1d":  timedelta(days=120),
        "1w":  timedelta(days=365),
        "1mo": timedelta(days=1825),
    }

    def get_klines(self, symbol: str, interval: str, limit: int = 45) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

        try:
            lookback = self._LOOKBACK.get(interval, timedelta(days=60))
            end = datetime.utcnow()
            start = end - lookback

            ticker = self.SYMBOL_MAP.get(symbol, symbol)
            yf_interval = self.INTERVAL_MAP.get(interval, interval)

            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval=yf_interval,
                auto_adjust=True,
                prepost=False,
                progress=False,
            )
            if df is None or df.empty:
                return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

            df = df.reset_index()
            df = self._normalise(df)
            return df.tail(limit).reset_index(drop=True)

        except Exception as exc:
            print(f"[YFinanceFeed] get_klines error for {symbol}/{interval}: {exc}")
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

    def get_klines_by_range(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

        try:
            ticker = self.SYMBOL_MAP.get(symbol, symbol)
            yf_interval = self.INTERVAL_MAP.get(interval, interval)

            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval=yf_interval,
                auto_adjust=True,
                prepost=False,
                progress=False,
            )
            if df is None or df.empty:
                return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

            df = df.reset_index()
            return self._normalise(df)

        except Exception as exc:
            print(f"[YFinanceFeed] get_klines_by_range error for {symbol}/{interval}: {exc}")
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)


# ---------------------------------------------------------------------------
# Stub implementations (swap in when ready)
# ---------------------------------------------------------------------------

class BinanceFeed(DataFeed):
    """
    Stub for Binance WebSocket / REST feed.

    Replace the NotImplementedError bodies with the real implementation.
    Recommended library: python-binance or binance-connector-python.
    """

    def get_klines(self, symbol: str, interval: str, limit: int = 45) -> pd.DataFrame:
        raise NotImplementedError(
            "BinanceFeed.get_klines is not yet implemented. "
            "Install python-binance and implement using client.get_klines()."
        )

    def get_klines_by_range(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        raise NotImplementedError("BinanceFeed.get_klines_by_range is not yet implemented.")


class PolygonFeed(DataFeed):
    """
    Stub for Polygon.io REST feed (stocks, options, forex, crypto).

    Replace the NotImplementedError bodies with the real implementation.
    Recommended library: polygon-api-client.
    """

    def get_klines(self, symbol: str, interval: str, limit: int = 45) -> pd.DataFrame:
        raise NotImplementedError(
            "PolygonFeed.get_klines is not yet implemented. "
            "Install polygon-api-client and implement using RESTClient.list_aggs()."
        )

    def get_klines_by_range(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        raise NotImplementedError("PolygonFeed.get_klines_by_range is not yet implemented.")
