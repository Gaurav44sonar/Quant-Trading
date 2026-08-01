"""
nse_pipeline/fetcher.py
=======================
Fetch intraday OHLCV data for NASDAQ stocks using yfinance.

NASDAQ tickers require no suffix on Yahoo Finance.
Filters to official US market trading hours: 9:30 AM - 4:00 PM ET only.

Limitations:
  - yfinance provides max ~59 days of intraday data
  - Data is delayed (not real-time)
  - Suitable for research/backtesting, not live trading

Usage
-----
    from nse_pipeline.fetcher import NSEFetcher
    
    fetcher = NSEFetcher(ticker_suffix="")
    df = fetcher.fetch_ticker("AAPL", days=30, interval="5m")
    panels = fetcher.fetch_universe(["AAPL", "MSFT", "GOOGL"], days=30)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytz

log = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")

# Official US market trading hours
MARKET_OPEN  = pd.Timestamp("09:30:00").time()
MARKET_CLOSE = pd.Timestamp("16:00:00").time()


class NSEFetcher:
    """
    Fetch intraday OHLCV data for NASDAQ stocks via yfinance.
    
    Parameters
    ----------
    ticker_suffix : str
        Suffix for tickers on Yahoo (default "" for NASDAQ)
    rate_limit_sleep : float
        Seconds to sleep between requests (default 0.3)
    retry_attempts : int
        Max retries per ticker (default 3)
    retry_backoff : float
        Seconds to wait between retries (default 5.0)
    """
    
    def __init__(
        self,
        ticker_suffix: str = "",
        rate_limit_sleep: float = 0.3,
        retry_attempts: int = 3,
        retry_backoff: float = 5.0,
    ):
        self.suffix = ticker_suffix
        self.sleep = rate_limit_sleep
        self.retries = retry_attempts
        self.backoff = retry_backoff
        
        # Lazy import
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            raise ImportError(
                "yfinance is required. Install with: pip install yfinance"
            )
    
    def fetch_ticker(
        self,
        ticker: str,
        days: int = 59,
        interval: str = "5m",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch intraday OHLCV for a single NASDAQ ticker.
        
        Parameters
        ----------
        ticker : str
            NASDAQ ticker symbol (e.g. "AAPL", "MSFT")
        days : int
            Number of days of history (max 59 for intraday)
        interval : str
            Bar interval ("1m", "5m", "15m", "30m", "1h")
        
        Returns
        -------
        DataFrame with columns [open, high, low, close, volume]
        Index: tz-aware DatetimeIndex in US/Eastern
        Returns None if fetch fails.
        """
        yf_ticker = f"{ticker}{self.suffix}"
        days = min(days, 59)  # yfinance hard limit
        
        end_dt = datetime.now(ET)
        start_dt = end_dt - timedelta(days=days)
        
        for attempt in range(1, self.retries + 1):
            try:
                data = self._yf.download(
                    tickers=yf_ticker,
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )
                
                if data is None or data.empty:
                    log.warning("  %s: empty data (attempt %d/%d)", ticker, attempt, self.retries)
                    time.sleep(self.backoff * attempt)
                    continue
                
                # Handle MultiIndex columns from yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Standardize column names
                data.columns = [c.lower().replace(" ", "_") for c in data.columns]
                
                # Keep only OHLCV
                keep_cols = ["open", "high", "low", "close", "volume"]
                available = [c for c in keep_cols if c in data.columns]
                if len(available) < 5:
                    log.warning("  %s: missing columns %s", ticker, 
                               set(keep_cols) - set(available))
                    return None
                data = data[keep_cols]
                
                # Ensure timezone-aware in US/Eastern
                if data.index.tz is None:
                    data.index = data.index.tz_localize("UTC").tz_convert(ET)
                elif str(data.index.tz) != "US/Eastern":
                    data.index = data.index.tz_convert(ET)
                
                # ── Filter to official US trading hours ONLY ──
                data = self._filter_trading_hours(data)
                
                if data.empty:
                    log.warning("  %s: no data within trading hours", ticker)
                    return None
                
                # Drop rows with all NaN
                data = data.dropna(how="all")
                
                log.debug(
                    "  [OK] %s: %d bars | %s -> %s",
                    ticker, len(data),
                    data.index.min().strftime("%Y-%m-%d"),
                    data.index.max().strftime("%Y-%m-%d"),
                )
                return data
                
            except Exception as e:
                log.warning(
                    "  %s: error on attempt %d/%d: %s",
                    ticker, attempt, self.retries, str(e)[:200]
                )
                time.sleep(self.backoff * attempt)
        
        log.error("  %s: FAILED after %d attempts", ticker, self.retries)
        return None
    
    def _filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to official US market trading hours: 9:30 AM - 4:00 PM ET."""
        times = df.index.time
        mask = (times >= MARKET_OPEN) & (times < MARKET_CLOSE)
        filtered = df[mask]
        
        n_removed = len(df) - len(filtered)
        if n_removed > 0:
            log.debug("  Removed %d bars outside US market hours", n_removed)
        
        return filtered
    
    def fetch_universe(
        self,
        tickers: list[str],
        days: int = 59,
        interval: str = "5m",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch intraday data for multiple NASDAQ tickers.
        
        Returns dict of {field: DataFrame[timestamp × ticker]} panels:
            {"open": df, "high": df, "low": df, "close": df, "volume": df}
        """
        log.debug("Fetching %d tickers from yfinance (NASDAQ) ...", len(tickers))
        
        all_data = {}
        failed = []
        
        for i, ticker in enumerate(tickers):
            log.debug("  [%d/%d] %s ...", i + 1, len(tickers), ticker)
            df = self.fetch_ticker(ticker, days=days, interval=interval)
            if df is not None and not df.empty:
                all_data[ticker] = df
            else:
                failed.append(ticker)
            
            # Rate limiting
            if i < len(tickers) - 1:
                time.sleep(self.sleep)
        
        if failed:
            log.warning("Failed tickers (%d): %s", len(failed), failed[:20])
        
        log.info(
            "Fetch complete: %d/%d tickers successful",
            len(all_data), len(tickers)
        )
        
        if not all_data:
            return {}
        
        return self._build_panels(all_data)
    
    def _build_panels(
        self,
        ticker_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        Convert per-ticker DataFrames into wide panels.
        
        Returns
        -------
        dict: {"open": DataFrame, "high": DataFrame, ...}
              Each DataFrame: [timestamp × ticker]
        """
        fields = ["open", "high", "low", "close", "volume"]
        panels = {}
        
        for field in fields:
            series_dict = {}
            for ticker, df in ticker_data.items():
                if field in df.columns:
                    series_dict[ticker] = df[field]
            
            if series_dict:
                panel = pd.DataFrame(series_dict)
                panel = panel.sort_index()
                panels[field] = panel
        
        if panels:
            n_bars = len(panels.get("close", pd.DataFrame()))
            n_tickers = len(panels.get("close", pd.DataFrame()).columns)
            log.info(
                "Panels built: %d bars x %d tickers | %s -> %s",
                n_bars, n_tickers,
                panels["close"].index.min().strftime("%Y-%m-%d %H:%M"),
                panels["close"].index.max().strftime("%Y-%m-%d %H:%M"),
            )
        
        return panels
    
    def fetch_index(
        self,
        ticker: str = "QQQ",
        days: int = 59,
        interval: str = "5m",
    ) -> tuple[pd.Series, pd.Series]:
        """
        Fetch NASDAQ index (QQQ) data for beta hedging.
        
        Returns
        -------
        prices : pd.Series — QQQ close prices
        returns : pd.Series — QQQ bar returns
        """
        log.info("Fetching NASDAQ index (%s) ...", ticker)
        
        end_dt = datetime.now(ET)
        start_dt = end_dt - timedelta(days=min(days, 59))
        
        data = self._yf.download(
            tickers=ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        
        if data is None or data.empty:
            log.error("Failed to fetch NASDAQ index data")
            return pd.Series(dtype=float), pd.Series(dtype=float)
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data.columns = [c.lower() for c in data.columns]
        
        # Timezone handling
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC").tz_convert(ET)
        elif str(data.index.tz) != "US/Eastern":
            data.index = data.index.tz_convert(ET)
        
        # Filter to trading hours
        data = self._filter_trading_hours(data)
        
        prices = data["close"]
        returns = prices.pct_change(1)
        
        log.info(
            "NASDAQ index (QQQ): %d bars | range: $%.2f -> $%.2f",
            len(prices), prices.min(), prices.max()
        )
        
        return prices, returns
    
    def fetch_daily(
        self,
        tickers: list[str],
        period: str = "1y",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch daily OHLCV for universe screening (no intraday limit).
        
        Parameters
        ----------
        tickers : list of NASDAQ ticker symbols
        period : str — yfinance period (e.g. "1y", "2y", "6mo")
        
        Returns dict panels: {"open": df, "high": df, ...}
        """
        log.info("Fetching daily data for %d tickers ...", len(tickers))
        
        yf_tickers = [f"{t}{self.suffix}" for t in tickers]
        ticker_str = " ".join(yf_tickers)
        
        try:
            data = self._yf.download(
                tickers=ticker_str,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )
        except Exception as e:
            log.error("Batch daily fetch failed: %s", str(e)[:200])
            return {}
        
        if data is None or data.empty:
            return {}
        
        # Build panels from multi-ticker download
        fields = ["Open", "High", "Low", "Close", "Volume"]
        panels = {}
        
        for field in fields:
            field_lower = field.lower()
            series_dict = {}
            
            for ticker, yf_ticker in zip(tickers, yf_tickers):
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        col_data = data[yf_ticker][field]
                    else:
                        col_data = data[field]
                    series_dict[ticker] = col_data
                except (KeyError, TypeError):
                    continue
            
            if series_dict:
                panels[field_lower] = pd.DataFrame(series_dict)
        
        if panels:
            log.info(
                "Daily panels: %d days × %d tickers",
                len(panels.get("close", pd.DataFrame())),
                len(panels.get("close", pd.DataFrame()).columns),
            )
        
        return panels
