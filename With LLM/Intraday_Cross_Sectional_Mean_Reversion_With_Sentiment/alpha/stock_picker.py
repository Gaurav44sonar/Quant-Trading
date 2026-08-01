"""
alpha/stock_picker.py
=====================
Concentrated Stock Picker for Single-Day Max Profit (NASDAQ)

From the 300-stock universe (180 volatile + 120 non-volatile),
this module picks the TOP 80 stocks with the highest conviction
signal for a long-only intraday trade.

Architecture
------------
  1. Score all tickers using DailySignalEngine composite score
  2. Filter: skip stocks with insufficient data or low liquidity
  3. Rank: pick top N by composite z-score
  4. Allocate: equal-weight capital among selected stocks

Usage
-----
    from alpha.stock_picker import StockPicker
    
    picker = StockPicker(panels, qqq_close, capital=100_000)
    picks = picker.pick(target_date)
    # picks = [
    #   {"ticker": "AAPL", "score": 2.8, "allocation": 1250, "shares": 7},
    #   {"ticker": "TSLA", "score": 2.3, "allocation": 1250, "shares": 5},
    #   ...
    # ]
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from features.daily_signals import DailySignalEngine

log = logging.getLogger(__name__)


# ── Sector mapping for diversification caps ───────────────────────────────
# Covers the most common correlated clusters in the universe.
# Stocks not in this map are assigned to "Other".
SECTOR_MAP = {
    # Crypto / Bitcoin proxies
    "MARA": "Crypto", "MSTR": "Crypto", "CLSK": "Crypto", "IREN": "Crypto",
    "HUT": "Crypto", "BTDR": "Crypto", "BTBT": "Crypto", "RIOT": "Crypto",
    "CIFR": "Crypto", "COIN": "Crypto", "HOOD": "Crypto",
    # Quantum computing
    "QUBT": "Quantum", "RGTI": "Quantum", "ARQQ": "Quantum",
    # Semiconductors
    "ARM": "Semiconductor", "MU": "Semiconductor", "AMKR": "Semiconductor",
    "MRVL": "Semiconductor", "RMBS": "Semiconductor", "QCOM": "Semiconductor",
    "ON": "Semiconductor", "SMCI": "Semiconductor", "SMTC": "Semiconductor",
    "ACLS": "Semiconductor", "AMD": "Semiconductor", "NVDA": "Semiconductor",
    "AVGO": "Semiconductor", "INTC": "Semiconductor", "TXN": "Semiconductor",
    "LRCX": "Semiconductor", "KLAC": "Semiconductor", "AMAT": "Semiconductor",
    "NXPI": "Semiconductor", "MCHP": "Semiconductor", "SWKS": "Semiconductor",
    "ADI": "Semiconductor", "MPWR": "Semiconductor",
    # Solar / Clean Energy
    "SEDG": "CleanEnergy", "ENPH": "CleanEnergy", "FSLR": "CleanEnergy",
    "SPWR": "CleanEnergy", "RUN": "CleanEnergy",
    # EV
    "RIVN": "EV", "LCID": "EV",
    # Consumer Staples
    "KDP": "ConsumerStaples", "KHC": "ConsumerStaples", "PEP": "ConsumerStaples",
    "MDLZ": "ConsumerStaples", "MNST": "ConsumerStaples",
    # NSE sectors
    "SAIL.NS": "Metals", "VEDL.NS": "Metals", "NATIONALUM.NS": "Metals",
    "HINDALCO.NS": "Metals", "TATASTEEL.NS": "Metals", "JSWSTEEL.NS": "Metals",
    "INFY.NS": "IT", "WIPRO.NS": "IT", "TECHM.NS": "IT", "LTTS.NS": "IT",
    "MPHASIS.NS": "IT", "COFORGE.NS": "IT", "PERSISTENT.NS": "IT",
    "YESBANK.NS": "Banking", "UCOBANK.NS": "Banking", "BANDHANBNK.NS": "Banking",
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking",
    "AXISBANK.NS": "Banking", "KOTAKBANK.NS": "Banking",
}

# ── Known chronic losers (stocks that repeatedly lose across sessions) ────
# These are blacklisted from selection. Review and update periodically.
KNOWN_LOSERS_US = {
    # Identified from Jun 17-24 logs: lost money in 2+ sessions
    "MSTR", "QUBT", "CLSK", "IREN", "ARM", "PRCT", "RMBS", "ACLS",
}
KNOWN_LOSERS_INDIA = set()  # India performance is acceptable


class StockPicker:
    """
    Selects top N stocks for long-only intraday trades.
    
    Parameters
    ----------
    panels : dict
        OHLCV panels
    nifty_close : Series
        NASDAQ index (QQQ) close prices
    capital : float
        Total trading capital (default $100K)
    n_picks : int
        Number of stocks to pick (default 80)
    min_score : float
        Minimum z-score to qualify (default 0.5)
    min_price : float
        Minimum stock price to trade (default $5)
    min_avg_volume : float
        Minimum 20-day avg daily volume (default 500,000 shares)
    preopen_weight : float
        Weight for pre-open signals in composite (default 0.6)
    confirm_weight : float
        Weight for confirmation signals in composite (default 0.4)
    """
    
    def __init__(
        self,
        panels: dict,
        nifty_close: pd.Series = None,
        capital: float = 100_000,
        n_picks: int = 80,
        min_score: float = 0.5,
        min_price: float = 5.0,
        min_avg_volume: float = 500_000,
        preopen_weight: float = 0.6,
        confirm_weight: float = 0.4,
        momentum_lookback: int = 5,
        momentum_threshold: float = 0.05,
        max_per_sector: int = 3,
        market: str = "us",
        min_basket_size: int = 5,
        momentum_mult: float | None = None,
        use_blacklist: bool = True,
    ):
        self.panels = panels
        self.nifty_close = nifty_close
        self.capital = capital
        self.n_picks = n_picks
        self.min_score = min_score
        self.min_price = min_price
        self.min_avg_volume = min_avg_volume
        self.preopen_weight = preopen_weight
        self.confirm_weight = confirm_weight
        self.momentum_lookback = momentum_lookback
        self.momentum_threshold = momentum_threshold
        self.max_per_sector = max_per_sector
        self.market = market
        self.min_basket_size = min_basket_size
        self.momentum_mult = momentum_mult
        self.use_blacklist = use_blacklist
        
        self.signal_engine = DailySignalEngine(panels, nifty_close)
        
        self._dates = panels["close"].index.normalize()
        self._unique_dates = sorted(self._dates.unique())
    
    def pick(self, target_date: pd.Timestamp) -> list[dict]:
        """
        Pick top N stocks for a given trading day.
        
        Parameters
        ----------
        target_date : the trading date
        
        Returns
        -------
        list of dicts, each with:
            ticker: str
            score: float (composite z-score)
            allocation: float (capital allocated in USD)
            entry_price: float (opening price at 9:45)
            shares: int (number of shares to buy)
            preopen_signals: dict (individual pre-open feature scores)
            confirm_signals: dict (individual confirmation scores)
        """
        target_date = pd.Timestamp(target_date).normalize()
        if len(self._unique_dates) > 0:
            first_date = self._unique_dates[0]
            if hasattr(first_date, "tz") and first_date.tz and not target_date.tz:
                target_date = target_date.tz_localize(first_date.tz)
        
        log.info("=" * 60)
        log.info("  Stock Picker: %s", target_date.date())
        log.info("=" * 60)
        
        # 1. Compute composite scores
        scores = self.signal_engine.composite_score(
            target_date,
            self.preopen_weight,
            self.confirm_weight,
        )
        
        if scores.empty:
            log.warning("  No scores computed for %s", target_date.date())
            return []
        
        # 2. Get individual signal DataFrames for reporting
        preopen_df = self.signal_engine.compute_preopen_signals(target_date)
        confirm_df = self.signal_engine.compute_confirmation(target_date)
        
        # 3. Apply filters (includes momentum filter)
        filtered = self._apply_filters(scores, target_date)
        
        if filtered.empty:
            log.warning("  No stocks passed filters for %s", target_date.date())
            return []
        
        # 4. Apply known-loser blacklist
        blacklist = (KNOWN_LOSERS_US if self.market == "us" else KNOWN_LOSERS_INDIA) if self.use_blacklist else set()
        if blacklist:
            before_bl = len(filtered)
            filtered = filtered[~filtered.index.isin(blacklist)]
            n_bl = before_bl - len(filtered)
            if n_bl > 0:
                log.info("  Blacklist removed %d chronic losers", n_bl)
        
        # 5. Pick top N above minimum score threshold
        qualified = filtered[filtered >= self.min_score]
        
        if len(qualified) < self.min_basket_size:
            log.info("  Only %d stocks above min_score=%.1f (target: %d). Lowering threshold to take top %d.",
                     len(qualified), self.min_score, self.min_basket_size, min(self.min_basket_size, len(filtered)))
            qualified = filtered.head(self.min_basket_size)
        
        # 6. Apply sector diversification cap
        qualified = self._apply_sector_cap(qualified)
        
        top_n = qualified.head(self.n_picks)
        
        # 5. Allocate capital equally
        n_stocks = len(top_n)
        per_stock_capital = self.capital / n_stocks
        
        # 6. Get entry prices (bar 4 open = 9:45 AM, after confirmation window)
        day_mask = self._dates == target_date
        day_open = self.panels["open"][day_mask]
        
        if len(day_open) == 0:
            entry_prices = self.panels["close"].iloc[-1]
            entry_bar = 0
        elif len(day_open) < 4:
            entry_prices = day_open.iloc[0]  # Fallback to first bar
            entry_bar = 0
        else:
            entry_prices = day_open.iloc[3]  # 9:45 AM bar open
            entry_bar = 3
        
        # 7. Build picks list
        picks = []
        for ticker in top_n.index:
            price = entry_prices.get(ticker, np.nan)
            if pd.isna(price) or price <= 0:
                continue
            
            shares = int(per_stock_capital / price)
            if shares <= 0:
                continue
            
            pick = {
                "ticker": ticker,
                "score": float(top_n[ticker]),
                "allocation": per_stock_capital,
                "entry_price": float(price),
                "shares": shares,
                "entry_bar_idx": entry_bar,
                "preopen_signals": (
                    preopen_df.loc[ticker].to_dict()
                    if ticker in preopen_df.index else {}
                ),
                "confirm_signals": (
                    confirm_df.loc[ticker].to_dict()
                    if not confirm_df.empty and ticker in confirm_df.index else {}
                ),
            }
            picks.append(pick)
        
        # Log picks
        log.info("  Selected %d stocks (capital: $%.0f each):", len(picks),
                per_stock_capital)
        for p in picks:
            log.info("    %-12s  score=%+.2fz  entry=$%.2f  shares=%d",
                    p["ticker"], p["score"], p["entry_price"], p["shares"])
        
        return picks
    
    def _apply_filters(
        self,
        scores: pd.Series,
        target_date: pd.Timestamp,
    ) -> pd.Series:
        """
        Filter out stocks that shouldn't be traded.
        
        Removes:
          - Stocks below min_price
          - Stocks with insufficient volume
          - Stocks with missing data
        """
        dates = self._unique_dates
        if target_date not in dates:
            log.warning("  Target date %s not found in data. Cannot apply filters.", target_date.date())
            return pd.Series(dtype=float)
        tgt_idx = dates.index(target_date)
        
        # Price filter
        day_mask = self._dates == target_date
        day_close = self.panels["close"][day_mask]
        if len(day_close) > 0:
            last_price = day_close.iloc[0]
        else:
            prev_date = dates[tgt_idx - 1]
            prev_mask = self._dates == prev_date
            last_price = self.panels["close"][prev_mask].iloc[-1]
        
        price_ok = last_price >= self.min_price
        
        # Volume filter (20-day avg)
        lookback_dates = dates[max(0, tgt_idx - 20):tgt_idx]
        daily_vol = self.panels["volume"].groupby(self._dates).sum()
        if len(lookback_dates) > 0 and len(daily_vol) > 0:
            avg_vol = daily_vol.reindex(lookback_dates).mean()
            vol_ok = avg_vol >= self.min_avg_volume
        else:
            vol_ok = pd.Series(True, index=scores.index)
        
        # Data completeness (must have data for today)
        has_data = scores.notna()
        
        # Combine filters
        mask = price_ok.reindex(scores.index).fillna(False) & \
               vol_ok.reindex(scores.index).fillna(False) & \
               has_data
        
        n_filtered = (~mask).sum()
        if n_filtered > 0:
            log.info("  Filtered out %d stocks (price/volume/data)", n_filtered)
        
        # Dynamic momentum threshold calculation
        if self.momentum_mult is not None:
            # Trailing ATR% over 20 days (excluding today)
            lookback_vol_dates = dates[max(0, tgt_idx - 20):tgt_idx]
            daily_range = self.signal_engine._daily["range"]
            daily_close = self.signal_engine._daily["close"]
            if len(lookback_vol_dates) > 0 and len(daily_close) > 0:
                avg_range = daily_range.reindex(lookback_vol_dates)
                avg_close = daily_close.reindex(lookback_vol_dates)
                atr_pct = (avg_range / avg_close.replace(0, np.nan)).mean()
            else:
                atr_pct = pd.Series(0.05, index=scores.index)
            
            dyn_threshold = (self.momentum_mult * atr_pct).reindex(scores.index).fillna(self.momentum_threshold)
        else:
            dyn_threshold = pd.Series(self.momentum_threshold, index=scores.index)
            
        # Momentum filter: reject stocks in strong directional trends
        # Stocks with |5-day return| > threshold are trending, not mean-reverting (excluding today)
        lookback_dates = dates[max(0, tgt_idx - self.momentum_lookback):tgt_idx]
        if len(lookback_dates) >= 2:
            daily_close = self.panels["close"].groupby(self._dates).last()
            if len(daily_close) >= 2:
                start_close = daily_close.reindex(lookback_dates).iloc[0]
                end_close = daily_close.reindex(lookback_dates).iloc[-1]
                momentum_ret = (end_close - start_close) / start_close.replace(0, np.nan)
                momentum_ok = momentum_ret.abs() <= dyn_threshold.reindex(momentum_ret.index).fillna(self.momentum_threshold)
                
                # Align and combine with existing mask
                momentum_aligned = momentum_ok.reindex(scores.index).fillna(True)
                n_momentum_filtered = (~momentum_aligned & mask).sum()
                if n_momentum_filtered > 0:
                    log.info("  Momentum filter rejected %d trending stocks (%s)",
                            n_momentum_filtered,
                            "dynamic ATR-scaled threshold" if self.momentum_mult is not None else f"static {self.momentum_threshold * 100}% threshold")
                mask = mask & momentum_aligned
        
        n_filtered = (~mask).sum()
        if n_filtered > 0:
            log.info("  Total filtered out: %d stocks (price/volume/data/momentum)", n_filtered)
        
        return scores[mask].sort_values(ascending=False)
    
    def _apply_sector_cap(
        self,
        scores: pd.Series,
    ) -> pd.Series:
        """
        Cap the number of stocks per sector to prevent correlated concentration.
        
        Uses SECTOR_MAP for known sector assignments. Stocks not in the map
        are assigned to sector "Other" which is not capped.
        """
        if self.max_per_sector <= 0:
            return scores
        
        sector_counts = {}
        kept = []
        
        for ticker in scores.index:
            sector = SECTOR_MAP.get(ticker, "Other")
            
            if sector == "Other":
                # Don't cap uncategorized stocks
                kept.append(ticker)
                continue
            
            count = sector_counts.get(sector, 0)
            if count < self.max_per_sector:
                kept.append(ticker)
                sector_counts[sector] = count + 1
            else:
                log.info("  Sector cap: skipped %s (sector=%s, already %d/%d)",
                        ticker, sector, count, self.max_per_sector)
        
        n_capped = len(scores) - len(kept)
        if n_capped > 0:
            log.info("  Sector diversification removed %d stocks", n_capped)
        
        return scores[kept]
    
    def get_trading_dates(self) -> list:
        """Return all dates available for backtesting."""
        # Need at least lookback_days of history
        lookback = self.signal_engine.lookback
        if len(self._unique_dates) <= lookback:
            return []
        return self._unique_dates[lookback:]
