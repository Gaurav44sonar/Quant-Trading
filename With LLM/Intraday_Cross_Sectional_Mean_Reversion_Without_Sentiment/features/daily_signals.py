"""
features/daily_signals.py
=========================
9 High-Conviction Features for Single-Day Max Profit (NASDAQ)

PRE-OPEN features (computed from previous day's data, available before 9:30 AM ET):
  P1_overnight_gap        — Overnight gap reversal signal
  P2_prev_day_momentum    — Previous day's ATR-normalized return (reversal)
  P3_volume_surge         — Yesterday's volume vs 20-day avg
  P4_relative_strength    — Stock vs NASDAQ index over last 3 days (reversal)
  P5_range_expansion      — Yesterday's range vs 20-day avg (volatility predictor)
  P6_close_location       — Where the stock closed within its daily range

CONFIRMATION features (computed from first 15 min of trading, 9:30-9:45 AM ET):
  C1_opening_bar_reversal — First bar return reversal
  C2_opening_volume       — Opening volume intensity vs history
  C3_gap_fill_speed       — How fast the overnight gap is filling

Usage
-----
    from features.daily_signals import DailySignalEngine
    
    engine = DailySignalEngine(panels, qqq_close)
    pre_open = engine.compute_preopen_signals(target_date)  # {ticker: score}
    confirmed = engine.compute_confirmation(target_date)     # {ticker: score}
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class DailySignalEngine:
    """
    Computes 9 features for single-day stock picking.
    
    Parameters
    ----------
    panels : dict
        OHLCV panels: {"open": DataFrame, "high": DataFrame, ...}
        Each DataFrame: [timestamp × ticker], 5-min bars, US market hours only
    nifty_close : Series
        NASDAQ index (QQQ) close prices (5-min bars, same index as panels)
    lookback_days : int
        Days of history for rolling calculations (default 20)
    """
    
    def __init__(
        self,
        panels: dict,
        nifty_close: pd.Series = None,
        lookback_days: int = 20,
    ):
        self.O = panels["open"]
        self.H = panels["high"]
        self.L = panels["low"]
        self.C = panels["close"]
        self.V = panels["volume"]
        self.nifty = nifty_close
        self.lookback = lookback_days
        
        self._dates = self.C.index.normalize()
        self._unique_dates = sorted(self._dates.unique())
        
        # Pre-compute daily OHLCV summaries (one row per date per ticker)
        self._daily = self._build_daily_summary()
    
    def _build_daily_summary(self) -> dict:
        """Aggregate 5-min bars into daily summaries."""
        daily = {}
        daily["open"] = self.O.groupby(self._dates).first()
        daily["high"] = self.H.groupby(self._dates).max()
        daily["low"] = self.L.groupby(self._dates).min()
        daily["close"] = self.C.groupby(self._dates).last()
        daily["volume"] = self.V.groupby(self._dates).sum()
        daily["range"] = daily["high"] - daily["low"]
        daily["return"] = daily["close"].pct_change(1)
        
        if self.nifty is not None:
            nifty_daily_close = self.nifty.groupby(self.nifty.index.normalize()).last()
            daily["nifty_return"] = nifty_daily_close.pct_change(1, fill_method=None)
        
        return daily
    
    # ══════════════════════════════════════════════════════════════════════════
    # PRE-OPEN SIGNALS (available before 9:30 AM ET)
    # ══════════════════════════════════════════════════════════════════════════
    
    def compute_preopen_signals(self, target_date: pd.Timestamp) -> pd.DataFrame:
        """
        Compute all 6 pre-open features for a target date.
        
        Uses data UP TO (but not including) target_date to avoid lookahead.
        
        Returns
        -------
        DataFrame: [ticker × feature], one row per ticker, 6 columns
        """
        target_date = pd.Timestamp(target_date).normalize()
        if len(self._unique_dates) > 0 and hasattr(self._unique_dates[0], "tz") and self._unique_dates[0].tz and not target_date.tz:
            target_date = target_date.tz_localize(self._unique_dates[0].tz)
        
        # Get index of target_date in our date list
        dates = self._unique_dates
        if target_date not in dates:
            log.warning("Target date %s not in data", target_date)
            return pd.DataFrame()
        
        tgt_idx = dates.index(target_date)
        if tgt_idx < self.lookback:
            log.warning("Insufficient history for %s (need %d days, have %d)",
                       target_date, self.lookback, tgt_idx)
            return pd.DataFrame()
        
        tickers = self.C.columns.tolist()
        signals = pd.DataFrame(index=tickers)
        
        signals["P1_overnight_gap"] = self._overnight_gap(target_date)
        signals["P2_prev_day_momentum"] = self._prev_day_momentum(target_date)
        signals["P3_volume_surge"] = self._volume_surge(target_date)
        signals["P4_relative_strength"] = self._relative_strength(target_date)
        signals["P5_range_expansion"] = self._range_expansion(target_date)
        signals["P6_close_location"] = self._close_location(target_date)
        
        # Cross-sectional z-score each feature
        for col in signals.columns:
            mu = signals[col].mean()
            sigma = signals[col].std()
            if sigma > 0:
                signals[col] = (signals[col] - mu) / sigma
            else:
                signals[col] = 0.0
        
        # Clip outliers
        signals = signals.clip(-3.0, 3.0)
        
        log.info("  Pre-open signals for %s: %d tickers, %d features",
                target_date.date(), len(tickers), len(signals.columns))
        
        return signals
    
    def _overnight_gap(self, target_date) -> pd.Series:
        """
        P1: Overnight gap = today's open / yesterday's close - 1.
        
        Signal: -gap (stocks that gap UP are expected to REVERT down intraday).
        This is historically the strongest single-day reversal predictor.
        """
        today_open = self._daily["open"].loc[target_date]
        
        dates = self._unique_dates
        prev_idx = dates.index(target_date) - 1
        prev_date = dates[prev_idx]
        prev_close = self._daily["close"].loc[prev_date]
        
        gap = (today_open / prev_close.replace(0, np.nan)) - 1
        return -gap  # Negative gap = reversal signal (gap up → expect down)
    
    def _prev_day_momentum(self, target_date) -> pd.Series:
        """
        P2: Previous day's return normalized by trailing ATR.
        
        Signal: -prev_return / ATR. Big up yesterday → expect down today.
        """
        dates = self._unique_dates
        tgt_idx = dates.index(target_date)
        prev_date = dates[tgt_idx - 1]
        
        prev_ret = self._daily["return"].loc[prev_date]
        
        # Trailing ATR (average daily range / close)
        lookback_dates = dates[max(0, tgt_idx - self.lookback):tgt_idx]
        daily_range = self._daily["range"].loc[lookback_dates]
        daily_close = self._daily["close"].loc[lookback_dates]
        atr_pct = (daily_range / daily_close.replace(0, np.nan)).mean()
        
        return -(prev_ret / atr_pct.replace(0, np.nan))
    
    def _volume_surge(self, target_date) -> pd.Series:
        """
        P3: Yesterday's volume / 20-day average volume.
        
        High volume = institutional activity. Combined with reversal signals,
        this identifies stocks where big players have acted → continuation or reversal.
        """
        dates = self._unique_dates
        tgt_idx = dates.index(target_date)
        prev_date = dates[tgt_idx - 1]
        
        prev_vol = self._daily["volume"].loc[prev_date]
        
        lookback_dates = dates[max(0, tgt_idx - self.lookback):tgt_idx]
        avg_vol = self._daily["volume"].loc[lookback_dates].mean()
        
        return (prev_vol / avg_vol.replace(0, np.nan)) - 1  # >0 = higher than avg
    
    def _relative_strength(self, target_date) -> pd.Series:
        """
        P4: Stock return vs NASDAQ index return over last 3 days.
        
        Stocks that massively outperformed the NASDAQ index over 3 days tend to
        underperform next day (cross-sectional mean reversion).
        """
        dates = self._unique_dates
        tgt_idx = dates.index(target_date)
        
        if "nifty_return" not in self._daily:
            return pd.Series(0.0, index=self.C.columns)
        
        # 3-day cumulative returns
        lookback_3 = dates[max(0, tgt_idx - 3):tgt_idx]
        stock_ret_3d = (1 + self._daily["return"].loc[lookback_3]).prod() - 1
        nifty_ret_3d = (1 + self._daily["nifty_return"].loc[lookback_3]).prod() - 1
        
        relative = stock_ret_3d - nifty_ret_3d
        return -relative  # Outperformers → expect underperformance (reversal)
    
    def _range_expansion(self, target_date) -> pd.Series:
        """
        P5: Yesterday's intraday range vs 20-day average range.
        
        Wide-range days predict high volatility next day → more profit opportunity.
        We WANT stocks with range expansion (more room to move).
        """
        dates = self._unique_dates
        tgt_idx = dates.index(target_date)
        prev_date = dates[tgt_idx - 1]
        
        prev_range = self._daily["range"].loc[prev_date]
        prev_close = self._daily["close"].loc[prev_date]
        range_pct = prev_range / prev_close.replace(0, np.nan)
        
        lookback_dates = dates[max(0, tgt_idx - self.lookback):tgt_idx]
        daily_range = self._daily["range"].loc[lookback_dates]
        daily_close = self._daily["close"].loc[lookback_dates]
        avg_range_pct = (daily_range / daily_close.replace(0, np.nan)).mean()
        
        return (range_pct / avg_range_pct.replace(0, np.nan)) - 1
    
    def _close_location(self, target_date) -> pd.Series:
        """
        P6: Close Location Value = (close - low) / (high - low).
        
        CLV near 0 = closed near the low → bounce next morning.
        CLV near 1 = closed near the high → pullback next morning.
        Signal: (0.5 - CLV) → positive for near-low, negative for near-high.
        """
        dates = self._unique_dates
        tgt_idx = dates.index(target_date)
        prev_date = dates[tgt_idx - 1]
        
        c = self._daily["close"].loc[prev_date]
        h = self._daily["high"].loc[prev_date]
        l = self._daily["low"].loc[prev_date]
        
        hl_range = (h - l).replace(0, np.nan)
        clv = (c - l) / hl_range  # 0 = closed at low, 1 = closed at high
        
        return 0.5 - clv  # Positive = near low (bounce expected)
    
    # ══════════════════════════════════════════════════════════════════════════
    # CONFIRMATION SIGNALS (computed from first 15 min of trading)
    # ══════════════════════════════════════════════════════════════════════════
    
    def compute_confirmation(self, target_date: pd.Timestamp) -> pd.DataFrame:
        """
        Compute 3 confirmation features from the first 3 bars (9:30-9:45 AM ET).
        
        Returns
        -------
        DataFrame: [ticker × feature], one row per ticker, 3 columns
        """
        target_date = pd.Timestamp(target_date).normalize()
        if len(self._unique_dates) > 0:
            first_date = self._unique_dates[0]
            if hasattr(first_date, "tz") and first_date.tz and not target_date.tz:
                target_date = target_date.tz_localize(first_date.tz)
        
        # Get today's intraday bars
        day_mask = self._dates == target_date
        if day_mask.sum() == 0:
            return pd.DataFrame()
        
        day_open = self.O[day_mask]
        day_high = self.H[day_mask]
        day_low = self.L[day_mask]
        day_close = self.C[day_mask]
        day_vol = self.V[day_mask]
        
        if len(day_close) < 3:
            log.warning("Less than 3 bars for %s", target_date)
            return pd.DataFrame()
        
        tickers = self.C.columns.tolist()
        signals = pd.DataFrame(index=tickers)
        
        signals["C1_opening_bar_reversal"] = self._opening_bar_reversal(
            day_open, day_close)
        signals["C2_opening_volume"] = self._opening_volume(
            day_vol, target_date)
        signals["C3_gap_fill_speed"] = self._gap_fill_speed(
            day_close, target_date)
        
        # Cross-sectional z-score
        for col in signals.columns:
            mu = signals[col].mean()
            sigma = signals[col].std()
            if sigma > 0:
                signals[col] = (signals[col] - mu) / sigma
            else:
                signals[col] = 0.0
        
        signals = signals.clip(-3.0, 3.0)
        
        log.info("  Confirmation signals for %s: %d tickers",
                target_date.date(), len(tickers))
        
        return signals
    
    def _opening_bar_reversal(self, day_open, day_close) -> pd.Series:
        """
        C1: First bar return reversal.
        
        Large first bar drop → expect bounce. Large first bar rise → expect pullback.
        """
        first_open = day_open.iloc[0]
        # Use close of 3rd bar (9:30) as the "opening move"
        bar3_close = day_close.iloc[min(2, len(day_close) - 1)]
        
        opening_ret = (bar3_close - first_open) / first_open.replace(0, np.nan)
        return -opening_ret  # Reverse: big opening move → reversal
    
    def _opening_volume(self, day_vol, target_date) -> pd.Series:
        """
        C2: Opening 3-bar volume vs historical same-time volume.
        
        High opening volume = conviction behind the move → stronger signal.
        """
        opening_vol = day_vol.iloc[:3].sum()
        
        dates = self._unique_dates
        tgt_idx = dates.index(target_date)
        
        # Historical opening volume (first 3 bars of each day)
        hist_opening_vols = []
        for d in dates[max(0, tgt_idx - self.lookback):tgt_idx]:
            d_mask = self._dates == d
            d_vol = self.V[d_mask]
            if len(d_vol) >= 3:
                hist_opening_vols.append(d_vol.iloc[:3].sum())
        
        if not hist_opening_vols:
            return pd.Series(0.0, index=self.C.columns)
        
        avg_opening_vol = pd.concat(hist_opening_vols, axis=1).mean(axis=1)
        return (opening_vol / avg_opening_vol.replace(0, np.nan)) - 1
    
    def _gap_fill_speed(self, day_close, target_date) -> pd.Series:
        """
        C3: How fast is the overnight gap filling?
        
        If stock gapped UP and price is already falling by bar 3 → gap is filling → REVERSAL confirmed.
        If stock gapped UP and price is still rising → gap is NOT filling → momentum, skip.
        """
        dates = self._unique_dates
        tgt_idx = dates.index(target_date)
        prev_date = dates[tgt_idx - 1]
        
        prev_close = self._daily["close"].loc[prev_date]
        today_open = self._daily["open"].loc[target_date]
        bar3_close = day_close.iloc[min(2, len(day_close) - 1)]
        
        gap = (today_open - prev_close) / prev_close.replace(0, np.nan)
        move_since_open = (bar3_close - today_open) / today_open.replace(0, np.nan)
        
        # Gap fill = move is in OPPOSITE direction to gap
        # gap > 0 and move < 0 → filling (good for our reversal bet)
        gap_fill = np.where(
            gap.abs() > 0.001,  # Only meaningful gaps
            -move_since_open / gap.replace(0, np.nan),  # >0 means filling
            0.0
        )
        
        return pd.Series(gap_fill, index=self.C.columns)
    
    # ══════════════════════════════════════════════════════════════════════════
    # COMPOSITE SCORE
    # ══════════════════════════════════════════════════════════════════════════
    
    def composite_score(
        self,
        target_date: pd.Timestamp,
        preopen_weight: float = 0.6,
        confirm_weight: float = 0.4,
    ) -> pd.Series:
        """
        Combine pre-open and confirmation signals into a single score per ticker.
        
        Parameters
        ----------
        target_date : date to score
        preopen_weight : weight for pre-open signals (default 60%)
        confirm_weight : weight for confirmation signals (default 40%)
        
        Returns
        -------
        Series: [ticker] → composite z-score. Higher = stronger BUY signal.
        """
        preopen = self.compute_preopen_signals(target_date)
        confirm = self.compute_confirmation(target_date)
        
        if preopen.empty:
            return pd.Series(dtype=float)
        
        # Equal-weight within each group
        preopen_score = preopen.mean(axis=1)
        
        if confirm.empty:
            composite = preopen_score
        else:
            confirm_score = confirm.mean(axis=1)
            composite = (preopen_weight * preopen_score +
                        confirm_weight * confirm_score)
        
        # Final z-score
        mu = composite.mean()
        sigma = composite.std()
        if sigma > 0:
            composite = (composite - mu) / sigma
        
        return composite.sort_values(ascending=False)
