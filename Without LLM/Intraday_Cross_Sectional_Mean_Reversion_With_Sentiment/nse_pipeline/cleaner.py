"""
nse_pipeline/cleaner.py
=======================
OHLCV data cleaning for NASDAQ intraday bars.

US market cleaning rules:
  1. Extreme return detection (no per-stock circuit breakers in US)
  2. Standard spike removal, gap handling, OHLC ordering
  3. Volume spike filtering
  4. Opening/closing auction bar handling

Usage
-----
    from nse_pipeline.cleaner import NSECleaner
    
    cleaner = NSECleaner(config)
    clean_panels = cleaner.clean(raw_panels)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class NSECleaner:
    """
    Clean raw OHLCV panel data for NASDAQ stocks.
    
    Parameters
    ----------
    max_daily_return : float
        Max absolute daily return before flagging (0.50 = 50%, US has no per-stock circuits)
    max_intraday_range : float
        Max (high-low)/close ratio per bar
    max_volume_spike_mult : float
        Flag bars where volume > mult × rolling median volume
    volume_spike_window : int
        Rolling window for volume spike detection
    max_overnight_gap : float
        Max absolute overnight gap (close→open next day)
    enforce_ohlc_order : bool
        If True, enforce low ≤ open,close ≤ high
    """
    
    def __init__(
        self,
        max_daily_return: float = 0.50,
        max_intraday_range: float = 0.50,
        max_volume_spike_mult: float = 10.0,
        volume_spike_window: int = 20,
        max_overnight_gap: float = 0.50,
        enforce_ohlc_order: bool = True,
    ):
        self.max_daily_return = max_daily_return
        self.max_intraday_range = max_intraday_range
        self.max_vol_spike = max_volume_spike_mult
        self.vol_spike_win = volume_spike_window
        self.max_gap = max_overnight_gap
        self.enforce_ohlc = enforce_ohlc_order
    
    def clean(
        self,
        panels: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        Clean a set of OHLCV panels.
        
        Parameters
        ----------
        panels : dict
            {"open": DataFrame, "high": DataFrame, "low": DataFrame,
             "close": DataFrame, "volume": DataFrame}
            Each DataFrame: [timestamp × ticker]
        
        Returns
        -------
        dict of cleaned panels (same structure)
        """
        log.info("Cleaning NASDAQ data ...")
        
        O = panels["open"].copy()
        H = panels["high"].copy()
        L = panels["low"].copy()
        C = panels["close"].copy()
        V = panels["volume"].copy()
        
        n_before = C.notna().sum().sum()
        
        # ── Rule 1: Enforce OHLC ordering ─────────────────────────────────
        if self.enforce_ohlc:
            O, H, L, C = self._enforce_ohlc(O, H, L, C)
        
        # ── Rule 2: Flag extreme intraday returns ─────────────────────────
        bar_ret = C.pct_change(1).abs()
        extreme_mask = bar_ret > self.max_daily_return
        n_extreme = extreme_mask.sum().sum()
        if n_extreme > 0:
            log.info("  Extreme return bars flagged: %d", n_extreme)
            # NaN out extreme bars (they distort signals)
            for panel in [O, H, L, C, V]:
                panel[extreme_mask] = np.nan
        
        # ── Rule 3: Flag extreme intraday range ──────────────────────────
        intraday_range = (H - L) / C.replace(0.0, np.nan)
        range_mask = intraday_range.abs() > self.max_intraday_range
        n_range = range_mask.sum().sum()
        if n_range > 0:
            log.info("  Extreme range bars flagged: %d", n_range)
            for panel in [O, H, L, C, V]:
                panel[range_mask] = np.nan
        
        # ── Rule 4: Flag volume spikes ───────────────────────────────────
        vol_median = V.rolling(self.vol_spike_win, min_periods=5).median()
        vol_spike_mask = V > (self.max_vol_spike * vol_median)
        # Don't NaN volume spikes — they carry information — just flag
        n_vol_spikes = vol_spike_mask.sum().sum()
        if n_vol_spikes > 0:
            log.debug("  Volume spikes detected: %d (kept, not removed)", n_vol_spikes)
        
        # ── Rule 5: Remove zero/negative prices ─────────────────────────
        for panel in [O, H, L, C]:
            panel[panel <= 0] = np.nan
        V[V < 0] = 0
        
        # ── Rule 6: Forward-fill small gaps (max 2 bars) ────────────────
        O = O.ffill(limit=2)
        H = H.ffill(limit=2)
        L = L.ffill(limit=2)
        C = C.ffill(limit=2)
        
        n_after = C.notna().sum().sum()
        pct_removed = (1 - n_after / max(n_before, 1)) * 100
        
        log.info(
            "  Cleaning complete: %d -> %d valid cells (%.1f%% removed)",
            n_before, n_after, pct_removed,
        )
        
        return {"open": O, "high": H, "low": L, "close": C, "volume": V}
    
    def _enforce_ohlc(
        self,
        O: pd.DataFrame,
        H: pd.DataFrame,
        L: pd.DataFrame,
        C: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Enforce low ≤ min(open, close) and high ≥ max(open, close)."""
        min_oc = pd.DataFrame(
            np.minimum(O.values, C.values), index=O.index, columns=O.columns
        )
        max_oc = pd.DataFrame(
            np.maximum(O.values, C.values), index=O.index, columns=O.columns
        )
        
        # Correct high and low
        H_fixed = pd.DataFrame(
            np.maximum(H.values, max_oc.values), index=H.index, columns=H.columns
        )
        L_fixed = pd.DataFrame(
            np.minimum(L.values, min_oc.values), index=L.index, columns=L.columns
        )
        
        n_fixed = ((H != H_fixed).sum().sum() + (L != L_fixed).sum().sum())
        if n_fixed > 0:
            log.debug("  OHLC ordering fixed: %d cells", n_fixed)
        
        return O, H_fixed, L_fixed, C
