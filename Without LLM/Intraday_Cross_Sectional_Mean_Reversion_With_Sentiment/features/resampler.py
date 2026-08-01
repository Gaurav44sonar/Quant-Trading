"""
features/resampler.py
=====================
Converts 5-minute OHLCV panels to any target bar frequency.

PIPELINE CHANGE (2026-03-25): Now ingests 5-minute bars directly from Polygon API
instead of 1-minute bars. This reduces data volume (~3x) while maintaining sufficient
granularity for intraday mean reversion signals.

Frequency reference for this strategy:
    1-min  → raw market microstructure (noise)
    5-min  → ← CURRENT: Polygon fetch + minimal resampling
    15-min → optional downstream aggregation (if needed)
    30-min → lower noise but alpha nearly gone
    60-min → minimal signal remaining

The resampler now:
  - Optionally aggregates 5-min bars to coarser frequencies (15, 30, 60-min)
  - Masks flagged bars (extended hours noise, stale prices)
  - Aligns all tickers to a canonical bar grid
  - Drops bars outside regular session (09:30–16:00 ET)
  - Computes derived daily panels for universe builder

Usage
-----
    from features.resampler import Resampler

    r = Resampler(panels_5min, freq="5min")  # pass-through: no resampling
    panels_5m = r.resample()
    
    # Or aggregate to 15-min if needed:
    r = Resampler(panels_5min, freq="15min")
    panels_15m = r.resample()
    panels_15m["close"]   # DataFrame: timestamp × ticker
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Regular session boundaries (ET)
_SESSION_START = "09:30"
_SESSION_END   = "16:00"

# Aggregation rules for each OHLCV field
_AGG_MAP = {
    "open":   "first",
    "high":   "max",
    "low":    "min",
    "close":  "last",
    "volume": "sum",
}


class Resampler:
    """
    Resamples 1-minute OHLCV panels to a target bar frequency.

    Parameters
    ----------
    panels      : dict of {field: DataFrame}  (timestamp × ticker, 1-min)
    freq        : target pandas offset string — "15min", "5min", "30min"
    flagged     : optional bool DataFrame (timestamp × ticker)
                  bars where flagged=True are set to NaN before resampling
    session_only: strip pre/post market bars (default True)
    min_bars_pct: minimum fraction of 1-min bars needed in a resampled bar
                  to be kept (default 0.5 — half the 1-min bars must exist)
    """

    def __init__(
        self,
        panels:       dict[str, pd.DataFrame],
        freq:         str   = "5min",
        flagged:      Optional[pd.DataFrame] = None,
        session_only: bool  = True,
        min_bars_pct: float = 0.5,
    ):
        self.panels       = panels
        self.freq         = freq
        self.flagged      = flagged
        self.session_only = session_only
        self.min_bars_pct = min_bars_pct

        # Bars per resampled bar (for coverage check)
        freq_mins = self._freq_to_minutes(freq)
        self._min_bars = max(1, int(freq_mins * min_bars_pct))
        self._freq_mins = freq_mins

        log.info(
            "Resampler initialised: 5-min → %s  "
            "(min %.0f%% bar coverage required)",
            freq, min_bars_pct * 100
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def resample(self) -> dict[str, pd.DataFrame]:
        """
        Run the full resampling pipeline.

        Returns dict of {field: DataFrame} at target frequency.
        """
        close = self.panels.get("close")
        if close is None:
            raise ValueError("panels must contain 'close'")

        log.info(
            "Resampling %d tickers × %d 5-min bars → %s ...",
            len(close.columns), len(close), self.freq
        )

        # Step 1: mask flagged bars
        panels = self._mask_flagged(self.panels)

        # Step 2: strip pre/post market
        if self.session_only:
            panels = self._filter_session(panels)

        # Step 3: resample each field
        resampled = {}
        for field, agg in _AGG_MAP.items():
            if field in panels:
                r = panels[field].resample(self.freq, label="left",
                                           closed="left").agg(agg)
                resampled[field] = r

        # Step 4: drop bars with insufficient coverage
        resampled = self._apply_coverage_filter(panels, resampled)

        # Step 5: drop all-NaN bars
        close_r = resampled["close"]
        valid   = close_r.notna().any(axis=1)
        resampled = {f: df[valid] for f, df in resampled.items()}

        n_out = len(resampled["close"])
        log.info(
            "Resample complete: %d → %d bars  "
            "(%.1f× compression, %.1f bars/day)",
            len(close), n_out,
            len(close) / max(n_out, 1),
            n_out / max(close.index.normalize().nunique(), 1),
        )
        return resampled

    def resample_to_daily(self) -> dict[str, pd.DataFrame]:
        """
        Aggregate 5-min panels to daily OHLCV.
        Used by universe builder and for daily-level features (gap, overnight).
        """
        panels = self._filter_session(self.panels) if self.session_only \
                 else self.panels
        daily = {}
        for field, agg in _AGG_MAP.items():
            if field in panels:
                d = panels[field].resample("1D", label="left").agg(agg)
                # Keep only actual trading days (days with data)
                has_data = panels["close"].resample("1D").count().max(axis=1) > 0
                daily[field] = d[has_data]
        return daily

    # ── Private ───────────────────────────────────────────────────────────────

    def _mask_flagged(
        self,
        panels: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Set flagged bars to NaN so they don't contaminate resampled bars."""
        if self.flagged is None:
            return panels
        mask   = self.flagged.reindex_like(panels["close"]).fillna(False)
        result = {}
        for field, df in panels.items():
            result[field] = df.where(~mask)
        n_masked = mask.sum().sum()
        log.info("Masked %d flagged 5-min bars before resampling", n_masked)
        return result

    def _filter_session(
        self,
        panels: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Keep only bars within regular trading session (09:30–16:00 ET)."""
        idx  = panels["close"].index
        t    = idx.time
        mask = (t >= pd.Timestamp(_SESSION_START).time()) & \
               (t <= pd.Timestamp(_SESSION_END).time())
        dropped = (~mask).sum()
        if dropped:
            log.info("Dropped %d pre/post-market bars", dropped)
        return {f: df[mask] for f, df in panels.items()}

    def _apply_coverage_filter(
        self,
        panels_1min: dict[str, pd.DataFrame],
        resampled:   dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        Drop resampled bars where fewer than min_bars_pct of the underlying
        1-min bars existed.  Prevents 1-bar "resampled" results from thin
        periods from looking like full-coverage bars.
        """
        # Count non-NaN 1-min bars per resampled bucket
        bar_counts = (
            panels_1min["close"]
            .notna()
            .resample(self.freq, label="left", closed="left")
            .sum()
        )
        sufficient = bar_counts >= self._min_bars
        result = {}
        for field, df in resampled.items():
            aligned = sufficient.reindex_like(df).fillna(False)
            result[field] = df.where(aligned)
        n_dropped = (~sufficient).sum().sum()
        if n_dropped:
            log.debug("Coverage filter: %d resampled bars set to NaN", n_dropped)
        return result

    @staticmethod
    def _freq_to_minutes(freq: str) -> int:
        mapping = {
            "1min": 1, "5min": 5, "10min": 10, "15min": 15,
            "30min": 30, "60min": 60, "1h": 60,
        }
        return mapping.get(freq.lower().replace(" ", ""), 15)
