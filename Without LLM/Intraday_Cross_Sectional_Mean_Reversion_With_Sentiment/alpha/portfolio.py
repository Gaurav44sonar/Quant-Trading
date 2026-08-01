"""
Portfolio Construction — Cross-Sectional Intraday Mean Reversion
================================================================

Fixes vs previous version
--------------------------
  FIX-A  log.info "%,.0f" → logging uses %-formatting which does NOT support
         the "," thousands-separator.  Replaced with pre-formatted f-string arg.

  FIX-B  Index-alignment collapse (ROOT CAUSE of 0.09x gross):
         alpha is (1485, 44) but close/volume are (31858, 44).
         Dividing  self.alpha / vol_filled  lets pandas align on the FULL
         close index → 97% of rows become NaN → fillna(0.0) → gross ≈ 0.09x.
         Fix: reindex vol_filled AND the liquidity mask to alpha.index BEFORE
         any arithmetic.  EWM is still computed on full history for warm-up.

Other corrections (carried forward from v1)
--------------------------------------------
  1. Pipeline order: liquidity → dollar-neutral → limits → normalize → scale
  2. vol_filled.fillna uses .apply() so row-median broadcast actually works
  3. min_weight threshold applied AFTER gross-leverage scaling
  4. Turnover control removed (2-bar half-life makes it adversarial)
  5. Soft winsorisation before hard clip
  6. Re-neutralise after clipping
  7. _assert_portfolio_health raises loudly before saving garbage weights
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from features.core import ewm_std          # unchanged dependency

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
class PortfolioBuilder:
    """
    Builds bar-level portfolio weights for a cross-sectional intraday
    mean-reversion strategy.

    Parameters
    ----------
    alpha : pd.DataFrame
        Composite alpha signal, shape (n_signal_bars, n_tickers).
        May be a SUBSET of the close/volume index (e.g. last N bars only).
    close : pd.DataFrame
        Last-trade price panel.  Must contain alpha.index as a subset.
    volume : pd.DataFrame
        Bar volume panel.  Same requirement as close.
    halflife : int
        EWM half-life (bars) for realised-vol estimation.
    target_vol : float
        Annualised target vol (reserved for future risk-parity extension).
    max_weight : float
        Hard cap on |weight| per name as a fraction of gross NAV.
    min_weight : float
        Post-leverage threshold; positions below this are zeroed.
        Keep <= 0.001 to preserve breadth across 44 names.
    gross_lev : float
        Target sum(|weights|).  Typical intraday l/s = 2.0.
    min_adtv_usd : float
        Minimum average daily dollar volume ($) for a name to be included.
    bars_per_year : int
        Bars per calendar year.  15-min US equities = 6552.
    winsor_z : float | None
        Soft-winsorise weights at +/- winsor_z * cross-sectional sigma before
        the hard clip.  None disables.
    """

    def __init__(
        self,
        alpha: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        halflife: int = 13,
        target_vol: float = 0.15,
        max_weight: float = 0.10,
        min_weight: float = 0.001,
        gross_lev: float = 2.0,
        min_adtv_usd: float = 1_000_000,
        bars_per_year: int = 6_552,
        winsor_z: float | None = 3.0,
    ):
        self.alpha         = alpha
        self.close         = close
        self.volume        = volume
        self.halflife      = halflife
        self.target_vol    = target_vol
        self.max_weight    = max_weight
        self.min_weight    = min_weight
        self.gross_lev     = gross_lev
        self.min_adtv_usd  = min_adtv_usd
        self.bars_per_year = bars_per_year
        self.winsor_z      = winsor_z

        # FIX-A: pre-format dollar amount; logging % does not support "," flag
        adtv_str = f"{min_adtv_usd:,.0f}"
        log.info(
            "PortfolioBuilder init — gross_lev=%.1fx | max_weight=%.0f%% | "
            "min_weight=%.3f%% | winsor_z=%s | turnover_control=DISABLED | "
            "alpha=%s | close=%s | adtv=$%s",
            gross_lev,
            max_weight * 100,
            min_weight * 100,
            str(winsor_z),
            str(alpha.shape),
            str(close.shape),
            adtv_str,
        )

        # Guard: close must cover the entire alpha index
        missing = alpha.index.difference(close.index)
        if len(missing):
            raise ValueError(
                f"close panel is missing {len(missing)} bars that appear in "
                f"alpha.index.  First missing: {missing[0]}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def build(self) -> pd.DataFrame:
        """
        Return a (n_signal_bars x n_tickers) DataFrame of portfolio weights.

        Pipeline
        --------
        1. Volatility-scale alpha     -> risk-adjusted signal  [FIX-B]
        2. Liquidity filter           -> zero untradeable names early
        3. Dollar-neutralise          -> remove market-beta
        4. Winsorise + hard clip      -> handle outliers, enforce max_weight
        5. Re-neutralise              -> restore $ neutrality after clipping
        6. Normalise to unit gross    -> deterministic input to step 7
        7. Scale to gross leverage    -> apply min_weight threshold last
        """
        log.info("Building portfolio weights ...")

        log.info("  [1/6] Volatility scaling ...")
        w = self._volatility_scale()

        # FIX-A: pre-format for log.info
        adtv_str = f"{self.min_adtv_usd:,.0f}"
        log.info("  [2/6] Liquidity filter (min ADTV $%s) ...", adtv_str)
        w = self._liquidity_filter(w)

        log.info("  [3/6] Dollar neutralisation ...")
        w = self._dollar_neutralise(w)

        log.info("  [4/6] Position limits (max=%.0f%%) ...", self.max_weight * 100)
        w = self._apply_limits(w)

        log.info("  [5/6] Normalise to unit gross ...")
        w = self._normalize(w)

        log.info("  [6/6] Scale to %.1fx gross leverage ...", self.gross_lev)
        w = self._scale_to_gross_lev(w)

        self._assert_portfolio_health(w)
        return w

    # ──────────────────────────────────────────────────────────────────────────
    # Pipeline steps
    # ──────────────────────────────────────────────────────────────────────────

    def _volatility_scale(self) -> pd.DataFrame:
        """
        Divide alpha by EWM realised vol so each name contributes equal
        risk per unit of signal.

        FIX-B  (root cause of 0.09x gross)
        ------------------------------------
        close has 31858 rows; alpha has 1485 rows.

        OLD (broken):
            scaled = self.alpha / vol_filled
            # pandas aligns on full close index
            # -> 97% of rows are NaN (no alpha there)
            # -> fillna(0.0) -> portfolio is 97% zero

        NEW (fixed):
            vol_aligned = vol_filled.reindex(self.alpha.index)
            scaled = self.alpha / vol_aligned
            # both sides have identical (1485, 44) index -> no phantom NaNs
            # EWM is still computed on full 31858-row history for proper warm-up
        """
        ret = self.close.pct_change(1)
        vol = ewm_std(ret, self.halflife)

        # Annualise on full close history (warm-up uses all available data)
        vol_ann = vol * np.sqrt(self.bars_per_year)

        # Row-median fallback for tickers with insufficient history
        row_median = vol_ann.median(axis=1)

        # FIX from v1: .apply broadcasts row_median correctly (fillna axis mismatch)
        vol_filled = vol_ann.apply(lambda col: col.fillna(row_median), axis=0)
        vol_filled = vol_filled.replace(0.0, np.nan)
        vol_filled = vol_filled.apply(lambda col: col.fillna(row_median), axis=0)

        # Global fallback for any bars where the entire row is NaN
        global_median = vol_filled.stack().median()
        vol_filled = vol_filled.fillna(global_median)

        # FIX-B: align to alpha.index BEFORE dividing
        vol_aligned = vol_filled.reindex(self.alpha.index)

        scaled = self.alpha / vol_aligned
        return scaled.fillna(0.0)

    def _liquidity_filter(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Zero names whose 20-bar rolling avg dollar volume (scaled to daily)
        is below min_adtv_usd.

        bars_per_day = bars_per_year / 252  (approx 26 for 15-min bars)

        FIX-B: reindex liquid mask to weights.index (= alpha.index) so the
        boolean mask aligns correctly and does not silently become all-True
        when weights has fewer rows than close.
        """
        bars_per_day = self.bars_per_year / 252
        dollar_vol   = self.close * self.volume
        avg_dv_bar   = dollar_vol.rolling(20, min_periods=5).mean()
        avg_dv_day   = avg_dv_bar * bars_per_day

        liquid = avg_dv_day >= self.min_adtv_usd

        # FIX-B: align to weights (alpha) index
        liquid_aligned = liquid.reindex(weights.index).fillna(False)

        result = weights.copy()
        result[~liquid_aligned] = 0.0
        return result

    def _dollar_neutralise(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Subtract cross-sectional mean so long book ~= short book."""
        return weights.sub(weights.mean(axis=1), axis=0)

    def _apply_limits(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        1. Soft winsorise at +/- winsor_z * row_sigma  (preserves rank order)
        2. Hard clip at +/- max_weight
        3. Re-neutralise so $ neutrality is restored after clipping distorts mean
        """
        w = weights.copy()

        # Soft winsorisation
        if self.winsor_z is not None:
            row_std = w.std(axis=1).replace(0.0, np.nan)
            upper   = ( row_std * self.winsor_z).values[:, None]
            lower   = (-row_std * self.winsor_z).values[:, None]
            w = pd.DataFrame(
                np.clip(w.values, lower, upper),
                index=w.index, columns=w.columns,
            )

        # Hard clip
        w = w.clip(-self.max_weight, self.max_weight)

        # Re-neutralise after clipping shifts the cross-sectional mean
        w = w.sub(w.mean(axis=1), axis=0)
        return w

    def _normalize(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Scale each row so sum(|w|) == 1.0.  Zero rows stay zero."""
        gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
        return weights.div(gross, axis=0).fillna(0.0)

    def _scale_to_gross_lev(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Multiply unit-gross weights by gross_lev so sum(|w|) == gross_lev.

        FIX from v1: min_weight applied AFTER scaling so the threshold is
        meaningful in leverage-adjusted space (not on ~0.02-scale pre-lev).
        """
        w = weights * self.gross_lev

        # Drop sub-threshold positions after leverage is applied
        w = w.where(w.abs() >= self.min_weight, 0.0)
        return w

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def _assert_portfolio_health(self, weights: pd.DataFrame) -> None:
        """
        Raise immediately if the portfolio has silently collapsed.
        Better to crash here than to persist garbage weights to disk.
        """
        avg_gross = weights.abs().sum(axis=1).mean()
        avg_net   = weights.sum(axis=1).abs().mean()
        avg_n     = (weights.abs() > self.min_weight).sum(axis=1).mean()

        log.info(
            "Portfolio built -- avg %.1f positions | avg gross=%.3fx | avg |net|=%.4f",
            avg_n, avg_gross, avg_net,
        )

        # Allow up to 50% of gross to be lost to liquidity filter
        min_expected_gross = self.gross_lev * 0.50
        if avg_gross < min_expected_gross:
            raise RuntimeError(
                f"Portfolio collapsed: avg gross={avg_gross:.3f}x "
                f"(expected >= {min_expected_gross:.2f}x).\n"
                "Likely causes:\n"
                "  1. alpha.index not aligned to close.index before division  (FIX-B)\n"
                "  2. min_adtv_usd too high -- too many names zeroed by liquidity filter\n"
                "  3. alpha signal is near-zero for most bars"
            )

        max_allowed_net = 0.15 * self.gross_lev
        if avg_net > max_allowed_net:
            raise RuntimeError(
                f"Dollar neutrality broken: avg |net|={avg_net:.4f} "
                f"(allowed <= {max_allowed_net:.4f}).\n"
                "Check _dollar_neutralise and _apply_limits re-neutralisation."
            )

        if avg_n < 4:
            raise RuntimeError(
                f"Insufficient breadth: avg {avg_n:.1f} positions (expected >= 4).\n"
                "Lower min_weight or min_adtv_usd."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────────

    def turnover(self, weights: pd.DataFrame) -> pd.Series:
        """
        One-way turnover as fraction of gross NAV per bar.
        Annualised: turnover().mean() * bars_per_year
        """
        delta = weights.diff().abs().sum(axis=1)
        gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
        return (delta / gross).fillna(0.0)

    def stats(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame | None = None,
    ) -> dict:
        """
        Return a dict of portfolio-level statistics.

        Parameters
        ----------
        weights : output of build()
        returns : bar-level pct_change panel aligned to weights.index (optional)
                  If supplied, realised PnL statistics are included.
        """
        gross = weights.abs().sum(axis=1)
        net   = weights.sum(axis=1)
        n_pos = (weights.abs() > self.min_weight).sum(axis=1)
        to    = self.turnover(weights)

        out = {
            "avg_gross_lev"    : gross.mean(),
            "avg_net_lev"      : net.abs().mean(),
            "avg_n_positions"  : n_pos.mean(),
            "avg_turnover_bar" : to.mean(),
            "annual_turnover"  : to.mean() * self.bars_per_year,
        }

        if returns is not None:
            # Bar-level portfolio return = sum(w_{t-1} * r_t)
            port_ret = (weights.shift(1) * returns).sum(axis=1).dropna()
            ann_ret  = port_ret.mean() * self.bars_per_year
            ann_vol  = port_ret.std()  * np.sqrt(self.bars_per_year)
            sharpe   = ann_ret / ann_vol if ann_vol > 0 else float("nan")
            out.update({
                "gross_annual_return" : ann_ret,
                "gross_annual_vol"    : ann_vol,
                "gross_sharpe"        : sharpe,
            })

        return out

    def portfolio_stats(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame | None = None,
    ) -> dict:
        """
        Backwards-compatible wrapper used by run_alpha.py.

        Maps the keys from `stats()` to the names expected by the
        higher-level script (`gross_return_ann`, `gross_vol_ann`,
        `avg_positions`, etc.) so calling code does not have to change.
        """
        s = self.stats(weights, returns)

        mapped = {
            # returns mapping
            "gross_return_ann": s.get("gross_annual_return", float("nan")),
            "gross_vol_ann":    s.get("gross_annual_vol", float("nan")),
            "gross_sharpe":     s.get("gross_sharpe", float("nan")),

            # turnover / positions / leverage
            "annual_turnover":  s.get("annual_turnover", float("nan")),
            "avg_positions":    s.get("avg_n_positions", s.get("avg_positions", float("nan"))),
            "avg_gross_lev":    s.get("avg_gross_lev", float("nan")),
            "avg_net_lev":      s.get("avg_net_lev", float("nan")),
        }

        return mapped

    def annual_turnover(self, weights: pd.DataFrame) -> float:
        """Return a single annualised turnover value (×NAV per year)."""
        return self.turnover(weights).mean() * self.bars_per_year