"""
alpha/signal.py
===============
Alpha Signal Construction — Phase 2

Takes the 15 features from the feature engine and combines them into
ONE composite alpha score per stock per bar.

Architecture (how a systematic fund builds this):

  Step 1 — IC estimation
    For each feature, compute rolling Spearman IC vs 1-bar forward return.
    IC = correlation(signal_t, return_t+1) across all stocks at each bar.
    Rolling window: 60 bars (15 hours of data).

  Step 2 — IC-weighted combination
    Composite alpha = Σ (IC_k / Σ|IC_k|) × Feature_k
    Features with higher recent IC get higher weight.
    Features with near-zero or negative IC get near-zero weight.
    This is automatically adaptive: if a feature stops working, its weight
    drops toward zero without manual intervention.

  Step 3 — Signal decay adjustment
    Raw alpha decays exponentially with holding period.
    Model: alpha_t+h = alpha_t × exp(-λ × h)
    Estimate λ from IC decay curve: find h where IC(h) = 0.5 × IC(1).
    Scale position sizes inversely: larger position when decay is slow.

  Step 4 — Final normalisation
    Cross-sectional z-score at every bar: mean=0, std=1 across all stocks.
    Winsorise at ±3σ: prevents one extreme stock from dominating.
    This is the signal passed to the portfolio optimizer.

Output: pd.DataFrame [timestamp × ticker], values ∈ [-3, +3]
  Positive = expect this stock to go UP (buy)
  Negative = expect this stock to go DOWN (sell short)

Usage
-----
    from alpha.signal import AlphaModel
    model   = AlphaModel(features, forward_returns)
    alpha   = model.composite_alpha()
    weights = model.ic_weights()
    decay   = model.signal_decay_halflife()
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from features.core import cs_zscore, cs_rank, rolling_mean, rolling_std

# Suppress scipy ConstantInputWarning when computing correlation of constant signals
warnings.filterwarnings("ignore", message="An input array is constant")

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# IC utilities (used both in AlphaModel and diagnostics)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ic_series(
    signal:       pd.DataFrame,
    fwd_return:   pd.DataFrame,
    min_stocks:   int = 5,
) -> pd.Series:
    """
    Bar-by-bar Spearman IC between signal at t and forward return at t+1.

    Spearman (rank correlation) is preferred over Pearson because:
      - It is robust to outlier stocks
      - Returns and signals are not normally distributed intraday
      - It measures whether ranking is correct, not magnitude

    Parameters
    ----------
    signal      : DataFrame [timestamp × ticker] — the feature signal
    fwd_return  : DataFrame [timestamp × ticker] — already shifted by 1 bar
    min_stocks  : minimum tickers needed for a valid IC computation

    Returns
    -------
    pd.Series indexed by timestamp. NaN where insufficient data.
    """
    ic_vals = np.full(len(signal), np.nan)
    sig_arr = signal.values
    ret_arr = fwd_return.values

    for i in range(len(signal)):
        sig_row = sig_arr[i]
        ret_row = ret_arr[i]
        valid   = ~(np.isnan(sig_row) | np.isnan(ret_row))
        if valid.sum() < min_stocks:
            continue
        corr, _ = spearmanr(sig_row[valid], ret_row[valid])
        if not np.isnan(corr):
            ic_vals[i] = corr

    return pd.Series(ic_vals, index=signal.index, name="IC")


def compute_ic_decay(
    signal:     pd.DataFrame,
    close:      pd.DataFrame,
    max_lead:   int = 15,
    min_stocks: int = 5,
) -> pd.DataFrame:
    """
    IC at multiple forward horizons (1 to max_lead bars).

    Reveals:
      - Signal half-life (where IC drops to 50% of IC at lead=1)
      - Whether signal is true reversal (positive then decaying to zero)
      - Whether there is a momentum zone (IC goes negative before zero)

    Returns DataFrame: index=lead, columns=[IC_mean, IC_std, ICIR, t_stat]
    """
    # Align signal and close to common index and columns
    common_idx = signal.index.intersection(close.index)
    common_cols = signal.columns.intersection(close.columns)
    signal = signal.reindex(index=common_idx, columns=common_cols)
    close = close.reindex(index=common_idx, columns=common_cols)
    
    rows = []
    ret_1bar = close.pct_change(1)

    for lead in range(1, max_lead + 1):
        fwd_ret = ret_1bar.shift(-lead)
        ic      = compute_ic_series(signal, fwd_ret, min_stocks)
        ic_clean = ic.dropna()
        if len(ic_clean) < 10:
            continue
        n      = len(ic_clean)
        mu     = ic_clean.mean()
        sig    = ic_clean.std()
        icir   = mu / sig if sig > 0 else 0.0
        tstat  = mu / (sig / np.sqrt(n)) if sig > 0 else 0.0
        rows.append({
            "lead":    lead,
            "IC_mean": round(mu,   5),
            "IC_std":  round(sig,  5),
            "ICIR":    round(icir, 3),
            "t_stat":  round(tstat, 3),
            "n_bars":  n,
        })

    return pd.DataFrame(rows).set_index("lead") if rows else pd.DataFrame()


def estimate_halflife(ic_decay: pd.DataFrame) -> float:
    """
    Estimate signal half-life in bars from IC decay DataFrame.
    Half-life = lead at which IC drops to 50% of IC at lead=1.
    Returns np.inf if IC never decays to 50%.
    """
    if ic_decay.empty or "IC_mean" not in ic_decay.columns:
        return np.inf
    ic1 = ic_decay.loc[1, "IC_mean"] if 1 in ic_decay.index else np.nan
    if np.isnan(ic1) or ic1 <= 0:
        return np.inf
    threshold = ic1 * 0.5
    below = ic_decay[ic_decay["IC_mean"] <= threshold]
    if below.empty:
        return np.inf
    return float(below.index[0])


# ─────────────────────────────────────────────────────────────────────────────
# AlphaModel
# ─────────────────────────────────────────────────────────────────────────────

class AlphaModel:
    """
    IC-weighted composite alpha signal from multiple features.

    Parameters
    ----------
    features    : dict {name: DataFrame[timestamp × ticker]}
                  Output of FeatureEngine.compute_all()
    close       : DataFrame[timestamp × ticker] — close prices
    ic_window   : rolling bars to estimate IC weights (default 60 = 15 hrs)
    min_ic_tstat: features with |IC t-stat| below this are given zero weight
    decay_window: bars to estimate signal decay half-life
    """

    def __init__(
        self,
        features:     dict[str, pd.DataFrame],
        close:        pd.DataFrame,
        ic_window:    int   = 300,
        min_ic_tstat: float = 1.0,
        decay_window: int   = 300,
    ):
        # Align features and close to common index AND columns (tickers)
        all_indices = [close.index] + [df.index for df in features.values()]
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)
        
        all_columns = [close.columns] + [df.columns for df in features.values()]
        common_columns = all_columns[0]
        for cols in all_columns[1:]:
            common_columns = common_columns.intersection(cols)
        
        self.close = close.reindex(index=common_index, columns=common_columns)
        self.features = {name: df.reindex(index=common_index, columns=common_columns) for name, df in features.items()}
        
        self.ic_window    = ic_window
        self.min_ic_tstat = min_ic_tstat
        self.decay_window = decay_window

        # 1-bar forward return — used for IC estimation
        self._fwd_ret = self.close.pct_change(1).shift(-1)

        log.info(
            "AlphaModel: %d features | IC window=%d bars | min t-stat=%.1f",
            len(self.features), ic_window, min_ic_tstat,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def composite_alpha(self) -> pd.DataFrame:
        """
        Compute the final composite alpha signal.

        Pipeline:
          1. Rolling IC per feature
          2. IC-weighted combination
          3. Cross-sectional z-score + winsorise at ±3σ

        Returns DataFrame [timestamp × ticker] ∈ [-3, +3].
        Positive = buy, Negative = sell short.
        """
        log.info("Computing composite alpha ...")

        weights = self.ic_weights()           # {name: rolling IC weight Series}
        alpha   = self._weighted_combine(weights)
        alpha   = cs_zscore(alpha, clip=3.0)  # final normalisation

        n_valid = alpha.notna().values.mean() * 100
        log.info(
            "Composite alpha: shape=%s  valid=%.1f%%  "
            "mean=%.4f  std=%.4f",
            alpha.shape, n_valid,
            float(alpha.stack().mean()) if not alpha.empty else 0,
            float(alpha.stack().std())  if not alpha.empty else 0,
        )
        return alpha

    def ic_weights(self) -> dict[str, pd.Series]:
        """
        Compute rolling IC weight for every feature.

        Weight(k, t) = rolling_IC(k, t) / Σ|rolling_IC(k, t)|

        Features with negative or near-zero rolling IC get ~0 weight
        automatically. This makes the model self-correcting — if market
        regime shifts and a feature stops working, its weight collapses.

        Returns dict {feature_name: pd.Series[timestamp → weight]}
        """
        # Step 1: compute rolling IC for every feature
        raw_ic = {}
        for name, feat in self.features.items():
            ic_series = compute_ic_series(feat, self._fwd_ret, min_stocks=5)
            # Smooth IC with rolling mean to reduce noise (shifted by 1 to prevent lookahead)
            raw_ic[name] = (
                ic_series.shift(1)
                .rolling(self.ic_window, min_periods=max(5, self.ic_window // 4))
                .mean()
                .fillna(0.0)
            )

        # Step 2: compute rolling IC t-stat for each feature
        ic_tstat = {}
        for name, feat in self.features.items():
            ic_series = compute_ic_series(feat, self._fwd_ret, min_stocks=5)
            ic_shifted = ic_series.shift(1)
            ic_mu  = ic_shifted.rolling(self.ic_window, min_periods=10).mean()
            ic_sig = ic_shifted.rolling(self.ic_window, min_periods=10).std()
            n_obs  = ic_shifted.rolling(self.ic_window, min_periods=10).count()
            tstat  = ic_mu / (ic_sig / np.sqrt(n_obs)).replace(0.0, np.nan)
            ic_tstat[name] = tstat.fillna(0.0)

        # Step 3: zero out features below min IC t-stat threshold
        for name in raw_ic:
            below_threshold = ic_tstat[name].abs() < self.min_ic_tstat
            raw_ic[name] = raw_ic[name].where(~below_threshold, 0.0)

        # Step 4: normalise weights to sum to 1 at each bar
        ic_df      = pd.DataFrame(raw_ic)
        sum_abs_ic = ic_df.abs().sum(axis=1).replace(0.0, np.nan)
        weights_df = ic_df.div(sum_abs_ic, axis=0).fillna(0.0)

        log.info("IC weights computed. Active features per bar (avg): %.1f",
                 (weights_df.abs() > 0.01).sum(axis=1).mean())

        return {col: weights_df[col] for col in weights_df.columns}

    def static_ic_weights(self) -> dict[str, float]:
        """
        Compute static (full-sample) IC weights for reporting.
        More stable than rolling — use for strategy analysis and reporting.
        Not used in live trading (rolling weights are used there).
        """
        ic_summary = {}
        for name, feat in self.features.items():
            ic = compute_ic_series(feat, self._fwd_ret, min_stocks=5).dropna()
            if len(ic) < 10:
                ic_summary[name] = 0.0
                continue
            mu    = ic.mean()
            sig   = ic.std()
            n     = len(ic)
            tstat = mu / (sig / np.sqrt(n)) if sig > 0 else 0.0
            # Only use features with significant IC
            ic_summary[name] = mu if abs(tstat) >= self.min_ic_tstat else 0.0

        # Normalise
        total = sum(abs(v) for v in ic_summary.values())
        if total == 0:
            return {k: 1.0 / len(ic_summary) for k in ic_summary}
        return {k: v / total for k, v in ic_summary.items()}

    def ic_summary_table(self) -> pd.DataFrame:
        """
        Full IC summary table for every feature.
        Columns: IC_mean, IC_std, ICIR, t_stat, pct_positive, weight
        This is the table you review to understand signal quality.
        """
        rows = []
        static_weights = self.static_ic_weights()

        for name, feat in self.features.items():
            ic = compute_ic_series(feat, self._fwd_ret, min_stocks=5).dropna()
            if len(ic) < 5:
                continue
            mu     = ic.mean()
            sig    = ic.std()
            n      = len(ic)
            icir   = mu / sig if sig > 0 else 0.0
            tstat  = mu / (sig / np.sqrt(n)) if sig > 0 else 0.0
            rows.append({
                "feature":      name,
                "IC_mean":      round(mu,    5),
                "IC_std":       round(sig,   5),
                "ICIR":         round(icir,  3),
                "t_stat":       round(tstat, 3),
                "pct_positive": round((ic > 0).mean(), 3),
                "n_bars":       n,
                "weight":       round(static_weights.get(name, 0.0), 4),
                "active":       abs(tstat) >= self.min_ic_tstat,
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("feature")
        return df.sort_values("t_stat", ascending=False)

    def signal_decay_halflife(self) -> dict[str, float]:
        """
        Estimate IC half-life for every feature.
        Returns {feature_name: halflife_in_bars}.
        Half-life = bars until IC drops to 50% of IC at 1-bar lead.
        This tells you: how long can you hold a position before the edge is gone?
        """
        halflives = {}
        for name, feat in self.features.items():
            decay = compute_ic_decay(feat, self.close, max_lead=15)
            halflives[name] = estimate_halflife(decay)
        return halflives

    def composite_decay_halflife(self) -> float:
        """Half-life of the composite alpha signal."""
        alpha  = self.composite_alpha()
        decay  = compute_ic_decay(alpha, self.close, max_lead=15)
        return estimate_halflife(decay)

    # ── Private ───────────────────────────────────────────────────────────────

    def _weighted_combine(
        self,
        weights: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Combine features using time-varying IC weights.

        Key insight: weights already carry the IC sign.
        A feature with negative IC gets a negative weight, which
        effectively flips the feature direction before combining.

        alpha(i,t) = Σ_k  w_k(t) × feature_k(i,t)

        Since w_k(t) = IC_k / Σ|IC_k|, a feature with IC=-0.5 gets
        weight=-0.5/total, which negates that feature's contribution.
        This means the composite correctly combines all features.
        """
        ref_idx  = self.close.index
        ref_cols = self.close.columns
        weighted = pd.DataFrame(0.0, index=ref_idx, columns=ref_cols)

        for name, feat in self.features.items():
            w_series  = weights.get(name, pd.Series(0.0, index=ref_idx))
            w_aligned = w_series.reindex(ref_idx).fillna(0.0)
            # Skip features with near-zero weight (saves compute)
            if w_aligned.abs().max() < 1e-6:
                continue
            feat_aligned  = feat.reindex(index=ref_idx, columns=ref_cols)
            contribution  = feat_aligned.mul(w_aligned, axis=0)
            weighted      = weighted.add(contribution.fillna(0.0))

        return weighted