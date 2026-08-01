"""
features/diagnostics.py
========================
Feature quality diagnostics for intraday cross-sectional mean reversion.

Key outputs
-----------
  IC (Information Coefficient)   — Spearman correlation of signal with forward return
  ICIR                           — IC mean / IC std (signal consistency)
  IC Decay curve                 — how fast does signal lose predictive power?
  Signal statistics              — mean, std, skew, kurtosis per feature
  Turnover                       — how much does the signal change bar to bar?
  Feature correlation matrix     — detect redundant signals

Usage
-----
    from features.diagnostics import FeatureDiagnostics
    diag = FeatureDiagnostics(features, panels["close"])
    report = diag.full_report()
    diag.print_summary(report)
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# Suppress scipy ConstantInputWarning when computing correlation of constant signals
warnings.filterwarnings("ignore", message="An input array is constant")

log = logging.getLogger(__name__)


class FeatureDiagnostics:
    """
    Runs IC analysis, decay curves, and signal quality checks on all features.

    Parameters
    ----------
    features : dict of {name: DataFrame}  — output of FeatureEngine.compute_all()
    close    : DataFrame  [timestamp × ticker]  — close price panel
    """

    def __init__(
        self,
        features: dict[str, pd.DataFrame],
        close:    pd.DataFrame,
    ):
        self.features = features
        self.close    = close

    # ── Information Coefficient ───────────────────────────────────────────────

    def ic_series(
        self,
        feature_name: str,
        forward_bars: int = 1,
    ) -> pd.Series:
        """
        Compute bar-by-bar IC (Spearman rank correlation between signal at t
        and forward return at t+forward_bars).

        IC > 0.02 = weakly predictive (typical for intraday)
        IC > 0.05 = strongly predictive

        Parameters
        ----------
        feature_name  : key in self.features dict
        forward_bars  : how many bars ahead to measure return
        """
        signal  = self.features[feature_name]
        fwd_ret = self.close.pct_change(forward_bars).shift(-forward_bars)

        ic_vals = []
        for ts in signal.index:
            sig_row = signal.loc[ts].dropna()
            ret_row = fwd_ret.loc[ts].dropna()
            common  = sig_row.index.intersection(ret_row.index)
            if len(common) < 3:
                ic_vals.append(np.nan)
                continue
            corr, _ = stats.spearmanr(sig_row[common], ret_row[common])
            ic_vals.append(corr)

        return pd.Series(ic_vals, index=signal.index, name=f"IC_{feature_name}")

    def ic_summary(
        self,
        feature_name: str,
        forward_bars: int = 1,
    ) -> dict:
        """
        Compute IC statistics for one feature at one forward horizon.
        Returns mean IC, IC std, ICIR, t-stat, and % positive bars.
        """
        ic = self.ic_series(feature_name, forward_bars).dropna()
        if len(ic) == 0:
            return {}

        n      = len(ic)
        mean   = ic.mean()
        std    = ic.std()
        icir   = mean / std if std > 0 else 0
        tstat  = mean / (std / np.sqrt(n)) if std > 0 else 0
        pct_pos = (ic > 0).mean()

        return {
            "feature":      feature_name,
            "forward_bars": forward_bars,
            "n_bars":       n,
            "IC_mean":      round(mean, 5),
            "IC_std":       round(std, 5),
            "ICIR":         round(icir, 3),
            "t_stat":       round(tstat, 3),
            "pct_positive": round(pct_pos, 3),
        }

    def ic_decay(
        self,
        feature_name: str,
        max_lead:     int = 20,
    ) -> pd.DataFrame:
        """
        IC at multiple forward horizons (1 to max_lead bars).
        Shows how quickly the signal's predictive power decays.

        For a mean-reversion signal:
          - IC should be positive at lead=1 (signal predicts next bar)
          - IC should decay toward 0 by lead=10-15
          - IC may go slightly negative (momentum zone) before reverting to 0

        Returns DataFrame with columns: lead, IC_mean, IC_std, ICIR
        """
        rows = []
        for lead in range(1, max_lead + 1):
            summary = self.ic_summary(feature_name, forward_bars=lead)
            if summary:
                rows.append(summary)
        return pd.DataFrame(rows).set_index("forward_bars") if rows else pd.DataFrame()

    # ── Signal Statistics ─────────────────────────────────────────────────────

    def signal_stats(self) -> pd.DataFrame:
        """
        Descriptive statistics for each feature.
        Flags: very low std (dead signal), extreme skew (outlier issue),
               very high turnover (noisy signal).
        """
        rows = []
        for name, feat in self.features.items():
            vals  = feat.values.flatten()
            vals  = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue

            # Turnover: mean absolute change bar to bar
            diff  = feat.diff().abs()
            turnover = diff.mean().mean()

            rows.append({
                "feature":     name,
                "mean":        round(float(np.mean(vals)), 4),
                "std":         round(float(np.std(vals)),  4),
                "skew":        round(float(stats.skew(vals)), 3),
                "kurtosis":    round(float(stats.kurtosis(vals)), 3),
                "pct_null":    round(float(np.isnan(feat.values).mean()), 4),
                "turnover_bpb":round(float(turnover), 4),  # bar-per-bar turnover
            })
        return pd.DataFrame(rows).set_index("feature") if rows else pd.DataFrame()

    # ── Feature Correlation ───────────────────────────────────────────────────

    def feature_correlation(self) -> pd.DataFrame:
        """
        Cross-feature correlation matrix (time × ticker stacked).
        High correlation (> 0.7) between two features = redundant signal.
        Keep only one, or blend carefully.
        """
        stacked = {}
        for name, feat in self.features.items():
            flat = feat.stack(future_stack=True).dropna()
            stacked[name] = flat

        combined = pd.DataFrame(stacked).dropna()
        return combined.corr(method="spearman").round(3)

    # ── Full Report ───────────────────────────────────────────────────────────

    def full_report(
        self,
        forward_bars: int = 1,
        ic_decay_bars: int = 15,
    ) -> dict:
        """
        Run all diagnostics and return a comprehensive report dict.

        Keys:
          "ic_summary"    — IC stats for each feature at forward_bars lead
          "ic_decay"      — IC decay curve for best feature
          "signal_stats"  — descriptive stats for all features
          "correlation"   — cross-feature Spearman correlation
        """
        log.info("Running feature diagnostics ...")

        # IC summary for all features
        ic_rows = []
        best_ic = -np.inf
        best_feature = None
        log.debug("Computing IC summary for %d features", len(self.features))
        for name in self.features:
            summary = self.ic_summary(name, forward_bars)
            if summary:
                ic_rows.append(summary)
                log.debug("  %s: IC=%.4f, ICIR=%.3f, t_stat=%.3f", 
                         name, summary["IC_mean"], summary["ICIR"], summary["t_stat"])
                if summary["IC_mean"] > best_ic:
                    best_ic      = summary["IC_mean"]
                    best_feature = name
        
        ic_summary_df = pd.DataFrame(ic_rows).set_index("feature") if ic_rows else pd.DataFrame()
        
        if best_feature:
            log.info("✓ Best IC feature: %s (IC=%.4f, ICIR=%.3f)", 
                    best_feature, best_ic, ic_summary_df.loc[best_feature, "ICIR"])

        # IC decay for the best-IC feature
        decay_df = pd.DataFrame()
        if best_feature:
            log.info("Computing IC decay for: %s", best_feature)
            decay_df = self.ic_decay(best_feature, max_lead=ic_decay_bars)
            log.debug("IC decay computed: %.4f → %.4f over %d bars", 
                     decay_df.iloc[0]["IC_mean"], decay_df.iloc[-1]["IC_mean"], ic_decay_bars)

        # Signal stats
        log.info("Computing signal statistics...")
        stats_df = self.signal_stats()
        
        # Warn about dead signals
        if not stats_df.empty:
            dead = stats_df[stats_df["std"] < 0.01]
            if not dead.empty:
                log.warning("⚠ Low-std signals (potential dead signals): %s", 
                           ", ".join(dead.index.tolist()))
            
            high_turnover = stats_df[stats_df["turnover_bpb"] > 0.5]
            if not high_turnover.empty:
                log.warning("⚠ High bar-to-bar turnover (noisy): %s", 
                           ", ".join(high_turnover.index.tolist()))

        # Correlation
        log.info("Computing feature correlation matrix...")
        corr_df = self.feature_correlation()
        
        # Detect redundant pairs
        if not corr_df.empty:
            redundant = []
            for i, col1 in enumerate(corr_df.columns):
                for col2 in corr_df.columns[i+1:]:
                    if abs(corr_df.loc[col1, col2]) > 0.7:
                        redundant.append((col1, col2, corr_df.loc[col1, col2]))
            if redundant:
                log.warning("⚠ Redundant feature pairs (corr > 0.7):")
                for f1, f2, corr in redundant[:5]:
                    log.warning("    %s ↔ %s (corr=%.3f)", f1, f2, corr)

        report = {
            "ic_summary":   ic_summary_df,
            "ic_decay":     decay_df,
            "signal_stats": stats_df,
            "correlation":  corr_df,
            "best_feature": best_feature,
        }
        log.info("Diagnostics complete. Report keys: %s", list(report.keys()))
        return report

    def print_summary(self, report: dict) -> None:
        """Print a formatted summary of the diagnostic report to stdout."""
        sep = "─" * 65

        print(f"\n{sep}")
        print("  FEATURE DIAGNOSTICS SUMMARY")
        print(sep)

        print("\n📊 IC Summary (1-bar forward return):")
        ic = report.get("ic_summary")
        if ic is not None and not ic.empty:
            print(ic[["IC_mean", "IC_std", "ICIR", "t_stat", "pct_positive"]]
                  .sort_values("IC_mean", ascending=False)
                  .to_string())

        print(f"\n📉 IC Decay — {report.get('best_feature', 'N/A')}:")
        decay = report.get("ic_decay")
        if decay is not None and not decay.empty:
            print(decay[["IC_mean", "ICIR"]].to_string())

        print("\n📐 Signal Statistics:")
        stats_df = report.get("signal_stats")
        if stats_df is not None and not stats_df.empty:
            print(stats_df[["mean", "std", "skew", "pct_null", "turnover_bpb"]]
                  .to_string())

        print(f"\n{sep}\n")

    def save_report(self, report: dict, output_dir: str = "reports/") -> None:
        """Save all report DataFrames to CSV files."""
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        log.info("Saving diagnostic reports to %s", out)
        saved_count = 0

        for key, df in report.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filepath = out / f"feature_{key}.csv"
                df.to_csv(filepath)
                file_size = filepath.stat().st_size / 1024  # KB
                log.info("  ✓ feature_%s.csv (%.1f KB, %d rows)", key, file_size, len(df))
                saved_count += 1
            elif isinstance(df, pd.DataFrame) and df.empty:
                log.debug("  (skipped empty: feature_%s.csv)", key)
            elif isinstance(df, str):
                log.debug("  (skipped string: %s = %s)", key, df)
        
        log.info("Reports saved: %d files", saved_count)
