"""
features/
=========
Feature engineering pipeline for NASDAQ intraday cross-sectional mean reversion.

Modules
-------
core.py        — cross-sectional math primitives (cs_zscore, cs_rank, ATR, etc.)
engine.py      — 7 alpha features (6 core + momentum exhaustion)
resampler.py   — frequency resampling
diagnostics.py — IC analysis, decay curves, correlation
store.py       — Parquet feature store (save/load)

Quick start
-----------
    from features.engine import FeatureEngine

    engine = FeatureEngine(panels)
    features = engine.compute_all()
    signal = features["A1_bar_reversal"]
"""
from features.core import cs_zscore, cs_rank, atr, atr_pct
from features.engine import FeatureEngine
from features.diagnostics import FeatureDiagnostics
from features.store import FeatureStore
