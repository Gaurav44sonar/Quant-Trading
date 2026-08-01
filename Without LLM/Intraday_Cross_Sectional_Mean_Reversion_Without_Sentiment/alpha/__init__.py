"""
alpha/
======
Phase 2: Alpha Signal Construction + Portfolio Construction.

Modules
-------
signal.py                — IC-weighted composite alpha from features
portfolio.py             — Dollar-neutral, vol-scaled position weights
rank_alpha.py            — Rank-based alpha construction
positions_beta_neutral.py — Beta-neutral positioning with NASDAQ index (QQQ)
regularized_zscore.py    — Regularized z-score for signals
risk_management.py       — Extreme move + drawdown risk controls

Quick start
-----------
    from alpha.signal    import AlphaModel
    from alpha.portfolio import PortfolioBuilder

    model   = AlphaModel(features, close, ic_window=120, min_ic_tstat=0.5)
    alpha   = model.composite_alpha()
    
    builder = PortfolioBuilder(alpha, close, volume, bars_per_year=19656)
    weights = builder.build()
"""
from alpha.signal    import AlphaModel, compute_ic_series, compute_ic_decay
from alpha.portfolio import PortfolioBuilder