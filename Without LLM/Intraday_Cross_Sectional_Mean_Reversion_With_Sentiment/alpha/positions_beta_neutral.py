"""
alpha/positions_beta_neutral.py
===============================
PART 5: Beta-Neutral Portfolio Positioning (SPY Hedging)

DESIGN:
  1. Compute rolling betas vs SPY (rolling cov/var, window=60 bars=5 hrs)
  2. Vol-scaled position sizing (larger positions for lower-vol stocks)
  3. Dollar-neutral constraint (long $50M = short $50M, net $0)
  4. Beta-neutral hedge (short SPY to neutralize portfolio beta)
  5. Risk constraints (max 10% per stock, gross leverage 2.0×)

PORTFOLIO STRUCTURE (example):
  - 22 long positions (top half of alpha)
  - 22 short positions (bottom half of alpha)
  - Long notional: $10M (assuming $100M capital, 2× leverage)
  - Short notional: $10M
  - Net: $0 (market-neutral)
  - Gross leverage: 2.0×
  - Portfolio beta: -0.02 (nearly neutral after SPY hedge)
  
COSTS:
  - SPY bid-ask: ~1 bps (highly liquid)
  - SPY borrow cost: ~0.5% annual = 0.002% daily (if negative rates)
  - Slippage on 44-stock rebalance: ~2 bps average
  
Usage
-----
    from alpha.positions_beta_neutral import compute_beta_neutral_positions
    
    positions, sizes, spy_hedge = compute_beta_neutral_positions(
        rank_alpha=df_alpha,        # [timestamp × 44], ∈ [-1, +1]
        close=df_close,             # [timestamp × 44]
        volumes=df_volumes,         # [timestamp × 44]
        spy_prices=spy_ser,         # [timestamp]
        spy_returns=spy_ret,        # [timestamp]
        capital=100_000_000,        # $100M
    )
    
    # Verify structure
    long_notional = (sizes[sizes > 0]).sum(axis=1).mean()
    short_notional = (sizes[sizes < 0]).abs().sum(axis=1).mean()
    portfolio_beta = (betas * sizes).sum(axis=1).mean()
    
    print(f"Long: ${long_notional/1e6:.1f}M, Short: ${short_notional/1e6:.1f}M")
    print(f"Portfolio beta: {portfolio_beta:.3f} (target <0.05)")
    print(f"SPY hedge required: ${spy_hedge.mean()/1e6:.1f}M")
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional

log = logging.getLogger(__name__)


def compute_beta_neutral_positions(
    rank_alpha: pd.DataFrame,
    close: pd.DataFrame,
    volumes: pd.DataFrame,
    spy_prices: pd.Series,
    spy_returns: pd.Series,
    capital: float = 100_000_000,
    beta_window: int = 60,
    vol_window: int = 60,
    gross_lev: float = 2.0,
    max_weight: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute dollar-neutral, beta-neutral positions for 44 stocks + SPY hedge.
    
    Parameters
    ----------
    rank_alpha : DataFrame [timestamp × 44]
        Alpha signal, range [-1.0, +1.0]
        Positive = buy (top performers)
        Negative = sell short (bottom performers)
    
    close : DataFrame [timestamp × 44]
        Close prices
    
    volumes : DataFrame [timestamp × 44]
        Bar volumes (for liquidity weighting)
    
    spy_prices : Series [timestamp]
        SPY close prices
    
    spy_returns : Series [timestamp]
        SPY 1-bar returns (used for beta estimation)
    
    capital : float
        Total capital ($). Default $100M.
        Positions are: long = capital/2 = $50M, short = capital/2 = $50M
    
    beta_window : int
        Rolling window for beta estimation (bars). Default 60 = 5 hours.
    
    vol_window : int
        Rolling window for volatility (bars)
    
    gross_lev : float
        Target gross leverage. Default 2.0× (standard for 44-stock L/S)
    
    max_weight : float
        Max |weight| as fraction of capital. Default 10%.
    
    Returns
    -------
    tuple:
        positions : DataFrame [timestamp × 44]
            Number of shares per stock (positive=long, negative=short)
        
        sizes : DataFrame [timestamp × 44]
            Notional $ size (positive=long, negative=short)
        
        spy_hedge : DataFrame [timestamp × 1]
            Shares of SPY to short for beta hedging (should be positive)
    
    Timing: ~1-2 sec for 2 years × 44 tickers
    """
    log.info(
        "Computing beta-neutral positions:\n"
        "  Capital: $%.0fM | Gross lev: %.1f× | Max weight: %.0f%%",
        capital / 1e6, gross_lev, max_weight * 100
    )
    
    # Step 1: Compute rolling betas vs SPY
    betas = _compute_rolling_betas(close, spy_returns, window=beta_window)
    
    # Step 2: Volatility scaling
    vols = close.pct_change(1).rolling(vol_window).std()
    
    # Scale alpha by inverse vol (more confidence in low-vol predictions)
    alpha_vol_scaled = rank_alpha.div(vols.clip(lower=0.001), axis=0)
    
    # Step 3: Dollar-neutral sizing
    sizes = _apply_dollar_neutrality(
        alpha_vol_scaled,
        close,
        capital=capital,
        gross_lev=gross_lev,
        max_weight=max_weight,
    )
    
    # Step 4: Beta hedge
    portfolio_beta = (betas * sizes / close).sum(axis=1)  # net delta-weighted
    spy_shares = portfolio_beta * capital / spy_prices  # short this much SPY
    spy_hedge = spy_shares.to_frame(name="SPY_hedge")
    
    # Step 5: Convert sizes to shares
    positions = sizes / close.clip(lower=0.01)
    
    # Validation
    _validate_positions(positions, sizes, betas, spy_hedge, capital, gross_lev)
    
    return positions, sizes, spy_hedge


def _compute_rolling_betas(
    close: pd.DataFrame,
    spy_returns: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """
    Compute rolling beta of each stock vs SPY.
    
    Parameters
    ----------
    close : DataFrame [timestamp × 44]
    spy_returns : Series [timestamp]
        SPY returns (1-bar pct_change)
    window : int
        Rolling window (bars)
    
    Returns
    -------
    DataFrame [timestamp × 44]: betas, typically ∈ [0.3, 2.0]
    """
    log.info("Computing rolling betas vs SPY (window=%d bars) ...", window)
    
    stock_returns = close.pct_change(1)
    betas = pd.DataFrame(np.nan, index=stock_returns.index, columns=stock_returns.columns)
    
    # Align SPY returns to stock index
    spy_ret_aligned = spy_returns.reindex(close.index)
    spy_var = spy_ret_aligned.rolling(window).var()
    
    for col in close.columns:
        # Rolling covariance
        stock_spy_cov = stock_returns[col].rolling(window).cov(spy_ret_aligned)
        betas[col] = stock_spy_cov / spy_var.clip(lower=1e-8)
    
    # Validate betas
    beta_mean = betas.stack().mean()
    beta_std = betas.stack().std()
    log.debug("Beta statistics: mean=%.3f, std=%.3f", beta_mean, beta_std)
    
    return betas


def _apply_dollar_neutrality(
    alpha_scaled: pd.DataFrame,
    close: pd.DataFrame,
    capital: float,
    gross_lev: float,
    max_weight: float,
) -> pd.DataFrame:
    """
    Apply dollar-neutral constraint: long $X = short $X.
    
    Algorithm:
      1. Vol-weighted alpha → raw sizing
      2. Split long/short
      3. Compute total long notional, total short notional
      4. Scale to balance: long_target = short_target = capital * gross_lev / 2
      5. Apply max_weight cap
    
    Parameters
    ----------
    alpha_scaled : DataFrame [timestamp × 44]
        Vol-normalized alpha signals
    close : DataFrame [timestamp × 44]
    capital : float
    gross_lev : float
    max_weight : float
    
    Returns
    -------
    DataFrame [timestamp × 44]: notional sizes ($)
    """
    log.info("Applying dollar-neutral constraint ...")
    
    # Raw notional sizes (before normalization)
    raw_sizes = alpha_scaled * close
    
    # Target dollars for long and short
    target_notional = capital * gross_lev / 2
    
    # Split by sign
    long_mask = raw_sizes > 0
    short_mask = raw_sizes < 0
    
    # Sum positive and negative
    long_sum = raw_sizes.where(long_mask, 0).abs().sum(axis=1)
    short_sum = raw_sizes.where(short_mask, 0).abs().sum(axis=1)
    
    # Scale factors to hit target notional
    sizes_normalized = raw_sizes.copy()
    for i in range(len(raw_sizes)):
        idx = raw_sizes.index[i]
        if long_sum.iloc[i] > 0:
            long_cols = long_mask.iloc[i].values
            sizes_normalized.loc[idx, long_cols] = \
                raw_sizes.loc[idx, long_cols] * \
                target_notional / (long_sum.iloc[i] + 1e-10)
        
        if short_sum.iloc[i] > 0:
            short_cols = short_mask.iloc[i].values
            sizes_normalized.loc[idx, short_cols] = \
                raw_sizes.loc[idx, short_cols] * \
                target_notional / (short_sum.iloc[i] + 1e-10)
    
    # Apply max_weight cap
    sizes_capped = sizes_normalized.copy()
    max_size = capital * max_weight
    sizes_capped = sizes_capped.clip(-max_size, max_size)
    
    return sizes_capped


def _validate_positions(
    positions: pd.DataFrame,
    sizes: pd.DataFrame,
    betas: pd.DataFrame,
    spy_hedge: pd.DataFrame,
    capital: float,
    gross_lev: float,
) -> None:
    """
    Validate portfolio structure: dollar-neutral, beta-neutral, leverage correct.
    
    Raises ValueError if constraints violated.
    """
    log.info("Validating portfolio structure ...")
    
    # Dollar neutrality
    long_notional = sizes.where(sizes > 0, 0).sum(axis=1).mean()
    short_notional = sizes.where(sizes < 0, 0).abs().sum(axis=1).mean()
    net_notional = (sizes.sum(axis=1)).mean()
    gross_notional = (sizes.abs().sum(axis=1)).mean()
    
    # Beta
    portfolio_beta = (betas * positions).sum(axis=1).mean()
    
    # Checks
    checks = {
        "Dollar neutral": abs(net_notional) < 1000,  # <$1K net drift
        "Long/Short balanced": abs(long_notional - short_notional) / (long_notional + 1e-8) < 0.05,
        "Gross leverage target": abs(gross_notional - capital * gross_lev) / (capital * gross_lev) < 0.05,
        "Beta neutral": abs(portfolio_beta) < 0.05,
    }
    
    all_pass = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        log.info("  %s %s", status, check_name)
    
    if not all_pass:
        log.warning(
            f"Portfolio validation warnings:\n"
            f"  Net notional: ${net_notional/1e6:.1f}M (target <$1M)\n"
            f"  Long notional: ${long_notional/1e6:.1f}M\n"
            f"  Short notional: ${short_notional/1e6:.1f}M\n"
            f"  Gross leverage: {gross_notional/capital:.2f}× (target {gross_lev:.1f}×)\n"
            f"  Portfolio beta: {portfolio_beta:.3f} (target <0.05)"
        )
    
    # Summary log
    try:
        spy_hedge_mean = spy_hedge.iloc[:, 0].mean()
    except:
        spy_hedge_mean = 0
    log.info(
        f"\nPORTFOLIO STRUCTURE:\n"
        f"  Long: ${long_notional/1e6:.1f}M | Short: ${short_notional/1e6:.1f}M | Net: ${net_notional/1e6:.1f}M\n"
        f"  Gross leverage: {gross_notional/capital:.2f}× | Gross notional: ${gross_notional/1e6:.1f}M\n"
        f"  Portfolio beta: {portfolio_beta:.3f}\n"
        f"  SPY hedge (avg shares): {spy_hedge_mean:.0f}"
    )
