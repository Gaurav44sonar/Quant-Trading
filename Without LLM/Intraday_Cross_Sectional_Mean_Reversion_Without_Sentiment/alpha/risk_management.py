"""
alpha/risk_management.py
========================
Risk management for NASDAQ intraday mean reversion.

Handles:
  1. Extreme move detection (stock-level intraday limits)
  2. Market-wide stress detection (QQQ/NASDAQ ±3% rolling)
  3. Position-level risk limits
  4. Drawdown-based position scaling
  5. Time-of-day position adjustments

Usage
-----
    from alpha.risk_management import RiskManager
    
    rm = RiskManager(capital=100_000)
    weights = rm.apply(weights, close, qqq_returns)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class RiskManager:
    """
    Risk controls for NASDAQ intraday mean reversion.
    
    Parameters
    ----------
    capital : float
        Total capital in USD (default $100K)
    max_single_stock_pct : float
        Max allocation to single stock (default 5%)
    max_drawdown_pct : float
        Max portfolio drawdown before reducing positions (default 5%)
    extreme_move_buffer : float
        Reduce position when stock has an extreme intraday move (default 8%)
    """
    
    def __init__(
        self,
        capital: float = 100_000,
        max_single_stock_pct: float = 0.05,
        max_drawdown_pct: float = 0.05,
        extreme_move_buffer: float = 0.08,
    ):
        self.capital = capital
        self.max_stock_pct = max_single_stock_pct
        self.max_dd = max_drawdown_pct
        self.extreme_buffer = extreme_move_buffer
    
    def apply(
        self,
        weights: pd.DataFrame,
        close: pd.DataFrame,
        nifty_returns: pd.Series = None,
    ) -> pd.DataFrame:
        """
        Apply all risk controls to portfolio weights.
        
        Parameters
        ----------
        weights : DataFrame [timestamp × ticker] — raw weights
        close : DataFrame [timestamp × ticker] — prices
        nifty_returns : Series — NASDAQ index (QQQ) bar returns
        
        Returns
        -------
        DataFrame: risk-adjusted weights
        """
        w = weights.copy()
        
        # 1. Extreme intraday move detection
        w = self._extreme_move_scale(w, close)
        
        # 2. Market-wide stress detection
        if nifty_returns is not None:
            w = self._market_stress_check(w, nifty_returns)
        
        # 3. Position concentration limits
        w = self._concentration_limit(w)
        
        # 4. Drawdown-based scaling
        w = self._drawdown_scale(w, close)
        
        return w
    
    def _extreme_move_scale(
        self,
        weights: pd.DataFrame,
        close: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Reduce positions for stocks with extreme intraday moves.
        
        US markets have no per-stock circuit breakers like NSE,
        but stocks with extreme session moves (>8%) tend to be
        unpredictable. Reduce exposure by 50%.
        """
        # Session return from day's open
        dates = close.index.normalize()
        session_open = close.groupby(dates).transform("first")
        session_ret = (close - session_open) / session_open.replace(0, np.nan)
        
        # Check if move exceeds extreme threshold
        extreme_mask = session_ret.abs() > self.extreme_buffer
        
        # Align to weights index
        extreme_aligned = extreme_mask.reindex(weights.index).fillna(False)
        
        # Scale down positions with extreme moves
        scale = pd.DataFrame(1.0, index=weights.index, columns=weights.columns)
        scale[extreme_aligned] = 0.5
        
        n_scaled = extreme_aligned.sum().sum()
        if n_scaled > 0:
            log.info("  Extreme move: reduced %d positions by 50%%", n_scaled)
        
        return weights * scale
    
    def _market_stress_check(
        self,
        weights: pd.DataFrame,
        qqq_returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Reduce all positions during market-wide stress.
        
        If QQQ/NASDAQ index moves > ±3% in a rolling 30-bar window,
        reduce all positions by 50%.
        """
        qqq_aligned = qqq_returns.reindex(weights.index)
        rolling_qqq = qqq_aligned.rolling(30, min_periods=5).sum().abs()
        
        stress_mask = rolling_qqq > 0.03
        
        scale = pd.Series(1.0, index=weights.index)
        scale[stress_mask] = 0.5
        
        n_stress = stress_mask.sum()
        if n_stress > 0:
            log.info("  Market stress: reduced positions for %d bars", n_stress)
        
        return weights.mul(scale, axis=0)
    
    def _concentration_limit(
        self,
        weights: pd.DataFrame,
    ) -> pd.DataFrame:
        """Hard clip individual stock weights at max_stock_pct."""
        return weights.clip(-self.max_stock_pct, self.max_stock_pct)
    
    def _drawdown_scale(
        self,
        weights: pd.DataFrame,
        close: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Scale down positions during portfolio drawdowns.
        
        If cumulative portfolio return drops below -max_dd,
        reduce position sizes linearly (min 30% of original).
        """
        # Compute simple portfolio equity proxy
        ret = close.pct_change(1)
        port_ret = (weights.shift(1) * ret.reindex(weights.index)).sum(axis=1)
        cum_ret = (1 + port_ret.fillna(0)).cumprod()
        
        running_max = cum_ret.expanding().max()
        drawdown = (cum_ret - running_max) / running_max
        
        # Scale: 1.0 at dd=0, 0.3 at dd=-max_dd, 0.3 below
        scale = (1.0 + drawdown / self.max_dd * 0.7).clip(0.3, 1.0)
        
        if (scale < 1.0).any():
            log.info("  Drawdown scaling active: min scale=%.2f", scale.min())
        
        return weights.mul(scale, axis=0)
