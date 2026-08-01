"""
alpha/execution.py
==================
Intraday Execution Engine for Single-Day Max Profit (NASDAQ)

Simulates realistic intraday position management with:
  - Entry at 9:45 AM ET (bar 4, after confirmation window)
  - Hard stop-loss at -2.0% per position (wider for US volatility)
  - Trailing stop after +2.5% profit
  - Profit taking at +1.5%, +3%, +4.5% levels
  - Mandatory time exit at 3:50 PM ET (bar 76, 10 min before close)

Usage
-----
    from alpha.execution import IntradayExecutor
    
    executor = IntradayExecutor(stop_loss_pct=0.02)
    result = executor.simulate_day(picks, intraday_bars)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single intraday trade."""
    ticker: str
    shares: int
    entry_price: float
    entry_time: pd.Timestamp
    entry_bar: int
    exit_price: float = 0.0
    exit_time: pd.Timestamp = None
    exit_bar: int = -1
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    max_gain_pct: float = 0.0
    max_loss_pct: float = 0.0
    
    @property
    def capital_deployed(self) -> float:
        return self.shares * self.entry_price
    
    def close(self, price: float, time: pd.Timestamp, bar: int, reason: str):
        self.exit_price = price
        self.exit_time = time
        self.exit_bar = bar
        self.exit_reason = reason
        self.pnl = (price - self.entry_price) * self.shares
        self.pnl_pct = (price / self.entry_price) - 1
    
    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "entry_time": str(self.entry_time),
            "exit_price": self.exit_price,
            "exit_time": str(self.exit_time),
            "exit_reason": self.exit_reason,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "max_gain_pct": self.max_gain_pct,
            "max_loss_pct": self.max_loss_pct,
        }


class IntradayExecutor:
    """
    Simulates intraday execution with risk controls.
    
    US Market Stop-Loss Adjustments vs NSE:
      - Wider hard stop (2.0% vs 1.5%) — US stocks have wider intraday ranges
      - Wider trailing trigger (2.5% vs 2.0%) — prevents premature activation
      - Wider trailing distance (0.75% vs 0.5%) — accounts for US tick noise
      - Wider profit targets (1.5/3/4.5% vs 1/2/3%) — US moves are larger
    
    Parameters
    ----------
    stop_loss_pct : float
        Hard stop-loss percentage (default 2.0% = 0.02)
    trailing_stop_trigger : float
        Profit level to activate trailing stop (default 2.5% = 0.025)
    trailing_stop_pct : float
        Trailing stop distance (default 0.75% = 0.0075)
    profit_take_1 : float
        First profit-taking level (default 1.5% = 0.015). Take 25% off.
    profit_take_2 : float
        Second profit-taking level (default 3% = 0.03). Take 25% off.
    profit_take_3 : float
        Third profit-taking level (default 4.5% = 0.045). Take 25% off.
    exit_bar : int
        Bar number to force exit (default 76 = 3:50 PM ET, 10 min before close)
    """
    
    def __init__(
        self,
        stop_loss_pct: float = 0.02,
        trailing_stop_trigger: float = 0.025,
        trailing_stop_pct: float = 0.0075,
        profit_take_1: float = 0.01,
        profit_take_2: float = 0.03,
        profit_take_3: float = 0.045,
        pt_weights: tuple[float, float, float] = (0.25, 0.25, 0.25),
        exit_bar: int = 76,
    ):
        self.stop_loss = stop_loss_pct
        self.trail_trigger = trailing_stop_trigger
        self.trail_pct = trailing_stop_pct
        self.pt1 = profit_take_1
        self.pt2 = profit_take_2
        self.pt3 = profit_take_3
        self.pt_weights = pt_weights
        self.exit_bar = exit_bar
    
    def simulate_day(
        self,
        picks: list[dict],
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        date_mask: pd.Series,
    ) -> dict:
        """
        Simulate one full trading day.
        
        Parameters
        ----------
        picks : list of stock pick dicts from StockPicker
        close : full close DataFrame
        high : full high DataFrame
        low : full low DataFrame
        date_mask : boolean mask for today's bars
        
        Returns
        -------
        dict with:
            trades: list[Trade]
            day_pnl: float (total P&L in USD)
            day_pnl_pct: float (P&L as % of deployed capital)
            capital_deployed: float
            n_stopped: int
            n_profit_taken: int
            n_time_exit: int
        """
        today_close = close[date_mask]
        today_high = high[date_mask]
        today_low = low[date_mask]
        
        if len(today_close) == 0 or not picks:
            return self._empty_result()
        
        n_bars = len(today_close)
        bar_times = today_close.index
        
        trades = []
        
        for pick in picks:
            ticker = pick["ticker"]
            if ticker not in today_close.columns:
                continue
            
            entry_bar = pick.get("entry_bar_idx", 3)
            if entry_bar >= n_bars:
                entry_bar = 0
            
            entry_price = pick["entry_price"]
            shares = pick["shares"]
            
            if shares <= 0 or entry_price <= 0:
                continue
            
            trade = Trade(
                ticker=ticker,
                shares=shares,
                entry_price=entry_price,
                entry_time=bar_times[entry_bar],
                entry_bar=entry_bar,
            )
            
            # Simulate bar-by-bar from entry
            self._simulate_trade(
                trade, today_close, today_high, today_low, bar_times
            )
            
            trades.append(trade)
        
        # Compute summary
        total_pnl = sum(t.pnl for t in trades)
        total_capital = sum(t.capital_deployed for t in trades)
        
        result = {
            "trades": trades,
            "day_pnl": total_pnl,
            "day_pnl_pct": total_pnl / total_capital if total_capital > 0 else 0,
            "capital_deployed": total_capital,
            "n_stopped": sum(1 for t in trades if t.exit_reason == "STOP_LOSS"),
            "n_profit_taken": sum(1 for t in trades if "PROFIT" in t.exit_reason),
            "n_trailing": sum(1 for t in trades if t.exit_reason == "TRAILING_STOP"),
            "n_time_exit": sum(1 for t in trades if t.exit_reason == "TIME_EXIT"),
        }
        
        return result
    
    def _simulate_trade(
        self,
        trade: Trade,
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        bar_times: pd.DatetimeIndex,
    ):
        """
        Simulate a single trade bar-by-bar with all risk controls.
        """
        ticker = trade.ticker
        entry = trade.entry_price
        remaining_shares = trade.shares
        high_water = entry  # Highest price seen for trailing stop
        
        # Profit-taking tracking
        pt1_done = False
        pt2_done = False
        pt3_done = False
        partial_exits = 0.0  # Accumulated partial P&L
        
        for bar_idx in range(trade.entry_bar + 1, len(close)):
            bar_high = high.iloc[bar_idx][ticker]
            bar_low = low.iloc[bar_idx][ticker]
            bar_close = close.iloc[bar_idx][ticker]
            bar_time = bar_times[bar_idx]
            
            if pd.isna(bar_close) or bar_close <= 0:
                continue
            
            # Update high water mark
            high_water = max(high_water, bar_high)
            
            # Track max gain/loss
            current_pct = (bar_close / entry) - 1
            trade.max_gain_pct = max(trade.max_gain_pct, (bar_high / entry) - 1)
            trade.max_loss_pct = min(trade.max_loss_pct, (bar_low / entry) - 1)
            
            # ── CHECK 1: Hard Stop-Loss ──────────────────────────────────
            if bar_low <= entry * (1 - self.stop_loss):
                stop_price = entry * (1 - self.stop_loss)
                trade.close(stop_price, bar_time, bar_idx, "STOP_LOSS")
                return
            
            # ── CHECK 2: Trailing Stop ───────────────────────────────────
            if high_water >= entry * (1 + self.trail_trigger):
                trail_price = high_water * (1 - self.trail_pct)
                if bar_low <= trail_price:
                    trade.close(trail_price, bar_time, bar_idx, "TRAILING_STOP")
                    return
            
            # ── CHECK 3: Profit Taking (partial exits) ───────────────────
            if not pt1_done and bar_high >= entry * (1 + self.pt1):
                pt1_done = True
                pt_shares = int(trade.shares * self.pt_weights[0])
                if pt_shares > 0:
                    partial_exits += pt_shares * (entry * (1 + self.pt1) - entry)
                    remaining_shares -= pt_shares
            
            if not pt2_done and bar_high >= entry * (1 + self.pt2):
                pt2_done = True
                pt_shares = int(trade.shares * self.pt_weights[1])
                if pt_shares > 0:
                    partial_exits += pt_shares * (entry * (1 + self.pt2) - entry)
                    remaining_shares -= pt_shares
            
            if not pt3_done and bar_high >= entry * (1 + self.pt3):
                pt3_done = True
                pt_shares = int(trade.shares * self.pt_weights[2])
                if pt_shares > 0:
                    partial_exits += pt_shares * (entry * (1 + self.pt3) - entry)
                    remaining_shares -= pt_shares
            
            # ── CHECK 4: Time Exit ───────────────────────────────────────
            if bar_idx >= self.exit_bar:
                # Close remaining shares at market
                final_pnl = remaining_shares * (bar_close - entry) + partial_exits
                trade.pnl = final_pnl
                trade.pnl_pct = final_pnl / trade.capital_deployed
                trade.exit_price = bar_close
                trade.exit_time = bar_time
                trade.exit_bar = bar_idx
                trade.exit_reason = "PROFIT_TIME_EXIT" if partial_exits > 0 else "TIME_EXIT"
                return
        
        # If we reach end of data without exiting (shouldn't happen)
        last_close = close.iloc[-1][ticker]
        final_pnl = remaining_shares * (last_close - entry) + partial_exits
        trade.pnl = final_pnl
        trade.pnl_pct = final_pnl / trade.capital_deployed
        trade.exit_price = last_close
        trade.exit_time = bar_times[-1]
        trade.exit_bar = len(close) - 1
        trade.exit_reason = "PROFIT_END_OF_DATA" if partial_exits > 0 else "END_OF_DATA"
    
    def _empty_result(self) -> dict:
        return {
            "trades": [],
            "day_pnl": 0.0,
            "day_pnl_pct": 0.0,
            "capital_deployed": 0.0,
            "n_stopped": 0,
            "n_profit_taken": 0,
            "n_trailing": 0,
            "n_time_exit": 0,
        }
