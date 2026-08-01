"""
run_live.py
===========
Live & Paper Trading Execution Pipeline forNASDAQ/NSE Mean-Reversion Strategy

Now supports both US and Indian markets, automatic multi-universe execution,
fail-safe report generation, and isolated crash recovery.
"""

from __future__ import annotations

import os
import sys
import json
import copy
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import time
import argparse
import logging
from datetime import datetime, time as dt_time, timedelta
import pytz

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Alpaca Imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Custom modules
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from alpha.stock_picker import StockPicker
from features.daily_signals import DailySignalEngine
from nse_pipeline.universe import SEED_POOL
from nse_pipeline.universe_nse import NSE_SEED_POOL
from sentiment.engine import SentimentDecisionEngine

# Constants
EST = pytz.timezone('US/Eastern')
SEP = "=" * 70
THIN = "-" * 70

# ── Execution Parameters (single source of truth) ────────────────────────
# Used by both get_expected_state_so_far() and the live monitoring loop.
# Update values HERE ONLY to keep crash-recovery and live logic in sync.
EXEC_PARAMS = {
    # ── Stop-Loss Tiers (tightened based on live performance analysis) ──
    "sl_tier1_pct": 0.01,       # 1.0% stop → sell 50% (was 2.5%, never triggered)
    "sl_tier2_pct": 0.02,       # 2.0% stop → sell 25% (was 5.0%)
    "sl_tier3_pct": 0.035,      # 3.5% stop → sell remaining (was 10.0%)
    "sl_tier1_weight": 0.50,
    "sl_tier2_weight": 0.25,
    "sl_tier3_weight": 0.25,
    
    # ── Trailing Stop ──
    "trail_trigger": 0.025,     # 2.5% trailing stop trigger (was 4.0%)
    "trail_pct": 0.0075,
    
    # ── Profit Targets (fixed fallback) ──
    "profit_take_1": 0.015,
    "profit_take_2": 0.03,
    "profit_take_3": 0.045,
    
    # ── Profit Targets (ATR-based) ──
    "atr_pt_1": 0.25,           # 0.25 ATR
    "atr_pt_2": 0.50,           # 0.50 ATR
    "atr_pt_3": 1.00,           # 1.00 ATR
    "pt_weight_1": 0.50,
    "pt_weight_2": 0.25,
    "pt_weight_3": 0.25,
    
    # ── Time-Based Partial Exits (new: captures MFE before MAE) ──
    "time_exit_1_minutes": 60,   # At 60 min, sell 50% if profitable
    "time_exit_1_sell_pct": 0.50,
    "time_exit_2_minutes": 120,  # At 120 min, sell 75% of remaining if profitable
    "time_exit_2_sell_pct": 0.75,
    
    # ── Market Stress Detection (from RiskManager) ──
    "market_stress_threshold": 0.03,  # If index drops >3% intraday, flatten all
    
    # ── Extreme Move Detection (from RiskManager) ──
    "extreme_move_pct": 0.08,   # If stock moves >8% intraday, close position
    
    # ── Drawdown-Based Portfolio Scaling ──
    "max_portfolio_drawdown": 0.03,  # If portfolio drops >3% from peak, flatten all
}

log = logging.getLogger("run_live")


def setup_logging(log_level: str = "INFO", log_file: str = "live_logs/pipeline.log") -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def get_unique_filepath(directory: str, filename_prefix: str, timestamp: str, extension: str = ".txt") -> str:
    filename = f"{filename_prefix}_{timestamp}{extension}"
    filepath = os.path.join(directory, filename)
    counter = 1
    while os.path.exists(filepath):
        filename = f"{filename_prefix}_{timestamp}_{counter}{extension}"
        filepath = os.path.join(directory, filename)
        counter += 1
    return filepath


def fetch_yfinance_panels(
    tickers: list[str],
    index_ticker: str = "QQQ",
    lookback_days: int = 35,
    market_tz: pytz.BaseTzInfo = EST,
    market_open: dt_time = dt_time(9, 30),
    market_close: dt_time = dt_time(16, 0)
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """
    Fetch 5-minute bars from Yahoo Finance and build clean panels.
    Ensures correct alignment to index bars.
    """
    log.info(f"Fetching 5-minute bar data from Yahoo Finance for last {lookback_days} days...")
    
    all_symbols = list(set(tickers + [index_ticker]))
    
    if lookback_days <= 5:
        period_str = "5d"
    elif lookback_days <= 20:
        period_str = "1mo"
    else:
        period_str = "60d"
        
    try:
        df_hist = yf.download(
            tickers=all_symbols,
            period=period_str,
            interval="5m",
            auto_adjust=True,
            progress=False
        )
        if lookback_days > 5:
            df_today = yf.download(
                tickers=all_symbols,
                period="1d",
                interval="5m",
                auto_adjust=True,
                progress=False
            )
            if df_today is not None and not df_today.empty:
                df = pd.concat([df_hist, df_today])
                df = df[~df.index.duplicated(keep='last')].sort_index()
            else:
                df = df_hist
        else:
            df = df_hist
    except Exception as e:
        log.error(f"Yahoo Finance download failed: {e}")
        raise e
        
    if df is None or df.empty:
        raise ValueError("No bar data returned from Yahoo Finance.")
        
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(market_tz)
    else:
        df.index = pd.to_datetime(df.index).tz_convert(market_tz)
    
    # Filter for standard market hours
    market_mask = (df.index.time >= market_open) & (df.index.time < market_close)
    df = df[market_mask]
    
    close_panel = df["Close"].copy()
    open_panel = df["Open"].copy()
    high_panel = df["High"].copy()
    low_panel = df["Low"].copy()
    volume_panel = df["Volume"].copy()
    
    if index_ticker not in close_panel.columns:
        raise ValueError(f"Market index {index_ticker} not found in downloaded Yahoo Finance bars.")
        
    index_close = close_panel[index_ticker]
    
    # Drop index ticker from main panels
    for p in [close_panel, open_panel, high_panel, low_panel, volume_panel]:
        if index_ticker in p.columns:
            p.drop(columns=[index_ticker], inplace=True)
            
    # Forward-fill gaps
    close_panel = close_panel.ffill()
    open_panel = open_panel.fillna(close_panel)
    high_panel = high_panel.fillna(close_panel)
    low_panel = low_panel.fillna(close_panel)
    volume_panel = volume_panel.fillna(0.0)
    
    # Reindex to match index bars exactly
    common_idx = index_close.dropna().index
    if len(common_idx) == 0:
        raise ValueError(f"No valid {index_ticker} benchmark index bars downloaded.")
        
    close_panel = close_panel.reindex(common_idx).ffill()
    open_panel = open_panel.reindex(common_idx).fillna(close_panel)
    high_panel = high_panel.reindex(common_idx).fillna(close_panel)
    low_panel = low_panel.reindex(common_idx).fillna(close_panel)
    volume_panel = volume_panel.reindex(common_idx).fillna(0.0)
    
    valid_tickers = close_panel.dropna(how='all', axis=1).columns
    if len(valid_tickers) == 0:
        raise ValueError("No valid price data downloaded for any of the watchlist tickers.")
    
    panels = {
        "close": close_panel[valid_tickers],
        "open": open_panel[valid_tickers],
        "high": high_panel[valid_tickers],
        "low": low_panel[valid_tickers],
        "volume": volume_panel[valid_tickers]
    }
    
    log.info(f"Loaded panels from Yahoo Finance for {len(valid_tickers)} tickers + {index_ticker} index across {len(common_idx)} bars.")
    return panels, index_close


def classify_tickers(
    panels: dict[str, pd.DataFrame],
    lookback_days: int | None = None
) -> tuple[list[str], list[str], pd.Series]:
    """
    Compute daily ATR% for each ticker from 5-minute panels resampled to daily.
    Classify them into volatile (top half) and non-volatile (bottom half) groups.
    """
    close = panels["close"]
    high = panels["high"]
    low = panels["low"]
    
    # Resample 5-min data to daily
    daily_close = close.resample("D").last().dropna(how="all")
    daily_high = high.resample("D").max().reindex(daily_close.index)
    daily_low = low.resample("D").min().reindex(daily_close.index)
    
    if lookback_days is not None:
        daily_close = daily_close.iloc[-lookback_days:]
        daily_high = daily_high.reindex(daily_close.index)
        daily_low = daily_low.reindex(daily_close.index)
        
    atr_window = min(20, len(daily_close))
    if atr_window <= 0:
        atr_window = 1
    
    prev_close = daily_close.shift(1)
    tr1 = daily_high - daily_low
    tr2 = (daily_high - prev_close).abs()
    tr3 = (daily_low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
    
    atr = tr.rolling(window=atr_window, min_periods=min(5, len(tr))).mean()
    atr_pct = (atr / daily_close.replace(0, np.nan)).iloc[-1]
    
    atr_ranked = atr_pct.dropna().sort_values(ascending=False)
    
    tickers = atr_ranked.index.tolist()
    n = len(tickers)
    
    n_vol = (n + 1) // 2
    volatile = tickers[:n_vol]
    nonvolatile = tickers[n_vol:]
    
    return volatile, nonvolatile, atr_ranked


def compute_watchlist_atr(panels: dict[str, pd.DataFrame], today_date: pd.Timestamp) -> dict[str, float]:
    """
    Calculate the daily ATR(14) for each ticker using data strictly before today_date.
    """
    atr_dict = {}
    close = panels["close"]
    high = panels["high"]
    low = panels["low"]
    
    # Exclude today's data to prevent lookahead bias
    hist_mask = close.index < today_date
    if not hist_mask.any():
        return atr_dict
        
    for ticker in close.columns:
        try:
            df_hist = pd.DataFrame({
                "High": high[ticker].loc[hist_mask],
                "Low": low[ticker].loc[hist_mask],
                "Close": close[ticker].loc[hist_mask]
            }).dropna()
            
            if len(df_hist) == 0:
                continue
                
            # Resample to daily
            daily = df_hist.resample('D').agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
            
            if len(daily) > 0:
                tr1 = daily["High"] - daily["Low"]
                tr2 = (daily["High"] - daily["Close"].shift(1)).abs()
                tr3 = (daily["Low"] - daily["Close"].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                if len(daily) >= 14:
                    atr_series = tr.rolling(14).mean()
                    atr_val = atr_series.iloc[-1]
                else:
                    atr_val = tr.mean()
                
                if not pd.isna(atr_val) and atr_val > 0:
                    atr_dict[ticker] = float(atr_val)
        except Exception as e:
            log.warning(f"Could not compute ATR for {ticker} in live pipeline: {e}")
            
    return atr_dict


def get_expected_state_so_far(
    picks: list[dict],
    panels: dict[str, pd.DataFrame],
    today_date: pd.Timestamp,
    time_entry: dt_time,
    time_flatten: dt_time
) -> dict[str, dict]:
    """
    Simulate trade entries and management up to the current bar for today.
    Allows recovery of tracking state if the script restarts mid-day.
    """
    close = panels["close"]
    high = panels["high"]
    low = panels["low"]
    
    dates = close.index.normalize()
    day_mask = dates == today_date
    
    today_close = close[day_mask]
    today_high = high[day_mask]
    today_low = low[day_mask]
    
    n_bars = len(today_close)
    tracking = {}
    
    entry_bar = 3
    exit_bar = 76
    if n_bars > 0:
        times_list = today_close.index.time
        for idx, t in enumerate(times_list):
            if t >= time_entry:
                entry_bar = idx
                break
        for idx, t in enumerate(times_list):
            if t >= time_flatten:
                exit_bar = idx
                break
                
    entry_time_ts = today_close.index[entry_bar] if len(today_close) > entry_bar else today_date

    # Precalculate ATR for the picks to prevent lookahead bias
    atr_values = compute_watchlist_atr(panels, today_date)

    # Unpack execution parameters from single source of truth (EXEC_PARAMS)
    sl_tier1_pct = EXEC_PARAMS["sl_tier1_pct"]
    sl_tier2_pct = EXEC_PARAMS["sl_tier2_pct"]
    sl_tier3_pct = EXEC_PARAMS["sl_tier3_pct"]
    sl_tier1_weight = EXEC_PARAMS["sl_tier1_weight"]
    sl_tier2_weight = EXEC_PARAMS["sl_tier2_weight"]
    sl_tier3_weight = EXEC_PARAMS["sl_tier3_weight"]
    trail_trigger = EXEC_PARAMS["trail_trigger"]
    trail_pct = EXEC_PARAMS["trail_pct"]
    profit_take_1 = EXEC_PARAMS["profit_take_1"]
    profit_take_2 = EXEC_PARAMS["profit_take_2"]
    profit_take_3 = EXEC_PARAMS["profit_take_3"]
    atr_pt_1 = EXEC_PARAMS["atr_pt_1"]
    atr_pt_2 = EXEC_PARAMS["atr_pt_2"]
    atr_pt_3 = EXEC_PARAMS["atr_pt_3"]
    pt_weight_1 = EXEC_PARAMS["pt_weight_1"]
    pt_weight_2 = EXEC_PARAMS["pt_weight_2"]
    pt_weight_3 = EXEC_PARAMS["pt_weight_3"]
    
    for pick in picks:
        ticker = pick["ticker"]
        entry_price = pick["entry_price"]
        initial_qty = pick["shares"]
        
        # Get ATR value for this stock
        atr_val = atr_values.get(ticker, None)
        
        state = {
            "ticker": ticker,
            "group": pick.get("group", "unclassified"),
            "entry_price": entry_price,
            "initial_qty": initial_qty,
            "qty": initial_qty,
            "high_water": entry_price,
            "atr_value": atr_val,
            "sl1_done": False,
            "sl2_done": False,
            "sl3_done": False,
            "pt1_done": False,
            "pt2_done": False,
            "pt3_done": False,
            "sl_exited": 0.0,
            "time_exit_1_done": False,
            "time_exit_2_done": False,
            "active": True,
            "exit_reason": "",
            "exits": [],
            "entry_time": entry_time_ts
        }
        
        if n_bars <= entry_bar + 1:
            tracking[ticker] = state
            continue
            
        for bar_idx in range(entry_bar + 1, n_bars):
            if not state["active"]:
                break
                
            if ticker not in today_high.columns or ticker not in today_low.columns or ticker not in today_close.columns:
                continue
                
            bar_high = today_high.iloc[bar_idx][ticker]
            bar_low = today_low.iloc[bar_idx][ticker]
            bar_close = today_close.iloc[bar_idx][ticker]
            bar_time = today_high.index[bar_idx]
            
            if pd.isna(bar_close) or bar_close <= 0:
                continue
                
            state["high_water"] = max(state["high_water"], bar_high)
            
            # 1. Tiered Stop-Loss
            if not state["sl1_done"] and bar_low <= entry_price * (1 - sl_tier1_pct):
                remaining = 1.0 - state["sl_exited"]
                if remaining > 0:
                    portion = min(1.0, sl_tier1_weight / remaining)
                    sold = int(state["qty"] * portion)
                    if sold > 0:
                        state["qty"] -= sold
                        state["exits"].append((sold, entry_price * (1 - sl_tier1_pct), bar_time))
                state["sl_exited"] += sl_tier1_weight
                state["sl1_done"] = True
                if state["qty"] <= 0:
                    state["active"] = False
                    state["exit_reason"] = "STOP_LOSS_TIER_1"
                    continue
                    
            if not state["sl2_done"] and bar_low <= entry_price * (1 - sl_tier2_pct):
                remaining = 1.0 - state["sl_exited"]
                if remaining > 0:
                    portion = min(1.0, sl_tier2_weight / remaining)
                    sold = int(state["qty"] * portion)
                    if sold > 0:
                        state["qty"] -= sold
                        state["exits"].append((sold, entry_price * (1 - sl_tier2_pct), bar_time))
                state["sl_exited"] += sl_tier2_weight
                state["sl2_done"] = True
                if state["qty"] <= 0:
                    state["active"] = False
                    state["exit_reason"] = "STOP_LOSS_TIER_2"
                    continue
                    
            if not state["sl3_done"] and bar_low <= entry_price * (1 - sl_tier3_pct):
                if state["qty"] > 0:
                    state["exits"].append((state["qty"], entry_price * (1 - sl_tier3_pct), bar_time))
                state["qty"] = 0
                state["sl3_done"] = True
                state["active"] = False
                state["exit_reason"] = "STOP_LOSS_TIER_3"
                continue
                
            # 2. Trailing Stop
            if state["high_water"] >= entry_price * (1 + trail_trigger):
                trail_price = state["high_water"] * (1 - trail_pct)
                if bar_low <= trail_price:
                    if state["qty"] > 0:
                        state["exits"].append((state["qty"], trail_price, bar_time))
                    state["qty"] = 0
                    state["active"] = False
                    state["exit_reason"] = "TRAILING_STOP"
                    continue
                    
            # 2.5 Extreme Move Detection (Stock-level limits)
            extreme_move_pct = EXEC_PARAMS.get("extreme_move_pct", 0.08)
            if extreme_move_pct is not None and extreme_move_pct > 0:
                sess_open = today_close.iloc[0][ticker] if len(today_close) > 0 else np.nan
                if pd.isna(sess_open) or sess_open <= 0:
                    sess_open = entry_price
                
                move_high = (bar_high - sess_open) / sess_open
                move_low = (bar_low - sess_open) / sess_open
                if abs(move_high) >= extreme_move_pct or abs(move_low) >= extreme_move_pct:
                    if state["qty"] > 0:
                        state["exits"].append((state["qty"], bar_close, bar_time))
                    state["qty"] = 0
                    state["active"] = False
                    state["exit_reason"] = "EXTREME_MOVE"
                    continue

            # 2.6 Time-Based Partial Exits
            elapsed_td = bar_time - entry_time_ts
            minutes_elapsed = elapsed_td.total_seconds() / 60.0
            
            time_exit_1_min = EXEC_PARAMS["time_exit_1_minutes"]
            time_exit_1_sell = EXEC_PARAMS["time_exit_1_sell_pct"]
            time_exit_2_min = EXEC_PARAMS["time_exit_2_minutes"]
            time_exit_2_sell = EXEC_PARAMS["time_exit_2_sell_pct"]

            if not state.setdefault("time_exit_1_done", False) and minutes_elapsed >= time_exit_1_min:
                if bar_close > entry_price:
                    qty_to_sell = int(state["qty"] * time_exit_1_sell)
                    if qty_to_sell > 0:
                        state["qty"] -= qty_to_sell
                        state["exits"].append((qty_to_sell, bar_close, bar_time))
                state["time_exit_1_done"] = True
                if state["qty"] <= 0:
                    state["active"] = False
                    state["exit_reason"] = "TIME_EXIT_1"
                    continue

            if not state.setdefault("time_exit_2_done", False) and minutes_elapsed >= time_exit_2_min:
                if bar_close > entry_price:
                    qty_to_sell = int(state["qty"] * time_exit_2_sell)
                    if qty_to_sell > 0:
                        state["qty"] -= qty_to_sell
                        state["exits"].append((qty_to_sell, bar_close, bar_time))
                state["time_exit_2_done"] = True
                if state["qty"] <= 0:
                    state["active"] = False
                    state["exit_reason"] = "TIME_EXIT_2"
                    continue
                    
            # 3. Partial Profit Taking (ATR-based or Fixed Fallback)
            if atr_pt_1 is not None and atr_val is not None:
                target_1 = entry_price + (atr_pt_1 * atr_val)
            else:
                target_1 = entry_price * (1 + profit_take_1)
                
            if atr_pt_2 is not None and atr_val is not None:
                target_2 = entry_price + (atr_pt_2 * atr_val)
            else:
                target_2 = entry_price * (1 + profit_take_2)
                
            if atr_pt_3 is not None and atr_val is not None:
                target_3 = entry_price + (atr_pt_3 * atr_val)
            else:
                target_3 = entry_price * (1 + profit_take_3)
            
            if not state["pt1_done"] and bar_high >= target_1:
                sold = int(initial_qty * pt_weight_1)
                qty_to_sell = min(sold, state["qty"])
                if qty_to_sell > 0:
                    state["exits"].append((qty_to_sell, target_1, bar_time))
                    state["qty"] -= qty_to_sell
                state["pt1_done"] = True
                if state["qty"] <= 0:
                    state["active"] = False
                    state["exit_reason"] = "PROFIT_TAKE_1"
                    continue
                    
            if not state["pt2_done"] and bar_high >= target_2:
                sold = int(initial_qty * pt_weight_2)
                qty_to_sell = min(sold, state["qty"])
                if qty_to_sell > 0:
                    state["exits"].append((qty_to_sell, target_2, bar_time))
                    state["qty"] -= qty_to_sell
                state["pt2_done"] = True
                if state["qty"] <= 0:
                    state["active"] = False
                    state["exit_reason"] = "PROFIT_TAKE_2"
                    continue
                    
            if not state["pt3_done"] and bar_high >= target_3:
                # Final target closes remaining
                qty_to_sell = state["qty"]
                if qty_to_sell > 0:
                    state["exits"].append((qty_to_sell, target_3, bar_time))
                    state["qty"] = 0
                state["pt3_done"] = True
                state["active"] = False
                state["exit_reason"] = "PROFIT_TAKE_3"
                continue
                    
            # 4. Mandatory Time Exit
            if bar_idx >= exit_bar:
                if state["qty"] > 0:
                    state["exits"].append((state["qty"], bar_close, bar_time))
                state["qty"] = 0
                state["active"] = False
                state["exit_reason"] = "TIME_EXIT"
                continue
                
        tracking[ticker] = state
        
    return tracking


def execute_order(
    trading_client: TradingClient,
    symbol: str,
    qty: int,
    side: OrderSide,
    dry_run: bool = False
) -> bool:
    """Submit a market order to Alpaca."""
    if qty <= 0:
        return False
        
    action_str = "BUY" if side == OrderSide.BUY else "SELL"
    log.info(f"[ORDER] {action_str} {qty} shares of {symbol}...")
    
    if dry_run:
        log.info(f"[DRY RUN] Order simulation complete. No real order sent.")
        return True
        
    try:
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(order_data=order_data)
        log.info(f"[ORDER SENT] Ticker: {symbol} | Status: {order.status} | ID: {order.id}")
        return True
    except Exception as e:
        log.error(f"[ORDER FAILED] Could not place {action_str} order for {symbol}: {e}")
        return False


def serialize_states(tracking_states):
    serialized = copy.deepcopy(tracking_states)
    for ticker, tstate in serialized.items():
        if "entry_time" in tstate:
            ts = tstate["entry_time"]
            tstate["entry_time"] = ts.isoformat() if isinstance(ts, datetime) else ts
        if "exits" in tstate:
            new_exits = []
            for item in tstate["exits"]:
                if len(item) == 3:
                    qty, price, ts = item
                    ts_str = ts.isoformat() if isinstance(ts, datetime) else ts
                    new_exits.append((qty, price, ts_str))
                else:
                    new_exits.append(item)
            tstate["exits"] = new_exits
    return serialized


def save_live_state(
    state_file_path: str,
    today_date: pd.Timestamp,
    time_entry: dt_time | None,
    time_flatten: dt_time | None,
    trades_entered: bool,
    picks: list[dict],
    tracking_states: dict[str, dict]
) -> None:
    """Save the live pipeline execution state to a JSON file."""
    try:
        serialized_states = serialize_states(tracking_states)
        serialized = {
            "today_date": today_date.strftime("%Y-%m-%d") if today_date is not None else None,
            "time_entry": time_entry.strftime("%H:%M:%S") if time_entry is not None else None,
            "time_flatten": time_flatten.strftime("%H:%M:%S") if time_flatten is not None else None,
            "trades_entered": trades_entered,
            "picks": picks,
            "tracking_states": serialized_states
        }
        with open(state_file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)
        log.info(f"[STATE] Saved execution state to {state_file_path}")
    except Exception as e:
        log.error(f"[STATE ERROR] Failed to save execution state: {e}")


def load_live_state(state_file_path: str, today_date: pd.Timestamp) -> dict | None:
    """Load the live pipeline execution state if it exists and matches today's date."""
    if not os.path.exists(state_file_path):
        return None
    try:
        with open(state_file_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        today_date_str = today_date.strftime("%Y-%m-%d")
        if state.get("today_date") != today_date_str:
            log.info(f"[STATE] Found state file from a different date ({state.get('today_date')}), ignoring.")
            return None
            
        if state.get("time_entry"):
            state["time_entry"] = datetime.strptime(state["time_entry"], "%H:%M:%S").time()
        if state.get("time_flatten"):
            state["time_flatten"] = datetime.strptime(state["time_flatten"], "%H:%M:%S").time()
            
        tracking_states = state.get("tracking_states", {})
        for ticker, tstate in tracking_states.items():
            if "entry_time" in tstate:
                ts_str = tstate["entry_time"]
                try:
                    tstate["entry_time"] = datetime.fromisoformat(ts_str) if ts_str else None
                except Exception:
                    pass
            if "exits" in tstate:
                parsed_exits = []
                for x in tstate["exits"]:
                    if len(x) == 3:
                        qty, price, ts_str = x
                        try:
                            ts = datetime.fromisoformat(ts_str) if ts_str else None
                        except Exception:
                            ts = ts_str
                        parsed_exits.append((qty, price, ts))
                    else:
                        parsed_exits.append(tuple(x))
                tstate["exits"] = parsed_exits
                
        log.info(f"[STATE] Loaded execution state for today from {state_file_path}")
        return state
    except Exception as e:
        log.error(f"[STATE ERROR] Failed to load execution state: {e}")
        return None


def clear_live_state(state_file_path: str) -> None:
    """Delete the persistent state file."""
    if os.path.exists(state_file_path):
        try:
            os.remove(state_file_path)
            log.info(f"[STATE] Cleared state file {state_file_path}")
        except Exception as e:
            log.error(f"[STATE ERROR] Failed to clear execution state file: {e}")


def calculate_portfolio_value(capital: float, tracking_states: dict, current_prices: dict) -> float:
    net_pnl = 0.0
    for ticker, state in tracking_states.items():
        entry_price = state["entry_price"]
        qty = state["qty"]
        initial_qty = state["initial_qty"]
        exits = state["exits"]
        
        exited_qty = sum(eqty for eqty, eprice, etime in exits)
        realized_revenue = sum(eqty * eprice for eqty, eprice, etime in exits)
        realized_cost = exited_qty * entry_price
        realized_pnl = realized_revenue - realized_cost
        
        unrealized_pnl = 0.0
        if qty > 0:
            current_price = current_prices.get(ticker, entry_price)
            unrealized_pnl = qty * (current_price - entry_price)
            
        net_pnl += (realized_pnl + unrealized_pnl)
        
    return capital + net_pnl


def graceful_shutdown(trading_client: TradingClient, tracking_states: dict, dry_run: bool) -> None:
    log.info("[SHUTDOWN] Executing graceful shutdown. Flattening all active positions...")
    try:
        if not dry_run:
            positions = trading_client.get_all_positions()
            for pos in positions:
                ticker = pos.symbol
                qty = int(float(pos.qty))
                if qty == 0:
                    continue
                log.info(f"Graceful shutdown: flattening broker position of {qty} shares of {ticker}")
                if qty > 0:
                    execute_order(trading_client, ticker, qty, OrderSide.SELL, dry_run)
                else:
                    execute_order(trading_client, ticker, abs(qty), OrderSide.BUY, dry_run)
    except Exception as e:
        log.error(f"Failed to retrieve broker positions for graceful shutdown: {e}")
        for ticker, state in tracking_states.items():
            if state.get("active") and state.get("qty", 0) > 0:
                log.info(f"Graceful shutdown fallback: flattening tracking position of {state['qty']} shares of {ticker}")
                execute_order(trading_client, ticker, state["qty"], OrderSide.SELL, dry_run)


def run_universe_session(
    market: str,
    universe_type: str,
    watchlist: list[str],
    trading_client: TradingClient,
    data_client: StockHistoricalDataClient,
    capital: float,
    n_picks: int,
    duration_hours: float = 6.08,
    dry_run: bool = False,
    run_once: bool = False,
    log_file: str = "live_logs/pipeline.log",
    report_save_path: str = "",
    run_timestamp: str = "",
    min_score: float = 0.8,
    min_basket_size: int = 8,
    momentum_mult: float | None = 2.0,
    momentum_threshold: float = 0.05,
    max_per_sector: int = 3,
    use_blacklist: bool = True,
) -> dict:
    """
    Executes a single session run for a specific universe category and returns computed metrics.
    Uses robust try/except/finally wrappers to ensure report generation.
    """
    log.info(SEP)
    log.info(f"  EXECUTION SESSION: Market={market.upper()} | Universe={universe_type.upper()} | Duration={duration_hours}h")
    log.info(SEP)
    
    sentiment_report = None
    
    # 1. Market Selection and Parameters
    if market == "us":
        market_tz = pytz.timezone('US/Eastern')
        index_ticker = "QQQ"
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        default_entry_time = dt_time(9, 45)
        safety_limit = dt_time(15, 55)
    else:  # india
        market_tz = pytz.timezone('Asia/Kolkata')
        index_ticker = "^NSEI"
        market_open = dt_time(9, 15)
        market_close = dt_time(15, 30)
        default_entry_time = dt_time(9, 30)
        safety_limit = dt_time(15, 25)
        
    state_file_path = os.path.join(ROOT, f"live_state_{market}_{universe_type}.json")
    
    # Local State Variables
    picks = []
    tracking_states = {}
    trades_entered = False
    time_entry = None
    time_flatten = None
    last_trade_date = None
    
    portfolio_history = []
    exposure_pcts = []
    runtime_errors = []
    api_issues = []
    termination_reason = "DURATION_COMPLETED"
    duration_completed = False
    manual_stop = False
    
    selection_timestamp = None
    volatile_list = []
    nonvolatile_list = []
    
    start_time = datetime.now(market_tz)
    
    # Pre-check Alpaca connection
    if not dry_run:
        try:
            account = trading_client.get_account()
            log.info(f"[ALPACA] Connected. Account Cash Available: ${float(account.cash):,.2f}")
        except Exception as e:
            log.error(f"[ALPACA] Connectivity check failed: {e}")
            api_issues.append(f"Alpaca connectivity check failed: {e}")
            termination_reason = "CONNECTION_FAILURE"
            # Fallback to dry run style or fail
            raise e
            
    try:
        log.info(f"Starting execution loop for {universe_type.upper()}...")
        while True:
            now = datetime.now(market_tz)
            current_time = now.time()
            
            # Weekday check
            if now.weekday() >= 5:
                log.info("Weekend detected. Execution suspended. Sleeping for 1 hour...")
                time.sleep(3600)
                continue
                
            today_date = pd.Timestamp(now.date()).tz_localize(market_tz)
            
            if last_trade_date is not None and today_date != last_trade_date:
                log.info(f"New trading day detected ({today_date.date()}). Exiting daily execution loop.")
                break
                
            last_trade_date = today_date
            
            # Lock timings
            if time_entry is None:
                saved_state = load_live_state(state_file_path, today_date)
                if saved_state is not None:
                    time_entry = saved_state["time_entry"]
                    time_flatten = saved_state["time_flatten"]
                    trades_entered = saved_state["trades_entered"]
                    picks = saved_state["picks"]
                    tracking_states = saved_state["tracking_states"]
                else:
                    now_time = now.time()
                    if now_time < market_open:
                        time_entry = default_entry_time
                        log.info(f"Script launched before market open. Scheduling entry for standard {time_entry.strftime('%H:%M:%S')} {market_tz}.")
                    else:
                        time_entry = now_time
                        log.info(f"Script launched during market hours. Initializing entry immediately at {time_entry.strftime('%H:%M:%S')} {market_tz}.")
                        
                    try:
                        entry_dt = datetime.combine(now.date(), time_entry)
                        calc_flatten_dt = entry_dt + timedelta(hours=duration_hours)
                        
                        safety_limit_dt = datetime.combine(now.date(), safety_limit)
                        
                        if calc_flatten_dt > safety_limit_dt:
                            time_flatten = safety_limit
                            log.info(f"Calculated exit time exceeds market hours safety limit. Capping exit at {time_flatten.strftime('%H:%M:%S')} {market_tz}.")
                        else:
                            time_flatten = calc_flatten_dt.time()
                    except Exception as e:
                        log.error(f"Error calculating flatten time: {e}")
                        raise e
                        
                    log.info(f"Execution schedule locked: Start/Entry={time_entry.strftime('%H:%M:%S')} | Flatten/Exit={time_flatten.strftime('%H:%M:%S')}")
                    save_live_state(state_file_path, today_date, time_entry, time_flatten, trades_entered, picks, tracking_states)
                    
            time_recovery_threshold = (datetime.combine(now.date(), time_entry) + timedelta(minutes=5)).time()
            time_market_close = market_close
            
            # 1. MORNING ENTRY STAGE
            if time_entry <= current_time < time_flatten and not trades_entered:
                log.info(f"\n[ENTRY PHASE] Market is open. Current Time: {now.strftime('%H:%M:%S')} {market_tz}")
                
                # Fetch data up to current bar for entire watchlist
                try:
                    panels, nifty_close = fetch_yfinance_panels(
                        tickers=watchlist,
                        index_ticker=index_ticker,
                        lookback_days=35,
                        market_tz=market_tz,
                        market_open=market_open,
                        market_close=market_close
                    )
                except Exception as e:
                    log.error(f"Failed to fetch market bars: {e}. Retrying in 1 minute...")
                    api_issues.append(f"Failed to fetch market bars: {e}")
                    time.sleep(60)
                    continue
                    
                # Classify tickers using lookback based on universe_type
                try:
                    log.info("Universe selection started...")
                    lookback_val = 15 if universe_type == "15D" else None
                    volatile_all, nonvolatile_all, atr_ranked = classify_tickers(panels, lookback_days=lookback_val)
                    
                    # Target top 25 volatile and bottom 55 non-volatile
                    # (shifted ratio: non-volatile stocks mean-revert better)
                    volatile_list = atr_ranked.head(25).index.tolist()
                    nonvolatile_list = atr_ranked.tail(55).index.tolist()
                    selection_timestamp = datetime.now(market_tz)
                    
                    log.info("Universe selection completed.")
                    log.info(f"Selected volatile stocks: {', '.join(volatile_list)}")
                    log.info(f"Selected non-volatile stocks: {', '.join(nonvolatile_list)}")
                except Exception as e:
                    log.error(f"Error classifying tickers: {e}")
                    volatile_list = []
                    nonvolatile_list = []
                    atr_ranked = pd.Series()
                    raise e
                    
                try:
                    log.info("Calculating ATR values for watchlist...")
                    atr_values = compute_watchlist_atr(panels, today_date)
                except Exception as e:
                    log.warning(f"Error calculating ATR values: {e}")
                    atr_values = {}
                    
                active_universe_tickers = list(set(volatile_list + nonvolatile_list))
                filtered_panels = {}
                for key in ["close", "open", "high", "low", "volume"]:
                    cols = [c for c in active_universe_tickers if c in panels[key].columns]
                    filtered_panels[key] = panels[key][cols].copy()
                    
                log.info("Running DailySignalEngine & StockPicker...")
                picker = StockPicker(
                    panels=filtered_panels,
                    nifty_close=nifty_close,
                    capital=capital,
                    n_picks=n_picks,
                    min_score=min_score,
                    min_avg_volume=200_000,
                    momentum_lookback=5,
                    momentum_threshold=momentum_threshold,
                    max_per_sector=max_per_sector,
                    market=market,
                    min_basket_size=min_basket_size,
                    momentum_mult=momentum_mult,
                    use_blacklist=use_blacklist,
                )
                
                picks = picker.pick(today_date)
                if not picks:
                    log.warning("No stock picks identified today. Suspending trading for this universe.")
                    termination_reason = "NO_PICKS_IDENTIFIED"
                    break
                    
                # Override entry price and shares
                if len(filtered_panels["open"]) > 0:
                    day_mask = filtered_panels["open"].index.normalize() == today_date
                    day_open = filtered_panels["open"][day_mask]
                    
                    entry_idx = 3
                    times_list = day_open.index.time
                    found = False
                    for idx, t in enumerate(times_list):
                        if t >= time_entry:
                            entry_idx = idx
                            found = True
                            break
                    if not found and len(times_list) > 0:
                        entry_idx = len(times_list) - 1
                            
                    if len(day_open) > 0:
                        entry_idx_safe = min(entry_idx, len(day_open) - 1)
                        entry_prices = day_open.iloc[entry_idx_safe]
                    else:
                        entry_idx_safe = 0
                        entry_prices = filtered_panels["close"].iloc[-1]
                        
                    per_stock_capital = capital / len(picks)
                    
                    log.info(f"Overriding entry prices and allocations at target entry time: {time_entry.strftime('%H:%M')} {market_tz}")
                    for pick in picks:
                        ticker = pick["ticker"]
                        price = entry_prices.get(ticker, np.nan)
                        if not pd.isna(price) and price > 0:
                            pick["entry_price"] = float(price)
                            pick["shares"] = int(per_stock_capital / price)
                            pick["entry_bar_idx"] = entry_idx_safe
                            
                for pick in picks:
                    ticker = pick["ticker"]
                    if ticker in volatile_list:
                        pick["group"] = "volatile"
                    elif ticker in nonvolatile_list:
                        pick["group"] = "nonvolatile"
                    else:
                        pick["group"] = "unclassified"
                    pick["atr_value"] = atr_values.get(ticker, None)
                        
                # Check for crash recovery / mid-day startup
                if current_time >= time_recovery_threshold:
                    log.info("\n[CRASH RECOVERY] Mid-day startup detected. Simulating state so far...")
                    tracking_states = get_expected_state_so_far(picks, filtered_panels, today_date, time_entry, time_flatten)
                    
                    actual_positions = {}
                    if not dry_run:
                        try:
                            al_positions = trading_client.get_all_positions()
                            actual_positions = {p.symbol: int(float(p.qty)) for p in al_positions}
                        except Exception as e:
                            log.error(f"Could not retrieve active Alpaca positions: {e}")
                            api_issues.append(f"Alpaca get_all_positions failed: {e}")
                            
                    log.info(f"Reconstructed {len(tracking_states)} states. Syncing with broker...")
                    for ticker, state in tracking_states.items():
                        exp_qty = state["qty"] if state["active"] else 0
                        act_qty = actual_positions.get(ticker, 0)
                        
                        if exp_qty == 0 and act_qty > 0:
                            execute_order(trading_client, ticker, act_qty, OrderSide.SELL, dry_run)
                        elif exp_qty > 0 and act_qty == 0:
                            state["active"] = False
                        elif exp_qty > 0 and act_qty != exp_qty:
                            diff = act_qty - exp_qty
                            if diff > 0:
                                execute_order(trading_client, ticker, diff, OrderSide.SELL, dry_run)
                                state["qty"] = exp_qty
                            else:
                                state["qty"] = act_qty
                        state["entry_time"] = start_time  # Set default entry time in recovery
                    trades_entered = True
                else:
                    # ── Sentiment Confirmation Layer ──
                    try:
                        sentiment_engine = SentimentDecisionEngine(market=market)
                        if sentiment_engine.enabled:
                            confirmed_picks, sentiment_report = sentiment_engine.evaluate(picks, today_date)
                            picks = confirmed_picks
                    except Exception as e:
                        log.warning(f"[SENTIMENT] Sentiment module failed. Bypassing and proceeding with original picks: {e}")
                        
                    if not picks:
                        log.warning("No stock picks confirmed after news sentiment filtering today. Suspending trading.")
                        termination_reason = "NO_CONFIRMED_PICKS"
                        break
                        
                    log.info("\n[ORDER SUBMISSION] Submitting entries at market entry time...")
                    log.info("Execution started.")
                    for pick in picks:
                        ticker = pick["ticker"]
                        qty = pick["shares"]
                        success = execute_order(trading_client, ticker, qty, OrderSide.BUY, dry_run)
                        
                        if success:
                            tracking_states[ticker] = {
                                "ticker": ticker,
                                "group": pick.get("group", "unclassified"),
                                "entry_price": pick["entry_price"],
                                "initial_qty": qty,
                                "qty": qty,
                                "high_water": pick["entry_price"],
                                "atr_value": pick.get("atr_value"),
                                "sl1_done": False,
                                "sl2_done": False,
                                "sl3_done": False,
                                "pt1_done": False,
                                "pt2_done": False,
                                "pt3_done": False,
                                "sl_exited": 0.0,
                                "time_exit_1_done": False,
                                "time_exit_2_done": False,
                                "active": True,
                                "exit_reason": "",
                                "exits": [],
                                "entry_time": datetime.now(market_tz)
                            }
                    
                    if not dry_run and trading_client is not None and tracking_states:
                        log.info("Sleeping 3 seconds for orders to fill, then syncing entry prices with broker...")
                        time.sleep(3)
                        try:
                            positions = trading_client.get_all_positions()
                            pos_dict = {p.symbol: float(p.avg_entry_price) for p in positions}
                            for ticker, state in tracking_states.items():
                                if ticker in pos_dict:
                                    actual_price = pos_dict[ticker]
                                    log.info(f"[PRICE SYNC] {ticker}: yfinance target price {state['entry_price']:.2f} -> broker fill price {actual_price:.2f}")
                                    state["entry_price"] = actual_price
                                    state["high_water"] = actual_price
                        except Exception as e:
                            log.warning(f"Could not fetch Alpaca positions to sync entry prices: {e}")
                            
                    trades_entered = True
                    save_live_state(state_file_path, today_date, time_entry, time_flatten, trades_entered, picks, tracking_states)
                    
                if run_once:
                    log.info("[RUN ONCE] Entry stage complete. Exiting script loop.")
                    break
                    
            # 2. INTRADAY MONITORING STAGE
            elif time_entry <= current_time < time_flatten and trades_entered:
                active_count = sum(1 for s in tracking_states.values() if s["active"])
                if active_count == 0:
                    log.info(f"No active positions remaining. Sleeping until scheduled flatten close at {time_flatten.strftime('%I:%M %p')}...")
                    time.sleep(30)
                    continue
                    
                mins_now = now.minute
                secs_now = now.second
                sleep_sec = ((4 - (mins_now % 5)) * 60) + (60 - secs_now) + 20
                log.info(f"[MONITORING] {active_count} active trades. Sleeping {sleep_sec}s until next completed bar...")
                time.sleep(sleep_sec)
                
                now = datetime.now(market_tz)
                try:
                    panels, index_close_series = fetch_yfinance_panels(
                        tickers=watchlist,
                        index_ticker=index_ticker,
                        lookback_days=2,
                        market_tz=market_tz,
                        market_open=market_open,
                        market_close=market_close
                    )
                except Exception as e:
                    log.error(f"[MONITORING FAILED] Could not fetch live bars: {e}")
                    api_issues.append(f"Failed to fetch live bars during monitoring: {e}")
                    continue
                    
                close = panels["close"]
                high = panels["high"]
                low = panels["low"]
                
                day_mask = close.index.normalize() == today_date
                today_close = close[day_mask]
                today_high = high[day_mask]
                today_low = low[day_mask]
                
                if len(today_close) == 0:
                    log.warning("No intraday bars found for today. Retrying on next check.")
                    continue
                    
                last_idx = len(today_close) - 1
                last_timestamp = today_close.index[-1]
                
                log.info(f"\n[BAR CHECK] Completed Bar: {last_timestamp.strftime('%H:%M:%S')} {market_tz}")
                log.info(f"  %-12s  %-12s  %8s  %8s  %8s  %8s  %s", "Ticker", "Group", "Entry", "Current", "Low", "High", "Status")
                log.info("  " + "-" * 77)
                
                # Unpack execution parameters from single source of truth (EXEC_PARAMS)
                sl_tier1_pct = EXEC_PARAMS["sl_tier1_pct"]
                sl_tier2_pct = EXEC_PARAMS["sl_tier2_pct"]
                sl_tier3_pct = EXEC_PARAMS["sl_tier3_pct"]
                sl_tier1_weight = EXEC_PARAMS["sl_tier1_weight"]
                sl_tier2_weight = EXEC_PARAMS["sl_tier2_weight"]
                sl_tier3_weight = EXEC_PARAMS["sl_tier3_weight"]
                trail_trigger = EXEC_PARAMS["trail_trigger"]
                trail_pct = EXEC_PARAMS["trail_pct"]
                profit_take_1 = EXEC_PARAMS["profit_take_1"]
                profit_take_2 = EXEC_PARAMS["profit_take_2"]
                profit_take_3 = EXEC_PARAMS["profit_take_3"]
                atr_pt_1 = EXEC_PARAMS["atr_pt_1"]
                atr_pt_2 = EXEC_PARAMS["atr_pt_2"]
                atr_pt_3 = EXEC_PARAMS["atr_pt_3"]
                pt_weight_1 = EXEC_PARAMS["pt_weight_1"]
                pt_weight_2 = EXEC_PARAMS["pt_weight_2"]
                pt_weight_3 = EXEC_PARAMS["pt_weight_3"]
                
                # Unpack new risk management parameters
                time_exit_1_min = EXEC_PARAMS["time_exit_1_minutes"]
                time_exit_1_sell = EXEC_PARAMS["time_exit_1_sell_pct"]
                time_exit_2_min = EXEC_PARAMS["time_exit_2_minutes"]
                time_exit_2_sell = EXEC_PARAMS["time_exit_2_sell_pct"]
                market_stress_threshold = EXEC_PARAMS["market_stress_threshold"]
                extreme_move_pct = EXEC_PARAMS["extreme_move_pct"]
                max_portfolio_dd = EXEC_PARAMS["max_portfolio_drawdown"]
                
                # ── MARKET STRESS DETECTION ──
                # If index has dropped >3% intraday, flatten all to prevent catastrophic losses
                try:
                    if index_close_series is not None and not index_close_series.empty:
                        idx_day_close = index_close_series[day_mask].dropna()
                        if len(idx_day_close) >= 2:
                            idx_open = idx_day_close.iloc[0]
                            idx_current = idx_day_close.iloc[-1]
                            idx_intraday_ret = (idx_current - idx_open) / idx_open
                            if abs(idx_intraday_ret) > market_stress_threshold:
                                log.warning(f"  [MARKET STRESS] {index_ticker} intraday return: {idx_intraday_ret*100:.2f}% exceeds ±{market_stress_threshold*100:.1f}% threshold")
                                log.warning(f"  [MARKET STRESS] Force-flattening all positions to prevent further losses")
                                for t_ticker, t_state in tracking_states.items():
                                    if t_state["active"] and t_state["qty"] > 0:
                                        execute_order(trading_client, t_ticker, t_state["qty"], OrderSide.SELL, dry_run)
                                        t_state["qty"] = 0
                                        t_state["active"] = False
                                        t_state["exit_reason"] = "MARKET_STRESS_EXIT"
                                        t_state["exits"].append((t_state["initial_qty"], idx_current, now))
                                save_live_state(state_file_path, today_date, time_entry, time_flatten, trades_entered, picks, tracking_states)
                                continue
                except Exception as e:
                    log.warning(f"  Market stress check failed: {e}")
                
                # ── PORTFOLIO DRAWDOWN CHECK ──
                # If portfolio value has dropped >3% from starting capital, flatten all
                current_prices = {
                    t: today_close.iloc[last_idx][t]
                    for t in tracking_states.keys()
                    if t in today_close.columns and not pd.isna(today_close.iloc[last_idx][t])
                }
                current_est = calculate_portfolio_value(capital, tracking_states, current_prices)
                peak_value = max([v for _, v in portfolio_history] + [capital])
                if peak_value > 0:
                    dd_from_peak = (current_est - peak_value) / peak_value
                    if dd_from_peak < -max_portfolio_dd:
                        log.warning(f"  [DRAWDOWN PROTECTION] Portfolio drawdown {dd_from_peak*100:.2f}% exceeds -{max_portfolio_dd*100:.1f}% limit (Peak: ${peak_value:,.2f}, Current Est: ${current_est:,.2f})")
                        log.warning(f"  [DRAWDOWN PROTECTION] Force-flattening all positions")
                        for t_ticker, t_state in tracking_states.items():
                            if t_state["active"] and t_state["qty"] > 0:
                                execute_order(trading_client, t_ticker, t_state["qty"], OrderSide.SELL, dry_run)
                                t_state["qty"] = 0
                                t_state["active"] = False
                                t_state["exit_reason"] = "DRAWDOWN_PROTECTION"
                                last_px = current_prices.get(t_ticker, t_state["entry_price"])
                                t_state["exits"].append((t_state["initial_qty"], last_px, now))
                        save_live_state(state_file_path, today_date, time_entry, time_flatten, trades_entered, picks, tracking_states)
                        continue
                
                # ── Intraday Sentiment Exit check ────────────────────────
                active_tickers = [t for t, s in tracking_states.items() if s["active"]]
                if active_tickers:
                    try:
                        sentiment_engine = SentimentDecisionEngine(market=market)
                        if sentiment_engine.enabled:
                            # Use 2 hours lookback to fetch breaking news
                            sentiment_engine.collector.max_age_hours = 2
                            articles = sentiment_engine.collector.fetch(tickers=active_tickers, market_news=False, market=market)
                            
                            for ticker in active_tickers:
                                t_arts = [a for a in articles if a.ticker == ticker]
                                if t_arts:
                                    results = [sentiment_engine.analyzer.analyze(a.headline + " " + a.summary) for a in t_arts]
                                    avg_pol = sum(r.polarity for r in results) / len(results)
                                    
                                    if avg_pol <= sentiment_engine.strong_neg_threshold:
                                        t_state = tracking_states[ticker]
                                        qty_to_sell = t_state["qty"]
                                        if qty_to_sell > 0:
                                            log.warning(f"  [SENTIMENT EXIT] Strong negative sentiment detected for {ticker} (Avg Polarity: {avg_pol:+.2f}). Selling all {qty_to_sell} shares immediately.")
                                            if execute_order(trading_client, ticker, qty_to_sell, OrderSide.SELL, dry_run):
                                                t_state["qty"] = 0
                                                t_state["active"] = False
                                                t_state["exit_reason"] = "SENTIMENT_EXIT"
                                                bar_close = today_close.iloc[last_idx].get(ticker, np.nan)
                                                if pd.isna(bar_close) or bar_close <= 0:
                                                    bar_close = t_state["entry_price"]
                                                t_state["exits"].append((qty_to_sell, float(bar_close), now))
                    except Exception as e:
                        log.debug(f"[SENTIMENT MONITORING] Failed: {e}")

                current_bar_check_prices = {}
                active_positions_value = 0.0
                
                for ticker, state in tracking_states.items():
                    if not state["active"]:
                        continue
                        
                    if ticker not in today_close.columns or ticker not in today_high.columns or ticker not in today_low.columns:
                        log.warning(f"  Missing price data for {ticker} in current bar.")
                        continue
                        
                    bar_close = today_close.iloc[last_idx][ticker]
                    bar_high = today_high.iloc[last_idx][ticker]
                    bar_low = today_low.iloc[last_idx][ticker]
                    
                    if pd.isna(bar_close) or bar_close <= 0:
                        continue
                        
                    current_bar_check_prices[ticker] = bar_close
                    active_positions_value += state["qty"] * bar_close
                    
                    entry_price = state["entry_price"]
                    initial_qty = state["initial_qty"]
                    state["high_water"] = max(state["high_water"], bar_high)
                    
                    status_str = f"Qty: {state['qty']} | HW: ${state['high_water']:.2f}"
                    log.info(f"  %-12s  %-12s  %8.2f  %8.2f  %8.2f  %8.2f  %s", 
                             ticker, state.get("group", "unclassified"), entry_price, bar_close, bar_low, bar_high, status_str)
                    
                    # Stop Loss
                    if not state["sl1_done"] and bar_low <= entry_price * (1 - sl_tier1_pct):
                        remaining = 1.0 - state["sl_exited"]
                        if remaining > 0:
                            portion = min(1.0, sl_tier1_weight / remaining)
                            sold = int(state["qty"] * portion)
                            if sold > 0:
                                log.info(f"  [STOP LOSS TIER 1] Breached: {ticker} Low: {bar_low:.2f} <= Limit: {entry_price * (1 - sl_tier1_pct):.2f}")
                                if execute_order(trading_client, ticker, sold, OrderSide.SELL, dry_run):
                                    state["qty"] -= sold
                                    state["exits"].append((sold, entry_price * (1 - sl_tier1_pct), now))
                        state["sl_exited"] += sl_tier1_weight
                        state["sl1_done"] = True
                        if state["qty"] <= 0:
                            state["active"] = False
                            state["exit_reason"] = "STOP_LOSS_TIER_1"
                            continue
                            
                    if not state["sl2_done"] and bar_low <= entry_price * (1 - sl_tier2_pct):
                        remaining = 1.0 - state["sl_exited"]
                        if remaining > 0:
                            portion = min(1.0, sl_tier2_weight / remaining)
                            sold = int(state["qty"] * portion)
                            if sold > 0:
                                log.info(f"  [STOP LOSS TIER 2] Breached: {ticker} Low: {bar_low:.2f} <= Limit: {entry_price * (1 - sl_tier2_pct):.2f}")
                                if execute_order(trading_client, ticker, sold, OrderSide.SELL, dry_run):
                                    state["qty"] -= sold
                                    state["exits"].append((sold, entry_price * (1 - sl_tier2_pct), now))
                        state["sl_exited"] += sl_tier2_weight
                        state["sl2_done"] = True
                        if state["qty"] <= 0:
                            state["active"] = False
                            state["exit_reason"] = "STOP_LOSS_TIER_2"
                            continue
                            
                    if not state["sl3_done"] and bar_low <= entry_price * (1 - sl_tier3_pct):
                        log.info(f"  [STOP LOSS TIER 3] Breached: {ticker} Low: {bar_low:.2f} <= Limit: {entry_price * (1 - sl_tier3_pct):.2f}")
                        qty_to_sell = state["qty"]
                        if execute_order(trading_client, ticker, qty_to_sell, OrderSide.SELL, dry_run):
                            state["qty"] = 0
                            state["sl3_done"] = True
                            state["active"] = False
                            state["exit_reason"] = "STOP_LOSS_TIER_3"
                            state["exits"].append((qty_to_sell, entry_price * (1 - sl_tier3_pct), now))
                            continue
                            
                    # Trailing Stop
                    if state["high_water"] >= entry_price * (1 + trail_trigger):
                        trail_price = state["high_water"] * (1 - trail_pct)
                        if bar_low <= trail_price:
                            log.info(f"  [TRAILING STOP] Breached: {ticker} Low: {bar_low:.2f} <= Trail Price: {trail_price:.2f}")
                            qty_to_sell = state["qty"]
                            if execute_order(trading_client, ticker, qty_to_sell, OrderSide.SELL, dry_run):
                                state["qty"] = 0
                                state["active"] = False
                                state["exit_reason"] = "TRAILING_STOP"
                                state["exits"].append((qty_to_sell, trail_price, now))
                                continue
                                
                    # 2.5 Extreme Move Detection
                    if extreme_move_pct is not None and extreme_move_pct > 0:
                        sess_open = today_close.iloc[0][ticker] if len(today_close) > 0 else np.nan
                        if pd.isna(sess_open) or sess_open <= 0:
                            sess_open = entry_price
                        
                        move_high = (bar_high - sess_open) / sess_open
                        move_low = (bar_low - sess_open) / sess_open
                        if abs(move_high) >= extreme_move_pct or abs(move_low) >= extreme_move_pct:
                            log.info(f"  [EXTREME MOVE] Breached: {ticker} High: {bar_high:.2f}, Low: {bar_low:.2f} relative to Session Open: {sess_open:.2f} exceeds threshold {extreme_move_pct*100:.1f}%")
                            qty_to_sell = state["qty"]
                            if qty_to_sell > 0:
                                if execute_order(trading_client, ticker, qty_to_sell, OrderSide.SELL, dry_run):
                                    state["qty"] = 0
                                    state["active"] = False
                                    state["exit_reason"] = "EXTREME_MOVE"
                                    state["exits"].append((qty_to_sell, bar_close, now))
                                    continue

                    # 2.6 Time-Based Partial Exits
                    minutes_elapsed = 0.0
                    if state.get("entry_time") is not None:
                        elapsed_td = now - state["entry_time"]
                        minutes_elapsed = elapsed_td.total_seconds() / 60.0

                    if not state.setdefault("time_exit_1_done", False) and minutes_elapsed >= time_exit_1_min:
                        if bar_close > entry_price:
                            qty_to_sell = int(state["qty"] * time_exit_1_sell)
                            if qty_to_sell > 0:
                                log.info(f"  [TIME EXIT 1] Breached: {ticker} elapsed: {minutes_elapsed:.1f}m >= {time_exit_1_min}m | Profit: {(bar_close - entry_price)/entry_price*100:.2f}% | Selling {qty_to_sell} shares")
                                if execute_order(trading_client, ticker, qty_to_sell, OrderSide.SELL, dry_run):
                                    state["qty"] -= qty_to_sell
                                    state["exits"].append((qty_to_sell, bar_close, now))
                        state["time_exit_1_done"] = True
                        if state["qty"] <= 0:
                            state["active"] = False
                            state["exit_reason"] = "TIME_EXIT_1"
                            continue

                    if not state.setdefault("time_exit_2_done", False) and minutes_elapsed >= time_exit_2_min:
                        if bar_close > entry_price:
                            qty_to_sell = int(state["qty"] * time_exit_2_sell)
                            if qty_to_sell > 0:
                                log.info(f"  [TIME EXIT 2] Breached: {ticker} elapsed: {minutes_elapsed:.1f}m >= {time_exit_2_min}m | Profit: {(bar_close - entry_price)/entry_price*100:.2f}% | Selling {qty_to_sell} shares")
                                if execute_order(trading_client, ticker, qty_to_sell, OrderSide.SELL, dry_run):
                                    state["qty"] -= qty_to_sell
                                    state["exits"].append((qty_to_sell, bar_close, now))
                        state["time_exit_2_done"] = True
                        if state["qty"] <= 0:
                            state["active"] = False
                            state["exit_reason"] = "TIME_EXIT_2"
                            continue

                    # Profit Targets (ATR-based or Fixed Fallback)
                    atr_val = state.get("atr_value", None)
                    
                    if atr_pt_1 is not None and atr_val is not None:
                        target_1 = entry_price + (atr_pt_1 * atr_val)
                    else:
                        target_1 = entry_price * (1 + profit_take_1)
                        
                    if atr_pt_2 is not None and atr_val is not None:
                        target_2 = entry_price + (atr_pt_2 * atr_val)
                    else:
                        target_2 = entry_price * (1 + profit_take_2)
                        
                    if atr_pt_3 is not None and atr_val is not None:
                        target_3 = entry_price + (atr_pt_3 * atr_val)
                    else:
                        target_3 = entry_price * (1 + profit_take_3)
                    
                    if not state["pt1_done"] and bar_high >= target_1:
                        sold = int(initial_qty * pt_weight_1)
                        qty_to_sell = min(sold, state["qty"])
                        if qty_to_sell > 0:
                            log.info(f"  [PROFIT TAKE 1] Breached: {ticker} High: {bar_high:.2f} >= Target: {target_1:.2f}")
                            if execute_order(trading_client, ticker, qty_to_sell, OrderSide.SELL, dry_run):
                                state["qty"] -= qty_to_sell
                                state["exits"].append((qty_to_sell, target_1, now))
                        state["pt1_done"] = True
                        if state["qty"] <= 0:
                            state["active"] = False
                            state["exit_reason"] = "PROFIT_TAKE_1"
                            continue
                            
                    if not state["pt2_done"] and bar_high >= target_2:
                        sold = int(initial_qty * pt_weight_2)
                        qty_to_sell = min(sold, state["qty"])
                        if qty_to_sell > 0:
                            log.info(f"  [PROFIT TAKE 2] Breached: {ticker} High: {bar_high:.2f} >= Target: {target_2:.2f}")
                            if execute_order(trading_client, ticker, qty_to_sell, OrderSide.SELL, dry_run):
                                state["qty"] -= qty_to_sell
                                state["exits"].append((qty_to_sell, target_2, now))
                        state["pt2_done"] = True
                        if state["qty"] <= 0:
                            state["active"] = False
                            state["exit_reason"] = "PROFIT_TAKE_2"
                            continue
                            
                    if not state["pt3_done"] and bar_high >= target_3:
                        qty_to_sell = state["qty"]  # closes remaining
                        if qty_to_sell > 0:
                            log.info(f"  [PROFIT TAKE 3] Breached: {ticker} High: {bar_high:.2f} >= Target: {target_3:.2f}")
                            if execute_order(trading_client, ticker, qty_to_sell, OrderSide.SELL, dry_run):
                                state["qty"] = 0
                                state["exits"].append((qty_to_sell, target_3, now))
                                state["pt3_done"] = True
                                state["active"] = False
                                state["exit_reason"] = "PROFIT_TAKE_3"
                            else:
                                log.warning(f"  [PROFIT TAKE 3] Order FAILED for {ticker}. Position stays active for retry.")
                        else:
                            # Already fully exited by prior stops
                            state["pt3_done"] = True
                            state["active"] = False
                            state["exit_reason"] = "PROFIT_TAKE_3"
                        continue
                            
                # Save state, record portfolio values and exposure
                save_live_state(state_file_path, today_date, time_entry, time_flatten, trades_entered, picks, tracking_states)
                current_portfolio_value = calculate_portfolio_value(capital, tracking_states, current_bar_check_prices)
                portfolio_history.append((last_timestamp, current_portfolio_value))
                
                exposure_pct_current = (active_positions_value / capital) * 100.0
                exposure_pcts.append(exposure_pct_current)
                
                log.info(f"Current Portfolio Value: ${current_portfolio_value:,.2f} | Exposure: {exposure_pct_current:.2f}%")
                
            # 3. DYNAMIC FLATTEN STAGE
            elif (time_flatten <= current_time <= time_market_close) and trades_entered:
                log.info(f"\n[FLATTEN PHASE] Exiting all remaining positions at scheduled exit time ({time_flatten.strftime('%I:%M %p')})...")
                
                if not dry_run:
                    try:
                        positions = trading_client.get_all_positions()
                        for pos in positions:
                            ticker = pos.symbol
                            qty = int(float(pos.qty))
                            if qty == 0:
                                continue
                            log.info(f"Flattening broker position: {qty} shares of {ticker}")
                            if qty > 0:
                                execute_order(trading_client, ticker, qty, OrderSide.SELL, dry_run)
                            else:
                                execute_order(trading_client, ticker, abs(qty), OrderSide.BUY, dry_run)
                    except Exception as e:
                        log.error(f"Could not retrieve broker positions: {e}. Falling back to tracking states.")
                        api_issues.append(f"Broker position retrieval failed on flatten: {e}")
                        for ticker, state in tracking_states.items():
                            if state["active"] and state["qty"] > 0:
                                execute_order(trading_client, ticker, state["qty"], OrderSide.SELL, dry_run)
                else:
                    log.info("[DRY RUN] Simulating flatten of all tracking positions.")
                    for ticker, state in tracking_states.items():
                        if state["active"] and state["qty"] > 0:
                            log.info(f"[DRY RUN] Flatten: {state['qty']} shares of {ticker}")
                            
                log.info("Flattening process completed successfully.")
                log.info("Execution stopped.")
                trades_entered = False
                termination_reason = "DURATION_COMPLETED"
                duration_completed = True
                break
                
            # 4. MARKET IS CLOSED STAGE
            else:
                trades_entered = False
                log.info(f"[CLOSED] Market is closed. Current Time: {now.strftime('%H:%M:%S')} {market_tz}. Waiting for next market session...")
                if run_once or dry_run:
                    log.info("[CLOSED] Exiting loop for this universe because run_once or dry_run is set.")
                    termination_reason = "MARKET_CLOSED"
                    break
                time.sleep(300)
                
    except KeyboardInterrupt:
        log.warning(f"\n[{universe_type.upper()}] Manual stop detected via KeyboardInterrupt.")
        termination_reason = "MANUAL_STOP"
        manual_stop = True
        log.info("Execution stopped.")
        graceful_shutdown(trading_client, tracking_states, dry_run)
    except Exception as e:
        log.error(f"\n[{universe_type.upper()}] Runtime Exception: {e}", exc_info=True)
        termination_reason = "EXCEPTION"
        runtime_errors.append(str(e))
        log.info("Execution stopped.")
        graceful_shutdown(trading_client, tracking_states, dry_run)
    finally:
        log.info(f"[{universe_type.upper()}] Post-session summary starts. Generating reports...")
        end_time = datetime.now(market_tz)
        duration_seconds = (end_time - start_time).total_seconds()
        
        # Value remaining tracking states at last known prices
        final_closes = {}
        try:
            temp_panels, _ = fetch_yfinance_panels(
                tickers=watchlist,
                index_ticker=index_ticker,
                lookback_days=2,
                market_tz=market_tz,
                market_open=market_open,
                market_close=market_close
            )
            if len(temp_panels["close"]) > 0:
                final_closes = temp_panels["close"].iloc[-1].to_dict()
        except Exception as e:
            log.warning(f"Could not fetch final close prices: {e}")
            
        for ticker, state in tracking_states.items():
            if state.get("active") and state.get("qty", 0) > 0:
                rem_qty = state["qty"]
                last_px = final_closes.get(ticker, state["entry_price"])
                state["exits"].append((rem_qty, last_px, end_time))
                state["qty"] = 0
                state["active"] = False
                state["exit_reason"] = "TERMINATION_EXIT"
                
        # Clean state exits list for report generator
        trades_list = []
        for ticker, state in tracking_states.items():
            trades_list.append({
                "ticker": ticker,
                "group": state.get("group", "unclassified"),
                "entry_price": state["entry_price"],
                "initial_qty": state["initial_qty"],
                "exits": state["exits"],
                "entry_time": state.get("entry_time", start_time),
                "exit_reason": state.get("exit_reason", "")
            })
            
        exposure_pct = np.mean(exposure_pcts) if exposure_pcts else 0.0
        
        # Save the report to report_save_path
        master_report_path = report_save_path
        
        log.info(f"Report text generation started for {universe_type.upper()}...")
        from live_report_generator import generate_individual_report_text
        session_metrics, report_text = generate_individual_report_text(
            market=market,
            universe_type=universe_type,
            universe_size=len(watchlist),
            volatile_list=volatile_list,
            nonvolatile_list=nonvolatile_list,
            ranking_method="trailing_atr",
            selection_timestamp=selection_timestamp if selection_timestamp else start_time,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            termination_reason=termination_reason,
            duration_completed=(termination_reason == "DURATION_COMPLETED"),
            manual_stop=manual_stop,
            runtime_errors=runtime_errors,
            api_issues=api_issues,
            trades=trades_list,
            capital=capital,
            portfolio_history=portfolio_history,
            exposure_pct=exposure_pct,
            save_path=master_report_path if master_report_path else "",
            sentiment_summary=sentiment_report,
        )
        log.info(f"Report text generated for {universe_type.upper()}.")
        
        if master_report_path:
            try:
                os.makedirs(os.path.dirname(master_report_path), exist_ok=True)
                with open(master_report_path, "w", encoding="utf-8") as f:
                    f.write(report_text)
                log.info(f"Report successfully saved to {master_report_path}")
            except Exception as e:
                log.error(f"Failed to write report to {master_report_path}: {e}")
                
        clear_live_state(state_file_path)
        
        return session_metrics, report_text


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NASDAQ/NSE Mean-Reversion Active Execution Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--paper", action="store_true", default=True,
                   help="Run in Alpaca Paper Trading Mode (default)")
    p.add_argument("--live", action="store_true",
                   help="Run in Alpaca Live Trading Mode (WARNING: Real Money)")
    p.add_argument("--dry-run", action="store_true",
                   help="Simulate signal computation and order routing without executing transactions")
    p.add_argument("--capital", type=float, default=100000,
                   help="Total simulated or active capital pool size")
    p.add_argument("--picks", type=int, default=15,
                   help="Number of stocks to select for daily trading")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Watchlist of tickers to screen and trade (defaults to full market seed pool)")
    p.add_argument("--run-once", action="store_true",
                   help="Process only the immediate cycle (entry or exit) and exit immediately (useful for cron jobs)")
    p.add_argument("--duration", type=float, default=6.08,
                   help="Duration in hours to hold positions before flattening")
    p.add_argument("--min-score", type=float, default=0.8,
                   help="Minimum z-score to qualify a stock pick")
    p.add_argument("--min-basket-size", type=int, default=8,
                   help="Minimum number of stocks to select for daily trading")
    p.add_argument("--momentum-mult", type=float, default=2.0,
                   help="Volatility multiplier (x ATR) for dynamic momentum filtering. Set to 0 to disable.")
    p.add_argument("--momentum-threshold", type=float, default=0.05,
                   help="Static momentum threshold (used if momentum-mult is disabled)")
    p.add_argument("--max-per-sector", type=int, default=3,
                   help="Maximum number of stocks allowed in the same sector. Set to 0 to disable.")
    p.add_argument("--disable-blacklist", action="store_true",
                   help="Disable the known-loser blacklist of chronic losers")
    p.add_argument("--log-file", default="live_logs/pipeline.log",
                   help="Path to the log file for live trading")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--market", type=str, required=True,
                   help="Market selection: 'us' or 'india'")
    p.add_argument("--universe", type=str, required=True,
                   choices=["15d", "alltime"],
                   help="Universe selection: '15d' or 'alltime'")
    
    args = p.parse_args()
    if args.market.lower() not in ["us", "india"]:
        p.error(f"Invalid market value: '{args.market}'. Valid choices are 'us' or 'india'.")
    args.market = args.market.lower()
    args.universe = args.universe.lower()
    return args


if __name__ == "__main__":
    args = _parse_args()
    setup_logging(args.log_level, args.log_file)
    
    load_dotenv()
    
    market = args.market
    log.info(f"Market selected: {market.upper()}")
    
    if args.live:
        log.warning("!!! RUNNING IN LIVE TRADING MODE (REAL MONEY) !!!")
        api_key = os.getenv('ALPACA_LIVE_API_KEY') or os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_LIVE_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')
        paper_trading = False
    else:
        log.info("[PAPER TRADING MODE]")
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        paper_trading = True
        
    if not api_key or not secret_key:
        if args.dry_run:
            log.warning("Alpaca API credentials missing, but running in DRY RUN mode. Bypassing credentials check.")
            trading_client = None
            data_client = None
        else:
            log.error("Alpaca API credentials missing. Please define ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file.")
            sys.exit(1)
    else:
        trading_client = TradingClient(api_key, secret_key, paper=paper_trading)
        data_client = StockHistoricalDataClient(api_key, secret_key)
    
    # Load correct seed pool
    if market == "us":
        seed_pool = list(dict.fromkeys(SEED_POOL))
    else:
        seed_pool = list(dict.fromkeys(NSE_SEED_POOL))
        
    if args.tickers is None:
        watchlist = seed_pool
    else:
        watchlist = args.tickers
        
    if market == "india":
        watchlist = [t if t.endswith(".NS") else f"{t}.NS" for t in watchlist]
        
    # Set up reporting folders
    base_report_dir = os.path.join(ROOT, "live_reports")
    if not os.path.exists(base_report_dir):
        base_report_dir = os.path.join(ROOT, "live_report")
        
    market_report_dir = os.path.join(base_report_dir, f"reports_{market}")
    os.makedirs(market_report_dir, exist_ok=True)
    
    # Determine universe subfolder and type string
    universe_selected = args.universe.lower()
    if universe_selected == "15d":
        subfolder_name = "top_15"
        uni_type = "15D"
    else:
        subfolder_name = "alltime"
        uni_type = "ALLTIME"
        
    subfolder_dir = os.path.join(market_report_dir, subfolder_name)
    os.makedirs(subfolder_dir, exist_ok=True)
    
    # Setup run timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get unique report file path
    report_save_path = get_unique_filepath(
        subfolder_dir,
        f"{market.upper()}_{uni_type}",
        run_timestamp,
        ".txt"
    )
    
    try:
        log.info(SEP)
        log.info(f"STARTING RUN for Market: {market.upper()} | Universe: {uni_type} | Timestamp: {run_timestamp}")
        log.info(SEP)
        
        metrics, report_text = run_universe_session(
            market=market,
            universe_type=uni_type,
            watchlist=watchlist,
            trading_client=trading_client,
            data_client=data_client,
            capital=args.capital,
            n_picks=args.picks,
            duration_hours=args.duration,
            dry_run=args.dry_run,
            run_once=args.run_once,
            log_file=args.log_file,
            report_save_path=report_save_path,
            run_timestamp=run_timestamp,
            min_score=args.min_score,
            min_basket_size=args.min_basket_size,
            momentum_mult=args.momentum_mult if args.momentum_mult > 0 else None,
            momentum_threshold=args.momentum_threshold,
            max_per_sector=args.max_per_sector,
            use_blacklist=not args.disable_blacklist,
        )
        
        log.info(SEP)
        log.info(f"COMPLETED RUN for Market: {market.upper()} | Universe: {uni_type}")
        log.info(f"Report saved to: {report_save_path}")
        log.info(SEP)
            
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt caught in main manager. Aborting.")
