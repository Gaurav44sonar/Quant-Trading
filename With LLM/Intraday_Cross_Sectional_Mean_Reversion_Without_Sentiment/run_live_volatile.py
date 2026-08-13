"""
run_live_volatile.py
====================
Live & Paper Trading Execution Pipeline for Volatile Stocks Only (NASDAQ/NSE).

Dynamically identifies the top 40 most volatile stocks over the last 15 trading
days and executes the intraday mean reversion strategy on them.

Results and logs are saved inside the newly created 'live_volatile_results' folder.
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

# Custom modules
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from alpha.stock_picker import StockPicker
from features.daily_signals import DailySignalEngine
from nse_pipeline.universe import SEED_POOL
from nse_pipeline.universe_nse import NSE_SEED_POOL

# Constants
EST = pytz.timezone('US/Eastern')
SEP = "=" * 70
THIN = "-" * 70

# Execution Parameters (synchronized with strategy defaults)
EXEC_PARAMS = {
    "sl_tier1_pct": 0.01,       # 1.0% stop → sell 50%
    "sl_tier2_pct": 0.02,       # 2.0% stop → sell 25%
    "sl_tier3_pct": 0.035,      # 3.5% stop → sell remaining
    "sl_tier1_weight": 0.50,
    "sl_tier2_weight": 0.25,
    "sl_tier3_weight": 0.25,
    
    "trail_trigger": 0.025,     # 2.5% trailing stop trigger
    "trail_pct": 0.0075,
    
    "profit_take_1": 0.015,
    "profit_take_2": 0.03,
    "profit_take_3": 0.045,
    
    "atr_pt_1": 0.25,           # 0.25 ATR
    "atr_pt_2": 0.50,           # 0.50 ATR
    "atr_pt_3": 1.00,           # 1.00 ATR
    "pt_weight_1": 0.50,
    "pt_weight_2": 0.25,
    "pt_weight_3": 0.25,
    
    "time_exit_1_minutes": 60,   # At 60 min, sell 50% if profitable
    "time_exit_1_sell_pct": 0.50,
    "time_exit_2_minutes": 120,  # At 120 min, sell 75% of remaining if profitable
    "time_exit_2_sell_pct": 0.75,
    
    "market_stress_threshold": 0.03,  # If index drops >3% intraday, flatten all
    "extreme_move_pct": 0.08,         # If stock moves >8% intraday, close position
    "max_portfolio_drawdown": 0.03,   # If portfolio drops >3% from peak, flatten all
}

log = logging.getLogger("run_live_volatile")


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
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


def select_top_40_volatile(market: str, market_tz: pytz.BaseTzInfo) -> tuple[list[str], pd.Series]:
    """
    Downloads last 25 trading days of daily bars from Yahoo Finance for all tickers
    in the seed pool, computes ATR% over the last 15 trading days, and selects
    the top 40 most volatile tickers.
    """
    log.info("Starting screening for top 40 volatile stocks over the last 15 trading days...")
    
    if market == "us":
        seed_pool = list(dict.fromkeys(SEED_POOL))
    else:
        seed_pool = list(dict.fromkeys(NSE_SEED_POOL))
        seed_pool = [t if t.endswith(".NS") else f"{t}.NS" for t in seed_pool]
        
    log.info(f"Downloading daily bars for {len(seed_pool)} seed pool tickers...")
    
    # Download in parallel using yfinance
    try:
        # Use period='1mo' to cover at least 20 trading days
        df = yf.download(
            tickers=seed_pool,
            period="1mo",
            interval="1d",
            auto_adjust=True,
            progress=False
        )
    except Exception as e:
        log.error(f"Failed to download daily data from Yahoo Finance: {e}")
        raise e
        
    if df is None or df.empty:
        raise ValueError("No daily data returned from Yahoo Finance for universe screening.")
        
    # Extract Close, High, Low
    close_df = df["Close"].copy() if "Close" in df else pd.DataFrame()
    high_df = df["High"].copy() if "High" in df else pd.DataFrame()
    low_df = df["Low"].copy() if "Low" in df else pd.DataFrame()
    
    if close_df.empty or high_df.empty or low_df.empty:
        raise ValueError("Close, High, or Low data missing from yfinance download.")
        
    # Align to last 15 trading days
    n_days = len(close_df)
    lookback = min(15, n_days)
    log.info(f"Using last {lookback} trading days of daily history for screening.")
    
    close_sub = close_df.iloc[-lookback:]
    high_sub = high_df.iloc[-lookback:]
    low_sub = low_df.iloc[-lookback:]
    
    # True range computation
    prev_close = close_df.shift(1).reindex(close_sub.index)
    
    tr1 = high_sub - low_sub
    tr2 = (high_sub - prev_close).abs()
    tr3 = (low_sub - prev_close).abs()
    
    tr = tr1.combine(tr2, np.maximum).combine(tr3, np.maximum)
    
    # Average True Range (ATR) is the average TR over the period
    atr = tr.mean()
    
    # Get the last close price for each ticker
    last_close = close_sub.iloc[-1]
    
    # ATR% = ATR / Last Close
    atr_pct = atr / last_close.replace(0, np.nan)
    
    # Sort and take top 40
    atr_ranked = atr_pct.dropna().sort_values(ascending=False)
    top_40 = atr_ranked.head(40).index.tolist()
    
    log.info(f"Top 40 Volatile Stocks identified:")
    for rank, (ticker, val) in enumerate(atr_ranked.head(40).items(), 1):
        log.info(f"  {rank:2d}. {ticker:6s}: ATR% = {val*100:.2f}%")
        
    return top_40, atr_ranked


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


def compute_watchlist_atr(panels: dict[str, pd.DataFrame], today_date: pd.Timestamp) -> dict[str, float]:
    """
    Calculate the daily ATR(14) for each ticker using data strictly before today_date.
    """
    atr_dict = {}
    close = panels["close"]
    high = panels["high"]
    low = panels["low"]
    
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

    # Unpack execution parameters
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
        atr_val = atr_values.get(ticker, None)
        
        state = {
            "ticker": ticker,
            "group": pick.get("group", "volatile"),
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
            
            # Stop-Loss
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
                
            # Trailing Stop
            if state["high_water"] >= entry_price * (1 + trail_trigger):
                trail_price = state["high_water"] * (1 - trail_pct)
                if bar_low <= trail_price:
                    if state["qty"] > 0:
                        state["exits"].append((state["qty"], trail_price, bar_time))
                    state["qty"] = 0
                    state["active"] = False
                    state["exit_reason"] = "TRAILING_STOP"
                    continue
                    
            # Extreme Move Detection
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

            # Time-Based Partial Exits
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
                    
            # Partial Profit Taking (ATR-based or Fixed Fallback)
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
                qty_to_sell = state["qty"]
                if qty_to_sell > 0:
                    state["exits"].append((qty_to_sell, target_3, bar_time))
                    state["qty"] = 0
                state["pt3_done"] = True
                state["active"] = False
                state["exit_reason"] = "PROFIT_TAKE_3"
                continue
                    
            # Mandatory Time Exit
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
    tracking_states: dict[str, dict],
    volatile_list: list[str],
    nonvolatile_list: list[str],
    last_reselection_hour: int = -1,
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
            "tracking_states": serialized_states,
            "volatile_list": volatile_list,
            "nonvolatile_list": nonvolatile_list,
            "last_reselection_hour": last_reselection_hour,
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
                
        # Restore last_reselection_hour (default -1 for backward-compat with old state files)
        if "last_reselection_hour" not in state:
            state["last_reselection_hour"] = -1
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
        if not dry_run and trading_client is not None:
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
    watchlist_seed: list[str],
    trading_client: TradingClient | None,
    data_client: StockHistoricalDataClient | None,
    capital: float,
    n_picks: int,
    duration_hours: float = 6.08,
    dry_run: bool = False,
    run_once: bool = False,
    log_file: str = "",
    report_save_path: str = "",
    run_timestamp: str = "",
    min_score: float = 0.8,
    min_basket_size: int = 5,
    momentum_mult: float | None = 2.0,
    momentum_threshold: float = 0.05,
    max_per_sector: int = 3,
    use_blacklist: bool = True,
    no_llm: bool = False,
) -> dict:

    """
    Executes the session run strictly for volatile stocks and returns computed metrics.
    """
    log.info(SEP)
    log.info(f"  VOLATILE STOCKS LIVE SESSION: Market={market.upper()} | Duration={duration_hours}h")
    log.info(SEP)
    
    sentiment_report = None
    
    # Market parameters
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
        
    state_file_path = os.path.join(ROOT, f"live_state_{market}_volatile.json")
    
    # State tracking
    picks = []
    tracking_states = {}
    watchlist = []
    trades_entered = False
    time_entry = None
    time_flatten = None
    last_trade_date = None
    last_reselection_hour = -1  # tracks which clock-hour last ran re-selection
    
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
    if not dry_run and trading_client is not None:
        try:
            account = trading_client.get_account()
            log.info(f"[ALPACA] Connected. Account Cash Available: ${float(account.cash):,.2f}")
        except Exception as e:
            log.error(f"[ALPACA] Connectivity check failed: {e}")
            api_issues.append(f"Alpaca connectivity check failed: {e}")
            termination_reason = "CONNECTION_FAILURE"
            raise e
            
    try:
        log.info("Starting volatile execution loop...")
        while True:
            now = datetime.now(market_tz)
            current_time = now.time()
            
            # Weekend Check
            if now.weekday() >= 5:
                log.info("Weekend detected. Execution suspended. Sleeping for 1 hour...")
                time.sleep(3600)
                continue
                
            today_date = pd.Timestamp(now.date()).tz_localize(market_tz)
            
            if last_trade_date is not None and today_date != last_trade_date:
                log.info(f"New trading day detected ({today_date.date()}). Exiting daily loop.")
                break
                
            last_trade_date = today_date
            
            # Initialize / Load state
            if time_entry is None:
                saved_state = load_live_state(state_file_path, today_date)
                if saved_state is not None:
                    time_entry = saved_state["time_entry"]
                    time_flatten = saved_state["time_flatten"]
                    trades_entered = saved_state["trades_entered"]
                    picks = saved_state["picks"]
                    tracking_states = saved_state["tracking_states"]
                    volatile_list = saved_state.setdefault("volatile_list", [])
                    nonvolatile_list = saved_state.setdefault("nonvolatile_list", [])
                    last_reselection_hour = saved_state.get("last_reselection_hour", -1)
                    watchlist = volatile_list
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
                    save_live_state(state_file_path, today_date, time_entry, time_flatten, trades_entered, picks, tracking_states, volatile_list, nonvolatile_list)
                    
            time_recovery_threshold = (datetime.combine(now.date(), time_entry) + timedelta(minutes=5)).time()
            time_market_close = market_close
            
            # 1. MORNING ENTRY STAGE
            if time_entry <= current_time < time_flatten and not trades_entered:
                log.info(f"\n[ENTRY PHASE] Market is open. Current Time: {now.strftime('%H:%M:%S')} {market_tz}")
                
                # Dynamic top 40 volatile stocks of last 15 days screening
                try:
                    volatile_list, atr_ranked = select_top_40_volatile(market, market_tz)
                    nonvolatile_list = []
                    selection_timestamp = datetime.now(market_tz)
                except Exception as e:
                    log.error(f"Failed to identify top 40 volatile stocks: {e}")
                    raise e
                
                watchlist = volatile_list
                
                # Fetch 5-minute bars for these 40 volatile tickers
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
                    
                try:
                    log.info("Calculating ATR values for watchlist...")
                    atr_values = compute_watchlist_atr(panels, today_date)
                except Exception as e:
                    log.warning(f"Error calculating ATR values: {e}")
                    atr_values = {}
                    
                active_universe_tickers = volatile_list
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

                # ── LLM ENTRY DECISION LAYER ───────────────────────────
                if not no_llm:
                    try:
                        log.info("\n[LLM ENTRY LAYER] Validating stock picks with Gemini 2.5 Flash...")
                        from ai_decision import DecisionEngine
                        llm_engine = DecisionEngine.from_config()
                        picks = llm_engine.evaluate_entry(
                            picks=picks,
                            panels=filtered_panels,
                            nifty_close=nifty_close,
                            today_date=today_date,
                            capital=capital,
                            market=market,
                        )
                    except Exception as e:
                        log.warning(f"[LLM ENTRY ERROR] Validation error: {e}. Proceeding with original picks.")
                # ───────────────────────────────────────────────────────

                    
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
                    
                    log.info(f"Overriding entry prices and allocations at entry time: {time_entry.strftime('%H:%M')} {market_tz}")
                    for pick in picks:
                        ticker = pick["ticker"]
                        price = entry_prices.get(ticker, np.nan)
                        if not pd.isna(price) and price > 0:
                            pick["entry_price"] = float(price)
                            pick["shares"] = int(per_stock_capital / price)
                            pick["entry_bar_idx"] = entry_idx_safe
                            
                for pick in picks:
                    pick["group"] = "volatile"
                    pick["atr_value"] = atr_values.get(pick["ticker"], None)
                        
                # Check for crash recovery / mid-day startup
                if current_time >= time_recovery_threshold:
                    log.info("\n[CRASH RECOVERY] Mid-day startup detected. Reconstructing state...")
                    tracking_states = get_expected_state_so_far(picks, filtered_panels, today_date, time_entry, time_flatten)
                    
                    actual_positions = {}
                    if not dry_run and trading_client is not None:
                        try:
                            al_positions = trading_client.get_all_positions()
                            actual_positions = {p.symbol: int(float(p.qty)) for p in al_positions}
                        except Exception as e:
                            log.error(f"Could not retrieve active Alpaca positions: {e}")
                            api_issues.append(f"Alpaca get_all_positions failed: {e}")
                            
                    log.info(f"Syncing reconstructed state with broker...")
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
                        state["entry_time"] = start_time
                    trades_entered = True
                else:
                    if not picks:
                        log.warning("No stock picks today. Suspending trading.")
                        termination_reason = "NO_PICKS"
                        break
                        
                    log.info("\n[ORDER SUBMISSION] Submitting entry orders...")
                    for pick in picks:
                        ticker = pick["ticker"]
                        qty = pick["shares"]
                        success = execute_order(trading_client, ticker, qty, OrderSide.BUY, dry_run)
                        
                        if success:
                            tracking_states[ticker] = {
                                "ticker": ticker,
                                "group": "volatile",
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
                    last_reselection_hour = now.hour
                    save_live_state(state_file_path, today_date, time_entry, time_flatten, trades_entered, picks, tracking_states, volatile_list, nonvolatile_list, last_reselection_hour)
                    
                if run_once:
                    log.info("[RUN ONCE] Entry stage complete. Exiting script.")
                    break
                    
            # 2. INTRADAY MONITORING STAGE
            elif time_entry <= current_time < time_flatten and trades_entered:
                active_count = sum(1 for s in tracking_states.values() if s["active"])

                # ── HOURLY RE-SELECTION ─────────────────────────────────────
                # Every clock-hour boundary, re-run stock screening and enter
                # new picks alongside any still-open positions from prior cycles.
                current_hour = current_time.hour
                mins_to_flatten = (
                    market_tz.localize(datetime.combine(now.date(), time_flatten)) - now
                ).total_seconds() / 60.0

                if current_hour > last_reselection_hour and mins_to_flatten >= 60:
                    log.info(f"\n[HOURLY RE-SELECTION] Hour boundary {current_hour}:00 reached. "
                             f"Re-running stock selection ({active_count} positions currently active)...")
                    last_reselection_hour = current_hour
                    try:
                        # 1. Re-screen top 40 volatile stocks
                        new_volatile_list, _ = select_top_40_volatile(market, market_tz)
                        watchlist = list(dict.fromkeys(volatile_list + new_volatile_list))  # preserve + extend
                        volatile_list = new_volatile_list
                        selection_timestamp = datetime.now(market_tz)

                        # 2. Fetch fresh 5-min panels
                        re_panels, re_nifty_close = fetch_yfinance_panels(
                            tickers=new_volatile_list,
                            index_ticker=index_ticker,
                            lookback_days=35,
                            market_tz=market_tz,
                            market_open=market_open,
                            market_close=market_close
                        )
                        re_atr_values = compute_watchlist_atr(re_panels, today_date)

                        re_filtered_panels = {}
                        for key in ["close", "open", "high", "low", "volume"]:
                            cols = [c for c in new_volatile_list if c in re_panels[key].columns]
                            re_filtered_panels[key] = re_panels[key][cols].copy()

                        # 3. Compute remaining investable capital
                        invested_capital = sum(
                            s["qty"] * s["entry_price"]
                            for s in tracking_states.values()
                            if s.get("active") and s.get("qty", 0) > 0
                        )
                        remaining_capital = max(capital - invested_capital, 0.0)
                        if remaining_capital < 1000:
                            log.info("[HOURLY RE-SELECTION] Remaining capital too low for new positions. Skipping.")
                        else:
                            # 4. Run StockPicker on remaining capital
                            re_picker = StockPicker(
                                panels=re_filtered_panels,
                                nifty_close=re_nifty_close,
                                capital=remaining_capital,
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
                            re_picks = re_picker.pick(today_date)

                            # 5. Exclude tickers already tracked this session
                            existing_tickers = set(tracking_states.keys())
                            re_picks = [p for p in re_picks if p["ticker"] not in existing_tickers]

                            if not re_picks:
                                log.info("[HOURLY RE-SELECTION] No new picks after excluding already-traded tickers.")
                            else:
                                # 6. Set entry prices from current bar
                                re_day_mask = re_filtered_panels["open"].index.normalize() == today_date
                                re_day_open = re_filtered_panels["open"][re_day_mask]
                                if len(re_day_open) > 0:
                                    re_entry_prices = re_day_open.iloc[-1]
                                else:
                                    re_entry_prices = re_filtered_panels["close"].iloc[-1]
                                re_per_stock_capital = remaining_capital / len(re_picks)
                                for pick in re_picks:
                                    ticker = pick["ticker"]
                                    price = re_entry_prices.get(ticker, np.nan)
                                    if not pd.isna(price) and price > 0:
                                        pick["entry_price"] = float(price)
                                        pick["shares"] = int(re_per_stock_capital / price)
                                    pick["group"] = "volatile"
                                    pick["atr_value"] = re_atr_values.get(ticker, None)

                                re_picks = [p for p in re_picks if p.get("shares", 0) > 0]

                                if not re_picks:
                                    log.info("[HOURLY RE-SELECTION] No valid picks with valid share counts.")
                                else:
                                    # 7. LLM entry validation (With LLM variant; no Sentiment in this variant)
                                    if not no_llm:
                                        try:
                                            log.info("[HOURLY RE-SELECTION] Running LLM entry validation...")
                                            from ai_decision import DecisionEngine
                                            re_llm_engine = DecisionEngine.from_config()
                                            re_picks = re_llm_engine.evaluate_entry(
                                                picks=re_picks,
                                                panels=re_filtered_panels,
                                                nifty_close=re_nifty_close,
                                                today_date=today_date,
                                                capital=remaining_capital,
                                                market=market,
                                            )
                                        except Exception as _le:
                                            log.warning(f"[HOURLY RE-SELECTION] LLM validation failed: {_le}. Proceeding.")

                                    if not re_picks:
                                        log.info("[HOURLY RE-SELECTION] All new picks filtered out by LLM.")
                                    else:
                                        # 8. Submit new entry orders
                                        log.info(f"[HOURLY RE-SELECTION] Entering {len(re_picks)} new position(s): "
                                                 f"{[p['ticker'] for p in re_picks]}")
                                        for pick in re_picks:
                                            ticker = pick["ticker"]
                                            qty = pick["shares"]
                                            success = execute_order(trading_client, ticker, qty, OrderSide.BUY, dry_run)
                                            if success:
                                                tracking_states[ticker] = {
                                                    "ticker": ticker,
                                                    "group": "volatile",
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
                                                picks.append(pick)

                                        # Sync entry prices with broker
                                        if not dry_run and trading_client is not None and re_picks:
                                            time.sleep(3)
                                            try:
                                                re_positions = trading_client.get_all_positions()
                                                re_pos_dict = {p.symbol: float(p.avg_entry_price) for p in re_positions}
                                                for pick in re_picks:
                                                    t = pick["ticker"]
                                                    if t in re_pos_dict and t in tracking_states:
                                                        tracking_states[t]["entry_price"] = re_pos_dict[t]
                                                        tracking_states[t]["high_water"] = re_pos_dict[t]
                                            except Exception as _ps:
                                                log.warning(f"[HOURLY RE-SELECTION] Price sync failed: {_ps}")

                                        save_live_state(
                                            state_file_path, today_date, time_entry, time_flatten,
                                            trades_entered, picks, tracking_states,
                                            volatile_list, nonvolatile_list, last_reselection_hour
                                        )
                                        active_count = sum(1 for s in tracking_states.values() if s["active"])
                                        log.info(f"[HOURLY RE-SELECTION] Complete. Total active positions: {active_count}")
                    except Exception as _re_err:
                        log.warning(f"[HOURLY RE-SELECTION] Failed with error: {_re_err}. Continuing with existing positions.")
                # ── END HOURLY RE-SELECTION ────────────────────────────────

                if active_count == 0:
                    log.info(f"No active positions remaining. Sleeping 30s (next hourly check at hour {last_reselection_hour + 1}:00)...")
                    time.sleep(30)
                    continue

                mins_now = now.minute
                secs_now = now.second
                sleep_sec = ((4 - (mins_now % 5)) * 60) + (60 - secs_now) + 20
                log.info(f"[MONITORING] {active_count} active trades. Sleeping {sleep_sec}s until next bar...")
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
                
                # Unpack execution parameters
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
                
                time_exit_1_min = EXEC_PARAMS["time_exit_1_minutes"]
                time_exit_1_sell = EXEC_PARAMS["time_exit_1_sell_pct"]
                time_exit_2_min = EXEC_PARAMS["time_exit_2_minutes"]
                time_exit_2_sell = EXEC_PARAMS["time_exit_2_sell_pct"]
                market_stress_threshold = EXEC_PARAMS["market_stress_threshold"]
                extreme_move_pct = EXEC_PARAMS["extreme_move_pct"]
                max_portfolio_dd = EXEC_PARAMS["max_portfolio_drawdown"]
                
                # Market Stress Detection
                try:
                    if index_close_series is not None and not index_close_series.empty:
                        idx_day_close = index_close_series[day_mask].dropna()
                        if len(idx_day_close) >= 2:
                            idx_open = idx_day_close.iloc[0]
                            idx_current = idx_day_close.iloc[-1]
                            idx_intraday_ret = (idx_current - idx_open) / idx_open
                            if abs(idx_intraday_ret) > market_stress_threshold:
                                log.warning(f"  [MARKET STRESS] {index_ticker} return: {idx_intraday_ret*100:.2f}% exceeds limit. Force-flattening.")
                                for t_ticker, t_state in tracking_states.items():
                                    if t_state["active"] and t_state["qty"] > 0:
                                        execute_order(trading_client, t_ticker, t_state["qty"], OrderSide.SELL, dry_run)
                                        t_state["qty"] = 0
                                        t_state["active"] = False
                                        t_state["exit_reason"] = "MARKET_STRESS_EXIT"
                                        t_state["exits"].append((t_state["initial_qty"], idx_current, now))
                                save_live_state(state_file_path, today_date, time_entry, time_flatten, trades_entered, picks, tracking_states, volatile_list, nonvolatile_list)
                                continue
                except Exception as e:
                    log.warning(f"  Market stress check failed: {e}")
                
                # Portfolio Drawdown Check
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
                        log.warning(f"  [DRAWDOWN PROTECTION] Portfolio drawdown {dd_from_peak*100:.2f}% exceeds limit. Force-flattening.")
                        for t_ticker, t_state in tracking_states.items():
                            if t_state["active"] and t_state["qty"] > 0:
                                execute_order(trading_client, t_ticker, t_state["qty"], OrderSide.SELL, dry_run)
                                t_state["qty"] = 0
                                t_state["active"] = False
                                t_state["exit_reason"] = "DRAWDOWN_PROTECTION"
                                last_px = current_prices.get(t_ticker, t_state["entry_price"])
                                t_state["exits"].append((t_state["initial_qty"], last_px, now))
                        save_live_state(state_file_path, today_date, time_entry, time_flatten, trades_entered, picks, tracking_states, volatile_list, nonvolatile_list)
                        continue
                

                current_bar_check_prices = {}
                active_positions_value = 0.0

                # ── LLM EXIT DECISION LAYER ───────────────────────────
                llm_exit_decisions = {}
                if not no_llm:
                    try:
                        from ai_decision import DecisionEngine
                        llm_engine_exit = DecisionEngine.from_config()
                        llm_exit_decisions = llm_engine_exit.evaluate_exit(
                            tracking_states=tracking_states,
                            today_close=today_close,
                            today_high=today_high,
                            today_low=today_low,
                            last_idx=last_idx,
                            index_data=index_close_series,
                            capital=capital,
                            now=now,
                        )
                    except Exception as e:
                        log.warning(f"[LLM EXIT ERROR] Validation error: {e}. Proceeding with standard risk rules.")
                # ───────────────────────────────────────────────────────

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
                             ticker, "volatile", entry_price, bar_close, bar_low, bar_high, status_str)
                    
                    # ── LLM Exit Action Check ──────────────────────────────
                    if ticker in llm_exit_decisions:
                        llm_dec = llm_exit_decisions[ticker]
                        if llm_dec.action.upper() == "SELL" and state["active"] and state["qty"] > 0:
                            log.info(f"  [LLM EXIT SELL] LLM recommended proactive exit for {ticker} (Reason: {llm_dec.reasoning})")
                            sold = state["qty"]
                            if execute_order(trading_client, ticker, sold, OrderSide.SELL, dry_run):
                                state["qty"] = 0
                                state["active"] = False
                                state["exit_reason"] = "LLM_PROACTIVE_SELL"
                                state["exits"].append((sold, bar_close, now))
                                continue
                        elif llm_dec.action.upper() == "REDUCE" and state["active"] and state["qty"] > 1:
                            sold = max(1, int(state["qty"] * 0.5))
                            log.info(f"  [LLM EXIT REDUCE] LLM recommended partial exit for {ticker} ({sold} shares) (Reason: {llm_dec.reasoning})")
                            if execute_order(trading_client, ticker, sold, OrderSide.SELL, dry_run):
                                state["qty"] -= sold
                                state["exits"].append((sold, bar_close, now))
                        elif llm_dec.action.upper() == "TIGHTEN_STOPS" and llm_dec.adjusted_trail_trigger:
                            trail_trigger = min(trail_trigger, llm_dec.adjusted_trail_trigger)
                            log.info(f"  [LLM TIGHTEN STOPS] Adjusted trailing trigger for {ticker} to {trail_trigger*100:.2f}%")
                    # ───────────────────────────────────────────────────────

                    
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
                                
                    # Extreme Move Detection
                    if extreme_move_pct is not None and extreme_move_pct > 0:
                        sess_open = today_close.iloc[0][ticker] if len(today_close) > 0 else np.nan
                        if pd.isna(sess_open) or sess_open <= 0:
                            sess_open = entry_price
                        
                        move_high = (bar_high - sess_open) / sess_open
                        move_low = (bar_low - sess_open) / sess_open
                        if abs(move_high) >= extreme_move_pct or abs(move_low) >= extreme_move_pct:
                            log.info(f"  [EXTREME MOVE] Breached: {ticker} High: {bar_high:.2f}, Low: {bar_low:.2f} relative to Session Open: {sess_open:.2f}")
                            qty_to_sell = state["qty"]
                            if qty_to_sell > 0:
                                if execute_order(trading_client, ticker, qty_to_sell, OrderSide.SELL, dry_run):
                                    state["qty"] = 0
                                    state["active"] = False
                                    state["exit_reason"] = "EXTREME_MOVE"
                                    state["exits"].append((qty_to_sell, bar_close, now))
                                    continue

                    # Time-Based Partial Exits
                    minutes_elapsed = 0.0
                    if state.get("entry_time") is not None:
                        elapsed_td = now - state["entry_time"]
                        minutes_elapsed = elapsed_td.total_seconds() / 60.0

                    if not state.setdefault("time_exit_1_done", False) and minutes_elapsed >= time_exit_1_min:
                        if bar_close > entry_price:
                            qty_to_sell = int(state["qty"] * time_exit_1_sell)
                            if qty_to_sell > 0:
                                log.info(f"  [TIME EXIT 1] Breached: {ticker} elapsed: {minutes_elapsed:.1f}m >= {time_exit_1_min}m | Selling {qty_to_sell} shares")
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
                                log.info(f"  [TIME EXIT 2] Breached: {ticker} elapsed: {minutes_elapsed:.1f}m >= {time_exit_2_min}m | Selling {qty_to_sell} shares")
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
                        qty_to_sell = state["qty"]
                        if qty_to_sell > 0:
                            log.info(f"  [PROFIT TAKE 3] Breached: {ticker} High: {bar_high:.2f} >= Target: {target_3:.2f}")
                            if execute_order(trading_client, ticker, qty_to_sell, OrderSide.SELL, dry_run):
                                state["qty"] = 0
                                state["exits"].append((qty_to_sell, target_3, now))
                                state["pt3_done"] = True
                                state["active"] = False
                                state["exit_reason"] = "PROFIT_TAKE_3"
                        else:
                            state["pt3_done"] = True
                            state["active"] = False
                            state["exit_reason"] = "PROFIT_TAKE_3"
                        continue
                            
                # Save state
                save_live_state(state_file_path, today_date, time_entry, time_flatten, trades_entered, picks, tracking_states, volatile_list, nonvolatile_list, last_reselection_hour)
                current_portfolio_value = calculate_portfolio_value(capital, tracking_states, current_bar_check_prices)
                portfolio_history.append((last_timestamp, current_portfolio_value))
                
                exposure_pct_current = (active_positions_value / capital) * 100.0
                exposure_pcts.append(exposure_pct_current)
                
                log.info(f"Current Portfolio Value: ${current_portfolio_value:,.2f} | Exposure: {exposure_pct_current:.2f}%")
                
            # 3. DYNAMIC FLATTEN STAGE
            elif (time_flatten <= current_time <= time_market_close) and trades_entered:
                log.info(f"\n[FLATTEN PHASE] Exiting all remaining positions at exit time ({time_flatten.strftime('%I:%M %p')})...")
                
                if not dry_run and trading_client is not None:
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
                log.info(f"[CLOSED] Market is closed. Current Time: {now.strftime('%H:%M:%S')} {market_tz}. Waiting for next session...")
                if run_once or dry_run:
                    log.info("[CLOSED] Exiting loop because run_once or dry_run is set.")
                    termination_reason = "MARKET_CLOSED"
                    break
                time.sleep(300)
                
    except KeyboardInterrupt:
        log.warning(f"\n[VOLATILE] Manual stop detected via KeyboardInterrupt.")
        termination_reason = "MANUAL_STOP"
        manual_stop = True
        log.info("Execution stopped.")
        graceful_shutdown(trading_client, tracking_states, dry_run)
    except Exception as e:
        log.error(f"\n[VOLATILE] Runtime Exception: {e}", exc_info=True)
        termination_reason = "EXCEPTION"
        runtime_errors.append(str(e))
        log.info("Execution stopped.")
        graceful_shutdown(trading_client, tracking_states, dry_run)
    finally:
        log.info(f"[VOLATILE] Post-session summary starts. Generating reports...")
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
                "group": "volatile",
                "entry_price": state["entry_price"],
                "initial_qty": state["initial_qty"],
                "exits": state["exits"],
                "entry_time": state.get("entry_time", start_time),
                "exit_reason": state.get("exit_reason", "")
            })
            
        exposure_pct = np.mean(exposure_pcts) if exposure_pcts else 0.0
        
        index_ret_val = None
        idx_src = None
        if 'idx_day_close' in locals() and idx_day_close is not None and len(idx_day_close) >= 2:
            idx_src = idx_day_close
        elif 'index_close_series' in locals() and index_close_series is not None and len(index_close_series) >= 2:
            idx_src = index_close_series

        if idx_src is not None and len(idx_src) >= 2:
            try:
                index_ret_val = ((float(idx_src.iloc[-1]) - float(idx_src.iloc[0])) / float(idx_src.iloc[0])) * 100.0
            except Exception:
                pass

        log.info(f"Report text generation started for VOLATILE...")
        from live_report_generator import generate_individual_report_text
        session_metrics, report_text = generate_individual_report_text(
            market=market,
            universe_type=universe_type,
            universe_size=len(watchlist),
            volatile_list=volatile_list,
            nonvolatile_list=[],
            ranking_method="trailing_atr_15d",
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
            save_path=report_save_path if report_save_path else "",
            sentiment_summary=sentiment_report,
            index_return=index_ret_val,
        )
        
        if report_save_path:
            try:
                os.makedirs(os.path.dirname(report_save_path), exist_ok=True)
                with open(report_save_path, "w", encoding="utf-8") as f:
                    f.write(report_text)
                log.info(f"Report successfully saved to {report_save_path}")
                try:
                    from update_volatile_excel import update_volatile_excel
                    update_volatile_excel()
                    log.info("Automatically updated Excel volatile results summary files.")
                except Exception as ex_excel:
                    log.warning(f"Could not auto-update Excel results summary files: {ex_excel}")
            except Exception as e:
                log.error(f"Failed to write report to {report_save_path}: {e}")
                
        clear_live_state(state_file_path)
        
        return session_metrics, report_text


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NASDAQ/NSE Mean-Reversion Live Execution on Top 40 Volatile Stocks",
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
    p.add_argument("--picks", type=int, default=40,
                   help="Number of stocks to select for daily trading")
    p.add_argument("--run-once", action="store_true",
                   help="Process only the immediate cycle (entry or exit) and exit immediately")
    p.add_argument("--duration", type=float, default=6.08,
                   help="Duration in hours to hold positions before flattening")
    p.add_argument("--min-score", type=float, default=0.8,
                   help="Minimum z-score to qualify a stock pick")
    p.add_argument("--min-basket-size", type=int, default=5,
                   help="Minimum number of stocks to select for daily trading")
    p.add_argument("--momentum-mult", type=float, default=2.0,
                   help="Volatility multiplier (x ATR) for dynamic momentum filtering. Set to 0 to disable.")
    p.add_argument("--momentum-threshold", type=float, default=0.05,
                   help="Static momentum threshold (used if momentum-mult is disabled)")
    p.add_argument("--max-per-sector", type=int, default=3,
                   help="Maximum number of stocks allowed in the same sector. Set to 0 to disable.")
    p.add_argument("--disable-blacklist", action="store_true",
                   help="Disable the known-loser blacklist of chronic losers")
    p.add_argument("--no-llm", action="store_true",
                   help="Disable LLM Decision Layer (fallback to pure alpha engine)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--market", type=str, default="us",
                   choices=["us", "india"],
                   help="Market selection: 'us' or 'india'")
    
    args = p.parse_args()
    args.market = args.market.lower()
    return args



if __name__ == "__main__":
    args = _parse_args()
    load_dotenv()
    
    market = args.market
    
    # Set up reporting folders inside live_volatile_results (or Indian_log_volatile for Indian market)
    if market == "india":
        results_dir = os.path.join(ROOT, "Indian_log_volatile")
    else:
        results_dir = os.path.join(ROOT, "live_volatile_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup run timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup log file
    log_file_path = os.path.join(results_dir, f"live_volatile_run_{run_timestamp}.log")
    
    # Setup logging
    setup_logging(args.log_level, log_file_path)
    
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
    
    # Get unique report file path formatted as US_VOLATILE_<timestamp>.txt
    report_save_path = get_unique_filepath(
        results_dir,
        f"{market.upper()}_VOLATILE",
        run_timestamp,
        ".txt"
    )
    
    try:
        log.info(SEP)
        log.info(f"STARTING RUN for Market: {market.upper()} | Universe: VOLATILE | Timestamp: {run_timestamp}")
        log.info(SEP)
        
        metrics, report_text = run_universe_session(
            market=market,
            universe_type="VOLATILE",
            watchlist_seed=[],  # Not used directly, watchlist dynamically built inside entry phase
            trading_client=trading_client,
            data_client=data_client,
            capital=args.capital,
            n_picks=args.picks,
            duration_hours=args.duration,
            dry_run=args.dry_run,
            run_once=args.run_once,
            log_file=log_file_path,
            report_save_path=report_save_path,
            run_timestamp=run_timestamp,
            min_score=args.min_score,
            min_basket_size=args.min_basket_size,
            momentum_mult=args.momentum_mult if args.momentum_mult > 0 else None,
            momentum_threshold=args.momentum_threshold,
            max_per_sector=args.max_per_sector,
            use_blacklist=not args.disable_blacklist,
            no_llm=args.no_llm,
        )

        
        log.info(SEP)
        log.info(f"COMPLETED RUN for Market: {market.upper()} | Universe: VOLATILE")
        log.info(f"Report saved to: {report_save_path}")
        log.info(f"Log saved to: {log_file_path}")
        try:
            from update_volatile_excel import update_volatile_excel
            update_volatile_excel()
            log.info("Updated Excel volatile results summary files.")
        except Exception as e:
            log.warning(f"Could not auto-update Excel results summary files: {e}")
        log.info(SEP)
            
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt caught in main manager. Aborting.")

