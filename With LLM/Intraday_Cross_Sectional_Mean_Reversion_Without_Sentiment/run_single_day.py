"""
run_single_day.py
=================
Single-Day Backtester for NASDAQ Intraday Max-Profit Strategy

Simulates the full day-trading workflow:
  1. Pre-open: Score all stocks using overnight & previous-day features
  2. Opening: Confirm picks using first 15 min of trading
  3. Execute: Enter at 9:45 AM ET, manage stops/targets intraday
  4. Exit: Flatten all positions by 3:50 PM ET
  5. Report: Print detailed P&L and trade log

Usage
-----
    python run_single_day.py --date 2026-04-11
    python run_single_day.py --last 5
    python run_single_day.py --from 2026-03-01 --to 2026-04-11
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nse_pipeline.orchestrator import NSEPipeline
from alpha.stock_picker import StockPicker
from alpha.execution import IntradayExecutor

SEP = "=" * 60
THIN = "-" * 60
log = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
        
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def get_top_n_volatile(panels: dict[str, pd.DataFrame], target_date: pd.Timestamp, n: int) -> list[str]:
    """
    Select the top n most volatile tickers based on ATR% over the last 15 trading days
    strictly before target_date.
    """
    close = panels["close"]
    high = panels["high"]
    low = panels["low"]
    
    dates = close.index.normalize()
    unique_dates = sorted(dates.unique())
    
    if target_date not in unique_dates:
        return list(close.columns)
        
    tgt_idx = unique_dates.index(target_date)
    lookback_dates = unique_dates[max(0, tgt_idx - 15):tgt_idx]
    
    if len(lookback_dates) == 0:
        return list(close.columns)
        
    # Build daily close, high, low for the lookback period
    mask = dates.isin(lookback_dates)
    daily_close = close[mask].groupby(dates[mask]).last()
    daily_high = high[mask].groupby(dates[mask]).max()
    daily_low = low[mask].groupby(dates[mask]).min()
    
    # Calculate ATR
    prev_close = daily_close.shift(1).reindex(daily_close.index)
    tr1 = daily_high - daily_low
    tr2 = (daily_high - prev_close).abs()
    tr3 = (daily_low - prev_close).abs()
    tr = tr1.combine(tr2, np.maximum).combine(tr3, np.maximum)
    
    atr = tr.mean()
    last_close = daily_close.iloc[-1]
    atr_pct = atr / last_close.replace(0, np.nan)
    
    # Sort and return top n
    ranked = atr_pct.dropna().sort_values(ascending=False)
    return ranked.head(n).index.tolist()


def run_single_day(
    panels: dict[str, pd.DataFrame],
    nifty_close: pd.Series,
    target_date: pd.Timestamp,
    capital: float = 1_000_000,
    n_picks: int = 80,
    profit_take_1: float = 0.015,
    profit_take_2: float = 0.03,
    profit_take_3: float = 0.045,
    atr_pt_1: float = 0.25,          # Updated: same as stop loss (0.25 ATR)
    atr_pt_2: float = 0.50,          # Updated: same as stop loss (0.50 ATR)
    atr_pt_3: float = 1.00,          # Updated: same as stop loss (1.00 ATR)
    trail_trigger: float = 0.04,     # Old: 0.025
    pt_weights: tuple = (0.50, 0.25, 0.25), # Updated: same as stop loss weights
    sl_tier1_pct: float = 0.025,    # Old: 0.03
    sl_tier2_pct: float = 0.05,     # Old: 0.06
    sl_tier3_pct: float = 0.10,     # Old: 0.12
    sl_tier1_weight: float = 0.50,
    sl_tier2_weight: float = 0.25,
    sl_tier3_weight: float = 0.25,
    volatile: list[str] = None,
    nonvolatile: list[str] = None,
    min_score: float = 0.3,
    min_basket_size: int = 8,
    momentum_mult: float | None = 2.0,
    momentum_threshold: float = 0.05,
    max_per_sector: int = 3,
    use_blacklist: bool = True,
    top_n_volatile: int = None,
) -> dict:
    """
    Run the strategy for a single day.
    
    Returns dict with full day result.
    """
    target_date = pd.Timestamp(target_date).normalize()
    
    # Restrict to top N volatile stocks dynamically
    if top_n_volatile is not None:
        top_tickers = get_top_n_volatile(panels, target_date, top_n_volatile)
        log.info("  Dynamically selected Top %d Volatile Stocks for %s: %s",
                 top_n_volatile, target_date.date(), ", ".join(top_tickers[:5]) + "...")
        
        filtered_panels = {}
        for key in ["close", "open", "high", "low", "volume"]:
            cols = [c for c in top_tickers if c in panels[key].columns]
            filtered_panels[key] = panels[key][cols].copy()
        panels = filtered_panels
    
    # 1. Pick stocks
    picker = StockPicker(
        panels=panels,
        nifty_close=nifty_close,
        capital=capital,
        n_picks=n_picks,
        min_score=min_score,
        min_avg_volume=200_000,
        min_basket_size=min_basket_size,
        momentum_mult=momentum_mult,
        momentum_threshold=momentum_threshold,
        max_per_sector=max_per_sector,
        use_blacklist=use_blacklist,
    )
    
    picks = picker.pick(target_date)
    
    if not picks:
        log.warning("  No picks for %s — skipping", target_date.date())
        return {"date": target_date, "pnl": 0, "trades": [], "status": "no_picks"}
    
    # 2. Execute intraday using backtesting.py
    from alpha.backtesting_strategy import IntradayMaxProfitStrategy, prep_ticker_data
    from backtesting import Backtest
    from alpha.execution import Trade
    
    dates_idx = panels["close"].index.normalize()
    date_mask = dates_idx == target_date
    
    trades = []
    total_pnl = 0.0
    total_capital = 0.0
    
    pick_capital = capital / max(1, len(picks))
    
    for pick in picks:
        ticker = pick["ticker"]
        
        # Calculate ATR using data strictly before target_date to avoid lookahead bias
        atr_val = None
        try:
            hist_mask = panels["close"].index < target_date
            if hist_mask.any():
                df_hist = pd.DataFrame()
                df_hist["High"] = panels["high"][ticker].loc[hist_mask]
                df_hist["Low"] = panels["low"][ticker].loc[hist_mask]
                df_hist["Close"] = panels["close"][ticker].loc[hist_mask]
                
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
        except Exception as e:
            log.warning(f"  Could not compute ATR for {ticker}: {e}")
                
        df = prep_ticker_data(ticker, panels, date_mask)
        if len(df) == 0:
            continue
            
        bt = Backtest(
            df, 
            IntradayMaxProfitStrategy, 
            cash=pick_capital, 
            margin=1.0, 
            trade_on_close=True,
            exclusive_orders=False, # allow multiple partial exits
            finalize_trades=True
        )
        
        # Run with dynamic parameters
        stats = bt.run(
            atr_value=atr_val,
            atr_pt_1=atr_pt_1,
            atr_pt_2=atr_pt_2,
            atr_pt_3=atr_pt_3,
            trailing_stop_trigger=trail_trigger,
            profit_take_1=profit_take_1,
            profit_take_2=profit_take_2,
            profit_take_3=profit_take_3,
            pt_weight_1=pt_weights[0],
            pt_weight_2=pt_weights[1],
            pt_weight_3=pt_weights[2],
            sl_tier1_pct=sl_tier1_pct,
            sl_tier2_pct=sl_tier2_pct,
            sl_tier3_pct=sl_tier3_pct,
            sl_tier1_weight=sl_tier1_weight,
            sl_tier2_weight=sl_tier2_weight,
            sl_tier3_weight=sl_tier3_weight,
        )
        
        pnl = stats['Equity Final [$]'] - pick_capital
        total_pnl += pnl
        total_capital += pick_capital
        
        bt_trades = stats['_trades']
        if len(bt_trades) > 0:
            for _, t in bt_trades.iterrows():
                pnl_pct = t['ReturnPct']
                exit_bar = t['ExitBar']
                
                # Classify exit reason based on exit bar and return percentage
                if exit_bar >= 76:
                    exit_reason = "TIME_EXIT"
                elif pnl_pct < 0:
                    exit_reason = "STOP_LOSS"
                else:
                    exit_reason = "PROFIT_TAKE"
                
                trade = Trade(
                    ticker=ticker,
                    shares=t['Size'],
                    entry_price=t['EntryPrice'],
                    entry_time=t['EntryTime'],
                    entry_bar=t['EntryBar'],
                    exit_price=t['ExitPrice'],
                    exit_time=t['ExitTime'],
                    exit_bar=t['ExitBar'],
                    exit_reason=exit_reason,
                    pnl=t['PnL'],
                    pnl_pct=pnl_pct,
                )
                trades.append(trade)
    
    n_stopped = sum(1 for t in trades if t.exit_reason == "STOP_LOSS")
    n_profit_taken = sum(1 for t in trades if t.exit_reason == "PROFIT_TAKE")
    n_trailing = sum(1 for t in trades if t.exit_reason == "TRAILING_STOP")
    n_time_exit = sum(1 for t in trades if t.exit_reason == "TIME_EXIT")
    
    result = {
        "date": target_date,
        "picks": picks,
        "status": "ok",
        "trades": trades,
        "day_pnl": total_pnl,
        "day_pnl_pct": total_pnl / total_capital if total_capital > 0 else 0,
        "capital_deployed": total_capital,
        "n_stopped": n_stopped,
        "n_profit_taken": n_profit_taken,
        "n_trailing": n_trailing,
        "n_time_exit": n_time_exit,
        "volatile": volatile,
        "nonvolatile": nonvolatile,
    }
    
    # 3. Print report
    _print_day_report(result, capital)
    
    return result


def run_backtest(
    config_path: str = "config/config.yaml",
    days: int = 59,
    target_date: str = None,
    last_n: int = None,
    from_date: str = None,
    to_date: str = None,
    capital: float = 1_000_000,
    n_picks: int = 80,
    profit_take: float = 0.015,
    trail_trigger: float = 0.025,
    skip_universe: bool = True,
    tickers: list[str] = None,
    min_score: float = 0.3,
    min_basket_size: int = 8,
    momentum_mult: float | None = 2.0,
    momentum_threshold: float = 0.05,
    max_per_sector: int = 3,
    use_blacklist: bool = True,
    top_n_volatile: int = None,
) -> list[dict]:
    """
    Run the full backtest pipeline.
    """
    t0 = time.perf_counter()
    
    log.info(SEP)
    log.info("  NASDAQ Single-Day Max-Profit Backtester")
    log.info(SEP)
    
    # ── Fetch Data ────────────────────────────────────────────────────
    log.info("\n[DATA] Fetching market data ...")
    pipeline = NSEPipeline(config_path)
    result = pipeline.run(days=days, skip_universe_screen=skip_universe, tickers=tickers)
    
    panels = result["panels"]
    nifty_close = result.get("nifty_prices", pd.Series())
    volatile = result.get("volatile", [])
    nonvolatile = result.get("nonvolatile", [])
    
    if not panels:
        log.error("No data — aborting")
        return []
    
    close = panels["close"]
    dates_idx = close.index.normalize()
    unique_dates = sorted(dates_idx.unique())
    
    log.info("  Data: %d bars × %d tickers | %d trading days",
             len(close), len(close.columns), len(unique_dates))
    
    # ── Attach backtest-only file logger (after data fetch) ────────────
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    bt_log_name = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    bt_log_path = reports_dir / bt_log_name
    bt_handler = logging.FileHandler(bt_log_path, mode="w", encoding="utf-8")
    bt_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logging.getLogger().addHandler(bt_handler)
    log.info("  Backtest log: %s", bt_log_path)
    log.info(SEP)
    log.info("  BACKTEST CONFIGURATION:")
    log.info("  Capital       : $%sK", f"{capital / 1_000:.0f}")
    log.info("  Stock Picks   : %d", n_picks)
    log.info("  Stop-Loss     : Tiered (2.5%% -> 50%%, 5%% -> 25%%, 10%% -> 25%%)")
    log.info("  Profit Target : %.2f%%", profit_take * 100)
    log.info("  Trail Trigger : %.2f%%", trail_trigger * 100)
    log.info(SEP)
    
    # ── Determine which dates to backtest ──────────────────────────
    # Need at least 20 days of history for features
    min_lookback = 20
    if len(unique_dates) <= min_lookback:
        log.error("Insufficient history: found %d trading days, but need at least %d trading days for feature computation. Please fetch more days (e.g. increase --days to 45 or 59).",
                  len(unique_dates), min_lookback)
        return []
        
    tradeable_dates = unique_dates[min_lookback:]
    
    if target_date:
        td = pd.Timestamp(target_date).normalize()
        if len(tradeable_dates) > 0 and hasattr(tradeable_dates[0], "tz") and tradeable_dates[0].tz:
            td = td.tz_localize(tradeable_dates[0].tz)
        if td in tradeable_dates:
            test_dates = [td]
        else:
            log.error("Date %s not available. Available: %s to %s",
                     td.date(), tradeable_dates[0].date(), tradeable_dates[-1].date())
            return []
    elif last_n:
        test_dates = tradeable_dates[-last_n:]
    elif from_date and to_date:
        fd = pd.Timestamp(from_date).normalize()
        td = pd.Timestamp(to_date).normalize()
        if len(tradeable_dates) > 0 and hasattr(tradeable_dates[0], "tz") and tradeable_dates[0].tz:
            fd = fd.tz_localize(tradeable_dates[0].tz)
            td = td.tz_localize(tradeable_dates[0].tz)
        test_dates = [d for d in tradeable_dates if fd <= d <= td]
    else:
        test_dates = tradeable_dates
        
    if not test_dates:
        log.error("No test dates found matching the requested backtest criteria.")
        return []
    
    log.info("  Backtesting %d days: %s -> %s",
             len(test_dates), test_dates[0].date(), test_dates[-1].date())
    log.info(THIN)
    
    # ── Run each day ──────────────────────────────────────────────────
    all_results = []
    
    for date in test_dates:
        day_result = run_single_day(
            panels=panels,
            nifty_close=nifty_close,
            target_date=date,
            capital=capital,
            n_picks=n_picks,
            profit_take_1=profit_take,
            trail_trigger=trail_trigger,
            volatile=volatile,
            nonvolatile=nonvolatile,
            min_score=min_score,
            min_basket_size=min_basket_size,
            momentum_mult=momentum_mult,
            momentum_threshold=momentum_threshold,
            max_per_sector=max_per_sector,
            use_blacklist=use_blacklist,
            top_n_volatile=top_n_volatile,
        )
        all_results.append(day_result)
    
    # ── Summary ────────────────────────────────────────────────────────
    _print_summary(all_results, capital, volatile, nonvolatile)
    
    elapsed = time.perf_counter() - t0
    log.info("  Backtest complete in %.1fs", elapsed)
    
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _print_day_report(result: dict, capital: float) -> None:
    """Print detailed trade log for a single day."""
    trades = result.get("trades", [])
    day_pnl = result.get("day_pnl", 0)
    date = result.get("date", "")
    
    if isinstance(date, pd.Timestamp):
        date_str = date.strftime("%Y-%m-%d (%A)")
    else:
        date_str = str(date)
    
    log.info("")
    log.info(SEP)
    log.info("  Date: %s", date_str)
    log.info(SEP)
    
    if not trades:
        log.info("  No trades executed.")
        return
    
    # Trade log header
    log.info("  %-12s  %-5s  %-8s  %10s  %10s  %8s  %10s  %s",
             "Ticker", "Side", "Shares", "Entry", "Exit", "P&L%", "P&L $", "Reason")
    log.info("  " + "-" * 85)
    
    for t in trades:
        pnl_str = f"${t.pnl:+,.0f}"
        pnl_pct = f"{t.pnl_pct*100:+.2f}%"
        entry_str = f"${t.entry_price:,.2f}"
        exit_str = f"${t.exit_price:,.2f}"
        
        log.info("  %-12s  %-5s  %-8d  %10s  %10s  %8s  %10s  %s",
                t.ticker, "LONG", t.shares,
                entry_str, exit_str, pnl_pct, pnl_str, t.exit_reason)
    
    log.info("  " + "-" * 85)
    
    # Day summary
    pnl_color = "+" if day_pnl >= 0 else ""
    day_pnl_pct = result.get("day_pnl_pct", 0)
    deployed = result.get("capital_deployed", capital)
    
    log.info("  Day P&L: %s  (%s on %s deployed)",
             f"${day_pnl:+,.0f}", f"{day_pnl_pct*100:+.2f}%", f"${deployed:,.0f}")
    log.info("  Stopped: %d | Profit-taken: %d | Trailing: %d | Time-exit: %d",
             result.get("n_stopped", 0),
             result.get("n_profit_taken", 0),
             result.get("n_trailing", 0),
             result.get("n_time_exit", 0))
             
    # Category-wise trade breakdown
    volatile = result.get("volatile", [])
    nonvolatile = result.get("nonvolatile", [])
    if trades and volatile and nonvolatile:
        vol_trades = [t for t in trades if t.ticker in volatile]
        nonvol_trades = [t for t in trades if t.ticker in nonvolatile]
        
        vol_wins = sum(1 for t in vol_trades if t.pnl > 0)
        nonvol_wins = sum(1 for t in nonvol_trades if t.pnl > 0)
        vol_pnl = sum(t.pnl for t in vol_trades)
        nonvol_pnl = sum(t.pnl for t in nonvol_trades)
        
        vol_wr = vol_wins / len(vol_trades) * 100 if vol_trades else 0.0
        nonvol_wr = nonvol_wins / len(nonvol_trades) * 100 if nonvol_trades else 0.0
        
        log.info("  Category-wise Trade Breakdown:")
        log.info("    %-28s  %d / %d (%.1f%%) | P&L: %s", "- Volatile:", vol_wins, len(vol_trades), vol_wr, f"${vol_pnl:+,.0f}")
        log.info("    %-28s  %d / %d (%.1f%%) | P&L: %s", "- Non-Volatile:", nonvol_wins, len(nonvol_trades), nonvol_wr, f"${nonvol_pnl:+,.0f}")
    log.info(SEP)


def _print_summary(
    results: list[dict],
    capital: float,
    volatile: list[str] = None,
    nonvolatile: list[str] = None,
) -> None:
    """Print multi-day backtest summary."""
    if not results:
        return
    
    valid = [r for r in results if r.get("status") == "ok"]
    if not valid:
        log.info("  No valid trading days found.")
        return
    
    pnls = [r.get("day_pnl", 0) for r in valid]
    pnl_pcts = [r.get("day_pnl_pct", 0) for r in valid]
    
    total_pnl = sum(pnls)
    avg_pnl = np.mean(pnls)
    win_days = sum(1 for p in pnls if p > 0)
    lose_days = sum(1 for p in pnls if p < 0)
    flat_days = sum(1 for p in pnls if p == 0)
    win_rate = win_days / len(pnls) * 100 if pnls else 0
    
    # Category-wise daily breakdown
    vol_pnls = []
    nonvol_pnls = []
    for r in valid:
        day_trades = r.get("trades", [])
        day_vol_pnl = sum(t.pnl for t in day_trades if t.ticker in (volatile or []))
        day_nonvol_pnl = sum(t.pnl for t in day_trades if t.ticker in (nonvolatile or []))
        vol_pnls.append(day_vol_pnl)
        nonvol_pnls.append(day_nonvol_pnl)
        
    vol_win_days = sum(1 for p in vol_pnls if p > 0)
    nonvol_win_days = sum(1 for p in nonvol_pnls if p > 0)
    vol_win_rate = vol_win_days / len(vol_pnls) * 100 if vol_pnls else 0.0
    nonvol_win_rate = nonvol_win_days / len(nonvol_pnls) * 100 if nonvol_pnls else 0.0
    
    # Best and worst days
    best_idx = np.argmax(pnls)
    worst_idx = np.argmin(pnls)
    
    # Count stops
    total_trades = sum(len(r.get("trades", [])) for r in valid)
    total_stops = sum(r.get("n_stopped", 0) for r in valid)
    
    log.info("")
    log.info(SEP)
    log.info("  BACKTEST SUMMARY: %d trading days", len(valid))
    log.info(SEP)
    log.info("")
    log.info("  %-30s  %s", "Total P&L", f"${total_pnl:+,.0f}")
    if volatile and nonvolatile:
        log.info("    %-28s  %s", "- Volatile contribution:", f"${sum(vol_pnls):+,.0f}")
        log.info("    %-28s  %s", "- Non-Volatile contribution:", f"${sum(nonvol_pnls):+,.0f}")
        
    log.info("  %-30s  %.2f%%", "Total Return on Capital",
             total_pnl / capital * 100)
    if volatile and nonvolatile:
        log.info("    %-28s  %+.2f%%", "- Volatile contribution:", sum(vol_pnls) / capital * 100)
        log.info("    %-28s  %+.2f%%", "- Non-Volatile contribution:", sum(nonvol_pnls) / capital * 100)
        
    log.info("  %-30s  %s", "Average Daily P&L", f"${avg_pnl:+,.0f}")
    if volatile and nonvolatile:
        log.info("    %-28s  %s", "- Volatile contribution:", f"${np.mean(vol_pnls):+,.0f}")
        log.info("    %-28s  %s", "- Non-Volatile contribution:", f"${np.mean(nonvol_pnls):+,.0f}")
        
    log.info("  %-30s  %+.3f%%", "Average Daily Return",
             np.mean(pnl_pcts) * 100)
    if volatile and nonvolatile:
        vol_pcts = [p / capital for p in vol_pnls]
        nonvol_pcts = [p / capital for p in nonvol_pnls]
        log.info("    %-28s  %+.3f%%", "- Volatile contribution:", np.mean(vol_pcts) * 100)
        log.info("    %-28s  %+.3f%%", "- Non-Volatile contribution:", np.mean(nonvol_pcts) * 100)
        
    log.info("")
    log.info("  %-30s  %d / %d (%.1f%%)", "Daily Win Rate (Overall)",
             win_days, len(pnls), win_rate)
    if volatile and nonvolatile:
        log.info("    %-28s  %d / %d (%.1f%%)", "- Volatile win rate:", vol_win_days, len(vol_pnls), vol_win_rate)
        log.info("    %-28s  %d / %d (%.1f%%)", "- Non-Volatile win rate:", nonvol_win_days, len(nonvol_pnls), nonvol_win_rate)
        
    log.info("  %-30s  %d", "Losing Days", lose_days)
    log.info("  %-30s  %d", "Flat Days", flat_days)
    log.info("")
    log.info("  %-30s  %s (%s)", "Best Day",
             f"${pnls[best_idx]:+,.0f}", valid[best_idx]["date"].strftime("%Y-%m-%d"))
    log.info("  %-30s  %s (%s)", "Worst Day",
             f"${pnls[worst_idx]:+,.0f}", valid[worst_idx]["date"].strftime("%Y-%m-%d"))
    log.info("")
    log.info("  %-30s  %d", "Total Trades", total_trades)
    log.info("  %-30s  %d (%.1f%%)", "Stop-Losses Hit",
             total_stops, total_stops / max(total_trades, 1) * 100)
             
    # Category-wise Trade Win Rates
    all_trades = [t for r in valid for t in r.get("trades", [])]
    if all_trades:
        total_wins = sum(1 for t in all_trades if t.pnl > 0)
        total_wr = total_wins / len(all_trades) * 100
        log.info("  %-30s  %d / %d (%.1f%%)", "Trade Win Rate (Overall)", total_wins, len(all_trades), total_wr)
        
        if volatile is not None and nonvolatile is not None:
            vol_trades = [t for t in all_trades if t.ticker in volatile]
            nonvol_trades = [t for t in all_trades if t.ticker in nonvolatile]
            
            vol_wins = sum(1 for t in vol_trades if t.pnl > 0)
            nonvol_wins = sum(1 for t in nonvol_trades if t.pnl > 0)
            
            vol_wr = vol_wins / len(vol_trades) * 100 if vol_trades else 0.0
            nonvol_wr = nonvol_wins / len(nonvol_trades) * 100 if nonvol_trades else 0.0
            
            log.info("  %-30s  %d / %d (%.1f%%)", "Trade Win Rate (Volatile)", vol_wins, len(vol_trades), vol_wr)
            log.info("  %-30s  %d / %d (%.1f%%)", "Trade Win Rate (Non-Volatile)", nonvol_wins, len(nonvol_trades), nonvol_wr)
    
    # Sharpe ratio (daily)
    if len(pnl_pcts) > 1:
        daily_sharpe = np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(252)
        log.info("  %-30s  %.3f", "Annualized Sharpe", daily_sharpe)
    
    # Max drawdown
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_dd = drawdown.min()
    log.info("  %-30s  %s", "Max Drawdown", f"${max_dd:,.0f}")
    
    log.info("")
    log.info("  Daily P&L Breakdown:")
    for r in valid:
        pnl = r.get("day_pnl", 0)
        n_trades = len(r.get("trades", []))
        marker = "[+]" if pnl > 0 else ("[-]" if pnl < 0 else "[ ]")
        log.info("    %s  %s  %s  (%d trades)",
                marker, r["date"].strftime("%Y-%m-%d"), f"${pnl:+,.0f}", n_trades)
    
    log.info(SEP)


# ─────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NASDAQ Single-Day Max-Profit Backtester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--days", type=int, default=59,
                   help="Days of intraday history to fetch")
    p.add_argument("--date", default=None,
                   help="Specific date to backtest (YYYY-MM-DD)")
    p.add_argument("--last", type=int, default=None,
                   help="Backtest last N trading days")
    p.add_argument("--from", dest="from_date", default=None)
    p.add_argument("--to", dest="to_date", default=None)
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--picks", type=int, default=80,
                   help="Number of stocks to pick per day")
    p.add_argument("--profit-take", type=float, default=0.015,
                   help="First profit target percentage")
    p.add_argument("--trail-trigger", type=float, default=0.04, # Old: 0.025
                   help="Trailing stop trigger percentage")
    p.add_argument("--skip-universe", action="store_true", default=True)
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--min-score", type=float, default=0.3,
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
    p.add_argument("--top-n-volatile", type=int, default=None,
                   help="Dynamically select and restrict the universe to the top N volatile stocks each day")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    setup_logging(args.log_level)
    
    run_backtest(
        config_path=args.config,
        days=args.days,
        target_date=args.date,
        last_n=args.last,
        from_date=args.from_date,
        to_date=args.to_date,
        capital=args.capital,
        n_picks=args.picks,
        profit_take=args.profit_take,
        trail_trigger=args.trail_trigger,
        skip_universe=args.skip_universe,
        tickers=args.tickers,
        min_score=args.min_score,
        min_basket_size=args.min_basket_size,
        momentum_mult=args.momentum_mult if args.momentum_mult > 0 else None,
        momentum_threshold=args.momentum_threshold,
        max_per_sector=args.max_per_sector,
        use_blacklist=not args.disable_blacklist,
        top_n_volatile=args.top_n_volatile,
    )
