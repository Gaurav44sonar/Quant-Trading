import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, time as dt_time
import pytz

# Add project root to path
sys.path.append(os.path.abspath("."))

# Mock log inside modules that import it, or initialize logging
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("regenerate_report")

from run_live_volatile import fetch_yfinance_panels, EST
from live_report_generator import generate_individual_report_text

def main():
    state_file_path = "live_state_us_volatile.json"
    if not os.path.exists(state_file_path):
        print(f"State file {state_file_path} not found!")
        return

    with open(state_file_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    market = "US"
    universe_type = "VOLATILE"
    market_tz = EST
    
    # Parse times
    today_date = pd.Timestamp(state["today_date"])
    time_entry = datetime.strptime(state["time_entry"], "%H:%M:%S").time()
    time_flatten = datetime.strptime(state["time_flatten"], "%H:%M:%S").time()
    
    # Reconstruct timestamps
    start_dt = datetime.combine(today_date.date(), time_entry)
    start_time = market_tz.localize(start_dt)
    
    end_dt = datetime.combine(today_date.date(), time_flatten)
    end_time = market_tz.localize(end_dt)
    
    duration_seconds = (end_time - start_time).total_seconds()
    
    # We can reconstruct tracking states
    tracking_states = state.get("tracking_states", {})
    
    # Let's restore entry_time and exits timestamps in tracking_states
    for ticker, tstate in tracking_states.items():
        if "entry_time" in tstate and tstate["entry_time"]:
            tstate["entry_time"] = datetime.fromisoformat(tstate["entry_time"])
        if "exits" in tstate:
            parsed_exits = []
            for x in tstate["exits"]:
                qty, price, ts_str = x
                ts = datetime.fromisoformat(ts_str) if ts_str else None
                parsed_exits.append((qty, price, ts))
            tstate["exits"] = parsed_exits
            
    # We need final closes to value remaining active states
    watchlist = state.get("volatile_list", [])
    index_ticker = "QQQ"
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    
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
        print(f"Error fetching final closes: {e}")
        
    # Value remaining tracking states at last known prices (just like run_live_volatile.py line 1568)
    for ticker, tstate in tracking_states.items():
        if tstate.get("active") or tstate.get("qty", 0) > 0:
            rem_qty = tstate["qty"]
            last_px = final_closes.get(ticker, tstate["entry_price"])
            
            # Check if this exit isn't already appended
            already_appended = False
            for exit_item in tstate["exits"]:
                if len(exit_item) >= 3 and isinstance(exit_item[2], datetime) and abs((exit_item[2] - end_time).total_seconds()) < 60:
                    already_appended = True
                    break
                    
            if not already_appended and rem_qty > 0:
                tstate["exits"].append((rem_qty, last_px, end_time))
            tstate["qty"] = 0
            tstate["active"] = False
            tstate["exit_reason"] = "TERMINATION_EXIT"
            
    # Clean state exits list for report generator
    trades_list = []
    for ticker, tstate in tracking_states.items():
        trades_list.append({
            "ticker": ticker,
            "group": "volatile",
            "entry_price": tstate["entry_price"],
            "initial_qty": tstate["initial_qty"],
            "exits": tstate["exits"],
            "entry_time": tstate.get("entry_time", start_time),
            "exit_reason": tstate.get("exit_reason", "")
        })
        
    # Reconstruct portfolio history and exposure_pcts from log file!
    portfolio_history = []
    exposure_pcts = []
    log_path = "live_volatile_results/live_volatile_run_20260730_220613.log"
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Current Portfolio Value:" in line:
                    parts = line.split("Current Portfolio Value:")
                    if len(parts) > 1:
                        subparts = parts[1].split("|")
                        val_str = subparts[0].replace("$", "").replace(",", "").strip()
                        exp_str = subparts[1].replace("Exposure:", "").replace("%", "").strip()
                        try:
                            # Reconstruct time from the start of the line (e.g. 2026-07-31 00:30:31)
                            time_part = line[:19]
                            ts = datetime.strptime(time_part, "%Y-%m-%d %H:%M:%S")
                            ts = market_tz.localize(ts)
                            portfolio_history.append((ts, float(val_str)))
                            exposure_pcts.append(float(exp_str))
                        except Exception as ex:
                            print(f"Error parsing log line: {ex}")
                            
    capital = 100000.0
    exposure_pct = np.mean(exposure_pcts) if exposure_pcts else 0.0
    
    report_save_path = "live_volatile_results/US_VOLATILE_20260730_220613.txt"
    
    print(f"Generating report text...")
    session_metrics, report_text = generate_individual_report_text(
        market=market,
        universe_type=universe_type,
        universe_size=len(watchlist),
        volatile_list=watchlist,
        nonvolatile_list=[],
        ranking_method="trailing_atr_15d",
        selection_timestamp=start_time,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration_seconds,
        termination_reason="DURATION_COMPLETED",
        duration_completed=True,
        manual_stop=False,
        runtime_errors=[],
        api_issues=[],
        trades=trades_list,
        capital=capital,
        portfolio_history=portfolio_history,
        exposure_pct=exposure_pct,
        save_path=report_save_path,
        sentiment_summary=None,
    )
    
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved report to {report_save_path}")
    
    # Run the Excel update script to update all Excel summaries
    try:
        from update_volatile_excel import update_volatile_excel
        update_volatile_excel()
        print("Updated Excel sheets successfully.")
    except Exception as e:
        print(f"Error updating Excel: {e}")

if __name__ == "__main__":
    main()
