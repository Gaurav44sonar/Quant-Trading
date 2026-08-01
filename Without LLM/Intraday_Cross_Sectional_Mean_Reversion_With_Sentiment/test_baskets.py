import sys
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nse_pipeline.orchestrator import NSEPipeline
from nse_pipeline.fetcher import NSEFetcher
from nse_pipeline.universe import SEED_POOL
from run_single_day import run_single_day, setup_logging

def compute_basket(daily_panels, target_date, atr_window=20, vol_count=40, nonvol_count=40):
    close = daily_panels["close"]
    high = daily_panels["high"]
    low = daily_panels["low"]
    
    # Filter data up to target_date (exclusive of target date, or inclusive? Usually universe is determined before open)
    # Actually, we can just use data up to target_date - 1 day to be strict, but the prompt doesn't specify.
    # Let's filter up to the target date.
    mask = close.index < pd.Timestamp(target_date).tz_localize(close.index.tz) if close.index.tz else close.index < pd.Timestamp(target_date)
    close = close[mask]
    high = high[mask]
    low = low[mask]
    
    # Filter out stocks with price < 5 or insufficient history
    last_price = close.iloc[-1]
    valid = last_price[last_price >= 5.0].index.tolist()
    
    # compute ATR%
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
    
    atr = tr.rolling(atr_window, min_periods=max(1, atr_window//2)).mean()
    atr_pct = (atr / close.replace(0, pd.NA)).iloc[-1]
    
    atr_ranked = atr_pct.reindex(valid).dropna().sort_values(ascending=False)
    
    volatile = atr_ranked.head(vol_count).index.tolist()
    nonvolatile = atr_ranked.tail(nonvol_count).index.tolist()
    
    return volatile + nonvolatile

def main():
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"test_baskets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging("INFO", str(reports_dir / log_name))
    log = logging.getLogger(__name__)
    
    target_dates = ["2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29"]
    
    fetcher = NSEFetcher()
    log.info("Fetching daily data for seed pool...")
    daily_panels = fetcher.fetch_daily(SEED_POOL, period="6mo")
    
    pipeline = NSEPipeline()
    
    for target_date in target_dates:
        log.info(f"\n==========================================")
        log.info(f"=== Running Backtest for Date: {target_date} ===")
        log.info(f"==========================================")
        
        basket = compute_basket(daily_panels, target_date, atr_window=20, vol_count=40, nonvol_count=0)
        log.info(f"Basket Optimized (100% Volatile) tickers: {len(basket)}")
        
        # Fetch intraday data for the basket
        result = pipeline.run(days=59, skip_universe_screen=True, tickers=basket)
        
        panels = result["panels"]
        nifty_close = result.get("nifty_prices", pd.Series())
        
        if not panels:
            log.error("No intraday data fetched.")
            continue
            
        if hasattr(panels["close"].index, "tz") and panels["close"].index.tz is not None:
            tz = panels["close"].index.tz
            localized_date = pd.Timestamp(target_date).tz_localize(tz)
        else:
            localized_date = pd.Timestamp(target_date)
            
        sl_configs = [
            (0.02, 0.04, 0.08),
            (0.03, 0.06, 0.09)
        ]
        
        for sl_set in sl_configs:
            log.info(f"--- Running with Stop Loss tiers: {sl_set[0]:.0%}, {sl_set[1]:.0%}, {sl_set[2]:.0%} ---")
            run_single_day(
                panels=panels,
                nifty_close=nifty_close,
                target_date=localized_date,
                capital=1_000_000,
                n_picks=80,
                # Keep profit-taking from the best run but converted to ATR multipliers
                # 0.25 ATR ~ 2.5%, 0.5 ATR ~ 5%, 1.0 ATR ~ 10%
                atr_pt_1=0.25,
                atr_pt_2=0.50,
                atr_pt_3=1.00,
                trail_trigger=0.04,
                pt_weights=(0.50, 0.25, 0.25),
                sl_tier1_pct=sl_set[0],
                sl_tier2_pct=sl_set[1],
                sl_tier3_pct=sl_set[2],
                sl_tier1_weight=0.50,
                sl_tier2_weight=0.25,
                sl_tier3_weight=0.25,
            )

if __name__ == "__main__":
    main()
