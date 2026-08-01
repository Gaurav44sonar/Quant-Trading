"""
nse_pipeline/orchestrator.py
============================
End-to-end pipeline orchestration for NASDAQ data.

Pipeline steps:
  1. Screen universe (180 volatile + 120 non-volatile)
  2. Fetch intraday 5-min data for selected stocks
  3. Fetch NASDAQ index (QQQ) for beta hedging
  4. Clean data (spikes, OHLC ordering)
  5. Save to Parquet

Usage
-----
    from nse_pipeline.orchestrator import NSEPipeline
    
    pipe = NSEPipeline()
    result = pipe.run()
    panels = result["panels"]
    volatile = result["volatile"]
    nonvolatile = result["nonvolatile"]
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import yaml
import pandas as pd

from nse_pipeline.fetcher import NSEFetcher
from nse_pipeline.cleaner import NSECleaner
from nse_pipeline.storage import save_panels, read_panels
from nse_pipeline.universe import UniverseBuilder, SEED_POOL

log = logging.getLogger(__name__)
SEP = "-" * 60


class NSEPipeline:
    """
    End-to-end pipeline for NASDAQ intraday data.
    
    Parameters
    ----------
    config_path : str
        Path to config.yaml
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        
        uni_cfg = self.config.get("universe", {})
        data_cfg = self.config.get("data", {})
        clean_cfg = self.config.get("cleaning", {})
        
        self.fetcher = NSEFetcher(
            ticker_suffix=data_cfg.get("ticker_suffix", ""),
            rate_limit_sleep=data_cfg.get("rate_limit_sleep", 0.3),
            retry_attempts=data_cfg.get("retry_attempts", 3),
            retry_backoff=data_cfg.get("retry_backoff_s", 5.0),
        )
        
        self.universe_builder = UniverseBuilder(
            seed_pool=SEED_POOL,
            volatile_count=uni_cfg.get("volatile_count", 180),
            nonvolatile_count=uni_cfg.get("non_volatile_count", 120),
            atr_window=uni_cfg.get("atr_window", 20),
            min_adtv_usd=uni_cfg.get("min_adtv_usd", 1_000_000),
            min_price=uni_cfg.get("min_price", 5.0),
            min_history_days=uni_cfg.get("min_history_days", 60),
        )
        
        self.cleaner = NSECleaner(
            max_daily_return=clean_cfg.get("max_daily_return", 0.50),
            max_intraday_range=clean_cfg.get("max_intraday_range", 0.50),
            max_volume_spike_mult=clean_cfg.get("max_volume_spike_mult", 10.0),
            volume_spike_window=clean_cfg.get("volume_spike_window", 20),
            max_overnight_gap=clean_cfg.get("max_overnight_gap", 0.50),
            enforce_ohlc_order=clean_cfg.get("enforce_ohlc_order", True),
        )
        
        self.storage_cfg = self.config.get("storage", {})
    
    def run(
        self,
        days: int = 59,
        skip_universe_screen: bool = False,
        tickers: Optional[list[str]] = None,
    ) -> dict:
        """
        Execute the full pipeline.
        
        Parameters
        ----------
        days : int
            Days of intraday history to fetch (max 59)
        skip_universe_screen : bool
            If True, skip daily fetch for screening (use provided tickers)
        tickers : list[str], optional
            If provided, use these tickers instead of dynamic selection
        
        Returns
        -------
        dict with keys:
            panels: dict of clean OHLCV panels
            volatile: list of volatile tickers
            nonvolatile: list of non-volatile tickers
            nifty_prices: NASDAQ index prices
            nifty_returns: NASDAQ index returns
        """
        t0 = time.perf_counter()
        
        log.info(SEP)
        log.info("  NASDAQ Intraday Data Pipeline")
        log.info("  Exchange: NASDAQ | Freq: 5m | Days: %d", days)
        log.info(SEP)
        
        # ── Step 1: Universe selection ────────────────────────────────────
        if tickers:
            volatile = tickers[:len(tickers) // 2]
            nonvolatile = tickers[len(tickers) // 2:]
            all_tickers = tickers
            log.info("[1/4] Using provided tickers: %d stocks", len(all_tickers))
        elif skip_universe_screen:
            # Use first 300 from seed pool
            all_tickers = SEED_POOL[:300]
            volatile = all_tickers[:180]
            nonvolatile = all_tickers[180:]
            log.info("[1/4] Skipping screen, using seed pool: %d stocks", len(all_tickers))
        else:
            log.info("[1/4] Screening universe (fetching daily data) ...")
            daily_panels = self.fetcher.fetch_daily(SEED_POOL, period="6mo")
            
            if not daily_panels:
                log.error("Failed to fetch daily data for screening")
                return {"panels": {}, "volatile": [], "nonvolatile": []}
            
            volatile, nonvolatile = self.universe_builder.select(daily_panels)
            all_tickers = volatile + nonvolatile
            log.info("  Selected %d tickers (%d volatile + %d non-volatile)",
                     len(all_tickers), len(volatile), len(nonvolatile))
        
        # ── Step 2: Fetch intraday data ──────────────────────────────────
        log.info("[2/4] Fetching 5-min intraday data for %d tickers ...",
                 len(all_tickers))
        panels = self.fetcher.fetch_universe(
            all_tickers, days=days, interval="5m"
        )
        
        if not panels:
            log.error("No intraday data fetched")
            return {"panels": {}, "volatile": volatile, "nonvolatile": nonvolatile}
        
        # ── Step 3: Clean data ───────────────────────────────────────────
        log.info("[3/4] Cleaning data ...")
        panels = self.cleaner.clean(panels)
        
        # ── Step 4: Fetch NASDAQ index for hedging ───────────────────────
        log.info("[4/4] Fetching NASDAQ index (QQQ) ...")
        index_ticker = self.config.get("data", {}).get("index_ticker", "QQQ")
        nifty_prices, nifty_returns = self.fetcher.fetch_index(
            ticker=index_ticker, days=days, interval="5m"
        )
        
        # ── Save to disk ─────────────────────────────────────────────────
        clean_dir = self.storage_cfg.get("clean_dir", "data/clean/")
        save_panels(panels, clean_dir)
        
        # Save universe info
        report_dir = self.storage_cfg.get("report_dir", "reports/")
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        
        universe_info = pd.DataFrame({
            "ticker": all_tickers,
            "group": ["volatile"] * len(volatile) + ["nonvolatile"] * len(nonvolatile),
        })
        universe_info.to_csv(Path(report_dir) / "universe.csv", index=False)
        
        elapsed = time.perf_counter() - t0
        log.info(SEP)
        log.info("  Pipeline complete in %.1fs", elapsed)
        log.info("  Clean data: %s", clean_dir)
        log.info("  Tickers: %d volatile + %d non-volatile = %d total",
                 len(volatile), len(nonvolatile), len(all_tickers))
        log.info(SEP)
        
        return {
            "panels": panels,
            "volatile": volatile,
            "nonvolatile": nonvolatile,
            "nifty_prices": nifty_prices,
            "nifty_returns": nifty_returns,
        }
    
    def _load_config(self, path: str) -> dict:
        """Load YAML config file."""
        p = Path(path)
        if not p.exists():
            log.warning("Config not found: %s — using defaults", path)
            return {}
        
        with open(p, "r") as f:
            return yaml.safe_load(f)
