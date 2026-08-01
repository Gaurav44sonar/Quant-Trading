"""
nse_pipeline/
=============
NASDAQ data pipeline for intraday cross-sectional mean reversion.

Modules
-------
fetcher.py     — yfinance-based data fetcher for NASDAQ stocks
cleaner.py     — OHLCV cleaning for US market data
storage.py     — Parquet read/write for panel data
universe.py    — Dynamic 180+120 universe selection (volatile/non-volatile)
orchestrator.py — End-to-end pipeline orchestration

Quick start
-----------
    from nse_pipeline.orchestrator import NSEPipeline
    
    pipe = NSEPipeline("config/config.yaml")
    panels = pipe.run()
"""
from nse_pipeline.fetcher import NSEFetcher
from nse_pipeline.cleaner import NSECleaner
from nse_pipeline.storage import read_panels, save_panels
from nse_pipeline.universe import UniverseBuilder
