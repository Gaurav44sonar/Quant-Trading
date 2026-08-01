"""
nse_pipeline/storage.py
=======================
Parquet-based storage for panel data (timestamp × ticker).

Handles read/write of wide-format OHLCV panels to/from Parquet files.
Market-agnostic — works for any exchange.

Usage
-----
    from nse_pipeline.storage import save_panels, read_panels
    
    save_panels(panels, "data/clean/")
    panels = read_panels("data/clean/")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


def save_panels(
    panels: dict[str, pd.DataFrame],
    output_dir: str,
    compression: str = "snappy",
) -> None:
    """
    Save OHLCV panels to Parquet files.
    
    Each field (open, high, low, close, volume) is saved as a
    separate Parquet file in long format: [timestamp, ticker, value].
    
    Parameters
    ----------
    panels : dict[str, DataFrame]
        {"open": df, "high": df, ...} — each [timestamp × ticker]
    output_dir : str
        Directory to write Parquet files
    compression : str
        Parquet compression (default "snappy")
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    for field, df in panels.items():
        path = out / f"{field}.parquet"
        
        # Convert wide → long for efficient Parquet storage
        long = df.stack(future_stack=True).reset_index()
        long.columns = ["timestamp", "ticker", "value"]
        long["timestamp"] = pd.to_datetime(long["timestamp"])
        
        table = pa.Table.from_pandas(long, preserve_index=False)
        pq.write_table(table, path, compression=compression)
        
        log.debug("  Saved %s: %d rows → %s", field, len(long), path)
    
    log.info(
        "Panels saved to %s (%d fields, %d tickers, %d bars)",
        output_dir,
        len(panels),
        len(panels.get("close", pd.DataFrame()).columns),
        len(panels.get("close", pd.DataFrame())),
    )


def read_panels(
    input_dir: str,
    tickers: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: tuple[str, ...] = ("open", "high", "low", "close", "volume"),
) -> dict[str, pd.DataFrame]:
    """
    Read OHLCV panels from Parquet files.
    
    Parameters
    ----------
    input_dir : str
        Directory containing {field}.parquet files
    tickers : list[str], optional
        Filter to specific tickers
    start_date, end_date : str, optional
        Date range filter (YYYY-MM-DD)
    fields : tuple of field names to load
    
    Returns
    -------
    dict[str, DataFrame] — {"open": df, "high": df, ...}
    Each DataFrame: [timestamp × ticker] wide format
    """
    inp = Path(input_dir)
    if not inp.exists():
        log.warning("Panel directory does not exist: %s", input_dir)
        return {}
    
    panels = {}
    
    for field in fields:
        path = inp / f"{field}.parquet"
        if not path.exists():
            log.warning("  Missing: %s", path)
            continue
        
        long = pd.read_parquet(path)
        long["timestamp"] = pd.to_datetime(long["timestamp"])
        
        # Filter by ticker
        if tickers:
            long = long[long["ticker"].isin(tickers)]
        
        # Filter by date
        if start_date:
            long = long[long["timestamp"] >= pd.Timestamp(start_date)]
        if end_date:
            long = long[long["timestamp"] <= pd.Timestamp(end_date)]
        
        # Pivot to wide format
        wide = long.pivot(index="timestamp", columns="ticker", values="value")
        wide = wide.sort_index()
        
        panels[field] = wide
    
    if panels:
        log.info(
            "Panels loaded from %s: %d bars × %d tickers",
            input_dir,
            len(panels.get("close", pd.DataFrame())),
            len(panels.get("close", pd.DataFrame()).columns),
        )
    
    return panels
