"""features/store.py — FeatureStore: persist computed features to Parquet."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

class FeatureStore:
    def __init__(self, root: str = "data/features/"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        log.info("FeatureStore initialized at: %s", self.root)

    def save(self, features: dict[str, pd.DataFrame], compression: str = "snappy") -> None:
        log.info("Saving %d features to %s with compression=%s", len(features), self.root, compression)
        saved_count = 0
        total_size = 0
        
        for name, df in features.items():
            file_size = self._save_one(name, df, compression)
            if file_size > 0:
                saved_count += 1
                total_size += file_size
        
        log.info("✓ Saved %d features | Total: %.2f MB", saved_count, total_size / (1024**2))

    def _save_one(self, name: str, df: pd.DataFrame, compression: str = "snappy") -> int:
        """Save one feature and return file size in bytes."""
        feat_root = self.root / name
        feat_root.mkdir(parents=True, exist_ok=True)
        # Ensure index/columns are named so stack produces correct column names
        df = df.copy()
        df.index.name   = "timestamp"
        df.columns.name = "ticker"
        long = df.stack(future_stack=True).reset_index(name='value')
        long["timestamp"] = pd.to_datetime(long["timestamp"])
        if long["timestamp"].dt.tz is None:
            long["timestamp"] = long["timestamp"].dt.tz_localize("America/New_York")
        long["value"] = long["value"].astype(np.float32)
        
        total_size = 0
        year_count = 0
        
        for year, part in long.groupby(long["timestamp"].dt.year):
            part_dir = feat_root / f"year={year}"
            part_dir.mkdir(parents=True, exist_ok=True)
            schema = pa.schema([
                pa.field("timestamp", pa.timestamp("us", tz="America/New_York")),
                pa.field("ticker",    pa.string()),
                pa.field("value",     pa.float32()),
            ])
            table = pa.Table.from_pandas(
                part[["timestamp","ticker","value"]], schema=schema, preserve_index=False)
            filepath = part_dir/"data.parquet"
            pq.write_table(table, filepath, compression=compression)
            file_size = filepath.stat().st_size
            total_size += file_size
            year_count += 1
            log.debug("  %s/year=%d: %d rows, %.2f MB", name, year, len(part), file_size / (1024**2))
        
        if year_count > 0:
            log.debug("  ✓ %s: %d year partitions, %.2f MB total", name, year_count, total_size / (1024**2))
        
        return total_size

    def load(self, feature_names: Optional[list[str]] = None,
             tickers: Optional[list[str]] = None,
             start_date: Optional[str] = None,
             end_date: Optional[str] = None) -> dict[str, pd.DataFrame]:
        available = self.list_features()
        if not available:
            raise FileNotFoundError(f"No features in {self.root}. Run save() first.")
        names = [n for n in (feature_names or available) if n in available]
        
        log.info("Loading %d features from %s (tickers=%s, dates=%s→%s)", 
                len(names), self.root, 
                f"{len(tickers)} tickers" if tickers else "all",
                start_date or "all", end_date or "latest")
        
        features = {}
        for name in names:
            df = self._load_one(name, tickers, start_date, end_date)
            if df is not None:
                features[name] = df
                log.debug("  ✓ Loaded %s: %s", name, df.shape)
        
        log.info("✓ Loaded %d features from %s", len(features), self.root)
        return features

    def _load_one(self, name: str, tickers, start_date, end_date) -> Optional[pd.DataFrame]:
        feat_root = self.root / name
        if not feat_root.exists():
            log.warning("Feature directory not found: %s", feat_root)
            return None
        
        filters = [("ticker","in",tickers)] if tickers else None
        dataset = pq.ParquetDataset(feat_root, filters=filters)
        long = dataset.read().to_pandas()
        
        if long.empty:
            log.warning("No data found for feature %s", name)
            return None
        
        long["timestamp"] = pd.to_datetime(long["timestamp"]).dt.tz_convert("America/New_York")
        if start_date:
            long = long[long["timestamp"] >= pd.Timestamp(start_date, tz="America/New_York")]
        if end_date:
            long = long[long["timestamp"] <= pd.Timestamp(end_date+" 23:59:59", tz="America/New_York")]
        
        if long.empty:
            log.warning("No data remaining for feature %s after date filtering", name)
            return None
        
        wide = long.pivot_table(index="timestamp", columns="ticker", values="value", aggfunc="last")
        return wide.astype(np.float64)

    def list_features(self) -> list[str]:
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()])

    def feature_info(self) -> pd.DataFrame:
        rows = []
        log.debug("Computing feature storage info...")
        
        for name in self.list_features():
            files = list((self.root/name).rglob("*.parquet"))
            size  = sum(f.stat().st_size for f in files)/1024/1024
            rows.append({"feature": name, "n_files": len(files), "size_mb": round(size,2)})
            log.debug("  %s: %d files, %.2f MB", name, len(files), size)
        
        result = pd.DataFrame(rows).set_index("feature") if rows else pd.DataFrame()
        if not result.empty:
            total = result["size_mb"].sum()
            log.info("Feature storage summary: %d features, %.2f MB total", len(result), total)
        return result
