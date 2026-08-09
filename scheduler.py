"""
scheduler.py
============
Render Background Worker Scheduler — Quant Trading Strategies (Without LLM)
=============================================================================

Runs FOUR strategy jobs automatically every trading day:
  1. Without_Sentiment  → India  (NSE):   9:15 AM IST  → ~3:30 PM IST
  2. With_Sentiment     → India  (NSE):   9:15 AM IST  → ~3:30 PM IST
  3. Without_Sentiment  → US (NASDAQ): 9:30 AM EST  → ~4:00 PM EST
  4. With_Sentiment     → US (NASDAQ): 9:30 AM EST  → ~4:00 PM EST

India and US jobs run concurrently (separate threads) within the same day.
After each session finishes, results are synced to Google Sheets.

Usage (Render start command):
    python scheduler.py

Usage (local test / dry-run):
    python scheduler.py --dry-run

Environment Variables (set in Render dashboard):
    ALPACA_API_KEY           — Alpaca paper trading API key
    ALPACA_SECRET_KEY        — Alpaca paper trading secret key
    GOOGLE_SHEET_ID          — Google Sheets ID for results sync
    GOOGLE_SERVICE_ACCOUNT_JSON — Full JSON content of service account key (string)
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import argparse
import threading
import subprocess
from datetime import datetime, date, timedelta
from typing import Dict, Optional

import pytz

# ── Timezone Constants ────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")
EST = pytz.timezone("US/Eastern")
UTC = pytz.utc

# ── Market Windows ────────────────────────────────────────────────────────────
# India (NSE): 9:15 AM – 3:30 PM IST
INDIA_OPEN_H,  INDIA_OPEN_M  = 9, 15
INDIA_CLOSE_H, INDIA_CLOSE_M = 15, 30

# US (NASDAQ): 9:30 AM – 4:00 PM EST
US_OPEN_H,  US_OPEN_M  = 9, 30
US_CLOSE_H, US_CLOSE_M = 16, 0

# ── Strategy Configuration ────────────────────────────────────────────────────
# Based on: python run_live_volatile.py --market us --duration 1.5 --picks 40
# India uses 6.25 hours (full NSE session ≈ 9:15–3:30 = 6h15m)
COMMON_ARGS = ["--picks", "40", "--paper"]

STRATEGY_CONFIGS = [
    {
        "name": "without_sentiment_india",
        "label": "WITHOUT_LLM_WITHOUT_SENTIMENT [INDIA]",
        "folder": "Without LLM/Intraday_Cross_Sectional_Mean_Reversion_Without_Sentiment",
        "market": "india",
        "duration": "6.25",
        "tz": IST,
        "open_h": INDIA_OPEN_H,
        "open_m": INDIA_OPEN_M,
    },
    {
        "name": "with_sentiment_india",
        "label": "WITHOUT_LLM_WITH_SENTIMENT [INDIA]",
        "folder": "Without LLM/Intraday_Cross_Sectional_Mean_Reversion_With_Sentiment",
        "market": "india",
        "duration": "6.25",
        "tz": IST,
        "open_h": INDIA_OPEN_H,
        "open_m": INDIA_OPEN_M,
    },
    {
        "name": "without_sentiment_us",
        "label": "WITHOUT_LLM_WITHOUT_SENTIMENT [US]",
        "folder": "Without LLM/Intraday_Cross_Sectional_Mean_Reversion_Without_Sentiment",
        "market": "us",
        "duration": "6.5",
        "tz": EST,
        "open_h": US_OPEN_H,
        "open_m": US_OPEN_M,
    },
    {
        "name": "with_sentiment_us",
        "label": "WITHOUT_LLM_WITH_SENTIMENT [US]",
        "folder": "Without LLM/Intraday_Cross_Sectional_Mean_Reversion_With_Sentiment",
        "market": "us",
        "duration": "6.5",
        "tz": EST,
        "open_h": US_OPEN_H,
        "open_m": US_OPEN_M,
    },
]

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [scheduler]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("scheduler")

# ── Root directory (where this file lives = repo root) ─────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

# ── State tracking: last run date per strategy ────────────────────────────────
_last_run: Dict[str, Optional[date]] = {cfg["name"]: None for cfg in STRATEGY_CONFIGS}
_lock = threading.Lock()

# ── Google credentials bootstrap ─────────────────────────────────────────────

def _bootstrap_google_credentials() -> None:
    """
    On Render, the service account JSON is injected as an environment variable
    GOOGLE_SERVICE_ACCOUNT_JSON (full JSON string) because we can't commit the
    file to git. This function writes it to disk so gspread can read it.
    """
    json_content = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    creds_path = os.path.join(ROOT, "google_credentials.json")

    if json_content and not os.path.isfile(creds_path):
        try:
            parsed = json.loads(json_content)
            with open(creds_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
            log.info(f"Google credentials written to {creds_path}")
        except Exception as e:
            log.warning(f"Could not write Google credentials from env var: {e}")
    elif os.path.isfile(creds_path):
        log.info("Google credentials file already exists on disk.")
    else:
        log.warning(
            "GOOGLE_SERVICE_ACCOUNT_JSON env var not set and google_credentials.json not found. "
            "Google Sheets sync will be skipped."
        )

# ── Sync results to Google Sheets ─────────────────────────────────────────────

def _sync_to_sheets(strategy_name: str, strategy_label: str, folder: str, market: str) -> None:
    """
    After a strategy session finishes, parse its latest report and push to Google Sheets.
    Strategy label maps to the tab name in google_sheets_sync.STRATEGY_MAP.
    """
    try:
        # Determine results directory
        strategy_dir = os.path.join(ROOT, folder)
        if market == "india":
            results_dir = os.path.join(strategy_dir, "Indian_log_volatile")
        else:
            results_dir = os.path.join(strategy_dir, "live_volatile_results")

        if not os.path.isdir(results_dir):
            log.warning(f"[{strategy_label}] Results directory not found: {results_dir}")
            return

        # Find the most recently modified .txt report
        txt_files = [
            os.path.join(results_dir, f)
            for f in os.listdir(results_dir)
            if f.endswith(".txt") and f.upper().startswith(market.upper())
        ]
        if not txt_files:
            log.warning(f"[{strategy_label}] No report .txt files found in {results_dir}")
            return

        latest_report = max(txt_files, key=os.path.getmtime)
        log.info(f"[{strategy_label}] Syncing latest report to Google Sheets: {os.path.basename(latest_report)}")

        # Import sync module (from repo root)
        sys.path.insert(0, ROOT)
        from google_sheets_sync import parse_report_txt, sync_dataframe_to_tab, STRATEGY_MAP
        import pandas as pd

        # Map strategy_name → tab_name
        # without_sentiment_india → WITHOUT_LLM_WITHOUT_SENTIMENT → "Without LLM Without Sentiment"
        if "without_sentiment" in strategy_name:
            map_key = "WITHOUT_LLM_WITHOUT_SENTIMENT"
        else:
            map_key = "WITHOUT_LLM_WITH_SENTIMENT"

        tab_name = STRATEGY_MAP.get(map_key, "Without LLM Without Sentiment")

        market_label = "INDIA" if market == "india" else "US"
        row_data = parse_report_txt(latest_report, market_label=market_label)
        df = pd.DataFrame([row_data])

        sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
        creds_path = os.path.join(ROOT, "google_credentials.json")

        success = sync_dataframe_to_tab(tab_name, df, sheet_id=sheet_id, creds_path=creds_path)
        if success:
            log.info(f"[{strategy_label}] ✓ Google Sheets sync complete → tab: '{tab_name}'")
        else:
            log.warning(f"[{strategy_label}] Google Sheets sync returned False (check credentials/sheet ID).")

    except Exception as e:
        log.error(f"[{strategy_label}] Google Sheets sync failed: {e}")


# ── Single strategy runner ────────────────────────────────────────────────────

def _run_strategy(cfg: dict, dry_run: bool = False) -> None:
    """
    Launches run_live_volatile.py for a single strategy config as a subprocess.
    Blocks until the process finishes, then triggers Google Sheets sync.
    """
    strategy_name = cfg["name"]
    label = cfg["label"]
    folder = cfg["folder"]
    market = cfg["market"]
    duration = cfg["duration"]

    strategy_dir = os.path.join(ROOT, folder)
    script_path = os.path.join(strategy_dir, "run_live_volatile.py")

    if not os.path.isfile(script_path):
        log.error(f"[{label}] Script not found: {script_path}")
        return

    cmd = [
        sys.executable, script_path,
        "--market", market,
        "--duration", duration,
        "--picks", "40",
        "--paper",
    ]

    if dry_run:
        cmd.append("--dry-run")

    log.info(f"[{label}] ▶ Starting session: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=strategy_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # Stream subprocess output to scheduler log in real-time
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log.info(f"[{label}]   {line}")

        proc.wait()
        rc = proc.returncode
        if rc == 0:
            log.info(f"[{label}] ✓ Session completed successfully (exit code 0).")
        else:
            log.warning(f"[{label}] ⚠ Session exited with code {rc}.")

    except Exception as e:
        log.error(f"[{label}] ✗ Failed to launch strategy: {e}")
        return

    # Post-session: sync results to Google Sheets
    log.info(f"[{label}] Syncing results to Google Sheets...")
    _sync_to_sheets(strategy_name, label, folder, market)


# ── Thread wrapper (records last_run date) ────────────────────────────────────

def _run_strategy_thread(cfg: dict, dry_run: bool, today: date) -> None:
    """Thread target — runs strategy and records the date it ran."""
    _run_strategy(cfg, dry_run=dry_run)
    with _lock:
        _last_run[cfg["name"]] = today
    log.info(f"[{cfg['label']}] Last-run date recorded: {today}")


# ── Market open checker ───────────────────────────────────────────────────────

def _is_market_open_window(cfg: dict) -> tuple[bool, date]:
    """
    Returns (should_run_now, today_date_in_tz).
    Fires only at market open (within a 5-minute grace window after open).
    """
    tz = cfg["tz"]
    now_local = datetime.now(tz)
    today_local = now_local.date()

    # Skip weekends
    if today_local.weekday() >= 5:  # 5=Sat, 6=Sun
        return False, today_local

    open_h = cfg["open_h"]
    open_m = cfg["open_m"]

    # Fire window: [open_time, open_time + 5 minutes]
    open_minute_of_day = open_h * 60 + open_m
    current_minute_of_day = now_local.hour * 60 + now_local.minute

    in_window = open_minute_of_day <= current_minute_of_day <= open_minute_of_day + 4

    return in_window, today_local


# ── Main scheduler loop ───────────────────────────────────────────────────────

def run_scheduler(dry_run: bool = False) -> None:
    """
    Infinite loop that checks every 60 seconds whether any strategy should fire.
    """
    log.info("=" * 70)
    log.info("  Quant-Trading Scheduler Starting")
    log.info(f"  Monitoring {len(STRATEGY_CONFIGS)} strategies")
    log.info(f"  Dry-run mode: {dry_run}")
    log.info("=" * 70)

    _bootstrap_google_credentials()

    # Track active threads to avoid double-launch
    active_threads: Dict[str, threading.Thread] = {}

    while True:
        for cfg in STRATEGY_CONFIGS:
            name = cfg["name"]
            label = cfg["label"]

            should_run, today = _is_market_open_window(cfg)

            if not should_run:
                continue

            with _lock:
                already_ran_today = (_last_run[name] == today)

            if already_ran_today:
                continue

            # Check if still running from a previous thread
            existing = active_threads.get(name)
            if existing and existing.is_alive():
                log.debug(f"[{label}] Thread still running, skipping duplicate launch.")
                continue

            log.info(f"[{label}] ⏰ Market open window detected — launching strategy thread.")

            t = threading.Thread(
                target=_run_strategy_thread,
                args=(cfg, dry_run, today),
                name=f"strategy-{name}",
                daemon=True,
            )
            t.start()
            active_threads[name] = t

        # Sleep 60 seconds between checks
        time.sleep(60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quant-Trading Scheduler for Render — auto-runs strategies at market open."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to all strategy scripts (no real orders placed).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_scheduler(dry_run=args.dry_run)
