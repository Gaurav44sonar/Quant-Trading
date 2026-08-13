"""
sync_results.py
===============
Post-session helper: reads the latest .txt report from a strategy folder
and pushes it to the correct Google Sheets tab.

Called by GitHub Actions after each run_live_volatile.py session finishes.

Usage:
    python sync_results.py --strategy without_sentiment --market india
    python sync_results.py --strategy with_sentiment    --market us
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [sync]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sync_results")

ROOT = os.path.dirname(os.path.abspath(__file__))

STRATEGY_FOLDERS = {
    # Without LLM strategies
    "without_sentiment":         "Without LLM/Intraday_Cross_Sectional_Mean_Reversion_Without_Sentiment",
    "with_sentiment":            "Without LLM/Intraday_Cross_Sectional_Mean_Reversion_With_Sentiment",
    # With LLM strategies
    "with_llm_with_sentiment":   "With LLM/Intraday_Cross_Sectional_Mean_Reversion_With_Sentiment",
    "with_llm_without_sentiment": "With LLM/Intraday_Cross_Sectional_Mean_Reversion_Without_Sentiment",
}

SHEET_TAB_MAP = {
    # Without LLM
    "without_sentiment":         "Without LLM Without Sentiment",
    "with_sentiment":            "Without LLM With Sentiment",
    # With LLM
    "with_llm_with_sentiment":   "With LLM With Sentiment",
    "with_llm_without_sentiment": "With LLM Without Sentiment",
}


def bootstrap_google_credentials() -> None:
    """
    On GitHub Actions / Render, the service account JSON is passed as an
    env var GOOGLE_SERVICE_ACCOUNT_JSON (full JSON string).
    Write it to google_credentials.json at repo root so gspread can find it.
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
            log.error(f"Could not write Google credentials: {e}")
            sys.exit(1)
    elif os.path.isfile(creds_path):
        log.info("google_credentials.json already present.")
    else:
        log.error(
            "GOOGLE_SERVICE_ACCOUNT_JSON env var is not set and "
            "google_credentials.json was not found. Cannot sync."
        )
        sys.exit(1)


def find_latest_report(strategy: str, market: str) -> str | None:
    """Find the most recently modified .txt report for this strategy+market."""
    folder = STRATEGY_FOLDERS[strategy]
    strategy_dir = os.path.join(ROOT, folder)

    if market == "india":
        results_dir = os.path.join(strategy_dir, "Indian_log_volatile")
    else:
        results_dir = os.path.join(strategy_dir, "live_volatile_results")

    if not os.path.isdir(results_dir):
        log.warning(f"Results directory not found: {results_dir}")
        return None

    prefix = market.upper()
    txt_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".txt") and f.upper().startswith(prefix)
    ]

    if not txt_files:
        log.warning(f"No .txt report files found in {results_dir}")
        return None

    latest = max(txt_files, key=os.path.getmtime)
    log.info(f"Latest report: {os.path.basename(latest)}")
    return latest


def sync(strategy: str, market: str) -> None:
    bootstrap_google_credentials()

    report_path = find_latest_report(strategy, market)
    if not report_path:
        log.error("No report to sync. Exiting.")
        sys.exit(1)

    sys.path.insert(0, ROOT)
    from google_sheets_sync import parse_report_txt, sync_dataframe_to_tab
    import pandas as pd

    market_label = "INDIA" if market == "india" else "US"
    row_data = parse_report_txt(report_path, market_label=market_label)
    df = pd.DataFrame([row_data])

    tab_name = SHEET_TAB_MAP[strategy]
    sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
    creds_path = os.path.join(ROOT, "google_credentials.json")

    log.info(f"Syncing to Google Sheets tab: '{tab_name}' | Market: {market_label}")
    success = sync_dataframe_to_tab(tab_name, df, sheet_id=sheet_id, creds_path=creds_path)

    if success:
        log.info(f"✓ Sync complete → '{tab_name}'")
    else:
        log.error("✗ Sync failed. Check GOOGLE_SHEET_ID and credentials.")
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync latest trading report to Google Sheets.")
    p.add_argument(
        "--strategy",
        required=True,
        choices=["without_sentiment", "with_sentiment",
                 "with_llm_with_sentiment", "with_llm_without_sentiment"],
        help="Which strategy's results to sync.",
    )
    p.add_argument(
        "--market",
        required=True,
        choices=["india", "us"],
        help="Which market to sync results for.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sync(args.strategy, args.market)
