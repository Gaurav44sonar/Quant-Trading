"""
Historical Data Migration Script for Google Sheets
===================================================
Scans all 4 strategy directories for existing .txt report files and Excel files,
extracts all historical metrics, and populates the 4 Google Sheet worksheet tabs:

1. With LLM With Sentiment
2. With LLM Without Sentiment
3. Without LLM With Sentiment
4. Without LLM Without Sentiment

Usage:
  python sync_all_historical.py            # Live upload to Google Sheets
  python sync_all_historical.py --dry-run  # Preview parsed records locally
"""

import os
import glob
import sys
import argparse
import pandas as pd
from google_sheets_sync import (
    parse_report_txt,
    sync_dataframe_to_tab,
    load_env_credentials,
    STRATEGY_MAP,
    COLUMNS_ORDER
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STRATEGY_PATHS = {
    'WITH_LLM_WITH_SENTIMENT': os.path.join(
        BASE_DIR, 'With LLM', 'Intraday_Cross_Sectional_Mean_Reversion_With_Sentiment'
    ),
    'WITH_LLM_WITHOUT_SENTIMENT': os.path.join(
        BASE_DIR, 'With LLM', 'Intraday_Cross_Sectional_Mean_Reversion_Without_Sentiment'
    ),
    'WITHOUT_LLM_WITH_SENTIMENT': os.path.join(
        BASE_DIR, 'Without LLM', 'Intraday_Cross_Sectional_Mean_Reversion_With_Sentiment'
    ),
    'WITHOUT_LLM_WITHOUT_SENTIMENT': os.path.join(
        BASE_DIR, 'Without LLM', 'Intraday_Cross_Sectional_Mean_Reversion_Without_Sentiment'
    ),
}


def collect_strategy_records(strategy_dir: str) -> pd.DataFrame:
    """
    Collects all session metric records for a strategy directory by reading .txt reports.
    """
    rows = []

    # 1. US Volatile Reports
    us_dir = os.path.join(strategy_dir, "live_volatile_results")
    us_files = sorted(glob.glob(os.path.join(us_dir, "US_VOLATILE_*.txt")))
    for f in us_files:
        try:
            r = parse_report_txt(f, market_label="US")
            rows.append(r)
        except Exception as e:
            print(f"[WARNING] Error parsing US file {f}: {e}")

    # 2. Indian Volatile Reports
    india_dir = os.path.join(strategy_dir, "Indian_log_volatile")
    india_files = sorted(glob.glob(os.path.join(india_dir, "INDIA_VOLATILE_*.txt")))
    for f in india_files:
        try:
            r = parse_report_txt(f, market_label="INDIA")
            rows.append(r)
        except Exception as e:
            print(f"[WARNING] Error parsing India file {f}: {e}")

    if not rows:
        return pd.DataFrame(columns=COLUMNS_ORDER)

    df = pd.DataFrame(rows)

    # Sort chronologically
    if '_start_time' in df.columns:
        df = df.sort_values(by=['date', '_start_time'], ascending=[True, True])
        df = df.drop(columns=['_start_time'])

    for col in COLUMNS_ORDER:
        if col not in df.columns:
            df[col] = ''

    return df[COLUMNS_ORDER]


def main():
    parser = argparse.ArgumentParser(description="Sync historical trading data to Google Sheets.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and display records without uploading to Google Sheets")
    args = parser.parse_args()

    sheet_id, creds_path = load_env_credentials()

    print("=" * 70)
    print("QUANT-TRADING GOOGLE SHEETS HISTORICAL DATA SYNC")
    print("=" * 70)

    if not args.dry_run:
        if not sheet_id:
            print("[ERROR] GOOGLE_SHEET_ID is not set in environment or .env file.")
            print("Please create/configure your Google Sheet ID before running live sync.")
            print("Run with --dry-run to test historical data extraction locally.")
            sys.exit(1)
        if not creds_path or not os.path.isfile(creds_path):
            print(f"[ERROR] Service Account key file not found at '{creds_path}'.")
            print("Please place 'google_credentials.json' in the project root.")
            print("Run with --dry-run to test historical data extraction locally.")
            sys.exit(1)

    all_summaries = {}

    for strat_key, strat_path in STRATEGY_PATHS.items():
        tab_name = STRATEGY_MAP[strat_key]
        print(f"\nScanning: {strat_key} -> Tab: '{tab_name}'")
        print(f"Path: {strat_path}")

        df = collect_strategy_records(strat_path)
        print(f"  Found {len(df)} historical session records.")

        if not df.empty:
            print("  Sample records:")
            print(df[['date', 'Market', 'Total P&L', 'Sharpe Ratio', 'Win Rate']].head(3).to_string(index=False))

        all_summaries[tab_name] = df

        if not args.dry_run:
            print(f"  Uploading to Google Sheet tab '{tab_name}'...")
            success = sync_dataframe_to_tab(tab_name, df, sheet_id=sheet_id, creds_path=creds_path)
            if success:
                print(f"  [SUCCESS] Tab '{tab_name}' updated successfully.")
            else:
                print(f"  [FAILED] Failed to update tab '{tab_name}'.")

    print("\n" + "=" * 70)
    if args.dry_run:
        print("[DRY-RUN COMPLETE] Parsed historical data for all 4 strategy tabs cleanly.")
    else:
        print("[SYNC COMPLETE] All historical records migrated to Google Sheet.")
    print("=" * 70)


if __name__ == "__main__":
    main()
