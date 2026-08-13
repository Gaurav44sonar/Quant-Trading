"""
test_sheets_sync.py
===================
Quick test to verify Google Sheets sync works correctly.
Creates dummy trading results and pushes to all 4 strategy tabs.

Run locally:
    python test_sheets_sync.py

If successful, you'll see test rows appear in your Google Sheet tabs.
"""

import os
import sys
import json
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

# ── Check credentials ─────────────────────────────────────────────────────────
sheet_id   = os.getenv("GOOGLE_SHEET_ID", "1jvLfiSnkHkNR8eI2S80IcjhJUsR2IdqcXimvPm1X7jc")
creds_path = os.path.join(ROOT, "google_credentials.json")

print("=" * 60)
print("  Google Sheets Sync — Connection Test")
print("=" * 60)
print(f"  Sheet ID   : {sheet_id}")
print(f"  Creds file : {creds_path}")
print(f"  Creds exist: {os.path.isfile(creds_path)}")
print("=" * 60)

if not os.path.isfile(creds_path):
    print("❌ ERROR: google_credentials.json not found!")
    sys.exit(1)

if not sheet_id:
    print("❌ ERROR: GOOGLE_SHEET_ID not set!")
    sys.exit(1)

# ── Create dummy test rows ────────────────────────────────────────────────────
from google_sheets_sync import sync_dataframe_to_tab

test_rows = [
    {
        "date": "2026-08-13",
        "Market": "TEST-INDIA",
        "Market Condition": "Positive",
        "Avg Return": "0.85%",
        "Cumulative Return": "1.20%",
        "Best Trade": "2.50%",
        "Worst Trade": "-0.80%",
        "Total P&L": "$850",
        "CAGR (annualized)": "214.0%",
        "Sharpe Ratio": "1.85",
        "Sortino Ratio": "2.10",
        "Max Drawdown": "-1.20%",
        "Volatility (Ann.)": "18.5%",
        "Win Rate": "62.5%",
        "Win/Loss Ratio": "1.65",
        "Avg Win": "1.40%",
        "Avg Loss": "-0.55%",
    },
    {
        "date": "2026-08-13",
        "Market": "TEST-US",
        "Market Condition": "Neutral",
        "Avg Return": "0.72%",
        "Cumulative Return": "0.95%",
        "Best Trade": "3.10%",
        "Worst Trade": "-1.20%",
        "Total P&L": "$720",
        "CAGR (annualized)": "181.0%",
        "Sharpe Ratio": "1.62",
        "Sortino Ratio": "1.88",
        "Max Drawdown": "-1.80%",
        "Volatility (Ann.)": "21.0%",
        "Win Rate": "58.0%",
        "Win/Loss Ratio": "1.42",
        "Avg Win": "1.25%",
        "Avg Loss": "-0.70%",
    }
]

df = pd.DataFrame(test_rows)

# ── Push to all 4 strategy tabs ──────────────────────────────────────────────
tabs = [
    "Without LLM Without Sentiment",
    "Without LLM With Sentiment",
    "With LLM Without Sentiment",
    "With LLM With Sentiment",
]

all_ok = True
for tab in tabs:
    print(f"\n  Syncing test row -> tab: '{tab}'...")
    try:
        success = sync_dataframe_to_tab(tab, df.copy(), sheet_id=sheet_id, creds_path=creds_path)
        if success:
            print(f"  [OK] SUCCESS -- data pushed to '{tab}'")
        else:
            print(f"  [FAIL] FAILED -- sync returned False for '{tab}'")
            all_ok = False
    except Exception as e:
        print(f"  [ERROR] -- {e}")
        all_ok = False

print()
print("=" * 60)
if all_ok:
    print("  [OK] ALL GOOD -- Check your Google Sheet now!")
    print(f"  >> https://docs.google.com/spreadsheets/d/{sheet_id}")
else:
    print("  [FAIL] SYNC FAILED -- Check errors above")
print("=" * 60)
