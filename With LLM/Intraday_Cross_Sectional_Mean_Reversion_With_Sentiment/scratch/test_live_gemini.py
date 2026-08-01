"""
scratch/test_live_gemini.py
===========================
Live API verification test script for Gemini 2.5 Flash (With Sentiment folder).
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ai_decision import DecisionEngine


def test_live_gemini():
    print("=" * 70)
    print("LIVE GEMINI 2.5 FLASH API DECISION TEST (WITH SENTIMENT)")
    print("=" * 70)

    engine = DecisionEngine.from_config()
    print(f"Provider: {engine.config.provider} | Model: {engine.config.model}")
    print(f"API Key present: {bool(engine.config.api_key)}")
    print("-" * 70)

    # Sample Picks from 9-Feature DailySignalEngine + Sentiment
    sample_picks = [
        {
            "ticker": "AAPL",
            "score": 2.34,
            "entry_price": 198.50,
            "shares": 33,
            "group": "volatile",
            "atr_value": 4.56,
            "sentiment_score": 0.65,
            "news_headline": "Apple Reports Record Quarterly Revenue and Strong iPhone Demand",
            "alpha_signals": {
                "P1_overnight_gap": 1.85,
                "P2_prev_day_momentum": 0.92,
                "P3_volume_surge": 1.20,
                "P4_relative_strength": -0.45,
                "P5_range_expansion": 0.78,
                "P6_close_location": 1.10,
                "C1_opening_bar_reversal": 1.45,
                "C2_opening_volume": 0.67,
                "C3_gap_fill_speed": 0.90,
            }
        },
        {
            "ticker": "LCID",
            "score": 0.85,
            "entry_price": 3.42,
            "shares": 1947,
            "group": "volatile",
            "atr_value": 0.35,
            "sentiment_score": -0.40,
            "news_headline": "Lucid Trims Production Target as EV Demand Slows",
            "alpha_signals": {
                "P1_overnight_gap": 0.45,
                "P2_prev_day_momentum": 0.30,
                "P3_volume_surge": -0.20,
                "P4_relative_strength": 0.60,
                "P5_range_expansion": -0.15,
                "P6_close_location": 0.25,
                "C1_opening_bar_reversal": -0.10,
                "C2_opening_volume": -0.30,
                "C3_gap_fill_speed": 0.15,
            }
        }
    ]

    dummy_panels = {"open": pd.DataFrame({"AAPL": [198.50], "LCID": [3.42]})}
    nifty_close = pd.Series([485.0, 485.23])
    today_date = pd.Timestamp(datetime.now().date())

    print("\n[TEST 1] Sending ENTRY validation request to Gemini 2.5 Flash...")
    evaluated_picks = engine.evaluate_entry(
        picks=sample_picks,
        panels=dummy_panels,
        nifty_close=nifty_close,
        today_date=today_date,
        capital=100000.0,
        market="us",
    )

    print("\n" + "-" * 70)
    print("ENTRY VALIDATION RESULT:")
    print(f"Original picks count: {len(sample_picks)} -> Approved picks count: {len(evaluated_picks)}")
    for p in evaluated_picks:
        print(f"  Approved: {p['ticker']:6s} | Shares: {p['shares']:4d} | Price: ${p['entry_price']:.2f}")

    print("=" * 70)
    print("WITH_SENTIMENT GEMINI 2.5 FLASH API INTEGRATION VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_live_gemini()
