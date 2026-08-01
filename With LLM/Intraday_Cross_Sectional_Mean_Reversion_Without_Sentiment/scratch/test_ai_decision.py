"""
scratch/test_ai_decision.py
===========================
Test suite for LLM Decision Layer (Gemini 2.5 Flash).
"""

import sys
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


from ai_decision import DecisionEngine
from ai_decision.schemas import EntryResponse, ExitResponse
from ai_decision.fallback import FallbackHandler
from ai_decision.circuit_breaker import CircuitBreaker

def test_ai_decision_layer():
    print("=" * 60)
    print("Testing LLM Decision Layer Modules...")
    print("=" * 60)

    # 1. Circuit Breaker Test
    cb = CircuitBreaker(failure_threshold=2, reset_seconds=1.0)
    assert cb.is_allowed() == True
    cb.record_failure()
    assert cb.is_allowed() == True
    cb.record_failure()
    assert cb.is_allowed() == False
    print("[OK] Circuit Breaker test passed.")


    # 2. Fallback Test
    sample_picks = [
        {"ticker": "AAPL", "score": 2.1, "entry_price": 200.0, "shares": 50},
        {"ticker": "TSLA", "score": 1.8, "entry_price": 220.0, "shares": 45},
    ]
    fb_entry = FallbackHandler.fallback_entry(sample_picks, "Test timeout")
    assert len(fb_entry.decisions) == 2
    assert fb_entry.decisions[0].action == "BUY"
    print("[OK] Fallback Handler test passed.")

    # 3. Decision Engine Initialization Test
    engine = DecisionEngine.from_config()
    print(f"[OK] DecisionEngine initialized (Enabled: {engine.config.enabled}, Provider: {engine.config.provider}, Model: {engine.config.model}).")

    # 4. Entry Evaluation Test (with mock fallback or live call)
    dummy_panels = {"open": pd.DataFrame({"AAPL": [200.0], "TSLA": [220.0]})}
    nifty_close = pd.Series([480.0, 482.0])
    today_date = pd.Timestamp(datetime.now().date())

    filtered_picks = engine.evaluate_entry(
        picks=sample_picks,
        panels=dummy_panels,
        nifty_close=nifty_close,
        today_date=today_date,
        capital=100000.0,
        market="us",
    )
    print(f"[OK] Entry evaluation completed. Returns {len(filtered_picks)} picks.")

    for p in filtered_picks:
        print(f"   -> Pick: {p['ticker']} | Shares: {p['shares']}")

    print("=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_ai_decision_layer()
