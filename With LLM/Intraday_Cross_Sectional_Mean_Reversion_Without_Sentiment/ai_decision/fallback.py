"""
ai_decision/fallback.py
=======================
Fallback handlers to ensure original strategy signals proceed seamlessly if LLM fails.
"""

import logging
from typing import List, Dict, Any
from ai_decision.schemas import EntryResponse, EntryDecisionItem, ExitResponse, ExitDecisionItem

log = logging.getLogger(__name__)


class FallbackHandler:
    @staticmethod
    def fallback_entry(picks: List[Dict[str, Any]], reason: str) -> EntryResponse:
        log.warning(f"[LLM FALLBACK ENTRY] Fallback triggered due to: {reason}. Approving all original picks unchanged.")
        decisions = []
        for pick in picks:
            ticker = pick.get("ticker", pick.get("symbol", "UNKNOWN"))
            decisions.append(
                EntryDecisionItem(
                    symbol=ticker,
                    action="BUY",
                    confidence=100,
                    position_multiplier=1.0,
                    reasoning=[f"Fallback triggered: {reason}"]
                )
            )
        return EntryResponse(decisions=decisions)

    @staticmethod
    def fallback_exit(tracking_states: Dict[str, Any], reason: str) -> ExitResponse:
        log.warning(f"[LLM FALLBACK EXIT] Fallback triggered due to: {reason}. Returning HOLD for active positions.")
        decisions = []
        for ticker, state in tracking_states.items():
            if state.get("active"):
                decisions.append(
                    ExitDecisionItem(
                        symbol=ticker,
                        action="HOLD",
                        confidence=100,
                        reasoning=[f"Fallback triggered: {reason}"]
                    )
                )
        return ExitResponse(exit_decisions=decisions)
