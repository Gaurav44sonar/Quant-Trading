"""
ai_decision package
===================
LLM Decision Layer for Intraday Cross-Sectional Mean Reversion Strategy.

Provides entry and exit signal validation using Google Gemini 2.5 Flash.
"""

from ai_decision.decision_engine import DecisionEngine

__all__ = ["DecisionEngine"]
