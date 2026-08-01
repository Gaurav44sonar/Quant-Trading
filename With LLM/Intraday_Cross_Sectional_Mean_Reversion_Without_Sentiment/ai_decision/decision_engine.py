"""
ai_decision/decision_engine.py
==============================
Main Orchestrator for the LLM Decision Layer.
"""

import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from ai_decision.config import LLMConfig
from ai_decision.llm_client import GeminiClient
from ai_decision.circuit_breaker import CircuitBreaker
from ai_decision.prompt_builder import PromptBuilder
from ai_decision.response_parser import ResponseParser
from ai_decision.fallback import FallbackHandler
from ai_decision.logger import DecisionLogger
from ai_decision.schemas import EntryResponse, ExitResponse

log = logging.getLogger(__name__)


class DecisionEngine:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.load()
        self.client = GeminiClient(self.config)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_threshold,
            reset_seconds=self.config.circuit_breaker_reset_seconds,
        )
        self.logger = DecisionLogger(self.config.log_file)

    @classmethod
    def from_config(cls, config_path: str = "config/config.yaml") -> "DecisionEngine":
        cfg = LLMConfig.load(config_path)
        return cls(cfg)

    def evaluate_entry(
        self,
        picks: List[Dict[str, Any]],
        panels: Dict[str, Any],
        nifty_close: Any,
        today_date: Any,
        capital: float,
        market: str = "us",
    ) -> List[Dict[str, Any]]:
        """
        Evaluates stock picks before entry.
        Returns filtered/adjusted picks list.
        """
        if not picks or not self.config.enabled or not self.config.entry_validation:
            return picks

        if not self.circuit_breaker.is_allowed():
            log.info("[LLM ENTRY] Circuit breaker OPEN. Using original Alpha picks.")
            return picks

        timestamp_str = datetime.now().isoformat()
        
        # Market context
        market_info = {
            "index_ticker": "QQQ" if market == "us" else "^NSEI",
            "index_price": float(nifty_close.iloc[-1]) if nifty_close is not None and len(nifty_close) > 0 else None,
            "index_intraday_return_pct": float((nifty_close.iloc[-1] / nifty_close.iloc[0] - 1) * 100) if nifty_close is not None and len(nifty_close) > 1 else None,
            "market_tz": "US/Eastern" if market == "us" else "Asia/Kolkata",
        }
        portfolio_info = {"capital": capital, "existing_positions": 0, "current_exposure_pct": 0.0}

        system_prompt = PromptBuilder.get_system_prompt()
        user_prompt = PromptBuilder.build_entry_prompt(picks, market_info, portfolio_info, timestamp_str)

        try:
            raw_text, latency_ms = self.client.generate_json(system_prompt, user_prompt)
            response: EntryResponse = ResponseParser.parse_entry_response(raw_text)
            
            self.circuit_breaker.record_success()
            self.logger.log_call("entry_validation", latency_ms, user_prompt, raw_text, response.model_dump())
            
            # Map LLM decisions back to picks
            decision_map = {d.symbol: d for d in response.decisions}
            adjusted_picks = []

            for pick in picks:
                ticker = pick["ticker"]
                dec = decision_map.get(ticker)

                if dec is None:
                    # Default if ticker omitted by LLM
                    adjusted_picks.append(pick)
                    continue

                if dec.action.upper() == "HOLD":
                    log.info(f"  [LLM ENTRY VETO] {ticker} rejected by LLM (Reason: {dec.reasoning})")
                    continue

                new_pick = pick.copy()
                if dec.action.upper() == "REDUCE":
                    new_pick["shares"] = int(new_pick["shares"] * max(0.25, dec.position_multiplier))
                    log.info(f"  [LLM ENTRY REDUCE] {ticker} size multiplied by {dec.position_multiplier:.2f} (Reason: {dec.reasoning})")
                else:
                    if dec.position_multiplier != 1.0 and dec.position_multiplier > 0:
                        new_pick["shares"] = int(new_pick["shares"] * dec.position_multiplier)
                        log.info(f"  [LLM ENTRY APPROVE] {ticker} multiplier={dec.position_multiplier:.2f} (Reason: {dec.reasoning})")
                    else:
                        log.info(f"  [LLM ENTRY APPROVE] {ticker} approved (Confidence: {dec.confidence}%)")

                if new_pick["shares"] > 0:
                    adjusted_picks.append(new_pick)

            log.info(f"[LLM ENTRY COMPLETE] Original picks: {len(picks)} -> Approved picks: {len(adjusted_picks)}")
            return adjusted_picks

        except Exception as e:
            self.circuit_breaker.record_failure()
            reason = str(e)
            log.warning(f"[LLM ENTRY FAILURE] {reason}. Falling back to original picks.")
            fb_response = FallbackHandler.fallback_entry(picks, reason)
            self.logger.log_call("entry_validation", 0.0, user_prompt, "", fb_response.model_dump(), fallback_used=True, error_msg=reason)
            return picks

    def evaluate_exit(
        self,
        tracking_states: Dict[str, Any],
        today_close: Any,
        today_high: Any,
        today_low: Any,
        last_idx: int,
        index_data: Any,
        capital: float,
        now: datetime,
    ) -> Dict[str, Any]:
        """
        Evaluates active positions during 5-min bar checks.
        Returns dict of {ticker: ExitDecisionItem}.
        """
        active_states = {k: v for k, v in tracking_states.items() if v.get("active")}
        if not active_states or not self.config.enabled or not self.config.exit_validation:
            return {}

        if not self.circuit_breaker.is_allowed():
            log.info("[LLM EXIT] Circuit breaker OPEN. Using original risk rules.")
            return {}

        # Update current price in tracking state temporary view
        for ticker, state in active_states.items():
            if ticker in today_close.columns:
                state["current_price"] = float(today_close.iloc[last_idx][ticker])
                state["bar_high"] = float(today_high.iloc[last_idx][ticker])
                state["bar_low"] = float(today_low.iloc[last_idx][ticker])
                if state.get("entry_time"):
                    elapsed_s = (now - state["entry_time"]).total_seconds()
                    state["minutes_held"] = elapsed_s / 60.0

        timestamp_str = now.isoformat()
        market_info = {
            "index_ticker": "QQQ",
            "index_price": float(index_data.iloc[-1]) if index_data is not None and len(index_data) > 0 else None,
            "index_intraday_return_pct": None,
            "market_tz": str(now.tzinfo) if now.tzinfo else "US/Eastern",
        }
        portfolio_info = {"capital": capital, "existing_positions": len(active_states), "current_exposure_pct": 0.0}

        system_prompt = PromptBuilder.get_system_prompt()
        user_prompt = PromptBuilder.build_exit_prompt(active_states, market_info, portfolio_info, timestamp_str)

        try:
            raw_text, latency_ms = self.client.generate_json(system_prompt, user_prompt)
            response: ExitResponse = ResponseParser.parse_exit_response(raw_text)

            self.circuit_breaker.record_success()
            self.logger.log_call("exit_validation", latency_ms, user_prompt, raw_text, response.model_dump())

            dec_map = {d.symbol: d for d in response.exit_decisions}
            return dec_map

        except Exception as e:
            self.circuit_breaker.record_failure()
            reason = str(e)
            log.warning(f"[LLM EXIT FAILURE] {reason}. Falling back to standard risk rules.")
            fb_response = FallbackHandler.fallback_exit(active_states, reason)
            self.logger.log_call("exit_validation", 0.0, user_prompt, "", fb_response.model_dump(), fallback_used=True, error_msg=reason)
            return {}
