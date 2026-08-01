"""
ai_decision/logger.py
=====================
Structured JSONL logging for all LLM calls, latency, tokens, and decisions.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any

log = logging.getLogger(__name__)


class DecisionLogger:
    def __init__(self, log_file_path: str = "live_volatile_results/llm_decisions.jsonl"):
        self.log_file_path = log_file_path
        os.makedirs(os.path.dirname(os.path.abspath(log_file_path)), exist_ok=True)

    def log_call(
        self,
        task: str,
        latency_ms: float,
        prompt: str,
        response_raw: str,
        parsed_response: Dict[str, Any],
        fallback_used: bool = False,
        error_msg: str = "",
    ):
        record = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "latency_ms": round(latency_ms, 2),
            "fallback_used": fallback_used,
            "error_msg": error_msg,
            "prompt_snippet": prompt[:300] if prompt else "",
            "response_snippet": response_raw[:300] if response_raw else "",
            "parsed_response": parsed_response,
        }
        
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning(f"Could not append to LLM log file {self.log_file_path}: {e}")
