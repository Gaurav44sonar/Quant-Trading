"""
ai_decision/circuit_breaker.py
================================
Circuit breaker pattern to guard trading pipeline against cascading LLM failures.
"""

import time
import logging

log = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_state_change = time.time()

    def is_allowed(self) -> bool:
        now = time.time()
        if self.state == "OPEN":
            if now - self.last_state_change > self.reset_seconds:
                self.state = "HALF_OPEN"
                self.last_state_change = now
                log.info("[CIRCUIT BREAKER] Transitioned from OPEN to HALF_OPEN. Retrying LLM call...")
                return True
            return False
        return True

    def record_success(self):
        if self.state in ("OPEN", "HALF_OPEN"):
            log.info("[CIRCUIT BREAKER] Success recorded. Resetting state to CLOSED.")
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        now = time.time()
        if self.failure_count >= self.failure_threshold and self.state != "OPEN":
            self.state = "OPEN"
            self.last_state_change = now
            log.warning(
                f"[CIRCUIT BREAKER] Threshold of {self.failure_threshold} failures reached. "
                f"Circuit OPENED. Skipping LLM calls for {self.reset_seconds} seconds."
            )
