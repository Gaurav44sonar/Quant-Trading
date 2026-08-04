"""
ai_decision/config.py
======================
Configuration settings for the LLM Decision Layer (With Sentiment).
"""

from dataclasses import dataclass
import os
import yaml
import logging

log = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    enabled: bool = True
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    api_key: str = ""
    timeout_seconds: float = 10.0
    max_retries: int = 2
    circuit_breaker_threshold: int = 3
    circuit_breaker_reset_seconds: float = 60.0
    entry_validation: bool = True
    exit_validation: bool = True
    exit_check_interval_bars: int = 3  # Only call LLM every Nth bar (1=every bar, 3=every 15min)
    temperature: float = 0.0
    max_tokens: int = 4000
    log_file: str = "live_volatile_results/llm_decisions.jsonl"

    @classmethod
    def load(cls, config_path: str = "config/config.yaml") -> "LLMConfig":
        cfg = cls()
        
        # 1. Try reading from YAML
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if data and "llm" in data:
                        llm_data = data["llm"]
                        cfg.enabled = llm_data.get("enabled", cfg.enabled)
                        cfg.provider = llm_data.get("provider", cfg.provider)
                        cfg.model = llm_data.get("model", cfg.model)
                        cfg.timeout_seconds = float(llm_data.get("timeout_seconds", cfg.timeout_seconds))
                        cfg.max_retries = int(llm_data.get("max_retries", cfg.max_retries))
                        cfg.circuit_breaker_threshold = int(llm_data.get("circuit_breaker_threshold", cfg.circuit_breaker_threshold))
                        cfg.circuit_breaker_reset_seconds = float(llm_data.get("circuit_breaker_reset_seconds", cfg.circuit_breaker_reset_seconds))
                        cfg.entry_validation = llm_data.get("entry_validation", cfg.entry_validation)
                        cfg.exit_validation = llm_data.get("exit_validation", cfg.exit_validation)
                        cfg.exit_check_interval_bars = int(llm_data.get("exit_check_interval_bars", cfg.exit_check_interval_bars))
                        cfg.temperature = float(llm_data.get("temperature", cfg.temperature))
                        cfg.max_tokens = int(llm_data.get("max_tokens", cfg.max_tokens))
                        cfg.log_file = llm_data.get("log_file", cfg.log_file)
            except Exception as e:
                log.warning(f"Could not load LLM config from {config_path}: {e}")

        # 2. Check environment variables
        env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if env_key:
            cfg.api_key = env_key

        return cfg
