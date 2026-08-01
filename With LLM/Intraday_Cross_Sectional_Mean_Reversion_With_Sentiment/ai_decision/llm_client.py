"""
ai_decision/llm_client.py
=========================
LLM API Client for Google Gemini 2.5 Flash.
Supports google-generativeai SDK and direct HTTP REST API fallback.
"""

import time
import json
import logging
import urllib.request
import urllib.error
from typing import Optional, Tuple
from ai_decision.config import LLMConfig

log = logging.getLogger(__name__)


class GeminiClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key
        self.model = config.model or "gemini-2.5-flash"
        self.timeout = config.timeout_seconds
        
        # Try initializing google-generativeai SDK if installed
        self.genai_sdk = None
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                
                gen_cfg = {"temperature": self.config.temperature, "max_output_tokens": self.config.max_tokens}
                try:
                    from google.generativeai.types import GenerationConfig
                    config_obj = GenerationConfig(temperature=self.config.temperature, response_mime_type="application/json")
                    self.genai_sdk = genai.GenerativeModel(model_name=self.model, generation_config=config_obj)
                except Exception:
                    self.genai_sdk = genai.GenerativeModel(model_name=self.model, generation_config=gen_cfg)
                    
                log.info(f"[LLM CLIENT] Initialized google-generativeai SDK with model {self.model}")
            except Exception as e:
                log.info(f"[LLM CLIENT] google-generativeai SDK not used ({e}). Falling back to Direct REST API.")

    def generate_json(self, system_prompt: str, user_prompt: str) -> Tuple[str, float]:
        """
        Sends system + user prompt to Gemini and returns (raw_json_str, latency_ms).
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        start_t = time.time()

        # Method 1: SDK if available
        if self.genai_sdk is not None:
            try:
                prompt_content = f"{system_prompt}\n\n{user_prompt}"
                response = self.genai_sdk.generate_content(prompt_content)
                latency_ms = (time.time() - start_t) * 1000.0
                return response.text, latency_ms
            except Exception as e:
                log.warning(f"[LLM CLIENT] SDK call failed ({e}). Attempting REST API fallback...")

        # Method 2: Direct REST API Call
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]
                }
            ],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
                "responseMimeType": "application/json"
            }
        }

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_bytes = resp.read()
                res_json = json.loads(resp_bytes.decode("utf-8"))
                
                # Extract text
                candidates = res_json.get("candidates", [])
                if not candidates:
                    raise ValueError("No candidates returned from Gemini REST API.")
                
                text = candidates[0]["content"]["parts"][0]["text"]
                latency_ms = (time.time() - start_t) * 1000.0
                return text, latency_ms

        except urllib.error.HTTPError as http_err:
            error_body = http_err.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Gemini API HTTP Error {http_err.code}: {error_body}")
        except Exception as ex:
            raise RuntimeError(f"Gemini REST API Call failed: {ex}")
