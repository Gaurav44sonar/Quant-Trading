"""
ai_decision/response_parser.py
==============================
Robust JSON parser and validator for LLM decisions.
Handles common Gemini response quirks (markdown fences, trailing commas,
unescaped characters, thinking blocks, etc.)
"""

import json
import re
import logging
from typing import Dict, Any
from ai_decision.schemas import EntryResponse, ExitResponse

log = logging.getLogger(__name__)


class ResponseParser:
    @staticmethod
    def _clean_json_str(raw: str) -> str:
        """Remove markdown fences and extract the JSON object."""
        s = raw.strip()
        # Strip markdown code fences
        if s.startswith("```json"):
            s = s[7:]
        elif s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
        return s

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """Extract the first complete JSON object {...} from text."""
        # Find the first { and last matching }
        start = text.find("{")
        if start == -1:
            return text
        
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        
        return text[start:end + 1]

    @staticmethod
    def _fix_common_json_issues(text: str) -> str:
        """Fix common JSON issues from LLM outputs."""
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([\]}])', r'\1', text)
        # Remove any control characters except newline/tab
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        return text

    @classmethod
    def _robust_parse(cls, raw_response: str) -> dict:
        """
        Multi-strategy JSON parsing:
        1. Direct json.loads on cleaned text
        2. Extract JSON object and retry
        3. Fix common issues and retry
        """
        cleaned = cls._clean_json_str(raw_response)
        
        # Strategy 1: Direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON object from text
        try:
            extracted = cls._extract_json_object(cleaned)
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Fix common issues then extract + parse
        try:
            fixed = cls._fix_common_json_issues(cleaned)
            extracted = cls._extract_json_object(fixed)
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
        
        # All strategies failed — log raw response for debugging and re-raise
        log.error("[RESPONSE PARSER] All JSON parse strategies failed. Raw response (first 500 chars): %s", 
                  raw_response[:500])
        # Final attempt — let it raise with the original error
        return json.loads(cleaned)

    @classmethod
    def parse_entry_response(cls, raw_response: str) -> EntryResponse:
        data = cls._robust_parse(raw_response)
        return EntryResponse.model_validate(data)

    @classmethod
    def parse_exit_response(cls, raw_response: str) -> ExitResponse:
        data = cls._robust_parse(raw_response)
        return ExitResponse.model_validate(data)
