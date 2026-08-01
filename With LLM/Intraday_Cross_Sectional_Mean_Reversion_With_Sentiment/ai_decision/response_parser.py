"""
ai_decision/response_parser.py
==============================
Strict JSON parser and validator for LLM decisions.
"""

import json
import logging
from typing import Dict, Any
from ai_decision.schemas import EntryResponse, ExitResponse

log = logging.getLogger(__name__)


class ResponseParser:
    @staticmethod
    def _clean_json_str(raw: str) -> str:
        s = raw.strip()
        if s.startswith("```json"):
            s = s[7:]
        elif s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        return s.strip()

    @classmethod
    def parse_entry_response(cls, raw_response: str) -> EntryResponse:
        cleaned = cls._clean_json_str(raw_response)
        data = json.loads(cleaned)
        return EntryResponse.model_validate(data)

    @classmethod
    def parse_exit_response(cls, raw_response: str) -> ExitResponse:
        cleaned = cls._clean_json_str(raw_response)
        data = json.loads(cleaned)
        return ExitResponse.model_validate(data)
