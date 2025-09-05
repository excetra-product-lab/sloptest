from __future__ import annotations

import json
import logging
import requests
from typing import Dict, Optional

from .base import LLMTransport

logger = logging.getLogger(__name__)


class ClaudeTransport(LLMTransport):
    """Anthropic Claude transport using HTTP API."""

    def __init__(self, *, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
        self._last_usage: Optional[Dict[str, int]] = None

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _capture_usage(self, result: Dict) -> None:
        usage = result.get('usage') or {}
        input_tokens = usage.get('input_tokens', 0) or 0
        output_tokens = usage.get('output_tokens', 0) or 0
        self._last_usage = {'input': input_tokens, 'output': output_tokens}

    def generate(self, *, system_prompt: str, user_content: str, max_tokens: int,
                 temperature: float = 0.3, response_json: bool = True) -> str:
        payload: Dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
        }
        if temperature is not None:
            payload["temperature"] = temperature

        response = requests.post(self.api_url, headers=self._headers(), json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        self._capture_usage(result)

        # Extract text content from Claude response
        content = ""
        for block in result.get('content', []) or []:
            if block.get('type') == 'text' and 'text' in block:
                content = block['text']
                break
        if not content and (result.get('content') and 'text' in result['content'][0]):
            content = result['content'][0]['text']
        return content or ""

    def refine(self, *, system_prompt: str, user_content: str, max_tokens: int,
               temperature: float = 0.2, response_json: bool = True) -> str:
        # Same endpoint with different defaults
        return self.generate(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=temperature,
            response_json=response_json,
        )

    def get_token_usage(self) -> Optional[Dict[str, int]]:
        return self._last_usage


