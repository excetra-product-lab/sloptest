from __future__ import annotations

import logging
from typing import Dict, Optional

from .base import LLMTransport

logger = logging.getLogger(__name__)


class OpenAITransport(LLMTransport):
    """OpenAI transport using official SDK. Only handles API I/O."""

    def __init__(self, api_key: str, model: str = "gpt-4.1"):
        try:
            from openai import OpenAI
        except Exception as e:  # pragma: no cover - optional import
            raise RuntimeError("openai library is not installed") from e

        self.model = model
        self._client = OpenAI(api_key=api_key, timeout=300.0)
        self._last_usage: Optional[Dict[str, int]] = None

    def generate(self, *, system_prompt: str, user_content: str, max_tokens: int,
                 temperature: float = 0.3, response_json: bool = True) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"} if response_json else None,
        )
        self._capture_usage(response)
        return response.choices[0].message.content or ""

    def refine(self, *, system_prompt: str, user_content: str, max_tokens: int,
               temperature: float = 0.2, response_json: bool = True) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"} if response_json else None,
        )
        self._capture_usage(response)
        return response.choices[0].message.content or ""

    def _capture_usage(self, response) -> None:
        try:
            usage = getattr(response, 'usage', None)
            if usage:
                self._last_usage = {
                    'input': getattr(usage, 'prompt_tokens', 0) or 0,
                    'output': getattr(usage, 'completion_tokens', 0) or 0,
                }
            else:
                self._last_usage = None
        except Exception:
            self._last_usage = None

    def get_token_usage(self) -> Optional[Dict[str, int]]:
        return self._last_usage


