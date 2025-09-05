from __future__ import annotations

import json
import logging
import requests
from typing import Dict, Optional

from .base import LLMTransport

logger = logging.getLogger(__name__)


class AzureOpenAITransport(LLMTransport):
    """Azure OpenAI transport using REST API. Only performs HTTP I/O."""

    def __init__(self, *, endpoint: str, api_key: str, deployment_name: str, api_version: str = "2024-10-21"):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = api_version
        self._last_usage: Optional[Dict[str, int]] = None

    def _post(self, payload: Dict) -> Dict:
        url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
        headers = {"Content-Type": "application/json", "api-key": self.api_key}
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()

    def generate(self, *, system_prompt: str, user_content: str, max_tokens: int,
                 temperature: float = 0.3, response_json: bool = True) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}

        result = self._post(payload)
        # Azure response may not always contain usage; preserve if present
        usage = result.get("usage") or {}
        self._last_usage = {
            'input': usage.get('prompt_tokens', 0) or 0,
            'output': usage.get('completion_tokens', 0) or 0,
        }
        return (result.get('choices', [{}])[0].get('message', {}) or {}).get('content', "")

    def refine(self, *, system_prompt: str, user_content: str, max_tokens: int,
               temperature: float = 0.2, response_json: bool = True) -> str:
        # Identical to generate but different default temperature
        return self.generate(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=temperature,
            response_json=response_json,
        )

    def get_token_usage(self) -> Optional[Dict[str, int]]:
        return self._last_usage


