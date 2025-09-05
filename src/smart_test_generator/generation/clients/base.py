"""Abstract transport layer for LLM providers.

Each concrete transport should ONLY perform HTTP/API I/O and return raw content
strings. Prompt engineering, context assembly, JSON parsing, validation, and
cost logging are handled by higher-level orchestrators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class LLMTransport(ABC):
    """Abstract base class for provider transports."""

    @abstractmethod
    def generate(self, *, system_prompt: str, user_content: str, max_tokens: int,
                 temperature: float = 0.3, response_json: bool = True) -> str:
        """Send a generation request and return the raw response content (string)."""

    @abstractmethod
    def refine(self, *, system_prompt: str, user_content: str, max_tokens: int,
               temperature: float = 0.2, response_json: bool = True) -> str:
        """Send a refinement request and return the raw response content (string)."""

    def get_token_usage(self) -> Optional[Dict[str, int]]:
        """Return last-request token usage as a dict with 'input' and 'output' keys if available."""
        return None


