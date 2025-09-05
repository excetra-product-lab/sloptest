from __future__ import annotations

import logging
from typing import Dict, Optional

from .base import LLMTransport

logger = logging.getLogger(__name__)


class BedrockTransport(LLMTransport):
    """AWS Bedrock transport via langchain_aws ChatBedrock. API I/O only."""

    def __init__(self, *, chat_bedrock):
        # Accept an initialized ChatBedrock instance from the caller
        self._chat = chat_bedrock
        self._last_usage: Optional[Dict[str, int]] = None

    def generate(self, *, system_prompt: str, user_content: str, max_tokens: int,
                 temperature: float = 0.3, response_json: bool = True) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        result = self._chat.invoke(messages, max_tokens=max_tokens, temperature=temperature)
        raw_content = getattr(result, "content", "") or ""
        return _extract_text_from_bedrock(raw_content)

    def refine(self, *, system_prompt: str, user_content: str, max_tokens: int,
               temperature: float = 0.2, response_json: bool = True) -> str:
        return self.generate(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=temperature,
            response_json=response_json,
        )

    def get_token_usage(self) -> Optional[Dict[str, int]]:
        # LangChain result does not expose usage reliably; return None
        return None


def _extract_text_from_bedrock(raw_content) -> str:
    if isinstance(raw_content, list):
        for part in raw_content:
            if isinstance(part, dict):
                if part.get('type') == 'text' and 'text' in part:
                    return part['text']
                if 'text' in part:
                    return part['text']
            else:
                try:
                    maybe_text = getattr(part, "text", None)
                    if maybe_text:
                        return maybe_text
                except Exception:
                    continue
        # Join all parts if nothing matched
        text_parts = []
        for part in raw_content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            else:
                try:
                    maybe_text = getattr(part, "text", None)
                    if maybe_text:
                        text_parts.append(maybe_text)
                except Exception:
                    continue
        return "\n".join(text_parts)
    return str(raw_content or "")


