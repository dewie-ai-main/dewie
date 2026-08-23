# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Anthropic Claude provider — chat via api.anthropic.com.

Uses httpx directly (no anthropic SDK) to keep dependencies minimal.
Embeddings are not supported by Anthropic; calls to embed() raise NotImplementedError.
"""

from __future__ import annotations

import logging
import os

import httpx

from .base import ChatProvider, EmbeddingProvider

log = logging.getLogger(__name__)

ANTHROPIC_CHAT_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Anthropic removed the sampling params (temperature/top_p/top_k) on its current
# top-tier models — sending `temperature` to any of these returns a 400. Older
# models (Haiku 4.5, Sonnet 4.5, Opus 4.5 and earlier) still accept it. We omit
# the param for the rejecting families so those models are usable for enrichment.
_SAMPLING_REJECTED_MARKERS = (
    "opus-4-6", "opus-4-7", "opus-4-8",
    "sonnet-5", "fable-5", "mythos-5", "mythos-preview",
)


def _rejects_sampling(model: str) -> bool:
    m = model.lower()
    return any(marker in m for marker in _SAMPLING_REJECTED_MARKERS)


class AnthropicChatProvider(ChatProvider):
    """Chat completions via Anthropic Claude API."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._chat_url = f"{base_url.rstrip('/')}/messages" if base_url else ANTHROPIC_CHAT_URL

    @property
    def name(self) -> str:
        return "anthropic"

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int = 600,
        temperature: float = 0.3,
    ) -> str:
        if not self._api_key:
            log.error("AnthropicChatProvider: ANTHROPIC_API_KEY is not set")
            return ""

        # Extract system message if present; Anthropic handles it separately
        system_content: str | None = None
        user_messages: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                user_messages.append({"role": msg["role"], "content": msg["content"]})

        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": user_messages,
        }
        # Current top-tier models reject temperature with a 400; only send it to
        # models that still accept sampling params.
        if not _rejects_sampling(self.model):
            payload["temperature"] = temperature
        if system_content:
            payload["system"] = system_content

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        try:
            async with httpx.AsyncClient() as http:
                r = await http.post(
                    self._chat_url,
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()
                # Anthropic returns content as a list of blocks
                content_blocks = data.get("content", [])
                text_parts = [
                    block["text"] for block in content_blocks if block.get("type") == "text"
                ]
                return "".join(text_parts).strip()
        except Exception as e:
            log.warning(f"AnthropicChatProvider.complete failed: {e}")
            return ""


class AnthropicEmbeddingProvider(EmbeddingProvider):
    """Anthropic does not support embeddings — always raises NotImplementedError."""

    @property
    def name(self) -> str:
        return "anthropic"

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. "
            "Use openai or ollama as your embedding provider."
        )
