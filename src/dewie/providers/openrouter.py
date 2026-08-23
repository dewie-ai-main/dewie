# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
OpenRouter provider — OpenAI-compatible chat via openrouter.ai.

Embeddings are not supported via OpenRouter; use openai or ollama instead.
"""

from __future__ import annotations

import logging
import os

import httpx

from .base import ChatProvider, EmbeddingProvider

log = logging.getLogger(__name__)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

_DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"


class OpenRouterChatProvider(ChatProvider):
    """Chat completions via OpenRouter (OpenAI-compatible)."""

    def __init__(self, model: str = _DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

    @property
    def name(self) -> str:
        return "openrouter"

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int = 600,
        temperature: float = 0.3,
    ) -> str:
        if not self._api_key:
            log.error("OpenRouterChatProvider: OPENROUTER_API_KEY is not set")
            return ""

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://dewie.ai",
            "X-Title": "Dewie",
        }

        try:
            async with httpx.AsyncClient() as http:
                r = await http.post(
                    OPENROUTER_CHAT_URL,
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=30,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.warning(f"OpenRouterChatProvider.complete failed: {e}")
            return ""


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """OpenRouter does not provide a dedicated embeddings endpoint."""

    @property
    def name(self) -> str:
        return "openrouter"

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        raise NotImplementedError(
            "OpenRouter does not provide an embeddings API. "
            "Use openai or ollama as your embedding provider."
        )
