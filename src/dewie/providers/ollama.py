# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Ollama provider — local LLM inference via Ollama's REST API.

Uses the OpenAI-compatible /v1/chat/completions endpoint for chat,
and /api/embeddings for embeddings.
"""

from __future__ import annotations

import logging
import os

import httpx

from .base import ChatProvider, EmbeddingDimensionMismatchError, EmbeddingProvider

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_CHAT_MODEL = "llama3.2"
_DEFAULT_EMBED_MODEL = "nomic-embed-text"


class OllamaChatProvider(ChatProvider):
    """Chat completions via Ollama's OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = _DEFAULT_CHAT_MODEL,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self._base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)).rstrip(
            "/"
        )

    @property
    def name(self) -> str:
        return "ollama"

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int = 600,
        temperature: float = 0.3,
    ) -> str:
        url = f"{self._base_url}/v1/chat/completions"
        try:
            async with httpx.AsyncClient() as http:
                r = await http.post(
                    url,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False,
                    },
                    timeout=60,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.warning(f"OllamaChatProvider.complete failed: {e}")
            return ""


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embeddings via Ollama's native /api/embeddings endpoint."""

    def __init__(
        self,
        model: str = _DEFAULT_EMBED_MODEL,
        base_url: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self._base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)).rstrip(
            "/"
        )

    @property
    def name(self) -> str:
        return "ollama"

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        url = f"{self._base_url}/api/embeddings"
        embeddings: list[list[float]] = []
        try:
            async with httpx.AsyncClient() as http:
                for text in texts:
                    r = await http.post(
                        url,
                        json={"model": self.model, "prompt": text},
                        timeout=60,
                    )
                    r.raise_for_status()
                    embeddings.append(r.json()["embedding"])
        except Exception as e:
            log.warning(f"OllamaEmbeddingProvider.embed failed: {e}")
            return None

        if self.dimensions is not None and embeddings:
            first_dim = len(embeddings[0])
            if first_dim != self.dimensions:
                # Check if truncation is possible (actual > expected)
                if first_dim > self.dimensions:
                    log.warning(
                        "OllamaEmbeddingProvider: truncating %d-dim embedding to %d dims "
                        "(model=%r). Set EMBED_DIMENSIONS=%d to match.",
                        first_dim, self.dimensions, self.model, self.dimensions,
                    )
                    embeddings = [v[: self.dimensions] for v in embeddings]
                else:
                    raise EmbeddingDimensionMismatchError(
                        expected=self.dimensions, actual=first_dim, model=self.model
                    )

        return embeddings
