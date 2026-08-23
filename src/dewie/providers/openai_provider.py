# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Direct OpenAI provider — chat and embeddings via api.openai.com.
"""

from __future__ import annotations

import logging
import os

import httpx

from .base import ChatProvider, EmbeddingProvider

log = logging.getLogger(__name__)

OPENAI_API_URL = "https://api.openai.com/v1"
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"

_DEFAULT_CHAT_MODEL = ""
_DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def _requested_embed_dimensions() -> int | None:
    raw = os.environ.get("EMBED_OUTPUT_DIMENSIONS") or os.environ.get("EMBED_DIMENSIONS")
    if not raw:
        return None
    try:
        dims = int(raw)
        return dims if dims > 0 else None
    except (TypeError, ValueError):
        return None


def _model_supports_mrl(model: str) -> bool:
    mid = model.lower()
    return (
        "qwen3-embedding" in mid
        or "text-embedding-3-small" in mid
        or "text-embedding-3-large" in mid
    )


def _downsample_vectors(vectors: list[list[float]], target_dims: int) -> list[list[float]]:
    return [vec[:target_dims] for vec in vectors]


class OpenAIChatProvider(ChatProvider):
    """Chat completions via direct OpenAI API."""

    def __init__(
        self,
        model: str = _DEFAULT_CHAT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        api_type: str = "chat/completions",
        extra_body: dict | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if base_url:
            self._chat_url = f"{base_url.rstrip('/')}/{api_type}"
        else:
            self._chat_url = f"{OPENAI_API_URL}/{api_type}"
        self._using_custom_base = bool(base_url) and self._chat_url != f"{OPENAI_API_URL}/{api_type}"
        self._extra_body = extra_body

    @property
    def name(self) -> str:
        return "openai"

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int = 600,
        temperature: float = 0.3,
    ) -> str:
        if not self._api_key and not self._using_custom_base:
            log.error("OpenAIChatProvider: OPENAI_API_KEY is not set")
            return ""

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with httpx.AsyncClient() as http:
                r = await http.post(
                    self._chat_url,
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        **(self._extra_body or {}),
                    },
                    timeout=120,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # str(e) is often empty for httpx timeouts (ConnectTimeout/ReadTimeout
            # carry no message) — include the exception type or the warning is blank.
            log.warning(f"OpenAIChatProvider.complete failed: {type(e).__name__}: {e}")
            return ""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embeddings via direct OpenAI API."""

    def __init__(
        self,
        model: str = _DEFAULT_EMBED_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        # Explicit arg wins; otherwise EMBED_OUTPUT_DIMENSIONS / EMBED_DIMENSIONS env.
        self.dimensions = dimensions if dimensions is not None else _requested_embed_dimensions()
        # Allow custom base URL for any OpenAI-compatible endpoint.
        self._embed_url = f"{base_url.rstrip('/')}/embeddings" if base_url else OPENAI_EMBED_URL
        # Extra request headers from env (e.g. provider-specific integration IDs)
        import json as _json

        _extra = os.environ.get("EMBED_EXTRA_HEADERS", "")
        self._extra_headers: dict = _json.loads(_extra) if _extra else {}
        # Set by embed() only when MRL truncation actually happened — lets
        # callers optionally persist the untruncated vector (embed_store_full_vector).
        self.last_full_vectors: list[list[float]] | None = None

    @property
    def name(self) -> str:
        return "openai"

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        using_custom_base = self._embed_url != OPENAI_EMBED_URL
        if not self._api_key and not using_custom_base:
            log.error("OpenAIEmbeddingProvider: OPENAI_API_KEY is not set")
            return None

        headers: dict = {"Content-Type": "application/json", **self._extra_headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        try:
            async with httpx.AsyncClient() as http:
                r = await http.post(
                    self._embed_url,
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                if r.status_code == 400 and "dimensions" in payload:
                    # Some OpenAI-compatible servers (llama.cpp, LM Studio) reject
                    # the dimensions param. Retry bare; downsample client-side below.
                    payload = {k: v for k, v in payload.items() if k != "dimensions"}
                    r = await http.post(
                        self._embed_url,
                        headers=headers,
                        json=payload,
                        timeout=30,
                    )
                r.raise_for_status()
                data = r.json()
                vectors = [
                    item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])
                ]
                self.last_full_vectors = None
                if (
                    self.dimensions
                    and vectors
                    and len(vectors[0]) > self.dimensions
                    and _model_supports_mrl(self.model)
                ):
                    self.last_full_vectors = vectors
                    vectors = _downsample_vectors(vectors, self.dimensions)
                return vectors
        except Exception as e:
            log.warning(f"OpenAIEmbeddingProvider.embed failed: {type(e).__name__}: {e}")
            return None
