# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""AgentBackend — delegates enrichment to an OpenClaw gateway (or any OpenAI-compatible endpoint).

Instead of calling an LLM API directly, this backend POSTs the enrichment prompt
to a running OpenClaw gateway's /v1/chat/completions endpoint and returns the
raw response text. The gateway handles its own model auth — the worker only needs
a gateway bearer token.

Primary use case: local workers offload LLM calls to a remote agent backend,
which runs completions under its own model credentials. Clean separation:
The local instance owns DB + queue + orchestration; the remote backend owns LLM inference.

Config example (dewie.yml):

    backends:
      - name: remote_agent
        type: agent
        provider: custom
        endpoint: http://localhost:18789
        auth_token_env: GATEWAY_TOKEN
        model: gpt-4o
        timeout: 60

The ``provider`` field is informational — any OpenAI-compatible endpoint works
regardless of what you put there.
"""

from __future__ import annotations

import logging
import os

import httpx

from dewie.enrichment.base import BackendError, EnrichmentBackend

logger = logging.getLogger(__name__)


class AgentBackend(EnrichmentBackend):
    """Enrichment backend that delegates to an OpenClaw gateway (or any
    OpenAI-compatible /v1/chat/completions endpoint)."""

    def __init__(
        self,
        name: str,
        endpoint: str,
        model: str,
        provider: str = "custom",
        auth_token_env: str | None = None,
        auth_token: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._name = name
        self._endpoint = endpoint.rstrip("/") + "/v1/chat/completions"
        self._model = model
        self._provider = provider
        self._auth_token_env = auth_token_env
        self._auth_token = auth_token
        self._timeout = timeout
        self._extra_headers = extra_headers or {}
        # Persistent HTTP client — reuses TCP connection to gateway, avoids per-call
        # connection overhead and token-refresh churn on the gateway endpoint.
        http_timeout = httpx.Timeout(connect=10.0, read=self._timeout * 2, write=10.0, pool=5.0)
        self._http_client = httpx.AsyncClient(timeout=http_timeout)

    @property
    def name(self) -> str:
        return self._name

    def _bearer_token(self) -> str | None:
        if self._auth_token:
            return self._auth_token
        if self._auth_token_env:
            return os.environ.get(self._auth_token_env)
        return None

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._extra_headers}
        token = self._bearer_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def complete(self, prompt: str) -> str:
        """Send enrichment prompt to the agent gateway, return raw response text.

        Args:
            prompt: Full extraction prompt (same as HttpBackend receives).

        Returns:
            Raw response content string from the model.

        Raises:
            BackendError: On HTTP failure, timeout, connection error, or empty response.
        """
        headers = self._build_headers()
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000,  # raised: 2000 too low, 3000 still truncated large keyword sets
        }

        logger.debug(
            "AgentBackend[%s] POST %s (provider=%s model=%s timeout=%.1fs)",
            self._name,
            self._endpoint,
            self._provider,
            self._model,
            self._timeout,
        )

        try:
            response = await self._http_client.post(self._endpoint, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise BackendError(
                f"AgentBackend[{self._name}] request timed out after {self._timeout}s"
            ) from exc
        except httpx.ConnectError as exc:
            raise BackendError(
                f"AgentBackend[{self._name}] connection refused: {self._endpoint}"
            ) from exc
        except httpx.RequestError as exc:
            raise BackendError(f"AgentBackend[{self._name}] request error: {exc}") from exc

        if response.status_code == 401:
            raise BackendError(
                f"AgentBackend[{self._name}] auth failed (401) — check gateway token"
            )
        if response.status_code == 429:
            raise BackendError(f"AgentBackend[{self._name}] rate limited (429)")
        if response.status_code >= 400:
            raise BackendError(
                f"AgentBackend[{self._name}] HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise BackendError(
                f"AgentBackend[{self._name}] unexpected response shape: {response.text[:200]}"
            ) from exc

        if not content or not content.strip():
            raise BackendError(f"AgentBackend[{self._name}] empty response from gateway")

        return content
