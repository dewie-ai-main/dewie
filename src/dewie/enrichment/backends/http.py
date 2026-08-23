# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
HttpBackend — generic async HTTP enrichment backend.

Supports three API shapes via the ``mode`` parameter — ``mode`` is normally
derived automatically from the registered server's ``api_format``
(see ``providers/servers.py``), not set by hand:

``ollama`` mode
    Targets the Ollama completion API (``POST /api/generate``).  Works with
    any Ollama-served model (llama3.2, phi3, qwen2.5, mistral, …) and with
    LM Studio's Ollama-compatible endpoint.

``openai`` mode
    Targets any OpenAI-compatible chat completion API
    (``POST /v1/chat/completions``).  Works with OpenAI, OpenRouter, and any
    self-hosted OpenAI-compatible server (LM Studio, llama.cpp, vLLM, …).

``anthropic`` mode
    Targets Anthropic's native Messages API (``POST /v1/messages``), with
    the ``x-api-key`` header and content-block response shape Anthropic
    actually uses.

Configuration
-------------
Backends reference a registered server by label (see ``servers:`` in
dewie.yml / ``providers/servers.py``) via ``ENRICHMENT_BACKENDS``::

    ENRICHMENT_BACKENDS='[
      {
        "name":   "llama_local",
        "type":   "http",
        "server": "llamacpp",
        "model":  "your-model-id"
      }
    ]'

Error handling
--------------
- Non-2xx HTTP responses → ``BackendError`` with status code and body excerpt.
- Request timeout        → ``BackendError`` with timeout duration.
- Missing response field → ``BackendError`` describing the missing path.
- Connection refused     → ``BackendError`` wrapping the underlying ``ConnectError``.

The ``MetadataProcessor`` catches all ``BackendError`` instances and applies
fallback / retry logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Literal

import httpx

from dewie.enrichment.base import BackendError

logger = logging.getLogger(__name__)

ApiMode = Literal["ollama", "openai", "anthropic"]


class HttpBackend:
    """
    Async HTTP enrichment backend supporting Ollama and OpenAI-compatible APIs.

    This backend is the primary integration point for LLM-based extraction.
    It is deliberately thin — it handles authentication, request shaping, and
    response extraction, but all prompt logic lives in ``enrichment/schema.py``.

    Args:
        name:          Registry identifier for this backend.  Must be unique.
        base_url:      Base URL of the API server (no trailing slash).
                       Examples: ``"http://localhost:11434"``,
                       ``"https://api.anthropic.com"``.
        model:         Model name as accepted by the target API.
                       Examples: ``"llama3.2:3b"``, ``"claude-haiku-4-5-20251001"``.
        mode:          API shape to use.  ``"ollama"`` or ``"openai"``.
        api_key_env:   Name of the environment variable holding the API key.
                       If ``None``, no ``Authorization`` header is added.
        timeout:       Request timeout in seconds.  Applies to the full
                       round-trip including model inference time.
        extra_headers:        Additional HTTP headers merged into every request.
                              Use for API-specific requirements (e.g. Anthropic's
                              ``anthropic-version`` header).
        max_tokens:           Maximum tokens in the completion response.  Passed to
                              the API where supported.  Ignored by Ollama.
        chat_template_kwargs: Passed verbatim as ``chat_template_kwargs`` in the
                              OpenAI-mode payload.  Use ``{"thinking": false}`` to
                              suppress chain-of-thought on llama.cpp thinking models
                              (Gemma 4, Qwen3, etc.).

    Raises:
        BackendError:  On any failure to obtain a valid response string.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        mode: ApiMode = "ollama",
        api_key_env: str | None = None,
        timeout: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int = 0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._mode = mode
        self._api_key_env = api_key_env
        self._timeout = timeout
        self._extra_headers = extra_headers or {}
        self._max_tokens = max_tokens
        self._extra_body = extra_body

    @property
    def name(self) -> str:
        """Registry identifier for this backend."""
        return self._name

    async def complete(self, prompt: str) -> str:
        """
        Send the extraction prompt to the configured HTTP endpoint.

        Selects the correct API shape based on ``self._mode`` and extracts
        the response text from the API-specific response structure.

        Args:
            prompt: Full extraction prompt from ``build_extraction_prompt()``.

        Returns:
            Raw response text from the model.  Should be a JSON object string.

        Raises:
            BackendError: On HTTP failure, timeout, or missing response field.
        """
        headers = self._build_headers()
        payload = self._build_payload(prompt)
        endpoint = self._endpoint()

        logger.debug(
            "HttpBackend[%s] POST %s (model=%s, timeout=%.1fs)",
            self._name,
            endpoint,
            self._model,
            self._timeout,
        )

        try:
            # Use separate connect vs read timeouts — the API can be slow to
            # finish streaming long responses; connect stays tight but we give
            # the read phase 3× the overall timeout to avoid mid-stream cuts.
            http_timeout = httpx.Timeout(
                connect=10.0,
                read=self._timeout * 3,
                write=10.0,
                pool=5.0,
            )
            async with httpx.AsyncClient(timeout=http_timeout) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise BackendError(
                f"HttpBackend[{self._name}] request timed out after {self._timeout}s"
            ) from exc
        except httpx.ConnectError as exc:
            raise BackendError(f"HttpBackend[{self._name}] connection refused: {endpoint}") from exc
        except httpx.RequestError as exc:
            raise BackendError(f"HttpBackend[{self._name}] request error: {exc}") from exc

        # 429 — rate limited: retry up to 4 times with exponential backoff
        for attempt in range(4):
            if response.status_code != 429:
                break
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    wait: float = float(retry_after)
                except ValueError:
                    wait = min(5 * (2**attempt), 120)
            else:
                wait = min(5 * (2**attempt), 120)
            logger.warning(
                "HttpBackend[%s] got 429 — waiting %.0fs (attempt %d/4)",
                self._name,
                wait,
                attempt + 1,
            )
            await asyncio.sleep(wait)
            try:
                http_timeout_429 = httpx.Timeout(
                    connect=10.0,
                    read=self._timeout * 3,
                    write=10.0,
                    pool=5.0,
                )
                async with httpx.AsyncClient(timeout=http_timeout_429) as client:
                    response = await client.post(endpoint, json=payload, headers=headers)
            except httpx.RequestError as exc:
                raise BackendError(f"HttpBackend[{self._name}] retry request error: {exc}") from exc
        else:
            snippet = response.text[:200]
            raise BackendError(f"HttpBackend[{self._name}] HTTP 429 after 4 retries: {snippet}")

        if response.status_code >= 400:
            snippet = response.text[:200]
            raise BackendError(f"HttpBackend[{self._name}] HTTP {response.status_code}: {snippet}")

        # Use streaming text parser when stream=True was requested
        logger.debug(
            "HttpBackend[%s] payload has stream=%s, response status=%d",
            self._name,
            payload.get("stream"),
            response.status_code,
        )
        if payload.get("stream"):
            logger.debug("HttpBackend[%s] using streaming parser", self._name)
            return self._extract_text_streaming(response.text)
        logger.debug("HttpBackend[%s] using non-streaming parser", self._name)
        return self._extract_text(response)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _endpoint(self) -> str:
        """Return the full URL for the API endpoint based on mode."""
        if self._mode == "ollama":
            return f"{self._base_url}/api/generate"
        suffix = "messages" if self._mode == "anthropic" else "chat/completions"
        # base_url should already include /v1 for hosted providers
        # (e.g. https://api.openai.com/v1), but also support bare base URLs for
        # self-hosted servers that expose /v1/<suffix>.
        if self._base_url.endswith("/v1"):
            return f"{self._base_url}/{suffix}"
        return f"{self._base_url}/v1/{suffix}"

    def _build_headers(self) -> dict[str, str]:
        """Build request headers, injecting the API key if configured."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        headers.update(self._extra_headers)

        if self._api_key_env:
            api_key = os.environ.get(self._api_key_env, "")
            if not api_key:
                logger.warning(
                    "HttpBackend[%s] api_key_env=%r is set but the environment "
                    "variable is empty or missing.",
                    self._name,
                    self._api_key_env,
                )
            elif self._mode == "anthropic":
                headers["x-api-key"] = api_key
                headers.setdefault("anthropic-version", "2023-06-01")
            else:
                headers["Authorization"] = f"Bearer {api_key}"

        return headers

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        """
        Build the API request payload for the selected mode.

        ``ollama`` mode: ``{ "model": ..., "prompt": ..., "stream": false }``
        ``openai`` mode: ``{ "model": ..., "messages": [...], "max_tokens": ... }``
        """
        if self._mode == "ollama":
            payload: dict = {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
            }
            if self._max_tokens:
                payload["num_predict"] = self._max_tokens
            return payload
        if self._mode == "anthropic":
            return {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self._max_tokens or 4096,
            }
        # openai mode
        payload: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._max_tokens:
            payload["max_tokens"] = self._max_tokens
        if self._extra_body:
            payload.update(self._extra_body)
        return payload

    def _extract_text(self, response: httpx.Response) -> str:
        """
        Extract the response text from the API payload.

        Raises:
            BackendError: If the expected fields are absent from the response.
        """
        try:
            data: dict[str, Any] = response.json()
        except json.JSONDecodeError as exc:
            raise BackendError(
                f"HttpBackend[{self._name}] response is not valid JSON: {exc}"
            ) from exc

        if self._mode == "ollama":
            text = data.get("response")
            if text is None:
                raise BackendError(
                    f"HttpBackend[{self._name}] 'response' field missing from Ollama reply. "
                    f"Keys present: {list(data.keys())}"
                )
            return str(text)

        if self._mode == "anthropic":
            content_blocks = data.get("content", [])
            text_parts = [b["text"] for b in content_blocks if b.get("type") == "text"]
            if not text_parts:
                raise BackendError(
                    f"HttpBackend[{self._name}] no text content in Anthropic reply. "
                    f"Keys present: {list(data.keys())}"
                )
            return "".join(text_parts)

        # openai mode
        try:
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            reasoning_content = message.get("reasoning_content") or ""

            # Thinking models (GLM-4.7-Flash, DeepSeek-R1, QwQ, etc.) sometimes
            # write their answer into reasoning_content and return empty content
            # when max_tokens is too low or the model streams only the think block.
            # Fall back to extracting JSON from reasoning_content in that case.
            if not content.strip() and reasoning_content.strip():
                logger.debug(
                    "HttpBackend[%s] content is empty but reasoning_content is non-empty "
                    "(%d chars) — attempting JSON extraction fallback",
                    self._name,
                    len(reasoning_content),
                )
                extracted = self._extract_json_from_reasoning(reasoning_content)
                if extracted:
                    logger.info(
                        "HttpBackend[%s] extracted %d-char JSON from reasoning_content",
                        self._name,
                        len(extracted),
                    )
                    return extracted
                # reasoning_content present but no JSON found — return it raw so
                # the caller can surface a useful error rather than empty string
                logger.warning(
                    "HttpBackend[%s] reasoning_content had no extractable JSON; returning raw",
                    self._name,
                )
                return reasoning_content

            return str(content)
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(
                f"HttpBackend[{self._name}] could not extract choices[0].message.content: {exc}. "
                f"Keys present: {list(data.keys())}"
            ) from exc

    @staticmethod
    def _extract_json_from_reasoning(text: str) -> str:
        """
        Extract the first top-level JSON object or array from a reasoning block.

        Thinking models embed their final answer inside ``<think>`` tags or as
        plain JSON within the reasoning text.  This helper scans for the first
        ``{`` or ``[`` and returns the balanced JSON substring, stripping any
        surrounding markdown code fences first.

        Returns an empty string when no valid JSON can be found.
        """
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
        # Remove <think> ... </think> wrappers if present — some models include
        # the answer *outside* the think block
        outside_think = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
        if outside_think:
            cleaned = outside_think

        # Find the first JSON object or array
        for start_char, end_char in (("{" , "}"), ("[", "]" )):
            idx = cleaned.find(start_char)
            if idx == -1:
                continue
            depth = 0
            in_string = False
            escape_next = False
            for i, ch in enumerate(cleaned[idx:], start=idx):
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\" and in_string:
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[idx : i + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            break
        return ""

    def _extract_text_streaming(self, raw: str) -> str:
        """
        Parse SSE streaming response (data: {...} lines) and concatenate content chunks.
        Used when stream=True is required by the target API.
        """
        parts = []
        line_count = 0
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            line_count += 1
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
                choices = obj.get("choices") or []
                if not choices:
                    continue  # skip filter/empty chunks (e.g. prompt_filter_results)
                content = choices[0].get("delta", {}).get("content")
                if content:
                    parts.append(content)
            except (json.JSONDecodeError, IndexError, KeyError) as exc:
                logger.warning(
                    "HttpBackend[%s] streaming chunk %d failed to parse: %s (chunk: %s...)",
                    self._name,
                    line_count,
                    exc,
                    chunk[:100],
                )
                continue
        result = "".join(parts)
        # Strip markdown code fences that some models wrap around JSON
        result = re.sub(r"^```(?:json)?\s*|\s*```$", "", result.strip()).strip()
        logger.debug(
            "HttpBackend[%s] streaming: %d lines, %d chunks, result len=%d",
            self._name,
            line_count,
            len(parts),
            len(result),
        )
        if not result:
            raise BackendError(f"HttpBackend[{self._name}] streaming response yielded no content")
        return result
