# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Pluggable web-search providers for the corpus-first web_search tool.

Providers normalize to SearchHit. The key capability difference:
  - Brave returns links + snippets only — page content needs a separate fetch.
  - Exa and You.com can return full page text in the search response, which
    lets the caller skip the fetch step entirely (hit.content is set).

Select via SEARCH_PROVIDER env / settings: brave | exa | you | stub.
The stub provider serves canned results from DEWIE_STUB_SEARCH_RESULTS
(JSON list of hits) and exists for tests and offline development.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx

log = logging.getLogger("dewie.search")

_TIMEOUT = 15.0


@dataclass
class SearchHit:
    """One normalized web search result."""

    title: str
    url: str
    snippet: str = ""
    content: str | None = None  # full page text, when the provider returns it


@runtime_checkable
class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, limit: int = 5) -> list[SearchHit]: ...


class BraveProvider:
    """Brave Search API — links + snippets, no page content."""

    name = "brave"
    _URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, *, transport: httpx.AsyncBaseTransport | None = None):
        self._api_key = api_key
        self._transport = transport

    async def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        async with httpx.AsyncClient(timeout=_TIMEOUT, transport=self._transport) as client:
            resp = await client.get(
                self._URL,
                headers={"Accept": "application/json", "X-Subscription-Token": self._api_key},
                params={"q": query, "count": limit, "text_decorations": False},
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            SearchHit(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
            )
            for item in data.get("web", {}).get("results", [])[:limit]
        ]


class ExaProvider:
    """Exa neural search — returns full page text with the results."""

    name = "exa"
    _URL = "https://api.exa.ai/search"

    def __init__(self, api_key: str, *, transport: httpx.AsyncBaseTransport | None = None):
        self._api_key = api_key
        self._transport = transport

    async def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        async with httpx.AsyncClient(timeout=_TIMEOUT, transport=self._transport) as client:
            resp = await client.post(
                self._URL,
                headers={"x-api-key": self._api_key, "Content-Type": "application/json"},
                json={
                    "query": query,
                    "numResults": limit,
                    "contents": {"text": True},
                },
            )
            resp.raise_for_status()
            data = resp.json()
        hits = []
        for item in data.get("results", [])[:limit]:
            text = item.get("text") or None
            hits.append(
                SearchHit(
                    title=item.get("title") or item.get("url", ""),
                    url=item.get("url", ""),
                    snippet=(text or "")[:500],
                    content=text,
                )
            )
        return hits


class YouProvider:
    """You.com search API — returns long snippets per hit; we join them as content."""

    name = "you"
    _URL = "https://api.ydc-index.io/search"

    # Joined snippets shorter than this aren't "page content", just a teaser.
    _MIN_CONTENT_CHARS = 800

    def __init__(self, api_key: str, *, transport: httpx.AsyncBaseTransport | None = None):
        self._api_key = api_key
        self._transport = transport

    async def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        async with httpx.AsyncClient(timeout=_TIMEOUT, transport=self._transport) as client:
            resp = await client.get(
                self._URL,
                headers={"X-API-Key": self._api_key},
                params={"query": query, "num_web_results": limit},
            )
            resp.raise_for_status()
            data = resp.json()
        hits = []
        for item in data.get("hits", [])[:limit]:
            snippets = item.get("snippets") or []
            joined = "\n\n".join(s for s in snippets if s)
            hits.append(
                SearchHit(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=(item.get("description") or joined)[:500],
                    content=joined if len(joined) >= self._MIN_CONTENT_CHARS else None,
                )
            )
        return hits


class StubProvider:
    """Canned results for tests/offline dev. Set DEWIE_STUB_SEARCH_RESULTS to a JSON list."""

    name = "stub"

    def __init__(self, hits: list[SearchHit] | None = None):
        if hits is None:
            raw = os.environ.get("DEWIE_STUB_SEARCH_RESULTS", "[]")
            try:
                hits = [SearchHit(**h) for h in json.loads(raw)]
            except (json.JSONDecodeError, TypeError) as exc:
                log.warning("StubProvider: bad DEWIE_STUB_SEARCH_RESULTS: %s", exc)
                hits = []
        self.hits = hits
        self.calls: list[str] = []  # recorded queries, for test assertions

    async def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        self.calls.append(query)
        return self.hits[:limit]


def get_search_provider(
    provider_name: str | None = None,
) -> SearchProvider | None:
    """
    Build the configured provider, or None when web search is not configured.

    Resolution: explicit arg → settings.search_provider → SEARCH_PROVIDER env.
    Missing API key for a configured provider logs a warning and returns None
    (the web_search tool then degrades to corpus-only with a clear message).
    """
    if provider_name is None:
        try:
            from dewie.config import settings

            provider_name = settings.search_provider
        except Exception:
            provider_name = os.environ.get("SEARCH_PROVIDER", "")

    provider_name = (provider_name or "").strip().lower()
    if not provider_name or provider_name == "none":
        return None

    if provider_name == "stub":
        return StubProvider()

    key_env = {"brave": "BRAVE_API_KEY", "exa": "EXA_API_KEY", "you": "YOU_API_KEY"}
    cls = {"brave": BraveProvider, "exa": ExaProvider, "you": YouProvider}

    if provider_name not in cls:
        log.warning("Unknown SEARCH_PROVIDER %r — valid: brave|exa|you|stub", provider_name)
        return None

    api_key = os.environ.get(key_env[provider_name], "")
    if not api_key:
        log.warning(
            "SEARCH_PROVIDER=%s but %s is unset — web fallback disabled",
            provider_name,
            key_env[provider_name],
        )
        return None

    return cls[provider_name](api_key)
