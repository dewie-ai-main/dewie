# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""Unit tests for the pluggable web-search providers (brave/exa/you/stub)."""

from __future__ import annotations

import json

import httpx
import pytest

from dewie.search.providers import (
    BraveProvider,
    ExaProvider,
    SearchHit,
    StubProvider,
    YouProvider,
    get_search_provider,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _transport(payload: dict, capture: dict | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["url"] = str(request.url)
            capture["headers"] = dict(request.headers)
            if request.content:
                capture["body"] = json.loads(request.content)
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


# ── Brave ─────────────────────────────────────────────────────────────────────


async def test_brave_parses_results_and_has_no_content():
    payload = {
        "web": {
            "results": [
                {"title": "Doc A", "url": "https://a.example", "description": "about A"},
                {"title": "Doc B", "url": "https://b.example", "description": "about B"},
            ]
        }
    }
    capture: dict = {}
    provider = BraveProvider("test-key", transport=_transport(payload, capture))
    hits = await provider.search("query terms", limit=2)

    assert [h.url for h in hits] == ["https://a.example", "https://b.example"]
    assert hits[0].snippet == "about A"
    assert all(h.content is None for h in hits), "Brave never returns page content"
    assert capture["headers"]["x-subscription-token"] == "test-key"


async def test_brave_respects_limit():
    payload = {
        "web": {"results": [{"title": f"D{i}", "url": f"https://{i}.example"} for i in range(10)]}
    }
    provider = BraveProvider("k", transport=_transport(payload))
    hits = await provider.search("q", limit=3)
    assert len(hits) == 3


# ── Exa ───────────────────────────────────────────────────────────────────────


async def test_exa_returns_full_text_as_content():
    payload = {
        "results": [
            {"title": "Deep doc", "url": "https://deep.example", "text": "full page text " * 50},
        ]
    }
    capture: dict = {}
    provider = ExaProvider("exa-key", transport=_transport(payload, capture))
    hits = await provider.search("q", limit=5)

    assert hits[0].content is not None and hits[0].content.startswith("full page text")
    assert hits[0].snippet == hits[0].content[:500]
    assert capture["headers"]["x-api-key"] == "exa-key"
    assert capture["body"]["contents"] == {"text": True}


async def test_exa_missing_text_means_no_content():
    payload = {"results": [{"title": "T", "url": "https://t.example"}]}
    provider = ExaProvider("k", transport=_transport(payload))
    hits = await provider.search("q")
    assert hits[0].content is None


# ── You.com ───────────────────────────────────────────────────────────────────


async def test_you_joins_long_snippets_into_content():
    long_snippets = ["paragraph one " * 30, "paragraph two " * 30]
    payload = {
        "hits": [
            {"title": "You doc", "url": "https://y.example", "snippets": long_snippets},
        ]
    }
    capture: dict = {}
    provider = YouProvider("you-key", transport=_transport(payload, capture))
    hits = await provider.search("q")

    assert hits[0].content is not None
    assert "paragraph one" in hits[0].content and "paragraph two" in hits[0].content
    assert capture["headers"]["x-api-key"] == "you-key"


async def test_you_short_snippets_are_not_content():
    payload = {"hits": [{"title": "T", "url": "https://y.example", "snippets": ["tiny"]}]}
    provider = YouProvider("k", transport=_transport(payload))
    hits = await provider.search("q")
    assert hits[0].content is None
    assert hits[0].snippet == "tiny"


# ── Stub ──────────────────────────────────────────────────────────────────────


async def test_stub_serves_canned_hits_and_records_calls():
    provider = StubProvider([SearchHit(title="S", url="https://s.example", content="body")])
    hits = await provider.search("anything")
    assert hits[0].url == "https://s.example"
    assert provider.calls == ["anything"]


async def test_stub_reads_env(monkeypatch):
    monkeypatch.setenv(
        "DEWIE_STUB_SEARCH_RESULTS",
        json.dumps([{"title": "E", "url": "https://e.example", "snippet": "s", "content": "c"}]),
    )
    provider = StubProvider()
    hits = await provider.search("q")
    assert hits[0].title == "E" and hits[0].content == "c"


# ── Factory ───────────────────────────────────────────────────────────────────


def test_factory_none_when_unconfigured():
    assert get_search_provider("") is None
    assert get_search_provider("none") is None


def test_factory_none_when_key_missing(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    assert get_search_provider("brave") is None


@pytest.mark.parametrize(
    ("name", "env", "cls"),
    [("brave", "BRAVE_API_KEY", BraveProvider), ("exa", "EXA_API_KEY", ExaProvider), ("you", "YOU_API_KEY", YouProvider)],
)
def test_factory_builds_each_provider(monkeypatch, name, env, cls):
    monkeypatch.setenv(env, "k")
    provider = get_search_provider(name)
    assert isinstance(provider, cls)


def test_factory_unknown_provider_is_none():
    assert get_search_provider("altavista") is None


def test_factory_stub():
    assert isinstance(get_search_provider("stub"), StubProvider)
