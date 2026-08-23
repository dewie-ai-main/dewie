# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""E2E tests for the corpus-first `web_search` MCP tool.

Exercises the full request cycle through POST /api/mcp:
manifest advertises the tool → corpus-hit path serves from corpus without
touching the provider → gap path falls back to the stub provider and the
fetched page is persisted to the corpus (upsert + body write) via the
fire-and-forget background task.

Storage is mocked per the e2e suite convention (no live DB/Redis); the
dispatcher, gap gate, provider plumbing, and persistence wiring are real.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

_USER_ID = "00000000-0000-0000-0000-000000000042"

QUERY = "volcanic eruption monitoring iceland"

_GOOD_CORPUS = [
    (
        SimpleNamespace(
            id=uuid.uuid4(),
            title="iceland volcano guide",
            url="https://corpus.example/iceland",
            summary="all about icelandic volcanoes",
            answers_questions=["how is volcanic eruption monitoring done in iceland"],
            topics=["volcanic activity", "iceland", "eruption monitoring"],
            ingested_at=datetime(2026, 6, 1),
        ),
        0.9,
    ),
    (
        SimpleNamespace(
            id=uuid.uuid4(),
            title="cooking blog",
            url="https://corpus.example/cooking",
            summary="recipes",
            answers_questions=[],
            topics=[],
            ingested_at=datetime(2026, 5, 1),
        ),
        0.2,
    ),
    (
        SimpleNamespace(
            id=uuid.uuid4(),
            title="filler",
            url="https://corpus.example/filler",
            summary="filler",
            answers_questions=[],
            topics=[],
            ingested_at=datetime(2026, 5, 1),
        ),
        0.1,
    ),
]


@pytest.fixture
def saved_bodies(monkeypatch):
    """Intercept body_store writes so nothing touches the real filesystem."""
    saved: dict = {}
    import dewie.storage.body_store as body_store

    monkeypatch.setattr(body_store, "save_body", lambda doc_id, body: saved.update({str(doc_id): body}))
    return saved


@pytest.fixture
def stub_search(monkeypatch):
    """Route get_search_provider() to the stub with one content-bearing hit."""
    from dewie.config import settings

    monkeypatch.setattr(settings, "search_provider", "stub")
    monkeypatch.setenv(
        "DEWIE_STUB_SEARCH_RESULTS",
        json.dumps(
            [
                {
                    "title": "Fresh volcano report",
                    "url": "https://fresh.example/volcano-report",
                    "snippet": "live data",
                    "content": "Fresh volcanic monitoring data from the web. " * 40,
                }
            ]
        ),
    )


def _make_app(pg: AsyncMock) -> FastAPI:
    from dewie.api.middleware import limiter
    from dewie.api.routes.mcp import router

    app = FastAPI()
    app.state.limiter = limiter

    async def _auth(request, call_next):
        request.state.user_id = _USER_ID
        request.state.workspace_ids = []
        request.state.key_id = None
        return await call_next(request)

    app.middleware("http")(_auth)
    app.include_router(router, prefix="/api")
    app.state.postgres = pg
    app.state.processor = None
    return app


async def _call(app: FastAPI, tool_input: dict) -> dict:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/mcp", json={"tool": "web_search", "input": tool_input})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── Manifest ──────────────────────────────────────────────────────────────────


async def test_manifest_advertises_web_search():
    pg = AsyncMock()
    app = _make_app(pg)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/mcp")
    assert resp.status_code == 200
    tools = {t["name"]: t for t in resp.json()["tools"]}
    assert "web_search" in tools
    schema = tools["web_search"]["input_schema"]["properties"]
    assert {"query", "limit", "force_web", "corpus_only"} <= set(schema)


# ── Corpus-hit path ───────────────────────────────────────────────────────────


async def test_corpus_hit_serves_from_corpus_without_web(stub_search, saved_bodies):
    pg = AsyncMock()
    pg.search = AsyncMock(return_value=_GOOD_CORPUS)
    app = _make_app(pg)

    data = await _call(app, {"query": QUERY})

    content = data["content"]
    assert content["source"] == "corpus"
    assert content["corpus_hits"][0]["title"] == "iceland volcano guide"
    assert content["corpus_hits"][0]["ingested_at"] is not None
    assert "gap" not in content
    assert "answers_questions" not in json.dumps(content)
    pg.upsert.assert_not_called()
    assert saved_bodies == {}


# ── Gap → web fallback path ───────────────────────────────────────────────────


async def test_gap_falls_back_to_web_and_persists(stub_search, saved_bodies):
    pg = AsyncMock()
    pg.search = AsyncMock(return_value=[])  # empty corpus → hard gap
    app = _make_app(pg)

    data = await _call(app, {"query": QUERY})

    content = data["content"]
    assert content["source"] == "web"
    assert "gap" in content and "absent from the corpus" in content["gap"]
    assert content["content_url"] == "https://fresh.example/volcano-report"
    assert content["content"].startswith("Fresh volcanic monitoring data")
    assert content["ingested_doc_id"]

    # Fire-and-forget persistence ran as a background task within the response cycle.
    pg.upsert.assert_called_once()
    persisted = pg.upsert.call_args.args[0]
    assert persisted.url == "https://fresh.example/volcano-report"
    assert str(persisted.id) == content["ingested_doc_id"]
    assert persisted.corpus_id == f"user:{_USER_ID}"
    assert content["ingested_doc_id"] in saved_bodies


# ── Agent overrides ───────────────────────────────────────────────────────────


async def test_corpus_only_reports_miss_with_gap(stub_search, saved_bodies):
    pg = AsyncMock()
    pg.search = AsyncMock(return_value=[])
    app = _make_app(pg)

    data = await _call(app, {"query": QUERY, "corpus_only": True})

    content = data["content"]
    assert content["source"] == "miss"
    assert "gap" in content
    pg.upsert.assert_not_called()


async def test_force_web_bypasses_good_corpus(stub_search, saved_bodies):
    pg = AsyncMock()
    pg.search = AsyncMock(return_value=_GOOD_CORPUS)
    app = _make_app(pg)

    data = await _call(app, {"query": QUERY, "force_web": True})

    content = data["content"]
    assert content["source"] == "web"
    assert content["corpus_hits"], "corpus context still included"
    pg.upsert.assert_called_once()


# ── Validation ────────────────────────────────────────────────────────────────


async def test_empty_query_is_422(stub_search):
    pg = AsyncMock()
    app = _make_app(pg)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/mcp", json={"tool": "web_search", "input": {}})
    assert resp.status_code == 422


async def test_unknown_tool_lists_web_search(stub_search):
    pg = AsyncMock()
    app = _make_app(pg)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/mcp", json={"tool": "nope", "input": {}})
    assert resp.status_code == 422
    assert "Unknown tool: 'nope'" in resp.json()["detail"]
