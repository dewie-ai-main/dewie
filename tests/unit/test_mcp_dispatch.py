"""Tests for dewie.api.mcp_dispatch — transport-agnostic MCP tool dispatch.

dispatch_mcp_tool() is shared by the REST /api/mcp route (tests in
test_mcp_routes.py exercise it through that transport) and the in-process
Streamable HTTP transport (test_mcp_streamable.py). These tests call it
directly to verify all branches, independent of either transport.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from dewie.api.mcp_dispatch import dispatch_mcp_tool

WORKSPACE_ID = uuid.UUID("00000000-0000-0000-0000-000000000010")


def _enqueue_recorder():
    """Returns (enqueue_fn, calls) — calls collects the coroutines passed in."""
    calls = []

    def _enqueue(coro):
        calls.append(coro)
        coro.close()  # avoid "never awaited" warnings; we only assert it was called

    return _enqueue, calls


async def _dispatch(tool_name, input_data, pg, **overrides):
    enqueue, calls = _enqueue_recorder()
    kwargs = dict(
        pg=pg,
        user_id="user123",
        workspace_ids=[WORKSPACE_ID],
        is_admin=False,
        key_id=None,
        enqueue_background=enqueue,
    )
    kwargs.update(overrides)
    result = await dispatch_mcp_tool(tool_name, input_data, **kwargs)
    return result, calls


# ── search_corpus ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_corpus_returns_safe_results():
    from dewie.models.content import DocumentType

    pg = AsyncMock()
    doc = MagicMock()
    doc.title = "Test"
    doc.url = "https://example.com"
    doc.summary = "summary"
    doc.document_type = DocumentType.NEWS_ARTICLE
    doc.source = "web"
    pg.search = AsyncMock(return_value=[(doc, 0.9)])

    result, _ = await _dispatch("search_corpus", {"query": "test"}, pg)
    assert result["count"] == 1
    assert result["results"][0]["title"] == "Test"
    assert "answers_questions" not in result["results"][0]


@pytest.mark.asyncio
async def test_search_corpus_handles_explicit_none_source():
    """FastMCP always passes optional kwargs, even at their None default —
    unlike the REST route, which only includes keys the caller actually sent.
    Caught by the live OpenClaw verification script (search_corpus crashed
    with AttributeError: 'NoneType' object has no attribute 'strip')."""
    from dewie.models.content import DocumentType

    pg = AsyncMock()
    doc = MagicMock()
    doc.title = "Test"
    doc.url = None
    doc.summary = "summary"
    doc.document_type = DocumentType.NEWS_ARTICLE
    doc.source = "web"
    pg.search = AsyncMock(return_value=[(doc, 0.9)])

    result, _ = await _dispatch("search_corpus", {"query": "test", "corpus_id": None, "source": None}, pg)
    assert result["count"] == 1


@pytest.mark.asyncio
async def test_search_corpus_empty_query_rejected():
    pg = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _dispatch("search_corpus", {"query": "  "}, pg)
    assert exc.value.status_code == 422


# ── ingest_url / dewie_ingest ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_url_does_not_eagerly_enqueue_enrichment():
    """No enqueue_background(_enrich_one(...)) here — the doc is upserted as
    status='pending' and the throttled enrichment poller picks it up on its
    own schedule, instead of one unbounded task per ingest_url call."""
    pg = AsyncMock()
    pg.upsert = AsyncMock()
    pg.write_body_text = AsyncMock()

    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.body = "body text"

    class _FakeIngester:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def fetch(self, url):
            yield doc

    import dewie.ingestion.web as web_mod
    orig = web_mod.WebIngester
    web_mod.WebIngester = _FakeIngester
    try:
        result, calls = await _dispatch("ingest_url", {"url": "https://example.com"}, pg)
    finally:
        web_mod.WebIngester = orig

    assert result["status"] == "pending"
    assert len(calls) == 0


@pytest.mark.asyncio
async def test_ingest_url_requires_url():
    pg = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _dispatch("ingest_url", {"url": ""}, pg)
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_ingest_url_requires_user_id():
    pg = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _dispatch("ingest_url", {"url": "https://example.com"}, pg, user_id=None)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_dewie_ingest_shares_ingest_url_branch():
    pg = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _dispatch("dewie_ingest", {"url": ""}, pg)
    assert exc.value.status_code == 422


# ── expand ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_expand_requires_doc_id():
    pg = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _dispatch("expand", {"doc_id": ""}, pg)
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_expand_returns_neighbors(monkeypatch):
    pg = AsyncMock()

    async def _fake_neighbors(pg, doc_id, limit=20, *, workspace_ids=None):
        return [{"doc_id": "abc", "title": "Neighbor"}]

    monkeypatch.setattr("dewie.api.mcp_dispatch._neighbors", _fake_neighbors)

    result, _ = await _dispatch("expand", {"doc_id": "abc123"}, pg)
    assert result["count"] == 1
    assert result["neighbors"][0]["title"] == "Neighbor"


# ── read ───────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_requires_doc_id():
    pg = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _dispatch("read", {"doc_id": ""}, pg)
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_read_returns_body(monkeypatch):
    pg = AsyncMock()

    async def _fake_read_body(pg, doc_id):
        return "the full body text"

    monkeypatch.setattr("dewie.api.mcp_dispatch._read_body", _fake_read_body)

    result, _ = await _dispatch("read", {"doc_id": "abc123"}, pg)
    assert result["body"] == "the full body text"


# ── intersect ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_intersect_requires_2_plus_doc_ids():
    pg = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _dispatch("intersect", {"doc_ids": ["a"]}, pg)
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_intersect_returns_docs(monkeypatch):
    pg = AsyncMock()

    async def _fake_intersection(pg, doc_ids, min_overlap=None, limit=10, *, workspace_ids=None):
        return {"docs": [{"doc_id": "x"}], "pinned_count": 2, "min_overlap": 2}

    monkeypatch.setattr("dewie.api.mcp_dispatch._intersection", _fake_intersection)

    result, _ = await _dispatch("intersect", {"doc_ids": ["a", "b"]}, pg)
    assert result["pinned_count"] == 2


# ── bridge ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bridge_requires_source_and_target():
    pg = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _dispatch("bridge", {"source_id": "a", "target_id": ""}, pg)
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_bridge_returns_path(monkeypatch):
    pg = AsyncMock()

    async def _fake_bridge_path(pg, source_id, target_id, max_depth=5, *, workspace_ids=None):
        return {"path": [source_id, target_id], "hops": 1}

    monkeypatch.setattr("dewie.api.mcp_dispatch._bridge_path", _fake_bridge_path)

    result, _ = await _dispatch("bridge", {"source_id": "a", "target_id": "b"}, pg)
    assert result["hops"] == 1


# ── browse ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_browse_requires_query():
    pg = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _dispatch("browse", {"query": ""}, pg)
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_browse_returns_formatted_list():
    pg = AsyncMock()
    doc = MagicMock()
    doc.title = "Article"
    doc.source = "bbc.co.uk"
    doc.summary = "summary text"
    doc.published_at = None
    doc.id = "doc-1"
    pg.search = AsyncMock(return_value=[(doc, 0.5)])

    result, _ = await _dispatch("browse", {"query": "news"}, pg)
    assert result["count"] == 1
    assert "Article" in result["formatted"]


# ── web_search ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_web_search_requires_query():
    pg = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _dispatch("web_search", {"query": ""}, pg)
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_web_search_enqueues_persist_only(monkeypatch):
    """persist_document is enqueued, but not enrichment — the doc lands as
    status='pending' and the throttled enrichment poller (run_enrichment_loop)
    picks it up on its own schedule, instead of an unbounded eager task per call."""
    pg = AsyncMock()

    new_doc = MagicMock()
    new_doc.corpus_id = None

    lookup = MagicMock()
    lookup.source = "web"
    lookup.corpus_hits = []
    lookup.web_hits = [MagicMock()]
    lookup.ingested_doc_id = "doc-1"
    lookup.to_content.return_value = {"source": "web", "results": []}

    async def _fake_web_lookup(query, *, pg, provider, limit, workspace_ids, force_web, corpus_only):
        return lookup, new_doc

    monkeypatch.setattr("dewie.search.web_lookup.web_lookup", _fake_web_lookup)
    monkeypatch.setattr("dewie.search.providers.get_search_provider", lambda: None)

    result, calls = await _dispatch("web_search", {"query": "test"}, pg)

    assert result["source"] == "web"
    assert len(calls) == 1


# ── add_catalog ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_catalog_requires_admin():
    pg = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _dispatch("add_catalog", {"name": "x", "type": "mcp"}, pg, is_admin=False)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_add_catalog_rejects_invalid_type():
    pg = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _dispatch("add_catalog", {"name": "x", "type": "bogus"}, pg, is_admin=True)
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_add_catalog_creates_source():
    pg = AsyncMock()
    pg.create_source = AsyncMock(return_value={"id": str(uuid.uuid4())})

    result, _ = await _dispatch(
        "add_catalog", {"name": "remote-node", "type": "mcp", "endpoint": "http://x"}, pg, is_admin=True
    )
    assert result["ok"] is True
    assert result["name"] == "remote-node"


# ── list_sources / list_catalogs ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_sources_returns_distinct_sources():
    pg = AsyncMock()
    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [("wikipedia",), ("bbc.co.uk",)]
    mock_conn.execute = AsyncMock(return_value=mock_result)
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    pg._engine.connect = MagicMock(return_value=mock_ctx)

    result, _ = await _dispatch("list_sources", {}, pg, workspace_ids=[])
    assert result["count"] == 2
    assert "wikipedia" in result["sources"]


@pytest.mark.asyncio
async def test_list_catalogs_requires_admin_when_auth_enabled(monkeypatch):
    from dewie.config import settings

    monkeypatch.setattr(settings, "auth_enabled", True)
    pg = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _dispatch("list_catalogs", {}, pg, is_admin=False)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_list_catalogs_returns_safe_fields():
    pg = AsyncMock()
    pg.list_sources = AsyncMock(
        return_value=[{"id": "1", "name": "node-a", "type": "mcp", "enabled": True, "secret": "hide-me"}]
    )

    result, _ = await _dispatch("list_catalogs", {}, pg, is_admin=True)
    assert result["count"] == 1
    assert "secret" not in result["catalogs"][0]


# ── dewie_fetch ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dewie_fetch_requires_url():
    pg = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _dispatch("dewie_fetch", {"url": ""}, pg)
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_dewie_fetch_enqueues_save_when_requested():
    pg = AsyncMock()
    doc = MagicMock()
    doc.title = "Fetched"
    doc.body = "content body"

    class _FakeIngester:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def fetch(self, url):
            yield doc

    import dewie.ingestion.web as web_mod
    orig = web_mod.WebIngester
    web_mod.WebIngester = _FakeIngester
    try:
        result, calls = await _dispatch("dewie_fetch", {"url": "https://example.com", "save": True}, pg)
    finally:
        web_mod.WebIngester = orig

    assert result["title"] == "Fetched"
    assert len(calls) == 1  # _bg_ingest enqueued


@pytest.mark.asyncio
async def test_dewie_fetch_no_save_does_not_enqueue():
    pg = AsyncMock()
    doc = MagicMock()
    doc.title = "Fetched"
    doc.body = "content body"

    class _FakeIngester:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def fetch(self, url):
            yield doc

    import dewie.ingestion.web as web_mod
    orig = web_mod.WebIngester
    web_mod.WebIngester = _FakeIngester
    try:
        result, calls = await _dispatch("dewie_fetch", {"url": "https://example.com", "save": False}, pg)
    finally:
        web_mod.WebIngester = orig

    assert len(calls) == 0


# ── unknown tool ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_tool_raises_422():
    pg = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _dispatch("not_a_real_tool", {}, pg)
    assert exc.value.status_code == 422
