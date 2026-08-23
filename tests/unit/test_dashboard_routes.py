"""Tests for dewie.api.routes.dashboard route handlers."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_request(pg=None, cache=None):
    req = MagicMock()
    req.app.state.postgres = pg or MagicMock()
    req.app.state.cache = cache or MagicMock()
    return req


def _make_session_cm(rows=None):
    """Mock for sqlalchemy async engine connect() context manager."""
    conn = AsyncMock()
    result = MagicMock()
    result.scalar.return_value = 0
    result.fetchall.return_value = rows or []
    result.mappings.return_value.one.return_value = {}
    conn.execute = AsyncMock(return_value=result)

    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ── stats ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stats_empty():
    from dewie.api.routes.dashboard import stats

    pg = MagicMock()
    pg.count_by_status = AsyncMock(return_value={})
    pg.list_crawl_sessions = AsyncMock(return_value=[])
    req = _make_request(pg=pg)
    result = await stats(req)
    assert result["total"] == 0
    assert result["crawl_sessions"] == []


@pytest.mark.asyncio
async def test_stats_with_sessions():
    from dewie.api.routes.dashboard import stats

    pg = MagicMock()
    pg.count_by_status = AsyncMock(return_value={"ready": 10, "pending": 2})
    session = {
        "crawl_session": "abc",
        "total": 5,
        "ready": 3,
        "processing": 1,
        "failed": 1,
        "started_at": datetime(2024, 1, 1),
        "last_seen_at": None,
    }
    pg.list_crawl_sessions = AsyncMock(return_value=[session])
    req = _make_request(pg=pg)
    result = await stats(req)
    assert result["total"] == 12
    assert len(result["crawl_sessions"]) == 1
    assert result["crawl_sessions"][0]["session"] == "abc"
    assert result["crawl_sessions"][0]["last_seen_at"] is None


# ── list_documents ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_documents_empty():
    from dewie.api.routes.dashboard import list_documents

    pg = MagicMock()
    pg.list_recent = AsyncMock(return_value=[])
    req = _make_request(pg=pg)
    result = await list_documents(req)
    assert result["docs"] == []


@pytest.mark.asyncio
async def test_list_documents_with_docs():
    import uuid

    from dewie.api.routes.dashboard import list_documents

    pg = MagicMock()
    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.url = "https://example.com"
    doc.title = "Test"
    doc.source = "web"
    doc.status = MagicMock()
    doc.status.value = "ready"
    doc.topics = ["ai", "ml"]
    doc.entities = ["OpenAI"]
    doc.sentiment = 0.5
    doc.ingested_at = datetime(2024, 1, 1)
    doc.crawl_session = None
    pg.list_recent = AsyncMock(return_value=[doc])
    req = _make_request(pg=pg)
    result = await list_documents(req)
    assert len(result["docs"]) == 1
    assert result["docs"][0]["title"] == "Test"
    assert result["docs"][0]["sentiment"] == 0.5


# ── HTML endpoints ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dashboard_returns_html():
    from dewie.api.routes.dashboard import dashboard

    result = await dashboard()
    assert "Dewie" in result.body.decode()


@pytest.mark.asyncio
async def test_ingest_status_ui_returns_html():
    from dewie.api.routes.dashboard import ingest_status_ui

    result = await ingest_status_ui()
    assert "Dewie" in result.body.decode()


@pytest.mark.asyncio
async def test_mcp_config_ui_replaces_url():
    from dewie.api.routes.dashboard import mcp_config_ui

    req = MagicMock()
    req.base_url = "http://localhost:8000/"
    result = await mcp_config_ui(req)
    html = result.body.decode()
    assert "localhost:8000" in html


# ── ingest_stats ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_stats_returns_expected_keys():
    from dewie.api.routes.dashboard import ingest_stats

    # Build a mock conn that returns scalar=0 for all queries, [] for fetchall
    conn = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar.return_value = 0
    fetchall_result = MagicMock()
    fetchall_result.fetchall.return_value = []
    # Return different results depending on call count
    conn.execute = AsyncMock(
        side_effect=[
            scalar_result,  # total
            scalar_result,  # edges
            scalar_result,  # enriched_aq
            scalar_result,  # has_embedding
            scalar_result,  # has_search_vec
            scalar_result,  # last_24h
            scalar_result,  # last_1h
            scalar_result,  # last_10min
            scalar_result,  # distinct_sources
            fetchall_result,  # by_source
            fetchall_result,  # recent_docs
            scalar_result,  # newest_at
        ]
    )

    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)

    pg = MagicMock()
    pg._is_sqlite = False
    pg._engine = MagicMock()
    pg._engine.connect.return_value = cm

    req = _make_request(pg=pg)
    response = MagicMock()
    response.headers = {}

    result = await ingest_stats(req, response)

    assert "total_docs" in result
    assert "total_edges" in result
    assert "by_source" in result


# ── ingest_tool_stats ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_tool_stats_returns_keys():
    from dewie.api.routes.dashboard import ingest_tool_stats

    conn = AsyncMock()
    empty_result = MagicMock()
    empty_result.fetchall.return_value = []
    conn.execute = AsyncMock(return_value=empty_result)

    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)

    pg = MagicMock()
    pg._is_sqlite = False
    pg._engine = MagicMock()
    pg._engine.connect.return_value = cm

    req = _make_request(pg=pg)
    response = MagicMock()
    response.headers = {}

    result = await ingest_tool_stats(req, response)

    assert "tools" in result
    assert "errors" in result
    assert result["tools"] == []
    assert result["errors"] == {}


# ── issue #242 regression tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_documents_none_ingested_at():
    """list_documents must not crash when ingested_at is None (issue #242)."""
    import uuid

    from dewie.api.routes.dashboard import list_documents

    pg = MagicMock()
    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.url = "https://example.com"
    doc.title = "Test"
    doc.source = "web"
    doc.status = MagicMock()
    doc.status.value = "ready"
    doc.topics = ["ai"]
    doc.entities = []
    doc.sentiment = None
    doc.ingested_at = None  # ← triggers the 500
    doc.crawl_session = None
    pg.list_recent = AsyncMock(return_value=[doc])
    req = _make_request(pg=pg)
    result = await list_documents(req)
    assert result["docs"][0]["ingested_at"] is None


@pytest.mark.asyncio
async def test_list_documents_none_topics_entities():
    """list_documents must not crash when topics/entities are None (issue #242 guard)."""
    import uuid

    from dewie.api.routes.dashboard import list_documents

    pg = MagicMock()
    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.url = "https://example.com"
    doc.title = "Test"
    doc.source = "web"
    doc.status = MagicMock()
    doc.status.value = "pending"
    doc.topics = None
    doc.entities = None
    doc.sentiment = None
    doc.ingested_at = None
    doc.crawl_session = None
    pg.list_recent = AsyncMock(return_value=[doc])
    req = _make_request(pg=pg)
    # Should not raise TypeError: 'NoneType' object is not subscriptable
    result = await list_documents(req)
    assert result["docs"][0]["topics"] is None or isinstance(result["docs"][0]["topics"], list)
