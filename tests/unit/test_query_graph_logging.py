"""Tests for logging instrumentation in query and graph endpoints (issue #787)."""

from __future__ import annotations

import logging
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_graph_request(pg=None):
    req = MagicMock()
    req.app.state.postgres = pg or MagicMock()
    req.state.request_id = "test-rid-123"
    req.state.tenant_id = "tenant-1"
    req.state.user_id = "user-1"
    req.state.workspace_ids = []
    return req


def _make_pg_session(session_rows=None):
    pg = MagicMock()
    session = AsyncMock()
    session_rows = session_rows or []

    mappings_result = MagicMock()
    mappings_result.all.return_value = session_rows
    execute_result = MagicMock()
    execute_result.mappings.return_value = mappings_result
    session.execute = AsyncMock(return_value=execute_result)

    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    pg._session_factory = MagicMock(return_value=cm)
    pg._apply_tenant = AsyncMock()
    return pg


# ── Graph endpoint logging ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_neighbors_logs_start(caplog):
    """get_neighbors logs a start message with request_id."""
    from dewie.api.routes.graph import get_neighbors

    pg = _make_pg_session([])
    req = _make_graph_request(pg)
    doc_id = uuid.uuid4()

    with caplog.at_level(logging.INFO, logger="dewie.api"):
        await get_neighbors(doc_id, req, limit=10)

    messages = [r.message for r in caplog.records if r.name == "dewie.api"]
    assert any("graph_neighbors started" in m for m in messages)


@pytest.mark.asyncio
async def test_graph_neighbors_logs_request_id(caplog):
    """get_neighbors includes request_id in log extra."""
    from dewie.api.routes.graph import get_neighbors

    pg = _make_pg_session([])
    req = _make_graph_request(pg)
    req.state.request_id = "req-abc"
    doc_id = uuid.uuid4()

    with caplog.at_level(logging.INFO, logger="dewie.api"):
        await get_neighbors(doc_id, req, limit=5)

    records = [r for r in caplog.records if r.name == "dewie.api"]
    start_record = next((r for r in records if "graph_neighbors started" in r.message), None)
    assert start_record is not None
    assert start_record.__dict__.get("request_id") == "req-abc"


@pytest.mark.asyncio
async def test_graph_neighbors_logs_succeeded(caplog):
    """get_neighbors logs success after returning results."""
    from dewie.api.routes.graph import get_neighbors

    pg = _make_pg_session([])
    req = _make_graph_request(pg)
    doc_id = uuid.uuid4()

    with caplog.at_level(logging.INFO, logger="dewie.api"):
        await get_neighbors(doc_id, req)

    messages = [r.message for r in caplog.records if r.name == "dewie.api"]
    assert any("graph_neighbors succeeded" in m for m in messages)


@pytest.mark.asyncio
async def test_graph_intersection_logs_start(caplog):
    """intersection logs a start message with request_id."""
    from dewie.api.routes.graph import intersection

    pg = _make_pg_session([])
    req = _make_graph_request(pg)

    with caplog.at_level(logging.INFO, logger="dewie.api"):
        await intersection(req, {"doc_ids": ["d1", "d2"]})

    messages = [r.message for r in caplog.records if r.name == "dewie.api"]
    assert any("graph_intersection started" in m for m in messages)


@pytest.mark.asyncio
async def test_graph_intersection_logs_succeeded(caplog):
    """intersection logs success."""
    from dewie.api.routes.graph import intersection

    pg = _make_pg_session([])
    req = _make_graph_request(pg)

    with caplog.at_level(logging.INFO, logger="dewie.api"):
        await intersection(req, {"doc_ids": ["d1", "d2"]})

    messages = [r.message for r in caplog.records if r.name == "dewie.api"]
    assert any("graph_intersection succeeded" in m for m in messages)


@pytest.mark.asyncio
async def test_graph_intersection_short_circuit_no_success_log(caplog):
    """intersection with <2 doc_ids returns early — no succeeded log."""
    from dewie.api.routes.graph import intersection

    req = _make_graph_request()

    with caplog.at_level(logging.INFO, logger="dewie.api"):
        await intersection(req, {"doc_ids": ["only-one"]})

    messages = [r.message for r in caplog.records if r.name == "dewie.api"]
    # started is logged, succeeded is NOT because we returned early
    assert any("graph_intersection started" in m for m in messages)
    assert not any("graph_intersection succeeded" in m for m in messages)


@pytest.mark.asyncio
async def test_graph_bridge_logs_start(caplog):
    """bridge_path logs start with source/target."""
    from dewie.api.routes.graph import bridge_path

    pg = _make_pg_session([])
    req = _make_graph_request(pg)

    with caplog.at_level(logging.INFO, logger="dewie.api"):
        await bridge_path(req, {"source_id": "d1", "target_id": "d2", "max_depth": 2})

    messages = [r.message for r in caplog.records if r.name == "dewie.api"]
    assert any("graph_bridge started" in m for m in messages)


@pytest.mark.asyncio
async def test_graph_bridge_logs_request_id(caplog):
    """bridge_path includes request_id in log records."""
    from dewie.api.routes.graph import bridge_path

    pg = _make_pg_session([])
    req = _make_graph_request(pg)
    req.state.request_id = "bridge-rid"

    with caplog.at_level(logging.INFO, logger="dewie.api"):
        await bridge_path(req, {"source_id": "d1", "target_id": "d2", "max_depth": 2})

    records = [r for r in caplog.records if r.name == "dewie.api"]
    start_rec = next((r for r in records if "graph_bridge started" in r.message), None)
    assert start_rec is not None
    assert start_rec.__dict__.get("request_id") == "bridge-rid"


# ── Query endpoint logging ────────────────────────────────────────────────────


def _make_query_app():
    """Build a minimal FastAPI app with the query router for TestClient use."""
    from fastapi import FastAPI

    from dewie.api.middleware import limiter
    from dewie.api.routes.query import router

    app = FastAPI()
    app.state.limiter = limiter

    from unittest.mock import AsyncMock, MagicMock

    from dewie.models.content import ContentDocument, ContentStatus

    doc = ContentDocument(
        url="https://example.com/doc",
        title="Test Doc",
        status=ContentStatus.READY,
        summary="Summary",
        keywords=["k1"],
        entities=[],
        topics=["topic"],
        answers_questions=["Q?"],
        embed_summary="embed",
    )

    pg = AsyncMock()
    pg.search = AsyncMock(return_value=[(doc, 0.9)])
    pg.get_edge_count = AsyncMock(return_value=2)
    pg.enqueue_search = AsyncMock(return_value=(False, None))
    pg.search_chunks_for_docs = AsyncMock(return_value={})
    pg.get_source = AsyncMock(return_value=None)

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    pg._session_factory = MagicMock(return_value=session_cm)

    conn = AsyncMock()
    qid_result = MagicMock()
    qid_result.fetchone = MagicMock(return_value=None)
    conn.execute = AsyncMock(return_value=qid_result)
    conn_cm = MagicMock()
    conn_cm.__aenter__ = AsyncMock(return_value=conn)
    conn_cm.__aexit__ = AsyncMock(return_value=None)
    pg._engine = MagicMock()
    pg._engine.connect = MagicMock(return_value=conn_cm)

    cache = AsyncMock()
    cache.get_query_result = AsyncMock(return_value=None)
    cache.set_query_result = AsyncMock()
    cache._redis = AsyncMock()
    cache._redis.get = AsyncMock(return_value=None)

    app.state.postgres = pg
    app.state.cache = cache


    async def add_state(request, call_next):
        request.state.request_id = "test-query-rid"
        request.state.workspace_ids = []
        request.state.user_id = None
        return await call_next(request)

    app.middleware("http")(add_state)
    app.include_router(router)
    return app


@pytest.mark.asyncio
async def test_query_endpoint_logs_started(caplog):
    """POST /query logs 'query started' with request_id."""
    from fastapi.testclient import TestClient

    from dewie.api.middleware import limiter

    app = _make_query_app()
    limiter.enabled = False

    with caplog.at_level(logging.INFO, logger="dewie.api"):
        with TestClient(app) as client:
            resp = client.post("/query", json={"query": "test query", "limit": 5})

    assert resp.status_code == 200
    messages = [r.message for r in caplog.records if r.name == "dewie.api"]
    assert any("query started" in m for m in messages)


@pytest.mark.asyncio
async def test_query_endpoint_logs_succeeded(caplog):
    """POST /query logs 'query succeeded' after returning results."""
    from fastapi.testclient import TestClient

    from dewie.api.middleware import limiter

    app = _make_query_app()
    limiter.enabled = False

    with caplog.at_level(logging.INFO, logger="dewie.api"):
        with TestClient(app) as client:
            resp = client.post("/query", json={"query": "another query", "limit": 3})

    assert resp.status_code == 200
    messages = [r.message for r in caplog.records if r.name == "dewie.api"]
    assert any("query succeeded" in m for m in messages)


def test_query_logs_truncate_long_query(caplog):
    """Query text is truncated to <=1000 chars in log output.

    SearchRequest enforces max_length=500, so we use 500 chars (the model max).
    The log layer slices at [:1000] as a defensive guard; 500 < 1000, so logged
    query should always be <= 1000 chars.
    """
    from fastapi.testclient import TestClient

    from dewie.api.middleware import limiter

    app = _make_query_app()
    limiter.enabled = False

    # SearchRequest.query has max_length=500; use exactly 500 chars
    long_query = "x" * 500

    with caplog.at_level(logging.INFO, logger="dewie.api"):
        with TestClient(app) as client:
            resp = client.post("/query", json={"query": long_query, "limit": 5})

    assert resp.status_code == 200
    # The logged query should be at most 1000 chars
    records = [r for r in caplog.records if r.name == "dewie.api" and "query started" in r.message]
    assert records, "No 'query started' log record found"
    logged_query = records[0].__dict__.get("query", "")
    assert len(logged_query) <= 1000
