"""Tests for dewie.api.routes.graph route handlers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_request(pg=None):
    """Build a minimal FastAPI-like Request mock."""
    req = MagicMock()
    req.app.state.postgres = pg or MagicMock()
    req.state.tenant_id = "tenant-1"
    req.state.user_id = "user-1"
    return req


def _make_pg(session_rows=None):
    """Build a mock PostgresClient with a session factory."""
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


# ── intersection ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_intersection_requires_two_doc_ids():
    from dewie.api.routes.graph import intersection

    req = _make_request()
    result = await intersection(req, {"doc_ids": ["only-one"]})
    assert "error" in result


@pytest.mark.asyncio
async def test_intersection_empty_returns_error():
    from dewie.api.routes.graph import intersection

    req = _make_request()
    result = await intersection(req, {"doc_ids": []})
    assert "error" in result


@pytest.mark.asyncio
async def test_intersection_with_results():
    from dewie.api.routes.graph import intersection

    row = {
        "doc_id": "d3",
        "title": "Common",
        "summary": "S",
        "keywords": [],
        "entities": [],
        "overlap_count": 2,
        "avg_weight": 0.8,
        "from_docs": ["d1", "d2"],
    }
    pg = _make_pg([row])
    req = _make_request(pg)
    result = await intersection(req, {"doc_ids": ["d1", "d2"], "limit": 10})
    assert "docs" in result
    assert result["pinned_count"] == 2


@pytest.mark.asyncio
async def test_intersection_no_results():
    from dewie.api.routes.graph import intersection

    pg = _make_pg([])
    req = _make_request(pg)
    result = await intersection(req, {"doc_ids": ["d1", "d2"]})
    assert result["docs"] == []


# ── bridge_path ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bridge_path_missing_source():
    from dewie.api.routes.graph import bridge_path

    req = _make_request()
    result = await bridge_path(req, {"target_id": "d2"})
    assert "error" in result


@pytest.mark.asyncio
async def test_bridge_path_same_source_target():
    from dewie.api.routes.graph import bridge_path

    req = _make_request()
    result = await bridge_path(req, {"source_id": "d1", "target_id": "d1"})
    assert result["path"] == ["d1"]
    assert result["hops"] == 0


@pytest.mark.asyncio
async def test_bridge_path_no_path_found():
    from dewie.api.routes.graph import bridge_path

    pg = _make_pg([])  # no edges
    req = _make_request(pg)
    result = await bridge_path(req, {"source_id": "d1", "target_id": "d2", "max_depth": 2})
    assert "error" in result
    assert result["hops"] == -1


@pytest.mark.asyncio
async def test_bridge_path_direct_connection():
    from dewie.api.routes.graph import bridge_path

    pg = MagicMock()
    session = AsyncMock()
    pg._apply_tenant = AsyncMock()

    # First execute call returns edges (d1 -> d2), second returns doc titles
    edge_row = MagicMock()
    edge_row.__getitem__ = lambda self, k: {"source_id": "d1", "target_id": "d2", "weight": 0.9}[k]

    doc_row = MagicMock()
    doc_row_dict = {"id": "d1", "title": "Doc 1", "summary": "s", "keywords": [], "entities": []}
    doc_row2_dict = {"id": "d2", "title": "Doc 2", "summary": "s", "keywords": [], "entities": []}

    call_count = 0

    async def mock_execute(sql, params):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        if call_count == 1:
            # Edge query
            m = MagicMock()
            m.all.return_value = [{"source_id": "d1", "target_id": "d2", "weight": 0.9}]
            result.mappings.return_value = m
        else:
            # Doc fetch
            m = MagicMock()
            m.all.return_value = [doc_row_dict, doc_row2_dict]
            result.mappings.return_value = m
        return result

    session.execute = mock_execute
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    pg._session_factory = MagicMock(return_value=cm)

    req = _make_request(pg)
    result = await bridge_path(req, {"source_id": "d1", "target_id": "d2"})
    assert result["hops"] == 1


# ── _neighbors_raw ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_neighbors_raw_returns_list():
    from dewie.api.routes.graph import _neighbors_raw

    session = AsyncMock()
    row = {
        "doc_id": "d2",
        "title": "T",
        "summary": "S",
        "keywords": [],
        "entities": [],
        "answers_questions": [],
        "weight": 0.8,
    }
    m = MagicMock()
    m.all.return_value = [row]
    result = MagicMock()
    result.mappings.return_value = m
    session.execute = AsyncMock(return_value=result)

    rows = await _neighbors_raw(session, "d1", limit=10)
    assert len(rows) == 1
    assert "answers_questions" not in rows[0]  # filtered out


@pytest.mark.asyncio
async def test_neighbors_raw_empty():
    from dewie.api.routes.graph import _neighbors_raw

    session = AsyncMock()
    m = MagicMock()
    m.all.return_value = []
    result = MagicMock()
    result.mappings.return_value = m
    session.execute = AsyncMock(return_value=result)

    rows = await _neighbors_raw(session, "d1")
    assert rows == []
