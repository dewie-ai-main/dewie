"""Optional dev-only coding-harness E2E tests.

Exercises key tool and endpoint contracts from an agent/harness perspective.
Excluded from default CI runs.

Run with:
    PYTHONPATH=src pytest -m dev_agent_harness -o addopts='' -v

No live services required — all storage is mocked.
To run against a real localhost instance set:
    DEWIE_TEST_API_BASE=http://localhost:8000/api
in .env.remote-catalog.local and the suite will additionally run live checks.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.dev_agent_harness

_USER_ID = "00000000-0000-0000-0000-000000000042"
_WORKSPACE_ID = uuid.UUID("00000000-0000-0000-0000-000000000010")


# ── Shared app factories ──────────────────────────────────────────────────────


def _make_mcp_app(pg: AsyncMock | None = None) -> FastAPI:
    from dewie.api.middleware_base import limiter
    from dewie.api.routes.mcp import router

    pg = pg or AsyncMock()

    app = FastAPI()
    app.state.limiter = limiter
    app.state.postgres = pg
    app.state.processor = None

    async def _auth(request, call_next):
        request.state.user_id = _USER_ID
        request.state.is_admin = False
        request.state.workspace_ids = [_WORKSPACE_ID]
        request.state.key_id = None
        request.state.key_scopes = ["read"]
        return await call_next(request)

    app.middleware("http")(_auth)
    app.include_router(router)
    return app


def _make_query_app(pg: AsyncMock | None = None) -> FastAPI:
    from dewie.api.middleware_base import limiter
    from dewie.api.routes.query import router

    pg = pg or AsyncMock()
    pg.search = AsyncMock(return_value=[])
    pg.get_edge_counts = AsyncMock(return_value={})
    pg.enqueue_search = AsyncMock(return_value=(False, None))
    pg.search_chunks_for_docs = AsyncMock(return_value={})

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    pg._session_factory = MagicMock(return_value=session_cm)

    cache = AsyncMock()
    cache.get_query_result = AsyncMock(return_value=None)
    cache.set_query_result = AsyncMock()
    cache.get_tenant_plan = AsyncMock(return_value="free")
    cache.incr_quota = AsyncMock(return_value=1)
    cache.decr_quota = AsyncMock()

    app = FastAPI()
    app.state.limiter = limiter
    app.state.postgres = pg
    app.state.cache = cache
    app.include_router(router)
    return app


# ── MCP: tool manifest contract ───────────────────────────────────────────────


def test_mcp_manifest_has_schema_version():
    client = TestClient(_make_mcp_app())
    resp = client.get("/mcp")
    assert resp.status_code == 200
    assert resp.json()["schema_version"] == "1.0"


def test_mcp_manifest_required_tools_present():
    """All tools expected by coding agents must be in the manifest."""
    client = TestClient(_make_mcp_app())
    resp = client.get("/mcp")
    tools = {t["name"] for t in resp.json()["tools"]}
    required = {"search_corpus", "ingest_url", "expand", "read", "intersect", "bridge", "web_search"}
    missing = required - tools
    assert not missing, f"Missing tools: {missing}"


def test_mcp_manifest_each_tool_has_input_schema():
    client = TestClient(_make_mcp_app())
    resp = client.get("/mcp")
    for tool in resp.json()["tools"]:
        assert "input_schema" in tool, f"Tool {tool['name']} missing input_schema"
        assert "required" in tool["input_schema"] or "properties" in tool["input_schema"]


# ── MCP: search_corpus contract ───────────────────────────────────────────────


def test_mcp_search_corpus_empty_query_rejected():
    client = TestClient(_make_mcp_app())
    resp = client.post("/mcp", json={"tool": "search_corpus", "input": {"query": ""}})
    assert resp.status_code == 422


def test_mcp_search_corpus_returns_count_field():
    pg = AsyncMock()
    pg.search = AsyncMock(return_value=[])
    client = TestClient(_make_mcp_app(pg))
    resp = client.post("/mcp", json={"tool": "search_corpus", "input": {"query": "test"}})
    assert resp.status_code == 200
    assert "count" in resp.json()["content"]


def test_mcp_search_corpus_never_exposes_aq():
    """answers_questions must never appear anywhere in the response."""
    pg = AsyncMock()
    doc = SimpleNamespace(
        id=uuid.uuid4(),
        title="Test Doc",
        url="https://example.com",
        summary="A test document about testing.",
        answers_questions=["how do you test?"],  # must not leak
        source="example.com",
        document_type=None,
        topics=[],
        keywords=[],
        entities=[],
        sentiment=None,
        ingested_at=None,
        published_at=None,
    )
    pg.search = AsyncMock(return_value=[(doc, 0.9)])
    client = TestClient(_make_mcp_app(pg))
    resp = client.post("/mcp", json={"tool": "search_corpus", "input": {"query": "test"}})
    assert resp.status_code == 200
    assert "answers_questions" not in resp.text


# ── MCP: unknown tool contract ────────────────────────────────────────────────


def test_mcp_unknown_tool_returns_422_with_available_list():
    client = TestClient(_make_mcp_app())
    resp = client.post("/mcp", json={"tool": "nonexistent_tool", "input": {}})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "search_corpus" in detail  # must enumerate valid tools


# ── MCP: expand contract ──────────────────────────────────────────────────────


def test_mcp_expand_empty_doc_id_rejected():
    client = TestClient(_make_mcp_app())
    resp = client.post("/mcp", json={"tool": "expand", "input": {"doc_id": ""}})
    assert resp.status_code == 422


# ── MCP: bridge contract ──────────────────────────────────────────────────────


def test_mcp_bridge_missing_target_rejected():
    client = TestClient(_make_mcp_app())
    resp = client.post(
        "/mcp",
        json={"tool": "bridge", "input": {"source_id": str(uuid.uuid4())}},
    )
    assert resp.status_code == 422


# ── MCP: intersect contract ───────────────────────────────────────────────────


def test_mcp_intersect_requires_at_least_two_docs():
    client = TestClient(_make_mcp_app())
    resp = client.post(
        "/mcp",
        json={"tool": "intersect", "input": {"doc_ids": [str(uuid.uuid4())]}},
    )
    assert resp.status_code == 422


# ── Query endpoint contract ───────────────────────────────────────────────────


def test_query_empty_string_returns_422():
    client = TestClient(_make_query_app())
    resp = client.post("/query", json={"query": ""})
    assert resp.status_code == 422


def test_query_missing_body_returns_422():
    client = TestClient(_make_query_app())
    resp = client.post("/query", json={})
    assert resp.status_code == 422


def test_query_returns_results_field():
    client = TestClient(_make_query_app())
    resp = client.post("/query", json={"query": "what is dewie?"})
    assert resp.status_code == 200
    assert "results" in resp.json()


def test_query_rankers_endpoint_returns_list():
    client = TestClient(_make_query_app())
    resp = client.get("/query/rankers")
    assert resp.status_code == 200
    rankers = resp.json()
    assert isinstance(rankers, list)
    assert len(rankers) > 0
    assert any(r.get("id") == "rrf" for r in rankers)


def test_query_response_never_exposes_aq():
    """answers_questions must never leak in any query response."""
    from dewie.models.content import ContentDocument, ContentStatus

    doc = ContentDocument(
        url="https://example.com/doc",
        title="Doc",
        status=ContentStatus.READY,
        answers_questions=["how?"],  # trade secret — must not appear in response
        summary="Summary.",
    )
    pg = AsyncMock()
    pg.search = AsyncMock(return_value=[(doc, 0.9)])
    pg.get_edge_counts = AsyncMock(return_value={})
    pg.enqueue_search = AsyncMock(return_value=(False, None))
    pg.search_chunks_for_docs = AsyncMock(return_value={})
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    pg._session_factory = MagicMock(return_value=session_cm)

    client = TestClient(_make_query_app(pg))
    resp = client.post("/query", json={"query": "test query"})
    assert resp.status_code == 200
    assert "answers_questions" not in resp.text


# ── Service status contract ───────────────────────────────────────────────────


def test_service_status_endpoint_returns_expected_shape():
    from dewie.api.routes.service_status import router

    app = FastAPI()

    async def _auth(request, call_next):
        request.state.user_id = _USER_ID
        request.state.is_admin = False
        return await call_next(request)

    app.middleware("http")(_auth)
    app.include_router(router)
    app.state.postgres = AsyncMock()
    app.state.cache = AsyncMock()

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/service-status")

    assert resp.status_code == 200
    body = resp.json()
    # Minimal contract: must have some status indicator
    assert isinstance(body, dict)
    assert len(body) > 0


# ── Optional: live localhost smoke ────────────────────────────────────────────


def _try_load_live_base() -> str | None:
    """Return live API base if .env.remote-catalog.local has DEWIE_TEST_API_BASE."""
    try:
        from tests.e2e.conftest import load_dev_env_file
        load_dev_env_file()
        import os
        return os.environ.get("DEWIE_TEST_API_BASE")
    except Exception:
        return None


_LIVE_BASE = _try_load_live_base()


@pytest.mark.skipif(not _LIVE_BASE, reason="DEWIE_TEST_API_BASE not set in .env.remote-catalog.local")
def test_live_health_endpoint():
    """Smoke: confirm localhost API health endpoint responds."""
    import httpx

    resp = httpx.get(f"{_LIVE_BASE}/health", timeout=5)
    assert resp.status_code == 200


@pytest.mark.skipif(not _LIVE_BASE, reason="DEWIE_TEST_API_BASE not set in .env.remote-catalog.local")
def test_live_mcp_manifest():
    """Smoke: confirm localhost MCP manifest lists tools."""
    import httpx

    resp = httpx.get(f"{_LIVE_BASE}/mcp", timeout=5)
    assert resp.status_code == 200
    tools = {t["name"] for t in resp.json()["tools"]}
    assert "search_corpus" in tools


@pytest.mark.skipif(not _LIVE_BASE, reason="DEWIE_TEST_API_BASE not set in .env.remote-catalog.local")
def test_live_query_endpoint():
    """Smoke: confirm localhost /query accepts a valid request."""
    import httpx

    resp = httpx.post(
        f"{_LIVE_BASE}/query",
        json={"query": "test harness ping"},
        timeout=10,
    )
    # Accept 200 (results) or 503 (no documents yet) — both indicate the pipeline is wired
    assert resp.status_code in (200, 503)
