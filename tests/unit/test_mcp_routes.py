"""Tests for dewie.api.routes.mcp — MCP tool endpoint."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dewie.api.routes.mcp import router


def _make_app(pg: MagicMock | None = None) -> FastAPI:
    """Create a minimal FastAPI app with the MCP router."""
    from dewie.api.middleware import limiter

    pg = pg or AsyncMock()

    app = FastAPI()
    app.state.limiter = limiter

    async def _inject_state(request, call_next):
        request.state.workspace_ids = [uuid.UUID("00000000-0000-0000-0000-000000000010")]
        request.state.user_id = "user123"
        request.state.key_id = None
        return await call_next(request)

    app.middleware("http")(_inject_state)
    app.include_router(router)
    app.state.postgres = pg
    app.state.processor = None
    return app


# ── GET /mcp — manifest ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_manifest_returns_schema():
    """GET /mcp returns the tool manifest with search_corpus and ingest_url."""
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/mcp")
    assert resp.status_code == 200
    data = resp.json()
    assert "tools" in data
    tool_names = [t["name"] for t in data["tools"]]
    assert "search_corpus" in tool_names
    assert "ingest_url" in tool_names


@pytest.mark.asyncio
async def test_mcp_manifest_schema_version():
    """GET /mcp has schema_version field."""
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/mcp")
    assert resp.json()["schema_version"] == "1.0"


# ── POST /mcp — search_corpus ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_search_corpus_returns_results():
    """POST /mcp search_corpus returns safe results without answers_questions."""
    pg = AsyncMock()
    doc = MagicMock()
    doc.title = "Test Document"
    doc.url = "https://example.com/doc"
    doc.summary = "A test document."
    from dewie.models.content import DocumentType

    doc.document_type = DocumentType.NEWS_ARTICLE
    doc.source = "web"
    pg.search = AsyncMock(return_value=[(doc, 0.92)])

    app = _make_app(pg=pg)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/mcp", json={"tool": "search_corpus", "input": {"query": "test query"}}
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["tool"] == "search_corpus"
    assert data["content"]["count"] == 1
    result = data["content"]["results"][0]
    assert result["title"] == "Test Document"
    assert "answers_questions" not in result  # AQ never exposed


@pytest.mark.asyncio
async def test_mcp_search_corpus_empty_query_rejected():
    """POST /mcp search_corpus with empty query returns 422."""
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/mcp", json={"tool": "search_corpus", "input": {"query": ""}})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_mcp_search_corpus_respects_limit():
    """POST /mcp search_corpus limits results to max 25."""
    pg = AsyncMock()
    pg.search = AsyncMock(return_value=[])
    app = _make_app(pg=pg)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/mcp", json={"tool": "search_corpus", "input": {"query": "test", "limit": 100}}
        )
    # Should have been called with limit=25 (max cap)
    call_kwargs = pg.search.call_args.kwargs if pg.search.call_args else {}
    assert call_kwargs.get("limit", 25) <= 25


@pytest.mark.asyncio
async def test_mcp_search_corpus_db_error_returns_500():
    """POST /mcp search_corpus returns 500 when DB raises an exception."""
    pg = AsyncMock()
    pg.search = AsyncMock(side_effect=Exception("DB down"))
    app = _make_app(pg=pg)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/mcp", json={"tool": "search_corpus", "input": {"query": "test"}})
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_mcp_search_corpus_with_corpus_id():
    """POST /mcp search_corpus with corpus_id calls pg.search successfully."""
    pg = AsyncMock()
    pg.search = AsyncMock(return_value=[])
    app = _make_app(pg=pg)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/mcp",
            json={
                "tool": "search_corpus",
                "input": {"query": "test", "corpus_id": "my_corpus"},
            },
        )
    assert resp.status_code == 200
    assert pg.search.called


# ── POST /mcp — auth guard ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_call_requires_auth():
    """POST /mcp without auth returns 401 when auth is enabled."""
    from unittest.mock import patch

    from dewie.api.middleware import limiter
    from dewie.config import settings as _real_settings

    app = FastAPI()
    app.state.limiter = limiter

    async def _no_auth(request, call_next):
        request.state.user_id = None
        request.state.workspace_ids = []
        request.state.key_id = None
        return await call_next(request)

    app.middleware("http")(_no_auth)
    app.include_router(router)
    app.state.postgres = AsyncMock()
    app.state.processor = None

    with patch.object(_real_settings, "auth_enabled", True):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/mcp", json={"tool": "search_corpus", "input": {"query": "test"}})
    assert resp.status_code == 401


# ── POST /mcp — ingest_url ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_ingest_url_empty_url_rejected():
    """POST /mcp ingest_url with empty url returns 422."""
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/mcp", json={"tool": "ingest_url", "input": {"url": ""}})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_mcp_ingest_url_no_user_id_rejected():
    """POST /mcp ingest_url without user-level auth returns 403."""
    from dewie.api.middleware import limiter

    app = FastAPI()
    app.state.limiter = limiter

    # Has workspace_ids but no user_id — ingest still requires user-level auth
    async def _key_only(request, call_next):
        request.state.user_id = None
        request.state.workspace_ids = [uuid.UUID("00000000-0000-0000-0000-000000000010")]
        request.state.key_id = None
        return await call_next(request)

    app.middleware("http")(_key_only)
    app.include_router(router)
    app.state.postgres = AsyncMock()
    app.state.processor = None

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/mcp", json={"tool": "ingest_url", "input": {"url": "https://example.com"}}
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_mcp_ingest_url_fetches_and_upserts():
    """POST /mcp ingest_url fetches the URL and upserts the document."""
    pg = AsyncMock()
    pg.upsert = AsyncMock()
    pg.write_body_text = AsyncMock()

    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.body = "Some body text"
    doc.corpus_id = None

    mock_ingester = AsyncMock()
    mock_ingester.__aenter__ = AsyncMock(return_value=mock_ingester)
    mock_ingester.__aexit__ = AsyncMock(return_value=None)

    async def _fake_fetch(url):
        yield doc

    mock_ingester.fetch = _fake_fetch

    app = _make_app(pg=pg)
    app.state.processor = None
    with (
        patch("dewie.ingestion.web.WebIngester", return_value=mock_ingester),
        patch("dewie.storage.body_store.save_body"),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/mcp", json={"tool": "ingest_url", "input": {"url": "https://example.com"}}
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["tool"] == "ingest_url"


@pytest.mark.asyncio
async def test_mcp_ingest_url_no_content_returns_422():
    """POST /mcp ingest_url when ingester returns nothing gives 422."""
    pg = AsyncMock()
    app = _make_app(pg=pg)

    mock_ingester = AsyncMock()
    mock_ingester.__aenter__ = AsyncMock(return_value=mock_ingester)
    mock_ingester.__aexit__ = AsyncMock(return_value=None)

    async def _empty_fetch(url):
        return
        yield  # make it an async generator

    mock_ingester.fetch = _empty_fetch

    with patch("dewie.ingestion.web.WebIngester", return_value=mock_ingester):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/mcp", json={"tool": "ingest_url", "input": {"url": "https://example.com"}}
            )

    assert resp.status_code == 422


# ── POST /mcp — logging ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_call_logging_has_request_id():
    """POST /mcp log entries include the request_id from request.state."""
    pg = AsyncMock()
    doc = MagicMock()
    doc.title = "Test Document"
    doc.url = "https://example.com/doc"
    doc.summary = "A test document."
    from dewie.models.content import DocumentType
    doc.document_type = DocumentType.NEWS_ARTICLE
    doc.source = "web"
    pg.search = AsyncMock(return_value=[(doc, 0.92)])

    request_id = "test-req-abc123"

    app = FastAPI()
    app.state.limiter = _make_app().state.limiter

    async def _inject_request_id(request, call_next):
        request.state.request_id = request_id
        request.state.workspace_ids = [uuid.UUID("00000000-0000-0000-0000-000000000010")]
        request.state.user_id = "user123"
        request.state.key_id = None
        return await call_next(request)

    app.middleware("http")(_inject_request_id)
    app.include_router(router)
    app.state.postgres = pg
    app.state.processor = None

    import io
    import logging

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    mcp_log = logging.getLogger("dewie.api.routes.mcp")
    mcp_log.addHandler(handler)
    mcp_log.setLevel(logging.DEBUG)

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/mcp", json={"tool": "search_corpus", "input": {"query": "test query"}}
            )

        assert resp.status_code == 200
        log_output = log_stream.getvalue()
        assert request_id in log_output
    finally:
        mcp_log.removeHandler(handler)


@pytest.mark.asyncio
async def test_mcp_call_redacts_sensitive_fields():
    """POST /mcp redacts sensitive fields like api_key in log output."""
    from dewie.api.mcp_dispatch import _redact

    assert _redact("my_api_key_is_here") == "***REDACTED***"
    assert _redact("password123") == "***REDACTED***"
    assert _redact("bearer_token_here") == "***REDACTED***"
    assert _redact("secret_value") == "***REDACTED***"
    assert _redact("https://example.com/doc") == "https://example.com/doc"
    assert _redact(None) is None


@pytest.mark.asyncio
async def test_mcp_call_truncates_long_values():
    """POST /mcp truncates values longer than 1000 chars for logging."""
    from dewie.api.mcp_dispatch import _redact

    long_value = "a" * 1500
    result = _redact(long_value)
    assert len(result) == 1000 + len("... [truncated]")
    assert "... [truncated]" in result


@pytest.mark.asyncio
async def test_mcp_call_request_id_defaults_to_unknown():
    """POST /mcp uses 'unknown' when request_id is not in request.state."""
    pg = AsyncMock()
    doc = MagicMock()
    doc.title = "Test"
    doc.url = None
    doc.summary = "summary"
    from dewie.models.content import DocumentType
    doc.document_type = DocumentType.NEWS_ARTICLE
    doc.source = "web"
    pg.search = AsyncMock(return_value=[(doc, 0.5)])

    import io
    import logging

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    mcp_log = logging.getLogger("dewie.api.routes.mcp")
    mcp_log.addHandler(handler)
    mcp_log.setLevel(logging.DEBUG)

    app = FastAPI()
    app.state.limiter = _make_app().state.limiter

    async def _no_request_id(request, call_next):
        # No request_id set
        request.state.workspace_ids = [uuid.UUID("00000000-0000-0000-0000-000000000010")]
        request.state.user_id = "user123"
        request.state.key_id = None
        return await call_next(request)

    app.middleware("http")(_no_request_id)
    app.include_router(router)
    app.state.postgres = pg
    app.state.processor = None

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/mcp", json={"tool": "search_corpus", "input": {"query": "test"}}
            )

        assert resp.status_code == 200
        log_output = log_stream.getvalue()
        assert "unknown" in log_output
    finally:
        mcp_log.removeHandler(handler)


@pytest.mark.asyncio
async def test_mcp_call_success_has_timing():
    """POST /mcp search success log includes elapsed_ms timing."""
    from dewie.models.content import DocumentType

    pg = AsyncMock()
    doc = MagicMock()
    doc.title = "Test"
    doc.url = None
    doc.summary = "summary"
    doc.document_type = DocumentType.NEWS_ARTICLE
    doc.source = "web"
    pg.search = AsyncMock(return_value=[(doc, 0.5)])

    import io
    import logging

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    mcp_log = logging.getLogger("dewie.api.mcp_dispatch")
    mcp_log.addHandler(handler)
    mcp_log.setLevel(logging.DEBUG)

    app = FastAPI()
    app.state.limiter = _make_app().state.limiter

    async def _inject_request_id(request, call_next):
        request.state.request_id = "timing-test"
        request.state.workspace_ids = [uuid.UUID("00000000-0000-0000-0000-000000000010")]
        request.state.user_id = "user123"
        request.state.key_id = None
        return await call_next(request)

    app.middleware("http")(_inject_request_id)
    app.include_router(router)
    app.state.postgres = pg
    app.state.processor = None

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/mcp", json={"tool": "search_corpus", "input": {"query": "test"}}
            )

        assert resp.status_code == 200
        log_output = log_stream.getvalue()
        assert "elapsed_ms" in log_output
    finally:
        mcp_log.removeHandler(handler)


@pytest.mark.asyncio
async def test_mcp_call_error_logging():
    """POST /mcp search failure logs with traceback using log.exception."""
    pg = AsyncMock()
    pg.search = AsyncMock(side_effect=Exception("DB down"))

    import io
    import logging

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    mcp_log = logging.getLogger("dewie.api.mcp_dispatch")
    mcp_log.addHandler(handler)
    mcp_log.setLevel(logging.DEBUG)

    app = FastAPI()
    app.state.limiter = _make_app().state.limiter

    async def _inject_request_id(request, call_next):
        request.state.request_id = "error-test"
        request.state.workspace_ids = [uuid.UUID("00000000-0000-0000-0000-000000000010")]
        request.state.user_id = "user123"
        request.state.key_id = None
        return await call_next(request)

    app.middleware("http")(_inject_request_id)
    app.include_router(router)
    app.state.postgres = pg
    app.state.processor = None

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/mcp", json={"tool": "search_corpus", "input": {"query": "test"}}
            )

        assert resp.status_code == 500
        log_output = log_stream.getvalue()
        assert "error-test" in log_output
        assert "dispatch search_corpus failed" in log_output
    finally:
        mcp_log.removeHandler(handler)
