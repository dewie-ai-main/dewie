"""Tests for dewie.api.routes.admin — admin API endpoints."""

from __future__ import annotations

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Pydantic model tests ───────────────────────────────────────────────────────


def test_create_key_request_defaults():
    from dewie.api.routes.admin import CreateKeyRequest

    req = CreateKeyRequest()
    assert req.scopes == ["read"]
    assert req.live is True
    assert req.name is None


def test_key_response_model():
    from dewie.api.routes.admin import KeyResponse

    key_id = uuid.uuid4()
    resp = KeyResponse(
        id=key_id,
        tenant_id=uuid.uuid4(),
        key_prefix="ck_live_abc",
        scopes=["read"],
        name="test",
        created_at="2026-01-01",
    )
    assert resp.id == key_id


def test_create_workspace_request_defaults():
    from dewie.api.routes.admin import CreateWorkspaceRequest

    req = CreateWorkspaceRequest(name="test-ws")
    assert req.sharing_tier == "internal_only"
    assert req.parent_id is None


def test_create_corpus_request_fields():
    from dewie.api.routes.admin import CreateCorpusRequest

    ws_id = uuid.uuid4()
    req = CreateCorpusRequest(name="My Corpus", slug="my-corpus", workspace_id=ws_id)
    assert req.slug == "my-corpus"
    assert req.workspace_id == ws_id


# ── Helper function tests ──────────────────────────────────────────────────────


def test_require_admin_session_passes_when_is_admin():
    from fastapi import Request

    from dewie.api.routes.admin import _require_admin_session

    mock_request = MagicMock(spec=Request)
    mock_request.state.is_admin = True
    _require_admin_session(mock_request)  # should not raise


def test_require_admin_session_raises_when_not_admin():
    from fastapi import HTTPException, Request

    from dewie.api.routes.admin import _require_admin_session

    mock_request = MagicMock(spec=Request)
    mock_request.state.is_admin = False
    with pytest.raises(HTTPException) as exc_info:
        _require_admin_session(mock_request)
    assert exc_info.value.status_code == 403


def test_pg_helper():
    from fastapi import Request

    from dewie.api.routes.admin import _pg

    mock_pg = MagicMock()
    mock_request = MagicMock(spec=Request)
    mock_request.app.state.postgres = mock_pg
    result = _pg(mock_request)
    assert result is mock_pg


# ── Workspace endpoints ────────────────────────────────────────────────────────


def _make_request(is_admin: bool = True, pg: object = None) -> MagicMock:
    from fastapi import Request

    mock_pg = pg or MagicMock()
    req = MagicMock(spec=Request)
    req.state.is_admin = is_admin
    req.state.tenant_id = uuid.uuid4()
    req.app.state.postgres = mock_pg
    return req


@pytest.mark.asyncio
async def test_create_workspace_success():
    from dewie.api.routes.admin import CreateWorkspaceRequest, create_workspace

    ws_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.create_workspace = AsyncMock(
        return_value={
            "id": ws_id,
            "name": "Test WS",
            "parent_id": None,
            "sharing_tier": "internal_only",
            "created_at": "2026-01-01",
        }
    )
    req = _make_request(pg=mock_pg)
    body = CreateWorkspaceRequest(name="Test WS")
    result = await create_workspace(body, req)
    assert result.id == ws_id
    assert result.name == "Test WS"


@pytest.mark.asyncio
async def test_list_workspaces_success():
    from dewie.api.routes.admin import list_workspaces

    mock_pg = AsyncMock()
    mock_pg.get_workspaces = AsyncMock(
        return_value=[
            {
                "id": uuid.uuid4(),
                "name": "WS1",
                "parent_id": None,
                "sharing_tier": "internal_only",
                "created_at": "2026-01-01",
            }
        ]
    )
    req = _make_request(pg=mock_pg)
    result = await list_workspaces(req)
    assert len(result) == 1
    assert result[0].name == "WS1"


@pytest.mark.asyncio
async def test_delete_workspace():
    from dewie.api.routes.admin import delete_workspace

    mock_pg = AsyncMock()
    mock_pg.delete_workspace = AsyncMock(return_value=None)
    req = _make_request(pg=mock_pg)
    await delete_workspace(uuid.uuid4(), req)
    mock_pg.delete_workspace.assert_awaited_once()


# ── Corpus endpoints ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_corpus_success():
    from dewie.api.routes.admin import CreateCorpusRequest, create_corpus

    corpus_id = uuid.uuid4()
    ws_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.create_corpus = AsyncMock(
        return_value={
            "id": corpus_id,
            "name": "Test Corpus",
            "slug": "test-corpus",
            "workspace_id": ws_id,
            "sharing_tier": "internal_only",
            "created_at": "2026-01-01",
        }
    )
    req = _make_request(pg=mock_pg)
    body = CreateCorpusRequest(name="Test Corpus", slug="test-corpus", workspace_id=ws_id)
    result = await create_corpus(body, req)
    assert result.id == corpus_id
    assert result.slug == "test-corpus"


@pytest.mark.asyncio
async def test_list_corpora_success():
    from dewie.api.routes.admin import list_corpora

    ws_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.get_corpora = AsyncMock(
        return_value=[
            {
                "id": uuid.uuid4(),
                "name": "C1",
                "slug": "c1",
                "workspace_id": ws_id,
                "sharing_tier": "internal_only",
                "created_at": "2026-01-01",
            }
        ]
    )
    req = _make_request(pg=mock_pg)
    result = await list_corpora(req)
    assert len(result) == 1


@pytest.mark.asyncio
async def test_delete_corpus():
    from dewie.api.routes.admin import delete_corpus

    mock_pg = AsyncMock()
    mock_pg.delete_corpus = AsyncMock(return_value=None)
    req = _make_request(pg=mock_pg)
    await delete_corpus(uuid.uuid4(), req)
    mock_pg.delete_corpus.assert_awaited_once()


# ── Query log endpoints (tenant-scoped) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_query_log_tenant_scoped():
    from dewie.api.routes.admin import list_query_log

    tenant_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.get_query_log = AsyncMock(
        return_value=[
            {"id": 1, "ts": "2026-01-01", "question": "q", "answer": "a"}
        ]
    )
    req = _make_request(pg=mock_pg)
    req.state.tenant_id = tenant_id
    result = await list_query_log(req)
    assert len(result) == 1
    mock_pg.get_query_log.assert_awaited_once_with(tenant_id=tenant_id, limit=100)


@pytest.mark.asyncio
async def test_get_query_log_entry_tenant_scoped():
    from dewie.api.routes.admin import get_query_log_entry

    tenant_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.get_query_log_entry = AsyncMock(
        return_value={"id": 1, "ts": "2026-01-01", "question": "q", "answer": "a"}
    )
    req = _make_request(pg=mock_pg)
    req.state.tenant_id = tenant_id
    result = await get_query_log_entry(999, req)
    assert result["id"] == 1
    mock_pg.get_query_log_entry.assert_awaited_once_with(query_id=999, tenant_id=tenant_id)


@pytest.mark.asyncio
async def test_get_query_log_entry_not_found():
    from fastapi import HTTPException

    from dewie.api.routes.admin import get_query_log_entry

    tenant_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.get_query_log_entry = AsyncMock(return_value=None)
    req = _make_request(pg=mock_pg)
    req.state.tenant_id = tenant_id

    with pytest.raises(HTTPException) as exc_info:
        await get_query_log_entry(999, req)
    assert exc_info.value.status_code == 404


# ── Catalog connection tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_test_postgres_connection_with_dsn():
    from dewie.api.routes.admin import _test_postgres_connection

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_engine = AsyncMock()
    mock_engine.connect = MagicMock(return_value=mock_conn)
    mock_engine.dispose = AsyncMock()

    with patch(
        "sqlalchemy.ext.asyncio.create_async_engine", return_value=mock_engine
    ):
        ok, error = await _test_postgres_connection({"dsn": "postgresql://localhost/db"})

    assert ok is True
    assert error is None


@pytest.mark.asyncio
async def test_test_postgres_connection_with_host_params():
    from dewie.api.routes.admin import _test_postgres_connection

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_engine = AsyncMock()
    mock_engine.connect = MagicMock(return_value=mock_conn)
    mock_engine.dispose = AsyncMock()

    with patch(
        "sqlalchemy.ext.asyncio.create_async_engine", return_value=mock_engine
    ):
        ok, error = await _test_postgres_connection({
            "host": "db.example.com",
            "port": "5432",
            "database": "mydb",
            "user": "admin",
            "password": "secret",
        })

    assert ok is True
    assert error is None


@pytest.mark.asyncio
async def test_test_postgres_connection_failure():
    from dewie.api.routes.admin import _test_postgres_connection

    mock_engine = AsyncMock()
    mock_engine.connect = MagicMock(side_effect=Exception("connection refused"))

    with patch(
        "sqlalchemy.ext.asyncio.create_async_engine", return_value=mock_engine
    ):
        ok, error = await _test_postgres_connection({"dsn": "postgresql://localhost/db"})

    assert ok is False
    assert "connection refused" in error


# ── Catalog MCP connection tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_test_mcp_connection_success():
    from dewie.api.routes.admin import _test_mcp_connection

    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        ok, error = await _test_mcp_connection({"endpoint": "http://mcp.example.com"})

    assert ok is True
    assert error is None


@pytest.mark.asyncio
async def test_test_mcp_connection_missing_endpoint():
    from dewie.api.routes.admin import _test_mcp_connection

    ok, error = await _test_mcp_connection({})

    assert ok is False
    assert "Missing endpoint" in error


@pytest.mark.asyncio
async def test_test_mcp_connection_auth_failure():
    from dewie.api.routes.admin import _test_mcp_connection

    mock_response = MagicMock()
    mock_response.status_code = 401

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        ok, error = await _test_mcp_connection({
            "endpoint": "http://mcp.example.com",
            "api_key": "secret-key",
        })

    assert ok is False
    assert "Authentication failed" in error


@pytest.mark.asyncio
async def test_test_mcp_connection_no_compatible_endpoint():
    from dewie.api.routes.admin import _test_mcp_connection

    mock_response = MagicMock()
    mock_response.status_code = 404

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        ok, error = await _test_mcp_connection({"endpoint": "http://mcp.example.com"})

    assert ok is False
    assert "No compatible API endpoint found" in error


# ── Test source endpoint (query a catalog) ────────────────────────────────────


@ pytest.mark.asyncio
async def test_test_source_postgres_success():
    from dewie.api.routes.admin import TestCatalogResponse, test_catalog

    source_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.get_source = AsyncMock(return_value={
        "id": str(source_id),
        "type": "postgres",
        "config": {
            "host": "localhost",
            "port": "5432",
            "database": "testdb",
            "user": "admin",
        },
    })
    mock_pg.set_source_test_result = AsyncMock()

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.commit = AsyncMock()
    mock_pg._engine.connect = MagicMock(return_value=mock_conn)

    req = _make_request(pg=mock_pg)

    with patch("sqlalchemy.ext.asyncio.create_async_engine") as mock_engine_factory:
        mock_engine = AsyncMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)
        mock_engine.dispose = AsyncMock()
        mock_engine_factory.return_value = mock_engine

        result = await test_catalog(source_id, req)

    assert isinstance(result, TestCatalogResponse)
    assert result.ok is True
    assert result.error is None
    mock_pg.set_source_test_result.assert_awaited_once_with(source_id, ok=True, error=None)


@ pytest.mark.asyncio
async def test_test_source_mcp_success():
    from dewie.api.routes.admin import TestCatalogResponse, test_catalog

    source_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.get_source = AsyncMock(return_value={
        "id": str(source_id),
        "type": "mcp",
        "config": {
            "endpoint": "http://mcp.example.com",
        },
    })
    mock_pg.set_source_test_result = AsyncMock()

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.commit = AsyncMock()
    mock_pg._engine.connect = MagicMock(return_value=mock_conn)

    req = _make_request(pg=mock_pg)

    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await test_catalog(source_id, req)

    assert isinstance(result, TestCatalogResponse)
    assert result.ok is True
    assert result.error is None


@ pytest.mark.asyncio
async def test_test_source_sqlite_success():
    import tempfile

    from dewie.api.routes.admin import TestCatalogResponse, test_catalog

    source_id = uuid.uuid4()
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        filepath = f.name

    try:
        mock_pg = AsyncMock()
        mock_pg.get_source = AsyncMock(return_value={
            "id": str(source_id),
            "type": "sqlite",
            "config": {"filepath": filepath},
        })
        mock_pg.set_source_test_result = AsyncMock()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        mock_pg._engine.connect = MagicMock(return_value=mock_conn)

        req = _make_request(pg=mock_pg)
        result = await test_catalog(source_id, req)

        assert isinstance(result, TestCatalogResponse)
        assert result.ok is True
        assert result.error is None
    finally:
        os.unlink(filepath)


@ pytest.mark.asyncio
async def test_test_source_sqlite_file_not_found():
    from dewie.api.routes.admin import TestCatalogResponse, test_catalog

    source_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.get_source = AsyncMock(return_value={
        "id": str(source_id),
        "type": "sqlite",
        "config": {"filepath": "/nonexistent/path/to/file.db"},
    })
    mock_pg.set_source_test_result = AsyncMock()

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.commit = AsyncMock()
    mock_pg._engine.connect = MagicMock(return_value=mock_conn)

    req = _make_request(pg=mock_pg)
    result = await test_catalog(source_id, req)

    assert isinstance(result, TestCatalogResponse)
    assert result.ok is False
    assert "File not found" in result.error


@ pytest.mark.asyncio
async def test_test_source_not_found():
    from fastapi import HTTPException

    from dewie.api.routes.admin import test_catalog

    source_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.get_source = AsyncMock(return_value=None)

    req = _make_request(pg=mock_pg)

    with pytest.raises(HTTPException) as exc_info:
        await test_catalog(source_id, req)

    assert exc_info.value.status_code == 404


@ pytest.mark.asyncio
async def test_test_source_unknown_type():
    from dewie.api.routes.admin import TestCatalogResponse, test_catalog

    source_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.get_source = AsyncMock(return_value={
        "id": str(source_id),
        "type": "unknown_type",
        "config": {},
    })
    mock_pg.set_source_test_result = AsyncMock()

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.commit = AsyncMock()
    mock_pg._engine.connect = MagicMock(return_value=mock_conn)

    req = _make_request(pg=mock_pg)
    result = await test_catalog(source_id, req)

    assert isinstance(result, TestCatalogResponse)
    assert result.ok is False
    assert "Unknown catalog type" in result.error
