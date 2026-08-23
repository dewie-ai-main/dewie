"""Tests for dewie.api.routes.admin — dewie catalogs CRUD and test endpoints."""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Pydantic model tests ───────────────────────────────────────────────────────


def test_create_catalog_request_defaults():
    from dewie.api.routes.admin import CreateCatalogRequest

    req = CreateCatalogRequest(name="test", type="sqlite")
    assert req.enabled is True
    assert req.config == {}


def test_update_catalog_request_defaults():
    from dewie.api.routes.admin import UpdateCatalogRequest

    req = UpdateCatalogRequest()
    assert req.name is None
    assert req.type is None
    assert req.config is None
    assert req.enabled is None


def test_catalog_source_response_model():
    from dewie.api.routes.admin import CatalogSourceResponse

    src_id = uuid.uuid4()
    resp = CatalogSourceResponse(
        id=src_id,
        name="test",
        type="sqlite",
        config={"filepath": "/tmp/test.db"},
        enabled=True,
        created_by=None,
        created_at="2026-01-01",
        tested_at="2026-01-02",
        test_status="ok",
        test_error=None,
        updated_at="2026-01-01",
    )
    assert resp.id == src_id
    assert resp.tested_at == "2026-01-02"
    assert resp.test_status == "ok"


def test_test_catalog_response_model():
    from dewie.api.routes.admin import TestCatalogResponse

    resp = TestCatalogResponse(ok=True)
    assert resp.ok is True
    assert resp.error is None

    resp2 = TestCatalogResponse(ok=False, error="connection failed")
    assert resp2.ok is False
    assert resp2.error == "connection failed"


# ── Helper ─────────────────────────────────────────────────────────────────────


def _make_request(is_admin: bool = True, pg: object = None) -> MagicMock:
    from fastapi import Request

    mock_pg = pg or MagicMock()
    req = MagicMock(spec=Request)
    req.state.is_admin = is_admin
    req.app.state.postgres = mock_pg
    return req


def _build_mock_conn(rows=None, rowcount=1, fetchone_row=None):
    """Build a mock DB connection context for raw SQL endpoints."""
    fake_rows = rows or []

    mock_result = MagicMock()
    mock_result.mappings.return_value.fetchall.return_value = fake_rows
    if fetchone_row is not None:
        mock_result.mappings.return_value.fetchone.return_value = fetchone_row
    else:
        mock_result.mappings.return_value.fetchone.return_value = (
            fake_rows[0] if fake_rows else None
        )
    mock_result.rowcount = rowcount

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)

    async def mock_aexit(*args, **kwargs):
        mock_conn.commit()
        return False

    mock_connect_ctx = MagicMock()
    mock_connect_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_connect_ctx.__aexit__ = mock_aexit

    mock_pg = MagicMock()
    mock_pg._engine.connect.return_value = mock_connect_ctx
    mock_pg._engine.begin.return_value = mock_connect_ctx
    return mock_pg, mock_conn


# ── create_catalog endpoint ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_catalog_invalid_type():
    from fastapi import HTTPException

    from dewie.api.routes.admin import CreateCatalogRequest, create_catalog

    req = _make_request()
    body = CreateCatalogRequest(name="test", type="invalid")
    with pytest.raises(HTTPException) as exc_info:
        await create_catalog(body, req)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_create_catalog_success():
    from dewie.api.routes.admin import CreateCatalogRequest, create_catalog

    src_id = uuid.uuid4()
    mock_pg, mock_conn = _build_mock_conn(
        rows=[
            {
                "id": str(src_id),
                "name": "test-db",
                "type": "sqlite",
                "config_json": {"filepath": "/tmp/test.db"},
                "enabled": True,
                "created_by": None,
                "created_at": "2026-01-01",
                "tested_at": None,
                "test_status": None,
                "test_error": None,
                "updated_at": "2026-01-01",
            }
        ]
    )

    req = _make_request(pg=mock_pg)
    body = CreateCatalogRequest(name="test-db", type="sqlite", config={"filepath": "/tmp/test.db"})

    # Patch uuid.uuid4 to return a predictable ID
    import unittest.mock as _mock

    with _mock.patch("dewie.api.routes.admin.uuid.uuid4", return_value=src_id):
        result = await create_catalog(body, req)

    assert result.id == src_id
    assert result.name == "test-db"
    assert result.type == "sqlite"
    assert result.enabled is True

    # Verify the INSERT was called
    mock_conn.execute.assert_called()
    call_args = mock_conn.execute.call_args
    # Check that commit was called
    mock_conn.commit.assert_called()


@pytest.mark.asyncio
async def test_create_catalog_unique_name_conflict():
    from fastapi import HTTPException

    from dewie.api.routes.admin import CreateCatalogRequest, create_catalog

    mock_pg, mock_conn = _build_mock_conn(rows=[])

    # Simulate a database unique constraint error

    async def raise_unique_error(*args, **kwargs):
        raise Exception("UNIQUE constraint failed: dewie_sources.name")

    mock_conn.execute = AsyncMock(side_effect=raise_unique_error)
    mock_conn.rollback = AsyncMock()

    req = _make_request(pg=mock_pg)
    body = CreateCatalogRequest(name="duplicate", type="sqlite")

    with pytest.raises(HTTPException) as exc_info:
        await create_catalog(body, req)
    assert exc_info.value.status_code == 409
    assert "duplicate" in exc_info.value.detail


@pytest.mark.asyncio
async def test_create_catalog_unknown_db_error():
    from fastapi import HTTPException

    from dewie.api.routes.admin import CreateCatalogRequest, create_catalog

    mock_pg, mock_conn = _build_mock_conn(rows=[])

    async def raise_db_error(*args, **kwargs):
        raise Exception("some random database error")

    mock_conn.execute = AsyncMock(side_effect=raise_db_error)
    mock_conn.rollback = AsyncMock()

    req = _make_request(pg=mock_pg)
    body = CreateCatalogRequest(name="test", type="postgres")

    with pytest.raises(HTTPException) as exc_info:
        await create_catalog(body, req)
    assert exc_info.value.status_code == 500


# ── list_catalogs endpoint ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_catalogs_empty():
    from dewie.api.routes.admin import list_catalogs

    mock_pg, mock_conn = _build_mock_conn(rows=[])

    req = _make_request(pg=mock_pg)
    result = await list_catalogs(req)
    assert result == []


@pytest.mark.asyncio
async def test_list_catalogs_with_data():
    from dewie.api.routes.admin import list_catalogs

    src_id = uuid.uuid4()
    fake_rows = [
        {
            "id": str(src_id),
            "name": "test-db",
            "type": "sqlite",
            "config_json": '{"filepath": "/tmp/test.db"}',
            "enabled": True,
            "created_by": None,
            "created_at": "2026-01-01",
            "tested_at": None,
            "test_status": None,
            "test_error": None,
            "updated_at": "2026-01-01",
        }
    ]

    mock_pg, mock_conn = _build_mock_conn(rows=fake_rows)

    req = _make_request(pg=mock_pg)
    result = await list_catalogs(req)

    assert len(result) == 1
    assert result[0].id == src_id
    assert result[0].name == "test-db"
    assert result[0].config == {"filepath": "/tmp/test.db"}
    assert result[0].tested_at is None


@pytest.mark.asyncio
async def test_list_catalogs_json_config_string_parsing():
    """Ensure config_json stored as string is correctly parsed back."""
    from dewie.api.routes.admin import list_catalogs

    src_id = uuid.uuid4()
    fake_rows = [
        {
            "id": str(src_id),
            "name": "test-db",
            "type": "postgres",
            "config_json": json.dumps({"host": "localhost", "port": 5432}),
            "enabled": False,
            "created_by": None,
            "created_at": "2026-01-01",
            "tested_at": "2026-01-02",
            "test_status": "ok",
            "test_error": None,
            "updated_at": "2026-01-02",
        }
    ]

    mock_pg, mock_conn = _build_mock_conn(rows=fake_rows)

    req = _make_request(pg=mock_pg)
    result = await list_catalogs(req)

    assert len(result) == 1
    assert result[0].config == {"host": "localhost", "port": 5432}
    assert result[0].enabled is False
    assert result[0].test_status == "ok"


# ── update_catalog endpoint ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_catalog_invalid_type():
    from fastapi import HTTPException

    from dewie.api.routes.admin import UpdateCatalogRequest, update_catalog

    req = _make_request()
    src_id = uuid.uuid4()
    body = UpdateCatalogRequest(type="invalid")
    with pytest.raises(HTTPException) as exc_info:
        await update_catalog(src_id, body, req)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_update_catalog_not_found():
    from fastapi import HTTPException

    from dewie.api.routes.admin import UpdateCatalogRequest, update_catalog

    mock_pg, mock_conn = _build_mock_conn(rows=[], rowcount=0)

    req = _make_request(pg=mock_pg)
    src_id = uuid.uuid4()
    body = UpdateCatalogRequest(name="updated")
    with pytest.raises(HTTPException) as exc_info:
        await update_catalog(src_id, body, req)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_update_catalog_success():
    from dewie.api.routes.admin import UpdateCatalogRequest, update_catalog

    src_id = uuid.uuid4()
    fake_row = {
        "id": str(src_id),
        "name": "updated-name",
        "type": "sqlite",
        "config_json": '{"filepath": "/tmp/updated.db"}',
        "enabled": False,
        "created_by": None,
        "created_at": "2026-01-01",
        "tested_at": None,
        "test_status": None,
        "test_error": None,
        "updated_at": "2026-01-02",
    }

    mock_pg, mock_conn = _build_mock_conn(rows=[fake_row], fetchone_row=fake_row)

    req = _make_request(pg=mock_pg)
    body = UpdateCatalogRequest(name="updated-name", enabled=False)
    result = await update_catalog(src_id, body, req)

    assert result.name == "updated-name"
    assert result.enabled is False

    # Verify the UPDATE was called
    assert mock_conn.execute.call_count >= 1
    mock_conn.commit.assert_called()


@pytest.mark.asyncio
async def test_update_catalog_partial():
    """Test updating only one field (enabled) without changing others."""
    from dewie.api.routes.admin import UpdateCatalogRequest, update_catalog

    src_id = uuid.uuid4()
    fake_row = {
        "id": str(src_id),
        "name": "test-db",
        "type": "postgres",
        "config_json": '{"host": "localhost", "port": 5432}',
        "enabled": False,
        "created_by": None,
        "created_at": "2026-01-01",
        "tested_at": None,
        "test_status": None,
        "test_error": None,
        "updated_at": "2026-01-02",
    }

    mock_pg, mock_conn = _build_mock_conn(rows=[fake_row], fetchone_row=fake_row)

    req = _make_request(pg=mock_pg)
    body = UpdateCatalogRequest(enabled=False)
    result = await update_catalog(src_id, body, req)

    assert result.name == "test-db"
    assert result.type == "postgres"
    assert result.enabled is False


# ── delete_catalog endpoint ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_catalog_not_found():
    from fastapi import HTTPException

    from dewie.api.routes.admin import delete_catalog

    mock_pg, mock_conn = _build_mock_conn(rows=[], rowcount=0)

    req = _make_request(pg=mock_pg)
    src_id = uuid.uuid4()
    with pytest.raises(HTTPException) as exc_info:
        await delete_catalog(src_id, req)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_catalog_success():
    from dewie.api.routes.admin import delete_catalog

    mock_pg, mock_conn = _build_mock_conn(rows=[], rowcount=1)

    req = _make_request(pg=mock_pg)
    src_id = uuid.uuid4()
    result = await delete_catalog(src_id, req)
    assert result is None
    mock_conn.commit.assert_called()


# ── test_catalog endpoint ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_test_catalog_not_found():
    from fastapi import HTTPException

    from dewie.api.routes.admin import test_catalog

    mock_pg, mock_conn = _build_mock_conn(rows=[], fetchone_row=None)

    req = _make_request(pg=mock_pg)
    src_id = uuid.uuid4()
    with pytest.raises(HTTPException) as exc_info:
        await test_catalog(src_id, req)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_test_catalog_sqlite_missing_filepath():
    from dewie.api.routes.admin import test_catalog

    src_id = uuid.uuid4()
    fake_row = {
        "type": "sqlite",
        "config_json": "{}",
    }

    mock_pg, mock_conn = _build_mock_conn(rows=[fake_row], fetchone_row=fake_row)

    req = _make_request(pg=mock_pg)
    result = await test_catalog(src_id, req)

    assert result.ok is False
    assert result.error == "Missing filepath in config"


@pytest.mark.asyncio
async def test_test_catalog_sqlite_file_not_found():
    from dewie.api.routes.admin import test_catalog

    src_id = uuid.uuid4()
    fake_row = {
        "type": "sqlite",
        "config_json": '{"filepath": "/nonexistent/path/test.db"}',
    }

    mock_pg, mock_conn = _build_mock_conn(rows=[fake_row], fetchone_row=fake_row)

    req = _make_request(pg=mock_pg)
    result = await test_catalog(src_id, req)

    assert result.ok is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_test_catalog_sqlite_file_exists(tmp_path):
    from dewie.api.routes.admin import test_catalog

    src_id = uuid.uuid4()
    db_file = tmp_path / "test.db"
    db_file.write_text("test")

    fake_row = {
        "type": "sqlite",
        "config_json": f'{{"filepath": "{db_file}"}}',
    }

    mock_pg, mock_conn = _build_mock_conn(rows=[fake_row], fetchone_row=fake_row)

    req = _make_request(pg=mock_pg)
    result = await test_catalog(src_id, req)

    assert result.ok is True
    assert result.error is None


@pytest.mark.asyncio
async def test_test_catalog_postgres_missing_fields():
    from dewie.api.routes.admin import test_catalog

    src_id = uuid.uuid4()
    fake_row = {
        "type": "postgres",
        "config_json": '{"host": "localhost"}',
    }

    mock_pg, mock_conn = _build_mock_conn(rows=[fake_row], fetchone_row=fake_row)

    req = _make_request(pg=mock_pg)
    result = await test_catalog(src_id, req)

    assert result.ok is False
    assert "missing" in result.error.lower()


@pytest.mark.asyncio
async def test_test_catalog_mcp_missing_endpoint():
    from dewie.api.routes.admin import test_catalog

    src_id = uuid.uuid4()
    fake_row = {
        "type": "mcp",
        "config_json": '{"method": "sse"}',
    }

    mock_pg, mock_conn = _build_mock_conn(rows=[fake_row], fetchone_row=fake_row)

    req = _make_request(pg=mock_pg)
    result = await test_catalog(src_id, req)

    assert result.ok is False
    assert result.error == "Missing endpoint in config"


@pytest.mark.asyncio
async def test_test_catalog_mcp_has_endpoint():
    from dewie.api.routes.admin import test_catalog

    src_id = uuid.uuid4()
    fake_row = {
        "type": "mcp",
        "config_json": '{"endpoint": "http://localhost:3000/mcp"}',
    }

    mock_pg, mock_conn = _build_mock_conn(rows=[fake_row], fetchone_row=fake_row)

    import unittest.mock as _mock

    req = _make_request(pg=mock_pg)
    with _mock.patch("dewie.api.routes.admin._test_mcp_connection", new=AsyncMock(return_value=(True, None))):
        result = await test_catalog(src_id, req)

    assert result.ok is True
    assert result.error is None


@pytest.mark.asyncio
async def test_test_catalog_updates_test_status_in_db():
    """Verify that test results are persisted back to the database."""
    from dewie.api.routes.admin import test_catalog

    src_id = uuid.uuid4()
    fake_row = {
        "type": "mcp",
        "config_json": '{"endpoint": "http://localhost:3000/mcp"}',
    }

    mock_pg, mock_conn = _build_mock_conn(rows=[fake_row], fetchone_row=fake_row)

    import unittest.mock as _mock

    req = _make_request(pg=mock_pg)
    with _mock.patch("dewie.api.routes.admin._test_mcp_connection", new=AsyncMock(return_value=(True, None))):
        result = await test_catalog(src_id, req)

    assert result.ok is True

    # Verify the UPDATE was called to persist test result
    assert mock_conn.execute.call_count >= 2
    mock_conn.commit.assert_called()


@pytest.mark.asyncio
async def test_test_catalog_unknown_type():
    from dewie.api.routes.admin import test_catalog

    src_id = uuid.uuid4()
    fake_row = {
        "type": "unknown_type",
        "config_json": "{}",
    }

    mock_pg, mock_conn = _build_mock_conn(rows=[fake_row], fetchone_row=fake_row)

    req = _make_request(pg=mock_pg)
    result = await test_catalog(src_id, req)

    assert result.ok is False
    assert "unknown" in result.error.lower()


# ── Validation ─────────────────────────────────────────────────────────────────


def test_valid_source_types():
    from dewie.api.routes.admin import _VALID_SOURCE_TYPES

    assert "sqlite" in _VALID_SOURCE_TYPES
    assert "postgres" in _VALID_SOURCE_TYPES
    assert "mcp" in _VALID_SOURCE_TYPES
    assert "mysql" not in _VALID_SOURCE_TYPES
    assert "mongodb" not in _VALID_SOURCE_TYPES


@pytest.mark.asyncio
async def test_test_catalog_postgres_dsn_only_supported():
    from dewie.api.routes.admin import test_catalog

    src_id = uuid.uuid4()
    fake_row = {
        "type": "postgres",
        "config_json": '{"dsn": "postgresql+asyncpg://user:pass@localhost:5432/db"}',
    }

    mock_pg, mock_conn = _build_mock_conn(rows=[fake_row], fetchone_row=fake_row)

    import unittest.mock as _mock

    req = _make_request(pg=mock_pg)
    with _mock.patch(
        "dewie.api.routes.admin._test_postgres_connection",
        new=AsyncMock(return_value=(True, None)),
    ):
        result = await test_catalog(src_id, req)

    assert result.ok is True
    assert result.error is None
