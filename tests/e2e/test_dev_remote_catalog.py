"""Optional dev-only E2E tests for remote catalog lifecycle.

These tests are excluded from default CI runs.

Run them explicitly with:
    PYTHONPATH=src pytest -m dev_remote_catalog -o addopts='' -v

Prerequisites — create .env.remote-catalog.local in the repo root:

    DEWIE_TEST_API_BASE=http://localhost:8000/api
    DEWIE_TEST_ADMIN_KEY=your-admin-key
    DEWIE_REMOTE_CATALOG_ENDPOINT=http://remote-dewie-node:8000/api
    DEWIE_REMOTE_CATALOG_API_KEY=ck_live_...

The suite skips cleanly if any required key is missing.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.dev_remote_catalog

# ── App factory (mirrors admin route tests) ───────────────────────────────────

_ADMIN_KEY = "test-admin-key-dev"
_TEST_USER_ID = "00000000-0000-0000-0000-000000000099"
_TEST_TENANT_ID = "00000000-0000-0000-0000-000000000001"


def _make_admin_app(pg: AsyncMock | None = None) -> FastAPI:
    """Minimal FastAPI app with admin router and a bypassed admin-auth fixture."""
    from dewie.api.routes.admin import router

    pg = pg or _make_pg()

    app = FastAPI()
    app.state.postgres = pg

    async def _inject_admin(request, call_next):
        request.state.user_id = _TEST_USER_ID
        request.state.is_admin = True
        request.state.tenant_id = _TEST_TENANT_ID
        request.state.key_scopes = ["read", "write"]
        return await call_next(request)

    app.middleware("http")(_inject_admin)
    app.include_router(router)
    return app


_SAMPLE_SOURCE = {
    "id": uuid.UUID("00000000-0000-0000-0000-000000000099"),
    "name": "Remote Dewie",
    "type": "mcp",
    "config": {"endpoint": "http://remote:8000/api", "api_key": "ck_live_test"},
    "enabled": True,
    "created_by": _TEST_USER_ID,
    "created_at": "2026-06-14T00:00:00",
    "updated_at": "2026-06-14T00:00:00",
    "tested_at": None,
    "test_status": None,
    "test_error": None,
}


def _make_pg(sources: list | None = None) -> AsyncMock:
    pg = AsyncMock()
    pg.list_sources = AsyncMock(return_value=sources or [])
    pg.get_source = AsyncMock(return_value=_SAMPLE_SOURCE)
    pg.create_source = AsyncMock(return_value=_SAMPLE_SOURCE)
    pg.update_source = AsyncMock(
        return_value={
            **_SAMPLE_SOURCE,
            "name": "Remote Dewie Updated",
            "enabled": False,
            "updated_at": "2026-06-14T01:00:00",
        }
    )
    # Truthy return (the deleted record) → route returns 204
    pg.delete_source = AsyncMock(return_value=_SAMPLE_SOURCE)
    pg._engine = MagicMock()
    return pg


# ── Admin: CREATE catalog ─────────────────────────────────────────────────────


def test_create_mcp_catalog_returns_201():
    app = _make_admin_app()
    client = TestClient(app)

    resp = client.post(
        "/admin/catalogs",
        json={
            "name": "Remote Dewie",
            "type": "mcp",
            "config": {"endpoint": "http://remote:8000/api", "api_key": "ck_live_test"},
            "enabled": True,
        },
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["type"] == "mcp"
    assert body["name"] == "Remote Dewie"
    assert body["enabled"] is True
    assert "id" in body


def test_create_catalog_invalid_type_returns_400():
    app = _make_admin_app()
    client = TestClient(app)

    resp = client.post(
        "/admin/catalogs",
        json={"name": "Bad Catalog", "type": "neo4j", "config": {}},
    )

    assert resp.status_code == 400
    assert "neo4j" in resp.json()["detail"]


def test_create_catalog_duplicate_name_returns_409():
    pg = _make_pg()
    pg.create_source = AsyncMock(side_effect=Exception("unique constraint failed"))
    app = _make_admin_app(pg)
    client = TestClient(app)

    resp = client.post(
        "/admin/catalogs",
        json={"name": "Remote Dewie", "type": "mcp", "config": {}},
    )

    assert resp.status_code == 409


# ── Admin: LIST catalogs ──────────────────────────────────────────────────────


def test_list_catalogs_returns_existing():
    existing = [
        {
            "id": uuid.uuid4(),
            "name": "Remote A",
            "type": "mcp",
            "config": {"endpoint": "http://a:8000/api"},
            "enabled": True,
            "created_by": None,
            "created_at": "2026-06-14T00:00:00",
            "updated_at": "2026-06-14T00:00:00",
            "tested_at": None,
            "test_status": None,
            "test_error": None,
        }
    ]
    app = _make_admin_app(_make_pg(sources=existing))
    client = TestClient(app)

    resp = client.get("/admin/catalogs")

    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    assert items[0]["name"] == "Remote A"


def test_list_catalogs_empty_returns_empty_list():
    app = _make_admin_app(_make_pg(sources=[]))
    client = TestClient(app)

    resp = client.get("/admin/catalogs")

    assert resp.status_code == 200
    assert resp.json() == []


# ── Admin: UPDATE catalog ─────────────────────────────────────────────────────


def test_update_catalog_disable():
    source_id = uuid.uuid4()
    pg = _make_pg()

    app = _make_admin_app(pg)
    client = TestClient(app)

    resp = client.patch(f"/admin/catalogs/{source_id}", json={"enabled": False})

    assert resp.status_code == 200
    assert resp.json()["enabled"] is False


def test_update_catalog_invalid_type_returns_400():
    source_id = uuid.uuid4()
    app = _make_admin_app()
    client = TestClient(app)

    resp = client.patch(f"/admin/catalogs/{source_id}", json={"type": "neo4j"})

    assert resp.status_code == 400


# ── Admin: TEST catalog connection ────────────────────────────────────────────


def test_test_mcp_catalog_missing_endpoint_returns_ok_false():
    source_id = uuid.uuid4()
    pg = _make_pg()
    # get_source returns a catalog with empty config → no endpoint
    pg.get_source = AsyncMock(
        return_value={
            **_SAMPLE_SOURCE,
            "id": source_id,
            "name": "Broken",
            "config": {},
        }
    )
    pg.update_source = AsyncMock(return_value=pg.get_source.return_value)

    app = _make_admin_app(pg)
    client = TestClient(app)

    resp = client.post(f"/admin/catalogs/{source_id}/test")

    assert resp.status_code == 200
    assert resp.json()["ok"] is False
    assert resp.json()["error"] is not None


def test_test_mcp_catalog_with_endpoint_probes_connection():
    source_id = uuid.uuid4()
    pg = _make_pg()
    pg.get_source = AsyncMock(
        return_value={
            **_SAMPLE_SOURCE,
            "id": source_id,
            "config": {"endpoint": "http://remote:8000/api", "api_key": "ck_live_test"},
        }
    )
    pg.update_source = AsyncMock(return_value=pg.get_source.return_value)

    app = _make_admin_app(pg)
    client = TestClient(app)

    # Patch the actual connection helper so no live network call is needed
    with patch(
        "dewie.api.routes.admin._test_mcp_connection",
        AsyncMock(return_value=(True, None)),
    ):
        resp = client.post(f"/admin/catalogs/{source_id}/test")

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert resp.json()["error"] is None


# ── Admin: DELETE catalog ─────────────────────────────────────────────────────


def test_delete_catalog_returns_204():
    source_id = uuid.uuid4()
    app = _make_admin_app()
    client = TestClient(app)

    resp = client.delete(f"/admin/catalogs/{source_id}")

    assert resp.status_code == 204


def test_delete_nonexistent_catalog_returns_404():
    source_id = uuid.uuid4()
    pg = _make_pg()
    # Route returns 404 when delete_source returns falsy (None / empty)
    pg.delete_source = AsyncMock(return_value=None)

    app = _make_admin_app(pg)
    client = TestClient(app)

    resp = client.delete(f"/admin/catalogs/{source_id}")

    assert resp.status_code == 404


# ── Response field contract ───────────────────────────────────────────────────


def test_catalog_response_shape_has_all_required_fields():
    app = _make_admin_app()
    client = TestClient(app)

    resp = client.post(
        "/admin/catalogs",
        json={"name": "Shape Check", "type": "mcp", "config": {"endpoint": "http://x"}},
    )

    assert resp.status_code == 201
    body = resp.json()
    for field in ("id", "name", "type", "config", "enabled", "created_at", "updated_at"):
        assert field in body, f"Missing field: {field}"


def test_catalog_response_does_not_leak_api_key_in_top_level():
    """Config sub-object may hold api_key, but it must not appear at top level."""
    app = _make_admin_app()
    client = TestClient(app)

    resp = client.post(
        "/admin/catalogs",
        json={
            "name": "Key Leak Check",
            "type": "mcp",
            "config": {"endpoint": "http://x", "api_key": "ck_live_secret"},
        },
    )

    assert resp.status_code == 201
    body = resp.json()
    assert "api_key" not in body  # must not be at top-level
    assert "ck_live_secret" not in str(body.get("name", ""))


# ── MCP tool: add_catalog ─────────────────────────────────────────────────────


def _make_mcp_app(pg: AsyncMock | None = None) -> FastAPI:
    from dewie.api.middleware_base import limiter
    from dewie.api.routes.mcp import router as mcp_router

    pg = pg or _make_pg()

    app = FastAPI()
    app.state.limiter = limiter
    app.state.postgres = pg
    app.state.processor = None

    async def _inject_admin(request, call_next):
        request.state.user_id = _TEST_USER_ID
        request.state.is_admin = True
        request.state.workspace_ids = []
        request.state.key_id = None
        request.state.key_scopes = ["read", "write"]
        return await call_next(request)

    app.middleware("http")(_inject_admin)
    app.include_router(mcp_router)
    return app


def test_mcp_manifest_lists_add_catalog_tool():
    """GET /mcp manifest should advertise the add_catalog tool."""
    app = _make_mcp_app()
    client = TestClient(app)

    resp = client.get("/mcp")

    assert resp.status_code == 200
    tools = {t["name"] for t in resp.json()["tools"]}
    assert "add_catalog" in tools


def test_mcp_add_catalog_creates_source():
    """POST /mcp add_catalog dispatches to admin catalog creation."""
    pg = _make_pg()
    app = _make_mcp_app(pg)
    client = TestClient(app)

    resp = client.post(
        "/mcp",
        json={
            "tool": "add_catalog",
            "input": {
                "name": "Via MCP",
                "type": "mcp",
                "endpoint": "http://remote:8000/api",
                "api_key": "ck_live_test",
            },
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["tool"] == "add_catalog"
    assert body["content"]["ok"] is True
    assert "id" in body["content"]


def test_mcp_add_catalog_invalid_type_returns_422():
    app = _make_mcp_app()
    client = TestClient(app)

    resp = client.post(
        "/mcp",
        json={
            "tool": "add_catalog",
            "input": {"name": "Bad", "type": "neo4j", "endpoint": "http://x"},
        },
    )

    assert resp.status_code == 422
    assert "neo4j" in resp.json()["detail"].lower()


def test_mcp_add_catalog_missing_name_returns_422():
    app = _make_mcp_app()
    client = TestClient(app)

    resp = client.post(
        "/mcp",
        json={
            "tool": "add_catalog",
            "input": {"type": "mcp", "endpoint": "http://x"},
        },
    )

    assert resp.status_code == 422


def test_mcp_add_catalog_non_admin_returns_403():
    """add_catalog tool must be restricted to admin users."""
    from dewie.api.middleware_base import limiter
    from dewie.api.routes.mcp import router as mcp_router

    app = FastAPI()
    app.state.limiter = limiter
    app.state.postgres = _make_pg()
    app.state.processor = None

    async def _inject_non_admin(request, call_next):
        request.state.user_id = _TEST_USER_ID
        request.state.is_admin = False
        request.state.workspace_ids = []
        request.state.key_id = None
        request.state.key_scopes = ["read"]
        return await call_next(request)

    app.middleware("http")(_inject_non_admin)
    app.include_router(mcp_router)
    client = TestClient(app)

    resp = client.post(
        "/mcp",
        json={
            "tool": "add_catalog",
            "input": {
                "name": "Hack",
                "type": "mcp",
                "endpoint": "http://remote:8000/api",
            },
        },
    )

    assert resp.status_code == 403
