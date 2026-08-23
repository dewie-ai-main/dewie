"""Tests for dewie.api.routes.service_status — service status indicators."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── _check_database ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_database_ok():
    from dewie.api.routes.service_status import _check_database

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=MagicMock())
    begin_cm = AsyncMock()
    begin_cm.__aenter__ = AsyncMock(return_value=conn)
    begin_cm.__aexit__ = AsyncMock(return_value=False)

    pg = MagicMock()
    pg._engine.begin.return_value = begin_cm

    app = MagicMock()
    app.state.postgres = pg

    result = await _check_database(app)

    assert result["status"] == "ok"
    assert "connected" in result["message"].lower()


@pytest.mark.asyncio
async def test_check_database_failure():
    from dewie.api.routes.service_status import _check_database

    pg = MagicMock()
    pg._engine.begin.side_effect = Exception("connection refused")

    app = MagicMock()
    app.state.postgres = pg

    result = await _check_database(app)

    assert result["status"] == "error"
    assert "connection refused" in result["message"]


# ── _check_cache ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_cache_ok():
    from dewie.api.routes.service_status import _check_cache

    cache = MagicMock()
    cache._redis.ping = AsyncMock(return_value=True)

    app = MagicMock()
    app.state.cache = cache

    result = await _check_cache(app)

    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_check_cache_failure():
    from dewie.api.routes.service_status import _check_cache

    cache = MagicMock()
    cache._redis.ping = AsyncMock(side_effect=Exception("redis down"))

    app = MagicMock()
    app.state.cache = cache

    result = await _check_cache(app)

    assert result["status"] == "degraded"
    assert "redis down" in result["message"]


# ── _check_mcp ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_mcp_routes_present():
    """MCP check returns ok when /mcp routes are mounted."""
    from dewie.api.routes.service_status import _check_mcp

    route1 = MagicMock()
    route1.path = "/mcp/tools"
    route2 = MagicMock()
    route2.path = "/mcp/call"

    app = MagicMock()
    app.routes = [route1, route2]

    result = await _check_mcp(app)
    assert result["status"] == "ok"
    assert "2" in result["message"]


@pytest.mark.asyncio
async def test_check_mcp_no_routes():
    """MCP check returns degraded when no /mcp routes found."""
    from dewie.api.routes.service_status import _check_mcp

    route1 = MagicMock()
    route1.path = "/health"

    app = MagicMock()
    app.routes = [route1]

    result = await _check_mcp(app)
    assert result["status"] == "degraded"


# ── get_service_status (integration-style) ────────────────────────────────────


@pytest.mark.asyncio
async def test_get_service_status_all_ok():
    from dewie.api.routes.service_status import get_service_status

    req = MagicMock()
    req.app = MagicMock()

    static_mount = MagicMock()
    static_mount.name = "static"

    mcp_route = MagicMock()
    mcp_route.path = "/mcp/tools"

    req.app.routes = [static_mount, mcp_route]

    ok_result = {"status": "ok", "message": "OK"}

    with (
        patch("dewie.api.routes.service_status._check_database", AsyncMock(return_value=ok_result)),
        patch("dewie.api.routes.service_status._check_cache",    AsyncMock(return_value=ok_result)),
        patch("dewie.api.routes.service_status._check_mcp",      AsyncMock(return_value=ok_result)),
        patch("dewie.api.routes.service_status._check_ingestion",AsyncMock(return_value=ok_result)),
    ):
        response = await get_service_status(req)

    import json
    body = json.loads(response.body)

    assert body["overall"] == "ok"
    assert body["services"]["api"]["status"] == "ok"
    assert body["services"]["database"]["status"] == "ok"
    assert body["services"]["mcp"]["status"] == "ok"
    assert body["services"]["web_panel"]["status"] == "ok"
    assert body["services"]["ingestion"]["status"] == "ok"
    assert "checked_at" in body


@pytest.mark.asyncio
async def test_get_service_status_db_error():
    from dewie.api.routes.service_status import get_service_status

    req = MagicMock()
    req.app = MagicMock()

    static_mount = MagicMock()
    static_mount.name = "static"
    mcp_route = MagicMock()
    mcp_route.path = "/mcp/tools"
    req.app.routes = [static_mount, mcp_route]

    ok_result   = {"status": "ok",    "message": "OK"}
    fail_result = {"status": "error", "message": "connection refused"}

    with (
        patch("dewie.api.routes.service_status._check_database", AsyncMock(return_value=fail_result)),
        patch("dewie.api.routes.service_status._check_cache",    AsyncMock(return_value=ok_result)),
        patch("dewie.api.routes.service_status._check_mcp",      AsyncMock(return_value=ok_result)),
        patch("dewie.api.routes.service_status._check_ingestion",AsyncMock(return_value=ok_result)),
    ):
        response = await get_service_status(req)

    import json
    body = json.loads(response.body)

    assert body["overall"] == "error"
    assert body["services"]["database"]["status"] == "error"


@pytest.mark.asyncio
async def test_get_service_status_cache_degraded():
    from dewie.api.routes.service_status import get_service_status

    req = MagicMock()
    req.app = MagicMock()

    static_mount = MagicMock()
    static_mount.name = "static"
    mcp_route = MagicMock()
    mcp_route.path = "/mcp/tools"
    req.app.routes = [static_mount, mcp_route]

    ok_result       = {"status": "ok",       "message": "OK"}
    degraded_result = {"status": "degraded", "message": "redis down"}

    with (
        patch("dewie.api.routes.service_status._check_database", AsyncMock(return_value=ok_result)),
        patch("dewie.api.routes.service_status._check_cache",    AsyncMock(return_value=degraded_result)),
        patch("dewie.api.routes.service_status._check_mcp",      AsyncMock(return_value=ok_result)),
        patch("dewie.api.routes.service_status._check_ingestion",AsyncMock(return_value=ok_result)),
    ):
        response = await get_service_status(req)

    import json
    body = json.loads(response.body)

    assert body["overall"] == "degraded"


@pytest.mark.asyncio
async def test_get_service_status_web_panel_no_static():
    """Web panel is degraded when no static mount is found."""
    from dewie.api.routes.service_status import get_service_status

    req = MagicMock()
    req.app = MagicMock()

    mcp_route = MagicMock()
    mcp_route.path = "/mcp/tools"
    # No static mount
    req.app.routes = [mcp_route]

    ok_result = {"status": "ok", "message": "OK"}

    with (
        patch("dewie.api.routes.service_status._check_database", AsyncMock(return_value=ok_result)),
        patch("dewie.api.routes.service_status._check_cache",    AsyncMock(return_value=ok_result)),
        patch("dewie.api.routes.service_status._check_mcp",      AsyncMock(return_value=ok_result)),
        patch("dewie.api.routes.service_status._check_ingestion",AsyncMock(return_value=ok_result)),
    ):
        response = await get_service_status(req)

    import json
    body = json.loads(response.body)

    assert body["services"]["web_panel"]["status"] == "degraded"
    assert body["overall"] == "degraded"


@pytest.mark.asyncio
async def test_get_service_status_response_shape():
    """Response always has overall, checked_at, services with expected keys."""
    from dewie.api.routes.service_status import get_service_status

    req = MagicMock()
    req.app = MagicMock()
    req.app.routes = []

    any_result = {"status": "ok", "message": "x"}

    with (
        patch("dewie.api.routes.service_status._check_database", AsyncMock(return_value=any_result)),
        patch("dewie.api.routes.service_status._check_cache",    AsyncMock(return_value=any_result)),
        patch("dewie.api.routes.service_status._check_mcp",      AsyncMock(return_value=any_result)),
    ):
        response = await get_service_status(req)

    import json
    body = json.loads(response.body)

    assert set(body.keys()) >= {"overall", "checked_at", "services"}
    expected_services = {"api", "database", "cache", "mcp", "web_panel", "ingestion"}
    assert set(body["services"].keys()) == expected_services
