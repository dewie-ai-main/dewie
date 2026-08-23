"""Tests for #213 — admin panel service status indicators.

Verifies:
  1. The admin.html static file contains the Status tab and panel.
  2. The JS loadServiceStatus function is present.
  3. The /service-status API endpoint returns the expected shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_STATIC = Path(__file__).parents[2] / "static"


# ── admin.html structure ──────────────────────────────────────────────────────


def test_admin_html_has_status_tab():
    """admin.html should have a sidebar tab button for Status."""
    content = (_STATIC / "admin.html").read_text()
    assert 'data-tab="status"' in content, "Missing data-tab=status sidebar button"


def test_admin_html_has_status_panel():
    """admin.html should have the panel-status section."""
    content = (_STATIC / "admin.html").read_text()
    assert 'id="panel-status"' in content, "Missing id=panel-status panel"


def test_admin_html_has_service_grid():
    """admin.html should render a service status grid."""
    content = (_STATIC / "admin.html").read_text()
    assert 'id="status-grid"' in content, "Missing status-grid element"


def test_admin_html_has_overall_indicator():
    """admin.html should show an overall status dot."""
    content = (_STATIC / "admin.html").read_text()
    assert 'id="status-overall-dot"' in content, "Missing status-overall-dot element"


def test_admin_html_has_load_service_status_fn():
    """admin.html should define the loadServiceStatus JS function."""
    content = (_STATIC / "admin.html").read_text()
    assert "loadServiceStatus" in content, "Missing loadServiceStatus JS function"


def test_admin_html_has_status_dot_css():
    """admin.html should define status-dot CSS classes with green/red/amber."""
    content = (_STATIC / "admin.html").read_text()
    assert ".status-dot.ok" in content, "Missing .status-dot.ok CSS"
    assert ".status-dot.error" in content, "Missing .status-dot.error CSS"
    assert ".status-dot.degraded" in content, "Missing .status-dot.degraded CSS"


def test_admin_html_fetches_service_status_endpoint():
    """admin.html should call /service-status endpoint."""
    content = (_STATIC / "admin.html").read_text()
    assert "/service-status" in content, "admin.html does not reference /service-status"


# ── /service-status API ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_service_status_covers_required_services():
    """service-status must include mcp, database, api, and web_panel keys."""
    from dewie.api.routes.service_status import get_service_status

    req = MagicMock()
    req.app = MagicMock()

    static_mount = MagicMock()
    static_mount.name = "static"
    mcp_route = MagicMock()
    mcp_route.path = "/mcp/tools"
    req.app.routes = [static_mount, mcp_route]

    ok = {"status": "ok", "message": "OK"}
    with (
        patch("dewie.api.routes.service_status._check_database", AsyncMock(return_value=ok)),
        patch("dewie.api.routes.service_status._check_cache",    AsyncMock(return_value=ok)),
        patch("dewie.api.routes.service_status._check_mcp",      AsyncMock(return_value=ok)),
    ):
        response = await get_service_status(req)

    body = json.loads(response.body)
    required = {"mcp", "database", "api", "web_panel"}
    assert required.issubset(set(body["services"].keys())), (
        f"Missing required service keys: {required - set(body['services'].keys())}"
    )


@pytest.mark.asyncio
async def test_service_status_each_has_status_and_message():
    """Every service entry must have 'status' and 'message' fields."""
    from dewie.api.routes.service_status import get_service_status

    req = MagicMock()
    req.app = MagicMock()
    req.app.routes = []

    ok = {"status": "ok", "message": "OK"}
    with (
        patch("dewie.api.routes.service_status._check_database", AsyncMock(return_value=ok)),
        patch("dewie.api.routes.service_status._check_cache",    AsyncMock(return_value=ok)),
        patch("dewie.api.routes.service_status._check_mcp",      AsyncMock(return_value=ok)),
    ):
        response = await get_service_status(req)

    body = json.loads(response.body)
    for name, svc in body["services"].items():
        assert "status" in svc, f"Service '{name}' missing 'status'"
        assert "message" in svc, f"Service '{name}' missing 'message'"
        assert svc["status"] in ("ok", "degraded", "error"), (
            f"Service '{name}' has unexpected status: {svc['status']}"
        )
