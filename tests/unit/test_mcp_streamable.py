"""Tests for dewie.api.mcp_streamable — in-process MCP Streamable HTTP transport.

Drives the real mcp_app.streamable_http_app() ASGI app through a real MCP
initialize -> tools/list -> tools/call handshake via httpx.ASGITransport, the
same in-memory-ASGI pattern test_mcp_routes.py already uses for the REST
endpoint. A tiny middleware stands in for Dewie's real _api_key_middleware,
injecting request.state the same way it would in production.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.middleware.base import BaseHTTPMiddleware

from dewie.api import mcp_shared_state
from dewie.api.mcp_streamable import mcp_app

EXPECTED_TOOL_NAMES = {
    "search_corpus",
    "ingest_url",
    "dewie_ingest",
    "expand",
    "read",
    "intersect",
    "bridge",
    "browse",
    "research",
    "web_search",
    "add_catalog",
    "list_sources",
    "list_catalogs",
    "dewie_fetch",
}


class _InjectAuthState(BaseHTTPMiddleware):
    """Stands in for _api_key_middleware: sets request.state before the request
    reaches the mounted FastMCP sub-app, mirroring how Mount preserves
    scope["state"] (but not scope["app"]) across the boundary in production."""

    def __init__(self, app, *, user_id="user123", workspace_ids=None, is_admin=False, key_id=None):
        super().__init__(app)
        self._user_id = user_id
        self._workspace_ids = workspace_ids if workspace_ids is not None else [
            uuid.UUID("00000000-0000-0000-0000-000000000010")
        ]
        self._is_admin = is_admin
        self._key_id = key_id

    async def dispatch(self, request, call_next):
        request.state.user_id = self._user_id
        request.state.workspace_ids = self._workspace_ids
        request.state.is_admin = self._is_admin
        request.state.key_id = self._key_id
        return await call_next(request)


def _build_app(**auth_kwargs):
    app = mcp_app.streamable_http_app()
    app.add_middleware(_InjectAuthState, **auth_kwargs)
    return app


@asynccontextmanager
async def _client_session(app, *, base_url="http://localhost:8000"):
    """Open a real MCP ClientSession against the ASGI app in-memory."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    url = base_url + "/mcp"

    async with mcp_app.session_manager.run():
        http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)
        async with http_client:
            async with streamable_http_client(url, http_client=http_client) as (
                read_stream,
                write_stream,
                _get_session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session


@pytest.fixture(autouse=True)
def _reset_session_manager():
    """mcp_app is a module-level singleton; its StreamableHTTPSessionManager
    can only .run() once per instance ever, so give each test a fresh one."""
    mcp_app._session_manager = None
    yield
    mcp_app._session_manager = None


@pytest.fixture(autouse=True)
def _configure_shared_state():
    pg = AsyncMock()
    mcp_shared_state.configure(pg, None)
    yield pg


@pytest.mark.asyncio
async def test_tools_list_returns_exactly_14_tools():
    app = _build_app()
    async with _client_session(app) as session:
        result = await session.list_tools()
    names = {t.name for t in result.tools}
    assert names == EXPECTED_TOOL_NAMES


@pytest.mark.asyncio
async def test_tools_call_search_corpus_dispatches(monkeypatch):
    app = _build_app()

    captured = {}

    async def _fake_dispatch(tool_name, ctx, input_data):
        captured["tool_name"] = tool_name
        captured["input_data"] = input_data
        return {"results": [], "count": 0}

    monkeypatch.setattr("dewie.api.mcp_streamable._dispatch", _fake_dispatch)

    async with _client_session(app) as session:
        result = await session.call_tool("search_corpus", {"query": "hello"})

    assert not result.isError
    assert captured["tool_name"] == "search_corpus"
    assert captured["input_data"]["query"] == "hello"


@pytest.mark.asyncio
async def test_tools_call_each_tool_dispatches_with_matching_name(monkeypatch):
    """Every registered tool calls dispatch_mcp_tool with its own name."""
    app = _build_app(is_admin=True)

    captured_calls = []

    async def _fake_dispatch_tool(tool_name, input_data, **kwargs):
        captured_calls.append(tool_name)
        return {"ok": True}

    monkeypatch.setattr("dewie.api.mcp_dispatch.dispatch_mcp_tool", _fake_dispatch_tool)
    # mcp_streamable imported dispatch_mcp_tool by reference at module load time;
    # patch the name it actually calls.
    monkeypatch.setattr("dewie.api.mcp_streamable.dispatch_mcp_tool", _fake_dispatch_tool)

    tool_args = {
        "search_corpus": {"query": "q"},
        "ingest_url": {"url": "http://x"},
        "dewie_ingest": {"url": "http://x"},
        "expand": {"doc_id": "d1"},
        "read": {"doc_id": "d1"},
        "intersect": {"doc_ids": ["a", "b"]},
        "bridge": {"source_id": "a", "target_id": "b"},
        "browse": {"query": "q"},
        "research": {"query": "q"},
        "web_search": {"query": "q"},
        "add_catalog": {"name": "n", "type": "mcp"},
        "list_sources": {},
        "list_catalogs": {},
        "dewie_fetch": {"url": "http://x"},
    }

    async with _client_session(app) as session:
        for name, args in tool_args.items():
            result = await session.call_tool(name, args)
            assert not result.isError, f"{name} returned an error: {result}"

    assert set(captured_calls) == EXPECTED_TOOL_NAMES


@pytest.mark.asyncio
async def test_unauthenticated_request_surfaces_as_mcp_error(monkeypatch):
    """No user_id/workspace_ids + auth_enabled=True -> MCP isError, not a raw 500."""
    from dewie.config import settings

    monkeypatch.setattr(settings, "auth_enabled", True)

    app = mcp_app.streamable_http_app()
    app.add_middleware(_InjectAuthState, user_id=None, workspace_ids=[])

    async with _client_session(app) as session:
        result = await session.call_tool("search_corpus", {"query": "hello"})

    assert result.isError
