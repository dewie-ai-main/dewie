"""
tests/test_mcp_server.py — MCP server tests.

Unit tests: tool registration, schema validation, error handling (mocked HTTP).
Integration test: live round-trip against running API (skipped if API is down).
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("mcp")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_call_request(name: str, arguments: dict):
    """Build a minimal CallToolRequest-like object."""
    from mcp.types import CallToolRequest, CallToolRequestParams

    return CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name=name, arguments=arguments),
    )


def _make_list_request():
    from mcp.types import ListToolsRequest

    return ListToolsRequest(method="tools/list")


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------


def test_auth_headers_with_valid_token():
    """_auth_headers returns Bearer token for a valid token."""
    from dewie.mcp_server import _auth_headers

    result = _auth_headers("test-token")
    assert result == {"Authorization": "Bearer test-token"}


def test_auth_headers_with_empty_token():
    """_auth_headers raises ValueError for empty token."""
    from dewie.mcp_server import _auth_headers

    with pytest.raises(ValueError, match="Authentication token is required"):
        _auth_headers("")


def test_auth_headers_with_none_token():
    """_auth_headers raises ValueError for None token."""
    from dewie.mcp_server import _auth_headers

    with pytest.raises(ValueError, match="Authentication token is required"):
        _auth_headers(None)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_tools_registered():
    """Server exposes all registered tools."""
    from dewie.mcp_server import TOOLS

    names = {t.name for t in TOOLS}
    assert names == {"dewie_search", "dewie_expand", "dewie_read", "dewie_intersect", "dewie_bridge", "dewie_browse", "dewie_research"}


def test_tool_schemas_have_required_fields():
    """Each tool schema has the required fields documented."""
    from dewie.mcp_server import TOOLS

    tool_map = {t.name: t for t in TOOLS}

    search = tool_map["dewie_search"]
    assert "query" in search.inputSchema["required"]
    assert "query" in search.inputSchema["properties"]
    assert "limit" in search.inputSchema["properties"]

    expand = tool_map["dewie_expand"]
    assert "doc_id" in expand.inputSchema["required"]

    read = tool_map["dewie_read"]
    assert "doc_id" in read.inputSchema["required"]


# ---------------------------------------------------------------------------
# Unit: happy path (mocked HTTP)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_results():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [{"doc_id": "abc", "title": "Test Doc", "summary": "Test summary"}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("dewie.mcp_server.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(return_value=mock_response)

        from dewie.mcp_server import _search

        result = await _search("test query", limit=3, token="test-token")
        data = json.loads(result)
        assert len(data["results"]) == 1
        assert data["results"][0]["doc_id"] == "abc"


@pytest.mark.asyncio
async def test_expand_returns_neighbors():
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"doc_id": "n1", "title": "Neighbor 1"},
        {"doc_id": "n2", "title": "Neighbor 2"},
    ]
    mock_response.raise_for_status = MagicMock()

    with patch("dewie.mcp_server.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.get = AsyncMock(return_value=mock_response)

        from dewie.mcp_server import _expand

        result = await _expand("abc-123", token="test-token")
        data = json.loads(result)
        assert len(data) == 2
        assert data[0]["doc_id"] == "n1"


@pytest.mark.asyncio
async def test_read_returns_content():
    mock_response = MagicMock()
    mock_response.text = "Full article text here."
    mock_response.raise_for_status = MagicMock()

    with patch("dewie.mcp_server.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.get = AsyncMock(return_value=mock_response)

        from dewie.mcp_server import _read

        result = await _read("abc-123", token="test-token")
        assert result == "Full article text here."


@pytest.mark.asyncio
async def test_read_truncates_at_8k():
    mock_response = MagicMock()
    mock_response.text = "x" * 20_000
    mock_response.raise_for_status = MagicMock()

    with patch("dewie.mcp_server.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.get = AsyncMock(return_value=mock_response)

        from dewie.mcp_server import _read

        result = await _read("abc-123", token="test-token")
        assert len(result) == 8000


# ---------------------------------------------------------------------------
# Unit: intersect / bridge / browse / research (mocked HTTP)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intersect_returns_docs():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "docs": [
            {"doc_id": "x1", "title": "Overlap Doc 1"},
            {"doc_id": "x2", "title": "Overlap Doc 2"},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("dewie.mcp_server.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(return_value=mock_response)

        from dewie.mcp_server import _intersect

        result = await _intersect(["abc", "def"], token="test-token")
        data = json.loads(result)
        assert len(data) == 2
        assert data[0]["doc_id"] == "x1"


@pytest.mark.asyncio
async def test_intersect_respects_limit():
    many_docs = [{"doc_id": f"d{i}", "title": f"Doc {i}"} for i in range(20)]
    mock_response = MagicMock()
    mock_response.json.return_value = {"docs": many_docs}
    mock_response.raise_for_status = MagicMock()

    with patch("dewie.mcp_server.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(return_value=mock_response)

        from dewie.mcp_server import _intersect

        result = await _intersect(["a", "b"], limit=5, token="test-token")
        data = json.loads(result)
        assert len(data) == 5


@pytest.mark.asyncio
async def test_bridge_returns_path():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "path": ["src-id", "mid-id", "tgt-id"],
        "hops": 2,
    }
    mock_response.raise_for_status = MagicMock()

    with patch("dewie.mcp_server.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(return_value=mock_response)

        from dewie.mcp_server import _bridge

        result = await _bridge("src-id", "tgt-id", max_depth=4, token="test-token")
        data = json.loads(result)
        assert "path" in data
        assert len(data["path"]) == 3


@pytest.mark.asyncio
async def test_browse_returns_formatted_list():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {
                "doc_id": "b1",
                "title": "Browse Result",
                "source": "web",
                "published_at": "2024-01-15T00:00:00Z",
                "summary": "A summary of the browse result.",
                "score": 0.88,
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("dewie.mcp_server.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(return_value=mock_response)

        from dewie.mcp_server import _browse

        result = await _browse("machine learning", limit=5, token="test-token")
        assert "Browse Result" in result
        assert "b1" in result
        assert "1 articles found" in result


@pytest.mark.asyncio
async def test_browse_caps_limit_at_15():
    """_browse enforces max limit of 15 regardless of caller request."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status = MagicMock()

    with patch("dewie.mcp_server.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(return_value=mock_response)

        from dewie.mcp_server import _browse

        await _browse("test", limit=50, token="test-token")
        call_kwargs = instance.post.call_args.kwargs or {}
        body = call_kwargs.get("json", {})
        assert body.get("limit", 0) <= 15


@pytest.mark.asyncio
async def test_research_returns_answer():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "answer": "The answer to your question is 42.",
        "docs_used": [
            {"doc_id": "r1", "title": "Source Doc", "url": "https://example.com", "relevance": 0.95}
        ],
        "docs_discarded": 1,
        "gaps": ["missing detail on X"],
        "confidence": 0.87,
        "usage": {"total_tokens": 512, "estimated_cost_usd": 0.0015},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("dewie.mcp_server.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(return_value=mock_response)

        from dewie.mcp_server import _research

        result = await _research("what is 42?", mode="quick", token="test-token")
        assert "The answer to your question is 42." in result
        assert "Source Doc" in result
        assert "missing detail on X" in result
        assert "confidence: 0.87" in result


@pytest.mark.asyncio
async def test_research_deep_mode_passes_max_iterations():
    """_research passes max_iterations to the API in deep mode."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "answer": "done",
        "docs_used": [],
        "docs_discarded": 0,
        "gaps": [],
        "confidence": 0.5,
        "usage": {},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("dewie.mcp_server.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(return_value=mock_response)

        from dewie.mcp_server import _research

        await _research("deep question", mode="deep", max_iterations=6, token="test-token")
        call_kwargs = instance.post.call_args.kwargs or {}
        body = call_kwargs.get("json", {})
        assert body.get("mode") == "deep"
        assert body.get("max_iterations") == 6


# ---------------------------------------------------------------------------
# Unit: error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_call_tool_connect_error_returns_error_result():
    """ConnectError in handle_call_tool returns isError CallToolResult with helpful message."""
    import httpx
    from mcp.types import CallToolRequest

    from dewie.mcp_server import create_server

    server = create_server(auth_token="test-token")
    req = _make_call_request("dewie_search", {"query": "test"})

    with patch("dewie.mcp_server.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

        handler = server.request_handlers[CallToolRequest]
        result = await handler(req)
        assert result.root.isError is True
        assert "Dewie API" in result.root.content[0].text


@pytest.mark.asyncio
async def test_handle_call_tool_http_error_returns_error_result():
    """HTTP 500 from API surfaces as isError=True CallToolResult."""
    import httpx
    from mcp.types import CallToolRequest

    from dewie.mcp_server import create_server

    server = create_server(auth_token="test-token")
    req = _make_call_request("dewie_search", {"query": "test"})

    with patch("dewie.mcp_server.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        mock_req = MagicMock()
        mock_req.url = "http://localhost:8000/query"
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        instance.post = AsyncMock(
            side_effect=httpx.HTTPStatusError("500", request=mock_req, response=mock_resp)
        )

        handler = server.request_handlers[CallToolRequest]
        result = await handler(req)
        assert result.root.isError is True
        assert "500" in result.root.content[0].text


@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    """Calling a non-existent tool returns isError=True."""
    from mcp.types import CallToolRequest

    from dewie.mcp_server import create_server

    server = create_server(auth_token="test-token")
    req = _make_call_request("nonexistent_tool", {})

    with (
        patch("dewie.mcp_server._search") as mock_search,
        patch("dewie.mcp_server._expand") as mock_expand,
        patch("dewie.mcp_server._read") as mock_read,
    ):
        handler = server.request_handlers[CallToolRequest]
        result = await handler(req)
        assert result.root.isError is True
        assert "Unknown tool" in result.root.content[0].text
        mock_search.assert_not_called()
        mock_expand.assert_not_called()
        mock_read.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: live API round-trip (skipped if API unreachable)
# ---------------------------------------------------------------------------

import httpx as _httpx

_LIVE_API_URL = os.environ.get("DEWIE_API_URL", "http://localhost:10946/api")


def _api_is_up() -> bool:
    # Health endpoint is at root, not under /api
    base = _LIVE_API_URL.rstrip("/api").rstrip("/")
    try:
        r = _httpx.get(f"{base}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.asyncio
@pytest.mark.skipif(not _api_is_up(), reason="Dewie API not running at localhost:10946")
async def test_live_search_returns_results():
    """Live round-trip: dewie_search returns at least one result."""
    import dewie.mcp_server as _mcp
    _orig = _mcp.DEWIE_API_URL
    _mcp.DEWIE_API_URL = _LIVE_API_URL
    try:
        from dewie.mcp_server import _search
        result = await _search("information retrieval", limit=3, token="open-mode")
        data = json.loads(result)
        assert "results" in data
        assert isinstance(data["results"], list)
    finally:
        _mcp.DEWIE_API_URL = _orig


@pytest.mark.asyncio
@pytest.mark.skipif(not _api_is_up(), reason="Dewie API not running at localhost:10946")
async def test_live_expand_returns_neighbors():
    """Live round-trip: dewie_expand on a known doc returns neighbors."""
    import dewie.mcp_server as _mcp
    _orig = _mcp.DEWIE_API_URL
    _mcp.DEWIE_API_URL = _LIVE_API_URL
    try:
        from dewie.mcp_server import _expand, _search
        search_result = await _search("information retrieval", limit=1, token="open-mode")
        data = json.loads(search_result)
        results = data.get("results", [])
        if not results:
            pytest.skip("No docs in corpus to expand")
        doc_id = results[0]["doc_id"]
        expand_result = await _expand(doc_id, token="open-mode")
        neighbors = json.loads(expand_result)
        assert isinstance(neighbors, list)
    finally:
        _mcp.DEWIE_API_URL = _orig


@pytest.mark.asyncio
@pytest.mark.skipif(not _api_is_up(), reason="Dewie API not running at localhost:10946")
async def test_live_read_returns_content():
    """Live round-trip: dewie_read on a known doc returns non-empty text."""
    import dewie.mcp_server as _mcp
    _orig = _mcp.DEWIE_API_URL
    _mcp.DEWIE_API_URL = _LIVE_API_URL
    try:
        from dewie.mcp_server import _read, _search
        search_result = await _search("information retrieval", limit=1, token="open-mode")
        data = json.loads(search_result)
        results = data.get("results", [])
        if not results:
            pytest.skip("No docs in corpus to read")
        doc_id = results[0]["doc_id"]
        content = await _read(doc_id, token="open-mode")
        assert isinstance(content, str)
        assert len(content) > 0
    finally:
        _mcp.DEWIE_API_URL = _orig


@pytest.mark.asyncio
@pytest.mark.skipif(not _api_is_up(), reason="Dewie API not running at localhost:10946")
async def test_live_server_lists_tools():
    """Live: MCP server lists all tools correctly."""
    from mcp.types import ListToolsRequest

    from dewie.mcp_server import create_server

    server = create_server(auth_token="test-token")
    req = _make_list_request()
    handler = server.request_handlers[ListToolsRequest]
    result = await handler(req)
    names = {t.name for t in result.root.tools}
    assert names == {"dewie_search", "dewie_expand", "dewie_read", "dewie_intersect", "dewie_bridge", "dewie_browse", "dewie_research"}
