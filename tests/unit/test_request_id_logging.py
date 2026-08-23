"""Tests for request-ID middleware, error handler, and logging config.

Covers:
  - Request-ID generation and header passthrough
  - Sensitive field redaction in logged data
  - Structured error responses with request ID
  - Log format includes request ID
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from dewie.api.logging_config import (
    get_request_id,
    redact_sensitive,
    reset_request_id,
    set_request_id,
)

# ── Request-ID middleware ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_request_id_middleware_generates_id():
    """When no X-Request-ID header is provided, a uuid4 is generated."""
    from dewie.api.middleware import RequestIDMiddleware

    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/test")
    async def test_endpoint(request: Request):
        return {"request_id": getattr(request.state, "request_id", "")}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["request_id"]
    try:
        uuid.UUID(body["request_id"])
    except ValueError:
        pytest.fail(f"Generated request_id is not a valid UUID: {body['request_id']}")


@pytest.mark.asyncio
async def test_request_id_middleware_uses_provided_id():
    """When X-Request-ID header is provided, it's used as-is."""
    from dewie.api.middleware import RequestIDMiddleware

    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/test")
    async def test_endpoint(request: Request):
        return {"request_id": getattr(request.state, "request_id", "")}

    provided_id = "my-custom-request-id-12345"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/test", headers={"X-Request-ID": provided_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["request_id"] == provided_id


@pytest.mark.asyncio
async def test_request_id_response_header():
    """The X-Request-ID is echoed back in the response headers."""
    from dewie.api.middleware import RequestIDMiddleware

    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/test")
    async def test_endpoint(request: Request):
        return {"ok": True}

    provided_id = "header-test-uuid"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/test", headers={"X-Request-ID": provided_id})
    assert resp.status_code == 200
    assert resp.headers.get("x-request-id") == provided_id


# ── Redaction ─────────────────────────────────────────────────────────────────


def test_redact_sensitive_fields():
    """Redact known-sensitive keys in a flat dict."""
    data = {
        "api_key": "sk-1234567890abcdef",
        "username": "alice",
        "password": "hunter2",
        "token": "bearer abc123",
    }
    result = redact_sensitive(data)
    assert result["api_key"] == "***REDACTED***"
    assert result["username"] == "alice"
    assert result["password"] == "***REDACTED***"
    assert result["token"] == "***REDACTED***"


def test_redact_nested_fields():
    """Redaction recurses into nested dicts and lists."""
    data = {
        "user": {
            "api_key": "secret-key",
            "name": "bob",
        },
        "auth": [
            {"token": "tok-1"},
            {"token": "tok-2", "password": "p@ssw0rd"},
        ],
    }
    result = redact_sensitive(data)
    assert result["user"]["api_key"] == "***REDACTED***"
    assert result["user"]["name"] == "bob"
    assert result["auth"][0]["token"] == "***REDACTED***"
    assert result["auth"][1]["token"] == "***REDACTED***"
    assert result["auth"][1]["password"] == "***REDACTED***"


# ── Error handler ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_error_handler_returns_request_id():
    """Error responses include the request_id field."""
    from dewie.api.middleware import ErrorHandlerMiddleware, RequestIDMiddleware

    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)

    @app.get("/fail")
    async def fail_endpoint(request: Request):
        raise ValueError("test error message")

    provided_id = "error-test-uuid"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/fail", headers={"X-Request-ID": provided_id})
    assert resp.status_code == 500
    body = resp.json()
    assert "request_id" in body
    assert body["request_id"] == provided_id
    assert body["detail"] == "test error message"


@pytest.mark.asyncio
async def test_error_handler_logs_traceback():
    """Unhandled exceptions are logged with full traceback at ERROR level."""
    from dewie.api.middleware import ErrorHandlerMiddleware, RequestIDMiddleware

    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)

    @app.get("/fail")
    async def fail_endpoint(request: Request):
        raise RuntimeError("boom")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch("dewie.api.middleware.error_handler.logger") as mock_logger:
            mock_logger.error = AsyncMock()
            resp = await client.get("/fail")
            assert resp.status_code == 500
            mock_logger.error.assert_called_once()
            # Verify the exception message appears in the log
            call_args = mock_logger.error.call_args
            assert "boom" in str(call_args)


# ── Integration: all endpoints return request_id ──────────────────────────────


@pytest.mark.asyncio
async def test_all_endpoints_return_request_id():
    """Every endpoint should echo back the X-Request-ID in response headers."""
    from dewie.api.middleware import ErrorHandlerMiddleware, RequestIDMiddleware

    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)

    @app.get("/a")
    async def endpoint_a(request: Request):
        return {"ok": True}

    @app.get("/b")
    async def endpoint_b(request: Request):
        return {"ok": True}

    @app.post("/c")
    async def endpoint_c(request: Request):
        return {"ok": True}

    provided_id = "integration-uuid-001"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for path in ("/a", "/b", "/c"):
            if path == "/c":
                resp = await client.post(path, headers={"X-Request-ID": provided_id})
            else:
                resp = await client.get(path, headers={"X-Request-ID": provided_id})
            assert resp.status_code == 200, f"Failed on {path}: {resp.text}"
            assert resp.headers.get("x-request-id") == provided_id, f"Missing X-Request-ID on {path}"


# ── Contextvar helpers ────────────────────────────────────────────────────────


def test_contextvar_set_and_get():
    """set_request_id and get_request_id work correctly."""
    token = set_request_id("ctx-test-001")
    assert get_request_id() == "ctx-test-001"
    reset_request_id(token)


def test_contextvar_isolation():
    """get_request_id returns empty string outside of set context."""
    assert get_request_id() == ""


def test_redact_bearer_token_in_string():
    """Redacts bearer token fragments in string values."""
    data = {"header": "Bearer eyJhbGciOiJIUzI1NiJ9.secret"}
    result = redact_sensitive(data)
    assert "secret" not in result["header"]
    assert "Bearer" in result["header"]
    assert "***REDACTED***" in result["header"]
