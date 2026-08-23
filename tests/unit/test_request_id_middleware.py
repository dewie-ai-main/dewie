"""
Tests for request ID middleware, logging/redaction utilities, and error handler.

Covers the requirements from issue #783:
  - test_request_id_middleware_generates_id
  - test_request_id_middleware_uses_provided_id
  - test_request_id_response_header
  - test_redact_sensitive_fields
  - test_redact_nested_fields
  - test_error_handler_returns_request_id
  - test_error_handler_logs_traceback
  - test_all_endpoints_return_request_id
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dewie.api.logging_config import redact_sensitive as redact
from dewie.api.middleware.error_handler import ErrorHandlerMiddleware
from dewie.api.middleware.request_id import RequestIDMiddleware

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_app() -> FastAPI:
    app = FastAPI()
    # Same order as middleware/__init__.py: RequestID added last = outermost,
    # so error responses from ErrorHandlerMiddleware still get the header.
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(RequestIDMiddleware)

    @app.get("/ping")
    async def ping():
        return {"pong": True}

    @app.get("/boom")
    async def boom():
        raise RuntimeError("something went wrong")

    return app


# ── Request ID middleware ─────────────────────────────────────────────────────


def test_request_id_middleware_generates_id():
    """A request without X-Request-ID gets a generated uuid4 id."""
    client = TestClient(_make_app(), raise_server_exceptions=False)
    resp = client.get("/ping")
    assert resp.status_code == 200
    rid = resp.headers.get("X-Request-ID", "")
    assert rid != ""
    # Should look like a uuid4
    assert len(rid) == 36
    assert rid.count("-") == 4


def test_request_id_middleware_uses_provided_id():
    """A request with X-Request-ID reuses the caller's value."""
    client = TestClient(_make_app(), raise_server_exceptions=False)
    resp = client.get("/ping", headers={"X-Request-ID": "my-custom-id-123"})
    assert resp.headers["X-Request-ID"] == "my-custom-id-123"


def test_request_id_response_header():
    """Every response includes the X-Request-ID header."""
    client = TestClient(_make_app(), raise_server_exceptions=False)
    resp = client.get("/ping")
    assert "X-Request-ID" in resp.headers


# ── Redaction ─────────────────────────────────────────────────────────────────


def test_redact_sensitive_fields():
    data = {"api_key": "secret", "password": "hunter2", "token": "tok123", "name": "Alice"}
    result = redact(data)
    assert result["api_key"] == "***REDACTED***"
    assert result["password"] == "***REDACTED***"
    assert result["token"] == "***REDACTED***"
    assert result["name"] == "Alice"


def test_redact_nested_fields():
    data = {"user": {"password": "secret", "email": "a@b.com"}, "authorization": "Bearer xyz"}
    result = redact(data)
    assert result["user"]["password"] == "***REDACTED***"
    assert result["user"]["email"] == "a@b.com"
    assert result["authorization"] == "***REDACTED***"


def test_redact_case_insensitive():
    data = {"API_KEY": "val", "Password": "p"}
    result = redact(data)
    assert result["API_KEY"] == "***REDACTED***"
    assert result["Password"] == "***REDACTED***"


def test_redact_non_dict_passthrough():
    assert redact("plain string") == "plain string"
    assert redact(42) == 42
    assert redact(None) is None


def test_redact_list_of_dicts():
    data = [{"api_key": "x"}, {"name": "Bob"}]
    result = redact(data)
    assert result[0]["api_key"] == "***REDACTED***"
    assert result[1]["name"] == "Bob"


# ── Error handler ─────────────────────────────────────────────────────────────


def test_error_handler_returns_request_id():
    """Unhandled exceptions return JSON with the request_id field."""
    client = TestClient(_make_app(), raise_server_exceptions=False)
    resp = client.get("/boom", headers={"X-Request-ID": "err-req-id"})
    assert resp.status_code == 500
    body = resp.json()
    assert body.get("request_id") == "err-req-id"
    assert "detail" in body
    # Must NOT leak the traceback to the client
    assert "RuntimeError" not in body.get("detail", "")
    assert "Traceback" not in body.get("detail", "")


def test_error_handler_logs_traceback(caplog):
    """Unhandled exceptions are logged at ERROR level with the full traceback."""
    app = _make_app()
    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger="dewie.api"):
        client.get("/boom", headers={"X-Request-ID": "log-test-id"})
    assert any("RuntimeError" in r.message or "something went wrong" in r.message
               for r in caplog.records)


# ── All responses carry the header ────────────────────────────────────────────


def test_all_endpoints_return_request_id():
    """Both success and error responses carry X-Request-ID."""
    client = TestClient(_make_app(), raise_server_exceptions=False)
    for path, expected_status in [("/ping", 200), ("/boom", 500)]:
        resp = client.get(path, headers={"X-Request-ID": "global-id"})
        assert resp.status_code == expected_status
        assert resp.headers.get("X-Request-ID") == "global-id", (
            f"{path} missing X-Request-ID header"
        )
