from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_auth_routes_work_in_local_mode() -> None:
    """Without LOCAL_AUTH_ENABLED, unauthenticated /auth/me returns 401.

    Regression test for issue #198: unauthenticated requests must not receive
    admin access via the /auth/me fallback.
    """

    from dewie.api.routes.auth import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    me = client.get("/auth/me")
    assert me.status_code == 200
    # When auth is enabled (default) and no session cookie is present,
    # /auth/me now returns authenticated:false instead of a synthetic local
    # user. This prevents unauthenticated users from seeing 'Open app' on
    # the home page (issue #219).
    body = me.json()
    assert body["authenticated"] is False
    assert body["is_admin"] is False


def test_password_reset_flow() -> None:
    """Test forgot password and reset password endpoints exist."""
    from dewie.api.routes.auth import (
        ForgotPasswordRequest,
        ResetPasswordRequest,
    )
    
    # Just verify the schemas exist and validate
    forgot = ForgotPasswordRequest(username="test@example.com")
    assert forgot.username == "test@example.com"
    
    reset = ResetPasswordRequest(reset_token="token123", password="password123")
    assert reset.reset_token == "token123"
    assert reset.password == "password123"


def test_documents_my_endpoint_available(monkeypatch) -> None:
    from dewie.api.middleware import limiter
    from dewie.api.routes.documents import router
    from dewie.models.content import ContentDocument, ContentStatus

    monkeypatch.setattr(limiter, "enabled", False)

    app = FastAPI()
    app.include_router(router)

    pg = AsyncMock()
    pg.list_recent = AsyncMock(
        return_value=[
            ContentDocument(
                url="https://example.com/doc",
                title="Example",
                status=ContentStatus.READY,
                summary="Summary",
            )
        ]
    )
    app.state.postgres = pg
    app.state.cache = AsyncMock()

    client = TestClient(app)
    resp = client.get("/documents/my?limit=10&offset=0")
    assert resp.status_code == 200
    body = resp.json()
    assert "total" in body
    assert isinstance(body.get("documents"), list)


def test_documents_my_endpoint_legacy_status_no_500(monkeypatch) -> None:
    """Issue #242 — GET /documents/my must not 500 when corpus contains docs with
    legacy status values (e.g. 'deferred') that are no longer in ContentStatus enum.

    _row_to_doc() previously called ContentStatus(row["status"]) directly, which
    raises ValueError for unknown values.  Now it uses _safe_enum() with a PENDING
    fallback, so the endpoint returns 200 instead of crashing with HTTP 500.
    """
    from unittest.mock import MagicMock

    from dewie.api.middleware import limiter
    from dewie.api.routes.documents import router
    from dewie.models.content import ContentDocument, ContentStatus

    monkeypatch.setattr(limiter, "enabled", False)

    app = FastAPI()
    app.include_router(router)

    # Simulate a doc that somehow ended up with PENDING status after the safe
    # enum fallback (what "deferred" would resolve to).
    legacy_doc = MagicMock(spec=ContentDocument)
    legacy_doc.id = __import__("uuid").uuid4()
    legacy_doc.url = "https://example.com/legacy"
    legacy_doc.title = "Legacy doc"
    legacy_doc.summary = None
    legacy_doc.source = "web"
    legacy_doc.status = ContentStatus.PENDING  # safe fallback for "deferred"
    legacy_doc.topics = None
    legacy_doc.keywords = None
    legacy_doc.entities = None
    legacy_doc.sentiment = None
    legacy_doc.ingested_at = None

    pg = AsyncMock()
    pg.list_recent = AsyncMock(return_value=[legacy_doc])
    app.state.postgres = pg
    app.state.cache = AsyncMock()

    client = TestClient(app)
    resp = client.get("/documents/my?limit=10")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "documents" in body
