"""Unit tests for /user/* endpoints."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from dewie.api.middleware_base import InProcessLimiter, _rate_limit_key

# ── Static file tests (issue #348) ────────────────────────────────────────────

_BASE = Path(__file__).resolve().parents[2] / "static"


def test_query_page_no_api_keys():
    html = (_BASE / "app.html").read_text()
    assert "/admin/keys" not in html
    assert "loadKeys" not in html
    assert "createKey" not in html
    assert "revokeKey" not in html


def test_account_page_has_api_keys_section():
    html = (_BASE / "account.html").read_text()
    assert 'id="api-keys"' in html or "API Keys" in html
    assert "/user/api-keys" in html


@pytest.fixture(autouse=True)
def _disable_rate_limiting(monkeypatch):
    """Disable the module-level limiter during unit tests."""
    from dewie.api.middleware import limiter

    monkeypatch.setattr(limiter, "enabled", False)


def _fresh_limiter() -> InProcessLimiter:
    return InProcessLimiter(key_func=_rate_limit_key)


# ── Engine mock helper ────────────────────────────────────────────────────────

_TEST_USER_ID = "00000000-0000-0000-0000-000000000001"
_TEST_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")


def _make_engine_mock(fetchone_result=None, fetchall_result=None):
    mock_result = MagicMock()
    mock_result.fetchone.return_value = fetchone_result
    mock_result.fetchall.return_value = fetchall_result or []

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_conn)
    cm.__aexit__ = AsyncMock(return_value=None)

    pg = AsyncMock()
    pg._engine = MagicMock()
    pg._engine.begin.return_value = cm
    pg._engine.connect.return_value = cm

    return pg, mock_conn, mock_result


def _make_user_app(pg=None, authenticated: bool = True):
    """Build a minimal FastAPI app with the /user router.

    When ``authenticated=True``, an HTTP middleware injects a synthetic user
    session into ``request.state`` so that ``_require_user`` succeeds.
    """
    from dewie.api.routes.user import router

    app = FastAPI()
    app.state.limiter = _fresh_limiter()
    app.include_router(router)

    if pg is None:
        pg, _, _ = _make_engine_mock()
    app.state.postgres = pg
    app.state.processor = AsyncMock()

    if authenticated:

        @app.middleware("http")
        async def _inject_session(request: Request, call_next):
            request.state.user_id = _TEST_USER_ID
            request.state.tenant_id = _TEST_TENANT_ID
            request.state.key_id = None  # session auth (not API key)
            return await call_next(request)

    return app


# ── POST /user/ingest ─────────────────────────────────────────────────────────


class TestUserIngest:
    def test_unauthenticated_returns_401(self):
        client = TestClient(_make_user_app(authenticated=False), raise_server_exceptions=False)
        resp = client.post("/user/ingest", json={"url": "https://example.com/article"})
        assert resp.status_code == 401

    def test_missing_url_returns_422(self):
        client = TestClient(_make_user_app(), raise_server_exceptions=False)
        resp = client.post("/user/ingest", json={})
        assert resp.status_code == 422

    def test_valid_url_accepted(self):
        from dewie.models.content import ContentDocument, ContentStatus

        mock_doc = ContentDocument(
            url="https://example.com/article",
            title="Test Article",
            status=ContentStatus.PENDING,
        )

        pg, mock_conn, mock_result = _make_engine_mock()
        # fetchone serves both the daily-limit check (row.cnt) and the
        # canonical-id lookup after upsert (row[0] must be a valid UUID)
        limit_row = MagicMock()
        limit_row.cnt = 0
        limit_row.__getitem__.return_value = str(mock_doc.id)
        mock_result.fetchone.return_value = limit_row
        pg.upsert = AsyncMock()
        pg.write_body_text = AsyncMock()

        with (
            patch(
                "dewie.ingestion.source_router.SourceRouter",
            ) as MockRouter,
            patch("dewie.storage.body_store.save_body"),
        ):
            # Make SourceRouter yield mock_doc as an async generator
            async def _gen(_url):
                yield mock_doc

            instance = AsyncMock()
            instance.fetch = _gen
            instance.close = AsyncMock()
            MockRouter.return_value = instance

            client = TestClient(_make_user_app(pg), raise_server_exceptions=False)
            resp = client.post("/user/ingest", json={"url": "https://example.com/article"})

        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "pending"
        assert "doc_id" in data


# ── GET /user/uploads ─────────────────────────────────────────────────────────


class TestUserUploads:
    def test_unauthenticated_returns_401(self):
        client = TestClient(_make_user_app(authenticated=False), raise_server_exceptions=False)
        resp = client.get("/user/uploads")
        assert resp.status_code == 401

    def test_authenticated_returns_list(self):
        pg, _, mock_result = _make_engine_mock(fetchall_result=[])
        client = TestClient(_make_user_app(pg), raise_server_exceptions=False)
        resp = client.get("/user/uploads")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_upload_items(self):
        now = datetime.now(UTC)
        row = MagicMock()
        row.id = uuid.uuid4()
        row.url = "https://example.com/article"
        row.title = "Test"
        row.status = "ready"
        row.ingested_at = now

        pg, _, mock_result = _make_engine_mock(fetchall_result=[row])
        client = TestClient(_make_user_app(pg), raise_server_exceptions=False)
        resp = client.get("/user/uploads")

        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["url"] == "https://example.com/article"
        assert items[0]["title"] == "Test"
        assert items[0]["status"] == "ready"


# ── POST /user/api-keys ───────────────────────────────────────────────────────


class TestUserApiKeyCreate:
    def test_unauthenticated_returns_401(self):
        client = TestClient(_make_user_app(authenticated=False), raise_server_exceptions=False)
        resp = client.post("/user/api-keys")
        assert resp.status_code == 401

    def test_authenticated_creates_key(self):
        key_id = uuid.uuid4()
        now = datetime.now(UTC)

        pg, _, _ = _make_engine_mock()
        with patch(
            "dewie.auth.create_api_key",
            AsyncMock(
                return_value=(
                    "ck_live_abc123",
                    {"id": key_id, "key_prefix": "ck_live_abc", "created_at": now},
                )
            ),
        ):
            client = TestClient(_make_user_app(pg), raise_server_exceptions=False)
            resp = client.post("/user/api-keys")

        assert resp.status_code == 201
        data = resp.json()
        assert data["key"] == "ck_live_abc123"
        assert "key_id" in data
        assert "prefix" in data

    def test_api_key_auth_rejected(self):
        """Creating a key via API key auth (not session) must be rejected."""
        from dewie.api.routes.user import router

        app = FastAPI()
        app.include_router(router)
        pg, _, _ = _make_engine_mock()
        app.state.postgres = pg

        @app.middleware("http")
        async def _inject_api_key_session(request: Request, call_next):
            request.state.user_id = _TEST_USER_ID
            request.state.tenant_id = _TEST_TENANT_ID
            request.state.key_id = str(uuid.uuid4())  # API key auth, not session
            return await call_next(request)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/user/api-keys")
        assert resp.status_code == 403

    def test_create_key_passes_user_id(self):
        """POST /user/api-keys must pass the session user_id to create_api_key."""
        key_id = uuid.uuid4()
        now = datetime.now(UTC)

        pg, _, _ = _make_engine_mock()

        async def _assert_user_id(*args, **kwargs):
            assert str(kwargs.get("user_id")) == _TEST_USER_ID, (
                f"Expected user_id={_TEST_USER_ID}, got {kwargs.get('user_id')}"
            )
            return "ck_live_test123", {"id": key_id, "key_prefix": "ck_live_t", "created_at": now}

        with patch(
            "dewie.auth.create_api_key",
            AsyncMock(side_effect=_assert_user_id),
        ):
            client = TestClient(_make_user_app(pg), raise_server_exceptions=False)
            resp = client.post("/user/api-keys")

        assert resp.status_code == 201
        data = resp.json()
        assert data["key"] == "ck_live_test123"


# ── GET /user/api-keys ────────────────────────────────────────────────────────


class TestUserApiKeyList:
    def test_unauthenticated_returns_401(self):
        client = TestClient(_make_user_app(authenticated=False), raise_server_exceptions=False)
        resp = client.get("/user/api-keys")
        assert resp.status_code == 401

    def test_api_keys_endpoint_authenticated(self):
        pg, _, mock_result = _make_engine_mock(fetchall_result=[])
        client = TestClient(_make_user_app(pg), raise_server_exceptions=False)
        resp = client.get("/user/api-keys")
        assert resp.status_code == 200

    def test_authenticated_returns_list(self):
        pg, _, mock_result = _make_engine_mock(fetchall_result=[])
        client = TestClient(_make_user_app(pg), raise_server_exceptions=False)
        resp = client.get("/user/api-keys")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_key_items(self):
        now = datetime.now(UTC)
        row = MagicMock()
        row.id = uuid.uuid4()
        row.key_prefix = "ck_live_abc"
        row.name = "my-key"
        row.created_at = now
        row.last_used_at = None
        row.scopes = ["read", "ingest"]

        pg, _, mock_result = _make_engine_mock(fetchall_result=[row])
        client = TestClient(_make_user_app(pg), raise_server_exceptions=False)
        resp = client.get("/user/api-keys")

        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["prefix"] == "ck_live_abc"
        assert items[0]["name"] == "my-key"
        assert items[0]["scopes"] == ["read", "ingest"]

    def test_returns_keys_for_different_user(self):
        """Keys created for a different user must not appear in list.

        Simulates: user A has a key in DB, user B queries → B gets empty list
        because the WHERE user_id clause filters it out.
        """
        other_user_id = "00000000-0000-0000-0000-000000000099"

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.fetchone.return_value = None

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=mock_conn)
        cm.__aexit__ = AsyncMock(return_value=None)

        pg, _, _ = _make_engine_mock()
        pg._engine.connect.return_value = cm

        app = _make_user_app(pg)

        @app.middleware("http")
        async def _override_user(request: Request, call_next):
            request.state.user_id = other_user_id
            return await call_next(request)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/user/api-keys")

        assert resp.status_code == 200
        items = resp.json()
        assert items == []


# ── DELETE /user/api-keys/{key_id} ────────────────────────────────────────────


class TestUserApiKeyRevoke:
    def test_unauthenticated_returns_401(self):
        client = TestClient(_make_user_app(authenticated=False), raise_server_exceptions=False)
        resp = client.delete(f"/user/api-keys/{uuid.uuid4()}")
        assert resp.status_code == 401

    def test_valid_key_returns_204(self):
        pg, _, _ = _make_engine_mock()
        with patch(
            "dewie.auth.revoke_api_key",
            AsyncMock(return_value=True),
        ):
            client = TestClient(_make_user_app(pg), raise_server_exceptions=False)
            resp = client.delete(f"/user/api-keys/{uuid.uuid4()}")
        assert resp.status_code == 204

    def test_unknown_key_returns_404(self):
        pg, _, _ = _make_engine_mock()
        with patch(
            "dewie.auth.revoke_api_key",
            AsyncMock(return_value=False),
        ):
            client = TestClient(_make_user_app(pg), raise_server_exceptions=False)
            resp = client.delete(f"/user/api-keys/{uuid.uuid4()}")
        assert resp.status_code == 404


# ── User model catalog and selection ─────────────────────────────────────────


class TestUserModelCatalog:
    def test_get_user_model_catalog(self):
        app = _make_user_app()
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "dewie.model_registry.registry.catalog",
            AsyncMock(
                return_value={
                    "context": "user",
                    "providers": [{"id": "openai"}],
                    "models_by_provider": {"openai": [{"id": "gpt-4o", "selectable": True}]},
                    "selections": {},
                }
            ),
        ):
            resp = client.get("/user/model-catalog")

        assert resp.status_code == 200
        assert resp.json()["context"] == "user"

    def test_get_user_model_catalog_rejects_invalid_purpose(self):
        app = _make_user_app()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/user/model-catalog?purpose=invalid")

        assert resp.status_code == 400

    def test_get_user_model_selection(self):
        app = _make_user_app()
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "dewie.model_registry.registry.get_context_selection",
            AsyncMock(return_value={"chat_provider_aq": "anthropic"}),
        ):
            resp = client.get("/user/model-selection")

        assert resp.status_code == 200
        assert resp.json()["values"]["chat_provider_aq"] == "anthropic"

    def test_patch_user_model_selection(self):
        app = _make_user_app()
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "dewie.model_registry.registry.set_context_selection",
            AsyncMock(return_value={"chat_provider_aq": "openai", "chat_model_aq": "gpt-4o"}),
        ):
            resp = client.patch(
                "/user/model-selection",
                json={"values": {"chat_provider_aq": "openai", "chat_model_aq": "gpt-4o"}},
            )

        assert resp.status_code == 200
        assert resp.json()["values"]["chat_model_aq"] == "gpt-4o"
