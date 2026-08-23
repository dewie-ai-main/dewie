"""Tests for dewie.api.routes.admin — admin API endpoints."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Pydantic model tests ───────────────────────────────────────────────────────


def test_create_key_request_defaults():
    from dewie.api.routes.admin import CreateKeyRequest

    req = CreateKeyRequest()
    assert req.scopes == ["read"]
    assert req.live is True
    assert req.workspace_ids == []
    assert req.name is None


def test_key_response_model():
    from dewie.api.routes.admin import KeyResponse

    key_id = uuid.uuid4()
    ws_id = uuid.uuid4()
    resp = KeyResponse(
        id=key_id,
        workspace_ids=[ws_id],
        key_prefix="ck_live_abc",
        scopes=["read"],
        name="test",
        created_at="2026-01-01",
    )
    assert resp.id == key_id
    assert resp.workspace_ids == [ws_id]


def test_create_workspace_request_defaults():
    from dewie.api.routes.admin import CreateWorkspaceRequest

    req = CreateWorkspaceRequest(name="test-ws")
    assert req.sharing_tier == "internal_only"
    assert req.parent_id is None


def test_create_corpus_request_fields():
    from dewie.api.routes.admin import CreateCorpusRequest

    ws_id = uuid.uuid4()
    req = CreateCorpusRequest(name="My Corpus", slug="my-corpus", workspace_id=ws_id)
    assert req.slug == "my-corpus"
    assert req.workspace_id == ws_id


# ── Helper function tests ──────────────────────────────────────────────────────


def test_require_admin_passes_when_is_admin():
    from fastapi import Request

    from dewie.api.routes.admin import _require_admin

    mock_request = MagicMock(spec=Request)
    mock_request.state.is_admin = True
    _require_admin(mock_request)  # should not raise


def test_require_admin_raises_when_not_admin():
    from fastapi import HTTPException, Request

    from dewie.api.routes.admin import _require_admin

    mock_request = MagicMock(spec=Request)
    mock_request.state.is_admin = False
    with pytest.raises(HTTPException) as exc_info:
        _require_admin(mock_request)
    assert exc_info.value.status_code == 403


def test_pg_helper():
    from fastapi import Request

    from dewie.api.routes.admin import _pg

    mock_pg = MagicMock()
    mock_request = MagicMock(spec=Request)
    mock_request.app.state.postgres = mock_pg
    result = _pg(mock_request)
    assert result is mock_pg


# ── Model catalog endpoints ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_model_catalog_success(monkeypatch):
    from dewie.api.routes.admin import get_model_catalog

    mock_registry = MagicMock()
    mock_registry.catalog = AsyncMock(
        return_value={
            "context": "admin",
            "providers": [{"id": "openai"}],
            "models_by_provider": {"openai": [{"id": "gpt-4o", "selectable": True}]},
            "selections": {},
        }
    )
    monkeypatch.setattr("dewie.api.routes.admin._model_registry", lambda: mock_registry)

    req = _make_request()
    resp = await get_model_catalog(req, context="admin", include_hidden=True)
    assert resp.context == "admin"
    assert resp.providers[0]["id"] == "openai"


@pytest.mark.asyncio
async def test_get_model_catalog_rejects_invalid_purpose(monkeypatch):
    from fastapi import HTTPException

    from dewie.api.routes.admin import get_model_catalog

    mock_registry = AsyncMock()
    monkeypatch.setattr("dewie.api.routes.admin._model_registry", lambda: mock_registry)

    req = _make_request()
    with pytest.raises(HTTPException) as exc:
        await get_model_catalog(req, context="admin", include_hidden=True, purpose="invalid")

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_get_embedding_status_success(monkeypatch):
    from dewie.api.routes.admin import get_embedding_status

    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = MagicMock(
        base_url="http://localhost:8080/v1",
        probe_url="http://localhost:8080/v1/models",
        available=True,
    )
    mock_registry.catalog = AsyncMock(
        return_value={
            "context": "admin",
            "purpose": "embedding",
            "providers": [],
            "models_by_provider": {
                "custom": [
                    {
                        "id": "Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M",
                        "is_embedding_model": True,
                        "embedding": {
                            "supports_mrl": True,
                            "min_dimensions": 32,
                            "default_dimensions": 4096,
                            "max_dimensions": 4096,
                            "source": "qwen_model_card",
                        },
                    }
                ]
            },
            "selections": {},
        }
    )

    monkeypatch.setattr("dewie.api.routes.admin._model_registry", lambda: mock_registry)
    monkeypatch.setattr("dewie.providers.factory._resolve_embed", lambda: ("custom", "Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M", 1024))
    monkeypatch.setattr("dewie.storage.postgres._embed_dimensions_for_model", lambda _m: 1024)
    monkeypatch.setenv(
        "SERVERS_JSON",
        '[{"label": "custom", "api_format": "openai", "endpoint": "http://localhost:8080/v1"}]',
    )
    monkeypatch.setenv("EMBED_OUTPUT_DIMENSIONS", "1024")

    req = _make_request()
    resp = await get_embedding_status(req)

    assert resp.provider == "custom"
    assert resp.model == "Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M"
    assert resp.base_url == "http://localhost:8080/v1"
    assert resp.supports_mrl is True
    assert resp.requested_output_dimensions == 1024
    assert resp.storage_dimensions == 1024


@pytest.mark.asyncio
async def test_register_model_provider_success(monkeypatch):
    from dewie.api.routes.admin import CatalogProviderRequest, register_model_provider

    mock_registry = AsyncMock()
    mock_registry.register_provider = AsyncMock(return_value=None)
    monkeypatch.setattr("dewie.api.routes.admin._model_registry", lambda: mock_registry)

    req = _make_request()
    body = CatalogProviderRequest(provider_id="custom-main", base_url="http://localhost:1234/v1")
    resp = await register_model_provider(body, req)
    assert resp.ok is True
    mock_registry.register_provider.assert_awaited_once()


@pytest.mark.asyncio
async def test_set_context_selection_success(monkeypatch):
    from dewie.api.routes.admin import ContextSelectionRequest, set_context_selection

    mock_registry = AsyncMock()
    mock_registry.set_context_selection = AsyncMock(
        return_value={"chat_provider_aq": "openai", "chat_model_aq": "gpt-4o"}
    )
    monkeypatch.setattr("dewie.api.routes.admin._model_registry", lambda: mock_registry)

    req = _make_request()
    body = ContextSelectionRequest(values={"chat_provider_aq": "openai", "chat_model_aq": "gpt-4o"})
    resp = await set_context_selection("admin", body, req)
    assert resp["context"] == "admin"
    assert resp["values"]["chat_model_aq"] == "gpt-4o"


# ── Config endpoints ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_config_reads_file(monkeypatch, tmp_path):
    from dewie.api.routes.admin import get_config

    cfg_path = tmp_path / "dewie.yml"
    cfg_path.write_text("query_default_ranker: rrf_chunks\n", encoding="utf-8")
    monkeypatch.setenv("DEWIE_CONFIG_PATH", str(cfg_path))

    req = _make_request()
    result = await get_config(req)

    assert result.file_path == str(cfg_path)
    keys = {v.key for v in result.values}
    assert "query_default_ranker" in keys
    ranker_value = next(v for v in result.values if v.key == "query_default_ranker")
    assert ranker_value.value == "rrf_chunks"
    assert ranker_value.source == "file"


@pytest.mark.asyncio
async def test_set_config_updates_ranker(monkeypatch, tmp_path):
    from dewie.api.routes.admin import ConfigSetRequest, set_config

    cfg_path = tmp_path / "dewie.yml"
    cfg_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("DEWIE_CONFIG_PATH", str(cfg_path))

    req = _make_request()
    body = ConfigSetRequest(path="query_default_ranker", value="rrf_chunks", value_type="str")
    result = await set_config(body, req)

    assert result.ok is True
    assert result.path == "query_default_ranker"
    assert result.reload_behavior == "hot_reload"
    assert "query_default_ranker: rrf_chunks" in cfg_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_set_config_rejects_invalid_ranker(monkeypatch, tmp_path):
    from fastapi import HTTPException

    from dewie.api.routes.admin import ConfigSetRequest, set_config

    cfg_path = tmp_path / "dewie.yml"
    cfg_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("DEWIE_CONFIG_PATH", str(cfg_path))

    req = _make_request()
    body = ConfigSetRequest(path="query_default_ranker", value="not_a_ranker", value_type="str")

    with pytest.raises(HTTPException) as exc_info:
        await set_config(body, req)
    assert exc_info.value.status_code == 400


# ── create_key endpoint ────────────────────────────────────────────────────────


def _make_request(is_admin: bool = True, pg: object = None) -> MagicMock:
    from fastapi import Request

    mock_pg = pg or MagicMock()
    req = MagicMock(spec=Request)
    req.state.is_admin = is_admin
    req.app.state.postgres = mock_pg
    return req


@pytest.mark.asyncio
async def test_create_key_invalid_scope():
    from fastapi import HTTPException

    from dewie.api.routes.admin import CreateKeyRequest, create_key

    req = _make_request()
    body = CreateKeyRequest(scopes=["not_a_real_scope"])
    with pytest.raises(HTTPException) as exc_info:
        await create_key(body, req)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_create_key_success():
    import dewie.api.routes.admin as admin_mod
    from dewie.api.routes.admin import CreateKeyRequest, create_key

    mock_pg = MagicMock()
    ws_id = uuid.uuid4()
    mock_pg.return_value = mock_pg


    raw = "ck_live_abc123"
    record = {
        "id": uuid.uuid4(),
        "workspace_ids": [ws_id],
        "key_prefix": "ck_live_abc",
        "scopes": ["read"],
        "name": "my key",
        "created_at": "2026-01-01",
    }


    original = admin_mod.create_key
    req = _make_request(pg=mock_pg)
    body = CreateKeyRequest(name="my key", workspace_ids=[ws_id])

    import unittest.mock as _mock

    with _mock.patch("dewie.auth.create_api_key", new=AsyncMock(return_value=(raw, record))):
        result = await create_key(body, req)
    assert result.key == raw
    assert result.record.name == "my key"


# ── list_keys endpoint ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_keys():
    from dewie.api.routes.admin import list_keys

    ws_id = uuid.uuid4()
    key_id = uuid.uuid4()
    fake_rows = [
        {
            "id": key_id,
            "workspace_ids": [ws_id],
            "key_prefix": "ck_live_abc",
            "scopes": ["read"],
            "name": "k",
            "created_at": "2026-01-01",
        }
    ]

    mock_result = MagicMock()
    mock_result.mappings.return_value.fetchall.return_value = fake_rows

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)
    mock_connect_ctx = MagicMock()
    mock_connect_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_connect_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_pg = MagicMock()
    mock_pg._engine.connect.return_value = mock_connect_ctx

    req = _make_request(pg=mock_pg)
    result = await list_keys(req)
    assert len(result) == 1
    assert result[0].id == key_id


# ── revoke_key endpoint ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_revoke_key_not_found():
    import unittest.mock as _mock

    from fastapi import HTTPException

    from dewie.api.routes.admin import revoke_key

    req = _make_request()
    with _mock.patch("dewie.auth.revoke_api_key", new=AsyncMock(return_value=False)):
        with pytest.raises(HTTPException) as exc_info:
            await revoke_key(uuid.uuid4(), req)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_revoke_key_success():
    import unittest.mock as _mock

    from dewie.api.routes.admin import revoke_key

    req = _make_request()
    with _mock.patch("dewie.auth.revoke_api_key", new=AsyncMock(return_value=True)):
        result = await revoke_key(uuid.uuid4(), req)
    assert result is None  # 204 No Content


# ── Workspace endpoints ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_workspace_success():
    from dewie.api.routes.admin import CreateWorkspaceRequest, create_workspace

    ws_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.create_workspace = AsyncMock(
        return_value={
            "id": ws_id,
            "name": "Test WS",
            "parent_id": None,
            "sharing_tier": "internal_only",
            "created_at": "2026-01-01",
        }
    )
    req = _make_request(pg=mock_pg)
    body = CreateWorkspaceRequest(name="Test WS")
    result = await create_workspace(body, req)
    assert result.id == ws_id
    assert result.name == "Test WS"


@pytest.mark.asyncio
async def test_list_workspaces_success():
    from dewie.api.routes.admin import list_workspaces

    mock_pg = AsyncMock()
    mock_pg.get_workspaces = AsyncMock(
        return_value=[
            {
                "id": uuid.uuid4(),
                "name": "WS1",
                "parent_id": None,
                "sharing_tier": "internal_only",
                "created_at": "2026-01-01",
            }
        ]
    )
    req = _make_request(pg=mock_pg)
    result = await list_workspaces(req)
    assert len(result) == 1
    assert result[0].name == "WS1"


@pytest.mark.asyncio
async def test_delete_workspace():
    from dewie.api.routes.admin import delete_workspace

    mock_pg = AsyncMock()
    mock_pg.delete_workspace = AsyncMock(return_value=None)
    req = _make_request(pg=mock_pg)
    await delete_workspace(uuid.uuid4(), req)
    mock_pg.delete_workspace.assert_awaited_once()


# ── Corpus endpoints ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_corpus_success():
    from dewie.api.routes.admin import CreateCorpusRequest, create_corpus

    corpus_id = uuid.uuid4()
    ws_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.create_corpus = AsyncMock(
        return_value={
            "id": corpus_id,
            "name": "Test Corpus",
            "slug": "test-corpus",
            "workspace_id": ws_id,
            "sharing_tier": "internal_only",
            "created_at": "2026-01-01",
        }
    )
    req = _make_request(pg=mock_pg)
    body = CreateCorpusRequest(name="Test Corpus", slug="test-corpus", workspace_id=ws_id)
    result = await create_corpus(body, req)
    assert result.id == corpus_id
    assert result.slug == "test-corpus"


@pytest.mark.asyncio
async def test_list_corpora_success():
    from dewie.api.routes.admin import list_corpora

    ws_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.get_corpora = AsyncMock(
        return_value=[
            {
                "id": uuid.uuid4(),
                "name": "C1",
                "slug": "c1",
                "workspace_id": ws_id,
                "sharing_tier": "internal_only",
                "created_at": "2026-01-01",
            }
        ]
    )
    req = _make_request(pg=mock_pg)
    result = await list_corpora(req)
    assert len(result) == 1


# ── Local user management endpoints ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_users_success():
    from dewie.api.routes.admin import list_users

    user_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.get_local_users = AsyncMock(
        return_value=[
            {
                "id": user_id,
                "email": "user@example.com",
                "name": "User",
                "is_admin": False,
                "activation_status": "approved",
                "has_password": True,
                "created_at": "2026-01-01",
                "last_login_at": None,
            }
        ]
    )
    req = _make_request(pg=mock_pg)
    result = await list_users(req)
    assert len(result) == 1
    assert result[0].email == "user@example.com"
    assert result[0].id == str(user_id)


@pytest.mark.asyncio
async def test_update_user_invalid_activation_status():
    from fastapi import HTTPException

    from dewie.api.routes.admin import UpdateAdminUserRequest, update_user

    req = _make_request(pg=AsyncMock())
    body = UpdateAdminUserRequest(activation_status="bad-status")
    with pytest.raises(HTTPException) as exc_info:
        await update_user(uuid.uuid4(), body, req)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_update_user_not_found():
    from fastapi import HTTPException

    from dewie.api.routes.admin import UpdateAdminUserRequest, update_user

    mock_pg = AsyncMock()
    mock_pg.update_local_user = AsyncMock(return_value=None)
    req = _make_request(pg=mock_pg)
    body = UpdateAdminUserRequest(name="Updated")
    with pytest.raises(HTTPException) as exc_info:
        await update_user(uuid.uuid4(), body, req)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_set_user_password_short_rejected():
    from fastapi import HTTPException

    from dewie.api.routes.admin import SetUserPasswordRequest, set_user_password

    req = _make_request(pg=AsyncMock())
    body = SetUserPasswordRequest(password="short")
    with pytest.raises(HTTPException) as exc_info:
        await set_user_password(uuid.uuid4(), body, req)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_delete_user_not_found():
    from fastapi import HTTPException

    from dewie.api.routes.admin import delete_user

    mock_pg = AsyncMock()
    mock_pg.delete_local_user = AsyncMock(return_value=False)
    req = _make_request(pg=mock_pg)
    with pytest.raises(HTTPException) as exc_info:
        await delete_user(uuid.uuid4(), req)
    assert exc_info.value.status_code == 404


# ── Query log endpoints ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_query_log_entries():
    from dewie.api.routes.admin import list_query_log

    fake_rows = [
        MagicMock(_mapping={"id": 1, "ts": "2026-01-01", "question": "q", "answer": "a"})
    ]
    mock_result = MagicMock()
    mock_result.fetchall.return_value = fake_rows

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_pg = MagicMock()
    mock_pg._engine.connect.return_value = mock_ctx

    req = _make_request(pg=mock_pg)
    result = await list_query_log(req)
    assert len(result) == 1


@pytest.mark.asyncio
async def test_get_query_log_entry_not_found():
    from fastapi import HTTPException

    from dewie.api.routes.admin import get_query_log_entry

    mock_result = MagicMock()
    mock_result.fetchone.return_value = None

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_pg = MagicMock()
    mock_pg._engine.connect.return_value = mock_ctx

    req = _make_request(pg=mock_pg)
    req.state.is_admin = True

    with pytest.raises(HTTPException) as exc_info:
        await get_query_log_entry(999, req)
    assert exc_info.value.status_code == 404


# ── Create local user endpoint ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_user_success():
    """POST /admin/users creates a new local user and returns 201."""
    import uuid
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.api.routes.admin import CreateAdminUserRequest, create_user

    user_id = uuid.uuid4()
    new_user = {
        "id": user_id,
        "email": "newuser@example.com",
        "name": "New User",
        "is_admin": False,
        "activation_status": "approved",
        "has_password": True,
        "created_at": "2024-01-01T00:00:00",
        "last_login_at": None,
    }

    mock_pg = MagicMock()
    req = _make_request(pg=mock_pg)

    with patch("dewie.local_auth.create_local_user", new=AsyncMock(return_value=new_user)):
        body = CreateAdminUserRequest(
            email="newuser@example.com",
            password="securepassword",
            name="New User",
        )
        result = await create_user(body, req)

    assert result.email == "newuser@example.com"
    assert result.id == str(user_id)
    assert result.is_admin is False


@pytest.mark.asyncio
async def test_create_user_short_password_rejected():
    """POST /admin/users rejects passwords shorter than 8 characters."""
    from fastapi import HTTPException

    from dewie.api.routes.admin import CreateAdminUserRequest, create_user

    mock_pg = MagicMock()
    req = _make_request(pg=mock_pg)

    body = CreateAdminUserRequest(email="x@example.com", password="short")
    with pytest.raises(HTTPException) as exc_info:
        await create_user(body, req)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_create_user_duplicate_email_409():
    """POST /admin/users returns 409 when email already exists."""
    from unittest.mock import AsyncMock, patch

    from fastapi import HTTPException

    from dewie.api.routes.admin import CreateAdminUserRequest, create_user

    mock_pg = MagicMock()
    req = _make_request(pg=mock_pg)

    with patch(
        "dewie.local_auth.create_local_user",
        new=AsyncMock(side_effect=ValueError("Email already exists: x@example.com")),
    ):
        body = CreateAdminUserRequest(email="x@example.com", password="longpassword")
        with pytest.raises(HTTPException) as exc_info:
            await create_user(body, req)
    assert exc_info.value.status_code == 409


# ── Suspend / unsuspend user endpoints ────────────────────────────────────────


@pytest.mark.asyncio
async def test_suspend_user_success():
    """POST /admin/users/{id}/suspend sets activation_status to rejected."""
    from dewie.api.routes.admin import suspend_user

    user_id = uuid.uuid4()
    updated_user = {
        "id": user_id,
        "email": "user@example.com",
        "name": "User",
        "is_admin": False,
        "activation_status": "rejected",
        "has_password": True,
        "created_at": "2026-01-01",
        "last_login_at": None,
    }
    mock_pg = AsyncMock()
    mock_pg.update_local_user = AsyncMock(return_value=updated_user)
    req = _make_request(pg=mock_pg)

    result = await suspend_user(user_id, req)
    assert result.activation_status == "rejected"
    mock_pg.update_local_user.assert_awaited_once_with(
        user_id=user_id, activation_status="rejected"
    )


@pytest.mark.asyncio
async def test_suspend_user_not_found():
    """POST /admin/users/{id}/suspend returns 404 when user doesn't exist."""
    from fastapi import HTTPException

    from dewie.api.routes.admin import suspend_user

    mock_pg = AsyncMock()
    mock_pg.update_local_user = AsyncMock(return_value=None)
    req = _make_request(pg=mock_pg)

    with pytest.raises(HTTPException) as exc_info:
        await suspend_user(uuid.uuid4(), req)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_unsuspend_user_success():
    """POST /admin/users/{id}/unsuspend sets activation_status to approved."""
    from dewie.api.routes.admin import unsuspend_user

    user_id = uuid.uuid4()
    updated_user = {
        "id": user_id,
        "email": "user@example.com",
        "name": "User",
        "is_admin": False,
        "activation_status": "approved",
        "has_password": True,
        "created_at": "2026-01-01",
        "last_login_at": None,
    }
    mock_pg = AsyncMock()
    mock_pg.update_local_user = AsyncMock(return_value=updated_user)
    req = _make_request(pg=mock_pg)

    result = await unsuspend_user(user_id, req)
    assert result.activation_status == "approved"
    mock_pg.update_local_user.assert_awaited_once_with(
        user_id=user_id, activation_status="approved"
    )


@pytest.mark.asyncio
async def test_unsuspend_user_not_found():
    from fastapi import HTTPException

    from dewie.api.routes.admin import unsuspend_user

    mock_pg = AsyncMock()
    mock_pg.update_local_user = AsyncMock(return_value=None)
    req = _make_request(pg=mock_pg)

    with pytest.raises(HTTPException) as exc_info:
        await unsuspend_user(uuid.uuid4(), req)
    assert exc_info.value.status_code == 404


# ── User document management endpoints ────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_user_documents_success():
    """GET /admin/users/{id}/documents returns docs owned by the user."""
    from dewie.api.routes.admin import list_user_documents

    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    mock_pg = AsyncMock()
    mock_pg.get_user_documents = AsyncMock(
        return_value=[
            {
                "id": doc_id,
                "title": "Test Doc",
                "source_url": "https://example.com/doc",
                "status": "complete",
                "corpus_id": uuid.uuid4(),
                "owner_user_id": str(user_id),
                "created_at": "2026-01-01",
            }
        ]
    )
    req = _make_request(pg=mock_pg)
    result = await list_user_documents(user_id, req)
    assert len(result) == 1
    assert result[0].id == str(doc_id)
    assert result[0].title == "Test Doc"
    mock_pg.get_user_documents.assert_awaited_once_with(user_id=user_id, limit=100, offset=0)


@pytest.mark.asyncio
async def test_list_user_documents_empty():
    """GET /admin/users/{id}/documents returns empty list when no docs."""
    from dewie.api.routes.admin import list_user_documents

    mock_pg = AsyncMock()
    mock_pg.get_user_documents = AsyncMock(return_value=[])
    req = _make_request(pg=mock_pg)
    result = await list_user_documents(uuid.uuid4(), req)
    assert result == []


@pytest.mark.asyncio
async def test_delete_user_document_success():
    """DELETE /admin/users/{id}/documents/{doc_id} hard-deletes a user document."""
    from dewie.api.routes.admin import delete_user_document

    mock_pg = AsyncMock()
    mock_pg.delete_user_document = AsyncMock(return_value=True)
    req = _make_request(pg=mock_pg)
    result = await delete_user_document(uuid.uuid4(), uuid.uuid4(), req)
    assert result is None  # 204


@pytest.mark.asyncio
async def test_delete_user_document_not_found():
    """DELETE /admin/users/{id}/documents/{doc_id} returns 404 when not found."""
    from fastapi import HTTPException

    from dewie.api.routes.admin import delete_user_document

    mock_pg = AsyncMock()
    mock_pg.delete_user_document = AsyncMock(return_value=False)
    req = _make_request(pg=mock_pg)

    with pytest.raises(HTTPException) as exc_info:
        await delete_user_document(uuid.uuid4(), uuid.uuid4(), req)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_disconnect_user_document_success():
    """POST /admin/users/{id}/documents/{doc_id}/disconnect clears owner_user_id."""
    from dewie.api.routes.admin import disconnect_user_document

    mock_pg = AsyncMock()
    mock_pg.disconnect_user_document = AsyncMock(return_value=True)
    req = _make_request(pg=mock_pg)
    result = await disconnect_user_document(uuid.uuid4(), uuid.uuid4(), req)
    assert result is None  # 204


@pytest.mark.asyncio
async def test_disconnect_user_document_not_found():
    """POST /admin/users/{id}/documents/{doc_id}/disconnect returns 404 when not found."""
    from fastapi import HTTPException

    from dewie.api.routes.admin import disconnect_user_document

    mock_pg = AsyncMock()
    mock_pg.disconnect_user_document = AsyncMock(return_value=False)
    req = _make_request(pg=mock_pg)

    with pytest.raises(HTTPException) as exc_info:
        await disconnect_user_document(uuid.uuid4(), uuid.uuid4(), req)
    assert exc_info.value.status_code == 404
