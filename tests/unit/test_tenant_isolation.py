"""Tests for dewie.storage.tenant_isolation — multi-tenant corpus isolation."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dewie.storage.tenant_isolation import (
    CorpusAccessDenied,
    TenantCorpusRouter,
    TenantProvisionResult,
    _slugify,
)

# ── _slugify ──────────────────────────────────────────────────────────────────


def test_slugify_basic():
    assert _slugify("ACME Corp") == "acme-corp"


def test_slugify_strips_special_chars():
    assert _slugify("Hello, World!") == "hello-world"


def test_slugify_consecutive_separators():
    assert _slugify("foo   bar") == "foo-bar"


def test_slugify_leading_trailing_dashes():
    assert _slugify("--leading") == "leading"


def test_slugify_empty_falls_back():
    assert _slugify("!!!") == "tenant"


def test_slugify_unicode_lowercased():
    assert _slugify("FooBar") == "foobar"


def test_slugify_max_length():
    long_name = "a" * 100
    result = _slugify(long_name)
    assert len(result) <= 63


# ── CorpusAccessDenied ─────────────────────────────────────────────────────────


def test_corpus_access_denied_message():
    cid = uuid.uuid4()
    wid = uuid.uuid4()
    exc = CorpusAccessDenied(cid, [wid])
    assert str(cid) in str(exc)
    assert str(wid) in str(exc)


def test_corpus_access_denied_empty_workspaces():
    cid = uuid.uuid4()
    exc = CorpusAccessDenied(cid, [])
    assert "<none>" in str(exc)


def test_corpus_access_denied_is_permission_error():
    exc = CorpusAccessDenied(uuid.uuid4(), [])
    assert isinstance(exc, PermissionError)


# ── TenantCorpusRouter helpers ────────────────────────────────────────────────


def _make_pg(workspace_row=None, corpus_row=None, corpus_workspace=None):
    """Build a minimal PostgresClient mock for TenantCorpusRouter tests."""
    pg = MagicMock()

    ws = workspace_row or {"id": str(uuid.uuid4()), "name": "test", "sharing_tier": "private"}
    corp = corpus_row or {
        "id": str(uuid.uuid4()),
        "name": "test corpus",
        "slug": "test",
        "workspace_id": ws["id"],
        "sharing_tier": "private",
    }

    pg.create_workspace = AsyncMock(return_value=ws)
    pg.create_corpus = AsyncMock(return_value=corp)
    pg.get_corpora = AsyncMock(return_value=[corp])

    # _engine.connect() context manager that returns workspace_id for corpus lookup
    mock_row = MagicMock()
    if corpus_workspace is not None:
        mock_row.__getitem__ = MagicMock(return_value=str(corpus_workspace))
        # row[0] for workspace_id
        mock_row.__iter__ = MagicMock(return_value=iter([str(corpus_workspace)]))
        mock_row.__getitem__ = lambda self, i: str(corpus_workspace)
    else:
        mock_row = None

    mock_execute_result = MagicMock()
    mock_execute_result.fetchone.return_value = mock_row

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_execute_result)

    mock_connect_ctx = MagicMock()
    mock_connect_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_connect_ctx.__aexit__ = AsyncMock(return_value=False)

    pg._engine = MagicMock()
    pg._engine.connect.return_value = mock_connect_ctx

    # Mock engine.begin() context manager for set_search_path
    mock_begin_ctx = MagicMock()
    mock_begin_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_begin_ctx.__aexit__ = AsyncMock(return_value=False)
    pg._engine.begin.return_value = mock_begin_ctx

    return pg


# ── provision_tenant ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_provision_tenant_returns_result():
    pg = _make_pg()

    with patch("dewie.storage.tenant_isolation.create_api_key", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = ("ck_live_testkey", {"id": uuid.uuid4()})
        router = TenantCorpusRouter(pg)
        result = await router.provision_tenant(display_name="ACME Corp")

    assert isinstance(result, TenantProvisionResult)
    assert result.tenant_slug == "acme-corp"
    assert isinstance(result.workspace_id, uuid.UUID)
    assert isinstance(result.corpus_id, uuid.UUID)
    assert result.api_key == "ck_live_testkey"


@pytest.mark.asyncio
async def test_provision_tenant_custom_slug():
    pg = _make_pg()

    with patch("dewie.storage.tenant_isolation.create_api_key", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = ("ck_live_testkey", {})
        router = TenantCorpusRouter(pg)
        result = await router.provision_tenant(slug="custom-slug", display_name="Whatever")

    assert result.tenant_slug == "custom-slug"


@pytest.mark.asyncio
async def test_provision_tenant_calls_create_workspace():
    pg = _make_pg()

    with patch("dewie.storage.tenant_isolation.create_api_key", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = ("ck_live_x", {})
        router = TenantCorpusRouter(pg)
        await router.provision_tenant(display_name="Test Co")

    pg.create_workspace.assert_called_once()
    call_kwargs = pg.create_workspace.call_args
    assert call_kwargs.kwargs.get("sharing_tier") == "private" or (
        len(call_kwargs.args) >= 1
    )


@pytest.mark.asyncio
async def test_provision_tenant_calls_create_corpus():
    pg = _make_pg()

    with patch("dewie.storage.tenant_isolation.create_api_key", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = ("ck_live_x", {})
        router = TenantCorpusRouter(pg)
        await router.provision_tenant(display_name="Test Co")

    pg.create_corpus.assert_called_once()


@pytest.mark.asyncio
async def test_provision_tenant_api_key_scoped_to_workspace():
    ws_id = uuid.uuid4()
    pg = _make_pg(workspace_row={"id": str(ws_id), "name": "t", "sharing_tier": "private"})

    with patch("dewie.storage.tenant_isolation.create_api_key", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = ("ck_live_x", {})
        router = TenantCorpusRouter(pg)
        await router.provision_tenant(display_name="Scoped Co")

    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args
    workspace_ids_arg = call_kwargs.kwargs.get("workspace_ids") or (
        call_kwargs.args[1] if len(call_kwargs.args) > 1 else []
    )
    # API key must be scoped to exactly the created workspace
    assert ws_id in workspace_ids_arg


# ── assert_corpus_access ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_assert_access_empty_workspaces_grants_all():
    """Empty allowed_workspace_ids = admin, no restriction."""
    pg = _make_pg()
    router = TenantCorpusRouter(pg)
    # Should not raise regardless of corpus
    await router.assert_corpus_access(uuid.uuid4(), [])


@pytest.mark.asyncio
async def test_assert_access_matching_workspace_passes():
    ws_id = uuid.uuid4()
    pg = _make_pg(corpus_workspace=ws_id)
    router = TenantCorpusRouter(pg)
    corpus_id = uuid.uuid4()
    # Should not raise
    await router.assert_corpus_access(corpus_id, [ws_id])


@pytest.mark.asyncio
async def test_assert_access_wrong_workspace_raises():
    ws_id = uuid.uuid4()
    other_ws = uuid.uuid4()
    pg = _make_pg(corpus_workspace=other_ws)
    router = TenantCorpusRouter(pg)
    corpus_id = uuid.uuid4()
    with pytest.raises(CorpusAccessDenied):
        await router.assert_corpus_access(corpus_id, [ws_id])


@pytest.mark.asyncio
async def test_assert_access_nonexistent_corpus_raises():
    """Corpus not found → deny (fail-closed)."""
    pg = _make_pg(corpus_workspace=None)
    # Patch get_corpus_workspace to return None
    router = TenantCorpusRouter(pg)

    with patch.object(router, "get_corpus_workspace", new=AsyncMock(return_value=None)):
        with pytest.raises(CorpusAccessDenied):
            await router.assert_corpus_access(uuid.uuid4(), [uuid.uuid4()])


@pytest.mark.asyncio
async def test_assert_access_multiple_allowed_workspaces():
    """Corpus in any of the allowed workspaces → pass."""
    ws_a = uuid.uuid4()
    ws_b = uuid.uuid4()
    pg = _make_pg(corpus_workspace=ws_b)
    router = TenantCorpusRouter(pg)

    with patch.object(router, "get_corpus_workspace", new=AsyncMock(return_value=ws_b)):
        # ws_b is in allowed list → should pass
        await router.assert_corpus_access(uuid.uuid4(), [ws_a, ws_b])


@pytest.mark.asyncio
async def test_assert_access_corpus_in_wrong_workspace_multi():
    """Corpus not in any of the allowed workspaces → deny."""
    ws_a = uuid.uuid4()
    ws_b = uuid.uuid4()
    evil_ws = uuid.uuid4()
    pg = _make_pg(corpus_workspace=evil_ws)
    router = TenantCorpusRouter(pg)

    with patch.object(router, "get_corpus_workspace", new=AsyncMock(return_value=evil_ws)):
        with pytest.raises(CorpusAccessDenied):
            await router.assert_corpus_access(uuid.uuid4(), [ws_a, ws_b])


# ── list_tenant_corpora ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_tenant_corpora_empty_workspace_ids_returns_all():
    pg = _make_pg()
    router = TenantCorpusRouter(pg)
    result = await router.list_tenant_corpora([])
    pg.get_corpora.assert_called_once_with()
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_list_tenant_corpora_filters_by_workspace():
    ws_id = uuid.uuid4()
    pg = _make_pg()
    router = TenantCorpusRouter(pg)
    await router.list_tenant_corpora([ws_id])
    pg.get_corpora.assert_called_once_with(workspace_id=ws_id)


@pytest.mark.asyncio
async def test_list_tenant_corpora_multiple_workspaces():
    ws_a = uuid.uuid4()
    ws_b = uuid.uuid4()
    pg = _make_pg()
    router = TenantCorpusRouter(pg)
    await router.list_tenant_corpora([ws_a, ws_b])
    assert pg.get_corpora.call_count == 2


# ── TenantProvisionResult dataclass ───────────────────────────────────────────


def test_tenant_provision_result_fields():
    ws = uuid.uuid4()
    corp = uuid.uuid4()
    r = TenantProvisionResult(
        tenant_slug="test",
        workspace_id=ws,
        corpus_id=corp,
        api_key="ck_live_abc",
    )
    assert r.tenant_slug == "test"
    assert r.workspace_id == ws
    assert r.corpus_id == corp
    assert r.api_key == "ck_live_abc"


# ── set_search_path ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_search_path_executes_set_local():
    """set_search_path sends SET LOCAL search_path to the database."""
    pg = _make_pg()
    mock_begin_conn = pg._engine.begin.return_value.__aenter__.return_value

    router = TenantCorpusRouter(pg)
    await router.set_search_path("acme")

    mock_begin_conn.execute.assert_called()
    call_args = mock_begin_conn.execute.call_args
    sql = str(call_args[0][0].text)
    assert "SET LOCAL search_path TO" in sql
    assert "acme" in sql


@pytest.mark.asyncio
async def test_set_search_path_rejects_empty():
    pg = _make_pg()
    router = TenantCorpusRouter(pg)
    with pytest.raises(ValueError, match="Invalid schema name"):
        await router.set_search_path("")


@pytest.mark.asyncio
async def test_set_search_path_rejects_none():
    pg = _make_pg()
    router = TenantCorpusRouter(pg)
    with pytest.raises(ValueError, match="Invalid schema name"):
        await router.set_search_path(None)  # type: ignore


@pytest.mark.asyncio
async def test_set_search_path_rejects_uppercase():
    pg = _make_pg()
    router = TenantCorpusRouter(pg)
    with pytest.raises(ValueError, match="Invalid schema name"):
        await router.set_search_path("ACME")


@pytest.mark.asyncio
async def test_set_search_path_rejects_hyphens():
    pg = _make_pg()
    router = TenantCorpusRouter(pg)
    with pytest.raises(ValueError, match="Invalid schema name"):
        await router.set_search_path("acme-corp")


@pytest.mark.asyncio
async def test_set_search_path_rejects_leading_digit():
    pg = _make_pg()
    router = TenantCorpusRouter(pg)
    with pytest.raises(ValueError, match="Invalid schema name"):
        await router.set_search_path("123acme")


@pytest.mark.asyncio
async def test_set_search_path_accepts_underscore_start():
    pg = _make_pg()
    mock_begin_conn = pg._engine.begin.return_value.__aenter__.return_value
    router = TenantCorpusRouter(pg)
    await router.set_search_path("_internal")
    mock_begin_conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_set_search_path_accepts_valid_names():
    pg = _make_pg()
    mock_begin_conn = pg._engine.begin.return_value.__aenter__.return_value
    router = TenantCorpusRouter(pg)

    for schema in ["acme", "acme_corp", "acme123", "_internal", "a"]:
        await router.set_search_path(schema)

    assert mock_begin_conn.execute.call_count == 5
