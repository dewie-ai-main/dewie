"""Tests for dewie.storage.postgres utility functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── _embed_dimensions_for_model ───────────────────────────────────────────────


def test_embed_dims_3small():
    from dewie.storage.postgres import _embed_dimensions_for_model

    assert _embed_dimensions_for_model("text-embedding-3-small") == 1536


def test_embed_dims_3large():
    from dewie.storage.postgres import _embed_dimensions_for_model

    assert _embed_dimensions_for_model("text-embedding-3-large") == 3072


def test_embed_dims_embeddinggemma():
    from dewie.storage.postgres import _embed_dimensions_for_model

    assert _embed_dimensions_for_model("ggml-org/embeddinggemma-300m-qat-q8_0-GGUF") == 768
    assert _embed_dimensions_for_model("google/embeddinggemma-300m") == 768


def test_embed_dims_ada():
    from dewie.storage.postgres import _embed_dimensions_for_model

    assert _embed_dimensions_for_model("text-embedding-ada-002") == 1536


def test_embed_dims_nomic():
    from dewie.storage.postgres import _embed_dimensions_for_model

    assert _embed_dimensions_for_model("nomic-embed-text") == 768


def test_embed_dims_default():
    from dewie.storage.postgres import _embed_dimensions_for_model

    assert _embed_dimensions_for_model("unknown-model") == 1536


def test_embed_dims_qwen3_embedding_8b():
    from dewie.storage.postgres import _embed_dimensions_for_model

    assert _embed_dimensions_for_model("Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M") == 4096


def test_embed_dims_env_override(monkeypatch):
    from dewie.storage.postgres import _embed_dimensions_for_model

    monkeypatch.setenv("EMBED_DIMENSIONS", "2048")
    assert _embed_dimensions_for_model("any-model") == 2048
    monkeypatch.delenv("EMBED_DIMENSIONS")


def test_embed_dims_output_dimensions_override(monkeypatch):
    from dewie.storage.postgres import _embed_dimensions_for_model

    monkeypatch.setenv("EMBED_OUTPUT_DIMENSIONS", "1024")
    assert _embed_dimensions_for_model("Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M") == 1024
    monkeypatch.delenv("EMBED_OUTPUT_DIMENSIONS")


# ── _expand_query_with_session ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_expand_query_empty_query():
    from dewie.storage.postgres import _expand_query_with_session

    session = AsyncMock()
    result = await _expand_query_with_session("", session)
    assert result == ""


@pytest.mark.asyncio
async def test_expand_query_adds_alternate_terms():
    from dewie.storage.postgres import _expand_query_with_session

    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = [("basketball",), ("NBA",)]
    session.execute = AsyncMock(return_value=mock_result)

    result = await _expand_query_with_session("NBA scores", session)
    assert "NBA scores" in result
    assert "basketball" in result


@pytest.mark.asyncio
async def test_expand_query_no_new_terms():
    from dewie.storage.postgres import _expand_query_with_session

    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    session.execute = AsyncMock(return_value=mock_result)

    result = await _expand_query_with_session("machine learning", session)
    assert result == "machine learning"


@pytest.mark.asyncio
async def test_expand_query_skips_existing_terms():
    from dewie.storage.postgres import _expand_query_with_session

    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = [("learning",)]  # already in query
    session.execute = AsyncMock(return_value=mock_result)

    result = await _expand_query_with_session("machine learning", session)
    # "learning" already in query — not re-added
    assert result.count("learning") == 1


@pytest.mark.asyncio
async def test_expand_query_handles_exception():
    from dewie.storage.postgres import _expand_query_with_session

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=Exception("db error"))

    result = await _expand_query_with_session("some query", session)
    assert result == "some query"


# ── _get_embedding ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_embedding_returns_vector():
    from dewie.storage.postgres import _get_embedding

    mock_provider = AsyncMock()
    mock_provider.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    with patch("dewie.providers.factory.get_embedding_provider", return_value=mock_provider):
        result = await _get_embedding("test query")

    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_get_embedding_returns_none_on_non_200():
    from dewie.storage.postgres import _get_embedding

    mock_provider = AsyncMock()
    mock_provider.embed = AsyncMock(return_value=None)

    with patch("dewie.providers.factory.get_embedding_provider", return_value=mock_provider):
        result = await _get_embedding("test query 401")

    assert result is None


@pytest.mark.asyncio
async def test_get_embedding_returns_none_on_exception():
    from dewie.storage.postgres import _get_embedding

    with patch("dewie.providers.factory.get_embedding_provider", side_effect=Exception("oops")):
        result = await _get_embedding("query")

    assert result is None


@pytest.mark.asyncio
async def test_get_embedding_uses_openrouter():
    from dewie.storage.postgres import _get_embedding

    mock_provider = AsyncMock()
    mock_provider.embed = AsyncMock(return_value=[[0.5]])

    with patch("dewie.providers.factory.get_embedding_provider", return_value=mock_provider) as p:
        result = await _get_embedding("q")

    assert p.called
    assert result == [0.5]


# ── .env.local override (Issue #197) ─────────────────────────────────────────


def test_env_local_overrides_env_for_postgres_dsn(tmp_path, monkeypatch):
    """
    .env.local must take precedence over .env when both are present.
    This is the root cause of Issue #197: dev service falls back to the
    localhost default because .env.local is not loaded.
    """
    import os

    from dewie.config import Settings

    # Real env vars outrank env files — scrub any leaked DSN first.
    monkeypatch.delenv("POSTGRES_DSN", raising=False)

    # Write a .env with a default-style DSN
    env_file = tmp_path / ".env"
    env_file.write_text("POSTGRES_DSN=postgresql+asyncpg://dewie:dewie@localhost:5432/dewie\n")

    # Write a .env.local with the production DSN (gitignored, higher priority)
    env_local_file = tmp_path / ".env.local"
    env_local_file.write_text(
        "POSTGRES_DSN=postgresql+asyncpg://dewie:dewie@prod-host:5432/dewie?ssl=disable\n"
    )

    # Temporarily change cwd so relative env_file paths resolve correctly
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        s = Settings(
            _env_file=(str(env_file), str(env_local_file)),  # type: ignore[call-arg]
        )
        assert "prod-host" in s.postgres_dsn, (
            ".env.local should override .env — dev service won't connect if it doesn't"
        )
    finally:
        os.chdir(orig_cwd)


def test_env_local_missing_falls_back_to_env(tmp_path, monkeypatch):
    """When .env.local doesn't exist, .env values are used normally."""
    import os

    from dewie.config import Settings

    monkeypatch.delenv("POSTGRES_DSN", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "POSTGRES_DSN=postgresql+asyncpg://dewie:dewie@myhost:5432/dewie\n"
    )

    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # .env.local does not exist — should gracefully fall back to .env
        s = Settings(
            _env_file=(str(env_file), str(tmp_path / ".env.local")),  # type: ignore[call-arg]
        )
        assert "myhost" in s.postgres_dsn
    finally:
        os.chdir(orig_cwd)


# ── check_db_credentials ──────────────────────────────────────────────────────


def test_default_password_triggers_warning(monkeypatch, caplog):
    """POSTGRES_PASSWORD=changeme → warning logged."""
    import logging

    monkeypatch.setenv("POSTGRES_PASSWORD", "changeme")

    from dewie.storage.postgres import check_db_credentials

    strong_dsn = "postgresql+asyncpg://dewie:V3ry$ecre7P@ss!@localhost:5432/dewie"
    with caplog.at_level(logging.WARNING):
        check_db_credentials(strong_dsn)

    assert any("SECURITY WARNING" in rec.message for rec in caplog.records)
    monkeypatch.delenv("POSTGRES_PASSWORD")


def test_strong_password_no_warning(monkeypatch, caplog):
    """POSTGRES_PASSWORD=Str0ng!Pass → no warning."""
    import logging

    monkeypatch.setenv("POSTGRES_PASSWORD", "Str0ng!Pass")

    from dewie.storage.postgres import check_db_credentials

    # Pass a strong DSN explicitly to avoid the module-level settings default
    strong_dsn = "postgresql+asyncpg://dewie:V3ry$ecre7P@ss!@localhost:5432/dewie"
    with caplog.at_level(logging.WARNING):
        check_db_credentials(strong_dsn)

    assert not any("SECURITY WARNING" in rec.message for rec in caplog.records)
    monkeypatch.delenv("POSTGRES_PASSWORD")


def test_empty_password_triggers_warning(monkeypatch, caplog):
    """Empty POSTGRES_PASSWORD → warning logged."""
    import logging

    monkeypatch.setenv("POSTGRES_PASSWORD", "")

    from dewie.storage.postgres import check_db_credentials

    strong_dsn = "postgresql+asyncpg://dewie:V3ry$ecre7P@ss!@localhost:5432/dewie"
    with caplog.at_level(logging.WARNING):
        check_db_credentials(strong_dsn)

    assert any("SECURITY WARNING" in rec.message for rec in caplog.records)
    monkeypatch.delenv("POSTGRES_PASSWORD")


def test_dsn_password_triggers_warning(monkeypatch, caplog):
    """Weak password embedded in DSN → warning logged."""
    import logging

    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    from dewie.storage.postgres import check_db_credentials

    with caplog.at_level(logging.WARNING):
        check_db_credentials("postgresql+asyncpg://dewie:postgres@localhost:5432/dewie")

    assert any("SECURITY WARNING" in rec.message for rec in caplog.records)


def test_dsn_strong_password_no_warning(monkeypatch, caplog):
    """Strong password in DSN → no warning."""
    import logging

    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    from dewie.storage.postgres import check_db_credentials

    strong_dsn = "postgresql+asyncpg://dewie:V3ry$ecre7P@ss!@localhost:5432/dewie"
    with caplog.at_level(logging.WARNING):
        check_db_credentials(strong_dsn)

    assert not any("SECURITY WARNING" in rec.message for rec in caplog.records)


def test_weak_passwords_set_contains_expected_values():
    """Verify the WEAK_PASSWORDS set contains the documented defaults."""
    from dewie.storage.postgres import WEAK_PASSWORDS

    expected = {"dewie", "postgres", "password", "admin", "secret", ""}
    assert expected.issubset(WEAK_PASSWORDS)


def test_dsn_case_insensitive_check(monkeypatch, caplog):
    """Password 'POSTGRES' in DSN should still be flagged (case-insensitive)."""
    import logging

    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    from dewie.storage.postgres import check_db_credentials

    with caplog.at_level(logging.WARNING):
        check_db_credentials("postgresql+asyncpg://dewie:POSTGRES@localhost:5432/dewie")

    assert any("SECURITY WARNING" in rec.message for rec in caplog.records)
