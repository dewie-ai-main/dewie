"""
Shared pytest fixtures for Dewie tests.

Uses fakeredis for in-process Redis emulation; PostgreSQL
integration tests require running services (see docker-compose.yml).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import fakeredis.aioredis
import httpx
import pytest
import pytest_asyncio

from dewie.models.content import ContentDocument, ContentStatus
from dewie.storage.cache import CacheClient


@pytest.fixture(autouse=True)
def _scrub_leaked_env(monkeypatch):
    """Tests must not inherit auth state from the host environment.

    magika (imported via markitdown) calls load_dotenv() at import time and
    can plant a developer's .env into the process:
    - INTERNAL_SERVICE_KEY activates the /ingest auth gate;
    - empty-string KEY= entries defeat os.environ.setdefault in registry
      helpers, silently deregistering providers.
    Tests that need this state set it explicitly via patch.dict/monkeypatch.

    Also resets the already-loaded settings singleton to auth_enabled=True so
    .env.local (which sets AUTH_ENABLED=false for local dev) doesn't leak into
    tests that check auth enforcement. Tests wanting auth off patch it explicitly.
    """
    import dewie.config as _cfg

    monkeypatch.setattr(_cfg.settings, "auth_enabled", True)
    monkeypatch.delenv("INTERNAL_SERVICE_KEY", raising=False)
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "BRAVE_API_KEY",
        "EXA_API_KEY",
        "YOU_API_KEY",
        "CUSTOM_LLM_API_KEY",
        "JWT_SECRET",
        "ADMIN_KEY",
    ):
        if os.environ.get(var) == "":
            monkeypatch.delenv(var, raising=False)


# ── Fake storage fixtures ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def fake_cache() -> AsyncIterator[CacheClient]:
    """In-process Redis cache backed by fakeredis."""
    cache = CacheClient.__new__(CacheClient)
    cache._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    cache._ttl = 300
    yield cache
    await cache._redis.aclose()


@pytest.fixture
def mock_postgres() -> AsyncMock:
    """Mock PostgresClient with sensible defaults."""
    pg = AsyncMock()
    pg.search.return_value = []
    pg.get_by_id.return_value = None
    pg.find_by_topics.return_value = []
    pg.find_by_entities.return_value = []
    pg.find_by_keywords.return_value = []
    return pg


# ── Sample document fixtures ──────────────────────────────────────────────────


@pytest.fixture
def sample_doc() -> ContentDocument:
    return ContentDocument(
        id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        url="https://example.com/article-1",
        title="OpenAI releases new model",
        summary=(
            "OpenAI has released a new large language model that shows "
            "significant improvements in reasoning and coding tasks. "
            "The model, named GPT-5, was announced at a San Francisco event."
        ),
        source="example.com",
        status=ContentStatus.READY,
        topics=["language model", "artificial intelligence"],
        keywords=["openai", "gpt", "model", "reasoning"],
        entities=["OpenAI", "GPT-5", "San Francisco"],
        sentiment=0.3,
    )


@pytest.fixture
def related_doc() -> ContentDocument:
    return ContentDocument(
        id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
        url="https://example.com/article-2",
        title="Google DeepMind announces Gemini Ultra",
        summary=(
            "Google DeepMind has unveiled Gemini Ultra, a competing AI model "
            "with strong performance on language benchmarks."
        ),
        source="example.com",
        status=ContentStatus.READY,
        topics=["language model", "artificial intelligence"],
        keywords=["google", "deepmind", "gemini", "model"],
        entities=["Google", "DeepMind", "Gemini Ultra"],
        sentiment=0.2,
    )


# ── HTTP-level test client ────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def app_client(
    mock_postgres: AsyncMock, fake_cache: CacheClient
) -> AsyncIterator[httpx.AsyncClient]:
    """
    AsyncClient wired directly to the FastAPI app via ASGI transport.

    Both ``app.state.postgres`` and ``app.state.cache`` are replaced with
    mocks so no live services are needed.  Auth is bypassed by patching the
    middleware to inject a default dev session.
    """

    from dewie.main import app

    # Attach mocks to app state before each request
    app.state.postgres = mock_postgres
    app.state.cache = fake_cache

    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
