"""E2E tests for the /query endpoint pipeline.

Covers the full request/response cycle including:
- Auth middleware (disabled by default, enabled for specific tests)
- Rate limiting
- Hybrid search with mocked postgres
- Response schema validation (AQ must NOT appear)
- Caching (second call served from Redis cache)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dewie.models.content import ContentDocument, ContentStatus


@pytest.fixture(autouse=True)
def _disable_rate_limiting(monkeypatch):
    """Disable the module-level limiter so inline decorator checks don't rate-limit tests."""
    from dewie.api.middleware import limiter

    monkeypatch.setattr(limiter, "enabled", False)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_doc(url: str = "https://example.com/article") -> ContentDocument:
    doc = ContentDocument(
        url=url,
        title="Test Article",
        status=ContentStatus.READY,
        summary="A great article about testing.",
        keywords=["testing", "pytest", "coverage"],
        entities=["GitHub", "Python"],
        topics=["software engineering"],
        answers_questions=["How do you test Python code?"],
        embed_summary="testing pytest coverage",
    )
    return doc


def _make_cache(results=None):
    cache = AsyncMock()
    # First call — cache miss; second call would return cached value
    cache.get_query_result = AsyncMock(return_value=None)
    cache.set_query_result = AsyncMock()
    cache.get_tenant_plan = AsyncMock(return_value="free")
    cache.set_tenant_plan = AsyncMock()
    cache.incr_quota = AsyncMock(return_value=1)
    cache.decr_quota = AsyncMock()
    return cache


def _make_pg(docs_with_scores=None):
    pg = AsyncMock()
    if docs_with_scores is None:
        docs_with_scores = [(_make_doc(), 0.95)]
    pg.search = AsyncMock(return_value=docs_with_scores)
    pg.get_edge_counts = AsyncMock(return_value={})
    pg.get_tenant_plan = AsyncMock(return_value="free")
    pg.log_access = AsyncMock()
    pg.enqueue_search = AsyncMock(return_value=(False, None))
    # Return a plain dict so chunk_matches.get() works correctly
    pg.search_chunks_for_docs = AsyncMock(return_value={})

    # pg._session_factory() must return an async context manager yielding a session
    mock_session = AsyncMock()
    # _edge_counts calls session.execute(select(...)) and returns rows
    mock_result = MagicMock()
    mock_result.all.return_value = []  # no edges for test docs
    mock_session.execute = AsyncMock(return_value=mock_result)

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    pg._session_factory = MagicMock(return_value=session_cm)

    return pg


def _make_query_app(pg=None, cache=None):
    from dewie.api.middleware import limiter
    from dewie.api.routes.query import router

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router)

    app.state.postgres = pg or _make_pg()
    app.state.cache = cache or _make_cache()
    return app


# ── Basic query pipeline ───────────────────────────────────────────────────────


class TestQueryPipeline:
    def test_query_returns_200_with_results(self):
        app = _make_query_app()
        client = TestClient(app)
        resp = client.post("/query", json={"query": "how to test Python code"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) >= 1

    def test_query_missing_body_returns_422(self):
        app = _make_query_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_query_empty_string_returns_422(self):
        app = _make_query_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 422

    def test_aq_not_in_response(self):
        """answers_questions must NEVER be present in any API response (trade secret)."""
        app = _make_query_app()
        client = TestClient(app)
        resp = client.post("/query", json={"query": "how to test Python code"})
        assert resp.status_code == 200

        # Check the raw JSON body — AQ must not appear anywhere
        body_text = resp.text
        assert "answers_questions" not in body_text

    def test_result_fields_present(self):
        """Core search result fields must be present."""
        app = _make_query_app()
        client = TestClient(app)
        resp = client.post("/query", json={"query": "pytest testing"})
        assert resp.status_code == 200

        results = resp.json()["results"]
        assert len(results) >= 1
        result = results[0]

        assert "doc_id" in result
        assert "url" in result
        assert "title" in result
        assert "score" in result
        # AQ must not be present
        assert "answers_questions" not in result

    def test_empty_corpus_returns_empty_results(self):
        pg = _make_pg(docs_with_scores=[])
        app = _make_query_app(pg=pg)
        client = TestClient(app)
        resp = client.post("/query", json={"query": "nothing here"})
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_query_with_limit(self):
        pg = _make_pg(
            docs_with_scores=[
                (_make_doc(f"https://example.com/{i}"), 0.9 - i * 0.1) for i in range(5)
            ]
        )
        app = _make_query_app(pg=pg)
        client = TestClient(app)
        resp = client.post("/query", json={"query": "test", "limit": 3})
        assert resp.status_code == 200

    def test_cache_miss_then_hit(self):
        """Second identical query should be served from cache without calling pg.search again."""
        pg = _make_pg()
        cache = _make_cache()

        cached_payload = {
            "results": [],
            "query": "cached query",
            "latency_ms": 10,
            "result_confidence": None,
            "total": 0,
        }
        call_count = {"n": 0}

        async def _get_cached(key, kind):
            call_count["n"] += 1
            if call_count["n"] > 1:
                return cached_payload
            return None

        cache.get_query_result = _get_cached

        app = _make_query_app(pg=pg, cache=cache)
        client = TestClient(app)

        # First request → cache miss → hits pg.search
        client.post("/query", json={"query": "cached query"})
        # Second request → cache hit → pg.search must NOT be called again
        client.post("/query", json={"query": "cached query"})

        assert pg.search.call_count == 1, (
            f"Expected pg.search called once (cache miss), got {pg.search.call_count}"
        )

    def test_rankers_endpoint_returns_list(self):
        app = _make_query_app()
        client = TestClient(app)
        resp = client.get("/query/rankers")
        assert resp.status_code == 200
        rankers = resp.json()
        assert isinstance(rankers, list)
        assert len(rankers) > 0
        assert any(r.get("id") == "rrf" for r in rankers)


# ── Quota enforcement ──────────────────────────────────────────────────────────


