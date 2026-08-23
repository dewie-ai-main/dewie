"""Tests for uncovered query.py endpoint paths.

Covers:
- GET /query/benchmark — benchmark_rerank endpoint
- POST /query/category — category query endpoint
- Reranking and staleness penalty paths in POST /query
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dewie.models.content import ContentDocument, ContentStatus


@pytest.fixture(autouse=True)
def _disable_rate_limiting(monkeypatch):
    from dewie.api.middleware import limiter

    monkeypatch.setattr(limiter, "enabled", False)


def _make_doc(url: str = "https://example.com/article", edge_count: int = 5) -> ContentDocument:
    return ContentDocument(
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


def _make_cache():
    cache = AsyncMock()
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
    pg.get_edge_count = AsyncMock(return_value=3)
    pg.get_edge_counts = AsyncMock(return_value={})
    pg.get_tenant_plan = AsyncMock(return_value="free")
    pg.log_access = AsyncMock()
    pg.enqueue_search = AsyncMock(return_value=(False, None))
    pg.search_chunks_for_docs = AsyncMock(return_value={})

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    pg._session_factory = MagicMock(return_value=session_cm)

    # Mock engine.connect() for query_id fetch
    conn = AsyncMock()
    qid_result = MagicMock()
    qid_result.fetchone.return_value = None
    conn.execute = AsyncMock(return_value=qid_result)
    conn_cm = MagicMock()
    conn_cm.__aenter__ = AsyncMock(return_value=conn)
    conn_cm.__aexit__ = AsyncMock(return_value=None)
    pg._engine = MagicMock()
    pg._engine.connect.return_value = conn_cm

    return pg


def _make_app(pg=None, cache=None):
    from dewie.api.middleware import limiter
    from dewie.api.routes.query import router

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router)
    app.state.postgres = pg or _make_pg()
    app.state.cache = cache or _make_cache()
    return app


# ── GET /query/benchmark ──────────────────────────────────────────────────────


class TestBenchmarkRerank:
    def test_benchmark_returns_200(self):
        """GET /query/benchmark returns 200 with standard and reranked results."""
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/query/benchmark?q=test+query")
        assert resp.status_code == 200

    def test_benchmark_response_has_required_fields(self):
        """GET /query/benchmark response contains comparison, standard_results, reranked_results."""
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/query/benchmark?q=test+query")
        data = resp.json()
        assert "query" in data
        assert "standard_results" in data
        assert "reranked_results" in data
        assert "comparison" in data

    def test_benchmark_with_chunk_scores(self):
        """GET /query/benchmark with chunk matches annotates reranked results."""
        doc = _make_doc()
        pg = _make_pg(docs_with_scores=[(doc, 0.9)])
        pg.search_chunks_for_docs = AsyncMock(
            return_value={
                str(doc.id): {"chunk_index": 0, "text": "matching chunk text", "score": 0.85}
            }
        )
        app = _make_app(pg=pg)
        client = TestClient(app)
        resp = client.get("/query/benchmark?q=test+query&limit=5")
        assert resp.status_code == 200

    def test_benchmark_empty_corpus(self):
        """GET /query/benchmark with empty results returns empty lists."""
        pg = _make_pg(docs_with_scores=[])
        app = _make_app(pg=pg)
        client = TestClient(app)
        resp = client.get("/query/benchmark?q=test+query")
        assert resp.status_code == 200
        data = resp.json()
        assert data["standard_results"] == []

    def test_benchmark_query_required(self):
        """GET /query/benchmark without q param returns 422."""
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/query/benchmark")
        assert resp.status_code == 422

    def test_benchmark_multi_doc_comparison(self):
        """GET /query/benchmark with multiple docs builds a comparison table."""
        docs = [_make_doc(f"https://example.com/doc{i}") for i in range(3)]
        pg = _make_pg(docs_with_scores=[(d, 0.9 - i * 0.1) for i, d in enumerate(docs)])
        app = _make_app(pg=pg)
        client = TestClient(app)
        resp = client.get("/query/benchmark?q=test+query")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["comparison"]) >= 1


# ── POST /query/category ──────────────────────────────────────────────────────


class TestCategoryQuery:
    def test_category_query_returns_200(self):
        """POST /query/category returns 200 with category distribution."""
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/query/category", json={"query": "test query"})
        assert resp.status_code == 200

    def test_category_response_has_results(self):
        """POST /query/category response includes results."""
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/query/category", json={"query": "test query"})
        data = resp.json()
        assert "results" in data
        assert "query" in data

    def test_category_query_aq_not_exposed(self):
        """POST /query/category must not expose answers_questions in results."""
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/query/category", json={"query": "test query"})
        data = resp.json()
        for result in data.get("results", []):
            assert "answers_questions" not in result

    def test_category_query_with_rerank(self):
        """POST /query/category with rerank=true applies chunk reranking."""
        pg = _make_pg()
        doc = _make_doc()
        pg.search = AsyncMock(return_value=[(doc, 0.9)])
        pg.search_chunks_for_docs = AsyncMock(
            return_value={str(doc.id): {"chunk_index": 0, "text": "relevant chunk", "score": 0.88}}
        )
        app = _make_app(pg=pg)
        client = TestClient(app)
        resp = client.post("/query/category?rerank=true", json={"query": "test query"})
        assert resp.status_code == 200

    def test_category_query_empty_corpus(self):
        """POST /query/category with empty corpus returns empty results."""
        pg = _make_pg(docs_with_scores=[])
        app = _make_app(pg=pg)
        client = TestClient(app)
        resp = client.post("/query/category", json={"query": "test query"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []

    def test_category_query_required(self):
        """POST /query/category without query field returns 422."""
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/query/category", json={})
        assert resp.status_code == 422


# ── POST /query with reranking paths ─────────────────────────────────────────


class TestQueryWithRerank:
    def test_query_with_rerank_flag(self):
        """POST /query with rerank=true goes through chunk reranking."""
        pg = _make_pg()
        doc = _make_doc()
        pg.search = AsyncMock(return_value=[(doc, 0.9)])
        pg.search_chunks_for_docs = AsyncMock(
            return_value={str(doc.id): {"chunk_index": 0, "text": "chunk text", "score": 0.8}}
        )
        app = _make_app(pg=pg)
        client = TestClient(app)
        resp = client.post("/query?rerank=true", json={"query": "test query"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"][0]["chunk_match"] == "chunk text"

    def test_query_with_staleness_penalty(self):
        """POST /query with staleness_penalty=true applies staleness sorting."""
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/query", json={"query": "test query", "staleness_penalty": True})
        assert resp.status_code == 200

    def test_query_results_with_multiple_docs(self):
        """POST /query with multiple docs triggers result_confidence computation."""
        docs = [_make_doc(f"https://example.com/{i}") for i in range(3)]
        # Docs with varied topics to trigger distributed confidence path
        docs[0].topics = ["tech", "ai"]
        docs[1].topics = ["finance", "market"]
        docs[2].topics = ["sports", "game"]
        docs[0].answers_questions = ["What is AI?"]

        pg = _make_pg(docs_with_scores=[(d, 0.9 - i * 0.3) for i, d in enumerate(docs)])
        app = _make_app(pg=pg)
        client = TestClient(app)
        resp = client.post("/query", json={"query": "artificial intelligence research"})
        assert resp.status_code == 200
        data = resp.json()
        assert "result_confidence" in data


class _FakeRemoteResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            req = httpx.Request("POST", "https://remote.invalid/query")
            resp = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError("remote error", request=req, response=resp)

    def json(self):
        return self._payload


def _remote_result_payload() -> dict:
    return {
        "query": "remote",
        "results": [
            {
                "doc_type": "other",
                "doc_id": str(uuid.uuid4()),
                "title": "Remote Result",
                "summary": "from remote",
                "url": "https://remote.example/doc",
                "source": "remote",
                "topics": [],
                "keywords": [],
                "entities": [],
                "sentiment": 0.0,
                "score": 1.0,
                "edge_count": 0,
            }
        ],
    }


class TestQueryRemoteSourceRouting:
    def test_mcp_query_uses_api_query_first(self, monkeypatch):
        called_urls: list[str] = []

        class FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json=None, headers=None, follow_redirects=True):  # noqa: A002
                called_urls.append(url)
                if url.endswith("/api/query"):
                    return _FakeRemoteResponse(200, _remote_result_payload())
                return _FakeRemoteResponse(404, {"detail": "not found"})

        monkeypatch.setattr("httpx.AsyncClient", lambda *args, **kwargs: FakeAsyncClient())

        pg = _make_pg(docs_with_scores=[])
        pg.get_source = AsyncMock(
            return_value={
                "id": str(uuid.uuid4()),
                "type": "mcp",
                "config": {"endpoint": "https://dewie.example", "api_key": "abc"},
            }
        )
        app = _make_app(pg=pg)
        client = TestClient(app)

        source_id = str(uuid.uuid4())
        resp = client.post("/query", json={"query": "remote", "limit": 3, "source_id": source_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_id"] == source_id
        assert data["total"] == 1
        assert called_urls == ["https://dewie.example/api/query"]

    def test_mcp_query_falls_back_to_legacy_query_path(self, monkeypatch):
        called_urls: list[str] = []

        class FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json=None, headers=None, follow_redirects=True):  # noqa: A002
                called_urls.append(url)
                if url.endswith("/api/query"):
                    return _FakeRemoteResponse(404, {"detail": "not found"})
                if url.endswith("/query"):
                    return _FakeRemoteResponse(200, _remote_result_payload())
                return _FakeRemoteResponse(404, {"detail": "not found"})

        monkeypatch.setattr("httpx.AsyncClient", lambda *args, **kwargs: FakeAsyncClient())

        pg = _make_pg(docs_with_scores=[])
        pg.get_source = AsyncMock(
            return_value={
                "id": str(uuid.uuid4()),
                "type": "mcp",
                "config": {"endpoint": "https://dewie.example", "api_key": "abc"},
            }
        )
        app = _make_app(pg=pg)
        client = TestClient(app)

        resp = client.post(
            "/query",
            json={"query": "remote", "limit": 3, "source_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 1
        assert called_urls == [
            "https://dewie.example/api/query",
            "https://dewie.example/query",
        ]

    def test_source_query_does_not_use_local_cache_entry(self, monkeypatch):
        called_urls: list[str] = []

        class FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json=None, headers=None, follow_redirects=True):  # noqa: A002
                called_urls.append(url)
                return _FakeRemoteResponse(200, _remote_result_payload())

        monkeypatch.setattr("httpx.AsyncClient", lambda *args, **kwargs: FakeAsyncClient())

        local_cache_payload = {
            "query": "iran",
            "results": [],
            "total": 0,
            "fallback_triggered": True,
            "gap_enrichment_queued": False,
            "location_results": None,
            "source_id": None,
        }

        cache = _make_cache()

        async def _cache_get(key, kind):  # noqa: ARG001
            # Simulate existing local-cache hit for non-source queries only.
            if "::source:" in key:
                return None
            return local_cache_payload

        cache.get_query_result = AsyncMock(side_effect=_cache_get)

        pg = _make_pg(docs_with_scores=[])
        src_id = str(uuid.uuid4())
        pg.get_source = AsyncMock(
            return_value={
                "id": src_id,
                "type": "mcp",
                "config": {"endpoint": "https://dewie.example", "api_key": "abc"},
            }
        )

        app = _make_app(pg=pg, cache=cache)
        client = TestClient(app)
        resp = client.post(
            "/query",
            json={"query": "iran", "limit": 5, "source_id": src_id},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["source_id"] == src_id
        assert data["total"] == 1
        assert called_urls == ["https://dewie.example/api/query"]
