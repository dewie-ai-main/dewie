"""
Integration tests for the HTTP API layer using FastAPI's TestClient.

Storage backends are replaced with mocks so no running services are needed.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

from dewie.main import app
from dewie.models.content import ContentDocument, ContentStatus
from dewie.models.query import SearchResponse, SearchResult


def _make_search_response(query: str = "test", count: int = 1) -> SearchResponse:
    results = [
        SearchResult(
            doc_id=str(uuid.uuid4()),
            title=f"Doc {i}",
            summary="A test summary.",
            url=f"https://example.com/{i}",
            source="example.com",
            topics=["ai"],
            keywords=["test"],
            entities=[],
            sentiment=0.1,
            answers_questions=[],
            score=1.0 / (i + 1),
            edge_count=3,
        )
        for i in range(count)
    ]
    return SearchResponse(query=query, results=results, total=count)


@pytest.fixture
def client():
    """TestClient with mocked postgres attached to app state."""
    mock_pg = AsyncMock()
    mock_pg.search.return_value = []
    mock_pg.get_edge_count.return_value = 0

    mock_cache = AsyncMock()
    mock_cache.get_query_result.return_value = None
    mock_cache.set_query_result.return_value = None

    with TestClient(app, raise_server_exceptions=True) as c:
        app.state.postgres = mock_pg
        app.state.cache = mock_cache
        yield c, mock_pg


class TestQueryEndpoint:
    def test_post_query_returns_200(self, client):
        c, pg = client
        resp = c.post("/query", json={"query": "OpenAI GPT"})
        assert resp.status_code == 200

    def test_response_shape(self, client):
        c, pg = client
        data = c.post("/query", json={"query": "OpenAI GPT"}).json()
        assert "results" in data
        assert "total" in data
        assert "query" in data

    def test_empty_query_rejected(self, client):
        c, _ = client
        resp = c.post("/query", json={"query": ""})
        assert resp.status_code == 422

    def test_query_too_long_rejected(self, client):
        c, _ = client
        resp = c.post("/query", json={"query": "x" * 501})
        assert resp.status_code == 422

    def test_limit_param_passed_to_pg(self, client):
        c, pg = client
        c.post("/query", json={"query": "AI", "limit": 5})
        pg.search.assert_called_once()
        call_kwargs = pg.search.call_args
        assert call_kwargs[1].get("limit") == 5 or call_kwargs[0][1] == 5


class TestQueryRerank:
    def test_rerank_false_no_behavior_change(self, client):
        """rerank=false must behave identically to omitting the parameter."""
        c, pg = client
        resp = c.post("/query?rerank=false", json={"query": "AI"})
        assert resp.status_code == 200
        # rerank=false → pg.search called with body.limit (default 10), not 20
        pg.search.assert_called_once()
        call_kwargs = pg.search.call_args
        limit_arg = call_kwargs[1].get("limit") or call_kwargs[0][1]
        assert limit_arg == 10

    def test_rerank_true_fetches_candidate_pool(self, client):
        """rerank=true must fetch _RERANK_CANDIDATE_COUNT=20 candidates from pg.search."""
        c, pg = client
        resp = c.post("/query?rerank=true", json={"query": "AI"})
        assert resp.status_code == 200
        pg.search.assert_called_once()
        call_kwargs = pg.search.call_args
        limit_arg = call_kwargs[1].get("limit") or call_kwargs[0][1]
        assert limit_arg == 20

    def test_rerank_true_calls_search_chunks_for_docs(self, client):
        """rerank=true must call search_chunks_for_docs to get chunk scores."""
        c, pg = client
        doc_id = str(uuid.uuid4())
        mock_doc = ContentDocument(
            id=doc_id,
            url="https://example.com",
            title="Test Doc",
            status=ContentStatus.READY,
            topics=["ai"],
            keywords=["test"],
            entities=[],
        )
        pg.search.return_value = [(mock_doc, 0.9)]
        pg.search_chunks_for_docs = AsyncMock(
            return_value={
                doc_id: {"chunk_index": 0, "text": "relevant chunk text", "score": 0.85},
            }
        )
        resp = c.post("/query?rerank=true", json={"query": "AI"})
        assert resp.status_code == 200
        pg.search_chunks_for_docs.assert_called_once()
        data = resp.json()
        assert data["results"][0]["chunk_match"] == "relevant chunk text"
        assert data["results"][0]["chunk_score"] == pytest.approx(0.85)

    def test_rerank_true_result_trimmed_to_limit(self, client):
        """After reranking, results must be trimmed to body.limit."""
        c, pg = client
        # Return 20 docs from pg.search (simulating the candidate pool)
        docs = [
            ContentDocument(
                id=str(uuid.uuid4()),
                url=f"https://example.com/{i}",
                title=f"Doc {i}",
                status=ContentStatus.READY,
                topics=["ai"],
                keywords=["test"],
                entities=[],
            )
            for i in range(20)
        ]
        pg.search.return_value = [(doc, 0.5) for doc in docs]
        pg.search_chunks_for_docs = AsyncMock(return_value={})
        resp = c.post("/query?rerank=true", json={"query": "AI", "limit": 5})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) <= 5

    def test_rerank_true_ranks_by_chunk_score(self, client):
        """Docs with higher chunk scores should rank above docs with lower chunk scores."""
        c, pg = client
        doc_low = ContentDocument(
            id=str(uuid.uuid4()),
            url="https://low.com",
            title="Low Chunk Score",
            status=ContentStatus.READY,
            topics=["ai"],
            keywords=[],
            entities=[],
        )
        doc_high = ContentDocument(
            id=str(uuid.uuid4()),
            url="https://high.com",
            title="High Chunk Score",
            status=ContentStatus.READY,
            topics=["ai"],
            keywords=[],
            entities=[],
        )
        # doc_low comes first in doc-level search but doc_high has a better chunk score.
        pg.search.return_value = [(doc_low, 0.9), (doc_high, 0.8)]
        pg.search_chunks_for_docs = AsyncMock(
            return_value={
                str(doc_low.id): {"chunk_index": 0, "text": "low chunk", "score": 0.3},
                str(doc_high.id): {"chunk_index": 0, "text": "high chunk", "score": 0.95},
            }
        )
        resp = c.post("/query?rerank=true", json={"query": "AI"})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert results[0]["title"] == "High Chunk Score"
        assert results[1]["title"] == "Low Chunk Score"

    def test_chunks_true_no_rerank_populates_chunk_match_only(self, client):
        """chunks=true without rerank should populate chunk_match but not chunk_score."""
        c, pg = client
        doc_id = str(uuid.uuid4())
        mock_doc = ContentDocument(
            id=doc_id,
            url="https://example.com",
            title="Test Doc",
            status=ContentStatus.READY,
            topics=["ai"],
            keywords=[],
            entities=[],
        )
        pg.search.return_value = [(mock_doc, 0.9)]
        pg.search_chunks_for_docs = AsyncMock(
            return_value={
                doc_id: {"chunk_index": 0, "text": "chunk text", "score": 0.77},
            }
        )
        resp = c.post("/query?chunks=true", json={"query": "AI"})
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["chunk_match"] == "chunk text"
        assert result.get("chunk_score") is None  # no chunk_score when only chunks=true


class TestBenchmarkEndpoint:
    def test_benchmark_returns_200(self, client):
        c, _ = client
        resp = c.get("/query/benchmark?q=AI")
        assert resp.status_code == 200

    def test_benchmark_response_shape(self, client):
        c, _ = client
        data = c.get("/query/benchmark?q=AI").json()
        assert "query" in data
        assert "limit" in data
        assert "standard_results" in data
        assert "reranked_results" in data
        assert "comparison" in data

    def test_benchmark_missing_query_returns_422(self, client):
        c, _ = client
        resp = c.get("/query/benchmark")
        assert resp.status_code == 422

    def test_benchmark_query_too_long_rejected(self, client):
        c, _ = client
        resp = c.get(f"/query/benchmark?q={'x' * 501}")
        assert resp.status_code == 422

    def test_benchmark_comparison_has_rank_fields(self, client):
        """Each comparison row must have standard_rank, reranked_rank, and rank_change."""
        c, pg = client
        doc_id = str(uuid.uuid4())
        mock_doc = ContentDocument(
            id=doc_id,
            url="https://example.com",
            title="Test Doc",
            status=ContentStatus.READY,
            topics=["ai"],
            keywords=[],
            entities=[],
        )
        pg.search.return_value = [(mock_doc, 0.8)]
        pg.search_chunks_for_docs = AsyncMock(
            return_value={
                doc_id: {"chunk_index": 0, "text": "chunk", "score": 0.7},
            }
        )
        data = c.get("/query/benchmark?q=AI&limit=5").json()
        assert data["limit"] == 5
        for row in data["comparison"]:
            assert "standard_rank" in row
            assert "reranked_rank" in row
            assert "rank_change" in row
            assert "doc_score" in row


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
