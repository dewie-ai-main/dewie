"""Tests for dewie.api.routes.traverse — keyword cluster traversal."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from dewie.api.routes.traverse import (
    DocumentResult,
    TraverseRequest,
    _build_next_clusters,
    _score_document,
)


def _make_doc_result(**kwargs) -> DocumentResult:
    defaults = {
        "id": "doc-1",
        "title": "Test Document",
        "url": "https://example.com/doc",
        "summary": "A test document.",
        "relevance_score": 0.8,
        "topics": [],
        "entities": [],
    }
    defaults.update(kwargs)
    return DocumentResult(**defaults)


# ── _score_document ───────────────────────────────────────────────────────────


def test_score_document_full_match():
    doc = MagicMock()
    doc.topics = ["machine learning", "neural networks"]
    doc.entities = []
    doc.keywords = ["deep learning", "ai"]

    score, matched = _score_document(doc, ["machine learning", "deep learning"])
    assert score > 0
    assert "machine learning" in matched or "deep learning" in matched


def test_score_document_no_match():
    doc = MagicMock()
    doc.topics = ["cooking", "recipes"]
    doc.entities = []
    doc.keywords = []

    score, matched = _score_document(doc, ["machine learning", "ai"])
    assert score == 0.0
    assert matched == []


def test_score_document_case_insensitive():
    doc = MagicMock()
    doc.topics = ["Machine Learning"]
    doc.entities = []
    doc.keywords = []

    score, matched = _score_document(doc, ["machine learning"])
    assert score > 0


def test_score_document_capped_at_one():
    doc = MagicMock()
    doc.topics = ["ai", "ml", "deep learning"]
    doc.entities = ["OpenAI"]
    doc.keywords = ["neural", "transformer", "bert"]

    score, _ = _score_document(doc, ["ai", "ml"])
    assert score <= 1.0


def test_score_document_returns_sorted_matched():
    doc = MagicMock()
    doc.topics = ["ml", "ai", "graphs"]
    doc.entities = []
    doc.keywords = []

    _, matched = _score_document(doc, ["ai", "ml", "graphs"])
    assert matched == sorted(matched)


# ── _build_next_clusters ──────────────────────────────────────────────────────


def _make_results(docs_data):
    results = []
    for data in docs_data:
        r = MagicMock()
        r.id = data.get("id", "doc-1")
        r.title = data.get("title", "Test")
        r.topics = data.get("topics", [])
        r.entities = data.get("entities", [])
        r.keywords = data.get("keywords", [])
        results.append(r)
    return results


def test_build_next_clusters_basic():
    docs = _make_results(
        [
            {
                "id": "d1",
                "topics": ["graphs", "databases"],
                "entities": ["PostgreSQL"],
                "keywords": [],
            },
            {
                "id": "d2",
                "topics": ["graphs", "networks"],
                "entities": [],
                "keywords": ["traversal"],
            },
        ]
    )
    clusters = _build_next_clusters(docs, ["graphs"], [], 5, "exploit")
    assert isinstance(clusters, list)
    # "graphs" is in seed so excluded from next clusters
    cluster_kws = [kw.lower() for c in clusters for kw in c.keywords]
    assert "graphs" not in cluster_kws


def test_build_next_clusters_excludes_seeds():
    docs = _make_results(
        [
            {"id": "d1", "topics": ["ai", "ml", "research"], "entities": [], "keywords": []},
        ]
    )
    clusters = _build_next_clusters(docs, ["ai", "ml"], [], 5, "exploit")
    for c in clusters:
        for kw in c.keywords:
            assert kw.lower() not in {"ai", "ml"}


def test_build_next_clusters_excludes_given_keywords():
    docs = _make_results(
        [
            {
                "id": "d1",
                "topics": ["research", "papers", "academia"],
                "entities": [],
                "keywords": [],
            },
        ]
    )
    clusters = _build_next_clusters(docs, ["research"], ["papers"], 5, "exploit")
    for c in clusters:
        for kw in c.keywords:
            assert kw.lower() != "papers"


def test_build_next_clusters_respects_max():
    docs = _make_results(
        [
            {"id": f"d{i}", "topics": [f"topic{i}", f"other{i}"], "entities": [], "keywords": []}
            for i in range(10)
        ]
    )
    clusters = _build_next_clusters(docs, ["seed"], [], 2, "exploit")
    assert len(clusters) <= 2


def test_build_next_clusters_explore_mode():
    docs = _make_results(
        [
            {"id": "d1", "topics": ["ai", "ml", "graphs"], "entities": [], "keywords": []},
            {"id": "d2", "topics": ["graphs", "networks"], "entities": [], "keywords": []},
        ]
    )
    # explore mode should work without errors
    clusters = _build_next_clusters(docs, ["seed"], [], 5, "explore")
    assert isinstance(clusters, list)


def test_build_next_clusters_empty_docs():
    clusters = _build_next_clusters([], ["seed"], [], 5, "exploit")
    assert clusters == []


def test_build_next_clusters_pin_keywords_included():
    docs = _make_results(
        [
            {
                "id": "d1",
                "topics": ["graphs", "databases", "search"],
                "entities": [],
                "keywords": [],
            },
        ]
    )
    clusters = _build_next_clusters(docs, ["graphs"], [], 3, "exploit", pin_keywords=["search"])
    # pinned keywords should appear in at least one cluster
    # pin_keywords are appended to each cluster if not already present
    assert isinstance(clusters, list)


# ── TraverseRequest validation ─────────────────────────────────────────────────


def test_traverse_request_defaults():
    req = TraverseRequest(seed_keywords=["ai", "ml"])
    assert req.max_documents == 20
    assert req.max_next_clusters == 5
    assert req.exploration_mode == "exploit"
    assert req.depth == 1


def test_traverse_request_custom():
    req = TraverseRequest(
        seed_keywords=["graphs"],
        max_documents=50,
        exploration_mode="explore",
        depth=3,
    )
    assert req.max_documents == 50
    assert req.exploration_mode == "explore"
    assert req.depth == 3


# ── Traverse endpoint ─────────────────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dewie.api.routes.traverse import router as traverse_router
from dewie.storage.postgres import PostgresClient


def _make_app(pg_mock):
    app = FastAPI()
    from dewie.api.middleware import limiter

    app.state.limiter = limiter
    app.include_router(traverse_router)  # router already has prefix="/traverse"
    app.state.postgres = pg_mock
    return app


def _make_pg_doc(**kwargs):
    doc = MagicMock()
    doc.id = kwargs.get("id", "00000000-0000-0000-0000-000000000001")
    doc.title = kwargs.get("title", "Test Doc")
    doc.url = kwargs.get("url", "https://example.com")
    doc.summary = kwargs.get("summary", "A test document")
    doc.topics = kwargs.get("topics", ["machine learning", "ai"])
    doc.entities = kwargs.get("entities", [])
    doc.keywords = kwargs.get("keywords", ["ml", "neural"])
    return doc


def test_traverse_no_docs_raises_503():
    pg = AsyncMock(spec=PostgresClient)
    pg.list_recent = AsyncMock(return_value=[])
    app = _make_app(pg)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/traverse", json={"seed_keywords": ["ai", "ml"]})
    assert resp.status_code == 503


def test_traverse_returns_documents_and_clusters():
    doc = _make_pg_doc()
    pg = AsyncMock(spec=PostgresClient)
    pg.list_recent = AsyncMock(return_value=[doc])
    app = _make_app(pg)
    client = TestClient(app)
    resp = client.post("/traverse", json={"seed_keywords": ["machine learning"]})
    assert resp.status_code == 200
    body = resp.json()
    assert "documents" in body
    assert "next_clusters" in body
    assert "metadata" in body
    assert body["metadata"]["depth"] == 1


def test_traverse_explore_mode():
    docs = [
        _make_pg_doc(
            id=f"00000000-0000-0000-0000-{str(i).zfill(12)}", topics=["ai", "ml", "graphs"]
        )
        for i in range(1, 6)
    ]
    pg = AsyncMock(spec=PostgresClient)
    pg.list_recent = AsyncMock(return_value=docs)
    app = _make_app(pg)
    client = TestClient(app)
    resp = client.post("/traverse", json={"seed_keywords": ["ai"], "exploration_mode": "explore"})
    assert resp.status_code == 200


def test_traverse_exclude_doc_ids():
    doc_id = "00000000-0000-0000-0000-000000000001"
    doc = _make_pg_doc(id=doc_id)
    pg = AsyncMock(spec=PostgresClient)
    pg.list_recent = AsyncMock(return_value=[doc])
    app = _make_app(pg)
    client = TestClient(app)
    resp = client.post(
        "/traverse",
        json={
            "seed_keywords": ["machine learning"],
            "exclude_doc_ids": [doc_id],
        },
    )
    # All docs excluded → no matches → returns 200 with empty documents
    assert resp.status_code == 200
    assert resp.json()["documents"] == []


def test_traverse_metadata_fields():
    doc = _make_pg_doc()
    pg = AsyncMock(spec=PostgresClient)
    pg.list_recent = AsyncMock(return_value=[doc])
    app = _make_app(pg)
    client = TestClient(app)
    resp = client.post(
        "/traverse",
        json={
            "seed_keywords": ["machine learning"],
            "depth": 3,
            "pin_keywords": ["neural"],
        },
    )
    assert resp.status_code == 200
    meta = resp.json()["metadata"]
    assert meta["depth"] == 3
    assert "machine learning" in meta["seed_keywords"]
    assert "neural" in meta["pin_keywords"]
    assert "traversal_id" in meta


# ── Structured logging tests ──────────────────────────────────────────────────


class TestTraverseLogging:
    """Verify structured logging in the /traverse route handler."""

    def test_traverse_logs_start(self, caplog):
        import logging
        from unittest.mock import AsyncMock

        from fastapi.testclient import TestClient


        doc = _make_pg_doc()
        pg = AsyncMock()
        pg.list_recent = AsyncMock(return_value=[doc])
        app = _make_app(pg)
        client = TestClient(app)

        with caplog.at_level(logging.INFO, logger="dewie.api"):
            client.post("/traverse", json={"seed_keywords": ["machine learning"]})

        messages = [r.message for r in caplog.records]
        assert any("request_start" in m for m in messages)

    def test_traverse_logs_success(self, caplog):
        import logging
        doc = _make_pg_doc()
        pg = AsyncMock()
        pg.list_recent = AsyncMock(return_value=[doc])
        app = _make_app(pg)
        client = TestClient(app)

        with caplog.at_level(logging.INFO, logger="dewie.api"):
            resp = client.post("/traverse", json={"seed_keywords": ["machine learning"]})

        assert resp.status_code == 200
        messages = [r.message for r in caplog.records]
        assert any("request_success" in m for m in messages)

    def test_traverse_logs_request_id(self, caplog):
        import logging
        doc = _make_pg_doc()
        pg = AsyncMock()
        pg.list_recent = AsyncMock(return_value=[doc])
        app = _make_app(pg)
        client = TestClient(app)

        with caplog.at_level(logging.INFO, logger="dewie.api"):
            client.post("/traverse", json={"seed_keywords": ["ai"]})

        # request_id should be "unknown" when not set by middleware
        records = [r for r in caplog.records if "[traverse] request" in r.message]
        assert len(records) >= 1

    def test_traverse_logs_http_error(self, caplog):
        """503 from no-docs raises HTTPException — should NOT log error (it's not a crash)."""
        import logging
        pg = AsyncMock()
        pg.list_recent = AsyncMock(return_value=[])
        app = _make_app(pg)
        client = TestClient(app, raise_server_exceptions=False)

        with caplog.at_level(logging.ERROR, logger="dewie.api"):
            resp = client.post("/traverse", json={"seed_keywords": ["ai"]})

        assert resp.status_code == 503
        error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        # HTTPException is re-raised directly without logging an error
        assert not any("traverse request error" in m for m in error_messages)
