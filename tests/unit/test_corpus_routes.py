"""Unit tests for dewie.api.routes.corpus — corpus sources and quality endpoints."""

from __future__ import annotations

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_corpus_app(pg=None):
    """Build a minimal FastAPI app with the corpus router and mock DB."""
    from dewie.api.middleware import limiter
    from dewie.api.routes.corpus import router

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router)
    app.state.postgres = pg or AsyncMock()
    return app


def _make_pipeline_app(pg=None):
    """Build a minimal FastAPI app with the pipeline router and mock DB."""
    from dewie.api.middleware import limiter
    from dewie.api.routes.pipeline import router

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router)
    app.state.postgres = pg or AsyncMock()
    return app


# ── Corpus sources tests ──────────────────────────────────────────────────────


class TestCorpusSources:
    """Tests for GET /api/corpus/sources."""

    def test_returns_source_list(self):
        """When corpus_sources_cache has rows, return them ordered by name."""
        mock_result = MagicMock()
        mock_result.mappings.return_value.fetchall.return_value = [
            {"name": "alpha", "count": 10},
            {"name": "beta", "count": 5},
        ]

        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(return_value=mock_result)

        mock_engine = AsyncMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)

        pg = AsyncMock()
        pg._engine = mock_engine

        app = _make_corpus_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/corpus/sources")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        assert body[0]["name"] == "alpha"
        assert body[1]["name"] == "beta"

    def test_returns_empty_list_when_no_sources(self):
        """When corpus_sources_cache is empty, return an empty list."""
        mock_result = MagicMock()
        mock_result.mappings.return_value.fetchall.return_value = []

        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(return_value=mock_result)

        mock_engine = AsyncMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)

        pg = AsyncMock()
        pg._engine = mock_engine

        app = _make_corpus_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/corpus/sources")

        assert resp.status_code == 200
        assert resp.json() == []

    def test_500_on_database_error(self):
        """When the DB query raises, return 500."""
        pg = AsyncMock()
        pg._engine.connect = AsyncMock(side_effect=RuntimeError("connection lost"))

        app = _make_corpus_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/corpus/sources")

        assert resp.status_code == 500


# ── Corpus quality tests ──────────────────────────────────────────────────────


class TestCorpusQuality:
    """Tests for GET /pipeline/corpus/quality."""

    def _build_mock_result(self, mappings):
        """Helper to create a mock SQLAlchemy result with mappings."""
        mock_result = MagicMock()
        mock_result.mappings.return_value.one.return_value = mappings
        return mock_result

    def _build_mock_conn(self, result, is_sqlite=False):
        """Helper to build a mock connection with the given result."""
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        async def mock_execute(stmt):
            # Check if it's a SELECT from corpus_quality_cache
            stmt_str = str(stmt) if hasattr(stmt, '__str__') else ''
            if 'corpus_quality_cache' in stmt_str:
                return result
            if 'corpus_sources_cache' in stmt_str:
                mock_sources = MagicMock()
                mock_sources.mappings.return_value.all.return_value = []
                return mock_sources
            if 'document_chunks' in stmt_str:
                mock_chunks = MagicMock()
                mock_chunks.mappings.return_value.one.return_value = {
                    "docs_with_chunks": 0,
                    "total_chunks": 0,
                    "chunks_with_embed": 0,
                    "chunks_with_aq_embed": 0,
                    "avg_chunks_per_doc": 0,
                    "avg_chunk_tokens": 0,
                    "avg_chunk_chars": 0,
                    "docs_with_aq_embed": 0,
                }
                return mock_chunks
            if 'chunk_status' in stmt_str:
                mock_status = MagicMock()
                mock_status.mappings.return_value.one.return_value = {
                    "status_none": 0,
                    "status_chunked": 0,
                    "status_skipped": 0,
                    "status_failed": 0,
                }
                return mock_status
            return result

        mock_conn.execute = mock_execute

        mock_engine = AsyncMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)

        pg = AsyncMock()
        pg._engine = mock_engine
        pg._is_sqlite = is_sqlite
        return pg

    def test_returns_quality_metrics_postgres(self):
        """Corpus quality endpoint returns summary, quality_distribution, and by_source."""
        summary = {
            "total": 100,
            "ready": 80,
            "pending": 15,
            "failed": 5,
            "with_embedding": 90,
            "embed_summary_good": 75,
            "embed_summary_stub": 5,
            "embed_summary_none": 10,
            "avg_embed_summary_len": 500,
            "avg_body_len": 2000,
            "avg_aqs": 3.5,
            "with_aqs": 60,
            "empty_aqs": 40,
            "quality_high": 50,
            "quality_medium": 20,
            "quality_low": 15,
            "quality_stub": 15,
            "refreshed_at": "2024-01-01T00:00:00Z",
        }

        mock_result = MagicMock()
        mock_result.mappings.return_value.one.return_value = summary

        pg = self._build_mock_conn(mock_result)

        app = _make_pipeline_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/pipeline/corpus/quality")

        assert resp.status_code == 200
        body = resp.json()

        assert body["summary"]["total"] == 100
        assert body["summary"]["ready"] == 80
        assert body["quality_distribution"]["high"] == 50
        assert body["quality_distribution"]["medium"] == 20
        assert body["quality_distribution"]["low"] == 15
        assert body["quality_distribution"]["stub"] == 15
        assert body["refreshed_at"] == "2024-01-01T00:00:00Z"

    def test_returns_quality_metrics_sqlite(self):
        """SQLite path runs live aggregate queries and returns same shape."""
        summary = {
            "total": 50,
            "ready": 40,
            "pending": 8,
            "failed": 2,
            "with_embedding": 45,
            "embed_summary_good": 35,
            "embed_summary_stub": 3,
            "embed_summary_none": 5,
            "avg_embed_summary_len": 300,
            "avg_body_len": 1500,
            "avg_aqs": 2.0,
            "with_aqs": 30,
            "empty_aqs": 20,
            "quality_high": 30,
            "quality_medium": 10,
            "quality_low": 7,
            "quality_stub": 3,
            "refreshed_at": None,
        }

        mock_result = MagicMock()
        mock_result.mappings.return_value.one.return_value = summary

        pg = self._build_mock_conn(mock_result, is_sqlite=True)

        app = _make_pipeline_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/pipeline/corpus/quality")

        assert resp.status_code == 200
        body = resp.json()

        assert body["summary"]["total"] == 50
        assert body["quality_distribution"]["high"] == 30
        assert body["refreshed_at"] is None

    def test_returns_500_on_query_failure(self):
        """When the database query fails, return 500 with error detail."""
        pg = AsyncMock()
        pg._is_sqlite = False
        pg._engine.connect = AsyncMock(side_effect=RuntimeError("query failed"))

        app = _make_pipeline_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/pipeline/corpus/quality")

        assert resp.status_code == 500
        assert "Query failed" in resp.json()["detail"]


# ── Corpus quality refresh tests ──────────────────────────────────────────────


class TestCorpusQualityRefresh:
    """Tests for POST /pipeline/corpus/quality/refresh."""

    def _build_mock_begin_conn(self):
        """Build a mock connection for BEGIN context manager."""
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(return_value=MagicMock())
        return mock_conn

    def test_refresh_succeeds(self):
        """Successful refresh returns ok=True."""
        mock_conn = self._build_mock_begin_conn()
        mock_engine = AsyncMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        pg = AsyncMock()
        pg._engine = mock_engine
        pg._is_sqlite = False

        app = _make_pipeline_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/pipeline/corpus/quality/refresh")

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert "refreshed" in body["message"].lower()

    def test_refresh_returns_500_on_error(self):
        """Database error during refresh returns 500."""
        mock_conn = self._build_mock_begin_conn()
        mock_engine = AsyncMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)
        mock_conn.execute = AsyncMock(side_effect=RuntimeError("refresh failed"))

        pg = AsyncMock()
        pg._engine = mock_engine
        pg._is_sqlite = False

        app = _make_pipeline_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/pipeline/corpus/quality/refresh")

        assert resp.status_code == 500


# ── Corpus export tests ───────────────────────────────────────────────────────


class TestCorpusExport:
    """Tests for GET /corpus/export."""

    def _build_mock_session(self, rows):
        """Build a mock async session with the given rows."""
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = rows

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        return mock_session

    def _build_mock_pg(self, batches, is_sqlite=False):
        """Build a mock PostgresClient that returns documents in batches.

        The generator calls `pg._session_factory()` which should return
        an async context manager yielding a session with `execute()`.
        """
        batch_idx = [0]

        def make_factory():
            if batch_idx[0] < len(batches):
                rows = batches[batch_idx[0]]
                batch_idx[0] += 1
                return self._build_mock_session(rows)
            return self._build_mock_session([])

        # _session_factory is called with () and returns an async context manager
        # It's NOT an async function — just a regular function returning a CM
        def session_factory():
            return make_factory()

        pg = AsyncMock()
        pg._session_factory = session_factory
        pg._is_sqlite = is_sqlite

        return pg

    def test_jsonl_export_returns_streaming_response(self):
        """JSONL export returns a StreamingResponse with application/jsonl media type."""
        rows = [
            {
                "id": "doc-1",
                "url": "https://example.com/1",
                "title": "Doc 1",
                "summary": "Summary 1",
                "topics": '["ai", "ml"]',
                "keywords": '["test"]',
                "entities": '["OrgA"]',
                "published_at": None,
                "enriched_at": None,
                "corpus_id": "test-corpus",
            }
        ]
        pg = self._build_mock_pg([rows])

        app = _make_corpus_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/corpus/export", params={"corpus_id": "test-corpus", "format": "jsonl"})

        assert resp.status_code == 200
        assert "application/jsonl" in resp.headers["content-type"]
        assert "corpus_test-corpus.jsonl" in resp.headers["content-disposition"]

    def test_jsonl_export_contains_expected_fields(self):
        """JSONL export contains only allowed fields."""
        rows = [
            {
                "id": "doc-1",
                "url": "https://example.com/1",
                "title": "Test Title",
                "summary": "Test Summary",
                "topics": '["topic-a", "topic-b"]',
                "keywords": '["kw-1", "kw-2"]',
                "entities": '["EntityX"]',
                "published_at": None,
                "enriched_at": None,
                "corpus_id": "test-corpus",
            }
        ]
        pg = self._build_mock_pg([rows])

        app = _make_corpus_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/corpus/export", params={"corpus_id": "test-corpus", "format": "jsonl"})

        assert resp.status_code == 200
        body = resp.text
        line = body.strip().split("\n")[0]
        import json

        doc = json.loads(line)
        expected_keys = {
            "id", "url", "title", "summary", "topics", "keywords",
            "entities", "published_at", "enriched_at", "corpus_id", "tags",
        }
        assert set(doc.keys()) == expected_keys
        assert "answers_questions" not in doc
        assert doc["topics"] == ["topic-a", "topic-b"]
        assert doc["keywords"] == ["kw-1", "kw-2"]
        assert doc["entities"] == ["EntityX"]
        assert doc["tags"] == []

    def test_jsonl_export_handles_null_summary(self):
        """Null summary is converted to empty string."""
        rows = [
            {
                "id": "doc-1",
                "url": "https://example.com/1",
                "title": "No Summary",
                "summary": None,
                "topics": None,
                "keywords": None,
                "entities": None,
                "published_at": None,
                "enriched_at": None,
                "corpus_id": "test-corpus",
            }
        ]
        pg = self._build_mock_pg([rows])

        app = _make_corpus_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/corpus/export", params={"corpus_id": "test-corpus", "format": "jsonl"})

        assert resp.status_code == 200
        body = resp.text.strip().split("\n")[0]
        import json

        doc = json.loads(body)
        assert doc["summary"] == ""
        assert doc["topics"] == []
        assert doc["keywords"] == []
        assert doc["entities"] == []

    def test_jsonl_export_with_dates(self):
        """Dates are serialized as ISO 8601 strings."""
        from datetime import datetime

        rows = [
            {
                "id": "doc-1",
                "url": "https://example.com/1",
                "title": "Dated Doc",
                "summary": "Has dates",
                "topics": None,
                "keywords": None,
                "entities": None,
                "published_at": datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC),
                "enriched_at": datetime(2024, 6, 16, 12, 0, 0, tzinfo=UTC),
                "corpus_id": "test-corpus",
            }
        ]
        pg = self._build_mock_pg([rows])

        app = _make_corpus_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/corpus/export", params={"corpus_id": "test-corpus", "format": "jsonl"})

        assert resp.status_code == 200
        body = resp.text.strip().split("\n")[0]
        import json

        doc = json.loads(body)
        assert doc["published_at"] == "2024-06-15T10:30:00+00:00"
        assert doc["enriched_at"] == "2024-06-16T12:00:00+00:00"

    def test_json_export_returns_array(self):
        """JSON format export returns a JSON array."""
        rows = [
            {
                "id": "doc-1",
                "url": "https://example.com/1",
                "title": "Doc 1",
                "summary": "Summary 1",
                "topics": '["ai"]',
                "keywords": None,
                "entities": None,
                "published_at": None,
                "enriched_at": None,
                "corpus_id": "test-corpus",
            }
        ]
        pg = self._build_mock_pg([rows])

        app = _make_corpus_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/corpus/export", params={"corpus_id": "test-corpus", "format": "json"})

        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]
        body = resp.text
        assert body.startswith("[")
        assert body.rstrip().endswith("]")
        import json

        data = json.loads(body)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_jsonl_export_empty_corpus(self):
        """Empty corpus returns an empty response."""
        pg = self._build_mock_pg([])

        app = _make_corpus_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/corpus/export", params={"corpus_id": "empty-corpus", "format": "jsonl"})

        assert resp.status_code == 200
        body = resp.text.strip()
        assert body == ""

    def test_json_export_empty_corpus(self):
        """JSON export of empty corpus returns empty array."""
        pg = self._build_mock_pg([])

        app = _make_corpus_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/corpus/export", params={"corpus_id": "empty-corpus", "format": "json"})

        assert resp.status_code == 200
        import json

        data = json.loads(resp.text)
        assert data == []

    def test_jsonl_export_batches_multiple_batches(self):
        """Export across multiple batches yields all documents."""
        batch1 = [
            {
                "id": "doc-1",
                "url": "https://example.com/1",
                "title": "Doc 1",
                "summary": "S1",
                "topics": None,
                "keywords": None,
                "entities": None,
                "published_at": None,
                "enriched_at": None,
                "corpus_id": "test-corpus",
            },
            {
                "id": "doc-2",
                "url": "https://example.com/2",
                "title": "Doc 2",
                "summary": "S2",
                "topics": None,
                "keywords": None,
                "entities": None,
                "published_at": None,
                "enriched_at": None,
                "corpus_id": "test-corpus",
            },
        ]
        batch2 = [
            {
                "id": "doc-3",
                "url": "https://example.com/3",
                "title": "Doc 3",
                "summary": "S3",
                "topics": None,
                "keywords": None,
                "entities": None,
                "published_at": None,
                "enriched_at": None,
                "corpus_id": "test-corpus",
            }
        ]
        pg = self._build_mock_pg([batch1, batch2])

        app = _make_corpus_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/corpus/export", params={"corpus_id": "test-corpus", "format": "jsonl"})

        assert resp.status_code == 200
        lines = [l for l in resp.text.strip().split("\n") if l.strip()]
        assert len(lines) == 3

    def test_jsonl_export_status_filter(self):
        """The status query parameter is accepted."""
        rows = [
            {
                "id": "doc-1",
                "url": "https://example.com/1",
                "title": "Doc 1",
                "summary": "Summary 1",
                "topics": None,
                "keywords": None,
                "entities": None,
                "published_at": None,
                "enriched_at": None,
                "corpus_id": "test-corpus",
            }
        ]
        pg = self._build_mock_pg([rows])

        app = _make_corpus_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/corpus/export",
            params={"corpus_id": "test-corpus", "format": "jsonl", "status": "ready"},
        )

        assert resp.status_code == 200

    def test_jsonl_export_default_status(self):
        """Default status filter is 'ready'."""
        rows = [
            {
                "id": "doc-1",
                "url": "https://example.com/1",
                "title": "Doc 1",
                "summary": "Summary 1",
                "topics": None,
                "keywords": None,
                "entities": None,
                "published_at": None,
                "enriched_at": None,
                "corpus_id": "test-corpus",
            }
        ]
        pg = self._build_mock_pg([rows])

        app = _make_corpus_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/corpus/export", params={"corpus_id": "test-corpus"})

        assert resp.status_code == 200


class TestExportToExportDict:
    """Tests for ContentDocument.to_export_dict()."""

    def test_exports_allowed_fields_only(self):
        """to_export_dict returns only allowed export fields."""
        from datetime import datetime

        from dewie.models.content import ContentDocument

        doc = ContentDocument(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            url="https://example.com/test",
            title="Test Document",
            summary="This is a test summary.",
            topics=["ai", "ml"],
            keywords=["test", "example"],
            entities=["OrgA", "PersonB"],
            published_at=datetime(2024, 1, 15, 8, 0, 0, tzinfo=UTC),
            enriched_at=datetime(2024, 1, 16, 9, 0, 0, tzinfo=UTC),
            corpus_id="my-corpus",
            answers_questions=["What is AI?"],
        )

        result = doc.to_export_dict()

        expected_keys = {
            "id", "url", "title", "summary", "topics", "keywords",
            "entities", "published_at", "enriched_at", "corpus_id", "tags",
        }
        assert set(result.keys()) == expected_keys
        assert "answers_questions" not in result
        assert "body" not in result

    def test_to_export_dict_excludes_answers_questions(self):
        """answers_questions must never appear in export dict."""
        from dewie.models.content import ContentDocument

        doc = ContentDocument(
            url="https://example.com",
            answers_questions=["Q1", "Q2", "Q3"],
        )

        result = doc.to_export_dict()
        assert "answers_questions" not in result

    def test_to_export_dict_serializes_dates(self):
        """Dates are serialized as ISO 8601 strings."""
        from datetime import datetime

        from dewie.models.content import ContentDocument

        doc = ContentDocument(
            url="https://example.com",
            published_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC),
            enriched_at=datetime(2024, 6, 2, 14, 30, 0, tzinfo=UTC),
        )

        result = doc.to_export_dict()
        assert result["published_at"] == "2024-06-01T12:00:00+00:00"
        assert result["enriched_at"] == "2024-06-02T14:30:00+00:00"

    def test_to_export_dict_null_dates(self):
        """Null dates produce None in export dict."""
        from dewie.models.content import ContentDocument

        doc = ContentDocument(
            url="https://example.com",
            published_at=None,
            enriched_at=None,
        )

        result = doc.to_export_dict()
        assert result["published_at"] is None
        assert result["enriched_at"] is None

    def test_to_export_dict_empty_lists(self):
        """Empty or None fields become empty lists."""
        from dewie.models.content import ContentDocument

        doc = ContentDocument(
            url="https://example.com",
            topics=[],
            keywords=[],
            entities=[],
        )

        result = doc.to_export_dict()
        assert result["topics"] == []
        assert result["keywords"] == []
        assert result["entities"] == []

    def test_to_export_dict_tags_is_empty_list(self):
        """tags is always an empty list in export."""
        from dewie.models.content import ContentDocument

        doc = ContentDocument(url="https://example.com")

        result = doc.to_export_dict()
        assert result["tags"] == []

    def test_to_export_dict_id_is_string(self):
        """id is serialized as a string."""
        from uuid import UUID

        from dewie.models.content import ContentDocument

        test_id = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        doc = ContentDocument(id=test_id, url="https://example.com")

        result = doc.to_export_dict()
        assert result["id"] == str(test_id)
        assert isinstance(result["id"], str)

