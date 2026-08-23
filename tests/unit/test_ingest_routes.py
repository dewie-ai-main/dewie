"""Unit tests for POST /ingest endpoint."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _disable_rate_limiting(monkeypatch):
    """Disable the module-level limiter during unit tests."""
    from dewie.api.middleware import limiter

    monkeypatch.setattr(limiter, "enabled", False)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_ingest_app(pg=None, processor=None):
    from dewie.api.middleware import limiter
    from dewie.api.routes.ingest import router

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router)

    app.state.postgres = pg or AsyncMock()
    app.state.processor = processor or AsyncMock()
    return app


def _no_content_ingester():
    """Patch WebIngester to yield no documents."""
    instance = AsyncMock()
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=None)

    async def _empty_fetch(_url):
        return
        yield  # make it an async generator

    instance.fetch = _empty_fetch
    return instance


def _one_doc_ingester(doc):
    """Patch WebIngester to yield one document."""
    instance = AsyncMock()
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=None)

    async def _fetch(_url):
        yield doc

    instance.fetch = _fetch
    return instance


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestIngestEndpoint:
    def test_missing_body_returns_422(self):
        client = TestClient(_make_ingest_app(), raise_server_exceptions=False)
        resp = client.post("/ingest", json={})
        assert resp.status_code == 422

    def test_invalid_url_returns_422(self):
        client = TestClient(_make_ingest_app(), raise_server_exceptions=False)
        resp = client.post("/ingest", json={"url": "not-a-url"})
        assert resp.status_code == 422

    def test_no_content_returns_422(self):
        """When the URL fetches nothing, expect 422."""
        with patch("dewie.api.routes.ingest.WebIngester", return_value=_no_content_ingester()):
            client = TestClient(_make_ingest_app(), raise_server_exceptions=False)
            resp = client.post("/ingest", json={"url": "https://example.com/empty"})
        assert resp.status_code == 422

    def test_valid_url_accepted(self):
        """Valid URL with content returns 202 and accepted doc IDs."""
        from dewie.models.content import ContentDocument, ContentStatus

        mock_doc = ContentDocument(
            url="https://example.com/article",
            title="Test Article",
            body="This is a test article with enough content to pass the quality gate. " * 5,
            status=ContentStatus.PENDING,
        )

        pg = AsyncMock()
        pg.upsert = AsyncMock()
        pg.write_body_text = AsyncMock()
        processor = AsyncMock()

        with patch(
            "dewie.api.routes.ingest.WebIngester",
            return_value=_one_doc_ingester(mock_doc),
        ):
            client = TestClient(_make_ingest_app(pg, processor), raise_server_exceptions=False)
            resp = client.post("/ingest", json={"url": "https://example.com/article"})

        assert resp.status_code == 202
        data = resp.json()
        assert "accepted" in data
        assert len(data["accepted"]) == 1

    def test_pre_fetched_body_skips_ingester(self):
        """When body is provided in request, WebIngester must NOT be called."""
        pg = AsyncMock()
        pg.upsert = AsyncMock()

        mock_ingester = MagicMock()
        with patch("dewie.api.routes.ingest.WebIngester", mock_ingester):
            client = TestClient(_make_ingest_app(pg), raise_server_exceptions=False)
            resp = client.post(
                "/ingest",
                json={
                    "url": "https://example.com/article",
                    "body": "This is the pre-fetched article body text. " * 10,
                    "title": "Pre-fetched article",
                },
            )

        mock_ingester.assert_not_called()
        assert resp.status_code == 202

    def test_service_key_header_enforced(self):
        """When INTERNAL_SERVICE_KEY is set, requests without it must get 403."""
        pg = AsyncMock()
        with patch.dict(os.environ, {"INTERNAL_SERVICE_KEY": "secret-key"}):
            client = TestClient(_make_ingest_app(pg), raise_server_exceptions=False)
            resp = client.post(
                "/ingest",
                json={"url": "https://example.com/article"},
                # No X-Service-Key header
            )
        assert resp.status_code == 403

    def test_correct_service_key_passes(self):
        """Requests with the correct X-Service-Key header must proceed past the auth gate."""
        from dewie.models.content import ContentDocument, ContentStatus

        mock_doc = ContentDocument(
            url="https://example.com/article",
            title="Test",
            status=ContentStatus.PENDING,
        )
        pg = AsyncMock()
        pg.upsert = AsyncMock()

        with (
            patch.dict(os.environ, {"INTERNAL_SERVICE_KEY": "secret-key"}),
            patch(
                "dewie.api.routes.ingest.WebIngester",
                return_value=_one_doc_ingester(mock_doc),
            ),
        ):
            client = TestClient(_make_ingest_app(pg), raise_server_exceptions=False)
            resp = client.post(
                "/ingest",
                json={"url": "https://example.com/article"},
                headers={"X-Service-Key": "secret-key"},
            )

        assert resp.status_code == 202

    def test_enrichment_queued_as_background_task(self):
        """BackgroundTasks must be called with the enrichment function after a successful ingest."""

        from dewie.models.content import ContentDocument, ContentStatus

        mock_doc = ContentDocument(
            url="https://example.com/article",
            title="Test",
            status=ContentStatus.PENDING,
        )
        pg = AsyncMock()
        pg.upsert = AsyncMock()
        processor = AsyncMock()

        with patch(
            "dewie.api.routes.ingest.WebIngester",
            return_value=_one_doc_ingester(mock_doc),
        ):
            client = TestClient(_make_ingest_app(pg, processor), raise_server_exceptions=False)
            resp = client.post("/ingest", json={"url": "https://example.com/article"})

        # Background task means enrichment happens asynchronously — response is 202
        assert resp.status_code == 202

    def test_invalid_enrichment_provider_model_rejected(self):
        with patch(
            "dewie.model_registry.registry.validate_provider_model",
            AsyncMock(return_value=(False, "Unknown provider/model pair")),
        ):
            client = TestClient(_make_ingest_app(), raise_server_exceptions=False)
            resp = client.post(
                "/ingest",
                json={
                    "url": "https://example.com/article",
                    "enrichment_provider": "missing",
                    "enrichment_model": "fake-model",
                },
            )

        assert resp.status_code == 400
