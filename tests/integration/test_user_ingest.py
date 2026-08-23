# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Integration tests for the user URL ingest flow.

Requires a running dev server at DEWIE_TEST_URL (default: http://localhost:10946).
Set DEWIE_TEST_API_KEY if the server has auth_enabled=True.

Run with:
    pytest tests/integration/test_user_ingest.py -v
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

BASE_URL = os.environ.get("DEWIE_TEST_URL", "http://localhost:10946")
API = f"{BASE_URL}/api"

# A stable, fast-loading public URL for ingest tests
TEST_URL = "https://en.wikipedia.org/wiki/Information_retrieval"


def _server_up() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _auth_available() -> bool:
    """Return True if the server is reachable and we have credentials (or auth is disabled)."""
    if not _server_up():
        return False
    api_key = os.environ.get("DEWIE_TEST_API_KEY", "").strip()
    # Probe the ingest endpoint — if it returns 401/403 without a key and we have no key, skip
    if not api_key:
        try:
            r = httpx.post(f"{API}/user/ingest", json={"url": "https://example.com"}, timeout=3)
            if r.status_code in (401, 403):
                return False
        except Exception:
            return False
    return True


_SKIP = pytest.mark.skipif(
    not _auth_available(),
    reason=f"Server at {BASE_URL} requires auth — set DEWIE_TEST_API_KEY",
)


@pytest.fixture(scope="module")
def client():
    headers = {}
    api_key = os.environ.get("DEWIE_TEST_API_KEY", "").strip()
    if api_key:
        headers["X-API-Key"] = api_key
    with httpx.Client(base_url=BASE_URL, timeout=30, headers=headers) as c:
        yield c


@_SKIP
def test_server_healthy(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@_SKIP
def test_ingest_returns_202_with_doc_id(client):
    r = client.post(f"{API}/user/ingest", json={"url": TEST_URL})
    assert r.status_code == 202, f"Expected 202, got {r.status_code}: {r.text}"
    body = r.json()
    assert "doc_id" in body, f"No doc_id in response: {body}"
    assert body["status"] == "pending"
    assert len(body["doc_id"]) == 36  # UUID format


@_SKIP
def test_ingest_doc_appears_in_uploads(client):
    # Ingest a URL
    r = client.post(f"{API}/user/ingest", json={"url": TEST_URL})
    assert r.status_code == 202
    doc_id = r.json()["doc_id"]

    # Poll uploads list — doc should appear within a few seconds
    for _ in range(10):
        uploads = client.get(f"{API}/user/uploads").json()
        assert isinstance(uploads, list), f"Expected list, got: {uploads}"
        if any(u["id"] == doc_id for u in uploads):
            break
        time.sleep(1)
    else:
        ids = [u["id"] for u in uploads]
        pytest.fail(f"doc_id {doc_id} never appeared in uploads after 10s. Got: {ids}")


@_SKIP
def test_ingest_doc_fetchable_by_id(client):
    r = client.post(f"{API}/user/ingest", json={"url": TEST_URL})
    assert r.status_code == 202
    doc_id = r.json()["doc_id"]

    # Doc should be fetchable immediately after ingest
    doc = client.get(f"{API}/documents/{doc_id}").json()
    assert "id" in doc or "url" in doc, f"Unexpected doc response: {doc}"


@_SKIP
def test_ingest_invalid_url_returns_error(client):
    r = client.post(f"{API}/user/ingest", json={"url": "not-a-url"})
    assert r.status_code in (422, 400), f"Expected 4xx for invalid URL, got {r.status_code}"


@_SKIP
def test_ingest_unreachable_url_returns_error(client):
    r = client.post(f"{API}/user/ingest", json={"url": "https://this-domain-does-not-exist-xyz.invalid/"})
    assert r.status_code in (422, 400, 500), f"Got {r.status_code}: {r.text}"


@_SKIP
def test_ingest_enrichment_completes(client):
    """Verify the full pipeline: ingest → enrichment → status=ready with summary and embedding."""
    r = client.post(f"{API}/user/ingest", json={"url": TEST_URL})
    assert r.status_code == 202
    doc_id = r.json()["doc_id"]

    # Poll up to 300s for enrichment to finish
    doc = None
    for _ in range(60):
        time.sleep(5)
        resp = client.get(f"{API}/documents/{doc_id}")
        if resp.status_code != 200:
            continue
        doc = resp.json()
        status = doc.get("status")
        if status in ("ready", "failed"):
            break

    assert doc is not None, "Document never became fetchable"
    assert doc.get("status") == "ready", (
        f"Expected status=ready, got {doc.get('status')!r}. "
        f"summary={bool(doc.get('summary'))}, "
        f"answers_questions={len(doc.get('answers_questions') or [])} items"
    )
    assert doc.get("summary"), "status=ready but summary is empty"
    assert doc.get("answers_questions"), "status=ready but answers_questions is empty"


@_SKIP
def test_youtube_url_ingest(client):
    """YouTube URLs need special handling — verify they don't silently succeed then vanish."""
    yt_url = "https://www.youtube.com/watch?v=ShGT-fY7S98"
    r = client.post(f"{API}/user/ingest", json={"url": yt_url})

    if r.status_code == 202:
        doc_id = r.json()["doc_id"]
        # Must actually appear in uploads
        found = False
        for _ in range(10):
            uploads = client.get(f"{API}/user/uploads").json()
            if isinstance(uploads, list) and any(u["id"] == doc_id for u in uploads):
                found = True
                break
            time.sleep(1)
        assert found, f"YouTube doc {doc_id} accepted but never appeared in uploads"
    else:
        # Explicitly rejected is also acceptable — just not silent success + disappear
        assert r.status_code in (422, 400), f"Unexpected status {r.status_code}: {r.text}"
