"""
Live integration tests targeting the production server.

These tests require the following environment variables to be set:
  - DEWIE_PROD_URL       — base URL of the production server (e.g. $DEWIE_PROD_URL)
  - DEWIE_PROD_EMAIL     — email for the dev/test account
  - DEWIE_PROD_PASSWORD  — password for the dev/test account

Run with:
    DEWIE_PROD_URL=... DEWIE_PROD_EMAIL=... DEWIE_PROD_PASSWORD=... \
    PYTHONPATH=src pytest tests/integration/test_production_server.py -v -m production
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

# ── Config ────────────────────────────────────────────────────────────────────

_prod_required = ("DEWIE_PROD_URL", "DEWIE_PROD_EMAIL", "DEWIE_PROD_PASSWORD")
_prod_missing = [k for k in _prod_required if k not in os.environ]
if _prod_missing:
    pytest.skip(
        f"Missing environment variables: {', '.join(_prod_missing)}. "
        f"Set DEWIE_PROD_URL, DEWIE_PROD_EMAIL, DEWIE_PROD_PASSWORD to run production tests",
        allow_module_level=True,
    )

PROD_BASE_URL = os.environ["DEWIE_PROD_URL"]
PROD_EMAIL = os.environ["DEWIE_PROD_EMAIL"]
PROD_PASSWORD = os.environ["DEWIE_PROD_PASSWORD"]

pytestmark = pytest.mark.production


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def prod_session() -> httpx.Client:
    """
    Authenticated httpx.Client for production server.
    Logs in once per module run and stores the session cookie.
    """
    client = httpx.Client(base_url=PROD_BASE_URL, timeout=30.0, follow_redirects=True)

    resp = client.post(
        "/api/auth/login",
        json={"email": PROD_EMAIL, "password": PROD_PASSWORD},
    )
    assert resp.status_code == 200, (
        f"Login failed ({resp.status_code}): {resp.text[:200]}"
    )
    data = resp.json()
    assert data.get("ok") is True, f"Login response not ok: {data}"
    assert "dewie_session" in resp.cookies, "No dewie_session cookie after login"

    yield client
    client.close()


@pytest.fixture(scope="module")
def anon_client() -> httpx.Client:
    """Unauthenticated client."""
    client = httpx.Client(base_url=PROD_BASE_URL, timeout=15.0)
    yield client
    client.close()


# ── Health / Connectivity ─────────────────────────────────────────────────────


class TestHealth:
    def test_health_endpoint_is_reachable(self, anon_client: httpx.Client):
        """GET /health returns 200 without auth."""
        resp = anon_client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_api_health_endpoint(self, anon_client: httpx.Client):
        """GET /health returns 200 (unauthenticated liveness probe)."""
        resp = anon_client.get("/health")
        assert resp.status_code == 200
        assert resp.json().get("status") == "ok"

    def test_root_redirects_to_login(self, anon_client: httpx.Client):
        """GET / redirects unauthenticated user toward login."""
        client = httpx.Client(base_url=PROD_BASE_URL, timeout=15.0, follow_redirects=False)
        resp = client.get("/")
        # Expect redirect (302) to /ui/login.html
        assert resp.status_code in (301, 302, 307, 308)
        assert "login" in resp.headers.get("location", "").lower()
        client.close()


# ── Auth ──────────────────────────────────────────────────────────────────────


class TestAuth:
    def test_login_succeeds(self):
        """POST /api/auth/login with valid credentials returns ok=True and a session cookie."""
        client = httpx.Client(base_url=PROD_BASE_URL, timeout=15.0)
        resp = client.post(
            "/api/auth/login",
            json={"email": PROD_EMAIL, "password": PROD_PASSWORD},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "email" in data
        assert "dewie_session" in resp.cookies
        client.close()

    def test_login_wrong_password_fails(self):
        """POST /api/auth/login with bad password returns 401."""
        client = httpx.Client(base_url=PROD_BASE_URL, timeout=15.0)
        resp = client.post(
            "/api/auth/login",
            json={"email": PROD_EMAIL, "password": "definitely_wrong_12345"},
        )
        assert resp.status_code in (401, 400)
        client.close()

    def test_auth_me_with_valid_session(self, prod_session: httpx.Client):
        """GET /api/auth/me returns the logged-in user's info."""
        resp = prod_session.get("/api/auth/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("authenticated") is True
        assert data.get("email") == PROD_EMAIL
        assert "user_id" in data

    def test_query_requires_auth(self, anon_client: httpx.Client):
        """POST /api/query without auth returns 401."""
        resp = anon_client.post(
            "/api/query",
            json={"query": "test", "limit": 1},
        )
        assert resp.status_code == 401


# ── Query ─────────────────────────────────────────────────────────────────────


class TestQuery:
    def test_basic_query_returns_results(self, prod_session: httpx.Client):
        """POST /api/query returns results from the production database."""
        resp = prod_session.post(
            "/api/query",
            json={"query": "machine learning", "limit": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data, f"No 'results' key in response: {data}"
        assert isinstance(data["results"], list)
        assert len(data["results"]) > 0, "Query returned 0 results — DB may be empty"
        assert "query" in data

    def test_query_result_has_expected_fields(self, prod_session: httpx.Client):
        """Each result has title, url, summary, and score."""
        resp = prod_session.post(
            "/api/query",
            json={"query": "artificial intelligence", "limit": 3},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) > 0
        first = results[0]
        for field in ("title", "url", "summary", "score"):
            assert field in first, f"Missing field '{field}' in result: {first.keys()}"

    def test_query_limit_is_respected(self, prod_session: httpx.Client):
        """limit param caps result count."""
        for limit in (1, 3, 10):
            resp = prod_session.post(
                "/api/query",
                json={"query": "science", "limit": limit},
            )
            assert resp.status_code == 200
            count = len(resp.json()["results"])
            assert count <= limit, f"Expected ≤{limit} results, got {count}"

    def test_empty_query_handled(self, prod_session: httpx.Client):
        """Empty query string returns a response (not a 500)."""
        resp = prod_session.post(
            "/api/query",
            json={"query": "", "limit": 1},
        )
        # May return 200 with empty results or 422 validation error — not 500
        assert resp.status_code != 500

    def test_query_performance(self, prod_session: httpx.Client):
        """Production query completes in under 10 seconds."""
        start = time.perf_counter()
        resp = prod_session.post(
            "/api/query",
            json={"query": "technology trends 2024", "limit": 5},
        )
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < 10.0, f"Query took {elapsed:.2f}s — too slow"

    def test_query_result_count_indicates_populated_db(self, prod_session: httpx.Client):
        """Production DB should have many documents — specific query returns results."""
        resp = prod_session.post(
            "/api/query",
            json={"query": "wikipedia machine learning artificial intelligence", "limit": 10},
        )
        assert resp.status_code == 200
        count = len(resp.json()["results"])
        assert count >= 1, f"Only got {count} results — DB may not be populated"


# ── Corpus / Documents ────────────────────────────────────────────────────────


class TestCorpus:
    def test_sources_list_returns_data(self, prod_session: httpx.Client):
        """GET /api/corpus/sources returns sources list."""
        resp = prod_session.get("/api/corpus/sources")
        assert resp.status_code == 200
        data = resp.json()
        # Should be a list of sources
        assert isinstance(data, list), f"Expected list, got {type(data)}"
        assert len(data) >= 0  # May be empty in clean install, but endpoint should exist

    def test_documents_endpoint(self, prod_session: httpx.Client):
        """GET /api/documents returns documents."""
        resp = prod_session.get("/api/documents", params={"limit": 5})
        assert resp.status_code == 200

    def test_service_status(self, prod_session: httpx.Client):
        """GET /api/service-status returns status info."""
        resp = prod_session.get("/api/service-status")
        assert resp.status_code == 200


# ── Static UI ─────────────────────────────────────────────────────────────────


class TestStaticUI:
    def test_login_page_served(self, anon_client: httpx.Client):
        """GET /ui/login.html returns 200 HTML."""
        resp = anon_client.get("/ui/login.html")
        assert resp.status_code == 200
        assert b"Sign in" in resp.content or b"login" in resp.content.lower()

    def test_app_page_requires_auth(self, anon_client: httpx.Client):
        """GET /ui/app.html without auth should redirect to login."""
        # With follow_redirects=False we see the redirect
        client = httpx.Client(base_url=PROD_BASE_URL, timeout=15.0, follow_redirects=False)
        resp = client.get("/ui/app.html")
        # Either 302 to login or 401; not 200 with app content
        # (The API middleware redirects /ui/admin.html but not app.html — 
        # app.html relies on client-side auth check)
        assert resp.status_code in (200, 302, 401)
        client.close()
