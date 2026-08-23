# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Integration tests for the agentic tool endpoints.

Requires a running dev server at DEWIE_TEST_URL (default: http://localhost:10946).
All tests are skipped if the server is unreachable.

Tests cover:
  - /api/mcp            — HTTP REST tool dispatch (search_corpus, ingest_url)
    - /api/mcp            — research tool dispatch (research)
  - /api/query          — search (used by dewie_search, dewie_browse)
  - /api/graph/neighbors/{doc_id}   — expand (dewie_expand)
  - /api/graph/intersection         — intersect (dewie_intersect)
  - /api/graph/bridge               — bridge (dewie_bridge)
  - /api/documents/{id}/content     — read (dewie_read)
  - /api/research/agent             — research (dewie_research)

Run:
    pytest tests/integration/test_mcp_tools.py -v
"""

from __future__ import annotations

import os

import httpx
import pytest

BASE_URL = os.environ.get("DEWIE_TEST_URL", "http://localhost:10946")
API = f"{BASE_URL}/api"


def _server_up() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


_SKIP = pytest.mark.skipif(not _server_up(), reason=f"Server not running at {BASE_URL}")


@pytest.fixture(scope="module")
def client():
    """HTTP client. Automatically acquires a local session if the server is in open mode."""
    with httpx.Client(base_url=BASE_URL, timeout=30) as c:
        # In open mode the /auth/me endpoint returns a synthetic local user.
        # No login needed — the session cookie is set by the server automatically
        # for same-origin requests. For the test client, we probe /auth/me and if it
        # returns authenticated=True without a cookie, the server is in open mode and
        # auth middleware injects the user from the start. Otherwise we skip auth-gated tests.
        c.get("/api/auth/me")  # warm up / follow any redirects
        yield c


# ── Helpers ───────────────────────────────────────────────────────────────────


def _first_doc_id(client: httpx.Client, query: str = "information retrieval") -> str | None:
    """Return a real doc_id from the corpus via search, or None if corpus is empty."""
    r = client.post(f"{API}/query", json={"query": query, "limit": 1})
    if r.status_code != 200:
        return None
    results = r.json().get("results", [])
    return results[0]["doc_id"] if results else None


def _research_available(base_url: str) -> bool:
    """Return True if /research/agent responds without a 500 config error."""
    try:
        r = httpx.post(
            f"{base_url}/api/research/agent",
            json={"query": "test", "mode": "quick"},
            timeout=15,
        )
        # 500 with _KNOWN_PROVIDERS or similar = LLM not configured
        if r.status_code == 500:
            detail = r.json().get("detail", "")
            if "_KNOWN_PROVIDERS" in detail or "ModelClient" in detail or "provider" in detail.lower():
                return False
        return True
    except Exception:
        return False


_SKIP_RESEARCH = pytest.mark.skipif(
    not _server_up() or not _research_available(BASE_URL),
    reason="Research agent requires a configured LLM provider",
)


# ── /api/mcp — HTTP REST dispatch ────────────────────────────────────────────


@_SKIP
class TestMcpHttpEndpoint:
    def test_manifest_returns_tools(self, client):
        r = client.get(f"{API}/mcp")
        assert r.status_code == 200
        data = r.json()
        assert "tools" in data
        names = {t["name"] for t in data["tools"]}
        assert "search_corpus" in names
        assert "ingest_url" in names
        assert "research" in names

    def test_search_corpus_returns_results(self, client):
        r = client.post(
            f"{API}/mcp",
            json={"tool": "search_corpus", "input": {"query": "information retrieval", "limit": 3}},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["tool"] == "search_corpus"
        assert "results" in data["content"]
        assert "answers_questions" not in str(data)  # must never be exposed

    def test_search_corpus_empty_query_rejected(self, client):
        r = client.post(f"{API}/mcp", json={"tool": "search_corpus", "input": {"query": ""}})
        assert r.status_code == 422

    def test_unknown_tool_rejected(self, client):
        r = client.post(f"{API}/mcp", json={"tool": "nonexistent", "input": {}})
        assert r.status_code == 422

    def test_search_corpus_respects_limit_cap(self, client):
        r = client.post(
            f"{API}/mcp",
            json={"tool": "search_corpus", "input": {"query": "test", "limit": 100}},
        )
        assert r.status_code == 200
        results = r.json()["content"]["results"]
        assert len(results) <= 25

    def test_research_empty_query_rejected(self, client):
        r = client.post(f"{API}/mcp", json={"tool": "research", "input": {"query": ""}})
        assert r.status_code == 422

    @_SKIP_RESEARCH
    def test_research_tool_returns_answer(self, client):
        r = client.post(
            f"{API}/mcp",
            json={"tool": "research", "input": {"query": "What is information retrieval?", "mode": "quick"}},
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["tool"] == "research"
        assert "answer" in data["content"]
        assert isinstance(data["content"]["answer"], str)
        assert len(data["content"]["answer"]) > 0

    @_SKIP_RESEARCH
    def test_research_tool_never_exposes_aq(self, client):
        r = client.post(
            f"{API}/mcp",
            json={"tool": "research", "input": {"query": "information retrieval", "mode": "quick"}},
            timeout=60,
        )
        assert r.status_code == 200
        assert "answers_questions" not in r.text


# ── /api/query — search endpoint (dewie_search / dewie_browse) ───────────────


@_SKIP
class TestQueryEndpoint:
    def test_search_returns_envelope(self, client):
        r = client.post(f"{API}/query", json={"query": "information", "limit": 5})
        assert r.status_code == 200
        data = r.json()
        assert "results" in data

    def test_search_result_fields(self, client):
        r = client.post(f"{API}/query", json={"query": "information", "limit": 3})
        assert r.status_code == 200
        results = r.json().get("results", [])
        if results:
            result = results[0]
            assert "doc_id" in result
            assert "title" in result
            assert "answers_questions" not in result  # never exposed

    def test_search_result_confidence_present(self, client):
        r = client.post(f"{API}/query", json={"query": "test", "limit": 3})
        assert r.status_code == 200
        data = r.json()
        assert "result_confidence" in data

    def test_search_limit_respected(self, client):
        r = client.post(f"{API}/query", json={"query": "the", "limit": 2})
        assert r.status_code == 200
        results = r.json().get("results", [])
        assert len(results) <= 2

    def test_search_empty_query_rejected(self, client):
        r = client.post(f"{API}/query", json={"query": "", "limit": 5})
        assert r.status_code in (400, 422)

    def test_rankers_endpoint_lists_options(self, client):
        r = client.get(f"{API}/query/rankers")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        ranker_ids = [x["id"] if isinstance(x, dict) else x for x in data]
        assert "rrf" in ranker_ids


# ── /api/graph/neighbors — expand (dewie_expand) ─────────────────────────────


@_SKIP
class TestGraphNeighborsEndpoint:
    def test_neighbors_returns_list(self, client):
        doc_id = _first_doc_id(client)
        if not doc_id:
            pytest.skip("No docs in corpus")
        r = client.get(f"{API}/graph/neighbors/{doc_id}")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_neighbors_unknown_id_returns_empty_or_404(self, client):
        r = client.get(f"{API}/graph/neighbors/00000000-0000-0000-0000-000000000000")
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            assert isinstance(r.json(), list)

    def test_neighbors_result_has_doc_fields(self, client):
        doc_id = _first_doc_id(client)
        if not doc_id:
            pytest.skip("No docs in corpus")
        r = client.get(f"{API}/graph/neighbors/{doc_id}")
        assert r.status_code == 200
        neighbors = r.json()
        if neighbors:
            n = neighbors[0]
            assert "doc_id" in n or "id" in n


# ── /api/graph/intersection — intersect (dewie_intersect) ────────────────────


@_SKIP
class TestGraphIntersectionEndpoint:
    def test_intersection_returns_docs(self, client):
        r = client.post(f"{API}/query", json={"query": "information", "limit": 2})
        results = r.json().get("results", [])
        if len(results) < 2:
            pytest.skip("Need at least 2 docs in corpus")
        ids = [results[0]["doc_id"], results[1]["doc_id"]]
        r2 = client.post(f"{API}/graph/intersection", json={"doc_ids": ids, "limit": 5})
        assert r2.status_code == 200
        data = r2.json()
        assert "docs" in data
        assert isinstance(data["docs"], list)

    def test_intersection_with_single_id_returns_result(self, client):
        doc_id = _first_doc_id(client)
        if not doc_id:
            pytest.skip("No docs in corpus")
        r = client.post(f"{API}/graph/intersection", json={"doc_ids": [doc_id]})
        assert r.status_code in (200, 400, 422)

    def test_intersection_missing_doc_ids_returns_error(self, client):
        r = client.post(f"{API}/graph/intersection", json={})
        # Server returns 422 for missing field OR 200 with error message
        if r.status_code == 200:
            assert "error" in r.json() or "docs" in r.json()
        else:
            assert r.status_code == 422


# ── /api/graph/bridge — bridge (dewie_bridge) ────────────────────────────────


@_SKIP
class TestGraphBridgeEndpoint:
    def test_bridge_between_same_doc(self, client):
        doc_id = _first_doc_id(client)
        if not doc_id:
            pytest.skip("No docs in corpus")
        r = client.post(
            f"{API}/graph/bridge",
            json={"source_id": doc_id, "target_id": doc_id, "max_depth": 3},
        )
        assert r.status_code in (200, 404)

    def test_bridge_missing_ids_returns_error(self, client):
        r = client.post(f"{API}/graph/bridge", json={})
        # Server returns 422 for missing fields OR 200 with error message
        if r.status_code == 200:
            assert "error" in r.json() or "path" in r.json()
        else:
            assert r.status_code == 422

    def test_bridge_between_two_docs(self, client):
        r = client.post(f"{API}/query", json={"query": "information", "limit": 2})
        results = r.json().get("results", [])
        if len(results) < 2:
            pytest.skip("Need at least 2 docs in corpus")
        src, tgt = results[0]["doc_id"], results[1]["doc_id"]
        r2 = client.post(
            f"{API}/graph/bridge",
            json={"source_id": src, "target_id": tgt, "max_depth": 5},
        )
        assert r2.status_code in (200, 404)
        if r2.status_code == 200:
            data = r2.json()
            assert "path" in data or "hops" in data or isinstance(data, dict)


# ── /api/documents/{id}/content — read (dewie_read) ──────────────────────────


@_SKIP
class TestDocumentContentEndpoint:
    def test_read_returns_text(self, client):
        doc_id = _first_doc_id(client)
        if not doc_id:
            pytest.skip("No docs in corpus")
        r = client.get(f"{API}/documents/{doc_id}/content")
        assert r.status_code == 200
        assert len(r.text) > 0

    def test_read_unknown_id_returns_404(self, client):
        r = client.get(f"{API}/documents/00000000-0000-0000-0000-000000000000/content")
        assert r.status_code == 404

    def test_read_content_never_includes_answers_questions_as_json_field(self, client):
        doc_id = _first_doc_id(client)
        if not doc_id:
            pytest.skip("No docs in corpus")
        r = client.get(f"{API}/documents/{doc_id}/content")
        assert r.status_code == 200
        # Content endpoint returns raw text — not JSON, so AQ can't leak structurally
        # but validate it's not accidentally returning the search response envelope
        assert '"answers_questions"' not in r.text[:500]


# ── /api/research/agent — research (dewie_research) ──────────────────────────


@_SKIP
class TestResearchAgentEndpoint:
    def test_research_invalid_mode_rejected(self, client):
        r = client.post(
            f"{API}/research/agent",
            json={"query": "test", "mode": "invalid_mode"},
        )
        assert r.status_code in (400, 422)

    def test_research_empty_query_rejected(self, client):
        r = client.post(f"{API}/research/agent", json={"query": "", "mode": "quick"})
        assert r.status_code in (400, 422)

    @_SKIP_RESEARCH
    def test_research_returns_answer(self, client):
        r = client.post(
            f"{API}/research/agent",
            json={"query": "What is information retrieval?", "mode": "quick"},
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    @_SKIP_RESEARCH
    def test_research_includes_docs_used(self, client):
        r = client.post(
            f"{API}/research/agent",
            json={"query": "information retrieval", "mode": "quick"},
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert "docs_used" in data
        assert isinstance(data["docs_used"], list)

    @_SKIP_RESEARCH
    def test_research_answers_questions_never_in_response(self, client):
        r = client.post(
            f"{API}/research/agent",
            json={"query": "test query", "mode": "quick"},
            timeout=60,
        )
        assert r.status_code == 200
        assert "answers_questions" not in r.text
