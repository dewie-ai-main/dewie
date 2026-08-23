"""Tests for #241 — corpus tab 404 fix.

The /api/pipeline/corpus/* endpoints were gated behind admin scope by the
blanket "/api/pipeline" -> "admin" rule.  Verify that the longest-prefix-wins
logic correctly applies "read" scope to /api/pipeline/corpus/* requests and
"admin" scope to other /api/pipeline/* requests.
"""

from __future__ import annotations


def _scope_for_path(path: str) -> str | None:
    """
    Mirror the _SCOPE_PREFIX_RULES longest-prefix-wins logic from middleware
    so we can test it without spinning up a full app.
    """
    from dewie.api.middleware import _SCOPE_PREFIX_RULES

    matched_prefix: str | None = None
    matched_scope: str | None = None
    for prefix, required_scope in _SCOPE_PREFIX_RULES:
        if path.startswith(prefix):
            if matched_prefix is None or len(prefix) > len(matched_prefix):
                matched_prefix = prefix
                matched_scope = required_scope
    return matched_scope


# ── /api/pipeline/corpus/* should require "read" (not "admin") ───────────────────

def test_corpus_quality_requires_read_scope():
    assert _scope_for_path("/api/pipeline/corpus/quality") == "read"


def test_corpus_sources_requires_read_scope():
    assert _scope_for_path("/api/pipeline/corpus/sources") == "read"


def test_corpus_source_docs_requires_read_scope():
    assert _scope_for_path("/api/pipeline/corpus/sources/arxiv.org/docs") == "read"


def test_corpus_quality_refresh_requires_read_scope():
    assert _scope_for_path("/api/pipeline/corpus/quality/refresh") == "read"


# ── /api/pipeline/workers/status should require "read" ────────────────────────────

def test_workers_status_requires_read_scope():
    assert _scope_for_path("/api/pipeline/workers/status") == "read"


# ── Other /api/pipeline/* routes still require "admin" ───────────────────────────

def test_pipeline_errors_requires_admin_scope():
    assert _scope_for_path("/api/pipeline/errors") == "admin"


def test_pipeline_workers_pause_requires_admin_scope():
    assert _scope_for_path("/api/pipeline/workers/pause") == "admin"


def test_pipeline_workers_resume_requires_admin_scope():
    assert _scope_for_path("/api/pipeline/workers/resume") == "admin"


def test_pipeline_inject_body_requires_admin_scope():
    assert _scope_for_path("/api/pipeline/inject-body") == "admin"


# ── Unrelated routes are unaffected ──────────────────────────────────────────

def test_query_requires_read_scope():
    assert _scope_for_path("/api/query") == "read"


def test_admin_route_requires_admin_scope():
    assert _scope_for_path("/api/admin/keys") == "admin"


def test_ingest_requires_ingest_scope():
    assert _scope_for_path("/api/ingest") == "ingest"
