"""Tests for dewie.api.routes.pipeline — pure helpers and route handlers."""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── _docker ───────────────────────────────────────────────────────────────────


def test_docker_success():
    from dewie.api.routes.pipeline import _docker

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")
        rc, output = _docker("ps")
    assert rc == 0
    assert "output" in output


def test_docker_file_not_found():
    from fastapi import HTTPException

    from dewie.api.routes.pipeline import _docker

    with patch("subprocess.run", side_effect=FileNotFoundError()):
        with pytest.raises(HTTPException) as exc:
            _docker("ps")
    assert exc.value.status_code == 503
    assert "docker not found" in exc.value.detail


def test_docker_timeout():
    from fastapi import HTTPException

    from dewie.api.routes.pipeline import _docker

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 15)):
        with pytest.raises(HTTPException) as exc:
            _docker("ps")
    assert exc.value.status_code == 503


def test_docker_other_error():
    from fastapi import HTTPException

    from dewie.api.routes.pipeline import _docker

    with patch("subprocess.run", side_effect=RuntimeError("oops")):
        with pytest.raises(HTTPException) as exc:
            _docker("ps")
    assert exc.value.status_code == 503


# ── _worker_containers ────────────────────────────────────────────────────────


def test_worker_containers_parses_output():
    from dewie.api.routes.pipeline import _worker_containers

    with patch(
        "dewie.api.routes.pipeline._docker",
        return_value=(0, "worker-1\tUp 5 hours\nworker-2\tExited"),
    ):
        containers = _worker_containers()
    assert len(containers) == 2
    assert containers[0]["name"] == "worker-1"
    assert containers[0]["status"] == "Up 5 hours"


def test_worker_containers_skips_dev():
    from dewie.api.routes.pipeline import _worker_containers

    with patch("dewie.api.routes.pipeline._docker", return_value=(0, "worker-dev\tUp 5 hours")):
        containers = _worker_containers()
    assert len(containers) == 0


def test_worker_containers_returns_empty_on_error():
    from dewie.api.routes.pipeline import _worker_containers

    with patch("dewie.api.routes.pipeline._docker", return_value=(1, "error")):
        containers = _worker_containers()
    assert containers == []


def test_worker_containers_empty_output():
    from dewie.api.routes.pipeline import _worker_containers

    with patch("dewie.api.routes.pipeline._docker", return_value=(0, "")):
        containers = _worker_containers()
    assert containers == []


# ── _worker_status ────────────────────────────────────────────────────────────


def test_worker_status_running():
    from dewie.api.routes.pipeline import _worker_status

    with patch(
        "dewie.api.routes.pipeline._worker_containers",
        return_value=[
            {"name": "worker-1", "status": "Up 5 hours"},
        ],
    ):
        status = _worker_status()
    assert status["worker-1"] == "RUNNING"


def test_worker_status_stopped():
    from dewie.api.routes.pipeline import _worker_status

    with patch(
        "dewie.api.routes.pipeline._worker_containers",
        return_value=[
            {"name": "worker-1", "status": "Exited (1) 5 minutes ago"},
        ],
    ):
        status = _worker_status()
    assert status["worker-1"] == "STOPPED"


def test_worker_status_empty():
    from dewie.api.routes.pipeline import _worker_status

    with patch("dewie.api.routes.pipeline._worker_containers", return_value=[]):
        status = _worker_status()
    assert status == {}


# ── Pydantic models ───────────────────────────────────────────────────────────


def test_inject_body_request():
    import uuid

    from dewie.api.routes.pipeline import InjectBodyRequest

    doc_id = uuid.uuid4()
    req = InjectBodyRequest(doc_id=doc_id, body_text="some text here")
    assert req.doc_id == doc_id
    assert req.body_text == "some text here"


def test_resolve_request():
    from dewie.api.routes.pipeline import ResolveRequest

    req = ResolveRequest(error_ids=[1, 2, 3])
    assert req.error_ids == [1, 2, 3]


# ── workers_status route ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_workers_status_returns_dict():
    from dewie.api.routes.pipeline import workers_status

    req = MagicMock()
    with patch(
        "dewie.api.routes.pipeline._worker_status", return_value={"worker-1": "RUNNING"}
    ):
        result = await workers_status(req)
    assert "workers" in result
    assert result["running"] == 1


@pytest.mark.asyncio
async def test_workers_status_empty_returns_warning():
    from dewie.api.routes.pipeline import workers_status

    req = MagicMock()
    with patch("dewie.api.routes.pipeline._worker_status", return_value={}):
        result = await workers_status(req)
    assert "warning" in result
    assert result["total"] == 0


# ── inject_body route ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inject_body_not_found():
    import uuid

    from fastapi import HTTPException

    from dewie.api.routes.pipeline import InjectBodyRequest, inject_body

    pg = MagicMock()
    pg.get_by_id = AsyncMock(return_value=None)
    req = MagicMock()
    req.app.state.postgres = pg
    body = InjectBodyRequest(doc_id=uuid.uuid4(), body_text="hello")
    with pytest.raises(HTTPException) as exc:
        await inject_body(req, body)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_inject_body_success():
    import uuid

    from dewie.api.routes.pipeline import InjectBodyRequest, inject_body

    doc_id = uuid.uuid4()
    pg = MagicMock()
    pg.get_by_id = AsyncMock(return_value={"id": doc_id})
    pg.write_body_text = AsyncMock()

    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    pg._session_factory = MagicMock(return_value=session_cm)

    req = MagicMock()
    req.app.state.postgres = pg
    body = InjectBodyRequest(doc_id=doc_id, body_text="Hello World")
    result = await inject_body(req, body)
    assert result["bytes_written"] > 0
    assert result["doc_id"] == str(doc_id)


# ── priority_enrich route ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_priority_enrich_not_found():
    import uuid

    from fastapi import HTTPException

    from dewie.api.routes.pipeline import PriorityInjectRequest, priority_enrich

    pg = MagicMock()
    pg.get_by_id = AsyncMock(return_value=None)
    req = MagicMock()
    req.app.state.postgres = pg
    body = PriorityInjectRequest(doc_id=uuid.uuid4())
    with pytest.raises(HTTPException) as exc:
        await priority_enrich(req, body)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_priority_enrich_success():
    import uuid

    from dewie.api.routes.pipeline import PriorityInjectRequest, priority_enrich

    doc_id = uuid.uuid4()
    pg = MagicMock()
    pg.get_by_id = AsyncMock(return_value={"id": doc_id})

    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    pg._session_factory = MagicMock(return_value=session_cm)

    req = MagicMock()
    req.app.state.postgres = pg
    body = PriorityInjectRequest(doc_id=doc_id)
    result = await priority_enrich(req, body)
    assert result["doc_id"] == str(doc_id)
    assert result["status"] == "queued"


# ── resolve_pipeline_errors route ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_pipeline_errors():
    from dewie.api.routes.pipeline import ResolveRequest, resolve_pipeline_errors

    pg = MagicMock()
    req = MagicMock()
    req.app.state.postgres = pg
    body = ResolveRequest(error_ids=[1, 2, 3])
    with patch("dewie.storage.pipeline_errors.mark_resolved", AsyncMock(return_value=(3, 3))):
        result = await resolve_pipeline_errors(req, body)
    assert result["resolved"] == 3
    assert result["requeued"] == 3


# ── workers_pause / workers_resume routes ─────────────────────────────────────


@pytest.mark.asyncio
async def test_workers_pause_already_stopped():
    from dewie.api.routes.pipeline import workers_pause

    req = MagicMock()
    with patch("dewie.api.routes.pipeline._worker_status", return_value={"w1": "STOPPED"}):
        result = await workers_pause(req)
    assert result["ok"] is True
    assert "already stopped" in result["message"].lower()


@pytest.mark.asyncio
async def test_workers_resume_already_running():
    from dewie.api.routes.pipeline import workers_resume

    req = MagicMock()
    with patch("dewie.api.routes.pipeline._worker_status", return_value={"w1": "RUNNING"}):
        result = await workers_resume(req)
    assert result["ok"] is True
    assert "already running" in result["message"].lower()


@pytest.mark.asyncio
async def test_workers_pause_stops_running():
    from dewie.api.routes.pipeline import workers_pause

    req = MagicMock()
    with (
        patch("dewie.api.routes.pipeline._worker_status", return_value={"w1": "RUNNING"}),
        patch("dewie.api.routes.pipeline._docker", return_value=(0, "w1")),
    ):
        result = await workers_pause(req)
    assert result["ok"] is True
    assert "1 worker" in result["message"]


@pytest.mark.asyncio
async def test_workers_resume_starts_stopped():
    from dewie.api.routes.pipeline import workers_resume

    req = MagicMock()
    with (
        patch(
            "dewie.api.routes.pipeline._worker_status",
            side_effect=[
                {"w1": "STOPPED"},
                {"w1": "RUNNING"},
            ],
        ),
        patch("dewie.api.routes.pipeline._docker", return_value=(0, "w1")),
    ):
        result = await workers_resume(req)
    assert result["ok"] is True


# ── corpus_sources route ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_corpus_sources_success():
    from dewie.api.routes.pipeline import corpus_sources

    row = {"source": "arxiv.org", "ready": 100, "pending": 5, "failed": 2, "total": 107}
    conn = AsyncMock()
    result = MagicMock()
    result.mappings.return_value = [row]
    conn.execute = AsyncMock(return_value=result)
    begin_cm = MagicMock()
    begin_cm.__aenter__ = AsyncMock(return_value=conn)
    begin_cm.__aexit__ = AsyncMock(return_value=None)
    pg = MagicMock()
    pg._engine.begin.return_value = begin_cm
    req = MagicMock()
    req.app.state.postgres = pg
    sources = await corpus_sources(req)
    assert isinstance(sources, list)
    assert len(sources) == 1
    assert sources[0]["source"] == "arxiv.org"


@pytest.mark.asyncio
async def test_corpus_sources_db_failure():
    from fastapi import HTTPException

    from dewie.api.routes.pipeline import corpus_sources

    pg = MagicMock()
    pg._engine.begin.side_effect = RuntimeError("DB down")
    req = MagicMock()
    req.app.state.postgres = pg
    with pytest.raises(HTTPException) as exc:
        await corpus_sources(req)
    assert exc.value.status_code == 500


# ── corpus_quality ────────────────────────────────────────────────────────────


def _make_engine_mock(**rows):
    """Returns a pg mock whose connect() returns data for given queries."""
    conn = AsyncMock()

    # We'll use side_effect to return different rows per call
    def execute_side_effect(query, *args, **kwargs):
        result = MagicMock()
        result.mappings.return_value.one.return_value = rows.get("default", {})
        result.mappings.return_value.all.return_value = rows.get("all", [])
        return result

    conn.execute = AsyncMock(side_effect=execute_side_effect)
    connect_cm = MagicMock()
    connect_cm.__aenter__ = AsyncMock(return_value=conn)
    connect_cm.__aexit__ = AsyncMock(return_value=None)
    begin_cm = MagicMock()
    begin_cm.__aenter__ = AsyncMock(return_value=conn)
    begin_cm.__aexit__ = AsyncMock(return_value=None)
    pg = MagicMock()
    pg._engine.connect.return_value = connect_cm
    pg._engine.begin.return_value = begin_cm
    return pg, conn


@pytest.mark.asyncio
async def test_corpus_quality_success():
    from dewie.api.routes.pipeline import corpus_quality

    pg, conn = _make_engine_mock()
    # For corpus_quality, connect() runs 4 queries; mock returns dicts
    quality_row = {
        "total": 100,
        "ready": 80,
        "pending": 10,
        "failed": 5,
        "with_embedding": 70,
        "embed_summary_good": 60,
        "embed_summary_stub": 5,
        "embed_summary_none": 15,
        "avg_embed_summary_len": 500,
        "avg_body_len": 2000,
        "avg_aqs": 3.5,
        "with_aqs": 65,
        "empty_aqs": 15,
        "quality_high": 40,
        "quality_medium": 30,
        "quality_low": 10,
        "quality_stub": 0,
        "refreshed_at": None,
    }
    chunk_row = {
        "docs_with_chunks": 50,
        "total_chunks": 500,
        "chunks_with_embed": 400,
        "chunks_with_aq_embed": 350,
        "avg_chunks_per_doc": 10,
        "avg_chunk_tokens": 200,
        "avg_chunk_chars": 1000,
        "docs_with_aq_embed": 45,
    }
    chunk_status_row = {
        "status_none": 30,
        "status_chunked": 50,
        "status_skipped": 10,
        "status_failed": 2,
    }
    call_count = [0]

    async def execute_side(*a, **kw):
        n = call_count[0]
        call_count[0] += 1
        r = MagicMock()
        if n == 0:
            r.mappings.return_value.one.return_value = quality_row
        elif n == 1:
            r.mappings.return_value.all.return_value = [{"source": "arxiv", "ready": 50}]
        elif n == 2:
            r.mappings.return_value.one.return_value = chunk_row
        else:
            r.mappings.return_value.one.return_value = chunk_status_row
        return r

    conn.execute = AsyncMock(side_effect=execute_side)
    req = MagicMock()
    req.app.state.postgres = pg
    result = await corpus_quality(req)
    assert result["summary"]["total"] == 100
    assert "chunks" in result
    assert "by_source" in result


@pytest.mark.asyncio
async def test_corpus_quality_db_failure():
    from fastapi import HTTPException

    from dewie.api.routes.pipeline import corpus_quality

    pg = MagicMock()
    pg._engine.connect.side_effect = RuntimeError("DB down")
    req = MagicMock()
    req.app.state.postgres = pg
    with pytest.raises(HTTPException) as exc:
        await corpus_quality(req)
    assert exc.value.status_code == 500


# ── corpus_quality_refresh ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_corpus_quality_refresh_success():
    from dewie.api.routes.pipeline import corpus_quality_refresh

    pg, conn = _make_engine_mock()
    req = MagicMock()
    req.app.state.postgres = pg
    result = await corpus_quality_refresh(req)
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_corpus_quality_refresh_failure():
    from fastapi import HTTPException

    from dewie.api.routes.pipeline import corpus_quality_refresh

    pg = MagicMock()
    pg._is_sqlite = False
    pg._engine.begin.side_effect = RuntimeError("DB down")
    req = MagicMock()
    req.app.state.postgres = pg
    with pytest.raises(HTTPException) as exc:
        await corpus_quality_refresh(req)
    assert exc.value.status_code == 500


# ── corpus_source_docs ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_corpus_source_docs_success():
    from dewie.api.routes.pipeline import corpus_source_docs

    pg, conn = _make_engine_mock()
    rows = [
        {
            "id": "abc",
            "url": "https://example.com",
            "title": "Test",
            "status": "ready",
            "enriched_at": None,
        }
    ]
    conn.execute = AsyncMock(return_value=MagicMock(mappings=MagicMock(return_value=rows)))
    req = MagicMock()
    req.app.state.postgres = pg
    result = await corpus_source_docs(req, source="arxiv", limit=10)
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_corpus_source_docs_unknown_source():
    from dewie.api.routes.pipeline import corpus_source_docs

    pg, conn = _make_engine_mock()
    conn.execute = AsyncMock(return_value=MagicMock(mappings=MagicMock(return_value=[])))
    req = MagicMock()
    req.app.state.postgres = pg
    result = await corpus_source_docs(req, source="(unknown)", limit=10)
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_corpus_source_docs_failure():
    from fastapi import HTTPException

    from dewie.api.routes.pipeline import corpus_source_docs

    pg = MagicMock()
    pg._engine.begin.side_effect = RuntimeError("DB down")
    req = MagicMock()
    req.app.state.postgres = pg
    with pytest.raises(HTTPException) as exc:
        await corpus_source_docs(req, source="arxiv", limit=10)
    assert exc.value.status_code == 500
