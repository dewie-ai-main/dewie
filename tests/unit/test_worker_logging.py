"""Tests for logging instrumentation in background worker loops."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── helpers ────────────────────────────────────────────────────────────────────


def _make_settings(**kwargs):
    s = MagicMock()
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


# ── chunk_embedder worker ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chunk_embedder_loop_logs_started(caplog):
    from dewie.workers.chunk_embedder import run_chunk_embedder_loop

    pg = AsyncMock()
    stop = asyncio.Event()
    settings = _make_settings(chunk_embedder_batch_size=10, chunk_embedder_sleep_secs=0)

    with patch("dewie.workers.chunk_embedder.run_once", new=AsyncMock(return_value=0)):
        stop.set()
        with caplog.at_level(logging.INFO, logger="dewie.workers.chunk_embedder"):
            await run_chunk_embedder_loop(pg, settings, stop=stop)

    messages = [r.message for r in caplog.records]
    assert any("chunk_embedder_loop started" in m for m in messages)


@pytest.mark.asyncio
async def test_chunk_embedder_loop_logs_tick_done(caplog):
    from dewie.workers.chunk_embedder import run_chunk_embedder_loop

    pg = AsyncMock()
    settings = _make_settings(chunk_embedder_batch_size=5, chunk_embedder_sleep_secs=0)
    stop = asyncio.Event()

    call_count = 0

    async def _run_once_side_effect(pg, batch_size):
        nonlocal call_count
        call_count += 1
        stop.set()  # stop after first tick
        return 3

    with patch("dewie.workers.chunk_embedder.run_once", side_effect=_run_once_side_effect):
        with caplog.at_level(logging.INFO, logger="dewie.workers.chunk_embedder"):
            await run_chunk_embedder_loop(pg, settings, stop=stop)

    records_with_extra = [r for r in caplog.records if hasattr(r, "processed")]
    # At least one tick-done record with processed count
    tick_done = [r for r in caplog.records if "tick done" in r.message]
    assert tick_done, "Expected a 'tick done' log record"
    assert call_count == 1


@pytest.mark.asyncio
async def test_chunk_embedder_loop_logs_exception(caplog):
    from dewie.workers.chunk_embedder import run_chunk_embedder_loop

    pg = AsyncMock()
    settings = _make_settings(chunk_embedder_batch_size=5, chunk_embedder_sleep_secs=0)
    stop = asyncio.Event()

    call_count = 0

    async def _exploding_run_once(pg, batch_size):
        nonlocal call_count
        call_count += 1
        stop.set()
        raise RuntimeError("boom")

    with patch("dewie.workers.chunk_embedder.run_once", side_effect=_exploding_run_once):
        with caplog.at_level(logging.ERROR, logger="dewie.workers.chunk_embedder"):
            await run_chunk_embedder_loop(pg, settings, stop=stop)

    error_msgs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("tick failed" in m for m in error_msgs)


@pytest.mark.asyncio
async def test_chunk_embedder_loop_logs_cancelled(caplog):
    from dewie.workers.chunk_embedder import run_chunk_embedder_loop

    pg = AsyncMock()
    settings = _make_settings(chunk_embedder_batch_size=5, chunk_embedder_sleep_secs=0)
    stop = asyncio.Event()

    async def _cancel_run_once(pg, batch_size):
        raise asyncio.CancelledError()

    with patch("dewie.workers.chunk_embedder.run_once", side_effect=_cancel_run_once):
        with caplog.at_level(logging.INFO, logger="dewie.workers.chunk_embedder"):
            await run_chunk_embedder_loop(pg, settings, stop=stop)

    messages = [r.message for r in caplog.records]
    assert any("cancelled" in m for m in messages)


# ── enrichment worker ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_enrichment_loop_logs_started(caplog):
    from dewie.workers.enrichment import run_enrichment_loop

    pg = AsyncMock()
    pg.get_pending_docs = AsyncMock(return_value=[])
    processor = AsyncMock()
    stop = asyncio.Event()
    settings = _make_settings(enrichment_batch_size=2, enrichment_sleep_secs=0)

    stop.set()
    with caplog.at_level(logging.INFO, logger="dewie.workers.enrichment"):
        await run_enrichment_loop(pg, processor, settings, stop=stop)

    messages = [r.message for r in caplog.records]
    assert any("enrichment_loop started" in m for m in messages)


@pytest.mark.asyncio
async def test_enrichment_loop_serial_mode_log(caplog):
    from dewie.workers.enrichment import run_enrichment_loop

    pg = AsyncMock()
    pg.get_pending_docs = AsyncMock(return_value=[])
    processor = AsyncMock()
    stop = asyncio.Event()
    settings = _make_settings(enrichment_batch_size=1, enrichment_sleep_secs=0)

    stop.set()
    with caplog.at_level(logging.INFO, logger="dewie.workers.enrichment"):
        await run_enrichment_loop(pg, processor, settings, stop=stop)

    messages = [r.message for r in caplog.records]
    assert any("SERIAL" in m for m in messages)


@pytest.mark.asyncio
async def test_enrichment_loop_logs_doc_done(caplog):
    from uuid import uuid4

    from dewie.workers.enrichment import run_enrichment_loop

    doc_id = str(uuid4())

    pg = AsyncMock()
    call_count = 0

    async def _get_pending_side(limit):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [doc_id]
        return []

    pg.get_pending_docs = AsyncMock(side_effect=_get_pending_side)

    mock_doc = MagicMock()
    mock_doc.id = doc_id
    pg.get_by_id = AsyncMock(return_value=mock_doc)

    processor = AsyncMock()
    processor.enrich_and_persist = AsyncMock()
    stop = asyncio.Event()

    settings = _make_settings(enrichment_batch_size=2, enrichment_sleep_secs=0)

    # Stop after second call (empty queue) so we don't loop forever
    original_get = pg.get_pending_docs.side_effect

    async def _stopping_get(limit):
        result = await original_get(limit)
        if not result:
            stop.set()
        return result

    pg.get_pending_docs = AsyncMock(side_effect=_stopping_get)

    with (
        patch("dewie.workers.enrichment.load_body", return_value="body text"),
        caplog.at_level(logging.INFO, logger="dewie.workers.enrichment"),
    ):
        await run_enrichment_loop(pg, processor, settings, stop=stop)

    messages = [r.message for r in caplog.records]
    assert any("doc done" in m for m in messages)


@pytest.mark.asyncio
async def test_enrichment_loop_logs_doc_failed(caplog):
    from uuid import uuid4

    from dewie.workers.enrichment import run_enrichment_loop

    doc_id = str(uuid4())

    pg = AsyncMock()
    pg.get_pending_docs = AsyncMock(return_value=[doc_id])
    pg.mark_status = AsyncMock()

    mock_doc = MagicMock()
    mock_doc.id = doc_id
    pg.get_by_id = AsyncMock(return_value=mock_doc)

    processor = AsyncMock()
    processor.enrich_and_persist = AsyncMock(side_effect=RuntimeError("enrich failed"))
    stop = asyncio.Event()

    call_count = 0

    async def _get_pending(limit):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [doc_id]
        stop.set()
        return []

    pg.get_pending_docs = AsyncMock(side_effect=_get_pending)

    settings = _make_settings(enrichment_batch_size=2, enrichment_sleep_secs=0)

    with (
        patch("dewie.workers.enrichment.load_body", return_value="body"),
        caplog.at_level(logging.ERROR, logger="dewie.workers.enrichment"),
    ):
        await run_enrichment_loop(pg, processor, settings, stop=stop)

    error_msgs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("doc failed" in m for m in error_msgs)


# ── ingest poller ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_poller_loop_logs_started(caplog):
    from dewie.workers.ingest_poller import run_ingest_loop

    stop = asyncio.Event()
    pg = AsyncMock()
    pg.list_feeds = AsyncMock(return_value=[])
    pg.get_feeds_due_for_poll = AsyncMock(return_value=[])

    async def _fake_sleep(secs):
        stop.set()

    with patch("asyncio.sleep", side_effect=_fake_sleep):
        with caplog.at_level(logging.INFO, logger="dewie.workers.ingest_poller"):
            await run_ingest_loop(pg, None, stop=stop)

    messages = [r.message for r in caplog.records]
    assert any("ingest_poller_loop started" in m for m in messages)


@pytest.mark.asyncio
async def test_ingest_poller_loop_logs_no_feeds(caplog):
    from dewie.workers.ingest_poller import run_ingest_loop

    stop = asyncio.Event()
    pg = AsyncMock()
    pg.list_feeds = AsyncMock(return_value=[])
    pg.get_feeds_due_for_poll = AsyncMock(return_value=[])

    async def _fake_sleep(secs):
        stop.set()

    with patch("asyncio.sleep", side_effect=_fake_sleep):
        with caplog.at_level(logging.DEBUG, logger="dewie.workers.ingest_poller"):
            await run_ingest_loop(pg, None, stop=stop)

    messages = [r.message for r in caplog.records]
    assert any("no feeds due" in m for m in messages)


@pytest.mark.asyncio
async def test_ingest_poller_loop_logs_cancelled(caplog):
    from dewie.workers.ingest_poller import run_ingest_loop

    stop = asyncio.Event()
    pg = AsyncMock()
    pg.list_feeds = AsyncMock(return_value=[])
    pg.get_feeds_due_for_poll = AsyncMock(return_value=[])

    async def _fake_sleep(secs):
        raise asyncio.CancelledError()

    with patch("asyncio.sleep", side_effect=_fake_sleep):
        with caplog.at_level(logging.INFO, logger="dewie.workers.ingest_poller"):
            await run_ingest_loop(pg, None, stop=stop)

    messages = [r.message for r in caplog.records]
    assert any("cancelled" in m for m in messages)


# ── llm_queue dispatcher logging ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_dispatcher_execute_job_logs_done(caplog):

    from dewie.llm_queue import LLMDispatcher

    dispatcher = LLMDispatcher.__new__(LLMDispatcher)
    dispatcher._r = None
    dispatcher._semaphore = asyncio.Semaphore(4)
    dispatcher._min_interval = 0.0
    dispatcher._last_fire = 0.0
    dispatcher._processed = 0
    dispatcher._errors = 0

    fake_result = {"content": "hi", "output_tokens": 10}

    mock_r = AsyncMock()
    mock_r.setex = AsyncMock()

    async def _fake_redis():
        return mock_r

    dispatcher._redis = _fake_redis
    dispatcher._run_llm = AsyncMock(return_value=fake_result)

    job = {"job_id": "abc123de-0000-0000-0000-000000000000", "type": "llm", "messages": []}

    with caplog.at_level(logging.INFO, logger="dewie.llm_queue"):
        await dispatcher._execute_job(job)

    messages = [r.message for r in caplog.records]
    assert any("job done" in m for m in messages)
    assert dispatcher._processed == 1


@pytest.mark.asyncio
async def test_llm_dispatcher_execute_job_logs_failure(caplog):
    from dewie.llm_queue import LLMDispatcher

    dispatcher = LLMDispatcher.__new__(LLMDispatcher)
    dispatcher._r = None
    dispatcher._semaphore = asyncio.Semaphore(4)
    dispatcher._min_interval = 0.0
    dispatcher._last_fire = 0.0
    dispatcher._processed = 0
    dispatcher._errors = 0

    mock_r = AsyncMock()
    mock_r.setex = AsyncMock()

    async def _fake_redis():
        return mock_r

    dispatcher._redis = _fake_redis
    dispatcher._run_llm = AsyncMock(side_effect=RuntimeError("llm failed"))

    job = {"job_id": "abc123de-0000-0000-0000-000000000000", "type": "llm", "messages": []}

    with caplog.at_level(logging.ERROR, logger="dewie.llm_queue"):
        await dispatcher._execute_job(job)

    error_msgs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("job failed" in m for m in error_msgs)
    assert dispatcher._errors == 1


@pytest.mark.asyncio
async def test_llm_dispatcher_execute_job_logs_job_id(caplog):
    """job_id[:8] must appear in log output."""
    from dewie.llm_queue import LLMDispatcher

    dispatcher = LLMDispatcher.__new__(LLMDispatcher)
    dispatcher._r = None
    dispatcher._semaphore = asyncio.Semaphore(4)
    dispatcher._min_interval = 0.0
    dispatcher._last_fire = 0.0
    dispatcher._processed = 0
    dispatcher._errors = 0

    fake_result = {"content": "ok", "output_tokens": 5}
    mock_r = AsyncMock()
    mock_r.setex = AsyncMock()

    async def _fake_redis():
        return mock_r

    dispatcher._redis = _fake_redis
    dispatcher._run_llm = AsyncMock(return_value=fake_result)

    full_job_id = "deadbeef-1234-5678-abcd-000000000000"
    job = {"job_id": full_job_id, "type": "llm", "messages": []}

    with caplog.at_level(logging.DEBUG, logger="dewie.llm_queue"):
        await dispatcher._execute_job(job)

    # job_id[:8] is stored in the LogRecord extra field
    all_job_ids = [
        getattr(r, "job_id", "") for r in caplog.records
    ]
    assert any("deadbeef" in jid for jid in all_job_ids), (
        f"Expected job_id 'deadbeef' in log extra fields, got: {all_job_ids}"
    )
