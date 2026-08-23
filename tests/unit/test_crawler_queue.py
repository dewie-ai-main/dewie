"""Tests for dewie.crawler.queue — QueueManager and CrawlJob."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_pg():
    session = AsyncMock()
    result = MagicMock()
    result.mappings.return_value.first.return_value = None
    result.scalar.return_value = 0
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    pg = MagicMock()
    pg._session_factory.return_value = cm
    return pg, session


# ── CrawlJob ──────────────────────────────────────────────────────────────────


def test_crawl_job_model():
    from dewie.crawler.queue import CrawlJob

    now = datetime.now(UTC)
    job = CrawlJob(
        id=1,
        url="https://example.com/page",
        domain="example.com",
        depth=1,
        parent_url="https://example.com",
        status="pending",
        crawl_session=uuid.uuid4(),
        error_msg=None,
        discovered_at=now,
        claimed_at=None,
        completed_at=None,
    )
    assert job.url == "https://example.com/page"
    assert job.depth == 1
    assert job.status == "pending"


# ── QueueManager ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_seed_calls_execute_and_commit():
    from dewie.crawler.queue import QueueManager

    pg, session = _make_pg()
    qm = QueueManager(pg)
    await qm.seed("https://example.com", uuid.uuid4(), "example.com")
    session.execute.assert_called_once()
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_enqueue_batch_empty_returns_zero():
    from dewie.crawler.queue import QueueManager

    pg, session = _make_pg()
    qm = QueueManager(pg)
    count = await qm.enqueue_batch(
        [], depth=1, parent_url="https://example.com", session_id=uuid.uuid4(), domain="example.com"
    )
    assert count == 0
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_enqueue_batch_returns_count():
    from dewie.crawler.queue import QueueManager

    pg, session = _make_pg()
    qm = QueueManager(pg)
    urls = ["https://a.com/1", "https://a.com/2", "https://a.com/3"]
    count = await qm.enqueue_batch(
        urls, depth=2, parent_url="https://a.com", session_id=uuid.uuid4(), domain="a.com"
    )
    assert count == 3
    session.execute.assert_called_once()
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_dequeue_returns_none_when_no_jobs():
    from dewie.crawler.queue import QueueManager

    pg, _ = _make_pg()
    qm = QueueManager(pg)
    result = await qm.dequeue(uuid.uuid4(), max_depth=3)
    assert result is None


@pytest.mark.asyncio
async def test_mark_done_calls_commit():
    from dewie.crawler.queue import QueueManager

    pg, session = _make_pg()
    qm = QueueManager(pg)
    await qm.mark_done(42)
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_mark_failed_calls_commit():
    from dewie.crawler.queue import QueueManager

    pg, session = _make_pg()
    qm = QueueManager(pg)
    await qm.mark_failed(42, "connection refused")
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_mark_needs_js_calls_commit():
    from dewie.crawler.queue import QueueManager

    pg, session = _make_pg()
    qm = QueueManager(pg)
    await qm.mark_needs_js(42)
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_count_done_returns_zero():
    from dewie.crawler.queue import QueueManager

    pg, _ = _make_pg()
    qm = QueueManager(pg)
    count = await qm.count_done(uuid.uuid4())
    assert count == 0


@pytest.mark.asyncio
async def test_count_pending_and_processing_returns_zero():
    from dewie.crawler.queue import QueueManager

    pg, _ = _make_pg()
    qm = QueueManager(pg)
    count = await qm.count_pending_and_processing(uuid.uuid4())
    assert count == 0


# ── _row_to_job ───────────────────────────────────────────────────────────────


def test_row_to_job():
    from dewie.crawler.queue import _row_to_job

    now = datetime.now(UTC)
    row = {
        "id": 5,
        "url": "https://ex.com",
        "domain": "ex.com",
        "depth": 0,
        "parent_url": None,
        "status": "done",
        "crawl_session": uuid.uuid4(),
        "error_msg": None,
        "discovered_at": now,
        "claimed_at": now,
        "completed_at": now,
    }
    job = _row_to_job(row)
    assert job.id == 5
    assert job.status == "done"
