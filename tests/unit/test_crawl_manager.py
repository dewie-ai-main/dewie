"""Unit tests for CrawlManager."""

from __future__ import annotations

import json
import time

import pytest

from dewie.ingestion.crawl_manager import CrawlManager

# ---------------------------------------------------------------------------
# 1. Register
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_source():
    manager = CrawlManager()
    manager.register("rss:techcrunch")
    s = manager.status()
    assert "rss:techcrunch" in s
    assert s["rss:techcrunch"]["paused"] is False
    assert s["rss:techcrunch"]["docs_ingested"] == 0


# ---------------------------------------------------------------------------
# 2. Pause / resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pause_and_resume():
    manager = CrawlManager()
    manager.register("wiki:machine-learning")

    manager.pause("wiki:machine-learning")
    assert manager.status()["wiki:machine-learning"]["paused"] is True

    manager.resume("wiki:machine-learning")
    assert manager.status()["wiki:machine-learning"]["paused"] is False


# ---------------------------------------------------------------------------
# 3. record_ingested increments count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_ingested_increments_count():
    manager = CrawlManager()
    manager.register("rss:techcrunch")
    manager.record_ingested("rss:techcrunch")
    manager.record_ingested("rss:techcrunch")
    manager.record_ingested("rss:techcrunch")
    assert manager.status()["rss:techcrunch"]["docs_ingested"] == 3


# ---------------------------------------------------------------------------
# 4. record_error increments count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_error_increments_count():
    manager = CrawlManager()
    manager.register("rss:techcrunch")
    manager.record_error("rss:techcrunch")
    manager.record_error("rss:techcrunch")
    assert manager.status()["rss:techcrunch"]["error_count"] == 2


# ---------------------------------------------------------------------------
# 5. acquire() auto-registers unknown source
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_registers_unknown_source():
    manager = CrawlManager(max_rps=100.0)
    async with manager.acquire("rss:new-source"):
        pass
    assert "rss:new-source" in manager.status()


# ---------------------------------------------------------------------------
# 6. Rate limiter allows burst (high max_rps → no significant delay)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limiter_allows_burst():
    manager = CrawlManager(max_rps=100.0)
    manager.register("rss:burst")

    start = time.monotonic()
    for _ in range(5):
        async with manager.acquire("rss:burst"):
            pass
    elapsed = time.monotonic() - start

    # At max_rps=100, bucket capacity=200; 5 tokens should be available
    # immediately. Allow generous 0.5 s for CI scheduling noise.
    assert elapsed < 0.5, f"Burst took too long: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# 7. save_status() writes valid JSON to STATUS_FILE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_status_writes_json(tmp_path, monkeypatch):
    # Redirect STATUS_FILE to a temp location so we don't pollute /tmp.
    import dewie.ingestion.crawl_manager as cm_mod

    fake_path = tmp_path / "crawl-manager.json"
    monkeypatch.setattr(cm_mod, "STATUS_FILE", fake_path)

    manager = CrawlManager()
    manager.register("rss:techcrunch")
    manager.record_ingested("rss:techcrunch", 2)
    manager.save_status()

    assert fake_path.exists()
    data = json.loads(fake_path.read_text())
    assert "rss:techcrunch" in data
    assert data["rss:techcrunch"]["docs_ingested"] == 2
