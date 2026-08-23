"""
tests/test_watcher.py — File system watcher tests.

Unit tests: ingest logic, skip-known-doc, debounce behaviour.
All tests use mocked PostgresClient and a mocked enrichment trigger.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pg(existing_doc=None):
    """Build a mock PostgresClient."""
    pg = MagicMock()
    pg.get_by_id = AsyncMock(return_value=existing_doc)
    pg.upsert = AsyncMock()
    pg.init_schema = AsyncMock()
    return pg


# ---------------------------------------------------------------------------
# Unit: ingest_body_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_new_file_upserts_doc(tmp_path: Path):
    """New .txt file with valid UUID name → upsert called with pending status."""
    from dewie.models.content import ContentStatus
    from dewie.watcher import ingest_body_file

    doc_id = uuid.uuid4()
    body_file = tmp_path / f"{doc_id}.txt"
    body_file.write_text("This is the article body text.\nMore content here.")

    pg = _make_pg(existing_doc=None)
    result = await ingest_body_file(body_file, pg)

    assert result is True
    pg.upsert.assert_awaited_once()
    upserted_doc = pg.upsert.call_args[0][0]
    assert upserted_doc.id == doc_id
    assert upserted_doc.status == ContentStatus.PENDING
    assert upserted_doc.source == "file-watcher"


@pytest.mark.asyncio
async def test_ingest_skips_non_uuid_filename(tmp_path: Path):
    """Files with non-UUID names are silently skipped."""
    from dewie.watcher import ingest_body_file

    bad_file = tmp_path / "not-a-uuid.txt"
    bad_file.write_text("some content")

    pg = _make_pg()
    result = await ingest_body_file(bad_file, pg)

    assert result is False
    pg.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_skips_empty_file(tmp_path: Path):
    """Empty .txt files are skipped."""
    from dewie.watcher import ingest_body_file

    doc_id = uuid.uuid4()
    empty_file = tmp_path / f"{doc_id}.txt"
    empty_file.write_text("   ")  # whitespace only

    pg = _make_pg(existing_doc=None)
    result = await ingest_body_file(empty_file, pg)

    assert result is False
    pg.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_skips_already_ready_doc(tmp_path: Path):
    """Doc already in DB with status=ready is not re-upserted."""

    from dewie.models.content import ContentDocument, ContentStatus
    from dewie.watcher import ingest_body_file

    doc_id = uuid.uuid4()
    body_file = tmp_path / f"{doc_id}.txt"
    body_file.write_text("Some body text here.")

    existing = MagicMock(spec=ContentDocument)
    existing.status = ContentStatus.READY
    pg = _make_pg(existing_doc=existing)

    result = await ingest_body_file(body_file, pg)

    assert result is False
    pg.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_skips_already_processing_doc(tmp_path: Path):
    """Doc already in DB with status=processing is not re-upserted."""
    from dewie.models.content import ContentDocument, ContentStatus
    from dewie.watcher import ingest_body_file

    doc_id = uuid.uuid4()
    body_file = tmp_path / f"{doc_id}.txt"
    body_file.write_text("Some body text.")

    existing = MagicMock(spec=ContentDocument)
    existing.status = ContentStatus.PROCESSING
    pg = _make_pg(existing_doc=existing)

    result = await ingest_body_file(body_file, pg)

    assert result is False
    pg.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_re_ingests_pending_doc(tmp_path: Path):
    """Doc in DB with status=pending should be upserted again (re-queue)."""
    from dewie.models.content import ContentDocument, ContentStatus
    from dewie.watcher import ingest_body_file

    doc_id = uuid.uuid4()
    body_file = tmp_path / f"{doc_id}.txt"
    body_file.write_text("Updated body content.")

    existing = MagicMock(spec=ContentDocument)
    existing.status = ContentStatus.PENDING
    pg = _make_pg(existing_doc=existing)

    result = await ingest_body_file(body_file, pg)

    assert result is True
    pg.upsert.assert_awaited_once()


# ---------------------------------------------------------------------------
# Unit: debounce — rapid events for same file fire handler only once
# ---------------------------------------------------------------------------


def test_debounce_fires_once_for_rapid_events(tmp_path: Path):
    """Multiple rapid on_created events for the same file debounce to one ingest call."""
    from dewie.watcher import DEBOUNCE_SECONDS, BodyFileHandler

    loop = asyncio.new_event_loop()
    pg = _make_pg()

    ingest_calls: list[str] = []

    async def fake_ingest(path, _pg):
        ingest_calls.append(str(path))
        return True

    with (
        patch("dewie.watcher.ingest_body_file", side_effect=fake_ingest),
    ):
        handler = BodyFileHandler(pg=pg, loop=loop)

        doc_id = uuid.uuid4()
        test_file = tmp_path / f"{doc_id}.txt"
        test_file.write_text("content")

        from watchdog.events import FileCreatedEvent

        event = FileCreatedEvent(str(test_file))

        # Fire 5 rapid events
        for _ in range(5):
            handler.on_created(event)
            time.sleep(0.05)

        # Wait for debounce to fire (DEBOUNCE_SECONDS + a little buffer)
        deadline = time.monotonic() + DEBOUNCE_SECONDS + 1.0
        loop.run_until_complete(asyncio.sleep(0))

        # Run the loop briefly to let coroutines execute
        end = time.monotonic() + DEBOUNCE_SECONDS + 1.0
        while time.monotonic() < end:
            loop.run_until_complete(asyncio.sleep(0.1))
            if ingest_calls:
                break

    loop.close()

    # Only one ingest call despite 5 events
    assert len(ingest_calls) == 1
    assert str(doc_id) in ingest_calls[0]


# ---------------------------------------------------------------------------
# Unit: config_file WatcherConfig
# ---------------------------------------------------------------------------


def test_watcher_config_defaults():
    """WatcherConfig has correct defaults."""
    from dewie.config_file import WatcherConfig

    cfg = WatcherConfig()
    assert cfg.enabled is False
    assert cfg.watch_dir == "data/bodies"


def test_dewie_config_has_watcher():
    """DewieConfig exposes watcher field."""
    from dewie.config_file import DewieConfig

    cfg = DewieConfig()
    assert hasattr(cfg, "watcher")
    assert cfg.watcher.enabled is False
