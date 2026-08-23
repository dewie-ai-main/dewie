# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
dewie/watcher.py — File system watcher for data/bodies/ auto-ingest.

Watches a directory for new .txt body files and upserts them into the
documents table as pending docs. The enrichment pipeline picks them up
automatically on its next poll cycle.

Expected file layout (matches body_store.py convention):
  <watch_dir>/<doc_id[:2]>/<doc_id>.txt   — shard subdirs
  <watch_dir>/<doc_id>.txt                 — flat (also accepted)

Configuration (dewie.yml):
  watcher:
    enabled: false
    watch_dir: data/bodies

Environment variables:
  DATABASE_URL     — PostgreSQL DSN (asyncpg format)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from dewie.models.content import ContentDocument, ContentStatus
from dewie.storage.postgres import PostgresClient

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Core ingest logic
# ---------------------------------------------------------------------------


async def ingest_body_file(path: Path, pg: PostgresClient) -> bool:
    """
    Upsert a body file as a pending document.

    Returns True if the file was ingested, False if it was skipped.
    """
    # Derive doc_id from filename stem
    stem = path.stem
    try:
        doc_id = uuid.UUID(stem)
    except ValueError:
        logger.debug("Skipping non-UUID filename: %s", path.name)
        return False

    # Skip if already processed
    existing = await pg.get_by_id(str(doc_id))
    if existing and existing.status in (ContentStatus.READY, ContentStatus.PROCESSING):
        logger.debug("Doc %s already %s — skipping", doc_id, existing.status)
        return False

    # Read body
    try:
        body = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return False

    if not body:
        logger.debug("Empty body file %s — skipping", path)
        return False

    # Use first line as title fallback
    first_line = body.splitlines()[0][:200]

    doc = ContentDocument(
        id=doc_id,
        url=f"file://bodies/{doc_id}",  # synthetic URL — unique by doc_id
        title=first_line,
        summary="",
        source="file-watcher",
        ingested_at=datetime.now(UTC),
        status=ContentStatus.PENDING,
    )

    await pg.upsert(doc)
    logger.info("Ingested body file → doc %s (%d chars)", doc_id, len(body))
    return True


# ---------------------------------------------------------------------------
# Watchdog event handler
# ---------------------------------------------------------------------------


class BodyFileHandler(FileSystemEventHandler):
    """Handles new .txt file creation events with debounce."""

    def __init__(self, pg: PostgresClient, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._pg = pg
        self._loop = loop
        self._pending: dict[str, float] = {}  # path → last event timestamp
        self._lock = threading.Lock()
        self._debounce_thread = threading.Thread(target=self._debounce_loop, daemon=True)
        self._debounce_thread.start()

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix != ".txt":
            return
        logger.debug("Detected new file: %s", path)
        with self._lock:
            self._pending[str(path)] = time.monotonic()

    def _debounce_loop(self) -> None:
        """Background thread — fires ingest after debounce window."""
        while True:
            time.sleep(DEBOUNCE_SECONDS / 2)
            now = time.monotonic()
            with self._lock:
                ready = [p for p, t in self._pending.items() if now - t >= DEBOUNCE_SECONDS]
                for p in ready:
                    del self._pending[p]

            for path_str in ready:
                asyncio.run_coroutine_threadsafe(self._process(Path(path_str)), self._loop)

    async def _process(self, path: Path) -> None:
        ingested = await ingest_body_file(path, self._pg)
        if ingested:
            logger.debug("Doc ingested; enrichment pipeline will pick it up on next poll")


# ---------------------------------------------------------------------------
# Main watcher entrypoint
# ---------------------------------------------------------------------------


async def run_watcher(watch_dir: str | Path) -> None:
    """Start watching `watch_dir` and block until interrupted."""
    watch_dir = Path(watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)

    from dewie.config import settings

    pg = PostgresClient(settings.postgres_dsn)
    await pg.init_schema()

    loop = asyncio.get_event_loop()
    handler = BodyFileHandler(pg=pg, loop=loop)
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=True)
    observer.start()
    logger.info("Watching %s for new .txt body files", watch_dir.resolve())

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        observer.stop()
        observer.join()
        logger.info("Watcher stopped")
