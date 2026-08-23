# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Pydantic model and queue manager for crawl_jobs rows.

Uses the same SQLAlchemy async engine as PostgresClient so no second
connection pool is needed — callers pass in the existing PostgresClient.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import text

if TYPE_CHECKING:
    from dewie.storage.postgres import PostgresClient


class CrawlJob(BaseModel):
    id: int
    url: str
    domain: str
    depth: int
    parent_url: str | None
    status: str
    crawl_session: UUID
    error_msg: str | None
    discovered_at: datetime
    claimed_at: datetime | None
    completed_at: datetime | None

    model_config = {"from_attributes": True}


class QueueManager:
    """Async FIFO queue backed by the crawl_jobs table."""

    def __init__(self, pg: PostgresClient) -> None:
        self._pg = pg

    # ── Writers ───────────────────────────────────────────────────────────────

    async def seed(self, url: str, session_id: UUID, domain: str) -> None:
        """Insert the seed URL as a depth-0 job (idempotent via ON CONFLICT DO NOTHING)."""
        sql = text("""
            INSERT INTO crawl_jobs (url, domain, depth, crawl_session)
            VALUES (:url, :domain, 0, :session_id)
            ON CONFLICT (crawl_session, url) DO NOTHING
        """)
        async with self._pg._session_factory() as session:
            await session.execute(
                sql,
                {"url": url, "domain": domain, "session_id": str(session_id)},
            )
            await session.commit()

    async def enqueue_batch(
        self,
        urls: list[str],
        depth: int,
        parent_url: str,
        session_id: UUID,
        domain: str,
    ) -> int:
        """
        Bulk-insert discovered URLs. Returns the number of rows actually inserted.
        Existing (session, url) pairs are silently skipped.
        """
        if not urls:
            return 0
        sql = text("""
            INSERT INTO crawl_jobs (url, domain, depth, parent_url, crawl_session)
            VALUES (:url, :domain, :depth, :parent_url, :session_id)
            ON CONFLICT (crawl_session, url) DO NOTHING
        """)
        rows = [
            {
                "url": u,
                "domain": domain,
                "depth": depth,
                "parent_url": parent_url,
                "session_id": str(session_id),
            }
            for u in urls
        ]
        async with self._pg._session_factory() as session:
            await session.execute(sql, rows)
            await session.commit()
            return len(rows)

    async def dequeue(self, session_id: UUID, max_depth: int) -> CrawlJob | None:
        """
        Atomically claim one pending job (BFS order: lowest depth first, then FIFO).
        Uses FOR UPDATE SKIP LOCKED so concurrent workers don't collide.
        """
        sql = text("""
            UPDATE crawl_jobs SET status='processing', claimed_at=NOW()
            WHERE id = (
                SELECT id FROM crawl_jobs
                WHERE crawl_session = :session_id
                  AND status = 'pending'
                  AND depth <= :max_depth
                ORDER BY depth ASC, id ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING *
        """)
        async with self._pg._session_factory() as session:
            row = (
                (
                    await session.execute(
                        sql, {"session_id": str(session_id), "max_depth": max_depth}
                    )
                )
                .mappings()
                .first()
            )
            await session.commit()
        return _row_to_job(row) if row else None

    async def mark_done(self, job_id: int) -> None:
        sql = text("""
            UPDATE crawl_jobs SET status='done', completed_at=NOW()
            WHERE id = :id
        """)
        async with self._pg._session_factory() as session:
            await session.execute(sql, {"id": job_id})
            await session.commit()

    async def mark_failed(self, job_id: int, error_msg: str) -> None:
        sql = text("""
            UPDATE crawl_jobs SET status='failed', completed_at=NOW(), error_msg=:err
            WHERE id = :id
        """)
        async with self._pg._session_factory() as session:
            await session.execute(sql, {"id": job_id, "err": error_msg[:2000]})
            await session.commit()

    async def mark_needs_js(self, job_id: int) -> None:
        sql = text("""
            UPDATE crawl_jobs SET status='needs_js', completed_at=NOW()
            WHERE id = :id
        """)
        async with self._pg._session_factory() as session:
            await session.execute(sql, {"id": job_id})
            await session.commit()

    # ── Counters ──────────────────────────────────────────────────────────────

    async def count_done(self, session_id: UUID) -> int:
        sql = text("""
            SELECT COUNT(*) FROM crawl_jobs
            WHERE crawl_session = :sid AND status = 'done'
        """)
        async with self._pg._session_factory() as session:
            result = await session.execute(sql, {"sid": str(session_id)})
            return int(result.scalar() or 0)

    async def count_pending_and_processing(self, session_id: UUID) -> int:
        sql = text("""
            SELECT COUNT(*) FROM crawl_jobs
            WHERE crawl_session = :sid AND status IN ('pending', 'processing')
        """)
        async with self._pg._session_factory() as session:
            result = await session.execute(sql, {"sid": str(session_id)})
            return int(result.scalar() or 0)


def _row_to_job(row: dict) -> CrawlJob:  # type: ignore[type-arg]
    return CrawlJob(
        id=row["id"],
        url=row["url"],
        domain=row["domain"],
        depth=row["depth"],
        parent_url=row["parent_url"],
        status=row["status"],
        crawl_session=row["crawl_session"],
        error_msg=row["error_msg"],
        discovered_at=row["discovered_at"],
        claimed_at=row["claimed_at"],
        completed_at=row["completed_at"],
    )
