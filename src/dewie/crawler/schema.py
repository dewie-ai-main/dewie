# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
DDL for the crawl_jobs table. Kept separate from postgres.py so the crawler
can be deployed as an independent service without touching core storage schema.
"""

from __future__ import annotations

from dewie.storage.postgres import PostgresClient

_CRAWL_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS crawl_jobs (
        id            BIGSERIAL   PRIMARY KEY,
        url           TEXT        NOT NULL,
        domain        TEXT        NOT NULL,
        depth         INTEGER     NOT NULL DEFAULT 0,
        parent_url    TEXT,
        status        TEXT        NOT NULL DEFAULT 'pending',
        crawl_session UUID        NOT NULL,
        error_msg     TEXT,
        discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        claimed_at    TIMESTAMPTZ,
        completed_at  TIMESTAMPTZ
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS uidx_crawl_jobs_session_url ON crawl_jobs (crawl_session, url)",
    "CREATE INDEX IF NOT EXISTS idx_crawl_jobs_dequeue ON crawl_jobs (crawl_session, status, depth) WHERE status = 'pending'",
]


async def init_crawl_schema(pg: PostgresClient) -> None:
    """Create crawl_jobs table and indexes if they do not already exist."""
    async with pg._engine.begin() as conn:
        for stmt in _CRAWL_SCHEMA_STATEMENTS:
            await conn.exec_driver_sql(stmt.strip())
