# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
/feeds endpoints for managing RSS feed subscriptions.

Feeds are polled on a configurable schedule. This module provides
CRUD endpoints and a manual poll trigger for each feed.
"""

from __future__ import annotations

import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from dewie.ingestion.rss import poll_rss_feed
from dewie.models.feed import RSSFeed, RSSFeedCreate, RSSFeedUpdate

log = logging.getLogger(__name__)

router = APIRouter(prefix="/feeds", tags=["feeds"])


def _get_pg(request: Request):
    return request.app.state.postgres


def _get_processor(request: Request):
    return getattr(request.app.state, "processor", None)


def _caller_tenant(request: Request) -> UUID:
    tid = getattr(request.state, "tenant_id", None)
    if tid:
        return UUID(str(tid))
    return UUID("00000000-0000-0000-0000-000000000001")


# ── CRUD ──────────────────────────────────────────────────────────────────────

@router.post("", response_model=RSSFeed, status_code=status.HTTP_201_CREATED)
async def create_feed(
    request: Request,
    body: RSSFeedCreate,
    pg=Depends(_get_pg),
) -> RSSFeed:
    """Create a new RSS feed subscription."""
    feed = RSSFeed(
        id=uuid4(),
        url=body.url,
        name=body.name,
        corpus_id=body.corpus_id,
        tags=body.tags,
        enabled=body.enabled,
        poll_interval_minutes=body.poll_interval_minutes,
        tenant_id=_caller_tenant(request),
    )
    return await pg.create_feed(feed)


@router.get("", response_model=list[RSSFeed])
async def list_feeds(
    request: Request,
    pg=Depends(_get_pg),
) -> list[RSSFeed]:
    """List all RSS feed subscriptions for the caller's tenant."""
    return await pg.list_feeds(tenant_id=_caller_tenant(request))


@router.get("/{feed_id}", response_model=RSSFeed)
async def get_feed(
    request: Request,
    feed_id: UUID,
    pg=Depends(_get_pg),
) -> RSSFeed:
    """Get a single RSS feed by ID."""
    feed = await pg.get_feed(feed_id)
    if feed is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feed not found.")
    return feed


@router.patch("/{feed_id}", response_model=RSSFeed)
async def update_feed(
    request: Request,
    feed_id: UUID,
    body: RSSFeedUpdate,
    pg=Depends(_get_pg),
) -> RSSFeed:
    """Update fields on an RSS feed."""
    existing = await pg.get_feed(feed_id)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feed not found.")
    updates = body.model_dump(exclude_none=True)
    feed = await pg.update_feed(feed_id, **updates)
    if feed is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feed not found.")
    return feed


@router.delete("/{feed_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_feed(
    request: Request,
    feed_id: UUID,
    pg=Depends(_get_pg),
) -> None:
    """Delete an RSS feed subscription."""
    deleted = await pg.delete_feed(feed_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feed not found.")


# ── Manual poll ───────────────────────────────────────────────────────────────

@router.post("/{feed_id}/poll", status_code=status.HTTP_202_ACCEPTED)
async def poll_feed(
    request: Request,
    feed_id: UUID,
    background_tasks: BackgroundTasks,
    pg=Depends(_get_pg),
    processor=Depends(_get_processor),
) -> dict:
    """Manually trigger a poll for a feed. Returns immediately (fire-and-forget)."""
    feed = await pg.get_feed(feed_id)
    if feed is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feed not found.")
    background_tasks.add_task(poll_rss_feed, feed, pg, processor)
    return {"message": f"Poll triggered for feed '{feed.name}'."}


async def _poll_feed(feed: RSSFeed, pg, processor) -> None:
    """Fetch and ingest all documents from a feed, then mark it as polled."""
    from dewie.ingestion.rss import poll_rss_feed
    await poll_rss_feed(feed, pg, processor)
