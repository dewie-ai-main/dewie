# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
POST /search-queue/enqueue — insert a search gap into the enrichment queue.

Callers: query.py fires this automatically when gap_signal is detected or
top-result score < 0.3.  External callers can also submit queries directly.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from dewie.storage.postgres import PostgresClient

router = APIRouter(prefix="/search-queue", tags=["search-queue"])


def _get_pg(request: Request) -> PostgresClient:
    return request.app.state.postgres


class EnqueueRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    category: str | None = None
    priority: int = Field(default=5, ge=1, le=10)


class EnqueueResponse(BaseModel):
    queued: bool
    id: str | None = None


@router.post(
    "/enqueue",
    response_model=EnqueueResponse,
    summary="Enqueue a search query for async Brave enrichment.",
)
async def enqueue(request: Request, body: EnqueueRequest) -> EnqueueResponse:
    """
    Insert a pending search_queue item for the given query.
    Deduplicated by query text (case-insensitive): if a pending row already
    exists for this query, returns queued=false without inserting a duplicate.
    """
    pg = _get_pg(request)
    queued, item_id = await pg.enqueue_search(
        query=body.query,
        category=body.category,
        priority=body.priority,
    )
    return EnqueueResponse(queued=queued, id=item_id)
