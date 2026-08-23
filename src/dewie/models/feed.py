# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Pydantic models for RSS feed subscriptions.

An RSSFeed represents a configured feed URL that the poller
checks on a schedule and ingests into the document pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

_PUBLIC_TENANT = UUID("00000000-0000-0000-0000-000000000001")


class RSSFeed(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    url: str
    name: str
    corpus_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    enabled: bool = True
    poll_interval_minutes: int = 60
    last_polled_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tenant_id: UUID = Field(default_factory=lambda: _PUBLIC_TENANT)

    model_config = {"from_attributes": True}


class RSSFeedCreate(BaseModel):
    url: str
    name: str
    corpus_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    enabled: bool = True
    poll_interval_minutes: int = 60


class RSSFeedUpdate(BaseModel):
    url: str | None = None
    name: str | None = None
    corpus_id: str | None = None
    tags: list[str] | None = None
    enabled: bool | None = None
    poll_interval_minutes: int | None = None
