# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""Sources endpoint — returns all allowed (enabled) registered sources."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/sources", tags=["sources"])


class SourceItem(BaseModel):
    id: str
    name: str
    type: str
    enabled: bool


@router.get("", response_model=list[SourceItem])
async def list_sources(request: Request) -> list[SourceItem]:
    """Return all allowed (enabled) registered sources."""
    pg = request.app.state.postgres
    async with pg._engine.connect() as conn:
        from sqlalchemy import text

        result = await conn.execute(
            text(
                """
                SELECT id, name, type, enabled
                FROM dewie_sources
                WHERE enabled = true
                ORDER BY name
                """
            )
        )
        rows = result.mappings().fetchall()

    return [
        SourceItem(
            id=str(row["id"]),
            name=str(row["name"]),
            type=str(row["type"]),
            enabled=bool(row.get("enabled", True)),
        )
        for row in rows
    ]