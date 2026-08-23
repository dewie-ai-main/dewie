# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
dewie/api/routes/me.py — Per-user self-service routes (convenience aliases).

Routes:
  GET    /me/queries         — list recent query_log entries for the caller
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request
from sqlalchemy import text

router = APIRouter(prefix="/me", tags=["user"])


def _require_user(request: Request) -> str:
    """Return user_id or raise 401. In open auth mode returns a synthetic local user."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        from dewie.config import settings

        if not settings.auth_enabled:
            return "00000000-0000-0000-0000-000000000002"
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Authentication required")
    return str(user_id)


def _pg(request: Request) -> Any:  # type: ignore[name-defined]
    return request.app.state.postgres


@router.get("/queries")
async def list_user_queries(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200, description="Max number of queries to return."),
) -> list[dict]:
    """Return recent query_log entries for the authenticated user, newest first."""
    user_id = _require_user(request)

    try:
        pg = _pg(request)
    except AttributeError:
        return []

    async with pg._engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT id, question, source, ts, elapsed_ms, answer, hops "
                    "FROM query_log WHERE user_id = :uid ORDER BY ts DESC LIMIT :limit"
                ),
                {"uid": user_id, "limit": limit},
            )
        ).fetchall()

    return [
        {
            "id": str(r.id),
            "question": r.question,
            "source": r.source,
            "ts": str(r.ts) if r.ts else None,
            "elapsed_ms": r.elapsed_ms,
            "answer": r.answer,
            "hops": r.hops,
        }
        for r in rows
    ]
