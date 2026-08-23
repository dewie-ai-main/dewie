# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
LLM response cache backed by Postgres.

Before any LLM call, check the cache. On miss, call the LLM and insert.
This means:
- Tests replay from DB without API calls
- Re-enrichment iterations don't re-call the LLM unless the cache is explicitly busted
- We have a full audit trail of every LLM response per document per step
"""

from __future__ import annotations

import hashlib
from uuid import UUID

from sqlalchemy import text


async def get_cached(pg, doc_id: UUID, step: str, model: str, prompt: str) -> str | None:
    """
    Return cached LLM response if available for this doc+step+model+prompt.
    Returns None on cache miss.
    Checks prompt_hash to invalidate if prompt changed.
    """
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    sql = text("""
        SELECT raw_response FROM llm_cache
        WHERE doc_id = :doc_id AND step = :step AND model = :model
          AND prompt_hash = :prompt_hash
        LIMIT 1
    """)
    async with pg._session_factory() as session:
        row = (
            (
                await session.execute(
                    sql,
                    {
                        "doc_id": str(doc_id),
                        "step": step,
                        "model": model,
                        "prompt_hash": prompt_hash,
                    },
                )
            )
            .mappings()
            .first()
        )
    return row["raw_response"] if row else None


async def set_cached(pg, doc_id: UUID, step: str, model: str, prompt: str, response: str) -> None:
    """
    Store an LLM response in the cache.
    Upserts — replaces existing entry for same doc+step+model.
    Silently skips if doc_id no longer exists (FK violation).
    """
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    sql = text("""
        INSERT INTO llm_cache (doc_id, step, model, prompt_hash, raw_response)
        VALUES (:doc_id, :step, :model, :prompt_hash, :raw_response)
        ON CONFLICT (doc_id, step, model) DO UPDATE SET
            prompt_hash  = EXCLUDED.prompt_hash,
            raw_response = EXCLUDED.raw_response
    """)
    try:
        async with pg._session_factory() as session:
            await session.execute(
                sql,
                {
                    "doc_id": str(doc_id),
                    "step": step,
                    "model": model,
                    "prompt_hash": prompt_hash,
                    "raw_response": response,
                },
            )
            await session.commit()
    except Exception:
        # FK violation (doc deleted) or other DB error — cache miss is safe to ignore
        import logging

        logging.getLogger(__name__).debug(
            "llm_cache.set_cached skipped for doc %s: FK or DB error", doc_id
        )


async def bust_cache(pg, doc_id: UUID, step: str | None = None) -> int:
    """
    Delete cache entries for a doc. If step is None, delete all steps for the doc.
    Returns number of rows deleted.
    """
    if step is None:
        sql = text("DELETE FROM llm_cache WHERE doc_id = :doc_id")
        params: dict = {"doc_id": str(doc_id)}
    else:
        sql = text("DELETE FROM llm_cache WHERE doc_id = :doc_id AND step = :step")
        params = {"doc_id": str(doc_id), "step": step}

    async with pg._session_factory() as session:
        result = await session.execute(sql, params)
        await session.commit()
        return result.rowcount
