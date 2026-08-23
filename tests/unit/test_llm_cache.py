"""Unit tests for the LLM response cache (dewie.storage.llm_cache)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

from dewie.storage.llm_cache import bust_cache, get_cached, set_cached

# ── Fixtures ──────────────────────────────────────────────────────────────────

DOC_ID = uuid.uuid4()
STEP = "extraction"
MODEL = "test-model"
PROMPT = "Extract metadata from this document about AI."
RESPONSE = '{"summary": "A test document about AI.", "keywords": ["ai"]}'


def _make_pg(first_row=None, rowcount=0):
    """
    Return a (pg_mock, session_mock) pair.

    pg._session_factory() is set up as an async context manager whose
    __aenter__ returns session_mock. session_mock.execute() returns a result
    whose .mappings().first() gives first_row and whose .rowcount gives rowcount.
    """
    session = AsyncMock()
    exec_result = MagicMock()
    exec_result.mappings.return_value.first.return_value = first_row
    exec_result.rowcount = rowcount
    session.execute.return_value = exec_result

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=None)

    pg = MagicMock()
    pg._session_factory.return_value = cm
    return pg, session


# ── Tests ─────────────────────────────────────────────────────────────────────


async def test_cache_miss_returns_none():
    """get_cached returns None when the DB returns no row."""
    pg, _ = _make_pg(first_row=None)
    result = await get_cached(pg, DOC_ID, STEP, MODEL, PROMPT)
    assert result is None


async def test_cache_set_and_get():
    """set_cached stores the response; get_cached retrieves it."""
    # Verify set_cached executes an INSERT
    set_pg, set_session = _make_pg()
    await set_cached(set_pg, DOC_ID, STEP, MODEL, PROMPT, RESPONSE)
    set_session.execute.assert_called_once()
    set_session.commit.assert_called_once()

    # Verify get_cached returns the stored response
    row = {"raw_response": RESPONSE}
    get_pg, _ = _make_pg(first_row=row)
    result = await get_cached(get_pg, DOC_ID, STEP, MODEL, PROMPT)
    assert result == RESPONSE


async def test_cache_prompt_hash_mismatch():
    """get_cached returns None when the prompt differs (hash mismatch → no DB row)."""
    # The query filters on prompt_hash, so a different prompt produces a different
    # hash and the DB returns no match — simulated by first_row=None.
    pg, _ = _make_pg(first_row=None)
    result = await get_cached(pg, DOC_ID, STEP, MODEL, "a completely different prompt")
    assert result is None


async def test_cache_bust_single_step():
    """bust_cache with a specific step deletes only that step's entry."""
    pg, session = _make_pg(rowcount=1)
    deleted = await bust_cache(pg, DOC_ID, STEP)
    assert deleted == 1
    session.execute.assert_called_once()
    session.commit.assert_called_once()
    # SQL must reference the step column
    executed_sql = str(session.execute.call_args[0][0]).lower()
    assert "step" in executed_sql


async def test_cache_bust_all_steps():
    """bust_cache with step=None deletes all entries for the doc."""
    pg, session = _make_pg(rowcount=2)
    deleted = await bust_cache(pg, DOC_ID, step=None)
    assert deleted == 2
    session.execute.assert_called_once()
    # SQL must NOT include a step filter
    executed_sql = str(session.execute.call_args[0][0]).lower()
    assert "step" not in executed_sql
    assert "doc_id" in executed_sql


async def test_cache_upsert():
    """set_cached called twice for the same doc+step+model uses ON CONFLICT upsert."""
    pg, session = _make_pg()
    await set_cached(pg, DOC_ID, STEP, MODEL, PROMPT, RESPONSE)
    await set_cached(pg, DOC_ID, STEP, MODEL, PROMPT, "updated response")

    # Both calls execute (DB handles deduplication via ON CONFLICT)
    assert session.execute.call_count == 2

    # The SQL uses ON CONFLICT upsert semantics
    last_sql = str(session.execute.call_args_list[-1][0][0]).upper()
    assert "ON CONFLICT" in last_sql
