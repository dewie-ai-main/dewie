# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""Unit tests for the corpus-first web_lookup engine (the gate decision table)."""

from __future__ import annotations

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from dewie.search.providers import SearchHit, StubProvider
from dewie.search.web_lookup import web_lookup

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _doc(title: str, *, aq: list[str] | None = None, topics: list[str] | None = None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        title=title,
        url=f"https://corpus.example/{title.replace(' ', '-')}",
        summary=f"summary of {title}",
        answers_questions=aq or [],
        topics=topics or [],
        ingested_at=datetime(2026, 6, 1, 12, 0, 0),
    )


def _pg_with(results):
    pg = AsyncMock()
    pg.search = AsyncMock(return_value=results)
    return pg


QUERY = "volcanic eruption monitoring iceland"

# A corpus that genuinely answers the query: one doc with strong AQ coverage
# and discriminated scores (top result clearly wins).
_GOOD_CORPUS = [
    (
        _doc(
            "iceland volcano guide",
            aq=["how is volcanic eruption monitoring done in iceland"],
            topics=["volcanic activity", "iceland", "eruption monitoring"],
        ),
        0.9,
    ),
    (_doc("unrelated cooking blog"), 0.2),
    (_doc("another filler doc"), 0.1),
]

# A corpus that only has adjacent junk: no AQ coverage, flat scores, no topics.
_THIN_CORPUS = [
    (_doc("knitting patterns"), 0.016),
    (_doc("tax law summary"), 0.015),
    (_doc("celebrity gossip"), 0.015),
]


# ── Corpus-hit path ───────────────────────────────────────────────────────────


async def test_corpus_hit_returns_corpus_and_never_calls_provider():
    provider = StubProvider([SearchHit(title="W", url="https://w.example", content="web text")])
    result, new_doc = await web_lookup(QUERY, pg=_pg_with(_GOOD_CORPUS), provider=provider)

    assert result.source == "corpus"
    assert result.gap is None
    assert result.corpus_hits[0]["title"] == "iceland volcano guide"
    assert result.corpus_hits[0]["ingested_at"] == "2026-06-01T12:00:00"
    assert provider.calls == [], "provider must not be called when the corpus suffices"
    assert new_doc is None


async def test_corpus_hits_never_expose_answers_questions():
    result, _ = await web_lookup(QUERY, pg=_pg_with(_GOOD_CORPUS), provider=None)
    for hit in result.corpus_hits:
        assert "answers_questions" not in hit
    assert "answers_questions" not in str(result.to_content())


# ── Gap path ──────────────────────────────────────────────────────────────────


async def test_gap_with_provider_content_returns_web_and_builds_doc():
    provider = StubProvider(
        [SearchHit(title="Fresh", url="https://fresh.example/page", content="long web text " * 100)]
    )
    result, new_doc = await web_lookup(QUERY, pg=_pg_with(_THIN_CORPUS), provider=provider)

    assert result.source == "web"
    assert result.gap is not None
    assert result.content_url == "https://fresh.example/page"
    assert result.content and result.content.startswith("long web text")
    assert provider.calls == [QUERY]
    assert new_doc is not None
    assert new_doc.url == "https://fresh.example/page"
    assert str(new_doc.id) == result.ingested_doc_id


async def test_gap_zero_results_is_hard_gap():
    provider = StubProvider([SearchHit(title="W", url="https://w.example", content="text " * 200)])
    result, new_doc = await web_lookup(QUERY, pg=_pg_with([]), provider=provider)
    assert result.source == "web"
    assert "absent from the corpus" in result.gap
    assert new_doc is not None


async def test_gap_without_provider_is_miss_with_reason():
    result, new_doc = await web_lookup(QUERY, pg=_pg_with(_THIN_CORPUS), provider=None)
    assert result.source == "miss"
    assert "No web search provider configured" in result.gap
    assert new_doc is None


async def test_gap_provider_returns_nothing_is_miss():
    result, new_doc = await web_lookup(QUERY, pg=_pg_with(_THIN_CORPUS), provider=StubProvider([]))
    assert result.source == "miss"
    assert new_doc is None


async def test_provider_exception_degrades_to_miss():
    provider = AsyncMock()
    provider.name = "boom"
    provider.search = AsyncMock(side_effect=RuntimeError("rate limited"))
    result, new_doc = await web_lookup(QUERY, pg=_pg_with(_THIN_CORPUS), provider=provider)
    assert result.source == "miss"
    assert "rate limited" in result.gap
    assert new_doc is None


# ── Agent override flags ──────────────────────────────────────────────────────


async def test_force_web_skips_the_gate():
    provider = StubProvider(
        [SearchHit(title="W", url="https://w.example", content="forced web text " * 50)]
    )
    result, new_doc = await web_lookup(
        QUERY, pg=_pg_with(_GOOD_CORPUS), provider=provider, force_web=True
    )
    assert result.source == "web"
    assert provider.calls == [QUERY]
    assert result.corpus_hits, "corpus hits still returned for context"
    assert new_doc is not None


async def test_corpus_only_never_hits_web_even_on_gap():
    provider = StubProvider([SearchHit(title="W", url="https://w.example", content="x " * 500)])
    result, new_doc = await web_lookup(
        QUERY, pg=_pg_with(_THIN_CORPUS), provider=provider, corpus_only=True
    )
    assert result.source == "miss"
    assert result.gap is not None
    assert provider.calls == []
    assert new_doc is None


# ── Content shaping ───────────────────────────────────────────────────────────


async def test_content_is_capped_in_tool_output():
    provider = StubProvider(
        [SearchHit(title="Big", url="https://big.example", content="y" * 50_000)]
    )
    result, _ = await web_lookup(QUERY, pg=_pg_with(_THIN_CORPUS), provider=provider)
    assert len(result.to_content()["content"]) <= 8000
    # but the full body goes into the corpus document
    assert result.content == "y" * 50_000


async def test_unenriched_corpus_with_term_coverage_is_not_a_gap():
    """No LLM yet → no aq/topics anywhere; title+body coverage must satisfy the gate."""
    doc = _doc("How Iceland monitors volcanic eruptions")
    doc.body = "Seismometers track eruption monitoring across iceland."
    result, new_doc = await web_lookup(
        QUERY,
        pg=_pg_with([(doc, 10.0)]),
        provider=StubProvider([SearchHit(title="W", url="https://w.example", content="x " * 500)]),
    )
    assert result.source == "corpus"
    assert new_doc is None
