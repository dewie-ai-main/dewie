"""Unit tests for the 'plan' strategy in investigate_v2.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_llm_json(obj) -> str:
    return json.dumps(obj)


# ── Plan generation schema ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_generation_returns_correct_schema():
    """_llm_call returning a valid plan JSON should populate the stored plan dict."""
    from dewie.api.routes.investigate_v2 import InvestigateJobRequest, _run_plan_job

    plan_payload = {
        "angles": ["best LLM 2024", "open source LLM models", "LLM benchmark comparison"],
        "entity_type": "LLM models",
        "attributes": ["param count", "benchmark scores", "designed for", "usage notes"],
        "completeness_criteria": "all major orgs covered, at least 15 entities",
        "adversarial_questions": ["what niche models might we miss?"],
    }

    pg = MagicMock()
    pg._engine = MagicMock()

    captured_plan: dict = {}

    async def fake_update(p, job_id, **kwargs):
        if "plan" in kwargs:
            captured_plan.update(kwargs["plan"])

    req = InvestigateJobRequest(query="best LLM models 2024", strategy="plan")

    with (
        patch("dewie.api.routes.investigate_v2._update_job", side_effect=fake_update),
        patch("dewie.api.routes.investigate_v2._llm_call", new_callable=AsyncMock) as mock_llm,
        patch(
            "dewie.api.routes.investigate_v2._brave_search",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "dewie.api.routes.investigate._dewie_search",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        # Step 0: plan, Step 1: entity extraction (empty → no entities → no gap pass), Step 4: synthesis
        mock_llm.side_effect = [
            _make_llm_json(plan_payload),  # plan
            _make_llm_json([]),  # entity extraction returns empty
            "synthesis result",  # synthesis (0 entities)
        ]

        await _run_plan_job("test-job-1", req, pg)

    assert captured_plan.get("angles") == plan_payload["angles"]
    assert captured_plan.get("entity_type") == "LLM models"
    assert "param count" in captured_plan.get("attributes", [])
    assert len(captured_plan.get("adversarial_questions", [])) >= 1


# ── Entity discovery deduplication ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_entity_discovery_deduplicates():
    """Exact-duplicate entity names returned by LLM should appear only once."""
    from dewie.api.routes.investigate_v2 import InvestigateJobRequest, _run_plan_job

    plan_payload = {
        "angles": ["query 1"],
        "entity_type": "tools",
        "attributes": ["feature"],
        "completeness_criteria": "ok",
        "adversarial_questions": [],
    }
    entities_with_dupes = ["ToolA", "ToolB", "ToolA", "ToolC"]

    investigated: list[str] = []

    async def fake_dewie(query, limit=5):
        investigated.append(query)
        return []

    req = InvestigateJobRequest(query="best tools", strategy="plan")
    pg = MagicMock()
    pg._engine = MagicMock()

    async def fake_update(p, job_id, **kwargs):
        pass

    with (
        patch("dewie.api.routes.investigate_v2._update_job", side_effect=fake_update),
        patch(
            "dewie.api.routes.investigate_v2._llm_call",
            new_callable=AsyncMock,
        ) as mock_llm,
        patch(
            "dewie.api.routes.investigate._brave_search",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "dewie.api.routes.investigate._dewie_search",
            side_effect=fake_dewie,
        ),
        patch(
            "dewie.api.routes.investigate._fetch_url",
            new_callable=AsyncMock,
            return_value=(False, "", "skip"),
        ),
    ):
        mock_llm.side_effect = [
            _make_llm_json(plan_payload),
            _make_llm_json(entities_with_dupes),
            "synthesis",
        ]

        await _run_plan_job("test-job-2", req, pg)

    # dewie is called once per unique entity — ToolA should appear exactly once
    toola_calls = [q for q in investigated if q == "ToolA"]
    assert len(toola_calls) == 1, f"ToolA investigated {len(toola_calls)} times, expected 1"
    # Total unique entities: ToolA, ToolB, ToolC = 3
    assert len(investigated) == 3, f"Expected 3 unique entity lookups, got {len(investigated)}"


# ── Dewie search called before Brave per entity ──────────────────────────


@pytest.mark.asyncio
async def test_dewie_search_called_before_brave_per_entity():
    """For each entity, dewie_search must be attempted before falling back to Brave."""
    from dewie.api.routes.investigate_v2 import InvestigateJobRequest, _run_plan_job

    plan_payload = {
        "angles": ["query"],
        "entity_type": "models",
        "attributes": ["accuracy"],
        "completeness_criteria": "ok",
        "adversarial_questions": [],
    }

    call_order: list[str] = []

    async def fake_dewie(query, limit=5):
        call_order.append(f"dewie:{query}")
        return []  # no corpus hit → should fall back to Brave

    async def fake_brave(query, count):
        call_order.append(f"brave:{query}")
        return []

    req = InvestigateJobRequest(query="top models", strategy="plan")
    pg = MagicMock()
    pg._engine = MagicMock()

    async def fake_update(p, job_id, **kwargs):
        pass

    with (
        patch("dewie.api.routes.investigate_v2._update_job", side_effect=fake_update),
        patch(
            "dewie.api.routes.investigate_v2._llm_call",
            new_callable=AsyncMock,
        ) as mock_llm,
        patch("dewie.api.routes.investigate._brave_search", side_effect=fake_brave),
        patch("dewie.api.routes.investigate._dewie_search", side_effect=fake_dewie),
        patch(
            "dewie.api.routes.investigate._fetch_url",
            new_callable=AsyncMock,
            return_value=(False, "", "skip"),
        ),
    ):
        mock_llm.side_effect = [
            _make_llm_json(plan_payload),
            _make_llm_json(["ModelX", "ModelY"]),
            "synthesis",
        ]

        await _run_plan_job("test-job-3", req, pg)

    # dewie must be called for each entity
    cat_calls = [c for c in call_order if c.startswith("dewie:")]
    assert len(cat_calls) == 2, f"Expected 2 dewie calls, got: {cat_calls}"

    # Since dewie returned no sufficient hit, Brave must also be called for each entity
    brave_entity_calls = [
        c for c in call_order if c.startswith("brave:ModelX") or c.startswith("brave:ModelY")
    ]
    assert len(brave_entity_calls) == 2, f"Expected brave fallback per entity, got: {call_order}"

    # For each entity, dewie index must precede brave index
    for entity in ("ModelX", "ModelY"):
        cat_idx = next((i for i, c in enumerate(call_order) if f"dewie:{entity}" in c), None)
        brave_idx = next((i for i, c in enumerate(call_order) if f"brave:{entity}" in c), None)
        assert cat_idx is not None, f"No dewie call for {entity}"
        assert brave_idx is not None, f"No brave call for {entity}"
        assert cat_idx < brave_idx, f"dewie for {entity} should come before brave"


# ── Adversarial pass adds new entities ───────────────────────────────────────


@pytest.mark.asyncio
async def test_adversarial_pass_adds_new_entities():
    """Gap entities surfaced by the adversarial pass should appear in the final result."""
    from dewie.api.routes.investigate_v2 import InvestigateJobRequest, _run_plan_job

    plan_payload = {
        "angles": ["query"],
        "entity_type": "tools",
        "attributes": ["feature"],
        "completeness_criteria": "ok",
        "adversarial_questions": ["what niche tools are missing?"],
    }

    req = InvestigateJobRequest(query="best tools", strategy="plan")
    pg = MagicMock()
    pg._engine = MagicMock()

    final_result: dict = {}

    async def fake_update(p, job_id, **kwargs):
        if "result" in kwargs:
            final_result.update(kwargs["result"])

    with (
        patch("dewie.api.routes.investigate_v2._update_job", side_effect=fake_update),
        patch(
            "dewie.api.routes.investigate_v2._llm_call",
            new_callable=AsyncMock,
        ) as mock_llm,
        patch(
            "dewie.api.routes.investigate_v2._brave_search",
            new_callable=AsyncMock,
            return_value=[
                {
                    "title": "NicheTool site",
                    "url": "http://x.com/niche",
                    "snippet": "NicheTool is useful",
                }
            ],
        ),
        patch(
            "dewie.api.routes.investigate._dewie_search",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "dewie.api.routes.investigate_v2._fetch_url",
            new_callable=AsyncMock,
            return_value=(False, "", "skip"),
        ),
    ):
        mock_llm.side_effect = [
            _make_llm_json(plan_payload),  # Step 0: plan
            _make_llm_json(["ToolA"]),  # Step 1: initial entities
            _make_llm_json(["niche tool search"]),  # Step 3: gap queries
            _make_llm_json(["NicheTool"]),  # Step 3: gap entity extraction
            "synthesis report",  # Step 4: synthesis
        ]

        await _run_plan_job("test-job-4", req, pg)

    assert final_result.get("entities_evaluated") == 2, (
        f"Expected 2 entities (ToolA + NicheTool), got {final_result.get('entities_evaluated')}"
    )
    entity_names = [s["entity"] for s in final_result.get("sources", [])]
    assert "NicheTool" in entity_names, f"NicheTool missing from sources: {entity_names}"


# ── _dewie_search helper ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dewie_search_returns_empty_on_error():
    """_dewie_search should return [] on any network error, never raise."""
    import httpx

    from dewie.api.routes.investigate import _dewie_search

    with patch("dewie.api.routes.investigate.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("connection refused")
        mock_cls.return_value = mock_client

        result = await _dewie_search("test query")

    assert result == []


@pytest.mark.asyncio
async def test_dewie_search_returns_results():
    """_dewie_search parses and returns results from a successful response."""
    from dewie.api.routes.investigate import _dewie_search

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(
        return_value={
            "results": [{"title": "Doc A", "score": 0.9}, {"title": "Doc B", "score": 0.7}]
        }
    )

    with patch("dewie.api.routes.investigate.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=fake_resp)
        mock_cls.return_value = mock_client

        result = await _dewie_search("test query", limit=2)

    assert len(result) == 2
    assert result[0]["title"] == "Doc A"


# ── _dewey_search: corpus-first, web fallback ─────────────────────────────────


@pytest.mark.asyncio
async def test_dewey_search_returns_corpus_hit_without_brave():
    """When corpus returns a high-score hit, Brave must NOT be called."""
    import asyncio

    from dewie.api.routes.investigate_v2 import _dewey_search

    sem = asyncio.Semaphore(5)
    corpus_results = [{"score": 0.8, "snippet": "This is the corpus answer.", "url": ""}]

    brave_called = []

    async def fake_brave(query, count):
        brave_called.append(query)
        return []

    with (
        patch(
            "dewie.api.routes.investigate._dewie_search",
            new_callable=AsyncMock,
            return_value=corpus_results,
        ),
        patch("dewie.api.routes.investigate._brave_search", side_effect=fake_brave),
    ):
        text, source_url = await _dewey_search("some query", sem)

    assert text == "This is the corpus answer."
    assert source_url is None, "corpus hit should return source_url=None"
    assert brave_called == [], (
        f"Brave should not be called on corpus hit, called with: {brave_called}"
    )


@pytest.mark.asyncio
async def test_dewey_search_falls_back_to_brave_on_corpus_miss():
    """When corpus returns nothing above threshold, Brave + fetch should be used."""
    import asyncio

    from dewie.api.routes.investigate_v2 import _dewey_search

    sem = asyncio.Semaphore(5)
    corpus_results = [{"score": 0.2, "snippet": "low score hit"}]  # below threshold

    with (
        patch(
            "dewie.api.routes.investigate._dewie_search",
            new_callable=AsyncMock,
            return_value=corpus_results,
        ),
        patch(
            "dewie.api.routes.investigate._brave_search",
            new_callable=AsyncMock,
            return_value=[{"title": "Web page", "url": "http://example.com/page"}],
        ),
        patch(
            "dewie.api.routes.investigate._fetch_url",
            new_callable=AsyncMock,
            return_value=(True, "Fetched web content here.", None),
        ),
    ):
        text, source_url = await _dewey_search("some query", sem)

    assert text == "Fetched web content here."
    assert source_url == "http://example.com/page"


@pytest.mark.asyncio
async def test_dewey_search_returns_empty_on_total_miss():
    """When both corpus and web return nothing, _dewey_search returns ('', None)."""
    import asyncio

    from dewie.api.routes.investigate_v2 import _dewey_search

    sem = asyncio.Semaphore(5)

    with (
        patch(
            "dewie.api.routes.investigate._dewie_search",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "dewie.api.routes.investigate._brave_search",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        text, source_url = await _dewey_search("obscure query", sem)

    assert text == ""
    assert source_url is None
