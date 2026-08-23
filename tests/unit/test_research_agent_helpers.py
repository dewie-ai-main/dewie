"""
Unit tests for pure-Python helpers in dewie.api.routes.research_agent.

Covers:
  - UsageAccumulator — token counting + cost estimation
  - AgentResearchRequest — Pydantic field validation
  - _extract_json_block — JSON extraction from LLM text
  - _resolve_model — env-based model resolution
  - POST /research/agent — HTTP endpoint (quick + deep mode, validation)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dewie.api.middleware import limiter

# ── _disable_rate_limiting fixture ────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _disable_rate_limiting(monkeypatch):
    monkeypatch.setattr(limiter, "enabled", False)


# ══════════════════════════════════════════════════════════════════════════════
# UsageAccumulator
# ══════════════════════════════════════════════════════════════════════════════


class TestUsageAccumulator:
    def test_initial_state(self):
        from dewie.api.routes.research_agent import UsageAccumulator

        acc = UsageAccumulator()
        assert acc.prompt_tokens == 0
        assert acc.completion_tokens == 0
        assert acc.total_tokens == 0
        assert acc.estimated_cost_usd == pytest.approx(0.0)

    def test_add_increments_tokens(self):
        from dewie.api.routes.research_agent import UsageAccumulator
        from dewie.model_adapter import LLMResponse

        acc = UsageAccumulator()
        resp = MagicMock(spec=LLMResponse)
        resp.input_tokens = 100
        resp.output_tokens = 50
        acc.add(resp)
        assert acc.prompt_tokens == 100
        assert acc.completion_tokens == 50
        assert acc.total_tokens == 150

    def test_add_multiple_responses(self):
        from dewie.api.routes.research_agent import UsageAccumulator
        from dewie.model_adapter import LLMResponse

        acc = UsageAccumulator()
        for _ in range(3):
            resp = MagicMock(spec=LLMResponse)
            resp.input_tokens = 10
            resp.output_tokens = 20
            acc.add(resp)
        assert acc.prompt_tokens == 30
        assert acc.completion_tokens == 60
        assert acc.total_tokens == 90

    def test_add_none_tokens_handled(self):
        from dewie.api.routes.research_agent import UsageAccumulator
        from dewie.model_adapter import LLMResponse

        acc = UsageAccumulator()
        resp = MagicMock(spec=LLMResponse)
        resp.input_tokens = None
        resp.output_tokens = None
        acc.add(resp)
        assert acc.prompt_tokens == 0
        assert acc.completion_tokens == 0

    def test_cost_known_model(self):
        from dewie.api.routes.research_agent import UsageAccumulator

        acc = UsageAccumulator(prompt_tokens=1000, completion_tokens=1000, model="gpt-4o")
        expected = 1.0 * 0.005 + 1.0 * 0.015  # 1k input + 1k output at gpt-4o rates
        assert acc.estimated_cost_usd == pytest.approx(expected)

    def test_cost_unknown_model_uses_default(self):
        from dewie.api.routes.research_agent import COST_PER_1K, UsageAccumulator

        acc = UsageAccumulator(prompt_tokens=1000, completion_tokens=1000, model="unknown-xyz")
        default = COST_PER_1K["default"]
        expected = 1.0 * default["input"] + 1.0 * default["output"]
        assert acc.estimated_cost_usd == pytest.approx(expected)

    def test_cost_zero_tokens(self):
        from dewie.api.routes.research_agent import UsageAccumulator

        acc = UsageAccumulator(model="gpt-4o")
        assert acc.estimated_cost_usd == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# AgentResearchRequest validation
# ══════════════════════════════════════════════════════════════════════════════


class TestAgentResearchRequest:
    def test_defaults(self):
        from dewie.api.routes.research_agent import AgentResearchRequest

        req = AgentResearchRequest(query="What is AI?")
        assert req.mode == "quick"
        assert req.max_iterations == 3
        assert req.max_docs_per_search == 5
        assert req.max_docs_total == 20
        assert req.web_fallback is False
        assert req.model is None
        assert req.corpus_id is None

    def test_deep_mode_accepted(self):
        from dewie.api.routes.research_agent import AgentResearchRequest

        req = AgentResearchRequest(query="Q?", mode="deep")
        assert req.mode == "deep"

    def test_invalid_mode_rejected(self):
        import pydantic

        from dewie.api.routes.research_agent import AgentResearchRequest

        with pytest.raises(pydantic.ValidationError):
            AgentResearchRequest(query="Q?", mode="invalid")

    def test_empty_query_rejected(self):
        import pydantic

        from dewie.api.routes.research_agent import AgentResearchRequest

        with pytest.raises(pydantic.ValidationError):
            AgentResearchRequest(query="")

    def test_query_too_long_rejected(self):
        import pydantic

        from dewie.api.routes.research_agent import AgentResearchRequest

        with pytest.raises(pydantic.ValidationError):
            AgentResearchRequest(query="x" * 2001)

    def test_max_iterations_bounds(self):
        import pydantic

        from dewie.api.routes.research_agent import AgentResearchRequest

        with pytest.raises(pydantic.ValidationError):
            AgentResearchRequest(query="Q?", max_iterations=0)
        with pytest.raises(pydantic.ValidationError):
            AgentResearchRequest(query="Q?", max_iterations=9)

    def test_max_docs_per_search_bounds(self):
        import pydantic

        from dewie.api.routes.research_agent import AgentResearchRequest

        with pytest.raises(pydantic.ValidationError):
            AgentResearchRequest(query="Q?", max_docs_per_search=0)
        with pytest.raises(pydantic.ValidationError):
            AgentResearchRequest(query="Q?", max_docs_per_search=16)

    def test_corpus_id_optional(self):
        from dewie.api.routes.research_agent import AgentResearchRequest

        req = AgentResearchRequest(query="Q?", corpus_id="my-corpus")
        assert req.corpus_id == "my-corpus"

    def test_model_override(self):
        from dewie.api.routes.research_agent import AgentResearchRequest

        req = AgentResearchRequest(query="Q?", model="gpt-4o-mini")
        assert req.model == "gpt-4o-mini"


# ══════════════════════════════════════════════════════════════════════════════
# _extract_json_block
# ══════════════════════════════════════════════════════════════════════════════


class TestExtractJsonBlock:
    def test_plain_json_object(self):
        from dewie.api.routes.research_agent import _extract_json_block

        result = _extract_json_block('{"key": "value"}')
        assert result == {"key": "value"}

    def test_plain_json_array(self):
        from dewie.api.routes.research_agent import _extract_json_block

        result = _extract_json_block('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_json_in_markdown_fence(self):
        from dewie.api.routes.research_agent import _extract_json_block

        text = '```json\n{"key": "value"}\n```'
        result = _extract_json_block(text)
        assert result == {"key": "value"}

    def test_json_in_plain_fence(self):
        from dewie.api.routes.research_agent import _extract_json_block

        text = '```\n["a", "b"]\n```'
        result = _extract_json_block(text)
        assert result == ["a", "b"]

    def test_json_embedded_in_prose(self):
        from dewie.api.routes.research_agent import _extract_json_block

        text = 'Here is the answer: {"score": 0.9} That is it.'
        result = _extract_json_block(text)
        assert result == {"score": 0.9}

    def test_array_embedded_in_prose(self):
        from dewie.api.routes.research_agent import _extract_json_block

        text = 'The sub-questions are: ["q1", "q2", "q3"]. Done.'
        result = _extract_json_block(text)
        assert result == ["q1", "q2", "q3"]

    def test_invalid_json_returns_none(self):
        from dewie.api.routes.research_agent import _extract_json_block

        result = _extract_json_block("This has no JSON at all.")
        assert result is None

    def test_empty_string_returns_none(self):
        from dewie.api.routes.research_agent import _extract_json_block

        result = _extract_json_block("")
        assert result is None

    def test_nested_object(self):
        from dewie.api.routes.research_agent import _extract_json_block

        result = _extract_json_block('{"a": {"b": [1, 2, 3]}}')
        assert result == {"a": {"b": [1, 2, 3]}}


# ══════════════════════════════════════════════════════════════════════════════
# _resolve_model
# ══════════════════════════════════════════════════════════════════════════════


class TestResolveModel:
    def test_override_takes_priority(self):
        from dewie.api.routes.research_agent import _resolve_model

        assert _resolve_model("gpt-4o-mini") == "gpt-4o-mini"

    def test_agent_model_env(self, monkeypatch):
        from dewie.api.routes.research_agent import _resolve_model

        monkeypatch.setenv("AGENT_MODEL", "claude-3-opus")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        assert _resolve_model(None) == "claude-3-opus"

    def test_llm_model_env_fallback(self, monkeypatch):
        from dewie.api.routes.research_agent import _resolve_model

        monkeypatch.delenv("AGENT_MODEL", raising=False)
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        assert _resolve_model(None) == "gpt-4o"

    def test_config_fallback_when_no_env(self, monkeypatch):
        """No override/env → falls back to the configured chat model."""
        from dewie.api.routes.research_agent import _resolve_model
        from dewie.config import settings

        monkeypatch.delenv("AGENT_MODEL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.setattr(settings, "chat_model_aq", "configured-model")
        assert _resolve_model(None) == "configured-model"

    def test_default_when_no_env_and_no_config(self, monkeypatch):
        from dewie.api.routes.research_agent import _resolve_model
        from dewie.config import settings

        monkeypatch.delenv("AGENT_MODEL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.setattr(settings, "chat_model_aq", "")
        assert _resolve_model(None) == ""


# ══════════════════════════════════════════════════════════════════════════════
# HTTP endpoint tests
# ══════════════════════════════════════════════════════════════════════════════

_QUICK_DECOMPOSE_RESP = json.dumps(["sub-q 1", "sub-q 2"])
_RELEVANCE_RESP = json.dumps([{"doc_id": "doc-1", "relevance": 0.9, "reason": "relevant"}])
_SYNTHESIS_RESP = "The answer is: AI is useful."


def _make_doc(doc_id: str = "doc-1", title: str = "Test Doc"):
    doc = MagicMock()
    doc.id = doc_id
    doc.title = title
    doc.url = "https://example.com"
    doc.answers_questions = None  # AQ must never be exposed
    return doc


def _make_session_mock():
    session = AsyncMock()
    result = MagicMock()
    result.fetchall.return_value = [
        MagicMock(
            doc_id="doc-1",
            title="Test Doc",
            url="https://example.com",
            embed_summary="AI is useful.",
        )
    ]
    session.execute = AsyncMock(return_value=result)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _make_agent_app(pg):
    from dewie.api.middleware import limiter
    from dewie.api.routes.research_agent import router

    app = FastAPI()
    app.state.limiter = limiter

    @app.middleware("http")
    async def _inject_state(request, call_next):
        request.app.state.postgres = pg
        return await call_next(request)

    app.include_router(router)
    return app


@pytest.mark.asyncio
async def test_agent_endpoint_422_missing_body():
    """Missing body returns 422."""
    pg = AsyncMock()
    app = _make_agent_app(pg)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/research/agent")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_agent_endpoint_422_empty_query():
    """Empty query string returns 422."""
    pg = AsyncMock()
    app = _make_agent_app(pg)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/research/agent", json={"query": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_agent_endpoint_quick_mode_returns_200():
    """Quick mode returns 200 with answer."""
    pg = AsyncMock()
    doc = _make_doc()
    pg.search = AsyncMock(return_value=[(doc, 0.85)])
    pg._session_factory = MagicMock(return_value=_make_session_mock())

    app = _make_agent_app(pg)

    llm_calls = iter([_QUICK_DECOMPOSE_RESP, _RELEVANCE_RESP, _SYNTHESIS_RESP])

    async def fake_llm(messages, model=None, accumulator=None, max_tokens=800):
        resp = MagicMock()
        resp.content = next(llm_calls, _SYNTHESIS_RESP)
        resp.input_tokens = 100
        resp.output_tokens = 50
        if accumulator is not None:
            accumulator.add(resp)
        return resp.content

    with patch("dewie.api.routes.research_agent._llm", new=AsyncMock(side_effect=fake_llm)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/research/agent", json={"query": "What is AI?"})

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "confidence" in data
    assert data["mode"] == "quick"


@pytest.mark.asyncio
async def test_agent_response_no_answers_questions():
    """AQ field must never appear in response."""
    pg = AsyncMock()
    doc = _make_doc()
    pg.search = AsyncMock(return_value=[(doc, 0.85)])
    pg._session_factory = MagicMock(return_value=_make_session_mock())

    app = _make_agent_app(pg)

    async def fake_llm(messages, model=None, accumulator=None, max_tokens=800):
        resp = MagicMock()
        resp.content = _SYNTHESIS_RESP
        resp.input_tokens = 10
        resp.output_tokens = 10
        if accumulator is not None:
            accumulator.add(resp)
        return resp.content

    with patch("dewie.api.routes.research_agent._llm", new=AsyncMock(side_effect=fake_llm)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/research/agent", json={"query": "Test question"})

    body = resp.text
    assert "answers_questions" not in body


@pytest.mark.asyncio
async def test_agent_endpoint_deep_mode():
    """Deep mode returns 200 with iterations > 0."""
    pg = AsyncMock()
    doc = _make_doc()
    pg.search = AsyncMock(return_value=[(doc, 0.75)])
    pg._session_factory = MagicMock(return_value=_make_session_mock())

    app = _make_agent_app(pg)

    # Deep mode LLM calls: decompose, relevance, reflect (may repeat), synthesize
    def make_resp(content):
        r = MagicMock()
        r.content = content
        r.input_tokens = 50
        r.output_tokens = 25
        return r

    call_count = [0]

    async def fake_llm(messages, model=None, accumulator=None, max_tokens=800):
        call_count[0] += 1
        if call_count[0] == 1:
            content = json.dumps(["sub-q 1"])  # decompose
        elif call_count[0] == 2:
            content = json.dumps([{"doc_id": "doc-1", "relevance": 0.8, "reason": "ok"}])
        elif call_count[0] == 3:
            content = json.dumps({"continue": False, "gaps": [], "new_questions": []})
        else:
            content = "Final synthesis answer."
        r = make_resp(content)
        if accumulator is not None:
            accumulator.add(r)
        return content

    with patch("dewie.api.routes.research_agent._llm", new=AsyncMock(side_effect=fake_llm)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/research/agent",
                json={"query": "Deep research question", "mode": "deep", "max_iterations": 1},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "deep"
    assert data["iterations"] >= 1


@pytest.mark.asyncio
async def test_agent_usage_info_in_response():
    """Response includes usage tracking info."""
    pg = AsyncMock()
    doc = _make_doc()
    pg.search = AsyncMock(return_value=[(doc, 0.85)])
    pg._session_factory = MagicMock(return_value=_make_session_mock())

    app = _make_agent_app(pg)

    async def fake_llm(messages, model=None, accumulator=None, max_tokens=800):
        resp = MagicMock()
        resp.content = _SYNTHESIS_RESP
        resp.input_tokens = 200
        resp.output_tokens = 100
        if accumulator is not None:
            accumulator.add(resp)
        return resp.content

    with patch("dewie.api.routes.research_agent._llm", new=AsyncMock(side_effect=fake_llm)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/research/agent", json={"query": "What is AI?"})

    assert resp.status_code == 200
    data = resp.json()
    assert "usage" in data
    assert "total_tokens" in data["usage"]
    assert "estimated_cost_usd" in data["usage"]


@pytest.mark.asyncio
async def test_agent_empty_corpus_returns_answer():
    """When no documents match, endpoint still returns 200 with an answer."""
    pg = AsyncMock()
    pg.search = AsyncMock(return_value=[])
    pg._session_factory = MagicMock(return_value=_make_session_mock())

    app = _make_agent_app(pg)

    async def fake_llm(messages, model=None, accumulator=None, max_tokens=800):
        resp = MagicMock()
        resp.content = "No relevant documents found."
        resp.input_tokens = 50
        resp.output_tokens = 20
        if accumulator is not None:
            accumulator.add(resp)
        return resp.content

    with patch("dewie.api.routes.research_agent._llm", new=AsyncMock(side_effect=fake_llm)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/research/agent", json={"query": "obscure question"})

    assert resp.status_code == 200


# ── _llm helper ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_accumulates_usage_and_returns_content():
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.api.routes.research_agent import UsageAccumulator, _llm

    usage = UsageAccumulator()
    mock_resp = MagicMock()
    mock_resp.content = "Hello world"
    mock_resp.usage = MagicMock()
    mock_resp.usage.prompt_tokens = 10
    mock_resp.usage.completion_tokens = 5
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.complete = AsyncMock(return_value=mock_resp)
    with patch("dewie.api.routes.research_agent.ModelClient", return_value=mock_client):
        result = await _llm(
            [{"role": "user", "content": "hi"}], model="gpt-4o-mini", accumulator=usage
        )
    assert result == "Hello world"
    assert usage.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_llm_returns_empty_string_for_none_content():
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.api.routes.research_agent import UsageAccumulator, _llm

    usage = UsageAccumulator()
    mock_resp = MagicMock()
    mock_resp.content = None
    mock_resp.usage = None
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.complete = AsyncMock(return_value=mock_resp)
    with patch("dewie.api.routes.research_agent.ModelClient", return_value=mock_client):
        result = await _llm([], model="gpt-4o-mini", accumulator=usage)
    assert result == ""


# ── _search helper ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_returns_results_normally():
    from unittest.mock import AsyncMock, MagicMock

    from dewie.api.routes.research_agent import _search

    pg = MagicMock()
    doc = MagicMock()
    pg.search = AsyncMock(return_value=[(doc, 0.9)])
    results = await _search(pg, "machine learning", limit=5, corpus_id=None)
    assert len(results) == 1
    assert results[0][1] == 0.9
    pg.search.assert_called_once_with(
        query="machine learning", limit=5, ranker="rrf", corpus_id=None
    )


@pytest.mark.asyncio
async def test_search_typeerror_fallback_without_corpus():
    """TypeError (old pg client) falls back to call without corpus_id."""
    from unittest.mock import AsyncMock, MagicMock

    from dewie.api.routes.research_agent import _search

    pg = MagicMock()
    doc = MagicMock()
    call_count = {"n": 0}

    async def search_side_effect(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise TypeError("unexpected keyword argument 'corpus_id'")
        return [(doc, 0.85)]

    pg.search = AsyncMock(side_effect=search_side_effect)
    results = await _search(pg, "query", limit=3, corpus_id=None)
    assert len(results) == 1
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_search_typeerror_fallback_with_corpus_filters_results():
    """TypeError fallback with corpus_id filters by corpus_id attribute."""
    from unittest.mock import AsyncMock, MagicMock

    from dewie.api.routes.research_agent import _search

    pg = MagicMock()
    doc_in = MagicMock()
    doc_in.corpus_id = "corp-1"
    doc_out = MagicMock()
    doc_out.corpus_id = "corp-2"
    call_count = {"n": 0}

    async def search_side_effect(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise TypeError("unexpected")
        return [(doc_in, 0.9), (doc_out, 0.8)]

    pg.search = AsyncMock(side_effect=search_side_effect)
    results = await _search(pg, "query", limit=10, corpus_id="corp-1")
    # Only doc_in should pass the corpus filter
    assert len(results) == 1
    assert results[0][0] is doc_in


@pytest.mark.asyncio
async def test_search_exception_fallback_to_rrf():
    """General exception falls back to rrf ranker."""
    from unittest.mock import AsyncMock, MagicMock

    from dewie.api.routes.research_agent import _search

    pg = MagicMock()
    doc = MagicMock()
    call_count = {"n": 0}

    async def search_side_effect(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("connection failed")
        return [(doc, 0.7)]

    pg.search = AsyncMock(side_effect=search_side_effect)
    results = await _search(pg, "query", limit=5, corpus_id=None)
    assert len(results) == 1
    # Second call should use rrf ranker
    second_call = pg.search.call_args_list[1]
    assert second_call.kwargs.get("ranker") == "rrf"


@pytest.mark.asyncio
async def test_search_double_exception_returns_empty():
    """When both attempts fail, returns empty list."""
    from unittest.mock import AsyncMock, MagicMock

    from dewie.api.routes.research_agent import _search

    pg = MagicMock()
    pg.search = AsyncMock(side_effect=RuntimeError("always fails"))
    results = await _search(pg, "query", limit=5, corpus_id=None)
    assert results == []


# ── _fetch_embed_summaries ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_embed_summaries_empty_ids():
    from unittest.mock import MagicMock

    from dewie.api.routes.research_agent import _fetch_embed_summaries

    pg = MagicMock()
    result = await _fetch_embed_summaries(pg, [])
    assert result == {}
    pg._session_factory.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_embed_summaries_returns_dict():
    from unittest.mock import AsyncMock, MagicMock

    from dewie.api.routes.research_agent import _fetch_embed_summaries

    rows_result = MagicMock()
    rows_result.fetchall.return_value = [
        ("doc-1", "Title A", "Embed summary A", "Summary A", "https://a.com", "arxiv"),
        ("doc-2", "Title B", None, "Summary B", "https://b.com", "web"),
    ]
    session = AsyncMock()
    session.execute = AsyncMock(return_value=rows_result)
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    pg = MagicMock()
    pg._session_factory = MagicMock(return_value=session_cm)
    result = await _fetch_embed_summaries(pg, ["doc-1", "doc-2"])
    assert "doc-1" in result
    assert result["doc-1"]["title"] == "Title A"
    assert result["doc-1"]["embed_summary"] == "Embed summary A"
    assert result["doc-2"]["embed_summary"] == "Summary B"  # falls back to summary


# ── _reflect helper ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reflect_returns_not_sufficient_when_no_docs():
    from dewie.api.routes.research_agent import UsageAccumulator, _reflect

    usage = UsageAccumulator()
    result = await _reflect("test query", [], model="gpt-4o-mini", usage=usage)
    assert result["sufficient"] is False
    assert result["missing"] == ["test query"]


@pytest.mark.asyncio
async def test_reflect_parses_valid_llm_response():
    import json
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.api.routes.research_agent import UsageAccumulator, _reflect

    usage = UsageAccumulator()
    docs = [{"title": "Doc A", "embed_summary": "Summary A"}]
    llm_output = json.dumps(
        {
            "sufficient": True,
            "missing": [],
            "next_queries": [],
        }
    )
    mock_resp = MagicMock()
    mock_resp.content = llm_output
    mock_resp.usage = None
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.complete = AsyncMock(return_value=mock_resp)
    with patch("dewie.api.routes.research_agent.ModelClient", return_value=mock_client):
        result = await _reflect("test query", docs, model="gpt-4o-mini", usage=usage)
    assert result["sufficient"] is True
    assert result["missing"] == []


@pytest.mark.asyncio
async def test_reflect_handles_invalid_json_from_llm():
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.api.routes.research_agent import UsageAccumulator, _reflect

    usage = UsageAccumulator()
    docs = [{"title": "Doc A", "embed_summary": "Summary A"}]
    mock_resp = MagicMock()
    mock_resp.content = "I cannot determine this."  # not JSON
    mock_resp.usage = None
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.complete = AsyncMock(return_value=mock_resp)
    with patch("dewie.api.routes.research_agent.ModelClient", return_value=mock_client):
        result = await _reflect("test query", docs, model="gpt-4o-mini", usage=usage)
    # Falls back to sufficient=True when parse fails
    assert result["sufficient"] is True
    assert result["missing"] == []
    assert result["next_queries"] == []


# ── Structured logging tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_endpoint_logs_start():
    """research_agent logs a start record at INFO level."""
    import logging
    pg = AsyncMock()
    doc = _make_doc()
    pg.search = AsyncMock(return_value=[(doc, 0.85)])
    pg._session_factory = MagicMock(return_value=_make_session_mock())

    app = _make_agent_app(pg)

    async def fake_llm(messages, model=None, accumulator=None, max_tokens=800):
        resp = MagicMock()
        resp.content = _SYNTHESIS_RESP
        resp.input_tokens = 50
        resp.output_tokens = 20
        if accumulator is not None:
            accumulator.add(resp)
        return resp.content

    import pytest
    with pytest.LogCaptureFixture if False else __import__("contextlib").nullcontext():
        pass  # just import path check


    with patch("dewie.api.routes.research_agent._llm", new=AsyncMock(side_effect=fake_llm)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            import logging
            handler = logging.handlers_list = []
            # Use caplog equivalent via direct propagation check
            with __import__("unittest").mock.patch.object(
                __import__("logging").getLogger("dewie.api"),
                "info",
            ) as mock_info:
                resp = await client.post("/research/agent", json={"query": "What is AI?"})
                assert resp.status_code == 200
                calls = [str(c) for c in mock_info.call_args_list]
                assert any("research_agent started" in c for c in calls)


@pytest.mark.asyncio
async def test_agent_endpoint_logs_success():
    """research_agent logs a success record at INFO level."""
    pg = AsyncMock()
    doc = _make_doc()
    pg.search = AsyncMock(return_value=[(doc, 0.85)])
    pg._session_factory = MagicMock(return_value=_make_session_mock())

    app = _make_agent_app(pg)

    async def fake_llm(messages, model=None, accumulator=None, max_tokens=800):
        resp = MagicMock()
        resp.content = _SYNTHESIS_RESP
        resp.input_tokens = 50
        resp.output_tokens = 20
        if accumulator is not None:
            accumulator.add(resp)
        return resp.content

    with patch("dewie.api.routes.research_agent._llm", new=AsyncMock(side_effect=fake_llm)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with __import__("unittest").mock.patch.object(
                __import__("logging").getLogger("dewie.api"),
                "info",
            ) as mock_info:
                resp = await client.post("/research/agent", json={"query": "What is AI?"})
                assert resp.status_code == 200
                calls = [str(c) for c in mock_info.call_args_list]
                assert any("research_agent succeeded" in c for c in calls)
