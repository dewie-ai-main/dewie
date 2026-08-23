# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Agentic research endpoint — /research/agent

Two-tier research loop:
  quick: decompose → parallel search → relevance filter → synthesize (~3 LLM calls)
  deep:  full iterative loop with up to max_iterations rounds

Design doc: docs/features/agentic-research-endpoint.md

AQ (answers_questions) is NEVER exposed in any response model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sqlalchemy import text

from dewie.api.middleware import limiter, rate_limit
from dewie.model_adapter import LLMResponse, ModelClient
from dewie.storage.postgres import PostgresClient

log = logging.getLogger("dewie.api")

# Fields that must be redacted from log output
_SENSITIVE_FIELDS = {"api_key", "password", "token", "secret", "authorization"}


def _redact(value: str | None) -> str | None:
    """Redact a potentially sensitive string value for logging."""
    if value is None:
        return None
    if len(value) > 1000:
        value = value[:1000] + "... [truncated]"
    for field in _SENSITIVE_FIELDS:
        if field in value.lower():
            return "***REDACTED***"
    return value


def _extract_request_id(request: Request) -> str:
    """Extract request_id from request state, falling back to 'unknown'."""
    return getattr(request.state, "request_id", "unknown")

router = APIRouter(tags=["research"])

# ── Constants ─────────────────────────────────────────────────────────────────

RELEVANCE_THRESHOLD = 0.4
DEFAULT_LOOP_MODEL = None  # falls back to env AGENT_MODEL / LLM_MODEL
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_MAX_DOCS_PER_SEARCH = 5
DEFAULT_MAX_DOCS_TOTAL = 20
COST_PER_1K = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-5-mini": {"input": 0.00015, "output": 0.0006},
    "default": {"input": 0.001, "output": 0.002},
}

# ── Usage tracking ────────────────────────────────────────────────────────────


@dataclass
class UsageAccumulator:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    def add(self, response: LLMResponse) -> None:
        self.prompt_tokens += response.input_tokens or 0
        self.completion_tokens += response.output_tokens or 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        pricing = COST_PER_1K.get(self.model, COST_PER_1K["default"])
        return (
            self.prompt_tokens / 1000 * pricing["input"]
            + self.completion_tokens / 1000 * pricing["output"]
        )


# ── Request / Response models ─────────────────────────────────────────────────


class AgentResearchRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=2000, description="The research question to answer."
    )
    mode: Literal["quick", "deep"] = Field(
        default="quick", description="quick (~3 LLM calls) or deep (iterative)."
    )
    max_iterations: int = Field(
        default=DEFAULT_MAX_ITERATIONS,
        ge=1,
        le=8,
        description="Max search-evaluate rounds (deep mode only).",
    )
    max_docs_per_search: int = Field(
        default=DEFAULT_MAX_DOCS_PER_SEARCH, ge=1, le=15, description="Per-search result limit."
    )
    max_docs_total: int = Field(
        default=DEFAULT_MAX_DOCS_TOTAL, ge=1, le=50, description="Hard cap on docs considered."
    )
    model: str | None = Field(
        default=None, description="LLM override for all calls in this request."
    )
    web_fallback: bool = Field(
        default=False, description="Fall back to web search when corpus has gaps."
    )
    corpus_id: str | None = Field(
        default=None, description="Scope to a specific corpus (default: all)."
    )


class DocRef(BaseModel):
    doc_id: str
    title: str
    url: str | None = None
    source: str | None = None
    score: float
    relevance: float = Field(description="Relevance score from the filter step (0.0-1.0).")


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    estimated_cost_usd: float


class AgentResearchResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    mode: Literal["quick", "deep"]
    docs_used: list[DocRef]
    docs_discarded: int
    gaps: list[str]
    web_results_used: int
    iterations: int
    usage: UsageInfo
    trace: list[str]
    query: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_pg(request: Request) -> PostgresClient:
    return request.app.state.postgres


def _resolve_model(override: str | None) -> str:
    """Explicit override → env vars → configured chat model (dewie.yml)."""
    explicit = override or os.environ.get("AGENT_MODEL") or os.environ.get("LLM_MODEL")
    if explicit:
        return explicit
    from dewie.config import settings

    return settings.chat_model_aq or ""


async def _llm(
    messages: list[dict[str, str]],
    model: str,
    accumulator: UsageAccumulator,
    max_tokens: int = 800,
) -> str:
    """Single LLM call, accumulates usage, returns content string."""
    accumulator.model = model
    async with ModelClient(model=model) as client:
        resp = await client.complete(messages, max_tokens=max_tokens)
    accumulator.add(resp)
    return resp.content or ""


async def _search(
    pg: PostgresClient,
    query: str,
    limit: int,
    corpus_id: str | None,
) -> list[tuple[Any, float]]:
    """Search with the default hybrid ranker, optional corpus scope.

    Uses "rrf" (same ranker as the search_corpus MCP tool) rather than
    "answers_questions_rrf": the AQ ranker degrades badly on corpora where
    document_aq was never populated, and its AQ-FTS signal depends on
    enrichment writing self-contained questions.
    """
    try:
        results = await pg.search(
            query=query,
            limit=limit,
            ranker="rrf",
            corpus_id=corpus_id,
        )
        return results
    except TypeError:
        # corpus_id param may not be supported on older pg client
        results = await pg.search(query=query, limit=limit, ranker="rrf")
        if corpus_id:
            results = [
                (d, s) for d, s in results if str(getattr(d, "corpus_id", "") or "") == corpus_id
            ]
        return results
    except Exception as exc:
        log.warning("Search failed (%s), falling back to rrf: %s", query[:50], exc)
        try:
            return await pg.search(query=query, limit=limit, ranker="rrf")
        except Exception:
            return []


async def _fetch_embed_summaries(
    pg: PostgresClient, doc_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Fetch doc metadata including embed_summary for relevance evaluation."""
    if not doc_ids:
        return {}
    is_sqlite = getattr(pg, "_is_sqlite", False)
    if is_sqlite:
        placeholders = ",".join(f":id{i}" for i in range(len(doc_ids)))
        params = {f"id{i}": v for i, v in enumerate(doc_ids)}
        sql = text(f"""
            SELECT id, title, embed_summary, summary, url, source
            FROM documents
            WHERE id IN ({placeholders}) AND status = 'ready'
        """)
    else:
        params = {"ids": doc_ids}
        sql = text("""
            SELECT id::text, title, embed_summary, summary, url, source
            FROM documents
            WHERE id = ANY(:ids) AND status = 'ready'
        """)
    async with pg._session_factory() as session:
        rows = await session.execute(sql, params)
        return {
            str(r[0]): {
                "title": r[1],
                "embed_summary": r[2] or r[3] or "",
                "summary": r[3] or "",
                "url": r[4],
                "source": r[5],
            }
            for r in rows.fetchall()
        }


def _extract_json_block(text: str) -> Any:
    """Extract first JSON object/array from an LLM response."""
    # Try raw parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find JSON block in markdown fences
    import re

    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Find first { or [ and try to parse from there
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        idx = text.find(start_char)
        if idx >= 0:
            # Find matching close
            depth = 0
            for i, ch in enumerate(text[idx:], idx):
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[idx : i + 1])
                        except json.JSONDecodeError:
                            break
    return None


# ── Step implementations ──────────────────────────────────────────────────────


async def _decompose(query: str, model: str, usage: UsageAccumulator) -> list[str]:
    """Decompose query into 2-4 sub-questions for parallel search."""
    prompt = (
        f"Given this research question, generate 2-4 focused sub-questions that would help "
        f"retrieve the most relevant documents from a knowledge corpus. "
        f"Return ONLY a JSON array of strings, nothing else.\n\n"
        f"Research question: {query}"
    )
    raw = await _llm(
        [{"role": "user", "content": prompt}],
        model=model,
        accumulator=usage,
        max_tokens=300,
    )
    parsed = _extract_json_block(raw)
    if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
        return parsed[:4]
    # Fallback: use original query
    log.warning("Decompose failed to parse JSON, using original query. raw=%r", raw[:200])
    return [query]


async def _evaluate_relevance(
    query: str,
    docs: list[dict[str, Any]],
    model: str,
    usage: UsageAccumulator,
) -> list[float]:
    """
    Score each doc's relevance to the original query (0.0-1.0).
    Returns a list of floats in the same order as docs.
    """
    if not docs:
        return []

    doc_summaries = []
    for i, d in enumerate(docs):
        title = d.get("title") or "(untitled)"
        text_content = (d.get("embed_summary") or d.get("summary") or "")[:1200]
        doc_summaries.append(f"[{i}] {title}\n{text_content}")

    prompt = (
        f"Rate how relevant each document is to answering this question:\n"
        f"Question: {query}\n\n"
        f"Documents:\n" + "\n\n".join(doc_summaries) + "\n\n"
        f"Return ONLY a JSON array of {len(docs)} floats between 0.0 and 1.0 "
        f"(one per document, in order). Higher = more relevant."
    )
    raw = await _llm(
        [{"role": "user", "content": prompt}],
        model=model,
        accumulator=usage,
        max_tokens=200,
    )
    parsed = _extract_json_block(raw)
    if isinstance(parsed, list) and len(parsed) == len(docs):
        try:
            return [min(1.0, max(0.0, float(v))) for v in parsed]
        except (TypeError, ValueError):
            pass
    # Fallback: score everything as 0.5
    log.warning("Relevance eval failed to parse. raw=%r", raw[:200])
    return [0.5] * len(docs)


async def _reflect(
    query: str,
    relevant_docs: list[dict[str, Any]],
    model: str,
    usage: UsageAccumulator,
) -> dict[str, Any]:
    """
    Given the docs collected so far, decide if we have enough to answer.
    Returns {sufficient: bool, missing: [str], next_queries: [str]}
    """
    if not relevant_docs:
        return {"sufficient": False, "missing": [query], "next_queries": [query]}

    summaries = "\n\n".join(
        f"[{i}] {d.get('title', '')}: {(d.get('embed_summary') or d.get('summary') or '')[:600]}"
        for i, d in enumerate(relevant_docs[:10])
    )
    prompt = (
        f"Given these documents, can you fully answer: '{query}'?\n\n"
        f"Documents:\n{summaries}\n\n"
        f"Return ONLY a JSON object with:\n"
        f"  sufficient: true/false (do the docs contain enough to answer the question?)\n"
        f"  missing: array of strings (aspects still not covered, max 3)\n"
        f"  next_queries: array of strings (follow-up searches if not sufficient, max 2)"
    )
    raw = await _llm(
        [{"role": "user", "content": prompt}],
        model=model,
        accumulator=usage,
        max_tokens=300,
    )
    parsed = _extract_json_block(raw)
    if isinstance(parsed, dict):
        return {
            "sufficient": bool(parsed.get("sufficient", False)),
            "missing": parsed.get("missing", [])[:3],
            "next_queries": parsed.get("next_queries", [])[:2],
        }
    return {"sufficient": True, "missing": [], "next_queries": []}


async def _synthesize(
    query: str,
    docs: list[dict[str, Any]],
    gaps: list[str],
    model: str,
    usage: UsageAccumulator,
) -> tuple[str, float]:
    """Synthesize final answer. Returns (answer, confidence)."""
    context_parts = []
    for i, d in enumerate(docs[:15], start=1):
        title = d.get("title") or "(untitled)"
        text_content = (d.get("embed_summary") or d.get("summary") or "").strip()
        if text_content:
            context_parts.append(f"[{i}] {title}\n{text_content[:800]}")

    context = "\n\n".join(context_parts) if context_parts else "(no relevant documents found)"
    gap_note = f"\n\nNote: The corpus has limited coverage of: {', '.join(gaps)}." if gaps else ""

    system = (
        "You are a research assistant. Answer the user's question using ONLY the provided documents. "
        "Cite document numbers in brackets (e.g. [1], [2]) where relevant. "
        "If documents are insufficient, say so clearly and explain what's missing. "
        "Be concise and factual."
    )
    user_msg = (
        f"Question: {query}\n\nDocuments:\n{context}{gap_note}\n\nProvide a thorough, cited answer."
    )
    answer = await _llm(
        [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        model=model,
        accumulator=usage,
        max_tokens=1500,
    )

    cited = sum(1 for i in range(1, len(docs) + 1) if f"[{i}]" in answer)
    confidence = min(1.0, 0.3 + 0.1 * cited + (0.2 if docs else 0.0))
    return answer or "(no answer generated)", confidence


# ── Quick mode ────────────────────────────────────────────────────────────────


async def _run_quick(
    pg: PostgresClient,
    body: AgentResearchRequest,
    model: str,
    usage: UsageAccumulator,
    trace: list[str],
) -> tuple[list[DocRef], list[DocRef], list[str], str, float, int]:
    """
    Quick mode: decompose → parallel search → filter → synthesize.
    Returns (docs_used, docs_discarded_refs, gaps, answer, confidence, web_used).
    """
    # 1. Decompose
    trace.append("decompose: generating sub-questions")
    sub_questions = await _decompose(body.query, model, usage)
    trace.append(f"decompose: {len(sub_questions)} sub-questions: {sub_questions}")

    # 2. Parallel search
    trace.append("search: running parallel searches")
    search_tasks = [_search(pg, q, body.max_docs_per_search, body.corpus_id) for q in sub_questions]
    all_results_nested = await asyncio.gather(*search_tasks, return_exceptions=True)

    seen: dict[str, float] = {}
    for results in all_results_nested:
        if isinstance(results, Exception):
            continue
        for doc, score in results:
            doc_id = str(doc.id)
            if doc_id not in seen or score > seen[doc_id]:
                seen[doc_id] = score

    trace.append(f"search: {len(seen)} unique docs retrieved")

    # 3. Fetch embed summaries
    doc_ids = list(seen.keys())[: body.max_docs_total]
    doc_meta = await _fetch_embed_summaries(pg, doc_ids)

    # Build ordered list by score
    ordered = sorted(
        [(doc_id, seen[doc_id], doc_meta[doc_id]) for doc_id in doc_ids if doc_id in doc_meta],
        key=lambda x: -x[1],
    )

    # 4. Relevance filter
    trace.append("filter: evaluating relevance")
    docs_for_eval = [m for _, _, m in ordered]
    scores = await _evaluate_relevance(body.query, docs_for_eval, model, usage)

    docs_used: list[DocRef] = []
    discarded: list[DocRef] = []
    for (doc_id, search_score, meta), rel_score in zip(ordered, scores):
        ref = DocRef(
            doc_id=doc_id,
            title=meta.get("title") or "",
            url=meta.get("url"),
            source=meta.get("source"),
            score=round(search_score, 4),
            relevance=round(rel_score, 3),
        )
        if rel_score >= RELEVANCE_THRESHOLD:
            docs_used.append(ref)
        else:
            discarded.append(ref)

    trace.append(
        f"filter: {len(docs_used)} relevant, {len(discarded)} discarded (threshold {RELEVANCE_THRESHOLD})"
    )

    # 5. Synthesize
    trace.append("synthesize: generating answer")
    relevant_metas = [doc_meta[r.doc_id] for r in docs_used if r.doc_id in doc_meta]
    answer, confidence = await _synthesize(body.query, relevant_metas, [], model, usage)
    trace.append(f"synthesize: done (confidence {confidence:.2f})")

    return docs_used, discarded, [], answer, confidence, 0


# ── Deep mode ─────────────────────────────────────────────────────────────────


async def _run_deep(
    pg: PostgresClient,
    body: AgentResearchRequest,
    model: str,
    usage: UsageAccumulator,
    trace: list[str],
) -> tuple[list[DocRef], list[DocRef], list[str], str, float, int, int]:
    """
    Deep mode: iterative search-evaluate-reflect loop.
    Returns (docs_used, discarded, gaps, answer, confidence, iterations, web_used).
    """
    all_relevant: dict[
        str, tuple[float, float, dict[str, Any]]
    ] = {}  # doc_id → (search_score, rel_score, meta)
    all_discarded: dict[str, float] = {}
    gaps: list[str] = []
    web_results_used = 0
    queries_tried: set[str] = set()

    # Initial decompose
    trace.append("decompose: generating sub-questions")
    sub_questions = await _decompose(body.query, model, usage)
    trace.append(f"decompose: {sub_questions}")
    pending_queries = sub_questions[:]

    for iteration in range(1, body.max_iterations + 1):
        if not pending_queries:
            trace.append(f"iteration {iteration}: no more queries, stopping")
            break

        trace.append(f"iteration {iteration}: searching {len(pending_queries)} queries")

        # Search
        new_doc_ids: dict[str, float] = {}
        for q in pending_queries:
            if q in queries_tried:
                continue
            queries_tried.add(q)
            results = await _search(pg, q, body.max_docs_per_search, body.corpus_id)
            for doc, score in results:
                doc_id = str(doc.id)
                if doc_id not in all_relevant and doc_id not in all_discarded:
                    if doc_id not in new_doc_ids or score > new_doc_ids[doc_id]:
                        new_doc_ids[doc_id] = score

        # Fetch metadata
        new_meta = await _fetch_embed_summaries(pg, list(new_doc_ids.keys()))
        ordered_new = sorted(
            [(did, new_doc_ids[did], new_meta[did]) for did in new_doc_ids if did in new_meta],
            key=lambda x: -x[1],
        )
        trace.append(f"iteration {iteration}: {len(ordered_new)} new docs retrieved")

        # Evaluate
        if ordered_new:
            docs_for_eval = [m for _, _, m in ordered_new]
            rel_scores = await _evaluate_relevance(body.query, docs_for_eval, model, usage)
            for (doc_id, search_score, meta), rel_score in zip(ordered_new, rel_scores):
                if len(all_relevant) >= body.max_docs_total:
                    break
                if rel_score >= RELEVANCE_THRESHOLD:
                    all_relevant[doc_id] = (search_score, rel_score, meta)
                else:
                    all_discarded[doc_id] = rel_score

        trace.append(f"iteration {iteration}: {len(all_relevant)} relevant total")

        # Reflect — decide if we have enough
        relevant_metas = [m for _, _, m in all_relevant.values()]
        reflection = await _reflect(body.query, relevant_metas, model, usage)
        trace.append(
            f"iteration {iteration}: sufficient={reflection['sufficient']}, missing={reflection.get('missing', [])}"
        )

        if reflection["sufficient"]:
            break

        # Not sufficient — use next_queries for next round
        new_gaps = reflection.get("missing", [])
        for g in new_gaps:
            if g not in gaps:
                gaps.append(g)

        pending_queries = reflection.get("next_queries", [])

        # Web fallback for gaps
        if body.web_fallback and gaps and iteration == body.max_iterations:
            trace.append("web_fallback: fetching web results for gaps")
            web_results_used = await _web_fallback(gaps[:3], relevant_metas)

    # Synthesize
    trace.append("synthesize: generating final answer")
    relevant_metas = [m for _, _, m in all_relevant.values()]
    answer, confidence = await _synthesize(body.query, relevant_metas, gaps, model, usage)
    trace.append(f"synthesize: done (confidence {confidence:.2f})")

    docs_used = [
        DocRef(
            doc_id=doc_id,
            title=meta.get("title") or "",
            url=meta.get("url"),
            source=meta.get("source"),
            score=round(search_score, 4),
            relevance=round(rel_score, 3),
        )
        for doc_id, (search_score, rel_score, meta) in all_relevant.items()
    ]
    discarded = [
        DocRef(doc_id=doc_id, title="", score=0.0, relevance=round(rel_score, 3))
        for doc_id, rel_score in all_discarded.items()
    ]

    return docs_used, discarded, gaps, answer, confidence, len(queries_tried), web_results_used


async def _web_fallback(gaps: list[str], relevant_metas: list[dict[str, Any]]) -> int:
    """Placeholder: web fallback not yet wired. Returns 0."""
    return 0


# ── Route ─────────────────────────────────────────────────────────────────────


@router.post("/research/agent", response_model=AgentResearchResponse)
@limiter.limit(rate_limit(5))
async def research_agent(request: Request, body: AgentResearchRequest) -> AgentResearchResponse:
    """
    Agentic research with active relevance filtering and gap detection.

    quick mode (~3 LLM calls, 2-8s):
      decompose → parallel search → relevance filter → synthesize

    deep mode (iterative, up to max_iterations rounds):
      decompose → search → evaluate → reflect → (repeat if needed) → synthesize

    Token usage is tracked per request. AQ data is never exposed.
    """
    request_id = _extract_request_id(request)
    log.info("research_agent started request_id=%s mode=%s query=%s", request_id, body.mode, _redact(body.query))
    t0 = time.monotonic()
    pg = _get_pg(request)
    model = _resolve_model(body.model)
    usage = UsageAccumulator(model=model)
    trace: list[str] = [f"mode={body.mode}, model={model}"]

    if body.mode == "quick":
        docs_used, discarded, gaps, answer, confidence, web_used = await _run_quick(
            pg, body, model, usage, trace
        )
        iterations = 1
    else:
        docs_used, discarded, gaps, answer, confidence, iterations, web_used = await _run_deep(
            pg, body, model, usage, trace
        )

    elapsed = round((time.monotonic() - t0) * 1000, 2)
    log.info("research_agent succeeded request_id=%s mode=%s iterations=%d docs_used=%d elapsed_ms=%.2f", request_id, body.mode, iterations, len(docs_used), elapsed)

    return AgentResearchResponse(
        answer=answer,
        confidence=confidence,
        mode=body.mode,
        docs_used=docs_used,
        docs_discarded=len(discarded),
        gaps=gaps,
        web_results_used=web_used,
        iterations=iterations,
        usage=UsageInfo(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            model=usage.model,
            estimated_cost_usd=round(usage.estimated_cost_usd, 6),
        ),
        trace=trace,
        query=body.query,
    )
