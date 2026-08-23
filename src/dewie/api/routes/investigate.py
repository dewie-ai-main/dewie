# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Investigate endpoint — map-reduce investigation pipeline with web research.

POST /investigate

Pipeline stages:
  1. Decompose (LLM): query → 6-10 specific sub-questions
  2. Search (Brave): 3-5 URLs per sub-question, run in parallel
  3. Fetch: raw HTML from each unique URL (semaphore=5)
  4. Distill (LLM): extract concrete facts per page (semaphore=3)
  5. Aggregate: group facts by sub-question (no LLM)
  6. Synthesize (LLM): full report from all aggregated facts
  7. Compress (optional): reduce to token_budget if specified

Unlike /research (corpus-only), /investigate goes to the web first,
building the corpus on demand. Use when the corpus is sparse on a topic.

AQ data is never exposed in any response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from dewie.api.middleware import limiter, rate_limit
from dewie.config import settings

log = logging.getLogger("dewie.api")

router = APIRouter(tags=["investigate"])

# ── Sensitive field redaction ────────────────────────────────────────────────────

_SENSITIVE_KEYS = ("api_key", "api_key_", "token", "password", "secret")


def _redact_fields(d: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *d* with sensitive values redacted."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        k_lower = k.lower()
        if any(s in k_lower for s in _SENSITIVE_KEYS):
            if isinstance(v, str):
                out[k] = "***REDACTED***"
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _truncate(value: str, maxlen: int = 1000) -> str:
    """Truncate *value* to *maxlen* characters."""
    if len(value) > maxlen:
        return value[:maxlen] + f"... ({len(value)} chars total)"
    return value

# ── Config ────────────────────────────────────────────────────────────────────

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8080")
LLM_MODEL = os.environ.get("INVESTIGATE_LLM_MODEL", "gpt-4o-mini")
DEWIE_BASE = os.environ.get("DEWIE_API_URL", "http://localhost:10946")

FETCH_TIMEOUT = 15
FETCH_SEMAPHORE = 5
DISTILL_SEMAPHORE = 20  # max concurrent distill calls

# Browser fetch proxy — set BROWSER_FETCH_URL env var to use browser-backed fetching
# instead of raw httpx. Proxy runs on the Mac (scripts/browser_fetch_proxy.py).
# Example: BROWSER_FETCH_URL=http://localhost:8888/fetch
BROWSER_FETCH_URL = os.environ.get("BROWSER_FETCH_URL", "")


# ── Request / Response models ─────────────────────────────────────────────────


class InvestigateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    num_sources: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Max sources per sub-question (Brave results)",
    )
    ingest: bool = Field(
        default=True,
        description="Ingest fetched sources into Dewie corpus",
    )
    token_budget: int | None = Field(
        default=None,
        description="If set, compress final report to ~this many tokens",
    )
    model: str = Field(
        default=LLM_MODEL,
        description="Local model to use",
    )


class SourceResult(BaseModel):
    title: str
    url: str
    sub_question: str
    fetched: bool = False
    fetch_error: str | None = None
    fact_count: int = 0
    ingested: bool = False
    doc_id: str | None = None


class InvestigateResponse(BaseModel):
    report: str
    summary: str | None = None
    sub_questions: list[str]
    sources: list[SourceResult]
    total_facts: int
    query: str
    trace: list[str]


# ── LLM helpers ───────────────────────────────────────────────────────────────


async def _llm_call(
    prompt: str, system: str = "", max_tokens: int = 1000, model: str = LLM_MODEL
) -> str:
    """Call the local llama-server via OpenAI-compatible API."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{LLM_BASE_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        # UD/thinking models: content holds the answer, reasoning_content holds the thinking trace
        content = (msg.get("content") or "").strip()
        if not content:
            # Don't use full reasoning_content (it's a thinking trace, not structured output)
            # Instead, scan reasoning_content for any embedded JSON array/object
            reasoning = (msg.get("reasoning_content") or "").strip()
            # Look for JSON array or object embedded in the reasoning
            json_match = re.search(r'(\[\s*["\{]|\{\s*")', reasoning)
            if json_match:
                content = reasoning[json_match.start() :]
            else:
                content = reasoning  # last resort — synthesis can use prose
        return content


def _extract_json(text: str) -> Any:
    """Extract JSON from LLM response that may have markdown fencing."""
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return json.loads(match.group(1))
    text = text.strip()
    if text.startswith("[") or text.startswith("{"):
        return json.loads(text)
    raise ValueError(f"No JSON found in: {text[:200]}")


# ── Stage 1: Decompose ────────────────────────────────────────────────────────


async def _decompose(query: str, model: str) -> list[str]:
    """Break the research question into 6-10 independently searchable sub-questions."""
    prompt = (
        "You are a research planner. Decompose this research question into 6-10 specific "
        "sub-questions that together would give exhaustive coverage.\n\n"
        "Rules:\n"
        "- Each sub-question should be independently searchable\n"
        "- Cover different angles: financial, regulatory, geographic, risk, trends\n"
        "- Be specific enough that a web search would find direct answers\n"
        "- Return ONLY a JSON array of strings\n\n"
        f"Question: {query}"
    )
    try:
        raw = await _llm_call(prompt, max_tokens=4000, model=model)
        sub_questions = _extract_json(raw)
        if not isinstance(sub_questions, list) or not sub_questions:
            raise ValueError("Empty or non-list response")
        return [str(q) for q in sub_questions]
    except Exception as exc:
        log.warning("Decompose failed (%s) — falling back to original query", exc)
        return [query]


# ── Stage 2: Search ───────────────────────────────────────────────────────────


async def _brave_search(query: str, count: int) -> list[dict[str, Any]]:
    """Run a Brave web search. Returns list of {title, url, description}."""
    if not BRAVE_API_KEY:
        log.warning("Brave search not configured (BRAVE_API_KEY unset) — returning empty results")
        return []

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            BRAVE_SEARCH_URL,
            headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
            params={"q": query, "count": count, "text_decorations": False},
        )
        resp.raise_for_status()
        data = resp.json()

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
        }
        for item in data.get("web", {}).get("results", [])
    ]


async def _search_all(
    sub_questions: list[str], num_sources: int
) -> dict[str, list[dict[str, Any]]]:
    """
    Search Brave for each sub-question in parallel.
    Returns {sub_question: [result, ...]} with deduped URLs across sub-questions.
    """

    async def _search_one(sq: str) -> tuple[str, list[dict[str, Any]]]:
        try:
            results = await _brave_search(sq, num_sources)
            return sq, results
        except Exception as exc:
            log.debug("Brave search failed for %r: %s", sq, exc)
            return sq, []

    pairs = await asyncio.gather(*[_search_one(sq) for sq in sub_questions])

    # Deduplicate URLs globally — first sub-question to claim a URL owns it
    seen_urls: set[str] = set()
    results_by_sq: dict[str, list[dict[str, Any]]] = {}
    for sq, hits in pairs:
        unique = []
        for hit in hits:
            if hit["url"] not in seen_urls:
                seen_urls.add(hit["url"])
                unique.append(hit)
        results_by_sq[sq] = unique

    return results_by_sq


# ── Stage 3: Fetch ────────────────────────────────────────────────────────────


async def _fetch_url(url: str, sem: asyncio.Semaphore) -> tuple[bool, str, str | None]:
    """
    Fetch a URL and return (success, readable_text, error).

    If BROWSER_FETCH_URL is configured, delegates to the browser fetch proxy
    (scripts/browser_fetch_proxy.py) running on the Mac for JS-rendered pages,
    soft paywalls, and cookie-gated content. Falls back to raw httpx on proxy failure.

    Without the proxy: strips scripts/styles/nav from raw HTML.
    """
    async with sem:
        # ── Browser proxy path (preferred) ────────────────────────────────────
        if BROWSER_FETCH_URL:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        BROWSER_FETCH_URL,
                        json={"url": url, "max_words": 4000, "timeout_ms": 20000},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get("ok") and data.get("text"):
                        return True, data["text"], None
                    # Proxy returned ok=False — fall through to raw fetch
                    log.debug(
                        "Browser proxy returned ok=False for %s: %s", url, data.get("error")
                    )
            except Exception as exc:
                log.debug(
                    "Browser proxy failed for %s (%s), falling back to raw fetch", url, exc
                )

        # ── Raw httpx fallback ────────────────────────────────────────────────
        try:
            async with httpx.AsyncClient(
                timeout=FETCH_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": settings.user_agent},
            ) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return False, "", f"HTTP {resp.status_code}"

                raw = resp.text
                clean = re.sub(
                    r"(?is)<(script|style|nav|header|footer|noscript)[^>]*>.*?</\1>", " ", raw
                )
                clean = re.sub(r"<[^>]+>", " ", clean)
                clean = (
                    clean.replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&nbsp;", " ")
                    .replace("&#039;", "'")
                    .replace("&quot;", '"')
                )
                clean = re.sub(r"\s+", " ", clean).strip()
                words = clean.split()
                text_words = [w for w in words if not re.search(r"[{}%;]", w) and len(w) < 50]
                return True, " ".join(text_words[:4000]), None

        except Exception as exc:
            log.debug("Fetch failed for %s: %s", url, exc)
            return False, "", str(exc)


# ── Stage 4: Distill ──────────────────────────────────────────────────────────


async def _distill(
    sub_question: str,
    title: str,
    url: str,
    text: str,
    sem: asyncio.Semaphore,
    model: str,
) -> list[str]:
    """
    Extract concrete, specific facts from a fetched page relevant to the sub-question.
    Returns a list of fact strings (may be empty).
    """
    async with sem:
        prompt = (
            f"Extract all concrete, specific facts from this document relevant to: {sub_question}\n\n"
            "Rules:\n"
            "- Extract facts WITH their specifics: numbers, dates, names, percentages, dollar amounts\n"
            '- NOT: "the article discusses insurance costs"\n'
            '- YES: "Florida coastal insurance: ~$7,000/yr, double inland average"\n'
            '- NOT: "various locations are mentioned"\n'
            '- YES: "Destin FL: 38% occupancy, $251 ADR, RevPAR $169"\n'
            "- If the document contains nothing relevant, return an empty array\n"
            "- Return ONLY a JSON array of fact strings, max 20 facts\n\n"
            f"Document title: {title}\n"
            f"Document URL: {url}\n\n"
            f"Document:\n{text[:2500]}"
        )
        try:
            raw = await _llm_call(prompt, max_tokens=6000, model=model)
            log.warning("DISTILL raw for %s: %r", url[:50], raw[:200])
            facts = _extract_json(raw)
            if isinstance(facts, list):
                return [str(f) for f in facts if f]
            return []
        except Exception as exc:
            log.warning("Distill failed for %s: %s", url, exc)
            return []


# ── Stage 5: Aggregate ────────────────────────────────────────────────────────


def _aggregate(
    results_by_sq: dict[str, list[dict[str, Any]]],
    facts_by_url: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Group all distilled facts by sub-question.
    No LLM needed — pure dict grouping.
    """
    aggregated: dict[str, list[str]] = {}
    for sq, hits in results_by_sq.items():
        all_facts: list[str] = []
        for hit in hits:
            url = hit["url"]
            facts = facts_by_url.get(url, [])
            # Prefix each fact with its source URL for inline citation in synthesis
            for fact in facts:
                all_facts.append(f"{fact} [source: {url}]")
        aggregated[sq] = all_facts
    return aggregated


# ── Stage 6: Synthesize ───────────────────────────────────────────────────────


async def _synthesize(query: str, aggregated: dict[str, list[str]], model: str) -> str:
    """
    Write a comprehensive research report from all aggregated facts.
    Falls back to a plain-text dump of facts if the LLM call fails.
    """
    # Format findings section
    sections: list[str] = []
    for sq, facts in aggregated.items():
        if facts:
            bullet_facts = "\n".join(f"  - {f}" for f in facts)
            sections.append(f"Sub-question: {sq}\nFindings:\n{bullet_facts}")
        else:
            sections.append(f"Sub-question: {sq}\nFindings: (no data found)")

    aggregated_text = "\n\n".join(sections)

    prompt = (
        "You are a research analyst. Write a comprehensive report answering this question "
        "based on the research findings below.\n\n"
        "Rules:\n"
        "- Lead with a TL;DR (3-5 bullet points, most important findings)\n"
        "- Then full analysis organized by theme\n"
        "- Cite facts inline with [source: url] notation\n"
        "- Include specific numbers wherever available\n"
        "- Flag any areas where data was sparse or missing\n"
        "- This is a full report — be thorough, not brief\n\n"
        f"Question: {query}\n\n"
        f"Research findings by sub-question:\n{aggregated_text}"
    )
    try:
        return await _llm_call(prompt, max_tokens=8000, model=model)
    except Exception as exc:
        log.warning("Synthesize failed (%s) — returning raw facts", exc)
        return f"Synthesis unavailable ({exc}).\n\nRaw findings:\n\n{aggregated_text}"


# ── Stage 7: Compress (optional) ─────────────────────────────────────────────


async def _compress(report: str, token_budget: int, model: str) -> str:
    """Reduce the full report to approximately token_budget tokens."""
    prompt = (
        f"Summarize this report to approximately {token_budget} tokens while preserving:\n"
        "- All key recommendations\n"
        "- All specific numbers and data points\n"
        "- Source attributions\n\n"
        f"Report:\n{report}"
    )
    try:
        return await _llm_call(prompt, max_tokens=token_budget, model=model)
    except Exception as exc:
        log.warning("Compress failed (%s) — returning full report", exc)
        return report


# ── Dewey search: corpus-first, web fallback ─────────────────────────────────

DEWIE_SCORE_THRESHOLD = 0.5  # minimum score to trust a corpus hit


def _dewie_auth_headers() -> dict[str, str]:
    api_key = os.environ.get("DEWIE_API_KEY", "")
    return {"X-API-Key": api_key} if api_key else {}


async def _dewie_search(query: str, limit: int = 5) -> list[dict]:
    """Search Dewie corpus. Returns list of result dicts or [] on any error."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{DEWIE_BASE}/search",
                json={"query": query, "limit": limit, "ranker": "rrf_aq"},
                headers=_dewie_auth_headers(),
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
    except Exception:
        return []


async def _ingest_background(url: str, title: str) -> None:
    """Fire-and-forget ingest into Dewie corpus. Never raises."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{DEWIE_BASE}/ingest",
                json={"url": url, "title": title},
                headers=_dewie_auth_headers(),
            )
    except Exception:
        pass


async def _dewey_search(
    query: str,
    fetch_sem: asyncio.Semaphore,
    num_web_results: int = 3,
    ingest: bool = True,
) -> tuple[str, str | None]:
    """
    Corpus-first search with web fallback + auto-ingest.

    Returns (text, source_url):
      - Corpus hit:  (snippet_text, None)       — no ingest needed, already there
      - Web hit:     (fetched_page_text, url)    — ingested in background
      - Total miss:  ("", None)
    """
    hits = await _dewie_search(query, limit=5)
    best = next((h for h in hits if h.get("score", 0) >= DEWIE_SCORE_THRESHOLD), None)
    if best:
        text = best.get("snippet") or best.get("body", "")[:1000]
        if text:
            return text, None

    try:
        web_results = await _brave_search(query, num_web_results)
    except Exception:
        web_results = []

    for item in web_results:
        url = item.get("url", "")
        if not url:
            continue
        ok, page_text, _err = await _fetch_url(url, fetch_sem)
        if ok and page_text:
            if ingest:
                asyncio.create_task(_ingest_background(url, item.get("title", url)))
            return page_text, url

    return "", None


async def _dewey_read(
    url: str,
    fetch_sem: asyncio.Semaphore,
    ingest: bool = True,
) -> tuple[str, str]:
    """
    Read content for a URL: corpus lookup first, then fetch + auto-ingest.

    Returns (text, source):
      - ("...content...", "corpus")  if already in corpus
      - ("...content...", "web")     if fetched from web (and ingested)
      - ("", "miss")                 if both paths failed
    """
    # Search corpus by URL as query — often finds it if the URL was previously ingested
    hits = await _dewie_search(url, limit=3)
    best = next((h for h in hits if h.get("score", 0) >= DEWIE_SCORE_THRESHOLD), None)
    if best:
        text = best.get("snippet") or best.get("body", "")[:4000]
        if text:
            return text, "corpus"

    ok, page_text, _err = await _fetch_url(url, fetch_sem)
    if ok and page_text:
        if ingest:
            asyncio.create_task(_ingest_background(url, url))
        return page_text, "web"

    return "", "miss"


# ── Ingest helper ─────────────────────────────────────────────────────────────


async def _ingest_url(request: Request, url: str, title: str) -> tuple[bool, str | None]:
    """Fire-and-forget ingest via internal HTTP call. Never raises."""
    try:
        api_key = request.headers.get("X-API-Key", "")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{DEWIE_BASE}/ingest",
                json={"url": url, "title": title},
                headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            )
            if resp.status_code in (200, 202):
                data = resp.json()
                doc_ids = data.get("doc_ids") or data.get("accepted") or []
                return True, str(doc_ids[0]) if doc_ids else None
    except Exception as exc:
        log.debug("Ingest skipped for %s: %s", url, exc)
    return False, None


# ── Route ─────────────────────────────────────────────────────────────────────


@router.post("/investigate", response_model=InvestigateResponse)
@limiter.limit(rate_limit(3))
async def investigate(
    request: Request,
    body: InvestigateRequest,
) -> InvestigateResponse:
    """
    Map-reduce investigation pipeline.

    Decomposes the query into sub-questions, searches the web for each,
    fetches and distills concrete facts from every source, then synthesizes
    a full research report. Sources are optionally ingested into the corpus.
    """
    import time as _time

    request_id = getattr(request.state, "request_id", "unknown")
    redacted_body = _redact_fields(body.model_dump())
    log.info(
        "investigate started",
        extra={
            "request_id": request_id,
            "query": _truncate(body.query),
            "num_sources": body.num_sources,
            "ingest": body.ingest,
            "model": body.model,
            "body": _truncate(json.dumps(redacted_body, default=str)),
        },
    )
    _start = _time.time()
    try:
        trace: list[str] = []
        model = body.model

        # ── Stage 1: Decompose ────────────────────────────────────────────────────
        sub_questions = await _decompose(body.query, model)
        trace.append(f"Decomposed into {len(sub_questions)} sub-questions")

        # ── Stage 2: Search ───────────────────────────────────────────────────────
        results_by_sq = await _search_all(sub_questions, body.num_sources)
        total_hits = sum(len(v) for v in results_by_sq.values())
        trace.append(f"Search returned {total_hits} unique URLs across all sub-questions")

        # Build flat list of (sub_question, hit) pairs for fetching
        fetch_targets: list[tuple[str, dict[str, Any]]] = [
            (sq, hit) for sq, hits in results_by_sq.items() for hit in hits
        ]

        # ── Stage 3: Fetch ────────────────────────────────────────────────────────
        fetch_sem = asyncio.Semaphore(FETCH_SEMAPHORE)
        fetch_tasks = [_fetch_url(hit["url"], fetch_sem) for _, hit in fetch_targets]
        fetch_results = await asyncio.gather(*fetch_tasks)

        fetched_ok = sum(1 for ok, _, _ in fetch_results if ok)
        trace.append(f"Fetched {fetched_ok}/{len(fetch_targets)} URLs successfully")

        # Map url → (success, text, error) for easy lookup
        url_fetch: dict[str, tuple[bool, str, str | None]] = {
            hit["url"]: fetch_results[i] for i, (_, hit) in enumerate(fetch_targets)
        }

        # ── Stage 4: Distill ──────────────────────────────────────────────────────
        distill_sem = asyncio.Semaphore(DISTILL_SEMAPHORE)
        distill_tasks: list[tuple[str, str, asyncio.Task[list[str]]]] = []

        for sq, hits in results_by_sq.items():
            for hit in hits:
                url = hit["url"]
                ok, text, _ = url_fetch[url]
                if ok and text:
                    task = asyncio.create_task(
                        _distill(sq, hit["title"], url, text, distill_sem, model)
                    )
                    distill_tasks.append((sq, url, task))

        facts_by_url: dict[str, list[str]] = {}
        if distill_tasks:
            distilled = await asyncio.gather(*[t for _, _, t in distill_tasks])
            for (sq, url, _), facts in zip(distill_tasks, distilled):
                facts_by_url[url] = facts

        total_facts = sum(len(f) for f in facts_by_url.values())
        trace.append(f"Distilled {total_facts} facts from {len(facts_by_url)} pages")

        # ── Stage 5: Aggregate ───────────────────────────────────────────────────
        aggregated = _aggregate(results_by_sq, facts_by_url)

        # ── Stage 6: Synthesize ───────────────────────────────────────────────────
        report = await _synthesize(body.query, aggregated, model)
        trace.append("Synthesis complete")

        # ── Stage 7: Compress (optional) ──────────────────────────────────────────
        summary: str | None = None
        if body.token_budget is not None:
            summary = await _compress(report, body.token_budget, model)
            trace.append(f"Compressed to ~{body.token_budget} token budget")

        # ── Build source results + optional ingest ────────────────────────────────
        source_results: list[SourceResult] = []
        for sq, hits in results_by_sq.items():
            for hit in hits:
                url = hit["url"]
                ok, _, err = url_fetch[url]
                facts = facts_by_url.get(url, [])

                sr = SourceResult(
                    title=hit["title"],
                    url=url,
                    sub_question=sq,
                    fetched=ok,
                    fetch_error=err,
                    fact_count=len(facts),
                )

                if body.ingest and ok:
                    ingested, doc_id = await _ingest_url(request, url, hit["title"])
                    sr.ingested = ingested
                    sr.doc_id = doc_id

                source_results.append(sr)

        ingested_count = sum(1 for s in source_results if s.ingested)
        if body.ingest:
            trace.append(f"Ingested {ingested_count}/{len(source_results)} sources into corpus")

        elapsed = _time.time() - _start
        log.info(
            "investigate succeeded",
            extra={
                "request_id": request_id,
                "status": 200,
                "total_facts": total_facts,
                "sources": len(source_results),
                "elapsed_seconds": round(elapsed, 3),
            },
        )
        return InvestigateResponse(
            report=report,
            summary=summary,
            sub_questions=sub_questions,
            sources=source_results,
            total_facts=total_facts,
            query=body.query,
            trace=trace,
        )
    except Exception:
        elapsed = _time.time() - _start
        log.exception(
            "investigate failed",
            extra={
                "request_id": request_id,
                "query": _truncate(body.query),
                "elapsed_seconds": round(elapsed, 3),
            },
        )
        raise
