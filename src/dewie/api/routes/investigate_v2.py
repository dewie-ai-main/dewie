# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Async investigation jobs — v2.

Adds two endpoints on top of the synchronous /investigate route:

  POST /investigate/jobs
    Enqueues an investigation job (subquestion or matrix strategy).
    Returns immediately with {id, status: "pending", query}.

  GET /investigate/jobs/{job_id}
    Returns the current job state including result when done.

Matrix strategy
---------------
Instead of decomposing into sub-questions, the matrix strategy:
  1. Discovers 30-60 entities relevant to the query (locations, products, etc.)
  2. Defines 6-10 measurable attributes for those entities
  3. Runs search → fetch → distill for every (entity, attribute) cell in parallel
  4. Synthesises a ranked report from the filled matrix

All heavy helpers (_llm_call, _extract_json, _brave_search, _fetch_url,
_distill, _synthesize, _compress) are imported from investigate.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import text

from dewie.api.middleware import limiter, rate_limit
from dewie.api.routes.investigate import (
    DEWIE_BASE,
    LLM_MODEL,
    _brave_search,
    _compress,
    _dewey_search,
    _distill,
    _extract_json,
    _fetch_url,
    _llm_call,
    _synthesize,
)

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


# ── Request / Response models ─────────────────────────────────────────────────


class InvestigateJobRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    strategy: Literal["subquestion", "matrix", "plan"] = "matrix"
    context: str | None = None
    num_sources: int = Field(default=5, ge=1, le=50)
    ingest: bool = True
    token_budget: int | None = None
    model: str = LLM_MODEL  # distill + decompose model (local, cheap)
    synthesis_model: str = "gpt-4o"  # synthesis model (larger, one call)
    webhook_url: str | None = None


class InvestigateJobStatus(BaseModel):
    id: str
    query: str
    strategy: str
    status: str
    plan: dict | None = None  # type: ignore[type-arg]
    result: dict | None = None  # type: ignore[type-arg]
    error: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


# ── DB helpers ────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(UTC)


async def _update_job(pg: Any, job_id: str, **kwargs: Any) -> None:
    """SET arbitrary columns on investigate_jobs WHERE id = job_id."""
    if not kwargs:
        return
    set_clauses = []
    params: dict[str, Any] = {"job_id": job_id}
    for k, v in kwargs.items():
        if isinstance(v, (dict, list)):
            # JSONB: use CAST to avoid :: syntax conflict with named params
            set_clauses.append(f"{k} = CAST(:{k} AS jsonb)")
            params[k] = json.dumps(v)
        else:
            set_clauses.append(f"{k} = :{k}")
            params[k] = v
    async with pg._engine.begin() as conn:
        await conn.execute(
            text(
                f"UPDATE investigate_jobs SET {', '.join(set_clauses)} WHERE id = CAST(:job_id AS uuid)"
            ),
            params,
        )


async def _fetch_job(pg: Any, job_id: str) -> dict[str, Any] | None:
    async with pg._engine.connect() as conn:
        row = await conn.execute(
            text("""
                SELECT id, query, strategy, status, plan, result, error,
                       created_at, started_at, completed_at
                FROM investigate_jobs
                WHERE id = :job_id
            """),
            {"job_id": job_id},
        )
        r = row.fetchone()
    if r is None:
        return None
    keys = (
        "id",
        "query",
        "strategy",
        "status",
        "plan",
        "result",
        "error",
        "created_at",
        "started_at",
        "completed_at",
    )
    d: dict[str, Any] = dict(zip(keys, r))
    # Convert timestamps to ISO strings
    for ts_col in ("created_at", "started_at", "completed_at"):
        val = d[ts_col]
        if val is not None and not isinstance(val, str):
            d[ts_col] = val.isoformat()
    return d


# ── Matrix strategy ───────────────────────────────────────────────────────────


async def _run_matrix_job(job_id: str, req: InvestigateJobRequest, pg: Any) -> None:
    """Full matrix investigation pipeline, runs as a background asyncio task."""
    try:
        await _update_job(pg, job_id, status="running", started_at=_now())

        # ── Step 1: Entity discovery ──────────────────────────────────────────
        context_line = f"\nUser constraints: {req.context}" if req.context else ""
        entity_prompt = (
            "/no_think\n\n"
            f'For the research question: "{req.query}"\n'
            f"{context_line}\n\n"
            "Identify what the primary entities are that need to be evaluated.\n"
            "For 'best vacation home markets' the entities are specific geographic locations.\n"
            "For 'best cloud databases' the entities are specific database products.\n\n"
            "Generate a comprehensive list of 30-60 specific entities to evaluate.\n"
            "For geographic queries: include entities from across the full geographic range,\n"
            "not just obvious famous ones. Include smaller/emerging options.\n\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            '    "entity_type": "location|product|company|...",\n'
            '    "entities": ["entity1", "entity2", ...],\n'
            '    "search_queries": ["query to find data about entity1", ...]'
            "  // one per entity\n"
            "}"
        )

        entity_raw = await _llm_call(entity_prompt, max_tokens=6000, model=req.model)
        log.warning("ENTITY raw (first 200): %r", entity_raw[:200])
        try:
            entity_data = _extract_json(entity_raw)
        except Exception:
            entity_data = {}

        entities: list[str] = entity_data.get("entities", [])

        if not entities:
            # LLM returned empty — generate entity list directly from query using a simpler prompt
            log.warning("Entity discovery empty, trying simpler prompt for job %s", job_id)
            simple_prompt = (
                f"List 30 specific US locations relevant to: {req.query}\n"
                f"{('Context: ' + req.context) if req.context else ''}\n"
                "Include a mix of beach, mountain, and lake destinations.\n"
                'Return ONLY a JSON array of strings: ["Location 1, ST", "Location 2, ST", ...]'
            )
            simple_raw = await _llm_call(simple_prompt, max_tokens=4000, model=req.model)
            log.warning("SIMPLE raw (first 200): %r", simple_raw[:200])
            try:
                entities = _extract_json(simple_raw)
                if not isinstance(entities, list):
                    entities = []
            except Exception:
                entities = []

        if not entities:
            log.warning(
                "Entity discovery still empty, using hardcoded defaults for job %s", job_id
            )
            entities = [
                "Outer Banks, NC",
                "Destin, FL",
                "Myrtle Beach, SC",
                "Hilton Head, SC",
                "Cape Cod, MA",
                "Ocean City, MD",
                "Galveston, TX",
                "30A/Rosemary Beach, FL",
                "Gatlinburg, TN",
                "Pigeon Forge, TN",
                "Asheville, NC",
                "Boone, NC",
                "Lake Tahoe, CA",
                "Big Bear Lake, CA",
                "Sedona, AZ",
                "Park City, UT",
                "Lake Norman, NC",
                "Lake of the Ozarks, MO",
                "Table Rock Lake, MO",
                "Traverse City, MI",
                "Door County, WI",
                "Lake Winnipesaukee, NH",
                "Fredericksburg, TX",
                "Hill Country, TX",
                "Branson, MO",
                "Chattanooga, TN",
                "Breckenridge, CO",
                "Steamboat Springs, CO",
                "Palm Springs, CA",
                "Scottsdale, AZ",
            ]
            entity_data = {"entity_type": "vacation home market", "entities": entities}

        # ── Step 2: Define attribute columns ─────────────────────────────────
        sample = entities[:10]
        overflow = max(0, len(entities) - 10)
        attr_prompt = (
            "/no_think\n\n"
            f"For evaluating: {sample}"
            + (f" (and {overflow} more similar entities)" if overflow else "")
            + f'\nResearch question: "{req.query}"\n'
            f"{context_line}\n\n"
            "Define 6-10 measurable attributes that would allow ranking/filtering these entities.\n"
            "Each attribute should be searchable on the web.\n\n"
            "Return ONLY a JSON array of attribute objects:\n"
            "[\n"
            '    {"name": "short_name", "description": "what to measure",'
            ' "search_template": "{entity} [specific search terms]"},\n'
            "    ...\n"
            "]"
        )

        attr_raw = await _llm_call(attr_prompt, max_tokens=6000, model=req.model)
        log.warning("ATTR raw (first 200): %r", attr_raw[:200])
        try:
            attributes: list[dict[str, str]] = _extract_json(attr_raw)
            if not isinstance(attributes, list) or not attributes:
                log.warning("Attribute discovery returned empty, using defaults")
                attributes = [
                    {
                        "name": "median_home_price",
                        "description": "median sale price for vacation properties",
                        "search_template": "{entity} vacation home median price 2026",
                    },
                    {
                        "name": "str_occupancy",
                        "description": "short-term rental occupancy rate",
                        "search_template": "{entity} Airbnb occupancy rate 2026",
                    },
                    {
                        "name": "str_regulations",
                        "description": "short-term rental permit requirements",
                        "search_template": "{entity} short term rental regulations permit 2026",
                    },
                    {
                        "name": "cap_rate",
                        "description": "capitalization rate for STR properties",
                        "search_template": "{entity} vacation rental cap rate ROI 2026",
                    },
                    {
                        "name": "insurance_cost",
                        "description": "homeowners insurance cost and climate risk",
                        "search_template": "{entity} homeowners insurance cost climate risk 2026",
                    },
                    {
                        "name": "drive_access",
                        "description": "drive time and accessibility from major metros",
                        "search_template": "{entity} drive time from major city hours",
                    },
                ]
        except Exception as exc:
            log.warning("Attribute JSON parse failed: %s — using defaults", exc)
            attributes = [
                {
                    "name": "median_home_price",
                    "description": "median sale price",
                    "search_template": "{entity} vacation home median price 2026",
                },
                {
                    "name": "str_occupancy",
                    "description": "STR occupancy rate",
                    "search_template": "{entity} Airbnb occupancy rate 2026",
                },
                {
                    "name": "str_regulations",
                    "description": "STR permit requirements",
                    "search_template": "{entity} short term rental permit 2026",
                },
                {
                    "name": "cap_rate",
                    "description": "cap rate ROI",
                    "search_template": "{entity} vacation rental cap rate 2026",
                },
                {
                    "name": "insurance_cost",
                    "description": "insurance and climate risk",
                    "search_template": "{entity} homeowners insurance climate risk",
                },
            ]

        plan: dict[str, Any] = {
            "entity_type": entity_data.get("entity_type"),
            "entities": entities,
            "attributes": attributes,
        }
        await _update_job(pg, job_id, plan=plan)

        # ── Step 3: Matrix fill ───────────────────────────────────────────────
        matrix_data: dict[str, dict[str, list[str]]] = {}
        all_sources: list[dict[str, Any]] = []

        capped_entities = entities[:50]
        batch_size = 5
        fetch_sem = asyncio.Semaphore(5)
        distill_sem = asyncio.Semaphore(1)

        for i in range(0, len(capped_entities), batch_size):
            batch = capped_entities[i : i + batch_size]

            search_tasks: list[tuple[str, str, str]] = []
            for entity in batch:
                for attr in attributes:
                    tmpl = attr.get("search_template", f"{{entity}} {attr.get('name', '')}")
                    query_str = tmpl.replace("{entity}", entity)
                    search_tasks.append((entity, attr["name"], query_str))

            search_results: list[Any] = list(
                await asyncio.gather(
                    *[_brave_search(q, 2) for _, _, q in search_tasks],
                    return_exceptions=True,
                )
            )

            # Collect unique URLs → (entity, attr_name, title, snippet)
            urls_to_fetch: dict[str, tuple[str, str, str, str]] = {}
            for (entity, attr_name, _), res in zip(search_tasks, search_results):
                if isinstance(res, Exception):
                    continue
                for r in res:
                    url = r.get("url", "")
                    if url and url not in urls_to_fetch:
                        urls_to_fetch[url] = (
                            entity,
                            attr_name,
                            r.get("title", ""),
                            r.get("snippet", ""),
                        )

            fetch_results: list[Any] = list(
                await asyncio.gather(
                    *[_fetch_url(url, fetch_sem) for url in urls_to_fetch],
                    return_exceptions=True,
                )
            )

            for (url, (entity, attr_name, title, snippet)), fetch_res in zip(
                urls_to_fetch.items(), fetch_results
            ):
                if isinstance(fetch_res, Exception):
                    continue
                ok, page_text, _err = fetch_res
                if not ok or not page_text:
                    continue

                facts = await _distill(
                    sub_question=f"{attr_name} for {entity}",
                    title=title,
                    url=url,
                    text=page_text,
                    sem=distill_sem,
                    model=req.model,
                )
                if facts:
                    matrix_data.setdefault(entity, {}).setdefault(attr_name, []).extend(facts)
                    all_sources.append(
                        {
                            "url": url,
                            "title": title,
                            "entity": entity,
                            "attr": attr_name,
                            "fact_count": len(facts),
                        }
                    )

        # ── Step 4: Synthesise from matrix ────────────────────────────────────
        matrix_lines: list[str] = []
        for entity, attrs in matrix_data.items():
            if not attrs:
                continue
            lines = [f"\n### {entity}"]
            for attr_name, facts in attrs.items():
                if facts:
                    lines.append(f"**{attr_name}:** " + "; ".join(facts[:5]))
            matrix_lines.append("\n".join(lines))

        attr_names = [a["name"] for a in attributes]
        synth_prompt = (
            f'Research question: "{req.query}"\n'
            f"{context_line}\n\n"
            f"Below is a research matrix with data collected for {len(matrix_data)} entities "
            f"across {len(attributes)} attributes.\n"
            "Many cells are empty (data not found) — that is expected.\n\n"
            "Write a comprehensive research report that:\n"
            "1. Leads with TL;DR: top 5-8 entities that stand out based on available data\n"
            "2. Explains why each TL;DR entry was selected (cite specific data points)\n"
            "3. Notes significant gaps or risks revealed by the data\n"
            "4. Includes a section on entities worth investigating further (promising but data-sparse)\n"
            "5. Cites sources as [source: url] where data exists\n\n"
            "Research Matrix:\n" + "\n".join(matrix_lines[:100])
        )

        try:
            report = await _llm_call(synth_prompt, max_tokens=8000, model=req.synthesis_model)
        except Exception as exc:
            log.warning("Synthesis LLM call failed (%s), using raw matrix as report", exc)
            report = "## Research Matrix (synthesis failed)\n\n" + "\n".join(matrix_lines[:200])

        summary: str | None = None
        if req.token_budget is not None:
            summary = await _compress(report, req.token_budget, req.model)

        total_facts = sum(len(facts) for attrs in matrix_data.values() for facts in attrs.values())

        result: dict[str, Any] = {
            "report": report,
            "summary": summary,
            "entities_evaluated": len(matrix_data),
            "attributes": attr_names,
            "sources": all_sources,
            "total_facts": total_facts,
        }

        # ── Ingest sources ────────────────────────────────────────────────────
        if req.ingest:
            api_key = os.environ.get("DEWIE_API_KEY", "")
            for src in all_sources:
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        await client.post(
                            f"{DEWIE_BASE}/ingest",
                            json={"url": src["url"], "title": src["title"]},
                            headers={"X-API-Key": api_key},
                        )
                except Exception:
                    pass

        # ── Webhook delivery ──────────────────────────────────────────────────
        if req.webhook_url:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(req.webhook_url, json={"job_id": job_id, "result": result})
            except Exception:
                pass

        await _update_job(pg, job_id, status="done", completed_at=_now(), result=result)

    except Exception as exc:
        log.exception("Matrix job %s failed: %s", job_id, exc)
        try:
            await _update_job(pg, job_id, status="failed", error=str(exc), completed_at=_now())
        except Exception:
            pass


# ── Subquestion strategy (async wrapper) ─────────────────────────────────────


async def _run_subquestion_job(
    job_id: str,
    req: InvestigateJobRequest,
    pg: Any,
    *,
    _already_running: bool = False,
) -> None:
    """Run the existing subquestion pipeline as an async job."""
    try:
        if not _already_running:
            await _update_job(pg, job_id, status="running", started_at=_now())

        from dewie.api.routes.investigate import (
            DISTILL_SEMAPHORE,
            FETCH_SEMAPHORE,
            _aggregate,
            _decompose,
            _search_all,
        )

        sub_questions = await _decompose(req.query, req.model)
        plan = {"sub_questions": sub_questions}
        await _update_job(pg, job_id, plan=plan)

        results_by_sq = await _search_all(sub_questions, req.num_sources)

        fetch_targets = [(sq, hit) for sq, hits in results_by_sq.items() for hit in hits]

        fetch_sem = asyncio.Semaphore(FETCH_SEMAPHORE)
        fetch_results: list[Any] = list(
            await asyncio.gather(
                *[_fetch_url(hit["url"], fetch_sem) for _, hit in fetch_targets],
                return_exceptions=True,
            )
        )

        url_fetch: dict[str, tuple[bool, str, str | None]] = {}
        for i, (_, hit) in enumerate(fetch_targets):
            fr = fetch_results[i]
            if isinstance(fr, Exception):
                url_fetch[hit["url"]] = (False, "", str(fr))
            else:
                url_fetch[hit["url"]] = fr

        distill_sem = asyncio.Semaphore(DISTILL_SEMAPHORE)
        distill_tasks: list[tuple[str, str, asyncio.Task[list[str]]]] = []
        for sq, hits in results_by_sq.items():
            for hit in hits:
                url = hit["url"]
                ok, page_text, _ = url_fetch.get(url, (False, "", None))
                if ok and page_text:
                    t = asyncio.create_task(
                        _distill(sq, hit["title"], url, page_text, distill_sem, req.model)
                    )
                    distill_tasks.append((sq, url, t))

        facts_by_url: dict[str, list[str]] = {}
        if distill_tasks:
            distilled = await asyncio.gather(*[t for _, _, t in distill_tasks])
            for (_, url, _task), facts in zip(distill_tasks, distilled):
                facts_by_url[url] = facts

        aggregated = _aggregate(results_by_sq, facts_by_url)
        report = await _synthesize(req.query, aggregated, req.synthesis_model)

        summary: str | None = None
        if req.token_budget is not None:
            summary = await _compress(report, req.token_budget, req.model)

        all_sources = [
            {
                "url": hit["url"],
                "title": hit["title"],
                "sub_question": sq,
                "fact_count": len(facts_by_url.get(hit["url"], [])),
            }
            for sq, hits in results_by_sq.items()
            for hit in hits
        ]

        total_facts = sum(len(f) for f in facts_by_url.values())

        result: dict[str, Any] = {
            "report": report,
            "summary": summary,
            "sub_questions": sub_questions,
            "sources": all_sources,
            "total_facts": total_facts,
        }

        if req.ingest:
            api_key = os.environ.get("DEWIE_API_KEY", "")
            for src in all_sources:
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        await client.post(
                            f"{DEWIE_BASE}/ingest",
                            json={"url": src["url"], "title": src["title"]},
                            headers={"X-API-Key": api_key},
                        )
                except Exception:
                    pass

        if req.webhook_url:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(req.webhook_url, json={"job_id": job_id, "result": result})
            except Exception:
                pass

        await _update_job(pg, job_id, status="done", completed_at=_now(), result=result)

    except Exception as exc:
        log.exception("Subquestion job %s failed: %s", job_id, exc)
        try:
            await _update_job(pg, job_id, status="failed", error=str(exc), completed_at=_now())
        except Exception:
            pass


# ── Plan strategy ─────────────────────────────────────────────────────────────


def _auth_headers() -> dict[str, str]:
    """Return auth headers using DEWIE_API_KEY if set."""
    from dewie.api.routes.investigate import _dewie_auth_headers

    return _dewie_auth_headers()


async def _run_plan_job(job_id: str, req: InvestigateJobRequest, pg: Any) -> None:
    """Full plan-based investigation pipeline."""
    try:
        await _update_job(pg, job_id, status="running", started_at=_now())

        context_line = f"\nUser constraints: {req.context}" if req.context else ""

        # ── Step 0: Plan generation ───────────────────────────────────────────
        plan_prompt = (
            "/no_think\n\n"
            f'Generate a structured research plan for the query: "{req.query}"\n'
            f"{context_line}\n\n"
            "Return ONLY a JSON object with this exact shape:\n"
            "{\n"
            '  "angles": ["search query 1", "search query 2", ...],\n'
            '  "entity_type": "what kind of entities we are finding (e.g. LLM models)",\n'
            '  "attributes": ["attribute 1", "attribute 2", ...],\n'
            '  "completeness_criteria": "one sentence describing when we have enough coverage",\n'
            '  "adversarial_questions": ["question 1", "question 2", ...]\n'
            "}\n\n"
            "Rules:\n"
            "- angles: 6-10 search queries covering different facets of entity discovery\n"
            "- attributes: 4-8 per-entity fields to populate for each entity found\n"
            "- adversarial_questions: 2-4 questions a skeptic would ask about gaps in coverage"
        )

        plan_raw = await _llm_call(plan_prompt, max_tokens=2000, model=req.model)
        log.warning("PLAN raw (first 300): %r", plan_raw[:300])
        try:
            research_plan: dict[str, Any] = _extract_json(plan_raw)
        except Exception:
            research_plan = {}

        angles: list[str] = research_plan.get("angles", [])
        entity_type: str = research_plan.get("entity_type", "entity")
        attributes: list[str] = research_plan.get("attributes", [])
        adversarial_questions: list[str] = research_plan.get("adversarial_questions", [])

        if not angles:
            angles = [req.query]
        if not attributes:
            attributes = ["name", "description", "key features", "use cases"]

        await _update_job(
            pg,
            job_id,
            plan={
                "angles": angles,
                "entity_type": entity_type,
                "attributes": attributes,
                "completeness_criteria": research_plan.get("completeness_criteria", ""),
                "adversarial_questions": adversarial_questions,
            },
        )

        # ── Step 1: Entity discovery ───────────────────────────────────────────
        # Run all angles as Brave searches in parallel; orchestrator controls budget.
        search_results: list[Any] = list(
            await asyncio.gather(
                *[_brave_search(angle, req.num_sources) for angle in angles],
                return_exceptions=True,
            )
        )

        # Collect all snippets for entity extraction
        all_snippets: list[str] = []
        for res in search_results:
            if isinstance(res, Exception):
                continue
            for item in res:
                snippet = f"{item.get('title', '')} — {item.get('snippet', '')}"
                all_snippets.append(snippet)

        extract_prompt = (
            "/no_think\n\n"
            f'From these search result snippets about "{req.query}", '
            f"extract all specific {entity_type} names mentioned.\n\n"
            "Rules:\n"
            "- Return canonical, properly-spelled names only\n"
            "- Deduplicate (include each entity once)\n"
            "- Return ONLY a JSON array of strings\n\n"
            "Snippets:\n" + "\n".join(all_snippets[:80])
        )

        entities_raw = await _llm_call(extract_prompt, max_tokens=3000, model=req.model)
        try:
            entities: list[str] = _extract_json(entities_raw)
            if not isinstance(entities, list):
                entities = []
            entities = list(
                dict.fromkeys(str(e) for e in entities if e)
            )  # deduplicate, preserve order
        except Exception:
            entities = []

        log.warning("PLAN job %s: discovered %d entities", job_id, len(entities))

        # ── Step 2: Per-entity investigation ──────────────────────────────────
        fetch_sem = asyncio.Semaphore(5)
        distill_sem = asyncio.Semaphore(1)

        async def _investigate_entity(entity: str) -> dict[str, Any]:
            profile: dict[str, Any] = {"entity": entity, "attributes": {}, "source": None}

            text, source_url = await _dewey_search(entity, fetch_sem)
            if not text:
                return profile

            profile["source"] = "corpus" if source_url is None else "web"
            if source_url:
                profile["web_url"] = source_url

            facts = await _distill(
                sub_question=f"{entity}: {', '.join(attributes)}",
                title=entity,
                url=source_url or "",
                text=text,
                sem=distill_sem,
                model=req.model,
            )
            if facts:
                profile["attributes"] = {"facts": facts}

            return profile

        entity_profiles: list[dict[str, Any]] = list(
            await asyncio.gather(
                *[_investigate_entity(e) for e in entities],
                return_exceptions=True,
            )
        )
        # Drop any exceptions
        entity_profiles = [p for p in entity_profiles if isinstance(p, dict)]

        # ── Step 3: Adversarial gap pass ──────────────────────────────────────
        if adversarial_questions and entity_profiles:
            entity_names_so_far = [p["entity"] for p in entity_profiles]
            gap_prompt = (
                "/no_think\n\n"
                f'Research query: "{req.query}"\n'
                f"Current entity list ({len(entity_names_so_far)} found): "
                + ", ".join(entity_names_so_far[:40])
                + "\n\nAdversarial questions:\n"
                + "\n".join(f"- {q}" for q in adversarial_questions)
                + "\n\nWhat entities are likely missing? "
                "Return ONLY a JSON array of additional search queries (2-5 queries) "
                "that would surface gaps, or an empty array [] if coverage looks complete."
            )

            gap_raw = await _llm_call(gap_prompt, max_tokens=1000, model=req.model)
            try:
                gap_queries: list[str] = _extract_json(gap_raw)
                if not isinstance(gap_queries, list):
                    gap_queries = []
            except Exception:
                gap_queries = []

            if gap_queries:
                gap_search_results: list[Any] = list(
                    await asyncio.gather(
                        *[_brave_search(q, req.num_sources) for q in gap_queries],
                        return_exceptions=True,
                    )
                )
                gap_snippets: list[str] = []
                for res in gap_search_results:
                    if isinstance(res, Exception):
                        continue
                    for item in res:
                        gap_snippets.append(f"{item.get('title', '')} — {item.get('snippet', '')}")

                if gap_snippets:
                    gap_extract_prompt = (
                        "/no_think\n\n"
                        f"Extract additional {entity_type} names from these snippets "
                        f"that are NOT already in this list: {entity_names_so_far[:40]}\n\n"
                        "Return ONLY a JSON array of new entity name strings.\n\n"
                        "Snippets:\n" + "\n".join(gap_snippets[:50])
                    )
                    gap_entities_raw = await _llm_call(
                        gap_extract_prompt, max_tokens=1000, model=req.model
                    )
                    try:
                        gap_entities: list[str] = _extract_json(gap_entities_raw)
                        if not isinstance(gap_entities, list):
                            gap_entities = []
                    except Exception:
                        gap_entities = []

                    existing_names = {p["entity"].lower() for p in entity_profiles}
                    new_entities = [e for e in gap_entities if str(e).lower() not in existing_names]

                    if new_entities:
                        log.warning(
                            "PLAN job %s: gap pass adding %d entities", job_id, len(new_entities)
                        )
                        new_profiles: list[Any] = list(
                            await asyncio.gather(
                                *[_investigate_entity(e) for e in new_entities],
                                return_exceptions=True,
                            )
                        )
                        entity_profiles.extend(p for p in new_profiles if isinstance(p, dict))

        # ── Step 4: Synthesis ──────────────────────────────────────────────────
        profile_lines: list[str] = []
        for p in entity_profiles:
            facts = p.get("attributes", {}).get("facts", [])
            if not facts and p.get("corpus_snippet"):
                facts = [p["corpus_snippet"][:200]]
            if facts:
                profile_lines.append(
                    f"\n### {p['entity']}\n" + "\n".join(f"- {f}" for f in facts[:10])
                )

        synth_prompt = (
            f'Research query: "{req.query}"\n'
            f"{context_line}\n\n"
            f"Below are research profiles for {len(entity_profiles)} {entity_type}s.\n\n"
            "Write a comprehensive research report that:\n"
            "1. Leads with TL;DR: top 5-8 standout entities with specific supporting data\n"
            "2. Full analysis organized by theme or attribute\n"
            "3. Notes gaps, risks, or entities that warrant further investigation\n"
            "4. Cites sources as [source: url] where available\n\n"
            "Entity profiles:\n" + "\n".join(profile_lines[:120])
        )

        try:
            report = await _llm_call(synth_prompt, max_tokens=8000, model=req.synthesis_model)
        except Exception as exc:
            log.warning("Plan synthesis failed (%s), using raw profiles", exc)
            report = "## Entity Profiles (synthesis failed)\n\n" + "\n".join(profile_lines[:200])

        summary: str | None = None
        if req.token_budget is not None:
            summary = await _compress(report, req.token_budget, req.model)

        all_sources = [
            {"entity": p["entity"], "source": p.get("source"), "url": p.get("web_url", "")}
            for p in entity_profiles
        ]

        result: dict[str, Any] = {
            "report": report,
            "summary": summary,
            "entity_type": entity_type,
            "entities_evaluated": len(entity_profiles),
            "attributes": attributes,
            "sources": all_sources,
        }

        # ── Ingest ────────────────────────────────────────────────────────────
        if req.ingest:
            for src in all_sources:
                if src.get("url"):
                    try:
                        async with httpx.AsyncClient(timeout=10) as client:
                            await client.post(
                                f"{DEWIE_BASE}/ingest",
                                json={"url": src["url"], "title": src["entity"]},
                                headers=_auth_headers(),
                            )
                    except Exception:
                        pass

        # ── Webhook ───────────────────────────────────────────────────────────
        if req.webhook_url:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(req.webhook_url, json={"job_id": job_id, "result": result})
            except Exception:
                pass

        await _update_job(pg, job_id, status="done", completed_at=_now(), result=result)

    except Exception as exc:
        log.exception("Plan job %s failed: %s", job_id, exc)
        try:
            await _update_job(pg, job_id, status="failed", error=str(exc), completed_at=_now())
        except Exception:
            pass


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("/investigate/jobs")
@limiter.limit(rate_limit(3))
async def create_investigate_job(
    request: Request,
    body: InvestigateJobRequest,
) -> dict[str, Any]:
    """
    Enqueue an investigation job.

    Returns immediately with {id, status: "pending", query}.
    Poll GET /investigate/jobs/{id} for progress and results.
    """
    import time as _time

    request_id = getattr(request.state, "request_id", "unknown")
    redacted_body = _redact_fields(body.model_dump())
    log.info(
        "create_investigate_job started",
        extra={
            "request_id": request_id,
            "query": _truncate(body.query),
            "strategy": body.strategy,
            "num_sources": body.num_sources,
            "body": _truncate(json.dumps(redacted_body, default=str)),
        },
    )
    _start = _time.time()
    try:
        pg = request.app.state.postgres

        async with pg._engine.begin() as conn:
            row = await conn.execute(
                text("""
                    INSERT INTO investigate_jobs (query, strategy, context)
                    VALUES (:query, :strategy, :context)
                    RETURNING id, created_at
                """),
                {"query": body.query, "strategy": body.strategy, "context": body.context},
            )
            r = row.fetchone()

        job_id = str(r[0])
        created_at = r[1].isoformat() if hasattr(r[1], "isoformat") else str(r[1])

        async def _run_with_error_catch(coro):
            try:
                await coro
            except Exception as exc:
                log.error(
                    "Investigation job %s crashed: %s",
                    job_id,
                    exc,
                    exc_info=True,
                    request_id=request_id,
                )
                try:
                    await _update_job(pg, job_id, status="failed", error=str(exc))
                except Exception:
                    pass

        if body.strategy == "matrix":
            asyncio.create_task(_run_with_error_catch(_run_matrix_job(job_id, body, pg)))
        elif body.strategy == "plan":
            asyncio.create_task(_run_with_error_catch(_run_plan_job(job_id, body, pg)))
        else:
            asyncio.create_task(_run_with_error_catch(_run_subquestion_job(job_id, body, pg)))

        elapsed = _time.time() - _start
        log.info(
            "create_investigate_job succeeded",
            extra={
                "request_id": request_id,
                "job_id": job_id,
                "status": 200,
                "elapsed_seconds": round(elapsed, 3),
            },
        )
        return {"id": job_id, "status": "pending", "query": body.query, "created_at": created_at}
    except Exception:
        elapsed = _time.time() - _start
        log.exception(
            "create_investigate_job failed",
            extra={
                "request_id": request_id,
                "query": _truncate(body.query),
                "elapsed_seconds": round(elapsed, 3),
            },
        )
        raise


@router.get("/investigate/jobs/{job_id}", response_model=InvestigateJobStatus)
async def get_investigate_job(job_id: str, request: Request) -> InvestigateJobStatus:
    """Fetch the current state of an investigation job."""
    pg = request.app.state.postgres
    row = await _fetch_job(pg, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return InvestigateJobStatus(
        id=str(row["id"]),
        query=row["query"],
        strategy=row["strategy"],
        status=row["status"],
        plan=row["plan"],
        result=row["result"] if row["status"] == "done" else None,
        error=row["error"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )
