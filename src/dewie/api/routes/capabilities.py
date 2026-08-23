# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
/capabilities/probe — corpus capability map endpoint.

Pre-search endpoint for agents and researchers to understand what a corpus
knows before committing to a specific query. Returns topic clusters, hub
document AQ samples, gap signals, and a suggested first query.

Token budget: target <1,200 tokens per response so context-constrained
agents can probe without sacrificing synthesis budget.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from dewie.api.middleware import limiter, rate_limit

router = APIRouter(prefix="/capabilities", tags=["capabilities"])


# ── Request / Response models ─────────────────────────────────────────────────


class ProbeRequest(BaseModel):
    context: str = Field(
        min_length=1,
        max_length=300,
        description="Topic sketch — free text describing the domain to probe.",
    )
    depth: Literal["summary", "detailed"] = Field(
        default="summary",
        description=(
            "'summary' returns up to 5 clusters with 3 AQ samples each (~800 tokens). "
            "'detailed' returns up to 10 clusters with 5 AQ samples each (~1,800 tokens)."
        ),
    )


class CapabilityCluster(BaseModel):
    label: str = Field(description="Topic cluster label derived from dominant topic.")
    doc_count: int = Field(description="Number of documents in this cluster.")
    hub_doc_id: str | None = Field(description="Most-connected document in the cluster.")
    coverage_confidence: float = Field(description="0.0–1.0; higher = deeper coverage.")
    sample_answerable_questions: list[str] = Field(
        description="Questions the hub document can answer — natural-language capability description."
    )
    time_range: dict[str, str] | None = Field(
        default=None,
        description="Earliest and latest document ingestion dates in this cluster.",
    )


class SystemCapabilities(BaseModel):
    podcast_transcription_available: bool = Field(description="Whether Whisper backends are installed.")


class ProbeResponse(BaseModel):
    """
    Corpus capability map for a given topic context.

    Read complexity_signal first:
    - 'deep': corpus has strong coverage, proceed to dewie_search
    - 'moderate': targeted search will work, check gap_signals for missing angles
    - 'sparse': corpus may not answer this well — consider ingesting more documents
      or broaden your query

    suggested_first_query is the highest-signal query to fire after probing.
    gap_signals name what the corpus cannot answer — use these as an ingest checklist.

    Typical token cost: 'summary' ~800 tokens, 'detailed' ~1,800 tokens.
    """

    context: str
    as_of: date = Field(default_factory=date.today)
    coverage_signal: Literal["deep", "moderate", "sparse"] = Field(
        description="Overall coverage depth for the probed context."
    )
    total_matching_docs: int
    clusters: list[CapabilityCluster]
    gap_signals: list[str] = Field(
        description=(
            "Topics or question types the corpus cannot answer well. "
            "Use as an ingest checklist or to adjust your query strategy."
        )
    )
    suggested_first_query: str | None = Field(
        description="Highest-signal query to fire next. Derived from hub cluster AQ strings."
    )
    probe_hint: str = Field(
        default=(
            "Probe result: use clusters to plan retrieval. "
            "Fire suggested_first_query then check result_confidence.complexity."
        ),
        description="Guidance note for context-constrained agents.",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_gap_signals(clusters: list[dict], context: str) -> list[str]:  # type: ignore[type-arg]
    """
    Infer gap signals from cluster coverage_confidence and doc_count.
    Low-confidence clusters and the absence of expected sub-topics produce signals.
    """
    gaps = []
    for c in clusters:
        if c["coverage_confidence"] < 0.3:
            gaps.append(
                f"Thin coverage on '{c['label']}' ({c['doc_count']} docs) — "
                "consider ingesting more content on this subtopic."
            )
    if not clusters:
        gaps.append(
            f"No documents found matching '{context}'. "
            "The corpus may not cover this domain — ingest relevant content first."
        )
    return gaps[:3]  # cap at 3 to keep token budget tight


def _suggested_first_query(clusters: list[dict]) -> str | None:  # type: ignore[type-arg]
    """Pick the most answerable question from the highest-confidence cluster."""
    if not clusters:
        return None
    best = max(clusters, key=lambda c: c["coverage_confidence"] * c["doc_count"])
    aqs = best.get("sample_aqs") or []
    if isinstance(aqs, list) and aqs:
        return aqs[0]
    return best.get("label")


def _coverage_signal(clusters: list[dict]) -> Literal["deep", "moderate", "sparse"]:  # type: ignore[type-arg]
    if not clusters:
        return "sparse"
    total_docs = sum(c["doc_count"] for c in clusters)
    avg_confidence = sum(c["coverage_confidence"] for c in clusters) / len(clusters)
    if total_docs >= 100 and avg_confidence >= 0.6:
        return "deep"
    if total_docs >= 20 or avg_confidence >= 0.3:
        return "moderate"
    return "sparse"


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.post(
    "/probe",
    response_model=ProbeResponse,
    summary="Probe corpus coverage for a topic context before searching.",
)
@limiter.limit(rate_limit())
async def probe(
    request: Request,
    body: ProbeRequest,
) -> ProbeResponse:
    """
    Pre-search corpus capability map.

    **Call this before dewie_search** when:
    - Context remaining is under 60k tokens (avoid wasting budget on unfocused searches)
    - The query involves relationships between topics or multi-document synthesis
    - You need to understand what the corpus can and cannot answer

    Returns topic clusters with sample answerable questions, gap signals,
    and a suggested first query — typically ~800 tokens for 'summary' depth.

    After probing, fire `suggested_first_query` via POST /query, then read
    `result_confidence.complexity` to decide whether to expand, intersect, or stop.
    """
    pg = request.app.state.postgres

    max_clusters = 5 if body.depth == "summary" else 10
    aq_limit = 3 if body.depth == "summary" else 5

    raw_clusters = await pg.probe_capabilities(body.context, max_clusters=max_clusters)

    clusters: list[CapabilityCluster] = []
    for c in raw_clusters:
        raw_aqs = c.get("sample_aqs") or []
        if isinstance(raw_aqs, str):
            import json as _json

            raw_aqs = _json.loads(raw_aqs)

        time_range = None
        if c.get("earliest_doc") and c.get("latest_doc"):
            time_range = {
                "earliest": str(c["earliest_doc"])[:7],  # YYYY-MM
                "latest": str(c["latest_doc"])[:7],
            }

        clusters.append(
            CapabilityCluster(
                label=c["label"],
                doc_count=c["doc_count"],
                hub_doc_id=str(c["hub_doc_id"]) if c.get("hub_doc_id") else None,
                coverage_confidence=round(float(c["coverage_confidence"]), 3),
                sample_answerable_questions=raw_aqs[:aq_limit],
                time_range=time_range,
            )
        )

    gap_signals = _build_gap_signals(raw_clusters, body.context)
    suggested = _suggested_first_query(raw_clusters)
    coverage = _coverage_signal(raw_clusters)
    total_docs = sum(c["doc_count"] for c in raw_clusters)

    return ProbeResponse(
        context=body.context,
        coverage_signal=coverage,
        total_matching_docs=total_docs,
        clusters=clusters,
        gap_signals=gap_signals,
        suggested_first_query=suggested,
    )


@router.post(
    "/rebuild",
    summary="Trigger a capability cluster rebuild (admin).",
    include_in_schema=False,
)
@limiter.limit(rate_limit())
async def rebuild_clusters(request: Request) -> dict:  # type: ignore[type-arg]
    """Manually trigger a capability cluster rebuild. Background-safe."""
    import asyncio

    pg = request.app.state.postgres
    asyncio.create_task(pg.rebuild_capability_clusters())
    return {"status": "rebuild started"}


@router.get(
    "/",
    response_model=SystemCapabilities,
    summary="Get system capabilities.",
)
async def get_capabilities(request: Request) -> SystemCapabilities:
    """Returns a map of available system-wide capabilities."""
    from dewie.ingestion.podcast import is_podcast_transcription_available
    return SystemCapabilities(podcast_transcription_available=is_podcast_transcription_available())
