# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Keyword Cluster Traversal API

Agent-native content traversal via keyword concept clusters rather than
document-to-document edge traversal.

Traversal model:
  seed_keywords → { documents[], next_clusters[], metadata }
  agent picks next cluster → repeat

Each step returns:
  - documents: ranked list matching the keyword cluster
  - next_clusters: conceptually adjacent keyword groups with relevance scores,
    anchor docs (why it was suggested), and a context centroid (what it means)
  - metadata: depth, breadth, traversal_id for stateful tracking

Exploration modes:
  - exploit: maximize relevance to current cluster
  - explore: maximize coverage / diversity

Design notes (from peer review):
  - Include confidence scores on both docs and clusters
  - Anchor clusters to source documents to explain *why* they were suggested
  - Provide context_centroid (human-readable cluster summary) for agent reasoning
  - Mitigate keyword drift by using enriched metadata (topics/entities) not raw word co-occurrence
  - Offer exploration_mode param for breadth vs depth control
"""

from __future__ import annotations

import hashlib
import logging
import time as _time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from dewie.api.middleware import limiter, rate_limit
from dewie.storage.postgres import PostgresClient

log = logging.getLogger("dewie.api")

router = APIRouter(prefix="/traverse", tags=["traversal"])

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


# ── Models ──────────────────────────────────────────────────────────────────


class TraverseRequest(BaseModel):
    seed_keywords: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Keyword cluster to start from. Can be topics, entities, or keywords.",
    )
    max_documents: int = Field(
        default=20, ge=1, le=100, description="Max documents to return per traversal step."
    )
    max_next_clusters: int = Field(
        default=5, ge=1, le=20, description="Max next keyword clusters to suggest."
    )
    exploration_mode: Literal["exploit", "explore"] = Field(
        default="exploit",
        description="exploit=maximize relevance, explore=maximize coverage/diversity",
    )
    depth: int = Field(
        default=1, ge=1, description="Current traversal depth (agent-tracked, echoed back)."
    )
    exclude_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords to exclude from next_clusters (e.g., already visited).",
    )
    exclude_doc_ids: list[str] = Field(
        default_factory=list, description="Doc IDs already seen — won't appear in results."
    )
    pin_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Keywords that must persist in every next_cluster suggestion regardless of "
            "which direction the traversal drifts. Acts as an anchor back to the original "
            "query intent. Useful for weaker models that may otherwise lose the thread. "
            "Leave empty (default) to allow unconstrained exploration."
        ),
    )


class DocumentResult(BaseModel):
    id: str
    title: str
    url: str
    summary: str | None
    relevance_score: float = Field(description="0-1, how well this doc matches the cluster")
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    matched_on: list[str] = Field(description="Which seed keywords matched this doc")


class KeywordCluster(BaseModel):
    keywords: list[str]
    relevance_score: float = Field(description="0-1, how relevant this cluster is")
    context_centroid: str = Field(description="One-line summary of what this cluster is about")
    anchor_doc_ids: list[str] = Field(description="Doc IDs that contributed these keywords")
    anchor_doc_titles: list[str] = Field(description="Human-readable anchor doc titles")


class TraversalMetadata(BaseModel):
    traversal_id: str
    depth: int
    seed_keywords: list[str]
    pin_keywords: list[str]
    total_matched: int
    exploration_mode: str
    returned_documents: int
    returned_clusters: int


class TraverseResponse(BaseModel):
    documents: list[DocumentResult]
    next_clusters: list[KeywordCluster]
    metadata: TraversalMetadata


# ── Helpers ─────────────────────────────────────────────────────────────────


def _score_document(doc, seed_keywords: list[str]) -> tuple[float, list[str]]:
    """
    Score a document against seed keywords.
    Checks: topics, entities, keywords fields.
    Returns (score 0-1, matched_terms).
    """
    seed_lower = {k.lower() for k in seed_keywords}
    doc_terms = set()

    for field in (doc.topics or [], doc.entities or [], doc.keywords or []):
        for term in field:
            doc_terms.add(term.lower())

    matched = seed_lower & doc_terms
    if not matched:
        return 0.0, []

    # TF-style score: matched / seed_size, weighted by document coverage
    coverage = len(matched) / len(seed_lower)
    density = len(matched) / max(len(doc_terms), 1)
    score = (coverage * 0.7) + (density * 0.3)

    return round(min(score, 1.0), 4), sorted(matched)


def _build_next_clusters(
    matched_docs: list,
    seed_keywords: list[str],
    exclude_keywords: list[str],
    max_clusters: int,
    exploration_mode: str,
    pin_keywords: list[str] | None = None,
) -> list[KeywordCluster]:
    """
    From the matched documents, aggregate new keyword clusters as next hops.

    Each cluster is a group of co-occurring keywords that appear together
    frequently across multiple documents. Anchored to the docs that contributed.
    """
    seed_lower = {k.lower() for k in seed_keywords}
    exclude_lower = {k.lower() for k in exclude_keywords} | seed_lower

    # Collect keyword → list of (doc_id, doc_title) that have it
    keyword_to_docs: dict[str, list[tuple[str, str]]] = {}

    for doc_result in matched_docs:
        doc_id = doc_result.id
        doc_title = doc_result.title

        all_terms = (
            list(doc_result.topics or [])
            + list(doc_result.entities or [])
            + list(doc_result.keywords or [])
        )

        for term in all_terms:
            t = term.lower().strip()
            if not t or t in exclude_lower or len(t) < 3:
                continue
            if t not in keyword_to_docs:
                keyword_to_docs[t] = []
            keyword_to_docs[t].append((doc_id, doc_title))

    # Score each keyword by document frequency
    keyword_scores: dict[str, float] = {}
    for kw, docs in keyword_to_docs.items():
        doc_freq = len(set(d[0] for d in docs))

        if exploration_mode == "exploit":
            # Weight towards keywords that appear in high-scoring docs
            score = doc_freq / max(len(matched_docs), 1)
        else:  # explore
            # Weight towards keywords that spread across many docs
            score = doc_freq / max(len(matched_docs), 1) * (1 + 0.1 * doc_freq)

        keyword_scores[kw] = score

    # Sort and group into clusters by co-occurrence
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)

    # Simple clustering: group top keywords that share the same anchor docs
    # This keeps clusters coherent and explained
    cluster_map: dict[str, list[str]] = {}  # anchor_key → keywords
    anchor_docs_map: dict[str, list[tuple[str, str]]] = {}

    for kw, score in sorted_keywords[:50]:  # consider top 50 candidates
        # Key = frozenset of top-2 anchor doc IDs
        anchor_ids = sorted(set(d[0] for d in keyword_to_docs[kw]))[:3]
        anchor_key = "|".join(anchor_ids[:2])

        if anchor_key not in cluster_map:
            cluster_map[anchor_key] = []
            anchor_docs_map[anchor_key] = [
                (d[0], d[1]) for d in keyword_to_docs[kw] if d[0] in anchor_ids
            ]

        cluster_map[anchor_key].append(kw)

    # Build KeywordCluster objects
    clusters = []
    for anchor_key, kws in cluster_map.items():
        if not kws:
            continue

        anchor_doc_pairs = anchor_docs_map.get(anchor_key, [])
        anchor_ids = list(dict.fromkeys(d[0] for d in anchor_doc_pairs))[:3]
        anchor_titles = list(dict.fromkeys(d[1] for d in anchor_doc_pairs))[:3]

        # Cluster relevance = avg score of keywords in it
        cluster_score = sum(keyword_scores.get(k, 0) for k in kws[:5]) / max(len(kws[:5]), 1)

        # Context centroid: top keywords as readable phrase
        top_kws = kws[:4]
        centroid = ", ".join(top_kws)

        clusters.append(
            KeywordCluster(
                keywords=kws[:8],
                relevance_score=round(min(cluster_score, 1.0), 4),
                context_centroid=centroid,
                anchor_doc_ids=anchor_ids,
                anchor_doc_titles=anchor_titles,
            )
        )

    # Sort clusters by relevance
    clusters.sort(key=lambda c: c.relevance_score, reverse=True)

    # Inject pinned keywords at the front of every cluster's keyword list.
    # This ensures that when the agent uses a next_cluster as its next hop,
    # the original intent travels with it — acting as a breadcrumb.
    if pin_keywords:
        pin_lower = [p.lower().strip() for p in pin_keywords if p.strip()]
        for cluster in clusters:
            # Prepend pins that aren't already in the cluster
            existing = {k.lower() for k in cluster.keywords}
            prepend = [p for p in pin_lower if p not in existing]
            if prepend:
                cluster.keywords = prepend + cluster.keywords

    return clusters[:max_clusters]


# ── Endpoint ─────────────────────────────────────────────────────────────────


def _get_pg(request: Request) -> PostgresClient:
    return request.app.state.postgres


@router.post("", response_model=TraverseResponse)
@limiter.limit(rate_limit())
async def traverse(
    body: TraverseRequest,
    request: Request,
) -> TraverseResponse:
    """
    Keyword cluster traversal endpoint.

    Given a set of seed keywords, returns:
    - Ranked documents matching that keyword cluster
    - Next keyword clusters to explore (with relevance scores and anchor docs)
    - Traversal metadata for stateful agent tracking

    This is the primary agent-facing traversal API. Agents should:
    1. Start with intent-derived seed keywords
    2. Pick from next_clusters to continue exploration
    3. Accumulate documents across steps
    4. Use exclude_keywords/exclude_doc_ids to avoid revisiting
    """
    pg: PostgresClient = _get_pg(request)
    t0 = _time.time()

    request_id = _extract_request_id(request)
    log.info(
        "[traverse] request_start | request_id=%s | seed_keywords=%s",
        request_id,
        _redact(str(body.seed_keywords)),
    )

    try:
        # Load all ready documents
        # TODO: For large corpora, replace with indexed query
        all_docs = await pg.list_recent(limit=2000)

        if not all_docs:
            raise HTTPException(status_code=503, detail="No documents available.")

        # Score every document against seed keywords
        scored: list[tuple[float, list[str], object]] = []
        for doc in all_docs:
            if str(doc.id) in body.exclude_doc_ids:
                continue
            score, matched = _score_document(doc, body.seed_keywords)
            if score > 0:
                scored.append((score, matched, doc))

        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)
        total_matched = len(scored)

        if body.exploration_mode == "explore":
            # Diversity: interleave high and medium scorers
            top = scored[: body.max_documents // 2]
            mid = scored[body.max_documents // 2 : body.max_documents * 2]
            selected = (top + mid[::2])[: body.max_documents]
        else:
            selected = scored[: body.max_documents]

        # Build DocumentResult list
        doc_results: list[DocumentResult] = []
        for score, matched, doc in selected:
            doc_results.append(
                DocumentResult(
                    id=str(doc.id),
                    title=doc.title or "",
                    url=doc.url or "",
                    summary=doc.summary,
                    relevance_score=score,
                    topics=doc.topics or [],
                    entities=(doc.entities or [])[:10],
                    keywords=(doc.keywords or [])[:10],
                    matched_on=matched,
                )
            )

        # Build next keyword clusters from matched documents
        next_clusters = _build_next_clusters(
            matched_docs=doc_results,
            seed_keywords=body.seed_keywords,
            exclude_keywords=body.exclude_keywords,
            max_clusters=body.max_next_clusters,
            exploration_mode=body.exploration_mode,
            pin_keywords=body.pin_keywords,
        )

        # Stable traversal ID from seed
        traversal_id = hashlib.md5("|".join(sorted(body.seed_keywords)).encode()).hexdigest()[:12]

        response = TraverseResponse(
            documents=doc_results,
            next_clusters=next_clusters,
            metadata=TraversalMetadata(
                traversal_id=traversal_id,
                depth=body.depth,
                seed_keywords=body.seed_keywords,
                pin_keywords=body.pin_keywords,
                total_matched=total_matched,
                exploration_mode=body.exploration_mode,
                returned_documents=len(doc_results),
                returned_clusters=len(next_clusters),
            ),
        )

        elapsed = _time.time() - t0
        log.info(
            "[traverse] request_success | request_id=%s | elapsed_ms=%.1f | "
            "returned_documents=%d | returned_clusters=%d | total_matched=%d",
            request_id,
            elapsed * 1000,
            len(doc_results),
            len(next_clusters),
            total_matched,
        )

        try:
            from dewie.storage.query_logger import QueryLogEntry
            from dewie.storage.query_logger import log_query as _log_query

            await _log_query(
                QueryLogEntry(
                    question=", ".join(body.seed_keywords),
                    source="api",
                    workspace_ids=getattr(request.state, "workspace_ids", []),
                    user_id=getattr(request.state, "user_id", None),
                    elapsed_ms=int(elapsed * 1000),
                )
            )
        except Exception:
            pass

        return response

    except HTTPException:
        raise

    except Exception as e:
        elapsed = _time.time() - t0
        log.error(
            "[traverse] request_error | request_id=%s | elapsed_ms=%.1f | error=%s",
            request_id,
            elapsed * 1000,
            str(e),
            exc_info=True,
        )
        raise
