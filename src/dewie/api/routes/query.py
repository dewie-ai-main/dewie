# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
/query endpoint — agent-friendly document search.

Returns SearchResponse with full metadata per result so agents can
re-rank by their own user context (who is asking, why they're asking).
"""

from __future__ import annotations

import logging
import time as _time

from fastapi import APIRouter, HTTPException, Query, Request

from dewie.api.middleware import limiter, rate_limit
from dewie.config import settings
from dewie.models.query import (
    BenchmarkResponse,
    BenchmarkResult,
    CategoryHint,
    CategoryQueryResponse,
    ResultConfidence,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from dewie.query.expand import expand_query
from dewie.storage.cache import CacheClient
from dewie.storage.postgres import PostgresClient
from dewie.storage.query_logger import QueryLogEntry
from dewie.storage.query_logger import log_query as _log_query
from dewie.storage.rankers import _edge_counts, apply_staleness_penalty

log = logging.getLogger("dewie.api")

router = APIRouter(prefix="/query", tags=["query"])

# ── Reranking ─────────────────────────────────────────────────────────────────
# Number of doc-level candidates to fetch before chunk-based reranking.
_RERANK_CANDIDATE_COUNT = 20

# ── Diverse keyword selection ─────────────────────────────────────────────────
# Tuning knobs — all in one place.
_KW_TOP_N = 3  # keywords returned per result
_KW_MAX_OVERLAP = 0.4  # Jaccard threshold above which a candidate is too similar to a selected kw
_KW_MIN_TOKEN_LEN = 3  # tokens shorter than this are treated as stop-words and ignored
_KW_MIN_EDGE_COUNT = (
    3  # documents with fewer edges than this get no keywords — they're stubs, not hubs
)

_STOP_WORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "are",
        "was",
        "its",
        "has",
        "have",
        "had",
        "not",
        "but",
        "about",
        "also",
    }
)


def _token_set(phrase: str) -> frozenset[str]:
    """Lowercase, split, drop stop-words and short tokens."""
    return frozenset(
        t
        for t in phrase.lower().replace("-", " ").split()
        if len(t) >= _KW_MIN_TOKEN_LEN and t not in _STOP_WORDS
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _diverse_keywords(keywords: list[str], n: int = _KW_TOP_N) -> list[str]:
    """
    Greedy max-diversity selection: return up to ``n`` keywords that are
    mutually dissimilar by Jaccard token overlap.

    Candidates are taken in the order they arrive (caller controls ranking).
    Each candidate is skipped if its token overlap with any already-selected
    keyword exceeds ``_KW_MAX_OVERLAP``.
    """
    selected: list[str] = []
    selected_tokens: list[frozenset[str]] = []

    for kw in keywords:
        if len(selected) >= n:
            break
        tokens = _token_set(kw)
        if any(_jaccard(tokens, t) > _KW_MAX_OVERLAP for t in selected_tokens):
            continue
        selected.append(kw)
        selected_tokens.append(tokens)

    return selected


def _get_pg(request: Request) -> PostgresClient:
    return request.app.state.postgres


def _is_valid_result(r: dict) -> bool:
    """Check if a remote result dict has the minimum fields for SearchResult."""
    return bool(r.get("doc_id") and r.get("title"))


def _get_cache(request: Request) -> CacheClient:
    return request.app.state.cache


def _compute_gap_signal(query: str, results: list[SearchResult]) -> str | None:
    """
    Detect when a query hits a genuine coverage gap — the information is absent
    from the corpus, not just hard to retrieve.

    A gap is distinct from low confidence:
      - Low confidence (ambiguous/distributed): information exists, needs deeper traversal.
      - Gap: information is not here; deeper traversal won't help.

    Uses relative signals, not absolute score thresholds, because RRF floor scores
    are corpus-size dependent and even irrelevant docs return non-zero scores.

    Returns a human + agent-readable string explaining the gap, or None if no gap detected.
    """
    # Hard gap: zero results
    if not results:
        return (
            f"No documents found for '{query}'. "
            "This topic appears to be absent from the corpus. "
            "Broaden the query, ingest relevant sources, or check an external source."
        )

    # Collect distinctive query words (len > 4 to skip stop-words)
    query_words = {w.lower() for w in query.split() if len(w) > 4}
    if not query_words:
        return None

    # AQ coverage: check per-document, not combined across docs.
    # Combined AQ is deceptive — "icelandic" (venture fund), "geological" (Arizona soc),
    # "survey" (NHS) across 3 unrelated docs fake-covers a volcanic query.
    # A genuine match requires ONE document to answer multiple aspects of the query.
    max_single_doc_aq = 0
    for r in results[:3]:
        if r.answers_questions:
            aq_text = " ".join(r.answers_questions).lower()
            covered = sum(1 for w in query_words if w in aq_text)
            max_single_doc_aq = max(max_single_doc_aq, covered)
    # Also track combined for the gap message (uncovered words)
    combined_aq = " ".join(
        " ".join(r.answers_questions).lower() for r in results[:3] if r.answers_questions
    )
    aq_threshold = max(2, round(len(query_words) * 0.33)) if len(query_words) >= 3 else 1
    no_aq_match = max_single_doc_aq < aq_threshold

    # Score uniformity: if all results cluster tightly (no discrimination), corpus
    # didn't actually find relevant docs — it just matched tokens randomly.
    # A real match has a clear winner; a gap has flat scores across all results.
    top_score = results[0].score
    bottom_score = results[-1].score if len(results) > 1 else top_score
    score_spread = (top_score - bottom_score) / top_score if top_score > 0 else 0.0
    scores_undiscriminated = score_spread < 0.15  # all results within 15% of each other

    # Topic coherence: require majority of query words to appear in top-result topics.
    # "any" match is too loose — a finance topic containing "icelandic" looks like
    # a topic match for a volcanic geology query, which it isn't.
    all_result_topics = " ".join(t.lower() for r in results[:3] for t in (r.topics or []))
    topics_covered = sum(1 for w in query_words if w in all_result_topics)
    topics_match_query = topics_covered >= max(1, len(query_words) * 0.5)

    if no_aq_match and scores_undiscriminated and not topics_match_query:
        uncovered = sorted(w for w in query_words if w not in combined_aq)[:4]
        topic_hint = f" No coverage found for: {', '.join(uncovered)}." if uncovered else ""
        return (
            f"{len(results)} result(s) returned but none address '{query}' directly.{topic_hint} "
            "The corpus appears to cover adjacent or unrelated topics. "
            "Retrying with a rephrased query is unlikely to help — "
            "consider ingesting relevant sources or checking an external knowledge base."
        )

    return None


def _compute_confidence(query: str, results: list[SearchResult]) -> ResultConfidence | None:
    """
    Compute adaptive retrieval signals from already-ranked results.

    Three complexity buckets:
      lookup      — clear winner, corpus directly answers the query
      ambiguous   — multiple docs compete (low score_gap); neighbours may help
      distributed — answer is spread across the corpus; bridge or intersect needed
    """
    if not results:
        return None

    top = results[0]

    # score_gap: how dominant is the top result?
    score_gap = round(top.score - results[1].score, 4) if len(results) > 1 else 0.0

    # aq_coverage_ratio: what fraction of meaningful query words appear in AQ questions?
    query_words = {w.lower() for w in query.split() if len(w) > 3}
    if query_words and top.answers_questions:
        aq_text = " ".join(top.answers_questions).lower()
        covered = sum(1 for w in query_words if w in aq_text)
        aq_coverage_ratio = round(covered / len(query_words), 3)
    else:
        aq_coverage_ratio = 0.0

    # edge_density: how connected is the top result? Cap at 50 edges = 1.0
    edge_density = round(min(top.edge_count / 50.0, 1.0), 3)

    # topic spread: are top results from different topic clusters?
    # High spread + low aq_coverage = answer is distributed across documents
    topic_spread = 0.0
    if len(results) >= 3:
        top3_topics = [set(r.topics) for r in results[:3]]
        union = top3_topics[0] | top3_topics[1] | top3_topics[2]
        intersection = top3_topics[0] & top3_topics[1] & top3_topics[2]
        if union:
            # High ratio = topics overlap (same cluster); low = distributed
            overlap_ratio = len(intersection) / len(union)
            topic_spread = round(1.0 - overlap_ratio, 3)

    # ── Classify complexity ──────────────────────────────────────────────────
    if score_gap > 0.2 and aq_coverage_ratio > 0.5:
        # One doc clearly wins and it covers the query
        complexity = "lookup"
        confidence_level = "high"
        suggested_action = "none"
    elif topic_spread > 0.6 and aq_coverage_ratio < 0.3:
        # Results span different topic clusters, no single doc has the answer
        complexity = "distributed"
        confidence_level = "low"
        suggested_action = "bridge" if edge_density > 0.2 else "intersect"
    elif score_gap < 0.05 or aq_coverage_ratio < 0.2:
        # Multiple docs competing but not clearly distributed
        complexity = "ambiguous"
        confidence_level = "low"
        suggested_action = "expand" if edge_density > 0.3 else "intersect"
    else:
        complexity = "ambiguous"
        confidence_level = "medium"
        suggested_action = "expand" if edge_density > 0.3 else "intersect"

    probe_hint = None
    if complexity == "distributed":
        probe_hint = (
            "Answer appears distributed across document clusters. "
            "Call POST /capabilities/probe with your query context to map coverage "
            "before retrying — saves context budget and improves answer quality."
        )

    gap_signal = _compute_gap_signal(query, results)

    return ResultConfidence(
        score_gap=score_gap,
        aq_coverage_ratio=aq_coverage_ratio,
        edge_density=edge_density,
        complexity=complexity,
        confidence_level=confidence_level,
        suggested_action=suggested_action,
        probe_hint=probe_hint,
        gap_signal=gap_signal,
    )


async def _fan_out_query(
    src_type: str,
    config: dict,
    body: SearchRequest,
    source_id: str,
) -> list[SearchResult]:
    """Fan-out a query to a remote source and return merged results."""

    if src_type == "mcp":
        return await _fan_out_mcp(config, body, source_id)
    elif src_type == "postgres":
        return await _fan_out_postgres(config, body, source_id)
    return []


async def _fan_out_mcp(
    config: dict,
    body: SearchRequest,
    source_id: str,
) -> list[SearchResult]:
    """Forward query to a remote Dewie MCP instance."""

    import httpx

    endpoint = str(config.get("endpoint", "")).strip().rstrip("/")
    if not endpoint:
        return []

    api_key = str(config.get("api_key", "")).strip()
    headers: dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key

    # Try /api/query first, then fall back to /query
    endpoints_to_try = [f"{endpoint}/api/query", f"{endpoint}/query"]

    deduped: list[str] = []
    for url in endpoints_to_try:
        if url not in deduped:
            deduped.append(url)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            for url in deduped:
                try:
                    resp = await client.post(
                        url,
                        json={"query": body.query, "limit": body.limit, "ranker": body.ranker},
                        headers=headers,
                        follow_redirects=True,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return _parse_remote_results(data, endpoint, source_id)
                except httpx.HTTPStatusError:
                    continue
                except Exception:
                    continue
    except Exception:
        pass
    return []


def _parse_remote_results(
    data: dict, endpoint: str, source_id: str
) -> list[SearchResult]:
    """Parse remote search response into SearchResult list with source_node set."""
    results: list[SearchResult] = []
    remote_docs = data.get("results", [])
    seen_ids: set[str] = set()

    for doc in remote_docs:
        doc_id = str(doc.get("doc_id", ""))
        if not doc_id or doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        results.append(
            SearchResult(
                doc_type=doc.get("doc_type"),
                doc_id=doc_id,
                title=doc.get("title", ""),
                summary=doc.get("summary"),
                url=doc.get("url"),
                source=doc.get("source"),
                topics=doc.get("topics", []),
                keywords=doc.get("keywords", []),
                entities=doc.get("entities", []),
                sentiment=doc.get("sentiment"),
                score=doc.get("score", 0.0),
                edge_count=doc.get("edge_count", 0),
                ingested_at=doc.get("ingested_at"),
                enrichment_quality_score=doc.get("enrichment_quality_score"),
                reading_level=doc.get("reading_level"),
                source_node=endpoint,
            )
        )
    return results


async def _fan_out_postgres(
    config: dict,
    body: SearchRequest,
    source_id: str,
) -> list[SearchResult]:
    """Query a remote Postgres dewie instance directly."""
    try:
        from sqlalchemy import text as _sql_text
        from sqlalchemy.ext.asyncio import create_async_engine

        dsn = str(config.get("dsn", "")).strip()
        if not dsn:
            host = str(config.get("host", "localhost")).strip()
            port = str(config.get("port", "5432")).strip()
            database = str(config.get("database", config.get("dbname", "postgres"))).strip()
            user = str(config.get("user", config.get("username", "postgres"))).strip()
            password = str(config.get("password", "")) or ""
            if password:
                dsn = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}?ssl=disable"
            else:
                dsn = f"postgresql+asyncpg://{user}@{host}:{port}/{database}?ssl=disable"

        if not dsn:
            return []

        engine = create_async_engine(dsn, pool_size=1, max_overflow=0, echo=False)

        try:
            async with engine.connect() as conn:
                q = f"%{body.query[:100]}%"
                rows = (
                    await conn.execute(
                        _sql_text("""
                            SELECT id, title, summary, url, source, topics, keywords,
                                   entities, sentiment, document_type, ingested_at,
                                   enrichment_quality_score, reading_level
                            FROM documents
                            WHERE status = 'ready'
                              AND (
                                lower(title) LIKE lower(:q)
                                OR lower(summary) LIKE lower(:q)
                                OR lower(COALESCE(keywords::text, '')) LIKE lower(:q)
                                OR lower(COALESCE(topics::text, '')) LIKE lower(:q)
                              )
                            ORDER BY ingested_at DESC
                            LIMIT :limit
                        """),
                        {"q": q, "limit": body.limit},
                    )
                ).mappings().all()


            results: list[SearchResult] = []
            seen_ids: set[str] = set()

            for row in rows:
                doc_id = str(row.get("id", ""))
                if not doc_id or doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                dt = row.get("document_type")
                rl = row.get("reading_level")

                results.append(
                    SearchResult(
                        doc_type=str(dt) if dt else None,
                        doc_id=doc_id,
                        title=str(row.get("title", "")),
                        summary=row.get("summary"),
                        url=row.get("url"),
                        source=row.get("source"),
                        topics=row.get("topics") or [],
                        keywords=row.get("keywords") or [],
                        entities=row.get("entities") or [],
                        sentiment=float(row.get("sentiment", 0) or 0),
                        score=1.0,
                        edge_count=0,
                        ingested_at=row.get("ingested_at") or None,
                        enrichment_quality_score=int(row.get("enrichment_quality_score", 0) or 0) if row.get("enrichment_quality_score") is not None else None,
                        reading_level=str(rl) if rl else None,
                        source_node=config.get("endpoint", dsn),
                    )
                )
            return results
        finally:
            await engine.dispose()
    except Exception:
        return []


@router.get("/rankers", summary="List available ranking strategies.")
async def list_rankers() -> list[dict]:
    """Return all registered ranker strategies with id, label, description."""
    from dewie.storage.rankers import list_rankers as _list

    return _list()


@router.post(
    "",
    response_model=SearchResponse,
    summary="Search for content matching a query string.",
)
@limiter.limit(rate_limit())
async def query(
    request: Request,
    body: SearchRequest,
    chunks: bool = Query(
        default=False,
        description="When true, surface the best matching chunk text alongside each result.",
    ),
    rerank: bool = Query(
        default=False,
        description=(
            "When true, runs a two-stage hierarchical retrieval: "
            "fetch the top 20 doc-level candidates, then re-rank them by best chunk similarity. "
            "Returns the top body.limit results with chunk_match and chunk_score populated. "
            "Supersedes chunks=true."
        ),
    ),
) -> SearchResponse:
    """
    Hybrid full-text + semantic search.  Returns one ``SearchResult`` per
    document with topics, keywords, entities, answers_questions, and
    edge_count so agents can re-rank results by their user context.

    Pass ``ranker`` in the request body to select a ranking strategy.
    See GET /query/rankers for available options.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    log.info(
        "query started",
        extra={
            "request_id": request_id,
            "query": body.query[:1000],
            "ranker": getattr(body, "ranker", None),
            "limit": body.limit,
            "rerank": rerank,
        },
    )
    t0 = _time.time()
    pg = _get_pg(request)
    cache = _get_cache(request)
    ranker = getattr(body, "ranker", settings.query_default_ranker) or settings.query_default_ranker

    # Expand the query with synonyms for common terms.
    # The original query is preserved in the response; expansion only enriches search.
    expanded_query = expand_query(body.query)

    # Source filtering: if source_id is provided, fan-out to the remote source.
    source_id = getattr(body, "source_id", None)

    # Reranked results are cached separately — they differ in score order and chunk data.
    if not source_id:
        cache_key = f"{expanded_query}::{ranker}" + ("::rerank" if rerank else "")
        cached = await cache.get_query_result(cache_key, "search")
        if cached:
            return SearchResponse(**cached)

    # Resolve quality filter — request params override server config defaults
    from pathlib import Path

    import yaml as _yaml

    _cfg_path = (
        Path("/app/dewie.yml") if Path("/app/dewie.yml").exists() else Path("dewie.yml")
    )
    try:
        _cfg = _yaml.safe_load(_cfg_path.read_text()) if _cfg_path.exists() else {}
    except Exception:
        _cfg = {}
    _qf = _cfg.get("ingest", {}).get("quality_filter", {})
    cfg_min_qs: int = _qf.get("min_enrichment_quality_score", -1)
    cfg_excl_rl: list[str] = _qf.get("exclude_reading_levels", [])

    min_qs = (
        body.min_enrichment_quality_score
        if body.min_enrichment_quality_score is not None
        else (cfg_min_qs if cfg_min_qs >= 0 else None)
    )
    excl_rl = (
        body.exclude_reading_levels
        if body.exclude_reading_levels is not None
        else (cfg_excl_rl or None)
    )

    # When reranking, fetch a larger candidate pool so chunk reranking has something to work with.
    # Category → source filter mapping
    _CATEGORY_SOURCES: dict[str, list[str]] = {
        "sports": [
            "sports.yahoo.com",
            "www.espn.com",
            "www.sportingnews.com",
            "www.cbssports.com",
            "www.nba.com",
            "www.nfl.com",
            "www.mlb.com",
            "www.nhl.com",
            "www.bleacherreport.com",
            "theathletic.com",
        ],
        "finance": [
            "finance.yahoo.com",
            "www.fool.com",
            "www.wsj.com",
            "www.bloomberg.com",
            "www.ft.com",
            "www.marketwatch.com",
            "www.cnbc.com",
            "www.barrons.com",
            "www.sec.gov",
            "www.investopedia.com",
            "seekingalpha.com",
        ],
        "tech": [
            "techcrunch.com",
            "www.theverge.com",
            "arstechnica.com",
            "www.wired.com",
            "venturebeat.com",
            "news.ycombinator.com",
            "www.ycombinator.com",
            "arxiv.org",
        ],
        "news": [
            "www.reuters.com",
            "apnews.com",
            "www.bbc.com",
            "www.theguardian.com",
            "www.nytimes.com",
            "www.cbsnews.com",
            "abcnews.com",
            "www.al-monitor.com",
            "www.axios.com",
        ],
        "science": [
            "arxiv.org",
            "www.nature.com",
            "www.ncbi.nlm.nih.gov",
            "pubmed.ncbi.nlm.nih.gov",
            "www.sciencedirect.com",
        ],
    }
    category = getattr(body, "category", None)
    source_filter = _CATEGORY_SOURCES.get(category.lower()) if category else None

    # Source filtering: if source_id is provided, fan-out to the remote source.
    _remote_results: list[SearchResult] = []
    if source_id:
        src = await pg.get_source(source_id)
        if not src:
            raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found.")
        src_type = str(src.get("type", ""))
        config = src.get("config") if isinstance(src.get("config"), dict) else {}
        _remote_results = await _fan_out_query(src_type, config, body, source_id)

    # When source_id fan-out already returned results, skip local search.
    if _remote_results:
        results = _remote_results
    else:
        search_limit = _RERANK_CANDIDATE_COUNT if rerank else body.limit

        raw_results = await pg.search(
            expanded_query,
            limit=search_limit,
            ranker=ranker,
            min_enrichment_quality_score=min_qs,
            exclude_reading_levels=excl_rl,
            workspace_ids=getattr(request.state, "workspace_ids", []),
            published_after=getattr(body, "published_after", None),
            published_before=getattr(body, "published_before", None),
            source_filter=source_filter,
        )

        doc_ids = [str(doc.id) for doc, _ in raw_results]
        async with pg._session_factory() as _session:
            _edge_count_map: dict[str, int] = dict(await _edge_counts(_session, doc_ids))

        results: list[SearchResult] = []
        for doc, score in raw_results:
            edge_count = _edge_count_map.get(str(doc.id), 0)
            results.append(
                SearchResult(
                    doc_type=doc.document_type.value if doc.document_type else None,
                    doc_id=str(doc.id),
                    title=doc.title,
                    summary=doc.summary or None,
                    url=doc.url or None,
                    source=doc.source or None,
                    topics=doc.topics,
                    keywords=_diverse_keywords(doc.keywords)
                    if edge_count >= _KW_MIN_EDGE_COUNT
                    else [],
                    entities=doc.entities,
                    sentiment=doc.sentiment,
                    answers_questions=doc.answers_questions,
                    score=score,
                    edge_count=edge_count,
                    ingested_at=doc.ingested_at,
                    enrichment_quality_score=doc.enrichment_quality_score,
                    reading_level=doc.reading_level.value if doc.reading_level else None,
                )
            )

    # ── Normalize scores to 0-1 range ─────────────────────────────────────────
    # RRF scores are tiny (≈0.016) and non-interpretable. Normalize so rank-1 = 1.0
    # and other scores are proportional. Preserves relative ordering.
    if results:
        _max_score = results[0].score
        if _max_score > 1e-9:
            for r in results:
                r.score = round(r.score / _max_score, 4)
        else:
            for r in results:
                r.score = 0.0

    # ── Hierarchical chunk reranking ──────────────────────────────────────────
    if rerank and results:
        # Fetch the best-matching chunk for each candidate document.
        doc_ids = [r.doc_id for r in results]
        chunk_matches = await pg.search_chunks_for_docs(expanded_query, doc_ids)

        # Annotate results with chunk scores and text.
        for r in results:
            match = chunk_matches.get(r.doc_id)
            if match:
                r.chunk_score = match["score"]
                r.chunk_match = match["text"]

        # Re-rank by merging doc-level and chunk scores so that a strong
        # doc-level RRF score is never discarded just because a doc has no chunks.
        results.sort(
            key=lambda r: (
                max(r.score, 0.8 * r.chunk_score) if r.chunk_score is not None else r.score
            ),
            reverse=True,
        )

        # Trim to the originally-requested limit.
        results = results[: body.limit]

    # ── Plain chunk augmentation (always on — no reranking) ──────────────────
    elif results:
        doc_ids = [r.doc_id for r in results]
        chunk_matches = await pg.search_chunks_for_docs(expanded_query, doc_ids)
        for r in results:
            match = chunk_matches.get(r.doc_id)
            if match:
                r.chunk_match = match["text"]
                r.chunk_score = match["score"]

    # ── Staleness penalty (opt-in) ────────────────────────────────────────────
    if getattr(body, "staleness_penalty", False) and results:
        apply_staleness_penalty(results)
        results.sort(key=lambda r: r.score, reverse=True)

    confidence = _compute_confidence(body.query, results)
    # Gap signal fires even on empty results (_compute_confidence returns None then)
    if confidence is None and not results:
        gap_signal = _compute_gap_signal(body.query, results)
        if gap_signal:
            from dewie.models.query import ResultConfidence as _RC

            confidence = _RC(
                score_gap=0.0,
                aq_coverage_ratio=0.0,
                edge_density=0.0,
                complexity="ambiguous",
                confidence_level="low",
                suggested_action="none",
                gap_signal=gap_signal,
            )

    # ── Gap enrichment trigger ────────────────────────────────────────────────
    # Fire-and-forget: enqueue for Brave enrichment when coverage is thin.
    gap_signal_val = confidence.gap_signal if confidence else None
    top_score = results[0].score if results else 0.0
    # After normalization: top=1.0 always. Use score_gap as quality signal instead.
    # A gap_signal or very low aq_coverage means poor retrieval quality.
    aq_cov = confidence.aq_coverage_ratio if confidence else 0.0
    fallback_triggered = gap_signal_val is not None or (aq_cov < 0.1 and top_score < 0.8)
    gap_enrichment_queued = False
    if fallback_triggered and not getattr(body, "disable_enrichment", False):
        category = getattr(body, "category", None)
        import logging as _logging

        _log = _logging.getLogger(__name__)
        _log.info(
            "[query] fallback triggered for '%s' (category=%s, top_score=%.3f)",
            body.query,
            category,
            top_score,
        )
        try:
            queued, _qid = await pg.enqueue_search(query=body.query, category=category)
            gap_enrichment_queued = queued
        except Exception:
            pass

    # ── Federation parked for OSS release ─────────────────────────────────────
    # Tiered gap escalation (personal → workspace → public) and peer search
    # are future features. max_tier field is accepted but ignored for now.

    response = SearchResponse(
        query=body.query,
        results=results,
        total=len(results),
        result_confidence=confidence,
        fallback_triggered=fallback_triggered,
        gap_enrichment_queued=gap_enrichment_queued,
        source_id=source_id if source_id else None,
    )
    if not source_id:
        await cache.set_query_result(cache_key, "search", response.model_dump(mode="json"))

    elapsed = _time.time() - t0
    docs_log = [{"doc_id": r.doc_id, "title": r.title} for r in results]
    query_id: str | None = None
    try:
        await _log_query(
            QueryLogEntry(
                question=body.query,
                source="api",
                user_id=getattr(request.state, "user_id", None),
                hops=0,
                docs_returned=docs_log,
                full_results=response.model_dump(mode="json"),
                elapsed_ms=int(elapsed * 1000),
            )
        )
        # Fetch the log ID back for the response
        async with pg._engine.connect() as _conn:
            from sqlalchemy import text as _qid_text

            _row = await _conn.execute(
                _qid_text("""
                SELECT id FROM query_log
                WHERE question = :q AND user_id = :uid
                ORDER BY ts DESC LIMIT 1
            """),
                {"q": body.query, "uid": getattr(request.state, "user_id", None)},
            )
            _r = _row.fetchone()
            if _r:
                query_id = str(_r[0])
    except Exception:
        pass

    response.query_id = query_id
    log.info(
        "query succeeded",
        extra={
            "request_id": request_id,
            "total": response.total,
            "elapsed_s": round(_time.time() - t0, 3),
        },
    )
    return response


# ── Benchmark endpoint ────────────────────────────────────────────────────────


@router.get(
    "/benchmark",
    response_model=BenchmarkResponse,
    summary="Compare standard vs. chunk-reranked results for a query.",
)
@limiter.limit(rate_limit())
async def benchmark_rerank(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500, description="Query string to benchmark."),
    limit: int = Query(default=10, ge=1, le=50, description="Number of results per method."),
) -> BenchmarkResponse:
    """
    Runs the same query using both standard doc-level ranking and two-stage
    chunk-based reranking, then returns side-by-side results with rank-change
    analysis.

    Intended for offline evaluation of reranking quality, not production use.
    Bypasses cache and query logging. Expect higher latency than /query.
    """
    import asyncio

    pg = _get_pg(request)
    workspace_ids = getattr(request.state, "workspace_ids", [])

    # Run both searches concurrently — standard uses body.limit, rerank uses the
    # wider candidate pool so chunk scores have more candidates to re-order.
    standard_raw, rerank_raw = await asyncio.gather(
        pg.search(q, limit=limit, ranker="rrf", workspace_ids=workspace_ids),
        pg.search(q, limit=_RERANK_CANDIDATE_COUNT, ranker="rrf", workspace_ids=workspace_ids),
    )

    def _build_result(doc, score, edge_count: int) -> SearchResult:
        return SearchResult(
            doc_type=doc.document_type.value if doc.document_type else None,
            doc_id=str(doc.id),
            title=doc.title,
            summary=doc.summary or None,
            url=doc.url or None,
            source=doc.source or None,
            topics=doc.topics,
            keywords=_diverse_keywords(doc.keywords) if edge_count >= _KW_MIN_EDGE_COUNT else [],
            entities=doc.entities,
            sentiment=doc.sentiment,
            answers_questions=doc.answers_questions,
            score=score,
            edge_count=edge_count,
            ingested_at=doc.ingested_at,
            enrichment_quality_score=doc.enrichment_quality_score,
            reading_level=doc.reading_level.value if doc.reading_level else None,
        )

    # Fetch edge counts in parallel for both result sets.
    std_edge_counts: list[int] = (
        list(await asyncio.gather(*[pg.get_edge_count(doc.id) for doc, _ in standard_raw]))
        if standard_raw
        else []
    )
    rrk_edge_counts: list[int] = (
        list(await asyncio.gather(*[pg.get_edge_count(doc.id) for doc, _ in rerank_raw]))
        if rerank_raw
        else []
    )

    standard_results = [
        _build_result(doc, score, ec) for (doc, score), ec in zip(standard_raw, std_edge_counts)
    ]
    rerank_results = [
        _build_result(doc, score, ec) for (doc, score), ec in zip(rerank_raw, rrk_edge_counts)
    ]

    # Annotate rerank candidates with chunk scores.
    rerank_doc_ids = [r.doc_id for r in rerank_results]
    if rerank_doc_ids:
        chunk_matches = await pg.search_chunks_for_docs(q, rerank_doc_ids)
        for r in rerank_results:
            match = chunk_matches.get(r.doc_id)
            if match:
                r.chunk_score = match["score"]
                r.chunk_match = match["text"]

    # Sort rerank candidates by chunk score; docs without chunks fall to the bottom.
    rerank_results.sort(
        key=lambda r: r.chunk_score if r.chunk_score is not None else -1.0,
        reverse=True,
    )
    rerank_results = rerank_results[:limit]

    # ── Build comparison table ────────────────────────────────────────────────
    standard_rank_map = {r.doc_id: i + 1 for i, r in enumerate(standard_results)}
    rerank_rank_map = {r.doc_id: i + 1 for i, r in enumerate(rerank_results)}

    # Scores and titles accessible for any doc in either list.
    score_map: dict[str, float] = {r.doc_id: r.score for r in rerank_results}
    score_map.update({r.doc_id: r.score for r in standard_results})  # standard wins on tie
    chunk_score_map: dict[str, float | None] = {r.doc_id: r.chunk_score for r in rerank_results}
    title_map: dict[str, str] = {r.doc_id: r.title for r in rerank_results}
    title_map.update({r.doc_id: r.title for r in standard_results})

    # Union of all doc_ids, preserving standard order then rerank extras.
    seen: set[str] = set()
    all_doc_ids: list[str] = []
    for r in standard_results + rerank_results:
        if r.doc_id not in seen:
            all_doc_ids.append(r.doc_id)
            seen.add(r.doc_id)

    comparison: list[BenchmarkResult] = []
    for doc_id in all_doc_ids:
        std_rank = standard_rank_map.get(doc_id, 0)
        rrk_rank = rerank_rank_map.get(doc_id, 0)
        rank_change = std_rank - rrk_rank if std_rank > 0 and rrk_rank > 0 else 0
        comparison.append(
            BenchmarkResult(
                doc_id=doc_id,
                title=title_map.get(doc_id, ""),
                doc_score=score_map.get(doc_id, 0.0),
                chunk_score=chunk_score_map.get(doc_id),
                standard_rank=std_rank,
                reranked_rank=rrk_rank,
                rank_change=rank_change,
            )
        )

    # Sort by standard rank (unranked docs last).
    comparison.sort(key=lambda b: b.standard_rank if b.standard_rank > 0 else 999)

    return BenchmarkResponse(
        query=q,
        limit=limit,
        standard_results=standard_results,
        reranked_results=rerank_results,
        comparison=comparison,
    )


# ── Agent query endpoint ───────────────────────────────────────────────────────

from pydantic import BaseModel as _BaseModel


class AgentQueryRequest(_BaseModel):
    query: str
    model: str = ""
    max_hops: int = 5


class AgentToolCall(_BaseModel):
    tool: str
    args: dict
    response_preview: str
    # For search calls: full list of candidates with scores
    search_results: list[dict] | None = None
    # Whether this doc was read (for read calls)
    was_read: bool | None = None


class AgentQueryResponse(_BaseModel):
    query_id: str | None = None
    query: str
    answer: str
    searches: int  # number of dewie_search calls (actual hops)
    reads: int  # number of dewie_read calls
    tool_calls: list[AgentToolCall]
    docs_read: list[dict]


_AGENT_SYSTEM = (
    "You are a research assistant with access to a private document corpus. "
    "Your job: answer the user's question using information found in the corpus.\n\n"
    "RULES:\n"
    "1. Always call dewie_search first with the user's exact query.\n"
    "2. Call dewie_read on the most relevant doc_id to get its full metadata and URL.\n"
    "3. Base your answer on the document's summary and metadata. Cite the source URL.\n"
    "4. If the corpus doesn't contain the answer, say so — do not use your training data.\n"
    "5. Do NOT modify the query — no added years, no extra specificity unless the user stated it.\n\n"
    "If you cannot call tools natively, emit:\n"
    'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "exact user query"}}\n'
    "Then wait for results, call dewie_read, then answer with the source URL cited."
)

_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "dewie_search",
            "description": "Search the document corpus. Returns documents with topics, keywords, entities, and answers_questions. Use the category parameter to narrow results: 'sports', 'finance', 'tech', 'news', 'science'. Always set category when the query is domain-specific.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                    "category": {
                        "type": "string",
                        "enum": ["sports", "finance", "tech", "news", "science"],
                        "description": "Optional domain filter. Set this when the query is clearly domain-specific.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dewie_expand",
            "description": "Get graph neighbors for a document. Returns related docs by edge weight.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dewie_read",
            "description": "Get full metadata and URI for a document. Returns title, url, source, summary, topics, entities, and document_type. Use the url field to reference or cite the document in your answer.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
            },
        },
    },
]


@router.post(
    "/agent",
    response_model=AgentQueryResponse,
    summary="Run a query through the agent loop (multi-hop search + LLM synthesis).",
)
@limiter.limit(rate_limit())
async def agent_query(request: Request, body: AgentQueryRequest) -> AgentQueryResponse:
    """
    Multi-hop agent query: LLM uses dewie_search, dewie_read, and
    dewie_expand tools to gather evidence, then synthesises a final answer.
    """
    import json as _json
    import re as _re

    pg = _get_pg(request)
    workspace_ids = getattr(request.state, "workspace_ids", [])

    # Build per-request API key header from tenant if available
    messages: list[dict] = [
        {"role": "system", "content": _AGENT_SYSTEM},
        {"role": "user", "content": body.query},
    ]
    tool_calls_list: list[dict] = []
    docs_read_list: list[dict] = []
    docs_considered: dict[str, dict] = {}
    answer: str | None = None

    try:
        from dewie.model_adapter import ModelClient
    except ImportError:
        return AgentQueryResponse(
            query=body.query,
            answer="Agent loop unavailable: model adapter not found.",
            searches=0,
            reads=0,
            tool_calls=[],
            docs_read=[],
        )

    async with ModelClient(model=body.model) as llm:
        for _step in range(body.max_hops):
            try:
                response = await llm.complete(messages, tools=_AGENT_TOOLS, max_tokens=1000)
            except Exception as e:
                answer = f"LLM error: {e}"
                break

            if not response.has_tool_calls:
                answer = response.content or "No answer."
                break

            messages.append(llm.assistant_message(response))

            for tc in response.tool_calls:
                fn = tc.name
                args = tc.arguments
                tool_resp = ""

                if fn == "dewie_search":
                    q = args.get("query", "")
                    limit = int(args.get("limit", 5))
                    agent_category = args.get("category", None)
                    # Resolve category to source filter
                    _CAT_SOURCES = {
                        "sports": [
                            "sports.yahoo.com",
                            "www.espn.com",
                            "www.sportingnews.com",
                            "www.cbssports.com",
                            "www.nba.com",
                            "theathletic.com",
                        ],
                        "finance": [
                            "finance.yahoo.com",
                            "www.fool.com",
                            "www.wsj.com",
                            "www.bloomberg.com",
                            "www.cnbc.com",
                            "www.sec.gov",
                            "seekingalpha.com",
                        ],
                        "tech": [
                            "techcrunch.com",
                            "www.theverge.com",
                            "arstechnica.com",
                            "www.wired.com",
                            "venturebeat.com",
                            "www.ycombinator.com",
                            "arxiv.org",
                        ],
                        "news": [
                            "www.reuters.com",
                            "apnews.com",
                            "www.bbc.com",
                            "www.theguardian.com",
                            "www.cbsnews.com",
                            "abcnews.com",
                            "www.al-monitor.com",
                        ],
                        "science": [
                            "arxiv.org",
                            "www.nature.com",
                            "www.ncbi.nlm.nih.gov",
                            "pubmed.ncbi.nlm.nih.gov",
                        ],
                    }
                    agent_source_filter = (
                        _CAT_SOURCES.get(agent_category.lower()) if agent_category else None
                    )
                    try:
                        raw = await pg.search(
                            q,
                            limit=limit,
                            ranker="answers_questions_rrf",
                            workspace_ids=workspace_ids,
                            source_filter=agent_source_filter,
                        )
                        results = []
                        for doc, score in raw:
                            did = str(doc.id)
                            entry = {
                                "doc_id": did,
                                "title": doc.title or "",
                                "url": doc.url or "",
                                "source": doc.source or "",
                                "summary": (doc.summary or ""),
                                "topics": doc.topics or [],
                                "score": round(score, 4),
                            }
                            if did not in docs_considered:
                                docs_considered[did] = {
                                    "doc_id": did,
                                    "title": doc.title or "",
                                    "score": score,
                                }
                            results.append(entry)
                        tool_resp = _json.dumps(results)
                        tool_calls_list.append(
                            {
                                "tool": fn,
                                "args": args,
                                "response": tool_resp,
                                "search_results": results,
                                "was_read": None,
                            }
                        )
                    except Exception as e:
                        tool_resp = _json.dumps({"error": str(e)})
                        tool_calls_list.append(
                            {
                                "tool": fn,
                                "args": args,
                                "response": tool_resp,
                                "search_results": [],
                                "was_read": None,
                            }
                        )

                elif fn == "dewie_expand":
                    doc_id = args.get("doc_id", "")
                    try:
                        from sqlalchemy import text as _text

                        async with pg._engine.connect() as conn:
                            rows = await conn.execute(
                                _text("""
                                    SELECT d.id::text, d.title, d.source, de.weight
                                    FROM document_edges de
                                    JOIN documents d ON d.id = de.target_id
                                    WHERE de.source_id = CAST(:doc_id AS UUID)
                                    ORDER BY de.weight DESC LIMIT 8
                                """),
                                {"doc_id": doc_id},
                            )
                            neighbors = [
                                {
                                    "doc_id": r[0],
                                    "title": r[1],
                                    "source": r[2],
                                    "weight": float(r[3]),
                                }
                                for r in rows
                            ]
                        tool_resp = _json.dumps(neighbors)
                    except Exception:
                        tool_resp = _json.dumps([])
                    tool_calls_list.append(
                        {
                            "tool": fn,
                            "args": args,
                            "response": tool_resp,
                            "search_results": None,
                            "was_read": None,
                        }
                    )

                elif fn == "dewie_read":
                    # Returns structured metadata + URI — we are a retrieval engine, not a content store.
                    # The caller already has the content or can fetch it; we surface the reference.
                    doc_id = args.get("doc_id", "")
                    read_success = False
                    try:
                        from sqlalchemy import text as _text

                        async with pg._engine.connect() as _conn:
                            _row = await _conn.execute(
                                _text("""
                                    SELECT id::text, title, url, source, summary,
                                           topics, keywords, entities, document_type,
                                           ingested_at::text, enrichment_quality_score
                                    FROM documents WHERE id = CAST(:id AS UUID)
                                """),
                                {"id": doc_id},
                            )
                            _doc = _row.fetchone()
                        if _doc:
                            doc_meta = {
                                "doc_id": _doc[0],
                                "title": _doc[1] or "",
                                "url": _doc[2] or "",
                                "source": _doc[3] or "",
                                "summary": _doc[4] or "",
                                "topics": list(_doc[5] or []),
                                "keywords": list(_doc[6] or []),
                                "entities": list(_doc[7] or []),
                                "document_type": _doc[8] or "",
                                "ingested_at": _doc[9] or "",
                                "quality_score": _doc[10],
                            }
                            docs_read_list.append(
                                {
                                    "doc_id": doc_id,
                                    "title": doc_meta["title"],
                                    "url": doc_meta["url"],
                                }
                            )
                            tool_resp = _json.dumps(doc_meta)
                            read_success = True
                        else:
                            tool_resp = _json.dumps(
                                {"error": "Document not found", "doc_id": doc_id}
                            )
                    except Exception as e:
                        tool_resp = _json.dumps({"error": str(e), "doc_id": doc_id})
                    tool_calls_list.append(
                        {
                            "tool": fn,
                            "args": args,
                            "response": tool_resp,
                            "search_results": None,
                            "was_read": read_success,
                        }
                    )
                else:
                    tool_resp = "unknown tool"
                    tool_calls_list.append(
                        {
                            "tool": fn,
                            "args": args,
                            "response": tool_resp,
                            "search_results": None,
                            "was_read": None,
                        }
                    )

                messages.append(llm.tool_response(tc.id, tool_resp, tc.name))

    if answer is None:
        answer = "Agent did not produce an answer within the hop limit."

    answer = _re.sub(r"TOOL_CALL:\s*\{.*?\}[\n\r]*", "", answer, flags=_re.DOTALL).strip()
    answer = _re.sub(
        r"<\|tool_call\|?>call:\w+[\(\{].*?[\)\}][\n\r]*", "", answer, flags=_re.DOTALL
    ).strip()
    if not answer:
        answer = "Agent did not produce an answer."

    # Log to query_log and get back the ID for tracking
    query_id: str | None = None
    try:
        from dewie.storage.query_logger import QueryLogEntry
        from dewie.storage.query_logger import log_query as _log_query

        entry = QueryLogEntry(
            question=body.query,
            model=body.model,
            source="agent",
            user_id=getattr(request.state, "user_id", None),
            hops=sum(1 for t in tool_calls_list if t["tool"] == "dewie_search"),
            hop_trace=tool_calls_list,
            docs_returned=docs_read_list,
            answer=answer,
        )
        await _log_query(entry)
        # Fetch the most recent log ID for this user/query
        async with pg._engine.connect() as _conn:
            from sqlalchemy import text as _text2

            _row = await _conn.execute(
                _text2("""
                SELECT id FROM query_log
                WHERE question = :q AND user_id = :uid
                ORDER BY ts DESC LIMIT 1
            """),
                {"q": body.query, "uid": getattr(request.state, "user_id", None)},
            )
            _r = _row.fetchone()
            if _r:
                query_id = str(_r[0])
    except Exception:
        pass

    return AgentQueryResponse(
        query_id=query_id,
        query=body.query,
        answer=answer,
        searches=sum(1 for t in tool_calls_list if t["tool"] == "dewie_search"),
        reads=sum(1 for t in tool_calls_list if t["tool"] == "dewie_read"),
        tool_calls=[
            AgentToolCall(
                tool=t["tool"],
                args=t["args"],
                response_preview=t["response"],
                search_results=t.get("search_results"),
                was_read=t.get("was_read"),
            )
            for t in tool_calls_list
        ],
        docs_read=docs_read_list,
    )


# ── Blind query endpoint ───────────────────────────────────────────────────────


class BlindQueryRequest(_BaseModel):
    query: str
    model: str = ""


class BlindQueryResponse(_BaseModel):
    query: str
    answer: str
    model: str


@router.post(
    "/blind",
    response_model=BlindQueryResponse,
    summary="Ask the LLM directly with no corpus access (blind condition for comparison).",
)
@limiter.limit(rate_limit())
async def blind_query(request: Request, body: BlindQueryRequest) -> BlindQueryResponse:
    """
    Direct LLM call with no tools and no retrieval. Used for blind vs Dewie comparison.
    """

    try:
        from dewie.model_adapter import ModelClient

        async with ModelClient(model=body.model) as llm:
            response = await llm.complete(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer the question directly and concisely using only your training knowledge. Do not use any tools.",
                    },
                    {"role": "user", "content": body.query},
                ],
                max_tokens=500,
            )
        answer = response.content or "No answer."
    except Exception as e:
        answer = f"Error: {e}"

    return BlindQueryResponse(query=body.query, answer=answer, model=body.model)


# ── Category query endpoint ────────────────────────────────────────────────────


@router.post(
    "/category",
    response_model=CategoryQueryResponse,
    summary="Search with category distribution hints. Shows which domains have relevant content and suggests narrowing when one category dominates.",
)
@limiter.limit(rate_limit())
async def category_query(
    request: Request,
    body: SearchRequest,
    rerank: bool = Query(default=False),
) -> CategoryQueryResponse:
    """
    Same as POST /query but also returns category distribution hints.
    When one category dominates results and no category filter was specified,
    returns a suggestion to narrow the search.
    """
    pg = _get_pg(request)
    cache = _get_cache(request)

    t0 = _time.time()
    ranker = getattr(body, "ranker", "answers_questions_rrf") or "answers_questions_rrf"
    category = getattr(body, "category", None)

    # Category → source mapping
    _CAT_SOURCES: dict[str, list[str]] = {
        "sports": [
            "sports.yahoo.com",
            "www.espn.com",
            "www.sportingnews.com",
            "www.cbssports.com",
            "www.nba.com",
            "theathletic.com",
        ],
        "finance": [
            "finance.yahoo.com",
            "www.fool.com",
            "www.wsj.com",
            "www.bloomberg.com",
            "www.cnbc.com",
            "www.sec.gov",
            "seekingalpha.com",
        ],
        "tech": [
            "techcrunch.com",
            "www.theverge.com",
            "arstechnica.com",
            "www.wired.com",
            "venturebeat.com",
            "www.ycombinator.com",
            "arxiv.org",
        ],
        "news": [
            "www.reuters.com",
            "apnews.com",
            "www.bbc.com",
            "www.theguardian.com",
            "www.cbsnews.com",
            "abcnews.com",
            "www.al-monitor.com",
            "thehill.com",
        ],
        "science": [
            "arxiv.org",
            "www.nature.com",
            "www.ncbi.nlm.nih.gov",
            "pubmed.ncbi.nlm.nih.gov",
        ],
    }

    source_filter = _CAT_SOURCES.get(category.lower()) if category else None
    search_limit = 20 if rerank else body.limit

    raw_results = await pg.search(
        body.query,
        limit=search_limit,
        ranker=ranker,
        workspace_ids=getattr(request.state, "workspace_ids", []),
        source_filter=source_filter,
    )

    from dewie.storage.rankers import _edge_counts

    doc_ids = [str(doc.id) for doc, _ in raw_results]
    async with pg._session_factory() as _session:
        _edge_count_map: dict[str, int] = dict(await _edge_counts(_session, doc_ids))

    results: list[SearchResult] = []
    for doc, score in raw_results:
        edge_count = _edge_count_map.get(str(doc.id), 0)
        results.append(
            SearchResult(
                doc_type=doc.document_type.value if doc.document_type else None,
                doc_id=str(doc.id),
                title=doc.title,
                summary=doc.summary or None,
                url=doc.url or None,
                source=doc.source or None,
                topics=doc.topics,
                keywords=_diverse_keywords(doc.keywords)
                if edge_count >= _KW_MIN_EDGE_COUNT
                else [],
                entities=doc.entities,
                sentiment=doc.sentiment,
                answers_questions=doc.answers_questions,
                score=score,
                edge_count=edge_count,
                ingested_at=doc.ingested_at,
                enrichment_quality_score=doc.enrichment_quality_score,
                reading_level=doc.reading_level.value if doc.reading_level else None,
            )
        )

    # Normalize scores
    if results:
        _max = results[0].score
        if _max > 1e-9:
            for r in results:
                r.score = round(r.score / _max, 4)

    if rerank and results:
        results = results[: body.limit]

    confidence = _compute_confidence(body.query, results)

    # ── Category distribution ─────────────────────────────────────────────────
    # Get corpus counts per category (cached 1 hour)
    async def _get_corpus_count(cat: str, sources: list[str]) -> int:
        cache_key = f"dewie:cat_counts:{cat}"
        cached = await cache._redis.get(cache_key)
        if cached:
            return int(cached)
        from sqlalchemy import text as _t

        async with pg._engine.connect() as conn:
            row = await conn.execute(
                _t("SELECT COUNT(*) FROM documents WHERE status='ready' AND source = ANY(:s)"),
                {"s": sources},
            )
            n = row.scalar() or 0
        await cache._redis.setex(cache_key, 3600, str(n))
        return n

    # Count results per category by source
    result_cat_counts: dict[str, int] = {}
    source_to_cat: dict[str, str] = {}
    for cat, sources in _CAT_SOURCES.items():
        for s in sources:
            source_to_cat[s] = cat

    for r in results:
        if r.source:
            cat = source_to_cat.get(r.source)
            if cat:
                result_cat_counts[cat] = result_cat_counts.get(cat, 0) + 1

    category_hints: dict[str, CategoryHint] = {}
    dominant_cat: str | None = None
    dominant_count = 0

    for cat, rc in result_cat_counts.items():
        corpus_n = await _get_corpus_count(cat, _CAT_SOURCES[cat])
        category_hints[cat] = CategoryHint(
            result_count=rc,
            corpus_count=corpus_n,
            suggested=False,
        )
        if rc > dominant_count:
            dominant_count = rc
            dominant_cat = cat

    # Build suggestion: only when no category specified AND one dominates
    category_suggestion: str | None = None
    if not category and dominant_cat and len(results) >= 5 and dominant_count / len(results) > 0.5:
        hint = category_hints[dominant_cat]
        hint.suggested = True
        category_suggestion = (
            f"Try narrowing to '{dominant_cat}' — {dominant_count} of {len(results)} "
            f"results are from {dominant_cat} sources "
            f"({hint.corpus_count:,} {dominant_cat} docs available)"
        )

    # Log query
    query_id: str | None = None
    try:
        from dewie.storage.query_logger import QueryLogEntry
        from dewie.storage.query_logger import log_query as _log_query

        docs_log = [{"doc_id": r.doc_id, "title": r.title} for r in results]
        await _log_query(
            QueryLogEntry(
                question=body.query,
                source="api",
                user_id=getattr(request.state, "user_id", None),
                hops=0,
                docs_returned=docs_log,
                elapsed_ms=int((_time.time() - t0) * 1000),
            )
        )
        async with pg._engine.connect() as _conn:
            from sqlalchemy import text as _qid_t

            _r = (
                await _conn.execute(
                    _qid_t(
                        "SELECT id FROM query_log WHERE question=:q AND user_id=:uid ORDER BY ts DESC LIMIT 1"
                    ),
                    {"q": body.query, "uid": getattr(request.state, "user_id", None)},
                )
            ).fetchone()
            if _r:
                query_id = str(_r[0])
    except Exception:
        pass

    return CategoryQueryResponse(
        query_id=query_id,
        query=body.query,
        results=results,
        total=len(results),
        result_confidence=confidence,
        fallback_triggered=bool(confidence and confidence.gap_signal),
        gap_enrichment_queued=False,
        category_hints=category_hints,
        category_suggestion=category_suggestion,
    )
