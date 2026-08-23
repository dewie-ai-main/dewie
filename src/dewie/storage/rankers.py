# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Pluggable ranking strategies for Dewie search.

Each ranker is a callable:
  async def rank(query: str, session: AsyncSession, embedding: list[float] | None, limit: int)
      -> list[tuple[str, float]]   # [(doc_id, score), ...]

Scores must be in [0, 1] range where higher = more relevant.
"""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable
from datetime import UTC

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


def _session_is_sqlite(session: AsyncSession) -> bool:
    """Return True if the session is backed by SQLite."""
    try:
        url = str(session.get_bind().engine.url)  # type: ignore[union-attr]
        return url.startswith("sqlite")
    except Exception:
        try:
            url = str(session.bind.engine.url)  # type: ignore[union-attr]
            return url.startswith("sqlite")
        except Exception:
            return False


def _sqlite_in_clause(ids: list[str], prefix: str = "id") -> tuple[str, dict[str, str]]:
    """Build an IN clause with named params for SQLite (no ANY/uuid[] support)."""
    params = {f"{prefix}{i}": v for i, v in enumerate(ids)}
    placeholders = ", ".join(f":{prefix}{i}" for i in range(len(ids)))
    return placeholders, params

RankerFn = Callable[..., Awaitable[list[tuple[str, float]]]]

RANKER_REGISTRY: dict[str, dict] = {}


def ranker(name: str, label: str, description: str):
    """Decorator to register a ranker function."""

    def decorator(fn: RankerFn) -> RankerFn:
        RANKER_REGISTRY[name] = {"fn": fn, "label": label, "description": description}
        return fn

    return decorator


# ── Helpers ───────────────────────────────────────────────────────────────────


def _normalize(scores: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return scores
    mn = min(s for _, s in scores)
    mx = max(s for _, s in scores)
    if mx == mn:
        return [(d, 1.0) for d, _ in scores]
    return [(d, (s - mn) / (mx - mn)) for d, s in scores]


async def _fts_sqlite(session: AsyncSession, query: str, limit: int) -> list[tuple[str, float]]:
    """Term-match scoring for SQLite: title hits weigh 2x body hits.

    Not BM25 — a deliberate floor so single-node SQLite installs get working
    retrieval instead of the silent empty results the Postgres SQL produced.
    """
    words = [w.lower().strip("\"'.,;:!?()") for w in query.split()]
    words = [w for w in words if len(w) > 2][:8]
    if not words:
        return []

    score_terms = []
    where_terms = []
    params: dict[str, object] = {"limit": limit}
    for i, w in enumerate(words):
        params[f"w{i}"] = f"%{w}%"
        score_terms.append(
            f"(CASE WHEN lower(title) LIKE :w{i} THEN 2 ELSE 0 END"
            f" + CASE WHEN lower(COALESCE(body_text, summary, '')) LIKE :w{i} THEN 1 ELSE 0 END)"
        )
        where_terms.append(
            f"(lower(title) LIKE :w{i} OR lower(COALESCE(body_text, summary, '')) LIKE :w{i})"
        )

    sql = text(
        f"SELECT id, ({' + '.join(score_terms)}) AS rank FROM documents "
        f"WHERE status = 'ready' AND ({' OR '.join(where_terms)}) "
        "ORDER BY rank DESC LIMIT :limit"
    )
    rows = (await session.execute(sql, params)).fetchall()
    return [(str(r[0]), float(r[1])) for r in rows]


async def _fts(session: AsyncSession, query: str, limit: int) -> list[tuple[str, float]]:
    """BM25 full-text search via Postgres ts_rank_cd (term-match scoring on SQLite)."""
    if _session_is_sqlite(session):
        try:
            return await _fts_sqlite(session, query, limit)
        except Exception:
            return []

    rows: list[tuple[str, float]] = []
    try:
        sql = text("""
            SELECT id, ts_rank_cd(search_vec, websearch_to_tsquery('english', :q)) AS rank
            FROM documents
            WHERE status = 'ready' AND search_vec @@ websearch_to_tsquery('english', :q)
            ORDER BY rank DESC LIMIT :limit
        """)
        r = (await session.execute(sql, {"q": query, "limit": limit})).fetchall()
        rows = [(str(x[0]), float(x[1])) for x in r]
    except Exception:
        # A failed statement aborts the whole transaction; roll back so
        # subsequent queries on this session don't see
        # InFailedSQLTransactionError.
        await session.rollback()

    # OR fallback if few results
    if len(rows) < 3:
        # to_tsquery (unlike websearch_to_tsquery) parses its input as tsquery
        # syntax — strip punctuation so quotes/apostrophes can't break it.
        words = [w for w in (re.sub(r"[^A-Za-z0-9]", "", w) for w in query.split()) if len(w) > 3]
        if words:
            try:
                sql2 = text("""
                    SELECT id, ts_rank_cd(search_vec, to_tsquery('english', :q)) AS rank
                    FROM documents
                    WHERE status = 'ready' AND search_vec @@ to_tsquery('english', :q)
                    ORDER BY rank DESC LIMIT :limit
                """)
                or_q = " | ".join(words)
                r2 = (await session.execute(sql2, {"q": or_q, "limit": limit})).fetchall()
                seen = {x[0] for x in rows}
                rows += [(str(x[0]), float(x[1])) for x in r2 if x[0] not in seen]
            except Exception:
                await session.rollback()
    return rows


async def _vec(
    session: AsyncSession, embedding: list[float] | None, limit: int
) -> list[tuple[str, float]]:
    """Cosine similarity vector search."""
    if not embedding:
        return []
    sql = text("""
        SELECT id, 1 - (embedding <=> cast(:vec as vector)) AS score
        FROM documents
        WHERE status = 'ready' AND embedding IS NOT NULL
        ORDER BY embedding <=> cast(:vec as vector)
        LIMIT :limit
    """)
    rows = (await session.execute(sql, {"vec": str(embedding), "limit": limit})).fetchall()
    return [(str(r[0]), float(r[1])) for r in rows]


async def _aq_vec(
    session: AsyncSession, embedding: list[float] | None, limit: int
) -> list[tuple[str, float]]:
    """
    Per-AQ vector search over the document_aq table.

    For each query, find the `scan_limit` closest AQ embeddings, then
    collapse to one score per document (best-matching AQ wins).

    This answers: "which document has an AQ string that DIRECTLY asks
    what this query is asking?" — different from full-doc vector search,
    which asks "which document is most semantically similar overall?"

    Falls back gracefully if document_aq table doesn't exist yet.
    """
    if not embedding:
        return []
    scan_limit = limit * 8
    try:
        sql = text("""
            SELECT doc_id::text, MAX(score) AS best_score
            FROM (
                SELECT doc_id, 1 - (embedding <=> cast(:vec as vector)) AS score
                FROM document_aq
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> cast(:vec as vector)
                LIMIT :scan_limit
            ) candidates
            GROUP BY doc_id
            ORDER BY best_score DESC
            LIMIT :limit
        """)
        rows = (
            await session.execute(
                sql, {"vec": str(embedding), "scan_limit": scan_limit, "limit": limit}
            )
        ).fetchall()
        return [(str(r[0]), float(r[1])) for r in rows]
    except Exception:
        return []


async def _chunk_vec(
    session: AsyncSession, embedding: list[float] | None, limit: int
) -> list[tuple[str, float]]:
    """
    Vector search over document_chunks, collapsed to parent doc.

    Two-step: HNSW scan for top-N chunks, then group by parent doc taking best score.
    Uses aq_embedding when available, falls back to body embedding.
    Returns (doc_id, best_chunk_score) — one entry per parent doc, sorted by score desc.
    """
    if not embedding:
        return []
    scan_limit = limit * 20  # scan many chunks — multiple chunks per doc
    try:
        # Step 1: pure HNSW scan on aq_embedding (index-friendly — no JOIN, no filter)
        sql_aq = text("""
            SELECT doc_id::text,
                   1 - (aq_embedding <=> cast(:vec as vector)) AS score
            FROM document_chunks
            WHERE aq_embedding IS NOT NULL
            ORDER BY aq_embedding <=> cast(:vec as vector)
            LIMIT :scan_limit
        """)
        # Step 2: fallback scan on body embedding for chunks without aq_embedding
        sql_body = text("""
            SELECT doc_id::text,
                   1 - (embedding <=> cast(:vec as vector)) AS score
            FROM document_chunks
            WHERE aq_embedding IS NULL AND embedding IS NOT NULL
            ORDER BY embedding <=> cast(:vec as vector)
            LIMIT :scan_limit
        """)
        aq_rows = (
            await session.execute(sql_aq, {"vec": str(embedding), "scan_limit": scan_limit})
        ).fetchall()
        body_rows = (
            await session.execute(sql_body, {"vec": str(embedding), "scan_limit": scan_limit})
        ).fetchall()

        # Collapse both to best score per parent doc
        best: dict[str, float] = {}
        for doc_id, score in list(aq_rows) + list(body_rows):
            doc_id = str(doc_id)
            s = float(score)
            if doc_id not in best or s > best[doc_id]:
                best[doc_id] = s

        # Filter to ready docs only (lightweight — only on top candidates)
        if best:
            doc_ids = list(best.keys())[: limit * 4]
            placeholders = ", ".join(f":id{i}" for i in range(len(doc_ids)))
            ready_sql = text(
                f"SELECT id::text FROM documents WHERE id::text IN ({placeholders}) AND status = 'ready'"
            )
            ready_rows = (
                await session.execute(ready_sql, {f"id{i}": d for i, d in enumerate(doc_ids)})
            ).fetchall()
            ready_ids = {str(r[0]) for r in ready_rows}
            best = {k: v for k, v in best.items() if k in ready_ids}

        return sorted(best.items(), key=lambda x: -x[1])[:limit]
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning("_chunk_vec failed: %s", e)
        return []


async def _aq_match(
    session: AsyncSession, query: str, embedding: list[float] | None, limit: int
) -> list[tuple[str, float]]:
    """
    Match query against per-doc answers_questions strings via BM25 FTS.
    Uses the stored aq_tsvec column (GIN-indexed, materialized at write time).
    For semantic/vector AQ search, use _aq_vec() which queries document_aq.
    """
    if not query:
        return []
    # Score based on how many AQ strings contain query terms (via text search).
    # Uses aq_tsvec stored column + idx_documents_aq_tsvec_stored GIN index —
    # value is materialized at write time so no per-row recomputation here.
    sql = text("""
        SELECT id,
               ts_rank_cd(aq_tsvec, websearch_to_tsquery('english', :q)) AS rank
        FROM documents
        WHERE status = 'ready'
          AND aq_tsvec IS NOT NULL
          AND aq_tsvec @@ websearch_to_tsquery('english', :q)
        ORDER BY rank DESC LIMIT :limit
    """)
    rows = (await session.execute(sql, {"q": query, "limit": limit})).fetchall()
    return [(str(r[0]), float(r[1])) for r in rows]


# ── Registered rankers ────────────────────────────────────────────────────────


@ranker("bm25", "BM25 Only", "Pure Postgres full-text search (ts_rank_cd). No vectors.")
async def rank_bm25(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    rows = await _fts(session, query, limit)
    return _normalize(rows)


@ranker("vector", "Vector Only", "Pure cosine similarity on text-embedding-3-small embeddings.")
async def rank_vector(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    rows = await _vec(session, embedding, limit)
    return _normalize(rows)


@ranker(
    "rrf",
    "RRF (default)",
    "Hybrid semantic + keyword, recommended (k=60). Current default.",
)
async def rank_rrf(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    K = 60
    fts = await _fts(session, query, limit)
    vec = await _vec(session, embedding, limit)
    aq = await _aq_vec(session, embedding, limit)
    chunks = await _chunk_vec(session, embedding, limit)
    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(fts):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(vec):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(aq):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(chunks):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
    top = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    return [(d, scores[d]) for d in top]



@ranker("rrf_k10", "RRF k=10", "RRF with lower k — amplifies rank differences more aggressively.")
async def rank_rrf_k10(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    K = 10
    fts = await _fts(session, query, limit)
    vec = await _vec(session, embedding, limit)
    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(fts):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(vec):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
    top = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    return [(d, scores[d]) for d in top]


@ranker("linear_blend", "Linear Blend α=0.7", "0.7 × normalized_vector + 0.3 × normalized_BM25.")
async def rank_linear_blend(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    alpha = 0.7  # weight for semantic
    fts_raw = await _fts(session, query, limit * 2)
    vec_raw = await _vec(session, embedding, limit * 2)
    fts_norm = dict(_normalize(fts_raw))
    vec_norm = dict(_normalize(vec_raw))
    all_ids = set(fts_norm) | set(vec_norm)
    scores = {
        doc_id: alpha * vec_norm.get(doc_id, 0.0) + (1 - alpha) * fts_norm.get(doc_id, 0.0)
        for doc_id in all_ids
    }
    top = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    return [(d, scores[d]) for d in top]


@ranker("linear_blend_50", "Linear Blend α=0.5", "Equal weight: 0.5 × vector + 0.5 × BM25.")
async def rank_linear_blend_50(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    alpha = 0.5
    fts_raw = await _fts(session, query, limit * 2)
    vec_raw = await _vec(session, embedding, limit * 2)
    fts_norm = dict(_normalize(fts_raw))
    vec_norm = dict(_normalize(vec_raw))
    all_ids = set(fts_norm) | set(vec_norm)
    scores = {
        doc_id: alpha * vec_norm.get(doc_id, 0.0) + (1 - alpha) * fts_norm.get(doc_id, 0.0)
        for doc_id in all_ids
    }
    top = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    return [(d, scores[d]) for d in top]


@ranker(
    "aq_boosted",
    "AQ-Boosted BM25",
    "BM25 + bonus for docs whose answers_questions field matches the query.",
)
async def rank_aq_boosted(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    fts_raw = await _fts(session, query, limit * 2)
    fts_norm = dict(_normalize(fts_raw))

    # AQ match bonus — uses stored aq_tsvec column
    try:
        sql = text("""
            SELECT id,
                   ts_rank_cd(aq_tsvec, websearch_to_tsquery('english', :q)) AS aq_rank
            FROM documents
            WHERE status = 'ready'
              AND aq_tsvec IS NOT NULL
              AND aq_tsvec @@ websearch_to_tsquery('english', :q)
            ORDER BY aq_rank DESC LIMIT :limit
        """)
        aq_rows = (await session.execute(sql, {"q": query, "limit": limit})).fetchall()
        aq_scores = dict(_normalize([(str(r[0]), float(r[1])) for r in aq_rows]))
    except Exception:
        aq_scores = {}

    all_ids = set(fts_norm) | set(aq_scores)
    scores = {
        doc_id: 0.6 * fts_norm.get(doc_id, 0.0) + 0.4 * aq_scores.get(doc_id, 0.0)
        for doc_id in all_ids
    }
    top = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    return [(d, scores[d]) for d in top]


@ranker("vec_aq_boosted", "Vector + AQ Boost", "Vector similarity + AQ field match bonus.")
async def rank_vec_aq_boosted(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    vec_raw = await _vec(session, embedding, limit * 2)
    vec_norm = dict(_normalize(vec_raw))

    try:
        sql = text("""
            SELECT id,
                   ts_rank_cd(aq_tsvec, websearch_to_tsquery('english', :q)) AS aq_rank
            FROM documents
            WHERE status = 'ready'
              AND aq_tsvec IS NOT NULL
              AND aq_tsvec @@ websearch_to_tsquery('english', :q)
            ORDER BY aq_rank DESC LIMIT :limit
        """)
        aq_rows = (await session.execute(sql, {"q": query, "limit": limit})).fetchall()
        aq_scores = dict(_normalize([(str(r[0]), float(r[1])) for r in aq_rows]))
    except Exception:
        aq_scores = {}

    all_ids = set(vec_norm) | set(aq_scores)
    scores = {
        doc_id: 0.7 * vec_norm.get(doc_id, 0.0) + 0.3 * aq_scores.get(doc_id, 0.0)
        for doc_id in all_ids
    }
    top = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    return [(d, scores[d]) for d in top]


# ── Helpers: graph density + recency ──────────────────────────────────────────


async def _edge_counts(session: AsyncSession, doc_ids: list[str]) -> dict[str, int]:
    """Return {doc_id: total_degree} for a list of doc ids (inbound + outbound).

    Counts edges in both directions so that highly-cited authoritative hubs
    score at least as well as documents that merely point to many others.
    """
    if not doc_ids:
        return {}
    if _session_is_sqlite(session):
        placeholders, params = _sqlite_in_clause(doc_ids, prefix="id")
        sql = text(f"""
            SELECT node_id, COUNT(*) AS cnt FROM (
                SELECT source_id AS node_id FROM document_edges
                WHERE source_id IN ({placeholders})
                UNION ALL
                SELECT target_id AS node_id FROM document_edges
                WHERE target_id IN ({placeholders})
            ) edges
            GROUP BY node_id
        """)
        rows = (await session.execute(sql, params)).fetchall()
        return {str(r[0]): int(r[1]) for r in rows}
    sql = text("""
        SELECT node_id, COUNT(*) AS cnt FROM (
            SELECT source_id::text AS node_id FROM document_edges
            WHERE source_id = ANY(cast(:ids AS uuid[]))
            UNION ALL
            SELECT target_id::text AS node_id FROM document_edges
            WHERE target_id = ANY(cast(:ids AS uuid[]))
        ) edges
        GROUP BY node_id
    """)
    rows = (await session.execute(sql, {"ids": doc_ids})).fetchall()
    return {str(r[0]): int(r[1]) for r in rows}


async def _ingested_at(session: AsyncSession, doc_ids: list[str]) -> dict[str, float]:
    """Return {doc_id: recency_score} using exp decay with half-life 30 days.

    Uses published_at when available (actual article date), falls back to
    ingested_at (when we crawled it). This ensures news from yesterday ranks
    above news from last week even if we ingested both today.
    """
    if not doc_ids:
        return {}
    if _session_is_sqlite(session):
        placeholders, params = _sqlite_in_clause(doc_ids, prefix="id")
        sql = text(f"""
            SELECT id,
                   COALESCE(published_at, ingested_at) AS effective_date
            FROM documents
            WHERE id IN ({placeholders})
              AND COALESCE(published_at, ingested_at) IS NOT NULL
        """)
    else:
        sql = text("""
            SELECT id::text,
                   COALESCE(published_at, ingested_at) AS effective_date
            FROM documents
            WHERE id = ANY(cast(:ids AS uuid[]))
              AND COALESCE(published_at, ingested_at) IS NOT NULL
        """)
        params = {"ids": doc_ids}  # type: ignore[assignment]
    rows = (await session.execute(sql, params)).fetchall()
    from datetime import datetime

    now = datetime.now(UTC)
    result = {}
    for doc_id, effective_date in rows:
        age_days = max((now - effective_date).total_seconds() / 86400, 0)
        result[str(doc_id)] = math.exp(-age_days / 30)
    return result


async def _quality_scores(session: AsyncSession, doc_ids: list[str]) -> dict[str, float]:
    """Return {doc_id: quality_score_0_to_1} for a list of doc ids.

    Docs without an enrichment_quality_score (NULL) get a neutral 0.5
    so they are not penalised relative to unenriched docs.
    """
    if not doc_ids:
        return {}
    if _session_is_sqlite(session):
        placeholders, params = _sqlite_in_clause(doc_ids, prefix="id")
        sql = text(f"SELECT id, enrichment_quality_score FROM documents WHERE id IN ({placeholders})")
    else:
        sql = text("SELECT id::text, enrichment_quality_score FROM documents WHERE id = ANY(cast(:ids AS uuid[]))")
        params = {"ids": doc_ids}  # type: ignore[assignment]
    rows = (await session.execute(sql, params)).fetchall()
    return {str(r[0]): (r[1] / 100.0 if r[1] is not None else 0.5) for r in rows}


def _normalize_dict(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return d
    mn, mx = min(d.values()), max(d.values())
    if mx == mn:
        return {k: 1.0 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}


@ranker(
    "rrf_graph_boosted",
    "RRF + Graph Density",
    "RRF base + 0.15 × normalized edge count. Demotes isolated noise docs, promotes topically central ones.",
)
async def rank_rrf_graph_boosted(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    K = 60
    fts = await _fts(session, query, limit)
    vec = await _vec(session, embedding, limit)
    rrf: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(fts):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(vec):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (K + rank + 1)

    all_ids = list(rrf)
    edges = _normalize_dict(dict(await _edge_counts(session, all_ids)))

    scores = {doc_id: rrf[doc_id] + 0.15 * edges.get(doc_id, 0.0) for doc_id in all_ids}
    top = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    return [(d, scores[d]) for d in top]


@ranker(
    "rrf_recency",
    "RRF + Recency",
    "0.8 × RRF + 0.2 × exp(-age_days/30). Boosts recently ingested docs, decays old ones.",
)
async def rank_rrf_recency(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    K = 60
    fts = await _fts(session, query, limit)
    vec = await _vec(session, embedding, limit)
    rrf: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(fts):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(vec):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (K + rank + 1)

    all_ids = list(rrf)
    recency = _normalize_dict(await _ingested_at(session, all_ids))

    scores = {doc_id: 0.8 * rrf[doc_id] + 0.2 * recency.get(doc_id, 0.0) for doc_id in all_ids}
    top = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    return [(d, scores[d]) for d in top]


@ranker(
    "rrf_graph_recency",
    "RRF + Graph + Recency",
    "0.7 × RRF + 0.15 × edge density + 0.15 × recency decay. Combined signal for news/current-events queries.",
)
async def rank_rrf_graph_recency(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    K = 60
    fts = await _fts(session, query, limit)
    vec = await _vec(session, embedding, limit)
    rrf: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(fts):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(vec):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (K + rank + 1)

    all_ids = list(rrf)
    edges = await _edge_counts(session, all_ids)
    recency = await _ingested_at(session, all_ids)
    edges_norm = _normalize_dict(dict(edges))
    recency_norm = _normalize_dict(recency)

    scores = {
        doc_id: 0.7 * rrf[doc_id]
        + 0.15 * edges_norm.get(doc_id, 0.0)
        + 0.15 * recency_norm.get(doc_id, 0.0)
        for doc_id in all_ids
    }
    top = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    return [(d, scores[d]) for d in top]


@ranker(
    "rrf_quality",
    "RRF + Quality Score",
    "0.9 × RRF + 0.1 × enrichment_quality_score. Demotes low-quality enrichments; NULL scores treated as neutral (0.5).",
)
async def rank_rrf_quality(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    K = 60
    fts = await _fts(session, query, limit)
    vec = await _vec(session, embedding, limit)
    rrf: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(fts):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(vec):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (K + rank + 1)

    all_ids = list(rrf)
    quality = await _quality_scores(session, all_ids)

    scores = {doc_id: 0.9 * rrf[doc_id] + 0.1 * quality.get(doc_id, 0.5) for doc_id in all_ids}
    top = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    return [(d, scores[d]) for d in top]


@ranker(
    "answers_questions_rrf",
    "precision",
    "Three-way RRF: AQ field FTS + per-AQ vector search (document_aq) + body vector. "
    "Per-AQ vectors find docs whose individual AQ string directly answers the query, "
    "fixing the retrieval-miss failure mode on natural-language queries.",
)
async def rank_answers_questions_rrf(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    K = 60

    # Signal 1: AQ full-text search — uses stored aq_tsvec column
    aq_rows: list[tuple[str, float]] = []
    try:
        sql = text("""
            SELECT id::text,
                   ts_rank_cd(aq_tsvec, websearch_to_tsquery('english', :q)) AS rank
            FROM documents
            WHERE status = 'ready'
              AND aq_tsvec IS NOT NULL
              AND aq_tsvec @@ websearch_to_tsquery('english', :q)
            ORDER BY rank DESC LIMIT :limit
        """)
        rows = (await session.execute(sql, {"q": query, "limit": limit})).fetchall()
        aq_rows = [(str(r[0]), float(r[1])) for r in rows]
    except Exception:
        await session.rollback()

    # Fallback OR search if few AQ FTS hits
    if len(aq_rows) < 3:
        # to_tsquery parses tsquery syntax — strip punctuation from tokens
        # (see _fts) so quotes/apostrophes can't abort the transaction.
        words = [w for w in (re.sub(r"[^A-Za-z0-9]", "", w) for w in query.split()) if len(w) > 3]
        if words:
            try:
                sql2 = text("""
                    SELECT id::text,
                           ts_rank_cd(aq_tsvec, to_tsquery('english', :q)) AS rank
                    FROM documents
                    WHERE status = 'ready'
                      AND aq_tsvec IS NOT NULL
                      AND aq_tsvec @@ to_tsquery('english', :q)
                    ORDER BY rank DESC LIMIT :limit
                """)
                r2 = (
                    await session.execute(sql2, {"q": " | ".join(words), "limit": limit})
                ).fetchall()
                seen = {x[0] for x in aq_rows}
                aq_rows += [(str(r[0]), float(r[1])) for r in r2 if str(r[0]) not in seen]
            except Exception:
                await session.rollback()

    # Signal 2 + 3: per-AQ vector search and body vector
    aq_vec = await _aq_vec(session, embedding, limit)
    vec = await _vec(session, embedding, limit)

    # Three-way RRF fusion
    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(aq_rows):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(aq_vec):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(vec):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)

    top = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    return [(d, scores[d]) for d in top]


@ranker(
    "rrf_chunks",
    "RRF + Chunk Vectors",
    "Four-way RRF: BM25 + body vector + AQ vector + chunk vectors. "
    "Chunk vectors let long documents compete in retrieval via their content passages "
    "rather than their summary embedding alone. Best chunk score per doc is used.",
)
async def rank_rrf_chunks(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    K = 60

    fts_rows = await _fts(session, query, limit)
    vec_rows = await _vec(session, embedding, limit)
    aq_vec_rows = await _aq_vec(session, embedding, limit)
    chunk_rows = await _chunk_vec(session, embedding, limit)

    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(fts_rows):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(vec_rows):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(aq_vec_rows):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(chunk_rows):
        # Chunk signal weighted equal to other signals — a doc discovered only
        # via chunk search gets 1 vote, same as a doc found via BM25.
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)

    top = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    return [(d, scores[d]) for d in top]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def _fetch_embeddings(session: AsyncSession, doc_ids: list[str]) -> dict[str, list[float]]:
    """Fetch stored (truncated) embedding vectors for a list of doc IDs."""
    if not doc_ids:
        return {}
    if _session_is_sqlite(session):
        placeholders, params = _sqlite_in_clause(doc_ids, prefix="id")
        sql = text(f"SELECT id, embedding FROM documents WHERE id IN ({placeholders}) AND embedding IS NOT NULL")
    else:
        sql = text("""
            SELECT id::text, embedding::text
            FROM documents
            WHERE id = ANY(cast(:ids AS uuid[])) AND embedding IS NOT NULL
        """)
        params = {"ids": doc_ids}  # type: ignore[assignment]
    rows = (await session.execute(sql, params)).fetchall()
    result: dict[str, list[float]] = {}
    for doc_id, vec_str in rows:
        try:
            if vec_str:
                result[str(doc_id)] = [float(v) for v in str(vec_str).strip("[]").split(",")]
        except Exception:
            pass
    return result


async def _fetch_embeddings_full(session: AsyncSession, doc_ids: list[str]) -> dict[str, list[float]]:
    """Fetch stored full embedding vectors for a list of doc IDs."""
    if not doc_ids:
        return {}
    if _session_is_sqlite(session):
        placeholders, params = _sqlite_in_clause(doc_ids, prefix="id")
        sql = text(f"SELECT id, embedding_full FROM documents WHERE id IN ({placeholders}) AND embedding_full IS NOT NULL")
    else:
        sql = text("""
            SELECT id::text, embedding_full::text
            FROM documents
            WHERE id = ANY(cast(:ids AS uuid[])) AND embedding_full IS NOT NULL
        """)
        params = {"ids": doc_ids}  # type: ignore[assignment]
    rows = (await session.execute(sql, params)).fetchall()
    result: dict[str, list[float]] = {}
    for doc_id, vec_str in rows:
        try:
            if vec_str:
                result[str(doc_id)] = [float(v) for v in str(vec_str).strip("[]").split(",")]
        except Exception:
            pass
    return result


def _mmr_rerank(
    candidates: list[tuple[str, float]],
    embeddings: dict[str, list[float]],
    query_embedding: list[float] | None,
    limit: int,
    lam: float = 0.7,
) -> list[tuple[str, float]]:
    """
    Maximal Marginal Relevance re-ranking.

    lam=1.0 → pure relevance (same as input order)
    lam=0.0 → pure diversity
    lam=0.7 → recommended default (relevance-biased)

    Candidates without embeddings are appended at the end in original order.
    """
    # Split into those with/without embeddings
    with_emb = [(d, s) for d, s in candidates if d in embeddings]
    without_emb = [(d, s) for d, s in candidates if d not in embeddings]

    if not with_emb:
        return candidates[:limit]

    # Normalise relevance scores to [0,1]
    norm = _normalize(with_emb)
    rel = {d: s for d, s in norm}

    selected: list[str] = []

    while len(selected) < limit and with_emb:
        best_id = None
        best_score = -1.0

        for doc_id, _ in with_emb:
            relevance = rel[doc_id]
            if not selected:
                mmr_score = relevance
            else:
                max_sim = max(
                    _cosine(embeddings[doc_id], embeddings[s]) for s in selected if s in embeddings
                )
                mmr_score = lam * relevance - (1 - lam) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_id = doc_id

        if best_id is None:
            break

        selected.append(best_id)
        with_emb = [(d, s) for d, s in with_emb if d != best_id]

    # Re-attach scores (use original relevance for selected, preserve order)
    score_map = {d: s for d, s in candidates}
    result = [(d, score_map[d]) for d in selected]

    # Append any leftover (no embedding) docs if we still need to fill limit
    remaining = limit - len(result)
    if remaining > 0:
        result += without_emb[:remaining]

    return result


@ranker(
    "rrf_mmr",
    "RRF + MMR Diversity",
    "RRF retrieves top-30 candidates, then MMR re-ranks for relevance AND diversity. "
    "Prevents near-duplicate results (e.g. 5 identical EDGAR index pages).",
)
async def rank_rrf_mmr(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    K = 60
    CANDIDATE_N = max(limit * 3, 30)  # retrieve more, then diversity-filter down

    bm25 = await _fts(session, query, CANDIDATE_N)
    vec = await _vec(session, embedding, CANDIDATE_N)

    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(bm25):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(vec):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)

    candidates = sorted(scores, key=scores.__getitem__, reverse=True)[:CANDIDATE_N]
    candidates_scored = [(d, scores[d]) for d in candidates]

    # Fetch embeddings for MMR
    embeddings_map = await _fetch_embeddings(session, candidates)
    return _mmr_rerank(candidates_scored, embeddings_map, embedding, limit, lam=0.7)


# ── Two-stage RRF → cosine rerank ─────────────────────────────────────────────


@ranker(
    "rrf_rerank",
    "RRF + Cosine Rerank",
    "RRF retrieves top-30 candidates, then re-ranks by direct cosine similarity "
    "between query embedding and doc embedding. More precise than RRF alone.",
)
async def rank_rrf_rerank(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    """
    Two-stage retrieval:
      1. RRF (BM25 + vector) → top CANDIDATE_N for broad recall
      2. Cosine similarity against stored doc embeddings → precise final ranking

    Falls back to straight RRF if no query embedding available.
    """
    if not embedding:
        return await rank_rrf(query, session, embedding, limit)

    K = 60
    CANDIDATE_N = max(limit * 4, 40)

    bm25 = await _fts(session, query, CANDIDATE_N)
    vec = await _vec(session, embedding, CANDIDATE_N)

    rrf_scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(bm25):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(vec):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (K + rank + 1)

    candidates = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:CANDIDATE_N]

    # Re-rank by cosine similarity to query embedding
    embeddings_map = await _fetch_embeddings(session, candidates)

    reranked: list[tuple[str, float]] = []
    for doc_id in candidates:
        if doc_id in embeddings_map:
            cos = _cosine(embedding, embeddings_map[doc_id])
            reranked.append((doc_id, cos))
        else:
            # No stored embedding — keep with a score below any cosine score
            reranked.append((doc_id, -1.0))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:limit]


@ranker(
    "rrf_rerank_mmr",
    "RRF + Cosine Rerank + MMR",
    "Three-stage pipeline: RRF (broad recall) → cosine rerank (precision) → MMR (diversity). "
    "Best overall quality but slightly more compute.",
)
async def rank_rrf_rerank_mmr(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    """Three-stage: RRF → cosine rerank → MMR diversity filter."""
    if not embedding:
        return await rank_rrf_mmr(query, session, embedding, limit)

    K = 60
    CANDIDATE_N = max(limit * 5, 50)

    bm25 = await _fts(session, query, CANDIDATE_N)
    vec = await _vec(session, embedding, CANDIDATE_N)

    rrf_scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(bm25):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(vec):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (K + rank + 1)

    candidates = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:CANDIDATE_N]
    embeddings_map = await _fetch_embeddings(session, candidates)

    # Stage 2: cosine rerank
    reranked: list[tuple[str, float]] = []
    for doc_id in candidates:
        cos = _cosine(embedding, embeddings_map[doc_id]) if doc_id in embeddings_map else -1.0
        reranked.append((doc_id, cos))
    reranked.sort(key=lambda x: x[1], reverse=True)

    # Stage 3: MMR diversity
    mmr_pool = reranked[:CANDIDATE_N]
    return _mmr_rerank(mmr_pool, embeddings_map, embedding, limit, lam=0.7)


@ranker(
    "rrf_rerank_mmr_full",
    "RRF + Full-Precision Cosine Rerank + MMR",
    "Three-stage: RRF → cosine rerank (using full-precision embeddings) → MMR diversity filter. "
    "Highest potential quality, uses untruncated vectors for re-scoring.",
)
async def rank_rrf_rerank_mmr_full(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    """Three-stage: RRF → cosine rerank (full precision) → MMR diversity filter."""
    if not embedding:
        return await rank_rrf_mmr(query, session, embedding, limit)

    K = 60
    CANDIDATE_N = max(limit * 5, 50)

    bm25 = await _fts(session, query, CANDIDATE_N)
    vec = await _vec(session, embedding, CANDIDATE_N)

    rrf_scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(bm25):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
    for rank, (doc_id, _) in enumerate(vec):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (K + rank + 1)

    candidates = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:CANDIDATE_N]

    # Fetch both truncated and full embeddings
    truncated_embeddings_map = await _fetch_embeddings(session, candidates)
    full_embeddings_map = await _fetch_embeddings_full(session, candidates)

    # Combine them: prefer full_embedding if available
    combined_embeddings = {}
    for doc_id in candidates:
        if doc_id in full_embeddings_map:
            combined_embeddings[doc_id] = full_embeddings_map[doc_id]
        elif doc_id in truncated_embeddings_map:
            combined_embeddings[doc_id] = truncated_embeddings_map[doc_id]

    # Stage 2: cosine rerank
    reranked: list[tuple[str, float]] = []
    for doc_id in candidates:
        cos = _cosine(embedding, combined_embeddings[doc_id]) if doc_id in combined_embeddings else -1.0
        reranked.append((doc_id, cos))
    reranked.sort(key=lambda x: x[1], reverse=True)

    # Stage 3: MMR diversity
    mmr_pool = reranked[:CANDIDATE_N]
    return _mmr_rerank(mmr_pool, combined_embeddings, embedding, limit, lam=0.7)


def apply_staleness_penalty(results: list) -> None:  # type: ignore[type-arg]
    """
    Multiply each SearchResult score in-place by a staleness factor based on ingested_at.

    Age buckets (calendar days since ingested_at):
        0-7 days    → 1.00 (no penalty)
        7-30 days   → 0.97
        30-90 days  → 0.93
        90-365 days → 0.88
        >365 days   → 0.80

    Results with ingested_at=None are not penalised.
    """
    from datetime import datetime

    now = datetime.now(UTC)
    for r in results:
        if r.ingested_at is None:
            continue
        ia = r.ingested_at
        if ia.tzinfo is None:
            ia = ia.replace(tzinfo=UTC)
        age_days = (now - ia).total_seconds() / 86400
        if age_days <= 7:
            multiplier = 1.00
        elif age_days <= 30:
            multiplier = 0.97
        elif age_days <= 90:
            multiplier = 0.93
        elif age_days <= 365:
            multiplier = 0.88
        else:
            multiplier = 0.80
        r.score = round(r.score * multiplier, 6)


async def run_ranker(
    name: str,
    query: str,
    session: AsyncSession,
    embedding: list[float] | None,
    limit: int,
) -> list[tuple[str, float]]:
    """Execute a named ranker. Raises KeyError if not found."""
    entry = RANKER_REGISTRY[name]
    return await entry["fn"](query, session, embedding, limit)


def list_rankers() -> list[dict]:
    """Return enabled rankers as serializable dicts.
    
    Only rankers whose IDs appear in settings.enabled_rankers are included.
    """
    from dewie.config import settings

    enabled = set(settings.enabled_rankers)
    return [
        {"id": k, "label": v["label"], "description": v["description"]}
        for k, v in RANKER_REGISTRY.items()
        if k in enabled
    ]


# ── Entity/keyword match ranker ────────────────────────────────────────────────


async def _entity_match(
    session: AsyncSession,
    query_terms: list[str],
    limit: int,
) -> list[tuple[str, float]]:
    """
    Score docs by how many query terms appear in their entities + keywords + topics.
    Returns (doc_id, score) where score = matched_terms / total_query_terms.
    Fast: pure Postgres array intersection, no embedding needed.
    """
    if not query_terms:
        return []
    try:
        # Build array literal for Postgres
        terms_array = "{" + ",".join(f'"{t}"' for t in query_terms) + "}"
        # Score docs by entity/keyword/topic overlap with query terms
        # Uses a two-phase approach:
        # 1. FTS pre-filter with simple term matching (OR of all terms)
        # 2. Python-side exact scoring against entity/keyword/topic arrays
        if not query_terms:
            return []

        import json as _json

        # Phase 1: broad candidate fetch — any doc where title/entities/keywords
        # contain any of the query terms (simple word match, not stemmed)
        or_conditions = " OR ".join(
            f"(lower(title) LIKE '%' || lower(:t{i}) || '%' "
            f"OR lower(COALESCE((SELECT string_agg(v,' ') FROM jsonb_array_elements_text(entities) v),'')) LIKE '%' || lower(:t{i}) || '%' "
            f"OR lower(COALESCE((SELECT string_agg(v,' ') FROM jsonb_array_elements_text(keywords) v),'')) LIKE '%' || lower(:t{i}) || '%')"
            for i in range(len(query_terms))
        )
        params = {f"t{i}": t for i, t in enumerate(query_terms)}
        params["scan"] = limit * 30

        scan_sql = f"""
            SELECT id::text, title, entities, keywords, topics
            FROM documents
            WHERE status = 'ready' AND ({or_conditions})
            LIMIT :scan
        """
        scan_rows = (await session.execute(text(scan_sql), params)).fetchall()

        # Phase 2: score each candidate by fraction of query terms matched
        query_lower = [t.lower() for t in query_terms]
        scored = []
        for doc_id, title, ents, kws, tops in scan_rows:
            # Build set of searchable strings from this doc
            searchable = set()
            if title:
                searchable.add(title.lower())
                for w in title.lower().split():
                    searchable.add(w)
            for arr in [ents, kws, tops]:
                if arr:
                    items = arr if isinstance(arr, list) else _json.loads(arr)
                    for item in items:
                        il = item.lower()
                        searchable.add(il)
                        for w in il.split():
                            searchable.add(w)
            # Count query terms that appear in any searchable string
            matches = sum(
                1 for qt in query_lower if any(qt == s or qt in s or s in qt for s in searchable)
            )
            if matches > 0:
                scored.append((doc_id, matches / len(query_lower)))

        scored.sort(key=lambda x: (-x[1], x[0]))
        rows = scored[:limit]
        return [(str(r[0]), float(r[1])) for r in rows]
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning("_entity_match failed: %s", e)
        return []


def _extract_query_terms(query: str) -> list[str]:
    """
    Extract meaningful terms from a query for entity/keyword matching.
    Prioritizes: proper nouns (capitalized), stock tickers (ALL CAPS 2-5 chars),
    multi-word phrases. Excludes generic words that cause false positives.
    """
    STOP = frozenset(
        {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "was",
            "are",
            "were",
            "what",
            "when",
            "where",
            "who",
            "why",
            "how",
            "did",
            "do",
            "does",
            "has",
            "have",
            "had",
            "will",
            "would",
            "can",
            "could",
            "should",
            "that",
            "this",
            "it",
            "its",
            "be",
            "been",
            "being",
            "about",
            "any",
            # Generic words that cause too many false positives in entity matching:
            "score",
            "scores",
            "game",
            "games",
            "today",
            "news",
            "update",
            "latest",
            "rate",
            "rates",
            "stock",
            "price",
            "day",
            "week",
            "year",
            "time",
            "new",
            "old",
            "big",
            "high",
            "low",
            "top",
            "best",
            "last",
            "next",
            "win",
            "loss",
            "result",
            "results",
            "report",
            "data",
            "info",
        }
    )
    import re

    tokens = re.findall(r"[A-Z]{2,5}|[A-Za-z0-9]+(?:'[a-zA-Z]+)?", query)
    meaningful = []
    for t in tokens:
        # Always keep: stock tickers (2-5 ALL CAPS), proper nouns (starts uppercase)
        # and terms >= 5 chars not in stop list
        is_ticker = t.isupper() and 2 <= len(t) <= 5
        is_proper = t[0].isupper() and not t.isupper() and len(t) >= 3
        is_long_common = len(t) >= 5 and t.lower() not in STOP
        if is_ticker or is_proper or is_long_common:
            meaningful.append(t)
    return meaningful


@ranker(
    "entity_match",
    "Entity/Keyword Match",
    "Scores docs by how many query terms appear in entities, keywords, and topics. "
    "Best for short keyword queries, named entities, sports scores, stock tickers.",
)
async def rank_entity_match(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    terms = _extract_query_terms(query)
    return await _entity_match(session, terms, limit)


@ranker(
    "adaptive",
    "Adaptive (auto-select)",
    "Automatically selects the best ranker based on query characteristics. "
    "Short keyword/entity queries → entity_match + rrf hybrid. "
    "Natural language questions → answers_questions_rrf (AQ). "
    "Falls back to rrf_aq if uncertain.",
)
async def rank_adaptive(
    query: str, session: AsyncSession, embedding, limit: int
) -> list[tuple[str, float]]:
    """
    Route to the right ranker based on query shape:
    - Short (<= 5 tokens) or high entity density → blend entity_match + rrf
    - Natural language question → rrf_aq
    - Default → rrf_aq
    """
    terms = _extract_query_terms(query)
    word_count = len(query.split())
    has_question_word = any(
        query.lower().startswith(w)
        for w in (
            "what",
            "when",
            "where",
            "who",
            "why",
            "how",
            "which",
            "is ",
            "are ",
            "did ",
            "does ",
        )
    )

    # Short keyword query (tweet-style) or named entity heavy
    is_keyword_query = word_count <= 5 and not has_question_word

    if is_keyword_query:
        # Blend entity match + BM25 RRF for short queries
        K = 60
        entity_results = await _entity_match(session, terms, limit)
        fts_results = await _fts(session, query, limit)
        aq_results = await _aq_vec(session, embedding, limit)
        scores: dict[str, float] = {}
        # Entity match weighted 2x — it's the primary signal for keyword queries
        for rank, (doc_id, _) in enumerate(entity_results):
            scores[doc_id] = scores.get(doc_id, 0) + 2.0 / (K + rank + 1)
        for rank, (doc_id, _) in enumerate(fts_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
        for rank, (doc_id, _) in enumerate(aq_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (K + rank + 1)
        top = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
        return [(d, scores[d]) for d in top]
    else:
        # Natural language question → rrf_aq is the right tool
        return await rank_answers_questions_rrf(query, session, embedding, limit)
