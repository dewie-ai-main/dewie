# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
pipeline.py — Enrichment helpers shared across the ingest pipeline.

Provides:
  - generate_aq / extract_ke  — LLM enrichment steps (AQ + keyword/entity)
  - build_embed_text / embed_batch — embedding utilities
  - compute_quality_score — deterministic quality scoring
  - jaccard / tokenize — similarity primitives
  - rebuild_edges — full-corpus edge rebuild (Jaccard + vector)

enrich_docs (Pass B) was removed in commit 4b695ee; enrichment is now
handled exclusively by enrichment_flow.py.
"""

from __future__ import annotations

import json
import logging
import re
import time

import httpx
from sqlalchemy import text

from dewie.providers import get_chat_provider, get_embedding_provider

log = logging.getLogger(__name__)

EMBED_BATCH = 20
CONCURRENT = 5


# ── Step 1: answers_questions ─────────────────────────────────────────────────

AQ_SYSTEM = """You are a document indexing assistant. Given a document title, summary, and excerpt, generate exactly 6-8 natural-language questions that this document directly answers.

Rules:
- Questions must be answerable by THIS document specifically
- Use varied vocabulary — synonyms, different phrasings, related terms
- Include both specific ("How do I configure X?") and conceptual ("What is X?") questions
- Do NOT just paraphrase the title
- Output ONLY a JSON array of strings, no other text"""


async def generate_aq(http: httpx.AsyncClient, title: str, summary: str, content: str) -> list[str]:
    excerpt = content[:1200] if content else summary
    prompt = f"Title: {title}\nSummary: {summary}\nExcerpt: {excerpt[:800]}"
    try:
        provider = get_chat_provider("aq_generation")
        raw = await provider.complete(
            messages=[
                {"role": "system", "content": AQ_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.3,
        )
        if not raw:
            return []
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
        qs = json.loads(raw)
        if isinstance(qs, list):
            return [str(q) for q in qs if q][:8]
    except Exception as e:
        log.warning(f"AQ failed for '{title[:40]}': {e}")
    return []


# ── Step 2: keywords + entities ──────────────────────────────────────────────

KE_SYSTEM = """Extract keywords and named entities from this document for search indexing.

Output ONLY valid JSON with exactly these keys:
{
  "keywords": ["list", "of", "atomic", "terms"],
  "entities": ["Named Entity 1", "Named Entity 2"]
}

Keywords: CLI names, config keys, API names, technical terms, concepts (20-40 items)
Entities: proper nouns — tool names, product names, company names, protocol names (5-15 items)
No duplicates. No sentences. Lowercase keywords. Title-case entities."""


async def extract_ke(http: httpx.AsyncClient, title: str, summary: str, content: str) -> dict:
    excerpt = f"Title: {title}\nSummary: {summary}\nContent excerpt: {content[:1500]}"
    try:
        provider = get_chat_provider("keyword_extraction")
        raw = await provider.complete(
            messages=[
                {"role": "system", "content": KE_SYSTEM},
                {"role": "user", "content": excerpt},
            ],
            max_tokens=600,
            temperature=0.1,
        )
        if not raw:
            return {"keywords": [], "entities": []}
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
        data = json.loads(raw)
        return {
            "keywords": [str(k).lower() for k in data.get("keywords", [])][:40],
            "entities": [str(e) for e in data.get("entities", [])][:15],
        }
    except Exception as e:
        log.warning(f"KE failed for '{title[:40]}': {e}")
    return {"keywords": [], "entities": []}


# ── Step 3: embeddings ────────────────────────────────────────────────────────


def build_embed_text(
    title: str, summary: str, aq: list, content: str, embed_summary: str = ""
) -> str:
    parts = [f"Title: {title}"]
    # Use retrieval-dense embed_summary if available, fall back to display summary
    dense = embed_summary or summary
    if dense:
        parts.append(f"Summary: {dense}")
    if aq:
        parts.append("Questions: " + " | ".join(aq[:6]))
    # No raw body slicing — the embed_summary already distilled it
    return "\n".join(parts)


async def embed_batch(
    http: httpx.AsyncClient,
    texts: list[str],
    _retries: int = 4,
    full_out: list[list[float]] | None = None,
) -> list[list[float]] | None:
    """Generate embeddings. If `full_out` is passed, it's populated (in place)
    with the untruncated vectors when the provider applied MRL truncation —
    only meaningful when settings.embed_store_full_vector is set by the caller."""
    try:
        provider = get_embedding_provider()
    except RuntimeError as e:
        log.warning("embed_batch: no embedding provider configured, skipping: %s", e)
        return None
    vectors = await provider.embed(texts)
    if full_out is not None:
        full_out.extend(getattr(provider, "last_full_vectors", None) or [])
    return vectors


# ── Step 4: Deterministic quality score ──────────────────────────────────────


def compute_quality_score(
    body_text: str,
    keywords: list,
    entities: list,
    aq: list,
    summary: str,
) -> int:
    """
    Compute a document quality score (0-100) from observable enrichment signals.

    This replaces the LLM-assigned enrichment_quality_score, which is unreliable:
    LLMs cluster scores in the 70-85 band regardless of actual content quality.

    Scoring logic:
      - Body length (35 pts): raw information volume — stubs get 0
      - Keyword richness (20 pts): distinct meaningful terms extracted
      - Entity density (20 pts): named persons/orgs/places/products found
      - AQ depth (15 pts): distinct questions the doc answers
      - Summary quality (10 pts): whether enrichment produced a real summary

    A tweet-length stub scores ~0. A full investigative article scores ~95.
    """
    score = 0

    # Body length — primary signal (0-35 pts)
    chars = len(body_text or "")
    if chars >= 3000:
        score += 35
    elif chars >= 1500:
        score += 26
    elif chars >= 500:
        score += 16
    elif chars >= 150:
        score += 8

    # Keyword richness (0-20 pts)
    n_kw = len(keywords or [])
    if n_kw >= 15:
        score += 20
    elif n_kw >= 8:
        score += 14
    elif n_kw >= 4:
        score += 9
    elif n_kw >= 1:
        score += 4

    # Entity density (0-20 pts)
    n_ent = len(entities or [])
    if n_ent >= 8:
        score += 20
    elif n_ent >= 4:
        score += 14
    elif n_ent >= 2:
        score += 8
    elif n_ent >= 1:
        score += 3

    # AQ depth — how much value enrichment extracted (0-15 pts)
    n_aq = len(aq or [])
    if n_aq >= 6:
        score += 15
    elif n_aq >= 4:
        score += 10
    elif n_aq >= 2:
        score += 5
    elif n_aq >= 1:
        score += 2

    # Summary quality (0-10 pts)
    s_len = len(summary or "")
    if s_len >= 200:
        score += 10
    elif s_len >= 100:
        score += 6
    elif s_len >= 40:
        score += 3

    return min(score, 100)


# ── Edge rebuild (deferred, whole-corpus Jaccard) ────────────────────────────


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def tokenize(texts: list) -> set:
    tokens = set()
    for t in texts:
        if isinstance(t, str):
            for tok in re.split(r"[\s\-_/.,]+", t.lower()):
                if len(tok) > 2:
                    tokens.add(tok)
    return tokens


async def rebuild_edges(engine, min_weight: float = 0.05, limit_per_doc: int = 20):
    """
    Rebuild all document_edges using two SQL-side passes — no Python-side
    data loading, bounded memory regardless of corpus size.

    Pass 1: Keyword/topic/entity Jaccard similarity via SQL set operations.
            Uses a temp table of (doc_id, token) rows + self-join to find
            candidate pairs. Skips tokens appearing in >500 docs (noise).
    Pass 2: pgvector ANN per-doc top-N, processed in batches of 200.
            Results written directly to document_edges, never held in Python.
    """
    log.info("Rebuilding document edges (SQL-native, bounded memory)...")
    t0 = time.time()

    async with engine.begin() as conn:
        # ── Stage table for new edges (swap in atomically at end) ────────────
        await conn.execute(text("DROP TABLE IF EXISTS document_edges_new"))
        await conn.execute(
            text("""
            CREATE UNLOGGED TABLE document_edges_new (
                source_id UUID NOT NULL,
                target_id UUID NOT NULL,
                weight     FLOAT NOT NULL,
                rel_type   TEXT NOT NULL DEFAULT 'keyword_similarity',
                PRIMARY KEY (source_id, target_id)
            )
        """)
        )

        # ── Pass 1: Keyword Jaccard via SQL ───────────────────────────────────
        # Explode keywords+entities+topics into a (doc_id, token) temp table,
        # self-join on token to find pairs, compute Jaccard in SQL.
        # Skips hyper-common tokens (>500 docs) to avoid O(n²) explosions.
        await conn.execute(
            text("""
            WITH doc_tokens AS (
                SELECT
                    d.id AS doc_id,
                    lower(tok) AS token
                FROM documents d,
                     jsonb_array_elements_text(
                         COALESCE(d.keywords, '[]'::jsonb) ||
                         COALESCE(d.entities, '[]'::jsonb) ||
                         COALESCE(d.topics,   '[]'::jsonb)
                     ) AS tok
                WHERE d.status = 'ready'
                  AND tok IS NOT NULL AND length(lower(tok)) >= 3
            ),
            token_freq AS (
                SELECT token, COUNT(DISTINCT doc_id) AS freq
                FROM doc_tokens
                GROUP BY token
            ),
            rare_tokens AS (
                SELECT dt.doc_id, dt.token
                FROM doc_tokens dt
                JOIN token_freq tf USING (token)
                WHERE tf.freq BETWEEN 2 AND 500
            ),
            doc_sizes AS (
                SELECT doc_id, COUNT(*) AS n_tokens
                FROM rare_tokens
                GROUP BY doc_id
            ),
            pairs AS (
                SELECT
                    LEAST(a.doc_id, b.doc_id)    AS src,
                    GREATEST(a.doc_id, b.doc_id) AS tgt,
                    COUNT(*) AS intersection
                FROM rare_tokens a
                JOIN rare_tokens b USING (token)
                WHERE a.doc_id < b.doc_id
                GROUP BY src, tgt
                HAVING COUNT(*) >= 1
            ),
            scored AS (
                SELECT
                    p.src,
                    p.tgt,
                    p.intersection::float /
                        (sa.n_tokens + sb.n_tokens - p.intersection) AS jaccard
                FROM pairs p
                JOIN doc_sizes sa ON sa.doc_id = p.src
                JOIN doc_sizes sb ON sb.doc_id = p.tgt
                WHERE p.intersection::float /
                      (sa.n_tokens + sb.n_tokens - p.intersection) >= :min_w
            ),
            ranked AS (
                SELECT src, tgt, jaccard,
                       ROW_NUMBER() OVER (PARTITION BY src ORDER BY jaccard DESC) AS rn_src,
                       ROW_NUMBER() OVER (PARTITION BY tgt ORDER BY jaccard DESC) AS rn_tgt
                FROM scored
            )
            INSERT INTO document_edges_new (source_id, target_id, weight, rel_type)
            SELECT src::uuid, tgt::uuid, jaccard, 'keyword_similarity'
            FROM ranked
            WHERE rn_src <= :lim AND rn_tgt <= :lim
            ON CONFLICT DO NOTHING
        """),
            {"min_w": min_weight, "lim": limit_per_doc},
        )

        kw_count = (await conn.execute(text("SELECT COUNT(*) FROM document_edges_new"))).scalar()
        log.info(f"  Keyword edges: {kw_count}")

        # ── Pass 2: Vector ANN in batches of 200 docs ─────────────────────────
        id_rows = (
            await conn.execute(
                text(
                    "SELECT id FROM documents WHERE embedding IS NOT NULL AND status = 'ready' ORDER BY id"
                )
            )
        ).all()
        embed_ids = [str(r[0]) for r in id_rows]
        vec_count = 0
        batch_size = 200

    for i in range(0, len(embed_ids), batch_size):
        batch = embed_ids[i : i + batch_size]
        async with engine.begin() as conn:
            for doc_id in batch:
                await conn.execute(
                    text("""
                    INSERT INTO document_edges_new (source_id, target_id, weight, rel_type)
                    SELECT
                        LEAST(cast(:id as uuid), id)    AS source_id,
                        GREATEST(cast(:id as uuid), id) AS target_id,
                        LEAST((1 - (embedding <=> (SELECT embedding FROM documents WHERE id = cast(:id2 as uuid)))) * 0.7, 1.0) AS weight,
                        'vector_similarity'
                    FROM documents
                    WHERE id != cast(:id3 as uuid)
                      AND embedding IS NOT NULL
                      AND status = 'ready'
                      AND (1 - (embedding <=> (SELECT embedding FROM documents WHERE id = cast(:id4 as uuid)))) >= 0.5
                    ORDER BY embedding <=> (SELECT embedding FROM documents WHERE id = cast(:id5 as uuid))
                    LIMIT 10
                    ON CONFLICT (source_id, target_id) DO UPDATE
                        SET weight = GREATEST(document_edges_new.weight, EXCLUDED.weight)
                """),
                    {"id": doc_id, "id2": doc_id, "id3": doc_id, "id4": doc_id, "id5": doc_id},
                )
                vec_count += 1
        if i % 2000 == 0 and i > 0:
            log.info(f"  Vector pass: {i}/{len(embed_ids)} docs processed...")

    log.info(f"  Vector edges added for {vec_count} docs")

    async with engine.begin() as conn:
        # ── Atomic swap ───────────────────────────────────────────────────────
        total = (await conn.execute(text("SELECT COUNT(*) FROM document_edges_new"))).scalar()
        log.info(f"  Total edges: {total} — swapping in atomically...")
        await conn.execute(text("DELETE FROM document_edges"))
        await conn.execute(
            text("""
            INSERT INTO document_edges (source_id, target_id, weight, rel_type)
            SELECT source_id, target_id, weight, rel_type FROM document_edges_new
            ON CONFLICT DO NOTHING
        """)
        )
        await conn.execute(text("DROP TABLE IF EXISTS document_edges_new"))
        # Sync to relationships table — only for docs that exist (FK safe)
        await conn.execute(text("DELETE FROM relationships"))
        await conn.execute(
            text("""
            INSERT INTO relationships (source_id, target_id, weight, rel_type)
            SELECT de.source_id, de.target_id, de.weight, de.rel_type
            FROM document_edges de
            WHERE EXISTS (SELECT 1 FROM documents WHERE id = de.source_id)
              AND EXISTS (SELECT 1 FROM documents WHERE id = de.target_id)
            ON CONFLICT DO NOTHING
        """)
        )

        elapsed = time.time() - t0
    log.info(f"  Edge rebuild complete in {elapsed:.1f}s")


async def add_edges_for_doc(
    engine,
    doc_id: str,
    min_weight: float = 0.05,
    limit_per_doc: int = 20,
) -> int:
    """
    Incremental edge update for a single newly-enriched document.

    Uses the same inverted-index SQL approach as ``rebuild_edges`` but scoped
    to one document -- O(k * freq) where k = number of tokens in the new doc,
    not O(n) over the whole corpus.

    Pass 1: Keyword/entity/topic Jaccard via SQL inverted index.
            Finds every ready doc that shares >=1 rare token with *doc_id*,
            computes Jaccard in SQL, inserts bidirectional edges.
    Pass 2: Vector ANN -- top-10 nearest neighbours for *doc_id* (if it has
            an embedding), written symmetrically to document_edges.

    Returns the number of edges touching this document after the update.
    """
    async with engine.begin() as conn:
        # Pass 1: Keyword Jaccard via SQL inverted index
        await conn.execute(
            text("""
            WITH new_doc_tokens AS (
                SELECT lower(tok) AS token
                FROM documents,
                     jsonb_array_elements_text(
                         COALESCE(keywords, '[]'::jsonb) ||
                         COALESCE(entities, '[]'::jsonb) ||
                         COALESCE(topics,   '[]'::jsonb)
                     ) AS tok
                WHERE id = cast(:doc_id AS uuid)
                  AND tok IS NOT NULL AND length(lower(tok)) >= 3
            ),
            token_freq AS (
                SELECT lower(tok) AS token, COUNT(DISTINCT d.id) AS freq
                FROM documents d,
                     jsonb_array_elements_text(
                         COALESCE(d.keywords, '[]'::jsonb) ||
                         COALESCE(d.entities, '[]'::jsonb) ||
                         COALESCE(d.topics,   '[]'::jsonb)
                     ) AS tok
                WHERE d.status = 'ready'
                GROUP BY lower(tok)
            ),
            rare_new_tokens AS (
                SELECT ndt.token
                FROM new_doc_tokens ndt
                JOIN token_freq tf USING (token)
                WHERE tf.freq BETWEEN 2 AND 500
            ),
            new_doc_size AS (
                SELECT COUNT(*) AS n FROM rare_new_tokens
            ),
            candidates AS (
                SELECT DISTINCT d.id AS cand_id
                FROM documents d,
                     jsonb_array_elements_text(
                         COALESCE(d.keywords, '[]'::jsonb) ||
                         COALESCE(d.entities, '[]'::jsonb) ||
                         COALESCE(d.topics,   '[]'::jsonb)
                     ) AS tok
                WHERE d.status = 'ready'
                  AND d.id != cast(:doc_id2 AS uuid)
                  AND lower(tok) IN (SELECT token FROM rare_new_tokens)
            ),
            cand_tokens AS (
                SELECT c.cand_id, lower(tok) AS token
                FROM candidates c
                JOIN documents d ON d.id = c.cand_id,
                     jsonb_array_elements_text(
                         COALESCE(d.keywords, '[]'::jsonb) ||
                         COALESCE(d.entities, '[]'::jsonb) ||
                         COALESCE(d.topics,   '[]'::jsonb)
                     ) AS tok
                WHERE lower(tok) IS NOT NULL AND length(lower(tok)) >= 3
                  AND lower(tok) IN (SELECT token FROM rare_new_tokens)
            ),
            cand_sizes AS (
                SELECT c.cand_id, COUNT(*) AS n
                FROM candidates c
                JOIN documents d ON d.id = c.cand_id,
                     jsonb_array_elements_text(
                         COALESCE(d.keywords, '[]'::jsonb) ||
                         COALESCE(d.entities, '[]'::jsonb) ||
                         COALESCE(d.topics,   '[]'::jsonb)
                     ) AS tok
                JOIN token_freq tf ON tf.token = lower(tok)
                WHERE tf.freq BETWEEN 2 AND 500
                GROUP BY c.cand_id
            ),
            intersection AS (
                SELECT cand_id, COUNT(*) AS shared
                FROM cand_tokens
                GROUP BY cand_id
            ),
            scored AS (
                SELECT
                    i.cand_id,
                    i.shared::float / (nds.n + cs.n - i.shared) AS jaccard
                FROM intersection i
                JOIN new_doc_size nds ON true
                JOIN cand_sizes cs ON cs.cand_id = i.cand_id
                WHERE nds.n > 0
                  AND (nds.n + cs.n - i.shared) > 0
                  AND i.shared::float / (nds.n + cs.n - i.shared) >= :min_w
            ),
            ranked AS (
                SELECT cand_id, jaccard,
                       ROW_NUMBER() OVER (ORDER BY jaccard DESC) AS rn
                FROM scored
            )
            INSERT INTO document_edges (source_id, target_id, weight, rel_type)
            SELECT
                LEAST(cast(:doc_id3 AS uuid), cand_id)    AS source_id,
                GREATEST(cast(:doc_id4 AS uuid), cand_id) AS target_id,
                jaccard,
                'keyword_similarity'
            FROM ranked
            WHERE rn <= :lim
            ON CONFLICT (source_id, target_id, rel_type) DO UPDATE
                SET weight = GREATEST(document_edges.weight, EXCLUDED.weight),
                    rel_type = EXCLUDED.rel_type
        """),
            {
                "doc_id": doc_id,
                "doc_id2": doc_id,
                "doc_id3": doc_id,
                "doc_id4": doc_id,
                "min_w": min_weight,
                "lim": limit_per_doc,
            },
        )

        # Pass 2: Vector ANN top-10
        has_embedding = (
            await conn.execute(
                text(
                    "SELECT 1 FROM documents WHERE id = cast(:id AS uuid) AND embedding IS NOT NULL"
                ),
                {"id": doc_id},
            )
        ).scalar()

        if has_embedding:
            await conn.execute(
                text("""
                INSERT INTO document_edges (source_id, target_id, weight, rel_type)
                SELECT
                    LEAST(cast(:id AS uuid), id)    AS source_id,
                    GREATEST(cast(:id AS uuid), id) AS target_id,
                    LEAST(
                        (1 - (embedding <=> (
                            SELECT embedding FROM documents WHERE id = cast(:id2 AS uuid)
                        ))) * 0.7,
                        1.0
                    ) AS weight,
                    'vector_similarity'
                FROM documents
                WHERE id != cast(:id3 AS uuid)
                  AND embedding IS NOT NULL
                  AND status = 'ready'
                  AND (1 - (embedding <=> (
                      SELECT embedding FROM documents WHERE id = cast(:id4 AS uuid)
                  ))) >= 0.5
                ORDER BY embedding <=> (
                    SELECT embedding FROM documents WHERE id = cast(:id5 AS uuid)
                )
                LIMIT 10
                ON CONFLICT (source_id, target_id, rel_type) DO UPDATE
                    SET weight = GREATEST(document_edges.weight, EXCLUDED.weight)
            """),
                {"id": doc_id, "id2": doc_id, "id3": doc_id, "id4": doc_id, "id5": doc_id},
            )

        edge_count = (
            await conn.execute(
                text("""
                SELECT COUNT(*) FROM document_edges
                WHERE source_id = cast(:id AS uuid) OR target_id = cast(:id2 AS uuid)
            """),
                {"id": doc_id, "id2": doc_id},
            )
        ).scalar() or 0

    log.debug("add_edges_for_doc: %s -> %d edges", doc_id, edge_count)
    return edge_count
