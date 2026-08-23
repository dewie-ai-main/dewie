# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Corpus-first web lookup — the engine behind the `web_search` MCP tool.

Flow:
    1. Search the corpus.
    2. Gate on the gap signal (semantic, from query.py — NOT a static score
       threshold; RRF scores are corpus-size dependent and a fixed cutoff
       can never be calibrated).
    3. Gap quiet  → return corpus hits with provenance (source="corpus").
    4. Gap fired  → provider web search; use provider-returned page text when
       available (Exa/You), otherwise fetch + extract; persist the new document
       fire-and-forget; return content with source="web" and the gap reason.
    5. No provider configured → source="miss" with the gap reason, so the
       agent knows the corpus is thin AND why nothing else was tried.

Every result carries provenance: the agent (and the human auditing it) can
always tell where an answer came from and how stale it is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from dewie.search.providers import SearchHit, SearchProvider

log = logging.getLogger("dewie.search")

# Cap on content returned to the agent — full bodies live in the corpus.
_MAX_CONTENT_CHARS = 8000


@dataclass
class LookupResult:
    """Outcome of a corpus-first lookup, with provenance."""

    source: str  # "corpus" | "web" | "miss"
    gap: str | None = None
    corpus_hits: list[dict[str, Any]] = field(default_factory=list)
    web_hits: list[dict[str, Any]] = field(default_factory=list)
    content: str | None = None
    content_url: str | None = None
    content_title: str | None = None
    ingested_doc_id: str | None = None

    def to_content(self) -> dict[str, Any]:
        """Shape returned through the MCP tool. NEVER includes answers_questions."""
        out: dict[str, Any] = {"source": self.source}
        if self.gap:
            out["gap"] = self.gap
        if self.corpus_hits:
            out["corpus_hits"] = self.corpus_hits
        if self.web_hits:
            out["web_hits"] = self.web_hits
        if self.content is not None:
            out["content"] = self.content[:_MAX_CONTENT_CHARS]
            out["content_url"] = self.content_url
            out["content_title"] = self.content_title
        if self.ingested_doc_id:
            out["ingested_doc_id"] = self.ingested_doc_id
        return out


def _corpus_hit_dict(doc: Any, score: float) -> dict[str, Any]:
    # answers_questions is intentionally excluded — internal ranking signal only.
    return {
        "doc_id": str(doc.id),
        "title": doc.title,
        "url": str(doc.url) if doc.url else None,
        "summary": getattr(doc, "summary", "") or "",
        "score": round(score, 4),
        "ingested_at": doc.ingested_at.isoformat() if getattr(doc, "ingested_at", None) else None,
    }


def _gap_signal(query: str, results: list[tuple[Any, float]]) -> str | None:
    """Run the shared gap heuristic over (doc, score) pairs from pg.search."""
    # Imported lazily: query.py is a route module and pulls the FastAPI stack.
    from dewie.api.routes.query import _compute_gap_signal

    wrapped = [
        SimpleNamespace(
            score=score,
            answers_questions=list(getattr(doc, "answers_questions", None) or []),
            topics=list(getattr(doc, "topics", None) or []),
        )
        for doc, score in results
    ]

    # Unenriched corpus (no LLM configured yet): the AQ/topic-based heuristic
    # would fire on every query. Fall back to term coverage of the top hit's
    # title+summary — majority coverage means the corpus plausibly answers it.
    if results and not any(r.answers_questions or r.topics for r in wrapped):
        query_words = {w.lower() for w in query.split() if len(w) > 4}
        if query_words:
            top_doc = results[0][0]
            haystack = (
                f"{getattr(top_doc, 'title', '') or ''} "
                f"{getattr(top_doc, 'summary', '') or ''} "
                f"{getattr(top_doc, 'body', '') or ''}"
            ).lower()
            covered = sum(1 for w in query_words if w in haystack)
            if covered >= max(1, round(len(query_words) * 0.5)):
                return None

    return _compute_gap_signal(query, wrapped)


async def _fetch_page_text(url: str) -> tuple[str | None, str | None]:
    """Fetch a URL and extract article text. Returns (title, text) or (None, None)."""
    try:
        from dewie.ingestion.web import WebIngester

        async with WebIngester() as ingester:
            docs = [doc async for doc in ingester.fetch(url)]
        if docs and docs[0].body:
            return docs[0].title, docs[0].body
    except Exception as exc:
        log.debug("web_lookup: fetch failed for %s: %s", url, exc)
    return None, None


async def web_lookup(
    query: str,
    *,
    pg: Any,
    provider: SearchProvider | None,
    limit: int = 5,
    workspace_ids: list | None = None,
    force_web: bool = False,
    corpus_only: bool = False,
) -> tuple[LookupResult, Any | None]:
    """
    Corpus-first lookup. Returns (result, new_doc_or_None).

    The caller persists/enriches the returned new document (route layer owns
    BackgroundTasks); this function stays pure enough to unit test.
    """
    results = await pg.search(
        query=query,
        limit=limit,
        ranker="rrf",
        workspace_ids=workspace_ids or [],
    )

    gap = _gap_signal(query, results)
    corpus_hits = [_corpus_hit_dict(doc, score) for doc, score in results]

    if corpus_only or (gap is None and not force_web):
        if gap is None:
            return LookupResult(source="corpus", corpus_hits=corpus_hits), None
        return LookupResult(source="miss", gap=gap, corpus_hits=corpus_hits), None

    reason = gap if gap is not None else "force_web requested by caller"

    if provider is None:
        return (
            LookupResult(
                source="miss",
                gap=f"{reason} No web search provider configured "
                "(set SEARCH_PROVIDER=brave|exa|you and the matching API key).",
                corpus_hits=corpus_hits,
            ),
            None,
        )

    try:
        hits: list[SearchHit] = await provider.search(query, limit=limit)
    except Exception as exc:
        log.warning("web_lookup: provider %s failed: %s", provider.name, exc)
        return (
            LookupResult(
                source="miss",
                gap=f"{reason} Web search via {provider.name} failed: {exc}",
                corpus_hits=corpus_hits,
            ),
            None,
        )

    web_hits = [{"title": h.title, "url": h.url, "snippet": h.snippet} for h in hits]
    if not hits:
        return (
            LookupResult(source="miss", gap=reason, corpus_hits=corpus_hits, web_hits=[]),
            None,
        )

    # Take the first hit we can get text for: provider content first, fetch second.
    content: str | None = None
    title: str | None = None
    url: str | None = None
    for hit in hits:
        if hit.content:
            content, title, url = hit.content, hit.title, hit.url
            break
        fetched_title, fetched_text = await _fetch_page_text(hit.url)
        if fetched_text:
            content, title, url = fetched_text, fetched_title or hit.title, hit.url
            break

    if content is None:
        return (
            LookupResult(
                source="miss",
                gap=f"{reason} Web results found but no page content could be retrieved.",
                corpus_hits=corpus_hits,
                web_hits=web_hits,
            ),
            None,
        )

    new_doc = _build_document(url=url or "", title=title or url or query, body=content)

    return (
        LookupResult(
            source="web",
            gap=reason,
            corpus_hits=corpus_hits,
            web_hits=web_hits,
            content=content,
            content_url=url,
            content_title=title,
            ingested_doc_id=str(new_doc.id) if new_doc else None,
        ),
        new_doc,
    )


def _build_document(*, url: str, title: str, body: str) -> Any | None:
    """Build a pending ContentDocument from captured web content."""
    try:
        from urllib.parse import urlparse

        from dewie.models.content import ContentDocument, ContentStatus

        return ContentDocument(
            url=url,
            title=title,
            body=body,
            source=urlparse(url).netloc or "web_search",
            status=ContentStatus.PENDING,
        )
    except Exception as exc:
        log.warning("web_lookup: could not build document for %s: %s", url, exc)
        return None


async def persist_document(doc: Any, pg: Any) -> None:
    """Persist a captured document: upsert + body store. Never raises."""
    try:
        await pg.upsert(doc)
        from dewie.storage.body_store import save_body

        if getattr(doc, "body", None):
            save_body(doc.id, doc.body)
            try:
                await pg.write_body_text(doc.id, doc.body)
            except Exception:
                pass
    except Exception as exc:
        log.warning("web_lookup: persist failed for %s: %s", getattr(doc, "url", "?"), exc)
