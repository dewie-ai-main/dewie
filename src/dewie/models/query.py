# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Request and response models for the public query API.
"""

from datetime import date, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from dewie.config import settings
from dewie.models.content import ContentDocument


class ExpandBy(StrEnum):
    """Dimension along which the recursive traversal expands the graph."""

    TOPICS = "topics"
    ENTITIES = "entities"
    KEYWORDS = "keywords"
    SEMANTIC = "semantic"


class SearchRequest(BaseModel):
    """Agent search query."""

    query: str = Field(min_length=1, max_length=500, description="Free-text search query.")
    limit: int = Field(default=10, ge=1, le=100)
    ranker: str = Field(
        default=settings.query_default_ranker,
        description="Ranking strategy. See GET /query/rankers.",
    )
    min_enrichment_quality_score: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Minimum enrichment_quality_score for returned docs (0-100). "
            "Docs with NULL enrichment_quality_score (not yet enriched) are always included. "
            "None = use server default from dewie.yml ingest.quality_filter.min_enrichment_quality_score."
        ),
    )
    exclude_reading_levels: list[str] | None = Field(
        default=None,
        description=(
            "Reading levels to exclude from results. E.g. ['quick_read']. "
            "None = use server default from dewie.yml ingest.quality_filter.exclude_reading_levels."
        ),
    )
    category: str | None = Field(
        default=None,
        max_length=100,
        description=(
            "Optional category hint (e.g. 'finance', 'health'). "
            "Stored on the search_queue entry when gap enrichment fires, "
            "so the background worker can route to specialized sources. "
            "Does not filter results — category-based routing is a future feature."
        ),
    )
    published_after: str | None = Field(
        default=None,
        description=(
            "ISO date string (YYYY-MM-DD). Exclude docs ingested before this date. "
            "Applied as a WHERE ingested_at >= filter on the result fetch."
        ),
    )
    published_before: str | None = Field(
        default=None,
        description=(
            "ISO date string (YYYY-MM-DD). Exclude docs ingested after this date. "
            "Applied as a WHERE ingested_at <= filter on the result fetch."
        ),
    )
    staleness_penalty: bool = Field(
        default=False,
        description=(
            "When true, multiply each result score by a staleness factor based on ingested_at. "
            "7d: 1.0, 7-30d: 0.97, 30-90d: 0.93, 90-365d: 0.88, >1y: 0.80. "
            "Applied after RRF so existing behavior is unchanged when false."
        ),
    )
    disable_enrichment: bool = Field(
        default=False,
        description=(
            "When true, suppress gap-triggered Brave/Exa enrichment for this query. "
            "Use during benchmarking to keep the corpus stable."
        ),
    )
    max_tier: int = Field(
        default=0,
        description=(
            "Reserved for future federation support. Currently ignored — "
            "all searches are local corpus only. "
            "Peer search is not yet implemented; this field is accepted and stored for future use."
        ),
    )
    source_id: str | None = Field(
        default=None,
        description=(
            "Optional UUID of a registered dewie_source to route this query to. "
            "When set, the query is forwarded to that source (postgres or mcp remote instance) "
            "instead of the local corpus. Requires source type 'mcp' with a valid endpoint config."
        ),
    )


class SearchResult(BaseModel):
    """One search result with full metadata for agent re-ranking."""

    doc_type: str | None
    doc_id: str
    title: str
    summary: str | None
    url: str | None
    source: str | None
    topics: list[str]
    keywords: list[str]
    entities: list[str]
    sentiment: float | None
    answers_questions: list[str] = Field(default=[], exclude=True)
    score: float
    edge_count: int
    ingested_at: datetime | None = None
    enrichment_quality_score: int | None = Field(
        default=None,
        description="LLM-estimated quality (0-100). NULL = not yet enriched.",
    )
    reading_level: str | None = Field(
        default=None,
        description="Reading level: quick_read, standard, long_read, deep_dive, academic.",
    )
    chunk_match: str | None = Field(
        default=None,
        description=(
            "Best-matching chunk text for this document when chunks=true or rerank=true is passed to /query. "
            "None when chunk search was not requested or the document has no chunks."
        ),
    )
    chunk_score: float | None = Field(
        default=None,
        description=(
            "Cosine similarity score of the best matching chunk (0.0–1.0). "
            "Populated when rerank=true; None otherwise."
        ),
    )
    source_node: str | None = Field(
        default=None,
        description=(
            "Origin of this result. None = local corpus. "
            "Non-null = base URL of the peer node that returned this result."
        ),
    )


class ResultConfidence(BaseModel):
    """
    Adaptive retrieval signal — tells an agent whether to act on the current results or dig deeper.

    complexity:
      "lookup"      — one document clearly answers the query; stop here.
      "ambiguous"   — multiple docs compete (low score gap); expand neighbours or intersect.
      "distributed" — answer is spread across documents; use bridge or intersect to connect them.

    score_gap: rank-1 score minus rank-2 score. Small gap = ambiguous.
    aq_coverage_ratio: fraction of query terms in the top result's answers_questions.
                       High = document directly answers the query.
    edge_density: normalized edge_count of top result (0.0–1.0, capped at 50 edges = 1.0).
                  High = rich graph neighbourhood worth exploring.
    confidence_level: "high" | "medium" | "low"
    suggested_action: "none" | "expand" | "intersect" | "bridge"
      none      — top result is sufficient
      expand    — call dewie_expand on top result doc_id to pull neighbours
      intersect — call dewie_intersect across top-3 doc_ids to find shared context
      bridge    — call dewie_bridge; answer likely spans two disconnected document clusters
    """

    score_gap: float = Field(description="rank-1 score minus rank-2 score (0.0 if only one result)")
    aq_coverage_ratio: float = Field(
        description="fraction of query terms in top result's answers_questions (0.0–1.0)"
    )
    edge_density: float = Field(description="normalized edge_count of top result (0.0–1.0)")
    complexity: str = Field(description="lookup | ambiguous | distributed")
    confidence_level: str = Field(description="high | medium | low")
    suggested_action: str = Field(description="none | expand | intersect | bridge")
    probe_hint: str | None = Field(
        default=None,
        description=(
            "Set when complexity='distributed': suggests calling POST /capabilities/probe "
            "with the query context to map the knowledge space before retrying."
        ),
    )
    gap_signal: str | None = Field(
        default=None,
        description=(
            "Set when the query hits a coverage gap in the corpus. "
            "Indicates the information is absent or too thin to be useful — "
            "the agent should not retry with a rephrased query, but instead "
            "broaden scope, switch sources, or flag the gap to the user. "
            "Distinct from low confidence (ambiguous/distributed), which means "
            "the information exists but needs deeper traversal."
        ),
    )


class SearchResponse(BaseModel):
    """Response for POST /query."""

    query_id: str | None = Field(
        default=None, description="Unique ID for this query run — use for tracking and support."
    )
    query: str
    as_of: date = Field(
        default_factory=date.today,
        description="Date the corpus was queried — use this to orient temporal reasoning.",
    )
    results: list[SearchResult]
    total: int
    result_confidence: ResultConfidence | None = Field(
        default=None,
        description=(
            "Adaptive retrieval signal. Read complexity first: 'lookup' = stop here; "
            "'ambiguous' = try expand or intersect; 'distributed' = answer spans docs, use bridge. "
            "suggested_action is the recommended next MCP tool call."
        ),
    )
    fallback_triggered: bool = Field(
        default=False,
        description=(
            "True when gap_signal fired or top-result score < 0.3. "
            "Indicates the corpus coverage is thin and a background enrichment job has been queued."
        ),
    )
    gap_enrichment_queued: bool = Field(
        default=False,
        description=(
            "True when a search_queue entry was successfully inserted for async Brave enrichment. "
            "False if already queued, if the queue insert failed, or if no gap was detected."
        ),
    )
    source_id: str | None = Field(
        default=None,
        description="When set, indicates results came from a remote registered source.",
    )


class QueryRequest(BaseModel):
    """Initial search query submitted by an agent."""

    query: str = Field(min_length=1, max_length=500, description="Free-text search query.")
    max_results: int = Field(default=10, ge=1, le=100)
    expand_by: ExpandBy = Field(default=ExpandBy.TOPICS)


class RelatedQueryRequest(BaseModel):
    """Request to recursively expand from a known document."""

    document_id: UUID = Field(description="Starting node for recursive traversal.")
    max_depth: int = Field(
        default=settings.default_max_depth,
        ge=1,
        le=settings.absolute_max_depth,
        description="Maximum recursion depth.",
    )
    expand_by: ExpandBy = Field(default=ExpandBy.TOPICS)
    max_nodes_per_level: int = Field(
        default=settings.max_nodes_per_level,
        ge=1,
        le=100,
        description="Maximum related documents fetched per depth level.",
    )


class QueryNode(BaseModel):
    """One node in the recursive result tree."""

    document: ContentDocument
    depth: int = Field(ge=0)
    path: list[str] = Field(
        default_factory=list,
        description="Ordered list of document IDs traversed to reach this node.",
    )
    related: list["QueryNode"] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Top-level response returned for both /query and /query/related."""

    query: str
    total_nodes: int
    max_depth_reached: int
    results: list[QueryNode]

    # Pagination cursor for large result sets
    next_cursor: str | None = None


class BenchmarkResult(BaseModel):
    """Per-document comparison row for GET /query/benchmark."""

    doc_id: str
    title: str
    doc_score: float = Field(description="Doc-level ranker score (identical for both methods).")
    chunk_score: float | None = Field(
        default=None,
        description="Best chunk cosine score. None if the document has no embedded chunks.",
    )
    standard_rank: int = Field(
        description="1-based rank in the standard (doc-level) result list. 0 = not in top-limit."
    )
    reranked_rank: int = Field(
        description="1-based rank after chunk-based reranking. 0 = not in top-limit."
    )
    rank_change: int = Field(
        description=(
            "standard_rank minus reranked_rank. "
            "Positive = doc moved up; negative = moved down; 0 = unchanged or not comparable."
        )
    )


class BenchmarkResponse(BaseModel):
    """Response for GET /query/benchmark — side-by-side reranking comparison."""

    query: str
    limit: int = Field(description="Number of results returned per method.")
    standard_results: list[SearchResult] = Field(
        description="Doc-level ranked results (no chunk reranking)."
    )
    reranked_results: list[SearchResult] = Field(
        description="Results after chunk-based reranking, with chunk_match and chunk_score populated."
    )
    comparison: list[BenchmarkResult] = Field(
        description=(
            "Union of both result sets with rank-change analysis. "
            "Sorted by standard_rank (docs absent from standard list appear last)."
        )
    )


class CategoryHint(BaseModel):
    """Category distribution hint for a search result set."""

    result_count: int = Field(description="Number of results from this category")
    corpus_count: int = Field(description="Total docs in corpus from this category")
    suggested: bool = Field(description="True if this category dominates results")


class CategoryQueryResponse(BaseModel):
    """Response for POST /query/category — includes category distribution hints."""

    query_id: str | None = Field(default=None, description="Unique query ID for tracking")
    query: str
    as_of: date = Field(default_factory=date.today)
    results: list[SearchResult]
    total: int
    result_confidence: ResultConfidence | None = None
    fallback_triggered: bool = False
    gap_enrichment_queued: bool = False
    category_hints: dict[str, CategoryHint] = Field(
        default_factory=dict,
        description="Category distribution of results — shows which domains have relevant content",
    )
    category_suggestion: str | None = Field(
        default=None, description="Hint to narrow by category when one category dominates results"
    )


# Required for self-referential model
QueryNode.model_rebuild()
