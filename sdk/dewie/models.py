# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Pydantic models for the Dewie API — mirrors the actual response shapes.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, computed_field


class SearchResult(BaseModel):
    doc_id: str
    title: str
    summary: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    topics: list[str] = []
    keywords: list[str] = []
    entities: list[str] = []
    score: float
    edge_count: int = 0
    enrichment_quality_score: Optional[int] = None
    reading_level: Optional[str] = None
    doc_type: Optional[str] = None


class ResultConfidence(BaseModel):
    score_gap: float = 0.0
    aq_coverage_ratio: float = 0.0
    edge_density: float = 0.0
    complexity: str = "ambiguous"
    confidence_level: str = "low"
    suggested_action: str = "none"
    # gap_signal is the warning message string, or None if no gap detected
    gap_signal: Optional[str] = None
    probe_hint: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    as_of: Optional[str] = None
    results: list[SearchResult] = []
    total: int = 0
    result_confidence: Optional[ResultConfidence] = None

    @computed_field
    @property
    def has_gap(self) -> bool:
        """True when gap_signal fired — corpus likely doesn't cover this query."""
        return bool(
            self.result_confidence and self.result_confidence.gap_signal
        )

    @computed_field
    @property
    def gap_message(self) -> Optional[str]:
        """The gap_signal warning string, or None. Ready to surface to users/agents."""
        if not self.result_confidence:
            return None
        return self.result_confidence.gap_signal


class TraverseDocument(BaseModel):
    """A document returned by /traverse (different shape from /query results)."""
    id: str
    title: str
    url: Optional[str] = None
    summary: Optional[str] = None
    relevance_score: float = 0.0
    topics: list[str] = []
    entities: list[str] = []
    keywords: list[str] = []
    matched_on: list[str] = []


class NextCluster(BaseModel):
    """A suggested next traversal direction from /traverse."""
    keywords: list[str] = []
    relevance_score: float = 0.0
    context_centroid: Optional[str] = None
    anchor_doc_ids: list[str] = []
    anchor_doc_titles: list[str] = []


class TraverseResponse(BaseModel):
    """Response from POST /traverse — keyword-driven graph exploration."""
    documents: list[TraverseDocument] = []
    next_clusters: list[NextCluster] = []
    metadata: dict = {}


class BridgePath(BaseModel):
    """Response from POST /graph/bridge — shortest path between two doc IDs."""
    path: list[str] = []   # ordered doc_ids from source to target
    hops: int = 0
    error: Optional[str] = None
