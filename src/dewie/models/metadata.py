# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Models for metadata tags and inter-document relationships stored in the graph.

These models represent the enrichment output that is persisted to Postgres and
used to build the relationship graph that agents traverse during browse sessions.
"""

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class EntityTag(BaseModel):
    """
    A named entity extracted from a document.

    Entities are identified by the enrichment backend (spaCy NER labels or
    LLM-assigned labels).  They form one axis of inter-document relationship
    building — documents sharing high-salience entities receive SHARED_ENTITY
    edges in the graph.
    """

    text: str = Field(description="Surface form of the entity as it appears in the text.")
    label: str = Field(
        description=(
            "Entity type label.  spaCy labels: ORG, PERSON, GPE, PRODUCT, "
            "EVENT, WORK_OF_ART, LAW.  LLM backends may return broader labels."
        ),
    )
    salience: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Importance weight of this entity within the document (0–1).  "
            "Higher salience entities contribute more strongly to relationship scores."
        ),
    )


class RelationshipType(StrEnum):
    """
    Typed edge labels in the document relationship graph (Postgres).

    Each type represents a different axis along which documents are connected.
    Agents can restrict traversal to specific types via the ``expand_by``
    parameter on browse and query endpoints.
    """

    SHARED_TOPIC = "SHARED_TOPIC"
    """Documents share at least one topic label (Jaccard similarity)."""

    SHARED_ENTITY = "SHARED_ENTITY"
    """Documents mention the same named entities."""

    SHARED_KEYWORD = "SHARED_KEYWORD"
    """Documents share high-scoring keyword tokens."""

    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    """
    Documents are semantically similar beyond surface keyword overlap.
    Currently computed via topic + entity co-occurrence; future versions
    will use embedding cosine similarity.
    """


class Relationship(BaseModel):
    """
    A directed weighted edge between two documents in the recommendation graph.

    Relationships are upserted into Postgres after each document is enriched.
    The ``weight`` field is updated to the maximum observed value — the
    strongest known connection is always preserved.

    ``shared_attributes`` records the specific tokens/labels that created the
    edge, enabling agents to understand *why* two documents are connected.
    """

    source_id: UUID = Field(description="Origin document UUID.")
    target_id: UUID = Field(description="Destination document UUID.")
    relationship_type: RelationshipType = Field(
        description="The semantic axis this edge represents.",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Strength of the relationship (0–1).  Computed as Jaccard similarity "
            "over the relevant attribute sets.  Higher = more relevant."
        ),
    )
    shared_attributes: list[str] = Field(
        default_factory=list,
        description=(
            "The specific topics, entity texts, or keywords that created this edge.  "
            "Exposed in NeighborPreview.shared_attributes so agents know why two "
            "documents are connected."
        ),
    )


class ContentMetadata(BaseModel):
    """
    Enriched metadata produced by the metadata processing pipeline.

    This model is used internally by RelationshipBuilder to compute graph edges.
    It mirrors the enrichment fields on ContentDocument but is structured for
    relationship computation rather than general document representation.
    """

    document_id: UUID = Field(description="Foreign key to ContentDocument.id.")
    document_type: str | None = Field(
        default=None,
        description="Document format classification (mirrors ContentDocument.document_type).",
    )
    topics: list[str] = Field(default_factory=list)
    themes: list[str] = Field(
        default_factory=list,
        description="Higher-level thematic concepts (mirrors ContentDocument.themes).",
    )
    keywords: list[str] = Field(default_factory=list)
    entities: list[EntityTag] = Field(default_factory=list)
    sentiment: float | None = None
    enrichment_quality_score: int | None = Field(
        default=None,
        description="Quality score (0–100) assigned by the enrichment backend.",
    )
    tone: str | None = Field(
        default=None,
        description="Qualitative tone descriptor assigned by the enrichment backend.",
    )
    language: str = Field(default="en", description="ISO 639-1 language code.")
    content_hash: str = Field(
        default="",
        description=(
            "SHA-256 hex digest of the document body.  Used for deduplication "
            "and change detection on re-ingestion."
        ),
    )
