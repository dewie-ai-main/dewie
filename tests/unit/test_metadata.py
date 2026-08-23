"""
Unit tests for metadata enrichment components.

These tests do NOT require running services — spaCy is used in-process.
"""

from __future__ import annotations

import pytest

from dewie.enrichment.relationships import RelationshipBuilder, jaccard
from dewie.models.content import ContentDocument
from dewie.models.metadata import RelationshipType


class TestJaccard:
    def test_identical_sets(self):
        assert jaccard(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_disjoint_sets(self):
        assert jaccard(["a", "b"], ["c", "d"]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # |A ∩ B| = 1, |A ∪ B| = 3
        assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)

    def test_empty_sets(self):
        assert jaccard([], []) == pytest.approx(0.0)

    def test_one_empty(self):
        assert jaccard(["a"], []) == pytest.approx(0.0)


class TestRelationshipBuilder:
    def test_no_relationships_when_disjoint(
        self,
        sample_doc: ContentDocument,
        related_doc: ContentDocument,
    ):
        # Force completely different metadata
        sample_doc.topics = ["sports"]
        sample_doc.entities = ["FIFA"]
        sample_doc.keywords = ["football"]
        related_doc.topics = ["cooking"]
        related_doc.entities = ["Gordon Ramsay"]
        related_doc.keywords = ["recipe"]

        builder = RelationshipBuilder(min_weight=0.05)
        rels = builder.build(sample_doc, [related_doc])
        assert rels == []

    def test_shared_topic_creates_relationship(
        self,
        sample_doc: ContentDocument,
        related_doc: ContentDocument,
    ):
        builder = RelationshipBuilder(min_weight=0.05)
        rels = builder.build(sample_doc, [related_doc])

        types = {r.relationship_type for r in rels}
        assert RelationshipType.SHARED_TOPIC in types

    def test_self_relationship_excluded(self, sample_doc: ContentDocument):
        builder = RelationshipBuilder(min_weight=0.0)
        rels = builder.build(sample_doc, [sample_doc])
        assert rels == []

    def test_weight_is_bounded(
        self,
        sample_doc: ContentDocument,
        related_doc: ContentDocument,
    ):
        builder = RelationshipBuilder(min_weight=0.0)
        rels = builder.build(sample_doc, [related_doc])
        for rel in rels:
            assert 0.0 <= rel.weight <= 1.0

    def test_min_weight_filters_weak_relations(
        self,
        sample_doc: ContentDocument,
        related_doc: ContentDocument,
    ):
        # Only one keyword overlaps
        sample_doc.keywords = ["model", "alpha", "beta", "gamma", "delta"]
        related_doc.keywords = ["model", "epsilon", "zeta", "eta", "theta"]
        # Jaccard = 1/9 ≈ 0.11

        builder_strict = RelationshipBuilder(min_weight=0.5)
        rels = builder_strict.build(sample_doc, [related_doc])
        keyword_rels = [r for r in rels if r.relationship_type == RelationshipType.SHARED_KEYWORD]
        assert keyword_rels == []

    def test_strongest_relationship_kept_per_pair(
        self,
        sample_doc: ContentDocument,
        related_doc: ContentDocument,
    ):
        """build() should deduplicate and keep only the strongest edge per pair."""
        builder = RelationshipBuilder(min_weight=0.0)
        rels = builder.build(sample_doc, [related_doc])

        # Each (source, target) pair should appear at most once per type
        seen = set()
        for rel in rels:
            key = (str(rel.source_id), str(rel.target_id), rel.relationship_type)
            assert key not in seen, "Duplicate relationship type per pair"
            seen.add(key)
