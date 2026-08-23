# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Relationship builder: compares two ContentDocuments and creates typed edges.

Called after metadata enrichment.  For each new document, relationships
are computed against a candidate set fetched from PostgreSQL (documents
sharing at least one metadata attribute).

Weight formula (Jaccard similarity over sets):
    weight = |A ∩ B| / |A ∪ B|
"""

from __future__ import annotations

from dewie.models.content import ContentDocument
from dewie.models.metadata import Relationship, RelationshipType


def jaccard(a: list[str], b: list[str]) -> float:
    """
    Compute Jaccard similarity between two token lists.

    Returns 0.0 if both lists are empty.
    """
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


class RelationshipBuilder:
    """
    Compares a source document against a collection of candidates and
    emits Relationship objects for all pairs exceeding the weight threshold.
    """

    def __init__(self, min_weight: float = 0.05) -> None:
        self._min_weight = min_weight

    def build(
        self, source: ContentDocument, candidates: list[ContentDocument]
    ) -> list[Relationship]:
        """Compare *source* against every document in *candidates*."""
        relationships: list[Relationship] = []

        for candidate in candidates:
            if candidate.id == source.id:
                continue
            rels = self._compare(source, candidate)
            relationships.extend(rels)

        best: dict[tuple, Relationship] = {}
        for rel in relationships:
            key = (str(rel.source_id), str(rel.target_id))
            if key not in best or rel.weight > best[key].weight:
                best[key] = rel

        return sorted(best.values(), key=lambda r: r.weight, reverse=True)

    def _compare(self, source: ContentDocument, target: ContentDocument) -> list[Relationship]:
        """Emit relationships for each matching dimension."""
        results: list[Relationship] = []

        for rel_type, src_attrs, tgt_attrs in [
            (RelationshipType.SHARED_TOPIC, source.topics, target.topics),
            (RelationshipType.SHARED_ENTITY, source.entities, target.entities),
            (RelationshipType.SHARED_KEYWORD, source.keywords, target.keywords),
        ]:
            weight = jaccard(src_attrs, tgt_attrs)
            if weight >= self._min_weight:
                shared = list(set(src_attrs) & set(tgt_attrs))
                results.append(
                    Relationship(
                        source_id=source.id,
                        target_id=target.id,
                        relationship_type=rel_type,
                        weight=round(weight, 4),
                        shared_attributes=shared,
                    )
                )
        return results
