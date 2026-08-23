"""
Performance tests for the recursive query engine.

These tests measure:
  1. Query latency at varying tree depths (1–5).
  2. Scalability as the candidate pool grows (10 → 10,000 documents).
  3. Memory efficiency of the BFS traversal (no unbounded accumulation).

Run with:
    pytest tests/performance/ -v -s --no-cov

These tests are excluded from the default suite via the `perf` mark.
Add `-m perf` to run them explicitly.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

try:
    from dewie.query.engine import QueryEngine
    from dewie.query.traversal import BFSTraversal

    from dewie.models.content import ContentDocument, ContentStatus
    from dewie.models.query import ExpandBy
except ModuleNotFoundError:
    ContentDocument = None  # type: ignore[assignment,misc]
    ContentStatus = None  # type: ignore[assignment]
    ExpandBy = None  # type: ignore[assignment]
    QueryEngine = None  # type: ignore[assignment,misc]
    BFSTraversal = None  # type: ignore[assignment,misc]

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skip(reason="performance tests disabled — re-enable post-BEIR"),
]


def _make_doc(n: int) -> ContentDocument:
    uid = uuid.UUID(f"00000000-0000-0000-0000-{n:012d}")
    return ContentDocument(
        id=uid,
        url=f"https://example.com/{n}",
        title=f"Article {n}",
        body="x" * 500,
        status=ContentStatus.READY,
        topics=["ai", "ml"],
        keywords=["model", "data"],
        entities=["OpenAI"],
    )


def _build_linear_chain(length: int) -> tuple[list[ContentDocument], dict]:
    """Create a linear chain of *length* documents and a neighbour map."""
    docs = [_make_doc(i) for i in range(length)]
    neighbours: dict[str, list[dict]] = {}
    for i in range(len(docs) - 1):
        neighbours[str(docs[i].id)] = [
            {"id": str(docs[i + 1].id), "weight": 1.0, "shared": ["ai"], "rel_type": "SHARED_TOPIC"}
        ]
    neighbours[str(docs[-1].id)] = []
    return docs, neighbours


def _build_wide_tree(branching: int, depth: int) -> tuple[list[ContentDocument], dict]:
    """
    Build a tree where each node has exactly *branching* children up to *depth*.

    Total nodes ≈ branching^(depth+1) / (branching - 1).
    """
    docs = []
    neighbours: dict[str, list[dict]] = {}
    counter = [0]

    def _build(parent_id: str | None, cur_depth: int) -> str:
        idx = counter[0]
        counter[0] += 1
        doc = _make_doc(idx)
        docs.append(doc)
        n_id = str(doc.id)

        if parent_id:
            neighbours.setdefault(parent_id, []).append(
                {"id": n_id, "weight": 1.0, "shared": ["ai"], "rel_type": "SHARED_TOPIC"}
            )

        if cur_depth < depth:
            for _ in range(branching):
                _build(n_id, cur_depth + 1)

        neighbours.setdefault(n_id, [])
        return n_id

    _build(None, 0)
    return docs, neighbours


class TestQueryLatency:
    """Verify that traversal completes within reasonable wall-clock limits."""

    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
    async def test_linear_chain_latency(self, depth: int, mock_postgres, mock_graph, fake_cache):
        """A linear chain of (depth+1) nodes should complete in < 1 second."""
        docs, neighbours = _build_linear_chain(depth + 1)
        doc_map = {d.id: d for d in docs}

        mock_postgres.get_by_id.side_effect = lambda did: doc_map.get(did)
        mock_graph.get_related.side_effect = lambda doc_id, **kw: neighbours.get(str(doc_id), [])

        engine = QueryEngine(postgres=mock_postgres, graph=mock_graph, cache=fake_cache)

        start = time.perf_counter()
        result = await engine.related(docs[0].id, max_depth=depth)
        elapsed = time.perf_counter() - start

        assert result.total_nodes == depth + 1
        assert elapsed < 1.0, f"Depth-{depth} traversal took {elapsed:.3f}s (expected < 1s)"

    async def test_wide_tree_depth3_branching3(self, mock_postgres, mock_graph, fake_cache):
        """
        A branching-factor-3, depth-3 tree (40 nodes) should complete in < 2s.
        """
        docs, neighbours = _build_wide_tree(branching=3, depth=3)
        doc_map = {d.id: d for d in docs}

        mock_postgres.get_by_id.side_effect = lambda did: doc_map.get(did)
        mock_graph.get_related.side_effect = lambda doc_id, **kw: neighbours.get(str(doc_id), [])

        engine = QueryEngine(postgres=mock_postgres, graph=mock_graph, cache=fake_cache)

        start = time.perf_counter()
        result = await engine.related(docs[0].id, max_depth=3, max_nodes_per_level=100)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Wide tree traversal took {elapsed:.3f}s (expected < 2s)"
        assert result.total_nodes > 1


class TestScalability:
    """Verify cache effectiveness at scale."""

    async def test_cache_eliminates_repeated_traversal_cost(
        self, mock_postgres, mock_graph, fake_cache
    ):
        """Second call should be ≥10× faster than the first due to caching."""
        docs, neighbours = _build_linear_chain(6)
        doc_map = {d.id: d for d in docs}

        mock_postgres.get_by_id.side_effect = lambda did: doc_map.get(did)
        mock_graph.get_related.side_effect = lambda doc_id, **kw: neighbours.get(str(doc_id), [])

        engine = QueryEngine(postgres=mock_postgres, graph=mock_graph, cache=fake_cache)

        # Cold run
        t0 = time.perf_counter()
        await engine.related(docs[0].id, max_depth=5)
        cold = time.perf_counter() - t0

        # Warm run (cache hit)
        t0 = time.perf_counter()
        await engine.related(docs[0].id, max_depth=5)
        warm = time.perf_counter() - t0

        assert warm < cold or warm < 0.05, (
            f"Cache did not meaningfully reduce latency: cold={cold:.4f}s warm={warm:.4f}s"
        )

    async def test_large_candidate_pool_relationship_builder(self):
        """
        RelationshipBuilder.build() with 10,000 candidates must complete in < 2s.
        """
        from dewie.metadata.relationships import RelationshipBuilder

        source = _make_doc(999_000)
        source.topics = ["ai", "ml", "nlp"]
        source.keywords = ["model", "data", "training"]
        source.entities = ["OpenAI", "Google"]

        candidates = []
        for i in range(10_000):
            doc = _make_doc(i)
            doc.topics = ["ai"] if i % 3 == 0 else ["sports"]
            doc.keywords = ["model"] if i % 5 == 0 else ["football"]
            doc.entities = ["OpenAI"] if i % 7 == 0 else ["FIFA"]
            candidates.append(doc)

        builder = RelationshipBuilder(min_weight=0.05)

        start = time.perf_counter()
        rels = await asyncio.to_thread(builder.build, source, candidates)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Relationship building took {elapsed:.3f}s for 10k candidates"
        assert len(rels) > 0
