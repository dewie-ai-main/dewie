"""Tests for pure helper functions in dewie.api.routes.capabilities."""

from __future__ import annotations


def _make_cluster(label, confidence, doc_count, aqs=None):
    return {
        "label": label,
        "coverage_confidence": confidence,
        "doc_count": doc_count,
        "sample_aqs": aqs or [],
    }


# ── _build_gap_signals ────────────────────────────────────────────────────────


def test_build_gap_signals_empty_clusters():
    from dewie.api.routes.capabilities import _build_gap_signals

    signals = _build_gap_signals([], "machine learning")
    assert len(signals) == 1
    assert "No documents" in signals[0]


def test_build_gap_signals_low_confidence():
    from dewie.api.routes.capabilities import _build_gap_signals

    clusters = [_make_cluster("AI ethics", 0.2, 3)]
    signals = _build_gap_signals(clusters, "AI")
    assert len(signals) == 1
    assert "AI ethics" in signals[0]


def test_build_gap_signals_high_confidence_no_signal():
    from dewie.api.routes.capabilities import _build_gap_signals

    clusters = [_make_cluster("Machine Learning", 0.9, 50)]
    signals = _build_gap_signals(clusters, "ML")
    assert signals == []


def test_build_gap_signals_caps_at_three():
    from dewie.api.routes.capabilities import _build_gap_signals

    clusters = [_make_cluster(f"topic-{i}", 0.1, 1) for i in range(10)]
    signals = _build_gap_signals(clusters, "test")
    assert len(signals) <= 3


# ── _suggested_first_query ────────────────────────────────────────────────────


def test_suggested_first_query_empty():
    from dewie.api.routes.capabilities import _suggested_first_query

    result = _suggested_first_query([])
    assert result is None


def test_suggested_first_query_picks_best():
    from dewie.api.routes.capabilities import _suggested_first_query

    clusters = [
        _make_cluster("low", 0.2, 5, aqs=["low question"]),
        _make_cluster("high", 0.9, 20, aqs=["What is machine learning?"]),
    ]
    result = _suggested_first_query(clusters)
    assert result == "What is machine learning?"


def test_suggested_first_query_falls_back_to_label():
    from dewie.api.routes.capabilities import _suggested_first_query

    clusters = [_make_cluster("AI Ethics", 0.8, 10, aqs=[])]
    result = _suggested_first_query(clusters)
    assert result == "AI Ethics"


def test_suggested_first_query_none_aqs():
    from dewie.api.routes.capabilities import _suggested_first_query

    cluster = {"label": "ML", "coverage_confidence": 0.7, "doc_count": 15, "sample_aqs": None}
    result = _suggested_first_query([cluster])
    assert result == "ML"


# ── _coverage_signal ──────────────────────────────────────────────────────────


def test_coverage_signal_sparse_empty():
    from dewie.api.routes.capabilities import _coverage_signal

    assert _coverage_signal([]) == "sparse"


def test_coverage_signal_deep():
    from dewie.api.routes.capabilities import _coverage_signal

    clusters = [_make_cluster("ML", 0.8, 60), _make_cluster("DL", 0.7, 60)]
    assert _coverage_signal(clusters) == "deep"


def test_coverage_signal_moderate_by_confidence():
    from dewie.api.routes.capabilities import _coverage_signal

    clusters = [_make_cluster("ML", 0.4, 5)]
    assert _coverage_signal(clusters) == "moderate"


def test_coverage_signal_moderate_by_doc_count():
    from dewie.api.routes.capabilities import _coverage_signal

    clusters = [_make_cluster("ML", 0.1, 30)]
    assert _coverage_signal(clusters) == "moderate"


def test_coverage_signal_sparse():
    from dewie.api.routes.capabilities import _coverage_signal

    clusters = [_make_cluster("ML", 0.1, 5)]
    assert _coverage_signal(clusters) == "sparse"


# ── Pydantic models ───────────────────────────────────────────────────────────


def test_probe_request_model():
    from dewie.api.routes.capabilities import ProbeRequest

    req = ProbeRequest(context="machine learning")
    assert req.context == "machine learning"


def test_probe_response_model():
    from dewie.api.routes.capabilities import ProbeResponse

    resp = ProbeResponse(
        context="ml",
        coverage_signal="deep",
        total_matching_docs=100,
        clusters=[],
        gap_signals=[],
        suggested_first_query=None,
    )
    assert resp.coverage_signal == "deep"
