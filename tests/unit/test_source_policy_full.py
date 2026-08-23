"""Tests for dewie.source_policy."""

from __future__ import annotations

# ── get_policy ────────────────────────────────────────────────────────────────


def test_get_policy_known_owned():
    from dewie.source_policy import get_policy

    policy = get_policy("wikipedia")
    assert policy.store_body is True
    assert policy.rank_tier == 1


def test_get_policy_known_news():
    from dewie.source_policy import get_policy

    # reuters is in news_tlds heuristic
    policy = get_policy("reuters")
    assert policy.store_body is False


def test_get_policy_unknown_defaults_public():
    from dewie.source_policy import get_policy

    policy = get_policy("some-unknown-blog.io")
    assert isinstance(policy.store_body, bool)


def test_get_policy_news_heuristic_bloomberg():
    from dewie.source_policy import get_policy

    policy = get_policy("bloomberg-news")
    assert policy.store_body is False


def test_get_policy_news_heuristic_cnbc():
    from dewie.source_policy import get_policy

    policy = get_policy("cnbc-finance")
    assert policy.store_body is False


# ── should_store_body ─────────────────────────────────────────────────────────


def test_should_store_body_owned():
    from dewie.source_policy import should_store_body

    assert should_store_body("wikipedia") is True


def test_should_store_body_news():
    from dewie.source_policy import should_store_body

    assert should_store_body("reuters") is False


def test_should_store_body_unknown():
    from dewie.source_policy import should_store_body

    result = should_store_body("unknown-source")
    assert isinstance(result, bool)


# ── rank_tier ─────────────────────────────────────────────────────────────────


def test_rank_tier_owned():
    from dewie.source_policy import rank_tier

    assert rank_tier("wikipedia") == 1


def test_rank_tier_unknown_default():
    from dewie.source_policy import rank_tier

    result = rank_tier("unknown-blog")
    assert result in (1, 2, 3)


# ── SourcePolicy dataclass ────────────────────────────────────────────────────


def test_source_policy_dataclass():
    from dewie.source_policy import SourcePolicy

    p = SourcePolicy(store_body=True, source_type="owned", rank_tier=1, description="Test")
    assert p.store_body is True
    assert p.rank_tier == 1


# ── SOURCE_POLICIES dict ──────────────────────────────────────────────────────


def test_source_policies_not_empty():
    from dewie.source_policy import SOURCE_POLICIES

    assert len(SOURCE_POLICIES) > 0


def test_source_policies_values_valid():
    from dewie.source_policy import SOURCE_POLICIES

    for source, policy in SOURCE_POLICIES.items():
        assert isinstance(policy.store_body, bool)
        assert policy.source_type in ("owned", "licensed", "public")
        assert policy.rank_tier in (1, 2, 3)
