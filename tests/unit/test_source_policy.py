"""Tests for dewie.source_policy."""

from __future__ import annotations

from dewie.source_policy import (
    SOURCE_POLICIES,
    SourcePolicy,
    get_policy,
    rank_tier,
    should_store_body,
)

# ── SourcePolicy dataclass ────────────────────────────────────────────────────


def test_source_policy_fields():
    p = SourcePolicy(store_body=True, source_type="owned", rank_tier=1, description="test")
    assert p.store_body is True
    assert p.source_type == "owned"
    assert p.rank_tier == 1
    assert p.description == "test"


def test_source_policy_default_description():
    p = SourcePolicy(store_body=False, source_type="licensed", rank_tier=3)
    assert p.description == ""


# ── SOURCE_POLICIES registry ──────────────────────────────────────────────────


def test_registry_not_empty():
    assert len(SOURCE_POLICIES) > 0


def test_tier1_sources_store_body():
    for name in ["wikipedia", "dewie-docs", "docker-docs"]:
        assert SOURCE_POLICIES[name].store_body is True
        assert SOURCE_POLICIES[name].rank_tier == 1


def test_tier3_news_no_body():
    for name in ["reuters.com", "www.bbc.com", "www.theguardian.com"]:
        assert SOURCE_POLICIES[name].store_body is False
        assert SOURCE_POLICIES[name].rank_tier == 3


def test_tier2_research_store_body():
    for name in ["huggingface.co", "lilianweng.github.io", "bair.berkeley.edu"]:
        assert SOURCE_POLICIES[name].store_body is True
        assert SOURCE_POLICIES[name].rank_tier == 2


def test_hacker_news_public_no_body():
    policy = SOURCE_POLICIES["news.ycombinator.com"]
    assert policy.store_body is False
    assert policy.source_type == "public"


# ── get_policy ────────────────────────────────────────────────────────────────


def test_get_policy_exact_match():
    policy = get_policy("reuters.com")
    assert policy.store_body is False
    assert policy.rank_tier == 3


def test_get_policy_wikipedia():
    policy = get_policy("wikipedia")
    assert policy.store_body is True
    assert policy.rank_tier == 1


def test_get_policy_unknown_defaults_to_public():
    policy = get_policy("some-random-blog.io")
    assert policy.store_body is True
    assert policy.source_type == "public"
    assert policy.rank_tier == 2


def test_get_policy_news_heuristic_reuters_in_name():
    policy = get_policy("feeds.reuters-clone.com")
    assert policy.store_body is False
    assert policy.source_type == "licensed"


def test_get_policy_news_heuristic_bloomberg():
    policy = get_policy("bloomberg-news-feed.example.com")
    assert policy.store_body is False


def test_get_policy_news_heuristic_nytimes():
    policy = get_policy("feeds.nytimes-syndication.com")
    assert policy.store_body is False


def test_get_policy_news_heuristic_news_prefix():
    policy = get_policy("news.somesite.com")
    assert policy.store_body is False


def test_get_policy_news_heuristic_slash_news():
    policy = get_policy("example.com/news")
    assert policy.store_body is False


def test_get_policy_non_news_unknown_gets_default_public():
    policy = get_policy("research.mylab.edu")
    assert policy.store_body is True
    assert policy.rank_tier == 2


# ── should_store_body ─────────────────────────────────────────────────────────


def test_should_store_body_owned():
    assert should_store_body("wikipedia") is True


def test_should_store_body_licensed_news():
    assert should_store_body("www.bbc.com") is False


def test_should_store_body_unknown():
    assert should_store_body("my-personal-blog.com") is True


# ── rank_tier ─────────────────────────────────────────────────────────────────


def test_rank_tier_tier1():
    assert rank_tier("wikipedia") == 1


def test_rank_tier_tier2():
    assert rank_tier("huggingface.co") == 2


def test_rank_tier_tier3():
    assert rank_tier("techcrunch.com") == 3


def test_rank_tier_unknown_defaults_to_2():
    assert rank_tier("unknown-source.xyz") == 2
