"""Tests for dewie.enrichment.router — EnrichmentRouter + _resolve_backend_name."""

from __future__ import annotations

from unittest.mock import MagicMock

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_doc(body: str = "x" * 100, document_type: str | None = None) -> MagicMock:
    doc = MagicMock()
    doc.body = body
    doc.document_type = document_type
    doc.url = "https://example.com/article"
    return doc


def _make_registry(default_name: str = "default_backend") -> MagicMock:
    registry = MagicMock()
    registry.default.return_value = MagicMock(name=default_name)
    registry.get.side_effect = lambda name: MagicMock(name=name)
    return registry


# ── _resolve_backend_name ─────────────────────────────────────────────────────


def test_resolve_backend_name_use_backend():
    from dewie.enrichment.router import _resolve_backend_name

    rule = {"use_backend": "ollama_3b", "if_body_longer_than": 5000}
    assert _resolve_backend_name(rule) == "ollama_3b"


def test_resolve_backend_name_default():
    from dewie.enrichment.router import _resolve_backend_name

    rule = {"default": "gpt-4o-mini"}
    assert _resolve_backend_name(rule) == "gpt-4o-mini"


def test_resolve_backend_name_none_when_no_key():
    from dewie.enrichment.router import _resolve_backend_name

    rule = {"if_body_longer_than": 5000}  # missing both use_backend and default
    assert _resolve_backend_name(rule) is None


# ── EnrichmentRouter.select ───────────────────────────────────────────────────


def test_select_no_rules_returns_default():
    """With no rules, falls through to registry.default()."""
    from dewie.enrichment.router import EnrichmentRouter

    registry = _make_registry()
    router = EnrichmentRouter(registry=registry, rules=[])
    doc = _make_doc()
    backend = router.select(doc)
    registry.default.assert_called_once()


def test_select_body_shorter_than_matches():
    """if_body_shorter_than matches short documents."""
    from dewie.enrichment.router import EnrichmentRouter

    registry = _make_registry()
    rules = [{"if_body_shorter_than": 500, "use_backend": "tiny_model"}]
    router = EnrichmentRouter(registry=registry, rules=rules)
    doc = _make_doc(body="short")
    router.select(doc)
    registry.get.assert_called_with("tiny_model")


def test_select_body_shorter_than_not_matched():
    """if_body_shorter_than skips long documents."""
    from dewie.enrichment.router import EnrichmentRouter

    registry = _make_registry()
    rules = [{"if_body_shorter_than": 50, "use_backend": "tiny_model"}]
    router = EnrichmentRouter(registry=registry, rules=rules)
    doc = _make_doc(body="x" * 200)  # 200 > 50 → no match
    router.select(doc)
    registry.default.assert_called_once()  # Falls through to default


def test_select_body_longer_than_matches():
    """if_body_longer_than matches long documents."""
    from dewie.enrichment.router import EnrichmentRouter

    registry = _make_registry()
    rules = [{"if_body_longer_than": 100, "use_backend": "big_model"}]
    router = EnrichmentRouter(registry=registry, rules=rules)
    doc = _make_doc(body="x" * 500)
    router.select(doc)
    registry.get.assert_called_with("big_model")


def test_select_body_longer_than_not_matched():
    """if_body_longer_than skips short documents."""
    from dewie.enrichment.router import EnrichmentRouter

    registry = _make_registry()
    rules = [{"if_body_longer_than": 1000, "use_backend": "big_model"}]
    router = EnrichmentRouter(registry=registry, rules=rules)
    doc = _make_doc(body="short body")
    router.select(doc)
    registry.default.assert_called_once()


def test_select_boundary_body_longer_than_exact():
    """if_body_longer_than triggers at exactly the threshold (>= semantics)."""
    from dewie.enrichment.router import EnrichmentRouter

    registry = _make_registry()
    rules = [{"if_body_longer_than": 100, "use_backend": "big_model"}]
    router = EnrichmentRouter(registry=registry, rules=rules)
    doc = _make_doc(body="x" * 100)  # exactly 100 → matches (>=)
    router.select(doc)
    registry.get.assert_called_with("big_model")


def test_select_document_type_matches():
    """if_document_type matches by exact document_type string."""
    from dewie.enrichment.router import EnrichmentRouter

    registry = _make_registry()
    rules = [{"if_document_type": "research_paper", "use_backend": "academic_model"}]
    router = EnrichmentRouter(registry=registry, rules=rules)
    doc = _make_doc(document_type="research_paper")
    router.select(doc)
    registry.get.assert_called_with("academic_model")


def test_select_document_type_not_matched_when_none():
    """if_document_type skips docs with document_type=None."""
    from dewie.enrichment.router import EnrichmentRouter

    registry = _make_registry()
    rules = [{"if_document_type": "research_paper", "use_backend": "academic_model"}]
    router = EnrichmentRouter(registry=registry, rules=rules)
    doc = _make_doc(document_type=None)
    router.select(doc)
    registry.default.assert_called_once()


def test_select_document_type_not_matched_wrong_type():
    """if_document_type skips docs with different document_type."""
    from dewie.enrichment.router import EnrichmentRouter

    registry = _make_registry()
    rules = [{"if_document_type": "research_paper", "use_backend": "academic_model"}]
    router = EnrichmentRouter(registry=registry, rules=rules)
    doc = _make_doc(document_type="blog_post")
    router.select(doc)
    registry.default.assert_called_once()


def test_select_default_rule_always_matches():
    """default rule always matches and short-circuits further evaluation."""
    from dewie.enrichment.router import EnrichmentRouter

    registry = _make_registry()
    rules = [{"default": "gpt-4o-mini"}]
    router = EnrichmentRouter(registry=registry, rules=rules)
    doc = _make_doc()
    router.select(doc)
    registry.get.assert_called_with("gpt-4o-mini")


def test_select_first_matching_rule_wins():
    """Rules are evaluated in order — first match wins."""
    from dewie.enrichment.router import EnrichmentRouter

    registry = _make_registry()
    rules = [
        {"if_body_shorter_than": 200, "use_backend": "small_model"},
        {"default": "default_model"},
    ]
    router = EnrichmentRouter(registry=registry, rules=rules)
    doc = _make_doc(body="short")
    router.select(doc)
    registry.get.assert_called_with("small_model")


def test_select_falls_through_to_later_rule():
    """When first rule doesn't match, subsequent rules are tried."""
    from dewie.enrichment.router import EnrichmentRouter

    registry = _make_registry()
    rules = [
        {"if_body_shorter_than": 5, "use_backend": "tiny_model"},
        {"default": "default_model"},
    ]
    router = EnrichmentRouter(registry=registry, rules=rules)
    doc = _make_doc(body="this is a longer body that won't match")
    router.select(doc)
    # tiny_model rule not matched → falls through to default rule
    registry.get.assert_called_with("default_model")


def test_select_rule_missing_backend_name_is_skipped():
    """Rules with no resolvable backend name are skipped gracefully."""
    from dewie.enrichment.router import EnrichmentRouter

    registry = _make_registry()
    # First rule has no use_backend or default key → skipped
    rules = [
        {"if_body_longer_than": 10},  # no backend name!
        {"default": "fallback"},
    ]
    router = EnrichmentRouter(registry=registry, rules=rules)
    doc = _make_doc(body="x" * 100)
    router.select(doc)
    registry.get.assert_called_with("fallback")


def test_safe_get_unknown_backend_falls_back_to_default():
    """_safe_get returns registry default when backend name is unknown."""
    from dewie.enrichment.router import EnrichmentRouter

    registry = MagicMock()
    registry.get.side_effect = KeyError("unknown_backend")
    registry.default.return_value = MagicMock(name="default")
    router = EnrichmentRouter(registry=registry)
    result = router._safe_get("unknown_backend")
    registry.default.assert_called_once()


# ── EnrichmentRouter.from_config ──────────────────────────────────────────────


def test_from_config_parses_valid_json():
    """from_config parses a valid JSON rules array from settings."""
    import json

    from dewie.enrichment.router import EnrichmentRouter

    rules_json = json.dumps([{"default": "gpt-4o-mini"}])
    settings = MagicMock()
    settings.enrichment_routing_rules = rules_json
    registry = _make_registry()
    router = EnrichmentRouter.from_config(settings, registry)
    assert len(router._rules) == 1
    assert router._rules[0]["default"] == "gpt-4o-mini"


def test_from_config_empty_json():
    """from_config with [] returns no rules."""
    from dewie.enrichment.router import EnrichmentRouter

    settings = MagicMock()
    settings.enrichment_routing_rules = "[]"
    registry = _make_registry()
    router = EnrichmentRouter.from_config(settings, registry)
    assert router._rules == []


def test_from_config_missing_attribute_returns_no_rules():
    """from_config handles missing enrichment_routing_rules attribute."""
    from dewie.enrichment.router import EnrichmentRouter

    settings = MagicMock(spec=[])  # no attributes
    registry = _make_registry()
    router = EnrichmentRouter.from_config(settings, registry)
    assert router._rules == []


def test_from_config_invalid_json_returns_no_rules():
    """from_config handles malformed JSON gracefully."""
    from dewie.enrichment.router import EnrichmentRouter

    settings = MagicMock()
    settings.enrichment_routing_rules = "{not valid json["
    registry = _make_registry()
    router = EnrichmentRouter.from_config(settings, registry)
    assert router._rules == []
