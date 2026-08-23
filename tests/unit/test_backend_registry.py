"""Tests for dewie.enrichment.registry — BackendRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dewie.enrichment.registry import BackendRegistry


def _make_backend(name: str) -> MagicMock:
    b = MagicMock()
    b.name = name
    return b


# ── register ──────────────────────────────────────────────────────────────────


def test_register_adds_backend():
    reg = BackendRegistry()
    reg.register(_make_backend("alpha"))
    assert "alpha" in reg.names()


def test_register_duplicate_raises():
    reg = BackendRegistry()
    reg.register(_make_backend("alpha"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_make_backend("alpha"))


def test_register_multiple():
    reg = BackendRegistry()
    reg.register(_make_backend("a"))
    reg.register(_make_backend("b"))
    assert reg.names() == ["a", "b"]


# ── set_default / default ─────────────────────────────────────────────────────


def test_set_default_valid():
    reg = BackendRegistry()
    b = _make_backend("main")
    reg.register(b)
    reg.set_default("main")
    assert reg.default() is b


def test_set_default_unknown_raises():
    reg = BackendRegistry()
    with pytest.raises(KeyError, match="not registered"):
        reg.set_default("ghost")


def test_default_no_explicit_default_uses_first():
    reg = BackendRegistry()
    b1 = _make_backend("first")
    b2 = _make_backend("second")
    reg.register(b1)
    reg.register(b2)
    assert reg.default() is b1


def test_default_empty_registry_raises():
    reg = BackendRegistry()
    with pytest.raises(RuntimeError, match="empty"):
        reg.default()


# ── get ───────────────────────────────────────────────────────────────────────


def test_get_returns_backend():
    reg = BackendRegistry()
    b = _make_backend("foo")
    reg.register(b)
    assert reg.get("foo") is b


def test_get_unknown_raises():
    reg = BackendRegistry()
    with pytest.raises(KeyError, match="not registered"):
        reg.get("missing")


# ── names / list_backends ─────────────────────────────────────────────────────


def test_names_empty():
    reg = BackendRegistry()
    assert reg.names() == []


def test_list_backends_same_as_names():
    reg = BackendRegistry()
    reg.register(_make_backend("x"))
    assert reg.list_backends() == reg.names()


# ── backend_info ──────────────────────────────────────────────────────────────


def test_backend_info_structure():
    reg = BackendRegistry()
    b = _make_backend("mybackend")
    b.__class__ = type("HttpBackend", (), {"__doc__": "HTTP enrichment backend."})
    reg.register(b)
    info = reg.backend_info()
    assert len(info) == 1
    assert info[0]["name"] == "mybackend"
    assert "type" in info[0]
    assert "description" in info[0]


# ── from_config ───────────────────────────────────────────────────────────────


def test_from_config_empty_backends():
    mock_settings = MagicMock()
    mock_settings.enrichment_backends = "[]"
    mock_settings.enrichment_default_backend = "passthrough"

    reg = BackendRegistry.from_config(mock_settings)
    assert "passthrough" in reg.names()


def test_from_config_with_passthrough_descriptor():
    mock_settings = MagicMock()
    mock_settings.enrichment_backends = (
        '[{"type": "passthrough", "name": "noop", "response": "{}"}]'
    )
    mock_settings.enrichment_default_backend = "passthrough"

    reg = BackendRegistry.from_config(mock_settings)
    assert "passthrough" in reg.names()
    assert "noop" in reg.names()


def test_from_config_skips_spacy():
    mock_settings = MagicMock()
    mock_settings.enrichment_backends = '[{"type": "spacy", "name": "spacy_old"}]'
    mock_settings.enrichment_default_backend = "passthrough"

    reg = BackendRegistry.from_config(mock_settings)
    assert "spacy_old" not in reg.names()


def test_from_config_skips_missing_name():
    mock_settings = MagicMock()
    mock_settings.enrichment_backends = '[{"type": "passthrough"}]'
    mock_settings.enrichment_default_backend = "passthrough"

    reg = BackendRegistry.from_config(mock_settings)
    # No unnamed backend was added; passthrough (the always-added one) is there
    assert "passthrough" in reg.names()


def test_from_config_invalid_json_logs_error():
    mock_settings = MagicMock()
    mock_settings.enrichment_backends = "NOT JSON"
    mock_settings.enrichment_default_backend = "passthrough"

    # Should not raise
    reg = BackendRegistry.from_config(mock_settings)
    assert "passthrough" in reg.names()


def test_from_config_unknown_type_skipped():
    mock_settings = MagicMock()
    mock_settings.enrichment_backends = '[{"type": "alien", "name": "weird"}]'
    mock_settings.enrichment_default_backend = "passthrough"

    reg = BackendRegistry.from_config(mock_settings)
    assert "weird" not in reg.names()


def test_from_config_sets_correct_default():
    mock_settings = MagicMock()
    mock_settings.enrichment_backends = (
        '[{"type": "passthrough", "name": "custom_pt", "response": "{}"}]'
    )
    mock_settings.enrichment_default_backend = "custom_pt"

    reg = BackendRegistry.from_config(mock_settings)
    assert reg.default().name == "custom_pt"
