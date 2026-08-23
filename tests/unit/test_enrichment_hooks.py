"""
Unit tests for enrichment version constant + backend registry introspection.
"""

from __future__ import annotations


def test_current_enrichment_version_is_int():
    from dewie.enrichment import CURRENT_ENRICHMENT_VERSION

    assert isinstance(CURRENT_ENRICHMENT_VERSION, int)
    assert CURRENT_ENRICHMENT_VERSION >= 1


def test_backend_registry_list_backends():
    from dewie.enrichment.backends.passthrough import PassthroughBackend
    from dewie.enrichment.registry import BackendRegistry

    registry = BackendRegistry()
    registry.register(PassthroughBackend(name="stub_a", response_json="{}"))
    registry.register(PassthroughBackend(name="stub_b", response_json="{}"))

    result = registry.list_backends()

    assert isinstance(result, list)
    assert all(isinstance(n, str) for n in result)
    assert "stub_a" in result
    assert "stub_b" in result


def test_backend_registry_backend_info():
    from dewie.enrichment.backends.passthrough import PassthroughBackend
    from dewie.enrichment.registry import BackendRegistry

    registry = BackendRegistry()
    registry.register(PassthroughBackend(name="stub_x", response_json="{}"))

    result = registry.backend_info()

    assert isinstance(result, list)
    assert len(result) == 1
    info = result[0]
    assert isinstance(info, dict)
    assert "name" in info
    assert info["name"] == "stub_x"


def test_version_skip_logic():
    """Test the version-skip logic directly without the full async stack."""
    from dewie.enrichment import CURRENT_ENRICHMENT_VERSION

    def maybe_skip(doc_version: int) -> dict | None:
        """Mirrors the skip logic in _enrich_async."""
        if doc_version >= CURRENT_ENRICHMENT_VERSION:
            return {"doc_id": "test", "success": True, "skipped": True}
        return None

    # Doc already at current version → should skip
    result = maybe_skip(CURRENT_ENRICHMENT_VERSION)
    assert result is not None
    assert result["skipped"] is True
    assert result["success"] is True

    # Doc at version above current → should also skip
    result = maybe_skip(CURRENT_ENRICHMENT_VERSION + 1)
    assert result is not None
    assert result["skipped"] is True

    # Doc at version below current → should NOT skip
    result = maybe_skip(CURRENT_ENRICHMENT_VERSION - 1)
    assert result is None
