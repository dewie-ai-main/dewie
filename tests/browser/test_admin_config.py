"""Tests for model/provider config admin endpoints."""
from __future__ import annotations

import os

import httpx

BASE_URL = "http://localhost:10946"
API_KEY = os.environ.get("BROWSER_TEST_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def test_get_config_returns_values():
    r = httpx.get(f"{BASE_URL}/admin/config", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert "values" in data
    assert isinstance(data["values"], list)
    assert len(data["values"]) > 0


def test_config_has_expected_keys():
    r = httpx.get(f"{BASE_URL}/admin/config", headers=HEADERS)
    values = {v["key"]: v for v in r.json()["values"]}
    for key in ("chat_model_aq", "chat_server_aq", "embed_model", "embed_server"):
        assert key in values, f"Missing config key: {key}"


def test_patch_config_saves_value():
    """PATCH /admin/config saves a new value and returns ok=true."""
    r = httpx.patch(
        f"{BASE_URL}/admin/config",
        headers=HEADERS,
        json={"path": "chat_model_aq", "value": "gpt-4o", "value_type": "str"},
    )
    assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
    data = r.json()
    assert data["ok"] is True
    assert data["path"] == "chat_model_aq"
    assert data["value"] == "gpt-4o"


def test_patch_config_value_persists():
    """After patching, GET /admin/config returns the updated value."""
    sentinel = "test-model-e2e-check"
    httpx.patch(
        f"{BASE_URL}/admin/config",
        headers=HEADERS,
        json={"path": "chat_model_aq", "value": sentinel, "value_type": "str"},
    )
    r = httpx.get(f"{BASE_URL}/admin/config", headers=HEADERS)
    values = {v["key"]: v for v in r.json()["values"]}
    assert values["chat_model_aq"]["value"] == sentinel

    # Restore
    httpx.patch(
        f"{BASE_URL}/admin/config",
        headers=HEADERS,
        json={"path": "chat_model_aq", "value": "gpt-4o", "value_type": "str"},
    )


def test_patch_unknown_config_key_returns_error():
    """Patching a key that doesn't exist should return an error, not 200."""
    r = httpx.patch(
        f"{BASE_URL}/admin/config",
        headers=HEADERS,
        json={"path": "does_not_exist_xyz", "value": "whatever", "value_type": "str"},
    )
    assert r.status_code in (400, 422), f"Expected 400/422, got {r.status_code}: {r.text}"
