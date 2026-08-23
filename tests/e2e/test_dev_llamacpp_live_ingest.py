"""Optional live E2E: llama.cpp provider registration + ingest flow.

Run explicitly:
    PYTHONPATH=src pytest tests/e2e/test_dev_llamacpp_live_ingest.py -o addopts='' -v

Required keys in .env.remote-catalog.local (or process env):
    DEWIE_TEST_API_BASE=http://localhost:10946/api
    LLAMACPP_BASE_URL=http://localhost:8080/v1
    LLAMACPP_MODEL_ID=<model-id-from-llama.cpp-models-endpoint>

Optional:
    DEWIE_TEST_ADMIN_KEY=<admin-key-when-auth-enabled>
    DEWIE_TEST_SERVICE_KEY=<service-key-when-ingest-key-required>
"""

from __future__ import annotations

import os
import time
import uuid

import httpx
import pytest

from tests.e2e.conftest import load_dev_env_file

pytestmark = pytest.mark.dev_llamacpp_live

# Load persisted local config once for this optional suite.
load_dev_env_file()

DEWIE_TEST_API_BASE = os.environ.get("DEWIE_TEST_API_BASE", "").rstrip("/")
LLAMACPP_BASE_URL = os.environ.get("LLAMACPP_BASE_URL", "").rstrip("/")
LLAMACPP_MODEL_ID = os.environ.get("LLAMACPP_MODEL_ID", "").strip()
DEWIE_TEST_ADMIN_KEY = os.environ.get("DEWIE_TEST_ADMIN_KEY", "").strip()
DEWIE_TEST_SERVICE_KEY = os.environ.get("DEWIE_TEST_SERVICE_KEY", "").strip()

_REQUIRED = {
    "DEWIE_TEST_API_BASE": DEWIE_TEST_API_BASE,
    "LLAMACPP_BASE_URL": LLAMACPP_BASE_URL,
    "LLAMACPP_MODEL_ID": LLAMACPP_MODEL_ID,
}
_MISSING = [k for k, v in _REQUIRED.items() if not v]
if _MISSING:
    pytest.skip(
        "Missing required live config in .env.remote-catalog.local: " + ", ".join(_MISSING),
        allow_module_level=True,
    )


def _admin_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if DEWIE_TEST_ADMIN_KEY:
        headers["X-Admin-Key"] = DEWIE_TEST_ADMIN_KEY
    return headers


def _ingest_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if DEWIE_TEST_SERVICE_KEY:
        headers["X-Service-Key"] = DEWIE_TEST_SERVICE_KEY
    return headers


def _provider_id() -> str:
    return "llamacpp-e2e"


def _provider_payload() -> dict:
    probe_url = f"{LLAMACPP_BASE_URL}/models"
    return {
        "provider_id": _provider_id(),
        "base_url": LLAMACPP_BASE_URL,
        "api_key_env": None,
        "probe_url": probe_url,
        "probe_timeout": 10.0,
        "dynamic": True,
    }


def _preflight() -> None:
    with httpx.Client(timeout=8.0) as client:
        # Dewie must expose MCP manifest at <api_base>/mcp
        r_dewie = client.get(f"{DEWIE_TEST_API_BASE}/mcp")
        if r_dewie.status_code not in (200, 401, 403):
            pytest.skip(
                f"Dewie API not reachable at {DEWIE_TEST_API_BASE} (status={r_dewie.status_code})",
                allow_module_level=True,
            )

        # llama.cpp server should expose /models on the supplied base URL.
        r_llama = client.get(f"{LLAMACPP_BASE_URL}/models")
        if r_llama.status_code != 200:
            pytest.skip(
                f"llama.cpp not reachable at {LLAMACPP_BASE_URL}/models (status={r_llama.status_code})",
                allow_module_level=True,
            )


def _cleanup_provider_best_effort(client: httpx.Client) -> None:
    client.delete(
        f"{DEWIE_TEST_API_BASE}/admin/model-catalog/providers/{_provider_id()}",
        headers=_admin_headers(),
    )


def _register_and_refresh_provider(client: httpx.Client) -> None:
    # Ensure idempotent reruns.
    _cleanup_provider_best_effort(client)

    register_resp = client.post(
        f"{DEWIE_TEST_API_BASE}/admin/model-catalog/providers",
        headers=_admin_headers(),
        json=_provider_payload(),
    )
    assert register_resp.status_code == 200, register_resp.text

    refresh_resp = client.post(
        f"{DEWIE_TEST_API_BASE}/admin/model-catalog/providers/{_provider_id()}/refresh",
        headers=_admin_headers(),
    )
    assert refresh_resp.status_code == 200, refresh_resp.text


def _assert_model_visible(client: httpx.Client) -> None:
    catalog_resp = client.get(
        f"{DEWIE_TEST_API_BASE}/admin/model-catalog",
        headers=_admin_headers(),
        params={"context": "admin", "include_hidden": "true"},
    )
    assert catalog_resp.status_code == 200, catalog_resp.text
    payload = catalog_resp.json()

    models = payload.get("models_by_provider", {}).get(_provider_id(), [])
    model_ids = {m.get("id") for m in models if isinstance(m, dict)}
    assert LLAMACPP_MODEL_ID in model_ids, (
        f"Expected model {LLAMACPP_MODEL_ID!r} in provider {_provider_id()!r}; "
        f"found={sorted(x for x in model_ids if x)}"
    )


def _ingest_live_doc(client: httpx.Client) -> str:
    unique = str(uuid.uuid4())[:8]
    title = f"llamacpp live ingest {unique}"
    url = f"https://example.com/llamacpp-live-{unique}"
    body = (
        "This is a deterministic live ingestion test body for Dewie with llama cpp backend. "
        "It contains unique tokens: volcano-sensor-grid, basalt-monitoring, plume-telemetry. "
        * 8
    )

    ingest_resp = client.post(
        f"{DEWIE_TEST_API_BASE}/ingest",
        headers=_ingest_headers(),
        json={
            "url": url,
            "title": title,
            "body": body,
            "enrichment_provider": _provider_id(),
            "enrichment_model": LLAMACPP_MODEL_ID,
        },
    )
    assert ingest_resp.status_code == 202, ingest_resp.text

    accepted = ingest_resp.json().get("accepted", [])
    assert accepted, ingest_resp.text
    return str(accepted[0])


def _wait_until_searchable(client: httpx.Client, timeout_secs: int = 45) -> dict:
    deadline = time.monotonic() + timeout_secs
    last = None
    while time.monotonic() < deadline:
        r = client.post(
            f"{DEWIE_TEST_API_BASE}/mcp",
            headers={"Content-Type": "application/json"},
            json={
                "tool": "search_corpus",
                "input": {"query": "basalt-monitoring plume-telemetry", "limit": 5},
            },
        )
        if r.status_code == 200:
            payload = r.json()
            last = payload
            results = payload.get("content", {}).get("results", [])
            if results:
                return payload
        time.sleep(1.0)

    raise AssertionError(f"Ingested document was not searchable within timeout. last={last}")


def test_llamacpp_live_register_and_ingest_flow() -> None:
    """Register llama.cpp provider, ingest with override, and verify retrieval."""
    _preflight()

    with httpx.Client(timeout=20.0) as client:
        _register_and_refresh_provider(client)
        _assert_model_visible(client)
        _ingest_live_doc(client)
        result = _wait_until_searchable(client)

        # Contract checks
        assert result.get("tool") == "search_corpus"
        assert "answers_questions" not in str(result)

        # Cleanup best-effort
        _cleanup_provider_best_effort(client)
