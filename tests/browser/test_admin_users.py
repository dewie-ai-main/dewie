"""Tests for user admin endpoints."""
from __future__ import annotations

import os

import httpx

BASE_URL = "http://localhost:10946"
API_KEY = os.environ.get("BROWSER_TEST_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY}
DEV_USER_ID = "00000000-0000-0000-0000-000000000002"


def test_users_list_returns_at_least_one():
    r = httpx.get(f"{BASE_URL}/admin/users", headers=HEADERS)
    assert r.status_code == 200
    users = r.json()
    assert len(users) >= 1, "Expected at least one user"


def test_dev_user_exists():
    r = httpx.get(f"{BASE_URL}/admin/users", headers=HEADERS)
    assert r.status_code == 200
    emails = [u["email"] for u in r.json()]
    assert "dev@dewie.ai" in emails


def test_dev_user_is_admin():
    r = httpx.get(f"{BASE_URL}/admin/users", headers=HEADERS)
    users = {u["email"]: u for u in r.json()}
    assert users["dev@dewie.ai"]["is_admin"] is True


def test_dev_user_has_password():
    r = httpx.get(f"{BASE_URL}/admin/users", headers=HEADERS)
    users = {u["email"]: u for u in r.json()}
    assert users["dev@dewie.ai"]["has_password"] is True


def test_set_user_password():
    """PATCH user password returns success."""
    r = httpx.post(
        f"{BASE_URL}/admin/users/{DEV_USER_ID}/password",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"password": "dewie-admin-2026"},  # same password — idempotent
    )
    assert r.status_code in (200, 204), f"Got {r.status_code}: {r.text}"


def test_update_user_name():
    """PATCH /admin/users/{id} updates user name."""
    r = httpx.patch(
        f"{BASE_URL}/admin/users/{DEV_USER_ID}",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"name": "Dev (Internal)"},  # restore original
    )
    assert r.status_code in (200, 204), f"Got {r.status_code}: {r.text}"
