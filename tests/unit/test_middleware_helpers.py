"""Tests for pure helper functions in dewie.api.middleware."""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_request(forwarded_for=None, api_key=None, remote_addr="1.2.3.4"):
    req = MagicMock()
    headers = {}
    if forwarded_for:
        headers["X-Forwarded-For"] = forwarded_for
    if api_key:
        headers["X-API-Key"] = api_key
    req.headers = headers
    req.client = MagicMock()
    req.client.host = remote_addr
    return req


# ── _real_ip ──────────────────────────────────────────────────────────────────


def test_real_ip_no_forwarded_for():
    from dewie.api.middleware import _real_ip

    req = _make_request(remote_addr="10.0.0.1")
    # Without X-Forwarded-For, falls back to get_remote_address
    # which reads req.client.host
    ip = _real_ip(req)
    assert isinstance(ip, str)


def test_real_ip_with_forwarded_for_single():
    from dewie.api.middleware import _real_ip

    # XFF trusted only when client is a private/loopback proxy
    req = _make_request(forwarded_for="203.0.113.1", remote_addr="127.0.0.1")
    ip = _real_ip(req)
    assert ip == "203.0.113.1"


def test_real_ip_with_forwarded_for_multiple():
    from dewie.api.middleware import _real_ip

    req = _make_request(forwarded_for="203.0.113.1, 10.0.0.1, 172.16.0.1", remote_addr="10.0.0.1")
    ip = _real_ip(req)
    assert ip == "203.0.113.1"


def test_real_ip_strips_whitespace():
    from dewie.api.middleware import _real_ip

    req = _make_request(forwarded_for="  203.0.113.2 , 10.0.0.1", remote_addr="10.0.0.1")
    ip = _real_ip(req)
    assert ip == "203.0.113.2"


def test_real_ip_ignores_forwarded_for_from_public_client():
    from dewie.api.middleware import _real_ip

    # Public client sending XFF must NOT be trusted
    req = _make_request(forwarded_for="1.1.1.1", remote_addr="5.6.7.8")
    ip = _real_ip(req)
    assert ip == "5.6.7.8"


# ── _rate_limit_key ───────────────────────────────────────────────────────────


def test_rate_limit_key_no_api_key():
    from dewie.api.middleware import _rate_limit_key

    req = _make_request()
    key = _rate_limit_key(req)
    # No API key, falls back to IP
    assert isinstance(key, str)
    assert "key:" not in key


def test_rate_limit_key_with_api_key():
    from dewie.api.middleware import _rate_limit_key

    req = _make_request(api_key="ck_live_abcdefghijklmnop")
    key = _rate_limit_key(req)
    assert key.startswith("key:")
    assert "ck_live_abcde" in key


def test_rate_limit_key_truncates_to_16():
    from dewie.api.middleware import _rate_limit_key

    req = _make_request(api_key="12345678901234567890")
    key = _rate_limit_key(req)
    # Should have key: prefix + first 16 chars of api key
    assert key == "key:1234567890123456"


def test_rate_limit_key_empty_api_key():
    from dewie.api.middleware import _rate_limit_key

    req = _make_request(api_key="")
    key = _rate_limit_key(req)
    # Empty API key treated as absent
    assert "key:" not in key
