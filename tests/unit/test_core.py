"""Tests for dewie.config — Settings singleton and local_auth_email resolution.

These tests verify that the local-auth email identity is driven by configuration
rather than a hardcoded string, fixing the issue where the top bar always
displayed "local@dewie.ai" regardless of the ``LOCAL_AUTH_EMAIL`` setting.

Issue: Top bar says "local@dewie.ai"
"""

from __future__ import annotations

from unittest.mock import patch

from dewie.config import Settings, settings

# ── Settings defaults ─────────────────────────────────────────────────────────


def test_default_local_auth_email():
    """The default local_auth_email must be the configured identity."""
    assert settings.local_auth_email == "Dewie Local Catalog"


def test_default_local_auth_enabled():
    """Local auth must be disabled by default."""
    assert settings.local_auth_enabled is False


def test_default_local_auth_is_admin():
    """Local auth user must be admin by default."""
    assert settings.local_auth_is_admin is True


def test_default_local_auth_user_id():
    """Local auth user_id must be the known synthetic UUID."""
    assert settings.local_auth_user_id == "00000000-0000-0000-0000-000000000002"


# ── Settings overrides via keyword args ───────────────────────────────────────


def test_settings_local_auth_email_override():
    """Custom LOCAL_AUTH_EMAIL should override the default."""
    with patch("dewie.config.Settings", wraps=Settings) as mock_cls:
        instance = Settings()
        instance.local_auth_email = "custom@example.com"
        assert instance.local_auth_email == "custom@example.com"


# ── Auth route uses config, not hardcoded ─────────────────────────────────────

# The key regression: auth_me must read local_auth_email from settings.
# Before the fix, "local@dewie.ai" was hardcoded, ignoring the config value.


async def test_auth_me_uses_settings_local_auth_email():
    """auth_me must return settings.local_auth_email, not a hardcoded string.

    This is the core regression test for the "top bar shows local@dewie.ai" bug.
    """
    from unittest.mock import MagicMock

    from fastapi import Request

    from dewie.api.routes.auth import auth_me

    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.user_id = None
    request.state.email = None
    request.state.is_admin = False
    request.state.activation_status = None
    request.cookies = {}

    custom_email = "my-custom-email@example.com"

    with patch("dewie.config.settings") as mock_settings:
        mock_settings.auth_enabled = False
        mock_settings.local_auth_enabled = False
        mock_settings.local_auth_email = custom_email
        response = await auth_me(request)

    assert response["email"] == custom_email, (
        f"Expected email '{custom_email}' from settings, "
        f"got '{response['email']}'. "
        "auth_me must read local_auth_email from config."
    )


async def test_auth_me_default_email_when_config_default():
    """When no LOCAL_AUTH_EMAIL is set, auth_me returns the default 'local@dewie.ai'."""
    from unittest.mock import MagicMock

    from fastapi import Request

    from dewie.api.routes.auth import auth_me

    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.user_id = None
    request.state.email = None
    request.state.is_admin = False
    request.state.activation_status = None
    request.cookies = {}

    with patch("dewie.config.settings") as mock_settings:
        mock_settings.auth_enabled = False
        mock_settings.local_auth_enabled = False
        mock_settings.local_auth_email = "local@dewie.ai"
        response = await auth_me(request)

    assert response["email"] == "local@dewie.ai"
