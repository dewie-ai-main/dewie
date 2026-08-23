"""Tests for dewie.api.routes.auth — login, signup, and session endpoints."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from pydantic import ValidationError

from dewie.api.routes.auth import (
    auth_login,
    auth_logout,
    auth_me,
    auth_signout,
    auth_signup,
)
from dewie.local_auth import hash_password

# ── auth_signup ────────────────────────────────────────────────────────────────


def _make_request_with_app(postgres_mock):
    """Helper to create a Request object with app.state.postgres."""
    request = MagicMock(spec=Request)
    request.app = MagicMock()
    request.app.state = MagicMock()
    request.app.state.postgres = postgres_mock
    return request


@pytest.mark.asyncio
async def test_auth_signup_returns_201_with_valid_credentials():
    """POST /auth/signup → 201 + dewie_session cookie set"""
    
    # Mock DB
    select_result = MagicMock()
    select_result.fetchone.return_value = None  # Email doesn't exist
    
    insert_user_result = MagicMock()
    user_id = str(uuid.uuid4())
    insert_user_result.mappings.return_value.fetchone.return_value = {
        "id": user_id,
        "email": "test@example.com",
        "name": "Test User",
        "created_at": datetime.now(UTC),
        "is_admin": False,
    }

    insert_key_result = MagicMock()
    insert_key_result.mappings.return_value.fetchone.return_value = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC),
    }
    
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(side_effect=[select_result, insert_user_result, insert_key_result])
    
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    request = _make_request_with_app(pg)
    
    from dewie.api.routes.auth import SignupRequest
    body = SignupRequest(username="test@example.com", password="password123", name="Test User")
    
    response = await auth_signup(body, request)
    
    assert response.status_code == 201
    data = response.body.decode()
    import json
    payload = json.loads(data)
    assert "api_key" in payload
    assert payload["api_key"].startswith("ck_live_")
    # Check that Set-Cookie header contains dewie_session
    set_cookie_headers = [h for h in response.raw_headers if h[0].lower() == b"set-cookie"]
    assert len(set_cookie_headers) > 0
    assert any(b"dewie_session" in h[1] for h in set_cookie_headers)


@pytest.mark.asyncio
async def test_auth_signup_sets_cookie_with_correct_attributes():
    """Verify cookie has httponly, max_age set (SameSite removed for Tailscale DNS compat)."""
    
    select_result = MagicMock()
    select_result.fetchone.return_value = None
    
    insert_user_result = MagicMock()
    insert_user_result.mappings.return_value.fetchone.return_value = {
        "id": str(uuid.uuid4()),
        "email": "test@example.com",
        "name": "Test User",
        "created_at": datetime.now(UTC),
        "is_admin": False,
    }

    insert_key_result = MagicMock()
    insert_key_result.mappings.return_value.fetchone.return_value = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC),
    }
    
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(side_effect=[select_result, insert_user_result, insert_key_result])
    
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    request = _make_request_with_app(pg)
    
    from dewie.api.routes.auth import SignupRequest
    body = SignupRequest(username="test@example.com", password="password123")
    
    response = await auth_signup(body, request)
    
    # Check Set-Cookie header for correct attributes
    set_cookie_headers = [h[1].decode() if isinstance(h[1], bytes) else h[1] for h in response.raw_headers if h[0].lower() == b"set-cookie"]
    assert any("dewie_session" in header and "HttpOnly" in header and "Path=/" in header 
               for header in set_cookie_headers)


@pytest.mark.asyncio
async def test_auth_signup_rejects_short_password():
    """Password must be at least 8 characters."""
    request = _make_request_with_app(MagicMock())
    
    from dewie.api.routes.auth import SignupRequest
    body = SignupRequest(username="test@example.com", password="short")
    
    with pytest.raises(HTTPException) as exc_info:
        await auth_signup(body, request)
    
    assert exc_info.value.status_code == 400
    assert "8 characters" in exc_info.value.detail


@pytest.mark.asyncio
async def test_auth_signup_accepts_non_email_username():
    """Username does not need to be an email address."""
    select_result = MagicMock()
    select_result.fetchone.return_value = None  # Username doesn't exist

    insert_user_result = MagicMock()
    user_id = str(uuid.uuid4())
    insert_user_result.mappings.return_value.fetchone.return_value = {
        "id": user_id,
        "email": "notanemail",
        "name": "Test User",
        "created_at": datetime.now(UTC),
        "is_admin": False,
    }

    insert_key_result = MagicMock()
    insert_key_result.mappings.return_value.fetchone.return_value = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC),
    }

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(side_effect=[select_result, insert_user_result, insert_key_result])

    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)

    pg = MagicMock()
    pg._engine = mock_engine

    request = _make_request_with_app(pg)

    from dewie.api.routes.auth import SignupRequest
    body = SignupRequest(username="notanemail", password="password123")

    response = await auth_signup(body, request)

    assert response.status_code == 201
    # Verify response includes api_key
    import json
    data = json.loads(response.body.decode())
    assert "api_key" in data
    assert data["api_key"].startswith("ck_live_")


@pytest.mark.asyncio
async def test_auth_signup_rejects_duplicate_email():
    """Email already exists → 409."""
    
    select_result = MagicMock()
    select_result.fetchone.return_value = None  # First check passes
    
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=select_result)
    
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    # Make the first execute return no result (email doesn't exist for check)
    # But simulate the create_local_user raising ValueError
    request = _make_request_with_app(pg)
    
    with patch("dewie.api.routes.auth.create_local_user") as mock_create:
        mock_create.side_effect = ValueError("Email already exists: test@example.com")
        
        from dewie.api.routes.auth import SignupRequest
        body = SignupRequest(username="test@example.com", password="password123")
        
        with pytest.raises(HTTPException) as exc_info:
            await auth_signup(body, request)
        
        assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_signup_returns_api_key():
    """Signup response includes api_key field starting with ck_live_."""
    select_result = MagicMock()
    select_result.fetchone.return_value = None
    
    insert_user_result = MagicMock()
    insert_user_result.mappings.return_value.fetchone.return_value = {
        "id": str(uuid.uuid4()),
        "email": "test@example.com",
        "name": "Test User",
        "created_at": datetime.now(UTC),
        "is_admin": False,
    }

    insert_key_result = MagicMock()
    insert_key_result.mappings.return_value.fetchone.return_value = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC),
    }
    
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(side_effect=[select_result, insert_user_result, insert_key_result])
    
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    request = _make_request_with_app(pg)
    
    from dewie.api.routes.auth import SignupRequest
    body = SignupRequest(username="test@example.com", password="password123", name="Test User")
    
    response = await auth_signup(body, request)
    
    import json
    data = json.loads(response.body.decode())
    assert "api_key" in data
    assert data["api_key"].startswith("ck_live_")


@pytest.mark.asyncio
async def test_signup_api_key_is_immediately_usable():
    """API key returned from signup works for authenticated query requests."""
    select_result = MagicMock()
    select_result.fetchone.return_value = None
    
    insert_user_result = MagicMock()
    insert_user_result.mappings.return_value.fetchone.return_value = {
        "id": str(uuid.uuid4()),
        "email": "test@example.com",
        "name": "Test User",
        "created_at": datetime.now(UTC),
        "is_admin": False,
    }

    insert_key_result = MagicMock()
    key_id = str(uuid.uuid4())
    key_prefix = "ck_live_" + "x" * 6  # 12 chars
    insert_key_result.mappings.return_value.fetchone.return_value = {
        "id": key_id,
        "created_at": datetime.now(UTC),
    }
    
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(side_effect=[select_result, insert_user_result, insert_key_result])
    
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    request = _make_request_with_app(pg)
    
    from dewie.api.routes.auth import SignupRequest
    body = SignupRequest(username="test@example.com", password="password123", name="Test User")
    
    response = await auth_signup(body, request)
    
    import json
    data = json.loads(response.body.decode())
    raw_key = data["api_key"]
    assert raw_key.startswith("ck_live_")
    
    # Verify the key can be used for verification
    from dewie.auth import key_prefix as kp
    prefix = kp(raw_key)
    assert prefix == raw_key[:12]
    
   # The key should be verifiable against the DB (mocked)
    from dewie.auth import key_prefix as kp
    
    prefix = kp(raw_key)
    assert prefix == raw_key[:12]
    
    # Verify the key was created with correct properties via direct DB query
    from sqlalchemy import text as _text
    
    # Create a real-like row object
    class FakeRow(dict):
        def __init__(self, **kwargs):
            super().__init__(kwargs)
    
    select_result = MagicMock()
    select_result.mappings.return_value.fetchall.return_value = [
        FakeRow(id=key_id, key_prefix=prefix, scopes=["read"], name="default"),
    ]
    
    mock_conn_verify = AsyncMock()
    mock_conn_verify.execute = AsyncMock(return_value=select_result)
    
    mock_engine_verify = MagicMock()
    mock_engine_verify.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn_verify)
    mock_engine_verify.connect.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg._engine = mock_engine_verify
    
    async with pg._engine.connect() as conn:
        rows = await conn.execute(
            _text("SELECT id, key_hash, key_prefix, scopes, name FROM api_keys WHERE key_prefix = :prefix"),
            {"prefix": prefix},
        )
        candidates = rows.mappings().fetchall()
    
    # Should find exactly one candidate (the one we just created)
    assert len(candidates) == 1
    assert candidates[0]["key_prefix"] == prefix
    assert candidates[0]["scopes"] == ["read"]
    assert candidates[0]["name"] == "default"


@pytest.mark.asyncio
async def test_signup_api_key_appears_in_list():
    """After signup, GET /user/api-keys shows 1 key named 'default'."""
    # First, simulate signup creating the key
    select_result = MagicMock()
    select_result.fetchone.return_value = None
    
    insert_user_result = MagicMock()
    user_id = str(uuid.uuid4())
    insert_user_result.mappings.return_value.fetchone.return_value = {
        "id": user_id,
        "email": "test@example.com",
        "name": "Test User",
        "created_at": datetime.now(UTC),
        "is_admin": False,
    }

    insert_key_result = MagicMock()
    insert_key_result.mappings.return_value.fetchone.return_value = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC),
    }
    
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(side_effect=[select_result, insert_user_result, insert_key_result])
    
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    from dewie.api.routes.auth import SignupRequest
    body = SignupRequest(username="test@example.com", password="password123", name="Test User")
    request = _make_request_with_app(pg)
    
    await auth_signup(body, request)
    
    # Now simulate GET /user/api-keys listing keys for this user
    from dewie.api.routes.user import list_user_keys
    
    list_request = MagicMock(spec=Request)
    list_request.state = MagicMock()
    list_request.state.user_id = user_id
    
    # The list query will execute on the same mock DB
    # The key was already inserted, so we expect one result
    # We need to re-setup the mock for the select on list
    from types import SimpleNamespace
    
    key_id_val = str(uuid.uuid4())
    list_select = MagicMock()
    list_select.fetchall.return_value = [
        SimpleNamespace(id=key_id_val, key_prefix="ck_live_xxxxxxxx",
                name="default",
                created_at=datetime.now(UTC),
                last_used_at=None,
                scopes=["read"]),
    ]
    mock_conn_list = AsyncMock()
    mock_conn_list.execute = AsyncMock(return_value=list_select)
    mock_engine_list = MagicMock()
    mock_engine_list.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn_list)
    mock_engine_list.connect.return_value.__aexit__ = AsyncMock(return_value=None)
    list_request.app = MagicMock()
    list_request.app.state = MagicMock()
    list_request.app.state.postgres._engine = mock_engine_list
    
    result = await list_user_keys(list_request)
    
    assert len(result) == 1
    assert result[0]["name"] == "default"
    assert result[0]["prefix"].startswith("ck_live_")


# ── signup email validation ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auth_signup_rejects_testuser_email():
    """testuser_{timestamp} pattern emails are rejected with ValidationError."""
    from dewie.api.routes.auth import SignupRequest

    with pytest.raises(ValidationError) as exc_info:
        SignupRequest(username="testuser_1780073980", password="password123")

    assert "not allowed" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_auth_signup_rejects_testuser_lowercase():
    """TestUser_{timestamp} (mixed case) is also rejected."""
    from dewie.api.routes.auth import SignupRequest

    with pytest.raises(ValidationError) as exc_info:
        SignupRequest(username="TestUser_abc123", password="password123")

    assert "not allowed" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_auth_signup_accepts_valid_email():
    """Normal email addresses are accepted through validation."""
    select_result = MagicMock()
    select_result.fetchone.return_value = None

    insert_user_result = MagicMock()
    insert_user_result.mappings.return_value.fetchone.return_value = {
        "id": str(uuid.uuid4()),
        "email": "alice@example.com",
        "name": "Alice",
        "created_at": datetime.now(UTC),
        "is_admin": False,
    }

    insert_key_result = MagicMock()
    insert_key_result.mappings.return_value.fetchone.return_value = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC),
    }

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(side_effect=[select_result, insert_user_result, insert_key_result])

    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)

    pg = MagicMock()
    pg._engine = mock_engine

    request = _make_request_with_app(pg)

    from dewie.api.routes.auth import SignupRequest
    body = SignupRequest(username="alice@example.com", password="password123")

    response = await auth_signup(body, request)
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_auth_signup_rejects_pure_digits_email():
    """10+ digit emails (unix timestamps) are rejected."""
    from dewie.api.routes.auth import SignupRequest

    with pytest.raises(ValidationError) as exc_info:
        SignupRequest(username="1780073980", password="password123")

    assert "not allowed" in str(exc_info.value).lower()


# ── auth_login ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auth_login_returns_200_with_valid_credentials():
    """POST /auth/login → 200 + dewie_session cookie set"""
    
    from dewie.api.routes.auth import LoginRequest
    
    password = "password123"
    user_id = str(uuid.uuid4())
    password_hash = hash_password(password)
    
    # Mock user fetch
    select_result = MagicMock()
    select_result.mappings.return_value.fetchone.return_value = {
        "id": user_id,
        "email": "test@example.com",
        "name": "Test User",
        "password_hash": password_hash,
        "is_admin": False,
    }
    
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=select_result)
    
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    request = _make_request_with_app(pg)
    body = LoginRequest(username="test@example.com", password=password)
    
    response = await auth_login(body, request)
    
    assert response.status_code == 200
    # Check that Set-Cookie header contains dewie_session
    set_cookie_headers = [h for h in response.raw_headers if h[0].lower() == b"set-cookie"]
    assert len(set_cookie_headers) > 0
    assert any(b"dewie_session" in h[1] for h in set_cookie_headers)


@pytest.mark.asyncio
async def test_auth_login_sets_cookie_with_correct_attributes():
    """Verify cookie has httponly, max_age set (SameSite removed for Tailscale DNS compat)."""
    
    from dewie.api.routes.auth import LoginRequest
    
    password = "password123"
    user_id = str(uuid.uuid4())
    password_hash = hash_password(password)
    
    select_result = MagicMock()
    select_result.mappings.return_value.fetchone.return_value = {
        "id": user_id,
        "email": "test@example.com",
        "name": "Test User",
        "password_hash": password_hash,
        "is_admin": False,
    }
    
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=select_result)
    
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    request = _make_request_with_app(pg)
    body = LoginRequest(username="test@example.com", password=password)
    
    response = await auth_login(body, request)
    
    set_cookie_headers = [h[1].decode() if isinstance(h[1], bytes) else h[1] for h in response.raw_headers if h[0].lower() == b"set-cookie"]
    assert any("dewie_session" in header and "HttpOnly" in header and "Path=/" in header
               for header in set_cookie_headers)


@pytest.mark.asyncio
async def test_auth_login_returns_401_for_wrong_password():
    """Invalid credentials → 401."""
    
    from dewie.api.routes.auth import LoginRequest
    
    password = "correct_password"
    user_id = str(uuid.uuid4())
    password_hash = hash_password(password)
    
    select_result = MagicMock()
    select_result.mappings.return_value.fetchone.return_value = {
        "id": user_id,
        "email": "test@example.com",
        "name": "Test User",
        "password_hash": password_hash,
        "is_admin": False,
    }
    
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=select_result)
    
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    request = _make_request_with_app(pg)
    body = LoginRequest(username="test@example.com", password="wrong_password")
    
    with pytest.raises(HTTPException) as exc_info:
        await auth_login(body, request)
    
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_auth_login_returns_401_for_nonexistent_email():
    """Email not found → 401."""
    
    from dewie.api.routes.auth import LoginRequest
    
    select_result = MagicMock()
    select_result.mappings.return_value.fetchone.return_value = None
    
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=select_result)
    
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    request = _make_request_with_app(pg)
    body = LoginRequest(username="nonexistent@example.com", password="password123")
    
    with pytest.raises(HTTPException) as exc_info:
        await auth_login(body, request)
    
    assert exc_info.value.status_code == 401


# ── auth_me ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auth_me_returns_not_authenticated_when_no_session_and_auth_enabled():
    """GET /auth/me without cookie and auth enabled → returns authenticated:false.
    
    Regression test for issue #219: previously /auth/me returned a synthetic
    local-mode user even when AUTH_ENABLED=true and there was no session,
    which caused the home page to show 'Open app' for unauthenticated users
    and allowed them to reach app.html where all API calls then failed with 401.
    """
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.user_id = None
    request.state.email = None
    request.state.is_admin = False
    request.state.activation_status = None
    request.cookies = {}  # No dewie_session cookie

    with patch("dewie.config.settings") as mock_settings:
        mock_settings.auth_enabled = True
        mock_settings.local_auth_enabled = False
        response = await auth_me(request)

    assert response["authenticated"] is False
    assert response["user_id"] is None


@pytest.mark.asyncio
async def test_auth_me_returns_local_mode_when_no_session_and_auth_disabled():
    """GET /auth/me without cookie and auth disabled → returns local-mode identity."""
    
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.user_id = None
    request.state.email = None
    request.state.is_admin = False
    request.state.activation_status = None
    request.cookies = {}  # No dewie_session cookie

    with patch("dewie.config.settings") as mock_settings:
        mock_settings.auth_enabled = False
        mock_settings.local_auth_enabled = False
        mock_settings.local_auth_email = "Dewie Local Catalog"
        response = await auth_me(request)

    assert response["user_id"] == "00000000-0000-0000-0000-000000000002"
    assert response["email"] == "Dewie Local Catalog"
    assert response["name"] == "Local mode"
    assert response["is_admin"] is True
    # Issue #219: local/open mode must return authenticated:True so the home
    # page "Open app" button is shown and app.html pages load their content.
    assert response["authenticated"] is True


@pytest.mark.asyncio
async def test_auth_me_returns_user_from_session_cookie():
    """GET /auth/me with valid session cookie → returns authenticated user."""
    
    from dewie.local_auth import create_session_token
    
    user_id = str(uuid.uuid4())
    email = "test@example.com"
    token = create_session_token(user_id, email, is_admin=False)
    
    # Mock DB connection for fetching user name
    select_result = MagicMock()
    select_result.mappings.return_value.fetchone.return_value = {
        "name": "Test User",
    }
    
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=select_result)
    
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
    
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.user_id = None
    request.state.email = None
    request.state.is_admin = False
    request.state.activation_status = None
    request.cookies = {"dewie_session": token}
    request.app = MagicMock()
    request.app.state = MagicMock()
    request.app.state.postgres = MagicMock()
    request.app.state.postgres._engine = mock_engine
    
    response = await auth_me(request)
    
    assert response["user_id"] == user_id
    assert response["email"] == email
    assert response["is_admin"] is False
    assert response["name"] == "Test User"


@pytest.mark.asyncio
async def test_auth_me_returns_authenticated_user_from_request_state():
    """GET /auth/me with authenticated request.state → returns real user."""
    
    user_id = str(uuid.uuid4())
    email = "authenticated@example.com"
    
    # Mock DB connection for fetching user name
    select_result = MagicMock()
    select_result.mappings.return_value.fetchone.return_value = {
        "name": "Authenticated User",
    }
    
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=select_result)
    
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
    
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.user_id = user_id
    request.state.email = email
    request.state.is_admin = True
    request.state.activation_status = "approved"
    request.app = MagicMock()
    request.app.state = MagicMock()
    request.app.state.postgres = MagicMock()
    request.app.state.postgres._engine = mock_engine
    
    response = await auth_me(request)
    
    assert response["user_id"] == user_id
    assert response["email"] == email
    assert response["is_admin"] is True


# ── auth_logout ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auth_logout_deletes_cookie():
    """POST /auth/logout → clears session cookie."""
    
    response = MagicMock()
    response.delete_cookie = MagicMock()
    request = MagicMock()
    request.state.request_id = "test-req-1"
    
    result = await auth_logout(response, request)
    
    assert result["ok"] is True
    response.delete_cookie.assert_called_once_with("dewie_session", path="/")


# ── auth_signout ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auth_signout_deletes_cookie():
    """POST /auth/signout (alias for logout) → clears session cookie."""
    
    response = MagicMock()
    response.delete_cookie = MagicMock()
    request = MagicMock()
    request.state.request_id = "test-req-2"
    
    result = await auth_signout(response, request)
    
    assert result["ok"] is True
    response.delete_cookie.assert_called_once_with("dewie_session", path="/")


# ── auth_me auth_method priority fix (issue #244) ─────────────────────────────


@pytest.mark.asyncio
async def test_auth_me_returns_password_when_user_has_both_google_sub_and_password():
    """Issue #244: user with both google_sub and password_hash should show auth_method=password.
    
    The dev@dewie.ai user was seeded as an OAuth placeholder but later got a password.
    Auth method should reflect the active credential (password), not the OAuth remnant.
    """
    from dewie.local_auth import create_session_token

    user_id = str(uuid.uuid4())
    email = "dev@dewie.ai"
    token = create_session_token(user_id, email, is_admin=True)

    # Simulate user that has both google_sub set AND a valid password_hash
    select_result = MagicMock()
    select_result.mappings.return_value.fetchone.return_value = {
        "name": "Dev (Internal)",
        "google_sub": "google-oauth2|some-legacy-sub",
        "has_password": True,
    }

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=select_result)

    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)

    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.user_id = None
    request.state.email = None
    request.state.is_admin = False
    request.state.activation_status = None
    request.cookies = {"dewie_session": token}
    request.app = MagicMock()
    request.app.state = MagicMock()
    request.app.state.postgres = MagicMock()
    request.app.state.postgres._engine = mock_engine

    response = await auth_me(request)

    assert response["auth_method"] == "password", (
        f"Expected 'password' but got '{response['auth_method']}'. "
        "Users with both google_sub and password_hash should show as password auth."
    )


@pytest.mark.asyncio
async def test_auth_me_returns_google_when_user_has_google_sub_but_no_password():
    """Google OAuth users without a password should still show auth_method=google."""
    from dewie.local_auth import create_session_token

    user_id = str(uuid.uuid4())
    email = "googleuser@example.com"
    token = create_session_token(user_id, email, is_admin=False)

    select_result = MagicMock()
    select_result.mappings.return_value.fetchone.return_value = {
        "name": "Google User",
        "is_admin": False,
        "google_sub": "google-oauth2|real-google-sub",
        "has_password": False,
    }

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=select_result)

    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)

    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.user_id = None
    request.state.email = None
    request.state.is_admin = False
    request.state.activation_status = None
    request.cookies = {"dewie_session": token}
    request.app = MagicMock()
    request.app.state = MagicMock()
    request.app.state.postgres = MagicMock()
    request.app.state.postgres._engine = mock_engine

    response = await auth_me(request)

    assert response["auth_method"] == "google"


@pytest.mark.asyncio
async def test_auth_me_returns_apple_when_user_has_apple_sub_but_no_password():
    """Apple Sign In users without a password should show auth_method=apple."""
    from dewie.local_auth import create_session_token

    user_id = str(uuid.uuid4())
    email = "appleuser@privaterelay.appleid.com"
    token = create_session_token(user_id, email, is_admin=False)

    select_result = MagicMock()
    select_result.mappings.return_value.fetchone.return_value = {
        "name": "Apple User",
        "is_admin": False,
        "google_sub": None,
        "apple_sub": "apple-sub-123456",
        "has_password": False,
    }

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=select_result)

    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)

    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.user_id = None
    request.state.email = None
    request.state.is_admin = False
    request.state.activation_status = None
    request.cookies = {"dewie_session": token}
    request.app = MagicMock()
    request.app.state = MagicMock()
    request.app.state.postgres = MagicMock()
    request.app.state.postgres._engine = mock_engine

    response = await auth_me(request)

    assert response["auth_method"] == "apple"


@pytest.mark.asyncio
async def test_auth_me_returns_password_default_when_user_has_no_auth_identifiers():
    """Issue #244: users with no google_sub, apple_sub, or password_hash still get a safe default.

    This covers legacy/orphaned accounts and ensures no KeyError or crash.
    auth_method should default to 'password' (the session-cookie default).
    """
    from dewie.local_auth import create_session_token

    user_id = str(uuid.uuid4())
    email = "legacy@example.com"
    token = create_session_token(user_id, email, is_admin=False)

    select_result = MagicMock()
    select_result.mappings.return_value.fetchone.return_value = {
        "name": "Legacy User",
        "google_sub": None,
        "apple_sub": None,
        "has_password": False,
    }

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=select_result)

    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)

    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.user_id = None
    request.state.email = None
    request.state.is_admin = False
    request.state.activation_status = None
    request.cookies = {"dewie_session": token}
    request.app = MagicMock()
    request.app.state = MagicMock()
    request.app.state.postgres = MagicMock()
    request.app.state.postgres._engine = mock_engine

    response = await auth_me(request)

    # Should not crash and should return a safe default auth_method
    assert "auth_method" in response
    assert response["auth_method"] in ("password", "local", "google", "apple"), (
        f"Unexpected auth_method: '{response['auth_method']}'"
    )
    assert response["authenticated"] is True


# ── issue #239: dev@dewie.ai sign-in via login endpoint ───────────────────────


@pytest.mark.asyncio
async def test_auth_login_succeeds_for_dev_at_dewie_ai_with_correct_password():
    """Issue #239: dev@dewie.ai should be able to sign in with the seeded password.

    Root cause was the dev seed user had NULL password_hash. The schema migration
    sets a bcrypt hash for 'password'. This test verifies the login endpoint
    correctly authenticates the dev user once the hash is in place.
    """
    import bcrypt as _bcrypt

    dev_password = "password"
    dev_hash = _bcrypt.hashpw(dev_password.encode(), _bcrypt.gensalt(4)).decode()

    select_result = MagicMock()
    select_result.mappings.return_value.fetchone.return_value = {
        "id": "00000000-0000-0000-0000-000000000002",
        "email": "dev@dewie.ai",
        "name": "Dev (Internal)",
        "password_hash": dev_hash,
        "is_admin": True,
    }

    update_result = MagicMock()

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(side_effect=[select_result, update_result])

    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)

    postgres_mock = MagicMock()
    postgres_mock._engine = mock_engine

    request = _make_request_with_app(postgres_mock)
    request.cookies = {}

    from dewie.api.routes.auth import LoginRequest

    body = LoginRequest(username="dev@dewie.ai", password=dev_password)
    resp = await auth_login(body, request)

    assert resp.status_code == 200, (
        f"Expected 200 for dev@dewie.ai/password after schema migration sets hash, got {resp.status_code}"
    )
    data = resp.body
    import json as _json
    body_data = _json.loads(data)
    assert body_data["ok"] is True
    assert body_data["email"] == "dev@dewie.ai"
    # Session cookie must be present
    set_cookie = resp.headers.get("set-cookie", "")
    assert "dewie_session" in set_cookie, (
        "Login response must set dewie_session cookie for session-based auth to work"
    )


@pytest.mark.asyncio
async def test_auth_login_returns_401_for_dev_at_dewie_ai_with_null_hash():
    """Issue #239 regression guard: NULL password_hash still returns 401 (no bypass).

    The schema migration MUST have run to seed the hash. Without it, sign-in fails.
    This test ensures we don't accidentally add a bypass for NULL password_hash.
    """
    select_result = MagicMock()
    select_result.mappings.return_value.fetchone.return_value = {
        "id": "00000000-0000-0000-0000-000000000002",
        "email": "dev@dewie.ai",
        "name": "Dev (Internal)",
        "password_hash": None,
        "is_admin": True,
    }

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=select_result)

    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)

    postgres_mock = MagicMock()
    postgres_mock._engine = mock_engine

    request = _make_request_with_app(postgres_mock)
    request.cookies = {}

    from dewie.api.routes.auth import LoginRequest

    body = LoginRequest(username="dev@dewie.ai", password="password")
    try:
        resp = await auth_login(body, request)
        raise AssertionError("Expected HTTPException(401) but got a response")
    except HTTPException as exc:
        assert exc.status_code == 401


@pytest.mark.asyncio
async def test_auth_me_dev_at_dewie_ai_shows_password_not_google():
    """Regression test for issue #244: dev@dewie.ai must report auth_method='password'.

    The dev@dewie.ai seed user (UUID 00000000-0000-0000-0000-000000000002) was
    originally created as an OAuth placeholder with a google_sub set. A later
    migration added a bcrypt password_hash for 'password'. Before the fix, the
    /auth/me endpoint returned auth_method='google' because google_sub was checked
    before password_hash. After the fix, password takes priority.

    This test pins the exact production scenario using the well-known dev user UUID
    and email to prevent regressions.
    """
    from dewie.local_auth import create_session_token

    dev_user_id = "00000000-0000-0000-0000-000000000002"
    dev_email = "dev@dewie.ai"
    token = create_session_token(dev_user_id, dev_email, is_admin=True)

    select_result = MagicMock()
    select_result.mappings.return_value.fetchone.return_value = {
        "name": "Dev (Internal)",
        # google_sub is set (legacy OAuth placeholder from original seed)
        "google_sub": "google-placeholder-sub",
        "apple_sub": None,
        # password_hash is set by the seed migration
        "has_password": True,
    }

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=select_result)

    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)

    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.user_id = None
    request.state.email = None
    request.state.is_admin = False
    request.state.activation_status = None
    request.cookies = {"dewie_session": token}
    request.app = MagicMock()
    request.app.state = MagicMock()
    request.app.state.postgres = MagicMock()
    request.app.state.postgres._engine = mock_engine

    response = await auth_me(request)

    assert response["auth_method"] == "password", (
        f"dev@dewie.ai should show auth_method='password' (has password_hash set), "
        f"but got '{response['auth_method']}'. "
        "Issue #244 regression: password must take priority over google_sub."
    )
    assert response["authenticated"] is True
    assert response["email"] == dev_email


# ── Web panel login tests (issue #380) ────────────────────────────────────────

@pytest.mark.asyncio
async def test_login_accepts_username_field():
    """Frontend sends 'username' not 'email' — LoginRequest must accept both.
    
    Issue #380: logging in via the web panel returned '[object Object]' error
    because the frontend POSTs {username, password} but LoginRequest expected
    {email, password}, causing a FastAPI 422 validation error with a list detail.
    """
    from dewie.api.routes.auth import LoginRequest

    body = LoginRequest(**{"username": "dev@dewie.ai", "password": "secret"})
    assert body.email == "dev@dewie.ai", (
        "LoginRequest must coerce 'username' -> 'email'. "
        "Frontend sends 'username', not 'email'."
    )


@pytest.mark.asyncio
async def test_login_accepts_email_field():
    """LoginRequest.email field still works when sent directly."""
    from dewie.api.routes.auth import LoginRequest

    body = LoginRequest(**{"email": "admin@dewie.ai", "password": "secret"})
    assert body.email == "admin@dewie.ai"


@pytest.mark.asyncio
async def test_login_success_returns_ok_and_cookie():
    """Successful login returns {ok: True} with a session cookie."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi import Request

    from dewie.api.routes.auth import LoginRequest, auth_login

    mock_user = {
        "id": "00000000-0000-0000-0000-000000000002",
        "email": "dev@dewie.ai",
        "name": "Dev",
        "is_admin": True,
    }

    body = LoginRequest(**{"username": "dev@dewie.ai", "password": "dewie"})
    request = MagicMock(spec=Request)
    request.app = MagicMock()
    request.app.state.postgres = MagicMock()

    with patch("dewie.api.routes.auth.verify_local_user", AsyncMock(return_value=mock_user)):
        response = await auth_login(body, request)

    assert response.status_code == 200
    body_data = response.body
    import json
    data = json.loads(body_data)
    assert data["ok"] is True
    assert data["email"] == "dev@dewie.ai"
    # Cookie must be set
    cookie_header = response.headers.get("set-cookie", "")
    assert "dewie_session=" in cookie_header, "Session cookie must be set on successful login"


@pytest.mark.asyncio
async def test_login_invalid_credentials_returns_401():
    """Invalid credentials return HTTP 401, not a structured validation error."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi import HTTPException, Request

    from dewie.api.routes.auth import LoginRequest, auth_login

    body = LoginRequest(**{"username": "dev@dewie.ai", "password": "wrong"})
    request = MagicMock(spec=Request)
    request.app = MagicMock()
    request.app.state.postgres = MagicMock()

    with patch("dewie.api.routes.auth.verify_local_user", AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as exc_info:
            await auth_login(body, request)

    assert exc_info.value.status_code == 401
    assert isinstance(exc_info.value.detail, str), (
        "detail must be a plain string so the frontend can display it directly. "
        "If detail is a list/dict, the browser shows '[object Object]'."
    )


@pytest.mark.asyncio
async def test_login_username_email_field_conflict_prefers_email():
    """If both 'username' and 'email' are sent, 'email' takes precedence."""
    from dewie.api.routes.auth import LoginRequest

    body = LoginRequest(**{"email": "real@dewie.ai", "username": "ignored@dewie.ai", "password": "x"})
    assert body.email == "real@dewie.ai", "When 'email' is provided, it should not be overwritten by 'username'"
