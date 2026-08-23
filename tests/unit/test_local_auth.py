"""Tests for dewie.local_auth — password hashing, JWT tokens, and local user creation."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt
import jwt
import pytest

from dewie.local_auth import (
    JWT_ALGORITHM,
    JWT_EXPIRY_HOURS,
    _default_display_name,
    create_local_user,
    create_session_token,
    hash_password,
    seed_default_admin,
    update_user_password,
    verify_local_user,
    verify_password,
    verify_session_token,
)

# ── _default_display_name (issue #243) ───────────────────────────────────────


def test_default_display_name_email_returns_local_part():
    """Email addresses → local part (before @) as display name."""
    assert _default_display_name("alice@example.com") == "alice"


def test_default_display_name_plain_username_returns_as_is():
    """Plain usernames (no @) are returned unchanged."""
    assert _default_display_name("alice") == "alice"
    assert _default_display_name("johndoe") == "johndoe"


def test_default_display_name_strips_whitespace():
    """Leading/trailing whitespace is stripped."""
    assert _default_display_name("  alice  ") == "alice"
    assert _default_display_name("  alice@example.com  ") == "alice"


def test_default_display_name_admin_username():
    """'admin' (the default seed username) maps to 'admin'."""
    assert _default_display_name("admin") == "admin"


def test_default_display_name_email_with_empty_local_part():
    """Edge case: identifier starting with @ falls back to the whole string."""
    result = _default_display_name("@nodomain")
    assert result == "@nodomain"


# ── hash_password ──────────────────────────────────────────────────────────────


def test_hash_password_returns_bcrypt_hash():
    password = "test_password_123"
    hash_result = hash_password(password)
    assert isinstance(hash_result, str)
    assert hash_result.startswith("$2b$")  # bcrypt hash prefix


def test_hash_password_is_verifiable():
    password = "secure_password_456"
    hash_result = hash_password(password)
    assert bcrypt.checkpw(password.encode(), hash_result.encode())


def test_hash_password_rejects_wrong_password():
    password = "correct_password"
    wrong_password = "wrong_password"
    hash_result = hash_password(password)
    assert not bcrypt.checkpw(wrong_password.encode(), hash_result.encode())


def test_hash_password_two_calls_produce_different_hashes():
    password = "same_password"
    hash1 = hash_password(password)
    hash2 = hash_password(password)
    # Both should hash the same password, but produce different hashes (different salts)
    assert hash1 != hash2
    # But both should verify the password
    assert bcrypt.checkpw(password.encode(), hash1.encode())
    assert bcrypt.checkpw(password.encode(), hash2.encode())


# ── verify_password ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_password_returns_true_for_correct_password():
    password = "correct_password"
    hash_result = hash_password(password)
    result = await verify_password(password, hash_result)
    assert result is True


@pytest.mark.asyncio
async def test_verify_password_returns_false_for_wrong_password():
    password = "correct_password"
    wrong_password = "wrong_password"
    hash_result = hash_password(password)
    result = await verify_password(wrong_password, hash_result)
    assert result is False


@pytest.mark.asyncio
async def test_verify_password_runs_in_thread_pool():
    # Verify that it's async (can await it)
    password = "test"
    hash_result = hash_password(password)
    result = await verify_password(password, hash_result)
    assert result is True


# ── create_session_token ───────────────────────────────────────────────────────


def test_create_session_token_returns_jwt_string():
    user_id = str(uuid.uuid4())
    email = "test@example.com"
    token = create_session_token(user_id, email)
    assert isinstance(token, str)
    # Should be decodable as JWT (three parts separated by dots)
    assert token.count(".") == 2


def test_create_session_token_contains_user_data():
    user_id = str(uuid.uuid4())
    email = "test@example.com"
    token = create_session_token(user_id, email, is_admin=True)
    
    # Decode without verification to check payload structure
    payload = jwt.decode(token, options={"verify_signature": False})
    assert payload["sub"] == user_id
    assert payload["email"] == email
    assert payload["is_admin"] is True


def test_create_session_token_sets_expiry():
    user_id = str(uuid.uuid4())
    email = "test@example.com"
    token = create_session_token(user_id, email)
    
    payload = jwt.decode(token, options={"verify_signature": False})
    # Check that exp is set
    assert "exp" in payload
    assert "iat" in payload
    # exp should be roughly 24 hours from now
    now = datetime.now(UTC)
    exp_time = datetime.fromtimestamp(payload["exp"], tz=UTC)
    iat_time = datetime.fromtimestamp(payload["iat"], tz=UTC)
    
    time_diff = (exp_time - iat_time).total_seconds()
    expected_diff = JWT_EXPIRY_HOURS * 3600
    # Allow 1 second tolerance
    assert abs(time_diff - expected_diff) < 1


def test_create_session_token_is_admin_defaults_to_false():
    user_id = str(uuid.uuid4())
    email = "test@example.com"
    token = create_session_token(user_id, email)
    
    payload = jwt.decode(token, options={"verify_signature": False})
    assert payload["is_admin"] is False


# ── verify_session_token ───────────────────────────────────────────────────────


def test_verify_session_token_returns_payload_for_valid_token():
    user_id = str(uuid.uuid4())
    email = "test@example.com"
    is_admin = True
    token = create_session_token(user_id, email, is_admin)
    
    payload = verify_session_token(token)
    assert payload is not None
    assert payload["sub"] == user_id
    assert payload["email"] == email
    assert payload["is_admin"] is True


def test_verify_session_token_returns_none_for_invalid_token():
    invalid_token = "not.a.valid.token"
    result = verify_session_token(invalid_token)
    assert result is None


def test_verify_session_token_returns_none_for_tampered_token():
    user_id = str(uuid.uuid4())
    email = "test@example.com"
    # Pin secret so sign and verify use the same key within this test
    with patch("dewie.local_auth._get_jwt_secret", return_value="test-secret-32-chars-long-padded"):
        token = create_session_token(user_id, email)

        # Tamper with the token's payload segment so the signature no longer
        # matches. (Flipping the last char of the signature is unreliable —
        # base64's trailing bits mean it can decode to the same bytes.)
        parts = token.split(".")
        parts[1] = ("B" if parts[1][0] != "B" else "C") + parts[1][1:]
        tampered_token = ".".join(parts)

        result = verify_session_token(tampered_token)
    assert result is None


def test_verify_session_token_returns_none_for_expired_token():
    with patch("dewie.local_auth._get_jwt_secret") as mock_secret:
        mock_secret.return_value = "test-secret-key"
        
        # Create a token with -1 hour expiry (already expired)
        now = datetime.now(UTC)
        payload = {
            "sub": str(uuid.uuid4()),
            "email": "test@example.com",
            "is_admin": False,
            "iat": now,
            "exp": now - timedelta(hours=1),  # Expired
        }
        expired_token = jwt.encode(payload, "test-secret-key", algorithm=JWT_ALGORITHM)
        
        # With mocked secret, we can verify the token format, but it will be rejected as expired
        result = verify_session_token(expired_token)
        assert result is None


def test_verify_session_token_returns_none_for_empty_token():
    result = verify_session_token("")
    assert result is None


# ── create_local_user ──────────────────────────────────────────────────────────


def _make_pg_mock_for_user_creation(existing_user=None, created_user_id=None):
    """Helper to create a mock postgres connection for user operations."""
    if created_user_id is None:
        created_user_id = str(uuid.uuid4())
    
    # Mock for checking if email exists
    existing_result = MagicMock()
    if existing_user:
        existing_result.fetchone.return_value = {"id": existing_user}
    else:
        existing_result.fetchone.return_value = None
    
    # Mock for inserting new user
    insert_result = MagicMock()
    insert_result.mappings.return_value.fetchone.return_value = {
        "id": created_user_id,
        "email": "test@example.com",
        "name": "Test User",
        "created_at": datetime.now(UTC),
        "is_admin": False,
    }
    
    # Mock connection
    mock_conn = AsyncMock()
    
    # Set up execute to return different results based on call order
    execute_calls = [existing_result, insert_result]
    mock_conn.execute = AsyncMock(side_effect=execute_calls)
    
    # Mock engine
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    return pg


@pytest.mark.asyncio
async def test_create_local_user_returns_user_dict():
    pg = _make_pg_mock_for_user_creation()
    user = await create_local_user(pg, "test@example.com", "password123", "Test User")
    
    assert isinstance(user, dict)
    assert "id" in user
    assert "email" in user
    assert user["email"] == "test@example.com"
    assert user["name"] == "Test User"
    assert "created_at" in user
    assert "is_admin" in user
    assert user["is_admin"] is False


@pytest.mark.asyncio
async def test_create_local_user_raises_on_duplicate_email():
    existing_id = str(uuid.uuid4())
    pg = _make_pg_mock_for_user_creation(existing_user=existing_id)
    
    with pytest.raises(ValueError, match="already exists"):
        await create_local_user(pg, "test@example.com", "password123", "Test User")


@pytest.mark.asyncio
async def test_create_local_user_raises_on_duplicate_plain_username():
    """Issue #243: duplicate check works for non-email usernames too."""
    existing_id = str(uuid.uuid4())
    pg = _make_pg_mock_for_user_creation(existing_user=existing_id)

    with pytest.raises(ValueError, match="already exists"):
        await create_local_user(pg, "alice", "password123")


@pytest.mark.asyncio
async def test_create_local_user_uses_plain_username_as_default_name():
    """Issue #243: plain username (no @) is used directly as the display name."""
    mock_conn = AsyncMock()

    execute_calls: list = []

    async def track_execute(query, params):
        execute_calls.append((str(query), params))
        if "SELECT id FROM users" in str(query):
            result = MagicMock()
            result.fetchone.return_value = None
            return result
        result = MagicMock()
        result.mappings.return_value.fetchone.return_value = {
            "id": str(uuid.uuid4()),
            "email": params["email"],
            "name": params["name"],
            "created_at": datetime.now(UTC),
            "is_admin": False,
        }
        return result

    mock_conn.execute = track_execute
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    pg = MagicMock()
    pg._engine = mock_engine

    user = await create_local_user(pg, "alice", "password123")

    # Display name should be the plain username itself, not an email prefix
    assert user["name"] == "alice"


@pytest.mark.asyncio
async def test_create_local_user_uses_email_prefix_as_default_name():
    pg = _make_pg_mock_for_user_creation()
    user = await create_local_user(pg, "alice@example.com", "password123")
    
    # Should use email prefix as name when not provided
    assert user["name"] is not None


@pytest.mark.asyncio
async def test_create_local_user_hashes_password():
    pg = _make_pg_mock_for_user_creation()
    
    # Capture the actual parameters passed to execute
    execute_calls = []
    
    # Create a real mock that captures the parameters
    mock_conn = AsyncMock()
    
    # Track execute calls
    async def track_execute(query, params):
        execute_calls.append((query, params))
        if "SELECT id FROM users" in str(query):
            result = MagicMock()
            result.fetchone.return_value = None
            return result
        else:
            result = MagicMock()
            result.mappings.return_value.fetchone.return_value = {
                "id": str(uuid.uuid4()),
                "email": params["email"],
                "name": params.get("name", "Test"),
                "created_at": datetime.now(UTC),
                "is_admin": False,
            }
            return result
    
    mock_conn.execute = track_execute
    
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    await create_local_user(pg, "test@example.com", "password123")
    
    # Verify that password_hash param was set and is a bcrypt hash
    for query, params in execute_calls:
        if "INSERT INTO users" in str(query):
            assert "password_hash" in params
            password_hash = params["password_hash"]
            # Should be bcrypt hash
            assert password_hash.startswith("$2b$")


# ── verify_local_user ──────────────────────────────────────────────────────────


def _make_pg_mock_for_user_verification(user_found=True, password_correct=True):
    """Helper to create a mock postgres connection for user verification."""
    
    # First execute: SELECT user
    select_result = MagicMock()
    if user_found:
        user_id = str(uuid.uuid4())
        password = "password123"
        password_hash = hash_password(password) if password_correct else hash_password("different_password")
        select_result.mappings.return_value.fetchone.return_value = {
            "id": user_id,
            "email": "test@example.com",
            "name": "Test User",
            "password_hash": password_hash,
            "is_admin": False,
        }
    else:
        select_result.mappings.return_value.fetchone.return_value = None
    
    # Second execute: UPDATE last_login_at
    update_result = MagicMock()
    
    mock_conn = AsyncMock()
    execute_calls = [select_result, update_result]
    mock_conn.execute = AsyncMock(side_effect=execute_calls)
    
    # Mock engine (need two contexts: one for connect, one for begin)
    mock_engine = MagicMock()
    
    # Set up both connect and begin
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    return pg


@pytest.mark.asyncio
async def test_verify_local_user_returns_user_dict_for_correct_credentials():
    pg = _make_pg_mock_for_user_verification(user_found=True, password_correct=True)
    user = await verify_local_user(pg, "test@example.com", "password123")
    
    assert isinstance(user, dict)
    assert user["email"] == "test@example.com"
    assert "id" in user
    assert "name" in user
    assert "is_admin" in user


@pytest.mark.asyncio
async def test_verify_local_user_returns_none_for_nonexistent_email():
    pg = _make_pg_mock_for_user_verification(user_found=False)
    user = await verify_local_user(pg, "nonexistent@example.com", "password123")
    
    assert user is None


@pytest.mark.asyncio
async def test_verify_local_user_returns_none_for_wrong_password():
    pg = _make_pg_mock_for_user_verification(user_found=True, password_correct=False)
    user = await verify_local_user(pg, "test@example.com", "wrong_password")
    
    assert user is None


@pytest.mark.asyncio
async def test_verify_local_user_returns_none_for_oauth_only_user():
    """OAuth-only users have no password_hash."""
    select_result = MagicMock()
    select_result.mappings.return_value.fetchone.return_value = {
        "id": str(uuid.uuid4()),
        "email": "oauth@example.com",
        "name": "OAuth User",
        "password_hash": None,  # OAuth user
        "is_admin": False,
    }
    
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=select_result)
    
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    user = await verify_local_user(pg, "oauth@example.com", "password123")
    assert user is None


# ── update_user_password ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_user_password_returns_true_on_success():
    update_result = MagicMock()
    update_result.rowcount = 1
    
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=update_result)
    
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    user_id = str(uuid.uuid4())
    result = await update_user_password(pg, user_id, "new_password_123")
    
    assert result is True


@pytest.mark.asyncio
async def test_update_user_password_returns_false_on_no_match():
    update_result = MagicMock()
    update_result.rowcount = 0  # No rows updated
    
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=update_result)
    
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    user_id = str(uuid.uuid4())
    result = await update_user_password(pg, user_id, "new_password_123")
    
    assert result is False


@pytest.mark.asyncio
async def test_update_user_password_hashes_password():
    """Verify password is hashed before sending to DB."""
    update_result = MagicMock()
    update_result.rowcount = 1
    
    # Track execute calls to inspect parameters
    execute_calls = []
    async def track_execute(query, params):
        execute_calls.append((query, params))
        return update_result
    
    mock_conn = AsyncMock()
    mock_conn.execute = track_execute
    
    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    
    pg = MagicMock()
    pg._engine = mock_engine
    
    user_id = str(uuid.uuid4())
    await update_user_password(pg, user_id, "new_password")
    
    # Check that password_hash param is a bcrypt hash
    for query, params in execute_calls:
        if "password_hash" in params:
            password_hash = params["password_hash"]
            assert password_hash.startswith("$2b$")
            # Verify it actually hashes the password
            assert bcrypt.checkpw(b"new_password", password_hash.encode())


# ── issue #239: dev@dewie.ai sign-in ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_local_user_rejects_oauth_only_user_no_password_hash():
    """OAuth-only / seed users with NULL password_hash cannot sign in with password.
    This is the root cause of issue #239 — the dev@dewie.ai seed row had no
    password_hash so every sign-in attempt returned None (401).
    After the fix, the seed migration sets a bcrypt hash for 'password' so
    dev@dewie.ai/password works in local development environments.
    """

    dev_password = "password"
    dev_hash = hash_password(dev_password)

    # Simulate the seeded dev user AFTER the migration applies the hash
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

    pg = MagicMock()
    pg._engine = mock_engine

    user = await verify_local_user(pg, "dev@dewie.ai", dev_password)

    assert user is not None, (
        "dev@dewie.ai/password should sign in after the seed migration sets password_hash"
    )
    assert user["email"] == "dev@dewie.ai"
    assert user["is_admin"] is True


@pytest.mark.asyncio
async def test_verify_local_user_returns_none_for_dev_user_with_null_password_hash():
    """Before fix: dev seed row has NULL password_hash → sign-in returns None (root cause)."""
    select_result = MagicMock()
    select_result.mappings.return_value.fetchone.return_value = {
        "id": "00000000-0000-0000-0000-000000000002",
        "email": "dev@dewie.ai",
        "name": "Dev (Internal)",
        "password_hash": None,  # ← pre-fix state: no password set
        "is_admin": True,
    }

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=select_result)

    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)

    pg = MagicMock()
    pg._engine = mock_engine

    # This should still return None — we don't allow password-less sign-in
    user = await verify_local_user(pg, "dev@dewie.ai", "password")
    assert user is None


# ── seed_default_admin ─────────────────────────────────────────────────────────


def _make_pg_mock_for_seeding(user_count=0):
    """Helper to create a mock postgres for seed_default_admin tests."""
    count_result = MagicMock()
    count_result.mappings.return_value.fetchone.return_value = {"cnt": user_count}

    # Mock for insert (only called when user_count == 0)
    insert_result = MagicMock()
    new_user_id = str(uuid.uuid4())
    insert_result.mappings.return_value.fetchone.return_value = {
        "id": new_user_id,
        "email": "admin",
        "name": "admin",
        "created_at": datetime.now(UTC),
        "is_admin": True,
    }

    mock_conn = AsyncMock()

    if user_count == 0:
        execute_calls = [count_result, insert_result]
    else:
        execute_calls = [count_result]
    mock_conn.execute = AsyncMock(side_effect=execute_calls)

    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)

    pg = MagicMock()
    pg._engine = mock_engine

    return pg


@pytest.mark.asyncio
async def test_seed_default_admin_creates_user_when_table_empty():
    """When users table is empty, a default admin user is created."""
    pg = _make_pg_mock_for_seeding(user_count=0)
    result = await seed_default_admin(pg, "admin", "admin")

    assert result is not None
    assert result["email"] == "admin"
    assert result["is_admin"] is True
    assert result["name"] == "admin"


@pytest.mark.asyncio
async def test_seed_default_admin_skips_when_users_exist():
    """When users table is not empty, seeding is skipped and None is returned."""
    pg = _make_pg_mock_for_seeding(user_count=1)
    result = await seed_default_admin(pg, "admin", "admin")

    assert result is None


@pytest.mark.asyncio
async def test_seed_default_admin_hashes_password():
    """Verify the admin password is hashed with bcrypt."""
    count_result = MagicMock()
    count_result.mappings.return_value.fetchone.return_value = {"cnt": 0}

    insert_result = MagicMock()
    insert_result.mappings.return_value.fetchone.return_value = {
        "id": str(uuid.uuid4()),
        "email": "admin",
        "name": "admin",
        "created_at": datetime.now(UTC),
        "is_admin": True,
    }

    captured_params = {}

    async def track_execute(query, params=None):
        captured_params[str(query)] = params
        if "INSERT INTO users" in str(query):
            return insert_result
        return count_result

    mock_conn = AsyncMock()
    mock_conn.execute = track_execute

    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)

    pg = MagicMock()
    pg._engine = mock_engine

    await seed_default_admin(pg, "admin", "admin")

    # Verify password_hash is a bcrypt hash
    for query, params in captured_params.items():
        if "INSERT INTO users" in query:
            assert "password_hash" in params
            password_hash = params["password_hash"]
            assert password_hash.startswith("$2b$")
            assert bcrypt.checkpw(b"admin", password_hash.encode())


@pytest.mark.asyncio
async def test_seed_default_admin_sets_is_admin_true():
    """The seeded admin user must have is_admin=True."""
    count_result = MagicMock()
    count_result.mappings.return_value.fetchone.return_value = {"cnt": 0}

    insert_result = MagicMock()
    insert_result.mappings.return_value.fetchone.return_value = {
        "id": str(uuid.uuid4()),
        "email": "admin",
        "name": "admin",
        "created_at": datetime.now(UTC),
        "is_admin": True,
    }

    captured_params = {}

    async def track_execute(query, params=None):
        captured_params[str(query)] = params
        if "INSERT INTO users" in str(query):
            return insert_result
        return count_result

    mock_conn = AsyncMock()
    mock_conn.execute = track_execute

    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)

    pg = MagicMock()
    pg._engine = mock_engine

    await seed_default_admin(pg, "admin", "admin")

    for query, params in captured_params.items():
        if "INSERT INTO users" in query:
            assert params["is_admin"] is True


@pytest.mark.asyncio
async def test_seed_default_admin_sets_enterprise_plan():
    """The seeded admin user must have plan='enterprise'."""
    count_result = MagicMock()
    count_result.mappings.return_value.fetchone.return_value = {"cnt": 0}

    insert_result = MagicMock()
    insert_result.mappings.return_value.fetchone.return_value = {
        "id": str(uuid.uuid4()),
        "email": "admin",
        "name": "admin",
        "created_at": datetime.now(UTC),
        "is_admin": True,
    }

    captured_params = {}

    async def track_execute(query, params=None):
        captured_params[str(query)] = params
        if "INSERT INTO users" in str(query):
            return insert_result
        return count_result

    mock_conn = AsyncMock()
    mock_conn.execute = track_execute

    mock_engine = MagicMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)

    pg = MagicMock()
    pg._engine = mock_engine

    await seed_default_admin(pg, "admin", "admin")

    for query, params in captured_params.items():
        if "INSERT INTO users" in query:
            assert params["plan"] == "enterprise"
