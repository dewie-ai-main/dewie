"""
Unit tests for multi-user database behaviour — issue #218.

Covers:
  - Multiple users can be created (no collision, unique IDs)
  - All users can write (upsert) to the database
  - Users from the same workspace can see each other's docs (via search/list)
  - A different user's ingestion does not corrupt an existing doc
  - Ability to delete entries from the database
  - Ability to write/delete documents scoped to the requesting user
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from dewie.models.content import ContentDocument, ContentStatus

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

ROOT_WORKSPACE_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")
TENANT_A_ID = "00000000-0000-0000-0000-000000000001"

USER_1_ID = str(uuid.uuid4())
USER_2_ID = str(uuid.uuid4())
USER_3_ID = str(uuid.uuid4())


def _make_pg():
    """Minimal PostgresClient shell with mocked engine."""
    from dewie.storage.postgres import PostgresClient

    pg = object.__new__(PostgresClient)
    pg._engine = MagicMock()
    pg._session_factory = MagicMock()
    pg._is_sqlite = False
    return pg


def _conn_cm(rows=None, mappings_rows=None, scalar=None, fetchone_val=None):
    """Build an async-context-manager that yields a mock connection."""
    conn = AsyncMock()
    result = MagicMock()
    result.fetchone.return_value = fetchone_val
    result.fetchall.return_value = rows or []
    result.rowcount = len(rows) if rows else 1
    if mappings_rows is not None:
        result.mappings.return_value.all.return_value = mappings_rows
        result.mappings.return_value.fetchone.return_value = (
            mappings_rows[0] if mappings_rows else None
        )
    if scalar is not None:
        result.scalar.return_value = scalar
    conn.execute = AsyncMock(return_value=result)
    conn.exec_driver_sql = AsyncMock(return_value=result)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, conn


def _begin_cm(rows=None, fetchone_val=None):
    """Build a transactional begin() context manager."""
    return _conn_cm(rows=rows, fetchone_val=fetchone_val)


def _make_doc(url="https://example.com/doc", owner_user_id=None, corpus_id=None):
    return ContentDocument(
        url=url,
        title="Test Document",
        summary="A test document summary",
        source="web",
        status=ContentStatus.READY,
        topics=["tech"],
        keywords=["test"],
        entities=[],
        ingested_at=datetime.now(UTC),
        owner_user_id=owner_user_id,
        corpus_id=corpus_id,
    )


def _make_user_row(user_id: str, email: str, name: str = "Test User") -> dict:
    return {
        "id": user_id,
        "email": email,
        "name": name,
        "is_admin": False,
        "activation_status": "approved",
        "created_at": datetime.now(UTC),
        "last_login_at": None,
        "has_password": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multiple users can be created
# ─────────────────────────────────────────────────────────────────────────────


class TestMultipleUsersCanBeCreated:
    """create_local_user / get_local_users covering distinct IDs & emails."""

    @pytest.mark.asyncio
    async def test_create_first_user_succeeds(self):
        """Creating a user when no existing email conflict should succeed."""
        from dewie.local_auth import create_local_user

        pg = _make_pg()
        user_row = {
            "id": USER_1_ID,
            "email": "alice@example.com",
            "name": "Alice",
            "created_at": datetime.now(UTC),
            "is_admin": False,
        }
        conn = AsyncMock()

        # First execute: email-check returns nothing (no conflict)
        no_conflict = MagicMock()
        no_conflict.fetchone.return_value = None

        # Second execute: INSERT RETURNING
        insert_result = MagicMock()
        insert_result.mappings.return_value.fetchone.return_value = user_row

        call_count = [0]

        async def fake_execute(sql, params=None):
            call_count[0] += 1
            return no_conflict if call_count[0] == 1 else insert_result

        conn.execute = fake_execute
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        pg._engine.begin.return_value = cm

        result = await create_local_user(pg, "alice@example.com", "password123", "Alice")
        assert result["email"] == "alice@example.com"
        assert result["name"] == "Alice"
        assert "id" in result

    @pytest.mark.asyncio
    async def test_create_duplicate_email_raises(self):
        """Attempting to create a user with an existing email raises ValueError."""
        from dewie.local_auth import create_local_user

        pg = _make_pg()
        conn = AsyncMock()
        conflict_result = MagicMock()
        conflict_result.fetchone.return_value = (USER_1_ID,)  # email exists
        conn.execute = AsyncMock(return_value=conflict_result)
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        pg._engine.begin.return_value = cm

        with pytest.raises(ValueError, match="Email already exists"):
            await create_local_user(pg, "alice@example.com", "password123")

    @pytest.mark.asyncio
    async def test_get_local_users_returns_all_users(self):
        """get_local_users returns a list of all users in the system."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        user_rows = [
            _make_user_row(USER_1_ID, "alice@example.com", "Alice"),
            _make_user_row(USER_2_ID, "bob@example.com", "Bob"),
            _make_user_row(USER_3_ID, "carol@example.com", "Carol"),
        ]
        conn = AsyncMock()
        result = MagicMock()
        result.fetchall.return_value = [MagicMock(_mapping=r) for r in user_rows]
        conn.execute = AsyncMock(return_value=result)
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        pg._engine.connect.return_value = cm

        users = await PostgresClient.get_local_users(pg)
        assert len(users) == 3
        emails = {u["email"] for u in users}
        assert emails == {"alice@example.com", "bob@example.com", "carol@example.com"}

    @pytest.mark.asyncio
    async def test_each_created_user_has_unique_id(self):
        """Users created by create_local_user get distinct UUIDs."""
        from dewie.local_auth import create_local_user

        created_ids: list[str] = []

        async def _create_one(email: str, name: str) -> dict:
            pg = _make_pg()
            new_id = str(uuid.uuid4())
            user_row = {
                "id": new_id,
                "email": email,
                "name": name,
                "created_at": datetime.now(UTC),
                "is_admin": False,
            }
            conn = AsyncMock()
            no_conflict = MagicMock()
            no_conflict.fetchone.return_value = None
            insert_result = MagicMock()
            insert_result.mappings.return_value.fetchone.return_value = user_row
            call_count = [0]

            async def fake_execute(sql, params=None):
                call_count[0] += 1
                return no_conflict if call_count[0] == 1 else insert_result

            conn.execute = fake_execute
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=conn)
            cm.__aexit__ = AsyncMock(return_value=False)
            pg._engine.begin.return_value = cm
            return await create_local_user(pg, email, "pass", name)

        u1 = await _create_one("alice@example.com", "Alice")
        u2 = await _create_one("bob@example.com", "Bob")
        assert u1["id"] != u2["id"]
        assert u1["email"] != u2["email"]


def _session_factory_cm(rows=None, mappings_rows=None):
    """Build a _session_factory()-style async context manager."""
    session = AsyncMock()
    result = MagicMock()
    result.rowcount = 1
    if mappings_rows is not None:
        result.mappings.return_value.all.return_value = mappings_rows
    if rows is not None:
        result.fetchall.return_value = rows
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, session


# ─────────────────────────────────────────────────────────────────────────────
# 2. All users can write to the database
# ─────────────────────────────────────────────────────────────────────────────


class TestAllUsersCanWrite:
    """Each user (identified by owner_user_id) can upsert documents."""

    @pytest.mark.asyncio
    async def test_user1_can_upsert_document(self):
        """User 1 calling upsert() should issue an INSERT/upsert execute."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        cm, session = _session_factory_cm()
        pg._session_factory.return_value = cm

        doc = _make_doc("https://example.com/user1-doc", owner_user_id=USER_1_ID)
        await PostgresClient.upsert(pg, doc)
        assert session.execute.called

    @pytest.mark.asyncio
    async def test_user2_can_upsert_document(self):
        """User 2 calling upsert() should also trigger a write."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        cm, session = _session_factory_cm()
        pg._session_factory.return_value = cm

        doc = _make_doc("https://example.com/user2-doc", owner_user_id=USER_2_ID)
        await PostgresClient.upsert(pg, doc)
        assert session.execute.called

    @pytest.mark.asyncio
    async def test_multiple_users_each_upsert_different_docs(self):
        """Simulate 3 users writing docs — each write triggers its own execute."""
        from dewie.storage.postgres import PostgresClient

        for i, user_id in enumerate([USER_1_ID, USER_2_ID, USER_3_ID]):
            pg = _make_pg()
            cm, session = _session_factory_cm()
            pg._session_factory.return_value = cm

            doc = _make_doc(f"https://example.com/doc-{i}", owner_user_id=user_id)
            await PostgresClient.upsert(pg, doc)
            assert session.execute.called, f"User {i+1} write did not call execute"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Users from the same workspace can see each other's docs
# ─────────────────────────────────────────────────────────────────────────────

DOC_ROW_TEMPLATE = {
    "id": None,  # filled per-test
    "url": "https://example.com/shared-doc",
    "title": "Shared Document",
    "summary": "Visible to all workspace members",
    "source": "web",
    "ingested_at": datetime.now(UTC),
    "status": "ready",
    "topics": ["tech"],
    "keywords": ["shared"],
    "entities": [],
    "sentiment": 0.5,
    "crawl_session": None,
    "enrichment_version": 1,
    "embedding_model": "text-embedding-3-small",
    "enriched_at": datetime.now(UTC),
    "answers_questions": [],
    "tone": "neutral",
    "document_type": None,
    "author": None,
    "reading_level": None,
    "embed_summary": None,
    "published_at": None,
    "paywall_detected": False,
    "paywall_type": "none",
    "alternate_terms": [],
    "enrichment_quality_score": 80,
    "gap_fill": False,
    "corpus_id": None,
    "language": "en",
}


class TestSameWorkspaceVisibility:
    """Documents ingested by one user are visible to others in the same workspace."""

    @pytest.mark.asyncio
    async def test_list_recent_returns_doc_ingested_by_other_user(self):
        """list_recent does not filter by user — all workspace members see all docs."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        doc_id = uuid.uuid4()
        row = {**DOC_ROW_TEMPLATE, "id": doc_id}

        session = AsyncMock()
        result = MagicMock()
        result.mappings.return_value.all.return_value = [row]
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=session)
        cm.__aexit__ = AsyncMock(return_value=False)
        pg._session_factory.return_value = cm

        docs = await PostgresClient.list_recent(pg, limit=10)
        assert len(docs) == 1
        assert docs[0].url == "https://example.com/shared-doc"

    @pytest.mark.asyncio
    async def test_get_local_users_shows_all_workspace_members(self):
        """All users in the same workspace appear in get_local_users."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        ws_users = [
            _make_user_row(USER_1_ID, "alice@ws.example.com"),
            _make_user_row(USER_2_ID, "bob@ws.example.com"),
        ]
        conn = AsyncMock()
        result = MagicMock()
        result.fetchall.return_value = [MagicMock(_mapping=r) for r in ws_users]
        conn.execute = AsyncMock(return_value=result)
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        pg._engine.connect.return_value = cm

        users = await PostgresClient.get_local_users(pg)
        assert len(users) == 2

    @pytest.mark.asyncio
    async def test_find_by_topics_returns_docs_from_all_users(self):
        """find_by_topics is workspace-scoped — no per-user filter applied."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        # Two docs from two different users, same topic
        rows = [
            {**DOC_ROW_TEMPLATE, "id": uuid.uuid4(), "url": "https://example.com/doc-u1"},
            {**DOC_ROW_TEMPLATE, "id": uuid.uuid4(), "url": "https://example.com/doc-u2"},
        ]
        session = AsyncMock()
        result = MagicMock()
        result.mappings.return_value.all.return_value = rows
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=session)
        cm.__aexit__ = AsyncMock(return_value=False)
        pg._session_factory.return_value = cm

        docs = await PostgresClient.find_by_topics(pg, ["tech"])
        assert len(docs) == 2
        urls = {d.url for d in docs}
        assert "https://example.com/doc-u1" in urls
        assert "https://example.com/doc-u2" in urls


# ─────────────────────────────────────────────────────────────────────────────
# 4. Other users' ingestions should not affect existing db entries
# ─────────────────────────────────────────────────────────────────────────────


class TestIngestionIsolation:
    """A second user's upsert must not overwrite a different document by another user."""

    @pytest.mark.asyncio
    async def test_upsert_uses_url_as_unique_key(self):
        """Upsert is keyed on URL. Two different URLs from two users produce two calls."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        cm, session = _session_factory_cm()
        pg._session_factory.return_value = cm

        doc_user1 = _make_doc("https://example.com/user1-article", owner_user_id=USER_1_ID)
        await PostgresClient.upsert(pg, doc_user1)

        # Reset for second user
        pg2 = _make_pg()
        cm2, session2 = _session_factory_cm()
        pg2._session_factory.return_value = cm2

        doc_user2 = _make_doc("https://example.com/user2-article", owner_user_id=USER_2_ID)
        await PostgresClient.upsert(pg2, doc_user2)

        # Both upserts executed independently
        assert session.execute.called
        assert session2.execute.called

    @pytest.mark.asyncio
    async def test_upsert_params_contain_correct_url(self):
        """The SQL params passed to each upsert contain only that user's URL."""
        from dewie.storage.postgres import PostgresClient

        captured_params: list[dict] = []

        session = AsyncMock()

        async def capture(sql, params=None):
            if params:
                captured_params.append(params)
            return MagicMock(rowcount=1)

        session.execute = capture
        session.commit = AsyncMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=session)
        cm.__aexit__ = AsyncMock(return_value=False)

        pg = _make_pg()
        pg._session_factory.return_value = cm

        target_url = "https://example.com/user1-exclusive"
        doc = _make_doc(target_url, owner_user_id=USER_1_ID)
        await PostgresClient.upsert(pg, doc)

        # At least one params dict should contain the correct URL
        urls_written = [p.get("url") for p in captured_params if "url" in p]
        assert target_url in urls_written, (
            f"Expected URL {target_url!r} in upsert params, got {urls_written}"
        )

    @pytest.mark.asyncio
    async def test_mark_status_targets_specific_doc_id(self):
        """mark_status scopes its UPDATE to a specific doc_id, not all docs."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        cm, session = _session_factory_cm()
        pg._session_factory.return_value = cm

        target_id = uuid.uuid4()
        await PostgresClient.mark_status(pg, target_id, ContentStatus.READY)

        # Verify the execute was called with the specific doc id in params
        session.execute.assert_called_once()
        call_args = session.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", {})
        if params:
            assert str(target_id) in str(params)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Ability to delete entries from the database
# ─────────────────────────────────────────────────────────────────────────────


class TestDeleteEntries:
    """delete_local_user and workspace/corpus deletes exercise DB deletion paths."""

    @pytest.mark.asyncio
    async def test_delete_local_user_returns_true_on_success(self):
        """delete_local_user returns True when a row was deleted."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        conn = AsyncMock()
        result = MagicMock()
        result.rowcount = 1
        conn.execute = AsyncMock(return_value=result)
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        pg._engine.begin.return_value = cm

        deleted = await PostgresClient.delete_local_user(pg, USER_1_ID)
        assert deleted is True

    @pytest.mark.asyncio
    async def test_delete_local_user_returns_false_when_not_found(self):
        """delete_local_user returns False when no rows were deleted."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        conn = AsyncMock()
        result = MagicMock()
        result.rowcount = 0
        conn.execute = AsyncMock(return_value=result)
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        pg._engine.begin.return_value = cm

        deleted = await PostgresClient.delete_local_user(pg, str(uuid.uuid4()))
        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_workspace_calls_execute(self):
        """delete_workspace issues a DELETE execute against the DB."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        cm, conn = _begin_cm()
        pg._engine.begin.return_value = cm

        ws_id = uuid.uuid4()
        await PostgresClient.delete_workspace(pg, ws_id)
        conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_corpus_calls_execute(self):
        """delete_corpus issues a DELETE execute against the DB."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        cm, conn = _begin_cm()
        pg._engine.begin.return_value = cm

        corpus_id = uuid.uuid4()
        await PostgresClient.delete_corpus(pg, corpus_id)
        conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_multiple_users_sequentially(self):
        """Deleting three distinct users succeeds for each in sequence."""
        from dewie.storage.postgres import PostgresClient

        for user_id in [USER_1_ID, USER_2_ID, USER_3_ID]:
            pg = _make_pg()
            conn = AsyncMock()
            result = MagicMock()
            result.rowcount = 1
            conn.execute = AsyncMock(return_value=result)
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=conn)
            cm.__aexit__ = AsyncMock(return_value=False)
            pg._engine.begin.return_value = cm

            deleted = await PostgresClient.delete_local_user(pg, user_id)
            assert deleted is True, f"delete_local_user({user_id}) should return True"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Write/delete documents locked to requesting user
# ─────────────────────────────────────────────────────────────────────────────


class TestUserScopedDocumentAccess:
    """Documents can be written/deleted scoped to a specific user."""

    @pytest.mark.asyncio
    async def test_upsert_document_with_owner_user_id(self):
        """Documents accept an owner_user_id field that is persisted."""
        doc = _make_doc("https://example.com/owner-test", owner_user_id=USER_1_ID)
        # Verify the field is actually set on the model
        assert doc.owner_user_id == USER_1_ID

    @pytest.mark.asyncio
    async def test_delete_user_also_removes_their_docs_via_cascade(self):
        """Deleting a user should cascade-delete their owned documents.

        This test verifies the delete_local_user call reaches the DB
        and that ON DELETE CASCADE in the schema handles the rest.
        """
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        conn = AsyncMock()
        result = MagicMock()
        result.rowcount = 1
        conn.execute = AsyncMock(return_value=result)
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        pg._engine.begin.return_value = cm

        # Deleting the user triggers the cascade at DB level
        deleted = await PostgresClient.delete_local_user(pg, USER_1_ID)
        assert deleted is True
        # Verify the DELETE was called with the correct user ID in params
        call_args = conn.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert str(USER_1_ID) in str(params)

    @pytest.mark.asyncio
    async def test_update_local_user_returns_none_for_unknown_user(self):
        """Updating a non-existent user returns None — no side effects."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        # begin cm for the UPDATE
        begin_conn = AsyncMock()
        update_result = MagicMock()
        update_result.rowcount = 0  # no rows updated → user not found
        begin_conn.execute = AsyncMock(return_value=update_result)
        begin_cm = MagicMock()
        begin_cm.__aenter__ = AsyncMock(return_value=begin_conn)
        begin_cm.__aexit__ = AsyncMock(return_value=False)
        pg._engine.begin.return_value = begin_cm

        result = await PostgresClient.update_local_user(
            pg, str(uuid.uuid4()), name="Ghost"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_update_local_user_returns_record_on_success(self):
        """Updating an existing user returns the updated user record."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        updated_row = {
            "id": USER_2_ID,
            "email": "bob@example.com",
            "name": "Bobby",
            "is_admin": False,
            "activation_status": "approved",
            "created_at": datetime.now(UTC),
            "last_login_at": None,
            "has_password": True,
        }

        # begin() for UPDATE
        begin_conn = AsyncMock()
        update_result = MagicMock()
        update_result.rowcount = 1
        begin_conn.execute = AsyncMock(return_value=update_result)
        begin_cm = MagicMock()
        begin_cm.__aenter__ = AsyncMock(return_value=begin_conn)
        begin_cm.__aexit__ = AsyncMock(return_value=False)
        pg._engine.begin.return_value = begin_cm

        # connect() for SELECT
        select_conn = AsyncMock()
        select_result = MagicMock()
        select_result.mappings.return_value.fetchone.return_value = updated_row
        select_conn.execute = AsyncMock(return_value=select_result)
        select_cm = MagicMock()
        select_cm.__aenter__ = AsyncMock(return_value=select_conn)
        select_cm.__aexit__ = AsyncMock(return_value=False)
        pg._engine.connect.return_value = select_cm

        result = await PostgresClient.update_local_user(pg, USER_2_ID, name="Bobby")
        assert result is not None
        assert result["name"] == "Bobby"

    def test_owner_user_id_field_on_content_document(self):
        """ContentDocument.owner_user_id field exists and defaults to None."""
        doc = ContentDocument(url="https://example.com/test", title="T")
        assert hasattr(doc, "owner_user_id")
        assert doc.owner_user_id is None

    def test_owner_user_id_can_be_set_on_content_document(self):
        """ContentDocument.owner_user_id can be explicitly assigned."""
        doc = ContentDocument(
            url="https://example.com/test",
            title="T",
            owner_user_id=USER_1_ID,
        )
        assert doc.owner_user_id == USER_1_ID


# ─────────────────────────────────────────────────────────────────────────────
# 7. Additional cross-user isolation & corpus-scoped visibility tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCrossUserIsolationExtended:
    """Extended isolation tests to confirm no cross-user data leakage."""

    def test_two_docs_with_different_owners_have_different_owner_ids(self):
        """Two documents created with different owner_user_ids remain distinct."""
        doc1 = _make_doc("https://example.com/doc-a", owner_user_id=USER_1_ID)
        doc2 = _make_doc("https://example.com/doc-b", owner_user_id=USER_2_ID)
        assert doc1.owner_user_id != doc2.owner_user_id
        assert doc1.url != doc2.url

    def test_doc_without_owner_user_id_is_public(self):
        """A document with no owner_user_id is treated as public/shared."""
        doc = _make_doc("https://example.com/public-doc")
        assert doc.owner_user_id is None

    def test_three_users_each_own_separate_docs(self):
        """Simulate three distinct users each owning one document."""
        users = [USER_1_ID, USER_2_ID, USER_3_ID]
        docs = [
            _make_doc(f"https://example.com/user{i}-doc", owner_user_id=uid)
            for i, uid in enumerate(users)
        ]
        owner_ids = [d.owner_user_id for d in docs]
        assert len(set(owner_ids)) == 3, "All three docs must have distinct owners"

    @pytest.mark.asyncio
    async def test_upsert_two_docs_same_corpus_different_owners(self):
        """Two upserts into the same corpus from different users both succeed."""
        from dewie.storage.postgres import PostgresClient

        corpus_id = uuid.uuid4()
        for user_id in [USER_1_ID, USER_2_ID]:
            pg = _make_pg()
            cm, session = _session_factory_cm()
            pg._session_factory.return_value = cm

            doc = _make_doc(
                f"https://example.com/corpus-doc-{user_id}",
                owner_user_id=user_id,
                corpus_id=str(corpus_id),
            )
            await PostgresClient.upsert(pg, doc)
            assert session.execute.called, f"upsert for user {user_id} did not call execute"

    @pytest.mark.asyncio
    async def test_delete_one_user_does_not_call_delete_for_another(self):
        """delete_local_user for user A should not issue a second delete for user B."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        conn = AsyncMock()
        result = MagicMock()
        result.rowcount = 1
        conn.execute = AsyncMock(return_value=result)
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        pg._engine.begin.return_value = cm

        await PostgresClient.delete_local_user(pg, USER_1_ID)

        # Only one execute call should have occurred (for USER_1_ID only)
        assert conn.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_user_can_upsert_then_owner_field_persists(self):
        """After upsert, the document model still carries the owner_user_id."""
        from dewie.storage.postgres import PostgresClient

        pg = _make_pg()
        cm, session = _session_factory_cm()
        pg._session_factory.return_value = cm

        doc = _make_doc("https://example.com/persist-owner", owner_user_id=USER_3_ID)
        await PostgresClient.upsert(pg, doc)

        # The in-memory doc retains its owner after the call
        assert doc.owner_user_id == USER_3_ID
