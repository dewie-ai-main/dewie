# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
dewie/storage/tenant_isolation.py — Multi-tenant corpus isolation.

Implements the workspace → corpus → document hierarchy for customer isolation.
Each tenant gets their own workspace and corpus; API keys are scoped to that
workspace so cross-tenant data is never accessible.

Usage::

    router = TenantCorpusRouter(pg)

    # Provision a new tenant (once, at signup)
    result = await router.provision_tenant("acme-corp", "ACME Corporation")
    # result.api_key  — give to customer (shown once)
    # result.workspace_id, result.corpus_id

    # Enforce corpus access on every read/write operation
    await router.assert_corpus_access(corpus_id, allowed_workspace_ids)
    # Raises CorpusAccessDenied if corpus is not in the allowed workspaces
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from dewie.auth import SCOPE_INGEST, SCOPE_READ, create_api_key

__all__ = [
    "TenantCorpusRouter",
    "TenantProvisionResult",
    "CorpusAccessDenied",
]

# Well-known root workspace ID (seeded in init_schema)
ROOT_WORKSPACE_ID = uuid.UUID("00000000-0000-0000-0000-000000000010")


class CorpusAccessDenied(PermissionError):
    """Raised when a caller attempts to access a corpus outside their workspaces."""

    def __init__(self, corpus_id: uuid.UUID, workspace_ids: list[uuid.UUID]) -> None:
        self.corpus_id = corpus_id
        self.workspace_ids = workspace_ids
        ws_str = ", ".join(str(w) for w in workspace_ids) or "<none>"
        super().__init__(
            f"Corpus {corpus_id} is not accessible from workspace(s): {ws_str}"
        )


@dataclass
class TenantProvisionResult:
    """Returned by :meth:`TenantCorpusRouter.provision_tenant`."""

    tenant_slug: str
    workspace_id: uuid.UUID
    corpus_id: uuid.UUID
    api_key: str  # plaintext — show to customer once, never store


def _slugify(name: str) -> str:
    """Convert a name to a URL-safe slug, e.g. 'ACME Corp' → 'acme-corp'."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:63] or "tenant"


class TenantCorpusRouter:
    """
    High-level interface for multi-tenant corpus isolation.

    Each *tenant* owns exactly one workspace and one primary corpus.
    Workspace IDs are embedded in API keys (via ``workspace_ids``), so the
    existing :func:`dewie.auth.verify_api_key` + middleware path automatically
    restricts every search/ingest call to the tenant's workspace.

    This class handles:
    - Provisioning: create workspace + corpus + scoped API key in one call.
    - Access enforcement: raise ``CorpusAccessDenied`` when a corpus does not
      belong to any of the caller's allowed workspaces.
    - Corpus lookup: resolve a corpus by ID and return its workspace.
    """

    def __init__(self, pg: Any) -> None:
        """
        Args:
            pg: A :class:`dewie.storage.postgres.PostgresClient` instance.
        """
        self._pg = pg

    # ── Provisioning ──────────────────────────────────────────────────────────

    async def provision_tenant(
        self,
        slug: str | None = None,
        display_name: str = "New Tenant",
        *,
        scopes: list[str] | None = None,
    ) -> TenantProvisionResult:
        """
        Create a new isolated tenant: workspace + corpus + scoped API key.

        Args:
            slug: URL-safe identifier for the tenant. Auto-derived from
                  ``display_name`` when omitted.
            display_name: Human-readable name for the workspace/corpus.
            scopes: API key scopes. Defaults to ``["read", "ingest"]``.

        Returns:
            :class:`TenantProvisionResult` with workspace_id, corpus_id, and
            the plaintext API key (shown once — never persisted in plaintext).
        """
        if scopes is None:
            scopes = [SCOPE_READ, SCOPE_INGEST]

        tenant_slug = slug or _slugify(display_name)

        # 1. Create a private workspace for this tenant
        workspace = await self._pg.create_workspace(
            name=display_name,
            sharing_tier="private",
        )
        workspace_id = uuid.UUID(str(workspace["id"]))

        # 2. Create the tenant's primary corpus inside that workspace
        corpus = await self._pg.create_corpus(
            name=f"{display_name} Corpus",
            slug=tenant_slug,
            workspace_id=workspace_id,
            sharing_tier="private",
        )
        corpus_id = uuid.UUID(str(corpus["id"]))

        # 3. Create an API key scoped to this workspace only
        raw_key, _record = await create_api_key(
            self._pg,
            workspace_ids=[workspace_id],
            name=f"{tenant_slug}-key",
            scopes=scopes,
        )

        return TenantProvisionResult(
            tenant_slug=tenant_slug,
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            api_key=raw_key,
        )

    # ── Access enforcement ─────────────────────────────────────────────────────

    async def assert_corpus_access(
        self,
        corpus_id: uuid.UUID,
        allowed_workspace_ids: list[uuid.UUID],
    ) -> None:
        """
        Raise :class:`CorpusAccessDenied` if ``corpus_id`` is not in one of
        ``allowed_workspace_ids``.

        When ``allowed_workspace_ids`` is empty the caller is an admin (or auth
        is disabled) and access is unconditionally granted.

        Args:
            corpus_id: The corpus the caller wants to access.
            allowed_workspace_ids: Workspace UUIDs from the API key record
                (``request.state.workspace_ids``).

        Raises:
            CorpusAccessDenied: Corpus does not belong to any allowed workspace.
        """
        # Empty allowed list = admin / no restriction
        if not allowed_workspace_ids:
            return

        workspace_id = await self.get_corpus_workspace(corpus_id)
        if workspace_id is None:
            # Corpus doesn't exist — deny by default (fail-closed)
            raise CorpusAccessDenied(corpus_id, allowed_workspace_ids)

        if workspace_id not in allowed_workspace_ids:
            raise CorpusAccessDenied(corpus_id, allowed_workspace_ids)

    async def get_corpus_workspace(self, corpus_id: uuid.UUID) -> uuid.UUID | None:
        """
        Return the workspace_id for a given corpus, or ``None`` if not found.

        Args:
            corpus_id: UUID of the corpus to look up.
        """
        from sqlalchemy import text

        async with self._pg._engine.connect() as conn:
            row = (
                await conn.execute(
                    text("SELECT workspace_id FROM corpora WHERE id = :id"),
                    {"id": str(corpus_id)},
                )
            ).fetchone()
        if row is None:
            return None
        return uuid.UUID(str(row[0]))

    async def set_search_path(self, schema: str) -> None:
        """
        Set the PostgreSQL ``search_path`` to a tenant's schema within the current
        transaction.

        Uses ``SET LOCAL`` so the setting survives only for the duration of the
        current transaction — this is required for PgBouncer / Supavisor in
        transaction mode where pool handoffs would otherwise lose session-level
        settings.

        Args:
            schema: The tenant's database schema name (must be a valid identifier).

        Raises:
            ValueError: If ``schema`` is empty or contains unsafe characters.
        """
        if not schema or not re.match(r"^[a-z_][a-z0-9_]*$", schema):
            raise ValueError(f"Invalid schema name: {schema!r}")

        from sqlalchemy import text

        async with self._pg._engine.begin() as conn:
            await conn.execute(
                text(f'SET LOCAL search_path TO "{schema}", public')
            )

    # ── Tenant corpus listing ──────────────────────────────────────────────────

    async def list_tenant_corpora(
        self, workspace_ids: list[uuid.UUID]
    ) -> list[dict]:  # type: ignore[type-arg]
        """
        Return all corpora visible to the given workspaces.

        Args:
            workspace_ids: Allowed workspace UUIDs (from API key). Empty = all.
        """
        if not workspace_ids:
            return await self._pg.get_corpora()

        results: list[dict] = []  # type: ignore[type-arg]
        for wid in workspace_ids:
            results.extend(await self._pg.get_corpora(workspace_id=wid))
        return results
