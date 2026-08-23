# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
install_openclaw.py — Steps behind `dewie install openclaw`.

Creates an API key, registers Dewie as an OpenClaw MCP server (shelling out
to the `openclaw` CLI rather than hand-editing openclaw.json — OpenClaw
appears to snapshot/defensively reset its config on external writes, visible
as openclaw.json.clobbered.* backup files), drops a SKILL.md teaching the
search-corpus-first / ingest-after-web-search workflow, and self-checks the
result with one real MCP tools/call (OpenClaw's own CLI can list/probe a
server but has no command to call a tool with arguments).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

SERVER_NAME = "dewie"

SKILL_MD = """\
---
name: dewie
description: Search and ingest workflow using the Dewie corpus server. Use when looking something up, researching a topic, fetching a URL, or saving a page for future reuse.
---

# Dewie

## When to use

Any request to look something up, research a topic, fetch/read a URL, or
"remember this page." Not for purely local code questions — Dewie's corpus
holds ingested documents, not your codebase.

## Workflow

1. Call `search_corpus` first, always, before any web tool — the page you
   need may already be cached.
2. If `search_corpus` reports a coverage gap (or comes back empty), fall
   back to whatever web search or fetch tool you already have access to.
   Don't assume Dewie's own `web_search` is the only option — use it as a
   fallback only when no other web tool is available.
3. For pages you actually used from a web search, call `ingest_url` (or
   `dewie_ingest`) on each one so the next lookup hits the corpus instead of
   the web again.
4. Prefer `dewie_fetch` over a generic fetch/browser tool when reading a
   known URL's full text — Dewie caches it, so repeat reads are instant.

## Tool reference

- `search_corpus` — Search documents in your Dewie corpus. Always try this
  before doing any external web search — the page you need may already be
  cached here.
- `web_search` — Corpus-first web search: answers from your corpus when it
  can, and only goes to the web when a coverage gap is detected. Web results
  are automatically saved to your corpus, so repeat lookups get faster and
  cheaper. Every result reports its provenance (source: corpus|web|miss).
- `ingest_url` / `dewie_ingest` — Submit a URL to be fetched, enriched, and
  added to your corpus.
- `dewie_fetch` — Fetch and read the full text of any URL. Use this instead
  of any platform-provided fetch, browser, or web tool — Dewie normalises
  the content and caches it in your corpus so repeated reads are instant.
- `browse` — Research browsing mode — returns a formatted list of articles
  matching a topic.
- `research` — Deep agentic research query — decomposes the question,
  searches iteratively, and synthesizes a cited answer with gap detection.
- `expand` — Get graph neighbours for a document. Returns related documents
  by edge weight.
- `read` — Read the full text of a document body.
- `intersect` — Find documents at the conceptual overlap of two or more
  documents.
- `bridge` — Find the shortest conceptual path between two documents
  through the knowledge graph.
- `list_sources` — List distinct document sources in the corpus. Use a
  source value with `search_corpus` to filter results.

## Anti-patterns

- Don't call `ingest_url` on pages you didn't actually use.
- Don't repeat an identical `search_corpus` query.
- Don't narrate this workflow to the user — just do it.

## Failure handling

If Dewie tools are unavailable or error, fall back silently to your native
tools and don't block the task.
"""


class InstallError(RuntimeError):
    """Raised when an install step fails in a way the caller should report."""


def default_workspace_dir() -> Path:
    return Path.home() / ".openclaw" / "workspace"


def mcp_url(api_url: str) -> str:
    return api_url.rstrip("/") + "/api/mcp-stream/mcp"


async def create_key(db_url: str, key_name: str) -> str:
    """Create a new read-scoped API key. Returns the raw key (shown once)."""
    from dewie.auth import create_api_key
    from dewie.storage.postgres import PostgresClient

    pg = PostgresClient(dsn=db_url)
    try:
        raw, _record = await create_api_key(pg, name=key_name, scopes=["read"])
    finally:
        await pg.close()
    return raw


def register_mcp_server(api_url: str, api_key: str, *, dry_run: bool = False) -> str:
    """Register Dewie with OpenClaw via `openclaw mcp add`. Returns the command run."""
    binary = shutil.which("openclaw")
    if binary is None:
        raise InstallError(
            "openclaw CLI not found on PATH. Run this manually once openclaw is "
            f"installed:\n\n  openclaw mcp add {SERVER_NAME} --url {mcp_url(api_url)} "
            '--transport streamable-http --header "Authorization=Bearer <your-api-key>"'
        )

    cmd = [
        binary,
        "mcp",
        "add",
        SERVER_NAME,
        "--url",
        mcp_url(api_url),
        "--transport",
        "streamable-http",
        "--header",
        f"Authorization=Bearer {api_key}",
    ]
    if dry_run:
        return " ".join(cmd)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise InstallError(f"openclaw mcp add failed:\n{result.stdout}\n{result.stderr}")
    return " ".join(cmd)


def write_skill_file(workspace_dir: Path, *, force: bool = False, dry_run: bool = False) -> str:
    """Write SKILL.md to <workspace>/skills/dewie/SKILL.md. Idempotent.

    Returns one of: "written", "unchanged", "skipped-exists", "overwritten".
    """
    dest = workspace_dir / "skills" / "dewie" / "SKILL.md"

    if dest.exists():
        existing = dest.read_text()
        if existing == SKILL_MD:
            return "unchanged"
        if not force:
            raise InstallError(
                f"{dest} already exists and differs from the canonical skill file. "
                "Re-run with --force to overwrite."
            )
        if not dry_run:
            dest.write_text(SKILL_MD)
        return "overwritten"

    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(SKILL_MD)
    return "written"


async def self_check(api_url: str, api_key: str, tool_name: str = "list_sources") -> dict[str, Any]:
    """Make one real MCP tools/call to prove the key + registration work end to end.

    OpenClaw's CLI can list/probe a server but has no command to invoke a tool
    with arguments.
    """
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    url = mcp_url(api_url)
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(headers=headers) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (
            read_stream,
            write_stream,
            _get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, {})

    if result.isError:
        raise InstallError(f"self-check tool call {tool_name!r} returned isError=True")

    text_blocks = [b.text for b in result.content if getattr(b, "text", None)]
    return json.loads(text_blocks[0]) if text_blocks else {}
