# Quickstart

## Option A — SQLite, no Docker

```bash
pip install "dewie[local]"   # [local] enables in-process embeddings (EmbeddingGemma)
dewie setup      # interactive wizard, writes dewie.yml
dewie serve      # API on http://localhost:10946
```

Ingest something and search it:

```bash
dewie ingest https://en.wikipedia.org/wiki/Retrieval-augmented_generation
curl -s localhost:10946/api/query -X POST -H 'Content-Type: application/json' \
  -d '{"query":"what is RAG"}' | jq .
```

SQLite is the default. It works end-to-end but uses in-process backends for the
queue and cache — fine for personal use or evaluation, not for production.

## Option B — Docker Compose (Postgres + pgvector + Redis)

```bash
git clone https://github.com/dewie-ai-main/dewie
cd dewie
cp .env.example .env          # edit ADMIN_EMAIL, ADMIN_PASSWORD, LLM settings
docker compose up -d
```

The compose file boots two services: `postgres` (with pgvector) and `app`. On first
start the app runs all migrations and seeds the admin user.

Verify it's up:

```bash
curl http://localhost:10946/health
```

## Connect your agent (MCP)

Dewie serves the MCP protocol at `/api/mcp`. Any MCP-capable agent works —
here's Claude Code:

```bash
claude mcp add --transport http dewie http://localhost:10946/api/mcp
```

Or edit `.claude/settings.json` directly:

```json
{
  "mcpServers": {
    "dewie": {
      "transport": "http",
      "url": "http://localhost:10946/api/mcp"
    }
  }
}
```

Your agent now has `search_corpus`, `read`, `expand`, `intersect`, `bridge`,
`browse`, `research`, `ingest_url`, and `web_search`. See
[docs/mcp-tools.md](mcp-tools.md) for what each does.

## First ingest

Dewie accepts URLs, RSS feeds, and file uploads:

```bash
# Single URL
dewie ingest https://example.com/article

# RSS feed (polled on a schedule) — see docs/configuration.md for options
curl -X POST localhost:10946/api/feeds \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/feed.xml","name":"Example"}'
```

Ingestion is fire-and-forget (202). The enrichment pipeline picks docs up in
the background: fetch → extract → LLM enrich → embed → index. A typical doc
takes 30–120 seconds depending on body length and LLM speed.

Watch progress in the admin dashboard, or by checking a document's status via
the API (`GET /api/documents`). Enrichment runs in the background.

## Set up a web search provider (optional)

To give `web_search` a web fallback, add to `.env`:

```bash
SEARCH_PROVIDER=brave
BRAVE_API_KEY=your_key_here
```

Supported providers: `brave`, `exa`, `you`. Without a provider, `web_search`
is corpus-only.

## What's next

- [docs/configuration.md](configuration.md) — all env vars and `dewie.yml` fields
- [docs/mcp-tools.md](mcp-tools.md) — full reference for the nine agent tools
- [docs/deployment.md](deployment.md) — production hardening, TLS, auth
