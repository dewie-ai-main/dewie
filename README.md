# Dewie

**Agent-native retrieval: navigation tools over a self-enriching corpus, so small models can punch above their weight.**

Most retrieval hands your agent top-k chunks and hopes the model is big enough to figure out the rest. Dewie does the expensive thinking once, at ingest — LLM enrichment extracts topics, entities, summaries, and the *questions each document answers* — then gives your agent cheap navigation primitives over the resulting graph. Multi-hop questions become a sequence of small tool calls instead of one giant context window.

> The fetch → enrich → embed → search core is stable.

## The tools your agent gets (MCP)

Dewie speaks [MCP](https://modelcontextprotocol.io) natively at `/api/mcp` — any MCP-capable agent can use it without an SDK.

| Tool | What it does |
|---|---|
| `search_corpus` | Hybrid search (BM25 + vector, RRF-fused) over the corpus |
| `read` | Fetch a document's full body |
| `expand` | Highest-relevance neighbors of a document in the graph |
| `intersect` | What connects a set of documents |
| `bridge` | Path between two documents |
| `browse` | Search + navigate in one call |
| `research` | Multi-step research loop over corpus + web |
| `ingest_url` | Fire-and-forget ingestion of a URL |
| `web_search` | **Corpus-first web search**: serves from your corpus when it can, falls back to the web only on a detected coverage gap, and saves what it fetches — so your agent's corpus builds itself |

### `web_search`: the self-building corpus

`web_search` replaces your agent's built-in web tools, so every lookup becomes corpus-first. The gate is the semantic gap signal (not a score threshold), every result carries provenance (`source: corpus | web | miss`, `ingested_at`), and agents can override with `force_web` (freshness matters) or `corpus_only`. Configure the backup search engine:

```bash
SEARCH_PROVIDER=brave   # brave | exa | you   (empty = corpus-only, no web fallback)
BRAVE_API_KEY=...       # or EXA_API_KEY / YOU_API_KEY
```

Exa and You.com return page text with results (no second fetch); Brave results are fetched + extracted before ingest.

### Wiring it into Claude Code

```bash
# 1. Point Claude Code at dewie's MCP endpoint
claude mcp add --transport http dewie http://localhost:10946/api/mcp

# 2. Prefer dewie's corpus-first tools over the built-ins (.claude/settings.json)
{ "permissions": { "deny": ["WebFetch", "WebSearch"] } }
```

From then on the agent's web lookups flow through your corpus, and everything it reads accumulates there.

## Optional Dependencies

Some features require additional packages:

- **Podcast Transcription**: `pip install "dewie[podcast]"` (enables `openai-whisper` or `faster-whisper`)
- **Media/YouTube**: `pip install "dewie[media]"` (enables `yt-dlp` and `youtube-transcript-api`)

## Quickstart (SQLite, no Docker)

```bash
pip install "dewie[local]"   # [local] enables in-process embeddings (EmbeddingGemma)
dewie setup                  # interactive wizard, writes dewie.yml
dewie serve                  # API on http://localhost:10946
dewie ingest https://en.wikipedia.org/wiki/Retrieval-augmented_generation
```

Then point your agent at the MCP endpoint, or poke it by hand:

```bash
curl -s localhost:10946/api/mcp | jq .                   # list the available tools
curl -s localhost:10946/api/query -X POST -H 'Content-Type: application/json' \
  -d '{"query": "what is RAG"}' | jq .
```

### Production-ish (Postgres + pgvector + Redis)

```bash
docker compose up -d         # pgvector, redis, app
```

See `docker-compose.yml`; configuration is env-var driven (`.env` supported, `generate-env.sh` scaffolds one).

## How it works

```
ingest (fire-and-forget, 202)
   └─► enrichment workers: topics, entities, summary, answers_questions, chunks, embeddings
          └─► relationship edges computed at write time (Postgres/pgvector or SQLite)
                 └─► agent navigates: search / expand / intersect / bridge   (MCP or REST)
```

- **Enrich at write time, navigate at read time.** Query-time work is database lookups, not reasoning — which is what lets small models drive it.
- **`answers_questions`** — each document is indexed by the questions it can answer (embedded + full-text). It's a hidden ranking signal, never exposed in API responses.
- **Runs locally by default — no API key.** A fresh install embeds in-process with EmbeddingGemma (GGUF via llama.cpp): no external embedding service, no HuggingFace license gate. Falls back to full-text search if the optional `[local]` extra isn't installed.
- **Bring your own model for enrichment.** Point `chat_server_aq` / `embed_server` at any registered server — OpenAI, Anthropic, OpenRouter, Ollama, vLLM, or your own llama.cpp endpoint. See [`docs/configuration.md`](docs/configuration.md).
- **Verification UI.** `static/` ships inspection pages (walk, inspect, query-inspector, retrieval-explorer) so you can audit what the system actually did.

## Layout

```
src/dewie/         core: api/routes, storage (postgres, rankers, cache), enrichment, ingestion, workers
static/            verification / inspection UI
tests/             unit, integration, performance
```

## Documentation

| Doc | What's in it |
|---|---|
| [`docs/quickstart.md`](docs/quickstart.md) | Zero to a working corpus |
| [`docs/mcp-tools.md`](docs/mcp-tools.md) | Full reference for the nine agent tools |
| [`docs/configuration.md`](docs/configuration.md) | Every env var and `dewie.yml` field |
| [`docs/deployment.md`](docs/deployment.md) | SQLite quickstart → production hardening |
| [`SECURITY.md`](SECURITY.md) | Reporting + deployment hardening checklist |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Ground rules, CI gates, PR checklist |
| [`CHANGELOG.md`](CHANGELOG.md) | What changed |

## Development

```bash
pip install -e ".[dev]"
pytest tests/unit tests/e2e -q --no-cov   # fast suites (CI also runs the live smoke)
ruff check src tests
./scripts/smoke_sqlite.sh                  # boot a real server, drive the full loop
```

## License

[FSL-1.1-ALv2](LICENSE) (Functional Source License). You can use, modify, and self-host Dewie for anything except offering it as a competing commercial service — and **each release automatically becomes Apache-2.0 two years after it ships**. See [fsl.software](https://fsl.software) for the rationale behind this license family.
