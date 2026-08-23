# Configuration

Settings are resolved in priority order:

1. Environment variables (highest)
2. `.env.local` (gitignored, local overrides)
3. `.env`
4. `dewie.yml`
5. Defaults (lowest)

Run `dewie setup` to generate a `dewie.yml` interactively. Set
`DEWIE_CONFIG_PATH` or `DEWIE_DATA_DIR` to control where it's read from.

---

## Database

| Variable | Default | Notes |
|----------|---------|-------|
| `POSTGRES_HOST` | — | Assemble DSN from parts (preferred) |
| `POSTGRES_PORT` | `5432` | |
| `POSTGRES_USER` | `dewie` | |
| `POSTGRES_PASSWORD` | `dewie` | **Change in production** |
| `POSTGRES_DB` | `dewie` | |
| `POSTGRES_DSN` | `postgresql+asyncpg://dewie:dewie@localhost:5432/dewie` | Override the full DSN directly |
| `REDIS_URL` | — | Empty = in-process queue/cache (single-node only) |
| `DEWIE_DATA_DIR` | `./data` | Root for flat-file body store and instance_id |

For SQLite (dev/eval only): set `POSTGRES_DSN=sqlite+aiosqlite:///./dewie.db`.

---

## API server

| Variable | Default | Notes |
|----------|---------|-------|
| `API_HOST` | `0.0.0.0` | |
| `API_PORT` | `10946` | |
| `API_WORKERS` | `4` | Uvicorn workers |
| `CORS_ORIGINS` | See below | JSON array: `["https://yourdomain.com"]` |
| `BASE_URL` | `http://localhost:10946` | Used in outbound links (password reset emails) |
| `RATE_LIMIT_RPM` | `60` | Requests per minute per client IP |
| `CACHE_TTL_SECONDS` | `300` | Search result cache TTL |

Default CORS origins: `https://dewie.ai`, `http://localhost:10946`, `http://localhost:3000`.

---

## Auth

| Variable | Default | Notes |
|----------|---------|-------|
| `AUTH_ENABLED` | `true` | `false` = open access (dev only) |
| `LOCAL_AUTH_ENABLED` | `false` | Treat all requests as a local admin — dev only |
| `LOCAL_AUTH_USER_ID` | `00000000-…-000002` | Synthetic user ID for local auth |
| `LOCAL_AUTH_EMAIL` | `Dewie Local Catalog` | Identity shown in `/auth/me` |
| `LOCAL_AUTH_IS_ADMIN` | `true` | |
| `ADMIN_EMAIL` | `admin` | Seeded on first start when `users` table is empty |
| `ADMIN_PASSWORD` | `admin` | **Change before deploying** |
| `INTERNAL_SERVICE_KEY_REQUIRED` | `false` | Require `INTERNAL_SERVICE_KEY` on `/ingest` — recommended in production |
| `ENCRYPTION_MASTER_KEY` | — | Base64 32-byte Fernet key for encrypting LLM API keys at rest |

---

## LLM providers

Providers are configured via `dewie.yml` under the `servers:` key. Each server
entry has:

```yaml
servers:
  - label: local           # label wired into chat_server_aq / embed_server / etc.
    api_format: openai      # openai | anthropic
    endpoint: http://localhost:8080/v1
    api_key_env: MY_KEY     # env var holding the key (not the key itself)
    extra_body:             # merged into every request payload
      thinking_budget_tokens: 0
```

Built-in servers you can reference without declaring them: `openai`, `anthropic`,
and `openrouter` (all OpenAI-compatible except `anthropic`). Just set the matching
`*_API_KEY` env var and point a step's server label at them.

Then wire the labels to pipeline steps:

| Variable | Default | Notes |
|----------|---------|-------|
| `CHAT_SERVER_AQ` | — | Server label for enrichment (AQ generation) |
| `CHAT_MODEL_AQ` | — | Model name for enrichment |
| `CHAT_SERVER_KE` | — | Server label for KE extraction; falls back to AQ server |
| `CHAT_MODEL_KE` | — | Model for KE extraction; falls back to AQ model |
| `ENRICHMENT_MODE` | `single_pass` | `single_pass` (one LLM call) or `dual_pass` (separate AQ + KE calls) |
| `ENRICHMENT_MODEL` | — | Legacy: model name when not using the server registry |
| `OPENAI_API_TYPE` | `chat/completions` | `chat/completions` or `v1/responses` |

---

## Embeddings

| Variable | Default | Notes |
|----------|---------|-------|
| `EMBED_SERVER` | `local` | Server label, or `local` for in-process embeddings |
| `EMBED_MODEL` | `ggml-org/embeddinggemma-300m-qat-q8_0-GGUF` | Default: EmbeddingGemma-300m as a public GGUF, run in-process via llama.cpp (no API key, no HF gate). A GGUF spec uses `llama-cpp-python`; a plain model name with `EMBED_SERVER=local` uses sentence-transformers; otherwise the model is sent to `EMBED_SERVER`. |
| `EMBED_DIMENSIONS` | auto | Override vector size (MRL truncation applied) |
| `EMBED_STORE_FULL_VECTOR` | `false` | Also persist the untruncated vector for exact reranking |
| `LOCAL_EMBED_ALLOWED` | `true` | Host-level switch. `false` disables `EMBED_SERVER=local` regardless of `dewie.yml` — for managed/multi-tenant hosts. |

In-process embedding needs the optional `[local]` extra (`pip install "dewie[local]"`);
without it, embedding degrades gracefully and documents stay full-text searchable.

Auto-detected dimensions: `embeddinggemma` → 768, `text-embedding-3-small` → 1536,
`text-embedding-3-large` → 3072, `nomic-embed-text` → 768,
`qwen3-embedding-{0.6b,4b,8b}` → 1024/2560/4096, others → 1536.

---

## Enrichment pipeline

| Variable | Default | Notes |
|----------|---------|-------|
| `ENRICHMENT_WORKERS` | `1` | Concurrent enrichment loops. On Postgres, docs are claimed atomically (`FOR UPDATE SKIP LOCKED`) so workers never collide; SQLite stays single-worker. Match to the inference server's parallel slots. |
| `ENRICHMENT_BATCH_SIZE` | `2` | Docs per tick; set `1` for local LLMs |
| `ENRICHMENT_SLEEP_SECS` | `30` | Sleep between ticks when queue is empty |
| `MAX_ENRICHMENT_RETRIES` | `3` | Attempts before `FAILED` |
| `MAX_EXTRACTION_CHARS` | `80000` | Body chars sent to LLM (truncated beyond this) |
| `MAX_SUMMARY_CHARS` | `1500` | Summary field hard cap |
| `SAVE_RAW_DOCUMENTS` | `false` | Save raw bodies to `ingested_docs/` on disk |
| `ENRICHMENT_DEFAULT_BACKEND` | `passthrough` | Backend when no routing rule matches |
| `ENRICHMENT_FALLBACK_BACKEND` | `passthrough` | Backend on primary failure |
| `ENRICHMENT_MAX_RETRIES` | `2` | Backend attempts per document |

---

## Search

| Variable | Default | Notes |
|----------|---------|-------|
| `QUERY_DEFAULT_RANKER` | `rrf_chunks` | Default ranking strategy |
| `DEWIE_ENABLED_RANKERS` | JSON array | Rankers visible in UI and API |
| `QUERY_LOG_SAVE_FULL_RESULTS` | `true` | Store full ranked result in query log |
| `DEFAULT_MAX_DEPTH` | `3` | Max hops for graph traversal |
| `ABSOLUTE_MAX_DEPTH` | `10` | Hard cap |
| `MAX_NODES_PER_LEVEL` | `20` | Graph breadth limit |
| `QUERY_TIMEOUT_SECONDS` | `30` | |

---

## Web search

| Variable | Default | Notes |
|----------|---------|-------|
| `SEARCH_PROVIDER` | — | `brave` / `exa` / `you` — empty disables web fallback |
| `BRAVE_API_KEY` | — | |
| `EXA_API_KEY` | — | |
| `YOU_API_KEY` | — | |

---

## Crawler

| Variable | Default | Notes |
|----------|---------|-------|
| `CRAWLER_MAX_DEPTH` | `2` | |
| `CRAWLER_MAX_PAGES` | `100` | |
| `CRAWLER_CONCURRENCY` | `3` | |
| `CRAWLER_POLITENESS_DELAY` | `1.0` | Seconds between requests |
| `CRAWLER_SAME_DOMAIN` | `true` | Don't follow off-domain links |
| `CRAWLER_REQUEST_TIMEOUT` | `15.0` | |
| `USER_AGENT` | Chrome/124 | Outbound User-Agent |

---

## Email (password reset)

| Variable | Default |
|----------|---------|
| `SMTP_HOST` | — |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | — |
| `SMTP_PASSWORD` | — |
| `SMTP_FROM_EMAIL` | — |

---

## Browse sessions

| Variable | Default | Notes |
|----------|---------|-------|
| `BROWSE_SESSION_TTL_SECONDS` | `14400` | 4 hours; Redis TTL for browse sessions |
| `BROWSE_MAX_NEIGHBORS` | `20` | Default neighbors per expand call |

---

## Subsystem flags

| Variable | Default | Notes |
|----------|---------|-------|
| `ENABLE_API` | `true` | `false` = worker-only mode |
| `ENABLE_ENRICHMENT` | `true` | |
| `ENABLE_INGESTION` | `true` | RSS/background ingest workers |
| `ENABLE_POLLER` | `true` | Background tasks (backfill, cluster rebuild) |

---

## Compliance / audit

| Variable | Default | Notes |
|----------|---------|-------|
| `AUDIT_LOG_ENABLED` | `true` | |
| `RETENTION_POLICY_ENABLED` | `false` | |
| `RETENTION_DAYS` | `365` | |
| `PII_SCAN_ENABLED` | `false` | |
| `ANOMALY_DETECTION_ENABLED` | `false` | |
