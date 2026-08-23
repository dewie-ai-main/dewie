# MCP Tools Reference

Dewie exposes nine tools at `POST /api/mcp`. Every MCP-capable agent can use
them. The manifest (tool names + schemas) is at `GET /api/mcp`.

---

## `search_corpus`

Hybrid search over the corpus: BM25 + vector + `answers_questions` signal,
fused with Reciprocal Rank Fusion.

| Parameter | Type | Required | Default | Notes |
|-----------|------|----------|---------|-------|
| `query` | string | yes | — | Natural language query |
| `limit` | int | no | 10 | Max results, capped at 25 |
| `ranker` | string | no | `rrf_chunks` | Ranking strategy (see below) |

**Rankers:**
- `rrf_chunks` — default; BM25 + vector over chunks, RRF-fused
- `rrf` — BM25 + vector over full documents
- `answers_questions_rrf` — weighted toward the `answers_questions` signal
- `bm25` — keyword only
- `vector` — embedding similarity only

Returns: array of `{doc_id, url, title, score, summary, topics, reading_level}`.

---

## `read`

Fetch a document's full body from the corpus. Use after `search_corpus` when
the summary isn't enough.

| Parameter | Type | Required |
|-----------|------|----------|
| `doc_id` | string (UUID) | yes |

Returns: `{doc_id, url, title, body, enriched_at}`.

---

## `expand`

Returns the highest-relevance neighbors of a document in the knowledge graph.
Edges are computed at ingest time — this is a database lookup, not a similarity
search.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `doc_id` | string (UUID) | yes | — |
| `limit` | int | no | 20 |

Returns: array of neighbor documents ordered by edge weight.

---

## `intersect`

Finds what connects a set of documents — shared topics, entities, or concepts
that appear across all of them.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `doc_ids` | array of strings | yes | — |
| `limit` | int | no | 10 |

Useful for: "what do these five search results have in common?"

---

## `bridge`

Finds the shortest path between two documents in the knowledge graph.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `source_id` | string (UUID) | yes | — |
| `target_id` | string (UUID) | yes | — |
| `max_hops` | int | no | 5 (max 8) |

Returns: the path as an ordered list of documents. Useful for understanding how
two apparently unrelated documents are connected.

---

## `browse`

Search + expand in one call. Returns search results and the neighborhood of
each result. Designed for exploratory research sessions.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `query` | string | yes | — |
| `limit` | int | no | 10 (max 15) |
| `ranker` | string | no | `rrf_aq` |

---

## `research`

Multi-step research loop. Runs multiple searches, synthesizes, and returns a
structured answer with sources. Slower than `search_corpus` but better for
open-ended questions.

| Parameter | Type | Required | Default | Notes |
|-----------|------|----------|---------|-------|
| `query` | string | yes | — | |
| `depth` | string | no | `quick` | `quick` / `thorough` |
| `max_iterations` | int | no | 3 | |
| `web_fallback` | bool | no | false | Fall back to web if corpus coverage is low |

---

## `ingest_url`

Fire-and-forget URL ingestion. Returns immediately (202); enrichment happens
in the background.

| Parameter | Type | Required |
|-----------|------|----------|
| `url` | string | yes |

The agent doesn't need to wait. Use `search_corpus` for the same URL after
~60–120 seconds.

---

## `web_search`

Corpus-first web search. Before hitting the web, checks whether the corpus
already covers the query (using the `answers_questions` signal). Falls back to
the configured web provider only when coverage is low.

Everything fetched from the web is auto-ingested into the corpus — the agent's
reading trail accumulates there over time.

| Parameter | Type | Required | Default | Notes |
|-----------|------|----------|---------|-------|
| `query` | string | yes | — | |
| `limit` | int | no | 5 (max 10) | |
| `force_web` | bool | no | false | Skip corpus check, go straight to web |
| `corpus_only` | bool | no | false | Never fall back to web |

Results include `source: corpus | web | miss` and `ingested_at`.

**Requires** `SEARCH_PROVIDER` to be set for web fallback. Providers:
`brave` (`BRAVE_API_KEY`), `exa` (`EXA_API_KEY`), `you` (`YOU_API_KEY`).
Exa and You return page text directly; Brave results are fetched and extracted
before ingest.

### Replacing your agent's built-in web tools

```bash
# Claude Code — disable built-in web tools, use dewie instead
# Add to .claude/settings.json:
{ "permissions": { "deny": ["WebFetch", "WebSearch"] } }
```

Over time the corpus self-builds from everything the agent reads.

---

## Authentication

By default auth is enabled. Pass your API key:

```
X-API-Key: ck_live_...
```

Or use a session token from `POST /auth/login`. In local dev with
`LOCAL_AUTH_ENABLED=true`, the header is not required.
