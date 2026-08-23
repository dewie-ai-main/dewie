# Changelog

All notable changes to Dewie are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/); versioning is semantic-ish while pre-1.0.

## [0.1.0] — 2026-07-29

Initial public release.

### Highlights
- **Self-hostable knowledge corpus** — ingest content, enrich it with an LLM
  (summary, keywords, entities, answers-questions), and search it semantically
  and full-text. Exposed as a REST API and as an **MCP server** for agents.
- **Zero-config local embeddings** — a fresh install embeds in-process with
  EmbeddingGemma-300m (GGUF via llama.cpp): no API key, external server, or
  HuggingFace license gate. Falls back to full-text search if the optional
  `[local]` dependency is absent. `LOCAL_EMBED_ALLOWED` lets a managed host
  gate in-process embedding per account.
- **Pluggable enrichment providers** — local (llama.cpp / Ollama), OpenAI,
  Anthropic, and OpenRouter, selected via a server registry.
- **Parallel enrichment workers** — the `enrichment_workers` setting runs
  concurrent enrichment loops; Postgres claims documents atomically
  (`FOR UPDATE SKIP LOCKED`) so workers never collide.
- **Storage** — Postgres + pgvector for semantic search, or SQLite for a
  zero-dependency full-text-only setup.
- **Ingestion** — URLs, RSS feeds, podcasts (with transcription), and documents
  (PDF / Word / Excel / PowerPoint).
- **MCP setup helper** — `dewie install <harness>` registers Dewie with an MCP
  client so agents can search and ingest through it.
- **Ops** — Docker image and docker-compose stacks (pgvector and SQLite),
  Alembic migrations, API-key auth, and a CI suite (lint, unit + e2e, live
  SQLite smoke, and a Docker end-to-end gate).
