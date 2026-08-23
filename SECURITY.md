# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Email **dev@dewie.ai** with:

- a description of the issue and the affected component
- reproduction steps or a proof of concept
- the version / commit you tested against

You'll get an acknowledgment within a few days. Coordinated disclosure preferred; credit given unless you'd rather stay anonymous.

## Deployment hardening checklist

Dewie ships local-first with permissive defaults. Before exposing an instance beyond localhost:

| Setting | Default | Production requirement |
|---|---|---|
| `AUTH_ENABLED` | `true` | keep **`true`** — `false` opens every route |
| `JWT_SECRET` | unset (warned) | 32+ random bytes; sessions won't survive restarts without it |
| `ADMIN_PASSWORD` / `ADMIN_EMAIL` | `admin` / `admin` | change before first start — the first boot seeds this account |
| `INTERNAL_SERVICE_KEY` | unset | set + `INTERNAL_SERVICE_KEY_REQUIRED=true` to gate `/api/ingest` |
| `POSTGRES_DSN` | `dewie:dewie@localhost` | real credentials; the default is dev-only |
| `CORS_ORIGINS` | dewie.ai + localhost | restrict to your domains |

## Security properties worth knowing

- **Session auth enforces `activation_status`** server-side: pending/rejected accounts get 403 on protected APIs.
- **Route-scope rules** apply to both API keys and session cookies (admin routes require admin scope; benchmark routes are admin-only).
- **`answers_questions`** (internal ranking signal) is structurally excluded from all API and MCP outputs.
- **Log redaction**: secrets patterns are scrubbed by `api/logging_config.redact_sensitive` before logging.
- **Known footgun**: a third-party dependency (`magika`, via `markitdown`) calls `load_dotenv()` at import time and will load any `.env` it finds above its install path into the process environment. Don't keep production secrets in stray `.env` files on shared machines.
