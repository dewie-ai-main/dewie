# Deployment

## Dev / evaluation: SQLite

No external dependencies. Data lives in a single file.

```bash
pip install dewie
dewie setup          # wizard; choose SQLite when prompted
dewie serve
```

Limitations: in-process queue and cache (no Redis), no concurrent workers,
no pgvector (falls back to BM25 + cosine). Fine for a personal instance or
evaluation; not for production.

---

## Production: Docker Compose

The included `docker-compose.yml` runs `postgres` (with pgvector) and `app` as
separate services; the app uses its in-process queue, so Redis is not required.

```bash
cp .env.example .env
# Edit .env:
#   ADMIN_EMAIL, ADMIN_PASSWORD
#   CHAT_SERVER_AQ / CHAT_MODEL_AQ (or configure servers: in dewie.yml)
#   SMTP_* if you want password reset email
docker compose up -d
docker compose logs -f app
```

On first boot the app runs all migrations, creates the admin user, and starts
the enrichment workers. Check health:

```bash
curl http://localhost:10946/health
```

### Volumes

| Mount | Purpose |
|-------|---------|
| `./data` | Flat-file body store and `instance_id.txt` |
| `postgres_data` volume | Postgres data |

---

## Postgres: migrations

Dewie uses Alembic. Migrations run automatically on startup but you can run
them manually:

```bash
alembic upgrade head         # apply all pending (POSTGRES_DSN must be set)
alembic upgrade <revision>   # migrate to a specific revision
```

Migration chain: `000000000000` (baseline) → `000000000001` → … → `000000000004` (`corpus_id`).

---

## Reverse proxy (nginx)

```nginx
server {
    listen 443 ssl;
    server_name dewie.example.com;

    ssl_certificate     /etc/ssl/certs/dewie.crt;
    ssl_certificate_key /etc/ssl/private/dewie.key;

    location / {
        proxy_pass http://localhost:10946;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;   # enrichment status checks can be slow
    }
}
```

Update `BASE_URL` and `CORS_ORIGINS` to match your domain:

```bash
BASE_URL=https://dewie.example.com
CORS_ORIGINS=["https://dewie.example.com"]
```

---

## Production checklist

**Before going live:**

- [ ] Change `ADMIN_EMAIL` and `ADMIN_PASSWORD` (default is `admin`/`admin`)
- [ ] Set `POSTGRES_PASSWORD` to something strong
- [ ] Set `ENCRYPTION_MASTER_KEY` — generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
- [ ] Set `INTERNAL_SERVICE_KEY_REQUIRED=true` and `INTERNAL_SERVICE_KEY`
- [ ] Set `BASE_URL` to your public URL (used in password reset links)
- [ ] Set `CORS_ORIGINS` to your frontend origin
- [ ] Configure `SMTP_*` for password reset email
- [ ] Set `LOCAL_AUTH_ENABLED=false` (should already be default)
- [ ] Put a TLS-terminating reverse proxy in front

**Optional hardening:**

- [ ] Set `RATE_LIMIT_RPM` lower than the default (60) for public-facing instances
- [ ] Set `REDIS_URL` (in-process cache is lost on restart)
- [ ] Set `API_WORKERS` to match your CPU count
- [ ] Set `ENRICHMENT_BATCH_SIZE=1` for local LLMs (serial enrichment, no timeout races)
- [ ] Enable `AUDIT_LOG_ENABLED=true` (default) and review the `audit_events` table

---

## Scaling

Dewie is a single-process app by default. To scale:

- **Horizontal**: run multiple `app` instances behind a load balancer. They share
  Postgres and Redis. Enrichment workers will claim work atomically — no double
  processing.
- **Worker-only mode**: set `ENABLE_API=false` on dedicated worker nodes and
  `ENABLE_ENRICHMENT=false` on API-only nodes. Both read from the same Postgres.
- **Redis**: required for multi-node deployments. Set `REDIS_URL` and
  `LLM_QUEUE_BACKEND=redis`.

---

## Backup

Dewie's durable state lives in:

1. **Postgres** — documents, enrichment metadata, edges, users, API keys
2. **`data/` directory** — raw body text (flat files, keyed by doc UUID)

Redis is ephemeral by default (browse sessions, result cache). No backup needed
unless you use it for the queue backend.

```bash
# Postgres dump
pg_dump -Fc dewie > dewie_$(date +%Y%m%d).dump

# Data dir
tar czf data_$(date +%Y%m%d).tar.gz data/
```
