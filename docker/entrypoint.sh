#!/bin/sh
# docker/entrypoint.sh — Dewie container startup
#
# On first run: generates missing secrets and seeds the admin user.
# Override admin credentials via ADMIN_USERNAME / ADMIN_PASSWORD env vars.

set -e

# ── Auto-generate missing secrets ────────────────────────────────────────────

generated_any=0

if [ -z "$JWT_SECRET" ]; then
    JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
    export JWT_SECRET
    echo "⚠  JWT_SECRET not set — generated: ${JWT_SECRET}"
    echo "   Add to .env: JWT_SECRET=${JWT_SECRET}"
    generated_any=1
fi

if [ -z "$ADMIN_KEY" ]; then
    ADMIN_KEY=$(python -c "import secrets; print('dw_' + secrets.token_urlsafe(24))")
    export ADMIN_KEY
    echo "🔑 ADMIN_KEY not set — generated: ${ADMIN_KEY}"
    echo "   Add to .env: ADMIN_KEY=${ADMIN_KEY}"
    generated_any=1
fi

if [ -z "$INTERNAL_SERVICE_KEY" ]; then
    INTERNAL_SERVICE_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
    export INTERNAL_SERVICE_KEY
    generated_any=1
fi

if [ "$generated_any" = "1" ]; then
    echo ""
    echo "── Add the above to your .env so they persist across restarts. ──"
    echo "── Tip: run  bash generate-env.sh  to generate a complete .env.  ──"
    echo ""
fi

# ── Seed admin user (first boot only) ────────────────────────────────────────
#
# Skipped on subsequent starts. Delete /app/data/.admin_seeded to re-run.
# Override credentials via ADMIN_USERNAME / ADMIN_PASSWORD env vars.

ADMIN_SEED_FLAG="/app/data/.admin_seeded"

if [ ! -f "$ADMIN_SEED_FLAG" ]; then
    ADMIN_USERNAME="${ADMIN_USERNAME:-admin}"
    ADMIN_PASSWORD="${ADMIN_PASSWORD:-admin123}"

    echo "👤 First boot — seeding admin user '${ADMIN_USERNAME}'..."
    dewie admin-user \
        --username "${ADMIN_USERNAME}" \
        --password "${ADMIN_PASSWORD}" \
        ${POSTGRES_DSN:+--db-url "${POSTGRES_DSN}"} \
        && touch "$ADMIN_SEED_FLAG" \
        && echo "   ✅ Admin user seeded." \
        || echo "   ⚠  Admin seed failed — run: docker compose exec app dewie admin-user"
fi

# ── Start server ──────────────────────────────────────────────────────────────

exec python -m uvicorn dewie.main:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-10946}" \
    --workers "${API_WORKERS:-1}"
