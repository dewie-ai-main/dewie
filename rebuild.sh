#!/bin/bash
# Rebuild and restart Dewie containers
set -e

cd "$(dirname "$0")"

echo "==> Pulling latest images..."
docker compose pull

echo "==> Building app container..."
docker compose build app

echo "==> Restarting containers..."
docker compose up -d

echo "==> Waiting for postgres to be healthy..."
until docker compose exec -T postgres pg_isready -U dewie > /dev/null 2>&1; do
    echo "    waiting..."
    sleep 2
done

echo "==> Waiting for app to be healthy..."
until curl -sf http://localhost:10946/health > /dev/null 2>&1; do
    echo "    waiting..."
    sleep 2
done

echo ""
echo "==> Done. Dewie is up at http://localhost:10946"
docker compose ps