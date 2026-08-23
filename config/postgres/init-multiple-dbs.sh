#!/bin/bash
# init-multiple-dbs.sh — Create multiple postgres databases on first startup
# Referenced by the postgres service for multi-database setups.
# Creates: dewie (app)
set -e

function create_db() {
  local db=$1
  echo "Creating database: $db"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    SELECT 'CREATE DATABASE $db' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db')\gexec
    GRANT ALL PRIVILEGES ON DATABASE $db TO $POSTGRES_USER;
EOSQL
}

# POSTGRES_MULTIPLE_DATABASES is a comma-separated list, e.g. "dewie"
if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
  IFS=',' read -ra DATABASES <<< "$POSTGRES_MULTIPLE_DATABASES"
  for db in "${DATABASES[@]}"; do
    create_db "$(echo $db | xargs)"
  done
  echo "Multiple databases initialized."
fi
