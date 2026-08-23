# Dockerfile — Dewie application image
#
# Build:  docker build -t dewie .
# Run:    docker compose up   (see docker-compose.yml)
#
# Extras installed:
#   auth  — bcrypt + PyJWT (required for login / API keys)
#   docs  — PDF, Word, Excel, PowerPoint ingestion
# Optional extras not included by default (add to pip install line if needed):
#   nlp   — spaCy NER (~500MB)
#   media — YouTube / podcast transcription via whisper (~2GB)
#   local — local sentence-transformer embeddings (~500MB)

FROM python:3.12-slim AS base

# System deps: libpq for asyncpg, curl for healthcheck, build-essential for native exts
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Application source ────────────────────────────────────────────────────────
# The package build needs src/ and static/ present (pyproject force-includes
# static/ into the wheel), so source is copied before pip install.
COPY pyproject.toml README.md alembic.ini ./
COPY src/     ./src/
COPY static/  ./static/
COPY docker/  ./docker/
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[auth,docs,dev]"

# In-process embeddings for the zero-config default (EmbeddingGemma GGUF via
# llama.cpp). We install just llama-cpp-python (prebuilt CPU wheel — no compile)
# + huggingface_hub, NOT the full [local] extra: that extra also pulls
# sentence-transformers/torch (~2GB), which the GGUF path does not use. The
# ~300MB model itself is fetched on first enrichment into the HF cache.
RUN pip install --no-cache-dir huggingface_hub \
      "llama-cpp-python>=0.3.0" \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Extra pip packages baked in at build time, e.g. Whisper for podcast ingestion:
#   docker build --build-arg EXTRA_DEPS="faster-whisper" -t dewie/app:podcast .
ARG EXTRA_DEPS=""
RUN if [ -n "$EXTRA_DEPS" ]; then pip install --no-cache-dir $EXTRA_DEPS; fi

# Persistent data + model-cache directories (mount volumes here in production).
# Created before the chown below so named volumes inherit dewie-writable dirs.
RUN mkdir -p /app/data /app/.cache

# Non-root user
RUN useradd -m -u 1000 dewie && chown -R dewie:dewie /app
USER dewie

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["sh", "/app/docker/entrypoint.sh"]
