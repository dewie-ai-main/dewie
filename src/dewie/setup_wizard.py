# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
setup_wizard.py — Interactive first-run setup for Dewie.

Design rules:
- No hardcoded model name suggestions — the server tells us what's available
- Ask for address/credentials before probing — never probe a URL the user hasn't confirmed
- On probe failure: offer to fix the address, enter a model manually, or go back to provider selection
- Cloud providers: get the API key first, then probe — never offer model lists without a key
- Postgres: ask host/port/user/pass individually; offer to create a dewie-specific DB user
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

# ── Helpers ───────────────────────────────────────────────────────────────────


def _q():
    try:
        import questionary
        return questionary
    except ImportError:
        print("questionary is required. Run: pip install -e .", file=sys.stderr)
        sys.exit(1)


def _probe_models(base_url: str, api_key: str = "") -> list[str]:
    """Hit /v1/models. Returns [] on any failure — caller decides what to do."""
    try:
        import httpx  # noqa: PLC0415
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        with httpx.Client(timeout=4) as client:
            r = client.get(f"{url}/models", headers=headers)
            if r.status_code == 200:
                ids = [m.get("id", "") for m in r.json().get("data", [])]
                return [i for i in ids if i]
    except Exception:
        pass
    return []


def _section(title: str) -> None:
    q = _q()
    q.print(f"\n{'─' * 52}", style="bold")
    q.print(f"  {title}", style="bold")
    q.print(f"{'─' * 52}", style="bold")


def _note(text: str) -> None:
    _q().print(f"  ℹ  {text}", style="fg:cyan")


def _ok(text: str) -> None:
    _q().print(f"  ✓  {text}", style="fg:green")


def _warn(text: str) -> None:
    _q().print(f"  ⚠  {text}", style="fg:yellow")


def _exit_on_none(val: Any) -> Any:
    if val is None:
        sys.exit(0)
    return val


def _register_server(
    cfg: dict[str, Any],
    label: str,
    endpoint: str,
    api_format: str = "openai",
    api_key_env: str | None = None,
) -> None:
    """Upsert a server entry into cfg["servers"] (the dewie.yml `servers:` list)."""
    servers: list[dict[str, Any]] = [
        s for s in cfg.get("servers", []) if s.get("label") != label
    ]
    entry: dict[str, Any] = {"label": label, "api_format": api_format, "endpoint": endpoint}
    if api_key_env:
        entry["api_key_env"] = api_key_env
    servers.append(entry)
    cfg["servers"] = servers


# ── Provider definitions (labels only — no model name suggestions) ────────────


_CLOUD_PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "embed_capable": True,
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "base_url": "https://api.anthropic.com/v1",
        "key_env": "ANTHROPIC_API_KEY",
        "embed_capable": False,
    },
}


_LOCAL_PROVIDERS = {
    "ollama": {
        "label": "Ollama",
        "base_url": "http://localhost:11434",
    },
}


# ── Shared: pick a model from a probed server, with fallbacks ─────────────────


def _pick_model_from_server(
    base_url: str,
    api_key: str = "",
    purpose: str = "enrichment",
    embed_only: bool = False,
) -> tuple[str, str] | None:
    """
    Probe base_url, let the user pick a model.

    Returns (model_name, final_base_url) or None if user wants to go back.
    On probe failure: offer to fix URL, enter manually, or go back.
    """
    q = _q()

    while True:
        _note(f"Connecting to {base_url} …")
        models = _probe_models(base_url, api_key)

        if models:
            if embed_only:
                # Put embedding-looking models first, rest after
                embed_models = [m for m in models if any(
                    x in m.lower() for x in ("embed", "nomic", "e5", "gte", "bge", "minilm")
                )]
                other_models = [m for m in models if m not in embed_models]
                display = embed_models + other_models
            else:
                # Filter out obvious non-chat models for LLM use
                display = [m for m in models if not any(
                    x in m.lower() for x in ("embed", "whisper", "tts", "dall")
                )] or models

            _ok(f"Connected — {len(display)} model(s) available.")
            choices = [q.Choice(m, value=m) for m in display]
            choices.append(q.Choice("↩  Go back to provider selection", value="__back__"))

            model = _exit_on_none(q.select(
                f"  Select the model to use for {purpose}:",
                choices=choices,
            ).ask())

            if model == "__back__":
                return None

            return model, base_url

        else:
            _warn(f"Could not connect to {base_url}")
            action = _exit_on_none(q.select(
                "  What would you like to do?",
                choices=[
                    q.Choice("Fix the server address and try again", value="fix"),
                    q.Choice("Enter a model name manually (server is running but /models is unavailable)", value="manual"),
                    q.Choice("↩  Go back to provider selection", value="back"),
                ],
            ).ask())

            if action == "back":
                return None
            if action == "manual":
                model = _exit_on_none(q.text(f"  Model name for {purpose}:").ask())
                if not model:
                    return None
                return model, base_url
            if action == "fix":
                base_url = _exit_on_none(q.text("  Server URL:", default=base_url).ask())


# ── Step: LLM provider ────────────────────────────────────────────────────────


def _step_llm(cfg: dict[str, Any]) -> tuple[str, str | None, str | None]:
    """
    Returns (llm_choice, api_key_or_none, confirmed_base_url_or_none).
    Loops at top level if user picks 'go back' from model selection.
    """
    q = _q()

    while True:
        _section("Step 1 of 3 — LLM Provider")
        _note(
            "Dewie uses an LLM to extract metadata from every document you ingest:\n"
            "  summaries, topics, keywords, and a set of specific questions each\n"
            "  document answers. This 'AQ index' is what makes Dewie's search\n"
            "  dramatically more accurate than standard keyword or vector search.\n"
            "\n"
            "  You can use a cloud API or a local model running on your machine."
        )

        choices = []
        for k, v in _CLOUD_PROVIDERS.items():
            choices.append(q.Choice(f"{v['label']}  (cloud API — needs API key)", value=k))
        for k, v in _LOCAL_PROVIDERS.items():
            choices.append(q.Choice(f"{v['label']}  (local — free, runs on your machine)", value=k))
        choices.append(q.Choice(
            "Skip for now  (search works, quality is lower without enrichment)",
            value="skip",
        ))

        llm_choice = _exit_on_none(q.select(
            "  Which LLM provider?",
            choices=choices,
        ).ask())

        if llm_choice == "skip":
            cfg["enrichment_default_backend"] = "passthrough"
            cfg.pop("enrichment_backends", None)
            _warn("Enrichment disabled. Re-run `dewie setup` to enable it later.")
            return "skip", None, None

        # ── Local provider ────────────────────────────────────────────────────
        if llm_choice in _LOCAL_PROVIDERS:
            info = _LOCAL_PROVIDERS[llm_choice]

            # Ask for address first — never probe a URL the user hasn't confirmed
            base_url = _exit_on_none(q.text(
                f"  {info['label']} server URL:",
                default=info["default_base_url"],
            ).ask())

            result = _pick_model_from_server(base_url, purpose="enrichment")
            if result is None:
                continue  # back to provider selection

            model, confirmed_url = result

            _register_server(cfg, llm_choice, confirmed_url, api_format="openai")
            cfg["enrichment_default_backend"] = llm_choice
            cfg["enrichment_batch_size"] = 1
            cfg["enrichment_sleep_secs"] = 0
            cfg["enrichment_backends"] = [{
                "name": llm_choice,
                "type": "http",
                "server": llm_choice,
                "model": model,
            }]
            cfg["chat_server_aq"] = llm_choice
            cfg["chat_model_aq"] = model
            _ok(f"LLM: {info['label']} › {model}")
            return llm_choice, None, confirmed_url

        # ── Cloud provider ────────────────────────────────────────────────────
        info = _CLOUD_PROVIDERS[llm_choice]
        key_env = info["key_env"]
        existing_key = os.environ.get(key_env, "")

        if existing_key:
            _ok(f"{key_env} found in environment.")
            use_existing = _exit_on_none(q.confirm("  Use it?", default=True).ask())
            api_key = existing_key if use_existing else _exit_on_none(
                q.password(f"  Enter your {key_env}:").ask()
            )
        else:
            _note(f"You'll need a {info['label']} API key.")
            api_key = _exit_on_none(q.password(f"  {key_env}:").ask())

        if not api_key:
            continue

        os.environ[key_env] = api_key

        result = _pick_model_from_server(info["base_url"], api_key=api_key, purpose="enrichment")
        if result is None:
            continue  # back to provider selection

        model, confirmed_url = result

        api_format = "anthropic" if llm_choice == "anthropic" else "openai"
        _register_server(cfg, llm_choice, confirmed_url, api_format=api_format, api_key_env=key_env)
        cfg["enrichment_default_backend"] = llm_choice
        cfg["llm_model"] = model
        cfg["enrichment_backends"] = [{
            "name": llm_choice,
            "type": "http",
            "server": llm_choice,
            "model": model,
        }]
        cfg["chat_server_aq"] = llm_choice
        cfg["chat_model_aq"] = model
        _ok(f"LLM: {info['label']} › {model}")
        return llm_choice, api_key, confirmed_url


# ── Step: Embeddings ──────────────────────────────────────────────────────────


def _step_embeddings(
    cfg: dict[str, Any],
    llm_choice: str,
    llm_api_key: str | None,
    llm_base_url: str | None,
) -> None:
    q = _q()

    if llm_choice == "skip":
        _note("Skipping embeddings — enrichment is disabled.")
        cfg["embed_server"] = "openai"
        return

    while True:
        _section("Step 2 of 3 — Embedding Model")
        _note(
            "Dewie embeds every document to enable semantic (vector) search —\n"
            "  finding documents by meaning, not just keywords.\n"
            "\n"
            "  The embedding model runs once per document at ingest time.\n"
            "  Changing it later requires re-embedding your entire corpus.\n"
            "\n"
            "  Use the same server as your LLM if it supports embeddings — easiest option."
        )

        choices = []

        # Same-server option — always first when available
        if llm_choice in _LOCAL_PROVIDERS and llm_base_url:
            choices.append(q.Choice(
                f"Use the same {_LOCAL_PROVIDERS[llm_choice]['label']} server I just configured",
                value=("local_same", llm_base_url),
            ))

        if llm_choice in _CLOUD_PROVIDERS and _CLOUD_PROVIDERS[llm_choice]["embed_capable"]:
            choices.append(q.Choice(
                f"Use the same {_CLOUD_PROVIDERS[llm_choice]['label']} account (needs embed-capable model)",
                value=("cloud_same", _CLOUD_PROVIDERS[llm_choice]["base_url"]),
            ))
        elif llm_choice == "anthropic":
            _note("Anthropic doesn't provide embedding models. You'll need a different source.")

        # OpenAI as a standalone option — only if we have or can get a key
        has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
        if llm_choice != "openai":
            label = "OpenAI" + (" (API key already set)" if has_openai_key else " (requires API key)")
            choices.append(q.Choice(label, value=("openai_separate", "https://api.openai.com/v1")))

        # Local in-process option
        choices.append(q.Choice(
            "Run a model locally (in-process, no server needed — requires sentence-transformers)",
            value=("local_inprocess", None),
        ))

        # Another local/custom server
        choices.append(q.Choice(
            "A different server or provider (enter URL)…",
            value=("custom", None),
        ))

        result = _exit_on_none(q.select(
            "  Where should embeddings come from?",
            choices=choices,
        ).ask())

        provider_type, base_url = result

        # Local in-process: no server needed, just ask for model name
        if provider_type == "local_inprocess":
            _note("This downloads the model from HuggingFace the first time it runs. No server needed.")
            embed_model = _exit_on_none(q.text(
                "  HuggingFace model name for embeddings:",
            ).ask())
            cfg["embed_server"] = "local"
            cfg["embed_model"] = embed_model
            _ok(f"Embeddings: {embed_model} (local in-process)")
            return

        # Determine API key for probing
        probe_key = ""
        if provider_type == "cloud_same" and llm_choice in _CLOUD_PROVIDERS:
            probe_key = os.environ.get(_CLOUD_PROVIDERS[llm_choice]["key_env"], "")
        elif provider_type == "openai_separate":
            probe_key = os.environ.get("OPENAI_API_KEY", "")
            if not probe_key:
                probe_key = _exit_on_none(q.password("  OpenAI API key for embeddings:").ask())
                if probe_key:
                    os.environ["OPENAI_API_KEY"] = probe_key

        if provider_type == "custom":
            base_url = _exit_on_none(q.text(
                "  Embedding server URL (OpenAI-compatible):",
                default="http://localhost:1234/v1",
            ).ask())

        model_result = _pick_model_from_server(
            base_url, api_key=probe_key, purpose="embeddings", embed_only=True
        )

        if model_result is None:
            # User went back — loop to provider choice
            continue

        embed_model, confirmed_url = model_result

        # Write to cfg — reuse the LLM's server label when it's literally the
        # same server; otherwise register a new one.
        if provider_type == "local_same":
            cfg["embed_server"] = llm_choice  # already registered by _step_llm
        elif (provider_type == "cloud_same" and llm_choice == "openai") or provider_type == "openai_separate":
            cfg["embed_server"] = "openai"  # built-in
        else:
            _register_server(cfg, "embed-server", confirmed_url, api_format="openai")
            cfg["embed_server"] = "embed-server"

        cfg["embed_model"] = embed_model
        _ok(f"Embeddings: {embed_model} @ {confirmed_url}")
        return


# ── Step: Auth ────────────────────────────────────────────────────────────────


def _step_auth(cfg: dict[str, Any]) -> None:
    q = _q()
    _section("Step 3 of 3 — Admin Key")
    _note(
        "All Dewie API requests require an API key in the header:\n"
        "  X-API-Key: <your-key>\n"
        "\n"
        "  This is your admin key — it has full access. Keep it safe."
    )

    current = cfg.get("admin_key", "")
    if current and current != "dewie-admin-local":
        if not _exit_on_none(q.confirm(
            f"  Current admin key: {current[:16]}…  Change it?", default=False
        ).ask()):
            _ok("Keeping existing admin key.")

    # Always ensure internal_service_key exists (never regenerate if already set)
    if not cfg.get("internal_service_key"):
        cfg["internal_service_key"] = secrets.token_hex(32)
        _ok(f"Internal service key generated: {cfg['internal_service_key'][:16]}…")

    generated = f"dw_{secrets.token_urlsafe(24)}"
    choice = _exit_on_none(q.select(
        "  Admin key:",
        choices=[
            q.Choice("Generate a secure random key", value="generated"),
            q.Choice("Enter my own", value="custom"),
        ],
    ).ask())

    if choice == "generated":
        cfg["admin_key"] = generated
    else:
        key = _exit_on_none(q.password("  Admin key:").ask())
        if not key:
            sys.exit(0)
        cfg["admin_key"] = key

    _ok(f"Admin key: {cfg['admin_key']}")
    _note("Save this — you'll need it for every API request.")
    cfg["auth_enabled"] = True

    # JWT secret for signing session tokens
    if not cfg.get("jwt_secret"):
        jwt_secret = secrets.token_hex(32)
        cfg["jwt_secret"] = jwt_secret
        _ok(f"JWT secret: {jwt_secret[:16]}…")
        _note("This secret signs session tokens — changing it invalidates all active sessions.")

    # Internal service key for ingest/service-to-service authentication
    if not cfg.get("internal_service_key"):
        internal_key = secrets.token_hex(32)
        cfg["internal_service_key"] = internal_key
        _ok(f"Internal service key: {internal_key[:16]}…")
        _note("This key authenticates enrichment workers calling the ingest API.")


# ── Step: Storage ─────────────────────────────────────────────────────────────


def _step_storage(cfg: dict[str, Any]) -> str:
    q = _q()
    _section("Storage")
    _note(
        "Dewie stores all your documents, embeddings, and search indexes in a database.\n"
        "\n"
        "  SQLite  — stored in a single local file. Zero setup. Great for testing.\n"
        "  Postgres — required for production, multi-worker, or large corpora.\n"
        "             Needs pgvector extension installed."
    )

    current_db = cfg.get("postgres_dsn", "")  # Try new name first, fallback handled later
    if not current_db:
        current_db = cfg.get("dewie_db", "")  # Backward compat with old field name
    default_choice = "postgres" if (current_db and current_db.startswith("postgres")) else "sqlite"

    choice = _exit_on_none(q.select(
        "  Which database?",
        choices=[
            q.Choice("SQLite  (recommended to start)", value="sqlite"),
            q.Choice("PostgreSQL + pgvector", value="postgres"),
        ],
        default=default_choice,
    ).ask())

    if choice == "sqlite":
        default_path = (
            current_db.replace("sqlite+aiosqlite:///", "") if current_db
            else os.path.abspath("dewie.db")
        )
        path = _exit_on_none(q.text("  SQLite file path:", default=default_path).ask())
        db_url = f"sqlite+aiosqlite:///{os.path.abspath(path)}"
        _ok(f"Storage: {db_url}")
        return db_url

    # Postgres — ask individually
    _note("Enter your Postgres connection details.")
    host = _exit_on_none(q.text("  Host:", default="localhost").ask())
    port = _exit_on_none(q.text("  Port:", default="5432").ask())
    dbname = _exit_on_none(q.text("  Database name:", default="dewie").ask())

    _note(
        "You can use an existing superuser to connect, or create a dedicated\n"
        "  Dewie database user with minimal permissions.\n"
        "  Creating a dedicated user requires that your current Postgres user\n"
        "  has CREATE ROLE / CREATE DATABASE privileges."
    )

    create_user = _exit_on_none(q.confirm(
        "  Create a dedicated 'dewie' database user?", default=True
    ).ask())

    if create_user:
        _note(
            "We'll generate connection details for a new 'dewie' user.\n"
            "  You'll need to run the CREATE ROLE + GRANT commands as a Postgres superuser.\n"
            "  We'll print them at the end so you can run them yourself."
        )
        dewie_password = secrets.token_urlsafe(16)
        username = "dewie"
        password = dewie_password
        cfg["_pg_setup_commands"] = [
            f"CREATE ROLE dewie WITH LOGIN PASSWORD '{dewie_password}';",
            f"CREATE DATABASE {dbname} OWNER dewie;",
            f"GRANT ALL PRIVILEGES ON DATABASE {dbname} TO dewie;",
            f"-- Then connect to {dbname} and run:",
            "CREATE EXTENSION IF NOT EXISTS vector;",
        ]
    else:
        username = _exit_on_none(q.text("  Username:", default="dewie").ask())
        password = _exit_on_none(q.password("  Password:").ask())

    db_url = f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{dbname}"
    _ok(f"Storage: postgres @ {host}:{port}/{dbname} (user: {username})")
    return db_url


# ── Main entry point ──────────────────────────────────────────────────────────


def run_setup(config_dir: str, force: bool, compose_mode: bool) -> None:  # noqa: C901
    q = _q()
    config_path = Path(config_dir) / "dewie.yml"
    env_path = Path(config_dir) / ".env"

    q.print("")
    if compose_mode:
        q.print("  ╔══════════════════════════════════════╗", style="bold")
        q.print("  ║     Dewie — Docker Compose Setup      ║", style="bold")
        q.print("  ╚══════════════════════════════════════╝", style="bold")
        q.print("")
        q.print("  Postgres + pgvector is handled by docker-compose.")
        q.print("  Answer a few questions, then run: docker compose up")
    else:
        q.print("  ╔══════════════════════════════════════╗", style="bold")
        q.print("  ║          Dewie — First-Run Setup      ║", style="bold")
        q.print("  ╚══════════════════════════════════════╝", style="bold")
        q.print("")
        q.print("  Answer a few questions and you'll be running in minutes.")

    cfg: dict[str, Any] = {}
    if not compose_mode and config_path.exists() and not force:
        cfg = _load_yaml(config_path)
        _note(f"Existing config found at {config_path} — updating only what changes.")

    # Storage
    if compose_mode:
        _note("Storage: Postgres (docker-compose manages it)")
    else:
        db_url = _step_storage(cfg)

    # LLM
    llm_choice, llm_api_key, llm_base_url = _step_llm(cfg)

    # Embeddings
    _step_embeddings(cfg, llm_choice, llm_api_key, llm_base_url)

    # Auth
    _step_auth(cfg)

    # Write
    pg_setup = cfg.pop("_pg_setup_commands", None)
    cfg.setdefault("redis_url", "")
    if "internal_service_key" not in cfg:
        cfg["internal_service_key"] = secrets.token_hex(32)
    cfg["internal_service_key_required"] = True

    if compose_mode:
        _write_env(env_path, cfg, llm_api_key, llm_choice)
        _ok(f".env written to {env_path}")
    else:
        # Write to dewie.yml (not env vars - let dewie serve read it)
        cfg["postgres_dsn"] = db_url
        cfg.setdefault("api_host", "0.0.0.0")
        cfg.setdefault("api_port", 10946)
        cfg.setdefault("api_workers", 1)
        _save_yaml(config_path, cfg)
        _ok(f"Config written to {config_path}")
        q.print("\n  Initializing database…")
        _init_db_sync(db_url)

    # Done
    q.print("")
    q.print("  ╔══════════════════════════════════════╗", style="bold fg:green")
    q.print("  ║            🎉  Setup complete!        ║", style="bold fg:green")
    q.print("  ╚══════════════════════════════════════╝", style="bold fg:green")
    q.print("")
    if compose_mode:
        q.print("  Start Dewie:", style="bold")
        q.print("    docker compose up")
    else:
        q.print("  Start Dewie:", style="bold")
        q.print("    dewie serve")
    q.print("")
    q.print("  Ingest a document:")
    q.print("    dewie ingest https://example.com/your-doc")
    q.print("")

    if llm_choice == "skip":
        q.print("  ⚠  Enrichment is disabled. Re-run `dewie setup` to add a provider.",
                style="fg:yellow")
        q.print("")

    if pg_setup:
        q.print("  ── Postgres setup commands ──────────────────────────────", style="bold")
        q.print("  Run these as a Postgres superuser before starting Dewie:")
        q.print("")
        for cmd in pg_setup:
            q.print(f"    {cmd}")
        q.print("")
        q.print("  Admin key shown above — save it before closing this window.", style="fg:yellow")


# ── Utilities ─────────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # noqa: PLC0415
        loaded = yaml.safe_load(path.read_text())
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml  # noqa: PLC0415
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _write_env(
    env_path: Path,
    cfg: dict[str, Any],
    llm_api_key: str | None,
    llm_choice: str,
) -> None:
    # Postgres connection is set directly in docker-compose.yml's `environment:`
    # block (POSTGRES_HOST/PORT/USER/PASSWORD/DB) — not read from .env, since
    # there's no `env_file:` directive wiring this file into the app container.
    lines = [
        "# Generated by `dewie setup --compose`",
        "REDIS_URL=",
        f"JWT_SECRET={cfg.get('jwt_secret', '')}",
        f"ADMIN_KEY={cfg.get('admin_key', '')}",
        "AUTH_ENABLED=true",
        "ENABLE_ENRICHMENT=true",
        "ENABLE_INGESTION=true",
        "ENABLE_POLLER=true",
        f"INTERNAL_SERVICE_KEY={cfg.get('internal_service_key', secrets.token_hex(32))}",
        "INTERNAL_SERVICE_KEY_REQUIRED=true",
    ]
    if llm_api_key and llm_choice in _CLOUD_PROVIDERS:
        key_env = _CLOUD_PROVIDERS[llm_choice]["key_env"]
        lines.append(f"{key_env}={llm_api_key}")
    servers = cfg.get("servers")
    if servers:
        lines.append(f"SERVERS_JSON={json.dumps(servers)}")
    backends = cfg.get("enrichment_backends")
    if backends:
        lines.append(f"ENRICHMENT_BACKENDS={json.dumps(backends)}")
        lines.append(f"ENRICHMENT_DEFAULT_BACKEND={cfg.get('enrichment_default_backend', 'passthrough')}")
    if cfg.get("chat_server_aq"):
        lines.append(f"CHAT_SERVER_AQ={cfg['chat_server_aq']}")
    if cfg.get("chat_model_aq"):
        lines.append(f"CHAT_MODEL_AQ={cfg['chat_model_aq']}")
    if cfg.get("embed_server"):
        lines.append(f"EMBED_SERVER={cfg['embed_server']}")
    if cfg.get("embed_model"):
        lines.append(f"EMBED_MODEL={cfg['embed_model']}")
    if cfg.get("internal_service_key"):
        lines.append(f"INTERNAL_SERVICE_KEY={cfg['internal_service_key']}")
    env_path.write_text("\n".join(lines) + "\n")


def _init_db_sync(db_url: str) -> None:
    import asyncio  # noqa: PLC0415

    import click  # noqa: PLC0415

    async def _run() -> None:
        os.environ["POSTGRES_DSN"] = db_url
        os.environ["DEWIE_DB"] = db_url
        from dewie.storage.postgres import PostgresClient  # noqa: PLC0415
        client = PostgresClient(dsn=db_url)
        await client.init_schema()
        await client.close()

    try:
        asyncio.run(_run())
        _ok("Database initialized.")
    except Exception as exc:
        raise click.ClickException(f"Database initialization failed: {exc}") from exc
