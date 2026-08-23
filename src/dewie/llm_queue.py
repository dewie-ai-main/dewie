# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
llm_queue.py — Background LLM job queue for enrichment and batch work.

WHAT THIS IS:
  A rate-controlled background queue for non-latency-sensitive LLM work:
  enrichment, chunk embedding, batch benchmarks. Not for user-facing queries.

HOW IT WORKS:
  1. Caller submits a job → gets back a job_id immediately
  2. Dispatcher drains queue at configured rate, fires LLM calls
  3. Results written to Redis with TTL, keyed by job_id
  4. Caller polls job_id for completion (or fire-and-forget for enrichment)

JOB TYPES:
  "enrich"    — enrich a document (passes doc_id, result written to DB)
  "llm"       — raw LLM completion (result stored in Redis for caller to read)
  "benchmark" — like "llm" but tagged with a run_id for batch collection

USAGE — submit a job:
    from dewie.llm_queue import submit_job, poll_job, JOB_TTL

    job_id = await submit_job({
        "type": "llm",
        "messages": [...],
        "model": "gpt-4o-2024-11-20",
        "max_tokens": 1000,
        "run_id": "benchmark-xyz",   # optional grouping tag
    })
    # Fire and forget, or poll:
    result = await poll_job(job_id, timeout=60)

USAGE — batch (benchmark):
    job_ids = []
    for q in queries:
        jid = await submit_job({"type": "llm", "messages": [...], "run_id": run_id})
        job_ids.append(jid)
    results = await poll_batch(job_ids, timeout=120)

DISPATCHER:
    python scripts/run_llm_dispatcher.py
    (or docker-compose service: llm-dispatcher)

CONFIG (env vars):
    LLM_QUEUE_KEY       Redis list key (default: dewie:llm:queue)
    LLM_QUEUE_RATE      Max requests per second (default: 5)
    LLM_QUEUE_WORKERS   Max concurrent LLM calls (default: 4)
    LLM_JOB_TTL         Result TTL seconds (default: 300)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

import redis.asyncio as aioredis

log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
QUEUE_KEY = os.environ.get("LLM_QUEUE_KEY", "dewie:llm:queue")
RESULT_PREFIX = "dewie:llm:result:"
RATE_PER_SEC = float(os.environ.get("LLM_QUEUE_RATE", "5"))
WORKERS = int(os.environ.get("LLM_QUEUE_WORKERS", "4"))
JOB_TTL = int(os.environ.get("LLM_JOB_TTL", "300"))
POLL_INTERVAL = 0.5  # seconds between result polls


def _queue_backend() -> str:
    """Queue backend selector: redis (default) or memory."""
    return os.environ.get("LLM_QUEUE_BACKEND", "redis").strip().lower()


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


class InProcessLLMQueue:
    """In-memory async queue for single-process deployments."""

    def __init__(self) -> None:
        self._results: dict[str, dict[str, Any]] = {}
        self._pending_jobs: set[str] = set()
        self._result_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(WORKERS)
        self._min_interval = 1.0 / RATE_PER_SEC
        self._last_fire = 0.0
        self._rate_lock = asyncio.Lock()

    async def submit(self, job_data: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        job = dict(job_data)
        job["job_id"] = job_id
        job["submitted_at"] = time.time()
        async with self._result_lock:
            self._pending_jobs.add(job_id)
        asyncio.create_task(self._execute(job))
        return job_id

    async def poll(self, job_id: str, timeout: float = 60.0) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            async with self._result_lock:
                result = self._results.pop(job_id, None)
            if result is not None:
                return result
            await asyncio.sleep(POLL_INTERVAL)
        return None

    async def poll_many(
        self, job_ids: list[str], timeout: float = 120.0
    ) -> dict[str, dict[str, Any] | None]:
        results: dict[str, dict[str, Any] | None] = {jid: None for jid in job_ids}
        pending = set(job_ids)
        deadline = time.monotonic() + timeout
        while pending and time.monotonic() < deadline:
            done: set[str] = set()
            async with self._result_lock:
                for jid in pending:
                    hit = self._results.pop(jid, None)
                    if hit is not None:
                        results[jid] = hit
                        done.add(jid)
            pending -= done
            if pending:
                await asyncio.sleep(POLL_INTERVAL)
        return results

    async def depth(self) -> int:
        async with self._result_lock:
            return len(self._pending_jobs)

    async def _execute(self, job: dict[str, Any]) -> None:
        job_id = job["job_id"]
        job_type = job.get("type", "llm")
        async with self._semaphore:
            try:
                async with self._rate_lock:
                    now = time.monotonic()
                    wait = self._min_interval - (now - self._last_fire)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    self._last_fire = time.monotonic()

                if job_type == "enrich":
                    raise NotImplementedError(
                        "Job type 'enrich' is not supported by InProcessLLMQueue. "
                        "Use type='llm' and handle the result in the caller."
                    )
                result = await _run_llm_job(job)
            except Exception as exc:
                result = {"error": str(exc)}
            finally:
                async with self._result_lock:
                    self._pending_jobs.discard(job_id)
                    self._results[job_id] = result


_IN_PROCESS_QUEUE = InProcessLLMQueue()


async def _run_llm_job(job: dict[str, Any]) -> dict[str, Any]:
    """Execute a single LLM job and return a normalised result payload."""
    from dewie.model_adapter import ModelClient

    model = job.get("model") or os.environ.get("LLM_MODEL") or ""
    async with ModelClient(model=model) as llm:
        resp = await llm.complete(
            messages=job["messages"],
            tools=job.get("tools"),
            max_tokens=job.get("max_tokens", 1000),
            temperature=job.get("temperature", 0.0),
        )
    return {
        "content": resp.content,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in (resp.tool_calls or [])
        ],
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "finish_reason": resp.finish_reason,
        "run_id": job.get("run_id"),
    }


# ── Caller API ─────────────────────────────────────────────────────────────────


async def submit_job(job_data: dict[str, Any], redis_url: str | None = None) -> str:
    """
    Push a job onto the queue. Returns job_id immediately.
    job_data must include 'type' ('llm' or 'enrich') and 'messages' or 'doc_id'.
    """
    if _queue_backend() == "memory":
        return await _IN_PROCESS_QUEUE.submit(job_data)

    r = aioredis.from_url(redis_url or _redis_url(), decode_responses=True)
    try:
        job_id = str(uuid.uuid4())
        job_data["job_id"] = job_id
        job_data["submitted_at"] = time.time()
        await r.lpush(QUEUE_KEY, json.dumps(job_data))
        return job_id
    finally:
        await r.aclose()


async def poll_job(
    job_id: str,
    timeout: float = 60.0,
    redis_url: str | None = None,
) -> dict[str, Any] | None:
    """
    Poll for a job result. Returns the result dict or None on timeout.
    Deletes the result key after reading.
    """
    if _queue_backend() == "memory":
        return await _IN_PROCESS_QUEUE.poll(job_id, timeout=timeout)

    r = aioredis.from_url(redis_url or _redis_url(), decode_responses=True)
    try:
        deadline = time.monotonic() + timeout
        result_key = RESULT_PREFIX + job_id
        while time.monotonic() < deadline:
            raw = await r.get(result_key)
            if raw:
                await r.delete(result_key)
                return json.loads(raw)
            await asyncio.sleep(POLL_INTERVAL)
        return None
    finally:
        await r.aclose()


async def poll_batch(
    job_ids: list[str],
    timeout: float = 120.0,
    redis_url: str | None = None,
) -> dict[str, dict[str, Any] | None]:
    """
    Poll for multiple jobs. Returns {job_id: result} dict.
    Jobs that don't complete within timeout get None.
    """
    if _queue_backend() == "memory":
        return await _IN_PROCESS_QUEUE.poll_many(job_ids, timeout=timeout)

    r = aioredis.from_url(redis_url or _redis_url(), decode_responses=True)
    try:
        results: dict[str, dict | None] = {jid: None for jid in job_ids}
        pending = set(job_ids)
        deadline = time.monotonic() + timeout

        while pending and time.monotonic() < deadline:
            done = set()
            for job_id in list(pending):
                raw = await r.get(RESULT_PREFIX + job_id)
                if raw:
                    await r.delete(RESULT_PREFIX + job_id)
                    results[job_id] = json.loads(raw)
                    done.add(job_id)
            pending -= done
            if pending:
                await asyncio.sleep(POLL_INTERVAL)

        return results
    finally:
        await r.aclose()


async def queue_depth(redis_url: str | None = None) -> int:
    """Return the current number of jobs waiting in the queue."""
    if _queue_backend() == "memory":
        return await _IN_PROCESS_QUEUE.depth()

    r = aioredis.from_url(redis_url or _redis_url(), decode_responses=True)
    try:
        return await r.llen(QUEUE_KEY)
    finally:
        await r.aclose()


# ── Dispatcher ─────────────────────────────────────────────────────────────────


class LLMDispatcher:
    """
    Drains the LLM queue at a controlled rate. Run as a single background process.
    All enrichment, chunk embedding, and batch LLM work flows through here.
    User-facing queries (agent endpoint) bypass the queue and call the LLM directly.
    """

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or _redis_url()
        self._r: aioredis.Redis | None = None
        self._semaphore = asyncio.Semaphore(WORKERS)
        self._min_interval = 1.0 / RATE_PER_SEC
        self._last_fire = 0.0
        self._processed = 0
        self._errors = 0
        # Load token at startup
        self._reload_token()

    async def _redis(self) -> aioredis.Redis:
        if self._r is None:
            self._r = aioredis.from_url(self.redis_url, decode_responses=True)
        return self._r

    async def _execute_job(self, job: dict[str, Any]) -> None:
        job_id = job["job_id"]
        job_type = job.get("type", "llm")
        result_key = RESULT_PREFIX + job_id
        t0 = time.time()

        log.debug(
            "llm_dispatcher job started",
            extra={"job_id": job_id[:8], "job_type": job_type},
        )

        async with self._semaphore:
            try:
                if job_type == "enrich":
                    result = await self._run_enrichment(job)
                else:
                    result = await self._run_llm(job)

                r = await self._redis()
                await r.setex(result_key, JOB_TTL, json.dumps(result))
                self._processed += 1
                elapsed_ms = round((time.time() - t0) * 1000)
                log.info(
                    "llm_dispatcher job done",
                    extra={
                        "job_id": job_id[:8],
                        "job_type": job_type,
                        "output_tokens": result.get("output_tokens", 0),
                        "elapsed_ms": elapsed_ms,
                    },
                )

            except Exception as e:
                elapsed_ms = round((time.time() - t0) * 1000)
                log.error(
                    "llm_dispatcher job failed",
                    extra={
                        "job_id": job_id[:8],
                        "job_type": job_type,
                        "error": str(e),
                        "elapsed_ms": elapsed_ms,
                    },
                )
                self._errors += 1
                r = await self._redis()
                await r.setex(result_key, JOB_TTL, json.dumps({"error": str(e)}))

    def _reload_token(self) -> bool:
        """
        Verify the optional bearer-token file has a fresh token. Set
        LLM_TOKEN_FILE to point at a JSON token file that an external process
        (e.g. a host cron) refreshes; if expired, wait up to 15s for the
        refresh. Returns True if the token is valid, False if unconfigured.
        """
        token_file = os.environ.get("LLM_TOKEN_FILE", "")
        if not token_file:
            return False

        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                import json as _j

                with open(token_file) as f:
                    data = _j.loads(f.read())
                token = data.get("token") or data.get("access_token") or ""
                if not token:
                    time.sleep(2)
                    continue
                exp = data.get("expiresAt") or data.get("expires_at")
                if exp:
                    exp_sec = exp / 1000 if exp > 1e10 else exp
                    if exp_sec < time.time() + 30:
                        log.warning("Token still stale, waiting for host cron...")
                        time.sleep(2)
                        continue
                log.info(
                    "Token valid (expires in %.1f min)", (exp_sec - time.time()) / 60 if exp else 99
                )
                return True
            except Exception as e:
                log.warning("Token check error: %s", e)
                time.sleep(2)

        log.error("Token still expired after 15s wait — host cron may be broken")
        return False

    async def _run_llm(self, job: dict[str, Any]) -> dict[str, Any]:
        import httpx

        from dewie.model_adapter import ModelClient

        model = job.get("model") or os.environ.get("LLM_MODEL") or ""

        async def _attempt() -> dict[str, Any]:
            async with ModelClient(model=model) as llm:
                resp = await llm.complete(
                    messages=job["messages"],
                    tools=job.get("tools"),
                    max_tokens=job.get("max_tokens", 1000),
                    temperature=job.get("temperature", 0.0),
                )
            return {
                "content": resp.content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in (resp.tool_calls or [])
                ],
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "finish_reason": resp.finish_reason,
                "run_id": job.get("run_id"),
            }

        try:
            return await _attempt()
        except httpx.HTTPStatusError as e:
            body = e.response.text[:500]
            log.error("HTTP %d body: %s", e.response.status_code, body)
            if e.response.status_code == 401:
                log.warning("401 — running token refresh and retrying")
                if self._reload_token():
                    return await _attempt()
                raise RuntimeError("Token refresh failed — check node/refresh script") from e
            raise

    async def _run_enrichment(self, job: dict[str, Any]) -> dict[str, Any]:
        """
        'enrich' job type is not used — the enrichment worker submits 'llm' jobs
        and handles parsing + DB writes itself. Raise clearly so callers know.
        """
        raise NotImplementedError(
            "Job type 'enrich' is not supported by LLMDispatcher. "
            "Use type='llm' and handle the result in your enrichment worker."
        )

    async def run(self) -> None:
        r = await self._redis()
        log.info(
            "LLM dispatcher started — rate=%.1f/s workers=%d queue_key=%s",
            RATE_PER_SEC,
            WORKERS,
            QUEUE_KEY,
        )
        start = time.monotonic()

        while True:
            try:
                now = time.monotonic()
                wait = self._min_interval - (now - self._last_fire)
                if wait > 0:
                    await asyncio.sleep(wait)

                raw = await r.brpop(QUEUE_KEY, timeout=2)
                if raw is None:
                    continue

                self._last_fire = time.monotonic()
                _, job_json = raw
                job = json.loads(job_json)
                asyncio.create_task(self._execute_job(job))

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Dispatcher loop error: %s", e)
                await asyncio.sleep(1)

        elapsed = time.monotonic() - start
        log.info(
            "Dispatcher stopped — %d processed, %d errors, %.0fs uptime",
            self._processed,
            self._errors,
            elapsed,
        )
        if self._r:
            await self._r.aclose()


# ── Entry point ────────────────────────────────────────────────────────────────


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [llm-dispatcher] %(message)s",
    )
    dispatcher = LLMDispatcher()
    from contextlib import suppress

    with suppress(KeyboardInterrupt):
        await dispatcher.run()


if __name__ == "__main__":
    asyncio.run(_main())
