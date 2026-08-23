"""Tests for dewie.llm_queue — submit_job, poll_job, queue_depth, LLMDispatcher."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_redis_mock():
    r = AsyncMock()
    r.aclose = AsyncMock()
    return r


# ── submit_job ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_job_returns_uuid_string():
    from dewie.llm_queue import submit_job

    r = _make_redis_mock()
    r.lpush = AsyncMock()
    with patch("dewie.llm_queue.aioredis.from_url", return_value=r):
        job_id = await submit_job({"type": "llm", "messages": []})
    assert isinstance(job_id, str)
    assert len(job_id) == 36  # UUID length


@pytest.mark.asyncio
async def test_submit_job_pushes_to_queue():
    from dewie.llm_queue import QUEUE_KEY, submit_job

    r = _make_redis_mock()
    r.lpush = AsyncMock()
    with patch("dewie.llm_queue.aioredis.from_url", return_value=r):
        await submit_job({"type": "llm"})
    r.lpush.assert_called_once()
    args = r.lpush.call_args[0]
    assert args[0] == QUEUE_KEY
    payload = json.loads(args[1])
    assert "job_id" in payload
    assert "submitted_at" in payload


@pytest.mark.asyncio
async def test_submit_job_closes_redis():
    from dewie.llm_queue import submit_job

    r = _make_redis_mock()
    r.lpush = AsyncMock()
    with patch("dewie.llm_queue.aioredis.from_url", return_value=r):
        await submit_job({})
    r.aclose.assert_called_once()


# ── poll_job ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_poll_job_returns_result_when_present():
    from dewie.llm_queue import poll_job

    r = _make_redis_mock()
    result_data = {"content": "hello", "output_tokens": 10}
    r.get = AsyncMock(return_value=json.dumps(result_data))
    r.delete = AsyncMock()
    with patch("dewie.llm_queue.aioredis.from_url", return_value=r):
        result = await poll_job("test-job-id", timeout=5)
    assert result == result_data
    r.delete.assert_called_once()


@pytest.mark.asyncio
async def test_poll_job_returns_none_on_timeout():
    from dewie.llm_queue import poll_job

    r = _make_redis_mock()
    r.get = AsyncMock(return_value=None)
    with (
        patch("dewie.llm_queue.aioredis.from_url", return_value=r),
        patch("dewie.llm_queue.POLL_INTERVAL", 0),
    ):
        result = await poll_job("missing-job", timeout=0.0)
    assert result is None


# ── queue_depth ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_queue_depth_returns_count():
    from dewie.llm_queue import queue_depth

    r = _make_redis_mock()
    r.llen = AsyncMock(return_value=7)
    with patch("dewie.llm_queue.aioredis.from_url", return_value=r):
        depth = await queue_depth()
    assert depth == 7


# ── LLMDispatcher ─────────────────────────────────────────────────────────────


def test_dispatcher_init():
    from dewie.llm_queue import LLMDispatcher

    d = LLMDispatcher(redis_url="redis://localhost:6379/0")
    assert d.redis_url == "redis://localhost:6379/0"
    assert d._processed == 0
    assert d._errors == 0


@pytest.mark.asyncio
async def test_dispatcher_redis_cached():
    from dewie.llm_queue import LLMDispatcher

    d = LLMDispatcher()
    mock_redis = AsyncMock()
    with patch("dewie.llm_queue.aioredis.from_url", return_value=mock_redis) as mock_from_url:
        r1 = await d._redis()
        r2 = await d._redis()
    assert r1 is r2
    mock_from_url.assert_called_once()


def test_dispatcher_reload_token_no_file():
    import os

    from dewie.llm_queue import LLMDispatcher

    d = LLMDispatcher()
    with patch.dict("os.environ", {}, clear=False):
        os.environ.pop("LLM_TOKEN_FILE", None)
        result = d._reload_token()
    assert result is False


def test_dispatcher_reload_token_with_file(tmp_path):
    import json as _json

    from dewie.llm_queue import LLMDispatcher

    token_file = tmp_path / "token.json"
    token_file.write_text(_json.dumps({"token": "ghu_abc123"}))
    d = LLMDispatcher()
    with patch.dict("os.environ", {"LLM_TOKEN_FILE": str(token_file)}, clear=False):
        result = d._reload_token()
        assert result is True


def test_dispatcher_reload_token_invalid_file(tmp_path):
    from dewie.llm_queue import LLMDispatcher

    token_file = tmp_path / "bad.json"
    token_file.write_text("not json")
    d = LLMDispatcher()

    # First two calls (deadline calc + loop check) see t=0, everything after
    # sees t=1000 so the loop exits. A finite side_effect list breaks here:
    # patching time.time patches the stdlib module globally, and logging's
    # timestamp formatting consumes extra calls.
    calls = {"n": 0}

    def _fake_time():
        calls["n"] += 1
        return 0 if calls["n"] <= 2 else 1000

    with (
        patch.dict("os.environ", {"LLM_TOKEN_FILE": str(token_file)}),
        patch("dewie.llm_queue.time.sleep"),
        patch("dewie.llm_queue.time.time", side_effect=_fake_time),
    ):
        result = d._reload_token()
    assert result is False


@pytest.mark.asyncio
async def test_dispatcher_run_enrichment_raises():
    from dewie.llm_queue import LLMDispatcher

    d = LLMDispatcher()
    with pytest.raises(NotImplementedError, match="not supported"):
        await d._run_enrichment({"job_id": "x", "type": "enrich"})


@pytest.mark.asyncio
async def test_dispatcher_execute_job_error_stored():
    from dewie.llm_queue import JOB_TTL, RESULT_PREFIX, LLMDispatcher

    d = LLMDispatcher()
    mock_redis = AsyncMock()
    mock_redis.setex = AsyncMock()
    d._r = mock_redis

    job = {"job_id": "test-123", "type": "llm", "messages": []}
    with patch.object(d, "_run_llm", side_effect=Exception("boom")):
        await d._execute_job(job)

    assert d._errors == 1
    mock_redis.setex.assert_called_once()
    key, ttl, payload_str = mock_redis.setex.call_args[0]
    assert key == RESULT_PREFIX + "test-123"
    assert ttl == JOB_TTL
    assert json.loads(payload_str)["error"] == "boom"


@pytest.mark.asyncio
async def test_dispatcher_execute_job_success_stored():
    """_execute_job success path: result stored in Redis."""
    from dewie.llm_queue import RESULT_PREFIX, LLMDispatcher

    d = LLMDispatcher()
    mock_redis = AsyncMock()
    mock_redis.setex = AsyncMock()
    d._r = mock_redis

    job = {"job_id": "test-456", "type": "llm", "messages": []}
    result = {"content": "hello", "output_tokens": 5}
    with patch.object(d, "_run_llm", return_value=result):
        await d._execute_job(job)

    assert d._processed == 1
    mock_redis.setex.assert_called_once()
    key, ttl, payload_str = mock_redis.setex.call_args[0]
    assert key == RESULT_PREFIX + "test-456"
    assert json.loads(payload_str)["content"] == "hello"


@pytest.mark.asyncio
async def test_dispatcher_run_llm_success():
    """_run_llm calls ModelClient.complete and returns structured result."""
    from dewie.llm_queue import LLMDispatcher
    from dewie.model_adapter import LLMResponse

    d = LLMDispatcher()
    mock_resp = LLMResponse(
        content="The answer is 42",
        tool_calls=[],
        finish_reason="stop",
        input_tokens=10,
        output_tokens=5,
        raw={},
    )
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=mock_resp)
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    with patch("dewie.model_adapter.ModelClient", return_value=mock_cm):
        result = await d._run_llm(
            {
                "job_id": "job-1",
                "type": "llm",
                "messages": [{"role": "user", "content": "What is 6x7?"}],
            }
        )

    assert result["content"] == "The answer is 42"
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 5


@pytest.mark.asyncio
async def test_poll_batch_returns_results():
    """poll_batch returns a dict of job_id → result."""
    from dewie.llm_queue import poll_batch

    r = _make_redis_mock()
    result_data = json.dumps({"content": "done", "output_tokens": 3})
    r.get = AsyncMock(return_value=result_data)
    r.delete = AsyncMock()
    with patch("dewie.llm_queue.aioredis.from_url", return_value=r):
        results = await poll_batch(["job-a", "job-b"], timeout=0.1)
    assert results["job-a"] == {"content": "done", "output_tokens": 3}
    assert results["job-b"] == {"content": "done", "output_tokens": 3}


@pytest.mark.asyncio
async def test_poll_batch_timeout_returns_none():
    """poll_batch returns None for jobs not completed within timeout."""
    from dewie.llm_queue import poll_batch

    r = _make_redis_mock()
    r.get = AsyncMock(return_value=None)  # Always miss
    with patch("dewie.llm_queue.aioredis.from_url", return_value=r):
        results = await poll_batch(["job-x"], timeout=0.01)
    assert results["job-x"] is None


# ── _run_llm: 401 retry with token refresh ────────────────────────────────────
# Note: ModelClient is imported locally inside _run_llm, so patch the source module.


@pytest.mark.asyncio
async def test_run_llm_401_retries_after_token_refresh():
    """_run_llm: 401 triggers token reload, then retries successfully."""
    import httpx

    from dewie.llm_queue import LLMDispatcher
    from dewie.model_adapter import LLMResponse

    d = LLMDispatcher()
    call_count = [0]
    success_resp = LLMResponse(
        content="retry worked",
        tool_calls=[],
        finish_reason="stop",
        input_tokens=5,
        output_tokens=3,
        raw={},
    )
    mock_client = AsyncMock()

    async def fake_complete(*a, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            mock_http_resp = MagicMock()
            mock_http_resp.status_code = 401
            mock_http_resp.text = "Unauthorized"
            raise httpx.HTTPStatusError("401", request=MagicMock(), response=mock_http_resp)
        return success_resp

    mock_client.complete = fake_complete
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    # Patch the source module since _run_llm does `from dewie.model_adapter import ModelClient`
    with (
        patch("dewie.model_adapter.ModelClient", return_value=mock_cm),
        patch.object(d, "_reload_token", return_value=True),
    ):
        result = await d._run_llm(
            {
                "job_id": "job-401",
                "type": "llm",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )
    assert result["content"] == "retry worked"
    assert call_count[0] == 2


@pytest.mark.asyncio
async def test_run_llm_401_token_refresh_fails_raises():
    """_run_llm: 401 with failed token refresh raises RuntimeError."""
    import httpx

    from dewie.llm_queue import LLMDispatcher

    d = LLMDispatcher()
    mock_client = AsyncMock()
    mock_http_resp = MagicMock()
    mock_http_resp.status_code = 401
    mock_http_resp.text = "Unauthorized"
    mock_client.complete = AsyncMock(
        side_effect=httpx.HTTPStatusError("401", request=MagicMock(), response=mock_http_resp)
    )
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("dewie.model_adapter.ModelClient", return_value=mock_cm),
        patch.object(d, "_reload_token", return_value=False),
    ):
        with pytest.raises(RuntimeError, match="Token refresh failed"):
            await d._run_llm(
                {
                    "job_id": "job-401-fail",
                    "type": "llm",
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )


@pytest.mark.asyncio
async def test_run_llm_non_401_http_error_reraises():
    """_run_llm: non-401 HTTP errors are re-raised directly."""
    import httpx

    from dewie.llm_queue import LLMDispatcher

    d = LLMDispatcher()
    mock_client = AsyncMock()
    mock_http_resp = MagicMock()
    mock_http_resp.status_code = 503
    mock_http_resp.text = "Service Unavailable"
    mock_client.complete = AsyncMock(
        side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=mock_http_resp)
    )
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    with patch("dewie.model_adapter.ModelClient", return_value=mock_cm):
        with pytest.raises(httpx.HTTPStatusError):
            await d._run_llm(
                {
                    "job_id": "job-503",
                    "type": "llm",
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )
