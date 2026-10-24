"""The rate limiter wired into the gateway: 429 + Retry-After, per-client buckets, fail open."""

import time

import app.main as gateway
import httpx
import pytest

BODY = {"messages": [{"role": "user", "content": "hi"}], "stream": False, "max_tokens": 2}


@pytest.fixture
def limited(monkeypatch, real_redis_url):
    # 3 requests of burst, essentially no refill during the test.
    monkeypatch.setattr(gateway, "RATE_LIMIT_CAPACITY", 3)
    monkeypatch.setattr(gateway, "RATE_LIMIT_REFILL_PER_SEC", 0.001)
    return real_redis_url


async def post(url: str, key: str | None = None) -> httpx.Response:
    headers = {"X-API-Key": key} if key else {}
    async with httpx.AsyncClient(timeout=10) as c:
        return await c.post(f"{url}/v1/chat/completions", json=BODY, headers=headers)


async def test_over_budget_gets_429_with_retry_after(start_gateway, limited):
    url = start_gateway(redis_url=limited)
    key = f"test-{time.time_ns()}"
    statuses = [(await post(url, key)).status_code for _ in range(4)]
    assert statuses == [200, 200, 200, 429]

    r = await post(url, key)
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) >= 1
    assert r.json()["error"]["type"] == "rate_limited"


async def test_each_api_key_has_its_own_bucket(start_gateway, limited):
    url = start_gateway(redis_url=limited)
    a, b = f"a-{time.time_ns()}", f"b-{time.time_ns()}"
    for _ in range(3):
        assert (await post(url, a)).status_code == 200
    assert (await post(url, a)).status_code == 429
    assert (await post(url, b)).status_code == 200  # b is untouched by a's usage


async def test_rate_limited_requests_never_reach_the_upstream(start_gateway, limited, fake_url):
    """A 429 must be decided at the gateway. Streaming requests are used because the fake counts those."""
    url = start_gateway(redis_url=limited)
    key = f"test-{time.time_ns()}"
    statuses = []
    async with httpx.AsyncClient(timeout=10) as c:
        for _ in range(6):
            r = await c.post(f"{url}/v1/chat/completions", json={**BODY, "stream": True}, headers={"X-API-Key": key})
            statuses.append(r.status_code)  # reading r.content drained the stream
        stats = (await c.get(f"{fake_url}/_stats")).json()
    assert statuses == [200, 200, 200, 429, 429, 429]
    assert stats["started"] == 3, f"only the 3 allowed requests may reach the model server: {stats}"


async def test_fails_open_when_redis_is_down_and_counts_it(start_gateway):
    url = start_gateway()  # Redis unreachable
    for _ in range(5):
        assert (await post(url, "k")).status_code == 200
    async with httpx.AsyncClient() as c:
        health = (await c.get(f"{url}/health")).json()
    assert health["rate_limit_fail_open"] == 5
    assert health["redis"] == "down"


async def test_can_be_switched_off(start_gateway, limited, monkeypatch):
    monkeypatch.setattr(gateway, "RATE_LIMIT_ENABLED", False)
    url = start_gateway(redis_url=limited)
    key = f"test-{time.time_ns()}"
    assert [(await post(url, key)).status_code for _ in range(6)] == [200] * 6
