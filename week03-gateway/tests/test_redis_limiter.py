"""Behaviour of the Lua token bucket against the real Redis (docker compose up -d)."""

import asyncio
import uuid

import pytest
import redis.asyncio as redis
from app.redis_limiter import RedisTokenBucket


@pytest.fixture
async def client(real_redis_url):
    c = redis.from_url(real_redis_url)
    yield c
    await c.aclose()


def fresh_id() -> str:
    return uuid.uuid4().hex


async def test_burst_then_reject_with_a_retry_after(client):
    limiter = RedisTokenBucket(client, capacity=3, refill_rate=10)  # one permit every 0.1 s
    cid = fresh_id()
    results = [await limiter.take(cid) for _ in range(4)]
    assert [d.allowed for d in results] == [True, True, True, False]
    # The 4th was rejected with ~0 permits left, so the next permit is ~0.1 s away: that is the Retry-After.
    assert 0.03 < results[3].retry_after <= 0.1


async def test_refills_over_real_time(client):
    limiter = RedisTokenBucket(client, capacity=1, refill_rate=10)
    cid = fresh_id()
    assert (await limiter.take(cid)).allowed
    assert not (await limiter.take(cid)).allowed
    await asyncio.sleep(0.2)  # two refill periods; the cap keeps it at 1 permit
    assert (await limiter.take(cid)).allowed
    assert not (await limiter.take(cid)).allowed


async def test_clients_have_separate_buckets(client):
    limiter = RedisTokenBucket(client, capacity=1, refill_rate=0.001)
    a, b = fresh_id(), fresh_id()
    assert (await limiter.take(a)).allowed
    assert not (await limiter.take(a)).allowed
    assert (await limiter.take(b)).allowed  # b is unaffected by a


async def test_idle_buckets_expire_so_redis_does_not_fill_up(client):
    limiter = RedisTokenBucket(client, capacity=5, refill_rate=1)
    cid = fresh_id()
    await limiter.take(cid)
    ttl = await client.ttl("rl:" + cid)
    assert 0 < ttl <= 60, f"bucket key must have a TTL, got {ttl}"


async def test_cost_greater_than_one(client):
    limiter = RedisTokenBucket(client, capacity=5, refill_rate=0.001)
    cid = fresh_id()
    assert (await limiter.take(cid, cost=3)).allowed
    d = await limiter.take(cid, cost=3)  # only 2 left
    assert not d.allowed
    assert d.permits_left == pytest.approx(2, abs=0.01)  # a rejection takes nothing
