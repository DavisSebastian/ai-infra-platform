"""The race condition, shown and then fixed. Needs the real Redis (docker compose up -d).

capacity=10 and a refill so slow it is irrelevant, so exactly 10 requests may ever be allowed.
50 requests fire at the same moment. The correct answer is 10 allowed, 40 rejected.
"""

import asyncio
import uuid

import pytest
import redis.asyncio as redis
from app.redis_limiter import RedisTokenBucket
from naive_limiter import NaiveRedisTokenBucket

CAPACITY = 10
SLOW_REFILL = 0.0001  # permits per second: nothing refills during a test
CONCURRENT = 50


@pytest.fixture
async def client(real_redis_url):
    # Pool big enough that all 50 requests really run at once (a small pool would serialise them and hide the race).
    pool = redis.BlockingConnectionPool.from_url(real_redis_url, max_connections=CONCURRENT)
    c = redis.Redis(connection_pool=pool)
    yield c
    await c.aclose()


def fresh_id() -> str:
    return uuid.uuid4().hex  # a new bucket per test, so tests never see each other's state


async def test_naive_get_then_set_lets_too_many_through(client):
    limiter = NaiveRedisTokenBucket(client, CAPACITY, SLOW_REFILL)
    cid = fresh_id()
    results = await asyncio.gather(*[limiter.take(cid) for _ in range(CONCURRENT)])
    allowed = sum(results)
    print(f"\n  naive:  {allowed} of {CONCURRENT} allowed (limit is {CAPACITY})")
    assert allowed > CAPACITY, "the race did not show up; the demonstration needs real concurrency"


async def test_lua_limiter_allows_exactly_the_limit(client):
    limiter = RedisTokenBucket(client, CAPACITY, SLOW_REFILL)
    cid = fresh_id()
    results = await asyncio.gather(*[limiter.take(cid) for _ in range(CONCURRENT)])
    allowed = sum(d.allowed for d in results)
    print(f"\n  lua:    {allowed} of {CONCURRENT} allowed (limit is {CAPACITY})")
    assert allowed == CAPACITY


async def test_lua_limiter_is_exact_across_two_separate_connections_pools(real_redis_url):
    """Two gateway replicas = two clients with their own pools, sharing one Redis. The limit must still hold."""
    clients = [redis.from_url(real_redis_url, max_connections=CONCURRENT) for _ in range(2)]
    try:
        limiters = [RedisTokenBucket(c, CAPACITY, SLOW_REFILL) for c in clients]
        cid = fresh_id()
        results = await asyncio.gather(*[limiters[i % 2].take(cid) for i in range(CONCURRENT)])
        assert sum(d.allowed for d in results) == CAPACITY
    finally:
        for c in clients:
            await c.aclose()
