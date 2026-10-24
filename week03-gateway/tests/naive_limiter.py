"""The WRONG way, kept only to demonstrate the race. Never imported by the app.

It reuses the correct pure `take()`, so the arithmetic is right. The only bug is that GET and SET are two
separate Redis commands, with an `await` (a gap where other requests run) between them.
"""

import time

import redis.asyncio as redis
from app.limiter import Bucket, take


class NaiveRedisTokenBucket:
    def __init__(self, client: redis.Redis, capacity: float, refill_rate: float, prefix: str = "naive:"):
        self._client = client
        self._capacity = capacity
        self._rate = refill_rate
        self._prefix = prefix

    async def take(self, client_id: str) -> bool:
        key = self._prefix + client_id
        data = await self._client.hmget(key, "permits", "updated_at")  # 1. read
        bucket = Bucket(float(data[0]), float(data[1])) if data[0] is not None else None
        allowed, new = take(bucket, time.time(), self._capacity, self._rate)  # 2. compute (in Python)
        await self._client.hset(key, mapping={"permits": new.permits, "updated_at": new.updated_at})  # 3. write
        return allowed
