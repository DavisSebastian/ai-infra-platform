"""Token bucket in Redis, made atomic with a Lua script.

Why Lua: the check is read -> compute -> write. Done as separate Redis commands, two requests can both read
"1 permit left" before either writes, and both are let in (see tests/test_race.py for the demonstration).
Redis runs a whole script as ONE uninterruptible command, so no one can slip in between the read and the write.

The arithmetic is the same as the pure `limiter.take`, kept in sync by tests that compare both.
Time comes from Redis (`TIME`), not from the gateway, so replicas with drifting clocks still agree on "now".
"""

import math
from dataclasses import dataclass

import redis.asyncio as redis

# KEYS[1] = bucket key.  ARGV = capacity, refill_rate (permits/s), cost, ttl (seconds)
# Returns {allowed (0/1), retry_after seconds, permits left}, the last two as strings (Lua would truncate floats).
TOKEN_BUCKET_LUA = """
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000

local data = redis.call('HMGET', KEYS[1], 'permits', 'updated_at')
local permits = tonumber(data[1])
local updated_at = tonumber(data[2])
if permits == nil then
  permits = capacity
  updated_at = now
end

local elapsed = math.max(0, now - updated_at)
permits = math.min(capacity, permits + elapsed * rate)

local allowed = 0
local retry_after = 0
if permits >= cost then
  permits = permits - cost
  allowed = 1
elseif rate > 0 then
  retry_after = (cost - permits) / rate
end

-- permits and updated_at are stored together, always (the rule from the pure version).
redis.call('HSET', KEYS[1], 'permits', permits, 'updated_at', now)
redis.call('EXPIRE', KEYS[1], ttl)
return {allowed, tostring(retry_after), tostring(permits)}
"""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after: float  # seconds until `cost` permits will be available (0 when allowed)
    permits_left: float


class RedisTokenBucket:
    def __init__(self, client: redis.Redis, capacity: float, refill_rate: float, prefix: str = "rl:"):
        self._client = client
        self._capacity = capacity
        self._rate = refill_rate
        self._prefix = prefix
        # Idle clients' keys expire on their own: twice the time to refill from empty, at least a minute.
        self._ttl = max(60, math.ceil(2 * capacity / refill_rate)) if refill_rate > 0 else 3600
        # register_script sends the script once (EVALSHA afterwards) and re-sends it if Redis forgot it.
        self._script = client.register_script(TOKEN_BUCKET_LUA)

    async def take(self, client_id: str, cost: float = 1.0) -> Decision:
        allowed, retry_after, permits = await self._script(
            keys=[self._prefix + client_id], args=[self._capacity, self._rate, cost, self._ttl]
        )
        return Decision(bool(int(allowed)), float(retry_after), float(permits))
