"""Token bucket rate limiter: the pure part (no Redis, no clock of its own).

A bucket holds up to `capacity` permits. Permits refill at `refill_rate` per second. Each request takes one.
Nothing runs in the background: the refill is computed lazily, from the time elapsed since the last call.

`now` is a parameter, not `time.time()`, so tests can say "pretend 2 seconds passed" instead of sleeping.
The Redis (Lua) version in step 4b does the same arithmetic atomically inside Redis.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Bucket:
    permits: float  # how many permits were in the bucket at `updated_at` (may be fractional)
    updated_at: float  # the `now` of the last call, in seconds


def take(
    bucket: Bucket | None,
    now: float,
    capacity: float,
    refill_rate: float,
    cost: float = 1.0,
) -> tuple[bool, Bucket]:
    """Try to take `cost` permits. Returns (allowed, the bucket to store for next time).

    bucket=None means "never seen this client": it starts FULL (a new client may burst right away).
    Does not modify its input; returns a new Bucket.

    The lazy refill, in order: start full if new; add elapsed * refill_rate (never negative elapsed, never above
    capacity); take `cost` if there is enough; ALWAYS return permits and timestamp together. Storing one
    without the other loses refill (new time, old permits) or counts it twice (new permits, old time).
    """
    if bucket is None:
        bucket = Bucket(permits=capacity, updated_at=now)

    elapsed = max(0.0, now - bucket.updated_at)
    permits = min(capacity, bucket.permits + elapsed * refill_rate)

    if permits >= cost:
        return True, Bucket(permits - cost, now)
    return False, Bucket(permits, now)
