"""Week 3: the gateway. Step 1: a pass-through streaming proxy (no limits yet).

POST /v1/chat/completions   forwarded to UPSTREAM_URL, chunks relayed the moment they arrive
GET  /health

Run:  UPSTREAM_URL=http://localhost:9000 uv run uvicorn app.main:app --port 8080   (from week03-gateway/)
"""

import hashlib
import logging
import math
import os
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from app.redis_limiter import RedisTokenBucket
from app.upstream import UpstreamClient, make_client

log = logging.getLogger("uvicorn.error")

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://localhost:9000")
PROXY_CLIENT = os.getenv("PROXY_CLIENT", "aiohttp")  # "aiohttp" or "httpx"
UPSTREAM_MAX_CONNECTIONS = int(os.getenv("UPSTREAM_MAX_CONNECTIONS", "100"))
UPSTREAM_CONNECT_TIMEOUT = float(os.getenv("UPSTREAM_CONNECT_TIMEOUT", "5"))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
# Every wait on Redis has a deadline, so a slow or hung Redis can't freeze the gateway (Idea 5).
REDIS_POOL_TIMEOUT = float(os.getenv("REDIS_POOL_TIMEOUT", "0.5"))  # waiting for a free connection from the pool
REDIS_CONNECT_TIMEOUT = float(os.getenv("REDIS_CONNECT_TIMEOUT", "0.5"))  # opening a new connection
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "0.5"))  # waiting for a reply

# Per-client token bucket: a client may burst CAPACITY requests, then is held to REFILL_PER_SEC on average.
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1") == "1"
RATE_LIMIT_CAPACITY = float(os.getenv("RATE_LIMIT_CAPACITY", "10"))
RATE_LIMIT_REFILL_PER_SEC = float(os.getenv("RATE_LIMIT_REFILL_PER_SEC", "1"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One pooled client for the whole process, created at startup and closed at shutdown (Week 2's pattern).
    app.state.upstream = make_client(PROXY_CLIENT, UPSTREAM_URL, UPSTREAM_MAX_CONNECTIONS, UPSTREAM_CONNECT_TIMEOUT)
    # Redis: the pool is created lazily, so a Redis that is down at startup does not stop the gateway from booting.
    # BlockingConnectionPool makes request N+1 WAIT (up to REDIS_POOL_TIMEOUT) when all connections are busy,
    # instead of failing instantly like the plain ConnectionPool.
    pool = redis.BlockingConnectionPool.from_url(
        REDIS_URL,
        max_connections=REDIS_MAX_CONNECTIONS,
        timeout=REDIS_POOL_TIMEOUT,
        socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
        socket_timeout=REDIS_SOCKET_TIMEOUT,
    )
    app.state.redis = redis.Redis(connection_pool=pool)
    app.state.limiter = RedisTokenBucket(app.state.redis, RATE_LIMIT_CAPACITY, RATE_LIMIT_REFILL_PER_SEC)
    app.state.rate_limit_fail_open = 0  # how many requests were let through UNCHECKED because Redis failed
    yield
    await app.state.redis.aclose()
    await app.state.upstream.aclose()


app = FastAPI(title="Week 3 - Gateway", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    # Redis being down degrades the gateway (no rate limiting) but does not stop it from proxying,
    # so /health still answers 200 and says so. Step 4 decides what "degraded" means for traffic.
    try:
        await request.app.state.redis.ping()
        redis_status = "ok"
    except (redis.RedisError, OSError):
        redis_status = "down"
    status = "ok" if redis_status == "ok" else "degraded"
    return {
        "status": status,
        "redis": redis_status,
        "rate_limit_fail_open": request.app.state.rate_limit_fail_open,
        "upstream": UPSTREAM_URL,
        "client": PROXY_CLIENT,
    }


def client_id(request: Request) -> str:
    """Who is asking? The API key if there is one, else the caller's IP address.
    The key is hashed before it becomes a Redis key, so raw secrets are never stored in Redis."""
    key = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if key:
        return "key:" + hashlib.sha256(key.encode()).hexdigest()[:16]
    return "ip:" + (request.client.host if request.client else "unknown")


async def rate_limit(request: Request) -> Response | None:
    """Returns a 429 response if this client is over budget, else None (go ahead).

    Fail OPEN: if Redis errors or times out, let the request through and count it. The in-memory backpressure
    that arrives in step 5 still protects the model server, so a Redis outage costs fairness, not safety.
    """
    if not RATE_LIMIT_ENABLED:
        return None
    try:
        decision = await request.app.state.limiter.take(client_id(request))
    except (TimeoutError, redis.RedisError, OSError):
        request.app.state.rate_limit_fail_open += 1
        log.warning(
            "rate limiter unavailable, letting the request through (count=%s)", request.app.state.rate_limit_fail_open
        )
        return None
    if decision.allowed:
        return None
    retry_after = max(1, math.ceil(decision.retry_after))  # the header is whole seconds
    return Response(
        content='{"error": {"type": "rate_limited", "message": "Too many requests. Slow down."}}',
        status_code=429,
        media_type="application/json",
        headers={"Retry-After": str(retry_after)},
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # 1. Rate limit (per client), BEFORE touching the upstream or reading the body: rejecting is cheap.
    rejected = await rate_limit(request)
    if rejected is not None:
        return rejected

    upstream: UpstreamClient = request.app.state.upstream
    # Forward the body as-is. The gateway doesn't need to understand it (yet): fewer places to go wrong.
    body = await request.body()
    resp = await upstream.open("/v1/chat/completions", body)

    if resp.status != 200:
        # Upstream said no (or failed). Relay its answer unchanged; nothing to stream.
        try:
            data = await resp.read()
        finally:
            await resp.close()
        return Response(content=data, status_code=resp.status, media_type="application/json")

    async def relay():
        try:
            async for data in resp.chunks():
                yield data
        finally:
            # Normal end: connection returns to the pool. Client left mid-stream: connection is dropped.
            await resp.close()

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
