"""Week 3: the gateway in front of the model server.

POST /v1/chat/completions   rate limit (429) -> admission (503) -> forwarded to UPSTREAM_URL, chunks relayed live
GET  /health

Run:  UPSTREAM_URL=http://localhost:9000 uv run uvicorn app.main:app --port 8080   (from week03-gateway/)
"""

import hashlib
import json
import logging
import math
import os
from contextlib import asynccontextmanager

import aiohttp
import anyio
import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from app.admission import Admission, ClientGone, Rejected
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

# Backpressure (Idea 6). Defaults fit the Week 2 server: one generation at a time, ~3.7 s per answer on the GPU,
# so a queue of 2 keeps the worst wait under ~10 s. queue size ~= acceptable wait / time per request.
MAX_IN_FLIGHT = int(os.getenv("MAX_IN_FLIGHT", "1"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "2"))
QUEUE_TIMEOUT = float(os.getenv("QUEUE_TIMEOUT", "10"))
OVERLOAD_RETRY_AFTER = int(os.getenv("OVERLOAD_RETRY_AFTER", "5"))  # seconds, sent with every 503


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
    app.state.admission = Admission(MAX_IN_FLIGHT, MAX_QUEUE, QUEUE_TIMEOUT)
    app.state.counts = {"overloaded": 0, "queue_timeout": 0, "client_gone": 0, "upstream_error": 0}
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
        "in_flight": request.app.state.admission.in_flight,
        "queued": request.app.state.admission.queued,
        "rejected": request.app.state.counts,
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
    return error(429, "rate_limited", "Too many requests. Slow down.", retry_after)


def error(status: int, kind: str, message: str, retry_after: int | None = None) -> Response:
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    body = json.dumps({"error": {"type": kind, "message": message}})
    return Response(content=body, status_code=status, media_type="application/json", headers=headers)


class GuardedStreamingResponse(StreamingResponse):
    """A StreamingResponse that ALWAYS runs `cleanup` when it ends: normal finish, client gone, or never started.

    Why not a `finally` inside the generator: if the client leaves before Starlette starts iterating it, the
    generator body never runs, so its `finally` never runs either. With MAX_IN_FLIGHT=1 one leaked slot would
    block the gateway forever. Wrapping __call__ covers every path.
    """

    def __init__(self, content, cleanup, **kwargs):
        super().__init__(content, **kwargs)
        self._cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Shielded, as in Week 2: during cancellation a plain await here would be cancelled again at once.
            with anyio.CancelScope(shield=True):
                await self._cleanup()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # 1. Rate limit (per client), BEFORE anything expensive: rejecting is cheap.
    rejected = await rate_limit(request)
    if rejected is not None:
        return rejected

    # Forward the body as-is. The gateway doesn't need to understand it (yet): fewer places to go wrong.
    body = await request.body()

    # 2. Admission: wait for a turn at the upstream, or be turned away.
    admission: Admission = request.app.state.admission
    counts = request.app.state.counts

    async def wait_for_disconnect() -> None:
        # The body is fully read, so the next ASGI message can only be "http.disconnect".
        while (await request.receive())["type"] != "http.disconnect":
            pass

    try:
        slot = await admission.acquire(wait_for_disconnect)
    except Rejected as r:
        counts[r.reason] += 1
        return error(503, r.reason, "The model server is busy. Try again later.", OVERLOAD_RETRY_AFTER)
    except ClientGone:
        counts["client_gone"] += 1
        return Response(status_code=499)  # nobody is listening; nothing was sent upstream

    # 3. Forward. From here on, every path must release the slot exactly once.
    upstream: UpstreamClient = request.app.state.upstream
    try:
        resp = await upstream.open("/v1/chat/completions", body)
    except (TimeoutError, aiohttp.ClientError, httpx.HTTPError, OSError):
        slot.release()
        counts["upstream_error"] += 1
        return error(502, "upstream_unavailable", "Could not reach the model server.")
    except BaseException:
        slot.release()
        raise

    if resp.status != 200:
        # Upstream said no (or failed). Relay its answer unchanged; nothing to stream.
        try:
            data = await resp.read()
        finally:
            await resp.close()
            slot.release()
        return Response(content=data, status_code=resp.status, media_type=resp.content_type)

    async def relay():
        async for data in resp.chunks():
            yield data

    async def cleanup():
        # Normal end: connection returns to the pool. Client left mid-stream: connection is dropped (stops upstream).
        await resp.close()
        slot.release()

    # Streaming or not, the bytes are relayed as they arrive; the upstream's Content-Type says which it was.
    return GuardedStreamingResponse(
        relay(),
        cleanup,
        media_type=resp.content_type,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
