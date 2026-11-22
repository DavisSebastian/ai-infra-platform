"""Week 3: the gateway in front of the model server.

POST /v1/chat/completions   rate limit (429) -> clean prompt (C++) -> apply champion card -> admission (503)
                            -> forwarded, chunks relayed live
GET  /health
POST /admin/reload          Week 5: re-read the `champion` card from the MLflow registry (needs X-Admin-Token)
POST /admin/traffic         Week 6: set the Blue/Green split, e.g. {"green": 10} (needs X-Admin-Token)

Week 7: with DRIFT_LOG=1, every forwarded request is also recorded to a Redis stream for the drift check.

Run:  UPSTREAM_URL=http://localhost:9000 uv run uvicorn gateway.main:app --port 8080
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import secrets
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass

import aiohttp
import anyio
import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from gateway.admission import Admission, ClientGone, Rejected
from gateway.champion import CHAMPION, ChampionError, ServingCard, apply_card, resolve_champion
from gateway.redis_limiter import RedisTokenBucket
from gateway.request_log import Finished, RequestLog
from gateway.router import BLUE, GREEN, TrafficSplit, choose, split_from_env
from gateway.sanitize import clean_body
from gateway.upstream import UpstreamClient, make_client

log = logging.getLogger("uvicorn.error")

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://localhost:9000")  # the blue model server
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

# Prompt cleaning (Week 4, C++ via pybind11): strip invisible/control characters and excess whitespace before
# forwarding. PROMPT_MAX_CHARS caps each message's text (unset = no cap).
CLEAN_PROMPTS = os.getenv("CLEAN_PROMPTS", "1") == "1"
PROMPT_MAX_CHARS = int(os.environ["PROMPT_MAX_CHARS"]) if os.getenv("PROMPT_MAX_CHARS") else None

# Model registry (Week 5). Unset = registry off: requests are forwarded as they are, as in Weeks 3-4.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
REGISTRY_MODEL = os.getenv("REGISTRY_MODEL", "gateway-llm")
# A secret: POST /admin/reload changes what every user gets, so it needs this in X-Admin-Token. Unset = no reloads.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

# Blue/Green (Week 6). Blue is UPSTREAM_URL. GREEN_URL unset = one model server only, as in Weeks 3-5.
GREEN_URL = os.getenv("GREEN_URL")
TRAFFIC_GREEN = os.getenv("TRAFFIC_GREEN")  # starting percent of requests sent to green; unset = 0

# Request log for drift detection (Week 7). Off by default: records hold prompt and answer text.
DRIFT_LOG = os.getenv("DRIFT_LOG", "0") == "1"
DRIFT_STREAM = os.getenv("DRIFT_STREAM", "gateway:requests")
DRIFT_STREAM_MAXLEN = int(os.getenv("DRIFT_STREAM_MAXLEN", "10000"))  # newest N records kept
CAPTURE_LIMIT = 64 * 1024  # bytes of each answer kept for the log: bounds memory per request


@dataclass
class Side:
    """One model server and everything the gateway keeps for it.

    Each side has its OWN pooled client and its OWN admission limit: each server has its own capacity, so a full
    queue on blue must not turn away a request that is going to an idle green.
    """

    name: str
    url: str
    upstream: UpstreamClient
    admission: Admission
    card: ServingCard | None = None  # the settings this server's requests get (None = registry off)
    routed: int = 0  # requests choose() sent to this side

    @classmethod
    def create(cls, name: str, url: str, card: ServingCard | None) -> "Side":
        upstream = make_client(PROXY_CLIENT, url, UPSTREAM_MAX_CONNECTIONS, UPSTREAM_CONNECT_TIMEOUT)
        return cls(name, url, upstream, Admission(MAX_IN_FLIGHT, MAX_QUEUE, QUEUE_TIMEOUT), card)


def card_aliases() -> dict[str, str]:
    """Which registry alias each side's card comes from.

    One model server: blue serves `champion`, exactly as in Week 5. Blue/Green: each server follows its own alias
    (`blue`, `green`), moved by `release.py deploy N --to green`, and `champion` records what is officially released.
    """
    return {BLUE: BLUE, GREEN: GREEN} if GREEN_URL else {BLUE: CHAMPION}


def resolve_cards(client, aliases: dict[str, str]) -> dict[str, ServingCard]:
    """Every side's card, or ChampionError if any one fails: all or nothing, never half the sides updated."""
    return {side: resolve_champion(client, REGISTRY_MODEL, alias=alias) for side, alias in aliases.items()}


def describe(cards: dict[str, ServingCard]) -> str:
    """'v2' with one model server, 'blue v1, green v2' with two."""
    if len(cards) == 1:
        return f"v{next(iter(cards.values())).version}"
    return ", ".join(f"{side} v{card.version}" for side, card in cards.items())


def make_registry_client():
    # MLflow's client retries an unreachable server for about 4 minutes by default (measured: 247 s). At startup
    # that hangs the gateway; on reload it holds the admin call. Fail fast instead, unless set explicitly.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    from mlflow import MlflowClient  # imported only when the registry is on: it's a slow import

    return MlflowClient(MLFLOW_TRACKING_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Settings are checked FIRST, before any connection is opened: a typo in TRAFFIC_GREEN stops startup.
    app.state.split = split_from_env(TRAFFIC_GREEN)
    if app.state.split.green and not GREEN_URL:
        raise ValueError(f"TRAFFIC_GREEN={app.state.split.green} sends traffic to green, but GREEN_URL is not set")

    # Then the cards. With the registry on and no usable card for a side there is nothing correct to serve there,
    # so startup fails (fail fast: the container restarts and tries again).
    app.state.registry = None
    app.state.reload = {"ok": 0, "failed": 0, "last_error": None}
    aliases = card_aliases()
    cards: dict[str, ServingCard | None] = dict.fromkeys(aliases)
    if MLFLOW_TRACKING_URI:
        app.state.registry = make_registry_client()
        # The MLflow client is blocking (requests), so it runs on a worker thread: never block the event loop.
        cards = await asyncio.to_thread(resolve_cards, app.state.registry, aliases)
        log.info("serving %s %s", REGISTRY_MODEL, describe(cards))

    # One pooled client per model server, created at startup and closed at shutdown (Week 2's pattern).
    app.state.sides = {BLUE: Side.create(BLUE, UPSTREAM_URL, cards[BLUE])}
    if GREEN_URL:
        app.state.sides[GREEN] = Side.create(GREEN, GREEN_URL, cards[GREEN])
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
    app.state.counts = {"overloaded": 0, "queue_timeout": 0, "client_gone": 0, "upstream_error": 0}
    app.state.cleaning = {"requests_changed": 0, "chars_removed": 0}
    # The request log shares the rate limiter's Redis client and its timeouts. Its writer is a background task.
    app.state.request_log = RequestLog(app.state.redis, DRIFT_STREAM, DRIFT_STREAM_MAXLEN) if DRIFT_LOG else None
    writer = asyncio.create_task(app.state.request_log.run()) if app.state.request_log else None
    yield
    if writer:
        writer.cancel()
        with suppress(asyncio.CancelledError):
            await writer
    await app.state.redis.aclose()
    for side in app.state.sides.values():
        await side.upstream.aclose()


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
    # A failed reload also degrades: the gateway is serving an older card than the registry says it should.
    reload = request.app.state.reload
    status = "ok" if redis_status == "ok" and reload["last_error"] is None else "degraded"
    sides = request.app.state.sides.values()
    return {
        "status": status,
        "redis": redis_status,
        "rate_limit_fail_open": request.app.state.rate_limit_fail_open,
        "in_flight": sum(s.admission.in_flight for s in sides),  # totals across both sides
        "queued": sum(s.admission.queued for s in sides),
        "rejected": request.app.state.counts,
        "cleaning": request.app.state.cleaning,
        "request_log": request.app.state.request_log.counts if request.app.state.request_log else "off",
        "upstream": UPSTREAM_URL,
        "client": PROXY_CLIENT,
        "registry": registry_status(request.app),
        "traffic": traffic_status(request.app),
    }


def traffic_status(app: FastAPI) -> dict:
    """Per side: its share and how busy it is. in_flight 0 on a side with weight 0 = drained, safe to stop."""
    split: TrafficSplit = app.state.split
    weights = {BLUE: split.blue, GREEN: split.green}
    return {
        name: {
            "url": side.url,
            "weight": weights[name],
            "in_flight": side.admission.in_flight,
            "queued": side.admission.queued,
            "routed": side.routed,
            "card": side.card.version if side.card else None,
        }
        for name, side in app.state.sides.items()
    }


def registry_status(app: FastAPI) -> dict | str:
    if app.state.registry is None:
        return "off"

    def fields(card: ServingCard) -> dict:
        return {"version": card.version, "model_id": card.model_id, "temperature": card.temperature,
                "max_tokens": card.max_tokens}  # fmt: skip

    sides: dict[str, Side] = app.state.sides
    if len(sides) == 1:  # one model server: the Week 5 shape, the champion's card
        return {"model": REGISTRY_MODEL, **fields(sides[BLUE].card), "reloads": app.state.reload}
    aliases = card_aliases()
    return {
        "model": REGISTRY_MODEL,
        "sides": {name: {"alias": aliases[name], **fields(side.card)} for name, side in sides.items()},
        "reloads": app.state.reload,
    }


def admin_denied(request: Request) -> Response | None:
    """A 403 unless the request carries the admin token. Admin calls change what every user gets."""
    token = request.headers.get("x-admin-token", "")
    # compare_digest takes the same time however many characters match, so the token can't be guessed by timing.
    if not ADMIN_TOKEN or not secrets.compare_digest(token.encode(), ADMIN_TOKEN.encode()):
        return error(403, "forbidden", "Admin calls need a valid X-Admin-Token (and ADMIN_TOKEN set on the gateway).")
    return None


@app.post("/admin/reload")
async def reload_champion(request: Request):
    """Re-read every side's card. On any failure all the current cards stay in place and the answer says why."""
    if denied := admin_denied(request):
        return denied
    if request.app.state.registry is None:
        return error(409, "registry_off", "The registry is off (MLFLOW_TRACKING_URI is not set).")

    sides: dict[str, Side] = request.app.state.sides
    old = {name: side.card for name, side in sides.items()}
    reload = request.app.state.reload
    try:
        new = await asyncio.to_thread(resolve_cards, request.app.state.registry, card_aliases())
    except ChampionError as e:
        reload["failed"] += 1
        reload["last_error"] = str(e)
        log.warning("reload failed, still serving %s: %s", describe(old), e)
        return error(503, "reload_failed", f"Still serving {describe(old)}. {e}")

    # The switch: one reference assignment per side, with no `await` in between, so no request can run in the
    # middle of it (asyncio switches tasks only at an await). Requests already running keep the card object they
    # started with; every request that starts after this loop gets the new ones.
    for name, side in sides.items():
        side.card = new[name]
    reload["ok"] += 1
    reload["last_error"] = None
    changed = new != old
    if changed:
        log.info("cards changed: %s -> %s", describe(old), describe(new))
    if len(sides) == 1:  # one model server: the Week 5 answer
        return {"version": new[BLUE].version, "previous": old[BLUE].version, "changed": changed}
    return {
        "sides": {name: {"version": new[name].version, "previous": old[name].version} for name in sides},
        "changed": changed,
    }


@app.post("/admin/traffic")
async def set_traffic(request: Request):
    """Set the Blue/Green split: {"green": 10} sends 10% of NEW requests to green. Requests already running keep
    the side they started on. {"green": 0} is the rollback: all new requests back to blue at once."""
    if denied := admin_denied(request):
        return denied
    try:
        green = json.loads(await request.body())["green"]
        new = TrafficSplit(green)
    except (ValueError, KeyError, TypeError) as e:  # not JSON, no "green", or TrafficSplit refused the value
        return error(400, "bad_split", f'Send {{"green": <whole number from 0 to 100>}}. {e}')
    if new.green and GREEN not in request.app.state.sides:
        return error(409, "green_off", "There is no green model server (GREEN_URL is not set).")

    old: TrafficSplit = request.app.state.split
    # The switch: one reference assignment, as in /admin/reload. Every request that starts after this line uses it.
    request.app.state.split = new
    if new != old:
        log.info("traffic: green %s%% -> %s%%", old.green, new.green)
    return {"green": new.green, "blue": new.blue, "previous": old.green, "changed": new != old}


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

    body = await request.body()

    # 2. Clean the prompt text (C++), BEFORE taking a slot: a big clean shouldn't hold the model's only slot.
    if CLEAN_PROMPTS:
        body, removed = await clean_body(body, PROMPT_MAX_CHARS)
        if removed:
            request.app.state.cleaning["requests_changed"] += 1
            request.app.state.cleaning["chars_removed"] += removed

    # 2b. Blue or green (Week 6): decided ONCE, here. The request stays on this side to the end, even if the split
    # changes while it streams, so moving traffic never cuts a request off halfway.
    side: Side = request.app.state.sides[choose(request.app.state.split, random.random())]
    side.routed += 1

    # 2c. Apply THAT side's card (Week 5), after cleaning: the card is trusted operator config, not user input.
    # Read side.card ONCE: if a reload swaps it during this request, this request still uses one whole card.
    card: ServingCard | None = side.card
    if card is not None:
        body = apply_card(body, card)

    # 3. Admission: wait for a turn at THIS side's model server, or be turned away.
    admission: Admission = side.admission
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

    # 4. Forward. From here on, every path must release the slot exactly once.
    upstream: UpstreamClient = side.upstream
    request_log: RequestLog | None = request.app.state.request_log
    opened = time.monotonic()
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
        if request_log:
            ms = 1000 * (time.monotonic() - opened)
            request_log.record(Finished(side.name, card and card.version, resp.status, body, b"", [], ms, True))
        return Response(content=data, status_code=resp.status, media_type=resp.content_type)

    # For the request log: a copy of the answer and the timings, taken as the bytes go past. The bytes themselves
    # are relayed untouched; nothing is parsed here (the log's background writer does that, after the response).
    captured = bytearray()
    arrivals: list[tuple[int, float]] = []  # per chunk: (bytes captured so far, ms since opened); the writer finds
    timing = {"completed": False}  # which chunk carried the first real text, so nothing is parsed here

    async def relay():
        async for data in resp.chunks():
            if request_log and len(captured) < CAPTURE_LIMIT:
                captured.extend(data[: CAPTURE_LIMIT - len(captured)])
                arrivals.append((len(captured), 1000 * (time.monotonic() - opened)))
            yield data
        timing["completed"] = True

    async def cleanup():
        # Normal end: connection returns to the pool. Client left mid-stream: connection is dropped (stops upstream).
        await resp.close()
        slot.release()
        if request_log:  # after the slot is released, and record() never waits, so logging can't hold anything up
            request_log.record(Finished(side.name, card and card.version, 200, body, bytes(captured), arrivals,
                                        1000 * (time.monotonic() - opened), timing["completed"]))  # fmt: skip

    # Streaming or not, the bytes are relayed as they arrive; the upstream's Content-Type says which it was.
    return GuardedStreamingResponse(
        relay(),
        cleanup,
        media_type=resp.content_type,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
