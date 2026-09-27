# Week 1 — Async Serving Foundations

A small FastAPI app that shows, with measurements, why LLM servers are built on async I/O.

| Endpoint | Style | What it demonstrates |
|---|---|---|
| `GET /users/{id}?delay=0.5` | `async def` + `aiosqlite` + `await asyncio.sleep` | The correct, non-blocking way |
| `GET /threadpool/users/{id}?delay=0.5` | plain `def` + `sqlite3` + `time.sleep` | The "Java/Tomcat" model: a bounded thread pool |
| `GET /blocking/users/{id}?delay=0.5` | `async def` + `time.sleep` | The classic bug: blocking the event loop |
| `GET /stream?delay=0.05` | SSE via `StreamingResponse` | Token-by-token streaming, like an LLM |

`delay` simulates a slow downstream call, such as another service or a model server.

## Run it

```bash
uv sync --group week01                                   # from repo root
cd week01-async-serving
uv run uvicorn app.main:app --loop uvloop --port 8000

curl localhost:8000/users/1
curl -N "localhost:8000/stream?delay=0.1"                # -N = don't buffer, see tokens arrive live

uv run python scripts/loadtest.py "/users/1?delay=0.5" -n 1000 -c 100
uv run python scripts/loadtest.py "/stream?delay=0.02" -n 500 -c 500 --stream
```

## Results (M-series Mac, single uvicorn worker, uvloop)

| Endpoint (`delay=0.5s`) | Concurrency | Throughput | p50 latency | p99 latency |
|---|---|---|---|---|
| async `/users` | 100 | **197 req/s** | 505 ms | 522 ms |
| async `/users` | 500 | **943 req/s** | 508 ms | 590 ms |
| thread pool `/threadpool/users` | 100 | 79 req/s | 1,050 ms | 1,544 ms |
| thread pool `/threadpool/users` | 500 | 79 req/s | 6,065 ms | 6,608 ms |
| blocking `/blocking/users` | 10 | **2 req/s** | 5,053 ms | 5,054 ms |

No delay, async `/users/1`: ~14,000 req/s, p99 23 ms.
Streaming, 500 concurrent streams of 64 tokens each: first byte at ~65 ms, and every stream finished in ~1.4 s.

### How to read these numbers

Every result follows from one formula: **throughput ≈ (how many requests can wait at once) ÷ (how long each one waits)**.

- **Async:** each waiting request costs only a small Python object, not a thread, so the number that can wait is basically unlimited. 100 ÷ 0.5 s = 200 req/s; 500 ÷ 0.5 s = 1,000 req/s. Latency stays flat at about 0.5 s.
- **Thread pool:** FastAPI runs plain `def` handlers on a pool of **40 threads**. Only 40 requests can wait at once: 40 ÷ 0.5 s = **80 req/s**, whatever the load. Extra requests queue up, so latency rises with load (6 s at c=500). A Java service with a fixed Tomcat pool fails the same way.
- **Blocking:** `time.sleep` inside `async def` holds the event loop's only thread. Just 1 request can wait at once: 1 ÷ 0.5 s = **2 req/s**. Unrelated requests stall too, including health checks. This is the worst case, and it looks harmless in code review.

## What's non-blocking, and why

**The event loop** is one thread running a to-do list. When a handler hits `await` (a DB query, a network call, a sleep), it tells the loop "I'm waiting, go do something else." The loop switches to another request, then comes back once the result is ready. So one thread can juggle thousands of in-flight requests, because most of their time is spent *waiting*, not computing.

| Code | Non-blocking? | Why |
|---|---|---|
| `await asyncio.sleep(0.5)` | ✅ | Gives control back to the loop while waiting |
| `await cursor.fetchone()` (aiosqlite) | ✅ | The loop keeps serving others until the row comes back |
| `StreamingResponse` + `async` generator | ✅ | Each `yield` sends a chunk; each `await` between tokens frees the loop |
| `time.sleep(0.5)` inside `async def` | ❌ | Never yields, so the whole server freezes |
| `sqlite3` / `requests` inside `async def` | ❌ | Same problem: sync I/O blocks the loop |
| plain `def` handler | ⚠️ | Safe (runs in the thread pool) but capped at 40 concurrent |

**Rule of thumb:** inside `async def`, anything slow must be `await`ed. If a library has no async version, use a plain `def` handler, or wrap the call in `await asyncio.to_thread(fn)`.

**uvloop** is a drop-in replacement for Python's built-in event loop, written on top of libuv (the C library Node.js uses). It makes the loop's own overhead smaller. It doesn't change the model; it just runs the same model faster.

**About aiosqlite:** SQLite is a local file with no network protocol, so aiosqlite runs queries on a background thread and gives you an `await`-able interface. `asyncpg` (Postgres) is async all the way down, because it talks to the database over a socket. From the handler's point of view the two behave the same: the loop is never blocked.

## Streaming (SSE)

Server-Sent Events is plain HTTP with a long-lived response. Each event is a `data: ...` line followed by a blank line:

```
data: {"index": 0, "token": "Large "}

data: {"index": 1, "token": "language "}

data: [DONE]
```

This is the same format OpenAI-compatible LLM APIs use, vLLM's included. Why it matters:

- **Time to first token (TTFT)** is what users actually feel. Here, 500 concurrent streams get their first byte in ~65 ms even though each full answer takes 1.4 s.
- **Client disconnects:** when the client hangs up, Starlette cancels the generator (`asyncio.CancelledError`). The app logs `stream cancelled by client after 4/64 tokens`. With a real model, this is where you stop generation so the GPU isn't wasted on an answer nobody will read.
- The `Cache-Control: no-cache` and `X-Accel-Buffering: no` headers stop proxies from buffering the stream into one big response.

## Lesson learned: benchmark your benchmark tool

The first version of `loadtest.py` used `httpx`. It reported only **66 req/s** for the async endpoint, with multi-second tail latencies, even against an endpoint that does nothing. Apache Bench (`ab`) showed the server doing 21,000 req/s. The load *client* was the bottleneck. After switching to `aiohttp`, the numbers matched `ab`. Before trusting any benchmark, check the load generator against a known-trivial endpoint.
