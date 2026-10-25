# Week 3 — The Gateway

Week 2's model server handles one generation at a time. With 16 users arriving at once, the last one waited **56 seconds** for a first word on an RTX 4090 (measured through RunPod's proxy; 41 s measured inside the pod). The server can't say "I'm full", so the line just grows.

This week puts a gateway in front of it. The gateway decides who gets in, bounds how long anyone waits, and turns everyone else away in milliseconds.

```
Client ──SSE──► [ Gateway :8080 ] ─────────────────────────────► [ Model server ]
                  1. rate limit per client     → 429   (Redis, token bucket, Lua)
                  2. admission: 1 in flight,   → 503   (in memory)
                     queue of 2, 10 s timeout
                  3. pooled upstream client, bytes relayed as they arrive
                  4. client leaves → upstream request closed → generation stops
```

## Run it

Everything runs on a laptop; no GPU needed.

**The whole stack in Docker** (gateway, Redis, fake model server):

```bash
cd week03-gateway
docker compose up -d --build --wait
curl -N localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
curl localhost:8080/health
docker compose down
```

**In front of the real Week 2 server** (Qwen 0.5B on the Mac):

```bash
uv sync --group dev --group week02 --group week03
cd week02-thin-slice && uv run uvicorn app.main:app --port 8001          # terminal 1
cd week03-gateway && docker compose up -d redis --wait
UPSTREAM_URL=http://127.0.0.1:8001 uv run uvicorn app.main:app --port 8080  # terminal 2
uv run python scripts/loadtest.py /v1/chat/completions --prompt "Write a long story about a robot." \
  --max-tokens 100 -n 16 -c 16
```

**Tests** (Redis must be running: `docker compose up -d redis`):

```bash
uv run pytest -o pythonpath=week03-gateway -o testpaths=week03-gateway/tests
```

### Settings (all environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `UPSTREAM_URL` | `http://localhost:9000` | The model server |
| `REDIS_URL` | `redis://localhost:6379` | Rate-limit state |
| `RATE_LIMIT_CAPACITY` / `RATE_LIMIT_REFILL_PER_SEC` | 10 / 1 | Burst size, then the long-run rate per client |
| `RATE_LIMIT_ENABLED` | `1` | `0` turns rate limiting off |
| `MAX_IN_FLIGHT` | 1 | Requests at the model server at once (the Week 2 server does one) |
| `MAX_QUEUE` | 2 | Requests allowed to wait. About acceptable wait ÷ time per answer |
| `QUEUE_TIMEOUT` | 10 | Seconds a request may wait before a 503 |
| `OVERLOAD_RETRY_AFTER` | 5 | `Retry-After` seconds sent with a 503 |
| `PROXY_CLIENT` | `aiohttp` | `httpx` is available for comparison |

The image sets `REDIS_URL=redis://redis:6379` and `UPSTREAM_URL=http://fake:9000`, the compose service names.

## Results

### Backpressure on a GPU: Llama 3.1 8B, RTX 4090

The Week 2 server running Llama 3.1 8B on a RunPod RTX 4090, bare vs behind the gateway, in the same pod and the same session. Load test run **inside the pod** (no proxy, no network time). All users arrive at once, 100 tokens each, the prompt from Week 2. Gateway: 1 in flight, queue of 2, rate limiting off. Commit `53b757a`. Raw numbers: [`results/llama8b-4090.jsonl`](results/llama8b-4090.jsonl). Steps: [`RUNBOOK.md`](RUNBOOK.md).

| Users | Bare: worst wait for first word | Gateway: served / rejected | Gateway: worst wait | Gateway: time to reject |
|---|---|---|---|---|
| 1 | 0.06 s | 1 / 0 | 0.07 s | n/a |
| 4 | 8.2 s | 3 / 1 | 5.5 s | 3 ms |
| 16 | **41 s** | 3 / 13 | **5.5 s** | 11 ms |
| 64 | **172 s** | 3 / 61 | **5.5 s** | 54 ms |

c=16 repeated three times: bare worst 41.1 / 41.0 / 40.8 s, gateway worst 5.47 / 5.50 / 5.48 s, 13 rejected each time in 11–13 ms. No `queue_timeout` rejections.

**The gateway's worst wait is 5.5 s whether 4, 16 or 64 users arrive.** The bare server's grows with the crowd.

**The numbers follow from one figure.** One answer takes 2.73 s (the single-user run). The last of 16 users on the bare server waits behind 15 answers: 15 × 2.73 ≈ 41 s. The last user the gateway accepts waits for two answers and then its own first word: 2 × 2.73 + 0.07 ≈ 5.5 s. Little's Law from Week 1 predicts both to within a fraction of a second. The prediction written before the run (~55 s bare, ~8 s gateway) used Week 2's 3.7 s per answer, which was measured through RunPod's proxy and included network time; the formula was right, the input was not.

**Capacity did not change:** at 16 users the bare server finished 16 answers in 43.6 s, 2.72 s each, the same as for one user.

Setup: VRAM 15,960 MiB of 24,564 MiB with the model loaded and idle. Model download and load: 43.2 s. Session cost: **$0.53**.

### Backpressure on a laptop: Qwen 0.5B, Mac

Week 2 server, Qwen2.5-0.5B on an M-series Mac. All users arrive at once (`n = c`), 100 tokens each, the same shape as Week 2's measurements. Gateway: 1 in flight, queue of 2. Rate limiting off, so only backpressure decides. Raw numbers: [`results/qwen05b-mac.jsonl`](results/qwen05b-mac.jsonl).

| Users | Bare: served | Bare: worst wait for first word | Gateway: served / rejected | Gateway: worst wait | Gateway: time to reject |
|---|---|---|---|---|---|
| 1 | 1 | 0.05 s | 1 / 0 | 0.04 s | n/a |
| 4 | 4 | 3.2 s | 3 / 1 | 1.7 s | 1 ms |
| 16 | 16 | **12.1 s** | 3 / 13 | **1.7 s** | 4 ms |
| 64 | 64 | **64.5 s** | 3 / 61 | **1.8 s** | 14 ms |

c=16 three times on each, because answer lengths vary between runs:

| c=16 run | Bare: median / worst wait | Gateway: median / worst wait | Gateway rejected |
|---|---|---|---|
| 1 | 6.9 s / 12.1 s | 0.2 s / 1.7 s | 13 in ≤ 4 ms |
| 2 | 9.7 s / 17.0 s | 1.5 s / 3.3 s | 13 in ≤ 3 ms |
| 3 | 8.1 s / 14.2 s | 1.5 s / 3.0 s | 13 in ≤ 3 ms |

**How to read this:**

- **The wait is bounded.** The bare server's worst wait grows with the crowd (12 s at 16 users, 64 s at 64). Through the gateway it stays at about two answers plus a first word, whatever the crowd: one running, two queued. Answer lengths vary more on the Mac, so the gateway's worst wait here ranges from 1.7 to 3.3 s; on the GPU it was steady at 5.5 s.
- **Everyone else hears "no" fast.** 13 of 16 got a 503 with `Retry-After` in a few milliseconds, instead of waiting up to 17 s.
- **Capacity did not change.** Total output was about 55 chunks/s in every run, bare or gateway. Backpressure adds no capacity. It stops the server from accepting work it can't do in a reasonable time. More capacity is continuous batching (vLLM, Month 3).
- **The trade is real.** The bare server eventually served all 16; the gateway served 3. A client that honours `Retry-After` comes back later. Whether that is better depends on the product: for a chat UI, a fast "busy" beats a 17 s blank screen.


### The rate limiter race

Limit 10, 50 requests at the same moment, real Redis, 5 runs each:

| Implementation | Allowed |
|---|---|
| Naive: read, compute in Python, write (two Redis commands) | **50 of 50**, every run |
| Lua script (one atomic Redis command) | **exactly 10**, every run, also from two separate clients |

All 50 got through because on one event loop every read happens before any write. That is the worst case; real traffic interleaves less, but the limit still leaks. The naive version reuses the correct pure function, so the only bug is the gap between the read and the write ([`tests/naive_limiter.py`](tests/naive_limiter.py)).

### Proxy overhead

Fake model server (20 chunks, 20 ms apart), 200 users, 1200 requests:

| | First word, median | First word, p95 | Chunks/s | Errors |
|---|---|---|---|---|
| Fake directly | 35 ms | 57 ms | 9,016 | 0 |
| Gateway, aiohttp | 33 ms | 79 ms | 9,052 | 0 |
| Gateway, httpx | 2,912 ms | 5,123 ms | 1,072 | 2 |

aiohttp adds next to nothing. httpx fell over at this concurrency (cause not investigated), repeating Week 1's finding with a load generator. A first run with an upstream pool of 100 connections gave aiohttp a 458 ms median: 200 users were queueing for 100 connections. Pool exhaustion looks like a slow client until you check.

## How a request flows

```
POST /v1/chat/completions
  │
  ├─ 1. rate limit: Lua token bucket in Redis, keyed by hashed API key (else IP)
  │        over budget → 429 + Retry-After          Redis down or slow → let it through, count it
  │
  ├─ 2. admission (app/admission.py)
  │        slot free, nobody waiting → go
  │        queue full → 503 overloaded (immediately)
  │        else wait, racing three things:
  │          slot frees → go     client leaves → dropped, never sent upstream     timeout → 503
  │
  ├─ 3. open the upstream request (pooled aiohttp session)
  │        unreachable → 502, slot released
  │
  └─ 4. relay bytes as they arrive; on any ending (done, client gone, never started):
           close upstream (stops the generation) and release the slot, exactly once
```

## Design decisions

| Decision | Why |
|---|---|
| Rate limit in **Redis**, backpressure in **memory** | The rate limit is per client and must hold across gateway replicas. The queue protects *this* upstream's capacity and must keep working when Redis is down. |
| **Lua** for the token bucket | Read, compute and write become one atomic command. Separate commands race (see the table above). Time comes from Redis, so replicas with drifting clocks agree. |
| **Fail open** when Redis fails | A Redis outage then costs fairness, not availability: the in-memory queue still protects the model. Every fail-open is counted in `/health` so it can't go unnoticed. For login attempts you would fail closed. |
| Release the slot around the **whole response**, not in the generator | If the client leaves before Starlette starts iterating a streaming generator, the generator's `finally` never runs. With one slot, one leak blocks the gateway forever. |
| Forward **raw bytes**, don't parse SSE | No added latency, nothing to get wrong, works for streaming and non-streaming alike. |
| **Timeouts** on every Redis wait | A hung Redis without a socket timeout froze `/health` (shown by a test with a fake Redis that never answers). |

## Tests and CI

50 tests, run against both proxy clients where it matters and against a real Redis. Each guarded behaviour was broken on purpose to check its test fails. That found three things:

- Two gaps in the first token bucket tests: refill counted twice on rejection, and a negative elapsed time after a clock jump. Both have tests now.
- A real bug: a crashing disconnect watcher was treated as "client gone", silently dropping a live request with a 499. It now raises.

CI (`.github/workflows/`):

| Job | What it proves |
|---|---|
| `ci.yml` → `test-week03` | The tests, with Redis as a **service container**. `REQUIRE_REDIS=1` turns "Redis missing" into a failure instead of a skip |
| `docker-image-week03.yml` → `build` | The image builds; on `main` it is published to GHCR. A `paths` filter keeps it separate from the 9 GB model image |
| `docker-image-week03.yml` → `compose-smoke` | The containers are wired correctly: `docker compose up --wait`, then health, a streamed answer, and a 429 |

## Docker

| | Week 2 model server | Week 3 gateway |
|---|---|---|
| Image size | 9.34 GB | **293 MB** |
| Dependencies | torch, transformers, CUDA wheels | FastAPI, aiohttp, redis |

The gateway image also runs the fake model server (`uvicorn app.fake_upstream:app`), so compose builds once and uses it twice. Only the gateway publishes a port; the fake is reachable only inside the compose network, as `http://fake:9000`.

## Known gaps

- **Non-streaming** requests whose client leaves still run to the end upstream (carried over from Week 2).
- A **hung** (not down) Redis adds up to the 0.5 s socket timeout to every request before failing open. Tested for `/health`, not measured on the proxy path. A circuit breaker would fix it.
- `Retry-After` on a 503 is a fixed 5 s, not derived from measured answer times.
- The "generator never started" release path is not reproduced by a test; the guard exists because Starlette's code shows the gap.
- Weeks 2 and 3 both have a package named `app`, so Week 3's tests need the `-o pythonpath=... -o testpaths=...` overrides.
- The API key only selects a rate-limit bucket; nothing validates it.

## Lessons learned

1. **A fast no beats a slow yes.** On the RTX 4090, bounding the queue cut the worst wait from 41 s to 5.5 s at 16 users and from 172 s to 5.5 s at 64, without adding any capacity.
2. **Queueing math predicts the result, if the inputs are measured where the system runs.** 2 × 2.73 s + first word = 5.5 s, exactly. The prediction made from a per-answer time measured through a proxy was 45% too high (8 s predicted, 5.5 s measured).
3. **Correct arithmetic is not enough.** The naive limiter used the right formula and still let 5x the limit through, because of the gap between two commands.
4. **Check what you think you are measuring.** A 458 ms "slow proxy" was a connection pool that was too small; an aiohttp-vs-httpx gap of 90x only showed up at high concurrency.
5. **Cleanup has more exits than you think.** A streaming response can end normally, be cancelled, or never start. Only one of those runs a generator's `finally`.
6. **Break your code to test your tests.** Mutation checks found gaps in tests written minutes earlier, and one real bug.
