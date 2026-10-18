# Week 2 — The Thin Slice

A FastAPI server that streams answers from a real LLM, packaged in Docker and tested in CI. It is the naive baseline
the rest of the roadmap improves on: plain Hugging Face Transformers, one generation at a time.

| Endpoint | What it does |
|---|---|
| `POST /v1/chat/completions` | OpenAI-style chat. `"stream": true` returns SSE chunks, otherwise one JSON response |
| `GET /health` | Model name and device. Answers instantly even while the model is generating |

The API shape is OpenAI's on purpose: vLLM speaks the same format, so the Month 3 benchmark can point at either server
without changing the client.

## Run it

```bash
uv sync --group week02                                   # from repo root
cd week02-thin-slice
uv run uvicorn app.main:app --port 8000                  # defaults to Qwen2.5-0.5B-Instruct on the best device found

curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a short poem about servers."}],"max_tokens":150,"stream":true}'

uv run python scripts/loadtest.py /v1/chat/completions --prompt "Write a long story about a robot." \
  --max-tokens 100 -n 16 -c 16 --base-url http://127.0.0.1:8000
```

Settings come from environment variables, so the same code runs everywhere:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | Any Hugging Face chat model |
| `DEVICE` | best available (`cuda`, `mps`, `cpu`) | Set `cuda` on a GPU box so a broken driver fails loudly instead of silently using the CPU |
| `MAX_NEW_TOKENS` | `512` | Server-side cap on answer length, whatever the client asks for |
| `HF_TOKEN` | unset | Only needed for gated models such as Llama. Pass it at run time, never bake it into an image |

**Docker** (build from the repo root; the context needs the root `pyproject.toml` and `uv.lock`):

```bash
docker build -f week02-thin-slice/Dockerfile -t week02-thin-slice:dev .
docker run --rm -p 8000:8000 -v hf-cache:/models -e HF_TOKEN week02-thin-slice:dev
```

**Tests** (no GPU, no torch): `uv sync --group dev --group week02-serve && uv run pytest`

**GPU run:** see [RUNBOOK.md](RUNBOOK.md).

## Results

### Mac (Qwen2.5-0.5B on an M-series GPU, `max_tokens=100`, all users arrive at once)

| Users | TTFT p50 | TTFT p95 / max | ITL mean | Per-user speed | Total output |
|---|---|---|---|---|---|
| 1 | 32 ms | 33 ms | 12 ms | 85 chunks/s | 84 chunks/s |
| 4 | 2.1 s | 2.2 s | 12 ms | 80 chunks/s | 80 chunks/s |
| 16 | 6.6 s | 11.1 s | 12 ms | 81 chunks/s | 80 chunks/s |

### RunPod GPU (Llama 3.1 8B on an RTX 4090, `max_tokens=100`, all users arrive at once)

RTX 4090 (24 GB), $0.90/hour, driver 595 / CUDA 13.2. Measured from my laptop through RunPod's HTTPS proxy, so TTFT includes network time.

| Users | TTFT p50 | TTFT p95 / max | ITL mean | Per-user speed | Total output |
|---|---|---|---|---|---|
| 1 | 686 ms | 1.0 s | 47 ms (p50 38) | 21.6 chunks/s | 18.1 chunks/s |
| 4 | 7.6 s | 11.4 s | 43 ms (p50 35) | 23.4 chunks/s | 22.6 chunks/s |
| 16 | 31.4 s | 56.5 s | 45 ms (p50 35) | 23.0 chunks/s | 21.9 chunks/s |

| Resource | Measured |
|---|---|
| Cold start (deploy click to ready) | about 1 to 2 min (deploy 12:16, ready 12:17, timed to the minute). Model download + load to GPU: 55 s |
| VRAM, model loaded and idle | 15 GiB (8.03 B parameters x 2 bytes = 16 GB = 15 GiB) |
| VRAM, after the 16-user run | 16 GiB of 24 (PyTorch keeps what it grabbed, so this is about the peak; read from the pod's gauge, rounded) |
| Total cost of the session | $0.54 (balance $10.00 to $9.46), including two failed starts on bad hosts |

### How to read these numbers

Wait time grows with users, but speed doesn't change.

- **ITL and per-user speed stay flat** (12 ms, about 80 chunks/s) from 1 to 16 users. Each generation runs alone, so it never slows down.
- **TTFT grows with the queue.** The server runs one generation at a time, so user *k* waits for the *k - 1* answers ahead of them. The last user's wait (11.1 s) is about the whole run (11.2 s) minus their own answer.
- **Total output doesn't scale with users.** About 80 chunks/s whether 1 or 16 people ask. This is Little's Law from Week 1 with a single worker: more arrivals can only lengthen the line.

That is the problem Month 3 attacks. While decoding, the GPU is mostly waiting on memory, so it could serve many users at once; continuous batching (vLLM) does exactly that.

**GPU vs Mac.** What changed: the model is 16 times bigger and answers far better, but each chunk is slower (about 45 ms against 12 ms), so total output fell from 80 to about 22 chunks/s. What did not change: the shape. ITL and total output stay flat from 1 to 16 users, and TTFT grows with the queue. At 16 users the last person waited 56 s for the first word. A faster GPU does not fix a server that handles one request at a time. Rough estimate: reading 16 GB of weights per token at about 35 ms is about 460 GB/s, around half of a 4090's peak memory bandwidth, so there is room to serve several users at once. That is what Month 3 tests.

**Caveats.** These are single runs. The GPU numbers went through my laptop's internet and RunPod's proxy, so the mean ITL (45 ms against a 35 ms median) includes network jitter; running the load test inside the pod would be cleaner. The Mac numbers use a 0.5B model, so treat them as the shape of the result, not a benchmark. Answer lengths vary because the model decides when to stop (the average was 50 to 76 chunks per request). "ITL" is really inter-chunk latency: the streamer emits pieces of text, usually a word, not guaranteed single tokens.

## How a request flows

```
event loop thread                                    generation thread
─────────────────                                    ─────────────────
validate, answer headers
take the generation lock
start thread ──────────────────────────────────────▶ model.generate() ...
async for piece in queue  ◀─── words via a queue ─── one piece per token
send SSE event to the client
...
await thread to finish, release lock
send finish_reason and [DONE]
```

- The **model loads once** at startup (`lifespan`). Cached on disk, it loads in about 3 s; downloading takes minutes.
- **`generate()` never runs on the event loop.** It runs on its own thread and hands text over through a thread-safe queue, so `/health` and queued requests keep being served.
- **One generation at a time** (`asyncio.Lock`). Two at once on one GPU gives no speedup with plain Transformers and risks running out of memory. Waiting on the lock parks the request instead of blocking the loop.
- **The server caps `max_tokens`** whatever the client asks, so one request can't hold the GPU for minutes.

## Three bugs, found by measuring

| Bug | What happened | Fix | Test that guards it |
|---|---|---|---|
| **`generate()` called inside `async def`** | `/health` took **4.6 s** while a long answer generated (Week 1's `/blocking` again, with a real cause). A load balancer would mark the server dead and restart it | Run generation on a thread and stream through a queue: `/health` answers in about 0.00 s | `test_health_stays_fast_during_non_streaming_generation` |
| **Client disconnect left the GPU working** | The disconnect cancels the request's *coroutine*, not the *thread* doing the work. Generation ran **4.4 s more** for nobody (forced-length run on the 0.5B model), and the lock was released while the thread still ran | A stop flag checked after every token (`StoppingCriteria`), set on disconnect | `test_client_disconnect_stops_generation` |
| **The lock was released too early** | After a disconnect, the cleanup `await` is cancelled immediately, so the lock was released with the thread still alive. A short window (about one token, 30 ms here), but a second generation could start | Wait for the thread inside `anyio.CancelScope(shield=True)` before releasing the lock | `test_next_request_never_overlaps_a_cancelled_generation` |

The first two numbers come from a tiny model on a Mac, so the *size* of the waste isn't the point. The point is that
the disconnect and the thread live in different worlds, and nothing connects them unless you do.

## Tests and CI

13 tests run the **real app on real uvicorn** with a fake model that produces words on a real thread and stops on a
flag, the way the real one does. No torch, no GPU. Each concurrency test guards one of the bugs above.

I checked that the tests can fail by breaking the server four ways (blocking `generate()`, no shield, no lock, no
cancel). Each break failed the test meant to catch it.

| CI job | What it does |
|---|---|
| `lint` | `ruff check` and `ruff format --check` |
| `test` | `pytest` with only the `week02-serve` dependency group (no torch) and `uv sync --locked` |
| `Docker image` | Builds the **amd64** image on every PR. Publishes to GHCR (`:latest` and `:sha-<commit>`) only on pushes to `main` |

Not covered in CI: the real `Stream` with a real model. I checked that path by hand.

## Docker

Two stages: a builder installs the dependencies with a pinned `uv`; the runtime image receives only the finished
virtualenv. It runs as a non-root user and has a healthcheck.

| What I measured (Mac, Qwen 0.5B) | Result |
|---|---|
| First start (downloads the model into a volume) | 111 s |
| New container, same volume | 4 s |
| Rebuild after changing one line of app code | 2.9 s (first build: 8 min 44 s) |
| Image size (arm64) | 9.3 GB |

- **Layer order matters.** Dependency files are copied first, so the slow install is cached until `pyproject.toml` or `uv.lock` change.
- **Weights live in a volume, not the image.** Otherwise every fresh container re-downloads the model. Code and model are versioned separately.
- **Secrets are passed at run time** (`-e HF_TOKEN`), never `ENV`/`ARG` in the Dockerfile. I checked that `docker inspect` and `docker history` contain none.
- **`--host 0.0.0.0`** is required in a container; the default `127.0.0.1` accepts nothing from outside.
- **The image is big because PyTorch's Linux wheel bundles CUDA** (about 3.3 GB of `nvidia-*` libraries plus 0.8 GB of Triton), even on a Mac where they are unused. So the GPU host only needs the NVIDIA driver, not a CUDA base image. The lockfile pins CUDA 13 libraries, which needs a recent driver.
- **A Mac builds arm64; RunPod GPUs run amd64.** That is why CI builds the image.
- `.dockerignore` is deny-by-default. Without it, the 791 MB `.venv` would be uploaded to Docker on every build.

## Lessons learned

- **Cancelling a request cancels the coroutine, not the thread doing the work.** Cooperative stopping (a flag) plus cleanup that survives cancellation is what makes the lock honest.
- **A test is only worth trusting if it can fail.** Break the code on purpose and watch the right test go red.
- **Check the check.** The load tester reported 21 ms against a server that sleeps 20 ms before I trusted it on a model. (Same lesson as Week 1's httpx.)
- **Rented GPUs fail in ways your code can't predict.** Two starts failed before the third worked: one host's GPU reported "requires reset", and another's driver only supported CUDA 12.8 while the image's PyTorch needs CUDA 13. `DEVICE=cuda` turned the second into a loud crash instead of a silent, hours-long CPU run. Filtering for CUDA 13 hosts fixed it. Match the image's CUDA version to what the fleet's drivers support.
- **Verify a runbook before relying on it.** Checking my own instructions found that the image has no `curl`, so the first draft of the health check would have failed on the GPU box.
