# gateway

The async inference gateway that sits in front of the model server. Built in Week 3, given a C++ prompt cleaner in Week 4, and extended through Month 2 (model registry, Blue/Green switching, drift, rollback).

```
Client ──SSE──► [ gateway :8080 ] ──► [ model server ]
                  rate limit (429) → clean prompt (C++) → admission (503) → pooled upstream (502) → relay bytes
```

The *why* behind each piece, with measurements, is in the weekly write-ups: [Week 3](../week03-gateway/README.md) (streaming, rate limiting, backpressure) and [Week 4](../week04-cpp-cleaner/README.md) (the C++ cleaner).

## Layout

```
gateway/
  pyproject.toml          the package: its runtime dependencies and build backend
  src/gateway/            the code (`import gateway`)
    main.py               HTTP API and the order of the checks
    redis_limiter.py      token bucket as one atomic Lua script
    admission.py          in-flight cap, bounded queue, queue timeout
    upstream.py           one pooled HTTP client to the model server
    sanitize.py           prompt cleaning via promptclean (C++)
    fake_upstream.py      a stand-in model server with exact timing, for tests and compose
  tests/
  Dockerfile              build from the repo root
  docker-compose.yml      gateway + Redis + fake model server
```

The root `pyproject.toml` installs this package through the `week03-serve` dependency group as an *editable* install, so code edits apply without reinstalling. The Docker image installs it non-editable, into the virtualenv.

## Run, test

From the repo root:

```bash
uv sync --group dev --group week03
(cd gateway && docker compose up -d redis --wait)    # the tests need Redis
uv run pytest gateway/tests
UPSTREAM_URL=http://localhost:9000 uv run uvicorn gateway.main:app --port 8080
```

The whole stack in Docker:

```bash
cd gateway
docker compose up -d --build --wait
curl localhost:8080/health
docker compose down
```

Settings are environment variables; the table is in the [Week 3 README](../week03-gateway/README.md#settings-all-environment-variables).
