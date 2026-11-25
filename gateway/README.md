# gateway

The async inference gateway that sits in front of the model server. Built in Week 3, given a C++ prompt cleaner in Week 4, and extended through Month 2 (model registry, Blue/Green switching, drift, rollback).

```
Client ──SSE──► [ gateway :8080 ] ──► [ model server ]
                  rate limit (429) → clean prompt (C++) → blue or green → that side's card → its admission (503) → its upstream (502) → relay bytes → record (Week 7)
```

The *why* behind each piece, with measurements, is in the weekly write-ups: [Week 3](../week03-gateway/README.md) (streaming, rate limiting, backpressure), [Week 4](../week04-cpp-cleaner/README.md) (the C++ cleaner), [Week 5](../week05-mlflow-registry/README.md) (the model registry: `MLFLOW_TRACKING_URI`, `REGISTRY_MODEL`, `ADMIN_TOKEN`, `POST /admin/reload`), [Week 6](../week06-blue-green/README.md) (Blue/Green: `GREEN_URL`, `TRAFFIC_GREEN`, `POST /admin/traffic`) and [Week 7](../week07-drift/README.md) (the request log for drift detection: `DRIFT_LOG`).

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
    champion.py           Week 5: resolve a settings card from an MLflow alias, apply it to requests
    router.py             Week 6: the Blue/Green traffic split, and which side a request goes to
    request_log.py        Week 7: one record per request to a Redis stream, for drift detection (off the hot path)
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
