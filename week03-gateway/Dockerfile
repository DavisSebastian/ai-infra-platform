# syntax=docker/dockerfile:1
#
# The gateway image. Build from the REPO ROOT (the root pyproject.toml and uv.lock are needed):
#   docker build -f week03-gateway/Dockerfile -t week03-gateway:dev .
#
# Same two-stage pattern as week02-thin-slice/Dockerfile (build the virtualenv with uv, ship only the venv).
# The difference is what goes in: the `week03-serve` group has no torch and no model, so this image is small.
# The fake model server lives in the same app/ package, so this one image can run either:
#   gateway (default CMD)   or   uvicorn app.fake_upstream:app --host 0.0.0.0 --port 9000

# ---------- Stage 1: build the virtualenv ----------
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependency files first, so code edits don't re-run the install (Docker layer caching).
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group week03-serve

# ---------- Stage 2: the runtime image ----------
FROM python:3.12-slim-bookworm AS runtime

RUN useradd --create-home --uid 1000 app

# Defaults that make sense INSIDE docker compose: other services are reached by their service name, not
# localhost. Every one of these can be overridden at run time (12-factor config); none of them is a secret.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    REDIS_URL=redis://redis:6379 \
    UPSTREAM_URL=http://fake:9000

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY week03-gateway/app ./app

USER app
EXPOSE 8080

# /health answers 200 even when Redis is down (the gateway still proxies), so "unhealthy" means the gateway
# process itself is stuck, not a dependency.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
