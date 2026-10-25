# syntax=docker/dockerfile:1
#
# Build from the REPO ROOT (the root pyproject.toml and uv.lock are needed):
#   docker build -f week02-thin-slice/Dockerfile -t week02-thin-slice:dev .
#
# Two stages. The first has uv and does the slow work of installing dependencies. The second is a plain
# Python image that receives only the finished virtualenv, so uv and its caches never ship.
# Java analogy: build with Maven in a JDK image, then copy only the JAR into a JRE image.

# ---------- Stage 1: build the virtualenv ----------
FROM python:3.12-slim-bookworm AS builder

# Copy in the uv binary, pinned to the version this repo uses. Both stages share the same Python base image,
# so the virtualenv built here finds exactly the same interpreter path when copied into stage 2.
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv

# COMPILE_BYTECODE: precompile .pyc files now (the non-root runtime user can't write them later).
# LINK_MODE=copy:   copy files into the venv (the cache mount below won't exist in the final image).
# PYTHON_DOWNLOADS: never download a Python; use the one in this image.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependency files first. Docker caches each step, so this slow install is re-run only when
# pyproject.toml or uv.lock change, not on every code edit. (Same idea as Jib's separate dependency layer.)
COPY pyproject.toml uv.lock ./
# uv.lock also lists promptclean (Week 4), a package installed from a local folder. `uv sync --locked` must read
# its metadata to check the lockfile is current, even though this image never installs it. Its pyproject.toml is
# enough: nothing is compiled here. (Without it the build fails with "Failed to generate package metadata".)
COPY week04-cpp-cleaner/pyproject.toml ./week04-cpp-cleaner/pyproject.toml

# --locked: fail if uv.lock is out of date, so the image has exactly the pinned versions.
# --no-dev: leave out ruff, pytest and friends. --group week02: the server plus torch and transformers.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group week02

# ---------- Stage 2: the runtime image ----------
FROM python:3.12-slim-bookworm AS runtime

# Run as a normal user, not root: if the process is ever compromised, it has fewer rights.
# /models is where Hugging Face downloads weights. It is created here, owned by that user, so a Docker
# volume mounted on it starts out writable.
RUN useradd --create-home --uid 1000 app \
    && mkdir /models \
    && chown app:app /models

# HF_HOME:         where from_pretrained() caches model files. Mount a volume here so the weights live
#                  OUTSIDE the image and survive container restarts.
# PYTHONUNBUFFERED: print logs immediately instead of holding them in a buffer.
# No secrets here! HF_TOKEN is passed at `docker run` time and never written into the image.
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY week02-thin-slice/app ./app

USER app
EXPOSE 8000

# Reports "unhealthy" if /health stops answering. start-period gives a big model time to load first.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# --host 0.0.0.0 is essential in a container: the default, 127.0.0.1, only accepts connections from
# INSIDE the container, so `-p 8000:8000` would reach nothing.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
