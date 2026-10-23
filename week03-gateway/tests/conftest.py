import os
import socket
import threading
import time

import app.fake_upstream as fake_upstream
import app.main as gateway
import pytest
import uvicorn


def serve(asgi_app) -> tuple[uvicorn.Server, threading.Thread, str]:
    """Run an ASGI app on a free port in its own thread (its own event loop), like Week 2's `server` fixture."""
    config = uvicorn.Config(asgi_app, host="127.0.0.1", port=0, log_level="warning", timeout_graceful_shutdown=2)
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.01)
    port = srv.servers[0].sockets[0].getsockname()[1]
    return srv, thread, f"http://127.0.0.1:{port}"


def stop(srv: uvicorn.Server, thread: threading.Thread) -> None:
    srv.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def fake_url():
    """The fake model server."""
    for key in stats_keys():
        fake_upstream.stats[key] = 0
    srv, thread, url = serve(fake_upstream.app)
    yield url
    stop(srv, thread)


def stats_keys():
    return list(fake_upstream.stats)


REAL_REDIS = "redis://localhost:6379"  # started by `docker compose up -d` (CI uses a service container)


def closed_port_url() -> str:
    """A Redis URL where nothing is listening: grab a free port, then close it."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"redis://127.0.0.1:{port}"


@pytest.fixture
def real_redis_url():
    """The real Redis. Locally, tests that need it are skipped when it isn't running. In CI we set
    REQUIRE_REDIS=1, so a missing Redis FAILS the build instead of quietly skipping the tests that matter."""
    host_port = REAL_REDIS.removeprefix("redis://").split(":")
    try:
        socket.create_connection((host_port[0], int(host_port[1])), timeout=1).close()
    except OSError:
        if os.getenv("REQUIRE_REDIS"):
            pytest.fail("REQUIRE_REDIS is set but Redis is not reachable at " + REAL_REDIS)
        pytest.skip("Redis is not running: `docker compose up -d` in week03-gateway/")
    return REAL_REDIS


@pytest.fixture
def start_gateway(fake_url, monkeypatch):
    """Factory: start the real gateway against the fake model server and a chosen Redis URL."""
    started = []

    def _start(client: str = "aiohttp", redis_url: str | None = None) -> str:
        monkeypatch.setattr(gateway, "UPSTREAM_URL", fake_url)
        monkeypatch.setattr(gateway, "PROXY_CLIENT", client)
        monkeypatch.setattr(gateway, "REDIS_URL", redis_url or closed_port_url())
        srv, thread, url = serve(gateway.app)
        started.append((srv, thread))
        return url

    yield _start
    for srv, thread in started:
        stop(srv, thread)


@pytest.fixture(params=["aiohttp", "httpx"])
def gateway_url(request, start_gateway):
    """The real gateway pointed at the fake model server. Every test runs once per proxy client.
    Redis is deliberately unreachable here: the proxy must not depend on it."""
    return start_gateway(request.param)
