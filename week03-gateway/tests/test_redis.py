import asyncio
import socket
import threading
import time

import httpx


async def health(url: str) -> tuple[dict, float]:
    async with httpx.AsyncClient(timeout=10) as c:
        t0 = time.perf_counter()
        r = await c.get(f"{url}/health")
    assert r.status_code == 200  # /health answers even when Redis is down
    return r.json(), time.perf_counter() - t0


async def test_health_reports_redis_ok(start_gateway, real_redis_url):
    body, _ = await health(start_gateway(redis_url=real_redis_url))
    assert body["redis"] == "ok"
    assert body["status"] == "ok"


async def test_health_degrades_when_redis_is_down(start_gateway):
    """Nothing listens on the port: the gateway still boots and answers, and says Redis is down."""
    body, elapsed = await health(start_gateway())
    assert body["redis"] == "down"
    assert body["status"] == "degraded"
    assert elapsed < 1.5


async def test_proxy_still_works_when_redis_is_down(start_gateway):
    url = start_gateway()
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{url}/v1/chat/completions", json={"messages": [], "stream": False, "max_tokens": 2})
    assert r.status_code == 200


def hung_server() -> tuple[socket.socket, str]:
    """A fake 'Redis' that accepts connections and never answers: a hung server, not a dead one."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    held = []

    def accept_forever():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            held.append(conn)  # keep it open, say nothing

    threading.Thread(target=accept_forever, daemon=True).start()
    return srv, f"redis://127.0.0.1:{srv.getsockname()[1]}"


async def test_hung_redis_hits_the_timeout_instead_of_hanging(start_gateway):
    """The Idea 5 lesson: without a socket timeout this request would wait forever. With one, PING gives up
    after REDIS_SOCKET_TIMEOUT (0.5 s) and /health reports Redis as down."""
    srv, url = hung_server()
    try:
        body, elapsed = await health(start_gateway(redis_url=url))
    finally:
        srv.close()
    assert body["redis"] == "down"
    assert 0.4 < elapsed < 2.0, f"expected ~0.5s timeout, took {elapsed:.2f}s"


async def test_many_concurrent_pings_share_a_small_pool(start_gateway, real_redis_url, monkeypatch):
    """100 simultaneous /health calls against a pool of 5 connections all succeed: callers queue for a free
    connection (BlockingConnectionPool) instead of erroring."""
    import app.main as gateway

    monkeypatch.setattr(gateway, "REDIS_MAX_CONNECTIONS", 5)
    url = start_gateway(redis_url=real_redis_url)
    async with httpx.AsyncClient(timeout=10, limits=httpx.Limits(max_connections=100)) as c:
        results = await asyncio.gather(*[c.get(f"{url}/health") for _ in range(100)])
    assert {r.json()["redis"] for r in results} == {"ok"}
