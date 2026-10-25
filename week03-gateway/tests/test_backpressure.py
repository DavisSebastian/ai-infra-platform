"""Backpressure: in-flight cap, bounded queue, queue timeout, dropping clients who left, no leaked slots.

The fake upstream streams 20 chunks 20 ms apart, so one answer takes ~0.42 s. Redis is unreachable in these
tests, so rate limiting fails open and only admission control decides.
"""

import asyncio
import time

import app.main as gateway
import httpx
import pytest
from conftest import closed_port_url

BODY = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
ANSWER = 0.42  # seconds per answer from the fake


@pytest.fixture
def settings(monkeypatch):
    def _set(max_in_flight=1, max_queue=2, queue_timeout=10.0):
        monkeypatch.setattr(gateway, "MAX_IN_FLIGHT", max_in_flight)
        monkeypatch.setattr(gateway, "MAX_QUEUE", max_queue)
        monkeypatch.setattr(gateway, "QUEUE_TIMEOUT", queue_timeout)

    return _set


async def stream_one(url: str, t0: float) -> tuple[int, float, dict | None]:
    """One streaming request, read to the end. Returns (status, seconds until it finished, error body or None)."""
    async with httpx.AsyncClient(timeout=30) as c:
        async with c.stream("POST", f"{url}/v1/chat/completions", json=BODY) as r:
            await r.aread()
            err = None if r.status_code == 200 else {"body": r.json(), "retry_after": r.headers.get("retry-after")}
            return r.status_code, time.perf_counter() - t0, err


async def get(url: str, path: str) -> dict:
    async with httpx.AsyncClient() as c:
        return (await c.get(url + path)).json()


async def test_one_in_flight_two_queued_rest_rejected_fast(settings, start_gateway, fake_url):
    """The Idea 6 prediction: cap 1, queue 2, 6 users at once -> 3 served, 3 rejected in milliseconds."""
    settings(max_in_flight=1, max_queue=2)
    url = start_gateway()
    t0 = time.perf_counter()
    results = await asyncio.gather(*[stream_one(url, t0) for _ in range(6)])

    served = sorted(t for status, t, _ in results if status == 200)
    rejected = [(t, err) for status, t, err in results if status == 503]
    assert len(served) == 3 and len(rejected) == 3, results
    assert max(t for t, _ in rejected) < 0.15, "rejections must be immediate, not after waiting"
    for _, err in rejected:
        assert err["body"]["error"]["type"] == "overloaded"
        assert int(err["retry_after"]) >= 1
    # Served one after another: ~1x, ~2x, ~3x the answer time.
    assert served[2] == pytest.approx(3 * ANSWER, abs=0.25)

    stats = await get(fake_url, "/_stats")
    assert stats["max_active"] == 1, f"the model server must never see more than 1 request at once: {stats}"
    health = await get(url, "/health")
    assert health["rejected"]["overloaded"] == 3
    assert (health["in_flight"], health["queued"]) == (0, 0)


async def test_queue_timeout_rejects_requests_that_waited_too_long(settings, start_gateway):
    settings(max_in_flight=1, max_queue=2, queue_timeout=0.2)
    url = start_gateway()
    t0 = time.perf_counter()
    results = await asyncio.gather(*[stream_one(url, t0) for _ in range(3)])

    assert sorted(status for status, _, _ in results) == [200, 503, 503]
    for status, t, err in results:
        if status == 503:
            assert err["body"]["error"]["type"] == "queue_timeout"
            assert 0.15 < t < 0.4, f"should give up after ~0.2 s, took {t:.2f}"


async def test_client_who_leaves_the_queue_is_never_sent_upstream(settings, start_gateway, fake_url):
    """Piece 5 of Idea 6: a queued request whose client has gone would cost a whole generation for nobody."""
    settings(max_in_flight=1, max_queue=2)
    url = start_gateway()
    t0 = time.perf_counter()
    first = asyncio.create_task(stream_one(url, t0))  # takes the only slot for ~0.42 s
    await asyncio.sleep(0.05)
    leaver = asyncio.create_task(stream_one(url, t0))  # waits in the queue...
    await asyncio.sleep(0.1)
    assert (await get(url, "/health"))["queued"] == 1
    leaver.cancel()  # ...and walks away (its client closes the connection)

    assert (await first)[0] == 200
    await asyncio.sleep(0.2)
    stats = await get(fake_url, "/_stats")
    assert stats["started"] == 1, f"the departed client's request reached the model server: {stats}"
    health = await get(url, "/health")
    assert health["rejected"]["client_gone"] == 1
    assert (health["in_flight"], health["queued"]) == (0, 0)


async def test_slot_is_released_when_a_client_leaves_mid_stream(settings, start_gateway):
    """With a cap of 1, a single leaked slot would block every later request. Leave mid-stream, then try again."""
    settings(max_in_flight=1, max_queue=2, queue_timeout=1.0)
    url = start_gateway()
    async with httpx.AsyncClient(timeout=10) as c:
        async with c.stream("POST", f"{url}/v1/chat/completions", json=BODY) as r:
            async for _ in r.aiter_lines():
                break  # leave after the first line

    await asyncio.sleep(0.2)
    assert (await get(url, "/health"))["in_flight"] == 0
    status, _, _ = await stream_one(url, time.perf_counter())
    assert status == 200


async def test_dead_upstream_gives_502_and_frees_the_slot(settings, start_gateway):
    settings(max_in_flight=1, max_queue=2, queue_timeout=1.0)
    url = start_gateway(upstream_url=closed_port_url().replace("redis://", "http://"))
    t0 = time.perf_counter()
    first = await stream_one(url, t0)
    second = await stream_one(url, t0)  # would time out in the queue if the first had leaked its slot
    assert first[0] == 502 and second[0] == 502
    assert first[2]["body"]["error"]["type"] == "upstream_unavailable"
    health = await get(url, "/health")
    assert health["in_flight"] == 0
    assert health["rejected"]["upstream_error"] == 2


async def test_a_crashing_disconnect_watcher_is_not_mistaken_for_a_departed_client():
    """Found while mutation testing: a watcher that raised was treated as "client gone", silently dropping a live
    request. It must surface as an error instead."""
    from app.admission import Admission

    adm = Admission(max_in_flight=1, max_queue=2, queue_timeout=1.0)

    async def never():
        await asyncio.Event().wait()

    async def broken():
        raise RuntimeError("watcher bug")

    slot = await adm.acquire(never)  # takes the only slot
    with pytest.raises(RuntimeError, match="watcher bug"):
        await adm.acquire(broken)  # queued, and its watcher crashes
    slot.release()
    assert (adm.in_flight, adm.queued) == (0, 0)
