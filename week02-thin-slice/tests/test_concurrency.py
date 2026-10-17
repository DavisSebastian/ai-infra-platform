"""The behaviours Week 2 was really about: the event loop stays free, one generation at a time,
and a client disconnect stops generation without breaking the lock. Each test guards a bug we hit."""

import asyncio
import json
import time

from test_api import CHAT, stream_events

SLOW = {"n_pieces": 50, "delay": 0.02}  # about 1 second of generation


def has_text(line: str) -> bool:
    if not line.startswith("data: ") or line.endswith("[DONE]"):
        return False
    return bool(json.loads(line[len("data: ") :])["choices"][0]["delta"].get("content"))


async def read_pieces_then_disconnect(client, how_many: int) -> None:
    """Start a streaming request, read a few text pieces, then drop the connection."""
    async with client.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True}) as resp:
        seen = 0
        async for line in resp.aiter_lines():
            seen += has_text(line)
            if seen == how_many:
                break  # leaving the `async with` closes the connection: the client has "gone"


async def wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met in time")


async def test_health_stays_fast_during_streaming_generation(client, fake_llm):
    fake_llm.n_pieces, fake_llm.delay = SLOW["n_pieces"], SLOW["delay"]

    async with client.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True}) as resp:
        async for line in resp.aiter_lines():
            if has_text(line):
                break  # generation is definitely running now
        start = time.perf_counter()
        health = await client.get("/health")
        elapsed = time.perf_counter() - start

    assert health.status_code == 200
    assert elapsed < 0.25


async def test_health_stays_fast_during_non_streaming_generation(client, fake_llm):
    # Regression test for the step 2 bug: generate() called directly inside `async def` freezes the server.
    fake_llm.n_pieces, fake_llm.delay = SLOW["n_pieces"], SLOW["delay"]

    slow = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT))
    await wait_until(lambda: fake_llm.active == 1)
    start = time.perf_counter()
    health = await client.get("/health")
    elapsed = time.perf_counter() - start

    assert health.status_code == 200
    assert elapsed < 0.25
    assert not slow.done(), "the test must overlap with a generation that is still running"
    await slow


async def test_only_one_generation_runs_at_a_time(client, fake_llm):
    fake_llm.n_pieces, fake_llm.delay = 5, 0.02

    results = await asyncio.gather(*(stream_events(client) for _ in range(3)))

    assert all(events[-1] == "[DONE]" for events in results)
    assert len(fake_llm.starts) == 3
    assert fake_llm.max_active == 1


async def test_client_disconnect_stops_generation(client, fake_llm):
    fake_llm.n_pieces, fake_llm.delay = 100, 0.02  # about 2 seconds if nobody stops it

    await read_pieces_then_disconnect(client, how_many=3)
    await wait_until(lambda: len(fake_llm.finishes) == 1)

    stream = fake_llm.streams[0]
    assert stream.cancel_called
    assert stream.completion_tokens < 20  # stopped within a few pieces, not run to 100


async def test_next_request_never_overlaps_a_cancelled_generation(client, fake_llm):
    # Regression test for the lock bug: the lock must only be released once the thread has really finished.
    fake_llm.n_pieces, fake_llm.delay = 100, 0.02
    await read_pieces_then_disconnect(client, how_many=3)

    fake_llm.n_pieces = 3
    events = await stream_events(client)

    assert events[-1] == "[DONE]"
    assert len(fake_llm.starts) == 2
    assert fake_llm.max_active == 1
    assert fake_llm.starts[1] >= fake_llm.finishes[0]


async def test_client_leaving_the_queue_never_starts_a_generation(client, fake_llm):
    fake_llm.n_pieces, fake_llm.delay = SLOW["n_pieces"], SLOW["delay"]

    first = asyncio.create_task(stream_events(client))
    await wait_until(lambda: fake_llm.active == 1)
    async with client.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True}):
        pass  # queued behind the first request, then leaves straight away

    events = await first
    await asyncio.sleep(0.1)

    assert events[-1] == "[DONE]"
    assert len(fake_llm.starts) == 1
