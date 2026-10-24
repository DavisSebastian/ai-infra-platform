import json
import time

import httpx

BODY = {"messages": [{"role": "user", "content": "hi"}], "stream": True}


def parse(lines: list[str]) -> list[str]:
    return [line.removeprefix("data: ") for line in lines if line.startswith("data: ")]


async def test_health(gateway_url):
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{gateway_url}/health")
    assert r.status_code == 200
    # Redis is unreachable in these tests, so the gateway reports "degraded" but is still serving.
    assert r.json()["status"] == "degraded"


async def test_stream_matches_upstream_format(gateway_url):
    """Role delta first, content pieces, a finish event, then [DONE]: exactly what the model server sends."""
    async with httpx.AsyncClient(timeout=10) as c:
        async with c.stream("POST", f"{gateway_url}/v1/chat/completions", json=BODY) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            events = parse([line async for line in r.aiter_lines()])

    assert events[-1] == "[DONE]"
    first = json.loads(events[0])
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert json.loads(events[-2])["choices"][0]["finish_reason"] == "stop"
    assert len(events) == 20 + 3  # role + 20 content pieces + finish + [DONE]


async def test_chunks_are_relayed_as_they_arrive_not_buffered(gateway_url):
    """The core promise of a streaming proxy. The fake emits a chunk every 20 ms for ~0.4 s in total.
    If the gateway buffered the answer, the first chunk would arrive at ~0.4 s. Streamed, it arrives almost at once."""
    async with httpx.AsyncClient(timeout=10) as c:
        t0 = time.perf_counter()
        arrivals = []
        async with c.stream("POST", f"{gateway_url}/v1/chat/completions", json=BODY) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    arrivals.append(time.perf_counter() - t0)

    total = arrivals[-1]
    assert total > 0.3, "fake upstream should take ~0.4 s"
    assert arrivals[0] < 0.15, f"first event took {arrivals[0]:.3f}s: the gateway is buffering"
    assert arrivals[5] < total / 2, "events are bunched up at the end"


async def test_non_streaming_request_is_forwarded(gateway_url):
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{gateway_url}/v1/chat/completions", json={**BODY, "stream": False, "max_tokens": 3})
    assert r.status_code == 200
    assert r.json()["usage"]["completion_tokens"] == 3


async def test_upstream_error_status_is_relayed(gateway_url):
    """A body the upstream rejects (422 from validation) comes back with the same status, not a 200 or a crash."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{gateway_url}/v1/chat/completions", json={"nope": 1})
    assert r.status_code == 422
