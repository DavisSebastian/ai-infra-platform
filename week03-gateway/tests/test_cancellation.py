import asyncio
import time

import httpx

BODY = {"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 500}


async def stats(fake_url: str) -> dict:
    async with httpx.AsyncClient() as c:
        return (await c.get(f"{fake_url}/_stats")).json()


async def wait_for(fake_url: str, key: str, expected: int, timeout: float = 3.0) -> dict:
    """Poll the fake upstream until a counter reaches `expected` (cancellation travels over the network, so it
    takes a moment). Returns the last stats seen either way, so the assertion can print them."""
    deadline = time.monotonic() + timeout
    while True:
        s = await stats(fake_url)
        if s[key] >= expected or time.monotonic() > deadline:
            return s
        await asyncio.sleep(0.02)


async def test_client_leaving_the_gateway_stops_the_upstream(gateway_url, fake_url):
    """The whole point of step 2. The fake would emit 20 chunks (~0.4 s). We read 3 and walk away.
    If the gateway doesn't pass the departure upstream, the fake keeps going to the end and `finished` becomes 1
    (in Week 2 terms: the GPU keeps generating for a user who left)."""
    async with httpx.AsyncClient(timeout=10) as c:
        async with c.stream("POST", f"{gateway_url}/v1/chat/completions", json=BODY) as r:
            seen = 0
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    seen += 1
                if seen == 3:
                    break  # leaving the `async with` closes the connection to the gateway

    s = await wait_for(fake_url, "cancelled", 1)
    assert s["cancelled"] == 1, f"upstream was never told the client left: {s}"

    # Give a wrongly-running upstream time to finish its remaining chunks, then confirm it did not.
    await asyncio.sleep(0.6)
    s = await stats(fake_url)
    assert s["finished"] == 0, f"upstream ran to the end after the client left: {s}"


async def test_a_completed_stream_is_not_counted_as_cancelled(gateway_url, fake_url):
    """The opposite check, so the test above can't pass just because everything always looks cancelled."""
    async with httpx.AsyncClient(timeout=10) as c:
        async with c.stream("POST", f"{gateway_url}/v1/chat/completions", json=BODY) as r:
            async for _ in r.aiter_lines():
                pass

    s = await wait_for(fake_url, "finished", 1)
    assert s == {"started": 1, "finished": 1, "cancelled": 0}
