"""Week 7: the request log. It records what drift detection needs, and never slows or fails a request."""

import asyncio
import json
import time
import uuid

import gateway.fake_upstream as fake_upstream
import gateway.main as gateway
import httpx
import pytest
import redis.asyncio as redis
from gateway.request_log import Finished, RequestLog, answer_of, first_token_ms, prompt_of

CHAT = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}

# --- Pulling the prompt and the answer out of raw bytes ---------------------------------------------------


@pytest.mark.parametrize(
    ("body", "prompt"),
    [
        (b'{"messages": [{"role": "user", "content": "hi"}]}', "hi"),
        # the LAST user message is what was asked; the system prompt and earlier turns are context
        (
            b'{"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "first"},'
            b' {"role": "assistant", "content": "ok"}, {"role": "user", "content": "second"}]}',
            "second",
        ),  # fmt: skip
        (b'{"messages": [{"role": "user", "content": [{"type": "text", "text": "parts"}]}]}', "parts"),
        (b"not json", ""),
        (b'{"no": "messages"}', ""),
        (b"[1, 2]", ""),
    ],
)
def test_prompt_of(body, prompt):
    assert prompt_of(body) == prompt


def sse(*pieces: str) -> bytes:
    events = [{"choices": [{"delta": {"role": "assistant", "content": ""}}]}]
    events += [{"choices": [{"delta": {"content": p}}]} for p in pieces]
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode() + b"data: [DONE]\n\n"


def test_answer_of_a_stream_joins_the_pieces():
    assert answer_of(sse("Canberra ", "is the capital.")) == "Canberra is the capital."


def test_answer_of_a_plain_json_response():
    raw = json.dumps({"choices": [{"message": {"role": "assistant", "content": "Canberra."}}]}).encode()
    assert answer_of(raw) == "Canberra."


def test_answer_of_a_stream_cut_at_the_capture_limit_keeps_what_arrived():
    raw = sse("one ", "two ", "three")
    assert answer_of(raw[: raw.index(b"three") - 5]) == "one two "  # the cut-off event is skipped, not a crash


@pytest.mark.parametrize("raw", [b"", b"garbage", b"data: not json\n\n", b'{"choices": []}'])
def test_answer_of_junk_is_empty(raw):
    assert answer_of(raw) == ""


# --- Time to first token: the first event with TEXT, not the first chunk ---------------------------------


def test_first_token_is_the_first_text_event_not_the_role_event():
    raw = sse("Canberra", " is the capital.")
    role_end = raw.index(b"\n\n") + 2  # the role-only event, which servers send before generating anything
    text_end = raw.index(b"Canberra") + 40
    arrivals = [(role_end, 2.0), (text_end, 180.0), (len(raw), 400.0)]
    assert first_token_ms(raw, arrivals) == 180.0  # not 2.0: that would be time to first BYTE


def test_first_token_when_one_chunk_holds_several_events():
    raw = sse("Canberra")
    assert first_token_ms(raw, [(len(raw), 95.0)]) == 95.0


@pytest.mark.parametrize("raw", [sse(), b'{"choices": [{"message": {"content": "plain JSON"}}]}', b""])
def test_no_first_token_without_streamed_text(raw):
    assert first_token_ms(raw, [(len(raw), 5.0)]) is None


# --- The writer: never blocks, never raises ---------------------------------------------------------------


class FakeRedis:
    def __init__(self):
        self.entries: list[dict] = []
        self.down = False

    async def xadd(self, stream, fields, maxlen, approximate):
        if self.down:
            raise redis.ConnectionError("Redis is down")
        self.entries.append({"stream": stream, "maxlen": maxlen, **json.loads(fields["data"])})


def finished(**kw) -> Finished:
    raw = sse("w0 ", "w1")
    first_text_end = raw.index(b"w0") + 30  # somewhere after the first text event's line ends
    defaults = dict(side="blue", card="2", status=200, body=json.dumps(CHAT).encode(), answer=raw,
                    arrivals=[(10, 1.04), (first_text_end, 12.34), (len(raw), 50.0)], ms=56.78,
                    completed=True)  # fmt: skip
    return Finished(**{**defaults, **kw})


def test_record_never_waits_when_the_queue_is_full():
    rl = RequestLog(FakeRedis(), "s", 100, queue_size=2)
    start = time.perf_counter()
    for _ in range(5):
        rl.record(finished())
    assert time.perf_counter() - start < 0.01
    assert rl.counts == {"written": 0, "dropped": 3, "failed": 0}


async def test_the_writer_stores_one_parsed_record_per_request():
    fake = FakeRedis()
    rl = RequestLog(fake, "gateway:requests", 10_000)
    writer = asyncio.create_task(rl.run())
    rl.record(finished())
    await asyncio.wait_for(rl.queue.join(), 2)
    writer.cancel()
    e = fake.entries[0]
    assert (e["stream"], e["maxlen"]) == ("gateway:requests", 10_000)
    assert (e["side"], e["card"], e["status"], e["completed"]) == ("blue", "2", 200, True)
    assert (e["prompt"], e["answer"], e["first_byte_ms"], e["ttft_ms"], e["ms"]) == ("hi", "w0 w1", 1.0, 12.3, 56.8)
    assert rl.counts["written"] == 1


async def test_a_redis_outage_is_counted_and_the_writer_keeps_going():
    fake = FakeRedis()
    rl = RequestLog(fake, "s", 100)
    writer = asyncio.create_task(rl.run())
    fake.down = True
    rl.record(finished())
    await asyncio.wait_for(rl.queue.join(), 2)
    fake.down = False
    rl.record(finished())
    await asyncio.wait_for(rl.queue.join(), 2)
    writer.cancel()
    assert rl.counts == {"written": 1, "dropped": 0, "failed": 1}


# --- Through the real gateway, with a real Redis -----------------------------------------------------------


@pytest.fixture
def stream_name():
    return f"test:requests:{uuid.uuid4().hex[:8]}"


@pytest.fixture
def logging_gateway(start_gateway, real_redis_url, stream_name, monkeypatch):
    monkeypatch.setattr(gateway, "DRIFT_LOG", True)
    monkeypatch.setattr(gateway, "DRIFT_STREAM", stream_name)
    return start_gateway(redis_url=real_redis_url)


async def entries(redis_url: str, stream: str, n: int, timeout: float = 5) -> list[dict]:
    """Wait until the stream holds n records (the writer is asynchronous), then return them, oldest first."""
    client = redis.from_url(redis_url)
    try:
        deadline = time.monotonic() + timeout
        while True:
            rows = await client.xrange(stream)
            if len(rows) >= n or time.monotonic() > deadline:
                return [json.loads(fields[b"data"]) for _, fields in rows]
            await asyncio.sleep(0.05)
    finally:
        await client.delete(stream)
        await client.aclose()


async def test_a_streamed_request_is_recorded(logging_gateway, real_redis_url, stream_name):
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{logging_gateway}/v1/chat/completions", json={**CHAT, "stream": True})
    assert r.status_code == 200
    [e] = await entries(real_redis_url, stream_name, 1)
    assert (e["side"], e["status"], e["completed"], e["prompt"], e["answer"]) == ("blue", 200, True, "hi", "w0 w1 ")
    assert e["card"] is None  # registry off
    assert 0 < e["first_byte_ms"] <= e["ttft_ms"] <= e["ms"]


async def test_a_plain_json_request_is_recorded(logging_gateway, real_redis_url, stream_name):
    async with httpx.AsyncClient(timeout=10) as c:
        await c.post(f"{logging_gateway}/v1/chat/completions", json=CHAT)
    [e] = await entries(real_redis_url, stream_name, 1)
    assert (e["prompt"], e["answer"]) == ("hi", "w0 w1")


async def test_a_client_leaving_mid_stream_is_recorded_as_not_completed(
    logging_gateway, real_redis_url, stream_name, monkeypatch
):
    monkeypatch.setattr(fake_upstream, "CHUNK_DELAY", 0.05)
    async with httpx.AsyncClient(timeout=10) as c:
        body = {**CHAT, "max_tokens": 20, "stream": True}
        async with c.stream("POST", f"{logging_gateway}/v1/chat/completions", json=body) as s:
            await anext(s.aiter_lines())  # read the first event, then leave
    [e] = await entries(real_redis_url, stream_name, 1)
    assert e["completed"] is False


async def test_with_redis_down_requests_still_succeed_and_the_failure_is_counted(start_gateway, monkeypatch):
    monkeypatch.setattr(gateway, "DRIFT_LOG", True)
    url = start_gateway()  # the default Redis URL here points at a closed port
    async with httpx.AsyncClient(timeout=10) as c:
        assert (await c.post(f"{url}/v1/chat/completions", json=CHAT)).status_code == 200
        deadline = time.monotonic() + 5
        while (counts := (await c.get(f"{url}/health")).json()["request_log"])["failed"] == 0:
            assert time.monotonic() < deadline, counts
            await asyncio.sleep(0.05)
    assert counts["written"] == 0


async def test_off_by_default(gateway_url):
    async with httpx.AsyncClient() as c:
        assert (await c.get(f"{gateway_url}/health")).json()["request_log"] == "off"
