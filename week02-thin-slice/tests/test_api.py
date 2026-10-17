"""The API contract: response shapes, SSE format, token caps. Fast; no real model."""

import json

import app.main as main

CHAT = {"messages": [{"role": "user", "content": "hi"}]}


async def stream_events(client, **extra) -> list[str]:
    """POST a streaming chat request and return the payload of every `data:` line."""
    async with client.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True, **extra}) as resp:
        assert resp.status_code == 200
        return [line[len("data: ") :] async for line in resp.aiter_lines() if line.startswith("data: ")]


def text_of(chunks: list[dict]) -> str:
    return "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)


async def test_health_reports_model_and_device(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "fake-model", "device": "cpu"}


async def test_stream_follows_openai_chunk_format(client, fake_llm):
    fake_llm.n_pieces = 3
    events = await stream_events(client)

    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert {c["object"] for c in chunks} == {"chat.completion.chunk"}
    assert len({c["id"] for c in chunks}) == 1
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert text_of(chunks) == "w0 w1 w2 "
    assert chunks[-1]["choices"][0]["delta"] == {}
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


async def test_stream_reports_length_when_the_token_cap_is_hit(client, fake_llm):
    fake_llm.n_pieces = 10
    events = await stream_events(client, max_tokens=3)

    chunks = [json.loads(e) for e in events[:-1]]
    assert text_of(chunks) == "w0 w1 w2 "
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


async def test_server_caps_max_tokens_whatever_the_client_asks(client, fake_llm, monkeypatch):
    monkeypatch.setattr(main, "MAX_NEW_TOKENS", 4)
    fake_llm.n_pieces = 50
    events = await stream_events(client, max_tokens=1_000_000)

    assert text_of([json.loads(e) for e in events[:-1]]) == "w0 w1 w2 w3 "


async def test_non_streaming_returns_one_openai_style_json(client):
    resp = await client.post("/v1/chat/completions", json=CHAT)

    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "w0 w1 w2 w3 w4 "}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


async def test_invalid_request_is_rejected_before_any_generation(client, fake_llm):
    resp = await client.post("/v1/chat/completions", json={})

    assert resp.status_code == 422
    assert fake_llm.starts == []
