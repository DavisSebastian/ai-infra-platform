"""A stand-in for the Week 2 model server. No model, no GPU: it just emits chunks on a fixed schedule.

Same contract as Week 2: POST /v1/chat/completions (streaming SSE), GET /health.
Because the timing is known exactly (CHUNKS chunks, CHUNK_DELAY seconds apart), any extra delay measured
through the gateway is the gateway's own overhead. It also counts how many streams ended early, so a test
can prove "the client left -> the upstream stopped".

Run:  uv run uvicorn app.fake_upstream:app --port 9000   (from week03-gateway/)
"""

import asyncio
import json
import os
import time

from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

CHUNKS = int(os.getenv("FAKE_CHUNKS", "20"))
CHUNK_DELAY = float(os.getenv("FAKE_CHUNK_DELAY", "0.02"))  # seconds between chunks (20 ms, like a fast GPU)
FIRST_CHUNK_DELAY = float(os.getenv("FAKE_FIRST_DELAY", "0.0"))  # simulated prompt-processing time

app = FastAPI(title="Fake model server")

stats = {"started": 0, "finished": 0, "cancelled": 0, "active": 0, "max_active": 0}
last_request: dict = {}  # the last chat request received, so tests can see what the gateway actually forwarded


class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int | None = None
    stream: bool = False


def sse(payload: dict | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n"


def chunk(delta: dict, finish: str | None = None) -> str:
    return sse(
        {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "fake",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    )


async def generate(n_chunks: int):
    stats["started"] += 1
    stats["active"] += 1
    stats["max_active"] = max(stats["max_active"], stats["active"])
    try:
        await asyncio.sleep(FIRST_CHUNK_DELAY)
        yield chunk({"role": "assistant", "content": ""})
        for i in range(n_chunks):
            await asyncio.sleep(CHUNK_DELAY)
            yield chunk({"content": f"w{i} "})
        yield chunk({}, "stop")
        yield sse("[DONE]")
        stats["finished"] += 1
    except asyncio.CancelledError:
        # uvicorn cancels this task when the client disconnects.
        stats["cancelled"] += 1
        raise
    finally:
        stats["active"] -= 1


@app.get("/health")
async def health():
    return {"status": "ok", "model": "fake", "device": "none"}


@app.get("/_stats")
async def get_stats():
    return stats


@app.get("/_last_request")
async def get_last_request():
    # json.dumps escapes everything to ASCII by default ("\\ud800"), so even a lone surrogate the gateway forwarded
    # can be returned. FastAPI's default JSON response would fail to encode it as UTF-8.
    return Response(content=json.dumps(last_request), media_type="application/json")


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest):
    last_request.clear()
    last_request.update(body.model_dump())
    n = min(body.max_tokens or CHUNKS, CHUNKS)
    if body.stream:
        return StreamingResponse(
            generate(n),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": "fake",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": " ".join(f"w{i}" for i in range(n))},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": n, "total_tokens": n + 1},
    }
