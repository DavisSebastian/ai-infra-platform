"""Week 2: the Thin Slice. A real model behind an OpenAI-style API.

POST /v1/chat/completions   {"messages": [...], "max_tokens": 200, "stream": true|false}
GET  /health

Generation never runs on the event loop thread, so /health stays instant while the model works.
Only one generation runs at a time (see `gen_lock`); everyone else waits in line.

Run:  uv run uvicorn app.main:app --port 8000   (from week02-thin-slice/)
"""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from app.model import LLM

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = os.getenv("DEVICE")  # None means "pick the best available" (resolved in load_llm)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))

log = logging.getLogger("uvicorn.error")


def load_llm() -> "LLM":
    # Imported here, not at the top of the file, so importing this module does not pull in torch.
    # That keeps tests and CI light, and gives tests one place to swap in a fake model.
    from app.model import LLM, pick_device

    return LLM(MODEL_ID, DEVICE or pick_device())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded once, before the server accepts traffic. Like a heavy singleton @Bean.
    t0 = time.perf_counter()
    app.state.llm = load_llm()
    # One model, one generation at a time. Running two at once on one GPU gives no speedup
    # with plain HF Transformers and risks running out of memory. (vLLM fixes this in Month 3.)
    app.state.gen_lock = asyncio.Lock()
    log.info("loaded %s on %s in %.1fs", MODEL_ID, app.state.llm.device, time.perf_counter() - t0)
    yield


app = FastAPI(title="Week 2 - Thin Slice", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int | None = None
    stream: bool = False


@app.get("/health")
async def health(request: Request):
    llm: LLM = request.app.state.llm
    return {"status": "ok", "model": llm.model_id, "device": llm.device}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatRequest):
    llm: LLM = request.app.state.llm
    lock: asyncio.Lock = request.app.state.gen_lock
    max_tokens = min(body.max_tokens or MAX_NEW_TOKENS, MAX_NEW_TOKENS)
    messages = [m.model_dump() for m in body.messages]
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if body.stream:
        return StreamingResponse(
            stream_chunks(llm, lock, messages, max_tokens, completion_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with lock:
        # The fix from step 2: the slow call runs on a worker thread; the event loop stays free.
        result = await asyncio.to_thread(llm.generate, messages, max_tokens)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": llm.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": finish_reason(result.completion_tokens, max_tokens),
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }


def finish_reason(completion_tokens: int, max_tokens: int) -> str:
    return "length" if completion_tokens >= max_tokens else "stop"


def sse(payload: dict | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n"


async def stream_chunks(llm: "LLM", lock: asyncio.Lock, messages: list[dict], max_tokens: int, completion_id: str):
    created = int(time.time())

    def chunk(delta: dict, finish: str | None = None) -> str:
        # OpenAI streaming shape: each event carries a small `delta` (the new piece of text).
        return sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": llm.model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )

    async with lock:
        stream = llm.start_stream(messages, max_tokens)
        try:
            yield chunk({"role": "assistant", "content": ""})
            # `async for` awaits each piece, so the event loop serves other requests in between.
            async for piece in stream.text:
                if piece:
                    yield chunk({"content": piece})
        finally:
            # Runs on a normal finish AND when the client disconnects (the task is cancelled mid-loop).
            # 1. Tell the generation thread to stop (no-op if it already finished).
            # 2. Wait until it really has, so the lock is never released while the thread still runs.
            # The shield matters: during cancellation, a plain `await` here would be cancelled again at once.
            stream.cancel()
            with anyio.CancelScope(shield=True):
                await stream.wait()

    yield chunk({}, finish_reason(stream.completion_tokens, max_tokens))
    yield sse("[DONE]")
