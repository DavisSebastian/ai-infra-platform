"""Week 1: async serving foundations.

Three versions of the same `/users/{id}` endpoint, so you can *measure* the difference:

  /users/{id}             async def + aiosqlite + asyncio.sleep   -> the correct, non-blocking way
  /threadpool/users/{id}  plain def + sqlite3 + time.sleep        -> "Java style": runs on a bounded thread pool
  /blocking/users/{id}    async def + time.sleep                  -> the bug: freezes the whole event loop

Plus `/stream`, an SSE endpoint that emits a paragraph word-by-word, like an LLM emitting tokens.

Run:  uv run uvicorn app.main:app --loop uvloop --port 8000   (from week01-async-serving/)
"""

import asyncio
import json
import logging
import sqlite3
import time
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.db import DB_PATH, init_db

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Like a Spring @PostConstruct / @PreDestroy pair: open shared resources once, close on shutdown.
    init_db()
    app.state.db = await aiosqlite.connect(DB_PATH)
    app.state.db.row_factory = aiosqlite.Row
    yield
    await app.state.db.close()


app = FastAPI(title="Week 1 - Async Serving", lifespan=lifespan)


# --- 1. The correct async endpoint -------------------------------------------------------------


@app.get("/users/{user_id}")
async def get_user(request: Request, user_id: int, delay: float = Query(0.0, ge=0, le=5)):
    # `delay` simulates a slow downstream call (auth service, feature store, a model server...).
    # `await` hands control back to the event loop, so thousands of requests can wait at once.
    if delay:
        await asyncio.sleep(delay)

    db: aiosqlite.Connection = request.app.state.db
    async with db.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,)) as cursor:
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="user not found")
    return dict(row)


# --- 2. The "Java thread pool" version ---------------------------------------------------------


@app.get("/threadpool/users/{user_id}")
def get_user_threadpool(user_id: int, delay: float = Query(0.0, ge=0, le=5)):
    # A plain `def` handler: FastAPI runs it on a worker thread (default pool size: 40).
    # Nothing breaks, but concurrency is capped by the pool, exactly like a Tomcat thread pool.
    if delay:
        time.sleep(delay)

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,)).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="user not found")
    return dict(row)


# --- 3. The bug: blocking inside async def -----------------------------------------------------


@app.get("/blocking/users/{user_id}")
async def get_user_blocking(request: Request, user_id: int, delay: float = Query(0.0, ge=0, le=5)):
    # DON'T DO THIS. `time.sleep` inside `async def` never yields, so the single event-loop
    # thread is stuck and *every other request* on this server waits behind it.
    if delay:
        time.sleep(delay)

    db: aiosqlite.Connection = request.app.state.db
    async with db.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,)) as cursor:
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="user not found")
    return dict(row)


# --- 4. Streaming with Server-Sent Events ------------------------------------------------------

PARAGRAPH = (
    "Large language models generate text one token at a time. Each new token depends on every "
    "token before it, so the server cannot compute the whole answer in one shot. Streaming sends "
    "each token to the client the moment it exists, which makes a slow model feel fast because "
    "the user sees the first word almost immediately instead of waiting for the full reply."
)


def sse(payload: dict | str) -> str:
    # SSE wire format: a `data:` line, then a blank line marks the end of one event.
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n"


async def fake_token_stream(delay: float):
    words = PARAGRAPH.split()
    sent = 0
    try:
        for i, word in enumerate(words):
            await asyncio.sleep(delay)  # stand-in for "the model is computing the next token"
            yield sse({"index": i, "token": word + " "})
            sent += 1
        yield sse({"finish_reason": "stop", "tokens": sent})
        yield sse("[DONE]")  # same end-of-stream marker the OpenAI API uses
    except asyncio.CancelledError:
        # Client hung up (closed the tab, Ctrl-C'd curl). With a real model this is where you'd
        # tell the engine to stop generating, so you don't burn GPU time on an answer nobody reads.
        log.info("stream cancelled by client after %d/%d tokens", sent, len(words))
        raise


@app.get("/stream")
async def stream(request: Request, delay: float = Query(0.05, ge=0, le=2)):
    return StreamingResponse(
        fake_token_stream(delay),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tells nginx-style proxies not to buffer the stream
        },
    )
