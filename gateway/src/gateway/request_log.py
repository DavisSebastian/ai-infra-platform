"""Week 7: one record per finished request, written to a Redis stream for the drift check to read later.

The gateway only RECORDS. It never computes drift: one request says nothing about a distribution, and embeddings
and statistics are far too slow for the request path. A separate job (week07-drift/scripts/drift_check.py) reads
the stream every few minutes and does the analysis.

    request ends -> RequestLog.record(...)    instant: puts raw bytes on an in-memory queue, never waits
                    background task           takes them off the queue, extracts prompt and answer, XADDs to Redis

Two rules, both from Week 3's rate limiter:
  - never slow a request down: record() only does queue.put_nowait(). The parsing and the Redis call happen in the
    background task, after the response is finished.
  - fail OPEN: if the queue is full or Redis is down, the record is dropped and counted. The request already
    succeeded; monitoring must never take serving down with it.

Java analogy: Logback's AsyncAppender with neverBlock=true. A bounded queue in front of a slow appender, and when the
queue is full, events are discarded rather than making the caller wait.

Privacy: records hold the prompt and answer text, because the drift check embeds prompts and measures answers. It is
off by default (DRIFT_LOG=1 turns it on), and the stream keeps only the newest STREAM_MAXLEN records. A real product
would also need consent and a retention policy.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import redis.asyncio as redis

log = logging.getLogger("uvicorn.error")


@dataclass
class Finished:
    """What the request path hands over: raw bytes and timings, nothing parsed yet (parsing is the writer's job)."""

    side: str
    card: str | None  # the card version this request was served with, or None with the registry off
    status: int
    body: bytes  # the request body as forwarded (after cleaning and the card)
    answer: bytes  # the response bytes as relayed, possibly cut at the capture limit
    ttft_ms: float | None  # gateway-side: upstream opened -> first chunk relayed
    ms: float  # upstream opened -> response finished
    completed: bool  # False if the client left mid-stream


def prompt_of(body: bytes) -> str:
    """The last user message of a chat request: what the user asked. "" if the body isn't a chat request."""
    try:
        messages = json.loads(body).get("messages")
    except (ValueError, UnicodeDecodeError, AttributeError):
        return ""
    if not isinstance(messages, list):
        return ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            return _text(m.get("content"))
    return ""


def answer_of(raw: bytes) -> str:
    """The text the model answered, from a streamed (SSE) or a plain JSON chat response. "" if there's none."""
    text = raw.decode("utf-8", errors="replace")
    if text.lstrip().startswith("data:"):
        parts = []
        for line in text.splitlines():
            if not line.startswith("data: {"):
                continue  # blank lines, "data: [DONE]", or a line cut off at the capture limit
            try:
                choice = json.loads(line.removeprefix("data: "))["choices"][0]
                parts.append(choice.get("delta", {}).get("content") or "")
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                continue
        return "".join(parts)
    try:
        return _text(json.loads(text)["choices"][0]["message"]["content"])
    except (ValueError, KeyError, IndexError, TypeError):
        return ""


def _text(content) -> str:
    """Message content is a string, or a list of parts like [{"type": "text", "text": "..."}]."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class RequestLog:
    def __init__(self, client: redis.Redis, stream: str, maxlen: int, queue_size: int = 1000):
        self.client = client
        self.stream = stream
        self.maxlen = maxlen
        self.queue: asyncio.Queue[Finished] = asyncio.Queue(maxsize=queue_size)
        self.counts = {"written": 0, "dropped": 0, "failed": 0}  # failed = Redis errors; dropped = queue full
        self._last_warning = 0.0

    def record(self, finished: Finished) -> None:
        """Called by the request path. Never blocks and never raises."""
        try:
            self.queue.put_nowait(finished)
        except asyncio.QueueFull:
            self.counts["dropped"] += 1

    async def run(self) -> None:
        """The background writer: runs for the life of the gateway."""
        while True:
            finished = await self.queue.get()
            try:
                await self.write(finished)
            except (redis.RedisError, OSError, TimeoutError) as e:
                self.counts["failed"] += 1
                if time.monotonic() - self._last_warning > 10:  # a dead Redis must not flood the log
                    self._last_warning = time.monotonic()
                    log.warning("request log: could not write to Redis (count=%s): %s", self.counts["failed"], e)
            finally:
                self.queue.task_done()

    async def write(self, f: Finished) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "side": f.side,
            "card": f.card,
            "status": f.status,
            "completed": f.completed,
            "ttft_ms": None if f.ttft_ms is None else round(f.ttft_ms, 1),
            "ms": round(f.ms, 1),
            "prompt": prompt_of(f.body),
            "answer": answer_of(f.answer) if f.status == 200 else "",
        }
        # One JSON field per entry: readers get the whole record in one parse. MAXLEN ~ trims old entries in bulk
        # (the ~ lets Redis trim at a convenient boundary, which is much cheaper than an exact cap).
        await self.client.xadd(self.stream, {"data": json.dumps(record)}, maxlen=self.maxlen, approximate=True)
        self.counts["written"] += 1
