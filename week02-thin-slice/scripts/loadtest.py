"""Load generator for streaming (SSE) endpoints: TTFT, inter-chunk latency (ITL) and tokens/s.

Extends week01-async-serving/scripts/loadtest.py (aiohttp; httpx was the bottleneck there).
New: POST bodies, per-event timestamps, TTFT = first event that carries text, ITL = gaps between text events.

Verify the tool first against Week 1's known stream (expect TTFT ~20 ms and ITL ~20 ms):
  uv run python scripts/loadtest.py "/stream?delay=0.02" -n 5 -c 1 --base-url http://127.0.0.1:8000

Then measure Week 2:
  uv run python scripts/loadtest.py /v1/chat/completions --prompt "Write a long story about a robot." \
      --max-tokens 100 -n 4 -c 4 --base-url http://127.0.0.1:8001

Caveat: the server streams pieces of text (usually a word), not guaranteed single tokens, so "ITL" here is
really inter-chunk latency. It approximates ITL closely enough to see queueing and batching effects.
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import aiohttp


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))]


def extract_text(event: dict) -> str | None:
    """Text carried by one SSE event, or None if it carries none (role, finish and [DONE] events)."""
    if "choices" in event:  # OpenAI chunk: {"choices": [{"delta": {"content": "..."}}]}
        return event["choices"][0].get("delta", {}).get("content") or None
    return event.get("token") or None  # Week 1 /stream: {"index": 3, "token": "word "}


@dataclass
class Result:
    ok: bool
    start: float
    total: float
    stamps: list[float] = field(default_factory=list)  # perf_counter() at every text event

    @property
    def ttft(self) -> float | None:
        return self.stamps[0] - self.start if self.stamps else None

    @property
    def gaps(self) -> list[float]:
        return [b - a for a, b in zip(self.stamps, self.stamps[1:])]


async def one_request(session: aiohttp.ClientSession, url: str, body: dict | None) -> Result:
    start = time.perf_counter()
    stamps: list[float] = []
    ok = False
    try:
        request = session.post(url, json=body) if body else session.get(url)
        async with request as resp:
            ok = resp.status == 200
            if ok:
                # Line by line, not iter_any(): network chunks can hold half an event or several events.
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    if extract_text(json.loads(payload)):
                        stamps.append(time.perf_counter())
    except (aiohttp.ClientError, TimeoutError):
        ok = False
    return Result(ok=ok, start=start, total=time.perf_counter() - start, stamps=stamps)


def ms(seconds: float) -> str:
    return f"{seconds * 1000:7.0f} ms"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("-n", "--requests", type=int, default=4)
    parser.add_argument("-c", "--concurrency", type=int, default=4)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--prompt", help="send a POST chat request with this prompt (omit for a plain GET stream)")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1, help="throwaway requests before measuring")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    body = None
    if args.prompt:
        body = {"messages": [{"role": "user", "content": args.prompt}], "max_tokens": args.max_tokens, "stream": True}

    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    url = args.base_url + args.path

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # The first generation is much slower than later ones (GPU warm-up), so don't measure it.
        for _ in range(args.warmup):
            await one_request(session, url, body)

        async def bounded() -> Result:
            async with sem:
                return await one_request(session, url, body)

        wall_start = time.perf_counter()
        results = await asyncio.gather(*(bounded() for _ in range(args.requests)))
        wall = time.perf_counter() - wall_start

    good = [r for r in results if r.ok]
    print(f"{args.path}  n={args.requests} c={args.concurrency}")
    print(f"  wall time   {wall:8.2f} s")
    print(f"  errors      {len(results) - len(good):8d}")

    ttfts = [r.ttft for r in good if r.ttft is not None]
    if ttfts:
        print(f"  TTFT        p50 {ms(pct(ttfts, 0.50))}   p95 {ms(pct(ttfts, 0.95))}   max {ms(max(ttfts))}")

    gaps = [g for r in good for g in r.gaps]
    if gaps:
        print(
            f"  ITL         p50 {ms(pct(gaps, 0.50))}   p95 {ms(pct(gaps, 0.95))}   "
            f"p99 {ms(pct(gaps, 0.99))}   mean {ms(statistics.mean(gaps))}"
        )

    rates = [(len(r.stamps) - 1) / (r.stamps[-1] - r.stamps[0]) for r in good if len(r.stamps) > 1]
    chunks = sum(len(r.stamps) for r in good)
    if rates:
        print(f"  per-user    {statistics.median(rates):8.1f} chunks/s (median, while streaming)")
        print(f"  total       {chunks / wall:8.1f} chunks/s across all users ({chunks / len(good):.0f} chunks/request)")


if __name__ == "__main__":
    asyncio.run(main())
