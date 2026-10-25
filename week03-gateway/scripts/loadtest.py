"""Load generator for streaming (SSE) endpoints: TTFT, inter-chunk latency (ITL), tokens/s, and rejections.

Lineage: week01 (aiohttp; httpx was the bottleneck) -> week02 (TTFT, ITL) -> this copy (week03).
New in Week 3, because a gateway can say no:
  - every HTTP status is recorded; 429 (rate limited) and 503 (overloaded) are counted as REJECTED, not errors
  - time-to-reject: how fast a "no" arrives (the point of failing fast)
  - TTFT/ITL are computed over ACCEPTED requests only, so fast rejections can't make latency look better
  - --keys N spreads requests over N API keys (X-API-Key), so per-client rate limits see N clients, not one IP
  - --json prints one machine-readable summary line (for results tables)

Run from week03-gateway/. Verify the tool first against a case with a known answer, e.g. the gateway with
MAX_IN_FLIGHT=1 MAX_QUEUE=2 in front of the fake upstream: 6 users at once must give 3 accepted, 3 rejected (503).
  uv run python scripts/loadtest.py /v1/chat/completions --prompt hi -n 6 -c 6 --base-url http://127.0.0.1:8080

Week 2's measurement shape (all users arrive at once, n = c, 100 tokens), bare server vs through the gateway:
  uv run python scripts/loadtest.py /v1/chat/completions --prompt "Write a long story about a robot." \
      --max-tokens 100 -n 16 -c 16 --base-url http://127.0.0.1:8080

Caveat: the server streams pieces of text (usually a word), not guaranteed single tokens, so "ITL" here is
really inter-chunk latency. It approximates ITL closely enough to see queueing and batching effects.
"""

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
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


REJECT_STATUSES = {429, 503}


@dataclass
class Result:
    status: int  # HTTP status, or 0 if the request failed before any response (connection error, timeout)
    start: float
    total: float
    stamps: list[float] = field(default_factory=list)  # perf_counter() at every text event

    @property
    def ok(self) -> bool:
        return self.status == 200

    @property
    def rejected(self) -> bool:
        return self.status in REJECT_STATUSES

    @property
    def ttft(self) -> float | None:
        return self.stamps[0] - self.start if self.stamps else None

    @property
    def gaps(self) -> list[float]:
        return [b - a for a, b in zip(self.stamps, self.stamps[1:])]


async def one_request(
    session: aiohttp.ClientSession, url: str, body: dict | None, headers: dict | None = None
) -> Result:
    start = time.perf_counter()
    stamps: list[float] = []
    status = 0
    try:
        request = session.post(url, json=body, headers=headers) if body else session.get(url, headers=headers)
        async with request as resp:
            status = resp.status
            if status != 200:
                await resp.read()  # drain the small error body so the connection can be reused
            else:
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
        status = 0
    return Result(status=status, start=start, total=time.perf_counter() - start, stamps=stamps)


def ms(seconds: float) -> str:
    return f"{seconds * 1000:7.0f} ms"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("-n", "--requests", type=int, default=4)
    parser.add_argument("-c", "--concurrency", type=int, default=4)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")  # the gateway
    parser.add_argument("--prompt", help="send a POST chat request with this prompt (omit for a plain GET stream)")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1, help="throwaway requests before measuring")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--keys", type=int, default=0, help="spread requests over N API keys (0 = send no key)")
    parser.add_argument("--json", action="store_true", help="also print a one-line JSON summary")
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
            await one_request(session, url, body, {"X-API-Key": "loadtest-warmup"} if args.keys else None)

        async def bounded(i: int) -> Result:
            headers = {"X-API-Key": f"loadtest-{i % args.keys}"} if args.keys else None
            async with sem:
                return await one_request(session, url, body, headers)

        wall_start = time.perf_counter()
        results = await asyncio.gather(*(bounded(i) for i in range(args.requests)))
        wall = time.perf_counter() - wall_start

    good = [r for r in results if r.ok]  # accepted: every latency number below is over these only
    rejected = [r for r in results if r.rejected]
    errors = [r for r in results if not r.ok and not r.rejected]
    statuses = dict(sorted(Counter(r.status for r in results).items()))

    print(f"{args.path}  n={args.requests} c={args.concurrency}" + (f" keys={args.keys}" if args.keys else ""))
    print(f"  wall time   {wall:8.2f} s")
    print(f"  statuses    {statuses}")
    print(f"  accepted    {len(good):8d}   rejected {len(rejected)} (429/503)   errors {len(errors)}")
    rejects = [r.total for r in rejected]
    if rejects:
        print(f"  reject in   p50 {ms(pct(rejects, 0.50))}   p95 {ms(pct(rejects, 0.95))}   max {ms(max(rejects))}")

    ttfts = [r.ttft for r in good if r.ttft is not None]
    if ttfts:
        print(
            f"  TTFT        p50 {ms(pct(ttfts, 0.50))}   p95 {ms(pct(ttfts, 0.95))}   max {ms(max(ttfts))}  (accepted)"
        )

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

    if args.json:
        summary = {
            "path": args.path,
            "n": args.requests,
            "c": args.concurrency,
            "statuses": statuses,
            "accepted": len(good),
            "rejected": len(rejected),
            "errors": len(errors),
            "reject_p50_ms": round(pct(rejects, 0.5) * 1000) if rejects else None,
            "reject_max_ms": round(max(rejects) * 1000) if rejects else None,
            "ttft_p50_ms": round(pct(ttfts, 0.5) * 1000) if ttfts else None,
            "ttft_p95_ms": round(pct(ttfts, 0.95) * 1000) if ttfts else None,
            "ttft_max_ms": round(max(ttfts) * 1000) if ttfts else None,
            "wall_s": round(wall, 2),
        }
        print(json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(main())
