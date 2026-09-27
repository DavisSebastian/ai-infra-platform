"""Tiny async load generator: fire N requests with C in flight at once, report throughput + latency.

Examples:
  uv run python scripts/loadtest.py /users/1?delay=0.5 -n 500 -c 100
  uv run python scripts/loadtest.py /stream?delay=0.02 -n 50 -c 50 --stream
"""

import argparse
import asyncio
import statistics
import time

import aiohttp


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))]


async def one_request(session: aiohttp.ClientSession, url: str, stream: bool) -> tuple[float, float | None, bool]:
    """Returns (total_seconds, time_to_first_byte_seconds, ok)."""
    start = time.perf_counter()
    ttfb = None
    try:
        async with session.get(url) as resp:
            if stream:
                async for _ in resp.content.iter_any():
                    if ttfb is None:
                        ttfb = time.perf_counter() - start
            else:
                await resp.read()
            ok = resp.status == 200
    except aiohttp.ClientError:
        ok = False
    return time.perf_counter() - start, ttfb, ok


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("-n", "--requests", type=int, default=200)
    parser.add_argument("-c", "--concurrency", type=int, default=50)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--stream", action="store_true", help="read body as a stream and record time-to-first-byte")
    args = parser.parse_args()

    # A semaphore caps how many requests are in flight: the "C" in "N requests at concurrency C".
    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    timeout = aiohttp.ClientTimeout(total=60)
    url = args.base_url + args.path

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        async def bounded():
            async with sem:
                return await one_request(session, url, args.stream)

        wall_start = time.perf_counter()
        results = await asyncio.gather(*(bounded() for _ in range(args.requests)))
        wall = time.perf_counter() - wall_start

    latencies = [r[0] for r in results if r[2]]
    errors = sum(1 for r in results if not r[2])

    print(f"{args.path}  n={args.requests} c={args.concurrency}")
    print(f"  wall time   {wall:8.2f} s")
    print(f"  throughput  {args.requests / wall:8.1f} req/s")
    print(f"  errors      {errors:8d}")
    if latencies:
        print(
            f"  latency     p50 {pct(latencies, 0.50) * 1000:7.0f} ms   "
            f"p95 {pct(latencies, 0.95) * 1000:7.0f} ms   "
            f"p99 {pct(latencies, 0.99) * 1000:7.0f} ms   "
            f"max {max(latencies) * 1000:7.0f} ms"
        )
    ttfbs = [r[1] for r in results if r[1] is not None]
    if ttfbs:
        print(f"  first byte  p50 {pct(ttfbs, 0.50) * 1000:7.0f} ms   mean {statistics.mean(ttfbs) * 1000:7.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
