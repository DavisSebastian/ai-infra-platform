"""Week 7: send a set of prompts through the gateway, then save the records it logged as one window (a parquet file).

A "window" is a sample of traffic the drift check treats as a unit. The baseline window defines "normal"; every later
window is compared against it.

    # the gateway must run with DRIFT_LOG=1 (see the Week 7 README for the full setup)
    uv run python week07-drift/scripts/capture.py week07-drift/prompts/normal.txt --split baseline \
        --out week07-drift/data/baseline.parquet

--split baseline / holdout takes one half of the file (the same seeded shuffle every time), so the two halves are
the same KIND of traffic with different prompts. Without --split, every prompt is sent.

Why read the records back from the gateway's stream instead of saving the answers here: the drift check must judge
what the GATEWAY recorded, because in production that is all it will have. Capturing through the same path tests it.
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from pathlib import Path

import aiohttp
import pandas as pd
import redis.asyncio as redis

SEED = 7  # fixed: the baseline/holdout split must be the same on every run and every machine


def load_prompts(path: Path, split: str | None) -> list[str]:
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if split is None:
        return prompts
    shuffled = prompts[:]
    random.Random(SEED).shuffle(shuffled)
    half = len(shuffled) // 2
    return shuffled[:half] if split == "baseline" else shuffled[half:]


async def send(session: aiohttp.ClientSession, url: str, prompt: str, max_tokens: int) -> int:
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": True}
    async with session.post(f"{url}/v1/chat/completions", json=body) as r:
        await r.read()
        return r.status


async def collect(client: redis.Redis, stream: str, since: str, prompts: set[str], expected: int,
                  timeout: float) -> list[dict]:  # fmt: skip
    """The records logged since `since` for these prompts. The gateway's writer is asynchronous, so wait for them."""
    deadline = time.monotonic() + timeout
    while True:
        rows = [json.loads(f[b"data"]) for _, f in await client.xrange(stream, min=since)]
        mine = [r for r in rows if r["prompt"] in prompts]
        if len(mine) >= expected or time.monotonic() > deadline:
            return mine
        await asyncio.sleep(0.2)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompts", type=Path, help="a prompt file: one prompt per line, # for comments")
    parser.add_argument("--split", choices=["baseline", "holdout"], help="send only this half of the file")
    parser.add_argument("--out", type=Path, required=True, help="where to write the window (.parquet)")
    parser.add_argument("--gateway", default=os.getenv("GATEWAY_URL", "http://localhost:8080"))
    parser.add_argument("--redis", default=os.getenv("REDIS_URL", "redis://localhost:6379"))
    parser.add_argument("--stream", default=os.getenv("DRIFT_STREAM", "gateway:requests"))
    parser.add_argument("--concurrency", type=int, default=1, help="the Week 2 server answers one at a time")
    parser.add_argument("--max-tokens", type=int, default=200, help="the card's max_tokens caps this anyway")
    args = parser.parse_args()

    # Each prompt once, on purpose: repeated prompts inflate the embedding check. The same prompt lands in both the
    # classifier's training and test halves, so it memorizes "this one is current" (measured: AUC 0.51 -> 0.68).
    prompts = load_prompts(args.prompts, args.split)
    client = redis.from_url(args.redis)
    since = f"{int(time.time() * 1000)}-0"  # stream IDs start with a millisecond timestamp
    statuses: list[int] = []
    sem = asyncio.Semaphore(args.concurrency)
    started = time.monotonic()

    async def one(session: aiohttp.ClientSession, i: int, prompt: str) -> None:
        async with sem:
            statuses.append(await send(session, args.gateway, prompt, args.max_tokens))
            if len(statuses) % 25 == 0:
                print(f"  {len(statuses)}/{len(prompts)} sent ({time.monotonic() - started:.0f} s)", flush=True)

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(*(one(session, i, p) for i, p in enumerate(prompts)))
    records = await collect(client, args.stream, since, set(prompts), len(prompts), timeout=30)
    await client.aclose()

    window = pd.DataFrame(records)
    window["source"] = args.prompts.name
    window["split"] = args.split or "all"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    window.to_parquet(args.out, index=False)

    ok = window[window["status"] == 200]
    words = ok["answer"].str.split().str.len()
    print(f"\n{args.out}: {len(window)} records ({len(prompts)} prompts sent, statuses {dict(_count(statuses))})")
    print(f"  sides {ok['side'].value_counts().to_dict()}, cards {ok['card'].value_counts(dropna=False).to_dict()}")
    print(f"  answer words: median {words.median():.0f}, range {words.min()}-{words.max()}")
    print(f"  time to first token: median {statistics.median(ok['ttft_ms']):.0f} ms")
    print(f"  took {time.monotonic() - started:.0f} s")
    if len(window) < len(prompts):
        print(f"  WARNING: {len(prompts) - len(window)} records missing (request log dropped or failed?)")


def _count(values: list[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


if __name__ == "__main__":
    asyncio.run(main())
