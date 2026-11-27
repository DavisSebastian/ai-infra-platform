"""Week 8: steady user traffic for the rollback measurement. Open-loop, like Week 6's switch benchmark.

    uv run python week08-auto-rollback/scripts/traffic.py --rate 2 --duration 600 --out results/run-traffic.jsonl

A request starts every 1/rate seconds whether or not earlier ones finished, the way independent users arrive, and
cycles through week07-drift/prompts/normal.txt in a seeded random order. Each request's outcome (start time, status,
whether its stream completed) is appended to --out, so failed requests during a rollback can be counted afterwards.
Which side served each request is in the gateway's own request log.
"""

import argparse
import asyncio
import itertools
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent.parent.parent


async def one(session: aiohttp.ClientSession, url: str, prompt: str) -> dict:
    started = datetime.now(UTC).isoformat(timespec="milliseconds")
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 200, "stream": True}
    try:
        async with session.post(f"{url}/v1/chat/completions", json=body) as r:
            raw = await r.read()
            return {"ts": started, "status": r.status, "completed": b"data: [DONE]" in raw, "error": None}
    except (aiohttp.ClientError, TimeoutError) as e:
        return {"ts": started, "status": None, "completed": False, "error": type(e).__name__}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gateway", default="http://localhost:8080")
    parser.add_argument("--rate", type=float, default=2, help="requests started per second")
    parser.add_argument("--duration", type=float, default=600, help="seconds")
    parser.add_argument("--prompts", type=Path, default=ROOT / "week07-drift" / "prompts" / "normal.txt")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prompts = [p.strip() for p in args.prompts.read_text().splitlines() if p.strip() and not p.startswith("#")]
    random.Random(8).shuffle(prompts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = args.out.open("w")
    pending: set[asyncio.Task] = set()
    t0 = time.monotonic()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        for i, prompt in enumerate(itertools.cycle(prompts)):
            due = t0 + i / args.rate
            if due - t0 > args.duration:
                break
            await asyncio.sleep(max(0.0, due - time.monotonic()))
            task = asyncio.create_task(one(session, args.gateway, prompt))
            task.add_done_callback(lambda t: (out.write(json.dumps(t.result()) + "\n"), out.flush()))
            pending.add(task)
            task.add_done_callback(pending.discard)
        await asyncio.gather(*pending)
    out.close()
    rows = [json.loads(line) for line in args.out.read_text().splitlines()]
    failed = [r for r in rows if r["status"] != 200 or not r["completed"]]
    print(f"{len(rows)} requests, {len(failed)} failed")


if __name__ == "__main__":
    asyncio.run(main())
