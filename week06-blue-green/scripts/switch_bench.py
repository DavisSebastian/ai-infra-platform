"""Week 6: what does switching model servers cost the users who are mid-request? Measured, not argued.

Starts two fake model servers (blue, green) and the real gateway on this machine, sends steady streaming traffic,
switches from blue to green partway through, and counts every request that failed.

  uv run python week06-blue-green/scripts/switch_bench.py --mode naive
  uv run python week06-blue-green/scripts/switch_bench.py --mode bluegreen
  uv run python week06-blue-green/scripts/switch_bench.py --mode canary

Modes (all switch at --switch-at seconds):
  naive      the gateway knows one server. Switching = stop it (SIGTERM) and start it again pointed at green,
             i.e. "change UPSTREAM_URL and redeploy"
  bluegreen  the gateway knows both. Switching = POST /admin/traffic {"green": 100}, no restart
  canary     the same, in steps: green 10, then 50, then 100, --step seconds apart

The load is OPEN-LOOP: requests start on a fixed schedule (--rate per second) whether or not earlier ones have
finished, the way independent users arrive. A closed loop (each worker waits for its previous reply) would send
fewer requests during an outage and hide exactly the failures this measures.

A request succeeds only if it gets a 200 AND its stream reaches [DONE]: a stream cut off halfway is a failure.

In CI (exit code 1 when the check fails):
  --mode bluegreen --max-failed 0    a Blue/Green switch must not fail a single request
  --mode naive --min-failed 1        the tool must still SEE an outage: a benchmark that always reports 0 failures
                                     would pass the check above while measuring nothing

Lineage: week01 loadtest (aiohttp) -> week03 loadtest (statuses, rejections) -> this (load with an event in it).
"""

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass

import aiohttp

TOKEN = "bench-token"
BODY = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 20, "stream": True}


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))] if values else float("nan")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- The processes under test ---------------------------------------------------------------------------


def start(module: str, port: int, env: dict[str, str]) -> subprocess.Popen:
    """uvicorn in its own process, as in production: a restart really means a new process and a new socket."""
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port), "--log-level", "error"],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def wait_healthy(session: aiohttp.ClientSession, url: str, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with session.get(f"{url}/health") as r:
                if r.status == 200:
                    return
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(0.05)
    raise SystemExit(f"{url} did not become healthy in {timeout} s")


def gateway_env(blue: str, green: str | None) -> dict[str, str]:
    env = {
        "UPSTREAM_URL": blue,
        "ADMIN_TOKEN": TOKEN,
        "RATE_LIMIT_ENABLED": "0",  # measuring switching, not rate limiting
        "MAX_IN_FLIGHT": "256",  # the fakes take any concurrency; don't let admission turn requests away
        "MAX_QUEUE": "256",
        "REDIS_URL": f"redis://127.0.0.1:{free_port()}",  # nothing there; unused with rate limiting off
    }
    if green:
        env["GREEN_URL"] = green
    return env


# --- The load ---------------------------------------------------------------------------------------------


@dataclass
class Result:
    start: float  # seconds since the run began
    seconds: float  # until the stream ended (or failed)
    ok: bool
    failure: str | None  # why it failed, e.g. "connect refused", "cut off mid-stream", "HTTP 503"
    side: str | None  # which model server answered


async def one_request(session: aiohttp.ClientSession, url: str, t0: float) -> Result:
    start = time.monotonic()
    side, done = None, False
    try:
        async with session.post(f"{url}/v1/chat/completions", json=BODY) as r:
            if r.status != 200:
                await r.read()
                return Result(start - t0, time.monotonic() - start, False, f"HTTP {r.status}", None)
            async for line in r.content:
                text = line.decode().strip()
                if text == "data: [DONE]":
                    done = True
                elif text.startswith("data: {") and side is None:
                    side = json.loads(text.removeprefix("data: "))["model"]
    except aiohttp.ClientConnectorError:
        return Result(start - t0, time.monotonic() - start, False, "connect refused", side)
    except (TimeoutError, aiohttp.ClientError) as e:
        return Result(start - t0, time.monotonic() - start, False, f"cut off ({type(e).__name__})", side)
    failure = None if done else "cut off mid-stream"
    return Result(start - t0, time.monotonic() - start, done, failure, side)


async def load(session: aiohttp.ClientSession, url_of: dict, rate: float, duration: float, t0: float) -> list:
    """Start one request every 1/rate seconds for `duration` seconds. url_of["gateway"] can change mid-run."""
    tasks = []
    for i in range(int(rate * duration)):
        await asyncio.sleep(max(0.0, t0 + i / rate - time.monotonic()))
        tasks.append(asyncio.create_task(one_request(session, url_of["gateway"], t0)))
    return await asyncio.gather(*tasks)


# --- The switch -------------------------------------------------------------------------------------------


async def switch(mode: str, session, procs: dict, urls: dict, ports: dict, args, t0: float) -> list[str]:
    """Perform the switch at args.switch_at; return a timeline of what happened, for the report."""
    timeline = []

    def note(what: str) -> None:
        timeline.append(f"{time.monotonic() - t0:6.2f} s  {what}")

    await asyncio.sleep(max(0.0, t0 + args.switch_at - time.monotonic()))
    if mode == "naive":
        note("SIGTERM the gateway (blue only)")
        procs["gateway"].send_signal(signal.SIGTERM)
        await asyncio.to_thread(procs["gateway"].wait)
        note("old gateway exited; starting a new one pointed at green")
        procs["gateway"] = start("gateway.main:app", ports["gateway"], gateway_env(urls["green"], None))
        await wait_healthy(session, urls["gateway"])
        note("new gateway healthy")
        return timeline

    steps = [100] if mode == "bluegreen" else [10, 50, 100]
    for i, green in enumerate(steps):
        if i:
            await asyncio.sleep(args.step)
        async with session.post(f"{urls['gateway']}/admin/traffic", json={"green": green},
                                headers={"X-Admin-Token": TOKEN}) as r:  # fmt: skip
            note(f"POST /admin/traffic green={green} -> HTTP {r.status}")
    return timeline


# --- Report -----------------------------------------------------------------------------------------------


def report(mode: str, results: list[Result], timeline: list[str], args) -> dict:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    summary = {
        "mode": mode,
        "requests": len(results),
        "ok": len(ok),
        "failed": len(failed),
        "failed_pct": round(100 * len(failed) / len(results), 2),
        "failures": dict(Counter(r.failure for r in failed)),
        "served_by": dict(Counter(r.side for r in ok)),
        "p50_ms": round(1000 * pct([r.seconds for r in ok], 0.50), 1),
        "p99_ms": round(1000 * pct([r.seconds for r in ok], 0.99), 1),
        # the window when users notice: first failure to last failure
        "failure_window_s": round(max(r.start for r in failed) - min(r.start for r in failed), 2) if failed else 0,
        "rate": args.rate,
        "duration_s": args.duration,
    }
    print(f"\n== {mode}: {args.rate:g} req/s for {args.duration:g} s, switch at {args.switch_at:g} s")
    for line in timeline:
        print("  " + line)
    print(f"  requests   {summary['requests']}  ok {summary['ok']}  failed {summary['failed']} "
          f"({summary['failed_pct']}%)")  # fmt: skip
    if failed:
        print(f"  failures   {summary['failures']}  over {summary['failure_window_s']} s")
    print(f"  served by  {summary['served_by']}")
    print(f"  latency    p50 {summary['p50_ms']} ms   p99 {summary['p99_ms']} ms   (successful streams, end to end)")
    if args.json:
        print(json.dumps(summary))
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["naive", "bluegreen", "canary"], required=True)
    parser.add_argument("--rate", type=float, default=50, help="requests started per second (default 50)")
    parser.add_argument("--duration", type=float, default=10, help="seconds of load (default 10)")
    parser.add_argument("--switch-at", type=float, default=4, help="seconds into the run (default 4)")
    parser.add_argument("--step", type=float, default=2, help="canary: seconds between steps (default 2)")
    parser.add_argument("--json", action="store_true", help="also print one JSON summary line")
    parser.add_argument("--max-failed", type=int, help="exit 1 if more requests than this failed")
    parser.add_argument("--min-failed", type=int, help="exit 1 if fewer requests than this failed")
    args = parser.parse_args()

    ports = {name: free_port() for name in ("blue", "green", "gateway")}
    urls = {name: f"http://127.0.0.1:{port}" for name, port in ports.items()}
    procs = {name: start("gateway.fake_upstream:app", ports[name], {"FAKE_NAME": name}) for name in ("blue", "green")}
    green_for_gateway = None if args.mode == "naive" else urls["green"]
    procs["gateway"] = start("gateway.main:app", ports["gateway"], gateway_env(urls["blue"], green_for_gateway))
    try:
        connector = aiohttp.TCPConnector(limit=0)  # no client-side cap: the client must not be the bottleneck
        async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=30)) as session:
            for name in ("blue", "green", "gateway"):
                await wait_healthy(session, urls[name])
            t0 = time.monotonic() + 0.2
            results, timeline = await asyncio.gather(
                load(session, urls, args.rate, args.duration, t0),
                switch(args.mode, session, procs, urls, ports, args, t0),
            )
        summary = report(args.mode, results, timeline, args)
    finally:
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            p.wait(timeout=10)

    failed = summary["failed"]
    if args.max_failed is not None and failed > args.max_failed:
        raise SystemExit(f"FAIL: {failed} requests failed, at most {args.max_failed} allowed")
    if args.min_failed is not None and failed < args.min_failed:
        raise SystemExit(f"FAIL: only {failed} requests failed, expected at least {args.min_failed}")


if __name__ == "__main__":
    asyncio.run(main())
