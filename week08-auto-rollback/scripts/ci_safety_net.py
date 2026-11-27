"""Week 8, CI: exercise the safety net end to end. A rollback path that never runs is one that fails when needed.

    uv run python week08-auto-rollback/scripts/ci_safety_net.py        (needs Redis on localhost:6379)

Starts two fake model servers (blue answers in 20 words, green in 3: a "bad deploy"), the real gateway with the
request log on, steady open-loop traffic (traffic.py), deploys green at 50%, then runs the real controller
(stand-in embedding, no torch) until it acts. Exit code 1 unless:
  - the controller rolled green back on its own, before the timeout
  - not a single request failed, before, during or after the rollback
  - after the rollback, the gateway sends green 0% and the stream shows no new green requests

Fake servers make it fast (the whole run takes well under a minute). The real-model measurement is in the README.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "week07-drift" / "scripts"))
from controller import Controller, gateway_reader, stream_reader, traffic_setter  # noqa: E402
from drift import compare, hashing  # noqa: E402
from verdict import decide  # noqa: E402

TOKEN, STREAM, MIN_RECORDS = "ci-token", f"ci:safety-net:{os.getpid()}", 30


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(module: str, port: int, env: dict) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port), "--log-level", "error"]
    return subprocess.Popen(cmd, env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_healthy(url: str) -> None:
    for _ in range(200):
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise SystemExit(f"{url} never became healthy")


def check(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        raise SystemExit(1)


def main() -> None:
    ports = {name: free_port() for name in ("blue", "green", "gateway")}
    url = f"http://127.0.0.1:{ports['gateway']}"
    procs = [
        start("gateway.fake_upstream:app", ports["blue"], {"FAKE_NAME": "blue", "FAKE_CHUNKS": "20"}),
        start("gateway.fake_upstream:app", ports["green"], {"FAKE_NAME": "green", "FAKE_CHUNKS": "3"}),
    ]
    procs.append(start("gateway.main:app", ports["gateway"], {
        "UPSTREAM_URL": f"http://127.0.0.1:{ports['blue']}", "GREEN_URL": f"http://127.0.0.1:{ports['green']}",
        "TRAFFIC_GREEN": "0", "ADMIN_TOKEN": TOKEN, "DRIFT_LOG": "1", "DRIFT_STREAM": STREAM,
        "RATE_LIMIT_ENABLED": "0", "MAX_IN_FLIGHT": "64", "MAX_QUEUE": "64",
    }))  # fmt: skip
    out = Path(os.getenv("RUNNER_TEMP", "/tmp")) / f"safety-net-traffic-{os.getpid()}.jsonl"
    try:
        for p in ("blue", "green", "gateway"):
            wait_healthy(f"http://127.0.0.1:{ports[p]}")
        traffic = subprocess.Popen([sys.executable, str(HERE / "traffic.py"), "--gateway", url, "--rate", "20",
                                    "--duration", "30", "--out", str(out)])  # fmt: skip
        time.sleep(1)
        traffic_setter(url, TOKEN)(50)  # the deploy
        print("deployed green (3-word answers) at 50%; controller watching", flush=True)

        baseline = pd.read_parquet(ROOT / "week07-drift" / "data" / "baseline.parquet")
        embed = hashing()

        def judge(green, blue):
            vs_control = compare(blue, green, embed) if len(blue) >= MIN_RECORDS else None
            return decide(compare(baseline, green, embed), vs_control, MIN_RECORDS)

        pulled: set = set()
        controller = Controller(gateway_reader(url), stream_reader("redis://localhost:6379", STREAM), judge,
                                traffic_setter(url, TOKEN), pulled.__contains__,
                                lambda canary, verdict, action: pulled.add(canary.green_card),
                                window=100, min_records=MIN_RECORDS)  # fmt: skip
        deadline, rolled_back = time.monotonic() + 25, False
        while time.monotonic() < deadline and not rolled_back:
            tick = controller.tick()
            print(f"  tick: green {tick.green}%, records {tick.n_green}/{tick.n_blue}, verdict {tick.verdict} "
                  f"-> {tick.action} {tick.target if tick.target is not None else ''}", flush=True)  # fmt: skip
            rolled_back = tick.target == 0
            time.sleep(1)
        check(rolled_back, "the controller rolled green back on its own")
        rollback_at = time.time()

        traffic.wait(timeout=60)
        users = pd.read_json(out, lines=True)
        failed = users[(users["status"] != 200) | (~users["completed"])]
        check(len(failed) == 0, f"no request failed ({len(users)} sent, {len(failed)} failed)")
        weight = httpx.get(f"{url}/health").json()["traffic"]["green"]["weight"]
        check(weight == 0, f"the gateway sends green {weight}%")
        log = stream_reader("redis://localhost:6379", STREAM)("green", None, 100_000)
        late = pd.to_datetime(log["ts"]).max().timestamp() - rollback_at if len(log) else -1
        check(late < 1, "no new green requests after the rollback (in-flight ones were allowed to finish)")
    finally:
        for p in procs:
            p.send_signal(signal.SIGTERM)
        for p in procs:
            p.wait(timeout=10)
        import redis

        redis.from_url("redis://localhost:6379").delete(STREAM)


if __name__ == "__main__":
    main()
