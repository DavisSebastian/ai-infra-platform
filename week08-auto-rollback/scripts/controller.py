"""Week 8: the canary controller. Watches green's live traffic and rolls it back on its own when drift says so.

    uv run python week08-auto-rollback/scripts/controller.py --interval 10
    uv run python week08-auto-rollback/scripts/controller.py --interval 10 --promote     # also step 10->25->50->100

Every --interval seconds, one tick:
    read the gateway's /health          which card each side runs, and green's share of traffic
    read green's records                from the gateway's Redis stream (Week 7), for green's CURRENT card only
    judge them                          Week 7's compare() + decide(): green vs the baseline, and vs blue
    next_action()                       policy.py: hold / set traffic / done
    act                                 POST /admin/traffic (Week 6's atomic switch); on a rollback, write the
                                        reason and the evidence to MLflow

Design:
  - A separate process, not gateway code (control plane vs data plane). The gateway stays fast and needs no torch.
    If this process dies, the gateway keeps serving with the current split ("fail static").
  - Level-triggered and stateless, like a Kubernetes controller: each tick decides from the current state of the
    world, not from memory of earlier ticks. "Green's version was already rolled back" is a tag on that version in
    MLflow, so a restarted controller neither forgets a rollback nor repeats it. Without a registry (CI), it falls
    back to remembering in memory.
  - Only green's CURRENT card counts: records from before a deploy describe a different version.
  - It never acts on data it couldn't read: an unreachable gateway or Redis means hold (verdict None).
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import redis

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "week07-drift" / "scripts"))
from policy import DONE, SET_TRAFFIC, Action, next_action  # noqa: E402
from verdict import INSUFFICIENT, MIN_RECORDS, Verdict, decide  # noqa: E402

GREEN, BLUE = "green", "blue"


@dataclass(frozen=True)
class Canary:
    """What the gateway says right now."""

    green: int  # green's share of traffic, percent
    green_card: str | None  # the card version green serves (None: registry off)
    blue_card: str | None
    since: str = ""  # when the split last changed: each canary step is judged only on traffic after it


@dataclass(frozen=True)
class Tick:
    """One check, for the log and the measurement."""

    ts: str
    green: int | None
    green_card: str | None
    n_green: int
    n_blue: int
    verdict: str | None
    reasons: tuple[str, ...]
    action: str
    target: int | None
    why: str


class Controller:
    """One tick at a time. Every dependency is passed in, so tests can run it with fakes (no network, no model)."""

    def __init__(
        self,
        read_canary: Callable[[], Canary],
        read_records: Callable[[str, str | None, int, str], pd.DataFrame],
        judge: Callable[[pd.DataFrame, pd.DataFrame], Verdict],
        set_traffic: Callable[[int], None],
        is_rolled_back: Callable[[str | None], bool],
        record_rollback: Callable[[Canary, Verdict, Action], None],
        promote: bool = False,
        window: int = 150,
        min_records: int = MIN_RECORDS,
    ):
        self.read_canary, self.read_records, self.judge = read_canary, read_records, judge
        self.set_traffic, self.is_rolled_back, self.record_rollback = set_traffic, is_rolled_back, record_rollback
        self.promote, self.window, self.min_records = promote, window, min_records

    def tick(self) -> Tick:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        try:
            canary = self.read_canary()
        except (httpx.HTTPError, KeyError, ValueError) as e:  # gateway down, or Blue/Green off
            action = Action("hold", None, f"can't read the gateway: {e}")
            return Tick(now, None, None, 0, 0, None, (), action.kind, None, action.reason)

        rolled_back = self.is_rolled_back(canary.green_card)
        verdict: Verdict | None = None
        n_green = n_blue = 0
        if canary.green > 0 and not rolled_back:  # otherwise there's nothing to judge, so don't spend the time
            try:
                green = self.read_records(GREEN, canary.green_card, self.window, canary.since)
                n_green = len(green)
                if n_green < self.min_records:  # cheap early answer: no embeddings, no statistics
                    verdict = Verdict(INSUFFICIENT, (f"green has {n_green} records; need {self.min_records}",))
                else:
                    blue = self.read_records(BLUE, canary.blue_card, self.window, canary.since)
                    blue = blue[blue["ts"] >= green["ts"].min()] if len(blue) else blue  # the same period only
                    n_blue = len(blue)
                    verdict = self.judge(green, blue)
            except (redis.RedisError, OSError) as e:
                verdict = None
                action = Action("hold", None, f"can't read the request log: {e}")
                return Tick(now, canary.green, canary.green_card, 0, 0, None, (), action.kind, None, action.reason)

        action = next_action(canary.green, verdict and verdict.level, rolled_back, self.promote)
        if action.kind == SET_TRAFFIC:
            self.set_traffic(action.green)
            if action.green == 0:
                self.record_rollback(canary, verdict, action)
        return Tick(now, canary.green, canary.green_card, n_green, n_blue, verdict and verdict.level,
                    verdict.reasons if verdict else (), action.kind, action.green, action.reason)  # fmt: skip


# --- The real dependencies: gateway, Redis, Week 7's checks, MLflow ------------------------------------------


def gateway_reader(url: str) -> Callable[[], Canary]:
    def read() -> Canary:
        health = httpx.get(f"{url}/health", timeout=5).raise_for_status().json()
        traffic = health["traffic"]
        return Canary(traffic[GREEN]["weight"], traffic[GREEN].get("card"), traffic[BLUE].get("card"),
                      health.get("traffic_since", ""))  # fmt: skip

    return read


def traffic_setter(url: str, token: str) -> Callable[[int], None]:
    def set_traffic(green: int) -> None:
        r = httpx.post(f"{url}/admin/traffic", json={"green": green}, headers={"X-Admin-Token": token}, timeout=5)
        r.raise_for_status()

    return set_traffic


def stream_reader(redis_url: str, stream: str, scan: int = 5000) -> Callable[..., pd.DataFrame]:
    """The newest `last` completed records a side served with a given card since `since`, oldest first."""
    client = redis.from_url(redis_url)

    def read(side: str, card: str | None, last: int, since: str = "") -> pd.DataFrame:
        rows = []
        for _, fields in client.xrevrange(stream, count=scan):
            r = json.loads(fields[b"data"])
            if r["ts"] < since:
                break  # newest first: everything further back is older still
            if r["side"] == side and r["card"] == card and r["status"] == 200 and r["completed"]:
                rows.append(r)
                if len(rows) == last:
                    break
        return pd.DataFrame(rows[::-1], columns=None if rows else ["ts", "prompt", "answer", "ttft_ms"])

    return read


def drift_judge(baseline: pd.DataFrame, embed, min_records: int) -> Callable[[pd.DataFrame, pd.DataFrame], Verdict]:
    from drift import compare

    def judge(green: pd.DataFrame, blue: pd.DataFrame) -> Verdict:
        vs_baseline = compare(baseline, green, embed)
        vs_control = compare(blue, green, embed) if len(blue) >= min_records else None
        return decide(vs_baseline, vs_control, min_records)

    return judge


class Registry:
    """The MLflow side: is a version rolled back (a tag on it), and the audit trail when the controller does it."""

    def __init__(self, tracking_uri: str, model: str):
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")  # Week 5: fail in seconds, not minutes
        from mlflow import MlflowClient

        self.client, self.model, self.tracking_uri = MlflowClient(tracking_uri), model, tracking_uri

    def is_rolled_back(self, version: str | None) -> bool:
        return version is not None and "auto_rolled_back_at" in self.client.get_model_version(self.model, version).tags

    def record_rollback(self, canary: Canary, verdict: Verdict | None, action: Action) -> None:
        import mlflow

        if canary.green_card is None:
            return
        version, reason = canary.green_card, "; ".join(verdict.reasons) if verdict else action.reason
        self.client.set_model_version_tag(self.model, version, "auto_rolled_back_at", datetime.now(UTC).isoformat())
        self.client.set_model_version_tag(self.model, version, "auto_rollback_reason", reason[:5000])
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment("canary-controller")
        with mlflow.start_run(run_name=f"rollback v{version}"):
            mlflow.log_params({"model": self.model, "version": version, "green_before": canary.green})
            mlflow.log_dict({"verdict": verdict and verdict.to_dict(), "action": action.reason}, "verdict.json")


def main() -> None:
    from drift import hashing, minilm

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gateway", default=os.getenv("GATEWAY_URL", "http://localhost:8080"))
    parser.add_argument("--token", default=os.getenv("ADMIN_TOKEN", "dev-token"))
    parser.add_argument("--redis", default=os.getenv("REDIS_URL", "redis://localhost:6379"))
    parser.add_argument("--stream", default=os.getenv("DRIFT_STREAM", "gateway:requests"))
    parser.add_argument(
        "--mlflow", default=os.getenv("MLFLOW_TRACKING_URI"), help="unset: remember rollbacks in memory"
    )
    parser.add_argument("--model", default=os.getenv("REGISTRY_MODEL", "gateway-llm"))
    parser.add_argument("--baseline", type=Path, default=ROOT / "week07-drift" / "data" / "baseline.parquet")
    parser.add_argument("--embeddings", choices=["minilm", "hashing"], default="minilm")
    parser.add_argument("--interval", type=float, default=10, help="seconds between checks")
    parser.add_argument("--window", type=int, default=150, help="newest records per side to judge")
    parser.add_argument("--min-records", type=int, default=MIN_RECORDS)
    parser.add_argument("--promote", action="store_true", help="on ok, step green up 10 -> 25 -> 50 -> 100")
    parser.add_argument("--once", action="store_true", help="one tick, then exit")
    parser.add_argument("--stop-when-done", action="store_true", help="exit after a rollback or a full promotion")
    parser.add_argument("--events", type=Path, help="append every tick here as JSON lines (for measurements)")
    args = parser.parse_args()

    if args.mlflow:
        registry = Registry(args.mlflow, args.model)
        is_rolled_back, record_rollback = registry.is_rolled_back, registry.record_rollback
    else:
        pulled: set[str | None] = set()  # no registry: remember in memory (lost on restart)
        is_rolled_back = pulled.__contains__

        def record_rollback(canary: Canary, verdict, action) -> None:
            pulled.add(canary.green_card)

    embed = minilm() if args.embeddings == "minilm" else hashing()
    controller = Controller(
        read_canary=gateway_reader(args.gateway),
        read_records=stream_reader(args.redis, args.stream),
        judge=drift_judge(pd.read_parquet(args.baseline), embed, args.min_records),
        set_traffic=traffic_setter(args.gateway, args.token),
        is_rolled_back=is_rolled_back,
        record_rollback=record_rollback,
        promote=args.promote,
        window=args.window,
        min_records=args.min_records,
    )
    print(f"controller: gateway {args.gateway}, every {args.interval:g} s, promote={args.promote}", flush=True)
    while True:
        t = controller.tick()
        target = "" if t.target is None else f" {t.target}%"
        print(f"{t.ts}  green {t.green}% (card {t.green_card})  records {t.n_green}/{t.n_blue}  "
              f"verdict {t.verdict}  -> {t.action}{target}  {t.why}", flush=True)  # fmt: skip
        if args.events:
            with args.events.open("a") as f:
                f.write(json.dumps(t.__dict__) + "\n")
        if args.once or (args.stop_when_done and (t.action == DONE or t.target == 0)):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
