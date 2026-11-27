"""Week 7: is the current traffic different from the baseline? Compares two windows and prints every check.

    # two saved windows
    uv run python week07-drift/scripts/drift_check.py --current week07-drift/data/normal-holdout.parquet

    # the live gateway: the last 150 requests green served, straight from the Redis stream
    uv run python week07-drift/scripts/drift_check.py --stream-last 150 --side green

    # canary analysis: green vs the baseline AND green vs blue over the same period (blue is the control group)
    uv run python week07-drift/scripts/drift_check.py --stream-last 150 --side green --against-side blue

    # the same, from a saved window that holds both sides' records (as capture.py saves them)
    uv run python week07-drift/scripts/drift_check.py --current week07-drift/data/scenario-c.parquet \
        --side green --against-side blue

It ends with decide()'s verdict: insufficient_data / ok / shift / warn / alert (Week 8 rolls back on alert).
The reference defaults to week07-drift/data/baseline.parquet. --html saves Evidently's report (histograms of every
feature, reference vs current); --json saves the numbers for other tools (Week 8's rollback reads these).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import redis

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from drift import DriftResult, compare, hashing, minilm  # noqa: E402  (a sibling script, not an installed package)
from verdict import INSUFFICIENT, MIN_RECORDS, decide  # noqa: E402


def from_stream(redis_url: str, stream: str, side: str | None, last: int) -> pd.DataFrame:
    """The newest `last` successful records (for one side, if given), oldest first."""
    client = redis.from_url(redis_url)
    rows, cursor = [], "+"
    while len(rows) < last:
        batch = client.xrevrange(stream, max=cursor, count=500)
        if not batch:
            break
        for entry_id, fields in batch:
            record = json.loads(fields[b"data"])
            if record["status"] == 200 and record["completed"] and side in (None, record["side"]):
                rows.append(record)
        cursor = "(" + batch[-1][0].decode()  # exclusive: continue just before the oldest entry seen
    return pd.DataFrame(rows[:last][::-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=HERE.parent / "data" / "baseline.parquet")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--current", type=Path, help="a saved window (.parquet)")
    source.add_argument("--stream-last", type=int, metavar="N", help="the last N records from the gateway's stream")
    parser.add_argument("--side", help="only records this side served (blue or green)")
    parser.add_argument("--against-side", help="also compare with this side's recent records (the control group)")
    parser.add_argument("--min-records", type=int, default=MIN_RECORDS, help="smaller windows aren't judged")
    parser.add_argument("--embeddings", choices=["minilm", "hashing"], default="minilm")
    parser.add_argument("--redis", default=os.getenv("REDIS_URL", "redis://localhost:6379"))
    parser.add_argument("--stream", default=os.getenv("DRIFT_STREAM", "gateway:requests"))
    parser.add_argument("--html", type=Path, help="save Evidently's HTML report here")
    parser.add_argument("--json", type=Path, help="save the result as JSON here")
    args = parser.parse_args()

    control = None
    if args.current:  # a saved window: blue and green records are both in it, as the gateway logged them
        saved = pd.read_parquet(args.current)
        current = saved[saved["side"] == args.side] if args.side else saved
        if args.against_side:
            control = saved[saved["side"] == args.against_side]
    else:
        current = from_stream(args.redis, args.stream, args.side, args.stream_last)
        if args.against_side:
            control = from_stream(args.redis, args.stream, args.against_side, args.stream_last)
    smallest = min(len(current), len(control) if control is not None else len(current))
    if smallest < args.min_records:  # e.g. green just deployed: judge nothing yet rather than crash or guess
        print(f"\nVERDICT: {INSUFFICIENT.upper()}\n  - smallest window has {smallest} records; need {args.min_records}")
        return

    embed = minilm() if args.embeddings == "minilm" else hashing()
    cur_name = args.current.name if args.current else f"stream, side={args.side or 'any'}"
    vs_baseline = compare(pd.read_parquet(args.reference), current, embed, html=args.html)
    show(f"{cur_name} vs {args.reference.name}", vs_baseline)
    vs_control = None
    if control is not None:
        vs_control = compare(control, current, embed)
        show(f"{cur_name} vs side={args.against_side} (the control: same users, same period)", vs_control)

    verdict = decide(vs_baseline, vs_control, args.min_records)
    print(f"\nVERDICT: {verdict.level.upper()}")
    for reason in verdict.reasons:
        print(f"  - {reason}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        out = {"vs_baseline": vs_baseline.to_dict(), "vs_control": vs_control and vs_control.to_dict(),
               "verdict": verdict.to_dict()}  # fmt: skip
        args.json.write_text(json.dumps(out, indent=2))


def show(title: str, result: DriftResult) -> None:
    print(f"\n{title}  (reference {result.n_reference}, current {result.n_current})")
    print(f"  {'check':<18} {'kind':<7} {'method':<26} {'score':>8}  drift?")
    for c in result.checks:
        print(f"  {c.feature:<18} {c.kind:<7} {c.method:<26} {c.score:>8.4f}  {'YES' if c.drifted else 'no'}")


if __name__ == "__main__":
    main()
