"""Week 7: is the current traffic different from the baseline? Compares two windows and prints every check.

    # two saved windows
    uv run python week07-drift/scripts/drift_check.py --current week07-drift/data/normal-holdout.parquet

    # the live gateway: the last 150 requests green served, straight from the Redis stream
    uv run python week07-drift/scripts/drift_check.py --stream-last 150 --side green

    # canary analysis: green vs blue over the same period (same users, so differences point at the model)
    uv run python week07-drift/scripts/drift_check.py --stream-last 150 --side green --against-side blue

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
from drift import compare, hashing, minilm  # noqa: E402  (a sibling script, not an installed package)


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
    parser.add_argument("--against-side", help="use the same stream's records for THIS side as the reference")
    parser.add_argument("--embeddings", choices=["minilm", "hashing"], default="minilm")
    parser.add_argument("--redis", default=os.getenv("REDIS_URL", "redis://localhost:6379"))
    parser.add_argument("--stream", default=os.getenv("DRIFT_STREAM", "gateway:requests"))
    parser.add_argument("--html", type=Path, help="save Evidently's HTML report here")
    parser.add_argument("--json", type=Path, help="save the result as JSON here")
    args = parser.parse_args()

    if args.current:
        current = pd.read_parquet(args.current)
        if args.side:
            current = current[current["side"] == args.side]
    else:
        current = from_stream(args.redis, args.stream, args.side, args.stream_last)
    if args.against_side:
        reference = from_stream(args.redis, args.stream, args.against_side, args.stream_last or len(current))
        ref_name = f"stream, side={args.against_side}"
    else:
        reference = pd.read_parquet(args.reference)
        ref_name = args.reference.name

    embed = minilm() if args.embeddings == "minilm" else hashing()
    result = compare(reference, current, embed, html=args.html)

    cur_name = args.current.name if args.current else f"stream, side={args.side or 'any'}"
    print(f"\nreference: {ref_name} ({result.n_reference})   current: {cur_name} ({result.n_current})\n")
    print(f"  {'check':<18} {'kind':<7} {'method':<16} {'score':>8}  drift?")
    for c in result.checks:
        print(f"  {c.feature:<18} {c.kind:<7} {c.method:<16} {c.score:>8.4f}  {'YES' if c.drifted else 'no'}")
    print(f"\n  input drift:  {result.drifted('input') or 'none'}")
    print(f"  output drift: {result.drifted('output') or 'none'}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
