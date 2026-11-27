"""Week 8: what did a rollback run cost? Reads what actually happened, not what the controller printed.

    uv run python week08-auto-rollback/scripts/summarize.py --stream week08:bad-deploy \
        --traffic week08-auto-rollback/results/bad-deploy-traffic.jsonl --deploy-at 2026-11-27T05:05:42

  from the gateway's request log (Redis stream)   which side and card served every request, and when
  from traffic.py's file                          every request's outcome as the user saw it (failed or not)
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd
import redis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--traffic", required=True, help="traffic.py's output (JSON lines)")
    parser.add_argument("--deploy-at", required=True, help="UTC time green got traffic, e.g. 2026-11-27T05:05:42")
    parser.add_argument("--redis", default=os.getenv("REDIS_URL", "redis://localhost:6379"))
    parser.add_argument("--json", help="also write the summary here")
    args = parser.parse_args()

    client = redis.from_url(args.redis)
    log = pd.DataFrame([json.loads(f[b"data"]) for _, f in client.xrange(args.stream)])
    log["ts"] = pd.to_datetime(log["ts"])
    users = pd.DataFrame([json.loads(line) for line in open(args.traffic)])
    deploy = pd.Timestamp(datetime.fromisoformat(args.deploy_at)).tz_localize("UTC")

    green = log[log["side"] == "green"]
    last_green = green["ts"].max()
    after = log[log["ts"] > last_green]
    failed = users[(users["status"] != 200) | (~users["completed"])]
    summary = {
        "requests": len(users),
        "failed": len(failed),
        "served_by_green": len(green),
        "green_cards": green["card"].value_counts().to_dict(),
        "green_share_while_canary": round(len(green) / len(log[(log["ts"] >= deploy) & (log["ts"] <= last_green)]), 3),
        "deploy_to_last_green_request_s": round((last_green - deploy).total_seconds(), 1),
        "requests_after_rollback": len(after),
        "green_after_rollback": int((after["side"] == "green").sum()),
    }
    for k, v in summary.items():
        print(f"  {k:<32} {v}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
