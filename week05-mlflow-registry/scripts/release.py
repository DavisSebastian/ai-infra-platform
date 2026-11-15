"""Week 5, step 3: decide which registered version is live, by moving aliases.

    cd week05-mlflow-registry
    uv run --group week05 python scripts/release.py status
    uv run --group week05 python scripts/release.py promote 2
    uv run --group week05 python scripts/release.py rollback --reason "answers got worse"

Two aliases on `gateway-llm`:
  champion   the version the gateway serves
  fallback   the last good champion, kept ready so rollback is one pointer move

Rules:
  promote N   N must be tagged eval=passed. The old champion becomes fallback.
  rollback    champion goes back to fallback, and fallback is CLEARED: the version we just left is
              suspect, so it must not be one command away from going live again.

MLflow keeps only where an alias points now, not where it pointed before. So every move also tags
the version (promoted_at/by, rolled_back_at/reason): that's the audit trail.

Needs only mlflow-skinny (no torch), so it can run in CI or inside the gateway image.
"""

import argparse
import getpass
import os
from datetime import UTC, datetime

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = os.getenv("REGISTRY_MODEL", "gateway-llm")  # same variable the gateway reads
CHAMPION = "champion"
FALLBACK = "fallback"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")


class ReleaseError(Exception):
    """A move the rules don't allow. The registry is left unchanged."""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def current(client, name: str = MODEL_NAME) -> dict[str, str]:
    """Where the aliases point right now, e.g. {"champion": "2", "fallback": "1"}."""
    return dict(client.get_registered_model(name).aliases)


def promote(client, version: str, name: str = MODEL_NAME, by: str | None = None) -> str:
    """Make `version` the champion. Returns a one-line summary of what happened."""
    try:
        mv = client.get_model_version(name, version)
    except MlflowException as e:
        if e.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise  # e.g. the server is down: that's not a rules problem, so don't dress it up as one
        raise ReleaseError(f"{name} has no v{version}") from None
    verdict = mv.tags.get("eval")
    if verdict != "passed":
        raise ReleaseError(f"v{version} has eval={verdict or '(none)'}; only eval=passed versions can be promoted")

    aliases = current(client, name)
    old = aliases.get(CHAMPION)
    if old == version:
        return f"v{version} is already {CHAMPION}; nothing changed"

    # Order matters: two separate calls can't be atomic, so make every in-between state safe.
    # Crash after the first call -> champion is still the old version, and fallback points at it too. Harmless.
    if old is not None:
        client.set_registered_model_alias(name, FALLBACK, old)
    client.set_registered_model_alias(name, CHAMPION, version)

    client.set_model_version_tag(name, version, "promoted_at", now())
    client.set_model_version_tag(name, version, "promoted_by", by or getpass.getuser())
    if old is None:
        return f"{CHAMPION}: (none) -> v{version}"
    return f"{CHAMPION}: v{old} -> v{version}, {FALLBACK}: v{old}"


def rollback(client, name: str = MODEL_NAME, reason: str = "manual") -> str:
    """Put the fallback back in charge and clear fallback. Returns a one-line summary."""
    aliases = current(client, name)
    bad, good = aliases.get(CHAMPION), aliases.get(FALLBACK)
    if good is None:
        raise ReleaseError(f"no {FALLBACK} is set, so there is nothing safe to roll back to")

    # Champion moves first: from this call on, the gateway gets the good version.
    # Crash before the delete -> champion and fallback both point at the good version. Harmless.
    client.set_registered_model_alias(name, CHAMPION, good)
    client.delete_registered_model_alias(name, FALLBACK)

    if bad is not None:
        client.set_model_version_tag(name, bad, "rolled_back_at", now())
        client.set_model_version_tag(name, bad, "rollback_reason", reason)
    return f"{CHAMPION}: v{bad or '-'} -> v{good}, {FALLBACK}: cleared (reason: {reason})"


def status(client, name: str = MODEL_NAME) -> str:
    aliases = current(client, name)
    lines = []
    for mv in sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)):
        labels = [a for a, v in aliases.items() if v == mv.version]
        lines.append(
            f"  v{mv.version:<3} eval={mv.tags.get('eval', '-'):<7} {mv.tags.get('card_file', ''):<22}"
            f" {', '.join(labels)}"
        )
    return f"{name}\n" + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="list versions and where the aliases point")
    p = sub.add_parser("promote", help="make a version the champion")
    p.add_argument("version", help="version number, e.g. 2")
    r = sub.add_parser("rollback", help="switch champion back to fallback")
    r.add_argument("--reason", default="manual", help="why, stored as a tag on the version rolled back from")
    args = parser.parse_args()

    client = MlflowClient(TRACKING_URI)
    try:
        if args.command == "promote":
            print(promote(client, args.version))
        elif args.command == "rollback":
            print(rollback(client, reason=args.reason))
        print(status(client))
    except ReleaseError as e:
        raise SystemExit(f"refused: {e}") from None


if __name__ == "__main__":
    main()
