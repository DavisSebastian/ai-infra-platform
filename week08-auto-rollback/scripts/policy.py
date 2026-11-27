"""Week 8: the canary controller's brain. Given where the canary is and what decide() said, what should happen next?

    next_action(green=10, verdict="alert", rolled_back=False)   -> Action("set_traffic", 0, ...)   roll back
    next_action(green=10, verdict="ok", rolled_back=False, promote=True)  -> Action("set_traffic", 25, ...)

A pure function: no network, no clock, no Redis. The controller loop (controller.py) does the I/O around it and
calls this once per check, which makes every rule testable on its own, like Week 6's choose().

The rules (tests in week08-auto-rollback/tests/test_policy.py):
  - a rollback is final for this deployment: never undo it, whatever later checks say (no flapping)
  - green at 0% with no rollback: no canary is running, nothing to do
  - "alert": set green to 0. Always safe: blue is the known-good version, so moving traffic TO blue needs no proof
  - "insufficient_data", "warn", or None (the controller couldn't read the data): hold. Never act on missing or
    ambiguous evidence; a human looks at warnings
  - "ok": hold, or with promote=True step up to the next of STEPS (10 -> 25 -> 50 -> 100); at 100% the canary is done.
    Moving traffic TO green needs evidence, and "ok" is that evidence
  - "shift" (green matches blue; both differ from the baseline) counts as "ok" for the canary: the question here is
    "is green as good as blue?", and it is. The reason says so, for the humans watching the log
  - at 100% the canary analysis is over: blue gets no traffic, so there is no control group left to compare with.
    Anything but "alert" means done (the evidence was gathered at the earlier steps); "alert" still rolls back,
    because blue is still running and can take the traffic back
  - any other verdict is a bug in the caller: raise ValueError

"""

from dataclasses import dataclass

STEPS = (10, 25, 50, 100)  # canary traffic steps for green, in percent
HOLD, SET_TRAFFIC, DONE = "hold", "set_traffic", "done"
VERDICTS = ("insufficient_data", "ok", "shift", "warn", "alert")


@dataclass(frozen=True)
class Action:
    kind: str  # HOLD (check again later), SET_TRAFFIC (move green to `green` %), or DONE (stop watching)
    green: int | None  # the target percent for SET_TRAFFIC, else None
    reason: str  # one line for the log and the MLflow audit trail


def next_action(green: int, verdict: str | None, rolled_back: bool, promote: bool = False) -> Action:
    # Checked first: a typo like "alret" must fail loudly, not quietly fall through to "hold" and never roll back.
    if verdict is not None and verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}; expected one of {VERDICTS} or None")

    # Order matters: "rolled back" wins over everything, even a fresh alert. There is nothing left to undo.
    if rolled_back:
        return Action(DONE, None, "rolled back: final for this deployment, a human redeploys")
    if green == 0:
        return Action(DONE, None, "no canary running (green has 0% of traffic)")

    if verdict == "alert":
        return Action(SET_TRAFFIC, 0, f"alert at {green}%: rolling back, all traffic to blue")
    if green >= STEPS[-1]:
        return Action(DONE, None, f"at 100% ({verdict}): fully promoted; no control group left to compare against")
    if verdict is None:
        return Action(HOLD, None, "no verdict (the data couldn't be read): holding, never act blind")
    if verdict in ("insufficient_data", "warn"):
        return Action(HOLD, None, f"{verdict}: holding at {green}%")

    # verdict is "ok" or "shift": green is as good as blue
    note = "ok" if verdict == "ok" else "shift (green matches blue; both differ from the baseline)"
    if not promote:
        return Action(HOLD, None, f"{note} at {green}%: holding (promotion is off)")
    target = next(step for step in STEPS if step > green)  # 15% set by hand -> 25%, not 50%
    return Action(SET_TRAFFIC, target, f"{note} at {green}%: stepping up to {target}%")
