"""Week 7: turn drift checks into one answer. Week 8's automatic rollback acts on ALERT.

    decide(green_vs_baseline)                              one model server
    decide(green_vs_baseline, vs_control=green_vs_blue)    Blue/Green: blue is the control group

  INSUFFICIENT  a window is too small to judge (with 20 requests, everything looks random)
  OK            nothing drifted
  SHIFT         (Blue/Green only) green differs from the baseline, but blue shows the same difference over the same
                period: the users or the environment changed, not green's model. Green is as good as blue, so a canary
                may proceed; the shift is worth a human's attention (and maybe a fresh baseline)
  WARN          something changed and green might be the cause, but the evidence is too thin: hold, tell a human
  ALERT         the evidence points at the model: Week 8 rolls back

Why SHIFT is its own level (found in Week 8): monitoring asks "is the world different from last week?", a canary asks
"is green as good as blue right now?". Under 2 requests/s the model server queues, so TTFT rose for BOTH sides versus
a baseline captured one request at a time. As a WARN that blocked every promotion; as a SHIFT it doesn't.

Two ideas decide where the line is:

1. Blame the model only when the users don't explain it. Drift says "different", not "worse": answers get longer
   when users ask harder questions. During a canary, blue serves the same users at the same time (the split is random
   per request), so green differing from BLUE points at green's model. Without a control, output drift that comes
   with input drift is put down to the users.

2. Don't act on one noisy check. Each check is a test at p < 0.05: about 1 time in 20 it fires with nothing wrong.
   With four output checks and a check every few minutes, acting on any single one would roll back for nothing
   several times a day. So an alert needs two agreeing signals: two output checks, or the same output check
   drifting in both comparisons (vs blue now, and vs the baseline). If the four checks were independent, two firing
   by chance would happen about 1 time in 70 rather than about 1 in 5; they aren't fully independent (longer answers
   also take longer), so treat that as a rough guide, not a guarantee.
"""

from dataclasses import dataclass

from drift import DriftResult

INSUFFICIENT, OK, SHIFT, WARN, ALERT = "insufficient_data", "ok", "shift", "warn", "alert"
MIN_RECORDS = 50


@dataclass(frozen=True)
class Verdict:
    level: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"level": self.level, "reasons": list(self.reasons)}


def decide(vs_baseline: DriftResult, vs_control: DriftResult | None = None, min_records: int = MIN_RECORDS) -> Verdict:
    # 1. Enough data in every window?
    windows = [vs_baseline.n_reference, vs_baseline.n_current]
    if vs_control is not None:
        windows += [vs_control.n_reference, vs_control.n_current]
    if min(windows) < min_records:
        return Verdict(INSUFFICIENT, (f"smallest window has {min(windows)} records; need {min_records}",))

    inputs, outputs = vs_baseline.drifted("input"), vs_baseline.drifted("output")

    # 2. Blue/Green: green against blue over the same period. Same users, so a difference is the model.
    if vs_control is not None:
        vs_blue = vs_control.drifted("output")
        confirmed = [f for f in vs_blue if f in outputs]  # also differs from the baseline
        if len(vs_blue) >= 2:
            return Verdict(ALERT, (f"output differs from the control (blue) on the same traffic: {vs_blue}",))
        if confirmed:
            return Verdict(ALERT, (f"output differs from both the control (blue) and the baseline: {confirmed}",))
        if vs_blue:
            return Verdict(WARN, (f"one output check differs from the control (blue), unconfirmed: {vs_blue}",))
        if inputs or outputs:
            reason = (f"drift vs the baseline, but the control (blue) shows it too: the users or the environment "
                      f"changed, not green's model (input {inputs}, output {outputs})")  # fmt: skip
            return Verdict(SHIFT, (reason,))
        return Verdict(OK, ("no drift vs the baseline or the control",))

    # 3. One model server: no control group, so input drift explains output drift.
    if len(outputs) >= 2 and not inputs:
        return Verdict(ALERT, (f"output drift with no input drift: {outputs}",))
    if outputs or inputs:
        reasons = []
        if inputs:
            reasons.append(f"input drift (the users changed): {inputs}")
        if outputs:
            reasons.append(f"output drift: {outputs}" + (" (explained by input drift?)" if inputs else " (one check)"))
        return Verdict(WARN, tuple(reasons))
    return Verdict(OK, ("no drift vs the baseline",))
