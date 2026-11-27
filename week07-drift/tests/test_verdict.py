"""Week 7: the verdict rules. Written before verdict.py, as its spec.

decide() turns drift checks into one answer: insufficient_data / ok / warn / alert. Week 8 rolls back on alert,
so the rules are about two things: blame the model only when the evidence points at it, and don't act on one noisy
check (each has a ~5% false-alarm rate).
"""

import json

import pytest
from drift import Check, DriftResult
from verdict import ALERT, INSUFFICIENT, OK, SHIFT, WARN, decide

INPUTS = ("prompt_words", "prompt_embedding")
OUTPUTS = ("answer_words", "ttft_ms", "answer_empty", "refusal")


def result(*drifted: str, n: int = 150, n_ref: int = 150) -> DriftResult:
    """A DriftResult where exactly the named checks drifted."""
    checks = tuple(
        Check(f, "input" if f in INPUTS else "output", "test", 0.0 if f in drifted else 0.5, f in drifted)
        for f in INPUTS + OUTPUTS
    )
    return DriftResult(n_ref, n, checks)


# --- Not enough data to judge ----------------------------------------------------------------------------


def test_too_few_records_is_insufficient_even_if_everything_drifted():
    v = decide(result(*OUTPUTS, n=20))
    assert v.level == INSUFFICIENT
    assert "20" in v.reasons[0]


def test_a_small_control_window_is_insufficient_too():
    assert decide(result(), vs_control=result(n=30)).level == INSUFFICIENT


def test_the_minimum_is_adjustable():
    assert decide(result(n=20), min_records=10).level == OK


# --- Nothing changed -------------------------------------------------------------------------------------


def test_no_drift_is_ok():
    assert decide(result()).level == OK
    assert decide(result(), vs_control=result()).level == OK


# --- Users changed: warn a human, never roll back --------------------------------------------------------


def test_input_drift_alone_is_a_warning():
    v = decide(result("prompt_embedding"))
    assert v.level == WARN
    assert any("prompt_embedding" in r for r in v.reasons)


def test_without_a_control_output_drift_with_input_drift_is_only_a_warning():
    # Answers got longer AND the questions changed: the users explain it, not the model.
    assert decide(result("prompt_embedding", "answer_words", "ttft_ms")).level == WARN


def test_both_sides_drifting_the_same_way_is_a_shift_not_the_model():
    # Green differs from the baseline, but not from blue over the same period: blue sees the same change.
    v = decide(result("prompt_embedding", "answer_words", "ttft_ms"), vs_control=result())
    assert v.level == SHIFT
    assert any("blue" in r or "control" in r for r in v.reasons)


# --- The model changed: alert (Week 8 rolls back) ---------------------------------------------------------


def test_without_a_control_two_output_checks_and_no_input_drift_is_an_alert():
    assert decide(result("answer_words", "refusal")).level == ALERT


def test_without_a_control_one_output_check_is_only_a_warning():
    # One check at p < 0.05 happens by chance about 1 time in 20: not enough to roll back on.
    assert decide(result("answer_words")).level == WARN


def test_two_output_checks_differing_from_the_control_is_an_alert():
    assert decide(result(), vs_control=result("answer_words", "refusal")).level == ALERT


def test_one_output_check_confirmed_by_both_comparisons_is_an_alert():
    # Shorter answers than blue right now AND than the baseline: two comparisons agree.
    v = decide(result("answer_words"), vs_control=result("answer_words"))
    assert v.level == ALERT
    assert any("answer_words" in r for r in v.reasons)


def test_one_output_check_against_the_control_only_is_a_warning():
    assert decide(result(), vs_control=result("answer_words")).level == WARN


def test_a_model_problem_is_still_an_alert_while_users_are_also_changing():
    # With a control, input drift doesn't excuse green: blue gets the same users and didn't change like this.
    v = decide(result("prompt_embedding", "answer_words"), vs_control=result("answer_words", "refusal"))
    assert v.level == ALERT


# --- Output ----------------------------------------------------------------------------------------------


def test_the_verdict_is_frozen_and_serializable():
    v = decide(result("answer_words", "refusal"))
    with pytest.raises(AttributeError):
        v.level = OK
    data = json.loads(json.dumps(v.to_dict()))
    assert data["level"] == ALERT and data["reasons"]
