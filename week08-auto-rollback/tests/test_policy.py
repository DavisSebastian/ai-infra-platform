"""Week 8: the controller's rules. Written before policy.py, as its spec."""

import pytest
from policy import DONE, HOLD, SET_TRAFFIC, STEPS, Action, next_action

# --- Roll back on alert ----------------------------------------------------------------------------------


@pytest.mark.parametrize("green", [10, 25, 50, 100])
def test_an_alert_sets_green_to_zero_at_any_step(green):
    a = next_action(green, "alert", rolled_back=False)
    assert (a.kind, a.green) == (SET_TRAFFIC, 0)
    assert "alert" in a.reason


def test_an_alert_rolls_back_even_when_promotion_is_on():
    assert next_action(50, "alert", rolled_back=False, promote=True).green == 0


# --- Never act on missing or ambiguous evidence ----------------------------------------------------------


@pytest.mark.parametrize("verdict", ["insufficient_data", "warn", None])
def test_no_clear_verdict_means_hold(verdict):
    a = next_action(10, verdict, rolled_back=False, promote=True)
    assert (a.kind, a.green) == (HOLD, None)


def test_ok_without_promotion_holds():
    assert next_action(10, "ok", rolled_back=False).kind == HOLD


# --- Promotion needs evidence, one step at a time ---------------------------------------------------------


@pytest.mark.parametrize(("green", "to"), [(10, 25), (25, 50), (50, 100)])
def test_ok_with_promotion_takes_the_next_step(green, to):
    a = next_action(green, "ok", rolled_back=False, promote=True)
    assert (a.kind, a.green) == (SET_TRAFFIC, to)


def test_a_shift_shared_with_blue_promotes_like_ok():
    # Green matches blue; both differ from the baseline (e.g. more load made TTFT rise on both). Not green's fault.
    a = next_action(10, "shift", rolled_back=False, promote=True)
    assert (a.kind, a.green) == (SET_TRAFFIC, 25)
    assert "shift" in a.reason


def test_a_shift_without_promotion_holds():
    assert next_action(10, "shift", rolled_back=False).kind == HOLD


def test_a_split_between_steps_goes_to_the_next_step_up():
    assert next_action(15, "ok", rolled_back=False, promote=True).green == 25  # set by hand, e.g. 15%


def test_fully_promoted_is_done():
    assert next_action(100, "ok", rolled_back=False, promote=True).kind == DONE


@pytest.mark.parametrize("verdict", ["warn", "shift", "insufficient_data", None])
def test_at_100_percent_the_canary_analysis_is_over(verdict):
    # Blue gets no traffic at 100%, so there's no control group: found stuck on "warn" forever in the real run.
    a = next_action(100, verdict, rolled_back=False, promote=True)
    assert (a.kind, a.green) == (DONE, None)


def test_an_alert_at_100_percent_still_rolls_back():
    assert next_action(100, "alert", rolled_back=False).green == 0


def test_the_steps_go_up_and_end_at_100():
    assert list(STEPS) == sorted(STEPS) and STEPS[-1] == 100 and STEPS[0] > 0


# --- A rollback is final ---------------------------------------------------------------------------------


@pytest.mark.parametrize("verdict", ["ok", "shift", "alert", "warn", "insufficient_data", None])
def test_after_a_rollback_nothing_changes(verdict):
    # Even a later "ok" must not send traffic back to the version that was just pulled: no flapping.
    a = next_action(0, verdict, rolled_back=True, promote=True)
    assert (a.kind, a.green) == (DONE, None)


def test_no_canary_running_is_done():
    assert next_action(0, "ok", rolled_back=False, promote=True).kind == DONE


# --- Guard rails -----------------------------------------------------------------------------------------


def test_an_unknown_verdict_is_a_bug():
    with pytest.raises(ValueError, match="maybe"):
        next_action(10, "maybe", rolled_back=False)


def test_an_action_is_frozen():
    a = next_action(10, "alert", rolled_back=False)
    assert isinstance(a, Action)
    with pytest.raises(AttributeError):
        a.green = 50
