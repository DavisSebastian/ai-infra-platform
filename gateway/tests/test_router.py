"""Week 6: the rules of the Blue/Green split. Written before router.py, as its spec."""

import random

import pytest
from gateway.router import BLUE, GREEN, TrafficSplit, choose, split_from_env

# --- TrafficSplit ----------------------------------------------------------------------------------------


def test_default_is_all_blue():
    assert (TrafficSplit().green, TrafficSplit().blue) == (0, 100)


@pytest.mark.parametrize("green", [0, 1, 10, 50, 99, 100])
def test_blue_gets_whatever_green_does_not(green):
    assert TrafficSplit(green).blue == 100 - green


@pytest.mark.parametrize("green", [-1, 101, 1000])
def test_out_of_range_is_refused(green):
    with pytest.raises(ValueError, match=str(green)):
        TrafficSplit(green)


@pytest.mark.parametrize("green", [10.5, "10", None, True, False])
def test_only_whole_numbers(green):
    # True would otherwise be 1% green, and False 0%: bool is an int in Python (Week 5's trap again).
    with pytest.raises(ValueError):
        TrafficSplit(green)


def test_a_split_cannot_be_changed_in_place():
    split = TrafficSplit(10)
    with pytest.raises(AttributeError):  # dataclasses.FrozenInstanceError is an AttributeError
        split.green = 50


def test_splits_compare_by_value():
    # The gateway uses this to tell "the traffic changed" from "the same split was set again".
    assert TrafficSplit(10) == TrafficSplit(10) and TrafficSplit(10) != TrafficSplit(20)


# --- choose() --------------------------------------------------------------------------------------------


def test_all_blue_never_picks_green_even_at_the_lowest_r():
    assert choose(TrafficSplit(0), 0.0) == BLUE


def test_all_green_always_picks_green_even_at_the_highest_r():
    assert choose(TrafficSplit(100), 0.999999) == GREEN


@pytest.mark.parametrize("green", [0, 1, 10, 33, 50, 90, 100])
def test_the_share_is_exact_over_evenly_spread_r(green):
    # 1000 evenly spread values of r: exactly green% of them must go to green, not roughly.
    picks = [choose(TrafficSplit(green), i / 1000) for i in range(1000)]
    assert picks.count(GREEN) == green * 10


def test_low_r_goes_to_green_high_r_to_blue():
    # One fixed rule, so a given r always lands on the same side. Green takes the bottom slice of [0, 1).
    split = TrafficSplit(10)
    assert choose(split, 0.05) == GREEN and choose(split, 0.15) == BLUE


def test_with_real_randomness_the_share_is_close():
    rng = random.Random(6)  # seeded: the test gives the same answer every run
    picks = [choose(TrafficSplit(10), rng.random()) for _ in range(10_000)]
    assert 900 <= picks.count(GREEN) <= 1100


@pytest.mark.parametrize("r", [-0.1, 1.0, 1.5])
def test_r_outside_0_to_1_is_a_bug(r):
    with pytest.raises(ValueError):
        choose(TrafficSplit(50), r)


# --- split_from_env() ------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, ""])
def test_unset_means_all_blue(value):
    assert split_from_env(value) == TrafficSplit(0)


@pytest.mark.parametrize(("value", "green"), [("0", 0), ("10", 10), ("100", 100), (" 25 ", 25)])
def test_a_number_sets_green(value, green):
    assert split_from_env(value) == TrafficSplit(green)


@pytest.mark.parametrize("value", ["ten", "10%", "10.5", "-5", "150"])
def test_a_bad_value_names_the_setting(value):
    with pytest.raises(ValueError, match="TRAFFIC_GREEN"):
        split_from_env(value)
