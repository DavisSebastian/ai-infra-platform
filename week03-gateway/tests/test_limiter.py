"""Unit tests for the pure token bucket. No Redis, no sleeping: time is just a number we pass in."""

import pytest
from app.limiter import Bucket, take


def run(requests_at: list[float], capacity: float, refill_rate: float) -> list[bool]:
    """Feed a series of request times through one client's bucket; return allowed/rejected for each."""
    bucket = None
    results = []
    for now in requests_at:
        allowed, bucket = take(bucket, now, capacity, refill_rate)
        results.append(allowed)
    return results


def test_new_client_starts_full_and_can_burst_up_to_capacity():
    # The worked example from the concept walkthrough: capacity 3, four requests at t=0.
    assert run([0, 0, 0, 0], capacity=3, refill_rate=1) == [True, True, True, False]


def test_refill_over_time():
    # capacity 3, refill 1/s: empty the bucket, wait 2 s -> exactly 2 permits are back.
    assert run([0, 0, 0, 2, 2, 2], capacity=3, refill_rate=1) == [True, True, True, True, True, False]


def test_check_yourself_example_from_the_walkthrough():
    # capacity 4, refill 2/s. Six at t=0 (4 allowed, 2 rejected), one more at t=1 (2 permits refilled -> allowed).
    assert run([0] * 6 + [1], capacity=4, refill_rate=2) == [True] * 4 + [False] * 2 + [True]


def test_bucket_never_holds_more_than_capacity():
    # Idle for an hour: the bucket is full (3), not 3600. So only 3 requests get through, not more.
    assert run([0, 3600, 3600, 3600, 3600], capacity=3, refill_rate=1) == [True, True, True, True, False]


def test_fractional_refill():
    # 0.5 permits/s: after 1 s only half a permit is back (not enough), after 2 s a whole one is.
    results = run([0, 0, 1, 2], capacity=2, refill_rate=0.5)
    assert results == [True, True, False, True]


def test_rejected_request_does_not_consume_permits():
    # Hammering a drained bucket must not push it negative: after exactly 1 s one request fits.
    assert run([0, 0, 0, 0, 0, 0, 1], capacity=2, refill_rate=1) == [True, True, False, False, False, False, True]


def test_clock_going_backwards_does_not_add_or_remove_permits():
    # Clocks can jump back (NTP adjustments, different servers). Elapsed time must be treated as 0, never negative.
    assert run([10, 10, 9, 9], capacity=2, refill_rate=1) == [True, True, False, False]


def test_take_does_not_modify_the_bucket_it_was_given():
    original = Bucket(permits=2.0, updated_at=0.0)
    allowed, new = take(original, now=0.0, capacity=3, refill_rate=1)
    assert allowed
    assert original == Bucket(permits=2.0, updated_at=0.0)  # unchanged
    assert new.permits == pytest.approx(1.0)


def test_cost_greater_than_one():
    # A request can cost more than one permit (e.g. a big prompt). 2 permits left, cost 3 -> rejected, nothing taken.
    allowed, bucket = take(Bucket(2.0, 0.0), now=0.0, capacity=5, refill_rate=1, cost=3)
    assert not allowed
    assert bucket.permits == pytest.approx(2.0)


def test_clients_are_independent():
    a_allowed, _ = take(None, 0, capacity=1, refill_rate=1)
    a_again, a_bucket = take(Bucket(0.0, 0.0), 0, capacity=1, refill_rate=1)
    b_allowed, _ = take(None, 0, capacity=1, refill_rate=1)
    assert a_allowed and b_allowed and not a_again
    assert a_bucket.permits == pytest.approx(0.0)


def test_rejections_do_not_count_the_same_elapsed_time_twice():
    # Found by mutation testing: storing the refilled permits but keeping the OLD timestamp on a rejection counts
    # the same seconds again on the next call, so the limit leaks. capacity 2, refill 1/s: drain at t=0, then two
    # rejections at 0.6 and 0.9. Only 0.9 permits have really refilled, so the second one must still be rejected.
    assert run([0, 0, 0.6, 0.9], capacity=2, refill_rate=1) == [True, True, False, False]


def test_a_backwards_clock_step_does_not_eat_permits():
    # Also found by mutation testing. One permit left, then the clock jumps back 1 s. Elapsed must count as 0, so the
    # remaining permit is still usable. With a negative elapsed it would be wiped out and this request rejected.
    assert run([10, 9, 9], capacity=2, refill_rate=1) == [True, True, False]
