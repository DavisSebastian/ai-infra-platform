"""Week 6: the Blue/Green traffic split. Which model server does this request go to?

Two model servers stay running side by side: blue and green. A TrafficSplit says what percent of requests go to
green; blue gets the rest. The gateway picks a side ONCE per request with choose(), so a request that starts on
blue finishes on blue, even if the split changes while it is streaming.

    split = TrafficSplit(green=10)        # canary: 10% to green
    choose(split, random.random())        # -> "blue" about 9 times in 10
    TrafficSplit(green=0)                 # rollback: everything back to blue

Like Week 5's ServingCard, a split is frozen: changing the traffic means building a new split and swapping one
reference (app.state.split = new), never editing the one that running requests are reading.

The tests (written first, as the spec) are in gateway/tests/test_router.py:
    uv run pytest gateway/tests/test_router.py
"""

from dataclasses import dataclass

BLUE = "blue"
GREEN = "green"


@dataclass(frozen=True)
class TrafficSplit:
    green: int = 0  # percent of requests that go to green, a whole number from 0 to 100

    def __post_init__(self) -> None:
        # bool first: True is an int in Python, so without this TrafficSplit(True) would mean 1% green.
        if isinstance(self.green, bool) or not isinstance(self.green, int):
            raise ValueError(f"green must be a whole number from 0 to 100, got {self.green!r}")
        if not 0 <= self.green <= 100:
            raise ValueError(f"green must be from 0 to 100, got {self.green}")

    @property
    def blue(self) -> int:
        """Percent of requests that go to blue: whatever green doesn't get."""
        return 100 - self.green


def choose(split: TrafficSplit, r: float) -> str:
    """BLUE or GREEN for one request.

    r is a random number from 0 (included) to 1 (excluded), normally random.random(). It is passed in rather than
    drawn here so tests can pick it: the same idea as passing a Clock into Java code instead of calling now().
    Out-of-range r is a bug in the caller: raise ValueError.
    """
    if not 0 <= r < 1:
        raise ValueError(f"r must be from 0 (included) to 1 (excluded), got {r}")
    # Green owns the bottom slice [0, green%), blue the rest [green%, 100%). Strict `<`: the boundary is blue's,
    # so green=0 never picks green, even at r=0.0.
    return GREEN if r * 100 < split.green else BLUE


def split_from_env(value: str | None) -> TrafficSplit:
    """The starting split, from the TRAFFIC_GREEN environment variable.

    Unset or empty -> all blue. "10" -> 10% green. Anything else -> ValueError that names TRAFFIC_GREEN, so a
    typo in the deployment config stops the gateway at startup instead of silently sending traffic somewhere.
    """
    if value is None or not value.strip():
        return TrafficSplit()
    try:
        # int() accepts " 25 " and refuses "10.5", "10%" and "ten". TrafficSplit refuses -5 and 150.
        return TrafficSplit(int(value))
    except ValueError as e:
        raise ValueError(f"TRAFFIC_GREEN must be a whole number from 0 to 100, got {value!r}") from e
