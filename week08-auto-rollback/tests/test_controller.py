"""Week 8: the controller loop, with every dependency faked (no gateway, no Redis, no model, no MLflow)."""

import httpx
import pandas as pd
import pytest
import redis
from controller import Canary, Controller
from verdict import Verdict


class World:
    """Stands in for the gateway, the request log and the registry, and records what the controller did."""

    def __init__(self, green=10, green_card="3", blue_card="1", n_green=80, n_blue=90, verdict="alert"):
        self.canary = Canary(green, green_card, blue_card)
        self.n = {"green": n_green, "blue": n_blue}
        self.verdict = verdict
        self.rolled_back: set[str | None] = set()
        self.traffic_calls: list[int] = []
        self.audits: list[tuple] = []
        self.judged: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        self.reads: list[tuple] = []
        self.gateway_down = self.redis_down = False

    def read_canary(self) -> Canary:
        if self.gateway_down:
            raise httpx.ConnectError("connection refused")
        return self.canary

    def read_records(self, side: str, card: str | None, last: int, since: str = "") -> pd.DataFrame:
        if self.redis_down:
            raise redis.ConnectionError("Redis is down")
        self.reads.append((side, card, last, since))
        start = 0 if side == "blue" else 30  # blue has older records too; green's start later
        n = self.n[side]
        ts = [f"2026-11-27T10:{(start + i) // 60:02d}:{(start + i) % 60:02d}" for i in range(n)]
        ts = [t for t in ts if t >= since]  # like the real reader: only traffic since the split last changed
        return pd.DataFrame({"ts": ts, "prompt": ["hi"] * len(ts), "answer": ["hello"] * len(ts),
                             "ttft_ms": [30.0] * len(ts)})  # fmt: skip

    def judge(self, green: pd.DataFrame, blue: pd.DataFrame) -> Verdict:
        self.judged.append((green, blue))
        return Verdict(self.verdict, (f"test verdict {self.verdict}",))

    def set_traffic(self, green: int) -> None:
        self.traffic_calls.append(green)
        self.canary = Canary(green, self.canary.green_card, self.canary.blue_card, self.canary.since)

    def is_rolled_back(self, card: str | None) -> bool:
        return card in self.rolled_back

    def record_rollback(self, canary, verdict, action) -> None:
        self.audits.append((canary, verdict, action))
        self.rolled_back.add(canary.green_card)


def controller(world: World, promote: bool = False) -> Controller:
    return Controller(world.read_canary, world.read_records, world.judge, world.set_traffic, world.is_rolled_back,
                      world.record_rollback, promote=promote, window=150, min_records=50)  # fmt: skip


# --- The rollback --------------------------------------------------------------------------------------


def test_an_alert_rolls_green_back_and_records_why():
    world = World(verdict="alert")
    tick = controller(world).tick()
    assert world.traffic_calls == [0]
    assert (tick.action, tick.target, tick.verdict) == ("set_traffic", 0, "alert")
    [(canary, verdict, _)] = world.audits
    assert canary.green_card == "3" and verdict.level == "alert"


def test_after_a_rollback_it_stops_and_never_flips_back():
    world = World(verdict="alert")
    c = controller(world, promote=True)
    c.tick()
    world.verdict = "ok"  # even if later traffic looks fine
    for _ in range(3):
        assert c.tick().action == "done"
    assert world.traffic_calls == [0] and len(world.audits) == 1


def test_a_restarted_controller_remembers_the_rollback_through_the_registry():
    world = World(verdict="alert")
    controller(world).tick()
    world.canary = Canary(10, "3", "1")  # someone sets green back to 10% by hand
    assert controller(world).tick().action == "done"  # a NEW controller: the tag, not its memory, says so
    assert world.traffic_calls == [0]


def test_a_new_deploy_is_watched_again():
    world = World(verdict="alert")
    controller(world).tick()
    world.canary = Canary(10, "4", "1")  # v4 deployed to green
    world.verdict = "ok"
    assert controller(world).tick().action == "hold"


# --- What it reads -------------------------------------------------------------------------------------


def test_it_reads_only_greens_current_card():
    world = World()
    controller(world).tick()
    assert world.reads[0] == ("green", "3", 150, "")


def test_blue_is_compared_over_the_same_period_only():
    world = World()
    controller(world).tick()
    green, blue = world.judged[0]
    assert len(blue) < world.n["blue"]  # blue's older records were dropped
    assert blue["ts"].min() >= green["ts"].min()


def test_too_few_green_records_holds_without_running_the_checks():
    world = World(n_green=20)
    tick = controller(world).tick()
    assert (tick.action, tick.verdict) == ("hold", "insufficient_data")
    assert world.judged == [] and world.traffic_calls == []


def test_no_canary_means_nothing_is_read():
    world = World(green=0)
    assert controller(world).tick().action == "done"
    assert world.reads == []


# --- Never act blind -----------------------------------------------------------------------------------


def test_an_unreachable_gateway_means_hold():
    world = World()
    world.gateway_down = True
    tick = controller(world).tick()
    assert tick.action == "hold" and "gateway" in tick.why
    assert world.traffic_calls == []


def test_an_unreachable_request_log_means_hold():
    world = World()
    world.redis_down = True
    tick = controller(world).tick()
    assert tick.action == "hold" and "request log" in tick.why
    assert world.traffic_calls == []


# --- Promotion -----------------------------------------------------------------------------------------


def test_ok_with_promotion_steps_up_and_records_no_rollback():
    world = World(verdict="ok")
    c = controller(world, promote=True)
    assert [c.tick().target for _ in range(4)] == [25, 50, 100, None]
    assert world.traffic_calls == [25, 50, 100] and world.audits == []


def test_each_canary_step_is_judged_only_on_traffic_since_the_step():
    # 80 green records exist, but only 20 arrived since the split last changed: not enough evidence for this step.
    world = World(verdict="ok", n_green=80)
    world.canary = Canary(25, "3", "1", since="2026-11-27T10:01:30")
    tick = controller(world, promote=True).tick()
    assert (tick.action, tick.verdict, tick.n_green) == ("hold", "insufficient_data", 20)
    assert world.traffic_calls == []


@pytest.mark.parametrize("verdict", ["warn", "insufficient_data"])
def test_warnings_hold_even_with_promotion(verdict):
    world = World(verdict=verdict)
    assert controller(world, promote=True).tick().action == "hold"
    assert world.traffic_calls == []
