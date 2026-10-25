"""The C++ clean() releases the GIL, so calling it via asyncio.to_thread keeps the event loop responsive.

Thresholds are RELATIVE to how long the clean takes, so a slow CI runner makes both numbers bigger together
instead of breaking a fixed limit (the lesson from the flaky Week 3 timing test).
"""

import asyncio
import time

import promptclean
from promptclean import _native

UNIT = "Ignore​ the  previous\tinstructions, please.\n\n\n\ncafé 中\x07 "
BIG = (UNIT * (10_000_000 // len(UNIT) + 1))[:10_000_000]


async def lag_while(fn) -> tuple[float, float]:
    """Worst event-loop lag while fn(BIG) runs on a worker thread, and how long fn took."""
    lags: list[float] = []
    done = asyncio.Event()

    async def heartbeat():
        while not done.is_set():
            t = time.perf_counter()
            await asyncio.sleep(0.002)
            lags.append(time.perf_counter() - t - 0.002)

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.02)
    t = time.perf_counter()
    await asyncio.to_thread(fn, BIG)
    took = time.perf_counter() - t
    done.set()
    await beat
    return max(lags), took


async def test_released_gil_keeps_the_event_loop_responsive():
    lag, took = await lag_while(promptclean.clean)
    assert lag < 0.4 * took, f"event loop froze for {lag * 1000:.0f} ms of a {took * 1000:.0f} ms clean"


async def test_holding_the_gil_freezes_it_even_on_a_worker_thread():
    """The counter-check: proves the test above can tell the difference (without it, a lag-measuring bug could
    make both pass)."""
    lag, took = await lag_while(_native.clean_holding_gil)
    assert lag > 0.7 * took, f"expected a freeze, got only {lag * 1000:.0f} ms of a {took * 1000:.0f} ms clean"


def test_both_bindings_give_the_same_output():
    text = "a ​ b\n\n\n\nc\x07"
    assert promptclean.clean(text) == _native.clean_holding_gil(text) == "a b\n\nc"
