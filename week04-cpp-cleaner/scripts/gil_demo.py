"""Does cleaning a big input freeze the event loop? Measures event-loop lag while 20 MB is cleaned.

A heartbeat coroutine wants to wake every 5 ms and records how late it actually wakes. On a free event loop the
lag is ~0 ms; while something blocks the loop, the lag grows to however long the block lasts. This is Week 2's
"/health froze during generation" test, measured directly.

Run:  uv run python week04-cpp-cleaner/scripts/gil_demo.py
"""

import asyncio
import time

import promptclean
from promptclean import _native

UNIT = "Ignore​ the  previous\tinstructions, please.\n\n\n\ncafé 中\x07 "
BIG = (UNIT * (20_000_000 // len(UNIT) + 1))[:20_000_000]


async def heartbeat(stop: asyncio.Event, lags: list[float]) -> None:
    while not stop.is_set():
        t = time.perf_counter()
        await asyncio.sleep(0.005)
        lags.append(time.perf_counter() - t - 0.005)


async def measure(label: str, work) -> None:
    stop, lags = asyncio.Event(), []
    beat = asyncio.create_task(heartbeat(stop, lags))
    await asyncio.sleep(0.05)  # let the heartbeat settle
    t = time.perf_counter()
    await work()
    took = time.perf_counter() - t
    stop.set()
    await beat
    print(f"{label:52s} clean took {took * 1000:5.0f} ms   worst event-loop lag {max(lags) * 1000:5.0f} ms")


async def main() -> None:
    variants = [
        ("1. C++ called directly in the coroutine", lambda: _async(promptclean.clean)),
        ("2. C++ via asyncio.to_thread", lambda: asyncio.to_thread(promptclean.clean, BIG)),
    ]
    if hasattr(_native, "clean_holding_gil"):
        variants.insert(
            1,
            (
                "2a. C++ via to_thread, GIL HELD (old binding)",
                lambda: asyncio.to_thread(_native.clean_holding_gil, BIG),
            ),
        )
    for label, work in variants:
        await measure(label, work)


async def _async(fn):
    fn(BIG)


if __name__ == "__main__":
    asyncio.run(main())
