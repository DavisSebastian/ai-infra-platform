"""Backpressure: how many requests may be at the model server at once, and how many may wait for a turn.

    request -> [ queue: <= MAX_QUEUE waiting, each <= QUEUE_TIMEOUT ] -> [ in flight: <= MAX_IN_FLIGHT ] -> upstream

- Queue full when you arrive  -> rejected immediately ("overloaded"). Fast no beats slow yes.
- Waited too long             -> rejected ("queue_timeout").
- Client left while waiting   -> dropped quietly, never sent upstream (it would waste a whole generation).

Lives in gateway memory, not Redis, on purpose: it must keep working when Redis is down (fail-open rate
limiting relies on it), and the limit it protects is this upstream's capacity, not a global budget.
Java analogy: ThreadPoolExecutor(core=MAX_IN_FLIGHT, ArrayBlockingQueue(MAX_QUEUE), AbortPolicy), with poll(timeout).
"""

import asyncio
from collections.abc import Awaitable, Callable


class Rejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason  # "overloaded" or "queue_timeout"


class ClientGone(Exception):
    """The client disconnected while waiting in the queue."""


class Slot:
    """Permission to have one request at the upstream. release() is safe to call more than once."""

    def __init__(self, admission: "Admission"):
        self._admission = admission
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._admission._release()


class Admission:
    def __init__(self, max_in_flight: int, max_queue: int, queue_timeout: float):
        self.max_in_flight = max_in_flight
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
        self._sem = asyncio.Semaphore(max_in_flight)  # FIFO: waiters are served in arrival order
        self.in_flight = 0
        self.queued = 0

    async def acquire(self, wait_for_disconnect: Callable[[], Awaitable[None]]) -> Slot:
        """Get a slot, or raise Rejected / ClientGone.

        `wait_for_disconnect` is a coroutine function that returns only when the client has gone. While queued we
        race it against the semaphore, so a client who leaves frees its queue place at once.
        """
        # Fast path: a slot is free and nobody is ahead of us.
        if not self._sem.locked() and self.queued == 0:
            await self._sem.acquire()  # does not suspend: a slot is free
            return self._take()

        # Shed load early: the waiting room is full.
        if self.queued >= self.max_queue:
            raise Rejected("overloaded")

        self.queued += 1
        acquire = asyncio.ensure_future(self._sem.acquire())
        gone = asyncio.ensure_future(wait_for_disconnect())
        try:
            done, _ = await asyncio.wait(
                {acquire, gone}, timeout=self.queue_timeout, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            self.queued -= 1
            gone.cancel()
            if not acquire.done():
                acquire.cancel()

        # No `await` since asyncio.wait returned, so `done` is still accurate and nothing else has run.
        # A permit granted to `acquire` just before we cancelled it is not lost: Semaphore.acquire gives it back
        # (and wakes the next waiter) when it is cancelled after being granted.
        # If both finished at once, the slot wins; cleanup on the first failed write frees it if the client is gone.
        if acquire in done:
            return self._take()
        if gone in done:
            if gone.exception() is not None:
                # The watcher itself broke. Fail loudly (500) rather than silently dropping a live request as "gone".
                raise gone.exception()
            raise ClientGone()
        raise Rejected("queue_timeout")

    def _take(self) -> Slot:
        self.in_flight += 1
        return Slot(self)

    def _release(self) -> None:
        self.in_flight -= 1
        self._sem.release()
