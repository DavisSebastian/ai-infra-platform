"""A fake model for tests: same interface as app.model.LLM / Stream, but no torch and no GPU.

It is not a trivial stub on purpose. FakeStream produces pieces on a real background thread and stops when
cancel() sets a flag, exactly like the real Stream does with generate(). That is what lets the tests exercise
the server's real concurrency behaviour (lock, cancellation, cleanup) without a model.
"""

import asyncio
import threading
import time
from dataclasses import dataclass

_END = object()


@dataclass
class FakeCompletion:
    text: str
    prompt_tokens: int
    completion_tokens: int


class FakeStream:
    def __init__(self, llm: "FakeLLM", n_pieces: int):
        self._llm = llm
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._stop = threading.Event()
        self.prompt_tokens = 10
        self.completion_tokens = 0
        self.error = None
        self.cancel_called = False
        self.text = self._pieces()
        self._thread = threading.Thread(target=self._run, args=(n_pieces,), daemon=True)
        self._thread.start()

    async def _pieces(self):
        while (item := await self._queue.get()) is not _END:
            yield item

    def cancel(self) -> None:
        self.cancel_called = True
        self._stop.set()

    def _run(self, n_pieces: int) -> None:
        self._llm.generation_started()
        try:
            for i in range(n_pieces):
                if self._stop.is_set():
                    break
                time.sleep(self._llm.delay)
                self._loop.call_soon_threadsafe(self._queue.put_nowait, f"w{i} ")
                self.completion_tokens += 1
        finally:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, _END)
            self._llm.generation_finished()

    async def wait(self) -> None:
        await asyncio.to_thread(self._thread.join)


class FakeLLM:
    model_id = "fake-model"
    device = "cpu"

    def __init__(self, n_pieces: int = 5, delay: float = 0.0):
        self.n_pieces = n_pieces
        self.delay = delay  # seconds per piece, so tests can make generation as slow as they need
        self.active = 0
        self.max_active = 0  # the most generations ever running at the same moment
        self.starts: list[float] = []
        self.finishes: list[float] = []
        self.streams: list[FakeStream] = []
        self._lock = threading.Lock()

    def generation_started(self) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.starts.append(time.perf_counter())

    def generation_finished(self) -> None:
        with self._lock:
            self.active -= 1
            self.finishes.append(time.perf_counter())

    def start_stream(self, messages: list[dict], max_new_tokens: int) -> FakeStream:
        stream = FakeStream(self, min(self.n_pieces, max_new_tokens))
        self.streams.append(stream)
        return stream

    def generate(self, messages: list[dict], max_new_tokens: int) -> FakeCompletion:
        self.generation_started()
        try:
            n = min(self.n_pieces, max_new_tokens)
            time.sleep(self.delay * n)
            return FakeCompletion(text="".join(f"w{i} " for i in range(n)), prompt_tokens=10, completion_tokens=n)
        finally:
            self.generation_finished()
