import threading
import time

import app.main as main
import httpx
import pytest
import pytest_asyncio
import uvicorn
from fakes import FakeLLM


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def server(fake_llm, monkeypatch):
    """The real app, served by real uvicorn on a free port, with the fake model swapped in.

    It runs in its own thread with its own event loop. That matters: if a bug blocks the server's event loop,
    the test client (on a different loop) can still measure how long the server took to answer.
    Spring analogy: @SpringBootTest(webEnvironment = RANDOM_PORT) with a @MockBean model.
    """
    monkeypatch.setattr(main, "load_llm", lambda: fake_llm)
    config = uvicorn.Config(main.app, host="127.0.0.1", port=0, log_level="warning", timeout_graceful_shutdown=2)
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10
    while not srv.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.01)

    port = srv.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"

    srv.should_exit = True
    thread.join(timeout=5)


@pytest_asyncio.fixture
async def client(server):
    async with httpx.AsyncClient(base_url=server, timeout=10) as c:
        yield c
