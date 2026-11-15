"""Week 6: the gateway in front of TWO model servers, blue and green, moving traffic between them.

Each fake model server puts its name in the response's "model" field, so a test can see which side answered.
"""

import json

import gateway.fake_upstream as fake_upstream
import gateway.main as gateway
import httpx
import pytest
from conftest import serve, stop

TOKEN = "test-admin-token"
CHAT = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}


@pytest.fixture
def servers():
    """Two independent fake model servers: {"blue": url, "green": url}."""
    running = {name: serve(fake_upstream.create_app(name)) for name in ("blue", "green")}
    yield {name: url for name, (_, _, url) in running.items()}
    for srv, thread, _ in running.values():
        stop(srv, thread)


@pytest.fixture
def blue_green(start_gateway, servers, monkeypatch):
    """The gateway with blue = UPSTREAM_URL and green = GREEN_URL. Starts all blue."""
    monkeypatch.setattr(gateway, "GREEN_URL", servers["green"])
    monkeypatch.setattr(gateway, "ADMIN_TOKEN", TOKEN)
    return start_gateway(upstream_url=servers["blue"])


async def served_by(url: str, n: int = 1) -> list[str]:
    """Send n chat requests; return which server answered each one."""
    async with httpx.AsyncClient(timeout=10) as c:
        return [(await c.post(f"{url}/v1/chat/completions", json=CHAT)).json()["model"] for _ in range(n)]


async def set_green(url: str, body, token: str | None = TOKEN) -> httpx.Response:
    headers = {"x-admin-token": token} if token is not None else {}
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    async with httpx.AsyncClient(timeout=10) as c:
        return await c.post(f"{url}/admin/traffic", content=content, headers=headers)


async def traffic(url: str) -> dict:
    async with httpx.AsyncClient() as c:
        return (await c.get(f"{url}/health")).json()["traffic"]


# --- Moving traffic -------------------------------------------------------------------------------------


async def test_starts_all_blue(blue_green):
    assert set(await served_by(blue_green, 10)) == {"blue"}
    t = await traffic(blue_green)
    assert (t["blue"]["weight"], t["green"]["weight"]) == (100, 0)


async def test_green_100_sends_every_new_request_to_green(blue_green):
    r = await set_green(blue_green, {"green": 100})
    assert r.json() == {"green": 100, "blue": 0, "previous": 0, "changed": True}
    assert set(await served_by(blue_green, 10)) == {"green"}


async def test_green_0_is_an_instant_rollback(blue_green):
    await set_green(blue_green, {"green": 100})
    await set_green(blue_green, {"green": 0})
    assert set(await served_by(blue_green, 10)) == {"blue"}


async def test_a_split_sends_traffic_to_both_and_health_counts_it(blue_green):
    await set_green(blue_green, {"green": 50})
    sides = await served_by(blue_green, 40)
    assert sides.count("blue") and sides.count("green")  # all 40 on one side at 50/50: odds 1 in 500 billion
    t = await traffic(blue_green)
    assert (t["blue"]["routed"], t["green"]["routed"]) == (sides.count("blue"), sides.count("green"))


async def test_setting_the_same_split_again_changes_nothing(blue_green):
    await set_green(blue_green, {"green": 10})
    assert (await set_green(blue_green, {"green": 10})).json()["changed"] is False


# --- The point of Blue/Green: a switch never cuts a request off ------------------------------------------


async def test_a_running_request_finishes_on_the_side_it_started_on(start_gateway, servers, monkeypatch):
    # One slot and NO queue per side: if both sides shared one admission limit, the green request below would be
    # turned away (503) while the blue stream holds the slot.
    monkeypatch.setattr(gateway, "MAX_IN_FLIGHT", 1)
    monkeypatch.setattr(gateway, "MAX_QUEUE", 0)
    monkeypatch.setattr(gateway, "GREEN_URL", servers["green"])
    monkeypatch.setattr(gateway, "ADMIN_TOKEN", TOKEN)
    monkeypatch.setattr(fake_upstream, "CHUNK_DELAY", 0.05)  # 20 chunks: a ~1 s stream
    url = start_gateway(upstream_url=servers["blue"])

    async with httpx.AsyncClient(timeout=10) as c:
        body = {**CHAT, "max_tokens": 20, "stream": True}
        async with c.stream("POST", f"{url}/v1/chat/completions", json=body) as stream:
            lines = stream.aiter_lines()
            first = await anext(lines)  # the stream has started, on blue

            await set_green(url, {"green": 100})  # switch while it is streaming
            assert await served_by(url) == ["green"]  # new requests: green, admitted at once
            assert (await traffic(url))["blue"]["in_flight"] == 1  # blue is draining

            rest = [line async for line in lines]

    events = [json.loads(x.removeprefix("data: ")) for x in [first, *rest] if x.startswith("data: {")]
    assert {e["model"] for e in events} == {"blue"}  # every chunk came from blue, start to finish
    assert "data: [DONE]" in rest  # it completed, not cut off
    assert (await traffic(url))["blue"]["in_flight"] == 0  # drained: blue is safe to stop


# --- Guard rails ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("token", [None, "", "wrong-token"])
async def test_moving_traffic_needs_the_admin_token(blue_green, token):
    assert (await set_green(blue_green, {"green": 100}, token)).status_code == 403
    assert (await traffic(blue_green))["green"]["weight"] == 0


@pytest.mark.parametrize("body", [{"green": 150}, {"green": -1}, {"green": True}, {"green": "10"}, {"green": None},
                                  {}, [], b"not json"])  # fmt: skip
async def test_a_bad_split_is_refused_and_nothing_moves(blue_green, body):
    await set_green(blue_green, {"green": 10})
    r = await set_green(blue_green, body)
    assert r.status_code == 400 and r.json()["error"]["type"] == "bad_split"
    assert (await traffic(blue_green))["green"]["weight"] == 10


async def test_without_green_url_there_is_only_blue(start_gateway, monkeypatch):
    monkeypatch.setattr(gateway, "ADMIN_TOKEN", TOKEN)
    url = start_gateway()
    assert list(await traffic(url)) == ["blue"]
    assert (await set_green(url, {"green": 10})).status_code == 409
    assert (await set_green(url, {"green": 0})).status_code == 200  # all blue is always allowed


# --- Startup --------------------------------------------------------------------------------------------


async def test_the_starting_split_comes_from_traffic_green(start_gateway, servers, monkeypatch):
    monkeypatch.setattr(gateway, "GREEN_URL", servers["green"])
    monkeypatch.setattr(gateway, "TRAFFIC_GREEN", "100")
    url = start_gateway(upstream_url=servers["blue"])
    assert set(await served_by(url, 5)) == {"green"}


@pytest.mark.parametrize(("traffic_green", "green_url", "complaint"), [
    ("ten", "http://green:9000", "TRAFFIC_GREEN"),
    ("10", None, "GREEN_URL is not set"),
])  # fmt: skip
async def test_a_bad_traffic_setting_stops_startup(monkeypatch, traffic_green, green_url, complaint):
    monkeypatch.setattr(gateway, "TRAFFIC_GREEN", traffic_green)
    monkeypatch.setattr(gateway, "GREEN_URL", green_url)
    with pytest.raises(ValueError, match=complaint):
        async with gateway.lifespan(gateway.app):
            pass
