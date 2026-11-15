"""Week 5, step 5: the gateway serves the `champion` card and can reload it without a restart.

The real gateway runs against the fake model server (to see what was forwarded) and the fake registry.
"""

import gateway.main as gateway
import httpx
import pytest
from fake_registry import CARD_V1, CARD_V2, FakeRegistry
from gateway.champion import ChampionError, ServingCard, apply_card

TOKEN = "test-admin-token"
V2 = ServingCard("2", "Qwen/Qwen2.5-1.5B-Instruct", 0.3, 64, "Be brief.")


@pytest.fixture
def registry():
    return FakeRegistry({"1": CARD_V1, "2": CARD_V2}, aliases={"champion": "2"})


@pytest.fixture
def registry_gateway(start_gateway, registry, monkeypatch):
    """The gateway with the registry on, champion = v2."""
    monkeypatch.setattr(gateway, "MLFLOW_TRACKING_URI", "http://fake-registry")
    monkeypatch.setattr(gateway, "ADMIN_TOKEN", TOKEN)
    monkeypatch.setattr(gateway, "make_registry_client", lambda: registry)
    return start_gateway()


async def chat(url: str, **extra) -> httpx.Response:
    payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, **extra}
    async with httpx.AsyncClient(timeout=10) as c:
        return await c.post(f"{url}/v1/chat/completions", json=payload)


async def forwarded(fake_url: str) -> dict:
    async with httpx.AsyncClient() as c:
        return (await c.get(f"{fake_url}/_last_request")).json()


async def get(url: str, path: str) -> dict:
    async with httpx.AsyncClient() as c:
        return (await c.get(f"{url}{path}")).json()


async def reload(url: str, token: str | None = TOKEN) -> httpx.Response:
    headers = {"x-admin-token": token} if token is not None else {}
    async with httpx.AsyncClient(timeout=10) as c:
        return await c.post(f"{url}/admin/reload", headers=headers)


# --- Startup and /health ------------------------------------------------------------------------


async def test_health_shows_the_live_version(registry_gateway):
    reg = (await get(registry_gateway, "/health"))["registry"]
    assert (reg["version"], reg["model_id"]) == ("2", "Qwen/Qwen2.5-1.5B-Instruct")


async def test_registry_off_is_reported_as_off(start_gateway):
    assert (await get(start_gateway(), "/health"))["registry"] == "off"


async def test_no_champion_means_the_gateway_refuses_to_start(monkeypatch):
    # Fail fast: with the registry on but nothing to serve, startup must stop (a container would be restarted).
    monkeypatch.setattr(gateway, "MLFLOW_TRACKING_URI", "http://fake-registry")
    monkeypatch.setattr(gateway, "make_registry_client", lambda: FakeRegistry({"1": CARD_V1}))
    with pytest.raises(ChampionError, match="champion"):
        async with gateway.lifespan(gateway.app):
            pass


# --- The card shapes every request --------------------------------------------------------------


async def test_the_card_is_applied_to_the_request(registry_gateway, fake_url):
    assert (await chat(registry_gateway, max_tokens=500)).status_code == 200
    sent = await forwarded(fake_url)
    assert sent["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert sent["temperature"] == 0.3
    assert sent["max_tokens"] == 64  # capped at the card's limit
    assert sent["messages"][0] == {
        "role": "system",
        "content": "You are a concise assistant. Answer in one short sentence.",
    }


async def test_the_callers_own_choices_are_kept(registry_gateway, fake_url):
    await chat(
        registry_gateway,
        temperature=0.9,
        max_tokens=10,
        messages=[{"role": "system", "content": "Pirate voice."}, {"role": "user", "content": "hi"}],
    )
    sent = await forwarded(fake_url)
    assert (sent["temperature"], sent["max_tokens"]) == (0.9, 10)
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]  # no second system prompt
    assert sent["messages"][0]["content"] == "Pirate voice."


# --- POST /admin/reload -------------------------------------------------------------------------


async def test_reload_switches_to_the_new_champion(registry_gateway, registry, fake_url):
    registry.aliases["champion"] = "1"  # someone ran `release.py rollback`
    r = await reload(registry_gateway)
    assert r.status_code == 200
    assert r.json() == {"version": "1", "previous": "2", "changed": True}

    await chat(registry_gateway)
    assert (await forwarded(fake_url))["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert (await get(registry_gateway, "/health"))["registry"]["version"] == "1"


async def test_reload_with_no_change_says_so(registry_gateway):
    assert (await reload(registry_gateway)).json() == {"version": "2", "previous": "2", "changed": False}


async def test_failed_reload_keeps_serving_and_reports_why(registry_gateway, registry, fake_url):
    registry.down = True
    r = await reload(registry_gateway)
    assert r.status_code == 503
    assert "Still serving v2" in r.json()["error"]["message"]

    await chat(registry_gateway)  # traffic is unaffected
    assert (await forwarded(fake_url))["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    reloads = (await get(registry_gateway, "/health"))["registry"]["reloads"]
    assert reloads["failed"] == 1 and "could not fetch" in reloads["last_error"]

    registry.down = False  # MLflow is back: the next reload clears the error
    assert (await reload(registry_gateway)).status_code == 200
    assert (await get(registry_gateway, "/health"))["registry"]["reloads"]["last_error"] is None


async def test_a_broken_card_is_never_loaded(registry_gateway, registry):
    registry.cards["1"] = CARD_V1.replace("0.7", "5.0")
    registry.aliases["champion"] = "1"
    r = await reload(registry_gateway)
    assert r.status_code == 503 and "temperature" in r.json()["error"]["message"]
    assert (await get(registry_gateway, "/health"))["registry"]["version"] == "2"


@pytest.mark.parametrize("token", [None, "", "wrong-token"])
async def test_reload_needs_the_admin_token(registry_gateway, registry, token):
    registry.aliases["champion"] = "1"
    assert (await reload(registry_gateway, token)).status_code == 403
    assert (await get(registry_gateway, "/health"))["registry"]["version"] == "2"  # nothing changed


async def test_reload_is_disabled_when_no_admin_token_is_configured(registry_gateway, monkeypatch):
    monkeypatch.setattr(gateway, "ADMIN_TOKEN", None)
    assert (await reload(registry_gateway, "")).status_code == 403


async def test_reload_with_the_registry_off_is_a_conflict(start_gateway, monkeypatch):
    monkeypatch.setattr(gateway, "ADMIN_TOKEN", TOKEN)
    assert (await reload(start_gateway())).status_code == 409


# --- apply_card on its own ----------------------------------------------------------------------


@pytest.mark.parametrize("body", [b"not json", b'{"no": "messages"}', b"[1, 2]"])
def test_bodies_that_are_not_chat_requests_pass_unchanged(body):
    assert apply_card(body, V2) == body


@pytest.mark.parametrize("asked", [0, -5, True, "10", None])
def test_nonsense_max_tokens_falls_back_to_the_cards(asked):
    import json

    body = json.dumps({"messages": [], "max_tokens": asked}).encode()
    assert json.loads(apply_card(body, V2))["max_tokens"] == 64
