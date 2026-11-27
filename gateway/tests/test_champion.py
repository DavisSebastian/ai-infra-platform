"""What resolve_champion() must do (Week 5, step 4). The tests were written before the code."""

import logging

import pytest
from fake_registry import CARD_V1, CARD_V2, FakeRegistry
from gateway.champion import ChampionError, ServingCard, resolve_champion

LAST_GOOD = ServingCard(
    version="1",
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    temperature=0.7,
    max_tokens=64,
    system_prompt="You are a concise assistant. Answer in one short sentence.",
)


@pytest.fixture
def registry():
    """v1 and v2 registered, each with a valid card. No aliases yet."""
    return FakeRegistry({"1": CARD_V1, "2": CARD_V2})


# --- The happy path: follow the alias --------------------------------------------------------


def test_returns_the_card_champion_points_at(registry):
    registry.aliases["champion"] = "2"
    card = resolve_champion(registry)
    assert card == ServingCard(
        version="2",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        temperature=0.3,
        max_tokens=64,
        system_prompt="You are a concise assistant. Answer in one short sentence.",
    )


def test_follows_the_alias_when_it_moves(registry):
    registry.aliases["champion"] = "2"
    assert resolve_champion(registry).version == "2"
    registry.aliases["champion"] = "1"  # someone ran `release.py rollback`
    assert resolve_champion(registry).version == "1"


# --- Nothing to serve, and nothing to fall back to: fail with a clear message -----------------


def test_missing_alias_is_a_clear_error(registry):
    with pytest.raises(ChampionError, match="champion"):
        resolve_champion(registry)


def test_mlflow_down_is_a_champion_error_not_a_raw_mlflow_one(registry):
    # The caller (the gateway) should have to handle one exception type, not every library's.
    registry.aliases["champion"] = "2"
    registry.down = True
    with pytest.raises(ChampionError):
        resolve_champion(registry)


# --- Broken cards are rejected ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("card_text", "complaint"),
    [
        pytest.param(CARD_V1.replace("model_id: Qwen/Qwen2.5-0.5B-Instruct\n", ""), "model_id", id="no model_id"),
        pytest.param(CARD_V1.replace("Qwen/Qwen2.5-0.5B-Instruct", '""'), "model_id", id="empty model_id"),
        pytest.param(CARD_V1.replace("0.7", "5.0"), "temperature", id="temperature too high"),
        pytest.param(CARD_V1.replace("0.7", "-0.1"), "temperature", id="temperature negative"),
        pytest.param(CARD_V1.replace("0.7", "hot"), "temperature", id="temperature not a number"),
        pytest.param(CARD_V1.replace("64", "0"), "max_tokens", id="max_tokens zero"),
        pytest.param(CARD_V1.replace("64", "64.5"), "max_tokens", id="max_tokens not whole"),
        # YAML reads `true` as a bool, and in Python bool is a subclass of int: isinstance(True, int) is True.
        pytest.param(CARD_V1.replace("64", "true"), "max_tokens", id="max_tokens is a bool"),
        pytest.param(CARD_V1.split("system_prompt")[0], "system_prompt", id="no system_prompt"),
        pytest.param("- just\n- a list\n", "mapping", id="not a mapping"),
        pytest.param("model_id: [unclosed\n", "not valid YAML", id="not valid YAML"),
    ],
)
def test_broken_card_is_rejected(registry, card_text, complaint):
    registry.cards["2"] = card_text
    registry.aliases["champion"] = "2"
    with pytest.raises(ChampionError, match=complaint):
        resolve_champion(registry)


@pytest.mark.parametrize("temperature", ["0", "0.0", "2", "1"])
def test_temperature_edges_and_whole_numbers_are_fine(registry, temperature):
    registry.cards["2"] = CARD_V1.replace("0.7", temperature)
    registry.aliases["champion"] = "2"
    assert resolve_champion(registry).temperature == float(temperature)


def test_unknown_extra_fields_are_ignored(registry):
    # Forward compatible: a newer card with an extra setting still loads on an older gateway.
    # (Like Jackson with FAIL_ON_UNKNOWN_PROPERTIES = false.)
    registry.cards["2"] = CARD_V1 + "top_p: 0.9\n"
    registry.aliases["champion"] = "2"
    assert resolve_champion(registry).model_id == "Qwen/Qwen2.5-0.5B-Instruct"


# --- With a last-known-good card: keep serving it, and say so loudly ---------------------------
# Falling back silently would hide the problem: exactly the "ML fails silently" trap of Month 2.


def test_mlflow_down_keeps_last_good_and_warns(registry, caplog):
    registry.aliases["champion"] = "2"
    registry.down = True
    with caplog.at_level(logging.WARNING):
        assert resolve_champion(registry, last_good=LAST_GOOD) is LAST_GOOD
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_missing_alias_keeps_last_good_and_warns(registry, caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_champion(registry, last_good=LAST_GOOD) is LAST_GOOD
    assert any(r.levelno == logging.WARNING and "champion" in r.getMessage() for r in caplog.records)


def test_broken_card_keeps_last_good_and_the_warning_says_why(registry, caplog):
    registry.cards["2"] = CARD_V1.replace("0.7", "5.0")
    registry.aliases["champion"] = "2"
    with caplog.at_level(logging.WARNING):
        assert resolve_champion(registry, last_good=LAST_GOOD) is LAST_GOOD
    assert any(r.levelno == logging.WARNING and "temperature" in r.getMessage() for r in caplog.records)


def test_a_good_champion_replaces_last_good(registry):
    registry.aliases["champion"] = "2"
    assert resolve_champion(registry, last_good=LAST_GOOD).version == "2"
