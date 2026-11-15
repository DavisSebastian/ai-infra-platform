"""Week 5: which settings card is live right now, and how it shapes each request.

resolve_champion()  follows the `champion` alias in the MLflow registry to a model version, downloads that
                    version's card (written by week05-mlflow-registry/scripts/register.py), checks it, returns it.
apply_card()        applies a card to one chat request before it is forwarded.

    from mlflow import MlflowClient
    card = resolve_champion(MlflowClient("http://localhost:5001"))
    card.model_id, card.temperature   # -> "Qwen/Qwen2.5-0.5B-Instruct", 0.7

The client is passed in rather than created here (dependency injection): tests hand in a fake, and this module
never imports mlflow, so importing the gateway stays fast when the registry is off.
"""

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger("uvicorn.error")

MODEL_NAME = "gateway-llm"
CHAMPION = "champion"
CARD_PATH = "card/serving_config.yaml"  # where register.py puts the card inside each run


class ChampionError(Exception):
    """No usable card could be resolved, and there was no last-known-good card to fall back to."""


@dataclass(frozen=True)  # frozen: every request shares one card, so nobody may change it in place
class ServingCard:
    version: str  # the model version it came from, e.g. "2"
    model_id: str
    temperature: float
    max_tokens: int
    system_prompt: str


def resolve_champion(client, name: str = MODEL_NAME, last_good: ServingCard | None = None) -> ServingCard:
    """Return the card the `champion` alias points at.

    If anything goes wrong (MLflow down, no champion alias, a broken card):
      - with last_good: log a WARNING saying what went wrong, and return last_good (the same object)
      - without it:     raise ChampionError saying what went wrong
    """
    try:
        return _fetch(client, name)
    except ChampionError as e:
        if last_good is None:
            raise
        # Loud on purpose: a silent fallback would hide exactly the kind of failure Month 2 is about.
        log.warning("could not load the %s card, still serving v%s: %s", CHAMPION, last_good.version, e)
        return last_good


def _fetch(client, name: str) -> ServingCard:
    # 1. Network: two calls to MLflow. Any failure here (server down, alias missing, artifact missing) becomes a
    #    ChampionError, so callers handle one exception type, not mlflow's, requests' and the OS's.
    try:
        mv = client.get_model_version_by_alias(name, CHAMPION)
        # Download by the version we just got (its run), not by "models:/name@champion" again. If someone moves
        # the alias between the two calls, the card and the version number still match.
        with tempfile.TemporaryDirectory() as tmp:
            text = Path(client.download_artifacts(mv.run_id, CARD_PATH, tmp)).read_text()
    except Exception as e:  # deliberately broad: see above
        raise ChampionError(f"could not fetch the {CHAMPION} card of {name}: {e}") from e

    # 2. Parse and check: the card is data typed by a human, so trust nothing about its shape.
    try:
        raw = yaml.safe_load(text)  # safe_load: plain data only, never constructs arbitrary Python objects
    except yaml.YAMLError as e:
        raise ChampionError(f"v{mv.version}: the card is not valid YAML") from e
    if not isinstance(raw, dict):
        raise ChampionError(f"v{mv.version}: the card must be a mapping of settings")
    return ServingCard(version=str(mv.version), **_check(raw, f"v{mv.version}"))


def _check(raw: dict, where: str) -> dict:
    """The validity rules. Unknown extra keys are ignored, so a newer card still loads on an older gateway."""
    model_id, temperature = raw.get("model_id"), raw.get("temperature")
    max_tokens, system_prompt = raw.get("max_tokens"), raw.get("system_prompt")

    if not isinstance(model_id, str) or not model_id.strip():
        raise ChampionError(f"{where}: model_id must be a non-empty string, got {model_id!r}")
    # bool first: in Python True is an int (isinstance(True, int) is True), and YAML turns `true` into True.
    if isinstance(temperature, bool) or not isinstance(temperature, int | float) or not 0 <= temperature <= 2:
        raise ChampionError(f"{where}: temperature must be a number from 0 to 2, got {temperature!r}")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ChampionError(f"{where}: max_tokens must be a whole number above 0, got {max_tokens!r}")
    if not isinstance(system_prompt, str):
        raise ChampionError(f"{where}: system_prompt must be a string, got {system_prompt!r}")
    return {"model_id": model_id, "temperature": float(temperature), "max_tokens": max_tokens,
            "system_prompt": system_prompt}  # fmt: skip


def apply_card(body: bytes, card: ServingCard) -> bytes:
    """The card's settings applied to one chat request. A body that isn't a JSON chat request passes unchanged.

    model          always the card's: the registry decides what is served, not the caller
    max_tokens     capped at the card's (a caller may ask for fewer, never more)
    temperature    the card's, unless the caller chose one
    system prompt  the card's is added at the front, unless the caller sent their own
    """
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list):
        return body

    payload["model"] = card.model_id
    asked = payload.get("max_tokens")
    if isinstance(asked, int) and not isinstance(asked, bool) and 0 < asked < card.max_tokens:
        payload["max_tokens"] = asked
    else:
        payload["max_tokens"] = card.max_tokens
    payload.setdefault("temperature", card.temperature)
    if card.system_prompt and not any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
        payload["messages"] = [{"role": "system", "content": card.system_prompt}, *messages]
    return json.dumps(payload).encode()
