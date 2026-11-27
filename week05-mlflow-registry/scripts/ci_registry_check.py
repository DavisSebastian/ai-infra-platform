"""Week 5, step 6: check the registry against a REAL MLflow server, in two halves with a restart in between.

    REGISTRY_MODEL=gateway-llm-ci uv run python week05-mlflow-registry/scripts/ci_registry_check.py seed
    (cd week05-mlflow-registry && docker compose down && docker compose up -d --wait)   # volumes kept
    REGISTRY_MODEL=gateway-llm-ci uv run python week05-mlflow-registry/scripts/ci_registry_check.py verify

seed    registers three cards with register.py's own code (no scoring: CI has no GPU or model, so the eval
        verdicts are set directly), then promotes with release.py: v3 (eval=failed) must be refused, v1 then v2.
verify  after the restart: the aliases survived, resolve_champion() returns v2's card from the real server, and
        the real server raises the error codes the tests' fake registries copy.

Run it under its own REGISTRY_MODEL (CI uses gateway-llm-ci) so your real gateway-llm versions are never touched.
"""

import sys
from pathlib import Path

import yaml
from gateway.champion import ServingCard, resolve_champion
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from register import MODEL_NAME, TRACKING_URI, log_and_register
from release import ReleaseError, current, promote, status

CARDS = Path(__file__).resolve().parent.parent / "cards"


def check(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        sys.exit(1)


def seed(client: MlflowClient) -> None:
    check(not client.search_model_versions(f"name='{MODEL_NAME}'"), f"{MODEL_NAME} starts empty")
    plan = [("v1-qwen-0.5b.yaml", "passed"), ("v2-qwen-1.5b.yaml", "passed"), ("v2-qwen-1.5b.yaml", "failed")]
    for card_file, verdict in plan:
        card = yaml.safe_load((CARDS / card_file).read_text())
        version, _ = log_and_register(card, card_file, {}, [], verdict, extra_params={"ci_fixture": True})
        print(f"  registered v{version.version} ({card_file}, eval={verdict})")

    try:
        promote(client, "3", MODEL_NAME, by="ci")
        check(False, "v3 (eval=failed) is refused")
    except ReleaseError:
        check(True, "v3 (eval=failed) is refused")
    promote(client, "1", MODEL_NAME, by="ci")
    promote(client, "2", MODEL_NAME, by="ci")
    check(current(client, MODEL_NAME) == {"champion": "2", "fallback": "1"}, "champion=v2, fallback=v1")
    print(status(client, MODEL_NAME))


def error_code(call) -> str | None:
    try:
        call()
    except MlflowException as e:
        return e.error_code
    return None


def verify(client: MlflowClient) -> None:
    aliases = current(client, MODEL_NAME)
    check(aliases == {"champion": "2", "fallback": "1"}, f"aliases survived the restart: {aliases}")

    card = resolve_champion(client, MODEL_NAME)
    v2 = yaml.safe_load((CARDS / "v2-qwen-1.5b.yaml").read_text())
    check(card == ServingCard(version="2", **v2), f"resolve_champion() -> v{card.version} {card.model_id}")

    # The fakes in gateway/tests/fake_registry.py and week05-mlflow-registry/tests/conftest.py copy these.
    code = error_code(lambda: client.get_model_version_by_alias(MODEL_NAME, "no-such-alias"))
    check(code == "INVALID_PARAMETER_VALUE", f"missing alias -> {code}")
    code = error_code(lambda: client.get_model_version(MODEL_NAME, "99"))
    check(code == "RESOURCE_DOES_NOT_EXIST", f"missing version -> {code}")


def main() -> None:
    step = sys.argv[1] if len(sys.argv) == 2 else ""
    if step not in ("seed", "verify"):
        sys.exit(__doc__)
    if MODEL_NAME == "gateway-llm":
        sys.exit("refusing to run against the real gateway-llm: set REGISTRY_MODEL=gateway-llm-ci")
    print(f"{step}: {MODEL_NAME} at {TRACKING_URI}")
    client = MlflowClient(TRACKING_URI)
    if step == "seed":
        seed(client)
    else:
        verify(client)


if __name__ == "__main__":
    main()
