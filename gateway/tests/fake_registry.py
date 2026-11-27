"""A fake MLflow registry: the two MlflowClient calls resolve_champion() makes, backed by dicts.

Like an in-memory repository in a Spring test: same method names as the real client, no network. Its errors
copy what the real MLflow 3.17 client was seen to raise (checked by hand against the Week 5 compose stack, and
in CI by week05-mlflow-registry/scripts/ci_registry_check.py).
"""

from pathlib import Path
from types import SimpleNamespace

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST

# The same text register.py logged for the real v1 and v2.
CARD_V1 = """\
model_id: Qwen/Qwen2.5-0.5B-Instruct
temperature: 0.7
max_tokens: 64
system_prompt: You are a concise assistant. Answer in one short sentence.
"""
CARD_V2 = CARD_V1.replace("0.5B", "1.5B").replace("0.7", "0.3")


class FakeRegistry:
    def __init__(self, cards: dict[str, str], aliases: dict[str, str] | None = None):
        self.cards = dict(cards)  # version -> the text of its serving_config.yaml
        self.aliases = dict(aliases or {})  # alias -> version
        self.down = False  # True: every call fails the way the real client does when the server is unreachable

    def _maybe_down(self) -> None:
        if self.down:
            # Error codes must be the enum constants: MlflowException silently turns a string into INTERNAL_ERROR.
            raise MlflowException("API request to http://localhost:5001 failed: Max retries exceeded", INTERNAL_ERROR)

    def get_model_version_by_alias(self, name, alias):
        self._maybe_down()
        if alias not in self.aliases:
            # Real MLflow 3.17 says INVALID_PARAMETER_VALUE here, not RESOURCE_DOES_NOT_EXIST.
            raise MlflowException(f"Registered model alias {alias} not found.", INVALID_PARAMETER_VALUE)
        version = self.aliases[alias]
        return SimpleNamespace(name=name, version=version, run_id=f"run-{version}")

    def download_artifacts(self, run_id, path, dst_path):
        self._maybe_down()
        version = run_id.removeprefix("run-")
        if version not in self.cards:
            raise MlflowException(f"no artifact {path} in run {run_id}", RESOURCE_DOES_NOT_EXIST)
        target = Path(dst_path) / Path(path).name  # the real client also keeps just the file name
        target.write_text(self.cards[version])
        return str(target)
