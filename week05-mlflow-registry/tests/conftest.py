"""Shared test setup for Week 5's release.py: a fake MLflow registry, so tests need no server.

(resolve_champion() lives in the gateway package; its tests and fake are in gateway/tests/.)

Run:  uv run --group dev --group week05 pytest week05-mlflow-registry/tests
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

# The scripts aren't an installed package; make `import release` work from the tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class FakeRegistry:
    """The handful of MlflowClient methods the scripts use, backed by dicts.

    Like an in-memory repository in a Spring test: same method names as the real client, no network.
    """

    def __init__(self, versions: dict[str, dict[str, str]], aliases: dict[str, str] | None = None):
        self.tags = {v: dict(t) for v, t in versions.items()}  # version -> its tags
        self.aliases = dict(aliases or {})  # alias -> version
        self.fail_on: str | None = None  # name of a method that should "crash", to test half-done moves

    def _maybe_fail(self, method: str) -> None:
        if self.fail_on == method:
            raise ConnectionError(f"simulated crash in {method}")

    def get_registered_model(self, name):
        return SimpleNamespace(name=name, aliases=dict(self.aliases))

    def get_model_version(self, name, version):
        if version not in self.tags:
            # The enum constant, not the string "RESOURCE_DOES_NOT_EXIST": MlflowException silently turns an
            # unknown string into INTERNAL_ERROR, and then the fake no longer behaves like the real server.
            raise MlflowException(f"version {version} not found", error_code=RESOURCE_DOES_NOT_EXIST)
        return SimpleNamespace(name=name, version=version, tags=dict(self.tags[version]))

    def search_model_versions(self, filter_string):
        return [SimpleNamespace(version=v, tags=dict(t)) for v, t in self.tags.items()]

    def set_registered_model_alias(self, name, alias, version):
        self._maybe_fail("set_registered_model_alias")
        self.aliases[alias] = version

    def delete_registered_model_alias(self, name, alias):
        self._maybe_fail("delete_registered_model_alias")
        del self.aliases[alias]

    def set_model_version_tag(self, name, version, key, value):
        self.tags[version][key] = value


@pytest.fixture
def registry():
    """v1 and v2 passed their eval, v3 failed. No aliases yet."""
    return FakeRegistry({"1": {"eval": "passed"}, "2": {"eval": "passed"}, "3": {"eval": "failed"}})
