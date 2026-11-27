"""The promote/rollback rules, one test per rule."""

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from release import ReleaseError, deploy, promote, rollback


def test_first_promotion_sets_champion_and_no_fallback(registry):
    promote(registry, "1", by="davis")
    assert registry.aliases == {"champion": "1"}
    assert registry.tags["1"]["promoted_by"] == "davis"
    assert "promoted_at" in registry.tags["1"]


def test_promotion_keeps_old_champion_as_fallback(registry):
    promote(registry, "1", by="davis")
    promote(registry, "2", by="davis")
    assert registry.aliases == {"champion": "2", "fallback": "1"}


def test_failed_eval_is_refused_and_nothing_moves(registry):
    promote(registry, "1", by="davis")
    with pytest.raises(ReleaseError, match="eval=failed"):
        promote(registry, "3", by="davis")
    assert registry.aliases == {"champion": "1"}


def test_untagged_version_is_refused(registry):
    registry.tags["4"] = {}  # registered by hand, never scored
    with pytest.raises(ReleaseError, match=r"eval=\(none\)"):
        promote(registry, "4", by="davis")


def test_unknown_version_is_refused(registry):
    with pytest.raises(ReleaseError, match="no v99"):
        promote(registry, "99", by="davis")


def test_server_errors_are_not_disguised_as_refusals(registry):
    def broken(name, version):
        raise MlflowException("internal error", error_code=INTERNAL_ERROR)

    registry.get_model_version = broken
    with pytest.raises(MlflowException):
        promote(registry, "1", by="davis")


def test_promoting_the_current_champion_changes_nothing(registry):
    promote(registry, "1", by="davis")
    promote(registry, "2", by="davis")
    assert "already champion" in promote(registry, "2", by="davis")
    assert registry.aliases == {"champion": "2", "fallback": "1"}  # fallback NOT overwritten with v2


def test_rollback_restores_fallback_and_clears_it(registry):
    promote(registry, "1", by="davis")
    promote(registry, "2", by="davis")
    rollback(registry, reason="answers got worse")
    assert registry.aliases == {"champion": "1"}
    assert registry.tags["2"]["rollback_reason"] == "answers got worse"


def test_second_rollback_is_refused(registry):
    # Option A: after a rollback there's no fallback, so the suspect version can't come back by accident.
    promote(registry, "1", by="davis")
    promote(registry, "2", by="davis")
    rollback(registry)
    with pytest.raises(ReleaseError, match="nothing safe"):
        rollback(registry)
    assert registry.aliases == {"champion": "1"}


@pytest.mark.parametrize("crash_in", ["set_registered_model_alias", "delete_registered_model_alias"])
def test_rollback_interrupted_halfway_never_leaves_the_bad_version_live(registry, crash_in):
    promote(registry, "1", by="davis")
    promote(registry, "2", by="davis")
    registry.fail_on = crash_in
    with pytest.raises(ConnectionError):
        rollback(registry)
    # Whatever step failed, champion is never "neither" and never stuck on a missing alias.
    assert registry.aliases["champion"] in {"1", "2"}
    if crash_in == "delete_registered_model_alias":
        assert registry.aliases == {"champion": "1", "fallback": "1"}  # the good version is already live


# --- Week 6: deploy to a Blue/Green side ----------------------------------------------------------------


def test_deploy_points_the_side_at_the_version_and_leaves_champion_alone(registry):
    promote(registry, "1", by="davis")
    deploy(registry, "2", "green", by="davis")
    assert registry.aliases == {"champion": "1", "green": "2"}
    assert registry.tags["2"]["deployed_by"] == "davis" and "deployed_to_green_at" in registry.tags["2"]


def test_deploy_has_the_same_gate_as_promote(registry):
    with pytest.raises(ReleaseError, match="eval=failed"):
        deploy(registry, "3", "green", by="davis")
    assert registry.aliases == {}


def test_deploy_to_an_unknown_side_is_refused(registry):
    with pytest.raises(ReleaseError, match="purple"):
        deploy(registry, "2", "purple", by="davis")


def test_deploying_the_same_version_again_changes_nothing(registry):
    deploy(registry, "2", "green", by="davis")
    assert "already on green" in deploy(registry, "2", "green", by="davis")
