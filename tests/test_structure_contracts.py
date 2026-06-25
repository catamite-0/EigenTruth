"""Contract tests for product-architecture skeleton modules."""

import pytest

from eigentruth.adapters import WorldModelPrediction
from eigentruth.calibration import CalibrationArtifact, CalibrationScore, SteeringPolicyConfig
from eigentruth.control import (
    ActionExecutionPolicy,
    ActionExecutionStatus,
    ActionRequest,
    ActionResult,
    ControlAction,
    JsonActionExecutionLedger,
    RiskDecision,
    RiskLevel,
    SQLiteActionExecutionLedger,
)
from eigentruth.registry import RegistryRecord
from eigentruth.verify import Claim, ClaimDependency, VerificationResult, VerificationStatus


def test_calibration_artifact_score_lookup():
    artifact = CalibrationArtifact(
        model_id="tiny-model",
        target_layer=-4,
        scores=(CalibrationScore("maha", threshold=3.0, conformal_alpha=0.1),),
        eigentruth_version="0.1.0",
        steering_policy=SteeringPolicyConfig(mode="disabled"),
    )

    assert artifact.score_names() == ("maha",)
    assert artifact.get_score("maha").threshold == 3.0
    with pytest.raises(KeyError):
        artifact.get_score("missing")


def test_calibration_score_validates_direction_and_alpha():
    with pytest.raises(ValueError):
        CalibrationScore("bad", threshold=1.0, direction="sideways")
    with pytest.raises(ValueError):
        CalibrationScore("bad", threshold=1.0, conformal_alpha=1.5)


def test_risk_decision_validation_and_values():
    decision = RiskDecision(
        action=ControlAction.RETRIEVE,
        risk_level=RiskLevel.MEDIUM,
        confidence=0.7,
        reason="diagnostic threshold exceeded",
    )

    assert decision.action.value == "retrieve"
    assert decision.risk_level.value == "medium"
    with pytest.raises(ValueError):
        RiskDecision(ControlAction.ACCEPT, RiskLevel.LOW, 1.2, "bad")


def test_action_request_json_roundtrip():
    request = ActionRequest(
        action=ControlAction.RETRIEVE,
        reason="unsupported claim",
        payload={"claim_ids": ("c1",)},
        metadata={"policy": "test"},
        request_id="a1",
    )

    payload = request.to_dict()
    loaded = ActionRequest.from_dict(payload)

    assert payload["action"] == "retrieve"
    assert loaded.action is ControlAction.RETRIEVE
    assert loaded.reason == "unsupported claim"
    assert loaded.payload["claim_ids"] == ("c1",)


def test_execute_tool_action_request_roundtrip():
    request = ActionRequest(
        action=ControlAction.EXECUTE_TOOL,
        reason="run verified local tool",
        payload={"tool": "reserve_inventory", "input": {"order_id": "ord_1"}},
        request_id="tool-1",
    )

    payload = request.to_dict()
    loaded = ActionRequest.from_dict(payload)

    assert payload["action"] == "execute_tool"
    assert loaded.action is ControlAction.EXECUTE_TOOL
    assert loaded.payload["tool"] == "reserve_inventory"


def test_action_execution_policy_json_roundtrip():
    policy = ActionExecutionPolicy(
        side_effecting=True,
        require_request_id=True,
        require_idempotency_key=True,
        default_timeout_seconds=2.5,
        max_timeout_seconds=10.0,
        required_metadata_keys=("tenant_id",),
    )

    payload = policy.to_dict()
    loaded = ActionExecutionPolicy.from_dict(payload)

    assert payload["side_effecting"] is True
    assert loaded.side_effecting is True
    assert loaded.require_request_id is True
    assert loaded.require_idempotency_key is True
    assert loaded.default_timeout_seconds == 2.5
    assert loaded.max_timeout_seconds == 10.0
    assert loaded.required_metadata_keys == ("tenant_id",)


def test_action_execution_timeout_values_must_be_positive_finite(tmp_path):
    for value in (float("nan"), float("inf"), True, 0.0):
        with pytest.raises(ValueError, match="positive finite"):
            ActionExecutionPolicy(default_timeout_seconds=value)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="positive finite"):
            SQLiteActionExecutionLedger(tmp_path / "action-ledger.sqlite", timeout_seconds=value)  # type: ignore[arg-type]

    policy = ActionExecutionPolicy(max_timeout_seconds=5.0)
    request = ActionRequest(
        action=ControlAction.EXECUTE_TOOL,
        reason="reserve inventory",
        metadata={"timeout_seconds": "nan"},
    )

    violations = policy.validate_request(request)

    assert violations == ("timeout_seconds must be a positive finite number.",)


def test_action_result_json_roundtrip():
    result = ActionResult(
        action=ControlAction.ABSTAIN,
        status=ActionExecutionStatus.DRY_RUN,
        output={"message": "not enough evidence"},
        metadata={"side_effects": False},
        request_id="r1",
    )

    payload = result.to_dict()
    loaded = ActionResult.from_dict(payload)

    assert payload["status"] == "dry_run"
    assert loaded.action is ControlAction.ABSTAIN
    assert loaded.status is ActionExecutionStatus.DRY_RUN
    assert loaded.output["message"] == "not enough evidence"


def test_json_action_execution_ledger_roundtrip(tmp_path):
    ledger_path = tmp_path / "action-ledger.json"
    ledger = JsonActionExecutionLedger(ledger_path)
    result = ActionResult(
        action=ControlAction.EXECUTE_TOOL,
        status=ActionExecutionStatus.SUCCEEDED,
        output={"reserved": 5},
        metadata={"side_effects": True},
        request_id="reserve-1",
    )
    replacement = ActionResult(
        action=ControlAction.EXECUTE_TOOL,
        status=ActionExecutionStatus.SUCCEEDED,
        output={"reserved": 99},
        metadata={"side_effects": True},
        request_id="reserve-2",
    )

    ledger.record("reserve:1", result)
    ledger.record("reserve:1", replacement)
    loaded = JsonActionExecutionLedger(ledger_path).get("reserve:1")

    assert loaded == result
    assert JsonActionExecutionLedger(ledger_path).get("missing") is None


def test_sqlite_action_execution_ledger_roundtrip(tmp_path):
    ledger_path = tmp_path / "action-ledger.sqlite"
    ledger = SQLiteActionExecutionLedger(ledger_path)
    result = ActionResult(
        action=ControlAction.EXECUTE_TOOL,
        status=ActionExecutionStatus.SUCCEEDED,
        output={"reserved": 5},
        metadata={"side_effects": True},
        request_id="reserve-1",
    )
    replacement = ActionResult(
        action=ControlAction.EXECUTE_TOOL,
        status=ActionExecutionStatus.SUCCEEDED,
        output={"reserved": 99},
        metadata={"side_effects": True},
        request_id="reserve-2",
    )

    ledger.record("reserve:1", result)
    ledger.record("reserve:1", replacement)
    loaded = SQLiteActionExecutionLedger(ledger_path).get("reserve:1")

    assert loaded == result
    assert SQLiteActionExecutionLedger(ledger_path).get("missing") is None
    with pytest.raises(ValueError, match="table_name"):
        SQLiteActionExecutionLedger(ledger_path, table_name="bad-name")
    with pytest.raises(ValueError, match="timeout_seconds"):
        SQLiteActionExecutionLedger(ledger_path, timeout_seconds=None)  # type: ignore[arg-type]


def test_verification_result_and_world_model_prediction_validate_confidence():
    claim = Claim("The trial has 120 participants.", claim_id="c1")
    result = VerificationResult(
        status=VerificationStatus.SUPPORTED,
        confidence=0.9,
        evidence=("trial registry",),
    )
    prediction = WorldModelPrediction(state={"inventory": 10}, confidence=0.8)

    assert claim.claim_id == "c1"
    assert result.status is VerificationStatus.SUPPORTED
    assert prediction.state["inventory"] == 10
    with pytest.raises(ValueError):
        VerificationResult(VerificationStatus.ERROR, confidence=-0.1)
    with pytest.raises(ValueError):
        WorldModelPrediction(state={}, confidence=1.1)


def test_claim_dependency_json_roundtrip_and_validation():
    dependency = ClaimDependency(
        parent_id="c1",
        child_id="c2",
        relation="requires",
        source="unit",
        reason="child claim depends on parent claim",
    )

    payload = dependency.to_dict()
    loaded = ClaimDependency.from_mapping(payload)

    assert payload == {
        "parent_id": "c1",
        "child_id": "c2",
        "relation": "requires",
        "source": "unit",
        "reason": "child claim depends on parent claim",
    }
    assert loaded == dependency
    with pytest.raises(ValueError, match="cannot point to itself"):
        ClaimDependency(parent_id="c1", child_id="c1")


def test_registry_record_key():
    record = RegistryRecord(
        name="truthfulqa-gpt2-l8",
        artifact_type="calibration",
        path="artifacts/calibration.json",
        version="2026-06-16",
    )

    assert record.key() == "calibration:truthfulqa-gpt2-l8:2026-06-16"


def test_artifact_registry_records_score_fusion_artifact(tmp_path):
    from eigentruth.registry import ArtifactRegistry

    registry = ArtifactRegistry(tmp_path / "registry.json").record_score_fusion_artifact(
        name="fusion",
        path=tmp_path / "fusion.json",
        version="0.1",
        metadata={"method": "max_rank"},
    )

    record = registry.get("score_fusion_artifact:fusion:0.1")
    assert record.artifact_type == "score_fusion_artifact"
    assert record.metadata["method"] == "max_rank"
