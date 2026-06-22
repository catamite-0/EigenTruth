"""Contract tests for product-architecture skeleton modules."""

import pytest

from eigentruth.adapters import WorldModelPrediction
from eigentruth.calibration import CalibrationArtifact, CalibrationScore, SteeringPolicyConfig
from eigentruth.control import (
    ActionExecutionStatus,
    ActionRequest,
    ActionResult,
    ControlAction,
    RiskDecision,
    RiskLevel,
)
from eigentruth.registry import RegistryRecord
from eigentruth.verify import Claim, VerificationResult, VerificationStatus


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


def test_registry_record_key():
    record = RegistryRecord(
        name="truthfulqa-gpt2-l8",
        artifact_type="calibration",
        path="artifacts/calibration.json",
        version="2026-06-16",
    )

    assert record.key() == "calibration:truthfulqa-gpt2-l8:2026-06-16"
