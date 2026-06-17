"""Product trace and artifact registry tests."""

import json

from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control import (
    ActionExecutionStatus,
    ActionRequest,
    ActionResult,
    ControlAction,
    ProductTrace,
    RiskController,
    RiskLevel,
    TraceEvent,
)
from eigentruth.registry import ArtifactRegistry, RegistryRecord
from eigentruth.verify import InMemoryVerifier, VerificationStatus, extract_claims, normalize_claim_text


def test_product_trace_serializes_risk_decision_and_verification_results():
    artifact = CalibrationArtifact(
        model_id="tiny",
        target_layer=-1,
        scores=(CalibrationScore("maha_last", threshold=3.0),),
        eigentruth_version="0.1.0",
    )
    decision = RiskController(artifact).decide({"maha_last": 4.0})
    claims = extract_claims("Paris is the capital of France.")
    verifier = InMemoryVerifier(
        facts={normalize_claim_text("Paris is the capital of France"): VerificationStatus.SUPPORTED},
        evidence={normalize_claim_text("Paris is the capital of France"): ("atlas",)},
    )
    results = verifier.verify_many(claims)

    trace = ProductTrace(
        request_id="req-1",
        diagnostics={"maha_last": 4.0},
        claims=claims,
        verification_results=results,
        risk_decision=decision,
        actions=(
            ActionRequest(
                action=ControlAction.RETRIEVE,
                reason="diagnostic threshold exceeded",
                payload={"claim_ids": ("c1",)},
            ),
        ),
        action_results=(
            ActionResult(
                action=ControlAction.RETRIEVE,
                status=ActionExecutionStatus.DRY_RUN,
                output={"would_execute": "retriever"},
            ),
        ),
        events=(TraceEvent("risk_decision", {"action": decision.action}),),
        metadata={"model_id": "tiny"},
    )
    payload = trace.to_dict()

    assert payload["risk_decision"]["action"] == "retrieve"
    assert payload["risk_decision"]["risk_level"] == RiskLevel.MEDIUM.value
    assert payload["verification_results"][0]["status"] == "supported"
    assert payload["actions"][0]["action"] == "retrieve"
    assert tuple(payload["actions"][0]["payload"]["claim_ids"]) == ("c1",)
    assert payload["action_results"][0]["status"] == "dry_run"
    assert payload["action_results"][0]["output"]["would_execute"] == "retriever"
    json.dumps(payload)


def test_artifact_registry_json_roundtrip(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ArtifactRegistry.load_json(registry_path)
    record = RegistryRecord(
        name="tiny-sweep",
        artifact_type="calibration_report",
        path="artifacts/tiny-sweep.json",
        version="0.2",
        metadata={"best_score": "maha_last"},
    )

    registry.add(record).save_json()
    loaded = ArtifactRegistry.load_json(registry_path)

    assert loaded.get(record.key()) == record
    assert loaded.list_records(artifact_type="calibration_report") == (record,)
    assert loaded.to_dict()["schema_version"] == 1
