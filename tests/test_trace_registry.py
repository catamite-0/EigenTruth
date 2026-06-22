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
from eigentruth.registry import ArtifactRegistry, RegistryRecord, build_artifact_manifest, fingerprint_path
from eigentruth.verify import (
    InMemoryVerifier,
    VerificationResult,
    VerificationStatus,
    extract_claims,
    normalize_claim_text,
)


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


def test_artifact_fingerprint_hashes_files_and_directories(tmp_path):
    file_path = tmp_path / "result.json"
    file_path.write_text('{"ok": true}\n', encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text('{"records": 2}\n', encoding="utf-8")
    (cache_dir / "records-00000.pt").write_bytes(b"tensor-bytes")

    file_record = fingerprint_path(file_path, root=tmp_path).to_dict()
    directory_record = fingerprint_path(cache_dir, root=tmp_path).to_dict()
    manifest = build_artifact_manifest(
        {"result": file_path, "cache": cache_dir, "missing": tmp_path / "missing.json"},
        root=tmp_path,
        metadata={"runner": "unit-test"},
    )

    assert file_record["path"] == "result.json"
    assert file_record["kind"] == "file"
    assert file_record["sha256"]
    assert directory_record["path"] == "cache"
    assert directory_record["kind"] == "directory"
    assert directory_record["file_count"] == 2
    assert directory_record["size_bytes"] == len('{"records": 2}\n') + len(b"tensor-bytes")
    assert manifest["metadata"]["runner"] == "unit-test"
    assert manifest["summary"]["artifact_count"] == 3
    assert manifest["summary"]["missing_count"] == 1

    before = directory_record["sha256"]
    (cache_dir / "records-00000.pt").write_bytes(b"changed")
    assert fingerprint_path(cache_dir, root=tmp_path).to_dict()["sha256"] != before


def test_product_trace_action_execution_summary_counts_results():
    trace = ProductTrace(
        action_results=(
            ActionResult(action=ControlAction.RETRIEVE, status=ActionExecutionStatus.SUCCEEDED),
            ActionResult(action=ControlAction.ABSTAIN, status=ActionExecutionStatus.DRY_RUN),
            ActionResult(action=ControlAction.RETRIEVE, status=ActionExecutionStatus.SUCCEEDED),
        )
    )

    summary = trace.action_execution_summary()

    assert summary["total"] == 3
    assert summary["counts_by_status"] == {"succeeded": 2, "dry_run": 1}
    assert summary["counts_by_action"] == {"retrieve": 2, "abstain": 1}
    assert summary["side_effects"] is False


def test_product_trace_verification_route_summary_counts_runtime_routes():
    trace = ProductTrace(
        verification_results=(
            VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=0.9,
                metadata={
                    "selected_route": "structured_qa",
                    "selected_verifier": "QuestionAnswerVerifier",
                    "matched_routes": ("structured_qa", "fallback"),
                    "skipped_routes": (),
                },
            ),
            VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=0.8,
                metadata={
                    "selected_route": "fallback",
                    "selected_verifier": "InMemoryVerifier",
                    "matched_routes": ("structured_qa", "fallback"),
                    "skipped_routes": (
                        {
                            "route": "structured_qa",
                            "status": "insufficient_evidence",
                            "match_reasons": ("context:statement.question",),
                        },
                    ),
                },
            ),
            VerificationResult(status=VerificationStatus.NOT_APPLICABLE, confidence=1.0),
        )
    )

    summary = trace.verification_route_summary()

    assert summary["total"] == 3
    assert summary["routed_total"] == 2
    assert summary["unrouted_total"] == 1
    assert summary["counts_by_status"] == {"supported": 2, "not_applicable": 1}
    assert summary["counts_by_selected_route"] == {"structured_qa": 1, "fallback": 1}
    assert summary["counts_by_matched_route"] == {"structured_qa": 2, "fallback": 2}
    assert summary["counts_by_skipped_route"] == {"structured_qa": 1}
    assert summary["skipped_routes"][0]["match_reasons"] == ("context:statement.question",)
    json.dumps(summary)


def test_artifact_registry_records_trace_report_and_action_result(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ArtifactRegistry.load_json(registry_path)

    registry.record_trace(
        name="req-1",
        path="artifacts/req-1.trace.json",
        version="0.3",
        metadata={"total_actions": 1},
    ).record_report(
        name="tiny-report",
        path="artifacts/report.json",
        version="0.3",
    ).record_action_result(
        name="req-1-actions",
        path="artifacts/req-1.actions.json",
        version="0.3",
    ).save_json()
    loaded = ArtifactRegistry.load_json(registry_path)

    assert loaded.list_records(artifact_type="product_trace")[0].metadata["total_actions"] == 1
    assert loaded.list_records(artifact_type="report")[0].name == "tiny-report"
    assert loaded.list_records(artifact_type="action_result")[0].name == "req-1-actions"
