"""Product trace and artifact registry tests."""

import json
import math
import os
from pathlib import Path

import pytest

import eigentruth.control.trace as trace_module
import eigentruth.registry.provenance as registry_provenance
from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control import (
    ActionExecutionStatus,
    ActionRequest,
    ActionResult,
    ControlAction,
    FeedbackOutcome,
    FinalAnswer,
    FinalAnswerStatus,
    ProductFeedbackRecord,
    ProductFeedbackStore,
    ProductPromotionContract,
    ProductRuntimeBudgetPolicy,
    ProductTrace,
    RiskController,
    RiskDecision,
    RiskLevel,
    RuntimePhaseTiming,
    RuntimeTrace,
    TraceEvent,
    evaluate_product_runtime_budget,
    first_existing_product_promotion_contract_path,
    load_product_promotion_contract,
    load_product_runtime_evidence_bundle,
    product_promotion_contract_metadata,
    product_runtime_budget_policy_from_release_candidate,
    product_runtime_metrics,
    product_trace_fingerprint,
    write_feedback_jsonl,
)
from eigentruth.registry import (
    ArtifactRegistry,
    ArtifactVerificationContext,
    RegistryRecord,
    build_artifact_manifest,
    fingerprint_path,
    load_and_verify_artifact_manifest,
    load_fingerprint_cache,
    load_json_cache,
    save_fingerprint_cache,
    save_json_cache,
)
from eigentruth.verify import (
    ClaimVerificationPlanner,
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
        final_answer=FinalAnswer(
            status=FinalAnswerStatus.NEEDS_RETRIEVAL,
            text="I need more evidence before answering reliably.",
            answerable=False,
            action=ControlAction.RETRIEVE,
            risk_level=RiskLevel.MEDIUM,
            confidence=decision.confidence,
            reason=decision.reason,
            claim_summary={"total_claims": 1, "status_counts": {"supported": 1}},
            evidence=({"claim_id": "c1", "text": "atlas"},),
            followup={"requires_followup": True},
        ),
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
    assert payload["final_answer"]["status"] == "needs_retrieval"
    assert payload["final_answer"]["answerable"] is False
    assert trace.final_answer_summary()["status"] == "needs_retrieval"
    assert trace.final_answer_summary()["evidence_count"] == 1
    json.dumps(payload)


def test_product_trace_serializes_claim_verification_plan_and_bounded_summary():
    claims = extract_claims("As of 2026, AlphaCorp has 10 offices. 2 plus 2 is 5.")
    plan = ClaimVerificationPlanner().plan(claims)
    trace = ProductTrace(
        request_id="req-plan",
        claims=claims,
        verification_plan=plan,
        metadata={"large_unselected_metadata": tuple(range(100))},
    )

    payload = trace.to_dict()
    bounded = trace.to_bounded_dict(max_nested_items=2)

    assert payload["verification_plan"]["run_verifier"] is True
    assert payload["verification_plan"]["verification_scope"] == "all"
    assert payload["verification_plan"]["route_hints"][0]["routes"] == (
        "retrieval",
        "triple_evidence",
        "groundedness",
    )
    assert payload["verification_plan"]["calculation_checks"][0]["expression"] == "2 + 2"
    assert payload["verification_plan"]["cost_estimate"]["estimated_cost_units"] == pytest.approx(4.95)
    assert bounded["summaries"]["verification_plan"]["available"] is True
    assert bounded["summaries"]["verification_plan"]["claim_count"] == 2
    assert bounded["summaries"]["verification_plan"]["route_counts"]["retrieval"] == 2
    assert bounded["summaries"]["verification_plan"]["route_counts"]["triple_evidence"] == 2
    assert bounded["summaries"]["verification_plan"]["tool_payload_counts"]["calculation_checks"] == 1
    assert bounded["summaries"]["verification_plan"]["cost_estimate"]["claim_count"] == 2
    assert bounded["summaries"]["verification_plan"]["cost_estimate"]["_truncated"] is True
    assert "verification_plan" not in bounded

    metrics = product_runtime_metrics(trace)
    bounded_metrics = product_runtime_metrics(bounded)
    assert metrics["verification_plan_available"] is True
    assert metrics["verification_plan_source"] == "full_trace"
    assert metrics["verification_plan_claim_count"] == 2.0
    assert metrics["verification_plan_route_hint_count"] == 2.0
    assert metrics["verification_plan_route_counts"]["retrieval"] == 2
    assert metrics["verification_plan_route_counts"]["triple_evidence"] == 2
    assert metrics["verification_plan_calculation_check_count"] == 1.0
    assert bounded_metrics["verification_plan_available"] is True
    assert bounded_metrics["verification_plan_source"] == "bounded_summary"
    assert bounded_metrics["verification_plan_claim_count"] == 2.0
    assert bounded_metrics["verification_plan_route_hint_count"] is None
    assert bounded_metrics["verification_plan_route_counts"]["retrieval"] == 2
    assert bounded_metrics["verification_plan_route_counts"]["triple_evidence"] == 2
    json.dumps(payload)
    json.dumps(bounded)


def test_product_trace_feedback_and_registry_normalize_strict_json_values(tmp_path):
    trace = ProductTrace(
        request_id="req-json",
        diagnostics={"bad": math.inf, "path": tmp_path / "diag.json", "tags": {"b", "a"}},
        risk_decision=RiskDecision(
            action=ControlAction.CLARIFY,
            risk_level=RiskLevel.UNKNOWN,
            confidence=1.0,
            reason="invalid input",
            diagnostics={"raw": math.nan, "blob": b"abc"},
        ),
        metadata={"path": tmp_path / "trace.json", "items": {"z", "a"}},
    )
    payload = trace.to_dict()

    json.dumps(payload, allow_nan=False)
    assert payload["diagnostics"]["bad"] == "inf"
    assert payload["diagnostics"]["path"] == str(tmp_path / "diag.json")
    assert tuple(payload["diagnostics"]["tags"]) == ("a", "b")
    assert payload["risk_decision"]["diagnostics"]["raw"] == "nan"
    assert payload["risk_decision"]["diagnostics"]["blob"]["encoding"] == "base64"

    fingerprint = product_trace_fingerprint(trace)
    assert fingerprint == product_trace_fingerprint(trace)

    feedback_path = tmp_path / "feedback.jsonl"
    write_feedback_jsonl(
        feedback_path,
        (ProductFeedbackRecord(request_id="req-json", outcome=FeedbackOutcome.UNKNOWN, metadata={"raw": math.inf}),),
    )
    assert json.loads(feedback_path.read_text(encoding="utf-8"))["metadata"]["raw"] == "inf"

    registry_path = tmp_path / "registry.json"
    ArtifactRegistry(
        registry_path,
        records=(
            RegistryRecord(
                name="trace",
                artifact_type="trace",
                path=str(tmp_path / "trace.json"),
                version="1",
                metadata={"path": tmp_path / "trace.json", "raw": math.nan},
            ),
        ),
    ).save_json()
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry_payload["records"][0]["metadata"]["raw"] == "nan"


def test_product_feedback_record_jsonl_roundtrip_and_trace_fingerprint(tmp_path):
    trace = ProductTrace(
        request_id="req-feedback",
        diagnostics={"maha_last": 0.1},
        risk_decision={
            "action": "accept",
            "risk_level": "low",
            "confidence": 0.9,
            "reason": "low risk",
        },
    )
    fingerprint = product_trace_fingerprint(trace)
    record = ProductFeedbackRecord(
        request_id="req-feedback",
        trace_fingerprint=fingerprint,
        claim_id="claim-1",
        outcome=FeedbackOutcome.INCORRECT,
        feedback_source="human_review",
        corrected_text="Corrected answer.",
        evidence_refs=("doc:1",),
        metadata={"reviewer": "unit"},
        created_at="2026-06-24T00:00:00+00:00",
    )
    path = tmp_path / "feedback.jsonl"

    write_feedback_jsonl(path, (record,))
    store = ProductFeedbackStore(path)
    store.append({
        "request_id": "req-2",
        "outcome": "correct",
        "feedback_source": "automated_eval",
    })
    loaded = store.read_all()

    assert len(loaded) == 2
    assert loaded[0].to_dict()["trace_fingerprint"] == fingerprint
    assert loaded[0].outcome is FeedbackOutcome.INCORRECT
    assert loaded[0].evidence_refs == ("doc:1",)
    assert loaded[1].outcome is FeedbackOutcome.CORRECT
    assert product_trace_fingerprint(trace.to_dict()) == fingerprint

    with pytest.raises(ValueError, match="outcome"):
        ProductFeedbackRecord(request_id="req", outcome="maybe")


def test_product_trace_bounded_payload_summarizes_large_fields():
    trace = ProductTrace(
        request_id="req-bounded",
        diagnostics={f"score_{index}": float(index) for index in range(5)},
        claims=tuple(
            {
                "claim_id": f"c{index}",
                "text": f"Claim {index}",
                "metadata": {"feature": index},
            }
            for index in range(4)
        ),
        verification_results=tuple(
            {
                "status": "supported",
                "confidence": 0.9,
                "evidence": tuple(f"evidence-{index}-{item}" for item in range(5)),
                "explanation": "x" * 80,
                "metadata": {
                    "selected_route": "structured_qa",
                    "retrieval_hits": tuple({"doc": item} for item in range(5)),
                    "total_duration_seconds": 0.01,
                },
            }
            for index in range(4)
        ),
        actions=tuple(
            ActionRequest(
                action=ControlAction.RETRIEVE,
                reason="unsupported",
                payload={"claim_ids": (f"c{index}",), "extra": tuple(range(10))},
            )
            for index in range(3)
        ),
        action_results=tuple(
            ActionResult(
                action=ControlAction.RETRIEVE,
                status=ActionExecutionStatus.SUCCEEDED,
                output={"hits": tuple({"id": item, "text": "y" * 80} for item in range(6))},
                metadata={"side_effects": False},
            )
            for _ in range(3)
        ),
        events=tuple(
            TraceEvent("event", {"items": tuple(range(10))})
            for _ in range(3)
        ),
        metadata={
            "artifact_source": "artifact.json",
            "promotion_contract_source": "contract.json",
            "promotion_contract_selfcheck_signal_fusion_workflow": {
                "report_path": "selfcheck.json"
            },
            "promotion_contract_world_model_signal_workflow": {
                "release_gate_status": "promote"
            },
            "runtime_budget": {"passed": True},
            "large_unselected_metadata": tuple(range(100)),
        },
        runtime_trace=RuntimeTrace(
            phases=(RuntimePhaseTiming("phase", 0.01),),
        ),
        final_answer=FinalAnswer(
            status=FinalAnswerStatus.ANSWERED,
            text="Final answer " + ("z" * 80),
            answerable=True,
            action=ControlAction.ACCEPT,
            risk_level=RiskLevel.LOW,
            confidence=0.97,
            reason="accepted",
            claim_summary={"total_claims": 4, "status_counts": {"supported": 4}},
            evidence=tuple({"claim_id": f"c{index}", "text": "evidence " + "w" * 80} for index in range(4)),
            followup={"requires_followup": False},
        ),
    )

    payload = trace.to_bounded_dict(
        max_diagnostics=2,
        max_claims=1,
        max_verification_results=2,
        max_actions=1,
        max_action_results=1,
        max_events=1,
        max_nested_items=2,
        max_string_length=40,
    )

    assert payload["trace_format"] == "bounded_product_trace"
    assert payload["request_id"] == "req-bounded"
    assert len(payload["diagnostics"]) == 2
    assert payload["truncation"]["diagnostics"] == {"total": 5, "included": 2, "omitted": 3}
    assert payload["truncation"]["claims"]["omitted"] == 3
    assert payload["truncation"]["verification_results"]["omitted"] == 2
    assert payload["truncation"]["actions"]["omitted"] == 2
    assert payload["truncation"]["action_results"]["omitted"] == 2
    assert payload["truncation"]["events"]["omitted"] == 2
    assert payload["runtime_trace"] is None
    assert payload["summaries"]["runtime"]["measured_phases"] == 1
    assert payload["summaries"]["action_execution"]["total"] == 3
    assert payload["summaries"]["final_answer"]["status"] == "answered"
    assert payload["summaries"]["final_answer"]["answerable"] is True
    assert payload["summaries"]["final_answer"]["evidence_count"] == 4
    assert payload["final_answer"]["text"].endswith("chars]")
    assert len(payload["final_answer"]["evidence"]) == 2
    assert payload["verification_results"][0]["evidence_count"] == 5
    assert len(payload["verification_results"][0]["evidence"]) == 2
    assert len(payload["verification_results"][0]["explanation"]) <= 40
    assert payload["action_results"][0]["output_summary"]["key_count"] == 1
    assert "large_unselected_metadata" not in payload["metadata"]
    assert payload["metadata"]["artifact_source"] == "artifact.json"
    assert payload["metadata"]["promotion_contract_source"] == "contract.json"
    assert payload["metadata"]["promotion_contract_selfcheck_signal_fusion_workflow"] == {
        "report_path": "selfcheck.json"
    }
    assert payload["metadata"]["promotion_contract_world_model_signal_workflow"] == {
        "release_gate_status": "promote"
    }
    metrics = product_runtime_metrics(payload)
    assert metrics["final_answer_available"] is True
    assert metrics["final_answer_source"] == "bounded_summary"
    assert metrics["final_answer_status"] == "answered"
    assert metrics["final_answer_answerable"] is True
    assert metrics["final_answer_evidence_count"] == 4.0
    json.dumps(payload)


def test_product_trace_bounded_payload_reuses_prepared_trace_payload(monkeypatch):
    original_verification = trace_module._verification_result_to_dict
    original_action_result = trace_module._action_result_to_dict
    original_event = trace_module._event_to_dict
    calls = {"verification": 0, "action_result": 0, "event": 0}

    def counted_verification(result):
        calls["verification"] += 1
        return original_verification(result)

    def counted_action_result(result):
        calls["action_result"] += 1
        return original_action_result(result)

    def counted_event(event):
        calls["event"] += 1
        return original_event(event)

    monkeypatch.setattr(trace_module, "_verification_result_to_dict", counted_verification)
    monkeypatch.setattr(trace_module, "_action_result_to_dict", counted_action_result)
    monkeypatch.setattr(trace_module, "_event_to_dict", counted_event)
    trace = ProductTrace(
        verification_results=tuple(
            VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=0.9,
                metadata={
                    "selected_route": "structured_qa",
                    "matched_routes": ("structured_qa",),
                    "total_duration_seconds": 0.01,
                },
            )
            for _ in range(5)
        ),
        action_results=tuple(
            ActionResult(action=ControlAction.RETRIEVE, status=ActionExecutionStatus.SUCCEEDED)
            for _ in range(4)
        ),
        events=(
            TraceEvent("verification_stage_decision", {"run_verifier": True}),
            TraceEvent("initial_verification", {"n_claims": 5, "verification_result_count": 5}),
        ),
    )

    payload = trace.to_bounded_dict()

    assert payload["summaries"]["verification_route"]["total"] == 5
    assert payload["summaries"]["verification_route_cost"]["total"] == 5
    assert payload["summaries"]["action_execution"]["total"] == 4
    assert calls == {"verification": 5, "action_result": 4, "event": 2}


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


def test_directory_fingerprint_reuses_single_directory_scan(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    nested_dir = cache_dir / "nested"
    nested_dir.mkdir(parents=True)
    (cache_dir / "manifest.json").write_text('{"records": 2}\n', encoding="utf-8")
    (nested_dir / "records-00000.pt").write_bytes(b"tensor-bytes")
    original_rglob = Path.rglob
    scan_count = 0

    def counted_rglob(self, pattern):
        nonlocal scan_count
        if self == cache_dir:
            scan_count += 1
        return original_rglob(self, pattern)

    monkeypatch.setattr(Path, "rglob", counted_rglob)

    fingerprint = fingerprint_path(cache_dir, root=tmp_path).to_dict()

    assert fingerprint["kind"] == "directory"
    assert fingerprint["file_count"] == 2
    assert scan_count == 1


def test_artifact_manifest_verification_detects_drift_and_nested_drift(tmp_path):
    data_path = tmp_path / "result.json"
    data_path.write_text('{"score": 1}\n', encoding="utf-8")
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(build_artifact_manifest({"result": data_path}, root=tmp_path)),
        encoding="utf-8",
    )

    clean = load_and_verify_artifact_manifest(manifest_path)
    assert clean.passed is True
    assert clean.checked == 1

    data_path.write_text('{"score": 200}\n', encoding="utf-8")
    drifted = load_and_verify_artifact_manifest(manifest_path)
    assert drifted.passed is False
    assert drifted.failures[0].name == "result"
    assert {failure.field for failure in drifted.failures} == {"sha256", "size_bytes"}

    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child_data = child_dir / "result.json"
    child_data.write_text('{"score": 1}\n', encoding="utf-8")
    child_manifest_path = child_dir / "artifact-manifest.json"
    child_manifest_path.write_text(
        json.dumps(build_artifact_manifest({"result": child_data}, root=child_dir)),
        encoding="utf-8",
    )
    root_manifest_path = tmp_path / "root-manifest.json"
    root_manifest_path.write_text(
        json.dumps(build_artifact_manifest({"child_manifest": child_manifest_path}, root=tmp_path)),
        encoding="utf-8",
    )
    child_data.write_text('{"score": 3}\n', encoding="utf-8")

    assert load_and_verify_artifact_manifest(root_manifest_path).passed is True
    recursive = load_and_verify_artifact_manifest(root_manifest_path, recursive=True)
    assert recursive.passed is False
    assert recursive.nested[0].failures[0].name == "result"


def test_artifact_manifest_verification_rejects_schema_and_summary_drift(tmp_path):
    data_path = tmp_path / "result.json"
    data_path.write_text('{"score": 1}\n', encoding="utf-8")
    manifest = build_artifact_manifest({"result": data_path}, root=tmp_path)
    manifest_path = tmp_path / "artifact-manifest.json"

    tampered = dict(manifest)
    tampered["schema_version"] = 999
    tampered["digest_algorithm"] = "md5"
    tampered["summary"] = dict(manifest["summary"])
    tampered["summary"]["artifact_count"] = 100
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")

    verification = load_and_verify_artifact_manifest(manifest_path)

    assert verification.passed is False
    assert {(failure.name, failure.field) for failure in verification.failures} == {
        ("manifest", "schema_version"),
        ("manifest", "digest_algorithm"),
        ("manifest", "summary.artifact_count"),
    }
    assert verification.checked == 1


def test_artifact_manifest_verification_resolves_sibling_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    shared_dir = tmp_path / "shared"
    run_dir.mkdir()
    shared_dir.mkdir()
    shared_report = shared_dir / "inside-sampling-profile-comparison.json"
    shared_report.write_text('{"status": "promote"}\n', encoding="utf-8")

    manifest = build_artifact_manifest({"inside_sampling": shared_report}, root=run_dir)
    manifest_path = run_dir / "artifact-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert manifest["artifacts"]["inside_sampling"]["path"] == "../shared/inside-sampling-profile-comparison.json"
    assert load_and_verify_artifact_manifest(manifest_path).passed is True


def test_artifact_manifest_verification_reuses_run_local_fingerprint_cache(tmp_path, monkeypatch):
    data_path = tmp_path / "shared-result.json"
    data_path.write_text('{"score": 1}\n', encoding="utf-8")
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child_manifest_path = child_dir / "artifact-manifest.json"
    child_manifest_path.write_text(
        json.dumps(build_artifact_manifest({"shared_result": data_path}, root=child_dir)),
        encoding="utf-8",
    )
    root_manifest_path = tmp_path / "artifact-manifest.json"
    root_manifest_path.write_text(
        json.dumps(build_artifact_manifest({
            "child_manifest": child_manifest_path,
            "direct_result": data_path,
        }, root=tmp_path)),
        encoding="utf-8",
    )
    original_sha256_file = registry_provenance._sha256_file
    calls_by_path: dict[str, int] = {}

    def counted_sha256_file(path):
        key = str(path.resolve())
        calls_by_path[key] = calls_by_path.get(key, 0) + 1
        return original_sha256_file(path)

    monkeypatch.setattr(registry_provenance, "_sha256_file", counted_sha256_file)

    verification = load_and_verify_artifact_manifest(root_manifest_path, recursive=True)

    assert verification.passed is True
    assert calls_by_path[str(data_path.resolve())] == 1
    assert calls_by_path[str(child_manifest_path.resolve())] == 1


def test_explicit_fingerprint_cache_invalidates_changed_file(tmp_path, monkeypatch):
    data_path = tmp_path / "result.json"
    data_path.write_text('{"score": 1}\n', encoding="utf-8")
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(build_artifact_manifest({"result": data_path}, root=tmp_path)),
        encoding="utf-8",
    )
    original_sha256_file = registry_provenance._sha256_file
    call_count = 0

    def counted_sha256_file(path):
        nonlocal call_count
        call_count += 1
        return original_sha256_file(path)

    monkeypatch.setattr(registry_provenance, "_sha256_file", counted_sha256_file)
    fingerprint_cache = {}

    assert load_and_verify_artifact_manifest(
        manifest_path,
        fingerprint_cache=fingerprint_cache,
    ).passed is True
    data_path.write_text('{"score": 2}\n', encoding="utf-8")
    drifted = load_and_verify_artifact_manifest(
        manifest_path,
        fingerprint_cache=fingerprint_cache,
    )

    assert drifted.passed is False
    assert {failure.field for failure in drifted.failures} >= {"sha256"}
    assert call_count == 2


def test_persisted_fingerprint_cache_reuses_unchanged_file(tmp_path, monkeypatch):
    data_path = tmp_path / "result.json"
    cache_path = tmp_path / "fingerprints.json"
    data_path.write_text('{"score": 1}\n', encoding="utf-8")
    fingerprint_cache = {}
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(build_artifact_manifest(
            {"result": data_path},
            root=tmp_path,
            fingerprint_cache=fingerprint_cache,
        )),
        encoding="utf-8",
    )
    save_fingerprint_cache(cache_path, fingerprint_cache)
    loaded_cache = load_fingerprint_cache(cache_path)
    original_sha256_file = registry_provenance._sha256_file
    call_count = 0

    def counted_sha256_file(path):
        nonlocal call_count
        call_count += 1
        return original_sha256_file(path)

    monkeypatch.setattr(registry_provenance, "_sha256_file", counted_sha256_file)

    verification = load_and_verify_artifact_manifest(
        manifest_path,
        fingerprint_cache=loaded_cache,
    )

    assert verification.passed is True
    assert call_count == 0
    assert load_fingerprint_cache(tmp_path / "missing-cache.json") == {}


def test_persisted_json_cache_reuses_unchanged_object(tmp_path, monkeypatch):
    data_path = tmp_path / "payload.json"
    cache_path = tmp_path / "json-cache.json"
    data_path.write_text('{"score": 1}\n', encoding="utf-8")
    context = ArtifactVerificationContext()

    payload, error = context.load_json_object(data_path)
    save_json_cache(cache_path, context.json_cache or {})
    loaded_cache = load_json_cache(cache_path)
    warm_context = ArtifactVerificationContext(json_cache=loaded_cache)
    original_read_text = Path.read_text

    def blocked_read_text(path, *args, **kwargs):
        if path == data_path:
            raise AssertionError("warm JSON cache should avoid reading the unchanged artifact")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", blocked_read_text)
    warm_payload, warm_error = warm_context.load_json_object(data_path)

    assert payload == {"score": 1}
    assert error is None
    assert warm_payload == {"score": 1}
    assert warm_error is None
    assert warm_context.json_cache_summary() == {
        "requests": 1,
        "hits": 1,
        "misses": 0,
        "errors": 0,
        "entries": 1,
        "hit_rate": 1.0,
    }
    assert load_json_cache(tmp_path / "missing-json-cache.json") == {}


def test_save_json_cache_prunes_stale_same_path_signatures(tmp_path):
    cache_path = tmp_path / "json-cache.json"
    data_path = tmp_path / "payload.json"
    old_key = f"{data_path}:16:1:1:100:old"
    latest_key = f"{data_path}:16:2:2:100:latest"
    unrelated_key = f"{tmp_path / 'other.json'}:16:1:1:200:other"

    save_json_cache(
        cache_path,
        {
            old_key: {"payload": {"score": 1}, "error": None},
            unrelated_key: {"payload": {"score": 3}, "error": None},
            latest_key: {"payload": {"score": 2}, "error": None},
        },
    )
    payload = json.loads(cache_path.read_text(encoding="utf-8"))

    assert old_key not in payload
    assert payload[latest_key]["payload"] == {"score": 2}
    assert payload[unrelated_key]["payload"] == {"score": 3}
    assert len(payload) == 2


def test_json_cache_returns_isolated_nested_payload_copies(tmp_path):
    data_path = tmp_path / "nested.json"
    data_path.write_text(
        json.dumps({"nested": {"items": [1], "value": {"score": 2}}}) + "\n",
        encoding="utf-8",
    )
    context = ArtifactVerificationContext()

    first_payload, first_error = context.load_json_object(data_path)
    first_payload["nested"]["items"].append(99)
    first_payload["nested"]["value"]["score"] = 7
    second_payload, second_error = context.load_json_object(data_path)
    assert second_payload == {"nested": {"items": [1], "value": {"score": 2}}}
    second_payload["nested"]["items"].append(42)
    third_payload, third_error = context.load_json_object(data_path)

    assert first_error is None
    assert second_error is None
    assert third_error is None
    assert third_payload == {"nested": {"items": [1], "value": {"score": 2}}}


def test_artifact_verification_context_caches_manifest_json_and_fingerprints(tmp_path):
    data_path = tmp_path / "result.json"
    manifest_path = tmp_path / "artifact-manifest.json"
    data_path.write_text('{"score": 1}\n', encoding="utf-8")
    context = ArtifactVerificationContext()
    manifest_path.write_text(
        json.dumps(context.build_artifact_manifest({"result": data_path}, root=tmp_path)),
        encoding="utf-8",
    )

    first = context.load_and_verify_artifact_manifest(manifest_path)
    second = context.load_and_verify_artifact_manifest(manifest_path)

    assert first.passed is True
    assert second.passed is True
    assert context.json_cache_summary() == {
        "requests": 2,
        "hits": 1,
        "misses": 1,
        "errors": 0,
        "entries": 1,
        "hit_rate": 0.5,
    }
    fingerprint_summary = context.cache_summary()["artifact_fingerprint_cache"]
    assert fingerprint_summary["requests"] == 3
    assert fingerprint_summary["hits"] == 2
    assert fingerprint_summary["misses"] == 1
    assert fingerprint_summary["entries"] >= 1
    assert fingerprint_summary["hit_rate"] == 2 / 3

    data_path.write_text('{"score": 200}\n', encoding="utf-8")
    drifted = context.load_and_verify_artifact_manifest(manifest_path)

    assert drifted.passed is False
    assert {failure.field for failure in drifted.failures} >= {"sha256"}
    assert context.json_cache_summary()["hits"] == 2
    json.dumps(context.cache_summary())


def test_artifact_json_cache_invalidates_same_size_same_mtime_content_change(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text('{"value":"aaaa"}\n', encoding="utf-8")
    stat = path.stat()
    context = ArtifactVerificationContext()

    first, first_error = context.load_json_object(path)
    path.write_text('{"value":"bbbb"}\n', encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second, second_error = context.load_json_object(path)

    assert first_error is None
    assert second_error is None
    assert first == {"value": "aaaa"}
    assert second == {"value": "bbbb"}
    assert context.json_cache_summary()["hits"] == 0
    assert context.json_cache_summary()["misses"] == 2
    assert context.json_cache_summary()["entries"] == 2


def test_artifact_manifest_parallel_fingerprinting_matches_serial_and_reuses_cache(tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    first_path.write_text('{"score": 1}\n', encoding="utf-8")
    second_path.write_text('{"score": 2}\n', encoding="utf-8")
    (cache_dir / "records.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    artifacts = {
        "first": first_path,
        "second": second_path,
        "cache": cache_dir,
        "missing": tmp_path / "missing.json",
    }

    serial = build_artifact_manifest(artifacts, root=tmp_path)
    parallel = build_artifact_manifest(artifacts, root=tmp_path, max_workers=3)

    assert parallel == serial

    context = ArtifactVerificationContext()
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(context.build_artifact_manifest(artifacts, root=tmp_path, max_workers=3)),
        encoding="utf-8",
    )

    first = context.load_and_verify_artifact_manifest(manifest_path, max_workers=3)
    second = context.load_and_verify_artifact_manifest(manifest_path, max_workers=3)

    assert first.passed is True
    assert second.passed is True
    fingerprint_summary = context.cache_summary()["artifact_fingerprint_cache"]
    assert fingerprint_summary["requests"] == 12
    assert fingerprint_summary["hits"] == 8
    assert fingerprint_summary["misses"] == 4
    assert fingerprint_summary["entries"] == 4

    with pytest.raises(ValueError, match="max_workers"):
        build_artifact_manifest(artifacts, root=tmp_path, max_workers=True)  # type: ignore[arg-type]


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


def test_product_trace_runtime_summary_counts_phase_timings():
    trace = ProductTrace(
        runtime_trace=RuntimeTrace(
            total_seconds=0.40,
            phases=(
                RuntimePhaseTiming("diagnostic_risk_decision", 0.05),
                RuntimePhaseTiming("initial_verification", 0.20, metadata={"n_claims": 2}),
                RuntimePhaseTiming("initial_verification", 0.10, metadata={"n_claims": 1}),
            ),
        )
    )

    payload = trace.to_dict()
    summary = trace.runtime_summary()

    assert payload["runtime_trace"]["summary"]["measured_phases"] == 3
    assert summary["total_seconds"] == 0.40
    assert summary["phase_counts"] == {
        "diagnostic_risk_decision": 1,
        "initial_verification": 2,
    }
    assert round(summary["phase_seconds"]["initial_verification"], 6) == 0.30
    assert summary["phase_stats"]["initial_verification"]["count"] == 2
    assert round(summary["phase_stats"]["initial_verification"]["mean_seconds"], 6) == 0.15
    assert round(summary["phase_p95_seconds"]["initial_verification"], 6) == 0.195
    assert round(summary["phase_p99_seconds"]["initial_verification"], 6) == 0.199
    assert summary["slowest_phase"] == {"name": "initial_verification", "seconds": 0.20}
    json.dumps(payload)


def test_product_trace_cache_summary_aggregates_named_cache_stats():
    trace = ProductTrace(
        metadata={
            "cache": {
                "verifier": {"size": 2, "hits": 3, "misses": 1},
                "retriever": {"size": 1, "hits": 1, "misses": 3},
            },
        },
    )

    summary = trace.cache_summary()

    assert summary["total_caches"] == 2
    assert summary["aggregate"]["size"] == 3
    assert summary["aggregate"]["hits"] == 4
    assert summary["aggregate"]["misses"] == 4
    assert summary["aggregate"]["requests"] == 8
    assert summary["aggregate"]["hit_rate"] == 0.5
    assert summary["caches"]["verifier"]["hit_rate"] == 0.75
    json.dumps(summary)


def test_product_trace_verification_stage_summary_counts_saved_claims():
    claims = extract_claims("Paris is the capital of France. Lyon is in France.")
    trace = ProductTrace(
        claims=claims,
        verification_results=(),
        events=(
            TraceEvent(
                "verification_stage_decision",
                {
                    "run_verifier": False,
                    "reason": "diagnostics and claim metadata did not require verification",
                },
            ),
            TraceEvent(
                "initial_verification",
                {"n_claims": len(claims), "skipped": True, "results": ()},
            ),
        ),
        metadata={"staged_verification_enabled": True},
    )

    summary = trace.verification_stage_summary()
    metrics = product_runtime_metrics(trace)
    report = evaluate_product_runtime_budget(
        trace,
        ProductRuntimeBudgetPolicy(
            min_verification_skip_rate=0.90,
            max_verified_claim_count=0,
            require_runtime_trace=False,
        ),
    )

    assert summary["enabled"] is True
    assert summary["skipped"] is True
    assert summary["claim_count"] == 2
    assert summary["saved_claim_count"] == 2
    assert summary["verified_claim_count"] == 0
    assert summary["skip_rate"] == 1.0
    assert metrics["verification_skip_rate"] == 1.0
    assert metrics["verifier_saved_claim_count"] == 2.0
    assert report["passed"] is True
    assert report["metrics"]["verification_skip_rate"] == 1.0
    assert report["policy"]["min_verification_skip_rate"] == 0.9
    json.dumps(summary)

    stage_only = ProductTrace(
        claims=claims[:1],
        events=(TraceEvent("verification_stage_decision", {"run_verifier": False}),),
    ).verification_stage_summary()
    assert stage_only["skipped"] is True
    assert stage_only["saved_claim_count"] == 1

    partial_trace = ProductTrace(
        claims=claims,
        verification_results=(
            VerificationResult(status=VerificationStatus.SUPPORTED, confidence=0.9),
        ),
        events=(
            TraceEvent(
                "verification_stage_decision",
                {
                    "run_verifier": True,
                    "verification_scope": "triggered",
                    "triggered_claim_ids": ("c2",),
                },
            ),
            TraceEvent(
                "initial_verification",
                {
                    "n_claims": len(claims),
                    "verification_scope": "triggered",
                    "verified_claim_ids": ("c2",),
                    "skipped_claim_ids": ("c1",),
                    "results": (
                        {"status": "supported", "confidence": 0.9, "evidence": ()},
                    ),
                },
            ),
        ),
    )
    partial = partial_trace.verification_stage_summary()
    partial_metrics = product_runtime_metrics(partial_trace)
    partial_report = evaluate_product_runtime_budget(
        partial_trace,
        ProductRuntimeBudgetPolicy(
            min_selective_claim_skip_rate=0.5,
            require_runtime_trace=False,
        ),
    )
    failing_partial_report = evaluate_product_runtime_budget(
        partial_trace,
        ProductRuntimeBudgetPolicy(
            min_selective_claim_skip_rate=0.75,
            require_runtime_trace=False,
        ),
    )
    assert partial["skipped"] is False
    assert partial["verification_scope"] == "triggered"
    assert partial["verified_claim_count"] == 1
    assert partial["saved_claim_count"] == 1
    assert partial["skip_rate"] == 0.5
    assert partial_metrics["selective_claim_skip_rate"] == 0.5
    assert partial_report["passed"] is True
    assert partial_report["policy"]["min_selective_claim_skip_rate"] == 0.5
    assert failing_partial_report["passed"] is False
    assert failing_partial_report["failures"][0]["metric"] == "selective_claim_skip_rate"


def test_product_runtime_budget_evaluates_trace_phase_limits():
    trace = ProductTrace(
        runtime_trace=RuntimeTrace(
            total_seconds=0.40,
            phases=(
                RuntimePhaseTiming("diagnostic_risk_decision", 0.05),
                RuntimePhaseTiming("initial_verification", 0.20),
            ),
        )
    )

    metrics = product_runtime_metrics(trace)
    report = evaluate_product_runtime_budget(
        trace,
        ProductRuntimeBudgetPolicy(
            max_total_seconds=0.50,
            max_phase_seconds={"initial_verification": 0.10},
        ),
    )

    assert metrics["total_seconds"] == 0.40
    assert metrics["phase_seconds"]["initial_verification"] == 0.20
    assert report["enabled"] is True
    assert report["passed"] is False
    assert report["failures"][0]["metric"] == "phase_seconds.initial_verification"
    assert report["failures"][0]["reason"] == "above 0.1"
    json.dumps(report)


def test_product_runtime_budget_checks_cache_hit_rates():
    trace = ProductTrace(
        metadata={
            "cache": {
                "verifier": {"size": 2, "hits": 1, "misses": 3},
                "retriever": {"size": 1, "hits": 3, "misses": 1},
            },
        },
        runtime_trace=RuntimeTrace(
            total_seconds=0.10,
            phases=(RuntimePhaseTiming("initial_verification", 0.05),),
        ),
    )

    report = evaluate_product_runtime_budget(
        trace,
        ProductRuntimeBudgetPolicy(
            min_cache_hit_rate=0.70,
            min_named_cache_hit_rate={"verifier": 0.50},
        ),
    )

    assert report["passed"] is False
    assert report["metrics"]["cache_hit_rate"] == 0.5
    assert report["metrics"]["named_cache_hit_rates"]["verifier"] == 0.25
    assert [failure["metric"] for failure in report["failures"]] == [
        "cache_hit_rate",
        "named_cache_hit_rate.verifier",
    ]


def test_product_runtime_budget_checks_phase_tail_latency():
    trace = ProductTrace(
        runtime_trace=RuntimeTrace(
            total_seconds=0.08,
            phases=(
                RuntimePhaseTiming("initial_verification", 0.01),
                RuntimePhaseTiming("initial_verification", 0.02),
                RuntimePhaseTiming("initial_verification", 0.05),
            ),
        ),
    )

    report = evaluate_product_runtime_budget(
        trace,
        ProductRuntimeBudgetPolicy(
            max_phase_p95_seconds={"initial_verification": 0.045},
            max_phase_p99_seconds={"initial_verification": 0.049},
        ),
    )

    assert report["passed"] is False
    assert round(report["metrics"]["phase_p95_seconds"]["initial_verification"], 6) == 0.047
    assert round(report["metrics"]["phase_p99_seconds"]["initial_verification"], 6) == 0.0494
    assert [failure["metric"] for failure in report["failures"]] == [
        "phase_p95_seconds.initial_verification",
        "phase_p99_seconds.initial_verification",
    ]


def test_product_runtime_budget_cache_only_policy_does_not_require_runtime_trace():
    trace = ProductTrace(
        metadata={"cache": {"verifier": {"size": 1, "hits": 1, "misses": 0}}},
        runtime_trace=None,
    )

    report = evaluate_product_runtime_budget(
        trace,
        ProductRuntimeBudgetPolicy(min_cache_hit_rate=0.90),
    )

    assert report["passed"] is True
    assert report["checks"][0]["metric"] == "cache_hit_rate"


def test_product_runtime_budget_fails_closed_when_trace_is_missing():
    report = evaluate_product_runtime_budget(
        ProductTrace(runtime_trace=None),
        ProductRuntimeBudgetPolicy(max_total_seconds=1.0),
    )

    assert report["enabled"] is True
    assert report["passed"] is False
    assert report["failures"][0]["metric"] == "runtime_trace"
    assert report["failures"][0]["reason"] == "missing"


def test_product_runtime_budget_policy_direct_constructor_parses_bool_strings():
    trace = ProductTrace(runtime_trace=None)
    policy = ProductRuntimeBudgetPolicy(
        max_total_seconds=1.0,
        require_runtime_trace="false",  # type: ignore[arg-type]
    )

    report = evaluate_product_runtime_budget(trace, policy)

    assert policy.require_runtime_trace is False
    assert report["passed"] is False
    assert [failure["metric"] for failure in report["failures"]] == ["total_seconds"]
    with pytest.raises(ValueError, match="require_runtime_trace"):
        ProductRuntimeBudgetPolicy(
            max_total_seconds=1.0,
            require_runtime_trace="maybe",  # type: ignore[arg-type]
        )


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


def test_product_trace_verification_route_cost_summary_matches_benchmark_fields():
    trace = ProductTrace(
        verification_results=(
            VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=0.9,
                metadata={
                    "selected_route": "structured_qa",
                    "matched_routes": ("structured_qa", "fallback"),
                    "total_duration_seconds": 0.01,
                    "selected_route_duration_seconds": 0.01,
                    "retrieval_hits": ({"id": "doc-1"}, {"id": "doc-2"}),
                },
            ),
            VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.5,
                metadata={
                    "selected_route": "retrieval_groundedness",
                    "matched_routes": ("structured_qa", "retrieval_groundedness"),
                    "skipped_routes": (
                        {
                            "route": "structured_qa",
                            "status": "insufficient_evidence",
                        },
                    ),
                    "total_duration_seconds": 0.04,
                    "selected_route_duration_seconds": 0.03,
                    "used_retrieval": True,
                    "retrieval_hit_count": 3,
                    "route_budget_limit": 2,
                    "route_budget_exhausted": True,
                    "unattempted_routes": ("fallback",),
                    "selected_route_was_fallthrough": True,
                },
            ),
            VerificationResult(status=VerificationStatus.NOT_APPLICABLE, confidence=1.0),
        )
    )

    summary = trace.verification_route_cost_summary()

    assert summary["total"] == 3
    assert summary["routed_total"] == 2
    assert summary["duration_observations"] == 2
    assert summary["mean_duration_seconds"] == 0.025
    assert summary["attempted_route_count_observations"] == 2
    assert summary["mean_attempted_route_count"] == 1.5
    assert summary["used_retrieval_count"] == 2
    assert summary["retrieval_use_rate"] == 2 / 3
    assert summary["retrieval_hit_count"] == 5
    assert summary["mean_retrieval_hits"] == 5 / 3
    assert summary["route_budget_limit_observations"] == 1
    assert summary["route_budget_exhausted_count"] == 1
    assert summary["route_budget_exhaustion_rate"] == 1.0
    assert summary["selected_fallthrough_budget_stop_count"] == 1
    assert summary["unattempted_route_count"] == 1
    assert summary["mean_unattempted_route_count"] == 1 / 3
    assert summary["by_route"]["retrieval_groundedness"]["mean_attempted_route_count"] == 2.0
    assert summary["by_route"]["retrieval_groundedness"]["route_budget_exhaustion_rate"] == 1.0
    json.dumps(summary)


def test_product_trace_route_cost_summary_treats_skipped_verifier_as_zero_cost():
    trace = ProductTrace(verification_results=())

    summary = trace.verification_route_cost_summary()
    report = evaluate_product_runtime_budget(
        trace,
        ProductRuntimeBudgetPolicy(
            max_mean_attempted_route_count=1.1,
            max_p99_route_duration_seconds=0.01,
            max_retrieval_use_rate=0.0,
            require_runtime_trace=False,
        ),
    )

    assert summary["total"] == 0
    assert summary["routed_total"] == 0
    assert summary["duration_observations"] == 0
    assert summary["mean_duration_seconds"] == 0.0
    assert summary["p99_duration_seconds"] == 0.0
    assert summary["mean_attempted_route_count"] == 0.0
    assert summary["retrieval_use_rate"] == 0.0
    assert report["passed"] is True
    assert [check["metric"] for check in report["checks"]] == [
        "p99_route_duration_seconds",
        "mean_attempted_route_count",
        "retrieval_use_rate",
    ]
    json.dumps(summary)


def test_product_trace_route_cost_summary_treats_unrouted_results_as_zero_cost():
    trace = ProductTrace(
        verification_results=(
            VerificationResult(status=VerificationStatus.NOT_APPLICABLE, confidence=1.0),
        )
    )

    summary = trace.verification_route_cost_summary()
    report = evaluate_product_runtime_budget(
        trace,
        ProductRuntimeBudgetPolicy(
            max_mean_attempted_route_count=0.0,
            max_p99_route_duration_seconds=0.0,
            max_retrieval_use_rate=0.0,
            require_runtime_trace=False,
        ),
    )

    assert summary["total"] == 1
    assert summary["routed_total"] == 0
    assert summary["unrouted_total"] == 1
    assert summary["mean_duration_seconds"] == 0.0
    assert summary["mean_attempted_route_count"] == 0.0
    assert summary["retrieval_use_rate"] == 0.0
    assert summary["by_route"]["unrouted"]["mean_duration_seconds"] == 0.0
    assert report["passed"] is True


def test_product_runtime_budget_checks_route_cost_without_runtime_trace():
    trace = ProductTrace(
        verification_results=(
            VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=0.9,
                metadata={
                    "selected_route": "structured_qa",
                    "total_duration_seconds": 0.01,
                    "retrieval_hits": (),
                },
            ),
            VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=0.9,
                metadata={
                    "selected_route": "retrieval_groundedness",
                    "skipped_routes": ({"route": "structured_qa"},),
                    "total_duration_seconds": 0.05,
                    "used_retrieval": True,
                    "retrieval_hit_count": 2,
                    "route_budget_limit": 1,
                    "route_budget_exhausted": True,
                    "unattempted_routes": ("fallback",),
                },
            ),
        ),
        runtime_trace=None,
    )

    report = evaluate_product_runtime_budget(
        trace,
        ProductRuntimeBudgetPolicy(
            max_mean_route_duration_seconds=0.02,
            max_route_duration_seconds=0.04,
            max_mean_attempted_route_count=1.2,
            max_route_budget_exhaustion_rate=0.0,
            max_retrieval_use_rate=0.25,
            max_retrieval_hit_count=1,
        ),
    )

    assert report["passed"] is False
    assert report["metrics"]["has_runtime_trace"] is False
    assert round(report["metrics"]["mean_route_duration_seconds"], 6) == 0.03
    assert report["metrics"]["max_route_duration_seconds"] == 0.05
    assert report["metrics"]["mean_attempted_route_count"] == 1.5
    assert report["metrics"]["route_budget_exhaustion_rate"] == 1.0
    assert report["metrics"]["route_budget_exhausted_count"] == 1.0
    assert report["metrics"]["unattempted_route_count"] == 1.0
    assert report["metrics"]["retrieval_use_rate"] == 0.5
    assert report["metrics"]["retrieval_hit_count"] == 2.0
    assert [failure["metric"] for failure in report["failures"]] == [
        "mean_route_duration_seconds",
        "max_route_duration_seconds",
        "mean_attempted_route_count",
        "route_budget_exhaustion_rate",
        "retrieval_use_rate",
        "retrieval_hit_count",
    ]


def test_product_promotion_contract_maps_release_candidate_budget(tmp_path):
    release_report = {
        "workflow": "release_candidate_comparison",
        "config": {
            "runtime_profile": "balanced",
            "inside_trigger_budget_policy": "quality_balanced",
            "max_runtime_total_seconds": 1.0,
            "max_mean_duration_seconds": 0.05,
            "max_p99_duration_seconds": 0.20,
            "max_max_duration_seconds": 0.25,
            "max_mean_attempted_route_count": 1.5,
            "max_route_budget_exhaustion_rate": 0.0,
            "max_retrieval_use_rate": 0.5,
            "max_retrieval_hit_count": 4,
            "min_claims_cache_hit_rate": 0.8,
            "min_verifier_trace_cache_hit_rate": 0.9,
            "required_route_min_selected": 200,
            "required_route_max_runtime_total_seconds": 8.0,
            "required_route_max_retrieval_hit_count": 450.0,
            "required_route_require_non_oracle_evidence": True,
            "required_route_require_retrieval_stress_control": True,
            "required_route_retrieval_stress_manifest": "artifacts/retrieval-stress/artifact-manifest.json",
            "required_route_min_stress_false_supported_rate": 0.90,
            "required_route_max_stress_false_refuted_rate": 0.05,
            "require_performance_score_dump_cache": True,
            "min_performance_score_dump_cache_jsonl_view_hit_rate": 0.5,
            "performance_drift_baseline_key": "performance_baseline:runtime-reference:0.8",
            "max_covariance_maha_last_auroc_drop": 0.05,
        },
        "decision": {
            "status": "promote",
            "recommended_readiness_record": "benchmark_manifest:readiness:0.8",
            "recommended_route_record": "benchmark_manifest:route:0.8",
            "recommended_performance_baseline_record": "performance_baseline:runtime:0.9",
            "recommended_selector_replay_candidate": "default",
            "recommended_product_runtime_drift_report": (
                "artifacts/runtime-drift/product-runtime-drift.json"
            ),
            "product_trace_replay_workflow_status": "promote",
            "world_model_signal_workflow_status": "promote",
            "recommended_world_model_signal_workflow_report": (
                "artifacts/world-model-signal/world-model-signal-workflow.json"
            ),
            "triple_extraction_fixture_matrix_status": "promote",
            "recommended_triple_extraction_fixture_matrix_report": (
                "artifacts/triple-extraction-fixture-matrix/"
                "triple-extraction-fixture-matrix.json"
            ),
            "feedback_policy_workflow_status": "promote",
            "recommended_feedback_policy_workflow_report": (
                "artifacts/feedback-policy-workflow/feedback-policy-workflow.json"
            ),
            "recommended_feedback_policy_candidate_control_policy": (
                "artifacts/feedback-policy-workflow/candidate-control-policy.json"
            ),
            "recommended_feedback_policy_candidate_control_defaults": (
                "artifacts/feedback-policy-workflow/candidate-control-defaults.json"
            ),
            "selector_replay_status": "promote",
            "product_runtime_drift_status": "promote",
            "recommended_route": "structured_state",
            "required_route_baseline_records": [
                "benchmark_manifest:retrieval-structured-qa:0.5"
            ],
            "required_route_baseline_status": "promote",
        },
        "release_candidate": {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "runtime": {
                "layer": -12,
                "batch_size": 2,
                "covariance_mode": "low_rank",
                "covariance_low_rank": 8,
            },
            "quality": {
                "covariance_tradeoff_gate": {
                    "passed": True,
                    "status": "quality_preserved",
                    "selected_covariance_mode": "low_rank",
                    "selected_covariance_low_rank": 8,
                    "selected_maha_last_delta_vs_baseline": -0.01,
                },
            },
            "performance_baseline_record": "performance_baseline:runtime:0.9",
            "performance_evidence_bundle": {
                "status": "promote",
                "release_ready": True,
                "recommendation": {
                    "cache_tuning_status": "ok",
                    "best_quality_signal": "truth_proj",
                    "best_quality_auroc": 0.91,
                },
                "cost": {
                    "uncached_total_seconds": 10.0,
                    "cached_total_ratio": 0.50,
                    "cache_only_total_ratio": 0.02,
                },
                "score_dump_cache": {
                    "enabled": True,
                    "source_count": 1,
                    "cache_entries": 5,
                    "totals": {
                        "fingerprint": {
                            "hits": 1,
                            "misses": 2,
                            "writes": 2,
                            "attempts": 3,
                            "hit_rate": 1 / 3,
                        },
                        "jsonl_summary": {
                            "hits": 1,
                            "misses": 1,
                            "writes": 1,
                            "attempts": 2,
                            "hit_rate": 0.5,
                        },
                        "jsonl_view": {
                            "hits": 3,
                            "misses": 2,
                            "writes": 2,
                            "attempts": 5,
                            "hit_rate": 0.6,
                        },
                    },
                },
            },
            "product_trace_replay_workflow": {
                "report_path": "artifacts/trace-replay-workflow/product-trace-replay-workflow.json",
                "manifest_path": "artifacts/trace-replay-workflow/artifact-manifest.json",
                "source": "registry",
                "registry": "artifacts/release-registry.json",
                "record_key": "report:trace-replay-workflow:0.1",
                "report_status": "promote",
                "selector_replay_report_path": (
                    "artifacts/selector/runtime-profile-selector-replay.json"
                ),
                "product_runtime_drift_report_path": (
                    "artifacts/runtime-drift/product-runtime-drift.json"
                ),
            },
            "world_model_signal_workflow": {
                "report_path": "artifacts/world-model-signal/world-model-signal-workflow.json",
                "manifest_path": "artifacts/world-model-signal/artifact-manifest.json",
                "source": "registry",
                "registry": "artifacts/release-registry.json",
                "record_key": "report:world-model-signal-workflow:0.1",
                "workflow": "world_model_signal_calibration_workflow",
                "status": "promote",
                "release_gate_status": "promote",
                "trace_gap_max": 0.0,
                "conflict_positive_count": 4,
                "calibrated_conflict_signal_count": 1,
                "blocking_reasons": [],
            },
            "triple_extraction_fixture_matrix": {
                "report_path": (
                    "artifacts/triple-extraction-fixture-matrix/"
                    "triple-extraction-fixture-matrix.json"
                ),
                "manifest_path": (
                    "artifacts/triple-extraction-fixture-matrix/artifact-manifest.json"
                ),
                "source": "registry",
                "registry": "artifacts/release-registry.json",
                "record_key": "report:triple-extraction-fixture-matrix:0.1",
                "workflow": "triple_extraction_fixture_matrix",
                "status": "promote",
                "n_corpora": 2,
                "promoted_corpora": 2,
                "distinct_predicate_count": 6,
                "distinct_predicates": [
                    "capital_of",
                    "currency_of",
                    "headquarters_location_of",
                    "inception_of",
                    "manufacturer_of",
                    "official_language_of",
                ],
                "mean_baseline_f1": 0.5,
                "mean_best_f1": 1.0,
                "mean_f1_lift": 0.5,
            },
            "feedback_policy_workflow": {
                "report_path": "artifacts/feedback-policy-workflow/feedback-policy-workflow.json",
                "manifest_path": "artifacts/feedback-policy-workflow/artifact-manifest.json",
                "source": "registry",
                "registry": "artifacts/release-registry.json",
                "record_key": "report:feedback-policy-workflow:0.1",
                "report_status": "recommend",
                "promotion_decision": "promote_candidate_policy",
                "candidate_control_policy": (
                    "artifacts/feedback-policy-workflow/candidate-control-policy.json"
                ),
                "candidate_control_policy_config": {
                    "unsupported_action": "clarify",
                    "compound_risk_action": "abstain",
                    "compound_verification_escalates": False,
                },
                "candidate_control_defaults": (
                    "artifacts/feedback-policy-workflow/candidate-control-defaults.json"
                ),
                "candidate_control_defaults_config": {
                    "staged_verification": True,
                    "max_verifier_route_attempts": 2,
                },
                "matched_feedback_count": 30,
                "accepted_but_wrong_rate": 0.03,
                "retrieved_failure_rate": 0.04,
                "abstain_false_positive_rate": 0.02,
                "final_answered_but_wrong_rate": 0.07,
                "final_answer_false_block_rate": 0.01,
                "safety_coverage_rate": 1.0,
                "unknown_safety_issue_rate": 0.0,
            },
            "release_efficiency": {
                "report_path": "artifacts/efficiency/release-efficiency-report.json",
                "manifest_path": "artifacts/efficiency/artifact-manifest.json",
                "workflow": "release_efficiency_report",
                "status": "promote",
                "decision": {
                    "recommended_profile": "balanced",
                    "recommended_efficiency_score": 2.0,
                },
                "summary": {
                    "profile_count": 3,
                    "quality_passed": True,
                    "trace_record_cache_hit_profile_count": 1,
                },
            },
            "selector_replay": {
                "report_path": "artifacts/selector/runtime-profile-selector-replay.json",
                "manifest_path": "artifacts/selector/artifact-manifest.json",
                "recommended_candidate": "default",
                "recommended_policy_path": "artifacts/selector/policies/default.json",
                "recommended": {
                    "candidate": "default",
                    "status": "promote",
                    "policy_path": "artifacts/selector/policies/default.json",
                    "estimated_cost_units_mean": 1.2,
                    "observed_runtime_coverage_rate": 1.0,
                    "observed_runtime_delta_coverage_rate": 1.0,
                    "observed_selected_total_seconds_mean": 0.10,
                    "observed_selected_minus_original_seconds_mean": -0.02,
                    "observed_selected_to_original_ratio_mean": 0.80,
                },
            },
            "product_runtime_drift": {
                "report_path": "artifacts/runtime-drift/product-runtime-drift.json",
                "manifest_path": "artifacts/runtime-drift/artifact-manifest.json",
                "baseline": {"path": "artifacts/runtime-baseline/product-runtime-baseline.json"},
                "current": {
                    "path": "artifacts/runtime-current/product-runtime-baseline.json",
                    "optimization": {
                        "policy_hints": {
                            "candidate_control_defaults": {
                                "max_verifier_route_attempts": 2,
                            },
                        },
                    },
                },
                "summary": {
                    "gate_enabled": True,
                    "compared_metric_count": 9,
                    "blocked_metric_count": 0,
                },
            },
            "adapter_family_matrix": {
                "matrix_path": "artifacts/adapter-family-matrix.json",
                "required_routes": ["structured_state", "state_transition", "retrieval_groundedness"],
                "routes": ["structured_qa", "structured_state", "state_transition", "retrieval_groundedness"],
                "promoted_routes": [
                    "structured_qa",
                    "structured_state",
                    "state_transition",
                    "retrieval_groundedness",
                ],
                "promotion_status": "promote",
            },
            "verifier_route": {
                "route": "structured_state",
                "mean_duration_seconds": 0.01,
                "p99_duration_seconds": 0.02,
                "max_duration_seconds": 0.03,
                "mean_attempted_route_count": 1.0,
                "retrieval_use_rate": 0.0,
            },
            "required_route_baselines": {
                "records": ["benchmark_manifest:retrieval-structured-qa:0.5"],
                "routes": ["retrieval_structured_qa"],
                "manifest_paths": ["artifacts/retrieval/audit-manifest.json"],
                "registry": "artifacts/staged-route-registry.json",
            },
            "manifests": {
                "readiness_manifest": "artifacts/readiness/artifact-manifest.json",
                "route_manifest": "artifacts/route/artifact-manifest.json",
                "performance_manifest": "artifacts/performance/artifact-manifest.json",
                "product_trace_replay_workflow_manifest": (
                    "artifacts/trace-replay-workflow/artifact-manifest.json"
                ),
                "world_model_signal_workflow_manifest": (
                    "artifacts/world-model-signal/artifact-manifest.json"
                ),
                "triple_extraction_fixture_matrix_manifest": (
                    "artifacts/triple-extraction-fixture-matrix/artifact-manifest.json"
                ),
                "feedback_policy_workflow_manifest": (
                    "artifacts/feedback-policy-workflow/artifact-manifest.json"
                ),
                "release_efficiency_manifest": "artifacts/efficiency/artifact-manifest.json",
                "selector_replay_manifest": "artifacts/selector/artifact-manifest.json",
                "product_runtime_drift_manifest": "artifacts/runtime-drift/artifact-manifest.json",
                "adapter_family_matrix_report": "artifacts/adapter-family-matrix.json",
            },
        },
        "performance_baseline_gate": {
            "covariance_tradeoff_gate": {
                "passed": True,
                "status": "quality_preserved",
                "selected_covariance_mode": "low_rank",
                "selected_covariance_low_rank": 8,
                "selected_maha_last_delta_vs_baseline": -0.02,
            },
            "performance_trend_gate": {
                "passed": True,
                "reference_record_key": "performance_baseline:runtime-reference:0.8",
                "metrics": {
                    "uncached_total_seconds": {"observed_ratio": 1.25},
                    "cached_total_seconds": {"observed_ratio": 1.20},
                    "cache_only_total_seconds": {"observed_ratio": 1.0},
                    "score_dump_cache_jsonl_view_hit_rate": {"observed_drop": 0.3},
                },
            },
        },
    }
    registry_workflow = {
        "workflow": "release_candidate_registry_workflow",
        "release_candidate_comparison": release_report,
    }
    contract_path = tmp_path / "release-workflow.json"
    contract_path.write_text(json.dumps(registry_workflow), encoding="utf-8")

    contract = ProductPromotionContract.from_json(contract_path)
    direct_policy = product_runtime_budget_policy_from_release_candidate(release_report)
    roundtrip = ProductPromotionContract.from_mapping(contract.to_dict())

    assert contract.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert roundtrip.world_model_signal_workflow == contract.world_model_signal_workflow
    assert roundtrip.triple_extraction_fixture_matrix == (
        contract.triple_extraction_fixture_matrix
    )
    assert contract.runtime["layer"] == -12
    assert contract.verifier_route["route"] == "structured_state"
    assert contract.metadata["runtime_profile"] == "balanced"
    assert contract.metadata["recommended_performance_baseline_record"] == "performance_baseline:runtime:0.9"
    assert contract.metadata["performance_baseline_record"] == "performance_baseline:runtime:0.9"
    assert contract.metadata["performance_evidence_bundle_status"] == "promote"
    assert contract.metadata["performance_evidence_bundle_release_ready"] is True
    assert contract.metadata["performance_cache_tuning_status"] == "ok"
    assert contract.metadata["performance_uncached_total_seconds"] == 10.0
    assert contract.metadata["performance_cached_total_ratio"] == 0.50
    assert contract.metadata["performance_cache_only_total_ratio"] == 0.02
    assert contract.metadata["performance_score_dump_cache_required"] is True
    assert contract.metadata["performance_score_dump_cache_min_jsonl_view_hit_rate"] == 0.5
    assert contract.metadata["performance_score_dump_cache_source_count"] == 1
    assert contract.metadata["performance_score_dump_cache_jsonl_view_hit_rate"] == 0.6
    assert contract.metadata["performance_drift_baseline_record"] == (
        "performance_baseline:runtime-reference:0.8"
    )
    assert contract.metadata["performance_trend_gate_passed"] is True
    assert contract.metadata["performance_trend_reference_record"] == (
        "performance_baseline:runtime-reference:0.8"
    )
    assert contract.metadata["performance_uncached_total_seconds_ratio_to_drift_baseline"] == 1.25
    assert contract.metadata[
        "performance_score_dump_cache_jsonl_view_hit_rate_drop_from_drift_baseline"
    ] == 0.3
    assert contract.metadata["max_covariance_maha_last_auroc_drop"] == 0.05
    assert contract.metadata["readiness_covariance_tradeoff_gate_passed"] is True
    assert contract.metadata["readiness_covariance_tradeoff_status"] == "quality_preserved"
    assert contract.metadata["readiness_covariance_selected_mode"] == "low_rank"
    assert contract.metadata["readiness_covariance_selected_low_rank"] == 8
    assert contract.metadata["readiness_covariance_maha_last_delta_vs_baseline"] == -0.01
    assert contract.metadata["performance_covariance_tradeoff_gate_passed"] is True
    assert contract.metadata["performance_covariance_tradeoff_status"] == "quality_preserved"
    assert contract.metadata["performance_covariance_selected_mode"] == "low_rank"
    assert contract.metadata["performance_covariance_selected_low_rank"] == 8
    assert contract.metadata["performance_covariance_maha_last_delta_vs_baseline"] == -0.02
    assert contract.metadata["performance_manifest"] == "artifacts/performance/artifact-manifest.json"
    assert contract.metadata["recommended_selector_replay_candidate"] == "default"
    assert contract.metadata["recommended_product_runtime_drift_report"] == (
        "artifacts/runtime-drift/product-runtime-drift.json"
    )
    assert contract.product_trace_replay_workflow == {
        "report_path": "artifacts/trace-replay-workflow/product-trace-replay-workflow.json",
        "manifest_path": "artifacts/trace-replay-workflow/artifact-manifest.json",
        "source": "registry",
        "registry": "artifacts/release-registry.json",
        "record_key": "report:trace-replay-workflow:0.1",
        "report_status": "promote",
        "selector_replay_report_path": "artifacts/selector/runtime-profile-selector-replay.json",
        "product_runtime_drift_report_path": "artifacts/runtime-drift/product-runtime-drift.json",
    }
    assert contract.metadata["product_trace_replay_workflow_status"] == "promote"
    assert contract.metadata["product_trace_replay_workflow_report"] == (
        "artifacts/trace-replay-workflow/product-trace-replay-workflow.json"
    )
    assert contract.metadata["product_trace_replay_workflow_manifest"] == (
        "artifacts/trace-replay-workflow/artifact-manifest.json"
    )
    assert contract.metadata["product_trace_replay_workflow_source"] == "registry"
    assert contract.metadata["product_trace_replay_workflow_record"] == (
        "report:trace-replay-workflow:0.1"
    )
    assert contract.metadata["product_trace_replay_workflow_selector_replay_report"] == (
        "artifacts/selector/runtime-profile-selector-replay.json"
    )
    assert contract.metadata["product_trace_replay_workflow_runtime_drift_report"] == (
        "artifacts/runtime-drift/product-runtime-drift.json"
    )
    assert contract.world_model_signal_workflow == {
        "report_path": "artifacts/world-model-signal/world-model-signal-workflow.json",
        "manifest_path": "artifacts/world-model-signal/artifact-manifest.json",
        "source": "registry",
        "registry": "artifacts/release-registry.json",
        "record_key": "report:world-model-signal-workflow:0.1",
        "workflow": "world_model_signal_calibration_workflow",
        "status": "promote",
        "release_gate_status": "promote",
        "trace_gap_max": 0.0,
        "conflict_positive_count": 4,
        "calibrated_conflict_signal_count": 1,
        "blocking_reasons": [],
    }
    assert contract.metadata["world_model_signal_workflow_status"] == "promote"
    assert contract.metadata["recommended_world_model_signal_workflow_report"] == (
        "artifacts/world-model-signal/world-model-signal-workflow.json"
    )
    assert contract.metadata["world_model_signal_workflow_release_gate_status"] == "promote"
    assert contract.metadata["world_model_signal_workflow_trace_gap_max"] == 0.0
    assert contract.metadata["world_model_signal_workflow_conflict_positive_count"] == 4
    assert contract.metadata["world_model_signal_workflow_calibrated_conflict_signal_count"] == 1
    assert contract.triple_extraction_fixture_matrix["record_key"] == (
        "report:triple-extraction-fixture-matrix:0.1"
    )
    assert contract.triple_extraction_fixture_matrix["distinct_predicate_count"] == 6
    assert contract.metadata["triple_extraction_fixture_matrix_status"] == "promote"
    assert contract.metadata["recommended_triple_extraction_fixture_matrix_report"].endswith(
        "triple-extraction-fixture-matrix.json"
    )
    assert contract.metadata["triple_extraction_fixture_matrix_report"].endswith(
        "triple-extraction-fixture-matrix.json"
    )
    assert contract.metadata["triple_extraction_fixture_matrix_record"] == (
        "report:triple-extraction-fixture-matrix:0.1"
    )
    assert contract.metadata["triple_extraction_fixture_matrix_distinct_predicate_count"] == 6
    assert contract.metadata["triple_extraction_fixture_matrix_mean_best_f1"] == 1.0
    assert contract.metadata["triple_extraction_fixture_matrix_mean_f1_lift"] == 0.5
    assert contract.metadata["triple_extraction_fixture_matrix_min_corpora"] is None
    assert contract.control_policy_config["unsupported_action"] == "clarify"
    assert contract.control_policy_config["compound_verification_escalates"] is False
    assert contract.feedback_policy_workflow["record_key"] == "report:feedback-policy-workflow:0.1"
    assert contract.feedback_policy_workflow["manifest_path"] == (
        "artifacts/feedback-policy-workflow/artifact-manifest.json"
    )
    assert contract.feedback_policy_workflow["candidate_control_policy_config"][
        "unsupported_action"
    ] == "clarify"
    assert contract.feedback_policy_workflow["candidate_control_defaults_config"][
        "max_verifier_route_attempts"
    ] == 2
    assert contract.feedback_policy_workflow["final_answered_but_wrong_rate"] == 0.07
    assert contract.feedback_policy_workflow["final_answer_false_block_rate"] == 0.01
    assert contract.metadata["recommended_feedback_policy_workflow_report"] == (
        "artifacts/feedback-policy-workflow/feedback-policy-workflow.json"
    )
    assert contract.metadata["feedback_policy_workflow_status"] == "promote"
    assert contract.metadata["feedback_policy_workflow_final_answered_but_wrong_rate"] == 0.07
    assert contract.metadata["feedback_policy_workflow_final_answer_false_block_rate"] == 0.01
    assert contract.release_efficiency["recommended_profile"] == "balanced"
    assert contract.release_efficiency["recommended_efficiency_score"] == 2.0
    assert contract.metadata["release_efficiency_report"] == (
        "artifacts/efficiency/release-efficiency-report.json"
    )
    assert contract.metadata["release_efficiency_manifest"] == (
        "artifacts/efficiency/artifact-manifest.json"
    )
    assert contract.metadata["release_efficiency_recommended_profile"] == "balanced"
    assert contract.metadata["release_efficiency_score"] == 2.0
    assert contract.metadata["release_efficiency_quality_passed"] is True
    assert contract.metadata["release_efficiency_trace_record_cache_hit_profile_count"] == 1
    assert contract.metadata["selector_replay_status"] == "promote"
    assert contract.metadata["selector_replay_report"] == (
        "artifacts/selector/runtime-profile-selector-replay.json"
    )
    assert contract.metadata["selector_replay_manifest"] == "artifacts/selector/artifact-manifest.json"
    assert contract.metadata["selector_replay_recommended_policy_path"] == (
        "artifacts/selector/policies/default.json"
    )
    assert contract.metadata["selector_replay_recommended"]["candidate"] == "default"
    assert contract.metadata["selector_replay_estimated_cost_units_mean"] == 1.2
    assert contract.metadata["selector_replay_observed_runtime_coverage_rate"] == 1.0
    assert contract.metadata["selector_replay_observed_runtime_delta_coverage_rate"] == 1.0
    assert contract.metadata["selector_replay_observed_selected_total_seconds_mean"] == 0.10
    assert contract.metadata["selector_replay_observed_selected_minus_original_seconds_mean"] == -0.02
    assert contract.metadata["selector_replay_observed_selected_to_original_ratio_mean"] == 0.80
    assert contract.metadata["product_runtime_drift_status"] == "promote"
    assert contract.metadata["product_runtime_drift_report"] == (
        "artifacts/runtime-drift/product-runtime-drift.json"
    )
    assert contract.metadata["product_runtime_drift_manifest"] == (
        "artifacts/runtime-drift/artifact-manifest.json"
    )
    assert contract.metadata["product_runtime_drift_baseline_path"] == (
        "artifacts/runtime-baseline/product-runtime-baseline.json"
    )
    assert contract.metadata["product_runtime_drift_current_path"] == (
        "artifacts/runtime-current/product-runtime-baseline.json"
    )
    assert contract.control_defaults == {"max_verifier_route_attempts": 2}
    assert contract.to_dict()["control_policy_config"]["unsupported_action"] == "clarify"
    assert contract.to_dict()["control_defaults"] == {"max_verifier_route_attempts": 2}
    assert contract.metadata["product_runtime_drift_gate_enabled"] is True
    assert contract.metadata["product_runtime_drift_compared_metric_count"] == 9
    assert contract.metadata["product_runtime_drift_blocked_metric_count"] == 0
    assert contract.metadata["adapter_family_matrix_report"] == "artifacts/adapter-family-matrix.json"
    assert contract.metadata["adapter_family_required_routes"] == [
        "structured_state",
        "state_transition",
        "retrieval_groundedness",
    ]
    assert contract.metadata["adapter_family_promotion_status"] == "promote"
    assert contract.metadata["required_route_baseline_status"] == "promote"
    assert contract.metadata["required_route_baseline_records"] == [
        "benchmark_manifest:retrieval-structured-qa:0.5"
    ]
    assert contract.metadata["required_route_baseline_routes"] == ["retrieval_structured_qa"]
    assert contract.metadata["required_route_baseline_manifests"] == [
        "artifacts/retrieval/audit-manifest.json"
    ]
    assert contract.metadata["required_route_budget_policy"]["required_route_min_selected"] == 200
    assert contract.metadata["required_route_budget_policy"]["required_route_max_retrieval_hit_count"] == 450.0
    assert (
        contract.metadata["required_route_budget_policy"]["required_route_require_non_oracle_evidence"]
        is True
    )
    assert (
        contract.metadata["required_route_budget_policy"][
            "required_route_require_retrieval_stress_control"
        ]
        is True
    )
    assert contract.metadata["required_route_budget_policy"]["required_route_retrieval_stress_manifest"] == (
        "artifacts/retrieval-stress/artifact-manifest.json"
    )
    assert contract.metadata["required_route_budget_policy"][
        "required_route_min_stress_false_supported_rate"
    ] == 0.90
    assert contract.metadata["required_route_budget_policy"][
        "required_route_max_stress_false_refuted_rate"
    ] == 0.05
    assert contract.runtime_budget_policy == direct_policy
    assert contract.runtime_budget_policy.max_total_seconds == 1.0
    assert contract.runtime_budget_policy.max_mean_route_duration_seconds == 0.05
    assert contract.runtime_budget_policy.max_p99_route_duration_seconds == 0.20
    assert contract.runtime_budget_policy.max_route_duration_seconds == 0.25
    assert contract.runtime_budget_policy.max_mean_attempted_route_count == 1.5
    assert contract.runtime_budget_policy.max_route_budget_exhaustion_rate == 0.0
    assert contract.runtime_budget_policy.max_retrieval_use_rate == 0.5
    assert contract.runtime_budget_policy.max_retrieval_hit_count == 4.0
    assert contract.runtime_budget_policy.min_named_cache_hit_rate == {
        "claims": 0.8,
        "verifier_trace": 0.9,
    }
    assert roundtrip == contract
    assert roundtrip.control_policy_config == contract.control_policy_config
    json.dumps(contract.to_dict())


def test_product_promotion_contract_loader_selects_default_and_metadata(tmp_path):
    missing_path = tmp_path / "missing.json"
    contract_path = tmp_path / "product-promotion-contract.json"
    ProductPromotionContract(
        model_id="demo-model",
        runtime={"layer": -2},
        verifier_route={"route": "structured_qa"},
        runtime_budget_policy=ProductRuntimeBudgetPolicy(max_mean_attempted_route_count=1.1),
        source_workflow="release_candidate_comparison",
        source_status="promote",
        product_trace_replay_workflow={
            "report_path": "trace-replay-workflow.json",
            "record_key": "report:trace-replay-workflow:0.1",
        },
        world_model_signal_workflow={
            "report_path": "world-model-signal-workflow.json",
            "record_key": "report:world-model-signal-workflow:0.1",
            "release_gate_status": "promote",
            "trace_gap_max": 0.0,
        },
        control_policy_config={
            "unsupported_action": "clarify",
            "compound_verification_escalates": False,
        },
        feedback_policy_workflow={
            "report_path": "feedback-policy-workflow.json",
            "record_key": "report:feedback-policy-workflow:0.1",
            "promotion_decision": "promote_candidate_policy",
        },
        triple_extraction_fixture_matrix={
            "report_path": "triple-extraction-fixture-matrix.json",
            "record_key": "report:triple-extraction-fixture-matrix:0.1",
            "status": "promote",
            "distinct_predicate_count": 6,
        },
        release_efficiency={
            "report_path": "release-efficiency.json",
            "recommended_profile": "balanced",
        },
        control_defaults={"max_verifier_route_attempts": 3},
        metadata={"selector_replay_status": "promote"},
    ).save_json(contract_path)

    assert first_existing_product_promotion_contract_path((missing_path, contract_path)) == contract_path
    assert load_product_promotion_contract(default_paths=(missing_path,)) is None

    loaded = load_product_promotion_contract(default_paths=(missing_path, contract_path))
    assert loaded is not None
    assert loaded.path == contract_path
    assert loaded.source == str(contract_path)
    assert loaded.contract.model_id == "demo-model"
    assert loaded.contract.runtime_budget_policy.max_mean_attempted_route_count == 1.1

    metadata = loaded.runtime_metadata(budget_enabled=True)
    assert metadata["promotion_contract_source"] == str(contract_path)
    assert metadata["promotion_contract_budget_enabled"] is True
    assert metadata["promotion_contract_model_id"] == "demo-model"
    assert metadata["promotion_contract_runtime"] == {"layer": -2}
    assert metadata["promotion_contract_verifier_route"] == {"route": "structured_qa"}
    assert metadata["promotion_contract_control_policy_config"]["unsupported_action"] == "clarify"
    assert metadata["promotion_contract_control_policy_config"][
        "compound_verification_escalates"
    ] is False
    assert metadata["promotion_contract_control_defaults"] == {
        "max_verifier_route_attempts": 3
    }
    assert metadata["promotion_contract_product_trace_replay_workflow"] == {
        "report_path": "trace-replay-workflow.json",
        "record_key": "report:trace-replay-workflow:0.1",
    }
    assert metadata["promotion_contract_world_model_signal_workflow"] == {
        "report_path": "world-model-signal-workflow.json",
        "record_key": "report:world-model-signal-workflow:0.1",
        "release_gate_status": "promote",
        "trace_gap_max": 0.0,
    }
    assert metadata["promotion_contract_feedback_policy_workflow"] == {
        "report_path": "feedback-policy-workflow.json",
        "record_key": "report:feedback-policy-workflow:0.1",
        "promotion_decision": "promote_candidate_policy",
    }
    assert metadata["promotion_contract_triple_extraction_fixture_matrix"] == {
        "report_path": "triple-extraction-fixture-matrix.json",
        "record_key": "report:triple-extraction-fixture-matrix:0.1",
        "status": "promote",
        "distinct_predicate_count": 6,
    }
    assert metadata["promotion_contract_release_efficiency"] == {
        "report_path": "release-efficiency.json",
        "recommended_profile": "balanced",
    }
    assert metadata["promotion_contract_metadata"] == {"selector_replay_status": "promote"}
    assert product_promotion_contract_metadata(None, source=None, budget_enabled=True) == {
        "promotion_contract_source": None,
        "promotion_contract_budget_enabled": False,
    }


def test_product_runtime_evidence_bundle_loads_manifest_and_registry_lazily(tmp_path):
    contract_path = tmp_path / "product-promotion-contract.json"
    manifest_path = tmp_path / "artifact-manifest.json"
    registry_path = tmp_path / "registry.json"
    selfcheck_dir = tmp_path / "selfcheck"
    selfcheck_dir.mkdir()
    selfcheck_report_path = selfcheck_dir / "workflow.json"
    selfcheck_manifest_path = selfcheck_dir / "artifact-manifest.json"
    selfcheck_registry_path = selfcheck_dir / "registry.json"
    world_model_dir = tmp_path / "world-model-signal"
    world_model_dir.mkdir()
    world_model_report_path = world_model_dir / "workflow.json"
    world_model_manifest_path = world_model_dir / "artifact-manifest.json"
    world_model_registry_path = world_model_dir / "registry.json"
    triple_matrix_dir = tmp_path / "triple-extraction-fixture-matrix"
    triple_matrix_dir.mkdir()
    triple_matrix_report_path = triple_matrix_dir / "matrix.json"
    triple_matrix_manifest_path = triple_matrix_dir / "artifact-manifest.json"
    triple_matrix_registry_path = triple_matrix_dir / "registry.json"
    selfcheck_report_path.write_text(
        json.dumps({
            "workflow": "selfcheck_signal_fusion_workflow",
            "status": "promote",
            "sample_quality": {"status": "pass", "passed": True},
            "fusion_summary": {"runs": [{"name": "tiny", "auroc": 0.7}]},
        }),
        encoding="utf-8",
    )
    selfcheck_manifest_path.write_text(
        json.dumps(
            build_artifact_manifest(
                {"selfcheck_signal_fusion_workflow": selfcheck_report_path},
                root=selfcheck_dir,
                metadata={"workflow": "selfcheck_signal_fusion_workflow"},
            )
        ),
        encoding="utf-8",
    )
    ArtifactRegistry.load_json(selfcheck_registry_path).record_report(
        name="selfcheck-signal-fusion-workflow",
        path=selfcheck_report_path,
        version="0.1",
        metadata={"artifact_manifest": str(selfcheck_manifest_path)},
    ).save_json()
    world_model_report_path.write_text(
        json.dumps({
            "workflow": "world_model_signal_calibration_workflow",
            "release_gate": {
                "status": "promote",
                "passed": True,
                "score_summary": {
                    "world_model_trace_gap": {"max": 0.0},
                    "world_model_conflict": {"positive_count": 4},
                },
                "calibrated_conflict_signals": [
                    {"signal": "world_model_conflict", "passes_calibration_gate": True}
                ],
            },
        }),
        encoding="utf-8",
    )
    world_model_manifest_path.write_text(
        json.dumps(
            build_artifact_manifest(
                {"world_model_signal_workflow": world_model_report_path},
                root=world_model_dir,
                metadata={"workflow": "world_model_signal_calibration_workflow"},
            )
        ),
        encoding="utf-8",
    )
    ArtifactRegistry.load_json(world_model_registry_path).record_report(
        name="world-model-signal-workflow",
        path=world_model_report_path,
        version="0.1",
        metadata={"artifact_manifest": str(world_model_manifest_path)},
    ).save_json()
    triple_matrix_report_path.write_text(
        json.dumps({
            "workflow": "triple_extraction_fixture_matrix",
            "status": "promote",
            "n_corpora": 2,
            "promoted_corpora": 2,
            "distinct_predicate_count": 6,
            "distinct_predicates": [
                "capital_of",
                "currency_of",
                "headquarters_location_of",
                "inception_of",
                "manufacturer_of",
                "official_language_of",
            ],
            "mean_best_f1": 1.0,
            "mean_f1_lift": 0.5,
        }),
        encoding="utf-8",
    )
    triple_matrix_manifest_path.write_text(
        json.dumps(
            build_artifact_manifest(
                {"triple_extraction_fixture_matrix": triple_matrix_report_path},
                root=triple_matrix_dir,
                metadata={"workflow": "triple_extraction_fixture_matrix"},
            )
        ),
        encoding="utf-8",
    )
    ArtifactRegistry.load_json(triple_matrix_registry_path).record_report(
        name="triple-extraction-fixture-matrix",
        path=triple_matrix_report_path,
        version="0.1",
        metadata={"artifact_manifest": str(triple_matrix_manifest_path)},
    ).save_json()
    ProductPromotionContract(
        model_id="demo-model",
        runtime={"layer": -2},
        verifier_route={"route": "structured_qa"},
        runtime_budget_policy=ProductRuntimeBudgetPolicy(max_retrieval_use_rate=0.0),
        source_workflow="release_candidate_comparison",
        source_status="promote",
        product_trace_replay_workflow={
            "report_path": "trace-replay-workflow.json",
            "selector_replay_report_path": "selector-replay.json",
            "product_runtime_drift_report_path": "runtime-drift.json",
        },
        selfcheck_signal_fusion_workflow={
            "report_path": "selfcheck/workflow.json",
            "manifest_path": "selfcheck/artifact-manifest.json",
            "registry": "selfcheck/registry.json",
            "record_key": "report:selfcheck-signal-fusion-workflow:0.1",
            "status": "promote",
            "sample_quality_status": "pass",
            "sample_quality_passed": True,
            "fusion_run_count": 1,
            "geometry_fusion_artifact_count": 1,
            "enhanced_score_dump_count": 1,
        },
        world_model_signal_workflow={
            "report_path": "world-model-signal/workflow.json",
            "manifest_path": "world-model-signal/artifact-manifest.json",
            "registry": "world-model-signal/registry.json",
            "record_key": "report:world-model-signal-workflow:0.1",
            "status": "promote",
            "release_gate_status": "promote",
            "trace_gap_max": 0.0,
            "conflict_positive_count": 4,
            "calibrated_conflict_signal_count": 1,
        },
        triple_extraction_fixture_matrix={
            "report_path": "triple-extraction-fixture-matrix/matrix.json",
            "manifest_path": "triple-extraction-fixture-matrix/artifact-manifest.json",
            "registry": "triple-extraction-fixture-matrix/registry.json",
            "record_key": "report:triple-extraction-fixture-matrix:0.1",
            "status": "promote",
            "n_corpora": 2,
            "promoted_corpora": 2,
            "distinct_predicate_count": 6,
            "mean_best_f1": 1.0,
            "mean_f1_lift": 0.5,
        },
        metadata={"product_runtime_drift_status": "promote"},
    ).save_json(contract_path)
    manifest = build_artifact_manifest(
        {"product_promotion_contract": contract_path},
        root=tmp_path,
        metadata={"release": "demo"},
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ArtifactRegistry.load_json(registry_path).record_product_promotion_contract(
        name="demo-product-promotion-contract",
        path=contract_path,
        version="1.0",
        metadata={"artifact_manifest": str(manifest_path)},
    ).save_json()

    bundle = load_product_runtime_evidence_bundle(
        default_contract_paths=(contract_path,),
        registry_path=registry_path,
    )
    assert bundle is not None
    assert bundle.contract.model_id == "demo-model"
    assert bundle.manifest_path == manifest_path
    assert bundle.registry_record() is not None
    assert bundle.registry_record().key() == "product_promotion_contract:demo-product-promotion-contract:1.0"

    metadata_without_verification = bundle.runtime_metadata(budget_enabled=True)
    assert metadata_without_verification["promotion_contract_manifest"] == str(manifest_path)
    assert metadata_without_verification["promotion_contract_manifest_verification"] is None
    assert metadata_without_verification["promotion_contract_registry"] == str(registry_path)
    assert metadata_without_verification["promotion_contract_registry_key"] == (
        "product_promotion_contract:demo-product-promotion-contract:1.0"
    )
    assert metadata_without_verification["promotion_contract_registry_record"]["metadata"] == {
        "artifact_manifest": str(manifest_path)
    }
    assert metadata_without_verification["promotion_contract_product_trace_replay_workflow"] == {
        "report_path": "trace-replay-workflow.json",
        "selector_replay_report_path": "selector-replay.json",
        "product_runtime_drift_report_path": "runtime-drift.json",
    }
    assert metadata_without_verification["selfcheck_signal_fusion_workflow_report"] == str(
        selfcheck_report_path
    )
    assert metadata_without_verification["selfcheck_signal_fusion_workflow_manifest"] == str(
        selfcheck_manifest_path
    )
    assert (
        metadata_without_verification[
            "selfcheck_signal_fusion_workflow_manifest_verification"
        ]
        is None
    )
    assert (
        metadata_without_verification["selfcheck_signal_fusion_workflow_registry"]
        == str(selfcheck_registry_path)
    )
    assert (
        metadata_without_verification["selfcheck_signal_fusion_workflow_registry_key"]
        == "report:selfcheck-signal-fusion-workflow:0.1"
    )
    assert metadata_without_verification["selfcheck_signal_fusion_workflow_registry_record"] is None
    assert (
        metadata_without_verification[
            "selfcheck_signal_fusion_workflow_sample_quality_passed"
        ]
        is True
    )
    assert metadata_without_verification["selfcheck_signal_fusion_workflow_fusion_run_count"] == 1
    assert metadata_without_verification["world_model_signal_workflow_report"] == str(
        world_model_report_path
    )
    assert metadata_without_verification["world_model_signal_workflow_manifest"] == str(
        world_model_manifest_path
    )
    assert (
        metadata_without_verification["world_model_signal_workflow_manifest_verification"]
        is None
    )
    assert metadata_without_verification["world_model_signal_workflow_registry"] == str(
        world_model_registry_path
    )
    assert metadata_without_verification["world_model_signal_workflow_registry_key"] == (
        "report:world-model-signal-workflow:0.1"
    )
    assert metadata_without_verification["world_model_signal_workflow_registry_record"] is None
    assert metadata_without_verification["world_model_signal_workflow_release_gate_status"] == "promote"
    assert metadata_without_verification["world_model_signal_workflow_trace_gap_max"] == 0.0
    assert metadata_without_verification["world_model_signal_workflow_conflict_positive_count"] == 4
    assert metadata_without_verification["triple_extraction_fixture_matrix_report"] == str(
        triple_matrix_report_path
    )
    assert metadata_without_verification["triple_extraction_fixture_matrix_manifest"] == str(
        triple_matrix_manifest_path
    )
    assert (
        metadata_without_verification[
            "triple_extraction_fixture_matrix_manifest_verification"
        ]
        is None
    )
    assert metadata_without_verification["triple_extraction_fixture_matrix_registry"] == str(
        triple_matrix_registry_path
    )
    assert metadata_without_verification["triple_extraction_fixture_matrix_registry_key"] == (
        "report:triple-extraction-fixture-matrix:0.1"
    )
    assert metadata_without_verification["triple_extraction_fixture_matrix_registry_record"] is None
    assert metadata_without_verification["triple_extraction_fixture_matrix_status"] == "promote"
    assert metadata_without_verification["triple_extraction_fixture_matrix_n_corpora"] == 2
    assert metadata_without_verification[
        "triple_extraction_fixture_matrix_distinct_predicate_count"
    ] == 6

    metadata_with_verification = bundle.runtime_metadata(
        budget_enabled=True,
        verify_manifest=True,
    )
    assert metadata_with_verification["promotion_contract_manifest_verification"]["passed"] is True
    assert metadata_with_verification["promotion_contract_manifest_verification"]["checked"] == 1
    assert (
        metadata_with_verification[
            "selfcheck_signal_fusion_workflow_manifest_verification"
        ]
        is None
    )

    metadata_with_selfcheck_verification = bundle.runtime_metadata(
        budget_enabled=True,
        verify_selfcheck_signal_fusion_manifest=True,
        include_selfcheck_signal_fusion_record=True,
    )
    assert (
        metadata_with_selfcheck_verification[
            "selfcheck_signal_fusion_workflow_manifest_verification"
        ]["passed"]
        is True
    )
    assert (
        metadata_with_selfcheck_verification[
            "selfcheck_signal_fusion_workflow_manifest_verification"
        ]["checked"]
        == 1
    )
    assert (
        metadata_with_selfcheck_verification["selfcheck_signal_fusion_workflow_registry_key"]
        == "report:selfcheck-signal-fusion-workflow:0.1"
    )
    assert metadata_with_selfcheck_verification[
        "selfcheck_signal_fusion_workflow_registry_record"
    ]["metadata"] == {"artifact_manifest": str(selfcheck_manifest_path)}

    metadata_with_world_model_verification = bundle.runtime_metadata(
        budget_enabled=True,
        verify_world_model_signal_workflow_manifest=True,
        include_world_model_signal_workflow_record=True,
    )
    assert (
        metadata_with_world_model_verification[
            "world_model_signal_workflow_manifest_verification"
        ]["passed"]
        is True
    )
    assert (
        metadata_with_world_model_verification[
            "world_model_signal_workflow_manifest_verification"
        ]["checked"]
        == 1
    )
    assert (
        metadata_with_world_model_verification["world_model_signal_workflow_registry_key"]
        == "report:world-model-signal-workflow:0.1"
    )
    assert metadata_with_world_model_verification[
        "world_model_signal_workflow_registry_record"
    ]["metadata"] == {"artifact_manifest": str(world_model_manifest_path)}

    metadata_with_triple_matrix_verification = bundle.runtime_metadata(
        budget_enabled=True,
        verify_triple_extraction_fixture_matrix_manifest=True,
        include_triple_extraction_fixture_matrix_record=True,
    )
    assert (
        metadata_with_triple_matrix_verification[
            "triple_extraction_fixture_matrix_manifest_verification"
        ]["passed"]
        is True
    )
    assert (
        metadata_with_triple_matrix_verification[
            "triple_extraction_fixture_matrix_manifest_verification"
        ]["checked"]
        == 1
    )
    assert (
        metadata_with_triple_matrix_verification[
            "triple_extraction_fixture_matrix_registry_key"
        ]
        == "report:triple-extraction-fixture-matrix:0.1"
    )
    assert metadata_with_triple_matrix_verification[
        "triple_extraction_fixture_matrix_registry_record"
    ]["metadata"] == {"artifact_manifest": str(triple_matrix_manifest_path)}


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
    ).record_benchmark_manifest(
        name="qwen-mini-matrix",
        path="artifacts/qwen-mini/artifact-manifest.json",
        version="0.3",
        metadata={"verified": True},
    ).record_manifest_verification(
        name="qwen-mini-matrix-verification",
        path="artifacts/qwen-mini/manifest-verification.json",
        version="0.3",
    ).save_json()
    loaded = ArtifactRegistry.load_json(registry_path)

    assert loaded.list_records(artifact_type="product_trace")[0].metadata["total_actions"] == 1
    assert loaded.list_records(artifact_type="report")[0].name == "tiny-report"
    assert loaded.list_records(artifact_type="action_result")[0].name == "req-1-actions"
    assert loaded.list_records(artifact_type="benchmark_manifest")[0].metadata["verified"] is True
    assert loaded.list_records(artifact_type="manifest_verification")[0].name == "qwen-mini-matrix-verification"
