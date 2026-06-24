"""Product trace and artifact registry tests."""

import json

from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control import (
    ActionExecutionStatus,
    ActionRequest,
    ActionResult,
    ControlAction,
    ProductPromotionContract,
    ProductRuntimeBudgetPolicy,
    ProductTrace,
    RiskController,
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
)
from eigentruth.registry import (
    ArtifactRegistry,
    RegistryRecord,
    build_artifact_manifest,
    fingerprint_path,
    load_and_verify_artifact_manifest,
)
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
            "runtime_budget": {"passed": True},
            "large_unselected_metadata": tuple(range(100)),
        },
        runtime_trace=RuntimeTrace(
            phases=(RuntimePhaseTiming("phase", 0.01),),
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
    assert payload["verification_results"][0]["evidence_count"] == 5
    assert len(payload["verification_results"][0]["evidence"]) == 2
    assert len(payload["verification_results"][0]["explanation"]) <= 40
    assert payload["action_results"][0]["output_summary"]["key_count"] == 1
    assert "large_unselected_metadata" not in payload["metadata"]
    assert payload["metadata"]["artifact_source"] == "artifact.json"
    assert payload["metadata"]["promotion_contract_source"] == "contract.json"
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
    assert summary["by_route"]["retrieval_groundedness"]["mean_attempted_route_count"] == 2.0
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
            max_retrieval_use_rate=0.25,
            max_retrieval_hit_count=1,
        ),
    )

    assert report["passed"] is False
    assert report["metrics"]["has_runtime_trace"] is False
    assert round(report["metrics"]["mean_route_duration_seconds"], 6) == 0.03
    assert report["metrics"]["max_route_duration_seconds"] == 0.05
    assert report["metrics"]["mean_attempted_route_count"] == 1.5
    assert report["metrics"]["retrieval_use_rate"] == 0.5
    assert report["metrics"]["retrieval_hit_count"] == 2.0
    assert [failure["metric"] for failure in report["failures"]] == [
        "mean_route_duration_seconds",
        "max_route_duration_seconds",
        "mean_attempted_route_count",
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
            "max_retrieval_use_rate": 0.5,
            "max_retrieval_hit_count": 4,
            "min_claims_cache_hit_rate": 0.8,
            "min_verifier_trace_cache_hit_rate": 0.9,
            "required_route_min_selected": 200,
            "required_route_max_runtime_total_seconds": 8.0,
            "required_route_max_retrieval_hit_count": 450.0,
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
            "runtime": {"layer": -12, "batch_size": 2},
            "performance_baseline_record": "performance_baseline:runtime:0.9",
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
                "current": {"path": "artifacts/runtime-current/product-runtime-baseline.json"},
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
                "selector_replay_manifest": "artifacts/selector/artifact-manifest.json",
                "product_runtime_drift_manifest": "artifacts/runtime-drift/artifact-manifest.json",
                "adapter_family_matrix_report": "artifacts/adapter-family-matrix.json",
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
    assert contract.runtime["layer"] == -12
    assert contract.verifier_route["route"] == "structured_state"
    assert contract.metadata["runtime_profile"] == "balanced"
    assert contract.metadata["recommended_performance_baseline_record"] == "performance_baseline:runtime:0.9"
    assert contract.metadata["performance_baseline_record"] == "performance_baseline:runtime:0.9"
    assert contract.metadata["performance_manifest"] == "artifacts/performance/artifact-manifest.json"
    assert contract.metadata["recommended_selector_replay_candidate"] == "default"
    assert contract.metadata["recommended_product_runtime_drift_report"] == (
        "artifacts/runtime-drift/product-runtime-drift.json"
    )
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
    assert contract.runtime_budget_policy == direct_policy
    assert contract.runtime_budget_policy.max_total_seconds == 1.0
    assert contract.runtime_budget_policy.max_mean_route_duration_seconds == 0.05
    assert contract.runtime_budget_policy.max_p99_route_duration_seconds == 0.20
    assert contract.runtime_budget_policy.max_route_duration_seconds == 0.25
    assert contract.runtime_budget_policy.max_mean_attempted_route_count == 1.5
    assert contract.runtime_budget_policy.max_retrieval_use_rate == 0.5
    assert contract.runtime_budget_policy.max_retrieval_hit_count == 4.0
    assert contract.runtime_budget_policy.min_named_cache_hit_rate == {
        "claims": 0.8,
        "verifier_trace": 0.9,
    }
    assert roundtrip == contract
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
    assert metadata["promotion_contract_metadata"] == {"selector_replay_status": "promote"}
    assert product_promotion_contract_metadata(None, source=None, budget_enabled=True) == {
        "promotion_contract_source": None,
        "promotion_contract_budget_enabled": False,
    }


def test_product_runtime_evidence_bundle_loads_manifest_and_registry_lazily(tmp_path):
    contract_path = tmp_path / "product-promotion-contract.json"
    manifest_path = tmp_path / "artifact-manifest.json"
    registry_path = tmp_path / "registry.json"
    ProductPromotionContract(
        model_id="demo-model",
        runtime={"layer": -2},
        verifier_route={"route": "structured_qa"},
        runtime_budget_policy=ProductRuntimeBudgetPolicy(max_retrieval_use_rate=0.0),
        source_workflow="release_candidate_comparison",
        source_status="promote",
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

    metadata_with_verification = bundle.runtime_metadata(
        budget_enabled=True,
        verify_manifest=True,
    )
    assert metadata_with_verification["promotion_contract_manifest_verification"]["passed"] is True
    assert metadata_with_verification["promotion_contract_manifest_verification"]["checked"] == 1


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
