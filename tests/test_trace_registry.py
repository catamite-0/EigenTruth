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
    product_promotion_contract_summary,
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
    VerificationBudgetPolicy,
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

    budgeted_plan = ClaimVerificationPlanner().plan(
        claims,
        budget_policy=VerificationBudgetPolicy(max_verify_claims=1, max_route_attempts=1),
    )
    budgeted_trace = ProductTrace(
        request_id="req-plan-budgeted",
        claims=claims,
        verification_plan=budgeted_plan,
    )
    budgeted_bounded = budgeted_trace.to_bounded_dict(max_nested_items=2)
    budgeted_metrics = product_runtime_metrics(budgeted_trace)
    budgeted_bounded_metrics = product_runtime_metrics(budgeted_bounded)

    assert budgeted_plan.verification_scope == "budgeted"
    assert budgeted_bounded["summaries"]["verification_plan"]["budget"]["enabled"] is True
    assert budgeted_metrics["verification_plan_budget_enabled"] is True
    assert budgeted_metrics["verification_plan_budget_selected_claim_count"] == 1.0
    assert budgeted_metrics["verification_plan_budget_dropped_claim_count"] == 1.0
    assert budgeted_metrics["verification_plan_budget_route_budget_exhausted"] is True
    assert budgeted_bounded_metrics["verification_plan_budget_enabled"] is True
    assert budgeted_bounded_metrics["verification_plan_budget_selected_claim_count"] == 1.0

    hidden_plan = ClaimVerificationPlanner().plan(
        claims,
        hidden_evidence={
            "selected": (
                {
                    "record_id": "c2",
                    "record_index": 1,
                    "score_name": "subspace_resid",
                    "score": 5.0,
                    "direction": "higher",
                    "anomaly_score": 1.0,
                    "layer": "-8",
                    "evidence_ref": "sweep:layer:-8:subspace_resid:c2",
                },
            ),
        },
    )
    hidden_trace = ProductTrace(
        request_id="req-plan-hidden-evidence",
        claims=claims,
        verification_plan=hidden_plan,
    )
    hidden_bounded = hidden_trace.to_bounded_dict()

    assert hidden_bounded["summaries"]["verification_plan"]["hidden_evidence"]["available"] is True
    assert hidden_bounded["summaries"]["verification_plan"]["hidden_evidence"]["selected_count"] == 1
    assert hidden_bounded["summaries"]["verification_plan"]["hidden_evidence"]["claim_ids"] == ["c2"]
    assert hidden_bounded["summaries"]["verification_plan"]["hidden_evidence"]["score_counts"] == {
        "subspace_resid": 1,
    }


def test_product_trace_summarizes_triple_and_slot_coverage():
    triple = {"subject": "France", "predicate": "capital_of", "object": "Paris"}
    trace = ProductTrace(
        claims=(
            {
                "claim_id": "c1",
                "text": "Paris is the capital of France.",
                "metadata": {"claim_triples": (triple,)},
            },
            {
                "claim_id": "c2",
                "text": "Berlin is a large city.",
                "metadata": {},
            },
        ),
        verification_results=(
            VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=0.85,
                evidence=("atlas",),
                metadata={
                    "selected_route": "triple_evidence",
                    "audit_report": {
                        "claim_id": "c1",
                        "triple_count": 1,
                        "passed_count": 1,
                        "failed_count": 0,
                        "covered_slot_count": 3,
                        "missing_slot_count": 0,
                        "passed": True,
                        "audits": (
                            {
                                "triple": triple,
                                "passed": True,
                                "covered_slots": ("subject", "predicate", "object"),
                                "missing_slots": (),
                                "slot_coverage": {
                                    "subject": 1.0,
                                    "predicate": 1.0,
                                    "object": 1.0,
                                },
                            },
                        ),
                    },
                    "all_triple_results": (
                        {
                            "status": "supported",
                            "metadata": {"triple": triple},
                        },
                    ),
                },
            ),
        ),
    )

    summary = trace.triple_coverage_summary()
    bounded = trace.to_bounded_dict()
    metrics = product_runtime_metrics(trace)
    bounded_metrics = product_runtime_metrics(bounded)

    assert summary["claim_count"] == 2
    assert summary["claims_with_triples"] == 1
    assert summary["claim_triple_count"] == 1
    assert summary["claim_triple_coverage_rate"] == pytest.approx(0.5)
    assert summary["audit_available"] is True
    assert summary["audit_claim_covered_count"] == 1
    assert summary["audit_claim_coverage_rate"] == pytest.approx(1.0)
    assert summary["audit_pass_rate"] == pytest.approx(1.0)
    assert summary["slot_coverage_rate"] == pytest.approx(1.0)
    assert summary["claim_predicate_counts"] == {"capital_of": 1}
    assert summary["audit_predicate_counts"] == {"capital_of": 1}
    assert summary["structured_fact_status_counts"] == {"supported": 1}
    assert bounded["summaries"]["triple_coverage"]["audit_report_count"] == 1
    assert metrics["triple_coverage_source"] == "full_trace"
    assert metrics["triple_audit_pass_rate"] == pytest.approx(1.0)
    assert bounded_metrics["triple_coverage_source"] == "bounded_summary"
    assert bounded_metrics["triple_slot_coverage_rate"] == pytest.approx(1.0)
    json.dumps(bounded)


def test_product_runtime_metrics_exposes_triple_audit_evidence_provenance():
    trace = ProductTrace(
        request_id="req-provenance",
        metadata={
            "promotion_contract_metadata": {
                "triple_audit_evidence_source": "claim_correction_workflow",
                "triple_audit_evidence_report": "claim-correction-workflow.json",
                "triple_audit_evidence_workflow": (
                    "source_family_structured_qa_claim_correction_workflow"
                ),
                "triple_audit_evidence_status": "promote",
            }
        },
    )

    metrics = product_runtime_metrics(trace)

    assert metrics["promotion_contract_available"] is True
    assert metrics["promotion_contract_triple_audit_evidence_available"] is True
    assert metrics["promotion_contract_triple_audit_evidence_source"] == (
        "claim_correction_workflow"
    )
    assert metrics["promotion_contract_triple_audit_evidence_report"] == (
        "claim-correction-workflow.json"
    )
    assert metrics["promotion_contract_triple_audit_evidence_workflow"] == (
        "source_family_structured_qa_claim_correction_workflow"
    )
    assert metrics["promotion_contract_triple_audit_evidence_status"] == "promote"
    assert metrics["promotion_contract_summary"]["triple_audit_evidence"] == {
        "available": True,
        "source": "claim_correction_workflow",
        "report": "claim-correction-workflow.json",
        "workflow": "source_family_structured_qa_claim_correction_workflow",
        "status": "promote",
    }


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
            "promotion_contract_promotion_summary": {
                "status": "promote",
                "blocking_gate_count": 0,
                "source_status": "promote",
                "runtime": {"layer": -2, "recommended_runtime_seconds": 0.2},
                "verifier_route": {"route": "structured_fact"},
                "action_gates": {"action_audit_status": "promote"},
            },
            "promotion_contract_verifier_route": {
                "route": "structured_fact",
                "covered_fact_property_count": 3,
                "covered_fact_properties": ["P36", "P37", "P38"],
            },
            "promotion_contract_recommended_route_covered_fact_properties": [
                "P36",
                "P37",
                "P38",
            ],
            "promotion_contract_required_route_baseline_covered_fact_property_counts": {
                "benchmark_manifest:structured-fact:0.1": 3
            },
            "promotion_contract_structured_fact_robustness_property_counts": {
                "benchmark_manifest:structured-fact:0.1": 3
            },
            "promotion_contract_selfcheck_signal_fusion_workflow": {
                "report_path": "selfcheck.json"
            },
            "promotion_contract_world_model_signal_workflow": {
                "release_gate_status": "promote"
            },
            "promotion_contract_pathway_intervention_workflow": {
                "release_ready": True
            },
            "promotion_contract_external_evidence_baseline_comparison": {
                "status": "promote",
                "recommended_route": "structured_fact",
                "route_passed": True,
            },
            "promotion_contract_frontier_release_evidence": {
                "status": "promote",
                "report_path": "frontier-evidence.json",
                "manifest_path": "frontier-manifest.json",
                "decision_status": "promote",
                "verifier_track_status": "promote",
                "abstention_track_status": "promote",
                "run_names": ["verifier", "abstention"],
            },
            "promotion_contract_frontier_release_evidence_decision_status": "promote",
            "promotion_contract_frontier_release_evidence_verifier_track_status": "promote",
            "promotion_contract_frontier_release_evidence_abstention_track_status": (
                "promote"
            ),
            "promotion_contract_frontier_release_evidence_run_names": [
                "verifier",
                "abstention",
            ],
            "promotion_contract_frontier_release_evidence_run_count": 2,
            "external_evidence_baseline_comparison_source": "registry",
            "external_evidence_baseline_comparison_status": "promote",
            "external_evidence_baseline_comparison_decision_status": "promote",
            "external_evidence_baseline_comparison_recommended_route": "structured_fact",
            "external_evidence_baseline_comparison_route_passed": True,
            "external_evidence_baseline_comparison_text_redline_passed": True,
            "promotion_contract_triple_extraction_fixture_matrix": {
                "status": "promote",
                "distinct_predicate_count": 6,
            },
            "triple_extraction_fixture_matrix_source": "registry",
            "triple_extraction_fixture_matrix_manifest_verification": {"passed": True},
            "triple_extraction_fixture_matrix_n_corpora": 2,
            "triple_extraction_fixture_matrix_promoted_corpora": 2,
            "triple_extraction_fixture_matrix_mean_best_f1": 1.0,
            "triple_extraction_fixture_matrix_mean_f1_lift": 0.5,
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
    assert payload["metadata"]["promotion_contract_promotion_summary"]["status"] == (
        "promote"
    )
    assert payload["metadata"]["promotion_contract_promotion_summary"][
        "blocking_gate_count"
    ] == 0
    assert payload["metadata"]["promotion_contract_verifier_route"][
        "covered_fact_property_count"
    ] == 3
    assert payload["metadata"][
        "promotion_contract_recommended_route_covered_fact_properties"
    ] == ["P36", "P37", {"_truncated": True, "_omitted_items": 1}]
    assert payload["metadata"][
        "promotion_contract_required_route_baseline_covered_fact_property_counts"
    ] == {"benchmark_manifest:structured-fact:0.1": 3}
    assert payload["metadata"]["promotion_contract_selfcheck_signal_fusion_workflow"] == {
        "report_path": "selfcheck.json"
    }
    assert payload["metadata"]["promotion_contract_world_model_signal_workflow"] == {
        "release_gate_status": "promote"
    }
    assert payload["metadata"]["promotion_contract_pathway_intervention_workflow"] == {
        "release_ready": True
    }
    assert payload["metadata"][
        "promotion_contract_external_evidence_baseline_comparison"
    ] == {
        "status": "promote",
        "recommended_route": "structured_fact",
        "_truncated": True,
        "_omitted_keys": 1,
    }
    assert payload["metadata"]["promotion_contract_frontier_release_evidence"] == {
        "status": "promote",
        "report_path": "frontier-evidence.json",
        "_truncated": True,
        "_omitted_keys": 5,
    }
    assert payload["metadata"]["promotion_contract_frontier_release_evidence_run_count"] == 2
    assert payload["metadata"]["promotion_contract_triple_extraction_fixture_matrix"] == {
        "status": "promote",
        "distinct_predicate_count": 6,
    }
    assert payload["metadata"]["triple_extraction_fixture_matrix_manifest_verification"] == {
        "passed": True
    }
    metrics = product_runtime_metrics(payload)
    assert metrics["final_answer_available"] is True
    assert metrics["final_answer_source"] == "bounded_summary"
    assert metrics["final_answer_status"] == "answered"
    assert metrics["final_answer_answerable"] is True
    assert metrics["final_answer_evidence_count"] == 4.0
    assert metrics["promotion_contract_available"] is True
    assert metrics["promotion_contract_source"] == "contract.json"
    assert metrics["promotion_contract_promotion_summary_status"] == "promote"
    assert metrics["promotion_contract_promotion_summary_blocking_gate_count"] == (
        pytest.approx(0.0)
    )
    assert metrics["promotion_contract_recommended_route_covered_fact_property_count"] == 3.0
    assert metrics["promotion_contract_recommended_route_covered_fact_properties"] == [
        "P36",
        "P37",
    ]
    assert metrics["promotion_contract_required_route_baseline_covered_fact_property_counts"] == {
        "benchmark_manifest:structured-fact:0.1": 3
    }
    assert (
        metrics["promotion_contract_external_evidence_baseline_comparison_available"]
        is True
    )
    assert metrics["promotion_contract_frontier_release_evidence_available"] is True
    assert metrics["promotion_contract_frontier_release_evidence_status"] == "promote"
    assert metrics["promotion_contract_frontier_release_evidence_report"] == (
        "frontier-evidence.json"
    )
    assert metrics["promotion_contract_frontier_release_evidence_decision_status"] == (
        "promote"
    )
    assert metrics[
        "promotion_contract_frontier_release_evidence_run_count"
    ] == pytest.approx(2.0)
    assert metrics[
        "promotion_contract_external_evidence_baseline_comparison_source"
    ] == "registry"
    assert metrics[
        "promotion_contract_external_evidence_baseline_comparison_status"
    ] == "promote"
    assert metrics[
        "promotion_contract_external_evidence_baseline_comparison_recommended_route"
    ] == "structured_fact"
    assert (
        metrics[
            "promotion_contract_external_evidence_baseline_comparison_route_passed"
        ]
        is True
    )
    assert (
        metrics[
            "promotion_contract_external_evidence_baseline_comparison_text_redline_passed"
        ]
        is True
    )
    assert metrics["promotion_contract_triple_extraction_fixture_matrix_available"] is True
    assert metrics["promotion_contract_triple_extraction_fixture_matrix_source"] == "registry"
    assert metrics["promotion_contract_triple_extraction_fixture_matrix_status"] == "promote"
    assert metrics["promotion_contract_triple_extraction_fixture_matrix_manifest_verified"] is True
    assert (
        metrics["promotion_contract_triple_extraction_fixture_matrix_distinct_predicate_count"]
        == 6.0
    )
    assert metrics["promotion_contract_triple_extraction_fixture_matrix_n_corpora"] == 2.0
    assert metrics["promotion_contract_triple_extraction_fixture_matrix_mean_best_f1"] == 1.0
    assert metrics["promotion_contract_triple_extraction_fixture_matrix_mean_f1_lift"] == 0.5
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
        actions=(
            ActionRequest(action=ControlAction.RETRIEVE, reason="fetch", request_id="r1"),
            ActionRequest(action=ControlAction.ABSTAIN, reason="stop", request_id="r2"),
            ActionRequest(action=ControlAction.RETRIEVE, reason="fetch", request_id="r3"),
        ),
        action_results=(
            ActionResult(
                action=ControlAction.RETRIEVE,
                status=ActionExecutionStatus.SUCCEEDED,
                request_id="r1",
            ),
            ActionResult(
                action=ControlAction.ABSTAIN,
                status=ActionExecutionStatus.DRY_RUN,
                request_id="r2",
            ),
            ActionResult(
                action=ControlAction.RETRIEVE,
                status=ActionExecutionStatus.SUCCEEDED,
                request_id="r3",
            ),
        )
    )

    summary = trace.action_execution_summary()

    assert summary["total"] == 3
    assert summary["counts_by_status"] == {"succeeded": 2, "dry_run": 1}
    assert summary["counts_by_action"] == {"retrieve": 2, "abstain": 1}
    assert summary["planned_action_count"] == 3
    assert summary["result_count"] == 3
    assert summary["planned_counts_by_action"] == {"retrieve": 2, "abstain": 1}
    assert summary["alignment_passed"] is True
    assert summary["alignment"]["passed"] is True
    assert summary["missing_result_count"] == 0
    assert summary["unexpected_result_count"] == 0
    assert summary["request_id_mismatch_count"] == 0
    assert summary["side_effects"] is False


def test_product_trace_action_execution_summary_flags_result_alignment_gaps():
    trace = ProductTrace(
        actions=(
            ActionRequest(action=ControlAction.RETRIEVE, reason="fetch", request_id="r1"),
            ActionRequest(action=ControlAction.ABSTAIN, reason="stop", request_id="r2"),
        ),
        action_results=(
            ActionResult(
                action=ControlAction.RETRIEVE,
                status=ActionExecutionStatus.SUCCEEDED,
                request_id="r1",
            ),
            ActionResult(
                action=ControlAction.REWRITE,
                status=ActionExecutionStatus.DRY_RUN,
                request_id="r-extra",
            ),
        ),
    )

    summary = trace.action_execution_summary()
    bounded = trace.to_bounded_dict()

    assert summary["alignment_passed"] is False
    assert summary["missing_result_count"] == 1
    assert summary["unexpected_result_count"] == 1
    assert summary["request_id_mismatch_count"] == 2
    assert summary["alignment"]["missing_results_by_action"] == {"abstain": 1}
    assert summary["alignment"]["unexpected_results_by_action"] == {"rewrite": 1}
    assert summary["alignment"]["missing_request_ids"] == ("r2",)
    assert summary["alignment"]["unexpected_request_ids"] == ("r-extra",)
    assert bounded["summaries"]["action_execution"]["alignment_passed"] is False
    assert bounded["summaries"]["action_execution"]["missing_result_count"] == 1


def test_product_trace_action_execution_summary_handles_many_request_ids():
    actions = tuple(
        ActionRequest(
            action=ControlAction.RETRIEVE,
            reason="fetch",
            request_id=f"request-{index}",
        )
        for index in range(32)
    )
    action_results = tuple(
        ActionResult(
            action=ControlAction.RETRIEVE,
            status=ActionExecutionStatus.SUCCEEDED,
            request_id=f"request-{index}",
        )
        for index in range(24)
    ) + (
        ActionResult(
            action=ControlAction.RETRIEVE,
            status=ActionExecutionStatus.SUCCEEDED,
            request_id="request-extra",
        ),
    )
    trace = ProductTrace(actions=actions, action_results=action_results)

    summary = trace.action_execution_summary()

    assert summary["alignment_passed"] is False
    assert summary["missing_result_count"] == 7
    assert summary["unexpected_result_count"] == 0
    assert summary["request_id_mismatch_count"] == 9
    assert summary["alignment"]["missing_request_ids"] == tuple(
        f"request-{index}" for index in range(24, 32)
    )
    assert summary["alignment"]["unexpected_request_ids"] == ("request-extra",)


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


def test_product_trace_world_model_summary_counts_conflicts_and_trace_gaps():
    trace = ProductTrace(
        verification_results=(
            VerificationResult(
                status=VerificationStatus.REFUTED,
                confidence=0.82,
                metadata={
                    "world_model": "RuleBasedWorldModelAdapter",
                    "world_model_reference": {
                        "reference_id": "orders",
                        "adapter": "RuleBasedWorldModelAdapter",
                    },
                    "world_model_view": {
                        "base_state_fingerprint": "base",
                        "predicted_state_fingerprint": "predicted",
                        "postcondition": {"path": "inventory.sku_123.available"},
                    },
                    "world_model_conflict": {
                        "path": "inventory.sku_123.available",
                        "expected": 10,
                        "actual": 7,
                    },
                    "prediction_confidence": 0.9,
                    "prediction_metadata": {
                        "decision_rule": "rule_transition_applied",
                        "agreement_rate": 1.0,
                    },
                    "decision_rule": "transition_postcondition_failed",
                },
            ),
            VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                metadata={
                    "verifier": "world_model_ensemble",
                    "world_model_reference": {
                        "reference_id": "orders",
                        "adapter": "EnsembleWorldModelAdapter",
                    },
                    "prediction_metadata": {
                        "below_min_agreement": True,
                        "agreement_rate": 0.5,
                        "decision_rule": "prediction_agreement_below_threshold",
                    },
                    "decision_rule": "prediction_agreement_below_threshold",
                },
            ),
            VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=0.7,
                metadata={
                    "verifier": "world_model_ensemble",
                    "prediction_metadata": {
                        "world_model_reference": {
                            "reference_id": "payments",
                            "adapter": "EnsembleWorldModelAdapter",
                        },
                        "world_model_view": {
                            "base_state_fingerprint": "base-nested",
                            "predicted_state_fingerprint": "predicted-nested",
                            "postcondition": {"path": "payments.invoice_1.status"},
                        },
                        "agreement_rate": 0.8,
                        "decision_rule": "prediction_consensus",
                    },
                },
            ),
            VerificationResult(status=VerificationStatus.SUPPORTED, confidence=0.8),
        )
    )

    summary = trace.world_model_summary()
    bounded = trace.to_bounded_dict()
    metrics = product_runtime_metrics(trace)
    bounded_metrics = product_runtime_metrics(bounded)

    assert summary["total"] == 4
    assert summary["world_model_total"] == 3
    assert summary["coverage_rate"] == pytest.approx(3 / 4)
    assert summary["conflict_count"] == 1
    assert summary["low_agreement_count"] == 1
    assert summary["trace_gap_count"] == 1
    assert summary["traceable"] is False
    assert summary["counts_by_status"] == {
        "refuted": 1,
        "insufficient_evidence": 1,
        "supported": 1,
    }
    assert summary["counts_by_adapter"] == {
        "RuleBasedWorldModelAdapter": 1,
        "EnsembleWorldModelAdapter": 2,
    }
    assert summary["counts_by_reference_id"] == {"orders": 2, "payments": 1}
    assert summary["conflict_paths"] == {"inventory.sku_123.available": 1}
    assert summary["prediction_confidence_mean"] == pytest.approx(0.9)
    assert summary["agreement_rate_min"] == pytest.approx(0.5)
    assert bounded["summaries"]["world_model"]["world_model_total"] == 3
    assert bounded["summaries"]["world_model"]["trace_gap_count"] == 1
    assert metrics["world_model_source"] == "full_trace"
    assert metrics["world_model_conflict_count"] == pytest.approx(1.0)
    assert metrics["world_model_trace_gap_rate"] == pytest.approx(1 / 3)
    assert metrics["world_model_counts_by_adapter"] == {
        "RuleBasedWorldModelAdapter": 1,
        "EnsembleWorldModelAdapter": 2,
    }
    assert bounded_metrics["world_model_source"] == "bounded_summary"
    assert bounded_metrics["world_model_low_agreement_count"] == pytest.approx(1.0)
    assert bounded_metrics["world_model_traceable"] is False
    json.dumps(summary)
    json.dumps(bounded)


def test_product_trace_world_model_summary_ignores_generic_prediction_metadata():
    trace = ProductTrace(
        verification_results=(
            VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=0.91,
                metadata={
                    "verifier": "calibrated_classifier",
                    "prediction_metadata": {
                        "decision_rule": "calibrated_softmax",
                        "below_min_agreement": True,
                        "agreement_rate": 0.2,
                    },
                },
            ),
        )
    )

    summary = trace.world_model_summary()

    assert summary["total"] == 1
    assert summary["world_model_total"] == 0
    assert summary["trace_gap_count"] == 0
    assert summary["low_agreement_count"] == 0
    assert summary["counts_by_status"] == {}
    assert summary["traceable"] is False


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
            "require_structured_fact_robustness": True,
            "structured_fact_canonical_route_key": (
                "benchmark_manifest:retrieval-structured-qa:0.5"
            ),
            "require_product_runtime_drift_covered_fact_property_evidence": True,
            "require_performance_score_dump_cache": True,
            "min_performance_score_dump_cache_jsonl_view_hit_rate": 0.5,
            "performance_drift_baseline_key": "performance_baseline:runtime-reference:0.8",
            "max_covariance_maha_last_auroc_drop": 0.05,
            "max_uncached_forward_seconds": None,
            "max_recommended_runtime_seconds": 1.0,
        },
        "decision": {
            "status": "promote",
            "readiness_status": "promote",
            "route_status": "promote",
            "performance_status": "promote",
            "adapter_family_status": "promote",
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
            "context_sensitivity_workflow_status": "promote",
            "recommended_context_sensitivity_workflow_report": (
                "artifacts/context-sensitivity/context-sensitivity-workflow.json"
            ),
            "pathway_intervention_workflow_status": "promote",
            "recommended_pathway_intervention_workflow_report": (
                "artifacts/pathway-intervention/pathway-intervention-workflow.json"
            ),
            "external_evidence_baseline_comparison_status": "promote",
            "recommended_external_evidence_baseline_comparison_report": (
                "artifacts/external-evidence/external-evidence-baseline-comparison.json"
            ),
            "pre_generation_probe_comparison_status": "promote",
            "recommended_pre_generation_probe_comparison_report": (
                "artifacts/pre-generation-probe-comparison/comparison.json"
            ),
            "claim_factuality_probe_comparison_status": "promote",
            "recommended_claim_factuality_probe_comparison_report": (
                "artifacts/claim-factuality-probe-comparison/comparison.json"
            ),
            "counterfactual_verification_status": "promote",
            "recommended_counterfactual_verification_report": (
                "artifacts/counterfactual/counterfactual-verification.json"
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
            "runtime_cost": {
                "recommended_runtime_seconds": 0.20,
                "recommended_runtime_cost_source": "cache_only_total_seconds",
                "uncached_forward_cost_seconds": 37.5,
                "uncached_forward_cost_source": "uncached_forced_answer_forward_seconds",
                "cache_only_total_seconds": 0.20,
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
                "require_action_audit_gate": True,
                "action_audit_gate": {
                    "status": "promote",
                    "gate_enabled": True,
                    "passed": True,
                    "error_rate": 0.0,
                },
                "require_action_execution_gate": True,
                "action_execution_gate": {
                    "status": "promote",
                    "gate_enabled": True,
                    "passed": True,
                    "missing_result_rate": 0.0,
                    "request_id_mismatch_rate": 0.0,
                },
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
            "context_sensitivity_workflow": {
                "report_path": (
                    "artifacts/context-sensitivity/context-sensitivity-workflow.json"
                ),
                "manifest_path": "artifacts/context-sensitivity/artifact-manifest.json",
                "source": "registry",
                "registry": "artifacts/release-registry.json",
                "record_key": "report:context-sensitivity-workflow:0.1",
                "workflow": "context_sensitivity_workflow",
                "status": "promote",
                "paired_logprob_record_count": 6,
                "enriched_record_count": 6,
                "enhanced_score_signal_count": 4,
                "max_flagged_rate": 0.25,
                "mean_flagged_rate": 0.125,
                "max_context_sensitivity_ratio": 1.35,
                "manifest_verified": True,
                "blocking_reasons": [],
            },
            "pathway_intervention_workflow": {
                "report_path": (
                    "artifacts/pathway-intervention/pathway-intervention-workflow.json"
                ),
                "manifest_path": "artifacts/pathway-intervention/artifact-manifest.json",
                "source": "registry",
                "registry": "artifacts/release-registry.json",
                "record_key": "report:pathway-intervention-workflow:0.1",
                "workflow": "pathway_intervention_workflow",
                "status": "promote",
                "report_status": "complete",
                "release_ready": True,
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "layer": -8,
                "intervention_layer": -8,
                "patch_layer": -8,
                "signals": ["pathway_disagreement", "truth_proj", "nll_answer"],
                "activation_ablation_gate_status": "promote",
                "source_patch_gate_status": "promote",
                "best_signals": {
                    "activation_ablation": "pathway_disagreement",
                    "source_patch": "truth_proj",
                },
                "blocking_reasons": [],
            },
            "external_evidence_baseline_comparison": {
                "report_path": (
                    "artifacts/external-evidence/"
                    "external-evidence-baseline-comparison.json"
                ),
                "source": "registry",
                "registry": "artifacts/release-registry.json",
                "record_key": "report:covered-facts-external-evidence-handoff:0.4",
                "workflow": "external_evidence_baseline_comparison",
                "decision_status": "promote",
                "recommended_route": "structured_fact",
                "recommended_route_record": (
                    "benchmark_manifest:structured-fact-canonical-route:0.1"
                ),
                "route_passed": True,
                "text_redline_passed": True,
                "text_redline_run_count": 2,
            },
            "pre_generation_probe_comparison": {
                "report_path": "artifacts/pre-generation-probe-comparison/comparison.json",
                "manifest_path": (
                    "artifacts/pre-generation-probe-comparison/artifact-manifest.json"
                ),
                "source": "registry",
                "registry": "artifacts/release-registry.json",
                "record_key": "report:pre-generation-probe-comparison:0.1",
                "workflow": "pre_generation_probe_workflow_comparison",
                "status": "promote",
                "model_count": 2,
                "run_count": 2,
                "redline_passed": True,
                "redline_run_count": 2,
                "best_run": {
                    "name": "qwen05",
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "recommended_layer": -12,
                    "test_label_auroc": 0.74,
                    "redline_best_signal": "answer_token_count",
                    "redline_best_auroc": 0.61,
                    "redline_margin": 0.13,
                },
            },
            "claim_factuality_probe_comparison": {
                "report_path": "artifacts/claim-factuality-probe-comparison/comparison.json",
                "manifest_path": (
                    "artifacts/claim-factuality-probe-comparison/artifact-manifest.json"
                ),
                "source": "registry",
                "registry": "artifacts/release-registry.json",
                "record_key": "report:claim-factuality-probe-comparison:0.1",
                "workflow": "claim_factuality_probe_workflow_comparison",
                "status": "promote",
                "report_status": "ready",
                "model_count": 2,
                "run_count": 2,
                "redline_passed": True,
                "redline_run_count": 2,
                "best_run": {
                    "name": "qwen05",
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "record_count": 96,
                    "recommended_layer": -4,
                    "test_label_auroc": 0.84,
                    "test_selective_accuracy": 0.91,
                    "test_selective_coverage": 0.78,
                    "conformal_threshold": 0.62,
                    "redline_best_signal": "answer_negation_flag",
                    "redline_best_auroc": 0.66,
                    "redline_margin": 0.18,
                },
            },
            "counterfactual_verification": {
                "report_path": "artifacts/counterfactual/counterfactual-verification.json",
                "manifest_path": "artifacts/counterfactual/artifact-manifest.json",
                "source": "registry",
                "registry": "artifacts/release-registry.json",
                "record_key": "report:counterfactual-verifier-audit:0.1",
                "workflow": "counterfactual_verification_eval",
                "status": "promote",
                "record_count": 12,
                "pass_rate": 1.0,
                "false_invariance_rate": 0.0,
                "flip_success_count": 12,
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
                    "promotion_evidence_metric_count": 4,
                    "promotion_evidence_blocked_metric_count": 0,
                    "triple_audit_evidence_metric_count": 4,
                    "triple_audit_evidence_blocked_metric_count": 0,
                    "covered_fact_property_evidence_metric_count": 6,
                    "covered_fact_property_evidence_blocked_metric_count": 0,
                    "world_model_evidence_required": True,
                    "world_model_evidence_metric_count": 5,
                    "world_model_evidence_blocked_metric_count": 0,
                    "promotion_contract_coverage_rate_baseline": 1.0,
                    "promotion_contract_coverage_rate_current": 1.0,
                    "promotion_contract_coverage_rate_status": "pass",
                    "triple_extraction_fixture_matrix_mean_best_f1_baseline": 0.9,
                    "triple_extraction_fixture_matrix_mean_best_f1_current": 0.88,
                    "triple_extraction_fixture_matrix_mean_best_f1_status": "pass",
                    "triple_audit_pass_rate_baseline": 1.0,
                    "triple_audit_pass_rate_current": 1.0,
                    "triple_audit_pass_rate_status": "pass",
                    "triple_slot_coverage_rate_baseline": 1.0,
                    "triple_slot_coverage_rate_current": 1.0,
                    "triple_slot_coverage_rate_status": "pass",
                    "covered_fact_recommended_route_property_metric_count_baseline": 3,
                    "covered_fact_recommended_route_property_metric_count_current": 3,
                    "covered_fact_recommended_route_property_metric_count_status": "pass",
                    "covered_fact_recommended_route_min_records_baseline": 9,
                    "covered_fact_recommended_route_min_records_current": 9,
                    "covered_fact_recommended_route_min_records_status": "pass",
                    "covered_fact_recommended_route_min_source_documents_baseline": 120,
                    "covered_fact_recommended_route_min_source_documents_current": 118,
                    "covered_fact_recommended_route_min_source_documents_status": "pass",
                    "covered_fact_recommended_route_min_decision_accuracy_baseline": 1.0,
                    "covered_fact_recommended_route_min_decision_accuracy_current": 0.99,
                    "covered_fact_recommended_route_min_decision_accuracy_status": "pass",
                    "covered_fact_recommended_route_max_false_supported_rate_baseline": 0.01,
                    "covered_fact_recommended_route_max_false_supported_rate_current": 0.02,
                    "covered_fact_recommended_route_max_false_supported_rate_status": "pass",
                    "covered_fact_recommended_route_min_false_refuted_rate_baseline": 0.98,
                    "covered_fact_recommended_route_min_false_refuted_rate_current": 0.97,
                    "covered_fact_recommended_route_min_false_refuted_rate_status": "pass",
                    "world_model_participating_trace_rate_baseline": 1.0,
                    "world_model_participating_trace_rate_current": 1.0,
                    "world_model_participating_trace_rate_status": "pass",
                    "world_model_coverage_rate_baseline": 1.0,
                    "world_model_coverage_rate_current": 1.0,
                    "world_model_coverage_rate_status": "pass",
                    "world_model_conflict_rate_baseline": 0.0,
                    "world_model_conflict_rate_current": 0.0,
                    "world_model_conflict_rate_status": "pass",
                    "world_model_low_agreement_rate_baseline": 0.0,
                    "world_model_low_agreement_rate_current": 0.0,
                    "world_model_low_agreement_rate_status": "pass",
                    "world_model_trace_gap_rate_baseline": 0.0,
                    "world_model_trace_gap_rate_current": 0.0,
                    "world_model_trace_gap_rate_status": "pass",
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
                "covered_fact_property_count": 3,
                "covered_fact_properties": ["P36", "P37", "P38"],
                "covered_fact_property_metrics": {
                    "P36": {"decision_accuracy": 1.0, "n_records": 16},
                    "P37": {"decision_accuracy": 1.0, "n_records": 12},
                    "P38": {"decision_accuracy": 1.0, "n_records": 9},
                },
            },
            "required_route_baselines": {
                "records": ["benchmark_manifest:retrieval-structured-qa:0.5"],
                "routes": ["retrieval_structured_qa"],
                "manifest_paths": ["artifacts/retrieval/audit-manifest.json"],
                "registry": "artifacts/staged-route-registry.json",
                "covered_fact_property_counts": {
                    "benchmark_manifest:retrieval-structured-qa:0.5": 3
                },
                "covered_fact_properties": {
                    "benchmark_manifest:retrieval-structured-qa:0.5": [
                        "P36",
                        "P37",
                        "P38",
                    ]
                },
                "covered_fact_property_metrics": {
                    "benchmark_manifest:retrieval-structured-qa:0.5": {
                        "P36": {"decision_accuracy": 1.0, "n_records": 16},
                        "P37": {"decision_accuracy": 1.0, "n_records": 12},
                        "P38": {"decision_accuracy": 1.0, "n_records": 9},
                    }
                },
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
                "context_sensitivity_workflow_manifest": (
                    "artifacts/context-sensitivity/artifact-manifest.json"
                ),
                "pathway_intervention_workflow_manifest": (
                    "artifacts/pathway-intervention/artifact-manifest.json"
                ),
                "triple_extraction_fixture_matrix_manifest": (
                    "artifacts/triple-extraction-fixture-matrix/artifact-manifest.json"
                ),
                "counterfactual_verification_manifest": (
                    "artifacts/counterfactual/artifact-manifest.json"
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
    summary = contract.to_summary_dict()
    direct_summary = product_promotion_contract_summary(contract.to_dict())

    assert contract.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert contract.to_dict()["summary"] == summary
    assert direct_summary == summary
    assert summary["status"] == "promote"
    assert summary["source_status"] == "promote"
    assert summary["runtime"]["layer"] == -12
    assert summary["runtime"]["recommended_runtime_seconds"] == 0.20
    assert summary["verifier_route"]["route"] == "structured_state"
    assert summary["verifier_route"]["covered_fact_property_count"] == 3
    assert summary["gate_statuses"]["readiness"] == "promote"
    assert summary["gate_statuses"]["performance"] == "promote"
    assert summary["gate_statuses"]["product_runtime_drift"] == "promote"
    assert summary["gate_statuses"]["adapter_family"] == "promote"
    assert summary["gate_statuses"]["context_sensitivity"] == "promote"
    assert summary["blocking_gate_count"] == 0
    assert summary["evidence_groups"]["covered_fact_property"]["metric_count"] == 6
    assert summary["evidence_groups"]["covered_fact_property"]["blocked_metric_count"] == 0
    assert summary["action_gates"]["action_audit_status"] == "promote"
    assert summary["action_gates"]["action_execution_status"] == "promote"
    assert roundtrip.world_model_signal_workflow == contract.world_model_signal_workflow
    assert roundtrip.context_sensitivity_workflow == (
        contract.context_sensitivity_workflow
    )
    assert roundtrip.pathway_intervention_workflow == (
        contract.pathway_intervention_workflow
    )
    assert roundtrip.external_evidence_baseline_comparison == (
        contract.external_evidence_baseline_comparison
    )
    assert roundtrip.pre_generation_probe_comparison == (
        contract.pre_generation_probe_comparison
    )
    assert roundtrip.counterfactual_verification == (
        contract.counterfactual_verification
    )
    assert roundtrip.triple_extraction_fixture_matrix == (
        contract.triple_extraction_fixture_matrix
    )
    assert contract.runtime["layer"] == -12
    assert contract.verifier_route["route"] == "structured_state"
    assert contract.verifier_route["covered_fact_properties"] == ["P36", "P37", "P38"]
    assert contract.metadata["runtime_profile"] == "balanced"
    assert contract.metadata["max_uncached_forward_seconds"] is None
    assert contract.metadata["max_recommended_runtime_seconds"] == 1.0
    assert contract.metadata["recommended_runtime_seconds"] == 0.20
    assert contract.metadata["recommended_runtime_cost_source"] == (
        "cache_only_total_seconds"
    )
    assert contract.metadata["uncached_forward_cost_seconds"] == 37.5
    assert contract.metadata["uncached_forward_cost_source"] == (
        "uncached_forced_answer_forward_seconds"
    )
    assert contract.metadata["cache_only_total_seconds"] == 0.20
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
    assert contract.metadata["recommended_route_covered_fact_property_count"] == 3
    assert contract.metadata["recommended_route_covered_fact_properties"] == [
        "P36",
        "P37",
        "P38",
    ]
    assert contract.metadata["recommended_route_covered_fact_property_metrics"] == {
        "P36": {"decision_accuracy": 1.0, "n_records": 16},
        "P37": {"decision_accuracy": 1.0, "n_records": 12},
        "P38": {"decision_accuracy": 1.0, "n_records": 9},
    }
    assert contract.metadata["required_route_baseline_covered_fact_property_counts"] == {
        "benchmark_manifest:retrieval-structured-qa:0.5": 3
    }
    assert contract.metadata["required_route_baseline_covered_fact_properties"] == {
        "benchmark_manifest:retrieval-structured-qa:0.5": ["P36", "P37", "P38"]
    }
    assert contract.metadata["required_route_baseline_covered_fact_property_metrics"] == {
        "benchmark_manifest:retrieval-structured-qa:0.5": {
            "P36": {"decision_accuracy": 1.0, "n_records": 16},
            "P37": {"decision_accuracy": 1.0, "n_records": 12},
            "P38": {"decision_accuracy": 1.0, "n_records": 9},
        }
    }
    assert contract.metadata["structured_fact_robustness_property_counts"] == {
        "benchmark_manifest:retrieval-structured-qa:0.5": 3
    }
    assert contract.metadata["structured_fact_robustness_properties"] == {
        "benchmark_manifest:retrieval-structured-qa:0.5": ["P36", "P37", "P38"]
    }
    assert contract.metadata["structured_fact_robustness_property_metrics"] == {
        "benchmark_manifest:retrieval-structured-qa:0.5": {
            "P36": {"decision_accuracy": 1.0, "n_records": 16},
            "P37": {"decision_accuracy": 1.0, "n_records": 12},
            "P38": {"decision_accuracy": 1.0, "n_records": 9},
        }
    }
    assert contract.metadata["recommended_product_runtime_drift_report"] == (
        "artifacts/runtime-drift/product-runtime-drift.json"
    )
    expected_trace_replay_workflow = {
        "report_path": "artifacts/trace-replay-workflow/product-trace-replay-workflow.json",
        "manifest_path": "artifacts/trace-replay-workflow/artifact-manifest.json",
        "source": "registry",
        "registry": "artifacts/release-registry.json",
        "record_key": "report:trace-replay-workflow:0.1",
        "report_status": "promote",
        "selector_replay_report_path": "artifacts/selector/runtime-profile-selector-replay.json",
        "product_runtime_drift_report_path": "artifacts/runtime-drift/product-runtime-drift.json",
    }
    for key, value in expected_trace_replay_workflow.items():
        assert contract.product_trace_replay_workflow[key] == value
    assert contract.product_trace_replay_workflow["action_audit_gate"]["status"] == "promote"
    assert (
        contract.product_trace_replay_workflow["action_execution_gate"]["status"]
        == "promote"
    )
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
    assert contract.context_sensitivity_workflow == {
        "report_path": "artifacts/context-sensitivity/context-sensitivity-workflow.json",
        "manifest_path": "artifacts/context-sensitivity/artifact-manifest.json",
        "source": "registry",
        "registry": "artifacts/release-registry.json",
        "record_key": "report:context-sensitivity-workflow:0.1",
        "workflow": "context_sensitivity_workflow",
        "status": "promote",
        "paired_logprob_record_count": 6,
        "enriched_record_count": 6,
        "enhanced_score_signal_count": 4,
        "max_flagged_rate": 0.25,
        "mean_flagged_rate": 0.125,
        "max_context_sensitivity_ratio": 1.35,
        "manifest_verified": True,
        "blocking_reasons": [],
    }
    assert contract.metadata["context_sensitivity_workflow_status"] == "promote"
    assert contract.metadata["recommended_context_sensitivity_workflow_report"] == (
        "artifacts/context-sensitivity/context-sensitivity-workflow.json"
    )
    assert contract.metadata["context_sensitivity_workflow_paired_logprob_record_count"] == 6
    assert contract.metadata["context_sensitivity_workflow_enriched_record_count"] == 6
    assert contract.metadata["context_sensitivity_workflow_enhanced_score_signal_count"] == 4
    assert contract.metadata["context_sensitivity_workflow_max_flagged_rate"] == 0.25
    assert contract.metadata["context_sensitivity_workflow_mean_flagged_rate"] == 0.125
    assert (
        contract.metadata["context_sensitivity_workflow_max_context_sensitivity_ratio"]
        == 1.35
    )
    assert contract.metadata["context_sensitivity_workflow_manifest_verified"] is True
    assert contract.pathway_intervention_workflow == {
        "report_path": "artifacts/pathway-intervention/pathway-intervention-workflow.json",
        "manifest_path": "artifacts/pathway-intervention/artifact-manifest.json",
        "source": "registry",
        "registry": "artifacts/release-registry.json",
        "record_key": "report:pathway-intervention-workflow:0.1",
        "workflow": "pathway_intervention_workflow",
        "status": "promote",
        "report_status": "complete",
        "release_ready": True,
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "layer": -8,
        "intervention_layer": -8,
        "patch_layer": -8,
        "signals": ["pathway_disagreement", "truth_proj", "nll_answer"],
        "activation_ablation_gate_status": "promote",
        "source_patch_gate_status": "promote",
        "best_signals": {
            "activation_ablation": "pathway_disagreement",
            "source_patch": "truth_proj",
        },
        "blocking_reasons": [],
    }
    assert contract.metadata["pathway_intervention_workflow_status"] == "promote"
    assert contract.metadata["recommended_pathway_intervention_workflow_report"] == (
        "artifacts/pathway-intervention/pathway-intervention-workflow.json"
    )
    assert contract.metadata["pathway_intervention_workflow_release_ready"] is True
    assert contract.metadata["pathway_intervention_workflow_activation_ablation_gate"] == (
        "promote"
    )
    assert contract.metadata["pathway_intervention_workflow_source_patch_gate"] == "promote"
    assert contract.external_evidence_baseline_comparison["record_key"] == (
        "report:covered-facts-external-evidence-handoff:0.4"
    )
    assert contract.external_evidence_baseline_comparison["recommended_route"] == (
        "structured_fact"
    )
    assert contract.metadata["external_evidence_baseline_comparison_status"] == "promote"
    assert contract.metadata[
        "recommended_external_evidence_baseline_comparison_report"
    ] == "artifacts/external-evidence/external-evidence-baseline-comparison.json"
    assert contract.metadata["external_evidence_baseline_comparison_record"] == (
        "report:covered-facts-external-evidence-handoff:0.4"
    )
    assert contract.metadata[
        "external_evidence_baseline_comparison_recommended_route"
    ] == "structured_fact"
    assert contract.metadata["external_evidence_baseline_comparison_route_passed"] is True
    assert (
        contract.metadata["external_evidence_baseline_comparison_text_redline_run_count"]
        == 2
    )
    assert contract.pre_generation_probe_comparison["record_key"] == (
        "report:pre-generation-probe-comparison:0.1"
    )
    assert contract.metadata["pre_generation_probe_comparison_status"] == "promote"
    assert contract.metadata["recommended_pre_generation_probe_comparison_report"] == (
        "artifacts/pre-generation-probe-comparison/comparison.json"
    )
    assert contract.metadata["pre_generation_probe_comparison_report"] == (
        "artifacts/pre-generation-probe-comparison/comparison.json"
    )
    assert contract.metadata["pre_generation_probe_comparison_record"] == (
        "report:pre-generation-probe-comparison:0.1"
    )
    assert contract.metadata["pre_generation_probe_comparison_model_count"] == 2
    assert contract.metadata["pre_generation_probe_comparison_redline_passed"] is True
    assert contract.metadata["pre_generation_probe_comparison_best_run"] == "qwen05"
    assert contract.metadata["pre_generation_probe_comparison_best_redline_signal"] == (
        "answer_token_count"
    )
    assert contract.metadata["pre_generation_probe_comparison_best_redline_margin"] == 0.13
    assert contract.claim_factuality_probe_comparison["record_key"] == (
        "report:claim-factuality-probe-comparison:0.1"
    )
    assert contract.metadata["claim_factuality_probe_comparison_status"] == "promote"
    assert contract.metadata["recommended_claim_factuality_probe_comparison_report"] == (
        "artifacts/claim-factuality-probe-comparison/comparison.json"
    )
    assert contract.metadata["claim_factuality_probe_comparison_report"] == (
        "artifacts/claim-factuality-probe-comparison/comparison.json"
    )
    assert contract.metadata["claim_factuality_probe_comparison_record"] == (
        "report:claim-factuality-probe-comparison:0.1"
    )
    assert contract.metadata["claim_factuality_probe_comparison_model_count"] == 2
    assert contract.metadata[
        "claim_factuality_probe_comparison_best_test_selective_accuracy"
    ] == 0.91
    assert contract.metadata["claim_factuality_probe_comparison_best_conformal_threshold"] == 0.62
    assert contract.metadata["claim_factuality_probe_comparison_best_redline_signal"] == (
        "answer_negation_flag"
    )
    assert contract.metadata["claim_factuality_probe_comparison_best_redline_margin"] == 0.18
    assert contract.counterfactual_verification["record_key"] == (
        "report:counterfactual-verifier-audit:0.1"
    )
    assert contract.metadata["counterfactual_verification_status"] == "promote"
    assert contract.metadata["recommended_counterfactual_verification_report"] == (
        "artifacts/counterfactual/counterfactual-verification.json"
    )
    assert contract.metadata["counterfactual_verification_manifest"] == (
        "artifacts/counterfactual/artifact-manifest.json"
    )
    assert contract.metadata["counterfactual_verification_record_count"] == 12
    assert contract.metadata["counterfactual_verification_pass_rate"] == pytest.approx(1.0)
    assert contract.metadata["counterfactual_verification_false_invariance_rate"] == (
        pytest.approx(0.0)
    )
    assert contract.metadata["counterfactual_verification_flip_success_count"] == 12
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
    assert contract.metadata["product_runtime_drift_promotion_evidence_metric_count"] == 4
    assert contract.metadata["product_runtime_drift_promotion_evidence_blocked_metric_count"] == 0
    assert contract.metadata["product_runtime_drift_triple_audit_evidence_metric_count"] == 4
    assert contract.metadata["product_runtime_drift_triple_audit_evidence_blocked_metric_count"] == 0
    assert contract.metadata["product_runtime_drift_covered_fact_property_evidence_required"] is True
    assert contract.metadata["product_runtime_drift_covered_fact_property_evidence_metric_count"] == 6
    assert (
        contract.metadata[
            "product_runtime_drift_covered_fact_property_evidence_blocked_metric_count"
        ]
        == 0
    )
    assert contract.metadata["product_runtime_drift_world_model_evidence_required"] is True
    assert contract.metadata["product_runtime_drift_world_model_evidence_metric_count"] == 5
    assert contract.metadata["product_runtime_drift_world_model_evidence_blocked_metric_count"] == 0
    assert contract.metadata["product_runtime_drift_promotion_contract_coverage_rate_current"] == 1.0
    assert contract.metadata["product_runtime_drift_promotion_contract_coverage_rate_status"] == "pass"
    assert (
        contract.metadata[
            "product_runtime_drift_triple_extraction_fixture_matrix_mean_best_f1_current"
        ]
        == 0.88
    )
    assert contract.metadata["product_runtime_drift_triple_audit_pass_rate_current"] == 1.0
    assert contract.metadata["product_runtime_drift_triple_slot_coverage_rate_status"] == "pass"
    assert (
        contract.metadata[
            "product_runtime_drift_covered_fact_recommended_route_min_records_current"
        ]
        == 9
    )
    assert (
        contract.metadata[
            "product_runtime_drift_covered_fact_recommended_route_min_records_status"
        ]
        == "pass"
    )
    assert (
        contract.metadata[
            "product_runtime_drift_covered_fact_recommended_route_max_false_supported_rate_current"
        ]
        == 0.02
    )
    assert contract.metadata["product_runtime_drift_world_model_trace_gap_rate_status"] == "pass"
    assert contract.metadata["product_runtime_drift_world_model_conflict_rate_current"] == 0.0
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
    trace_metadata = product_promotion_contract_metadata(
        contract,
        source=str(contract_path),
        budget_enabled=True,
    )
    assert trace_metadata["promotion_contract_recommended_runtime_seconds"] == 0.20
    assert trace_metadata["promotion_contract_recommended_runtime_cost_source"] == (
        "cache_only_total_seconds"
    )
    assert trace_metadata["promotion_contract_uncached_forward_cost_seconds"] == 37.5
    assert trace_metadata["promotion_contract_max_recommended_runtime_seconds"] == 1.0
    json.dumps(contract.to_dict())


def test_product_promotion_contract_loader_selects_default_and_metadata(tmp_path):
    missing_path = tmp_path / "missing.json"
    contract_path = tmp_path / "product-promotion-contract.json"
    ProductPromotionContract(
        model_id="demo-model",
        runtime={"layer": -2},
        verifier_route={
            "route": "structured_qa",
            "covered_fact_property_count": 3,
            "covered_fact_properties": ["P36", "P37", "P38"],
            "covered_fact_property_metrics": {
                "P36": {"decision_accuracy": 1.0, "n_records": 16},
            },
        },
        runtime_budget_policy=ProductRuntimeBudgetPolicy(max_mean_attempted_route_count=1.1),
        source_workflow="release_candidate_comparison",
        source_status="promote",
        product_trace_replay_workflow={
            "report_path": "trace-replay-workflow.json",
            "manifest_path": "trace-replay-manifest.json",
            "source": "registry",
            "registry": "release-registry.json",
            "record_key": "report:trace-replay-workflow:0.1",
            "report_status": "promote",
            "selector_replay_report_path": "selector-replay.json",
            "product_runtime_drift_report_path": "product-runtime-drift.json",
            "require_action_audit_gate": True,
            "action_audit_gate_report_path": "action-audit-gate.json",
            "action_audit_gate": {
                "status": "promote",
                "gate_enabled": True,
                "passed": True,
                "error_rate": 0.0,
                "missing_retrieval_action_rate": 0.0,
                "missing_plan_retrieval_query_rate": 0.0,
                "malformed_payload_rate": 0.0,
                "unexpected_action_rate": 0.0,
                "unknown_claim_id_rate": 0.0,
            },
            "require_action_execution_gate": True,
            "action_execution_gate_report_path": "action-execution-gate.json",
            "action_execution_gate": {
                "status": "promote",
                "gate_enabled": True,
                "passed": True,
                "alignment_failed_trace_rate": 0.0,
                "missing_result_rate": 0.0,
                "unexpected_result_rate": 0.0,
                "request_id_mismatch_rate": 0.0,
            },
        },
        world_model_signal_workflow={
            "report_path": "world-model-signal-workflow.json",
            "record_key": "report:world-model-signal-workflow:0.1",
            "release_gate_status": "promote",
            "trace_gap_max": 0.0,
        },
        context_sensitivity_workflow={
            "report_path": "context-sensitivity-workflow.json",
            "record_key": "report:context-sensitivity-workflow:0.1",
            "status": "promote",
            "paired_logprob_record_count": 6,
            "enriched_record_count": 6,
            "enhanced_score_signal_count": 4,
            "max_flagged_rate": 0.25,
            "mean_flagged_rate": 0.125,
            "max_context_sensitivity_ratio": 1.35,
        },
        pathway_intervention_workflow={
            "report_path": "pathway-intervention-workflow.json",
            "manifest_path": "pathway-artifact-manifest.json",
            "record_key": "report:pathway-intervention-workflow:0.1",
            "status": "promote",
            "report_status": "complete",
            "release_ready": True,
            "model": "demo-model",
            "layer": -8,
            "intervention_layer": -8,
            "patch_layer": -8,
            "signals": ["pathway_disagreement", "truth_proj"],
            "activation_ablation_gate_status": "promote",
            "source_patch_gate_status": "promote",
            "best_signals": {
                "activation_ablation": "pathway_disagreement",
                "source_patch": "truth_proj",
            },
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
        external_evidence_baseline_comparison={
            "report_path": "external-evidence-baseline-comparison.json",
            "record_key": "report:covered-facts-external-evidence-handoff:0.4",
            "status": "promote",
            "recommended_route": "structured_fact",
            "route_passed": True,
            "text_redline_passed": True,
        },
        pre_generation_probe_comparison={
            "report_path": "pre-generation-probe-comparison.json",
            "record_key": "report:pre-generation-probe-comparison:0.1",
            "status": "promote",
            "model_count": 2,
            "run_count": 2,
            "redline_passed": True,
            "redline_run_count": 2,
            "best_run": {
                "name": "qwen05",
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "recommended_layer": -12,
                "test_label_auroc": 0.74,
                "redline_best_signal": "answer_token_count",
                "redline_best_auroc": 0.61,
                "redline_margin": 0.13,
            },
        },
        claim_factuality_probe_comparison={
            "report_path": "claim-factuality-probe-comparison.json",
            "record_key": "report:claim-factuality-probe-comparison:0.1",
            "status": "promote",
            "report_status": "ready",
            "model_count": 2,
            "run_count": 2,
            "redline_passed": True,
            "redline_run_count": 2,
            "best_run": {
                "name": "qwen05",
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "record_count": 96,
                "recommended_layer": -4,
                "test_label_auroc": 0.84,
                "test_selective_accuracy": 0.91,
                "test_selective_coverage": 0.78,
                "conformal_threshold": 0.62,
                "redline_best_signal": "answer_negation_flag",
                "redline_best_auroc": 0.66,
                "redline_margin": 0.18,
            },
        },
        counterfactual_verification={
            "report_path": "counterfactual-verification.json",
            "record_key": "report:counterfactual-verifier-audit:0.1",
            "workflow": "counterfactual_verification_eval",
            "status": "promote",
            "record_count": 4,
            "pass_rate": 1.0,
            "false_invariance_rate": 0.0,
            "flip_success_count": 4,
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
        frontier_release_evidence={
            "report_path": "frontier-release-evidence.json",
            "manifest_path": "frontier-release-evidence-manifest.json",
            "source": "registry",
            "registry": "release-registry.json",
            "record_key": "report:frontier-release-evidence:0.1",
            "workflow": "frontier_release_evidence_comparison",
            "status": "promote",
            "report_status": "complete",
            "decision_status": "promote",
            "verifier_track_status": "promote",
            "abstention_track_status": "promote",
            "citation_batch_track_status": "promote",
            "frontier_rerun_rollup_track_status": "promote",
            "base_verifier_track_status": "promote",
            "base_abstention_track_status": "blocked",
            "base_detectability_track_status": "blocked",
            "base_multiple_testing_track_status": "promote",
            "frontier_rerun_rollup_promoted_tracks": [
                "abstention",
                "detectability",
            ],
            "frontier_rerun_rollup_report_count": 4,
            "frontier_rerun_rollup_candidate_count": 4,
            "frontier_rerun_rollup_missing_report_count": 0,
            "frontier_rerun_rollup_invalid_report_count": 0,
            "frontier_rerun_rollup_blocked_candidate_count": 0,
            "frontier_rerun_rollup_promotion_ready_count": 4,
            "citation_batch_rollup_count": 1,
            "citation_batch_expected_batch_count": 2,
            "citation_batch_observed_batch_count": 2,
            "citation_batch_missing_expected_batch_count": 0,
            "run_names": ["verifier-stability", "abstention-stability"],
        },
        control_defaults={"max_verifier_route_attempts": 3},
        metadata={
            "selector_replay_status": "promote",
            "product_runtime_drift_covered_fact_property_evidence_required": True,
            "product_runtime_drift_covered_fact_property_evidence_metric_count": 6,
            "product_runtime_drift_covered_fact_property_evidence_blocked_metric_count": 0,
            "product_runtime_drift_world_model_evidence_required": True,
            "product_runtime_drift_world_model_evidence_metric_count": 5,
            "product_runtime_drift_world_model_evidence_blocked_metric_count": 0,
            "product_runtime_drift_world_model_trace_gap_rate_baseline": 0.0,
            "product_runtime_drift_world_model_trace_gap_rate_current": 0.0,
            "product_runtime_drift_world_model_trace_gap_rate_status": "pass",
            "product_runtime_drift_covered_fact_recommended_route_min_records_baseline": 16,
            "product_runtime_drift_covered_fact_recommended_route_min_records_current": 15,
            "product_runtime_drift_covered_fact_recommended_route_min_records_status": "pass",
            "product_runtime_drift_covered_fact_recommended_route_min_decision_accuracy_baseline": 1.0,
            "product_runtime_drift_covered_fact_recommended_route_min_decision_accuracy_current": 0.99,
            "product_runtime_drift_covered_fact_recommended_route_min_decision_accuracy_status": "pass",
            "required_route_baseline_covered_fact_property_counts": {
                "benchmark_manifest:structured-fact:0.1": 3
            },
            "required_route_baseline_covered_fact_properties": {
                "benchmark_manifest:structured-fact:0.1": ["P36", "P37", "P38"]
            },
            "required_route_baseline_covered_fact_property_metrics": {
                "benchmark_manifest:structured-fact:0.1": {
                    "P36": {"decision_accuracy": 1.0, "n_records": 16}
                }
            },
        },
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
    assert metadata["promotion_contract_promotion_summary"]["status"] == "promote"
    assert metadata["promotion_contract_promotion_summary"]["source_status"] == "promote"
    assert (
        metadata["promotion_contract_promotion_summary"]["blocking_gate_count"] == 0
    )
    assert (
        metadata["promotion_contract_promotion_summary"][
            "blocked_evidence_group_count"
        ]
        == 0
    )
    assert metadata["promotion_contract_promotion_summary"]["runtime"]["layer"] == -2
    assert (
        metadata["promotion_contract_promotion_summary"]["verifier_route"]["route"]
        == "structured_qa"
    )
    assert (
        metadata["promotion_contract_promotion_summary"]["action_gates"][
            "action_audit_status"
        ]
        == "promote"
    )
    assert metadata["promotion_contract_promotion_summary"]["gate_statuses"][
        "frontier_release_evidence"
    ] == "promote"
    assert metadata["promotion_contract_promotion_summary"]["recommended_records"][
        "frontier_release_evidence"
    ] == "frontier-release-evidence.json"
    assert metadata["promotion_contract_frontier_release_evidence"]["status"] == (
        "promote"
    )
    assert metadata["promotion_contract_frontier_release_evidence_report"] == (
        "frontier-release-evidence.json"
    )
    assert metadata["promotion_contract_frontier_release_evidence_manifest"] == (
        "frontier-release-evidence-manifest.json"
    )
    assert metadata[
        "promotion_contract_frontier_release_evidence_decision_status"
    ] == "promote"
    assert metadata[
        "promotion_contract_frontier_release_evidence_verifier_track_status"
    ] == "promote"
    assert metadata[
        "promotion_contract_frontier_release_evidence_abstention_track_status"
    ] == "promote"
    assert metadata[
        "promotion_contract_frontier_release_evidence_citation_batch_track_status"
    ] == "promote"
    assert metadata[
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_track_status"
    ] == "promote"
    assert metadata[
        "promotion_contract_frontier_release_evidence_base_abstention_track_status"
    ] == "blocked"
    assert metadata[
        "promotion_contract_frontier_release_evidence_base_detectability_track_status"
    ] == "blocked"
    assert metadata[
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_promoted_tracks"
    ] == ["abstention", "detectability"]
    assert metadata[
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_report_count"
    ] == 4
    assert metadata[
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_missing_report_count"
    ] == 0
    assert metadata[
        "promotion_contract_frontier_release_evidence_citation_batch_expected_batch_count"
    ] == 2
    assert metadata[
        "promotion_contract_frontier_release_evidence_citation_batch_observed_batch_count"
    ] == 2
    assert metadata[
        "promotion_contract_frontier_release_evidence_citation_batch_missing_expected_batch_count"
    ] == 0
    assert metadata["promotion_contract_runtime"] == {"layer": -2}
    assert metadata["promotion_contract_verifier_route"] == {
        "route": "structured_qa",
        "covered_fact_property_count": 3,
        "covered_fact_properties": ["P36", "P37", "P38"],
        "covered_fact_property_metrics": {
            "P36": {"decision_accuracy": 1.0, "n_records": 16},
        },
    }
    assert metadata["promotion_contract_recommended_route_covered_fact_property_count"] == 3
    assert metadata["promotion_contract_recommended_route_covered_fact_properties"] == [
        "P36",
        "P37",
        "P38",
    ]
    assert metadata[
        "promotion_contract_required_route_baseline_covered_fact_property_counts"
    ] == {"benchmark_manifest:structured-fact:0.1": 3}
    assert metadata[
        "promotion_contract_recommended_route_covered_fact_property_metrics"
    ] == {"P36": {"decision_accuracy": 1.0, "n_records": 16}}
    assert metadata[
        "promotion_contract_required_route_baseline_covered_fact_property_metrics"
    ] == {
        "benchmark_manifest:structured-fact:0.1": {
            "P36": {"decision_accuracy": 1.0, "n_records": 16}
        }
    }
    runtime_metrics = product_runtime_metrics({"metadata": metadata})
    assert runtime_metrics["promotion_contract_promotion_summary_status"] == "promote"
    assert runtime_metrics["promotion_contract_promotion_summary_source_status"] == (
        "promote"
    )
    assert runtime_metrics[
        "promotion_contract_promotion_summary_blocking_gate_count"
    ] == pytest.approx(0.0)
    assert runtime_metrics[
        "promotion_contract_promotion_summary_blocked_evidence_group_count"
    ] == pytest.approx(0.0)
    assert runtime_metrics[
        "promotion_contract_promotion_summary_runtime_layer"
    ] == pytest.approx(-2.0)
    assert runtime_metrics["promotion_contract_promotion_summary_route"] == (
        "structured_qa"
    )
    assert runtime_metrics[
        "promotion_contract_promotion_summary_action_audit_status"
    ] == "promote"
    assert runtime_metrics["promotion_contract_frontier_release_evidence_available"] is True
    assert runtime_metrics["promotion_contract_frontier_release_evidence_status"] == (
        "promote"
    )
    assert runtime_metrics["promotion_contract_frontier_release_evidence_report"] == (
        "frontier-release-evidence.json"
    )
    assert runtime_metrics[
        "promotion_contract_frontier_release_evidence_decision_status"
    ] == "promote"
    assert runtime_metrics[
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_track_status"
    ] == "promote"
    assert runtime_metrics[
        "promotion_contract_frontier_release_evidence_base_abstention_track_status"
    ] == "blocked"
    assert runtime_metrics[
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_promoted_tracks"
    ] == ["abstention", "detectability"]
    assert runtime_metrics[
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_report_count"
    ] == pytest.approx(4.0)
    assert runtime_metrics[
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_promotion_ready_count"
    ] == pytest.approx(4.0)
    assert runtime_metrics[
        "promotion_contract_frontier_release_evidence_run_count"
    ] == pytest.approx(2.0)
    assert runtime_metrics[
        "promotion_contract_recommended_route_covered_fact_property_metric_count"
    ] == pytest.approx(1.0)
    assert runtime_metrics[
        "promotion_contract_recommended_route_covered_fact_min_records"
    ] == pytest.approx(16.0)
    assert runtime_metrics[
        "promotion_contract_recommended_route_covered_fact_min_decision_accuracy"
    ] == pytest.approx(1.0)
    assert runtime_metrics[
        "promotion_contract_required_route_baseline_covered_fact_property_metric_count"
    ] == pytest.approx(1.0)
    assert runtime_metrics[
        "promotion_contract_required_route_baseline_covered_fact_min_records"
    ] == pytest.approx(16.0)
    assert runtime_metrics["promotion_contract_summary"]["covered_fact_property_rollups"][
        "recommended_route"
    ]["min_records"] == pytest.approx(16.0)
    assert (
        runtime_metrics[
            "promotion_contract_external_evidence_baseline_comparison_available"
        ]
        is True
    )
    assert runtime_metrics[
        "promotion_contract_external_evidence_baseline_comparison_record"
    ] == "report:covered-facts-external-evidence-handoff:0.4"
    assert runtime_metrics[
        "promotion_contract_external_evidence_baseline_comparison_status"
    ] == "promote"
    assert runtime_metrics[
        "promotion_contract_external_evidence_baseline_comparison_recommended_route"
    ] == "structured_fact"
    assert (
        runtime_metrics[
            "promotion_contract_external_evidence_baseline_comparison_route_passed"
        ]
        is True
    )
    assert (
        runtime_metrics["promotion_contract_summary"][
            "external_evidence_baseline_comparison"
        ]["text_redline_passed"]
        is True
    )
    assert (
        runtime_metrics["promotion_contract_pre_generation_probe_comparison_available"]
        is True
    )
    assert runtime_metrics[
        "promotion_contract_pre_generation_probe_comparison_record"
    ] == "report:pre-generation-probe-comparison:0.1"
    assert runtime_metrics[
        "promotion_contract_pre_generation_probe_comparison_redline_passed"
    ] is True
    assert runtime_metrics[
        "promotion_contract_pre_generation_probe_comparison_best_redline_margin"
    ] == pytest.approx(0.13)
    assert runtime_metrics["promotion_contract_summary"][
        "pre_generation_probe_comparison"
    ]["best_redline_signal"] == "answer_token_count"
    assert (
        runtime_metrics["promotion_contract_claim_factuality_probe_comparison_available"]
        is True
    )
    assert runtime_metrics[
        "promotion_contract_claim_factuality_probe_comparison_record"
    ] == "report:claim-factuality-probe-comparison:0.1"
    assert runtime_metrics[
        "promotion_contract_claim_factuality_probe_comparison_best_test_selective_accuracy"
    ] == pytest.approx(0.91)
    assert runtime_metrics[
        "promotion_contract_claim_factuality_probe_comparison_best_conformal_threshold"
    ] == pytest.approx(0.62)
    assert runtime_metrics[
        "promotion_contract_claim_factuality_probe_comparison_best_redline_margin"
    ] == pytest.approx(0.18)
    assert runtime_metrics["promotion_contract_summary"][
        "claim_factuality_probe_comparison"
    ]["best_redline_signal"] == "answer_negation_flag"
    assert runtime_metrics["promotion_contract_counterfactual_verification_available"] is True
    assert runtime_metrics[
        "promotion_contract_counterfactual_verification_record"
    ] == "report:counterfactual-verifier-audit:0.1"
    assert runtime_metrics[
        "promotion_contract_counterfactual_verification_status"
    ] == "promote"
    assert runtime_metrics[
        "promotion_contract_counterfactual_verification_pass_rate"
    ] == pytest.approx(1.0)
    assert runtime_metrics[
        "promotion_contract_counterfactual_verification_false_invariance_rate"
    ] == pytest.approx(0.0)
    assert runtime_metrics["promotion_contract_summary"][
        "counterfactual_verification"
    ]["record_count"] == pytest.approx(4.0)
    assert runtime_metrics[
        "promotion_contract_pathway_intervention_workflow_available"
    ] is True
    assert runtime_metrics[
        "promotion_contract_pathway_intervention_workflow_record"
    ] == "report:pathway-intervention-workflow:0.1"
    assert runtime_metrics[
        "promotion_contract_pathway_intervention_workflow_release_ready"
    ] is True
    assert runtime_metrics[
        "promotion_contract_pathway_intervention_workflow_activation_ablation_gate"
    ] == "promote"
    assert runtime_metrics["promotion_contract_summary"][
        "pathway_intervention_workflow"
    ]["best_signals"] == {
        "activation_ablation": "pathway_disagreement",
        "source_patch": "truth_proj",
    }
    assert metadata[
        "promotion_contract_product_runtime_drift_covered_fact_property_evidence_required"
    ] is True
    assert metadata[
        "promotion_contract_product_runtime_drift_covered_fact_property_evidence_metric_count"
    ] == 6
    assert metadata[
        "promotion_contract_product_runtime_drift_covered_fact_property_evidence_blocked_metric_count"
    ] == 0
    assert metadata[
        "promotion_contract_product_runtime_drift_world_model_evidence_required"
    ] is True
    assert metadata[
        "promotion_contract_product_runtime_drift_world_model_evidence_metric_count"
    ] == 5
    assert metadata[
        "promotion_contract_product_runtime_drift_world_model_evidence_blocked_metric_count"
    ] == 0
    assert metadata[
        "promotion_contract_product_runtime_drift_world_model_trace_gap_rate_status"
    ] == "pass"
    assert metadata[
        "promotion_contract_product_runtime_drift_covered_fact_recommended_route_min_records_current"
    ] == 15
    assert metadata[
        "promotion_contract_product_runtime_drift_covered_fact_recommended_route_min_records_status"
    ] == "pass"
    assert (
        metadata[
            "promotion_contract_product_runtime_drift_covered_fact_recommended_route_min_decision_accuracy_current"
        ]
        == 0.99
    )
    assert runtime_metrics["promotion_contract_summary"]["product_runtime_drift"][
        "covered_fact_property_evidence_metric_count"
    ] == pytest.approx(6.0)
    assert runtime_metrics["promotion_contract_summary"]["product_runtime_drift"][
        "covered_fact_property_evidence"
    ]["covered_fact_recommended_route_min_records"]["current"] == pytest.approx(15.0)
    assert runtime_metrics["promotion_contract_summary"]["product_runtime_drift"][
        "world_model_evidence_metric_count"
    ] == pytest.approx(5.0)
    assert runtime_metrics["promotion_contract_summary"]["product_runtime_drift"][
        "world_model_evidence"
    ]["world_model_trace_gap_rate"]["status"] == "pass"
    assert runtime_metrics["promotion_contract_product_trace_replay_available"] is True
    assert runtime_metrics["promotion_contract_product_trace_replay_workflow_report"] == (
        "trace-replay-workflow.json"
    )
    assert runtime_metrics[
        "promotion_contract_product_trace_replay_workflow_manifest"
    ] == "trace-replay-manifest.json"
    assert runtime_metrics[
        "promotion_contract_product_trace_replay_workflow_selector_replay_report"
    ] == "selector-replay.json"
    assert runtime_metrics[
        "promotion_contract_product_trace_replay_workflow_runtime_drift_report"
    ] == "product-runtime-drift.json"
    assert runtime_metrics[
        "promotion_contract_product_trace_action_audit_gate_required"
    ] is True
    assert runtime_metrics[
        "promotion_contract_product_trace_action_audit_gate_status"
    ] == "promote"
    assert runtime_metrics[
        "promotion_contract_product_trace_action_audit_gate_enabled"
    ] is True
    assert runtime_metrics[
        "promotion_contract_product_trace_action_audit_gate_passed"
    ] is True
    assert runtime_metrics[
        "promotion_contract_product_trace_action_audit_missing_retrieval_action_rate"
    ] == pytest.approx(0.0)
    assert runtime_metrics[
        "promotion_contract_product_trace_action_execution_gate_required"
    ] is True
    assert runtime_metrics[
        "promotion_contract_product_trace_action_execution_gate_status"
    ] == "promote"
    assert runtime_metrics[
        "promotion_contract_product_trace_action_execution_gate_enabled"
    ] is True
    assert runtime_metrics[
        "promotion_contract_product_trace_action_execution_gate_passed"
    ] is True
    assert runtime_metrics[
        "promotion_contract_product_trace_action_execution_request_id_mismatch_rate"
    ] == pytest.approx(0.0)
    trace_replay_summary = runtime_metrics["promotion_contract_summary"][
        "product_trace_replay"
    ]
    assert trace_replay_summary["available"] is True
    assert trace_replay_summary["action_audit_gate"]["error_rate"] == pytest.approx(0.0)
    assert trace_replay_summary["action_execution_gate"][
        "request_id_mismatch_rate"
    ] == pytest.approx(0.0)
    assert metadata["promotion_contract_control_policy_config"]["unsupported_action"] == "clarify"
    assert metadata["promotion_contract_control_policy_config"][
        "compound_verification_escalates"
    ] is False
    assert metadata["promotion_contract_control_defaults"] == {
        "max_verifier_route_attempts": 3
    }
    assert metadata["promotion_contract_product_trace_replay_workflow"] == {
        "report_path": "trace-replay-workflow.json",
        "manifest_path": "trace-replay-manifest.json",
        "source": "registry",
        "registry": "release-registry.json",
        "record_key": "report:trace-replay-workflow:0.1",
        "report_status": "promote",
        "selector_replay_report_path": "selector-replay.json",
        "product_runtime_drift_report_path": "product-runtime-drift.json",
        "require_action_audit_gate": True,
        "action_audit_gate_report_path": "action-audit-gate.json",
        "action_audit_gate": {
            "status": "promote",
            "gate_enabled": True,
            "passed": True,
            "error_rate": 0.0,
            "missing_retrieval_action_rate": 0.0,
            "missing_plan_retrieval_query_rate": 0.0,
            "malformed_payload_rate": 0.0,
            "unexpected_action_rate": 0.0,
            "unknown_claim_id_rate": 0.0,
        },
        "require_action_execution_gate": True,
        "action_execution_gate_report_path": "action-execution-gate.json",
        "action_execution_gate": {
            "status": "promote",
            "gate_enabled": True,
            "passed": True,
            "alignment_failed_trace_rate": 0.0,
            "missing_result_rate": 0.0,
            "unexpected_result_rate": 0.0,
            "request_id_mismatch_rate": 0.0,
        },
    }
    assert metadata["promotion_contract_product_trace_replay_workflow_report"] == (
        "trace-replay-workflow.json"
    )
    assert metadata["promotion_contract_product_trace_replay_workflow_manifest"] == (
        "trace-replay-manifest.json"
    )
    assert metadata["promotion_contract_product_trace_replay_workflow_record"] == (
        "report:trace-replay-workflow:0.1"
    )
    assert metadata["promotion_contract_product_trace_replay_workflow_report_status"] == (
        "promote"
    )
    assert metadata[
        "promotion_contract_product_trace_replay_workflow_selector_replay_report"
    ] == "selector-replay.json"
    assert metadata[
        "promotion_contract_product_trace_replay_workflow_runtime_drift_report"
    ] == "product-runtime-drift.json"
    assert metadata["promotion_contract_product_trace_action_audit_gate_required"] is True
    assert metadata["promotion_contract_product_trace_action_audit_gate_status"] == "promote"
    assert metadata["promotion_contract_product_trace_action_audit_gate_enabled"] is True
    assert metadata["promotion_contract_product_trace_action_audit_gate_passed"] is True
    assert metadata["promotion_contract_product_trace_action_audit_gate_report"] == (
        "action-audit-gate.json"
    )
    assert metadata[
        "promotion_contract_product_trace_action_audit_missing_retrieval_action_rate"
    ] == 0.0
    assert metadata[
        "promotion_contract_product_trace_action_audit_missing_plan_retrieval_query_rate"
    ] == 0.0
    assert metadata[
        "promotion_contract_product_trace_action_execution_gate_required"
    ] is True
    assert metadata[
        "promotion_contract_product_trace_action_execution_gate_status"
    ] == "promote"
    assert metadata[
        "promotion_contract_product_trace_action_execution_gate_enabled"
    ] is True
    assert metadata[
        "promotion_contract_product_trace_action_execution_gate_passed"
    ] is True
    assert metadata["promotion_contract_product_trace_action_execution_gate_report"] == (
        "action-execution-gate.json"
    )
    assert metadata[
        "promotion_contract_product_trace_action_execution_request_id_mismatch_rate"
    ] == 0.0
    assert metadata["promotion_contract_world_model_signal_workflow"] == {
        "report_path": "world-model-signal-workflow.json",
        "record_key": "report:world-model-signal-workflow:0.1",
        "release_gate_status": "promote",
        "trace_gap_max": 0.0,
    }
    assert metadata["promotion_contract_context_sensitivity_workflow"] == {
        "report_path": "context-sensitivity-workflow.json",
        "record_key": "report:context-sensitivity-workflow:0.1",
        "status": "promote",
        "paired_logprob_record_count": 6,
        "enriched_record_count": 6,
        "enhanced_score_signal_count": 4,
        "max_flagged_rate": 0.25,
        "mean_flagged_rate": 0.125,
        "max_context_sensitivity_ratio": 1.35,
    }
    assert (
        metadata["promotion_contract_context_sensitivity_workflow_report"]
        == "context-sensitivity-workflow.json"
    )
    assert (
        metadata[
            "promotion_contract_context_sensitivity_workflow_paired_logprob_record_count"
        ]
        == 6
    )
    assert (
        metadata["promotion_contract_context_sensitivity_workflow_max_flagged_rate"]
        == 0.25
    )
    assert (
        metadata[
            "promotion_contract_context_sensitivity_workflow_max_context_sensitivity_ratio"
        ]
        == 1.35
    )
    assert metadata["promotion_contract_pathway_intervention_workflow"] == {
        "report_path": "pathway-intervention-workflow.json",
        "manifest_path": "pathway-artifact-manifest.json",
        "record_key": "report:pathway-intervention-workflow:0.1",
        "status": "promote",
        "report_status": "complete",
        "release_ready": True,
        "model": "demo-model",
        "layer": -8,
        "intervention_layer": -8,
        "patch_layer": -8,
        "signals": ["pathway_disagreement", "truth_proj"],
        "activation_ablation_gate_status": "promote",
        "source_patch_gate_status": "promote",
        "best_signals": {
            "activation_ablation": "pathway_disagreement",
            "source_patch": "truth_proj",
        },
    }
    assert metadata["promotion_contract_pathway_intervention_workflow_report"] == (
        "pathway-intervention-workflow.json"
    )
    assert metadata["promotion_contract_pathway_intervention_workflow_release_ready"] is True
    assert metadata[
        "promotion_contract_pathway_intervention_workflow_source_patch_gate"
    ] == "promote"
    assert metadata["promotion_contract_feedback_policy_workflow"] == {
        "report_path": "feedback-policy-workflow.json",
        "record_key": "report:feedback-policy-workflow:0.1",
        "promotion_decision": "promote_candidate_policy",
    }
    assert metadata["promotion_contract_external_evidence_baseline_comparison"] == {
        "report_path": "external-evidence-baseline-comparison.json",
        "record_key": "report:covered-facts-external-evidence-handoff:0.4",
        "status": "promote",
        "recommended_route": "structured_fact",
        "route_passed": True,
        "text_redline_passed": True,
    }
    assert metadata[
        "promotion_contract_external_evidence_baseline_comparison_report"
    ] == "external-evidence-baseline-comparison.json"
    assert metadata[
        "promotion_contract_external_evidence_baseline_comparison_record"
    ] == "report:covered-facts-external-evidence-handoff:0.4"
    assert metadata[
        "promotion_contract_external_evidence_baseline_comparison_route_passed"
    ] is True
    assert metadata["promotion_contract_pre_generation_probe_comparison"] == {
        "report_path": "pre-generation-probe-comparison.json",
        "record_key": "report:pre-generation-probe-comparison:0.1",
        "status": "promote",
        "model_count": 2,
        "run_count": 2,
        "redline_passed": True,
        "redline_run_count": 2,
        "best_run": {
            "name": "qwen05",
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "recommended_layer": -12,
            "test_label_auroc": 0.74,
            "redline_best_signal": "answer_token_count",
            "redline_best_auroc": 0.61,
            "redline_margin": 0.13,
        },
    }
    assert metadata["promotion_contract_pre_generation_probe_comparison_report"] == (
        "pre-generation-probe-comparison.json"
    )
    assert metadata["promotion_contract_pre_generation_probe_comparison_record"] == (
        "report:pre-generation-probe-comparison:0.1"
    )
    assert (
        metadata["promotion_contract_pre_generation_probe_comparison_best_redline_signal"]
        == "answer_token_count"
    )
    assert metadata["promotion_contract_counterfactual_verification"] == {
        "report_path": "counterfactual-verification.json",
        "record_key": "report:counterfactual-verifier-audit:0.1",
        "workflow": "counterfactual_verification_eval",
        "status": "promote",
        "record_count": 4,
        "pass_rate": 1.0,
        "false_invariance_rate": 0.0,
        "flip_success_count": 4,
    }
    assert metadata["promotion_contract_counterfactual_verification_report"] == (
        "counterfactual-verification.json"
    )
    assert metadata["promotion_contract_counterfactual_verification_record"] == (
        "report:counterfactual-verifier-audit:0.1"
    )
    assert metadata["promotion_contract_counterfactual_verification_pass_rate"] == 1.0
    assert (
        metadata["promotion_contract_counterfactual_verification_false_invariance_rate"]
        == 0.0
    )
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
    assert metadata["promotion_contract_metadata"]["selector_replay_status"] == "promote"
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
    context_sensitivity_dir = tmp_path / "context-sensitivity"
    context_sensitivity_dir.mkdir()
    context_sensitivity_report_path = context_sensitivity_dir / "workflow.json"
    context_sensitivity_manifest_path = context_sensitivity_dir / "artifact-manifest.json"
    context_sensitivity_registry_path = context_sensitivity_dir / "registry.json"
    pathway_dir = tmp_path / "pathway-intervention"
    pathway_dir.mkdir()
    pathway_report_path = pathway_dir / "workflow.json"
    pathway_manifest_path = pathway_dir / "artifact-manifest.json"
    pathway_registry_path = pathway_dir / "registry.json"
    external_evidence_dir = tmp_path / "external-evidence"
    external_evidence_dir.mkdir()
    external_evidence_report_path = external_evidence_dir / "comparison.json"
    external_evidence_registry_path = external_evidence_dir / "registry.json"
    pre_generation_dir = tmp_path / "pre-generation"
    pre_generation_dir.mkdir()
    pre_generation_report_path = pre_generation_dir / "comparison.json"
    pre_generation_manifest_path = pre_generation_dir / "artifact-manifest.json"
    pre_generation_registry_path = pre_generation_dir / "registry.json"
    triple_matrix_dir = tmp_path / "triple-extraction-fixture-matrix"
    triple_matrix_dir.mkdir()
    triple_matrix_report_path = triple_matrix_dir / "matrix.json"
    triple_matrix_manifest_path = triple_matrix_dir / "artifact-manifest.json"
    triple_matrix_registry_path = triple_matrix_dir / "registry.json"
    counterfactual_dir = tmp_path / "counterfactual"
    counterfactual_dir.mkdir()
    counterfactual_report_path = counterfactual_dir / "counterfactual-verification.json"
    counterfactual_manifest_path = counterfactual_dir / "artifact-manifest.json"
    counterfactual_registry_path = counterfactual_dir / "registry.json"
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
    context_sensitivity_report_path.write_text(
        json.dumps({
            "workflow": "context_sensitivity_workflow",
            "paired_logprobs": {"record_count": 6},
            "enrichment": {"record_count": 6},
            "enhanced_score_dump": {
                "score_dump_summary": {
                    "score_names": [
                        "context_sensitivity_flagged_rate",
                        "context_sensitivity_max_shift",
                        "context_sensitivity_mean_shift",
                        "context_sensitivity_max_ratio",
                    ]
                }
            },
            "signal_summary": {
                "context_sensitivity_flagged_rate": {
                    "max": 0.25,
                    "mean": 0.125,
                },
                "context_sensitivity_max_ratio": {"max": 1.35},
            },
            "manifest_verification": {"passed": True},
        }),
        encoding="utf-8",
    )
    context_sensitivity_manifest_path.write_text(
        json.dumps(
            build_artifact_manifest(
                {"context_sensitivity_workflow": context_sensitivity_report_path},
                root=context_sensitivity_dir,
                metadata={"workflow": "context_sensitivity_workflow"},
            )
        ),
        encoding="utf-8",
    )
    ArtifactRegistry.load_json(context_sensitivity_registry_path).record_report(
        name="context-sensitivity-workflow",
        path=context_sensitivity_report_path,
        version="0.1",
        metadata={"artifact_manifest": str(context_sensitivity_manifest_path)},
    ).save_json()
    pathway_report_path.write_text(
        json.dumps({
            "workflow": "pathway_intervention_workflow",
            "status": "complete",
            "evidence_bundle": {
                "release_ready": True,
                "model": "demo-model",
                "layer": -8,
                "intervention_layer": -8,
                "patch_layer": -8,
                "signals": ["pathway_disagreement", "truth_proj"],
                "best_signals": {
                    "activation_ablation": "pathway_disagreement",
                    "source_patch": "truth_proj",
                },
            },
            "comparisons": {
                "activation_ablation": {"gate": {"status": "promote"}},
                "source_patch": {"gate": {"status": "promote"}},
            },
        }),
        encoding="utf-8",
    )
    pathway_manifest_path.write_text(
        json.dumps(
            build_artifact_manifest(
                {"pathway_intervention_workflow": pathway_report_path},
                root=pathway_dir,
                metadata={"workflow": "pathway_intervention_workflow"},
            )
        ),
        encoding="utf-8",
    )
    ArtifactRegistry.load_json(pathway_registry_path).record_report(
        name="pathway-intervention-workflow",
        path=pathway_report_path,
        version="0.1",
        metadata={"artifact_manifest": str(pathway_manifest_path)},
    ).save_json()
    external_evidence_report_path.write_text(
        json.dumps({
            "workflow": "external_evidence_baseline_comparison",
            "decision": {
                "status": "promote",
                "recommended_route": "structured_fact",
                "recommended_route_record": (
                    "benchmark_manifest:structured-fact-canonical-route:0.1"
                ),
            },
        }),
        encoding="utf-8",
    )
    ArtifactRegistry.load_json(external_evidence_registry_path).record_report(
        name="covered-facts-external-evidence-handoff",
        path=external_evidence_report_path,
        version="0.4",
        metadata={"workflow": "external_evidence_baseline_comparison"},
    ).save_json()
    pre_generation_report_path.write_text(
        json.dumps({
            "workflow": "pre_generation_probe_workflow_comparison",
            "status": "ready",
            "model_count": 2,
            "redline_passed": True,
            "redline_run_count": 2,
        }),
        encoding="utf-8",
    )
    pre_generation_manifest_path.write_text(
        json.dumps(
            build_artifact_manifest(
                {"pre_generation_probe_comparison": pre_generation_report_path},
                root=pre_generation_dir,
                metadata={"workflow": "pre_generation_probe_workflow_comparison"},
            )
        ),
        encoding="utf-8",
    )
    ArtifactRegistry.load_json(pre_generation_registry_path).record_report(
        name="pre-generation-probe-comparison",
        path=pre_generation_report_path,
        version="0.1",
        metadata={"artifact_manifest": str(pre_generation_manifest_path)},
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
    counterfactual_report_path.write_text(
        json.dumps({
            "workflow": "counterfactual_verification_eval",
            "status": "promote",
            "summary": {
                "record_count": 4,
                "pass_rate": 1.0,
                "false_invariance_rate": 0.0,
                "flip_success_count": 4,
            },
        }),
        encoding="utf-8",
    )
    counterfactual_manifest_path.write_text(
        json.dumps(
            build_artifact_manifest(
                {"counterfactual_verification": counterfactual_report_path},
                root=counterfactual_dir,
                metadata={"workflow": "counterfactual_verification_eval"},
            )
        ),
        encoding="utf-8",
    )
    ArtifactRegistry.load_json(counterfactual_registry_path).record_report(
        name="counterfactual-verifier-audit",
        path=counterfactual_report_path,
        version="0.1",
        metadata={"artifact_manifest": str(counterfactual_manifest_path)},
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
        context_sensitivity_workflow={
            "report_path": "context-sensitivity/workflow.json",
            "manifest_path": "context-sensitivity/artifact-manifest.json",
            "registry": "context-sensitivity/registry.json",
            "record_key": "report:context-sensitivity-workflow:0.1",
            "status": "promote",
            "paired_logprob_record_count": 6,
            "enriched_record_count": 6,
            "enhanced_score_signal_count": 4,
            "max_flagged_rate": 0.25,
            "mean_flagged_rate": 0.125,
            "max_context_sensitivity_ratio": 1.35,
            "manifest_verified": True,
        },
        pathway_intervention_workflow={
            "report_path": "pathway-intervention/workflow.json",
            "manifest_path": "pathway-intervention/artifact-manifest.json",
            "registry": "pathway-intervention/registry.json",
            "record_key": "report:pathway-intervention-workflow:0.1",
            "status": "promote",
            "report_status": "complete",
            "release_ready": True,
            "model": "demo-model",
            "layer": -8,
            "intervention_layer": -8,
            "patch_layer": -8,
            "signals": ["pathway_disagreement", "truth_proj"],
            "activation_ablation_gate_status": "promote",
            "source_patch_gate_status": "promote",
            "best_signals": {
                "activation_ablation": "pathway_disagreement",
                "source_patch": "truth_proj",
            },
        },
        external_evidence_baseline_comparison={
            "report_path": "external-evidence/comparison.json",
            "registry": "external-evidence/registry.json",
            "record_key": "report:covered-facts-external-evidence-handoff:0.4",
            "status": "promote",
            "decision_status": "promote",
            "recommended_route": "structured_fact",
            "recommended_route_record": (
                "benchmark_manifest:structured-fact-canonical-route:0.1"
            ),
            "route_passed": True,
            "text_redline_passed": True,
            "text_redline_run_count": 1,
        },
        pre_generation_probe_comparison={
            "report_path": "pre-generation/comparison.json",
            "manifest_path": "pre-generation/artifact-manifest.json",
            "registry": "pre-generation/registry.json",
            "record_key": "report:pre-generation-probe-comparison:0.1",
            "status": "promote",
            "model_count": 2,
            "run_count": 2,
            "redline_passed": True,
            "redline_run_count": 2,
            "best_run": {
                "name": "qwen05",
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "recommended_layer": -12,
                "test_label_auroc": 0.74,
                "redline_best_signal": "answer_token_count",
                "redline_best_auroc": 0.61,
                "redline_margin": 0.13,
            },
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
        counterfactual_verification={
            "report_path": "counterfactual/counterfactual-verification.json",
            "manifest_path": "counterfactual/artifact-manifest.json",
            "registry": "counterfactual/registry.json",
            "record_key": "report:counterfactual-verifier-audit:0.1",
            "workflow": "counterfactual_verification_eval",
            "status": "promote",
            "record_count": 4,
            "pass_rate": 1.0,
            "false_invariance_rate": 0.0,
            "flip_success_count": 4,
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
    assert metadata_without_verification["context_sensitivity_workflow_report"] == str(
        context_sensitivity_report_path
    )
    assert metadata_without_verification["context_sensitivity_workflow_manifest"] == str(
        context_sensitivity_manifest_path
    )
    assert (
        metadata_without_verification[
            "context_sensitivity_workflow_manifest_verification"
        ]
        is None
    )
    assert metadata_without_verification["context_sensitivity_workflow_registry"] == str(
        context_sensitivity_registry_path
    )
    assert (
        metadata_without_verification["context_sensitivity_workflow_registry_key"]
        == "report:context-sensitivity-workflow:0.1"
    )
    assert (
        metadata_without_verification["context_sensitivity_workflow_registry_record"]
        is None
    )
    assert metadata_without_verification["context_sensitivity_workflow_status"] == (
        "promote"
    )
    assert (
        metadata_without_verification[
            "context_sensitivity_workflow_paired_logprob_record_count"
        ]
        == 6
    )
    assert metadata_without_verification["context_sensitivity_workflow_max_flagged_rate"] == (
        pytest.approx(0.25)
    )
    assert metadata_without_verification["pathway_intervention_workflow_report"] == str(
        pathway_report_path
    )
    assert metadata_without_verification["pathway_intervention_workflow_manifest"] == str(
        pathway_manifest_path
    )
    assert (
        metadata_without_verification[
            "pathway_intervention_workflow_manifest_verification"
        ]
        is None
    )
    assert metadata_without_verification["pathway_intervention_workflow_registry"] == str(
        pathway_registry_path
    )
    assert metadata_without_verification["pathway_intervention_workflow_registry_key"] == (
        "report:pathway-intervention-workflow:0.1"
    )
    assert metadata_without_verification["pathway_intervention_workflow_registry_record"] is None
    assert metadata_without_verification["pathway_intervention_workflow_release_ready"] is True
    assert metadata_without_verification["pathway_intervention_workflow_model"] == "demo-model"
    assert metadata_without_verification["pathway_intervention_workflow_layer"] == -8
    assert metadata_without_verification[
        "pathway_intervention_workflow_activation_ablation_gate"
    ] == "promote"
    assert (
        metadata_without_verification["pathway_intervention_workflow_source_patch_gate"]
        == "promote"
    )
    assert metadata_without_verification[
        "promotion_contract_pathway_intervention_workflow"
    ]["record_key"] == "report:pathway-intervention-workflow:0.1"
    assert metadata_without_verification[
        "promotion_contract_external_evidence_baseline_comparison"
    ]["record_key"] == "report:covered-facts-external-evidence-handoff:0.4"
    assert metadata_without_verification["external_evidence_baseline_comparison_report"] == (
        str(external_evidence_report_path)
    )
    assert metadata_without_verification[
        "external_evidence_baseline_comparison_registry"
    ] == str(external_evidence_registry_path)
    assert metadata_without_verification[
        "external_evidence_baseline_comparison_registry_key"
    ] == "report:covered-facts-external-evidence-handoff:0.4"
    assert (
        metadata_without_verification[
            "external_evidence_baseline_comparison_registry_record"
        ]
        is None
    )
    assert metadata_without_verification[
        "external_evidence_baseline_comparison_status"
    ] == "promote"
    assert metadata_without_verification[
        "external_evidence_baseline_comparison_route_passed"
    ] is True
    assert metadata_without_verification[
        "external_evidence_baseline_comparison_text_redline_run_count"
    ] == 1
    assert metadata_without_verification[
        "promotion_contract_pre_generation_probe_comparison"
    ]["record_key"] == "report:pre-generation-probe-comparison:0.1"
    assert metadata_without_verification["pre_generation_probe_comparison_report"] == str(
        pre_generation_report_path
    )
    assert metadata_without_verification["pre_generation_probe_comparison_manifest"] == str(
        pre_generation_manifest_path
    )
    assert (
        metadata_without_verification[
            "pre_generation_probe_comparison_manifest_verification"
        ]
        is None
    )
    assert metadata_without_verification[
        "pre_generation_probe_comparison_registry"
    ] == str(pre_generation_registry_path)
    assert metadata_without_verification[
        "pre_generation_probe_comparison_registry_key"
    ] == "report:pre-generation-probe-comparison:0.1"
    assert metadata_without_verification[
        "pre_generation_probe_comparison_registry_record"
    ] is None
    assert metadata_without_verification[
        "pre_generation_probe_comparison_redline_passed"
    ] is True
    assert metadata_without_verification[
        "pre_generation_probe_comparison_best_redline_margin"
    ] == pytest.approx(0.13)
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
    assert metadata_without_verification[
        "promotion_contract_counterfactual_verification"
    ]["record_key"] == "report:counterfactual-verifier-audit:0.1"
    assert metadata_without_verification["counterfactual_verification_report"] == str(
        counterfactual_report_path
    )
    assert metadata_without_verification["counterfactual_verification_manifest"] == str(
        counterfactual_manifest_path
    )
    assert (
        metadata_without_verification[
            "counterfactual_verification_manifest_verification"
        ]
        is None
    )
    assert metadata_without_verification[
        "counterfactual_verification_registry"
    ] == str(counterfactual_registry_path)
    assert metadata_without_verification[
        "counterfactual_verification_registry_key"
    ] == "report:counterfactual-verifier-audit:0.1"
    assert metadata_without_verification[
        "counterfactual_verification_registry_record"
    ] is None
    assert metadata_without_verification["counterfactual_verification_status"] == "promote"
    assert metadata_without_verification["counterfactual_verification_pass_rate"] == (
        pytest.approx(1.0)
    )
    assert metadata_without_verification[
        "counterfactual_verification_false_invariance_rate"
    ] == pytest.approx(0.0)

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

    metadata_with_context_sensitivity_verification = bundle.runtime_metadata(
        budget_enabled=True,
        verify_context_sensitivity_workflow_manifest=True,
        include_context_sensitivity_workflow_record=True,
    )
    assert (
        metadata_with_context_sensitivity_verification[
            "context_sensitivity_workflow_manifest_verification"
        ]["passed"]
        is True
    )
    assert (
        metadata_with_context_sensitivity_verification[
            "context_sensitivity_workflow_manifest_verification"
        ]["checked"]
        == 1
    )
    assert (
        metadata_with_context_sensitivity_verification[
            "context_sensitivity_workflow_registry_key"
        ]
        == "report:context-sensitivity-workflow:0.1"
    )
    assert metadata_with_context_sensitivity_verification[
        "context_sensitivity_workflow_registry_record"
    ]["metadata"] == {"artifact_manifest": str(context_sensitivity_manifest_path)}

    metadata_with_pathway_verification = bundle.runtime_metadata(
        budget_enabled=True,
        verify_pathway_intervention_workflow_manifest=True,
        include_pathway_intervention_workflow_record=True,
    )
    assert (
        metadata_with_pathway_verification[
            "pathway_intervention_workflow_manifest_verification"
        ]["passed"]
        is True
    )
    assert (
        metadata_with_pathway_verification[
            "pathway_intervention_workflow_manifest_verification"
        ]["checked"]
        == 1
    )
    assert (
        metadata_with_pathway_verification["pathway_intervention_workflow_registry_key"]
        == "report:pathway-intervention-workflow:0.1"
    )
    assert metadata_with_pathway_verification[
        "pathway_intervention_workflow_registry_record"
    ]["metadata"] == {"artifact_manifest": str(pathway_manifest_path)}

    metadata_with_external_evidence_record = bundle.runtime_metadata(
        budget_enabled=True,
        include_external_evidence_baseline_comparison_record=True,
    )
    assert (
        metadata_with_external_evidence_record[
            "external_evidence_baseline_comparison_registry_key"
        ]
        == "report:covered-facts-external-evidence-handoff:0.4"
    )
    assert metadata_with_external_evidence_record[
        "external_evidence_baseline_comparison_registry_record"
    ]["metadata"] == {"workflow": "external_evidence_baseline_comparison"}

    metadata_with_pre_generation_verification = bundle.runtime_metadata(
        budget_enabled=True,
        verify_pre_generation_probe_comparison_manifest=True,
        include_pre_generation_probe_comparison_record=True,
    )
    assert (
        metadata_with_pre_generation_verification[
            "pre_generation_probe_comparison_manifest_verification"
        ]["passed"]
        is True
    )
    assert (
        metadata_with_pre_generation_verification[
            "pre_generation_probe_comparison_manifest_verification"
        ]["checked"]
        == 1
    )
    assert (
        metadata_with_pre_generation_verification[
            "pre_generation_probe_comparison_registry_key"
        ]
        == "report:pre-generation-probe-comparison:0.1"
    )
    assert metadata_with_pre_generation_verification[
        "pre_generation_probe_comparison_registry_record"
    ]["metadata"] == {"artifact_manifest": str(pre_generation_manifest_path)}

    metadata_with_counterfactual_verification = bundle.runtime_metadata(
        budget_enabled=True,
        verify_counterfactual_verification_manifest=True,
        include_counterfactual_verification_record=True,
    )
    assert (
        metadata_with_counterfactual_verification[
            "counterfactual_verification_manifest_verification"
        ]["passed"]
        is True
    )
    assert (
        metadata_with_counterfactual_verification[
            "counterfactual_verification_manifest_verification"
        ]["checked"]
        == 1
    )
    assert (
        metadata_with_counterfactual_verification[
            "counterfactual_verification_registry_key"
        ]
        == "report:counterfactual-verifier-audit:0.1"
    )
    assert metadata_with_counterfactual_verification[
        "counterfactual_verification_registry_record"
    ]["metadata"] == {"artifact_manifest": str(counterfactual_manifest_path)}

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


def test_product_runtime_evidence_bundle_exposes_evidence_handoff_manifest(tmp_path):
    contract_path = tmp_path / "product-promotion-contract.json"
    handoff_path = tmp_path / "product-promotion-contract-evidence-handoff.json"
    audit_path = tmp_path / "product-promotion-contract-evidence-handoff-audit.json"
    handoff_manifest_path = tmp_path / "evidence-handoff-artifact-manifest.json"

    ProductPromotionContract(
        model_id="demo-model",
        source_status="promote",
        metadata={"recommended_runtime_seconds": 0.2},
    ).save_json(contract_path)
    handoff_path.write_text(
        json.dumps({
            "workflow": "product_promotion_contract",
            "source_contract": str(contract_path),
            "contract": {"workflow": "product_promotion_contract"},
            "audit": {"status": "promote"},
        }),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps({
            "workflow": "product_promotion_evidence_handoff_audit",
            "status": "promote",
            "summary": {
                "blocked_group_count": 0,
                "expected_metric_count": 38,
                "groups": {
                    "promotion": "promote",
                    "pre_generation": "promote",
                    "counterfactual": "promote",
                    "action_gate": "promote",
                    "triple_audit": "promote",
                    "covered_fact_property": "promote",
                },
                "missing_metric_count": 0,
                "present_metric_count": 38,
            },
        }),
        encoding="utf-8",
    )
    handoff_manifest = build_artifact_manifest(
        {
            "source_contract": contract_path,
            "product_promotion_contract_evidence_handoff": handoff_path,
            "product_promotion_contract_evidence_handoff_audit": audit_path,
        },
        root=tmp_path,
        metadata={
            "after_missing_metric_count": 0,
            "before_missing_metric_count": 24,
            "filled_groups": [
                "promotion",
                "pre_generation",
                "counterfactual",
                "action_gate",
                "triple_audit",
                "covered_fact_property",
            ],
            "resolved_missing_metric_count": 24,
            "status": "promote",
        },
    )
    handoff_manifest_path.write_text(json.dumps(handoff_manifest), encoding="utf-8")

    bundle = load_product_runtime_evidence_bundle(
        default_contract_paths=(contract_path,),
    )

    assert bundle is not None
    assert bundle.evidence_handoff_manifest_path == handoff_manifest_path
    metadata = bundle.runtime_metadata(budget_enabled=True)
    assert metadata["promotion_contract_evidence_handoff_manifest"] == str(
        handoff_manifest_path
    )
    assert metadata["promotion_contract_evidence_handoff_contract"] == str(handoff_path)
    assert metadata["promotion_contract_evidence_handoff_audit"] == str(audit_path)
    assert metadata["promotion_contract_evidence_handoff_status"] == "promote"
    assert metadata["promotion_contract_evidence_handoff_before_missing_metric_count"] == 24
    assert metadata["promotion_contract_evidence_handoff_after_missing_metric_count"] == 0
    assert metadata["promotion_contract_evidence_handoff_resolved_missing_metric_count"] == 24
    assert metadata["promotion_contract_evidence_handoff_present_metric_count"] == 38
    assert metadata["promotion_contract_evidence_handoff_missing_metric_count"] == 0
    assert metadata["promotion_contract_evidence_handoff_group_statuses"] == {
        "action_gate": "promote",
        "counterfactual": "promote",
        "covered_fact_property": "promote",
        "pre_generation": "promote",
        "promotion": "promote",
        "triple_audit": "promote",
    }
    assert metadata["promotion_contract_evidence_handoff_manifest_verification"] is None
    runtime_metrics = product_runtime_metrics({"metadata": metadata})
    assert runtime_metrics["promotion_contract_evidence_handoff_available"] is True
    assert runtime_metrics["promotion_contract_evidence_handoff_manifest"] == str(
        handoff_manifest_path
    )
    assert runtime_metrics["promotion_contract_evidence_handoff_present_metric_count"] == (
        pytest.approx(38.0)
    )
    assert runtime_metrics["promotion_contract_evidence_handoff_missing_metric_count"] == (
        pytest.approx(0.0)
    )
    assert runtime_metrics["promotion_contract_evidence_handoff_present_metric_rate"] == (
        pytest.approx(1.0)
    )
    assert runtime_metrics["promotion_contract_evidence_handoff_group_count"] == (
        pytest.approx(6.0)
    )
    assert runtime_metrics["promotion_contract_evidence_handoff_promoted_group_count"] == (
        pytest.approx(6.0)
    )
    assert runtime_metrics["promotion_contract_evidence_handoff_promoted_group_rate"] == (
        pytest.approx(1.0)
    )
    handoff_summary = runtime_metrics["promotion_contract_summary"]["evidence_handoff"]
    assert handoff_summary["status"] == "promote"
    assert handoff_summary["manifest_verified"] is None
    assert handoff_summary["group_statuses"]["promotion"] == "promote"

    verified_metadata = bundle.runtime_metadata(
        budget_enabled=True,
        verify_evidence_handoff_manifest=True,
    )
    assert (
        verified_metadata["promotion_contract_evidence_handoff_manifest_verification"][
            "passed"
        ]
        is True
    )
    assert (
        verified_metadata["promotion_contract_evidence_handoff_manifest_verification"][
            "checked"
        ]
        == 3
    )
    verified_metrics = product_runtime_metrics({"metadata": verified_metadata})
    assert verified_metrics[
        "promotion_contract_evidence_handoff_manifest_verified"
    ] is True
    assert verified_metrics["promotion_contract_summary"]["evidence_handoff"][
        "manifest_verified"
    ] is True


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
