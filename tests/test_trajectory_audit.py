"""Trace-level hallucination audit tests."""

import json

from eigentruth.control import (
    ActionExecutionStatus,
    ActionRequest,
    ActionResult,
    ControlAction,
    FinalAnswer,
    FinalAnswerStatus,
    ProductTrace,
    RiskDecision,
    RiskLevel,
    TrajectoryAuditReport,
    TrajectoryHallucinationType,
    audit_product_trace_trajectory,
    product_runtime_metrics,
)
from eigentruth.verify import Claim, ClaimVerificationPlan, VerificationResult, VerificationStatus


def test_trajectory_audit_flags_refuted_claim_that_is_accepted():
    trace = ProductTrace(
        request_id="trace-refuted",
        claims=(Claim("Mars is the Sun.", claim_id="c1"),),
        verification_results=(
            VerificationResult(
                status=VerificationStatus.REFUTED,
                confidence=0.98,
                evidence=("Mars is a planet.",),
                metadata={"claim_id": "c1"},
            ),
        ),
        risk_decision=RiskDecision(
            action=ControlAction.ACCEPT,
            risk_level=RiskLevel.LOW,
            confidence=0.91,
            reason="incorrectly accepted",
        ),
        actions=(
            ActionRequest(
                action=ControlAction.ACCEPT,
                reason="pass through",
                request_id="accept-1",
            ),
        ),
        action_results=(
            ActionResult(
                action=ControlAction.ACCEPT,
                status=ActionExecutionStatus.DRY_RUN,
                request_id="accept-1",
            ),
        ),
        final_answer=FinalAnswer(
            status=FinalAnswerStatus.ANSWERED,
            text="Mars is the Sun.",
            answerable=True,
            action=ControlAction.ACCEPT,
            risk_level=RiskLevel.LOW,
            confidence=0.91,
            reason="accepted",
        ),
    )

    report = audit_product_trace_trajectory(trace)
    summary = report.summary()
    bounded = trace.to_bounded_dict()
    metrics = product_runtime_metrics(trace)
    bounded_metrics = product_runtime_metrics(bounded)
    loaded = TrajectoryAuditReport.from_dict(report.to_dict())

    assert report.passed is False
    assert summary["counts_by_type"][TrajectoryHallucinationType.FACTUAL.value] == 1
    assert summary["counts_by_type"][TrajectoryHallucinationType.LOGICAL.value] == 1
    assert summary["counts_by_code"]["accepted_refuted_claim"] == 1
    assert bounded["summaries"]["trajectory_audit"]["counts_by_code"]["accepted_refuted_claim"] == 1
    assert metrics["trajectory_audit_passed"] is False
    assert metrics["trajectory_audit_factual_count"] == 1.0
    assert bounded_metrics["trajectory_audit_source"] == "bounded_summary"
    assert loaded.summary()["counts_by_code"] == summary["counts_by_code"]
    json.dumps(report.to_dict())


def test_trajectory_audit_clean_supported_accept_path_passes():
    trace = ProductTrace(
        request_id="trace-clean",
        claims=(Claim("Paris is the capital of France.", claim_id="c1"),),
        verification_results=(
            VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=0.99,
                evidence=("France government page",),
                metadata={"claim_id": "c1"},
            ),
        ),
        risk_decision=RiskDecision(
            action=ControlAction.ACCEPT,
            risk_level=RiskLevel.LOW,
            confidence=0.97,
            reason="supported",
        ),
        actions=(
            ActionRequest(
                action=ControlAction.ACCEPT,
                reason="pass through",
                request_id="accept-1",
            ),
        ),
        action_results=(
            ActionResult(
                action=ControlAction.ACCEPT,
                status=ActionExecutionStatus.DRY_RUN,
                request_id="accept-1",
            ),
        ),
        final_answer=FinalAnswer(
            status=FinalAnswerStatus.ANSWERED,
            text="Paris is the capital of France.",
            answerable=True,
            action=ControlAction.ACCEPT,
            risk_level=RiskLevel.LOW,
            confidence=0.97,
            reason="supported",
        ),
    )

    report = audit_product_trace_trajectory(trace)
    metrics = product_runtime_metrics(trace)

    assert report.passed is True
    assert report.summary()["issue_count"] == 0
    assert metrics["trajectory_audit_passed"] is True
    assert metrics["trajectory_audit_issue_count"] == 0.0


def test_trajectory_audit_maps_action_and_execution_failures_to_taxonomy():
    plan = ClaimVerificationPlan(
        run_verifier=True,
        reason="needs retrieval",
        claims=(Claim("Paris is the capital of France.", claim_id="c1"),),
        verify_claim_ids=("c1",),
        retrieval_queries=({"claim_id": "c1", "query": "Paris capital France"},),
    )
    trace = ProductTrace(
        request_id="trace-action-drift",
        verification_plan=plan,
        risk_decision={"action": "retrieve", "risk_level": "medium"},
        actions=(
            ActionRequest(
                action=ControlAction.ABSTAIN,
                reason="wrong action for retrieve decision",
                request_id="abstain-1",
            ),
        ),
        action_results=(
            ActionResult(
                action=ControlAction.RETRIEVE,
                status=ActionExecutionStatus.FAILED,
                error="retriever unavailable",
                request_id="retrieve-2",
            ),
        ),
    )

    report = audit_product_trace_trajectory(trace)
    summary = report.summary()

    assert report.passed is False
    assert summary["counts_by_type"][TrajectoryHallucinationType.PROCEDURAL.value] >= 2
    assert summary["counts_by_type"][TrajectoryHallucinationType.SCOPE.value] >= 1
    assert summary["counts_by_type"][TrajectoryHallucinationType.REFERENTIAL.value] >= 1
    assert summary["counts_by_code"]["missing_decision_action"] == 1
    assert summary["counts_by_code"]["action_result_request_id_mismatch"] == 1
