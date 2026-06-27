"""Action/tool-selection audit tests."""

import json

import pytest

from eigentruth.control import (
    ActionAuditPolicy,
    ActionAuditReport,
    ActionAuditSeverity,
    ActionRequest,
    ControlAction,
    ProductTrace,
    RiskDecision,
    RiskLevel,
    audit_action_requests,
    product_runtime_metrics,
)
from eigentruth.verify import Claim, ClaimVerificationPlan


def _plan_with_retrieval() -> ClaimVerificationPlan:
    return ClaimVerificationPlan(
        run_verifier=True,
        reason="claim requires retrieval",
        claims=(Claim("Paris is the capital of France.", claim_id="c1"),),
        verify_claim_ids=("c1",),
        retrieval_queries=({"claim_id": "c1", "query": "Paris capital France"},),
    )


def test_action_audit_flags_missing_plan_retrieval_action():
    decision = RiskDecision(
        action=ControlAction.ACCEPT,
        risk_level=RiskLevel.LOW,
        confidence=0.9,
        reason="low diagnostic risk",
    )

    report = audit_action_requests(
        (
            ActionRequest(
                action=ControlAction.ACCEPT,
                reason="low risk",
                payload={"mode": "pass_through"},
            ),
        ),
        decision=decision,
        verification_plan=_plan_with_retrieval(),
    )

    assert report.passed is False
    assert report.summary()["counts_by_code"]["missing_retrieval_action"] == 1
    assert report.required_actions == ("accept", "retrieve")
    json.dumps(report.to_dict())


def test_action_audit_accepts_executable_retrieval_payload():
    decision = RiskDecision(
        action=ControlAction.RETRIEVE,
        risk_level=RiskLevel.MEDIUM,
        confidence=0.8,
        reason="unsupported claim",
    )

    report = audit_action_requests(
        (
            ActionRequest(
                action=ControlAction.RETRIEVE,
                reason="retrieve evidence",
                payload={"retrieval_targets": ({"claim_id": "c1", "text": "Paris capital France"},)},
            ),
        ),
        decision=decision,
        verification_plan=_plan_with_retrieval(),
    )

    assert report.passed is True
    assert report.summary()["issue_count"] == 0


def test_action_audit_flags_retrieval_payload_that_misses_plan_query():
    trace = ProductTrace(
        verification_plan=_plan_with_retrieval(),
        risk_decision={"action": "retrieve"},
        actions=(
            ActionRequest(
                action=ControlAction.RETRIEVE,
                reason="retrieve evidence",
                payload={
                    "retrieval_targets": (
                        {"claim_id": "c1", "text": "Berlin capital Germany"},
                    ),
                },
            ),
        ),
    )
    report = audit_action_requests(
        (
            ActionRequest(
                action=ControlAction.RETRIEVE,
                reason="retrieve evidence",
                payload={
                    "retrieval_targets": (
                        {"claim_id": "c1", "text": "Berlin capital Germany"},
                    ),
                },
            ),
        ),
        decision=ControlAction.RETRIEVE,
        verification_plan=_plan_with_retrieval(),
    )

    summary = report.summary()
    metrics = product_runtime_metrics(trace)
    assert report.passed is False
    assert summary["counts_by_code"]["missing_plan_retrieval_query"] == 1
    assert metrics["action_audit_missing_plan_retrieval_query_count"] == 1.0
    issue = next(
        issue for issue in report.issues
        if issue.code == "missing_plan_retrieval_query"
    )
    assert issue.severity is ActionAuditSeverity.ERROR
    assert issue.claim_ids == ("c1",)
    assert issue.metadata["plan_retrieval_query_count"] == 1
    assert issue.metadata["covered_query_count"] == 0
    assert issue.metadata["missing_query_count"] == 1


def test_action_audit_flags_retrieval_queries_without_executable_target():
    report = audit_action_requests(
        (
            ActionRequest(
                action=ControlAction.RETRIEVE,
                reason="manual retrieval",
                payload={"retrieval_queries": ({"claim_id": "c1", "query": "Paris capital France"},)},
            ),
        ),
        decision=ControlAction.RETRIEVE,
        verification_plan=_plan_with_retrieval(),
    )

    summary = report.summary()
    assert report.passed is False
    assert summary["counts_by_code"]["malformed_retrieval_payload"] == 1
    assert summary["counts_by_code"]["retrieval_queries_not_executable"] == 1


def test_action_audit_flags_tool_parameter_problems_and_unknown_claim_ids():
    plan = _plan_with_retrieval()

    report = audit_action_requests(
        (
            ActionRequest(
                action=ControlAction.EXECUTE_TOOL,
                reason="call local tool",
                payload={"input": "not-a-json-object", "claim_ids": ("c404",)},
            ),
        ),
        decision=ControlAction.EXECUTE_TOOL,
        verification_plan=plan,
    )

    summary = report.summary()
    assert report.passed is False
    assert summary["counts_by_code"]["malformed_tool_payload"] == 1
    assert summary["counts_by_code"]["malformed_tool_arguments"] == 1
    assert summary["counts_by_code"]["unknown_claim_id"] == 1


def test_action_audit_policy_bool_roundtrip_and_rejects_ambiguous_values():
    policy = ActionAuditPolicy.from_dict({
        "require_decision_action": "false",
        "require_plan_retrieval": "on",
        "validate_claim_ids": "0",
    })

    assert policy.require_decision_action is False
    assert policy.require_plan_retrieval is True
    assert policy.validate_claim_ids is False
    assert ActionAuditPolicy.from_dict(policy.to_dict()) == policy
    with pytest.raises(ValueError, match="boolean"):
        ActionAuditPolicy.from_dict({"require_decision_action": "maybe"})


def test_action_audit_report_roundtrip():
    report = audit_action_requests(
        ("retrieve",),
        decision=ControlAction.RETRIEVE,
        verification_plan=_plan_with_retrieval(),
    )

    loaded = ActionAuditReport.from_dict(report.to_dict())

    assert loaded.summary()["counts_by_code"] == report.summary()["counts_by_code"]
    assert loaded.issues[0].severity is ActionAuditSeverity.ERROR


def test_product_trace_exposes_action_audit_summary_and_runtime_metrics():
    trace = ProductTrace(
        verification_plan=_plan_with_retrieval(),
        risk_decision=RiskDecision(
            action=ControlAction.ACCEPT,
            risk_level=RiskLevel.LOW,
            confidence=0.95,
            reason="low diagnostic risk",
        ),
        actions=(
            ActionRequest(
                action=ControlAction.ACCEPT,
                reason="low risk",
                payload={"mode": "pass_through"},
            ),
        ),
    )

    summary = trace.action_audit_summary()
    bounded = trace.to_bounded_dict()
    metrics = product_runtime_metrics(trace)
    bounded_metrics = product_runtime_metrics(bounded)

    assert summary["passed"] is False
    assert bounded["summaries"]["action_audit"]["counts_by_code"]["missing_retrieval_action"] == 1
    assert metrics["action_audit_available"] is True
    assert metrics["action_audit_passed"] is False
    assert metrics["action_audit_missing_retrieval_action_count"] == 1.0
    assert bounded_metrics["action_audit_source"] == "bounded_summary"
    assert bounded_metrics["action_audit_missing_retrieval_action_count"] == 1.0
