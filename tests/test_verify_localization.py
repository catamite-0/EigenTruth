"""Tests for claim-level risk localization."""

from __future__ import annotations

import json

import pytest

from eigentruth.control import ProductTrace, product_runtime_metrics
from eigentruth.verify import (
    Claim,
    ClaimRiskLocalizationReport,
    ClaimRiskSpan,
    ClaimVerificationPlanner,
    InMemoryVerifier,
    VerificationBudgetPolicy,
    VerificationResult,
    VerificationStatus,
    extract_claims,
    localize_claim_risk_spans,
    normalize_claim_text,
)


def test_localize_claim_risk_spans_uses_verification_status_and_claim_spans():
    text = "Paris is the capital of France. 2 plus 2 is 5."
    claims = extract_claims(text)
    verifier = InMemoryVerifier(
        facts={
            normalize_claim_text("Paris is the capital of France"): VerificationStatus.SUPPORTED,
            normalize_claim_text("2 plus 2 is 5"): VerificationStatus.REFUTED,
        },
        evidence={
            normalize_claim_text("Paris is the capital of France"): ("atlas",),
            normalize_claim_text("2 plus 2 is 5"): ("calculator",),
        },
    )
    results = verifier.verify_many(claims)
    plan = ClaimVerificationPlanner().plan(claims)

    report = localize_claim_risk_spans(
        claims,
        verification_results=results,
        verification_plan=plan,
        source_text=text,
    )

    assert isinstance(report, ClaimRiskLocalizationReport)
    assert [span.claim_id for span in report.spans] == ["c1", "c2"]
    assert report.spans[0].risk_level == "low"
    assert report.spans[1].risk_level == "high"
    assert report.spans[1].status == "refuted"
    assert report.spans[1].span == claims[1].span
    assert "calculator" in report.spans[1].routes
    assert "verification_status:refuted" in report.spans[1].reasons
    summary = report.summary()
    assert summary["span_count"] == 2
    assert summary["localized_span_count"] == 2
    assert summary["high_risk_claim_ids"] == ("c2",)
    json.dumps(report.to_dict())


def test_localize_claim_risk_spans_marks_budget_dropped_claims_and_filters_level():
    claims = (
        Claim("A low-risk descriptive claim.", claim_id="plain", span=(0, 29)),
        Claim(
            "2 plus 2 is 5.",
            claim_id="calc",
            span=(30, 45),
            metadata={
                "calculation": {"expression": "2 + 2", "expected": 5},
                "features": {"has_number": True, "has_calculation": True},
            },
        ),
    )
    plan = ClaimVerificationPlanner().plan(
        claims,
        budget_policy=VerificationBudgetPolicy(max_verify_claims=1, max_route_attempts=1),
    )
    results = (
        VerificationResult(
            status=VerificationStatus.REFUTED,
            confidence=0.9,
            evidence=("calculator",),
            metadata={"claim_id": "calc"},
        ),
    )

    report = localize_claim_risk_spans(
        claims,
        verification_results=results,
        verification_plan=plan,
        min_risk_level="medium",
    )

    assert [span.claim_id for span in report.spans] == ["plain", "calc"]
    by_claim = {span.claim_id: span for span in report.spans}
    assert by_claim["plain"].risk_level == "medium"
    assert by_claim["plain"].metadata["budget_dropped_routes"] == ()
    assert "budget:dropped_claim" in by_claim["plain"].reasons
    assert by_claim["calc"].risk_level == "high"
    assert by_claim["calc"].routes == ("calculator",)


def test_product_trace_exposes_claim_risk_localization_summary_and_metrics():
    claims = extract_claims("Paris is the capital of France. 2 plus 2 is 5.")
    plan = ClaimVerificationPlanner().plan(claims)
    results = (
        VerificationResult(
            status=VerificationStatus.SUPPORTED,
            confidence=0.9,
            evidence=("atlas",),
            metadata={"claim_id": "c1"},
        ),
        VerificationResult(
            status=VerificationStatus.REFUTED,
            confidence=0.95,
            evidence=("calculator",),
            metadata={"claim_id": "c2"},
        ),
    )
    trace = ProductTrace(claims=claims, verification_plan=plan, verification_results=results)

    summary = trace.claim_risk_localization_summary()
    bounded = trace.to_bounded_dict(max_nested_items=3)
    metrics = product_runtime_metrics(trace)
    bounded_metrics = product_runtime_metrics(bounded)

    assert summary["high_risk_claim_ids"] == ("c2",)
    assert summary["counts_by_risk_level"]["high"] == 1
    assert bounded["summaries"]["claim_risk_localization"]["high_risk_claim_count"] == 1
    assert bounded["summaries"]["claim_risk_localization"]["top_risk_spans"][0]["claim_id"] == "c2"
    assert metrics["claim_risk_localization_available"] is True
    assert metrics["claim_risk_high_count"] == 1.0
    assert metrics["claim_risk_medium_or_high_count"] == 1.0
    assert bounded_metrics["claim_risk_localization_source"] == "bounded_summary"
    assert bounded_metrics["claim_risk_high_count"] == 1.0


def test_claim_risk_span_validation():
    with pytest.raises(ValueError, match="risk_level"):
        ClaimRiskSpan(claim_id="c1", text="x", span=(0, 1), risk_level="unknown", risk_score=0.1)
    with pytest.raises(ValueError, match="span end"):
        ClaimRiskSpan(claim_id="c1", text="x", span=(2, 1), risk_level="low", risk_score=0.1)
