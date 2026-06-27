"""Tests for dependency-free claim verification planning."""

from __future__ import annotations

import json

import pytest

from eigentruth.verify import (
    Claim,
    ClaimVerificationPlan,
    ClaimVerificationPlanner,
    VerificationBudgetPolicy,
    VerificationEscalationPolicy,
    VerificationPlanCostEstimate,
    VerificationResult,
    VerificationRouteHint,
    VerificationStatus,
    budget_verification_plan,
    escalate_uncertain_verification_plan,
    estimate_verification_plan_cost,
    extract_claims,
)


def test_claim_verification_planner_extracts_and_routes_text_claims():
    planner = ClaimVerificationPlanner()

    plan = planner.plan("As of 2026, AlphaCorp has 10 offices [1]. 2 plus 2 is 5.")

    assert plan.run_verifier is True
    assert plan.verification_scope == "all"
    assert plan.verify_claim_ids == ("c1", "c2")
    assert plan.triggered_features["c1"] == ("has_number", "has_citation", "is_time_sensitive")
    assert plan.triggered_features["c2"] == ("has_number", "has_calculation")
    route_by_claim = {hint.claim_id: hint.routes for hint in plan.route_hints}
    assert route_by_claim["c1"] == ("citation", "retrieval", "triple_evidence", "groundedness")
    assert route_by_claim["c2"] == ("calculator", "retrieval", "triple_evidence", "groundedness")
    assert plan.retrieval_queries[0]["claim_id"] == "c1"
    assert plan.retrieval_queries[0]["query"].startswith("As of 2026")
    assert plan.citation_checks[0]["claim_id"] == "c1"
    assert plan.citation_checks[0]["references"][0]["citation_id"] == "1"
    assert plan.calculation_checks[0]["claim_id"] == "c2"
    assert plan.calculation_checks[0]["expression"] == "2 + 2"
    assert plan.calculation_checks[0]["expected"] == 5.0
    assert plan.selected_claims() == tuple(plan.claims)
    cost = plan.cost_estimate()
    assert isinstance(cost, VerificationPlanCostEstimate)
    assert cost.route_counts == {
        "citation": 1,
        "retrieval": 2,
        "triple_evidence": 2,
        "groundedness": 2,
        "calculator": 1,
    }
    assert cost.tool_payload_counts == {
        "retrieval_queries": 2,
        "citation_checks": 1,
        "calculation_checks": 1,
        "state_checks": 0,
        "world_model_checks": 0,
    }
    assert cost.estimated_route_attempts == 8
    assert cost.estimated_tool_payloads == 4
    assert cost.estimated_cost_units == pytest.approx(5.45)
    payload = plan.to_dict()
    assert payload["cost_estimate"]["estimated_cost_units"] == pytest.approx(5.45)
    assert estimate_verification_plan_cost(payload).estimated_cost_units == pytest.approx(5.45)
    json.dumps(payload)


def test_claim_verification_planner_preserves_existing_claims_and_bool_semantics():
    claims = (
        Claim(
            "Paris is the capital of France.",
            claim_id="safe",
            metadata={"features": {"has_number": "false"}, "requires_verification": "false"},
        ),
        Claim(
            "A cited claim needs review.",
            claim_id="explicit",
            metadata={"requires_verification": "true", "features": {"has_citation": "false"}},
        ),
        Claim(
            "Ambiguous flags fail closed.",
            claim_id="ambiguous",
            metadata={"requires_verification": "maybe"},
        ),
    )
    planner = ClaimVerificationPlanner(
        verify_all_by_default=False,
        verify_triggered_claims_only=True,
    )

    plan = planner.plan(claims)

    assert plan.run_verifier is True
    assert plan.verification_scope == "triggered"
    assert plan.verify_claim_ids == ("explicit", "ambiguous")
    assert plan.skipped_claim_ids == ("safe",)
    assert plan.selected_claims() == claims[1:]
    assert "safe" not in plan.triggered_claim_ids
    assert plan.triggered_metadata["explicit"] == ("requires_verification",)
    assert plan.triggered_metadata["ambiguous"] == ("requires_verification",)
    assert [claim.text for claim in plan.claims] == [claim.text for claim in claims]


def test_claim_verification_planner_emits_tool_payloads_from_metadata():
    claim = Claim(
        "Inventory should remain available.",
        claim_id="inventory",
        metadata={
            "state_check": {"path": "inventory.sku_123.available", "operator": ">", "value": 0},
            "world_model_check": {"transition": {"decrement": {"inventory.sku_123.available": 1}}},
            "retrieval_query": {"query": "inventory sku_123 availability", "source": "planner-test"},
            "route_hints": ("state", "world_model"),
        },
    )
    planner = ClaimVerificationPlanner(
        verify_all_by_default=False,
        verify_triggered_claims_only=True,
    )

    plan = planner.plan((claim,))

    assert plan.verify_claim_ids == ("inventory",)
    assert plan.triggered_metadata["inventory"] == (
        "state_check",
        "world_model_check",
        "retrieval_query",
        "route_hints",
    )
    assert plan.route_hints[0].routes == ("state", "world_model", "retrieval", "groundedness")
    assert plan.retrieval_queries == (
        {
            "query": "inventory sku_123 availability",
            "claim_id": "inventory",
            "metadata": {"source": "planner-test"},
        },
    )
    assert plan.state_checks[0]["path"] == "inventory.sku_123.available"
    assert plan.state_checks[0]["claim_id"] == "inventory"
    assert plan.world_model_checks[0]["claim_id"] == "inventory"
    json.dumps(plan.to_dict())


def test_claim_verification_planner_routes_metadata_triples_to_triple_evidence():
    claim = Claim(
        "Revenue grew to 10 million.",
        claim_id="rev",
        metadata={
            "requires_triple_audit": "true",
            "claim_triples": {
                "subject": "Revenue",
                "predicate": "grew_to",
                "object": "10 million",
            },
        },
    )
    planner = ClaimVerificationPlanner(
        verify_all_by_default=False,
        verify_triggered_claims_only=True,
        retrieval_feature_flags=(),
    )

    plan = planner.plan((claim,))

    assert plan.run_verifier is True
    assert plan.verify_claim_ids == ("rev",)
    assert plan.triggered_metadata["rev"] == ("requires_triple_audit", "claim_triples")
    assert plan.route_hints[0].routes == ("triple_evidence", "groundedness")
    assert plan.route_hints[0].reasons == (
        "metadata:requires_triple_audit",
        "metadata:claim_triples",
    )
    assert plan.cost_estimate().route_counts == {"triple_evidence": 1, "groundedness": 1}


def test_claim_extraction_can_attach_rule_based_fact_triples():
    claims = extract_claims(
        "Paris is the capital of France. AlphaCorp has 10 offices in Europe.",
        include_triples=True,
        require_triple_audit=True,
    )

    assert claims[0].metadata["requires_triple_audit"] is True
    assert claims[0].metadata["claim_triples"][0]["subject"] == "France"
    assert claims[0].metadata["claim_triples"][0]["predicate"] == "capital_of"
    assert claims[0].metadata["claim_triples"][0]["object"] == "Paris"
    assert claims[1].metadata["claim_triples"][0]["subject"] == "AlphaCorp"
    assert claims[1].metadata["claim_triples"][0]["object"] == "10 offices in Europe"


def test_claim_verification_planner_can_route_extracted_fact_triples():
    planner = ClaimVerificationPlanner(
        include_extracted_triples=True,
        verify_all_by_default=False,
        verify_triggered_claims_only=True,
        verify_claim_feature_flags=(),
        retrieval_feature_flags=(),
        triple_evidence_feature_flags=(),
    )

    plan = planner.plan("Paris is the capital of France.")

    assert plan.run_verifier is True
    assert plan.verify_claim_ids == ("c1",)
    assert plan.triggered_metadata["c1"] == ("claim_triples",)
    assert plan.route_hints[0].routes == ("triple_evidence", "groundedness")
    assert plan.route_hints[0].reasons == ("metadata:claim_triples",)
    assert plan.claims[0].metadata["claim_triples"][0]["predicate"] == "capital_of"


def test_verification_budget_policy_selects_high_value_claims_and_routes():
    claims = (
        Claim("A low-risk descriptive claim.", claim_id="plain"),
        Claim(
            "2 plus 2 is 5.",
            claim_id="calc",
            metadata={
                "calculation": {"expression": "2 + 2", "expected": 5},
                "features": {"has_number": True, "has_calculation": True},
            },
        ),
        Claim(
            "Shipping the order will reduce inventory.",
            claim_id="transition",
            metadata={
                "world_model_check": {"action": {"ship": "sku-1"}},
                "features": {"is_time_sensitive": True},
            },
        ),
    )
    planner = ClaimVerificationPlanner()
    original = planner.plan(claims)

    budgeted = budget_verification_plan(
        original,
        VerificationBudgetPolicy(
            max_verify_claims=2,
            max_route_attempts=3,
            max_tool_payloads=2,
        ),
    )

    assert budgeted.verification_scope == "budgeted"
    assert budgeted.verify_claim_ids == ("calc", "transition")
    assert budgeted.skipped_claim_ids == ("plain",)
    routes = {hint.claim_id: hint.routes for hint in budgeted.route_hints}
    assert routes == {
        "calc": ("calculator", "triple_evidence"),
        "transition": ("world_model",),
    }
    assert budgeted.calculation_checks[0]["claim_id"] == "calc"
    assert budgeted.world_model_checks[0]["claim_id"] == "transition"
    assert budgeted.retrieval_queries == ()
    assert budgeted.cost_estimate().route_counts == {
        "calculator": 1,
        "triple_evidence": 1,
        "world_model": 1,
    }
    assert budgeted.cost_estimate().tool_payload_counts["calculation_checks"] == 1
    assert budgeted.cost_estimate().tool_payload_counts["world_model_checks"] == 1
    assert budgeted.budget["claim_budget_exhausted"] is True
    assert budgeted.budget["route_budget_exhausted"] is True
    assert budgeted.budget["dropped_claim_ids"] == ("plain",)
    assert budgeted.budget["dropped_routes"]["calc"] == ("retrieval", "groundedness")
    assert budgeted.budget["dropped_routes"]["transition"] == (
        "triple_evidence",
        "retrieval",
        "groundedness",
    )
    assert (
        budgeted.budget["selected_cost_estimate"]["estimated_cost_units"]
        < budgeted.budget["original_cost_estimate"]["estimated_cost_units"]
    )
    json.dumps(budgeted.to_dict())


def test_claim_verification_planner_accepts_budget_policy_mapping():
    planner = ClaimVerificationPlanner()

    plan = planner.plan(
        "As of 2026, AlphaCorp has 10 offices [1]. 2 plus 2 is 5.",
        budget_policy={
            "max_verify_claims": 1,
            "max_route_attempts": 1,
            "route_priority": ("calculator", "citation", "retrieval", "groundedness"),
        },
    )

    assert plan.verification_scope == "budgeted"
    assert plan.verify_claim_ids == ("c2",)
    assert plan.route_hints[0].routes == ("calculator",)
    assert plan.budget["policy"]["max_verify_claims"] == 1
    assert plan.budget["selected_cost_estimate"]["estimated_route_attempts"] == 1


def test_verification_escalation_policy_selects_uncertain_claims_and_fallback_query():
    plan = ClaimVerificationPlan(
        run_verifier=True,
        reason="manual",
        verification_scope="all",
        claims=(
            Claim("Paris is the capital of France.", claim_id="c1"),
            Claim("2 plus 2 is 4.", claim_id="c2"),
        ),
        verify_claim_ids=("c1", "c2"),
        route_hints=(
            VerificationRouteHint("c1", ("groundedness",), ("cheap:first_pass",)),
            VerificationRouteHint("c2", ("groundedness",), ("cheap:first_pass",)),
        ),
    )

    escalated = escalate_uncertain_verification_plan(
        plan,
        (
            VerificationResult(
                VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.4,
                metadata={"claim_id": "c1"},
            ),
            VerificationResult(
                VerificationStatus.SUPPORTED,
                confidence=0.92,
                metadata={"claim_id": "c2"},
            ),
        ),
    )

    assert escalated.verification_scope == "budgeted"
    assert escalated.verify_claim_ids == ("c1",)
    assert escalated.skipped_claim_ids == ("c2",)
    assert escalated.route_hints[0].routes == ("retrieval",)
    assert escalated.route_hints[0].metadata["verification_escalation"]["uncertainty_reasons"] == (
        "status:insufficient_evidence",
        "confidence_below:0.65",
    )
    assert escalated.retrieval_queries == (
        {
            "query": "Paris is the capital of France.",
            "claim_id": "c1",
            "metadata": {"source": "uncertainty_escalation.fallback_retrieval"},
        },
    )
    summary = escalated.budget["uncertainty_escalation"]
    assert summary["uncertain_claim_ids"] == ("c1",)
    assert summary["selected_claim_ids"] == ("c1",)
    assert summary["preliminary_results"]["c1"]["status"] == "insufficient_evidence"
    json.dumps(escalated.to_dict())


def test_verification_escalation_policy_applies_claim_and_route_budgets():
    claims = (
        Claim("Claim one.", claim_id="c1"),
        Claim("Claim two.", claim_id="c2"),
        Claim("Claim three.", claim_id="c3"),
    )
    plan = ClaimVerificationPlan(
        run_verifier=True,
        reason="manual",
        verification_scope="all",
        claims=claims,
        verify_claim_ids=("c1", "c2", "c3"),
        route_hints=tuple(
            VerificationRouteHint(
                claim.claim_id or "",
                ("retrieval", "triple_evidence", "world_model", "groundedness"),
            )
            for claim in claims
        ),
        retrieval_queries=tuple(
            {
                "query": claim.text,
                "claim_id": claim.claim_id,
                "metadata": {"source": "test"},
            }
            for claim in claims
        ),
    )

    escalated = escalate_uncertain_verification_plan(
        plan,
        (
            {"claim_id": "c1", "status": "supported", "confidence": 0.55},
            {"claim_id": "c2", "status": "supported", "confidence": 0.20},
            {"claim_id": "c3", "status": "supported", "confidence": 0.40},
        ),
        VerificationEscalationPolicy(
            min_confidence=0.6,
            max_escalated_claims=2,
            max_route_attempts=2,
        ),
    )

    assert escalated.verify_claim_ids == ("c2", "c3")
    assert {hint.claim_id: hint.routes for hint in escalated.route_hints} == {
        "c2": ("retrieval",),
        "c3": ("retrieval",),
    }
    assert [query["claim_id"] for query in escalated.retrieval_queries] == ["c2", "c3"]
    assert escalated.budget["route_budget_exhausted"] is True
    summary = escalated.budget["uncertainty_escalation"]
    assert summary["dropped_claim_cap_ids"] == ("c1",)
    assert summary["uncertain_claim_ids"] == ("c2", "c3", "c1")
    assert summary["selected_claim_ids"] == ("c2", "c3")
    assert escalated.cost_estimate().route_counts == {"retrieval": 2}
    json.dumps(escalated.to_dict())


def test_claim_verification_plan_json_shape_matches_stage_decision_fields():
    plan = ClaimVerificationPlan(
        run_verifier=True,
        reason="manual",
        verification_scope="triggered",
        claims=(Claim("A claim.", claim_id="c1"),),
        verify_claim_ids=("c1",),
        triggered_claim_ids=("c1",),
        triggered_features={"c1": ("has_number",)},
        triggered_metadata={"c1": ("requires_verification",)},
        route_hints=(VerificationRouteHint("c1", ("groundedness",), ("default:groundedness",)),),
    )

    payload = plan.to_dict()

    assert payload["run_verifier"] is True
    assert payload["verification_scope"] == "triggered"
    assert payload["verify_claim_ids"] == ("c1",)
    assert payload["skipped_claim_ids"] == ()
    assert payload["triggered_claim_ids"] == ("c1",)
    assert payload["triggered_features"] == {"c1": ("has_number",)}
    assert payload["triggered_metadata"] == {"c1": ("requires_verification",)}
    assert payload["route_hints"][0]["routes"] == ("groundedness",)
    json.dumps(payload)


def test_claim_verification_planner_returns_none_scope_for_empty_inputs():
    planner = ClaimVerificationPlanner()

    plan = planner.plan("")

    assert plan.run_verifier is False
    assert plan.verification_scope == "none"
    assert plan.selected_claims() == ()
    assert plan.to_dict()["claims"] == ()


def test_claim_verification_planner_rejects_invalid_config():
    with pytest.raises(ValueError, match="min_chars"):
        ClaimVerificationPlanner(min_chars=0)
    planner = ClaimVerificationPlanner(
        verify_all_by_default="false",  # type: ignore[arg-type]
        verify_triggered_claims_only="true",  # type: ignore[arg-type]
    )

    assert planner.verify_all_by_default is False
    assert planner.verify_triggered_claims_only is True
    with pytest.raises(ValueError, match="boolean"):
        ClaimVerificationPlanner(verify_all_by_default="maybe")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_route_attempts"):
        VerificationBudgetPolicy(max_route_attempts=-1)
    with pytest.raises(ValueError, match="min_confidence"):
        VerificationEscalationPolicy(min_confidence=1.5)
    with pytest.raises(ValueError, match="unknown verification status"):
        VerificationEscalationPolicy(uncertain_statuses=("maybe",))
