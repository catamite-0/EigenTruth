"""Tests for dependency-free claim verification planning."""

from __future__ import annotations

import json

import pytest

from eigentruth.verify import (
    Claim,
    ClaimVerificationPlan,
    ClaimVerificationPlanner,
    VerificationPlanCostEstimate,
    VerificationRouteHint,
    estimate_verification_plan_cost,
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
    assert route_by_claim["c1"] == ("retrieval", "triple_evidence", "groundedness")
    assert route_by_claim["c2"] == ("calculator", "retrieval", "triple_evidence", "groundedness")
    assert plan.retrieval_queries[0]["claim_id"] == "c1"
    assert plan.retrieval_queries[0]["query"].startswith("As of 2026")
    assert plan.calculation_checks[0]["claim_id"] == "c2"
    assert plan.calculation_checks[0]["expression"] == "2 + 2"
    assert plan.calculation_checks[0]["expected"] == 5.0
    assert plan.selected_claims() == tuple(plan.claims)
    cost = plan.cost_estimate()
    assert isinstance(cost, VerificationPlanCostEstimate)
    assert cost.route_counts == {"retrieval": 2, "triple_evidence": 2, "groundedness": 2, "calculator": 1}
    assert cost.tool_payload_counts == {
        "retrieval_queries": 2,
        "calculation_checks": 1,
        "state_checks": 0,
        "world_model_checks": 0,
    }
    assert cost.estimated_route_attempts == 7
    assert cost.estimated_tool_payloads == 3
    assert cost.estimated_cost_units == pytest.approx(4.95)
    payload = plan.to_dict()
    assert payload["cost_estimate"]["estimated_cost_units"] == pytest.approx(4.95)
    assert estimate_verification_plan_cost(payload).estimated_cost_units == pytest.approx(4.95)
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
