"""Tests for promotion-contract evidence handoff audits."""

import json

from benchmarks.audit_product_promotion_contract_evidence import (
    build_product_promotion_evidence_audit,
)
from eigentruth.control import audit_product_promotion_contract_evidence
from eigentruth.json_utils import strict_json_dumps
from eigentruth.registry import ArtifactRegistry


def test_promotion_contract_evidence_audit_blocks_missing_frontier_groups():
    audit = audit_product_promotion_contract_evidence({
        "workflow": "product_promotion_contract",
        "source_status": "promote",
        "model_id": "tiny",
    })
    payload = audit.to_dict()

    assert payload["status"] == "blocked"
    assert payload["source_workflow"] == "product_promotion_contract"
    assert payload["summary"]["expected_metric_count"] == 38
    assert payload["summary"]["present_metric_count"] == 1
    assert payload["summary"]["missing_metric_count"] == 37
    assert payload["summary"]["groups"]["promotion"] == "blocked"
    assert "run_pre_generation_probe_comparison" in payload["recommended_action_ids"]
    assert "promotion_contract.pre_generation_probe_comparison.coverage_rate" in (
        payload["missing_metrics"]
    )
    strict_json_dumps(payload, sort_keys=True)


def test_promotion_contract_evidence_audit_passes_complete_synthetic_contract():
    audit = audit_product_promotion_contract_evidence(_complete_contract())
    payload = audit.to_dict()

    assert payload["status"] == "promote"
    assert payload["summary"]["expected_metric_count"] == 38
    assert payload["summary"]["present_metric_count"] == 38
    assert payload["summary"]["missing_metric_count"] == 0
    assert payload["recommended_action_ids"] == ()


def test_promotion_contract_evidence_audit_can_require_subset():
    audit = audit_product_promotion_contract_evidence(
        {
            "workflow": "product_promotion_contract",
            "source_status": "promote",
            "model_id": "tiny",
        },
        required_groups=("pre_generation",),
    )

    assert audit.status == "blocked"
    assert audit.required_groups == ("pre_generation",)
    assert audit.recommended_action_ids == ("run_pre_generation_probe_comparison",)
    assert len(audit.missing_metrics) == 8


def test_product_promotion_evidence_audit_cli_helper_writes_and_registers(tmp_path):
    contract = tmp_path / "contract.json"
    output = tmp_path / "handoff-audit.json"
    registry_path = tmp_path / "registry.json"
    contract.write_text(json.dumps(_complete_contract()), encoding="utf-8")

    payload = build_product_promotion_evidence_audit(
        contract=contract,
        json_path=output,
        registry_path=registry_path,
        name="contract-handoff",
        version="0.1",
        required_groups=("promotion", "action_gate"),
        metadata={"scope": "unit-test"},
    )

    assert output.exists()
    assert payload["status"] == "promote"
    assert payload["required_groups"] == ("promotion", "action_gate")
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get("product_promotion_evidence_audit:contract-handoff:0.1")
    assert record.path == str(output)
    assert record.metadata["status"] == "promote"
    assert record.metadata["missing_metric_count"] == 0
    assert record.metadata["scope"] == "unit-test"


def _complete_contract():
    return {
        "workflow": "product_promotion_contract",
        "source_status": "promote",
        "model_id": "tiny",
        "triple_extraction_fixture_matrix": {
            "source": "registry",
            "status": "promote",
            "manifest_verified": True,
            "mean_best_f1": 0.95,
            "mean_f1_lift": 0.35,
        },
        "pre_generation_probe_comparison": {
            "source": "registry",
            "status": "promote",
            "manifest_verified": True,
            "model_count": 2,
            "run_count": 3,
            "redline_passed": True,
            "best_test_label_auroc": 0.82,
            "best_redline_auroc": 0.74,
            "best_redline_margin": 0.08,
        },
        "counterfactual_verification": {
            "source": "registry",
            "status": "promote",
            "manifest_verified": True,
            "record_count": 12,
            "pass_rate": 1.0,
            "false_invariance_rate": 0.0,
            "flip_success_count": 12,
        },
        "metadata": {
            "triple_claim_coverage_rate": 1.0,
            "triple_audit_claim_coverage_rate": 1.0,
            "triple_audit_pass_rate": 1.0,
            "triple_slot_coverage_rate": 1.0,
            "recommended_route_covered_fact_property_metrics": {
                "property_metric_count": 3,
                "min_records": 9,
                "min_source_documents": 100,
                "min_decision_accuracy": 1.0,
                "max_false_supported_rate": 0.0,
                "min_false_refuted_rate": 1.0,
            },
            "product_trace_action_audit_error_rate": 0.0,
            "product_trace_action_audit_missing_retrieval_action_rate": 0.0,
            "product_trace_action_audit_missing_plan_retrieval_query_rate": 0.0,
            "product_trace_action_audit_malformed_payload_rate": 0.0,
            "product_trace_action_audit_unexpected_action_rate": 0.0,
            "product_trace_action_audit_unknown_claim_id_rate": 0.0,
            "product_trace_action_execution_alignment_failed_trace_rate": 0.0,
            "product_trace_action_execution_missing_result_rate": 0.0,
            "product_trace_action_execution_unexpected_result_rate": 0.0,
            "product_trace_action_execution_request_id_mismatch_rate": 0.0,
        },
    }
