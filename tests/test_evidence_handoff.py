"""Tests for promotion-contract evidence handoff audits."""

import json

from benchmarks.audit_product_promotion_contract_evidence import (
    build_product_promotion_evidence_audit,
)
from benchmarks.export_product_promotion_contract_evidence_handoff import (
    export_product_promotion_contract_evidence_handoff,
)
from benchmarks.export_product_promotion_contract_evidence_handoff import (
    main as export_product_promotion_contract_evidence_handoff_main,
)
from eigentruth.control import (
    audit_product_promotion_contract_evidence,
    enrich_product_promotion_contract_evidence,
)
from eigentruth.json_utils import strict_json_dumps
from eigentruth.registry import ArtifactRegistry


def test_promotion_contract_evidence_audit_blocks_missing_frontier_groups():
    audit = audit_product_promotion_contract_evidence(
        {
            "workflow": "product_promotion_contract",
            "source_status": "promote",
            "model_id": "tiny",
        }
    )
    payload = audit.to_dict()

    assert payload["status"] == "blocked"
    assert payload["source_workflow"] == "product_promotion_contract"
    assert payload["summary"]["expected_metric_count"] == 65
    assert payload["summary"]["present_metric_count"] == 1
    assert payload["summary"]["missing_metric_count"] == 64
    assert payload["summary"]["groups"]["promotion"] == "blocked"
    assert "run_pre_generation_probe_comparison" in payload["recommended_action_ids"]
    assert "run_frontier_release_evidence_comparison" in payload["recommended_action_ids"]
    assert "promotion_contract.pre_generation_probe_comparison.coverage_rate" in (payload["missing_metrics"])
    assert "promotion_contract.frontier_release_evidence.decision_status" in (payload["missing_metrics"])
    assert "promotion_contract.frontier_release_evidence.multiple_testing_track_status" in (
        payload["missing_metrics"]
    )
    assert "promotion_contract.frontier_release_evidence.citation_batch_track_status" in (
        payload["missing_metrics"]
    )
    assert "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_track_status" in (
        payload["missing_metrics"]
    )
    strict_json_dumps(payload, sort_keys=True)


def test_promotion_contract_evidence_audit_passes_complete_synthetic_contract():
    audit = audit_product_promotion_contract_evidence(_complete_contract())
    payload = audit.to_dict()

    assert payload["status"] == "promote"
    assert payload["summary"]["expected_metric_count"] == 65
    assert payload["summary"]["present_metric_count"] == 65
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


def test_promotion_contract_evidence_audit_can_require_optional_runtime_groups():
    audit = audit_product_promotion_contract_evidence(
        {
            "workflow": "product_promotion_contract",
            "source_status": "promote",
            "model_id": "tiny",
        },
        required_groups=(
            "claim_risk_localization",
            "trajectory_audit",
            "evidence_handoff",
            "world_model",
        ),
    )
    payload = audit.to_dict()

    assert payload["status"] == "blocked"
    assert payload["required_groups"] == (
        "claim_risk_localization",
        "trajectory_audit",
        "evidence_handoff",
        "world_model",
    )
    assert payload["summary"]["expected_metric_count"] == 91
    assert payload["summary"]["missing_metric_count"] == 26
    assert payload["summary"]["groups"]["claim_risk_localization"] == "blocked"
    assert payload["summary"]["groups"]["trajectory_audit"] == "blocked"
    assert payload["summary"]["groups"]["evidence_handoff"] == "blocked"
    assert payload["summary"]["groups"]["world_model"] == "blocked"
    assert payload["recommended_action_ids"] == (
        "rerun_product_trace_claim_risk_localization_evidence",
        "rerun_product_trace_trajectory_audit_evidence",
        "refresh_product_promotion_evidence_handoff",
        "rerun_product_trace_world_model_evidence",
    )
    assert "claim_risk_localization.coverage_rate" in payload["missing_metrics"]
    assert "trajectory_audit.error_rate" in payload["missing_metrics"]
    assert "promotion_contract.evidence_handoff.promoted_group_rate.mean" in (
        payload["missing_metrics"]
    )
    assert "world_model.trace_gap_rate" in payload["missing_metrics"]


def test_promotion_contract_evidence_audit_passes_optional_runtime_groups():
    audit = audit_product_promotion_contract_evidence(
        _complete_contract_with_optional_runtime_groups(),
        required_groups=(
            "claim_factuality",
            "claim_risk_localization",
            "trajectory_audit",
            "evidence_handoff",
            "world_model",
            "context_sensitivity",
            "counterfactual_robustness",
        ),
    )
    payload = audit.to_dict()

    assert payload["status"] == "promote"
    assert payload["summary"]["expected_metric_count"] == 113
    assert payload["summary"]["missing_metric_count"] == 0
    assert payload["recommended_action_ids"] == ()
    assert payload["summary"]["groups"]["claim_factuality"] == "promote"
    assert payload["summary"]["groups"]["claim_risk_localization"] == "promote"
    assert payload["summary"]["groups"]["trajectory_audit"] == "promote"
    assert payload["summary"]["groups"]["evidence_handoff"] == "promote"
    assert payload["summary"]["groups"]["world_model"] == "promote"
    assert payload["summary"]["groups"]["context_sensitivity"] == "promote"
    assert payload["summary"]["groups"]["counterfactual_robustness"] == "promote"


def test_promotion_contract_evidence_audit_reads_exported_runtime_current_metadata():
    audit = audit_product_promotion_contract_evidence(
        {
            "workflow": "product_promotion_contract",
            "source_status": "promote",
            "model_id": "tiny",
            "metadata": {
                "product_runtime_drift_claim_risk_localization_coverage_rate_current": 1.0,
                "product_runtime_drift_claim_risk_localization_high_risk_claim_count_current": 1,
                "product_runtime_drift_claim_risk_localization_medium_or_high_risk_claim_count_current": 2,
                "product_runtime_drift_claim_risk_localization_entity_candidate_observation_count_current": 4,
                "product_runtime_drift_claim_risk_localization_unique_entity_candidate_count_current": 3,
                "product_runtime_drift_claim_risk_localization_high_risk_entity_candidate_count_current": 1,
                "product_runtime_drift_claim_risk_localization_medium_or_high_entity_candidate_count_current": 2,
            },
        },
        required_groups=("claim_risk_localization",),
    )
    payload = audit.to_dict()
    claim_risk = next(
        group
        for group in payload["groups"]
        if group["group"] == "claim_risk_localization"
    )

    assert payload["status"] == "promote"
    assert payload["missing_metrics"] == ()
    assert claim_risk["status"] == "promote"
    assert claim_risk["metrics"][0]["source_path"] == (
        "metadata",
        "product_runtime_drift_claim_risk_localization_coverage_rate_current",
    )


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


def test_product_promotion_evidence_handoff_export_fills_explicit_sources():
    result = enrich_product_promotion_contract_evidence(
        {
            "workflow": "product_promotion_contract",
            "source_status": "promote",
            "model_id": "tiny",
        },
        pre_generation_probe_comparison=_pre_generation_comparison_report(),
        triple_extraction_fixture_matrix=_triple_matrix_report(),
        counterfactual_verification=_counterfactual_report(),
        product_trace_replay_workflow=_product_trace_replay_workflow(),
        frontier_release_evidence=_frontier_release_evidence_report(),
        runtime_baseline=_runtime_baseline_with_triple_audit(),
        covered_fact_property_metrics=_covered_fact_property_rollup(),
    )
    payload = result.to_dict()
    contract = payload["contract"]

    assert payload["before_audit"]["summary"]["missing_metric_count"] == 64
    assert payload["after_audit"]["status"] == "promote"
    assert payload["after_audit"]["summary"]["present_metric_count"] == 65
    assert payload["summary"]["resolved_missing_metric_count"] == 64
    assert set(payload["filled_groups"]) == {
        "promotion",
        "pre_generation",
        "counterfactual",
        "triple_audit",
        "covered_fact_property",
        "action_gate",
        "frontier_release_evidence",
    }
    assert contract["pre_generation_probe_comparison"]["best_redline_margin"] == 0.08
    assert contract["metadata"]["pre_generation_probe_comparison_best_redline_auroc"] == 0.74
    assert contract["metadata"]["product_trace_action_audit_error_rate"] == 0.0
    assert contract["metadata"]["triple_slot_coverage_rate"] == 1.0
    assert contract["metadata"]["triple_audit_evidence_source"] == "runtime_baseline"
    assert (
        contract["metadata"]["triple_audit_evidence_workflow"]
        == "product_runtime_baseline"
    )
    assert contract["frontier_release_evidence"]["decision_status"] == "promote"
    assert contract["metadata"]["frontier_release_evidence_abstention_track_status"] == "promote"
    assert (
        contract["frontier_release_evidence"]["frontier_rerun_rollup_track_status"]
        == "not_required"
    )
    assert contract["metadata"]["frontier_release_evidence_frontier_rerun_rollup_report_count"] == 0
    assert contract["metadata"]["frontier_release_evidence_citation_batch_rollup_count"] == 0
    assert contract["metadata"]["evidence_handoff_coverage_rate"] == 1.0
    assert contract["metadata"]["evidence_handoff_status"] == "promote"
    assert contract["metadata"]["evidence_handoff_missing_metric_count"] == 0.0
    assert contract["metadata"]["evidence_handoff_present_metric_rate"] == 1.0
    assert contract["metadata"]["evidence_handoff_promoted_group_rate"] == 1.0


def test_product_promotion_evidence_handoff_accepts_triple_audit_enrichment_report():
    result = enrich_product_promotion_contract_evidence(
        {
            "workflow": "product_promotion_contract",
            "source_status": "promote",
            "model_id": "tiny",
        },
        triple_audit_enrichment=_triple_audit_enrichment_report(),
        triple_audit_enrichment_path="triple-audit.json",
        required_groups=("triple_audit",),
    )
    payload = result.to_dict()
    contract = payload["contract"]

    assert payload["after_audit"]["status"] == "promote"
    assert payload["filled_groups"] == ("triple_audit",)
    assert payload["metadata"]["sources"]["triple_audit_enrichment"] == "triple-audit.json"
    assert contract["metadata"]["triple_claim_coverage_rate"] == 1.0
    assert contract["metadata"]["triple_audit_claim_coverage_rate"] == 1.0
    assert contract["metadata"]["triple_audit_pass_rate"] == 1.0
    assert contract["metadata"]["triple_slot_coverage_rate"] == 1.0
    assert contract["metadata"]["triple_audit_evidence_source"] == (
        "triple_audit_enrichment"
    )
    assert contract["metadata"]["triple_audit_evidence_report"] == "triple-audit.json"
    assert contract["metadata"]["triple_audit_evidence_workflow"] == (
        "product_trace_triple_audit_enrichment"
    )
    assert contract["metadata"]["triple_audit_evidence_status"] == "promote"

    blocked = enrich_product_promotion_contract_evidence(
        {
            "workflow": "product_promotion_contract",
            "source_status": "promote",
            "model_id": "tiny",
        },
        triple_audit_enrichment={
            **_triple_audit_enrichment_report(),
            "status": "blocked",
        },
        required_groups=("triple_audit",),
    )
    blocked_payload = blocked.to_dict()
    assert blocked_payload["after_audit"]["status"] == "blocked"
    assert "triple_audit" not in blocked_payload["filled_groups"]


def test_product_promotion_evidence_handoff_prefers_complete_triple_audit_enrichment():
    result = enrich_product_promotion_contract_evidence(
        {
            "workflow": "product_promotion_contract",
            "source_status": "promote",
            "model_id": "tiny",
        },
        runtime_baseline={
            "workflow": "product_runtime_baseline",
            "summary": {
                "triple_coverage": {
                    "claim_triple_coverage_rate": 0.0,
                }
            },
        },
        runtime_baseline_path="runtime-baseline.json",
        triple_audit_enrichment=_triple_audit_enrichment_report(),
        triple_audit_enrichment_path="triple-audit.json",
        required_groups=("triple_audit",),
    )
    payload = result.to_dict()
    contract = payload["contract"]

    assert payload["after_audit"]["status"] == "promote"
    assert payload["filled_groups"] == ("triple_audit",)
    assert contract["metadata"]["triple_claim_coverage_rate"] == 1.0
    assert contract["metadata"]["triple_audit_claim_coverage_rate"] == 1.0
    assert contract["metadata"]["triple_audit_pass_rate"] == 1.0
    assert contract["metadata"]["triple_slot_coverage_rate"] == 1.0
    assert contract["metadata"]["triple_audit_evidence_source"] == (
        "triple_audit_enrichment"
    )
    assert contract["metadata"]["triple_audit_evidence_report"] == "triple-audit.json"


def test_product_promotion_evidence_handoff_accepts_claim_correction_workflow_report():
    result = enrich_product_promotion_contract_evidence(
        {
            "workflow": "product_promotion_contract",
            "source_status": "promote",
            "model_id": "tiny",
        },
        triple_audit_enrichment=_claim_correction_workflow_report(),
        triple_audit_enrichment_path="claim-correction-workflow.json",
        required_groups=("triple_audit",),
    )
    payload = result.to_dict()

    assert payload["after_audit"]["status"] == "promote"
    assert payload["filled_groups"] == ("triple_audit",)
    assert payload["contract"]["metadata"]["triple_audit_pass_rate"] == 1.0
    assert payload["contract"]["metadata"]["triple_audit_evidence_source"] == (
        "claim_correction_workflow"
    )
    assert payload["contract"]["metadata"]["triple_audit_evidence_report"] == (
        "claim-correction-workflow.json"
    )
    assert payload["contract"]["metadata"]["triple_audit_evidence_workflow"] == (
        "source_family_structured_qa_claim_correction_workflow"
    )
    assert payload["contract"]["metadata"]["triple_audit_evidence_status"] == "promote"


def test_product_promotion_evidence_handoff_rolls_up_covered_fact_route_summary():
    result = enrich_product_promotion_contract_evidence(
        {
            "workflow": "product_promotion_contract",
            "source_status": "promote",
            "model_id": "tiny",
        },
        covered_fact_property_metrics=_covered_fact_route_summary(),
    )
    payload = result.to_dict()
    contract = payload["contract"]
    groups = {group["group"]: group for group in payload["after_audit"]["groups"]}
    rollup = contract["metadata"]["recommended_route_covered_fact_property_metrics"]

    assert groups["covered_fact_property"]["status"] == "promote"
    assert rollup["property_metric_count"] == 2
    assert rollup["min_records"] == 4
    assert rollup["min_source_documents"] == 2
    assert rollup["min_decision_accuracy"] == 1.0
    assert rollup["max_false_supported_rate"] == 0.0
    assert rollup["min_false_refuted_rate"] == 1.0


def test_product_promotion_evidence_handoff_cli_helper_writes_and_registers(tmp_path):
    contract = tmp_path / "contract.json"
    pre_generation = tmp_path / "pre-generation-comparison.json"
    matrix = tmp_path / "triple-matrix.json"
    workflow = tmp_path / "product-trace-workflow.json"
    frontier_evidence = tmp_path / "frontier-release-evidence.json"
    triple_audit = tmp_path / "triple-audit-enrichment.json"
    output = tmp_path / "contract-enriched.json"
    audit = tmp_path / "contract-enriched-audit.json"
    manifest = tmp_path / "artifact-manifest.json"
    registry_path = tmp_path / "registry.json"
    contract.write_text(
        json.dumps(
            {
                "workflow": "product_promotion_contract",
                "source_status": "promote",
                "model_id": "tiny",
            }
        ),
        encoding="utf-8",
    )
    pre_generation.write_text(json.dumps(_pre_generation_comparison_report()), encoding="utf-8")
    matrix.write_text(json.dumps(_triple_matrix_report()), encoding="utf-8")
    workflow.write_text(json.dumps(_product_trace_replay_workflow()), encoding="utf-8")
    frontier_evidence.write_text(
        json.dumps(_frontier_release_evidence_report()),
        encoding="utf-8",
    )
    triple_audit.write_text(json.dumps(_triple_audit_enrichment_report()), encoding="utf-8")

    payload = export_product_promotion_contract_evidence_handoff(
        contract=contract,
        json_path=output,
        audit_json_path=audit,
        pre_generation_probe_comparison=pre_generation,
        triple_extraction_fixture_matrix=matrix,
        product_trace_replay_workflow=workflow,
        frontier_release_evidence=frontier_evidence,
        triple_audit_enrichment=triple_audit,
        artifact_manifest_path=manifest,
        registry_path=registry_path,
        name="contract-enriched",
        version="0.2",
        metadata={"scope": "unit-test"},
    )

    assert output.exists()
    assert audit.exists()
    assert manifest.exists()
    assert payload["summary"]["before_missing_metric_count"] == 64
    assert payload["summary"]["after_missing_metric_count"] == 12
    assert payload["summary"]["resolved_missing_metric_count"] == 52
    output_payload = json.loads(output.read_text(encoding="utf-8"))
    assert output_payload["metadata"]["evidence_handoff_manifest"] == str(manifest)
    assert output_payload["metadata"]["evidence_handoff_contract"] == str(output)
    assert output_payload["metadata"]["evidence_handoff_audit"] == str(audit)
    assert output_payload["metadata"]["evidence_handoff_manifest_verified"] is True
    assert output_payload["metadata"]["evidence_handoff_manifest_verified_rate"] == 1.0
    assert output_payload["metadata"]["evidence_handoff_missing_metric_count"] == 12.0
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["summary"]["missing_count"] == 0
    assert set(manifest_payload["artifacts"]) == {
        "source_contract",
        "product_promotion_contract_evidence_handoff",
        "product_promotion_contract_evidence_handoff_audit",
        "pre_generation_probe_comparison",
        "triple_extraction_fixture_matrix",
        "product_trace_replay_workflow",
        "frontier_release_evidence",
        "triple_audit_enrichment",
    }
    registry = ArtifactRegistry.load_json(registry_path)
    contract_record = registry.get("product_promotion_contract:contract-enriched:0.2")
    audit_record = registry.get("product_promotion_evidence_audit:contract-enriched-audit:0.2")
    assert contract_record.metadata["resolved_missing_metric_count"] == 52
    assert contract_record.metadata["artifact_manifest"] == str(manifest)
    assert audit_record.metadata["missing_metric_count"] == 12
    assert audit_record.metadata["scope"] == "unit-test"


def test_product_promotion_evidence_handoff_export_accepts_required_runtime_groups(
    tmp_path,
):
    contract = tmp_path / "contract.json"
    output = tmp_path / "contract-enriched.json"
    audit = tmp_path / "contract-enriched-audit.json"
    manifest = tmp_path / "artifact-manifest.json"
    registry_path = tmp_path / "registry.json"
    contract.write_text(
        json.dumps(_complete_contract_with_optional_runtime_groups()),
        encoding="utf-8",
    )

    payload = export_product_promotion_contract_evidence_handoff(
        contract=contract,
        json_path=output,
        audit_json_path=audit,
        artifact_manifest_path=manifest,
        registry_path=registry_path,
        name="strict-contract-enriched",
        version="0.3",
        required_groups=(
            "claim_risk_localization",
            "trajectory_audit",
            "world_model",
        ),
    )
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    contract_record = registry.get(
        "product_promotion_contract:strict-contract-enriched:0.3"
    )
    audit_record = registry.get(
        "product_promotion_evidence_audit:strict-contract-enriched-audit:0.3"
    )

    assert payload["status"] == "promote"
    assert payload["after_audit"]["required_groups"] == (
        "claim_risk_localization",
        "trajectory_audit",
        "world_model",
    )
    assert payload["summary"]["after_missing_metric_count"] == 0
    assert audit_payload["summary"]["groups"]["claim_risk_localization"] == "promote"
    assert audit_payload["summary"]["groups"]["trajectory_audit"] == "promote"
    assert audit_payload["summary"]["groups"]["world_model"] == "promote"
    assert manifest_payload["metadata"]["required_groups"] == [
        "claim_risk_localization",
        "trajectory_audit",
        "world_model",
    ]
    assert contract_record.metadata["required_groups"] == [
        "claim_risk_localization",
        "trajectory_audit",
        "world_model",
    ]
    assert audit_record.metadata["required_groups"] == [
        "claim_risk_localization",
        "trajectory_audit",
        "world_model",
    ]


def test_product_promotion_evidence_handoff_cli_parses_required_groups(tmp_path):
    contract = tmp_path / "contract.json"
    output = tmp_path / "contract-enriched.json"
    audit = tmp_path / "contract-enriched-audit.json"
    contract.write_text(
        json.dumps(_complete_contract_with_optional_runtime_groups()),
        encoding="utf-8",
    )

    export_product_promotion_contract_evidence_handoff_main(
        [
            "--contract",
            str(contract),
            "--json",
            str(output),
            "--audit-json",
            str(audit),
            "--required-groups",
            "claim_risk_localization,trajectory_audit,world_model",
        ]
    )
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))

    assert output.exists()
    assert audit_payload["status"] == "promote"
    assert audit_payload["required_groups"] == [
        "claim_risk_localization",
        "trajectory_audit",
        "world_model",
    ]
    assert audit_payload["summary"]["missing_metric_count"] == 0


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
        "frontier_release_evidence": {
            "report_path": "frontier-release-evidence.json",
            "manifest_path": "frontier-release-evidence-manifest.json",
            "source": "registry",
            "workflow": "frontier_release_evidence_comparison",
            "status": "promote",
            "report_status": "complete",
            "decision_status": "promote",
            "verifier_track_status": "promote",
            "abstention_track_status": "promote",
            "multiple_testing_track_status": "not_required",
            "citation_batch_track_status": "not_required",
            "frontier_rerun_rollup_track_status": "not_required",
            "base_verifier_track_status": "promote",
            "base_abstention_track_status": "promote",
            "base_detectability_track_status": "not_required",
            "base_multiple_testing_track_status": "not_required",
            "frontier_rerun_rollup_report_count": 0,
            "frontier_rerun_rollup_candidate_count": 0,
            "frontier_rerun_rollup_missing_report_count": 0,
            "frontier_rerun_rollup_invalid_report_count": 0,
            "frontier_rerun_rollup_blocked_candidate_count": 0,
            "frontier_rerun_rollup_promotion_ready_count": 0,
            "citation_batch_rollup_count": 0,
            "citation_batch_expected_batch_count": 0,
            "citation_batch_observed_batch_count": 0,
            "citation_batch_missing_expected_batch_count": 0,
            "citation_batch_duplicate_batch_count": 0,
            "citation_batch_unexpected_batch_count": 0,
            "run_names": ("verifier-stability", "abstention-stability"),
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


def _complete_contract_with_optional_runtime_groups():
    contract = _complete_contract()
    contract.update({
        "claim_factuality_probe_comparison": {
            "coverage_rate": 1.0,
            "manifest_verified_rate": 1.0,
            "model_count": 2,
            "run_count": 3,
            "redline_pass_rate": 1.0,
            "best_test_label_auroc": 0.82,
            "best_test_selective_accuracy": 0.91,
            "best_test_selective_coverage": 0.78,
            "best_redline_auroc": 0.74,
            "best_redline_margin": 0.08,
        },
        "claim_risk_localization": {
            "coverage_rate": 1.0,
            "high_risk_claim_count": 1,
            "medium_or_high_risk_claim_count": 2,
            "entity_candidate_observation_count": 4,
            "unique_entity_candidate_count": 3,
            "high_risk_entity_candidate_count": 1,
            "medium_or_high_entity_candidate_count": 2,
        },
        "trajectory_audit": {
            "failed_trace_rate": 0.0,
            "error_rate": 0.0,
            "factual_rate": 0.0,
            "referential_rate": 0.0,
            "logical_rate": 0.0,
            "procedural_rate": 0.0,
            "scope_rate": 0.0,
        },
        "evidence_handoff": {
            "coverage_rate": 1.0,
            "manifest_verified_rate": 1.0,
            "present_metric_rate": 1.0,
            "missing_metric_rate": 0.0,
            "missing_metric_count": 0,
            "blocked_group_count": 0,
            "promoted_group_rate": 1.0,
        },
        "world_model": {
            "participating_trace_rate": 1.0,
            "coverage_rate": 1.0,
            "conflict_rate": 0.0,
            "low_agreement_rate": 0.0,
            "trace_gap_rate": 0.0,
        },
        "context_sensitivity": {
            "participating_trace_rate": 1.0,
            "coverage_rate": 1.0,
            "flagged_result_rate": 0.0,
            "trace_gap_rate": 0.0,
            "max_flagged_rate": 0.0,
            "max_context_sensitivity_ratio": 1.0,
        },
        "counterfactual_robustness": {
            "participating_trace_rate": 1.0,
            "coverage_rate": 1.0,
            "pass_rate": 1.0,
            "flip_success_rate": 1.0,
            "false_invariance_rate": 0.0,
            "trace_gap_rate": 0.0,
        },
    })
    return contract


def _pre_generation_comparison_report():
    return {
        "workflow": "pre_generation_probe_workflow_comparison",
        "status": "ready",
        "artifact_manifest_summary": {"missing_count": 0},
        "paths": {"artifact_manifest": "pre-generation/artifact-manifest.json"},
        "promotion_gate": {
            "failures": [],
            "model_count": 2,
            "models": ["tiny-a", "tiny-b"],
            "redline_passed": True,
            "redline_run_count": 2,
        },
        "runs": [{"name": "tiny-a"}, {"name": "tiny-b"}],
        "leaderboard": [
            {
                "name": "tiny-a",
                "effective_model": "tiny-a",
                "recommended_layer": -4,
                "test_label_auroc": 0.82,
                "redline_best_signal": "answer_negation_flag",
                "redline_best_auroc": 0.74,
                "redline_margin": 0.08,
            }
        ],
    }


def _triple_matrix_report():
    return {
        "workflow": "triple_extraction_fixture_matrix",
        "status": "promote",
        "n_corpora": 2,
        "promoted_corpora": 2,
        "distinct_predicate_count": 6,
        "distinct_predicates": ["P36", "P37"],
        "mean_baseline_f1": 0.45,
        "mean_best_f1": 0.95,
        "mean_f1_lift": 0.5,
    }


def _counterfactual_report():
    return {
        "workflow": "counterfactual_verification_eval",
        "artifact_manifest_summary": {"missing_count": 0},
        "paths": {"artifact_manifest": "counterfactual/artifact-manifest.json"},
        "report": {
            "summary": {
                "record_count": 12,
                "pass_rate": 1.0,
                "false_invariance_rate": 0.0,
                "flip_success_count": 12,
            }
        },
    }


def _product_trace_replay_workflow():
    return {
        "workflow": "product_trace_replay_workflow",
        "status": "promote",
        "artifact_manifest_summary": {"missing_count": 0},
        "paths": {
            "artifact_manifest": "trace-workflow/artifact-manifest.json",
            "selector_replay_report": "trace-workflow/selector-replay.json",
        },
        "action_audit_gate": {
            "error_rate": 0.0,
            "missing_retrieval_action_rate": 0.0,
            "missing_plan_retrieval_query_rate": 0.0,
            "malformed_payload_rate": 0.0,
            "unexpected_action_rate": 0.0,
            "unknown_claim_id_rate": 0.0,
        },
        "action_execution_gate": {
            "alignment_failed_trace_rate": 0.0,
            "missing_result_rate": 0.0,
            "unexpected_result_rate": 0.0,
            "request_id_mismatch_rate": 0.0,
        },
    }


def _frontier_release_evidence_report():
    return {
        "workflow": "frontier_release_evidence_comparison",
        "status": "complete",
        "paths": {
            "report": "frontier-release/frontier-release-evidence.json",
            "artifact_manifest": "frontier-release/artifact-manifest.json",
        },
        "decision": {
            "status": "promote",
            "verifier_track_status": "promote",
            "abstention_track_status": "promote",
            "multiple_testing_track_status": "not_required",
            "citation_batch_track_status": "not_required",
            "frontier_rerun_rollup_track_status": "not_required",
            "base_verifier_track_status": "promote",
            "base_abstention_track_status": "promote",
            "base_detectability_track_status": "not_required",
            "base_multiple_testing_track_status": "not_required",
            "blocking_reasons": [],
        },
        "evidence_summary": {
            "run_names": ["verifier-stability", "abstention-stability"],
            "frontier_rerun_rollup_report_count": 0,
            "frontier_rerun_rollup_candidate_count": 0,
            "frontier_rerun_rollup_missing_report_count": 0,
            "frontier_rerun_rollup_invalid_report_count": 0,
            "frontier_rerun_rollup_blocked_candidate_count": 0,
            "frontier_rerun_rollup_promotion_ready_count": 0,
            "citation_batch_rollup_count": 0,
            "citation_batch_expected_batch_count": 0,
            "citation_batch_observed_batch_count": 0,
            "citation_batch_missing_expected_batch_count": 0,
            "citation_batch_duplicate_batch_count": 0,
            "citation_batch_unexpected_batch_count": 0,
        },
    }


def _runtime_baseline_with_triple_audit():
    return {
        "workflow": "product_runtime_baseline",
        "summary": {
            "triple_coverage": {
                "claim_triple_coverage_rate": 1.0,
                "audit_claim_coverage_rate": 1.0,
                "audit_pass_rate": 1.0,
                "slot_coverage_rate": 1.0,
            }
        },
    }


def _triple_audit_enrichment_report():
    return {
        "workflow": "product_trace_triple_audit_enrichment",
        "status": "promote",
        "summary": {
            "claim_triple_coverage_rate": 1.0,
            "audit_claim_coverage_rate": 1.0,
            "audit_pass_rate": 1.0,
            "slot_coverage_rate": 1.0,
        },
    }


def _claim_correction_workflow_report():
    return {
        "workflow": "source_family_structured_qa_claim_correction_workflow",
        "status": "promote",
        "summary": {
            "triple_audit_status": "promote",
            "triple_audit_claim_triple_coverage_rate": 1.0,
            "triple_audit_audit_claim_coverage_rate": 1.0,
            "triple_audit_audit_pass_rate": 1.0,
            "triple_audit_slot_coverage_rate": 1.0,
        },
    }


def _covered_fact_property_rollup():
    return {
        "property_metric_count": 3,
        "min_records": 9,
        "min_source_documents": 100,
        "min_decision_accuracy": 1.0,
        "max_false_supported_rate": 0.0,
        "min_false_refuted_rate": 1.0,
    }


def _covered_fact_route_summary():
    return {
        "workflow": "source_family_structured_qa_route_workflow",
        "status": "promote",
        "fact_group_metrics": {
            "wikidata:reference:p31": {
                "n_records": 8,
                "n_source_documents": 0,
                "decision_accuracy": 1.0,
                "false_supported_rate": 0.0,
                "false_refuted_rate": 1.0,
            },
            "worldbank:official_statistics:sp_pop_totl": {
                "n_records": 4,
                "n_source_documents": 0,
                "decision_accuracy": 1.0,
                "false_supported_rate": 0.0,
                "false_refuted_rate": 1.0,
            },
        },
        "score_dump_summary": {
            "by_fact_group": {
                "wikidata:reference:p31": {"n_source_documents": 4},
                "worldbank:official_statistics:sp_pop_totl": {"n_source_documents": 2},
            }
        },
    }
