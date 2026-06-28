"""Shared product-runtime drift evidence metric keys."""

from __future__ import annotations

PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_KEYS: tuple[str, ...] = (
    "promotion_contract_coverage_rate",
    "triple_extraction_fixture_matrix_coverage_rate",
    "triple_extraction_fixture_matrix_mean_best_f1",
    "triple_extraction_fixture_matrix_mean_f1_lift",
)
PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_KEYS: tuple[str, ...] = (
    "pre_generation_probe_comparison_coverage_rate",
    "pre_generation_probe_comparison_manifest_verified_rate",
    "pre_generation_probe_comparison_model_count",
    "pre_generation_probe_comparison_run_count",
    "pre_generation_probe_comparison_redline_pass_rate",
    "pre_generation_probe_comparison_best_test_label_auroc",
    "pre_generation_probe_comparison_best_redline_auroc",
    "pre_generation_probe_comparison_best_redline_margin",
)
PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_EVIDENCE_KEYS: tuple[str, ...] = (
    "counterfactual_verification_coverage_rate",
    "counterfactual_verification_manifest_verified_rate",
    "counterfactual_verification_record_count",
    "counterfactual_verification_pass_rate",
    "counterfactual_verification_false_invariance_rate",
    "counterfactual_verification_flip_success_count",
)
PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_KEYS: tuple[str, ...] = (
    "triple_claim_coverage_rate",
    "triple_audit_claim_coverage_rate",
    "triple_audit_pass_rate",
    "triple_slot_coverage_rate",
)
PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_KEYS: tuple[str, ...] = (
    "covered_fact_recommended_route_property_metric_count",
    "covered_fact_recommended_route_min_records",
    "covered_fact_recommended_route_min_source_documents",
    "covered_fact_recommended_route_min_decision_accuracy",
    "covered_fact_recommended_route_max_false_supported_rate",
    "covered_fact_recommended_route_min_false_refuted_rate",
)
PRODUCT_RUNTIME_DRIFT_ACTION_GATE_EVIDENCE_KEYS: tuple[str, ...] = (
    "product_trace_action_audit_error_rate",
    "product_trace_action_audit_missing_retrieval_action_rate",
    "product_trace_action_audit_missing_plan_retrieval_query_rate",
    "product_trace_action_audit_malformed_payload_rate",
    "product_trace_action_audit_unexpected_action_rate",
    "product_trace_action_audit_unknown_claim_id_rate",
    "product_trace_action_execution_alignment_failed_trace_rate",
    "product_trace_action_execution_missing_result_rate",
    "product_trace_action_execution_unexpected_result_rate",
    "product_trace_action_execution_request_id_mismatch_rate",
)

PRODUCT_RUNTIME_DRIFT_EVIDENCE_GROUPS: dict[str, tuple[str, ...]] = {
    "promotion": PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_KEYS,
    "pre_generation": PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_KEYS,
    "counterfactual": PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_EVIDENCE_KEYS,
    "triple_audit": PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_KEYS,
    "covered_fact_property": PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_KEYS,
    "action_gate": PRODUCT_RUNTIME_DRIFT_ACTION_GATE_EVIDENCE_KEYS,
}

PRODUCT_RUNTIME_DRIFT_EVIDENCE_KEYS: tuple[str, ...] = tuple(
    key
    for group_keys in PRODUCT_RUNTIME_DRIFT_EVIDENCE_GROUPS.values()
    for key in group_keys
)
