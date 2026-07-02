"""Shared product-runtime drift evidence metric keys."""

from __future__ import annotations

PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_KEYS: tuple[str, ...] = (
    "promotion_contract_coverage_rate",
    "triple_extraction_fixture_matrix_coverage_rate",
    "triple_extraction_fixture_matrix_mean_best_f1",
    "triple_extraction_fixture_matrix_mean_f1_lift",
)
PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_RISK_EVIDENCE_KEYS: tuple[str, ...] = (
    "pre_generation_risk_coverage_rate",
    "pre_generation_learned_risk_coverage_rate",
    "pre_generation_audit_profile_rate",
    "pre_generation_learned_risk_routed_rate",
    "pre_generation_learned_risk_probability_mean",
)
PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_PROBE_EVIDENCE_KEYS: tuple[str, ...] = (
    "pre_generation_probe_comparison_coverage_rate",
    "pre_generation_probe_comparison_manifest_verified_rate",
    "pre_generation_probe_comparison_model_count",
    "pre_generation_probe_comparison_run_count",
    "pre_generation_probe_comparison_redline_pass_rate",
    "pre_generation_probe_comparison_best_test_label_auroc",
    "pre_generation_probe_comparison_best_redline_auroc",
    "pre_generation_probe_comparison_best_redline_margin",
)
PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_KEYS: tuple[str, ...] = (
    PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_RISK_EVIDENCE_KEYS
    + PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_PROBE_EVIDENCE_KEYS
)
PRODUCT_RUNTIME_DRIFT_CLAIM_FACTUALITY_EVIDENCE_KEYS: tuple[str, ...] = (
    "claim_factuality_probe_comparison_coverage_rate",
    "claim_factuality_probe_comparison_manifest_verified_rate",
    "claim_factuality_probe_comparison_model_count",
    "claim_factuality_probe_comparison_run_count",
    "claim_factuality_probe_comparison_redline_pass_rate",
    "claim_factuality_probe_comparison_best_test_label_auroc",
    "claim_factuality_probe_comparison_best_test_selective_accuracy",
    "claim_factuality_probe_comparison_best_test_selective_coverage",
    "claim_factuality_probe_comparison_best_redline_auroc",
    "claim_factuality_probe_comparison_best_redline_margin",
)
PRODUCT_RUNTIME_DRIFT_CLAIM_RISK_LOCALIZATION_EVIDENCE_KEYS: tuple[str, ...] = (
    "claim_risk_localization_coverage_rate",
    "claim_risk_localization_high_risk_claim_count",
    "claim_risk_localization_medium_or_high_risk_claim_count",
    "claim_risk_localization_entity_candidate_observation_count",
    "claim_risk_localization_unique_entity_candidate_count",
    "claim_risk_localization_high_risk_entity_candidate_count",
    "claim_risk_localization_medium_or_high_entity_candidate_count",
)
PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_EVIDENCE_KEYS: tuple[str, ...] = (
    "counterfactual_verification_coverage_rate",
    "counterfactual_verification_manifest_verified_rate",
    "counterfactual_verification_record_count",
    "counterfactual_verification_pass_rate",
    "counterfactual_verification_false_invariance_rate",
    "counterfactual_verification_flip_success_count",
)
PRODUCT_RUNTIME_DRIFT_FACT_SELFCHECK_GATE_EVIDENCE_KEYS: tuple[str, ...] = (
    "fact_selfcheck_gate_coverage_rate",
    "fact_selfcheck_gate_report_present_rate",
    "fact_selfcheck_gate_manifest_present_rate",
    "fact_selfcheck_gate_manifest_verified_rate",
    "fact_selfcheck_gate_passed_rate",
    "fact_selfcheck_gate_run_count",
    "fact_selfcheck_gate_failed_run_count",
    "fact_selfcheck_gate_min_executed_rate",
    "fact_selfcheck_gate_min_decided_rate",
    "fact_selfcheck_gate_max_not_applicable_rate",
    "fact_selfcheck_gate_min_claim_triples_per_record",
    "fact_selfcheck_gate_min_sample_triples_per_record",
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
PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_ACTION_GATE_EVIDENCE_KEYS: tuple[str, ...] = (
    "world_model_action_gate_coverage_rate",
    "world_model_action_gate_pass_rate",
    "world_model_action_gate_blocked_rate",
    "world_model_action_gate_side_effect_block_violation_rate",
    "world_model_action_gate_low_prediction_confidence_rate",
    "world_model_action_gate_low_agreement_rate",
    "world_model_action_gate_no_rule_matched_rate",
    "world_model_action_gate_postcondition_refuted_rate",
    "world_model_action_gate_postcondition_insufficient_evidence_rate",
    "world_model_action_gate_postcondition_error_rate",
)
PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_ROLLOUT_EVIDENCE_KEYS: tuple[str, ...] = (
    "world_model_rollout_coverage_rate",
    "world_model_rollout_sync_rate",
    "world_model_rollout_drift_rate",
    "world_model_rollout_trace_gap_rate",
    "world_model_rollout_path_mismatch_rate",
)
PRODUCT_RUNTIME_DRIFT_ACTION_RECEIPTS_EVIDENCE_KEYS: tuple[str, ...] = (
    "product_trace_action_receipts_coverage_rate",
    "product_trace_action_receipts_missing_receipt_rate",
    "product_trace_action_receipts_invalid_receipt_rate",
    "product_trace_action_receipts_fingerprint_mismatch_rate",
    "product_trace_action_receipts_unsigned_receipt_rate",
)
PRODUCT_RUNTIME_DRIFT_RECEIPT_CLAIM_SUPPORT_EVIDENCE_KEYS: tuple[str, ...] = (
    "product_trace_receipt_claim_support_reference_support_rate",
    "product_trace_receipt_claim_support_unsupported_reference_rate",
    "product_trace_receipt_claim_support_missing_reference_rate",
    "product_trace_receipt_claim_support_unreceipted_reference_rate",
    "product_trace_receipt_claim_support_failed_result_reference_rate",
    "product_trace_receipt_claim_support_fingerprint_mismatch_reference_rate",
    "product_trace_receipt_claim_support_unsigned_reference_rate",
)
PRODUCT_RUNTIME_DRIFT_TRAJECTORY_AUDIT_EVIDENCE_KEYS: tuple[str, ...] = (
    "product_trace_trajectory_audit_failed_trace_rate",
    "product_trace_trajectory_audit_error_rate",
    "product_trace_trajectory_audit_factual_rate",
    "product_trace_trajectory_audit_referential_rate",
    "product_trace_trajectory_audit_logical_rate",
    "product_trace_trajectory_audit_procedural_rate",
    "product_trace_trajectory_audit_scope_rate",
    "product_trace_trajectory_audit_cascade_rate",
)
PRODUCT_RUNTIME_DRIFT_PROVENANCE_EVIDENCE_KEYS: tuple[str, ...] = (
    "product_trace_provenance_coverage_rate",
    "product_trace_provenance_supported_claim_evidence_coverage",
    "product_trace_provenance_missing_reference_rate",
    "product_trace_provenance_unsupported_supported_claim_rate",
    "product_trace_provenance_error_rate",
    "product_trace_provenance_final_answer_evidence_reference_rate",
    "product_trace_evidence_graph_consistency_coverage_rate",
    "product_trace_evidence_graph_consistency_supported_claim_consistency_rate",
    "product_trace_evidence_graph_consistency_missing_number_rate",
    "product_trace_evidence_graph_consistency_cross_claim_hit_rate",
    "product_trace_evidence_graph_consistency_error_rate",
)
PRODUCT_RUNTIME_DRIFT_CITATION_INTEGRITY_EVIDENCE_KEYS: tuple[str, ...] = (
    "product_trace_citation_integrity_participating_trace_rate",
    "product_trace_citation_integrity_coverage_rate",
    "product_trace_citation_integrity_mismatch_rate",
    "product_trace_citation_integrity_unresolved_rate",
    "product_trace_citation_integrity_issue_rate",
    "product_trace_citation_integrity_trace_gap_rate",
)
PRODUCT_RUNTIME_DRIFT_EVIDENCE_QUALITY_EVIDENCE_KEYS: tuple[str, ...] = (
    "product_trace_evidence_quality_trace_coverage_rate",
    "product_trace_evidence_quality_coverage_rate",
    "product_trace_evidence_quality_pass_rate",
    "product_trace_evidence_quality_failure_rate",
    "product_trace_evidence_quality_failed_result_rate",
    "product_trace_evidence_quality_stale_evidence_rate",
    "product_trace_evidence_quality_untrusted_source_rate",
    "product_trace_evidence_quality_missing_source_rate",
    "product_trace_evidence_quality_missing_timestamp_rate",
)
PRODUCT_RUNTIME_DRIFT_METACOGNITION_EVIDENCE_KEYS: tuple[str, ...] = (
    "product_trace_metacognition_trace_coverage_rate",
    "product_trace_metacognition_pass_rate",
    "product_trace_metacognition_overconfident_risk_rate",
    "product_trace_metacognition_miscalibration_score_mean",
)
PRODUCT_RUNTIME_DRIFT_EVIDENCE_HANDOFF_EVIDENCE_KEYS: tuple[str, ...] = (
    "evidence_handoff_coverage_rate",
    "evidence_handoff_manifest_verified_rate",
    "evidence_handoff_present_metric_rate",
    "evidence_handoff_missing_metric_rate",
    "evidence_handoff_missing_metric_count",
    "evidence_handoff_blocked_group_count",
    "evidence_handoff_promoted_group_rate",
)
PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_EVIDENCE_KEYS: tuple[str, ...] = (
    "world_model_participating_trace_rate",
    "world_model_coverage_rate",
    "world_model_conflict_rate",
    "world_model_low_agreement_rate",
    "world_model_trace_gap_rate",
)
PRODUCT_RUNTIME_DRIFT_CONTEXT_SENSITIVITY_EVIDENCE_KEYS: tuple[str, ...] = (
    "context_sensitivity_participating_trace_rate",
    "context_sensitivity_coverage_rate",
    "context_sensitivity_flagged_result_rate",
    "context_sensitivity_trace_gap_rate",
    "context_sensitivity_max_flagged_rate",
    "context_sensitivity_max_context_sensitivity_ratio",
)
PRODUCT_RUNTIME_DRIFT_EVIDENCE_ALIGNMENT_EVIDENCE_KEYS: tuple[str, ...] = (
    "evidence_alignment_participating_trace_rate",
    "evidence_alignment_coverage_rate",
    "evidence_alignment_alignment_rate",
    "evidence_alignment_misalignment_rate",
    "evidence_alignment_insufficient_evidence_rate",
    "evidence_alignment_citation_reference_coverage_rate",
    "evidence_alignment_issue_rate",
    "evidence_alignment_trace_gap_rate",
)
PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_ROBUSTNESS_EVIDENCE_KEYS: tuple[str, ...] = (
    "counterfactual_robustness_participating_trace_rate",
    "counterfactual_robustness_coverage_rate",
    "counterfactual_robustness_pass_rate",
    "counterfactual_robustness_flip_success_rate",
    "counterfactual_robustness_false_invariance_rate",
    "counterfactual_robustness_trace_gap_rate",
)
PRODUCT_RUNTIME_DRIFT_FRONTIER_RELEASE_EVIDENCE_KEYS: tuple[str, ...] = (
    "frontier_release_evidence_coverage_rate",
    "frontier_release_evidence_report_present_rate",
    "frontier_release_evidence_manifest_present_rate",
    "frontier_release_evidence_status_promote_rate",
    "frontier_release_evidence_decision_promote_rate",
    "frontier_release_evidence_verifier_track_promote_rate",
    "frontier_release_evidence_abstention_track_promote_rate",
    "frontier_release_evidence_citation_batch_track_promote_rate",
    "frontier_release_evidence_frontier_rerun_rollup_track_promote_rate",
    "frontier_release_evidence_run_count",
    "frontier_release_evidence_frontier_rerun_rollup_report_count",
    "frontier_release_evidence_frontier_rerun_rollup_candidate_count",
    "frontier_release_evidence_frontier_rerun_rollup_missing_report_count",
    "frontier_release_evidence_frontier_rerun_rollup_invalid_report_count",
    "frontier_release_evidence_frontier_rerun_rollup_blocked_candidate_count",
    "frontier_release_evidence_frontier_rerun_rollup_promotion_ready_count",
    "frontier_release_evidence_citation_batch_rollup_count",
    "frontier_release_evidence_citation_batch_expected_batch_count",
    "frontier_release_evidence_citation_batch_observed_batch_count",
    "frontier_release_evidence_citation_batch_missing_expected_batch_count",
    "frontier_release_evidence_citation_batch_duplicate_batch_count",
    "frontier_release_evidence_citation_batch_unexpected_batch_count",
)

PRODUCT_RUNTIME_DRIFT_EVIDENCE_GROUPS: dict[str, tuple[str, ...]] = {
    "promotion": PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_KEYS,
    "pre_generation": PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_KEYS,
    "claim_factuality": PRODUCT_RUNTIME_DRIFT_CLAIM_FACTUALITY_EVIDENCE_KEYS,
    "claim_risk_localization": (
        PRODUCT_RUNTIME_DRIFT_CLAIM_RISK_LOCALIZATION_EVIDENCE_KEYS
    ),
    "counterfactual": PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_EVIDENCE_KEYS,
    "fact_selfcheck_gate": PRODUCT_RUNTIME_DRIFT_FACT_SELFCHECK_GATE_EVIDENCE_KEYS,
    "triple_audit": PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_KEYS,
    "covered_fact_property": PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_KEYS,
    "action_gate": PRODUCT_RUNTIME_DRIFT_ACTION_GATE_EVIDENCE_KEYS,
    "world_model_action_gate": PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_ACTION_GATE_EVIDENCE_KEYS,
    "world_model_rollout": PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_ROLLOUT_EVIDENCE_KEYS,
    "action_receipts": PRODUCT_RUNTIME_DRIFT_ACTION_RECEIPTS_EVIDENCE_KEYS,
    "receipt_claim_support": (
        PRODUCT_RUNTIME_DRIFT_RECEIPT_CLAIM_SUPPORT_EVIDENCE_KEYS
    ),
    "trajectory_audit": PRODUCT_RUNTIME_DRIFT_TRAJECTORY_AUDIT_EVIDENCE_KEYS,
    "provenance": PRODUCT_RUNTIME_DRIFT_PROVENANCE_EVIDENCE_KEYS,
    "citation_integrity": PRODUCT_RUNTIME_DRIFT_CITATION_INTEGRITY_EVIDENCE_KEYS,
    "evidence_quality": PRODUCT_RUNTIME_DRIFT_EVIDENCE_QUALITY_EVIDENCE_KEYS,
    "metacognition": PRODUCT_RUNTIME_DRIFT_METACOGNITION_EVIDENCE_KEYS,
    "evidence_handoff": PRODUCT_RUNTIME_DRIFT_EVIDENCE_HANDOFF_EVIDENCE_KEYS,
    "world_model": PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_EVIDENCE_KEYS,
    "context_sensitivity": PRODUCT_RUNTIME_DRIFT_CONTEXT_SENSITIVITY_EVIDENCE_KEYS,
    "evidence_alignment": PRODUCT_RUNTIME_DRIFT_EVIDENCE_ALIGNMENT_EVIDENCE_KEYS,
    "counterfactual_robustness": (
        PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_ROBUSTNESS_EVIDENCE_KEYS
    ),
    "frontier_release_evidence": PRODUCT_RUNTIME_DRIFT_FRONTIER_RELEASE_EVIDENCE_KEYS,
}

PRODUCT_RUNTIME_DRIFT_EVIDENCE_KEYS: tuple[str, ...] = tuple(
    key
    for group_keys in PRODUCT_RUNTIME_DRIFT_EVIDENCE_GROUPS.values()
    for key in group_keys
)
