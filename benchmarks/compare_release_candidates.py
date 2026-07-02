"""Compare registered readiness and route baselines as one release candidate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.compare_readiness_baselines import (  # noqa: E402
    compare_readiness_baselines,
    covariance_tradeoff_gate,
)
from benchmarks.compare_route_baselines import compare_route_baselines  # noqa: E402
from benchmarks.recommend_runtime_config import INSIDE_TRIGGER_BUDGET_POLICIES  # noqa: E402
from benchmarks.release_policy_profiles import (  # noqa: E402
    RELEASE_POLICY_PROFILE_NAMES,
    append_unique,
    apply_release_policy_profile_defaults,
    clean_optional_key,
)
from eigentruth.control import (  # noqa: E402
    RUNTIME_PROFILE_NAMES,
    ControlPolicyConfig,
    get_runtime_profile,
)
from eigentruth.registry import (  # noqa: E402
    ArtifactRegistry,
    ArtifactVerificationContext,
)

ADAPTER_FAMILY_PROFILES: Mapping[str, tuple[str, ...]] = {
    "strict_audit": ("structured_state", "state_transition", "triple_evidence"),
}
ADAPTER_FAMILY_PROFILE_NAMES = tuple(sorted(ADAPTER_FAMILY_PROFILES))
ADAPTER_FAMILY_PROFILES_REQUIRING_STATE_TRANSITION_WORLD_MODEL: frozenset[str] = frozenset({
    "strict_audit",
})
_PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("promotion_contract.coverage_rate", "promotion_contract_coverage_rate"),
    (
        "promotion_contract.triple_extraction_fixture_matrix.coverage_rate",
        "triple_extraction_fixture_matrix_coverage_rate",
    ),
    (
        "promotion_contract.triple_extraction_fixture_matrix.mean_best_f1.mean",
        "triple_extraction_fixture_matrix_mean_best_f1",
    ),
    (
        "promotion_contract.triple_extraction_fixture_matrix.mean_f1_lift.mean",
        "triple_extraction_fixture_matrix_mean_f1_lift",
    ),
)
_PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_RISK_EVIDENCE_FIELDS: tuple[
    tuple[str, str],
    ...
] = (
    ("pre_generation_risk.coverage_rate", "pre_generation_risk_coverage_rate"),
    (
        "pre_generation_risk.learned_risk_coverage_rate",
        "pre_generation_learned_risk_coverage_rate",
    ),
    (
        "pre_generation_risk.selected_profile.audit_rate",
        "pre_generation_audit_profile_rate",
    ),
    (
        "pre_generation_risk.learned_risk_routed_rate",
        "pre_generation_learned_risk_routed_rate",
    ),
    (
        "pre_generation_risk.learned_risk_probability.mean",
        "pre_generation_learned_risk_probability_mean",
    ),
)
_PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_PROBE_EVIDENCE_FIELDS: tuple[
    tuple[str, str],
    ...
] = (
    (
        "promotion_contract.pre_generation_probe_comparison.coverage_rate",
        "pre_generation_probe_comparison_coverage_rate",
    ),
    (
        "promotion_contract.pre_generation_probe_comparison.manifest_verified_rate",
        "pre_generation_probe_comparison_manifest_verified_rate",
    ),
    (
        "promotion_contract.pre_generation_probe_comparison.model_count.mean",
        "pre_generation_probe_comparison_model_count",
    ),
    (
        "promotion_contract.pre_generation_probe_comparison.run_count.mean",
        "pre_generation_probe_comparison_run_count",
    ),
    (
        "promotion_contract.pre_generation_probe_comparison.redline_pass_rate",
        "pre_generation_probe_comparison_redline_pass_rate",
    ),
    (
        "promotion_contract.pre_generation_probe_comparison.best_test_label_auroc.mean",
        "pre_generation_probe_comparison_best_test_label_auroc",
    ),
    (
        "promotion_contract.pre_generation_probe_comparison.best_redline_auroc.mean",
        "pre_generation_probe_comparison_best_redline_auroc",
    ),
    (
        "promotion_contract.pre_generation_probe_comparison.best_redline_margin.mean",
        "pre_generation_probe_comparison_best_redline_margin",
    ),
)
_PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    _PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_RISK_EVIDENCE_FIELDS
    + _PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_PROBE_EVIDENCE_FIELDS
)
_PRODUCT_RUNTIME_DRIFT_CLAIM_FACTUALITY_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    (
        "promotion_contract.claim_factuality_probe_comparison.coverage_rate",
        "claim_factuality_probe_comparison_coverage_rate",
    ),
    (
        "promotion_contract.claim_factuality_probe_comparison.manifest_verified_rate",
        "claim_factuality_probe_comparison_manifest_verified_rate",
    ),
    (
        "promotion_contract.claim_factuality_probe_comparison.model_count.mean",
        "claim_factuality_probe_comparison_model_count",
    ),
    (
        "promotion_contract.claim_factuality_probe_comparison.run_count.mean",
        "claim_factuality_probe_comparison_run_count",
    ),
    (
        "promotion_contract.claim_factuality_probe_comparison.dataset_count.mean",
        "claim_factuality_probe_comparison_dataset_count",
    ),
    (
        "promotion_contract.claim_factuality_probe_comparison.redline_pass_rate",
        "claim_factuality_probe_comparison_redline_pass_rate",
    ),
    (
        "promotion_contract.claim_factuality_probe_comparison.best_test_label_auroc.mean",
        "claim_factuality_probe_comparison_best_test_label_auroc",
    ),
    (
        "promotion_contract.claim_factuality_probe_comparison.best_test_selective_accuracy.mean",
        "claim_factuality_probe_comparison_best_test_selective_accuracy",
    ),
    (
        "promotion_contract.claim_factuality_probe_comparison.best_test_selective_coverage.mean",
        "claim_factuality_probe_comparison_best_test_selective_coverage",
    ),
    (
        "promotion_contract.claim_factuality_probe_comparison.best_redline_auroc.mean",
        "claim_factuality_probe_comparison_best_redline_auroc",
    ),
    (
        "promotion_contract.claim_factuality_probe_comparison.best_redline_margin.mean",
        "claim_factuality_probe_comparison_best_redline_margin",
    ),
)
_PRODUCT_RUNTIME_DRIFT_CLAIM_RISK_LOCALIZATION_EVIDENCE_FIELDS: tuple[
    tuple[str, str],
    ...
] = (
    (
        "claim_risk_localization.coverage_rate",
        "claim_risk_localization_coverage_rate",
    ),
    (
        "claim_risk_localization.high_risk_claim_count",
        "claim_risk_localization_high_risk_claim_count",
    ),
    (
        "claim_risk_localization.medium_or_high_risk_claim_count",
        "claim_risk_localization_medium_or_high_risk_claim_count",
    ),
    (
        "claim_risk_localization.entity_candidate_observation_count",
        "claim_risk_localization_entity_candidate_observation_count",
    ),
    (
        "claim_risk_localization.unique_entity_candidate_count",
        "claim_risk_localization_unique_entity_candidate_count",
    ),
    (
        "claim_risk_localization.high_risk_entity_candidate_count",
        "claim_risk_localization_high_risk_entity_candidate_count",
    ),
    (
        "claim_risk_localization.medium_or_high_entity_candidate_count",
        "claim_risk_localization_medium_or_high_entity_candidate_count",
    ),
)
_PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    (
        "promotion_contract.counterfactual_verification.coverage_rate",
        "counterfactual_verification_coverage_rate",
    ),
    (
        "promotion_contract.counterfactual_verification.manifest_verified_rate",
        "counterfactual_verification_manifest_verified_rate",
    ),
    (
        "promotion_contract.counterfactual_verification.record_count.mean",
        "counterfactual_verification_record_count",
    ),
    (
        "promotion_contract.counterfactual_verification.pass_rate.mean",
        "counterfactual_verification_pass_rate",
    ),
    (
        "promotion_contract.counterfactual_verification.false_invariance_rate.mean",
        "counterfactual_verification_false_invariance_rate",
    ),
    (
        "promotion_contract.counterfactual_verification.flip_success_count.mean",
        "counterfactual_verification_flip_success_count",
    ),
)
_PRODUCT_RUNTIME_DRIFT_FACT_SELFCHECK_GATE_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    (
        "promotion_contract.fact_selfcheck_gate.coverage_rate",
        "fact_selfcheck_gate_coverage_rate",
    ),
    (
        "promotion_contract.fact_selfcheck_gate.report_present_rate",
        "fact_selfcheck_gate_report_present_rate",
    ),
    (
        "promotion_contract.fact_selfcheck_gate.manifest_present_rate",
        "fact_selfcheck_gate_manifest_present_rate",
    ),
    (
        "promotion_contract.fact_selfcheck_gate.manifest_verified_rate",
        "fact_selfcheck_gate_manifest_verified_rate",
    ),
    (
        "promotion_contract.fact_selfcheck_gate.passed_rate",
        "fact_selfcheck_gate_passed_rate",
    ),
    (
        "promotion_contract.fact_selfcheck_gate.run_count.mean",
        "fact_selfcheck_gate_run_count",
    ),
    (
        "promotion_contract.fact_selfcheck_gate.failed_run_count.mean",
        "fact_selfcheck_gate_failed_run_count",
    ),
    (
        "promotion_contract.fact_selfcheck_gate.min_executed_rate.mean",
        "fact_selfcheck_gate_min_executed_rate",
    ),
    (
        "promotion_contract.fact_selfcheck_gate.min_decided_rate.mean",
        "fact_selfcheck_gate_min_decided_rate",
    ),
    (
        "promotion_contract.fact_selfcheck_gate.max_not_applicable_rate.mean",
        "fact_selfcheck_gate_max_not_applicable_rate",
    ),
    (
        "promotion_contract.fact_selfcheck_gate.min_claim_triples_per_record.mean",
        "fact_selfcheck_gate_min_claim_triples_per_record",
    ),
    (
        "promotion_contract.fact_selfcheck_gate.min_sample_triples_per_record.mean",
        "fact_selfcheck_gate_min_sample_triples_per_record",
    ),
)
_PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("triple_coverage.claim_triple_coverage_rate", "triple_claim_coverage_rate"),
    ("triple_coverage.audit_claim_coverage_rate", "triple_audit_claim_coverage_rate"),
    ("triple_coverage.audit_pass_rate", "triple_audit_pass_rate"),
    ("triple_coverage.slot_coverage_rate", "triple_slot_coverage_rate"),
)
_PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    (
        "promotion_contract.covered_fact_properties.recommended_route_property_metrics.property_metric_count.mean",
        "covered_fact_recommended_route_property_metric_count",
    ),
    (
        "promotion_contract.covered_fact_properties.recommended_route_property_metrics.min_records.mean",
        "covered_fact_recommended_route_min_records",
    ),
    (
        "promotion_contract.covered_fact_properties.recommended_route_property_metrics.min_source_documents.mean",
        "covered_fact_recommended_route_min_source_documents",
    ),
    (
        "promotion_contract.covered_fact_properties.recommended_route_property_metrics.min_decision_accuracy.mean",
        "covered_fact_recommended_route_min_decision_accuracy",
    ),
    (
        "promotion_contract.covered_fact_properties.recommended_route_property_metrics.max_false_supported_rate.mean",
        "covered_fact_recommended_route_max_false_supported_rate",
    ),
    (
        "promotion_contract.covered_fact_properties.recommended_route_property_metrics.min_false_refuted_rate.mean",
        "covered_fact_recommended_route_min_false_refuted_rate",
    ),
)
_PRODUCT_RUNTIME_DRIFT_ACTION_GATE_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    (
        "promotion_contract.product_trace_replay.action_audit_gate.error_rate.mean",
        "product_trace_action_audit_error_rate",
    ),
    (
        "promotion_contract.product_trace_replay.action_audit_gate.missing_retrieval_action_rate.mean",
        "product_trace_action_audit_missing_retrieval_action_rate",
    ),
    (
        "promotion_contract.product_trace_replay.action_audit_gate.missing_plan_retrieval_query_rate.mean",
        "product_trace_action_audit_missing_plan_retrieval_query_rate",
    ),
    (
        "promotion_contract.product_trace_replay.action_audit_gate.malformed_payload_rate.mean",
        "product_trace_action_audit_malformed_payload_rate",
    ),
    (
        "promotion_contract.product_trace_replay.action_audit_gate.unexpected_action_rate.mean",
        "product_trace_action_audit_unexpected_action_rate",
    ),
    (
        "promotion_contract.product_trace_replay.action_audit_gate.unknown_claim_id_rate.mean",
        "product_trace_action_audit_unknown_claim_id_rate",
    ),
    (
        "promotion_contract.product_trace_replay.action_execution_gate.alignment_failed_trace_rate.mean",
        "product_trace_action_execution_alignment_failed_trace_rate",
    ),
    (
        "promotion_contract.product_trace_replay.action_execution_gate.missing_result_rate.mean",
        "product_trace_action_execution_missing_result_rate",
    ),
    (
        "promotion_contract.product_trace_replay.action_execution_gate.unexpected_result_rate.mean",
        "product_trace_action_execution_unexpected_result_rate",
    ),
    (
        "promotion_contract.product_trace_replay.action_execution_gate.request_id_mismatch_rate.mean",
        "product_trace_action_execution_request_id_mismatch_rate",
    ),
)
_PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_ACTION_GATE_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("world_model_action_gate.coverage_rate", "world_model_action_gate_coverage_rate"),
    ("world_model_action_gate.pass_rate", "world_model_action_gate_pass_rate"),
    ("world_model_action_gate.blocked_rate", "world_model_action_gate_blocked_rate"),
    (
        "world_model_action_gate.side_effect_block_violation_rate",
        "world_model_action_gate_side_effect_block_violation_rate",
    ),
    (
        "world_model_action_gate.low_prediction_confidence_rate",
        "world_model_action_gate_low_prediction_confidence_rate",
    ),
    ("world_model_action_gate.low_agreement_rate", "world_model_action_gate_low_agreement_rate"),
    (
        "world_model_action_gate.no_rule_matched_rate",
        "world_model_action_gate_no_rule_matched_rate",
    ),
    (
        "world_model_action_gate.postcondition_refuted_rate",
        "world_model_action_gate_postcondition_refuted_rate",
    ),
    (
        "world_model_action_gate.postcondition_insufficient_evidence_rate",
        "world_model_action_gate_postcondition_insufficient_evidence_rate",
    ),
    (
        "world_model_action_gate.postcondition_error_rate",
        "world_model_action_gate_postcondition_error_rate",
    ),
)
_PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_ROLLOUT_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("world_model_rollout.coverage_rate", "world_model_rollout_coverage_rate"),
    ("world_model_rollout.sync_rate", "world_model_rollout_sync_rate"),
    ("world_model_rollout.drift_rate", "world_model_rollout_drift_rate"),
    ("world_model_rollout.trace_gap_rate", "world_model_rollout_trace_gap_rate"),
    (
        "world_model_rollout.path_mismatch_rate",
        "world_model_rollout_path_mismatch_rate",
    ),
)
_PRODUCT_RUNTIME_DRIFT_ACTION_RECEIPTS_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("action_receipts.coverage_rate", "product_trace_action_receipts_coverage_rate"),
    (
        "action_receipts.missing_receipt_rate",
        "product_trace_action_receipts_missing_receipt_rate",
    ),
    (
        "action_receipts.invalid_receipt_rate",
        "product_trace_action_receipts_invalid_receipt_rate",
    ),
    (
        "action_receipts.fingerprint_mismatch_rate",
        "product_trace_action_receipts_fingerprint_mismatch_rate",
    ),
    (
        "action_receipts.unsigned_receipt_rate",
        "product_trace_action_receipts_unsigned_receipt_rate",
    ),
)
_PRODUCT_RUNTIME_DRIFT_RECEIPT_CLAIM_SUPPORT_EVIDENCE_FIELDS: tuple[
    tuple[str, str],
    ...
] = (
    (
        "receipt_claim_support.reference_support_rate",
        "product_trace_receipt_claim_support_reference_support_rate",
    ),
    (
        "receipt_claim_support.unsupported_reference_rate",
        "product_trace_receipt_claim_support_unsupported_reference_rate",
    ),
    (
        "receipt_claim_support.missing_reference_rate",
        "product_trace_receipt_claim_support_missing_reference_rate",
    ),
    (
        "receipt_claim_support.unreceipted_reference_rate",
        "product_trace_receipt_claim_support_unreceipted_reference_rate",
    ),
    (
        "receipt_claim_support.failed_result_reference_rate",
        "product_trace_receipt_claim_support_failed_result_reference_rate",
    ),
    (
        "receipt_claim_support.fingerprint_mismatch_reference_rate",
        "product_trace_receipt_claim_support_fingerprint_mismatch_reference_rate",
    ),
    (
        "receipt_claim_support.unsigned_reference_rate",
        "product_trace_receipt_claim_support_unsigned_reference_rate",
    ),
)
_PRODUCT_RUNTIME_DRIFT_TRAJECTORY_AUDIT_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("trajectory_audit.failed_trace_rate", "product_trace_trajectory_audit_failed_trace_rate"),
    ("trajectory_audit.error_rate", "product_trace_trajectory_audit_error_rate"),
    ("trajectory_audit.factual_rate", "product_trace_trajectory_audit_factual_rate"),
    ("trajectory_audit.referential_rate", "product_trace_trajectory_audit_referential_rate"),
    ("trajectory_audit.logical_rate", "product_trace_trajectory_audit_logical_rate"),
    ("trajectory_audit.procedural_rate", "product_trace_trajectory_audit_procedural_rate"),
    ("trajectory_audit.scope_rate", "product_trace_trajectory_audit_scope_rate"),
    ("trajectory_audit.cascade_rate", "product_trace_trajectory_audit_cascade_rate"),
)
_PRODUCT_RUNTIME_DRIFT_PROVENANCE_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("provenance.coverage_rate", "product_trace_provenance_coverage_rate"),
    (
        "provenance.supported_claim_evidence_coverage",
        "product_trace_provenance_supported_claim_evidence_coverage",
    ),
    ("provenance.missing_reference_rate", "product_trace_provenance_missing_reference_rate"),
    (
        "provenance.unsupported_supported_claim_rate",
        "product_trace_provenance_unsupported_supported_claim_rate",
    ),
    ("provenance.error_rate", "product_trace_provenance_error_rate"),
    (
        "provenance.final_answer_evidence_reference_rate",
        "product_trace_provenance_final_answer_evidence_reference_rate",
    ),
    (
        "evidence_graph_consistency.consistency_coverage_rate",
        "product_trace_evidence_graph_consistency_coverage_rate",
    ),
    (
        "evidence_graph_consistency.supported_claim_consistency_rate",
        "product_trace_evidence_graph_consistency_supported_claim_consistency_rate",
    ),
    (
        "evidence_graph_consistency.missing_number_rate",
        "product_trace_evidence_graph_consistency_missing_number_rate",
    ),
    (
        "evidence_graph_consistency.cross_claim_retrieval_hit_rate",
        "product_trace_evidence_graph_consistency_cross_claim_hit_rate",
    ),
    (
        "evidence_graph_consistency.error_rate",
        "product_trace_evidence_graph_consistency_error_rate",
    ),
)
_PRODUCT_RUNTIME_DRIFT_CITATION_INTEGRITY_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    (
        "citation_integrity.participating_trace_rate",
        "product_trace_citation_integrity_participating_trace_rate",
    ),
    ("citation_integrity.coverage_rate", "product_trace_citation_integrity_coverage_rate"),
    ("citation_integrity.mismatch_rate", "product_trace_citation_integrity_mismatch_rate"),
    ("citation_integrity.unresolved_rate", "product_trace_citation_integrity_unresolved_rate"),
    ("citation_integrity.issue_rate", "product_trace_citation_integrity_issue_rate"),
    ("citation_integrity.trace_gap_rate", "product_trace_citation_integrity_trace_gap_rate"),
)
_PRODUCT_RUNTIME_DRIFT_EVIDENCE_QUALITY_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    (
        "evidence_quality.trace_coverage_rate",
        "product_trace_evidence_quality_trace_coverage_rate",
    ),
    ("evidence_quality.coverage_rate", "product_trace_evidence_quality_coverage_rate"),
    ("evidence_quality.pass_rate", "product_trace_evidence_quality_pass_rate"),
    ("evidence_quality.failure_rate", "product_trace_evidence_quality_failure_rate"),
    (
        "evidence_quality.failed_result_rate",
        "product_trace_evidence_quality_failed_result_rate",
    ),
    (
        "evidence_quality.stale_evidence_rate",
        "product_trace_evidence_quality_stale_evidence_rate",
    ),
    (
        "evidence_quality.untrusted_source_rate",
        "product_trace_evidence_quality_untrusted_source_rate",
    ),
    (
        "evidence_quality.missing_source_rate",
        "product_trace_evidence_quality_missing_source_rate",
    ),
    (
        "evidence_quality.missing_timestamp_rate",
        "product_trace_evidence_quality_missing_timestamp_rate",
    ),
)
_PRODUCT_RUNTIME_DRIFT_METACOGNITION_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    (
        "metacognition.trace_coverage_rate",
        "product_trace_metacognition_trace_coverage_rate",
    ),
    ("metacognition.pass_rate", "product_trace_metacognition_pass_rate"),
    (
        "metacognition.overconfident_risk_rate",
        "product_trace_metacognition_overconfident_risk_rate",
    ),
    (
        "metacognition.miscalibration_score.mean",
        "product_trace_metacognition_miscalibration_score_mean",
    ),
)
_PRODUCT_RUNTIME_DRIFT_EVIDENCE_HANDOFF_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    (
        "promotion_contract.evidence_handoff.coverage_rate",
        "evidence_handoff_coverage_rate",
    ),
    (
        "promotion_contract.evidence_handoff.manifest_verified_rate",
        "evidence_handoff_manifest_verified_rate",
    ),
    (
        "promotion_contract.evidence_handoff.present_metric_rate.mean",
        "evidence_handoff_present_metric_rate",
    ),
    (
        "promotion_contract.evidence_handoff.missing_metric_rate.mean",
        "evidence_handoff_missing_metric_rate",
    ),
    (
        "promotion_contract.evidence_handoff.missing_metric_count.mean",
        "evidence_handoff_missing_metric_count",
    ),
    (
        "promotion_contract.evidence_handoff.blocked_group_count.mean",
        "evidence_handoff_blocked_group_count",
    ),
    (
        "promotion_contract.evidence_handoff.promoted_group_rate.mean",
        "evidence_handoff_promoted_group_rate",
    ),
)
_PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("world_model.participating_trace_rate", "world_model_participating_trace_rate"),
    ("world_model.coverage_rate", "world_model_coverage_rate"),
    ("world_model.conflict_rate", "world_model_conflict_rate"),
    ("world_model.low_agreement_rate", "world_model_low_agreement_rate"),
    ("world_model.trace_gap_rate", "world_model_trace_gap_rate"),
)
_PRODUCT_RUNTIME_DRIFT_CONTEXT_SENSITIVITY_EVIDENCE_FIELDS: tuple[
    tuple[str, str],
    ...
] = (
    (
        "context_sensitivity.participating_trace_rate",
        "context_sensitivity_participating_trace_rate",
    ),
    ("context_sensitivity.coverage_rate", "context_sensitivity_coverage_rate"),
    (
        "context_sensitivity.flagged_result_rate",
        "context_sensitivity_flagged_result_rate",
    ),
    ("context_sensitivity.trace_gap_rate", "context_sensitivity_trace_gap_rate"),
    ("context_sensitivity.max_flagged_rate", "context_sensitivity_max_flagged_rate"),
    (
        "context_sensitivity.max_context_sensitivity_ratio",
        "context_sensitivity_max_context_sensitivity_ratio",
    ),
)
_PRODUCT_RUNTIME_DRIFT_EVIDENCE_ALIGNMENT_EVIDENCE_FIELDS: tuple[
    tuple[str, str],
    ...
] = (
    (
        "evidence_alignment.participating_trace_rate",
        "evidence_alignment_participating_trace_rate",
    ),
    ("evidence_alignment.coverage_rate", "evidence_alignment_coverage_rate"),
    ("evidence_alignment.alignment_rate", "evidence_alignment_alignment_rate"),
    ("evidence_alignment.misalignment_rate", "evidence_alignment_misalignment_rate"),
    (
        "evidence_alignment.insufficient_evidence_rate",
        "evidence_alignment_insufficient_evidence_rate",
    ),
    (
        "evidence_alignment.citation_reference_coverage_rate",
        "evidence_alignment_citation_reference_coverage_rate",
    ),
    ("evidence_alignment.issue_rate", "evidence_alignment_issue_rate"),
    ("evidence_alignment.trace_gap_rate", "evidence_alignment_trace_gap_rate"),
)
_PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_ROBUSTNESS_EVIDENCE_FIELDS: tuple[
    tuple[str, str],
    ...
] = (
    (
        "counterfactual_robustness.participating_trace_rate",
        "counterfactual_robustness_participating_trace_rate",
    ),
    (
        "counterfactual_robustness.coverage_rate",
        "counterfactual_robustness_coverage_rate",
    ),
    (
        "counterfactual_robustness.pass_rate",
        "counterfactual_robustness_pass_rate",
    ),
    (
        "counterfactual_robustness.flip_success_rate",
        "counterfactual_robustness_flip_success_rate",
    ),
    (
        "counterfactual_robustness.false_invariance_rate",
        "counterfactual_robustness_false_invariance_rate",
    ),
    (
        "counterfactual_robustness.trace_gap_rate",
        "counterfactual_robustness_trace_gap_rate",
    ),
)
_PRODUCT_RUNTIME_DRIFT_FRONTIER_RELEASE_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    (
        "promotion_contract.frontier_release_evidence.coverage_rate",
        "frontier_release_evidence_coverage_rate",
    ),
    (
        "promotion_contract.frontier_release_evidence.report_present_rate",
        "frontier_release_evidence_report_present_rate",
    ),
    (
        "promotion_contract.frontier_release_evidence.manifest_present_rate",
        "frontier_release_evidence_manifest_present_rate",
    ),
    (
        "promotion_contract.frontier_release_evidence.status_promote_rate",
        "frontier_release_evidence_status_promote_rate",
    ),
    (
        "promotion_contract.frontier_release_evidence.decision_promote_rate",
        "frontier_release_evidence_decision_promote_rate",
    ),
    (
        "promotion_contract.frontier_release_evidence.verifier_track_promote_rate",
        "frontier_release_evidence_verifier_track_promote_rate",
    ),
    (
        "promotion_contract.frontier_release_evidence.abstention_track_promote_rate",
        "frontier_release_evidence_abstention_track_promote_rate",
    ),
    (
        "promotion_contract.frontier_release_evidence.citation_batch_track_promote_rate",
        "frontier_release_evidence_citation_batch_track_promote_rate",
    ),
    (
        "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_track_promote_rate",
        "frontier_release_evidence_frontier_rerun_rollup_track_promote_rate",
    ),
    (
        "promotion_contract.frontier_release_evidence.run_count.mean",
        "frontier_release_evidence_run_count",
    ),
    (
        "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_report_count.mean",
        "frontier_release_evidence_frontier_rerun_rollup_report_count",
    ),
    (
        "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_candidate_count.mean",
        "frontier_release_evidence_frontier_rerun_rollup_candidate_count",
    ),
    (
        "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_missing_report_count.mean",
        "frontier_release_evidence_frontier_rerun_rollup_missing_report_count",
    ),
    (
        "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_invalid_report_count.mean",
        "frontier_release_evidence_frontier_rerun_rollup_invalid_report_count",
    ),
    (
        "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_blocked_candidate_count.mean",
        "frontier_release_evidence_frontier_rerun_rollup_blocked_candidate_count",
    ),
    (
        "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_promotion_ready_count.mean",
        "frontier_release_evidence_frontier_rerun_rollup_promotion_ready_count",
    ),
    (
        "promotion_contract.frontier_release_evidence.citation_batch_rollup_count.mean",
        "frontier_release_evidence_citation_batch_rollup_count",
    ),
    (
        "promotion_contract.frontier_release_evidence.citation_batch_expected_batch_count.mean",
        "frontier_release_evidence_citation_batch_expected_batch_count",
    ),
    (
        "promotion_contract.frontier_release_evidence.citation_batch_observed_batch_count.mean",
        "frontier_release_evidence_citation_batch_observed_batch_count",
    ),
    (
        "promotion_contract.frontier_release_evidence.citation_batch_missing_expected_batch_count.mean",
        "frontier_release_evidence_citation_batch_missing_expected_batch_count",
    ),
    (
        "promotion_contract.frontier_release_evidence.citation_batch_duplicate_batch_count.mean",
        "frontier_release_evidence_citation_batch_duplicate_batch_count",
    ),
    (
        "promotion_contract.frontier_release_evidence.citation_batch_unexpected_batch_count.mean",
        "frontier_release_evidence_citation_batch_unexpected_batch_count",
    ),
)


def compare_release_candidates(
    *,
    readiness_registry_path: str | Path,
    route_registry_path: str | Path | None = None,
    readiness_baseline_keys: Sequence[str] = (),
    route_baseline_keys: Sequence[str] = (),
    required_route_baseline_keys: Sequence[str] = (),
    release_policy_profile: str | None = None,
    require_structured_fact_robustness: bool = False,
    structured_fact_canonical_route_key: str | None = None,
    structured_fact_paraphrase_route_key: str | None = None,
    structured_fact_robustness_min_selected: int | None = None,
    structured_fact_robustness_min_decision_accuracy: float | None = None,
    structured_fact_robustness_max_false_supported_rate: float | None = None,
    structured_fact_robustness_min_false_refuted_rate: float | None = None,
    structured_fact_robustness_min_covered_fact_properties: int | None = None,
    structured_fact_robustness_min_covered_fact_property_records: int | None = None,
    structured_fact_robustness_min_covered_fact_property_source_documents: int | None = None,
    structured_fact_robustness_min_covered_fact_property_decision_accuracy: float | None = None,
    structured_fact_robustness_max_covered_fact_property_false_supported_rate: float | None = None,
    structured_fact_robustness_min_covered_fact_property_false_refuted_rate: float | None = None,
    performance_registry_path: str | Path | None = None,
    performance_baseline_key: str | None = None,
    selector_replay_report_path: str | Path | None = None,
    product_runtime_drift_report_path: str | Path | None = None,
    require_product_runtime_drift_promotion_evidence: bool = False,
    require_product_runtime_drift_pre_generation_evidence: bool = False,
    require_product_runtime_drift_claim_factuality_evidence: bool = False,
    require_product_runtime_drift_claim_risk_localization_evidence: bool = False,
    require_product_runtime_drift_counterfactual_evidence: bool = False,
    require_product_runtime_drift_fact_selfcheck_gate_evidence: bool = False,
    require_product_runtime_drift_triple_audit_evidence: bool = False,
    require_product_runtime_drift_covered_fact_property_evidence: bool = False,
    require_product_runtime_drift_action_gate_evidence: bool = False,
    require_product_runtime_drift_world_model_action_gate_evidence: bool = False,
    require_product_runtime_drift_world_model_rollout_evidence: bool = False,
    require_product_runtime_drift_action_receipts_evidence: bool = False,
    require_product_runtime_drift_receipt_claim_support_evidence: bool = False,
    require_product_runtime_drift_trajectory_audit_evidence: bool = False,
    require_product_runtime_drift_provenance_evidence: bool = False,
    require_product_runtime_drift_citation_integrity_evidence: bool = False,
    require_product_runtime_drift_evidence_quality_evidence: bool = False,
    require_product_runtime_drift_metacognition_evidence: bool = False,
    require_product_runtime_drift_evidence_handoff_evidence: bool = False,
    require_product_runtime_drift_world_model_evidence: bool = False,
    require_product_runtime_drift_context_sensitivity_evidence: bool = False,
    require_product_runtime_drift_evidence_alignment_evidence: bool = False,
    require_product_runtime_drift_counterfactual_robustness_evidence: bool = False,
    require_product_runtime_drift_frontier_release_evidence: bool = False,
    release_efficiency_report_path: str | Path | None = None,
    external_evidence_baseline_comparison_path: str | Path | None = None,
    external_evidence_baseline_comparison_registry_path: str | Path | None = None,
    external_evidence_baseline_comparison_key: str | None = None,
    pre_generation_probe_comparison_path: str | Path | None = None,
    pre_generation_probe_comparison_registry_path: str | Path | None = None,
    pre_generation_probe_comparison_key: str | None = None,
    claim_factuality_probe_comparison_path: str | Path | None = None,
    claim_factuality_probe_comparison_registry_path: str | Path | None = None,
    claim_factuality_probe_comparison_key: str | None = None,
    frontier_release_evidence_path: str | Path | None = None,
    frontier_release_evidence_registry_path: str | Path | None = None,
    frontier_release_evidence_key: str | None = None,
    require_frontier_release_input_manifests: bool = False,
    world_model_signal_workflow_path: str | Path | None = None,
    world_model_signal_workflow_registry_path: str | Path | None = None,
    world_model_signal_workflow_key: str | None = None,
    context_sensitivity_workflow_path: str | Path | None = None,
    context_sensitivity_workflow_registry_path: str | Path | None = None,
    context_sensitivity_workflow_key: str | None = None,
    mechanism_handoff_evidence_bundle_path: str | Path | None = None,
    mechanism_handoff_evidence_bundle_registry_path: str | Path | None = None,
    mechanism_handoff_evidence_bundle_key: str | None = None,
    pathway_intervention_workflow_path: str | Path | None = None,
    pathway_intervention_workflow_registry_path: str | Path | None = None,
    pathway_intervention_workflow_key: str | None = None,
    product_trace_replay_workflow_path: str | Path | None = None,
    product_trace_replay_workflow_registry_path: str | Path | None = None,
    product_trace_replay_workflow_key: str | None = None,
    require_product_trace_action_audit_gate: bool = False,
    require_product_trace_action_execution_gate: bool = False,
    selfcheck_signal_fusion_workflow_path: str | Path | None = None,
    selfcheck_signal_fusion_workflow_registry_path: str | Path | None = None,
    selfcheck_signal_fusion_workflow_key: str | None = None,
    uncertainty_escalation_workflow_path: str | Path | None = None,
    uncertainty_escalation_workflow_registry_path: str | Path | None = None,
    uncertainty_escalation_workflow_key: str | None = None,
    min_uncertainty_escalation_records: int | None = None,
    min_uncertainty_escalation_trigger_rate: float | None = None,
    min_uncertainty_escalation_retrieval_evidence_rate: float | None = None,
    max_uncertainty_escalation_final_false_accept_rate: float | None = None,
    max_uncertainty_escalation_false_accept_delta: float | None = None,
    feedback_policy_workflow_path: str | Path | None = None,
    feedback_policy_workflow_registry_path: str | Path | None = None,
    feedback_policy_workflow_key: str | None = None,
    feedback_policy_min_matched_feedback_count: int | None = None,
    feedback_policy_min_safety_coverage: float | None = None,
    feedback_policy_max_unknown_safety_issue_rate: float | None = None,
    adapter_family_matrix_path: str | Path | None = None,
    adapter_family_profile: str | None = None,
    required_adapter_routes: Sequence[str] = (),
    require_state_transition_world_model: bool = False,
    triple_extraction_fixture_matrix_path: str | Path | None = None,
    triple_extraction_fixture_matrix_registry_path: str | Path | None = None,
    triple_extraction_fixture_matrix_key: str | None = None,
    min_triple_extraction_corpora: int | None = None,
    min_triple_extraction_distinct_predicates: int | None = None,
    min_triple_extraction_external_prediction_count: int | None = None,
    min_triple_extraction_external_prediction_corpora: int | None = None,
    min_triple_extraction_mean_best_external_f1: float | None = None,
    counterfactual_verification_report_path: str | Path | None = None,
    counterfactual_verification_registry_path: str | Path | None = None,
    counterfactual_verification_key: str | None = None,
    min_counterfactual_verification_records: int | None = None,
    min_counterfactual_verification_pass_rate: float | None = None,
    max_counterfactual_verification_false_invariance_rate: float | None = None,
    require_performance_score_dump_cache: bool = False,
    min_performance_score_dump_cache_jsonl_view_hit_rate: float | None = None,
    performance_drift_baseline_key: str | None = None,
    max_performance_uncached_total_seconds_ratio: float | None = None,
    max_performance_cached_total_seconds_ratio: float | None = None,
    max_performance_cache_only_total_seconds_ratio: float | None = None,
    max_performance_score_dump_cache_jsonl_view_hit_rate_drop: float | None = None,
    recursive: bool = True,
    allow_unverified: bool = False,
    manifest_fingerprint_workers: int = 1,
    runtime_profile: str | None = None,
    inside_trigger_budget_policy: str | None = None,
    min_best_quality_auroc: float | None = None,
    max_uncached_forward_seconds: float | None = None,
    max_cache_only_seconds: float | None = None,
    max_recommended_runtime_seconds: float | None = None,
    max_covariance_maha_last_auroc_drop: float | None = None,
    max_inside_sample_count_ratio: float | None = None,
    max_inside_generation_seconds_ratio: float | None = None,
    min_selected: int | None = None,
    min_decision_accuracy: float | None = None,
    max_false_supported_rate: float | None = None,
    min_false_refuted_rate: float | None = None,
    max_verified_false_alarm: float | None = None,
    min_verified_detection: float | None = None,
    max_mean_duration_seconds: float | None = None,
    max_p99_duration_seconds: float | None = None,
    max_max_duration_seconds: float | None = None,
    max_mean_attempted_route_count: float | None = None,
    max_retrieval_use_rate: float | None = None,
    max_runtime_total_seconds: float | None = None,
    max_retrieval_hit_count: float | None = None,
    min_claims_cache_hit_rate: float | None = None,
    min_verifier_trace_cache_hit_rate: float | None = None,
    min_covered_fact_properties: int | None = None,
    min_covered_fact_property_records: int | None = None,
    min_covered_fact_property_source_documents: int | None = None,
    min_covered_fact_property_decision_accuracy: float | None = None,
    max_covered_fact_property_false_supported_rate: float | None = None,
    min_covered_fact_property_false_refuted_rate: float | None = None,
    require_non_oracle_evidence: bool = False,
    require_retrieval_provenance_filter: bool = False,
    required_retrieval_source_prefixes: Sequence[str] = (),
    required_retrieval_metadata: Mapping[str, Any] | None = None,
    min_retrieval_filter_score: float | None = None,
    require_retrieval_stress_control: bool = False,
    retrieval_stress_manifest: str | Path | None = None,
    min_stress_false_supported_rate: float | None = None,
    max_stress_false_refuted_rate: float | None = None,
    required_route_min_selected: int | None = None,
    required_route_min_decision_accuracy: float | None = None,
    required_route_max_false_supported_rate: float | None = None,
    required_route_min_false_refuted_rate: float | None = None,
    required_route_max_verified_false_alarm: float | None = None,
    required_route_min_verified_detection: float | None = None,
    required_route_max_mean_duration_seconds: float | None = None,
    required_route_max_p99_duration_seconds: float | None = None,
    required_route_max_max_duration_seconds: float | None = None,
    required_route_max_mean_attempted_route_count: float | None = None,
    required_route_max_retrieval_use_rate: float | None = None,
    required_route_max_runtime_total_seconds: float | None = None,
    required_route_max_retrieval_hit_count: float | None = None,
    required_route_min_claims_cache_hit_rate: float | None = None,
    required_route_min_verifier_trace_cache_hit_rate: float | None = None,
    required_route_min_covered_fact_properties: int | None = None,
    required_route_min_covered_fact_property_records: int | None = None,
    required_route_min_covered_fact_property_source_documents: int | None = None,
    required_route_min_covered_fact_property_decision_accuracy: float | None = None,
    required_route_max_covered_fact_property_false_supported_rate: float | None = None,
    required_route_min_covered_fact_property_false_refuted_rate: float | None = None,
    required_route_require_non_oracle_evidence: bool = False,
    required_route_require_retrieval_provenance_filter: bool = False,
    required_route_required_retrieval_source_prefixes: Sequence[str] = (),
    required_route_required_retrieval_metadata: Mapping[str, Any] | None = None,
    required_route_min_retrieval_filter_score: float | None = None,
    required_route_require_retrieval_stress_control: bool = False,
    required_route_retrieval_stress_manifest: str | Path | None = None,
    required_route_min_stress_false_supported_rate: float | None = None,
    required_route_max_stress_false_refuted_rate: float | None = None,
    notes: Sequence[str] = (),
    fingerprint_cache: MutableMapping[str, dict[str, Any]] | None = None,
    json_cache: MutableMapping[str, dict[str, Any]] | None = None,
    json_cache_stats: MutableMapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed deployable release candidate from saved baselines."""
    disabled_profile_defaults: list[str] = []
    if external_evidence_baseline_comparison_path is not None:
        disabled_profile_defaults.append("external_evidence_baseline_comparison_key")
    if triple_extraction_fixture_matrix_path is not None:
        disabled_profile_defaults.append("triple_extraction_fixture_matrix_key")
    if mechanism_handoff_evidence_bundle_path is not None:
        disabled_profile_defaults.append("mechanism_handoff_evidence_bundle_key")
    release_policy_profile, release_policy_values, release_policy_applied = (
        apply_release_policy_profile_defaults(
            release_policy_profile,
            {
                "require_structured_fact_robustness": require_structured_fact_robustness,
                "min_best_quality_auroc": min_best_quality_auroc,
                "max_uncached_forward_seconds": max_uncached_forward_seconds,
                "max_recommended_runtime_seconds": max_recommended_runtime_seconds,
                "min_selected": min_selected,
                "min_decision_accuracy": min_decision_accuracy,
                "max_false_supported_rate": max_false_supported_rate,
                "min_false_refuted_rate": min_false_refuted_rate,
                "required_route_min_selected": required_route_min_selected,
                "required_route_min_decision_accuracy": required_route_min_decision_accuracy,
                "required_route_max_false_supported_rate": required_route_max_false_supported_rate,
                "required_route_min_false_refuted_rate": required_route_min_false_refuted_rate,
                "required_route_min_covered_fact_properties": required_route_min_covered_fact_properties,
                "required_route_min_covered_fact_property_records": (
                    required_route_min_covered_fact_property_records
                ),
                "required_route_min_covered_fact_property_source_documents": (
                    required_route_min_covered_fact_property_source_documents
                ),
                "required_route_min_covered_fact_property_decision_accuracy": (
                    required_route_min_covered_fact_property_decision_accuracy
                ),
                "required_route_max_covered_fact_property_false_supported_rate": (
                    required_route_max_covered_fact_property_false_supported_rate
                ),
                "required_route_min_covered_fact_property_false_refuted_rate": (
                    required_route_min_covered_fact_property_false_refuted_rate
                ),
                "structured_fact_robustness_min_selected": (
                    structured_fact_robustness_min_selected
                ),
                "structured_fact_robustness_min_decision_accuracy": (
                    structured_fact_robustness_min_decision_accuracy
                ),
                "structured_fact_robustness_max_false_supported_rate": (
                    structured_fact_robustness_max_false_supported_rate
                ),
                "structured_fact_robustness_min_false_refuted_rate": (
                    structured_fact_robustness_min_false_refuted_rate
                ),
                "structured_fact_robustness_min_covered_fact_properties": (
                    structured_fact_robustness_min_covered_fact_properties
                ),
                "structured_fact_robustness_min_covered_fact_property_records": (
                    structured_fact_robustness_min_covered_fact_property_records
                ),
                "structured_fact_robustness_min_covered_fact_property_source_documents": (
                    structured_fact_robustness_min_covered_fact_property_source_documents
                ),
                "structured_fact_robustness_min_covered_fact_property_decision_accuracy": (
                    structured_fact_robustness_min_covered_fact_property_decision_accuracy
                ),
                "structured_fact_robustness_max_covered_fact_property_false_supported_rate": (
                    structured_fact_robustness_max_covered_fact_property_false_supported_rate
                ),
                "structured_fact_robustness_min_covered_fact_property_false_refuted_rate": (
                    structured_fact_robustness_min_covered_fact_property_false_refuted_rate
                ),
                "adapter_family_profile": adapter_family_profile,
                "require_state_transition_world_model": require_state_transition_world_model,
                "require_product_runtime_drift_promotion_evidence": (
                    require_product_runtime_drift_promotion_evidence
                ),
                "require_product_runtime_drift_pre_generation_evidence": (
                    require_product_runtime_drift_pre_generation_evidence
                ),
                "require_product_runtime_drift_claim_factuality_evidence": (
                    require_product_runtime_drift_claim_factuality_evidence
                ),
                "require_product_runtime_drift_claim_risk_localization_evidence": (
                    require_product_runtime_drift_claim_risk_localization_evidence
                ),
                "require_product_runtime_drift_counterfactual_evidence": (
                    require_product_runtime_drift_counterfactual_evidence
                ),
                "require_product_runtime_drift_fact_selfcheck_gate_evidence": (
                    require_product_runtime_drift_fact_selfcheck_gate_evidence
                ),
                "require_product_runtime_drift_triple_audit_evidence": (
                    require_product_runtime_drift_triple_audit_evidence
                ),
                "require_product_runtime_drift_covered_fact_property_evidence": (
                    require_product_runtime_drift_covered_fact_property_evidence
                ),
                "require_product_runtime_drift_action_gate_evidence": (
                    require_product_runtime_drift_action_gate_evidence
                ),
                "require_product_runtime_drift_world_model_action_gate_evidence": (
                    require_product_runtime_drift_world_model_action_gate_evidence
                ),
                "require_product_runtime_drift_world_model_rollout_evidence": (
                    require_product_runtime_drift_world_model_rollout_evidence
                ),
                "require_product_runtime_drift_action_receipts_evidence": (
                    require_product_runtime_drift_action_receipts_evidence
                ),
                "require_product_runtime_drift_receipt_claim_support_evidence": (
                    require_product_runtime_drift_receipt_claim_support_evidence
                ),
                "require_product_runtime_drift_trajectory_audit_evidence": (
                    require_product_runtime_drift_trajectory_audit_evidence
                ),
                "require_product_runtime_drift_provenance_evidence": (
                    require_product_runtime_drift_provenance_evidence
                ),
                "require_product_runtime_drift_citation_integrity_evidence": (
                    require_product_runtime_drift_citation_integrity_evidence
                ),
                "require_product_runtime_drift_evidence_quality_evidence": (
                    require_product_runtime_drift_evidence_quality_evidence
                ),
                "require_product_runtime_drift_metacognition_evidence": (
                    require_product_runtime_drift_metacognition_evidence
                ),
                "require_product_runtime_drift_evidence_handoff_evidence": (
                    require_product_runtime_drift_evidence_handoff_evidence
                ),
                "require_product_runtime_drift_world_model_evidence": (
                    require_product_runtime_drift_world_model_evidence
                ),
                "require_product_runtime_drift_context_sensitivity_evidence": (
                    require_product_runtime_drift_context_sensitivity_evidence
                ),
                "require_product_runtime_drift_evidence_alignment_evidence": (
                    require_product_runtime_drift_evidence_alignment_evidence
                ),
                "require_product_runtime_drift_counterfactual_robustness_evidence": (
                    require_product_runtime_drift_counterfactual_robustness_evidence
                ),
                "require_product_runtime_drift_frontier_release_evidence": (
                    require_product_runtime_drift_frontier_release_evidence
                ),
                "require_frontier_release_input_manifests": (
                    require_frontier_release_input_manifests
                ),
                "require_product_trace_action_audit_gate": require_product_trace_action_audit_gate,
                "require_product_trace_action_execution_gate": (
                    require_product_trace_action_execution_gate
                ),
                "external_evidence_baseline_comparison_key": (
                    external_evidence_baseline_comparison_key
                ),
                "triple_extraction_fixture_matrix_key": triple_extraction_fixture_matrix_key,
                "mechanism_handoff_evidence_bundle_key": (
                    mechanism_handoff_evidence_bundle_key
                ),
                "min_triple_extraction_corpora": min_triple_extraction_corpora,
                "min_triple_extraction_distinct_predicates": min_triple_extraction_distinct_predicates,
                "min_triple_extraction_external_prediction_count": (
                    min_triple_extraction_external_prediction_count
                ),
                "min_triple_extraction_external_prediction_corpora": (
                    min_triple_extraction_external_prediction_corpora
                ),
                "min_triple_extraction_mean_best_external_f1": (
                    min_triple_extraction_mean_best_external_f1
                ),
            },
            disabled_defaults=disabled_profile_defaults,
        )
    )
    require_structured_fact_robustness = bool(
        release_policy_values["require_structured_fact_robustness"]
    )
    min_best_quality_auroc = release_policy_values["min_best_quality_auroc"]
    max_uncached_forward_seconds = release_policy_values["max_uncached_forward_seconds"]
    max_recommended_runtime_seconds = release_policy_values["max_recommended_runtime_seconds"]
    min_selected = release_policy_values["min_selected"]
    min_decision_accuracy = release_policy_values["min_decision_accuracy"]
    max_false_supported_rate = release_policy_values["max_false_supported_rate"]
    min_false_refuted_rate = release_policy_values["min_false_refuted_rate"]
    required_route_min_selected = release_policy_values["required_route_min_selected"]
    required_route_min_decision_accuracy = release_policy_values["required_route_min_decision_accuracy"]
    required_route_max_false_supported_rate = release_policy_values["required_route_max_false_supported_rate"]
    required_route_min_false_refuted_rate = release_policy_values["required_route_min_false_refuted_rate"]
    required_route_min_covered_fact_properties = release_policy_values["required_route_min_covered_fact_properties"]
    required_route_min_covered_fact_property_records = release_policy_values[
        "required_route_min_covered_fact_property_records"
    ]
    required_route_min_covered_fact_property_source_documents = release_policy_values[
        "required_route_min_covered_fact_property_source_documents"
    ]
    required_route_min_covered_fact_property_decision_accuracy = release_policy_values[
        "required_route_min_covered_fact_property_decision_accuracy"
    ]
    required_route_max_covered_fact_property_false_supported_rate = release_policy_values[
        "required_route_max_covered_fact_property_false_supported_rate"
    ]
    required_route_min_covered_fact_property_false_refuted_rate = release_policy_values[
        "required_route_min_covered_fact_property_false_refuted_rate"
    ]
    structured_fact_robustness_min_selected = release_policy_values[
        "structured_fact_robustness_min_selected"
    ]
    structured_fact_robustness_min_decision_accuracy = release_policy_values[
        "structured_fact_robustness_min_decision_accuracy"
    ]
    structured_fact_robustness_max_false_supported_rate = release_policy_values[
        "structured_fact_robustness_max_false_supported_rate"
    ]
    structured_fact_robustness_min_false_refuted_rate = release_policy_values[
        "structured_fact_robustness_min_false_refuted_rate"
    ]
    structured_fact_robustness_min_covered_fact_properties = release_policy_values[
        "structured_fact_robustness_min_covered_fact_properties"
    ]
    structured_fact_robustness_min_covered_fact_property_records = release_policy_values[
        "structured_fact_robustness_min_covered_fact_property_records"
    ]
    structured_fact_robustness_min_covered_fact_property_source_documents = (
        release_policy_values[
            "structured_fact_robustness_min_covered_fact_property_source_documents"
        ]
    )
    structured_fact_robustness_min_covered_fact_property_decision_accuracy = (
        release_policy_values[
            "structured_fact_robustness_min_covered_fact_property_decision_accuracy"
        ]
    )
    structured_fact_robustness_max_covered_fact_property_false_supported_rate = (
        release_policy_values[
            "structured_fact_robustness_max_covered_fact_property_false_supported_rate"
        ]
    )
    structured_fact_robustness_min_covered_fact_property_false_refuted_rate = (
        release_policy_values[
            "structured_fact_robustness_min_covered_fact_property_false_refuted_rate"
        ]
    )
    adapter_family_profile = release_policy_values["adapter_family_profile"]
    require_state_transition_world_model = bool(
        release_policy_values["require_state_transition_world_model"]
    )
    require_product_runtime_drift_promotion_evidence = bool(
        release_policy_values["require_product_runtime_drift_promotion_evidence"]
    )
    require_product_runtime_drift_pre_generation_evidence = bool(
        release_policy_values["require_product_runtime_drift_pre_generation_evidence"]
    )
    require_product_runtime_drift_claim_factuality_evidence = bool(
        release_policy_values.get("require_product_runtime_drift_claim_factuality_evidence", False)
    )
    require_product_runtime_drift_claim_risk_localization_evidence = bool(
        release_policy_values.get(
            "require_product_runtime_drift_claim_risk_localization_evidence",
            False,
        )
    )
    require_product_runtime_drift_counterfactual_evidence = bool(
        release_policy_values["require_product_runtime_drift_counterfactual_evidence"]
    )
    require_product_runtime_drift_fact_selfcheck_gate_evidence = bool(
        release_policy_values.get(
            "require_product_runtime_drift_fact_selfcheck_gate_evidence",
            False,
        )
    )
    require_product_runtime_drift_triple_audit_evidence = bool(
        release_policy_values["require_product_runtime_drift_triple_audit_evidence"]
    )
    require_product_runtime_drift_covered_fact_property_evidence = bool(
        release_policy_values["require_product_runtime_drift_covered_fact_property_evidence"]
    )
    require_product_runtime_drift_action_gate_evidence = bool(
        release_policy_values["require_product_runtime_drift_action_gate_evidence"]
    )
    require_product_runtime_drift_world_model_action_gate_evidence = bool(
        release_policy_values.get(
            "require_product_runtime_drift_world_model_action_gate_evidence",
            False,
        )
    )
    require_product_runtime_drift_world_model_rollout_evidence = bool(
        release_policy_values.get(
            "require_product_runtime_drift_world_model_rollout_evidence",
            False,
        )
    )
    require_product_runtime_drift_action_receipts_evidence = bool(
        release_policy_values.get(
            "require_product_runtime_drift_action_receipts_evidence",
            False,
        )
    )
    require_product_runtime_drift_receipt_claim_support_evidence = bool(
        release_policy_values.get(
            "require_product_runtime_drift_receipt_claim_support_evidence",
            False,
        )
    )
    require_product_runtime_drift_trajectory_audit_evidence = bool(
        release_policy_values.get("require_product_runtime_drift_trajectory_audit_evidence", False)
    )
    require_product_runtime_drift_provenance_evidence = bool(
        release_policy_values.get("require_product_runtime_drift_provenance_evidence", False)
    )
    require_product_runtime_drift_citation_integrity_evidence = bool(
        release_policy_values.get(
            "require_product_runtime_drift_citation_integrity_evidence",
            False,
        )
    )
    require_product_runtime_drift_evidence_quality_evidence = bool(
        release_policy_values.get(
            "require_product_runtime_drift_evidence_quality_evidence",
            False,
        )
    )
    require_product_runtime_drift_metacognition_evidence = bool(
        release_policy_values.get(
            "require_product_runtime_drift_metacognition_evidence",
            False,
        )
    )
    require_product_runtime_drift_evidence_handoff_evidence = bool(
        release_policy_values["require_product_runtime_drift_evidence_handoff_evidence"]
    )
    require_product_runtime_drift_world_model_evidence = bool(
        release_policy_values.get("require_product_runtime_drift_world_model_evidence", False)
    )
    require_product_runtime_drift_context_sensitivity_evidence = bool(
        release_policy_values.get(
            "require_product_runtime_drift_context_sensitivity_evidence",
            False,
        )
    )
    require_product_runtime_drift_evidence_alignment_evidence = bool(
        release_policy_values.get(
            "require_product_runtime_drift_evidence_alignment_evidence",
            False,
        )
    )
    require_product_runtime_drift_counterfactual_robustness_evidence = bool(
        release_policy_values.get(
            "require_product_runtime_drift_counterfactual_robustness_evidence",
            False,
        )
    )
    require_product_runtime_drift_frontier_release_evidence = bool(
        release_policy_values.get(
            "require_product_runtime_drift_frontier_release_evidence",
            False,
        )
    )
    require_frontier_release_input_manifests = bool(
        release_policy_values.get("require_frontier_release_input_manifests", False)
    )
    require_product_trace_action_audit_gate = bool(
        release_policy_values["require_product_trace_action_audit_gate"]
    )
    require_product_trace_action_execution_gate = bool(
        release_policy_values["require_product_trace_action_execution_gate"]
    )
    external_evidence_baseline_comparison_key = clean_optional_key(
        release_policy_values["external_evidence_baseline_comparison_key"]
    )
    triple_extraction_fixture_matrix_key = clean_optional_key(
        release_policy_values["triple_extraction_fixture_matrix_key"]
    )
    mechanism_handoff_evidence_bundle_key = clean_optional_key(
        release_policy_values["mechanism_handoff_evidence_bundle_key"]
    )
    min_triple_extraction_corpora = release_policy_values["min_triple_extraction_corpora"]
    min_triple_extraction_distinct_predicates = release_policy_values[
        "min_triple_extraction_distinct_predicates"
    ]
    min_triple_extraction_external_prediction_count = release_policy_values[
        "min_triple_extraction_external_prediction_count"
    ]
    min_triple_extraction_external_prediction_corpora = release_policy_values[
        "min_triple_extraction_external_prediction_corpora"
    ]
    min_triple_extraction_mean_best_external_f1 = release_policy_values[
        "min_triple_extraction_mean_best_external_f1"
    ]
    structured_fact_canonical_route_key = clean_optional_key(structured_fact_canonical_route_key)
    structured_fact_paraphrase_route_key = clean_optional_key(structured_fact_paraphrase_route_key)
    if require_structured_fact_robustness and (
        structured_fact_canonical_route_key is None
        or structured_fact_paraphrase_route_key is None
    ):
        raise ValueError(
            "structured_fact robustness requires both "
            "structured_fact_canonical_route_key and structured_fact_paraphrase_route_key."
        )
    if not require_structured_fact_robustness and (
        structured_fact_canonical_route_key is not None
        or structured_fact_paraphrase_route_key is not None
    ):
        raise ValueError("structured_fact route keys require require_structured_fact_robustness=True.")
    structured_fact_route_keys: tuple[str, ...] = ()
    if require_structured_fact_robustness:
        structured_fact_route_keys = tuple(
            str(key)
            for key in (structured_fact_canonical_route_key, structured_fact_paraphrase_route_key)
            if key is not None
        )
    required_route_baseline_keys = tuple(str(key) for key in required_route_baseline_keys)
    if require_structured_fact_robustness:
        required_route_baseline_keys = append_unique(
            required_route_baseline_keys,
            (structured_fact_canonical_route_key, structured_fact_paraphrase_route_key),
        )
    structured_fact_route_key_set = set(structured_fact_route_keys)
    ordinary_required_route_keys = tuple(
        key
        for key in required_route_baseline_keys
        if key not in structured_fact_route_key_set
    )
    if performance_baseline_key is None and (
        require_performance_score_dump_cache
        or min_performance_score_dump_cache_jsonl_view_hit_rate is not None
        or performance_drift_baseline_key is not None
        or max_performance_uncached_total_seconds_ratio is not None
        or max_performance_cached_total_seconds_ratio is not None
        or max_performance_cache_only_total_seconds_ratio is not None
        or max_performance_score_dump_cache_jsonl_view_hit_rate_drop is not None
    ):
        raise ValueError("performance gates require performance_baseline_key.")
    if performance_drift_baseline_key is None and (
        max_performance_uncached_total_seconds_ratio is not None
        or max_performance_cached_total_seconds_ratio is not None
        or max_performance_cache_only_total_seconds_ratio is not None
        or max_performance_score_dump_cache_jsonl_view_hit_rate_drop is not None
    ):
        raise ValueError("performance drift thresholds require performance_drift_baseline_key.")
    if min_performance_score_dump_cache_jsonl_view_hit_rate is not None:
        cache_hit_rate_threshold = _float_or_none(
            min_performance_score_dump_cache_jsonl_view_hit_rate
        )
        if (
            cache_hit_rate_threshold is None
            or cache_hit_rate_threshold < 0
            or cache_hit_rate_threshold > 1
        ):
            raise ValueError(
                "min_performance_score_dump_cache_jsonl_view_hit_rate must be between 0 and 1."
            )
        min_performance_score_dump_cache_jsonl_view_hit_rate = cache_hit_rate_threshold
    max_performance_uncached_total_seconds_ratio = _validate_optional_non_negative_float(
        max_performance_uncached_total_seconds_ratio,
        name="max_performance_uncached_total_seconds_ratio",
    )
    max_performance_cached_total_seconds_ratio = _validate_optional_non_negative_float(
        max_performance_cached_total_seconds_ratio,
        name="max_performance_cached_total_seconds_ratio",
    )
    max_performance_cache_only_total_seconds_ratio = _validate_optional_non_negative_float(
        max_performance_cache_only_total_seconds_ratio,
        name="max_performance_cache_only_total_seconds_ratio",
    )
    max_performance_score_dump_cache_jsonl_view_hit_rate_drop = _validate_optional_unit_float(
        max_performance_score_dump_cache_jsonl_view_hit_rate_drop,
        name="max_performance_score_dump_cache_jsonl_view_hit_rate_drop",
    )
    min_triple_extraction_external_prediction_count = _validate_optional_non_negative_int(
        min_triple_extraction_external_prediction_count,
        name="min_triple_extraction_external_prediction_count",
    )
    min_triple_extraction_external_prediction_corpora = _validate_optional_non_negative_int(
        min_triple_extraction_external_prediction_corpora,
        name="min_triple_extraction_external_prediction_corpora",
    )
    min_triple_extraction_mean_best_external_f1 = _validate_optional_unit_float(
        min_triple_extraction_mean_best_external_f1,
        name="min_triple_extraction_mean_best_external_f1",
    )
    feedback_policy_min_matched_feedback_count = _validate_optional_non_negative_int(
        feedback_policy_min_matched_feedback_count,
        name="feedback_policy_min_matched_feedback_count",
    )
    feedback_policy_min_safety_coverage = _validate_optional_unit_float(
        feedback_policy_min_safety_coverage,
        name="feedback_policy_min_safety_coverage",
    )
    feedback_policy_max_unknown_safety_issue_rate = _validate_optional_unit_float(
        feedback_policy_max_unknown_safety_issue_rate,
        name="feedback_policy_max_unknown_safety_issue_rate",
    )
    min_uncertainty_escalation_records = _validate_optional_non_negative_int(
        min_uncertainty_escalation_records,
        name="min_uncertainty_escalation_records",
    )
    min_uncertainty_escalation_trigger_rate = _validate_optional_unit_float(
        min_uncertainty_escalation_trigger_rate,
        name="min_uncertainty_escalation_trigger_rate",
    )
    min_uncertainty_escalation_retrieval_evidence_rate = _validate_optional_unit_float(
        min_uncertainty_escalation_retrieval_evidence_rate,
        name="min_uncertainty_escalation_retrieval_evidence_rate",
    )
    max_uncertainty_escalation_final_false_accept_rate = _validate_optional_unit_float(
        max_uncertainty_escalation_final_false_accept_rate,
        name="max_uncertainty_escalation_final_false_accept_rate",
    )
    max_uncertainty_escalation_false_accept_delta = _validate_optional_finite_float(
        max_uncertainty_escalation_false_accept_delta,
        name="max_uncertainty_escalation_false_accept_delta",
    )
    max_covariance_maha_last_auroc_drop = _validate_optional_non_negative_float(
        max_covariance_maha_last_auroc_drop,
        name="max_covariance_maha_last_auroc_drop",
    )
    manifest_fingerprint_workers = _validate_positive_int(
        manifest_fingerprint_workers,
        name="manifest_fingerprint_workers",
    )
    verification_context = ArtifactVerificationContext(
        fingerprint_cache=fingerprint_cache,
        json_cache=json_cache,
        json_cache_stats=json_cache_stats,
    )
    cache = verification_context.fingerprint_cache
    payload_cache = verification_context.json_cache
    payload_cache_stats = verification_context.json_cache_stats
    route_registry_path = readiness_registry_path if route_registry_path is None else route_registry_path
    performance_registry_path = (
        readiness_registry_path if performance_registry_path is None else performance_registry_path
    )
    profile, profile_values, profile_applied = _apply_runtime_profile(
        runtime_profile,
        {
            "inside_trigger_budget_policy": inside_trigger_budget_policy,
            "max_inside_sample_count_ratio": max_inside_sample_count_ratio,
            "max_inside_generation_seconds_ratio": max_inside_generation_seconds_ratio,
            "max_mean_attempted_route_count": max_mean_attempted_route_count,
            "max_retrieval_use_rate": max_retrieval_use_rate,
        },
    )
    inside_trigger_budget_policy = profile_values["inside_trigger_budget_policy"]
    max_inside_sample_count_ratio = profile_values["max_inside_sample_count_ratio"]
    max_inside_generation_seconds_ratio = profile_values["max_inside_generation_seconds_ratio"]
    max_mean_attempted_route_count = profile_values["max_mean_attempted_route_count"]
    max_retrieval_use_rate = profile_values["max_retrieval_use_rate"]
    inside_trigger_budget_policy = _normalize_inside_trigger_budget_policy(
        inside_trigger_budget_policy
    )
    adapter_profile_name, adapter_profile_routes = _adapter_family_profile_routes(adapter_family_profile)
    required_adapter_routes = _merge_routes(adapter_profile_routes, required_adapter_routes)
    adapter_profile_requires_world_model = adapter_family_profile_requires_state_transition_world_model(
        adapter_profile_name
    )
    require_state_transition_world_model = bool(
        require_state_transition_world_model or adapter_profile_requires_world_model
    )
    if adapter_profile_name is not None and adapter_family_matrix_path is None:
        raise ValueError("adapter_family_profile requires adapter_family_matrix_path.")
    min_triple_extraction_corpora = _validate_optional_non_negative_int(
        min_triple_extraction_corpora,
        name="min_triple_extraction_corpora",
    )
    min_triple_extraction_distinct_predicates = _validate_optional_non_negative_int(
        min_triple_extraction_distinct_predicates,
        name="min_triple_extraction_distinct_predicates",
    )
    product_trace_replay_workflow_source = _resolve_product_trace_replay_workflow_source(
        product_trace_replay_workflow_path=product_trace_replay_workflow_path,
        product_trace_replay_workflow_registry_path=(
            product_trace_replay_workflow_registry_path
            if product_trace_replay_workflow_key is not None
            else None
        ),
        product_trace_replay_workflow_key=product_trace_replay_workflow_key,
        default_registry_path=readiness_registry_path,
    )
    product_trace_replay_workflow = _product_trace_replay_workflow_gate(
        product_trace_replay_workflow_source=product_trace_replay_workflow_source,
        selector_replay_report_path=selector_replay_report_path,
        product_runtime_drift_report_path=product_runtime_drift_report_path,
        require_action_audit_gate=require_product_trace_action_audit_gate,
        require_action_execution_gate=require_product_trace_action_execution_gate,
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        verification_context=verification_context,
    )
    if product_trace_replay_workflow is not None:
        if selector_replay_report_path is None and product_trace_replay_workflow.get(
            "selector_replay_report_path"
        ):
            selector_replay_report_path = str(
                product_trace_replay_workflow["selector_replay_report_path"]
            )
        if product_runtime_drift_report_path is None and product_trace_replay_workflow.get(
            "product_runtime_drift_report_path"
        ):
            product_runtime_drift_report_path = str(
                product_trace_replay_workflow["product_runtime_drift_report_path"]
            )
    selfcheck_signal_fusion_workflow_source = _resolve_selfcheck_signal_fusion_workflow_source(
        selfcheck_signal_fusion_workflow_path=selfcheck_signal_fusion_workflow_path,
        selfcheck_signal_fusion_workflow_registry_path=(
            selfcheck_signal_fusion_workflow_registry_path
            if selfcheck_signal_fusion_workflow_key is not None
            else None
        ),
        selfcheck_signal_fusion_workflow_key=selfcheck_signal_fusion_workflow_key,
        default_registry_path=readiness_registry_path,
    )
    selfcheck_signal_fusion_workflow = _selfcheck_signal_fusion_workflow_gate(
        selfcheck_signal_fusion_workflow_source=selfcheck_signal_fusion_workflow_source,
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        verification_context=verification_context,
    )
    uncertainty_escalation_workflow_source = _resolve_uncertainty_escalation_workflow_source(
        uncertainty_escalation_workflow_path=uncertainty_escalation_workflow_path,
        uncertainty_escalation_workflow_registry_path=(
            uncertainty_escalation_workflow_registry_path
            if uncertainty_escalation_workflow_key is not None
            else None
        ),
        uncertainty_escalation_workflow_key=uncertainty_escalation_workflow_key,
        default_registry_path=readiness_registry_path,
    )
    uncertainty_escalation_workflow = _uncertainty_escalation_workflow_gate(
        uncertainty_escalation_workflow_source=uncertainty_escalation_workflow_source,
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        min_records=min_uncertainty_escalation_records,
        min_trigger_rate=min_uncertainty_escalation_trigger_rate,
        min_retrieval_evidence_rate=min_uncertainty_escalation_retrieval_evidence_rate,
        max_final_false_accept_rate=max_uncertainty_escalation_final_false_accept_rate,
        max_false_accept_delta=max_uncertainty_escalation_false_accept_delta,
        verification_context=verification_context,
    )
    feedback_policy_workflow_source = _resolve_feedback_policy_workflow_source(
        feedback_policy_workflow_path=feedback_policy_workflow_path,
        feedback_policy_workflow_registry_path=(
            feedback_policy_workflow_registry_path
            if feedback_policy_workflow_key is not None
            else None
        ),
        feedback_policy_workflow_key=feedback_policy_workflow_key,
        default_registry_path=readiness_registry_path,
    )
    feedback_policy_workflow = _feedback_policy_workflow_gate(
        feedback_policy_workflow_source=feedback_policy_workflow_source,
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        min_matched_feedback_count=feedback_policy_min_matched_feedback_count,
        min_safety_coverage=feedback_policy_min_safety_coverage,
        max_unknown_safety_issue_rate=feedback_policy_max_unknown_safety_issue_rate,
        verification_context=verification_context,
    )
    readiness = compare_readiness_baselines(
        registry_path=readiness_registry_path,
        baseline_keys=readiness_baseline_keys,
        recursive=recursive,
        allow_unverified=allow_unverified,
        inside_trigger_budget_policy=inside_trigger_budget_policy,
        min_best_quality_auroc=min_best_quality_auroc,
        max_uncached_forward_seconds=max_uncached_forward_seconds,
        max_cache_only_seconds=max_cache_only_seconds,
        max_recommended_runtime_seconds=max_recommended_runtime_seconds,
        max_covariance_maha_last_auroc_drop=max_covariance_maha_last_auroc_drop,
        max_inside_sample_count_ratio=max_inside_sample_count_ratio,
        max_inside_generation_seconds_ratio=max_inside_generation_seconds_ratio,
        notes=("release candidate readiness comparison",),
        fingerprint_cache=cache,
        json_cache=payload_cache,
        json_cache_stats=payload_cache_stats,
    )
    route = compare_route_baselines(
        registry_path=route_registry_path,
        baseline_keys=route_baseline_keys,
        recursive=recursive,
        allow_unverified=allow_unverified,
        min_selected=min_selected,
        min_decision_accuracy=min_decision_accuracy,
        max_false_supported_rate=max_false_supported_rate,
        min_false_refuted_rate=min_false_refuted_rate,
        max_verified_false_alarm=max_verified_false_alarm,
        min_verified_detection=min_verified_detection,
        max_mean_duration_seconds=max_mean_duration_seconds,
        max_p99_duration_seconds=max_p99_duration_seconds,
        max_max_duration_seconds=max_max_duration_seconds,
        max_mean_attempted_route_count=max_mean_attempted_route_count,
        max_retrieval_use_rate=max_retrieval_use_rate,
        max_runtime_total_seconds=max_runtime_total_seconds,
        max_retrieval_hit_count=max_retrieval_hit_count,
        min_claims_cache_hit_rate=min_claims_cache_hit_rate,
        min_verifier_trace_cache_hit_rate=min_verifier_trace_cache_hit_rate,
        min_covered_fact_properties=min_covered_fact_properties,
        min_covered_fact_property_records=min_covered_fact_property_records,
        min_covered_fact_property_source_documents=min_covered_fact_property_source_documents,
        min_covered_fact_property_decision_accuracy=min_covered_fact_property_decision_accuracy,
        max_covered_fact_property_false_supported_rate=max_covered_fact_property_false_supported_rate,
        min_covered_fact_property_false_refuted_rate=min_covered_fact_property_false_refuted_rate,
        require_non_oracle_evidence=require_non_oracle_evidence,
        require_retrieval_provenance_filter=require_retrieval_provenance_filter,
        required_retrieval_source_prefixes=required_retrieval_source_prefixes,
        required_retrieval_metadata=required_retrieval_metadata,
        min_retrieval_filter_score=min_retrieval_filter_score,
        require_retrieval_stress_control=require_retrieval_stress_control,
        retrieval_stress_manifest=retrieval_stress_manifest,
        min_stress_false_supported_rate=min_stress_false_supported_rate,
        max_stress_false_refuted_rate=max_stress_false_refuted_rate,
        notes=("release candidate route comparison",),
        fingerprint_cache=cache,
        json_cache=payload_cache,
        json_cache_stats=payload_cache_stats,
    )
    raw_candidate = _release_candidate(readiness, route)
    ordinary_required_routes = _required_route_baseline_gate(
        route_registry_path=route_registry_path,
        required_route_baseline_keys=ordinary_required_route_keys,
        recursive=recursive,
        allow_unverified=allow_unverified,
        min_selected=required_route_min_selected,
        min_decision_accuracy=required_route_min_decision_accuracy,
        max_false_supported_rate=required_route_max_false_supported_rate,
        min_false_refuted_rate=required_route_min_false_refuted_rate,
        max_verified_false_alarm=required_route_max_verified_false_alarm,
        min_verified_detection=required_route_min_verified_detection,
        max_mean_duration_seconds=required_route_max_mean_duration_seconds,
        max_p99_duration_seconds=required_route_max_p99_duration_seconds,
        max_max_duration_seconds=required_route_max_max_duration_seconds,
        max_mean_attempted_route_count=required_route_max_mean_attempted_route_count,
        max_retrieval_use_rate=required_route_max_retrieval_use_rate,
        max_runtime_total_seconds=required_route_max_runtime_total_seconds,
        max_retrieval_hit_count=required_route_max_retrieval_hit_count,
        min_claims_cache_hit_rate=required_route_min_claims_cache_hit_rate,
        min_verifier_trace_cache_hit_rate=required_route_min_verifier_trace_cache_hit_rate,
        min_covered_fact_properties=required_route_min_covered_fact_properties,
        min_covered_fact_property_records=required_route_min_covered_fact_property_records,
        min_covered_fact_property_source_documents=required_route_min_covered_fact_property_source_documents,
        min_covered_fact_property_decision_accuracy=required_route_min_covered_fact_property_decision_accuracy,
        max_covered_fact_property_false_supported_rate=(
            required_route_max_covered_fact_property_false_supported_rate
        ),
        min_covered_fact_property_false_refuted_rate=required_route_min_covered_fact_property_false_refuted_rate,
        require_non_oracle_evidence=required_route_require_non_oracle_evidence,
        require_retrieval_provenance_filter=required_route_require_retrieval_provenance_filter,
        required_retrieval_source_prefixes=required_route_required_retrieval_source_prefixes,
        required_retrieval_metadata=required_route_required_retrieval_metadata,
        min_retrieval_filter_score=required_route_min_retrieval_filter_score,
        require_retrieval_stress_control=required_route_require_retrieval_stress_control,
        retrieval_stress_manifest=required_route_retrieval_stress_manifest,
        min_stress_false_supported_rate=required_route_min_stress_false_supported_rate,
        max_stress_false_refuted_rate=required_route_max_stress_false_refuted_rate,
        fingerprint_cache=cache,
        json_cache=payload_cache,
        json_cache_stats=payload_cache_stats,
    )
    structured_fact_routes = _required_route_baseline_gate(
        route_registry_path=route_registry_path,
        required_route_baseline_keys=structured_fact_route_keys,
        recursive=recursive,
        allow_unverified=allow_unverified,
        min_selected=_first_set(
            structured_fact_robustness_min_selected,
            required_route_min_selected,
        ),
        min_decision_accuracy=_first_set(
            structured_fact_robustness_min_decision_accuracy,
            required_route_min_decision_accuracy,
        ),
        max_false_supported_rate=_first_set(
            structured_fact_robustness_max_false_supported_rate,
            required_route_max_false_supported_rate,
        ),
        min_false_refuted_rate=_first_set(
            structured_fact_robustness_min_false_refuted_rate,
            required_route_min_false_refuted_rate,
        ),
        max_verified_false_alarm=None,
        min_verified_detection=None,
        max_mean_duration_seconds=None,
        max_p99_duration_seconds=None,
        max_max_duration_seconds=None,
        max_mean_attempted_route_count=None,
        max_retrieval_use_rate=None,
        max_runtime_total_seconds=None,
        max_retrieval_hit_count=None,
        min_claims_cache_hit_rate=None,
        min_verifier_trace_cache_hit_rate=None,
        min_covered_fact_properties=_first_set(
            structured_fact_robustness_min_covered_fact_properties,
            required_route_min_covered_fact_properties,
        ),
        min_covered_fact_property_records=_first_set(
            structured_fact_robustness_min_covered_fact_property_records,
            required_route_min_covered_fact_property_records,
        ),
        min_covered_fact_property_source_documents=_first_set(
            structured_fact_robustness_min_covered_fact_property_source_documents,
            required_route_min_covered_fact_property_source_documents,
        ),
        min_covered_fact_property_decision_accuracy=_first_set(
            structured_fact_robustness_min_covered_fact_property_decision_accuracy,
            required_route_min_covered_fact_property_decision_accuracy,
        ),
        max_covered_fact_property_false_supported_rate=_first_set(
            structured_fact_robustness_max_covered_fact_property_false_supported_rate,
            required_route_max_covered_fact_property_false_supported_rate,
        ),
        min_covered_fact_property_false_refuted_rate=_first_set(
            structured_fact_robustness_min_covered_fact_property_false_refuted_rate,
            required_route_min_covered_fact_property_false_refuted_rate,
        ),
        require_non_oracle_evidence=False,
        require_retrieval_provenance_filter=False,
        required_retrieval_source_prefixes=(),
        required_retrieval_metadata=None,
        min_retrieval_filter_score=None,
        require_retrieval_stress_control=False,
        retrieval_stress_manifest=None,
        min_stress_false_supported_rate=None,
        max_stress_false_refuted_rate=None,
        fingerprint_cache=cache,
        json_cache=payload_cache,
        json_cache_stats=payload_cache_stats,
    )
    required_routes = _combine_required_route_baseline_gates(
        route_registry_path=route_registry_path,
        required_route_baseline_keys=required_route_baseline_keys,
        ordinary_gate=ordinary_required_routes,
        structured_fact_gate=structured_fact_routes,
        ordinary_required_route_keys=ordinary_required_route_keys,
        structured_fact_route_keys=structured_fact_route_keys,
    )
    adapter_family = _adapter_family_matrix_gate(
        adapter_family_matrix_path=adapter_family_matrix_path,
        required_routes=required_adapter_routes,
        require_state_transition_world_model=bool(require_state_transition_world_model),
        verification_context=verification_context,
    )
    triple_extraction_fixture_matrix_source = _resolve_triple_extraction_fixture_matrix_source(
        triple_extraction_fixture_matrix_path=triple_extraction_fixture_matrix_path,
        triple_extraction_fixture_matrix_registry_path=(
            triple_extraction_fixture_matrix_registry_path
            if triple_extraction_fixture_matrix_key is not None
            else None
        ),
        triple_extraction_fixture_matrix_key=triple_extraction_fixture_matrix_key,
        default_registry_path=readiness_registry_path,
    )
    triple_extraction_fixture_matrix = _triple_extraction_fixture_matrix_gate(
        triple_extraction_fixture_matrix_source=triple_extraction_fixture_matrix_source,
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        min_corpora=min_triple_extraction_corpora,
        min_distinct_predicates=min_triple_extraction_distinct_predicates,
        min_external_prediction_count=min_triple_extraction_external_prediction_count,
        min_external_prediction_corpora=min_triple_extraction_external_prediction_corpora,
        min_mean_best_external_f1=min_triple_extraction_mean_best_external_f1,
        verification_context=verification_context,
    )
    counterfactual_verification_source = _resolve_counterfactual_verification_source(
        counterfactual_verification_report_path=counterfactual_verification_report_path,
        counterfactual_verification_registry_path=(
            counterfactual_verification_registry_path
            if counterfactual_verification_key is not None
            else None
        ),
        counterfactual_verification_key=counterfactual_verification_key,
        default_registry_path=readiness_registry_path,
    )
    counterfactual_verification = _counterfactual_verification_gate(
        counterfactual_verification_source=counterfactual_verification_source,
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        min_records=min_counterfactual_verification_records,
        min_pass_rate=min_counterfactual_verification_pass_rate,
        max_false_invariance_rate=max_counterfactual_verification_false_invariance_rate,
        verification_context=verification_context,
    )
    performance = _performance_baseline_gate(
        performance_registry_path=performance_registry_path,
        performance_baseline_key=performance_baseline_key,
        recursive=recursive,
        allow_unverified=allow_unverified,
        candidate=raw_candidate,
        require_score_dump_cache=require_performance_score_dump_cache,
        min_score_dump_cache_jsonl_view_hit_rate=min_performance_score_dump_cache_jsonl_view_hit_rate,
        drift_baseline_key=performance_drift_baseline_key,
        max_uncached_total_seconds_ratio=max_performance_uncached_total_seconds_ratio,
        max_cached_total_seconds_ratio=max_performance_cached_total_seconds_ratio,
        max_cache_only_total_seconds_ratio=max_performance_cache_only_total_seconds_ratio,
        max_score_dump_cache_jsonl_view_hit_rate_drop=(
            max_performance_score_dump_cache_jsonl_view_hit_rate_drop
        ),
        max_covariance_maha_last_auroc_drop=max_covariance_maha_last_auroc_drop,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        verification_context=verification_context,
    )
    selector_replay = _selector_replay_gate(
        selector_replay_report_path=selector_replay_report_path,
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        verification_context=verification_context,
    )
    product_runtime_drift = _product_runtime_drift_gate(
        product_runtime_drift_report_path=product_runtime_drift_report_path,
        require_promotion_evidence=require_product_runtime_drift_promotion_evidence,
        require_pre_generation_evidence=require_product_runtime_drift_pre_generation_evidence,
        require_claim_factuality_evidence=(
            require_product_runtime_drift_claim_factuality_evidence
        ),
        require_claim_risk_localization_evidence=(
            require_product_runtime_drift_claim_risk_localization_evidence
        ),
        require_counterfactual_evidence=require_product_runtime_drift_counterfactual_evidence,
        require_fact_selfcheck_gate_evidence=(
            require_product_runtime_drift_fact_selfcheck_gate_evidence
        ),
        require_triple_audit_evidence=require_product_runtime_drift_triple_audit_evidence,
        require_covered_fact_property_evidence=(
            require_product_runtime_drift_covered_fact_property_evidence
        ),
        require_action_gate_evidence=require_product_runtime_drift_action_gate_evidence,
        require_world_model_action_gate_evidence=(
            require_product_runtime_drift_world_model_action_gate_evidence
        ),
        require_world_model_rollout_evidence=(
            require_product_runtime_drift_world_model_rollout_evidence
        ),
        require_action_receipts_evidence=(
            require_product_runtime_drift_action_receipts_evidence
        ),
        require_receipt_claim_support_evidence=(
            require_product_runtime_drift_receipt_claim_support_evidence
        ),
        require_trajectory_audit_evidence=(
            require_product_runtime_drift_trajectory_audit_evidence
        ),
        require_provenance_evidence=require_product_runtime_drift_provenance_evidence,
        require_citation_integrity_evidence=(
            require_product_runtime_drift_citation_integrity_evidence
        ),
        require_evidence_quality_evidence=(
            require_product_runtime_drift_evidence_quality_evidence
        ),
        require_metacognition_evidence=(
            require_product_runtime_drift_metacognition_evidence
        ),
        require_evidence_handoff_evidence=(
            require_product_runtime_drift_evidence_handoff_evidence
        ),
        require_world_model_evidence=require_product_runtime_drift_world_model_evidence,
        require_context_sensitivity_evidence=(
            require_product_runtime_drift_context_sensitivity_evidence
        ),
        require_evidence_alignment_evidence=(
            require_product_runtime_drift_evidence_alignment_evidence
        ),
        require_counterfactual_robustness_evidence=(
            require_product_runtime_drift_counterfactual_robustness_evidence
        ),
        require_frontier_release_evidence=(
            require_product_runtime_drift_frontier_release_evidence
        ),
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        verification_context=verification_context,
    )
    release_efficiency = _release_efficiency_gate(
        release_efficiency_report_path=release_efficiency_report_path,
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        verification_context=verification_context,
    )
    external_evidence_baseline_comparison_source = (
        _resolve_external_evidence_baseline_comparison_source(
            external_evidence_baseline_comparison_path=external_evidence_baseline_comparison_path,
            external_evidence_baseline_comparison_registry_path=(
                external_evidence_baseline_comparison_registry_path
            ),
            external_evidence_baseline_comparison_key=external_evidence_baseline_comparison_key,
            default_registry_path=readiness_registry_path,
        )
    )
    external_evidence_baseline_comparison = _external_evidence_baseline_comparison_gate(
        external_evidence_baseline_comparison_source=external_evidence_baseline_comparison_source,
        verification_context=verification_context,
    )
    pre_generation_probe_comparison_source = _resolve_pre_generation_probe_comparison_source(
        pre_generation_probe_comparison_path=pre_generation_probe_comparison_path,
        pre_generation_probe_comparison_registry_path=(
            pre_generation_probe_comparison_registry_path
            if pre_generation_probe_comparison_key is not None
            else None
        ),
        pre_generation_probe_comparison_key=pre_generation_probe_comparison_key,
        default_registry_path=readiness_registry_path,
    )
    pre_generation_probe_comparison = _pre_generation_probe_comparison_gate(
        pre_generation_probe_comparison_source=pre_generation_probe_comparison_source,
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        verification_context=verification_context,
    )
    claim_factuality_probe_comparison_source = _resolve_claim_factuality_probe_comparison_source(
        claim_factuality_probe_comparison_path=claim_factuality_probe_comparison_path,
        claim_factuality_probe_comparison_registry_path=(
            claim_factuality_probe_comparison_registry_path
            if claim_factuality_probe_comparison_key is not None
            else None
        ),
        claim_factuality_probe_comparison_key=claim_factuality_probe_comparison_key,
        default_registry_path=readiness_registry_path,
    )
    claim_factuality_probe_comparison = _claim_factuality_probe_comparison_gate(
        claim_factuality_probe_comparison_source=claim_factuality_probe_comparison_source,
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        verification_context=verification_context,
    )
    frontier_release_evidence_source = _resolve_frontier_release_evidence_source(
        frontier_release_evidence_path=frontier_release_evidence_path,
        frontier_release_evidence_registry_path=(
            frontier_release_evidence_registry_path
            if frontier_release_evidence_key is not None
            else None
        ),
        frontier_release_evidence_key=frontier_release_evidence_key,
        default_registry_path=readiness_registry_path,
    )
    frontier_release_evidence = _frontier_release_evidence_gate(
        frontier_release_evidence_source=frontier_release_evidence_source,
        require_input_manifests=require_frontier_release_input_manifests,
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        verification_context=verification_context,
    )
    world_model_signal_workflow_source = _resolve_world_model_signal_workflow_source(
        world_model_signal_workflow_path=world_model_signal_workflow_path,
        world_model_signal_workflow_registry_path=(
            world_model_signal_workflow_registry_path
            if world_model_signal_workflow_key is not None
            else None
        ),
        world_model_signal_workflow_key=world_model_signal_workflow_key,
        default_registry_path=readiness_registry_path,
    )
    world_model_signal_workflow = _world_model_signal_workflow_gate(
        world_model_signal_workflow_source=world_model_signal_workflow_source,
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        verification_context=verification_context,
    )
    context_sensitivity_workflow_source = _resolve_context_sensitivity_workflow_source(
        context_sensitivity_workflow_path=context_sensitivity_workflow_path,
        context_sensitivity_workflow_registry_path=(
            context_sensitivity_workflow_registry_path
            if context_sensitivity_workflow_key is not None
            else None
        ),
        context_sensitivity_workflow_key=context_sensitivity_workflow_key,
        default_registry_path=readiness_registry_path,
    )
    context_sensitivity_workflow = _context_sensitivity_workflow_gate(
        context_sensitivity_workflow_source=context_sensitivity_workflow_source,
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        verification_context=verification_context,
    )
    mechanism_handoff_evidence_bundle_source = _resolve_mechanism_handoff_evidence_bundle_source(
        mechanism_handoff_evidence_bundle_path=mechanism_handoff_evidence_bundle_path,
        mechanism_handoff_evidence_bundle_registry_path=(
            mechanism_handoff_evidence_bundle_registry_path
            if mechanism_handoff_evidence_bundle_key is not None
            else None
        ),
        mechanism_handoff_evidence_bundle_key=mechanism_handoff_evidence_bundle_key,
        default_registry_path=readiness_registry_path,
    )
    mechanism_handoff_evidence_bundle = _mechanism_handoff_evidence_bundle_gate(
        mechanism_handoff_evidence_bundle_source=mechanism_handoff_evidence_bundle_source,
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        verification_context=verification_context,
    )
    pathway_intervention_workflow_source = _resolve_pathway_intervention_workflow_source(
        pathway_intervention_workflow_path=pathway_intervention_workflow_path,
        pathway_intervention_workflow_registry_path=(
            pathway_intervention_workflow_registry_path
            if pathway_intervention_workflow_key is not None
            else None
        ),
        pathway_intervention_workflow_key=pathway_intervention_workflow_key,
        default_registry_path=readiness_registry_path,
    )
    pathway_intervention_workflow = _pathway_intervention_workflow_gate(
        pathway_intervention_workflow_source=pathway_intervention_workflow_source,
        recursive=recursive,
        allow_unverified=allow_unverified,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
        verification_context=verification_context,
    )
    decision = _decision(
        readiness,
        route,
        raw_candidate,
        performance,
        adapter_family,
        triple_extraction_fixture_matrix,
        counterfactual_verification,
        required_routes,
        product_trace_replay_workflow,
        selector_replay,
        product_runtime_drift,
        release_efficiency,
        external_evidence_baseline_comparison,
        pre_generation_probe_comparison,
        claim_factuality_probe_comparison,
        frontier_release_evidence,
        world_model_signal_workflow,
        context_sensitivity_workflow,
        mechanism_handoff_evidence_bundle,
        pathway_intervention_workflow,
        selfcheck_signal_fusion_workflow,
        uncertainty_escalation_workflow,
        feedback_policy_workflow,
    )
    candidate = (
        _candidate_with_gates(
            raw_candidate,
            performance,
            adapter_family,
            triple_extraction_fixture_matrix,
            counterfactual_verification,
            required_routes,
            product_trace_replay_workflow,
            selector_replay,
            product_runtime_drift,
            release_efficiency,
            external_evidence_baseline_comparison,
            pre_generation_probe_comparison,
            claim_factuality_probe_comparison,
            frontier_release_evidence,
            world_model_signal_workflow,
            context_sensitivity_workflow,
            mechanism_handoff_evidence_bundle,
            pathway_intervention_workflow,
            selfcheck_signal_fusion_workflow,
            uncertainty_escalation_workflow,
            feedback_policy_workflow,
        )
        if decision["status"] == "promote"
        else None
    )
    return {
        "schema_version": 1,
        "workflow": "release_candidate_comparison",
        "summary": {
            "artifact_json_cache": verification_context.json_cache_summary(),
        },
        "config": {
            "readiness_registry": str(readiness_registry_path),
            "route_registry": str(route_registry_path),
            "performance_registry": str(performance_registry_path),
            "release_policy_profile": release_policy_profile,
            "release_policy_profile_applied_defaults": release_policy_applied,
            "readiness_baseline_keys": list(readiness_baseline_keys),
            "route_baseline_keys": list(route_baseline_keys),
            "required_route_baseline_keys": list(required_route_baseline_keys),
            "require_structured_fact_robustness": require_structured_fact_robustness,
            "structured_fact_canonical_route_key": structured_fact_canonical_route_key,
            "structured_fact_paraphrase_route_key": structured_fact_paraphrase_route_key,
            "performance_baseline_key": performance_baseline_key,
            "selector_replay_report": (
                None if selector_replay_report_path is None else str(selector_replay_report_path)
            ),
            "product_runtime_drift_report": (
                None
                if product_runtime_drift_report_path is None
                else str(product_runtime_drift_report_path)
            ),
            "require_product_runtime_drift_promotion_evidence": bool(
                require_product_runtime_drift_promotion_evidence
            ),
            "require_product_runtime_drift_pre_generation_evidence": bool(
                require_product_runtime_drift_pre_generation_evidence
            ),
            "require_product_runtime_drift_claim_factuality_evidence": bool(
                require_product_runtime_drift_claim_factuality_evidence
            ),
            "require_product_runtime_drift_claim_risk_localization_evidence": bool(
                require_product_runtime_drift_claim_risk_localization_evidence
            ),
            "require_product_runtime_drift_counterfactual_evidence": bool(
                require_product_runtime_drift_counterfactual_evidence
            ),
            "require_product_runtime_drift_fact_selfcheck_gate_evidence": bool(
                require_product_runtime_drift_fact_selfcheck_gate_evidence
            ),
            "require_product_runtime_drift_triple_audit_evidence": bool(
                require_product_runtime_drift_triple_audit_evidence
            ),
            "require_product_runtime_drift_covered_fact_property_evidence": bool(
                require_product_runtime_drift_covered_fact_property_evidence
            ),
            "require_product_runtime_drift_action_gate_evidence": bool(
                require_product_runtime_drift_action_gate_evidence
            ),
            "require_product_runtime_drift_world_model_action_gate_evidence": bool(
                require_product_runtime_drift_world_model_action_gate_evidence
            ),
            "require_product_runtime_drift_world_model_rollout_evidence": bool(
                require_product_runtime_drift_world_model_rollout_evidence
            ),
            "require_product_runtime_drift_action_receipts_evidence": bool(
                require_product_runtime_drift_action_receipts_evidence
            ),
            "require_product_runtime_drift_receipt_claim_support_evidence": bool(
                require_product_runtime_drift_receipt_claim_support_evidence
            ),
            "require_product_runtime_drift_trajectory_audit_evidence": bool(
                require_product_runtime_drift_trajectory_audit_evidence
            ),
            "require_product_runtime_drift_provenance_evidence": bool(
                require_product_runtime_drift_provenance_evidence
            ),
            "require_product_runtime_drift_citation_integrity_evidence": bool(
                require_product_runtime_drift_citation_integrity_evidence
            ),
            "require_product_runtime_drift_evidence_quality_evidence": bool(
                require_product_runtime_drift_evidence_quality_evidence
            ),
            "require_product_runtime_drift_metacognition_evidence": bool(
                require_product_runtime_drift_metacognition_evidence
            ),
            "require_product_runtime_drift_evidence_handoff_evidence": bool(
                require_product_runtime_drift_evidence_handoff_evidence
            ),
            "require_product_runtime_drift_world_model_evidence": bool(
                require_product_runtime_drift_world_model_evidence
            ),
            "require_product_runtime_drift_context_sensitivity_evidence": bool(
                require_product_runtime_drift_context_sensitivity_evidence
            ),
            "require_product_runtime_drift_evidence_alignment_evidence": bool(
                require_product_runtime_drift_evidence_alignment_evidence
            ),
            "require_product_runtime_drift_counterfactual_robustness_evidence": bool(
                require_product_runtime_drift_counterfactual_robustness_evidence
            ),
            "require_product_runtime_drift_frontier_release_evidence": bool(
                require_product_runtime_drift_frontier_release_evidence
            ),
            "require_frontier_release_input_manifests": bool(
                require_frontier_release_input_manifests
            ),
            "require_product_trace_action_audit_gate": bool(
                require_product_trace_action_audit_gate
            ),
            "require_product_trace_action_execution_gate": bool(
                require_product_trace_action_execution_gate
            ),
            "release_efficiency_report": (
                None
                if release_efficiency_report_path is None
                else str(release_efficiency_report_path)
            ),
            "external_evidence_baseline_comparison": (
                None
                if external_evidence_baseline_comparison_source is None
                else str(external_evidence_baseline_comparison_source["path"])
            ),
            "external_evidence_baseline_comparison_registry": (
                None
                if external_evidence_baseline_comparison_source is None
                else external_evidence_baseline_comparison_source.get("registry")
            ),
            "external_evidence_baseline_comparison_key": (
                external_evidence_baseline_comparison_key
            ),
            "pre_generation_probe_comparison": (
                None
                if pre_generation_probe_comparison_source is None
                else str(pre_generation_probe_comparison_source["path"])
            ),
            "pre_generation_probe_comparison_registry": (
                None
                if pre_generation_probe_comparison_source is None
                else pre_generation_probe_comparison_source.get("registry")
            ),
            "pre_generation_probe_comparison_key": pre_generation_probe_comparison_key,
            "claim_factuality_probe_comparison": (
                None
                if claim_factuality_probe_comparison_source is None
                else str(claim_factuality_probe_comparison_source["path"])
            ),
            "claim_factuality_probe_comparison_registry": (
                None
                if claim_factuality_probe_comparison_source is None
                else claim_factuality_probe_comparison_source.get("registry")
            ),
            "claim_factuality_probe_comparison_key": claim_factuality_probe_comparison_key,
            "frontier_release_evidence": (
                None
                if frontier_release_evidence_source is None
                else str(frontier_release_evidence_source["path"])
            ),
            "frontier_release_evidence_registry": (
                None
                if frontier_release_evidence_source is None
                else frontier_release_evidence_source.get("registry")
            ),
            "frontier_release_evidence_key": frontier_release_evidence_key,
            "world_model_signal_workflow": (
                None
                if world_model_signal_workflow_source is None
                else str(world_model_signal_workflow_source["path"])
            ),
            "world_model_signal_workflow_registry": (
                None
                if world_model_signal_workflow_source is None
                else world_model_signal_workflow_source.get("registry")
            ),
            "world_model_signal_workflow_key": world_model_signal_workflow_key,
            "context_sensitivity_workflow": (
                None
                if context_sensitivity_workflow_source is None
                else str(context_sensitivity_workflow_source["path"])
            ),
            "context_sensitivity_workflow_registry": (
                None
                if context_sensitivity_workflow_source is None
                else context_sensitivity_workflow_source.get("registry")
            ),
            "context_sensitivity_workflow_key": context_sensitivity_workflow_key,
            "mechanism_handoff_evidence_bundle": (
                None
                if mechanism_handoff_evidence_bundle_source is None
                else str(mechanism_handoff_evidence_bundle_source["path"])
            ),
            "mechanism_handoff_evidence_bundle_registry": (
                None
                if mechanism_handoff_evidence_bundle_source is None
                else mechanism_handoff_evidence_bundle_source.get("registry")
            ),
            "mechanism_handoff_evidence_bundle_key": mechanism_handoff_evidence_bundle_key,
            "pathway_intervention_workflow": (
                None
                if pathway_intervention_workflow_source is None
                else str(pathway_intervention_workflow_source["path"])
            ),
            "pathway_intervention_workflow_registry": (
                None
                if pathway_intervention_workflow_source is None
                else pathway_intervention_workflow_source.get("registry")
            ),
            "pathway_intervention_workflow_key": pathway_intervention_workflow_key,
            "product_trace_replay_workflow": (
                None
                if product_trace_replay_workflow_source is None
                else str(product_trace_replay_workflow_source["path"])
            ),
            "product_trace_replay_workflow_registry": (
                None
                if product_trace_replay_workflow_source is None
                else product_trace_replay_workflow_source.get("registry")
            ),
            "product_trace_replay_workflow_key": product_trace_replay_workflow_key,
            "selfcheck_signal_fusion_workflow": (
                None
                if selfcheck_signal_fusion_workflow_source is None
                else str(selfcheck_signal_fusion_workflow_source["path"])
            ),
            "selfcheck_signal_fusion_workflow_registry": (
                None
                if selfcheck_signal_fusion_workflow_source is None
                else selfcheck_signal_fusion_workflow_source.get("registry")
            ),
            "selfcheck_signal_fusion_workflow_key": selfcheck_signal_fusion_workflow_key,
            "uncertainty_escalation_workflow": (
                None
                if uncertainty_escalation_workflow_source is None
                else str(uncertainty_escalation_workflow_source["path"])
            ),
            "uncertainty_escalation_workflow_registry": (
                None
                if uncertainty_escalation_workflow_source is None
                else uncertainty_escalation_workflow_source.get("registry")
            ),
            "uncertainty_escalation_workflow_key": uncertainty_escalation_workflow_key,
            "min_uncertainty_escalation_records": min_uncertainty_escalation_records,
            "min_uncertainty_escalation_trigger_rate": (
                min_uncertainty_escalation_trigger_rate
            ),
            "min_uncertainty_escalation_retrieval_evidence_rate": (
                min_uncertainty_escalation_retrieval_evidence_rate
            ),
            "max_uncertainty_escalation_final_false_accept_rate": (
                max_uncertainty_escalation_final_false_accept_rate
            ),
            "max_uncertainty_escalation_false_accept_delta": (
                max_uncertainty_escalation_false_accept_delta
            ),
            "feedback_policy_workflow": (
                None
                if feedback_policy_workflow_source is None
                else str(feedback_policy_workflow_source["path"])
            ),
            "feedback_policy_workflow_registry": (
                None
                if feedback_policy_workflow_source is None
                else feedback_policy_workflow_source.get("registry")
            ),
            "feedback_policy_workflow_key": feedback_policy_workflow_key,
            "feedback_policy_min_matched_feedback_count": feedback_policy_min_matched_feedback_count,
            "feedback_policy_min_safety_coverage": feedback_policy_min_safety_coverage,
            "feedback_policy_max_unknown_safety_issue_rate": feedback_policy_max_unknown_safety_issue_rate,
            "adapter_family_matrix": None if adapter_family_matrix_path is None else str(adapter_family_matrix_path),
            "adapter_family_profile": adapter_profile_name,
            "adapter_family_profile_required_routes": list(adapter_profile_routes),
            "adapter_family_profile_requires_state_transition_world_model": (
                adapter_profile_requires_world_model
            ),
            "required_adapter_routes": list(required_adapter_routes),
            "require_state_transition_world_model": bool(require_state_transition_world_model),
            "triple_extraction_fixture_matrix": (
                None
                if triple_extraction_fixture_matrix_source is None
                else str(triple_extraction_fixture_matrix_source["path"])
            ),
            "triple_extraction_fixture_matrix_registry": (
                None
                if triple_extraction_fixture_matrix_source is None
                else triple_extraction_fixture_matrix_source.get("registry")
            ),
            "triple_extraction_fixture_matrix_key": triple_extraction_fixture_matrix_key,
            "min_triple_extraction_corpora": min_triple_extraction_corpora,
            "min_triple_extraction_distinct_predicates": min_triple_extraction_distinct_predicates,
            "min_triple_extraction_external_prediction_count": (
                min_triple_extraction_external_prediction_count
            ),
            "min_triple_extraction_external_prediction_corpora": (
                min_triple_extraction_external_prediction_corpora
            ),
            "min_triple_extraction_mean_best_external_f1": (
                min_triple_extraction_mean_best_external_f1
            ),
            "counterfactual_verification_report": (
                None
                if counterfactual_verification_source is None
                else str(counterfactual_verification_source["path"])
            ),
            "counterfactual_verification_registry": (
                None
                if counterfactual_verification_source is None
                else counterfactual_verification_source.get("registry")
            ),
            "counterfactual_verification_key": counterfactual_verification_key,
            "min_counterfactual_verification_records": min_counterfactual_verification_records,
            "min_counterfactual_verification_pass_rate": min_counterfactual_verification_pass_rate,
            "max_counterfactual_verification_false_invariance_rate": (
                max_counterfactual_verification_false_invariance_rate
            ),
            "require_performance_score_dump_cache": require_performance_score_dump_cache,
            "min_performance_score_dump_cache_jsonl_view_hit_rate": (
                min_performance_score_dump_cache_jsonl_view_hit_rate
            ),
            "performance_drift_baseline_key": performance_drift_baseline_key,
            "max_performance_uncached_total_seconds_ratio": (
                max_performance_uncached_total_seconds_ratio
            ),
            "max_performance_cached_total_seconds_ratio": (
                max_performance_cached_total_seconds_ratio
            ),
            "max_performance_cache_only_total_seconds_ratio": (
                max_performance_cache_only_total_seconds_ratio
            ),
            "max_performance_score_dump_cache_jsonl_view_hit_rate_drop": (
                max_performance_score_dump_cache_jsonl_view_hit_rate_drop
            ),
            "recursive": recursive,
            "allow_unverified": allow_unverified,
            "manifest_fingerprint_workers": manifest_fingerprint_workers,
            "runtime_profile": None if profile is None else profile.name,
            "runtime_profile_defaults": None if profile is None else dict(profile.defaults),
            "runtime_profile_applied_defaults": profile_applied,
            "inside_trigger_budget_policy": inside_trigger_budget_policy,
            "min_best_quality_auroc": min_best_quality_auroc,
            "max_uncached_forward_seconds": max_uncached_forward_seconds,
            "max_cache_only_seconds": max_cache_only_seconds,
            "max_recommended_runtime_seconds": max_recommended_runtime_seconds,
            "max_covariance_maha_last_auroc_drop": max_covariance_maha_last_auroc_drop,
            "max_inside_sample_count_ratio": max_inside_sample_count_ratio,
            "max_inside_generation_seconds_ratio": max_inside_generation_seconds_ratio,
            "min_selected": min_selected,
            "min_decision_accuracy": min_decision_accuracy,
            "max_false_supported_rate": max_false_supported_rate,
            "min_false_refuted_rate": min_false_refuted_rate,
            "max_verified_false_alarm": max_verified_false_alarm,
            "min_verified_detection": min_verified_detection,
            "max_mean_duration_seconds": max_mean_duration_seconds,
            "max_p99_duration_seconds": max_p99_duration_seconds,
            "max_max_duration_seconds": max_max_duration_seconds,
            "max_mean_attempted_route_count": max_mean_attempted_route_count,
            "max_retrieval_use_rate": max_retrieval_use_rate,
            "max_runtime_total_seconds": max_runtime_total_seconds,
            "max_retrieval_hit_count": max_retrieval_hit_count,
            "min_claims_cache_hit_rate": min_claims_cache_hit_rate,
            "min_verifier_trace_cache_hit_rate": min_verifier_trace_cache_hit_rate,
            "min_covered_fact_properties": min_covered_fact_properties,
            "min_covered_fact_property_records": min_covered_fact_property_records,
            "min_covered_fact_property_source_documents": min_covered_fact_property_source_documents,
            "min_covered_fact_property_decision_accuracy": min_covered_fact_property_decision_accuracy,
            "max_covered_fact_property_false_supported_rate": max_covered_fact_property_false_supported_rate,
            "min_covered_fact_property_false_refuted_rate": min_covered_fact_property_false_refuted_rate,
            "require_non_oracle_evidence": require_non_oracle_evidence,
            "require_retrieval_provenance_filter": require_retrieval_provenance_filter,
            "required_retrieval_source_prefixes": list(required_retrieval_source_prefixes),
            "required_retrieval_metadata": dict(required_retrieval_metadata or {}),
            "min_retrieval_filter_score": min_retrieval_filter_score,
            "require_retrieval_stress_control": require_retrieval_stress_control,
            "retrieval_stress_manifest": None if retrieval_stress_manifest is None else str(retrieval_stress_manifest),
            "min_stress_false_supported_rate": min_stress_false_supported_rate,
            "max_stress_false_refuted_rate": max_stress_false_refuted_rate,
            "required_route_min_selected": required_route_min_selected,
            "required_route_min_decision_accuracy": required_route_min_decision_accuracy,
            "required_route_max_false_supported_rate": required_route_max_false_supported_rate,
            "required_route_min_false_refuted_rate": required_route_min_false_refuted_rate,
            "required_route_max_verified_false_alarm": required_route_max_verified_false_alarm,
            "required_route_min_verified_detection": required_route_min_verified_detection,
            "required_route_max_mean_duration_seconds": required_route_max_mean_duration_seconds,
            "required_route_max_p99_duration_seconds": required_route_max_p99_duration_seconds,
            "required_route_max_max_duration_seconds": required_route_max_max_duration_seconds,
            "required_route_max_mean_attempted_route_count": required_route_max_mean_attempted_route_count,
            "required_route_max_retrieval_use_rate": required_route_max_retrieval_use_rate,
            "required_route_max_runtime_total_seconds": required_route_max_runtime_total_seconds,
            "required_route_max_retrieval_hit_count": required_route_max_retrieval_hit_count,
            "required_route_min_claims_cache_hit_rate": required_route_min_claims_cache_hit_rate,
            "required_route_min_verifier_trace_cache_hit_rate": required_route_min_verifier_trace_cache_hit_rate,
            "required_route_min_covered_fact_properties": required_route_min_covered_fact_properties,
            "required_route_min_covered_fact_property_records": required_route_min_covered_fact_property_records,
            "required_route_min_covered_fact_property_source_documents": (
                required_route_min_covered_fact_property_source_documents
            ),
            "required_route_min_covered_fact_property_decision_accuracy": (
                required_route_min_covered_fact_property_decision_accuracy
            ),
            "required_route_max_covered_fact_property_false_supported_rate": (
                required_route_max_covered_fact_property_false_supported_rate
            ),
            "required_route_min_covered_fact_property_false_refuted_rate": (
                required_route_min_covered_fact_property_false_refuted_rate
            ),
            "structured_fact_robustness_min_selected": structured_fact_robustness_min_selected,
            "structured_fact_robustness_min_decision_accuracy": (
                structured_fact_robustness_min_decision_accuracy
            ),
            "structured_fact_robustness_max_false_supported_rate": (
                structured_fact_robustness_max_false_supported_rate
            ),
            "structured_fact_robustness_min_false_refuted_rate": (
                structured_fact_robustness_min_false_refuted_rate
            ),
            "structured_fact_robustness_min_covered_fact_properties": (
                structured_fact_robustness_min_covered_fact_properties
            ),
            "structured_fact_robustness_min_covered_fact_property_records": (
                structured_fact_robustness_min_covered_fact_property_records
            ),
            "structured_fact_robustness_min_covered_fact_property_source_documents": (
                structured_fact_robustness_min_covered_fact_property_source_documents
            ),
            "structured_fact_robustness_min_covered_fact_property_decision_accuracy": (
                structured_fact_robustness_min_covered_fact_property_decision_accuracy
            ),
            "structured_fact_robustness_max_covered_fact_property_false_supported_rate": (
                structured_fact_robustness_max_covered_fact_property_false_supported_rate
            ),
            "structured_fact_robustness_min_covered_fact_property_false_refuted_rate": (
                structured_fact_robustness_min_covered_fact_property_false_refuted_rate
            ),
            "required_route_require_non_oracle_evidence": required_route_require_non_oracle_evidence,
            "required_route_require_retrieval_provenance_filter": (
                required_route_require_retrieval_provenance_filter
            ),
            "required_route_required_retrieval_source_prefixes": list(
                required_route_required_retrieval_source_prefixes
            ),
            "required_route_required_retrieval_metadata": dict(
                required_route_required_retrieval_metadata or {}
            ),
            "required_route_min_retrieval_filter_score": required_route_min_retrieval_filter_score,
            "required_route_require_retrieval_stress_control": required_route_require_retrieval_stress_control,
            "required_route_retrieval_stress_manifest": (
                None
                if required_route_retrieval_stress_manifest is None
                else str(required_route_retrieval_stress_manifest)
            ),
            "required_route_min_stress_false_supported_rate": required_route_min_stress_false_supported_rate,
            "required_route_max_stress_false_refuted_rate": required_route_max_stress_false_refuted_rate,
        },
        "readiness_baseline_comparison": readiness,
        "route_baseline_comparison": route,
        "required_route_baseline_gate": required_routes,
        "performance_baseline_gate": performance,
        "product_trace_replay_workflow_gate": product_trace_replay_workflow,
        "selfcheck_signal_fusion_workflow_gate": selfcheck_signal_fusion_workflow,
        "uncertainty_escalation_workflow_gate": uncertainty_escalation_workflow,
        "feedback_policy_workflow_gate": feedback_policy_workflow,
        "selector_replay_gate": selector_replay,
        "product_runtime_drift_gate": product_runtime_drift,
        "release_efficiency_gate": release_efficiency,
        "external_evidence_baseline_comparison_gate": external_evidence_baseline_comparison,
        "pre_generation_probe_comparison_gate": pre_generation_probe_comparison,
        "claim_factuality_probe_comparison_gate": claim_factuality_probe_comparison,
        "frontier_release_evidence_gate": frontier_release_evidence,
        "world_model_signal_workflow_gate": world_model_signal_workflow,
        "context_sensitivity_workflow_gate": context_sensitivity_workflow,
        "mechanism_handoff_evidence_bundle_gate": mechanism_handoff_evidence_bundle,
        "pathway_intervention_workflow_gate": pathway_intervention_workflow,
        "adapter_family_matrix_gate": adapter_family,
        "triple_extraction_fixture_matrix_gate": triple_extraction_fixture_matrix,
        "counterfactual_verification_gate": counterfactual_verification,
        "release_candidate": candidate,
        "decision": decision,
        "notes": list(notes),
    }


def _release_candidate(
    readiness: Mapping[str, Any],
    route: Mapping[str, Any],
) -> dict[str, Any] | None:
    readiness_decision = _mapping(readiness.get("decision"))
    route_decision = _mapping(route.get("decision"))
    if readiness_decision.get("status") != "promote" or route_decision.get("status") != "promote":
        return None
    readiness_row = _recommended_row(readiness, readiness_decision.get("recommended_record"))
    route_row = _recommended_row(route, route_decision.get("recommended_record"))
    if not readiness_row or not route_row:
        return None
    route_property_summary = _covered_fact_property_summary(route_row)
    return {
        "readiness_record": readiness_row.get("record_key"),
        "route_record": route_row.get("record_key"),
        "model": readiness_row.get("model"),
        "runtime": {
            "layer": readiness_row.get("layer"),
            "batch_size": readiness_row.get("batch_size"),
            "hidden_state_capture": readiness_row.get("hidden_state_capture"),
            "covariance_mode": readiness_row.get("covariance_mode"),
            "covariance_low_rank": readiness_row.get("covariance_low_rank"),
            "max_batch_tokens": readiness_row.get("max_batch_tokens"),
            "prefix_kv_cache": readiness_row.get("prefix_kv_cache"),
            "max_workers": readiness_row.get("max_workers"),
            "inside_sampling": readiness_row.get("inside_sampling"),
            "inside_trigger_budget_sweep": readiness_row.get("inside_trigger_budget_sweep"),
            "inside_trigger_budget_policy": readiness_row.get("inside_trigger_budget_policy"),
            "performance_cell": readiness_row.get("recommended_performance_cell"),
            "benchmark_flags": readiness_row.get("benchmark_flags"),
        },
        "quality": {
            "best_quality_signal": readiness_row.get("best_quality_signal"),
            "quality_signals": readiness_row.get("quality_signals"),
            "truth_proj_auroc": readiness_row.get("truth_proj_auroc"),
            "covariance_tradeoff": readiness_row.get("covariance_tradeoff"),
            "covariance_tradeoff_gate": readiness_row.get("covariance_tradeoff_gate"),
        },
        "runtime_cost": {
            "uncached_forward_cost_seconds": readiness_row.get("uncached_forward_cost_seconds"),
            "uncached_forward_cost_source": readiness_row.get("uncached_forward_cost_source"),
            "cache_only_total_seconds": readiness_row.get("cache_only_total_seconds"),
            "recommended_runtime_seconds": readiness_row.get("recommended_runtime_seconds"),
            "recommended_runtime_cost_source": readiness_row.get(
                "recommended_runtime_cost_source"
            ),
            "inside_sampling_recommended_run": readiness_row.get("inside_sampling_recommended_run"),
            "inside_sampling_total_generated_samples": readiness_row.get(
                "inside_sampling_total_generated_samples"
            ),
            "inside_sampling_sample_count_ratio_to_baseline": readiness_row.get(
                "inside_sampling_sample_count_ratio_to_baseline"
            ),
            "inside_sampling_sample_count_ratio_to_reference": readiness_row.get(
                "inside_sampling_sample_count_ratio_to_reference"
            ),
            "inside_sampling_sample_count_ratio_for_gate": readiness_row.get(
                "inside_sampling_sample_count_ratio_for_gate"
            ),
            "inside_sampling_sample_count_ratio_source": readiness_row.get(
                "inside_sampling_sample_count_ratio_source"
            ),
            "inside_generation_seconds": readiness_row.get("inside_generation_seconds"),
            "inside_generation_seconds_ratio_to_baseline": readiness_row.get(
                "inside_generation_seconds_ratio_to_baseline"
            ),
            "inside_generation_seconds_ratio_to_reference": readiness_row.get(
                "inside_generation_seconds_ratio_to_reference"
            ),
            "inside_generation_seconds_ratio_for_gate": readiness_row.get(
                "inside_generation_seconds_ratio_for_gate"
            ),
            "inside_generation_seconds_ratio_source": readiness_row.get(
                "inside_generation_seconds_ratio_source"
            ),
            "inside_sampling_stop_reason_counts": readiness_row.get("inside_sampling_stop_reason_counts"),
            "inside_trigger_budget_id": readiness_row.get("inside_trigger_budget_id"),
            "inside_trigger_budget_policy": readiness_row.get("inside_trigger_budget_policy"),
            "inside_trigger_budget_derive_from_max_budget": readiness_row.get(
                "inside_trigger_budget_derive_from_max_budget"
            ),
        },
        "verifier_route": {
            "route": route_row.get("recommended_route"),
            "selected": route_row.get("selected"),
            "decision_accuracy": route_row.get("decision_accuracy"),
            "false_supported_rate": route_row.get("false_supported_rate"),
            "false_refuted_rate": route_row.get("false_refuted_rate"),
            "verified_false_alarm": route_row.get("verified_false_alarm"),
            "verified_detection": route_row.get("verified_detection"),
            "mean_duration_seconds": route_row.get("mean_duration_seconds"),
            "p99_duration_seconds": route_row.get("p99_duration_seconds"),
            "max_duration_seconds": route_row.get("max_duration_seconds"),
            "mean_attempted_route_count": route_row.get("mean_attempted_route_count"),
            "retrieval_use_rate": route_row.get("retrieval_use_rate"),
            "runtime_total_seconds": route_row.get("runtime_total_seconds"),
            "runtime_retrieval_hit_count": route_row.get("runtime_retrieval_hit_count"),
            "claims_cache_hit_rate": route_row.get("claims_cache_hit_rate"),
            "verifier_trace_cache_hit_rate": route_row.get("verifier_trace_cache_hit_rate"),
            "covered_fact_property_count": route_property_summary["count"],
            "covered_fact_properties": route_property_summary["properties"],
            "covered_fact_property_metrics": route_property_summary["metrics"],
        },
        "manifests": {
            "readiness_manifest": readiness_row.get("manifest_path"),
            "route_manifest": route_row.get("manifest_path"),
        },
    }


def _recommended_row(
    report: Mapping[str, Any],
    record_key: Any,
) -> dict[str, Any]:
    if record_key is None:
        return {}
    for row in report.get("leaderboard", ()):
        row_map = _mapping(row)
        if row_map.get("record_key") == record_key:
            return row_map
    return {}


def _covered_fact_property_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {
        str(property_id): dict(_mapping(property_metrics))
        for property_id, property_metrics in _mapping(
            row.get("covered_fact_property_metrics")
        ).items()
        if str(property_id)
    }
    properties = tuple(sorted(metrics))
    count = row.get("covered_fact_property_count")
    if count is None and properties:
        count = len(properties)
    return {
        "count": count,
        "properties": properties,
        "metrics": {property_id: metrics[property_id] for property_id in properties},
    }


def _first_set(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _combine_required_route_baseline_gates(
    *,
    route_registry_path: str | Path,
    required_route_baseline_keys: Sequence[str],
    ordinary_gate: Mapping[str, Any] | None,
    structured_fact_gate: Mapping[str, Any] | None,
    ordinary_required_route_keys: Sequence[str],
    structured_fact_route_keys: Sequence[str],
) -> dict[str, Any] | None:
    gates = tuple(gate for gate in (ordinary_gate, structured_fact_gate) if gate is not None)
    if not gates:
        return None
    if len(gates) == 1:
        gate = dict(gates[0])
        gate["required_keys"] = tuple(required_route_baseline_keys)
        gate["ordinary_required_route_keys"] = tuple(ordinary_required_route_keys)
        gate["structured_fact_route_keys"] = tuple(structured_fact_route_keys)
        if structured_fact_gate is not None:
            gate["structured_fact_robustness_gate"] = structured_fact_gate
        else:
            gate["ordinary_required_route_gate"] = ordinary_gate
        return gate

    rows: list[Mapping[str, Any]] = []
    failures: list[str] = []
    for label, gate in (
        ("ordinary_required_routes", ordinary_gate),
        ("structured_fact_robustness", structured_fact_gate),
    ):
        if gate is None:
            continue
        rows.extend(_mapping(row) for row in gate.get("rows", ()))
        gate_state = _mapping(gate.get("gate"))
        if gate_state.get("passed") is True:
            continue
        reasons = list(gate_state.get("blocking_reasons", ())) or [
            "route baseline gate did not pass"
        ]
        failures.extend(f"{label}: {reason}" for reason in reasons)
    gate_state = {
        "passed": not failures,
        "blocking_reasons": failures,
    }
    ordinary_comparison = None if ordinary_gate is None else ordinary_gate.get("comparison")
    structured_comparison = (
        None if structured_fact_gate is None else structured_fact_gate.get("comparison")
    )
    comparison = {
        "schema_version": 1,
        "status": "promote" if gate_state["passed"] else "blocked",
        "config": {
            "ordinary_required_route_keys": tuple(ordinary_required_route_keys),
            "structured_fact_route_keys": tuple(structured_fact_route_keys),
            "ordinary_required_routes": (
                None
                if ordinary_comparison is None
                else _mapping(ordinary_comparison).get("config")
            ),
            "structured_fact_robustness": (
                None
                if structured_comparison is None
                else _mapping(structured_comparison).get("config")
            ),
        },
        "leaderboard": tuple(rows),
        "ordinary_required_route_comparison": ordinary_comparison,
        "structured_fact_robustness_comparison": structured_comparison,
    }
    return {
        "schema_version": 1,
        "status": "promote" if gate_state["passed"] else "blocked",
        "registry": str(route_registry_path),
        "required_keys": tuple(required_route_baseline_keys),
        "ordinary_required_route_keys": tuple(ordinary_required_route_keys),
        "structured_fact_route_keys": tuple(structured_fact_route_keys),
        "comparison": comparison,
        "rows": tuple(rows),
        "gate": gate_state,
        "ordinary_required_route_gate": ordinary_gate,
        "structured_fact_robustness_gate": structured_fact_gate,
    }


def _decision(
    readiness: Mapping[str, Any],
    route: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    performance: Mapping[str, Any] | None = None,
    adapter_family: Mapping[str, Any] | None = None,
    triple_extraction_fixture_matrix: Mapping[str, Any] | None = None,
    counterfactual_verification: Mapping[str, Any] | None = None,
    required_routes: Mapping[str, Any] | None = None,
    product_trace_replay_workflow: Mapping[str, Any] | None = None,
    selector_replay: Mapping[str, Any] | None = None,
    product_runtime_drift: Mapping[str, Any] | None = None,
    release_efficiency: Mapping[str, Any] | None = None,
    external_evidence_baseline_comparison: Mapping[str, Any] | None = None,
    pre_generation_probe_comparison: Mapping[str, Any] | None = None,
    claim_factuality_probe_comparison: Mapping[str, Any] | None = None,
    frontier_release_evidence: Mapping[str, Any] | None = None,
    world_model_signal_workflow: Mapping[str, Any] | None = None,
    context_sensitivity_workflow: Mapping[str, Any] | None = None,
    mechanism_handoff_evidence_bundle: Mapping[str, Any] | None = None,
    pathway_intervention_workflow: Mapping[str, Any] | None = None,
    selfcheck_signal_fusion_workflow: Mapping[str, Any] | None = None,
    uncertainty_escalation_workflow: Mapping[str, Any] | None = None,
    feedback_policy_workflow: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    readiness_decision = _mapping(readiness.get("decision"))
    route_decision = _mapping(route.get("decision"))
    readiness_status = readiness_decision.get("status")
    route_status = route_decision.get("status")
    performance_gate = _mapping(None if performance is None else performance.get("gate"))
    performance_status = None if performance is None else performance.get("status")
    adapter_family_gate = _mapping(None if adapter_family is None else adapter_family.get("gate"))
    adapter_family_status = None if adapter_family is None else adapter_family.get("status")
    triple_extraction_fixture_matrix_gate = _mapping(
        None
        if triple_extraction_fixture_matrix is None
        else triple_extraction_fixture_matrix.get("gate")
    )
    triple_extraction_fixture_matrix_status = (
        None
        if triple_extraction_fixture_matrix is None
        else triple_extraction_fixture_matrix.get("status")
    )
    counterfactual_verification_gate = _mapping(
        None
        if counterfactual_verification is None
        else counterfactual_verification.get("gate")
    )
    counterfactual_verification_status = (
        None
        if counterfactual_verification is None
        else counterfactual_verification.get("status")
    )
    required_routes_gate = _mapping(None if required_routes is None else required_routes.get("gate"))
    required_route_status = None if required_routes is None else required_routes.get("status")
    product_trace_replay_workflow_gate = _mapping(
        None if product_trace_replay_workflow is None else product_trace_replay_workflow.get("gate")
    )
    product_trace_replay_workflow_status = (
        None if product_trace_replay_workflow is None else product_trace_replay_workflow.get("status")
    )
    selector_replay_gate = _mapping(None if selector_replay is None else selector_replay.get("gate"))
    selector_replay_status = None if selector_replay is None else selector_replay.get("status")
    product_runtime_drift_gate = _mapping(
        None if product_runtime_drift is None else product_runtime_drift.get("gate")
    )
    product_runtime_drift_status = (
        None if product_runtime_drift is None else product_runtime_drift.get("status")
    )
    release_efficiency_gate = _mapping(
        None if release_efficiency is None else release_efficiency.get("gate")
    )
    release_efficiency_status = (
        None if release_efficiency is None else release_efficiency.get("status")
    )
    external_evidence_baseline_comparison_gate = _mapping(
        None
        if external_evidence_baseline_comparison is None
        else external_evidence_baseline_comparison.get("gate")
    )
    external_evidence_baseline_comparison_status = (
        None
        if external_evidence_baseline_comparison is None
        else external_evidence_baseline_comparison.get("status")
    )
    pre_generation_probe_comparison_gate = _mapping(
        None
        if pre_generation_probe_comparison is None
        else pre_generation_probe_comparison.get("gate")
    )
    pre_generation_probe_comparison_status = (
        None
        if pre_generation_probe_comparison is None
        else pre_generation_probe_comparison.get("status")
    )
    claim_factuality_probe_comparison_gate = _mapping(
        None
        if claim_factuality_probe_comparison is None
        else claim_factuality_probe_comparison.get("gate")
    )
    claim_factuality_probe_comparison_status = (
        None
        if claim_factuality_probe_comparison is None
        else claim_factuality_probe_comparison.get("status")
    )
    frontier_release_evidence_gate = _mapping(
        None if frontier_release_evidence is None else frontier_release_evidence.get("gate")
    )
    frontier_release_evidence_status = (
        None if frontier_release_evidence is None else frontier_release_evidence.get("status")
    )
    world_model_signal_workflow_gate = _mapping(
        None
        if world_model_signal_workflow is None
        else world_model_signal_workflow.get("gate")
    )
    world_model_signal_workflow_status = (
        None
        if world_model_signal_workflow is None
        else world_model_signal_workflow.get("status")
    )
    context_sensitivity_workflow_gate = _mapping(
        None
        if context_sensitivity_workflow is None
        else context_sensitivity_workflow.get("gate")
    )
    context_sensitivity_workflow_status = (
        None
        if context_sensitivity_workflow is None
        else context_sensitivity_workflow.get("status")
    )
    mechanism_handoff_evidence_bundle_gate = _mapping(
        None
        if mechanism_handoff_evidence_bundle is None
        else mechanism_handoff_evidence_bundle.get("gate")
    )
    mechanism_handoff_evidence_bundle_status = (
        None
        if mechanism_handoff_evidence_bundle is None
        else mechanism_handoff_evidence_bundle.get("status")
    )
    pathway_intervention_workflow_gate = _mapping(
        None
        if pathway_intervention_workflow is None
        else pathway_intervention_workflow.get("gate")
    )
    pathway_intervention_workflow_status = (
        None
        if pathway_intervention_workflow is None
        else pathway_intervention_workflow.get("status")
    )
    selfcheck_signal_fusion_workflow_gate = _mapping(
        None
        if selfcheck_signal_fusion_workflow is None
        else selfcheck_signal_fusion_workflow.get("gate")
    )
    selfcheck_signal_fusion_workflow_status = (
        None
        if selfcheck_signal_fusion_workflow is None
        else selfcheck_signal_fusion_workflow.get("status")
    )
    uncertainty_escalation_workflow_gate = _mapping(
        None
        if uncertainty_escalation_workflow is None
        else uncertainty_escalation_workflow.get("gate")
    )
    uncertainty_escalation_workflow_status = (
        None
        if uncertainty_escalation_workflow is None
        else uncertainty_escalation_workflow.get("status")
    )
    feedback_policy_workflow_gate = _mapping(
        None if feedback_policy_workflow is None else feedback_policy_workflow.get("gate")
    )
    feedback_policy_workflow_status = (
        None if feedback_policy_workflow is None else feedback_policy_workflow.get("status")
    )
    blocking_reasons = []
    if readiness_status != "promote":
        blocking_reasons.append({
            "gate": "readiness_baseline",
            "status": readiness_status,
            "reasons": list(readiness_decision.get("blocking_reasons", ())),
        })
    if route_status != "promote":
        blocking_reasons.append({
            "gate": "route_baseline",
            "status": route_status,
            "reasons": list(route_decision.get("blocking_reasons", ())),
        })
    if performance is not None and performance_gate.get("passed") is not True:
        blocking_reasons.append({
            "gate": "performance_baseline",
            "status": performance_status,
            "reasons": list(performance_gate.get("blocking_reasons", ())),
        })
    if adapter_family is not None and adapter_family_gate.get("passed") is not True:
        blocking_reasons.append({
            "gate": "adapter_family_matrix",
            "status": adapter_family_status,
            "reasons": list(adapter_family_gate.get("blocking_reasons", ())),
        })
    if (
        triple_extraction_fixture_matrix is not None
        and triple_extraction_fixture_matrix_gate.get("passed") is not True
    ):
        blocking_reasons.append({
            "gate": "triple_extraction_fixture_matrix",
            "status": triple_extraction_fixture_matrix_status,
            "reasons": list(triple_extraction_fixture_matrix_gate.get("blocking_reasons", ())),
        })
    if (
        counterfactual_verification is not None
        and counterfactual_verification_gate.get("passed") is not True
    ):
        blocking_reasons.append({
            "gate": "counterfactual_verification",
            "status": counterfactual_verification_status,
            "reasons": list(counterfactual_verification_gate.get("blocking_reasons", ())),
        })
    if required_routes is not None and required_routes_gate.get("passed") is not True:
        blocking_reasons.append({
            "gate": "required_route_baselines",
            "status": required_route_status,
            "reasons": list(required_routes_gate.get("blocking_reasons", ())),
        })
    if (
        product_trace_replay_workflow is not None
        and product_trace_replay_workflow_gate.get("passed") is not True
    ):
        blocking_reasons.append({
            "gate": "product_trace_replay_workflow",
            "status": product_trace_replay_workflow_status,
            "reasons": list(product_trace_replay_workflow_gate.get("blocking_reasons", ())),
        })
    if selector_replay is not None and selector_replay_gate.get("passed") is not True:
        blocking_reasons.append({
            "gate": "selector_replay",
            "status": selector_replay_status,
            "reasons": list(selector_replay_gate.get("blocking_reasons", ())),
        })
    if product_runtime_drift is not None and product_runtime_drift_gate.get("passed") is not True:
        blocking_reasons.append({
            "gate": "product_runtime_drift",
            "status": product_runtime_drift_status,
            "reasons": list(product_runtime_drift_gate.get("blocking_reasons", ())),
        })
    if release_efficiency is not None and release_efficiency_gate.get("passed") is not True:
        blocking_reasons.append({
            "gate": "release_efficiency",
            "status": release_efficiency_status,
            "reasons": list(release_efficiency_gate.get("blocking_reasons", ())),
        })
    if (
        external_evidence_baseline_comparison is not None
        and external_evidence_baseline_comparison_gate.get("passed") is not True
    ):
        blocking_reasons.append({
            "gate": "external_evidence_baseline_comparison",
            "status": external_evidence_baseline_comparison_status,
            "reasons": list(external_evidence_baseline_comparison_gate.get("blocking_reasons", ())),
        })
    if (
        pre_generation_probe_comparison is not None
        and pre_generation_probe_comparison_gate.get("passed") is not True
    ):
        blocking_reasons.append({
            "gate": "pre_generation_probe_comparison",
            "status": pre_generation_probe_comparison_status,
            "reasons": list(pre_generation_probe_comparison_gate.get("blocking_reasons", ())),
        })
    if (
        claim_factuality_probe_comparison is not None
        and claim_factuality_probe_comparison_gate.get("passed") is not True
    ):
        blocking_reasons.append({
            "gate": "claim_factuality_probe_comparison",
            "status": claim_factuality_probe_comparison_status,
            "reasons": list(claim_factuality_probe_comparison_gate.get("blocking_reasons", ())),
        })
    if (
        frontier_release_evidence is not None
        and frontier_release_evidence_gate.get("passed") is not True
    ):
        blocking_reasons.append({
            "gate": "frontier_release_evidence",
            "status": frontier_release_evidence_status,
            "reasons": list(frontier_release_evidence_gate.get("blocking_reasons", ())),
        })
    if (
        world_model_signal_workflow is not None
        and world_model_signal_workflow_gate.get("passed") is not True
    ):
        blocking_reasons.append({
            "gate": "world_model_signal_workflow",
            "status": world_model_signal_workflow_status,
            "reasons": list(world_model_signal_workflow_gate.get("blocking_reasons", ())),
        })
    if (
        context_sensitivity_workflow is not None
        and context_sensitivity_workflow_gate.get("passed") is not True
    ):
        blocking_reasons.append({
            "gate": "context_sensitivity_workflow",
            "status": context_sensitivity_workflow_status,
            "reasons": list(context_sensitivity_workflow_gate.get("blocking_reasons", ())),
        })
    if (
        mechanism_handoff_evidence_bundle is not None
        and mechanism_handoff_evidence_bundle_gate.get("passed") is not True
    ):
        blocking_reasons.append({
            "gate": "mechanism_handoff_evidence_bundle",
            "status": mechanism_handoff_evidence_bundle_status,
            "reasons": list(mechanism_handoff_evidence_bundle_gate.get("blocking_reasons", ())),
        })
    if (
        pathway_intervention_workflow is not None
        and pathway_intervention_workflow_gate.get("passed") is not True
    ):
        blocking_reasons.append({
            "gate": "pathway_intervention_workflow",
            "status": pathway_intervention_workflow_status,
            "reasons": list(pathway_intervention_workflow_gate.get("blocking_reasons", ())),
        })
    if (
        selfcheck_signal_fusion_workflow is not None
        and selfcheck_signal_fusion_workflow_gate.get("passed") is not True
    ):
        blocking_reasons.append({
            "gate": "selfcheck_signal_fusion_workflow",
            "status": selfcheck_signal_fusion_workflow_status,
            "reasons": list(selfcheck_signal_fusion_workflow_gate.get("blocking_reasons", ())),
        })
    if (
        uncertainty_escalation_workflow is not None
        and uncertainty_escalation_workflow_gate.get("passed") is not True
    ):
        blocking_reasons.append({
            "gate": "uncertainty_escalation_workflow",
            "status": uncertainty_escalation_workflow_status,
            "reasons": list(
                uncertainty_escalation_workflow_gate.get("blocking_reasons", ())
            ),
        })
    if (
        feedback_policy_workflow is not None
        and feedback_policy_workflow_gate.get("passed") is not True
    ):
        blocking_reasons.append({
            "gate": "feedback_policy_workflow",
            "status": feedback_policy_workflow_status,
            "reasons": list(feedback_policy_workflow_gate.get("blocking_reasons", ())),
        })
    if candidate is None and not blocking_reasons:
        blocking_reasons.append({
            "gate": "release_candidate",
            "status": "blocked",
            "reasons": ["promoted baseline comparisons did not expose recommended rows"],
        })
    status = "promote" if candidate is not None and not blocking_reasons else (
        "no_candidate" if "no_candidate" in {readiness_status, route_status} else "blocked"
    )
    return {
        "status": status,
        "readiness_status": readiness_status,
        "route_status": route_status,
        "performance_status": performance_status,
        "adapter_family_status": adapter_family_status,
        "triple_extraction_fixture_matrix_status": triple_extraction_fixture_matrix_status,
        "counterfactual_verification_status": counterfactual_verification_status,
        "required_route_baseline_status": required_route_status,
        "product_trace_replay_workflow_status": product_trace_replay_workflow_status,
        "selector_replay_status": selector_replay_status,
        "product_runtime_drift_status": product_runtime_drift_status,
        "release_efficiency_status": release_efficiency_status,
        "external_evidence_baseline_comparison_status": external_evidence_baseline_comparison_status,
        "pre_generation_probe_comparison_status": pre_generation_probe_comparison_status,
        "claim_factuality_probe_comparison_status": claim_factuality_probe_comparison_status,
        "frontier_release_evidence_status": frontier_release_evidence_status,
        "world_model_signal_workflow_status": world_model_signal_workflow_status,
        "context_sensitivity_workflow_status": context_sensitivity_workflow_status,
        "mechanism_handoff_evidence_bundle_status": mechanism_handoff_evidence_bundle_status,
        "pathway_intervention_workflow_status": pathway_intervention_workflow_status,
        "selfcheck_signal_fusion_workflow_status": selfcheck_signal_fusion_workflow_status,
        "uncertainty_escalation_workflow_status": uncertainty_escalation_workflow_status,
        "feedback_policy_workflow_status": feedback_policy_workflow_status,
        "recommended_readiness_record": None if candidate is None else candidate.get("readiness_record"),
        "recommended_route_record": None if candidate is None else candidate.get("route_record"),
        "recommended_performance_baseline_record": (
            None if performance is None or performance_gate.get("passed") is not True else performance.get("record_key")
        ),
        "required_adapter_routes": (
            ()
            if adapter_family is None or adapter_family_gate.get("passed") is not True
            else tuple(adapter_family.get("required_routes", ()))
        ),
        "recommended_triple_extraction_fixture_matrix_report": (
            None
            if (
                triple_extraction_fixture_matrix is None
                or triple_extraction_fixture_matrix_gate.get("passed") is not True
            )
            else triple_extraction_fixture_matrix.get("report_path")
        ),
        "recommended_counterfactual_verification_report": (
            None
            if (
                counterfactual_verification is None
                or counterfactual_verification_gate.get("passed") is not True
            )
            else counterfactual_verification.get("report_path")
        ),
        "required_route_baseline_records": (
            ()
            if required_routes is None or required_routes_gate.get("passed") is not True
            else tuple(required_routes.get("required_keys", ()))
        ),
        "recommended_selector_replay_candidate": (
            None
            if selector_replay is None or selector_replay_gate.get("passed") is not True
            else selector_replay.get("recommended_candidate")
        ),
        "recommended_product_runtime_drift_report": (
            None
            if product_runtime_drift is None or product_runtime_drift_gate.get("passed") is not True
            else product_runtime_drift.get("report_path")
        ),
        "recommended_release_efficiency_report": (
            None
            if release_efficiency is None or release_efficiency_gate.get("passed") is not True
            else release_efficiency.get("report_path")
        ),
        "recommended_release_efficiency_profile": (
            None
            if release_efficiency is None or release_efficiency_gate.get("passed") is not True
            else release_efficiency.get("recommended_profile")
        ),
        "recommended_external_evidence_baseline_comparison_report": (
            None
            if (
                external_evidence_baseline_comparison is None
                or external_evidence_baseline_comparison_gate.get("passed") is not True
            )
            else external_evidence_baseline_comparison.get("report_path")
        ),
        "recommended_pre_generation_probe_comparison_report": (
            None
            if (
                pre_generation_probe_comparison is None
                or pre_generation_probe_comparison_gate.get("passed") is not True
            )
            else pre_generation_probe_comparison.get("report_path")
        ),
        "recommended_claim_factuality_probe_comparison_report": (
            None
            if (
                claim_factuality_probe_comparison is None
                or claim_factuality_probe_comparison_gate.get("passed") is not True
            )
            else claim_factuality_probe_comparison.get("report_path")
        ),
        "recommended_frontier_release_evidence_report": (
            None
            if (
                frontier_release_evidence is None
                or frontier_release_evidence_gate.get("passed") is not True
            )
            else frontier_release_evidence.get("report_path")
        ),
        "recommended_world_model_signal_workflow_report": (
            None
            if (
                world_model_signal_workflow is None
                or world_model_signal_workflow_gate.get("passed") is not True
            )
            else world_model_signal_workflow.get("report_path")
        ),
        "recommended_context_sensitivity_workflow_report": (
            None
            if (
                context_sensitivity_workflow is None
                or context_sensitivity_workflow_gate.get("passed") is not True
            )
            else context_sensitivity_workflow.get("report_path")
        ),
        "recommended_mechanism_handoff_evidence_bundle_report": (
            None
            if (
                mechanism_handoff_evidence_bundle is None
                or mechanism_handoff_evidence_bundle_gate.get("passed") is not True
            )
            else mechanism_handoff_evidence_bundle.get("report_path")
        ),
        "recommended_pathway_intervention_workflow_report": (
            None
            if (
                pathway_intervention_workflow is None
                or pathway_intervention_workflow_gate.get("passed") is not True
            )
            else pathway_intervention_workflow.get("report_path")
        ),
        "recommended_selfcheck_signal_fusion_workflow_report": (
            None
            if (
                selfcheck_signal_fusion_workflow is None
                or selfcheck_signal_fusion_workflow_gate.get("passed") is not True
            )
            else selfcheck_signal_fusion_workflow.get("report_path")
        ),
        "recommended_uncertainty_escalation_workflow_report": (
            None
            if (
                uncertainty_escalation_workflow is None
                or uncertainty_escalation_workflow_gate.get("passed") is not True
            )
            else uncertainty_escalation_workflow.get("report_path")
        ),
        "recommended_feedback_policy_workflow_report": (
            None
            if feedback_policy_workflow is None or feedback_policy_workflow_gate.get("passed") is not True
            else feedback_policy_workflow.get("report_path")
        ),
        "recommended_feedback_policy_candidate_control_policy": (
            None
            if feedback_policy_workflow is None or feedback_policy_workflow_gate.get("passed") is not True
            else feedback_policy_workflow.get("candidate_control_policy")
        ),
        "recommended_feedback_policy_candidate_control_defaults": (
            None
            if feedback_policy_workflow is None or feedback_policy_workflow_gate.get("passed") is not True
            else feedback_policy_workflow.get("candidate_control_defaults")
        ),
        "recommended_model": None if candidate is None else candidate.get("model"),
        "recommended_route": None if candidate is None else _mapping(candidate.get("verifier_route")).get("route"),
        "blocking_reasons": blocking_reasons,
    }


def _required_route_baseline_gate(
    *,
    route_registry_path: str | Path,
    required_route_baseline_keys: Sequence[str],
    recursive: bool,
    allow_unverified: bool,
    min_selected: int | None,
    min_decision_accuracy: float | None,
    max_false_supported_rate: float | None,
    min_false_refuted_rate: float | None,
    max_verified_false_alarm: float | None,
    min_verified_detection: float | None,
    max_mean_duration_seconds: float | None,
    max_p99_duration_seconds: float | None,
    max_max_duration_seconds: float | None,
    max_mean_attempted_route_count: float | None,
    max_retrieval_use_rate: float | None,
    max_runtime_total_seconds: float | None,
    max_retrieval_hit_count: float | None,
    min_claims_cache_hit_rate: float | None,
    min_verifier_trace_cache_hit_rate: float | None,
    min_covered_fact_properties: int | None,
    min_covered_fact_property_records: int | None,
    min_covered_fact_property_source_documents: int | None,
    min_covered_fact_property_decision_accuracy: float | None,
    max_covered_fact_property_false_supported_rate: float | None,
    min_covered_fact_property_false_refuted_rate: float | None,
    require_non_oracle_evidence: bool,
    require_retrieval_provenance_filter: bool,
    required_retrieval_source_prefixes: Sequence[str],
    required_retrieval_metadata: Mapping[str, Any] | None,
    min_retrieval_filter_score: float | None,
    require_retrieval_stress_control: bool,
    retrieval_stress_manifest: str | Path | None,
    min_stress_false_supported_rate: float | None,
    max_stress_false_refuted_rate: float | None,
    fingerprint_cache: MutableMapping[str, dict[str, Any]],
    json_cache: MutableMapping[str, dict[str, Any]],
    json_cache_stats: MutableMapping[str, int],
) -> dict[str, Any] | None:
    required_keys = tuple(str(key) for key in required_route_baseline_keys if str(key))
    if not required_keys:
        return None
    comparison = compare_route_baselines(
        registry_path=route_registry_path,
        baseline_keys=required_keys,
        recursive=recursive,
        allow_unverified=allow_unverified,
        min_selected=min_selected,
        min_decision_accuracy=min_decision_accuracy,
        max_false_supported_rate=max_false_supported_rate,
        min_false_refuted_rate=min_false_refuted_rate,
        max_verified_false_alarm=max_verified_false_alarm,
        min_verified_detection=min_verified_detection,
        max_mean_duration_seconds=max_mean_duration_seconds,
        max_p99_duration_seconds=max_p99_duration_seconds,
        max_max_duration_seconds=max_max_duration_seconds,
        max_mean_attempted_route_count=max_mean_attempted_route_count,
        max_retrieval_use_rate=max_retrieval_use_rate,
        max_runtime_total_seconds=max_runtime_total_seconds,
        max_retrieval_hit_count=max_retrieval_hit_count,
        min_claims_cache_hit_rate=min_claims_cache_hit_rate,
        min_verifier_trace_cache_hit_rate=min_verifier_trace_cache_hit_rate,
        min_covered_fact_properties=min_covered_fact_properties,
        min_covered_fact_property_records=min_covered_fact_property_records,
        min_covered_fact_property_source_documents=min_covered_fact_property_source_documents,
        min_covered_fact_property_decision_accuracy=min_covered_fact_property_decision_accuracy,
        max_covered_fact_property_false_supported_rate=max_covered_fact_property_false_supported_rate,
        min_covered_fact_property_false_refuted_rate=min_covered_fact_property_false_refuted_rate,
        require_non_oracle_evidence=require_non_oracle_evidence,
        require_retrieval_provenance_filter=require_retrieval_provenance_filter,
        required_retrieval_source_prefixes=required_retrieval_source_prefixes,
        required_retrieval_metadata=required_retrieval_metadata,
        min_retrieval_filter_score=min_retrieval_filter_score,
        require_retrieval_stress_control=require_retrieval_stress_control,
        retrieval_stress_manifest=retrieval_stress_manifest,
        min_stress_false_supported_rate=min_stress_false_supported_rate,
        max_stress_false_refuted_rate=max_stress_false_refuted_rate,
        notes=("release candidate required route baseline gate",),
        fingerprint_cache=fingerprint_cache,
        json_cache=json_cache,
        json_cache_stats=json_cache_stats,
    )
    rows = tuple(_mapping(row) for row in comparison.get("leaderboard", ()))
    rows_by_key = {str(row.get("record_key")): row for row in rows if row.get("record_key") is not None}
    failures = []
    for key in required_keys:
        row = rows_by_key.get(key)
        if row is None:
            failures.append(f"required route baseline {key!r} is missing from comparison")
            continue
        gate = _mapping(row.get("gate"))
        if gate.get("passed") is True:
            continue
        reasons = list(gate.get("blocking_reasons", ())) or ["route baseline gate did not pass"]
        failures.extend(f"{key}: {reason}" for reason in reasons)
    gate = {
        "passed": not failures,
        "blocking_reasons": failures,
    }
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "registry": str(route_registry_path),
        "required_keys": required_keys,
        "comparison": comparison,
        "rows": rows,
        "gate": gate,
    }


def _adapter_family_matrix_gate(
    *,
    adapter_family_matrix_path: str | Path | None,
    required_routes: Sequence[str],
    require_state_transition_world_model: bool,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if adapter_family_matrix_path is None:
        return None
    matrix_path = Path(adapter_family_matrix_path)
    report, report_error = verification_context.load_json_object(matrix_path)
    routes = tuple(str(route) for route in report.get("routes", ()) if str(route))
    required = tuple(str(route) for route in required_routes if str(route))
    family_statuses = _adapter_family_statuses(report)
    family_details = _adapter_family_details(report)
    promotion_decision = _mapping(report.get("promotion_decision"))
    route_comparison = _mapping(report.get("route_comparison"))
    quality_gate = _mapping(route_comparison.get("quality_gate"))
    retrieval_routes = tuple(str(route) for route in report.get("retrieval_routes", ()) if str(route))
    audit_routes = tuple(str(route) for route in report.get("audit_routes", ()) if str(route))
    gate = _adapter_family_gate(
        report_error=report_error,
        promotion_decision=promotion_decision,
        quality_gate=quality_gate,
        routes=routes,
        family_statuses=family_statuses,
        family_details=family_details,
        required_routes=required,
        require_state_transition_world_model=bool(require_state_transition_world_model),
    )
    state_transition = _mapping(family_details.get("state_transition"))
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "matrix_path": str(matrix_path),
        "workflow": report.get("workflow"),
        "alpha": report.get("alpha"),
        "n_records": report.get("n_records"),
        "routes": routes,
        "required_routes": required,
        "retrieval_routes": retrieval_routes,
        "audit_routes": audit_routes,
        "promoted_routes": tuple(route for route, status in family_statuses.items() if status == "promote"),
        "family_statuses": family_statuses,
        "require_state_transition_world_model": bool(require_state_transition_world_model),
        "state_transition_world_model_adapter": state_transition.get("world_model_adapter"),
        "state_transition_world_model_rule_count": state_transition.get("world_model_rule_count"),
        "promotion_status": promotion_decision.get("status"),
        "quality_gate_passed": quality_gate.get("passed"),
        "gate": gate,
    }


def _adapter_family_gate(
    *,
    report_error: str | None,
    promotion_decision: Mapping[str, Any],
    quality_gate: Mapping[str, Any],
    routes: Sequence[str],
    family_statuses: Mapping[str, str],
    family_details: Mapping[str, Mapping[str, Any]],
    required_routes: Sequence[str],
    require_state_transition_world_model: bool,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"adapter family matrix could not be loaded: {report_error}")
    if promotion_decision.get("status") != "promote":
        failures.append(
            f"adapter family promotion status is {promotion_decision.get('status')!r}, expected 'promote'"
        )
    if quality_gate and quality_gate.get("passed") is not True:
        failures.append("adapter family route quality gate did not pass")
    route_set = set(routes)
    for route in required_routes:
        if route not in route_set:
            failures.append(f"required adapter route {route!r} is missing from matrix")
            continue
        status = family_statuses.get(route)
        if status != "promote":
            failures.append(f"required adapter route {route!r} status is {status!r}, expected 'promote'")
    if require_state_transition_world_model:
        if "state_transition" not in route_set:
            failures.append("state_transition world-model evidence requires a state_transition route")
        else:
            transition = _mapping(family_details.get("state_transition"))
            adapter = transition.get("world_model_adapter")
            if adapter != "RuleBasedWorldModelAdapter":
                failures.append(
                    "state_transition world-model adapter is "
                    f"{adapter!r}, expected 'RuleBasedWorldModelAdapter'"
                )
            rule_count = _float_or_none(transition.get("world_model_rule_count"))
            if rule_count is None or rule_count <= 0:
                failures.append(
                    "state_transition world-model rule count must be positive "
                    f"for rule-based evidence, got {transition.get('world_model_rule_count')!r}"
                )
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _adapter_family_details(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    details = {}
    for item in report.get("families", ()):
        family = _mapping(item)
        route = family.get("route")
        if route is not None:
            details[str(route)] = family
    return details


def _adapter_family_statuses(report: Mapping[str, Any]) -> dict[str, str]:
    statuses = {}
    for item in report.get("families", ()):
        family = _mapping(item)
        route = family.get("route")
        if route is not None:
            statuses[str(route)] = str(family.get("status"))
    by_route = _mapping(_mapping(report.get("route_comparison")).get("by_route"))
    for route, item in by_route.items():
        if str(route) not in statuses:
            status = _mapping(item).get("promotion_status")
            if status is not None:
                statuses[str(route)] = str(status)
    return statuses


def _triple_extraction_fixture_matrix_gate(
    *,
    triple_extraction_fixture_matrix_source: Mapping[str, Any] | None,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    min_corpora: int | None,
    min_distinct_predicates: int | None,
    min_external_prediction_count: int | None,
    min_external_prediction_corpora: int | None,
    min_mean_best_external_f1: float | None,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if triple_extraction_fixture_matrix_source is None:
        return None
    report_path = Path(triple_extraction_fixture_matrix_source["path"])
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _triple_extraction_fixture_matrix_manifest_path(report, report_path=report_path)
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="triple_extraction_fixture_matrix_manifest",
        verification_context=verification_context,
    )
    gate = _triple_extraction_fixture_matrix_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        allow_unverified=allow_unverified,
        min_corpora=min_corpora,
        min_distinct_predicates=min_distinct_predicates,
        min_external_prediction_count=min_external_prediction_count,
        min_external_prediction_corpora=min_external_prediction_corpora,
        min_mean_best_external_f1=min_mean_best_external_f1,
    )
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "source": triple_extraction_fixture_matrix_source.get("source"),
        "registry": triple_extraction_fixture_matrix_source.get("registry"),
        "record_key": triple_extraction_fixture_matrix_source.get("record_key"),
        "record": triple_extraction_fixture_matrix_source.get("record"),
        "workflow": report.get("workflow"),
        "report_status": report.get("status"),
        "n_corpora": report.get("n_corpora"),
        "promoted_corpora": report.get("promoted_corpora"),
        "distinct_predicate_count": report.get("distinct_predicate_count"),
        "distinct_predicates": tuple(report.get("distinct_predicates", ())),
        "mean_baseline_f1": _float_or_none(report.get("mean_baseline_f1")),
        "mean_best_f1": _float_or_none(report.get("mean_best_f1")),
        "mean_f1_lift": _float_or_none(report.get("mean_f1_lift")),
        "external_prediction_count": _float_or_none(report.get("external_prediction_count")),
        "external_prediction_corpora": _tuple_or_empty_sequence(
            report.get("external_prediction_corpora", ())
        ),
        "mean_best_external_f1": _float_or_none(report.get("mean_best_external_f1")),
        "verification": verification,
        "gate": gate,
    }


def _triple_extraction_fixture_matrix_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    allow_unverified: bool,
    min_corpora: int | None,
    min_distinct_predicates: int | None,
    min_external_prediction_count: int | None,
    min_external_prediction_corpora: int | None,
    min_mean_best_external_f1: float | None,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"triple extraction fixture matrix could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("triple extraction fixture matrix artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("triple extraction fixture matrix manifest verification failed")
    if report.get("workflow") != "triple_extraction_fixture_matrix":
        failures.append(
            "triple extraction fixture matrix workflow is "
            f"{report.get('workflow')!r}, expected 'triple_extraction_fixture_matrix'"
        )
    if report.get("status") != "promote":
        failures.append(
            f"triple extraction fixture matrix status is {report.get('status')!r}, expected 'promote'"
        )
    n_corpora = _float_or_none(report.get("n_corpora"))
    promoted_corpora = _float_or_none(report.get("promoted_corpora"))
    if n_corpora is None:
        failures.append("triple extraction fixture matrix n_corpora is missing")
    if promoted_corpora is None:
        failures.append("triple extraction fixture matrix promoted_corpora is missing")
    if min_corpora is not None:
        if n_corpora is None or n_corpora < min_corpora:
            failures.append(
                "triple extraction fixture matrix corpus count below "
                f"{min_corpora}: {report.get('n_corpora')!r}"
            )
        if promoted_corpora is None or promoted_corpora < min_corpora:
            failures.append(
                "triple extraction fixture matrix promoted corpus count below "
                f"{min_corpora}: {report.get('promoted_corpora')!r}"
            )
    distinct_predicate_count = _float_or_none(report.get("distinct_predicate_count"))
    if distinct_predicate_count is None:
        failures.append("triple extraction fixture matrix distinct predicate count is missing")
    if min_distinct_predicates is not None and (
        distinct_predicate_count is None
        or distinct_predicate_count < min_distinct_predicates
    ):
        failures.append(
            "triple extraction fixture matrix distinct predicate count below "
            f"{min_distinct_predicates}: {report.get('distinct_predicate_count')!r}"
        )
    mean_best_f1 = _float_or_none(report.get("mean_best_f1"))
    if mean_best_f1 is None:
        failures.append("triple extraction fixture matrix mean_best_f1 is missing")
    elif mean_best_f1 <= 0.0:
        failures.append(f"triple extraction fixture matrix mean_best_f1 is non-positive: {mean_best_f1!r}")
    external_prediction_count = _float_or_none(report.get("external_prediction_count"))
    if min_external_prediction_count is not None:
        if external_prediction_count is None:
            failures.append("triple extraction fixture matrix external prediction count is missing")
        elif external_prediction_count < min_external_prediction_count:
            failures.append(
                "triple extraction fixture matrix external prediction count below "
                f"{min_external_prediction_count}: {report.get('external_prediction_count')!r}"
            )
    external_prediction_corpus_count = len(
        _tuple_or_empty_sequence(report.get("external_prediction_corpora", ()))
    )
    if min_external_prediction_corpora is not None:
        if external_prediction_corpus_count < min_external_prediction_corpora:
            failures.append(
                "triple extraction fixture matrix external prediction corpus count below "
                f"{min_external_prediction_corpora}: {external_prediction_corpus_count!r}"
            )
    mean_best_external_f1 = _float_or_none(report.get("mean_best_external_f1"))
    if min_mean_best_external_f1 is not None:
        if mean_best_external_f1 is None:
            failures.append("triple extraction fixture matrix mean_best_external_f1 is missing")
        elif mean_best_external_f1 < min_mean_best_external_f1:
            failures.append(
                "triple extraction fixture matrix mean_best_external_f1 below "
                f"{min_mean_best_external_f1}: {mean_best_external_f1!r}"
            )
    return {
        "passed": not failures,
        "blocking_reasons": failures,
        "policy": {
            "min_corpora": min_corpora,
            "min_distinct_predicates": min_distinct_predicates,
            "min_external_prediction_count": min_external_prediction_count,
            "min_external_prediction_corpora": min_external_prediction_corpora,
            "min_mean_best_external_f1": min_mean_best_external_f1,
        },
    }


def _triple_extraction_fixture_matrix_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _first_present(
        report.get("artifact_manifest_path"),
        _nested(report, "paths", "artifact_manifest"),
    )
    if raw_path is None:
        sibling = report_path.parent / "artifact-manifest.json"
        return sibling if sibling.exists() else None
    return _resolve_path(raw_path, base_path=report_path)


def _resolve_triple_extraction_fixture_matrix_source(
    *,
    triple_extraction_fixture_matrix_path: str | Path | None,
    triple_extraction_fixture_matrix_registry_path: str | Path | None,
    triple_extraction_fixture_matrix_key: str | None,
    default_registry_path: str | Path,
) -> dict[str, Any] | None:
    if triple_extraction_fixture_matrix_path is not None:
        if triple_extraction_fixture_matrix_key is not None:
            raise ValueError(
                "triple_extraction_fixture_matrix_path is mutually exclusive with "
                "triple_extraction_fixture_matrix_key."
            )
        return {"source": "file", "path": Path(triple_extraction_fixture_matrix_path)}
    if triple_extraction_fixture_matrix_key is None:
        if triple_extraction_fixture_matrix_registry_path is not None:
            raise ValueError(
                "triple_extraction_fixture_matrix_registry_path requires "
                "triple_extraction_fixture_matrix_key."
            )
        return None
    registry_path = Path(
        default_registry_path
        if triple_extraction_fixture_matrix_registry_path is None
        else triple_extraction_fixture_matrix_registry_path
    )
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get(str(triple_extraction_fixture_matrix_key))
    if record.artifact_type != "report":
        raise ValueError(f"registry record {record.key()!r} is not a report.")
    return {
        "source": "registry",
        "registry": str(registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "path": _resolve_registry_record_path(registry_path, record),
    }


def _counterfactual_verification_gate(
    *,
    counterfactual_verification_source: Mapping[str, Any] | None,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    min_records: int | None,
    min_pass_rate: float | None,
    max_false_invariance_rate: float | None,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if counterfactual_verification_source is None:
        return None
    report_path = Path(counterfactual_verification_source["path"])
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _counterfactual_verification_manifest_path(report, report_path=report_path)
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="counterfactual_verification_manifest",
        verification_context=verification_context,
    )
    summary = _counterfactual_verification_summary(report)
    gate = _counterfactual_verification_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        summary=summary,
        allow_unverified=allow_unverified,
        min_records=min_records,
        min_pass_rate=min_pass_rate,
        max_false_invariance_rate=max_false_invariance_rate,
    )
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "source": counterfactual_verification_source.get("source"),
        "registry": counterfactual_verification_source.get("registry"),
        "record_key": counterfactual_verification_source.get("record_key"),
        "record": counterfactual_verification_source.get("record"),
        "workflow": report.get("workflow"),
        "record_count": _float_or_none(summary.get("record_count")),
        "pass_rate": _float_or_none(summary.get("pass_rate")),
        "false_invariance_rate": _float_or_none(summary.get("false_invariance_rate")),
        "flip_success_count": _float_or_none(summary.get("flip_success_count")),
        "verification": verification,
        "gate": gate,
    }


def _counterfactual_verification_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    summary: Mapping[str, Any],
    allow_unverified: bool,
    min_records: int | None,
    min_pass_rate: float | None,
    max_false_invariance_rate: float | None,
) -> dict[str, Any]:
    failures = []
    effective_min_records = 1 if min_records is None else int(min_records)
    effective_min_pass_rate = 1.0 if min_pass_rate is None else float(min_pass_rate)
    effective_max_false_invariance_rate = (
        0.0 if max_false_invariance_rate is None else float(max_false_invariance_rate)
    )
    if report_error is not None:
        failures.append(f"counterfactual verification report could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("counterfactual verification artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("counterfactual verification manifest verification failed")
    workflow = report.get("workflow")
    if workflow not in {"counterfactual_verification_eval", "counterfactual_verification_audit"}:
        failures.append(
            f"counterfactual verification workflow is {workflow!r}, expected "
            "'counterfactual_verification_eval' or 'counterfactual_verification_audit'"
        )
    record_count = _float_or_none(summary.get("record_count"))
    pass_rate = _float_or_none(summary.get("pass_rate"))
    false_invariance_rate = _float_or_none(summary.get("false_invariance_rate"))
    if record_count is None:
        failures.append("counterfactual verification record_count is missing")
    elif record_count < effective_min_records:
        failures.append(
            "counterfactual verification record count below "
            f"{effective_min_records}: {summary.get('record_count')!r}"
        )
    if pass_rate is None:
        failures.append("counterfactual verification pass_rate is missing")
    elif pass_rate < effective_min_pass_rate:
        failures.append(
            "counterfactual verification pass_rate below "
            f"{effective_min_pass_rate}: {pass_rate!r}"
        )
    if false_invariance_rate is None:
        failures.append("counterfactual verification false_invariance_rate is missing")
    elif false_invariance_rate > effective_max_false_invariance_rate:
        failures.append(
            "counterfactual verification false_invariance_rate above "
            f"{effective_max_false_invariance_rate}: {false_invariance_rate!r}"
        )
    return {
        "passed": not failures,
        "blocking_reasons": failures,
        "policy": {
            "min_records": effective_min_records,
            "min_pass_rate": effective_min_pass_rate,
            "max_false_invariance_rate": effective_max_false_invariance_rate,
        },
    }


def _counterfactual_verification_summary(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(
        _first_present(
            _nested(report, "report", "summary"),
            report.get("summary"),
        )
    )


def _counterfactual_verification_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _first_present(
        _nested(report, "paths", "artifact_manifest"),
        report.get("artifact_manifest_path"),
    )
    if raw_path is None:
        sibling = report_path.parent / "artifact-manifest.json"
        return sibling if sibling.exists() else None
    return _resolve_path(raw_path, base_path=report_path)


def _resolve_counterfactual_verification_source(
    *,
    counterfactual_verification_report_path: str | Path | None,
    counterfactual_verification_registry_path: str | Path | None,
    counterfactual_verification_key: str | None,
    default_registry_path: str | Path,
) -> dict[str, Any] | None:
    if counterfactual_verification_report_path is not None:
        if counterfactual_verification_key is not None:
            raise ValueError(
                "counterfactual_verification_report_path is mutually exclusive with "
                "counterfactual_verification_key."
            )
        return {"source": "file", "path": Path(counterfactual_verification_report_path)}
    if counterfactual_verification_key is None:
        if counterfactual_verification_registry_path is not None:
            raise ValueError(
                "counterfactual_verification_registry_path requires "
                "counterfactual_verification_key."
            )
        return None
    registry_path = Path(
        default_registry_path
        if counterfactual_verification_registry_path is None
        else counterfactual_verification_registry_path
    )
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get(str(counterfactual_verification_key))
    if record.artifact_type != "report":
        raise ValueError(f"registry record {record.key()!r} is not a report.")
    return {
        "source": "registry",
        "registry": str(registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "path": _resolve_registry_record_path(registry_path, record),
    }


def _performance_baseline_gate(
    *,
    performance_registry_path: str | Path,
    performance_baseline_key: str | None,
    recursive: bool,
    allow_unverified: bool,
    candidate: Mapping[str, Any] | None,
    require_score_dump_cache: bool,
    min_score_dump_cache_jsonl_view_hit_rate: float | None,
    drift_baseline_key: str | None,
    max_uncached_total_seconds_ratio: float | None,
    max_cached_total_seconds_ratio: float | None,
    max_cache_only_total_seconds_ratio: float | None,
    max_score_dump_cache_jsonl_view_hit_rate_drop: float | None,
    max_covariance_maha_last_auroc_drop: float | None,
    manifest_fingerprint_workers: int,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if performance_baseline_key is None:
        return None
    registry = ArtifactRegistry.load_json(performance_registry_path)
    record = registry.get(performance_baseline_key)
    if record.artifact_type != "performance_baseline":
        raise ValueError(f"registry record {record.key()!r} is not a performance_baseline.")
    report_path = Path(record.path)
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _performance_manifest_path(record, report, report_path=report_path)
    verification = _verify_performance_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        verification_context=verification_context,
    )
    runtime_recommendation, runtime_source = _performance_runtime_recommendation(
        record,
        report,
        report_path=report_path,
        verification_context=verification_context,
    )
    performance_evidence_bundle = _mapping(report.get("performance_evidence_bundle"))
    performance_score_dump_cache = _mapping(performance_evidence_bundle.get("score_dump_cache"))
    performance_score_dump_cache_totals = _mapping(performance_score_dump_cache.get("totals"))
    performance_jsonl_view_cache = _mapping(performance_score_dump_cache_totals.get("jsonl_view"))
    drift_record = None
    drift_report_path = None
    drift_report: dict[str, Any] = {}
    drift_report_error = None
    drift_manifest_path = None
    drift_verification = None
    drift_evidence_bundle: dict[str, Any] = {}
    if drift_baseline_key is not None:
        drift_record = registry.get(drift_baseline_key)
        if drift_record.artifact_type != "performance_baseline":
            raise ValueError(
                f"registry record {drift_record.key()!r} is not a performance_baseline."
        )
        drift_report_path = Path(drift_record.path)
        drift_report, drift_report_error = verification_context.load_json_object(drift_report_path)
        drift_manifest_path = _performance_manifest_path(
            drift_record,
            drift_report,
            report_path=drift_report_path,
        )
        drift_verification = _verify_performance_manifest(
            drift_manifest_path,
            recursive=recursive,
            max_workers=manifest_fingerprint_workers,
            verification_context=verification_context,
        )
        drift_evidence_bundle = _mapping(drift_report.get("performance_evidence_bundle"))
    performance_trend_gate = _performance_trend_gate(
        current_bundle=performance_evidence_bundle,
        reference_bundle=drift_evidence_bundle,
        reference_report_error=drift_report_error,
        reference_manifest_path=drift_manifest_path,
        reference_verification=drift_verification,
        reference_record_key=None if drift_record is None else drift_record.key(),
        reference_report_path=drift_report_path,
        allow_unverified=allow_unverified,
        max_uncached_total_seconds_ratio=max_uncached_total_seconds_ratio,
        max_cached_total_seconds_ratio=max_cached_total_seconds_ratio,
        max_cache_only_total_seconds_ratio=max_cache_only_total_seconds_ratio,
        max_score_dump_cache_jsonl_view_hit_rate_drop=(
            max_score_dump_cache_jsonl_view_hit_rate_drop
        ),
    )
    gate = _performance_gate(
        verification=verification,
        allow_unverified=allow_unverified,
        report_error=report_error,
        manifest_path=manifest_path,
        runtime_recommendation=runtime_recommendation,
        performance_evidence_bundle=performance_evidence_bundle,
        candidate=candidate,
        require_score_dump_cache=require_score_dump_cache,
        min_score_dump_cache_jsonl_view_hit_rate=min_score_dump_cache_jsonl_view_hit_rate,
        performance_trend_gate=performance_trend_gate,
        max_covariance_maha_last_auroc_drop=max_covariance_maha_last_auroc_drop,
    )
    recommendation = _mapping(runtime_recommendation.get("recommendation"))
    best_quality = _mapping(recommendation.get("best_quality_signal"))
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "registry": str(performance_registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "verification": verification,
        "performance_drift_baseline_record": None if drift_record is None else drift_record.key(),
        "performance_drift_baseline_report_path": (
            None if drift_report_path is None else str(drift_report_path)
        ),
        "performance_drift_baseline_manifest_path": (
            None if drift_manifest_path is None else str(drift_manifest_path)
        ),
        "performance_drift_baseline_verification": drift_verification,
        "runtime_recommendation_source": runtime_source,
        "runtime_recommendation_status": runtime_recommendation.get("status"),
        "performance_evidence_bundle": (
            None if not performance_evidence_bundle else performance_evidence_bundle
        ),
        "performance_evidence_bundle_status": performance_evidence_bundle.get("status"),
        "performance_evidence_bundle_release_ready": performance_evidence_bundle.get("release_ready"),
        "performance_score_dump_cache": (
            None if not performance_score_dump_cache else performance_score_dump_cache
        ),
        "performance_score_dump_cache_source_count": performance_score_dump_cache.get("source_count"),
        "performance_score_dump_cache_jsonl_view_hit_rate": performance_jsonl_view_cache.get("hit_rate"),
        "performance_score_dump_cache_gate": gate.get("score_dump_cache"),
        "performance_trend_gate": gate.get("performance_trend"),
        "covariance_tradeoff_gate": gate.get("covariance_tradeoff"),
        "runtime": {
            "cell_id": recommendation.get("cell_id"),
            "layer": recommendation.get("layer"),
            "batch_size": recommendation.get("batch_size"),
            "hidden_state_capture": recommendation.get("hidden_state_capture"),
            "covariance_mode": recommendation.get("covariance_mode"),
            "covariance_low_rank": recommendation.get("covariance_low_rank"),
            "max_batch_tokens": recommendation.get("max_batch_tokens"),
            "prefix_kv_cache": recommendation.get("prefix_kv_cache"),
            "max_workers": recommendation.get("max_workers"),
            "inside_trigger_budget_id": _performance_inside_trigger_budget_id(recommendation),
            "inside_trigger_budget_policy": _performance_inside_trigger_budget_policy(recommendation),
            "best_quality_signal": (
                None
                if not best_quality
                else {
                    "name": best_quality.get("name"),
                    "auroc": _float_or_none(best_quality.get("auroc")),
                }
            ),
        },
        "gate": gate,
    }


def _product_trace_replay_workflow_gate(
    *,
    product_trace_replay_workflow_source: Mapping[str, Any] | None,
    selector_replay_report_path: str | Path | None,
    product_runtime_drift_report_path: str | Path | None,
    require_action_audit_gate: bool,
    require_action_execution_gate: bool,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if product_trace_replay_workflow_source is None:
        return None
    report_path = Path(product_trace_replay_workflow_source["path"])
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _product_trace_replay_workflow_manifest_path(report, report_path=report_path)
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="product_trace_replay_workflow_manifest",
        verification_context=verification_context,
    )
    workflow_selector_path = _product_trace_replay_workflow_child_path(
        report,
        report_path=report_path,
        child_path_key="selector_replay_report",
    )
    workflow_drift_path = _product_trace_replay_workflow_child_path(
        report,
        report_path=report_path,
        child_path_key="runtime_drift_report",
    )
    workflow_action_audit_path = _product_trace_replay_workflow_child_path(
        report,
        report_path=report_path,
        child_path_key="action_audit_gate_report",
    )
    workflow_action_execution_path = _product_trace_replay_workflow_child_path(
        report,
        report_path=report_path,
        child_path_key="action_execution_gate_report",
    )
    action_audit_gate = _product_trace_action_audit_gate_summary(
        report,
        report_path=report_path,
        action_audit_gate_report_path=workflow_action_audit_path,
    )
    action_execution_gate = _product_trace_action_execution_gate_summary(
        report,
        report_path=report_path,
        action_execution_gate_report_path=workflow_action_execution_path,
    )
    action_audit_report_path = (
        None
        if action_audit_gate.get("report_path") is None
        else Path(str(action_audit_gate["report_path"]))
    )
    action_audit_report: dict[str, Any] = {}
    action_audit_report_error = None
    if require_action_audit_gate and action_audit_report_path is not None:
        action_audit_report, action_audit_report_error = verification_context.load_json_object(
            action_audit_report_path
        )
    action_execution_report_path = (
        None
        if action_execution_gate.get("report_path") is None
        else Path(str(action_execution_gate["report_path"]))
    )
    action_execution_report: dict[str, Any] = {}
    action_execution_report_error = None
    if require_action_execution_gate and action_execution_report_path is not None:
        action_execution_report, action_execution_report_error = verification_context.load_json_object(
            action_execution_report_path
        )
    action_audit_manifest_artifact_present = None
    action_audit_manifest_artifact_path = None
    action_audit_manifest_error = None
    if require_action_audit_gate:
        (
            action_audit_manifest_artifact_present,
            action_audit_manifest_artifact_path,
            action_audit_manifest_error,
        ) = _artifact_manifest_artifact_path(
            manifest_path,
            artifact_name="action_audit_gate_report",
            verification_context=verification_context,
        )
    action_execution_manifest_artifact_present = None
    action_execution_manifest_artifact_path = None
    action_execution_manifest_error = None
    if require_action_execution_gate:
        (
            action_execution_manifest_artifact_present,
            action_execution_manifest_artifact_path,
            action_execution_manifest_error,
        ) = _artifact_manifest_artifact_path(
            manifest_path,
            artifact_name="action_execution_gate_report",
            verification_context=verification_context,
        )
    resolved_selector_path = (
        Path(selector_replay_report_path)
        if selector_replay_report_path is not None
        else workflow_selector_path
    )
    resolved_drift_path = (
        Path(product_runtime_drift_report_path)
        if product_runtime_drift_report_path is not None
        else workflow_drift_path
    )
    gate = _product_trace_replay_workflow_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        selector_replay_report_path=resolved_selector_path,
        product_runtime_drift_report_path=resolved_drift_path,
        action_audit_gate=action_audit_gate,
        action_audit_report_path=action_audit_report_path,
        action_audit_report=action_audit_report,
        action_audit_report_error=action_audit_report_error,
        action_audit_manifest_artifact_present=action_audit_manifest_artifact_present,
        action_audit_manifest_artifact_path=action_audit_manifest_artifact_path,
        action_audit_manifest_error=action_audit_manifest_error,
        require_action_audit_gate=require_action_audit_gate,
        action_execution_gate=action_execution_gate,
        action_execution_report_path=action_execution_report_path,
        action_execution_report=action_execution_report,
        action_execution_report_error=action_execution_report_error,
        action_execution_manifest_artifact_present=action_execution_manifest_artifact_present,
        action_execution_manifest_artifact_path=action_execution_manifest_artifact_path,
        action_execution_manifest_error=action_execution_manifest_error,
        require_action_execution_gate=require_action_execution_gate,
        allow_unverified=allow_unverified,
    )
    action_audit_gate = _mapping(gate.get("action_audit_gate"))
    action_execution_gate = _mapping(gate.get("action_execution_gate"))
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "source": product_trace_replay_workflow_source.get("source"),
        "registry": product_trace_replay_workflow_source.get("registry"),
        "record_key": product_trace_replay_workflow_source.get("record_key"),
        "record": product_trace_replay_workflow_source.get("record"),
        "workflow": report.get("workflow"),
        "report_status": report.get("status"),
        "selector_replay_report_path": (
            None if resolved_selector_path is None else str(resolved_selector_path)
        ),
        "selector_replay_report_source": (
            None
            if resolved_selector_path is None
            else ("explicit" if selector_replay_report_path is not None else "workflow")
        ),
        "product_runtime_drift_report_path": (
            None if resolved_drift_path is None else str(resolved_drift_path)
        ),
        "product_runtime_drift_report_source": (
            None
            if resolved_drift_path is None
            else ("explicit" if product_runtime_drift_report_path is not None else "workflow")
        ),
        "action_audit_gate": action_audit_gate,
        "action_audit_gate_report_path": action_audit_gate.get("report_path"),
        "require_action_audit_gate": bool(require_action_audit_gate),
        "action_execution_gate": action_execution_gate,
        "action_execution_gate_report_path": action_execution_gate.get("report_path"),
        "require_action_execution_gate": bool(require_action_execution_gate),
        "verification": verification,
        "gate": gate,
    }


def _resolve_product_trace_replay_workflow_source(
    *,
    product_trace_replay_workflow_path: str | Path | None,
    product_trace_replay_workflow_registry_path: str | Path | None,
    product_trace_replay_workflow_key: str | None,
    default_registry_path: str | Path,
) -> dict[str, Any] | None:
    if product_trace_replay_workflow_path is not None:
        if product_trace_replay_workflow_key is not None:
            raise ValueError(
                "product_trace_replay_workflow_path is mutually exclusive with "
                "product_trace_replay_workflow_key."
            )
        return {"source": "file", "path": Path(product_trace_replay_workflow_path)}
    if product_trace_replay_workflow_key is None:
        if product_trace_replay_workflow_registry_path is not None:
            raise ValueError(
                "product_trace_replay_workflow_registry_path requires "
                "product_trace_replay_workflow_key."
            )
        return None
    registry_path = Path(
        default_registry_path
        if product_trace_replay_workflow_registry_path is None
        else product_trace_replay_workflow_registry_path
    )
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get(str(product_trace_replay_workflow_key))
    if record.artifact_type != "report":
        raise ValueError(f"registry record {record.key()!r} is not a report.")
    return {
        "source": "registry",
        "registry": str(registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "path": _resolve_registry_record_path(registry_path, record),
    }


def _product_trace_replay_workflow_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    selector_replay_report_path: Path | None,
    product_runtime_drift_report_path: Path | None,
    action_audit_gate: Mapping[str, Any],
    action_audit_report_path: Path | None,
    action_audit_report: Mapping[str, Any],
    action_audit_report_error: str | None,
    action_audit_manifest_artifact_present: bool | None,
    action_audit_manifest_artifact_path: Path | None,
    action_audit_manifest_error: str | None,
    require_action_audit_gate: bool,
    action_execution_gate: Mapping[str, Any],
    action_execution_report_path: Path | None,
    action_execution_report: Mapping[str, Any],
    action_execution_report_error: str | None,
    action_execution_manifest_artifact_present: bool | None,
    action_execution_manifest_artifact_path: Path | None,
    action_execution_manifest_error: str | None,
    require_action_execution_gate: bool,
    allow_unverified: bool,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"product trace replay workflow report could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("product trace replay workflow artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("product trace replay workflow manifest verification failed")
    if report.get("workflow") != "product_trace_replay_workflow":
        failures.append(
            f"product trace replay workflow is {report.get('workflow')!r}, "
            "expected 'product_trace_replay_workflow'"
        )
    if report.get("status") != "promote":
        failures.append(
            f"product trace replay workflow status is {report.get('status')!r}, expected 'promote'"
        )
    if selector_replay_report_path is None:
        failures.append("product trace replay workflow selector replay report is missing")
    if product_runtime_drift_report_path is None:
        failures.append("product trace replay workflow runtime drift report is missing")
    if require_action_audit_gate:
        if action_audit_gate.get("gate_enabled") is not True:
            failures.append("product trace replay workflow action-audit gate is not enabled")
        if (
            action_audit_gate.get("status") != "promote"
            or action_audit_gate.get("passed") is not True
        ):
            failures.append(
                "product trace replay workflow action-audit gate status is "
                f"{action_audit_gate.get('status')!r}, expected 'promote'"
            )
        if action_audit_gate.get("report_path") is None:
            failures.append("product trace replay workflow action-audit gate report is missing")
        elif action_audit_report_error is not None:
            failures.append(
                "product trace replay workflow action-audit gate report could not be loaded: "
                f"{action_audit_report_error}"
            )
        else:
            action_audit_report_summary = _mapping(action_audit_report.get("summary"))
            if action_audit_report.get("workflow") != "product_trace_action_audit_gate":
                failures.append(
                    "product trace replay workflow action-audit gate report workflow is "
                    f"{action_audit_report.get('workflow')!r}, expected "
                    "'product_trace_action_audit_gate'"
                )
            if action_audit_report.get("status") != "promote":
                failures.append(
                    "product trace replay workflow action-audit gate report status is "
                    f"{action_audit_report.get('status')!r}, expected 'promote'"
                )
            if action_audit_report_summary.get("gate_enabled") is not True:
                failures.append(
                    "product trace replay workflow action-audit gate report is not enabled"
                )
            if action_audit_report_summary.get("passed") is not True:
                failures.append(
                    "product trace replay workflow action-audit gate report did not pass"
                )
            failures.extend(
                _action_audit_report_internal_failures(
                    action_audit_report,
                    action_audit_report_summary=action_audit_report_summary,
                )
            )
            failures.extend(
                _action_audit_report_path_failures(
                    action_audit_report,
                    action_audit_report_path=action_audit_report_path,
                )
            )
            failures.extend(
                _action_audit_summary_mismatches(
                    action_audit_gate,
                    action_audit_report_summary,
                )
            )
        if not allow_unverified:
            if action_audit_manifest_error is not None:
                failures.append(
                    "product trace replay workflow artifact manifest could not be inspected for "
                    f"action-audit gate report: {action_audit_manifest_error}"
                )
            elif action_audit_manifest_artifact_present is not True:
                failures.append(
                    "product trace replay workflow artifact manifest does not include "
                    "action-audit gate report"
                )
            elif not _paths_match(action_audit_manifest_artifact_path, action_audit_report_path):
                failures.append(
                    "product trace replay workflow artifact manifest action-audit gate report "
                    f"path is {action_audit_manifest_artifact_path!s}, expected "
                    f"{action_audit_report_path!s}"
                )
    if require_action_execution_gate:
        if action_execution_gate.get("gate_enabled") is not True:
            failures.append("product trace replay workflow action-execution gate is not enabled")
        if (
            action_execution_gate.get("status") != "promote"
            or action_execution_gate.get("passed") is not True
        ):
            failures.append(
                "product trace replay workflow action-execution gate status is "
                f"{action_execution_gate.get('status')!r}, expected 'promote'"
            )
        if action_execution_gate.get("report_path") is None:
            failures.append("product trace replay workflow action-execution gate report is missing")
        elif action_execution_report_error is not None:
            failures.append(
                "product trace replay workflow action-execution gate report could not be loaded: "
                f"{action_execution_report_error}"
            )
        else:
            action_execution_report_summary = _mapping(action_execution_report.get("summary"))
            if action_execution_report.get("workflow") != "product_trace_action_execution_gate":
                failures.append(
                    "product trace replay workflow action-execution gate report workflow is "
                    f"{action_execution_report.get('workflow')!r}, expected "
                    "'product_trace_action_execution_gate'"
                )
            if action_execution_report.get("status") != "promote":
                failures.append(
                    "product trace replay workflow action-execution gate report status is "
                    f"{action_execution_report.get('status')!r}, expected 'promote'"
                )
            if action_execution_report_summary.get("gate_enabled") is not True:
                failures.append(
                    "product trace replay workflow action-execution gate report is not enabled"
                )
            if action_execution_report_summary.get("passed") is not True:
                failures.append(
                    "product trace replay workflow action-execution gate report did not pass"
                )
            failures.extend(
                _action_execution_report_internal_failures(
                    action_execution_report,
                    action_execution_report_summary=action_execution_report_summary,
                )
            )
            failures.extend(
                _action_execution_report_path_failures(
                    action_execution_report,
                    action_execution_report_path=action_execution_report_path,
                )
            )
            failures.extend(
                _action_execution_summary_mismatches(
                    action_execution_gate,
                    action_execution_report_summary,
                )
            )
        if not allow_unverified:
            if action_execution_manifest_error is not None:
                failures.append(
                    "product trace replay workflow artifact manifest could not be inspected for "
                    f"action-execution gate report: {action_execution_manifest_error}"
                )
            elif action_execution_manifest_artifact_present is not True:
                failures.append(
                    "product trace replay workflow artifact manifest does not include "
                    "action-execution gate report"
                )
            elif not _paths_match(action_execution_manifest_artifact_path, action_execution_report_path):
                failures.append(
                    "product trace replay workflow artifact manifest action-execution gate report "
                    f"path is {action_execution_manifest_artifact_path!s}, expected "
                    f"{action_execution_report_path!s}"
                )
    return {
        "passed": not failures,
        "blocking_reasons": failures,
        "require_action_audit_gate": bool(require_action_audit_gate),
        "action_audit_gate": dict(action_audit_gate),
        "action_audit_gate_report_status": action_audit_report.get("status"),
        "action_audit_gate_report_workflow": action_audit_report.get("workflow"),
        "action_audit_manifest_artifact_present": action_audit_manifest_artifact_present,
        "action_audit_manifest_artifact_path": (
            None
            if action_audit_manifest_artifact_path is None
            else str(action_audit_manifest_artifact_path)
        ),
        "require_action_execution_gate": bool(require_action_execution_gate),
        "action_execution_gate": dict(action_execution_gate),
        "action_execution_gate_report_status": action_execution_report.get("status"),
        "action_execution_gate_report_workflow": action_execution_report.get("workflow"),
        "action_execution_manifest_artifact_present": action_execution_manifest_artifact_present,
        "action_execution_manifest_artifact_path": (
            None
            if action_execution_manifest_artifact_path is None
            else str(action_execution_manifest_artifact_path)
        ),
    }


def _product_trace_action_audit_gate_summary(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    action_audit_gate_report_path: Path | None,
) -> dict[str, Any]:
    summary = _mapping(report.get("action_audit_gate"))
    if not summary:
        return {
            "status": None,
            "gate_enabled": None,
            "passed": None,
            "report_path": (
                None
                if action_audit_gate_report_path is None
                else str(action_audit_gate_report_path)
            ),
        }
    report_path_value = action_audit_gate_report_path
    if report_path_value is None:
        raw_report_path = summary.get("report_path") or _nested(
            report,
            "paths",
            "action_audit_gate_report",
        )
        if raw_report_path is not None:
            report_path_value = _resolve_path(raw_report_path, base_path=report_path)
    return {
        "status": summary.get("status"),
        "gate_enabled": summary.get("gate_enabled"),
        "passed": summary.get("passed"),
        "trace_count": summary.get("trace_count"),
        "failed_trace_count": summary.get("failed_trace_count"),
        "failed_trace_rate": summary.get("failed_trace_rate"),
        "error_count": summary.get("error_count"),
        "error_rate": summary.get("error_rate"),
        "missing_retrieval_action_rate": summary.get("missing_retrieval_action_rate"),
        "missing_plan_retrieval_query_rate": summary.get(
            "missing_plan_retrieval_query_rate"
        ),
        "malformed_payload_rate": summary.get("malformed_payload_rate"),
        "unexpected_action_rate": summary.get("unexpected_action_rate"),
        "unknown_claim_id_rate": summary.get("unknown_claim_id_rate"),
        "blocked_metric_count": summary.get("blocked_metric_count"),
        "checked_metric_count": summary.get("checked_metric_count"),
        "report_path": None if report_path_value is None else str(report_path_value),
    }


def _product_trace_action_execution_gate_summary(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    action_execution_gate_report_path: Path | None,
) -> dict[str, Any]:
    summary = _mapping(report.get("action_execution_gate"))
    if not summary:
        return {
            "status": None,
            "gate_enabled": None,
            "passed": None,
            "report_path": (
                None
                if action_execution_gate_report_path is None
                else str(action_execution_gate_report_path)
            ),
        }
    report_path_value = action_execution_gate_report_path
    if report_path_value is None:
        raw_report_path = summary.get("report_path") or _nested(
            report,
            "paths",
            "action_execution_gate_report",
        )
        if raw_report_path is not None:
            report_path_value = _resolve_path(raw_report_path, base_path=report_path)
    return {
        "status": summary.get("status"),
        "gate_enabled": summary.get("gate_enabled"),
        "passed": summary.get("passed"),
        "trace_count": summary.get("trace_count"),
        "available_trace_count": summary.get("available_trace_count"),
        "alignment_available_trace_count": summary.get("alignment_available_trace_count"),
        "alignment_failed_trace_count": summary.get("alignment_failed_trace_count"),
        "alignment_failed_trace_rate": summary.get("alignment_failed_trace_rate"),
        "planned_action_count": summary.get("planned_action_count"),
        "result_count": summary.get("result_count"),
        "missing_result_count": summary.get("missing_result_count"),
        "missing_result_rate": summary.get("missing_result_rate"),
        "unexpected_result_count": summary.get("unexpected_result_count"),
        "unexpected_result_rate": summary.get("unexpected_result_rate"),
        "request_id_mismatch_count": summary.get("request_id_mismatch_count"),
        "request_id_mismatch_rate": summary.get("request_id_mismatch_rate"),
        "blocked_metric_count": summary.get("blocked_metric_count"),
        "checked_metric_count": summary.get("checked_metric_count"),
        "report_path": None if report_path_value is None else str(report_path_value),
    }


def _action_audit_summary_mismatches(
    workflow_summary: Mapping[str, Any],
    report_summary: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    for field in (
        "gate_enabled",
        "passed",
        "trace_count",
        "available_trace_count",
        "failed_trace_count",
        "failed_trace_rate",
        "error_count",
        "error_rate",
        "missing_retrieval_action_rate",
        "missing_plan_retrieval_query_rate",
        "malformed_payload_rate",
        "unexpected_action_rate",
        "unknown_claim_id_rate",
        "blocked_metric_count",
        "checked_metric_count",
    ):
        workflow_value = workflow_summary.get(field)
        report_value = report_summary.get(field)
        if workflow_value is None or report_value is None:
            continue
        if workflow_value != report_value:
            failures.append(
                "product trace replay workflow action-audit summary field "
                f"{field!r} is {workflow_value!r}, but child report has {report_value!r}"
            )
    return failures


def _action_execution_summary_mismatches(
    workflow_summary: Mapping[str, Any],
    report_summary: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    for field in (
        "gate_enabled",
        "passed",
        "trace_count",
        "available_trace_count",
        "alignment_available_trace_count",
        "alignment_failed_trace_count",
        "alignment_failed_trace_rate",
        "planned_action_count",
        "result_count",
        "missing_result_count",
        "missing_result_rate",
        "unexpected_result_count",
        "unexpected_result_rate",
        "request_id_mismatch_count",
        "request_id_mismatch_rate",
        "blocked_metric_count",
        "checked_metric_count",
    ):
        workflow_value = workflow_summary.get(field)
        report_value = report_summary.get(field)
        if workflow_value is None or report_value is None:
            continue
        if workflow_value != report_value:
            failures.append(
                "product trace replay workflow action-execution summary field "
                f"{field!r} is {workflow_value!r}, but child report has {report_value!r}"
            )
    return failures


def _action_audit_report_internal_failures(
    action_audit_report: Mapping[str, Any],
    *,
    action_audit_report_summary: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    report_status = action_audit_report.get("status")
    decision = _mapping(action_audit_report.get("decision"))
    decision_status = decision.get("status")
    if decision_status is not None and decision_status != report_status:
        failures.append(
            "product trace replay workflow action-audit gate report decision status is "
            f"{decision_status!r}, expected {report_status!r}"
        )
    decision_passed = decision.get("passed")
    if decision_passed is not None and decision_passed != action_audit_report_summary.get("passed"):
        failures.append(
            "product trace replay workflow action-audit gate report decision passed is "
            f"{decision_passed!r}, expected {action_audit_report_summary.get('passed')!r}"
        )
    blocking_reasons = _tuple_or_empty_sequence(decision.get("blocking_reasons"))
    if report_status == "promote" and blocking_reasons:
        failures.append(
            "product trace replay workflow action-audit gate report promoted with "
            f"{len(blocking_reasons)} blocking reasons"
        )
    summary_blocked_metric_count = action_audit_report_summary.get("blocked_metric_count")
    if summary_blocked_metric_count is not None and blocking_reasons:
        if summary_blocked_metric_count != len(blocking_reasons):
            failures.append(
                "product trace replay workflow action-audit gate report blocked metric count is "
                f"{summary_blocked_metric_count!r}, but decision has {len(blocking_reasons)} "
                "blocking reasons"
            )

    checks = tuple(
        _mapping(check)
        for check in _tuple_or_empty_sequence(action_audit_report.get("checks"))
        if isinstance(check, Mapping)
    )
    blocked_checks = tuple(check for check in checks if check.get("status") == "blocked")
    if report_status == "promote" and blocked_checks:
        failures.append(
            "product trace replay workflow action-audit gate report promoted with "
            f"{len(blocked_checks)} blocked checks"
        )
    summary_checked_metric_count = action_audit_report_summary.get("checked_metric_count")
    if summary_checked_metric_count is not None and checks:
        if summary_checked_metric_count != len(checks):
            failures.append(
                "product trace replay workflow action-audit gate report checked metric count is "
                f"{summary_checked_metric_count!r}, but report has {len(checks)} checks"
            )
    if summary_blocked_metric_count is not None and checks:
        if summary_blocked_metric_count != len(blocked_checks):
            failures.append(
                "product trace replay workflow action-audit gate report blocked metric count is "
                f"{summary_blocked_metric_count!r}, but report has {len(blocked_checks)} "
                "blocked checks"
            )
    return failures


def _action_execution_report_internal_failures(
    action_execution_report: Mapping[str, Any],
    *,
    action_execution_report_summary: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    report_status = action_execution_report.get("status")
    decision = _mapping(action_execution_report.get("decision"))
    decision_status = decision.get("status")
    if decision_status is not None and decision_status != report_status:
        failures.append(
            "product trace replay workflow action-execution gate report decision status is "
            f"{decision_status!r}, expected {report_status!r}"
        )
    decision_passed = decision.get("passed")
    if (
        decision_passed is not None
        and decision_passed != action_execution_report_summary.get("passed")
    ):
        failures.append(
            "product trace replay workflow action-execution gate report decision passed is "
            f"{decision_passed!r}, expected {action_execution_report_summary.get('passed')!r}"
        )
    blocking_reasons = _tuple_or_empty_sequence(decision.get("blocking_reasons"))
    if report_status == "promote" and blocking_reasons:
        failures.append(
            "product trace replay workflow action-execution gate report promoted with "
            f"{len(blocking_reasons)} blocking reasons"
        )
    summary_blocked_metric_count = action_execution_report_summary.get("blocked_metric_count")
    if summary_blocked_metric_count is not None and blocking_reasons:
        if summary_blocked_metric_count != len(blocking_reasons):
            failures.append(
                "product trace replay workflow action-execution gate report blocked metric count is "
                f"{summary_blocked_metric_count!r}, but decision has {len(blocking_reasons)} "
                "blocking reasons"
            )

    checks = tuple(
        _mapping(check)
        for check in _tuple_or_empty_sequence(action_execution_report.get("checks"))
        if isinstance(check, Mapping)
    )
    blocked_checks = tuple(check for check in checks if check.get("status") == "blocked")
    if report_status == "promote" and blocked_checks:
        failures.append(
            "product trace replay workflow action-execution gate report promoted with "
            f"{len(blocked_checks)} blocked checks"
        )
    summary_checked_metric_count = action_execution_report_summary.get("checked_metric_count")
    if summary_checked_metric_count is not None and checks:
        if summary_checked_metric_count != len(checks):
            failures.append(
                "product trace replay workflow action-execution gate report checked metric count is "
                f"{summary_checked_metric_count!r}, but report has {len(checks)} checks"
            )
    if summary_blocked_metric_count is not None and checks:
        if summary_blocked_metric_count != len(blocked_checks):
            failures.append(
                "product trace replay workflow action-execution gate report blocked metric count is "
                f"{summary_blocked_metric_count!r}, but report has {len(blocked_checks)} "
                "blocked checks"
            )
    return failures


def _action_audit_report_path_failures(
    action_audit_report: Mapping[str, Any],
    *,
    action_audit_report_path: Path | None,
) -> list[str]:
    raw_report_path = _nested(action_audit_report, "paths", "report")
    if raw_report_path is None:
        return ["product trace replay workflow action-audit gate report path is missing"]
    expected_path = (
        None
        if action_audit_report_path is None
        else _resolve_path(raw_report_path, base_path=action_audit_report_path)
    )
    if not _paths_match(expected_path, action_audit_report_path):
        return [
            "product trace replay workflow action-audit gate report self path is "
            f"{expected_path!s}, expected {action_audit_report_path!s}"
        ]
    return []


def _action_execution_report_path_failures(
    action_execution_report: Mapping[str, Any],
    *,
    action_execution_report_path: Path | None,
) -> list[str]:
    raw_report_path = _nested(action_execution_report, "paths", "report")
    if raw_report_path is None:
        return ["product trace replay workflow action-execution gate report path is missing"]
    expected_path = (
        None
        if action_execution_report_path is None
        else _resolve_path(raw_report_path, base_path=action_execution_report_path)
    )
    if not _paths_match(expected_path, action_execution_report_path):
        return [
            "product trace replay workflow action-execution gate report self path is "
            f"{expected_path!s}, expected {action_execution_report_path!s}"
        ]
    return []


def _paths_match(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.resolve() == right.resolve()


def _product_trace_replay_workflow_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _nested(report, "paths", "artifact_manifest")
    if raw_path is None:
        return None
    return _resolve_path(raw_path, base_path=report_path)


def _product_trace_replay_workflow_child_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    child_path_key: str,
) -> Path | None:
    raw_path = _nested(report, "paths", child_path_key)
    if raw_path is None:
        return None
    path = _resolve_path(raw_path, base_path=report_path)
    return path.resolve() if path.exists() else path


def _artifact_manifest_artifact_path(
    manifest_path: Path | None,
    *,
    artifact_name: str,
    verification_context: ArtifactVerificationContext,
) -> tuple[bool | None, Path | None, str | None]:
    if manifest_path is None:
        return None, None, "artifact manifest path is missing"
    manifest, error = verification_context.load_json_object(manifest_path)
    if error is not None:
        return None, None, error
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None, None, "artifact manifest artifacts is not a mapping"
    artifact = artifacts.get(artifact_name)
    if not isinstance(artifact, Mapping):
        return False, None, None
    raw_path = artifact.get("path")
    if raw_path is None:
        return True, None, "artifact manifest action-audit gate report path is missing"
    return True, _resolve_path(raw_path, base_path=manifest_path), None


def _feedback_policy_workflow_gate(
    *,
    feedback_policy_workflow_source: Mapping[str, Any] | None,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    min_matched_feedback_count: int | None,
    min_safety_coverage: float | None,
    max_unknown_safety_issue_rate: float | None,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if feedback_policy_workflow_source is None:
        return None
    report_path = Path(feedback_policy_workflow_source["path"])
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _feedback_policy_workflow_manifest_path(report, report_path=report_path)
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="feedback_policy_workflow_manifest",
        verification_context=verification_context,
    )
    gate = _feedback_policy_workflow_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        allow_unverified=allow_unverified,
        min_matched_feedback_count=min_matched_feedback_count,
        min_safety_coverage=min_safety_coverage,
        max_unknown_safety_issue_rate=max_unknown_safety_issue_rate,
    )
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "source": feedback_policy_workflow_source.get("source"),
        "registry": feedback_policy_workflow_source.get("registry"),
        "record_key": feedback_policy_workflow_source.get("record_key"),
        "record": feedback_policy_workflow_source.get("record"),
        "workflow": report.get("workflow"),
        "report_status": report.get("status"),
        "promotion_decision": _nested(report, "decision", "promotion_decision"),
        "candidate_control_policy": _first_present(
            _nested(report, "decision", "candidate_control_policy"),
            _nested(report, "paths", "candidate_control_policy"),
        ),
        "candidate_control_policy_config": _feedback_policy_candidate_control_policy_config(
            report
        ),
        "candidate_control_defaults": _first_present(
            _nested(report, "decision", "candidate_control_defaults"),
            _nested(report, "paths", "candidate_control_defaults"),
        ),
        "candidate_control_defaults_config": _feedback_policy_candidate_control_defaults_config(
            report
        ),
        "matched_feedback_count": _float_or_none(_feedback_policy_matched_feedback_count(report)),
        "accepted_but_wrong_rate": _float_or_none(
            _nested(report, "feedback_summary", "accepted_but_wrong_rate", "estimate")
        ),
        "retrieved_failure_rate": _float_or_none(
            _nested(report, "feedback_summary", "retrieved_failure_rate", "estimate")
        ),
        "abstain_false_positive_rate": _float_or_none(
            _nested(report, "feedback_summary", "abstain_false_positive_rate", "estimate")
        ),
        "final_answered_but_wrong_rate": _float_or_none(
            _nested(report, "feedback_summary", "final_answered_but_wrong_rate", "estimate")
        ),
        "final_answer_false_block_rate": _float_or_none(
            _nested(report, "feedback_summary", "final_answer_false_block_rate", "estimate")
        ),
        "safety_coverage_rate": _float_or_none(_feedback_policy_safety_coverage_rate(report)),
        "unknown_safety_issue_rate": _float_or_none(_feedback_policy_unknown_safety_issue_rate(report)),
        "verification": verification,
        "gate": gate,
    }


def _selfcheck_signal_fusion_workflow_gate(
    *,
    selfcheck_signal_fusion_workflow_source: Mapping[str, Any] | None,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if selfcheck_signal_fusion_workflow_source is None:
        return None
    report_path = Path(selfcheck_signal_fusion_workflow_source["path"])
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _selfcheck_signal_fusion_workflow_manifest_path(
        report,
        report_path=report_path,
    )
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="selfcheck_signal_fusion_workflow_manifest",
        verification_context=verification_context,
    )
    gate = _selfcheck_signal_fusion_workflow_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        allow_unverified=allow_unverified,
    )
    sample_quality = _mapping(report.get("sample_quality"))
    fusion_summary = _mapping(report.get("fusion_summary"))
    fusion_runs = tuple(fusion_summary.get("runs") or ())
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "source": selfcheck_signal_fusion_workflow_source.get("source"),
        "registry": selfcheck_signal_fusion_workflow_source.get("registry"),
        "record_key": selfcheck_signal_fusion_workflow_source.get("record_key"),
        "record": selfcheck_signal_fusion_workflow_source.get("record"),
        "workflow": report.get("workflow"),
        "sample_quality_status": sample_quality.get("status"),
        "sample_quality_passed": sample_quality.get("passed"),
        "sample_quality_failed_runs": tuple(sample_quality.get("failed_runs") or ()),
        "sample_quality_run_count": len(_mapping(sample_quality.get("runs"))),
        "sample_quality_runs": _selfcheck_sample_quality_run_summaries(sample_quality),
        "fusion_summary": fusion_summary,
        "fusion_run_count": len(fusion_runs),
        "geometry_fusion_artifact_count": len(_mapping(report.get("geometry_fusion_artifacts"))),
        "enhanced_score_dump_count": len(_mapping(report.get("enhanced_score_dumps"))),
        "verification": verification,
        "gate": gate,
    }


def _resolve_selfcheck_signal_fusion_workflow_source(
    *,
    selfcheck_signal_fusion_workflow_path: str | Path | None,
    selfcheck_signal_fusion_workflow_registry_path: str | Path | None,
    selfcheck_signal_fusion_workflow_key: str | None,
    default_registry_path: str | Path,
) -> dict[str, Any] | None:
    if selfcheck_signal_fusion_workflow_path is not None:
        if selfcheck_signal_fusion_workflow_key is not None:
            raise ValueError(
                "selfcheck_signal_fusion_workflow_path is mutually exclusive with "
                "selfcheck_signal_fusion_workflow_key."
            )
        return {"source": "file", "path": Path(selfcheck_signal_fusion_workflow_path)}
    if selfcheck_signal_fusion_workflow_key is None:
        if selfcheck_signal_fusion_workflow_registry_path is not None:
            raise ValueError(
                "selfcheck_signal_fusion_workflow_registry_path requires "
                "selfcheck_signal_fusion_workflow_key."
            )
        return None
    registry_path = Path(
        default_registry_path
        if selfcheck_signal_fusion_workflow_registry_path is None
        else selfcheck_signal_fusion_workflow_registry_path
    )
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get(str(selfcheck_signal_fusion_workflow_key))
    if record.artifact_type != "report":
        raise ValueError(f"registry record {record.key()!r} is not a report.")
    return {
        "source": "registry",
        "registry": str(registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "path": _resolve_registry_record_path(registry_path, record),
    }


def _selfcheck_signal_fusion_workflow_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    allow_unverified: bool,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"selfcheck signal fusion workflow report could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("selfcheck signal fusion workflow artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("selfcheck signal fusion workflow manifest verification failed")
    if report.get("workflow") != "selfcheck_signal_fusion_workflow":
        failures.append(
            "selfcheck signal fusion workflow is "
            f"{report.get('workflow')!r}, expected 'selfcheck_signal_fusion_workflow'"
        )
    sample_quality = _mapping(report.get("sample_quality"))
    if sample_quality.get("passed") is not True:
        failed_runs = tuple(sample_quality.get("failed_runs") or ())
        failures.append(
            "selfcheck signal fusion sample quality gate did not pass"
            + (f": failed_runs={failed_runs!r}" if failed_runs else "")
        )
    fusion_summary = _mapping(report.get("fusion_summary"))
    if not tuple(fusion_summary.get("runs") or ()):
        failures.append("selfcheck signal fusion summary has no runs")
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _selfcheck_signal_fusion_workflow_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _first_present(
        report.get("artifact_manifest_path"),
        _nested(report, "paths", "artifact_manifest"),
    )
    if raw_path is None:
        return None
    return _resolve_path(raw_path, base_path=report_path)


def _selfcheck_sample_quality_run_summaries(
    sample_quality: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    summaries = []
    for name, payload in _mapping(sample_quality.get("runs")).items():
        run = _mapping(payload)
        summaries.append({
            "name": str(name),
            "status": run.get("status"),
            "passed": run.get("passed"),
            "n_total": run.get("n_total"),
            "records_meeting_min_samples": run.get("records_meeting_min_samples"),
            "coverage": _float_or_none(run.get("coverage")),
            "average_samples_per_record": _float_or_none(
                run.get("average_samples_per_record")
            ),
            "not_applicable_rate": _float_or_none(run.get("not_applicable_rate")),
            "best_overlap_mean": _float_or_none(run.get("best_overlap_mean")),
        })
    return tuple(summaries)


def _uncertainty_escalation_workflow_gate(
    *,
    uncertainty_escalation_workflow_source: Mapping[str, Any] | None,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    min_records: int | None,
    min_trigger_rate: float | None,
    min_retrieval_evidence_rate: float | None,
    max_final_false_accept_rate: float | None,
    max_false_accept_delta: float | None,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if uncertainty_escalation_workflow_source is None:
        return None
    report_path = Path(uncertainty_escalation_workflow_source["path"])
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _uncertainty_escalation_workflow_manifest_path(
        report,
        report_path=report_path,
    )
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="uncertainty_escalation_workflow_manifest",
        verification_context=verification_context,
    )
    summary = _uncertainty_escalation_workflow_summary(report)
    gate = _uncertainty_escalation_workflow_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        summary=summary,
        allow_unverified=allow_unverified,
        min_records=min_records,
        min_trigger_rate=min_trigger_rate,
        min_retrieval_evidence_rate=min_retrieval_evidence_rate,
        max_final_false_accept_rate=max_final_false_accept_rate,
        max_false_accept_delta=max_false_accept_delta,
    )
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "source": uncertainty_escalation_workflow_source.get("source"),
        "registry": uncertainty_escalation_workflow_source.get("registry"),
        "record_key": uncertainty_escalation_workflow_source.get("record_key"),
        "record": uncertainty_escalation_workflow_source.get("record"),
        "workflow": report.get("workflow"),
        "record_count": _float_or_none(summary.get("record_count")),
        "triggered_records": _float_or_none(summary.get("triggered_records")),
        "trigger_rate": _float_or_none(summary.get("trigger_rate")),
        "retrieval_evidence_rate": _float_or_none(
            summary.get("retrieval_evidence_rate")
        ),
        "final_false_accept_rate": _float_or_none(
            summary.get("final_false_accept_rate")
        ),
        "false_accept_delta": _float_or_none(summary.get("false_accept_delta")),
        "accepted_false_delta": _float_or_none(summary.get("accepted_false_delta")),
        "verification": verification,
        "gate": gate,
    }


def _resolve_uncertainty_escalation_workflow_source(
    *,
    uncertainty_escalation_workflow_path: str | Path | None,
    uncertainty_escalation_workflow_registry_path: str | Path | None,
    uncertainty_escalation_workflow_key: str | None,
    default_registry_path: str | Path,
) -> dict[str, Any] | None:
    if uncertainty_escalation_workflow_path is not None:
        if uncertainty_escalation_workflow_key is not None:
            raise ValueError(
                "uncertainty_escalation_workflow_path is mutually exclusive with "
                "uncertainty_escalation_workflow_key."
            )
        return {"source": "file", "path": Path(uncertainty_escalation_workflow_path)}
    if uncertainty_escalation_workflow_key is None:
        if uncertainty_escalation_workflow_registry_path is not None:
            raise ValueError(
                "uncertainty_escalation_workflow_registry_path requires "
                "uncertainty_escalation_workflow_key."
            )
        return None
    registry_path = Path(
        default_registry_path
        if uncertainty_escalation_workflow_registry_path is None
        else uncertainty_escalation_workflow_registry_path
    )
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get(str(uncertainty_escalation_workflow_key))
    if record.artifact_type != "report":
        raise ValueError(f"registry record {record.key()!r} is not a report.")
    return {
        "source": "registry",
        "registry": str(registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "path": _resolve_registry_record_path(registry_path, record),
    }


def _uncertainty_escalation_workflow_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    summary: Mapping[str, Any],
    allow_unverified: bool,
    min_records: int | None,
    min_trigger_rate: float | None,
    min_retrieval_evidence_rate: float | None,
    max_final_false_accept_rate: float | None,
    max_false_accept_delta: float | None,
) -> dict[str, Any]:
    failures = []
    effective_min_records = 1 if min_records is None else int(min_records)
    if report_error is not None:
        failures.append(
            f"uncertainty escalation workflow report could not be loaded: {report_error}"
        )
    if manifest_path is None:
        failures.append("uncertainty escalation workflow artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("uncertainty escalation workflow manifest verification failed")
    if report.get("workflow") != "uncertainty_escalation_fixture_workflow":
        failures.append(
            "uncertainty escalation workflow is "
            f"{report.get('workflow')!r}, expected "
            "'uncertainty_escalation_fixture_workflow'"
        )
    record_count = _float_or_none(summary.get("record_count"))
    if record_count is None:
        failures.append("uncertainty escalation workflow record_count is missing")
    elif record_count < effective_min_records:
        failures.append(
            "uncertainty escalation workflow record count below "
            f"{effective_min_records}: {record_count!r}"
        )
    trigger_rate = _float_or_none(summary.get("trigger_rate"))
    if min_trigger_rate is not None:
        if trigger_rate is None:
            failures.append("uncertainty escalation workflow trigger_rate is missing")
        elif trigger_rate < min_trigger_rate:
            failures.append(
                "uncertainty escalation workflow trigger_rate below "
                f"{min_trigger_rate}: {trigger_rate!r}"
            )
    retrieval_rate = _float_or_none(summary.get("retrieval_evidence_rate"))
    if min_retrieval_evidence_rate is not None:
        if retrieval_rate is None:
            failures.append(
                "uncertainty escalation workflow retrieval_evidence_rate is missing"
            )
        elif retrieval_rate < min_retrieval_evidence_rate:
            failures.append(
                "uncertainty escalation workflow retrieval_evidence_rate below "
                f"{min_retrieval_evidence_rate}: {retrieval_rate!r}"
            )
    final_false_accept = _float_or_none(summary.get("final_false_accept_rate"))
    if max_final_false_accept_rate is not None:
        if final_false_accept is None:
            failures.append(
                "uncertainty escalation workflow final_false_accept_rate is missing"
            )
        elif final_false_accept > max_final_false_accept_rate:
            failures.append(
                "uncertainty escalation workflow final_false_accept_rate above "
                f"{max_final_false_accept_rate}: {final_false_accept!r}"
            )
    false_accept_delta = _float_or_none(summary.get("false_accept_delta"))
    if max_false_accept_delta is not None:
        if false_accept_delta is None:
            failures.append(
                "uncertainty escalation workflow false_accept_delta is missing"
            )
        elif false_accept_delta > max_false_accept_delta:
            failures.append(
                "uncertainty escalation workflow false_accept_delta above "
                f"{max_false_accept_delta}: {false_accept_delta!r}"
            )
    return {
        "passed": not failures,
        "blocking_reasons": failures,
        "policy": {
            "min_records": effective_min_records,
            "min_trigger_rate": min_trigger_rate,
            "min_retrieval_evidence_rate": min_retrieval_evidence_rate,
            "max_final_false_accept_rate": max_final_false_accept_rate,
            "max_false_accept_delta": max_false_accept_delta,
        },
    }


def _uncertainty_escalation_workflow_summary(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _mapping(_first_present(report.get("report"), report.get("summary")))
    return {
        "record_count": _first_present(
            summary.get("n_total"),
            _nested(report, "input", "record_count"),
        ),
        "triggered_records": _nested(
            summary,
            "uncertainty_escalation",
            "triggered_records",
        ),
        "trigger_rate": _nested(
            summary,
            "uncertainty_escalation",
            "trigger_rate",
            "estimate",
        ),
        "retrieval_evidence_rate": _nested(
            summary,
            "action_execution",
            "retrieval_evidence_rate",
            "estimate",
        ),
        "final_false_accept_rate": _nested(
            summary,
            "quality",
            "final",
            "false_accept_rate",
            "estimate",
        ),
        "false_accept_delta": _nested(
            summary,
            "quality",
            "delta",
            "false_accept_rate",
        ),
        "accepted_false_delta": _nested(
            summary,
            "quality",
            "delta",
            "accepted_false",
        ),
    }


def _uncertainty_escalation_workflow_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _first_present(
        _nested(report, "paths", "artifact_manifest"),
        report.get("artifact_manifest_path"),
    )
    if raw_path is None:
        sibling = report_path.parent / "artifact-manifest.json"
        return sibling if sibling.exists() else None
    return _resolve_path(raw_path, base_path=report_path)


def _resolve_feedback_policy_workflow_source(
    *,
    feedback_policy_workflow_path: str | Path | None,
    feedback_policy_workflow_registry_path: str | Path | None,
    feedback_policy_workflow_key: str | None,
    default_registry_path: str | Path,
) -> dict[str, Any] | None:
    if feedback_policy_workflow_path is not None:
        if feedback_policy_workflow_key is not None:
            raise ValueError(
                "feedback_policy_workflow_path is mutually exclusive with "
                "feedback_policy_workflow_key."
            )
        return {"source": "file", "path": Path(feedback_policy_workflow_path)}
    if feedback_policy_workflow_key is None:
        if feedback_policy_workflow_registry_path is not None:
            raise ValueError(
                "feedback_policy_workflow_registry_path requires "
                "feedback_policy_workflow_key."
            )
        return None
    registry_path = Path(
        default_registry_path
        if feedback_policy_workflow_registry_path is None
        else feedback_policy_workflow_registry_path
    )
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get(str(feedback_policy_workflow_key))
    if record.artifact_type != "report":
        raise ValueError(f"registry record {record.key()!r} is not a report.")
    return {
        "source": "registry",
        "registry": str(registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "path": _resolve_registry_record_path(registry_path, record),
    }


def _feedback_policy_workflow_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    allow_unverified: bool,
    min_matched_feedback_count: int | None,
    min_safety_coverage: float | None,
    max_unknown_safety_issue_rate: float | None,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"feedback policy workflow report could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("feedback policy workflow artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("feedback policy workflow manifest verification failed")
    if report.get("workflow") != "feedback_policy_workflow":
        failures.append(
            f"feedback policy workflow is {report.get('workflow')!r}, "
            "expected 'feedback_policy_workflow'"
        )
    status = report.get("status")
    if status not in {"recommend", "observed"}:
        failures.append(
            f"feedback policy workflow status is {status!r}, expected 'recommend' or 'observed'"
        )
    promotion_decision = _nested(report, "decision", "promotion_decision")
    if promotion_decision not in {"promote_candidate_policy", "keep_current_policy"}:
        failures.append(
            "feedback policy workflow promotion_decision is "
            f"{promotion_decision!r}, expected a policy recommendation or keep-current decision"
        )
    if status == "recommend":
        if _first_present(
            _nested(report, "decision", "candidate_control_policy"),
            _nested(report, "paths", "candidate_control_policy"),
        ) is None:
            failures.append("feedback policy workflow candidate control policy is missing")
        policy_config = _feedback_policy_candidate_control_policy_config(report)
        if not policy_config:
            failures.append("feedback policy workflow candidate control policy config is missing")
        else:
            try:
                ControlPolicyConfig.from_dict(policy_config)
            except ValueError as exc:
                failures.append(
                    "feedback policy workflow candidate control policy config "
                    f"is invalid: {exc}"
                )
        if _first_present(
            _nested(report, "decision", "candidate_control_defaults"),
            _nested(report, "paths", "candidate_control_defaults"),
        ) is None:
            failures.append("feedback policy workflow candidate control defaults are missing")

    matched = _float_or_none(_feedback_policy_matched_feedback_count(report))
    if min_matched_feedback_count is not None:
        if matched is None:
            failures.append("feedback policy workflow matched feedback count is missing")
        elif matched < min_matched_feedback_count:
            failures.append(
                "feedback policy workflow matched feedback count below "
                f"{min_matched_feedback_count}: {matched}"
            )
    safety_coverage = _float_or_none(_feedback_policy_safety_coverage_rate(report))
    if min_safety_coverage is not None:
        if safety_coverage is None:
            failures.append("feedback policy workflow safety coverage rate is missing")
        elif safety_coverage < min_safety_coverage:
            failures.append(
                "feedback policy workflow safety coverage below "
                f"{min_safety_coverage}: {safety_coverage}"
            )
    unknown_rate = _float_or_none(_feedback_policy_unknown_safety_issue_rate(report))
    if max_unknown_safety_issue_rate is not None:
        if unknown_rate is None:
            failures.append("feedback policy workflow unknown safety issue rate is missing")
        elif unknown_rate > max_unknown_safety_issue_rate:
            failures.append(
                "feedback policy workflow unknown safety issue rate above "
                f"{max_unknown_safety_issue_rate}: {unknown_rate}"
            )
    return {
        "passed": not failures,
        "blocking_reasons": failures,
        "policy": {
            "min_matched_feedback_count": min_matched_feedback_count,
            "min_safety_coverage": min_safety_coverage,
            "max_unknown_safety_issue_rate": max_unknown_safety_issue_rate,
        },
    }


def _feedback_policy_workflow_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _nested(report, "paths", "artifact_manifest")
    if raw_path is None:
        return None
    return _resolve_path(raw_path, base_path=report_path)


def _feedback_policy_matched_feedback_count(report: Mapping[str, Any]) -> Any:
    return _first_present(
        _nested(report, "decision", "matched_feedback_count"),
        _nested(report, "feedback_summary", "trace_matched_feedback_count"),
        _nested(report, "replay_summary", "matched_feedback_count"),
    )


def _feedback_policy_safety_coverage_rate(report: Mapping[str, Any]) -> Any:
    return _first_present(
        _nested(report, "decision", "safety_coverage_rate"),
        _nested(report, "replay_summary", "safety_coverage_rate", "estimate"),
    )


def _feedback_policy_unknown_safety_issue_rate(report: Mapping[str, Any]) -> Any:
    return _first_present(
        _nested(report, "decision", "unknown_safety_issue_rate"),
        _nested(report, "replay_summary", "unknown_safety_issue_rate", "estimate"),
    )


def _feedback_policy_candidate_control_policy_config(report: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(
        _first_present(
            _nested(report, "decision", "candidate_control_policy_config"),
            _nested(report, "recommendation", "candidate_control_policy_config"),
        )
    )


def _feedback_policy_candidate_control_defaults_config(report: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(
        _first_present(
            _nested(report, "decision", "candidate_control_defaults_config"),
            _nested(report, "recommendation", "candidate_control_defaults"),
        )
    )


def _selector_replay_gate(
    *,
    selector_replay_report_path: str | Path | None,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if selector_replay_report_path is None:
        return None
    report_path = Path(selector_replay_report_path)
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _selector_replay_manifest_path(report, report_path=report_path)
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="selector_replay_manifest",
        verification_context=verification_context,
    )
    recommended = _selector_replay_recommended_row(report)
    gate = _selector_replay_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        recommended=recommended,
        allow_unverified=allow_unverified,
    )
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "workflow": report.get("workflow"),
        "report_status": report.get("status"),
        "recommended_candidate": _nested(report, "decision", "recommended_candidate"),
        "recommended_policy_path": _nested(report, "decision", "recommended_policy_path"),
        "recommended": _selector_replay_summary(recommended),
        "verification": verification,
        "gate": gate,
    }


def _resolve_external_evidence_baseline_comparison_source(
    *,
    external_evidence_baseline_comparison_path: str | Path | None,
    external_evidence_baseline_comparison_registry_path: str | Path | None,
    external_evidence_baseline_comparison_key: str | None,
    default_registry_path: str | Path,
) -> dict[str, Any] | None:
    if external_evidence_baseline_comparison_path is not None:
        if external_evidence_baseline_comparison_key is not None:
            raise ValueError(
                "external_evidence_baseline_comparison_path is mutually exclusive with "
                "external_evidence_baseline_comparison_key."
            )
        return {"source": "file", "path": Path(external_evidence_baseline_comparison_path)}
    if external_evidence_baseline_comparison_key is None:
        if external_evidence_baseline_comparison_registry_path is not None:
            raise ValueError(
                "external_evidence_baseline_comparison_registry_path requires "
                "external_evidence_baseline_comparison_key."
            )
        return None
    registry_path = Path(
        default_registry_path
        if external_evidence_baseline_comparison_registry_path is None
        else external_evidence_baseline_comparison_registry_path
    )
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get(str(external_evidence_baseline_comparison_key))
    if record.artifact_type != "report":
        raise ValueError(f"registry record {record.key()!r} is not a report.")
    return {
        "source": "registry",
        "registry": str(registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "path": _resolve_registry_record_path(registry_path, record),
    }


def _external_evidence_baseline_comparison_gate(
    *,
    external_evidence_baseline_comparison_source: Mapping[str, Any] | None,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if external_evidence_baseline_comparison_source is None:
        return None
    report_path = Path(external_evidence_baseline_comparison_source["path"])
    report, report_error = verification_context.load_json_object(report_path)
    decision = _mapping(report.get("decision"))
    route_comparison = _mapping(report.get("route_baseline_comparison"))
    text_redline = _mapping(report.get("text_redline_comparison"))
    gate = _external_evidence_baseline_comparison_report_gate(
        report=report,
        report_error=report_error,
    )
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "source": external_evidence_baseline_comparison_source.get("source"),
        "registry": external_evidence_baseline_comparison_source.get("registry"),
        "record_key": external_evidence_baseline_comparison_source.get("record_key"),
        "record": external_evidence_baseline_comparison_source.get("record"),
        "workflow": report.get("workflow"),
        "decision_status": decision.get("status"),
        "recommended_route": decision.get("recommended_route"),
        "recommended_route_record": decision.get("recommended_route_record"),
        "route_passed": route_comparison.get("passed"),
        "text_redline_passed": text_redline.get("passed"),
        "text_redline_run_count": text_redline.get("run_count"),
        "blocking_reasons": tuple(decision.get("blocking_reasons", ())),
        "gate": gate,
    }


def _external_evidence_baseline_comparison_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"external evidence baseline comparison could not be loaded: {report_error}")
    if report.get("workflow") != "external_evidence_baseline_comparison":
        failures.append(
            "external evidence baseline comparison workflow is "
            f"{report.get('workflow')!r}, expected 'external_evidence_baseline_comparison'"
        )
    decision = _mapping(report.get("decision"))
    if decision.get("status") != "promote":
        failures.append(
            "external evidence baseline comparison status is "
            f"{decision.get('status')!r}, expected 'promote'"
        )
    route_comparison = _mapping(report.get("route_baseline_comparison"))
    if route_comparison and route_comparison.get("passed") is not True:
        failures.append("external evidence route baseline comparison did not pass")
    text_redline = _mapping(report.get("text_redline_comparison"))
    if text_redline and text_redline.get("passed") is not True:
        failures.append("external evidence text redline comparison did not pass")
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _pre_generation_probe_comparison_gate(
    *,
    pre_generation_probe_comparison_source: Mapping[str, Any] | None,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if pre_generation_probe_comparison_source is None:
        return None
    report_path = Path(pre_generation_probe_comparison_source["path"])
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _pre_generation_probe_comparison_manifest_path(report, report_path=report_path)
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="pre_generation_probe_comparison_manifest",
        verification_context=verification_context,
    )
    promotion_gate = _mapping(report.get("promotion_gate"))
    leaderboard = tuple(
        _mapping(item)
        for item in report.get("leaderboard") or ()
        if isinstance(item, Mapping)
    )
    best_run = leaderboard[0] if leaderboard else {}
    gate = _pre_generation_probe_comparison_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        promotion_gate=promotion_gate,
        leaderboard=leaderboard,
        allow_unverified=allow_unverified,
    )
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "source": pre_generation_probe_comparison_source.get("source"),
        "registry": pre_generation_probe_comparison_source.get("registry"),
        "record_key": pre_generation_probe_comparison_source.get("record_key"),
        "record": pre_generation_probe_comparison_source.get("record"),
        "workflow": report.get("workflow"),
        "report_status": report.get("status"),
        "run_count": sum(1 for run in report.get("runs") or () if isinstance(run, Mapping)),
        "model_count": _float_or_none(promotion_gate.get("model_count")),
        "models": tuple(promotion_gate.get("models") or ()),
        "redline_passed": promotion_gate.get("redline_passed"),
        "redline_run_count": _float_or_none(promotion_gate.get("redline_run_count")),
        "best_run": {
            "name": best_run.get("name"),
            "model": best_run.get("model"),
            "recommended_layer": best_run.get("recommended_layer"),
            "test_label_auroc": _float_or_none(best_run.get("test_label_auroc")),
            "redline_best_signal": best_run.get("redline_best_signal"),
            "redline_best_auroc": _float_or_none(best_run.get("redline_best_auroc")),
            "redline_margin": _float_or_none(best_run.get("redline_margin")),
        },
        "blocking_reasons": tuple(gate.get("blocking_reasons", ())),
        "verification": verification,
        "gate": gate,
    }


def _pre_generation_probe_comparison_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    promotion_gate: Mapping[str, Any],
    leaderboard: Sequence[Mapping[str, Any]],
    allow_unverified: bool,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"pre-generation probe comparison report could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("pre-generation probe comparison artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("pre-generation probe comparison manifest verification failed")
    if report.get("workflow") != "pre_generation_probe_workflow_comparison":
        failures.append(
            "pre-generation probe comparison workflow is "
            f"{report.get('workflow')!r}, expected 'pre_generation_probe_workflow_comparison'"
        )
    if report.get("status") != "ready":
        failures.append(
            f"pre-generation probe comparison status is {report.get('status')!r}, expected 'ready'"
        )
    if not promotion_gate:
        failures.append("pre-generation probe comparison promotion_gate is missing")
    else:
        gate_failures = tuple(promotion_gate.get("failures", ()))
        if gate_failures:
            failures.append(
                "pre-generation probe comparison promotion gate did not pass"
                + _format_gate_reasons({"blocking_reasons": gate_failures})
            )
        model_count = _float_or_none(promotion_gate.get("model_count"))
        if model_count is None or model_count < 2:
            failures.append("pre-generation probe comparison model_count is below 2")
        redline_run_count = _float_or_none(promotion_gate.get("redline_run_count"))
        if redline_run_count is None or redline_run_count < 1:
            failures.append("pre-generation probe comparison redline evidence is missing")
        if promotion_gate.get("redline_passed") is not True:
            failures.append("pre-generation probe comparison redline gate did not pass")
    if not leaderboard:
        failures.append("pre-generation probe comparison leaderboard is missing")
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _pre_generation_probe_comparison_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _nested(report, "paths", "artifact_manifest")
    if raw_path is None:
        return None
    return _resolve_path(raw_path, base_path=report_path)


def _resolve_pre_generation_probe_comparison_source(
    *,
    pre_generation_probe_comparison_path: str | Path | None,
    pre_generation_probe_comparison_registry_path: str | Path | None,
    pre_generation_probe_comparison_key: str | None,
    default_registry_path: str | Path,
) -> dict[str, Any] | None:
    if pre_generation_probe_comparison_path is not None:
        if pre_generation_probe_comparison_key is not None:
            raise ValueError(
                "pre_generation_probe_comparison_path is mutually exclusive with "
                "pre_generation_probe_comparison_key."
            )
        return {"source": "file", "path": Path(pre_generation_probe_comparison_path)}
    if pre_generation_probe_comparison_key is None:
        if pre_generation_probe_comparison_registry_path is not None:
            raise ValueError(
                "pre_generation_probe_comparison_registry_path requires "
                "pre_generation_probe_comparison_key."
            )
        return None
    registry_path = Path(
        default_registry_path
        if pre_generation_probe_comparison_registry_path is None
        else pre_generation_probe_comparison_registry_path
    )
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get(str(pre_generation_probe_comparison_key))
    if record.artifact_type != "report":
        raise ValueError(f"registry record {record.key()!r} is not a report.")
    return {
        "source": "registry",
        "registry": str(registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "path": _resolve_registry_record_path(registry_path, record),
    }


def _claim_factuality_probe_comparison_gate(
    *,
    claim_factuality_probe_comparison_source: Mapping[str, Any] | None,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if claim_factuality_probe_comparison_source is None:
        return None
    report_path = Path(claim_factuality_probe_comparison_source["path"])
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _claim_factuality_probe_comparison_manifest_path(
        report,
        report_path=report_path,
    )
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="claim_factuality_probe_comparison_manifest",
        verification_context=verification_context,
    )
    promotion_gate = _mapping(report.get("promotion_gate"))
    leaderboard = tuple(
        _mapping(item)
        for item in report.get("leaderboard") or ()
        if isinstance(item, Mapping)
    )
    best_run = leaderboard[0] if leaderboard else {}
    gate = _claim_factuality_probe_comparison_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        promotion_gate=promotion_gate,
        leaderboard=leaderboard,
        allow_unverified=allow_unverified,
    )
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "source": claim_factuality_probe_comparison_source.get("source"),
        "registry": claim_factuality_probe_comparison_source.get("registry"),
        "record_key": claim_factuality_probe_comparison_source.get("record_key"),
        "record": claim_factuality_probe_comparison_source.get("record"),
        "workflow": report.get("workflow"),
        "report_status": report.get("status"),
        "run_count": sum(1 for run in report.get("runs") or () if isinstance(run, Mapping)),
        "model_count": _float_or_none(promotion_gate.get("model_count")),
        "models": tuple(promotion_gate.get("models") or ()),
        "dataset_count": _float_or_none(promotion_gate.get("dataset_count")),
        "datasets": tuple(promotion_gate.get("datasets") or ()),
        "redline_passed": promotion_gate.get("redline_passed"),
        "redline_run_count": _float_or_none(promotion_gate.get("redline_run_count")),
        "best_run": {
            "name": best_run.get("name"),
            "model": best_run.get("effective_model") or best_run.get("model"),
            "record_count": _float_or_none(best_run.get("record_count")),
            "recommended_layer": best_run.get("recommended_layer"),
            "test_label_auroc": _float_or_none(best_run.get("test_label_auroc")),
            "test_selective_accuracy": _float_or_none(best_run.get("test_selective_accuracy")),
            "test_selective_coverage": _float_or_none(best_run.get("test_selective_coverage")),
            "conformal_threshold": _float_or_none(best_run.get("conformal_threshold")),
            "redline_best_signal": best_run.get("redline_best_signal"),
            "redline_best_auroc": _float_or_none(best_run.get("redline_best_auroc")),
            "redline_margin": _float_or_none(best_run.get("redline_margin")),
        },
        "blocking_reasons": tuple(gate.get("blocking_reasons", ())),
        "verification": verification,
        "gate": gate,
    }


def _claim_factuality_probe_comparison_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    promotion_gate: Mapping[str, Any],
    leaderboard: Sequence[Mapping[str, Any]],
    allow_unverified: bool,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"claim factuality probe comparison report could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("claim factuality probe comparison artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("claim factuality probe comparison manifest verification failed")
    if report.get("workflow") != "claim_factuality_probe_workflow_comparison":
        failures.append(
            "claim factuality probe comparison workflow is "
            f"{report.get('workflow')!r}, expected 'claim_factuality_probe_workflow_comparison'"
        )
    if report.get("status") != "ready":
        failures.append(
            f"claim factuality probe comparison status is {report.get('status')!r}, expected 'ready'"
        )
    if not promotion_gate:
        failures.append("claim factuality probe comparison promotion_gate is missing")
    else:
        gate_failures = tuple(promotion_gate.get("failures", ()))
        if gate_failures:
            failures.append(
                "claim factuality probe comparison promotion gate did not pass"
                + _format_gate_reasons({"blocking_reasons": gate_failures})
            )
        model_count = _float_or_none(promotion_gate.get("model_count"))
        if model_count is None or model_count < 2:
            failures.append("claim factuality probe comparison model_count is below 2")
        redline_run_count = _float_or_none(promotion_gate.get("redline_run_count"))
        if redline_run_count is None or redline_run_count < 1:
            failures.append("claim factuality probe comparison redline evidence is missing")
        if promotion_gate.get("redline_passed") is not True:
            failures.append("claim factuality probe comparison redline gate did not pass")
    if not leaderboard:
        failures.append("claim factuality probe comparison leaderboard is missing")
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _claim_factuality_probe_comparison_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _nested(report, "paths", "artifact_manifest")
    if raw_path is None:
        return None
    return _resolve_path(raw_path, base_path=report_path)


def _resolve_claim_factuality_probe_comparison_source(
    *,
    claim_factuality_probe_comparison_path: str | Path | None,
    claim_factuality_probe_comparison_registry_path: str | Path | None,
    claim_factuality_probe_comparison_key: str | None,
    default_registry_path: str | Path,
) -> dict[str, Any] | None:
    if claim_factuality_probe_comparison_path is not None:
        if claim_factuality_probe_comparison_key is not None:
            raise ValueError(
                "claim_factuality_probe_comparison_path is mutually exclusive with "
                "claim_factuality_probe_comparison_key."
            )
        return {"source": "file", "path": Path(claim_factuality_probe_comparison_path)}
    if claim_factuality_probe_comparison_key is None:
        if claim_factuality_probe_comparison_registry_path is not None:
            raise ValueError(
                "claim_factuality_probe_comparison_registry_path requires "
                "claim_factuality_probe_comparison_key."
            )
        return None
    registry_path = Path(
        default_registry_path
        if claim_factuality_probe_comparison_registry_path is None
        else claim_factuality_probe_comparison_registry_path
    )
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get(str(claim_factuality_probe_comparison_key))
    if record.artifact_type != "report":
        raise ValueError(f"registry record {record.key()!r} is not a report.")
    return {
        "source": "registry",
        "registry": str(registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "path": _resolve_registry_record_path(registry_path, record),
    }


def _frontier_release_evidence_gate(
    *,
    frontier_release_evidence_source: Mapping[str, Any] | None,
    require_input_manifests: bool,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if frontier_release_evidence_source is None:
        return None
    report_path = Path(frontier_release_evidence_source["path"])
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _frontier_release_evidence_manifest_path(report, report_path=report_path)
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="frontier_release_evidence_manifest",
        verification_context=verification_context,
    )
    decision = _mapping(report.get("decision"))
    summary = _mapping(report.get("evidence_summary"))
    gate = _frontier_release_evidence_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        allow_unverified=allow_unverified,
        require_input_manifests=require_input_manifests,
    )
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "source": frontier_release_evidence_source.get("source"),
        "registry": frontier_release_evidence_source.get("registry"),
        "record_key": frontier_release_evidence_source.get("record_key"),
        "record": frontier_release_evidence_source.get("record"),
        "workflow": report.get("workflow"),
        "report_status": report.get("status"),
        "decision_status": decision.get("status"),
        "verifier_track_status": decision.get("verifier_track_status"),
        "abstention_track_status": decision.get("abstention_track_status"),
        "multiple_testing_track_status": decision.get("multiple_testing_track_status"),
        "citation_batch_track_status": decision.get("citation_batch_track_status"),
        "frontier_rerun_rollup_track_status": decision.get(
            "frontier_rerun_rollup_track_status"
        ),
        "base_verifier_track_status": decision.get("base_verifier_track_status"),
        "base_abstention_track_status": decision.get("base_abstention_track_status"),
        "base_detectability_track_status": decision.get(
            "base_detectability_track_status"
        ),
        "base_multiple_testing_track_status": decision.get(
            "base_multiple_testing_track_status"
        ),
        "frontier_rerun_rollup_promoted_tracks": tuple(
            decision.get("frontier_rerun_rollup_promoted_tracks", ())
        ),
        "run_names": tuple(summary.get("run_names", ())),
        "frontier_rerun_rollup_report_count": summary.get(
            "frontier_rerun_rollup_report_count"
        ),
        "frontier_rerun_rollup_candidate_count": summary.get(
            "frontier_rerun_rollup_candidate_count"
        ),
        "frontier_rerun_rollup_missing_report_count": summary.get(
            "frontier_rerun_rollup_missing_report_count"
        ),
        "frontier_rerun_rollup_invalid_report_count": summary.get(
            "frontier_rerun_rollup_invalid_report_count"
        ),
        "frontier_rerun_rollup_blocked_candidate_count": summary.get(
            "frontier_rerun_rollup_blocked_candidate_count"
        ),
        "frontier_rerun_rollup_promotion_ready_count": summary.get(
            "frontier_rerun_rollup_promotion_ready_count"
        ),
        "citation_batch_rollup_count": summary.get("citation_batch_rollup_count"),
        "citation_batch_expected_batch_count": summary.get(
            "citation_batch_expected_batch_count"
        ),
        "citation_batch_observed_batch_count": summary.get(
            "citation_batch_observed_batch_count"
        ),
        "citation_batch_missing_expected_batch_count": summary.get(
            "citation_batch_missing_expected_batch_count"
        ),
        "citation_batch_duplicate_batch_count": summary.get(
            "citation_batch_duplicate_batch_count"
        ),
        "citation_batch_unexpected_batch_count": summary.get(
            "citation_batch_unexpected_batch_count"
        ),
        "require_input_manifests": bool(require_input_manifests),
        "input_manifest_required": summary.get("input_manifest_required"),
        "input_manifest_required_count": summary.get("input_manifest_required_count"),
        "input_manifest_verified_count": summary.get("input_manifest_verified_count"),
        "input_manifest_failed_count": summary.get("input_manifest_failed_count"),
        "input_manifest_missing_count": summary.get("input_manifest_missing_count"),
        "input_manifest_failure_count": summary.get("input_manifest_failure_count"),
        "blocking_reasons": tuple(decision.get("blocking_reasons", ())),
        "verification": verification,
        "gate": gate,
    }


def _frontier_release_evidence_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    allow_unverified: bool,
    require_input_manifests: bool,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"frontier release evidence report could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("frontier release evidence artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("frontier release evidence manifest verification failed")
    if report.get("workflow") != "frontier_release_evidence_comparison":
        failures.append(
            f"frontier release evidence workflow is {report.get('workflow')!r}, "
            "expected 'frontier_release_evidence_comparison'"
        )
    if report.get("status") != "complete":
        failures.append(
            f"frontier release evidence status is {report.get('status')!r}, expected 'complete'"
        )
    decision = _mapping(report.get("decision"))
    decision_status = decision.get("status")
    if decision_status != "promote":
        failures.append(
            f"frontier release evidence decision status is {decision_status!r}, expected 'promote'"
        )
    for track in ("verifier_track_status", "abstention_track_status"):
        if decision.get(track) != "promote":
            failures.append(
                f"frontier release evidence {track} is {decision.get(track)!r}, expected 'promote'"
            )
    multiple_testing_track_status = decision.get("multiple_testing_track_status")
    if multiple_testing_track_status not in {None, "promote", "not_required"}:
        failures.append(
            "frontier release evidence multiple_testing_track_status is "
            f"{multiple_testing_track_status!r}, expected 'promote' or 'not_required'"
        )
    citation_batch_track_status = decision.get("citation_batch_track_status")
    if citation_batch_track_status not in {None, "promote", "not_required"}:
        failures.append(
            "frontier release evidence citation_batch_track_status is "
            f"{citation_batch_track_status!r}, expected 'promote' or 'not_required'"
        )
    summary = _mapping(report.get("evidence_summary"))
    run_count = _float_or_none(summary.get("run_count"))
    if run_count is None:
        failures.append("frontier release evidence run count is missing")
    elif run_count < 1:
        failures.append("frontier release evidence run count is zero")
    if require_input_manifests:
        failures.extend(_frontier_release_input_manifest_failures(summary))
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _frontier_release_input_manifest_failures(
    summary: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    required = summary.get("input_manifest_required")
    required_count = _float_or_none(summary.get("input_manifest_required_count"))
    verified_count = _float_or_none(summary.get("input_manifest_verified_count"))
    failed_count = _float_or_none(summary.get("input_manifest_failed_count"))
    missing_count = _float_or_none(summary.get("input_manifest_missing_count"))
    failure_count = _float_or_none(summary.get("input_manifest_failure_count"))
    if required is not True:
        failures.append(
            "frontier release evidence input manifests were not required by the source report"
        )
    if required_count is None or required_count < 1:
        failures.append("frontier release evidence input manifest required count is missing or zero")
    if verified_count is None:
        failures.append("frontier release evidence input manifest verified count is missing")
    if required_count is not None and verified_count is not None and verified_count != required_count:
        failures.append(
            "frontier release evidence input manifest verified count "
            f"{verified_count:g} does not match required count {required_count:g}"
        )
    if missing_count is None:
        failures.append("frontier release evidence input manifest missing count is missing")
    elif missing_count > 0:
        failures.append(
            f"frontier release evidence input manifest missing count {missing_count:g} is non-zero"
        )
    if failed_count is None:
        failures.append("frontier release evidence input manifest failed count is missing")
    elif failed_count > 0:
        failures.append(
            f"frontier release evidence input manifest failed count {failed_count:g} is non-zero"
        )
    if failure_count is None:
        failures.append("frontier release evidence input manifest failure count is missing")
    elif failure_count > 0:
        failures.append(
            f"frontier release evidence input manifest failure count {failure_count:g} is non-zero"
        )
    return failures


def _frontier_release_evidence_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _nested(report, "paths", "artifact_manifest")
    if raw_path is None:
        return None
    return _resolve_path(raw_path, base_path=report_path)


def _resolve_frontier_release_evidence_source(
    *,
    frontier_release_evidence_path: str | Path | None,
    frontier_release_evidence_registry_path: str | Path | None,
    frontier_release_evidence_key: str | None,
    default_registry_path: str | Path,
) -> dict[str, Any] | None:
    if frontier_release_evidence_path is not None:
        if frontier_release_evidence_key is not None:
            raise ValueError(
                "frontier_release_evidence_path is mutually exclusive with "
                "frontier_release_evidence_key."
            )
        return {"source": "file", "path": Path(frontier_release_evidence_path)}
    if frontier_release_evidence_key is None:
        if frontier_release_evidence_registry_path is not None:
            raise ValueError(
                "frontier_release_evidence_registry_path requires "
                "frontier_release_evidence_key."
            )
        return None
    registry_path = (
        default_registry_path
        if frontier_release_evidence_registry_path is None
        else frontier_release_evidence_registry_path
    )
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get(str(frontier_release_evidence_key))
    return {
        "source": "registry",
        "path": Path(record.path),
        "registry": str(registry_path),
        "record_key": str(frontier_release_evidence_key),
        "record": record.to_dict(),
    }


def _world_model_signal_workflow_gate(
    *,
    world_model_signal_workflow_source: Mapping[str, Any] | None,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if world_model_signal_workflow_source is None:
        return None
    report_path = Path(world_model_signal_workflow_source["path"])
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _world_model_signal_workflow_manifest_path(report, report_path=report_path)
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="world_model_signal_workflow_manifest",
        verification_context=verification_context,
    )
    release_gate = _mapping(report.get("release_gate"))
    gate = _world_model_signal_workflow_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        release_gate=release_gate,
        allow_unverified=allow_unverified,
    )
    trace_gap_summary = _mapping(
        _nested(release_gate, "score_summary", "world_model_trace_gap")
    )
    conflict_summary = _mapping(
        _nested(release_gate, "score_summary", "world_model_conflict")
    )
    calibrated = tuple(
        _mapping(item)
        for item in release_gate.get("calibrated_conflict_signals", ())
        if isinstance(item, Mapping)
    )
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "source": world_model_signal_workflow_source.get("source"),
        "registry": world_model_signal_workflow_source.get("registry"),
        "record_key": world_model_signal_workflow_source.get("record_key"),
        "record": world_model_signal_workflow_source.get("record"),
        "workflow": report.get("workflow"),
        "release_gate_status": release_gate.get("status"),
        "release_gate_passed": release_gate.get("passed"),
        "trace_gap_max": _float_or_none(trace_gap_summary.get("max")),
        "conflict_positive_count": _float_or_none(conflict_summary.get("positive_count")),
        "calibrated_conflict_signal_count": len(calibrated),
        "calibrated_conflict_signals": calibrated,
        "blocking_reasons": tuple(release_gate.get("blocking_reasons", ())),
        "verification": verification,
        "gate": gate,
    }


def _world_model_signal_workflow_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    release_gate: Mapping[str, Any],
    allow_unverified: bool,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"world-model signal workflow report could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("world-model signal workflow artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("world-model signal workflow manifest verification failed")
    if report.get("workflow") != "world_model_signal_calibration_workflow":
        failures.append(
            "world-model signal workflow is "
            f"{report.get('workflow')!r}, expected 'world_model_signal_calibration_workflow'"
        )
    if not release_gate:
        failures.append("world-model signal workflow release_gate is missing")
    elif release_gate.get("passed") is not True:
        failures.append(
            "world-model signal workflow release gate did not pass"
            + _format_gate_reasons(release_gate)
        )
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _world_model_signal_workflow_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _first_present(
        report.get("artifact_manifest_path"),
        _nested(report, "paths", "artifact_manifest"),
    )
    if raw_path is None:
        return None
    return _resolve_path(raw_path, base_path=report_path)


def _resolve_world_model_signal_workflow_source(
    *,
    world_model_signal_workflow_path: str | Path | None,
    world_model_signal_workflow_registry_path: str | Path | None,
    world_model_signal_workflow_key: str | None,
    default_registry_path: str | Path,
) -> dict[str, Any] | None:
    if world_model_signal_workflow_path is not None:
        if world_model_signal_workflow_key is not None:
            raise ValueError(
                "world_model_signal_workflow_path is mutually exclusive with "
                "world_model_signal_workflow_key."
            )
        return {"source": "file", "path": Path(world_model_signal_workflow_path)}
    if world_model_signal_workflow_key is None:
        if world_model_signal_workflow_registry_path is not None:
            raise ValueError(
                "world_model_signal_workflow_registry_path requires "
                "world_model_signal_workflow_key."
            )
        return None
    registry_path = Path(
        default_registry_path
        if world_model_signal_workflow_registry_path is None
        else world_model_signal_workflow_registry_path
    )
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get(str(world_model_signal_workflow_key))
    if record.artifact_type != "report":
        raise ValueError(f"registry record {record.key()!r} is not a report.")
    return {
        "source": "registry",
        "registry": str(registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "path": _resolve_registry_record_path(registry_path, record),
    }


def _context_sensitivity_workflow_gate(
    *,
    context_sensitivity_workflow_source: Mapping[str, Any] | None,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if context_sensitivity_workflow_source is None:
        return None
    report_path = Path(context_sensitivity_workflow_source["path"])
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _context_sensitivity_workflow_manifest_path(
        report,
        report_path=report_path,
    )
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="context_sensitivity_workflow_manifest",
        verification_context=verification_context,
    )
    summary = _context_sensitivity_workflow_summary(report)
    gate = _context_sensitivity_workflow_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        summary=summary,
        allow_unverified=allow_unverified,
    )
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "source": context_sensitivity_workflow_source.get("source"),
        "registry": context_sensitivity_workflow_source.get("registry"),
        "record_key": context_sensitivity_workflow_source.get("record_key"),
        "record": context_sensitivity_workflow_source.get("record"),
        "workflow": report.get("workflow"),
        "paired_logprob_record_count": summary.get("paired_logprob_record_count"),
        "enriched_record_count": summary.get("enriched_record_count"),
        "enhanced_score_signal_count": summary.get("enhanced_score_signal_count"),
        "max_flagged_rate": summary.get("max_flagged_rate"),
        "mean_flagged_rate": summary.get("mean_flagged_rate"),
        "max_context_sensitivity_ratio": summary.get("max_context_sensitivity_ratio"),
        "manifest_verified": summary.get("manifest_verified"),
        "blocking_reasons": tuple(gate.get("blocking_reasons", ())),
        "verification": verification,
        "gate": gate,
    }


def _context_sensitivity_workflow_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    summary: Mapping[str, Any],
    allow_unverified: bool,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"context-sensitivity workflow report could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("context-sensitivity workflow artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("context-sensitivity workflow manifest verification failed")
    if report.get("workflow") != "context_sensitivity_workflow":
        failures.append(
            "context-sensitivity workflow is "
            f"{report.get('workflow')!r}, expected 'context_sensitivity_workflow'"
        )
    embedded_manifest_verified = summary.get("manifest_verified")
    if embedded_manifest_verified is False and not allow_unverified:
        failures.append("context-sensitivity workflow embedded manifest verification failed")
    paired_count = _float_or_none(summary.get("paired_logprob_record_count"))
    if paired_count is None:
        failures.append("context-sensitivity workflow paired_logprob_record_count is missing")
    elif paired_count < 1:
        failures.append("context-sensitivity workflow paired_logprob_record_count is zero")
    enriched_count = _float_or_none(summary.get("enriched_record_count"))
    if enriched_count is None:
        failures.append("context-sensitivity workflow enriched_record_count is missing")
    elif enriched_count < 1:
        failures.append("context-sensitivity workflow enriched_record_count is zero")
    signal_count = _float_or_none(summary.get("enhanced_score_signal_count"))
    if signal_count is None:
        failures.append("context-sensitivity workflow enhanced score signal count is missing")
    elif signal_count < 1:
        failures.append("context-sensitivity workflow enhanced score signal count is zero")
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _context_sensitivity_workflow_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    paired_summary = _mapping(report.get("paired_summary"))
    enrichment_summary = _mapping(report.get("enrichment_summary"))
    enhanced_summary = _mapping(report.get("enhanced_score_summary"))
    signal_summary = _mapping(report.get("signal_summary"))
    manifest_verification = _mapping(report.get("manifest_verification"))
    return {
        "paired_logprob_record_count": _float_or_none(
            paired_summary.get("paired_logprob_record_count")
        ),
        "enriched_record_count": _float_or_none(report.get("enriched_record_count")),
        "enhanced_score_signal_count": len(signal_summary) or len(enhanced_summary),
        "max_flagged_rate": _float_or_none(enrichment_summary.get("max_flagged_rate")),
        "mean_flagged_rate": _float_or_none(enrichment_summary.get("mean_flagged_rate")),
        "max_context_sensitivity_ratio": _float_or_none(
            enrichment_summary.get("max_context_sensitivity_ratio")
        ),
        "manifest_verified": (
            None
            if "passed" not in manifest_verification
            else bool(manifest_verification.get("passed"))
        ),
    }


def _context_sensitivity_workflow_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _first_present(
        report.get("artifact_manifest_path"),
        _nested(report, "paths", "artifact_manifest"),
    )
    if raw_path is None:
        sibling = report_path.parent / "artifact-manifest.json"
        return sibling if sibling.exists() else None
    return _resolve_path(raw_path, base_path=report_path)


def _resolve_context_sensitivity_workflow_source(
    *,
    context_sensitivity_workflow_path: str | Path | None,
    context_sensitivity_workflow_registry_path: str | Path | None,
    context_sensitivity_workflow_key: str | None,
    default_registry_path: str | Path,
) -> dict[str, Any] | None:
    if context_sensitivity_workflow_path is not None:
        if context_sensitivity_workflow_key is not None:
            raise ValueError(
                "context_sensitivity_workflow_path is mutually exclusive with "
                "context_sensitivity_workflow_key."
            )
        return {"source": "file", "path": Path(context_sensitivity_workflow_path)}
    if context_sensitivity_workflow_key is None:
        if context_sensitivity_workflow_registry_path is not None:
            raise ValueError(
                "context_sensitivity_workflow_registry_path requires "
                "context_sensitivity_workflow_key."
            )
        return None
    registry_path = Path(
        default_registry_path
        if context_sensitivity_workflow_registry_path is None
        else context_sensitivity_workflow_registry_path
    )
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get(str(context_sensitivity_workflow_key))
    if record.artifact_type != "report":
        raise ValueError(f"registry record {record.key()!r} is not a report.")
    return {
        "source": "registry",
        "registry": str(registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "path": _resolve_registry_record_path(registry_path, record),
    }


def _mechanism_handoff_evidence_bundle_gate(
    *,
    mechanism_handoff_evidence_bundle_source: Mapping[str, Any] | None,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if mechanism_handoff_evidence_bundle_source is None:
        return None
    report_path = Path(mechanism_handoff_evidence_bundle_source["path"])
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _mechanism_handoff_evidence_bundle_manifest_path(
        report,
        report_path=report_path,
    )
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="mechanism_handoff_evidence_bundle_manifest",
        verification_context=verification_context,
    )
    bundle_gate = _mapping(report.get("gate"))
    summary = _mapping(report.get("summary"))
    gate = _mechanism_handoff_evidence_bundle_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        bundle_gate=bundle_gate,
        summary=summary,
        allow_unverified=allow_unverified,
    )
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "source": mechanism_handoff_evidence_bundle_source.get("source"),
        "registry": mechanism_handoff_evidence_bundle_source.get("registry"),
        "record_key": mechanism_handoff_evidence_bundle_source.get("record_key"),
        "record": mechanism_handoff_evidence_bundle_source.get("record"),
        "workflow": report.get("workflow"),
        "report_status": report.get("status"),
        "handoff_count": _float_or_none(summary.get("handoff_count")),
        "trace_count": _float_or_none(summary.get("trace_count")),
        "target_count": _float_or_none(summary.get("target_count")),
        "target_coverage_rate": _float_or_none(summary.get("target_coverage_rate")),
        "source_citation_count": _float_or_none(summary.get("source_citation_count")),
        "verification_status_counts": dict(
            _mapping(summary.get("verification_status_counts"))
        ),
        "action_counts": dict(_mapping(summary.get("action_counts"))),
        "source_family_counts": dict(_mapping(summary.get("source_family_counts"))),
        "blocking_reasons": tuple(bundle_gate.get("blocking_reasons", ())),
        "verification": verification,
        "gate": gate,
    }


def _mechanism_handoff_evidence_bundle_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    bundle_gate: Mapping[str, Any],
    summary: Mapping[str, Any],
    allow_unverified: bool,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"mechanism handoff evidence bundle could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("mechanism handoff evidence bundle artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("mechanism handoff evidence bundle manifest verification failed")
    if report.get("workflow") != "mechanism_handoff_evidence_bundle":
        failures.append(
            "mechanism handoff evidence bundle workflow is "
            f"{report.get('workflow')!r}, expected 'mechanism_handoff_evidence_bundle'"
        )
    if report.get("status") != "promote":
        failures.append(
            f"mechanism handoff evidence bundle status is {report.get('status')!r}, "
            "expected 'promote'"
        )
    if not bundle_gate:
        failures.append("mechanism handoff evidence bundle gate is missing")
    elif bundle_gate.get("passed") is not True:
        failures.append(
            "mechanism handoff evidence bundle gate did not pass"
            + _format_gate_reasons(bundle_gate)
        )
    handoff_count = _float_or_none(summary.get("handoff_count"))
    trace_count = _float_or_none(summary.get("trace_count"))
    source_citation_count = _float_or_none(summary.get("source_citation_count"))
    if handoff_count is None or handoff_count < 1:
        failures.append("mechanism handoff evidence bundle handoff_count is below 1")
    if trace_count is None or trace_count < 1:
        failures.append("mechanism handoff evidence bundle trace_count is below 1")
    if source_citation_count is None:
        failures.append("mechanism handoff evidence bundle source_citation_count is missing")
    elif trace_count is not None and source_citation_count < trace_count:
        failures.append(
            "mechanism handoff evidence bundle source citations do not cover all traces: "
            f"{source_citation_count} < {trace_count}"
        )
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _mechanism_handoff_evidence_bundle_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _nested(report, "paths", "artifact_manifest")
    if raw_path is None:
        sibling = report_path.parent / "artifact-manifest.json"
        return sibling if sibling.exists() else None
    return _resolve_path(raw_path, base_path=report_path)


def _resolve_mechanism_handoff_evidence_bundle_source(
    *,
    mechanism_handoff_evidence_bundle_path: str | Path | None,
    mechanism_handoff_evidence_bundle_registry_path: str | Path | None,
    mechanism_handoff_evidence_bundle_key: str | None,
    default_registry_path: str | Path,
) -> dict[str, Any] | None:
    if mechanism_handoff_evidence_bundle_path is not None:
        if mechanism_handoff_evidence_bundle_key is not None:
            raise ValueError(
                "mechanism_handoff_evidence_bundle_path is mutually exclusive with "
                "mechanism_handoff_evidence_bundle_key."
            )
        return {"source": "file", "path": Path(mechanism_handoff_evidence_bundle_path)}
    if mechanism_handoff_evidence_bundle_key is None:
        if mechanism_handoff_evidence_bundle_registry_path is not None:
            raise ValueError(
                "mechanism_handoff_evidence_bundle_registry_path requires "
                "mechanism_handoff_evidence_bundle_key."
            )
        return None
    registry_path = Path(
        default_registry_path
        if mechanism_handoff_evidence_bundle_registry_path is None
        else mechanism_handoff_evidence_bundle_registry_path
    )
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get(str(mechanism_handoff_evidence_bundle_key))
    if record.artifact_type != "report":
        raise ValueError(f"registry record {record.key()!r} is not a report.")
    return {
        "source": "registry",
        "registry": str(registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "path": _resolve_registry_record_path(registry_path, record),
    }


def _pathway_intervention_workflow_gate(
    *,
    pathway_intervention_workflow_source: Mapping[str, Any] | None,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if pathway_intervention_workflow_source is None:
        return None
    report_path = Path(pathway_intervention_workflow_source["path"])
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _pathway_intervention_workflow_manifest_path(report, report_path=report_path)
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="pathway_intervention_workflow_manifest",
        verification_context=verification_context,
    )
    evidence_bundle = _mapping(report.get("evidence_bundle"))
    comparisons = _mapping(report.get("comparisons"))
    activation_ablation = _mapping(comparisons.get("activation_ablation"))
    source_patch = _mapping(comparisons.get("source_patch"))
    gate = _pathway_intervention_workflow_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        evidence_bundle=evidence_bundle,
        activation_ablation=activation_ablation,
        source_patch=source_patch,
        allow_unverified=allow_unverified,
    )
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "source": pathway_intervention_workflow_source.get("source"),
        "registry": pathway_intervention_workflow_source.get("registry"),
        "record_key": pathway_intervention_workflow_source.get("record_key"),
        "record": pathway_intervention_workflow_source.get("record"),
        "workflow": report.get("workflow"),
        "report_status": report.get("status"),
        "release_ready": evidence_bundle.get("release_ready"),
        "model": evidence_bundle.get("model"),
        "layer": evidence_bundle.get("layer"),
        "intervention_layer": evidence_bundle.get("intervention_layer"),
        "patch_layer": evidence_bundle.get("patch_layer"),
        "signals": tuple(evidence_bundle.get("signals", ())),
        "activation_ablation_gate_status": _nested(activation_ablation, "gate", "status"),
        "source_patch_gate_status": _nested(source_patch, "gate", "status"),
        "best_signals": dict(evidence_bundle.get("best_signals") or {}),
        "blocking_reasons": tuple(gate.get("blocking_reasons", ())),
        "verification": verification,
        "gate": gate,
    }


def _pathway_intervention_workflow_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    evidence_bundle: Mapping[str, Any],
    activation_ablation: Mapping[str, Any],
    source_patch: Mapping[str, Any],
    allow_unverified: bool,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"pathway intervention workflow report could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("pathway intervention workflow artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("pathway intervention workflow manifest verification failed")
    if report.get("workflow") != "pathway_intervention_workflow":
        failures.append(
            "pathway intervention workflow is "
            f"{report.get('workflow')!r}, expected 'pathway_intervention_workflow'"
        )
    if report.get("status") != "complete":
        failures.append(
            f"pathway intervention workflow status is {report.get('status')!r}, expected 'complete'"
        )
    if evidence_bundle.get("release_ready") is not True:
        failures.append("pathway intervention workflow evidence_bundle.release_ready is not true")
    for name, comparison in (
        ("activation_ablation", activation_ablation),
        ("source_patch", source_patch),
    ):
        gate_status = _nested(comparison, "gate", "status")
        if gate_status != "promote":
            failures.append(
                f"pathway intervention workflow {name} gate status is "
                f"{gate_status!r}, expected 'promote'"
            )
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _pathway_intervention_workflow_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _first_present(
        report.get("artifact_manifest_path"),
        _nested(report, "paths", "artifact_manifest"),
    )
    if raw_path is None:
        sibling = report_path.parent / "artifact-manifest.json"
        return sibling if sibling.exists() else None
    return _resolve_path(raw_path, base_path=report_path)


def _resolve_pathway_intervention_workflow_source(
    *,
    pathway_intervention_workflow_path: str | Path | None,
    pathway_intervention_workflow_registry_path: str | Path | None,
    pathway_intervention_workflow_key: str | None,
    default_registry_path: str | Path,
) -> dict[str, Any] | None:
    if pathway_intervention_workflow_path is not None:
        if pathway_intervention_workflow_key is not None:
            raise ValueError(
                "pathway_intervention_workflow_path is mutually exclusive with "
                "pathway_intervention_workflow_key."
            )
        return {"source": "file", "path": Path(pathway_intervention_workflow_path)}
    if pathway_intervention_workflow_key is None:
        if pathway_intervention_workflow_registry_path is not None:
            raise ValueError(
                "pathway_intervention_workflow_registry_path requires "
                "pathway_intervention_workflow_key."
            )
        return None
    registry_path = Path(
        default_registry_path
        if pathway_intervention_workflow_registry_path is None
        else pathway_intervention_workflow_registry_path
    )
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get(str(pathway_intervention_workflow_key))
    if record.artifact_type != "report":
        raise ValueError(f"registry record {record.key()!r} is not a report.")
    return {
        "source": "registry",
        "registry": str(registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "path": _resolve_registry_record_path(registry_path, record),
    }


def _format_gate_reasons(gate: Mapping[str, Any]) -> str:
    reasons = tuple(gate.get("blocking_reasons", ()) or ())
    return f": {reasons!r}" if reasons else ""


def _selector_replay_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    recommended: Mapping[str, Any],
    allow_unverified: bool,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"selector replay report could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("selector replay artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("selector replay manifest verification failed")
    if report.get("status") != "promote":
        failures.append(f"selector replay status is {report.get('status')!r}, expected 'promote'")
    if _nested(report, "decision", "recommended_candidate") is None:
        failures.append("selector replay recommended candidate is missing")
    if not recommended:
        failures.append("selector replay recommended candidate is missing from leaderboard")
    elif recommended.get("status") == "blocked" or bool(recommended.get("blocked")):
        failures.append(
            f"selector replay recommended candidate status is {recommended.get('status')!r}, expected promoted"
        )
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _selector_replay_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _nested(report, "paths", "artifact_manifest")
    if raw_path is None:
        return None
    return _resolve_path(raw_path, base_path=report_path)


def _selector_replay_recommended_row(report: Mapping[str, Any]) -> dict[str, Any]:
    recommended_candidate = _nested(report, "decision", "recommended_candidate")
    if recommended_candidate is None:
        return {}
    for row in report.get("leaderboard", ()):
        row_map = _mapping(row)
        if row_map.get("candidate") == recommended_candidate:
            return row_map
    return {}


def _selector_replay_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "candidate": row.get("candidate"),
        "status": row.get("status"),
        "policy_path": row.get("policy_path"),
        "estimated_cost_units_mean": _float_or_none(row.get("estimated_cost_units_mean")),
        "observed_runtime_coverage_rate": _float_or_none(row.get("observed_runtime_coverage_rate")),
        "observed_selected_total_seconds_mean": _float_or_none(
            row.get("observed_selected_total_seconds_mean")
        ),
        "observed_selected_total_seconds_p95": _float_or_none(
            row.get("observed_selected_total_seconds_p95")
        ),
        "observed_runtime_delta_coverage_rate": _float_or_none(
            row.get("observed_runtime_delta_coverage_rate")
        ),
        "observed_selected_minus_original_seconds_mean": _float_or_none(
            row.get("observed_selected_minus_original_seconds_mean")
        ),
        "observed_selected_to_original_ratio_mean": _float_or_none(
            row.get("observed_selected_to_original_ratio_mean")
        ),
        "changed_rate": _float_or_none(row.get("changed_rate")),
    }


def _product_runtime_drift_gate(
    *,
    product_runtime_drift_report_path: str | Path | None,
    require_promotion_evidence: bool,
    require_pre_generation_evidence: bool,
    require_claim_factuality_evidence: bool,
    require_claim_risk_localization_evidence: bool,
    require_counterfactual_evidence: bool,
    require_fact_selfcheck_gate_evidence: bool,
    require_triple_audit_evidence: bool,
    require_covered_fact_property_evidence: bool,
    require_action_gate_evidence: bool,
    require_world_model_action_gate_evidence: bool,
    require_world_model_rollout_evidence: bool,
    require_action_receipts_evidence: bool,
    require_receipt_claim_support_evidence: bool,
    require_trajectory_audit_evidence: bool,
    require_provenance_evidence: bool,
    require_citation_integrity_evidence: bool,
    require_evidence_quality_evidence: bool,
    require_metacognition_evidence: bool,
    require_evidence_handoff_evidence: bool,
    require_world_model_evidence: bool,
    require_context_sensitivity_evidence: bool,
    require_evidence_alignment_evidence: bool,
    require_counterfactual_robustness_evidence: bool,
    require_frontier_release_evidence: bool,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if product_runtime_drift_report_path is None:
        if (
            require_promotion_evidence
            or require_pre_generation_evidence
            or require_claim_factuality_evidence
            or require_claim_risk_localization_evidence
            or require_counterfactual_evidence
            or require_fact_selfcheck_gate_evidence
            or require_triple_audit_evidence
            or require_covered_fact_property_evidence
            or require_action_gate_evidence
            or require_world_model_action_gate_evidence
            or require_world_model_rollout_evidence
            or require_action_receipts_evidence
            or require_receipt_claim_support_evidence
            or require_trajectory_audit_evidence
            or require_provenance_evidence
            or require_citation_integrity_evidence
            or require_evidence_quality_evidence
            or require_metacognition_evidence
            or require_evidence_handoff_evidence
            or require_world_model_evidence
            or require_context_sensitivity_evidence
            or require_evidence_alignment_evidence
            or require_counterfactual_robustness_evidence
            or require_frontier_release_evidence
        ):
            gate = {
                "passed": False,
                "blocking_reasons": [
                    "product runtime drift report is required when drift evidence is required"
                ],
            }
            return {
                "schema_version": 1,
                "status": "blocked",
                "report_path": None,
                "manifest_path": None,
                "workflow": None,
                "report_status": None,
                "decision_status": None,
                "baseline": {},
                "current": {},
                "summary": {
                    "gate_enabled": None,
                    "compared_metric_count": None,
                    "blocked_metric_count": None,
                    "observed_metric_count": None,
                    "promotion_evidence_required": bool(require_promotion_evidence),
                    "promotion_evidence_metric_count": 0,
                    "promotion_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in _PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_FIELDS
                    ) if require_promotion_evidence else (),
                    "promotion_evidence_blocked_metric_count": 0,
                    "pre_generation_evidence_required": bool(require_pre_generation_evidence),
                    "pre_generation_evidence_metric_count": 0,
                    "pre_generation_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_FIELDS
                        )
                    ) if require_pre_generation_evidence else (),
                    "pre_generation_evidence_blocked_metric_count": 0,
                    "claim_factuality_evidence_required": bool(
                        require_claim_factuality_evidence
                    ),
                    "claim_factuality_evidence_metric_count": 0,
                    "claim_factuality_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_CLAIM_FACTUALITY_EVIDENCE_FIELDS
                        )
                    ) if require_claim_factuality_evidence else (),
                    "claim_factuality_evidence_blocked_metric_count": 0,
                    "claim_risk_localization_evidence_required": bool(
                        require_claim_risk_localization_evidence
                    ),
                    "claim_risk_localization_evidence_metric_count": 0,
                    "claim_risk_localization_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_CLAIM_RISK_LOCALIZATION_EVIDENCE_FIELDS
                        )
                    ) if require_claim_risk_localization_evidence else (),
                    "claim_risk_localization_evidence_blocked_metric_count": 0,
                    "counterfactual_evidence_required": bool(require_counterfactual_evidence),
                    "counterfactual_evidence_metric_count": 0,
                    "counterfactual_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_EVIDENCE_FIELDS
                        )
                    ) if require_counterfactual_evidence else (),
                    "counterfactual_evidence_blocked_metric_count": 0,
                    "fact_selfcheck_gate_evidence_required": bool(
                        require_fact_selfcheck_gate_evidence
                    ),
                    "fact_selfcheck_gate_evidence_metric_count": 0,
                    "fact_selfcheck_gate_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_FACT_SELFCHECK_GATE_EVIDENCE_FIELDS
                        )
                    ) if require_fact_selfcheck_gate_evidence else (),
                    "fact_selfcheck_gate_evidence_blocked_metric_count": 0,
                    "triple_audit_evidence_required": bool(require_triple_audit_evidence),
                    "triple_audit_evidence_metric_count": 0,
                    "triple_audit_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_FIELDS
                        )
                    ) if require_triple_audit_evidence else (),
                    "triple_audit_evidence_blocked_metric_count": 0,
                    "covered_fact_property_evidence_required": bool(
                        require_covered_fact_property_evidence
                    ),
                    "covered_fact_property_evidence_metric_count": 0,
                    "covered_fact_property_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_FIELDS
                        )
                    ) if require_covered_fact_property_evidence else (),
                    "covered_fact_property_evidence_blocked_metric_count": 0,
                    "action_gate_evidence_required": bool(require_action_gate_evidence),
                    "action_gate_evidence_metric_count": 0,
                    "action_gate_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_ACTION_GATE_EVIDENCE_FIELDS
                        )
                    ) if require_action_gate_evidence else (),
                    "action_gate_evidence_blocked_metric_count": 0,
                    "world_model_action_gate_evidence_required": bool(
                        require_world_model_action_gate_evidence
                    ),
                    "world_model_action_gate_evidence_metric_count": 0,
                    "world_model_action_gate_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_ACTION_GATE_EVIDENCE_FIELDS
                        )
                    ) if require_world_model_action_gate_evidence else (),
                    "world_model_action_gate_evidence_blocked_metric_count": 0,
                    "world_model_rollout_evidence_required": bool(
                        require_world_model_rollout_evidence
                    ),
                    "world_model_rollout_evidence_metric_count": 0,
                    "world_model_rollout_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_ROLLOUT_EVIDENCE_FIELDS
                        )
                    ) if require_world_model_rollout_evidence else (),
                    "world_model_rollout_evidence_blocked_metric_count": 0,
                    "action_receipts_evidence_required": bool(
                        require_action_receipts_evidence
                    ),
                    "action_receipts_evidence_metric_count": 0,
                    "action_receipts_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_ACTION_RECEIPTS_EVIDENCE_FIELDS
                        )
                    ) if require_action_receipts_evidence else (),
                    "action_receipts_evidence_blocked_metric_count": 0,
                    "receipt_claim_support_evidence_required": bool(
                        require_receipt_claim_support_evidence
                    ),
                    "receipt_claim_support_evidence_metric_count": 0,
                    "receipt_claim_support_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_RECEIPT_CLAIM_SUPPORT_EVIDENCE_FIELDS
                        )
                    ) if require_receipt_claim_support_evidence else (),
                    "receipt_claim_support_evidence_blocked_metric_count": 0,
                    "trajectory_audit_evidence_required": bool(
                        require_trajectory_audit_evidence
                    ),
                    "trajectory_audit_evidence_metric_count": 0,
                    "trajectory_audit_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_TRAJECTORY_AUDIT_EVIDENCE_FIELDS
                        )
                    ) if require_trajectory_audit_evidence else (),
                    "trajectory_audit_evidence_blocked_metric_count": 0,
                    "provenance_evidence_required": bool(
                        require_provenance_evidence
                    ),
                    "provenance_evidence_metric_count": 0,
                    "provenance_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_PROVENANCE_EVIDENCE_FIELDS
                        )
                    ) if require_provenance_evidence else (),
                    "provenance_evidence_blocked_metric_count": 0,
                    "citation_integrity_evidence_required": bool(
                        require_citation_integrity_evidence
                    ),
                    "citation_integrity_evidence_metric_count": 0,
                    "citation_integrity_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_CITATION_INTEGRITY_EVIDENCE_FIELDS
                        )
                    ) if require_citation_integrity_evidence else (),
                    "citation_integrity_evidence_blocked_metric_count": 0,
                    "evidence_quality_evidence_required": bool(
                        require_evidence_quality_evidence
                    ),
                    "evidence_quality_evidence_metric_count": 0,
                    "evidence_quality_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_EVIDENCE_QUALITY_EVIDENCE_FIELDS
                        )
                    ) if require_evidence_quality_evidence else (),
                    "evidence_quality_evidence_blocked_metric_count": 0,
                    "metacognition_evidence_required": bool(
                        require_metacognition_evidence
                    ),
                    "metacognition_evidence_metric_count": 0,
                    "metacognition_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_METACOGNITION_EVIDENCE_FIELDS
                        )
                    ) if require_metacognition_evidence else (),
                    "metacognition_evidence_blocked_metric_count": 0,
                    "evidence_handoff_evidence_required": bool(
                        require_evidence_handoff_evidence
                    ),
                    "evidence_handoff_evidence_metric_count": 0,
                    "evidence_handoff_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_EVIDENCE_HANDOFF_EVIDENCE_FIELDS
                        )
                    ) if require_evidence_handoff_evidence else (),
                    "evidence_handoff_evidence_blocked_metric_count": 0,
                    "world_model_evidence_required": bool(require_world_model_evidence),
                    "world_model_evidence_metric_count": 0,
                    "world_model_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_EVIDENCE_FIELDS
                        )
                    ) if require_world_model_evidence else (),
                    "world_model_evidence_blocked_metric_count": 0,
                    "context_sensitivity_evidence_required": bool(
                        require_context_sensitivity_evidence
                    ),
                    "context_sensitivity_evidence_metric_count": 0,
                    "context_sensitivity_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_CONTEXT_SENSITIVITY_EVIDENCE_FIELDS
                        )
                    ) if require_context_sensitivity_evidence else (),
                    "context_sensitivity_evidence_blocked_metric_count": 0,
                    "evidence_alignment_evidence_required": bool(
                        require_evidence_alignment_evidence
                    ),
                    "evidence_alignment_evidence_metric_count": 0,
                    "evidence_alignment_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_EVIDENCE_ALIGNMENT_EVIDENCE_FIELDS
                        )
                    ) if require_evidence_alignment_evidence else (),
                    "evidence_alignment_evidence_blocked_metric_count": 0,
                    "counterfactual_robustness_evidence_required": bool(
                        require_counterfactual_robustness_evidence
                    ),
                    "counterfactual_robustness_evidence_metric_count": 0,
                    "counterfactual_robustness_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_ROBUSTNESS_EVIDENCE_FIELDS
                        )
                    ) if require_counterfactual_robustness_evidence else (),
                    "counterfactual_robustness_evidence_blocked_metric_count": 0,
                    "frontier_release_evidence_required": bool(
                        require_frontier_release_evidence
                    ),
                    "frontier_release_evidence_metric_count": 0,
                    "frontier_release_evidence_missing_metrics": tuple(
                        metric_name
                        for metric_name, _prefix in (
                            _PRODUCT_RUNTIME_DRIFT_FRONTIER_RELEASE_EVIDENCE_FIELDS
                        )
                    ) if require_frontier_release_evidence else (),
                    "frontier_release_evidence_blocked_metric_count": 0,
                },
                "metrics": (),
                "verification": {"passed": False, "reason": "missing product runtime drift report"},
                "gate": gate,
            }
        return None
    report_path = Path(product_runtime_drift_report_path)
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _product_runtime_drift_manifest_path(report, report_path=report_path)
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="product_runtime_drift_manifest",
        verification_context=verification_context,
    )
    metrics = _product_runtime_drift_metric_summary(report)
    promotion_evidence_summary = _product_runtime_drift_promotion_evidence_summary(
        metrics,
        required=require_promotion_evidence,
    )
    pre_generation_evidence_summary = _product_runtime_drift_pre_generation_evidence_summary(
        metrics,
        required=require_pre_generation_evidence,
    )
    claim_factuality_evidence_summary = (
        _product_runtime_drift_claim_factuality_evidence_summary(
            metrics,
            required=require_claim_factuality_evidence,
        )
    )
    claim_risk_localization_evidence_summary = (
        _product_runtime_drift_claim_risk_localization_evidence_summary(
            metrics,
            required=require_claim_risk_localization_evidence,
        )
    )
    counterfactual_evidence_summary = _product_runtime_drift_counterfactual_evidence_summary(
        metrics,
        required=require_counterfactual_evidence,
    )
    fact_selfcheck_gate_evidence_summary = (
        _product_runtime_drift_fact_selfcheck_gate_evidence_summary(
            metrics,
            required=require_fact_selfcheck_gate_evidence,
        )
    )
    triple_audit_evidence_summary = _product_runtime_drift_triple_audit_evidence_summary(
        metrics,
        required=require_triple_audit_evidence,
    )
    covered_fact_property_evidence_summary = (
        _product_runtime_drift_covered_fact_property_evidence_summary(
            metrics,
            required=require_covered_fact_property_evidence,
        )
    )
    action_gate_evidence_summary = _product_runtime_drift_action_gate_evidence_summary(
        metrics,
        required=require_action_gate_evidence,
    )
    world_model_action_gate_evidence_summary = (
        _product_runtime_drift_world_model_action_gate_evidence_summary(
            metrics,
            required=require_world_model_action_gate_evidence,
        )
    )
    world_model_rollout_evidence_summary = (
        _product_runtime_drift_world_model_rollout_evidence_summary(
            metrics,
            required=require_world_model_rollout_evidence,
        )
    )
    action_receipts_evidence_summary = (
        _product_runtime_drift_action_receipts_evidence_summary(
            metrics,
            required=require_action_receipts_evidence,
        )
    )
    receipt_claim_support_evidence_summary = (
        _product_runtime_drift_receipt_claim_support_evidence_summary(
            metrics,
            required=require_receipt_claim_support_evidence,
        )
    )
    trajectory_audit_evidence_summary = (
        _product_runtime_drift_trajectory_audit_evidence_summary(
            metrics,
            required=require_trajectory_audit_evidence,
        )
    )
    provenance_evidence_summary = _product_runtime_drift_provenance_evidence_summary(
        metrics,
        required=require_provenance_evidence,
    )
    citation_integrity_evidence_summary = (
        _product_runtime_drift_citation_integrity_evidence_summary(
            metrics,
            required=require_citation_integrity_evidence,
        )
    )
    evidence_quality_evidence_summary = (
        _product_runtime_drift_evidence_quality_evidence_summary(
            metrics,
            required=require_evidence_quality_evidence,
        )
    )
    metacognition_evidence_summary = (
        _product_runtime_drift_metacognition_evidence_summary(
            metrics,
            required=require_metacognition_evidence,
        )
    )
    evidence_handoff_evidence_summary = (
        _product_runtime_drift_evidence_handoff_evidence_summary(
            metrics,
            required=require_evidence_handoff_evidence,
        )
    )
    frontier_release_evidence_summary = (
        _product_runtime_drift_frontier_release_evidence_summary(
            metrics,
            required=require_frontier_release_evidence,
        )
    )
    world_model_evidence_summary = _product_runtime_drift_world_model_evidence_summary(
        metrics,
        required=require_world_model_evidence,
    )
    context_sensitivity_evidence_summary = (
        _product_runtime_drift_context_sensitivity_evidence_summary(
            metrics,
            required=require_context_sensitivity_evidence,
        )
    )
    evidence_alignment_evidence_summary = (
        _product_runtime_drift_evidence_alignment_evidence_summary(
            metrics,
            required=require_evidence_alignment_evidence,
        )
    )
    counterfactual_robustness_evidence_summary = (
        _product_runtime_drift_counterfactual_robustness_evidence_summary(
            metrics,
            required=require_counterfactual_robustness_evidence,
        )
    )
    gate = _product_runtime_drift_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        promotion_evidence_summary=promotion_evidence_summary,
        require_promotion_evidence=require_promotion_evidence,
        pre_generation_evidence_summary=pre_generation_evidence_summary,
        require_pre_generation_evidence=require_pre_generation_evidence,
        claim_factuality_evidence_summary=claim_factuality_evidence_summary,
        require_claim_factuality_evidence=require_claim_factuality_evidence,
        claim_risk_localization_evidence_summary=(
            claim_risk_localization_evidence_summary
        ),
        require_claim_risk_localization_evidence=(
            require_claim_risk_localization_evidence
        ),
        counterfactual_evidence_summary=counterfactual_evidence_summary,
        require_counterfactual_evidence=require_counterfactual_evidence,
        fact_selfcheck_gate_evidence_summary=fact_selfcheck_gate_evidence_summary,
        require_fact_selfcheck_gate_evidence=require_fact_selfcheck_gate_evidence,
        triple_audit_evidence_summary=triple_audit_evidence_summary,
        require_triple_audit_evidence=require_triple_audit_evidence,
        covered_fact_property_evidence_summary=covered_fact_property_evidence_summary,
        require_covered_fact_property_evidence=require_covered_fact_property_evidence,
        action_gate_evidence_summary=action_gate_evidence_summary,
        require_action_gate_evidence=require_action_gate_evidence,
        world_model_action_gate_evidence_summary=(
            world_model_action_gate_evidence_summary
        ),
        require_world_model_action_gate_evidence=require_world_model_action_gate_evidence,
        world_model_rollout_evidence_summary=world_model_rollout_evidence_summary,
        require_world_model_rollout_evidence=require_world_model_rollout_evidence,
        action_receipts_evidence_summary=action_receipts_evidence_summary,
        require_action_receipts_evidence=require_action_receipts_evidence,
        receipt_claim_support_evidence_summary=receipt_claim_support_evidence_summary,
        require_receipt_claim_support_evidence=require_receipt_claim_support_evidence,
        trajectory_audit_evidence_summary=trajectory_audit_evidence_summary,
        require_trajectory_audit_evidence=require_trajectory_audit_evidence,
        provenance_evidence_summary=provenance_evidence_summary,
        require_provenance_evidence=require_provenance_evidence,
        citation_integrity_evidence_summary=citation_integrity_evidence_summary,
        require_citation_integrity_evidence=require_citation_integrity_evidence,
        evidence_quality_evidence_summary=evidence_quality_evidence_summary,
        require_evidence_quality_evidence=require_evidence_quality_evidence,
        metacognition_evidence_summary=metacognition_evidence_summary,
        require_metacognition_evidence=require_metacognition_evidence,
        evidence_handoff_evidence_summary=evidence_handoff_evidence_summary,
        require_evidence_handoff_evidence=require_evidence_handoff_evidence,
        world_model_evidence_summary=world_model_evidence_summary,
        require_world_model_evidence=require_world_model_evidence,
        context_sensitivity_evidence_summary=context_sensitivity_evidence_summary,
        require_context_sensitivity_evidence=require_context_sensitivity_evidence,
        evidence_alignment_evidence_summary=evidence_alignment_evidence_summary,
        require_evidence_alignment_evidence=require_evidence_alignment_evidence,
        counterfactual_robustness_evidence_summary=(
            counterfactual_robustness_evidence_summary
        ),
        require_counterfactual_robustness_evidence=(
            require_counterfactual_robustness_evidence
        ),
        frontier_release_evidence_summary=frontier_release_evidence_summary,
        require_frontier_release_evidence=require_frontier_release_evidence,
        allow_unverified=allow_unverified,
    )
    summary = _mapping(report.get("summary"))
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "workflow": report.get("workflow"),
        "report_status": report.get("status"),
        "decision_status": _nested(report, "decision", "status"),
        "baseline": _mapping(report.get("baseline")),
        "current": _mapping(report.get("current")),
        "summary": {
            "gate_enabled": summary.get("gate_enabled"),
            "compared_metric_count": summary.get("compared_metric_count"),
            "blocked_metric_count": summary.get("blocked_metric_count"),
            "observed_metric_count": summary.get("observed_metric_count"),
            **promotion_evidence_summary,
            **pre_generation_evidence_summary,
            **claim_factuality_evidence_summary,
            **claim_risk_localization_evidence_summary,
            **counterfactual_evidence_summary,
            **fact_selfcheck_gate_evidence_summary,
            **triple_audit_evidence_summary,
            **covered_fact_property_evidence_summary,
            **action_gate_evidence_summary,
            **world_model_action_gate_evidence_summary,
            **world_model_rollout_evidence_summary,
            **action_receipts_evidence_summary,
            **receipt_claim_support_evidence_summary,
            **trajectory_audit_evidence_summary,
            **provenance_evidence_summary,
            **citation_integrity_evidence_summary,
            **evidence_quality_evidence_summary,
            **metacognition_evidence_summary,
            **evidence_handoff_evidence_summary,
            **world_model_evidence_summary,
            **context_sensitivity_evidence_summary,
            **evidence_alignment_evidence_summary,
            **counterfactual_robustness_evidence_summary,
            **frontier_release_evidence_summary,
        },
        "metrics": metrics,
        "verification": verification,
        "gate": gate,
    }


def _product_runtime_drift_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    promotion_evidence_summary: Mapping[str, Any],
    require_promotion_evidence: bool,
    pre_generation_evidence_summary: Mapping[str, Any],
    require_pre_generation_evidence: bool,
    claim_factuality_evidence_summary: Mapping[str, Any],
    require_claim_factuality_evidence: bool,
    claim_risk_localization_evidence_summary: Mapping[str, Any],
    require_claim_risk_localization_evidence: bool,
    counterfactual_evidence_summary: Mapping[str, Any],
    require_counterfactual_evidence: bool,
    fact_selfcheck_gate_evidence_summary: Mapping[str, Any],
    require_fact_selfcheck_gate_evidence: bool,
    triple_audit_evidence_summary: Mapping[str, Any],
    require_triple_audit_evidence: bool,
    covered_fact_property_evidence_summary: Mapping[str, Any],
    require_covered_fact_property_evidence: bool,
    action_gate_evidence_summary: Mapping[str, Any],
    require_action_gate_evidence: bool,
    world_model_action_gate_evidence_summary: Mapping[str, Any],
    require_world_model_action_gate_evidence: bool,
    world_model_rollout_evidence_summary: Mapping[str, Any],
    require_world_model_rollout_evidence: bool,
    action_receipts_evidence_summary: Mapping[str, Any],
    require_action_receipts_evidence: bool,
    receipt_claim_support_evidence_summary: Mapping[str, Any],
    require_receipt_claim_support_evidence: bool,
    trajectory_audit_evidence_summary: Mapping[str, Any],
    require_trajectory_audit_evidence: bool,
    provenance_evidence_summary: Mapping[str, Any],
    require_provenance_evidence: bool,
    citation_integrity_evidence_summary: Mapping[str, Any],
    require_citation_integrity_evidence: bool,
    evidence_quality_evidence_summary: Mapping[str, Any],
    require_evidence_quality_evidence: bool,
    metacognition_evidence_summary: Mapping[str, Any],
    require_metacognition_evidence: bool,
    evidence_handoff_evidence_summary: Mapping[str, Any],
    require_evidence_handoff_evidence: bool,
    world_model_evidence_summary: Mapping[str, Any],
    require_world_model_evidence: bool,
    context_sensitivity_evidence_summary: Mapping[str, Any],
    require_context_sensitivity_evidence: bool,
    evidence_alignment_evidence_summary: Mapping[str, Any],
    require_evidence_alignment_evidence: bool,
    counterfactual_robustness_evidence_summary: Mapping[str, Any],
    require_counterfactual_robustness_evidence: bool,
    frontier_release_evidence_summary: Mapping[str, Any],
    require_frontier_release_evidence: bool,
    allow_unverified: bool,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"product runtime drift report could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("product runtime drift artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("product runtime drift manifest verification failed")
    if report.get("workflow") != "product_runtime_drift_comparison":
        failures.append(
            f"product runtime drift workflow is {report.get('workflow')!r}, "
            "expected 'product_runtime_drift_comparison'"
        )
    if report.get("status") != "promote":
        failures.append(f"product runtime drift status is {report.get('status')!r}, expected 'promote'")
    decision_status = _nested(report, "decision", "status")
    if decision_status != "promote":
        failures.append(
            f"product runtime drift decision status is {decision_status!r}, expected 'promote'"
        )
    summary = _mapping(report.get("summary"))
    if summary.get("gate_enabled") is not True:
        failures.append("product runtime drift gate was not enabled")
    blocked_count = _float_or_none(summary.get("blocked_metric_count"))
    if blocked_count is None:
        failures.append("product runtime drift blocked metric count is missing")
    elif blocked_count > 0:
        failures.append(f"product runtime drift blocked {int(blocked_count)} metric(s)")
    if require_promotion_evidence:
        missing_metrics = tuple(promotion_evidence_summary.get("promotion_evidence_missing_metrics") or ())
        if missing_metrics:
            failures.append(
                "product runtime drift promotion evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
    if require_pre_generation_evidence:
        missing_metrics = tuple(
            pre_generation_evidence_summary.get("pre_generation_evidence_missing_metrics") or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift pre-generation evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            pre_generation_evidence_summary.get("pre_generation_evidence_blocked_metric_count")
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift pre-generation evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_claim_factuality_evidence:
        missing_metrics = tuple(
            claim_factuality_evidence_summary.get(
                "claim_factuality_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift claim factuality evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            claim_factuality_evidence_summary.get(
                "claim_factuality_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift claim factuality evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_claim_risk_localization_evidence:
        missing_metrics = tuple(
            claim_risk_localization_evidence_summary.get(
                "claim_risk_localization_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift claim-risk localization evidence metrics are "
                "incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            claim_risk_localization_evidence_summary.get(
                "claim_risk_localization_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift claim-risk localization evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_counterfactual_evidence:
        missing_metrics = tuple(
            counterfactual_evidence_summary.get("counterfactual_evidence_missing_metrics") or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift counterfactual evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            counterfactual_evidence_summary.get("counterfactual_evidence_blocked_metric_count")
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift counterfactual evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_fact_selfcheck_gate_evidence:
        missing_metrics = tuple(
            fact_selfcheck_gate_evidence_summary.get(
                "fact_selfcheck_gate_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift fact-selfcheck gate evidence metrics are "
                "incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            fact_selfcheck_gate_evidence_summary.get(
                "fact_selfcheck_gate_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift fact-selfcheck gate evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_triple_audit_evidence:
        missing_metrics = tuple(
            triple_audit_evidence_summary.get("triple_audit_evidence_missing_metrics") or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift triple audit evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            triple_audit_evidence_summary.get("triple_audit_evidence_blocked_metric_count")
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift triple audit evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_covered_fact_property_evidence:
        missing_metrics = tuple(
            covered_fact_property_evidence_summary.get(
                "covered_fact_property_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift covered-fact property evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            covered_fact_property_evidence_summary.get(
                "covered_fact_property_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift covered-fact property evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_action_gate_evidence:
        missing_metrics = tuple(
            action_gate_evidence_summary.get("action_gate_evidence_missing_metrics") or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift action-gate evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            action_gate_evidence_summary.get("action_gate_evidence_blocked_metric_count")
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift action-gate evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_world_model_action_gate_evidence:
        missing_metrics = tuple(
            world_model_action_gate_evidence_summary.get(
                "world_model_action_gate_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift world-model action-gate evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            world_model_action_gate_evidence_summary.get(
                "world_model_action_gate_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift world-model action-gate evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_world_model_rollout_evidence:
        missing_metrics = tuple(
            world_model_rollout_evidence_summary.get(
                "world_model_rollout_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift world-model rollout evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            world_model_rollout_evidence_summary.get(
                "world_model_rollout_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift world-model rollout evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_action_receipts_evidence:
        missing_metrics = tuple(
            action_receipts_evidence_summary.get(
                "action_receipts_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift action-receipts evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            action_receipts_evidence_summary.get(
                "action_receipts_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift action-receipts evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_receipt_claim_support_evidence:
        missing_metrics = tuple(
            receipt_claim_support_evidence_summary.get(
                "receipt_claim_support_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift receipt claim-support evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            receipt_claim_support_evidence_summary.get(
                "receipt_claim_support_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift receipt claim-support evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_trajectory_audit_evidence:
        missing_metrics = tuple(
            trajectory_audit_evidence_summary.get(
                "trajectory_audit_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift trajectory-audit evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            trajectory_audit_evidence_summary.get(
                "trajectory_audit_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift trajectory-audit evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_provenance_evidence:
        missing_metrics = tuple(
            provenance_evidence_summary.get("provenance_evidence_missing_metrics") or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift provenance evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            provenance_evidence_summary.get(
                "provenance_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift provenance evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_citation_integrity_evidence:
        missing_metrics = tuple(
            citation_integrity_evidence_summary.get(
                "citation_integrity_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift citation-integrity evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            citation_integrity_evidence_summary.get(
                "citation_integrity_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift citation-integrity evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_evidence_quality_evidence:
        missing_metrics = tuple(
            evidence_quality_evidence_summary.get(
                "evidence_quality_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift evidence-quality evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            evidence_quality_evidence_summary.get(
                "evidence_quality_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift evidence-quality evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_metacognition_evidence:
        missing_metrics = tuple(
            metacognition_evidence_summary.get(
                "metacognition_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift metacognition evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            metacognition_evidence_summary.get(
                "metacognition_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift metacognition evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_evidence_handoff_evidence:
        missing_metrics = tuple(
            evidence_handoff_evidence_summary.get(
                "evidence_handoff_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift evidence-handoff evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            evidence_handoff_evidence_summary.get(
                "evidence_handoff_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift evidence-handoff evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_world_model_evidence:
        missing_metrics = tuple(
            world_model_evidence_summary.get("world_model_evidence_missing_metrics") or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift world-model evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            world_model_evidence_summary.get("world_model_evidence_blocked_metric_count")
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift world-model evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_context_sensitivity_evidence:
        missing_metrics = tuple(
            context_sensitivity_evidence_summary.get(
                "context_sensitivity_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift context-sensitivity evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            context_sensitivity_evidence_summary.get(
                "context_sensitivity_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift context-sensitivity evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_evidence_alignment_evidence:
        missing_metrics = tuple(
            evidence_alignment_evidence_summary.get(
                "evidence_alignment_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift evidence-alignment evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            evidence_alignment_evidence_summary.get(
                "evidence_alignment_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift evidence-alignment evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_counterfactual_robustness_evidence:
        missing_metrics = tuple(
            counterfactual_robustness_evidence_summary.get(
                "counterfactual_robustness_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift counterfactual-robustness evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            counterfactual_robustness_evidence_summary.get(
                "counterfactual_robustness_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift counterfactual-robustness evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    if require_frontier_release_evidence:
        missing_metrics = tuple(
            frontier_release_evidence_summary.get(
                "frontier_release_evidence_missing_metrics"
            ) or ()
        )
        if missing_metrics:
            failures.append(
                "product runtime drift frontier release evidence metrics are incomplete: "
                + ", ".join(str(metric) for metric in missing_metrics)
            )
        blocked_metric_count = _float_or_none(
            frontier_release_evidence_summary.get(
                "frontier_release_evidence_blocked_metric_count"
            )
        )
        if blocked_metric_count is not None and blocked_metric_count > 0:
            failures.append(
                "product runtime drift frontier release evidence blocked "
                f"{int(blocked_metric_count)} metric(s)"
            )
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _product_runtime_drift_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _nested(report, "paths", "artifact_manifest")
    if raw_path is None:
        return None
    return _resolve_path(raw_path, base_path=report_path)


def _product_runtime_drift_metric_summary(report: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = []
    for row in report.get("metrics", ()):
        metric = _mapping(row)
        if not metric:
            continue
        rows.append({
            "metric": metric.get("metric"),
            "status": metric.get("status"),
            "comparison": metric.get("comparison"),
            "baseline": _float_or_none(metric.get("baseline")),
            "current": _float_or_none(metric.get("current")),
            "ratio_to_baseline": _float_or_none(metric.get("ratio_to_baseline")),
            "absolute_delta": _float_or_none(metric.get("absolute_delta")),
            "absolute_drop": _float_or_none(metric.get("absolute_drop")),
            "threshold": metric.get("threshold"),
            "reason": metric.get("reason"),
        })
    return tuple(rows)


def _product_runtime_drift_promotion_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    summary: dict[str, Any] = {
        "promotion_evidence_required": bool(required),
        "promotion_evidence_metric_count": 0,
        "promotion_evidence_missing_metrics": (),
        "promotion_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_FIELDS:
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary["promotion_evidence_blocked_metric_count"] += 1
    summary["promotion_evidence_metric_count"] = metric_count
    summary["promotion_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _product_runtime_drift_pre_generation_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    summary: dict[str, Any] = {
        "pre_generation_evidence_required": bool(required),
        "pre_generation_evidence_metric_count": 0,
        "pre_generation_evidence_missing_metrics": (),
        "pre_generation_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_FIELDS:
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary["pre_generation_evidence_blocked_metric_count"] += 1
    summary["pre_generation_evidence_metric_count"] = metric_count
    summary["pre_generation_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _product_runtime_drift_claim_factuality_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    summary: dict[str, Any] = {
        "claim_factuality_evidence_required": bool(required),
        "claim_factuality_evidence_metric_count": 0,
        "claim_factuality_evidence_missing_metrics": (),
        "claim_factuality_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_RUNTIME_DRIFT_CLAIM_FACTUALITY_EVIDENCE_FIELDS:
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary["claim_factuality_evidence_blocked_metric_count"] += 1
    summary["claim_factuality_evidence_metric_count"] = metric_count
    summary["claim_factuality_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _product_runtime_drift_claim_risk_localization_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    summary: dict[str, Any] = {
        "claim_risk_localization_evidence_required": bool(required),
        "claim_risk_localization_evidence_metric_count": 0,
        "claim_risk_localization_evidence_missing_metrics": (),
        "claim_risk_localization_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in (
        _PRODUCT_RUNTIME_DRIFT_CLAIM_RISK_LOCALIZATION_EVIDENCE_FIELDS
    ):
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary["claim_risk_localization_evidence_blocked_metric_count"] += 1
    summary["claim_risk_localization_evidence_metric_count"] = metric_count
    summary["claim_risk_localization_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _product_runtime_drift_counterfactual_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    summary: dict[str, Any] = {
        "counterfactual_evidence_required": bool(required),
        "counterfactual_evidence_metric_count": 0,
        "counterfactual_evidence_missing_metrics": (),
        "counterfactual_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_EVIDENCE_FIELDS:
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary["counterfactual_evidence_blocked_metric_count"] += 1
    summary["counterfactual_evidence_metric_count"] = metric_count
    summary["counterfactual_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _product_runtime_drift_fact_selfcheck_gate_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    return _product_runtime_drift_named_evidence_summary(
        metrics,
        fields=_PRODUCT_RUNTIME_DRIFT_FACT_SELFCHECK_GATE_EVIDENCE_FIELDS,
        evidence_prefix="fact_selfcheck_gate",
        required=required,
    )


def _product_runtime_drift_triple_audit_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    summary: dict[str, Any] = {
        "triple_audit_evidence_required": bool(required),
        "triple_audit_evidence_metric_count": 0,
        "triple_audit_evidence_missing_metrics": (),
        "triple_audit_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_FIELDS:
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary["triple_audit_evidence_blocked_metric_count"] += 1
    summary["triple_audit_evidence_metric_count"] = metric_count
    summary["triple_audit_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _product_runtime_drift_covered_fact_property_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    summary: dict[str, Any] = {
        "covered_fact_property_evidence_required": bool(required),
        "covered_fact_property_evidence_metric_count": 0,
        "covered_fact_property_evidence_missing_metrics": (),
        "covered_fact_property_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_FIELDS:
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary["covered_fact_property_evidence_blocked_metric_count"] += 1
    summary["covered_fact_property_evidence_metric_count"] = metric_count
    summary["covered_fact_property_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _product_runtime_drift_action_gate_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    summary: dict[str, Any] = {
        "action_gate_evidence_required": bool(required),
        "action_gate_evidence_metric_count": 0,
        "action_gate_evidence_missing_metrics": (),
        "action_gate_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_RUNTIME_DRIFT_ACTION_GATE_EVIDENCE_FIELDS:
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary["action_gate_evidence_blocked_metric_count"] += 1
    summary["action_gate_evidence_metric_count"] = metric_count
    summary["action_gate_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _product_runtime_drift_world_model_action_gate_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    return _product_runtime_drift_named_evidence_summary(
        metrics,
        fields=_PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_ACTION_GATE_EVIDENCE_FIELDS,
        evidence_prefix="world_model_action_gate",
        required=required,
    )


def _product_runtime_drift_world_model_rollout_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    return _product_runtime_drift_named_evidence_summary(
        metrics,
        fields=_PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_ROLLOUT_EVIDENCE_FIELDS,
        evidence_prefix="world_model_rollout",
        required=required,
    )


def _product_runtime_drift_named_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[tuple[str, str]],
    evidence_prefix: str,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    blocked_count_key = f"{evidence_prefix}_evidence_blocked_metric_count"
    summary: dict[str, Any] = {
        f"{evidence_prefix}_evidence_required": bool(required),
        f"{evidence_prefix}_evidence_metric_count": 0,
        f"{evidence_prefix}_evidence_missing_metrics": (),
        blocked_count_key: 0,
    }
    for metric_name, prefix in fields:
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary[blocked_count_key] += 1
    summary[f"{evidence_prefix}_evidence_metric_count"] = metric_count
    summary[f"{evidence_prefix}_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _product_runtime_drift_action_receipts_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    return _product_runtime_drift_named_evidence_summary(
        metrics,
        fields=_PRODUCT_RUNTIME_DRIFT_ACTION_RECEIPTS_EVIDENCE_FIELDS,
        evidence_prefix="action_receipts",
        required=required,
    )


def _product_runtime_drift_receipt_claim_support_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    return _product_runtime_drift_named_evidence_summary(
        metrics,
        fields=_PRODUCT_RUNTIME_DRIFT_RECEIPT_CLAIM_SUPPORT_EVIDENCE_FIELDS,
        evidence_prefix="receipt_claim_support",
        required=required,
    )


def _product_runtime_drift_provenance_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    return _product_runtime_drift_named_evidence_summary(
        metrics,
        fields=_PRODUCT_RUNTIME_DRIFT_PROVENANCE_EVIDENCE_FIELDS,
        evidence_prefix="provenance",
        required=required,
    )


def _product_runtime_drift_citation_integrity_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    return _product_runtime_drift_named_evidence_summary(
        metrics,
        fields=_PRODUCT_RUNTIME_DRIFT_CITATION_INTEGRITY_EVIDENCE_FIELDS,
        evidence_prefix="citation_integrity",
        required=required,
    )


def _product_runtime_drift_evidence_quality_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    return _product_runtime_drift_named_evidence_summary(
        metrics,
        fields=_PRODUCT_RUNTIME_DRIFT_EVIDENCE_QUALITY_EVIDENCE_FIELDS,
        evidence_prefix="evidence_quality",
        required=required,
    )


def _product_runtime_drift_metacognition_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    return _product_runtime_drift_named_evidence_summary(
        metrics,
        fields=_PRODUCT_RUNTIME_DRIFT_METACOGNITION_EVIDENCE_FIELDS,
        evidence_prefix="metacognition",
        required=required,
    )


def _product_runtime_drift_trajectory_audit_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    summary: dict[str, Any] = {
        "trajectory_audit_evidence_required": bool(required),
        "trajectory_audit_evidence_metric_count": 0,
        "trajectory_audit_evidence_missing_metrics": (),
        "trajectory_audit_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_RUNTIME_DRIFT_TRAJECTORY_AUDIT_EVIDENCE_FIELDS:
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary["trajectory_audit_evidence_blocked_metric_count"] += 1
    summary["trajectory_audit_evidence_metric_count"] = metric_count
    summary["trajectory_audit_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _product_runtime_drift_evidence_handoff_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    summary: dict[str, Any] = {
        "evidence_handoff_evidence_required": bool(required),
        "evidence_handoff_evidence_metric_count": 0,
        "evidence_handoff_evidence_missing_metrics": (),
        "evidence_handoff_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_RUNTIME_DRIFT_EVIDENCE_HANDOFF_EVIDENCE_FIELDS:
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary["evidence_handoff_evidence_blocked_metric_count"] += 1
    summary["evidence_handoff_evidence_metric_count"] = metric_count
    summary["evidence_handoff_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _product_runtime_drift_world_model_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    summary: dict[str, Any] = {
        "world_model_evidence_required": bool(required),
        "world_model_evidence_metric_count": 0,
        "world_model_evidence_missing_metrics": (),
        "world_model_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_EVIDENCE_FIELDS:
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary["world_model_evidence_blocked_metric_count"] += 1
    summary["world_model_evidence_metric_count"] = metric_count
    summary["world_model_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _product_runtime_drift_context_sensitivity_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    summary: dict[str, Any] = {
        "context_sensitivity_evidence_required": bool(required),
        "context_sensitivity_evidence_metric_count": 0,
        "context_sensitivity_evidence_missing_metrics": (),
        "context_sensitivity_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_RUNTIME_DRIFT_CONTEXT_SENSITIVITY_EVIDENCE_FIELDS:
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary["context_sensitivity_evidence_blocked_metric_count"] += 1
    summary["context_sensitivity_evidence_metric_count"] = metric_count
    summary["context_sensitivity_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _product_runtime_drift_evidence_alignment_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    summary: dict[str, Any] = {
        "evidence_alignment_evidence_required": bool(required),
        "evidence_alignment_evidence_metric_count": 0,
        "evidence_alignment_evidence_missing_metrics": (),
        "evidence_alignment_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_RUNTIME_DRIFT_EVIDENCE_ALIGNMENT_EVIDENCE_FIELDS:
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary["evidence_alignment_evidence_blocked_metric_count"] += 1
    summary["evidence_alignment_evidence_metric_count"] = metric_count
    summary["evidence_alignment_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _product_runtime_drift_counterfactual_robustness_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    summary: dict[str, Any] = {
        "counterfactual_robustness_evidence_required": bool(required),
        "counterfactual_robustness_evidence_metric_count": 0,
        "counterfactual_robustness_evidence_missing_metrics": (),
        "counterfactual_robustness_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_ROBUSTNESS_EVIDENCE_FIELDS:
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary["counterfactual_robustness_evidence_blocked_metric_count"] += 1
    summary["counterfactual_robustness_evidence_metric_count"] = metric_count
    summary["counterfactual_robustness_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _product_runtime_drift_frontier_release_evidence_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    required: bool = False,
) -> dict[str, Any]:
    metrics_by_name = {
        str(metric["metric"]): metric
        for metric in metrics
        if isinstance(metric, Mapping) and isinstance(metric.get("metric"), str)
    }
    missing_metrics: list[str] = []
    metric_count = 0
    summary: dict[str, Any] = {
        "frontier_release_evidence_required": bool(required),
        "frontier_release_evidence_metric_count": 0,
        "frontier_release_evidence_missing_metrics": (),
        "frontier_release_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_RUNTIME_DRIFT_FRONTIER_RELEASE_EVIDENCE_FIELDS:
        metric = metrics_by_name.get(metric_name)
        summary[f"{prefix}_baseline"] = None if metric is None else metric.get("baseline")
        summary[f"{prefix}_current"] = None if metric is None else metric.get("current")
        summary[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is None or metric.get("current") is None:
            missing_metrics.append(metric_name)
            continue
        metric_count += 1
        if metric.get("status") == "blocked":
            summary["frontier_release_evidence_blocked_metric_count"] += 1
    summary["frontier_release_evidence_metric_count"] = metric_count
    summary["frontier_release_evidence_missing_metrics"] = tuple(missing_metrics)
    return summary


def _release_efficiency_gate(
    *,
    release_efficiency_report_path: str | Path | None,
    recursive: bool,
    allow_unverified: bool,
    manifest_fingerprint_workers: int,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if release_efficiency_report_path is None:
        return None
    report_path = Path(release_efficiency_report_path)
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _release_efficiency_manifest_path(report, report_path=report_path)
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=manifest_fingerprint_workers,
        artifact_name="release_efficiency_manifest",
        verification_context=verification_context,
    )
    recommended = _release_efficiency_recommended_row(report)
    gate = _release_efficiency_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
        recommended=recommended,
        allow_unverified=allow_unverified,
    )
    summary = _mapping(report.get("summary"))
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "report_path": str(report_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "workflow": report.get("workflow"),
        "report_status": report.get("status"),
        "decision_status": _nested(report, "decision", "status"),
        "recommended_profile": _nested(report, "decision", "recommended_profile"),
        "recommended_efficiency_score": _float_or_none(
            _nested(report, "decision", "recommended_efficiency_score")
        ),
        "blocking_reasons": tuple(_nested(report, "decision", "blocking_reasons") or ()),
        "summary": {
            "profile_count": summary.get("profile_count"),
            "blocked_profile_count": summary.get("blocked_profile_count"),
            "quality_report_count": summary.get("quality_report_count"),
            "quality_passed": summary.get("quality_passed"),
            "generated_trace_count": summary.get("generated_trace_count"),
            "reused_trace_count": summary.get("reused_trace_count"),
            "trace_record_cache_enabled_profile_count": summary.get(
                "trace_record_cache_enabled_profile_count"
            ),
            "trace_record_cache_hit_profile_count": summary.get(
                "trace_record_cache_hit_profile_count"
            ),
            "trace_record_cache_written_profile_count": summary.get(
                "trace_record_cache_written_profile_count"
            ),
        },
        "leaderboard_top": _release_efficiency_summary(recommended),
        "verification": verification,
        "gate": gate,
    }


def _release_efficiency_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
    recommended: Mapping[str, Any],
    allow_unverified: bool,
) -> dict[str, Any]:
    failures = []
    if report_error is not None:
        failures.append(f"release efficiency report could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("release efficiency artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("release efficiency manifest verification failed")
    if report.get("workflow") != "release_efficiency_report":
        failures.append(
            f"release efficiency workflow is {report.get('workflow')!r}, "
            "expected 'release_efficiency_report'"
        )
    if report.get("status") != "promote":
        failures.append(f"release efficiency status is {report.get('status')!r}, expected 'promote'")
    decision_status = _nested(report, "decision", "status")
    if decision_status != "promote":
        failures.append(
            f"release efficiency decision status is {decision_status!r}, expected 'promote'"
        )
    recommended_profile = _nested(report, "decision", "recommended_profile")
    allowed_profiles = set(RUNTIME_PROFILE_NAMES) | {"auto"}
    if recommended_profile is None:
        failures.append("release efficiency recommended profile is missing")
    elif str(recommended_profile) not in allowed_profiles:
        choices = ", ".join(sorted(allowed_profiles))
        failures.append(
            f"release efficiency recommended profile is {recommended_profile!r}, "
            f"expected one of: {choices}"
        )
    if _float_or_none(_nested(report, "decision", "recommended_efficiency_score")) is None:
        failures.append("release efficiency recommended efficiency score is missing or non-finite")
    summary = _mapping(report.get("summary"))
    profile_count = _float_or_none(summary.get("profile_count"))
    if profile_count is None:
        failures.append("release efficiency profile count is missing")
    elif profile_count < 1:
        failures.append("release efficiency profile count is zero")
    if summary.get("quality_passed") is False:
        failures.append("release efficiency quality reports did not pass")
    if not recommended:
        failures.append("release efficiency recommended profile is missing from leaderboard")
    elif recommended.get("status") == "blocked" or bool(recommended.get("blocked")):
        failures.append(
            f"release efficiency recommended profile status is {recommended.get('status')!r}, "
            "expected promoted"
        )
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _release_efficiency_manifest_path(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    raw_path = _nested(report, "paths", "artifact_manifest")
    if raw_path is None:
        return None
    return _resolve_path(raw_path, base_path=report_path)


def _release_efficiency_recommended_row(report: Mapping[str, Any]) -> dict[str, Any]:
    recommended_profile = _nested(report, "decision", "recommended_profile")
    if recommended_profile is None:
        return {}
    for row in report.get("leaderboard", ()):
        row_map = _mapping(row)
        if row_map.get("profile") == recommended_profile:
            return row_map
    return {}


def _release_efficiency_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    efficiency = _mapping(row.get("efficiency"))
    trace_record_cache = _mapping(row.get("trace_record_cache"))
    return {
        "profile": row.get("profile"),
        "status": row.get("status"),
        "blocked": row.get("blocked"),
        "efficiency": {
            "score": _float_or_none(efficiency.get("score")),
            "quality_score": _float_or_none(efficiency.get("quality_score")),
            "runtime_score": _float_or_none(efficiency.get("runtime_score")),
            "route_fanout_score": _float_or_none(efficiency.get("route_fanout_score")),
            "cache_bonus": _float_or_none(efficiency.get("cache_bonus")),
        },
        "total_seconds_mean": _float_or_none(row.get("total_seconds_mean")),
        "mean_attempted_route_count": _float_or_none(row.get("mean_attempted_route_count")),
        "verification_skip_rate_mean": _float_or_none(row.get("verification_skip_rate_mean")),
        "trace_record_cache_hit": trace_record_cache.get("cache_hit"),
    }


def _performance_gate(
    *,
    verification: Mapping[str, Any],
    allow_unverified: bool,
    report_error: str | None,
    manifest_path: Path | None,
    runtime_recommendation: Mapping[str, Any],
    performance_evidence_bundle: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    require_score_dump_cache: bool,
    min_score_dump_cache_jsonl_view_hit_rate: float | None,
    performance_trend_gate: Mapping[str, Any] | None,
    max_covariance_maha_last_auroc_drop: float | None,
) -> dict[str, Any]:
    failures = []
    performance_score_dump_cache = _mapping(performance_evidence_bundle.get("score_dump_cache"))
    score_dump_cache_gate = _performance_score_dump_cache_gate(
        performance_score_dump_cache,
        required=require_score_dump_cache,
        min_jsonl_view_hit_rate=min_score_dump_cache_jsonl_view_hit_rate,
    )
    covariance_gate = covariance_tradeoff_gate(
        runtime_recommendation,
        max_covariance_maha_last_auroc_drop=max_covariance_maha_last_auroc_drop,
    )
    if report_error is not None:
        failures.append(f"performance baseline report could not be loaded: {report_error}")
    if manifest_path is None:
        failures.append("performance baseline artifact manifest is missing")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("performance baseline manifest verification failed")
    if runtime_recommendation.get("status") != "promote":
        failures.append(
            f"performance runtime_recommendation_status is {runtime_recommendation.get('status')!r}, "
            "expected 'promote'"
        )
    if performance_evidence_bundle and performance_evidence_bundle.get("release_ready") is not True:
        failures.append(
            "performance evidence bundle release_ready is "
            f"{performance_evidence_bundle.get('release_ready')!r}, expected True"
        )
    failures.extend(score_dump_cache_gate["blocking_reasons"])
    failures.extend(covariance_gate["blocking_reasons"])
    if performance_trend_gate is not None:
        failures.extend(performance_trend_gate.get("blocking_reasons", ()))
    if candidate is None:
        failures.append("release candidate is unavailable for performance baseline comparison")
        return {
            "passed": False,
            "blocking_reasons": failures,
            "score_dump_cache": score_dump_cache_gate,
            "covariance_tradeoff": covariance_gate,
            "performance_trend": (
                None if performance_trend_gate is None else dict(performance_trend_gate)
            ),
        }

    recommendation = _mapping(runtime_recommendation.get("recommendation"))
    runtime = _mapping(candidate.get("runtime"))
    quality = _mapping(candidate.get("quality"))
    runtime_cost = _mapping(candidate.get("runtime_cost"))
    for field in (
        "layer",
        "batch_size",
        "hidden_state_capture",
        "covariance_mode",
        "covariance_low_rank",
        "max_batch_tokens",
        "prefix_kv_cache",
        "max_workers",
    ):
        _append_runtime_mismatch(failures, field, recommendation.get(field), runtime.get(field))
    _append_runtime_mismatch(
        failures,
        "inside_trigger_budget_id",
        _performance_inside_trigger_budget_id(recommendation),
        runtime_cost.get("inside_trigger_budget_id"),
    )
    _append_runtime_mismatch(
        failures,
        "inside_trigger_budget_policy",
        _performance_inside_trigger_budget_policy(recommendation),
        runtime_cost.get("inside_trigger_budget_policy"),
    )
    _append_best_quality_mismatch(
        failures,
        expected=_mapping(recommendation.get("best_quality_signal")),
        observed=_mapping(quality.get("best_quality_signal")),
    )
    return {
        "passed": not failures,
        "blocking_reasons": failures,
        "score_dump_cache": score_dump_cache_gate,
        "covariance_tradeoff": covariance_gate,
        "performance_trend": None if performance_trend_gate is None else dict(performance_trend_gate),
    }


def _performance_score_dump_cache_gate(
    score_dump_cache: Mapping[str, Any],
    *,
    required: bool,
    min_jsonl_view_hit_rate: float | None,
) -> dict[str, Any]:
    totals = _mapping(score_dump_cache.get("totals"))
    jsonl_view = _mapping(totals.get("jsonl_view"))
    source_count = _float_or_none(score_dump_cache.get("source_count"))
    jsonl_view_hit_rate = _float_or_none(jsonl_view.get("hit_rate"))
    failures = []
    enabled = bool(required or min_jsonl_view_hit_rate is not None)
    if required and (
        score_dump_cache.get("enabled") is not True
        or source_count is None
        or source_count < 1
    ):
        failures.append("performance score-dump cache evidence is required but missing")
    if min_jsonl_view_hit_rate is not None:
        if jsonl_view_hit_rate is None:
            failures.append("performance score-dump cache jsonl_view hit rate is missing")
        elif jsonl_view_hit_rate < min_jsonl_view_hit_rate:
            failures.append(
                "performance score-dump cache jsonl_view hit rate below "
                f"{min_jsonl_view_hit_rate}: {jsonl_view_hit_rate}"
            )
    return {
        "enabled": enabled,
        "required": required,
        "min_jsonl_view_hit_rate": min_jsonl_view_hit_rate,
        "observed_source_count": source_count,
        "observed_jsonl_view_hit_rate": jsonl_view_hit_rate,
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _performance_trend_gate(
    *,
    current_bundle: Mapping[str, Any],
    reference_bundle: Mapping[str, Any],
    reference_report_error: str | None,
    reference_manifest_path: Path | None,
    reference_verification: Mapping[str, Any] | None,
    reference_record_key: str | None,
    reference_report_path: Path | None,
    allow_unverified: bool,
    max_uncached_total_seconds_ratio: float | None,
    max_cached_total_seconds_ratio: float | None,
    max_cache_only_total_seconds_ratio: float | None,
    max_score_dump_cache_jsonl_view_hit_rate_drop: float | None,
) -> dict[str, Any] | None:
    thresholds = {
        "max_uncached_total_seconds_ratio": max_uncached_total_seconds_ratio,
        "max_cached_total_seconds_ratio": max_cached_total_seconds_ratio,
        "max_cache_only_total_seconds_ratio": max_cache_only_total_seconds_ratio,
        "max_score_dump_cache_jsonl_view_hit_rate_drop": (
            max_score_dump_cache_jsonl_view_hit_rate_drop
        ),
    }
    enabled = any(value is not None for value in thresholds.values())
    if reference_record_key is None and not enabled:
        return None

    failures = []
    if enabled:
        if reference_record_key is None:
            failures.append("performance drift baseline record is missing")
        if reference_report_error is not None:
            failures.append(f"performance drift baseline report could not be loaded: {reference_report_error}")
        if reference_manifest_path is None:
            failures.append("performance drift baseline artifact manifest is missing")
        if (
            reference_verification is not None
            and not bool(reference_verification.get("passed", False))
            and not allow_unverified
        ):
            failures.append("performance drift baseline manifest verification failed")
        if reference_bundle.get("release_ready") is not True:
            failures.append(
                "performance drift baseline evidence bundle release_ready is "
                f"{reference_bundle.get('release_ready')!r}, expected True"
            )

    metrics = {
        "uncached_total_seconds": _performance_ratio_metric(
            "uncached_total_seconds",
            current=_performance_bundle_cost_metric(current_bundle, "uncached_total_seconds"),
            baseline=_performance_bundle_cost_metric(reference_bundle, "uncached_total_seconds"),
            threshold=max_uncached_total_seconds_ratio,
        ),
        "cached_total_seconds": _performance_ratio_metric(
            "cached_total_seconds",
            current=_performance_bundle_cost_metric(current_bundle, "cached_total_seconds"),
            baseline=_performance_bundle_cost_metric(reference_bundle, "cached_total_seconds"),
            threshold=max_cached_total_seconds_ratio,
        ),
        "cache_only_total_seconds": _performance_ratio_metric(
            "cache_only_total_seconds",
            current=_performance_bundle_cost_metric(current_bundle, "cache_only_total_seconds"),
            baseline=_performance_bundle_cost_metric(reference_bundle, "cache_only_total_seconds"),
            threshold=max_cache_only_total_seconds_ratio,
        ),
        "score_dump_cache_jsonl_view_hit_rate": _performance_drop_metric(
            "score_dump_cache_jsonl_view_hit_rate",
            current=_performance_bundle_jsonl_hit_rate(current_bundle),
            baseline=_performance_bundle_jsonl_hit_rate(reference_bundle),
            threshold=max_score_dump_cache_jsonl_view_hit_rate_drop,
        ),
    }
    for metric in metrics.values():
        failures.extend(metric["blocking_reasons"])

    return {
        "schema_version": 1,
        "enabled": enabled,
        "passed": not failures,
        "blocking_reasons": failures,
        "reference_record_key": reference_record_key,
        "reference_report_path": None if reference_report_path is None else str(reference_report_path),
        "reference_manifest_path": (
            None if reference_manifest_path is None else str(reference_manifest_path)
        ),
        "reference_manifest_verified": (
            None if reference_verification is None else bool(reference_verification.get("passed", False))
        ),
        "thresholds": thresholds,
        "metrics": metrics,
    }


def _performance_bundle_cost_metric(bundle: Mapping[str, Any], name: str) -> float | None:
    return _float_or_none(_mapping(bundle.get("cost")).get(name))


def _performance_bundle_jsonl_hit_rate(bundle: Mapping[str, Any]) -> float | None:
    score_dump_cache = _mapping(bundle.get("score_dump_cache"))
    totals = _mapping(score_dump_cache.get("totals"))
    jsonl_view = _mapping(totals.get("jsonl_view"))
    return _float_or_none(jsonl_view.get("hit_rate"))


def _performance_ratio_metric(
    metric: str,
    *,
    current: float | None,
    baseline: float | None,
    threshold: float | None,
) -> dict[str, Any]:
    failures = []
    observed_ratio = None
    enabled = threshold is not None
    if enabled:
        if current is None:
            failures.append(f"performance drift {metric} current value is missing")
        if baseline is None:
            failures.append(f"performance drift {metric} baseline value is missing")
        elif baseline <= 0:
            failures.append(f"performance drift {metric} baseline value must be positive")
        if current is not None and baseline is not None and baseline > 0:
            observed_ratio = current / baseline
            if observed_ratio > threshold:
                failures.append(
                    f"performance drift {metric} ratio above {threshold}: {observed_ratio}"
                )
    return {
        "enabled": enabled,
        "metric": metric,
        "comparison": "ratio_to_baseline",
        "current": current,
        "baseline": baseline,
        "observed_ratio": observed_ratio,
        "threshold": threshold,
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _performance_drop_metric(
    metric: str,
    *,
    current: float | None,
    baseline: float | None,
    threshold: float | None,
) -> dict[str, Any]:
    failures = []
    observed_drop = None
    enabled = threshold is not None
    if enabled:
        if current is None:
            failures.append(f"performance drift {metric} current value is missing")
        if baseline is None:
            failures.append(f"performance drift {metric} baseline value is missing")
        if current is not None and baseline is not None:
            observed_drop = baseline - current
            if observed_drop > threshold:
                failures.append(
                    f"performance drift {metric} drop above {threshold}: {observed_drop}"
                )
    return {
        "enabled": enabled,
        "metric": metric,
        "comparison": "absolute_drop_from_baseline",
        "current": current,
        "baseline": baseline,
        "observed_drop": observed_drop,
        "threshold": threshold,
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _append_runtime_mismatch(
    failures: list[str],
    field: str,
    expected: Any,
    observed: Any,
) -> None:
    if expected is None:
        return
    if not _json_value_equal(expected, observed):
        failures.append(f"performance baseline runtime {field} mismatch: expected {expected!r}, got {observed!r}")


def _append_best_quality_mismatch(
    failures: list[str],
    *,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> None:
    expected_name = expected.get("name")
    if expected_name is None:
        return
    observed_name = observed.get("name")
    if expected_name != observed_name:
        failures.append(
            "performance baseline best quality signal mismatch: "
            f"expected {expected_name!r}, got {observed_name!r}"
        )
        return
    expected_auroc = _float_or_none(expected.get("auroc"))
    observed_auroc = _float_or_none(observed.get("auroc"))
    if expected_auroc is not None and (
        observed_auroc is None or abs(expected_auroc - observed_auroc) > 1e-12
    ):
        failures.append(
            "performance baseline best quality AUROC mismatch: "
            f"expected {expected_auroc!r}, got {observed_auroc!r}"
        )


def _json_value_equal(left: Any, right: Any) -> bool:
    left_float = _float_or_none(left)
    right_float = _float_or_none(right)
    if left_float is not None or right_float is not None:
        return left_float is not None and right_float is not None and abs(left_float - right_float) <= 1e-12
    return left == right


def _performance_manifest_path(
    record: Any,
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> Path | None:
    metadata = _mapping(record.metadata)
    raw_path = _first_present(
        metadata.get("artifact_manifest"),
        _mapping(report.get("paths")).get("artifact_manifest"),
    )
    if raw_path is None:
        return None
    return _resolve_path(raw_path, base_path=report_path)


def _performance_runtime_recommendation(
    record: Any,
    report: Mapping[str, Any],
    *,
    report_path: Path,
    verification_context: ArtifactVerificationContext,
) -> tuple[dict[str, Any], str | None]:
    runtime = _mapping(report.get("runtime_recommendation"))
    if runtime:
        return runtime, str(report_path)
    metadata = _mapping(record.metadata)
    raw_path = _first_present(
        metadata.get("runtime_recommendation"),
        _mapping(report.get("paths")).get("runtime_recommendation"),
    )
    if raw_path is None:
        return {}, None
    path = _resolve_path(raw_path, base_path=report_path)
    payload, _ = verification_context.load_json_object(path)
    return payload, str(path)


def _verify_performance_manifest(
    manifest_path: Path | None,
    *,
    recursive: bool,
    max_workers: int,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any]:
    return _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=max_workers,
        artifact_name="performance_baseline_manifest",
        verification_context=verification_context,
    )


def _verify_artifact_manifest(
    manifest_path: Path | None,
    *,
    recursive: bool,
    max_workers: int,
    artifact_name: str,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any]:
    if manifest_path is None:
        return {
            "manifest_path": None,
            "passed": False,
            "checked": 0,
            "failures": [{
                "name": artifact_name,
                "path": "",
                "field": "path",
                "expected": "artifact manifest path",
                "actual": None,
            }],
            "nested": [],
        }
    try:
        return verification_context.load_and_verify_artifact_manifest(
            manifest_path,
            recursive=recursive,
            max_workers=max_workers,
        ).to_dict()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "manifest_path": str(manifest_path),
            "passed": False,
            "checked": 0,
            "failures": [{
                "name": artifact_name,
                "path": str(manifest_path),
                "field": "load",
                "expected": "readable artifact manifest",
                "actual": str(exc),
            }],
            "nested": [],
        }


def _candidate_with_gates(
    candidate: Mapping[str, Any] | None,
    performance: Mapping[str, Any] | None,
    adapter_family: Mapping[str, Any] | None,
    triple_extraction_fixture_matrix: Mapping[str, Any] | None,
    counterfactual_verification: Mapping[str, Any] | None,
    required_routes: Mapping[str, Any] | None,
    product_trace_replay_workflow: Mapping[str, Any] | None,
    selector_replay: Mapping[str, Any] | None,
    product_runtime_drift: Mapping[str, Any] | None,
    release_efficiency: Mapping[str, Any] | None,
    external_evidence_baseline_comparison: Mapping[str, Any] | None,
    pre_generation_probe_comparison: Mapping[str, Any] | None,
    claim_factuality_probe_comparison: Mapping[str, Any] | None,
    frontier_release_evidence: Mapping[str, Any] | None,
    world_model_signal_workflow: Mapping[str, Any] | None,
    context_sensitivity_workflow: Mapping[str, Any] | None,
    mechanism_handoff_evidence_bundle: Mapping[str, Any] | None,
    pathway_intervention_workflow: Mapping[str, Any] | None,
    selfcheck_signal_fusion_workflow: Mapping[str, Any] | None,
    uncertainty_escalation_workflow: Mapping[str, Any] | None,
    feedback_policy_workflow: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if candidate is None:
        return None
    payload = dict(candidate)
    manifests = dict(payload.get("manifests") or {})
    if performance is not None:
        payload["performance_baseline_record"] = performance.get("record_key")
        performance_evidence_bundle = _mapping(performance.get("performance_evidence_bundle"))
        if performance_evidence_bundle:
            payload["performance_evidence_bundle"] = performance_evidence_bundle
        manifests["performance_manifest"] = performance.get("manifest_path")
    if adapter_family is not None:
        payload["adapter_family_matrix"] = {
            "matrix_path": adapter_family.get("matrix_path"),
            "required_routes": tuple(adapter_family.get("required_routes", ())),
            "routes": tuple(adapter_family.get("routes", ())),
            "retrieval_routes": tuple(adapter_family.get("retrieval_routes", ())),
            "audit_routes": tuple(adapter_family.get("audit_routes", ())),
            "promoted_routes": tuple(adapter_family.get("promoted_routes", ())),
            "require_state_transition_world_model": adapter_family.get(
                "require_state_transition_world_model"
            ),
            "state_transition_world_model_adapter": adapter_family.get(
                "state_transition_world_model_adapter"
            ),
            "state_transition_world_model_rule_count": adapter_family.get(
                "state_transition_world_model_rule_count"
            ),
            "promotion_status": adapter_family.get("promotion_status"),
        }
        manifests["adapter_family_matrix_report"] = adapter_family.get("matrix_path")
    if triple_extraction_fixture_matrix is not None:
        payload["triple_extraction_fixture_matrix"] = {
            "report_path": triple_extraction_fixture_matrix.get("report_path"),
            "manifest_path": triple_extraction_fixture_matrix.get("manifest_path"),
            "source": triple_extraction_fixture_matrix.get("source"),
            "registry": triple_extraction_fixture_matrix.get("registry"),
            "record_key": triple_extraction_fixture_matrix.get("record_key"),
            "status": triple_extraction_fixture_matrix.get("report_status"),
            "n_corpora": triple_extraction_fixture_matrix.get("n_corpora"),
            "promoted_corpora": triple_extraction_fixture_matrix.get("promoted_corpora"),
            "distinct_predicate_count": triple_extraction_fixture_matrix.get("distinct_predicate_count"),
            "distinct_predicates": tuple(
                triple_extraction_fixture_matrix.get("distinct_predicates", ())
            ),
            "mean_best_f1": triple_extraction_fixture_matrix.get("mean_best_f1"),
            "mean_f1_lift": triple_extraction_fixture_matrix.get("mean_f1_lift"),
            "external_prediction_count": triple_extraction_fixture_matrix.get(
                "external_prediction_count"
            ),
            "external_prediction_corpora": tuple(
                triple_extraction_fixture_matrix.get("external_prediction_corpora", ())
            ),
            "mean_best_external_f1": triple_extraction_fixture_matrix.get(
                "mean_best_external_f1"
            ),
        }
        manifests["triple_extraction_fixture_matrix_report"] = (
            triple_extraction_fixture_matrix.get("report_path")
        )
        manifests["triple_extraction_fixture_matrix_manifest"] = (
            triple_extraction_fixture_matrix.get("manifest_path")
        )
    if counterfactual_verification is not None:
        payload["counterfactual_verification"] = {
            "report_path": counterfactual_verification.get("report_path"),
            "manifest_path": counterfactual_verification.get("manifest_path"),
            "source": counterfactual_verification.get("source"),
            "registry": counterfactual_verification.get("registry"),
            "record_key": counterfactual_verification.get("record_key"),
            "workflow": counterfactual_verification.get("workflow"),
            "record_count": counterfactual_verification.get("record_count"),
            "pass_rate": counterfactual_verification.get("pass_rate"),
            "false_invariance_rate": counterfactual_verification.get("false_invariance_rate"),
            "flip_success_count": counterfactual_verification.get("flip_success_count"),
        }
        manifests["counterfactual_verification_report"] = (
            counterfactual_verification.get("report_path")
        )
        manifests["counterfactual_verification_manifest"] = (
            counterfactual_verification.get("manifest_path")
        )
    if required_routes is not None:
        required_rows = tuple(_mapping(row) for row in required_routes.get("rows", ()))
        required_records = tuple(row.get("record_key") for row in required_rows if row.get("record_key") is not None)
        required_manifest_paths = tuple(
            row.get("manifest_path") for row in required_rows if row.get("manifest_path") is not None
        )
        property_counts: dict[str, Any] = {}
        property_sets: dict[str, tuple[str, ...]] = {}
        property_metrics: dict[str, dict[str, Any]] = {}
        for row in required_rows:
            record_key = row.get("record_key")
            if record_key is None:
                continue
            summary = _covered_fact_property_summary(row)
            if (
                summary["count"] is None
                and not summary["properties"]
                and not summary["metrics"]
            ):
                continue
            key = str(record_key)
            property_counts[key] = summary["count"]
            property_sets[key] = summary["properties"]
            property_metrics[key] = summary["metrics"]
        payload["required_route_baselines"] = {
            "registry": required_routes.get("registry"),
            "records": required_records,
            "routes": tuple(row.get("recommended_route") for row in required_rows),
            "manifest_paths": required_manifest_paths,
            "covered_fact_property_counts": property_counts,
            "covered_fact_properties": property_sets,
            "covered_fact_property_metrics": property_metrics,
        }
        for idx, manifest_path in enumerate(required_manifest_paths, start=1):
            manifests[f"required_route_manifest_{idx}"] = manifest_path
    if product_trace_replay_workflow is not None:
        payload["product_trace_replay_workflow"] = {
            "report_path": product_trace_replay_workflow.get("report_path"),
            "manifest_path": product_trace_replay_workflow.get("manifest_path"),
            "source": product_trace_replay_workflow.get("source"),
            "registry": product_trace_replay_workflow.get("registry"),
            "record_key": product_trace_replay_workflow.get("record_key"),
            "report_status": product_trace_replay_workflow.get("report_status"),
            "selector_replay_report_path": product_trace_replay_workflow.get(
                "selector_replay_report_path"
            ),
            "product_runtime_drift_report_path": product_trace_replay_workflow.get(
                "product_runtime_drift_report_path"
            ),
            "require_action_audit_gate": product_trace_replay_workflow.get(
                "require_action_audit_gate"
            ),
            "action_audit_gate": dict(
                _mapping(product_trace_replay_workflow.get("action_audit_gate"))
            ),
            "action_audit_gate_report_path": product_trace_replay_workflow.get(
                "action_audit_gate_report_path"
            ),
            "require_action_execution_gate": product_trace_replay_workflow.get(
                "require_action_execution_gate"
            ),
            "action_execution_gate": dict(
                _mapping(product_trace_replay_workflow.get("action_execution_gate"))
            ),
            "action_execution_gate_report_path": product_trace_replay_workflow.get(
                "action_execution_gate_report_path"
            ),
        }
        manifests["product_trace_replay_workflow_manifest"] = product_trace_replay_workflow.get(
            "manifest_path"
        )
        if product_trace_replay_workflow.get("action_execution_gate_report_path") is not None:
            manifests["product_trace_action_execution_gate_report"] = (
                product_trace_replay_workflow.get("action_execution_gate_report_path")
            )
    if selector_replay is not None:
        payload["selector_replay"] = {
            "report_path": selector_replay.get("report_path"),
            "manifest_path": selector_replay.get("manifest_path"),
            "recommended_candidate": selector_replay.get("recommended_candidate"),
            "recommended_policy_path": selector_replay.get("recommended_policy_path"),
            "recommended": _mapping(selector_replay.get("recommended")),
        }
        manifests["selector_replay_manifest"] = selector_replay.get("manifest_path")
    if product_runtime_drift is not None:
        payload["product_runtime_drift"] = {
            "report_path": product_runtime_drift.get("report_path"),
            "manifest_path": product_runtime_drift.get("manifest_path"),
            "report_status": product_runtime_drift.get("report_status"),
            "decision_status": product_runtime_drift.get("decision_status"),
            "baseline": _mapping(product_runtime_drift.get("baseline")),
            "current": _mapping(product_runtime_drift.get("current")),
            "summary": _mapping(product_runtime_drift.get("summary")),
            "metrics": tuple(_mapping(metric) for metric in product_runtime_drift.get("metrics") or ()),
        }
        manifests["product_runtime_drift_manifest"] = product_runtime_drift.get("manifest_path")
    if release_efficiency is not None:
        leaderboard_top = _mapping(release_efficiency.get("leaderboard_top"))
        payload["release_efficiency"] = {
            "report_path": release_efficiency.get("report_path"),
            "manifest_path": release_efficiency.get("manifest_path"),
            "workflow": release_efficiency.get("workflow"),
            "status": release_efficiency.get("report_status"),
            "recommended_profile": release_efficiency.get("recommended_profile"),
            "recommended_efficiency_score": release_efficiency.get(
                "recommended_efficiency_score"
            ),
            "decision": {
                "status": release_efficiency.get("decision_status"),
                "recommended_profile": release_efficiency.get("recommended_profile"),
                "recommended_efficiency_score": release_efficiency.get(
                    "recommended_efficiency_score"
                ),
                "blocking_reasons": tuple(release_efficiency.get("blocking_reasons", ())),
            },
            "summary": _mapping(release_efficiency.get("summary")),
            "leaderboard": (leaderboard_top,) if leaderboard_top else (),
        }
        manifests["release_efficiency_manifest"] = release_efficiency.get("manifest_path")
    if external_evidence_baseline_comparison is not None:
        payload["external_evidence_baseline_comparison"] = {
            "report_path": external_evidence_baseline_comparison.get("report_path"),
            "source": external_evidence_baseline_comparison.get("source"),
            "registry": external_evidence_baseline_comparison.get("registry"),
            "record_key": external_evidence_baseline_comparison.get("record_key"),
            "workflow": external_evidence_baseline_comparison.get("workflow"),
            "decision_status": external_evidence_baseline_comparison.get("decision_status"),
            "recommended_route": external_evidence_baseline_comparison.get("recommended_route"),
            "recommended_route_record": external_evidence_baseline_comparison.get(
                "recommended_route_record"
            ),
            "route_passed": external_evidence_baseline_comparison.get("route_passed"),
            "text_redline_passed": external_evidence_baseline_comparison.get(
                "text_redline_passed"
            ),
            "text_redline_run_count": external_evidence_baseline_comparison.get(
                "text_redline_run_count"
            ),
            "blocking_reasons": tuple(
                external_evidence_baseline_comparison.get("blocking_reasons", ())
            ),
        }
        manifests["external_evidence_baseline_comparison_report"] = (
            external_evidence_baseline_comparison.get("report_path")
        )
    if pre_generation_probe_comparison is not None:
        best_run = _mapping(pre_generation_probe_comparison.get("best_run"))
        payload["pre_generation_probe_comparison"] = {
            "report_path": pre_generation_probe_comparison.get("report_path"),
            "manifest_path": pre_generation_probe_comparison.get("manifest_path"),
            "source": pre_generation_probe_comparison.get("source"),
            "registry": pre_generation_probe_comparison.get("registry"),
            "record_key": pre_generation_probe_comparison.get("record_key"),
            "workflow": pre_generation_probe_comparison.get("workflow"),
            "status": pre_generation_probe_comparison.get("report_status"),
            "model_count": pre_generation_probe_comparison.get("model_count"),
            "run_count": pre_generation_probe_comparison.get("run_count"),
            "redline_passed": pre_generation_probe_comparison.get("redline_passed"),
            "redline_run_count": pre_generation_probe_comparison.get("redline_run_count"),
            "best_run": best_run,
            "blocking_reasons": tuple(pre_generation_probe_comparison.get("blocking_reasons", ())),
        }
        manifests["pre_generation_probe_comparison_manifest"] = (
            pre_generation_probe_comparison.get("manifest_path")
        )
    if claim_factuality_probe_comparison is not None:
        best_run = _mapping(claim_factuality_probe_comparison.get("best_run"))
        payload["claim_factuality_probe_comparison"] = {
            "report_path": claim_factuality_probe_comparison.get("report_path"),
            "manifest_path": claim_factuality_probe_comparison.get("manifest_path"),
            "source": claim_factuality_probe_comparison.get("source"),
            "registry": claim_factuality_probe_comparison.get("registry"),
            "record_key": claim_factuality_probe_comparison.get("record_key"),
            "workflow": claim_factuality_probe_comparison.get("workflow"),
            "status": claim_factuality_probe_comparison.get("report_status"),
            "model_count": claim_factuality_probe_comparison.get("model_count"),
            "run_count": claim_factuality_probe_comparison.get("run_count"),
            "dataset_count": claim_factuality_probe_comparison.get("dataset_count"),
            "datasets": claim_factuality_probe_comparison.get("datasets"),
            "redline_passed": claim_factuality_probe_comparison.get("redline_passed"),
            "redline_run_count": claim_factuality_probe_comparison.get("redline_run_count"),
            "best_run": best_run,
            "blocking_reasons": tuple(
                claim_factuality_probe_comparison.get("blocking_reasons", ())
            ),
        }
        manifests["claim_factuality_probe_comparison_manifest"] = (
            claim_factuality_probe_comparison.get("manifest_path")
        )
    if frontier_release_evidence is not None:
        payload["frontier_release_evidence"] = {
            "report_path": frontier_release_evidence.get("report_path"),
            "manifest_path": frontier_release_evidence.get("manifest_path"),
            "source": frontier_release_evidence.get("source"),
            "registry": frontier_release_evidence.get("registry"),
            "record_key": frontier_release_evidence.get("record_key"),
            "workflow": frontier_release_evidence.get("workflow"),
            "report_status": frontier_release_evidence.get("report_status"),
            "decision_status": frontier_release_evidence.get("decision_status"),
            "verifier_track_status": frontier_release_evidence.get("verifier_track_status"),
            "abstention_track_status": frontier_release_evidence.get("abstention_track_status"),
            "multiple_testing_track_status": frontier_release_evidence.get(
                "multiple_testing_track_status"
            ),
            "citation_batch_track_status": frontier_release_evidence.get(
                "citation_batch_track_status"
            ),
            "frontier_rerun_rollup_track_status": frontier_release_evidence.get(
                "frontier_rerun_rollup_track_status"
            ),
            "base_verifier_track_status": frontier_release_evidence.get(
                "base_verifier_track_status"
            ),
            "base_abstention_track_status": frontier_release_evidence.get(
                "base_abstention_track_status"
            ),
            "base_detectability_track_status": frontier_release_evidence.get(
                "base_detectability_track_status"
            ),
            "base_multiple_testing_track_status": frontier_release_evidence.get(
                "base_multiple_testing_track_status"
            ),
            "frontier_rerun_rollup_promoted_tracks": tuple(
                frontier_release_evidence.get("frontier_rerun_rollup_promoted_tracks", ())
            ),
            "frontier_rerun_rollup_report_count": frontier_release_evidence.get(
                "frontier_rerun_rollup_report_count"
            ),
            "frontier_rerun_rollup_candidate_count": frontier_release_evidence.get(
                "frontier_rerun_rollup_candidate_count"
            ),
            "frontier_rerun_rollup_missing_report_count": frontier_release_evidence.get(
                "frontier_rerun_rollup_missing_report_count"
            ),
            "frontier_rerun_rollup_invalid_report_count": frontier_release_evidence.get(
                "frontier_rerun_rollup_invalid_report_count"
            ),
            "frontier_rerun_rollup_blocked_candidate_count": frontier_release_evidence.get(
                "frontier_rerun_rollup_blocked_candidate_count"
            ),
            "frontier_rerun_rollup_promotion_ready_count": frontier_release_evidence.get(
                "frontier_rerun_rollup_promotion_ready_count"
            ),
            "citation_batch_rollup_count": frontier_release_evidence.get(
                "citation_batch_rollup_count"
            ),
            "citation_batch_expected_batch_count": frontier_release_evidence.get(
                "citation_batch_expected_batch_count"
            ),
            "citation_batch_observed_batch_count": frontier_release_evidence.get(
                "citation_batch_observed_batch_count"
            ),
            "citation_batch_missing_expected_batch_count": frontier_release_evidence.get(
                "citation_batch_missing_expected_batch_count"
            ),
            "citation_batch_duplicate_batch_count": frontier_release_evidence.get(
                "citation_batch_duplicate_batch_count"
            ),
            "citation_batch_unexpected_batch_count": frontier_release_evidence.get(
                "citation_batch_unexpected_batch_count"
            ),
            "require_input_manifests": frontier_release_evidence.get(
                "require_input_manifests"
            ),
            "input_manifest_required": frontier_release_evidence.get(
                "input_manifest_required"
            ),
            "input_manifest_required_count": frontier_release_evidence.get(
                "input_manifest_required_count"
            ),
            "input_manifest_verified_count": frontier_release_evidence.get(
                "input_manifest_verified_count"
            ),
            "input_manifest_failed_count": frontier_release_evidence.get(
                "input_manifest_failed_count"
            ),
            "input_manifest_missing_count": frontier_release_evidence.get(
                "input_manifest_missing_count"
            ),
            "input_manifest_failure_count": frontier_release_evidence.get(
                "input_manifest_failure_count"
            ),
            "run_names": tuple(frontier_release_evidence.get("run_names", ())),
            "blocking_reasons": tuple(frontier_release_evidence.get("blocking_reasons", ())),
        }
        manifests["frontier_release_evidence_manifest"] = frontier_release_evidence.get(
            "manifest_path"
        )
    if world_model_signal_workflow is not None:
        payload["world_model_signal_workflow"] = {
            "report_path": world_model_signal_workflow.get("report_path"),
            "manifest_path": world_model_signal_workflow.get("manifest_path"),
            "source": world_model_signal_workflow.get("source"),
            "registry": world_model_signal_workflow.get("registry"),
            "record_key": world_model_signal_workflow.get("record_key"),
            "workflow": world_model_signal_workflow.get("workflow"),
            "release_gate_status": world_model_signal_workflow.get("release_gate_status"),
            "trace_gap_max": world_model_signal_workflow.get("trace_gap_max"),
            "conflict_positive_count": world_model_signal_workflow.get(
                "conflict_positive_count"
            ),
            "calibrated_conflict_signal_count": world_model_signal_workflow.get(
                "calibrated_conflict_signal_count"
            ),
            "blocking_reasons": tuple(world_model_signal_workflow.get("blocking_reasons", ())),
        }
        manifests["world_model_signal_workflow_manifest"] = (
            world_model_signal_workflow.get("manifest_path")
        )
    if context_sensitivity_workflow is not None:
        payload["context_sensitivity_workflow"] = {
            "report_path": context_sensitivity_workflow.get("report_path"),
            "manifest_path": context_sensitivity_workflow.get("manifest_path"),
            "source": context_sensitivity_workflow.get("source"),
            "registry": context_sensitivity_workflow.get("registry"),
            "record_key": context_sensitivity_workflow.get("record_key"),
            "workflow": context_sensitivity_workflow.get("workflow"),
            "paired_logprob_record_count": context_sensitivity_workflow.get(
                "paired_logprob_record_count"
            ),
            "enriched_record_count": context_sensitivity_workflow.get(
                "enriched_record_count"
            ),
            "enhanced_score_signal_count": context_sensitivity_workflow.get(
                "enhanced_score_signal_count"
            ),
            "max_flagged_rate": context_sensitivity_workflow.get("max_flagged_rate"),
            "max_context_sensitivity_ratio": context_sensitivity_workflow.get(
                "max_context_sensitivity_ratio"
            ),
            "manifest_verified": context_sensitivity_workflow.get("manifest_verified"),
            "blocking_reasons": tuple(
                context_sensitivity_workflow.get("blocking_reasons", ())
            ),
        }
        manifests["context_sensitivity_workflow_manifest"] = (
            context_sensitivity_workflow.get("manifest_path")
        )
    if mechanism_handoff_evidence_bundle is not None:
        payload["mechanism_handoff_evidence_bundle"] = {
            "report_path": mechanism_handoff_evidence_bundle.get("report_path"),
            "manifest_path": mechanism_handoff_evidence_bundle.get("manifest_path"),
            "source": mechanism_handoff_evidence_bundle.get("source"),
            "registry": mechanism_handoff_evidence_bundle.get("registry"),
            "record_key": mechanism_handoff_evidence_bundle.get("record_key"),
            "workflow": mechanism_handoff_evidence_bundle.get("workflow"),
            "status": mechanism_handoff_evidence_bundle.get("report_status"),
            "handoff_count": mechanism_handoff_evidence_bundle.get("handoff_count"),
            "trace_count": mechanism_handoff_evidence_bundle.get("trace_count"),
            "target_count": mechanism_handoff_evidence_bundle.get("target_count"),
            "target_coverage_rate": mechanism_handoff_evidence_bundle.get(
                "target_coverage_rate"
            ),
            "source_citation_count": mechanism_handoff_evidence_bundle.get(
                "source_citation_count"
            ),
            "verification_status_counts": dict(
                mechanism_handoff_evidence_bundle.get("verification_status_counts") or {}
            ),
            "action_counts": dict(mechanism_handoff_evidence_bundle.get("action_counts") or {}),
            "source_family_counts": dict(
                mechanism_handoff_evidence_bundle.get("source_family_counts") or {}
            ),
            "blocking_reasons": tuple(
                mechanism_handoff_evidence_bundle.get("blocking_reasons", ())
            ),
        }
        manifests["mechanism_handoff_evidence_bundle_manifest"] = (
            mechanism_handoff_evidence_bundle.get("manifest_path")
        )
    if pathway_intervention_workflow is not None:
        payload["pathway_intervention_workflow"] = {
            "report_path": pathway_intervention_workflow.get("report_path"),
            "manifest_path": pathway_intervention_workflow.get("manifest_path"),
            "source": pathway_intervention_workflow.get("source"),
            "registry": pathway_intervention_workflow.get("registry"),
            "record_key": pathway_intervention_workflow.get("record_key"),
            "workflow": pathway_intervention_workflow.get("workflow"),
            "report_status": pathway_intervention_workflow.get("report_status"),
            "release_ready": pathway_intervention_workflow.get("release_ready"),
            "model": pathway_intervention_workflow.get("model"),
            "layer": pathway_intervention_workflow.get("layer"),
            "intervention_layer": pathway_intervention_workflow.get("intervention_layer"),
            "patch_layer": pathway_intervention_workflow.get("patch_layer"),
            "signals": tuple(pathway_intervention_workflow.get("signals", ())),
            "activation_ablation_gate_status": (
                pathway_intervention_workflow.get("activation_ablation_gate_status")
            ),
            "source_patch_gate_status": pathway_intervention_workflow.get("source_patch_gate_status"),
            "best_signals": dict(pathway_intervention_workflow.get("best_signals") or {}),
            "blocking_reasons": tuple(pathway_intervention_workflow.get("blocking_reasons", ())),
        }
        manifests["pathway_intervention_workflow_manifest"] = (
            pathway_intervention_workflow.get("manifest_path")
        )
    if selfcheck_signal_fusion_workflow is not None:
        payload["selfcheck_signal_fusion_workflow"] = {
            "report_path": selfcheck_signal_fusion_workflow.get("report_path"),
            "manifest_path": selfcheck_signal_fusion_workflow.get("manifest_path"),
            "source": selfcheck_signal_fusion_workflow.get("source"),
            "registry": selfcheck_signal_fusion_workflow.get("registry"),
            "record_key": selfcheck_signal_fusion_workflow.get("record_key"),
            "workflow": selfcheck_signal_fusion_workflow.get("workflow"),
            "status": selfcheck_signal_fusion_workflow.get("status"),
            "sample_quality_status": selfcheck_signal_fusion_workflow.get(
                "sample_quality_status"
            ),
            "sample_quality_passed": selfcheck_signal_fusion_workflow.get(
                "sample_quality_passed"
            ),
            "sample_quality_failed_runs": tuple(
                selfcheck_signal_fusion_workflow.get("sample_quality_failed_runs", ())
            ),
            "sample_quality_run_count": selfcheck_signal_fusion_workflow.get(
                "sample_quality_run_count"
            ),
            "sample_quality_runs": tuple(
                _mapping(run)
                for run in selfcheck_signal_fusion_workflow.get("sample_quality_runs", ())
            ),
            "fusion_run_count": selfcheck_signal_fusion_workflow.get("fusion_run_count"),
            "geometry_fusion_artifact_count": selfcheck_signal_fusion_workflow.get(
                "geometry_fusion_artifact_count"
            ),
            "enhanced_score_dump_count": selfcheck_signal_fusion_workflow.get(
                "enhanced_score_dump_count"
            ),
        }
        manifests["selfcheck_signal_fusion_workflow_manifest"] = (
            selfcheck_signal_fusion_workflow.get("manifest_path")
        )
    if uncertainty_escalation_workflow is not None:
        payload["uncertainty_escalation_workflow"] = {
            "report_path": uncertainty_escalation_workflow.get("report_path"),
            "manifest_path": uncertainty_escalation_workflow.get("manifest_path"),
            "source": uncertainty_escalation_workflow.get("source"),
            "registry": uncertainty_escalation_workflow.get("registry"),
            "record_key": uncertainty_escalation_workflow.get("record_key"),
            "workflow": uncertainty_escalation_workflow.get("workflow"),
            "record_count": uncertainty_escalation_workflow.get("record_count"),
            "trigger_rate": uncertainty_escalation_workflow.get("trigger_rate"),
            "retrieval_evidence_rate": uncertainty_escalation_workflow.get(
                "retrieval_evidence_rate"
            ),
            "final_false_accept_rate": uncertainty_escalation_workflow.get(
                "final_false_accept_rate"
            ),
            "false_accept_delta": uncertainty_escalation_workflow.get(
                "false_accept_delta"
            ),
            "accepted_false_delta": uncertainty_escalation_workflow.get(
                "accepted_false_delta"
            ),
        }
        manifests["uncertainty_escalation_workflow_manifest"] = (
            uncertainty_escalation_workflow.get("manifest_path")
        )
    if feedback_policy_workflow is not None:
        payload["feedback_policy_workflow"] = {
            "report_path": feedback_policy_workflow.get("report_path"),
            "manifest_path": feedback_policy_workflow.get("manifest_path"),
            "source": feedback_policy_workflow.get("source"),
            "registry": feedback_policy_workflow.get("registry"),
            "record_key": feedback_policy_workflow.get("record_key"),
            "report_status": feedback_policy_workflow.get("report_status"),
            "promotion_decision": feedback_policy_workflow.get("promotion_decision"),
            "candidate_control_policy": feedback_policy_workflow.get("candidate_control_policy"),
            "candidate_control_policy_config": feedback_policy_workflow.get(
                "candidate_control_policy_config"
            ),
            "candidate_control_defaults": feedback_policy_workflow.get("candidate_control_defaults"),
            "candidate_control_defaults_config": feedback_policy_workflow.get(
                "candidate_control_defaults_config"
            ),
            "matched_feedback_count": feedback_policy_workflow.get("matched_feedback_count"),
            "accepted_but_wrong_rate": feedback_policy_workflow.get("accepted_but_wrong_rate"),
            "retrieved_failure_rate": feedback_policy_workflow.get("retrieved_failure_rate"),
            "abstain_false_positive_rate": feedback_policy_workflow.get("abstain_false_positive_rate"),
            "final_answered_but_wrong_rate": feedback_policy_workflow.get(
                "final_answered_but_wrong_rate"
            ),
            "final_answer_false_block_rate": feedback_policy_workflow.get(
                "final_answer_false_block_rate"
            ),
            "safety_coverage_rate": feedback_policy_workflow.get("safety_coverage_rate"),
            "unknown_safety_issue_rate": feedback_policy_workflow.get("unknown_safety_issue_rate"),
        }
        manifests["feedback_policy_workflow_manifest"] = feedback_policy_workflow.get("manifest_path")
    payload["manifests"] = manifests
    return payload


def _performance_inside_trigger_budget_id(recommendation: Mapping[str, Any]) -> Any:
    inside_sampling = _mapping(recommendation.get("inside_sampling"))
    trigger_budget = _mapping(recommendation.get("inside_trigger_budget_sweep"))
    return _first_present(
        inside_sampling.get("inside_trigger_budget_id"),
        trigger_budget.get("recommended_budget_id"),
    )


def _performance_inside_trigger_budget_policy(recommendation: Mapping[str, Any]) -> Any:
    inside_sampling = _mapping(recommendation.get("inside_sampling"))
    trigger_budget = _mapping(recommendation.get("inside_trigger_budget_sweep"))
    return _first_present(
        inside_sampling.get("inside_trigger_budget_policy"),
        trigger_budget.get("selection_policy"),
        recommendation.get("inside_trigger_budget_policy"),
    )


def _resolve_path(raw_path: Any, *, base_path: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    report_relative = base_path.parent / path
    if report_relative.exists() or not path.exists():
        return report_relative
    return path


def _resolve_registry_record_path(registry_path: Path, record: Any) -> Path:
    path = Path(record.path)
    if path.is_absolute():
        return path
    sibling = registry_path.parent / path
    return sibling if sibling.exists() else path


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        numeric = float(value)
    else:
        try:
            numeric = float(str(value))
        except (TypeError, ValueError):
            return None
    return numeric if math.isfinite(numeric) else None


def _tuple_or_empty_sequence(value: Any) -> tuple[Any, ...]:
    if value is None or isinstance(value, str):
        return ()
    try:
        return tuple(value)
    except TypeError:
        return ()


def _validate_optional_non_negative_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    numeric = _float_or_none(value)
    if numeric is None or numeric < 0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return numeric


def _validate_optional_finite_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    numeric = _float_or_none(value)
    if numeric is None:
        raise ValueError(f"{name} must be a finite number.")
    return numeric


def _validate_optional_unit_float(value: Any, *, name: str) -> float | None:
    numeric = _validate_optional_non_negative_float(value, name=name)
    if numeric is not None and numeric > 1:
        raise ValueError(f"{name} must be between 0 and 1.")
    return numeric


def _validate_optional_non_negative_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if numeric < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return numeric


def _validate_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if numeric < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return numeric


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _apply_runtime_profile(
    runtime_profile: str | None,
    values: Mapping[str, Any],
) -> tuple[Any | None, dict[str, Any], dict[str, Any]]:
    profile = get_runtime_profile(runtime_profile)
    if profile is None:
        return None, dict(values), {}
    merged, applied = profile.apply_defaults(values)
    return profile, merged, applied


def _adapter_family_profile_routes(profile: str | None) -> tuple[str | None, tuple[str, ...]]:
    if profile is None:
        return None, ()
    normalized = str(profile).strip().lower().replace("-", "_")
    if normalized not in ADAPTER_FAMILY_PROFILES:
        choices = ", ".join(ADAPTER_FAMILY_PROFILE_NAMES)
        raise ValueError(f"adapter_family_profile must be one of: {choices}")
    return normalized, ADAPTER_FAMILY_PROFILES[normalized]


def adapter_family_profile_requires_state_transition_world_model(profile: str | None) -> bool:
    if profile is None:
        return False
    normalized = str(profile).strip().lower().replace("-", "_")
    return normalized in ADAPTER_FAMILY_PROFILES_REQUIRING_STATE_TRANSITION_WORLD_MODEL


def _merge_routes(*route_groups: Sequence[str]) -> tuple[str, ...]:
    routes = []
    seen = set()
    for group in route_groups:
        for route in group:
            route_text = str(route).strip()
            if not route_text or route_text in seen:
                continue
            routes.append(route_text)
            seen.add(route_text)
    return tuple(routes)


def _normalize_inside_trigger_budget_policy(policy: str | None) -> str | None:
    if policy is None:
        return None
    normalized = str(policy).strip().lower().replace("-", "_")
    if normalized not in INSIDE_TRIGGER_BUDGET_POLICIES:
        choices = ", ".join(INSIDE_TRIGGER_BUDGET_POLICIES)
        raise ValueError(f"inside_trigger_budget_policy must be one of: {choices}")
    return normalized


def _parse_non_negative_float(value: str, *, flag: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{flag} must be a non-negative finite number.")
    return numeric


def _parse_finite_float(value: str, *, flag: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{flag} must be a finite number.")
    return numeric


def _parse_non_negative_int(value: str, *, flag: str) -> int:
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{flag} must be a non-negative integer.")
    return numeric


def _parse_positive_int(value: str, *, flag: str) -> int:
    numeric = int(value)
    if numeric < 1:
        raise ValueError(f"{flag} must be a positive integer.")
    return numeric


def _parse_unit_float(value: str, *, flag: str) -> float:
    numeric = _parse_non_negative_float(value, flag=flag)
    if numeric > 1:
        raise ValueError(f"{flag} must be between 0 and 1.")
    return numeric


def _parse_csv(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in str(value).split(",") if part.strip())
    return tuple(parsed)


def _parse_key_values(values: Sequence[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not values:
        return parsed
    for value in values:
        for part in str(value).split(","):
            item = part.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"metadata requirement {item!r} must use key=value format.")
            key, raw = item.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError("metadata requirement key must be non-empty.")
            parsed[key] = raw.strip()
    return parsed


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = compare_release_candidates(
        readiness_registry_path=args.readiness_registry,
        route_registry_path=args.route_registry,
        readiness_baseline_keys=tuple(args.readiness_baseline_key or ()),
        route_baseline_keys=tuple(args.route_baseline_key or ()),
        required_route_baseline_keys=tuple(args.required_route_baseline_key or ()),
        release_policy_profile=args.release_policy_profile,
        require_structured_fact_robustness=bool(args.require_structured_fact_robustness),
        structured_fact_canonical_route_key=args.structured_fact_canonical_route_key,
        structured_fact_paraphrase_route_key=args.structured_fact_paraphrase_route_key,
        structured_fact_robustness_min_selected=args.structured_fact_robustness_min_selected,
        structured_fact_robustness_min_decision_accuracy=(
            args.structured_fact_robustness_min_decision_accuracy
        ),
        structured_fact_robustness_max_false_supported_rate=(
            args.structured_fact_robustness_max_false_supported_rate
        ),
        structured_fact_robustness_min_false_refuted_rate=(
            args.structured_fact_robustness_min_false_refuted_rate
        ),
        structured_fact_robustness_min_covered_fact_properties=(
            args.structured_fact_robustness_min_covered_fact_properties
        ),
        structured_fact_robustness_min_covered_fact_property_records=(
            args.structured_fact_robustness_min_covered_fact_property_records
        ),
        structured_fact_robustness_min_covered_fact_property_source_documents=(
            args.structured_fact_robustness_min_covered_fact_property_source_documents
        ),
        structured_fact_robustness_min_covered_fact_property_decision_accuracy=(
            args.structured_fact_robustness_min_covered_fact_property_decision_accuracy
        ),
        structured_fact_robustness_max_covered_fact_property_false_supported_rate=(
            args.structured_fact_robustness_max_covered_fact_property_false_supported_rate
        ),
        structured_fact_robustness_min_covered_fact_property_false_refuted_rate=(
            args.structured_fact_robustness_min_covered_fact_property_false_refuted_rate
        ),
        performance_registry_path=args.performance_registry,
        performance_baseline_key=args.performance_baseline_key,
        selector_replay_report_path=args.selector_replay_report,
        product_runtime_drift_report_path=args.product_runtime_drift_report,
        require_product_runtime_drift_promotion_evidence=bool(
            args.require_product_runtime_drift_promotion_evidence
        ),
        require_product_runtime_drift_pre_generation_evidence=bool(
            args.require_product_runtime_drift_pre_generation_evidence
        ),
        require_product_runtime_drift_claim_factuality_evidence=bool(
            args.require_product_runtime_drift_claim_factuality_evidence
        ),
        require_product_runtime_drift_claim_risk_localization_evidence=bool(
            args.require_product_runtime_drift_claim_risk_localization_evidence
        ),
        require_product_runtime_drift_counterfactual_evidence=bool(
            args.require_product_runtime_drift_counterfactual_evidence
        ),
        require_product_runtime_drift_fact_selfcheck_gate_evidence=bool(
            args.require_product_runtime_drift_fact_selfcheck_gate_evidence
        ),
        require_product_runtime_drift_triple_audit_evidence=bool(
            args.require_product_runtime_drift_triple_audit_evidence
        ),
        require_product_runtime_drift_covered_fact_property_evidence=bool(
            args.require_product_runtime_drift_covered_fact_property_evidence
        ),
        require_product_runtime_drift_action_gate_evidence=bool(
            args.require_product_runtime_drift_action_gate_evidence
        ),
        require_product_runtime_drift_world_model_action_gate_evidence=bool(
            args.require_product_runtime_drift_world_model_action_gate_evidence
        ),
        require_product_runtime_drift_world_model_rollout_evidence=bool(
            args.require_product_runtime_drift_world_model_rollout_evidence
        ),
        require_product_runtime_drift_action_receipts_evidence=bool(
            args.require_product_runtime_drift_action_receipts_evidence
        ),
        require_product_runtime_drift_receipt_claim_support_evidence=bool(
            args.require_product_runtime_drift_receipt_claim_support_evidence
        ),
        require_product_runtime_drift_trajectory_audit_evidence=bool(
            args.require_product_runtime_drift_trajectory_audit_evidence
        ),
        require_product_runtime_drift_provenance_evidence=bool(
            args.require_product_runtime_drift_provenance_evidence
        ),
        require_product_runtime_drift_citation_integrity_evidence=bool(
            args.require_product_runtime_drift_citation_integrity_evidence
        ),
        require_product_runtime_drift_evidence_quality_evidence=bool(
            args.require_product_runtime_drift_evidence_quality_evidence
        ),
        require_product_runtime_drift_metacognition_evidence=bool(
            args.require_product_runtime_drift_metacognition_evidence
        ),
        require_product_runtime_drift_evidence_handoff_evidence=bool(
            args.require_product_runtime_drift_evidence_handoff_evidence
        ),
        require_product_runtime_drift_world_model_evidence=bool(
            args.require_product_runtime_drift_world_model_evidence
        ),
        require_product_runtime_drift_context_sensitivity_evidence=bool(
            args.require_product_runtime_drift_context_sensitivity_evidence
        ),
        require_product_runtime_drift_evidence_alignment_evidence=bool(
            args.require_product_runtime_drift_evidence_alignment_evidence
        ),
        require_product_runtime_drift_counterfactual_robustness_evidence=bool(
            args.require_product_runtime_drift_counterfactual_robustness_evidence
        ),
        require_product_runtime_drift_frontier_release_evidence=bool(
            args.require_product_runtime_drift_frontier_release_evidence
        ),
        release_efficiency_report_path=args.release_efficiency_report,
        external_evidence_baseline_comparison_path=args.external_evidence_baseline_comparison,
        external_evidence_baseline_comparison_registry_path=(
            args.external_evidence_baseline_comparison_registry
        ),
        external_evidence_baseline_comparison_key=args.external_evidence_baseline_comparison_key,
        pre_generation_probe_comparison_path=args.pre_generation_probe_comparison,
        pre_generation_probe_comparison_registry_path=args.pre_generation_probe_comparison_registry,
        pre_generation_probe_comparison_key=args.pre_generation_probe_comparison_key,
        claim_factuality_probe_comparison_path=args.claim_factuality_probe_comparison,
        claim_factuality_probe_comparison_registry_path=(
            args.claim_factuality_probe_comparison_registry
        ),
        claim_factuality_probe_comparison_key=args.claim_factuality_probe_comparison_key,
        frontier_release_evidence_path=args.frontier_release_evidence,
        frontier_release_evidence_registry_path=args.frontier_release_evidence_registry,
        frontier_release_evidence_key=args.frontier_release_evidence_key,
        require_frontier_release_input_manifests=bool(
            args.require_frontier_release_input_manifests
        ),
        world_model_signal_workflow_path=args.world_model_signal_workflow,
        world_model_signal_workflow_registry_path=args.world_model_signal_workflow_registry,
        world_model_signal_workflow_key=args.world_model_signal_workflow_key,
        context_sensitivity_workflow_path=args.context_sensitivity_workflow,
        context_sensitivity_workflow_registry_path=args.context_sensitivity_workflow_registry,
        context_sensitivity_workflow_key=args.context_sensitivity_workflow_key,
        mechanism_handoff_evidence_bundle_path=args.mechanism_handoff_evidence_bundle,
        mechanism_handoff_evidence_bundle_registry_path=(
            args.mechanism_handoff_evidence_bundle_registry
        ),
        mechanism_handoff_evidence_bundle_key=args.mechanism_handoff_evidence_bundle_key,
        pathway_intervention_workflow_path=args.pathway_intervention_workflow,
        pathway_intervention_workflow_registry_path=args.pathway_intervention_workflow_registry,
        pathway_intervention_workflow_key=args.pathway_intervention_workflow_key,
        product_trace_replay_workflow_path=args.product_trace_replay_workflow,
        product_trace_replay_workflow_registry_path=args.product_trace_replay_workflow_registry,
        product_trace_replay_workflow_key=args.product_trace_replay_workflow_key,
        require_product_trace_action_audit_gate=bool(args.require_product_trace_action_audit_gate),
        require_product_trace_action_execution_gate=bool(
            args.require_product_trace_action_execution_gate
        ),
        selfcheck_signal_fusion_workflow_path=args.selfcheck_signal_fusion_workflow,
        selfcheck_signal_fusion_workflow_registry_path=args.selfcheck_signal_fusion_workflow_registry,
        selfcheck_signal_fusion_workflow_key=args.selfcheck_signal_fusion_workflow_key,
        uncertainty_escalation_workflow_path=args.uncertainty_escalation_workflow,
        uncertainty_escalation_workflow_registry_path=args.uncertainty_escalation_workflow_registry,
        uncertainty_escalation_workflow_key=args.uncertainty_escalation_workflow_key,
        min_uncertainty_escalation_records=args.min_uncertainty_escalation_records,
        min_uncertainty_escalation_trigger_rate=args.min_uncertainty_escalation_trigger_rate,
        min_uncertainty_escalation_retrieval_evidence_rate=(
            args.min_uncertainty_escalation_retrieval_evidence_rate
        ),
        max_uncertainty_escalation_final_false_accept_rate=(
            args.max_uncertainty_escalation_final_false_accept_rate
        ),
        max_uncertainty_escalation_false_accept_delta=(
            args.max_uncertainty_escalation_false_accept_delta
        ),
        feedback_policy_workflow_path=args.feedback_policy_workflow,
        feedback_policy_workflow_registry_path=args.feedback_policy_workflow_registry,
        feedback_policy_workflow_key=args.feedback_policy_workflow_key,
        feedback_policy_min_matched_feedback_count=args.feedback_policy_min_matched_feedback_count,
        feedback_policy_min_safety_coverage=args.feedback_policy_min_safety_coverage,
        feedback_policy_max_unknown_safety_issue_rate=args.feedback_policy_max_unknown_safety_issue_rate,
        adapter_family_matrix_path=args.adapter_family_matrix,
        adapter_family_profile=args.adapter_family_profile,
        required_adapter_routes=tuple(args.required_adapter_route or ()),
        require_state_transition_world_model=bool(args.require_state_transition_world_model),
        triple_extraction_fixture_matrix_path=args.triple_extraction_fixture_matrix,
        triple_extraction_fixture_matrix_registry_path=args.triple_extraction_fixture_matrix_registry,
        triple_extraction_fixture_matrix_key=args.triple_extraction_fixture_matrix_key,
        min_triple_extraction_corpora=args.min_triple_extraction_corpora,
        min_triple_extraction_distinct_predicates=args.min_triple_extraction_distinct_predicates,
        min_triple_extraction_external_prediction_count=(
            args.min_triple_extraction_external_prediction_count
        ),
        min_triple_extraction_external_prediction_corpora=(
            args.min_triple_extraction_external_prediction_corpora
        ),
        min_triple_extraction_mean_best_external_f1=(
            args.min_triple_extraction_mean_best_external_f1
        ),
        counterfactual_verification_report_path=args.counterfactual_verification_report,
        counterfactual_verification_registry_path=args.counterfactual_verification_registry,
        counterfactual_verification_key=args.counterfactual_verification_key,
        min_counterfactual_verification_records=args.min_counterfactual_verification_records,
        min_counterfactual_verification_pass_rate=args.min_counterfactual_verification_pass_rate,
        max_counterfactual_verification_false_invariance_rate=(
            args.max_counterfactual_verification_false_invariance_rate
        ),
        require_performance_score_dump_cache=bool(args.require_performance_score_dump_cache),
        min_performance_score_dump_cache_jsonl_view_hit_rate=(
            args.min_performance_score_dump_cache_jsonl_view_hit_rate
        ),
        performance_drift_baseline_key=args.performance_drift_baseline_key,
        max_performance_uncached_total_seconds_ratio=(
            args.max_performance_uncached_total_seconds_ratio
        ),
        max_performance_cached_total_seconds_ratio=(
            args.max_performance_cached_total_seconds_ratio
        ),
        max_performance_cache_only_total_seconds_ratio=(
            args.max_performance_cache_only_total_seconds_ratio
        ),
        max_performance_score_dump_cache_jsonl_view_hit_rate_drop=(
            args.max_performance_score_dump_cache_jsonl_view_hit_rate_drop
        ),
        recursive=not args.no_recursive,
        allow_unverified=bool(args.allow_unverified),
        manifest_fingerprint_workers=args.manifest_fingerprint_workers,
        runtime_profile=args.runtime_profile,
        inside_trigger_budget_policy=args.inside_trigger_budget_policy,
        min_best_quality_auroc=args.min_best_quality_auroc,
        max_uncached_forward_seconds=args.max_uncached_forward_seconds,
        max_cache_only_seconds=args.max_cache_only_seconds,
        max_recommended_runtime_seconds=args.max_recommended_runtime_seconds,
        max_covariance_maha_last_auroc_drop=args.max_covariance_maha_last_auroc_drop,
        max_inside_sample_count_ratio=args.max_inside_sample_count_ratio,
        max_inside_generation_seconds_ratio=args.max_inside_generation_seconds_ratio,
        min_selected=args.min_selected,
        min_decision_accuracy=args.min_decision_accuracy,
        max_false_supported_rate=args.max_false_supported_rate,
        min_false_refuted_rate=args.min_false_refuted_rate,
        max_verified_false_alarm=args.max_verified_false_alarm,
        min_verified_detection=args.min_verified_detection,
        max_mean_duration_seconds=args.max_mean_duration_seconds,
        max_p99_duration_seconds=args.max_p99_duration_seconds,
        max_max_duration_seconds=args.max_max_duration_seconds,
        max_mean_attempted_route_count=args.max_mean_attempted_route_count,
        max_retrieval_use_rate=args.max_retrieval_use_rate,
        max_runtime_total_seconds=args.max_runtime_total_seconds,
        max_retrieval_hit_count=args.max_retrieval_hit_count,
        min_claims_cache_hit_rate=args.min_claims_cache_hit_rate,
        min_verifier_trace_cache_hit_rate=args.min_verifier_trace_cache_hit_rate,
        min_covered_fact_properties=args.min_covered_fact_properties,
        min_covered_fact_property_records=args.min_covered_fact_property_records,
        min_covered_fact_property_source_documents=args.min_covered_fact_property_source_documents,
        min_covered_fact_property_decision_accuracy=args.min_covered_fact_property_decision_accuracy,
        max_covered_fact_property_false_supported_rate=args.max_covered_fact_property_false_supported_rate,
        min_covered_fact_property_false_refuted_rate=args.min_covered_fact_property_false_refuted_rate,
        require_non_oracle_evidence=bool(args.require_non_oracle_evidence),
        require_retrieval_provenance_filter=bool(args.require_retrieval_provenance_filter),
        required_retrieval_source_prefixes=_parse_csv(args.required_retrieval_source_prefix),
        required_retrieval_metadata=_parse_key_values(args.required_retrieval_metadata),
        min_retrieval_filter_score=args.min_retrieval_filter_score,
        require_retrieval_stress_control=bool(args.require_retrieval_stress_control),
        retrieval_stress_manifest=args.retrieval_stress_manifest,
        min_stress_false_supported_rate=args.min_stress_false_supported_rate,
        max_stress_false_refuted_rate=args.max_stress_false_refuted_rate,
        required_route_min_selected=args.required_route_min_selected,
        required_route_min_decision_accuracy=args.required_route_min_decision_accuracy,
        required_route_max_false_supported_rate=args.required_route_max_false_supported_rate,
        required_route_min_false_refuted_rate=args.required_route_min_false_refuted_rate,
        required_route_max_verified_false_alarm=args.required_route_max_verified_false_alarm,
        required_route_min_verified_detection=args.required_route_min_verified_detection,
        required_route_max_mean_duration_seconds=args.required_route_max_mean_duration_seconds,
        required_route_max_p99_duration_seconds=args.required_route_max_p99_duration_seconds,
        required_route_max_max_duration_seconds=args.required_route_max_max_duration_seconds,
        required_route_max_mean_attempted_route_count=args.required_route_max_mean_attempted_route_count,
        required_route_max_retrieval_use_rate=args.required_route_max_retrieval_use_rate,
        required_route_max_runtime_total_seconds=args.required_route_max_runtime_total_seconds,
        required_route_max_retrieval_hit_count=args.required_route_max_retrieval_hit_count,
        required_route_min_claims_cache_hit_rate=args.required_route_min_claims_cache_hit_rate,
        required_route_min_verifier_trace_cache_hit_rate=args.required_route_min_verifier_trace_cache_hit_rate,
        required_route_min_covered_fact_properties=args.required_route_min_covered_fact_properties,
        required_route_min_covered_fact_property_records=args.required_route_min_covered_fact_property_records,
        required_route_min_covered_fact_property_source_documents=(
            args.required_route_min_covered_fact_property_source_documents
        ),
        required_route_min_covered_fact_property_decision_accuracy=(
            args.required_route_min_covered_fact_property_decision_accuracy
        ),
        required_route_max_covered_fact_property_false_supported_rate=(
            args.required_route_max_covered_fact_property_false_supported_rate
        ),
        required_route_min_covered_fact_property_false_refuted_rate=(
            args.required_route_min_covered_fact_property_false_refuted_rate
        ),
        required_route_require_non_oracle_evidence=bool(args.required_route_require_non_oracle_evidence),
        required_route_require_retrieval_provenance_filter=bool(
            args.required_route_require_retrieval_provenance_filter
        ),
        required_route_required_retrieval_source_prefixes=_parse_csv(
            args.required_route_required_retrieval_source_prefix
        ),
        required_route_required_retrieval_metadata=_parse_key_values(
            args.required_route_required_retrieval_metadata
        ),
        required_route_min_retrieval_filter_score=args.required_route_min_retrieval_filter_score,
        required_route_require_retrieval_stress_control=bool(
            args.required_route_require_retrieval_stress_control
        ),
        required_route_retrieval_stress_manifest=args.required_route_retrieval_stress_manifest,
        required_route_min_stress_false_supported_rate=args.required_route_min_stress_false_supported_rate,
        required_route_max_stress_false_refuted_rate=args.required_route_max_stress_false_refuted_rate,
        notes=args.note,
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote release candidate comparison to {output_path}")
    decision = payload["decision"]
    print(
        "release_candidate_comparison="
        f"{decision['status']} readiness={decision.get('recommended_readiness_record')} "
        f"route={decision.get('recommended_route_record')} "
        f"performance={decision.get('recommended_performance_baseline_record')} "
        f"selector_replay={decision.get('recommended_selector_replay_candidate')} "
        f"product_runtime_drift={decision.get('product_runtime_drift_status')} "
        f"release_efficiency={decision.get('recommended_release_efficiency_profile')} "
        f"external_evidence={decision.get('external_evidence_baseline_comparison_status')} "
        f"pre_generation_probe={decision.get('pre_generation_probe_comparison_status')} "
        f"claim_factuality_probe={decision.get('claim_factuality_probe_comparison_status')} "
        f"frontier_release_evidence={decision.get('frontier_release_evidence_status')} "
        f"world_model_signal={decision.get('world_model_signal_workflow_status')} "
        f"context_sensitivity={decision.get('context_sensitivity_workflow_status')} "
        f"pathway_intervention={decision.get('pathway_intervention_workflow_status')} "
        f"selfcheck_signal_fusion={decision.get('selfcheck_signal_fusion_workflow_status')} "
        f"uncertainty_escalation={decision.get('uncertainty_escalation_workflow_status')} "
        f"triple_extraction_matrix={decision.get('triple_extraction_fixture_matrix_status')} "
        f"counterfactual_verification={decision.get('counterfactual_verification_status')}"
    )
    if args.fail_on_blocked and decision["status"] != "promote":
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare readiness and verifier-route baselines as one release candidate"
    )
    parser.add_argument("--readiness-registry", required=True,
                        help="ArtifactRegistry JSON path containing readiness baselines")
    parser.add_argument("--route-registry", default=None,
                        help="ArtifactRegistry JSON path containing route baselines; defaults to readiness registry")
    parser.add_argument("--performance-registry", default=None,
                        help="ArtifactRegistry JSON path containing performance_baseline records; defaults to "
                             "readiness registry")
    parser.add_argument("--readiness-baseline-key", action="append", default=[],
                        help="readiness benchmark_manifest registry key to compare; repeatable")
    parser.add_argument("--route-baseline-key", action="append", default=[],
                        help="route benchmark_manifest registry key to compare; repeatable")
    parser.add_argument("--required-route-baseline-key", action="append", default=[],
                        help="additional promoted route benchmark_manifest key that must verify without "
                             "becoming the selected product route; repeatable")
    parser.add_argument("--release-policy-profile", default=None,
                        choices=RELEASE_POLICY_PROFILE_NAMES,
                        help="optional named release gate defaults; explicit CLI thresholds override profile values")
    parser.add_argument("--require-structured-fact-robustness", action="store_true",
                        help="require both canonical and paraphrase structured_fact covered-facts route "
                             "baselines as additional release evidence")
    parser.add_argument("--structured-fact-canonical-route-key", default=None,
                        help="benchmark_manifest:<name>:<version> key for the canonical structured_fact route")
    parser.add_argument("--structured-fact-paraphrase-route-key", default=None,
                        help="benchmark_manifest:<name>:<version> key for the paraphrase robustness "
                             "structured_fact route")
    parser.add_argument("--performance-baseline-key", default=None,
                        help="optional performance_baseline registry key that must match the selected runtime")
    parser.add_argument("--performance-drift-baseline-key", default=None,
                        help="optional reference performance_baseline key for trend regression gates")
    parser.add_argument("--selector-replay-report", default=None,
                        help="optional runtime-profile selector replay report that must promote and verify")
    parser.add_argument("--product-runtime-drift-report", default=None,
                        help="optional product runtime drift report that must promote and verify")
    parser.add_argument("--require-product-runtime-drift-promotion-evidence", action="store_true",
                        help="require the product runtime drift report to include promotion-contract "
                             "coverage and triple-extraction fixture-matrix quality metrics")
    parser.add_argument("--require-product-runtime-drift-pre-generation-evidence", action="store_true",
                        help="require the product runtime drift report to include pre-generation "
                             "probe comparison coverage, redline, and quality metrics")
    parser.add_argument("--require-product-runtime-drift-claim-factuality-evidence", action="store_true",
                        help="require the product runtime drift report to include claim factuality "
                             "probe comparison coverage, conformal/selective, redline, and quality metrics")
    parser.add_argument("--require-product-runtime-drift-claim-risk-localization-evidence",
                        action="store_true",
                        help="require the product runtime drift report to include claim-risk localization "
                             "coverage, claim counts, and entity-candidate risk drift metrics")
    parser.add_argument("--require-product-runtime-drift-counterfactual-evidence", action="store_true",
                        help="require the product runtime drift report to include counterfactual "
                             "verifier-audit coverage, manifest, pass-rate, false-invariance, "
                             "and flip-success metrics")
    parser.add_argument("--require-product-runtime-drift-fact-selfcheck-gate-evidence",
                        action="store_true",
                        help="require the product runtime drift report to include fact-selfcheck "
                             "signal-fusion gate coverage, manifest, pass, run, and "
                             "triple-count metrics")
    parser.add_argument("--require-product-runtime-drift-triple-audit-evidence", action="store_true",
                        help="require the product runtime drift report to include trace-level "
                             "triple coverage, audit coverage, audit pass-rate, and slot coverage metrics")
    parser.add_argument("--require-product-runtime-drift-covered-fact-property-evidence", action="store_true",
                        help="require the product runtime drift report to include covered-fact property metrics")
    parser.add_argument("--require-product-runtime-drift-action-gate-evidence", action="store_true",
                        help="require the product runtime drift report to include product-trace "
                             "action-audit and action-execution drift metrics")
    parser.add_argument("--require-product-runtime-drift-world-model-action-gate-evidence",
                        action="store_true",
                        help="require the product runtime drift report to include world-model "
                             "guarded-action coverage, pass/block, and failure-reason metrics")
    parser.add_argument("--require-product-runtime-drift-world-model-rollout-evidence",
                        action="store_true",
                        help="require the product runtime drift report to include post-action "
                             "world-model rollout coverage, sync, drift, trace-gap, and "
                             "path-mismatch metrics")
    parser.add_argument("--require-product-runtime-drift-action-receipts-evidence", action="store_true",
                        help="require the product runtime drift report to include product-trace "
                             "action receipt coverage, validity, signature, and fingerprint metrics")
    parser.add_argument("--require-product-runtime-drift-receipt-claim-support-evidence",
                        action="store_true",
                        help="require the product runtime drift report to include receipt-backed "
                             "claim-support reference quality metrics")
    parser.add_argument("--require-product-runtime-drift-trajectory-audit-evidence", action="store_true",
                        help="require the product runtime drift report to include trajectory-audit "
                             "failed-trace/error and hallucination-taxonomy drift metrics")
    parser.add_argument("--require-product-runtime-drift-provenance-evidence", action="store_true",
                        help="require the product runtime drift report to include trace-provenance "
                             "coverage, support-reference, and missing-evidence drift metrics")
    parser.add_argument("--require-product-runtime-drift-citation-integrity-evidence",
                        action="store_true",
                        help="require the product runtime drift report to include citation-integrity "
                             "participation, coverage, mismatch, unresolved, issue, and trace-gap metrics")
    parser.add_argument("--require-product-runtime-drift-evidence-quality-evidence",
                        action="store_true",
                        help="require the product runtime drift report to include retrieval evidence-quality "
                             "coverage, pass/failure, source, freshness, and timestamp metrics")
    parser.add_argument("--require-product-runtime-drift-metacognition-evidence",
                        action="store_true",
                        help="require the product runtime drift report to include metacognition "
                             "coverage, pass-rate, overconfidence, and miscalibration metrics")
    parser.add_argument("--require-product-runtime-drift-evidence-handoff-evidence", action="store_true",
                        help="require the product runtime drift report to include promotion-contract "
                             "evidence handoff coverage, manifest, metric completeness, and promoted-group "
                             "drift metrics")
    parser.add_argument("--require-product-runtime-drift-world-model-evidence", action="store_true",
                        help="require the product runtime drift report to include trace-level world-model "
                             "participation, coverage, conflict, low-agreement, and trace-gap metrics")
    parser.add_argument("--require-product-runtime-drift-context-sensitivity-evidence", action="store_true",
                        help="require the product runtime drift report to include trace-level "
                             "context-sensitivity participation, coverage, flagged, trace-gap, "
                             "and ratio metrics")
    parser.add_argument("--require-product-runtime-drift-evidence-alignment-evidence", action="store_true",
                        help="require the product runtime drift report to include trace-level "
                             "claim/evidence alignment participation, coverage, alignment, "
                             "misalignment, citation-coverage, issue, and trace-gap metrics")
    parser.add_argument("--require-product-runtime-drift-counterfactual-robustness-evidence", action="store_true",
                        help="require the product runtime drift report to include trace-level "
                             "counterfactual robustness participation, coverage, pass, "
                             "false-invariance, flip-success, and trace-gap metrics")
    parser.add_argument("--require-product-runtime-drift-frontier-release-evidence", action="store_true",
                        help="require the product runtime drift report to include frontier release "
                             "evidence coverage, artifact presence, promote-rate, and run-count metrics")
    parser.add_argument("--release-efficiency-report", default=None,
                        help="optional release efficiency report that must promote and verify")
    parser.add_argument("--external-evidence-baseline-comparison", default=None,
                        help="optional compare_external_evidence_baselines.py report that must promote")
    parser.add_argument("--external-evidence-baseline-comparison-registry", default=None,
                        help="optional ArtifactRegistry JSON path for "
                             "--external-evidence-baseline-comparison-key; defaults to "
                             "--readiness-registry")
    parser.add_argument("--external-evidence-baseline-comparison-key", default=None,
                        help="optional report:<name>:<version> registry key for an external "
                             "evidence baseline comparison")
    parser.add_argument("--pre-generation-probe-comparison", default=None,
                        help="optional compare_pre_generation_probe_workflows.py report that must "
                             "pass multi-model and text-redline release gates")
    parser.add_argument("--pre-generation-probe-comparison-registry", default=None,
                        help="optional ArtifactRegistry JSON path for "
                             "--pre-generation-probe-comparison-key; defaults to --readiness-registry")
    parser.add_argument("--pre-generation-probe-comparison-key", default=None,
                        help="optional report:<name>:<version> registry key for a pre-generation "
                             "probe workflow comparison")
    parser.add_argument("--claim-factuality-probe-comparison", default=None,
                        help="optional compare_claim_factuality_probe_workflows.py report that must "
                             "pass multi-model, conformal, and text-redline release gates")
    parser.add_argument("--claim-factuality-probe-comparison-registry", default=None,
                        help="optional ArtifactRegistry JSON path for "
                             "--claim-factuality-probe-comparison-key; defaults to "
                             "--readiness-registry")
    parser.add_argument("--claim-factuality-probe-comparison-key", default=None,
                        help="optional report:<name>:<version> registry key for a claim factuality "
                             "probe workflow comparison")
    parser.add_argument("--frontier-release-evidence", default=None,
                        help="optional frontier release-evidence report that must promote and verify")
    parser.add_argument("--frontier-release-evidence-registry", default=None,
                        help="optional ArtifactRegistry JSON path for --frontier-release-evidence-key; "
                             "defaults to --readiness-registry")
    parser.add_argument("--frontier-release-evidence-key", default=None,
                        help="optional report:<name>:<version> registry key for frontier release evidence")
    parser.add_argument("--require-frontier-release-input-manifests", action="store_true",
                        help="require the frontier release-evidence report to prove all input "
                             "artifact manifests were required and verified")
    parser.add_argument("--world-model-signal-workflow", default=None,
                        help="optional world-model signal calibration workflow report that must pass its "
                             "conflict/trace-gap release gate")
    parser.add_argument("--world-model-signal-workflow-registry", default=None,
                        help="optional ArtifactRegistry JSON path for --world-model-signal-workflow-key; "
                             "defaults to --readiness-registry")
    parser.add_argument("--world-model-signal-workflow-key", default=None,
                        help="optional report:<name>:<version> registry key for a world-model signal workflow")
    parser.add_argument("--context-sensitivity-workflow", default=None,
                        help="optional context-sensitivity workflow report that must verify and contain "
                             "paired/enriched/enhanced score evidence")
    parser.add_argument("--context-sensitivity-workflow-registry", default=None,
                        help="optional ArtifactRegistry JSON path for --context-sensitivity-workflow-key; "
                             "defaults to --readiness-registry")
    parser.add_argument("--context-sensitivity-workflow-key", default=None,
                        help="optional report:<name>:<version> registry key for a context-sensitivity workflow")
    parser.add_argument("--mechanism-handoff-evidence-bundle", default=None,
                        help="optional mechanism handoff evidence bundle report that must promote "
                             "and verify")
    parser.add_argument("--mechanism-handoff-evidence-bundle-registry", default=None,
                        help="optional ArtifactRegistry JSON path for "
                             "--mechanism-handoff-evidence-bundle-key; defaults to "
                             "--readiness-registry")
    parser.add_argument("--mechanism-handoff-evidence-bundle-key", default=None,
                        help="optional report:<name>:<version> registry key for a mechanism "
                             "handoff evidence bundle")
    parser.add_argument("--pathway-intervention-workflow", default=None,
                        help="optional pathway intervention workflow report that must be release-ready "
                             "and manifest-verified")
    parser.add_argument("--pathway-intervention-workflow-registry", default=None,
                        help="optional ArtifactRegistry JSON path for --pathway-intervention-workflow-key; "
                             "defaults to --readiness-registry")
    parser.add_argument("--pathway-intervention-workflow-key", default=None,
                        help="optional report:<name>:<version> registry key for a pathway intervention workflow")
    parser.add_argument("--product-trace-replay-workflow", default=None,
                        help="optional product trace replay workflow report; when supplied, its selector "
                             "replay and runtime-drift child reports are used unless explicit child report "
                             "paths are provided")
    parser.add_argument("--product-trace-replay-workflow-registry", default=None,
                        help="optional ArtifactRegistry JSON path for --product-trace-replay-workflow-key; "
                             "defaults to --readiness-registry")
    parser.add_argument("--product-trace-replay-workflow-key", default=None,
                        help="optional report:<name>:<version> registry key for a product trace replay workflow")
    parser.add_argument("--require-product-trace-action-audit-gate", action="store_true",
                        help="require the supplied product trace replay workflow to have enabled and promoted "
                             "its action-audit gate")
    parser.add_argument("--require-product-trace-action-execution-gate", action="store_true",
                        help="require the supplied product trace replay workflow to have enabled and promoted "
                             "its action-execution alignment gate")
    parser.add_argument("--selfcheck-signal-fusion-workflow", default=None,
                        help="optional selfcheck signal fusion workflow report that must pass sample-quality "
                             "and manifest gates")
    parser.add_argument("--selfcheck-signal-fusion-workflow-registry", default=None,
                        help="optional ArtifactRegistry JSON path for "
                             "--selfcheck-signal-fusion-workflow-key; defaults to --readiness-registry")
    parser.add_argument("--selfcheck-signal-fusion-workflow-key", default=None,
                        help="optional report:<name>:<version> registry key for a selfcheck signal fusion workflow")
    parser.add_argument("--uncertainty-escalation-workflow", default=None,
                        help="optional uncertainty escalation workflow report that must verify and meet "
                             "configured escalation/evidence quality thresholds")
    parser.add_argument("--uncertainty-escalation-workflow-registry", default=None,
                        help="optional ArtifactRegistry JSON path for "
                             "--uncertainty-escalation-workflow-key; defaults to --readiness-registry")
    parser.add_argument("--uncertainty-escalation-workflow-key", default=None,
                        help="optional report:<name>:<version> registry key for an uncertainty "
                             "escalation workflow")
    parser.add_argument("--min-uncertainty-escalation-records",
                        type=lambda value: _parse_non_negative_int(
                            value,
                            flag="--min-uncertainty-escalation-records",
                        ), default=None,
                        help="optional minimum record count; defaults to 1 when workflow evidence is supplied")
    parser.add_argument("--min-uncertainty-escalation-trigger-rate",
                        type=lambda value: _parse_unit_float(
                            value,
                            flag="--min-uncertainty-escalation-trigger-rate",
                        ), default=None,
                        help="optional minimum uncertainty escalation trigger rate")
    parser.add_argument("--min-uncertainty-escalation-retrieval-evidence-rate",
                        type=lambda value: _parse_unit_float(
                            value,
                            flag="--min-uncertainty-escalation-retrieval-evidence-rate",
                        ), default=None,
                        help="optional minimum retrieval evidence rate after escalation")
    parser.add_argument("--max-uncertainty-escalation-final-false-accept-rate",
                        type=lambda value: _parse_unit_float(
                            value,
                            flag="--max-uncertainty-escalation-final-false-accept-rate",
                        ), default=None,
                        help="optional maximum final false-accept rate after escalation")
    parser.add_argument("--max-uncertainty-escalation-false-accept-delta",
                        type=lambda value: _parse_finite_float(
                            value,
                            flag="--max-uncertainty-escalation-false-accept-delta",
                        ), default=None,
                        help="optional maximum false-accept-rate delta after escalation; negative values "
                             "require improvement")
    parser.add_argument("--feedback-policy-workflow", default=None,
                        help="optional feedback-policy workflow report that must recommend/observe and verify")
    parser.add_argument("--feedback-policy-workflow-registry", default=None,
                        help="optional ArtifactRegistry JSON path for --feedback-policy-workflow-key; "
                             "defaults to --readiness-registry")
    parser.add_argument("--feedback-policy-workflow-key", default=None,
                        help="optional report:<name>:<version> registry key for a feedback-policy workflow")
    parser.add_argument("--feedback-policy-min-matched-feedback-count", type=lambda value: _parse_non_negative_int(
        value,
        flag="--feedback-policy-min-matched-feedback-count",
    ), default=None,
                        help="optional minimum matched feedback count for the feedback-policy workflow gate")
    parser.add_argument("--feedback-policy-min-safety-coverage", type=lambda value: _parse_unit_float(
        value,
        flag="--feedback-policy-min-safety-coverage",
    ), default=None,
                        help="optional minimum feedback replay safety coverage for the feedback-policy workflow gate")
    parser.add_argument("--feedback-policy-max-unknown-safety-issue-rate", type=lambda value: _parse_unit_float(
        value,
        flag="--feedback-policy-max-unknown-safety-issue-rate",
    ), default=None,
                        help="optional maximum unknown safety issue rate for the feedback-policy workflow gate")
    parser.add_argument("--adapter-family-matrix", default=None,
                        help="optional adapter-family matrix JSON report that must promote before release")
    parser.add_argument("--adapter-family-profile", default=None,
                        choices=ADAPTER_FAMILY_PROFILE_NAMES,
                        help="optional adapter-family route profile; strict_audit requires structured_state, "
                             "state_transition, triple_evidence, and rule-based state-transition "
                             "world-model evidence")
    parser.add_argument("--required-adapter-route", action="append", default=[],
                        help="route that must be present and promoted in --adapter-family-matrix; repeatable")
    parser.add_argument("--require-state-transition-world-model", action="store_true",
                        help="require adapter-family state_transition evidence to use RuleBasedWorldModelAdapter "
                             "with at least one rule; enabled automatically by strict_audit")
    parser.add_argument("--triple-extraction-fixture-matrix", default=None,
                        help="optional triple-extraction fixture matrix report that must promote and verify")
    parser.add_argument("--triple-extraction-fixture-matrix-registry", default=None,
                        help="optional ArtifactRegistry JSON path for "
                             "--triple-extraction-fixture-matrix-key; defaults to --readiness-registry")
    parser.add_argument("--triple-extraction-fixture-matrix-key", default=None,
                        help="optional report:<name>:<version> registry key for a triple-extraction matrix")
    parser.add_argument("--min-triple-extraction-corpora", type=lambda value: _parse_non_negative_int(
        value,
        flag="--min-triple-extraction-corpora",
    ), default=None,
                        help="optional minimum corpus count and promoted corpus count for the "
                             "triple-extraction fixture matrix")
    parser.add_argument("--min-triple-extraction-distinct-predicates",
                        type=lambda value: _parse_non_negative_int(
                            value,
                            flag="--min-triple-extraction-distinct-predicates",
                        ), default=None,
                        help="optional minimum distinct predicate count for the triple-extraction fixture matrix")
    parser.add_argument("--min-triple-extraction-external-prediction-count",
                        type=lambda value: _parse_non_negative_int(
                            value,
                            flag="--min-triple-extraction-external-prediction-count",
                        ), default=None,
                        help="optional minimum external prediction file count for the "
                             "triple-extraction fixture matrix")
    parser.add_argument("--min-triple-extraction-external-prediction-corpora",
                        type=lambda value: _parse_non_negative_int(
                            value,
                            flag="--min-triple-extraction-external-prediction-corpora",
                        ), default=None,
                        help="optional minimum number of corpora with external prediction files")
    parser.add_argument("--min-triple-extraction-mean-best-external-f1",
                        type=lambda value: _parse_unit_float(
                            value,
                            flag="--min-triple-extraction-mean-best-external-f1",
                        ), default=None,
                        help="optional minimum mean best external prediction F1 for the "
                             "triple-extraction fixture matrix")
    parser.add_argument("--counterfactual-verification-report", default=None,
                        help="optional counterfactual verifier audit report that must promote and verify")
    parser.add_argument("--counterfactual-verification-registry", default=None,
                        help="optional ArtifactRegistry JSON path for "
                             "--counterfactual-verification-key; defaults to --readiness-registry")
    parser.add_argument("--counterfactual-verification-key", default=None,
                        help="optional report:<name>:<version> registry key for counterfactual verifier audit")
    parser.add_argument("--min-counterfactual-verification-records",
                        type=lambda value: _parse_non_negative_int(
                            value,
                            flag="--min-counterfactual-verification-records",
                        ), default=None,
                        help="optional minimum counterfactual probe count; defaults to 1 when report is supplied")
    parser.add_argument("--min-counterfactual-verification-pass-rate",
                        type=lambda value: _parse_unit_float(
                            value,
                            flag="--min-counterfactual-verification-pass-rate",
                        ), default=None,
                        help="optional minimum audit pass rate; defaults to 1.0 when report is supplied")
    parser.add_argument("--max-counterfactual-verification-false-invariance-rate",
                        type=lambda value: _parse_unit_float(
                            value,
                            flag="--max-counterfactual-verification-false-invariance-rate",
                        ), default=None,
                        help="optional maximum false-invariance rate; defaults to 0.0 when report is supplied")
    parser.add_argument("--require-performance-score-dump-cache", action="store_true",
                        help="require the selected performance baseline to include score-dump cache evidence")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--note", action="append", default=[],
                        help="optional note to include in the comparison report; repeatable")
    parser.add_argument("--no-recursive", action="store_true", help="only verify root manifests")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="allow unverified manifests to become candidates")
    parser.add_argument("--manifest-fingerprint-workers", type=lambda value: _parse_positive_int(
        value,
        flag="--manifest-fingerprint-workers",
    ), default=1,
                        help="maximum worker threads for recursive artifact-manifest fingerprinting")
    parser.add_argument("--runtime-profile", default=None, choices=RUNTIME_PROFILE_NAMES,
                        help="optional release profile that fills unset runtime/cost gates; explicit flags "
                             "override profile defaults")
    parser.add_argument("--inside-trigger-budget-policy", default=None,
                        choices=INSIDE_TRIGGER_BUDGET_POLICIES,
                        help="optional release-time override for trigger-budget sweep selection; omit to use "
                             "the readiness baseline policy")
    parser.add_argument("--min-best-quality-auroc", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-best-quality-auroc",
    ), default=None)
    parser.add_argument("--max-uncached-forward-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-uncached-forward-seconds",
    ), default=None)
    parser.add_argument("--max-cache-only-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-cache-only-seconds",
    ), default=None)
    parser.add_argument("--max-recommended-runtime-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-recommended-runtime-seconds",
    ), default=None,
                        help="maximum selected deployment-path runtime cost from the readiness recommendation")
    parser.add_argument("--max-covariance-maha-last-auroc-drop", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-covariance-maha-last-auroc-drop",
    ), default=None,
                        help="max allowed selected covariance maha_last AUROC drop versus the full-covariance "
                             "baseline; readiness/performance candidates without covariance tradeoff data fail "
                             "closed when set")
    parser.add_argument("--max-inside-sample-count-ratio", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-inside-sample-count-ratio",
    ), default=None)
    parser.add_argument("--max-inside-generation-seconds-ratio", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-inside-generation-seconds-ratio",
    ), default=None)
    parser.add_argument("--min-selected", type=lambda value: _parse_non_negative_int(
        value,
        flag="--min-selected",
    ), default=None)
    parser.add_argument("--min-decision-accuracy", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-decision-accuracy",
    ), default=None)
    parser.add_argument("--max-false-supported-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-false-supported-rate",
    ), default=None)
    parser.add_argument("--min-false-refuted-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-false-refuted-rate",
    ), default=None)
    parser.add_argument("--max-verified-false-alarm", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-verified-false-alarm",
    ), default=None)
    parser.add_argument("--min-verified-detection", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-verified-detection",
    ), default=None)
    parser.add_argument("--max-mean-duration-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-mean-duration-seconds",
    ), default=None)
    parser.add_argument("--max-p99-duration-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-p99-duration-seconds",
    ), default=None)
    parser.add_argument("--max-max-duration-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-max-duration-seconds",
    ), default=None)
    parser.add_argument("--max-mean-attempted-route-count", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-mean-attempted-route-count",
    ), default=None)
    parser.add_argument("--max-retrieval-use-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-retrieval-use-rate",
    ), default=None)
    parser.add_argument("--max-runtime-total-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-runtime-total-seconds",
    ), default=None)
    parser.add_argument("--max-retrieval-hit-count", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-retrieval-hit-count",
    ), default=None)
    parser.add_argument("--min-claims-cache-hit-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-claims-cache-hit-rate",
    ), default=None)
    parser.add_argument("--min-verifier-trace-cache-hit-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-verifier-trace-cache-hit-rate",
    ), default=None)
    parser.add_argument("--min-covered-fact-properties", type=lambda value: _parse_non_negative_int(
        value,
        flag="--min-covered-fact-properties",
    ), default=None)
    parser.add_argument("--min-covered-fact-property-records", type=lambda value: _parse_non_negative_int(
        value,
        flag="--min-covered-fact-property-records",
    ), default=None)
    parser.add_argument("--min-covered-fact-property-source-documents", type=lambda value: _parse_non_negative_int(
        value,
        flag="--min-covered-fact-property-source-documents",
    ), default=None)
    parser.add_argument("--min-covered-fact-property-decision-accuracy", type=lambda value: _parse_unit_float(
        value,
        flag="--min-covered-fact-property-decision-accuracy",
    ), default=None)
    parser.add_argument("--max-covered-fact-property-false-supported-rate", type=lambda value: _parse_unit_float(
        value,
        flag="--max-covered-fact-property-false-supported-rate",
    ), default=None)
    parser.add_argument("--min-covered-fact-property-false-refuted-rate", type=lambda value: _parse_unit_float(
        value,
        flag="--min-covered-fact-property-false-refuted-rate",
    ), default=None)
    parser.add_argument(
        "--require-non-oracle-evidence",
        action="store_true",
        help="require selected route claims to omit labels and include input provenance",
    )
    parser.add_argument(
        "--require-retrieval-provenance-filter",
        action="store_true",
        help="require selected route evidence to record a source-requiring retrieval provenance filter",
    )
    parser.add_argument(
        "--required-retrieval-source-prefix",
        action="append",
        default=None,
        help="source prefix that must appear in the selected route provenance-filter allow-list; "
             "comma-separated or repeatable",
    )
    parser.add_argument(
        "--required-retrieval-metadata",
        action="append",
        default=None,
        help="required selected-route provenance-filter metadata key=value; comma-separated or repeatable",
    )
    parser.add_argument(
        "--min-retrieval-filter-score",
        type=lambda value: _parse_non_negative_float(
            value,
            flag="--min-retrieval-filter-score",
        ),
        default=None,
        help="minimum min_score required in the selected route retrieval provenance filter",
    )
    parser.add_argument(
        "--require-retrieval-stress-control",
        action="store_true",
        help="require an answer-echo retrieval stress control for selected route baselines",
    )
    parser.add_argument(
        "--retrieval-stress-manifest",
        default=None,
        help="optional answer-echo retrieval stress artifact manifest for selected route baselines",
    )
    parser.add_argument(
        "--min-stress-false-supported-rate",
        type=lambda value: _parse_unit_float(
            value,
            flag="--min-stress-false-supported-rate",
        ),
        default=None,
    )
    parser.add_argument(
        "--max-stress-false-refuted-rate",
        type=lambda value: _parse_unit_float(
            value,
            flag="--max-stress-false-refuted-rate",
        ),
        default=None,
    )
    parser.add_argument(
        "--min-performance-score-dump-cache-jsonl-view-hit-rate",
        type=lambda value: _parse_unit_float(
            value,
            flag="--min-performance-score-dump-cache-jsonl-view-hit-rate",
        ),
        default=None,
        help="minimum selected JSONL score-dump cache hit rate required from the performance baseline",
    )
    parser.add_argument(
        "--max-performance-uncached-total-seconds-ratio",
        type=lambda value: _parse_non_negative_float(
            value,
            flag="--max-performance-uncached-total-seconds-ratio",
        ),
        default=None,
        help="maximum selected performance uncached_total_seconds ratio versus drift baseline",
    )
    parser.add_argument(
        "--max-performance-cached-total-seconds-ratio",
        type=lambda value: _parse_non_negative_float(
            value,
            flag="--max-performance-cached-total-seconds-ratio",
        ),
        default=None,
        help="maximum selected performance cached_total_seconds ratio versus drift baseline",
    )
    parser.add_argument(
        "--max-performance-cache-only-total-seconds-ratio",
        type=lambda value: _parse_non_negative_float(
            value,
            flag="--max-performance-cache-only-total-seconds-ratio",
        ),
        default=None,
        help="maximum selected performance cache_only_total_seconds ratio versus drift baseline",
    )
    parser.add_argument(
        "--max-performance-score-dump-cache-jsonl-view-hit-rate-drop",
        type=lambda value: _parse_unit_float(
            value,
            flag="--max-performance-score-dump-cache-jsonl-view-hit-rate-drop",
        ),
        default=None,
        help="maximum allowed JSONL score-dump cache hit-rate drop versus drift baseline",
    )
    parser.add_argument("--required-route-min-selected", type=lambda value: _parse_non_negative_int(
        value,
        flag="--required-route-min-selected",
    ), default=None)
    parser.add_argument("--required-route-min-decision-accuracy", type=lambda value: _parse_non_negative_float(
        value,
        flag="--required-route-min-decision-accuracy",
    ), default=None)
    parser.add_argument("--required-route-max-false-supported-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--required-route-max-false-supported-rate",
    ), default=None)
    parser.add_argument("--required-route-min-false-refuted-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--required-route-min-false-refuted-rate",
    ), default=None)
    parser.add_argument("--required-route-max-verified-false-alarm", type=lambda value: _parse_non_negative_float(
        value,
        flag="--required-route-max-verified-false-alarm",
    ), default=None)
    parser.add_argument("--required-route-min-verified-detection", type=lambda value: _parse_non_negative_float(
        value,
        flag="--required-route-min-verified-detection",
    ), default=None)
    parser.add_argument("--required-route-max-mean-duration-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--required-route-max-mean-duration-seconds",
    ), default=None)
    parser.add_argument("--required-route-max-p99-duration-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--required-route-max-p99-duration-seconds",
    ), default=None)
    parser.add_argument("--required-route-max-max-duration-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--required-route-max-max-duration-seconds",
    ), default=None)
    parser.add_argument("--required-route-max-mean-attempted-route-count", type=lambda value: _parse_non_negative_float(
        value,
        flag="--required-route-max-mean-attempted-route-count",
    ), default=None)
    parser.add_argument("--required-route-max-retrieval-use-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--required-route-max-retrieval-use-rate",
    ), default=None)
    parser.add_argument("--required-route-max-runtime-total-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--required-route-max-runtime-total-seconds",
    ), default=None)
    parser.add_argument("--required-route-max-retrieval-hit-count", type=lambda value: _parse_non_negative_float(
        value,
        flag="--required-route-max-retrieval-hit-count",
    ), default=None)
    parser.add_argument("--required-route-min-claims-cache-hit-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--required-route-min-claims-cache-hit-rate",
    ), default=None)
    parser.add_argument(
        "--required-route-min-verifier-trace-cache-hit-rate",
        type=lambda value: _parse_non_negative_float(
            value,
            flag="--required-route-min-verifier-trace-cache-hit-rate",
        ),
        default=None,
    )
    parser.add_argument("--required-route-min-covered-fact-properties", type=lambda value: _parse_non_negative_int(
        value,
        flag="--required-route-min-covered-fact-properties",
    ), default=None)
    parser.add_argument("--required-route-min-covered-fact-property-records", type=lambda value: (
        _parse_non_negative_int(
            value,
            flag="--required-route-min-covered-fact-property-records",
        )
    ), default=None)
    parser.add_argument("--required-route-min-covered-fact-property-source-documents", type=lambda value: (
        _parse_non_negative_int(
            value,
            flag="--required-route-min-covered-fact-property-source-documents",
        )
    ), default=None)
    parser.add_argument("--required-route-min-covered-fact-property-decision-accuracy", type=lambda value: (
        _parse_unit_float(
            value,
            flag="--required-route-min-covered-fact-property-decision-accuracy",
        )
    ), default=None)
    parser.add_argument("--required-route-max-covered-fact-property-false-supported-rate", type=lambda value: (
        _parse_unit_float(
            value,
            flag="--required-route-max-covered-fact-property-false-supported-rate",
        )
    ), default=None)
    parser.add_argument("--required-route-min-covered-fact-property-false-refuted-rate", type=lambda value: (
        _parse_unit_float(
            value,
            flag="--required-route-min-covered-fact-property-false-refuted-rate",
        )
    ), default=None)
    parser.add_argument("--structured-fact-robustness-min-selected", type=lambda value: _parse_non_negative_int(
        value,
        flag="--structured-fact-robustness-min-selected",
    ), default=None)
    parser.add_argument("--structured-fact-robustness-min-decision-accuracy", type=lambda value: (
        _parse_non_negative_float(
            value,
            flag="--structured-fact-robustness-min-decision-accuracy",
        )
    ), default=None)
    parser.add_argument("--structured-fact-robustness-max-false-supported-rate", type=lambda value: (
        _parse_non_negative_float(
            value,
            flag="--structured-fact-robustness-max-false-supported-rate",
        )
    ), default=None)
    parser.add_argument("--structured-fact-robustness-min-false-refuted-rate", type=lambda value: (
        _parse_non_negative_float(
            value,
            flag="--structured-fact-robustness-min-false-refuted-rate",
        )
    ), default=None)
    parser.add_argument("--structured-fact-robustness-min-covered-fact-properties", type=lambda value: (
        _parse_non_negative_int(
            value,
            flag="--structured-fact-robustness-min-covered-fact-properties",
        )
    ), default=None)
    parser.add_argument("--structured-fact-robustness-min-covered-fact-property-records", type=lambda value: (
        _parse_non_negative_int(
            value,
            flag="--structured-fact-robustness-min-covered-fact-property-records",
        )
    ), default=None)
    parser.add_argument(
        "--structured-fact-robustness-min-covered-fact-property-source-documents",
        type=lambda value: _parse_non_negative_int(
            value,
            flag="--structured-fact-robustness-min-covered-fact-property-source-documents",
        ),
        default=None,
    )
    parser.add_argument(
        "--structured-fact-robustness-min-covered-fact-property-decision-accuracy",
        type=lambda value: _parse_unit_float(
            value,
            flag="--structured-fact-robustness-min-covered-fact-property-decision-accuracy",
        ),
        default=None,
    )
    parser.add_argument(
        "--structured-fact-robustness-max-covered-fact-property-false-supported-rate",
        type=lambda value: _parse_unit_float(
            value,
            flag="--structured-fact-robustness-max-covered-fact-property-false-supported-rate",
        ),
        default=None,
    )
    parser.add_argument(
        "--structured-fact-robustness-min-covered-fact-property-false-refuted-rate",
        type=lambda value: _parse_unit_float(
            value,
            flag="--structured-fact-robustness-min-covered-fact-property-false-refuted-rate",
        ),
        default=None,
    )
    parser.add_argument(
        "--required-route-require-non-oracle-evidence",
        action="store_true",
        help="require required route claims to omit labels and include input provenance",
    )
    parser.add_argument(
        "--required-route-require-retrieval-provenance-filter",
        action="store_true",
        help="require required route evidence to record a source-requiring retrieval provenance filter",
    )
    parser.add_argument(
        "--required-route-required-retrieval-source-prefix",
        action="append",
        default=None,
        help="source prefix that must appear in the required route provenance-filter allow-list; "
             "comma-separated or repeatable",
    )
    parser.add_argument(
        "--required-route-required-retrieval-metadata",
        action="append",
        default=None,
        help="required required-route provenance-filter metadata key=value; comma-separated or repeatable",
    )
    parser.add_argument(
        "--required-route-min-retrieval-filter-score",
        type=lambda value: _parse_non_negative_float(
            value,
            flag="--required-route-min-retrieval-filter-score",
        ),
        default=None,
        help="minimum min_score required in the required route retrieval provenance filter",
    )
    parser.add_argument(
        "--required-route-require-retrieval-stress-control",
        action="store_true",
        help="require an answer-echo retrieval stress control for required route baselines",
    )
    parser.add_argument(
        "--required-route-retrieval-stress-manifest",
        default=None,
        help="optional answer-echo retrieval stress artifact manifest for required route baselines",
    )
    parser.add_argument(
        "--required-route-min-stress-false-supported-rate",
        type=lambda value: _parse_unit_float(
            value,
            flag="--required-route-min-stress-false-supported-rate",
        ),
        default=None,
    )
    parser.add_argument(
        "--required-route-max-stress-false-refuted-rate",
        type=lambda value: _parse_unit_float(
            value,
            flag="--required-route-max-stress-false-refuted-rate",
        ),
        default=None,
    )
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless the release candidate promotes")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
