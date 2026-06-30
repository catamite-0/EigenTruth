"""Compare ProductTrace runtime baselines for drift.

This workflow compares two ``run_product_runtime_baseline.py`` reports without
rerunning models, verifiers, retrieval, or product demos. It is intended for
continuous runtime-verification handoff: a promoted product baseline can be kept
in the local registry, while fresh production/demo traces are compared against
that baseline under explicit drift gates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import planned_artifact_manifest_summary, strict_bool  # noqa: E402
from eigentruth.control import ProductRuntimeBudgetPolicy  # noqa: E402
from eigentruth.registry import ArtifactRegistry, RegistryRecord, build_artifact_manifest  # noqa: E402

_PROMOTION_EVIDENCE_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
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
_PRE_GENERATION_PROBE_COMPARISON_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
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
_CLAIM_FACTUALITY_PROBE_COMPARISON_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
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
_COUNTERFACTUAL_VERIFICATION_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
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
_EVIDENCE_HANDOFF_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
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
_FRONTIER_RELEASE_EVIDENCE_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
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

_TRIPLE_COVERAGE_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
    ("triple_coverage.claim_triple_coverage_rate", "triple_claim_coverage_rate"),
    ("triple_coverage.audit_claim_coverage_rate", "triple_audit_claim_coverage_rate"),
    ("triple_coverage.audit_pass_rate", "triple_audit_pass_rate"),
    ("triple_coverage.slot_coverage_rate", "triple_slot_coverage_rate"),
)
_WORLD_MODEL_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
    ("world_model.participating_trace_rate", "world_model_participating_trace_rate"),
    ("world_model.coverage_rate", "world_model_coverage_rate"),
    ("world_model.conflict_rate", "world_model_conflict_rate"),
    ("world_model.low_agreement_rate", "world_model_low_agreement_rate"),
    ("world_model.trace_gap_rate", "world_model_trace_gap_rate"),
)
_CONTEXT_SENSITIVITY_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
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
_COUNTERFACTUAL_ROBUSTNESS_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
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
_COVERED_FACT_PROPERTY_SCOPES: dict[str, str] = {
    "recommended_route": "recommended_route_property_metrics",
    "required_route_baseline": "required_route_baseline_property_metrics",
    "structured_fact_robustness": "structured_fact_robustness_property_metrics",
}
_COVERED_FACT_PROPERTY_METADATA_FIELDS: tuple[tuple[str, str], ...] = tuple(
    (
        f"promotion_contract.covered_fact_properties.{scope_key}.property_metric_count.mean",
        f"covered_fact_{scope_name}_property_metric_count",
    )
    for scope_name, scope_key in _COVERED_FACT_PROPERTY_SCOPES.items()
) + tuple(
    (
        f"promotion_contract.covered_fact_properties.{scope_key}.{metric_name}.mean",
        f"covered_fact_{scope_name}_{metric_name}",
    )
    for scope_name, scope_key in _COVERED_FACT_PROPERTY_SCOPES.items()
    for metric_name in (
        "min_records",
        "min_source_documents",
        "min_decision_accuracy",
        "max_false_supported_rate",
        "min_false_refuted_rate",
    )
)
_PRODUCT_TRACE_ACTION_GATE_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
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
_PRODUCT_TRACE_TRAJECTORY_AUDIT_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
    ("trajectory_audit.failed_trace_rate", "product_trace_trajectory_audit_failed_trace_rate"),
    ("trajectory_audit.error_rate", "product_trace_trajectory_audit_error_rate"),
    ("trajectory_audit.factual_rate", "product_trace_trajectory_audit_factual_rate"),
    ("trajectory_audit.referential_rate", "product_trace_trajectory_audit_referential_rate"),
    ("trajectory_audit.logical_rate", "product_trace_trajectory_audit_logical_rate"),
    ("trajectory_audit.procedural_rate", "product_trace_trajectory_audit_procedural_rate"),
    ("trajectory_audit.scope_rate", "product_trace_trajectory_audit_scope_rate"),
)
_PRODUCT_TRACE_ACTION_GATE_METRIC_SPECS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "promotion_contract.product_trace_replay.action_audit_gate.error_rate.mean",
        ("promotion_contract", "product_trace_replay", "action_audit_gate", "error_rate", "mean"),
        "max_product_trace_action_audit_error_rate_increase",
    ),
    (
        "promotion_contract.product_trace_replay.action_audit_gate.missing_retrieval_action_rate.mean",
        (
            "promotion_contract",
            "product_trace_replay",
            "action_audit_gate",
            "missing_retrieval_action_rate",
            "mean",
        ),
        "max_product_trace_action_audit_missing_retrieval_action_rate_increase",
    ),
    (
        "promotion_contract.product_trace_replay.action_audit_gate.missing_plan_retrieval_query_rate.mean",
        (
            "promotion_contract",
            "product_trace_replay",
            "action_audit_gate",
            "missing_plan_retrieval_query_rate",
            "mean",
        ),
        "max_product_trace_action_audit_missing_plan_retrieval_query_rate_increase",
    ),
    (
        "promotion_contract.product_trace_replay.action_audit_gate.malformed_payload_rate.mean",
        ("promotion_contract", "product_trace_replay", "action_audit_gate", "malformed_payload_rate", "mean"),
        "max_product_trace_action_audit_malformed_payload_rate_increase",
    ),
    (
        "promotion_contract.product_trace_replay.action_audit_gate.unexpected_action_rate.mean",
        ("promotion_contract", "product_trace_replay", "action_audit_gate", "unexpected_action_rate", "mean"),
        "max_product_trace_action_audit_unexpected_action_rate_increase",
    ),
    (
        "promotion_contract.product_trace_replay.action_audit_gate.unknown_claim_id_rate.mean",
        ("promotion_contract", "product_trace_replay", "action_audit_gate", "unknown_claim_id_rate", "mean"),
        "max_product_trace_action_audit_unknown_claim_id_rate_increase",
    ),
    (
        "promotion_contract.product_trace_replay.action_execution_gate.alignment_failed_trace_rate.mean",
        (
            "promotion_contract",
            "product_trace_replay",
            "action_execution_gate",
            "alignment_failed_trace_rate",
            "mean",
        ),
        "max_product_trace_action_execution_alignment_failed_trace_rate_increase",
    ),
    (
        "promotion_contract.product_trace_replay.action_execution_gate.missing_result_rate.mean",
        ("promotion_contract", "product_trace_replay", "action_execution_gate", "missing_result_rate", "mean"),
        "max_product_trace_action_execution_missing_result_rate_increase",
    ),
    (
        "promotion_contract.product_trace_replay.action_execution_gate.unexpected_result_rate.mean",
        ("promotion_contract", "product_trace_replay", "action_execution_gate", "unexpected_result_rate", "mean"),
        "max_product_trace_action_execution_unexpected_result_rate_increase",
    ),
    (
        "promotion_contract.product_trace_replay.action_execution_gate.request_id_mismatch_rate.mean",
        (
            "promotion_contract",
            "product_trace_replay",
            "action_execution_gate",
            "request_id_mismatch_rate",
            "mean",
        ),
        "max_product_trace_action_execution_request_id_mismatch_rate_increase",
    ),
)
_PRODUCT_TRACE_TRAJECTORY_AUDIT_METRIC_SPECS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "trajectory_audit.failed_trace_rate",
        ("trajectory_audit", "failed_trace_rate"),
        "max_product_trace_trajectory_audit_failed_trace_rate_increase",
    ),
    (
        "trajectory_audit.error_rate",
        ("trajectory_audit", "error_rate"),
        "max_product_trace_trajectory_audit_error_rate_increase",
    ),
    (
        "trajectory_audit.factual_rate",
        ("trajectory_audit", "factual_rate"),
        "max_product_trace_trajectory_audit_factual_rate_increase",
    ),
    (
        "trajectory_audit.referential_rate",
        ("trajectory_audit", "referential_rate"),
        "max_product_trace_trajectory_audit_referential_rate_increase",
    ),
    (
        "trajectory_audit.logical_rate",
        ("trajectory_audit", "logical_rate"),
        "max_product_trace_trajectory_audit_logical_rate_increase",
    ),
    (
        "trajectory_audit.procedural_rate",
        ("trajectory_audit", "procedural_rate"),
        "max_product_trace_trajectory_audit_procedural_rate_increase",
    ),
    (
        "trajectory_audit.scope_rate",
        ("trajectory_audit", "scope_rate"),
        "max_product_trace_trajectory_audit_scope_rate_increase",
    ),
)
_WORLD_MODEL_MIN_METRIC_SPECS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "world_model.participating_trace_rate",
        ("world_model", "participating_trace_rate"),
        "min_world_model_participating_trace_rate",
    ),
    (
        "world_model.coverage_rate",
        ("world_model", "coverage_rate"),
        "min_world_model_coverage_rate",
    ),
)
_WORLD_MODEL_INCREASE_METRIC_SPECS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "world_model.conflict_rate",
        ("world_model", "conflict_rate"),
        "max_world_model_conflict_rate_increase",
    ),
    (
        "world_model.low_agreement_rate",
        ("world_model", "low_agreement_rate"),
        "max_world_model_low_agreement_rate_increase",
    ),
    (
        "world_model.trace_gap_rate",
        ("world_model", "trace_gap_rate"),
        "max_world_model_trace_gap_rate_increase",
    ),
)
_CONTEXT_SENSITIVITY_MIN_METRIC_SPECS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "context_sensitivity.participating_trace_rate",
        ("context_sensitivity", "participating_trace_rate"),
        "min_context_sensitivity_participating_trace_rate",
    ),
    (
        "context_sensitivity.coverage_rate",
        ("context_sensitivity", "coverage_rate"),
        "min_context_sensitivity_coverage_rate",
    ),
)
_CONTEXT_SENSITIVITY_INCREASE_METRIC_SPECS: tuple[
    tuple[str, tuple[str, ...], str],
    ...
] = (
    (
        "context_sensitivity.flagged_result_rate",
        ("context_sensitivity", "flagged_result_rate"),
        "max_context_sensitivity_flagged_result_rate_increase",
    ),
    (
        "context_sensitivity.trace_gap_rate",
        ("context_sensitivity", "trace_gap_rate"),
        "max_context_sensitivity_trace_gap_rate_increase",
    ),
    (
        "context_sensitivity.max_flagged_rate",
        ("context_sensitivity", "max_flagged_rate"),
        "max_context_sensitivity_max_flagged_rate_increase",
    ),
    (
        "context_sensitivity.max_context_sensitivity_ratio",
        ("context_sensitivity", "max_context_sensitivity_ratio"),
        "max_context_sensitivity_max_ratio_increase",
    ),
)
_COUNTERFACTUAL_ROBUSTNESS_MIN_METRIC_SPECS: tuple[
    tuple[str, tuple[str, ...], str],
    ...
] = (
    (
        "counterfactual_robustness.participating_trace_rate",
        ("counterfactual_robustness", "participating_trace_rate"),
        "min_counterfactual_robustness_participating_trace_rate",
    ),
    (
        "counterfactual_robustness.coverage_rate",
        ("counterfactual_robustness", "coverage_rate"),
        "min_counterfactual_robustness_coverage_rate",
    ),
    (
        "counterfactual_robustness.pass_rate",
        ("counterfactual_robustness", "pass_rate"),
        "min_counterfactual_robustness_pass_rate",
    ),
    (
        "counterfactual_robustness.flip_success_rate",
        ("counterfactual_robustness", "flip_success_rate"),
        "min_counterfactual_robustness_flip_success_rate",
    ),
)
_COUNTERFACTUAL_ROBUSTNESS_INCREASE_METRIC_SPECS: tuple[
    tuple[str, tuple[str, ...], str],
    ...
] = (
    (
        "counterfactual_robustness.false_invariance_rate",
        ("counterfactual_robustness", "false_invariance_rate"),
        "max_counterfactual_robustness_false_invariance_rate_increase",
    ),
    (
        "counterfactual_robustness.trace_gap_rate",
        ("counterfactual_robustness", "trace_gap_rate"),
        "max_counterfactual_robustness_trace_gap_rate_increase",
    ),
)


def compare_product_runtime_baselines(
    *,
    current_path: str | Path,
    baseline_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    baseline_key: str | None = None,
    baseline_name: str | None = None,
    baseline_version: str | None = None,
    runtime_budget_policy_path: str | Path | None = None,
    runtime_budget_policy_key: str | None = None,
    report_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    max_total_seconds_mean_ratio: float | None = None,
    max_total_seconds_p95_ratio: float | None = None,
    max_mean_route_duration_ratio: float | None = None,
    max_p95_route_duration_ratio: float | None = None,
    max_mean_attempted_route_count_delta: float | None = None,
    max_retrieval_use_rate_delta: float | None = None,
    max_cache_hit_rate_drop: float | None = None,
    max_verification_skip_rate_drop: float | None = None,
    min_promotion_contract_coverage: float | None = None,
    min_pre_generation_probe_comparison_coverage: float | None = None,
    min_pre_generation_probe_comparison_manifest_verified_rate: float | None = None,
    min_pre_generation_probe_comparison_model_count: float | None = None,
    min_pre_generation_probe_comparison_run_count: float | None = None,
    min_pre_generation_probe_comparison_redline_pass_rate: float | None = None,
    max_pre_generation_probe_comparison_best_test_label_auroc_drop: float | None = None,
    max_pre_generation_probe_comparison_best_redline_auroc_drop: float | None = None,
    max_pre_generation_probe_comparison_best_redline_margin_drop: float | None = None,
    min_claim_factuality_probe_comparison_coverage: float | None = None,
    min_claim_factuality_probe_comparison_manifest_verified_rate: float | None = None,
    min_claim_factuality_probe_comparison_model_count: float | None = None,
    min_claim_factuality_probe_comparison_run_count: float | None = None,
    min_claim_factuality_probe_comparison_redline_pass_rate: float | None = None,
    max_claim_factuality_probe_comparison_best_test_label_auroc_drop: float | None = None,
    max_claim_factuality_probe_comparison_best_test_selective_accuracy_drop: (
        float | None
    ) = None,
    max_claim_factuality_probe_comparison_best_test_selective_coverage_drop: (
        float | None
    ) = None,
    max_claim_factuality_probe_comparison_best_redline_auroc_drop: float | None = None,
    max_claim_factuality_probe_comparison_best_redline_margin_drop: float | None = None,
    min_counterfactual_verification_coverage: float | None = None,
    min_counterfactual_verification_manifest_verified_rate: float | None = None,
    min_counterfactual_verification_record_count: float | None = None,
    min_counterfactual_verification_pass_rate: float | None = None,
    max_counterfactual_verification_false_invariance_rate: float | None = None,
    max_counterfactual_verification_flip_success_count_drop: float | None = None,
    min_evidence_handoff_coverage: float | None = None,
    min_evidence_handoff_manifest_verified_rate: float | None = None,
    min_evidence_handoff_present_metric_rate: float | None = None,
    max_evidence_handoff_missing_metric_rate: float | None = None,
    max_evidence_handoff_missing_metric_count: float | None = None,
    max_evidence_handoff_blocked_group_count: float | None = None,
    min_evidence_handoff_promoted_group_rate: float | None = None,
    min_frontier_release_evidence_coverage: float | None = None,
    min_frontier_release_evidence_report_present_rate: float | None = None,
    min_frontier_release_evidence_manifest_present_rate: float | None = None,
    min_frontier_release_evidence_status_promote_rate: float | None = None,
    min_frontier_release_evidence_decision_promote_rate: float | None = None,
    min_frontier_release_evidence_verifier_track_promote_rate: float | None = None,
    min_frontier_release_evidence_abstention_track_promote_rate: float | None = None,
    min_frontier_release_evidence_citation_batch_track_promote_rate: (
        float | None
    ) = None,
    min_frontier_release_evidence_frontier_rerun_rollup_track_promote_rate: (
        float | None
    ) = None,
    min_frontier_release_evidence_run_count: float | None = None,
    min_frontier_release_evidence_frontier_rerun_rollup_report_count: (
        float | None
    ) = None,
    min_frontier_release_evidence_frontier_rerun_rollup_candidate_count: (
        float | None
    ) = None,
    max_frontier_release_evidence_frontier_rerun_rollup_missing_report_count: (
        float | None
    ) = None,
    max_frontier_release_evidence_frontier_rerun_rollup_invalid_report_count: (
        float | None
    ) = None,
    max_frontier_release_evidence_frontier_rerun_rollup_blocked_candidate_count: (
        float | None
    ) = None,
    min_frontier_release_evidence_frontier_rerun_rollup_promotion_ready_count: (
        float | None
    ) = None,
    min_frontier_release_evidence_citation_batch_rollup_count: float | None = None,
    max_frontier_release_evidence_citation_batch_missing_expected_batch_count: (
        float | None
    ) = None,
    max_frontier_release_evidence_citation_batch_duplicate_batch_count: (
        float | None
    ) = None,
    max_frontier_release_evidence_citation_batch_unexpected_batch_count: (
        float | None
    ) = None,
    min_triple_extraction_fixture_matrix_coverage: float | None = None,
    max_triple_extraction_fixture_matrix_mean_best_f1_drop: float | None = None,
    max_triple_extraction_fixture_matrix_mean_f1_lift_drop: float | None = None,
    min_triple_claim_coverage: float | None = None,
    min_triple_audit_claim_coverage: float | None = None,
    min_triple_audit_pass_rate: float | None = None,
    min_triple_slot_coverage: float | None = None,
    min_world_model_participating_trace_rate: float | None = None,
    min_world_model_coverage_rate: float | None = None,
    max_world_model_conflict_rate_increase: float | None = None,
    max_world_model_low_agreement_rate_increase: float | None = None,
    max_world_model_trace_gap_rate_increase: float | None = None,
    min_context_sensitivity_participating_trace_rate: float | None = None,
    min_context_sensitivity_coverage_rate: float | None = None,
    max_context_sensitivity_flagged_result_rate_increase: float | None = None,
    max_context_sensitivity_trace_gap_rate_increase: float | None = None,
    max_context_sensitivity_max_flagged_rate_increase: float | None = None,
    max_context_sensitivity_max_ratio_increase: float | None = None,
    min_counterfactual_robustness_participating_trace_rate: float | None = None,
    min_counterfactual_robustness_coverage_rate: float | None = None,
    min_counterfactual_robustness_pass_rate: float | None = None,
    min_counterfactual_robustness_flip_success_rate: float | None = None,
    max_counterfactual_robustness_false_invariance_rate_increase: float | None = None,
    max_counterfactual_robustness_trace_gap_rate_increase: float | None = None,
    promotion_contract_covered_fact_property_scopes: Sequence[str] | None = None,
    min_promotion_contract_covered_fact_property_metric_count: float | None = None,
    min_promotion_contract_covered_fact_min_records: float | None = None,
    min_promotion_contract_covered_fact_min_source_documents: float | None = None,
    max_promotion_contract_covered_fact_min_decision_accuracy_drop: float | None = None,
    max_promotion_contract_covered_fact_max_false_supported_rate_increase: float | None = None,
    max_promotion_contract_covered_fact_min_false_refuted_rate_drop: float | None = None,
    max_product_trace_action_audit_error_rate_increase: float | None = None,
    max_product_trace_action_audit_missing_retrieval_action_rate_increase: float | None = None,
    max_product_trace_action_audit_missing_plan_retrieval_query_rate_increase: float | None = None,
    max_product_trace_action_audit_malformed_payload_rate_increase: float | None = None,
    max_product_trace_action_audit_unexpected_action_rate_increase: float | None = None,
    max_product_trace_action_audit_unknown_claim_id_rate_increase: float | None = None,
    max_product_trace_action_execution_alignment_failed_trace_rate_increase: float | None = None,
    max_product_trace_action_execution_missing_result_rate_increase: float | None = None,
    max_product_trace_action_execution_unexpected_result_rate_increase: float | None = None,
    max_product_trace_action_execution_request_id_mismatch_rate_increase: float | None = None,
    max_product_trace_trajectory_audit_failed_trace_rate_increase: float | None = None,
    max_product_trace_trajectory_audit_error_rate_increase: float | None = None,
    max_product_trace_trajectory_audit_factual_rate_increase: float | None = None,
    max_product_trace_trajectory_audit_referential_rate_increase: float | None = None,
    max_product_trace_trajectory_audit_logical_rate_increase: float | None = None,
    max_product_trace_trajectory_audit_procedural_rate_increase: float | None = None,
    max_product_trace_trajectory_audit_scope_rate_increase: float | None = None,
    min_current_trace_count: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build a fail-closed drift report between two product runtime baselines."""
    compact = strict_bool(compact_json, name="compact_json")
    source = _resolve_baseline_source(
        baseline_path=baseline_path,
        registry_path=registry_path,
        baseline_key=baseline_key,
        baseline_name=baseline_name,
        baseline_version=baseline_version,
    )
    policy_source = _resolve_runtime_budget_policy_source(
        runtime_budget_policy_path=runtime_budget_policy_path,
        registry_path=registry_path,
        runtime_budget_policy_key=runtime_budget_policy_key,
    )
    current_report_path = Path(current_path)
    baseline_report = _load_runtime_baseline(source["path"])
    current_report = _load_runtime_baseline(current_report_path)
    baseline_summary = _mapping(baseline_report.get("summary"))
    current_summary = _mapping(current_report.get("summary"))
    gates = {
        "max_total_seconds_mean_ratio": _optional_non_negative_float(max_total_seconds_mean_ratio),
        "max_total_seconds_p95_ratio": _optional_non_negative_float(max_total_seconds_p95_ratio),
        "max_mean_route_duration_ratio": _optional_non_negative_float(max_mean_route_duration_ratio),
        "max_p95_route_duration_ratio": _optional_non_negative_float(max_p95_route_duration_ratio),
        "max_mean_attempted_route_count_delta": _optional_non_negative_float(
            max_mean_attempted_route_count_delta
        ),
        "max_retrieval_use_rate_delta": _optional_non_negative_float(max_retrieval_use_rate_delta),
        "max_cache_hit_rate_drop": _optional_non_negative_float(max_cache_hit_rate_drop),
        "max_verification_skip_rate_drop": _optional_non_negative_float(max_verification_skip_rate_drop),
        "min_promotion_contract_coverage": _optional_rate_float(min_promotion_contract_coverage),
        "min_pre_generation_probe_comparison_coverage": _optional_rate_float(
            min_pre_generation_probe_comparison_coverage
        ),
        "min_pre_generation_probe_comparison_manifest_verified_rate": _optional_rate_float(
            min_pre_generation_probe_comparison_manifest_verified_rate
        ),
        "min_pre_generation_probe_comparison_model_count": _optional_non_negative_float(
            min_pre_generation_probe_comparison_model_count
        ),
        "min_pre_generation_probe_comparison_run_count": _optional_non_negative_float(
            min_pre_generation_probe_comparison_run_count
        ),
        "min_pre_generation_probe_comparison_redline_pass_rate": _optional_rate_float(
            min_pre_generation_probe_comparison_redline_pass_rate
        ),
        "max_pre_generation_probe_comparison_best_test_label_auroc_drop": (
            _optional_rate_float(max_pre_generation_probe_comparison_best_test_label_auroc_drop)
        ),
        "max_pre_generation_probe_comparison_best_redline_auroc_drop": (
            _optional_rate_float(max_pre_generation_probe_comparison_best_redline_auroc_drop)
        ),
        "max_pre_generation_probe_comparison_best_redline_margin_drop": (
            _optional_non_negative_float(max_pre_generation_probe_comparison_best_redline_margin_drop)
        ),
        "min_claim_factuality_probe_comparison_coverage": _optional_rate_float(
            min_claim_factuality_probe_comparison_coverage
        ),
        "min_claim_factuality_probe_comparison_manifest_verified_rate": (
            _optional_rate_float(
                min_claim_factuality_probe_comparison_manifest_verified_rate
            )
        ),
        "min_claim_factuality_probe_comparison_model_count": (
            _optional_non_negative_float(min_claim_factuality_probe_comparison_model_count)
        ),
        "min_claim_factuality_probe_comparison_run_count": (
            _optional_non_negative_float(min_claim_factuality_probe_comparison_run_count)
        ),
        "min_claim_factuality_probe_comparison_redline_pass_rate": (
            _optional_rate_float(min_claim_factuality_probe_comparison_redline_pass_rate)
        ),
        "max_claim_factuality_probe_comparison_best_test_label_auroc_drop": (
            _optional_rate_float(
                max_claim_factuality_probe_comparison_best_test_label_auroc_drop
            )
        ),
        "max_claim_factuality_probe_comparison_best_test_selective_accuracy_drop": (
            _optional_rate_float(
                max_claim_factuality_probe_comparison_best_test_selective_accuracy_drop
            )
        ),
        "max_claim_factuality_probe_comparison_best_test_selective_coverage_drop": (
            _optional_rate_float(
                max_claim_factuality_probe_comparison_best_test_selective_coverage_drop
            )
        ),
        "max_claim_factuality_probe_comparison_best_redline_auroc_drop": (
            _optional_rate_float(
                max_claim_factuality_probe_comparison_best_redline_auroc_drop
            )
        ),
        "max_claim_factuality_probe_comparison_best_redline_margin_drop": (
            _optional_non_negative_float(
                max_claim_factuality_probe_comparison_best_redline_margin_drop
            )
        ),
        "min_counterfactual_verification_coverage": _optional_rate_float(
            min_counterfactual_verification_coverage
        ),
        "min_counterfactual_verification_manifest_verified_rate": _optional_rate_float(
            min_counterfactual_verification_manifest_verified_rate
        ),
        "min_counterfactual_verification_record_count": _optional_non_negative_float(
            min_counterfactual_verification_record_count
        ),
        "min_counterfactual_verification_pass_rate": _optional_rate_float(
            min_counterfactual_verification_pass_rate
        ),
        "max_counterfactual_verification_false_invariance_rate": _optional_rate_float(
            max_counterfactual_verification_false_invariance_rate
        ),
        "max_counterfactual_verification_flip_success_count_drop": _optional_non_negative_float(
            max_counterfactual_verification_flip_success_count_drop
        ),
        "min_evidence_handoff_coverage": _optional_rate_float(
            min_evidence_handoff_coverage
        ),
        "min_evidence_handoff_manifest_verified_rate": _optional_rate_float(
            min_evidence_handoff_manifest_verified_rate
        ),
        "min_evidence_handoff_present_metric_rate": _optional_rate_float(
            min_evidence_handoff_present_metric_rate
        ),
        "max_evidence_handoff_missing_metric_rate": _optional_rate_float(
            max_evidence_handoff_missing_metric_rate
        ),
        "max_evidence_handoff_missing_metric_count": _optional_non_negative_float(
            max_evidence_handoff_missing_metric_count
        ),
        "max_evidence_handoff_blocked_group_count": _optional_non_negative_float(
            max_evidence_handoff_blocked_group_count
        ),
        "min_evidence_handoff_promoted_group_rate": _optional_rate_float(
            min_evidence_handoff_promoted_group_rate
        ),
        "min_frontier_release_evidence_coverage": _optional_rate_float(
            min_frontier_release_evidence_coverage
        ),
        "min_frontier_release_evidence_report_present_rate": _optional_rate_float(
            min_frontier_release_evidence_report_present_rate
        ),
        "min_frontier_release_evidence_manifest_present_rate": _optional_rate_float(
            min_frontier_release_evidence_manifest_present_rate
        ),
        "min_frontier_release_evidence_status_promote_rate": _optional_rate_float(
            min_frontier_release_evidence_status_promote_rate
        ),
        "min_frontier_release_evidence_decision_promote_rate": _optional_rate_float(
            min_frontier_release_evidence_decision_promote_rate
        ),
        "min_frontier_release_evidence_verifier_track_promote_rate": _optional_rate_float(
            min_frontier_release_evidence_verifier_track_promote_rate
        ),
        "min_frontier_release_evidence_abstention_track_promote_rate": _optional_rate_float(
            min_frontier_release_evidence_abstention_track_promote_rate
        ),
        "min_frontier_release_evidence_citation_batch_track_promote_rate": _optional_rate_float(
            min_frontier_release_evidence_citation_batch_track_promote_rate
        ),
        "min_frontier_release_evidence_frontier_rerun_rollup_track_promote_rate": _optional_rate_float(
            min_frontier_release_evidence_frontier_rerun_rollup_track_promote_rate
        ),
        "min_frontier_release_evidence_run_count": _optional_non_negative_float(
            min_frontier_release_evidence_run_count
        ),
        "min_frontier_release_evidence_frontier_rerun_rollup_report_count": (
            _optional_non_negative_float(
                min_frontier_release_evidence_frontier_rerun_rollup_report_count
            )
        ),
        "min_frontier_release_evidence_frontier_rerun_rollup_candidate_count": (
            _optional_non_negative_float(
                min_frontier_release_evidence_frontier_rerun_rollup_candidate_count
            )
        ),
        "max_frontier_release_evidence_frontier_rerun_rollup_missing_report_count": (
            _optional_non_negative_float(
                max_frontier_release_evidence_frontier_rerun_rollup_missing_report_count
            )
        ),
        "max_frontier_release_evidence_frontier_rerun_rollup_invalid_report_count": (
            _optional_non_negative_float(
                max_frontier_release_evidence_frontier_rerun_rollup_invalid_report_count
            )
        ),
        "max_frontier_release_evidence_frontier_rerun_rollup_blocked_candidate_count": (
            _optional_non_negative_float(
                max_frontier_release_evidence_frontier_rerun_rollup_blocked_candidate_count
            )
        ),
        "min_frontier_release_evidence_frontier_rerun_rollup_promotion_ready_count": (
            _optional_non_negative_float(
                min_frontier_release_evidence_frontier_rerun_rollup_promotion_ready_count
            )
        ),
        "min_frontier_release_evidence_citation_batch_rollup_count": _optional_non_negative_float(
            min_frontier_release_evidence_citation_batch_rollup_count
        ),
        "max_frontier_release_evidence_citation_batch_missing_expected_batch_count": (
            _optional_non_negative_float(
                max_frontier_release_evidence_citation_batch_missing_expected_batch_count
            )
        ),
        "max_frontier_release_evidence_citation_batch_duplicate_batch_count": (
            _optional_non_negative_float(
                max_frontier_release_evidence_citation_batch_duplicate_batch_count
            )
        ),
        "max_frontier_release_evidence_citation_batch_unexpected_batch_count": (
            _optional_non_negative_float(
                max_frontier_release_evidence_citation_batch_unexpected_batch_count
            )
        ),
        "min_triple_extraction_fixture_matrix_coverage": _optional_rate_float(
            min_triple_extraction_fixture_matrix_coverage
        ),
        "max_triple_extraction_fixture_matrix_mean_best_f1_drop": _optional_non_negative_float(
            max_triple_extraction_fixture_matrix_mean_best_f1_drop
        ),
        "max_triple_extraction_fixture_matrix_mean_f1_lift_drop": _optional_non_negative_float(
            max_triple_extraction_fixture_matrix_mean_f1_lift_drop
        ),
        "min_triple_claim_coverage": _optional_rate_float(min_triple_claim_coverage),
        "min_triple_audit_claim_coverage": _optional_rate_float(min_triple_audit_claim_coverage),
        "min_triple_audit_pass_rate": _optional_rate_float(min_triple_audit_pass_rate),
        "min_triple_slot_coverage": _optional_rate_float(min_triple_slot_coverage),
        "min_world_model_participating_trace_rate": _optional_rate_float(
            min_world_model_participating_trace_rate
        ),
        "min_world_model_coverage_rate": _optional_rate_float(min_world_model_coverage_rate),
        "max_world_model_conflict_rate_increase": _optional_rate_float(
            max_world_model_conflict_rate_increase
        ),
        "max_world_model_low_agreement_rate_increase": _optional_rate_float(
            max_world_model_low_agreement_rate_increase
        ),
        "max_world_model_trace_gap_rate_increase": _optional_rate_float(
            max_world_model_trace_gap_rate_increase
        ),
        "min_context_sensitivity_participating_trace_rate": _optional_rate_float(
            min_context_sensitivity_participating_trace_rate
        ),
        "min_context_sensitivity_coverage_rate": _optional_rate_float(
            min_context_sensitivity_coverage_rate
        ),
        "max_context_sensitivity_flagged_result_rate_increase": _optional_rate_float(
            max_context_sensitivity_flagged_result_rate_increase
        ),
        "max_context_sensitivity_trace_gap_rate_increase": _optional_rate_float(
            max_context_sensitivity_trace_gap_rate_increase
        ),
        "max_context_sensitivity_max_flagged_rate_increase": _optional_rate_float(
            max_context_sensitivity_max_flagged_rate_increase
        ),
        "max_context_sensitivity_max_ratio_increase": _optional_non_negative_float(
            max_context_sensitivity_max_ratio_increase
        ),
        "min_counterfactual_robustness_participating_trace_rate": _optional_rate_float(
            min_counterfactual_robustness_participating_trace_rate
        ),
        "min_counterfactual_robustness_coverage_rate": _optional_rate_float(
            min_counterfactual_robustness_coverage_rate
        ),
        "min_counterfactual_robustness_pass_rate": _optional_rate_float(
            min_counterfactual_robustness_pass_rate
        ),
        "min_counterfactual_robustness_flip_success_rate": _optional_rate_float(
            min_counterfactual_robustness_flip_success_rate
        ),
        "max_counterfactual_robustness_false_invariance_rate_increase": (
            _optional_rate_float(
                max_counterfactual_robustness_false_invariance_rate_increase
            )
        ),
        "max_counterfactual_robustness_trace_gap_rate_increase": _optional_rate_float(
            max_counterfactual_robustness_trace_gap_rate_increase
        ),
        "promotion_contract_covered_fact_property_scopes": _covered_fact_property_scopes(
            promotion_contract_covered_fact_property_scopes
        ),
        "min_promotion_contract_covered_fact_property_metric_count": _optional_non_negative_float(
            min_promotion_contract_covered_fact_property_metric_count
        ),
        "min_promotion_contract_covered_fact_min_records": _optional_non_negative_float(
            min_promotion_contract_covered_fact_min_records
        ),
        "min_promotion_contract_covered_fact_min_source_documents": _optional_non_negative_float(
            min_promotion_contract_covered_fact_min_source_documents
        ),
        "max_promotion_contract_covered_fact_min_decision_accuracy_drop": _optional_rate_float(
            max_promotion_contract_covered_fact_min_decision_accuracy_drop
        ),
        "max_promotion_contract_covered_fact_max_false_supported_rate_increase": _optional_rate_float(
            max_promotion_contract_covered_fact_max_false_supported_rate_increase
        ),
        "max_promotion_contract_covered_fact_min_false_refuted_rate_drop": _optional_rate_float(
            max_promotion_contract_covered_fact_min_false_refuted_rate_drop
        ),
        "max_product_trace_action_audit_error_rate_increase": _optional_rate_float(
            max_product_trace_action_audit_error_rate_increase
        ),
        "max_product_trace_action_audit_missing_retrieval_action_rate_increase": _optional_rate_float(
            max_product_trace_action_audit_missing_retrieval_action_rate_increase
        ),
        "max_product_trace_action_audit_missing_plan_retrieval_query_rate_increase": (
            _optional_rate_float(
                max_product_trace_action_audit_missing_plan_retrieval_query_rate_increase
            )
        ),
        "max_product_trace_action_audit_malformed_payload_rate_increase": _optional_rate_float(
            max_product_trace_action_audit_malformed_payload_rate_increase
        ),
        "max_product_trace_action_audit_unexpected_action_rate_increase": _optional_rate_float(
            max_product_trace_action_audit_unexpected_action_rate_increase
        ),
        "max_product_trace_action_audit_unknown_claim_id_rate_increase": _optional_rate_float(
            max_product_trace_action_audit_unknown_claim_id_rate_increase
        ),
        "max_product_trace_action_execution_alignment_failed_trace_rate_increase": (
            _optional_rate_float(
                max_product_trace_action_execution_alignment_failed_trace_rate_increase
            )
        ),
        "max_product_trace_action_execution_missing_result_rate_increase": _optional_rate_float(
            max_product_trace_action_execution_missing_result_rate_increase
        ),
        "max_product_trace_action_execution_unexpected_result_rate_increase": (
            _optional_rate_float(
                max_product_trace_action_execution_unexpected_result_rate_increase
            )
        ),
        "max_product_trace_action_execution_request_id_mismatch_rate_increase": (
            _optional_rate_float(
                max_product_trace_action_execution_request_id_mismatch_rate_increase
            )
        ),
        "max_product_trace_trajectory_audit_failed_trace_rate_increase": (
            _optional_rate_float(max_product_trace_trajectory_audit_failed_trace_rate_increase)
        ),
        "max_product_trace_trajectory_audit_error_rate_increase": _optional_rate_float(
            max_product_trace_trajectory_audit_error_rate_increase
        ),
        "max_product_trace_trajectory_audit_factual_rate_increase": _optional_rate_float(
            max_product_trace_trajectory_audit_factual_rate_increase
        ),
        "max_product_trace_trajectory_audit_referential_rate_increase": _optional_rate_float(
            max_product_trace_trajectory_audit_referential_rate_increase
        ),
        "max_product_trace_trajectory_audit_logical_rate_increase": _optional_rate_float(
            max_product_trace_trajectory_audit_logical_rate_increase
        ),
        "max_product_trace_trajectory_audit_procedural_rate_increase": _optional_rate_float(
            max_product_trace_trajectory_audit_procedural_rate_increase
        ),
        "max_product_trace_trajectory_audit_scope_rate_increase": _optional_rate_float(
            max_product_trace_trajectory_audit_scope_rate_increase
        ),
        "min_current_trace_count": _optional_non_negative_int(min_current_trace_count),
    }
    metrics = _comparison_metrics(
        baseline_summary,
        current_summary,
        gates=gates,
    )
    runtime_budget_policy_gate = _runtime_budget_policy_gate(
        current_summary,
        source=policy_source,
    )
    drift_gate_enabled = any(
        value is not None
        for key, value in gates.items()
        if key != "promotion_contract_covered_fact_property_scopes"
    )
    runtime_budget_policy_gate_enabled = bool(runtime_budget_policy_gate.get("enabled"))
    gate_enabled = drift_gate_enabled or runtime_budget_policy_gate_enabled
    blocking_reasons = tuple(
        str(metric["reason"])
        for metric in metrics
        if metric.get("status") == "blocked" and metric.get("reason")
    ) + tuple(
        str(failure["reason"])
        for failure in runtime_budget_policy_gate.get("failures", ())
        if isinstance(failure, Mapping) and failure.get("reason")
    )
    status = "blocked" if blocking_reasons else ("promote" if gate_enabled else "observed")
    resolved_report_path = None if report_path is None else Path(report_path)
    resolved_manifest_path = _artifact_manifest_path(
        report_path=resolved_report_path,
        artifact_manifest_path=None if artifact_manifest_path is None else Path(artifact_manifest_path),
    )
    report = {
        "schema_version": 1,
        "workflow": "product_runtime_drift_comparison",
        "status": status,
        "decision": {
            "status": status,
            "blocking_reasons": blocking_reasons,
        },
        "summary": {
            "gate_enabled": gate_enabled,
            "drift_gate_enabled": drift_gate_enabled,
            "runtime_budget_policy_gate_enabled": runtime_budget_policy_gate_enabled,
            "runtime_budget_policy_passed": runtime_budget_policy_gate.get("passed"),
            "runtime_budget_policy_check_count": runtime_budget_policy_gate.get("check_count"),
            "runtime_budget_policy_failed_count": runtime_budget_policy_gate.get("failed_count"),
            "compared_metric_count": len(metrics),
            "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
            "observed_metric_count": sum(1 for metric in metrics if metric.get("status") == "observed"),
        },
        "baseline": {
            "path": str(source["path"]),
            "status": baseline_report.get("status"),
            "source": source["source"],
            "registry": source.get("registry"),
            "record_key": source.get("record_key"),
            "record": source.get("record"),
            "optimization": _runtime_optimization_handoff(baseline_report),
        },
        "current": {
            "path": str(current_report_path),
            "status": current_report.get("status"),
            "optimization": _runtime_optimization_handoff(current_report),
        },
        "metrics": metrics,
        "runtime_budget_policy_gate": runtime_budget_policy_gate,
        "paths": {
            "report": None if resolved_report_path is None else str(resolved_report_path),
            "artifact_manifest": None if resolved_manifest_path is None else str(resolved_manifest_path),
            "runtime_budget_policy": None if policy_source is None else str(policy_source["path"]),
        },
        "config": {
            **gates,
            "runtime_budget_policy_path": (
                None if runtime_budget_policy_path is None else str(runtime_budget_policy_path)
            ),
            "runtime_budget_policy_key": runtime_budget_policy_key,
            "compact_json": compact,
            "metadata": dict(metadata or {}),
        },
    }
    return _write_outputs(
        report,
        baseline_path=Path(source["path"]),
        current_path=current_report_path,
        report_path=resolved_report_path,
        artifact_manifest_path=resolved_manifest_path,
        registry_path=None if registry_path is None else Path(registry_path),
        name=name,
        version=version,
        compact=compact,
    )


def _comparison_metrics(
    baseline_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    metrics = [
        _ratio_metric(
            "total_seconds.mean",
            _nested_float(baseline_summary, ("total_seconds", "mean")),
            _nested_float(current_summary, ("total_seconds", "mean")),
            gates.get("max_total_seconds_mean_ratio"),
        ),
        _ratio_metric(
            "total_seconds.p95",
            _nested_float(baseline_summary, ("total_seconds", "p95")),
            _nested_float(current_summary, ("total_seconds", "p95")),
            gates.get("max_total_seconds_p95_ratio"),
        ),
        _ratio_metric(
            "route.mean_duration_seconds.mean",
            _nested_float(baseline_summary, ("mean_route_duration_seconds", "mean")),
            _nested_float(current_summary, ("mean_route_duration_seconds", "mean")),
            gates.get("max_mean_route_duration_ratio"),
        ),
        _ratio_metric(
            "route.p95_duration_seconds.mean",
            _nested_float(baseline_summary, ("p95_route_duration_seconds", "mean")),
            _nested_float(current_summary, ("p95_route_duration_seconds", "mean")),
            gates.get("max_p95_route_duration_ratio"),
        ),
        _delta_metric(
            "mean_attempted_route_count.mean",
            _nested_float(baseline_summary, ("mean_attempted_route_count", "mean")),
            _nested_float(current_summary, ("mean_attempted_route_count", "mean")),
            gates.get("max_mean_attempted_route_count_delta"),
        ),
        _delta_metric(
            "retrieval_use_rate.mean",
            _nested_float(baseline_summary, ("retrieval_use_rate", "mean")),
            _nested_float(current_summary, ("retrieval_use_rate", "mean")),
            gates.get("max_retrieval_use_rate_delta"),
        ),
        _drop_metric(
            "cache_hit_rate.mean",
            _nested_float(baseline_summary, ("cache_hit_rate", "mean")),
            _nested_float(current_summary, ("cache_hit_rate", "mean")),
            gates.get("max_cache_hit_rate_drop"),
        ),
        _drop_metric(
            "verification_skip_rate.mean",
            _nested_float(baseline_summary, ("verification_skip_rate", "mean")),
            _nested_float(current_summary, ("verification_skip_rate", "mean")),
            gates.get("max_verification_skip_rate_drop"),
        ),
        _min_current_metric(
            "promotion_contract.coverage_rate",
            _nested_float(baseline_summary, ("promotion_contract", "coverage_rate")),
            _nested_float(current_summary, ("promotion_contract", "coverage_rate")),
            gates.get("min_promotion_contract_coverage"),
        ),
        _min_current_metric(
            "promotion_contract.triple_extraction_fixture_matrix.coverage_rate",
            _nested_float(
                baseline_summary,
                ("promotion_contract", "triple_extraction_fixture_matrix", "coverage_rate"),
            ),
            _nested_float(
                current_summary,
                ("promotion_contract", "triple_extraction_fixture_matrix", "coverage_rate"),
            ),
            gates.get("min_triple_extraction_fixture_matrix_coverage"),
        ),
        _drop_metric(
            "promotion_contract.triple_extraction_fixture_matrix.mean_best_f1.mean",
            _nested_float(
                baseline_summary,
                (
                    "promotion_contract",
                    "triple_extraction_fixture_matrix",
                    "mean_best_f1",
                    "mean",
                ),
            ),
            _nested_float(
                current_summary,
                (
                    "promotion_contract",
                    "triple_extraction_fixture_matrix",
                    "mean_best_f1",
                    "mean",
                ),
            ),
            gates.get("max_triple_extraction_fixture_matrix_mean_best_f1_drop"),
        ),
        _drop_metric(
            "promotion_contract.triple_extraction_fixture_matrix.mean_f1_lift.mean",
            _nested_float(
                baseline_summary,
                (
                    "promotion_contract",
                    "triple_extraction_fixture_matrix",
                    "mean_f1_lift",
                    "mean",
                ),
            ),
            _nested_float(
                current_summary,
                (
                    "promotion_contract",
                    "triple_extraction_fixture_matrix",
                    "mean_f1_lift",
                    "mean",
                ),
            ),
            gates.get("max_triple_extraction_fixture_matrix_mean_f1_lift_drop"),
        ),
        _min_current_metric(
            "triple_coverage.claim_triple_coverage_rate",
            _nested_float(baseline_summary, ("triple_coverage", "claim_triple_coverage_rate")),
            _nested_float(current_summary, ("triple_coverage", "claim_triple_coverage_rate")),
            gates.get("min_triple_claim_coverage"),
        ),
        _min_current_metric(
            "triple_coverage.audit_claim_coverage_rate",
            _nested_float(baseline_summary, ("triple_coverage", "audit_claim_coverage_rate")),
            _nested_float(current_summary, ("triple_coverage", "audit_claim_coverage_rate")),
            gates.get("min_triple_audit_claim_coverage"),
        ),
        _min_current_metric(
            "triple_coverage.audit_pass_rate",
            _nested_float(baseline_summary, ("triple_coverage", "audit_pass_rate")),
            _nested_float(current_summary, ("triple_coverage", "audit_pass_rate")),
            gates.get("min_triple_audit_pass_rate"),
        ),
        _min_current_metric(
            "triple_coverage.slot_coverage_rate",
            _nested_float(baseline_summary, ("triple_coverage", "slot_coverage_rate")),
            _nested_float(current_summary, ("triple_coverage", "slot_coverage_rate")),
            gates.get("min_triple_slot_coverage"),
        ),
        _min_metric(
            "n_traces",
            _finite_float(current_summary.get("n_traces")),
            gates.get("min_current_trace_count"),
        ),
    ]
    metrics.extend(_covered_fact_property_metrics(baseline_summary, current_summary, gates=gates))
    metrics.extend(_product_trace_action_gate_metrics(baseline_summary, current_summary, gates=gates))
    metrics.extend(_product_trace_trajectory_audit_metrics(baseline_summary, current_summary, gates=gates))
    metrics.extend(_world_model_metrics(baseline_summary, current_summary, gates=gates))
    metrics.extend(_context_sensitivity_metrics(baseline_summary, current_summary, gates=gates))
    metrics.extend(_counterfactual_robustness_metrics(baseline_summary, current_summary, gates=gates))
    metrics.extend(_pre_generation_probe_comparison_metrics(baseline_summary, current_summary, gates=gates))
    metrics.extend(_claim_factuality_probe_comparison_metrics(baseline_summary, current_summary, gates=gates))
    metrics.extend(_counterfactual_verification_metrics(baseline_summary, current_summary, gates=gates))
    metrics.extend(_evidence_handoff_metrics(baseline_summary, current_summary, gates=gates))
    metrics.extend(
        _frontier_release_evidence_metrics(baseline_summary, current_summary, gates=gates)
    )
    return metrics


def _pre_generation_probe_comparison_metrics(
    baseline_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not _pre_generation_probe_comparison_gate_enabled(gates):
        return []
    baseline = _mapping(_nested_value(baseline_summary, ("promotion_contract", "pre_generation_probe_comparison")))
    current = _mapping(_nested_value(current_summary, ("promotion_contract", "pre_generation_probe_comparison")))
    return [
        _min_current_metric(
            "promotion_contract.pre_generation_probe_comparison.coverage_rate",
            _finite_float(baseline.get("coverage_rate")),
            _finite_float(current.get("coverage_rate")),
            gates.get("min_pre_generation_probe_comparison_coverage"),
        ),
        _min_current_metric(
            "promotion_contract.pre_generation_probe_comparison.manifest_verified_rate",
            _pre_generation_manifest_verified_rate(baseline),
            _pre_generation_manifest_verified_rate(current),
            gates.get("min_pre_generation_probe_comparison_manifest_verified_rate"),
        ),
        _min_current_metric(
            "promotion_contract.pre_generation_probe_comparison.model_count.mean",
            _nested_float(baseline, ("model_count", "mean")),
            _nested_float(current, ("model_count", "mean")),
            gates.get("min_pre_generation_probe_comparison_model_count"),
        ),
        _min_current_metric(
            "promotion_contract.pre_generation_probe_comparison.run_count.mean",
            _nested_float(baseline, ("run_count", "mean")),
            _nested_float(current, ("run_count", "mean")),
            gates.get("min_pre_generation_probe_comparison_run_count"),
        ),
        _min_current_metric(
            "promotion_contract.pre_generation_probe_comparison.redline_pass_rate",
            _pre_generation_redline_pass_rate(baseline),
            _pre_generation_redline_pass_rate(current),
            gates.get("min_pre_generation_probe_comparison_redline_pass_rate"),
        ),
        _drop_metric(
            "promotion_contract.pre_generation_probe_comparison.best_test_label_auroc.mean",
            _nested_float(baseline, ("best_test_label_auroc", "mean")),
            _nested_float(current, ("best_test_label_auroc", "mean")),
            gates.get("max_pre_generation_probe_comparison_best_test_label_auroc_drop"),
        ),
        _drop_metric(
            "promotion_contract.pre_generation_probe_comparison.best_redline_auroc.mean",
            _nested_float(baseline, ("best_redline_auroc", "mean")),
            _nested_float(current, ("best_redline_auroc", "mean")),
            gates.get("max_pre_generation_probe_comparison_best_redline_auroc_drop"),
        ),
        _drop_metric(
            "promotion_contract.pre_generation_probe_comparison.best_redline_margin.mean",
            _nested_float(baseline, ("best_redline_margin", "mean")),
            _nested_float(current, ("best_redline_margin", "mean")),
            gates.get("max_pre_generation_probe_comparison_best_redline_margin_drop"),
        ),
    ]


def _pre_generation_probe_comparison_gate_enabled(gates: Mapping[str, Any]) -> bool:
    return any(
        gates.get(key) is not None
        for key in (
            "min_pre_generation_probe_comparison_coverage",
            "min_pre_generation_probe_comparison_manifest_verified_rate",
            "min_pre_generation_probe_comparison_model_count",
            "min_pre_generation_probe_comparison_run_count",
            "min_pre_generation_probe_comparison_redline_pass_rate",
            "max_pre_generation_probe_comparison_best_test_label_auroc_drop",
            "max_pre_generation_probe_comparison_best_redline_auroc_drop",
            "max_pre_generation_probe_comparison_best_redline_margin_drop",
        )
    )


def _claim_factuality_probe_comparison_metrics(
    baseline_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not _claim_factuality_probe_comparison_gate_enabled(gates):
        return []
    baseline = _mapping(
        _nested_value(
            baseline_summary,
            ("promotion_contract", "claim_factuality_probe_comparison"),
        )
    )
    current = _mapping(
        _nested_value(
            current_summary,
            ("promotion_contract", "claim_factuality_probe_comparison"),
        )
    )
    return [
        _min_current_metric(
            "promotion_contract.claim_factuality_probe_comparison.coverage_rate",
            _finite_float(baseline.get("coverage_rate")),
            _finite_float(current.get("coverage_rate")),
            gates.get("min_claim_factuality_probe_comparison_coverage"),
        ),
        _min_current_metric(
            "promotion_contract.claim_factuality_probe_comparison.manifest_verified_rate",
            _pre_generation_manifest_verified_rate(baseline),
            _pre_generation_manifest_verified_rate(current),
            gates.get("min_claim_factuality_probe_comparison_manifest_verified_rate"),
        ),
        _min_current_metric(
            "promotion_contract.claim_factuality_probe_comparison.model_count.mean",
            _nested_float(baseline, ("model_count", "mean")),
            _nested_float(current, ("model_count", "mean")),
            gates.get("min_claim_factuality_probe_comparison_model_count"),
        ),
        _min_current_metric(
            "promotion_contract.claim_factuality_probe_comparison.run_count.mean",
            _nested_float(baseline, ("run_count", "mean")),
            _nested_float(current, ("run_count", "mean")),
            gates.get("min_claim_factuality_probe_comparison_run_count"),
        ),
        _min_current_metric(
            "promotion_contract.claim_factuality_probe_comparison.redline_pass_rate",
            _pre_generation_redline_pass_rate(baseline),
            _pre_generation_redline_pass_rate(current),
            gates.get("min_claim_factuality_probe_comparison_redline_pass_rate"),
        ),
        _drop_metric(
            "promotion_contract.claim_factuality_probe_comparison.best_test_label_auroc.mean",
            _nested_float(baseline, ("best_test_label_auroc", "mean")),
            _nested_float(current, ("best_test_label_auroc", "mean")),
            gates.get(
                "max_claim_factuality_probe_comparison_best_test_label_auroc_drop"
            ),
        ),
        _drop_metric(
            "promotion_contract.claim_factuality_probe_comparison.best_test_selective_accuracy.mean",
            _nested_float(baseline, ("best_test_selective_accuracy", "mean")),
            _nested_float(current, ("best_test_selective_accuracy", "mean")),
            gates.get(
                "max_claim_factuality_probe_comparison_best_test_selective_accuracy_drop"
            ),
        ),
        _drop_metric(
            "promotion_contract.claim_factuality_probe_comparison.best_test_selective_coverage.mean",
            _nested_float(baseline, ("best_test_selective_coverage", "mean")),
            _nested_float(current, ("best_test_selective_coverage", "mean")),
            gates.get(
                "max_claim_factuality_probe_comparison_best_test_selective_coverage_drop"
            ),
        ),
        _drop_metric(
            "promotion_contract.claim_factuality_probe_comparison.best_redline_auroc.mean",
            _nested_float(baseline, ("best_redline_auroc", "mean")),
            _nested_float(current, ("best_redline_auroc", "mean")),
            gates.get("max_claim_factuality_probe_comparison_best_redline_auroc_drop"),
        ),
        _drop_metric(
            "promotion_contract.claim_factuality_probe_comparison.best_redline_margin.mean",
            _nested_float(baseline, ("best_redline_margin", "mean")),
            _nested_float(current, ("best_redline_margin", "mean")),
            gates.get("max_claim_factuality_probe_comparison_best_redline_margin_drop"),
        ),
    ]


def _claim_factuality_probe_comparison_gate_enabled(gates: Mapping[str, Any]) -> bool:
    return any(
        gates.get(key) is not None
        for key in (
            "min_claim_factuality_probe_comparison_coverage",
            "min_claim_factuality_probe_comparison_manifest_verified_rate",
            "min_claim_factuality_probe_comparison_model_count",
            "min_claim_factuality_probe_comparison_run_count",
            "min_claim_factuality_probe_comparison_redline_pass_rate",
            "max_claim_factuality_probe_comparison_best_test_label_auroc_drop",
            "max_claim_factuality_probe_comparison_best_test_selective_accuracy_drop",
            "max_claim_factuality_probe_comparison_best_test_selective_coverage_drop",
            "max_claim_factuality_probe_comparison_best_redline_auroc_drop",
            "max_claim_factuality_probe_comparison_best_redline_margin_drop",
        )
    )


def _pre_generation_manifest_verified_rate(summary: Mapping[str, Any]) -> float | None:
    verified = _finite_float(summary.get("manifest_verified_count"))
    failed = _finite_float(summary.get("manifest_failed_count"))
    unknown = _finite_float(summary.get("manifest_unknown_count"))
    finite = tuple(value for value in (verified, failed, unknown) if value is not None)
    total = sum(finite)
    if total <= 0.0:
        return None
    return (verified or 0.0) / total


def _pre_generation_redline_pass_rate(summary: Mapping[str, Any]) -> float | None:
    counts = _mapping(summary.get("redline_passed_counts"))
    return _count_rate(counts, ("True", "true", "1", "yes", "on"))


def _counterfactual_verification_metrics(
    baseline_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not _counterfactual_verification_gate_enabled(gates):
        return []
    baseline = _mapping(_nested_value(baseline_summary, ("promotion_contract", "counterfactual_verification")))
    current = _mapping(_nested_value(current_summary, ("promotion_contract", "counterfactual_verification")))
    return [
        _min_current_metric(
            "promotion_contract.counterfactual_verification.coverage_rate",
            _finite_float(baseline.get("coverage_rate")),
            _finite_float(current.get("coverage_rate")),
            gates.get("min_counterfactual_verification_coverage"),
        ),
        _min_current_metric(
            "promotion_contract.counterfactual_verification.manifest_verified_rate",
            _manifest_verified_rate(baseline),
            _manifest_verified_rate(current),
            gates.get("min_counterfactual_verification_manifest_verified_rate"),
        ),
        _min_current_metric(
            "promotion_contract.counterfactual_verification.record_count.mean",
            _nested_float(baseline, ("record_count", "mean")),
            _nested_float(current, ("record_count", "mean")),
            gates.get("min_counterfactual_verification_record_count"),
        ),
        _min_current_metric(
            "promotion_contract.counterfactual_verification.pass_rate.mean",
            _nested_float(baseline, ("pass_rate", "mean")),
            _nested_float(current, ("pass_rate", "mean")),
            gates.get("min_counterfactual_verification_pass_rate"),
        ),
        _max_current_metric(
            "promotion_contract.counterfactual_verification.false_invariance_rate.mean",
            _nested_float(baseline, ("false_invariance_rate", "mean")),
            _nested_float(current, ("false_invariance_rate", "mean")),
            gates.get("max_counterfactual_verification_false_invariance_rate"),
        ),
        _drop_metric(
            "promotion_contract.counterfactual_verification.flip_success_count.mean",
            _nested_float(baseline, ("flip_success_count", "mean")),
            _nested_float(current, ("flip_success_count", "mean")),
            gates.get("max_counterfactual_verification_flip_success_count_drop"),
        ),
    ]


def _counterfactual_verification_gate_enabled(gates: Mapping[str, Any]) -> bool:
    return any(
        gates.get(key) is not None
        for key in (
            "min_counterfactual_verification_coverage",
            "min_counterfactual_verification_manifest_verified_rate",
            "min_counterfactual_verification_record_count",
            "min_counterfactual_verification_pass_rate",
            "max_counterfactual_verification_false_invariance_rate",
            "max_counterfactual_verification_flip_success_count_drop",
        )
    )


def _evidence_handoff_metrics(
    baseline_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not _evidence_handoff_gate_enabled(gates):
        return []
    baseline = _mapping(_nested_value(baseline_summary, ("promotion_contract", "evidence_handoff")))
    current = _mapping(_nested_value(current_summary, ("promotion_contract", "evidence_handoff")))
    return [
        _min_current_metric(
            "promotion_contract.evidence_handoff.coverage_rate",
            _finite_float(baseline.get("coverage_rate")),
            _finite_float(current.get("coverage_rate")),
            gates.get("min_evidence_handoff_coverage"),
        ),
        _min_current_metric(
            "promotion_contract.evidence_handoff.manifest_verified_rate",
            _manifest_verified_rate(baseline),
            _manifest_verified_rate(current),
            gates.get("min_evidence_handoff_manifest_verified_rate"),
        ),
        _min_current_metric(
            "promotion_contract.evidence_handoff.present_metric_rate.mean",
            _nested_float(baseline, ("present_metric_rate", "mean")),
            _nested_float(current, ("present_metric_rate", "mean")),
            gates.get("min_evidence_handoff_present_metric_rate"),
        ),
        _max_current_metric(
            "promotion_contract.evidence_handoff.missing_metric_rate.mean",
            _nested_float(baseline, ("missing_metric_rate", "mean")),
            _nested_float(current, ("missing_metric_rate", "mean")),
            gates.get("max_evidence_handoff_missing_metric_rate"),
        ),
        _max_current_metric(
            "promotion_contract.evidence_handoff.missing_metric_count.mean",
            _nested_float(baseline, ("missing_metric_count", "mean")),
            _nested_float(current, ("missing_metric_count", "mean")),
            gates.get("max_evidence_handoff_missing_metric_count"),
        ),
        _max_current_metric(
            "promotion_contract.evidence_handoff.blocked_group_count.mean",
            _nested_float(baseline, ("blocked_group_count", "mean")),
            _nested_float(current, ("blocked_group_count", "mean")),
            gates.get("max_evidence_handoff_blocked_group_count"),
        ),
        _min_current_metric(
            "promotion_contract.evidence_handoff.promoted_group_rate.mean",
            _nested_float(baseline, ("promoted_group_rate", "mean")),
            _nested_float(current, ("promoted_group_rate", "mean")),
            gates.get("min_evidence_handoff_promoted_group_rate"),
        ),
    ]


def _evidence_handoff_gate_enabled(gates: Mapping[str, Any]) -> bool:
    return any(
        gates.get(key) is not None
        for key in (
            "min_evidence_handoff_coverage",
            "min_evidence_handoff_manifest_verified_rate",
            "min_evidence_handoff_present_metric_rate",
            "max_evidence_handoff_missing_metric_rate",
            "max_evidence_handoff_missing_metric_count",
            "max_evidence_handoff_blocked_group_count",
            "min_evidence_handoff_promoted_group_rate",
        )
    )


def _frontier_release_evidence_metrics(
    baseline_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not _frontier_release_evidence_gate_enabled(gates):
        return []
    baseline = _mapping(
        _nested_value(
            baseline_summary,
            ("promotion_contract", "frontier_release_evidence"),
        )
    )
    current = _mapping(
        _nested_value(
            current_summary,
            ("promotion_contract", "frontier_release_evidence"),
        )
    )
    return [
        _min_current_metric(
            "promotion_contract.frontier_release_evidence.coverage_rate",
            _finite_float(baseline.get("coverage_rate")),
            _finite_float(current.get("coverage_rate")),
            gates.get("min_frontier_release_evidence_coverage"),
        ),
        _min_current_metric(
            "promotion_contract.frontier_release_evidence.report_present_rate",
            _finite_float(baseline.get("report_present_rate")),
            _finite_float(current.get("report_present_rate")),
            gates.get("min_frontier_release_evidence_report_present_rate"),
        ),
        _min_current_metric(
            "promotion_contract.frontier_release_evidence.manifest_present_rate",
            _finite_float(baseline.get("manifest_present_rate")),
            _finite_float(current.get("manifest_present_rate")),
            gates.get("min_frontier_release_evidence_manifest_present_rate"),
        ),
        _min_current_metric(
            "promotion_contract.frontier_release_evidence.status_promote_rate",
            _finite_float(baseline.get("status_promote_rate")),
            _finite_float(current.get("status_promote_rate")),
            gates.get("min_frontier_release_evidence_status_promote_rate"),
        ),
        _min_current_metric(
            "promotion_contract.frontier_release_evidence.decision_promote_rate",
            _finite_float(baseline.get("decision_promote_rate")),
            _finite_float(current.get("decision_promote_rate")),
            gates.get("min_frontier_release_evidence_decision_promote_rate"),
        ),
        _min_current_metric(
            "promotion_contract.frontier_release_evidence.verifier_track_promote_rate",
            _finite_float(baseline.get("verifier_track_promote_rate")),
            _finite_float(current.get("verifier_track_promote_rate")),
            gates.get("min_frontier_release_evidence_verifier_track_promote_rate"),
        ),
        _min_current_metric(
            "promotion_contract.frontier_release_evidence.abstention_track_promote_rate",
            _finite_float(baseline.get("abstention_track_promote_rate")),
            _finite_float(current.get("abstention_track_promote_rate")),
            gates.get("min_frontier_release_evidence_abstention_track_promote_rate"),
        ),
        _min_current_metric(
            "promotion_contract.frontier_release_evidence.citation_batch_track_promote_rate",
            _finite_float(baseline.get("citation_batch_track_promote_rate")),
            _finite_float(current.get("citation_batch_track_promote_rate")),
            gates.get(
                "min_frontier_release_evidence_citation_batch_track_promote_rate"
            ),
        ),
        _min_current_metric(
            "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_track_promote_rate",
            _finite_float(baseline.get("frontier_rerun_rollup_track_promote_rate")),
            _finite_float(current.get("frontier_rerun_rollup_track_promote_rate")),
            gates.get(
                "min_frontier_release_evidence_frontier_rerun_rollup_track_promote_rate"
            ),
        ),
        _min_current_metric(
            "promotion_contract.frontier_release_evidence.run_count.mean",
            _nested_float(baseline, ("run_count", "mean")),
            _nested_float(current, ("run_count", "mean")),
            gates.get("min_frontier_release_evidence_run_count"),
        ),
        _min_current_metric(
            "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_report_count.mean",
            _nested_float(baseline, ("frontier_rerun_rollup_report_count", "mean")),
            _nested_float(current, ("frontier_rerun_rollup_report_count", "mean")),
            gates.get(
                "min_frontier_release_evidence_frontier_rerun_rollup_report_count"
            ),
        ),
        _min_current_metric(
            "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_candidate_count.mean",
            _nested_float(baseline, ("frontier_rerun_rollup_candidate_count", "mean")),
            _nested_float(current, ("frontier_rerun_rollup_candidate_count", "mean")),
            gates.get(
                "min_frontier_release_evidence_frontier_rerun_rollup_candidate_count"
            ),
        ),
        _max_current_metric(
            "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_missing_report_count.mean",
            _nested_float(
                baseline,
                ("frontier_rerun_rollup_missing_report_count", "mean"),
            ),
            _nested_float(
                current,
                ("frontier_rerun_rollup_missing_report_count", "mean"),
            ),
            gates.get(
                "max_frontier_release_evidence_frontier_rerun_rollup_missing_report_count"
            ),
        ),
        _max_current_metric(
            "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_invalid_report_count.mean",
            _nested_float(
                baseline,
                ("frontier_rerun_rollup_invalid_report_count", "mean"),
            ),
            _nested_float(
                current,
                ("frontier_rerun_rollup_invalid_report_count", "mean"),
            ),
            gates.get(
                "max_frontier_release_evidence_frontier_rerun_rollup_invalid_report_count"
            ),
        ),
        _max_current_metric(
            "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_blocked_candidate_count.mean",
            _nested_float(
                baseline,
                ("frontier_rerun_rollup_blocked_candidate_count", "mean"),
            ),
            _nested_float(
                current,
                ("frontier_rerun_rollup_blocked_candidate_count", "mean"),
            ),
            gates.get(
                "max_frontier_release_evidence_frontier_rerun_rollup_blocked_candidate_count"
            ),
        ),
        _min_current_metric(
            "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_promotion_ready_count.mean",
            _nested_float(
                baseline,
                ("frontier_rerun_rollup_promotion_ready_count", "mean"),
            ),
            _nested_float(
                current,
                ("frontier_rerun_rollup_promotion_ready_count", "mean"),
            ),
            gates.get(
                "min_frontier_release_evidence_frontier_rerun_rollup_promotion_ready_count"
            ),
        ),
        _min_current_metric(
            "promotion_contract.frontier_release_evidence.citation_batch_rollup_count.mean",
            _nested_float(baseline, ("citation_batch_rollup_count", "mean")),
            _nested_float(current, ("citation_batch_rollup_count", "mean")),
            gates.get("min_frontier_release_evidence_citation_batch_rollup_count"),
        ),
        _max_current_metric(
            "promotion_contract.frontier_release_evidence.citation_batch_missing_expected_batch_count.mean",
            _nested_float(
                baseline,
                ("citation_batch_missing_expected_batch_count", "mean"),
            ),
            _nested_float(
                current,
                ("citation_batch_missing_expected_batch_count", "mean"),
            ),
            gates.get(
                "max_frontier_release_evidence_citation_batch_missing_expected_batch_count"
            ),
        ),
        _max_current_metric(
            "promotion_contract.frontier_release_evidence.citation_batch_duplicate_batch_count.mean",
            _nested_float(
                baseline,
                ("citation_batch_duplicate_batch_count", "mean"),
            ),
            _nested_float(
                current,
                ("citation_batch_duplicate_batch_count", "mean"),
            ),
            gates.get(
                "max_frontier_release_evidence_citation_batch_duplicate_batch_count"
            ),
        ),
        _max_current_metric(
            "promotion_contract.frontier_release_evidence.citation_batch_unexpected_batch_count.mean",
            _nested_float(
                baseline,
                ("citation_batch_unexpected_batch_count", "mean"),
            ),
            _nested_float(
                current,
                ("citation_batch_unexpected_batch_count", "mean"),
            ),
            gates.get(
                "max_frontier_release_evidence_citation_batch_unexpected_batch_count"
            ),
        ),
    ]


def _frontier_release_evidence_gate_enabled(gates: Mapping[str, Any]) -> bool:
    return any(
        gates.get(key) is not None
        for key in (
            "min_frontier_release_evidence_coverage",
            "min_frontier_release_evidence_report_present_rate",
            "min_frontier_release_evidence_manifest_present_rate",
            "min_frontier_release_evidence_status_promote_rate",
            "min_frontier_release_evidence_decision_promote_rate",
            "min_frontier_release_evidence_verifier_track_promote_rate",
            "min_frontier_release_evidence_abstention_track_promote_rate",
            "min_frontier_release_evidence_citation_batch_track_promote_rate",
            "min_frontier_release_evidence_frontier_rerun_rollup_track_promote_rate",
            "min_frontier_release_evidence_run_count",
            "min_frontier_release_evidence_frontier_rerun_rollup_report_count",
            "min_frontier_release_evidence_frontier_rerun_rollup_candidate_count",
            "max_frontier_release_evidence_frontier_rerun_rollup_missing_report_count",
            "max_frontier_release_evidence_frontier_rerun_rollup_invalid_report_count",
            "max_frontier_release_evidence_frontier_rerun_rollup_blocked_candidate_count",
            "min_frontier_release_evidence_frontier_rerun_rollup_promotion_ready_count",
            "min_frontier_release_evidence_citation_batch_rollup_count",
            "max_frontier_release_evidence_citation_batch_missing_expected_batch_count",
            "max_frontier_release_evidence_citation_batch_duplicate_batch_count",
            "max_frontier_release_evidence_citation_batch_unexpected_batch_count",
        )
    )


def _manifest_verified_rate(summary: Mapping[str, Any]) -> float | None:
    verified = _finite_float(summary.get("manifest_verified_count"))
    failed = _finite_float(summary.get("manifest_failed_count"))
    unknown = _finite_float(summary.get("manifest_unknown_count"))
    finite = tuple(value for value in (verified, failed, unknown) if value is not None)
    total = sum(finite)
    if total <= 0.0:
        return None
    return (verified or 0.0) / total


def _covered_fact_property_metrics(
    baseline_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not _covered_fact_property_gate_enabled(gates):
        return []
    rows: list[dict[str, Any]] = []
    for scope_name in gates.get("promotion_contract_covered_fact_property_scopes", ()):
        scope_key = _COVERED_FACT_PROPERTY_SCOPES[str(scope_name)]
        rows.extend((
            _min_current_metric(
                f"promotion_contract.covered_fact_properties.{scope_key}.property_metric_count.mean",
                _nested_float(
                    baseline_summary,
                    ("promotion_contract", "covered_fact_properties", scope_key, "property_metric_count", "mean"),
                ),
                _nested_float(
                    current_summary,
                    ("promotion_contract", "covered_fact_properties", scope_key, "property_metric_count", "mean"),
                ),
                gates.get("min_promotion_contract_covered_fact_property_metric_count"),
            ),
            _min_current_metric(
                f"promotion_contract.covered_fact_properties.{scope_key}.min_records.mean",
                _nested_float(
                    baseline_summary,
                    ("promotion_contract", "covered_fact_properties", scope_key, "min_records", "mean"),
                ),
                _nested_float(
                    current_summary,
                    ("promotion_contract", "covered_fact_properties", scope_key, "min_records", "mean"),
                ),
                gates.get("min_promotion_contract_covered_fact_min_records"),
            ),
            _min_current_metric(
                f"promotion_contract.covered_fact_properties.{scope_key}.min_source_documents.mean",
                _nested_float(
                    baseline_summary,
                    ("promotion_contract", "covered_fact_properties", scope_key, "min_source_documents", "mean"),
                ),
                _nested_float(
                    current_summary,
                    ("promotion_contract", "covered_fact_properties", scope_key, "min_source_documents", "mean"),
                ),
                gates.get("min_promotion_contract_covered_fact_min_source_documents"),
            ),
            _drop_metric(
                f"promotion_contract.covered_fact_properties.{scope_key}.min_decision_accuracy.mean",
                _nested_float(
                    baseline_summary,
                    ("promotion_contract", "covered_fact_properties", scope_key, "min_decision_accuracy", "mean"),
                ),
                _nested_float(
                    current_summary,
                    ("promotion_contract", "covered_fact_properties", scope_key, "min_decision_accuracy", "mean"),
                ),
                gates.get("max_promotion_contract_covered_fact_min_decision_accuracy_drop"),
            ),
            _delta_metric(
                f"promotion_contract.covered_fact_properties.{scope_key}.max_false_supported_rate.mean",
                _nested_float(
                    baseline_summary,
                    (
                        "promotion_contract",
                        "covered_fact_properties",
                        scope_key,
                        "max_false_supported_rate",
                        "mean",
                    ),
                ),
                _nested_float(
                    current_summary,
                    (
                        "promotion_contract",
                        "covered_fact_properties",
                        scope_key,
                        "max_false_supported_rate",
                        "mean",
                    ),
                ),
                gates.get("max_promotion_contract_covered_fact_max_false_supported_rate_increase"),
            ),
            _drop_metric(
                f"promotion_contract.covered_fact_properties.{scope_key}.min_false_refuted_rate.mean",
                _nested_float(
                    baseline_summary,
                    ("promotion_contract", "covered_fact_properties", scope_key, "min_false_refuted_rate", "mean"),
                ),
                _nested_float(
                    current_summary,
                    ("promotion_contract", "covered_fact_properties", scope_key, "min_false_refuted_rate", "mean"),
                ),
                gates.get("max_promotion_contract_covered_fact_min_false_refuted_rate_drop"),
            ),
        ))
    return rows


def _covered_fact_property_gate_enabled(gates: Mapping[str, Any]) -> bool:
    return any(
        gates.get(key) is not None
        for key in (
            "min_promotion_contract_covered_fact_property_metric_count",
            "min_promotion_contract_covered_fact_min_records",
            "min_promotion_contract_covered_fact_min_source_documents",
            "max_promotion_contract_covered_fact_min_decision_accuracy_drop",
            "max_promotion_contract_covered_fact_max_false_supported_rate_increase",
            "max_promotion_contract_covered_fact_min_false_refuted_rate_drop",
        )
    )


def _product_trace_action_gate_metrics(
    baseline_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not _product_trace_action_gate_gate_enabled(gates):
        return []
    return [
        _delta_metric(
            metric_name,
            _nested_float(baseline_summary, metric_path),
            _nested_float(current_summary, metric_path),
            gates.get(gate_key),
        )
        for metric_name, metric_path, gate_key in _PRODUCT_TRACE_ACTION_GATE_METRIC_SPECS
    ]


def _product_trace_action_gate_gate_enabled(gates: Mapping[str, Any]) -> bool:
    return any(gates.get(gate_key) is not None for _, _, gate_key in _PRODUCT_TRACE_ACTION_GATE_METRIC_SPECS)


def _product_trace_trajectory_audit_metrics(
    baseline_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not _product_trace_trajectory_audit_gate_enabled(gates):
        return []
    return [
        _delta_metric(
            metric_name,
            _nested_float(baseline_summary, metric_path),
            _nested_float(current_summary, metric_path),
            gates.get(gate_key),
        )
        for metric_name, metric_path, gate_key in _PRODUCT_TRACE_TRAJECTORY_AUDIT_METRIC_SPECS
    ]


def _product_trace_trajectory_audit_gate_enabled(gates: Mapping[str, Any]) -> bool:
    return any(
        gates.get(gate_key) is not None
        for _, _, gate_key in _PRODUCT_TRACE_TRAJECTORY_AUDIT_METRIC_SPECS
    )


def _world_model_metrics(
    baseline_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not _world_model_gate_enabled(gates):
        return []
    rows = [
        _min_current_metric(
            metric_name,
            _nested_float(baseline_summary, metric_path),
            _nested_float(current_summary, metric_path),
            gates.get(gate_key),
        )
        for metric_name, metric_path, gate_key in _WORLD_MODEL_MIN_METRIC_SPECS
    ]
    rows.extend(
        _delta_metric(
            metric_name,
            _nested_float(baseline_summary, metric_path),
            _nested_float(current_summary, metric_path),
            gates.get(gate_key),
        )
        for metric_name, metric_path, gate_key in _WORLD_MODEL_INCREASE_METRIC_SPECS
    )
    return rows


def _world_model_gate_enabled(gates: Mapping[str, Any]) -> bool:
    return any(
        gates.get(gate_key) is not None
        for _, _, gate_key in (
            *_WORLD_MODEL_MIN_METRIC_SPECS,
            *_WORLD_MODEL_INCREASE_METRIC_SPECS,
        )
    )


def _context_sensitivity_metrics(
    baseline_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not _context_sensitivity_gate_enabled(gates):
        return []
    rows = [
        _min_current_metric(
            metric_name,
            _nested_float(baseline_summary, metric_path),
            _nested_float(current_summary, metric_path),
            gates.get(gate_key),
        )
        for metric_name, metric_path, gate_key in _CONTEXT_SENSITIVITY_MIN_METRIC_SPECS
    ]
    rows.extend(
        _delta_metric(
            metric_name,
            _nested_float(baseline_summary, metric_path),
            _nested_float(current_summary, metric_path),
            gates.get(gate_key),
        )
        for metric_name, metric_path, gate_key in _CONTEXT_SENSITIVITY_INCREASE_METRIC_SPECS
    )
    return rows


def _context_sensitivity_gate_enabled(gates: Mapping[str, Any]) -> bool:
    return any(
        gates.get(gate_key) is not None
        for _, _, gate_key in (
            *_CONTEXT_SENSITIVITY_MIN_METRIC_SPECS,
            *_CONTEXT_SENSITIVITY_INCREASE_METRIC_SPECS,
        )
    )


def _counterfactual_robustness_metrics(
    baseline_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not _counterfactual_robustness_gate_enabled(gates):
        return []
    rows = [
        _min_current_metric(
            metric_name,
            _nested_float(baseline_summary, metric_path),
            _nested_float(current_summary, metric_path),
            gates.get(gate_key),
        )
        for metric_name, metric_path, gate_key in _COUNTERFACTUAL_ROBUSTNESS_MIN_METRIC_SPECS
    ]
    rows.extend(
        _delta_metric(
            metric_name,
            _nested_float(baseline_summary, metric_path),
            _nested_float(current_summary, metric_path),
            gates.get(gate_key),
        )
        for metric_name, metric_path, gate_key in _COUNTERFACTUAL_ROBUSTNESS_INCREASE_METRIC_SPECS
    )
    return rows


def _counterfactual_robustness_gate_enabled(gates: Mapping[str, Any]) -> bool:
    return any(
        gates.get(gate_key) is not None
        for _, _, gate_key in (
            *_COUNTERFACTUAL_ROBUSTNESS_MIN_METRIC_SPECS,
            *_COUNTERFACTUAL_ROBUSTNESS_INCREASE_METRIC_SPECS,
        )
    )


def _runtime_optimization_handoff(report: Mapping[str, Any]) -> dict[str, Any]:
    optimization = _mapping(report.get("optimization"))
    if not optimization:
        return {}
    policy_hints = _mapping(optimization.get("policy_hints"))
    return {
        "status": optimization.get("status"),
        "policy_hints": {
            "source": policy_hints.get("source"),
            "candidate_control_defaults": _mapping(
                policy_hints.get("candidate_control_defaults")
            ),
            "candidate_runtime_budget_policy": _mapping(
                policy_hints.get("candidate_runtime_budget_policy")
            ),
        },
    }


def _ratio_metric(
    name: str,
    baseline: float | None,
    current: float | None,
    max_ratio: float | None,
) -> dict[str, Any]:
    ratio = None if baseline in (None, 0.0) or current is None else current / baseline
    row = _base_metric(name, baseline=baseline, current=current)
    row.update({
        "comparison": "max_ratio",
        "ratio_to_baseline": ratio,
        "threshold": max_ratio,
    })
    return _gate_metric(row, value=ratio, threshold=max_ratio, fail=lambda value, threshold: value > threshold)


def _delta_metric(
    name: str,
    baseline: float | None,
    current: float | None,
    max_delta: float | None,
) -> dict[str, Any]:
    delta = None if baseline is None or current is None else current - baseline
    row = _base_metric(name, baseline=baseline, current=current)
    row.update({
        "comparison": "max_increase",
        "absolute_delta": delta,
        "threshold": max_delta,
    })
    return _gate_metric(row, value=delta, threshold=max_delta, fail=lambda value, threshold: value > threshold)


def _drop_metric(
    name: str,
    baseline: float | None,
    current: float | None,
    max_drop: float | None,
) -> dict[str, Any]:
    drop = None if baseline is None or current is None else baseline - current
    row = _base_metric(name, baseline=baseline, current=current)
    row.update({
        "comparison": "max_drop",
        "absolute_drop": drop,
        "threshold": max_drop,
    })
    return _gate_metric(row, value=drop, threshold=max_drop, fail=lambda value, threshold: value > threshold)


def _min_metric(name: str, current: float | None, minimum: int | None) -> dict[str, Any]:
    row = {
        "metric": name,
        "baseline": None,
        "current": current,
        "comparison": "min_current",
        "threshold": minimum,
    }
    return _gate_metric(
        row,
        value=current,
        threshold=None if minimum is None else float(minimum),
        fail=lambda value, threshold: value < threshold,
    )


def _min_current_metric(
    name: str,
    baseline: float | None,
    current: float | None,
    minimum: float | None,
) -> dict[str, Any]:
    row = _base_metric(name, baseline=baseline, current=current)
    row.update({
        "comparison": "min_current",
        "threshold": minimum,
    })
    return _gate_metric(row, value=current, threshold=minimum, fail=lambda value, threshold: value < threshold)


def _max_current_metric(
    name: str,
    baseline: float | None,
    current: float | None,
    maximum: float | None,
) -> dict[str, Any]:
    row = _base_metric(name, baseline=baseline, current=current)
    row.update({
        "comparison": "max_current",
        "threshold": maximum,
    })
    return _gate_metric(row, value=current, threshold=maximum, fail=lambda value, threshold: value > threshold)


def _base_metric(name: str, *, baseline: float | None, current: float | None) -> dict[str, Any]:
    return {
        "metric": name,
        "baseline": baseline,
        "current": current,
        "absolute_delta": None if baseline is None or current is None else current - baseline,
    }


def _gate_metric(
    row: dict[str, Any],
    *,
    value: float | None,
    threshold: float | None,
    fail: Any,
) -> dict[str, Any]:
    if threshold is None:
        row["status"] = "observed"
        row["reason"] = None
        return row
    if value is None:
        row["status"] = "blocked"
        row["reason"] = f"{row['metric']}: missing or non-finite comparison value"
        return row
    if fail(value, threshold):
        row["status"] = "blocked"
        if row.get("comparison") == "min_current":
            row["reason"] = f"{row['metric']}: {value:.6g} below gate {threshold:.6g}"
        elif row.get("comparison") == "max_current":
            row["reason"] = f"{row['metric']}: {value:.6g} above gate {threshold:.6g}"
        else:
            row["reason"] = f"{row['metric']}: {value:.6g} exceeded gate {threshold:.6g}"
        return row
    row["status"] = "pass"
    row["reason"] = None
    return row


def _resolve_baseline_source(
    *,
    baseline_path: str | Path | None,
    registry_path: str | Path | None,
    baseline_key: str | None,
    baseline_name: str | None,
    baseline_version: str | None,
) -> dict[str, Any]:
    if baseline_path is not None:
        if baseline_key or baseline_name or baseline_version:
            raise ValueError("baseline_path is mutually exclusive with registry baseline selection.")
        return {"source": "file", "path": Path(baseline_path)}
    if registry_path is None:
        raise ValueError("provide baseline_path or registry_path with a product_runtime_baseline key.")
    registry = ArtifactRegistry.load_json(registry_path)
    record = _select_runtime_baseline_record(
        registry,
        baseline_key=baseline_key,
        baseline_name=baseline_name,
        baseline_version=baseline_version,
    )
    return {
        "source": "registry",
        "registry": str(registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "path": _resolve_record_path(Path(registry_path), record),
    }


def _resolve_runtime_budget_policy_source(
    *,
    runtime_budget_policy_path: str | Path | None,
    registry_path: str | Path | None,
    runtime_budget_policy_key: str | None,
) -> dict[str, Any] | None:
    if runtime_budget_policy_path is None and runtime_budget_policy_key is None:
        return None
    if runtime_budget_policy_path is not None and runtime_budget_policy_key is not None:
        raise ValueError("runtime_budget_policy_path and runtime_budget_policy_key are mutually exclusive.")
    if runtime_budget_policy_path is not None:
        return {"source": "file", "path": Path(runtime_budget_policy_path)}
    if registry_path is None:
        raise ValueError("runtime_budget_policy_key requires registry_path.")
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get(str(runtime_budget_policy_key))
    if record.artifact_type != "product_runtime_budget_policy":
        raise ValueError(f"registry record {record.key()!r} is not a product_runtime_budget_policy.")
    return {
        "source": "registry",
        "registry": str(registry_path),
        "record_key": record.key(),
        "record": record.to_dict(),
        "path": _resolve_record_path(Path(registry_path), record),
    }


def _select_runtime_baseline_record(
    registry: ArtifactRegistry,
    *,
    baseline_key: str | None,
    baseline_name: str | None,
    baseline_version: str | None,
) -> RegistryRecord:
    if baseline_key:
        record = registry.get(baseline_key)
    else:
        if not baseline_name or not baseline_version:
            raise ValueError("provide --baseline-key or both --baseline-name and --baseline-version.")
        record = registry.get(f"product_runtime_baseline:{baseline_name}:{baseline_version}")
    if record.artifact_type != "product_runtime_baseline":
        raise ValueError(f"registry record {record.key()!r} is not a product_runtime_baseline.")
    return record


def _resolve_record_path(registry_path: Path, record: RegistryRecord) -> Path:
    path = Path(record.path)
    if path.is_absolute():
        return path
    sibling = registry_path.parent / path
    return sibling if sibling.exists() else path


def _load_runtime_baseline(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"runtime baseline report must be a JSON object: {path}")
    if payload.get("workflow") != "product_runtime_baseline":
        raise ValueError(f"runtime baseline report has unexpected workflow: {path}")
    if not isinstance(payload.get("summary"), Mapping):
        raise ValueError(f"runtime baseline report is missing summary: {path}")
    return dict(payload)


def _runtime_budget_policy_gate(
    current_summary: Mapping[str, Any],
    *,
    source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if source is None:
        return {
            "enabled": False,
            "passed": None,
            "source": None,
            "aggregation": "current_product_runtime_baseline_summary",
            "policy": None,
            "policy_metadata": {},
            "check_count": 0,
            "failed_count": 0,
            "checks": (),
            "failures": (),
        }
    payload = _load_runtime_budget_policy(source["path"])
    policy = ProductRuntimeBudgetPolicy.from_mapping(payload)
    checks = tuple(_runtime_budget_policy_checks(current_summary, policy))
    failures = tuple(check for check in checks if check.get("status") == "blocked")
    return {
        "enabled": policy.enabled(),
        "passed": None if not policy.enabled() else not failures,
        "source": _jsonable_source(source),
        "aggregation": "current_product_runtime_baseline_summary",
        "policy": policy.to_dict(),
        "policy_metadata": dict(_mapping(payload.get("metadata"))),
        "check_count": len(checks),
        "failed_count": len(failures),
        "checks": checks,
        "failures": failures,
    }


def _runtime_budget_policy_checks(
    summary: Mapping[str, Any],
    policy: ProductRuntimeBudgetPolicy,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if not policy.enabled():
        return checks
    if policy.require_runtime_trace and _policy_requires_runtime_trace(policy):
        n_traces = _finite_float(summary.get("n_traces"))
        runtime_trace_count = _finite_float(summary.get("runtime_trace_count"))
        if n_traces is not None:
            checks.append(_policy_min_check(
                "runtime_trace_count",
                runtime_trace_count,
                n_traces,
                policy_field="require_runtime_trace",
            ))
    checks.extend((
        _policy_max_check(
            "total_seconds.p95",
            _nested_float(summary, ("total_seconds", "p95")),
            policy.max_total_seconds,
            policy_field="max_total_seconds",
        ),
        _policy_max_check(
            "route.mean_duration_seconds",
            _first_nested_float(
                summary,
                (
                    ("routes", "overall", "mean_duration_seconds"),
                    ("mean_route_duration_seconds", "mean"),
                ),
            ),
            policy.max_mean_route_duration_seconds,
            policy_field="max_mean_route_duration_seconds",
        ),
        _policy_max_check(
            "route.p95_duration_seconds.p95",
            _first_nested_float(
                summary,
                (
                    ("p95_route_duration_seconds", "p95"),
                    ("routes", "overall", "per_trace_p95_duration_seconds", "p95"),
                ),
            ),
            policy.max_p95_route_duration_seconds,
            policy_field="max_p95_route_duration_seconds",
        ),
        _policy_max_check(
            "route.p99_duration_seconds.p99",
            _first_nested_float(
                summary,
                (
                    ("p99_route_duration_seconds", "p99"),
                    ("routes", "overall", "per_trace_p99_duration_seconds", "p99"),
                ),
            ),
            policy.max_p99_route_duration_seconds,
            policy_field="max_p99_route_duration_seconds",
        ),
        _policy_max_check(
            "route.max_duration_seconds",
            _first_nested_float(
                summary,
                (
                    ("routes", "overall", "max_duration_seconds"),
                    ("max_route_duration_seconds", "max"),
                ),
            ),
            policy.max_route_duration_seconds,
            policy_field="max_route_duration_seconds",
        ),
        _policy_max_check(
            "route.mean_attempted_route_count",
            _first_nested_float(
                summary,
                (
                    ("routes", "overall", "mean_attempted_route_count"),
                    ("mean_attempted_route_count", "mean"),
                ),
            ),
            policy.max_mean_attempted_route_count,
            policy_field="max_mean_attempted_route_count",
        ),
        _policy_max_check(
            "route.route_budget_exhaustion_rate",
            _first_nested_float(
                summary,
                (
                    ("routes", "overall", "route_budget_exhaustion_rate"),
                    ("route_budget_exhaustion_rate", "mean"),
                ),
            ),
            policy.max_route_budget_exhaustion_rate,
            policy_field="max_route_budget_exhaustion_rate",
        ),
        _policy_max_check(
            "route.retrieval_use_rate",
            _first_nested_float(
                summary,
                (
                    ("routes", "overall", "retrieval_use_rate"),
                    ("retrieval_use_rate", "mean"),
                ),
            ),
            policy.max_retrieval_use_rate,
            policy_field="max_retrieval_use_rate",
        ),
        _policy_max_check(
            "retrieval_hit_count.p95",
            _first_nested_float(
                summary,
                (
                    ("retrieval_hit_count", "p95"),
                    ("routes", "overall", "mean_retrieval_hits"),
                ),
            ),
            policy.max_retrieval_hit_count,
            policy_field="max_retrieval_hit_count",
        ),
        _policy_min_check(
            "cache_hit_rate.mean",
            _nested_float(summary, ("cache_hit_rate", "mean")),
            policy.min_cache_hit_rate,
            policy_field="min_cache_hit_rate",
        ),
        _policy_min_check(
            "verification_skip_rate.mean",
            _nested_float(summary, ("verification_skip_rate", "mean")),
            policy.min_verification_skip_rate,
            policy_field="min_verification_skip_rate",
        ),
        _policy_min_check(
            "verification_stage.selective_claim_skip_rate",
            _nested_float(summary, ("verification_stage", "selective_claim_skip_rate")),
            policy.min_selective_claim_skip_rate,
            policy_field="min_selective_claim_skip_rate",
        ),
        _policy_max_check(
            "verified_claim_count.p95",
            _nested_float(summary, ("verified_claim_count", "p95")),
            policy.max_verified_claim_count,
            policy_field="max_verified_claim_count",
        ),
    ))
    for phase_name, limit in policy.max_phase_seconds.items():
        checks.append(_policy_max_check(
            f"phase_seconds.{phase_name}.max",
            _nested_float(summary, ("phases", phase_name, "seconds", "max")),
            limit,
            policy_field=f"max_phase_seconds.{phase_name}",
        ))
    for phase_name, limit in policy.max_phase_p95_seconds.items():
        checks.append(_policy_max_check(
            f"phase_seconds.{phase_name}.p95",
            _nested_float(summary, ("phases", phase_name, "seconds", "p95")),
            limit,
            policy_field=f"max_phase_p95_seconds.{phase_name}",
        ))
    for phase_name, limit in policy.max_phase_p99_seconds.items():
        checks.append(_policy_max_check(
            f"phase_seconds.{phase_name}.p99",
            _nested_float(summary, ("phases", phase_name, "seconds", "p99")),
            limit,
            policy_field=f"max_phase_p99_seconds.{phase_name}",
        ))
    for cache_name, limit in policy.min_named_cache_hit_rate.items():
        checks.append(_policy_min_check(
            f"named_cache_hit_rate.{cache_name}.mean",
            _nested_float(summary, ("named_cache_hit_rates", cache_name, "mean")),
            limit,
            policy_field=f"min_named_cache_hit_rate.{cache_name}",
        ))
    return [check for check in checks if check.get("limit") is not None]


def _policy_max_check(
    metric: str,
    value: float | None,
    limit: float | None,
    *,
    policy_field: str,
) -> dict[str, Any]:
    return _policy_gate_check(
        metric,
        value,
        limit,
        policy_field=policy_field,
        limit_type="max",
        failed=lambda observed, threshold: observed > threshold,
        failure_text="exceeded",
    )


def _policy_min_check(
    metric: str,
    value: float | None,
    limit: float | None,
    *,
    policy_field: str,
) -> dict[str, Any]:
    return _policy_gate_check(
        metric,
        value,
        limit,
        policy_field=policy_field,
        limit_type="min",
        failed=lambda observed, threshold: observed < threshold,
        failure_text="below",
    )


def _policy_gate_check(
    metric: str,
    value: float | None,
    limit: float | None,
    *,
    policy_field: str,
    limit_type: str,
    failed: Any,
    failure_text: str,
) -> dict[str, Any]:
    row = {
        "metric": metric,
        "policy_field": policy_field,
        "limit_type": limit_type,
        "limit": limit,
        "value": value,
        "status": "disabled" if limit is None else "pass",
        "reason": None,
    }
    if limit is None:
        return row
    if value is None:
        row["status"] = "blocked"
        row["reason"] = f"runtime_budget_policy: {metric} missing or non-finite in current baseline summary"
        return row
    if failed(value, limit):
        row["status"] = "blocked"
        row["reason"] = f"runtime_budget_policy: {metric} {value:.6g} {failure_text} gate {limit:.6g}"
    return row


def _policy_requires_runtime_trace(policy: ProductRuntimeBudgetPolicy) -> bool:
    return (
        policy.max_total_seconds is not None
        or bool(policy.max_phase_seconds)
        or bool(policy.max_phase_p95_seconds)
        or bool(policy.max_phase_p99_seconds)
    )


def _load_runtime_budget_policy(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"runtime budget policy must be a JSON object: {path}")
    return dict(payload)


def _jsonable_source(source: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in source.items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def _write_outputs(
    report: Mapping[str, Any],
    *,
    baseline_path: Path,
    current_path: Path,
    report_path: Path | None,
    artifact_manifest_path: Path | None,
    registry_path: Path | None,
    name: str | None,
    version: str | None,
    compact: bool,
) -> dict[str, Any]:
    if (name or version) and (registry_path is None or report_path is None):
        raise ValueError("recording a drift report requires registry_path, report_path, name, and version.")
    if (name is None) != (version is None):
        raise ValueError("registry drift report recording requires both name and version.")
    output = dict(report)
    if report_path is None:
        return output
    runtime_budget_policy_gate = _mapping(report.get("runtime_budget_policy_gate"))
    runtime_budget_policy_source = _mapping(runtime_budget_policy_gate.get("source"))
    runtime_budget_policy_path = runtime_budget_policy_source.get("path")
    artifacts = {
        "product_runtime_drift_report": report_path,
        "baseline_product_runtime_baseline": baseline_path,
        "current_product_runtime_baseline": current_path,
    }
    if runtime_budget_policy_path is not None:
        artifacts["runtime_budget_policy"] = Path(str(runtime_budget_policy_path))
    drift_metadata = _drift_metadata(report)
    manifest_summary = planned_artifact_manifest_summary(
        artifacts,
        assume_file_paths=(report_path,),
    )
    output["artifact_manifest_summary"] = manifest_summary
    _write_json(report_path, output, compact=compact)
    if artifact_manifest_path is not None:
        manifest = build_artifact_manifest(
            artifacts,
            root=artifact_manifest_path.parent,
            metadata={
                "runner": "compare_product_runtime_baselines",
                "status": report.get("status"),
                "baseline_path": str(baseline_path),
                "current_path": str(current_path),
                "runtime_budget_policy_path": runtime_budget_policy_path,
                "runtime_budget_policy_key": runtime_budget_policy_source.get("record_key"),
                "runtime_budget_policy_gate_enabled": runtime_budget_policy_gate.get("enabled"),
                "runtime_budget_policy_passed": runtime_budget_policy_gate.get("passed"),
                "runtime_budget_policy_failed_count": runtime_budget_policy_gate.get("failed_count"),
                "compact_json": compact,
                **drift_metadata,
            },
        )
        _write_json(artifact_manifest_path, manifest, compact=compact)
    if name and version and registry_path is not None:
        registry = ArtifactRegistry.load_json(registry_path)
        registry.record_product_runtime_drift_report(
            name=name,
            path=report_path,
            version=version,
            metadata={
                "workflow": "compare_product_runtime_baselines",
                "status": report.get("status"),
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                "runtime_budget_policy_path": runtime_budget_policy_path,
                "runtime_budget_policy_key": runtime_budget_policy_source.get("record_key"),
                "runtime_budget_policy_gate_enabled": runtime_budget_policy_gate.get("enabled"),
                "runtime_budget_policy_passed": runtime_budget_policy_gate.get("passed"),
                "runtime_budget_policy_failed_count": runtime_budget_policy_gate.get("failed_count"),
                "compact_json": compact,
                **drift_metadata,
            },
        )
        registry.save_json()
    return output


def _drift_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    return {
        "gate_enabled": summary.get("gate_enabled"),
        "drift_gate_enabled": summary.get("drift_gate_enabled"),
        "compared_metric_count": summary.get("compared_metric_count"),
        "blocked_metric_count": summary.get("blocked_metric_count"),
        "observed_metric_count": summary.get("observed_metric_count"),
        **_promotion_evidence_metadata(report),
        **_pre_generation_probe_comparison_metadata(report),
        **_claim_factuality_probe_comparison_metadata(report),
        **_counterfactual_verification_metadata(report),
        **_evidence_handoff_metadata(report),
        **_frontier_release_evidence_metadata(report),
        **_covered_fact_property_metadata(report),
        **_triple_coverage_metadata(report),
        **_world_model_metadata(report),
        **_context_sensitivity_metadata(report),
        **_counterfactual_robustness_metadata(report),
        **_product_trace_action_gate_metadata(report),
        **_product_trace_trajectory_audit_metadata(report),
    }


def _promotion_evidence_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics_by_name(report.get("metrics"))
    metadata: dict[str, Any] = {
        "promotion_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PROMOTION_EVIDENCE_METADATA_FIELDS:
        metric = metrics.get(metric_name)
        metadata[f"{prefix}_baseline"] = _finite_float(None if metric is None else metric.get("baseline"))
        metadata[f"{prefix}_current"] = _finite_float(None if metric is None else metric.get("current"))
        metadata[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is not None and metric.get("status") == "blocked":
            metadata["promotion_evidence_blocked_metric_count"] += 1
    return metadata


def _pre_generation_probe_comparison_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics_by_name(report.get("metrics"))
    metadata: dict[str, Any] = {
        "pre_generation_probe_comparison_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRE_GENERATION_PROBE_COMPARISON_METADATA_FIELDS:
        metric = metrics.get(metric_name)
        metadata[f"{prefix}_baseline"] = _finite_float(None if metric is None else metric.get("baseline"))
        metadata[f"{prefix}_current"] = _finite_float(None if metric is None else metric.get("current"))
        metadata[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is not None and metric.get("status") == "blocked":
            metadata["pre_generation_probe_comparison_blocked_metric_count"] += 1
    return metadata


def _claim_factuality_probe_comparison_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics_by_name(report.get("metrics"))
    metadata: dict[str, Any] = {
        "claim_factuality_probe_comparison_blocked_metric_count": 0,
    }
    for metric_name, prefix in _CLAIM_FACTUALITY_PROBE_COMPARISON_METADATA_FIELDS:
        metric = metrics.get(metric_name)
        metadata[f"{prefix}_baseline"] = _finite_float(
            None if metric is None else metric.get("baseline")
        )
        metadata[f"{prefix}_current"] = _finite_float(
            None if metric is None else metric.get("current")
        )
        metadata[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is not None and metric.get("status") == "blocked":
            metadata["claim_factuality_probe_comparison_blocked_metric_count"] += 1
    return metadata


def _counterfactual_verification_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics_by_name(report.get("metrics"))
    metadata: dict[str, Any] = {
        "counterfactual_verification_blocked_metric_count": 0,
    }
    for metric_name, prefix in _COUNTERFACTUAL_VERIFICATION_METADATA_FIELDS:
        metric = metrics.get(metric_name)
        metadata[f"{prefix}_baseline"] = _finite_float(None if metric is None else metric.get("baseline"))
        metadata[f"{prefix}_current"] = _finite_float(None if metric is None else metric.get("current"))
        metadata[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is not None and metric.get("status") == "blocked":
            metadata["counterfactual_verification_blocked_metric_count"] += 1
    return metadata


def _evidence_handoff_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics_by_name(report.get("metrics"))
    metadata: dict[str, Any] = {
        "evidence_handoff_blocked_metric_count": 0,
    }
    for metric_name, prefix in _EVIDENCE_HANDOFF_METADATA_FIELDS:
        metric = metrics.get(metric_name)
        metadata[f"{prefix}_baseline"] = _finite_float(None if metric is None else metric.get("baseline"))
        metadata[f"{prefix}_current"] = _finite_float(None if metric is None else metric.get("current"))
        metadata[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is not None and metric.get("status") == "blocked":
            metadata["evidence_handoff_blocked_metric_count"] += 1
    return metadata


def _frontier_release_evidence_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics_by_name(report.get("metrics"))
    metadata: dict[str, Any] = {
        "frontier_release_evidence_blocked_metric_count": 0,
    }
    for metric_name, prefix in _FRONTIER_RELEASE_EVIDENCE_METADATA_FIELDS:
        metric = metrics.get(metric_name)
        metadata[f"{prefix}_baseline"] = _finite_float(
            None if metric is None else metric.get("baseline")
        )
        metadata[f"{prefix}_current"] = _finite_float(
            None if metric is None else metric.get("current")
        )
        metadata[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is not None and metric.get("status") == "blocked":
            metadata["frontier_release_evidence_blocked_metric_count"] += 1
    return metadata


def _covered_fact_property_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics_by_name(report.get("metrics"))
    metadata: dict[str, Any] = {
        "covered_fact_property_blocked_metric_count": 0,
    }
    for metric_name, prefix in _COVERED_FACT_PROPERTY_METADATA_FIELDS:
        metric = metrics.get(metric_name)
        metadata[f"{prefix}_baseline"] = _finite_float(None if metric is None else metric.get("baseline"))
        metadata[f"{prefix}_current"] = _finite_float(None if metric is None else metric.get("current"))
        metadata[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is not None and metric.get("status") == "blocked":
            metadata["covered_fact_property_blocked_metric_count"] += 1
    return metadata


def _triple_coverage_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics_by_name(report.get("metrics"))
    metadata: dict[str, Any] = {
        "triple_coverage_blocked_metric_count": 0,
    }
    for metric_name, prefix in _TRIPLE_COVERAGE_METADATA_FIELDS:
        metric = metrics.get(metric_name)
        metadata[f"{prefix}_baseline"] = _finite_float(None if metric is None else metric.get("baseline"))
        metadata[f"{prefix}_current"] = _finite_float(None if metric is None else metric.get("current"))
        metadata[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is not None and metric.get("status") == "blocked":
            metadata["triple_coverage_blocked_metric_count"] += 1
    return metadata


def _world_model_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics_by_name(report.get("metrics"))
    metadata: dict[str, Any] = {
        "world_model_blocked_metric_count": 0,
    }
    for metric_name, prefix in _WORLD_MODEL_METADATA_FIELDS:
        metric = metrics.get(metric_name)
        metadata[f"{prefix}_baseline"] = _finite_float(None if metric is None else metric.get("baseline"))
        metadata[f"{prefix}_current"] = _finite_float(None if metric is None else metric.get("current"))
        metadata[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is not None and metric.get("status") == "blocked":
            metadata["world_model_blocked_metric_count"] += 1
    return metadata


def _context_sensitivity_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics_by_name(report.get("metrics"))
    metadata: dict[str, Any] = {
        "context_sensitivity_blocked_metric_count": 0,
    }
    for metric_name, prefix in _CONTEXT_SENSITIVITY_METADATA_FIELDS:
        metric = metrics.get(metric_name)
        metadata[f"{prefix}_baseline"] = _finite_float(
            None if metric is None else metric.get("baseline")
        )
        metadata[f"{prefix}_current"] = _finite_float(
            None if metric is None else metric.get("current")
        )
        metadata[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is not None and metric.get("status") == "blocked":
            metadata["context_sensitivity_blocked_metric_count"] += 1
    return metadata


def _counterfactual_robustness_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics_by_name(report.get("metrics"))
    metadata: dict[str, Any] = {
        "counterfactual_robustness_blocked_metric_count": 0,
    }
    for metric_name, prefix in _COUNTERFACTUAL_ROBUSTNESS_METADATA_FIELDS:
        metric = metrics.get(metric_name)
        metadata[f"{prefix}_baseline"] = _finite_float(
            None if metric is None else metric.get("baseline")
        )
        metadata[f"{prefix}_current"] = _finite_float(
            None if metric is None else metric.get("current")
        )
        metadata[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is not None and metric.get("status") == "blocked":
            metadata["counterfactual_robustness_blocked_metric_count"] += 1
    return metadata


def _product_trace_action_gate_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics_by_name(report.get("metrics"))
    metadata: dict[str, Any] = {
        "product_trace_action_gate_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_TRACE_ACTION_GATE_METADATA_FIELDS:
        metric = metrics.get(metric_name)
        metadata[f"{prefix}_baseline"] = _finite_float(None if metric is None else metric.get("baseline"))
        metadata[f"{prefix}_current"] = _finite_float(None if metric is None else metric.get("current"))
        metadata[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is not None and metric.get("status") == "blocked":
            metadata["product_trace_action_gate_blocked_metric_count"] += 1
    return metadata


def _product_trace_trajectory_audit_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics_by_name(report.get("metrics"))
    metadata: dict[str, Any] = {
        "product_trace_trajectory_audit_blocked_metric_count": 0,
    }
    for metric_name, prefix in _PRODUCT_TRACE_TRAJECTORY_AUDIT_METADATA_FIELDS:
        metric = metrics.get(metric_name)
        metadata[f"{prefix}_baseline"] = _finite_float(None if metric is None else metric.get("baseline"))
        metadata[f"{prefix}_current"] = _finite_float(None if metric is None else metric.get("current"))
        metadata[f"{prefix}_status"] = None if metric is None else metric.get("status")
        if metric is not None and metric.get("status") == "blocked":
            metadata["product_trace_trajectory_audit_blocked_metric_count"] += 1
    return metadata


def _metrics_by_name(value: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    for item in value:
        if isinstance(item, Mapping) and isinstance(item.get("metric"), str):
            rows[str(item["metric"])] = item
    return rows


def _artifact_manifest_path(
    *,
    report_path: Path | None,
    artifact_manifest_path: Path | None,
) -> Path | None:
    if artifact_manifest_path is not None:
        return artifact_manifest_path
    if report_path is None:
        return None
    return report_path.with_name("product-runtime-drift-artifact-manifest.json")


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output_path.write_text(text, encoding="utf-8")


def _nested_float(payload: Mapping[str, Any], path: Sequence[str]) -> float | None:
    return _finite_float(_nested_value(payload, path))


def _nested_value(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _first_nested_float(
    payload: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
) -> float | None:
    for path in paths:
        value = _nested_float(payload, path)
        if value is not None:
            return value
    return None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _optional_non_negative_float(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError("drift gate values must be finite and non-negative.")
    return numeric


def _optional_rate_float(value: float | None) -> float | None:
    numeric = _optional_non_negative_float(value)
    if numeric is not None and numeric > 1.0:
        raise ValueError("rate drift gate values must be between 0 and 1.")
    return numeric


def _optional_non_negative_int(value: int | None) -> int | None:
    if value is None:
        return None
    numeric = int(value)
    if numeric < 0:
        raise ValueError("min_current_trace_count must be non-negative.")
    return numeric


def _covered_fact_property_scopes(scopes: Sequence[str] | None) -> tuple[str, ...]:
    if scopes is None:
        return ("recommended_route",)
    normalized = tuple(str(scope).strip() for scope in scopes if str(scope).strip())
    if not normalized:
        return ("recommended_route",)
    invalid = tuple(scope for scope in normalized if scope not in _COVERED_FACT_PROPERTY_SCOPES)
    if invalid:
        allowed = ", ".join(sorted(_COVERED_FACT_PROPERTY_SCOPES))
        raise ValueError(
            f"unknown promotion-contract covered-fact property scope(s): {invalid}; allowed: {allowed}"
        )
    return normalized


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _count_rate(counts: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    values = [_finite_float(value) for value in counts.values()]
    finite_values = tuple(value for value in values if value is not None)
    total = sum(finite_values)
    if total <= 0.0:
        return None
    numerator = sum(_finite_float(counts.get(key)) or 0.0 for key in keys)
    return numerator / total


def _parse_metadata(values: Sequence[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--metadata entries must be formatted as key=value.")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--metadata keys must be non-empty.")
        try:
            metadata[key] = json.loads(raw)
        except json.JSONDecodeError:
            metadata[key] = raw
    return metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = compare_product_runtime_baselines(
        baseline_path=Path(args.baseline) if args.baseline else None,
        current_path=Path(args.current),
        registry_path=Path(args.registry) if args.registry else None,
        baseline_key=args.baseline_key,
        baseline_name=args.baseline_name,
        baseline_version=args.baseline_version,
        runtime_budget_policy_path=(
            Path(args.runtime_budget_policy) if args.runtime_budget_policy else None
        ),
        runtime_budget_policy_key=args.runtime_budget_policy_key,
        report_path=Path(args.json) if args.json else None,
        artifact_manifest_path=Path(args.artifact_manifest) if args.artifact_manifest else None,
        name=args.name,
        version=args.version,
        max_total_seconds_mean_ratio=args.max_total_seconds_mean_ratio,
        max_total_seconds_p95_ratio=args.max_total_seconds_p95_ratio,
        max_mean_route_duration_ratio=args.max_mean_route_duration_ratio,
        max_p95_route_duration_ratio=args.max_p95_route_duration_ratio,
        max_mean_attempted_route_count_delta=args.max_mean_attempted_route_count_delta,
        max_retrieval_use_rate_delta=args.max_retrieval_use_rate_delta,
        max_cache_hit_rate_drop=args.max_cache_hit_rate_drop,
        max_verification_skip_rate_drop=args.max_verification_skip_rate_drop,
        min_promotion_contract_coverage=args.min_promotion_contract_coverage,
        min_pre_generation_probe_comparison_coverage=(
            args.min_pre_generation_probe_comparison_coverage
        ),
        min_pre_generation_probe_comparison_manifest_verified_rate=(
            args.min_pre_generation_probe_comparison_manifest_verified_rate
        ),
        min_pre_generation_probe_comparison_model_count=(
            args.min_pre_generation_probe_comparison_model_count
        ),
        min_pre_generation_probe_comparison_run_count=(
            args.min_pre_generation_probe_comparison_run_count
        ),
        min_pre_generation_probe_comparison_redline_pass_rate=(
            args.min_pre_generation_probe_comparison_redline_pass_rate
        ),
        max_pre_generation_probe_comparison_best_test_label_auroc_drop=(
            args.max_pre_generation_probe_comparison_best_test_label_auroc_drop
        ),
        max_pre_generation_probe_comparison_best_redline_auroc_drop=(
            args.max_pre_generation_probe_comparison_best_redline_auroc_drop
        ),
        max_pre_generation_probe_comparison_best_redline_margin_drop=(
            args.max_pre_generation_probe_comparison_best_redline_margin_drop
        ),
        min_claim_factuality_probe_comparison_coverage=(
            args.min_claim_factuality_probe_comparison_coverage
        ),
        min_claim_factuality_probe_comparison_manifest_verified_rate=(
            args.min_claim_factuality_probe_comparison_manifest_verified_rate
        ),
        min_claim_factuality_probe_comparison_model_count=(
            args.min_claim_factuality_probe_comparison_model_count
        ),
        min_claim_factuality_probe_comparison_run_count=(
            args.min_claim_factuality_probe_comparison_run_count
        ),
        min_claim_factuality_probe_comparison_redline_pass_rate=(
            args.min_claim_factuality_probe_comparison_redline_pass_rate
        ),
        max_claim_factuality_probe_comparison_best_test_label_auroc_drop=(
            args.max_claim_factuality_probe_comparison_best_test_label_auroc_drop
        ),
        max_claim_factuality_probe_comparison_best_test_selective_accuracy_drop=(
            args.max_claim_factuality_probe_comparison_best_test_selective_accuracy_drop
        ),
        max_claim_factuality_probe_comparison_best_test_selective_coverage_drop=(
            args.max_claim_factuality_probe_comparison_best_test_selective_coverage_drop
        ),
        max_claim_factuality_probe_comparison_best_redline_auroc_drop=(
            args.max_claim_factuality_probe_comparison_best_redline_auroc_drop
        ),
        max_claim_factuality_probe_comparison_best_redline_margin_drop=(
            args.max_claim_factuality_probe_comparison_best_redline_margin_drop
        ),
        min_counterfactual_verification_coverage=(
            args.min_counterfactual_verification_coverage
        ),
        min_counterfactual_verification_manifest_verified_rate=(
            args.min_counterfactual_verification_manifest_verified_rate
        ),
        min_counterfactual_verification_record_count=(
            args.min_counterfactual_verification_record_count
        ),
        min_counterfactual_verification_pass_rate=(
            args.min_counterfactual_verification_pass_rate
        ),
        max_counterfactual_verification_false_invariance_rate=(
            args.max_counterfactual_verification_false_invariance_rate
        ),
        max_counterfactual_verification_flip_success_count_drop=(
            args.max_counterfactual_verification_flip_success_count_drop
        ),
        min_evidence_handoff_coverage=args.min_evidence_handoff_coverage,
        min_evidence_handoff_manifest_verified_rate=(
            args.min_evidence_handoff_manifest_verified_rate
        ),
        min_evidence_handoff_present_metric_rate=(
            args.min_evidence_handoff_present_metric_rate
        ),
        max_evidence_handoff_missing_metric_rate=(
            args.max_evidence_handoff_missing_metric_rate
        ),
        max_evidence_handoff_missing_metric_count=(
            args.max_evidence_handoff_missing_metric_count
        ),
        max_evidence_handoff_blocked_group_count=(
            args.max_evidence_handoff_blocked_group_count
        ),
        min_evidence_handoff_promoted_group_rate=(
            args.min_evidence_handoff_promoted_group_rate
        ),
        min_frontier_release_evidence_coverage=(
            args.min_frontier_release_evidence_coverage
        ),
        min_frontier_release_evidence_report_present_rate=(
            args.min_frontier_release_evidence_report_present_rate
        ),
        min_frontier_release_evidence_manifest_present_rate=(
            args.min_frontier_release_evidence_manifest_present_rate
        ),
        min_frontier_release_evidence_status_promote_rate=(
            args.min_frontier_release_evidence_status_promote_rate
        ),
        min_frontier_release_evidence_decision_promote_rate=(
            args.min_frontier_release_evidence_decision_promote_rate
        ),
        min_frontier_release_evidence_verifier_track_promote_rate=(
            args.min_frontier_release_evidence_verifier_track_promote_rate
        ),
        min_frontier_release_evidence_abstention_track_promote_rate=(
            args.min_frontier_release_evidence_abstention_track_promote_rate
        ),
        min_frontier_release_evidence_citation_batch_track_promote_rate=(
            args.min_frontier_release_evidence_citation_batch_track_promote_rate
        ),
        min_frontier_release_evidence_frontier_rerun_rollup_track_promote_rate=(
            args.min_frontier_release_evidence_frontier_rerun_rollup_track_promote_rate
        ),
        min_frontier_release_evidence_run_count=(
            args.min_frontier_release_evidence_run_count
        ),
        min_frontier_release_evidence_frontier_rerun_rollup_report_count=(
            args.min_frontier_release_evidence_frontier_rerun_rollup_report_count
        ),
        min_frontier_release_evidence_frontier_rerun_rollup_candidate_count=(
            args.min_frontier_release_evidence_frontier_rerun_rollup_candidate_count
        ),
        max_frontier_release_evidence_frontier_rerun_rollup_missing_report_count=(
            args.max_frontier_release_evidence_frontier_rerun_rollup_missing_report_count
        ),
        max_frontier_release_evidence_frontier_rerun_rollup_invalid_report_count=(
            args.max_frontier_release_evidence_frontier_rerun_rollup_invalid_report_count
        ),
        max_frontier_release_evidence_frontier_rerun_rollup_blocked_candidate_count=(
            args.max_frontier_release_evidence_frontier_rerun_rollup_blocked_candidate_count
        ),
        min_frontier_release_evidence_frontier_rerun_rollup_promotion_ready_count=(
            args.min_frontier_release_evidence_frontier_rerun_rollup_promotion_ready_count
        ),
        min_frontier_release_evidence_citation_batch_rollup_count=(
            args.min_frontier_release_evidence_citation_batch_rollup_count
        ),
        max_frontier_release_evidence_citation_batch_missing_expected_batch_count=(
            args.max_frontier_release_evidence_citation_batch_missing_expected_batch_count
        ),
        max_frontier_release_evidence_citation_batch_duplicate_batch_count=(
            args.max_frontier_release_evidence_citation_batch_duplicate_batch_count
        ),
        max_frontier_release_evidence_citation_batch_unexpected_batch_count=(
            args.max_frontier_release_evidence_citation_batch_unexpected_batch_count
        ),
        min_triple_extraction_fixture_matrix_coverage=(
            args.min_triple_extraction_fixture_matrix_coverage
        ),
        max_triple_extraction_fixture_matrix_mean_best_f1_drop=(
            args.max_triple_extraction_fixture_matrix_mean_best_f1_drop
        ),
        max_triple_extraction_fixture_matrix_mean_f1_lift_drop=(
            args.max_triple_extraction_fixture_matrix_mean_f1_lift_drop
        ),
        min_triple_claim_coverage=args.min_triple_claim_coverage,
        min_triple_audit_claim_coverage=args.min_triple_audit_claim_coverage,
        min_triple_audit_pass_rate=args.min_triple_audit_pass_rate,
        min_triple_slot_coverage=args.min_triple_slot_coverage,
        min_world_model_participating_trace_rate=(
            args.min_world_model_participating_trace_rate
        ),
        min_world_model_coverage_rate=args.min_world_model_coverage_rate,
        max_world_model_conflict_rate_increase=(
            args.max_world_model_conflict_rate_increase
        ),
        max_world_model_low_agreement_rate_increase=(
            args.max_world_model_low_agreement_rate_increase
        ),
        max_world_model_trace_gap_rate_increase=(
            args.max_world_model_trace_gap_rate_increase
        ),
        min_context_sensitivity_participating_trace_rate=(
            args.min_context_sensitivity_participating_trace_rate
        ),
        min_context_sensitivity_coverage_rate=(
            args.min_context_sensitivity_coverage_rate
        ),
        max_context_sensitivity_flagged_result_rate_increase=(
            args.max_context_sensitivity_flagged_result_rate_increase
        ),
        max_context_sensitivity_trace_gap_rate_increase=(
            args.max_context_sensitivity_trace_gap_rate_increase
        ),
        max_context_sensitivity_max_flagged_rate_increase=(
            args.max_context_sensitivity_max_flagged_rate_increase
        ),
        max_context_sensitivity_max_ratio_increase=(
            args.max_context_sensitivity_max_ratio_increase
        ),
        min_counterfactual_robustness_participating_trace_rate=(
            args.min_counterfactual_robustness_participating_trace_rate
        ),
        min_counterfactual_robustness_coverage_rate=(
            args.min_counterfactual_robustness_coverage_rate
        ),
        min_counterfactual_robustness_pass_rate=(
            args.min_counterfactual_robustness_pass_rate
        ),
        min_counterfactual_robustness_flip_success_rate=(
            args.min_counterfactual_robustness_flip_success_rate
        ),
        max_counterfactual_robustness_false_invariance_rate_increase=(
            args.max_counterfactual_robustness_false_invariance_rate_increase
        ),
        max_counterfactual_robustness_trace_gap_rate_increase=(
            args.max_counterfactual_robustness_trace_gap_rate_increase
        ),
        promotion_contract_covered_fact_property_scopes=(
            args.promotion_contract_covered_fact_property_scope
        ),
        min_promotion_contract_covered_fact_property_metric_count=(
            args.min_promotion_contract_covered_fact_property_metric_count
        ),
        min_promotion_contract_covered_fact_min_records=(
            args.min_promotion_contract_covered_fact_min_records
        ),
        min_promotion_contract_covered_fact_min_source_documents=(
            args.min_promotion_contract_covered_fact_min_source_documents
        ),
        max_promotion_contract_covered_fact_min_decision_accuracy_drop=(
            args.max_promotion_contract_covered_fact_min_decision_accuracy_drop
        ),
        max_promotion_contract_covered_fact_max_false_supported_rate_increase=(
            args.max_promotion_contract_covered_fact_max_false_supported_rate_increase
        ),
        max_promotion_contract_covered_fact_min_false_refuted_rate_drop=(
            args.max_promotion_contract_covered_fact_min_false_refuted_rate_drop
        ),
        max_product_trace_action_audit_error_rate_increase=(
            args.max_product_trace_action_audit_error_rate_increase
        ),
        max_product_trace_action_audit_missing_retrieval_action_rate_increase=(
            args.max_product_trace_action_audit_missing_retrieval_action_rate_increase
        ),
        max_product_trace_action_audit_missing_plan_retrieval_query_rate_increase=(
            args.max_product_trace_action_audit_missing_plan_retrieval_query_rate_increase
        ),
        max_product_trace_action_audit_malformed_payload_rate_increase=(
            args.max_product_trace_action_audit_malformed_payload_rate_increase
        ),
        max_product_trace_action_audit_unexpected_action_rate_increase=(
            args.max_product_trace_action_audit_unexpected_action_rate_increase
        ),
        max_product_trace_action_audit_unknown_claim_id_rate_increase=(
            args.max_product_trace_action_audit_unknown_claim_id_rate_increase
        ),
        max_product_trace_action_execution_alignment_failed_trace_rate_increase=(
            args.max_product_trace_action_execution_alignment_failed_trace_rate_increase
        ),
        max_product_trace_action_execution_missing_result_rate_increase=(
            args.max_product_trace_action_execution_missing_result_rate_increase
        ),
        max_product_trace_action_execution_unexpected_result_rate_increase=(
            args.max_product_trace_action_execution_unexpected_result_rate_increase
        ),
        max_product_trace_action_execution_request_id_mismatch_rate_increase=(
            args.max_product_trace_action_execution_request_id_mismatch_rate_increase
        ),
        max_product_trace_trajectory_audit_failed_trace_rate_increase=(
            args.max_product_trace_trajectory_audit_failed_trace_rate_increase
        ),
        max_product_trace_trajectory_audit_error_rate_increase=(
            args.max_product_trace_trajectory_audit_error_rate_increase
        ),
        max_product_trace_trajectory_audit_factual_rate_increase=(
            args.max_product_trace_trajectory_audit_factual_rate_increase
        ),
        max_product_trace_trajectory_audit_referential_rate_increase=(
            args.max_product_trace_trajectory_audit_referential_rate_increase
        ),
        max_product_trace_trajectory_audit_logical_rate_increase=(
            args.max_product_trace_trajectory_audit_logical_rate_increase
        ),
        max_product_trace_trajectory_audit_procedural_rate_increase=(
            args.max_product_trace_trajectory_audit_procedural_rate_increase
        ),
        max_product_trace_trajectory_audit_scope_rate_increase=(
            args.max_product_trace_trajectory_audit_scope_rate_increase
        ),
        min_current_trace_count=args.min_current_trace_count,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_drift and payload["status"] == "blocked":
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare ProductTrace runtime baselines for drift")
    parser.add_argument("--current", required=True, help="current product runtime baseline report JSON")
    parser.add_argument("--baseline", default=None, help="baseline product runtime baseline report JSON")
    parser.add_argument("--registry", default=None, help="local ArtifactRegistry JSON path")
    parser.add_argument("--baseline-key", default=None, help="product_runtime_baseline registry key")
    parser.add_argument("--baseline-name", default=None, help="product runtime baseline record name")
    parser.add_argument("--baseline-version", default=None, help="product runtime baseline record version")
    parser.add_argument("--runtime-budget-policy", default=None,
                        help="optional ProductRuntimeBudgetPolicy JSON path to gate current baseline summary")
    parser.add_argument("--runtime-budget-policy-key", default=None,
                        help="product_runtime_budget_policy registry key")
    parser.add_argument("--json", default=None, help="optional drift report JSON output path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest output path")
    parser.add_argument("--name", default=None, help="optional registry drift report name")
    parser.add_argument("--version", default=None, help="optional registry drift report version")
    parser.add_argument("--metadata", action="append", default=[], help="metadata key=value; repeatable")
    parser.add_argument("--max-total-seconds-mean-ratio", type=float, default=None)
    parser.add_argument("--max-total-seconds-p95-ratio", type=float, default=None)
    parser.add_argument("--max-mean-route-duration-ratio", type=float, default=None)
    parser.add_argument("--max-p95-route-duration-ratio", type=float, default=None)
    parser.add_argument("--max-mean-attempted-route-count-delta", type=float, default=None)
    parser.add_argument("--max-retrieval-use-rate-delta", type=float, default=None)
    parser.add_argument("--max-cache-hit-rate-drop", type=float, default=None)
    parser.add_argument("--max-verification-skip-rate-drop", type=float, default=None)
    parser.add_argument("--min-promotion-contract-coverage", type=float, default=None)
    parser.add_argument("--min-pre-generation-probe-comparison-coverage", type=float, default=None)
    parser.add_argument(
        "--min-pre-generation-probe-comparison-manifest-verified-rate",
        type=float,
        default=None,
    )
    parser.add_argument("--min-pre-generation-probe-comparison-model-count", type=float, default=None)
    parser.add_argument("--min-pre-generation-probe-comparison-run-count", type=float, default=None)
    parser.add_argument("--min-pre-generation-probe-comparison-redline-pass-rate", type=float, default=None)
    parser.add_argument(
        "--max-pre-generation-probe-comparison-best-test-label-auroc-drop",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-pre-generation-probe-comparison-best-redline-auroc-drop",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-pre-generation-probe-comparison-best-redline-margin-drop",
        type=float,
        default=None,
    )
    parser.add_argument("--min-claim-factuality-probe-comparison-coverage", type=float, default=None)
    parser.add_argument(
        "--min-claim-factuality-probe-comparison-manifest-verified-rate",
        type=float,
        default=None,
    )
    parser.add_argument("--min-claim-factuality-probe-comparison-model-count", type=float, default=None)
    parser.add_argument("--min-claim-factuality-probe-comparison-run-count", type=float, default=None)
    parser.add_argument(
        "--min-claim-factuality-probe-comparison-redline-pass-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-claim-factuality-probe-comparison-best-test-label-auroc-drop",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-claim-factuality-probe-comparison-best-test-selective-accuracy-drop",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-claim-factuality-probe-comparison-best-test-selective-coverage-drop",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-claim-factuality-probe-comparison-best-redline-auroc-drop",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-claim-factuality-probe-comparison-best-redline-margin-drop",
        type=float,
        default=None,
    )
    parser.add_argument("--min-counterfactual-verification-coverage", type=float, default=None)
    parser.add_argument(
        "--min-counterfactual-verification-manifest-verified-rate",
        type=float,
        default=None,
    )
    parser.add_argument("--min-counterfactual-verification-record-count", type=float, default=None)
    parser.add_argument("--min-counterfactual-verification-pass-rate", type=float, default=None)
    parser.add_argument(
        "--max-counterfactual-verification-false-invariance-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-counterfactual-verification-flip-success-count-drop",
        type=float,
        default=None,
    )
    parser.add_argument("--min-evidence-handoff-coverage", type=float, default=None)
    parser.add_argument(
        "--min-evidence-handoff-manifest-verified-rate",
        type=float,
        default=None,
    )
    parser.add_argument("--min-evidence-handoff-present-metric-rate", type=float, default=None)
    parser.add_argument("--max-evidence-handoff-missing-metric-rate", type=float, default=None)
    parser.add_argument("--max-evidence-handoff-missing-metric-count", type=float, default=None)
    parser.add_argument("--max-evidence-handoff-blocked-group-count", type=float, default=None)
    parser.add_argument("--min-evidence-handoff-promoted-group-rate", type=float, default=None)
    parser.add_argument("--min-frontier-release-evidence-coverage", type=float, default=None)
    parser.add_argument(
        "--min-frontier-release-evidence-report-present-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-frontier-release-evidence-manifest-present-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-frontier-release-evidence-status-promote-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-frontier-release-evidence-decision-promote-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-frontier-release-evidence-verifier-track-promote-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-frontier-release-evidence-abstention-track-promote-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-frontier-release-evidence-citation-batch-track-promote-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-frontier-release-evidence-frontier-rerun-rollup-track-promote-rate",
        type=float,
        default=None,
    )
    parser.add_argument("--min-frontier-release-evidence-run-count", type=float, default=None)
    parser.add_argument(
        "--min-frontier-release-evidence-frontier-rerun-rollup-report-count",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-frontier-release-evidence-frontier-rerun-rollup-candidate-count",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-frontier-release-evidence-frontier-rerun-rollup-missing-report-count",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-frontier-release-evidence-frontier-rerun-rollup-invalid-report-count",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-frontier-release-evidence-frontier-rerun-rollup-blocked-candidate-count",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-frontier-release-evidence-frontier-rerun-rollup-promotion-ready-count",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-frontier-release-evidence-citation-batch-rollup-count",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-frontier-release-evidence-citation-batch-missing-expected-batch-count",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-frontier-release-evidence-citation-batch-duplicate-batch-count",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-frontier-release-evidence-citation-batch-unexpected-batch-count",
        type=float,
        default=None,
    )
    parser.add_argument("--min-triple-extraction-fixture-matrix-coverage", type=float, default=None)
    parser.add_argument("--max-triple-extraction-fixture-matrix-mean-best-f1-drop", type=float, default=None)
    parser.add_argument("--max-triple-extraction-fixture-matrix-mean-f1-lift-drop", type=float, default=None)
    parser.add_argument("--min-triple-claim-coverage", type=float, default=None)
    parser.add_argument("--min-triple-audit-claim-coverage", type=float, default=None)
    parser.add_argument("--min-triple-audit-pass-rate", type=float, default=None)
    parser.add_argument("--min-triple-slot-coverage", type=float, default=None)
    parser.add_argument("--min-world-model-participating-trace-rate", type=float, default=None)
    parser.add_argument("--min-world-model-coverage-rate", type=float, default=None)
    parser.add_argument("--max-world-model-conflict-rate-increase", type=float, default=None)
    parser.add_argument("--max-world-model-low-agreement-rate-increase", type=float, default=None)
    parser.add_argument("--max-world-model-trace-gap-rate-increase", type=float, default=None)
    parser.add_argument("--min-context-sensitivity-participating-trace-rate", type=float, default=None)
    parser.add_argument("--min-context-sensitivity-coverage-rate", type=float, default=None)
    parser.add_argument(
        "--max-context-sensitivity-flagged-result-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-context-sensitivity-trace-gap-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-context-sensitivity-max-flagged-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-context-sensitivity-max-ratio-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-counterfactual-robustness-participating-trace-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-counterfactual-robustness-coverage-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-counterfactual-robustness-pass-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-counterfactual-robustness-flip-success-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-counterfactual-robustness-false-invariance-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-counterfactual-robustness-trace-gap-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--promotion-contract-covered-fact-property-scope",
        action="append",
        default=None,
        choices=tuple(_COVERED_FACT_PROPERTY_SCOPES),
        help=(
            "covered-fact property rollup scope to gate; repeatable; defaults to recommended_route "
            "when any covered-fact property gate is supplied"
        ),
    )
    parser.add_argument("--min-promotion-contract-covered-fact-property-metric-count", type=float, default=None)
    parser.add_argument("--min-promotion-contract-covered-fact-min-records", type=float, default=None)
    parser.add_argument("--min-promotion-contract-covered-fact-min-source-documents", type=float, default=None)
    parser.add_argument(
        "--max-promotion-contract-covered-fact-min-decision-accuracy-drop",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-promotion-contract-covered-fact-max-false-supported-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-promotion-contract-covered-fact-min-false-refuted-rate-drop",
        type=float,
        default=None,
    )
    parser.add_argument("--max-product-trace-action-audit-error-rate-increase", type=float, default=None)
    parser.add_argument(
        "--max-product-trace-action-audit-missing-retrieval-action-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-product-trace-action-audit-missing-plan-retrieval-query-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument("--max-product-trace-action-audit-malformed-payload-rate-increase", type=float, default=None)
    parser.add_argument("--max-product-trace-action-audit-unexpected-action-rate-increase", type=float, default=None)
    parser.add_argument("--max-product-trace-action-audit-unknown-claim-id-rate-increase", type=float, default=None)
    parser.add_argument(
        "--max-product-trace-action-execution-alignment-failed-trace-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument("--max-product-trace-action-execution-missing-result-rate-increase", type=float, default=None)
    parser.add_argument(
        "--max-product-trace-action-execution-unexpected-result-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-product-trace-action-execution-request-id-mismatch-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-product-trace-trajectory-audit-failed-trace-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-product-trace-trajectory-audit-error-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-product-trace-trajectory-audit-factual-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-product-trace-trajectory-audit-referential-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-product-trace-trajectory-audit-logical-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-product-trace-trajectory-audit-procedural-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-product-trace-trajectory-audit-scope-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument("--min-current-trace-count", type=int, default=None)
    parser.add_argument("--compact-json", action="store_true",
                        help="write minified drift report and manifest JSON")
    parser.add_argument("--fail-on-drift", action="store_true",
                        help="exit non-zero when drift gates block")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
