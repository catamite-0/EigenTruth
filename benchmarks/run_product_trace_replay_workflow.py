"""Build a replay-ready ProductTrace corpus and run replay reports.

This workflow is the one-command handoff from raw saved product traces to
redacted corpus artifacts, product runtime baselines, and selector-policy replay.
It performs no model, verifier, retriever, or external-service work.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence, TypeVar

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.build_product_trace_corpus import (  # noqa: E402
    ProductTraceCorpusConfig,
    build_product_trace_corpus,
)
from benchmarks.compare_product_runtime_baselines import compare_product_runtime_baselines  # noqa: E402
from benchmarks.config_utils import (  # noqa: E402
    planned_artifact_manifest_summary,
    strict_bool,
    strict_positive_int,
)
from benchmarks.run_product_runtime_baseline import (  # noqa: E402
    ProductRuntimeBaselineConfig,
    build_product_runtime_baseline,
)
from benchmarks.run_runtime_profile_selector_replay import (  # noqa: E402
    RuntimeProfileSelectorReplayConfig,
    run_runtime_profile_selector_replay,
)
from benchmarks.run_runtime_profile_selector_tuning import RuntimeProfileSelectorCandidate  # noqa: E402
from eigentruth.registry import (  # noqa: E402
    ArtifactRegistry,
    build_artifact_manifest,
    fingerprint_path,
    load_and_verify_artifact_manifest,
    load_fingerprint_cache,
    save_fingerprint_cache,
)

_T = TypeVar("_T")
_TRACE_SUMMARY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProductTraceReplayWorkflowConfig:
    """Configuration for a full ProductTrace replay workflow."""

    trace_paths: Sequence[str | Path] = ()
    jsonl_paths: Sequence[str | Path] = ()
    output_dir: str | Path = "artifacts/product_trace_replay_workflow"
    candidates: Sequence[RuntimeProfileSelectorCandidate | Mapping[str, Any]] = ()
    replay_policy_path: str | Path | None = None
    runtime_policy_path: str | Path | None = None
    promotion_contract_path: str | Path | None = None
    runtime_recommended_policy_path: str | Path | None = None
    runtime_drift_baseline_path: str | Path | None = None
    runtime_drift_baseline_key: str | None = None
    runtime_drift_baseline_name: str | None = None
    runtime_drift_baseline_version: str | None = None
    runtime_drift_budget_policy_path: str | Path | None = None
    runtime_drift_budget_policy_key: str | None = None
    runtime_drift_report_path: str | Path | None = None
    runtime_drift_artifact_manifest_path: str | Path | None = None
    max_runtime_drift_total_seconds_mean_ratio: float | None = None
    max_runtime_drift_total_seconds_p95_ratio: float | None = None
    max_runtime_drift_mean_route_duration_ratio: float | None = None
    max_runtime_drift_p95_route_duration_ratio: float | None = None
    max_runtime_drift_mean_attempted_route_count_delta: float | None = None
    max_runtime_drift_retrieval_use_rate_delta: float | None = None
    max_runtime_drift_cache_hit_rate_drop: float | None = None
    max_runtime_drift_verification_skip_rate_drop: float | None = None
    min_runtime_drift_pre_generation_risk_coverage_rate: float | None = None
    min_runtime_drift_pre_generation_learned_risk_coverage_rate: float | None = None
    max_runtime_drift_pre_generation_audit_profile_rate_increase: float | None = None
    max_runtime_drift_pre_generation_learned_risk_routed_rate_increase: float | None = None
    max_runtime_drift_pre_generation_learned_risk_probability_mean_increase: float | None = None
    min_runtime_drift_promotion_contract_coverage: float | None = None
    min_runtime_drift_pre_generation_probe_comparison_coverage: float | None = None
    min_runtime_drift_pre_generation_probe_comparison_manifest_verified_rate: float | None = None
    min_runtime_drift_pre_generation_probe_comparison_model_count: float | None = None
    min_runtime_drift_pre_generation_probe_comparison_run_count: float | None = None
    min_runtime_drift_pre_generation_probe_comparison_redline_pass_rate: float | None = None
    max_runtime_drift_pre_generation_probe_comparison_best_test_label_auroc_drop: float | None = None
    max_runtime_drift_pre_generation_probe_comparison_best_redline_auroc_drop: float | None = None
    max_runtime_drift_pre_generation_probe_comparison_best_redline_margin_drop: float | None = None
    min_runtime_drift_claim_factuality_probe_comparison_coverage: float | None = None
    min_runtime_drift_claim_factuality_probe_comparison_manifest_verified_rate: float | None = None
    min_runtime_drift_claim_factuality_probe_comparison_model_count: float | None = None
    min_runtime_drift_claim_factuality_probe_comparison_run_count: float | None = None
    min_runtime_drift_claim_factuality_probe_comparison_dataset_count: float | None = None
    min_runtime_drift_claim_factuality_probe_comparison_redline_pass_rate: float | None = None
    max_runtime_drift_claim_factuality_probe_comparison_best_test_label_auroc_drop: float | None = None
    max_runtime_drift_claim_factuality_probe_comparison_best_test_selective_accuracy_drop: float | None = None
    max_runtime_drift_claim_factuality_probe_comparison_best_test_selective_coverage_drop: float | None = None
    max_runtime_drift_claim_factuality_probe_comparison_best_redline_auroc_drop: float | None = None
    max_runtime_drift_claim_factuality_probe_comparison_best_redline_margin_drop: float | None = None
    min_runtime_drift_counterfactual_verification_coverage: float | None = None
    min_runtime_drift_counterfactual_verification_manifest_verified_rate: float | None = None
    min_runtime_drift_counterfactual_verification_record_count: float | None = None
    min_runtime_drift_counterfactual_verification_pass_rate: float | None = None
    max_runtime_drift_counterfactual_verification_false_invariance_rate: float | None = None
    max_runtime_drift_counterfactual_verification_flip_success_count_drop: float | None = None
    min_runtime_drift_evidence_handoff_coverage: float | None = None
    min_runtime_drift_evidence_handoff_manifest_verified_rate: float | None = None
    min_runtime_drift_evidence_handoff_present_metric_rate: float | None = None
    max_runtime_drift_evidence_handoff_missing_metric_rate: float | None = None
    max_runtime_drift_evidence_handoff_missing_metric_count: float | None = None
    max_runtime_drift_evidence_handoff_blocked_group_count: float | None = None
    min_runtime_drift_evidence_handoff_promoted_group_rate: float | None = None
    min_runtime_drift_fact_selfcheck_gate_coverage: float | None = None
    min_runtime_drift_fact_selfcheck_gate_report_present_rate: float | None = None
    min_runtime_drift_fact_selfcheck_gate_manifest_present_rate: float | None = None
    min_runtime_drift_fact_selfcheck_gate_manifest_verified_rate: float | None = None
    min_runtime_drift_fact_selfcheck_gate_passed_rate: float | None = None
    min_runtime_drift_fact_selfcheck_gate_run_count: float | None = None
    max_runtime_drift_fact_selfcheck_gate_failed_run_count: float | None = None
    min_runtime_drift_fact_selfcheck_gate_min_executed_rate: float | None = None
    min_runtime_drift_fact_selfcheck_gate_min_decided_rate: float | None = None
    max_runtime_drift_fact_selfcheck_gate_max_not_applicable_rate: float | None = None
    min_runtime_drift_fact_selfcheck_gate_min_claim_triples_per_record: float | None = None
    min_runtime_drift_fact_selfcheck_gate_min_sample_triples_per_record: float | None = None
    min_runtime_drift_triple_extraction_fixture_matrix_coverage: float | None = None
    max_runtime_drift_triple_extraction_fixture_matrix_mean_best_f1_drop: float | None = None
    max_runtime_drift_triple_extraction_fixture_matrix_mean_f1_lift_drop: float | None = None
    min_runtime_drift_triple_claim_coverage: float | None = None
    min_runtime_drift_triple_audit_claim_coverage: float | None = None
    min_runtime_drift_triple_audit_pass_rate: float | None = None
    min_runtime_drift_triple_slot_coverage: float | None = None
    min_runtime_drift_world_model_participating_trace_rate: float | None = None
    min_runtime_drift_world_model_coverage_rate: float | None = None
    max_runtime_drift_world_model_conflict_rate_increase: float | None = None
    max_runtime_drift_world_model_low_agreement_rate_increase: float | None = None
    max_runtime_drift_world_model_trace_gap_rate_increase: float | None = None
    min_runtime_drift_context_sensitivity_participating_trace_rate: float | None = None
    min_runtime_drift_context_sensitivity_coverage_rate: float | None = None
    max_runtime_drift_context_sensitivity_flagged_result_rate_increase: float | None = None
    max_runtime_drift_context_sensitivity_trace_gap_rate_increase: float | None = None
    max_runtime_drift_context_sensitivity_max_flagged_rate_increase: float | None = None
    max_runtime_drift_context_sensitivity_max_ratio_increase: float | None = None
    min_runtime_drift_counterfactual_robustness_participating_trace_rate: float | None = None
    min_runtime_drift_counterfactual_robustness_coverage_rate: float | None = None
    min_runtime_drift_counterfactual_robustness_pass_rate: float | None = None
    min_runtime_drift_counterfactual_robustness_flip_success_rate: float | None = None
    max_runtime_drift_counterfactual_robustness_false_invariance_rate_increase: float | None = None
    max_runtime_drift_counterfactual_robustness_trace_gap_rate_increase: float | None = None
    min_runtime_drift_claim_risk_localization_coverage_rate: float | None = None
    max_runtime_drift_claim_risk_localization_high_risk_claim_count_increase: float | None = None
    max_runtime_drift_claim_risk_localization_medium_or_high_risk_claim_count_increase: (
        float | None
    ) = None
    max_runtime_drift_claim_risk_localization_entity_candidate_observation_count_increase: (
        float | None
    ) = None
    max_runtime_drift_claim_risk_localization_unique_entity_candidate_count_increase: (
        float | None
    ) = None
    max_runtime_drift_claim_risk_localization_high_risk_entity_candidate_count_increase: (
        float | None
    ) = None
    max_runtime_drift_claim_risk_localization_medium_or_high_entity_candidate_count_increase: (
        float | None
    ) = None
    runtime_drift_covered_fact_property_scopes: Sequence[str] = ()
    min_runtime_drift_covered_fact_property_metric_count: float | None = None
    min_runtime_drift_covered_fact_min_records: float | None = None
    min_runtime_drift_covered_fact_min_source_documents: float | None = None
    max_runtime_drift_covered_fact_min_decision_accuracy_drop: float | None = None
    max_runtime_drift_covered_fact_max_false_supported_rate_increase: float | None = None
    max_runtime_drift_covered_fact_min_false_refuted_rate_drop: float | None = None
    max_runtime_drift_product_trace_action_audit_error_rate_increase: float | None = None
    max_runtime_drift_product_trace_action_audit_missing_retrieval_action_rate_increase: float | None = None
    max_runtime_drift_product_trace_action_audit_missing_plan_retrieval_query_rate_increase: float | None = None
    max_runtime_drift_product_trace_action_audit_malformed_payload_rate_increase: float | None = None
    max_runtime_drift_product_trace_action_audit_unexpected_action_rate_increase: float | None = None
    max_runtime_drift_product_trace_action_audit_unknown_claim_id_rate_increase: float | None = None
    max_runtime_drift_product_trace_action_execution_alignment_failed_trace_rate_increase: float | None = None
    max_runtime_drift_product_trace_action_execution_missing_result_rate_increase: float | None = None
    max_runtime_drift_product_trace_action_execution_unexpected_result_rate_increase: float | None = None
    max_runtime_drift_product_trace_action_execution_request_id_mismatch_rate_increase: float | None = None
    min_runtime_drift_product_trace_receipt_claim_support_reference_support_rate: (
        float | None
    ) = None
    max_runtime_drift_product_trace_receipt_claim_support_unsupported_reference_rate_increase: (
        float | None
    ) = None
    max_runtime_drift_product_trace_receipt_claim_support_missing_reference_rate_increase: (
        float | None
    ) = None
    max_runtime_drift_product_trace_receipt_claim_support_unreceipted_reference_rate_increase: (
        float | None
    ) = None
    max_runtime_drift_product_trace_receipt_claim_support_failed_result_reference_rate_increase: (
        float | None
    ) = None
    max_runtime_drift_product_trace_receipt_claim_support_fingerprint_mismatch_reference_rate_increase: (
        float | None
    ) = None
    max_runtime_drift_product_trace_receipt_claim_support_unsigned_reference_rate_increase: (
        float | None
    ) = None
    max_runtime_drift_product_trace_trajectory_audit_failed_trace_rate_increase: float | None = None
    max_runtime_drift_product_trace_trajectory_audit_error_rate_increase: float | None = None
    max_runtime_drift_product_trace_trajectory_audit_factual_rate_increase: float | None = None
    max_runtime_drift_product_trace_trajectory_audit_referential_rate_increase: float | None = None
    max_runtime_drift_product_trace_trajectory_audit_logical_rate_increase: float | None = None
    max_runtime_drift_product_trace_trajectory_audit_procedural_rate_increase: float | None = None
    max_runtime_drift_product_trace_trajectory_audit_scope_rate_increase: float | None = None
    max_runtime_drift_product_trace_trajectory_audit_cascade_rate_increase: float | None = None
    min_runtime_drift_product_trace_provenance_coverage_rate: float | None = None
    min_runtime_drift_product_trace_provenance_supported_claim_evidence_coverage: (
        float | None
    ) = None
    max_runtime_drift_product_trace_provenance_missing_reference_rate_increase: (
        float | None
    ) = None
    max_runtime_drift_product_trace_provenance_unsupported_supported_claim_rate_increase: (
        float | None
    ) = None
    max_runtime_drift_product_trace_provenance_error_rate_increase: float | None = None
    min_runtime_drift_product_trace_provenance_final_answer_evidence_reference_rate: (
        float | None
    ) = None
    min_runtime_drift_product_trace_citation_integrity_participating_trace_rate: (
        float | None
    ) = None
    min_runtime_drift_product_trace_citation_integrity_coverage_rate: float | None = None
    max_runtime_drift_product_trace_citation_integrity_mismatch_rate_increase: (
        float | None
    ) = None
    max_runtime_drift_product_trace_citation_integrity_unresolved_rate_increase: (
        float | None
    ) = None
    max_runtime_drift_product_trace_citation_integrity_issue_rate_increase: (
        float | None
    ) = None
    max_runtime_drift_product_trace_citation_integrity_trace_gap_rate_increase: (
        float | None
    ) = None
    min_runtime_drift_current_trace_count: int | None = None
    max_action_audit_error_rate: float | None = None
    max_action_audit_missing_retrieval_rate: float | None = None
    max_action_audit_missing_plan_retrieval_query_rate: float | None = None
    max_action_audit_malformed_payload_rate: float | None = None
    max_action_audit_unexpected_action_rate: float | None = None
    max_action_audit_unknown_claim_id_rate: float | None = None
    max_action_execution_missing_result_rate: float | None = None
    max_action_execution_unexpected_result_rate: float | None = None
    max_action_execution_request_id_mismatch_rate: float | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    redact_text: bool = True
    require_runtime_trace: bool = False
    strict: bool = False
    limit: int | None = None
    compact_json: bool = False
    verify_manifest: bool = False
    verification_report_path: str | Path | None = None
    allow_manifest_verification_failures: bool = False
    fingerprint_cache_path: str | Path | None = None
    corpus_cache_path: str | Path | None = None
    refresh_corpus_cache: bool = False
    corpus_source_cache_path: str | Path | None = None
    refresh_corpus_source_cache: bool = False
    runtime_trace_records_cache_path: str | Path | None = None
    refresh_runtime_trace_records_cache: bool = False
    runtime_trace_scan_workers: int = 1
    selector_trace_inputs_path: str | Path | None = None
    refresh_selector_trace_inputs: bool = False

    def __post_init__(self) -> None:
        trace_paths = tuple(Path(path) for path in self.trace_paths)
        jsonl_paths = tuple(Path(path) for path in self.jsonl_paths)
        if not trace_paths and not jsonl_paths:
            raise ValueError("at least one ProductTrace JSON or JSONL path is required.")
        candidates = tuple(_candidate_from_value(candidate) for candidate in self.candidates)
        if not candidates:
            raise ValueError("at least one selector candidate is required.")
        names = [candidate.name for candidate in candidates]
        if len(set(names)) != len(names):
            raise ValueError("selector candidate names must be unique.")
        if self.runtime_policy_path is not None and self.promotion_contract_path is not None:
            raise ValueError("runtime_policy_path and promotion_contract_path are mutually exclusive.")
        drift_baseline_selected = any(
            value is not None
            for value in (
                self.runtime_drift_baseline_path,
                self.runtime_drift_baseline_key,
                self.runtime_drift_baseline_name,
                self.runtime_drift_baseline_version,
            )
        )
        drift_gate_selected = bool(self.runtime_drift_covered_fact_property_scopes) or any(
            value is not None
            for value in (
                self.runtime_drift_budget_policy_path,
                self.runtime_drift_budget_policy_key,
                self.max_runtime_drift_total_seconds_mean_ratio,
                self.max_runtime_drift_total_seconds_p95_ratio,
                self.max_runtime_drift_mean_route_duration_ratio,
                self.max_runtime_drift_p95_route_duration_ratio,
                self.max_runtime_drift_mean_attempted_route_count_delta,
                self.max_runtime_drift_retrieval_use_rate_delta,
                self.max_runtime_drift_cache_hit_rate_drop,
                self.max_runtime_drift_verification_skip_rate_drop,
                self.min_runtime_drift_pre_generation_risk_coverage_rate,
                self.min_runtime_drift_pre_generation_learned_risk_coverage_rate,
                self.max_runtime_drift_pre_generation_audit_profile_rate_increase,
                self.max_runtime_drift_pre_generation_learned_risk_routed_rate_increase,
                self.max_runtime_drift_pre_generation_learned_risk_probability_mean_increase,
                self.min_runtime_drift_promotion_contract_coverage,
                self.min_runtime_drift_pre_generation_probe_comparison_coverage,
                self.min_runtime_drift_pre_generation_probe_comparison_manifest_verified_rate,
                self.min_runtime_drift_pre_generation_probe_comparison_model_count,
                self.min_runtime_drift_pre_generation_probe_comparison_run_count,
                self.min_runtime_drift_pre_generation_probe_comparison_redline_pass_rate,
                self.max_runtime_drift_pre_generation_probe_comparison_best_test_label_auroc_drop,
                self.max_runtime_drift_pre_generation_probe_comparison_best_redline_auroc_drop,
                self.max_runtime_drift_pre_generation_probe_comparison_best_redline_margin_drop,
                self.min_runtime_drift_claim_factuality_probe_comparison_coverage,
                self.min_runtime_drift_claim_factuality_probe_comparison_manifest_verified_rate,
                self.min_runtime_drift_claim_factuality_probe_comparison_model_count,
                self.min_runtime_drift_claim_factuality_probe_comparison_run_count,
                self.min_runtime_drift_claim_factuality_probe_comparison_dataset_count,
                self.min_runtime_drift_claim_factuality_probe_comparison_redline_pass_rate,
                self.max_runtime_drift_claim_factuality_probe_comparison_best_test_label_auroc_drop,
                self.max_runtime_drift_claim_factuality_probe_comparison_best_test_selective_accuracy_drop,
                self.max_runtime_drift_claim_factuality_probe_comparison_best_test_selective_coverage_drop,
                self.max_runtime_drift_claim_factuality_probe_comparison_best_redline_auroc_drop,
                self.max_runtime_drift_claim_factuality_probe_comparison_best_redline_margin_drop,
                self.min_runtime_drift_counterfactual_verification_coverage,
                self.min_runtime_drift_counterfactual_verification_manifest_verified_rate,
                self.min_runtime_drift_counterfactual_verification_record_count,
                self.min_runtime_drift_counterfactual_verification_pass_rate,
                self.max_runtime_drift_counterfactual_verification_false_invariance_rate,
                self.max_runtime_drift_counterfactual_verification_flip_success_count_drop,
                self.min_runtime_drift_evidence_handoff_coverage,
                self.min_runtime_drift_evidence_handoff_manifest_verified_rate,
                self.min_runtime_drift_evidence_handoff_present_metric_rate,
                self.max_runtime_drift_evidence_handoff_missing_metric_rate,
                self.max_runtime_drift_evidence_handoff_missing_metric_count,
                self.max_runtime_drift_evidence_handoff_blocked_group_count,
                self.min_runtime_drift_evidence_handoff_promoted_group_rate,
                self.min_runtime_drift_fact_selfcheck_gate_coverage,
                self.min_runtime_drift_fact_selfcheck_gate_report_present_rate,
                self.min_runtime_drift_fact_selfcheck_gate_manifest_present_rate,
                self.min_runtime_drift_fact_selfcheck_gate_manifest_verified_rate,
                self.min_runtime_drift_fact_selfcheck_gate_passed_rate,
                self.min_runtime_drift_fact_selfcheck_gate_run_count,
                self.max_runtime_drift_fact_selfcheck_gate_failed_run_count,
                self.min_runtime_drift_fact_selfcheck_gate_min_executed_rate,
                self.min_runtime_drift_fact_selfcheck_gate_min_decided_rate,
                self.max_runtime_drift_fact_selfcheck_gate_max_not_applicable_rate,
                self.min_runtime_drift_fact_selfcheck_gate_min_claim_triples_per_record,
                self.min_runtime_drift_fact_selfcheck_gate_min_sample_triples_per_record,
                self.min_runtime_drift_triple_extraction_fixture_matrix_coverage,
                self.max_runtime_drift_triple_extraction_fixture_matrix_mean_best_f1_drop,
                self.max_runtime_drift_triple_extraction_fixture_matrix_mean_f1_lift_drop,
                self.min_runtime_drift_triple_claim_coverage,
                self.min_runtime_drift_triple_audit_claim_coverage,
                self.min_runtime_drift_triple_audit_pass_rate,
                self.min_runtime_drift_triple_slot_coverage,
                self.min_runtime_drift_world_model_participating_trace_rate,
                self.min_runtime_drift_world_model_coverage_rate,
                self.max_runtime_drift_world_model_conflict_rate_increase,
                self.max_runtime_drift_world_model_low_agreement_rate_increase,
                self.max_runtime_drift_world_model_trace_gap_rate_increase,
                self.min_runtime_drift_context_sensitivity_participating_trace_rate,
                self.min_runtime_drift_context_sensitivity_coverage_rate,
                self.max_runtime_drift_context_sensitivity_flagged_result_rate_increase,
                self.max_runtime_drift_context_sensitivity_trace_gap_rate_increase,
                self.max_runtime_drift_context_sensitivity_max_flagged_rate_increase,
                self.max_runtime_drift_context_sensitivity_max_ratio_increase,
                self.min_runtime_drift_counterfactual_robustness_participating_trace_rate,
                self.min_runtime_drift_counterfactual_robustness_coverage_rate,
                self.min_runtime_drift_counterfactual_robustness_pass_rate,
                self.min_runtime_drift_counterfactual_robustness_flip_success_rate,
                self.max_runtime_drift_counterfactual_robustness_false_invariance_rate_increase,
                self.max_runtime_drift_counterfactual_robustness_trace_gap_rate_increase,
                self.min_runtime_drift_claim_risk_localization_coverage_rate,
                self.max_runtime_drift_claim_risk_localization_high_risk_claim_count_increase,
                self.max_runtime_drift_claim_risk_localization_medium_or_high_risk_claim_count_increase,
                self.max_runtime_drift_claim_risk_localization_entity_candidate_observation_count_increase,
                self.max_runtime_drift_claim_risk_localization_unique_entity_candidate_count_increase,
                self.max_runtime_drift_claim_risk_localization_high_risk_entity_candidate_count_increase,
                self.max_runtime_drift_claim_risk_localization_medium_or_high_entity_candidate_count_increase,
                self.min_runtime_drift_covered_fact_property_metric_count,
                self.min_runtime_drift_covered_fact_min_records,
                self.min_runtime_drift_covered_fact_min_source_documents,
                self.max_runtime_drift_covered_fact_min_decision_accuracy_drop,
                self.max_runtime_drift_covered_fact_max_false_supported_rate_increase,
                self.max_runtime_drift_covered_fact_min_false_refuted_rate_drop,
                self.max_runtime_drift_product_trace_action_audit_error_rate_increase,
                self.max_runtime_drift_product_trace_action_audit_missing_retrieval_action_rate_increase,
                self.max_runtime_drift_product_trace_action_audit_missing_plan_retrieval_query_rate_increase,
                self.max_runtime_drift_product_trace_action_audit_malformed_payload_rate_increase,
                self.max_runtime_drift_product_trace_action_audit_unexpected_action_rate_increase,
                self.max_runtime_drift_product_trace_action_audit_unknown_claim_id_rate_increase,
                self.max_runtime_drift_product_trace_action_execution_alignment_failed_trace_rate_increase,
                self.max_runtime_drift_product_trace_action_execution_missing_result_rate_increase,
                self.max_runtime_drift_product_trace_action_execution_unexpected_result_rate_increase,
                self.max_runtime_drift_product_trace_action_execution_request_id_mismatch_rate_increase,
                self.min_runtime_drift_product_trace_receipt_claim_support_reference_support_rate,
                self.max_runtime_drift_product_trace_receipt_claim_support_unsupported_reference_rate_increase,
                self.max_runtime_drift_product_trace_receipt_claim_support_missing_reference_rate_increase,
                self.max_runtime_drift_product_trace_receipt_claim_support_unreceipted_reference_rate_increase,
                self.max_runtime_drift_product_trace_receipt_claim_support_failed_result_reference_rate_increase,
                self.max_runtime_drift_product_trace_receipt_claim_support_fingerprint_mismatch_reference_rate_increase,
                self.max_runtime_drift_product_trace_receipt_claim_support_unsigned_reference_rate_increase,
                self.max_runtime_drift_product_trace_trajectory_audit_failed_trace_rate_increase,
                self.max_runtime_drift_product_trace_trajectory_audit_error_rate_increase,
                self.max_runtime_drift_product_trace_trajectory_audit_factual_rate_increase,
                self.max_runtime_drift_product_trace_trajectory_audit_referential_rate_increase,
                self.max_runtime_drift_product_trace_trajectory_audit_logical_rate_increase,
                self.max_runtime_drift_product_trace_trajectory_audit_procedural_rate_increase,
                self.max_runtime_drift_product_trace_trajectory_audit_scope_rate_increase,
                self.max_runtime_drift_product_trace_trajectory_audit_cascade_rate_increase,
                self.min_runtime_drift_product_trace_provenance_coverage_rate,
                self.min_runtime_drift_product_trace_provenance_supported_claim_evidence_coverage,
                self.max_runtime_drift_product_trace_provenance_missing_reference_rate_increase,
                self.max_runtime_drift_product_trace_provenance_unsupported_supported_claim_rate_increase,
                self.max_runtime_drift_product_trace_provenance_error_rate_increase,
                self.min_runtime_drift_product_trace_provenance_final_answer_evidence_reference_rate,
                self.min_runtime_drift_product_trace_citation_integrity_participating_trace_rate,
                self.min_runtime_drift_product_trace_citation_integrity_coverage_rate,
                self.max_runtime_drift_product_trace_citation_integrity_mismatch_rate_increase,
                self.max_runtime_drift_product_trace_citation_integrity_unresolved_rate_increase,
                self.max_runtime_drift_product_trace_citation_integrity_issue_rate_increase,
                self.max_runtime_drift_product_trace_citation_integrity_trace_gap_rate_increase,
                self.min_runtime_drift_current_trace_count,
            )
        )
        if drift_gate_selected and not drift_baseline_selected:
            raise ValueError("runtime drift gates require a runtime_drift_baseline path or registry record.")
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        for field_name in (
            "max_action_audit_error_rate",
            "max_action_audit_missing_retrieval_rate",
            "max_action_audit_missing_plan_retrieval_query_rate",
            "max_action_audit_malformed_payload_rate",
            "max_action_audit_unexpected_action_rate",
            "max_action_audit_unknown_claim_id_rate",
            "max_action_execution_missing_result_rate",
            "max_action_execution_unexpected_result_rate",
            "max_action_execution_request_id_mismatch_rate",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_rate_float(getattr(self, field_name), name=field_name),
            )
        limit = None if self.limit is None else int(self.limit)
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive when provided.")
        object.__setattr__(self, "trace_paths", trace_paths)
        object.__setattr__(self, "jsonl_paths", jsonl_paths)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "candidates", candidates)
        if self.replay_policy_path is not None:
            object.__setattr__(self, "replay_policy_path", Path(self.replay_policy_path))
        if self.runtime_policy_path is not None:
            object.__setattr__(self, "runtime_policy_path", Path(self.runtime_policy_path))
        if self.promotion_contract_path is not None:
            object.__setattr__(self, "promotion_contract_path", Path(self.promotion_contract_path))
        if self.runtime_recommended_policy_path is not None:
            object.__setattr__(
                self,
                "runtime_recommended_policy_path",
                Path(self.runtime_recommended_policy_path),
            )
        for field_name in (
            "runtime_drift_baseline_path",
            "runtime_drift_budget_policy_path",
            "runtime_drift_report_path",
            "runtime_drift_artifact_manifest_path",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, Path(value))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        if self.verification_report_path is not None:
            object.__setattr__(self, "verification_report_path", Path(self.verification_report_path))
        if self.fingerprint_cache_path is not None:
            object.__setattr__(self, "fingerprint_cache_path", Path(self.fingerprint_cache_path))
        if self.corpus_cache_path is not None:
            object.__setattr__(self, "corpus_cache_path", Path(self.corpus_cache_path))
        if self.corpus_source_cache_path is not None:
            object.__setattr__(self, "corpus_source_cache_path", Path(self.corpus_source_cache_path))
        if self.runtime_trace_records_cache_path is not None:
            object.__setattr__(
                self,
                "runtime_trace_records_cache_path",
                Path(self.runtime_trace_records_cache_path),
            )
        if self.selector_trace_inputs_path is not None:
            object.__setattr__(self, "selector_trace_inputs_path", Path(self.selector_trace_inputs_path))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        object.__setattr__(
            self,
            "runtime_drift_covered_fact_property_scopes",
            tuple(
                str(scope).strip()
                for scope in self.runtime_drift_covered_fact_property_scopes
                if str(scope).strip()
            ),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "redact_text", strict_bool(self.redact_text, name="redact_text"))
        object.__setattr__(
            self,
            "require_runtime_trace",
            strict_bool(self.require_runtime_trace, name="require_runtime_trace"),
        )
        object.__setattr__(self, "strict", strict_bool(self.strict, name="strict"))
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))
        object.__setattr__(self, "verify_manifest", strict_bool(self.verify_manifest, name="verify_manifest"))
        object.__setattr__(
            self,
            "allow_manifest_verification_failures",
            strict_bool(
                self.allow_manifest_verification_failures,
                name="allow_manifest_verification_failures",
            ),
        )
        object.__setattr__(
            self,
            "refresh_corpus_cache",
            strict_bool(
                self.refresh_corpus_cache,
                name="refresh_corpus_cache",
            ),
        )
        object.__setattr__(
            self,
            "refresh_corpus_source_cache",
            strict_bool(
                self.refresh_corpus_source_cache,
                name="refresh_corpus_source_cache",
            ),
        )
        object.__setattr__(
            self,
            "refresh_runtime_trace_records_cache",
            strict_bool(
                self.refresh_runtime_trace_records_cache,
                name="refresh_runtime_trace_records_cache",
            ),
        )
        object.__setattr__(
            self,
            "runtime_trace_scan_workers",
            strict_positive_int(self.runtime_trace_scan_workers, name="runtime_trace_scan_workers"),
        )
        object.__setattr__(
            self,
            "refresh_selector_trace_inputs",
            strict_bool(
                self.refresh_selector_trace_inputs,
                name="refresh_selector_trace_inputs",
            ),
        )

    @property
    def resolved_report_path(self) -> Path:
        """Return the top-level workflow report path."""
        return Path(self.output_dir) / "product-trace-replay-workflow.json"

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        """Return the top-level artifact manifest path."""
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return Path(self.output_dir) / "artifact-manifest.json"

    @property
    def resolved_verification_report_path(self) -> Path:
        """Return the top-level artifact manifest verification report path."""
        if self.verification_report_path is not None:
            return Path(self.verification_report_path)
        return Path(self.output_dir) / "manifest-verification.json"

    @property
    def resolved_runtime_drift_report_path(self) -> Path:
        """Return the child product runtime drift report path."""
        if self.runtime_drift_report_path is not None:
            return Path(self.runtime_drift_report_path)
        return Path(self.output_dir) / "runtime-drift" / "product-runtime-drift.json"

    @property
    def resolved_runtime_drift_artifact_manifest_path(self) -> Path:
        """Return the child product runtime drift artifact manifest path."""
        if self.runtime_drift_artifact_manifest_path is not None:
            return Path(self.runtime_drift_artifact_manifest_path)
        return Path(self.output_dir) / "runtime-drift" / "artifact-manifest.json"

    @property
    def resolved_action_audit_gate_report_path(self) -> Path:
        """Return the child action-audit gate report path."""
        return Path(self.output_dir) / "action-audit-gate.json"

    @property
    def resolved_action_execution_gate_report_path(self) -> Path:
        """Return the child action-execution alignment gate report path."""
        return Path(self.output_dir) / "action-execution-gate.json"


def run_product_trace_replay_workflow(config: ProductTraceReplayWorkflowConfig) -> dict[str, Any]:
    """Run corpus build, runtime baseline, and selector replay in one workflow."""
    fingerprint_cache = _load_fingerprint_cache(config)
    workflow_started = time.perf_counter()
    phase_timings: dict[str, dict[str, Any]] = {}
    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        corpus = _timed_phase(
            "corpus",
            phase_timings,
            lambda: _run_corpus(config, fingerprint_cache=fingerprint_cache),
        )
        corpus_trace_paths = tuple(Path(record["path"]) for record in _sequence(corpus.get("traces")))
        runtime_baseline = _timed_phase(
            "runtime_baseline",
            phase_timings,
            lambda: _run_runtime_baseline(config, corpus_trace_paths),
        )
        action_audit_gate = _timed_phase(
            "action_audit_gate",
            phase_timings,
            lambda: _run_action_audit_gate(config, runtime_baseline),
        )
        action_execution_gate = _timed_phase(
            "action_execution_gate",
            phase_timings,
            lambda: _run_action_execution_gate(config, runtime_baseline),
        )
        selector_replay = _timed_phase(
            "selector_replay",
            phase_timings,
            lambda: _run_selector_replay(
                config,
                corpus_trace_paths,
                runtime_pair_index_path=_nested(corpus, "paths", "runtime_pair_index"),
            ),
        )
        runtime_drift = _timed_phase(
            "runtime_drift",
            phase_timings,
            lambda: _run_runtime_drift_gate(config, runtime_baseline),
        )
        status = _workflow_status(
            corpus,
            runtime_baseline,
            action_audit_gate,
            action_execution_gate,
            selector_replay,
            runtime_drift,
        )
        report = {
            "schema_version": 1,
            "workflow": "product_trace_replay_workflow",
            "status": status,
            "decision": {
                "status": status,
                "blocking_reasons": _blocking_reasons(
                    corpus,
                    runtime_baseline,
                    action_audit_gate,
                    action_execution_gate,
                    selector_replay,
                    runtime_drift,
                ),
                "recommended_selector_candidate": _nested(
                    selector_replay,
                    "decision",
                    "recommended_candidate",
                ),
                "recommended_selector_policy_path": _nested(
                    selector_replay,
                    "decision",
                    "recommended_policy_path",
                ),
                "recommended_runtime_policy_path": _nested(
                    runtime_baseline,
                    "paths",
                    "recommended_policy",
                ),
            },
            "corpus": _corpus_summary(corpus),
            "runtime_baseline": _runtime_baseline_summary(runtime_baseline),
            "action_audit_gate": _action_audit_gate_summary(action_audit_gate),
            "action_execution_gate": _action_execution_gate_summary(action_execution_gate),
            "selector_replay": _selector_replay_summary(selector_replay),
            "runtime_drift": _runtime_drift_summary(runtime_drift),
            "cache_summary": _workflow_cache_summary(corpus, runtime_baseline, selector_replay),
            "optimization": _workflow_optimization_summary(runtime_baseline),
            "timing": _workflow_timing(phase_timings, started_at=workflow_started),
            "paths": {
                "report": str(config.resolved_report_path),
                "artifact_manifest": str(config.resolved_artifact_manifest_path),
                "output_dir": str(config.output_dir),
                "corpus_report": _nested(corpus, "paths", "report"),
                "corpus_manifest": _nested(corpus, "paths", "artifact_manifest"),
                "corpus_traces_dir": _nested(corpus, "paths", "traces_dir"),
                "corpus_runtime_pair_index": _nested(corpus, "paths", "runtime_pair_index"),
                "corpus_cache": _nested(corpus, "workflow_cache", "path"),
                "corpus_source_cache": _nested(corpus, "paths", "source_cache"),
                "runtime_baseline_report": _nested(runtime_baseline, "paths", "report"),
                "runtime_baseline_manifest": _nested(runtime_baseline, "paths", "artifact_manifest"),
                "runtime_recommended_policy": _nested(runtime_baseline, "paths", "recommended_policy"),
                "action_audit_gate_report": _nested(action_audit_gate, "paths", "report"),
                "action_execution_gate_report": _nested(action_execution_gate, "paths", "report"),
                "runtime_drift_report": _nested(runtime_drift, "paths", "report"),
                "runtime_drift_manifest": _nested(runtime_drift, "paths", "artifact_manifest"),
                "selector_replay_report": _nested(selector_replay, "paths", "report"),
                "selector_replay_manifest": _nested(selector_replay, "paths", "artifact_manifest"),
                "manifest_verification": (
                    str(config.resolved_verification_report_path) if config.verify_manifest else None
                ),
                "manifest_fingerprint_cache": (
                    None if config.fingerprint_cache_path is None else str(config.fingerprint_cache_path)
                ),
                "runtime_trace_records_cache": _nested(runtime_baseline, "paths", "trace_records_cache"),
                "selector_trace_inputs": _nested(selector_replay, "paths", "trace_inputs"),
            },
            "config": {
                "candidate_names": tuple(candidate.name for candidate in config.candidates),
                "trace_count": len(config.trace_paths),
                "jsonl_count": len(config.jsonl_paths),
                "replay_policy": None if config.replay_policy_path is None else str(config.replay_policy_path),
                "runtime_policy": None if config.runtime_policy_path is None else str(config.runtime_policy_path),
                "promotion_contract": (
                    None if config.promotion_contract_path is None else str(config.promotion_contract_path)
                ),
                "runtime_recommended_policy": (
                    None
                    if config.runtime_recommended_policy_path is None
                    else str(config.runtime_recommended_policy_path)
                ),
                "runtime_drift_baseline": (
                    None
                    if config.runtime_drift_baseline_path is None
                    else str(config.runtime_drift_baseline_path)
                ),
                "runtime_drift_baseline_key": config.runtime_drift_baseline_key,
                "runtime_drift_baseline_name": config.runtime_drift_baseline_name,
                "runtime_drift_baseline_version": config.runtime_drift_baseline_version,
                "runtime_drift_budget_policy": (
                    None
                    if config.runtime_drift_budget_policy_path is None
                    else str(config.runtime_drift_budget_policy_path)
                ),
                "runtime_drift_budget_policy_key": config.runtime_drift_budget_policy_key,
                "runtime_drift_report": (
                    None
                    if not _runtime_drift_configured(config)
                    else str(config.resolved_runtime_drift_report_path)
                ),
                "runtime_drift_artifact_manifest": (
                    None
                    if not _runtime_drift_configured(config)
                    else str(config.resolved_runtime_drift_artifact_manifest_path)
                ),
                "runtime_drift_gates": _runtime_drift_gate_config(config),
                "action_audit_gates": _action_audit_gate_config(config),
                "action_audit_gate_report": (
                    None
                    if not _action_audit_gate_configured(config)
                    else str(config.resolved_action_audit_gate_report_path)
                ),
                "action_execution_gates": _action_execution_gate_config(config),
                "action_execution_gate_report": (
                    None
                    if not _action_execution_gate_configured(config)
                    else str(config.resolved_action_execution_gate_report_path)
                ),
                "redact_text": config.redact_text,
                "require_runtime_trace": config.require_runtime_trace,
                "strict": config.strict,
                "limit": config.limit,
                "compact_json": config.compact_json,
                "fingerprint_cache": (
                    None if config.fingerprint_cache_path is None else str(config.fingerprint_cache_path)
                ),
                "corpus_cache": None if config.corpus_cache_path is None else str(config.corpus_cache_path),
                "refresh_corpus_cache": config.refresh_corpus_cache,
                "corpus_source_cache": (
                    None if config.corpus_source_cache_path is None else str(config.corpus_source_cache_path)
                ),
                "refresh_corpus_source_cache": config.refresh_corpus_source_cache,
                "runtime_trace_records_cache": (
                    None
                    if config.runtime_trace_records_cache_path is None
                    else str(config.runtime_trace_records_cache_path)
                ),
                "refresh_runtime_trace_records_cache": config.refresh_runtime_trace_records_cache,
                "runtime_trace_scan_workers": config.runtime_trace_scan_workers,
                "selector_trace_inputs": (
                    None if config.selector_trace_inputs_path is None else str(config.selector_trace_inputs_path)
                ),
                "refresh_selector_trace_inputs": config.refresh_selector_trace_inputs,
                "metadata": dict(config.metadata),
            },
        }
        _write_report_and_manifest(config, report, fingerprint_cache=fingerprint_cache)
        if config.verify_manifest:
            report["manifest_verification"] = _write_manifest_verification(
                config,
                fingerprint_cache=fingerprint_cache,
            )
        _record_registry(config, report, fingerprint_cache=fingerprint_cache)
        return report
    finally:
        _save_fingerprint_cache(config, fingerprint_cache)


def _timed_phase(
    name: str,
    timings: MutableMapping[str, dict[str, Any]],
    func: Callable[[], _T],
) -> _T:
    started = time.perf_counter()
    try:
        return func()
    finally:
        timings[name] = {
            "seconds": _round_seconds(time.perf_counter() - started),
        }


def _workflow_timing(
    phases: Mapping[str, Mapping[str, Any]],
    *,
    started_at: float,
) -> dict[str, Any]:
    phase_payload = {name: dict(payload) for name, payload in phases.items()}
    phase_total = sum(
        float(payload.get("seconds", 0.0))
        for payload in phase_payload.values()
        if not isinstance(payload.get("seconds"), bool)
    )
    return {
        "total_seconds": _round_seconds(time.perf_counter() - started_at),
        "phase_total_seconds": _round_seconds(phase_total),
        "phases": phase_payload,
    }


def _workflow_cache_summary(
    corpus: Mapping[str, Any],
    runtime_baseline: Mapping[str, Any],
    selector_replay: Mapping[str, Any],
) -> dict[str, Any]:
    corpus_cache = _mapping(corpus.get("workflow_cache"))
    corpus_source_cache = _mapping(corpus.get("source_cache"))
    runtime_cache = _mapping(_nested(runtime_baseline, "config", "trace_record_cache"))
    selector_cache = _mapping(_nested(selector_replay, "config", "trace_inputs"))
    corpus_source_entry = _cache_entry_summary(corpus_source_cache)
    if corpus_cache.get("source") == "corpus_cache" and corpus_source_entry.get("enabled") is True:
        corpus_source_entry.update({
            "enabled": False,
            "source": "covered_by_corpus_cache",
            "hit": None,
            "partial_hit": None,
            "written": False,
        })
    caches = {
        "corpus": _cache_entry_summary(corpus_cache),
        "corpus_source": corpus_source_entry,
        "runtime_trace_records": _cache_entry_summary(runtime_cache),
        "selector_trace_inputs": _cache_entry_summary(selector_cache),
    }
    enabled = [entry for entry in caches.values() if entry.get("enabled") is True]
    hit_count = sum(1 for entry in enabled if entry.get("hit") is True)
    written_count = sum(1 for entry in enabled if entry.get("written") is True)
    return {
        "enabled_count": len(enabled),
        "hit_count": hit_count,
        "written_count": written_count,
        "hit_rate": _safe_div(hit_count, len(enabled)),
        **caches,
    }


def _cache_entry_summary(cache: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": cache.get("enabled"),
        "source": cache.get("source"),
        "path": cache.get("path"),
        "hit": cache.get("cache_hit"),
        "partial_hit": cache.get("cache_partial_hit"),
        "written": cache.get("cache_written"),
        "refresh": cache.get("refresh"),
        "invalidation_reason": cache.get("invalidation_reason"),
    }


def _run_corpus(
    config: ProductTraceReplayWorkflowConfig,
    *,
    fingerprint_cache: MutableMapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cache_path = config.corpus_cache_path
    invalidation_reason = None
    if cache_path is not None and cache_path.exists() and not config.refresh_corpus_cache:
        cached = _load_corpus_cache(config, fingerprint_cache=fingerprint_cache)
        if cached is not None:
            corpus, payload = cached
            corpus["workflow_cache"] = {
                "enabled": True,
                "source": "corpus_cache",
                "path": str(cache_path),
                "cache_hit": True,
                "cache_written": False,
                "source_count": len(_sequence(payload.get("sources"))),
                "refresh": False,
                "invalidation_reason": None,
            }
            return corpus
        invalidation_reason = "fingerprint_config_or_schema_mismatch"

    corpus = build_product_trace_corpus(
        ProductTraceCorpusConfig(
            trace_paths=config.trace_paths,
            jsonl_paths=config.jsonl_paths,
            output_dir=Path(config.output_dir) / "corpus",
            redact_text=config.redact_text,
            require_runtime_trace=config.require_runtime_trace,
            strict=config.strict,
            limit=config.limit,
            compact_json=config.compact_json,
            source_cache_path=config.corpus_source_cache_path,
            refresh_source_cache=config.refresh_corpus_source_cache,
            metadata={
                "source": "run_product_trace_replay_workflow",
                **dict(config.metadata),
            },
        )
    )
    if cache_path is not None:
        payload = _corpus_cache_payload(config, corpus, fingerprint_cache=fingerprint_cache)
        _write_json(cache_path, payload, compact=config.compact_json)
    corpus["workflow_cache"] = {
        "enabled": cache_path is not None,
        "source": "corpus_build",
        "path": None if cache_path is None else str(cache_path),
        "cache_hit": False,
        "cache_written": cache_path is not None,
        "source_count": len(config.trace_paths) + len(config.jsonl_paths),
        "refresh": config.refresh_corpus_cache,
        "invalidation_reason": invalidation_reason,
    }
    return corpus


def _load_corpus_cache(
    config: ProductTraceReplayWorkflowConfig,
    *,
    fingerprint_cache: MutableMapping[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], Mapping[str, Any]] | None:
    if config.corpus_cache_path is None:
        return None
    try:
        payload = json.loads(Path(config.corpus_cache_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("schema_version") != 1:
        return None
    if payload.get("workflow") != "product_trace_replay_workflow_corpus_cache":
        return None
    if _mapping(payload.get("config")).get("signature") != _corpus_cache_signature(config):
        return None
    if not _corpus_sources_match(config, _sequence(payload.get("sources")), fingerprint_cache=fingerprint_cache):
        return None
    child_paths = _corpus_child_paths(config)
    outputs = _mapping(payload.get("outputs"))
    if not _corpus_outputs_match(
        outputs,
        child_paths=child_paths,
        fingerprint_cache=fingerprint_cache,
    ):
        return None
    report_path = child_paths["report"]
    try:
        corpus = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(corpus, Mapping):
        return None
    if corpus.get("workflow") != "product_trace_corpus":
        return None
    return dict(corpus), payload


def _corpus_cache_payload(
    config: ProductTraceReplayWorkflowConfig,
    corpus: Mapping[str, Any],
    *,
    fingerprint_cache: MutableMapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    child_paths = _corpus_child_paths(config)
    return {
        "schema_version": 1,
        "workflow": "product_trace_replay_workflow_corpus_cache",
        "config": {
            "signature": _corpus_cache_signature(config),
            "payload": _corpus_cache_config_payload(config),
        },
        "summary": {
            "accepted_count": _nested(corpus, "summary", "accepted_count"),
            "rejected_count": _nested(corpus, "summary", "rejected_count"),
            "runtime_pair_index_record_count": _nested(corpus, "runtime_pair_index", "record_count"),
        },
        "paths": {
            "corpus_cache": None if config.corpus_cache_path is None else str(config.corpus_cache_path),
            **{key: str(value) for key, value in child_paths.items()},
        },
        "sources": _corpus_source_fingerprints(config, fingerprint_cache=fingerprint_cache),
        "outputs": {
            key: fingerprint_path(value, fingerprint_cache=fingerprint_cache).to_dict()
            for key, value in child_paths.items()
        },
    }


def _corpus_child_paths(config: ProductTraceReplayWorkflowConfig) -> dict[str, Path]:
    output_dir = Path(config.output_dir) / "corpus"
    paths = {
        "report": output_dir / "product-trace-corpus.json",
        "artifact_manifest": output_dir / "artifact-manifest.json",
        "traces_dir": output_dir / "traces",
        "runtime_pair_index": output_dir / "runtime-pair-index.json",
    }
    if config.corpus_source_cache_path is not None:
        paths["source_cache"] = Path(config.corpus_source_cache_path)
    return paths


def _corpus_source_fingerprints(
    config: ProductTraceReplayWorkflowConfig,
    *,
    fingerprint_cache: MutableMapping[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    records = []
    for source_kind, paths in (("trace", config.trace_paths), ("jsonl", config.jsonl_paths)):
        for path in paths:
            records.append({
                "kind": source_kind,
                "path": str(path),
                "fingerprint": fingerprint_path(path, fingerprint_cache=fingerprint_cache).to_dict(),
            })
    return tuple(records)


def _corpus_sources_match(
    config: ProductTraceReplayWorkflowConfig,
    sources: Sequence[Any],
    *,
    fingerprint_cache: MutableMapping[str, dict[str, Any]] | None = None,
) -> bool:
    expected_sources = tuple(
        (source_kind, str(path))
        for source_kind, paths in (("trace", config.trace_paths), ("jsonl", config.jsonl_paths))
        for path in paths
    )
    if len(sources) != len(expected_sources):
        return False
    for source, expected in zip(sources, expected_sources, strict=True):
        if not isinstance(source, Mapping):
            return False
        expected_kind, expected_path = expected
        if source.get("kind") != expected_kind or str(source.get("path")) != expected_path:
            return False
        expected_fingerprint = _mapping(source.get("fingerprint"))
        if not expected_fingerprint:
            return False
        actual = fingerprint_path(expected_path, fingerprint_cache=fingerprint_cache).to_dict()
        if not _fingerprint_matches(expected_fingerprint, actual):
            return False
    return True


def _corpus_outputs_match(
    outputs: Mapping[str, Any],
    *,
    child_paths: Mapping[str, Path],
    fingerprint_cache: MutableMapping[str, dict[str, Any]] | None = None,
) -> bool:
    if set(outputs) != set(child_paths):
        return False
    for key, expected_path in child_paths.items():
        record = outputs.get(key)
        if not isinstance(record, Mapping):
            return False
        if str(record.get("path")) != str(expected_path):
            return False
        actual = fingerprint_path(expected_path, fingerprint_cache=fingerprint_cache).to_dict()
        if not _fingerprint_matches(record, actual):
            return False
    return True


def _corpus_cache_signature(config: ProductTraceReplayWorkflowConfig) -> str:
    return json.dumps(
        _corpus_cache_config_payload(config),
        sort_keys=True,
        separators=(",", ":"),
    )


def _corpus_cache_config_payload(config: ProductTraceReplayWorkflowConfig) -> dict[str, Any]:
    child_paths = _corpus_child_paths(config)
    return {
        "trace_summary_schema_version": _TRACE_SUMMARY_SCHEMA_VERSION,
        "trace_paths": [str(path) for path in config.trace_paths],
        "jsonl_paths": [str(path) for path in config.jsonl_paths],
        "redact_text": config.redact_text,
        "require_runtime_trace": config.require_runtime_trace,
        "strict": config.strict,
        "limit": config.limit,
        "compact_json": config.compact_json,
        "source_cache": None if config.corpus_source_cache_path is None else str(config.corpus_source_cache_path),
        "refresh_source_cache": config.refresh_corpus_source_cache,
        "metadata": {
            "source": "run_product_trace_replay_workflow",
            **dict(config.metadata),
        },
        "child_paths": {key: str(value) for key, value in child_paths.items()},
    }


def _fingerprint_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    return all(
        expected.get(field_name) == actual.get(field_name)
        for field_name in ("exists", "kind", "sha256", "size_bytes", "file_count")
    )


def _run_runtime_baseline(
    config: ProductTraceReplayWorkflowConfig,
    trace_paths: Sequence[Path],
) -> dict[str, Any]:
    if not trace_paths:
        return _skipped_child_report("product_runtime_baseline", reason="no valid corpus traces")
    output_dir = Path(config.output_dir) / "runtime-baseline"
    return build_product_runtime_baseline(
        ProductRuntimeBaselineConfig(
            trace_paths=trace_paths,
            report_path=output_dir / "product-runtime-baseline.json",
            policy_path=config.runtime_policy_path,
            promotion_contract_path=config.promotion_contract_path,
            trace_records_cache_path=config.runtime_trace_records_cache_path,
            refresh_trace_records_cache=config.refresh_runtime_trace_records_cache,
            trace_scan_workers=config.runtime_trace_scan_workers,
            recommended_policy_path=config.runtime_recommended_policy_path,
            artifact_manifest_path=output_dir / "artifact-manifest.json",
            compact_json=config.compact_json,
            metadata={
                "source": "run_product_trace_replay_workflow",
                **dict(config.metadata),
            },
        )
    )


def _run_selector_replay(
    config: ProductTraceReplayWorkflowConfig,
    trace_paths: Sequence[Path],
    *,
    runtime_pair_index_path: str | Path | None = None,
) -> dict[str, Any]:
    if not trace_paths:
        return _skipped_child_report("runtime_profile_selector_replay", reason="no valid corpus traces")
    output_dir = Path(config.output_dir) / "selector-replay"
    return run_runtime_profile_selector_replay(
        RuntimeProfileSelectorReplayConfig(
            trace_paths=trace_paths,
            output_dir=output_dir,
            candidates=config.candidates,
            replay_policy_path=config.replay_policy_path,
            runtime_pair_index_path=(
                None if runtime_pair_index_path is None else Path(runtime_pair_index_path)
            ),
            trace_inputs_path=config.selector_trace_inputs_path,
            refresh_trace_inputs=config.refresh_selector_trace_inputs,
            compact_json=config.compact_json,
            metadata={
                "source": "run_product_trace_replay_workflow",
                **dict(config.metadata),
            },
        )
    )


def _run_action_audit_gate(
    config: ProductTraceReplayWorkflowConfig,
    runtime_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if not _action_audit_gate_configured(config):
        return {
            "schema_version": 1,
            "workflow": "product_trace_action_audit_gate",
            "status": "not_configured",
            "decision": {
                "status": "not_configured",
                "blocking_reasons": (),
            },
            "summary": {
                "gate_enabled": False,
                "passed": None,
                "blocked_metric_count": 0,
            },
            "checks": (),
            "paths": {
                "report": None,
            },
        }
    action_audit = _mapping(_nested(runtime_baseline, "summary", "action_audit"))
    if not action_audit:
        payload = {
            "schema_version": 1,
            "workflow": "product_trace_action_audit_gate",
            "status": "blocked",
            "decision": {
                "status": "blocked",
                "blocking_reasons": ("runtime baseline action-audit summary is unavailable",),
            },
            "summary": {
                "gate_enabled": True,
                "passed": False,
                "blocked_metric_count": 1,
            },
            "checks": (),
            "paths": {
                "report": str(config.resolved_action_audit_gate_report_path),
            },
        }
        _write_json(config.resolved_action_audit_gate_report_path, payload, compact=config.compact_json)
        return payload

    gate_config = _action_audit_gate_config(config)
    checks = tuple(_action_audit_gate_checks(action_audit, gate_config))
    blocking_reasons = tuple(
        str(check.get("reason"))
        for check in checks
        if check.get("status") == "blocked" and check.get("reason")
    )
    status = "blocked" if blocking_reasons else "promote"
    summary = {
        "gate_enabled": True,
        "passed": not blocking_reasons,
        "trace_count": action_audit.get("source_trace_count"),
        "available_trace_count": action_audit.get("available_trace_count"),
        "passed_trace_count": action_audit.get("passed_trace_count"),
        "failed_trace_count": action_audit.get("failed_trace_count"),
        "failed_trace_rate": action_audit.get("failed_trace_rate"),
        "issue_count": action_audit.get("issue_count"),
        "error_count": action_audit.get("error_count"),
        "warning_count": action_audit.get("warning_count"),
        "error_rate": action_audit.get("error_rate"),
        "missing_decision_action_rate": action_audit.get("missing_decision_action_rate"),
        "missing_retrieval_action_rate": action_audit.get("missing_retrieval_action_rate"),
        "missing_plan_retrieval_query_rate": action_audit.get(
            "missing_plan_retrieval_query_rate"
        ),
        "malformed_payload_rate": action_audit.get("malformed_payload_rate"),
        "unexpected_action_rate": action_audit.get("unexpected_action_rate"),
        "unknown_claim_id_rate": action_audit.get("unknown_claim_id_rate"),
        "blocked_metric_count": len(blocking_reasons),
        "checked_metric_count": len(checks),
    }
    payload = {
        "schema_version": 1,
        "workflow": "product_trace_action_audit_gate",
        "status": status,
        "decision": {
            "status": status,
            "blocking_reasons": blocking_reasons,
        },
        "summary": summary,
        "checks": checks,
        "action_audit": dict(action_audit),
        "config": gate_config,
        "paths": {
            "report": str(config.resolved_action_audit_gate_report_path),
        },
    }
    _write_json(config.resolved_action_audit_gate_report_path, payload, compact=config.compact_json)
    return payload


def _run_action_execution_gate(
    config: ProductTraceReplayWorkflowConfig,
    runtime_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if not _action_execution_gate_configured(config):
        return {
            "schema_version": 1,
            "workflow": "product_trace_action_execution_gate",
            "status": "not_configured",
            "decision": {
                "status": "not_configured",
                "blocking_reasons": (),
            },
            "summary": {
                "gate_enabled": False,
                "passed": None,
                "blocked_metric_count": 0,
            },
            "checks": (),
            "paths": {
                "report": None,
            },
        }
    action_execution = _mapping(_nested(runtime_baseline, "summary", "action_execution"))
    if not action_execution:
        payload = {
            "schema_version": 1,
            "workflow": "product_trace_action_execution_gate",
            "status": "blocked",
            "decision": {
                "status": "blocked",
                "blocking_reasons": ("runtime baseline action-execution summary is unavailable",),
            },
            "summary": {
                "gate_enabled": True,
                "passed": False,
                "blocked_metric_count": 1,
            },
            "checks": (),
            "paths": {
                "report": str(config.resolved_action_execution_gate_report_path),
            },
        }
        _write_json(config.resolved_action_execution_gate_report_path, payload, compact=config.compact_json)
        return payload

    gate_config = _action_execution_gate_config(config)
    checks = tuple(_action_execution_gate_checks(action_execution, gate_config))
    blocking_reasons = tuple(
        str(check.get("reason"))
        for check in checks
        if check.get("status") == "blocked" and check.get("reason")
    )
    status = "blocked" if blocking_reasons else "promote"
    summary = {
        "gate_enabled": True,
        "passed": not blocking_reasons,
        "trace_count": action_execution.get("source_trace_count"),
        "available_trace_count": action_execution.get("available_trace_count"),
        "alignment_available_trace_count": action_execution.get("alignment_available_trace_count"),
        "alignment_failed_trace_count": action_execution.get("alignment_failed_trace_count"),
        "alignment_failed_trace_rate": action_execution.get("alignment_failed_trace_rate"),
        "planned_action_count": action_execution.get("planned_action_count"),
        "result_count": action_execution.get("result_count"),
        "missing_result_count": action_execution.get("missing_result_count"),
        "missing_result_rate": action_execution.get("missing_result_rate"),
        "unexpected_result_count": action_execution.get("unexpected_result_count"),
        "unexpected_result_rate": action_execution.get("unexpected_result_rate"),
        "request_id_mismatch_count": action_execution.get("request_id_mismatch_count"),
        "request_id_mismatch_rate": action_execution.get("request_id_mismatch_rate"),
        "blocked_metric_count": len(blocking_reasons),
        "checked_metric_count": len(checks),
    }
    payload = {
        "schema_version": 1,
        "workflow": "product_trace_action_execution_gate",
        "status": status,
        "decision": {
            "status": status,
            "blocking_reasons": blocking_reasons,
        },
        "summary": summary,
        "checks": checks,
        "action_execution": dict(action_execution),
        "config": gate_config,
        "paths": {
            "report": str(config.resolved_action_execution_gate_report_path),
        },
    }
    _write_json(config.resolved_action_execution_gate_report_path, payload, compact=config.compact_json)
    return payload


def _run_runtime_drift_gate(
    config: ProductTraceReplayWorkflowConfig,
    runtime_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if not _runtime_drift_configured(config):
        return {
            "schema_version": 1,
            "workflow": "product_runtime_drift_comparison",
            "status": "not_configured",
            "decision": {
                "status": "not_configured",
                "blocking_reasons": (),
            },
            "summary": {
                "gate_enabled": False,
                "drift_gate_enabled": False,
                "runtime_budget_policy_gate_enabled": False,
                "blocked_metric_count": 0,
            },
            "paths": {
                "report": None,
                "artifact_manifest": None,
            },
        }
    current_path = _nested(runtime_baseline, "paths", "report")
    if current_path is None:
        return _skipped_child_report(
            "product_runtime_drift_comparison",
            reason="runtime baseline report is unavailable",
        )
    child_name = None
    child_version = None
    if config.registry_path is not None and config.name and config.version:
        child_name = f"{config.name}-runtime-drift"
        child_version = str(config.version)
    return compare_product_runtime_baselines(
        current_path=Path(str(current_path)),
        baseline_path=config.runtime_drift_baseline_path,
        registry_path=config.registry_path,
        baseline_key=config.runtime_drift_baseline_key,
        baseline_name=config.runtime_drift_baseline_name,
        baseline_version=config.runtime_drift_baseline_version,
        runtime_budget_policy_path=config.runtime_drift_budget_policy_path,
        runtime_budget_policy_key=config.runtime_drift_budget_policy_key,
        report_path=config.resolved_runtime_drift_report_path,
        artifact_manifest_path=config.resolved_runtime_drift_artifact_manifest_path,
        name=child_name,
        version=child_version,
        metadata={
            "source": "run_product_trace_replay_workflow",
            **dict(config.metadata),
        },
        compact_json=config.compact_json,
        **_runtime_drift_gate_config(config),
    )


def _skipped_child_report(workflow: str, *, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow": workflow,
        "status": "blocked",
        "decision": {
            "status": "blocked",
            "blocking_reasons": (reason,),
        },
    }


def _workflow_status(
    corpus: Mapping[str, Any],
    runtime_baseline: Mapping[str, Any],
    action_audit_gate: Mapping[str, Any],
    action_execution_gate: Mapping[str, Any],
    selector_replay: Mapping[str, Any],
    runtime_drift: Mapping[str, Any],
) -> str:
    child_statuses = (
        corpus.get("status"),
        runtime_baseline.get("status"),
        action_audit_gate.get("status"),
        action_execution_gate.get("status"),
        selector_replay.get("status"),
        runtime_drift.get("status"),
    )
    if "blocked" in child_statuses:
        return "blocked"
    if corpus.get("status") == "partial":
        return "partial"
    if selector_replay.get("status") == "promote":
        return "promote"
    return "observed"


def _blocking_reasons(
    corpus: Mapping[str, Any],
    runtime_baseline: Mapping[str, Any],
    action_audit_gate: Mapping[str, Any],
    action_execution_gate: Mapping[str, Any],
    selector_replay: Mapping[str, Any],
    runtime_drift: Mapping[str, Any],
) -> tuple[str, ...]:
    reasons = []
    for child_name, child in (
        ("corpus", corpus),
        ("runtime_baseline", runtime_baseline),
        ("action_audit_gate", action_audit_gate),
        ("action_execution_gate", action_execution_gate),
        ("selector_replay", selector_replay),
        ("runtime_drift", runtime_drift),
    ):
        if child.get("status") != "blocked":
            continue
        child_reasons = _sequence(_nested(child, "decision", "blocking_reasons"))
        if not child_reasons:
            reasons.append(f"{child_name}: blocked")
            continue
        reasons.extend(f"{child_name}: {reason}" for reason in child_reasons)
    return tuple(str(reason) for reason in reasons)


def _runtime_drift_configured(config: ProductTraceReplayWorkflowConfig) -> bool:
    return bool(config.runtime_drift_covered_fact_property_scopes) or any(
        value is not None
        for value in (
            config.runtime_drift_baseline_path,
            config.runtime_drift_baseline_key,
            config.runtime_drift_baseline_name,
            config.runtime_drift_baseline_version,
            config.runtime_drift_budget_policy_path,
            config.runtime_drift_budget_policy_key,
            config.max_runtime_drift_total_seconds_mean_ratio,
            config.max_runtime_drift_total_seconds_p95_ratio,
            config.max_runtime_drift_mean_route_duration_ratio,
            config.max_runtime_drift_p95_route_duration_ratio,
            config.max_runtime_drift_mean_attempted_route_count_delta,
            config.max_runtime_drift_retrieval_use_rate_delta,
            config.max_runtime_drift_cache_hit_rate_drop,
            config.max_runtime_drift_verification_skip_rate_drop,
            config.min_runtime_drift_pre_generation_risk_coverage_rate,
            config.min_runtime_drift_pre_generation_learned_risk_coverage_rate,
            config.max_runtime_drift_pre_generation_audit_profile_rate_increase,
            config.max_runtime_drift_pre_generation_learned_risk_routed_rate_increase,
            config.max_runtime_drift_pre_generation_learned_risk_probability_mean_increase,
            config.min_runtime_drift_promotion_contract_coverage,
            config.min_runtime_drift_pre_generation_probe_comparison_coverage,
            config.min_runtime_drift_pre_generation_probe_comparison_manifest_verified_rate,
            config.min_runtime_drift_pre_generation_probe_comparison_model_count,
            config.min_runtime_drift_pre_generation_probe_comparison_run_count,
            config.min_runtime_drift_pre_generation_probe_comparison_redline_pass_rate,
            config.max_runtime_drift_pre_generation_probe_comparison_best_test_label_auroc_drop,
            config.max_runtime_drift_pre_generation_probe_comparison_best_redline_auroc_drop,
            config.max_runtime_drift_pre_generation_probe_comparison_best_redline_margin_drop,
            config.min_runtime_drift_claim_factuality_probe_comparison_coverage,
            config.min_runtime_drift_claim_factuality_probe_comparison_manifest_verified_rate,
            config.min_runtime_drift_claim_factuality_probe_comparison_model_count,
            config.min_runtime_drift_claim_factuality_probe_comparison_run_count,
            config.min_runtime_drift_claim_factuality_probe_comparison_dataset_count,
            config.min_runtime_drift_claim_factuality_probe_comparison_redline_pass_rate,
            config.max_runtime_drift_claim_factuality_probe_comparison_best_test_label_auroc_drop,
            config.max_runtime_drift_claim_factuality_probe_comparison_best_test_selective_accuracy_drop,
            config.max_runtime_drift_claim_factuality_probe_comparison_best_test_selective_coverage_drop,
            config.max_runtime_drift_claim_factuality_probe_comparison_best_redline_auroc_drop,
            config.max_runtime_drift_claim_factuality_probe_comparison_best_redline_margin_drop,
            config.min_runtime_drift_counterfactual_verification_coverage,
            config.min_runtime_drift_counterfactual_verification_manifest_verified_rate,
            config.min_runtime_drift_counterfactual_verification_record_count,
            config.min_runtime_drift_counterfactual_verification_pass_rate,
            config.max_runtime_drift_counterfactual_verification_false_invariance_rate,
            config.max_runtime_drift_counterfactual_verification_flip_success_count_drop,
            config.min_runtime_drift_evidence_handoff_coverage,
            config.min_runtime_drift_evidence_handoff_manifest_verified_rate,
            config.min_runtime_drift_evidence_handoff_present_metric_rate,
            config.max_runtime_drift_evidence_handoff_missing_metric_rate,
            config.max_runtime_drift_evidence_handoff_missing_metric_count,
            config.max_runtime_drift_evidence_handoff_blocked_group_count,
            config.min_runtime_drift_evidence_handoff_promoted_group_rate,
            config.min_runtime_drift_fact_selfcheck_gate_coverage,
            config.min_runtime_drift_fact_selfcheck_gate_report_present_rate,
            config.min_runtime_drift_fact_selfcheck_gate_manifest_present_rate,
            config.min_runtime_drift_fact_selfcheck_gate_manifest_verified_rate,
            config.min_runtime_drift_fact_selfcheck_gate_passed_rate,
            config.min_runtime_drift_fact_selfcheck_gate_run_count,
            config.max_runtime_drift_fact_selfcheck_gate_failed_run_count,
            config.min_runtime_drift_fact_selfcheck_gate_min_executed_rate,
            config.min_runtime_drift_fact_selfcheck_gate_min_decided_rate,
            config.max_runtime_drift_fact_selfcheck_gate_max_not_applicable_rate,
            config.min_runtime_drift_fact_selfcheck_gate_min_claim_triples_per_record,
            config.min_runtime_drift_fact_selfcheck_gate_min_sample_triples_per_record,
            config.min_runtime_drift_triple_extraction_fixture_matrix_coverage,
            config.max_runtime_drift_triple_extraction_fixture_matrix_mean_best_f1_drop,
            config.max_runtime_drift_triple_extraction_fixture_matrix_mean_f1_lift_drop,
            config.min_runtime_drift_triple_claim_coverage,
            config.min_runtime_drift_triple_audit_claim_coverage,
            config.min_runtime_drift_triple_audit_pass_rate,
            config.min_runtime_drift_triple_slot_coverage,
            config.min_runtime_drift_world_model_participating_trace_rate,
            config.min_runtime_drift_world_model_coverage_rate,
            config.max_runtime_drift_world_model_conflict_rate_increase,
            config.max_runtime_drift_world_model_low_agreement_rate_increase,
            config.max_runtime_drift_world_model_trace_gap_rate_increase,
            config.min_runtime_drift_context_sensitivity_participating_trace_rate,
            config.min_runtime_drift_context_sensitivity_coverage_rate,
            config.max_runtime_drift_context_sensitivity_flagged_result_rate_increase,
            config.max_runtime_drift_context_sensitivity_trace_gap_rate_increase,
            config.max_runtime_drift_context_sensitivity_max_flagged_rate_increase,
            config.max_runtime_drift_context_sensitivity_max_ratio_increase,
            config.min_runtime_drift_counterfactual_robustness_participating_trace_rate,
            config.min_runtime_drift_counterfactual_robustness_coverage_rate,
            config.min_runtime_drift_counterfactual_robustness_pass_rate,
            config.min_runtime_drift_counterfactual_robustness_flip_success_rate,
            config.max_runtime_drift_counterfactual_robustness_false_invariance_rate_increase,
            config.max_runtime_drift_counterfactual_robustness_trace_gap_rate_increase,
            config.min_runtime_drift_claim_risk_localization_coverage_rate,
            config.max_runtime_drift_claim_risk_localization_high_risk_claim_count_increase,
            config.max_runtime_drift_claim_risk_localization_medium_or_high_risk_claim_count_increase,
            config.max_runtime_drift_claim_risk_localization_entity_candidate_observation_count_increase,
            config.max_runtime_drift_claim_risk_localization_unique_entity_candidate_count_increase,
            config.max_runtime_drift_claim_risk_localization_high_risk_entity_candidate_count_increase,
            config.max_runtime_drift_claim_risk_localization_medium_or_high_entity_candidate_count_increase,
            config.min_runtime_drift_covered_fact_property_metric_count,
            config.min_runtime_drift_covered_fact_min_records,
            config.min_runtime_drift_covered_fact_min_source_documents,
            config.max_runtime_drift_covered_fact_min_decision_accuracy_drop,
            config.max_runtime_drift_covered_fact_max_false_supported_rate_increase,
            config.max_runtime_drift_covered_fact_min_false_refuted_rate_drop,
            config.max_runtime_drift_product_trace_action_audit_error_rate_increase,
            config.max_runtime_drift_product_trace_action_audit_missing_retrieval_action_rate_increase,
            config.max_runtime_drift_product_trace_action_audit_missing_plan_retrieval_query_rate_increase,
            config.max_runtime_drift_product_trace_action_audit_malformed_payload_rate_increase,
            config.max_runtime_drift_product_trace_action_audit_unexpected_action_rate_increase,
            config.max_runtime_drift_product_trace_action_audit_unknown_claim_id_rate_increase,
            config.max_runtime_drift_product_trace_action_execution_alignment_failed_trace_rate_increase,
            config.max_runtime_drift_product_trace_action_execution_missing_result_rate_increase,
            config.max_runtime_drift_product_trace_action_execution_unexpected_result_rate_increase,
            config.max_runtime_drift_product_trace_action_execution_request_id_mismatch_rate_increase,
            config.min_runtime_drift_product_trace_receipt_claim_support_reference_support_rate,
            config.max_runtime_drift_product_trace_receipt_claim_support_unsupported_reference_rate_increase,
            config.max_runtime_drift_product_trace_receipt_claim_support_missing_reference_rate_increase,
            config.max_runtime_drift_product_trace_receipt_claim_support_unreceipted_reference_rate_increase,
            config.max_runtime_drift_product_trace_receipt_claim_support_failed_result_reference_rate_increase,
            config.max_runtime_drift_product_trace_receipt_claim_support_fingerprint_mismatch_reference_rate_increase,
            config.max_runtime_drift_product_trace_receipt_claim_support_unsigned_reference_rate_increase,
            config.max_runtime_drift_product_trace_trajectory_audit_failed_trace_rate_increase,
            config.max_runtime_drift_product_trace_trajectory_audit_error_rate_increase,
            config.max_runtime_drift_product_trace_trajectory_audit_factual_rate_increase,
            config.max_runtime_drift_product_trace_trajectory_audit_referential_rate_increase,
            config.max_runtime_drift_product_trace_trajectory_audit_logical_rate_increase,
            config.max_runtime_drift_product_trace_trajectory_audit_procedural_rate_increase,
            config.max_runtime_drift_product_trace_trajectory_audit_scope_rate_increase,
            config.max_runtime_drift_product_trace_trajectory_audit_cascade_rate_increase,
            config.min_runtime_drift_product_trace_provenance_coverage_rate,
            config.min_runtime_drift_product_trace_provenance_supported_claim_evidence_coverage,
            config.max_runtime_drift_product_trace_provenance_missing_reference_rate_increase,
            config.max_runtime_drift_product_trace_provenance_unsupported_supported_claim_rate_increase,
            config.max_runtime_drift_product_trace_provenance_error_rate_increase,
            config.min_runtime_drift_product_trace_provenance_final_answer_evidence_reference_rate,
            config.min_runtime_drift_product_trace_citation_integrity_participating_trace_rate,
            config.min_runtime_drift_product_trace_citation_integrity_coverage_rate,
            config.max_runtime_drift_product_trace_citation_integrity_mismatch_rate_increase,
            config.max_runtime_drift_product_trace_citation_integrity_unresolved_rate_increase,
            config.max_runtime_drift_product_trace_citation_integrity_issue_rate_increase,
            config.max_runtime_drift_product_trace_citation_integrity_trace_gap_rate_increase,
            config.min_runtime_drift_current_trace_count,
        )
    )


def _runtime_drift_gate_config(config: ProductTraceReplayWorkflowConfig) -> dict[str, Any]:
    return {
        "max_total_seconds_mean_ratio": config.max_runtime_drift_total_seconds_mean_ratio,
        "max_total_seconds_p95_ratio": config.max_runtime_drift_total_seconds_p95_ratio,
        "max_mean_route_duration_ratio": config.max_runtime_drift_mean_route_duration_ratio,
        "max_p95_route_duration_ratio": config.max_runtime_drift_p95_route_duration_ratio,
        "max_mean_attempted_route_count_delta": config.max_runtime_drift_mean_attempted_route_count_delta,
        "max_retrieval_use_rate_delta": config.max_runtime_drift_retrieval_use_rate_delta,
        "max_cache_hit_rate_drop": config.max_runtime_drift_cache_hit_rate_drop,
        "max_verification_skip_rate_drop": config.max_runtime_drift_verification_skip_rate_drop,
        "min_pre_generation_risk_coverage_rate": (
            config.min_runtime_drift_pre_generation_risk_coverage_rate
        ),
        "min_pre_generation_learned_risk_coverage_rate": (
            config.min_runtime_drift_pre_generation_learned_risk_coverage_rate
        ),
        "max_pre_generation_audit_profile_rate_increase": (
            config.max_runtime_drift_pre_generation_audit_profile_rate_increase
        ),
        "max_pre_generation_learned_risk_routed_rate_increase": (
            config.max_runtime_drift_pre_generation_learned_risk_routed_rate_increase
        ),
        "max_pre_generation_learned_risk_probability_mean_increase": (
            config.max_runtime_drift_pre_generation_learned_risk_probability_mean_increase
        ),
        "min_promotion_contract_coverage": config.min_runtime_drift_promotion_contract_coverage,
        "min_pre_generation_probe_comparison_coverage": (
            config.min_runtime_drift_pre_generation_probe_comparison_coverage
        ),
        "min_pre_generation_probe_comparison_manifest_verified_rate": (
            config.min_runtime_drift_pre_generation_probe_comparison_manifest_verified_rate
        ),
        "min_pre_generation_probe_comparison_model_count": (
            config.min_runtime_drift_pre_generation_probe_comparison_model_count
        ),
        "min_pre_generation_probe_comparison_run_count": (
            config.min_runtime_drift_pre_generation_probe_comparison_run_count
        ),
        "min_pre_generation_probe_comparison_redline_pass_rate": (
            config.min_runtime_drift_pre_generation_probe_comparison_redline_pass_rate
        ),
        "max_pre_generation_probe_comparison_best_test_label_auroc_drop": (
            config.max_runtime_drift_pre_generation_probe_comparison_best_test_label_auroc_drop
        ),
        "max_pre_generation_probe_comparison_best_redline_auroc_drop": (
            config.max_runtime_drift_pre_generation_probe_comparison_best_redline_auroc_drop
        ),
        "max_pre_generation_probe_comparison_best_redline_margin_drop": (
            config.max_runtime_drift_pre_generation_probe_comparison_best_redline_margin_drop
        ),
        "min_claim_factuality_probe_comparison_coverage": (
            config.min_runtime_drift_claim_factuality_probe_comparison_coverage
        ),
        "min_claim_factuality_probe_comparison_manifest_verified_rate": (
            config.min_runtime_drift_claim_factuality_probe_comparison_manifest_verified_rate
        ),
        "min_claim_factuality_probe_comparison_model_count": (
            config.min_runtime_drift_claim_factuality_probe_comparison_model_count
        ),
        "min_claim_factuality_probe_comparison_run_count": (
            config.min_runtime_drift_claim_factuality_probe_comparison_run_count
        ),
        "min_claim_factuality_probe_comparison_dataset_count": (
            config.min_runtime_drift_claim_factuality_probe_comparison_dataset_count
        ),
        "min_claim_factuality_probe_comparison_redline_pass_rate": (
            config.min_runtime_drift_claim_factuality_probe_comparison_redline_pass_rate
        ),
        "max_claim_factuality_probe_comparison_best_test_label_auroc_drop": (
            config.max_runtime_drift_claim_factuality_probe_comparison_best_test_label_auroc_drop
        ),
        "max_claim_factuality_probe_comparison_best_test_selective_accuracy_drop": (
            config.max_runtime_drift_claim_factuality_probe_comparison_best_test_selective_accuracy_drop
        ),
        "max_claim_factuality_probe_comparison_best_test_selective_coverage_drop": (
            config.max_runtime_drift_claim_factuality_probe_comparison_best_test_selective_coverage_drop
        ),
        "max_claim_factuality_probe_comparison_best_redline_auroc_drop": (
            config.max_runtime_drift_claim_factuality_probe_comparison_best_redline_auroc_drop
        ),
        "max_claim_factuality_probe_comparison_best_redline_margin_drop": (
            config.max_runtime_drift_claim_factuality_probe_comparison_best_redline_margin_drop
        ),
        "min_counterfactual_verification_coverage": (
            config.min_runtime_drift_counterfactual_verification_coverage
        ),
        "min_counterfactual_verification_manifest_verified_rate": (
            config.min_runtime_drift_counterfactual_verification_manifest_verified_rate
        ),
        "min_counterfactual_verification_record_count": (
            config.min_runtime_drift_counterfactual_verification_record_count
        ),
        "min_counterfactual_verification_pass_rate": (
            config.min_runtime_drift_counterfactual_verification_pass_rate
        ),
        "max_counterfactual_verification_false_invariance_rate": (
            config.max_runtime_drift_counterfactual_verification_false_invariance_rate
        ),
        "max_counterfactual_verification_flip_success_count_drop": (
            config.max_runtime_drift_counterfactual_verification_flip_success_count_drop
        ),
        "min_evidence_handoff_coverage": config.min_runtime_drift_evidence_handoff_coverage,
        "min_evidence_handoff_manifest_verified_rate": (
            config.min_runtime_drift_evidence_handoff_manifest_verified_rate
        ),
        "min_evidence_handoff_present_metric_rate": (
            config.min_runtime_drift_evidence_handoff_present_metric_rate
        ),
        "max_evidence_handoff_missing_metric_rate": (
            config.max_runtime_drift_evidence_handoff_missing_metric_rate
        ),
        "max_evidence_handoff_missing_metric_count": (
            config.max_runtime_drift_evidence_handoff_missing_metric_count
        ),
        "max_evidence_handoff_blocked_group_count": (
            config.max_runtime_drift_evidence_handoff_blocked_group_count
        ),
        "min_evidence_handoff_promoted_group_rate": (
            config.min_runtime_drift_evidence_handoff_promoted_group_rate
        ),
        "min_fact_selfcheck_gate_coverage": config.min_runtime_drift_fact_selfcheck_gate_coverage,
        "min_fact_selfcheck_gate_report_present_rate": (
            config.min_runtime_drift_fact_selfcheck_gate_report_present_rate
        ),
        "min_fact_selfcheck_gate_manifest_present_rate": (
            config.min_runtime_drift_fact_selfcheck_gate_manifest_present_rate
        ),
        "min_fact_selfcheck_gate_manifest_verified_rate": (
            config.min_runtime_drift_fact_selfcheck_gate_manifest_verified_rate
        ),
        "min_fact_selfcheck_gate_passed_rate": (
            config.min_runtime_drift_fact_selfcheck_gate_passed_rate
        ),
        "min_fact_selfcheck_gate_run_count": config.min_runtime_drift_fact_selfcheck_gate_run_count,
        "max_fact_selfcheck_gate_failed_run_count": (
            config.max_runtime_drift_fact_selfcheck_gate_failed_run_count
        ),
        "min_fact_selfcheck_gate_min_executed_rate": (
            config.min_runtime_drift_fact_selfcheck_gate_min_executed_rate
        ),
        "min_fact_selfcheck_gate_min_decided_rate": (
            config.min_runtime_drift_fact_selfcheck_gate_min_decided_rate
        ),
        "max_fact_selfcheck_gate_max_not_applicable_rate": (
            config.max_runtime_drift_fact_selfcheck_gate_max_not_applicable_rate
        ),
        "min_fact_selfcheck_gate_min_claim_triples_per_record": (
            config.min_runtime_drift_fact_selfcheck_gate_min_claim_triples_per_record
        ),
        "min_fact_selfcheck_gate_min_sample_triples_per_record": (
            config.min_runtime_drift_fact_selfcheck_gate_min_sample_triples_per_record
        ),
        "min_triple_extraction_fixture_matrix_coverage": (
            config.min_runtime_drift_triple_extraction_fixture_matrix_coverage
        ),
        "max_triple_extraction_fixture_matrix_mean_best_f1_drop": (
            config.max_runtime_drift_triple_extraction_fixture_matrix_mean_best_f1_drop
        ),
        "max_triple_extraction_fixture_matrix_mean_f1_lift_drop": (
            config.max_runtime_drift_triple_extraction_fixture_matrix_mean_f1_lift_drop
        ),
        "min_triple_claim_coverage": config.min_runtime_drift_triple_claim_coverage,
        "min_triple_audit_claim_coverage": config.min_runtime_drift_triple_audit_claim_coverage,
        "min_triple_audit_pass_rate": config.min_runtime_drift_triple_audit_pass_rate,
        "min_triple_slot_coverage": config.min_runtime_drift_triple_slot_coverage,
        "min_world_model_participating_trace_rate": (
            config.min_runtime_drift_world_model_participating_trace_rate
        ),
        "min_world_model_coverage_rate": config.min_runtime_drift_world_model_coverage_rate,
        "max_world_model_conflict_rate_increase": (
            config.max_runtime_drift_world_model_conflict_rate_increase
        ),
        "max_world_model_low_agreement_rate_increase": (
            config.max_runtime_drift_world_model_low_agreement_rate_increase
        ),
        "max_world_model_trace_gap_rate_increase": (
            config.max_runtime_drift_world_model_trace_gap_rate_increase
        ),
        "min_context_sensitivity_participating_trace_rate": (
            config.min_runtime_drift_context_sensitivity_participating_trace_rate
        ),
        "min_context_sensitivity_coverage_rate": (
            config.min_runtime_drift_context_sensitivity_coverage_rate
        ),
        "max_context_sensitivity_flagged_result_rate_increase": (
            config.max_runtime_drift_context_sensitivity_flagged_result_rate_increase
        ),
        "max_context_sensitivity_trace_gap_rate_increase": (
            config.max_runtime_drift_context_sensitivity_trace_gap_rate_increase
        ),
        "max_context_sensitivity_max_flagged_rate_increase": (
            config.max_runtime_drift_context_sensitivity_max_flagged_rate_increase
        ),
        "max_context_sensitivity_max_ratio_increase": (
            config.max_runtime_drift_context_sensitivity_max_ratio_increase
        ),
        "min_counterfactual_robustness_participating_trace_rate": (
            config.min_runtime_drift_counterfactual_robustness_participating_trace_rate
        ),
        "min_counterfactual_robustness_coverage_rate": (
            config.min_runtime_drift_counterfactual_robustness_coverage_rate
        ),
        "min_counterfactual_robustness_pass_rate": (
            config.min_runtime_drift_counterfactual_robustness_pass_rate
        ),
        "min_counterfactual_robustness_flip_success_rate": (
            config.min_runtime_drift_counterfactual_robustness_flip_success_rate
        ),
        "max_counterfactual_robustness_false_invariance_rate_increase": (
            config.max_runtime_drift_counterfactual_robustness_false_invariance_rate_increase
        ),
        "max_counterfactual_robustness_trace_gap_rate_increase": (
            config.max_runtime_drift_counterfactual_robustness_trace_gap_rate_increase
        ),
        "min_claim_risk_localization_coverage_rate": (
            config.min_runtime_drift_claim_risk_localization_coverage_rate
        ),
        "max_claim_risk_localization_high_risk_claim_count_increase": (
            config.max_runtime_drift_claim_risk_localization_high_risk_claim_count_increase
        ),
        "max_claim_risk_localization_medium_or_high_risk_claim_count_increase": (
            config.max_runtime_drift_claim_risk_localization_medium_or_high_risk_claim_count_increase
        ),
        "max_claim_risk_localization_entity_candidate_observation_count_increase": (
            config.max_runtime_drift_claim_risk_localization_entity_candidate_observation_count_increase
        ),
        "max_claim_risk_localization_unique_entity_candidate_count_increase": (
            config.max_runtime_drift_claim_risk_localization_unique_entity_candidate_count_increase
        ),
        "max_claim_risk_localization_high_risk_entity_candidate_count_increase": (
            config.max_runtime_drift_claim_risk_localization_high_risk_entity_candidate_count_increase
        ),
        "max_claim_risk_localization_medium_or_high_entity_candidate_count_increase": (
            config.max_runtime_drift_claim_risk_localization_medium_or_high_entity_candidate_count_increase
        ),
        "promotion_contract_covered_fact_property_scopes": tuple(
            config.runtime_drift_covered_fact_property_scopes
        ),
        "min_promotion_contract_covered_fact_property_metric_count": (
            config.min_runtime_drift_covered_fact_property_metric_count
        ),
        "min_promotion_contract_covered_fact_min_records": (
            config.min_runtime_drift_covered_fact_min_records
        ),
        "min_promotion_contract_covered_fact_min_source_documents": (
            config.min_runtime_drift_covered_fact_min_source_documents
        ),
        "max_promotion_contract_covered_fact_min_decision_accuracy_drop": (
            config.max_runtime_drift_covered_fact_min_decision_accuracy_drop
        ),
        "max_promotion_contract_covered_fact_max_false_supported_rate_increase": (
            config.max_runtime_drift_covered_fact_max_false_supported_rate_increase
        ),
        "max_promotion_contract_covered_fact_min_false_refuted_rate_drop": (
            config.max_runtime_drift_covered_fact_min_false_refuted_rate_drop
        ),
        "max_product_trace_action_audit_error_rate_increase": (
            config.max_runtime_drift_product_trace_action_audit_error_rate_increase
        ),
        "max_product_trace_action_audit_missing_retrieval_action_rate_increase": (
            config.max_runtime_drift_product_trace_action_audit_missing_retrieval_action_rate_increase
        ),
        "max_product_trace_action_audit_missing_plan_retrieval_query_rate_increase": (
            config.max_runtime_drift_product_trace_action_audit_missing_plan_retrieval_query_rate_increase
        ),
        "max_product_trace_action_audit_malformed_payload_rate_increase": (
            config.max_runtime_drift_product_trace_action_audit_malformed_payload_rate_increase
        ),
        "max_product_trace_action_audit_unexpected_action_rate_increase": (
            config.max_runtime_drift_product_trace_action_audit_unexpected_action_rate_increase
        ),
        "max_product_trace_action_audit_unknown_claim_id_rate_increase": (
            config.max_runtime_drift_product_trace_action_audit_unknown_claim_id_rate_increase
        ),
        "max_product_trace_action_execution_alignment_failed_trace_rate_increase": (
            config.max_runtime_drift_product_trace_action_execution_alignment_failed_trace_rate_increase
        ),
        "max_product_trace_action_execution_missing_result_rate_increase": (
            config.max_runtime_drift_product_trace_action_execution_missing_result_rate_increase
        ),
        "max_product_trace_action_execution_unexpected_result_rate_increase": (
            config.max_runtime_drift_product_trace_action_execution_unexpected_result_rate_increase
        ),
        "max_product_trace_action_execution_request_id_mismatch_rate_increase": (
            config.max_runtime_drift_product_trace_action_execution_request_id_mismatch_rate_increase
        ),
        "min_product_trace_receipt_claim_support_reference_support_rate": (
            config.min_runtime_drift_product_trace_receipt_claim_support_reference_support_rate
        ),
        "max_product_trace_receipt_claim_support_unsupported_reference_rate_increase": (
            config.max_runtime_drift_product_trace_receipt_claim_support_unsupported_reference_rate_increase
        ),
        "max_product_trace_receipt_claim_support_missing_reference_rate_increase": (
            config.max_runtime_drift_product_trace_receipt_claim_support_missing_reference_rate_increase
        ),
        "max_product_trace_receipt_claim_support_unreceipted_reference_rate_increase": (
            config.max_runtime_drift_product_trace_receipt_claim_support_unreceipted_reference_rate_increase
        ),
        "max_product_trace_receipt_claim_support_failed_result_reference_rate_increase": (
            config.max_runtime_drift_product_trace_receipt_claim_support_failed_result_reference_rate_increase
        ),
        "max_product_trace_receipt_claim_support_fingerprint_mismatch_reference_rate_increase": (
            config.max_runtime_drift_product_trace_receipt_claim_support_fingerprint_mismatch_reference_rate_increase
        ),
        "max_product_trace_receipt_claim_support_unsigned_reference_rate_increase": (
            config.max_runtime_drift_product_trace_receipt_claim_support_unsigned_reference_rate_increase
        ),
        "max_product_trace_trajectory_audit_failed_trace_rate_increase": (
            config.max_runtime_drift_product_trace_trajectory_audit_failed_trace_rate_increase
        ),
        "max_product_trace_trajectory_audit_error_rate_increase": (
            config.max_runtime_drift_product_trace_trajectory_audit_error_rate_increase
        ),
        "max_product_trace_trajectory_audit_factual_rate_increase": (
            config.max_runtime_drift_product_trace_trajectory_audit_factual_rate_increase
        ),
        "max_product_trace_trajectory_audit_referential_rate_increase": (
            config.max_runtime_drift_product_trace_trajectory_audit_referential_rate_increase
        ),
        "max_product_trace_trajectory_audit_logical_rate_increase": (
            config.max_runtime_drift_product_trace_trajectory_audit_logical_rate_increase
        ),
        "max_product_trace_trajectory_audit_procedural_rate_increase": (
            config.max_runtime_drift_product_trace_trajectory_audit_procedural_rate_increase
        ),
        "max_product_trace_trajectory_audit_scope_rate_increase": (
            config.max_runtime_drift_product_trace_trajectory_audit_scope_rate_increase
        ),
        "max_product_trace_trajectory_audit_cascade_rate_increase": (
            config.max_runtime_drift_product_trace_trajectory_audit_cascade_rate_increase
        ),
        "min_product_trace_provenance_coverage_rate": (
            config.min_runtime_drift_product_trace_provenance_coverage_rate
        ),
        "min_product_trace_provenance_supported_claim_evidence_coverage": (
            config.min_runtime_drift_product_trace_provenance_supported_claim_evidence_coverage
        ),
        "max_product_trace_provenance_missing_reference_rate_increase": (
            config.max_runtime_drift_product_trace_provenance_missing_reference_rate_increase
        ),
        "max_product_trace_provenance_unsupported_supported_claim_rate_increase": (
            config.max_runtime_drift_product_trace_provenance_unsupported_supported_claim_rate_increase
        ),
        "max_product_trace_provenance_error_rate_increase": (
            config.max_runtime_drift_product_trace_provenance_error_rate_increase
        ),
        "min_product_trace_provenance_final_answer_evidence_reference_rate": (
            config.min_runtime_drift_product_trace_provenance_final_answer_evidence_reference_rate
        ),
        "min_product_trace_citation_integrity_participating_trace_rate": (
            config.min_runtime_drift_product_trace_citation_integrity_participating_trace_rate
        ),
        "min_product_trace_citation_integrity_coverage_rate": (
            config.min_runtime_drift_product_trace_citation_integrity_coverage_rate
        ),
        "max_product_trace_citation_integrity_mismatch_rate_increase": (
            config.max_runtime_drift_product_trace_citation_integrity_mismatch_rate_increase
        ),
        "max_product_trace_citation_integrity_unresolved_rate_increase": (
            config.max_runtime_drift_product_trace_citation_integrity_unresolved_rate_increase
        ),
        "max_product_trace_citation_integrity_issue_rate_increase": (
            config.max_runtime_drift_product_trace_citation_integrity_issue_rate_increase
        ),
        "max_product_trace_citation_integrity_trace_gap_rate_increase": (
            config.max_runtime_drift_product_trace_citation_integrity_trace_gap_rate_increase
        ),
        "min_current_trace_count": config.min_runtime_drift_current_trace_count,
    }


def _action_audit_gate_configured(config: ProductTraceReplayWorkflowConfig) -> bool:
    return any(
        value is not None
        for value in (
            config.max_action_audit_error_rate,
            config.max_action_audit_missing_retrieval_rate,
            config.max_action_audit_missing_plan_retrieval_query_rate,
            config.max_action_audit_malformed_payload_rate,
            config.max_action_audit_unexpected_action_rate,
            config.max_action_audit_unknown_claim_id_rate,
        )
    )


def _action_audit_gate_config(config: ProductTraceReplayWorkflowConfig) -> dict[str, Any]:
    return {
        "max_error_rate": config.max_action_audit_error_rate,
        "max_missing_retrieval_rate": config.max_action_audit_missing_retrieval_rate,
        "max_missing_plan_retrieval_query_rate": (
            config.max_action_audit_missing_plan_retrieval_query_rate
        ),
        "max_malformed_payload_rate": config.max_action_audit_malformed_payload_rate,
        "max_unexpected_action_rate": config.max_action_audit_unexpected_action_rate,
        "max_unknown_claim_id_rate": config.max_action_audit_unknown_claim_id_rate,
    }


def _action_audit_gate_checks(
    action_audit: Mapping[str, Any],
    gate_config: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    check_specs = (
        ("action_audit.error_rate", "error_rate", "max_error_rate"),
        (
            "action_audit.missing_retrieval_action_rate",
            "missing_retrieval_action_rate",
            "max_missing_retrieval_rate",
        ),
        (
            "action_audit.missing_plan_retrieval_query_rate",
            "missing_plan_retrieval_query_rate",
            "max_missing_plan_retrieval_query_rate",
        ),
        (
            "action_audit.malformed_payload_rate",
            "malformed_payload_rate",
            "max_malformed_payload_rate",
        ),
        (
            "action_audit.unexpected_action_rate",
            "unexpected_action_rate",
            "max_unexpected_action_rate",
        ),
        (
            "action_audit.unknown_claim_id_rate",
            "unknown_claim_id_rate",
            "max_unknown_claim_id_rate",
        ),
    )
    checks = []
    for metric_name, summary_key, limit_key in check_specs:
        limit = gate_config.get(limit_key)
        if limit is None:
            continue
        value = _finite_float(action_audit.get(summary_key))
        if value is None:
            checks.append({
                "metric": metric_name,
                "status": "blocked",
                "value": None,
                "limit": limit,
                "operator": "<=",
                "reason": f"{metric_name} is unavailable",
            })
            continue
        status = "pass" if value <= float(limit) else "blocked"
        checks.append({
            "metric": metric_name,
            "status": status,
            "value": value,
            "limit": float(limit),
            "operator": "<=",
            "reason": (
                None
                if status == "pass"
                else f"{metric_name} {value:.6g} exceeded limit {float(limit):.6g}"
            ),
        })
    return tuple(checks)


def _action_execution_gate_configured(config: ProductTraceReplayWorkflowConfig) -> bool:
    return any(
        value is not None
        for value in (
            config.max_action_execution_missing_result_rate,
            config.max_action_execution_unexpected_result_rate,
            config.max_action_execution_request_id_mismatch_rate,
        )
    )


def _action_execution_gate_config(config: ProductTraceReplayWorkflowConfig) -> dict[str, Any]:
    return {
        "max_missing_result_rate": config.max_action_execution_missing_result_rate,
        "max_unexpected_result_rate": config.max_action_execution_unexpected_result_rate,
        "max_request_id_mismatch_rate": config.max_action_execution_request_id_mismatch_rate,
    }


def _action_execution_gate_checks(
    action_execution: Mapping[str, Any],
    gate_config: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    check_specs = (
        (
            "action_execution.missing_result_rate",
            "missing_result_rate",
            "max_missing_result_rate",
        ),
        (
            "action_execution.unexpected_result_rate",
            "unexpected_result_rate",
            "max_unexpected_result_rate",
        ),
        (
            "action_execution.request_id_mismatch_rate",
            "request_id_mismatch_rate",
            "max_request_id_mismatch_rate",
        ),
    )
    checks = []
    for metric_name, summary_key, limit_key in check_specs:
        limit = gate_config.get(limit_key)
        if limit is None:
            continue
        value = _finite_float(action_execution.get(summary_key))
        if value is None:
            checks.append({
                "metric": metric_name,
                "status": "blocked",
                "value": None,
                "limit": limit,
                "operator": "<=",
                "reason": f"{metric_name} is unavailable",
            })
            continue
        status = "pass" if value <= float(limit) else "blocked"
        checks.append({
            "metric": metric_name,
            "status": status,
            "value": value,
            "limit": float(limit),
            "operator": "<=",
            "reason": (
                None
                if status == "pass"
                else f"{metric_name} {value:.6g} exceeded limit {float(limit):.6g}"
            ),
        })
    return tuple(checks)


def _corpus_summary(corpus: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(corpus.get("summary"))
    workflow_cache = _mapping(corpus.get("workflow_cache"))
    source_cache = _mapping(corpus.get("source_cache"))
    return {
        "status": corpus.get("status"),
        "accepted_count": summary.get("accepted_count"),
        "rejected_count": summary.get("rejected_count"),
        "runtime_trace_count": summary.get("runtime_trace_count"),
        "redacted_trace_count": summary.get("redacted_trace_count"),
        "unique_request_key_count": summary.get("unique_request_key_count"),
        "runtime_pair_index_record_count": _nested(corpus, "runtime_pair_index", "record_count"),
        "counts_by_runtime_profile": dict(_mapping(summary.get("counts_by_runtime_profile"))),
        "counts_by_risk_level": dict(_mapping(summary.get("counts_by_risk_level"))),
        "counts_by_action": dict(_mapping(summary.get("counts_by_action"))),
        "cache_source": workflow_cache.get("source"),
        "cache_hit": workflow_cache.get("cache_hit"),
        "cache_written": workflow_cache.get("cache_written"),
        "cache_path": workflow_cache.get("path"),
        "source_cache_source": source_cache.get("source"),
        "source_cache_hit": source_cache.get("cache_hit"),
        "source_cache_partial_hit": source_cache.get("cache_partial_hit"),
        "source_cache_hit_count": source_cache.get("hit_count"),
        "source_cache_miss_count": source_cache.get("miss_count"),
        "source_cache_written": source_cache.get("cache_written"),
        "source_cache_path": _nested(corpus, "paths", "source_cache"),
    }


def _runtime_baseline_summary(runtime_baseline: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(runtime_baseline.get("summary"))
    total_seconds = _mapping(summary.get("total_seconds"))
    action_audit = _mapping(summary.get("action_audit"))
    action_execution = _mapping(summary.get("action_execution"))
    trajectory_audit = _mapping(summary.get("trajectory_audit"))
    trace_record_cache = _mapping(_nested(runtime_baseline, "config", "trace_record_cache"))
    recommended_policy = _mapping(_nested(runtime_baseline, "config", "recommended_policy"))
    optimization = _mapping(runtime_baseline.get("optimization"))
    recommendations = tuple(_mapping(item) for item in _sequence(optimization.get("recommendations")))
    return {
        "status": runtime_baseline.get("status"),
        "budget_enabled": _nested(runtime_baseline, "budget", "enabled"),
        "budget_passed": _nested(runtime_baseline, "budget", "passed"),
        "n_traces": summary.get("n_traces"),
        "runtime_trace_count": summary.get("runtime_trace_count"),
        "total_seconds_mean": total_seconds.get("mean"),
        "total_seconds_p95": total_seconds.get("p95"),
        "total_seconds_max": total_seconds.get("max"),
        "trace_records_cache_source": trace_record_cache.get("source"),
        "trace_records_cache_hit": trace_record_cache.get("cache_hit"),
        "trace_records_cache_written": trace_record_cache.get("cache_written"),
        "trace_scan_workers": _nested(
            runtime_baseline,
            "config",
            "trace_scan_workers",
        ),
        "trace_scan_effective_workers": trace_record_cache.get("trace_scan_workers"),
        "action_audit_available_trace_count": action_audit.get("available_trace_count"),
        "action_audit_failed_trace_count": action_audit.get("failed_trace_count"),
        "action_audit_failed_trace_rate": action_audit.get("failed_trace_rate"),
        "action_audit_error_count": action_audit.get("error_count"),
        "action_audit_error_rate": action_audit.get("error_rate"),
        "action_audit_missing_retrieval_action_rate": action_audit.get(
            "missing_retrieval_action_rate"
        ),
        "action_audit_missing_plan_retrieval_query_rate": action_audit.get(
            "missing_plan_retrieval_query_rate"
        ),
        "action_audit_malformed_payload_rate": action_audit.get("malformed_payload_rate"),
        "action_audit_unexpected_action_rate": action_audit.get("unexpected_action_rate"),
        "action_audit_unknown_claim_id_rate": action_audit.get("unknown_claim_id_rate"),
        "action_execution_available_trace_count": action_execution.get("available_trace_count"),
        "action_execution_alignment_available_trace_count": action_execution.get(
            "alignment_available_trace_count"
        ),
        "action_execution_alignment_failed_trace_count": action_execution.get(
            "alignment_failed_trace_count"
        ),
        "action_execution_alignment_failed_trace_rate": action_execution.get(
            "alignment_failed_trace_rate"
        ),
        "action_execution_missing_result_rate": action_execution.get("missing_result_rate"),
        "action_execution_unexpected_result_rate": action_execution.get("unexpected_result_rate"),
        "action_execution_request_id_mismatch_rate": action_execution.get(
            "request_id_mismatch_rate"
        ),
        "trajectory_audit_available_trace_count": trajectory_audit.get("available_trace_count"),
        "trajectory_audit_failed_trace_rate": trajectory_audit.get("failed_trace_rate"),
        "trajectory_audit_error_rate": trajectory_audit.get("error_rate"),
        "trajectory_audit_factual_rate": trajectory_audit.get("factual_rate"),
        "trajectory_audit_referential_rate": trajectory_audit.get("referential_rate"),
        "trajectory_audit_logical_rate": trajectory_audit.get("logical_rate"),
        "trajectory_audit_procedural_rate": trajectory_audit.get("procedural_rate"),
        "trajectory_audit_scope_rate": trajectory_audit.get("scope_rate"),
        "trace_records_cache_path": _nested(runtime_baseline, "paths", "trace_records_cache"),
        "recommended_policy_path": _nested(runtime_baseline, "paths", "recommended_policy"),
        "recommended_policy_written": recommended_policy.get("written"),
        "recommended_policy_enabled": recommended_policy.get("policy_enabled"),
        "recommended_policy_threshold_count": recommended_policy.get("threshold_count"),
        "optimization_status": optimization.get("status"),
        "optimization_recommendation_count": len(recommendations),
    }


def _action_audit_gate_summary(action_audit_gate: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(action_audit_gate.get("summary"))
    return {
        "status": action_audit_gate.get("status"),
        "gate_enabled": summary.get("gate_enabled"),
        "passed": summary.get("passed"),
        "trace_count": summary.get("trace_count"),
        "available_trace_count": summary.get("available_trace_count"),
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
        "report_path": _nested(action_audit_gate, "paths", "report"),
    }


def _action_execution_gate_summary(action_execution_gate: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(action_execution_gate.get("summary"))
    return {
        "status": action_execution_gate.get("status"),
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
        "report_path": _nested(action_execution_gate, "paths", "report"),
    }


def _runtime_drift_summary(runtime_drift: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(runtime_drift.get("summary"))
    budget_gate = _mapping(runtime_drift.get("runtime_budget_policy_gate"))
    covered_fact_property = _covered_fact_property_metric_summary(runtime_drift)
    product_trace_action_gate = _product_trace_action_gate_metric_summary(runtime_drift)
    product_trace_receipt_claim_support = (
        _product_trace_receipt_claim_support_metric_summary(runtime_drift)
    )
    product_trace_trajectory_audit = _product_trace_trajectory_audit_metric_summary(runtime_drift)
    product_trace_provenance = _product_trace_provenance_metric_summary(runtime_drift)
    product_trace_citation_integrity = (
        _product_trace_citation_integrity_metric_summary(runtime_drift)
    )
    world_model = _world_model_metric_summary(runtime_drift)
    pre_generation_probe_comparison = _pre_generation_probe_comparison_metric_summary(runtime_drift)
    claim_factuality_probe_comparison = _claim_factuality_probe_comparison_metric_summary(runtime_drift)
    counterfactual_verification = _counterfactual_verification_metric_summary(runtime_drift)
    evidence_handoff = _evidence_handoff_metric_summary(runtime_drift)
    fact_selfcheck_gate = _fact_selfcheck_gate_metric_summary(runtime_drift)
    context_sensitivity = _context_sensitivity_metric_summary(runtime_drift)
    counterfactual_robustness = _counterfactual_robustness_metric_summary(runtime_drift)
    claim_risk_localization = _claim_risk_localization_metric_summary(runtime_drift)
    pre_generation_risk = _pre_generation_risk_metric_summary(runtime_drift)
    return {
        "status": runtime_drift.get("status"),
        "gate_enabled": summary.get("gate_enabled"),
        "drift_gate_enabled": summary.get("drift_gate_enabled"),
        "runtime_budget_policy_gate_enabled": summary.get("runtime_budget_policy_gate_enabled"),
        "runtime_budget_policy_passed": summary.get("runtime_budget_policy_passed"),
        "runtime_budget_policy_failed_count": summary.get("runtime_budget_policy_failed_count"),
        "compared_metric_count": summary.get("compared_metric_count"),
        "blocked_metric_count": summary.get("blocked_metric_count"),
        "observed_metric_count": summary.get("observed_metric_count"),
        "covered_fact_property_metric_count": covered_fact_property["metric_count"],
        "covered_fact_property_blocked_metric_count": covered_fact_property["blocked_metric_count"],
        "product_trace_action_gate_metric_count": product_trace_action_gate["metric_count"],
        "product_trace_action_gate_blocked_metric_count": product_trace_action_gate["blocked_metric_count"],
        "product_trace_receipt_claim_support_metric_count": product_trace_receipt_claim_support[
            "metric_count"
        ],
        "product_trace_receipt_claim_support_blocked_metric_count": (
            product_trace_receipt_claim_support["blocked_metric_count"]
        ),
        "product_trace_trajectory_audit_metric_count": product_trace_trajectory_audit[
            "metric_count"
        ],
        "product_trace_trajectory_audit_blocked_metric_count": (
            product_trace_trajectory_audit["blocked_metric_count"]
        ),
        "product_trace_provenance_metric_count": product_trace_provenance["metric_count"],
        "product_trace_provenance_blocked_metric_count": (
            product_trace_provenance["blocked_metric_count"]
        ),
        "product_trace_citation_integrity_metric_count": (
            product_trace_citation_integrity["metric_count"]
        ),
        "product_trace_citation_integrity_blocked_metric_count": (
            product_trace_citation_integrity["blocked_metric_count"]
        ),
        "world_model_metric_count": world_model["metric_count"],
        "world_model_blocked_metric_count": world_model["blocked_metric_count"],
        "context_sensitivity_metric_count": context_sensitivity["metric_count"],
        "context_sensitivity_blocked_metric_count": context_sensitivity[
            "blocked_metric_count"
        ],
        "counterfactual_robustness_metric_count": counterfactual_robustness[
            "metric_count"
        ],
        "counterfactual_robustness_blocked_metric_count": (
            counterfactual_robustness["blocked_metric_count"]
        ),
        "claim_risk_localization_metric_count": claim_risk_localization["metric_count"],
        "claim_risk_localization_blocked_metric_count": (
            claim_risk_localization["blocked_metric_count"]
        ),
        "pre_generation_risk_metric_count": pre_generation_risk["metric_count"],
        "pre_generation_risk_blocked_metric_count": (
            pre_generation_risk["blocked_metric_count"]
        ),
        "pre_generation_probe_comparison_metric_count": pre_generation_probe_comparison["metric_count"],
        "pre_generation_probe_comparison_blocked_metric_count": (
            pre_generation_probe_comparison["blocked_metric_count"]
        ),
        "claim_factuality_probe_comparison_metric_count": claim_factuality_probe_comparison["metric_count"],
        "claim_factuality_probe_comparison_blocked_metric_count": (
            claim_factuality_probe_comparison["blocked_metric_count"]
        ),
        "counterfactual_verification_metric_count": counterfactual_verification["metric_count"],
        "counterfactual_verification_blocked_metric_count": (
            counterfactual_verification["blocked_metric_count"]
        ),
        "evidence_handoff_metric_count": evidence_handoff["metric_count"],
        "evidence_handoff_blocked_metric_count": evidence_handoff["blocked_metric_count"],
        "fact_selfcheck_gate_metric_count": fact_selfcheck_gate["metric_count"],
        "fact_selfcheck_gate_blocked_metric_count": fact_selfcheck_gate["blocked_metric_count"],
        "baseline_path": _nested(runtime_drift, "baseline", "path"),
        "current_path": _nested(runtime_drift, "current", "path"),
        "report_path": _nested(runtime_drift, "paths", "report"),
        "artifact_manifest_path": _nested(runtime_drift, "paths", "artifact_manifest"),
        "runtime_budget_policy_path": _nested(runtime_drift, "paths", "runtime_budget_policy"),
        "runtime_budget_policy_check_count": budget_gate.get("check_count"),
    }


def _covered_fact_property_metric_summary(runtime_drift: Mapping[str, Any]) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith(
            "promotion_contract.covered_fact_properties."
        )
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _pre_generation_probe_comparison_metric_summary(runtime_drift: Mapping[str, Any]) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith(
            "promotion_contract.pre_generation_probe_comparison."
        )
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _pre_generation_risk_metric_summary(runtime_drift: Mapping[str, Any]) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith("pre_generation_risk.")
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _claim_factuality_probe_comparison_metric_summary(runtime_drift: Mapping[str, Any]) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith(
            "promotion_contract.claim_factuality_probe_comparison."
        )
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _counterfactual_verification_metric_summary(runtime_drift: Mapping[str, Any]) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith(
            "promotion_contract.counterfactual_verification."
        )
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _evidence_handoff_metric_summary(runtime_drift: Mapping[str, Any]) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith(
            "promotion_contract.evidence_handoff."
        )
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _fact_selfcheck_gate_metric_summary(runtime_drift: Mapping[str, Any]) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith(
            "promotion_contract.fact_selfcheck_gate."
        )
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _world_model_metric_summary(runtime_drift: Mapping[str, Any]) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith("world_model.")
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _context_sensitivity_metric_summary(runtime_drift: Mapping[str, Any]) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith("context_sensitivity.")
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _counterfactual_robustness_metric_summary(runtime_drift: Mapping[str, Any]) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith(
            "counterfactual_robustness."
        )
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _claim_risk_localization_metric_summary(runtime_drift: Mapping[str, Any]) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith(
            "claim_risk_localization."
        )
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _product_trace_action_gate_metric_summary(runtime_drift: Mapping[str, Any]) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith("promotion_contract.product_trace_replay.")
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _product_trace_receipt_claim_support_metric_summary(
    runtime_drift: Mapping[str, Any],
) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith("receipt_claim_support.")
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _product_trace_trajectory_audit_metric_summary(runtime_drift: Mapping[str, Any]) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith("trajectory_audit.")
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _product_trace_provenance_metric_summary(runtime_drift: Mapping[str, Any]) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith("provenance.")
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _product_trace_citation_integrity_metric_summary(
    runtime_drift: Mapping[str, Any],
) -> dict[str, int]:
    metrics = tuple(
        _mapping(metric)
        for metric in _sequence(runtime_drift.get("metrics"))
        if str(_mapping(metric).get("metric") or "").startswith("citation_integrity.")
    )
    return {
        "metric_count": len(metrics),
        "blocked_metric_count": sum(1 for metric in metrics if metric.get("status") == "blocked"),
    }


def _workflow_optimization_summary(runtime_baseline: Mapping[str, Any]) -> dict[str, Any]:
    optimization = _mapping(runtime_baseline.get("optimization"))
    if not optimization:
        return {
            "status": "missing",
            "recommendation_count": 0,
            "priority_counts": {},
            "area_counts": {},
            "top_recommendations": (),
            "phase_hotspots": (),
            "route_hotspots": (),
            "policy_hints": {},
        }
    recommendations = tuple(_mapping(item) for item in _sequence(optimization.get("recommendations")))
    phase_hotspots = tuple(
        _compact_hotspot(item, name_field="phase")
        for item in _sequence(_nested(optimization, "hotspots", "phases"))[:3]
        if isinstance(item, Mapping)
    )
    route_hotspots = tuple(
        _compact_hotspot(item, name_field="route")
        for item in _sequence(_nested(optimization, "hotspots", "routes"))[:3]
        if isinstance(item, Mapping)
    )
    return {
        "status": optimization.get("status"),
        "summary": dict(_mapping(optimization.get("summary"))),
        "recommendation_count": len(recommendations),
        "priority_counts": _counts(item.get("priority") for item in recommendations),
        "area_counts": _counts(item.get("area") for item in recommendations),
        "top_recommendations": tuple(
            _compact_recommendation(item)
            for item in recommendations[:5]
        ),
        "phase_hotspots": phase_hotspots,
        "route_hotspots": route_hotspots,
        "policy_hints": dict(_mapping(optimization.get("policy_hints"))),
    }


def _compact_recommendation(recommendation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": recommendation.get("id"),
        "priority": recommendation.get("priority"),
        "area": recommendation.get("area"),
        "title": recommendation.get("title"),
        "evidence": dict(_mapping(recommendation.get("evidence"))),
    }


def _compact_hotspot(hotspot: Mapping[str, Any], *, name_field: str) -> dict[str, Any]:
    keep_fields = (
        name_field,
        "total_observed_seconds",
        "total_duration_seconds",
        "mean_seconds",
        "mean_duration_seconds",
        "p95_seconds",
        "max_seconds",
        "max_duration_seconds",
        "retrieval_use_rate",
        "mean_attempted_route_count",
    )
    return {
        field_name: hotspot.get(field_name)
        for field_name in keep_fields
        if field_name in hotspot
    }


def _selector_replay_summary(selector_replay: Mapping[str, Any]) -> dict[str, Any]:
    recommended = _recommended_leaderboard_row(selector_replay)
    trace_inputs = _mapping(_nested(selector_replay, "config", "trace_inputs"))
    return {
        "status": selector_replay.get("status"),
        "recommended_candidate": _nested(selector_replay, "decision", "recommended_candidate"),
        "recommended_policy_path": _nested(selector_replay, "decision", "recommended_policy_path"),
        "recommended_estimated_cost_units_mean": recommended.get("estimated_cost_units_mean"),
        "recommended_observed_runtime_coverage_rate": recommended.get("observed_runtime_coverage_rate"),
        "recommended_observed_selected_total_seconds_mean": recommended.get(
            "observed_selected_total_seconds_mean"
        ),
        "recommended_observed_selected_total_seconds_p95": recommended.get(
            "observed_selected_total_seconds_p95"
        ),
        "trace_inputs_source": trace_inputs.get("source"),
        "trace_inputs_cache_hit": trace_inputs.get("cache_hit"),
        "trace_inputs_cache_written": trace_inputs.get("cache_written"),
        "trace_inputs_path": _nested(selector_replay, "paths", "trace_inputs"),
        "leaderboard": tuple(_sequence(selector_replay.get("leaderboard"))),
    }


def _recommended_leaderboard_row(report: Mapping[str, Any]) -> dict[str, Any]:
    recommended_candidate = _nested(report, "decision", "recommended_candidate")
    for row in _sequence(report.get("leaderboard")):
        if isinstance(row, Mapping) and row.get("candidate") == recommended_candidate:
            return dict(row)
    return {}


def _write_report_and_manifest(
    config: ProductTraceReplayWorkflowConfig,
    report: dict[str, Any],
    *,
    fingerprint_cache: MutableMapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    artifacts = _artifact_paths(config, report)
    report["artifact_manifest_summary"] = planned_artifact_manifest_summary(
        artifacts,
        assume_file_paths=(config.resolved_report_path,),
    )
    _write_json(config.resolved_report_path, report, compact=config.compact_json)
    return _write_artifact_manifest(
        config,
        report,
        artifacts=artifacts,
        fingerprint_cache=fingerprint_cache,
    )


def _artifact_paths(
    config: ProductTraceReplayWorkflowConfig,
    report: Mapping[str, Any],
) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {
        "product_trace_replay_workflow_report": config.resolved_report_path,
        "corpus_report": _nested(report, "paths", "corpus_report"),
        "corpus_manifest": _nested(report, "paths", "corpus_manifest"),
        "corpus_traces_dir": _nested(report, "paths", "corpus_traces_dir"),
        "corpus_runtime_pair_index": _nested(report, "paths", "corpus_runtime_pair_index"),
        "corpus_cache": _nested(report, "paths", "corpus_cache"),
        "corpus_source_cache": _nested(report, "paths", "corpus_source_cache"),
        "runtime_baseline_report": _nested(report, "paths", "runtime_baseline_report"),
        "runtime_baseline_manifest": _nested(report, "paths", "runtime_baseline_manifest"),
        "runtime_trace_records_cache": _nested(report, "paths", "runtime_trace_records_cache"),
        "runtime_recommended_policy": _nested(report, "paths", "runtime_recommended_policy"),
        "action_audit_gate_report": _nested(report, "paths", "action_audit_gate_report"),
        "action_execution_gate_report": _nested(report, "paths", "action_execution_gate_report"),
        "runtime_drift_report": _nested(report, "paths", "runtime_drift_report"),
        "runtime_drift_manifest": _nested(report, "paths", "runtime_drift_manifest"),
        "selector_replay_report": _nested(report, "paths", "selector_replay_report"),
        "selector_replay_manifest": _nested(report, "paths", "selector_replay_manifest"),
        "selector_trace_inputs": _nested(report, "paths", "selector_trace_inputs"),
        "runtime_policy": config.runtime_policy_path,
        "promotion_contract": config.promotion_contract_path,
        "replay_policy": config.replay_policy_path,
    }
    for index, path in enumerate(config.trace_paths):
        artifacts[f"input_trace_{index:04d}_{_safe_artifact_name(path.stem)}"] = path
    for index, path in enumerate(config.jsonl_paths):
        artifacts[f"input_jsonl_{index:04d}_{_safe_artifact_name(path.stem)}"] = path
    return artifacts


def _write_artifact_manifest(
    config: ProductTraceReplayWorkflowConfig,
    report: Mapping[str, Any],
    *,
    artifacts: Mapping[str, str | Path | None] | None = None,
    fingerprint_cache: MutableMapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest = build_artifact_manifest(
        _artifact_paths(config, report) if artifacts is None else artifacts,
        root=config.resolved_artifact_manifest_path.parent,
        metadata={
            "runner": "run_product_trace_replay_workflow",
            "status": report.get("status"),
            "corpus_status": _nested(report, "corpus", "status"),
            "runtime_baseline_status": _nested(report, "runtime_baseline", "status"),
            "action_audit_gate_status": _nested(report, "action_audit_gate", "status"),
            "action_audit_gate_enabled": _nested(report, "action_audit_gate", "gate_enabled"),
            "action_audit_gate_passed": _nested(report, "action_audit_gate", "passed"),
            "action_audit_gate_blocked_metric_count": _nested(
                report,
                "action_audit_gate",
                "blocked_metric_count",
            ),
            "action_audit_error_rate": _nested(report, "action_audit_gate", "error_rate"),
            "action_audit_missing_retrieval_action_rate": _nested(
                report,
                "action_audit_gate",
                "missing_retrieval_action_rate",
            ),
            "action_audit_missing_plan_retrieval_query_rate": _nested(
                report,
                "action_audit_gate",
                "missing_plan_retrieval_query_rate",
            ),
            "action_audit_malformed_payload_rate": _nested(
                report,
                "action_audit_gate",
                "malformed_payload_rate",
            ),
            "action_audit_unexpected_action_rate": _nested(
                report,
                "action_audit_gate",
                "unexpected_action_rate",
            ),
            "action_audit_unknown_claim_id_rate": _nested(
                report,
                "action_audit_gate",
                "unknown_claim_id_rate",
            ),
            "action_audit_gate_report": _nested(report, "paths", "action_audit_gate_report"),
            "action_execution_gate_status": _nested(report, "action_execution_gate", "status"),
            "action_execution_gate_enabled": _nested(report, "action_execution_gate", "gate_enabled"),
            "action_execution_gate_passed": _nested(report, "action_execution_gate", "passed"),
            "action_execution_gate_blocked_metric_count": _nested(
                report,
                "action_execution_gate",
                "blocked_metric_count",
            ),
            "action_execution_alignment_failed_trace_rate": _nested(
                report,
                "action_execution_gate",
                "alignment_failed_trace_rate",
            ),
            "action_execution_missing_result_rate": _nested(
                report,
                "action_execution_gate",
                "missing_result_rate",
            ),
            "action_execution_unexpected_result_rate": _nested(
                report,
                "action_execution_gate",
                "unexpected_result_rate",
            ),
            "action_execution_request_id_mismatch_rate": _nested(
                report,
                "action_execution_gate",
                "request_id_mismatch_rate",
            ),
            "action_execution_gate_report": _nested(
                report,
                "paths",
                "action_execution_gate_report",
            ),
            "selector_replay_status": _nested(report, "selector_replay", "status"),
            "workflow_total_seconds": _nested(report, "timing", "total_seconds"),
            "workflow_phase_total_seconds": _nested(report, "timing", "phase_total_seconds"),
            "workflow_corpus_seconds": _nested(report, "timing", "phases", "corpus", "seconds"),
            "workflow_runtime_baseline_seconds": _nested(
                report,
                "timing",
                "phases",
                "runtime_baseline",
                "seconds",
            ),
            "workflow_action_audit_gate_seconds": _nested(
                report,
                "timing",
                "phases",
                "action_audit_gate",
                "seconds",
            ),
            "workflow_action_execution_gate_seconds": _nested(
                report,
                "timing",
                "phases",
                "action_execution_gate",
                "seconds",
            ),
            "workflow_selector_replay_seconds": _nested(
                report,
                "timing",
                "phases",
                "selector_replay",
                "seconds",
            ),
            "workflow_runtime_drift_seconds": _nested(
                report,
                "timing",
                "phases",
                "runtime_drift",
                "seconds",
            ),
            "workflow_cache_enabled_count": _nested(report, "cache_summary", "enabled_count"),
            "workflow_cache_hit_count": _nested(report, "cache_summary", "hit_count"),
            "workflow_cache_hit_rate": _nested(report, "cache_summary", "hit_rate"),
            "optimization_status": _nested(report, "optimization", "status"),
            "optimization_recommendation_count": _nested(
                report,
                "optimization",
                "recommendation_count",
            ),
            "optimization_high_priority_count": _nested(
                report,
                "optimization",
                "priority_counts",
                "high",
            ),
            "optimization_top_recommendation": _nested(
                report,
                "optimization",
                "top_recommendations",
                0,
                "id",
            ),
            "optimization_slowest_phase": _nested(
                report,
                "optimization",
                "summary",
                "slowest_phase",
            ),
            "optimization_slowest_route": _nested(
                report,
                "optimization",
                "summary",
                "slowest_route",
            ),
            "recommended_runtime_policy_path": _nested(
                report,
                "decision",
                "recommended_runtime_policy_path",
            ),
            "recommended_runtime_policy_written": _nested(
                report,
                "runtime_baseline",
                "recommended_policy_written",
            ),
            "recommended_runtime_policy_enabled": _nested(
                report,
                "runtime_baseline",
                "recommended_policy_enabled",
            ),
            "recommended_runtime_policy_threshold_count": _nested(
                report,
                "runtime_baseline",
                "recommended_policy_threshold_count",
            ),
            "runtime_trace_scan_workers": _nested(
                report,
                "runtime_baseline",
                "trace_scan_workers",
            ),
            "runtime_trace_scan_effective_workers": _nested(
                report,
                "runtime_baseline",
                "trace_scan_effective_workers",
            ),
            "recommended_selector_candidate": _nested(
                report,
                "decision",
                "recommended_selector_candidate",
            ),
            "runtime_drift_status": _nested(report, "runtime_drift", "status"),
            "runtime_drift_gate_enabled": _nested(report, "runtime_drift", "gate_enabled"),
            "runtime_drift_drift_gate_enabled": _nested(report, "runtime_drift", "drift_gate_enabled"),
            "runtime_drift_budget_policy_gate_enabled": _nested(
                report,
                "runtime_drift",
                "runtime_budget_policy_gate_enabled",
            ),
            "runtime_drift_budget_policy_passed": _nested(
                report,
                "runtime_drift",
                "runtime_budget_policy_passed",
            ),
            "runtime_drift_compared_metric_count": _nested(
                report,
                "runtime_drift",
                "compared_metric_count",
            ),
            "runtime_drift_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "blocked_metric_count",
            ),
            "runtime_drift_covered_fact_property_metric_count": _nested(
                report,
                "runtime_drift",
                "covered_fact_property_metric_count",
            ),
            "runtime_drift_covered_fact_property_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "covered_fact_property_blocked_metric_count",
            ),
            "runtime_drift_product_trace_action_gate_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_action_gate_metric_count",
            ),
            "runtime_drift_product_trace_action_gate_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_action_gate_blocked_metric_count",
            ),
            "runtime_drift_product_trace_receipt_claim_support_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_receipt_claim_support_metric_count",
            ),
            "runtime_drift_product_trace_receipt_claim_support_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_receipt_claim_support_blocked_metric_count",
            ),
            "runtime_drift_product_trace_trajectory_audit_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_trajectory_audit_metric_count",
            ),
            "runtime_drift_product_trace_trajectory_audit_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_trajectory_audit_blocked_metric_count",
            ),
            "runtime_drift_product_trace_provenance_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_provenance_metric_count",
            ),
            "runtime_drift_product_trace_provenance_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_provenance_blocked_metric_count",
            ),
            "runtime_drift_product_trace_citation_integrity_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_citation_integrity_metric_count",
            ),
            "runtime_drift_product_trace_citation_integrity_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_citation_integrity_blocked_metric_count",
            ),
            "runtime_drift_world_model_metric_count": _nested(
                report,
                "runtime_drift",
                "world_model_metric_count",
            ),
            "runtime_drift_world_model_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "world_model_blocked_metric_count",
            ),
            "runtime_drift_context_sensitivity_metric_count": _nested(
                report,
                "runtime_drift",
                "context_sensitivity_metric_count",
            ),
            "runtime_drift_context_sensitivity_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "context_sensitivity_blocked_metric_count",
            ),
            "runtime_drift_counterfactual_robustness_metric_count": _nested(
                report,
                "runtime_drift",
                "counterfactual_robustness_metric_count",
            ),
            "runtime_drift_counterfactual_robustness_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "counterfactual_robustness_blocked_metric_count",
            ),
            "runtime_drift_claim_risk_localization_metric_count": _nested(
                report,
                "runtime_drift",
                "claim_risk_localization_metric_count",
            ),
            "runtime_drift_claim_risk_localization_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "claim_risk_localization_blocked_metric_count",
            ),
            "runtime_drift_pre_generation_risk_metric_count": _nested(
                report,
                "runtime_drift",
                "pre_generation_risk_metric_count",
            ),
            "runtime_drift_pre_generation_risk_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "pre_generation_risk_blocked_metric_count",
            ),
            "runtime_drift_pre_generation_probe_comparison_metric_count": _nested(
                report,
                "runtime_drift",
                "pre_generation_probe_comparison_metric_count",
            ),
            "runtime_drift_pre_generation_probe_comparison_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "pre_generation_probe_comparison_blocked_metric_count",
            ),
            "runtime_drift_claim_factuality_probe_comparison_metric_count": _nested(
                report,
                "runtime_drift",
                "claim_factuality_probe_comparison_metric_count",
            ),
            "runtime_drift_claim_factuality_probe_comparison_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "claim_factuality_probe_comparison_blocked_metric_count",
            ),
            "runtime_drift_counterfactual_verification_metric_count": _nested(
                report,
                "runtime_drift",
                "counterfactual_verification_metric_count",
            ),
            "runtime_drift_counterfactual_verification_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "counterfactual_verification_blocked_metric_count",
            ),
            "runtime_drift_evidence_handoff_metric_count": _nested(
                report,
                "runtime_drift",
                "evidence_handoff_metric_count",
            ),
            "runtime_drift_evidence_handoff_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "evidence_handoff_blocked_metric_count",
            ),
            "runtime_drift_fact_selfcheck_gate_metric_count": _nested(
                report,
                "runtime_drift",
                "fact_selfcheck_gate_metric_count",
            ),
            "runtime_drift_fact_selfcheck_gate_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "fact_selfcheck_gate_blocked_metric_count",
            ),
            "runtime_drift_report": _nested(report, "paths", "runtime_drift_report"),
            "runtime_drift_artifact_manifest": _nested(report, "paths", "runtime_drift_manifest"),
            "compact_json": config.compact_json,
            **dict(config.metadata),
        },
        fingerprint_cache=fingerprint_cache,
    )
    _write_json(config.resolved_artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _record_registry(
    config: ProductTraceReplayWorkflowConfig,
    report: Mapping[str, Any],
    *,
    fingerprint_cache: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    if config.registry_path is None:
        return
    manifest_verification = _mapping(report.get("manifest_verification"))
    verification_payload = _mapping(manifest_verification.get("verification"))
    verification_report = manifest_verification.get("path")
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_report(
        name=str(config.name),
        path=config.resolved_report_path,
        version=str(config.version),
        metadata={
            "workflow": "run_product_trace_replay_workflow",
            "status": report.get("status"),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "corpus_status": _nested(report, "corpus", "status"),
            "runtime_baseline_status": _nested(report, "runtime_baseline", "status"),
            "action_audit_gate_status": _nested(report, "action_audit_gate", "status"),
            "action_audit_gate_enabled": _nested(report, "action_audit_gate", "gate_enabled"),
            "action_audit_gate_passed": _nested(report, "action_audit_gate", "passed"),
            "action_audit_gate_blocked_metric_count": _nested(
                report,
                "action_audit_gate",
                "blocked_metric_count",
            ),
            "action_audit_error_rate": _nested(report, "action_audit_gate", "error_rate"),
            "action_audit_missing_retrieval_action_rate": _nested(
                report,
                "action_audit_gate",
                "missing_retrieval_action_rate",
            ),
            "action_audit_missing_plan_retrieval_query_rate": _nested(
                report,
                "action_audit_gate",
                "missing_plan_retrieval_query_rate",
            ),
            "action_audit_malformed_payload_rate": _nested(
                report,
                "action_audit_gate",
                "malformed_payload_rate",
            ),
            "action_audit_unexpected_action_rate": _nested(
                report,
                "action_audit_gate",
                "unexpected_action_rate",
            ),
            "action_audit_unknown_claim_id_rate": _nested(
                report,
                "action_audit_gate",
                "unknown_claim_id_rate",
            ),
            "action_audit_gate_report": _nested(report, "paths", "action_audit_gate_report"),
            "action_execution_gate_status": _nested(report, "action_execution_gate", "status"),
            "action_execution_gate_enabled": _nested(report, "action_execution_gate", "gate_enabled"),
            "action_execution_gate_passed": _nested(report, "action_execution_gate", "passed"),
            "action_execution_gate_blocked_metric_count": _nested(
                report,
                "action_execution_gate",
                "blocked_metric_count",
            ),
            "action_execution_alignment_failed_trace_rate": _nested(
                report,
                "action_execution_gate",
                "alignment_failed_trace_rate",
            ),
            "action_execution_missing_result_rate": _nested(
                report,
                "action_execution_gate",
                "missing_result_rate",
            ),
            "action_execution_unexpected_result_rate": _nested(
                report,
                "action_execution_gate",
                "unexpected_result_rate",
            ),
            "action_execution_request_id_mismatch_rate": _nested(
                report,
                "action_execution_gate",
                "request_id_mismatch_rate",
            ),
            "action_execution_gate_report": _nested(
                report,
                "paths",
                "action_execution_gate_report",
            ),
            "selector_replay_status": _nested(report, "selector_replay", "status"),
            "workflow_total_seconds": _nested(report, "timing", "total_seconds"),
            "workflow_phase_total_seconds": _nested(report, "timing", "phase_total_seconds"),
            "workflow_corpus_seconds": _nested(report, "timing", "phases", "corpus", "seconds"),
            "workflow_runtime_baseline_seconds": _nested(
                report,
                "timing",
                "phases",
                "runtime_baseline",
                "seconds",
            ),
            "workflow_action_audit_gate_seconds": _nested(
                report,
                "timing",
                "phases",
                "action_audit_gate",
                "seconds",
            ),
            "workflow_action_execution_gate_seconds": _nested(
                report,
                "timing",
                "phases",
                "action_execution_gate",
                "seconds",
            ),
            "workflow_selector_replay_seconds": _nested(
                report,
                "timing",
                "phases",
                "selector_replay",
                "seconds",
            ),
            "workflow_runtime_drift_seconds": _nested(
                report,
                "timing",
                "phases",
                "runtime_drift",
                "seconds",
            ),
            "workflow_cache_enabled_count": _nested(report, "cache_summary", "enabled_count"),
            "workflow_cache_hit_count": _nested(report, "cache_summary", "hit_count"),
            "workflow_cache_hit_rate": _nested(report, "cache_summary", "hit_rate"),
            "optimization_status": _nested(report, "optimization", "status"),
            "optimization_recommendation_count": _nested(
                report,
                "optimization",
                "recommendation_count",
            ),
            "optimization_high_priority_count": _nested(
                report,
                "optimization",
                "priority_counts",
                "high",
            ),
            "optimization_top_recommendation": _nested(
                report,
                "optimization",
                "top_recommendations",
                0,
                "id",
            ),
            "optimization_slowest_phase": _nested(
                report,
                "optimization",
                "summary",
                "slowest_phase",
            ),
            "optimization_slowest_route": _nested(
                report,
                "optimization",
                "summary",
                "slowest_route",
            ),
            "recommended_runtime_policy_path": _nested(
                report,
                "decision",
                "recommended_runtime_policy_path",
            ),
            "recommended_runtime_policy_written": _nested(
                report,
                "runtime_baseline",
                "recommended_policy_written",
            ),
            "recommended_runtime_policy_enabled": _nested(
                report,
                "runtime_baseline",
                "recommended_policy_enabled",
            ),
            "recommended_runtime_policy_threshold_count": _nested(
                report,
                "runtime_baseline",
                "recommended_policy_threshold_count",
            ),
            "recommended_selector_candidate": _nested(
                report,
                "decision",
                "recommended_selector_candidate",
            ),
            "recommended_selector_policy_path": _nested(
                report,
                "decision",
                "recommended_selector_policy_path",
            ),
            "runtime_drift_status": _nested(report, "runtime_drift", "status"),
            "runtime_drift_gate_enabled": _nested(report, "runtime_drift", "gate_enabled"),
            "runtime_drift_drift_gate_enabled": _nested(report, "runtime_drift", "drift_gate_enabled"),
            "runtime_drift_budget_policy_gate_enabled": _nested(
                report,
                "runtime_drift",
                "runtime_budget_policy_gate_enabled",
            ),
            "runtime_drift_budget_policy_passed": _nested(
                report,
                "runtime_drift",
                "runtime_budget_policy_passed",
            ),
            "runtime_drift_compared_metric_count": _nested(
                report,
                "runtime_drift",
                "compared_metric_count",
            ),
            "runtime_drift_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "blocked_metric_count",
            ),
            "runtime_drift_covered_fact_property_metric_count": _nested(
                report,
                "runtime_drift",
                "covered_fact_property_metric_count",
            ),
            "runtime_drift_covered_fact_property_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "covered_fact_property_blocked_metric_count",
            ),
            "runtime_drift_product_trace_action_gate_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_action_gate_metric_count",
            ),
            "runtime_drift_product_trace_action_gate_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_action_gate_blocked_metric_count",
            ),
            "runtime_drift_product_trace_receipt_claim_support_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_receipt_claim_support_metric_count",
            ),
            "runtime_drift_product_trace_receipt_claim_support_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_receipt_claim_support_blocked_metric_count",
            ),
            "runtime_drift_product_trace_trajectory_audit_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_trajectory_audit_metric_count",
            ),
            "runtime_drift_product_trace_trajectory_audit_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_trajectory_audit_blocked_metric_count",
            ),
            "runtime_drift_product_trace_provenance_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_provenance_metric_count",
            ),
            "runtime_drift_product_trace_provenance_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_provenance_blocked_metric_count",
            ),
            "runtime_drift_product_trace_citation_integrity_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_citation_integrity_metric_count",
            ),
            "runtime_drift_product_trace_citation_integrity_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "product_trace_citation_integrity_blocked_metric_count",
            ),
            "runtime_drift_world_model_metric_count": _nested(
                report,
                "runtime_drift",
                "world_model_metric_count",
            ),
            "runtime_drift_world_model_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "world_model_blocked_metric_count",
            ),
            "runtime_drift_context_sensitivity_metric_count": _nested(
                report,
                "runtime_drift",
                "context_sensitivity_metric_count",
            ),
            "runtime_drift_context_sensitivity_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "context_sensitivity_blocked_metric_count",
            ),
            "runtime_drift_counterfactual_robustness_metric_count": _nested(
                report,
                "runtime_drift",
                "counterfactual_robustness_metric_count",
            ),
            "runtime_drift_counterfactual_robustness_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "counterfactual_robustness_blocked_metric_count",
            ),
            "runtime_drift_claim_risk_localization_metric_count": _nested(
                report,
                "runtime_drift",
                "claim_risk_localization_metric_count",
            ),
            "runtime_drift_claim_risk_localization_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "claim_risk_localization_blocked_metric_count",
            ),
            "runtime_drift_pre_generation_risk_metric_count": _nested(
                report,
                "runtime_drift",
                "pre_generation_risk_metric_count",
            ),
            "runtime_drift_pre_generation_risk_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "pre_generation_risk_blocked_metric_count",
            ),
            "runtime_drift_pre_generation_probe_comparison_metric_count": _nested(
                report,
                "runtime_drift",
                "pre_generation_probe_comparison_metric_count",
            ),
            "runtime_drift_pre_generation_probe_comparison_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "pre_generation_probe_comparison_blocked_metric_count",
            ),
            "runtime_drift_claim_factuality_probe_comparison_metric_count": _nested(
                report,
                "runtime_drift",
                "claim_factuality_probe_comparison_metric_count",
            ),
            "runtime_drift_claim_factuality_probe_comparison_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "claim_factuality_probe_comparison_blocked_metric_count",
            ),
            "runtime_drift_counterfactual_verification_metric_count": _nested(
                report,
                "runtime_drift",
                "counterfactual_verification_metric_count",
            ),
            "runtime_drift_counterfactual_verification_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "counterfactual_verification_blocked_metric_count",
            ),
            "runtime_drift_evidence_handoff_metric_count": _nested(
                report,
                "runtime_drift",
                "evidence_handoff_metric_count",
            ),
            "runtime_drift_evidence_handoff_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "evidence_handoff_blocked_metric_count",
            ),
            "runtime_drift_fact_selfcheck_gate_metric_count": _nested(
                report,
                "runtime_drift",
                "fact_selfcheck_gate_metric_count",
            ),
            "runtime_drift_fact_selfcheck_gate_blocked_metric_count": _nested(
                report,
                "runtime_drift",
                "fact_selfcheck_gate_blocked_metric_count",
            ),
            "runtime_drift_report": _nested(report, "paths", "runtime_drift_report"),
            "runtime_drift_artifact_manifest": _nested(report, "paths", "runtime_drift_manifest"),
            "manifest_verified": verification_payload.get("passed"),
            "manifest_verification_report": verification_report,
            "manifest_verification_checked": verification_payload.get("checked"),
            "manifest_verification_failure_count": _failure_count(verification_payload),
            "manifest_fingerprint_cache": (
                None if config.fingerprint_cache_path is None else str(config.fingerprint_cache_path)
            ),
            "manifest_fingerprint_cache_entries": (
                None if fingerprint_cache is None else len(fingerprint_cache)
            ),
            "corpus_cache_path": _nested(report, "paths", "corpus_cache"),
            "corpus_cache_source": _nested(report, "corpus", "cache_source"),
            "corpus_cache_hit": _nested(report, "corpus", "cache_hit"),
            "corpus_cache_written": _nested(report, "corpus", "cache_written"),
            "corpus_source_cache_path": _nested(report, "paths", "corpus_source_cache"),
            "corpus_source_cache_source": _nested(report, "corpus", "source_cache_source"),
            "corpus_source_cache_hit_count": _nested(report, "corpus", "source_cache_hit_count"),
            "corpus_source_cache_miss_count": _nested(report, "corpus", "source_cache_miss_count"),
            "corpus_source_cache_written": _nested(report, "corpus", "source_cache_written"),
            "runtime_trace_records_cache_path": _nested(report, "paths", "runtime_trace_records_cache"),
            "runtime_trace_records_cache_source": _nested(
                report,
                "runtime_baseline",
                "trace_records_cache_source",
            ),
            "runtime_trace_records_cache_hit": _nested(
                report,
                "runtime_baseline",
                "trace_records_cache_hit",
            ),
            "runtime_trace_records_cache_written": _nested(
                report,
                "runtime_baseline",
                "trace_records_cache_written",
            ),
            "runtime_trace_scan_workers": _nested(
                report,
                "runtime_baseline",
                "trace_scan_workers",
            ),
            "runtime_trace_scan_effective_workers": _nested(
                report,
                "runtime_baseline",
                "trace_scan_effective_workers",
            ),
            "selector_trace_inputs_path": _nested(report, "paths", "selector_trace_inputs"),
            "selector_trace_inputs_source": _nested(report, "selector_replay", "trace_inputs_source"),
            "selector_trace_inputs_cache_hit": _nested(report, "selector_replay", "trace_inputs_cache_hit"),
            "selector_trace_inputs_cache_written": _nested(
                report,
                "selector_replay",
                "trace_inputs_cache_written",
            ),
            "compact_json": config.compact_json,
            **dict(config.metadata),
        },
    )
    runtime_policy_path = _nested(report, "paths", "runtime_recommended_policy")
    if runtime_policy_path is not None:
        registry.record_product_runtime_budget_policy(
            name=f"{config.name}-runtime-recommended-policy",
            path=str(runtime_policy_path),
            version=str(config.version),
            metadata={
                "workflow": "run_product_trace_replay_workflow",
                "source_workflow_record": f"report:{config.name}:{config.version}",
                "source_workflow_report": str(config.resolved_report_path),
                "source_runtime_baseline_report": _nested(report, "paths", "runtime_baseline_report"),
                "artifact_manifest": str(config.resolved_artifact_manifest_path),
                "policy_enabled": _nested(report, "runtime_baseline", "recommended_policy_enabled"),
                "threshold_count": _nested(
                    report,
                    "runtime_baseline",
                    "recommended_policy_threshold_count",
                ),
                "optimization_status": _nested(report, "optimization", "status"),
                "compact_json": config.compact_json,
                **dict(config.metadata),
            },
        )
    if verification_report is not None:
        registry.record_manifest_verification(
            name=f"{config.name}-verification",
            path=str(verification_report),
            version=str(config.version),
            metadata={
                "manifest_name": str(config.name),
                "manifest_path": str(config.resolved_artifact_manifest_path),
                "passed": verification_payload.get("passed"),
                "recursive": True,
            },
        )
    registry.save_json()


def _write_manifest_verification(
    config: ProductTraceReplayWorkflowConfig,
    *,
    fingerprint_cache: MutableMapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    verification = load_and_verify_artifact_manifest(
        config.resolved_artifact_manifest_path,
        recursive=True,
        fingerprint_cache=fingerprint_cache,
    )
    payload = verification.to_dict()
    path = config.resolved_verification_report_path
    _write_json(path, payload, compact=config.compact_json)
    if not verification.passed and not config.allow_manifest_verification_failures:
        raise ValueError("product trace replay artifact manifest verification failed")
    return {"path": str(path), "verification": payload}


def _load_fingerprint_cache(config: ProductTraceReplayWorkflowConfig) -> dict[str, dict[str, Any]]:
    return load_fingerprint_cache(config.fingerprint_cache_path)


def _save_fingerprint_cache(
    config: ProductTraceReplayWorkflowConfig,
    fingerprint_cache: Mapping[str, Mapping[str, Any]],
) -> None:
    save_fingerprint_cache(
        config.fingerprint_cache_path,
        fingerprint_cache,
        compact=config.compact_json,
    )


def _failure_count(verification_payload: Mapping[str, Any]) -> int | None:
    if not verification_payload:
        return None
    count = len(tuple(verification_payload.get("failures", ())))
    for nested in verification_payload.get("nested", ()):
        if isinstance(nested, Mapping):
            nested_count = _failure_count(nested)
            count += 0 if nested_count is None else nested_count
    return count


def _round_seconds(value: float) -> float:
    return round(max(0.0, float(value)), 6)


def _safe_div(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not (numeric == numeric and abs(numeric) != float("inf")):
        return None
    return numeric


def _optional_rate_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    numeric = _finite_float(value)
    if numeric is None or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be a finite rate between 0 and 1.")
    return numeric


def _candidate_from_value(
    value: RuntimeProfileSelectorCandidate | Mapping[str, Any],
) -> RuntimeProfileSelectorCandidate:
    if isinstance(value, RuntimeProfileSelectorCandidate):
        return value
    payload = dict(value)
    return RuntimeProfileSelectorCandidate(
        name=str(payload["name"]),
        policy=_mapping(payload.get("policy")),
        source=None if payload.get("source") is None else str(payload.get("source")),
    )


def _load_candidate(value: str) -> RuntimeProfileSelectorCandidate:
    if "=" in value:
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
    else:
        path = Path(value)
        name = path.stem
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"selector candidate JSON must be an object: {path}")
    return RuntimeProfileSelectorCandidate(name=name, policy=payload, source=str(path))


def _trace_paths_from_args(values: Sequence[str], globs: Sequence[str]) -> tuple[Path, ...]:
    paths = [Path(value) for value in values]
    for pattern in globs:
        paths.extend(Path(match) for match in sorted(glob.glob(pattern)))
    return _unique_paths(paths)


def _unique_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    unique = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _nested(payload: Mapping[str, Any], *keys: str | int) -> Any:
    current: Any = payload
    for key in keys:
        if isinstance(key, int):
            if not isinstance(current, Sequence) or isinstance(current, (str, bytes, bytearray)):
                return None
            if key < 0 or key >= len(current):
                return None
            current = current[key]
            continue
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value).strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _safe_artifact_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return cleaned or "artifact"


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json_text(payload, compact=compact), encoding="utf-8")


def _json_text(payload: Any, *, compact: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _parse_mapping_json(value: str | None, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return dict(payload)


def _config_from_args(args: argparse.Namespace) -> ProductTraceReplayWorkflowConfig:
    if not args.candidate:
        raise ValueError("--candidate is required for product trace replay workflow.")
    return ProductTraceReplayWorkflowConfig(
        trace_paths=_trace_paths_from_args(args.trace or (), args.trace_glob or ()),
        jsonl_paths=_unique_paths(tuple(Path(path) for path in args.jsonl or ())),
        output_dir=Path(args.output_dir),
        candidates=tuple(_load_candidate(value) for value in args.candidate),
        replay_policy_path=Path(args.replay_policy) if args.replay_policy else None,
        runtime_policy_path=Path(args.runtime_policy) if args.runtime_policy else None,
        promotion_contract_path=Path(args.promotion_contract) if args.promotion_contract else None,
        runtime_recommended_policy_path=(
            Path(args.save_runtime_recommended_policy)
            if args.save_runtime_recommended_policy
            else None
        ),
        runtime_drift_baseline_path=(
            Path(args.runtime_drift_baseline) if args.runtime_drift_baseline else None
        ),
        runtime_drift_baseline_key=args.runtime_drift_baseline_key,
        runtime_drift_baseline_name=args.runtime_drift_baseline_name,
        runtime_drift_baseline_version=args.runtime_drift_baseline_version,
        runtime_drift_budget_policy_path=(
            Path(args.runtime_drift_budget_policy) if args.runtime_drift_budget_policy else None
        ),
        runtime_drift_budget_policy_key=args.runtime_drift_budget_policy_key,
        runtime_drift_report_path=(
            Path(args.runtime_drift_report) if args.runtime_drift_report else None
        ),
        runtime_drift_artifact_manifest_path=(
            Path(args.runtime_drift_artifact_manifest)
            if args.runtime_drift_artifact_manifest
            else None
        ),
        max_runtime_drift_total_seconds_mean_ratio=args.max_runtime_drift_total_seconds_mean_ratio,
        max_runtime_drift_total_seconds_p95_ratio=args.max_runtime_drift_total_seconds_p95_ratio,
        max_runtime_drift_mean_route_duration_ratio=args.max_runtime_drift_mean_route_duration_ratio,
        max_runtime_drift_p95_route_duration_ratio=args.max_runtime_drift_p95_route_duration_ratio,
        max_runtime_drift_mean_attempted_route_count_delta=(
            args.max_runtime_drift_mean_attempted_route_count_delta
        ),
        max_runtime_drift_retrieval_use_rate_delta=args.max_runtime_drift_retrieval_use_rate_delta,
        max_runtime_drift_cache_hit_rate_drop=args.max_runtime_drift_cache_hit_rate_drop,
        max_runtime_drift_verification_skip_rate_drop=args.max_runtime_drift_verification_skip_rate_drop,
        min_runtime_drift_pre_generation_risk_coverage_rate=(
            args.min_runtime_drift_pre_generation_risk_coverage_rate
        ),
        min_runtime_drift_pre_generation_learned_risk_coverage_rate=(
            args.min_runtime_drift_pre_generation_learned_risk_coverage_rate
        ),
        max_runtime_drift_pre_generation_audit_profile_rate_increase=(
            args.max_runtime_drift_pre_generation_audit_profile_rate_increase
        ),
        max_runtime_drift_pre_generation_learned_risk_routed_rate_increase=(
            args.max_runtime_drift_pre_generation_learned_risk_routed_rate_increase
        ),
        max_runtime_drift_pre_generation_learned_risk_probability_mean_increase=(
            args.max_runtime_drift_pre_generation_learned_risk_probability_mean_increase
        ),
        min_runtime_drift_promotion_contract_coverage=args.min_runtime_drift_promotion_contract_coverage,
        min_runtime_drift_pre_generation_probe_comparison_coverage=(
            args.min_runtime_drift_pre_generation_probe_comparison_coverage
        ),
        min_runtime_drift_pre_generation_probe_comparison_manifest_verified_rate=(
            args.min_runtime_drift_pre_generation_probe_comparison_manifest_verified_rate
        ),
        min_runtime_drift_pre_generation_probe_comparison_model_count=(
            args.min_runtime_drift_pre_generation_probe_comparison_model_count
        ),
        min_runtime_drift_pre_generation_probe_comparison_run_count=(
            args.min_runtime_drift_pre_generation_probe_comparison_run_count
        ),
        min_runtime_drift_pre_generation_probe_comparison_redline_pass_rate=(
            args.min_runtime_drift_pre_generation_probe_comparison_redline_pass_rate
        ),
        max_runtime_drift_pre_generation_probe_comparison_best_test_label_auroc_drop=(
            args.max_runtime_drift_pre_generation_probe_comparison_best_test_label_auroc_drop
        ),
        max_runtime_drift_pre_generation_probe_comparison_best_redline_auroc_drop=(
            args.max_runtime_drift_pre_generation_probe_comparison_best_redline_auroc_drop
        ),
        max_runtime_drift_pre_generation_probe_comparison_best_redline_margin_drop=(
            args.max_runtime_drift_pre_generation_probe_comparison_best_redline_margin_drop
        ),
        min_runtime_drift_claim_factuality_probe_comparison_coverage=(
            args.min_runtime_drift_claim_factuality_probe_comparison_coverage
        ),
        min_runtime_drift_claim_factuality_probe_comparison_manifest_verified_rate=(
            args.min_runtime_drift_claim_factuality_probe_comparison_manifest_verified_rate
        ),
        min_runtime_drift_claim_factuality_probe_comparison_model_count=(
            args.min_runtime_drift_claim_factuality_probe_comparison_model_count
        ),
        min_runtime_drift_claim_factuality_probe_comparison_run_count=(
            args.min_runtime_drift_claim_factuality_probe_comparison_run_count
        ),
        min_runtime_drift_claim_factuality_probe_comparison_dataset_count=(
            args.min_runtime_drift_claim_factuality_probe_comparison_dataset_count
        ),
        min_runtime_drift_claim_factuality_probe_comparison_redline_pass_rate=(
            args.min_runtime_drift_claim_factuality_probe_comparison_redline_pass_rate
        ),
        max_runtime_drift_claim_factuality_probe_comparison_best_test_label_auroc_drop=(
            args.max_runtime_drift_claim_factuality_probe_comparison_best_test_label_auroc_drop
        ),
        max_runtime_drift_claim_factuality_probe_comparison_best_test_selective_accuracy_drop=(
            args.max_runtime_drift_claim_factuality_probe_comparison_best_test_selective_accuracy_drop
        ),
        max_runtime_drift_claim_factuality_probe_comparison_best_test_selective_coverage_drop=(
            args.max_runtime_drift_claim_factuality_probe_comparison_best_test_selective_coverage_drop
        ),
        max_runtime_drift_claim_factuality_probe_comparison_best_redline_auroc_drop=(
            args.max_runtime_drift_claim_factuality_probe_comparison_best_redline_auroc_drop
        ),
        max_runtime_drift_claim_factuality_probe_comparison_best_redline_margin_drop=(
            args.max_runtime_drift_claim_factuality_probe_comparison_best_redline_margin_drop
        ),
        min_runtime_drift_counterfactual_verification_coverage=(
            args.min_runtime_drift_counterfactual_verification_coverage
        ),
        min_runtime_drift_counterfactual_verification_manifest_verified_rate=(
            args.min_runtime_drift_counterfactual_verification_manifest_verified_rate
        ),
        min_runtime_drift_counterfactual_verification_record_count=(
            args.min_runtime_drift_counterfactual_verification_record_count
        ),
        min_runtime_drift_counterfactual_verification_pass_rate=(
            args.min_runtime_drift_counterfactual_verification_pass_rate
        ),
        max_runtime_drift_counterfactual_verification_false_invariance_rate=(
            args.max_runtime_drift_counterfactual_verification_false_invariance_rate
        ),
        max_runtime_drift_counterfactual_verification_flip_success_count_drop=(
            args.max_runtime_drift_counterfactual_verification_flip_success_count_drop
        ),
        min_runtime_drift_evidence_handoff_coverage=(
            args.min_runtime_drift_evidence_handoff_coverage
        ),
        min_runtime_drift_evidence_handoff_manifest_verified_rate=(
            args.min_runtime_drift_evidence_handoff_manifest_verified_rate
        ),
        min_runtime_drift_evidence_handoff_present_metric_rate=(
            args.min_runtime_drift_evidence_handoff_present_metric_rate
        ),
        max_runtime_drift_evidence_handoff_missing_metric_rate=(
            args.max_runtime_drift_evidence_handoff_missing_metric_rate
        ),
        max_runtime_drift_evidence_handoff_missing_metric_count=(
            args.max_runtime_drift_evidence_handoff_missing_metric_count
        ),
        max_runtime_drift_evidence_handoff_blocked_group_count=(
            args.max_runtime_drift_evidence_handoff_blocked_group_count
        ),
        min_runtime_drift_evidence_handoff_promoted_group_rate=(
            args.min_runtime_drift_evidence_handoff_promoted_group_rate
        ),
        min_runtime_drift_fact_selfcheck_gate_coverage=(
            args.min_runtime_drift_fact_selfcheck_gate_coverage
        ),
        min_runtime_drift_fact_selfcheck_gate_report_present_rate=(
            args.min_runtime_drift_fact_selfcheck_gate_report_present_rate
        ),
        min_runtime_drift_fact_selfcheck_gate_manifest_present_rate=(
            args.min_runtime_drift_fact_selfcheck_gate_manifest_present_rate
        ),
        min_runtime_drift_fact_selfcheck_gate_manifest_verified_rate=(
            args.min_runtime_drift_fact_selfcheck_gate_manifest_verified_rate
        ),
        min_runtime_drift_fact_selfcheck_gate_passed_rate=(
            args.min_runtime_drift_fact_selfcheck_gate_passed_rate
        ),
        min_runtime_drift_fact_selfcheck_gate_run_count=(
            args.min_runtime_drift_fact_selfcheck_gate_run_count
        ),
        max_runtime_drift_fact_selfcheck_gate_failed_run_count=(
            args.max_runtime_drift_fact_selfcheck_gate_failed_run_count
        ),
        min_runtime_drift_fact_selfcheck_gate_min_executed_rate=(
            args.min_runtime_drift_fact_selfcheck_gate_min_executed_rate
        ),
        min_runtime_drift_fact_selfcheck_gate_min_decided_rate=(
            args.min_runtime_drift_fact_selfcheck_gate_min_decided_rate
        ),
        max_runtime_drift_fact_selfcheck_gate_max_not_applicable_rate=(
            args.max_runtime_drift_fact_selfcheck_gate_max_not_applicable_rate
        ),
        min_runtime_drift_fact_selfcheck_gate_min_claim_triples_per_record=(
            args.min_runtime_drift_fact_selfcheck_gate_min_claim_triples_per_record
        ),
        min_runtime_drift_fact_selfcheck_gate_min_sample_triples_per_record=(
            args.min_runtime_drift_fact_selfcheck_gate_min_sample_triples_per_record
        ),
        min_runtime_drift_triple_extraction_fixture_matrix_coverage=(
            args.min_runtime_drift_triple_extraction_fixture_matrix_coverage
        ),
        max_runtime_drift_triple_extraction_fixture_matrix_mean_best_f1_drop=(
            args.max_runtime_drift_triple_extraction_fixture_matrix_mean_best_f1_drop
        ),
        max_runtime_drift_triple_extraction_fixture_matrix_mean_f1_lift_drop=(
            args.max_runtime_drift_triple_extraction_fixture_matrix_mean_f1_lift_drop
        ),
        min_runtime_drift_triple_claim_coverage=args.min_runtime_drift_triple_claim_coverage,
        min_runtime_drift_triple_audit_claim_coverage=args.min_runtime_drift_triple_audit_claim_coverage,
        min_runtime_drift_triple_audit_pass_rate=args.min_runtime_drift_triple_audit_pass_rate,
        min_runtime_drift_triple_slot_coverage=args.min_runtime_drift_triple_slot_coverage,
        min_runtime_drift_world_model_participating_trace_rate=(
            args.min_runtime_drift_world_model_participating_trace_rate
        ),
        min_runtime_drift_world_model_coverage_rate=(
            args.min_runtime_drift_world_model_coverage_rate
        ),
        max_runtime_drift_world_model_conflict_rate_increase=(
            args.max_runtime_drift_world_model_conflict_rate_increase
        ),
        max_runtime_drift_world_model_low_agreement_rate_increase=(
            args.max_runtime_drift_world_model_low_agreement_rate_increase
        ),
        max_runtime_drift_world_model_trace_gap_rate_increase=(
            args.max_runtime_drift_world_model_trace_gap_rate_increase
        ),
        min_runtime_drift_context_sensitivity_participating_trace_rate=(
            args.min_runtime_drift_context_sensitivity_participating_trace_rate
        ),
        min_runtime_drift_context_sensitivity_coverage_rate=(
            args.min_runtime_drift_context_sensitivity_coverage_rate
        ),
        max_runtime_drift_context_sensitivity_flagged_result_rate_increase=(
            args.max_runtime_drift_context_sensitivity_flagged_result_rate_increase
        ),
        max_runtime_drift_context_sensitivity_trace_gap_rate_increase=(
            args.max_runtime_drift_context_sensitivity_trace_gap_rate_increase
        ),
        max_runtime_drift_context_sensitivity_max_flagged_rate_increase=(
            args.max_runtime_drift_context_sensitivity_max_flagged_rate_increase
        ),
        max_runtime_drift_context_sensitivity_max_ratio_increase=(
            args.max_runtime_drift_context_sensitivity_max_ratio_increase
        ),
        min_runtime_drift_counterfactual_robustness_participating_trace_rate=(
            args.min_runtime_drift_counterfactual_robustness_participating_trace_rate
        ),
        min_runtime_drift_counterfactual_robustness_coverage_rate=(
            args.min_runtime_drift_counterfactual_robustness_coverage_rate
        ),
        min_runtime_drift_counterfactual_robustness_pass_rate=(
            args.min_runtime_drift_counterfactual_robustness_pass_rate
        ),
        min_runtime_drift_counterfactual_robustness_flip_success_rate=(
            args.min_runtime_drift_counterfactual_robustness_flip_success_rate
        ),
        max_runtime_drift_counterfactual_robustness_false_invariance_rate_increase=(
            args.max_runtime_drift_counterfactual_robustness_false_invariance_rate_increase
        ),
        max_runtime_drift_counterfactual_robustness_trace_gap_rate_increase=(
            args.max_runtime_drift_counterfactual_robustness_trace_gap_rate_increase
        ),
        min_runtime_drift_claim_risk_localization_coverage_rate=(
            args.min_runtime_drift_claim_risk_localization_coverage_rate
        ),
        max_runtime_drift_claim_risk_localization_high_risk_claim_count_increase=(
            args.max_runtime_drift_claim_risk_localization_high_risk_claim_count_increase
        ),
        max_runtime_drift_claim_risk_localization_medium_or_high_risk_claim_count_increase=(
            args.max_runtime_drift_claim_risk_localization_medium_or_high_risk_claim_count_increase
        ),
        max_runtime_drift_claim_risk_localization_entity_candidate_observation_count_increase=(
            args.max_runtime_drift_claim_risk_localization_entity_candidate_observation_count_increase
        ),
        max_runtime_drift_claim_risk_localization_unique_entity_candidate_count_increase=(
            args.max_runtime_drift_claim_risk_localization_unique_entity_candidate_count_increase
        ),
        max_runtime_drift_claim_risk_localization_high_risk_entity_candidate_count_increase=(
            args.max_runtime_drift_claim_risk_localization_high_risk_entity_candidate_count_increase
        ),
        max_runtime_drift_claim_risk_localization_medium_or_high_entity_candidate_count_increase=(
            args.max_runtime_drift_claim_risk_localization_medium_or_high_entity_candidate_count_increase
        ),
        runtime_drift_covered_fact_property_scopes=tuple(
            args.runtime_drift_covered_fact_property_scope or ()
        ),
        min_runtime_drift_covered_fact_property_metric_count=(
            args.min_runtime_drift_covered_fact_property_metric_count
        ),
        min_runtime_drift_covered_fact_min_records=args.min_runtime_drift_covered_fact_min_records,
        min_runtime_drift_covered_fact_min_source_documents=(
            args.min_runtime_drift_covered_fact_min_source_documents
        ),
        max_runtime_drift_covered_fact_min_decision_accuracy_drop=(
            args.max_runtime_drift_covered_fact_min_decision_accuracy_drop
        ),
        max_runtime_drift_covered_fact_max_false_supported_rate_increase=(
            args.max_runtime_drift_covered_fact_max_false_supported_rate_increase
        ),
        max_runtime_drift_covered_fact_min_false_refuted_rate_drop=(
            args.max_runtime_drift_covered_fact_min_false_refuted_rate_drop
        ),
        max_runtime_drift_product_trace_action_audit_error_rate_increase=(
            args.max_runtime_drift_product_trace_action_audit_error_rate_increase
        ),
        max_runtime_drift_product_trace_action_audit_missing_retrieval_action_rate_increase=(
            args.max_runtime_drift_product_trace_action_audit_missing_retrieval_action_rate_increase
        ),
        max_runtime_drift_product_trace_action_audit_missing_plan_retrieval_query_rate_increase=(
            args.max_runtime_drift_product_trace_action_audit_missing_plan_retrieval_query_rate_increase
        ),
        max_runtime_drift_product_trace_action_audit_malformed_payload_rate_increase=(
            args.max_runtime_drift_product_trace_action_audit_malformed_payload_rate_increase
        ),
        max_runtime_drift_product_trace_action_audit_unexpected_action_rate_increase=(
            args.max_runtime_drift_product_trace_action_audit_unexpected_action_rate_increase
        ),
        max_runtime_drift_product_trace_action_audit_unknown_claim_id_rate_increase=(
            args.max_runtime_drift_product_trace_action_audit_unknown_claim_id_rate_increase
        ),
        max_runtime_drift_product_trace_action_execution_alignment_failed_trace_rate_increase=(
            args.max_runtime_drift_product_trace_action_execution_alignment_failed_trace_rate_increase
        ),
        max_runtime_drift_product_trace_action_execution_missing_result_rate_increase=(
            args.max_runtime_drift_product_trace_action_execution_missing_result_rate_increase
        ),
        max_runtime_drift_product_trace_action_execution_unexpected_result_rate_increase=(
            args.max_runtime_drift_product_trace_action_execution_unexpected_result_rate_increase
        ),
        max_runtime_drift_product_trace_action_execution_request_id_mismatch_rate_increase=(
            args.max_runtime_drift_product_trace_action_execution_request_id_mismatch_rate_increase
        ),
        min_runtime_drift_product_trace_receipt_claim_support_reference_support_rate=(
            args.min_runtime_drift_product_trace_receipt_claim_support_reference_support_rate
        ),
        max_runtime_drift_product_trace_receipt_claim_support_unsupported_reference_rate_increase=(
            args.max_runtime_drift_product_trace_receipt_claim_support_unsupported_reference_rate_increase
        ),
        max_runtime_drift_product_trace_receipt_claim_support_missing_reference_rate_increase=(
            args.max_runtime_drift_product_trace_receipt_claim_support_missing_reference_rate_increase
        ),
        max_runtime_drift_product_trace_receipt_claim_support_unreceipted_reference_rate_increase=(
            args.max_runtime_drift_product_trace_receipt_claim_support_unreceipted_reference_rate_increase
        ),
        max_runtime_drift_product_trace_receipt_claim_support_failed_result_reference_rate_increase=(
            args.max_runtime_drift_product_trace_receipt_claim_support_failed_result_reference_rate_increase
        ),
        max_runtime_drift_product_trace_receipt_claim_support_fingerprint_mismatch_reference_rate_increase=(
            args.max_runtime_drift_product_trace_receipt_claim_support_fingerprint_mismatch_reference_rate_increase
        ),
        max_runtime_drift_product_trace_receipt_claim_support_unsigned_reference_rate_increase=(
            args.max_runtime_drift_product_trace_receipt_claim_support_unsigned_reference_rate_increase
        ),
        max_runtime_drift_product_trace_trajectory_audit_failed_trace_rate_increase=(
            args.max_runtime_drift_product_trace_trajectory_audit_failed_trace_rate_increase
        ),
        max_runtime_drift_product_trace_trajectory_audit_error_rate_increase=(
            args.max_runtime_drift_product_trace_trajectory_audit_error_rate_increase
        ),
        max_runtime_drift_product_trace_trajectory_audit_factual_rate_increase=(
            args.max_runtime_drift_product_trace_trajectory_audit_factual_rate_increase
        ),
        max_runtime_drift_product_trace_trajectory_audit_referential_rate_increase=(
            args.max_runtime_drift_product_trace_trajectory_audit_referential_rate_increase
        ),
        max_runtime_drift_product_trace_trajectory_audit_logical_rate_increase=(
            args.max_runtime_drift_product_trace_trajectory_audit_logical_rate_increase
        ),
        max_runtime_drift_product_trace_trajectory_audit_procedural_rate_increase=(
            args.max_runtime_drift_product_trace_trajectory_audit_procedural_rate_increase
        ),
        max_runtime_drift_product_trace_trajectory_audit_scope_rate_increase=(
            args.max_runtime_drift_product_trace_trajectory_audit_scope_rate_increase
        ),
        max_runtime_drift_product_trace_trajectory_audit_cascade_rate_increase=(
            args.max_runtime_drift_product_trace_trajectory_audit_cascade_rate_increase
        ),
        min_runtime_drift_product_trace_provenance_coverage_rate=(
            args.min_runtime_drift_product_trace_provenance_coverage_rate
        ),
        min_runtime_drift_product_trace_provenance_supported_claim_evidence_coverage=(
            args.min_runtime_drift_product_trace_provenance_supported_claim_evidence_coverage
        ),
        max_runtime_drift_product_trace_provenance_missing_reference_rate_increase=(
            args.max_runtime_drift_product_trace_provenance_missing_reference_rate_increase
        ),
        max_runtime_drift_product_trace_provenance_unsupported_supported_claim_rate_increase=(
            args.max_runtime_drift_product_trace_provenance_unsupported_supported_claim_rate_increase
        ),
        max_runtime_drift_product_trace_provenance_error_rate_increase=(
            args.max_runtime_drift_product_trace_provenance_error_rate_increase
        ),
        min_runtime_drift_product_trace_provenance_final_answer_evidence_reference_rate=(
            args.min_runtime_drift_product_trace_provenance_final_answer_evidence_reference_rate
        ),
        min_runtime_drift_product_trace_citation_integrity_participating_trace_rate=(
            args.min_runtime_drift_product_trace_citation_integrity_participating_trace_rate
        ),
        min_runtime_drift_product_trace_citation_integrity_coverage_rate=(
            args.min_runtime_drift_product_trace_citation_integrity_coverage_rate
        ),
        max_runtime_drift_product_trace_citation_integrity_mismatch_rate_increase=(
            args.max_runtime_drift_product_trace_citation_integrity_mismatch_rate_increase
        ),
        max_runtime_drift_product_trace_citation_integrity_unresolved_rate_increase=(
            args.max_runtime_drift_product_trace_citation_integrity_unresolved_rate_increase
        ),
        max_runtime_drift_product_trace_citation_integrity_issue_rate_increase=(
            args.max_runtime_drift_product_trace_citation_integrity_issue_rate_increase
        ),
        max_runtime_drift_product_trace_citation_integrity_trace_gap_rate_increase=(
            args.max_runtime_drift_product_trace_citation_integrity_trace_gap_rate_increase
        ),
        min_runtime_drift_current_trace_count=args.min_runtime_drift_current_trace_count,
        max_action_audit_error_rate=args.max_action_audit_error_rate,
        max_action_audit_missing_retrieval_rate=args.max_action_audit_missing_retrieval_rate,
        max_action_audit_missing_plan_retrieval_query_rate=(
            args.max_action_audit_missing_plan_retrieval_query_rate
        ),
        max_action_audit_malformed_payload_rate=args.max_action_audit_malformed_payload_rate,
        max_action_audit_unexpected_action_rate=args.max_action_audit_unexpected_action_rate,
        max_action_audit_unknown_claim_id_rate=args.max_action_audit_unknown_claim_id_rate,
        max_action_execution_missing_result_rate=(
            args.max_action_execution_missing_result_rate
        ),
        max_action_execution_unexpected_result_rate=(
            args.max_action_execution_unexpected_result_rate
        ),
        max_action_execution_request_id_mismatch_rate=(
            args.max_action_execution_request_id_mismatch_rate
        ),
        artifact_manifest_path=Path(args.artifact_manifest) if args.artifact_manifest else None,
        registry_path=Path(args.registry) if args.registry else None,
        name=args.name,
        version=args.version,
        metadata=_parse_mapping_json(args.metadata_json, name="--metadata-json"),
        redact_text=not bool(args.no_redact_text),
        require_runtime_trace=bool(args.require_runtime_trace),
        strict=bool(args.strict),
        limit=args.limit,
        compact_json=bool(args.compact_json),
        verify_manifest=bool(args.verify_manifest),
        verification_report_path=Path(args.verification_report) if args.verification_report else None,
        allow_manifest_verification_failures=bool(args.allow_manifest_verification_failures),
        fingerprint_cache_path=Path(args.fingerprint_cache) if args.fingerprint_cache else None,
        corpus_cache_path=Path(args.corpus_cache_json) if args.corpus_cache_json else None,
        refresh_corpus_cache=bool(args.refresh_corpus_cache),
        corpus_source_cache_path=(
            Path(args.corpus_source_cache_json) if args.corpus_source_cache_json else None
        ),
        refresh_corpus_source_cache=bool(args.refresh_corpus_source_cache),
        runtime_trace_records_cache_path=(
            Path(args.runtime_trace_records_cache_json) if args.runtime_trace_records_cache_json else None
        ),
        refresh_runtime_trace_records_cache=bool(args.refresh_runtime_trace_records_cache),
        runtime_trace_scan_workers=args.runtime_trace_scan_workers,
        selector_trace_inputs_path=(
            Path(args.selector_trace_inputs_json) if args.selector_trace_inputs_json else None
        ),
        refresh_selector_trace_inputs=bool(args.refresh_selector_trace_inputs),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = run_product_trace_replay_workflow(_config_from_args(args))
    print(_json_text(report, compact=bool(args.compact_json)), end="")
    if args.fail_on_blocked and report["status"] == "blocked":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a replay workflow over saved ProductTrace payloads")
    parser.add_argument("--trace", action="append", default=[], help="ProductTrace JSON path; repeatable")
    parser.add_argument("--trace-glob", action="append", default=[], help="glob for ProductTrace JSON files")
    parser.add_argument("--jsonl", action="append", default=[], help="ProductTrace JSONL path; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", action="append", default=[],
                        help="candidate selector policy JSON path, or name=path; repeatable")
    parser.add_argument("--replay-policy", default=None, help="RuntimeProfileSelectorReplayPolicy JSON path")
    parser.add_argument("--runtime-policy", default=None, help="ProductRuntimeBudgetPolicy JSON path")
    parser.add_argument("--promotion-contract", default=None, help="ProductPromotionContract/release report JSON path")
    parser.add_argument("--save-runtime-recommended-policy", default=None,
                        help="write the runtime baseline optimization candidate ProductRuntimeBudgetPolicy JSON")
    parser.add_argument("--runtime-drift-baseline", default=None,
                        help="prior product runtime baseline report JSON for workflow drift gate")
    parser.add_argument("--runtime-drift-baseline-key", default=None,
                        help="product_runtime_baseline registry key for workflow drift gate")
    parser.add_argument("--runtime-drift-baseline-name", default=None,
                        help="product runtime baseline registry record name for workflow drift gate")
    parser.add_argument("--runtime-drift-baseline-version", default=None,
                        help="product runtime baseline registry record version for workflow drift gate")
    parser.add_argument("--runtime-drift-budget-policy", default=None,
                        help="ProductRuntimeBudgetPolicy JSON for aggregate workflow drift gate")
    parser.add_argument("--runtime-drift-budget-policy-key", default=None,
                        help="product_runtime_budget_policy registry key for aggregate workflow drift gate")
    parser.add_argument("--runtime-drift-report", default=None,
                        help="optional child product runtime drift report path")
    parser.add_argument("--runtime-drift-artifact-manifest", default=None,
                        help="optional child product runtime drift artifact manifest path")
    parser.add_argument("--max-runtime-drift-total-seconds-mean-ratio", type=float, default=None)
    parser.add_argument("--max-runtime-drift-total-seconds-p95-ratio", type=float, default=None)
    parser.add_argument("--max-runtime-drift-mean-route-duration-ratio", type=float, default=None)
    parser.add_argument("--max-runtime-drift-p95-route-duration-ratio", type=float, default=None)
    parser.add_argument("--max-runtime-drift-mean-attempted-route-count-delta", type=float, default=None)
    parser.add_argument("--max-runtime-drift-retrieval-use-rate-delta", type=float, default=None)
    parser.add_argument("--max-runtime-drift-cache-hit-rate-drop", type=float, default=None)
    parser.add_argument("--max-runtime-drift-verification-skip-rate-drop", type=float, default=None)
    parser.add_argument("--min-runtime-drift-pre-generation-risk-coverage-rate", type=float, default=None)
    parser.add_argument(
        "--min-runtime-drift-pre-generation-learned-risk-coverage-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-pre-generation-audit-profile-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-pre-generation-learned-risk-routed-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-pre-generation-learned-risk-probability-mean-increase",
        type=float,
        default=None,
    )
    parser.add_argument("--min-runtime-drift-promotion-contract-coverage", type=float, default=None)
    parser.add_argument("--min-runtime-drift-pre-generation-probe-comparison-coverage", type=float, default=None)
    parser.add_argument(
        "--min-runtime-drift-pre-generation-probe-comparison-manifest-verified-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-pre-generation-probe-comparison-model-count",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-pre-generation-probe-comparison-run-count",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-pre-generation-probe-comparison-redline-pass-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-pre-generation-probe-comparison-best-test-label-auroc-drop",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-pre-generation-probe-comparison-best-redline-auroc-drop",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-pre-generation-probe-comparison-best-redline-margin-drop",
        type=float,
        default=None,
    )
    parser.add_argument("--min-runtime-drift-claim-factuality-probe-comparison-coverage", type=float, default=None)
    parser.add_argument(
        "--min-runtime-drift-claim-factuality-probe-comparison-manifest-verified-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-claim-factuality-probe-comparison-model-count",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-claim-factuality-probe-comparison-run-count",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-claim-factuality-probe-comparison-dataset-count",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-claim-factuality-probe-comparison-redline-pass-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-claim-factuality-probe-comparison-best-test-label-auroc-drop",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-claim-factuality-probe-comparison-best-test-selective-accuracy-drop",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-claim-factuality-probe-comparison-best-test-selective-coverage-drop",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-claim-factuality-probe-comparison-best-redline-auroc-drop",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-claim-factuality-probe-comparison-best-redline-margin-drop",
        type=float,
        default=None,
    )
    parser.add_argument("--min-runtime-drift-counterfactual-verification-coverage", type=float, default=None)
    parser.add_argument(
        "--min-runtime-drift-counterfactual-verification-manifest-verified-rate",
        type=float,
        default=None,
    )
    parser.add_argument("--min-runtime-drift-counterfactual-verification-record-count", type=float, default=None)
    parser.add_argument("--min-runtime-drift-counterfactual-verification-pass-rate", type=float, default=None)
    parser.add_argument(
        "--max-runtime-drift-counterfactual-verification-false-invariance-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-counterfactual-verification-flip-success-count-drop",
        type=float,
        default=None,
    )
    parser.add_argument("--min-runtime-drift-evidence-handoff-coverage", type=float, default=None)
    parser.add_argument(
        "--min-runtime-drift-evidence-handoff-manifest-verified-rate",
        type=float,
        default=None,
    )
    parser.add_argument("--min-runtime-drift-evidence-handoff-present-metric-rate", type=float, default=None)
    parser.add_argument("--max-runtime-drift-evidence-handoff-missing-metric-rate", type=float, default=None)
    parser.add_argument("--max-runtime-drift-evidence-handoff-missing-metric-count", type=float, default=None)
    parser.add_argument("--max-runtime-drift-evidence-handoff-blocked-group-count", type=float, default=None)
    parser.add_argument("--min-runtime-drift-evidence-handoff-promoted-group-rate", type=float, default=None)
    parser.add_argument("--min-runtime-drift-fact-selfcheck-gate-coverage", type=float, default=None)
    parser.add_argument("--min-runtime-drift-fact-selfcheck-gate-report-present-rate", type=float, default=None)
    parser.add_argument("--min-runtime-drift-fact-selfcheck-gate-manifest-present-rate", type=float, default=None)
    parser.add_argument("--min-runtime-drift-fact-selfcheck-gate-manifest-verified-rate", type=float, default=None)
    parser.add_argument("--min-runtime-drift-fact-selfcheck-gate-passed-rate", type=float, default=None)
    parser.add_argument("--min-runtime-drift-fact-selfcheck-gate-run-count", type=float, default=None)
    parser.add_argument("--max-runtime-drift-fact-selfcheck-gate-failed-run-count", type=float, default=None)
    parser.add_argument("--min-runtime-drift-fact-selfcheck-gate-min-executed-rate", type=float, default=None)
    parser.add_argument("--min-runtime-drift-fact-selfcheck-gate-min-decided-rate", type=float, default=None)
    parser.add_argument("--max-runtime-drift-fact-selfcheck-gate-max-not-applicable-rate", type=float, default=None)
    parser.add_argument(
        "--min-runtime-drift-fact-selfcheck-gate-min-claim-triples-per-record",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-fact-selfcheck-gate-min-sample-triples-per-record",
        type=float,
        default=None,
    )
    parser.add_argument("--min-runtime-drift-triple-extraction-fixture-matrix-coverage", type=float, default=None)
    parser.add_argument(
        "--max-runtime-drift-triple-extraction-fixture-matrix-mean-best-f1-drop",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-triple-extraction-fixture-matrix-mean-f1-lift-drop",
        type=float,
        default=None,
    )
    parser.add_argument("--min-runtime-drift-triple-claim-coverage", type=float, default=None)
    parser.add_argument("--min-runtime-drift-triple-audit-claim-coverage", type=float, default=None)
    parser.add_argument("--min-runtime-drift-triple-audit-pass-rate", type=float, default=None)
    parser.add_argument("--min-runtime-drift-triple-slot-coverage", type=float, default=None)
    parser.add_argument("--min-runtime-drift-world-model-participating-trace-rate", type=float, default=None)
    parser.add_argument("--min-runtime-drift-world-model-coverage-rate", type=float, default=None)
    parser.add_argument("--max-runtime-drift-world-model-conflict-rate-increase", type=float, default=None)
    parser.add_argument("--max-runtime-drift-world-model-low-agreement-rate-increase", type=float, default=None)
    parser.add_argument("--max-runtime-drift-world-model-trace-gap-rate-increase", type=float, default=None)
    parser.add_argument("--min-runtime-drift-context-sensitivity-participating-trace-rate", type=float, default=None)
    parser.add_argument("--min-runtime-drift-context-sensitivity-coverage-rate", type=float, default=None)
    parser.add_argument(
        "--max-runtime-drift-context-sensitivity-flagged-result-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-context-sensitivity-trace-gap-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-context-sensitivity-max-flagged-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-context-sensitivity-max-ratio-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-counterfactual-robustness-participating-trace-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-counterfactual-robustness-coverage-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-counterfactual-robustness-pass-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-counterfactual-robustness-flip-success-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-counterfactual-robustness-false-invariance-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-counterfactual-robustness-trace-gap-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-claim-risk-localization-coverage-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-claim-risk-localization-high-risk-claim-count-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-claim-risk-localization-medium-or-high-risk-claim-count-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-claim-risk-localization-entity-candidate-observation-count-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-claim-risk-localization-unique-entity-candidate-count-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-claim-risk-localization-high-risk-entity-candidate-count-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-claim-risk-localization-medium-or-high-entity-candidate-count-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--runtime-drift-covered-fact-property-scope",
        action="append",
        default=[],
        help="promotion contract covered-fact property scope for runtime drift gates; repeatable",
    )
    parser.add_argument("--min-runtime-drift-covered-fact-property-metric-count", type=float, default=None)
    parser.add_argument("--min-runtime-drift-covered-fact-min-records", type=float, default=None)
    parser.add_argument("--min-runtime-drift-covered-fact-min-source-documents", type=float, default=None)
    parser.add_argument("--max-runtime-drift-covered-fact-min-decision-accuracy-drop", type=float, default=None)
    parser.add_argument(
        "--max-runtime-drift-covered-fact-max-false-supported-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument("--max-runtime-drift-covered-fact-min-false-refuted-rate-drop", type=float, default=None)
    parser.add_argument("--max-runtime-drift-product-trace-action-audit-error-rate-increase", type=float, default=None)
    parser.add_argument(
        "--max-runtime-drift-product-trace-action-audit-missing-retrieval-action-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-action-audit-missing-plan-retrieval-query-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-action-audit-malformed-payload-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-action-audit-unexpected-action-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-action-audit-unknown-claim-id-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-action-execution-alignment-failed-trace-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-action-execution-missing-result-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-action-execution-unexpected-result-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-action-execution-request-id-mismatch-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-product-trace-receipt-claim-support-reference-support-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-receipt-claim-support-unsupported-reference-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-receipt-claim-support-missing-reference-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-receipt-claim-support-unreceipted-reference-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-receipt-claim-support-failed-result-reference-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-receipt-claim-support-fingerprint-mismatch-reference-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-receipt-claim-support-unsigned-reference-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-trajectory-audit-failed-trace-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-trajectory-audit-error-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-trajectory-audit-factual-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-trajectory-audit-referential-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-trajectory-audit-logical-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-trajectory-audit-procedural-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-trajectory-audit-scope-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-trajectory-audit-cascade-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-product-trace-provenance-coverage-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-product-trace-provenance-supported-claim-evidence-coverage",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-provenance-missing-reference-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-provenance-unsupported-supported-claim-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-provenance-error-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-product-trace-provenance-final-answer-evidence-reference-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-product-trace-citation-integrity-participating-trace-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-runtime-drift-product-trace-citation-integrity-coverage-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-citation-integrity-mismatch-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-citation-integrity-unresolved-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-citation-integrity-issue-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-runtime-drift-product-trace-citation-integrity-trace-gap-rate-increase",
        type=float,
        default=None,
    )
    parser.add_argument("--min-runtime-drift-current-trace-count", type=int, default=None)
    parser.add_argument("--max-action-audit-error-rate", type=float, default=None)
    parser.add_argument("--max-action-audit-missing-retrieval-rate", type=float, default=None)
    parser.add_argument("--max-action-audit-missing-plan-retrieval-query-rate", type=float, default=None)
    parser.add_argument("--max-action-audit-malformed-payload-rate", type=float, default=None)
    parser.add_argument("--max-action-audit-unexpected-action-rate", type=float, default=None)
    parser.add_argument("--max-action-audit-unknown-claim-id-rate", type=float, default=None)
    parser.add_argument("--max-action-execution-missing-result-rate", type=float, default=None)
    parser.add_argument("--max-action-execution-unexpected-result-rate", type=float, default=None)
    parser.add_argument("--max-action-execution-request-id-mismatch-rate", type=float, default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata-json", default=None)
    parser.add_argument("--no-redact-text", action="store_true")
    parser.add_argument("--require-runtime-trace", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--verify-manifest", action="store_true",
                        help="recursively verify the written workflow artifact manifest")
    parser.add_argument("--verification-report", default=None,
                        help="optional path for the manifest verification report")
    parser.add_argument("--allow-manifest-verification-failures", action="store_true",
                        help="write and register manifest verification even when it fails")
    parser.add_argument("--fingerprint-cache", default=None,
                        help="optional JSON cache for manifest fingerprint reads")
    parser.add_argument("--corpus-cache-json", default=None,
                        help="optional cache path for replay-ready corpus outputs")
    parser.add_argument("--refresh-corpus-cache", action="store_true",
                        help="rebuild --corpus-cache-json even when a valid cache exists")
    parser.add_argument("--corpus-source-cache-json", default=None,
                        help="optional per-source corpus cache for validated/redacted trace entries")
    parser.add_argument("--refresh-corpus-source-cache", action="store_true",
                        help="rebuild --corpus-source-cache-json even when a valid cache exists")
    parser.add_argument("--runtime-trace-records-cache-json", default=None,
                        help="optional runtime baseline trace-record cache path")
    parser.add_argument("--refresh-runtime-trace-records-cache", action="store_true",
                        help="rebuild --runtime-trace-records-cache-json even when a valid cache exists")
    parser.add_argument("--runtime-trace-scan-workers", type=int, default=1,
                        help="maximum worker threads for the child ProductTrace runtime baseline scan")
    parser.add_argument("--selector-trace-inputs-json", default=None,
                        help="optional selector replay input cache path")
    parser.add_argument("--refresh-selector-trace-inputs", action="store_true",
                        help="rebuild --selector-trace-inputs-json even when a valid cache exists")
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
