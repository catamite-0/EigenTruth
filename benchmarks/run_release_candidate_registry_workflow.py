"""Run release-candidate comparison and register its verified manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.compare_release_candidates import (  # noqa: E402
    ADAPTER_FAMILY_PROFILE_NAMES,
    adapter_family_profile_requires_state_transition_world_model,
    compare_release_candidates,
)
from benchmarks.config_utils import strict_positive_int  # noqa: E402
from benchmarks.promote_artifact_manifest import promote_artifact_manifest  # noqa: E402
from benchmarks.recommend_runtime_config import INSIDE_TRIGGER_BUDGET_POLICIES  # noqa: E402
from benchmarks.release_policy_profiles import (  # noqa: E402
    RELEASE_POLICY_PROFILE_NAMES,
    append_unique,
    apply_release_policy_profile_defaults,
    clean_optional_key,
)
from eigentruth.control import RUNTIME_PROFILE_NAMES, get_runtime_profile  # noqa: E402
from eigentruth.registry import (  # noqa: E402
    ArtifactVerificationContext,
    load_fingerprint_cache,
    load_json_cache,
    save_fingerprint_cache,
    save_json_cache,
)

_PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_PREFIXES: tuple[str, ...] = (
    "promotion_contract_coverage_rate",
    "triple_extraction_fixture_matrix_coverage_rate",
    "triple_extraction_fixture_matrix_mean_best_f1",
    "triple_extraction_fixture_matrix_mean_f1_lift",
)
_PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_PREFIXES: tuple[str, ...] = (
    "pre_generation_probe_comparison_coverage_rate",
    "pre_generation_probe_comparison_manifest_verified_rate",
    "pre_generation_probe_comparison_model_count",
    "pre_generation_probe_comparison_run_count",
    "pre_generation_probe_comparison_redline_pass_rate",
    "pre_generation_probe_comparison_best_test_label_auroc",
    "pre_generation_probe_comparison_best_redline_auroc",
    "pre_generation_probe_comparison_best_redline_margin",
)
_PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_EVIDENCE_PREFIXES: tuple[str, ...] = (
    "counterfactual_verification_coverage_rate",
    "counterfactual_verification_manifest_verified_rate",
    "counterfactual_verification_record_count",
    "counterfactual_verification_pass_rate",
    "counterfactual_verification_false_invariance_rate",
    "counterfactual_verification_flip_success_count",
)
_PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_PREFIXES: tuple[str, ...] = (
    "triple_claim_coverage_rate",
    "triple_audit_claim_coverage_rate",
    "triple_audit_pass_rate",
    "triple_slot_coverage_rate",
)
_PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_PREFIXES: tuple[str, ...] = (
    "covered_fact_recommended_route_property_metric_count",
    "covered_fact_recommended_route_min_records",
    "covered_fact_recommended_route_min_source_documents",
    "covered_fact_recommended_route_min_decision_accuracy",
    "covered_fact_recommended_route_max_false_supported_rate",
    "covered_fact_recommended_route_min_false_refuted_rate",
)
_PRODUCT_RUNTIME_DRIFT_ACTION_GATE_EVIDENCE_PREFIXES: tuple[str, ...] = (
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


def _apply_release_policy_profile_to_config(
    config: "ReleaseCandidateRegistryWorkflowConfig",
) -> None:
    disabled_profile_defaults: list[str] = []
    if config.external_evidence_baseline_comparison_path is not None:
        disabled_profile_defaults.append("external_evidence_baseline_comparison_key")
    if config.triple_extraction_fixture_matrix_path is not None:
        disabled_profile_defaults.append("triple_extraction_fixture_matrix_key")
    profile, values, applied = apply_release_policy_profile_defaults(
        config.release_policy_profile,
        {
            "require_structured_fact_robustness": config.require_structured_fact_robustness,
            "min_best_quality_auroc": config.min_best_quality_auroc,
            "max_uncached_forward_seconds": config.max_uncached_forward_seconds,
            "min_selected": config.min_selected,
            "min_decision_accuracy": config.min_decision_accuracy,
            "max_false_supported_rate": config.max_false_supported_rate,
            "min_false_refuted_rate": config.min_false_refuted_rate,
            "required_route_min_selected": config.required_route_min_selected,
            "required_route_min_decision_accuracy": config.required_route_min_decision_accuracy,
            "required_route_max_false_supported_rate": config.required_route_max_false_supported_rate,
            "required_route_min_false_refuted_rate": config.required_route_min_false_refuted_rate,
            "required_route_min_covered_fact_properties": config.required_route_min_covered_fact_properties,
            "required_route_min_covered_fact_property_records": (
                config.required_route_min_covered_fact_property_records
            ),
            "required_route_min_covered_fact_property_source_documents": (
                config.required_route_min_covered_fact_property_source_documents
            ),
            "required_route_min_covered_fact_property_decision_accuracy": (
                config.required_route_min_covered_fact_property_decision_accuracy
            ),
            "required_route_max_covered_fact_property_false_supported_rate": (
                config.required_route_max_covered_fact_property_false_supported_rate
            ),
            "required_route_min_covered_fact_property_false_refuted_rate": (
                config.required_route_min_covered_fact_property_false_refuted_rate
            ),
            "adapter_family_profile": config.adapter_family_profile,
            "require_state_transition_world_model": config.require_state_transition_world_model,
            "require_product_runtime_drift_promotion_evidence": (
                config.require_product_runtime_drift_promotion_evidence
            ),
            "require_product_runtime_drift_pre_generation_evidence": (
                config.require_product_runtime_drift_pre_generation_evidence
            ),
            "require_product_runtime_drift_counterfactual_evidence": (
                config.require_product_runtime_drift_counterfactual_evidence
            ),
            "require_product_runtime_drift_triple_audit_evidence": (
                config.require_product_runtime_drift_triple_audit_evidence
            ),
            "require_product_runtime_drift_covered_fact_property_evidence": (
                config.require_product_runtime_drift_covered_fact_property_evidence
            ),
            "require_product_runtime_drift_action_gate_evidence": (
                config.require_product_runtime_drift_action_gate_evidence
            ),
            "require_product_trace_action_audit_gate": config.require_product_trace_action_audit_gate,
            "require_product_trace_action_execution_gate": (
                config.require_product_trace_action_execution_gate
            ),
            "external_evidence_baseline_comparison_key": (
                config.external_evidence_baseline_comparison_key
            ),
            "triple_extraction_fixture_matrix_key": config.triple_extraction_fixture_matrix_key,
            "min_triple_extraction_corpora": config.min_triple_extraction_corpora,
            "min_triple_extraction_distinct_predicates": (
                config.min_triple_extraction_distinct_predicates
            ),
            "min_triple_extraction_external_prediction_count": (
                config.min_triple_extraction_external_prediction_count
            ),
            "min_triple_extraction_external_prediction_corpora": (
                config.min_triple_extraction_external_prediction_corpora
            ),
            "min_triple_extraction_mean_best_external_f1": (
                config.min_triple_extraction_mean_best_external_f1
            ),
        },
        disabled_defaults=disabled_profile_defaults,
    )
    object.__setattr__(config, "release_policy_profile", profile)
    for key, value in values.items():
        object.__setattr__(config, key, value)
    object.__setattr__(config, "release_policy_profile_applied_defaults", applied)


@dataclass(frozen=True)
class ReleaseCandidateRegistryWorkflowConfig:
    """Configuration for registering a promoted release-candidate manifest."""

    readiness_registry_path: Path
    release_registry_path: Path
    name: str
    version: str
    release_policy_profile: str | None = None
    route_registry_path: Path | None = None
    performance_registry_path: Path | None = None
    readiness_baseline_keys: Sequence[str] = ()
    route_baseline_keys: Sequence[str] = ()
    required_route_baseline_keys: Sequence[str] = ()
    require_structured_fact_robustness: bool = False
    structured_fact_canonical_route_key: str | None = None
    structured_fact_paraphrase_route_key: str | None = None
    performance_baseline_key: str | None = None
    selector_replay_report_path: Path | None = None
    product_runtime_drift_report_path: Path | None = None
    require_product_runtime_drift_promotion_evidence: bool = False
    require_product_runtime_drift_pre_generation_evidence: bool = False
    require_product_runtime_drift_counterfactual_evidence: bool = False
    require_product_runtime_drift_triple_audit_evidence: bool = False
    require_product_runtime_drift_covered_fact_property_evidence: bool = False
    require_product_runtime_drift_action_gate_evidence: bool = False
    release_efficiency_report_path: Path | None = None
    external_evidence_baseline_comparison_path: Path | None = None
    external_evidence_baseline_comparison_registry_path: Path | None = None
    external_evidence_baseline_comparison_key: str | None = None
    pre_generation_probe_comparison_path: Path | None = None
    pre_generation_probe_comparison_registry_path: Path | None = None
    pre_generation_probe_comparison_key: str | None = None
    frontier_release_evidence_path: Path | None = None
    frontier_release_evidence_registry_path: Path | None = None
    frontier_release_evidence_key: str | None = None
    world_model_signal_workflow_path: Path | None = None
    world_model_signal_workflow_registry_path: Path | None = None
    world_model_signal_workflow_key: str | None = None
    pathway_intervention_workflow_path: Path | None = None
    pathway_intervention_workflow_registry_path: Path | None = None
    pathway_intervention_workflow_key: str | None = None
    product_trace_replay_workflow_path: Path | None = None
    product_trace_replay_workflow_registry_path: Path | None = None
    product_trace_replay_workflow_key: str | None = None
    require_product_trace_action_audit_gate: bool = False
    require_product_trace_action_execution_gate: bool = False
    selfcheck_signal_fusion_workflow_path: Path | None = None
    selfcheck_signal_fusion_workflow_registry_path: Path | None = None
    selfcheck_signal_fusion_workflow_key: str | None = None
    uncertainty_escalation_workflow_path: Path | None = None
    uncertainty_escalation_workflow_registry_path: Path | None = None
    uncertainty_escalation_workflow_key: str | None = None
    min_uncertainty_escalation_records: int | None = None
    min_uncertainty_escalation_trigger_rate: float | None = None
    min_uncertainty_escalation_retrieval_evidence_rate: float | None = None
    max_uncertainty_escalation_final_false_accept_rate: float | None = None
    max_uncertainty_escalation_false_accept_delta: float | None = None
    feedback_policy_workflow_path: Path | None = None
    feedback_policy_workflow_registry_path: Path | None = None
    feedback_policy_workflow_key: str | None = None
    feedback_policy_min_matched_feedback_count: int | None = None
    feedback_policy_min_safety_coverage: float | None = None
    feedback_policy_max_unknown_safety_issue_rate: float | None = None
    adapter_family_matrix_path: Path | None = None
    adapter_family_profile: str | None = None
    required_adapter_routes: Sequence[str] = ()
    require_state_transition_world_model: bool = False
    triple_extraction_fixture_matrix_path: Path | None = None
    triple_extraction_fixture_matrix_registry_path: Path | None = None
    triple_extraction_fixture_matrix_key: str | None = None
    min_triple_extraction_corpora: int | None = None
    min_triple_extraction_distinct_predicates: int | None = None
    min_triple_extraction_external_prediction_count: int | None = None
    min_triple_extraction_external_prediction_corpora: int | None = None
    min_triple_extraction_mean_best_external_f1: float | None = None
    counterfactual_verification_report_path: Path | None = None
    counterfactual_verification_registry_path: Path | None = None
    counterfactual_verification_key: str | None = None
    min_counterfactual_verification_records: int | None = None
    min_counterfactual_verification_pass_rate: float | None = None
    max_counterfactual_verification_false_invariance_rate: float | None = None
    require_performance_score_dump_cache: bool = False
    min_performance_score_dump_cache_jsonl_view_hit_rate: float | None = None
    performance_drift_baseline_key: str | None = None
    max_performance_uncached_total_seconds_ratio: float | None = None
    max_performance_cached_total_seconds_ratio: float | None = None
    max_performance_cache_only_total_seconds_ratio: float | None = None
    max_performance_score_dump_cache_jsonl_view_hit_rate_drop: float | None = None
    release_report_path: Path | None = None
    artifact_manifest_path: Path | None = None
    verification_report_path: Path | None = None
    workflow_report_path: Path | None = None
    fingerprint_cache_path: Path | None = None
    json_cache_path: Path | None = None
    manifest_fingerprint_workers: int = 1
    recursive: bool = True
    allow_unverified: bool = False
    runtime_profile: str | None = None
    inside_trigger_budget_policy: str | None = None
    min_best_quality_auroc: float | None = None
    max_uncached_forward_seconds: float | None = None
    max_cache_only_seconds: float | None = None
    max_covariance_maha_last_auroc_drop: float | None = None
    max_inside_sample_count_ratio: float | None = None
    max_inside_generation_seconds_ratio: float | None = None
    min_selected: int | None = None
    min_decision_accuracy: float | None = None
    max_false_supported_rate: float | None = None
    min_false_refuted_rate: float | None = None
    max_verified_false_alarm: float | None = None
    min_verified_detection: float | None = None
    max_mean_duration_seconds: float | None = None
    max_p99_duration_seconds: float | None = None
    max_max_duration_seconds: float | None = None
    max_mean_attempted_route_count: float | None = None
    max_retrieval_use_rate: float | None = None
    max_runtime_total_seconds: float | None = None
    max_retrieval_hit_count: float | None = None
    min_claims_cache_hit_rate: float | None = None
    min_verifier_trace_cache_hit_rate: float | None = None
    require_non_oracle_evidence: bool = False
    require_retrieval_provenance_filter: bool = False
    required_retrieval_source_prefixes: Sequence[str] = ()
    required_retrieval_metadata: Mapping[str, Any] | None = None
    min_retrieval_filter_score: float | None = None
    require_retrieval_stress_control: bool = False
    retrieval_stress_manifest_path: Path | None = None
    min_stress_false_supported_rate: float | None = None
    max_stress_false_refuted_rate: float | None = None
    required_route_min_selected: int | None = None
    required_route_min_decision_accuracy: float | None = None
    required_route_max_false_supported_rate: float | None = None
    required_route_min_false_refuted_rate: float | None = None
    required_route_max_verified_false_alarm: float | None = None
    required_route_min_verified_detection: float | None = None
    required_route_max_mean_duration_seconds: float | None = None
    required_route_max_p99_duration_seconds: float | None = None
    required_route_max_max_duration_seconds: float | None = None
    required_route_max_mean_attempted_route_count: float | None = None
    required_route_max_retrieval_use_rate: float | None = None
    required_route_max_runtime_total_seconds: float | None = None
    required_route_max_retrieval_hit_count: float | None = None
    required_route_min_claims_cache_hit_rate: float | None = None
    required_route_min_verifier_trace_cache_hit_rate: float | None = None
    required_route_min_covered_fact_properties: int | None = None
    required_route_min_covered_fact_property_records: int | None = None
    required_route_min_covered_fact_property_source_documents: int | None = None
    required_route_min_covered_fact_property_decision_accuracy: float | None = None
    required_route_max_covered_fact_property_false_supported_rate: float | None = None
    required_route_min_covered_fact_property_false_refuted_rate: float | None = None
    required_route_require_non_oracle_evidence: bool = False
    required_route_require_retrieval_provenance_filter: bool = False
    required_route_required_retrieval_source_prefixes: Sequence[str] = ()
    required_route_required_retrieval_metadata: Mapping[str, Any] | None = None
    required_route_min_retrieval_filter_score: float | None = None
    required_route_require_retrieval_stress_control: bool = False
    required_route_retrieval_stress_manifest_path: Path | None = None
    required_route_min_stress_false_supported_rate: float | None = None
    required_route_max_stress_false_refuted_rate: float | None = None
    promotion_metadata: Mapping[str, Any] | None = None
    allow_non_promote: bool = False
    allow_promotion_failures: bool = False
    release_policy_profile_applied_defaults: Mapping[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _apply_release_policy_profile_to_config(self)
        object.__setattr__(self, "readiness_registry_path", Path(self.readiness_registry_path))
        object.__setattr__(self, "release_registry_path", Path(self.release_registry_path))
        if self.route_registry_path is not None:
            object.__setattr__(self, "route_registry_path", Path(self.route_registry_path))
        if self.performance_registry_path is not None:
            object.__setattr__(self, "performance_registry_path", Path(self.performance_registry_path))
        if self.selector_replay_report_path is not None:
            object.__setattr__(self, "selector_replay_report_path", Path(self.selector_replay_report_path))
        if self.product_runtime_drift_report_path is not None:
            object.__setattr__(
                self,
                "product_runtime_drift_report_path",
                Path(self.product_runtime_drift_report_path),
            )
        if self.release_efficiency_report_path is not None:
            object.__setattr__(
                self,
                "release_efficiency_report_path",
                Path(self.release_efficiency_report_path),
            )
        if self.external_evidence_baseline_comparison_path is not None:
            object.__setattr__(
                self,
                "external_evidence_baseline_comparison_path",
                Path(self.external_evidence_baseline_comparison_path),
            )
        if self.external_evidence_baseline_comparison_registry_path is not None:
            object.__setattr__(
                self,
                "external_evidence_baseline_comparison_registry_path",
                Path(self.external_evidence_baseline_comparison_registry_path),
            )
        if self.pre_generation_probe_comparison_path is not None:
            object.__setattr__(
                self,
                "pre_generation_probe_comparison_path",
                Path(self.pre_generation_probe_comparison_path),
            )
        if self.pre_generation_probe_comparison_registry_path is not None:
            object.__setattr__(
                self,
                "pre_generation_probe_comparison_registry_path",
                Path(self.pre_generation_probe_comparison_registry_path),
            )
        if self.frontier_release_evidence_path is not None:
            object.__setattr__(
                self,
                "frontier_release_evidence_path",
                Path(self.frontier_release_evidence_path),
            )
        if self.frontier_release_evidence_registry_path is not None:
            object.__setattr__(
                self,
                "frontier_release_evidence_registry_path",
                Path(self.frontier_release_evidence_registry_path),
            )
        if self.world_model_signal_workflow_path is not None:
            object.__setattr__(
                self,
                "world_model_signal_workflow_path",
                Path(self.world_model_signal_workflow_path),
            )
        if self.world_model_signal_workflow_registry_path is not None:
            object.__setattr__(
                self,
                "world_model_signal_workflow_registry_path",
                Path(self.world_model_signal_workflow_registry_path),
            )
        if self.pathway_intervention_workflow_path is not None:
            object.__setattr__(
                self,
                "pathway_intervention_workflow_path",
                Path(self.pathway_intervention_workflow_path),
            )
        if self.pathway_intervention_workflow_registry_path is not None:
            object.__setattr__(
                self,
                "pathway_intervention_workflow_registry_path",
                Path(self.pathway_intervention_workflow_registry_path),
            )
        if self.product_trace_replay_workflow_path is not None:
            object.__setattr__(
                self,
                "product_trace_replay_workflow_path",
                Path(self.product_trace_replay_workflow_path),
            )
        if self.product_trace_replay_workflow_registry_path is not None:
            object.__setattr__(
                self,
                "product_trace_replay_workflow_registry_path",
                Path(self.product_trace_replay_workflow_registry_path),
            )
        if self.selfcheck_signal_fusion_workflow_path is not None:
            object.__setattr__(
                self,
                "selfcheck_signal_fusion_workflow_path",
                Path(self.selfcheck_signal_fusion_workflow_path),
            )
        if self.selfcheck_signal_fusion_workflow_registry_path is not None:
            object.__setattr__(
                self,
                "selfcheck_signal_fusion_workflow_registry_path",
                Path(self.selfcheck_signal_fusion_workflow_registry_path),
            )
        if self.uncertainty_escalation_workflow_path is not None:
            object.__setattr__(
                self,
                "uncertainty_escalation_workflow_path",
                Path(self.uncertainty_escalation_workflow_path),
            )
        if self.uncertainty_escalation_workflow_registry_path is not None:
            object.__setattr__(
                self,
                "uncertainty_escalation_workflow_registry_path",
                Path(self.uncertainty_escalation_workflow_registry_path),
            )
        if self.feedback_policy_workflow_path is not None:
            object.__setattr__(
                self,
                "feedback_policy_workflow_path",
                Path(self.feedback_policy_workflow_path),
            )
        if self.feedback_policy_workflow_registry_path is not None:
            object.__setattr__(
                self,
                "feedback_policy_workflow_registry_path",
                Path(self.feedback_policy_workflow_registry_path),
            )
        if self.adapter_family_matrix_path is not None:
            object.__setattr__(self, "adapter_family_matrix_path", Path(self.adapter_family_matrix_path))
        if self.triple_extraction_fixture_matrix_path is not None:
            object.__setattr__(
                self,
                "triple_extraction_fixture_matrix_path",
                Path(self.triple_extraction_fixture_matrix_path),
            )
        if self.triple_extraction_fixture_matrix_registry_path is not None:
            object.__setattr__(
                self,
                "triple_extraction_fixture_matrix_registry_path",
                Path(self.triple_extraction_fixture_matrix_registry_path),
            )
        if self.counterfactual_verification_report_path is not None:
            object.__setattr__(
                self,
                "counterfactual_verification_report_path",
                Path(self.counterfactual_verification_report_path),
            )
        if self.counterfactual_verification_registry_path is not None:
            object.__setattr__(
                self,
                "counterfactual_verification_registry_path",
                Path(self.counterfactual_verification_registry_path),
            )
        if self.retrieval_stress_manifest_path is not None:
            object.__setattr__(
                self,
                "retrieval_stress_manifest_path",
                Path(self.retrieval_stress_manifest_path),
            )
        if self.required_route_retrieval_stress_manifest_path is not None:
            object.__setattr__(
                self,
                "required_route_retrieval_stress_manifest_path",
                Path(self.required_route_retrieval_stress_manifest_path),
            )
        if self.release_report_path is not None:
            object.__setattr__(self, "release_report_path", Path(self.release_report_path))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        if self.verification_report_path is not None:
            object.__setattr__(self, "verification_report_path", Path(self.verification_report_path))
        if self.workflow_report_path is not None:
            object.__setattr__(self, "workflow_report_path", Path(self.workflow_report_path))
        if self.fingerprint_cache_path is not None:
            object.__setattr__(self, "fingerprint_cache_path", Path(self.fingerprint_cache_path))
        if self.json_cache_path is not None:
            object.__setattr__(self, "json_cache_path", Path(self.json_cache_path))
        object.__setattr__(
            self,
            "required_retrieval_source_prefixes",
            tuple(str(value).strip() for value in self.required_retrieval_source_prefixes if str(value).strip()),
        )
        object.__setattr__(self, "required_retrieval_metadata", dict(self.required_retrieval_metadata or {}))
        object.__setattr__(
            self,
            "required_route_required_retrieval_source_prefixes",
            tuple(
                str(value).strip()
                for value in self.required_route_required_retrieval_source_prefixes
                if str(value).strip()
            ),
        )
        object.__setattr__(
            self,
            "required_route_required_retrieval_metadata",
            dict(self.required_route_required_retrieval_metadata or {}),
        )
        manifest_fingerprint_workers = strict_positive_int(
            self.manifest_fingerprint_workers,
            name="manifest_fingerprint_workers",
        )
        object.__setattr__(self, "manifest_fingerprint_workers", manifest_fingerprint_workers)
        if self.runtime_profile is not None:
            profile = get_runtime_profile(self.runtime_profile)
            object.__setattr__(self, "runtime_profile", profile.name)
        if self.adapter_family_profile is not None:
            profile = str(self.adapter_family_profile).strip().lower().replace("-", "_")
            if profile not in ADAPTER_FAMILY_PROFILE_NAMES:
                choices = ", ".join(ADAPTER_FAMILY_PROFILE_NAMES)
                raise ValueError(f"adapter_family_profile must be one of: {choices}")
            object.__setattr__(self, "adapter_family_profile", profile)
            if adapter_family_profile_requires_state_transition_world_model(profile):
                object.__setattr__(self, "require_state_transition_world_model", True)
        if self.inside_trigger_budget_policy is not None:
            policy = str(self.inside_trigger_budget_policy).strip().lower().replace("-", "_")
            if policy not in INSIDE_TRIGGER_BUDGET_POLICIES:
                choices = ", ".join(INSIDE_TRIGGER_BUDGET_POLICIES)
                raise ValueError(f"inside_trigger_budget_policy must be one of: {choices}")
            object.__setattr__(self, "inside_trigger_budget_policy", policy)
        object.__setattr__(
            self,
            "readiness_baseline_keys",
            tuple(str(key) for key in self.readiness_baseline_keys),
        )
        object.__setattr__(
            self,
            "route_baseline_keys",
            tuple(str(key) for key in self.route_baseline_keys),
        )
        canonical_key = clean_optional_key(self.structured_fact_canonical_route_key)
        paraphrase_key = clean_optional_key(self.structured_fact_paraphrase_route_key)
        object.__setattr__(self, "structured_fact_canonical_route_key", canonical_key)
        object.__setattr__(self, "structured_fact_paraphrase_route_key", paraphrase_key)
        if self.require_structured_fact_robustness and (
            canonical_key is None or paraphrase_key is None
        ):
            raise ValueError(
                "structured_fact robustness requires both "
                "structured_fact_canonical_route_key and structured_fact_paraphrase_route_key."
            )
        if not self.require_structured_fact_robustness and (
            canonical_key is not None or paraphrase_key is not None
        ):
            raise ValueError(
                "structured_fact route keys require require_structured_fact_robustness=True."
            )
        required_route_keys = tuple(str(key) for key in self.required_route_baseline_keys)
        if self.require_structured_fact_robustness:
            required_route_keys = append_unique(required_route_keys, (canonical_key, paraphrase_key))
        object.__setattr__(
            self,
            "required_route_baseline_keys",
            required_route_keys,
        )
        object.__setattr__(self, "required_adapter_routes", tuple(str(route) for route in self.required_adapter_routes))

    @property
    def output_root(self) -> Path:
        if self.workflow_report_path is not None:
            return self.workflow_report_path.parent
        if self.release_report_path is not None:
            return self.release_report_path.parent
        if self.artifact_manifest_path is not None:
            return self.artifact_manifest_path.parent
        return self.release_registry_path.parent

    @property
    def report_path(self) -> Path:
        return self.workflow_report_path or self.output_root / "release-candidate-registry-workflow.json"

    @property
    def comparison_path(self) -> Path:
        return self.release_report_path or self.output_root / "release-candidate-comparison.json"

    @property
    def manifest_path(self) -> Path:
        return self.artifact_manifest_path or self.output_root / "release-candidate-artifact-manifest.json"

    @property
    def verification_path(self) -> Path:
        return self.verification_report_path or self.output_root / "release-candidate-manifest-verification.json"


def run_release_candidate_registry_workflow(
    config: ReleaseCandidateRegistryWorkflowConfig,
) -> dict[str, Any]:
    """Run release comparison, write an artifact manifest, and register when eligible."""
    workflow_started = time.perf_counter()
    phase_timings: dict[str, dict[str, Any]] = {}
    verification_context = ArtifactVerificationContext(
        fingerprint_cache=load_fingerprint_cache(config.fingerprint_cache_path),
        json_cache=load_json_cache(config.json_cache_path),
    )
    fingerprint_cache = verification_context.fingerprint_cache
    json_cache = verification_context.json_cache
    json_cache_stats = verification_context.json_cache_stats
    phase_started = time.perf_counter()
    comparison = compare_release_candidates(
        readiness_registry_path=config.readiness_registry_path,
        route_registry_path=config.route_registry_path,
        readiness_baseline_keys=config.readiness_baseline_keys,
        route_baseline_keys=config.route_baseline_keys,
        required_route_baseline_keys=config.required_route_baseline_keys,
        release_policy_profile=config.release_policy_profile,
        require_structured_fact_robustness=config.require_structured_fact_robustness,
        structured_fact_canonical_route_key=config.structured_fact_canonical_route_key,
        structured_fact_paraphrase_route_key=config.structured_fact_paraphrase_route_key,
        performance_registry_path=config.performance_registry_path,
        performance_baseline_key=config.performance_baseline_key,
        selector_replay_report_path=config.selector_replay_report_path,
        product_runtime_drift_report_path=config.product_runtime_drift_report_path,
        require_product_runtime_drift_promotion_evidence=(
            config.require_product_runtime_drift_promotion_evidence
        ),
        require_product_runtime_drift_pre_generation_evidence=(
            config.require_product_runtime_drift_pre_generation_evidence
        ),
        require_product_runtime_drift_counterfactual_evidence=(
            config.require_product_runtime_drift_counterfactual_evidence
        ),
        require_product_runtime_drift_triple_audit_evidence=(
            config.require_product_runtime_drift_triple_audit_evidence
        ),
        require_product_runtime_drift_covered_fact_property_evidence=(
            config.require_product_runtime_drift_covered_fact_property_evidence
        ),
        require_product_runtime_drift_action_gate_evidence=(
            config.require_product_runtime_drift_action_gate_evidence
        ),
        release_efficiency_report_path=config.release_efficiency_report_path,
        external_evidence_baseline_comparison_path=(
            config.external_evidence_baseline_comparison_path
        ),
        external_evidence_baseline_comparison_registry_path=(
            config.external_evidence_baseline_comparison_registry_path
        ),
        external_evidence_baseline_comparison_key=(
            config.external_evidence_baseline_comparison_key
        ),
        pre_generation_probe_comparison_path=config.pre_generation_probe_comparison_path,
        pre_generation_probe_comparison_registry_path=(
            config.pre_generation_probe_comparison_registry_path
        ),
        pre_generation_probe_comparison_key=config.pre_generation_probe_comparison_key,
        frontier_release_evidence_path=config.frontier_release_evidence_path,
        frontier_release_evidence_registry_path=config.frontier_release_evidence_registry_path,
        frontier_release_evidence_key=config.frontier_release_evidence_key,
        world_model_signal_workflow_path=config.world_model_signal_workflow_path,
        world_model_signal_workflow_registry_path=config.world_model_signal_workflow_registry_path,
        world_model_signal_workflow_key=config.world_model_signal_workflow_key,
        pathway_intervention_workflow_path=config.pathway_intervention_workflow_path,
        pathway_intervention_workflow_registry_path=(
            config.pathway_intervention_workflow_registry_path
        ),
        pathway_intervention_workflow_key=config.pathway_intervention_workflow_key,
        product_trace_replay_workflow_path=config.product_trace_replay_workflow_path,
        product_trace_replay_workflow_registry_path=config.product_trace_replay_workflow_registry_path,
        product_trace_replay_workflow_key=config.product_trace_replay_workflow_key,
        require_product_trace_action_audit_gate=config.require_product_trace_action_audit_gate,
        require_product_trace_action_execution_gate=(
            config.require_product_trace_action_execution_gate
        ),
        selfcheck_signal_fusion_workflow_path=config.selfcheck_signal_fusion_workflow_path,
        selfcheck_signal_fusion_workflow_registry_path=(
            config.selfcheck_signal_fusion_workflow_registry_path
        ),
        selfcheck_signal_fusion_workflow_key=config.selfcheck_signal_fusion_workflow_key,
        uncertainty_escalation_workflow_path=config.uncertainty_escalation_workflow_path,
        uncertainty_escalation_workflow_registry_path=(
            config.uncertainty_escalation_workflow_registry_path
        ),
        uncertainty_escalation_workflow_key=config.uncertainty_escalation_workflow_key,
        min_uncertainty_escalation_records=config.min_uncertainty_escalation_records,
        min_uncertainty_escalation_trigger_rate=(
            config.min_uncertainty_escalation_trigger_rate
        ),
        min_uncertainty_escalation_retrieval_evidence_rate=(
            config.min_uncertainty_escalation_retrieval_evidence_rate
        ),
        max_uncertainty_escalation_final_false_accept_rate=(
            config.max_uncertainty_escalation_final_false_accept_rate
        ),
        max_uncertainty_escalation_false_accept_delta=(
            config.max_uncertainty_escalation_false_accept_delta
        ),
        feedback_policy_workflow_path=config.feedback_policy_workflow_path,
        feedback_policy_workflow_registry_path=config.feedback_policy_workflow_registry_path,
        feedback_policy_workflow_key=config.feedback_policy_workflow_key,
        feedback_policy_min_matched_feedback_count=config.feedback_policy_min_matched_feedback_count,
        feedback_policy_min_safety_coverage=config.feedback_policy_min_safety_coverage,
        feedback_policy_max_unknown_safety_issue_rate=config.feedback_policy_max_unknown_safety_issue_rate,
        adapter_family_matrix_path=config.adapter_family_matrix_path,
        adapter_family_profile=config.adapter_family_profile,
        required_adapter_routes=config.required_adapter_routes,
        require_state_transition_world_model=bool(config.require_state_transition_world_model),
        triple_extraction_fixture_matrix_path=config.triple_extraction_fixture_matrix_path,
        triple_extraction_fixture_matrix_registry_path=(
            config.triple_extraction_fixture_matrix_registry_path
        ),
        triple_extraction_fixture_matrix_key=config.triple_extraction_fixture_matrix_key,
        min_triple_extraction_corpora=config.min_triple_extraction_corpora,
        min_triple_extraction_distinct_predicates=(
            config.min_triple_extraction_distinct_predicates
        ),
        min_triple_extraction_external_prediction_count=(
            config.min_triple_extraction_external_prediction_count
        ),
        min_triple_extraction_external_prediction_corpora=(
            config.min_triple_extraction_external_prediction_corpora
        ),
        min_triple_extraction_mean_best_external_f1=(
            config.min_triple_extraction_mean_best_external_f1
        ),
        counterfactual_verification_report_path=config.counterfactual_verification_report_path,
        counterfactual_verification_registry_path=config.counterfactual_verification_registry_path,
        counterfactual_verification_key=config.counterfactual_verification_key,
        min_counterfactual_verification_records=config.min_counterfactual_verification_records,
        min_counterfactual_verification_pass_rate=(
            config.min_counterfactual_verification_pass_rate
        ),
        max_counterfactual_verification_false_invariance_rate=(
            config.max_counterfactual_verification_false_invariance_rate
        ),
        require_performance_score_dump_cache=config.require_performance_score_dump_cache,
        min_performance_score_dump_cache_jsonl_view_hit_rate=(
            config.min_performance_score_dump_cache_jsonl_view_hit_rate
        ),
        performance_drift_baseline_key=config.performance_drift_baseline_key,
        max_performance_uncached_total_seconds_ratio=(
            config.max_performance_uncached_total_seconds_ratio
        ),
        max_performance_cached_total_seconds_ratio=(
            config.max_performance_cached_total_seconds_ratio
        ),
        max_performance_cache_only_total_seconds_ratio=(
            config.max_performance_cache_only_total_seconds_ratio
        ),
        max_performance_score_dump_cache_jsonl_view_hit_rate_drop=(
            config.max_performance_score_dump_cache_jsonl_view_hit_rate_drop
        ),
        recursive=config.recursive,
        allow_unverified=config.allow_unverified,
        runtime_profile=config.runtime_profile,
        inside_trigger_budget_policy=config.inside_trigger_budget_policy,
        min_best_quality_auroc=config.min_best_quality_auroc,
        max_uncached_forward_seconds=config.max_uncached_forward_seconds,
        max_cache_only_seconds=config.max_cache_only_seconds,
        max_covariance_maha_last_auroc_drop=config.max_covariance_maha_last_auroc_drop,
        max_inside_sample_count_ratio=config.max_inside_sample_count_ratio,
        max_inside_generation_seconds_ratio=config.max_inside_generation_seconds_ratio,
        min_selected=config.min_selected,
        min_decision_accuracy=config.min_decision_accuracy,
        max_false_supported_rate=config.max_false_supported_rate,
        min_false_refuted_rate=config.min_false_refuted_rate,
        max_verified_false_alarm=config.max_verified_false_alarm,
        min_verified_detection=config.min_verified_detection,
        max_mean_duration_seconds=config.max_mean_duration_seconds,
        max_p99_duration_seconds=config.max_p99_duration_seconds,
        max_max_duration_seconds=config.max_max_duration_seconds,
        max_mean_attempted_route_count=config.max_mean_attempted_route_count,
        max_retrieval_use_rate=config.max_retrieval_use_rate,
        max_runtime_total_seconds=config.max_runtime_total_seconds,
        max_retrieval_hit_count=config.max_retrieval_hit_count,
        min_claims_cache_hit_rate=config.min_claims_cache_hit_rate,
        min_verifier_trace_cache_hit_rate=config.min_verifier_trace_cache_hit_rate,
        require_non_oracle_evidence=config.require_non_oracle_evidence,
        require_retrieval_provenance_filter=config.require_retrieval_provenance_filter,
        required_retrieval_source_prefixes=config.required_retrieval_source_prefixes,
        required_retrieval_metadata=config.required_retrieval_metadata,
        min_retrieval_filter_score=config.min_retrieval_filter_score,
        require_retrieval_stress_control=config.require_retrieval_stress_control,
        retrieval_stress_manifest=config.retrieval_stress_manifest_path,
        min_stress_false_supported_rate=config.min_stress_false_supported_rate,
        max_stress_false_refuted_rate=config.max_stress_false_refuted_rate,
        required_route_min_selected=config.required_route_min_selected,
        required_route_min_decision_accuracy=config.required_route_min_decision_accuracy,
        required_route_max_false_supported_rate=config.required_route_max_false_supported_rate,
        required_route_min_false_refuted_rate=config.required_route_min_false_refuted_rate,
        required_route_max_verified_false_alarm=config.required_route_max_verified_false_alarm,
        required_route_min_verified_detection=config.required_route_min_verified_detection,
        required_route_max_mean_duration_seconds=config.required_route_max_mean_duration_seconds,
        required_route_max_p99_duration_seconds=config.required_route_max_p99_duration_seconds,
        required_route_max_max_duration_seconds=config.required_route_max_max_duration_seconds,
        required_route_max_mean_attempted_route_count=config.required_route_max_mean_attempted_route_count,
        required_route_max_retrieval_use_rate=config.required_route_max_retrieval_use_rate,
        required_route_max_runtime_total_seconds=config.required_route_max_runtime_total_seconds,
        required_route_max_retrieval_hit_count=config.required_route_max_retrieval_hit_count,
        required_route_min_claims_cache_hit_rate=config.required_route_min_claims_cache_hit_rate,
        required_route_min_verifier_trace_cache_hit_rate=config.required_route_min_verifier_trace_cache_hit_rate,
        required_route_min_covered_fact_properties=config.required_route_min_covered_fact_properties,
        required_route_min_covered_fact_property_records=(
            config.required_route_min_covered_fact_property_records
        ),
        required_route_min_covered_fact_property_source_documents=(
            config.required_route_min_covered_fact_property_source_documents
        ),
        required_route_min_covered_fact_property_decision_accuracy=(
            config.required_route_min_covered_fact_property_decision_accuracy
        ),
        required_route_max_covered_fact_property_false_supported_rate=(
            config.required_route_max_covered_fact_property_false_supported_rate
        ),
        required_route_min_covered_fact_property_false_refuted_rate=(
            config.required_route_min_covered_fact_property_false_refuted_rate
        ),
        required_route_require_non_oracle_evidence=config.required_route_require_non_oracle_evidence,
        required_route_require_retrieval_provenance_filter=(
            config.required_route_require_retrieval_provenance_filter
        ),
        required_route_required_retrieval_source_prefixes=(
            config.required_route_required_retrieval_source_prefixes
        ),
        required_route_required_retrieval_metadata=config.required_route_required_retrieval_metadata,
        required_route_min_retrieval_filter_score=config.required_route_min_retrieval_filter_score,
        required_route_require_retrieval_stress_control=(
            config.required_route_require_retrieval_stress_control
        ),
        required_route_retrieval_stress_manifest=(
            config.required_route_retrieval_stress_manifest_path
        ),
        required_route_min_stress_false_supported_rate=(
            config.required_route_min_stress_false_supported_rate
        ),
        required_route_max_stress_false_refuted_rate=(
            config.required_route_max_stress_false_refuted_rate
        ),
        notes=("release candidate registry workflow",),
        fingerprint_cache=fingerprint_cache,
        json_cache=json_cache,
        json_cache_stats=json_cache_stats,
        manifest_fingerprint_workers=config.manifest_fingerprint_workers,
    )
    comparison = _comparison_with_registry_config(comparison, config)
    _record_phase_seconds("compare", phase_timings, phase_started)
    phase_started = time.perf_counter()
    _write_json_payload(config.comparison_path, comparison)
    _record_phase_seconds("comparison_write", phase_timings, phase_started)
    phase_started = time.perf_counter()
    _write_artifact_manifest(config, comparison, verification_context=verification_context)
    _record_phase_seconds("manifest_build", phase_timings, phase_started)

    release_decision = dict(comparison.get("decision") or {})
    release_status = str(release_decision.get("status"))
    promotion = None
    blocking_reasons = []
    if release_status != "promote":
        blocking_reasons.append("release candidate comparison did not promote")

    if release_status == "promote" or config.allow_non_promote:
        phase_started = time.perf_counter()
        promotion = promote_artifact_manifest(
            manifest_path=config.manifest_path,
            registry_path=config.release_registry_path,
            name=config.name,
            version=config.version,
            verification_report_path=config.verification_path,
            recursive=config.recursive,
            allow_failures=config.allow_promotion_failures,
            metadata=_promotion_metadata(config, comparison),
            verification_context=verification_context,
            manifest_fingerprint_workers=config.manifest_fingerprint_workers,
        )
        _record_phase_seconds("promotion", phase_timings, phase_started)
        if not dict(promotion.get("verification") or {}).get("passed", False):
            blocking_reasons.append("release candidate manifest verification did not pass")
    else:
        phase_timings["promotion"] = {
            "seconds": 0.0,
            "skipped": True,
        }

    decision = _registry_workflow_decision(
        release_status=release_status,
        promotion=promotion,
        blocking_reasons=blocking_reasons,
    )
    payload = {
        "schema_version": 1,
        "workflow": "release_candidate_registry_workflow",
        "config": {
            "readiness_registry": str(config.readiness_registry_path),
            "route_registry": str(config.route_registry_path or config.readiness_registry_path),
            "performance_registry": str(config.performance_registry_path or config.readiness_registry_path),
            "release_policy_profile": config.release_policy_profile,
            "release_policy_profile_applied_defaults": dict(config.release_policy_profile_applied_defaults),
            "readiness_baseline_keys": tuple(config.readiness_baseline_keys),
            "route_baseline_keys": tuple(config.route_baseline_keys),
            "performance_baseline_key": config.performance_baseline_key,
            "selector_replay_report": (
                None
                if config.selector_replay_report_path is None
                else str(config.selector_replay_report_path)
            ),
            "product_runtime_drift_report": (
                None
                if config.product_runtime_drift_report_path is None
                else str(config.product_runtime_drift_report_path)
            ),
            "require_product_runtime_drift_promotion_evidence": (
                config.require_product_runtime_drift_promotion_evidence
            ),
            "require_product_runtime_drift_pre_generation_evidence": (
                config.require_product_runtime_drift_pre_generation_evidence
            ),
            "require_product_runtime_drift_counterfactual_evidence": (
                config.require_product_runtime_drift_counterfactual_evidence
            ),
            "require_product_runtime_drift_triple_audit_evidence": (
                config.require_product_runtime_drift_triple_audit_evidence
            ),
            "require_product_runtime_drift_covered_fact_property_evidence": (
                config.require_product_runtime_drift_covered_fact_property_evidence
            ),
            "require_product_runtime_drift_action_gate_evidence": (
                config.require_product_runtime_drift_action_gate_evidence
            ),
            "release_efficiency_report": (
                None
                if config.release_efficiency_report_path is None
                else str(config.release_efficiency_report_path)
            ),
            "external_evidence_baseline_comparison": (
                None
                if config.external_evidence_baseline_comparison_path is None
                else str(config.external_evidence_baseline_comparison_path)
            ),
            "external_evidence_baseline_comparison_registry": (
                None
                if config.external_evidence_baseline_comparison_registry_path is None
                else str(config.external_evidence_baseline_comparison_registry_path)
            ),
            "external_evidence_baseline_comparison_key": (
                config.external_evidence_baseline_comparison_key
            ),
            "pre_generation_probe_comparison": (
                None
                if config.pre_generation_probe_comparison_path is None
                else str(config.pre_generation_probe_comparison_path)
            ),
            "pre_generation_probe_comparison_registry": (
                None
                if config.pre_generation_probe_comparison_registry_path is None
                else str(config.pre_generation_probe_comparison_registry_path)
            ),
            "pre_generation_probe_comparison_key": config.pre_generation_probe_comparison_key,
            "frontier_release_evidence": (
                None
                if config.frontier_release_evidence_path is None
                else str(config.frontier_release_evidence_path)
            ),
            "frontier_release_evidence_registry": (
                None
                if config.frontier_release_evidence_registry_path is None
                else str(config.frontier_release_evidence_registry_path)
            ),
            "frontier_release_evidence_key": config.frontier_release_evidence_key,
            "world_model_signal_workflow": (
                None
                if config.world_model_signal_workflow_path is None
                else str(config.world_model_signal_workflow_path)
            ),
            "world_model_signal_workflow_registry": (
                None
                if config.world_model_signal_workflow_registry_path is None
                else str(config.world_model_signal_workflow_registry_path)
            ),
            "world_model_signal_workflow_key": config.world_model_signal_workflow_key,
            "pathway_intervention_workflow": (
                None
                if config.pathway_intervention_workflow_path is None
                else str(config.pathway_intervention_workflow_path)
            ),
            "pathway_intervention_workflow_registry": (
                None
                if config.pathway_intervention_workflow_registry_path is None
                else str(config.pathway_intervention_workflow_registry_path)
            ),
            "pathway_intervention_workflow_key": config.pathway_intervention_workflow_key,
            "product_trace_replay_workflow": (
                None
                if config.product_trace_replay_workflow_path is None
                else str(config.product_trace_replay_workflow_path)
            ),
            "product_trace_replay_workflow_registry": (
                None
                if config.product_trace_replay_workflow_registry_path is None
                else str(config.product_trace_replay_workflow_registry_path)
            ),
            "product_trace_replay_workflow_key": config.product_trace_replay_workflow_key,
            "require_product_trace_action_audit_gate": config.require_product_trace_action_audit_gate,
            "require_product_trace_action_execution_gate": (
                config.require_product_trace_action_execution_gate
            ),
            "selfcheck_signal_fusion_workflow": (
                None
                if config.selfcheck_signal_fusion_workflow_path is None
                else str(config.selfcheck_signal_fusion_workflow_path)
            ),
            "selfcheck_signal_fusion_workflow_registry": (
                None
                if config.selfcheck_signal_fusion_workflow_registry_path is None
                else str(config.selfcheck_signal_fusion_workflow_registry_path)
            ),
            "selfcheck_signal_fusion_workflow_key": config.selfcheck_signal_fusion_workflow_key,
            "uncertainty_escalation_workflow": (
                None
                if config.uncertainty_escalation_workflow_path is None
                else str(config.uncertainty_escalation_workflow_path)
            ),
            "uncertainty_escalation_workflow_registry": (
                None
                if config.uncertainty_escalation_workflow_registry_path is None
                else str(config.uncertainty_escalation_workflow_registry_path)
            ),
            "uncertainty_escalation_workflow_key": config.uncertainty_escalation_workflow_key,
            "min_uncertainty_escalation_records": config.min_uncertainty_escalation_records,
            "min_uncertainty_escalation_trigger_rate": (
                config.min_uncertainty_escalation_trigger_rate
            ),
            "min_uncertainty_escalation_retrieval_evidence_rate": (
                config.min_uncertainty_escalation_retrieval_evidence_rate
            ),
            "max_uncertainty_escalation_final_false_accept_rate": (
                config.max_uncertainty_escalation_final_false_accept_rate
            ),
            "max_uncertainty_escalation_false_accept_delta": (
                config.max_uncertainty_escalation_false_accept_delta
            ),
            "feedback_policy_workflow": (
                None
                if config.feedback_policy_workflow_path is None
                else str(config.feedback_policy_workflow_path)
            ),
            "feedback_policy_workflow_registry": (
                None
                if config.feedback_policy_workflow_registry_path is None
                else str(config.feedback_policy_workflow_registry_path)
            ),
            "feedback_policy_workflow_key": config.feedback_policy_workflow_key,
            "feedback_policy_min_matched_feedback_count": config.feedback_policy_min_matched_feedback_count,
            "feedback_policy_min_safety_coverage": config.feedback_policy_min_safety_coverage,
            "feedback_policy_max_unknown_safety_issue_rate": (
                config.feedback_policy_max_unknown_safety_issue_rate
            ),
            "required_route_baseline_keys": tuple(config.required_route_baseline_keys),
            "require_structured_fact_robustness": config.require_structured_fact_robustness,
            "structured_fact_canonical_route_key": config.structured_fact_canonical_route_key,
            "structured_fact_paraphrase_route_key": config.structured_fact_paraphrase_route_key,
            "adapter_family_matrix": (
                None
                if config.adapter_family_matrix_path is None
                else str(config.adapter_family_matrix_path)
            ),
            "adapter_family_profile": config.adapter_family_profile,
            "adapter_family_profile_requires_state_transition_world_model": (
                adapter_family_profile_requires_state_transition_world_model(config.adapter_family_profile)
            ),
            "required_adapter_routes": tuple(config.required_adapter_routes),
            "require_state_transition_world_model": bool(config.require_state_transition_world_model),
            "triple_extraction_fixture_matrix": (
                None
                if config.triple_extraction_fixture_matrix_path is None
                else str(config.triple_extraction_fixture_matrix_path)
            ),
            "triple_extraction_fixture_matrix_registry": (
                None
                if config.triple_extraction_fixture_matrix_registry_path is None
                else str(config.triple_extraction_fixture_matrix_registry_path)
            ),
            "triple_extraction_fixture_matrix_key": config.triple_extraction_fixture_matrix_key,
            "min_triple_extraction_corpora": config.min_triple_extraction_corpora,
            "min_triple_extraction_distinct_predicates": (
                config.min_triple_extraction_distinct_predicates
            ),
            "min_triple_extraction_external_prediction_count": (
                config.min_triple_extraction_external_prediction_count
            ),
            "min_triple_extraction_external_prediction_corpora": (
                config.min_triple_extraction_external_prediction_corpora
            ),
            "min_triple_extraction_mean_best_external_f1": (
                config.min_triple_extraction_mean_best_external_f1
            ),
            "counterfactual_verification_report": (
                None
                if config.counterfactual_verification_report_path is None
                else str(config.counterfactual_verification_report_path)
            ),
            "counterfactual_verification_registry": (
                None
                if config.counterfactual_verification_registry_path is None
                else str(config.counterfactual_verification_registry_path)
            ),
            "counterfactual_verification_key": config.counterfactual_verification_key,
            "min_counterfactual_verification_records": (
                config.min_counterfactual_verification_records
            ),
            "min_counterfactual_verification_pass_rate": (
                config.min_counterfactual_verification_pass_rate
            ),
            "max_counterfactual_verification_false_invariance_rate": (
                config.max_counterfactual_verification_false_invariance_rate
            ),
            "require_performance_score_dump_cache": config.require_performance_score_dump_cache,
            "min_performance_score_dump_cache_jsonl_view_hit_rate": (
                config.min_performance_score_dump_cache_jsonl_view_hit_rate
            ),
            "performance_drift_baseline_key": config.performance_drift_baseline_key,
            "max_performance_uncached_total_seconds_ratio": (
                config.max_performance_uncached_total_seconds_ratio
            ),
            "max_performance_cached_total_seconds_ratio": (
                config.max_performance_cached_total_seconds_ratio
            ),
            "max_performance_cache_only_total_seconds_ratio": (
                config.max_performance_cache_only_total_seconds_ratio
            ),
            "max_performance_score_dump_cache_jsonl_view_hit_rate_drop": (
                config.max_performance_score_dump_cache_jsonl_view_hit_rate_drop
            ),
            "release_registry": str(config.release_registry_path),
            "name": config.name,
            "version": config.version,
            "release_report": str(config.comparison_path),
            "artifact_manifest": str(config.manifest_path),
            "fingerprint_cache": (
                None if config.fingerprint_cache_path is None else str(config.fingerprint_cache_path)
            ),
            "artifact_json_cache": (
                None if config.json_cache_path is None else str(config.json_cache_path)
            ),
            "manifest_fingerprint_workers": config.manifest_fingerprint_workers,
            "recursive": config.recursive,
            "allow_non_promote": config.allow_non_promote,
            "allow_promotion_failures": config.allow_promotion_failures,
            "runtime_profile": config.runtime_profile,
            "inside_trigger_budget_policy": config.inside_trigger_budget_policy,
            "max_covariance_maha_last_auroc_drop": config.max_covariance_maha_last_auroc_drop,
            "required_route_min_selected": config.required_route_min_selected,
            "required_route_min_decision_accuracy": config.required_route_min_decision_accuracy,
            "required_route_max_false_supported_rate": config.required_route_max_false_supported_rate,
            "required_route_min_false_refuted_rate": config.required_route_min_false_refuted_rate,
            "required_route_max_verified_false_alarm": config.required_route_max_verified_false_alarm,
            "required_route_min_verified_detection": config.required_route_min_verified_detection,
            "required_route_max_mean_duration_seconds": config.required_route_max_mean_duration_seconds,
            "required_route_max_p99_duration_seconds": config.required_route_max_p99_duration_seconds,
            "required_route_max_max_duration_seconds": config.required_route_max_max_duration_seconds,
            "required_route_max_mean_attempted_route_count": (
                config.required_route_max_mean_attempted_route_count
            ),
            "required_route_max_retrieval_use_rate": config.required_route_max_retrieval_use_rate,
            "required_route_max_runtime_total_seconds": config.required_route_max_runtime_total_seconds,
            "required_route_max_retrieval_hit_count": config.required_route_max_retrieval_hit_count,
            "required_route_min_claims_cache_hit_rate": config.required_route_min_claims_cache_hit_rate,
            "required_route_min_verifier_trace_cache_hit_rate": (
                config.required_route_min_verifier_trace_cache_hit_rate
            ),
            "required_route_min_covered_fact_properties": config.required_route_min_covered_fact_properties,
            "required_route_min_covered_fact_property_records": (
                config.required_route_min_covered_fact_property_records
            ),
            "required_route_min_covered_fact_property_source_documents": (
                config.required_route_min_covered_fact_property_source_documents
            ),
            "required_route_min_covered_fact_property_decision_accuracy": (
                config.required_route_min_covered_fact_property_decision_accuracy
            ),
            "required_route_max_covered_fact_property_false_supported_rate": (
                config.required_route_max_covered_fact_property_false_supported_rate
            ),
            "required_route_min_covered_fact_property_false_refuted_rate": (
                config.required_route_min_covered_fact_property_false_refuted_rate
            ),
            "require_non_oracle_evidence": config.require_non_oracle_evidence,
            "require_retrieval_provenance_filter": config.require_retrieval_provenance_filter,
            "required_retrieval_source_prefixes": list(config.required_retrieval_source_prefixes),
            "required_retrieval_metadata": dict(config.required_retrieval_metadata or {}),
            "min_retrieval_filter_score": config.min_retrieval_filter_score,
            "require_retrieval_stress_control": config.require_retrieval_stress_control,
            "retrieval_stress_manifest": (
                None
                if config.retrieval_stress_manifest_path is None
                else str(config.retrieval_stress_manifest_path)
            ),
            "min_stress_false_supported_rate": config.min_stress_false_supported_rate,
            "max_stress_false_refuted_rate": config.max_stress_false_refuted_rate,
            "required_route_require_non_oracle_evidence": config.required_route_require_non_oracle_evidence,
            "required_route_require_retrieval_stress_control": (
                config.required_route_require_retrieval_stress_control
            ),
            "required_route_retrieval_stress_manifest": (
                None
                if config.required_route_retrieval_stress_manifest_path is None
                else str(config.required_route_retrieval_stress_manifest_path)
            ),
            "required_route_min_stress_false_supported_rate": (
                config.required_route_min_stress_false_supported_rate
            ),
            "required_route_max_stress_false_refuted_rate": (
                config.required_route_max_stress_false_refuted_rate
            ),
            "required_route_require_retrieval_provenance_filter": (
                config.required_route_require_retrieval_provenance_filter
            ),
            "required_route_required_retrieval_source_prefixes": list(
                config.required_route_required_retrieval_source_prefixes
            ),
            "required_route_required_retrieval_metadata": dict(
                config.required_route_required_retrieval_metadata or {}
            ),
            "required_route_min_retrieval_filter_score": (
                config.required_route_min_retrieval_filter_score
            ),
        },
        "release_candidate_comparison": comparison,
        "promotion": promotion,
        "decision": decision,
        "artifact_cache": verification_context.cache_summary(),
        "timing": _workflow_timing(phase_timings, started_at=workflow_started),
    }
    phase_started = time.perf_counter()
    _write_json_payload(config.report_path, payload)
    _record_phase_seconds("workflow_report_write", phase_timings, phase_started)
    payload["timing"] = _workflow_timing(phase_timings, started_at=workflow_started)
    _write_json_payload(config.report_path, payload)
    save_fingerprint_cache(config.fingerprint_cache_path, verification_context.fingerprint_cache or {})
    save_json_cache(config.json_cache_path, verification_context.json_cache or {})
    return payload


def _record_phase_seconds(
    name: str,
    timings: dict[str, dict[str, Any]],
    started_at: float,
) -> None:
    timings[name] = {
        "seconds": _round_seconds(time.perf_counter() - started_at),
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


def _write_json_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _comparison_with_registry_config(
    comparison: Mapping[str, Any],
    config: ReleaseCandidateRegistryWorkflowConfig,
) -> dict[str, Any]:
    payload = dict(comparison)
    comparison_config = dict(payload.get("config") or {})
    comparison_config.update({
        "release_policy_profile": config.release_policy_profile,
        "release_policy_profile_applied_defaults": dict(config.release_policy_profile_applied_defaults),
        "required_route_baseline_keys": list(config.required_route_baseline_keys),
        "require_structured_fact_robustness": config.require_structured_fact_robustness,
        "structured_fact_canonical_route_key": config.structured_fact_canonical_route_key,
        "structured_fact_paraphrase_route_key": config.structured_fact_paraphrase_route_key,
        "adapter_family_profile": config.adapter_family_profile,
        "adapter_family_profile_requires_state_transition_world_model": (
            adapter_family_profile_requires_state_transition_world_model(config.adapter_family_profile)
        ),
        "required_adapter_routes": list(config.required_adapter_routes),
        "require_state_transition_world_model": bool(config.require_state_transition_world_model),
        "world_model_signal_workflow": (
            comparison_config.get("world_model_signal_workflow")
            if config.world_model_signal_workflow_path is None
            else str(config.world_model_signal_workflow_path)
        ),
        "world_model_signal_workflow_registry": (
            comparison_config.get("world_model_signal_workflow_registry")
            if config.world_model_signal_workflow_registry_path is None
            else str(config.world_model_signal_workflow_registry_path)
        ),
        "world_model_signal_workflow_key": config.world_model_signal_workflow_key,
        "pathway_intervention_workflow": (
            comparison_config.get("pathway_intervention_workflow")
            if config.pathway_intervention_workflow_path is None
            else str(config.pathway_intervention_workflow_path)
        ),
        "pathway_intervention_workflow_registry": (
            comparison_config.get("pathway_intervention_workflow_registry")
            if config.pathway_intervention_workflow_registry_path is None
            else str(config.pathway_intervention_workflow_registry_path)
        ),
        "pathway_intervention_workflow_key": config.pathway_intervention_workflow_key,
        "uncertainty_escalation_workflow": (
            comparison_config.get("uncertainty_escalation_workflow")
            if config.uncertainty_escalation_workflow_path is None
            else str(config.uncertainty_escalation_workflow_path)
        ),
        "uncertainty_escalation_workflow_registry": (
            comparison_config.get("uncertainty_escalation_workflow_registry")
            if config.uncertainty_escalation_workflow_registry_path is None
            else str(config.uncertainty_escalation_workflow_registry_path)
        ),
        "uncertainty_escalation_workflow_key": config.uncertainty_escalation_workflow_key,
        "min_uncertainty_escalation_records": config.min_uncertainty_escalation_records,
        "min_uncertainty_escalation_trigger_rate": (
            config.min_uncertainty_escalation_trigger_rate
        ),
        "min_uncertainty_escalation_retrieval_evidence_rate": (
            config.min_uncertainty_escalation_retrieval_evidence_rate
        ),
        "max_uncertainty_escalation_final_false_accept_rate": (
            config.max_uncertainty_escalation_final_false_accept_rate
        ),
        "max_uncertainty_escalation_false_accept_delta": (
            config.max_uncertainty_escalation_false_accept_delta
        ),
        "triple_extraction_fixture_matrix": (
            comparison_config.get("triple_extraction_fixture_matrix")
            if config.triple_extraction_fixture_matrix_path is None
            else str(config.triple_extraction_fixture_matrix_path)
        ),
        "triple_extraction_fixture_matrix_registry": (
            comparison_config.get("triple_extraction_fixture_matrix_registry")
            if config.triple_extraction_fixture_matrix_registry_path is None
            else str(config.triple_extraction_fixture_matrix_registry_path)
        ),
        "triple_extraction_fixture_matrix_key": config.triple_extraction_fixture_matrix_key,
        "min_triple_extraction_corpora": config.min_triple_extraction_corpora,
        "min_triple_extraction_distinct_predicates": (
            config.min_triple_extraction_distinct_predicates
        ),
        "min_triple_extraction_external_prediction_count": (
            config.min_triple_extraction_external_prediction_count
        ),
        "min_triple_extraction_external_prediction_corpora": (
            config.min_triple_extraction_external_prediction_corpora
        ),
        "min_triple_extraction_mean_best_external_f1": (
            config.min_triple_extraction_mean_best_external_f1
        ),
        "counterfactual_verification_report": (
            comparison_config.get("counterfactual_verification_report")
            if config.counterfactual_verification_report_path is None
            else str(config.counterfactual_verification_report_path)
        ),
        "counterfactual_verification_registry": (
            comparison_config.get("counterfactual_verification_registry")
            if config.counterfactual_verification_registry_path is None
            else str(config.counterfactual_verification_registry_path)
        ),
        "counterfactual_verification_key": config.counterfactual_verification_key,
        "min_counterfactual_verification_records": (
            config.min_counterfactual_verification_records
        ),
        "min_counterfactual_verification_pass_rate": (
            config.min_counterfactual_verification_pass_rate
        ),
        "max_counterfactual_verification_false_invariance_rate": (
            config.max_counterfactual_verification_false_invariance_rate
        ),
        "external_evidence_baseline_comparison": (
            comparison_config.get("external_evidence_baseline_comparison")
            if config.external_evidence_baseline_comparison_path is None
            else str(config.external_evidence_baseline_comparison_path)
        ),
        "external_evidence_baseline_comparison_registry": (
            comparison_config.get("external_evidence_baseline_comparison_registry")
            if config.external_evidence_baseline_comparison_registry_path is None
            else str(config.external_evidence_baseline_comparison_registry_path)
        ),
        "external_evidence_baseline_comparison_key": (
            comparison_config.get("external_evidence_baseline_comparison_key")
            if config.external_evidence_baseline_comparison_key is None
            else config.external_evidence_baseline_comparison_key
        ),
        "pre_generation_probe_comparison": (
            comparison_config.get("pre_generation_probe_comparison")
            if config.pre_generation_probe_comparison_path is None
            else str(config.pre_generation_probe_comparison_path)
        ),
        "pre_generation_probe_comparison_registry": (
            comparison_config.get("pre_generation_probe_comparison_registry")
            if config.pre_generation_probe_comparison_registry_path is None
            else str(config.pre_generation_probe_comparison_registry_path)
        ),
        "pre_generation_probe_comparison_key": (
            comparison_config.get("pre_generation_probe_comparison_key")
            if config.pre_generation_probe_comparison_key is None
            else config.pre_generation_probe_comparison_key
        ),
    })
    payload["config"] = comparison_config
    return payload


def _round_seconds(value: float) -> float:
    return round(max(0.0, float(value)), 6)


def _write_artifact_manifest(
    config: ReleaseCandidateRegistryWorkflowConfig,
    comparison: Mapping[str, Any],
    *,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any]:
    candidate = dict(comparison.get("release_candidate") or {})
    manifests = dict(candidate.get("manifests") or {})
    artifacts: dict[str, str | Path | None] = {
        "release_candidate_report": config.comparison_path,
        "readiness_manifest": manifests.get("readiness_manifest"),
        "route_manifest": manifests.get("route_manifest"),
        "performance_manifest": manifests.get("performance_manifest"),
        "selector_replay_manifest": manifests.get("selector_replay_manifest"),
        "product_runtime_drift_manifest": manifests.get("product_runtime_drift_manifest"),
        "release_efficiency_manifest": manifests.get("release_efficiency_manifest"),
        "external_evidence_baseline_comparison_report": manifests.get(
            "external_evidence_baseline_comparison_report"
        )
        or _nested(comparison, "external_evidence_baseline_comparison_gate", "report_path"),
        "pre_generation_probe_comparison_manifest": manifests.get(
            "pre_generation_probe_comparison_manifest"
        )
        or _nested(comparison, "pre_generation_probe_comparison_gate", "manifest_path"),
        "frontier_release_evidence_manifest": manifests.get("frontier_release_evidence_manifest")
        or _nested(comparison, "frontier_release_evidence_gate", "manifest_path"),
        "world_model_signal_workflow_manifest": manifests.get("world_model_signal_workflow_manifest")
        or _nested(comparison, "world_model_signal_workflow_gate", "manifest_path"),
        "pathway_intervention_workflow_manifest": manifests.get(
            "pathway_intervention_workflow_manifest"
        )
        or _nested(comparison, "pathway_intervention_workflow_gate", "manifest_path"),
        "product_trace_replay_workflow_manifest": manifests.get(
            "product_trace_replay_workflow_manifest"
        ),
        "product_trace_action_audit_gate_report": _nested(
            candidate,
            "product_trace_replay_workflow",
            "action_audit_gate_report_path",
        ),
        "product_trace_action_execution_gate_report": _nested(
            candidate,
            "product_trace_replay_workflow",
            "action_execution_gate_report_path",
        ),
        "selfcheck_signal_fusion_workflow_manifest": manifests.get(
            "selfcheck_signal_fusion_workflow_manifest"
        )
        or _nested(comparison, "selfcheck_signal_fusion_workflow_gate", "manifest_path"),
        "uncertainty_escalation_workflow_manifest": manifests.get(
            "uncertainty_escalation_workflow_manifest"
        )
        or _nested(comparison, "uncertainty_escalation_workflow_gate", "manifest_path"),
        "feedback_policy_workflow_manifest": manifests.get("feedback_policy_workflow_manifest"),
        "adapter_family_matrix_report": manifests.get("adapter_family_matrix_report"),
        "triple_extraction_fixture_matrix_report": manifests.get(
            "triple_extraction_fixture_matrix_report"
        ),
        "triple_extraction_fixture_matrix_manifest": manifests.get(
            "triple_extraction_fixture_matrix_manifest"
        ),
        "counterfactual_verification_report": manifests.get(
            "counterfactual_verification_report"
        ),
        "counterfactual_verification_manifest": manifests.get(
            "counterfactual_verification_manifest"
        ),
    }
    artifacts.update({
        str(name): path
        for name, path in manifests.items()
        if str(name).startswith("required_route_manifest_")
    })
    manifest = verification_context.build_artifact_manifest(
        artifacts,
        root=config.manifest_path.parent,
        metadata=_manifest_metadata(comparison),
        max_workers=config.manifest_fingerprint_workers,
    )
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _registry_workflow_decision(
    *,
    release_status: str,
    promotion: Mapping[str, Any] | None,
    blocking_reasons: Sequence[str],
) -> dict[str, Any]:
    verification = {} if promotion is None else dict(promotion.get("verification") or {})
    verified = bool(verification.get("passed", False))
    status = "promote" if release_status == "promote" and verified else "blocked"
    return {
        "status": status,
        "release_candidate_status": release_status,
        "manifest_promoted": promotion is not None,
        "manifest_verified": verified,
        "registry_record": None if promotion is None else dict(promotion.get("records") or {}).get(
            "benchmark_manifest"
        ),
        "blocking_reasons": tuple(blocking_reasons),
    }


def _manifest_metadata(comparison: Mapping[str, Any]) -> dict[str, Any]:
    decision = dict(comparison.get("decision") or {})
    config = dict(comparison.get("config") or {})
    candidate = dict(comparison.get("release_candidate") or {})
    runtime = dict(candidate.get("runtime") or {})
    quality = dict(candidate.get("quality") or {})
    best_quality = dict(quality.get("best_quality_signal") or {})
    readiness_covariance_gate = dict(quality.get("covariance_tradeoff_gate") or {})
    runtime_cost = dict(candidate.get("runtime_cost") or {})
    performance_evidence_bundle = dict(candidate.get("performance_evidence_bundle") or {})
    performance_evidence_recommendation = dict(
        performance_evidence_bundle.get("recommendation") or {}
    )
    performance_evidence_cost = dict(performance_evidence_bundle.get("cost") or {})
    performance_evidence = dict(performance_evidence_bundle.get("evidence") or {})
    performance_score_dump_cache = dict(performance_evidence_bundle.get("score_dump_cache") or {})
    performance_score_dump_cache_totals = dict(performance_score_dump_cache.get("totals") or {})
    performance_jsonl_view_cache = dict(performance_score_dump_cache_totals.get("jsonl_view") or {})
    performance_gate = dict(comparison.get("performance_baseline_gate") or {})
    performance_covariance_gate = dict(performance_gate.get("covariance_tradeoff_gate") or {})
    if not performance_covariance_gate:
        performance_covariance_gate = dict(
            dict(performance_gate.get("gate") or {}).get("covariance_tradeoff") or {}
        )
    performance_trend_gate = dict(performance_gate.get("performance_trend_gate") or {})
    performance_trend_metrics = dict(performance_trend_gate.get("metrics") or {})
    performance_uncached_trend = dict(performance_trend_metrics.get("uncached_total_seconds") or {})
    performance_cached_trend = dict(performance_trend_metrics.get("cached_total_seconds") or {})
    performance_cache_only_trend = dict(performance_trend_metrics.get("cache_only_total_seconds") or {})
    performance_cache_hit_rate_trend = dict(
        performance_trend_metrics.get("score_dump_cache_jsonl_view_hit_rate") or {}
    )
    verifier_route = dict(candidate.get("verifier_route") or {})
    manifests = dict(candidate.get("manifests") or {})
    adapter_family = dict(candidate.get("adapter_family_matrix") or {})
    triple_extraction_fixture_matrix = dict(candidate.get("triple_extraction_fixture_matrix") or {})
    if not triple_extraction_fixture_matrix:
        triple_extraction_fixture_matrix = dict(
            comparison.get("triple_extraction_fixture_matrix_gate") or {}
        )
    counterfactual_verification = dict(candidate.get("counterfactual_verification") or {})
    if not counterfactual_verification:
        counterfactual_verification = dict(
            comparison.get("counterfactual_verification_gate") or {}
        )
    required_route_baselines = dict(candidate.get("required_route_baselines") or {})
    required_route_property_counts = dict(
        required_route_baselines.get("covered_fact_property_counts") or {}
    )
    required_route_property_sets = dict(
        required_route_baselines.get("covered_fact_properties") or {}
    )
    structured_fact_robustness = _structured_fact_robustness_metadata(
        config,
        required_route_baselines,
    )
    selector_replay = dict(candidate.get("selector_replay") or {})
    selector_replay_recommended = dict(selector_replay.get("recommended") or {})
    product_runtime_drift = dict(candidate.get("product_runtime_drift") or {})
    product_runtime_drift_summary = dict(product_runtime_drift.get("summary") or {})
    product_runtime_drift_baseline = dict(product_runtime_drift.get("baseline") or {})
    product_runtime_drift_current = dict(product_runtime_drift.get("current") or {})
    release_efficiency = dict(candidate.get("release_efficiency") or {})
    release_efficiency_summary = dict(release_efficiency.get("summary") or {})
    release_efficiency_leaderboard = release_efficiency.get("leaderboard")
    release_efficiency_top = (
        dict(release_efficiency_leaderboard[0])
        if isinstance(release_efficiency_leaderboard, (list, tuple)) and release_efficiency_leaderboard
        else {}
    )
    external_evidence_baseline_comparison = dict(
        candidate.get("external_evidence_baseline_comparison") or {}
    )
    if not external_evidence_baseline_comparison:
        external_evidence_baseline_comparison = dict(
            comparison.get("external_evidence_baseline_comparison_gate") or {}
        )
    pre_generation_probe_comparison = dict(
        candidate.get("pre_generation_probe_comparison") or {}
    )
    if not pre_generation_probe_comparison:
        pre_generation_probe_comparison = dict(
            comparison.get("pre_generation_probe_comparison_gate") or {}
        )
    pre_generation_best_run = dict(pre_generation_probe_comparison.get("best_run") or {})
    product_trace_replay_workflow = dict(candidate.get("product_trace_replay_workflow") or {})
    product_trace_action_audit_gate = dict(
        product_trace_replay_workflow.get("action_audit_gate") or {}
    )
    product_trace_action_execution_gate = dict(
        product_trace_replay_workflow.get("action_execution_gate") or {}
    )
    selfcheck_signal_fusion_workflow = dict(
        candidate.get("selfcheck_signal_fusion_workflow") or {}
    )
    if not selfcheck_signal_fusion_workflow:
        selfcheck_signal_fusion_workflow = dict(
            comparison.get("selfcheck_signal_fusion_workflow_gate") or {}
        )
    uncertainty_escalation_workflow = dict(
        candidate.get("uncertainty_escalation_workflow") or {}
    )
    if not uncertainty_escalation_workflow:
        uncertainty_escalation_workflow = dict(
            comparison.get("uncertainty_escalation_workflow_gate") or {}
        )
    feedback_policy_workflow = dict(candidate.get("feedback_policy_workflow") or {})
    frontier_release_evidence = dict(candidate.get("frontier_release_evidence") or {})
    if not frontier_release_evidence:
        frontier_release_evidence = dict(comparison.get("frontier_release_evidence_gate") or {})
    world_model_signal_workflow = dict(candidate.get("world_model_signal_workflow") or {})
    if not world_model_signal_workflow:
        world_model_signal_workflow = dict(comparison.get("world_model_signal_workflow_gate") or {})
    pathway_intervention_workflow = dict(candidate.get("pathway_intervention_workflow") or {})
    if not pathway_intervention_workflow:
        pathway_intervention_workflow = dict(
            comparison.get("pathway_intervention_workflow_gate") or {}
        )
    return {
        "runner": "run_release_candidate_registry_workflow",
        "workflow": comparison.get("workflow"),
        "release_candidate_status": decision.get("status"),
        "release_readiness_status": decision.get("readiness_status"),
        "release_route_status": decision.get("route_status"),
        "release_performance_status": decision.get("performance_status"),
        "release_selector_replay_status": decision.get("selector_replay_status"),
        "release_product_runtime_drift_status": decision.get("product_runtime_drift_status"),
        "release_efficiency_status": decision.get("release_efficiency_status"),
        "release_external_evidence_baseline_comparison_status": decision.get(
            "external_evidence_baseline_comparison_status"
        ),
        "release_pre_generation_probe_comparison_status": decision.get(
            "pre_generation_probe_comparison_status"
        ),
        "release_frontier_release_evidence_status": decision.get(
            "frontier_release_evidence_status"
        ),
        "release_world_model_signal_workflow_status": decision.get(
            "world_model_signal_workflow_status"
        ),
        "release_pathway_intervention_workflow_status": decision.get(
            "pathway_intervention_workflow_status"
        ),
        "release_adapter_family_status": decision.get("adapter_family_status"),
        "release_triple_extraction_fixture_matrix_status": decision.get(
            "triple_extraction_fixture_matrix_status"
        ),
        "release_counterfactual_verification_status": decision.get(
            "counterfactual_verification_status"
        ),
        "release_required_route_baseline_status": decision.get("required_route_baseline_status"),
        "release_product_trace_replay_workflow_status": decision.get(
            "product_trace_replay_workflow_status"
        ),
        "release_selfcheck_signal_fusion_workflow_status": decision.get(
            "selfcheck_signal_fusion_workflow_status"
        ),
        "release_uncertainty_escalation_workflow_status": decision.get(
            "uncertainty_escalation_workflow_status"
        ),
        "release_feedback_policy_workflow_status": decision.get("feedback_policy_workflow_status"),
        "release_policy_profile": config.get("release_policy_profile"),
        "release_policy_profile_applied_defaults": config.get(
            "release_policy_profile_applied_defaults"
        ),
        "release_runtime_profile": config.get("runtime_profile"),
        "release_runtime_profile_defaults": config.get("runtime_profile_defaults"),
        "release_runtime_profile_applied_defaults": config.get("runtime_profile_applied_defaults"),
        "recommended_readiness_record": decision.get("recommended_readiness_record"),
        "recommended_route_record": decision.get("recommended_route_record"),
        "recommended_performance_baseline_record": decision.get("recommended_performance_baseline_record"),
        "recommended_selector_replay_candidate": decision.get("recommended_selector_replay_candidate"),
        "recommended_product_runtime_drift_report": decision.get("recommended_product_runtime_drift_report"),
        "recommended_release_efficiency_report": decision.get("recommended_release_efficiency_report"),
        "recommended_release_efficiency_profile": decision.get("recommended_release_efficiency_profile"),
        "recommended_external_evidence_baseline_comparison_report": decision.get(
            "recommended_external_evidence_baseline_comparison_report"
        ),
        "recommended_pre_generation_probe_comparison_report": decision.get(
            "recommended_pre_generation_probe_comparison_report"
        ),
        "recommended_frontier_release_evidence_report": decision.get(
            "recommended_frontier_release_evidence_report"
        ),
        "recommended_world_model_signal_workflow_report": decision.get(
            "recommended_world_model_signal_workflow_report"
        ),
        "recommended_pathway_intervention_workflow_report": decision.get(
            "recommended_pathway_intervention_workflow_report"
        ),
        "recommended_triple_extraction_fixture_matrix_report": decision.get(
            "recommended_triple_extraction_fixture_matrix_report"
        ),
        "recommended_counterfactual_verification_report": decision.get(
            "recommended_counterfactual_verification_report"
        ),
        "recommended_selfcheck_signal_fusion_workflow_report": decision.get(
            "recommended_selfcheck_signal_fusion_workflow_report"
        ),
        "recommended_uncertainty_escalation_workflow_report": decision.get(
            "recommended_uncertainty_escalation_workflow_report"
        ),
        "recommended_feedback_policy_workflow_report": decision.get(
            "recommended_feedback_policy_workflow_report"
        ),
        "recommended_feedback_policy_candidate_control_policy": decision.get(
            "recommended_feedback_policy_candidate_control_policy"
        ),
        "recommended_feedback_policy_candidate_control_defaults": decision.get(
            "recommended_feedback_policy_candidate_control_defaults"
        ),
        "required_adapter_routes": decision.get("required_adapter_routes"),
        "required_route_baseline_records": decision.get("required_route_baseline_records"),
        "recommended_model": decision.get("recommended_model"),
        "recommended_route": decision.get("recommended_route"),
        "recommended_route_covered_fact_property_count": (
            verifier_route.get("covered_fact_property_count")
        ),
        "recommended_route_covered_fact_properties": (
            verifier_route.get("covered_fact_properties")
        ),
        "recommended_layer": runtime.get("layer"),
        "recommended_batch_size": runtime.get("batch_size"),
        "recommended_hidden_state_capture": runtime.get("hidden_state_capture"),
        "recommended_covariance_mode": runtime.get("covariance_mode"),
        "recommended_covariance_low_rank": runtime.get("covariance_low_rank"),
        "recommended_max_batch_tokens": runtime.get("max_batch_tokens"),
        "recommended_prefix_kv_cache": runtime.get("prefix_kv_cache"),
        "recommended_max_workers": runtime.get("max_workers"),
        "recommended_best_quality_signal": best_quality.get("name"),
        "recommended_best_quality_auroc": best_quality.get("auroc"),
        "recommended_quality_signals": quality.get("quality_signals"),
        "performance_evidence_bundle_status": performance_evidence_bundle.get("status"),
        "performance_evidence_bundle_release_ready": performance_evidence_bundle.get(
            "release_ready"
        ),
        "performance_cache_tuning_status": performance_evidence_recommendation.get(
            "cache_tuning_status"
        ),
        "performance_score_ensemble_report": performance_evidence.get("score_ensemble_report"),
        "recommended_score_fusion_status": performance_evidence_recommendation.get(
            "score_fusion_status"
        ),
        "recommended_score_fusion_signal": performance_evidence_recommendation.get(
            "score_fusion_signal"
        ),
        "recommended_score_fusion_auroc": performance_evidence_recommendation.get(
            "score_fusion_auroc"
        ),
        "recommended_score_fusion_conformal_gate_passed": (
            performance_evidence_recommendation.get("score_fusion_conformal_gate_passed")
        ),
        "performance_selected_fusion_artifact_report": performance_evidence.get(
            "selected_fusion_artifact_report"
        ),
        "recommended_selected_fusion_status": performance_evidence_recommendation.get(
            "selected_fusion_status"
        ),
        "recommended_selected_fusion_run": performance_evidence_recommendation.get(
            "selected_fusion_run"
        ),
        "recommended_selected_fusion_candidate": performance_evidence_recommendation.get(
            "selected_fusion_candidate"
        ),
        "recommended_selected_fusion_signal": performance_evidence_recommendation.get(
            "selected_fusion_signal"
        ),
        "recommended_selected_fusion_auroc": performance_evidence_recommendation.get(
            "selected_fusion_auroc"
        ),
        "recommended_selected_fusion_artifact_path": performance_evidence_recommendation.get(
            "selected_fusion_artifact_path"
        ),
        "performance_uncached_total_seconds": performance_evidence_cost.get(
            "uncached_total_seconds"
        ),
        "performance_cached_total_ratio": performance_evidence_cost.get("cached_total_ratio"),
        "performance_cache_only_total_ratio": performance_evidence_cost.get(
            "cache_only_total_ratio"
        ),
        "performance_score_dump_cache_required": config.get("require_performance_score_dump_cache"),
        "performance_score_dump_cache_min_jsonl_view_hit_rate": (
            config.get("min_performance_score_dump_cache_jsonl_view_hit_rate")
        ),
        "performance_score_dump_cache_source_count": performance_score_dump_cache.get("source_count"),
        "performance_score_dump_cache_jsonl_view_hit_rate": performance_jsonl_view_cache.get("hit_rate"),
        "performance_drift_baseline_record": config.get("performance_drift_baseline_key"),
        "performance_trend_gate_passed": performance_trend_gate.get("passed"),
        "performance_trend_reference_record": performance_trend_gate.get("reference_record_key"),
        "performance_uncached_total_seconds_ratio_to_drift_baseline": (
            performance_uncached_trend.get("observed_ratio")
        ),
        "performance_cached_total_seconds_ratio_to_drift_baseline": (
            performance_cached_trend.get("observed_ratio")
        ),
        "performance_cache_only_total_seconds_ratio_to_drift_baseline": (
            performance_cache_only_trend.get("observed_ratio")
        ),
        "performance_score_dump_cache_jsonl_view_hit_rate_drop_from_drift_baseline": (
            performance_cache_hit_rate_trend.get("observed_drop")
        ),
        "max_covariance_maha_last_auroc_drop": config.get("max_covariance_maha_last_auroc_drop"),
        "readiness_covariance_tradeoff_gate_passed": readiness_covariance_gate.get("passed"),
        "readiness_covariance_tradeoff_status": readiness_covariance_gate.get("status"),
        "readiness_covariance_selected_mode": readiness_covariance_gate.get("selected_covariance_mode"),
        "readiness_covariance_selected_low_rank": (
            readiness_covariance_gate.get("selected_covariance_low_rank")
        ),
        "readiness_covariance_maha_last_delta_vs_baseline": (
            readiness_covariance_gate.get("selected_maha_last_delta_vs_baseline")
        ),
        "performance_covariance_tradeoff_gate_passed": performance_covariance_gate.get("passed"),
        "performance_covariance_tradeoff_status": performance_covariance_gate.get("status"),
        "performance_covariance_selected_mode": performance_covariance_gate.get(
            "selected_covariance_mode"
        ),
        "performance_covariance_selected_low_rank": (
            performance_covariance_gate.get("selected_covariance_low_rank")
        ),
        "performance_covariance_maha_last_delta_vs_baseline": (
            performance_covariance_gate.get("selected_maha_last_delta_vs_baseline")
        ),
        "recommended_uncached_forward_cost_seconds": runtime_cost.get("uncached_forward_cost_seconds"),
        "recommended_uncached_forward_cost_source": runtime_cost.get("uncached_forward_cost_source"),
        "recommended_cache_only_total_seconds": runtime_cost.get("cache_only_total_seconds"),
        "recommended_inside_sampling_run": runtime_cost.get("inside_sampling_recommended_run"),
        "recommended_inside_sampling_total_generated_samples": runtime_cost.get(
            "inside_sampling_total_generated_samples"
        ),
        "recommended_inside_sampling_sample_count_ratio_to_baseline": runtime_cost.get(
            "inside_sampling_sample_count_ratio_to_baseline"
        ),
        "recommended_inside_sampling_sample_count_ratio_to_reference": runtime_cost.get(
            "inside_sampling_sample_count_ratio_to_reference"
        ),
        "recommended_inside_sampling_sample_count_ratio_for_gate": runtime_cost.get(
            "inside_sampling_sample_count_ratio_for_gate"
        ),
        "recommended_inside_sampling_sample_count_ratio_source": runtime_cost.get(
            "inside_sampling_sample_count_ratio_source"
        ),
        "recommended_inside_generation_seconds": runtime_cost.get("inside_generation_seconds"),
        "recommended_inside_generation_seconds_ratio_to_baseline": runtime_cost.get(
            "inside_generation_seconds_ratio_to_baseline"
        ),
        "recommended_inside_generation_seconds_ratio_to_reference": runtime_cost.get(
            "inside_generation_seconds_ratio_to_reference"
        ),
        "recommended_inside_generation_seconds_ratio_for_gate": runtime_cost.get(
            "inside_generation_seconds_ratio_for_gate"
        ),
        "recommended_inside_generation_seconds_ratio_source": runtime_cost.get(
            "inside_generation_seconds_ratio_source"
        ),
        "recommended_inside_sampling_stop_reason_counts": runtime_cost.get("inside_sampling_stop_reason_counts"),
        "recommended_inside_trigger_budget_id": runtime_cost.get("inside_trigger_budget_id"),
        "recommended_inside_trigger_budget_policy": runtime_cost.get("inside_trigger_budget_policy"),
        "recommended_inside_trigger_budget_derive_from_max_budget": runtime_cost.get(
            "inside_trigger_budget_derive_from_max_budget"
        ),
        "recommended_route_selected": verifier_route.get("selected"),
        "recommended_route_decision_accuracy": verifier_route.get("decision_accuracy"),
        "recommended_route_false_supported_rate": verifier_route.get("false_supported_rate"),
        "recommended_route_false_refuted_rate": verifier_route.get("false_refuted_rate"),
        "recommended_route_verified_false_alarm": verifier_route.get("verified_false_alarm"),
        "recommended_route_verified_detection": verifier_route.get("verified_detection"),
        "recommended_route_mean_duration_seconds": verifier_route.get("mean_duration_seconds"),
        "recommended_route_p99_duration_seconds": verifier_route.get("p99_duration_seconds"),
        "recommended_route_max_duration_seconds": verifier_route.get("max_duration_seconds"),
        "recommended_route_mean_attempted_route_count": verifier_route.get("mean_attempted_route_count"),
        "recommended_route_retrieval_use_rate": verifier_route.get("retrieval_use_rate"),
        "recommended_route_runtime_total_seconds": verifier_route.get("runtime_total_seconds"),
        "recommended_route_runtime_retrieval_hit_count": verifier_route.get("runtime_retrieval_hit_count"),
        "recommended_route_claims_cache_hit_rate": verifier_route.get("claims_cache_hit_rate"),
        "recommended_route_verifier_trace_cache_hit_rate": verifier_route.get("verifier_trace_cache_hit_rate"),
        "release_route_require_non_oracle_evidence": config.get("require_non_oracle_evidence"),
        "release_route_require_retrieval_provenance_filter": config.get(
            "require_retrieval_provenance_filter"
        ),
        "release_route_required_retrieval_source_prefixes": config.get(
            "required_retrieval_source_prefixes"
        ),
        "release_route_required_retrieval_metadata": config.get("required_retrieval_metadata"),
        "release_route_min_retrieval_filter_score": config.get("min_retrieval_filter_score"),
        "release_route_require_retrieval_stress_control": config.get(
            "require_retrieval_stress_control"
        ),
        "release_route_retrieval_stress_manifest": config.get("retrieval_stress_manifest"),
        "release_route_min_stress_false_supported_rate": config.get(
            "min_stress_false_supported_rate"
        ),
        "release_route_max_stress_false_refuted_rate": config.get(
            "max_stress_false_refuted_rate"
        ),
        "structured_fact_robustness_required": structured_fact_robustness["required"],
        "structured_fact_robustness_canonical_route_key": (
            structured_fact_robustness["canonical_route_key"]
        ),
        "structured_fact_robustness_paraphrase_route_key": (
            structured_fact_robustness["paraphrase_route_key"]
        ),
        "structured_fact_robustness_records": structured_fact_robustness["records"],
        "structured_fact_robustness_routes": structured_fact_robustness["routes"],
        "structured_fact_robustness_manifests": structured_fact_robustness["manifests"],
        "structured_fact_robustness_property_counts": (
            structured_fact_robustness["property_counts"]
        ),
        "structured_fact_robustness_properties": structured_fact_robustness["properties"],
        "selector_replay_report": selector_replay.get("report_path"),
        "selector_replay_recommended_policy_path": selector_replay.get("recommended_policy_path"),
        "selector_replay_estimated_cost_units_mean": selector_replay_recommended.get(
            "estimated_cost_units_mean"
        ),
        "selector_replay_observed_runtime_coverage_rate": selector_replay_recommended.get(
            "observed_runtime_coverage_rate"
        ),
        "selector_replay_observed_runtime_delta_coverage_rate": selector_replay_recommended.get(
            "observed_runtime_delta_coverage_rate"
        ),
        "selector_replay_observed_selected_total_seconds_mean": selector_replay_recommended.get(
            "observed_selected_total_seconds_mean"
        ),
        "selector_replay_observed_selected_minus_original_seconds_mean": (
            selector_replay_recommended.get("observed_selected_minus_original_seconds_mean")
        ),
        "selector_replay_observed_selected_to_original_ratio_mean": selector_replay_recommended.get(
            "observed_selected_to_original_ratio_mean"
        ),
        "product_runtime_drift_report": product_runtime_drift.get("report_path"),
        "product_runtime_drift_baseline_path": product_runtime_drift_baseline.get("path"),
        "product_runtime_drift_current_path": product_runtime_drift_current.get("path"),
        "product_runtime_drift_gate_enabled": product_runtime_drift_summary.get("gate_enabled"),
        "product_runtime_drift_promotion_evidence_required": config.get(
            "require_product_runtime_drift_promotion_evidence"
        ),
        "product_runtime_drift_pre_generation_evidence_required": config.get(
            "require_product_runtime_drift_pre_generation_evidence"
        ),
        "product_runtime_drift_counterfactual_evidence_required": config.get(
            "require_product_runtime_drift_counterfactual_evidence"
        ),
        "product_runtime_drift_triple_audit_evidence_required": config.get(
            "require_product_runtime_drift_triple_audit_evidence"
        ),
        "product_runtime_drift_covered_fact_property_evidence_required": config.get(
            "require_product_runtime_drift_covered_fact_property_evidence"
        ),
        "product_runtime_drift_action_gate_evidence_required": config.get(
            "require_product_runtime_drift_action_gate_evidence"
        ),
        "product_runtime_drift_compared_metric_count": product_runtime_drift_summary.get(
            "compared_metric_count"
        ),
        "product_runtime_drift_blocked_metric_count": product_runtime_drift_summary.get(
            "blocked_metric_count"
        ),
        **_product_runtime_drift_promotion_metadata(product_runtime_drift_summary),
        "release_efficiency_report": release_efficiency.get("report_path"),
        "release_efficiency_recommended_profile": release_efficiency.get("recommended_profile"),
        "release_efficiency_score": release_efficiency.get("recommended_efficiency_score"),
        "release_efficiency_profile_count": release_efficiency_summary.get("profile_count"),
        "release_efficiency_quality_passed": release_efficiency_summary.get("quality_passed"),
        "release_efficiency_trace_record_cache_hit_profile_count": (
            release_efficiency_summary.get("trace_record_cache_hit_profile_count")
        ),
        "release_efficiency_leaderboard_top_profile": release_efficiency_top.get("profile"),
        "external_evidence_baseline_comparison_report": (
            external_evidence_baseline_comparison.get("report_path")
        ),
        "external_evidence_baseline_comparison_source": (
            external_evidence_baseline_comparison.get("source")
        ),
        "external_evidence_baseline_comparison_registry": (
            external_evidence_baseline_comparison.get("registry")
        ),
        "external_evidence_baseline_comparison_record": (
            external_evidence_baseline_comparison.get("record_key")
        ),
        "external_evidence_baseline_comparison_decision_status": (
            external_evidence_baseline_comparison.get("decision_status")
        ),
        "external_evidence_baseline_comparison_recommended_route": (
            external_evidence_baseline_comparison.get("recommended_route")
        ),
        "external_evidence_baseline_comparison_recommended_route_record": (
            external_evidence_baseline_comparison.get("recommended_route_record")
        ),
        "external_evidence_baseline_comparison_route_passed": (
            external_evidence_baseline_comparison.get("route_passed")
        ),
        "external_evidence_baseline_comparison_text_redline_passed": (
            external_evidence_baseline_comparison.get("text_redline_passed")
        ),
        "external_evidence_baseline_comparison_text_redline_run_count": (
            external_evidence_baseline_comparison.get("text_redline_run_count")
        ),
        "pre_generation_probe_comparison_report": (
            pre_generation_probe_comparison.get("report_path")
        ),
        "pre_generation_probe_comparison_manifest": (
            manifests.get("pre_generation_probe_comparison_manifest")
            or pre_generation_probe_comparison.get("manifest_path")
        ),
        "pre_generation_probe_comparison_source": (
            pre_generation_probe_comparison.get("source")
        ),
        "pre_generation_probe_comparison_registry": (
            pre_generation_probe_comparison.get("registry")
        ),
        "pre_generation_probe_comparison_record": (
            pre_generation_probe_comparison.get("record_key")
        ),
        "pre_generation_probe_comparison_model_count": (
            pre_generation_probe_comparison.get("model_count")
        ),
        "pre_generation_probe_comparison_run_count": (
            pre_generation_probe_comparison.get("run_count")
        ),
        "pre_generation_probe_comparison_redline_passed": (
            pre_generation_probe_comparison.get("redline_passed")
        ),
        "pre_generation_probe_comparison_redline_run_count": (
            pre_generation_probe_comparison.get("redline_run_count")
        ),
        "pre_generation_probe_comparison_best_run": pre_generation_best_run.get("name"),
        "pre_generation_probe_comparison_best_model": pre_generation_best_run.get("model"),
        "pre_generation_probe_comparison_best_layer": (
            pre_generation_best_run.get("recommended_layer")
        ),
        "pre_generation_probe_comparison_best_test_label_auroc": (
            pre_generation_best_run.get("test_label_auroc")
        ),
        "pre_generation_probe_comparison_best_redline_signal": (
            pre_generation_best_run.get("redline_best_signal")
        ),
        "pre_generation_probe_comparison_best_redline_auroc": (
            pre_generation_best_run.get("redline_best_auroc")
        ),
        "pre_generation_probe_comparison_best_redline_margin": (
            pre_generation_best_run.get("redline_margin")
        ),
        "product_trace_replay_workflow_report": product_trace_replay_workflow.get("report_path"),
        "product_trace_replay_workflow_source": product_trace_replay_workflow.get("source"),
        "product_trace_replay_workflow_registry": product_trace_replay_workflow.get("registry"),
        "product_trace_replay_workflow_record": product_trace_replay_workflow.get("record_key"),
        "product_trace_replay_workflow_selector_replay_report": product_trace_replay_workflow.get(
            "selector_replay_report_path"
        ),
        "product_trace_replay_workflow_runtime_drift_report": product_trace_replay_workflow.get(
            "product_runtime_drift_report_path"
        ),
        "product_trace_action_audit_gate_required": config.get(
            "require_product_trace_action_audit_gate"
        ),
        "product_trace_action_audit_gate_status": product_trace_action_audit_gate.get("status"),
        "product_trace_action_audit_gate_enabled": product_trace_action_audit_gate.get(
            "gate_enabled"
        ),
        "product_trace_action_audit_gate_passed": product_trace_action_audit_gate.get("passed"),
        "product_trace_action_audit_gate_report": product_trace_replay_workflow.get(
            "action_audit_gate_report_path"
        ),
        "product_trace_action_audit_error_rate": product_trace_action_audit_gate.get(
            "error_rate"
        ),
        "product_trace_action_audit_missing_retrieval_action_rate": (
            product_trace_action_audit_gate.get("missing_retrieval_action_rate")
        ),
        "product_trace_action_audit_missing_plan_retrieval_query_rate": (
            product_trace_action_audit_gate.get("missing_plan_retrieval_query_rate")
        ),
        "product_trace_action_audit_malformed_payload_rate": product_trace_action_audit_gate.get(
            "malformed_payload_rate"
        ),
        "product_trace_action_audit_unexpected_action_rate": product_trace_action_audit_gate.get(
            "unexpected_action_rate"
        ),
        "product_trace_action_audit_unknown_claim_id_rate": product_trace_action_audit_gate.get(
            "unknown_claim_id_rate"
        ),
        "product_trace_action_execution_gate_required": config.get(
            "require_product_trace_action_execution_gate"
        ),
        "product_trace_action_execution_gate_status": product_trace_action_execution_gate.get(
            "status"
        ),
        "product_trace_action_execution_gate_enabled": product_trace_action_execution_gate.get(
            "gate_enabled"
        ),
        "product_trace_action_execution_gate_passed": product_trace_action_execution_gate.get(
            "passed"
        ),
        "product_trace_action_execution_gate_report": product_trace_replay_workflow.get(
            "action_execution_gate_report_path"
        ),
        "product_trace_action_execution_alignment_failed_trace_rate": (
            product_trace_action_execution_gate.get("alignment_failed_trace_rate")
        ),
        "product_trace_action_execution_missing_result_rate": (
            product_trace_action_execution_gate.get("missing_result_rate")
        ),
        "product_trace_action_execution_unexpected_result_rate": (
            product_trace_action_execution_gate.get("unexpected_result_rate")
        ),
        "product_trace_action_execution_request_id_mismatch_rate": (
            product_trace_action_execution_gate.get("request_id_mismatch_rate")
        ),
        "selfcheck_signal_fusion_workflow_report": selfcheck_signal_fusion_workflow.get("report_path"),
        "selfcheck_signal_fusion_workflow_manifest": (
            manifests.get("selfcheck_signal_fusion_workflow_manifest")
            or selfcheck_signal_fusion_workflow.get("manifest_path")
        ),
        "selfcheck_signal_fusion_workflow_source": selfcheck_signal_fusion_workflow.get("source"),
        "selfcheck_signal_fusion_workflow_registry": selfcheck_signal_fusion_workflow.get("registry"),
        "selfcheck_signal_fusion_workflow_record": selfcheck_signal_fusion_workflow.get("record_key"),
        "selfcheck_signal_fusion_workflow_sample_quality_status": (
            selfcheck_signal_fusion_workflow.get("sample_quality_status")
        ),
        "selfcheck_signal_fusion_workflow_sample_quality_passed": (
            selfcheck_signal_fusion_workflow.get("sample_quality_passed")
        ),
        "selfcheck_signal_fusion_workflow_failed_runs": (
            selfcheck_signal_fusion_workflow.get("sample_quality_failed_runs")
        ),
        "selfcheck_signal_fusion_workflow_sample_quality_run_count": (
            selfcheck_signal_fusion_workflow.get("sample_quality_run_count")
        ),
        "selfcheck_signal_fusion_workflow_fusion_run_count": (
            selfcheck_signal_fusion_workflow.get("fusion_run_count")
        ),
        "selfcheck_signal_fusion_workflow_geometry_artifact_count": (
            selfcheck_signal_fusion_workflow.get("geometry_fusion_artifact_count")
        ),
        "selfcheck_signal_fusion_workflow_enhanced_score_dump_count": (
            selfcheck_signal_fusion_workflow.get("enhanced_score_dump_count")
        ),
        "uncertainty_escalation_workflow_report": (
            uncertainty_escalation_workflow.get("report_path")
        ),
        "uncertainty_escalation_workflow_manifest": (
            manifests.get("uncertainty_escalation_workflow_manifest")
            or uncertainty_escalation_workflow.get("manifest_path")
        ),
        "uncertainty_escalation_workflow_source": (
            uncertainty_escalation_workflow.get("source")
        ),
        "uncertainty_escalation_workflow_registry": (
            uncertainty_escalation_workflow.get("registry")
        ),
        "uncertainty_escalation_workflow_record": (
            uncertainty_escalation_workflow.get("record_key")
        ),
        "uncertainty_escalation_workflow_record_count": (
            uncertainty_escalation_workflow.get("record_count")
        ),
        "uncertainty_escalation_workflow_trigger_rate": (
            uncertainty_escalation_workflow.get("trigger_rate")
        ),
        "uncertainty_escalation_workflow_retrieval_evidence_rate": (
            uncertainty_escalation_workflow.get("retrieval_evidence_rate")
        ),
        "uncertainty_escalation_workflow_final_false_accept_rate": (
            uncertainty_escalation_workflow.get("final_false_accept_rate")
        ),
        "uncertainty_escalation_workflow_false_accept_delta": (
            uncertainty_escalation_workflow.get("false_accept_delta")
        ),
        "uncertainty_escalation_workflow_accepted_false_delta": (
            uncertainty_escalation_workflow.get("accepted_false_delta")
        ),
        "uncertainty_escalation_min_records": config.get(
            "min_uncertainty_escalation_records"
        ),
        "uncertainty_escalation_min_trigger_rate": config.get(
            "min_uncertainty_escalation_trigger_rate"
        ),
        "uncertainty_escalation_min_retrieval_evidence_rate": config.get(
            "min_uncertainty_escalation_retrieval_evidence_rate"
        ),
        "uncertainty_escalation_max_final_false_accept_rate": config.get(
            "max_uncertainty_escalation_final_false_accept_rate"
        ),
        "uncertainty_escalation_max_false_accept_delta": config.get(
            "max_uncertainty_escalation_false_accept_delta"
        ),
        "feedback_policy_workflow_report": feedback_policy_workflow.get("report_path"),
        "feedback_policy_workflow_source": feedback_policy_workflow.get("source"),
        "feedback_policy_workflow_registry": feedback_policy_workflow.get("registry"),
        "feedback_policy_workflow_record": feedback_policy_workflow.get("record_key"),
        "feedback_policy_workflow_promotion_decision": feedback_policy_workflow.get(
            "promotion_decision"
        ),
        "feedback_policy_workflow_candidate_control_policy": feedback_policy_workflow.get(
            "candidate_control_policy"
        ),
        "feedback_policy_workflow_candidate_control_defaults": feedback_policy_workflow.get(
            "candidate_control_defaults"
        ),
        "feedback_policy_workflow_matched_feedback_count": feedback_policy_workflow.get(
            "matched_feedback_count"
        ),
        "feedback_policy_workflow_accepted_but_wrong_rate": feedback_policy_workflow.get(
            "accepted_but_wrong_rate"
        ),
        "feedback_policy_workflow_retrieved_failure_rate": feedback_policy_workflow.get(
            "retrieved_failure_rate"
        ),
        "feedback_policy_workflow_abstain_false_positive_rate": feedback_policy_workflow.get(
            "abstain_false_positive_rate"
        ),
        "feedback_policy_workflow_final_answered_but_wrong_rate": feedback_policy_workflow.get(
            "final_answered_but_wrong_rate"
        ),
        "feedback_policy_workflow_final_answer_false_block_rate": feedback_policy_workflow.get(
            "final_answer_false_block_rate"
        ),
        "feedback_policy_workflow_safety_coverage_rate": feedback_policy_workflow.get(
            "safety_coverage_rate"
        ),
        "feedback_policy_workflow_unknown_safety_issue_rate": feedback_policy_workflow.get(
            "unknown_safety_issue_rate"
        ),
        "adapter_family_matrix_report": adapter_family.get("matrix_path"),
        "adapter_family_profile": config.get("adapter_family_profile"),
        "adapter_family_profile_requires_state_transition_world_model": config.get(
            "adapter_family_profile_requires_state_transition_world_model"
        ),
        "adapter_family_profile_required_routes": config.get("adapter_family_profile_required_routes"),
        "adapter_family_routes": adapter_family.get("routes"),
        "adapter_family_retrieval_routes": adapter_family.get("retrieval_routes"),
        "adapter_family_audit_routes": adapter_family.get("audit_routes"),
        "adapter_family_promoted_routes": adapter_family.get("promoted_routes"),
        "adapter_family_required_routes": adapter_family.get("required_routes"),
        "adapter_family_require_state_transition_world_model": adapter_family.get(
            "require_state_transition_world_model"
        ),
        "adapter_family_state_transition_world_model_adapter": adapter_family.get(
            "state_transition_world_model_adapter"
        ),
        "adapter_family_state_transition_world_model_rule_count": adapter_family.get(
            "state_transition_world_model_rule_count"
        ),
        "triple_extraction_fixture_matrix_report": triple_extraction_fixture_matrix.get("report_path"),
        "triple_extraction_fixture_matrix_manifest": (
            manifests.get("triple_extraction_fixture_matrix_manifest")
            or triple_extraction_fixture_matrix.get("manifest_path")
        ),
        "triple_extraction_fixture_matrix_source": triple_extraction_fixture_matrix.get("source"),
        "triple_extraction_fixture_matrix_registry": triple_extraction_fixture_matrix.get("registry"),
        "triple_extraction_fixture_matrix_record": triple_extraction_fixture_matrix.get("record_key"),
        "triple_extraction_fixture_matrix_status": triple_extraction_fixture_matrix.get("status")
        or triple_extraction_fixture_matrix.get("report_status"),
        "triple_extraction_fixture_matrix_n_corpora": (
            triple_extraction_fixture_matrix.get("n_corpora")
        ),
        "triple_extraction_fixture_matrix_promoted_corpora": (
            triple_extraction_fixture_matrix.get("promoted_corpora")
        ),
        "triple_extraction_fixture_matrix_distinct_predicate_count": (
            triple_extraction_fixture_matrix.get("distinct_predicate_count")
        ),
        "triple_extraction_fixture_matrix_distinct_predicates": (
            triple_extraction_fixture_matrix.get("distinct_predicates")
        ),
        "triple_extraction_fixture_matrix_mean_best_f1": (
            triple_extraction_fixture_matrix.get("mean_best_f1")
        ),
        "triple_extraction_fixture_matrix_mean_f1_lift": (
            triple_extraction_fixture_matrix.get("mean_f1_lift")
        ),
        "triple_extraction_fixture_matrix_external_prediction_count": (
            triple_extraction_fixture_matrix.get("external_prediction_count")
        ),
        "triple_extraction_fixture_matrix_external_prediction_corpora": (
            triple_extraction_fixture_matrix.get("external_prediction_corpora")
        ),
        "triple_extraction_fixture_matrix_mean_best_external_f1": (
            triple_extraction_fixture_matrix.get("mean_best_external_f1")
        ),
        "triple_extraction_fixture_matrix_min_corpora": config.get(
            "min_triple_extraction_corpora"
        ),
        "triple_extraction_fixture_matrix_min_distinct_predicates": config.get(
            "min_triple_extraction_distinct_predicates"
        ),
        "triple_extraction_fixture_matrix_min_external_prediction_count": config.get(
            "min_triple_extraction_external_prediction_count"
        ),
        "triple_extraction_fixture_matrix_min_external_prediction_corpora": config.get(
            "min_triple_extraction_external_prediction_corpora"
        ),
        "triple_extraction_fixture_matrix_min_mean_best_external_f1": config.get(
            "min_triple_extraction_mean_best_external_f1"
        ),
        "counterfactual_verification_report": counterfactual_verification.get("report_path"),
        "counterfactual_verification_manifest": (
            manifests.get("counterfactual_verification_manifest")
            or counterfactual_verification.get("manifest_path")
        ),
        "counterfactual_verification_source": counterfactual_verification.get("source"),
        "counterfactual_verification_registry": counterfactual_verification.get("registry"),
        "counterfactual_verification_record": counterfactual_verification.get("record_key"),
        "counterfactual_verification_status": counterfactual_verification.get("status")
        or counterfactual_verification.get("report_status"),
        "counterfactual_verification_record_count": (
            counterfactual_verification.get("record_count")
        ),
        "counterfactual_verification_pass_rate": counterfactual_verification.get("pass_rate"),
        "counterfactual_verification_false_invariance_rate": (
            counterfactual_verification.get("false_invariance_rate")
        ),
        "counterfactual_verification_flip_success_count": (
            counterfactual_verification.get("flip_success_count")
        ),
        "counterfactual_verification_min_records": config.get(
            "min_counterfactual_verification_records"
        ),
        "counterfactual_verification_min_pass_rate": config.get(
            "min_counterfactual_verification_pass_rate"
        ),
        "counterfactual_verification_max_false_invariance_rate": config.get(
            "max_counterfactual_verification_false_invariance_rate"
        ),
        "required_route_baseline_registry": required_route_baselines.get("registry"),
        "required_route_baseline_routes": required_route_baselines.get("routes"),
        "required_route_baseline_manifests": required_route_baselines.get("manifest_paths"),
        "required_route_baseline_covered_fact_property_counts": required_route_property_counts,
        "required_route_baseline_covered_fact_properties": required_route_property_sets,
        "required_route_budget_policy": {
            key: config.get(key)
            for key in (
                "required_route_min_selected",
                "required_route_min_decision_accuracy",
                "required_route_max_false_supported_rate",
                "required_route_min_false_refuted_rate",
                "required_route_max_verified_false_alarm",
                "required_route_min_verified_detection",
                "required_route_max_mean_duration_seconds",
                "required_route_max_p99_duration_seconds",
                "required_route_max_max_duration_seconds",
                "required_route_max_mean_attempted_route_count",
                "required_route_max_retrieval_use_rate",
                "required_route_max_runtime_total_seconds",
                "required_route_max_retrieval_hit_count",
                "required_route_min_claims_cache_hit_rate",
                "required_route_min_verifier_trace_cache_hit_rate",
                "required_route_min_covered_fact_properties",
                "required_route_min_covered_fact_property_records",
                "required_route_min_covered_fact_property_source_documents",
                "required_route_min_covered_fact_property_decision_accuracy",
                "required_route_max_covered_fact_property_false_supported_rate",
                "required_route_min_covered_fact_property_false_refuted_rate",
                "required_route_require_non_oracle_evidence",
                "required_route_require_retrieval_provenance_filter",
                "required_route_required_retrieval_source_prefixes",
                "required_route_required_retrieval_metadata",
                "required_route_min_retrieval_filter_score",
                "required_route_require_retrieval_stress_control",
                "required_route_retrieval_stress_manifest",
                "required_route_min_stress_false_supported_rate",
                "required_route_max_stress_false_refuted_rate",
            )
        },
        "readiness_manifest": manifests.get("readiness_manifest"),
        "route_manifest": manifests.get("route_manifest"),
        "performance_manifest": manifests.get("performance_manifest"),
        "selector_replay_manifest": manifests.get("selector_replay_manifest"),
        "product_runtime_drift_manifest": manifests.get("product_runtime_drift_manifest"),
        "release_efficiency_manifest": manifests.get("release_efficiency_manifest"),
        "frontier_release_evidence_manifest": manifests.get("frontier_release_evidence_manifest")
        or frontier_release_evidence.get("manifest_path"),
        "frontier_release_evidence_report": frontier_release_evidence.get("report_path"),
        "frontier_release_evidence_decision_status": frontier_release_evidence.get(
            "decision_status"
        ),
        "frontier_release_evidence_verifier_track_status": frontier_release_evidence.get(
            "verifier_track_status"
        ),
        "frontier_release_evidence_abstention_track_status": frontier_release_evidence.get(
            "abstention_track_status"
        ),
        "world_model_signal_workflow_report": world_model_signal_workflow.get("report_path"),
        "world_model_signal_workflow_manifest": (
            manifests.get("world_model_signal_workflow_manifest")
            or world_model_signal_workflow.get("manifest_path")
        ),
        "world_model_signal_workflow_source": world_model_signal_workflow.get("source"),
        "world_model_signal_workflow_registry": world_model_signal_workflow.get("registry"),
        "world_model_signal_workflow_record": world_model_signal_workflow.get("record_key"),
        "world_model_signal_workflow_release_gate_status": (
            world_model_signal_workflow.get("release_gate_status")
        ),
        "world_model_signal_workflow_trace_gap_max": (
            world_model_signal_workflow.get("trace_gap_max")
        ),
        "world_model_signal_workflow_conflict_positive_count": (
            world_model_signal_workflow.get("conflict_positive_count")
        ),
        "world_model_signal_workflow_calibrated_conflict_signal_count": (
            world_model_signal_workflow.get("calibrated_conflict_signal_count")
        ),
        "pathway_intervention_workflow_report": pathway_intervention_workflow.get("report_path"),
        "pathway_intervention_workflow_manifest": (
            manifests.get("pathway_intervention_workflow_manifest")
            or pathway_intervention_workflow.get("manifest_path")
        ),
        "pathway_intervention_workflow_source": pathway_intervention_workflow.get("source"),
        "pathway_intervention_workflow_registry": pathway_intervention_workflow.get("registry"),
        "pathway_intervention_workflow_record": pathway_intervention_workflow.get("record_key"),
        "pathway_intervention_workflow_release_ready": (
            pathway_intervention_workflow.get("release_ready")
        ),
        "pathway_intervention_workflow_model": pathway_intervention_workflow.get("model"),
        "pathway_intervention_workflow_layer": pathway_intervention_workflow.get("layer"),
        "pathway_intervention_workflow_activation_ablation_gate": (
            pathway_intervention_workflow.get("activation_ablation_gate_status")
        ),
        "pathway_intervention_workflow_source_patch_gate": (
            pathway_intervention_workflow.get("source_patch_gate_status")
        ),
        "product_trace_replay_workflow_manifest": manifests.get(
            "product_trace_replay_workflow_manifest"
        ),
        "feedback_policy_workflow_manifest": manifests.get("feedback_policy_workflow_manifest"),
        "adapter_family_matrix_manifest": manifests.get("adapter_family_matrix_report"),
    }


def _product_runtime_drift_promotion_metadata(summary: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        "product_runtime_drift_promotion_evidence_metric_count": summary.get(
            "promotion_evidence_metric_count"
        ),
        "product_runtime_drift_promotion_evidence_blocked_metric_count": summary.get(
            "promotion_evidence_blocked_metric_count"
        ),
        "product_runtime_drift_pre_generation_evidence_metric_count": summary.get(
            "pre_generation_evidence_metric_count"
        ),
        "product_runtime_drift_pre_generation_evidence_blocked_metric_count": summary.get(
            "pre_generation_evidence_blocked_metric_count"
        ),
        "product_runtime_drift_counterfactual_evidence_metric_count": summary.get(
            "counterfactual_evidence_metric_count"
        ),
        "product_runtime_drift_counterfactual_evidence_blocked_metric_count": summary.get(
            "counterfactual_evidence_blocked_metric_count"
        ),
        "product_runtime_drift_triple_audit_evidence_metric_count": summary.get(
            "triple_audit_evidence_metric_count"
        ),
        "product_runtime_drift_triple_audit_evidence_blocked_metric_count": summary.get(
            "triple_audit_evidence_blocked_metric_count"
        ),
        "product_runtime_drift_covered_fact_property_evidence_metric_count": summary.get(
            "covered_fact_property_evidence_metric_count"
        ),
        "product_runtime_drift_covered_fact_property_evidence_blocked_metric_count": summary.get(
            "covered_fact_property_evidence_blocked_metric_count"
        ),
        "product_runtime_drift_action_gate_evidence_metric_count": summary.get(
            "action_gate_evidence_metric_count"
        ),
        "product_runtime_drift_action_gate_evidence_blocked_metric_count": summary.get(
            "action_gate_evidence_blocked_metric_count"
        ),
    }
    for prefix in _PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_PREFIXES:
        for suffix in ("baseline", "current", "status"):
            metadata[f"product_runtime_drift_{prefix}_{suffix}"] = summary.get(f"{prefix}_{suffix}")
    for prefix in _PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_PREFIXES:
        for suffix in ("baseline", "current", "status"):
            metadata[f"product_runtime_drift_{prefix}_{suffix}"] = summary.get(f"{prefix}_{suffix}")
    for prefix in _PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_EVIDENCE_PREFIXES:
        for suffix in ("baseline", "current", "status"):
            metadata[f"product_runtime_drift_{prefix}_{suffix}"] = summary.get(f"{prefix}_{suffix}")
    for prefix in _PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_PREFIXES:
        for suffix in ("baseline", "current", "status"):
            metadata[f"product_runtime_drift_{prefix}_{suffix}"] = summary.get(f"{prefix}_{suffix}")
    for prefix in _PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_PREFIXES:
        for suffix in ("baseline", "current", "status"):
            metadata[f"product_runtime_drift_{prefix}_{suffix}"] = summary.get(f"{prefix}_{suffix}")
    for prefix in _PRODUCT_RUNTIME_DRIFT_ACTION_GATE_EVIDENCE_PREFIXES:
        for suffix in ("baseline", "current", "status"):
            metadata[f"product_runtime_drift_{prefix}_{suffix}"] = summary.get(f"{prefix}_{suffix}")
    return metadata


def _structured_fact_robustness_metadata(
    config: Mapping[str, Any],
    required_route_baselines: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_key = config.get("structured_fact_canonical_route_key")
    paraphrase_key = config.get("structured_fact_paraphrase_route_key")
    target_keys = tuple(
        str(key)
        for key in (canonical_key, paraphrase_key)
        if key is not None
    )
    records = list(required_route_baselines.get("records") or ())
    routes = list(required_route_baselines.get("routes") or ())
    manifests = list(required_route_baselines.get("manifest_paths") or ())
    property_counts_by_record = dict(required_route_baselines.get("covered_fact_property_counts") or {})
    properties_by_record = dict(required_route_baselines.get("covered_fact_properties") or {})
    by_record: dict[str, dict[str, Any]] = {}
    for idx, record in enumerate(records):
        key = str(record)
        by_record[key] = {
            "route": routes[idx] if idx < len(routes) else None,
            "manifest": manifests[idx] if idx < len(manifests) else None,
            "property_count": property_counts_by_record.get(key),
            "properties": properties_by_record.get(key),
        }
    selected_records: list[str] = []
    selected_routes: list[Any] = []
    selected_manifests: list[Any] = []
    selected_property_counts: dict[str, Any] = {}
    selected_properties: dict[str, Any] = {}
    for key in target_keys:
        entry = by_record.get(key)
        if entry is None:
            continue
        selected_records.append(key)
        selected_routes.append(entry["route"])
        selected_manifests.append(entry["manifest"])
        selected_property_counts[key] = entry["property_count"]
        selected_properties[key] = entry["properties"]
    return {
        "required": bool(config.get("require_structured_fact_robustness")),
        "canonical_route_key": canonical_key,
        "paraphrase_route_key": paraphrase_key,
        "records": selected_records,
        "routes": selected_routes,
        "manifests": selected_manifests,
        "property_counts": selected_property_counts,
        "properties": selected_properties,
    }


def _promotion_metadata(
    config: ReleaseCandidateRegistryWorkflowConfig,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = {
        **_manifest_metadata(comparison),
        "workflow": "run_release_candidate_registry_workflow",
        "manifest_fingerprint_cache": (
            None if config.fingerprint_cache_path is None else str(config.fingerprint_cache_path)
        ),
        "artifact_json_cache_path": (
            None if config.json_cache_path is None else str(config.json_cache_path)
        ),
    }
    if config.promotion_metadata is not None:
        metadata.update(dict(config.promotion_metadata))
    return metadata


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata entry must be key=value: {value!r}")
        key, text = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"metadata key must not be empty: {value!r}")
        metadata[key] = text
    return metadata


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


def _config_from_args(args: argparse.Namespace) -> ReleaseCandidateRegistryWorkflowConfig:
    return ReleaseCandidateRegistryWorkflowConfig(
        readiness_registry_path=Path(args.readiness_registry),
        route_registry_path=None if args.route_registry is None else Path(args.route_registry),
        performance_registry_path=None if args.performance_registry is None else Path(args.performance_registry),
        release_registry_path=Path(args.release_registry),
        name=args.name,
        version=args.version,
        release_policy_profile=args.release_policy_profile,
        readiness_baseline_keys=tuple(args.readiness_baseline_key or ()),
        route_baseline_keys=tuple(args.route_baseline_key or ()),
        required_route_baseline_keys=tuple(args.required_route_baseline_key or ()),
        require_structured_fact_robustness=bool(args.require_structured_fact_robustness),
        structured_fact_canonical_route_key=args.structured_fact_canonical_route_key,
        structured_fact_paraphrase_route_key=args.structured_fact_paraphrase_route_key,
        performance_baseline_key=args.performance_baseline_key,
        selector_replay_report_path=(
            None if args.selector_replay_report is None else Path(args.selector_replay_report)
        ),
        product_runtime_drift_report_path=(
            None
            if args.product_runtime_drift_report is None
            else Path(args.product_runtime_drift_report)
        ),
        require_product_runtime_drift_promotion_evidence=bool(
            args.require_product_runtime_drift_promotion_evidence
        ),
        require_product_runtime_drift_pre_generation_evidence=bool(
            args.require_product_runtime_drift_pre_generation_evidence
        ),
        require_product_runtime_drift_counterfactual_evidence=bool(
            args.require_product_runtime_drift_counterfactual_evidence
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
        release_efficiency_report_path=(
            None
            if args.release_efficiency_report is None
            else Path(args.release_efficiency_report)
        ),
        external_evidence_baseline_comparison_path=(
            None
            if args.external_evidence_baseline_comparison is None
            else Path(args.external_evidence_baseline_comparison)
        ),
        external_evidence_baseline_comparison_registry_path=(
            None
            if args.external_evidence_baseline_comparison_registry is None
            else Path(args.external_evidence_baseline_comparison_registry)
        ),
        external_evidence_baseline_comparison_key=(
            args.external_evidence_baseline_comparison_key
        ),
        pre_generation_probe_comparison_path=(
            None
            if args.pre_generation_probe_comparison is None
            else Path(args.pre_generation_probe_comparison)
        ),
        pre_generation_probe_comparison_registry_path=(
            None
            if args.pre_generation_probe_comparison_registry is None
            else Path(args.pre_generation_probe_comparison_registry)
        ),
        pre_generation_probe_comparison_key=args.pre_generation_probe_comparison_key,
        frontier_release_evidence_path=(
            None
            if args.frontier_release_evidence is None
            else Path(args.frontier_release_evidence)
        ),
        frontier_release_evidence_registry_path=(
            None
            if args.frontier_release_evidence_registry is None
            else Path(args.frontier_release_evidence_registry)
        ),
        frontier_release_evidence_key=args.frontier_release_evidence_key,
        world_model_signal_workflow_path=(
            None
            if args.world_model_signal_workflow is None
            else Path(args.world_model_signal_workflow)
        ),
        world_model_signal_workflow_registry_path=(
            None
            if args.world_model_signal_workflow_registry is None
            else Path(args.world_model_signal_workflow_registry)
        ),
        world_model_signal_workflow_key=args.world_model_signal_workflow_key,
        pathway_intervention_workflow_path=(
            None
            if args.pathway_intervention_workflow is None
            else Path(args.pathway_intervention_workflow)
        ),
        pathway_intervention_workflow_registry_path=(
            None
            if args.pathway_intervention_workflow_registry is None
            else Path(args.pathway_intervention_workflow_registry)
        ),
        pathway_intervention_workflow_key=args.pathway_intervention_workflow_key,
        product_trace_replay_workflow_path=(
            None
            if args.product_trace_replay_workflow is None
            else Path(args.product_trace_replay_workflow)
        ),
        product_trace_replay_workflow_registry_path=(
            None
            if args.product_trace_replay_workflow_registry is None
            else Path(args.product_trace_replay_workflow_registry)
        ),
        product_trace_replay_workflow_key=args.product_trace_replay_workflow_key,
        require_product_trace_action_audit_gate=bool(args.require_product_trace_action_audit_gate),
        require_product_trace_action_execution_gate=bool(
            args.require_product_trace_action_execution_gate
        ),
        selfcheck_signal_fusion_workflow_path=(
            None
            if args.selfcheck_signal_fusion_workflow is None
            else Path(args.selfcheck_signal_fusion_workflow)
        ),
        selfcheck_signal_fusion_workflow_registry_path=(
            None
            if args.selfcheck_signal_fusion_workflow_registry is None
            else Path(args.selfcheck_signal_fusion_workflow_registry)
        ),
        selfcheck_signal_fusion_workflow_key=args.selfcheck_signal_fusion_workflow_key,
        uncertainty_escalation_workflow_path=(
            None
            if args.uncertainty_escalation_workflow is None
            else Path(args.uncertainty_escalation_workflow)
        ),
        uncertainty_escalation_workflow_registry_path=(
            None
            if args.uncertainty_escalation_workflow_registry is None
            else Path(args.uncertainty_escalation_workflow_registry)
        ),
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
        feedback_policy_workflow_path=(
            None
            if args.feedback_policy_workflow is None
            else Path(args.feedback_policy_workflow)
        ),
        feedback_policy_workflow_registry_path=(
            None
            if args.feedback_policy_workflow_registry is None
            else Path(args.feedback_policy_workflow_registry)
        ),
        feedback_policy_workflow_key=args.feedback_policy_workflow_key,
        feedback_policy_min_matched_feedback_count=args.feedback_policy_min_matched_feedback_count,
        feedback_policy_min_safety_coverage=args.feedback_policy_min_safety_coverage,
        feedback_policy_max_unknown_safety_issue_rate=args.feedback_policy_max_unknown_safety_issue_rate,
        adapter_family_matrix_path=None if args.adapter_family_matrix is None else Path(args.adapter_family_matrix),
        adapter_family_profile=args.adapter_family_profile,
        required_adapter_routes=tuple(args.required_adapter_route or ()),
        require_state_transition_world_model=bool(args.require_state_transition_world_model),
        triple_extraction_fixture_matrix_path=(
            None
            if args.triple_extraction_fixture_matrix is None
            else Path(args.triple_extraction_fixture_matrix)
        ),
        triple_extraction_fixture_matrix_registry_path=(
            None
            if args.triple_extraction_fixture_matrix_registry is None
            else Path(args.triple_extraction_fixture_matrix_registry)
        ),
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
        counterfactual_verification_report_path=(
            None
            if args.counterfactual_verification_report is None
            else Path(args.counterfactual_verification_report)
        ),
        counterfactual_verification_registry_path=(
            None
            if args.counterfactual_verification_registry is None
            else Path(args.counterfactual_verification_registry)
        ),
        counterfactual_verification_key=args.counterfactual_verification_key,
        min_counterfactual_verification_records=args.min_counterfactual_verification_records,
        min_counterfactual_verification_pass_rate=(
            args.min_counterfactual_verification_pass_rate
        ),
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
        release_report_path=None if args.release_report_json is None else Path(args.release_report_json),
        artifact_manifest_path=None if args.artifact_manifest is None else Path(args.artifact_manifest),
        verification_report_path=None if args.verification_report is None else Path(args.verification_report),
        workflow_report_path=None if args.json is None else Path(args.json),
        fingerprint_cache_path=None if args.fingerprint_cache is None else Path(args.fingerprint_cache),
        json_cache_path=None if args.artifact_json_cache is None else Path(args.artifact_json_cache),
        manifest_fingerprint_workers=args.manifest_fingerprint_workers,
        recursive=not args.no_recursive,
        allow_unverified=bool(args.allow_unverified),
        runtime_profile=args.runtime_profile,
        inside_trigger_budget_policy=args.inside_trigger_budget_policy,
        min_best_quality_auroc=args.min_best_quality_auroc,
        max_uncached_forward_seconds=args.max_uncached_forward_seconds,
        max_cache_only_seconds=args.max_cache_only_seconds,
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
        require_non_oracle_evidence=bool(args.require_non_oracle_evidence),
        require_retrieval_provenance_filter=bool(args.require_retrieval_provenance_filter),
        required_retrieval_source_prefixes=_parse_csv(args.required_retrieval_source_prefix),
        required_retrieval_metadata=_parse_key_values(args.required_retrieval_metadata),
        min_retrieval_filter_score=args.min_retrieval_filter_score,
        require_retrieval_stress_control=bool(args.require_retrieval_stress_control),
        retrieval_stress_manifest_path=(
            None if args.retrieval_stress_manifest is None else Path(args.retrieval_stress_manifest)
        ),
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
        required_route_retrieval_stress_manifest_path=(
            None
            if args.required_route_retrieval_stress_manifest is None
            else Path(args.required_route_retrieval_stress_manifest)
        ),
        required_route_min_stress_false_supported_rate=args.required_route_min_stress_false_supported_rate,
        required_route_max_stress_false_refuted_rate=args.required_route_max_stress_false_refuted_rate,
        promotion_metadata=_parse_metadata(args.metadata or ()),
        allow_non_promote=bool(args.allow_non_promote),
        allow_promotion_failures=bool(args.allow_promotion_failures),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = run_release_candidate_registry_workflow(_config_from_args(args))
    decision = payload["decision"]
    print(
        "release_candidate_registry="
        f"{decision['status']} "
        f"release={decision.get('release_candidate_status')} "
        f"record={decision.get('registry_record')}"
    )
    if args.fail_on_blocked and decision["status"] != "promote":
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run release candidate gates and register the verified manifest")
    parser.add_argument("--readiness-registry", required=True)
    parser.add_argument("--route-registry", default=None)
    parser.add_argument("--performance-registry", default=None,
                        help="registry containing performance_baseline records; defaults to readiness registry")
    parser.add_argument("--release-registry", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--release-policy-profile",
        default=None,
        choices=RELEASE_POLICY_PROFILE_NAMES,
        help="optional named release gate defaults; explicit CLI thresholds override profile values",
    )
    parser.add_argument("--readiness-baseline-key", action="append", default=[])
    parser.add_argument("--route-baseline-key", action="append", default=[])
    parser.add_argument("--required-route-baseline-key", action="append", default=[],
                        help="additional promoted route benchmark_manifest key that must verify without "
                             "becoming the selected product route; repeatable")
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
    parser.add_argument("--require-product-runtime-drift-counterfactual-evidence", action="store_true",
                        help="require the product runtime drift report to include counterfactual "
                             "verifier-audit coverage, manifest, pass-rate, false-invariance, "
                             "and flip-success metrics")
    parser.add_argument("--require-product-runtime-drift-triple-audit-evidence", action="store_true",
                        help="require the product runtime drift report to include trace-level "
                             "triple coverage, audit coverage, audit pass-rate, and slot coverage metrics")
    parser.add_argument("--require-product-runtime-drift-covered-fact-property-evidence", action="store_true",
                        help="require the product runtime drift report to include covered-fact property metrics")
    parser.add_argument("--require-product-runtime-drift-action-gate-evidence", action="store_true",
                        help="require the product runtime drift report to include product-trace "
                             "action-audit and action-execution drift metrics")
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
    parser.add_argument("--frontier-release-evidence", default=None,
                        help="optional frontier release-evidence report that must promote and verify")
    parser.add_argument("--frontier-release-evidence-registry", default=None,
                        help="optional ArtifactRegistry JSON path for --frontier-release-evidence-key; "
                             "defaults to --readiness-registry")
    parser.add_argument("--frontier-release-evidence-key", default=None,
                        help="optional report:<name>:<version> registry key for frontier release evidence")
    parser.add_argument("--world-model-signal-workflow", default=None,
                        help="optional world-model signal calibration workflow report that must pass its "
                             "conflict/trace-gap release gate")
    parser.add_argument("--world-model-signal-workflow-registry", default=None,
                        help="optional ArtifactRegistry JSON path for --world-model-signal-workflow-key; "
                             "defaults to --readiness-registry")
    parser.add_argument("--world-model-signal-workflow-key", default=None,
                        help="optional report:<name>:<version> registry key for a world-model signal workflow")
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
                        help="optional minimum matched feedback count forwarded to the feedback-policy workflow gate")
    parser.add_argument("--feedback-policy-min-safety-coverage", type=lambda value: _parse_unit_float(
        value,
        flag="--feedback-policy-min-safety-coverage",
    ), default=None,
                        help="optional minimum feedback replay safety coverage forwarded to the feedback-policy gate")
    parser.add_argument("--feedback-policy-max-unknown-safety-issue-rate", type=lambda value: _parse_unit_float(
        value,
        flag="--feedback-policy-max-unknown-safety-issue-rate",
    ), default=None,
                        help="optional maximum unknown safety issue rate forwarded to the feedback-policy gate")
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
                        help="optional report:<name>:<version> registry key for a counterfactual verifier audit")
    parser.add_argument("--min-counterfactual-verification-records",
                        type=lambda value: _parse_non_negative_int(
                            value,
                            flag="--min-counterfactual-verification-records",
                        ), default=None,
                        help="optional minimum audited counterfactual pair count")
    parser.add_argument("--min-counterfactual-verification-pass-rate",
                        type=lambda value: _parse_unit_float(
                            value,
                            flag="--min-counterfactual-verification-pass-rate",
                        ), default=None,
                        help="optional minimum counterfactual verifier audit pass rate")
    parser.add_argument("--max-counterfactual-verification-false-invariance-rate",
                        type=lambda value: _parse_unit_float(
                            value,
                            flag="--max-counterfactual-verification-false-invariance-rate",
                        ), default=None,
                        help="optional maximum rate where false counterfactuals remain supported")
    parser.add_argument("--require-performance-score-dump-cache", action="store_true",
                        help="require the selected performance baseline to include score-dump cache evidence")
    parser.add_argument("--json", default=None, help="optional registry workflow report path")
    parser.add_argument("--release-report-json", default=None,
                        help="optional path for the release candidate comparison report")
    parser.add_argument("--artifact-manifest", default=None,
                        help="optional path for the release candidate artifact manifest")
    parser.add_argument("--verification-report", default=None)
    parser.add_argument("--fingerprint-cache", default=None,
                        help="optional JSON cache for recursive manifest fingerprint reads")
    parser.add_argument("--artifact-json-cache", default=None,
                        help="optional path-signature JSON artifact cache for release report/manifests")
    parser.add_argument(
        "--manifest-fingerprint-workers",
        type=lambda value: _parse_positive_int(value, flag="--manifest-fingerprint-workers"),
        default=1,
        help="bounded worker count for release manifest artifact fingerprinting",
    )
    parser.add_argument("--metadata", action="append", default=[], help="extra promotion metadata as key=value")
    parser.add_argument("--allow-non-promote", action="store_true",
                        help="register even when the release candidate comparison does not promote")
    parser.add_argument("--allow-promotion-failures", action="store_true",
                        help="register even when manifest verification fails")
    parser.add_argument("--no-recursive", action="store_true", help="only verify root manifests")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="allow unverified input baseline manifests to become candidates")
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
        type=lambda value: _parse_non_negative_float(value, flag="--min-retrieval-filter-score"),
        default=None,
        help="minimum min_score required in the selected route retrieval provenance filter",
    )
    parser.add_argument(
        "--require-retrieval-stress-control",
        action="store_true",
        help="require selected retrieval route evidence to pass an answer-echo stress control",
    )
    parser.add_argument(
        "--retrieval-stress-manifest",
        default=None,
        help="optional selected-route answer-echo retrieval stress artifact manifest path",
    )
    parser.add_argument(
        "--min-stress-false-supported-rate",
        type=lambda value: _parse_unit_float(value, flag="--min-stress-false-supported-rate"),
        default=None,
        help="minimum false-supported rate expected on answer-echo stress control",
    )
    parser.add_argument(
        "--max-stress-false-refuted-rate",
        type=lambda value: _parse_unit_float(value, flag="--max-stress-false-refuted-rate"),
        default=None,
        help="maximum false-refuted rate expected on answer-echo stress control",
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
        help="require required route evidence to pass an answer-echo retrieval stress control",
    )
    parser.add_argument(
        "--required-route-retrieval-stress-manifest",
        default=None,
        help="optional required-route answer-echo retrieval stress artifact manifest path",
    )
    parser.add_argument(
        "--required-route-min-stress-false-supported-rate",
        type=lambda value: _parse_unit_float(
            value,
            flag="--required-route-min-stress-false-supported-rate",
        ),
        default=None,
        help="minimum false-supported rate expected for required-route answer-echo stress control",
    )
    parser.add_argument(
        "--required-route-max-stress-false-refuted-rate",
        type=lambda value: _parse_unit_float(
            value,
            flag="--required-route-max-stress-false-refuted-rate",
        ),
        default=None,
        help="maximum false-refuted rate expected for required-route answer-echo stress control",
    )
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless the release candidate registry workflow promotes")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
