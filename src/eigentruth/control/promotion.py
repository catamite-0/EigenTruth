"""Promotion contracts that bridge offline release reports to product control."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from eigentruth.control import runtime_drift_keys as _runtime_drift_keys
from eigentruth.control.controller import ControlPolicyConfig
from eigentruth.control.runtime_budget import ProductRuntimeBudgetPolicy
from eigentruth.json_utils import strict_json_dumps
from eigentruth.registry import (
    ArtifactManifestVerification,
    ArtifactRegistry,
    RegistryRecord,
    load_and_verify_artifact_manifest,
)

_PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_CLAIM_FACTUALITY_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_CLAIM_FACTUALITY_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_ACTION_GATE_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_ACTION_GATE_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_TRAJECTORY_AUDIT_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_TRAJECTORY_AUDIT_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_EVIDENCE_HANDOFF_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_EVIDENCE_HANDOFF_EVIDENCE_KEYS
)


@dataclass(frozen=True)
class ProductPromotionContract:
    """Deployable product-control contract derived from a release candidate."""

    model_id: str | None = None
    runtime: Mapping[str, Any] = field(default_factory=dict)
    verifier_route: Mapping[str, Any] = field(default_factory=dict)
    runtime_budget_policy: ProductRuntimeBudgetPolicy | Mapping[str, Any] = field(
        default_factory=ProductRuntimeBudgetPolicy
    )
    control_policy_config: Mapping[str, Any] = field(default_factory=dict)
    control_defaults: Mapping[str, Any] = field(default_factory=dict)
    source_workflow: str | None = None
    source_status: str | None = None
    product_trace_replay_workflow: Mapping[str, Any] = field(default_factory=dict)
    selfcheck_signal_fusion_workflow: Mapping[str, Any] = field(default_factory=dict)
    world_model_signal_workflow: Mapping[str, Any] = field(default_factory=dict)
    pathway_intervention_workflow: Mapping[str, Any] = field(default_factory=dict)
    feedback_policy_workflow: Mapping[str, Any] = field(default_factory=dict)
    external_evidence_baseline_comparison: Mapping[str, Any] = field(default_factory=dict)
    pre_generation_probe_comparison: Mapping[str, Any] = field(default_factory=dict)
    claim_factuality_probe_comparison: Mapping[str, Any] = field(default_factory=dict)
    triple_extraction_fixture_matrix: Mapping[str, Any] = field(default_factory=dict)
    counterfactual_verification: Mapping[str, Any] = field(default_factory=dict)
    release_efficiency: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        policy = (
            self.runtime_budget_policy
            if isinstance(self.runtime_budget_policy, ProductRuntimeBudgetPolicy)
            else ProductRuntimeBudgetPolicy.from_mapping(self.runtime_budget_policy)
        )
        object.__setattr__(self, "runtime", dict(self.runtime))
        object.__setattr__(self, "verifier_route", dict(self.verifier_route))
        object.__setattr__(self, "runtime_budget_policy", policy)
        control_policy = _control_policy_config_dict(self.control_policy_config)
        object.__setattr__(self, "control_policy_config", control_policy)
        object.__setattr__(self, "control_defaults", dict(self.control_defaults))
        object.__setattr__(
            self,
            "product_trace_replay_workflow",
            dict(self.product_trace_replay_workflow),
        )
        object.__setattr__(
            self,
            "selfcheck_signal_fusion_workflow",
            dict(self.selfcheck_signal_fusion_workflow),
        )
        object.__setattr__(
            self,
            "world_model_signal_workflow",
            dict(self.world_model_signal_workflow),
        )
        object.__setattr__(
            self,
            "pathway_intervention_workflow",
            dict(self.pathway_intervention_workflow),
        )
        object.__setattr__(
            self,
            "feedback_policy_workflow",
            dict(self.feedback_policy_workflow),
        )
        object.__setattr__(
            self,
            "external_evidence_baseline_comparison",
            dict(self.external_evidence_baseline_comparison),
        )
        object.__setattr__(
            self,
            "pre_generation_probe_comparison",
            dict(self.pre_generation_probe_comparison),
        )
        object.__setattr__(
            self,
            "claim_factuality_probe_comparison",
            dict(self.claim_factuality_probe_comparison),
        )
        object.__setattr__(
            self,
            "triple_extraction_fixture_matrix",
            dict(self.triple_extraction_fixture_matrix),
        )
        object.__setattr__(
            self,
            "counterfactual_verification",
            dict(self.counterfactual_verification),
        )
        object.__setattr__(self, "release_efficiency", dict(self.release_efficiency))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        require_promoted: bool = True,
    ) -> "ProductPromotionContract":
        """Build a contract from either a contract payload or release report."""
        if payload.get("workflow") == "product_promotion_contract":
            return cls(
                schema_version=int(payload.get("schema_version", 1)),
                source_workflow=_optional_str(payload.get("source_workflow")),
                source_status=_optional_str(payload.get("source_status")),
                model_id=_optional_str(payload.get("model_id")),
                runtime=_mapping(payload.get("runtime")),
                verifier_route=_mapping(payload.get("verifier_route")),
                runtime_budget_policy=ProductRuntimeBudgetPolicy.from_mapping(
                    _mapping(payload.get("runtime_budget_policy"))
                ),
                control_policy_config=_mapping(payload.get("control_policy_config")),
                control_defaults=_mapping(payload.get("control_defaults")),
                product_trace_replay_workflow=_mapping(
                    payload.get("product_trace_replay_workflow")
                ),
                selfcheck_signal_fusion_workflow=_mapping(
                    payload.get("selfcheck_signal_fusion_workflow")
                ),
                world_model_signal_workflow=_mapping(
                    payload.get("world_model_signal_workflow")
                ),
                pathway_intervention_workflow=_mapping(
                    payload.get("pathway_intervention_workflow")
                ),
                feedback_policy_workflow=_mapping(payload.get("feedback_policy_workflow")),
                external_evidence_baseline_comparison=_mapping(
                    payload.get("external_evidence_baseline_comparison")
                ),
                pre_generation_probe_comparison=_mapping(
                    payload.get("pre_generation_probe_comparison")
                ),
                claim_factuality_probe_comparison=_mapping(
                    payload.get("claim_factuality_probe_comparison")
                ),
                triple_extraction_fixture_matrix=_mapping(
                    payload.get("triple_extraction_fixture_matrix")
                ),
                counterfactual_verification=_mapping(
                    payload.get("counterfactual_verification")
                ),
                release_efficiency=_mapping(payload.get("release_efficiency")),
                metadata=_mapping(payload.get("metadata")),
            )
        return cls.from_release_candidate_report(payload, require_promoted=require_promoted)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        require_promoted: bool = True,
    ) -> "ProductPromotionContract":
        """Load a contract or release-candidate report from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("promotion contract JSON must contain an object.")
        return cls.from_mapping(payload, require_promoted=require_promoted)

    @classmethod
    def from_release_candidate_report(
        cls,
        report: Mapping[str, Any],
        *,
        require_promoted: bool = True,
    ) -> "ProductPromotionContract":
        """Build a product contract from a release-candidate comparison report."""
        comparison = _release_candidate_comparison(report)
        decision = _mapping(comparison.get("decision"))
        status = _optional_str(decision.get("status"))
        if require_promoted and status != "promote":
            raise ValueError("release candidate report did not promote.")
        candidate = _mapping(comparison.get("release_candidate"))
        if not candidate:
            raise ValueError("release candidate report does not contain a release_candidate.")
        config = _mapping(comparison.get("config"))
        manifests = _mapping(candidate.get("manifests"))
        adapter_family = _mapping(candidate.get("adapter_family_matrix"))
        required_route_baselines = _mapping(candidate.get("required_route_baselines"))
        product_trace_replay_workflow = _mapping(
            candidate.get("product_trace_replay_workflow")
        )
        selfcheck_signal_fusion_workflow = _mapping(
            candidate.get("selfcheck_signal_fusion_workflow")
        )
        world_model_signal_workflow = _mapping(candidate.get("world_model_signal_workflow"))
        pathway_intervention_workflow = _mapping(
            candidate.get("pathway_intervention_workflow")
        )
        feedback_policy_workflow = _mapping(candidate.get("feedback_policy_workflow"))
        external_evidence_baseline_comparison = (
            _external_evidence_baseline_comparison_metadata(
                _first_mapping(
                    candidate.get("external_evidence_baseline_comparison"),
                    comparison.get("external_evidence_baseline_comparison_gate"),
                ),
            )
        )
        pre_generation_probe_comparison = _pre_generation_probe_comparison_metadata(
            _first_mapping(
                candidate.get("pre_generation_probe_comparison"),
                comparison.get("pre_generation_probe_comparison_gate"),
            ),
        )
        claim_factuality_probe_comparison = (
            _claim_factuality_probe_comparison_metadata(
                _first_mapping(
                    candidate.get("claim_factuality_probe_comparison"),
                    comparison.get("claim_factuality_probe_comparison_gate"),
                ),
            )
        )
        triple_extraction_fixture_matrix = _triple_extraction_fixture_matrix_metadata(
            _first_mapping(
                candidate.get("triple_extraction_fixture_matrix"),
                comparison.get("triple_extraction_fixture_matrix_gate"),
            ),
            manifests=manifests,
        )
        counterfactual_verification = _counterfactual_verification_metadata(
            _first_mapping(
                candidate.get("counterfactual_verification"),
                comparison.get("counterfactual_verification_gate"),
            ),
            manifests=manifests,
        )
        release_efficiency = _release_efficiency_metadata(
            _mapping(candidate.get("release_efficiency")),
            manifests=manifests,
        )
        runtime_cost = _mapping(candidate.get("runtime_cost"))
        product_trace_replay_metadata = _product_trace_replay_workflow_metadata(
            product_trace_replay_workflow,
            manifests=manifests,
        )
        selector_replay = _mapping(candidate.get("selector_replay"))
        selector_replay_recommended = _mapping(selector_replay.get("recommended"))
        product_runtime_drift = _mapping(candidate.get("product_runtime_drift"))
        product_runtime_drift_summary = _mapping(product_runtime_drift.get("summary"))
        product_runtime_drift_baseline = _mapping(product_runtime_drift.get("baseline"))
        product_runtime_drift_current = _mapping(product_runtime_drift.get("current"))
        performance_evidence_bundle = _mapping(candidate.get("performance_evidence_bundle"))
        performance_evidence_recommendation = _mapping(
            performance_evidence_bundle.get("recommendation")
        )
        performance_evidence = _mapping(performance_evidence_bundle.get("evidence"))
        performance_evidence_cost = _mapping(performance_evidence_bundle.get("cost"))
        performance_score_dump_cache = _mapping(performance_evidence_bundle.get("score_dump_cache"))
        performance_score_dump_cache_totals = _mapping(performance_score_dump_cache.get("totals"))
        performance_jsonl_view_cache = _mapping(performance_score_dump_cache_totals.get("jsonl_view"))
        performance_gate = _mapping(comparison.get("performance_baseline_gate"))
        performance_trend_gate = _mapping(performance_gate.get("performance_trend_gate"))
        performance_covariance_gate = _mapping(performance_gate.get("covariance_tradeoff_gate"))
        if not performance_covariance_gate:
            performance_covariance_gate = _mapping(
                _mapping(performance_gate.get("gate")).get("covariance_tradeoff")
            )
        performance_trend_metrics = _mapping(performance_trend_gate.get("metrics"))
        performance_uncached_trend = _mapping(
            performance_trend_metrics.get("uncached_total_seconds")
        )
        performance_cached_trend = _mapping(
            performance_trend_metrics.get("cached_total_seconds")
        )
        performance_cache_only_trend = _mapping(
            performance_trend_metrics.get("cache_only_total_seconds")
        )
        performance_cache_hit_rate_trend = _mapping(
            performance_trend_metrics.get("score_dump_cache_jsonl_view_hit_rate")
        )
        quality = _mapping(candidate.get("quality"))
        readiness_covariance_gate = _mapping(quality.get("covariance_tradeoff_gate"))
        verifier_route = _mapping(candidate.get("verifier_route"))
        required_route_property_counts = _mapping(
            required_route_baselines.get("covered_fact_property_counts")
        )
        required_route_property_sets = _mapping(
            required_route_baselines.get("covered_fact_properties")
        )
        required_route_property_metrics = _mapping(
            required_route_baselines.get("covered_fact_property_metrics")
        )
        structured_fact_robustness = _structured_fact_robustness_metadata(
            config,
            required_route_baselines,
        )
        return cls(
            source_workflow=_optional_str(comparison.get("workflow")),
            source_status=status,
            model_id=_optional_str(candidate.get("model")),
            runtime=_mapping(candidate.get("runtime")),
            verifier_route=verifier_route,
            runtime_budget_policy=product_runtime_budget_policy_from_release_candidate(
                comparison
            ),
            control_policy_config=_product_control_policy_config_from_release_candidate(
                feedback_policy_workflow
            ),
            control_defaults=_product_control_defaults_from_release_candidate(
                comparison
            ),
            product_trace_replay_workflow=product_trace_replay_metadata,
            selfcheck_signal_fusion_workflow=_selfcheck_signal_fusion_workflow_metadata(
                selfcheck_signal_fusion_workflow,
                manifests=manifests,
            ),
            world_model_signal_workflow=_world_model_signal_workflow_metadata(
                world_model_signal_workflow,
                manifests=manifests,
            ),
            pathway_intervention_workflow=_pathway_intervention_workflow_metadata(
                pathway_intervention_workflow,
                manifests=manifests,
            ),
            feedback_policy_workflow=_feedback_policy_workflow_metadata(
                feedback_policy_workflow,
                manifests=manifests,
            ),
            external_evidence_baseline_comparison=external_evidence_baseline_comparison,
            pre_generation_probe_comparison=pre_generation_probe_comparison,
            claim_factuality_probe_comparison=claim_factuality_probe_comparison,
            triple_extraction_fixture_matrix=triple_extraction_fixture_matrix,
            counterfactual_verification=counterfactual_verification,
            release_efficiency=release_efficiency,
            metadata={
                "recommended_readiness_record": decision.get("recommended_readiness_record"),
                "recommended_route_record": decision.get("recommended_route_record"),
                "recommended_performance_baseline_record": decision.get(
                    "recommended_performance_baseline_record"
                ),
                "recommended_selector_replay_candidate": decision.get(
                    "recommended_selector_replay_candidate"
                ),
                "recommended_route_covered_fact_property_count": (
                    verifier_route.get("covered_fact_property_count")
                ),
                "recommended_route_covered_fact_properties": (
                    verifier_route.get("covered_fact_properties")
                ),
                "recommended_route_covered_fact_property_metrics": (
                    verifier_route.get("covered_fact_property_metrics")
                ),
                "recommended_product_runtime_drift_report": decision.get(
                    "recommended_product_runtime_drift_report"
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
                "product_trace_replay_workflow_status": decision.get(
                    "product_trace_replay_workflow_status"
                ),
                "product_trace_replay_workflow_report": product_trace_replay_metadata.get(
                    "report_path"
                ),
                "product_trace_replay_workflow_manifest": (
                    product_trace_replay_metadata.get("manifest_path")
                ),
                "product_trace_replay_workflow_source": product_trace_replay_metadata.get(
                    "source"
                ),
                "product_trace_replay_workflow_registry": product_trace_replay_metadata.get(
                    "registry"
                ),
                "product_trace_replay_workflow_record": product_trace_replay_metadata.get(
                    "record_key"
                ),
                "product_trace_replay_workflow_report_status": (
                    product_trace_replay_metadata.get("report_status")
                ),
                "product_trace_replay_workflow_selector_replay_report": (
                    product_trace_replay_metadata.get("selector_replay_report_path")
                ),
                "product_trace_replay_workflow_runtime_drift_report": (
                    product_trace_replay_metadata.get("product_runtime_drift_report_path")
                ),
                "product_trace_action_audit_gate_required": _first_present(
                    product_trace_replay_metadata.get("require_action_audit_gate"),
                    config.get("require_product_trace_action_audit_gate"),
                ),
                "product_trace_action_audit_gate_status": (
                    product_trace_replay_metadata.get("action_audit_gate_status")
                ),
                "product_trace_action_audit_gate_enabled": (
                    product_trace_replay_metadata.get("action_audit_gate_enabled")
                ),
                "product_trace_action_audit_gate_passed": (
                    product_trace_replay_metadata.get("action_audit_gate_passed")
                ),
                "product_trace_action_audit_gate_report": (
                    product_trace_replay_metadata.get("action_audit_gate_report_path")
                ),
                "product_trace_action_audit_error_rate": (
                    product_trace_replay_metadata.get("action_audit_error_rate")
                ),
                "product_trace_action_audit_missing_retrieval_action_rate": (
                    product_trace_replay_metadata.get(
                        "action_audit_missing_retrieval_action_rate"
                    )
                ),
                "product_trace_action_audit_missing_plan_retrieval_query_rate": (
                    product_trace_replay_metadata.get(
                        "action_audit_missing_plan_retrieval_query_rate"
                    )
                ),
                "product_trace_action_audit_malformed_payload_rate": (
                    product_trace_replay_metadata.get("action_audit_malformed_payload_rate")
                ),
                "product_trace_action_audit_unexpected_action_rate": (
                    product_trace_replay_metadata.get("action_audit_unexpected_action_rate")
                ),
                "product_trace_action_audit_unknown_claim_id_rate": (
                    product_trace_replay_metadata.get("action_audit_unknown_claim_id_rate")
                ),
                "product_trace_action_execution_gate_required": _first_present(
                    product_trace_replay_metadata.get("require_action_execution_gate"),
                    config.get("require_product_trace_action_execution_gate"),
                ),
                "product_trace_action_execution_gate_status": (
                    product_trace_replay_metadata.get("action_execution_gate_status")
                ),
                "product_trace_action_execution_gate_enabled": (
                    product_trace_replay_metadata.get("action_execution_gate_enabled")
                ),
                "product_trace_action_execution_gate_passed": (
                    product_trace_replay_metadata.get("action_execution_gate_passed")
                ),
                "product_trace_action_execution_gate_report": (
                    product_trace_replay_metadata.get("action_execution_gate_report_path")
                ),
                "product_trace_action_execution_alignment_failed_trace_rate": (
                    product_trace_replay_metadata.get(
                        "action_execution_alignment_failed_trace_rate"
                    )
                ),
                "product_trace_action_execution_missing_result_rate": (
                    product_trace_replay_metadata.get("action_execution_missing_result_rate")
                ),
                "product_trace_action_execution_unexpected_result_rate": (
                    product_trace_replay_metadata.get("action_execution_unexpected_result_rate")
                ),
                "product_trace_action_execution_request_id_mismatch_rate": (
                    product_trace_replay_metadata.get(
                        "action_execution_request_id_mismatch_rate"
                    )
                ),
                "selfcheck_signal_fusion_workflow_status": decision.get(
                    "selfcheck_signal_fusion_workflow_status"
                ),
                "recommended_selfcheck_signal_fusion_workflow_report": decision.get(
                    "recommended_selfcheck_signal_fusion_workflow_report"
                ),
                "world_model_signal_workflow_status": decision.get(
                    "world_model_signal_workflow_status"
                ),
                "recommended_world_model_signal_workflow_report": decision.get(
                    "recommended_world_model_signal_workflow_report"
                ),
                "pathway_intervention_workflow_status": decision.get(
                    "pathway_intervention_workflow_status"
                ),
                "recommended_pathway_intervention_workflow_report": decision.get(
                    "recommended_pathway_intervention_workflow_report"
                ),
                "triple_extraction_fixture_matrix_status": _first_present(
                    decision.get("triple_extraction_fixture_matrix_status"),
                    triple_extraction_fixture_matrix.get("status"),
                ),
                "recommended_triple_extraction_fixture_matrix_report": decision.get(
                    "recommended_triple_extraction_fixture_matrix_report"
                ),
                **_triple_extraction_fixture_matrix_flat_metadata(
                    triple_extraction_fixture_matrix
                ),
                "counterfactual_verification_status": _first_present(
                    decision.get("counterfactual_verification_status"),
                    counterfactual_verification.get("status"),
                ),
                "recommended_counterfactual_verification_report": (
                    decision.get("recommended_counterfactual_verification_report")
                ),
                **_counterfactual_verification_flat_metadata(
                    counterfactual_verification
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
                "triple_extraction_fixture_matrix_min_corpora": config.get(
                    "min_triple_extraction_corpora"
                ),
                "triple_extraction_fixture_matrix_min_distinct_predicates": config.get(
                    "min_triple_extraction_distinct_predicates"
                ),
                "world_model_signal_workflow_report": (
                    world_model_signal_workflow.get("report_path")
                ),
                "world_model_signal_workflow_manifest": (
                    world_model_signal_workflow.get("manifest_path")
                    or manifests.get("world_model_signal_workflow_manifest")
                ),
                "world_model_signal_workflow_source": (
                    world_model_signal_workflow.get("source")
                ),
                "world_model_signal_workflow_registry": (
                    world_model_signal_workflow.get("registry")
                ),
                "world_model_signal_workflow_record": (
                    world_model_signal_workflow.get("record_key")
                ),
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
                "pathway_intervention_workflow_report": (
                    pathway_intervention_workflow.get("report_path")
                ),
                "pathway_intervention_workflow_manifest": (
                    pathway_intervention_workflow.get("manifest_path")
                    or manifests.get("pathway_intervention_workflow_manifest")
                ),
                "pathway_intervention_workflow_source": (
                    pathway_intervention_workflow.get("source")
                ),
                "pathway_intervention_workflow_registry": (
                    pathway_intervention_workflow.get("registry")
                ),
                "pathway_intervention_workflow_record": (
                    pathway_intervention_workflow.get("record_key")
                ),
                "pathway_intervention_workflow_report_status": (
                    pathway_intervention_workflow.get("report_status")
                ),
                "pathway_intervention_workflow_release_ready": (
                    pathway_intervention_workflow.get("release_ready")
                ),
                "pathway_intervention_workflow_model": (
                    pathway_intervention_workflow.get("model")
                ),
                "pathway_intervention_workflow_layer": (
                    pathway_intervention_workflow.get("layer")
                ),
                "pathway_intervention_workflow_intervention_layer": (
                    pathway_intervention_workflow.get("intervention_layer")
                ),
                "pathway_intervention_workflow_patch_layer": (
                    pathway_intervention_workflow.get("patch_layer")
                ),
                "pathway_intervention_workflow_activation_ablation_gate": (
                    pathway_intervention_workflow.get("activation_ablation_gate_status")
                ),
                "pathway_intervention_workflow_source_patch_gate": (
                    pathway_intervention_workflow.get("source_patch_gate_status")
                ),
                "pathway_intervention_workflow_signals": (
                    pathway_intervention_workflow.get("signals")
                ),
                "pathway_intervention_workflow_best_signals": (
                    pathway_intervention_workflow.get("best_signals")
                ),
                "selfcheck_signal_fusion_workflow_report": (
                    selfcheck_signal_fusion_workflow.get("report_path")
                ),
                "selfcheck_signal_fusion_workflow_manifest": (
                    selfcheck_signal_fusion_workflow.get("manifest_path")
                    or manifests.get("selfcheck_signal_fusion_workflow_manifest")
                ),
                "selfcheck_signal_fusion_workflow_source": (
                    selfcheck_signal_fusion_workflow.get("source")
                ),
                "selfcheck_signal_fusion_workflow_registry": (
                    selfcheck_signal_fusion_workflow.get("registry")
                ),
                "selfcheck_signal_fusion_workflow_record": (
                    selfcheck_signal_fusion_workflow.get("record_key")
                ),
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
                "feedback_policy_workflow_status": decision.get(
                    "feedback_policy_workflow_status"
                ),
                "feedback_policy_workflow_report": feedback_policy_workflow.get("report_path"),
                "feedback_policy_workflow_manifest": (
                    feedback_policy_workflow.get("manifest_path")
                    or manifests.get("feedback_policy_workflow_manifest")
                ),
                "feedback_policy_workflow_source": feedback_policy_workflow.get("source"),
                "feedback_policy_workflow_registry": feedback_policy_workflow.get("registry"),
                "feedback_policy_workflow_record": feedback_policy_workflow.get("record_key"),
                "feedback_policy_workflow_report_status": (
                    feedback_policy_workflow.get("report_status")
                ),
                "feedback_policy_workflow_promotion_decision": (
                    feedback_policy_workflow.get("promotion_decision")
                ),
                "feedback_policy_workflow_candidate_control_policy": (
                    feedback_policy_workflow.get("candidate_control_policy")
                ),
                "feedback_policy_workflow_candidate_control_defaults": (
                    feedback_policy_workflow.get("candidate_control_defaults")
                ),
                "feedback_policy_workflow_matched_feedback_count": (
                    feedback_policy_workflow.get("matched_feedback_count")
                ),
                "feedback_policy_workflow_accepted_but_wrong_rate": (
                    feedback_policy_workflow.get("accepted_but_wrong_rate")
                ),
                "feedback_policy_workflow_retrieved_failure_rate": (
                    feedback_policy_workflow.get("retrieved_failure_rate")
                ),
                "feedback_policy_workflow_abstain_false_positive_rate": (
                    feedback_policy_workflow.get("abstain_false_positive_rate")
                ),
                "feedback_policy_workflow_final_answered_but_wrong_rate": (
                    feedback_policy_workflow.get("final_answered_but_wrong_rate")
                ),
                "feedback_policy_workflow_final_answer_false_block_rate": (
                    feedback_policy_workflow.get("final_answer_false_block_rate")
                ),
                "feedback_policy_workflow_safety_coverage_rate": (
                    feedback_policy_workflow.get("safety_coverage_rate")
                ),
                "feedback_policy_workflow_unknown_safety_issue_rate": (
                    feedback_policy_workflow.get("unknown_safety_issue_rate")
                ),
                "external_evidence_baseline_comparison_status": _first_present(
                    decision.get("external_evidence_baseline_comparison_status"),
                    external_evidence_baseline_comparison.get("decision_status"),
                    external_evidence_baseline_comparison.get("status"),
                ),
                "recommended_external_evidence_baseline_comparison_report": (
                    decision.get("recommended_external_evidence_baseline_comparison_report")
                ),
                **_external_evidence_baseline_comparison_flat_metadata(
                    external_evidence_baseline_comparison
                ),
                "pre_generation_probe_comparison_status": _first_present(
                    decision.get("pre_generation_probe_comparison_status"),
                    pre_generation_probe_comparison.get("status"),
                ),
                "recommended_pre_generation_probe_comparison_report": (
                    decision.get("recommended_pre_generation_probe_comparison_report")
                ),
                **_pre_generation_probe_comparison_flat_metadata(
                    pre_generation_probe_comparison
                ),
                "claim_factuality_probe_comparison_status": _first_present(
                    decision.get("claim_factuality_probe_comparison_status"),
                    claim_factuality_probe_comparison.get("status"),
                    claim_factuality_probe_comparison.get("report_status"),
                ),
                "recommended_claim_factuality_probe_comparison_report": (
                    decision.get("recommended_claim_factuality_probe_comparison_report")
                ),
                **_claim_factuality_probe_comparison_flat_metadata(
                    claim_factuality_probe_comparison
                ),
                **_release_efficiency_flat_metadata(release_efficiency),
                "performance_baseline_record": candidate.get("performance_baseline_record"),
                "performance_evidence_bundle_status": performance_evidence_bundle.get("status"),
                "performance_evidence_bundle_release_ready": (
                    performance_evidence_bundle.get("release_ready")
                ),
                "performance_cache_tuning_status": (
                    performance_evidence_recommendation.get("cache_tuning_status")
                ),
                "performance_best_quality_signal": (
                    performance_evidence_recommendation.get("best_quality_signal")
                ),
                "performance_best_quality_auroc": (
                    performance_evidence_recommendation.get("best_quality_auroc")
                ),
                "performance_score_fusion_status": _first_present(
                    performance_evidence_recommendation.get("score_fusion_status"),
                    performance_evidence.get("score_fusion_status"),
                ),
                "performance_score_fusion_signal": _first_present(
                    performance_evidence_recommendation.get("score_fusion_signal"),
                    performance_evidence.get("score_fusion_signal"),
                ),
                "performance_score_fusion_auroc": _first_present(
                    performance_evidence_recommendation.get("score_fusion_auroc"),
                    performance_evidence.get("score_fusion_auroc"),
                ),
                "performance_score_fusion_conformal_gate_passed": _first_present(
                    performance_evidence_recommendation.get("score_fusion_conformal_gate_passed"),
                    performance_evidence.get("score_fusion_conformal_gate_passed"),
                ),
                "performance_selected_fusion_status": _first_present(
                    performance_evidence_recommendation.get("selected_fusion_status"),
                    performance_evidence.get("selected_fusion_status"),
                ),
                "performance_selected_fusion_run": _first_present(
                    performance_evidence_recommendation.get("selected_fusion_run"),
                    performance_evidence.get("selected_fusion_run"),
                ),
                "performance_selected_fusion_candidate": _first_present(
                    performance_evidence_recommendation.get("selected_fusion_candidate"),
                    performance_evidence.get("selected_fusion_candidate"),
                ),
                "performance_selected_fusion_signal": _first_present(
                    performance_evidence_recommendation.get("selected_fusion_signal"),
                    performance_evidence.get("selected_fusion_signal"),
                ),
                "performance_selected_fusion_auroc": _first_present(
                    performance_evidence_recommendation.get("selected_fusion_auroc"),
                    performance_evidence.get("selected_fusion_auroc"),
                ),
                "performance_selected_fusion_false_alarm": (
                    performance_evidence.get("selected_fusion_false_alarm")
                ),
                "performance_selected_fusion_detection": (
                    performance_evidence.get("selected_fusion_detection")
                ),
                "performance_selected_fusion_artifact_report": (
                    performance_evidence.get("selected_fusion_artifact_report")
                ),
                "performance_selected_fusion_artifact_path": _first_present(
                    performance_evidence_recommendation.get("selected_fusion_artifact_path"),
                    performance_evidence.get("selected_fusion_artifact_path"),
                ),
                "performance_uncached_total_seconds": (
                    performance_evidence_cost.get("uncached_total_seconds")
                ),
                "performance_cached_total_ratio": performance_evidence_cost.get(
                    "cached_total_ratio"
                ),
                "performance_cache_only_total_ratio": performance_evidence_cost.get(
                    "cache_only_total_ratio"
                ),
                "performance_score_dump_cache_required": (
                    config.get("require_performance_score_dump_cache")
                ),
                "performance_score_dump_cache_min_jsonl_view_hit_rate": (
                    config.get("min_performance_score_dump_cache_jsonl_view_hit_rate")
                ),
                "performance_score_dump_cache_source_count": (
                    performance_score_dump_cache.get("source_count")
                ),
                "performance_score_dump_cache_jsonl_view_hit_rate": (
                    performance_jsonl_view_cache.get("hit_rate")
                ),
                "performance_drift_baseline_record": config.get(
                    "performance_drift_baseline_key"
                ),
                "performance_trend_gate_passed": performance_trend_gate.get("passed"),
                "performance_trend_reference_record": performance_trend_gate.get(
                    "reference_record_key"
                ),
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
                "max_covariance_maha_last_auroc_drop": (
                    config.get("max_covariance_maha_last_auroc_drop")
                ),
                "readiness_covariance_tradeoff_gate_passed": (
                    readiness_covariance_gate.get("passed")
                ),
                "readiness_covariance_tradeoff_status": (
                    readiness_covariance_gate.get("status")
                ),
                "readiness_covariance_selected_mode": (
                    readiness_covariance_gate.get("selected_covariance_mode")
                ),
                "readiness_covariance_selected_low_rank": (
                    readiness_covariance_gate.get("selected_covariance_low_rank")
                ),
                "readiness_covariance_maha_last_delta_vs_baseline": (
                    readiness_covariance_gate.get("selected_maha_last_delta_vs_baseline")
                ),
                "performance_covariance_tradeoff_gate_passed": (
                    performance_covariance_gate.get("passed")
                ),
                "performance_covariance_tradeoff_status": (
                    performance_covariance_gate.get("status")
                ),
                "performance_covariance_selected_mode": (
                    performance_covariance_gate.get("selected_covariance_mode")
                ),
                "performance_covariance_selected_low_rank": (
                    performance_covariance_gate.get("selected_covariance_low_rank")
                ),
                "performance_covariance_maha_last_delta_vs_baseline": (
                    performance_covariance_gate.get("selected_maha_last_delta_vs_baseline")
                ),
                "recommended_route": decision.get("recommended_route"),
                "selector_replay_status": decision.get("selector_replay_status"),
                "selector_replay_report": selector_replay.get("report_path"),
                "selector_replay_manifest": (
                    selector_replay.get("manifest_path")
                    or manifests.get("selector_replay_manifest")
                ),
                "selector_replay_recommended_policy_path": selector_replay.get(
                    "recommended_policy_path"
                ),
                "selector_replay_recommended": selector_replay_recommended,
                "selector_replay_estimated_cost_units_mean": selector_replay_recommended.get(
                    "estimated_cost_units_mean"
                ),
                "selector_replay_observed_runtime_coverage_rate": selector_replay_recommended.get(
                    "observed_runtime_coverage_rate"
                ),
                "selector_replay_observed_runtime_delta_coverage_rate": (
                    selector_replay_recommended.get("observed_runtime_delta_coverage_rate")
                ),
                "selector_replay_observed_selected_total_seconds_mean": (
                    selector_replay_recommended.get("observed_selected_total_seconds_mean")
                ),
                "selector_replay_observed_selected_minus_original_seconds_mean": (
                    selector_replay_recommended.get("observed_selected_minus_original_seconds_mean")
                ),
                "selector_replay_observed_selected_to_original_ratio_mean": (
                    selector_replay_recommended.get("observed_selected_to_original_ratio_mean")
                ),
                "product_runtime_drift_status": decision.get("product_runtime_drift_status"),
                "product_runtime_drift_report": product_runtime_drift.get("report_path"),
                "product_runtime_drift_manifest": (
                    product_runtime_drift.get("manifest_path")
                    or manifests.get("product_runtime_drift_manifest")
                ),
                "product_runtime_drift_baseline_path": product_runtime_drift_baseline.get("path"),
                "product_runtime_drift_current_path": product_runtime_drift_current.get("path"),
                "product_runtime_drift_gate_enabled": product_runtime_drift_summary.get("gate_enabled"),
                "product_runtime_drift_promotion_evidence_required": config.get(
                    "require_product_runtime_drift_promotion_evidence"
                ),
                "product_runtime_drift_pre_generation_evidence_required": config.get(
                    "require_product_runtime_drift_pre_generation_evidence"
                ),
                "product_runtime_drift_claim_factuality_evidence_required": config.get(
                    "require_product_runtime_drift_claim_factuality_evidence"
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
                "product_runtime_drift_trajectory_audit_evidence_required": config.get(
                    "require_product_runtime_drift_trajectory_audit_evidence"
                ),
                "product_runtime_drift_evidence_handoff_evidence_required": config.get(
                    "require_product_runtime_drift_evidence_handoff_evidence"
                ),
                "product_runtime_drift_compared_metric_count": (
                    product_runtime_drift_summary.get("compared_metric_count")
                ),
                "product_runtime_drift_blocked_metric_count": (
                    product_runtime_drift_summary.get("blocked_metric_count")
                ),
                **_product_runtime_drift_promotion_metadata(product_runtime_drift_summary),
                "runtime_profile": config.get("runtime_profile"),
                "inside_trigger_budget_policy": config.get("inside_trigger_budget_policy"),
                "runtime_profile_applied_defaults": config.get(
                    "runtime_profile_applied_defaults"
                ),
                "max_uncached_forward_seconds": config.get(
                    "max_uncached_forward_seconds"
                ),
                "max_recommended_runtime_seconds": config.get(
                    "max_recommended_runtime_seconds"
                ),
                "recommended_runtime_seconds": runtime_cost.get(
                    "recommended_runtime_seconds"
                ),
                "recommended_runtime_cost_source": runtime_cost.get(
                    "recommended_runtime_cost_source"
                ),
                "uncached_forward_cost_seconds": runtime_cost.get(
                    "uncached_forward_cost_seconds"
                ),
                "uncached_forward_cost_source": runtime_cost.get(
                    "uncached_forward_cost_source"
                ),
                "cache_only_total_seconds": runtime_cost.get("cache_only_total_seconds"),
                "readiness_manifest": manifests.get("readiness_manifest"),
                "route_manifest": manifests.get("route_manifest"),
                "performance_manifest": manifests.get("performance_manifest"),
                "adapter_family_matrix_report": adapter_family.get("matrix_path"),
                "adapter_family_routes": adapter_family.get("routes"),
                "adapter_family_promoted_routes": adapter_family.get("promoted_routes"),
                "adapter_family_required_routes": adapter_family.get("required_routes"),
                "adapter_family_promotion_status": adapter_family.get("promotion_status"),
                "adapter_family_matrix_manifest": manifests.get("adapter_family_matrix_report"),
                "required_route_baseline_status": decision.get("required_route_baseline_status"),
                "required_route_baseline_records": (
                    required_route_baselines.get("records")
                    or decision.get("required_route_baseline_records")
                ),
                "required_route_baseline_routes": required_route_baselines.get("routes"),
                "required_route_baseline_manifests": required_route_baselines.get("manifest_paths"),
                "required_route_baseline_registry": required_route_baselines.get("registry"),
                "required_route_baseline_covered_fact_property_counts": (
                    required_route_property_counts
                ),
                "required_route_baseline_covered_fact_properties": (
                    required_route_property_sets
                ),
                "required_route_baseline_covered_fact_property_metrics": (
                    required_route_property_metrics
                ),
                "structured_fact_robustness_required": structured_fact_robustness[
                    "required"
                ],
                "structured_fact_robustness_canonical_route_key": (
                    structured_fact_robustness["canonical_route_key"]
                ),
                "structured_fact_robustness_paraphrase_route_key": (
                    structured_fact_robustness["paraphrase_route_key"]
                ),
                "structured_fact_robustness_records": structured_fact_robustness[
                    "records"
                ],
                "structured_fact_robustness_routes": structured_fact_robustness[
                    "routes"
                ],
                "structured_fact_robustness_manifests": structured_fact_robustness[
                    "manifests"
                ],
                "structured_fact_robustness_property_counts": (
                    structured_fact_robustness["property_counts"]
                ),
                "structured_fact_robustness_properties": (
                    structured_fact_robustness["properties"]
                ),
                "structured_fact_robustness_property_metrics": (
                    structured_fact_robustness["property_metrics"]
                ),
                "required_route_budget_policy": _required_route_budget_policy(config),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable contract payload."""
        return {
            "schema_version": self.schema_version,
            "workflow": "product_promotion_contract",
            "source_workflow": self.source_workflow,
            "source_status": self.source_status,
            "model_id": self.model_id,
            "runtime": dict(self.runtime),
            "verifier_route": dict(self.verifier_route),
            "runtime_budget_policy": self.runtime_budget_policy.to_dict(),
            "control_policy_config": dict(self.control_policy_config),
            "control_defaults": dict(self.control_defaults),
            "product_trace_replay_workflow": dict(self.product_trace_replay_workflow),
            "selfcheck_signal_fusion_workflow": dict(self.selfcheck_signal_fusion_workflow),
            "world_model_signal_workflow": dict(self.world_model_signal_workflow),
            "pathway_intervention_workflow": dict(self.pathway_intervention_workflow),
            "feedback_policy_workflow": dict(self.feedback_policy_workflow),
            "external_evidence_baseline_comparison": dict(
                self.external_evidence_baseline_comparison
            ),
            "pre_generation_probe_comparison": dict(
                self.pre_generation_probe_comparison
            ),
            "claim_factuality_probe_comparison": dict(
                self.claim_factuality_probe_comparison
            ),
            "triple_extraction_fixture_matrix": dict(
                self.triple_extraction_fixture_matrix
            ),
            "counterfactual_verification": dict(self.counterfactual_verification),
            "release_efficiency": dict(self.release_efficiency),
            "metadata": dict(self.metadata),
        }

    def save_json(self, path: str | Path) -> None:
        """Write the contract payload to JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(strict_json_dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class LoadedProductPromotionContract:
    """Promotion contract plus the source used to load it."""

    contract: ProductPromotionContract
    source: str
    path: Path | None = None

    def runtime_metadata(self, *, budget_enabled: bool) -> dict[str, Any]:
        """Return ProductTrace metadata for this contract."""
        return product_promotion_contract_metadata(
            self.contract,
            source=self.source,
            budget_enabled=budget_enabled,
        )


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
        "product_runtime_drift_claim_factuality_evidence_metric_count": summary.get(
            "claim_factuality_evidence_metric_count"
        ),
        "product_runtime_drift_claim_factuality_evidence_blocked_metric_count": summary.get(
            "claim_factuality_evidence_blocked_metric_count"
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
        "product_runtime_drift_trajectory_audit_evidence_metric_count": summary.get(
            "trajectory_audit_evidence_metric_count"
        ),
        "product_runtime_drift_trajectory_audit_evidence_blocked_metric_count": summary.get(
            "trajectory_audit_evidence_blocked_metric_count"
        ),
        "product_runtime_drift_evidence_handoff_evidence_metric_count": summary.get(
            "evidence_handoff_evidence_metric_count"
        ),
        "product_runtime_drift_evidence_handoff_evidence_blocked_metric_count": summary.get(
            "evidence_handoff_evidence_blocked_metric_count"
        ),
    }
    for prefix in _PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_PREFIXES:
        for suffix in ("baseline", "current", "status"):
            metadata[f"product_runtime_drift_{prefix}_{suffix}"] = summary.get(f"{prefix}_{suffix}")
    for prefix in _PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_PREFIXES:
        for suffix in ("baseline", "current", "status"):
            metadata[f"product_runtime_drift_{prefix}_{suffix}"] = summary.get(f"{prefix}_{suffix}")
    for prefix in _PRODUCT_RUNTIME_DRIFT_CLAIM_FACTUALITY_EVIDENCE_PREFIXES:
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
    for prefix in _PRODUCT_RUNTIME_DRIFT_TRAJECTORY_AUDIT_EVIDENCE_PREFIXES:
        for suffix in ("baseline", "current", "status"):
            metadata[f"product_runtime_drift_{prefix}_{suffix}"] = summary.get(f"{prefix}_{suffix}")
    for prefix in _PRODUCT_RUNTIME_DRIFT_EVIDENCE_HANDOFF_EVIDENCE_PREFIXES:
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
    property_counts_by_record = _mapping(
        required_route_baselines.get("covered_fact_property_counts")
    )
    properties_by_record = _mapping(
        required_route_baselines.get("covered_fact_properties")
    )
    property_metrics_by_record = _mapping(
        required_route_baselines.get("covered_fact_property_metrics")
    )
    by_record: dict[str, dict[str, Any]] = {}
    for idx, record in enumerate(records):
        key = str(record)
        by_record[key] = {
            "route": routes[idx] if idx < len(routes) else None,
            "manifest": manifests[idx] if idx < len(manifests) else None,
            "property_count": property_counts_by_record.get(key),
            "properties": properties_by_record.get(key),
            "property_metrics": property_metrics_by_record.get(key),
        }

    selected_records: list[str] = []
    selected_routes: list[Any] = []
    selected_manifests: list[Any] = []
    selected_property_counts: dict[str, Any] = {}
    selected_properties: dict[str, Any] = {}
    selected_property_metrics: dict[str, Any] = {}
    for key in target_keys:
        entry = by_record.get(key)
        if entry is None:
            continue
        selected_records.append(key)
        selected_routes.append(entry["route"])
        selected_manifests.append(entry["manifest"])
        selected_property_counts[key] = entry["property_count"]
        selected_properties[key] = entry["properties"]
        selected_property_metrics[key] = entry["property_metrics"]

    return {
        "required": bool(config.get("require_structured_fact_robustness")),
        "canonical_route_key": canonical_key,
        "paraphrase_route_key": paraphrase_key,
        "records": selected_records,
        "routes": selected_routes,
        "manifests": selected_manifests,
        "property_counts": selected_property_counts,
        "properties": selected_properties,
        "property_metrics": selected_property_metrics,
    }


def first_existing_product_promotion_contract_path(
    paths: Iterable[str | Path],
) -> Path | None:
    """Return the first existing promotion contract path from an ordered candidate list."""
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None


def load_product_promotion_contract(
    path: str | Path | None = None,
    *,
    default_paths: Iterable[str | Path] = (),
    require_promoted: bool = True,
) -> LoadedProductPromotionContract | None:
    """Load an explicit contract or the first existing default contract.

    Explicit paths are loaded directly and keep normal file errors. When no
    explicit path is supplied and none of the defaults exist, return ``None``.
    """
    resolved_path = (
        Path(path)
        if path is not None
        else first_existing_product_promotion_contract_path(default_paths)
    )
    if resolved_path is None:
        return None
    return LoadedProductPromotionContract(
        contract=ProductPromotionContract.from_json(
            resolved_path,
            require_promoted=require_promoted,
        ),
        source=str(resolved_path),
        path=resolved_path,
    )


@dataclass(frozen=True)
class ProductRuntimeEvidenceBundle:
    """Lazy runtime evidence bundle for a deployable promotion contract."""

    loaded_contract: LoadedProductPromotionContract
    manifest_path: Path | None = None
    evidence_handoff_manifest_path: Path | None = None
    registry_path: Path | None = None
    registry_key: str | None = None
    manifest_recursive: bool = True
    _manifest_verification: ArtifactManifestVerification | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _registry_record: RegistryRecord | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _evidence_handoff_manifest_payload: Mapping[str, Any] | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _evidence_handoff_manifest_verification: ArtifactManifestVerification | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _evidence_handoff_audit_payload: Mapping[str, Any] | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _selfcheck_signal_fusion_manifest_verification: ArtifactManifestVerification | None = (
        field(
            default=None,
            init=False,
            compare=False,
            repr=False,
        )
    )
    _selfcheck_signal_fusion_registry_record: RegistryRecord | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _world_model_signal_workflow_manifest_verification: ArtifactManifestVerification | None = (
        field(
            default=None,
            init=False,
            compare=False,
            repr=False,
        )
    )
    _world_model_signal_workflow_registry_record: RegistryRecord | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _pathway_intervention_workflow_manifest_verification: (
        ArtifactManifestVerification | None
    ) = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _pathway_intervention_workflow_registry_record: RegistryRecord | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _external_evidence_baseline_comparison_registry_record: RegistryRecord | None = (
        field(
            default=None,
            init=False,
            compare=False,
            repr=False,
        )
    )
    _pre_generation_probe_comparison_manifest_verification: (
        ArtifactManifestVerification | None
    ) = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _pre_generation_probe_comparison_registry_record: RegistryRecord | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _claim_factuality_probe_comparison_manifest_verification: (
        ArtifactManifestVerification | None
    ) = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _claim_factuality_probe_comparison_registry_record: RegistryRecord | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _triple_extraction_fixture_matrix_manifest_verification: (
        ArtifactManifestVerification | None
    ) = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _triple_extraction_fixture_matrix_registry_record: RegistryRecord | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _counterfactual_verification_manifest_verification: (
        ArtifactManifestVerification | None
    ) = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _counterfactual_verification_registry_record: RegistryRecord | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )

    @property
    def contract(self) -> ProductPromotionContract:
        """Return the loaded promotion contract."""
        return self.loaded_contract.contract

    @property
    def source(self) -> str:
        """Return the promotion contract source path or label."""
        return self.loaded_contract.source

    @property
    def contract_path(self) -> Path | None:
        """Return the local contract path when it came from a file."""
        return self.loaded_contract.path

    def verify_manifest(self) -> ArtifactManifestVerification | None:
        """Lazily verify the optional artifact manifest."""
        if self.manifest_path is None:
            return None
        if self._manifest_verification is None:
            object.__setattr__(
                self,
                "_manifest_verification",
                load_and_verify_artifact_manifest(
                    self.manifest_path,
                    recursive=self.manifest_recursive,
                ),
            )
        return self._manifest_verification

    def registry_record(self) -> RegistryRecord | None:
        """Lazily resolve the optional registry record."""
        if self.registry_path is None:
            return None
        if self._registry_record is None:
            registry = ArtifactRegistry.load_json(self.registry_path)
            record = (
                registry.get(self.registry_key)
                if self.registry_key is not None
                else _find_product_promotion_contract_record(
                    registry,
                    contract_path=self.contract_path,
                    source=self.source,
                )
            )
            object.__setattr__(self, "_registry_record", record)
        return self._registry_record

    def verify_evidence_handoff_manifest(self) -> ArtifactManifestVerification | None:
        """Lazily verify the optional enriched handoff artifact manifest."""
        if self.evidence_handoff_manifest_path is None:
            return None
        if self._evidence_handoff_manifest_verification is None:
            object.__setattr__(
                self,
                "_evidence_handoff_manifest_verification",
                load_and_verify_artifact_manifest(
                    self.evidence_handoff_manifest_path,
                    recursive=self.manifest_recursive,
                ),
            )
        return self._evidence_handoff_manifest_verification

    def evidence_handoff_manifest_payload(self) -> Mapping[str, Any] | None:
        """Return the parsed enriched handoff manifest payload, if available."""
        if self.evidence_handoff_manifest_path is None:
            return None
        if self._evidence_handoff_manifest_payload is None:
            payload = json.loads(
                self.evidence_handoff_manifest_path.read_text(encoding="utf-8")
            )
            if not isinstance(payload, Mapping):
                raise ValueError("promotion handoff manifest JSON must contain an object.")
            object.__setattr__(self, "_evidence_handoff_manifest_payload", payload)
        return self._evidence_handoff_manifest_payload

    @property
    def evidence_handoff_contract_path(self) -> Path | None:
        """Return the enriched handoff contract path referenced by its manifest."""
        return _artifact_manifest_entry_path(
            self.evidence_handoff_manifest_path,
            self.evidence_handoff_manifest_payload(),
            "product_promotion_contract_evidence_handoff",
        )

    @property
    def evidence_handoff_audit_path(self) -> Path | None:
        """Return the enriched handoff audit path referenced by its manifest."""
        return _artifact_manifest_entry_path(
            self.evidence_handoff_manifest_path,
            self.evidence_handoff_manifest_payload(),
            "product_promotion_contract_evidence_handoff_audit",
        )

    def evidence_handoff_audit_payload(self) -> Mapping[str, Any] | None:
        """Return the parsed enriched handoff audit payload, if available."""
        audit_path = self.evidence_handoff_audit_path
        if audit_path is None:
            return None
        if self._evidence_handoff_audit_payload is None:
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("promotion handoff audit JSON must contain an object.")
            object.__setattr__(self, "_evidence_handoff_audit_payload", payload)
        return self._evidence_handoff_audit_payload

    def evidence_handoff_metadata(
        self,
        *,
        verify_manifest: bool = False,
    ) -> dict[str, Any]:
        """Return JSON-ready enriched handoff provenance metadata."""
        manifest_payload = self.evidence_handoff_manifest_payload()
        manifest_metadata = _mapping(
            None if manifest_payload is None else manifest_payload.get("metadata")
        )
        manifest_summary = _mapping(
            None if manifest_payload is None else manifest_payload.get("summary")
        )
        audit_payload = self.evidence_handoff_audit_payload()
        audit_summary = _mapping(
            None if audit_payload is None else audit_payload.get("summary")
        )
        group_statuses = _mapping(audit_summary.get("groups"))
        manifest_verification = (
            self.verify_evidence_handoff_manifest() if verify_manifest else None
        )
        return {
            "promotion_contract_evidence_handoff_manifest": (
                None
                if self.evidence_handoff_manifest_path is None
                else str(self.evidence_handoff_manifest_path)
            ),
            "promotion_contract_evidence_handoff_manifest_verification": (
                None if manifest_verification is None else manifest_verification.to_dict()
            ),
            "promotion_contract_evidence_handoff_manifest_summary": (
                None if not manifest_summary else dict(manifest_summary)
            ),
            "promotion_contract_evidence_handoff_manifest_metadata": (
                None if not manifest_metadata else dict(manifest_metadata)
            ),
            "promotion_contract_evidence_handoff_contract": (
                None
                if self.evidence_handoff_contract_path is None
                else str(self.evidence_handoff_contract_path)
            ),
            "promotion_contract_evidence_handoff_audit": (
                None
                if self.evidence_handoff_audit_path is None
                else str(self.evidence_handoff_audit_path)
            ),
            "promotion_contract_evidence_handoff_workflow": (
                None if audit_payload is None else audit_payload.get("workflow")
            ),
            "promotion_contract_evidence_handoff_status": _first_present(
                None if audit_payload is None else audit_payload.get("status"),
                manifest_metadata.get("status"),
            ),
            "promotion_contract_evidence_handoff_before_missing_metric_count": (
                manifest_metadata.get("before_missing_metric_count")
            ),
            "promotion_contract_evidence_handoff_after_missing_metric_count": (
                manifest_metadata.get("after_missing_metric_count")
            ),
            "promotion_contract_evidence_handoff_resolved_missing_metric_count": (
                manifest_metadata.get("resolved_missing_metric_count")
            ),
            "promotion_contract_evidence_handoff_filled_groups": (
                manifest_metadata.get("filled_groups")
            ),
            "promotion_contract_evidence_handoff_expected_metric_count": (
                audit_summary.get("expected_metric_count")
            ),
            "promotion_contract_evidence_handoff_present_metric_count": (
                audit_summary.get("present_metric_count")
            ),
            "promotion_contract_evidence_handoff_missing_metric_count": (
                audit_summary.get("missing_metric_count")
            ),
            "promotion_contract_evidence_handoff_blocked_group_count": (
                audit_summary.get("blocked_group_count")
            ),
            "promotion_contract_evidence_handoff_group_statuses": dict(group_statuses),
        }

    def evidence_metadata(
        self,
        *,
        verify_manifest: bool = False,
        include_registry_record: bool = True,
    ) -> dict[str, Any]:
        """Return JSON-ready provenance metadata for this evidence bundle."""
        manifest_verification = self.verify_manifest() if verify_manifest else None
        registry_record = self.registry_record() if include_registry_record else None
        return {
            "promotion_contract_manifest": (
                None if self.manifest_path is None else str(self.manifest_path)
            ),
            "promotion_contract_manifest_verification": (
                None if manifest_verification is None else manifest_verification.to_dict()
            ),
            "promotion_contract_registry": (
                None if self.registry_path is None else str(self.registry_path)
            ),
            "promotion_contract_registry_key": (
                None if registry_record is None else registry_record.key()
            ),
            "promotion_contract_registry_record": (
                None if registry_record is None else registry_record.to_dict()
            ),
        }

    @property
    def selfcheck_signal_fusion_workflow(self) -> Mapping[str, Any]:
        """Return the selfcheck signal-fusion workflow contract, if present."""
        return self.contract.selfcheck_signal_fusion_workflow

    @property
    def selfcheck_signal_fusion_report_path(self) -> Path | None:
        """Return the selfcheck signal-fusion workflow report path."""
        return _resolve_contract_metadata_path(
            self.selfcheck_signal_fusion_workflow.get("report_path"),
            contract_path=self.contract_path,
        )

    @property
    def selfcheck_signal_fusion_manifest_path(self) -> Path | None:
        """Return the selfcheck signal-fusion workflow manifest path."""
        return _resolve_contract_metadata_path(
            self.selfcheck_signal_fusion_workflow.get("manifest_path"),
            contract_path=self.contract_path,
        )

    @property
    def selfcheck_signal_fusion_registry_path(self) -> Path | None:
        """Return the selfcheck signal-fusion workflow registry path."""
        return _resolve_contract_metadata_path(
            self.selfcheck_signal_fusion_workflow.get("registry"),
            contract_path=self.contract_path,
        )

    def verify_selfcheck_signal_fusion_manifest(
        self,
    ) -> ArtifactManifestVerification | None:
        """Lazily verify the optional selfcheck signal-fusion workflow manifest."""
        manifest_path = self.selfcheck_signal_fusion_manifest_path
        if manifest_path is None:
            return None
        if self._selfcheck_signal_fusion_manifest_verification is None:
            object.__setattr__(
                self,
                "_selfcheck_signal_fusion_manifest_verification",
                load_and_verify_artifact_manifest(
                    manifest_path,
                    recursive=self.manifest_recursive,
                ),
            )
        return self._selfcheck_signal_fusion_manifest_verification

    def selfcheck_signal_fusion_registry_record(self) -> RegistryRecord | None:
        """Lazily resolve the optional selfcheck signal-fusion workflow registry record."""
        registry_path = self.selfcheck_signal_fusion_registry_path
        record_key = self.selfcheck_signal_fusion_workflow.get("record_key")
        if registry_path is None or record_key is None:
            return None
        if self._selfcheck_signal_fusion_registry_record is None:
            registry = ArtifactRegistry.load_json(registry_path)
            object.__setattr__(
                self,
                "_selfcheck_signal_fusion_registry_record",
                registry.get(str(record_key)),
            )
        return self._selfcheck_signal_fusion_registry_record

    def selfcheck_signal_fusion_evidence_metadata(
        self,
        *,
        verify_manifest: bool = False,
        include_registry_record: bool = False,
    ) -> dict[str, Any]:
        """Return JSON-ready selfcheck signal-fusion provenance metadata."""
        workflow = self.selfcheck_signal_fusion_workflow
        report_path = self.selfcheck_signal_fusion_report_path
        manifest_path = self.selfcheck_signal_fusion_manifest_path
        registry_path = self.selfcheck_signal_fusion_registry_path
        manifest_verification = (
            self.verify_selfcheck_signal_fusion_manifest() if verify_manifest else None
        )
        record_key = workflow.get("record_key")
        registry_record = (
            self.selfcheck_signal_fusion_registry_record()
            if include_registry_record
            else None
        )
        return {
            "selfcheck_signal_fusion_workflow_report": (
                None if report_path is None else str(report_path)
            ),
            "selfcheck_signal_fusion_workflow_manifest": (
                None if manifest_path is None else str(manifest_path)
            ),
            "selfcheck_signal_fusion_workflow_manifest_verification": (
                None if manifest_verification is None else manifest_verification.to_dict()
            ),
            "selfcheck_signal_fusion_workflow_registry": (
                None if registry_path is None else str(registry_path)
            ),
            "selfcheck_signal_fusion_workflow_registry_key": (
                None if record_key is None else str(record_key)
            ),
            "selfcheck_signal_fusion_workflow_registry_record": (
                None if registry_record is None else registry_record.to_dict()
            ),
            "selfcheck_signal_fusion_workflow_status": workflow.get("status"),
            "selfcheck_signal_fusion_workflow_sample_quality_status": workflow.get(
                "sample_quality_status"
            ),
            "selfcheck_signal_fusion_workflow_sample_quality_passed": workflow.get(
                "sample_quality_passed"
            ),
            "selfcheck_signal_fusion_workflow_fusion_run_count": workflow.get(
                "fusion_run_count"
            ),
            "selfcheck_signal_fusion_workflow_geometry_artifact_count": workflow.get(
                "geometry_fusion_artifact_count"
            ),
            "selfcheck_signal_fusion_workflow_enhanced_score_dump_count": workflow.get(
                "enhanced_score_dump_count"
            ),
        }

    @property
    def world_model_signal_workflow(self) -> Mapping[str, Any]:
        """Return the world-model signal workflow contract, if present."""
        return self.contract.world_model_signal_workflow

    @property
    def world_model_signal_workflow_report_path(self) -> Path | None:
        """Return the world-model signal workflow report path."""
        return _resolve_contract_metadata_path(
            self.world_model_signal_workflow.get("report_path"),
            contract_path=self.contract_path,
        )

    @property
    def world_model_signal_workflow_manifest_path(self) -> Path | None:
        """Return the world-model signal workflow manifest path."""
        return _resolve_contract_metadata_path(
            self.world_model_signal_workflow.get("manifest_path"),
            contract_path=self.contract_path,
        )

    @property
    def world_model_signal_workflow_registry_path(self) -> Path | None:
        """Return the world-model signal workflow registry path."""
        return _resolve_contract_metadata_path(
            self.world_model_signal_workflow.get("registry"),
            contract_path=self.contract_path,
        )

    def verify_world_model_signal_workflow_manifest(
        self,
    ) -> ArtifactManifestVerification | None:
        """Lazily verify the optional world-model signal workflow manifest."""
        manifest_path = self.world_model_signal_workflow_manifest_path
        if manifest_path is None:
            return None
        if self._world_model_signal_workflow_manifest_verification is None:
            object.__setattr__(
                self,
                "_world_model_signal_workflow_manifest_verification",
                load_and_verify_artifact_manifest(
                    manifest_path,
                    recursive=self.manifest_recursive,
                ),
            )
        return self._world_model_signal_workflow_manifest_verification

    def world_model_signal_workflow_registry_record(self) -> RegistryRecord | None:
        """Lazily resolve the optional world-model signal workflow registry record."""
        registry_path = self.world_model_signal_workflow_registry_path
        record_key = self.world_model_signal_workflow.get("record_key")
        if registry_path is None or record_key is None:
            return None
        if self._world_model_signal_workflow_registry_record is None:
            registry = ArtifactRegistry.load_json(registry_path)
            object.__setattr__(
                self,
                "_world_model_signal_workflow_registry_record",
                registry.get(str(record_key)),
            )
        return self._world_model_signal_workflow_registry_record

    def world_model_signal_evidence_metadata(
        self,
        *,
        verify_manifest: bool = False,
        include_registry_record: bool = False,
    ) -> dict[str, Any]:
        """Return JSON-ready world-model signal provenance metadata."""
        workflow = self.world_model_signal_workflow
        report_path = self.world_model_signal_workflow_report_path
        manifest_path = self.world_model_signal_workflow_manifest_path
        registry_path = self.world_model_signal_workflow_registry_path
        manifest_verification = (
            self.verify_world_model_signal_workflow_manifest() if verify_manifest else None
        )
        record_key = workflow.get("record_key")
        registry_record = (
            self.world_model_signal_workflow_registry_record()
            if include_registry_record
            else None
        )
        return {
            "world_model_signal_workflow_report": (
                None if report_path is None else str(report_path)
            ),
            "world_model_signal_workflow_manifest": (
                None if manifest_path is None else str(manifest_path)
            ),
            "world_model_signal_workflow_manifest_verification": (
                None if manifest_verification is None else manifest_verification.to_dict()
            ),
            "world_model_signal_workflow_registry": (
                None if registry_path is None else str(registry_path)
            ),
            "world_model_signal_workflow_registry_key": (
                None if record_key is None else str(record_key)
            ),
            "world_model_signal_workflow_registry_record": (
                None if registry_record is None else registry_record.to_dict()
            ),
            "world_model_signal_workflow_status": workflow.get("status"),
            "world_model_signal_workflow_release_gate_status": workflow.get(
                "release_gate_status"
            ),
            "world_model_signal_workflow_trace_gap_max": workflow.get("trace_gap_max"),
            "world_model_signal_workflow_conflict_positive_count": workflow.get(
                "conflict_positive_count"
            ),
            "world_model_signal_workflow_calibrated_conflict_signal_count": (
                workflow.get("calibrated_conflict_signal_count")
            ),
        }

    @property
    def pathway_intervention_workflow(self) -> Mapping[str, Any]:
        """Return the pathway-intervention workflow contract, if present."""
        return self.contract.pathway_intervention_workflow

    @property
    def pathway_intervention_workflow_report_path(self) -> Path | None:
        """Return the pathway-intervention workflow report path."""
        return _resolve_contract_metadata_path(
            self.pathway_intervention_workflow.get("report_path"),
            contract_path=self.contract_path,
        )

    @property
    def pathway_intervention_workflow_manifest_path(self) -> Path | None:
        """Return the pathway-intervention workflow manifest path."""
        return _resolve_contract_metadata_path(
            self.pathway_intervention_workflow.get("manifest_path"),
            contract_path=self.contract_path,
        )

    @property
    def pathway_intervention_workflow_registry_path(self) -> Path | None:
        """Return the pathway-intervention workflow registry path."""
        return _resolve_contract_metadata_path(
            self.pathway_intervention_workflow.get("registry"),
            contract_path=self.contract_path,
        )

    def verify_pathway_intervention_workflow_manifest(
        self,
    ) -> ArtifactManifestVerification | None:
        """Lazily verify the optional pathway-intervention workflow manifest."""
        manifest_path = self.pathway_intervention_workflow_manifest_path
        if manifest_path is None:
            return None
        if self._pathway_intervention_workflow_manifest_verification is None:
            object.__setattr__(
                self,
                "_pathway_intervention_workflow_manifest_verification",
                load_and_verify_artifact_manifest(
                    manifest_path,
                    recursive=self.manifest_recursive,
                ),
            )
        return self._pathway_intervention_workflow_manifest_verification

    def pathway_intervention_workflow_registry_record(self) -> RegistryRecord | None:
        """Lazily resolve the optional pathway-intervention workflow registry record."""
        registry_path = self.pathway_intervention_workflow_registry_path
        record_key = self.pathway_intervention_workflow.get("record_key")
        if registry_path is None or record_key is None:
            return None
        if self._pathway_intervention_workflow_registry_record is None:
            registry = ArtifactRegistry.load_json(registry_path)
            object.__setattr__(
                self,
                "_pathway_intervention_workflow_registry_record",
                registry.get(str(record_key)),
            )
        return self._pathway_intervention_workflow_registry_record

    def pathway_intervention_evidence_metadata(
        self,
        *,
        verify_manifest: bool = False,
        include_registry_record: bool = False,
    ) -> dict[str, Any]:
        """Return JSON-ready pathway-intervention provenance metadata."""
        workflow = self.pathway_intervention_workflow
        report_path = self.pathway_intervention_workflow_report_path
        manifest_path = self.pathway_intervention_workflow_manifest_path
        registry_path = self.pathway_intervention_workflow_registry_path
        manifest_verification = (
            self.verify_pathway_intervention_workflow_manifest()
            if verify_manifest
            else None
        )
        record_key = workflow.get("record_key")
        registry_record = (
            self.pathway_intervention_workflow_registry_record()
            if include_registry_record
            else None
        )
        return {
            "pathway_intervention_workflow_report": (
                None if report_path is None else str(report_path)
            ),
            "pathway_intervention_workflow_manifest": (
                None if manifest_path is None else str(manifest_path)
            ),
            "pathway_intervention_workflow_manifest_verification": (
                None if manifest_verification is None else manifest_verification.to_dict()
            ),
            "pathway_intervention_workflow_registry": (
                None if registry_path is None else str(registry_path)
            ),
            "pathway_intervention_workflow_registry_key": (
                None if record_key is None else str(record_key)
            ),
            "pathway_intervention_workflow_registry_record": (
                None if registry_record is None else registry_record.to_dict()
            ),
            "pathway_intervention_workflow_status": workflow.get("status"),
            "pathway_intervention_workflow_report_status": workflow.get("report_status"),
            "pathway_intervention_workflow_release_ready": workflow.get("release_ready"),
            "pathway_intervention_workflow_model": workflow.get("model"),
            "pathway_intervention_workflow_layer": workflow.get("layer"),
            "pathway_intervention_workflow_intervention_layer": workflow.get(
                "intervention_layer"
            ),
            "pathway_intervention_workflow_patch_layer": workflow.get("patch_layer"),
            "pathway_intervention_workflow_activation_ablation_gate": workflow.get(
                "activation_ablation_gate_status"
            ),
            "pathway_intervention_workflow_source_patch_gate": workflow.get(
                "source_patch_gate_status"
            ),
            "pathway_intervention_workflow_signals": workflow.get("signals"),
            "pathway_intervention_workflow_best_signals": workflow.get("best_signals"),
        }

    @property
    def external_evidence_baseline_comparison(self) -> Mapping[str, Any]:
        """Return the external-evidence baseline-comparison contract, if present."""
        return self.contract.external_evidence_baseline_comparison

    @property
    def external_evidence_baseline_comparison_report_path(self) -> Path | None:
        """Return the external-evidence baseline-comparison report path."""
        return _resolve_contract_metadata_path(
            self.external_evidence_baseline_comparison.get("report_path"),
            contract_path=self.contract_path,
        )

    @property
    def external_evidence_baseline_comparison_registry_path(self) -> Path | None:
        """Return the external-evidence baseline-comparison registry path."""
        return _resolve_contract_metadata_path(
            self.external_evidence_baseline_comparison.get("registry"),
            contract_path=self.contract_path,
        )

    def external_evidence_baseline_comparison_registry_record(
        self,
    ) -> RegistryRecord | None:
        """Lazily resolve the optional external-evidence comparison record."""
        registry_path = self.external_evidence_baseline_comparison_registry_path
        record_key = self.external_evidence_baseline_comparison.get("record_key")
        if registry_path is None or record_key is None:
            return None
        if self._external_evidence_baseline_comparison_registry_record is None:
            registry = ArtifactRegistry.load_json(registry_path)
            object.__setattr__(
                self,
                "_external_evidence_baseline_comparison_registry_record",
                registry.get(str(record_key)),
            )
        return self._external_evidence_baseline_comparison_registry_record

    def external_evidence_baseline_comparison_evidence_metadata(
        self,
        *,
        include_registry_record: bool = False,
    ) -> dict[str, Any]:
        """Return JSON-ready external-evidence comparison provenance metadata."""
        comparison = self.external_evidence_baseline_comparison
        report_path = self.external_evidence_baseline_comparison_report_path
        registry_path = self.external_evidence_baseline_comparison_registry_path
        record_key = comparison.get("record_key")
        registry_record = (
            self.external_evidence_baseline_comparison_registry_record()
            if include_registry_record
            else None
        )
        return {
            "external_evidence_baseline_comparison_report": (
                None if report_path is None else str(report_path)
            ),
            "external_evidence_baseline_comparison_registry": (
                None if registry_path is None else str(registry_path)
            ),
            "external_evidence_baseline_comparison_registry_key": (
                None if record_key is None else str(record_key)
            ),
            "external_evidence_baseline_comparison_registry_record": (
                None if registry_record is None else registry_record.to_dict()
            ),
            "external_evidence_baseline_comparison_status": comparison.get("status"),
            "external_evidence_baseline_comparison_decision_status": (
                comparison.get("decision_status")
            ),
            "external_evidence_baseline_comparison_recommended_route": (
                comparison.get("recommended_route")
            ),
            "external_evidence_baseline_comparison_recommended_route_record": (
                comparison.get("recommended_route_record")
            ),
            "external_evidence_baseline_comparison_route_passed": (
                comparison.get("route_passed")
            ),
            "external_evidence_baseline_comparison_text_redline_passed": (
                comparison.get("text_redline_passed")
            ),
            "external_evidence_baseline_comparison_text_redline_run_count": (
                comparison.get("text_redline_run_count")
            ),
        }

    @property
    def pre_generation_probe_comparison(self) -> Mapping[str, Any]:
        """Return the pre-generation probe comparison contract, if present."""
        return self.contract.pre_generation_probe_comparison

    @property
    def pre_generation_probe_comparison_report_path(self) -> Path | None:
        """Return the pre-generation probe comparison report path."""
        return _resolve_contract_metadata_path(
            self.pre_generation_probe_comparison.get("report_path"),
            contract_path=self.contract_path,
        )

    @property
    def pre_generation_probe_comparison_manifest_path(self) -> Path | None:
        """Return the pre-generation probe comparison manifest path."""
        return _resolve_contract_metadata_path(
            self.pre_generation_probe_comparison.get("manifest_path"),
            contract_path=self.contract_path,
        )

    @property
    def pre_generation_probe_comparison_registry_path(self) -> Path | None:
        """Return the pre-generation probe comparison registry path."""
        return _resolve_contract_metadata_path(
            self.pre_generation_probe_comparison.get("registry"),
            contract_path=self.contract_path,
        )

    def verify_pre_generation_probe_comparison_manifest(
        self,
    ) -> ArtifactManifestVerification | None:
        """Lazily verify the optional pre-generation probe comparison manifest."""
        manifest_path = self.pre_generation_probe_comparison_manifest_path
        if manifest_path is None:
            return None
        if self._pre_generation_probe_comparison_manifest_verification is None:
            object.__setattr__(
                self,
                "_pre_generation_probe_comparison_manifest_verification",
                load_and_verify_artifact_manifest(
                    manifest_path,
                    recursive=self.manifest_recursive,
                ),
            )
        return self._pre_generation_probe_comparison_manifest_verification

    def pre_generation_probe_comparison_registry_record(
        self,
    ) -> RegistryRecord | None:
        """Lazily resolve the optional pre-generation probe comparison record."""
        registry_path = self.pre_generation_probe_comparison_registry_path
        record_key = self.pre_generation_probe_comparison.get("record_key")
        if registry_path is None or record_key is None:
            return None
        if self._pre_generation_probe_comparison_registry_record is None:
            registry = ArtifactRegistry.load_json(registry_path)
            object.__setattr__(
                self,
                "_pre_generation_probe_comparison_registry_record",
                registry.get(str(record_key)),
            )
        return self._pre_generation_probe_comparison_registry_record

    def pre_generation_probe_comparison_evidence_metadata(
        self,
        *,
        verify_manifest: bool = False,
        include_registry_record: bool = False,
    ) -> dict[str, Any]:
        """Return JSON-ready pre-generation probe comparison provenance metadata."""
        comparison = self.pre_generation_probe_comparison
        report_path = self.pre_generation_probe_comparison_report_path
        manifest_path = self.pre_generation_probe_comparison_manifest_path
        registry_path = self.pre_generation_probe_comparison_registry_path
        manifest_verification = (
            self.verify_pre_generation_probe_comparison_manifest()
            if verify_manifest
            else None
        )
        record_key = comparison.get("record_key")
        registry_record = (
            self.pre_generation_probe_comparison_registry_record()
            if include_registry_record
            else None
        )
        best_run = _mapping(comparison.get("best_run"))
        return {
            "pre_generation_probe_comparison_report": (
                None if report_path is None else str(report_path)
            ),
            "pre_generation_probe_comparison_manifest": (
                None if manifest_path is None else str(manifest_path)
            ),
            "pre_generation_probe_comparison_manifest_verification": (
                None if manifest_verification is None else manifest_verification.to_dict()
            ),
            "pre_generation_probe_comparison_registry": (
                None if registry_path is None else str(registry_path)
            ),
            "pre_generation_probe_comparison_registry_key": (
                None if record_key is None else str(record_key)
            ),
            "pre_generation_probe_comparison_registry_record": (
                None if registry_record is None else registry_record.to_dict()
            ),
            "pre_generation_probe_comparison_status": comparison.get("status"),
            "pre_generation_probe_comparison_model_count": comparison.get("model_count"),
            "pre_generation_probe_comparison_run_count": comparison.get("run_count"),
            "pre_generation_probe_comparison_redline_passed": comparison.get(
                "redline_passed"
            ),
            "pre_generation_probe_comparison_redline_run_count": comparison.get(
                "redline_run_count"
            ),
            "pre_generation_probe_comparison_best_run": best_run.get("name"),
            "pre_generation_probe_comparison_best_model": best_run.get("model"),
            "pre_generation_probe_comparison_best_layer": best_run.get(
                "recommended_layer"
            ),
            "pre_generation_probe_comparison_best_test_label_auroc": best_run.get(
                "test_label_auroc"
            ),
            "pre_generation_probe_comparison_best_redline_signal": best_run.get(
                "redline_best_signal"
            ),
            "pre_generation_probe_comparison_best_redline_auroc": best_run.get(
                "redline_best_auroc"
            ),
            "pre_generation_probe_comparison_best_redline_margin": best_run.get(
                "redline_margin"
            ),
        }

    @property
    def claim_factuality_probe_comparison(self) -> Mapping[str, Any]:
        """Return the claim-factuality probe comparison contract, if present."""
        return self.contract.claim_factuality_probe_comparison

    @property
    def claim_factuality_probe_comparison_report_path(self) -> Path | None:
        """Return the claim-factuality probe comparison report path."""
        return _resolve_contract_metadata_path(
            self.claim_factuality_probe_comparison.get("report_path"),
            contract_path=self.contract_path,
        )

    @property
    def claim_factuality_probe_comparison_manifest_path(self) -> Path | None:
        """Return the claim-factuality probe comparison manifest path."""
        return _resolve_contract_metadata_path(
            self.claim_factuality_probe_comparison.get("manifest_path"),
            contract_path=self.contract_path,
        )

    @property
    def claim_factuality_probe_comparison_registry_path(self) -> Path | None:
        """Return the claim-factuality probe comparison registry path."""
        return _resolve_contract_metadata_path(
            self.claim_factuality_probe_comparison.get("registry"),
            contract_path=self.contract_path,
        )

    def verify_claim_factuality_probe_comparison_manifest(
        self,
    ) -> ArtifactManifestVerification | None:
        """Lazily verify the optional claim-factuality probe comparison manifest."""
        manifest_path = self.claim_factuality_probe_comparison_manifest_path
        if manifest_path is None:
            return None
        if self._claim_factuality_probe_comparison_manifest_verification is None:
            object.__setattr__(
                self,
                "_claim_factuality_probe_comparison_manifest_verification",
                load_and_verify_artifact_manifest(
                    manifest_path,
                    recursive=self.manifest_recursive,
                ),
            )
        return self._claim_factuality_probe_comparison_manifest_verification

    def claim_factuality_probe_comparison_registry_record(
        self,
    ) -> RegistryRecord | None:
        """Lazily resolve the optional claim-factuality probe comparison record."""
        registry_path = self.claim_factuality_probe_comparison_registry_path
        record_key = self.claim_factuality_probe_comparison.get("record_key")
        if registry_path is None or record_key is None:
            return None
        if self._claim_factuality_probe_comparison_registry_record is None:
            registry = ArtifactRegistry.load_json(registry_path)
            object.__setattr__(
                self,
                "_claim_factuality_probe_comparison_registry_record",
                registry.get(str(record_key)),
            )
        return self._claim_factuality_probe_comparison_registry_record

    def claim_factuality_probe_comparison_evidence_metadata(
        self,
        *,
        verify_manifest: bool = False,
        include_registry_record: bool = False,
    ) -> dict[str, Any]:
        """Return JSON-ready claim-factuality probe comparison provenance metadata."""
        comparison = self.claim_factuality_probe_comparison
        report_path = self.claim_factuality_probe_comparison_report_path
        manifest_path = self.claim_factuality_probe_comparison_manifest_path
        registry_path = self.claim_factuality_probe_comparison_registry_path
        manifest_verification = (
            self.verify_claim_factuality_probe_comparison_manifest()
            if verify_manifest
            else None
        )
        record_key = comparison.get("record_key")
        registry_record = (
            self.claim_factuality_probe_comparison_registry_record()
            if include_registry_record
            else None
        )
        best_run = _mapping(comparison.get("best_run"))
        return {
            "claim_factuality_probe_comparison_report": (
                None if report_path is None else str(report_path)
            ),
            "claim_factuality_probe_comparison_manifest": (
                None if manifest_path is None else str(manifest_path)
            ),
            "claim_factuality_probe_comparison_manifest_verification": (
                None if manifest_verification is None else manifest_verification.to_dict()
            ),
            "claim_factuality_probe_comparison_registry": (
                None if registry_path is None else str(registry_path)
            ),
            "claim_factuality_probe_comparison_registry_key": (
                None if record_key is None else str(record_key)
            ),
            "claim_factuality_probe_comparison_registry_record": (
                None if registry_record is None else registry_record.to_dict()
            ),
            "claim_factuality_probe_comparison_status": comparison.get("status"),
            "claim_factuality_probe_comparison_report_status": comparison.get(
                "report_status"
            ),
            "claim_factuality_probe_comparison_model_count": comparison.get(
                "model_count"
            ),
            "claim_factuality_probe_comparison_run_count": comparison.get("run_count"),
            "claim_factuality_probe_comparison_redline_passed": comparison.get(
                "redline_passed"
            ),
            "claim_factuality_probe_comparison_redline_run_count": comparison.get(
                "redline_run_count"
            ),
            "claim_factuality_probe_comparison_best_run": best_run.get("name"),
            "claim_factuality_probe_comparison_best_model": best_run.get("model"),
            "claim_factuality_probe_comparison_best_record_count": best_run.get(
                "record_count"
            ),
            "claim_factuality_probe_comparison_best_layer": best_run.get(
                "recommended_layer"
            ),
            "claim_factuality_probe_comparison_best_test_label_auroc": best_run.get(
                "test_label_auroc"
            ),
            "claim_factuality_probe_comparison_best_test_selective_accuracy": (
                best_run.get("test_selective_accuracy")
            ),
            "claim_factuality_probe_comparison_best_test_selective_coverage": (
                best_run.get("test_selective_coverage")
            ),
            "claim_factuality_probe_comparison_best_conformal_threshold": best_run.get(
                "conformal_threshold"
            ),
            "claim_factuality_probe_comparison_best_redline_signal": best_run.get(
                "redline_best_signal"
            ),
            "claim_factuality_probe_comparison_best_redline_auroc": best_run.get(
                "redline_best_auroc"
            ),
            "claim_factuality_probe_comparison_best_redline_margin": best_run.get(
                "redline_margin"
            ),
        }

    @property
    def triple_extraction_fixture_matrix(self) -> Mapping[str, Any]:
        """Return the triple-extraction fixture-matrix contract, if present."""
        return self.contract.triple_extraction_fixture_matrix

    @property
    def triple_extraction_fixture_matrix_report_path(self) -> Path | None:
        """Return the triple-extraction fixture-matrix report path."""
        return _resolve_contract_metadata_path(
            self.triple_extraction_fixture_matrix.get("report_path"),
            contract_path=self.contract_path,
        )

    @property
    def triple_extraction_fixture_matrix_manifest_path(self) -> Path | None:
        """Return the triple-extraction fixture-matrix manifest path."""
        return _resolve_contract_metadata_path(
            self.triple_extraction_fixture_matrix.get("manifest_path"),
            contract_path=self.contract_path,
        )

    @property
    def triple_extraction_fixture_matrix_registry_path(self) -> Path | None:
        """Return the triple-extraction fixture-matrix registry path."""
        return _resolve_contract_metadata_path(
            self.triple_extraction_fixture_matrix.get("registry"),
            contract_path=self.contract_path,
        )

    def verify_triple_extraction_fixture_matrix_manifest(
        self,
    ) -> ArtifactManifestVerification | None:
        """Lazily verify the optional triple-extraction fixture-matrix manifest."""
        manifest_path = self.triple_extraction_fixture_matrix_manifest_path
        if manifest_path is None:
            return None
        if self._triple_extraction_fixture_matrix_manifest_verification is None:
            object.__setattr__(
                self,
                "_triple_extraction_fixture_matrix_manifest_verification",
                load_and_verify_artifact_manifest(
                    manifest_path,
                    recursive=self.manifest_recursive,
                ),
            )
        return self._triple_extraction_fixture_matrix_manifest_verification

    def triple_extraction_fixture_matrix_registry_record(self) -> RegistryRecord | None:
        """Lazily resolve the optional triple-extraction fixture-matrix record."""
        registry_path = self.triple_extraction_fixture_matrix_registry_path
        record_key = self.triple_extraction_fixture_matrix.get("record_key")
        if registry_path is None or record_key is None:
            return None
        if self._triple_extraction_fixture_matrix_registry_record is None:
            registry = ArtifactRegistry.load_json(registry_path)
            object.__setattr__(
                self,
                "_triple_extraction_fixture_matrix_registry_record",
                registry.get(str(record_key)),
            )
        return self._triple_extraction_fixture_matrix_registry_record

    def triple_extraction_fixture_matrix_evidence_metadata(
        self,
        *,
        verify_manifest: bool = False,
        include_registry_record: bool = False,
    ) -> dict[str, Any]:
        """Return JSON-ready triple-extraction fixture-matrix provenance metadata."""
        matrix = self.triple_extraction_fixture_matrix
        report_path = self.triple_extraction_fixture_matrix_report_path
        manifest_path = self.triple_extraction_fixture_matrix_manifest_path
        registry_path = self.triple_extraction_fixture_matrix_registry_path
        manifest_verification = (
            self.verify_triple_extraction_fixture_matrix_manifest()
            if verify_manifest
            else None
        )
        record_key = matrix.get("record_key")
        registry_record = (
            self.triple_extraction_fixture_matrix_registry_record()
            if include_registry_record
            else None
        )
        return {
            "triple_extraction_fixture_matrix_report": (
                None if report_path is None else str(report_path)
            ),
            "triple_extraction_fixture_matrix_manifest": (
                None if manifest_path is None else str(manifest_path)
            ),
            "triple_extraction_fixture_matrix_manifest_verification": (
                None if manifest_verification is None else manifest_verification.to_dict()
            ),
            "triple_extraction_fixture_matrix_registry": (
                None if registry_path is None else str(registry_path)
            ),
            "triple_extraction_fixture_matrix_registry_key": (
                None if record_key is None else str(record_key)
            ),
            "triple_extraction_fixture_matrix_registry_record": (
                None if registry_record is None else registry_record.to_dict()
            ),
            "triple_extraction_fixture_matrix_status": matrix.get("status"),
            "triple_extraction_fixture_matrix_n_corpora": matrix.get("n_corpora"),
            "triple_extraction_fixture_matrix_promoted_corpora": matrix.get(
                "promoted_corpora"
            ),
            "triple_extraction_fixture_matrix_distinct_predicate_count": matrix.get(
                "distinct_predicate_count"
            ),
            "triple_extraction_fixture_matrix_distinct_predicates": matrix.get(
                "distinct_predicates"
            ),
            "triple_extraction_fixture_matrix_mean_best_f1": matrix.get("mean_best_f1"),
            "triple_extraction_fixture_matrix_mean_f1_lift": matrix.get("mean_f1_lift"),
        }

    @property
    def counterfactual_verification(self) -> Mapping[str, Any]:
        """Return the counterfactual verifier-audit contract, if present."""
        return self.contract.counterfactual_verification

    @property
    def counterfactual_verification_report_path(self) -> Path | None:
        """Return the counterfactual verifier-audit report path."""
        return _resolve_contract_metadata_path(
            self.counterfactual_verification.get("report_path"),
            contract_path=self.contract_path,
        )

    @property
    def counterfactual_verification_manifest_path(self) -> Path | None:
        """Return the counterfactual verifier-audit manifest path."""
        return _resolve_contract_metadata_path(
            self.counterfactual_verification.get("manifest_path"),
            contract_path=self.contract_path,
        )

    @property
    def counterfactual_verification_registry_path(self) -> Path | None:
        """Return the counterfactual verifier-audit registry path."""
        return _resolve_contract_metadata_path(
            self.counterfactual_verification.get("registry"),
            contract_path=self.contract_path,
        )

    def verify_counterfactual_verification_manifest(
        self,
    ) -> ArtifactManifestVerification | None:
        """Lazily verify the optional counterfactual verifier-audit manifest."""
        manifest_path = self.counterfactual_verification_manifest_path
        if manifest_path is None:
            return None
        if self._counterfactual_verification_manifest_verification is None:
            object.__setattr__(
                self,
                "_counterfactual_verification_manifest_verification",
                load_and_verify_artifact_manifest(
                    manifest_path,
                    recursive=self.manifest_recursive,
                ),
            )
        return self._counterfactual_verification_manifest_verification

    def counterfactual_verification_registry_record(self) -> RegistryRecord | None:
        """Lazily resolve the optional counterfactual verifier-audit record."""
        registry_path = self.counterfactual_verification_registry_path
        record_key = self.counterfactual_verification.get("record_key")
        if registry_path is None or record_key is None:
            return None
        if self._counterfactual_verification_registry_record is None:
            registry = ArtifactRegistry.load_json(registry_path)
            object.__setattr__(
                self,
                "_counterfactual_verification_registry_record",
                registry.get(str(record_key)),
            )
        return self._counterfactual_verification_registry_record

    def counterfactual_verification_evidence_metadata(
        self,
        *,
        verify_manifest: bool = False,
        include_registry_record: bool = False,
    ) -> dict[str, Any]:
        """Return JSON-ready counterfactual verifier-audit provenance metadata."""
        audit = self.counterfactual_verification
        report_path = self.counterfactual_verification_report_path
        manifest_path = self.counterfactual_verification_manifest_path
        registry_path = self.counterfactual_verification_registry_path
        manifest_verification = (
            self.verify_counterfactual_verification_manifest()
            if verify_manifest
            else None
        )
        record_key = audit.get("record_key")
        registry_record = (
            self.counterfactual_verification_registry_record()
            if include_registry_record
            else None
        )
        return {
            "counterfactual_verification_report": (
                None if report_path is None else str(report_path)
            ),
            "counterfactual_verification_manifest": (
                None if manifest_path is None else str(manifest_path)
            ),
            "counterfactual_verification_manifest_verification": (
                None if manifest_verification is None else manifest_verification.to_dict()
            ),
            "counterfactual_verification_registry": (
                None if registry_path is None else str(registry_path)
            ),
            "counterfactual_verification_registry_key": (
                None if record_key is None else str(record_key)
            ),
            "counterfactual_verification_registry_record": (
                None if registry_record is None else registry_record.to_dict()
            ),
            "counterfactual_verification_source": audit.get("source"),
            "counterfactual_verification_status": audit.get("status"),
            "counterfactual_verification_workflow": audit.get("workflow"),
            "counterfactual_verification_record_count": audit.get("record_count"),
            "counterfactual_verification_pass_rate": audit.get("pass_rate"),
            "counterfactual_verification_false_invariance_rate": audit.get(
                "false_invariance_rate"
            ),
            "counterfactual_verification_flip_success_count": audit.get(
                "flip_success_count"
            ),
        }

    def runtime_metadata(
        self,
        *,
        budget_enabled: bool,
        verify_manifest: bool = False,
        verify_evidence_handoff_manifest: bool = False,
        include_registry_record: bool = True,
        verify_selfcheck_signal_fusion_manifest: bool = False,
        include_selfcheck_signal_fusion_record: bool = False,
        verify_world_model_signal_workflow_manifest: bool = False,
        include_world_model_signal_workflow_record: bool = False,
        verify_pathway_intervention_workflow_manifest: bool = False,
        include_pathway_intervention_workflow_record: bool = False,
        include_external_evidence_baseline_comparison_record: bool = False,
        verify_pre_generation_probe_comparison_manifest: bool = False,
        include_pre_generation_probe_comparison_record: bool = False,
        verify_claim_factuality_probe_comparison_manifest: bool = False,
        include_claim_factuality_probe_comparison_record: bool = False,
        verify_triple_extraction_fixture_matrix_manifest: bool = False,
        include_triple_extraction_fixture_matrix_record: bool = False,
        verify_counterfactual_verification_manifest: bool = False,
        include_counterfactual_verification_record: bool = False,
    ) -> dict[str, Any]:
        """Return ProductTrace metadata for contract and provenance evidence."""
        return {
            **self.loaded_contract.runtime_metadata(budget_enabled=budget_enabled),
            **self.evidence_metadata(
                verify_manifest=verify_manifest,
                include_registry_record=include_registry_record,
            ),
            **self.evidence_handoff_metadata(
                verify_manifest=verify_evidence_handoff_manifest,
            ),
            **self.selfcheck_signal_fusion_evidence_metadata(
                verify_manifest=verify_selfcheck_signal_fusion_manifest,
                include_registry_record=include_selfcheck_signal_fusion_record,
            ),
            **self.world_model_signal_evidence_metadata(
                verify_manifest=verify_world_model_signal_workflow_manifest,
                include_registry_record=include_world_model_signal_workflow_record,
            ),
            **self.pathway_intervention_evidence_metadata(
                verify_manifest=verify_pathway_intervention_workflow_manifest,
                include_registry_record=include_pathway_intervention_workflow_record,
            ),
            **self.external_evidence_baseline_comparison_evidence_metadata(
                include_registry_record=include_external_evidence_baseline_comparison_record,
            ),
            **self.pre_generation_probe_comparison_evidence_metadata(
                verify_manifest=verify_pre_generation_probe_comparison_manifest,
                include_registry_record=include_pre_generation_probe_comparison_record,
            ),
            **self.claim_factuality_probe_comparison_evidence_metadata(
                verify_manifest=verify_claim_factuality_probe_comparison_manifest,
                include_registry_record=include_claim_factuality_probe_comparison_record,
            ),
            **self.triple_extraction_fixture_matrix_evidence_metadata(
                verify_manifest=verify_triple_extraction_fixture_matrix_manifest,
                include_registry_record=include_triple_extraction_fixture_matrix_record,
            ),
            **self.counterfactual_verification_evidence_metadata(
                verify_manifest=verify_counterfactual_verification_manifest,
                include_registry_record=include_counterfactual_verification_record,
            ),
        }


def load_product_runtime_evidence_bundle(
    path: str | Path | None = None,
    *,
    default_contract_paths: Iterable[str | Path] = (),
    manifest_path: str | Path | None = None,
    evidence_handoff_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    registry_key: str | None = None,
    require_promoted: bool = True,
    manifest_recursive: bool = True,
) -> ProductRuntimeEvidenceBundle | None:
    """Load a promotion contract plus optional manifest and registry provenance."""
    loaded_contract = load_product_promotion_contract(
        path,
        default_paths=default_contract_paths,
        require_promoted=require_promoted,
    )
    if loaded_contract is None:
        return None
    resolved_manifest_path = _resolve_product_promotion_contract_manifest_path(
        loaded_contract.path,
        manifest_path=manifest_path,
    )
    resolved_evidence_handoff_manifest_path = (
        _resolve_product_promotion_contract_evidence_handoff_manifest_path(
            loaded_contract.path,
            evidence_handoff_manifest_path=evidence_handoff_manifest_path,
        )
    )
    return ProductRuntimeEvidenceBundle(
        loaded_contract=loaded_contract,
        manifest_path=resolved_manifest_path,
        evidence_handoff_manifest_path=resolved_evidence_handoff_manifest_path,
        registry_path=None if registry_path is None else Path(registry_path),
        registry_key=registry_key,
        manifest_recursive=manifest_recursive,
    )


def product_promotion_contract_metadata(
    contract: ProductPromotionContract | None,
    *,
    source: str | None,
    budget_enabled: bool,
) -> dict[str, Any]:
    """Return ProductTrace metadata for a promotion contract."""
    if contract is None:
        return {
            "promotion_contract_source": None,
            "promotion_contract_budget_enabled": False,
        }
    covered_fact_scope = _promotion_contract_covered_fact_scope_metadata(contract)
    return {
        "promotion_contract_source": source,
        "promotion_contract_budget_enabled": budget_enabled,
        "promotion_contract_model_id": contract.model_id,
        "promotion_contract_source_workflow": contract.source_workflow,
        "promotion_contract_source_status": contract.source_status,
        "promotion_contract_runtime": dict(contract.runtime),
        "promotion_contract_verifier_route": dict(contract.verifier_route),
        "promotion_contract_control_policy_config": dict(contract.control_policy_config),
        "promotion_contract_control_defaults": dict(contract.control_defaults),
        "promotion_contract_product_trace_replay_workflow": dict(
            contract.product_trace_replay_workflow
        ),
        "promotion_contract_selfcheck_signal_fusion_workflow": dict(
            contract.selfcheck_signal_fusion_workflow
        ),
        "promotion_contract_world_model_signal_workflow": dict(
            contract.world_model_signal_workflow
        ),
        "promotion_contract_pathway_intervention_workflow": dict(
            contract.pathway_intervention_workflow
        ),
        "promotion_contract_feedback_policy_workflow": dict(
            contract.feedback_policy_workflow
        ),
        "promotion_contract_external_evidence_baseline_comparison": dict(
            contract.external_evidence_baseline_comparison
        ),
        "promotion_contract_pre_generation_probe_comparison": dict(
            contract.pre_generation_probe_comparison
        ),
        "promotion_contract_claim_factuality_probe_comparison": dict(
            contract.claim_factuality_probe_comparison
        ),
        "promotion_contract_triple_extraction_fixture_matrix": dict(
            contract.triple_extraction_fixture_matrix
        ),
        "promotion_contract_counterfactual_verification": dict(
            contract.counterfactual_verification
        ),
        "promotion_contract_release_efficiency": dict(contract.release_efficiency),
        "promotion_contract_metadata": dict(contract.metadata),
        **_promotion_contract_product_trace_replay_metadata(contract),
        **_promotion_contract_product_runtime_drift_metadata(contract),
        **_promotion_contract_runtime_cost_metadata(contract),
        **_promotion_contract_external_evidence_baseline_comparison_metadata(contract),
        **_promotion_contract_pre_generation_probe_comparison_metadata(contract),
        **_promotion_contract_claim_factuality_probe_comparison_metadata(contract),
        **_promotion_contract_pathway_intervention_metadata(contract),
        **_promotion_contract_counterfactual_verification_metadata(contract),
        **covered_fact_scope,
    }


def _promotion_contract_runtime_cost_metadata(
    contract: ProductPromotionContract,
) -> dict[str, Any]:
    metadata = _mapping(contract.metadata)
    return _drop_none_values({
        "promotion_contract_max_uncached_forward_seconds": metadata.get(
            "max_uncached_forward_seconds"
        ),
        "promotion_contract_max_recommended_runtime_seconds": metadata.get(
            "max_recommended_runtime_seconds"
        ),
        "promotion_contract_recommended_runtime_seconds": metadata.get(
            "recommended_runtime_seconds"
        ),
        "promotion_contract_recommended_runtime_cost_source": metadata.get(
            "recommended_runtime_cost_source"
        ),
        "promotion_contract_uncached_forward_cost_seconds": metadata.get(
            "uncached_forward_cost_seconds"
        ),
        "promotion_contract_uncached_forward_cost_source": metadata.get(
            "uncached_forward_cost_source"
        ),
        "promotion_contract_cache_only_total_seconds": metadata.get(
            "cache_only_total_seconds"
        ),
    })


def _promotion_contract_product_trace_replay_metadata(
    contract: ProductPromotionContract,
) -> dict[str, Any]:
    metadata = _mapping(contract.metadata)
    workflow = _mapping(contract.product_trace_replay_workflow)
    action_audit_gate = _mapping(workflow.get("action_audit_gate"))
    action_execution_gate = _mapping(workflow.get("action_execution_gate"))
    return _drop_none_values({
        "promotion_contract_product_trace_replay_workflow_status": _first_present(
            metadata.get("product_trace_replay_workflow_status"),
            workflow.get("status"),
            workflow.get("report_status"),
        ),
        "promotion_contract_product_trace_replay_workflow_report": _first_present(
            workflow.get("report_path"),
            metadata.get("product_trace_replay_workflow_report"),
        ),
        "promotion_contract_product_trace_replay_workflow_manifest": _first_present(
            workflow.get("manifest_path"),
            metadata.get("product_trace_replay_workflow_manifest"),
        ),
        "promotion_contract_product_trace_replay_workflow_source": _first_present(
            workflow.get("source"),
            metadata.get("product_trace_replay_workflow_source"),
        ),
        "promotion_contract_product_trace_replay_workflow_registry": _first_present(
            workflow.get("registry"),
            metadata.get("product_trace_replay_workflow_registry"),
        ),
        "promotion_contract_product_trace_replay_workflow_record": _first_present(
            workflow.get("record_key"),
            metadata.get("product_trace_replay_workflow_record"),
        ),
        "promotion_contract_product_trace_replay_workflow_report_status": (
            _first_present(
                workflow.get("report_status"),
                metadata.get("product_trace_replay_workflow_report_status"),
            )
        ),
        "promotion_contract_product_trace_replay_workflow_selector_replay_report": (
            _first_present(
                workflow.get("selector_replay_report_path"),
                metadata.get("product_trace_replay_workflow_selector_replay_report"),
            )
        ),
        "promotion_contract_product_trace_replay_workflow_runtime_drift_report": (
            _first_present(
                workflow.get("product_runtime_drift_report_path"),
                metadata.get("product_trace_replay_workflow_runtime_drift_report"),
            )
        ),
        "promotion_contract_product_trace_action_audit_gate_required": _first_present(
            workflow.get("require_action_audit_gate"),
            metadata.get("product_trace_action_audit_gate_required"),
        ),
        "promotion_contract_product_trace_action_audit_gate_status": _first_present(
            workflow.get("action_audit_gate_status"),
            action_audit_gate.get("status"),
            metadata.get("product_trace_action_audit_gate_status"),
        ),
        "promotion_contract_product_trace_action_audit_gate_enabled": _first_present(
            workflow.get("action_audit_gate_enabled"),
            action_audit_gate.get("gate_enabled"),
            metadata.get("product_trace_action_audit_gate_enabled"),
        ),
        "promotion_contract_product_trace_action_audit_gate_passed": _first_present(
            workflow.get("action_audit_gate_passed"),
            action_audit_gate.get("passed"),
            metadata.get("product_trace_action_audit_gate_passed"),
        ),
        "promotion_contract_product_trace_action_audit_gate_report": _first_present(
            workflow.get("action_audit_gate_report_path"),
            action_audit_gate.get("report_path"),
            metadata.get("product_trace_action_audit_gate_report"),
        ),
        "promotion_contract_product_trace_action_audit_error_rate": _first_present(
            workflow.get("action_audit_error_rate"),
            action_audit_gate.get("error_rate"),
            metadata.get("product_trace_action_audit_error_rate"),
        ),
        "promotion_contract_product_trace_action_audit_missing_retrieval_action_rate": (
            _first_present(
                workflow.get("action_audit_missing_retrieval_action_rate"),
                action_audit_gate.get("missing_retrieval_action_rate"),
                metadata.get("product_trace_action_audit_missing_retrieval_action_rate"),
            )
        ),
        "promotion_contract_product_trace_action_audit_missing_plan_retrieval_query_rate": (
            _first_present(
                workflow.get("action_audit_missing_plan_retrieval_query_rate"),
                action_audit_gate.get("missing_plan_retrieval_query_rate"),
                metadata.get(
                    "product_trace_action_audit_missing_plan_retrieval_query_rate"
                ),
            )
        ),
        "promotion_contract_product_trace_action_audit_malformed_payload_rate": (
            _first_present(
                workflow.get("action_audit_malformed_payload_rate"),
                action_audit_gate.get("malformed_payload_rate"),
                metadata.get("product_trace_action_audit_malformed_payload_rate"),
            )
        ),
        "promotion_contract_product_trace_action_audit_unexpected_action_rate": (
            _first_present(
                workflow.get("action_audit_unexpected_action_rate"),
                action_audit_gate.get("unexpected_action_rate"),
                metadata.get("product_trace_action_audit_unexpected_action_rate"),
            )
        ),
        "promotion_contract_product_trace_action_audit_unknown_claim_id_rate": (
            _first_present(
                workflow.get("action_audit_unknown_claim_id_rate"),
                action_audit_gate.get("unknown_claim_id_rate"),
                metadata.get("product_trace_action_audit_unknown_claim_id_rate"),
            )
        ),
        "promotion_contract_product_trace_action_execution_gate_required": (
            _first_present(
                workflow.get("require_action_execution_gate"),
                metadata.get("product_trace_action_execution_gate_required"),
            )
        ),
        "promotion_contract_product_trace_action_execution_gate_status": _first_present(
            workflow.get("action_execution_gate_status"),
            action_execution_gate.get("status"),
            metadata.get("product_trace_action_execution_gate_status"),
        ),
        "promotion_contract_product_trace_action_execution_gate_enabled": (
            _first_present(
                workflow.get("action_execution_gate_enabled"),
                action_execution_gate.get("gate_enabled"),
                metadata.get("product_trace_action_execution_gate_enabled"),
            )
        ),
        "promotion_contract_product_trace_action_execution_gate_passed": _first_present(
            workflow.get("action_execution_gate_passed"),
            action_execution_gate.get("passed"),
            metadata.get("product_trace_action_execution_gate_passed"),
        ),
        "promotion_contract_product_trace_action_execution_gate_report": _first_present(
            workflow.get("action_execution_gate_report_path"),
            action_execution_gate.get("report_path"),
            metadata.get("product_trace_action_execution_gate_report"),
        ),
        "promotion_contract_product_trace_action_execution_alignment_failed_trace_rate": (
            _first_present(
                workflow.get("action_execution_alignment_failed_trace_rate"),
                action_execution_gate.get("alignment_failed_trace_rate"),
                metadata.get(
                    "product_trace_action_execution_alignment_failed_trace_rate"
                ),
            )
        ),
        "promotion_contract_product_trace_action_execution_missing_result_rate": (
            _first_present(
                workflow.get("action_execution_missing_result_rate"),
                action_execution_gate.get("missing_result_rate"),
                metadata.get("product_trace_action_execution_missing_result_rate"),
            )
        ),
        "promotion_contract_product_trace_action_execution_unexpected_result_rate": (
            _first_present(
                workflow.get("action_execution_unexpected_result_rate"),
                action_execution_gate.get("unexpected_result_rate"),
                metadata.get("product_trace_action_execution_unexpected_result_rate"),
            )
        ),
        "promotion_contract_product_trace_action_execution_request_id_mismatch_rate": (
            _first_present(
                workflow.get("action_execution_request_id_mismatch_rate"),
                action_execution_gate.get("request_id_mismatch_rate"),
                metadata.get("product_trace_action_execution_request_id_mismatch_rate"),
            )
        ),
    })


def _promotion_contract_product_runtime_drift_metadata(
    contract: ProductPromotionContract,
) -> dict[str, Any]:
    metadata = _mapping(contract.metadata)
    scalar_fields = (
        "product_runtime_drift_status",
        "product_runtime_drift_report",
        "product_runtime_drift_manifest",
        "product_runtime_drift_baseline_path",
        "product_runtime_drift_current_path",
        "product_runtime_drift_gate_enabled",
        "product_runtime_drift_promotion_evidence_required",
        "product_runtime_drift_pre_generation_evidence_required",
        "product_runtime_drift_claim_factuality_evidence_required",
        "product_runtime_drift_counterfactual_evidence_required",
        "product_runtime_drift_triple_audit_evidence_required",
        "product_runtime_drift_covered_fact_property_evidence_required",
        "product_runtime_drift_action_gate_evidence_required",
        "product_runtime_drift_trajectory_audit_evidence_required",
        "product_runtime_drift_evidence_handoff_evidence_required",
        "product_runtime_drift_compared_metric_count",
        "product_runtime_drift_blocked_metric_count",
        "product_runtime_drift_promotion_evidence_metric_count",
        "product_runtime_drift_promotion_evidence_blocked_metric_count",
        "product_runtime_drift_pre_generation_evidence_metric_count",
        "product_runtime_drift_pre_generation_evidence_blocked_metric_count",
        "product_runtime_drift_claim_factuality_evidence_metric_count",
        "product_runtime_drift_claim_factuality_evidence_blocked_metric_count",
        "product_runtime_drift_counterfactual_evidence_metric_count",
        "product_runtime_drift_counterfactual_evidence_blocked_metric_count",
        "product_runtime_drift_triple_audit_evidence_metric_count",
        "product_runtime_drift_triple_audit_evidence_blocked_metric_count",
        "product_runtime_drift_covered_fact_property_evidence_metric_count",
        "product_runtime_drift_covered_fact_property_evidence_blocked_metric_count",
        "product_runtime_drift_action_gate_evidence_metric_count",
        "product_runtime_drift_action_gate_evidence_blocked_metric_count",
        "product_runtime_drift_trajectory_audit_evidence_metric_count",
        "product_runtime_drift_trajectory_audit_evidence_blocked_metric_count",
        "product_runtime_drift_evidence_handoff_evidence_metric_count",
        "product_runtime_drift_evidence_handoff_evidence_blocked_metric_count",
    )
    flattened: dict[str, Any] = {
        f"promotion_contract_{key}": metadata.get(key)
        for key in scalar_fields
        if key in metadata
    }
    evidence_prefixes = (
        *_PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_PREFIXES,
        *_PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_PREFIXES,
        *_PRODUCT_RUNTIME_DRIFT_CLAIM_FACTUALITY_EVIDENCE_PREFIXES,
        *_PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_EVIDENCE_PREFIXES,
        *_PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_PREFIXES,
        *_PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_PREFIXES,
        *_PRODUCT_RUNTIME_DRIFT_ACTION_GATE_EVIDENCE_PREFIXES,
        *_PRODUCT_RUNTIME_DRIFT_TRAJECTORY_AUDIT_EVIDENCE_PREFIXES,
        *_PRODUCT_RUNTIME_DRIFT_EVIDENCE_HANDOFF_EVIDENCE_PREFIXES,
    )
    for prefix in evidence_prefixes:
        for suffix in ("baseline", "current", "status"):
            key = f"product_runtime_drift_{prefix}_{suffix}"
            if key in metadata:
                flattened[f"promotion_contract_{key}"] = metadata.get(key)
    return flattened


def _promotion_contract_external_evidence_baseline_comparison_metadata(
    contract: ProductPromotionContract,
) -> dict[str, Any]:
    metadata = _mapping(contract.metadata)
    comparison = _mapping(contract.external_evidence_baseline_comparison)
    return _drop_none_values({
        "promotion_contract_external_evidence_baseline_comparison_status": _first_present(
            metadata.get("external_evidence_baseline_comparison_status"),
            comparison.get("status"),
            comparison.get("decision_status"),
        ),
        "promotion_contract_external_evidence_baseline_comparison_report": _first_present(
            comparison.get("report_path"),
            metadata.get("external_evidence_baseline_comparison_report"),
        ),
        "promotion_contract_external_evidence_baseline_comparison_source": _first_present(
            comparison.get("source"),
            metadata.get("external_evidence_baseline_comparison_source"),
        ),
        "promotion_contract_external_evidence_baseline_comparison_registry": _first_present(
            comparison.get("registry"),
            metadata.get("external_evidence_baseline_comparison_registry"),
        ),
        "promotion_contract_external_evidence_baseline_comparison_record": _first_present(
            comparison.get("record_key"),
            metadata.get("external_evidence_baseline_comparison_record"),
        ),
        "promotion_contract_external_evidence_baseline_comparison_decision_status": (
            _first_present(
                comparison.get("decision_status"),
                metadata.get("external_evidence_baseline_comparison_decision_status"),
            )
        ),
        "promotion_contract_external_evidence_baseline_comparison_recommended_route": (
            _first_present(
                comparison.get("recommended_route"),
                metadata.get("external_evidence_baseline_comparison_recommended_route"),
            )
        ),
        "promotion_contract_external_evidence_baseline_comparison_recommended_route_record": (
            _first_present(
                comparison.get("recommended_route_record"),
                metadata.get(
                    "external_evidence_baseline_comparison_recommended_route_record"
                ),
            )
        ),
        "promotion_contract_external_evidence_baseline_comparison_route_passed": (
            _first_present(
                comparison.get("route_passed"),
                metadata.get("external_evidence_baseline_comparison_route_passed"),
            )
        ),
        "promotion_contract_external_evidence_baseline_comparison_text_redline_passed": (
            _first_present(
                comparison.get("text_redline_passed"),
                metadata.get("external_evidence_baseline_comparison_text_redline_passed"),
            )
        ),
        "promotion_contract_external_evidence_baseline_comparison_text_redline_run_count": (
            _first_present(
                comparison.get("text_redline_run_count"),
                metadata.get(
                    "external_evidence_baseline_comparison_text_redline_run_count"
                ),
            )
        ),
    })


def _promotion_contract_pre_generation_probe_comparison_metadata(
    contract: ProductPromotionContract,
) -> dict[str, Any]:
    metadata = _mapping(contract.metadata)
    comparison = _mapping(contract.pre_generation_probe_comparison)
    best_run = _mapping(comparison.get("best_run"))
    return _drop_none_values({
        "promotion_contract_pre_generation_probe_comparison_status": _first_present(
            metadata.get("pre_generation_probe_comparison_status"),
            comparison.get("status"),
        ),
        "promotion_contract_pre_generation_probe_comparison_report": _first_present(
            comparison.get("report_path"),
            metadata.get("pre_generation_probe_comparison_report"),
        ),
        "promotion_contract_pre_generation_probe_comparison_source": _first_present(
            comparison.get("source"),
            metadata.get("pre_generation_probe_comparison_source"),
        ),
        "promotion_contract_pre_generation_probe_comparison_registry": _first_present(
            comparison.get("registry"),
            metadata.get("pre_generation_probe_comparison_registry"),
        ),
        "promotion_contract_pre_generation_probe_comparison_record": _first_present(
            comparison.get("record_key"),
            metadata.get("pre_generation_probe_comparison_record"),
        ),
        "promotion_contract_pre_generation_probe_comparison_model_count": _first_present(
            comparison.get("model_count"),
            metadata.get("pre_generation_probe_comparison_model_count"),
        ),
        "promotion_contract_pre_generation_probe_comparison_run_count": _first_present(
            comparison.get("run_count"),
            metadata.get("pre_generation_probe_comparison_run_count"),
        ),
        "promotion_contract_pre_generation_probe_comparison_redline_passed": _first_present(
            comparison.get("redline_passed"),
            metadata.get("pre_generation_probe_comparison_redline_passed"),
        ),
        "promotion_contract_pre_generation_probe_comparison_redline_run_count": (
            _first_present(
                comparison.get("redline_run_count"),
                metadata.get("pre_generation_probe_comparison_redline_run_count"),
            )
        ),
        "promotion_contract_pre_generation_probe_comparison_best_run": _first_present(
            best_run.get("name"),
            metadata.get("pre_generation_probe_comparison_best_run"),
        ),
        "promotion_contract_pre_generation_probe_comparison_best_model": _first_present(
            best_run.get("model"),
            metadata.get("pre_generation_probe_comparison_best_model"),
        ),
        "promotion_contract_pre_generation_probe_comparison_best_layer": _first_present(
            best_run.get("recommended_layer"),
            metadata.get("pre_generation_probe_comparison_best_layer"),
        ),
        "promotion_contract_pre_generation_probe_comparison_best_test_label_auroc": (
            _first_present(
                best_run.get("test_label_auroc"),
                metadata.get("pre_generation_probe_comparison_best_test_label_auroc"),
            )
        ),
        "promotion_contract_pre_generation_probe_comparison_best_redline_margin": (
            _first_present(
                best_run.get("redline_margin"),
                metadata.get("pre_generation_probe_comparison_best_redline_margin"),
            )
        ),
        "promotion_contract_pre_generation_probe_comparison_best_redline_signal": (
            _first_present(
                best_run.get("redline_best_signal"),
                metadata.get("pre_generation_probe_comparison_best_redline_signal"),
            )
        ),
        "promotion_contract_pre_generation_probe_comparison_best_redline_auroc": (
            _first_present(
                best_run.get("redline_best_auroc"),
                metadata.get("pre_generation_probe_comparison_best_redline_auroc"),
            )
        ),
    })


def _promotion_contract_claim_factuality_probe_comparison_metadata(
    contract: ProductPromotionContract,
) -> dict[str, Any]:
    metadata = _mapping(contract.metadata)
    comparison = _mapping(contract.claim_factuality_probe_comparison)
    best_run = _mapping(comparison.get("best_run"))
    return _drop_none_values({
        "promotion_contract_claim_factuality_probe_comparison_status": _first_present(
            metadata.get("claim_factuality_probe_comparison_status"),
            comparison.get("status"),
        ),
        "promotion_contract_claim_factuality_probe_comparison_report_status": (
            _first_present(
                comparison.get("report_status"),
                metadata.get("claim_factuality_probe_comparison_report_status"),
            )
        ),
        "promotion_contract_claim_factuality_probe_comparison_report": _first_present(
            comparison.get("report_path"),
            metadata.get("claim_factuality_probe_comparison_report"),
        ),
        "promotion_contract_claim_factuality_probe_comparison_source": _first_present(
            comparison.get("source"),
            metadata.get("claim_factuality_probe_comparison_source"),
        ),
        "promotion_contract_claim_factuality_probe_comparison_registry": _first_present(
            comparison.get("registry"),
            metadata.get("claim_factuality_probe_comparison_registry"),
        ),
        "promotion_contract_claim_factuality_probe_comparison_record": _first_present(
            comparison.get("record_key"),
            metadata.get("claim_factuality_probe_comparison_record"),
        ),
        "promotion_contract_claim_factuality_probe_comparison_model_count": (
            _first_present(
                comparison.get("model_count"),
                metadata.get("claim_factuality_probe_comparison_model_count"),
            )
        ),
        "promotion_contract_claim_factuality_probe_comparison_run_count": _first_present(
            comparison.get("run_count"),
            metadata.get("claim_factuality_probe_comparison_run_count"),
        ),
        "promotion_contract_claim_factuality_probe_comparison_redline_passed": (
            _first_present(
                comparison.get("redline_passed"),
                metadata.get("claim_factuality_probe_comparison_redline_passed"),
            )
        ),
        "promotion_contract_claim_factuality_probe_comparison_redline_run_count": (
            _first_present(
                comparison.get("redline_run_count"),
                metadata.get("claim_factuality_probe_comparison_redline_run_count"),
            )
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_run": _first_present(
            best_run.get("name"),
            metadata.get("claim_factuality_probe_comparison_best_run"),
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_model": _first_present(
            best_run.get("model"),
            metadata.get("claim_factuality_probe_comparison_best_model"),
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_record_count": (
            _first_present(
                best_run.get("record_count"),
                metadata.get("claim_factuality_probe_comparison_best_record_count"),
            )
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_layer": (
            _first_present(
                best_run.get("recommended_layer"),
                metadata.get("claim_factuality_probe_comparison_best_layer"),
            )
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_test_label_auroc": (
            _first_present(
                best_run.get("test_label_auroc"),
                metadata.get("claim_factuality_probe_comparison_best_test_label_auroc"),
            )
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_test_selective_accuracy": (
            _first_present(
                best_run.get("test_selective_accuracy"),
                metadata.get(
                    "claim_factuality_probe_comparison_best_test_selective_accuracy"
                ),
            )
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_test_selective_coverage": (
            _first_present(
                best_run.get("test_selective_coverage"),
                metadata.get(
                    "claim_factuality_probe_comparison_best_test_selective_coverage"
                ),
            )
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_conformal_threshold": (
            _first_present(
                best_run.get("conformal_threshold"),
                metadata.get("claim_factuality_probe_comparison_best_conformal_threshold"),
            )
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_redline_margin": (
            _first_present(
                best_run.get("redline_margin"),
                metadata.get("claim_factuality_probe_comparison_best_redline_margin"),
            )
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_redline_signal": (
            _first_present(
                best_run.get("redline_best_signal"),
                metadata.get("claim_factuality_probe_comparison_best_redline_signal"),
            )
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_redline_auroc": (
            _first_present(
                best_run.get("redline_best_auroc"),
                metadata.get("claim_factuality_probe_comparison_best_redline_auroc"),
            )
        ),
    })


def _promotion_contract_pathway_intervention_metadata(
    contract: ProductPromotionContract,
) -> dict[str, Any]:
    metadata = _mapping(contract.metadata)
    workflow = _mapping(contract.pathway_intervention_workflow)
    return _drop_none_values({
        "promotion_contract_pathway_intervention_workflow_status": _first_present(
            metadata.get("pathway_intervention_workflow_status"),
            workflow.get("status"),
        ),
        "promotion_contract_pathway_intervention_workflow_report": _first_present(
            workflow.get("report_path"),
            metadata.get("pathway_intervention_workflow_report"),
        ),
        "promotion_contract_pathway_intervention_workflow_manifest": _first_present(
            workflow.get("manifest_path"),
            metadata.get("pathway_intervention_workflow_manifest"),
        ),
        "promotion_contract_pathway_intervention_workflow_source": _first_present(
            workflow.get("source"),
            metadata.get("pathway_intervention_workflow_source"),
        ),
        "promotion_contract_pathway_intervention_workflow_registry": _first_present(
            workflow.get("registry"),
            metadata.get("pathway_intervention_workflow_registry"),
        ),
        "promotion_contract_pathway_intervention_workflow_record": _first_present(
            workflow.get("record_key"),
            metadata.get("pathway_intervention_workflow_record"),
        ),
        "promotion_contract_pathway_intervention_workflow_report_status": _first_present(
            workflow.get("report_status"),
            metadata.get("pathway_intervention_workflow_report_status"),
        ),
        "promotion_contract_pathway_intervention_workflow_release_ready": _first_present(
            workflow.get("release_ready"),
            metadata.get("pathway_intervention_workflow_release_ready"),
        ),
        "promotion_contract_pathway_intervention_workflow_model": _first_present(
            workflow.get("model"),
            metadata.get("pathway_intervention_workflow_model"),
        ),
        "promotion_contract_pathway_intervention_workflow_layer": _first_present(
            workflow.get("layer"),
            metadata.get("pathway_intervention_workflow_layer"),
        ),
        "promotion_contract_pathway_intervention_workflow_intervention_layer": (
            _first_present(
                workflow.get("intervention_layer"),
                metadata.get("pathway_intervention_workflow_intervention_layer"),
            )
        ),
        "promotion_contract_pathway_intervention_workflow_patch_layer": _first_present(
            workflow.get("patch_layer"),
            metadata.get("pathway_intervention_workflow_patch_layer"),
        ),
        "promotion_contract_pathway_intervention_workflow_activation_ablation_gate": (
            _first_present(
                workflow.get("activation_ablation_gate_status"),
                metadata.get("pathway_intervention_workflow_activation_ablation_gate"),
            )
        ),
        "promotion_contract_pathway_intervention_workflow_source_patch_gate": (
            _first_present(
                workflow.get("source_patch_gate_status"),
                metadata.get("pathway_intervention_workflow_source_patch_gate"),
            )
        ),
        "promotion_contract_pathway_intervention_workflow_signals": _first_present(
            workflow.get("signals"),
            metadata.get("pathway_intervention_workflow_signals"),
        ),
        "promotion_contract_pathway_intervention_workflow_best_signals": _first_present(
            workflow.get("best_signals"),
            metadata.get("pathway_intervention_workflow_best_signals"),
        ),
    })


def _promotion_contract_counterfactual_verification_metadata(
    contract: ProductPromotionContract,
) -> dict[str, Any]:
    metadata = _mapping(contract.metadata)
    audit = _mapping(contract.counterfactual_verification)
    return _drop_none_values({
        "promotion_contract_counterfactual_verification_status": _first_present(
            metadata.get("counterfactual_verification_status"),
            audit.get("status"),
        ),
        "promotion_contract_counterfactual_verification_report": _first_present(
            audit.get("report_path"),
            metadata.get("counterfactual_verification_report"),
        ),
        "promotion_contract_counterfactual_verification_manifest": _first_present(
            audit.get("manifest_path"),
            metadata.get("counterfactual_verification_manifest"),
        ),
        "promotion_contract_counterfactual_verification_source": _first_present(
            audit.get("source"),
            metadata.get("counterfactual_verification_source"),
        ),
        "promotion_contract_counterfactual_verification_registry": _first_present(
            audit.get("registry"),
            metadata.get("counterfactual_verification_registry"),
        ),
        "promotion_contract_counterfactual_verification_record": _first_present(
            audit.get("record_key"),
            metadata.get("counterfactual_verification_record"),
        ),
        "promotion_contract_counterfactual_verification_workflow": _first_present(
            audit.get("workflow"),
            metadata.get("counterfactual_verification_workflow"),
        ),
        "promotion_contract_counterfactual_verification_record_count": _first_present(
            audit.get("record_count"),
            metadata.get("counterfactual_verification_record_count"),
        ),
        "promotion_contract_counterfactual_verification_pass_rate": _first_present(
            audit.get("pass_rate"),
            metadata.get("counterfactual_verification_pass_rate"),
        ),
        "promotion_contract_counterfactual_verification_false_invariance_rate": (
            _first_present(
                audit.get("false_invariance_rate"),
                metadata.get("counterfactual_verification_false_invariance_rate"),
            )
        ),
        "promotion_contract_counterfactual_verification_flip_success_count": (
            _first_present(
                audit.get("flip_success_count"),
                metadata.get("counterfactual_verification_flip_success_count"),
            )
        ),
    })


def _promotion_contract_covered_fact_scope_metadata(
    contract: ProductPromotionContract,
) -> dict[str, Any]:
    verifier_route = _mapping(contract.verifier_route)
    metadata = _mapping(contract.metadata)
    return {
        "promotion_contract_recommended_route_covered_fact_property_count": (
            _first_present(
                metadata.get("recommended_route_covered_fact_property_count"),
                verifier_route.get("covered_fact_property_count"),
            )
        ),
        "promotion_contract_recommended_route_covered_fact_properties": (
            _first_present(
                metadata.get("recommended_route_covered_fact_properties"),
                verifier_route.get("covered_fact_properties"),
            )
        ),
        "promotion_contract_recommended_route_covered_fact_property_metrics": (
            _first_present(
                metadata.get("recommended_route_covered_fact_property_metrics"),
                verifier_route.get("covered_fact_property_metrics"),
            )
        ),
        "promotion_contract_required_route_baseline_covered_fact_property_counts": (
            metadata.get("required_route_baseline_covered_fact_property_counts")
        ),
        "promotion_contract_required_route_baseline_covered_fact_properties": (
            metadata.get("required_route_baseline_covered_fact_properties")
        ),
        "promotion_contract_required_route_baseline_covered_fact_property_metrics": (
            metadata.get("required_route_baseline_covered_fact_property_metrics")
        ),
        "promotion_contract_structured_fact_robustness_property_counts": (
            metadata.get("structured_fact_robustness_property_counts")
        ),
        "promotion_contract_structured_fact_robustness_properties": (
            metadata.get("structured_fact_robustness_properties")
        ),
        "promotion_contract_structured_fact_robustness_property_metrics": (
            metadata.get("structured_fact_robustness_property_metrics")
        ),
    }


def product_runtime_budget_policy_from_release_candidate(
    report: Mapping[str, Any],
) -> ProductRuntimeBudgetPolicy:
    """Build a product runtime budget policy from release-candidate gate config."""
    comparison = _release_candidate_comparison(report)
    config = _mapping(comparison.get("config"))
    named_cache_hit_rates: dict[str, float] = {}
    if config.get("min_claims_cache_hit_rate") is not None:
        named_cache_hit_rates["claims"] = config["min_claims_cache_hit_rate"]
    if config.get("min_verifier_trace_cache_hit_rate") is not None:
        named_cache_hit_rates["verifier_trace"] = config["min_verifier_trace_cache_hit_rate"]
    return ProductRuntimeBudgetPolicy(
        max_total_seconds=config.get("max_runtime_total_seconds"),
        max_mean_route_duration_seconds=config.get("max_mean_duration_seconds"),
        max_p99_route_duration_seconds=config.get("max_p99_duration_seconds"),
        max_route_duration_seconds=config.get("max_max_duration_seconds"),
        max_mean_attempted_route_count=config.get("max_mean_attempted_route_count"),
        max_route_budget_exhaustion_rate=config.get("max_route_budget_exhaustion_rate"),
        max_retrieval_use_rate=config.get("max_retrieval_use_rate"),
        max_retrieval_hit_count=config.get("max_retrieval_hit_count"),
        min_cache_hit_rate=config.get("min_cache_hit_rate"),
        min_named_cache_hit_rate=named_cache_hit_rates,
        min_verification_skip_rate=config.get("min_verification_skip_rate"),
        min_selective_claim_skip_rate=config.get("min_selective_claim_skip_rate"),
        max_verified_claim_count=config.get("max_verified_claim_count"),
    )


def _product_control_defaults_from_release_candidate(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = _release_candidate_comparison(report)
    candidate = _mapping(comparison.get("release_candidate"))
    explicit = _drop_none_values(_mapping(candidate.get("control_defaults")))
    if explicit:
        return explicit
    drift = _mapping(candidate.get("product_runtime_drift"))
    for branch in ("current", "baseline"):
        defaults = _drop_none_values(
            _mapping(
                _nested(
                    drift,
                    branch,
                    "optimization",
                    "policy_hints",
                    "candidate_control_defaults",
                )
            )
        )
        if defaults:
            return defaults
    return {}


def _product_control_policy_config_from_release_candidate(
    feedback_policy_workflow: Mapping[str, Any],
) -> dict[str, Any]:
    return _control_policy_config_dict(
        _mapping(feedback_policy_workflow.get("candidate_control_policy_config"))
    )


def _control_policy_config_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    return ControlPolicyConfig.from_dict(payload).to_dict()


def _required_route_budget_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
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
            "required_route_require_non_oracle_evidence",
            "required_route_require_retrieval_stress_control",
            "required_route_retrieval_stress_manifest",
            "required_route_min_stress_false_supported_rate",
            "required_route_max_stress_false_refuted_rate",
        )
    }


def _release_candidate_comparison(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("workflow") == "release_candidate_registry_workflow":
        comparison = payload.get("release_candidate_comparison")
        if not isinstance(comparison, Mapping):
            raise ValueError("registry workflow payload is missing release_candidate_comparison.")
        return dict(comparison)
    return dict(payload)


def _product_trace_replay_workflow_metadata(
    workflow: Mapping[str, Any],
    *,
    manifests: Mapping[str, Any],
) -> dict[str, Any]:
    if not workflow:
        return {}
    action_audit_gate = _mapping(workflow.get("action_audit_gate"))
    action_execution_gate = _mapping(workflow.get("action_execution_gate"))
    metadata = {
        "report_path": workflow.get("report_path"),
        "manifest_path": (
            workflow.get("manifest_path")
            or manifests.get("product_trace_replay_workflow_manifest")
        ),
        "source": workflow.get("source"),
        "registry": workflow.get("registry"),
        "record_key": workflow.get("record_key"),
        "report_status": workflow.get("report_status"),
        "selector_replay_report_path": workflow.get("selector_replay_report_path"),
        "product_runtime_drift_report_path": workflow.get(
            "product_runtime_drift_report_path"
        ),
        "require_action_audit_gate": workflow.get("require_action_audit_gate"),
        "action_audit_gate_report_path": _first_present(
            workflow.get("action_audit_gate_report_path"),
            action_audit_gate.get("report_path"),
            manifests.get("product_trace_action_audit_gate_report"),
        ),
        "action_audit_gate_status": action_audit_gate.get("status"),
        "action_audit_gate_enabled": action_audit_gate.get("gate_enabled"),
        "action_audit_gate_passed": action_audit_gate.get("passed"),
        "action_audit_error_rate": action_audit_gate.get("error_rate"),
        "action_audit_missing_retrieval_action_rate": action_audit_gate.get(
            "missing_retrieval_action_rate"
        ),
        "action_audit_missing_plan_retrieval_query_rate": action_audit_gate.get(
            "missing_plan_retrieval_query_rate"
        ),
        "action_audit_malformed_payload_rate": action_audit_gate.get(
            "malformed_payload_rate"
        ),
        "action_audit_unexpected_action_rate": action_audit_gate.get(
            "unexpected_action_rate"
        ),
        "action_audit_unknown_claim_id_rate": action_audit_gate.get(
            "unknown_claim_id_rate"
        ),
        "require_action_execution_gate": workflow.get("require_action_execution_gate"),
        "action_execution_gate_report_path": _first_present(
            workflow.get("action_execution_gate_report_path"),
            action_execution_gate.get("report_path"),
            manifests.get("product_trace_action_execution_gate_report"),
        ),
        "action_execution_gate_status": action_execution_gate.get("status"),
        "action_execution_gate_enabled": action_execution_gate.get("gate_enabled"),
        "action_execution_gate_passed": action_execution_gate.get("passed"),
        "action_execution_alignment_failed_trace_rate": action_execution_gate.get(
            "alignment_failed_trace_rate"
        ),
        "action_execution_missing_result_rate": action_execution_gate.get(
            "missing_result_rate"
        ),
        "action_execution_unexpected_result_rate": action_execution_gate.get(
            "unexpected_result_rate"
        ),
        "action_execution_request_id_mismatch_rate": action_execution_gate.get(
            "request_id_mismatch_rate"
        ),
    }
    if action_audit_gate:
        metadata["action_audit_gate"] = action_audit_gate
    if action_execution_gate:
        metadata["action_execution_gate"] = action_execution_gate
    return _drop_none_values(metadata)


def _selfcheck_signal_fusion_workflow_metadata(
    workflow: Mapping[str, Any],
    *,
    manifests: Mapping[str, Any],
) -> dict[str, Any]:
    if not workflow:
        return {}
    return {
        "report_path": workflow.get("report_path"),
        "manifest_path": (
            workflow.get("manifest_path")
            or manifests.get("selfcheck_signal_fusion_workflow_manifest")
        ),
        "source": workflow.get("source"),
        "registry": workflow.get("registry"),
        "record_key": workflow.get("record_key"),
        "workflow": workflow.get("workflow"),
        "status": workflow.get("status"),
        "sample_quality_status": workflow.get("sample_quality_status"),
        "sample_quality_passed": workflow.get("sample_quality_passed"),
        "sample_quality_failed_runs": workflow.get("sample_quality_failed_runs"),
        "sample_quality_run_count": workflow.get("sample_quality_run_count"),
        "sample_quality_runs": workflow.get("sample_quality_runs"),
        "fusion_run_count": workflow.get("fusion_run_count"),
        "geometry_fusion_artifact_count": workflow.get("geometry_fusion_artifact_count"),
        "enhanced_score_dump_count": workflow.get("enhanced_score_dump_count"),
    }


def _world_model_signal_workflow_metadata(
    workflow: Mapping[str, Any],
    *,
    manifests: Mapping[str, Any],
) -> dict[str, Any]:
    if not workflow:
        return {}
    return {
        "report_path": workflow.get("report_path"),
        "manifest_path": (
            workflow.get("manifest_path")
            or manifests.get("world_model_signal_workflow_manifest")
        ),
        "source": workflow.get("source"),
        "registry": workflow.get("registry"),
        "record_key": workflow.get("record_key"),
        "workflow": workflow.get("workflow"),
        "status": workflow.get("status"),
        "release_gate_status": workflow.get("release_gate_status"),
        "trace_gap_max": workflow.get("trace_gap_max"),
        "conflict_positive_count": workflow.get("conflict_positive_count"),
        "calibrated_conflict_signal_count": workflow.get(
            "calibrated_conflict_signal_count"
        ),
        "blocking_reasons": workflow.get("blocking_reasons"),
    }


def _pathway_intervention_workflow_metadata(
    workflow: Mapping[str, Any],
    *,
    manifests: Mapping[str, Any],
) -> dict[str, Any]:
    if not workflow:
        return {}
    return {
        "report_path": workflow.get("report_path"),
        "manifest_path": (
            workflow.get("manifest_path")
            or manifests.get("pathway_intervention_workflow_manifest")
        ),
        "source": workflow.get("source"),
        "registry": workflow.get("registry"),
        "record_key": workflow.get("record_key"),
        "workflow": workflow.get("workflow"),
        "status": workflow.get("status"),
        "report_status": workflow.get("report_status"),
        "release_ready": workflow.get("release_ready"),
        "model": workflow.get("model"),
        "layer": workflow.get("layer"),
        "intervention_layer": workflow.get("intervention_layer"),
        "patch_layer": workflow.get("patch_layer"),
        "signals": workflow.get("signals"),
        "activation_ablation_gate_status": workflow.get(
            "activation_ablation_gate_status"
        ),
        "source_patch_gate_status": workflow.get("source_patch_gate_status"),
        "best_signals": workflow.get("best_signals"),
        "blocking_reasons": workflow.get("blocking_reasons"),
    }


def _feedback_policy_workflow_metadata(
    workflow: Mapping[str, Any],
    *,
    manifests: Mapping[str, Any],
) -> dict[str, Any]:
    if not workflow:
        return {}
    return {
        "report_path": workflow.get("report_path"),
        "manifest_path": (
            workflow.get("manifest_path")
            or manifests.get("feedback_policy_workflow_manifest")
        ),
        "source": workflow.get("source"),
        "registry": workflow.get("registry"),
        "record_key": workflow.get("record_key"),
        "report_status": workflow.get("report_status"),
        "promotion_decision": workflow.get("promotion_decision"),
        "candidate_control_policy": workflow.get("candidate_control_policy"),
        "candidate_control_policy_config": _control_policy_config_dict(
            _mapping(workflow.get("candidate_control_policy_config"))
        ),
        "candidate_control_defaults": workflow.get("candidate_control_defaults"),
        "candidate_control_defaults_config": _mapping(
            workflow.get("candidate_control_defaults_config")
        ),
        "matched_feedback_count": workflow.get("matched_feedback_count"),
        "accepted_but_wrong_rate": workflow.get("accepted_but_wrong_rate"),
        "retrieved_failure_rate": workflow.get("retrieved_failure_rate"),
        "abstain_false_positive_rate": workflow.get("abstain_false_positive_rate"),
        "final_answered_but_wrong_rate": workflow.get("final_answered_but_wrong_rate"),
        "final_answer_false_block_rate": workflow.get("final_answer_false_block_rate"),
        "safety_coverage_rate": workflow.get("safety_coverage_rate"),
        "unknown_safety_issue_rate": workflow.get("unknown_safety_issue_rate"),
    }


def _triple_extraction_fixture_matrix_metadata(
    matrix: Mapping[str, Any],
    *,
    manifests: Mapping[str, Any],
) -> dict[str, Any]:
    if not matrix:
        return {}
    distinct_predicates = matrix.get("distinct_predicates")
    return _drop_none_values({
        "report_path": matrix.get("report_path"),
        "manifest_path": (
            matrix.get("manifest_path")
            or manifests.get("triple_extraction_fixture_matrix_manifest")
        ),
        "source": matrix.get("source"),
        "registry": matrix.get("registry"),
        "record_key": matrix.get("record_key"),
        "workflow": matrix.get("workflow"),
        "status": _first_present(matrix.get("status"), matrix.get("report_status")),
        "n_corpora": matrix.get("n_corpora"),
        "promoted_corpora": matrix.get("promoted_corpora"),
        "distinct_predicate_count": matrix.get("distinct_predicate_count"),
        "distinct_predicates": (
            None if distinct_predicates is None else list(distinct_predicates)
        ),
        "mean_baseline_f1": matrix.get("mean_baseline_f1"),
        "mean_best_f1": matrix.get("mean_best_f1"),
        "mean_f1_lift": matrix.get("mean_f1_lift"),
    })


def _external_evidence_baseline_comparison_metadata(
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    if not comparison:
        return {}
    return _drop_none_values({
        "report_path": comparison.get("report_path"),
        "source": comparison.get("source"),
        "registry": comparison.get("registry"),
        "record_key": comparison.get("record_key"),
        "workflow": comparison.get("workflow"),
        "status": _first_present(
            comparison.get("status"),
            comparison.get("decision_status"),
            comparison.get("report_status"),
        ),
        "decision_status": comparison.get("decision_status"),
        "recommended_route": comparison.get("recommended_route"),
        "recommended_route_record": comparison.get("recommended_route_record"),
        "route_passed": comparison.get("route_passed"),
        "text_redline_passed": comparison.get("text_redline_passed"),
        "text_redline_run_count": comparison.get("text_redline_run_count"),
        "blocking_reasons": comparison.get("blocking_reasons"),
    })


def _external_evidence_baseline_comparison_flat_metadata(
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    return _drop_none_values({
        "external_evidence_baseline_comparison_report": comparison.get("report_path"),
        "external_evidence_baseline_comparison_source": comparison.get("source"),
        "external_evidence_baseline_comparison_registry": comparison.get("registry"),
        "external_evidence_baseline_comparison_record": comparison.get("record_key"),
        "external_evidence_baseline_comparison_status": comparison.get("status"),
        "external_evidence_baseline_comparison_decision_status": (
            comparison.get("decision_status")
        ),
        "external_evidence_baseline_comparison_recommended_route": (
            comparison.get("recommended_route")
        ),
        "external_evidence_baseline_comparison_recommended_route_record": (
            comparison.get("recommended_route_record")
        ),
        "external_evidence_baseline_comparison_route_passed": (
            comparison.get("route_passed")
        ),
        "external_evidence_baseline_comparison_text_redline_passed": (
            comparison.get("text_redline_passed")
        ),
        "external_evidence_baseline_comparison_text_redline_run_count": (
            comparison.get("text_redline_run_count")
        ),
    })


def _pre_generation_probe_comparison_metadata(
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    if not comparison:
        return {}
    best_run = _mapping(comparison.get("best_run"))
    return _drop_none_values({
        "report_path": comparison.get("report_path"),
        "manifest_path": comparison.get("manifest_path"),
        "source": comparison.get("source"),
        "registry": comparison.get("registry"),
        "record_key": comparison.get("record_key"),
        "workflow": comparison.get("workflow"),
        "status": _first_present(
            comparison.get("status"),
            comparison.get("report_status"),
        ),
        "model_count": comparison.get("model_count"),
        "run_count": comparison.get("run_count"),
        "redline_passed": comparison.get("redline_passed"),
        "redline_run_count": comparison.get("redline_run_count"),
        "best_run": {
            "name": best_run.get("name"),
            "model": best_run.get("model"),
            "recommended_layer": best_run.get("recommended_layer"),
            "test_label_auroc": best_run.get("test_label_auroc"),
            "redline_best_signal": best_run.get("redline_best_signal"),
            "redline_best_auroc": best_run.get("redline_best_auroc"),
            "redline_margin": best_run.get("redline_margin"),
        },
        "blocking_reasons": comparison.get("blocking_reasons"),
    })


def _pre_generation_probe_comparison_flat_metadata(
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    best_run = _mapping(comparison.get("best_run"))
    return _drop_none_values({
        "pre_generation_probe_comparison_report": comparison.get("report_path"),
        "pre_generation_probe_comparison_manifest": comparison.get("manifest_path"),
        "pre_generation_probe_comparison_source": comparison.get("source"),
        "pre_generation_probe_comparison_registry": comparison.get("registry"),
        "pre_generation_probe_comparison_record": comparison.get("record_key"),
        "pre_generation_probe_comparison_status": comparison.get("status"),
        "pre_generation_probe_comparison_model_count": comparison.get("model_count"),
        "pre_generation_probe_comparison_run_count": comparison.get("run_count"),
        "pre_generation_probe_comparison_redline_passed": comparison.get("redline_passed"),
        "pre_generation_probe_comparison_redline_run_count": (
            comparison.get("redline_run_count")
        ),
        "pre_generation_probe_comparison_best_run": best_run.get("name"),
        "pre_generation_probe_comparison_best_model": best_run.get("model"),
        "pre_generation_probe_comparison_best_layer": best_run.get("recommended_layer"),
        "pre_generation_probe_comparison_best_test_label_auroc": (
            best_run.get("test_label_auroc")
        ),
        "pre_generation_probe_comparison_best_redline_signal": (
            best_run.get("redline_best_signal")
        ),
        "pre_generation_probe_comparison_best_redline_auroc": (
            best_run.get("redline_best_auroc")
        ),
        "pre_generation_probe_comparison_best_redline_margin": best_run.get("redline_margin"),
    })


def _claim_factuality_probe_comparison_metadata(
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    if not comparison:
        return {}
    best_run = _mapping(comparison.get("best_run"))
    status = _first_present(
        comparison.get("status"),
        comparison.get("report_status"),
    )
    report_status = comparison.get("report_status")
    if report_status is None and status == "ready":
        report_status = status
    return _drop_none_values({
        "report_path": comparison.get("report_path"),
        "manifest_path": comparison.get("manifest_path"),
        "source": comparison.get("source"),
        "registry": comparison.get("registry"),
        "record_key": comparison.get("record_key"),
        "workflow": comparison.get("workflow"),
        "status": status,
        "report_status": report_status,
        "model_count": comparison.get("model_count"),
        "run_count": comparison.get("run_count"),
        "redline_passed": comparison.get("redline_passed"),
        "redline_run_count": comparison.get("redline_run_count"),
        "best_run": {
            "name": best_run.get("name"),
            "model": best_run.get("model"),
            "record_count": best_run.get("record_count"),
            "recommended_layer": best_run.get("recommended_layer"),
            "test_label_auroc": best_run.get("test_label_auroc"),
            "test_selective_accuracy": best_run.get("test_selective_accuracy"),
            "test_selective_coverage": best_run.get("test_selective_coverage"),
            "conformal_threshold": best_run.get("conformal_threshold"),
            "redline_best_signal": best_run.get("redline_best_signal"),
            "redline_best_auroc": best_run.get("redline_best_auroc"),
            "redline_margin": best_run.get("redline_margin"),
        },
        "blocking_reasons": comparison.get("blocking_reasons"),
    })


def _claim_factuality_probe_comparison_flat_metadata(
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    best_run = _mapping(comparison.get("best_run"))
    return _drop_none_values({
        "claim_factuality_probe_comparison_report": comparison.get("report_path"),
        "claim_factuality_probe_comparison_manifest": comparison.get("manifest_path"),
        "claim_factuality_probe_comparison_source": comparison.get("source"),
        "claim_factuality_probe_comparison_registry": comparison.get("registry"),
        "claim_factuality_probe_comparison_record": comparison.get("record_key"),
        "claim_factuality_probe_comparison_status": comparison.get("status"),
        "claim_factuality_probe_comparison_report_status": comparison.get("report_status"),
        "claim_factuality_probe_comparison_model_count": comparison.get("model_count"),
        "claim_factuality_probe_comparison_run_count": comparison.get("run_count"),
        "claim_factuality_probe_comparison_redline_passed": (
            comparison.get("redline_passed")
        ),
        "claim_factuality_probe_comparison_redline_run_count": (
            comparison.get("redline_run_count")
        ),
        "claim_factuality_probe_comparison_best_run": best_run.get("name"),
        "claim_factuality_probe_comparison_best_model": best_run.get("model"),
        "claim_factuality_probe_comparison_best_record_count": best_run.get(
            "record_count"
        ),
        "claim_factuality_probe_comparison_best_layer": best_run.get(
            "recommended_layer"
        ),
        "claim_factuality_probe_comparison_best_test_label_auroc": best_run.get(
            "test_label_auroc"
        ),
        "claim_factuality_probe_comparison_best_test_selective_accuracy": (
            best_run.get("test_selective_accuracy")
        ),
        "claim_factuality_probe_comparison_best_test_selective_coverage": (
            best_run.get("test_selective_coverage")
        ),
        "claim_factuality_probe_comparison_best_conformal_threshold": best_run.get(
            "conformal_threshold"
        ),
        "claim_factuality_probe_comparison_best_redline_signal": (
            best_run.get("redline_best_signal")
        ),
        "claim_factuality_probe_comparison_best_redline_auroc": (
            best_run.get("redline_best_auroc")
        ),
        "claim_factuality_probe_comparison_best_redline_margin": best_run.get(
            "redline_margin"
        ),
    })


def _counterfactual_verification_metadata(
    audit: Mapping[str, Any],
    *,
    manifests: Mapping[str, Any],
) -> dict[str, Any]:
    if not audit:
        return {}
    gate = _mapping(audit.get("gate"))
    return _drop_none_values({
        "report_path": audit.get("report_path"),
        "manifest_path": (
            audit.get("manifest_path")
            or manifests.get("counterfactual_verification_manifest")
        ),
        "source": audit.get("source"),
        "registry": audit.get("registry"),
        "record_key": audit.get("record_key"),
        "workflow": audit.get("workflow"),
        "status": _first_present(audit.get("status"), audit.get("report_status")),
        "record_count": audit.get("record_count"),
        "pass_rate": audit.get("pass_rate"),
        "false_invariance_rate": audit.get("false_invariance_rate"),
        "flip_success_count": audit.get("flip_success_count"),
        "blocking_reasons": _first_present(
            audit.get("blocking_reasons"),
            gate.get("blocking_reasons"),
        ),
    })


def _counterfactual_verification_flat_metadata(
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    return _drop_none_values({
        "counterfactual_verification_report": audit.get("report_path"),
        "counterfactual_verification_manifest": audit.get("manifest_path"),
        "counterfactual_verification_source": audit.get("source"),
        "counterfactual_verification_registry": audit.get("registry"),
        "counterfactual_verification_record": audit.get("record_key"),
        "counterfactual_verification_status": audit.get("status"),
        "counterfactual_verification_workflow": audit.get("workflow"),
        "counterfactual_verification_record_count": audit.get("record_count"),
        "counterfactual_verification_pass_rate": audit.get("pass_rate"),
        "counterfactual_verification_false_invariance_rate": (
            audit.get("false_invariance_rate")
        ),
        "counterfactual_verification_flip_success_count": (
            audit.get("flip_success_count")
        ),
    })


def _triple_extraction_fixture_matrix_flat_metadata(
    matrix: Mapping[str, Any],
) -> dict[str, Any]:
    return _drop_none_values({
        "triple_extraction_fixture_matrix_report": matrix.get("report_path"),
        "triple_extraction_fixture_matrix_manifest": matrix.get("manifest_path"),
        "triple_extraction_fixture_matrix_source": matrix.get("source"),
        "triple_extraction_fixture_matrix_registry": matrix.get("registry"),
        "triple_extraction_fixture_matrix_record": matrix.get("record_key"),
        "triple_extraction_fixture_matrix_status": matrix.get("status"),
        "triple_extraction_fixture_matrix_n_corpora": matrix.get("n_corpora"),
        "triple_extraction_fixture_matrix_promoted_corpora": matrix.get(
            "promoted_corpora"
        ),
        "triple_extraction_fixture_matrix_distinct_predicate_count": matrix.get(
            "distinct_predicate_count"
        ),
        "triple_extraction_fixture_matrix_distinct_predicates": matrix.get(
            "distinct_predicates"
        ),
        "triple_extraction_fixture_matrix_mean_baseline_f1": matrix.get(
            "mean_baseline_f1"
        ),
        "triple_extraction_fixture_matrix_mean_best_f1": matrix.get("mean_best_f1"),
        "triple_extraction_fixture_matrix_mean_f1_lift": matrix.get("mean_f1_lift"),
    })


def _release_efficiency_metadata(
    report: Mapping[str, Any],
    *,
    manifests: Mapping[str, Any],
) -> dict[str, Any]:
    if not report:
        return {}
    leaderboard = report.get("leaderboard")
    top = _mapping(leaderboard[0]) if isinstance(leaderboard, (list, tuple)) and leaderboard else {}
    return _drop_none_values({
        "report_path": _first_present(
            report.get("report_path"),
            _nested(report, "paths", "report"),
            report.get("path"),
        ),
        "manifest_path": _first_present(
            report.get("manifest_path"),
            _nested(report, "paths", "artifact_manifest"),
            manifests.get("release_efficiency_manifest"),
        ),
        "source": report.get("source"),
        "registry": report.get("registry"),
        "record_key": report.get("record_key"),
        "workflow": report.get("workflow"),
        "status": _first_present(report.get("status"), _nested(report, "decision", "status")),
        "recommended_profile": _first_present(
            report.get("recommended_profile"),
            _nested(report, "decision", "recommended_profile"),
        ),
        "recommended_efficiency_score": _first_present(
            report.get("recommended_efficiency_score"),
            _nested(report, "decision", "recommended_efficiency_score"),
            _nested(top, "efficiency", "score"),
        ),
        "profile_count": _first_present(
            report.get("profile_count"),
            _nested(report, "summary", "profile_count"),
        ),
        "quality_passed": _first_present(
            report.get("quality_passed"),
            _nested(report, "summary", "quality_passed"),
        ),
        "trace_record_cache_hit_profile_count": _first_present(
            report.get("trace_record_cache_hit_profile_count"),
            _nested(report, "summary", "trace_record_cache_hit_profile_count"),
        ),
        "leaderboard_top_profile": top.get("profile"),
    })


def _release_efficiency_flat_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    return _drop_none_values({
        "release_efficiency_report": report.get("report_path"),
        "release_efficiency_manifest": report.get("manifest_path"),
        "release_efficiency_status": report.get("status"),
        "release_efficiency_recommended_profile": report.get("recommended_profile"),
        "release_efficiency_score": report.get("recommended_efficiency_score"),
        "release_efficiency_profile_count": report.get("profile_count"),
        "release_efficiency_quality_passed": report.get("quality_passed"),
        "release_efficiency_trace_record_cache_hit_profile_count": report.get(
            "trace_record_cache_hit_profile_count"
        ),
    })


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, Mapping) and value:
            return dict(value)
    return {}


def _nested(payload: Mapping[str, Any], *path: str) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _drop_none_values(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if item is not None}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _resolve_product_promotion_contract_manifest_path(
    contract_path: Path | None,
    *,
    manifest_path: str | Path | None,
) -> Path | None:
    if manifest_path is not None:
        return Path(manifest_path)
    if contract_path is None:
        return None
    sibling_manifest = contract_path.parent / "artifact-manifest.json"
    if sibling_manifest.exists():
        return sibling_manifest
    return None


def _resolve_product_promotion_contract_evidence_handoff_manifest_path(
    contract_path: Path | None,
    *,
    evidence_handoff_manifest_path: str | Path | None,
) -> Path | None:
    if evidence_handoff_manifest_path is not None:
        return Path(evidence_handoff_manifest_path)
    if contract_path is None:
        return None
    sibling_manifest = contract_path.parent / "evidence-handoff-artifact-manifest.json"
    if sibling_manifest.exists():
        return sibling_manifest
    return None


def _artifact_manifest_entry_path(
    manifest_path: Path | None,
    manifest_payload: Mapping[str, Any] | None,
    artifact_key: str,
) -> Path | None:
    if manifest_path is None or manifest_payload is None:
        return None
    artifact = _mapping(_mapping(manifest_payload.get("artifacts")).get(artifact_key))
    raw_path = artifact.get("path")
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _resolve_contract_metadata_path(
    value: Any,
    *,
    contract_path: Path | None,
) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute() or path.exists() or contract_path is None:
        return path
    contract_relative = contract_path.parent / path
    if contract_relative.exists():
        return contract_relative
    return path


def _find_product_promotion_contract_record(
    registry: ArtifactRegistry,
    *,
    contract_path: Path | None,
    source: str,
) -> RegistryRecord | None:
    for record in registry.list_records(artifact_type="product_promotion_contract"):
        if _record_path_matches(record.path, contract_path=contract_path, source=source):
            return record
    return None


def _record_path_matches(
    record_path: str,
    *,
    contract_path: Path | None,
    source: str,
) -> bool:
    if record_path == source:
        return True
    if contract_path is None:
        return False
    raw_record_path = Path(record_path)
    try:
        return raw_record_path.resolve() == contract_path.resolve()
    except OSError:
        return False
