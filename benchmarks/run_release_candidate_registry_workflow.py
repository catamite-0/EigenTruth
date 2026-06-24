"""Run release-candidate comparison and register its verified manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.compare_release_candidates import compare_release_candidates  # noqa: E402
from benchmarks.promote_artifact_manifest import promote_artifact_manifest  # noqa: E402
from benchmarks.recommend_runtime_config import INSIDE_TRIGGER_BUDGET_POLICIES  # noqa: E402
from eigentruth.control import RUNTIME_PROFILE_NAMES, get_runtime_profile  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class ReleaseCandidateRegistryWorkflowConfig:
    """Configuration for registering a promoted release-candidate manifest."""

    readiness_registry_path: Path
    release_registry_path: Path
    name: str
    version: str
    route_registry_path: Path | None = None
    performance_registry_path: Path | None = None
    readiness_baseline_keys: Sequence[str] = ()
    route_baseline_keys: Sequence[str] = ()
    required_route_baseline_keys: Sequence[str] = ()
    performance_baseline_key: str | None = None
    selector_replay_report_path: Path | None = None
    product_runtime_drift_report_path: Path | None = None
    adapter_family_matrix_path: Path | None = None
    required_adapter_routes: Sequence[str] = ()
    release_report_path: Path | None = None
    artifact_manifest_path: Path | None = None
    verification_report_path: Path | None = None
    workflow_report_path: Path | None = None
    recursive: bool = True
    allow_unverified: bool = False
    runtime_profile: str | None = None
    inside_trigger_budget_policy: str | None = None
    min_best_quality_auroc: float | None = None
    max_uncached_forward_seconds: float | None = None
    max_cache_only_seconds: float | None = None
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
    promotion_metadata: Mapping[str, Any] | None = None
    allow_non_promote: bool = False
    allow_promotion_failures: bool = False

    def __post_init__(self) -> None:
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
        if self.adapter_family_matrix_path is not None:
            object.__setattr__(self, "adapter_family_matrix_path", Path(self.adapter_family_matrix_path))
        if self.release_report_path is not None:
            object.__setattr__(self, "release_report_path", Path(self.release_report_path))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        if self.verification_report_path is not None:
            object.__setattr__(self, "verification_report_path", Path(self.verification_report_path))
        if self.workflow_report_path is not None:
            object.__setattr__(self, "workflow_report_path", Path(self.workflow_report_path))
        if self.runtime_profile is not None:
            profile = get_runtime_profile(self.runtime_profile)
            object.__setattr__(self, "runtime_profile", profile.name)
        if self.inside_trigger_budget_policy is not None:
            policy = str(self.inside_trigger_budget_policy).strip().lower().replace("-", "_")
            if policy not in INSIDE_TRIGGER_BUDGET_POLICIES:
                choices = ", ".join(INSIDE_TRIGGER_BUDGET_POLICIES)
                raise ValueError(f"inside_trigger_budget_policy must be one of: {choices}")
            object.__setattr__(self, "inside_trigger_budget_policy", policy)
        object.__setattr__(self, "readiness_baseline_keys", tuple(str(key) for key in self.readiness_baseline_keys))
        object.__setattr__(self, "route_baseline_keys", tuple(str(key) for key in self.route_baseline_keys))
        object.__setattr__(
            self,
            "required_route_baseline_keys",
            tuple(str(key) for key in self.required_route_baseline_keys),
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
    comparison = compare_release_candidates(
        readiness_registry_path=config.readiness_registry_path,
        route_registry_path=config.route_registry_path,
        readiness_baseline_keys=config.readiness_baseline_keys,
        route_baseline_keys=config.route_baseline_keys,
        required_route_baseline_keys=config.required_route_baseline_keys,
        performance_registry_path=config.performance_registry_path,
        performance_baseline_key=config.performance_baseline_key,
        selector_replay_report_path=config.selector_replay_report_path,
        product_runtime_drift_report_path=config.product_runtime_drift_report_path,
        adapter_family_matrix_path=config.adapter_family_matrix_path,
        required_adapter_routes=config.required_adapter_routes,
        recursive=config.recursive,
        allow_unverified=config.allow_unverified,
        runtime_profile=config.runtime_profile,
        inside_trigger_budget_policy=config.inside_trigger_budget_policy,
        min_best_quality_auroc=config.min_best_quality_auroc,
        max_uncached_forward_seconds=config.max_uncached_forward_seconds,
        max_cache_only_seconds=config.max_cache_only_seconds,
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
        notes=("release candidate registry workflow",),
    )
    config.comparison_path.parent.mkdir(parents=True, exist_ok=True)
    config.comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_artifact_manifest(config, comparison)

    release_decision = dict(comparison.get("decision") or {})
    release_status = str(release_decision.get("status"))
    promotion = None
    blocking_reasons = []
    if release_status != "promote":
        blocking_reasons.append("release candidate comparison did not promote")

    if release_status == "promote" or config.allow_non_promote:
        promotion = promote_artifact_manifest(
            manifest_path=config.manifest_path,
            registry_path=config.release_registry_path,
            name=config.name,
            version=config.version,
            verification_report_path=config.verification_path,
            recursive=True,
            allow_failures=config.allow_promotion_failures,
            metadata=_promotion_metadata(config, comparison),
        )
        if not dict(promotion.get("verification") or {}).get("passed", False):
            blocking_reasons.append("release candidate manifest verification did not pass")

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
            "required_route_baseline_keys": tuple(config.required_route_baseline_keys),
            "adapter_family_matrix": (
                None
                if config.adapter_family_matrix_path is None
                else str(config.adapter_family_matrix_path)
            ),
            "required_adapter_routes": tuple(config.required_adapter_routes),
            "release_registry": str(config.release_registry_path),
            "name": config.name,
            "version": config.version,
            "release_report": str(config.comparison_path),
            "artifact_manifest": str(config.manifest_path),
            "allow_non_promote": config.allow_non_promote,
            "allow_promotion_failures": config.allow_promotion_failures,
            "runtime_profile": config.runtime_profile,
            "inside_trigger_budget_policy": config.inside_trigger_budget_policy,
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
        },
        "release_candidate_comparison": comparison,
        "promotion": promotion,
        "decision": decision,
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _write_artifact_manifest(
    config: ReleaseCandidateRegistryWorkflowConfig,
    comparison: Mapping[str, Any],
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
        "adapter_family_matrix_report": manifests.get("adapter_family_matrix_report"),
    }
    artifacts.update({
        str(name): path
        for name, path in manifests.items()
        if str(name).startswith("required_route_manifest_")
    })
    manifest = build_artifact_manifest(
        artifacts,
        root=config.manifest_path.parent,
        metadata=_manifest_metadata(comparison),
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
    runtime_cost = dict(candidate.get("runtime_cost") or {})
    verifier_route = dict(candidate.get("verifier_route") or {})
    manifests = dict(candidate.get("manifests") or {})
    adapter_family = dict(candidate.get("adapter_family_matrix") or {})
    required_route_baselines = dict(candidate.get("required_route_baselines") or {})
    selector_replay = dict(candidate.get("selector_replay") or {})
    selector_replay_recommended = dict(selector_replay.get("recommended") or {})
    product_runtime_drift = dict(candidate.get("product_runtime_drift") or {})
    product_runtime_drift_summary = dict(product_runtime_drift.get("summary") or {})
    product_runtime_drift_baseline = dict(product_runtime_drift.get("baseline") or {})
    product_runtime_drift_current = dict(product_runtime_drift.get("current") or {})
    return {
        "runner": "run_release_candidate_registry_workflow",
        "workflow": comparison.get("workflow"),
        "release_candidate_status": decision.get("status"),
        "release_readiness_status": decision.get("readiness_status"),
        "release_route_status": decision.get("route_status"),
        "release_performance_status": decision.get("performance_status"),
        "release_selector_replay_status": decision.get("selector_replay_status"),
        "release_product_runtime_drift_status": decision.get("product_runtime_drift_status"),
        "release_adapter_family_status": decision.get("adapter_family_status"),
        "release_required_route_baseline_status": decision.get("required_route_baseline_status"),
        "release_runtime_profile": config.get("runtime_profile"),
        "release_runtime_profile_defaults": config.get("runtime_profile_defaults"),
        "release_runtime_profile_applied_defaults": config.get("runtime_profile_applied_defaults"),
        "recommended_readiness_record": decision.get("recommended_readiness_record"),
        "recommended_route_record": decision.get("recommended_route_record"),
        "recommended_performance_baseline_record": decision.get("recommended_performance_baseline_record"),
        "recommended_selector_replay_candidate": decision.get("recommended_selector_replay_candidate"),
        "recommended_product_runtime_drift_report": decision.get("recommended_product_runtime_drift_report"),
        "required_adapter_routes": decision.get("required_adapter_routes"),
        "required_route_baseline_records": decision.get("required_route_baseline_records"),
        "recommended_model": decision.get("recommended_model"),
        "recommended_route": decision.get("recommended_route"),
        "recommended_layer": runtime.get("layer"),
        "recommended_batch_size": runtime.get("batch_size"),
        "recommended_hidden_state_capture": runtime.get("hidden_state_capture"),
        "recommended_max_batch_tokens": runtime.get("max_batch_tokens"),
        "recommended_prefix_kv_cache": runtime.get("prefix_kv_cache"),
        "recommended_max_workers": runtime.get("max_workers"),
        "recommended_best_quality_signal": best_quality.get("name"),
        "recommended_best_quality_auroc": best_quality.get("auroc"),
        "recommended_quality_signals": quality.get("quality_signals"),
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
        "product_runtime_drift_compared_metric_count": product_runtime_drift_summary.get(
            "compared_metric_count"
        ),
        "product_runtime_drift_blocked_metric_count": product_runtime_drift_summary.get(
            "blocked_metric_count"
        ),
        "adapter_family_matrix_report": adapter_family.get("matrix_path"),
        "adapter_family_routes": adapter_family.get("routes"),
        "adapter_family_promoted_routes": adapter_family.get("promoted_routes"),
        "adapter_family_required_routes": adapter_family.get("required_routes"),
        "required_route_baseline_registry": required_route_baselines.get("registry"),
        "required_route_baseline_routes": required_route_baselines.get("routes"),
        "required_route_baseline_manifests": required_route_baselines.get("manifest_paths"),
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
            )
        },
        "readiness_manifest": manifests.get("readiness_manifest"),
        "route_manifest": manifests.get("route_manifest"),
        "performance_manifest": manifests.get("performance_manifest"),
        "selector_replay_manifest": manifests.get("selector_replay_manifest"),
        "product_runtime_drift_manifest": manifests.get("product_runtime_drift_manifest"),
        "adapter_family_matrix_manifest": manifests.get("adapter_family_matrix_report"),
    }


def _promotion_metadata(
    config: ReleaseCandidateRegistryWorkflowConfig,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = {
        **_manifest_metadata(comparison),
        "workflow": "run_release_candidate_registry_workflow",
    }
    if config.promotion_metadata is not None:
        metadata.update(dict(config.promotion_metadata))
    return metadata


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


def _parse_non_negative_float(value: str, *, flag: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{flag} must be a non-negative finite number.")
    return numeric


def _parse_non_negative_int(value: str, *, flag: str) -> int:
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{flag} must be a non-negative integer.")
    return numeric


def _config_from_args(args: argparse.Namespace) -> ReleaseCandidateRegistryWorkflowConfig:
    return ReleaseCandidateRegistryWorkflowConfig(
        readiness_registry_path=Path(args.readiness_registry),
        route_registry_path=None if args.route_registry is None else Path(args.route_registry),
        performance_registry_path=None if args.performance_registry is None else Path(args.performance_registry),
        release_registry_path=Path(args.release_registry),
        name=args.name,
        version=args.version,
        readiness_baseline_keys=tuple(args.readiness_baseline_key or ()),
        route_baseline_keys=tuple(args.route_baseline_key or ()),
        required_route_baseline_keys=tuple(args.required_route_baseline_key or ()),
        performance_baseline_key=args.performance_baseline_key,
        selector_replay_report_path=(
            None if args.selector_replay_report is None else Path(args.selector_replay_report)
        ),
        product_runtime_drift_report_path=(
            None
            if args.product_runtime_drift_report is None
            else Path(args.product_runtime_drift_report)
        ),
        adapter_family_matrix_path=None if args.adapter_family_matrix is None else Path(args.adapter_family_matrix),
        required_adapter_routes=tuple(args.required_adapter_route or ()),
        release_report_path=None if args.release_report_json is None else Path(args.release_report_json),
        artifact_manifest_path=None if args.artifact_manifest is None else Path(args.artifact_manifest),
        verification_report_path=None if args.verification_report is None else Path(args.verification_report),
        workflow_report_path=None if args.json is None else Path(args.json),
        recursive=not args.no_recursive,
        allow_unverified=bool(args.allow_unverified),
        runtime_profile=args.runtime_profile,
        inside_trigger_budget_policy=args.inside_trigger_budget_policy,
        min_best_quality_auroc=args.min_best_quality_auroc,
        max_uncached_forward_seconds=args.max_uncached_forward_seconds,
        max_cache_only_seconds=args.max_cache_only_seconds,
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
    parser.add_argument("--readiness-baseline-key", action="append", default=[])
    parser.add_argument("--route-baseline-key", action="append", default=[])
    parser.add_argument("--required-route-baseline-key", action="append", default=[],
                        help="additional promoted route benchmark_manifest key that must verify without "
                             "becoming the selected product route; repeatable")
    parser.add_argument("--performance-baseline-key", default=None,
                        help="optional performance_baseline registry key that must match the selected runtime")
    parser.add_argument("--selector-replay-report", default=None,
                        help="optional runtime-profile selector replay report that must promote and verify")
    parser.add_argument("--product-runtime-drift-report", default=None,
                        help="optional product runtime drift report that must promote and verify")
    parser.add_argument("--adapter-family-matrix", default=None,
                        help="optional adapter-family matrix JSON report that must promote before release")
    parser.add_argument("--required-adapter-route", action="append", default=[],
                        help="route that must be present and promoted in --adapter-family-matrix; repeatable")
    parser.add_argument("--json", default=None, help="optional registry workflow report path")
    parser.add_argument("--release-report-json", default=None,
                        help="optional path for the release candidate comparison report")
    parser.add_argument("--artifact-manifest", default=None,
                        help="optional path for the release candidate artifact manifest")
    parser.add_argument("--verification-report", default=None)
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
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless the release candidate registry workflow promotes")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
