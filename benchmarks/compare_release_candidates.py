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
from eigentruth.control import RUNTIME_PROFILE_NAMES, get_runtime_profile  # noqa: E402
from eigentruth.registry import (  # noqa: E402
    ArtifactRegistry,
    ArtifactVerificationContext,
)


def compare_release_candidates(
    *,
    readiness_registry_path: str | Path,
    route_registry_path: str | Path | None = None,
    readiness_baseline_keys: Sequence[str] = (),
    route_baseline_keys: Sequence[str] = (),
    required_route_baseline_keys: Sequence[str] = (),
    performance_registry_path: str | Path | None = None,
    performance_baseline_key: str | None = None,
    selector_replay_report_path: str | Path | None = None,
    product_runtime_drift_report_path: str | Path | None = None,
    product_trace_replay_workflow_path: str | Path | None = None,
    product_trace_replay_workflow_registry_path: str | Path | None = None,
    product_trace_replay_workflow_key: str | None = None,
    adapter_family_matrix_path: str | Path | None = None,
    required_adapter_routes: Sequence[str] = (),
    require_performance_score_dump_cache: bool = False,
    min_performance_score_dump_cache_jsonl_view_hit_rate: float | None = None,
    performance_drift_baseline_key: str | None = None,
    max_performance_uncached_total_seconds_ratio: float | None = None,
    max_performance_cached_total_seconds_ratio: float | None = None,
    max_performance_cache_only_total_seconds_ratio: float | None = None,
    max_performance_score_dump_cache_jsonl_view_hit_rate_drop: float | None = None,
    recursive: bool = True,
    allow_unverified: bool = False,
    runtime_profile: str | None = None,
    inside_trigger_budget_policy: str | None = None,
    min_best_quality_auroc: float | None = None,
    max_uncached_forward_seconds: float | None = None,
    max_cache_only_seconds: float | None = None,
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
    require_non_oracle_evidence: bool = False,
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
    required_route_require_non_oracle_evidence: bool = False,
    notes: Sequence[str] = (),
    fingerprint_cache: MutableMapping[str, dict[str, Any]] | None = None,
    json_cache: MutableMapping[str, dict[str, Any]] | None = None,
    json_cache_stats: MutableMapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed deployable release candidate from saved baselines."""
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
    max_covariance_maha_last_auroc_drop = _validate_optional_non_negative_float(
        max_covariance_maha_last_auroc_drop,
        name="max_covariance_maha_last_auroc_drop",
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
        recursive=recursive,
        allow_unverified=allow_unverified,
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
    readiness = compare_readiness_baselines(
        registry_path=readiness_registry_path,
        baseline_keys=readiness_baseline_keys,
        recursive=recursive,
        allow_unverified=allow_unverified,
        inside_trigger_budget_policy=inside_trigger_budget_policy,
        min_best_quality_auroc=min_best_quality_auroc,
        max_uncached_forward_seconds=max_uncached_forward_seconds,
        max_cache_only_seconds=max_cache_only_seconds,
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
        require_non_oracle_evidence=require_non_oracle_evidence,
        notes=("release candidate route comparison",),
        fingerprint_cache=cache,
        json_cache=payload_cache,
        json_cache_stats=payload_cache_stats,
    )
    raw_candidate = _release_candidate(readiness, route)
    required_routes = _required_route_baseline_gate(
        route_registry_path=route_registry_path,
        required_route_baseline_keys=required_route_baseline_keys,
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
        require_non_oracle_evidence=required_route_require_non_oracle_evidence,
        fingerprint_cache=cache,
        json_cache=payload_cache,
        json_cache_stats=payload_cache_stats,
    )
    adapter_family = _adapter_family_matrix_gate(
        adapter_family_matrix_path=adapter_family_matrix_path,
        required_routes=required_adapter_routes,
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
        verification_context=verification_context,
    )
    selector_replay = _selector_replay_gate(
        selector_replay_report_path=selector_replay_report_path,
        recursive=recursive,
        allow_unverified=allow_unverified,
        verification_context=verification_context,
    )
    product_runtime_drift = _product_runtime_drift_gate(
        product_runtime_drift_report_path=product_runtime_drift_report_path,
        recursive=recursive,
        allow_unverified=allow_unverified,
        verification_context=verification_context,
    )
    decision = _decision(
        readiness,
        route,
        raw_candidate,
        performance,
        adapter_family,
        required_routes,
        product_trace_replay_workflow,
        selector_replay,
        product_runtime_drift,
    )
    candidate = (
        _candidate_with_gates(
            raw_candidate,
            performance,
            adapter_family,
            required_routes,
            product_trace_replay_workflow,
            selector_replay,
            product_runtime_drift,
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
            "readiness_baseline_keys": list(readiness_baseline_keys),
            "route_baseline_keys": list(route_baseline_keys),
            "required_route_baseline_keys": list(required_route_baseline_keys),
            "performance_baseline_key": performance_baseline_key,
            "selector_replay_report": (
                None if selector_replay_report_path is None else str(selector_replay_report_path)
            ),
            "product_runtime_drift_report": (
                None
                if product_runtime_drift_report_path is None
                else str(product_runtime_drift_report_path)
            ),
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
            "adapter_family_matrix": None if adapter_family_matrix_path is None else str(adapter_family_matrix_path),
            "required_adapter_routes": list(required_adapter_routes),
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
            "runtime_profile": None if profile is None else profile.name,
            "runtime_profile_defaults": None if profile is None else dict(profile.defaults),
            "runtime_profile_applied_defaults": profile_applied,
            "inside_trigger_budget_policy": inside_trigger_budget_policy,
            "min_best_quality_auroc": min_best_quality_auroc,
            "max_uncached_forward_seconds": max_uncached_forward_seconds,
            "max_cache_only_seconds": max_cache_only_seconds,
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
            "require_non_oracle_evidence": require_non_oracle_evidence,
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
            "required_route_require_non_oracle_evidence": required_route_require_non_oracle_evidence,
        },
        "readiness_baseline_comparison": readiness,
        "route_baseline_comparison": route,
        "required_route_baseline_gate": required_routes,
        "performance_baseline_gate": performance,
        "product_trace_replay_workflow_gate": product_trace_replay_workflow,
        "selector_replay_gate": selector_replay,
        "product_runtime_drift_gate": product_runtime_drift,
        "adapter_family_matrix_gate": adapter_family,
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
        },
        "runtime_cost": {
            "uncached_forward_cost_seconds": readiness_row.get("uncached_forward_cost_seconds"),
            "uncached_forward_cost_source": readiness_row.get("uncached_forward_cost_source"),
            "cache_only_total_seconds": readiness_row.get("cache_only_total_seconds"),
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


def _decision(
    readiness: Mapping[str, Any],
    route: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    performance: Mapping[str, Any] | None = None,
    adapter_family: Mapping[str, Any] | None = None,
    required_routes: Mapping[str, Any] | None = None,
    product_trace_replay_workflow: Mapping[str, Any] | None = None,
    selector_replay: Mapping[str, Any] | None = None,
    product_runtime_drift: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    readiness_decision = _mapping(readiness.get("decision"))
    route_decision = _mapping(route.get("decision"))
    readiness_status = readiness_decision.get("status")
    route_status = route_decision.get("status")
    performance_gate = _mapping(None if performance is None else performance.get("gate"))
    performance_status = None if performance is None else performance.get("status")
    adapter_family_gate = _mapping(None if adapter_family is None else adapter_family.get("gate"))
    adapter_family_status = None if adapter_family is None else adapter_family.get("status")
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
        "required_route_baseline_status": required_route_status,
        "product_trace_replay_workflow_status": product_trace_replay_workflow_status,
        "selector_replay_status": selector_replay_status,
        "product_runtime_drift_status": product_runtime_drift_status,
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
    require_non_oracle_evidence: bool,
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
        require_non_oracle_evidence=require_non_oracle_evidence,
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
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if adapter_family_matrix_path is None:
        return None
    matrix_path = Path(adapter_family_matrix_path)
    report, report_error = verification_context.load_json_object(matrix_path)
    routes = tuple(str(route) for route in report.get("routes", ()) if str(route))
    required = tuple(str(route) for route in required_routes if str(route))
    family_statuses = _adapter_family_statuses(report)
    promotion_decision = _mapping(report.get("promotion_decision"))
    route_comparison = _mapping(report.get("route_comparison"))
    quality_gate = _mapping(route_comparison.get("quality_gate"))
    gate = _adapter_family_gate(
        report_error=report_error,
        promotion_decision=promotion_decision,
        quality_gate=quality_gate,
        routes=routes,
        family_statuses=family_statuses,
        required_routes=required,
    )
    return {
        "schema_version": 1,
        "status": "promote" if gate["passed"] else "blocked",
        "matrix_path": str(matrix_path),
        "workflow": report.get("workflow"),
        "alpha": report.get("alpha"),
        "n_records": report.get("n_records"),
        "routes": routes,
        "required_routes": required,
        "promoted_routes": tuple(route for route, status in family_statuses.items() if status == "promote"),
        "family_statuses": family_statuses,
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
    required_routes: Sequence[str],
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
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


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
    recursive: bool,
    allow_unverified: bool,
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
        allow_unverified=allow_unverified,
    )
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
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


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


def _selector_replay_gate(
    *,
    selector_replay_report_path: str | Path | None,
    recursive: bool,
    allow_unverified: bool,
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
    recursive: bool,
    allow_unverified: bool,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any] | None:
    if product_runtime_drift_report_path is None:
        return None
    report_path = Path(product_runtime_drift_report_path)
    report, report_error = verification_context.load_json_object(report_path)
    manifest_path = _product_runtime_drift_manifest_path(report, report_path=report_path)
    verification = _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        artifact_name="product_runtime_drift_manifest",
        verification_context=verification_context,
    )
    gate = _product_runtime_drift_report_gate(
        report=report,
        report_error=report_error,
        manifest_path=manifest_path,
        verification=verification,
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
        },
        "metrics": _product_runtime_drift_metric_summary(report),
        "verification": verification,
        "gate": gate,
    }


def _product_runtime_drift_report_gate(
    *,
    report: Mapping[str, Any],
    report_error: str | None,
    manifest_path: Path | None,
    verification: Mapping[str, Any],
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
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any]:
    return _verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        artifact_name="performance_baseline_manifest",
        verification_context=verification_context,
    )


def _verify_artifact_manifest(
    manifest_path: Path | None,
    *,
    recursive: bool,
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
    required_routes: Mapping[str, Any] | None,
    product_trace_replay_workflow: Mapping[str, Any] | None,
    selector_replay: Mapping[str, Any] | None,
    product_runtime_drift: Mapping[str, Any] | None,
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
            "promoted_routes": tuple(adapter_family.get("promoted_routes", ())),
            "promotion_status": adapter_family.get("promotion_status"),
        }
        manifests["adapter_family_matrix_report"] = adapter_family.get("matrix_path")
    if required_routes is not None:
        required_rows = tuple(_mapping(row) for row in required_routes.get("rows", ()))
        required_records = tuple(row.get("record_key") for row in required_rows if row.get("record_key") is not None)
        required_manifest_paths = tuple(
            row.get("manifest_path") for row in required_rows if row.get("manifest_path") is not None
        )
        payload["required_route_baselines"] = {
            "registry": required_routes.get("registry"),
            "records": required_records,
            "routes": tuple(row.get("recommended_route") for row in required_rows),
            "manifest_paths": required_manifest_paths,
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
        }
        manifests["product_trace_replay_workflow_manifest"] = product_trace_replay_workflow.get(
            "manifest_path"
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


def _validate_optional_non_negative_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    numeric = _float_or_none(value)
    if numeric is None or numeric < 0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return numeric


def _validate_optional_unit_float(value: Any, *, name: str) -> float | None:
    numeric = _validate_optional_non_negative_float(value, name=name)
    if numeric is not None and numeric > 1:
        raise ValueError(f"{name} must be between 0 and 1.")
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


def _parse_non_negative_int(value: str, *, flag: str) -> int:
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{flag} must be a non-negative integer.")
    return numeric


def _parse_unit_float(value: str, *, flag: str) -> float:
    numeric = _parse_non_negative_float(value, flag=flag)
    if numeric > 1:
        raise ValueError(f"{flag} must be between 0 and 1.")
    return numeric


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = compare_release_candidates(
        readiness_registry_path=args.readiness_registry,
        route_registry_path=args.route_registry,
        readiness_baseline_keys=tuple(args.readiness_baseline_key or ()),
        route_baseline_keys=tuple(args.route_baseline_key or ()),
        required_route_baseline_keys=tuple(args.required_route_baseline_key or ()),
        performance_registry_path=args.performance_registry,
        performance_baseline_key=args.performance_baseline_key,
        selector_replay_report_path=args.selector_replay_report,
        product_runtime_drift_report_path=args.product_runtime_drift_report,
        product_trace_replay_workflow_path=args.product_trace_replay_workflow,
        product_trace_replay_workflow_registry_path=args.product_trace_replay_workflow_registry,
        product_trace_replay_workflow_key=args.product_trace_replay_workflow_key,
        adapter_family_matrix_path=args.adapter_family_matrix,
        required_adapter_routes=tuple(args.required_adapter_route or ()),
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
        required_route_require_non_oracle_evidence=bool(args.required_route_require_non_oracle_evidence),
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
        f"product_runtime_drift={decision.get('product_runtime_drift_status')}"
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
    parser.add_argument("--performance-baseline-key", default=None,
                        help="optional performance_baseline registry key that must match the selected runtime")
    parser.add_argument("--performance-drift-baseline-key", default=None,
                        help="optional reference performance_baseline key for trend regression gates")
    parser.add_argument("--selector-replay-report", default=None,
                        help="optional runtime-profile selector replay report that must promote and verify")
    parser.add_argument("--product-runtime-drift-report", default=None,
                        help="optional product runtime drift report that must promote and verify")
    parser.add_argument("--product-trace-replay-workflow", default=None,
                        help="optional product trace replay workflow report; when supplied, its selector "
                             "replay and runtime-drift child reports are used unless explicit child report "
                             "paths are provided")
    parser.add_argument("--product-trace-replay-workflow-registry", default=None,
                        help="optional ArtifactRegistry JSON path for --product-trace-replay-workflow-key; "
                             "defaults to --readiness-registry")
    parser.add_argument("--product-trace-replay-workflow-key", default=None,
                        help="optional report:<name>:<version> registry key for a product trace replay workflow")
    parser.add_argument("--adapter-family-matrix", default=None,
                        help="optional adapter-family matrix JSON report that must promote before release")
    parser.add_argument("--required-adapter-route", action="append", default=[],
                        help="route that must be present and promoted in --adapter-family-matrix; repeatable")
    parser.add_argument("--require-performance-score-dump-cache", action="store_true",
                        help="require the selected performance baseline to include score-dump cache evidence")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--note", action="append", default=[],
                        help="optional note to include in the comparison report; repeatable")
    parser.add_argument("--no-recursive", action="store_true", help="only verify root manifests")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="allow unverified manifests to become candidates")
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
    parser.add_argument(
        "--required-route-require-non-oracle-evidence",
        action="store_true",
        help="require required route claims to omit labels and include input provenance",
    )
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless the release candidate promotes")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
