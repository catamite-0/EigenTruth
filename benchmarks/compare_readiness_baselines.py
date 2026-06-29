"""Compare registered adapter readiness baselines and recommend a runtime."""

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

from benchmarks import artifact_json_cache as _artifact_json_cache  # noqa: E402
from benchmarks.recommend_runtime_config import (  # noqa: E402
    INSIDE_TRIGGER_BUDGET_POLICIES,
    build_runtime_recommendation,
)
from eigentruth.registry import (  # noqa: E402
    ArtifactRegistry,
    ArtifactVerificationContext,
    RegistryRecord,
    verify_artifact_manifest,
)

_load_optional_json = _artifact_json_cache.load_optional_json


def compare_readiness_baselines(
    *,
    registry_path: str | Path,
    baseline_keys: Sequence[str] = (),
    recursive: bool = True,
    allow_unverified: bool = False,
    inside_trigger_budget_policy: str | None = None,
    min_best_quality_auroc: float | None = None,
    max_uncached_forward_seconds: float | None = None,
    max_cache_only_seconds: float | None = None,
    max_recommended_runtime_seconds: float | None = None,
    max_covariance_maha_last_auroc_drop: float | None = None,
    max_inside_sample_count_ratio: float | None = None,
    max_inside_generation_seconds_ratio: float | None = None,
    notes: Sequence[str] = (),
    fingerprint_cache: MutableMapping[str, dict[str, Any]] | None = None,
    json_cache: MutableMapping[str, dict[str, Any]] | None = None,
    json_cache_stats: MutableMapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed comparison of registered readiness baselines."""
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
    inside_trigger_budget_policy = _normalize_inside_trigger_budget_policy(
        inside_trigger_budget_policy
    )
    registry = ArtifactRegistry.load_json(registry_path)
    records = _select_records(registry, baseline_keys=baseline_keys)
    rows = [
        _readiness_row(
            record,
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
            fingerprint_cache=cache,
            json_cache=payload_cache,
            json_cache_stats=payload_cache_stats,
        )
        for record in records
    ]
    leaderboard = sorted(rows, key=_leaderboard_key)
    recommendation = next((row for row in leaderboard if row["gate"]["passed"]), None)
    decision = _decision(leaderboard, recommendation)
    return {
        "schema_version": 1,
        "workflow": "readiness_baseline_comparison",
        "registry": str(registry_path),
        "config": {
            "baseline_keys": list(baseline_keys),
            "recursive": recursive,
            "allow_unverified": allow_unverified,
            "inside_trigger_budget_policy": inside_trigger_budget_policy,
            "min_best_quality_auroc": min_best_quality_auroc,
            "max_uncached_forward_seconds": max_uncached_forward_seconds,
            "max_cache_only_seconds": max_cache_only_seconds,
            "max_recommended_runtime_seconds": max_recommended_runtime_seconds,
            "max_covariance_maha_last_auroc_drop": max_covariance_maha_last_auroc_drop,
            "max_inside_sample_count_ratio": max_inside_sample_count_ratio,
            "max_inside_generation_seconds_ratio": max_inside_generation_seconds_ratio,
        },
        "summary": {
            "record_count": len(rows),
            "passing_count": sum(1 for row in rows if row["gate"]["passed"]),
            "recommended_record": None if recommendation is None else recommendation["record_key"],
            "artifact_json_cache": verification_context.json_cache_summary(),
        },
        "decision": decision,
        "leaderboard": leaderboard,
        "notes": list(notes),
    }


def _select_records(
    registry: ArtifactRegistry,
    *,
    baseline_keys: Sequence[str],
) -> tuple[RegistryRecord, ...]:
    if baseline_keys:
        records = tuple(registry.get(key) for key in baseline_keys)
    else:
        records = tuple(
            record
            for record in registry.list_records(artifact_type="benchmark_manifest")
            if _is_readiness_record(record)
        )
    for record in records:
        if record.artifact_type != "benchmark_manifest":
            raise ValueError(f"registry record {record.key()!r} is not a benchmark_manifest.")
    return records


def _is_readiness_record(record: RegistryRecord) -> bool:
    metadata = dict(record.metadata)
    manifest_metadata = _mapping(metadata.get("manifest_metadata"))
    return (
        metadata.get("workflow") == "run_adapter_readiness_registry_workflow"
        or manifest_metadata.get("runner") == "run_adapter_readiness_workflow"
        or metadata.get("readiness_status") is not None
    )


def _readiness_row(
    record: RegistryRecord,
    *,
    recursive: bool,
    allow_unverified: bool,
    inside_trigger_budget_policy: str | None,
    min_best_quality_auroc: float | None,
    max_uncached_forward_seconds: float | None,
    max_cache_only_seconds: float | None,
    max_recommended_runtime_seconds: float | None,
    max_covariance_maha_last_auroc_drop: float | None,
    max_inside_sample_count_ratio: float | None,
    max_inside_generation_seconds_ratio: float | None,
    fingerprint_cache: MutableMapping[str, dict[str, Any]],
    json_cache: MutableMapping[str, dict[str, Any]],
    json_cache_stats: MutableMapping[str, int],
) -> dict[str, Any]:
    manifest_path = Path(record.path)
    manifest, manifest_error = _load_optional_json(
        manifest_path,
        json_cache=json_cache,
        json_cache_stats=json_cache_stats,
    )
    verification = _verify_manifest(
        manifest_path,
        manifest=manifest,
        manifest_error=manifest_error,
        recursive=recursive,
        fingerprint_cache=fingerprint_cache,
    )
    manifest_metadata = _manifest_metadata(record, manifest)
    runtime_recommendation, runtime_source = _runtime_recommendation_from_manifest(
        manifest_path,
        manifest,
        manifest_metadata,
        inside_trigger_budget_policy=inside_trigger_budget_policy,
        json_cache=json_cache,
        json_cache_stats=json_cache_stats,
    )
    recommendation = _mapping(runtime_recommendation.get("recommendation"))
    best_quality = _best_quality(recommendation, manifest_metadata)
    quality_signals = _quality_signals(recommendation, manifest_metadata)
    uncached_cost = _uncached_forward_cost(recommendation)
    cache_only_seconds = _float_or_none(recommendation.get("cache_only_total_seconds"))
    recommended_runtime_cost = _recommended_runtime_cost(recommendation)
    covariance_tradeoff = _mapping(recommendation.get("covariance_tradeoff"))
    covariance_gate = covariance_tradeoff_gate(
        runtime_recommendation,
        max_covariance_maha_last_auroc_drop=max_covariance_maha_last_auroc_drop,
    )
    inside_sampling = _inside_sampling_summary(recommendation, manifest_metadata)
    gate = _gate(
        verification=verification,
        allow_unverified=allow_unverified,
        manifest_error=manifest_error,
        readiness_status=manifest_metadata.get("readiness_status"),
        runtime_status=runtime_recommendation.get("status"),
        best_quality=best_quality,
        uncached_forward_seconds=uncached_cost["seconds"],
        cache_only_seconds=cache_only_seconds,
        recommended_runtime_seconds=recommended_runtime_cost["seconds"],
        inside_sample_count_ratio=inside_sampling["sample_count_ratio_for_gate"],
        inside_generation_seconds_ratio=inside_sampling["inside_generation_seconds_ratio_for_gate"],
        min_best_quality_auroc=min_best_quality_auroc,
        max_uncached_forward_seconds=max_uncached_forward_seconds,
        max_cache_only_seconds=max_cache_only_seconds,
        max_recommended_runtime_seconds=max_recommended_runtime_seconds,
        covariance_tradeoff_gate=covariance_gate,
        max_inside_sample_count_ratio=max_inside_sample_count_ratio,
        max_inside_generation_seconds_ratio=max_inside_generation_seconds_ratio,
    )
    return {
        "record_key": record.key(),
        "name": record.name,
        "version": record.version,
        "manifest_path": str(manifest_path),
        "verification": verification,
        "gate": gate,
        "runtime_recommendation_source": runtime_source,
        "readiness_status": manifest_metadata.get("readiness_status"),
        "adapter_family_status": manifest_metadata.get("adapter_family_status"),
        "performance_status": manifest_metadata.get("performance_status"),
        "runtime_recommendation_status": runtime_recommendation.get("status"),
        "model": manifest_metadata.get("model"),
        "dtype": manifest_metadata.get("dtype"),
        "recommended_route": manifest_metadata.get("recommended_route"),
        "recommended_performance_cell": manifest_metadata.get("recommended_performance_cell"),
        "layer": recommendation.get("layer", manifest_metadata.get("recommended_layer")),
        "batch_size": recommendation.get("batch_size", manifest_metadata.get("recommended_batch_size")),
        "hidden_state_capture": recommendation.get(
            "hidden_state_capture",
            manifest_metadata.get("recommended_hidden_state_capture"),
        ),
        "max_batch_tokens": recommendation.get(
            "max_batch_tokens",
            manifest_metadata.get("recommended_max_batch_tokens"),
        ),
        "prefix_kv_cache": recommendation.get(
            "prefix_kv_cache",
            manifest_metadata.get("recommended_prefix_kv_cache"),
        ),
        "max_workers": recommendation.get("max_workers", manifest_metadata.get("recommended_max_workers")),
        "quality_signals": quality_signals,
        "best_quality_signal": best_quality,
        "truth_proj_auroc": _float_or_none(recommendation.get("truth_proj_auroc")),
        "uncached_total_seconds": _float_or_none(recommendation.get("uncached_total_seconds")),
        "uncached_forced_answer_forward_seconds": _float_or_none(
            recommendation.get("uncached_forced_answer_forward_seconds")
        ),
        "uncached_forward_cost_seconds": uncached_cost["seconds"],
        "uncached_forward_cost_source": uncached_cost["source"],
        "cache_only_total_seconds": cache_only_seconds,
        "recommended_runtime_seconds": recommended_runtime_cost["seconds"],
        "recommended_runtime_cost_source": recommended_runtime_cost["source"],
        "covariance_mode": recommendation.get(
            "covariance_mode",
            manifest_metadata.get("recommended_covariance_mode"),
        ),
        "covariance_low_rank": recommendation.get(
            "covariance_low_rank",
            manifest_metadata.get("recommended_covariance_low_rank"),
        ),
        "covariance_tradeoff": None if not covariance_tradeoff else covariance_tradeoff,
        "covariance_tradeoff_status": covariance_tradeoff.get("status"),
        "covariance_maha_last_delta_vs_baseline": covariance_gate.get(
            "selected_maha_last_delta_vs_baseline"
        ),
        "covariance_tradeoff_gate": covariance_gate,
        "inside_sampling": inside_sampling["payload"],
        "inside_sampling_recommended_run": inside_sampling["recommended_run"],
        "inside_sampling_total_generated_samples": inside_sampling["total_generated_samples"],
        "inside_sampling_sample_count_ratio_to_baseline": inside_sampling["sample_count_ratio_to_baseline"],
        "inside_sampling_sample_count_ratio_to_reference": inside_sampling["sample_count_ratio_to_reference"],
        "inside_sampling_sample_count_ratio_for_gate": inside_sampling["sample_count_ratio_for_gate"],
        "inside_sampling_sample_count_ratio_source": inside_sampling["sample_count_ratio_source"],
        "inside_generation_seconds": inside_sampling["inside_generation_seconds"],
        "inside_generation_seconds_ratio_to_baseline": inside_sampling[
            "inside_generation_seconds_ratio_to_baseline"
        ],
        "inside_generation_seconds_ratio_to_reference": inside_sampling[
            "inside_generation_seconds_ratio_to_reference"
        ],
        "inside_generation_seconds_ratio_for_gate": inside_sampling[
            "inside_generation_seconds_ratio_for_gate"
        ],
        "inside_generation_seconds_ratio_source": inside_sampling[
            "inside_generation_seconds_ratio_source"
        ],
        "inside_sampling_stop_reason_counts": inside_sampling["stop_reason_counts"],
        "inside_trigger_budget_sweep": inside_sampling["trigger_budget_sweep"],
        "inside_trigger_budget_id": inside_sampling["trigger_budget_id"],
        "inside_trigger_budget_policy": inside_sampling["trigger_budget_policy"],
        "inside_trigger_budget_derive_from_max_budget": inside_sampling["derive_from_max_budget"],
        "performance_wall_clock_seconds": _float_or_none(
            manifest_metadata.get("performance_wall_clock_seconds")
        ),
        "wall_clock_seconds": _float_or_none(manifest_metadata.get("wall_clock_seconds")),
        "benchmark_flags": runtime_recommendation.get("benchmark_flags"),
    }


def _runtime_recommendation_from_manifest(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
    *,
    inside_trigger_budget_policy: str | None,
    json_cache: MutableMapping[str, dict[str, Any]],
    json_cache_stats: MutableMapping[str, int],
) -> tuple[dict[str, Any], str | None]:
    performance_matrix_path = _resolve_artifact_path(
        manifest_path,
        manifest,
        artifact_name="performance_matrix_report",
    )
    if performance_matrix_path is not None:
        matrix_report, _ = _load_optional_json(
            performance_matrix_path,
            json_cache=json_cache,
            json_cache_stats=json_cache_stats,
        )
        if matrix_report:
            inside_sampling_path = _resolve_artifact_path(
                manifest_path,
                manifest,
                artifact_name="inside_sampling_profile_report",
            )
            inside_sampling_report, _ = (
                ({}, None)
                if inside_sampling_path is None
                else _load_optional_json(
                    inside_sampling_path,
                    json_cache=json_cache,
                    json_cache_stats=json_cache_stats,
                )
            )
            inside_trigger_budget_sweep_path = _resolve_artifact_path(
                manifest_path,
                manifest,
                artifact_name="inside_trigger_budget_sweep_report",
            )
            inside_trigger_budget_sweep_report, _ = (
                ({}, None)
                if inside_trigger_budget_sweep_path is None
                else _load_optional_json(
                    inside_trigger_budget_sweep_path,
                    json_cache=json_cache,
                    json_cache_stats=json_cache_stats,
                )
            )
            return (
                build_runtime_recommendation(
                    matrix_report,
                    inside_sampling_report=inside_sampling_report or None,
                    inside_trigger_budget_sweep_report=inside_trigger_budget_sweep_report or None,
                    inside_trigger_budget_policy=str(
                        _first_present(
                            inside_trigger_budget_policy,
                            manifest_metadata.get("recommended_inside_trigger_budget_policy"),
                            manifest_metadata.get("inside_trigger_budget_policy"),
                            "quality_balanced",
                        )
                    ),
                    matrix_report_path=performance_matrix_path,
                    inside_sampling_report_path=inside_sampling_path,
                    inside_trigger_budget_sweep_report_path=inside_trigger_budget_sweep_path,
                ),
                str(performance_matrix_path),
            )

    runtime_path = _resolve_artifact_path(
        manifest_path,
        manifest,
        artifact_name="runtime_recommendation",
    )
    if runtime_path is not None:
        runtime, _ = _load_optional_json(
            runtime_path,
            json_cache=json_cache,
            json_cache_stats=json_cache_stats,
        )
        if runtime:
            return runtime, str(runtime_path)

    runtime_status = manifest_metadata.get("runtime_recommendation_status")
    if runtime_status is None:
        return {}, None
    quality_signals = _mapping(manifest_metadata.get("recommended_quality_signals"))
    best_quality = {
        "name": manifest_metadata.get("recommended_best_quality_signal"),
        "auroc": manifest_metadata.get("recommended_best_quality_auroc"),
    }
    return (
        {
            "status": runtime_status,
            "recommendation": {
                "layer": manifest_metadata.get("recommended_layer"),
                "batch_size": manifest_metadata.get("recommended_batch_size"),
                "hidden_state_capture": manifest_metadata.get("recommended_hidden_state_capture"),
                "max_batch_tokens": manifest_metadata.get("recommended_max_batch_tokens"),
                "prefix_kv_cache": manifest_metadata.get("recommended_prefix_kv_cache"),
                "max_workers": manifest_metadata.get("recommended_max_workers"),
                "quality_signals": quality_signals,
                "best_quality_signal": best_quality,
                "inside_sampling": _mapping(manifest_metadata.get("recommended_inside_sampling")),
                "inside_trigger_budget_sweep": _mapping(
                    manifest_metadata.get("recommended_inside_trigger_budget_sweep")
                ),
            },
        },
        "registry_metadata",
    )


def _manifest_metadata(record: RegistryRecord, manifest: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(record.metadata)
    manifest_metadata = _mapping(metadata.get("manifest_metadata"))
    if not manifest_metadata:
        manifest_metadata = _mapping(manifest.get("metadata"))
    merged = dict(manifest_metadata)
    merged.update(metadata)
    nested = _mapping(metadata.get("manifest_metadata"))
    merged.update(nested)
    return merged


def _verify_manifest(
    manifest_path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_error: str | None,
    recursive: bool,
    fingerprint_cache: MutableMapping[str, dict[str, Any]],
) -> dict[str, Any]:
    if manifest_error is not None:
        return _manifest_load_failure(manifest_path, manifest_error)
    try:
        return verify_artifact_manifest(
            manifest,
            manifest_path=manifest_path,
            recursive=recursive,
            fingerprint_cache=fingerprint_cache,
        ).to_dict()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _manifest_load_failure(manifest_path, str(exc))


def _manifest_load_failure(manifest_path: Path, error: str) -> dict[str, Any]:
    return {
        "manifest_path": str(manifest_path),
        "passed": False,
        "checked": 0,
        "failures": [
            {
                "name": "manifest",
                "path": str(manifest_path),
                "field": "load",
                "expected": "readable artifact manifest",
                "actual": error,
            }
        ],
        "nested": [],
    }


def covariance_tradeoff_gate(
    runtime_recommendation: Mapping[str, Any],
    *,
    max_covariance_maha_last_auroc_drop: float | None,
) -> dict[str, Any]:
    """Return a fail-closed gate for selected covariance-mode quality drop."""
    threshold = _validate_optional_non_negative_float(
        max_covariance_maha_last_auroc_drop,
        name="max_covariance_maha_last_auroc_drop",
    )
    enabled = threshold is not None
    recommendation = _mapping(runtime_recommendation.get("recommendation"))
    tradeoff = _mapping(recommendation.get("covariance_tradeoff"))
    selected_cell = tradeoff.get("selected_cell")
    selected = _selected_covariance_tradeoff_candidate(tradeoff)
    delta = None if selected is None else _float_or_none(selected.get("maha_last_delta_vs_baseline"))
    failures = []

    if enabled:
        if not tradeoff:
            failures.append("covariance tradeoff data is missing")
        elif selected is None:
            failures.append("selected covariance candidate is missing from tradeoff candidates")
        elif delta is None:
            failures.append("selected covariance maha_last AUROC delta is missing")
        elif delta < -float(threshold):
            failures.append(
                "selected covariance maha_last AUROC drop "
                f"{abs(delta):.6g} exceeds {float(threshold):.6g}"
            )

    return {
        "enabled": enabled,
        "max_covariance_maha_last_auroc_drop": threshold,
        "passed": not failures,
        "blocking_reasons": failures,
        "status": tradeoff.get("status"),
        "baseline_cell": tradeoff.get("baseline_cell"),
        "selected_cell": selected_cell,
        "selected_covariance_mode": None if selected is None else selected.get("covariance_mode"),
        "selected_covariance_low_rank": None if selected is None else selected.get("covariance_low_rank"),
        "selected_maha_last_auroc": (
            None if selected is None else _float_or_none(selected.get("maha_last_auroc"))
        ),
        "selected_maha_last_delta_vs_baseline": delta,
    }


def _selected_covariance_tradeoff_candidate(
    tradeoff: Mapping[str, Any],
) -> dict[str, Any] | None:
    selected_cell = tradeoff.get("selected_cell")
    for item in tradeoff.get("candidates", ()):
        candidate = _mapping(item)
        if candidate.get("cell_id") == selected_cell:
            return candidate
    return None


def _gate(
    *,
    verification: Mapping[str, Any],
    allow_unverified: bool,
    manifest_error: str | None,
    readiness_status: Any,
    runtime_status: Any,
    best_quality: Mapping[str, Any] | None,
    uncached_forward_seconds: float | None,
    cache_only_seconds: float | None,
    recommended_runtime_seconds: float | None,
    covariance_tradeoff_gate: Mapping[str, Any],
    inside_sample_count_ratio: float | None,
    inside_generation_seconds_ratio: float | None,
    min_best_quality_auroc: float | None,
    max_uncached_forward_seconds: float | None,
    max_cache_only_seconds: float | None,
    max_recommended_runtime_seconds: float | None,
    max_inside_sample_count_ratio: float | None,
    max_inside_generation_seconds_ratio: float | None,
) -> dict[str, Any]:
    failures = []
    if manifest_error is not None:
        failures.append(f"manifest could not be loaded: {manifest_error}")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("manifest verification failed")
    if readiness_status != "promote":
        failures.append(f"readiness_status is {readiness_status!r}, expected 'promote'")
    if runtime_status != "promote":
        failures.append(f"runtime_recommendation_status is {runtime_status!r}, expected 'promote'")
    quality_auroc = None if best_quality is None else _float_or_none(best_quality.get("auroc"))
    quality_name = None if best_quality is None else best_quality.get("name")
    if not quality_name or quality_auroc is None:
        failures.append("best quality signal is missing")
    if min_best_quality_auroc is not None and (
        quality_auroc is None or quality_auroc < min_best_quality_auroc
    ):
        failures.append(f"best quality AUROC below {min_best_quality_auroc}")
    if max_uncached_forward_seconds is not None and (
        uncached_forward_seconds is None or uncached_forward_seconds > max_uncached_forward_seconds
    ):
        failures.append(f"uncached forward cost seconds above {max_uncached_forward_seconds}")
    if max_cache_only_seconds is not None and (
        cache_only_seconds is None or cache_only_seconds > max_cache_only_seconds
    ):
        failures.append(f"cache-only total seconds above {max_cache_only_seconds}")
    if max_recommended_runtime_seconds is not None and (
        recommended_runtime_seconds is None
        or recommended_runtime_seconds > max_recommended_runtime_seconds
    ):
        failures.append(f"recommended runtime seconds above {max_recommended_runtime_seconds}")
    failures.extend(covariance_tradeoff_gate.get("blocking_reasons", ()))
    if max_inside_sample_count_ratio is not None and (
        inside_sample_count_ratio is None or inside_sample_count_ratio > max_inside_sample_count_ratio
    ):
        failures.append(f"INSIDE sampling sample-count ratio above {max_inside_sample_count_ratio}")
    if max_inside_generation_seconds_ratio is not None and (
        inside_generation_seconds_ratio is None
        or inside_generation_seconds_ratio > max_inside_generation_seconds_ratio
    ):
        failures.append(
            f"INSIDE sampling generation-seconds ratio above {max_inside_generation_seconds_ratio}"
        )
    return {
        "passed": not failures,
        "blocking_reasons": failures,
        "covariance_tradeoff": dict(covariance_tradeoff_gate),
    }


def _decision(
    leaderboard: Sequence[Mapping[str, Any]],
    recommendation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not leaderboard:
        return {
            "status": "no_candidate",
            "recommended_record": None,
            "blocking_reasons": ("no readiness benchmark_manifest records selected",),
        }
    if recommendation is not None:
        return {
            "status": "promote",
            "recommended_record": recommendation["record_key"],
            "recommended_model": recommendation.get("model"),
            "recommended_best_quality_signal": recommendation.get("best_quality_signal"),
            "blocking_reasons": (),
        }
    return {
        "status": "blocked",
        "recommended_record": None,
        "blocking_reasons": tuple(
            f"{row['record_key']}: {reason}"
            for row in leaderboard
            for reason in row["gate"]["blocking_reasons"]
        ),
    }


def _leaderboard_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    best_quality = _mapping(row.get("best_quality_signal"))
    best_auroc = _float_or_none(best_quality.get("auroc"))
    recommended_runtime = _float_or_none(row.get("recommended_runtime_seconds"))
    forward = _float_or_none(row.get("uncached_forward_cost_seconds"))
    cache_only = _float_or_none(row.get("cache_only_total_seconds"))
    inside_sample_ratio = _float_or_none(row.get("inside_sampling_sample_count_ratio_for_gate"))
    inside_seconds_ratio = _float_or_none(row.get("inside_generation_seconds_ratio_for_gate"))
    return (
        not _mapping(row.get("gate")).get("passed", False),
        -(best_auroc if best_auroc is not None else -math.inf),
        recommended_runtime if recommended_runtime is not None else math.inf,
        forward if forward is not None else math.inf,
        cache_only if cache_only is not None else math.inf,
        inside_sample_ratio if inside_sample_ratio is not None else math.inf,
        inside_seconds_ratio if inside_seconds_ratio is not None else math.inf,
        str(row.get("record_key")),
    )


def _resolve_artifact_path(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    artifact_name: str,
) -> Path | None:
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return None
    artifact = artifacts.get(artifact_name)
    if not isinstance(artifact, Mapping):
        return None
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _best_quality(
    recommendation: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    best = _mapping(recommendation.get("best_quality_signal"))
    name = best.get("name")
    auroc = _float_or_none(best.get("auroc"))
    if name and auroc is not None:
        return {"name": str(name), "auroc": auroc}
    fallback_name = manifest_metadata.get("recommended_best_quality_signal")
    fallback_auroc = _float_or_none(manifest_metadata.get("recommended_best_quality_auroc"))
    if fallback_name and fallback_auroc is not None:
        return {"name": str(fallback_name), "auroc": fallback_auroc}
    truth_proj = _float_or_none(recommendation.get("truth_proj_auroc"))
    if truth_proj is not None:
        return {"name": "truth_proj", "auroc": truth_proj}
    return None


def _quality_signals(
    recommendation: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
) -> dict[str, float]:
    signals = _finite_float_mapping(_mapping(recommendation.get("quality_signals")))
    if not signals:
        signals = _finite_float_mapping(_mapping(manifest_metadata.get("recommended_quality_signals")))
    truth_proj = _float_or_none(recommendation.get("truth_proj_auroc"))
    if truth_proj is not None and "truth_proj" not in signals:
        signals["truth_proj"] = truth_proj
    return {name: signals[name] for name in sorted(signals)}


def _uncached_forward_cost(recommendation: Mapping[str, Any]) -> dict[str, Any]:
    forced = _float_or_none(recommendation.get("uncached_forced_answer_forward_seconds"))
    if forced is not None:
        return {
            "seconds": forced,
            "source": "uncached_forced_answer_forward_seconds",
        }
    total = _float_or_none(recommendation.get("uncached_total_seconds"))
    if total is not None:
        return {
            "seconds": total,
            "source": "uncached_total_seconds_fallback",
        }
    return {
        "seconds": None,
        "source": None,
    }


def _recommended_runtime_cost(recommendation: Mapping[str, Any]) -> dict[str, Any]:
    for field_name in (
        "cache_only_total_seconds",
        "cached_total_seconds",
        "uncached_forced_answer_forward_seconds",
        "uncached_total_seconds",
    ):
        seconds = _float_or_none(recommendation.get(field_name))
        if seconds is not None:
            return {
                "seconds": seconds,
                "source": field_name,
            }
    return {
        "seconds": None,
        "source": None,
    }


def _inside_sampling_summary(
    recommendation: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _mapping(recommendation.get("inside_sampling"))
    if not payload:
        payload = _mapping(manifest_metadata.get("recommended_inside_sampling"))
    trigger_budget = _mapping(recommendation.get("inside_trigger_budget_sweep"))
    if not trigger_budget:
        trigger_budget = _mapping(manifest_metadata.get("recommended_inside_trigger_budget_sweep"))
    stop_reason_counts = payload.get("stop_reason_counts")
    if not isinstance(stop_reason_counts, Mapping):
        stop_reason_counts = {}
    sample_ratio_to_baseline = _float_or_none(payload.get("sample_count_ratio_to_baseline"))
    sample_ratio_to_reference = _first_float(
        payload.get("sample_count_ratio_to_reference"),
        trigger_budget.get("sample_count_ratio_to_reference"),
    )
    sample_ratio_for_gate, sample_ratio_source = _ratio_for_gate(
        sample_ratio_to_baseline,
        baseline_source="sample_count_ratio_to_baseline",
        reference_ratio=sample_ratio_to_reference,
        reference_source="sample_count_ratio_to_reference",
    )
    seconds_ratio_to_baseline = _float_or_none(payload.get("inside_generation_seconds_ratio_to_baseline"))
    seconds_ratio_to_reference = _first_float(
        payload.get("inside_generation_seconds_ratio_to_reference"),
        trigger_budget.get("inside_generation_seconds_ratio_to_reference"),
    )
    seconds_ratio_for_gate, seconds_ratio_source = _ratio_for_gate(
        seconds_ratio_to_baseline,
        baseline_source="inside_generation_seconds_ratio_to_baseline",
        reference_ratio=seconds_ratio_to_reference,
        reference_source="inside_generation_seconds_ratio_to_reference",
    )
    return {
        "payload": payload or None,
        "recommended_run": payload.get("recommended_run"),
        "total_generated_samples": _int_or_none(payload.get("total_generated_samples")),
        "sample_count_ratio_to_baseline": sample_ratio_to_baseline,
        "sample_count_ratio_to_reference": sample_ratio_to_reference,
        "sample_count_ratio_for_gate": sample_ratio_for_gate,
        "sample_count_ratio_source": sample_ratio_source,
        "inside_generation_seconds": _float_or_none(payload.get("inside_generation_seconds")),
        "inside_generation_seconds_ratio_to_baseline": seconds_ratio_to_baseline,
        "inside_generation_seconds_ratio_to_reference": seconds_ratio_to_reference,
        "inside_generation_seconds_ratio_for_gate": seconds_ratio_for_gate,
        "inside_generation_seconds_ratio_source": seconds_ratio_source,
        "stop_reason_counts": dict(stop_reason_counts),
        "trigger_budget_sweep": trigger_budget or None,
        "trigger_budget_id": _first_present(
            payload.get("inside_trigger_budget_id"),
            trigger_budget.get("recommended_budget_id"),
        ),
        "trigger_budget_policy": _first_present(
            payload.get("inside_trigger_budget_policy"),
            trigger_budget.get("selection_policy"),
            manifest_metadata.get("recommended_inside_trigger_budget_policy"),
            manifest_metadata.get("inside_trigger_budget_policy"),
        ),
        "derive_from_max_budget": _first_present(
            payload.get("derive_from_max_budget"),
            trigger_budget.get("derive_from_max_budget"),
        ),
    }


def _first_float(*values: Any) -> float | None:
    for value in values:
        numeric = _float_or_none(value)
        if numeric is not None:
            return numeric
    return None


def _ratio_for_gate(
    baseline_ratio: float | None,
    *,
    baseline_source: str,
    reference_ratio: float | None,
    reference_source: str,
) -> tuple[float | None, str | None]:
    if baseline_ratio is not None:
        return baseline_ratio, baseline_source
    if reference_ratio is not None:
        return reference_ratio, reference_source
    return None, None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _finite_float_mapping(values: Mapping[str, Any]) -> dict[str, float]:
    signals = {}
    for key, value in values.items():
        numeric = _float_or_none(value)
        if numeric is not None:
            signals[str(key)] = numeric
    return signals


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
    if not math.isfinite(numeric):
        return None
    return numeric


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _validate_optional_non_negative_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    numeric = _float_or_none(value)
    if numeric is None or numeric < 0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return numeric


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = compare_readiness_baselines(
        registry_path=args.registry,
        baseline_keys=tuple(args.baseline_key or ()),
        recursive=not args.no_recursive,
        allow_unverified=bool(args.allow_unverified),
        inside_trigger_budget_policy=args.inside_trigger_budget_policy,
        min_best_quality_auroc=args.min_best_quality_auroc,
        max_uncached_forward_seconds=args.max_uncached_forward_seconds,
        max_cache_only_seconds=args.max_cache_only_seconds,
        max_recommended_runtime_seconds=args.max_recommended_runtime_seconds,
        max_covariance_maha_last_auroc_drop=args.max_covariance_maha_last_auroc_drop,
        max_inside_sample_count_ratio=args.max_inside_sample_count_ratio,
        max_inside_generation_seconds_ratio=args.max_inside_generation_seconds_ratio,
        notes=args.note,
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote readiness baseline comparison to {output_path}")
    decision = payload["decision"]
    print(
        "readiness_baseline_comparison="
        f"{decision['status']} recommended={decision.get('recommended_record')}"
    )
    if args.fail_on_blocked and decision["status"] != "promote":
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare registered adapter-readiness benchmark manifests"
    )
    parser.add_argument("--registry", required=True, help="local ArtifactRegistry JSON path")
    parser.add_argument("--baseline-key", action="append", default=[],
                        help="benchmark_manifest registry key to compare; repeatable")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--note", action="append", default=[],
                        help="optional note to include in the comparison report; repeatable")
    parser.add_argument("--no-recursive", action="store_true",
                        help="only verify root manifests")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="allow unverified manifests to become candidates")
    parser.add_argument("--inside-trigger-budget-policy", default=None,
                        choices=INSIDE_TRIGGER_BUDGET_POLICIES,
                        help="optional override for trigger-budget sweep selection; omit to use each readiness "
                             "baseline policy")
    parser.add_argument("--min-best-quality-auroc", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-best-quality-auroc",
    ), default=None)
    parser.add_argument("--max-uncached-forward-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-uncached-forward-seconds",
    ), default=None,
                        help="max uncached forward cost; uses uncached total as fallback for legacy reports")
    parser.add_argument("--max-cache-only-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-cache-only-seconds",
    ), default=None)
    parser.add_argument("--max-recommended-runtime-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-recommended-runtime-seconds",
    ), default=None,
                        help="max selected deployment-path runtime cost; prefers cache-only/cached cost when "
                             "available and falls back to uncached forward cost")
    parser.add_argument("--max-covariance-maha-last-auroc-drop", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-covariance-maha-last-auroc-drop",
    ), default=None,
                        help="max allowed selected covariance maha_last AUROC drop versus the full-covariance "
                             "baseline; candidates without covariance tradeoff data fail closed when set")
    parser.add_argument("--max-inside-sample-count-ratio", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-inside-sample-count-ratio",
    ), default=None)
    parser.add_argument("--max-inside-generation-seconds-ratio", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-inside-generation-seconds-ratio",
    ), default=None)
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless a readiness baseline promotes")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
