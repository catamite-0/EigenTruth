"""Compare registered verifier-route promotion baselines."""

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

from benchmarks.runtime_budget_policy import (  # noqa: E402
    RuntimeBudgetPolicy,
    evaluate_runtime_budget,
    runtime_metrics_from_metadata,
)
from eigentruth.registry import ArtifactRegistry, RegistryRecord, load_and_verify_artifact_manifest  # noqa: E402


def compare_route_baselines(
    *,
    registry_path: str | Path,
    baseline_keys: Sequence[str] = (),
    recursive: bool = True,
    allow_unverified: bool = False,
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
    notes: Sequence[str] = (),
    fingerprint_cache: MutableMapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed comparison of registered route-promotion baselines."""
    cache = fingerprint_cache if fingerprint_cache is not None else {}
    registry = ArtifactRegistry.load_json(registry_path)
    records = _select_records(registry, baseline_keys=baseline_keys)
    rows = [
        _route_baseline_row(
            record,
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
            fingerprint_cache=cache,
        )
        for record in records
    ]
    leaderboard = sorted(rows, key=_leaderboard_key)
    recommendation = next((row for row in leaderboard if row["gate"]["passed"]), None)
    decision = _decision(leaderboard, recommendation)
    return {
        "schema_version": 1,
        "workflow": "route_baseline_comparison",
        "registry": str(registry_path),
        "config": {
            "baseline_keys": list(baseline_keys),
            "recursive": recursive,
            "allow_unverified": allow_unverified,
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
        },
        "summary": {
            "record_count": len(rows),
            "passing_count": sum(1 for row in rows if row["gate"]["passed"]),
            "recommended_record": None if recommendation is None else recommendation["record_key"],
            "recommended_route": None if recommendation is None else recommendation.get("recommended_route"),
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
            if _is_route_record(record)
        )
    for record in records:
        if record.artifact_type != "benchmark_manifest":
            raise ValueError(f"registry record {record.key()!r} is not a benchmark_manifest.")
    return records


def _is_route_record(record: RegistryRecord) -> bool:
    metadata = dict(record.metadata)
    manifest_metadata = _mapping(metadata.get("manifest_metadata"))
    return (
        metadata.get("workflow") == "run_adapter_promotion_workflow"
        or metadata.get("workflow") == "adapter_promotion_workflow"
        or manifest_metadata.get("runner") == "run_adapter_promotion_workflow"
        or metadata.get("route_promotion_status") is not None
    )


def _route_baseline_row(
    record: RegistryRecord,
    *,
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
    fingerprint_cache: MutableMapping[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = Path(record.path)
    manifest, manifest_error = _load_optional_json(manifest_path)
    verification = _verify_manifest(
        manifest_path,
        recursive=recursive,
        fingerprint_cache=fingerprint_cache,
    )
    manifest_metadata = _manifest_metadata(record, manifest)
    route_report_path = _resolve_artifact_path(
        manifest_path,
        manifest,
        artifact_name="route_comparison_report",
    )
    route_comparison, route_report_error = (
        ({}, "route_comparison_report artifact missing")
        if route_report_path is None
        else _load_optional_json(route_report_path)
    )
    route_decision = _mapping(route_comparison.get("promotion_decision"))
    recommended_route = (
        route_decision.get("recommended_route")
        or manifest_metadata.get("recommended_route")
    )
    metrics = _recommended_route_metrics(
        route_comparison,
        manifest_metadata,
        recommended_route=None if recommended_route is None else str(recommended_route),
    )
    route_status = (
        route_decision.get("status")
        or manifest_metadata.get("route_promotion_status")
        or manifest_metadata.get("promotion_status")
    )
    runtime_budget = evaluate_runtime_budget(
        runtime_metrics_from_metadata(manifest_metadata),
        RuntimeBudgetPolicy(
            max_total_seconds=max_runtime_total_seconds,
            max_retrieval_hit_count=max_retrieval_hit_count,
            min_claims_cache_hit_rate=min_claims_cache_hit_rate,
            min_verifier_trace_cache_hit_rate=min_verifier_trace_cache_hit_rate,
        ),
    )
    gate = _gate(
        verification=verification,
        allow_unverified=allow_unverified,
        manifest_error=manifest_error,
        route_report_error=route_report_error,
        route_status=route_status,
        recommended_route=recommended_route,
        metrics=metrics,
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
        runtime_budget=runtime_budget,
    )
    runtime_metrics = dict(runtime_budget.get("metrics") or {})
    return {
        "record_key": record.key(),
        "name": record.name,
        "version": record.version,
        "manifest_path": str(manifest_path),
        "route_report_path": None if route_report_path is None else str(route_report_path),
        "verification": verification,
        "gate": gate,
        "route_promotion_status": route_status,
        "recommended_route": None if recommended_route is None else str(recommended_route),
        "selected": _int_or_none(metrics.get("selected")),
        "decision_accuracy": _float_or_none(metrics.get("decision_accuracy")),
        "false_supported_rate": _float_or_none(metrics.get("false_supported_rate")),
        "false_refuted_rate": _float_or_none(metrics.get("false_refuted_rate")),
        "verified_false_alarm": _float_or_none(metrics.get("verified_false_alarm")),
        "verified_detection": _float_or_none(metrics.get("verified_detection")),
        "mean_duration_seconds": _float_or_none(metrics.get("mean_duration_seconds")),
        "p95_duration_seconds": _float_or_none(metrics.get("p95_duration_seconds")),
        "p99_duration_seconds": _float_or_none(metrics.get("p99_duration_seconds")),
        "max_duration_seconds": _float_or_none(metrics.get("max_duration_seconds")),
        "mean_attempted_route_count": _float_or_none(metrics.get("mean_attempted_route_count")),
        "retrieval_use_rate": _float_or_none(metrics.get("retrieval_use_rate")),
        "invalid_metric_counts": _mapping(metrics.get("invalid_metric_counts")),
        "runtime_total_seconds": _float_or_none(runtime_metrics.get("total_seconds")),
        "runtime_retrieval_hit_count": _float_or_none(runtime_metrics.get("retrieval_hit_count")),
        "claims_cache_hit_rate": _float_or_none(runtime_metrics.get("claims_cache_hit_rate")),
        "verifier_trace_cache_hit_rate": _float_or_none(runtime_metrics.get("verifier_trace_cache_hit_rate")),
        "runtime_budget": runtime_budget,
    }


def _recommended_route_metrics(
    route_comparison: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
    *,
    recommended_route: str | None,
) -> dict[str, Any]:
    by_route = _mapping(route_comparison.get("by_route"))
    if recommended_route is not None:
        metrics = _mapping(by_route.get(recommended_route))
        if metrics:
            return metrics
    return {
        "selected": manifest_metadata.get("recommended_selected"),
        "decision_accuracy": manifest_metadata.get("recommended_decision_accuracy"),
        "false_supported_rate": manifest_metadata.get("recommended_false_supported_rate"),
        "false_refuted_rate": manifest_metadata.get("recommended_false_refuted_rate"),
        "verified_false_alarm": manifest_metadata.get("recommended_verified_false_alarm"),
        "verified_detection": manifest_metadata.get("recommended_verified_detection"),
        "mean_duration_seconds": manifest_metadata.get("recommended_mean_duration_seconds"),
        "p95_duration_seconds": manifest_metadata.get("recommended_p95_duration_seconds"),
        "p99_duration_seconds": manifest_metadata.get("recommended_p99_duration_seconds"),
        "max_duration_seconds": manifest_metadata.get("recommended_max_duration_seconds"),
        "mean_attempted_route_count": manifest_metadata.get("recommended_mean_attempted_route_count"),
        "retrieval_use_rate": manifest_metadata.get("recommended_retrieval_use_rate"),
        "invalid_metric_counts": manifest_metadata.get("recommended_invalid_metric_counts"),
    }


def _gate(
    *,
    verification: Mapping[str, Any],
    allow_unverified: bool,
    manifest_error: str | None,
    route_report_error: str | None,
    route_status: Any,
    recommended_route: Any,
    metrics: Mapping[str, Any],
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
    runtime_budget: Mapping[str, Any],
) -> dict[str, Any]:
    failures = []
    if manifest_error is not None:
        failures.append(f"manifest could not be loaded: {manifest_error}")
    if route_report_error is not None:
        failures.append(f"route comparison report could not be loaded: {route_report_error}")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("manifest verification failed")
    if route_status != "promote":
        failures.append(f"route_promotion_status is {route_status!r}, expected 'promote'")
    if recommended_route is None:
        failures.append("recommended route is missing")
    invalid_metric_counts = {
        key: int(value)
        for key, value in _mapping(metrics.get("invalid_metric_counts")).items()
        if _int_or_none(value) is not None and int(value) > 0
    }
    if invalid_metric_counts:
        failures.append(f"recommended route has invalid source metrics: {invalid_metric_counts}")
    _check_min(failures, "selected", _int_or_none(metrics.get("selected")), min_selected)
    _check_min(failures, "decision_accuracy", _float_or_none(metrics.get("decision_accuracy")), min_decision_accuracy)
    _check_max(
        failures,
        "false_supported_rate",
        _float_or_none(metrics.get("false_supported_rate")),
        max_false_supported_rate,
    )
    _check_min(
        failures,
        "false_refuted_rate",
        _float_or_none(metrics.get("false_refuted_rate")),
        min_false_refuted_rate,
    )
    _check_max(
        failures,
        "verified_false_alarm",
        _float_or_none(metrics.get("verified_false_alarm")),
        max_verified_false_alarm,
    )
    _check_min(
        failures,
        "verified_detection",
        _float_or_none(metrics.get("verified_detection")),
        min_verified_detection,
    )
    _check_max(
        failures,
        "mean_duration_seconds",
        _float_or_none(metrics.get("mean_duration_seconds")),
        max_mean_duration_seconds,
    )
    _check_max(
        failures,
        "p99_duration_seconds",
        _float_or_none(metrics.get("p99_duration_seconds")),
        max_p99_duration_seconds,
    )
    _check_max(
        failures,
        "max_duration_seconds",
        _float_or_none(metrics.get("max_duration_seconds")),
        max_max_duration_seconds,
    )
    _check_max(
        failures,
        "mean_attempted_route_count",
        _float_or_none(metrics.get("mean_attempted_route_count")),
        max_mean_attempted_route_count,
    )
    _check_max(
        failures,
        "retrieval_use_rate",
        _float_or_none(metrics.get("retrieval_use_rate")),
        max_retrieval_use_rate,
    )
    if runtime_budget.get("enabled") and not runtime_budget.get("passed"):
        failures.extend(_runtime_budget_reasons(runtime_budget))
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _runtime_budget_reasons(runtime_budget: Mapping[str, Any]) -> list[str]:
    reasons = []
    for failure in runtime_budget.get("failures", ()):
        if not isinstance(failure, Mapping):
            continue
        metric = failure.get("metric")
        reason = failure.get("reason") or "failed"
        reasons.append(f"runtime_budget: {metric} {reason}")
    return reasons


def _check_min(failures: list[str], metric: str, value: float | int | None, limit: float | int | None) -> None:
    if limit is None:
        return
    if value is None or value < limit:
        failures.append(f"{metric} below {limit}")


def _check_max(failures: list[str], metric: str, value: float | int | None, limit: float | int | None) -> None:
    if limit is None:
        return
    if value is None or value > limit:
        failures.append(f"{metric} above {limit}")


def _decision(
    leaderboard: Sequence[Mapping[str, Any]],
    recommendation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not leaderboard:
        return {
            "status": "no_candidate",
            "recommended_record": None,
            "blocking_reasons": ("no route benchmark_manifest records selected",),
        }
    if recommendation is not None:
        return {
            "status": "promote",
            "recommended_record": recommendation["record_key"],
            "recommended_route": recommendation.get("recommended_route"),
            "blocking_reasons": (),
        }
    return {
        "status": "blocked",
        "recommended_record": None,
        "recommended_route": None,
        "blocking_reasons": tuple(
            f"{row['record_key']}: {reason}"
            for row in leaderboard
            for reason in row["gate"]["blocking_reasons"]
        ),
    }


def _leaderboard_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        not _mapping(row.get("gate")).get("passed", False),
        -(_float_or_none(row.get("decision_accuracy")) or -math.inf),
        _float_or_none(row.get("false_supported_rate")) if row.get("false_supported_rate") is not None else math.inf,
        -(_float_or_none(row.get("false_refuted_rate")) or -math.inf),
        _float_or_none(row.get("p99_duration_seconds")) if row.get("p99_duration_seconds") is not None else math.inf,
        _float_or_none(row.get("mean_duration_seconds")) if row.get("mean_duration_seconds") is not None else math.inf,
        _float_or_none(row.get("retrieval_use_rate")) if row.get("retrieval_use_rate") is not None else math.inf,
        _float_or_none(row.get("runtime_total_seconds")) if row.get("runtime_total_seconds") is not None else math.inf,
        -(_int_or_none(row.get("selected")) or -1),
        str(row.get("record_key")),
    )


def _manifest_metadata(record: RegistryRecord, manifest: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(record.metadata)
    manifest_metadata = _mapping(manifest.get("metadata"))
    merged = dict(manifest_metadata)
    merged.update(metadata)
    nested = _mapping(metadata.get("manifest_metadata"))
    merged.update(nested)
    return merged


def _verify_manifest(
    manifest_path: Path,
    *,
    recursive: bool,
    fingerprint_cache: MutableMapping[str, dict[str, Any]],
) -> dict[str, Any]:
    try:
        return load_and_verify_artifact_manifest(
            manifest_path,
            recursive=recursive,
            fingerprint_cache=fingerprint_cache,
        ).to_dict()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
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
                    "actual": str(exc),
                }
            ],
            "nested": [],
        }


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


def _load_optional_json(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, str(exc)
    if not isinstance(payload, dict):
        return {}, f"{path} did not contain a JSON object"
    return payload, None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = compare_route_baselines(
        registry_path=args.registry,
        baseline_keys=tuple(args.baseline_key or ()),
        recursive=not args.no_recursive,
        allow_unverified=bool(args.allow_unverified),
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
        notes=args.note,
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote route baseline comparison to {output_path}")
    decision = payload["decision"]
    print(
        "route_baseline_comparison="
        f"{decision['status']} recommended={decision.get('recommended_record')} "
        f"route={decision.get('recommended_route')}"
    )
    if args.fail_on_blocked and decision["status"] != "promote":
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare registered verifier-route promotion benchmark manifests"
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
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless a route baseline promotes")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
