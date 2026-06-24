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
    drift_gate_enabled = any(value is not None for value in gates.values())
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
    return [
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
        _min_metric(
            "n_traces",
            _finite_float(current_summary.get("n_traces")),
            gates.get("min_current_trace_count"),
        ),
    ]


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
            },
        )
        registry.save_json()
    return output


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
    current: Any = payload
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return _finite_float(current)


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


def _optional_non_negative_int(value: int | None) -> int | None:
    if value is None:
        return None
    numeric = int(value)
    if numeric < 0:
        raise ValueError("min_current_trace_count must be non-negative.")
    return numeric


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


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
    parser.add_argument("--min-current-trace-count", type=int, default=None)
    parser.add_argument("--compact-json", action="store_true",
                        help="write minified drift report and manifest JSON")
    parser.add_argument("--fail-on-drift", action="store_true",
                        help="exit non-zero when drift gates block")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
