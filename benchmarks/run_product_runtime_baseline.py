"""Build a ProductTrace runtime baseline report.

This workflow aggregates already-emitted product traces. It does not run a
model, verifier, retriever, or external service. The purpose is to make the
control-plane runtime budget auditable across a sample of real or demo requests.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import (  # noqa: E402
    planned_artifact_manifest_summary,
    reject_bounded_product_trace,
    strict_bool,
)
from eigentruth.control import (  # noqa: E402
    ProductPromotionContract,
    ProductRuntimeBudgetPolicy,
    evaluate_product_runtime_budget,
    product_runtime_metrics,
)
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class ProductRuntimeBaselineConfig:
    """Configuration for a ProductTrace runtime baseline report."""

    trace_paths: Sequence[str | Path]
    report_path: str | Path
    policy: ProductRuntimeBudgetPolicy | Mapping[str, Any] | None = None
    policy_path: str | Path | None = None
    promotion_contract_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    compact_json: bool = False

    def __post_init__(self) -> None:
        trace_paths = tuple(Path(path) for path in self.trace_paths)
        if not trace_paths:
            raise ValueError("at least one ProductTrace path is required.")
        if self.policy is not None and (self.policy_path is not None or self.promotion_contract_path is not None):
            raise ValueError("policy object is mutually exclusive with policy_path and promotion_contract_path.")
        if self.policy_path is not None and self.promotion_contract_path is not None:
            raise ValueError("policy_path and promotion_contract_path are mutually exclusive.")
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        object.__setattr__(self, "trace_paths", trace_paths)
        object.__setattr__(self, "report_path", Path(self.report_path))
        if self.policy_path is not None:
            object.__setattr__(self, "policy_path", Path(self.policy_path))
        if self.promotion_contract_path is not None:
            object.__setattr__(self, "promotion_contract_path", Path(self.promotion_contract_path))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        """Return the output artifact manifest path."""
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return Path(self.report_path).with_name("product-runtime-baseline-artifact-manifest.json")


def build_product_runtime_baseline(config: ProductRuntimeBaselineConfig) -> dict[str, Any]:
    """Aggregate ProductTrace runtime metrics and optional budget results."""
    policy, policy_source = _load_policy(config)
    traces = tuple((path, _load_trace(path)) for path in config.trace_paths)
    records = tuple(
        _trace_record(path, trace, policy=policy)
        for path, trace in traces
    )
    budget_summary = _budget_summary(records, policy=policy)
    status = _status_from_budget(budget_summary)
    report = {
        "schema_version": 1,
        "workflow": "product_runtime_baseline",
        "status": status,
        "decision": {
            "status": status,
            "blocking_reasons": _blocking_reasons(budget_summary),
        },
        "summary": _aggregate_records(records),
        "budget": budget_summary,
        "traces": list(records),
        "paths": {
            "report": str(config.report_path),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "policy": None if config.policy_path is None else str(config.policy_path),
            "promotion_contract": (
                None if config.promotion_contract_path is None else str(config.promotion_contract_path)
            ),
            "traces": [str(path) for path, _trace in traces],
        },
        "config": {
            "trace_count": len(traces),
            "policy_source": policy_source,
            "compact_json": config.compact_json,
            "metadata": dict(config.metadata),
        },
    }
    _write_report_and_manifest(config, report)
    _record_registry(config, report)
    return report


def _write_report_and_manifest(
    config: ProductRuntimeBaselineConfig,
    report: dict[str, Any],
) -> dict[str, Any]:
    artifacts = _artifact_paths(config)
    report["artifact_manifest_summary"] = planned_artifact_manifest_summary(
        artifacts,
        assume_file_paths=(config.report_path,),
    )
    _write_report(config.report_path, report, compact=config.compact_json)
    return _write_artifact_manifest(config, report, artifacts=artifacts)


def _trace_record(
    path: Path,
    trace: Mapping[str, Any],
    *,
    policy: ProductRuntimeBudgetPolicy | None,
) -> dict[str, Any]:
    metrics = product_runtime_metrics(trace)
    budget = None if policy is None else evaluate_product_runtime_budget(trace, policy)
    return {
        "path": str(path),
        "request_id": trace.get("request_id"),
        "metrics": _compact_metrics(metrics),
        "budget": budget,
    }


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "has_runtime_trace": bool(metrics.get("has_runtime_trace")),
        "total_seconds": metrics.get("total_seconds"),
        "accounted_seconds": metrics.get("accounted_seconds"),
        "unaccounted_seconds": metrics.get("unaccounted_seconds"),
        "measured_phases": metrics.get("measured_phases"),
        "phase_seconds": dict(_mapping(metrics.get("phase_seconds"))),
        "phase_counts": dict(_mapping(metrics.get("phase_counts"))),
        "phase_p95_seconds": dict(_mapping(metrics.get("phase_p95_seconds"))),
        "phase_p99_seconds": dict(_mapping(metrics.get("phase_p99_seconds"))),
        "slowest_phase": metrics.get("slowest_phase"),
        "cache_hit_rate": metrics.get("cache_hit_rate"),
        "named_cache_hit_rates": dict(_mapping(metrics.get("named_cache_hit_rates"))),
        "route_cost_summary": dict(_mapping(metrics.get("route_cost_summary"))),
        "mean_route_duration_seconds": metrics.get("mean_route_duration_seconds"),
        "p95_route_duration_seconds": metrics.get("p95_route_duration_seconds"),
        "p99_route_duration_seconds": metrics.get("p99_route_duration_seconds"),
        "max_route_duration_seconds": metrics.get("max_route_duration_seconds"),
        "mean_attempted_route_count": metrics.get("mean_attempted_route_count"),
        "retrieval_use_rate": metrics.get("retrieval_use_rate"),
        "retrieval_hit_count": metrics.get("retrieval_hit_count"),
        "mean_retrieval_hits": metrics.get("mean_retrieval_hits"),
        "verification_stage_summary": dict(_mapping(metrics.get("verification_stage_summary"))),
        "verification_stage_enabled": bool(metrics.get("verification_stage_enabled")),
        "verification_stage_skipped": bool(metrics.get("verification_stage_skipped")),
        "verification_skip_rate": metrics.get("verification_skip_rate"),
        "verified_claim_count": metrics.get("verified_claim_count"),
        "verifier_saved_claim_count": metrics.get("verifier_saved_claim_count"),
    }


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [_mapping(record.get("metrics")) for record in records]
    return {
        "n_traces": len(records),
        "runtime_trace_count": sum(1 for item in metrics if bool(item.get("has_runtime_trace"))),
        "total_seconds": _numeric_summary(item.get("total_seconds") for item in metrics),
        "accounted_seconds": _numeric_summary(item.get("accounted_seconds") for item in metrics),
        "measured_phases": _numeric_summary(item.get("measured_phases") for item in metrics),
        "mean_route_duration_seconds": _numeric_summary(
            item.get("mean_route_duration_seconds") for item in metrics
        ),
        "p95_route_duration_seconds": _numeric_summary(
            item.get("p95_route_duration_seconds") for item in metrics
        ),
        "p99_route_duration_seconds": _numeric_summary(
            item.get("p99_route_duration_seconds") for item in metrics
        ),
        "max_route_duration_seconds": _numeric_summary(
            item.get("max_route_duration_seconds") for item in metrics
        ),
        "mean_attempted_route_count": _numeric_summary(
            item.get("mean_attempted_route_count") for item in metrics
        ),
        "retrieval_use_rate": _numeric_summary(item.get("retrieval_use_rate") for item in metrics),
        "retrieval_hit_count": _numeric_summary(item.get("retrieval_hit_count") for item in metrics),
        "cache_hit_rate": _numeric_summary(item.get("cache_hit_rate") for item in metrics),
        "verification_skip_rate": _numeric_summary(item.get("verification_skip_rate") for item in metrics),
        "verified_claim_count": _numeric_summary(item.get("verified_claim_count") for item in metrics),
        "verifier_saved_claim_count": _numeric_summary(item.get("verifier_saved_claim_count") for item in metrics),
        "verification_stage": _aggregate_verification_stage(metrics),
        "phases": _aggregate_phases(metrics),
        "routes": _aggregate_routes(metrics),
    }


def _aggregate_phases(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    phase_names = sorted(
        {
            str(name)
            for item in metrics
            for name in _mapping(item.get("phase_seconds")).keys()
        }
    )
    phases = {}
    for phase in phase_names:
        values = [
            _mapping(item.get("phase_seconds")).get(phase)
            for item in metrics
            if phase in _mapping(item.get("phase_seconds"))
        ]
        counts = [
            _finite_float(_mapping(item.get("phase_counts")).get(phase))
            for item in metrics
            if phase in _mapping(item.get("phase_counts"))
        ]
        phases[phase] = {
            "trace_observations": len(values),
            "phase_count": int(sum(value for value in counts if value is not None)),
            "seconds": _numeric_summary(values),
            "p95_seconds": _numeric_summary(
                _mapping(item.get("phase_p95_seconds")).get(phase)
                for item in metrics
                if phase in _mapping(item.get("phase_p95_seconds"))
            ),
            "p99_seconds": _numeric_summary(
                _mapping(item.get("phase_p99_seconds")).get(phase)
                for item in metrics
                if phase in _mapping(item.get("phase_p99_seconds"))
            ),
        }
    return phases


def _aggregate_routes(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = [_mapping(item.get("route_cost_summary")) for item in metrics]
    by_route_names = sorted(
        {
            str(route)
            for summary in summaries
            for route in _mapping(summary.get("by_route")).keys()
        }
    )
    return {
        "overall": _aggregate_route_summaries(summaries),
        "by_route": {
            route: _aggregate_route_summaries(
                _mapping(_mapping(summary.get("by_route")).get(route))
                for summary in summaries
                if route in _mapping(summary.get("by_route"))
            )
            for route in by_route_names
        },
    }


def _aggregate_verification_stage(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = [_mapping(item.get("verification_stage_summary")) for item in metrics]
    reason_counts: dict[str, int] = {}
    triggered_feature_counts: dict[str, int] = {}
    triggered_metadata_counts: dict[str, int] = {}
    for summary in summaries:
        reason = summary.get("reason")
        if reason is not None:
            reason_key = str(reason)
            reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
        _merge_counts(triggered_feature_counts, _mapping(summary.get("triggered_feature_counts")))
        _merge_counts(triggered_metadata_counts, _mapping(summary.get("triggered_metadata_counts")))
    enabled_count = sum(1 for summary in summaries if bool(summary.get("enabled")))
    skipped_count = sum(1 for summary in summaries if bool(summary.get("skipped")))
    saved_claim_count = _sum_float(summaries, "saved_claim_count")
    verified_claim_count = _sum_float(summaries, "verified_claim_count")
    claim_count = _sum_float(summaries, "claim_count")
    return {
        "source_trace_count": len(summaries),
        "enabled_trace_count": enabled_count,
        "skipped_trace_count": skipped_count,
        "run_verifier_trace_count": sum(1 for summary in summaries if summary.get("run_verifier") is True),
        "skip_decision_rate": _safe_div(skipped_count, len(summaries)),
        "claim_count": claim_count,
        "saved_claim_count": saved_claim_count,
        "verified_claim_count": verified_claim_count,
        "claim_skip_rate": _safe_div(saved_claim_count, claim_count),
        "per_trace_skip_rate": _numeric_summary(summary.get("skip_rate") for summary in summaries),
        "reason_counts": reason_counts,
        "triggered_feature_counts": triggered_feature_counts,
        "triggered_metadata_counts": triggered_metadata_counts,
    }


def _merge_counts(target: dict[str, int], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        numeric = _finite_float(value)
        if numeric is None:
            continue
        target[str(key)] = target.get(str(key), 0) + int(numeric)


def _aggregate_route_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = tuple(_mapping(summary) for summary in summaries)
    total = _sum_int(summaries, "total")
    routed_total = _sum_int(summaries, "routed_total")
    duration_observations = _sum_int(summaries, "duration_observations")
    selected_duration_observations = _sum_int(summaries, "selected_route_duration_observations")
    attempted_observations = _sum_int(summaries, "attempted_route_count_observations")
    total_duration = _sum_float(summaries, "total_duration_seconds")
    total_selected_duration = _sum_float(summaries, "total_selected_route_duration_seconds")
    total_attempted = _sum_float(summaries, "total_attempted_route_count")
    used_retrieval_count = _sum_int(summaries, "used_retrieval_count")
    retrieval_hit_count = _sum_int(summaries, "retrieval_hit_count")
    return {
        "source_trace_count": len(summaries),
        "total": total,
        "routed_total": routed_total,
        "unrouted_total": None if total is None or routed_total is None else total - routed_total,
        "duration_observations": duration_observations,
        "total_duration_seconds": total_duration,
        "mean_duration_seconds": _safe_div(total_duration, duration_observations),
        "per_trace_mean_duration_seconds": _numeric_summary(
            summary.get("mean_duration_seconds") for summary in summaries
        ),
        "per_trace_p95_duration_seconds": _numeric_summary(
            summary.get("p95_duration_seconds") for summary in summaries
        ),
        "per_trace_p99_duration_seconds": _numeric_summary(
            summary.get("p99_duration_seconds") for summary in summaries
        ),
        "max_duration_seconds": _max_numeric(summary.get("max_duration_seconds") for summary in summaries),
        "selected_route_duration_observations": selected_duration_observations,
        "total_selected_route_duration_seconds": total_selected_duration,
        "mean_selected_route_duration_seconds": _safe_div(
            total_selected_duration,
            selected_duration_observations,
        ),
        "attempted_route_count_observations": attempted_observations,
        "total_attempted_route_count": total_attempted,
        "mean_attempted_route_count": _safe_div(total_attempted, attempted_observations),
        "used_retrieval_count": used_retrieval_count,
        "retrieval_use_rate": _safe_div(used_retrieval_count, total),
        "retrieval_hit_count": retrieval_hit_count,
        "mean_retrieval_hits": _safe_div(retrieval_hit_count, total),
    }


def _budget_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    policy: ProductRuntimeBudgetPolicy | None,
) -> dict[str, Any]:
    if policy is None:
        return {
            "enabled": False,
            "passed": None,
            "policy": None,
            "passed_count": None,
            "failed_count": None,
            "failure_counts_by_metric": {},
        }
    budgets = [_mapping(record.get("budget")) for record in records]
    failed = [budget for budget in budgets if budget.get("passed") is not True]
    failure_counts: dict[str, int] = {}
    for budget in failed:
        for failure in _sequence(budget.get("failures")):
            if not isinstance(failure, Mapping):
                continue
            metric = str(failure.get("metric", "unknown"))
            failure_counts[metric] = failure_counts.get(metric, 0) + 1
    return {
        "enabled": policy.enabled(),
        "passed": not failed,
        "policy": policy.to_dict(),
        "passed_count": len(budgets) - len(failed),
        "failed_count": len(failed),
        "failure_counts_by_metric": failure_counts,
    }


def _status_from_budget(budget: Mapping[str, Any]) -> str:
    if not bool(budget.get("enabled")):
        return "observed"
    return "promote" if budget.get("passed") is True else "blocked"


def _blocking_reasons(budget: Mapping[str, Any]) -> tuple[str, ...]:
    if not bool(budget.get("enabled")):
        return ()
    if budget.get("passed") is True:
        return ()
    counts = _mapping(budget.get("failure_counts_by_metric"))
    if not counts:
        return ("one or more runtime budget checks failed",)
    return tuple(
        f"{metric}: failed {count} trace(s)"
        for metric, count in sorted(counts.items())
    )


def _load_policy(config: ProductRuntimeBaselineConfig) -> tuple[ProductRuntimeBudgetPolicy | None, str | None]:
    if config.policy is not None:
        return (
            config.policy
            if isinstance(config.policy, ProductRuntimeBudgetPolicy)
            else ProductRuntimeBudgetPolicy.from_mapping(config.policy),
            "inline",
        )
    if config.policy_path is not None:
        payload = _load_json(config.policy_path)
        return ProductRuntimeBudgetPolicy.from_mapping(payload), str(config.policy_path)
    if config.promotion_contract_path is not None:
        contract = ProductPromotionContract.from_json(config.promotion_contract_path)
        return contract.runtime_budget_policy, str(config.promotion_contract_path)
    return None, None


def _write_artifact_manifest(
    config: ProductRuntimeBaselineConfig,
    report: Mapping[str, Any],
    *,
    artifacts: Mapping[str, str | Path | None] | None = None,
) -> dict[str, Any]:
    manifest = build_artifact_manifest(
        _artifact_paths(config) if artifacts is None else artifacts,
        root=config.resolved_artifact_manifest_path.parent,
        metadata={
            "runner": "run_product_runtime_baseline",
            "status": report.get("status"),
            "trace_count": len(config.trace_paths),
            "budget_enabled": _mapping(report.get("budget")).get("enabled"),
            "budget_passed": _mapping(report.get("budget")).get("passed"),
            "compact_json": config.compact_json,
            **dict(config.metadata),
        },
    )
    _write_report(config.resolved_artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _artifact_paths(config: ProductRuntimeBaselineConfig) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {
        "product_runtime_baseline_report": config.report_path,
        "policy": config.policy_path,
        "promotion_contract": config.promotion_contract_path,
    }
    for index, trace_path in enumerate(config.trace_paths):
        artifacts[f"trace_{index:04d}_{_safe_artifact_name(trace_path.stem)}"] = trace_path
    return artifacts


def _record_registry(config: ProductRuntimeBaselineConfig, report: Mapping[str, Any]) -> None:
    if config.registry_path is None:
        return
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_product_runtime_baseline(
        name=str(config.name),
        path=config.report_path,
        version=str(config.version),
        metadata={
            "workflow": "run_product_runtime_baseline",
            "status": report.get("status"),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "trace_count": len(config.trace_paths),
            "budget_enabled": _mapping(report.get("budget")).get("enabled"),
            "budget_passed": _mapping(report.get("budget")).get("passed"),
            "failed_count": _mapping(report.get("budget")).get("failed_count"),
            "compact_json": config.compact_json,
            **dict(config.metadata),
        },
    )
    registry.save_json()


def _load_trace(path: str | Path) -> dict[str, Any]:
    payload = _load_json(path)
    reject_bounded_product_trace(payload, path=path)
    if "runtime_trace" not in payload and "verification_results" not in payload:
        raise ValueError(f"ProductTrace JSON is missing runtime/control fields: {path}")
    return payload


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be an object: {path}")
    return dict(payload)


def _write_report(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json_text(payload, compact=compact), encoding="utf-8")


def _json_text(payload: Any, *, compact: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _numeric_summary(values: Sequence[Any] | Any) -> dict[str, Any]:
    raw_values = tuple(values)
    finite_values = [
        numeric
        for value in raw_values
        if (numeric := _finite_float(value)) is not None
    ]
    if not finite_values:
        return {
            "count": 0,
            "missing_or_nonfinite": len(raw_values),
            "mean": None,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    total = sum(finite_values)
    return {
        "count": len(finite_values),
        "missing_or_nonfinite": len(raw_values) - len(finite_values),
        "mean": total / len(finite_values),
        "min": min(finite_values),
        "p50": _percentile(finite_values, 50.0),
        "p95": _percentile(finite_values, 95.0),
        "p99": _percentile(finite_values, 99.0),
        "max": max(finite_values),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return lower + (upper - lower) * (rank - lower_index)


def _sum_int(items: Sequence[Mapping[str, Any]], field_name: str) -> int | None:
    values = [_finite_float(item.get(field_name)) for item in items]
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return int(sum(finite_values))


def _sum_float(items: Sequence[Mapping[str, Any]], field_name: str) -> float | None:
    values = [_finite_float(item.get(field_name)) for item in items]
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return float(sum(finite_values))


def _max_numeric(values: Sequence[Any] | Any) -> float | None:
    finite_values = [
        numeric
        for value in values
        if (numeric := _finite_float(value)) is not None
    ]
    return None if not finite_values else max(finite_values)


def _safe_div(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


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


def _safe_artifact_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return cleaned or "trace"


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


def _config_from_args(args: argparse.Namespace) -> ProductRuntimeBaselineConfig:
    return ProductRuntimeBaselineConfig(
        trace_paths=tuple(args.trace),
        report_path=Path(args.json),
        policy_path=Path(args.policy) if args.policy else None,
        promotion_contract_path=Path(args.promotion_contract) if args.promotion_contract else None,
        artifact_manifest_path=Path(args.artifact_manifest) if args.artifact_manifest else None,
        registry_path=Path(args.registry) if args.registry else None,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = build_product_runtime_baseline(_config_from_args(args))
    print(_json_text(report, compact=bool(args.compact_json)), end="")
    if args.fail_on_blocked and report["status"] == "blocked":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a ProductTrace runtime baseline report")
    parser.add_argument("--trace", action="append", required=True, help="ProductTrace JSON path; repeatable")
    parser.add_argument("--json", required=True, help="output baseline report JSON path")
    parser.add_argument("--policy", default=None, help="ProductRuntimeBudgetPolicy JSON path")
    parser.add_argument("--promotion-contract", default=None, help="ProductPromotionContract/release report JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest output path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry product runtime baseline name")
    parser.add_argument("--version", default=None, help="registry product runtime baseline version")
    parser.add_argument("--metadata", action="append", default=[], help="metadata key=value; repeatable")
    parser.add_argument("--compact-json", action="store_true",
                        help="write minified baseline report and manifest JSON")
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
