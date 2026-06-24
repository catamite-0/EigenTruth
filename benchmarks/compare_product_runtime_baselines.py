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
from eigentruth.registry import ArtifactRegistry, RegistryRecord, build_artifact_manifest  # noqa: E402


def compare_product_runtime_baselines(
    *,
    current_path: str | Path,
    baseline_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    baseline_key: str | None = None,
    baseline_name: str | None = None,
    baseline_version: str | None = None,
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
    current_report_path = Path(current_path)
    baseline_report = _load_runtime_baseline(source["path"])
    current_report = _load_runtime_baseline(current_report_path)
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
        _mapping(baseline_report.get("summary")),
        _mapping(current_report.get("summary")),
        gates=gates,
    )
    gate_enabled = any(value is not None for value in gates.values())
    blocking_reasons = tuple(
        str(metric["reason"])
        for metric in metrics
        if metric.get("status") == "blocked" and metric.get("reason")
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
        },
        "current": {
            "path": str(current_report_path),
            "status": current_report.get("status"),
        },
        "metrics": metrics,
        "paths": {
            "report": None if resolved_report_path is None else str(resolved_report_path),
            "artifact_manifest": None if resolved_manifest_path is None else str(resolved_manifest_path),
        },
        "config": {
            **gates,
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
        if registry_path is not None or baseline_key or baseline_name or baseline_version:
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
    artifacts = {
        "product_runtime_drift_report": report_path,
        "baseline_product_runtime_baseline": baseline_path,
        "current_product_runtime_baseline": current_path,
    }
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
