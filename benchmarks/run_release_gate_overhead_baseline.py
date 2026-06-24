"""Aggregate release-candidate registry workflow overhead reports.

This script reads already-written ``run_release_candidate_registry_workflow.py``
JSON payloads. It does not rerun release gates or verify manifests; it turns the
workflow timing and artifact-cache summaries into a lightweight baseline report.
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

from benchmarks.config_utils import strict_bool  # noqa: E402
from eigentruth.registry import ArtifactRegistry  # noqa: E402


@dataclass(frozen=True)
class ReleaseGateOverheadBaselineConfig:
    """Configuration for aggregating release-gate overhead reports."""

    report_paths: Sequence[str | Path]
    output_path: str | Path
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    max_total_seconds: float | None = None
    max_phase_total_seconds: float | None = None
    min_fingerprint_cache_hit_rate: float | None = None
    min_last_fingerprint_cache_hit_rate: float | None = None
    min_report_count: int | None = None
    compact_json: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.report_paths:
            raise ValueError("at least one release workflow report path is required.")
        object.__setattr__(self, "report_paths", tuple(Path(path) for path in self.report_paths))
        object.__setattr__(self, "output_path", Path(self.output_path))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
            if not self.name or not self.version:
                raise ValueError("registry_path requires name and version.")
        if (self.name is None) != (self.version is None):
            raise ValueError("registry recording requires both name and version.")
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))
        for field_name in (
            "max_total_seconds",
            "max_phase_total_seconds",
            "min_fingerprint_cache_hit_rate",
            "min_last_fingerprint_cache_hit_rate",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _non_negative_float(value, name=field_name))
        if self.min_report_count is not None and int(self.min_report_count) < 0:
            raise ValueError("min_report_count must be non-negative.")
        if self.min_report_count is not None:
            object.__setattr__(self, "min_report_count", int(self.min_report_count))
        object.__setattr__(self, "metadata", dict(self.metadata))


def build_release_gate_overhead_baseline(config: ReleaseGateOverheadBaselineConfig) -> dict[str, Any]:
    """Build and write a release-gate overhead baseline from workflow reports."""
    records = tuple(_workflow_record(path) for path in config.report_paths)
    summary = _baseline_summary(records)
    gates = {
        "max_total_seconds": config.max_total_seconds,
        "max_phase_total_seconds": config.max_phase_total_seconds,
        "min_fingerprint_cache_hit_rate": config.min_fingerprint_cache_hit_rate,
        "min_last_fingerprint_cache_hit_rate": config.min_last_fingerprint_cache_hit_rate,
        "min_report_count": config.min_report_count,
    }
    metrics = _gate_metrics(summary, gates)
    blocking_reasons = tuple(
        str(metric["reason"]) for metric in metrics if metric.get("status") == "blocked" and metric.get("reason")
    )
    gate_enabled = any(value is not None for value in gates.values())
    status = "blocked" if blocking_reasons else ("promote" if gate_enabled else "observed")
    payload = {
        "schema_version": 1,
        "workflow": "release_gate_overhead_baseline",
        "status": status,
        "decision": {
            "status": status,
            "gate_enabled": gate_enabled,
            "blocking_reasons": blocking_reasons,
        },
        "summary": summary,
        "metrics": metrics,
        "records": records,
        "paths": {
            "report": str(config.output_path),
            "source_reports": tuple(str(path) for path in config.report_paths),
            "registry": None if config.registry_path is None else str(config.registry_path),
        },
        "config": {
            **gates,
            "compact_json": config.compact_json,
            "metadata": dict(config.metadata),
        },
        "registry_record": None if config.registry_path is None else f"report:{config.name}:{config.version}",
    }
    _write_json(config.output_path, payload, compact=config.compact_json)
    if config.registry_path is not None:
        _record_registry(config, payload)
    return payload


def _workflow_record(path: Path) -> dict[str, Any]:
    payload = _load_workflow_report(path)
    timing = _mapping(payload.get("timing"))
    phases = _mapping(timing.get("phases"))
    artifact_cache = _mapping(payload.get("artifact_cache"))
    fingerprint_cache = _cache_record(_mapping(artifact_cache.get("artifact_fingerprint_cache")))
    json_cache = _cache_record(_mapping(artifact_cache.get("artifact_json_cache")))
    return {
        "path": str(path),
        "status": _mapping(payload.get("decision")).get("status"),
        "release_candidate_status": _mapping(payload.get("decision")).get("release_candidate_status"),
        "registry_record": _mapping(payload.get("decision")).get("registry_record"),
        "fingerprint_cache_path": _mapping(payload.get("config")).get("fingerprint_cache"),
        "total_seconds": _finite_float(timing.get("total_seconds")),
        "phase_total_seconds": _finite_float(timing.get("phase_total_seconds")),
        "phases": {
            str(name): {
                "seconds": _finite_float(_mapping(phase).get("seconds")),
                "skipped": bool(_mapping(phase).get("skipped")),
            }
            for name, phase in phases.items()
            if isinstance(phase, Mapping)
        },
        "artifact_fingerprint_cache": fingerprint_cache,
        "artifact_json_cache": json_cache,
    }


def _load_workflow_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"release workflow report must be a JSON object: {path}")
    if payload.get("workflow") != "release_candidate_registry_workflow":
        raise ValueError(f"unexpected workflow in release workflow report: {path}")
    return dict(payload)


def _baseline_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    phase_names = sorted({str(name) for record in records for name in _mapping(record.get("phases")).keys()})
    fingerprint_cache = _cache_summary(records, cache_name="artifact_fingerprint_cache")
    return {
        "report_count": len(records),
        "status_counts": _counts(record.get("status") for record in records),
        "release_candidate_status_counts": _counts(record.get("release_candidate_status") for record in records),
        "total_seconds": _numeric_summary(record.get("total_seconds") for record in records),
        "phase_total_seconds": _numeric_summary(record.get("phase_total_seconds") for record in records),
        "phases": {
            phase: {
                "seconds": _numeric_summary(
                    _mapping(_mapping(record.get("phases")).get(phase)).get("seconds") for record in records
                ),
                "skipped_count": sum(
                    1 for record in records if bool(_mapping(_mapping(record.get("phases")).get(phase)).get("skipped"))
                ),
            }
            for phase in phase_names
        },
        "artifact_fingerprint_cache": fingerprint_cache,
        "artifact_json_cache": _cache_summary(records, cache_name="artifact_json_cache"),
        "optimization": _optimization_summary(records, fingerprint_cache=fingerprint_cache),
    }


def _cache_record(cache: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "requests": _finite_int(cache.get("requests")),
        "hits": _finite_int(cache.get("hits")),
        "misses": _finite_int(cache.get("misses")),
        "entries": _finite_int(cache.get("entries")),
        "hit_rate": _finite_float(cache.get("hit_rate")),
    }


def _cache_summary(records: Sequence[Mapping[str, Any]], *, cache_name: str) -> dict[str, Any]:
    cache_records = tuple(_mapping(record.get(cache_name)) for record in records)
    requests = _sum_int(cache_records, "requests")
    hits = _sum_int(cache_records, "hits")
    misses = _sum_int(cache_records, "misses")
    hit_rates = tuple(cache.get("hit_rate") for cache in cache_records)
    last_cache = cache_records[-1] if cache_records else {}
    return {
        "report_count": len(cache_records),
        "requests": requests,
        "hits": hits,
        "misses": misses,
        "weighted_hit_rate": _safe_div(hits, requests),
        "last_hit_rate": _finite_float(last_cache.get("hit_rate")),
        "hit_rate": _numeric_summary(hit_rates),
        "entries": _numeric_summary(cache.get("entries") for cache in cache_records),
        "warm_report_count": sum(1 for value in hit_rates if (_finite_float(value) or 0.0) > 0.0),
    }


def _optimization_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    fingerprint_cache: Mapping[str, Any],
) -> dict[str, Any]:
    phase_means = []
    for phase in sorted({str(name) for record in records for name in _mapping(record.get("phases")).keys()}):
        summary = _numeric_summary(
            _mapping(_mapping(record.get("phases")).get(phase)).get("seconds") for record in records
        )
        mean = _finite_float(summary.get("mean"))
        if mean is not None:
            phase_means.append((phase, mean))
    top_phase = None if not phase_means else max(phase_means, key=lambda item: item[1])
    cache_status = _fingerprint_cache_status(fingerprint_cache)
    recommendations = []
    if top_phase is not None:
        recommendations.append(
            {
                "kind": "phase_hotspot",
                "phase": top_phase[0],
                "mean_seconds": top_phase[1],
                "reason": "This release-gate phase has the highest observed mean runtime.",
            }
        )
    if cache_status != "warm_reuse_observed":
        recommendations.append(
            {
                "kind": "fingerprint_cache_reuse",
                "status": cache_status,
                "reason": "Repeat release gates should use --fingerprint-cache and show a high last-run hit rate.",
            }
        )
    return {
        "top_phase": None if top_phase is None else {"name": top_phase[0], "mean_seconds": top_phase[1]},
        "fingerprint_cache_status": cache_status,
        "recommendations": recommendations,
    }


def _fingerprint_cache_status(cache: Mapping[str, Any]) -> str:
    requests = _finite_float(cache.get("requests"))
    hits = _finite_float(cache.get("hits"))
    last_hit_rate = _finite_float(cache.get("last_hit_rate"))
    if requests in (None, 0.0):
        return "no_fingerprint_requests"
    if hits in (None, 0.0):
        return "no_reuse_observed"
    if last_hit_rate is not None and last_hit_rate >= 0.90:
        return "warm_reuse_observed"
    return "partial_reuse_observed"


def _gate_metrics(summary: Mapping[str, Any], gates: Mapping[str, Any]) -> list[dict[str, Any]]:
    fingerprint_cache = _mapping(summary.get("artifact_fingerprint_cache"))
    return [
        _max_metric(
            "total_seconds.max",
            _finite_float(_mapping(summary.get("total_seconds")).get("max")),
            gates.get("max_total_seconds"),
        ),
        _max_metric(
            "phase_total_seconds.max",
            _finite_float(_mapping(summary.get("phase_total_seconds")).get("max")),
            gates.get("max_phase_total_seconds"),
        ),
        _min_metric(
            "artifact_fingerprint_cache.weighted_hit_rate",
            _finite_float(fingerprint_cache.get("weighted_hit_rate")),
            gates.get("min_fingerprint_cache_hit_rate"),
        ),
        _min_metric(
            "artifact_fingerprint_cache.last_hit_rate",
            _finite_float(fingerprint_cache.get("last_hit_rate")),
            gates.get("min_last_fingerprint_cache_hit_rate"),
        ),
        _min_metric(
            "report_count",
            _finite_float(summary.get("report_count")),
            gates.get("min_report_count"),
        ),
    ]


def _max_metric(name: str, observed: float | None, threshold: Any) -> dict[str, Any]:
    row = {
        "metric": name,
        "comparison": "max",
        "observed": observed,
        "threshold": threshold,
    }
    return _gate_metric(row, value=observed, threshold=threshold, fail=lambda value, limit: value > limit)


def _min_metric(name: str, observed: float | None, threshold: Any) -> dict[str, Any]:
    row = {
        "metric": name,
        "comparison": "min",
        "observed": observed,
        "threshold": threshold,
    }
    return _gate_metric(row, value=observed, threshold=threshold, fail=lambda value, limit: value < limit)


def _gate_metric(
    row: dict[str, Any],
    *,
    value: float | None,
    threshold: Any,
    fail: Any,
) -> dict[str, Any]:
    limit = _finite_float(threshold)
    if limit is None:
        row["status"] = "observed"
        row["reason"] = None
        return row
    if value is None:
        row["status"] = "blocked"
        row["reason"] = f"{row['metric']}: missing or non-finite value"
        return row
    if fail(value, limit):
        row["status"] = "blocked"
        comparator = "below" if row.get("comparison") == "min" else "exceeded"
        row["reason"] = f"{row['metric']}: {value:.6g} {comparator} gate {limit:.6g}"
        return row
    row["status"] = "pass"
    row["reason"] = None
    return row


def _record_registry(config: ReleaseGateOverheadBaselineConfig, payload: Mapping[str, Any]) -> None:
    summary = _mapping(payload.get("summary"))
    fingerprint_cache = _mapping(summary.get("artifact_fingerprint_cache"))
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_report(
        name=str(config.name),
        path=config.output_path,
        version=str(config.version),
        metadata={
            "workflow": "release_gate_overhead_baseline",
            "status": payload.get("status"),
            "report_count": summary.get("report_count"),
            "total_seconds_mean": _mapping(summary.get("total_seconds")).get("mean"),
            "total_seconds_max": _mapping(summary.get("total_seconds")).get("max"),
            "phase_total_seconds_mean": _mapping(summary.get("phase_total_seconds")).get("mean"),
            "fingerprint_cache_weighted_hit_rate": fingerprint_cache.get("weighted_hit_rate"),
            "fingerprint_cache_last_hit_rate": fingerprint_cache.get("last_hit_rate"),
            "compact_json": config.compact_json,
            **dict(config.metadata),
        },
    )
    registry.save_json()


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output_path.write_text(text, encoding="utf-8")


def _numeric_summary(values: Sequence[Any] | Any) -> dict[str, Any]:
    raw_values = tuple(values)
    finite_values = [numeric for value in raw_values if (numeric := _finite_float(value)) is not None]
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


def _counts(values: Sequence[Any] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = "unknown" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _finite_int(value: Any) -> int | None:
    numeric = _finite_float(value)
    return None if numeric is None else int(numeric)


def _sum_int(items: Sequence[Mapping[str, Any]], field_name: str) -> int | None:
    values = [_finite_int(item.get(field_name)) for item in items]
    finite_values = [value for value in values if value is not None]
    return None if not finite_values else int(sum(finite_values))


def _safe_div(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _non_negative_float(value: Any, *, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return numeric


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


def _config_from_args(args: argparse.Namespace) -> ReleaseGateOverheadBaselineConfig:
    return ReleaseGateOverheadBaselineConfig(
        report_paths=tuple(args.report),
        output_path=Path(args.json),
        registry_path=None if args.registry is None else Path(args.registry),
        name=args.name,
        version=args.version,
        max_total_seconds=args.max_total_seconds,
        max_phase_total_seconds=args.max_phase_total_seconds,
        min_fingerprint_cache_hit_rate=args.min_fingerprint_cache_hit_rate,
        min_last_fingerprint_cache_hit_rate=args.min_last_fingerprint_cache_hit_rate,
        min_report_count=args.min_report_count,
        compact_json=bool(args.compact_json),
        metadata=_parse_metadata(args.metadata or ()),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = build_release_gate_overhead_baseline(_config_from_args(args))
    decision = payload["decision"]
    print(
        "release_gate_overhead="
        f"{decision['status']} "
        f"reports={payload['summary']['report_count']} "
        f"last_hit_rate={payload['summary']['artifact_fingerprint_cache']['last_hit_rate']}"
    )
    if args.fail_on_blocked and decision["status"] == "blocked":
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a release-gate overhead baseline report")
    parser.add_argument("--report", action="append", required=True, help="release workflow JSON report; repeatable")
    parser.add_argument("--json", required=True, help="output overhead baseline report")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--max-total-seconds", type=float, default=None)
    parser.add_argument("--max-phase-total-seconds", type=float, default=None)
    parser.add_argument("--min-fingerprint-cache-hit-rate", type=float, default=None)
    parser.add_argument("--min-last-fingerprint-cache-hit-rate", type=float, default=None)
    parser.add_argument("--min-report-count", type=int, default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
