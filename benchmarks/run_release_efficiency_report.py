"""Build a release efficiency report from product runtime sweep artifacts."""

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

from benchmarks.config_utils import planned_artifact_manifest_summary, strict_bool  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class ReleaseEfficiencyReportConfig:
    """Configuration for a release efficiency report."""

    profile_sweep_path: str | Path
    report_path: str | Path
    quality_report_paths: Sequence[str | Path] = ()
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    compact_json: bool = False

    def __post_init__(self) -> None:
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        object.__setattr__(self, "profile_sweep_path", Path(self.profile_sweep_path))
        object.__setattr__(self, "report_path", Path(self.report_path))
        object.__setattr__(
            self,
            "quality_report_paths",
            tuple(Path(path) for path in self.quality_report_paths),
        )
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
        return Path(self.report_path).with_name("release-efficiency-artifact-manifest.json")


def build_release_efficiency_report(config: ReleaseEfficiencyReportConfig) -> dict[str, Any]:
    """Build and write a release efficiency report."""
    profile_sweep = _load_json_object(config.profile_sweep_path)
    quality_reports = tuple(_quality_report(path) for path in config.quality_report_paths)
    profiles = tuple(
        _profile_efficiency_row(profile)
        for profile in _sequence(profile_sweep.get("profiles"))
        if isinstance(profile, Mapping)
    )
    leaderboard = _leaderboard(profiles)
    recommendation = leaderboard[0] if leaderboard else None
    quality_summary = _quality_summary(quality_reports)
    status = _status(
        profile_sweep=profile_sweep,
        quality_summary=quality_summary,
        recommendation=recommendation,
    )
    report = {
        "schema_version": 1,
        "workflow": "release_efficiency_report",
        "status": status,
        "decision": {
            "status": status,
            "recommended_profile": None if recommendation is None else recommendation.get("profile"),
            "recommended_efficiency_score": (
                None
                if recommendation is None
                else _nested(recommendation, "efficiency", "score")
            ),
            "blocking_reasons": _blocking_reasons(
                profile_sweep=profile_sweep,
                quality_summary=quality_summary,
                recommendation=recommendation,
            ),
        },
        "inputs": {
            "profile_sweep": {
                "path": str(config.profile_sweep_path),
                "workflow": profile_sweep.get("workflow"),
                "status": profile_sweep.get("status"),
                "recommended_profile": _nested(profile_sweep, "decision", "recommended_profile"),
            },
            "quality_reports": quality_reports,
        },
        "summary": _summary(
            profile_sweep=profile_sweep,
            profiles=profiles,
            quality_summary=quality_summary,
        ),
        "profiles": profiles,
        "leaderboard": leaderboard,
        "quality": quality_summary,
        "paths": {
            "report": str(config.report_path),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "profile_sweep": str(config.profile_sweep_path),
            "quality_reports": tuple(str(path) for path in config.quality_report_paths),
        },
        "config": {
            "compact_json": config.compact_json,
            "metadata": dict(config.metadata),
        },
    }
    _write_outputs(config, report)
    return report


def _profile_efficiency_row(profile: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _mapping(profile.get("metrics"))
    trace_sources = _mapping(profile.get("trace_sources"))
    trace_record_cache = _mapping(profile.get("trace_record_cache"))
    efficiency = _efficiency(metrics)
    return {
        "profile": profile.get("profile"),
        "status": profile.get("status"),
        "baseline_status": profile.get("baseline_status"),
        "trace_count": profile.get("trace_count"),
        "baseline_path": profile.get("baseline_path"),
        "metrics": {
            "total_seconds_mean": _finite_float(metrics.get("total_seconds_mean")),
            "total_seconds_p95": _finite_float(metrics.get("total_seconds_p95")),
            "measured_phases_mean": _finite_float(metrics.get("measured_phases_mean")),
            "mean_route_duration_seconds": _finite_float(metrics.get("mean_route_duration_seconds")),
            "mean_attempted_route_count": _finite_float(metrics.get("mean_attempted_route_count")),
            "route_budget_exhaustion_rate": _finite_float(metrics.get("route_budget_exhaustion_rate")),
            "retrieval_use_rate": _finite_float(metrics.get("retrieval_use_rate")),
            "cache_hit_rate_mean": _finite_float(metrics.get("cache_hit_rate_mean")),
            "verification_skip_rate_mean": _finite_float(metrics.get("verification_skip_rate_mean")),
            "verification_selective_claim_skip_rate": _finite_float(
                metrics.get("verification_selective_claim_skip_rate")
            ),
            "verified_claim_count_mean": _finite_float(metrics.get("verified_claim_count_mean")),
            "verifier_saved_claim_count_mean": _finite_float(
                metrics.get("verifier_saved_claim_count_mean")
            ),
            "max_verifier_route_attempts_mean": _finite_float(
                metrics.get("max_verifier_route_attempts_mean")
            ),
        },
        "efficiency": efficiency,
        "runtime_profile_selection": _mapping(profile.get("runtime_profile_selection")),
        "trace_sources": {
            "generated_count": _finite_int(trace_sources.get("generated_count")),
            "reused_count": _finite_int(trace_sources.get("reused_count")),
            "counts": _mapping(trace_sources.get("counts")),
        },
        "trace_record_cache": {
            "enabled": trace_record_cache.get("enabled"),
            "cache_hit": trace_record_cache.get("cache_hit"),
            "cache_written": trace_record_cache.get("cache_written"),
            "path": trace_record_cache.get("path"),
            "invalidation_reason": trace_record_cache.get("invalidation_reason"),
        },
    }


def _efficiency(metrics: Mapping[str, Any]) -> dict[str, Any]:
    total_seconds = _finite_float(metrics.get("total_seconds_mean")) or 0.0
    route_attempts = _finite_float(metrics.get("mean_attempted_route_count")) or 0.0
    verified_claims = _finite_float(metrics.get("verified_claim_count_mean")) or 0.0
    retrieval_use = _finite_float(metrics.get("retrieval_use_rate")) or 0.0
    route_exhaustion = _finite_float(metrics.get("route_budget_exhaustion_rate")) or 0.0
    verification_skip = _finite_float(metrics.get("verification_skip_rate_mean")) or 0.0
    selective_skip = _finite_float(metrics.get("verification_selective_claim_skip_rate")) or 0.0
    cache_hit = _finite_float(metrics.get("cache_hit_rate_mean")) or 0.0
    verifier_saved = _finite_float(metrics.get("verifier_saved_claim_count_mean")) or 0.0
    cost = (
        total_seconds
        + (0.05 * route_attempts)
        + (0.05 * verified_claims)
        + (0.10 * retrieval_use)
        + (0.20 * route_exhaustion)
    )
    savings = 1.0 + verification_skip + selective_skip + cache_hit + (0.10 * verifier_saved)
    score = savings / (1.0 + cost)
    return {
        "score": score,
        "cost_index": cost,
        "savings_index": savings,
        "formula": (
            "(1 + verification_skip + selective_skip + cache_hit + 0.10*verifier_saved) "
            "/ (1 + total_seconds + 0.05*route_attempts + 0.05*verified_claims "
            "+ 0.10*retrieval_use + 0.20*route_budget_exhaustion)"
        ),
    }


def _leaderboard(profiles: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    rows = []
    for profile in profiles:
        efficiency = _mapping(profile.get("efficiency"))
        metrics = _mapping(profile.get("metrics"))
        trace_record_cache = _mapping(profile.get("trace_record_cache"))
        rows.append({
            "profile": profile.get("profile"),
            "status": profile.get("status"),
            "efficiency": dict(efficiency),
            "total_seconds_mean": metrics.get("total_seconds_mean"),
            "mean_attempted_route_count": metrics.get("mean_attempted_route_count"),
            "verification_skip_rate_mean": metrics.get("verification_skip_rate_mean"),
            "verification_selective_claim_skip_rate": metrics.get(
                "verification_selective_claim_skip_rate"
            ),
            "trace_record_cache_hit": trace_record_cache.get("cache_hit"),
            "blocked": profile.get("status") == "blocked",
        })
    return tuple(sorted(
        rows,
        key=lambda row: (
            bool(row["blocked"]),
            -(_finite_float(_nested(row, "efficiency", "score")) or 0.0),
            _sort_float(row.get("total_seconds_mean")),
            str(row.get("profile")),
        ),
    ))


def _quality_report(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    return {
        "path": str(path),
        "workflow": payload.get("workflow"),
        "status": payload.get("status"),
        "decision_status": _nested(payload, "decision", "status"),
    }


def _quality_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = tuple(str(report.get("status")) for report in reports if report.get("status") is not None)
    blocked = tuple(report for report in reports if report.get("status") == "blocked")
    promoted = tuple(report for report in reports if report.get("status") == "promote")
    return {
        "report_count": len(reports),
        "status_counts": _counts(statuses),
        "blocked_count": len(blocked),
        "promoted_count": len(promoted),
        "passed": not blocked,
    }


def _summary(
    *,
    profile_sweep: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
    quality_summary: Mapping[str, Any],
) -> dict[str, Any]:
    trace_sources = tuple(_mapping(profile.get("trace_sources")) for profile in profiles)
    trace_record_caches = tuple(_mapping(profile.get("trace_record_cache")) for profile in profiles)
    return {
        "profile_sweep_status": profile_sweep.get("status"),
        "profile_count": len(profiles),
        "blocked_profile_count": sum(1 for profile in profiles if profile.get("status") == "blocked"),
        "quality_report_count": quality_summary.get("report_count"),
        "quality_passed": quality_summary.get("passed"),
        "generated_trace_count": sum((_finite_int(item.get("generated_count")) or 0) for item in trace_sources),
        "reused_trace_count": sum((_finite_int(item.get("reused_count")) or 0) for item in trace_sources),
        "trace_record_cache_enabled_profile_count": sum(
            1 for item in trace_record_caches if item.get("enabled") is True
        ),
        "trace_record_cache_hit_profile_count": sum(
            1 for item in trace_record_caches if item.get("cache_hit") is True
        ),
        "trace_record_cache_written_profile_count": sum(
            1 for item in trace_record_caches if item.get("cache_written") is True
        ),
    }


def _status(
    *,
    profile_sweep: Mapping[str, Any],
    quality_summary: Mapping[str, Any],
    recommendation: Mapping[str, Any] | None,
) -> str:
    if profile_sweep.get("status") == "blocked":
        return "blocked"
    if quality_summary.get("blocked_count"):
        return "blocked"
    if recommendation is None:
        return "blocked"
    if profile_sweep.get("status") == "promote" and quality_summary.get("passed") is True:
        return "promote" if quality_summary.get("report_count") else "observed"
    return "observed"


def _blocking_reasons(
    *,
    profile_sweep: Mapping[str, Any],
    quality_summary: Mapping[str, Any],
    recommendation: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    reasons = []
    if profile_sweep.get("status") == "blocked":
        reasons.append("profile sweep is blocked")
    if quality_summary.get("blocked_count"):
        reasons.append("one or more quality reports are blocked")
    if recommendation is None:
        reasons.append("no eligible profile recommendation")
    return tuple(reasons)


def _write_outputs(config: ReleaseEfficiencyReportConfig, report: dict[str, Any]) -> None:
    artifacts = _artifact_paths(config)
    report["artifact_manifest_summary"] = planned_artifact_manifest_summary(
        artifacts,
        assume_file_paths=(config.report_path,),
    )
    _write_json(config.report_path, report, compact=config.compact_json)
    manifest = _write_artifact_manifest(config, report, artifacts=artifacts)
    report["artifact_manifest_summary"] = manifest["summary"]
    _write_json(config.report_path, report, compact=config.compact_json)
    _record_registry(config, report)


def _write_artifact_manifest(
    config: ReleaseEfficiencyReportConfig,
    report: Mapping[str, Any],
    *,
    artifacts: Mapping[str, str | Path | None],
) -> dict[str, Any]:
    manifest = build_artifact_manifest(
        artifacts,
        root=config.resolved_artifact_manifest_path.parent,
        metadata={
            "runner": "run_release_efficiency_report",
            "status": report.get("status"),
            "recommended_profile": _nested(report, "decision", "recommended_profile"),
            "recommended_efficiency_score": _nested(
                report,
                "decision",
                "recommended_efficiency_score",
            ),
            "profile_sweep": str(config.profile_sweep_path),
            "quality_report_count": len(config.quality_report_paths),
            "compact_json": config.compact_json,
            **dict(config.metadata),
        },
    )
    _write_json(config.resolved_artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _artifact_paths(config: ReleaseEfficiencyReportConfig) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {
        "release_efficiency_report": config.report_path,
        "profile_sweep": config.profile_sweep_path,
    }
    for index, path in enumerate(config.quality_report_paths):
        artifacts[f"quality_report_{index:04d}"] = path
    return artifacts


def _record_registry(config: ReleaseEfficiencyReportConfig, report: Mapping[str, Any]) -> None:
    if config.registry_path is None:
        return
    ArtifactRegistry.load_json(config.registry_path).record_report(
        name=str(config.name),
        path=config.report_path,
        version=str(config.version),
        metadata={
            "workflow": "release_efficiency_report",
            "status": report.get("status"),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "recommended_profile": _nested(report, "decision", "recommended_profile"),
            "recommended_efficiency_score": _nested(
                report,
                "decision",
                "recommended_efficiency_score",
            ),
            "profile_count": _nested(report, "summary", "profile_count"),
            "quality_report_count": _nested(report, "summary", "quality_report_count"),
            "trace_record_cache_hit_profile_count": _nested(
                report,
                "summary",
                "trace_record_cache_hit_profile_count",
            ),
            "compact_json": config.compact_json,
            **dict(config.metadata),
        },
    ).save_json()


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must contain an object: {path}")
    return dict(payload)


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _counts(values: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


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
    if numeric is None or not numeric.is_integer():
        return None
    return int(numeric)


def _sort_float(value: Any) -> float:
    numeric = _finite_float(value)
    return float("inf") if numeric is None else numeric


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json_text(payload, compact=compact), encoding="utf-8")


def _json_text(payload: Any, *, compact: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


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


def _config_from_args(args: argparse.Namespace) -> ReleaseEfficiencyReportConfig:
    return ReleaseEfficiencyReportConfig(
        profile_sweep_path=Path(args.profile_sweep),
        report_path=Path(args.json),
        quality_report_paths=tuple(Path(path) for path in args.quality_report or ()),
        artifact_manifest_path=Path(args.artifact_manifest) if args.artifact_manifest else None,
        registry_path=Path(args.registry) if args.registry else None,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = build_release_efficiency_report(_config_from_args(args))
    print(_json_text(report, compact=bool(args.compact_json)), end="")
    if args.fail_on_blocked and report["status"] == "blocked":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a release efficiency report")
    parser.add_argument("--profile-sweep", required=True, help="product runtime profile sweep report JSON")
    parser.add_argument("--json", required=True, help="output release efficiency report JSON path")
    parser.add_argument("--quality-report", action="append", default=[],
                        help="optional release/readiness/performance quality report JSON; repeatable")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest output path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry release efficiency report name")
    parser.add_argument("--version", default=None, help="registry release efficiency report version")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
