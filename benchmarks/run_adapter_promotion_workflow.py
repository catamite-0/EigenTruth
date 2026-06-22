"""Run a fail-closed adapter promotion workflow from saved benchmark reports."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.compare_registry_baseline import compare_registry_baseline  # noqa: E402
from benchmarks.compare_verifier_routes import build_route_comparison_report  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class AdapterPromotionWorkflowConfig:
    """Configuration for a route-promotion plus optional baseline-gate workflow."""

    reports: Sequence[tuple[str, Path]]
    route_report_path: Path
    alpha: float = 0.10
    min_selected: int = 1
    notes: Sequence[str] = ()
    gate_routes: Sequence[str] = ()
    gate_min_selected: int | None = None
    min_decision_accuracy: float | None = None
    max_false_supported_rate: float | None = None
    min_false_refuted_rate: float | None = None
    max_verified_false_alarm: float | None = None
    min_verified_detection: float | None = None
    max_mean_duration_seconds: float | None = None
    max_p95_duration_seconds: float | None = None
    max_p99_duration_seconds: float | None = None
    max_max_duration_seconds: float | None = None
    max_mean_attempted_route_count: float | None = None
    max_retrieval_use_rate: float | None = None
    min_cache_hit_rate: float | None = None
    registry_path: Path | None = None
    baseline_key: str | None = None
    baseline_name: str | None = None
    baseline_version: str | None = None
    baseline_profile_artifact: str = "profiles.uncached"
    candidate_profiles: Sequence[tuple[str, Path]] = ()
    allow_unverified_compare: bool = False
    max_total_ratio: float | None = None
    max_run_total_ratios: Mapping[str, float] | None = None
    max_phase_ratios: Mapping[str, float] | None = None
    min_throughput_ratios: Mapping[str, float] | None = None
    compact_json: bool = False
    artifact_manifest_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reports", tuple((str(name), Path(path)) for name, path in self.reports))
        object.__setattr__(self, "route_report_path", Path(self.route_report_path))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))
        object.__setattr__(self, "gate_routes", tuple(str(route) for route in self.gate_routes))
        object.__setattr__(
            self,
            "candidate_profiles",
            tuple((str(name), Path(path)) for name, path in self.candidate_profiles),
        )


def run_adapter_promotion_workflow(config: AdapterPromotionWorkflowConfig) -> dict[str, Any]:
    """Run route promotion checks and optional registry-backed profile comparison."""
    route_comparison = build_route_comparison_report(
        config.reports,
        alpha=config.alpha,
        min_selected=config.min_selected,
        notes=config.notes,
        gate_routes=config.gate_routes,
        gate_min_selected=config.gate_min_selected,
        min_decision_accuracy=config.min_decision_accuracy,
        max_false_supported_rate=config.max_false_supported_rate,
        min_false_refuted_rate=config.min_false_refuted_rate,
        max_verified_false_alarm=config.max_verified_false_alarm,
        min_verified_detection=config.min_verified_detection,
        max_mean_duration_seconds=config.max_mean_duration_seconds,
        max_p95_duration_seconds=config.max_p95_duration_seconds,
        max_p99_duration_seconds=config.max_p99_duration_seconds,
        max_max_duration_seconds=config.max_max_duration_seconds,
        max_mean_attempted_route_count=config.max_mean_attempted_route_count,
        max_retrieval_use_rate=config.max_retrieval_use_rate,
        min_cache_hit_rate=config.min_cache_hit_rate,
    )
    config.route_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.route_report_path.write_text(
        _json_text(route_comparison, compact=config.compact_json, sort_keys=True),
        encoding="utf-8",
    )

    registry_comparison = None
    if config.candidate_profiles:
        if config.registry_path is None:
            raise ValueError("registry_path is required when candidate_profiles are provided.")
        registry_comparison = compare_registry_baseline(
            registry_path=config.registry_path,
            baseline_key=config.baseline_key,
            baseline_name=config.baseline_name,
            baseline_version=config.baseline_version,
            baseline_profile_artifact=config.baseline_profile_artifact,
            candidate_profiles=config.candidate_profiles,
            recursive=True,
            allow_unverified=config.allow_unverified_compare,
            max_total_ratio=config.max_total_ratio,
            max_run_total_ratios=config.max_run_total_ratios,
            max_phase_ratios=config.max_phase_ratios,
            min_throughput_ratios=config.min_throughput_ratios,
            notes=("adapter promotion workflow registry comparison",),
        )

    decision = _workflow_decision(route_comparison, registry_comparison)
    payload = {
        "schema_version": 1,
        "workflow": "adapter_promotion_workflow",
        "route_comparison_path": str(config.route_report_path),
        "artifact_manifest": None if config.artifact_manifest_path is None else str(config.artifact_manifest_path),
        "route_comparison": route_comparison,
        "registry_baseline_comparison": registry_comparison,
        "decision": decision,
    }
    if config.artifact_manifest_path is not None:
        _write_artifact_manifest(config, payload)
    return payload


def _write_artifact_manifest(
    config: AdapterPromotionWorkflowConfig,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "route_comparison_report": config.route_report_path,
    }
    for name, path in config.reports:
        artifacts[f"verifier_reports.{name}"] = path
    for name, path in config.candidate_profiles:
        artifacts[f"candidate_profiles.{name}"] = path

    route_comparison = dict(payload.get("route_comparison") or {})
    decision = dict(payload.get("decision") or {})
    route_decision = dict(route_comparison.get("promotion_decision") or {})
    recommended_route = route_decision.get("recommended_route") or decision.get("recommended_route")
    by_route = dict(route_comparison.get("by_route") or {})
    recommended_metrics = (
        dict(by_route.get(str(recommended_route)) or {})
        if recommended_route is not None
        else {}
    )
    quality_gate = dict(route_comparison.get("quality_gate") or {})
    manifest = build_artifact_manifest(
        artifacts,
        root=config.artifact_manifest_path.parent,
        metadata={
            "runner": "run_adapter_promotion_workflow",
            "workflow": payload.get("workflow"),
            "promotion_status": decision.get("status"),
            "route_promotion_status": route_decision.get("status"),
            "recommended_route": recommended_route,
            "route_promotion_passed": decision.get("route_promotion_passed"),
            "registry_baseline_checked": decision.get("registry_baseline_checked"),
            "registry_baseline_passed": decision.get("registry_baseline_passed"),
            "quality_gate_passed": quality_gate.get("passed"),
            "quality_gate_checked_routes": quality_gate.get("checked_routes"),
            "recommended_selected": recommended_metrics.get("selected"),
            "recommended_decision_accuracy": recommended_metrics.get("decision_accuracy"),
            "recommended_false_supported_rate": recommended_metrics.get("false_supported_rate"),
            "recommended_false_refuted_rate": recommended_metrics.get("false_refuted_rate"),
            "recommended_verified_false_alarm": recommended_metrics.get("verified_false_alarm"),
            "recommended_verified_detection": recommended_metrics.get("verified_detection"),
            "recommended_mean_duration_seconds": recommended_metrics.get("mean_duration_seconds"),
            "recommended_p95_duration_seconds": recommended_metrics.get("p95_duration_seconds"),
            "recommended_p99_duration_seconds": recommended_metrics.get("p99_duration_seconds"),
            "recommended_max_duration_seconds": recommended_metrics.get("max_duration_seconds"),
            "recommended_mean_attempted_route_count": recommended_metrics.get("mean_attempted_route_count"),
            "recommended_retrieval_use_rate": recommended_metrics.get("retrieval_use_rate"),
            "recommended_invalid_metric_counts": recommended_metrics.get("invalid_metric_counts"),
        },
    )
    config.artifact_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.artifact_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _json_text(payload: Mapping[str, Any], *, compact: bool, sort_keys: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=sort_keys, separators=(",", ":")) + "\n"
    return json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n"


def _workflow_decision(
    route_comparison: Mapping[str, Any],
    registry_comparison: Mapping[str, Any] | None,
) -> dict[str, Any]:
    blocking_reasons = []
    route_decision = route_comparison.get("promotion_decision", {})
    route_status = route_decision.get("status") if isinstance(route_decision, Mapping) else None
    if route_status != "promote":
        blocking_reasons.append({
            "gate": "route_promotion",
            "status": route_status,
            "reason": None if not isinstance(route_decision, Mapping) else route_decision.get("reason"),
        })

    registry_gate_passed = None
    if registry_comparison is not None:
        comparison = registry_comparison.get("comparison", {})
        gate = comparison.get("regression_gate", {}) if isinstance(comparison, Mapping) else {}
        if not isinstance(gate, Mapping) or "passed" not in gate:
            registry_gate_passed = None
            blocking_reasons.append({
                "gate": "registry_baseline",
                "status": "needs_gate",
                "reason": "configure a registry-backed performance regression gate before promoting an adapter",
                "failures": [],
            })
        else:
            registry_gate_passed = bool(gate.get("passed"))
        if registry_gate_passed is False:
            blocking_reasons.append({
                "gate": "registry_baseline",
                "status": "failed",
                "reason": "registry-backed performance regression gate failed",
                "failures": list(gate.get("failures", ())) if isinstance(gate, Mapping) else [],
            })

    return {
        "status": "promote" if not blocking_reasons else "blocked",
        "route_promotion_passed": route_status == "promote",
        "recommended_route": route_decision.get("recommended_route") if isinstance(route_decision, Mapping) else None,
        "registry_baseline_checked": registry_comparison is not None,
        "registry_baseline_passed": registry_gate_passed,
        "blocking_reasons": blocking_reasons,
    }


def _parse_named_path(value: str, *, default_name_from_stem: bool = True) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        if default_name_from_stem:
            return path.stem, path
        raise ValueError(f"named path must be formatted as name=path: {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("named path name cannot be empty.")
    return name, Path(path)


def _parse_named_float(value: str, *, flag: str) -> tuple[str, float]:
    if "=" not in value:
        raise ValueError(f"{flag} must be formatted as name=value.")
    name, raw_value = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"{flag} name cannot be empty.")
    threshold = float(raw_value)
    if threshold < 0:
        raise ValueError(f"{flag} value for {name!r} must be non-negative.")
    return name, threshold


def _config_from_args(args: argparse.Namespace) -> AdapterPromotionWorkflowConfig:
    return AdapterPromotionWorkflowConfig(
        reports=tuple(_parse_named_path(value) for value in args.report),
        route_report_path=Path(args.route_report_json),
        alpha=args.alpha,
        min_selected=args.min_selected,
        notes=args.note,
        gate_routes=tuple(args.gate_route),
        gate_min_selected=args.gate_min_selected,
        min_decision_accuracy=args.min_decision_accuracy,
        max_false_supported_rate=args.max_false_supported_rate,
        min_false_refuted_rate=args.min_false_refuted_rate,
        max_verified_false_alarm=args.max_verified_false_alarm,
        min_verified_detection=args.min_verified_detection,
        max_mean_duration_seconds=args.max_mean_duration_seconds,
        max_p95_duration_seconds=args.max_p95_duration_seconds,
        max_p99_duration_seconds=args.max_p99_duration_seconds,
        max_max_duration_seconds=args.max_max_duration_seconds,
        max_mean_attempted_route_count=args.max_mean_attempted_route_count,
        max_retrieval_use_rate=args.max_retrieval_use_rate,
        min_cache_hit_rate=args.min_cache_hit_rate,
        registry_path=None if args.registry is None else Path(args.registry),
        baseline_key=args.baseline_key,
        baseline_name=args.baseline_name,
        baseline_version=args.baseline_version,
        baseline_profile_artifact=args.baseline_profile_artifact,
        candidate_profiles=tuple(_parse_named_path(value) for value in args.candidate_profile),
        allow_unverified_compare=bool(args.allow_unverified_compare),
        max_total_ratio=args.max_total_ratio,
        max_run_total_ratios=dict(
            _parse_named_float(value, flag="--max-run-total-ratio")
            for value in args.max_run_total_ratio
        ),
        max_phase_ratios=dict(
            _parse_named_float(value, flag="--max-phase-ratio")
            for value in args.max_phase_ratio
        ),
        min_throughput_ratios=dict(
            _parse_named_float(value, flag="--min-throughput-ratio")
            for value in args.min_throughput_ratio
        ),
        compact_json=bool(args.compact_json),
        artifact_manifest_path=None if args.artifact_manifest is None else Path(args.artifact_manifest),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = run_adapter_promotion_workflow(_config_from_args(args))
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _json_text(payload, compact=bool(args.compact_json), sort_keys=True),
            encoding="utf-8",
        )
        print(f"Wrote adapter promotion workflow report to {output_path}")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run adapter route promotion and optional baseline gates")
    parser.add_argument("--report", action="append", required=True,
                        help="verifier route report path, optionally named as name=path; repeatable")
    parser.add_argument("--route-report-json", required=True,
                        help="path to write the generated route comparison JSON")
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="alpha key to use for route_control_impact")
    parser.add_argument("--min-selected", type=int, default=1,
                        help="minimum selected records required for route ranking")
    parser.add_argument("--note", action="append", default=[],
                        help="optional note to include in the route comparison; repeatable")
    parser.add_argument("--gate-route", action="append", default=[],
                        help="aggregate route to gate; repeatable. Defaults to all eligible routes")
    parser.add_argument("--gate-min-selected", type=int, default=None,
                        help="minimum selected records for gated routes; defaults to --min-selected")
    parser.add_argument("--min-decision-accuracy", type=float, default=None)
    parser.add_argument("--max-false-supported-rate", type=float, default=None)
    parser.add_argument("--min-false-refuted-rate", type=float, default=None)
    parser.add_argument("--max-verified-false-alarm", type=float, default=None)
    parser.add_argument("--min-verified-detection", type=float, default=None)
    parser.add_argument("--max-mean-duration-seconds", type=float, default=None)
    parser.add_argument("--max-p95-duration-seconds", type=float, default=None)
    parser.add_argument("--max-p99-duration-seconds", type=float, default=None)
    parser.add_argument("--max-max-duration-seconds", type=float, default=None)
    parser.add_argument("--max-mean-attempted-route-count", type=float, default=None)
    parser.add_argument("--max-retrieval-use-rate", type=float, default=None)
    parser.add_argument("--min-cache-hit-rate", type=float, default=None)
    parser.add_argument("--registry", default=None,
                        help="optional local ArtifactRegistry JSON path for baseline comparison")
    parser.add_argument("--baseline-key", default=None, help="benchmark_manifest registry key")
    parser.add_argument("--baseline-name", default=None, help="benchmark manifest record name")
    parser.add_argument("--baseline-version", default=None, help="benchmark manifest record version")
    parser.add_argument("--baseline-profile-artifact", default="profiles.uncached",
                        help="profile artifact name inside the baseline manifest")
    parser.add_argument("--candidate-profile", action="append", default=[],
                        help="candidate profile JSON path, optionally named as name=path; repeatable")
    parser.add_argument("--allow-unverified-compare", action="store_true",
                        help="compare even when baseline manifest verification fails")
    parser.add_argument("--max-total-ratio", type=float, default=None,
                        help="fail registry gate when any candidate exceeds this total-time ratio")
    parser.add_argument("--max-run-total-ratio", action="append", default=[],
                        help="fail registry gate when one named run exceeds this ratio, formatted as run=ratio")
    parser.add_argument("--max-phase-ratio", action="append", default=[],
                        help="fail registry gate when a phase exceeds this ratio, formatted as phase=ratio")
    parser.add_argument("--min-throughput-ratio", action="append", default=[],
                        help="fail registry gate when throughput drops below this ratio, formatted as metric=ratio")
    parser.add_argument("--json", default=None, help="optional path to write the final workflow report")
    parser.add_argument("--artifact-manifest", default=None,
                        help="optional path to write a registry-ready artifact manifest")
    parser.add_argument("--compact-json", action="store_true",
                        help="write minified JSON artifacts for lower artifact size and write latency")
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless the workflow decision status is promote")
    args = parser.parse_args(argv)
    payload = run(args)
    decision = payload["decision"]
    print(
        f"adapter_promotion={decision['status']} "
        f"route={decision.get('recommended_route')}"
    )
    if args.fail_on_blocked and decision["status"] != "promote":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
