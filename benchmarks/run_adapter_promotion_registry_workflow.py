"""Run adapter promotion, verify its manifest, and register the promoted route."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.promote_artifact_manifest import promote_artifact_manifest  # noqa: E402
from benchmarks.run_adapter_promotion_workflow import (  # noqa: E402
    AdapterPromotionWorkflowConfig,
    _parse_named_float,
    _parse_named_path,
    run_adapter_promotion_workflow,
)


@dataclass(frozen=True)
class AdapterPromotionRegistryWorkflowConfig:
    """Configuration for registering a promoted adapter route manifest."""

    promotion: AdapterPromotionWorkflowConfig
    registry_path: Path
    name: str
    version: str
    workflow_report_path: Path | None = None
    verification_report_path: Path | None = None
    promotion_metadata: Mapping[str, Any] | None = None
    allow_non_promote: bool = False
    allow_promotion_failures: bool = False

    def __post_init__(self) -> None:
        promotion = self.promotion
        if promotion.artifact_manifest_path is None:
            promotion = replace(
                promotion,
                artifact_manifest_path=promotion.route_report_path.with_name("artifact-manifest.json"),
            )
            object.__setattr__(self, "promotion", promotion)
        object.__setattr__(self, "registry_path", Path(self.registry_path))
        if self.workflow_report_path is not None:
            object.__setattr__(self, "workflow_report_path", Path(self.workflow_report_path))
        if self.verification_report_path is not None:
            object.__setattr__(self, "verification_report_path", Path(self.verification_report_path))

    @property
    def report_path(self) -> Path:
        return self.workflow_report_path or self.promotion.route_report_path.with_name(
            "adapter-promotion-registry-workflow.json"
        )

    @property
    def verification_path(self) -> Path:
        return self.verification_report_path or self.promotion.route_report_path.with_name(
            "manifest-verification.json"
        )


def run_adapter_promotion_registry_workflow(
    config: AdapterPromotionRegistryWorkflowConfig,
) -> dict[str, Any]:
    """Run route promotion and register its verified manifest when eligible."""
    promotion_report = run_adapter_promotion_workflow(config.promotion)
    adapter_decision = dict(promotion_report.get("decision") or {})
    adapter_status = str(adapter_decision.get("status"))
    manifest_promotion = None
    blocking_reasons = []
    if adapter_status != "promote":
        blocking_reasons.append("adapter promotion decision did not promote")

    if adapter_status == "promote" or config.allow_non_promote:
        manifest_path = promotion_report.get("artifact_manifest")
        if not manifest_path:
            raise ValueError("adapter promotion workflow did not write an artifact manifest")
        manifest_promotion = promote_artifact_manifest(
            manifest_path=manifest_path,
            registry_path=config.registry_path,
            name=config.name,
            version=config.version,
            verification_report_path=config.verification_path,
            recursive=True,
            allow_failures=config.allow_promotion_failures,
            metadata=_promotion_metadata(config, promotion_report),
        )
        if not dict(manifest_promotion.get("verification") or {}).get("passed", False):
            blocking_reasons.append("adapter promotion manifest verification did not pass")

    decision = _registry_workflow_decision(
        adapter_promotion_status=adapter_status,
        promotion=manifest_promotion,
        blocking_reasons=blocking_reasons,
    )
    payload = {
        "schema_version": 1,
        "workflow": "adapter_promotion_registry_workflow",
        "config": {
            "route_report_json": str(config.promotion.route_report_path),
            "artifact_manifest": str(config.promotion.artifact_manifest_path),
            "registry": str(config.registry_path),
            "name": config.name,
            "version": config.version,
            "allow_non_promote": config.allow_non_promote,
            "allow_promotion_failures": config.allow_promotion_failures,
        },
        "adapter_promotion": promotion_report,
        "promotion": manifest_promotion,
        "decision": decision,
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _registry_workflow_decision(
    *,
    adapter_promotion_status: str,
    promotion: Mapping[str, Any] | None,
    blocking_reasons: Sequence[str],
) -> dict[str, Any]:
    verification = {} if promotion is None else dict(promotion.get("verification") or {})
    verified = bool(verification.get("passed", False))
    status = "promote" if adapter_promotion_status == "promote" and verified else "blocked"
    return {
        "status": status,
        "adapter_promotion_status": adapter_promotion_status,
        "manifest_promoted": promotion is not None,
        "manifest_verified": verified,
        "registry_record": None if promotion is None else dict(promotion.get("records") or {}).get(
            "benchmark_manifest"
        ),
        "blocking_reasons": tuple(blocking_reasons),
    }


def _promotion_metadata(
    config: AdapterPromotionRegistryWorkflowConfig,
    promotion_report: Mapping[str, Any],
) -> dict[str, Any]:
    decision = dict(promotion_report.get("decision") or {})
    route_comparison = dict(promotion_report.get("route_comparison") or {})
    route_decision = dict(route_comparison.get("promotion_decision") or {})
    recommended_route = route_decision.get("recommended_route") or decision.get("recommended_route")
    by_route = dict(route_comparison.get("by_route") or {})
    recommended_metrics = (
        dict(by_route.get(str(recommended_route)) or {})
        if recommended_route is not None
        else {}
    )
    quality_gate = dict(route_comparison.get("quality_gate") or {})
    staged = dict(route_comparison.get("staged_verification") or {})
    metadata = {
        "workflow": "run_adapter_promotion_registry_workflow",
        "source_workflow": promotion_report.get("workflow"),
        "adapter_promotion_status": decision.get("status"),
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
        "staged_verification_enabled": staged.get("enabled"),
        "staged_skip_rate": staged.get("skip_rate"),
        "staged_verified_false_alarm": staged.get("verified_false_alarm"),
        "staged_verified_detection": staged.get("verified_detection"),
        "staged_delta_false_alarm": staged.get("delta_false_alarm"),
        "staged_delta_detection": staged.get("delta_detection"),
        "staged_verified_records": staged.get("verified_records"),
        "staged_skipped_records": staged.get("skipped_records"),
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


def _config_from_args(args: argparse.Namespace) -> AdapterPromotionRegistryWorkflowConfig:
    route_report_path = Path(args.route_report_json)
    artifact_manifest_path = (
        route_report_path.with_name("artifact-manifest.json")
        if args.artifact_manifest is None
        else Path(args.artifact_manifest)
    )
    promotion = AdapterPromotionWorkflowConfig(
        reports=tuple(_parse_named_path(value) for value in args.report),
        route_report_path=route_report_path,
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
        min_staged_skip_rate=args.min_staged_skip_rate,
        max_staged_verified_false_alarm=args.max_staged_verified_false_alarm,
        min_staged_verified_detection=args.min_staged_verified_detection,
        max_staged_delta_false_alarm=args.max_staged_delta_false_alarm,
        min_staged_delta_detection=args.min_staged_delta_detection,
        registry_path=None if args.baseline_registry is None else Path(args.baseline_registry),
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
        artifact_manifest_path=artifact_manifest_path,
    )
    return AdapterPromotionRegistryWorkflowConfig(
        promotion=promotion,
        registry_path=Path(args.registry),
        name=args.name,
        version=args.version,
        workflow_report_path=Path(args.json) if args.json else None,
        verification_report_path=Path(args.verification_report) if args.verification_report else None,
        promotion_metadata=_parse_metadata(args.metadata or ()),
        allow_non_promote=bool(args.allow_non_promote),
        allow_promotion_failures=bool(args.allow_promotion_failures),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = run_adapter_promotion_registry_workflow(_config_from_args(args))
    decision = payload["decision"]
    print(
        "adapter_promotion_registry="
        f"{decision['status']} "
        f"adapter_promotion={decision.get('adapter_promotion_status')} "
        f"record={decision.get('registry_record')}"
    )
    if args.fail_on_blocked and decision["status"] != "promote":
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run route promotion and register the verified manifest")
    parser.add_argument("--report", action="append", required=True,
                        help="verifier route report path, optionally named as name=path; repeatable")
    parser.add_argument("--route-report-json", required=True,
                        help="path to write the generated route comparison JSON")
    parser.add_argument("--registry", required=True, help="local ArtifactRegistry JSON path for route registration")
    parser.add_argument("--name", required=True, help="registry artifact name")
    parser.add_argument("--version", required=True, help="registry artifact version")
    parser.add_argument("--json", default=None, help="optional registry workflow report path")
    parser.add_argument("--artifact-manifest", default=None,
                        help="optional path to write the registry-ready artifact manifest")
    parser.add_argument("--verification-report", default=None, help="path for the manifest verification report")
    parser.add_argument("--metadata", action="append", default=[], help="extra promotion metadata as key=value")
    parser.add_argument("--allow-non-promote", action="store_true",
                        help="register even when adapter promotion decision is not promote")
    parser.add_argument("--allow-promotion-failures", action="store_true",
                        help="register even when manifest verification fails")
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
    parser.add_argument("--min-staged-skip-rate", type=float, default=None)
    parser.add_argument("--max-staged-verified-false-alarm", type=float, default=None)
    parser.add_argument("--min-staged-verified-detection", type=float, default=None)
    parser.add_argument("--max-staged-delta-false-alarm", type=float, default=None)
    parser.add_argument("--min-staged-delta-detection", type=float, default=None)
    parser.add_argument("--baseline-registry", default=None,
                        help="optional ArtifactRegistry path used only for performance baseline comparison")
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
    parser.add_argument("--compact-json", action="store_true",
                        help="write minified route-comparison and artifact-manifest JSON")
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless registry workflow decision is promote")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
