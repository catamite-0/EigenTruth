"""Run release-candidate comparison and register its verified manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.compare_release_candidates import compare_release_candidates  # noqa: E402
from benchmarks.promote_artifact_manifest import promote_artifact_manifest  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class ReleaseCandidateRegistryWorkflowConfig:
    """Configuration for registering a promoted release-candidate manifest."""

    readiness_registry_path: Path
    release_registry_path: Path
    name: str
    version: str
    route_registry_path: Path | None = None
    readiness_baseline_keys: Sequence[str] = ()
    route_baseline_keys: Sequence[str] = ()
    release_report_path: Path | None = None
    artifact_manifest_path: Path | None = None
    verification_report_path: Path | None = None
    workflow_report_path: Path | None = None
    recursive: bool = True
    allow_unverified: bool = False
    min_best_quality_auroc: float | None = None
    max_uncached_forward_seconds: float | None = None
    max_cache_only_seconds: float | None = None
    min_selected: int | None = None
    min_decision_accuracy: float | None = None
    max_false_supported_rate: float | None = None
    min_false_refuted_rate: float | None = None
    max_verified_false_alarm: float | None = None
    min_verified_detection: float | None = None
    max_mean_duration_seconds: float | None = None
    max_p99_duration_seconds: float | None = None
    max_max_duration_seconds: float | None = None
    max_mean_attempted_route_count: float | None = None
    max_retrieval_use_rate: float | None = None
    promotion_metadata: Mapping[str, Any] | None = None
    allow_non_promote: bool = False
    allow_promotion_failures: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "readiness_registry_path", Path(self.readiness_registry_path))
        object.__setattr__(self, "release_registry_path", Path(self.release_registry_path))
        if self.route_registry_path is not None:
            object.__setattr__(self, "route_registry_path", Path(self.route_registry_path))
        if self.release_report_path is not None:
            object.__setattr__(self, "release_report_path", Path(self.release_report_path))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        if self.verification_report_path is not None:
            object.__setattr__(self, "verification_report_path", Path(self.verification_report_path))
        if self.workflow_report_path is not None:
            object.__setattr__(self, "workflow_report_path", Path(self.workflow_report_path))
        object.__setattr__(self, "readiness_baseline_keys", tuple(str(key) for key in self.readiness_baseline_keys))
        object.__setattr__(self, "route_baseline_keys", tuple(str(key) for key in self.route_baseline_keys))

    @property
    def output_root(self) -> Path:
        if self.workflow_report_path is not None:
            return self.workflow_report_path.parent
        if self.release_report_path is not None:
            return self.release_report_path.parent
        if self.artifact_manifest_path is not None:
            return self.artifact_manifest_path.parent
        return self.release_registry_path.parent

    @property
    def report_path(self) -> Path:
        return self.workflow_report_path or self.output_root / "release-candidate-registry-workflow.json"

    @property
    def comparison_path(self) -> Path:
        return self.release_report_path or self.output_root / "release-candidate-comparison.json"

    @property
    def manifest_path(self) -> Path:
        return self.artifact_manifest_path or self.output_root / "release-candidate-artifact-manifest.json"

    @property
    def verification_path(self) -> Path:
        return self.verification_report_path or self.output_root / "release-candidate-manifest-verification.json"


def run_release_candidate_registry_workflow(
    config: ReleaseCandidateRegistryWorkflowConfig,
) -> dict[str, Any]:
    """Run release comparison, write an artifact manifest, and register when eligible."""
    comparison = compare_release_candidates(
        readiness_registry_path=config.readiness_registry_path,
        route_registry_path=config.route_registry_path,
        readiness_baseline_keys=config.readiness_baseline_keys,
        route_baseline_keys=config.route_baseline_keys,
        recursive=config.recursive,
        allow_unverified=config.allow_unverified,
        min_best_quality_auroc=config.min_best_quality_auroc,
        max_uncached_forward_seconds=config.max_uncached_forward_seconds,
        max_cache_only_seconds=config.max_cache_only_seconds,
        min_selected=config.min_selected,
        min_decision_accuracy=config.min_decision_accuracy,
        max_false_supported_rate=config.max_false_supported_rate,
        min_false_refuted_rate=config.min_false_refuted_rate,
        max_verified_false_alarm=config.max_verified_false_alarm,
        min_verified_detection=config.min_verified_detection,
        max_mean_duration_seconds=config.max_mean_duration_seconds,
        max_p99_duration_seconds=config.max_p99_duration_seconds,
        max_max_duration_seconds=config.max_max_duration_seconds,
        max_mean_attempted_route_count=config.max_mean_attempted_route_count,
        max_retrieval_use_rate=config.max_retrieval_use_rate,
        notes=("release candidate registry workflow",),
    )
    config.comparison_path.parent.mkdir(parents=True, exist_ok=True)
    config.comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_artifact_manifest(config, comparison)

    release_decision = dict(comparison.get("decision") or {})
    release_status = str(release_decision.get("status"))
    promotion = None
    blocking_reasons = []
    if release_status != "promote":
        blocking_reasons.append("release candidate comparison did not promote")

    if release_status == "promote" or config.allow_non_promote:
        promotion = promote_artifact_manifest(
            manifest_path=config.manifest_path,
            registry_path=config.release_registry_path,
            name=config.name,
            version=config.version,
            verification_report_path=config.verification_path,
            recursive=True,
            allow_failures=config.allow_promotion_failures,
            metadata=_promotion_metadata(config, comparison),
        )
        if not dict(promotion.get("verification") or {}).get("passed", False):
            blocking_reasons.append("release candidate manifest verification did not pass")

    decision = _registry_workflow_decision(
        release_status=release_status,
        promotion=promotion,
        blocking_reasons=blocking_reasons,
    )
    payload = {
        "schema_version": 1,
        "workflow": "release_candidate_registry_workflow",
        "config": {
            "readiness_registry": str(config.readiness_registry_path),
            "route_registry": str(config.route_registry_path or config.readiness_registry_path),
            "release_registry": str(config.release_registry_path),
            "name": config.name,
            "version": config.version,
            "release_report": str(config.comparison_path),
            "artifact_manifest": str(config.manifest_path),
            "allow_non_promote": config.allow_non_promote,
            "allow_promotion_failures": config.allow_promotion_failures,
        },
        "release_candidate_comparison": comparison,
        "promotion": promotion,
        "decision": decision,
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _write_artifact_manifest(
    config: ReleaseCandidateRegistryWorkflowConfig,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = dict(comparison.get("release_candidate") or {})
    manifests = dict(candidate.get("manifests") or {})
    artifacts: dict[str, str | Path | None] = {
        "release_candidate_report": config.comparison_path,
        "readiness_manifest": manifests.get("readiness_manifest"),
        "route_manifest": manifests.get("route_manifest"),
    }
    manifest = build_artifact_manifest(
        artifacts,
        root=config.manifest_path.parent,
        metadata=_manifest_metadata(comparison),
    )
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _registry_workflow_decision(
    *,
    release_status: str,
    promotion: Mapping[str, Any] | None,
    blocking_reasons: Sequence[str],
) -> dict[str, Any]:
    verification = {} if promotion is None else dict(promotion.get("verification") or {})
    verified = bool(verification.get("passed", False))
    status = "promote" if release_status == "promote" and verified else "blocked"
    return {
        "status": status,
        "release_candidate_status": release_status,
        "manifest_promoted": promotion is not None,
        "manifest_verified": verified,
        "registry_record": None if promotion is None else dict(promotion.get("records") or {}).get(
            "benchmark_manifest"
        ),
        "blocking_reasons": tuple(blocking_reasons),
    }


def _manifest_metadata(comparison: Mapping[str, Any]) -> dict[str, Any]:
    decision = dict(comparison.get("decision") or {})
    candidate = dict(comparison.get("release_candidate") or {})
    runtime = dict(candidate.get("runtime") or {})
    quality = dict(candidate.get("quality") or {})
    best_quality = dict(quality.get("best_quality_signal") or {})
    runtime_cost = dict(candidate.get("runtime_cost") or {})
    verifier_route = dict(candidate.get("verifier_route") or {})
    manifests = dict(candidate.get("manifests") or {})
    return {
        "runner": "run_release_candidate_registry_workflow",
        "workflow": comparison.get("workflow"),
        "release_candidate_status": decision.get("status"),
        "release_readiness_status": decision.get("readiness_status"),
        "release_route_status": decision.get("route_status"),
        "recommended_readiness_record": decision.get("recommended_readiness_record"),
        "recommended_route_record": decision.get("recommended_route_record"),
        "recommended_model": decision.get("recommended_model"),
        "recommended_route": decision.get("recommended_route"),
        "recommended_layer": runtime.get("layer"),
        "recommended_batch_size": runtime.get("batch_size"),
        "recommended_hidden_state_capture": runtime.get("hidden_state_capture"),
        "recommended_max_batch_tokens": runtime.get("max_batch_tokens"),
        "recommended_prefix_kv_cache": runtime.get("prefix_kv_cache"),
        "recommended_max_workers": runtime.get("max_workers"),
        "recommended_best_quality_signal": best_quality.get("name"),
        "recommended_best_quality_auroc": best_quality.get("auroc"),
        "recommended_quality_signals": quality.get("quality_signals"),
        "recommended_uncached_forward_cost_seconds": runtime_cost.get("uncached_forward_cost_seconds"),
        "recommended_uncached_forward_cost_source": runtime_cost.get("uncached_forward_cost_source"),
        "recommended_cache_only_total_seconds": runtime_cost.get("cache_only_total_seconds"),
        "recommended_route_selected": verifier_route.get("selected"),
        "recommended_route_decision_accuracy": verifier_route.get("decision_accuracy"),
        "recommended_route_false_supported_rate": verifier_route.get("false_supported_rate"),
        "recommended_route_false_refuted_rate": verifier_route.get("false_refuted_rate"),
        "recommended_route_p99_duration_seconds": verifier_route.get("p99_duration_seconds"),
        "recommended_route_mean_attempted_route_count": verifier_route.get("mean_attempted_route_count"),
        "readiness_manifest": manifests.get("readiness_manifest"),
        "route_manifest": manifests.get("route_manifest"),
    }


def _promotion_metadata(
    config: ReleaseCandidateRegistryWorkflowConfig,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = {
        **_manifest_metadata(comparison),
        "workflow": "run_release_candidate_registry_workflow",
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


def _config_from_args(args: argparse.Namespace) -> ReleaseCandidateRegistryWorkflowConfig:
    return ReleaseCandidateRegistryWorkflowConfig(
        readiness_registry_path=Path(args.readiness_registry),
        route_registry_path=None if args.route_registry is None else Path(args.route_registry),
        release_registry_path=Path(args.release_registry),
        name=args.name,
        version=args.version,
        readiness_baseline_keys=tuple(args.readiness_baseline_key or ()),
        route_baseline_keys=tuple(args.route_baseline_key or ()),
        release_report_path=None if args.release_report_json is None else Path(args.release_report_json),
        artifact_manifest_path=None if args.artifact_manifest is None else Path(args.artifact_manifest),
        verification_report_path=None if args.verification_report is None else Path(args.verification_report),
        workflow_report_path=None if args.json is None else Path(args.json),
        recursive=not args.no_recursive,
        allow_unverified=bool(args.allow_unverified),
        min_best_quality_auroc=args.min_best_quality_auroc,
        max_uncached_forward_seconds=args.max_uncached_forward_seconds,
        max_cache_only_seconds=args.max_cache_only_seconds,
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
        promotion_metadata=_parse_metadata(args.metadata or ()),
        allow_non_promote=bool(args.allow_non_promote),
        allow_promotion_failures=bool(args.allow_promotion_failures),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = run_release_candidate_registry_workflow(_config_from_args(args))
    decision = payload["decision"]
    print(
        "release_candidate_registry="
        f"{decision['status']} "
        f"release={decision.get('release_candidate_status')} "
        f"record={decision.get('registry_record')}"
    )
    if args.fail_on_blocked and decision["status"] != "promote":
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run release candidate gates and register the verified manifest")
    parser.add_argument("--readiness-registry", required=True)
    parser.add_argument("--route-registry", default=None)
    parser.add_argument("--release-registry", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--readiness-baseline-key", action="append", default=[])
    parser.add_argument("--route-baseline-key", action="append", default=[])
    parser.add_argument("--json", default=None, help="optional registry workflow report path")
    parser.add_argument("--release-report-json", default=None,
                        help="optional path for the release candidate comparison report")
    parser.add_argument("--artifact-manifest", default=None,
                        help="optional path for the release candidate artifact manifest")
    parser.add_argument("--verification-report", default=None)
    parser.add_argument("--metadata", action="append", default=[], help="extra promotion metadata as key=value")
    parser.add_argument("--allow-non-promote", action="store_true",
                        help="register even when the release candidate comparison does not promote")
    parser.add_argument("--allow-promotion-failures", action="store_true",
                        help="register even when manifest verification fails")
    parser.add_argument("--no-recursive", action="store_true", help="only verify root manifests")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="allow unverified input baseline manifests to become candidates")
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
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless the release candidate registry workflow promotes")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
