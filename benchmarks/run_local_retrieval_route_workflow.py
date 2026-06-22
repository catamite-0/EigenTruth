"""Build, promote, and optionally register a local retrieval route baseline."""

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

from benchmarks.build_evidence_fixture import build_evidence_fixture, load_corpus, load_score_dump  # noqa: E402
from benchmarks.eval_verifier_ensemble import build_verifier_ensemble_report  # noqa: E402
from benchmarks.promote_artifact_manifest import promote_artifact_manifest  # noqa: E402
from benchmarks.run_adapter_promotion_workflow import (  # noqa: E402
    AdapterPromotionWorkflowConfig,
    run_adapter_promotion_workflow,
)
from eigentruth.registry import build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class LocalRetrievalRouteWorkflowConfig:
    """Configuration for a local retrieval evidence route workflow."""

    scores_path: Path
    corpus_paths: Sequence[Path]
    output_dir: Path
    registry_path: Path | None = None
    name: str | None = None
    version: str | None = None
    score_name: str = "retrieval"
    signal: str = "truth_proj"
    direction: str | None = None
    alpha: float = 0.10
    repeats: int = 1
    seed: int = 0
    query_field: str = "answer"
    verifier_min_overlap: float = 0.65
    retriever_min_overlap: float = 0.20
    retrieval_limit: int = 5
    min_selected: int = 1
    gate_routes: Sequence[str] = ("retrieval_groundedness",)
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
    claims_path: Path | None = None
    verifier_report_path: Path | None = None
    route_report_path: Path | None = None
    promotion_report_path: Path | None = None
    artifact_manifest_path: Path | None = None
    verification_report_path: Path | None = None
    workflow_report_path: Path | None = None
    compact_json: bool = False
    allow_non_promote: bool = False
    allow_promotion_failures: bool = False
    promotion_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores_path", Path(self.scores_path))
        object.__setattr__(self, "corpus_paths", tuple(Path(path) for path in self.corpus_paths))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for attr in (
            "registry_path",
            "claims_path",
            "verifier_report_path",
            "route_report_path",
            "promotion_report_path",
            "artifact_manifest_path",
            "verification_report_path",
            "workflow_report_path",
        ):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, Path(value))
        object.__setattr__(self, "gate_routes", tuple(str(route) for route in self.gate_routes))
        if not self.corpus_paths:
            raise ValueError("corpus_paths must contain at least one local corpus path.")
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if int(self.repeats) < 1:
            raise ValueError("repeats must be >= 1.")
        if int(self.retrieval_limit) <= 0:
            raise ValueError("retrieval_limit must be positive.")
        registry_fields = (self.registry_path, self.name, self.version)
        if any(value is not None for value in registry_fields) and not all(registry_fields):
            raise ValueError("registry_path, name, and version must be provided together.")

    @property
    def resolved_claims_path(self) -> Path:
        return self.claims_path or self.output_dir / "retrieval-claims.json"

    @property
    def resolved_verifier_report_path(self) -> Path:
        return self.verifier_report_path or self.output_dir / "retrieval-verifier-report.json"

    @property
    def resolved_route_report_path(self) -> Path:
        return self.route_report_path or self.output_dir / "retrieval-route-comparison.json"

    @property
    def resolved_promotion_report_path(self) -> Path:
        return self.promotion_report_path or self.output_dir / "retrieval-route-promotion.json"

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        return self.artifact_manifest_path or self.output_dir / "retrieval-route-artifact-manifest.json"

    @property
    def resolved_verification_report_path(self) -> Path:
        return self.verification_report_path or self.output_dir / "retrieval-route-manifest-verification.json"

    @property
    def resolved_workflow_report_path(self) -> Path:
        return self.workflow_report_path or self.output_dir / "local-retrieval-route-workflow.json"


def run_local_retrieval_route_workflow(config: LocalRetrievalRouteWorkflowConfig) -> dict[str, Any]:
    """Run local retrieval evidence construction, route promotion, and optional registration."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    score_dump = load_score_dump(config.scores_path)
    corpus_documents = load_corpus(config.corpus_paths)
    claims_fixture = build_evidence_fixture(
        score_dump,
        corpus_documents,
        retriever_min_overlap=config.retriever_min_overlap,
        retrieval_limit=config.retrieval_limit,
        query_field=config.query_field,
    )
    _write_json(config.resolved_claims_path, claims_fixture, compact=config.compact_json)

    verifier_report = build_verifier_ensemble_report(
        ((config.score_name, config.scores_path),),
        signal=config.signal,
        claims_path=config.resolved_claims_path,
        direction=config.direction,
        alphas=(config.alpha,),
        repeats=config.repeats,
        seed=config.seed,
        verifier_min_overlap=config.verifier_min_overlap,
        retriever_min_overlap=config.retriever_min_overlap,
        retrieval_limit=config.retrieval_limit,
    )
    _write_json(config.resolved_verifier_report_path, verifier_report, compact=config.compact_json)

    promotion_report = run_adapter_promotion_workflow(
        AdapterPromotionWorkflowConfig(
            reports=((config.score_name, config.resolved_verifier_report_path),),
            route_report_path=config.resolved_route_report_path,
            alpha=config.alpha,
            min_selected=config.min_selected,
            notes=("local retrieval route workflow",),
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
            compact_json=config.compact_json,
        )
    )
    _write_json(config.resolved_promotion_report_path, promotion_report, compact=config.compact_json)

    metadata = _manifest_metadata(
        config,
        claims_fixture=claims_fixture,
        promotion_report=promotion_report,
    )
    _write_artifact_manifest(config, metadata=metadata)
    manifest_promotion = _maybe_register_manifest(config, promotion_report, metadata)
    decision = _workflow_decision(
        promotion_report=promotion_report,
        manifest_promotion=manifest_promotion,
        registry_requested=config.registry_path is not None,
    )
    payload = {
        "schema_version": 1,
        "workflow": "local_retrieval_route_workflow",
        "config": {
            "scores_path": str(config.scores_path),
            "corpus_paths": [str(path) for path in config.corpus_paths],
            "score_name": config.score_name,
            "signal": config.signal,
            "direction": config.direction,
            "alpha": float(config.alpha),
            "repeats": int(config.repeats),
            "seed": int(config.seed),
            "query_field": config.query_field,
            "verifier_min_overlap": float(config.verifier_min_overlap),
            "retriever_min_overlap": float(config.retriever_min_overlap),
            "retrieval_limit": int(config.retrieval_limit),
            "gate_routes": list(config.gate_routes),
            "registry": None if config.registry_path is None else str(config.registry_path),
            "name": config.name,
            "version": config.version,
        },
        "claims_path": str(config.resolved_claims_path),
        "verifier_report_path": str(config.resolved_verifier_report_path),
        "route_report_path": str(config.resolved_route_report_path),
        "promotion_report_path": str(config.resolved_promotion_report_path),
        "artifact_manifest_path": str(config.resolved_artifact_manifest_path),
        "claims_summary": claims_fixture["summary"],
        "adapter_promotion": promotion_report,
        "manifest_promotion": manifest_promotion,
        "decision": decision,
    }
    _write_json(config.resolved_workflow_report_path, payload, compact=config.compact_json)
    return payload


def _maybe_register_manifest(
    config: LocalRetrievalRouteWorkflowConfig,
    promotion_report: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    if config.registry_path is None:
        return None
    decision = dict(promotion_report.get("decision") or {})
    if decision.get("status") != "promote" and not config.allow_non_promote:
        return None
    return promote_artifact_manifest(
        manifest_path=config.resolved_artifact_manifest_path,
        registry_path=config.registry_path,
        name=str(config.name),
        version=str(config.version),
        verification_report_path=config.resolved_verification_report_path,
        recursive=True,
        allow_failures=config.allow_promotion_failures,
        metadata={
            **dict(metadata),
            "workflow": "run_local_retrieval_route_workflow",
        },
    )


def _write_artifact_manifest(
    config: LocalRetrievalRouteWorkflowConfig,
    *,
    metadata: Mapping[str, Any],
) -> None:
    artifacts: dict[str, str | Path | None] = {
        "score_dump": config.scores_path,
        "retrieval_claims": config.resolved_claims_path,
        "verifier_report": config.resolved_verifier_report_path,
        "route_comparison_report": config.resolved_route_report_path,
        "promotion_report": config.resolved_promotion_report_path,
    }
    for idx, path in enumerate(config.corpus_paths, start=1):
        artifacts[f"retrieval_corpora.{idx}.{path.stem}"] = path
    manifest = build_artifact_manifest(
        artifacts,
        root=config.resolved_artifact_manifest_path.parent,
        metadata=metadata,
    )
    _write_json(config.resolved_artifact_manifest_path, manifest, compact=False)


def _manifest_metadata(
    config: LocalRetrievalRouteWorkflowConfig,
    *,
    claims_fixture: Mapping[str, Any],
    promotion_report: Mapping[str, Any],
) -> dict[str, Any]:
    decision = dict(promotion_report.get("decision") or {})
    route_comparison = dict(promotion_report.get("route_comparison") or {})
    route_decision = dict(route_comparison.get("promotion_decision") or {})
    recommended_route = route_decision.get("recommended_route") or decision.get("recommended_route")
    by_route = dict(route_comparison.get("by_route") or {})
    recommended_metrics = dict(by_route.get(str(recommended_route)) or {}) if recommended_route else {}
    quality_gate = dict(route_comparison.get("quality_gate") or {})
    summary = dict(claims_fixture.get("summary") or {})
    metadata = {
        "runner": "run_local_retrieval_route_workflow",
        "workflow": "local_retrieval_route_workflow",
        "source_workflow": promotion_report.get("workflow"),
        "adapter_promotion_status": decision.get("status"),
        "route_promotion_status": route_decision.get("status"),
        "recommended_route": recommended_route,
        "quality_gate_passed": quality_gate.get("passed"),
        "quality_gate_checked_routes": quality_gate.get("checked_routes"),
        "score_name": config.score_name,
        "signal": config.signal,
        "query_field": config.query_field,
        "verifier_min_overlap": float(config.verifier_min_overlap),
        "retriever_min_overlap": float(config.retriever_min_overlap),
        "retrieval_limit": int(config.retrieval_limit),
        "corpus_count": len(config.corpus_paths),
        "claims_n_records": summary.get("n_records"),
        "claims_records_with_hits": summary.get("records_with_hits"),
        "claims_total_hits": summary.get("total_hits"),
        "claims_average_hits_per_record": summary.get("average_hits_per_record"),
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
    }
    if config.promotion_metadata is not None:
        metadata.update(dict(config.promotion_metadata))
    return metadata


def _workflow_decision(
    *,
    promotion_report: Mapping[str, Any],
    manifest_promotion: Mapping[str, Any] | None,
    registry_requested: bool,
) -> dict[str, Any]:
    adapter_decision = dict(promotion_report.get("decision") or {})
    adapter_status = str(adapter_decision.get("status"))
    blocking_reasons = []
    if adapter_status != "promote":
        blocking_reasons.append("adapter promotion decision did not promote")
    verification = {} if manifest_promotion is None else dict(manifest_promotion.get("verification") or {})
    manifest_verified = bool(verification.get("passed", False))
    if registry_requested and manifest_promotion is None:
        blocking_reasons.append("registry promotion was not attempted")
    if registry_requested and manifest_promotion is not None and not manifest_verified:
        blocking_reasons.append("retrieval route manifest verification did not pass")
    status = "promote" if not blocking_reasons else "blocked"
    return {
        "status": status,
        "adapter_promotion_status": adapter_status,
        "manifest_promoted": manifest_promotion is not None,
        "manifest_verified": manifest_verified,
        "registry_record": None if manifest_promotion is None else dict(manifest_promotion.get("records") or {}).get(
            "benchmark_manifest"
        ),
        "blocking_reasons": tuple(blocking_reasons),
    }


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if compact
        else json.dumps(payload, indent=2, sort_keys=True)
    )
    path.write_text(text + "\n", encoding="utf-8")


def _parse_non_negative_float(value: str, *, flag: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{flag} must be a non-negative finite number.")
    return numeric


def _parse_positive_float(value: str, *, flag: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{flag} must be a positive finite number.")
    return numeric


def _parse_non_negative_int(value: str, *, flag: str) -> int:
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{flag} must be a non-negative integer.")
    return numeric


def _parse_positive_int(value: str, *, flag: str) -> int:
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"{flag} must be a positive integer.")
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


def _config_from_args(args: argparse.Namespace) -> LocalRetrievalRouteWorkflowConfig:
    return LocalRetrievalRouteWorkflowConfig(
        scores_path=Path(args.scores),
        corpus_paths=tuple(Path(path) for path in args.corpus),
        output_dir=Path(args.output_dir),
        registry_path=None if args.registry is None else Path(args.registry),
        name=args.name,
        version=args.version,
        score_name=args.score_name,
        signal=args.signal,
        direction=args.direction,
        alpha=args.alpha,
        repeats=args.repeats,
        seed=args.seed,
        query_field=args.query_field,
        verifier_min_overlap=args.verifier_min_overlap,
        retriever_min_overlap=args.retriever_min_overlap,
        retrieval_limit=args.retrieval_limit,
        min_selected=args.min_selected,
        gate_routes=tuple(args.gate_route or ("retrieval_groundedness",)),
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
        claims_path=None if args.claims_json is None else Path(args.claims_json),
        verifier_report_path=None if args.verifier_report_json is None else Path(args.verifier_report_json),
        route_report_path=None if args.route_report_json is None else Path(args.route_report_json),
        promotion_report_path=None if args.promotion_report_json is None else Path(args.promotion_report_json),
        artifact_manifest_path=None if args.artifact_manifest is None else Path(args.artifact_manifest),
        verification_report_path=None if args.verification_report is None else Path(args.verification_report),
        workflow_report_path=None if args.json is None else Path(args.json),
        compact_json=bool(args.compact_json),
        allow_non_promote=bool(args.allow_non_promote),
        allow_promotion_failures=bool(args.allow_promotion_failures),
        promotion_metadata=_parse_metadata(args.metadata or ()),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = run_local_retrieval_route_workflow(_config_from_args(args))
    decision = payload["decision"]
    print(
        "local_retrieval_route="
        f"{decision['status']} "
        f"adapter={decision.get('adapter_promotion_status')} "
        f"record={decision.get('registry_record')}"
    )
    if args.fail_on_blocked and decision["status"] != "promote":
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build and optionally register a local retrieval-groundedness route baseline"
    )
    parser.add_argument("--scores", required=True, help="statement-bearing score dump JSON")
    parser.add_argument("--corpus", action="append", required=True, help="local JSON/JSONL/text corpus path")
    parser.add_argument("--output-dir", required=True, help="directory for workflow artifacts")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry baseline name; required with --registry")
    parser.add_argument("--version", default=None, help="registry baseline version; required with --registry")
    parser.add_argument("--score-name", default="retrieval")
    parser.add_argument("--signal", default="truth_proj")
    parser.add_argument("--direction", default=None)
    parser.add_argument("--alpha", type=lambda value: _parse_positive_float(value, flag="--alpha"), default=0.10)
    parser.add_argument("--repeats", type=lambda value: _parse_positive_int(value, flag="--repeats"), default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--query-field",
        choices=("text", "answer", "question", "question_answer"),
        default="answer",
    )
    parser.add_argument("--verifier-min-overlap", type=float, default=0.65)
    parser.add_argument("--retriever-min-overlap", type=float, default=0.20)
    parser.add_argument("--retrieval-limit", type=lambda value: _parse_positive_int(
        value,
        flag="--retrieval-limit",
    ), default=5)
    parser.add_argument("--min-selected", type=lambda value: _parse_non_negative_int(
        value,
        flag="--min-selected",
    ), default=1)
    parser.add_argument("--gate-route", action="append", default=None)
    parser.add_argument("--gate-min-selected", type=lambda value: _parse_non_negative_int(
        value,
        flag="--gate-min-selected",
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
    parser.add_argument("--max-p95-duration-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-p95-duration-seconds",
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
    parser.add_argument("--min-cache-hit-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-cache-hit-rate",
    ), default=None)
    parser.add_argument("--claims-json", default=None)
    parser.add_argument("--verifier-report-json", default=None)
    parser.add_argument("--route-report-json", default=None)
    parser.add_argument("--promotion-report-json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--verification-report", default=None)
    parser.add_argument("--json", default=None, help="optional workflow report path")
    parser.add_argument("--metadata", action="append", default=[], help="extra promotion metadata as key=value")
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--allow-non-promote", action="store_true")
    parser.add_argument("--allow-promotion-failures", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
