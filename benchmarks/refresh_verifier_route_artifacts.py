"""Refresh verifier route artifacts from saved score dumps without rerunning models."""

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

from benchmarks.eval_verifier_ensemble import build_verifier_ensemble_report  # noqa: E402
from benchmarks.run_adapter_promotion_workflow import (  # noqa: E402
    AdapterPromotionWorkflowConfig,
    run_adapter_promotion_workflow,
)


@dataclass(frozen=True)
class VerifierRouteArtifactRefreshConfig:
    """Configuration for refreshing new-schema verifier route artifacts."""

    score_dumps: Sequence[tuple[str, Path]]
    verifier_report_path: Path
    claims_path: Path | None = None
    qa_corpus_path: Path | None = None
    state_path: Path | None = None
    signal: str = "truth_proj"
    direction: str | None = None
    alphas: Sequence[float] = (0.10,)
    repeats: int = 5
    seed: int = 0
    verifier_min_overlap: float = 0.65
    retriever_min_overlap: float = 0.2
    retrieval_limit: int = 5
    enable_triple_evidence: bool = False
    triple_min_slot_coverage: float = 1.0
    promotion_report_path: Path | None = None
    route_report_path: Path | None = None
    promotion_notes: Sequence[str] = ()
    promotion_gate_routes: Sequence[str] = ()
    promotion_gate_min_selected: int | None = None
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "score_dumps", tuple((str(name), Path(path)) for name, path in self.score_dumps))
        object.__setattr__(self, "verifier_report_path", Path(self.verifier_report_path))
        if self.claims_path is not None:
            object.__setattr__(self, "claims_path", Path(self.claims_path))
        if self.qa_corpus_path is not None:
            object.__setattr__(self, "qa_corpus_path", Path(self.qa_corpus_path))
        if self.state_path is not None:
            object.__setattr__(self, "state_path", Path(self.state_path))
        if self.promotion_report_path is not None:
            object.__setattr__(self, "promotion_report_path", Path(self.promotion_report_path))
        if self.route_report_path is not None:
            object.__setattr__(self, "route_report_path", Path(self.route_report_path))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        object.__setattr__(self, "alphas", tuple(float(alpha) for alpha in self.alphas))
        object.__setattr__(self, "promotion_notes", tuple(str(note) for note in self.promotion_notes))
        object.__setattr__(
            self,
            "promotion_gate_routes",
            tuple(str(route) for route in self.promotion_gate_routes),
        )
        object.__setattr__(
            self,
            "candidate_profiles",
            tuple((str(name), Path(path)) for name, path in self.candidate_profiles),
        )


def refresh_verifier_route_artifacts(config: VerifierRouteArtifactRefreshConfig) -> dict[str, Any]:
    """Refresh verifier route artifacts and optionally run adapter promotion."""
    verifier_report = build_verifier_ensemble_report(
        config.score_dumps,
        signal=config.signal,
        claims_path=config.claims_path,
        qa_corpus_path=config.qa_corpus_path,
        state_path=config.state_path,
        direction=config.direction,
        alphas=config.alphas,
        repeats=config.repeats,
        seed=config.seed,
        verifier_min_overlap=config.verifier_min_overlap,
        retriever_min_overlap=config.retriever_min_overlap,
        retrieval_limit=config.retrieval_limit,
        enable_triple_evidence=config.enable_triple_evidence,
        triple_min_slot_coverage=config.triple_min_slot_coverage,
    )
    config.verifier_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.verifier_report_path.write_text(
        _json_text(verifier_report, compact=config.compact_json, sort_keys=True),
        encoding="utf-8",
    )

    promotion = None
    if config.promotion_report_path is not None:
        route_report_path = config.route_report_path or config.promotion_report_path.with_name("route-comparison.json")
        promotion = run_adapter_promotion_workflow(
            AdapterPromotionWorkflowConfig(
                reports=(("refreshed", config.verifier_report_path),),
                route_report_path=route_report_path,
                alpha=float(config.alphas[0]),
                min_selected=1,
                notes=config.promotion_notes,
                gate_routes=config.promotion_gate_routes,
                gate_min_selected=config.promotion_gate_min_selected,
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
                registry_path=config.registry_path,
                baseline_key=config.baseline_key,
                baseline_name=config.baseline_name,
                baseline_version=config.baseline_version,
                baseline_profile_artifact=config.baseline_profile_artifact,
                candidate_profiles=config.candidate_profiles,
                allow_unverified_compare=config.allow_unverified_compare,
                max_total_ratio=config.max_total_ratio,
                max_run_total_ratios=config.max_run_total_ratios,
                max_phase_ratios=config.max_phase_ratios,
                min_throughput_ratios=config.min_throughput_ratios,
                compact_json=config.compact_json,
            )
        )
        config.promotion_report_path.parent.mkdir(parents=True, exist_ok=True)
        config.promotion_report_path.write_text(
            _json_text(promotion, compact=config.compact_json, sort_keys=True),
            encoding="utf-8",
        )

    return {
        "schema_version": 1,
        "workflow": "refresh_verifier_route_artifacts",
        "verifier_report_path": str(config.verifier_report_path),
        "verifier_report_summary": _verifier_report_summary(verifier_report),
        "promotion_report_path": None if config.promotion_report_path is None else str(config.promotion_report_path),
        "promotion": promotion,
    }


def _json_text(payload: Mapping[str, Any], *, compact: bool, sort_keys: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=sort_keys, separators=(",", ":")) + "\n"
    return json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n"


def _verifier_report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    runs = []
    for run in report.get("runs", ()):
        if not isinstance(run, Mapping):
            continue
        route_quality = run.get("route_quality", {})
        routes = {}
        if isinstance(route_quality, Mapping):
            for route, payload in route_quality.items():
                if not isinstance(payload, Mapping):
                    continue
                routes[str(route)] = {
                    "selected": int(payload.get("selected", 0)),
                    "decision_accuracy": payload.get("decision_accuracy"),
                    "false_supported_rate": payload.get("false_supported_rate"),
                    "false_refuted_rate": payload.get("false_refuted_rate"),
                    "mean_duration_seconds": payload.get("mean_duration_seconds"),
                    "p95_duration_seconds": payload.get("p95_duration_seconds"),
                    "p99_duration_seconds": payload.get("p99_duration_seconds"),
                    "max_duration_seconds": payload.get("max_duration_seconds"),
                    "mean_attempted_route_count": payload.get("mean_attempted_route_count"),
                    "retrieval_use_rate": payload.get("retrieval_use_rate"),
                }
        runs.append({
            "name": str(run.get("name", "")),
            "n_total": int(run.get("n_total", 0)),
            "routes": routes,
            "cache_stats": _cache_summary(run.get("cache_stats", {})),
        })
    transition_verifier = report.get("transition_verifier", {})
    transition_summary = {}
    if isinstance(transition_verifier, Mapping):
        transition_summary = {
            "enabled": transition_verifier.get("enabled"),
            "world_model_adapter": transition_verifier.get("world_model_adapter"),
            "world_model_rule_count": transition_verifier.get("world_model_rule_count"),
            "min_prediction_confidence": transition_verifier.get("min_prediction_confidence"),
            "global_transitions": transition_verifier.get("global_transitions"),
        }
    return {
        "signal": report.get("signal"),
        "alphas": list(report.get("alphas", ())),
        "transition_verifier": transition_summary,
        "runs": runs,
    }


def _cache_summary(cache_stats: Any) -> dict[str, Any]:
    if not isinstance(cache_stats, Mapping):
        return {}
    summary = {}
    fields = ("requests", "hits", "misses", "hit_rate", "instances")
    for name, payload in cache_stats.items():
        if not isinstance(payload, Mapping):
            continue
        compact = {field: payload[field] for field in fields if field in payload}
        if compact:
            summary[str(name)] = compact
    return summary


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("named path name cannot be empty.")
    return name, Path(path)


def _parse_alphas(value: str) -> tuple[float, ...]:
    alphas = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not alphas:
        raise ValueError("--alphas must contain at least one value.")
    if any(not (0.0 < alpha < 1.0) for alpha in alphas):
        raise ValueError("alphas must be in (0, 1).")
    return alphas


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


def _config_from_args(args: argparse.Namespace) -> VerifierRouteArtifactRefreshConfig:
    return VerifierRouteArtifactRefreshConfig(
        score_dumps=tuple(_parse_named_path(value) for value in args.scores),
        verifier_report_path=Path(args.verifier_report_json),
        claims_path=None if args.claims is None else Path(args.claims),
        qa_corpus_path=None if args.qa_corpus is None else Path(args.qa_corpus),
        state_path=None if args.state_source is None else Path(args.state_source),
        signal=args.signal,
        direction=args.direction,
        alphas=_parse_alphas(args.alphas),
        repeats=args.repeats,
        seed=args.seed,
        verifier_min_overlap=args.verifier_min_overlap,
        retriever_min_overlap=args.retriever_min_overlap,
        retrieval_limit=args.retrieval_limit,
        enable_triple_evidence=bool(args.enable_triple_evidence),
        triple_min_slot_coverage=args.triple_min_slot_coverage,
        promotion_report_path=None if args.promotion_json is None else Path(args.promotion_json),
        route_report_path=None if args.route_report_json is None else Path(args.route_report_json),
        promotion_notes=args.note,
        promotion_gate_routes=tuple(args.gate_route),
        promotion_gate_min_selected=args.gate_min_selected,
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
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = refresh_verifier_route_artifacts(_config_from_args(args))
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _json_text(payload, compact=bool(args.compact_json), sort_keys=True),
            encoding="utf-8",
        )
        print(f"Wrote verifier route refresh workflow report to {output_path}")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Refresh verifier route artifacts from saved score dumps")
    parser.add_argument("--scores", action="append", required=True,
                        help="score dump path, optionally named as name=path; repeatable")
    parser.add_argument("--verifier-report-json", required=True,
                        help="path to write the refreshed verifier ensemble report")
    parser.add_argument("--claims", default=None, help="optional claim/evidence fixture JSON")
    parser.add_argument("--qa-corpus", default=None, help="optional structured QA corpus JSON")
    parser.add_argument("--state-source", default=None, help="optional structured state JSON")
    parser.add_argument("--signal", default="truth_proj")
    parser.add_argument("--direction", choices=("higher", "lower"), default=None)
    parser.add_argument("--alphas", default="0.1")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verifier-min-overlap", type=float, default=0.65)
    parser.add_argument("--retriever-min-overlap", type=float, default=0.2)
    parser.add_argument("--retrieval-limit", type=int, default=5)
    parser.add_argument("--enable-triple-evidence", action="store_true",
                        help="enable strict subject-predicate-object evidence audits")
    parser.add_argument("--triple-min-slot-coverage", type=float, default=1.0,
                        help="minimum per-slot evidence coverage for triple-evidence audits")
    parser.add_argument("--promotion-json", default=None,
                        help="optional path to write adapter promotion workflow output")
    parser.add_argument("--route-report-json", default=None,
                        help="optional route comparison path; defaults next to --promotion-json")
    parser.add_argument("--note", action="append", default=[],
                        help="optional note for promotion route comparison; repeatable")
    parser.add_argument("--gate-route", action="append", default=[],
                        help="promotion route to gate; repeatable")
    parser.add_argument("--gate-min-selected", type=int, default=None)
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
                        help="optional local ArtifactRegistry JSON path for promotion baseline comparison")
    parser.add_argument("--baseline-key", default=None)
    parser.add_argument("--baseline-name", default=None)
    parser.add_argument("--baseline-version", default=None)
    parser.add_argument("--baseline-profile-artifact", default="profiles.uncached")
    parser.add_argument("--candidate-profile", action="append", default=[],
                        help="candidate profile JSON path, optionally named as name=path; repeatable")
    parser.add_argument("--allow-unverified-compare", action="store_true")
    parser.add_argument("--max-total-ratio", type=float, default=None)
    parser.add_argument("--max-run-total-ratio", action="append", default=[])
    parser.add_argument("--max-phase-ratio", action="append", default=[])
    parser.add_argument("--min-throughput-ratio", action="append", default=[])
    parser.add_argument("--json", default=None, help="optional path to write refresh workflow summary")
    parser.add_argument("--compact-json", action="store_true",
                        help="write minified JSON artifacts for lower artifact size and write latency")
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero when --promotion-json is set and promotion does not pass")
    args = parser.parse_args(argv)
    payload = run(args)
    promotion = payload.get("promotion")
    if isinstance(promotion, Mapping):
        decision = promotion["decision"]
        print(
            f"adapter_promotion={decision['status']} "
            f"route={decision.get('recommended_route')}"
        )
        if args.fail_on_blocked and decision["status"] != "promote":
            raise SystemExit(1)
    elif args.fail_on_blocked:
        raise SystemExit("--fail-on-blocked requires --promotion-json")


if __name__ == "__main__":
    main()
