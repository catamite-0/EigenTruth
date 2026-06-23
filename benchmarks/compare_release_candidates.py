"""Compare registered readiness and route baselines as one release candidate."""

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

from benchmarks.compare_readiness_baselines import compare_readiness_baselines  # noqa: E402
from benchmarks.compare_route_baselines import compare_route_baselines  # noqa: E402


def compare_release_candidates(
    *,
    readiness_registry_path: str | Path,
    route_registry_path: str | Path | None = None,
    readiness_baseline_keys: Sequence[str] = (),
    route_baseline_keys: Sequence[str] = (),
    recursive: bool = True,
    allow_unverified: bool = False,
    min_best_quality_auroc: float | None = None,
    max_uncached_forward_seconds: float | None = None,
    max_cache_only_seconds: float | None = None,
    max_inside_sample_count_ratio: float | None = None,
    max_inside_generation_seconds_ratio: float | None = None,
    min_selected: int | None = None,
    min_decision_accuracy: float | None = None,
    max_false_supported_rate: float | None = None,
    min_false_refuted_rate: float | None = None,
    max_verified_false_alarm: float | None = None,
    min_verified_detection: float | None = None,
    max_mean_duration_seconds: float | None = None,
    max_p99_duration_seconds: float | None = None,
    max_max_duration_seconds: float | None = None,
    max_mean_attempted_route_count: float | None = None,
    max_retrieval_use_rate: float | None = None,
    max_runtime_total_seconds: float | None = None,
    max_retrieval_hit_count: float | None = None,
    min_claims_cache_hit_rate: float | None = None,
    min_verifier_trace_cache_hit_rate: float | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a fail-closed deployable release candidate from saved baselines."""
    route_registry_path = readiness_registry_path if route_registry_path is None else route_registry_path
    readiness = compare_readiness_baselines(
        registry_path=readiness_registry_path,
        baseline_keys=readiness_baseline_keys,
        recursive=recursive,
        allow_unverified=allow_unverified,
        min_best_quality_auroc=min_best_quality_auroc,
        max_uncached_forward_seconds=max_uncached_forward_seconds,
        max_cache_only_seconds=max_cache_only_seconds,
        max_inside_sample_count_ratio=max_inside_sample_count_ratio,
        max_inside_generation_seconds_ratio=max_inside_generation_seconds_ratio,
        notes=("release candidate readiness comparison",),
    )
    route = compare_route_baselines(
        registry_path=route_registry_path,
        baseline_keys=route_baseline_keys,
        recursive=recursive,
        allow_unverified=allow_unverified,
        min_selected=min_selected,
        min_decision_accuracy=min_decision_accuracy,
        max_false_supported_rate=max_false_supported_rate,
        min_false_refuted_rate=min_false_refuted_rate,
        max_verified_false_alarm=max_verified_false_alarm,
        min_verified_detection=min_verified_detection,
        max_mean_duration_seconds=max_mean_duration_seconds,
        max_p99_duration_seconds=max_p99_duration_seconds,
        max_max_duration_seconds=max_max_duration_seconds,
        max_mean_attempted_route_count=max_mean_attempted_route_count,
        max_retrieval_use_rate=max_retrieval_use_rate,
        max_runtime_total_seconds=max_runtime_total_seconds,
        max_retrieval_hit_count=max_retrieval_hit_count,
        min_claims_cache_hit_rate=min_claims_cache_hit_rate,
        min_verifier_trace_cache_hit_rate=min_verifier_trace_cache_hit_rate,
        notes=("release candidate route comparison",),
    )
    candidate = _release_candidate(readiness, route)
    decision = _decision(readiness, route, candidate)
    return {
        "schema_version": 1,
        "workflow": "release_candidate_comparison",
        "config": {
            "readiness_registry": str(readiness_registry_path),
            "route_registry": str(route_registry_path),
            "readiness_baseline_keys": list(readiness_baseline_keys),
            "route_baseline_keys": list(route_baseline_keys),
            "recursive": recursive,
            "allow_unverified": allow_unverified,
            "min_best_quality_auroc": min_best_quality_auroc,
            "max_uncached_forward_seconds": max_uncached_forward_seconds,
            "max_cache_only_seconds": max_cache_only_seconds,
            "max_inside_sample_count_ratio": max_inside_sample_count_ratio,
            "max_inside_generation_seconds_ratio": max_inside_generation_seconds_ratio,
            "min_selected": min_selected,
            "min_decision_accuracy": min_decision_accuracy,
            "max_false_supported_rate": max_false_supported_rate,
            "min_false_refuted_rate": min_false_refuted_rate,
            "max_verified_false_alarm": max_verified_false_alarm,
            "min_verified_detection": min_verified_detection,
            "max_mean_duration_seconds": max_mean_duration_seconds,
            "max_p99_duration_seconds": max_p99_duration_seconds,
            "max_max_duration_seconds": max_max_duration_seconds,
            "max_mean_attempted_route_count": max_mean_attempted_route_count,
            "max_retrieval_use_rate": max_retrieval_use_rate,
            "max_runtime_total_seconds": max_runtime_total_seconds,
            "max_retrieval_hit_count": max_retrieval_hit_count,
            "min_claims_cache_hit_rate": min_claims_cache_hit_rate,
            "min_verifier_trace_cache_hit_rate": min_verifier_trace_cache_hit_rate,
        },
        "readiness_baseline_comparison": readiness,
        "route_baseline_comparison": route,
        "release_candidate": candidate,
        "decision": decision,
        "notes": list(notes),
    }


def _release_candidate(
    readiness: Mapping[str, Any],
    route: Mapping[str, Any],
) -> dict[str, Any] | None:
    readiness_decision = _mapping(readiness.get("decision"))
    route_decision = _mapping(route.get("decision"))
    if readiness_decision.get("status") != "promote" or route_decision.get("status") != "promote":
        return None
    readiness_row = _recommended_row(readiness, readiness_decision.get("recommended_record"))
    route_row = _recommended_row(route, route_decision.get("recommended_record"))
    if not readiness_row or not route_row:
        return None
    return {
        "readiness_record": readiness_row.get("record_key"),
        "route_record": route_row.get("record_key"),
        "model": readiness_row.get("model"),
        "runtime": {
            "layer": readiness_row.get("layer"),
            "batch_size": readiness_row.get("batch_size"),
            "hidden_state_capture": readiness_row.get("hidden_state_capture"),
            "max_batch_tokens": readiness_row.get("max_batch_tokens"),
            "prefix_kv_cache": readiness_row.get("prefix_kv_cache"),
            "max_workers": readiness_row.get("max_workers"),
            "inside_sampling": readiness_row.get("inside_sampling"),
            "performance_cell": readiness_row.get("recommended_performance_cell"),
            "benchmark_flags": readiness_row.get("benchmark_flags"),
        },
        "quality": {
            "best_quality_signal": readiness_row.get("best_quality_signal"),
            "quality_signals": readiness_row.get("quality_signals"),
            "truth_proj_auroc": readiness_row.get("truth_proj_auroc"),
        },
        "runtime_cost": {
            "uncached_forward_cost_seconds": readiness_row.get("uncached_forward_cost_seconds"),
            "uncached_forward_cost_source": readiness_row.get("uncached_forward_cost_source"),
            "cache_only_total_seconds": readiness_row.get("cache_only_total_seconds"),
            "inside_sampling_recommended_run": readiness_row.get("inside_sampling_recommended_run"),
            "inside_sampling_total_generated_samples": readiness_row.get(
                "inside_sampling_total_generated_samples"
            ),
            "inside_sampling_sample_count_ratio_to_baseline": readiness_row.get(
                "inside_sampling_sample_count_ratio_to_baseline"
            ),
            "inside_generation_seconds": readiness_row.get("inside_generation_seconds"),
            "inside_generation_seconds_ratio_to_baseline": readiness_row.get(
                "inside_generation_seconds_ratio_to_baseline"
            ),
            "inside_sampling_stop_reason_counts": readiness_row.get("inside_sampling_stop_reason_counts"),
        },
        "verifier_route": {
            "route": route_row.get("recommended_route"),
            "selected": route_row.get("selected"),
            "decision_accuracy": route_row.get("decision_accuracy"),
            "false_supported_rate": route_row.get("false_supported_rate"),
            "false_refuted_rate": route_row.get("false_refuted_rate"),
            "verified_false_alarm": route_row.get("verified_false_alarm"),
            "verified_detection": route_row.get("verified_detection"),
            "mean_duration_seconds": route_row.get("mean_duration_seconds"),
            "p99_duration_seconds": route_row.get("p99_duration_seconds"),
            "max_duration_seconds": route_row.get("max_duration_seconds"),
            "mean_attempted_route_count": route_row.get("mean_attempted_route_count"),
            "retrieval_use_rate": route_row.get("retrieval_use_rate"),
            "runtime_total_seconds": route_row.get("runtime_total_seconds"),
            "runtime_retrieval_hit_count": route_row.get("runtime_retrieval_hit_count"),
            "claims_cache_hit_rate": route_row.get("claims_cache_hit_rate"),
            "verifier_trace_cache_hit_rate": route_row.get("verifier_trace_cache_hit_rate"),
        },
        "manifests": {
            "readiness_manifest": readiness_row.get("manifest_path"),
            "route_manifest": route_row.get("manifest_path"),
        },
    }


def _recommended_row(
    report: Mapping[str, Any],
    record_key: Any,
) -> dict[str, Any]:
    if record_key is None:
        return {}
    for row in report.get("leaderboard", ()):
        row_map = _mapping(row)
        if row_map.get("record_key") == record_key:
            return row_map
    return {}


def _decision(
    readiness: Mapping[str, Any],
    route: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    readiness_decision = _mapping(readiness.get("decision"))
    route_decision = _mapping(route.get("decision"))
    readiness_status = readiness_decision.get("status")
    route_status = route_decision.get("status")
    blocking_reasons = []
    if readiness_status != "promote":
        blocking_reasons.append({
            "gate": "readiness_baseline",
            "status": readiness_status,
            "reasons": list(readiness_decision.get("blocking_reasons", ())),
        })
    if route_status != "promote":
        blocking_reasons.append({
            "gate": "route_baseline",
            "status": route_status,
            "reasons": list(route_decision.get("blocking_reasons", ())),
        })
    if candidate is None and not blocking_reasons:
        blocking_reasons.append({
            "gate": "release_candidate",
            "status": "blocked",
            "reasons": ["promoted baseline comparisons did not expose recommended rows"],
        })
    status = "promote" if candidate is not None else (
        "no_candidate" if "no_candidate" in {readiness_status, route_status} else "blocked"
    )
    return {
        "status": status,
        "readiness_status": readiness_status,
        "route_status": route_status,
        "recommended_readiness_record": None if candidate is None else candidate.get("readiness_record"),
        "recommended_route_record": None if candidate is None else candidate.get("route_record"),
        "recommended_model": None if candidate is None else candidate.get("model"),
        "recommended_route": None if candidate is None else _mapping(candidate.get("verifier_route")).get("route"),
        "blocking_reasons": blocking_reasons,
    }


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = compare_release_candidates(
        readiness_registry_path=args.readiness_registry,
        route_registry_path=args.route_registry,
        readiness_baseline_keys=tuple(args.readiness_baseline_key or ()),
        route_baseline_keys=tuple(args.route_baseline_key or ()),
        recursive=not args.no_recursive,
        allow_unverified=bool(args.allow_unverified),
        min_best_quality_auroc=args.min_best_quality_auroc,
        max_uncached_forward_seconds=args.max_uncached_forward_seconds,
        max_cache_only_seconds=args.max_cache_only_seconds,
        max_inside_sample_count_ratio=args.max_inside_sample_count_ratio,
        max_inside_generation_seconds_ratio=args.max_inside_generation_seconds_ratio,
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
        max_runtime_total_seconds=args.max_runtime_total_seconds,
        max_retrieval_hit_count=args.max_retrieval_hit_count,
        min_claims_cache_hit_rate=args.min_claims_cache_hit_rate,
        min_verifier_trace_cache_hit_rate=args.min_verifier_trace_cache_hit_rate,
        notes=args.note,
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote release candidate comparison to {output_path}")
    decision = payload["decision"]
    print(
        "release_candidate_comparison="
        f"{decision['status']} readiness={decision.get('recommended_readiness_record')} "
        f"route={decision.get('recommended_route_record')}"
    )
    if args.fail_on_blocked and decision["status"] != "promote":
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare readiness and verifier-route baselines as one release candidate"
    )
    parser.add_argument("--readiness-registry", required=True,
                        help="ArtifactRegistry JSON path containing readiness baselines")
    parser.add_argument("--route-registry", default=None,
                        help="ArtifactRegistry JSON path containing route baselines; defaults to readiness registry")
    parser.add_argument("--readiness-baseline-key", action="append", default=[],
                        help="readiness benchmark_manifest registry key to compare; repeatable")
    parser.add_argument("--route-baseline-key", action="append", default=[],
                        help="route benchmark_manifest registry key to compare; repeatable")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--note", action="append", default=[],
                        help="optional note to include in the comparison report; repeatable")
    parser.add_argument("--no-recursive", action="store_true", help="only verify root manifests")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="allow unverified manifests to become candidates")
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
    parser.add_argument("--max-inside-sample-count-ratio", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-inside-sample-count-ratio",
    ), default=None)
    parser.add_argument("--max-inside-generation-seconds-ratio", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-inside-generation-seconds-ratio",
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
    parser.add_argument("--max-runtime-total-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-runtime-total-seconds",
    ), default=None)
    parser.add_argument("--max-retrieval-hit-count", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-retrieval-hit-count",
    ), default=None)
    parser.add_argument("--min-claims-cache-hit-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-claims-cache-hit-rate",
    ), default=None)
    parser.add_argument("--min-verifier-trace-cache-hit-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-verifier-trace-cache-hit-rate",
    ), default=None)
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless the release candidate promotes")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
