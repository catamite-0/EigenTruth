"""Compare registered adapter readiness baselines and recommend a runtime."""

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

from benchmarks.recommend_runtime_config import build_runtime_recommendation  # noqa: E402
from eigentruth.registry import ArtifactRegistry, RegistryRecord, load_and_verify_artifact_manifest  # noqa: E402


def compare_readiness_baselines(
    *,
    registry_path: str | Path,
    baseline_keys: Sequence[str] = (),
    recursive: bool = True,
    allow_unverified: bool = False,
    min_best_quality_auroc: float | None = None,
    max_uncached_forward_seconds: float | None = None,
    max_cache_only_seconds: float | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a fail-closed comparison of registered readiness baselines."""
    registry = ArtifactRegistry.load_json(registry_path)
    records = _select_records(registry, baseline_keys=baseline_keys)
    rows = [
        _readiness_row(
            record,
            recursive=recursive,
            allow_unverified=allow_unverified,
            min_best_quality_auroc=min_best_quality_auroc,
            max_uncached_forward_seconds=max_uncached_forward_seconds,
            max_cache_only_seconds=max_cache_only_seconds,
        )
        for record in records
    ]
    leaderboard = sorted(rows, key=_leaderboard_key)
    recommendation = next((row for row in leaderboard if row["gate"]["passed"]), None)
    decision = _decision(leaderboard, recommendation)
    return {
        "schema_version": 1,
        "workflow": "readiness_baseline_comparison",
        "registry": str(registry_path),
        "config": {
            "baseline_keys": list(baseline_keys),
            "recursive": recursive,
            "allow_unverified": allow_unverified,
            "min_best_quality_auroc": min_best_quality_auroc,
            "max_uncached_forward_seconds": max_uncached_forward_seconds,
            "max_cache_only_seconds": max_cache_only_seconds,
        },
        "summary": {
            "record_count": len(rows),
            "passing_count": sum(1 for row in rows if row["gate"]["passed"]),
            "recommended_record": None if recommendation is None else recommendation["record_key"],
        },
        "decision": decision,
        "leaderboard": leaderboard,
        "notes": list(notes),
    }


def _select_records(
    registry: ArtifactRegistry,
    *,
    baseline_keys: Sequence[str],
) -> tuple[RegistryRecord, ...]:
    if baseline_keys:
        records = tuple(registry.get(key) for key in baseline_keys)
    else:
        records = tuple(
            record
            for record in registry.list_records(artifact_type="benchmark_manifest")
            if _is_readiness_record(record)
        )
    for record in records:
        if record.artifact_type != "benchmark_manifest":
            raise ValueError(f"registry record {record.key()!r} is not a benchmark_manifest.")
    return records


def _is_readiness_record(record: RegistryRecord) -> bool:
    metadata = dict(record.metadata)
    manifest_metadata = _mapping(metadata.get("manifest_metadata"))
    return (
        metadata.get("workflow") == "run_adapter_readiness_registry_workflow"
        or manifest_metadata.get("runner") == "run_adapter_readiness_workflow"
        or metadata.get("readiness_status") is not None
    )


def _readiness_row(
    record: RegistryRecord,
    *,
    recursive: bool,
    allow_unverified: bool,
    min_best_quality_auroc: float | None,
    max_uncached_forward_seconds: float | None,
    max_cache_only_seconds: float | None,
) -> dict[str, Any]:
    manifest_path = Path(record.path)
    manifest, manifest_error = _load_optional_json(manifest_path)
    verification = _verify_manifest(manifest_path, recursive=recursive)
    manifest_metadata = _manifest_metadata(record, manifest)
    runtime_recommendation, runtime_source = _runtime_recommendation_from_manifest(
        manifest_path,
        manifest,
        manifest_metadata,
    )
    recommendation = _mapping(runtime_recommendation.get("recommendation"))
    best_quality = _best_quality(recommendation, manifest_metadata)
    quality_signals = _quality_signals(recommendation, manifest_metadata)
    uncached_cost = _uncached_forward_cost(recommendation)
    cache_only_seconds = _float_or_none(recommendation.get("cache_only_total_seconds"))
    gate = _gate(
        verification=verification,
        allow_unverified=allow_unverified,
        manifest_error=manifest_error,
        readiness_status=manifest_metadata.get("readiness_status"),
        runtime_status=runtime_recommendation.get("status"),
        best_quality=best_quality,
        uncached_forward_seconds=uncached_cost["seconds"],
        cache_only_seconds=cache_only_seconds,
        min_best_quality_auroc=min_best_quality_auroc,
        max_uncached_forward_seconds=max_uncached_forward_seconds,
        max_cache_only_seconds=max_cache_only_seconds,
    )
    return {
        "record_key": record.key(),
        "name": record.name,
        "version": record.version,
        "manifest_path": str(manifest_path),
        "verification": verification,
        "gate": gate,
        "runtime_recommendation_source": runtime_source,
        "readiness_status": manifest_metadata.get("readiness_status"),
        "adapter_family_status": manifest_metadata.get("adapter_family_status"),
        "performance_status": manifest_metadata.get("performance_status"),
        "runtime_recommendation_status": runtime_recommendation.get("status"),
        "model": manifest_metadata.get("model"),
        "dtype": manifest_metadata.get("dtype"),
        "recommended_route": manifest_metadata.get("recommended_route"),
        "recommended_performance_cell": manifest_metadata.get("recommended_performance_cell"),
        "layer": recommendation.get("layer", manifest_metadata.get("recommended_layer")),
        "batch_size": recommendation.get("batch_size", manifest_metadata.get("recommended_batch_size")),
        "hidden_state_capture": recommendation.get(
            "hidden_state_capture",
            manifest_metadata.get("recommended_hidden_state_capture"),
        ),
        "max_batch_tokens": recommendation.get(
            "max_batch_tokens",
            manifest_metadata.get("recommended_max_batch_tokens"),
        ),
        "prefix_kv_cache": recommendation.get(
            "prefix_kv_cache",
            manifest_metadata.get("recommended_prefix_kv_cache"),
        ),
        "max_workers": recommendation.get("max_workers", manifest_metadata.get("recommended_max_workers")),
        "quality_signals": quality_signals,
        "best_quality_signal": best_quality,
        "truth_proj_auroc": _float_or_none(recommendation.get("truth_proj_auroc")),
        "uncached_total_seconds": _float_or_none(recommendation.get("uncached_total_seconds")),
        "uncached_forced_answer_forward_seconds": _float_or_none(
            recommendation.get("uncached_forced_answer_forward_seconds")
        ),
        "uncached_forward_cost_seconds": uncached_cost["seconds"],
        "uncached_forward_cost_source": uncached_cost["source"],
        "cache_only_total_seconds": cache_only_seconds,
        "performance_wall_clock_seconds": _float_or_none(
            manifest_metadata.get("performance_wall_clock_seconds")
        ),
        "wall_clock_seconds": _float_or_none(manifest_metadata.get("wall_clock_seconds")),
        "benchmark_flags": runtime_recommendation.get("benchmark_flags"),
    }


def _runtime_recommendation_from_manifest(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    performance_matrix_path = _resolve_artifact_path(
        manifest_path,
        manifest,
        artifact_name="performance_matrix_report",
    )
    if performance_matrix_path is not None:
        matrix_report, _ = _load_optional_json(performance_matrix_path)
        if matrix_report:
            return (
                build_runtime_recommendation(
                    matrix_report,
                    matrix_report_path=performance_matrix_path,
                ),
                str(performance_matrix_path),
            )

    runtime_path = _resolve_artifact_path(
        manifest_path,
        manifest,
        artifact_name="runtime_recommendation",
    )
    if runtime_path is not None:
        runtime, _ = _load_optional_json(runtime_path)
        if runtime:
            return runtime, str(runtime_path)

    runtime_status = manifest_metadata.get("runtime_recommendation_status")
    if runtime_status is None:
        return {}, None
    quality_signals = _mapping(manifest_metadata.get("recommended_quality_signals"))
    best_quality = {
        "name": manifest_metadata.get("recommended_best_quality_signal"),
        "auroc": manifest_metadata.get("recommended_best_quality_auroc"),
    }
    return (
        {
            "status": runtime_status,
            "recommendation": {
                "layer": manifest_metadata.get("recommended_layer"),
                "batch_size": manifest_metadata.get("recommended_batch_size"),
                "hidden_state_capture": manifest_metadata.get("recommended_hidden_state_capture"),
                "max_batch_tokens": manifest_metadata.get("recommended_max_batch_tokens"),
                "prefix_kv_cache": manifest_metadata.get("recommended_prefix_kv_cache"),
                "max_workers": manifest_metadata.get("recommended_max_workers"),
                "quality_signals": quality_signals,
                "best_quality_signal": best_quality,
            },
        },
        "registry_metadata",
    )


def _manifest_metadata(record: RegistryRecord, manifest: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(record.metadata)
    manifest_metadata = _mapping(metadata.get("manifest_metadata"))
    if not manifest_metadata:
        manifest_metadata = _mapping(manifest.get("metadata"))
    merged = dict(manifest_metadata)
    merged.update(metadata)
    nested = _mapping(metadata.get("manifest_metadata"))
    merged.update(nested)
    return merged


def _verify_manifest(manifest_path: Path, *, recursive: bool) -> dict[str, Any]:
    try:
        return load_and_verify_artifact_manifest(manifest_path, recursive=recursive).to_dict()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "manifest_path": str(manifest_path),
            "passed": False,
            "checked": 0,
            "failures": [
                {
                    "name": "manifest",
                    "path": str(manifest_path),
                    "field": "load",
                    "expected": "readable artifact manifest",
                    "actual": str(exc),
                }
            ],
            "nested": [],
        }


def _gate(
    *,
    verification: Mapping[str, Any],
    allow_unverified: bool,
    manifest_error: str | None,
    readiness_status: Any,
    runtime_status: Any,
    best_quality: Mapping[str, Any] | None,
    uncached_forward_seconds: float | None,
    cache_only_seconds: float | None,
    min_best_quality_auroc: float | None,
    max_uncached_forward_seconds: float | None,
    max_cache_only_seconds: float | None,
) -> dict[str, Any]:
    failures = []
    if manifest_error is not None:
        failures.append(f"manifest could not be loaded: {manifest_error}")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("manifest verification failed")
    if readiness_status != "promote":
        failures.append(f"readiness_status is {readiness_status!r}, expected 'promote'")
    if runtime_status != "promote":
        failures.append(f"runtime_recommendation_status is {runtime_status!r}, expected 'promote'")
    quality_auroc = None if best_quality is None else _float_or_none(best_quality.get("auroc"))
    quality_name = None if best_quality is None else best_quality.get("name")
    if not quality_name or quality_auroc is None:
        failures.append("best quality signal is missing")
    if min_best_quality_auroc is not None and (
        quality_auroc is None or quality_auroc < min_best_quality_auroc
    ):
        failures.append(f"best quality AUROC below {min_best_quality_auroc}")
    if max_uncached_forward_seconds is not None and (
        uncached_forward_seconds is None or uncached_forward_seconds > max_uncached_forward_seconds
    ):
        failures.append(f"uncached forward cost seconds above {max_uncached_forward_seconds}")
    if max_cache_only_seconds is not None and (
        cache_only_seconds is None or cache_only_seconds > max_cache_only_seconds
    ):
        failures.append(f"cache-only total seconds above {max_cache_only_seconds}")
    return {
        "passed": not failures,
        "blocking_reasons": failures,
    }


def _decision(
    leaderboard: Sequence[Mapping[str, Any]],
    recommendation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not leaderboard:
        return {
            "status": "no_candidate",
            "recommended_record": None,
            "blocking_reasons": ("no readiness benchmark_manifest records selected",),
        }
    if recommendation is not None:
        return {
            "status": "promote",
            "recommended_record": recommendation["record_key"],
            "recommended_model": recommendation.get("model"),
            "recommended_best_quality_signal": recommendation.get("best_quality_signal"),
            "blocking_reasons": (),
        }
    return {
        "status": "blocked",
        "recommended_record": None,
        "blocking_reasons": tuple(
            f"{row['record_key']}: {reason}"
            for row in leaderboard
            for reason in row["gate"]["blocking_reasons"]
        ),
    }


def _leaderboard_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    best_quality = _mapping(row.get("best_quality_signal"))
    best_auroc = _float_or_none(best_quality.get("auroc"))
    forward = _float_or_none(row.get("uncached_forward_cost_seconds"))
    cache_only = _float_or_none(row.get("cache_only_total_seconds"))
    return (
        not _mapping(row.get("gate")).get("passed", False),
        -(best_auroc if best_auroc is not None else -math.inf),
        forward if forward is not None else math.inf,
        cache_only if cache_only is not None else math.inf,
        str(row.get("record_key")),
    )


def _resolve_artifact_path(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    artifact_name: str,
) -> Path | None:
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return None
    artifact = artifacts.get(artifact_name)
    if not isinstance(artifact, Mapping):
        return None
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _load_optional_json(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, str(exc)
    if not isinstance(payload, dict):
        return {}, f"{path} did not contain a JSON object"
    return payload, None


def _best_quality(
    recommendation: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    best = _mapping(recommendation.get("best_quality_signal"))
    name = best.get("name")
    auroc = _float_or_none(best.get("auroc"))
    if name and auroc is not None:
        return {"name": str(name), "auroc": auroc}
    fallback_name = manifest_metadata.get("recommended_best_quality_signal")
    fallback_auroc = _float_or_none(manifest_metadata.get("recommended_best_quality_auroc"))
    if fallback_name and fallback_auroc is not None:
        return {"name": str(fallback_name), "auroc": fallback_auroc}
    truth_proj = _float_or_none(recommendation.get("truth_proj_auroc"))
    if truth_proj is not None:
        return {"name": "truth_proj", "auroc": truth_proj}
    return None


def _quality_signals(
    recommendation: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
) -> dict[str, float]:
    signals = _finite_float_mapping(_mapping(recommendation.get("quality_signals")))
    if not signals:
        signals = _finite_float_mapping(_mapping(manifest_metadata.get("recommended_quality_signals")))
    truth_proj = _float_or_none(recommendation.get("truth_proj_auroc"))
    if truth_proj is not None and "truth_proj" not in signals:
        signals["truth_proj"] = truth_proj
    return {name: signals[name] for name in sorted(signals)}


def _uncached_forward_cost(recommendation: Mapping[str, Any]) -> dict[str, Any]:
    forced = _float_or_none(recommendation.get("uncached_forced_answer_forward_seconds"))
    if forced is not None:
        return {
            "seconds": forced,
            "source": "uncached_forced_answer_forward_seconds",
        }
    total = _float_or_none(recommendation.get("uncached_total_seconds"))
    if total is not None:
        return {
            "seconds": total,
            "source": "uncached_total_seconds_fallback",
        }
    return {
        "seconds": None,
        "source": None,
    }


def _finite_float_mapping(values: Mapping[str, Any]) -> dict[str, float]:
    signals = {}
    for key, value in values.items():
        numeric = _float_or_none(value)
        if numeric is not None:
            signals[str(key)] = numeric
    return signals


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        numeric = float(value)
    else:
        try:
            numeric = float(str(value))
        except (TypeError, ValueError):
            return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _parse_non_negative_float(value: str, *, flag: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{flag} must be a non-negative finite number.")
    return numeric


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = compare_readiness_baselines(
        registry_path=args.registry,
        baseline_keys=tuple(args.baseline_key or ()),
        recursive=not args.no_recursive,
        allow_unverified=bool(args.allow_unverified),
        min_best_quality_auroc=args.min_best_quality_auroc,
        max_uncached_forward_seconds=args.max_uncached_forward_seconds,
        max_cache_only_seconds=args.max_cache_only_seconds,
        notes=args.note,
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote readiness baseline comparison to {output_path}")
    decision = payload["decision"]
    print(
        "readiness_baseline_comparison="
        f"{decision['status']} recommended={decision.get('recommended_record')}"
    )
    if args.fail_on_blocked and decision["status"] != "promote":
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare registered adapter-readiness benchmark manifests"
    )
    parser.add_argument("--registry", required=True, help="local ArtifactRegistry JSON path")
    parser.add_argument("--baseline-key", action="append", default=[],
                        help="benchmark_manifest registry key to compare; repeatable")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--note", action="append", default=[],
                        help="optional note to include in the comparison report; repeatable")
    parser.add_argument("--no-recursive", action="store_true",
                        help="only verify root manifests")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="allow unverified manifests to become candidates")
    parser.add_argument("--min-best-quality-auroc", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-best-quality-auroc",
    ), default=None)
    parser.add_argument("--max-uncached-forward-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-uncached-forward-seconds",
    ), default=None,
                        help="max uncached forward cost; uses uncached total as fallback for legacy reports")
    parser.add_argument("--max-cache-only-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-cache-only-seconds",
    ), default=None)
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless a readiness baseline promotes")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
