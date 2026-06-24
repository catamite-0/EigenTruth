"""Sweep bounded worker counts for artifact-manifest fingerprint verification."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import strict_bool, strict_positive_int  # noqa: E402
from eigentruth.registry import (  # noqa: E402
    ArtifactRegistry,
    ArtifactVerificationContext,
    load_fingerprint_cache,
)


@dataclass(frozen=True)
class ManifestFingerprintWorkerSweepConfig:
    """Configuration for verifying manifests across fingerprint worker counts."""

    manifest_paths: Sequence[str | Path]
    output_path: str | Path
    worker_counts: Sequence[int] = (1, 2, 4)
    repeats: int = 1
    fingerprint_cache_path: str | Path | None = None
    recursive: bool = True
    allow_failures: bool = False
    compact_json: bool = False
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        manifest_paths = tuple(Path(path) for path in self.manifest_paths)
        if not manifest_paths:
            raise ValueError("manifest_paths must not be empty.")
        object.__setattr__(self, "manifest_paths", manifest_paths)
        object.__setattr__(self, "output_path", Path(self.output_path))
        worker_counts = tuple(strict_positive_int(value, name="worker_counts") for value in self.worker_counts)
        if not worker_counts:
            raise ValueError("worker_counts must not be empty.")
        if len(worker_counts) != len(set(worker_counts)):
            raise ValueError("worker_counts must not contain duplicates.")
        object.__setattr__(self, "worker_counts", worker_counts)
        repeats = strict_positive_int(self.repeats, name="repeats")
        object.__setattr__(self, "repeats", repeats)
        if self.fingerprint_cache_path is not None:
            object.__setattr__(self, "fingerprint_cache_path", Path(self.fingerprint_cache_path))
        object.__setattr__(self, "recursive", strict_bool(self.recursive, name="recursive"))
        object.__setattr__(self, "allow_failures", strict_bool(self.allow_failures, name="allow_failures"))
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
            if not self.name or not self.version:
                raise ValueError("registry_path requires name and version.")
        if (self.name is None) != (self.version is None):
            raise ValueError("registry recording requires both name and version.")
        object.__setattr__(self, "metadata", dict(self.metadata))


def build_manifest_fingerprint_worker_sweep(
    config: ManifestFingerprintWorkerSweepConfig,
) -> dict[str, Any]:
    """Verify manifests under each worker count and write a sweep report."""
    seed_cache = load_fingerprint_cache(config.fingerprint_cache_path)
    samples = []
    for worker_count in config.worker_counts:
        for repeat_index in range(config.repeats):
            samples.append(_run_sample(config, worker_count, repeat_index, seed_cache=seed_cache))
    summaries = [_worker_summary(worker_count, samples) for worker_count in config.worker_counts]
    leaderboard = _leaderboard(summaries)
    decision = _decision(summaries, leaderboard, allow_failures=config.allow_failures)
    status = (
        "blocked"
        if decision["recommended_worker_count"] is None or decision["blocking_reasons"]
        else "observed"
    )
    payload = {
        "schema_version": 1,
        "workflow": "manifest_fingerprint_worker_sweep",
        "status": status,
        "decision": decision,
        "leaderboard": leaderboard,
        "worker_summaries": summaries,
        "samples": samples,
        "paths": {
            "report": str(config.output_path),
            "manifests": tuple(str(path) for path in config.manifest_paths),
            "fingerprint_cache": (
                None if config.fingerprint_cache_path is None else str(config.fingerprint_cache_path)
            ),
            "registry": None if config.registry_path is None else str(config.registry_path),
        },
        "config": {
            "worker_counts": tuple(config.worker_counts),
            "repeats": config.repeats,
            "recursive": config.recursive,
            "allow_failures": config.allow_failures,
            "compact_json": config.compact_json,
            "metadata": dict(config.metadata),
        },
        "registry_record": None if config.registry_path is None else f"report:{config.name}:{config.version}",
    }
    _write_json(config.output_path, payload, compact=config.compact_json)
    if config.registry_path is not None:
        _record_registry(config, payload)
    return payload


def _run_sample(
    config: ManifestFingerprintWorkerSweepConfig,
    worker_count: int,
    repeat_index: int,
    *,
    seed_cache: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    context = ArtifactVerificationContext(fingerprint_cache=_copy_fingerprint_cache(seed_cache))
    started_at = time.perf_counter()
    verifications: list[dict[str, Any]] = []
    errors = []
    for manifest_path in config.manifest_paths:
        try:
            verification = context.load_and_verify_artifact_manifest(
                manifest_path,
                recursive=config.recursive,
                max_workers=worker_count,
            )
            verifications.append(verification.to_dict())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({
                "manifest_path": str(manifest_path),
                "error": str(exc),
            })
            if not config.allow_failures:
                break
    total_seconds = _round_seconds(time.perf_counter() - started_at)
    passed = not errors and all(bool(verification.get("passed")) for verification in verifications)
    return {
        "worker_count": int(worker_count),
        "repeat_index": int(repeat_index),
        "passed": passed,
        "total_seconds": total_seconds,
        "manifest_count": len(tuple(config.manifest_paths)),
        "checked": sum(_finite_int(verification.get("checked")) or 0 for verification in verifications),
        "failure_count": sum(_failure_count(verification) for verification in verifications),
        "error_count": len(errors),
        "errors": errors,
        "artifact_cache": context.cache_summary(),
        "verifications": verifications,
    }


def _worker_summary(worker_count: int, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    worker_samples = tuple(sample for sample in samples if int(sample.get("worker_count", 0)) == worker_count)
    seconds = _numeric_summary(sample.get("total_seconds") for sample in worker_samples)
    fingerprint_records = tuple(
        _mapping(_mapping(sample.get("artifact_cache")).get("artifact_fingerprint_cache"))
        for sample in worker_samples
    )
    requests = sum(_finite_int(record.get("requests")) or 0 for record in fingerprint_records)
    hits = sum(_finite_int(record.get("hits")) or 0 for record in fingerprint_records)
    return {
        "worker_count": int(worker_count),
        "sample_count": len(worker_samples),
        "passed_count": sum(1 for sample in worker_samples if bool(sample.get("passed"))),
        "error_count": sum(_finite_int(sample.get("error_count")) or 0 for sample in worker_samples),
        "failure_count": sum(_finite_int(sample.get("failure_count")) or 0 for sample in worker_samples),
        "checked": sum(_finite_int(sample.get("checked")) or 0 for sample in worker_samples),
        "total_seconds": seconds,
        "artifact_fingerprint_cache": {
            "requests": requests,
            "hits": hits,
            "misses": sum(_finite_int(record.get("misses")) or 0 for record in fingerprint_records),
            "weighted_hit_rate": _safe_div(hits, requests),
            "entries": _numeric_summary(record.get("entries") for record in fingerprint_records),
        },
    }


def _leaderboard(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    baseline_mean = _baseline_mean(summaries)
    for summary in summaries:
        mean_seconds = _finite_float(_mapping(summary.get("total_seconds")).get("mean"))
        worker_count = _finite_int(summary.get("worker_count"))
        sample_count = _finite_int(summary.get("sample_count")) or 0
        passed_count = _finite_int(summary.get("passed_count")) or 0
        row = {
            "worker_count": worker_count,
            "mean_seconds": mean_seconds,
            "min_seconds": _finite_float(_mapping(summary.get("total_seconds")).get("min")),
            "max_seconds": _finite_float(_mapping(summary.get("total_seconds")).get("max")),
            "passed": sample_count > 0 and passed_count == sample_count,
            "sample_count": sample_count,
            "speedup_vs_worker_1": None if baseline_mean in (None, 0.0) or mean_seconds in (None, 0.0) else (
                baseline_mean / mean_seconds
            ),
        }
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            not bool(row["passed"]),
            math.inf if row["mean_seconds"] is None else float(row["mean_seconds"]),
            math.inf if row["worker_count"] is None else int(row["worker_count"]),
        ),
    )


def _decision(
    summaries: Sequence[Mapping[str, Any]],
    leaderboard: Sequence[Mapping[str, Any]],
    *,
    allow_failures: bool,
) -> dict[str, Any]:
    recommended = next(
        (
            row
            for row in leaderboard
            if bool(row.get("passed")) and _finite_float(row.get("mean_seconds")) is not None
        ),
        None,
    )
    baseline_mean = _baseline_mean(summaries)
    recommended_mean = None if recommended is None else _finite_float(recommended.get("mean_seconds"))
    worker_failure_reasons = tuple(
        f"worker_count={summary.get('worker_count')} did not pass all verification samples"
        for summary in summaries
        if (_finite_int(summary.get("passed_count")) or 0) < (_finite_int(summary.get("sample_count")) or 0)
    )
    return {
        "recommended_worker_count": None if recommended is None else recommended.get("worker_count"),
        "recommended_mean_seconds": recommended_mean,
        "baseline_worker_count": 1 if baseline_mean is not None else None,
        "baseline_mean_seconds": baseline_mean,
        "speedup_vs_worker_1": (
            None
            if baseline_mean in (None, 0.0) or recommended_mean in (None, 0.0)
            else baseline_mean / recommended_mean
        ),
        "worker_failure_reasons": worker_failure_reasons,
        "blocking_reasons": () if allow_failures else worker_failure_reasons,
    }


def _baseline_mean(summaries: Sequence[Mapping[str, Any]]) -> float | None:
    baseline = next((summary for summary in summaries if int(summary.get("worker_count", 0)) == 1), None)
    if baseline is None:
        return None
    return _finite_float(_mapping(baseline.get("total_seconds")).get("mean"))


def _record_registry(config: ManifestFingerprintWorkerSweepConfig, payload: Mapping[str, Any]) -> None:
    decision = _mapping(payload.get("decision"))
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_report(
        name=str(config.name),
        path=config.output_path,
        version=str(config.version),
        metadata={
            "workflow": "manifest_fingerprint_worker_sweep",
            "status": payload.get("status"),
            "recommended_worker_count": decision.get("recommended_worker_count"),
            "recommended_mean_seconds": decision.get("recommended_mean_seconds"),
            "speedup_vs_worker_1": decision.get("speedup_vs_worker_1"),
            "manifest_count": len(tuple(config.manifest_paths)),
            "repeats": config.repeats,
            "recursive": config.recursive,
            "allow_failures": config.allow_failures,
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


def _copy_fingerprint_cache(cache: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(key): dict(value) for key, value in cache.items()}


def _failure_count(verification_payload: Mapping[str, Any]) -> int:
    count = len(tuple(verification_payload.get("failures", ())))
    for nested in verification_payload.get("nested", ()):
        if isinstance(nested, Mapping):
            count += _failure_count(nested)
    return count


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
            "max": None,
        }
    return {
        "count": len(finite_values),
        "missing_or_nonfinite": len(raw_values) - len(finite_values),
        "mean": sum(finite_values) / len(finite_values),
        "min": min(finite_values),
        "p50": _percentile(finite_values, 50.0),
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


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _finite_int(value: Any) -> int | None:
    numeric = _finite_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _round_seconds(value: float) -> float:
    return round(max(0.0, float(value)), 6)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one integer.")
    if any(value < 1 for value in values):
        raise ValueError("worker counts must be at least 1.")
    if len(values) != len(set(values)):
        raise ValueError("worker counts must not contain duplicates.")
    return values


def _parse_positive_int(value: str, *, flag: str) -> int:
    numeric = int(value)
    if numeric < 1:
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = build_manifest_fingerprint_worker_sweep(
        ManifestFingerprintWorkerSweepConfig(
            manifest_paths=tuple(args.manifest or ()),
            output_path=args.json,
            worker_counts=args.workers,
            repeats=args.repeats,
            fingerprint_cache_path=args.fingerprint_cache,
            recursive=not args.no_recursive,
            allow_failures=bool(args.allow_failures),
            compact_json=bool(args.compact_json),
            registry_path=args.registry,
            name=args.name,
            version=args.version,
            metadata=_parse_metadata(args.metadata or ()),
        )
    )
    print(
        "manifest_fingerprint_worker_sweep="
        f"{payload['status']} "
        f"recommended_workers={payload['decision'].get('recommended_worker_count')} "
        f"speedup_vs_worker_1={payload['decision'].get('speedup_vs_worker_1')}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sweep artifact-manifest fingerprint worker counts")
    parser.add_argument("--manifest", action="append", required=True, help="artifact manifest to verify; repeatable")
    parser.add_argument("--json", required=True, help="output worker-sweep report JSON")
    parser.add_argument(
        "--workers",
        type=_parse_int_list,
        default=(1, 2, 4),
        help="comma-separated worker counts to test; default: 1,2,4",
    )
    parser.add_argument(
        "--repeats",
        type=lambda value: _parse_positive_int(value, flag="--repeats"),
        default=1,
        help="number of verification samples per worker count",
    )
    parser.add_argument("--fingerprint-cache", default=None, help="optional seed fingerprint cache JSON")
    parser.add_argument("--no-recursive", action="store_true", help="only verify root manifests")
    parser.add_argument("--allow-failures", action="store_true", help="continue samples after verification failures")
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry report name")
    parser.add_argument("--version", default=None, help="registry report version")
    parser.add_argument("--metadata", action="append", default=[], help="extra registry metadata as key=value")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
