"""Compare a current profile against a verified registry baseline manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.compare_profiles import build_profile_comparison  # noqa: E402
from eigentruth.registry import ArtifactRegistry, RegistryRecord, load_and_verify_artifact_manifest  # noqa: E402


def compare_registry_baseline(
    *,
    registry_path: str | Path,
    candidate_profiles: Sequence[tuple[str, str | Path]],
    baseline_key: str | None = None,
    baseline_name: str | None = None,
    baseline_version: str | None = None,
    baseline_profile_artifact: str = "profiles.uncached",
    recursive: bool = True,
    allow_unverified: bool = False,
    max_total_ratio: float | None = None,
    max_run_total_ratios: Mapping[str, float] | None = None,
    max_phase_ratios: Mapping[str, float] | None = None,
    min_throughput_ratios: Mapping[str, float] | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a profile regression report from a registered benchmark manifest."""
    if not candidate_profiles:
        raise ValueError("at least one candidate profile is required.")
    registry = ArtifactRegistry.load_json(registry_path)
    record = _select_manifest_record(
        registry,
        baseline_key=baseline_key,
        baseline_name=baseline_name,
        baseline_version=baseline_version,
    )
    manifest_path = Path(record.path)
    verification = load_and_verify_artifact_manifest(manifest_path, recursive=recursive)
    verification_payload = verification.to_dict()
    if not verification.passed and not allow_unverified:
        raise ValueError("baseline artifact manifest verification failed; use --allow-unverified to compare anyway")

    baseline_profile_path = _resolve_manifest_artifact_path(
        manifest_path,
        artifact_name=baseline_profile_artifact,
    )
    comparison = build_profile_comparison(
        [("baseline", baseline_profile_path), *_normalize_candidate_profiles(candidate_profiles)],
        baseline="baseline",
        notes=(
            "registry-backed baseline profile comparison",
            *notes,
        ),
        max_total_ratio=max_total_ratio,
        max_run_total_ratios=max_run_total_ratios,
        max_phase_ratios=max_phase_ratios,
        min_throughput_ratios=min_throughput_ratios,
    )
    return {
        "registry_baseline": {
            "registry": str(registry_path),
            "record_key": record.key(),
            "record": record.to_dict(),
            "manifest_path": str(manifest_path),
            "profile_artifact": baseline_profile_artifact,
            "profile_path": str(baseline_profile_path),
            "verification": verification_payload,
            "allow_unverified": allow_unverified,
        },
        "comparison": comparison,
    }


def _select_manifest_record(
    registry: ArtifactRegistry,
    *,
    baseline_key: str | None,
    baseline_name: str | None,
    baseline_version: str | None,
) -> RegistryRecord:
    if baseline_key:
        record = registry.get(baseline_key)
    else:
        if not baseline_name or not baseline_version:
            raise ValueError("provide --baseline-key or both --baseline-name and --baseline-version.")
        record = registry.get(f"benchmark_manifest:{baseline_name}:{baseline_version}")
    if record.artifact_type != "benchmark_manifest":
        raise ValueError(f"registry record {record.key()!r} is not a benchmark_manifest.")
    return record


def _resolve_manifest_artifact_path(manifest_path: Path, *, artifact_name: str) -> Path:
    parts = tuple(part.strip() for part in artifact_name.split("::"))
    if not parts or any(not part for part in parts):
        raise ValueError("artifact reference must not be empty.")
    current_manifest_path = manifest_path
    for index, part in enumerate(parts):
        path = _resolve_single_manifest_artifact_path(current_manifest_path, artifact_name=part)
        if index == len(parts) - 1:
            return path
        current_manifest_path = path
    raise AssertionError("unreachable artifact reference resolution path")


def _resolve_single_manifest_artifact_path(manifest_path: Path, *, artifact_name: str) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifact manifest must contain an 'artifacts' mapping.")
    artifact = artifacts.get(artifact_name)
    if not isinstance(artifact, Mapping):
        raise ValueError(f"artifact manifest does not contain artifact {artifact_name!r}.")
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"artifact {artifact_name!r} does not contain a valid path.")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _normalize_candidate_profiles(
    profiles: Sequence[tuple[str, str | Path]],
) -> list[tuple[str, Path]]:
    normalized = []
    seen = {"baseline"}
    for name, path in profiles:
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("candidate profile name cannot be empty.")
        if clean_name in seen:
            raise ValueError(f"profile name {clean_name!r} is duplicated or reserved.")
        seen.add(clean_name)
        normalized.append((clean_name, Path(path)))
    return normalized


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("candidate profile name cannot be empty.")
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = compare_registry_baseline(
        registry_path=args.registry,
        baseline_key=args.baseline_key,
        baseline_name=args.baseline_name,
        baseline_version=args.baseline_version,
        baseline_profile_artifact=args.baseline_profile_artifact,
        candidate_profiles=[_parse_named_path(value) for value in args.candidate_profile],
        recursive=not args.no_recursive,
        allow_unverified=bool(args.allow_unverified),
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
        notes=args.note,
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote registry baseline comparison to {output_path}")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare current profiles against a verified registry baseline")
    parser.add_argument("--registry", required=True, help="local ArtifactRegistry JSON path")
    parser.add_argument("--baseline-key", default=None, help="benchmark_manifest registry key")
    parser.add_argument("--baseline-name", default=None, help="benchmark manifest record name")
    parser.add_argument("--baseline-version", default=None, help="benchmark manifest record version")
    parser.add_argument("--baseline-profile-artifact", default="profiles.uncached",
                        help="profile artifact name inside the manifest")
    parser.add_argument("--candidate-profile", action="append", required=True,
                        help="candidate profile JSON path, optionally named as name=path; repeatable")
    parser.add_argument("--note", action="append", default=[],
                        help="optional note to include in the comparison report; repeatable")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--no-recursive", action="store_true", help="only verify the root baseline manifest")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="compare even when baseline manifest verification fails")
    parser.add_argument("--max-total-ratio", type=float, default=None,
                        help="fail when any candidate exceeds this total-time ratio")
    parser.add_argument("--max-run-total-ratio", action="append", default=[],
                        help="fail when one named run exceeds this total-time ratio, formatted as run=ratio")
    parser.add_argument("--max-phase-ratio", action="append", default=[],
                        help="fail when a phase exceeds this ratio, formatted as phase=ratio; repeatable")
    parser.add_argument("--min-throughput-ratio", action="append", default=[],
                        help="fail when throughput drops below this ratio, formatted as metric=ratio; repeatable")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="exit non-zero when the generated comparison gate fails")
    args = parser.parse_args(argv)
    payload = run(args)
    comparison = payload["comparison"]
    for item in comparison["runs"]:
        if item["name"] == "baseline":
            continue
        delta = item["total_delta"]
        print(
            f"{item['name']}: total={item['total_seconds']:.3f}s "
            f"ratio={delta['ratio_to_baseline']} "
            f"bottleneck={item['bottleneck']}"
        )
    gate = comparison.get("regression_gate")
    if gate is not None:
        print(f"regression_gate={'passed' if gate['passed'] else 'failed'}")
        if not gate["passed"] and args.fail_on_regression:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
