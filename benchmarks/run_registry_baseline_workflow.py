"""Run, verify, promote, and optionally compare a registry baseline workflow."""

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
from benchmarks.promote_artifact_manifest import promote_artifact_manifest  # noqa: E402
from benchmarks.run_cache_profile_matrix import CacheProfileMatrixConfig, run_matrix  # noqa: E402


@dataclass(frozen=True)
class RegistryBaselineWorkflowConfig:
    """Configuration for one registry-backed baseline workflow."""

    output_dir: Path
    registry_path: Path
    name: str
    version: str
    model: str = "sshleifer/tiny-gpt2"
    dtype: str = "float32"
    layers: Sequence[int] = (-1,)
    batch_sizes: Sequence[int] = (4,)
    hidden_state_captures: Sequence[str] = ("outputs",)
    limit: int | None = None
    manifold_questions: int | None = None
    max_length: int = 64
    eval_reps_cache_shard_size: int = 4
    cached_max_total_ratio: float = 1.10
    cache_only_max_total_ratio: float = 0.35
    python_executable: str = sys.executable
    progress_every: int = 0
    length_bucketed_batches: bool = True
    offline: bool = True
    shared_cache_dir: Path | None = None
    matrix_mode: str = "triplet"
    clean: bool = False
    dry_run: bool = False
    verification_report_path: Path | None = None
    promotion_metadata: Mapping[str, Any] | None = None
    allow_promotion_failures: bool = False
    candidate_profiles: Sequence[tuple[str, Path]] = ()
    baseline_profile_artifact: str = "auto"
    allow_unverified_compare: bool = False
    max_total_ratio: float | None = None
    max_run_total_ratios: Mapping[str, float] | None = None
    max_phase_ratios: Mapping[str, float] | None = None
    min_throughput_ratios: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "registry_path", Path(self.registry_path))
        if self.shared_cache_dir is not None:
            object.__setattr__(self, "shared_cache_dir", Path(self.shared_cache_dir))
        if self.verification_report_path is not None:
            object.__setattr__(self, "verification_report_path", Path(self.verification_report_path))
        object.__setattr__(self, "layers", tuple(int(layer) for layer in self.layers))
        object.__setattr__(self, "batch_sizes", tuple(int(batch_size) for batch_size in self.batch_sizes))
        object.__setattr__(
            self,
            "hidden_state_captures",
            tuple(str(capture) for capture in self.hidden_state_captures),
        )
        object.__setattr__(
            self,
            "candidate_profiles",
            tuple((str(name), Path(path)) for name, path in self.candidate_profiles),
        )


def run_registry_baseline_workflow(config: RegistryBaselineWorkflowConfig) -> dict[str, Any]:
    """Run the matrix baseline workflow and return a JSON-serializable payload."""
    matrix_config = CacheProfileMatrixConfig(
        output_dir=config.output_dir,
        model=config.model,
        dtype=config.dtype,
        layers=config.layers,
        batch_sizes=config.batch_sizes,
        hidden_state_captures=config.hidden_state_captures,
        limit=config.limit,
        manifold_questions=config.manifold_questions,
        max_length=config.max_length,
        eval_reps_cache_shard_size=config.eval_reps_cache_shard_size,
        cached_max_total_ratio=config.cached_max_total_ratio,
        cache_only_max_total_ratio=config.cache_only_max_total_ratio,
        python_executable=config.python_executable,
        progress_every=config.progress_every,
        length_bucketed_batches=config.length_bucketed_batches,
        offline=config.offline,
        shared_cache_dir=config.shared_cache_dir,
        matrix_mode=config.matrix_mode,
    )
    matrix_report = run_matrix(matrix_config, clean=config.clean, dry_run=config.dry_run)
    manifest_path = Path(str(matrix_report["artifact_manifest"]))
    verification_report_path = config.verification_report_path or config.output_dir / "manifest-verification.json"
    promotion = promote_artifact_manifest(
        manifest_path=manifest_path,
        registry_path=config.registry_path,
        name=config.name,
        version=config.version,
        verification_report_path=verification_report_path,
        recursive=True,
        allow_failures=config.allow_promotion_failures,
        metadata=_promotion_metadata(config, matrix_report),
    )
    comparison = None
    if config.candidate_profiles:
        baseline_profile_artifact = _resolve_workflow_baseline_profile_artifact(
            config.baseline_profile_artifact,
            matrix_report,
        )
        comparison = compare_registry_baseline(
            registry_path=config.registry_path,
            baseline_name=config.name,
            baseline_version=config.version,
            baseline_profile_artifact=baseline_profile_artifact,
            candidate_profiles=config.candidate_profiles,
            recursive=True,
            allow_unverified=config.allow_unverified_compare,
            max_total_ratio=config.max_total_ratio,
            max_run_total_ratios=config.max_run_total_ratios,
            max_phase_ratios=config.max_phase_ratios,
            min_throughput_ratios=config.min_throughput_ratios,
            notes=["registry baseline workflow comparison"],
        )
    return {
        "workflow": {
            "output_dir": str(config.output_dir),
            "registry": str(config.registry_path),
            "name": config.name,
            "version": config.version,
            "dry_run": config.dry_run,
            "matrix_mode": config.matrix_mode,
            "baseline_profile_artifact": (
                config.baseline_profile_artifact
                if comparison is None
                else comparison["registry_baseline"]["profile_artifact"]
            ),
        },
        "matrix": matrix_report,
        "promotion": promotion,
        "comparison": comparison,
    }


def _resolve_workflow_baseline_profile_artifact(
    artifact: str,
    matrix_report: Mapping[str, Any],
) -> str:
    if artifact != "auto":
        return artifact
    for cell in matrix_report.get("cells", ()):
        if not isinstance(cell, Mapping):
            continue
        cell_id = str(cell.get("id", "")).strip()
        triplet = cell.get("triplet", {})
        if not cell_id or not isinstance(triplet, Mapping):
            continue
        profiles = triplet.get("profiles", {})
        if isinstance(profiles, Mapping) and "uncached" in profiles:
            return f"cells.{cell_id}.triplet_manifest::profiles.uncached"
    raise ValueError("could not auto-resolve a workflow baseline profile artifact.")


def _promotion_metadata(
    config: RegistryBaselineWorkflowConfig,
    matrix_report: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = {
        "workflow": "run_registry_baseline_workflow",
        "matrix_mode": config.matrix_mode,
        "dry_run": config.dry_run,
        "model": config.model,
        "layers": tuple(config.layers),
        "batch_sizes": tuple(config.batch_sizes),
        "hidden_state_captures": tuple(config.hidden_state_captures),
        "cell_count": len(tuple(matrix_report.get("cells", ()))),
    }
    if config.promotion_metadata is not None:
        metadata.update(dict(config.promotion_metadata))
    return metadata


def _parse_int_list(value: str, *, flag: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError(f"{flag} must not be empty.")
    return values


def _parse_str_list(value: str, *, flag: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError(f"{flag} must not be empty.")
    return values


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


def _config_from_args(args: argparse.Namespace) -> RegistryBaselineWorkflowConfig:
    return RegistryBaselineWorkflowConfig(
        output_dir=Path(args.output_dir),
        registry_path=Path(args.registry),
        name=args.name,
        version=args.version,
        model=args.model,
        dtype=args.dtype,
        layers=_parse_int_list(args.layers, flag="--layers"),
        batch_sizes=_parse_int_list(args.batch_sizes, flag="--batch-sizes"),
        hidden_state_captures=_parse_str_list(
            args.hidden_state_captures,
            flag="--hidden-state-captures",
        ),
        limit=args.limit,
        manifold_questions=args.manifold_questions,
        max_length=args.max_length,
        eval_reps_cache_shard_size=args.eval_reps_cache_shard_size,
        cached_max_total_ratio=args.cached_max_total_ratio,
        cache_only_max_total_ratio=args.cache_only_max_total_ratio,
        python_executable=args.python,
        progress_every=args.progress_every,
        length_bucketed_batches=not args.no_length_bucketed_batches,
        offline=not args.real_truthfulqa,
        shared_cache_dir=Path(args.shared_cache_dir) if args.shared_cache_dir else None,
        matrix_mode=args.matrix_mode,
        clean=bool(args.clean),
        dry_run=bool(args.dry_run),
        verification_report_path=Path(args.verification_report) if args.verification_report else None,
        promotion_metadata=_parse_metadata(args.metadata or ()),
        allow_promotion_failures=bool(args.allow_promotion_failures),
        candidate_profiles=tuple(_parse_named_path(value) for value in args.candidate_profile),
        baseline_profile_artifact=args.baseline_profile_artifact,
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
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = run_registry_baseline_workflow(_config_from_args(args))
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote registry baseline workflow report to {output_path}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    gate = None if payload["comparison"] is None else payload["comparison"]["comparison"].get("regression_gate")
    if args.fail_on_regression and gate is not None and not gate.get("passed", False):
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a verified registry baseline workflow")
    parser.add_argument("--output-dir", required=True,
                        help="directory for matrix outputs, manifests, and reports")
    parser.add_argument("--registry", required=True, help="local ArtifactRegistry JSON path")
    parser.add_argument("--name", required=True, help="registry benchmark manifest name")
    parser.add_argument("--version", required=True, help="registry benchmark manifest version")
    parser.add_argument("--verification-report", default=None, help="path for the manifest verification report")
    parser.add_argument("--metadata", action="append", default=[], help="extra promotion metadata as key=value")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2",
                        help="model id passed to eval_truthfulqa.py by the matrix runner")
    parser.add_argument("--dtype", default="float32", help="dtype passed through to eval_truthfulqa.py")
    parser.add_argument("--layers", default="-1", help="comma-list of layer indexes")
    parser.add_argument("--batch-sizes", default="4", help="comma-list of batch sizes")
    parser.add_argument("--hidden-state-captures", default="outputs",
                        help="comma-list of hidden-state capture modes")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--manifold-questions", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--eval-reps-cache-shard-size", type=int, default=4)
    parser.add_argument("--cached-max-total-ratio", type=float, default=1.10)
    parser.add_argument("--cache-only-max-total-ratio", type=float, default=0.35)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used for eval_truthfulqa.py subprocesses")
    parser.add_argument("--no-length-bucketed-batches", action="store_true")
    parser.add_argument("--real-truthfulqa", action="store_true",
                        help="load the configured model and TruthfulQA dataset instead of the offline fixture")
    parser.add_argument("--shared-cache-dir", default=None,
                        help="shared cache directory passed to the matrix runner")
    parser.add_argument("--matrix-mode", default="triplet", choices=["triplet", "rescore"])
    parser.add_argument("--clean", action="store_true", help="remove cell output directories before running")
    parser.add_argument("--dry-run", action="store_true", help="write commands/manifests without loading a model")
    parser.add_argument("--allow-promotion-failures", action="store_true",
                        help="register the manifest even if verification fails")
    parser.add_argument("--candidate-profile", action="append", default=[],
                        help="candidate profile JSON path, optionally named as name=path; repeatable")
    parser.add_argument("--baseline-profile-artifact", default="auto",
                        help="baseline profile artifact reference; 'auto' uses the first uncached matrix cell")
    parser.add_argument("--allow-unverified-compare", action="store_true",
                        help="compare even if the registered baseline verification fails")
    parser.add_argument("--max-total-ratio", type=float, default=None)
    parser.add_argument("--max-run-total-ratio", action="append", default=[])
    parser.add_argument("--max-phase-ratio", action="append", default=[])
    parser.add_argument("--min-throughput-ratio", action="append", default=[])
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="exit non-zero when the optional comparison gate fails")
    parser.add_argument("--json", default=None, help="optional path to write the workflow JSON report")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
