"""Run direct self-consistency signal fusion from sampled responses.

This workflow converts aligned sampled generations into standard score-dump
columns with ``build_selfcheck_signal_score_dump.py``, then evaluates the
resulting selfcheck signals alongside existing geometry scores. It does not
load models or call the network.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.build_selfcheck_signal_score_dump import (  # noqa: E402
    DEFAULT_SELFCHECK_SIGNALS,
)
from benchmarks.build_selfcheck_signal_score_dump import (  # noqa: E402
    build_report as build_selfcheck_signal_score_dump_report,
)
from benchmarks.eval_score_ensemble import (  # noqa: E402
    GEOMETRY_FUSION_METHODS,
    METHODS,
    build_ensemble_report,
    build_geometry_fusion_artifact_from_score_dump,
)
from eigentruth.registry import ArtifactVerificationContext, build_artifact_manifest  # noqa: E402

DEFAULT_FUSION_SIGNALS = (
    "truth_proj",
    "subspace_resid",
    "eigenscore",
    "selfcheck_support_rate",
    "selfcheck_refute_rate",
    "selfcheck_disagreement",
    "selfcheck_not_applicable",
    "selfcheck_best_overlap",
)
DEFAULT_GEOMETRY_SIGNALS = ("truth_proj", "subspace_resid", "eigenscore")
DEFAULT_UNCERTAINTY_SIGNALS = (
    "selfcheck_support_rate",
    "selfcheck_refute_rate",
    "selfcheck_disagreement",
    "selfcheck_not_applicable",
    "selfcheck_best_overlap",
)


@dataclass(frozen=True)
class SelfcheckSignalFusionWorkflowConfig:
    """Configuration for the direct selfcheck-signal fusion workflow."""

    score_dumps: Sequence[tuple[str, Path]]
    output_dir: Path
    sample_paths: Sequence[Path] = ()
    keep_signals: Sequence[str] | None = None
    selfcheck_signals: Sequence[str] = DEFAULT_SELFCHECK_SIGNALS
    fusion_signals: Sequence[str] = DEFAULT_FUSION_SIGNALS
    methods: Sequence[str] = METHODS
    geometry_signals: Sequence[str] = DEFAULT_GEOMETRY_SIGNALS
    uncertainty_signals: Sequence[str] = DEFAULT_UNCERTAINTY_SIGNALS
    geometry_method: str = "mean_rank"
    uncertainty_method: str = "mean_rank"
    geometry_fusion_methods: Sequence[str] = GEOMETRY_FUSION_METHODS
    alphas: Sequence[float] = (0.10,)
    repeats: int = 20
    seed: int = 0
    best_alpha: float = 0.10
    min_samples: int = 2
    min_overlap: float = 0.65
    support_threshold: float = 0.60
    refute_threshold: float = 0.50
    early_stop: bool = False
    max_samples: int | None = None
    compact_json: bool = False
    verify_manifest: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "score_dumps",
            tuple((str(name), Path(path)) for name, path in self.score_dumps),
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "sample_paths", tuple(Path(path) for path in self.sample_paths))
        for attr in (
            "selfcheck_signals",
            "fusion_signals",
            "methods",
            "geometry_signals",
            "uncertainty_signals",
            "geometry_fusion_methods",
            "alphas",
        ):
            object.__setattr__(self, attr, tuple(getattr(self, attr)))
        if self.keep_signals is not None:
            object.__setattr__(self, "keep_signals", tuple(self.keep_signals))
        if not self.score_dumps:
            raise ValueError("score_dumps must contain at least one score dump.")
        if len({name for name, _ in self.score_dumps}) != len(self.score_dumps):
            raise ValueError("score dump names must be unique.")
        if int(self.repeats) < 1:
            raise ValueError("repeats must be >= 1.")
        if int(self.min_samples) < 1:
            raise ValueError("min_samples must be >= 1.")
        if self.max_samples is not None and int(self.max_samples) < 1:
            raise ValueError("max_samples must be >= 1 when set.")
        for name in ("min_overlap", "support_threshold", "refute_threshold"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
            object.__setattr__(self, name, value)
        if any(not (0.0 < float(alpha) < 1.0) for alpha in self.alphas):
            raise ValueError("alphas must be in (0, 1).")
        if not (0.0 < float(self.best_alpha) < 1.0):
            raise ValueError("best_alpha must be in (0, 1).")
        if not self.selfcheck_signals:
            raise ValueError("selfcheck_signals must contain at least one signal.")
        if len(set(self.selfcheck_signals)) != len(self.selfcheck_signals):
            raise ValueError("selfcheck_signals must contain unique values.")
        if not self.fusion_signals:
            raise ValueError("fusion_signals must contain at least one signal.")
        if bool(self.geometry_signals) != bool(self.uncertainty_signals):
            raise ValueError("geometry_signals and uncertainty_signals must be provided together.")

    @property
    def score_ensemble_report_path(self) -> Path:
        return self.output_dir / "score-ensemble-report.json"

    @property
    def artifact_manifest_path(self) -> Path:
        return self.output_dir / "artifact-manifest.json"

    @property
    def manifest_verification_path(self) -> Path:
        return self.output_dir / "manifest-verification.json"

    @property
    def workflow_report_path(self) -> Path:
        return self.output_dir / "selfcheck-signal-fusion-workflow.json"


def run_selfcheck_signal_fusion_workflow(
    config: SelfcheckSignalFusionWorkflowConfig,
) -> dict[str, Any]:
    """Run selfcheck signal construction and calibrated score fusion."""
    started = time.perf_counter()
    profile: dict[str, float] = {}
    config.output_dir.mkdir(parents=True, exist_ok=True)

    enhanced_score_dumps: list[tuple[str, Path]] = []
    enhanced_reports: dict[str, str] = {}
    enhanced_summaries: dict[str, Any] = {}
    with _profile_phase(profile, "build_selfcheck_signal_score_dumps"):
        for run_name, source_path in config.score_dumps:
            output_path = config.output_dir / f"{run_name}-selfcheck-scores.manifest.json"
            report_path = config.output_dir / f"{run_name}-selfcheck-score-report.json"
            report = build_selfcheck_signal_score_dump_report(
                input_scores=source_path,
                sample_paths=config.sample_paths,
                output=output_path,
                output_format="jsonl",
                keep_signals=config.keep_signals,
                selfcheck_signals=config.selfcheck_signals,
                min_samples=int(config.min_samples),
                min_overlap=float(config.min_overlap),
                support_threshold=float(config.support_threshold),
                refute_threshold=float(config.refute_threshold),
                early_stop=bool(config.early_stop),
                max_samples=config.max_samples,
            )
            _write_json(report_path, report, compact=config.compact_json)
            enhanced_score_dumps.append((run_name, output_path))
            enhanced_reports[run_name] = str(report_path)
            enhanced_summaries[run_name] = {
                "n_total": report.get("n_total"),
                "selfcheck_signals": report.get("selfcheck_signals"),
                "summary": report.get("summary"),
            }

    with _profile_phase(profile, "build_score_ensemble_report"):
        score_ensemble_report = build_ensemble_report(
            enhanced_score_dumps,
            signals=config.fusion_signals,
            methods=config.methods,
            geometry_signals=config.geometry_signals,
            uncertainty_signals=config.uncertainty_signals,
            geometry_method=config.geometry_method,
            uncertainty_method=config.uncertainty_method,
            geometry_fusion_methods=config.geometry_fusion_methods,
            alphas=tuple(float(alpha) for alpha in config.alphas),
            repeats=int(config.repeats),
            seed=int(config.seed),
            best_alpha=float(config.best_alpha),
        )
    with _profile_phase(profile, "write_score_ensemble_report"):
        _write_json(config.score_ensemble_report_path, score_ensemble_report, compact=config.compact_json)

    geometry_artifacts: dict[str, str] = {}
    if config.geometry_signals and config.uncertainty_signals:
        with _profile_phase(profile, "build_geometry_fusion_artifacts"):
            for run_name, enhanced_path in enhanced_score_dumps:
                run_payload = _run_payload_by_name(score_ensemble_report, run_name)
                best_geometry = run_payload.get("best_geometry_fusion_at_alpha")
                if best_geometry is None:
                    continue
                artifact_path = config.output_dir / f"{run_name}-selfcheck-geometry-fusion-artifact.json"
                artifact = build_geometry_fusion_artifact_from_score_dump(
                    (run_name, enhanced_path),
                    geometry_signals=config.geometry_signals,
                    uncertainty_signals=config.uncertainty_signals,
                    geometry_method=config.geometry_method,
                    uncertainty_method=config.uncertainty_method,
                    fusion_method=str(best_geometry["name"]),
                    alpha=float(config.best_alpha),
                    cache={},
                )
                artifact.save_json(artifact_path)
                geometry_artifacts[run_name] = str(artifact_path)

    manifest_metadata = _manifest_metadata(
        config,
        enhanced_summaries=enhanced_summaries,
        score_ensemble_report=score_ensemble_report,
        geometry_artifacts=geometry_artifacts,
        profile=profile,
        total_seconds=time.perf_counter() - started,
    )
    with _profile_phase(profile, "write_artifact_manifest"):
        manifest = _write_artifact_manifest(
            config,
            enhanced_score_dumps=enhanced_score_dumps,
            enhanced_reports=enhanced_reports,
            geometry_artifacts=geometry_artifacts,
            metadata=manifest_metadata,
        )

    manifest_verification = None
    if config.verify_manifest:
        with _profile_phase(profile, "verify_artifact_manifest"):
            context = ArtifactVerificationContext()
            manifest_verification = context.load_and_verify_artifact_manifest(
                config.artifact_manifest_path,
                root=config.artifact_manifest_path.parent,
            ).to_dict()
            _write_json(config.manifest_verification_path, manifest_verification, compact=False)

    profile["total_seconds"] = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "workflow": "selfcheck_signal_fusion_workflow",
        "config": _config_payload(config),
        "enhanced_score_dumps": {name: str(path) for name, path in enhanced_score_dumps},
        "enhanced_score_reports": enhanced_reports,
        "score_ensemble_report_path": str(config.score_ensemble_report_path),
        "geometry_fusion_artifacts": geometry_artifacts,
        "artifact_manifest_path": str(config.artifact_manifest_path),
        "manifest_verification_path": (
            None if manifest_verification is None else str(config.manifest_verification_path)
        ),
        "selfcheck_summary": enhanced_summaries,
        "fusion_summary": _fusion_summary(score_ensemble_report),
        "manifest_summary": manifest.get("summary"),
        "manifest_verification": manifest_verification,
        "profile": dict(profile),
    }
    _write_json(config.workflow_report_path, payload, compact=config.compact_json)
    return payload


def _write_artifact_manifest(
    config: SelfcheckSignalFusionWorkflowConfig,
    *,
    enhanced_score_dumps: Sequence[tuple[str, Path]],
    enhanced_reports: Mapping[str, str],
    geometry_artifacts: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "score_ensemble_report": config.score_ensemble_report_path,
    }
    for idx, path in enumerate(config.sample_paths, start=1):
        artifacts[f"selfcheck_samples.{idx}.{path.stem}"] = path
    for run_name, path in config.score_dumps:
        artifacts[f"source_scores.{run_name}"] = path
    for run_name, path in enhanced_score_dumps:
        artifacts[f"enhanced_scores.{run_name}"] = path
        artifacts[f"enhanced_records.{run_name}"] = path.with_suffix(".records.jsonl")
    for run_name, path in enhanced_reports.items():
        artifacts[f"enhanced_report.{run_name}"] = path
    for run_name, path in geometry_artifacts.items():
        artifacts[f"geometry_fusion_artifact.{run_name}"] = path

    manifest = build_artifact_manifest(
        artifacts,
        root=config.artifact_manifest_path.parent,
        metadata=metadata,
    )
    _write_json(config.artifact_manifest_path, manifest, compact=False)
    return manifest


def _manifest_metadata(
    config: SelfcheckSignalFusionWorkflowConfig,
    *,
    enhanced_summaries: Mapping[str, Any],
    score_ensemble_report: Mapping[str, Any],
    geometry_artifacts: Mapping[str, str],
    profile: Mapping[str, float],
    total_seconds: float,
) -> dict[str, Any]:
    return {
        "runner": "run_selfcheck_signal_fusion_workflow",
        "workflow": "selfcheck_signal_fusion_workflow",
        "score_names": [name for name, _ in config.score_dumps],
        "sample_paths": [str(path) for path in config.sample_paths],
        "keep_signals": None if config.keep_signals is None else list(config.keep_signals),
        "selfcheck_signals": list(config.selfcheck_signals),
        "fusion_signals": list(config.fusion_signals),
        "geometry_signals": list(config.geometry_signals),
        "uncertainty_signals": list(config.uncertainty_signals),
        "best_alpha": float(config.best_alpha),
        "selfcheck_config": {
            "min_samples": int(config.min_samples),
            "min_overlap": float(config.min_overlap),
            "support_threshold": float(config.support_threshold),
            "refute_threshold": float(config.refute_threshold),
            "early_stop": bool(config.early_stop),
            "max_samples": config.max_samples,
        },
        "selfcheck_summary": dict(enhanced_summaries),
        "fusion_summary": _fusion_summary(score_ensemble_report),
        "geometry_artifact_count": len(geometry_artifacts),
        "uses_external_sample_paths": bool(config.sample_paths),
        "profile": dict(profile),
        "total_seconds": float(total_seconds),
    }


def _fusion_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    runs = []
    for run in report.get("runs", ()):
        if not isinstance(run, Mapping):
            continue
        runs.append({
            "name": run.get("name"),
            "best_single_at_alpha": run.get("best_single_at_alpha"),
            "best_ensemble_at_alpha": run.get("best_ensemble_at_alpha"),
            "best_geometry_fusion_at_alpha": run.get("best_geometry_fusion_at_alpha"),
        })
    return {"runs": runs}


def _run_payload_by_name(report: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for run in report.get("runs", ()):
        if isinstance(run, Mapping) and str(run.get("name")) == str(name):
            return run
    raise ValueError(f"score ensemble report has no run named {name!r}.")


def _config_payload(config: SelfcheckSignalFusionWorkflowConfig) -> dict[str, Any]:
    return {
        "score_dumps": {name: str(path) for name, path in config.score_dumps},
        "sample_paths": [str(path) for path in config.sample_paths],
        "keep_signals": None if config.keep_signals is None else list(config.keep_signals),
        "selfcheck_signals": list(config.selfcheck_signals),
        "fusion_signals": list(config.fusion_signals),
        "methods": list(config.methods),
        "geometry_signals": list(config.geometry_signals),
        "uncertainty_signals": list(config.uncertainty_signals),
        "geometry_method": config.geometry_method,
        "uncertainty_method": config.uncertainty_method,
        "geometry_fusion_methods": list(config.geometry_fusion_methods),
        "alphas": [float(alpha) for alpha in config.alphas],
        "repeats": int(config.repeats),
        "seed": int(config.seed),
        "best_alpha": float(config.best_alpha),
        "min_samples": int(config.min_samples),
        "min_overlap": float(config.min_overlap),
        "support_threshold": float(config.support_threshold),
        "refute_threshold": float(config.refute_threshold),
        "early_stop": bool(config.early_stop),
        "max_samples": config.max_samples,
        "verify_manifest": bool(config.verify_manifest),
    }


@contextmanager
def _profile_phase(profile: MutableMapping[str, float], name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        profile[name] = profile.get(name, 0.0) + (time.perf_counter() - started)


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("scores name cannot be empty.")
    return name, Path(path)


def _parse_csv(value: str | None, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError(f"{name} must contain at least one value.")
    if len(set(parts)) != len(parts):
        raise ValueError(f"{name} must contain unique values.")
    return parts


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from CLI-style arguments."""
    config = SelfcheckSignalFusionWorkflowConfig(
        score_dumps=tuple(_parse_named_path(value) for value in args.scores),
        output_dir=Path(args.output_dir),
        sample_paths=tuple(Path(path) for path in args.samples or ()),
        keep_signals=_parse_csv(args.keep_signals, name="keep_signals"),
        selfcheck_signals=_parse_csv(args.selfcheck_signals, name="selfcheck_signals") or DEFAULT_SELFCHECK_SIGNALS,
        fusion_signals=_parse_csv(args.fusion_signals, name="fusion_signals") or DEFAULT_FUSION_SIGNALS,
        methods=_parse_csv(args.methods, name="methods") or METHODS,
        geometry_signals=_parse_csv(args.geometry_signals, name="geometry_signals") or DEFAULT_GEOMETRY_SIGNALS,
        uncertainty_signals=(
            _parse_csv(args.uncertainty_signals, name="uncertainty_signals")
            or DEFAULT_UNCERTAINTY_SIGNALS
        ),
        geometry_method=args.geometry_method,
        uncertainty_method=args.uncertainty_method,
        geometry_fusion_methods=(
            _parse_csv(args.geometry_fusion_methods, name="geometry_fusion_methods")
            or GEOMETRY_FUSION_METHODS
        ),
        alphas=tuple(float(value) for value in (_parse_csv(args.alphas, name="alphas") or ())),
        repeats=args.repeats,
        seed=args.seed,
        best_alpha=args.best_alpha,
        min_samples=args.min_samples,
        min_overlap=args.min_overlap,
        support_threshold=args.support_threshold,
        refute_threshold=args.refute_threshold,
        early_stop=args.early_stop,
        max_samples=args.max_samples,
        compact_json=args.compact_json,
        verify_manifest=not bool(args.no_verify_manifest),
    )
    payload = run_selfcheck_signal_fusion_workflow(config)
    print(
        f"selfcheck_signal_fusion_workflow_ok runs={len(config.score_dumps)} "
        f"manifest={payload['artifact_manifest_path']}"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct selfcheck-signal fusion over sampled responses")
    parser.add_argument("--scores", action="append", required=True,
                        help="score dump path, optionally named as name=path; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", action="append", default=None,
                        help="sampled generation JSON/JSONL for selfcheck; repeatable")
    parser.add_argument("--keep-signals", default=None)
    parser.add_argument("--selfcheck-signals", default=",".join(DEFAULT_SELFCHECK_SIGNALS))
    parser.add_argument("--fusion-signals", default=",".join(DEFAULT_FUSION_SIGNALS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--geometry-signals", default=",".join(DEFAULT_GEOMETRY_SIGNALS))
    parser.add_argument("--uncertainty-signals", default=",".join(DEFAULT_UNCERTAINTY_SIGNALS))
    parser.add_argument("--geometry-method", default="mean_rank")
    parser.add_argument("--uncertainty-method", default="mean_rank")
    parser.add_argument("--geometry-fusion-methods", default=",".join(GEOMETRY_FUSION_METHODS))
    parser.add_argument("--alphas", default="0.1")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--best-alpha", type=float, default=0.10)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--min-overlap", type=float, default=0.65)
    parser.add_argument("--support-threshold", type=float, default=0.60)
    parser.add_argument("--refute-threshold", type=float, default=0.50)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--no-verify-manifest", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
