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
from benchmarks.plan_selfcheck_sample_collection import (  # noqa: E402
    SelfcheckSampleCollectionPlanConfig,
    write_selfcheck_sample_collection_plan,
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
    sample_quality_min_coverage: float = 0.50
    sample_quality_max_not_applicable_rate: float = 0.50
    sample_quality_min_average_samples_per_record: float = 1.0
    sample_quality_min_records_meeting_min_samples: int | None = None
    sample_quality_min_best_overlap_mean: float = 0.0
    write_sample_collection_plans: bool = True
    sample_collection_target_samples_per_record: int | None = None
    sample_collection_plan_max_records: int | None = None
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
        if self.sample_collection_target_samples_per_record is not None:
            target = int(self.sample_collection_target_samples_per_record)
            if target < int(self.min_samples):
                raise ValueError("sample_collection_target_samples_per_record must be >= min_samples.")
            object.__setattr__(self, "sample_collection_target_samples_per_record", target)
        if self.sample_collection_plan_max_records is not None:
            max_records = int(self.sample_collection_plan_max_records)
            if max_records < 0:
                raise ValueError("sample_collection_plan_max_records must be >= 0 when set.")
            object.__setattr__(self, "sample_collection_plan_max_records", max_records)
        if self.sample_quality_min_records_meeting_min_samples is not None:
            min_records = int(self.sample_quality_min_records_meeting_min_samples)
            if min_records < 0:
                raise ValueError("sample_quality_min_records_meeting_min_samples must be >= 0 when set.")
            object.__setattr__(self, "sample_quality_min_records_meeting_min_samples", min_records)
        for name in ("min_overlap", "support_threshold", "refute_threshold"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
            object.__setattr__(self, name, value)
        for name in (
            "sample_quality_min_coverage",
            "sample_quality_max_not_applicable_rate",
            "sample_quality_min_best_overlap_mean",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
            object.__setattr__(self, name, value)
        min_average_samples = float(self.sample_quality_min_average_samples_per_record)
        if min_average_samples < 0.0:
            raise ValueError("sample_quality_min_average_samples_per_record must be >= 0.")
        object.__setattr__(
            self,
            "sample_quality_min_average_samples_per_record",
            min_average_samples,
        )
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
    def sample_quality_report_path(self) -> Path:
        return self.output_dir / "sample-quality-report.json"

    @property
    def artifact_manifest_path(self) -> Path:
        return self.output_dir / "artifact-manifest.json"

    @property
    def manifest_verification_path(self) -> Path:
        return self.output_dir / "manifest-verification.json"

    @property
    def workflow_report_path(self) -> Path:
        return self.output_dir / "selfcheck-signal-fusion-workflow.json"

    def sample_collection_plan_path(self, run_name: str) -> Path:
        return self.output_dir / f"{run_name}-selfcheck-sample-collection-plan.json"


def run_selfcheck_signal_fusion_workflow(
    config: SelfcheckSignalFusionWorkflowConfig,
) -> dict[str, Any]:
    """Run selfcheck signal construction and calibrated score fusion."""
    started = time.perf_counter()
    profile: dict[str, float] = {}
    config.output_dir.mkdir(parents=True, exist_ok=True)

    sample_collection_plans: dict[str, str] = {}
    sample_collection_summaries: dict[str, Any] = {}
    if config.write_sample_collection_plans:
        with _profile_phase(profile, "write_sample_collection_plans"):
            for run_name, source_path in config.score_dumps:
                plan_path = config.sample_collection_plan_path(run_name)
                plan = write_selfcheck_sample_collection_plan(
                    SelfcheckSampleCollectionPlanConfig(
                        scores=source_path,
                        output=plan_path,
                        sample_paths=config.sample_paths,
                        min_samples=int(config.min_samples),
                        target_samples_per_record=config.sample_collection_target_samples_per_record,
                        max_records=config.sample_collection_plan_max_records,
                        include_ready_records=False,
                        sample_quality_min_coverage=float(config.sample_quality_min_coverage),
                        sample_quality_min_average_samples_per_record=float(
                            config.sample_quality_min_average_samples_per_record
                        ),
                        sample_quality_min_records_meeting_min_samples=(
                            config.sample_quality_min_records_meeting_min_samples
                        ),
                        compact_json=config.compact_json,
                    )
                )
                sample_collection_plans[run_name] = str(plan_path)
                sample_collection_summaries[run_name] = _sample_collection_summary(plan)

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
                "fixture_summary": report.get("fixture_summary"),
                "summary": report.get("summary"),
            }

    with _profile_phase(profile, "write_sample_quality_report"):
        sample_quality_report = _sample_quality_report(config, enhanced_summaries)
        _write_json(config.sample_quality_report_path, sample_quality_report, compact=config.compact_json)

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
        sample_collection_summaries=sample_collection_summaries,
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
            sample_collection_plans=sample_collection_plans,
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
        "sample_collection_plans": sample_collection_plans,
        "sample_collection_summary": sample_collection_summaries,
        "sample_quality_report_path": str(config.sample_quality_report_path),
        "sample_quality": sample_quality_report,
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
    sample_collection_plans: Mapping[str, str],
    geometry_artifacts: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "sample_quality_report": config.sample_quality_report_path,
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
    for run_name, path in sample_collection_plans.items():
        artifacts[f"sample_collection_plan.{run_name}"] = path
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
    sample_collection_summaries: Mapping[str, Any],
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
        "sample_quality_gate": _sample_quality_gate_config(config),
        "sample_collection_plans_enabled": bool(config.write_sample_collection_plans),
        "sample_collection_target_samples_per_record": (
            config.sample_collection_target_samples_per_record
        ),
        "sample_collection_plan_max_records": config.sample_collection_plan_max_records,
        "sample_collection_summary": dict(sample_collection_summaries),
        "sample_quality": _sample_quality_report(config, enhanced_summaries),
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
        "sample_quality_gate": _sample_quality_gate_config(config),
        "write_sample_collection_plans": bool(config.write_sample_collection_plans),
        "sample_collection_target_samples_per_record": (
            config.sample_collection_target_samples_per_record
        ),
        "sample_collection_plan_max_records": config.sample_collection_plan_max_records,
        "verify_manifest": bool(config.verify_manifest),
    }


def _sample_collection_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(plan.get("summary"))
    quality = _mapping(plan.get("sample_quality_gate_projection"))
    collection_plan = _mapping(plan.get("collection_plan"))
    return {
        "status": plan.get("status"),
        "records_to_collect_count": collection_plan.get("records_to_collect_count"),
        "recommended_min_new_samples": collection_plan.get("recommended_min_new_samples"),
        "n_records": summary.get("n_records"),
        "records_meeting_min_samples": summary.get("records_meeting_min_samples"),
        "records_below_min_samples": summary.get("records_below_min_samples"),
        "records_below_target_samples": summary.get("records_below_target_samples"),
        "sample_deficit_total": summary.get("sample_deficit_total"),
        "coverage": summary.get("min_sample_coverage"),
        "average_samples_per_record": summary.get("average_samples_per_record"),
        "sample_quality_gate_status": quality.get("status"),
        "sample_quality_gate_passed": quality.get("passed"),
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


def _sample_quality_gate_config(config: SelfcheckSignalFusionWorkflowConfig) -> dict[str, Any]:
    return {
        "min_coverage": float(config.sample_quality_min_coverage),
        "max_not_applicable_rate": float(config.sample_quality_max_not_applicable_rate),
        "min_average_samples_per_record": float(config.sample_quality_min_average_samples_per_record),
        "min_records_meeting_min_samples": config.sample_quality_min_records_meeting_min_samples,
        "min_best_overlap_mean": float(config.sample_quality_min_best_overlap_mean),
    }


def _sample_quality_report(
    config: SelfcheckSignalFusionWorkflowConfig,
    enhanced_summaries: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = _sample_quality_gate_config(config)
    runs = {}
    failed_runs = []
    for run_name, summary in enhanced_summaries.items():
        run_report = _sample_quality_run_report(str(run_name), _mapping(summary), thresholds)
        runs[str(run_name)] = run_report
        if not run_report["passed"]:
            failed_runs.append(str(run_name))
    passed = not failed_runs
    return {
        "schema_version": 1,
        "report_type": "selfcheck_sample_quality_gate",
        "status": "pass" if passed else "fail",
        "passed": passed,
        "thresholds": thresholds,
        "runs": runs,
        "failed_runs": failed_runs,
        "recommendation": (
            "selfcheck signals have enough sample coverage for calibrated replay"
            if passed else
            "collect better aligned samples before promoting selfcheck signals"
        ),
    }


def _sample_quality_run_report(
    run_name: str,
    summary: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    fixture = _mapping(summary.get("fixture_summary"))
    signals = _mapping(summary.get("summary"))
    n_total = int(_number(summary.get("n_total"), default=fixture.get("n_records", 0.0)))
    records_with_samples = int(_number(fixture.get("records_with_samples"), default=0.0))
    records_meeting_min = int(_number(fixture.get("records_meeting_min_samples"), default=0.0))
    total_samples = int(_number(fixture.get("total_samples"), default=0.0))
    average_samples = _number(fixture.get("average_samples_per_record"), default=0.0)
    coverage = (float(records_meeting_min) / float(n_total)) if n_total else 0.0
    sample_presence_rate = (float(records_with_samples) / float(n_total)) if n_total else 0.0
    not_applicable_rate = _signal_mean(signals, "selfcheck_not_applicable")
    best_overlap_mean = _signal_mean(signals, "selfcheck_best_overlap")
    sample_count_mean = _signal_mean(signals, "selfcheck_sample_count")

    failures = []
    if coverage < float(thresholds["min_coverage"]):
        failures.append({
            "metric": "coverage",
            "value": coverage,
            "threshold": float(thresholds["min_coverage"]),
            "rule": "value >= threshold",
        })
    if average_samples < float(thresholds["min_average_samples_per_record"]):
        failures.append({
            "metric": "average_samples_per_record",
            "value": average_samples,
            "threshold": float(thresholds["min_average_samples_per_record"]),
            "rule": "value >= threshold",
        })
    min_records = thresholds.get("min_records_meeting_min_samples")
    if min_records is not None and records_meeting_min < int(min_records):
        failures.append({
            "metric": "records_meeting_min_samples",
            "value": records_meeting_min,
            "threshold": int(min_records),
            "rule": "value >= threshold",
        })
    if not_applicable_rate is not None and not_applicable_rate > float(thresholds["max_not_applicable_rate"]):
        failures.append({
            "metric": "not_applicable_rate",
            "value": not_applicable_rate,
            "threshold": float(thresholds["max_not_applicable_rate"]),
            "rule": "value <= threshold",
        })
    if best_overlap_mean is not None and best_overlap_mean < float(thresholds["min_best_overlap_mean"]):
        failures.append({
            "metric": "best_overlap_mean",
            "value": best_overlap_mean,
            "threshold": float(thresholds["min_best_overlap_mean"]),
            "rule": "value >= threshold",
        })

    return {
        "name": run_name,
        "passed": not failures,
        "status": "pass" if not failures else "fail",
        "n_total": n_total,
        "records_with_samples": records_with_samples,
        "records_meeting_min_samples": records_meeting_min,
        "total_samples": total_samples,
        "coverage": coverage,
        "sample_presence_rate": sample_presence_rate,
        "average_samples_per_record": average_samples,
        "not_applicable_rate": not_applicable_rate,
        "best_overlap_mean": best_overlap_mean,
        "sample_count_mean": sample_count_mean,
        "failures": failures,
    }


def _signal_mean(signals: Mapping[str, Any], name: str) -> float | None:
    payload = signals.get(name)
    if not isinstance(payload, Mapping) or payload.get("mean") is None:
        return None
    return _number(payload.get("mean"), default=0.0)


def _number(value: Any, *, default: Any) -> float:
    if isinstance(value, bool) or value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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
        sample_quality_min_coverage=args.sample_quality_min_coverage,
        sample_quality_max_not_applicable_rate=args.sample_quality_max_not_applicable_rate,
        sample_quality_min_average_samples_per_record=args.sample_quality_min_average_samples_per_record,
        sample_quality_min_records_meeting_min_samples=args.sample_quality_min_records_meeting_min_samples,
        sample_quality_min_best_overlap_mean=args.sample_quality_min_best_overlap_mean,
        write_sample_collection_plans=not bool(args.no_sample_collection_plan),
        sample_collection_target_samples_per_record=args.sample_collection_target_samples_per_record,
        sample_collection_plan_max_records=args.sample_collection_plan_max_records,
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
    parser.add_argument("--sample-quality-min-coverage", type=float, default=0.50)
    parser.add_argument("--sample-quality-max-not-applicable-rate", type=float, default=0.50)
    parser.add_argument("--sample-quality-min-average-samples-per-record", type=float, default=1.0)
    parser.add_argument("--sample-quality-min-records-meeting-min-samples", type=int, default=None)
    parser.add_argument("--sample-quality-min-best-overlap-mean", type=float, default=0.0)
    parser.add_argument("--no-sample-collection-plan", action="store_true",
                        help="do not write per-run selfcheck sample collection preflight plans")
    parser.add_argument("--sample-collection-target-samples-per-record", type=int, default=None,
                        help="desired samples per record in preflight plans; defaults to --min-samples")
    parser.add_argument("--sample-collection-plan-max-records", type=int, default=None,
                        help="maximum missing-record details per plan; 0 keeps only plan summaries")
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--no-verify-manifest", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
