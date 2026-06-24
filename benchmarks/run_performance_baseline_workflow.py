"""Build a registry-ready performance baseline bundle.

This workflow is a thin orchestration layer over the existing performance
helpers. It runs or reuses cache-profile evidence, optional worker-count and
INSIDE sampling evidence, then writes one runtime recommendation and artifact
manifest that can be registered as the product performance baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import planned_artifact_manifest_summary  # noqa: E402
from benchmarks.recommend_runtime_config import (  # noqa: E402
    INSIDE_TRIGGER_BUDGET_POLICIES,
    build_runtime_recommendation,
)
from benchmarks.run_cache_profile_matrix import (  # noqa: E402
    MATRIX_MODES,
    CacheProfileMatrixConfig,
    _parse_int_list,
    _parse_max_batch_token_budgets,
    _parse_prefix_kv_cache_modes,
    _parse_str_list,
    run_matrix,
)
from benchmarks.run_cache_worker_sweep import CacheWorkerSweepConfig, run_worker_sweep  # noqa: E402
from benchmarks.run_inside_sampling_profile import (  # noqa: E402
    INSIDE_PROFILE_RUN_NAMES,
    InsideSamplingProfileConfig,
    _parse_run_names,
    run_inside_sampling_profile,
)
from benchmarks.run_inside_trigger_budget_sweep import (  # noqa: E402
    InsideTriggerBudgetSweepConfig,
    TriggerBudgetSpec,
    run_inside_trigger_budget_sweep,
)
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class PerformanceBaselineWorkflowConfig:
    """Configuration for a product performance baseline workflow."""

    output_dir: Path
    report_path: Path | None = None
    registry_path: Path | None = None
    name: str | None = None
    version: str | None = None
    model: str = "sshleifer/tiny-gpt2"
    dtype: str = "float32"
    layers: Sequence[int] = (-1,)
    batch_sizes: Sequence[int] = (4,)
    hidden_state_captures: Sequence[str] = ("outputs",)
    limit: int | None = None
    manifold_questions: int | None = None
    max_length: int = 64
    max_batch_tokens: int = 0
    max_batch_token_budgets: Sequence[int] | None = None
    prefix_kv_cache: bool = False
    prefix_kv_cache_modes: Sequence[bool] | None = None
    eval_reps_cache_shard_size: int = 4
    cached_max_total_ratio: float = 1.10
    cache_only_max_total_ratio: float = 0.35
    python_executable: str = sys.executable
    progress_every: int = 0
    length_bucketed_batches: bool = True
    offline: bool = True
    shared_cache_dir: Path | None = None
    matrix_mode: str = "triplet"
    max_workers: int = 1
    matrix_report_path: Path | None = None
    worker_sweep_report_path: Path | None = None
    run_worker_sweep: bool = False
    worker_counts: Sequence[int] = (1, 2)
    inside_sampling_report_path: Path | None = None
    run_inside_sampling: bool = False
    inside_trigger_budget_sweep_report_path: Path | None = None
    run_inside_trigger_budget_sweep: bool = False
    inside_trigger_budget_policy: str = "quality_balanced"
    inside_samples: int = 5
    inside_batch_size: int = 1
    inside_max_new_tokens: int = 12
    inside_temperature: float = 0.7
    inside_top_p: float = 0.9
    inside_pooling: str = "last"
    inside_embedding_threshold: float = 0.90
    inside_min_samples: int = 2
    inside_sample_step: int = 1
    inside_stability_delta: float = 0.05
    inside_selfcheck_min_overlap: float = 0.65
    inside_selfcheck_support_threshold: float = 0.60
    inside_selfcheck_refute_threshold: float = 0.50
    inside_adaptive_max_sample_ratio: float = 1.0
    inside_adaptive_selfcheck_max_sample_ratio: float = 1.0
    max_inside_generation_seconds_ratio: float | None = None
    inside_run_names: Sequence[str] = INSIDE_PROFILE_RUN_NAMES
    inside_trigger_signal: str = "truth_proj"
    inside_trigger_budgets: Sequence[TriggerBudgetSpec] = (
        TriggerBudgetSpec("top_fraction", 0.25),
        TriggerBudgetSpec("top_fraction", 0.5),
    )
    inside_reference_report_path: Path | None = None
    derive_trigger_from_max_budget: bool = False
    refresh_shared_caches: bool = False
    clean: bool = False
    dry_run: bool = False
    skip_existing: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.report_path is not None:
            object.__setattr__(self, "report_path", Path(self.report_path))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
            if not self.name or not self.version:
                raise ValueError("registry_path requires name and version.")
        if self.shared_cache_dir is not None:
            object.__setattr__(self, "shared_cache_dir", Path(self.shared_cache_dir))
        for field_name in (
            "matrix_report_path",
            "worker_sweep_report_path",
            "inside_sampling_report_path",
            "inside_trigger_budget_sweep_report_path",
            "inside_reference_report_path",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, Path(value))
        object.__setattr__(self, "layers", tuple(int(value) for value in self.layers))
        object.__setattr__(self, "batch_sizes", tuple(int(value) for value in self.batch_sizes))
        object.__setattr__(
            self,
            "hidden_state_captures",
            tuple(str(value) for value in self.hidden_state_captures),
        )
        object.__setattr__(self, "worker_counts", tuple(int(value) for value in self.worker_counts))
        object.__setattr__(self, "inside_run_names", _parse_run_names(",".join(self.inside_run_names)))
        object.__setattr__(self, "inside_trigger_budgets", tuple(self.inside_trigger_budgets))
        policy = str(self.inside_trigger_budget_policy).strip().lower().replace("-", "_")
        if policy not in INSIDE_TRIGGER_BUDGET_POLICIES:
            choices = ", ".join(INSIDE_TRIGGER_BUDGET_POLICIES)
            raise ValueError(f"inside_trigger_budget_policy must be one of: {choices}")
        object.__setattr__(self, "inside_trigger_budget_policy", policy)
        if self.run_worker_sweep and self.worker_sweep_report_path is not None:
            raise ValueError("run_worker_sweep and worker_sweep_report_path are mutually exclusive.")
        if self.run_inside_sampling and self.inside_sampling_report_path is not None:
            raise ValueError("run_inside_sampling and inside_sampling_report_path are mutually exclusive.")
        if self.run_inside_trigger_budget_sweep and self.inside_trigger_budget_sweep_report_path is not None:
            raise ValueError(
                "run_inside_trigger_budget_sweep and inside_trigger_budget_sweep_report_path are mutually exclusive."
            )

    @property
    def resolved_report_path(self) -> Path:
        """Return the workflow report path."""
        return self.report_path or self.output_dir / "performance-baseline-workflow.json"

    @property
    def runtime_recommendation_path(self) -> Path:
        """Return the runtime recommendation artifact path."""
        return self.output_dir / "runtime-recommendation.json"

    @property
    def artifact_manifest_path(self) -> Path:
        """Return the workflow artifact manifest path."""
        return self.output_dir / "artifact-manifest.json"


def run_performance_baseline_workflow(config: PerformanceBaselineWorkflowConfig) -> dict[str, Any]:
    """Run or reuse performance evidence and write a baseline workflow report."""
    started_at = time.perf_counter()
    if config.clean and config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    matrix_report, matrix_report_path = _matrix_report(config)
    worker_sweep_report, worker_sweep_report_path = _worker_sweep_report(config)
    inside_sampling_report, inside_sampling_report_path = _inside_sampling_report(config)
    trigger_sweep_report, trigger_sweep_report_path = _inside_trigger_budget_sweep_report(config)

    runtime_recommendation = build_runtime_recommendation(
        matrix_report,
        worker_sweep_report=worker_sweep_report,
        inside_sampling_report=inside_sampling_report,
        inside_trigger_budget_sweep_report=trigger_sweep_report,
        inside_trigger_budget_policy=config.inside_trigger_budget_policy,
        matrix_report_path=matrix_report_path,
        worker_sweep_report_path=worker_sweep_report_path,
        inside_sampling_report_path=inside_sampling_report_path,
        inside_trigger_budget_sweep_report_path=trigger_sweep_report_path,
    )
    score_dump_cache_evidence = _score_dump_cache_evidence_summary(
        matrix_report=matrix_report,
        worker_sweep_report=worker_sweep_report,
        inside_sampling_report=inside_sampling_report,
        inside_trigger_budget_sweep_report=trigger_sweep_report,
    )
    config.runtime_recommendation_path.write_text(
        json.dumps(runtime_recommendation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    decision = _workflow_decision(runtime_recommendation)
    artifacts = _artifact_paths(
        config,
        matrix_report=matrix_report,
        matrix_report_path=matrix_report_path,
        worker_sweep_report=worker_sweep_report,
        worker_sweep_report_path=worker_sweep_report_path,
        inside_sampling_report=inside_sampling_report,
        inside_sampling_report_path=inside_sampling_report_path,
        trigger_sweep_report=trigger_sweep_report,
        trigger_sweep_report_path=trigger_sweep_report_path,
    )
    artifact_manifest_summary = planned_artifact_manifest_summary(
        artifacts,
        assume_file_paths=(config.resolved_report_path,),
    )
    report = {
        "schema_version": 1,
        "workflow": "performance_baseline_workflow",
        "status": decision["status"],
        "decision": decision,
        "runtime_recommendation": runtime_recommendation,
        "paths": {
            "report": str(config.resolved_report_path),
            "artifact_manifest": str(config.artifact_manifest_path),
            "runtime_recommendation": str(config.runtime_recommendation_path),
            "matrix_report": None if matrix_report_path is None else str(matrix_report_path),
            "worker_sweep_report": None if worker_sweep_report_path is None else str(worker_sweep_report_path),
            "inside_sampling_report": None if inside_sampling_report_path is None else str(inside_sampling_report_path),
            "inside_trigger_budget_sweep_report": (
                None if trigger_sweep_report_path is None else str(trigger_sweep_report_path)
            ),
        },
        "config": _config_payload(config),
        "execution": {
            "wall_clock_seconds": time.perf_counter() - started_at,
            "matrix_report_reused": config.matrix_report_path is not None,
            "worker_sweep_report_reused": config.worker_sweep_report_path is not None,
            "inside_sampling_report_reused": config.inside_sampling_report_path is not None,
            "inside_trigger_budget_sweep_report_reused": (
                config.inside_trigger_budget_sweep_report_path is not None
            ),
        },
        "artifact_manifest_summary": artifact_manifest_summary,
    }
    if config.registry_path is not None:
        report["registry_record"] = f"performance_baseline:{config.name}:{config.version}"
    report["performance_evidence_bundle"] = _performance_evidence_bundle_summary(
        config,
        report=report,
        runtime_recommendation=runtime_recommendation,
        artifact_manifest_summary=artifact_manifest_summary,
        score_dump_cache_evidence=score_dump_cache_evidence,
    )

    config.resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.resolved_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = _write_artifact_manifest(
        config,
        matrix_report=matrix_report,
        matrix_report_path=matrix_report_path,
        worker_sweep_report=worker_sweep_report,
        worker_sweep_report_path=worker_sweep_report_path,
        inside_sampling_report=inside_sampling_report,
        inside_sampling_report_path=inside_sampling_report_path,
        trigger_sweep_report=trigger_sweep_report,
        trigger_sweep_report_path=trigger_sweep_report_path,
        report=report,
    )
    report["artifact_manifest_summary"] = manifest["summary"]
    report["performance_evidence_bundle"] = _performance_evidence_bundle_summary(
        config,
        report=report,
        runtime_recommendation=runtime_recommendation,
        artifact_manifest_summary=manifest["summary"],
        score_dump_cache_evidence=score_dump_cache_evidence,
    )
    _record_registry(config, report)
    return report


def _matrix_report(config: PerformanceBaselineWorkflowConfig) -> tuple[dict[str, Any], Path | None]:
    if config.matrix_report_path is not None:
        return _load_json(config.matrix_report_path), config.matrix_report_path
    matrix_config = CacheProfileMatrixConfig(
        output_dir=config.output_dir / "cache_matrix",
        model=config.model,
        dtype=config.dtype,
        layers=config.layers,
        batch_sizes=config.batch_sizes,
        hidden_state_captures=config.hidden_state_captures,
        limit=config.limit,
        manifold_questions=config.manifold_questions,
        max_length=config.max_length,
        max_batch_tokens=config.max_batch_tokens,
        max_batch_token_budgets=config.max_batch_token_budgets,
        prefix_kv_cache=config.prefix_kv_cache,
        prefix_kv_cache_modes=config.prefix_kv_cache_modes,
        eval_reps_cache_shard_size=config.eval_reps_cache_shard_size,
        cached_max_total_ratio=config.cached_max_total_ratio,
        cache_only_max_total_ratio=config.cache_only_max_total_ratio,
        python_executable=config.python_executable,
        progress_every=config.progress_every,
        length_bucketed_batches=config.length_bucketed_batches,
        offline=config.offline,
        shared_cache_dir=None if config.shared_cache_dir is None else config.shared_cache_dir / "cache_matrix",
        matrix_mode=config.matrix_mode,
        max_workers=config.max_workers,
    )
    report = run_matrix(matrix_config, clean=False, dry_run=config.dry_run)
    return report, Path(str(report.get("report_path", matrix_config.report_path)))


def _worker_sweep_report(config: PerformanceBaselineWorkflowConfig) -> tuple[dict[str, Any] | None, Path | None]:
    if config.worker_sweep_report_path is not None:
        return _load_json(config.worker_sweep_report_path), config.worker_sweep_report_path
    if not config.run_worker_sweep:
        return None, None
    sweep_config = CacheWorkerSweepConfig(
        output_dir=config.output_dir / "worker_sweep",
        worker_counts=config.worker_counts,
        model=config.model,
        dtype=config.dtype,
        layers=config.layers,
        batch_sizes=config.batch_sizes,
        hidden_state_captures=config.hidden_state_captures,
        limit=config.limit,
        manifold_questions=config.manifold_questions,
        max_length=config.max_length,
        max_batch_tokens=config.max_batch_tokens,
        max_batch_token_budgets=config.max_batch_token_budgets,
        prefix_kv_cache=config.prefix_kv_cache,
        prefix_kv_cache_modes=config.prefix_kv_cache_modes,
        eval_reps_cache_shard_size=config.eval_reps_cache_shard_size,
        cached_max_total_ratio=config.cached_max_total_ratio,
        cache_only_max_total_ratio=config.cache_only_max_total_ratio,
        python_executable=config.python_executable,
        progress_every=config.progress_every,
        length_bucketed_batches=config.length_bucketed_batches,
        offline=config.offline,
        shared_cache_dir=None if config.shared_cache_dir is None else config.shared_cache_dir / "worker_sweep",
        matrix_mode=config.matrix_mode,
    )
    report = run_worker_sweep(sweep_config, clean=False, dry_run=config.dry_run)
    return report, Path(str(report.get("report_path", sweep_config.report_path)))


def _inside_sampling_report(
    config: PerformanceBaselineWorkflowConfig,
) -> tuple[dict[str, Any] | None, Path | None]:
    if config.inside_sampling_report_path is not None:
        return _load_json(config.inside_sampling_report_path), config.inside_sampling_report_path
    if not config.run_inside_sampling:
        return None, None
    sampling_config = _inside_sampling_config(config, output_dir=config.output_dir / "inside_sampling")
    report = run_inside_sampling_profile(
        sampling_config,
        clean=False,
        dry_run=config.dry_run,
        skip_existing=config.skip_existing,
    )
    if report.get("dry_run"):
        return None, None
    path = Path(str(report.get("comparison_report", sampling_config.comparison_report)))
    comparison = _load_json(path)
    if report.get("artifact_manifest") is not None:
        comparison["artifact_manifest"] = report.get("artifact_manifest")
    return comparison, path


def _inside_trigger_budget_sweep_report(
    config: PerformanceBaselineWorkflowConfig,
) -> tuple[dict[str, Any] | None, Path | None]:
    if config.inside_trigger_budget_sweep_report_path is not None:
        return (
            _load_json(config.inside_trigger_budget_sweep_report_path),
            config.inside_trigger_budget_sweep_report_path,
        )
    if not config.run_inside_trigger_budget_sweep:
        return None, None
    sweep_config = InsideTriggerBudgetSweepConfig(
        output_dir=config.output_dir / "inside_trigger_budget_sweep",
        trigger_signal=config.inside_trigger_signal,
        budgets=config.inside_trigger_budgets,
        model=config.model,
        dtype=config.dtype,
        layer=config.layers[0],
        limit=config.limit,
        manifold_questions=config.manifold_questions,
        batch_size=config.batch_sizes[0],
        max_batch_tokens=config.max_batch_tokens,
        max_length=config.max_length,
        hidden_state_capture=config.hidden_state_captures[0],
        progress_every=config.progress_every,
        offline=config.offline,
        length_bucketed_batches=config.length_bucketed_batches,
        python_executable=config.python_executable,
        inside_samples=config.inside_samples,
        inside_batch_size=config.inside_batch_size,
        inside_max_new_tokens=config.inside_max_new_tokens,
        inside_temperature=config.inside_temperature,
        inside_top_p=config.inside_top_p,
        inside_pooling=config.inside_pooling,
        inside_embedding_threshold=config.inside_embedding_threshold,
        inside_min_samples=config.inside_min_samples,
        inside_sample_step=config.inside_sample_step,
        inside_stability_delta=config.inside_stability_delta,
        inside_selfcheck_min_overlap=config.inside_selfcheck_min_overlap,
        inside_selfcheck_support_threshold=config.inside_selfcheck_support_threshold,
        inside_selfcheck_refute_threshold=config.inside_selfcheck_refute_threshold,
        adaptive_max_sample_ratio=config.inside_adaptive_max_sample_ratio,
        adaptive_selfcheck_max_sample_ratio=config.inside_adaptive_selfcheck_max_sample_ratio,
        max_inside_generation_seconds_ratio=config.max_inside_generation_seconds_ratio,
        run_names=config.inside_run_names,
        reference_report_path=config.inside_reference_report_path,
        shared_cache_dir=None if config.shared_cache_dir is None else config.shared_cache_dir / "inside_trigger",
        eval_reps_cache_shard_size=config.eval_reps_cache_shard_size,
        refresh_shared_caches=config.refresh_shared_caches,
        derive_from_max_budget=config.derive_trigger_from_max_budget,
    )
    report = run_inside_trigger_budget_sweep(
        sweep_config,
        clean=False,
        dry_run=config.dry_run,
        skip_existing=config.skip_existing,
    )
    path = Path(str(report.get("report_path", sweep_config.report_path)))
    return (None if report.get("dry_run") else _load_json(path)), path


def _inside_sampling_config(
    config: PerformanceBaselineWorkflowConfig,
    *,
    output_dir: Path,
) -> InsideSamplingProfileConfig:
    return InsideSamplingProfileConfig(
        output_dir=output_dir,
        model=config.model,
        dtype=config.dtype,
        layer=config.layers[0],
        limit=config.limit,
        manifold_questions=config.manifold_questions,
        batch_size=config.batch_sizes[0],
        max_batch_tokens=config.max_batch_tokens,
        max_length=config.max_length,
        hidden_state_capture=config.hidden_state_captures[0],
        progress_every=config.progress_every,
        offline=config.offline,
        length_bucketed_batches=config.length_bucketed_batches,
        python_executable=config.python_executable,
        inside_samples=config.inside_samples,
        inside_batch_size=config.inside_batch_size,
        inside_max_new_tokens=config.inside_max_new_tokens,
        inside_temperature=config.inside_temperature,
        inside_top_p=config.inside_top_p,
        inside_pooling=config.inside_pooling,
        inside_embedding_threshold=config.inside_embedding_threshold,
        inside_min_samples=config.inside_min_samples,
        inside_sample_step=config.inside_sample_step,
        inside_stability_delta=config.inside_stability_delta,
        inside_selfcheck_min_overlap=config.inside_selfcheck_min_overlap,
        inside_selfcheck_support_threshold=config.inside_selfcheck_support_threshold,
        inside_selfcheck_refute_threshold=config.inside_selfcheck_refute_threshold,
        adaptive_max_sample_ratio=config.inside_adaptive_max_sample_ratio,
        adaptive_selfcheck_max_sample_ratio=config.inside_adaptive_selfcheck_max_sample_ratio,
        max_inside_generation_seconds_ratio=config.max_inside_generation_seconds_ratio,
        run_names=config.inside_run_names,
        refresh_shared_caches=config.refresh_shared_caches,
        eval_reps_cache_shard_size=config.eval_reps_cache_shard_size,
        statement_encoding_cache_path=_shared_cache_path(config, "statement-encoding-cache.json"),
        layer_stats_cache_path=_shared_cache_path(config, "layer-stats-cache.json"),
        eval_reps_cache_path=_shared_cache_path(config, "eval-reps-cache"),
        inside_diagnostics_cache_path=_shared_cache_path(config, "inside-diagnostics-cache.json"),
    )


def _shared_cache_path(config: PerformanceBaselineWorkflowConfig, name: str) -> Path | None:
    if config.shared_cache_dir is None:
        return None
    return config.shared_cache_dir / "inside_sampling" / name


def _workflow_decision(runtime_recommendation: Mapping[str, Any]) -> dict[str, Any]:
    status = str(runtime_recommendation.get("status") or "missing")
    blocking_reasons = tuple(runtime_recommendation.get("blocking_reasons") or ())
    if status == "promote":
        recommendation = runtime_recommendation.get("recommendation")
        if isinstance(recommendation, Mapping):
            return {
                "status": "promote",
                "recommended_cell": recommendation.get("cell_id"),
                "recommended_layer": recommendation.get("layer"),
                "recommended_batch_size": recommendation.get("batch_size"),
                "recommended_max_workers": recommendation.get("max_workers"),
                "recommended_best_quality_signal": (
                    None
                    if not isinstance(recommendation.get("best_quality_signal"), Mapping)
                    else recommendation["best_quality_signal"].get("name")
                ),
                "blocking_reasons": (),
            }
    if status in {"dry_run", "needs_evidence"}:
        return {"status": "needs_evidence", "blocking_reasons": blocking_reasons}
    return {"status": "blocked", "blocking_reasons": blocking_reasons}


def _write_artifact_manifest(
    config: PerformanceBaselineWorkflowConfig,
    *,
    matrix_report: Mapping[str, Any],
    matrix_report_path: Path | None,
    worker_sweep_report: Mapping[str, Any] | None,
    worker_sweep_report_path: Path | None,
    inside_sampling_report: Mapping[str, Any] | None,
    inside_sampling_report_path: Path | None,
    trigger_sweep_report: Mapping[str, Any] | None,
    trigger_sweep_report_path: Path | None,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = _artifact_paths(
        config,
        matrix_report=matrix_report,
        matrix_report_path=matrix_report_path,
        worker_sweep_report=worker_sweep_report,
        worker_sweep_report_path=worker_sweep_report_path,
        inside_sampling_report=inside_sampling_report,
        inside_sampling_report_path=inside_sampling_report_path,
        trigger_sweep_report=trigger_sweep_report,
        trigger_sweep_report_path=trigger_sweep_report_path,
    )
    manifest = build_artifact_manifest(
        artifacts,
        root=config.output_dir,
        metadata={
            "runner": "run_performance_baseline_workflow",
            "status": report.get("status"),
            "runtime_recommendation_status": dict(report.get("runtime_recommendation") or {}).get("status"),
            "model": config.model,
            "dtype": config.dtype,
            "layers": tuple(config.layers),
            "batch_sizes": tuple(config.batch_sizes),
            "hidden_state_captures": tuple(config.hidden_state_captures),
            "offline": config.offline,
            "dry_run": config.dry_run,
            "matrix_report_reused": config.matrix_report_path is not None,
            "worker_sweep_enabled": bool(worker_sweep_report_path),
            "inside_sampling_enabled": bool(inside_sampling_report_path),
            "inside_trigger_budget_sweep_enabled": bool(trigger_sweep_report_path),
            "inside_trigger_budget_policy": config.inside_trigger_budget_policy,
            "recommended_cell": dict(report.get("decision") or {}).get("recommended_cell"),
            "recommended_layer": dict(report.get("decision") or {}).get("recommended_layer"),
            "recommended_batch_size": dict(report.get("decision") or {}).get("recommended_batch_size"),
            "recommended_max_workers": dict(report.get("decision") or {}).get("recommended_max_workers"),
            "recommended_best_quality_signal": dict(report.get("decision") or {}).get(
                "recommended_best_quality_signal"
            ),
        },
    )
    config.artifact_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _artifact_paths(
    config: PerformanceBaselineWorkflowConfig,
    *,
    matrix_report: Mapping[str, Any],
    matrix_report_path: Path | None,
    worker_sweep_report: Mapping[str, Any] | None,
    worker_sweep_report_path: Path | None,
    inside_sampling_report: Mapping[str, Any] | None,
    inside_sampling_report_path: Path | None,
    trigger_sweep_report: Mapping[str, Any] | None,
    trigger_sweep_report_path: Path | None,
) -> dict[str, str | Path | None]:
    return {
        "performance_baseline_report": config.resolved_report_path,
        "runtime_recommendation": config.runtime_recommendation_path,
        "matrix_report": matrix_report_path,
        "matrix_manifest": matrix_report.get("artifact_manifest"),
        "worker_sweep_report": worker_sweep_report_path,
        "worker_sweep_manifest": None if worker_sweep_report is None else worker_sweep_report.get("artifact_manifest"),
        "inside_sampling_report": inside_sampling_report_path,
        "inside_sampling_manifest": (
            None if inside_sampling_report is None else inside_sampling_report.get("artifact_manifest")
        ),
        "inside_trigger_budget_sweep_report": trigger_sweep_report_path,
        "inside_trigger_budget_sweep_manifest": (
            None if trigger_sweep_report is None else trigger_sweep_report.get("artifact_manifest")
        ),
    }


def _performance_evidence_bundle_summary(
    config: PerformanceBaselineWorkflowConfig,
    *,
    report: Mapping[str, Any],
    runtime_recommendation: Mapping[str, Any],
    artifact_manifest_summary: Mapping[str, Any],
    score_dump_cache_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(report.get("status"))
    recommendation_status = str(runtime_recommendation.get("status") or "missing")
    recommendation = _mapping(runtime_recommendation.get("recommendation"))
    evidence = _mapping(runtime_recommendation.get("evidence"))
    cache_tuning = _mapping(recommendation.get("cache_tuning"))
    inside_sampling = _mapping(recommendation.get("inside_sampling"))
    inside_trigger_budget = _mapping(recommendation.get("inside_trigger_budget_sweep"))
    missing_count = _int_or_zero(artifact_manifest_summary.get("missing_count"))
    release_ready = status == "promote" and recommendation_status == "promote" and missing_count == 0
    uncached_total = _float_or_none(recommendation.get("uncached_total_seconds"))
    cached_total = _float_or_none(recommendation.get("cached_total_seconds"))
    cache_only_total = _float_or_none(recommendation.get("cache_only_total_seconds"))
    forced_answer_forward = _float_or_none(
        recommendation.get("uncached_forced_answer_forward_seconds")
    )
    best_quality = _mapping(recommendation.get("best_quality_signal"))
    return {
        "schema_version": 1,
        "status": status,
        "release_ready": release_ready,
        "runtime": {
            "model": config.model,
            "dtype": config.dtype,
            "layers": tuple(config.layers),
            "batch_sizes": tuple(config.batch_sizes),
            "hidden_state_captures": tuple(config.hidden_state_captures),
            "max_batch_tokens": config.max_batch_tokens,
            "max_batch_token_budgets": (
                None if config.max_batch_token_budgets is None else tuple(config.max_batch_token_budgets)
            ),
            "prefix_kv_cache": config.prefix_kv_cache,
            "prefix_kv_cache_modes": (
                None if config.prefix_kv_cache_modes is None else tuple(config.prefix_kv_cache_modes)
            ),
            "max_workers": config.max_workers,
            "length_bucketed_batches": config.length_bucketed_batches,
            "offline": config.offline,
            "matrix_mode": config.matrix_mode,
        },
        "recommendation": {
            "status": recommendation_status,
            "cell_id": recommendation.get("cell_id"),
            "layer": recommendation.get("layer"),
            "batch_size": recommendation.get("batch_size"),
            "hidden_state_capture": recommendation.get("hidden_state_capture"),
            "max_batch_tokens": recommendation.get("max_batch_tokens"),
            "prefix_kv_cache": recommendation.get("prefix_kv_cache"),
            "max_workers": recommendation.get("max_workers"),
            "recommendation_metric": recommendation.get("recommendation_metric"),
            "best_quality_signal": best_quality.get("name"),
            "best_quality_auroc": best_quality.get("auroc"),
            "quality_signal_count": evidence.get("quality_signal_count"),
            "cache_tuning_status": cache_tuning.get("status"),
            "inside_sampling_run": inside_sampling.get("recommended_run"),
            "inside_trigger_budget_id": inside_trigger_budget.get("recommended_budget_id"),
        },
        "cost": {
            "uncached_total_seconds": uncached_total,
            "cached_total_seconds": cached_total,
            "cache_only_total_seconds": cache_only_total,
            "uncached_forced_answer_forward_seconds": forced_answer_forward,
            "cached_total_ratio": _safe_ratio(cached_total, uncached_total),
            "cache_only_total_ratio": _safe_ratio(cache_only_total, uncached_total),
        },
        "evidence": {
            "matrix_report": evidence.get("matrix_report"),
            "matrix_status": evidence.get("matrix_status"),
            "matrix_recommended_cell": evidence.get("matrix_recommended_cell"),
            "matrix_candidate_count": evidence.get("matrix_candidate_count"),
            "matrix_checked_cell_count": evidence.get("matrix_checked_cell_count"),
            "matrix_wall_clock_seconds": evidence.get("matrix_wall_clock_seconds"),
            "configured_matrix_workers": evidence.get("configured_matrix_workers"),
            "worker_sweep_report": evidence.get("worker_sweep_report"),
            "worker_sweep_status": evidence.get("worker_sweep_status"),
            "worker_recommended_worker_count": evidence.get("worker_recommended_worker_count"),
            "inside_sampling_report": evidence.get("inside_sampling_report"),
            "inside_sampling_status": evidence.get("inside_sampling_status"),
            "inside_sampling_gate_passed": evidence.get("inside_sampling_gate_passed"),
            "inside_trigger_budget_sweep_report": evidence.get("inside_trigger_budget_sweep_report"),
            "inside_trigger_budget_policy": evidence.get("inside_trigger_budget_policy"),
            "inside_trigger_budget_sweep_status": evidence.get("inside_trigger_budget_sweep_status"),
            "inside_trigger_budget_gate_passed": evidence.get("inside_trigger_budget_gate_passed"),
        },
        "score_dump_cache": dict(score_dump_cache_evidence),
        "artifacts": {
            "summary": dict(artifact_manifest_summary),
            "artifact_manifest": str(config.artifact_manifest_path),
            "runtime_recommendation": str(config.runtime_recommendation_path),
        },
        "registry": {
            "record": report.get("registry_record"),
            "path": None if config.registry_path is None else str(config.registry_path),
        },
    }


def _record_registry(config: PerformanceBaselineWorkflowConfig, report: Mapping[str, Any]) -> None:
    if config.registry_path is None:
        return
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_performance_baseline(
        name=str(config.name),
        path=config.resolved_report_path,
        version=str(config.version),
        metadata={
            "workflow": "run_performance_baseline_workflow",
            "status": report.get("status"),
            "artifact_manifest": str(config.artifact_manifest_path),
            "runtime_recommendation": str(config.runtime_recommendation_path),
            "runtime_recommendation_status": dict(report.get("runtime_recommendation") or {}).get("status"),
            "recommended_cell": dict(report.get("decision") or {}).get("recommended_cell"),
            "recommended_layer": dict(report.get("decision") or {}).get("recommended_layer"),
            "recommended_batch_size": dict(report.get("decision") or {}).get("recommended_batch_size"),
            "recommended_max_workers": dict(report.get("decision") or {}).get("recommended_max_workers"),
            "recommended_best_quality_signal": dict(report.get("decision") or {}).get(
                "recommended_best_quality_signal"
            ),
            "performance_evidence_bundle_status": _nested(
                report,
                "performance_evidence_bundle",
                "status",
            ),
            "performance_evidence_bundle_release_ready": _nested(
                report,
                "performance_evidence_bundle",
                "release_ready",
            ),
            "performance_cache_tuning_status": _nested(
                report,
                "performance_evidence_bundle",
                "recommendation",
                "cache_tuning_status",
            ),
        },
    )
    registry.save_json()


_SCORE_DUMP_CACHE_SECTIONS = ("fingerprint", "jsonl_summary", "jsonl_view")


def _score_dump_cache_evidence_summary(
    *,
    matrix_report: Mapping[str, Any],
    worker_sweep_report: Mapping[str, Any] | None,
    inside_sampling_report: Mapping[str, Any] | None,
    inside_trigger_budget_sweep_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    sources = []
    totals = _empty_score_dump_cache_totals()
    for source_name, source_report in (
        ("matrix_report", matrix_report),
        ("worker_sweep_report", worker_sweep_report),
        ("inside_sampling_report", inside_sampling_report),
        ("inside_trigger_budget_sweep_report", inside_trigger_budget_sweep_report),
    ):
        if source_report is None:
            continue
        summary = _mapping(source_report.get("score_dump_cache"))
        if not summary or summary.get("enabled") is not True:
            continue
        source = _score_dump_cache_source_summary(source_name, summary)
        sources.append(source)
        totals["cache_entries"] += int(source.get("cache_entries", 0) or 0)
        for section in _SCORE_DUMP_CACHE_SECTIONS:
            _add_cache_counter(totals[section], _mapping(source.get(section)))
    return {
        "enabled": bool(sources),
        "source_count": len(sources),
        "cache_entries": totals["cache_entries"],
        "totals": _score_dump_cache_totals_payload(totals),
        "sources": sources,
    }


def _score_dump_cache_source_summary(source_name: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": source_name,
        "enabled": bool(summary.get("enabled", False)),
        "cache_entries": _int_or_zero(summary.get("cache_entries")),
        "fingerprint": _cache_counter_payload(_mapping(summary.get("fingerprint"))),
        "jsonl_summary": _cache_counter_payload(_mapping(summary.get("jsonl_summary"))),
        "jsonl_view": _cache_counter_payload(_mapping(summary.get("jsonl_view"))),
    }


def _empty_score_dump_cache_totals() -> dict[str, Any]:
    return {
        "cache_entries": 0,
        "fingerprint": {"hits": 0, "misses": 0, "writes": 0},
        "jsonl_summary": {"hits": 0, "misses": 0, "writes": 0},
        "jsonl_view": {"hits": 0, "misses": 0, "writes": 0},
    }


def _score_dump_cache_totals_payload(totals: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fingerprint": _cache_counter_payload(_mapping(totals.get("fingerprint"))),
        "jsonl_summary": _cache_counter_payload(_mapping(totals.get("jsonl_summary"))),
        "jsonl_view": _cache_counter_payload(_mapping(totals.get("jsonl_view"))),
    }


def _add_cache_counter(target: dict[str, int], source: Mapping[str, Any]) -> None:
    for key in ("hits", "misses", "writes"):
        target[key] = int(target.get(key, 0)) + _int_or_zero(source.get(key))


def _cache_counter_payload(counter: Mapping[str, Any]) -> dict[str, Any]:
    hits = _int_or_zero(counter.get("hits"))
    misses = _int_or_zero(counter.get("misses"))
    writes = _int_or_zero(counter.get("writes"))
    attempts = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "writes": writes,
        "attempts": attempts,
        "hit_rate": None if attempts == 0 else hits / attempts,
    }


def _config_payload(config: PerformanceBaselineWorkflowConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "dtype": config.dtype,
        "layers": tuple(config.layers),
        "batch_sizes": tuple(config.batch_sizes),
        "hidden_state_captures": tuple(config.hidden_state_captures),
        "limit": config.limit,
        "manifold_questions": config.manifold_questions,
        "max_length": config.max_length,
        "max_batch_tokens": config.max_batch_tokens,
        "max_batch_token_budgets": (
            None if config.max_batch_token_budgets is None else tuple(config.max_batch_token_budgets)
        ),
        "prefix_kv_cache": config.prefix_kv_cache,
        "prefix_kv_cache_modes": (
            None if config.prefix_kv_cache_modes is None else tuple(config.prefix_kv_cache_modes)
        ),
        "eval_reps_cache_shard_size": config.eval_reps_cache_shard_size,
        "cached_max_total_ratio": config.cached_max_total_ratio,
        "cache_only_max_total_ratio": config.cache_only_max_total_ratio,
        "length_bucketed_batches": config.length_bucketed_batches,
        "offline": config.offline,
        "shared_cache_dir": None if config.shared_cache_dir is None else str(config.shared_cache_dir),
        "matrix_mode": config.matrix_mode,
        "max_workers": config.max_workers,
        "run_worker_sweep": config.run_worker_sweep,
        "worker_counts": tuple(config.worker_counts),
        "run_inside_sampling": config.run_inside_sampling,
        "run_inside_trigger_budget_sweep": config.run_inside_trigger_budget_sweep,
        "inside_trigger_budget_policy": config.inside_trigger_budget_policy,
        "inside_trigger_signal": config.inside_trigger_signal,
        "inside_trigger_budgets": [
            {"kind": budget.kind, "value": budget.value, "id": budget.id}
            for budget in config.inside_trigger_budgets
        ],
        "dry_run": config.dry_run,
        "skip_existing": config.skip_existing,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _parse_trigger_budget(value: str) -> TriggerBudgetSpec:
    if "=" not in value:
        raise ValueError("--inside-trigger-budget must be formatted as kind=value.")
    kind, raw_value = value.split("=", 1)
    return TriggerBudgetSpec(kind.strip(), float(raw_value))


def _config_from_args(args: argparse.Namespace) -> PerformanceBaselineWorkflowConfig:
    return PerformanceBaselineWorkflowConfig(
        output_dir=Path(args.output_dir),
        report_path=Path(args.json) if args.json else None,
        registry_path=Path(args.registry) if args.registry else None,
        name=args.name,
        version=args.version,
        model=args.model,
        dtype=args.dtype,
        layers=_parse_int_list(args.layers, name="layers"),
        batch_sizes=_parse_int_list(args.batch_sizes, name="batch_sizes"),
        hidden_state_captures=_parse_str_list(args.hidden_state_captures, name="hidden_state_captures"),
        limit=args.limit,
        manifold_questions=args.manifold_questions,
        max_length=args.max_length,
        max_batch_tokens=args.max_batch_tokens,
        max_batch_token_budgets=_parse_max_batch_token_budgets(args.max_batch_token_budgets),
        prefix_kv_cache=args.prefix_kv_cache,
        prefix_kv_cache_modes=_parse_prefix_kv_cache_modes(args.prefix_kv_cache_modes),
        eval_reps_cache_shard_size=args.eval_reps_cache_shard_size,
        cached_max_total_ratio=args.cached_max_total_ratio,
        cache_only_max_total_ratio=args.cache_only_max_total_ratio,
        python_executable=args.python,
        progress_every=args.progress_every,
        length_bucketed_batches=not args.no_length_bucketed_batches,
        offline=not args.real_truthfulqa,
        shared_cache_dir=Path(args.shared_cache_dir) if args.shared_cache_dir else None,
        matrix_mode=args.matrix_mode,
        max_workers=args.max_workers,
        matrix_report_path=Path(args.matrix_report) if args.matrix_report else None,
        worker_sweep_report_path=Path(args.worker_sweep_report) if args.worker_sweep_report else None,
        run_worker_sweep=args.run_worker_sweep,
        worker_counts=_parse_int_list(args.worker_counts, name="worker_counts"),
        inside_sampling_report_path=Path(args.inside_sampling_report) if args.inside_sampling_report else None,
        run_inside_sampling=args.run_inside_sampling,
        inside_trigger_budget_sweep_report_path=(
            Path(args.inside_trigger_budget_sweep_report)
            if args.inside_trigger_budget_sweep_report
            else None
        ),
        run_inside_trigger_budget_sweep=args.run_inside_trigger_budget_sweep,
        inside_trigger_budget_policy=args.inside_trigger_budget_policy,
        inside_samples=args.inside_samples,
        inside_batch_size=args.inside_batch_size,
        inside_max_new_tokens=args.inside_max_new_tokens,
        inside_temperature=args.inside_temperature,
        inside_top_p=args.inside_top_p,
        inside_pooling=args.inside_pooling,
        inside_embedding_threshold=args.inside_embedding_threshold,
        inside_min_samples=args.inside_min_samples,
        inside_sample_step=args.inside_sample_step,
        inside_stability_delta=args.inside_stability_delta,
        inside_selfcheck_min_overlap=args.inside_selfcheck_min_overlap,
        inside_selfcheck_support_threshold=args.inside_selfcheck_support_threshold,
        inside_selfcheck_refute_threshold=args.inside_selfcheck_refute_threshold,
        inside_adaptive_max_sample_ratio=args.inside_adaptive_max_sample_ratio,
        inside_adaptive_selfcheck_max_sample_ratio=args.inside_adaptive_selfcheck_max_sample_ratio,
        max_inside_generation_seconds_ratio=args.max_inside_generation_seconds_ratio,
        inside_run_names=_parse_run_names(args.inside_run_names),
        inside_trigger_signal=args.inside_trigger_signal,
        inside_trigger_budgets=tuple(
            _parse_trigger_budget(value)
            for value in (
                args.inside_trigger_budget
                if args.inside_trigger_budget
                else ["top_fraction=0.25", "top_fraction=0.5"]
            )
        ),
        inside_reference_report_path=(
            Path(args.inside_reference_report) if args.inside_reference_report else None
        ),
        derive_trigger_from_max_budget=args.derive_trigger_from_max_budget,
        refresh_shared_caches=args.refresh_shared_caches,
        clean=args.clean,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = run_performance_baseline_workflow(_config_from_args(args))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_blocked and report["status"] != "promote":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a registry-ready performance baseline bundle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None, help="optional workflow report path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry performance baseline name")
    parser.add_argument("--version", default=None, help="registry performance baseline version")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--layers", default="-1")
    parser.add_argument("--batch-sizes", default="4")
    parser.add_argument("--hidden-state-captures", default="outputs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--manifold-questions", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=0)
    parser.add_argument("--max-batch-token-budgets", default=None)
    parser.add_argument("--prefix-kv-cache", action="store_true")
    parser.add_argument("--prefix-kv-cache-modes", default=None)
    parser.add_argument("--eval-reps-cache-shard-size", type=int, default=4)
    parser.add_argument("--cached-max-total-ratio", type=float, default=1.10)
    parser.add_argument("--cache-only-max-total-ratio", type=float, default=0.35)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--no-length-bucketed-batches", action="store_true")
    parser.add_argument("--real-truthfulqa", action="store_true")
    parser.add_argument("--shared-cache-dir", default=None)
    parser.add_argument("--matrix-mode", default="triplet", choices=MATRIX_MODES)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--matrix-report", default=None, help="reuse an existing cache-profile matrix report")
    parser.add_argument("--worker-sweep-report", default=None, help="reuse an existing worker sweep report")
    parser.add_argument("--run-worker-sweep", action="store_true")
    parser.add_argument("--worker-counts", default="1,2")
    parser.add_argument("--inside-sampling-report", default=None)
    parser.add_argument("--run-inside-sampling", action="store_true")
    parser.add_argument("--inside-trigger-budget-sweep-report", default=None)
    parser.add_argument("--run-inside-trigger-budget-sweep", action="store_true")
    parser.add_argument(
        "--inside-trigger-budget-policy",
        default="quality_balanced",
        choices=INSIDE_TRIGGER_BUDGET_POLICIES,
    )
    parser.add_argument("--inside-samples", type=int, default=5)
    parser.add_argument("--inside-batch-size", type=int, default=1)
    parser.add_argument("--inside-max-new-tokens", type=int, default=12)
    parser.add_argument("--inside-temperature", type=float, default=0.7)
    parser.add_argument("--inside-top-p", type=float, default=0.9)
    parser.add_argument("--inside-pooling", default="last", choices=["last", "mean"])
    parser.add_argument("--inside-embedding-threshold", type=float, default=0.90)
    parser.add_argument("--inside-min-samples", type=int, default=2)
    parser.add_argument("--inside-sample-step", type=int, default=1)
    parser.add_argument("--inside-stability-delta", type=float, default=0.05)
    parser.add_argument("--inside-selfcheck-min-overlap", type=float, default=0.65)
    parser.add_argument("--inside-selfcheck-support-threshold", type=float, default=0.60)
    parser.add_argument("--inside-selfcheck-refute-threshold", type=float, default=0.50)
    parser.add_argument("--inside-adaptive-max-sample-ratio", type=float, default=1.0)
    parser.add_argument("--inside-adaptive-selfcheck-max-sample-ratio", type=float, default=1.0)
    parser.add_argument("--max-inside-generation-seconds-ratio", type=float, default=None)
    parser.add_argument("--inside-run-names", default="fixed,adaptive,adaptive_selfcheck")
    parser.add_argument("--inside-trigger-signal", default="truth_proj")
    parser.add_argument(
        "--inside-trigger-budget",
        action="append",
        default=[],
        help="trigger budget as kind=value; repeatable",
    )
    parser.add_argument("--inside-reference-report", default=None)
    parser.add_argument("--derive-trigger-from-max-budget", action="store_true")
    parser.add_argument("--refresh-shared-caches", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
