"""Run multi-model TruthfulQA frontier calibrated-observability experiments.

This workflow is a thin orchestrator over the existing building blocks:

1. run or reuse one calibrated-observability closure per model/scale cell;
2. compare the resulting score dumps with rank-fusion ensembles;
3. write a top-level manifest and optional local registry record.

It is intended for Qwen/SmolLM2 l20/l80 style research runs while still
supporting bounded offline smoke tests.
"""

from __future__ import annotations

import argparse
import json
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
from benchmarks.eval_detectability_taxonomy import (  # noqa: E402
    build_detectability_taxonomy_report,
    write_detectability_artifact_manifest,
)
from benchmarks.eval_score_ensemble import build_ensemble_report  # noqa: E402
from benchmarks.run_calibrated_observability_workflow import (  # noqa: E402
    CalibratedObservabilityWorkflowConfig,
    run_calibrated_observability_workflow,
)
from eigentruth.registry import ArtifactRegistry, ArtifactVerificationContext  # noqa: E402

DEFAULT_FRONTIER_MODELS = (
    "qwen05=Qwen/Qwen2.5-0.5B-Instruct",
    "smollm2=HuggingFaceTB/SmolLM2-135M-Instruct",
)
DEFAULT_FRONTIER_SCALES = (
    "l20=20:40:-8:-16,-14,-12,-10,-8",
    "l80=80:80:-12:-16,-14,-12,-10,-8",
)
DEFAULT_FRONTIER_SIGNALS = (
    "truth_proj",
    "maha_last",
    "subspace_resid",
    "resid_update_norm",
    "resid_update_profile_area",
    "resid_update_profile_peak",
    "resid_update_profile_late_mass",
    "resid_update_profile_concentration",
    "prompt_answer_distance",
    "prompt_answer_cosine_gap",
    "answer_anchor_distance",
    "answer_path_length",
    "pathway_disagreement",
    "eigenscore",
)
DEFAULT_ENSEMBLE_METHODS = ("max_rank", "mean_rank")
DEFAULT_ALPHAS = (0.05, 0.10, 0.20)


@dataclass(frozen=True)
class ModelSpec:
    """Named model entry for a frontier run."""

    name: str
    model_id: str

    def __post_init__(self) -> None:
        name = self.name.strip()
        model_id = self.model_id.strip()
        if not name:
            raise ValueError("model name must be non-empty.")
        if not model_id:
            raise ValueError("model id must be non-empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "model_id", model_id)


@dataclass(frozen=True)
class ScaleSpec:
    """Named TruthfulQA scale and layer-band entry.

    ``name`` usually follows the existing l20/l80 convention where the number is
    the eval question limit, not the transformer layer.
    """

    name: str
    limit: int
    manifold_questions: int
    layer: int
    sweep_layers: Sequence[int]

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("scale name must be non-empty.")
        if int(self.limit) < 0:
            raise ValueError("scale limit must be >=0.")
        if int(self.manifold_questions) < 1:
            raise ValueError("scale manifold_questions must be >=1.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "limit", int(self.limit))
        object.__setattr__(self, "manifold_questions", int(self.manifold_questions))
        object.__setattr__(self, "layer", int(self.layer))
        object.__setattr__(self, "sweep_layers", tuple(int(layer) for layer in self.sweep_layers))


@dataclass(frozen=True)
class TruthfulQAFrontierWorkflowConfig:
    """Configuration for the multi-cell frontier experiment workflow."""

    output_dir: Path
    models: Sequence[ModelSpec]
    scales: Sequence[ScaleSpec]
    registry_path: Path | None = None
    name: str | None = None
    version: str | None = None
    cache_dir: Path | None = None
    scores_path: Path | None = None
    sweep_layers_from_band_report: Path | None = None
    sweep_band_strategy: str | None = None
    sweep_band_expand_radius: int = 0
    sweep_band_target_layer: str | None = None
    sweep_band_run_template: str = "{cell}"
    sweep_band_scales: Sequence[str] = ()
    dtype: str = "float32"
    batch_size: int = 4
    max_batch_tokens: int = 0
    max_length: int = 96
    hidden_state_capture: str = "hooks"
    attention_pathway: bool = False
    attn_implementation: str | None = None
    covariance_mode: str = "full"
    covariance_low_rank: int = 16
    progress_every: int = 0
    length_bucketed_batches: bool = True
    offline: bool = False
    auto_batch_size: bool = True
    cache_only: bool = False
    refresh_caches: bool = False
    warmup_checkpoint_every: int = 50
    eval_reps_cache_shard_size: int = 16
    eval_reps_shard_read_cache_size: int = 2
    dump_scores_format: str = "jsonl"
    refresh_scores: bool = False
    signals: Sequence[str] = DEFAULT_FRONTIER_SIGNALS
    conformal_signal: str = "maha_last"
    conformal_repeats: int = 20
    ensemble_repeats: int = 20
    seed: int = 0
    artifact_alpha: float = 0.10
    multiple_testing_signals: Sequence[str] = ()
    multiple_testing_alpha: float | None = None
    multiple_testing_method: str = "by"
    best_alpha: float = 0.10
    best_by: str = "auroc"
    ensemble_methods: Sequence[str] = DEFAULT_ENSEMBLE_METHODS
    alphas: Sequence[float] = DEFAULT_ALPHAS
    detectability_consistency_signal: str | None = None
    detectability_confidence_signal: str | None = None
    detectability_consistency_direction: str = "higher"
    detectability_confidence_direction: str = "higher"
    python_executable: str = sys.executable
    clean: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if self.scores_path is not None:
            object.__setattr__(self, "scores_path", Path(self.scores_path))
        if self.sweep_layers_from_band_report is not None:
            object.__setattr__(self, "sweep_layers_from_band_report", Path(self.sweep_layers_from_band_report))
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        models = tuple(self.models)
        scales = tuple(self.scales)
        if not models:
            raise ValueError("at least one model is required.")
        if not scales:
            raise ValueError("at least one scale is required.")
        if len({model.name for model in models}) != len(models):
            raise ValueError("model names must be unique.")
        if len({scale.name for scale in scales}) != len(scales):
            raise ValueError("scale names must be unique.")
        if self.scores_path is not None and len(models) * len(scales) != 1:
            raise ValueError("scores_path can only be used with exactly one model-scale cell.")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be >=1.")
        if int(self.max_batch_tokens) < 0:
            raise ValueError("max_batch_tokens must be >=0.")
        if int(self.max_length) < 1:
            raise ValueError("max_length must be >=1.")
        if int(self.covariance_low_rank) < 1:
            raise ValueError("covariance_low_rank must be >=1.")
        if int(self.warmup_checkpoint_every) < 0:
            raise ValueError("warmup_checkpoint_every must be >=0.")
        if int(self.eval_reps_cache_shard_size) < 0:
            raise ValueError("eval_reps_cache_shard_size must be >=0.")
        if int(self.eval_reps_shard_read_cache_size) < 1:
            raise ValueError("eval_reps_shard_read_cache_size must be >=1.")
        if int(self.sweep_band_expand_radius) < 0:
            raise ValueError("sweep_band_expand_radius must be >=0.")
        if int(self.conformal_repeats) < 1 or int(self.ensemble_repeats) < 1:
            raise ValueError("repeats must be >=1.")
        if self.cache_only and self.cache_dir is None:
            raise ValueError("cache_only requires cache_dir.")
        if self.refresh_caches and self.cache_dir is None:
            raise ValueError("refresh_caches requires cache_dir.")
        if self.cache_only and self.refresh_caches:
            raise ValueError("cache_only cannot refresh caches.")
        if self.hidden_state_capture not in {"outputs", "hooks"}:
            raise ValueError("hidden_state_capture must be one of: outputs, hooks.")
        if self.attn_implementation is not None:
            attn_implementation = str(self.attn_implementation).strip()
            if not attn_implementation:
                raise ValueError("attn_implementation must be non-empty when set.")
            object.__setattr__(self, "attn_implementation", attn_implementation)
        object.__setattr__(self, "attention_pathway", bool(self.attention_pathway))
        if self.multiple_testing_alpha is not None:
            object.__setattr__(self, "multiple_testing_alpha", float(self.multiple_testing_alpha))
        if self.covariance_mode not in {"full", "diag", "low_rank", "shrinkage"}:
            raise ValueError("covariance_mode must be one of: full, diag, low_rank, shrinkage.")
        if self.dump_scores_format not in {"json", "jsonl"}:
            raise ValueError("dump_scores_format must be one of: json, jsonl.")
        if self.best_by not in {"auroc", "detection"}:
            raise ValueError("best_by must be one of: auroc, detection.")
        if self.multiple_testing_alpha is not None and not 0.0 < self.multiple_testing_alpha < 1.0:
            raise ValueError("multiple_testing_alpha must be in (0, 1).")
        multiple_testing_method = str(self.multiple_testing_method).strip().lower().replace("_", "-")
        if multiple_testing_method not in {"by", "bh", "bonferroni"}:
            raise ValueError("multiple_testing_method must be one of: by, bh, bonferroni.")
        if self.detectability_consistency_direction not in {"higher", "lower"}:
            raise ValueError("detectability_consistency_direction must be one of: higher, lower.")
        if self.detectability_confidence_direction not in {"higher", "lower"}:
            raise ValueError("detectability_confidence_direction must be one of: higher, lower.")
        if bool(self.detectability_consistency_signal) != bool(self.detectability_confidence_signal):
            raise ValueError(
                "detectability_consistency_signal and detectability_confidence_signal must be set together."
            )
        if self.sweep_band_target_layer not in {None, "best", "band_best", "first"}:
            raise ValueError("sweep_band_target_layer must be one of: best, band_best, first.")
        signals = tuple(str(signal).strip() for signal in self.signals if str(signal).strip())
        multiple_testing_signals = tuple(
            str(signal).strip()
            for signal in self.multiple_testing_signals
            if str(signal).strip()
        )
        sweep_band_scales = tuple(str(scale).strip() for scale in self.sweep_band_scales if str(scale).strip())
        if self.sweep_layers_from_band_report is not None:
            missing_scales = sorted(set(sweep_band_scales) - {scale.name for scale in scales})
            if missing_scales:
                raise ValueError(f"sweep_band_scales contains unknown scale names: {', '.join(missing_scales)}")
        conformal_signal = str(self.conformal_signal).strip()
        detectability_consistency_signal = (
            None if self.detectability_consistency_signal is None
            else str(self.detectability_consistency_signal).strip()
        )
        detectability_confidence_signal = (
            None if self.detectability_confidence_signal is None
            else str(self.detectability_confidence_signal).strip()
        )
        methods = tuple(str(method) for method in self.ensemble_methods if str(method))
        alphas = tuple(float(alpha) for alpha in self.alphas)
        if not signals:
            raise ValueError("at least one signal is required.")
        if not conformal_signal:
            raise ValueError("conformal_signal must be non-empty.")
        if conformal_signal not in signals:
            raise ValueError("conformal_signal must be included in signals.")
        if bool(detectability_consistency_signal) != bool(detectability_confidence_signal):
            raise ValueError(
                "detectability_consistency_signal and detectability_confidence_signal must be non-empty together."
            )
        if (
            detectability_consistency_signal is not None
            and detectability_confidence_signal is not None
            and detectability_consistency_signal == detectability_confidence_signal
        ):
            raise ValueError("detectability_consistency_signal and detectability_confidence_signal must differ.")
        if not methods:
            raise ValueError("at least one ensemble method is required.")
        if any(not (0.0 < alpha < 1.0) for alpha in alphas):
            raise ValueError("alphas must be in (0, 1).")
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "multiple_testing_signals", multiple_testing_signals)
        object.__setattr__(self, "multiple_testing_method", multiple_testing_method)
        object.__setattr__(self, "sweep_band_scales", sweep_band_scales)
        object.__setattr__(self, "sweep_band_expand_radius", int(self.sweep_band_expand_radius))
        if self.sweep_band_strategy is not None:
            object.__setattr__(self, "sweep_band_strategy", str(self.sweep_band_strategy))
        if self.sweep_band_target_layer is not None:
            object.__setattr__(self, "sweep_band_target_layer", str(self.sweep_band_target_layer))
        object.__setattr__(self, "sweep_band_run_template", str(self.sweep_band_run_template))
        object.__setattr__(self, "conformal_signal", conformal_signal)
        object.__setattr__(self, "detectability_consistency_signal", detectability_consistency_signal)
        object.__setattr__(self, "detectability_confidence_signal", detectability_confidence_signal)
        object.__setattr__(self, "ensemble_methods", methods)
        object.__setattr__(self, "alphas", alphas)
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "max_batch_tokens", int(self.max_batch_tokens))
        object.__setattr__(self, "max_length", int(self.max_length))
        object.__setattr__(self, "covariance_low_rank", int(self.covariance_low_rank))
        object.__setattr__(self, "progress_every", int(self.progress_every))
        object.__setattr__(self, "warmup_checkpoint_every", int(self.warmup_checkpoint_every))
        object.__setattr__(self, "eval_reps_cache_shard_size", int(self.eval_reps_cache_shard_size))
        object.__setattr__(self, "eval_reps_shard_read_cache_size", int(self.eval_reps_shard_read_cache_size))
        object.__setattr__(self, "conformal_repeats", int(self.conformal_repeats))
        object.__setattr__(self, "ensemble_repeats", int(self.ensemble_repeats))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "artifact_alpha", float(self.artifact_alpha))
        object.__setattr__(self, "best_alpha", float(self.best_alpha))

    @property
    def report_path(self) -> Path:
        return self.output_dir / "truthfulqa-frontier-workflow.json"

    @property
    def ensemble_report_path(self) -> Path:
        return self.output_dir / "score-ensemble-report.json"

    @property
    def artifact_manifest_path(self) -> Path:
        return self.output_dir / "artifact-manifest.json"

    @property
    def multiple_testing_enabled(self) -> bool:
        return bool(self.multiple_testing_signals)

    @property
    def resolved_multiple_testing_alpha(self) -> float:
        return self.artifact_alpha if self.multiple_testing_alpha is None else self.multiple_testing_alpha


def run_truthfulqa_frontier_workflow(config: TruthfulQAFrontierWorkflowConfig) -> dict[str, Any]:
    """Run or plan the multi-model frontier workflow."""
    started_at = time.perf_counter()
    if config.clean and config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    cell_reports = []
    score_dumps: list[tuple[str, Path]] = []
    for model in config.models:
        for scale in config.scales:
            cell_name = f"{model.name}-{scale.name}"
            cell_dir = config.output_dir / cell_name
            cell_config = _cell_config(config, model=model, scale=scale, cell_dir=cell_dir)
            cell_report = run_calibrated_observability_workflow(cell_config)
            cell_reports.append(_cell_summary(cell_name, model, scale, cell_report))
            score_dumps.append((cell_name, cell_config.resolved_scores_path))

    ensemble_payload = None
    if not config.dry_run:
        ensemble_payload = build_ensemble_report(
            score_dumps,
            signals=config.signals,
            methods=config.ensemble_methods,
            alphas=config.alphas,
            repeats=config.ensemble_repeats,
            seed=config.seed,
            best_alpha=config.best_alpha,
        )
        config.ensemble_report_path.write_text(
            json.dumps(ensemble_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    detectability_reports = _build_detectability_reports(config, cell_reports)
    artifacts = _artifact_paths(config, cell_reports)
    manifest_summary = planned_artifact_manifest_summary(artifacts, assume_file_paths=(config.report_path,))
    status = _workflow_status(config, cell_reports, ensemble_payload)
    report: dict[str, Any] = {
        "schema_version": 1,
        "workflow": "truthfulqa_frontier_workflow",
        "status": status,
        "config": _config_payload(config),
        "paths": {
            "workflow_report": str(config.report_path),
            "score_ensemble_report": str(config.ensemble_report_path),
            "artifact_manifest": str(config.artifact_manifest_path),
        },
        "cells": cell_reports,
        "ensemble": _ensemble_summary(ensemble_payload, path=config.ensemble_report_path),
        "detectability_taxonomy": _detectability_summary(detectability_reports),
        "multiple_testing_gate": _multiple_testing_summary(config, cell_reports),
        "artifact_manifest_summary": manifest_summary,
        "execution": {
            "wall_clock_seconds": time.perf_counter() - started_at,
            "dry_run": config.dry_run,
        },
    }
    if config.registry_path is not None:
        report["registry_record"] = f"report:{config.name}:{config.version}"
    config.report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = _write_artifact_manifest(config, report, artifacts)
    report["artifact_manifest_summary"] = manifest["summary"]
    _record_registry(config, report)
    return report


def _cell_config(
    config: TruthfulQAFrontierWorkflowConfig,
    *,
    model: ModelSpec,
    scale: ScaleSpec,
    cell_dir: Path,
) -> CalibratedObservabilityWorkflowConfig:
    cell_name = f"{model.name}-{scale.name}"
    band_report = config.sweep_layers_from_band_report if _cell_uses_band_report(config, scale=scale) else None
    return CalibratedObservabilityWorkflowConfig(
        output_dir=cell_dir,
        model=model.model_id,
        dtype=config.dtype,
        layer=scale.layer,
        sweep=True,
        sweep_layers=() if band_report is not None else scale.sweep_layers,
        sweep_layers_from_band_report=band_report,
        sweep_band_strategy=config.sweep_band_strategy,
        sweep_band_run=(
            None if band_report is None else _format_sweep_band_run_template(config, model=model, scale=scale)
        ),
        sweep_band_expand_radius=config.sweep_band_expand_radius,
        sweep_band_target_layer=config.sweep_band_target_layer,
        limit=scale.limit,
        manifold_questions=scale.manifold_questions,
        max_length=config.max_length,
        scores_path=config.scores_path,
        batch_size=config.batch_size,
        max_batch_tokens=config.max_batch_tokens,
        hidden_state_capture=config.hidden_state_capture,
        attention_pathway=config.attention_pathway,
        attn_implementation=config.attn_implementation,
        covariance_mode=config.covariance_mode,
        covariance_low_rank=config.covariance_low_rank,
        progress_every=config.progress_every,
        length_bucketed_batches=config.length_bucketed_batches,
        offline=config.offline,
        auto_batch_size=config.auto_batch_size,
        cache_only=config.cache_only,
        **_cell_cache_config(config, cell_name=cell_name),
        dump_scores_format=config.dump_scores_format,
        refresh_scores=config.refresh_scores,
        signals=config.signals,
        signal=config.conformal_signal,
        repeats=config.conformal_repeats,
        seed=config.seed,
        artifact_alpha=config.artifact_alpha,
        multiple_testing_signals=config.multiple_testing_signals,
        multiple_testing_alpha=config.multiple_testing_alpha,
        multiple_testing_method=config.multiple_testing_method,
        best_by=config.best_by,
        python_executable=config.python_executable,
        dry_run=config.dry_run,
    )


def _cell_summary(
    cell_name: str,
    model: ModelSpec,
    scale: ScaleSpec,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = dict(report.get("evidence_bundle") or {})
    calibration = dict(evidence.get("calibration") or {})
    score_dump = dict(evidence.get("score_dump") or {})
    artifacts = dict(evidence.get("artifacts") or {})
    multiple_testing = dict(calibration.get("multiple_testing") or {})
    if multiple_testing:
        multiple_testing["report"] = artifacts.get("multiple_testing_report")
        multiple_testing["calibration"] = artifacts.get("multiple_testing_calibration")
    return {
        "name": cell_name,
        "model": {"name": model.name, "model_id": model.model_id},
        "scale": {
            "name": scale.name,
            "limit": scale.limit,
            "manifold_questions": scale.manifold_questions,
            "layer": scale.layer,
            "sweep_layers": tuple(scale.sweep_layers),
            "resolved_layer": _nested(report, "config", "layer"),
            "resolved_sweep_layers": _nested(report, "config", "sweep_layers"),
            "sweep_layers_source": _nested(report, "config", "sweep_layers_source"),
        },
        "status": report.get("status"),
        "workflow_report": _nested(report, "paths", "workflow_report"),
        "artifact_manifest": _nested(report, "paths", "artifact_manifest"),
        "score_dump": score_dump,
        "best": {
            "score_name": calibration.get("best_score_name"),
            "layer": calibration.get("best_layer"),
            "auroc": calibration.get("best_auroc"),
            "false_alarm": calibration.get("best_false_alarm"),
            "detection": calibration.get("best_detection"),
        },
        "multiple_testing": multiple_testing,
    }


def _cell_uses_band_report(config: TruthfulQAFrontierWorkflowConfig, *, scale: ScaleSpec) -> bool:
    if config.sweep_layers_from_band_report is None:
        return False
    if not config.sweep_band_scales:
        return True
    return scale.name in set(config.sweep_band_scales)


def _format_sweep_band_run_template(
    config: TruthfulQAFrontierWorkflowConfig,
    *,
    model: ModelSpec,
    scale: ScaleSpec,
) -> str:
    cell_name = f"{model.name}-{scale.name}"
    try:
        return config.sweep_band_run_template.format(
            cell=cell_name,
            model=model.name,
            model_id=model.model_id,
            scale=scale.name,
        )
    except KeyError as exc:
        raise ValueError(f"unknown sweep band run template key: {exc.args[0]}") from exc


def _workflow_status(
    config: TruthfulQAFrontierWorkflowConfig,
    cells: Sequence[Mapping[str, Any]],
    ensemble_payload: Mapping[str, Any] | None,
) -> str:
    if config.dry_run:
        return "needs_evidence"
    if any(cell.get("status") != "complete" for cell in cells):
        return "blocked"
    if ensemble_payload is None:
        return "blocked"
    for cell in cells:
        detectability = cell.get("detectability_taxonomy")
        if isinstance(detectability, Mapping) and detectability.get("status") not in {None, "complete"}:
            return "blocked"
    return "complete"


def _build_detectability_reports(
    config: TruthfulQAFrontierWorkflowConfig,
    cell_reports: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not _detectability_enabled(config):
        return []
    reports: list[dict[str, Any]] = []
    for cell in cell_reports:
        cell_name = str(cell["name"])
        output_path = _detectability_report_path(config, cell_name=cell_name)
        manifest_path = _detectability_manifest_path(config, cell_name=cell_name)
        summary: dict[str, Any] = {
            "path": str(output_path),
            "artifact_manifest": str(manifest_path),
            "consistency_signal": config.detectability_consistency_signal,
            "confidence_signal": config.detectability_confidence_signal,
            "consistency_direction": config.detectability_consistency_direction,
            "confidence_direction": config.detectability_confidence_direction,
        }
        if config.dry_run:
            summary["status"] = "planned"
        else:
            payload = build_detectability_taxonomy_report(
                score_dump_path=_nested(cell, "score_dump", "path"),
                consistency_signal=str(config.detectability_consistency_signal),
                confidence_signal=str(config.detectability_confidence_signal),
                consistency_direction=config.detectability_consistency_direction,
                confidence_direction=config.detectability_confidence_direction,
                metadata={"run_name": cell_name},
            )
            score_dump_path = Path(str(_nested(cell, "score_dump", "path")))
            payload["paths"] = {
                "detectability_taxonomy_report": str(output_path),
                "artifact_manifest": str(manifest_path),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            initial_manifest = write_detectability_artifact_manifest(
                manifest_path=manifest_path,
                output_path=output_path,
                score_dump_path=score_dump_path,
                payload=payload,
            )
            payload["artifact_manifest_summary"] = initial_manifest["summary"]
            output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest = write_detectability_artifact_manifest(
                manifest_path=manifest_path,
                output_path=output_path,
                score_dump_path=score_dump_path,
                payload=payload,
            )
            payload["artifact_manifest_summary"] = manifest["summary"]
            output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            false_distribution = _mapping(payload.get("report", {}).get("false_distribution"))
            entrenched = _mapping(false_distribution.get("entrenched"))
            summary.update({
                "status": payload.get("status"),
                "n_total": _nested(payload, "report", "n_total"),
                "n_false": _nested(payload, "report", "n_false"),
                "entrenched_false_count": entrenched.get("count"),
                "entrenched_false_rate": entrenched.get("rate"),
                "artifact_manifest_summary": manifest["summary"],
            })
        cell["detectability_taxonomy"] = summary
        reports.append({"cell": cell_name, **summary})
    return reports


def _detectability_enabled(config: TruthfulQAFrontierWorkflowConfig) -> bool:
    return bool(config.detectability_consistency_signal and config.detectability_confidence_signal)


def _detectability_report_path(config: TruthfulQAFrontierWorkflowConfig, *, cell_name: str) -> Path:
    return config.output_dir / cell_name / "detectability-taxonomy-report.json"


def _detectability_manifest_path(config: TruthfulQAFrontierWorkflowConfig, *, cell_name: str) -> Path:
    return config.output_dir / cell_name / "detectability-taxonomy-artifact-manifest.json"


def _detectability_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not reports:
        return None
    return {
        "report_count": len(reports),
        "reports": tuple(dict(report) for report in reports),
    }


def _multiple_testing_summary(
    config: TruthfulQAFrontierWorkflowConfig,
    cell_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not config.multiple_testing_enabled:
        return None
    cells = []
    pass_count = 0
    fail_count = 0
    unknown_count = 0
    for cell in cell_reports:
        multiple_testing = _mapping(cell.get("multiple_testing"))
        passed = multiple_testing.get("pass")
        if passed is True:
            pass_count += 1
        elif passed is False:
            fail_count += 1
        else:
            unknown_count += 1
        cells.append({
            "cell": cell.get("name"),
            "pass": passed,
            "false_alarm": multiple_testing.get("false_alarm"),
            "detection": multiple_testing.get("detection"),
            "report": multiple_testing.get("report"),
            "calibration": multiple_testing.get("calibration"),
        })
    return {
        "enabled": True,
        "signals": tuple(config.multiple_testing_signals),
        "alpha": config.resolved_multiple_testing_alpha,
        "method": config.multiple_testing_method,
        "cell_count": len(cells),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "unknown_count": unknown_count,
        "all_pass": bool(cells) and fail_count == 0 and unknown_count == 0,
        "cells": tuple(cells),
    }


def _ensemble_summary(payload: Mapping[str, Any] | None, *, path: Path) -> dict[str, Any] | None:
    if payload is None:
        return None
    runs = []
    for run in payload.get("runs", ()):
        if not isinstance(run, Mapping):
            continue
        runs.append({
            "name": run.get("name"),
            "best_single_at_alpha": run.get("best_single_at_alpha"),
            "best_ensemble_at_alpha": run.get("best_ensemble_at_alpha"),
        })
    return {
        "path": str(path),
        "signals": tuple(payload.get("signals", ())),
        "methods": tuple(payload.get("methods", ())),
        "best_alpha": payload.get("best_alpha"),
        "runs": tuple(runs),
    }


def _artifact_paths(
    config: TruthfulQAFrontierWorkflowConfig,
    cell_reports: Sequence[Mapping[str, Any]],
) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {
        "workflow_report": config.report_path,
        "score_ensemble_report": config.ensemble_report_path,
    }
    if config.sweep_layers_from_band_report is not None:
        artifacts["sweep_layer_band_report"] = config.sweep_layers_from_band_report
    for cell in cell_reports:
        name = str(cell["name"])
        artifacts[f"cells.{name}.workflow_report"] = cell.get("workflow_report")
        artifacts[f"cells.{name}.artifact_manifest"] = cell.get("artifact_manifest")
        artifacts[f"cells.{name}.score_dump"] = _nested(cell, "score_dump", "path")
        artifacts[f"cells.{name}.detectability_taxonomy_report"] = _nested(
            cell,
            "detectability_taxonomy",
            "path",
        )
        artifacts[f"cells.{name}.detectability_taxonomy_artifact_manifest"] = _nested(
            cell,
            "detectability_taxonomy",
            "artifact_manifest",
        )
        if config.multiple_testing_enabled:
            artifacts[f"cells.{name}.multiple_testing_report"] = _nested(
                cell,
                "multiple_testing",
                "report",
            )
            artifacts[f"cells.{name}.multiple_testing_calibration"] = _nested(
                cell,
                "multiple_testing",
                "calibration",
            )
    return artifacts


def _write_artifact_manifest(
    config: TruthfulQAFrontierWorkflowConfig,
    report: Mapping[str, Any],
    artifacts: Mapping[str, str | Path | None],
) -> dict[str, Any]:
    context = ArtifactVerificationContext()
    manifest = context.build_artifact_manifest(
        artifacts,
        root=config.output_dir,
        metadata={
            "runner": "run_truthfulqa_frontier_workflow",
            "status": report.get("status"),
            "dry_run": config.dry_run,
            "models": tuple(model.name for model in config.models),
            "scales": tuple(scale.name for scale in config.scales),
            "signals": tuple(config.signals),
            "ensemble_methods": tuple(config.ensemble_methods),
            "multiple_testing_enabled": config.multiple_testing_enabled,
            "multiple_testing_signals": tuple(config.multiple_testing_signals),
            "multiple_testing_alpha": config.resolved_multiple_testing_alpha,
            "multiple_testing_method": config.multiple_testing_method,
            "multiple_testing_all_pass": _nested(report, "multiple_testing_gate", "all_pass"),
            "detectability_consistency_signal": config.detectability_consistency_signal,
            "detectability_confidence_signal": config.detectability_confidence_signal,
            "detectability_consistency_direction": config.detectability_consistency_direction,
            "detectability_confidence_direction": config.detectability_confidence_direction,
            "sweep_layers_from_band_report": (
                None if config.sweep_layers_from_band_report is None else str(config.sweep_layers_from_band_report)
            ),
            "sweep_band_scales": tuple(config.sweep_band_scales),
            "sweep_band_expand_radius": config.sweep_band_expand_radius,
            "sweep_band_target_layer": config.sweep_band_target_layer,
        },
    )
    config.artifact_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _record_registry(config: TruthfulQAFrontierWorkflowConfig, report: Mapping[str, Any]) -> None:
    if config.registry_path is None or config.dry_run:
        return
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_report(
        name=str(config.name),
        path=config.report_path,
        version=str(config.version),
        metadata={
            "workflow": "run_truthfulqa_frontier_workflow",
            "status": report.get("status"),
            "artifact_manifest": str(config.artifact_manifest_path),
            "score_ensemble_report": str(config.ensemble_report_path),
            "models": tuple(model.name for model in config.models),
            "scales": tuple(scale.name for scale in config.scales),
            "signals": tuple(config.signals),
            "multiple_testing_enabled": config.multiple_testing_enabled,
            "multiple_testing_signals": tuple(config.multiple_testing_signals),
            "multiple_testing_alpha": config.resolved_multiple_testing_alpha,
            "multiple_testing_method": config.multiple_testing_method,
            "multiple_testing_all_pass": _nested(report, "multiple_testing_gate", "all_pass"),
            "detectability_consistency_signal": config.detectability_consistency_signal,
            "detectability_confidence_signal": config.detectability_confidence_signal,
            "sweep_layers_from_band_report": (
                None if config.sweep_layers_from_band_report is None else str(config.sweep_layers_from_band_report)
            ),
            "sweep_band_scales": tuple(config.sweep_band_scales),
        },
    )
    registry.save_json()


def _config_payload(config: TruthfulQAFrontierWorkflowConfig) -> dict[str, Any]:
    return {
        "models": tuple({"name": model.name, "model_id": model.model_id} for model in config.models),
        "scales": tuple({
            "name": scale.name,
            "limit": scale.limit,
            "manifold_questions": scale.manifold_questions,
            "layer": scale.layer,
            "sweep_layers": tuple(scale.sweep_layers),
        } for scale in config.scales),
        "dtype": config.dtype,
        "batch_size": config.batch_size,
        "max_batch_tokens": config.max_batch_tokens,
        "max_length": config.max_length,
        "hidden_state_capture": config.hidden_state_capture,
        "attention_pathway": config.attention_pathway,
        "attn_implementation": config.attn_implementation,
        "covariance_mode": config.covariance_mode,
        "covariance_low_rank": config.covariance_low_rank,
        "progress_every": config.progress_every,
        "length_bucketed_batches": config.length_bucketed_batches,
        "offline": config.offline,
        "auto_batch_size": config.auto_batch_size,
        "cache_only": config.cache_only,
        "cache_dir": None if config.cache_dir is None else str(config.cache_dir),
        "scores_path": None if config.scores_path is None else str(config.scores_path),
        "sweep_layers_from_band_report": (
            None if config.sweep_layers_from_band_report is None else str(config.sweep_layers_from_band_report)
        ),
        "sweep_band_strategy": config.sweep_band_strategy,
        "sweep_band_expand_radius": config.sweep_band_expand_radius,
        "sweep_band_target_layer": config.sweep_band_target_layer,
        "sweep_band_run_template": config.sweep_band_run_template,
        "sweep_band_scales": tuple(config.sweep_band_scales),
        "refresh_caches": config.refresh_caches,
        "warmup_checkpoint_every": config.warmup_checkpoint_every,
        "eval_reps_cache_shard_size": config.eval_reps_cache_shard_size,
        "eval_reps_shard_read_cache_size": config.eval_reps_shard_read_cache_size,
        "dump_scores_format": config.dump_scores_format,
        "refresh_scores": config.refresh_scores,
        "signals": tuple(config.signals),
        "conformal_signal": config.conformal_signal,
        "conformal_repeats": config.conformal_repeats,
        "ensemble_repeats": config.ensemble_repeats,
        "seed": config.seed,
        "artifact_alpha": config.artifact_alpha,
        "multiple_testing_signals": tuple(config.multiple_testing_signals),
        "multiple_testing_alpha": config.resolved_multiple_testing_alpha,
        "multiple_testing_method": config.multiple_testing_method,
        "best_alpha": config.best_alpha,
        "best_by": config.best_by,
        "ensemble_methods": tuple(config.ensemble_methods),
        "alphas": tuple(config.alphas),
        "detectability_consistency_signal": config.detectability_consistency_signal,
        "detectability_confidence_signal": config.detectability_confidence_signal,
        "detectability_consistency_direction": config.detectability_consistency_direction,
        "detectability_confidence_direction": config.detectability_confidence_direction,
        "dry_run": config.dry_run,
    }


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _cell_cache_config(config: TruthfulQAFrontierWorkflowConfig, *, cell_name: str) -> dict[str, Any]:
    if config.cache_dir is None:
        return {}
    cache_root = config.cache_dir / cell_name
    return {
        "statement_encoding_cache": cache_root / "statement-encodings.json",
        "refresh_statement_encoding_cache": config.refresh_caches,
        "layer_stats_cache": cache_root / "layer-stats.pt",
        "refresh_layer_stats_cache": config.refresh_caches,
        "warmup_checkpoint": cache_root / "warmup-checkpoint.pt",
        "warmup_checkpoint_every": config.warmup_checkpoint_every,
        "eval_reps_cache": cache_root / "eval-reps-cache",
        "eval_reps_cache_shard_size": config.eval_reps_cache_shard_size,
        "eval_reps_shard_read_cache_size": config.eval_reps_shard_read_cache_size,
        "refresh_eval_reps_cache": config.refresh_caches,
    }


def _parse_model(value: str) -> ModelSpec:
    if "=" not in value:
        model_id = value.strip()
        return ModelSpec(name=_slug(model_id), model_id=model_id)
    name, model_id = value.split("=", 1)
    return ModelSpec(name=name, model_id=model_id)


def _parse_scale(value: str) -> ScaleSpec:
    if "=" not in value:
        raise ValueError("--scale must be formatted as name=limit:manifold_questions:layer:sweep_layers.")
    name, raw = value.split("=", 1)
    parts = raw.split(":", 3)
    if len(parts) != 4:
        raise ValueError("--scale must be formatted as name=limit:manifold_questions:layer:sweep_layers.")
    sweep_layers = tuple(int(part.strip()) for part in parts[3].split(",") if part.strip())
    return ScaleSpec(
        name=name,
        limit=int(parts[0]),
        manifold_questions=int(parts[1]),
        layer=int(parts[2]),
        sweep_layers=sweep_layers,
    )


def _parse_csv(value: str | None, *, name: str) -> tuple[str, ...]:
    if value is None:
        raise ValueError(f"{name} is required.")
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError(f"{name} must contain at least one value.")
    return parts


def _parse_float_csv(value: str | None, *, name: str) -> tuple[float, ...]:
    return tuple(float(part) for part in _parse_csv(value, name=name))


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-") or "model"


def _config_from_args(args: argparse.Namespace) -> TruthfulQAFrontierWorkflowConfig:
    models = tuple(_parse_model(value) for value in (args.model or DEFAULT_FRONTIER_MODELS))
    scales = tuple(_parse_scale(value) for value in (args.scale or DEFAULT_FRONTIER_SCALES))
    return TruthfulQAFrontierWorkflowConfig(
        output_dir=Path(args.output_dir),
        models=models,
        scales=scales,
        registry_path=Path(args.registry) if args.registry else None,
        name=args.name,
        version=args.version,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        scores_path=Path(args.scores) if args.scores else None,
        sweep_layers_from_band_report=(
            Path(args.sweep_layers_from_band_report) if args.sweep_layers_from_band_report else None
        ),
        sweep_band_strategy=args.sweep_band_strategy,
        sweep_band_expand_radius=args.sweep_band_expand_radius,
        sweep_band_target_layer=args.sweep_band_target_layer,
        sweep_band_run_template=args.sweep_band_run_template,
        sweep_band_scales=_parse_csv(args.sweep_band_scales, name="--sweep-band-scales")
        if args.sweep_band_scales
        else (),
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_length=args.max_length,
        hidden_state_capture=args.hidden_state_capture,
        attention_pathway=args.attention_pathway,
        attn_implementation=args.attn_implementation,
        covariance_mode=args.covariance_mode,
        covariance_low_rank=args.covariance_low_rank,
        progress_every=args.progress_every,
        length_bucketed_batches=not args.no_length_bucketed_batches,
        offline=args.offline,
        auto_batch_size=not args.no_auto_batch_size,
        cache_only=args.cache_only,
        refresh_caches=args.refresh_caches,
        warmup_checkpoint_every=args.warmup_checkpoint_every,
        eval_reps_cache_shard_size=args.eval_reps_cache_shard_size,
        eval_reps_shard_read_cache_size=args.eval_reps_shard_read_cache_size,
        dump_scores_format=args.dump_scores_format,
        refresh_scores=args.refresh_scores,
        signals=_parse_csv(args.signals, name="--signals"),
        conformal_signal=args.conformal_signal,
        conformal_repeats=args.conformal_repeats,
        ensemble_repeats=args.ensemble_repeats,
        seed=args.seed,
        artifact_alpha=args.artifact_alpha,
        multiple_testing_signals=_parse_csv(args.multiple_testing_signals, name="--multiple-testing-signals")
        if args.multiple_testing_signals
        else (),
        multiple_testing_alpha=args.multiple_testing_alpha,
        multiple_testing_method=args.multiple_testing_method,
        best_alpha=args.best_alpha,
        best_by=args.best_by,
        ensemble_methods=_parse_csv(args.methods, name="--methods"),
        alphas=_parse_float_csv(args.alphas, name="--alphas"),
        detectability_consistency_signal=args.detectability_consistency_signal,
        detectability_confidence_signal=args.detectability_confidence_signal,
        detectability_consistency_direction=args.detectability_consistency_direction,
        detectability_confidence_direction=args.detectability_confidence_direction,
        python_executable=args.python,
        clean=args.clean,
        dry_run=args.dry_run,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = run_truthfulqa_frontier_workflow(_config_from_args(args))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_blocked and report["status"] == "blocked":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run multi-model TruthfulQA frontier workflow")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", action="append",
                        help="model entry formatted as name=model_id; repeatable")
    parser.add_argument(
        "--scale",
        action="append",
        help="scale formatted as name=limit:manifold_questions:layer:sweep_layers; repeatable",
    )
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--cache-dir", default=None,
                        help="optional root for per-cell statement/layer/eval cache artifacts")
    parser.add_argument(
        "--scores",
        default=None,
        help="reuse this score dump for a single-cell frontier rerun instead of materializing scores",
    )
    parser.add_argument(
        "--sweep-layers-from-band-report",
        default=None,
        help="optional compare_layer_band_selectors.py report used to derive per-cell sweep layers",
    )
    parser.add_argument(
        "--sweep-band-strategy",
        default=None,
        help="strategy name from the layer-band report; defaults to the report recommended_strategy",
    )
    parser.add_argument(
        "--sweep-band-expand-radius",
        type=int,
        default=0,
        help="expand each selected candidate layer by +/- this many hidden-state indexes",
    )
    parser.add_argument(
        "--sweep-band-target-layer",
        choices=("best", "band_best", "first"),
        default=None,
        help="set each cell target --layer from the selected layer-band report run",
    )
    parser.add_argument(
        "--sweep-band-run-template",
        default="{cell}",
        help="format string for matching report run names; keys: cell, model, model_id, scale",
    )
    parser.add_argument(
        "--sweep-band-scales",
        default=None,
        help="comma-list of scale names that should use the layer-band report; default applies to all scales",
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--hidden-state-capture", choices=("outputs", "hooks"), default="hooks")
    parser.add_argument("--attention-pathway", action="store_true",
                        help="request optional attention-pathway score columns in each cell")
    parser.add_argument("--attn-implementation", default=None,
                        help="optional Transformers attention backend, e.g. eager for output_attentions support")
    parser.add_argument("--covariance-mode", choices=("full", "diag", "low_rank", "shrinkage"),
                        default="full")
    parser.add_argument("--covariance-low-rank", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--no-length-bucketed-batches", action="store_true")
    parser.add_argument("--offline", action="store_true",
                        help="use eval_truthfulqa.py --offline for bounded fixture runs")
    parser.add_argument("--no-auto-batch-size", action="store_true")
    parser.add_argument("--cache-only", action="store_true",
                        help="score from existing per-cell caches under --cache-dir")
    parser.add_argument("--refresh-caches", action="store_true",
                        help="rebuild per-cell caches under --cache-dir")
    parser.add_argument("--warmup-checkpoint-every", type=int, default=50)
    parser.add_argument("--eval-reps-cache-shard-size", type=int, default=16)
    parser.add_argument("--eval-reps-shard-read-cache-size", type=int, default=2)
    parser.add_argument("--dump-scores-format", choices=("json", "jsonl"), default="jsonl")
    parser.add_argument("--refresh-scores", action="store_true")
    parser.add_argument("--signals", default=",".join(DEFAULT_FRONTIER_SIGNALS))
    parser.add_argument("--conformal-signal", default="maha_last")
    parser.add_argument("--conformal-repeats", type=int, default=20)
    parser.add_argument("--ensemble-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifact-alpha", type=float, default=0.10)
    parser.add_argument(
        "--multiple-testing-signals",
        default=None,
        help=(
            "optional comma-list forwarded to each calibrated cell for a multi-signal conformal "
            "gate report/runtime artifact"
        ),
    )
    parser.add_argument(
        "--multiple-testing-alpha",
        type=float,
        default=None,
        help="optional false-alarm budget for --multiple-testing-signals; defaults to --artifact-alpha",
    )
    parser.add_argument(
        "--multiple-testing-method",
        choices=("by", "bh", "bonferroni"),
        default="by",
        help="multiple-testing correction forwarded to each calibrated cell",
    )
    parser.add_argument("--best-alpha", type=float, default=0.10)
    parser.add_argument("--best-by", choices=("auroc", "detection"), default="auroc")
    parser.add_argument("--methods", default=",".join(DEFAULT_ENSEMBLE_METHODS))
    parser.add_argument("--alphas", default=",".join(str(alpha) for alpha in DEFAULT_ALPHAS))
    parser.add_argument("--detectability-consistency-signal", default=None)
    parser.add_argument("--detectability-confidence-signal", default=None)
    parser.add_argument("--detectability-consistency-direction", choices=("higher", "lower"), default="higher")
    parser.add_argument("--detectability-confidence-direction", choices=("higher", "lower"), default="higher")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
