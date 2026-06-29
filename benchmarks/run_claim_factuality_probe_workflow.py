"""Run the claim factuality probe evidence workflow end to end.

This workflow closes the benchmark-side path for claim-level hidden-state
factuality probing:

1. optionally run ``eval_truthfulqa.py`` to export candidate-claim hidden-state
   records from a forced-answer forward pass;
2. run ``eval_claim_factuality_probe.py`` in layer-sweep mode to select a probe
   layer and save the recommended probe plus split-conformal calibration;
3. run a cheap text redline baseline over the same records;
4. write a compact workflow report, artifact manifest, and optional registry
   records.

The workflow can reuse an existing records file. That is the preferred path for
iterating on probe training without rerunning a model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_TRUTHFULQA_SCRIPT = Path("benchmarks") / "eval_truthfulqa.py"
EVAL_CLAIM_FACTUALITY_PROBE_SCRIPT = Path("benchmarks") / "eval_claim_factuality_probe.py"
EVAL_TEXT_BASELINE_SCRIPT = Path("benchmarks") / "eval_pre_generation_text_baselines.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import planned_artifact_manifest_summary  # noqa: E402
from benchmarks.eval_claim_factuality_probe import CLAIM_FACTUALITY_SWEEP_BEST_BY  # noqa: E402
from benchmarks.eval_pre_generation_text_baselines import DEFAULT_TEXT_BASELINE_SIGNALS  # noqa: E402
from eigentruth.json_utils import to_jsonable  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

DEFAULT_CLAIM_FACTUALITY_LAYERS = (-1, -2)
DEFAULT_SWEEP_LAYERS: tuple[int, ...] | None = None
HIDDEN_STATE_CAPTURE_MODES = ("outputs", "hooks")
POOLING_MODES = ("mean", "first_token", "last_token")


@dataclass(frozen=True)
class ClaimFactualityProbeWorkflowConfig:
    """Configuration for the claim factuality probe evidence workflow."""

    output_dir: str | Path
    records_path: str | Path | None = None
    report_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    register_name: str | None = None
    register_version: str = "0.1"
    model: str = "sshleifer/tiny-gpt2"
    dtype: str = "float32"
    layer: int = -1
    limit: int = 12
    manifold_questions: int = 6
    max_length: int = 64
    batch_size: int = 1
    max_batch_tokens: int = 0
    hidden_state_capture: str = "hooks"
    claim_factuality_layers: Sequence[int] = DEFAULT_CLAIM_FACTUALITY_LAYERS
    offline: bool = True
    auto_batch_size: bool = False
    length_bucketed_batches: bool = True
    progress_every: int = 0
    refresh_records: bool = False
    sweep_layers: Sequence[int] | None = DEFAULT_SWEEP_LAYERS
    best_by: str = "auto"
    train_fraction: float = 0.7
    seed: int = 0
    pooling: str = "mean"
    steps: int = 300
    lr: float = 0.05
    l2: float = 1e-4
    conformal_alpha: float = 0.1
    soft_target_cutoff: float | None = None
    calibration_model_id: str = "claim_factuality_probe"
    run_text_baseline: bool = True
    baseline_signals: Sequence[str] = DEFAULT_TEXT_BASELINE_SIGNALS
    python_executable: str = sys.executable
    clean: bool = False
    dry_run: bool = False
    compact_json: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for field_name in ("records_path", "report_path", "artifact_manifest_path", "registry_path"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, Path(value))
        object.__setattr__(self, "layer", int(self.layer))
        object.__setattr__(self, "limit", int(self.limit))
        object.__setattr__(self, "manifold_questions", int(self.manifold_questions))
        object.__setattr__(self, "max_length", int(self.max_length))
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "max_batch_tokens", int(self.max_batch_tokens))
        object.__setattr__(
            self,
            "claim_factuality_layers",
            tuple(int(layer) for layer in self.claim_factuality_layers),
        )
        object.__setattr__(
            self,
            "sweep_layers",
            None if self.sweep_layers is None else tuple(int(layer) for layer in self.sweep_layers),
        )
        object.__setattr__(self, "train_fraction", float(self.train_fraction))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "steps", int(self.steps))
        object.__setattr__(self, "lr", float(self.lr))
        object.__setattr__(self, "l2", float(self.l2))
        object.__setattr__(self, "conformal_alpha", float(self.conformal_alpha))
        if self.soft_target_cutoff is not None:
            object.__setattr__(self, "soft_target_cutoff", float(self.soft_target_cutoff))
        object.__setattr__(self, "progress_every", int(self.progress_every))
        signals = tuple(str(signal) for signal in self.baseline_signals if str(signal))
        if not signals:
            raise ValueError("baseline_signals must not be empty.")
        if len(set(signals)) != len(signals):
            raise ValueError("baseline_signals must be unique.")
        object.__setattr__(self, "baseline_signals", signals)
        if self.register_name is not None:
            register_name = str(self.register_name).strip()
            if not register_name:
                raise ValueError("register_name must be non-empty when provided.")
            object.__setattr__(self, "register_name", register_name)
        object.__setattr__(self, "register_version", str(self.register_version))
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("dtype must be one of: float32, bfloat16, float16.")
        if self.hidden_state_capture not in HIDDEN_STATE_CAPTURE_MODES:
            raise ValueError(f"hidden_state_capture must be one of: {', '.join(HIDDEN_STATE_CAPTURE_MODES)}.")
        if self.best_by not in CLAIM_FACTUALITY_SWEEP_BEST_BY:
            raise ValueError(f"best_by must be one of: {', '.join(CLAIM_FACTUALITY_SWEEP_BEST_BY)}.")
        if self.pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of: {', '.join(POOLING_MODES)}.")
        if not self.claim_factuality_layers:
            raise ValueError("claim_factuality_layers must not be empty.")
        if self.limit < 0:
            raise ValueError("limit must be >=0.")
        if self.manifold_questions < 1:
            raise ValueError("manifold_questions must be >=1.")
        if self.max_length < 1:
            raise ValueError("max_length must be >=1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >=1.")
        if self.max_batch_tokens < 0:
            raise ValueError("max_batch_tokens must be >=0.")
        if not (0.0 < self.train_fraction < 1.0):
            raise ValueError("train_fraction must be in (0, 1).")
        if self.steps < 1:
            raise ValueError("steps must be >=1.")
        if self.lr <= 0.0:
            raise ValueError("lr must be >0.")
        if self.l2 < 0.0:
            raise ValueError("l2 must be >=0.")
        if not (0.0 < self.conformal_alpha < 1.0):
            raise ValueError("conformal_alpha must be in (0, 1).")
        if self.soft_target_cutoff is not None and not (0.0 <= self.soft_target_cutoff <= 1.0):
            raise ValueError("soft_target_cutoff must be in [0, 1].")
        if self.progress_every < 0:
            raise ValueError("progress_every must be >=0.")
        if self.clean and self.records_path is not None and Path(self.records_path).exists():
            raise ValueError("clean cannot be used with an existing external records_path.")
        if self.register_name is not None and self.registry_path is None:
            raise ValueError("register_name requires registry_path.")

    @property
    def resolved_records_path(self) -> Path:
        return Path(self.records_path) if self.records_path is not None else self.output_dir / "records.jsonl"

    @property
    def truthfulqa_report_path(self) -> Path:
        return self.output_dir / "truthfulqa-claim-factuality-export.json"

    @property
    def probe_report_path(self) -> Path:
        return self.output_dir / "claim-factuality-probe-layer-sweep.json"

    @property
    def text_baseline_report_path(self) -> Path:
        return self.output_dir / "claim-factuality-text-baseline.json"

    @property
    def text_baseline_manifest_path(self) -> Path:
        return self.output_dir / "claim-factuality-text-baseline-manifest.json"

    @property
    def best_probe_path(self) -> Path:
        return self.output_dir / "best-claim-factuality-probe.pt"

    @property
    def best_calibration_path(self) -> Path:
        return self.output_dir / "best-claim-factuality-calibration.json"

    @property
    def resolved_report_path(self) -> Path:
        if self.report_path is not None:
            return Path(self.report_path)
        return self.output_dir / "claim-factuality-probe-workflow.json"

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return self.output_dir / "artifact-manifest.json"


def run_claim_factuality_probe_workflow(config: ClaimFactualityProbeWorkflowConfig) -> dict[str, Any]:
    """Run or plan the claim factuality probe workflow."""
    started_at = time.perf_counter()
    if config.clean and config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    records_reused = config.resolved_records_path.exists() and not config.refresh_records
    truthfulqa_command = _truthfulqa_command(config)
    probe_command = _probe_command(config)
    text_baseline_command = _text_baseline_command(config)
    truthfulqa_payload = None
    probe_payload = None
    text_baseline_payload = None
    if not config.dry_run:
        if not records_reused:
            _run_command(truthfulqa_command)
        _run_command(probe_command)
        if config.run_text_baseline:
            _run_command(text_baseline_command)
        truthfulqa_payload = _load_json_if_exists(config.truthfulqa_report_path)
        probe_payload = _load_json_if_exists(config.probe_report_path)
        text_baseline_payload = _load_json_if_exists(config.text_baseline_report_path)
    records_summary = _records_summary(config.resolved_records_path)
    effective_model = _effective_model(
        config,
        truthfulqa_payload=truthfulqa_payload,
        records_summary=records_summary,
    )
    redline = _redline_summary(
        probe_payload=probe_payload,
        text_baseline_payload=text_baseline_payload,
        run_text_baseline=config.run_text_baseline,
    )
    registry_record_key = (
        None if config.register_name is None else f"report:{config.register_name}:{config.register_version}"
    )
    registry_manifest_record_key = (
        None
        if config.register_name is None
        else f"benchmark_manifest:{config.register_name}:{config.register_version}"
    )

    artifacts = _artifact_paths(
        config,
        include_truthfulqa_report=(not records_reused or config.truthfulqa_report_path.exists()),
    )
    artifact_manifest_summary = planned_artifact_manifest_summary(
        artifacts,
        assume_file_paths=(config.resolved_report_path,) if not config.dry_run else (),
    )
    report = {
        "schema_version": 1,
        "workflow": "claim_factuality_probe_workflow",
        "status": _workflow_status(
            dry_run=config.dry_run,
            probe_payload=probe_payload,
            text_baseline_payload=text_baseline_payload,
            run_text_baseline=config.run_text_baseline,
        ),
        "paths": _paths_payload(config),
        "config": _config_payload(config),
        "effective_model": effective_model,
        "records": records_summary,
        "execution": {
            "records_reused": records_reused,
            "truthfulqa_command": truthfulqa_command,
            "probe_command": probe_command,
            "text_baseline_command": text_baseline_command if config.run_text_baseline else None,
            "wall_clock_seconds": time.perf_counter() - started_at,
        },
        "truthfulqa": _truthfulqa_summary(truthfulqa_payload),
        "probe": _probe_summary(probe_payload),
        "redline": redline,
        "artifact_manifest_summary": artifact_manifest_summary,
        "registry_record": registry_record_key,
        "registry_manifest_record": registry_manifest_record_key,
        "evidence_scope": {
            "claim": "claim factuality probe workflow reproducibility and text-redline comparison",
            "not_a_claim": "model-level hallucination detector superiority",
            "notes": (
                "Candidate-claim TruthfulQA records validate the hidden-state probe handoff. Promote "
                "detector-quality claims only after larger held-out multi-model evaluation and redline gates."
            ),
        },
    }
    _write_json(config.resolved_report_path, report, compact=config.compact_json)
    manifest = None
    if not config.dry_run:
        manifest = build_artifact_manifest(
            artifacts,
            root=config.resolved_artifact_manifest_path.parent,
            metadata={
                "workflow": "claim_factuality_probe_workflow",
                "status": report["status"],
                "records_reused": records_reused,
                "effective_model": effective_model,
                "recommended_layer": _nested(probe_payload, "recommended", "layer"),
                "candidate_count": _nested(probe_payload, "candidate_count"),
                "redline_margin": redline.get("probe_vs_text_auroc_margin"),
                "text_redline_best_signal": _nested(text_baseline_payload, "best_signal", "name"),
            },
        )
        _write_json(config.resolved_artifact_manifest_path, manifest, compact=False)
    if not config.dry_run and config.registry_path is not None and config.register_name is not None:
        _record_registry(config, report=report, manifest=manifest)
    print(
        "claim_factuality_probe_workflow_ok "
        f"status={report['status']} "
        f"records_reused={records_reused} "
        f"recommended_layer={_nested(probe_payload, 'recommended', 'layer')} "
        f"redline_margin={redline.get('probe_vs_text_auroc_margin')} "
        f"output={config.resolved_report_path}"
    )
    return to_jsonable(report)


def _truthfulqa_command(config: ClaimFactualityProbeWorkflowConfig) -> list[str]:
    command = [
        str(config.python_executable),
        str(EVAL_TRUTHFULQA_SCRIPT),
        "--model",
        config.model,
        "--dtype",
        config.dtype,
        "--layer",
        str(config.layer),
        "--limit",
        str(config.limit),
        "--manifold-questions",
        str(config.manifold_questions),
        "--max-length",
        str(config.max_length),
        "--batch-size",
        str(config.batch_size),
        "--max-batch-tokens",
        str(config.max_batch_tokens),
        "--hidden-state-capture",
        config.hidden_state_capture,
        "--progress-every",
        str(config.progress_every),
        "--json",
        str(config.truthfulqa_report_path),
        "--dump-claim-factuality-probe-records",
        str(config.resolved_records_path),
        f"--claim-factuality-probe-layers={_comma_ints(config.claim_factuality_layers)}",
        "--seed",
        str(config.seed),
    ]
    if config.offline:
        command.append("--offline")
    if config.auto_batch_size:
        command.append("--auto-batch-size")
    if config.length_bucketed_batches:
        command.append("--length-bucketed-batches")
    return command


def _probe_command(config: ClaimFactualityProbeWorkflowConfig) -> list[str]:
    sweep_layers = "auto" if config.sweep_layers is None else _comma_ints(config.sweep_layers)
    command = [
        str(config.python_executable),
        str(EVAL_CLAIM_FACTUALITY_PROBE_SCRIPT),
        "--records",
        str(config.resolved_records_path),
        "--json",
        str(config.probe_report_path),
        f"--sweep-layers={sweep_layers}",
        "--best-by",
        config.best_by,
        "--save-artifact",
        str(config.best_probe_path),
        "--save-calibration",
        str(config.best_calibration_path),
        "--calibration-model-id",
        config.calibration_model_id,
        "--conformal-alpha",
        str(config.conformal_alpha),
        "--train-fraction",
        str(config.train_fraction),
        "--seed",
        str(config.seed),
        "--pooling",
        config.pooling,
        "--steps",
        str(config.steps),
        "--lr",
        str(config.lr),
        "--l2",
        str(config.l2),
    ]
    if config.soft_target_cutoff is not None:
        command.extend(["--soft-target-cutoff", str(config.soft_target_cutoff)])
    if config.compact_json:
        command.append("--compact-json")
    return command


def _text_baseline_command(config: ClaimFactualityProbeWorkflowConfig) -> list[str]:
    command = [
        str(config.python_executable),
        str(EVAL_TEXT_BASELINE_SCRIPT),
        "--records",
        str(config.resolved_records_path),
        "--json",
        str(config.text_baseline_report_path),
        "--artifact-manifest",
        str(config.text_baseline_manifest_path),
        "--baseline-signals",
        ",".join(config.baseline_signals),
    ]
    if config.compact_json:
        command.append("--compact-json")
    return command


def _artifact_paths(
    config: ClaimFactualityProbeWorkflowConfig,
    *,
    include_truthfulqa_report: bool,
) -> dict[str, Path]:
    artifacts = {
        "workflow_report": config.resolved_report_path,
        "records": config.resolved_records_path,
        "probe_report": config.probe_report_path,
        "best_probe": config.best_probe_path,
        "best_calibration": config.best_calibration_path,
    }
    if config.run_text_baseline:
        artifacts["text_baseline_report"] = config.text_baseline_report_path
        artifacts["text_baseline_manifest"] = config.text_baseline_manifest_path
    if include_truthfulqa_report:
        artifacts["truthfulqa_report"] = config.truthfulqa_report_path
    return artifacts


def _paths_payload(config: ClaimFactualityProbeWorkflowConfig) -> dict[str, str | None]:
    return {
        "workflow_report": str(config.resolved_report_path),
        "artifact_manifest": str(config.resolved_artifact_manifest_path),
        "registry": None if config.registry_path is None else str(config.registry_path),
        "truthfulqa_report": str(config.truthfulqa_report_path),
        "records": str(config.resolved_records_path),
        "probe_report": str(config.probe_report_path),
        "text_baseline_report": str(config.text_baseline_report_path),
        "text_baseline_manifest": str(config.text_baseline_manifest_path),
        "best_probe": str(config.best_probe_path),
        "best_calibration": str(config.best_calibration_path),
    }


def _config_payload(config: ClaimFactualityProbeWorkflowConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "dtype": config.dtype,
        "layer": config.layer,
        "limit": config.limit,
        "manifold_questions": config.manifold_questions,
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "max_batch_tokens": config.max_batch_tokens,
        "hidden_state_capture": config.hidden_state_capture,
        "claim_factuality_layers": tuple(config.claim_factuality_layers),
        "offline": config.offline,
        "auto_batch_size": config.auto_batch_size,
        "length_bucketed_batches": config.length_bucketed_batches,
        "progress_every": config.progress_every,
        "refresh_records": config.refresh_records,
        "sweep_layers": "auto" if config.sweep_layers is None else tuple(config.sweep_layers),
        "best_by": config.best_by,
        "train_fraction": config.train_fraction,
        "seed": config.seed,
        "pooling": config.pooling,
        "steps": config.steps,
        "lr": config.lr,
        "l2": config.l2,
        "conformal_alpha": config.conformal_alpha,
        "soft_target_cutoff": config.soft_target_cutoff,
        "calibration_model_id": config.calibration_model_id,
        "run_text_baseline": config.run_text_baseline,
        "baseline_signals": tuple(config.baseline_signals),
        "register_name": config.register_name,
        "register_version": config.register_version,
        "clean": config.clean,
        "dry_run": config.dry_run,
    }


def _truthfulqa_summary(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    auroc = _mapping(payload.get("auroc"))
    best_signal = None
    if auroc:
        best_name, best_value = max(
            ((name, float(value)) for name, value in auroc.items() if _is_number(value)),
            key=lambda item: item[1],
            default=(None, None),
        )
        if best_name is not None:
            best_signal = {"name": best_name, "auroc": best_value}
    return {
        "model": _nested(payload, "config", "model"),
        "n_pos": _nested(payload, "config", "n_pos"),
        "n_neg": _nested(payload, "config", "n_neg"),
        "records": _mapping(payload.get("claim_factuality_probe_records")),
        "best_signal": best_signal,
    }


def _records_summary(path: str | Path) -> dict[str, Any] | None:
    records_path = Path(path)
    if not records_path.exists():
        return None
    first_record: Mapping[str, Any] | None = None
    record_count = 0
    if records_path.suffix.lower() == ".jsonl":
        with records_path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"records JSONL line {line_number} must be an object.")
                if first_record is None:
                    first_record = payload
                record_count += 1
    else:
        payload = json.loads(records_path.read_text(encoding="utf-8"))
        raw_records = payload.get("records") if isinstance(payload, Mapping) else payload
        if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
            raise ValueError("records JSON must be a list or an object with a records list.")
        record_count = len(raw_records)
        if record_count:
            first = raw_records[0]
            if not isinstance(first, Mapping):
                raise ValueError("records JSON entries must be objects.")
            first_record = first
    metadata = _mapping(first_record.get("metadata")) if first_record is not None else {}
    return {
        "path": str(records_path),
        "record_count": int(record_count),
        "metadata_model": metadata.get("model"),
        "metadata_dataset": metadata.get("dataset"),
        "metadata_layers": metadata.get("layers"),
        "metadata_record_grain": metadata.get("record_grain"),
        "metadata_offline": metadata.get("offline"),
        "metadata_source": metadata.get("source"),
    }


def _effective_model(
    config: ClaimFactualityProbeWorkflowConfig,
    *,
    truthfulqa_payload: Mapping[str, Any] | None,
    records_summary: Mapping[str, Any] | None,
) -> str:
    truthfulqa_model = _nested(truthfulqa_payload, "config", "model")
    if truthfulqa_model:
        return str(truthfulqa_model)
    records_model = _nested(records_summary, "metadata_model")
    if records_model:
        return str(records_model)
    return str(config.model)


def _probe_summary(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    recommended = _mapping(payload.get("recommended"))
    metrics = _mapping(recommended.get("metrics"))
    test_metrics = _mapping(metrics.get("test"))
    conformal = _mapping(recommended.get("conformal"))
    test_selective = _mapping(conformal.get("test_selective"))
    return {
        "candidate_count": payload.get("candidate_count"),
        "resolved_best_by": _nested(payload, "config", "resolved_best_by"),
        "recommended_layer": recommended.get("layer"),
        "selection_metric": recommended.get("selection_metric"),
        "selection_value": recommended.get("selection_raw_value"),
        "test_label_auroc": test_metrics.get("label_auroc"),
        "test_target_bce": test_metrics.get("target_bce"),
        "conformal_available": conformal.get("available"),
        "conformal_threshold": conformal.get("threshold"),
        "test_selective_accuracy": test_selective.get("accuracy"),
        "test_selective_coverage": test_selective.get("coverage"),
        "saved_recommended": _mapping(payload.get("saved_recommended")).get("paths"),
    }


def _redline_summary(
    *,
    probe_payload: Mapping[str, Any] | None,
    text_baseline_payload: Mapping[str, Any] | None,
    run_text_baseline: bool,
) -> dict[str, Any]:
    if not run_text_baseline:
        return {"available": False, "reason": "disabled"}
    if text_baseline_payload is None:
        return {"available": False, "reason": "missing_text_baseline_report"}
    best_signal = _mapping(text_baseline_payload.get("best_signal"))
    probe_auc = _nested(probe_payload, "recommended", "metrics", "test", "label_auroc")
    text_auc = best_signal.get("auroc")
    margin = None
    if _is_number(probe_auc) and _is_number(text_auc):
        margin = float(probe_auc) - float(text_auc)
    return {
        "available": True,
        "workflow": text_baseline_payload.get("workflow"),
        "record_count": text_baseline_payload.get("record_count"),
        "best_text_signal": best_signal.get("name"),
        "best_text_direction": best_signal.get("direction"),
        "best_text_auroc": best_signal.get("auroc"),
        "probe_test_label_auroc": probe_auc,
        "probe_vs_text_auroc_margin": margin,
        "status": (
            "pass"
            if margin is not None and margin >= 0.0
            else "observed"
            if margin is not None
            else "unknown"
        ),
    }


def _workflow_status(
    *,
    dry_run: bool,
    probe_payload: Mapping[str, Any] | None,
    text_baseline_payload: Mapping[str, Any] | None,
    run_text_baseline: bool,
) -> str:
    if dry_run:
        return "planned"
    if not isinstance(probe_payload, Mapping):
        return "partial"
    if _nested(probe_payload, "recommended", "layer") is None:
        return "blocked"
    if _nested(probe_payload, "saved_recommended", "paths", "calibration") is None:
        return "partial"
    if run_text_baseline and not isinstance(text_baseline_payload, Mapping):
        return "partial"
    return "ready"


def _record_registry(
    config: ClaimFactualityProbeWorkflowConfig,
    *,
    report: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> None:
    if config.registry_path is None or config.register_name is None:
        return
    metadata = {
        "workflow": report.get("workflow"),
        "status": report.get("status"),
        "effective_model": report.get("effective_model"),
        "record_count": _nested(report, "records", "record_count"),
        "recommended_layer": _nested(report, "probe", "recommended_layer"),
        "test_label_auroc": _nested(report, "probe", "test_label_auroc"),
        "redline_margin": _nested(report, "redline", "probe_vs_text_auroc_margin"),
        "text_redline_best_signal": _nested(report, "redline", "best_text_signal"),
        "manifest_summary": None if manifest is None else manifest.get("summary"),
    }
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_report(
        name=config.register_name,
        path=config.resolved_report_path,
        version=config.register_version,
        metadata=metadata,
    )
    if manifest is not None:
        registry.record_benchmark_manifest(
            name=config.register_name,
            path=config.resolved_artifact_manifest_path,
            version=config.register_version,
            metadata=metadata,
        )
    registry.save_json()


def _run_command(command: Sequence[str]) -> None:
    subprocess.run(list(command), cwd=REPO_ROOT, check=True)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    indent = None if compact else 2
    output.write_text(json.dumps(to_jsonable(payload), indent=indent, sort_keys=True) + "\n", encoding="utf-8")


def _load_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    payload_path = Path(path)
    if not payload_path.exists():
        return None
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {payload_path}")
    return dict(payload)


def _parse_int_list(value: str | None, *, allow_auto: bool = False) -> tuple[int, ...] | None:
    if value is None:
        return None
    stripped = value.strip()
    if allow_auto and stripped.casefold() == "auto":
        return None
    if not stripped:
        raise ValueError("layer list must not be empty.")
    return tuple(int(part.strip()) for part in stripped.split(",") if part.strip())


def _parse_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    if not items:
        raise ValueError("CSV value must not be empty.")
    return items


def _comma_ints(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _config_from_args(args: argparse.Namespace) -> ClaimFactualityProbeWorkflowConfig:
    return ClaimFactualityProbeWorkflowConfig(
        output_dir=args.output_dir,
        records_path=args.records,
        report_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        register_name=args.register_name,
        register_version=args.register_version,
        model=args.model,
        dtype=args.dtype,
        layer=args.layer,
        limit=args.limit,
        manifold_questions=args.manifold_questions,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        hidden_state_capture=args.hidden_state_capture,
        claim_factuality_layers=_parse_int_list(args.claim_factuality_layers) or DEFAULT_CLAIM_FACTUALITY_LAYERS,
        offline=True if args.offline is None else bool(args.offline),
        auto_batch_size=bool(args.auto_batch_size),
        length_bucketed_batches=not bool(args.no_length_bucketed_batches),
        progress_every=args.progress_every,
        refresh_records=bool(args.refresh_records),
        sweep_layers=_parse_int_list(args.sweep_layers, allow_auto=True),
        best_by=args.best_by,
        train_fraction=args.train_fraction,
        seed=args.seed,
        pooling=args.pooling,
        steps=args.steps,
        lr=args.lr,
        l2=args.l2,
        conformal_alpha=args.conformal_alpha,
        soft_target_cutoff=args.soft_target_cutoff,
        calibration_model_id=args.calibration_model_id,
        run_text_baseline=not bool(args.skip_text_baseline),
        baseline_signals=_parse_csv(args.baseline_signals) or DEFAULT_TEXT_BASELINE_SIGNALS,
        python_executable=args.python_executable,
        clean=bool(args.clean),
        dry_run=bool(args.dry_run),
        compact_json=bool(args.compact_json),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the claim factuality probe evidence workflow")
    parser.add_argument("--output-dir", required=True, help="directory for workflow artifacts")
    parser.add_argument("--records", default=None, help="optional existing records JSON/JSONL to reuse")
    parser.add_argument("--json", default=None, help="optional workflow report path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact-manifest path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--register-name", default=None, help="optional registry report/manifest record name")
    parser.add_argument("--register-version", default="0.1")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--manifold-questions", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batch-tokens", type=int, default=0)
    parser.add_argument("--hidden-state-capture", default="hooks", choices=HIDDEN_STATE_CAPTURE_MODES)
    parser.add_argument("--claim-factuality-layers", default="-1,-2")
    offline_group = parser.add_mutually_exclusive_group()
    offline_group.add_argument(
        "--offline",
        dest="offline",
        action="store_true",
        default=None,
        help="use bundled TruthfulQA smoke fixture for record export",
    )
    offline_group.add_argument(
        "--real-truthfulqa",
        dest="offline",
        action="store_false",
        default=None,
        help="download/use real TruthfulQA instead of the offline smoke fixture",
    )
    parser.add_argument("--auto-batch-size", action="store_true")
    parser.add_argument("--no-length-bucketed-batches", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--refresh-records", action="store_true", help="rerun record export even if records exist")
    parser.add_argument("--sweep-layers", default="auto", help="comma-separated probe layers, or auto")
    parser.add_argument("--best-by", default="auto", choices=CLAIM_FACTUALITY_SWEEP_BEST_BY)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooling", default="mean", choices=POOLING_MODES)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--conformal-alpha", type=float, default=0.1)
    parser.add_argument("--soft-target-cutoff", type=float, default=None)
    parser.add_argument("--calibration-model-id", default="claim_factuality_probe")
    parser.add_argument("--skip-text-baseline", action="store_true")
    parser.add_argument("--baseline-signals", default=None, help="comma-separated text redline baseline signals")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compact-json", action="store_true")
    run_claim_factuality_probe_workflow(_config_from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
