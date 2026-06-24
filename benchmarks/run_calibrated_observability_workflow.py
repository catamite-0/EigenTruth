"""Run the calibrated-observability score-dump -> calibration closure.

This workflow orchestrates the low-cost post-hoc path:

1. optionally run ``eval_truthfulqa.py`` to produce a score dump, preferably as
   a JSONL manifest plus records sidecar;
2. run ``eval_conformal.py`` to produce a structured report, sweep report, best
   calibration artifact, and conformal artifact manifest;
3. write a top-level artifact manifest and optional local registry record.

The workflow can also reuse an existing score dump, which is the common path for
iterating on calibration without rerunning a model.
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
EVAL_CONFORMAL_SCRIPT = Path("benchmarks") / "eval_conformal.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import planned_artifact_manifest_summary  # noqa: E402
from eigentruth.eval.score_dump import JSONL_FORMAT, ScoreDumpJsonlManifest  # noqa: E402
from eigentruth.registry import (  # noqa: E402
    ArtifactRegistry,
    ArtifactVerificationContext,
)

DEFAULT_SWEEP_SIGNALS = (
    "maha_last",
    "truth_proj",
    "subspace_resid",
    "eigenscore",
    "inside_eigenscore",
    "inside_semantic_entropy",
    "inside_embedding_entropy",
)

RUNTIME_PRESETS = ("custom", "quick", "calibrate", "full")
RUNTIME_PRESET_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "custom": {},
    "quick": {
        "limit": 12,
        "manifold_questions": 6,
        "max_length": 48,
        "max_batch_tokens": 384,
        "sweep_layers": (-1, -2, -4),
        "signals": ("maha_last", "truth_proj", "subspace_resid"),
        "repeats": 3,
        "artifact_alpha": 0.20,
        "offline": True,
    },
    "calibrate": {
        "dump_scores_format": "jsonl",
        "max_batch_tokens": 768,
        "signals": DEFAULT_SWEEP_SIGNALS,
        "repeats": 20,
        "offline": True,
    },
    "full": {
        "max_length": 128,
        "max_batch_tokens": 1024,
        "auto_batch_size": True,
        "signals": DEFAULT_SWEEP_SIGNALS,
        "repeats": 30,
        "offline": False,
    },
}


@dataclass(frozen=True)
class CalibratedObservabilityWorkflowConfig:
    """Configuration for the calibrated-observability closure workflow."""

    output_dir: Path
    scores_path: Path | None = None
    report_path: Path | None = None
    registry_path: Path | None = None
    name: str | None = None
    version: str | None = None
    model: str = "sshleifer/tiny-gpt2"
    dtype: str = "float32"
    layer: int = -1
    sweep: bool = True
    sweep_layers: Sequence[int] = ()
    limit: int | None = None
    manifold_questions: int | None = None
    max_length: int = 64
    batch_size: int = 1
    max_batch_tokens: int = 0
    hidden_state_capture: str = "outputs"
    progress_every: int = 0
    length_bucketed_batches: bool = True
    offline: bool = True
    auto_batch_size: bool = False
    dump_scores_format: str = "jsonl"
    refresh_scores: bool = False
    signals: Sequence[str] = DEFAULT_SWEEP_SIGNALS
    signal: str = "maha_last"
    direction: str | None = None
    repeats: int = 20
    seed: int = 0
    artifact_alpha: float = 0.10
    best_by: str = "auroc"
    python_executable: str = sys.executable
    clean: bool = False
    dry_run: bool = False
    runtime_preset: str = "custom"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for field_name in ("scores_path", "report_path", "registry_path"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, Path(value))
        object.__setattr__(self, "layer", int(self.layer))
        object.__setattr__(self, "sweep_layers", tuple(int(layer) for layer in self.sweep_layers))
        object.__setattr__(self, "signals", tuple(str(signal) for signal in self.signals if str(signal)))
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "max_batch_tokens", int(self.max_batch_tokens))
        object.__setattr__(self, "max_length", int(self.max_length))
        object.__setattr__(self, "progress_every", int(self.progress_every))
        object.__setattr__(self, "repeats", int(self.repeats))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "artifact_alpha", float(self.artifact_alpha))
        if self.runtime_preset not in RUNTIME_PRESETS:
            raise ValueError(f"runtime_preset must be one of: {', '.join(RUNTIME_PRESETS)}.")
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        if self.dump_scores_format not in {"json", "jsonl"}:
            raise ValueError("dump_scores_format must be one of: json, jsonl.")
        if self.best_by not in {"auroc", "detection"}:
            raise ValueError("best_by must be one of: auroc, detection.")
        if self.direction not in {None, "higher", "lower"}:
            raise ValueError("direction must be one of: higher, lower.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >=1.")
        if self.max_batch_tokens < 0:
            raise ValueError("max_batch_tokens must be >=0.")
        if self.max_length < 1:
            raise ValueError("max_length must be >=1.")
        if self.progress_every < 0:
            raise ValueError("progress_every must be >=0.")
        if self.repeats < 1:
            raise ValueError("repeats must be >=1.")

    @property
    def resolved_scores_path(self) -> Path:
        """Return the score-dump path used by the workflow."""
        if self.scores_path is not None:
            return self.scores_path
        suffix = ".manifest.json" if self.dump_scores_format == "jsonl" else ".json"
        return self.output_dir / f"scores{suffix}"

    @property
    def truthfulqa_report_path(self) -> Path:
        """Return the structured TruthfulQA report path."""
        return self.output_dir / "truthfulqa-report.json"

    @property
    def truthfulqa_profile_path(self) -> Path:
        """Return the TruthfulQA profile report path."""
        return self.output_dir / "truthfulqa-profile.json"

    @property
    def conformal_report_path(self) -> Path:
        """Return the conformal report path."""
        return self.output_dir / "conformal-report.json"

    @property
    def sweep_report_path(self) -> Path:
        """Return the layer/score sweep report path."""
        return self.output_dir / "sweep-report.json"

    @property
    def best_calibration_path(self) -> Path:
        """Return the best calibration artifact path."""
        return self.output_dir / "best-calibration.json"

    @property
    def conformal_artifact_manifest_path(self) -> Path:
        """Return the conformal artifact manifest path."""
        return self.output_dir / "conformal-artifact-manifest.json"

    @property
    def resolved_report_path(self) -> Path:
        """Return the workflow report path."""
        return self.report_path or self.output_dir / "calibrated-observability-workflow.json"

    @property
    def artifact_manifest_path(self) -> Path:
        """Return the top-level artifact manifest path."""
        return self.output_dir / "artifact-manifest.json"


def run_calibrated_observability_workflow(
    config: CalibratedObservabilityWorkflowConfig,
) -> dict[str, Any]:
    """Run or plan the calibrated-observability workflow."""
    started_at = time.perf_counter()
    verification_context = ArtifactVerificationContext()
    if config.clean and config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    score_dump_reused = config.resolved_scores_path.exists() and not config.refresh_scores
    truthfulqa_command = _truthfulqa_command(config)
    conformal_command = _conformal_command(config)
    execution: dict[str, Any] = {
        "score_dump_reused": score_dump_reused,
        "truthfulqa_command": truthfulqa_command,
        "conformal_command": conformal_command,
    }

    conformal_payload: dict[str, Any] | None = None
    conformal_manifest_verification: dict[str, Any] | None = None
    if not config.dry_run:
        if not score_dump_reused:
            _run_command(truthfulqa_command)
        elif not config.resolved_scores_path.exists():
            raise FileNotFoundError(f"score dump does not exist: {config.resolved_scores_path}")
        _run_command(conformal_command)
        conformal_payload = _load_json(config.conformal_report_path)
        conformal_manifest_verification = verification_context.load_and_verify_artifact_manifest(
            config.conformal_artifact_manifest_path,
            recursive=True,
        ).to_dict()

    status = _workflow_status(
        dry_run=config.dry_run,
        conformal_manifest_verification=conformal_manifest_verification,
    )
    artifacts = _artifact_paths(config, score_dump_reused=score_dump_reused)
    artifact_manifest_summary = planned_artifact_manifest_summary(
        artifacts,
        assume_file_paths=(config.resolved_report_path,),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "workflow": "calibrated_observability_workflow",
        "status": status,
        "paths": _paths_payload(config),
        "config": _config_payload(config),
        "execution": {
            **execution,
            "wall_clock_seconds": time.perf_counter() - started_at,
        },
        "conformal": _conformal_summary(conformal_payload),
        "conformal_manifest_verification": conformal_manifest_verification,
        "artifact_manifest_summary": artifact_manifest_summary,
    }
    if config.registry_path is not None:
        report["registry_record"] = f"report:{config.name}:{config.version}"
    report["evidence_bundle"] = _evidence_bundle_summary(
        config,
        report=report,
        conformal_payload=conformal_payload,
        artifact_manifest_summary=artifact_manifest_summary,
    )
    config.resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.resolved_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = _write_artifact_manifest(config, report, artifacts, verification_context=verification_context)
    report["artifact_manifest_summary"] = manifest["summary"]
    report["evidence_bundle"] = _evidence_bundle_summary(
        config,
        report=report,
        conformal_payload=conformal_payload,
        artifact_manifest_summary=manifest["summary"],
    )
    report["artifact_cache"] = verification_context.cache_summary()
    _record_registry(config, report)
    return report


def _truthfulqa_command(config: CalibratedObservabilityWorkflowConfig) -> list[str]:
    command = [
        str(config.python_executable),
        str(EVAL_TRUTHFULQA_SCRIPT),
        "--model",
        config.model,
        "--dtype",
        config.dtype,
        "--layer",
        str(config.layer),
        "--batch-size",
        str(config.batch_size),
        "--max-batch-tokens",
        str(config.max_batch_tokens),
        "--max-length",
        str(config.max_length),
        "--hidden-state-capture",
        config.hidden_state_capture,
        "--progress-every",
        str(config.progress_every),
        "--json",
        str(config.truthfulqa_report_path),
        "--profile-json",
        str(config.truthfulqa_profile_path),
        "--dump-scores",
        str(config.resolved_scores_path),
        "--dump-scores-format",
        config.dump_scores_format,
    ]
    if config.sweep_layers:
        command.extend(["--sweep-layers", ",".join(str(layer) for layer in config.sweep_layers)])
    elif config.sweep:
        command.append("--sweep")
    if config.offline:
        command.append("--offline")
    if config.length_bucketed_batches:
        command.append("--length-bucketed-batches")
    if config.auto_batch_size:
        command.append("--auto-batch-size")
    if config.limit is not None:
        command.extend(["--limit", str(config.limit)])
    if config.manifold_questions is not None:
        command.extend(["--manifold-questions", str(config.manifold_questions)])
    return command


def _conformal_command(config: CalibratedObservabilityWorkflowConfig) -> list[str]:
    command = [
        str(config.python_executable),
        str(EVAL_CONFORMAL_SCRIPT),
        "--scores",
        str(config.resolved_scores_path),
        "--signal",
        config.signal,
        "--repeats",
        str(config.repeats),
        "--seed",
        str(config.seed),
        "--json",
        str(config.conformal_report_path),
        "--artifact-alpha",
        str(config.artifact_alpha),
        "--save-sweep-report",
        str(config.sweep_report_path),
        "--save-best-calibration",
        str(config.best_calibration_path),
        "--artifact-manifest",
        str(config.conformal_artifact_manifest_path),
        "--best-by",
        config.best_by,
    ]
    if config.signals:
        command.extend(["--signals", ",".join(config.signals)])
    if config.direction is not None:
        command.extend(["--direction", config.direction])
    return command


def _run_command(command: Sequence[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _workflow_status(
    *,
    dry_run: bool,
    conformal_manifest_verification: Mapping[str, Any] | None,
) -> str:
    if dry_run:
        return "needs_evidence"
    if not conformal_manifest_verification or not conformal_manifest_verification.get("passed"):
        return "blocked"
    return "complete"


def _conformal_summary(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    sweep_report = payload.get("sweep_report")
    best = sweep_report.get("best") if isinstance(sweep_report, Mapping) else None
    return {
        "verdict": payload.get("verdict"),
        "signal": dict(payload.get("config") or {}).get("signal"),
        "direction": dict(payload.get("config") or {}).get("direction"),
        "best": dict(best) if isinstance(best, Mapping) else None,
    }


def _evidence_bundle_summary(
    config: CalibratedObservabilityWorkflowConfig,
    *,
    report: Mapping[str, Any],
    conformal_payload: Mapping[str, Any] | None,
    artifact_manifest_summary: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(report.get("status"))
    conformal = dict(report.get("conformal") or {})
    verdict = conformal.get("verdict")
    manifest_verification = report.get("conformal_manifest_verification")
    manifest_passed = (
        bool(manifest_verification.get("passed"))
        if isinstance(manifest_verification, Mapping)
        else None
    )
    missing_count = int(artifact_manifest_summary.get("missing_count", 0))
    release_ready = (
        status == "complete"
        and verdict == "ACCEPT"
        and manifest_passed is True
        and missing_count == 0
    )
    score_dump_metadata = _conformal_score_dump_metadata(conformal_payload)
    score_dump_summary = dict(score_dump_metadata.get("summary") or {})
    best = dict(conformal.get("best") or {})
    registry_record = report.get("registry_record")
    return {
        "schema_version": 1,
        "status": status,
        "release_ready": release_ready,
        "runtime": {
            "runtime_preset": config.runtime_preset,
            "model": config.model,
            "dtype": config.dtype,
            "layer": config.layer,
            "offline": config.offline,
            "max_batch_tokens": config.max_batch_tokens,
            "auto_batch_size": config.auto_batch_size,
        },
        "score_dump": {
            "path": str(config.resolved_scores_path),
            "reused": dict(report.get("execution") or {}).get("score_dump_reused"),
            "source_format": score_dump_metadata.get("source_format"),
            "sha256": score_dump_metadata.get("sha256"),
            "records_sha256": _nested(score_dump_metadata, "records", "sha256"),
            "n_total": score_dump_summary.get("n_total"),
            "n_true": score_dump_summary.get("n_true"),
            "n_false": score_dump_summary.get("n_false"),
            "score_names": tuple(score_dump_summary.get("score_names", ())),
            "sweep_layers": tuple(score_dump_summary.get("sweep_layers", ())),
            "all_signal_names": tuple(score_dump_summary.get("all_signal_names", ())),
        },
        "calibration": {
            "conformal_verdict": verdict,
            "primary_signal": conformal.get("signal"),
            "direction": conformal.get("direction"),
            "best_by": config.best_by,
            "artifact_alpha": config.artifact_alpha,
            "best_layer": best.get("layer"),
            "best_score_name": best.get("score_name"),
            "best_direction": best.get("direction"),
            "best_threshold": best.get("threshold"),
            "best_auroc": best.get("auroc"),
            "best_false_alarm": best.get("false_alarm"),
            "best_detection": best.get("detection"),
        },
        "artifacts": {
            "summary": dict(artifact_manifest_summary),
            "artifact_manifest": str(config.artifact_manifest_path),
            "conformal_artifact_manifest": str(config.conformal_artifact_manifest_path),
            "best_calibration": str(config.best_calibration_path),
            "conformal_manifest_passed": manifest_passed,
            "conformal_manifest_checked": (
                manifest_verification.get("checked")
                if isinstance(manifest_verification, Mapping)
                else None
            ),
        },
        "registry": {
            "record": registry_record,
            "path": None if config.registry_path is None else str(config.registry_path),
        },
    }


def _conformal_score_dump_metadata(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    config = payload.get("config")
    if not isinstance(config, Mapping):
        return {}
    metadata = config.get("score_dump")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _paths_payload(config: CalibratedObservabilityWorkflowConfig) -> dict[str, str]:
    return {
        "score_dump": str(config.resolved_scores_path),
        "truthfulqa_report": str(config.truthfulqa_report_path),
        "truthfulqa_profile": str(config.truthfulqa_profile_path),
        "conformal_report": str(config.conformal_report_path),
        "sweep_report": str(config.sweep_report_path),
        "best_calibration": str(config.best_calibration_path),
        "conformal_artifact_manifest": str(config.conformal_artifact_manifest_path),
        "workflow_report": str(config.resolved_report_path),
        "artifact_manifest": str(config.artifact_manifest_path),
    }


def _config_payload(config: CalibratedObservabilityWorkflowConfig) -> dict[str, Any]:
    return {
        "runtime_preset": config.runtime_preset,
        "model": config.model,
        "dtype": config.dtype,
        "layer": config.layer,
        "sweep": config.sweep,
        "sweep_layers": tuple(config.sweep_layers),
        "limit": config.limit,
        "manifold_questions": config.manifold_questions,
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "max_batch_tokens": config.max_batch_tokens,
        "hidden_state_capture": config.hidden_state_capture,
        "progress_every": config.progress_every,
        "length_bucketed_batches": config.length_bucketed_batches,
        "offline": config.offline,
        "auto_batch_size": config.auto_batch_size,
        "dump_scores_format": config.dump_scores_format,
        "refresh_scores": config.refresh_scores,
        "signals": tuple(config.signals),
        "signal": config.signal,
        "direction": config.direction,
        "repeats": config.repeats,
        "seed": config.seed,
        "artifact_alpha": config.artifact_alpha,
        "best_by": config.best_by,
        "dry_run": config.dry_run,
    }


def _artifact_paths(
    config: CalibratedObservabilityWorkflowConfig,
    *,
    score_dump_reused: bool,
) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {
        "workflow_report": config.resolved_report_path,
        "score_dump": config.resolved_scores_path,
        "score_dump_records": _score_dump_records_path(config.resolved_scores_path),
        "conformal_report": config.conformal_report_path,
        "sweep_report": config.sweep_report_path,
        "best_calibration": config.best_calibration_path,
        "conformal_artifact_manifest": config.conformal_artifact_manifest_path,
    }
    if not score_dump_reused:
        artifacts["truthfulqa_report"] = config.truthfulqa_report_path
        artifacts["truthfulqa_profile"] = config.truthfulqa_profile_path
    return artifacts


def _score_dump_records_path(path: Path) -> Path | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, Mapping) or payload.get("format") != JSONL_FORMAT:
        return None
    return ScoreDumpJsonlManifest.from_mapping(payload).records_file(path)


def _write_artifact_manifest(
    config: CalibratedObservabilityWorkflowConfig,
    report: Mapping[str, Any],
    artifacts: Mapping[str, str | Path | None],
    *,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any]:
    manifest = verification_context.build_artifact_manifest(
        artifacts,
        root=config.output_dir,
        metadata={
            "runner": "run_calibrated_observability_workflow",
            "status": report.get("status"),
            "conformal_verdict": dict(report.get("conformal") or {}).get("verdict"),
            "model": config.model,
            "dtype": config.dtype,
            "layer": config.layer,
            "runtime_preset": config.runtime_preset,
            "offline": config.offline,
            "dry_run": config.dry_run,
            "score_dump_reused": dict(report.get("execution") or {}).get("score_dump_reused"),
            "dump_scores_format": config.dump_scores_format,
            "artifact_alpha": config.artifact_alpha,
            "best_by": config.best_by,
        },
    )
    config.artifact_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _record_registry(
    config: CalibratedObservabilityWorkflowConfig,
    report: Mapping[str, Any],
) -> None:
    if config.registry_path is None:
        return
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_report(
        name=str(config.name),
        path=config.resolved_report_path,
        version=str(config.version),
        metadata={
            "workflow": "run_calibrated_observability_workflow",
            "status": report.get("status"),
            "artifact_manifest": str(config.artifact_manifest_path),
            "conformal_artifact_manifest": str(config.conformal_artifact_manifest_path),
            "conformal_verdict": dict(report.get("conformal") or {}).get("verdict"),
            "score_dump": str(config.resolved_scores_path),
            "score_dump_reused": dict(report.get("execution") or {}).get("score_dump_reused"),
            "best_score_name": _nested(report, "conformal", "best", "score_name"),
            "best_layer": _nested(report, "conformal", "best", "layer"),
            "evidence_bundle_status": _nested(report, "evidence_bundle", "status"),
            "evidence_bundle_release_ready": _nested(report, "evidence_bundle", "release_ready"),
            "artifact_json_cache": _nested(report, "artifact_cache", "artifact_json_cache"),
            "artifact_fingerprint_cache_entries": _nested(
                report,
                "artifact_cache",
                "artifact_fingerprint_cache",
                "entries",
            ),
        },
    )
    registry.save_json()


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _parse_int_tuple(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_str_tuple(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _runtime_preset_defaults(runtime_preset: str) -> dict[str, Any]:
    if runtime_preset not in RUNTIME_PRESETS:
        raise ValueError(f"runtime_preset must be one of: {', '.join(RUNTIME_PRESETS)}.")
    return dict(RUNTIME_PRESET_DEFAULTS[runtime_preset])


def _arg_or_preset(args: argparse.Namespace, defaults: Mapping[str, Any], name: str, fallback: Any) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return defaults.get(name, fallback)


def _config_from_args(args: argparse.Namespace) -> CalibratedObservabilityWorkflowConfig:
    preset_defaults = _runtime_preset_defaults(args.runtime_preset)
    sweep = bool(_arg_or_preset(args, preset_defaults, "sweep", True))
    sweep_layers = (
        _parse_int_tuple(args.sweep_layers)
        if args.sweep_layers is not None
        else tuple(preset_defaults.get("sweep_layers", ()))
    )
    if not sweep:
        sweep_layers = ()
    signals = (
        _parse_str_tuple(args.signals)
        if args.signals is not None
        else tuple(preset_defaults.get("signals", DEFAULT_SWEEP_SIGNALS))
    )
    return CalibratedObservabilityWorkflowConfig(
        output_dir=Path(args.output_dir),
        runtime_preset=args.runtime_preset,
        scores_path=Path(args.scores) if args.scores else None,
        report_path=Path(args.json) if args.json else None,
        registry_path=Path(args.registry) if args.registry else None,
        name=args.name,
        version=args.version,
        model=_arg_or_preset(args, preset_defaults, "model", "sshleifer/tiny-gpt2"),
        dtype=_arg_or_preset(args, preset_defaults, "dtype", "float32"),
        layer=_arg_or_preset(args, preset_defaults, "layer", -1),
        sweep=sweep,
        sweep_layers=sweep_layers,
        limit=_arg_or_preset(args, preset_defaults, "limit", None),
        manifold_questions=_arg_or_preset(args, preset_defaults, "manifold_questions", None),
        max_length=_arg_or_preset(args, preset_defaults, "max_length", 64),
        batch_size=_arg_or_preset(args, preset_defaults, "batch_size", 1),
        max_batch_tokens=_arg_or_preset(args, preset_defaults, "max_batch_tokens", 0),
        hidden_state_capture=_arg_or_preset(args, preset_defaults, "hidden_state_capture", "outputs"),
        progress_every=_arg_or_preset(args, preset_defaults, "progress_every", 0),
        length_bucketed_batches=_arg_or_preset(args, preset_defaults, "length_bucketed_batches", True),
        offline=_arg_or_preset(args, preset_defaults, "offline", True),
        auto_batch_size=_arg_or_preset(args, preset_defaults, "auto_batch_size", False),
        dump_scores_format=_arg_or_preset(args, preset_defaults, "dump_scores_format", "jsonl"),
        refresh_scores=args.refresh_scores,
        signals=signals,
        signal=_arg_or_preset(args, preset_defaults, "signal", "maha_last"),
        direction=_arg_or_preset(args, preset_defaults, "direction", None),
        repeats=_arg_or_preset(args, preset_defaults, "repeats", 20),
        seed=_arg_or_preset(args, preset_defaults, "seed", 0),
        artifact_alpha=_arg_or_preset(args, preset_defaults, "artifact_alpha", 0.10),
        best_by=_arg_or_preset(args, preset_defaults, "best_by", "auroc"),
        python_executable=args.python,
        clean=args.clean,
        dry_run=args.dry_run,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = run_calibrated_observability_workflow(_config_from_args(args))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_blocked and report["status"] == "blocked":
        raise SystemExit(1)
    if args.fail_on_reject and dict(report.get("conformal") or {}).get("verdict") != "ACCEPT":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run calibrated-observability score dump and conformal closure")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--runtime-preset",
        default="custom",
        choices=RUNTIME_PRESETS,
        help=(
            "cost/control preset: custom preserves explicit defaults, quick bounds local smoke cost, "
            "calibrate favors score-dump reuse, full enables real TruthfulQA-oriented defaults"
        ),
    )
    parser.add_argument("--scores", default=None, help="reuse or write this score dump path")
    parser.add_argument("--json", default=None, help="optional workflow report path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry report name")
    parser.add_argument("--version", default=None, help="registry report version")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--layer", type=int, default=None)
    sweep_group = parser.add_mutually_exclusive_group()
    sweep_group.add_argument("--sweep", dest="sweep", action="store_true", default=None)
    sweep_group.add_argument("--no-sweep", dest="sweep", action="store_false", help="do not pass --sweep")
    parser.add_argument("--sweep-layers", default=None, help="comma-list passed to eval_truthfulqa.py --sweep-layers")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--manifold-questions", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batch-tokens", type=int, default=None)
    parser.add_argument("--hidden-state-capture", default=None, choices=("outputs", "hooks"))
    parser.add_argument("--progress-every", type=int, default=None)
    length_group = parser.add_mutually_exclusive_group()
    length_group.add_argument(
        "--length-bucketed-batches",
        dest="length_bucketed_batches",
        action="store_true",
        default=None,
    )
    length_group.add_argument(
        "--no-length-bucketed-batches",
        dest="length_bucketed_batches",
        action="store_false",
    )
    offline_group = parser.add_mutually_exclusive_group()
    offline_group.add_argument("--offline", dest="offline", action="store_true", default=None)
    offline_group.add_argument(
        "--real-truthfulqa",
        dest="offline",
        action="store_false",
        default=None,
        help="download/use real TruthfulQA instead of offline",
    )
    auto_batch_group = parser.add_mutually_exclusive_group()
    auto_batch_group.add_argument("--auto-batch-size", dest="auto_batch_size", action="store_true", default=None)
    auto_batch_group.add_argument("--no-auto-batch-size", dest="auto_batch_size", action="store_false")
    parser.add_argument("--dump-scores-format", default=None, choices=("json", "jsonl"))
    parser.add_argument("--refresh-scores", action="store_true", help="rerun eval_truthfulqa even if --scores exists")
    parser.add_argument("--signal", default=None, help="primary conformal signal")
    parser.add_argument("--signals", default=None, help="comma-list for sweep calibration")
    parser.add_argument("--direction", default=None, choices=("higher", "lower"))
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--artifact-alpha", type=float, default=None)
    parser.add_argument("--best-by", choices=("auroc", "detection"), default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    parser.add_argument("--fail-on-reject", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
