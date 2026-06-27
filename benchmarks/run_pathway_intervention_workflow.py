"""Run baseline/ablation/source-patch pathway intervention evidence.

This workflow produces the mechanism-evidence chain needed before promoting a
pathway-causality claim:

1. run ``eval_truthfulqa.py`` once as baseline;
2. rerun with ``--activation-intervention-layer`` for ablation evidence;
3. rerun with ``--activation-patch-layer`` for source-token patch evidence;
4. compare each intervention score dump against baseline with
   ``eval_pathway_intervention.py`` and write a manifest-backed report.
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
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_TRUTHFULQA_SCRIPT = Path("benchmarks") / "eval_truthfulqa.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import planned_artifact_manifest_summary  # noqa: E402
from benchmarks.eval_pathway_intervention import run_pathway_intervention_eval  # noqa: E402
from eigentruth.intervention import (  # noqa: E402
    ACTIVATION_INTERVENTION_MODES,
    ACTIVATION_INTERVENTION_SPANS,
    ACTIVATION_PATCH_ALIGNMENTS,
)
from eigentruth.registry import ArtifactRegistry, ArtifactVerificationContext  # noqa: E402

PATCH_SOURCE_MODES = ("opposite_label", "same_label")
DEFAULT_SIGNALS = ("pathway_disagreement", "truth_proj", "nll_answer")
CommandRunner = Callable[[Sequence[str]], None]


@dataclass(frozen=True)
class PathwayInterventionWorkflowConfig:
    """Configuration for the pathway intervention evidence workflow."""

    output_dir: Path
    report_path: Path | None = None
    registry_path: Path | None = None
    name: str | None = None
    version: str = "0.1"
    model: str = "sshleifer/tiny-gpt2"
    dtype: str = "float32"
    layer: int = -1
    intervention_layer: int | None = None
    patch_layer: int | None = None
    limit: int | None = 12
    manifold_questions: int | None = 6
    max_length: int = 48
    batch_size: int = 4
    max_batch_tokens: int = 384
    hidden_state_capture: str = "hooks"
    progress_every: int = 0
    dump_scores_format: str = "jsonl"
    offline: bool = True
    auto_batch_size: bool = True
    length_bucketed_batches: bool = True
    activation_intervention_span: str = "answer"
    activation_intervention_mode: str = "zero"
    activation_intervention_scale: float = 0.0
    activation_patch_target_span: str = "answer"
    activation_patch_source_span: str = "answer"
    activation_patch_alignment: str = "left"
    activation_patch_source: str = "opposite_label"
    signals: Sequence[str] = DEFAULT_SIGNALS
    directions: Mapping[str, str] | None = None
    min_mean_risk_reduction: float = 0.0
    min_improved_fraction: float = 0.5
    python_executable: str = sys.executable
    clean: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.report_path is not None:
            object.__setattr__(self, "report_path", Path(self.report_path))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        object.__setattr__(self, "layer", int(self.layer))
        object.__setattr__(
            self,
            "intervention_layer",
            int(self.layer) if self.intervention_layer is None else int(self.intervention_layer),
        )
        object.__setattr__(
            self,
            "patch_layer",
            int(self.layer) if self.patch_layer is None else int(self.patch_layer),
        )
        object.__setattr__(self, "max_length", int(self.max_length))
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "max_batch_tokens", int(self.max_batch_tokens))
        object.__setattr__(self, "progress_every", int(self.progress_every))
        object.__setattr__(self, "signals", tuple(str(signal) for signal in self.signals if str(signal)))
        if self.directions is not None:
            object.__setattr__(
                self,
                "directions",
                {str(key): _coerce_direction(value) for key, value in self.directions.items()},
            )
        if self.registry_path is not None and not self.name:
            raise ValueError("registry_path requires name.")
        if self.dump_scores_format not in {"json", "jsonl"}:
            raise ValueError("dump_scores_format must be one of: json, jsonl.")
        if self.hidden_state_capture not in {"outputs", "hooks"}:
            raise ValueError("hidden_state_capture must be one of: outputs, hooks.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >=1.")
        if self.max_batch_tokens < 0:
            raise ValueError("max_batch_tokens must be >=0.")
        if self.progress_every < 0:
            raise ValueError("progress_every must be >=0.")
        if self.activation_intervention_span not in ACTIVATION_INTERVENTION_SPANS:
            raise ValueError("activation_intervention_span is invalid.")
        if self.activation_intervention_mode not in ACTIVATION_INTERVENTION_MODES:
            raise ValueError("activation_intervention_mode is invalid.")
        if self.activation_patch_target_span not in ACTIVATION_INTERVENTION_SPANS:
            raise ValueError("activation_patch_target_span is invalid.")
        if self.activation_patch_source_span not in ACTIVATION_INTERVENTION_SPANS:
            raise ValueError("activation_patch_source_span is invalid.")
        if self.activation_patch_alignment not in ACTIVATION_PATCH_ALIGNMENTS:
            raise ValueError("activation_patch_alignment is invalid.")
        if self.activation_patch_source not in PATCH_SOURCE_MODES:
            raise ValueError("activation_patch_source is invalid.")
        if not self.signals:
            raise ValueError("signals must contain at least one score name.")
        if not 0.0 <= float(self.min_improved_fraction) <= 1.0:
            raise ValueError("min_improved_fraction must be in [0, 1].")

    @property
    def workflow_report_path(self) -> Path:
        """Return the workflow report path."""
        return self.report_path or self.output_dir / "pathway-intervention-workflow.json"

    @property
    def artifact_manifest_path(self) -> Path:
        """Return the top-level artifact manifest path."""
        return self.output_dir / "artifact-manifest.json"

    @property
    def score_suffix(self) -> str:
        """Return score dump suffix for this workflow."""
        return ".manifest.json" if self.dump_scores_format == "jsonl" else ".json"


def run_pathway_intervention_workflow(
    config: PathwayInterventionWorkflowConfig,
    *,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Run or plan the pathway intervention evidence workflow."""
    started_at = time.perf_counter()
    command_runner = _run_command if command_runner is None else command_runner
    verification_context = ArtifactVerificationContext()
    if config.clean and config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for run_dir in _run_dirs(config).values():
        run_dir.mkdir(parents=True, exist_ok=True)

    commands = {
        "baseline": _truthfulqa_command(config, run_name="baseline"),
        "activation_ablation": _truthfulqa_command(config, run_name="activation_ablation"),
        "source_patch": _truthfulqa_command(config, run_name="source_patch"),
    }
    compare_reports: dict[str, dict[str, Any] | None] = {
        "activation_ablation": None,
        "source_patch": None,
    }
    if not config.dry_run:
        for command in commands.values():
            command_runner(command)
        compare_reports["activation_ablation"] = run_pathway_intervention_eval(
            baseline_scores_path=_score_path(config, "baseline"),
            intervened_scores_path=_score_path(config, "activation_ablation"),
            output_path=_compare_report_path(config, "activation_ablation"),
            artifact_manifest_path=_compare_manifest_path(config, "activation_ablation"),
            signals=config.signals,
            directions=config.directions,
            pathway=config.activation_intervention_span,
            intervention_name=f"{config.activation_intervention_span}_activation_{config.activation_intervention_mode}",
            min_mean_risk_reduction=config.min_mean_risk_reduction,
            min_improved_fraction=config.min_improved_fraction,
            include_record_effects=False,
            metadata={"workflow": "run_pathway_intervention_workflow", "run": "activation_ablation"},
        )
        compare_reports["source_patch"] = run_pathway_intervention_eval(
            baseline_scores_path=_score_path(config, "baseline"),
            intervened_scores_path=_score_path(config, "source_patch"),
            output_path=_compare_report_path(config, "source_patch"),
            artifact_manifest_path=_compare_manifest_path(config, "source_patch"),
            signals=config.signals,
            directions=config.directions,
            pathway=config.activation_patch_target_span,
            intervention_name=f"{config.activation_patch_target_span}_source_patch",
            min_mean_risk_reduction=config.min_mean_risk_reduction,
            min_improved_fraction=config.min_improved_fraction,
            include_record_effects=False,
            metadata={"workflow": "run_pathway_intervention_workflow", "run": "source_patch"},
        )

    artifacts = _artifact_paths(config)
    artifact_manifest_summary = planned_artifact_manifest_summary(
        artifacts,
        assume_file_paths=(config.workflow_report_path,),
    )
    status = _workflow_status(config=config, compare_reports=compare_reports)
    report: dict[str, Any] = {
        "schema_version": 1,
        "workflow": "pathway_intervention_workflow",
        "status": status,
        "paths": _paths_payload(config),
        "config": _config_payload(config),
        "execution": {
            "commands": commands,
            "dry_run": config.dry_run,
            "wall_clock_seconds": time.perf_counter() - started_at,
        },
        "comparisons": _comparison_summary(compare_reports),
        "artifact_manifest_summary": artifact_manifest_summary,
    }
    if config.registry_path is not None:
        report["registry_record"] = f"report:{config.name}:{config.version}"
    report["evidence_bundle"] = _evidence_bundle_summary(config, report=report)
    config.workflow_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.workflow_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = _write_artifact_manifest(config, report, artifacts, verification_context=verification_context)
    report["artifact_manifest_summary"] = manifest["summary"]
    report["evidence_bundle"] = _evidence_bundle_summary(config, report=report)
    report["artifact_cache"] = verification_context.cache_summary()
    config.workflow_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # The report includes manifest/cache summaries, so rebuild the manifest
    # after the final report write to keep report fingerprints current.
    _write_artifact_manifest(config, report, artifacts, verification_context=verification_context)
    _record_registry(config, report)
    return report


def _truthfulqa_command(config: PathwayInterventionWorkflowConfig, *, run_name: str) -> list[str]:
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
        str(_truthfulqa_report_path(config, run_name)),
        "--dump-scores",
        str(_score_path(config, run_name)),
        "--dump-scores-format",
        config.dump_scores_format,
    ]
    if config.limit is not None:
        command.extend(["--limit", str(config.limit)])
    if config.manifold_questions is not None:
        command.extend(["--manifold-questions", str(config.manifold_questions)])
    if config.offline:
        command.append("--offline")
    if config.auto_batch_size:
        command.append("--auto-batch-size")
    if config.length_bucketed_batches:
        command.append("--length-bucketed-batches")
    if run_name == "activation_ablation":
        command.extend([
            "--activation-intervention-layer",
            str(config.intervention_layer),
            "--activation-intervention-span",
            config.activation_intervention_span,
            "--activation-intervention-mode",
            config.activation_intervention_mode,
            "--activation-intervention-scale",
            str(config.activation_intervention_scale),
        ])
    elif run_name == "source_patch":
        command.extend([
            "--activation-patch-layer",
            str(config.patch_layer),
            "--activation-patch-target-span",
            config.activation_patch_target_span,
            "--activation-patch-source-span",
            config.activation_patch_source_span,
            "--activation-patch-alignment",
            config.activation_patch_alignment,
            "--activation-patch-source",
            config.activation_patch_source,
        ])
    return command


def _run_command(command: Sequence[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _run_dirs(config: PathwayInterventionWorkflowConfig) -> dict[str, Path]:
    return {
        "baseline": config.output_dir / "baseline",
        "activation_ablation": config.output_dir / "activation-ablation",
        "source_patch": config.output_dir / "source-patch",
    }


def _score_path(config: PathwayInterventionWorkflowConfig, run_name: str) -> Path:
    return _run_dirs(config)[run_name] / f"scores{config.score_suffix}"


def _truthfulqa_report_path(config: PathwayInterventionWorkflowConfig, run_name: str) -> Path:
    return _run_dirs(config)[run_name] / "truthfulqa-report.json"


def _compare_report_path(config: PathwayInterventionWorkflowConfig, run_name: str) -> Path:
    return _run_dirs(config)[run_name] / "intervention-effect-report.json"


def _compare_manifest_path(config: PathwayInterventionWorkflowConfig, run_name: str) -> Path:
    return _run_dirs(config)[run_name] / "artifact-manifest.json"


def _artifact_paths(config: PathwayInterventionWorkflowConfig) -> dict[str, Path]:
    return {
        "workflow_report": config.workflow_report_path,
        "baseline_scores": _score_path(config, "baseline"),
        "baseline_truthfulqa_report": _truthfulqa_report_path(config, "baseline"),
        "activation_ablation_scores": _score_path(config, "activation_ablation"),
        "activation_ablation_truthfulqa_report": _truthfulqa_report_path(config, "activation_ablation"),
        "activation_ablation_report": _compare_report_path(config, "activation_ablation"),
        "activation_ablation_manifest": _compare_manifest_path(config, "activation_ablation"),
        "source_patch_scores": _score_path(config, "source_patch"),
        "source_patch_truthfulqa_report": _truthfulqa_report_path(config, "source_patch"),
        "source_patch_report": _compare_report_path(config, "source_patch"),
        "source_patch_manifest": _compare_manifest_path(config, "source_patch"),
    }


def _paths_payload(config: PathwayInterventionWorkflowConfig) -> dict[str, Any]:
    return {
        "workflow_report": str(config.workflow_report_path),
        "artifact_manifest": str(config.artifact_manifest_path),
        "runs": {
            name: {
                "score_dump": str(_score_path(config, name)),
                "truthfulqa_report": str(_truthfulqa_report_path(config, name)),
                "comparison_report": (
                    None if name == "baseline" else str(_compare_report_path(config, name))
                ),
                "artifact_manifest": (
                    None if name == "baseline" else str(_compare_manifest_path(config, name))
                ),
            }
            for name in ("baseline", "activation_ablation", "source_patch")
        },
    }


def _config_payload(config: PathwayInterventionWorkflowConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "dtype": config.dtype,
        "layer": config.layer,
        "intervention_layer": config.intervention_layer,
        "patch_layer": config.patch_layer,
        "limit": config.limit,
        "manifold_questions": config.manifold_questions,
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "max_batch_tokens": config.max_batch_tokens,
        "hidden_state_capture": config.hidden_state_capture,
        "progress_every": config.progress_every,
        "dump_scores_format": config.dump_scores_format,
        "offline": config.offline,
        "auto_batch_size": config.auto_batch_size,
        "length_bucketed_batches": config.length_bucketed_batches,
        "activation_intervention": {
            "span": config.activation_intervention_span,
            "mode": config.activation_intervention_mode,
            "scale": config.activation_intervention_scale,
        },
        "activation_patch": {
            "target_span": config.activation_patch_target_span,
            "source_span": config.activation_patch_source_span,
            "alignment": config.activation_patch_alignment,
            "source": config.activation_patch_source,
        },
        "signals": tuple(config.signals),
        "directions": dict(config.directions or {}),
        "min_mean_risk_reduction": config.min_mean_risk_reduction,
        "min_improved_fraction": config.min_improved_fraction,
        "dry_run": config.dry_run,
    }


def _workflow_status(
    *,
    config: PathwayInterventionWorkflowConfig,
    compare_reports: Mapping[str, Mapping[str, Any] | None],
) -> str:
    if config.dry_run:
        return "needs_evidence"
    if any(report is None for report in compare_reports.values()):
        return "blocked"
    return "complete"


def _comparison_summary(
    compare_reports: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    summary = {}
    for name, report in compare_reports.items():
        if report is None:
            summary[name] = None
            continue
        summary[name] = {
            "status": report.get("status"),
            "gate": dict(report.get("summary", {}).get("gate") or {}),
            "best_signal": report.get("summary", {}).get("best_signal"),
            "n_total": report.get("summary", {}).get("n_total"),
            "paths": dict(report.get("paths") or {}),
        }
    return summary


def _evidence_bundle_summary(
    config: PathwayInterventionWorkflowConfig,
    *,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    comparisons = dict(report.get("comparisons") or {})
    gates = {
        name: dict(value.get("gate") or {}) if isinstance(value, Mapping) else None
        for name, value in comparisons.items()
    }
    release_ready = (
        report.get("status") == "complete"
        and bool(gates.get("activation_ablation"))
        and bool(gates.get("source_patch"))
        and gates["activation_ablation"].get("status") == "promote"
        and gates["source_patch"].get("status") == "promote"
        and int(dict(report.get("artifact_manifest_summary") or {}).get("missing_count", 0)) == 0
    )
    return {
        "schema_version": 1,
        "status": report.get("status"),
        "release_ready": release_ready,
        "model": config.model,
        "layer": config.layer,
        "intervention_layer": config.intervention_layer,
        "patch_layer": config.patch_layer,
        "signals": tuple(config.signals),
        "gates": gates,
        "best_signals": {
            name: value.get("best_signal") if isinstance(value, Mapping) else None
            for name, value in comparisons.items()
        },
        "artifact_manifest": str(config.artifact_manifest_path),
    }


def _write_artifact_manifest(
    config: PathwayInterventionWorkflowConfig,
    report: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    *,
    verification_context: ArtifactVerificationContext,
) -> dict[str, Any]:
    manifest = verification_context.build_artifact_manifest(
        artifacts,
        root=config.output_dir,
        metadata={
            "runner": "run_pathway_intervention_workflow",
            "status": report.get("status"),
            "model": config.model,
            "layer": config.layer,
            "intervention_layer": config.intervention_layer,
            "patch_layer": config.patch_layer,
            "offline": config.offline,
            "dry_run": config.dry_run,
            "activation_ablation_gate": _nested(report, "comparisons", "activation_ablation", "gate", "status"),
            "source_patch_gate": _nested(report, "comparisons", "source_patch", "gate", "status"),
        },
    )
    config.artifact_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _record_registry(config: PathwayInterventionWorkflowConfig, report: Mapping[str, Any]) -> None:
    if config.registry_path is None or config.dry_run:
        return
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_report(
        name=str(config.name),
        path=config.workflow_report_path,
        version=str(config.version),
        metadata={
            "workflow": "run_pathway_intervention_workflow",
            "status": report.get("status"),
            "artifact_manifest": str(config.artifact_manifest_path),
            "activation_ablation_gate": _nested(report, "comparisons", "activation_ablation", "gate", "status"),
            "source_patch_gate": _nested(report, "comparisons", "source_patch", "gate", "status"),
            "release_ready": _nested(report, "evidence_bundle", "release_ready"),
            "model": config.model,
            "layer": config.layer,
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


def _parse_int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if not text else int(text)


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_SIGNALS
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("signals must contain at least one value.")
    return values


def _parse_directions(values: Sequence[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError("direction overrides must use name=higher or name=lower.")
        name, direction = value.split("=", 1)
        parsed[name.strip()] = _coerce_direction(direction)
    return parsed


def _coerce_direction(value: Any) -> str:
    text = str(value).strip().lower()
    if text not in {"higher", "lower"}:
        raise ValueError("direction must be one of: higher, lower.")
    return text


def _config_from_args(args: argparse.Namespace) -> PathwayInterventionWorkflowConfig:
    return PathwayInterventionWorkflowConfig(
        output_dir=Path(args.output_dir),
        report_path=Path(args.json) if args.json else None,
        registry_path=Path(args.registry) if args.registry else None,
        name=args.name,
        version=args.version,
        model=args.model,
        dtype=args.dtype,
        layer=args.layer,
        intervention_layer=_parse_int_or_none(args.intervention_layer),
        patch_layer=_parse_int_or_none(args.patch_layer),
        limit=_parse_int_or_none(args.limit),
        manifold_questions=_parse_int_or_none(args.manifold_questions),
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        hidden_state_capture=args.hidden_state_capture,
        progress_every=args.progress_every,
        dump_scores_format=args.dump_scores_format,
        offline=args.offline,
        auto_batch_size=args.auto_batch_size,
        length_bucketed_batches=not args.no_length_bucketed_batches,
        activation_intervention_span=args.activation_intervention_span,
        activation_intervention_mode=args.activation_intervention_mode,
        activation_intervention_scale=args.activation_intervention_scale,
        activation_patch_target_span=args.activation_patch_target_span,
        activation_patch_source_span=args.activation_patch_source_span,
        activation_patch_alignment=args.activation_patch_alignment,
        activation_patch_source=args.activation_patch_source,
        signals=_parse_csv(args.signals),
        directions=_parse_directions(args.direction),
        min_mean_risk_reduction=args.min_mean_risk_reduction,
        min_improved_fraction=args.min_improved_fraction,
        python_executable=args.python_executable,
        clean=args.clean,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run pathway intervention evidence workflow")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None, help="optional workflow report path")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry report name")
    parser.add_argument("--version", default="0.1")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--intervention-layer", default=None, help="defaults to --layer")
    parser.add_argument("--patch-layer", default=None, help="defaults to --layer")
    parser.add_argument("--limit", default="12", help="TruthfulQA eval question limit; empty disables")
    parser.add_argument("--manifold-questions", default="6", help="warmup question count; empty disables")
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=384)
    parser.add_argument("--hidden-state-capture", default="hooks", choices=("outputs", "hooks"))
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--dump-scores-format", default="jsonl", choices=("json", "jsonl"))
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--online", dest="offline", action="store_false")
    parser.add_argument("--auto-batch-size", action="store_true", default=True)
    parser.add_argument("--no-auto-batch-size", dest="auto_batch_size", action="store_false")
    parser.add_argument("--no-length-bucketed-batches", action="store_true")
    parser.add_argument("--activation-intervention-span", default="answer", choices=ACTIVATION_INTERVENTION_SPANS)
    parser.add_argument("--activation-intervention-mode", default="zero", choices=ACTIVATION_INTERVENTION_MODES)
    parser.add_argument("--activation-intervention-scale", type=float, default=0.0)
    parser.add_argument("--activation-patch-target-span", default="answer", choices=ACTIVATION_INTERVENTION_SPANS)
    parser.add_argument("--activation-patch-source-span", default="answer", choices=ACTIVATION_INTERVENTION_SPANS)
    parser.add_argument("--activation-patch-alignment", default="left", choices=ACTIVATION_PATCH_ALIGNMENTS)
    parser.add_argument("--activation-patch-source", default="opposite_label", choices=PATCH_SOURCE_MODES)
    parser.add_argument("--signals", default=",".join(DEFAULT_SIGNALS))
    parser.add_argument("--direction", action="append", default=())
    parser.add_argument("--min-mean-risk-reduction", type=float, default=0.0)
    parser.add_argument("--min-improved-fraction", type=float, default=0.5)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    payload = run_pathway_intervention_workflow(_config_from_args(args))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
