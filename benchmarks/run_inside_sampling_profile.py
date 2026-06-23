"""Profile fixed, adaptive, and self-check-bounded INSIDE sampling.

This workflow runs ``eval_truthfulqa.py`` with comparable INSIDE sampling
settings, then writes a compact report showing sample-budget and runtime
differences. It is intended to turn sampling-cost claims into reproducible
artifacts rather than one-off terminal notes.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = Path("benchmarks") / "eval_truthfulqa.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.registry import build_artifact_manifest  # noqa: E402

INSIDE_PROFILE_RUN_NAMES = ("fixed", "adaptive", "adaptive_selfcheck")


@dataclass(frozen=True)
class InsideSamplingProfileConfig:
    """Configuration for one INSIDE sampling profile comparison."""

    output_dir: Path
    model: str = "sshleifer/tiny-gpt2"
    dtype: str = "float32"
    layer: int = -1
    limit: int | None = None
    manifold_questions: int | None = None
    batch_size: int = 4
    max_batch_tokens: int = 0
    max_length: int = 64
    hidden_state_capture: str = "outputs"
    progress_every: int = 0
    offline: bool = True
    length_bucketed_batches: bool = True
    python_executable: str = sys.executable
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
    dump_scores: bool = False
    dump_inside_samples: bool = False
    adaptive_max_sample_ratio: float = 1.0
    adaptive_selfcheck_max_sample_ratio: float = 1.0
    max_inside_generation_seconds_ratio: float | None = None
    run_names: Sequence[str] = INSIDE_PROFILE_RUN_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.limit is not None and int(self.limit) < 0:
            raise ValueError("limit must be >=0.")
        if self.manifold_questions is not None and int(self.manifold_questions) < 1:
            raise ValueError("manifold_questions must be >=1.")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be >=1.")
        if int(self.max_batch_tokens) < 0:
            raise ValueError("max_batch_tokens must be >=0.")
        if int(self.max_length) < 1:
            raise ValueError("max_length must be >=1.")
        if int(self.inside_samples) < 2:
            raise ValueError("inside_samples must be >=2.")
        if int(self.inside_batch_size) < 1:
            raise ValueError("inside_batch_size must be >=1.")
        if int(self.inside_max_new_tokens) < 1:
            raise ValueError("inside_max_new_tokens must be >=1.")
        if float(self.inside_temperature) < 0.0:
            raise ValueError("inside_temperature must be non-negative.")
        _validate_unit_interval(self.inside_top_p, "inside_top_p")
        if int(self.inside_min_samples) < 2:
            raise ValueError("inside_min_samples must be >=2.")
        if int(self.inside_min_samples) > int(self.inside_samples):
            raise ValueError("inside_min_samples cannot exceed inside_samples.")
        if int(self.inside_sample_step) < 1:
            raise ValueError("inside_sample_step must be >=1.")
        if float(self.inside_stability_delta) < 0.0:
            raise ValueError("inside_stability_delta must be >=0.")
        if self.inside_pooling not in {"last", "mean"}:
            raise ValueError("inside_pooling must be 'last' or 'mean'.")
        _validate_unit_interval(self.inside_embedding_threshold, "inside_embedding_threshold", lower=-1.0)
        _validate_unit_interval(self.inside_selfcheck_min_overlap, "inside_selfcheck_min_overlap")
        _validate_unit_interval(self.inside_selfcheck_support_threshold, "inside_selfcheck_support_threshold")
        _validate_unit_interval(self.inside_selfcheck_refute_threshold, "inside_selfcheck_refute_threshold")
        if float(self.adaptive_max_sample_ratio) < 0.0:
            raise ValueError("adaptive_max_sample_ratio must be non-negative.")
        if float(self.adaptive_selfcheck_max_sample_ratio) < 0.0:
            raise ValueError("adaptive_selfcheck_max_sample_ratio must be non-negative.")
        if self.max_inside_generation_seconds_ratio is not None and self.max_inside_generation_seconds_ratio < 0.0:
            raise ValueError("max_inside_generation_seconds_ratio must be non-negative when set.")
        run_names = _normalize_run_names(self.run_names)
        object.__setattr__(self, "run_names", run_names)
        object.__setattr__(self, "dtype", str(self.dtype))
        object.__setattr__(self, "hidden_state_capture", str(self.hidden_state_capture))
        object.__setattr__(self, "inside_pooling", str(self.inside_pooling))

    @property
    def comparison_report(self) -> Path:
        return self.output_dir / "inside-sampling-profile-comparison.json"

    @property
    def artifact_manifest(self) -> Path:
        return self.output_dir / "artifact-manifest.json"

    def result_path(self, name: str) -> Path:
        return self.output_dir / f"result-{name}.json"

    def profile_path(self, name: str) -> Path:
        return self.output_dir / f"profile-{name}.json"

    def score_dump_path(self, name: str) -> Path:
        return self.output_dir / f"scores-{name}.json"


def build_eval_command(config: InsideSamplingProfileConfig, name: str) -> list[str]:
    """Build one ``eval_truthfulqa.py`` command for an INSIDE sampling run."""
    if name not in INSIDE_PROFILE_RUN_NAMES:
        raise ValueError(f"unknown inside sampling run name: {name}")
    command = [
        str(config.python_executable),
        str(EVAL_SCRIPT),
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
        "--inside-samples",
        str(config.inside_samples),
        "--inside-batch-size",
        str(config.inside_batch_size),
        "--inside-max-new-tokens",
        str(config.inside_max_new_tokens),
        "--inside-temperature",
        str(config.inside_temperature),
        "--inside-top-p",
        str(config.inside_top_p),
        "--inside-pooling",
        config.inside_pooling,
        "--inside-embedding-threshold",
        str(config.inside_embedding_threshold),
        "--json",
        str(config.result_path(name)),
        "--profile-json",
        str(config.profile_path(name)),
    ]
    if config.offline:
        command.append("--offline")
    if config.limit is not None:
        command.extend(["--limit", str(config.limit)])
    if config.manifold_questions is not None:
        command.extend(["--manifold-questions", str(config.manifold_questions)])
    if config.length_bucketed_batches:
        command.append("--length-bucketed-batches")
    if name in {"adaptive", "adaptive_selfcheck"}:
        command.extend([
            "--inside-adaptive-sampling",
            "--inside-min-samples",
            str(config.inside_min_samples),
            "--inside-sample-step",
            str(config.inside_sample_step),
            "--inside-stability-delta",
            str(config.inside_stability_delta),
        ])
    if name == "adaptive_selfcheck":
        command.extend([
            "--inside-selfcheck-early-stop",
            "--inside-selfcheck-min-overlap",
            str(config.inside_selfcheck_min_overlap),
            "--inside-selfcheck-support-threshold",
            str(config.inside_selfcheck_support_threshold),
            "--inside-selfcheck-refute-threshold",
            str(config.inside_selfcheck_refute_threshold),
        ])
    if config.dump_scores or config.dump_inside_samples:
        command.extend(["--dump-scores", str(config.score_dump_path(name))])
    if config.dump_inside_samples:
        command.append("--dump-inside-samples")
    return command


def build_inside_sampling_commands(config: InsideSamplingProfileConfig) -> dict[str, list[str]]:
    """Return commands for configured sampling profile runs in execution order."""
    return {name: build_eval_command(config, name) for name in config.run_names}


def run_inside_sampling_profile(
    config: InsideSamplingProfileConfig,
    *,
    clean: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the sampling profile workflow and write comparison artifacts."""
    if clean and config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    commands = build_inside_sampling_commands(config)
    command_log: dict[str, list[str]] = {}
    for name, command in commands.items():
        command_log[name] = command
        if not dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)

    if dry_run:
        payload = {
            "dry_run": True,
            "output_dir": str(config.output_dir),
            "commands": command_log,
            "run_names": tuple(command_log),
        }
    else:
        comparison = build_inside_sampling_comparison(
            {
                name: {
                    "result": config.result_path(name),
                    "profile": config.profile_path(name),
                }
                for name in command_log
            },
            baseline="fixed" if "fixed" in command_log else next(iter(command_log)),
            max_sample_ratios=_max_sample_ratios(config, command_log),
            max_inside_generation_seconds_ratio=config.max_inside_generation_seconds_ratio,
        )
        with open(config.comparison_report, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        payload = {
            "dry_run": False,
            "output_dir": str(config.output_dir),
            "commands": command_log,
            "run_names": tuple(command_log),
            "results": {name: str(config.result_path(name)) for name in command_log},
            "profiles": {name: str(config.profile_path(name)) for name in command_log},
            "score_dumps": (
                {name: str(config.score_dump_path(name)) for name in command_log}
                if config.dump_scores or config.dump_inside_samples
                else {}
            ),
            "comparison_report": str(config.comparison_report),
            "sample_efficiency_gate": comparison["sample_efficiency_gate"],
            "recommendation": comparison["recommendation"],
        }

    command_log_path = config.output_dir / "inside-sampling-profile-commands.json"
    with open(command_log_path, "w", encoding="utf-8") as f:
        json.dump(command_log, f, indent=2)
    payload["command_log"] = str(command_log_path)
    manifest = _write_artifact_manifest(config, payload)
    payload["artifact_manifest"] = str(config.artifact_manifest)
    payload["artifact_manifest_summary"] = manifest["summary"]
    return payload


def build_inside_sampling_comparison(
    runs: Mapping[str, Mapping[str, str | Path]],
    *,
    baseline: str = "fixed",
    max_sample_ratios: Mapping[str, float] | None = None,
    max_inside_generation_seconds_ratio: float | None = None,
) -> dict[str, Any]:
    """Build a report comparing sample count and runtime across run payloads."""
    if not runs:
        raise ValueError("at least one run is required.")
    if baseline not in runs:
        raise ValueError(f"baseline run {baseline!r} is not present.")
    max_sample_ratios = {} if max_sample_ratios is None else dict(max_sample_ratios)
    rows = []
    baseline_row = _inside_sampling_row(baseline, runs[baseline])
    baseline_samples = baseline_row["total_generated_samples"]
    baseline_inside_seconds = baseline_row["inside_generation_seconds"]
    for name, paths in runs.items():
        row = _inside_sampling_row(name, paths)
        row["sample_count_ratio_to_baseline"] = _ratio(row["total_generated_samples"], baseline_samples)
        row["inside_generation_seconds_ratio_to_baseline"] = _ratio(
            row["inside_generation_seconds"],
            baseline_inside_seconds,
        )
        rows.append(row)
    rows = sorted(rows, key=lambda item: (
        item["total_generated_samples"] is None,
        float("inf") if item["total_generated_samples"] is None else item["total_generated_samples"],
        item["inside_generation_seconds"] is None,
        float("inf") if item["inside_generation_seconds"] is None else item["inside_generation_seconds"],
        item["name"],
    ))
    gate = _sample_efficiency_gate(
        rows,
        baseline=baseline,
        max_sample_ratios=max_sample_ratios,
        max_inside_generation_seconds_ratio=max_inside_generation_seconds_ratio,
    )
    recommended = rows[0]["name"] if rows else None
    return {
        "schema_version": 1,
        "baseline": baseline,
        "runs": {row["name"]: row for row in rows},
        "leaderboard": rows,
        "sample_efficiency_gate": gate,
        "recommendation": {
            "recommended_run": recommended,
            "reason": "lowest_total_generated_samples_then_inside_generation_seconds",
        },
    }


def _inside_sampling_row(name: str, paths: Mapping[str, str | Path]) -> dict[str, Any]:
    result_path = Path(paths["result"])
    profile_path = Path(paths["profile"])
    result = _read_json(result_path)
    profile = _read_json(profile_path)
    inside = result.get("inside_sampling", {})
    if not isinstance(inside, Mapping):
        inside = {}
    phases = profile.get("phases", {})
    if not isinstance(phases, Mapping):
        phases = {}
    return {
        "name": name,
        "result_path": str(result_path),
        "profile_path": str(profile_path),
        "adaptive": inside.get("adaptive"),
        "selfcheck_early_stop": inside.get("selfcheck_early_stop"),
        "sampled": _optional_int(inside.get("sampled")),
        "total_generated_samples": _optional_int(inside.get("total_generated_samples")),
        "mean_samples_per_record": _optional_float(inside.get("mean_samples_per_record")),
        "mean_samples_per_sampled_record": _optional_float(inside.get("mean_samples_per_sampled_record")),
        "stopped_early": _optional_int(inside.get("stopped_early")),
        "stop_reason_counts": dict(inside.get("stop_reason_counts", {}))
        if isinstance(inside.get("stop_reason_counts", {}), Mapping)
        else {},
        "total_seconds": _optional_float(profile.get("total_seconds")),
        "inside_generation_seconds": _optional_float(phases.get("inside_generation")),
    }


def _sample_efficiency_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    max_sample_ratios: Mapping[str, float],
    max_inside_generation_seconds_ratio: float | None,
) -> dict[str, Any]:
    failures = []
    checked_runs = []
    for row in rows:
        name = str(row["name"])
        if name == baseline:
            continue
        checked = False
        max_sample_ratio = max_sample_ratios.get(name)
        if max_sample_ratio is not None:
            checked = True
            ratio = row.get("sample_count_ratio_to_baseline")
            if ratio is None or float(ratio) > float(max_sample_ratio):
                failures.append({
                    "run": name,
                    "metric": "sample_count_ratio_to_baseline",
                    "value": ratio,
                    "max_allowed": float(max_sample_ratio),
                })
        if max_inside_generation_seconds_ratio is not None:
            checked = True
            ratio = row.get("inside_generation_seconds_ratio_to_baseline")
            if ratio is None or float(ratio) > float(max_inside_generation_seconds_ratio):
                failures.append({
                    "run": name,
                    "metric": "inside_generation_seconds_ratio_to_baseline",
                    "value": ratio,
                    "max_allowed": float(max_inside_generation_seconds_ratio),
                })
        if checked:
            checked_runs.append(name)
    return {
        "passed": not failures,
        "baseline": baseline,
        "checked_runs": checked_runs,
        "failures": failures,
        "max_sample_ratios": dict(max_sample_ratios),
        "max_inside_generation_seconds_ratio": max_inside_generation_seconds_ratio,
    }


def _max_sample_ratios(
    config: InsideSamplingProfileConfig,
    commands: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    ratios = {}
    if "adaptive" in commands:
        ratios["adaptive"] = float(config.adaptive_max_sample_ratio)
    if "adaptive_selfcheck" in commands:
        ratios["adaptive_selfcheck"] = float(config.adaptive_selfcheck_max_sample_ratio)
    return ratios


def _normalize_run_names(run_names: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(str(name).strip() for name in run_names if str(name).strip())
    if not requested:
        raise ValueError("run_names must not be empty.")
    unknown = tuple(name for name in requested if name not in INSIDE_PROFILE_RUN_NAMES)
    if unknown:
        expected = ", ".join(INSIDE_PROFILE_RUN_NAMES)
        raise ValueError(f"unknown inside sampling run name(s): {', '.join(unknown)}; expected one of: {expected}.")
    if len(requested) != len(set(requested)):
        raise ValueError("run_names must not contain duplicates.")
    selected = tuple(name for name in INSIDE_PROFILE_RUN_NAMES if name in set(requested))
    return selected


def _write_artifact_manifest(config: InsideSamplingProfileConfig, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "command_log": payload.get("command_log"),
        "comparison_report": payload.get("comparison_report"),
    }
    for group_name in ("profiles", "results", "score_dumps"):
        for name, path in dict(payload.get(group_name, {})).items():
            artifacts[f"{group_name}.{name}"] = path
    manifest = build_artifact_manifest(
        artifacts,
        root=config.output_dir,
        metadata={
            "runner": "run_inside_sampling_profile",
            "model": config.model,
            "dtype": config.dtype,
            "layer": config.layer,
            "offline": config.offline,
            "inside_samples": config.inside_samples,
            "inside_min_samples": config.inside_min_samples,
            "inside_sample_step": config.inside_sample_step,
            "inside_stability_delta": config.inside_stability_delta,
            "run_names": tuple(config.run_names),
            "dry_run": bool(payload.get("dry_run")),
        },
    )
    config.artifact_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _ratio(value: int | float | None, baseline: int | float | None) -> float | None:
    if value is None or baseline is None or float(baseline) == 0.0:
        return None
    result = float(value) / float(baseline)
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _validate_unit_interval(value: float, name: str, *, lower: float = 0.0) -> None:
    if not (lower <= float(value) <= 1.0):
        raise ValueError(f"{name} must be in [{lower}, 1].")


def _parse_run_names(value: str) -> tuple[str, ...]:
    return _normalize_run_names(tuple(item.strip() for item in value.split(",") if item.strip()))


def _config_from_args(args: argparse.Namespace) -> InsideSamplingProfileConfig:
    return InsideSamplingProfileConfig(
        output_dir=Path(args.output_dir),
        model=args.model,
        dtype=args.dtype,
        layer=args.layer,
        limit=args.limit,
        manifold_questions=args.manifold_questions,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_length=args.max_length,
        hidden_state_capture=args.hidden_state_capture,
        progress_every=args.progress_every,
        offline=not args.real_truthfulqa,
        length_bucketed_batches=not args.no_length_bucketed_batches,
        python_executable=args.python,
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
        dump_scores=args.dump_scores,
        dump_inside_samples=args.dump_inside_samples,
        adaptive_max_sample_ratio=args.adaptive_max_sample_ratio,
        adaptive_selfcheck_max_sample_ratio=args.adaptive_selfcheck_max_sample_ratio,
        max_inside_generation_seconds_ratio=args.max_inside_generation_seconds_ratio,
        run_names=_parse_run_names(args.runs),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = run_inside_sampling_profile(
        _config_from_args(args),
        clean=bool(args.clean),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    gate = payload.get("sample_efficiency_gate")
    if args.fail_on_regression and gate is not None and not gate.get("passed", False):
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run fixed/adaptive/self-check INSIDE sampling profile comparison")
    parser.add_argument("--output-dir", required=True,
                        help="directory for profiles, result JSON, optional score dumps, and comparison report")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--manifold-questions", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--hidden-state-capture", default="outputs")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--real-truthfulqa", action="store_true",
                        help="load the configured model and TruthfulQA dataset instead of the offline fixture")
    parser.add_argument("--no-length-bucketed-batches", action="store_true")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used for eval_truthfulqa.py subprocesses")
    parser.add_argument("--inside-samples", type=int, default=5)
    parser.add_argument("--inside-batch-size", type=int, default=1)
    parser.add_argument("--inside-max-new-tokens", type=int, default=12)
    parser.add_argument("--inside-temperature", type=float, default=0.7)
    parser.add_argument("--inside-top-p", type=float, default=0.9)
    parser.add_argument("--inside-pooling", default="last", choices=("last", "mean"))
    parser.add_argument("--inside-embedding-threshold", type=float, default=0.90)
    parser.add_argument("--inside-min-samples", type=int, default=2)
    parser.add_argument("--inside-sample-step", type=int, default=1)
    parser.add_argument("--inside-stability-delta", type=float, default=0.05)
    parser.add_argument("--inside-selfcheck-min-overlap", type=float, default=0.65)
    parser.add_argument("--inside-selfcheck-support-threshold", type=float, default=0.60)
    parser.add_argument("--inside-selfcheck-refute-threshold", type=float, default=0.50)
    parser.add_argument("--dump-scores", action="store_true",
                        help="write per-run score dumps in addition to result/profile JSON")
    parser.add_argument("--dump-inside-samples", action="store_true",
                        help="include sampled continuation text in score dumps; implies --dump-scores")
    parser.add_argument("--adaptive-max-sample-ratio", type=float, default=1.0,
                        help="max adaptive/fixed total-generated-sample ratio for the sample gate")
    parser.add_argument("--adaptive-selfcheck-max-sample-ratio", type=float, default=1.0,
                        help="max adaptive_selfcheck/fixed total-generated-sample ratio for the sample gate")
    parser.add_argument("--max-inside-generation-seconds-ratio", type=float, default=None,
                        help="optional max candidate/fixed inside_generation runtime ratio")
    parser.add_argument("--runs", default="fixed,adaptive,adaptive_selfcheck",
                        help="comma-list of runs in canonical order: fixed,adaptive,adaptive_selfcheck")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="exit non-zero when the generated sample efficiency gate fails")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
