"""Run uncached, cached, and cache-only TruthfulQA profile runs.

This helper performs a real local benchmark workflow: it calls
``eval_truthfulqa.py`` three times in the same output directory, then writes a
``compare_profiles.py`` report with separate total-time gates for cached and
cache-only paths. It may download or load the configured model.
"""

from __future__ import annotations

import argparse
import json
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

from benchmarks.compare_profiles import build_profile_comparison  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402

TRIPLET_RUN_NAMES = ("uncached", "cached", "cache_only")


@dataclass(frozen=True)
class CacheProfileTripletConfig:
    """Configuration for one uncached/cached/cache-only profile triplet."""

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
    prefix_kv_cache: bool = False
    eval_reps_cache_shard_size: int = 4
    cached_max_total_ratio: float = 1.10
    cache_only_max_total_ratio: float = 0.35
    python_executable: str = sys.executable
    progress_every: int = 0
    length_bucketed_batches: bool = True
    offline: bool = True
    statement_encoding_cache_path: Path | None = None
    layer_stats_cache_path: Path | None = None
    eval_reps_cache_path: Path | None = None
    uncached_cache_mode: str = "refresh"
    run_names: Sequence[str] = TRIPLET_RUN_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.statement_encoding_cache_path is not None:
            object.__setattr__(self, "statement_encoding_cache_path", Path(self.statement_encoding_cache_path))
        if self.layer_stats_cache_path is not None:
            object.__setattr__(self, "layer_stats_cache_path", Path(self.layer_stats_cache_path))
        if self.eval_reps_cache_path is not None:
            object.__setattr__(self, "eval_reps_cache_path", Path(self.eval_reps_cache_path))
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
        if int(self.eval_reps_cache_shard_size) < 1:
            raise ValueError("eval_reps_cache_shard_size must be >=1.")
        if float(self.cached_max_total_ratio) < 0:
            raise ValueError("cached_max_total_ratio must be non-negative.")
        if float(self.cache_only_max_total_ratio) < 0:
            raise ValueError("cache_only_max_total_ratio must be non-negative.")
        if self.uncached_cache_mode not in {"refresh", "warm_start", "none"}:
            raise ValueError("uncached_cache_mode must be one of: refresh, warm_start, none.")
        if self.prefix_kv_cache and str(self.hidden_state_capture) != "outputs":
            raise ValueError("prefix_kv_cache requires hidden_state_capture='outputs'.")
        run_names = _normalize_run_names(self.run_names)
        object.__setattr__(self, "dtype", str(self.dtype))
        object.__setattr__(self, "max_batch_tokens", int(self.max_batch_tokens))
        object.__setattr__(self, "hidden_state_capture", str(self.hidden_state_capture))
        object.__setattr__(self, "uncached_cache_mode", str(self.uncached_cache_mode))
        object.__setattr__(self, "run_names", run_names)

    @property
    def statement_encoding_cache(self) -> Path:
        return self.statement_encoding_cache_path or self.output_dir / "statement-encodings.json"

    @property
    def layer_stats_cache(self) -> Path:
        return self.layer_stats_cache_path or self.output_dir / "layer-stats.pt"

    @property
    def eval_reps_cache(self) -> Path:
        return self.eval_reps_cache_path or self.output_dir / "eval-reps-cache"

    @property
    def comparison_report(self) -> Path:
        return self.output_dir / "cache-profile-comparison.json"

    @property
    def artifact_manifest(self) -> Path:
        return self.output_dir / "artifact-manifest.json"

    def profile_path(self, name: str) -> Path:
        return self.output_dir / f"profile-{name}.json"

    def result_path(self, name: str) -> Path:
        return self.output_dir / f"result-{name}.json"


def build_eval_command(config: CacheProfileTripletConfig, name: str) -> list[str]:
    """Build one eval command for a profile triplet run."""
    base = [
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
        "--json",
        str(config.result_path(name)),
        "--profile-json",
        str(config.profile_path(name)),
    ]
    if config.offline:
        base.append("--offline")
    if config.limit is not None:
        base.extend(["--limit", str(config.limit)])
    if config.manifold_questions is not None:
        base.extend(["--manifold-questions", str(config.manifold_questions)])
    if config.length_bucketed_batches:
        base.append("--length-bucketed-batches")
    if config.prefix_kv_cache and name != "cache_only":
        base.append("--prefix-kv-cache")

    if name == "uncached" and config.uncached_cache_mode == "refresh":
        return [
            *base,
            "--statement-encoding-cache",
            str(config.statement_encoding_cache),
            "--refresh-statement-encoding-cache",
            "--layer-stats-cache",
            str(config.layer_stats_cache),
            "--refresh-layer-stats-cache",
            "--eval-reps-cache",
            str(config.eval_reps_cache),
            "--eval-reps-cache-shard-size",
            str(config.eval_reps_cache_shard_size),
            "--refresh-eval-reps-cache",
        ]
    if name == "uncached" and config.uncached_cache_mode == "warm_start":
        return [
            *base,
            "--statement-encoding-cache",
            str(config.statement_encoding_cache),
            "--layer-stats-cache",
            str(config.layer_stats_cache),
        ]
    if name == "uncached" and config.uncached_cache_mode == "none":
        return base
    if name == "cached":
        return [
            *base,
            "--statement-encoding-cache",
            str(config.statement_encoding_cache),
            "--layer-stats-cache",
            str(config.layer_stats_cache),
            "--eval-reps-cache",
            str(config.eval_reps_cache),
        ]
    if name == "cache_only":
        return [
            *base,
            "--cache-only",
            "--layer-stats-cache",
            str(config.layer_stats_cache),
            "--eval-reps-cache",
            str(config.eval_reps_cache),
        ]
    raise ValueError(f"unknown triplet run name: {name}")


def build_triplet_commands(config: CacheProfileTripletConfig) -> dict[str, list[str]]:
    """Return commands for configured profile triplet runs in execution order."""
    return {name: build_eval_command(config, name) for name in config.run_names}


def run_triplet(
    config: CacheProfileTripletConfig,
    *,
    clean: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the profile triplet and write a comparison report."""
    if clean and config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    commands = build_triplet_commands(config)
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
            "caches": _cache_paths(config),
            "uncached_cache_mode": config.uncached_cache_mode,
        }
    else:
        comparison = _build_comparison_if_available(config)
        payload = {
            "dry_run": False,
            "output_dir": str(config.output_dir),
            "commands": command_log,
            "run_names": tuple(command_log),
            "profiles": {name: str(config.profile_path(name)) for name in command_log},
            "results": {name: str(config.result_path(name)) for name in command_log},
            "comparison_report": str(config.comparison_report) if comparison is not None else None,
            "comparison_skipped_reason": None if comparison is not None else "baseline run 'uncached' was not executed",
            "regression_gate": comparison.get("regression_gate") if comparison is not None else None,
            "caches": _cache_paths(config),
            "uncached_cache_mode": config.uncached_cache_mode,
        }

    command_log_path = config.output_dir / "cache-profile-triplet-commands.json"
    with open(command_log_path, "w", encoding="utf-8") as f:
        json.dump(command_log, f, indent=2)
    payload["command_log"] = str(command_log_path)
    manifest = _write_artifact_manifest(config, payload)
    payload["artifact_manifest"] = str(config.artifact_manifest)
    payload["artifact_manifest_summary"] = manifest["summary"]
    return payload


def _normalize_run_names(run_names: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(str(name).strip() for name in run_names if str(name).strip())
    if not requested:
        raise ValueError("run_names must not be empty.")
    unknown = tuple(name for name in requested if name not in TRIPLET_RUN_NAMES)
    if unknown:
        expected = ", ".join(TRIPLET_RUN_NAMES)
        raise ValueError(f"unknown triplet run name(s): {', '.join(unknown)}; expected one of: {expected}.")
    if len(requested) != len(set(requested)):
        raise ValueError("run_names must not contain duplicates.")
    selected = tuple(name for name in TRIPLET_RUN_NAMES if name in set(requested))
    return selected


def _build_comparison_if_available(config: CacheProfileTripletConfig) -> dict[str, Any] | None:
    if "uncached" not in config.run_names:
        return None
    profiles = [(name, config.profile_path(name)) for name in config.run_names]
    max_run_total_ratios = {}
    if "cached" in config.run_names:
        max_run_total_ratios["cached"] = config.cached_max_total_ratio
    if "cache_only" in config.run_names:
        max_run_total_ratios["cache_only"] = config.cache_only_max_total_ratio
    comparison = build_profile_comparison(
        profiles,
        baseline="uncached",
        notes=["same-machine uncached/cached/cache-only TruthfulQA profile triplet"],
        max_run_total_ratios=max_run_total_ratios,
    )
    with open(config.comparison_report, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    return comparison


def _cache_paths(config: CacheProfileTripletConfig) -> dict[str, str]:
    return {
        "statement_encoding_cache": str(config.statement_encoding_cache),
        "layer_stats_cache": str(config.layer_stats_cache),
        "eval_reps_cache": str(config.eval_reps_cache),
    }


def _write_artifact_manifest(config: CacheProfileTripletConfig, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "command_log": payload.get("command_log"),
        "comparison_report": payload.get("comparison_report"),
    }
    for group_name in ("profiles", "results", "caches"):
        for name, path in dict(payload.get(group_name, {})).items():
            artifacts[f"{group_name}.{name}"] = path
    manifest = build_artifact_manifest(
        artifacts,
        root=config.output_dir,
        metadata={
            "runner": "run_cache_profile_triplet",
            "model": config.model,
            "dtype": config.dtype,
            "layer": config.layer,
            "batch_size": config.batch_size,
            "max_batch_tokens": config.max_batch_tokens,
            "hidden_state_capture": config.hidden_state_capture,
            "prefix_kv_cache": config.prefix_kv_cache,
            "offline": config.offline,
            "run_names": tuple(config.run_names),
            "uncached_cache_mode": config.uncached_cache_mode,
            "dry_run": bool(payload.get("dry_run")),
        },
    )
    config.artifact_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _parse_run_names(value: str) -> tuple[str, ...]:
    return _normalize_run_names(tuple(item.strip() for item in value.split(",") if item.strip()))


def _config_from_args(args: argparse.Namespace) -> CacheProfileTripletConfig:
    return CacheProfileTripletConfig(
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
        prefix_kv_cache=args.prefix_kv_cache,
        eval_reps_cache_shard_size=args.eval_reps_cache_shard_size,
        cached_max_total_ratio=args.cached_max_total_ratio,
        cache_only_max_total_ratio=args.cache_only_max_total_ratio,
        python_executable=args.python,
        progress_every=args.progress_every,
        length_bucketed_batches=not args.no_length_bucketed_batches,
        offline=not args.real_truthfulqa,
        statement_encoding_cache_path=Path(args.statement_encoding_cache) if args.statement_encoding_cache else None,
        layer_stats_cache_path=Path(args.layer_stats_cache) if args.layer_stats_cache else None,
        eval_reps_cache_path=Path(args.eval_reps_cache) if args.eval_reps_cache else None,
        uncached_cache_mode=args.uncached_cache_mode,
        run_names=_parse_run_names(args.runs),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = run_triplet(
        _config_from_args(args),
        clean=bool(args.clean),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    gate = payload.get("regression_gate")
    if args.fail_on_regression and gate is not None and not gate.get("passed", False):
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run uncached/cached/cache-only TruthfulQA profile triplet")
    parser.add_argument("--output-dir", required=True,
                        help="directory for caches, profile payloads, result JSON, and comparison report")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2",
                        help="model id passed to eval_truthfulqa.py; may be downloaded by transformers")
    parser.add_argument("--dtype", default="float32",
                        help="dtype passed to eval_truthfulqa.py")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None,
                        help="eval question limit passed to eval_truthfulqa.py; 0 means all")
    parser.add_argument("--manifold-questions", type=int, default=None,
                        help="warmup question count passed to eval_truthfulqa.py")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=0,
                        help="padded-token budget per eval_truthfulqa.py warmup/eval forward batch; 0 disables")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--hidden-state-capture", default="outputs",
                        help="hidden state capture mode passed to eval_truthfulqa.py")
    parser.add_argument("--prefix-kv-cache", action="store_true",
                        help="pass --prefix-kv-cache to non-cache-only eval_truthfulqa.py runs")
    parser.add_argument("--eval-reps-cache-shard-size", type=int, default=4)
    parser.add_argument("--cached-max-total-ratio", type=float, default=1.10,
                        help="max cached/uncached total-time ratio for the comparison gate")
    parser.add_argument("--cache-only-max-total-ratio", type=float, default=0.35,
                        help="max cache-only/uncached total-time ratio for the comparison gate")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used for eval_truthfulqa.py subprocesses")
    parser.add_argument("--no-length-bucketed-batches", action="store_true",
                        help="omit --length-bucketed-batches from all eval runs")
    parser.add_argument("--real-truthfulqa", action="store_true",
                        help="load the configured model and TruthfulQA dataset instead of the offline fixture")
    parser.add_argument("--statement-encoding-cache", default=None,
                        help="override the statement encoding cache path used by triplet commands")
    parser.add_argument("--layer-stats-cache", default=None,
                        help="override the layer stats cache path used by triplet commands")
    parser.add_argument("--eval-reps-cache", default=None,
                        help="override the eval reps cache path used by triplet commands")
    parser.add_argument("--uncached-cache-mode", default="refresh", choices=["refresh", "warm_start", "none"],
                        help="cache behavior for the uncached run: refresh all caches, warm-start from "
                             "statement/layer caches without eval reps, or avoid caches")
    parser.add_argument("--runs", default="uncached,cached,cache_only",
                        help="comma-list of triplet runs to execute in canonical order: "
                             "uncached,cached,cache_only")
    parser.add_argument("--clean", action="store_true",
                        help="remove --output-dir before running")
    parser.add_argument("--dry-run", action="store_true",
                        help="only write and print the commands; do not run eval_truthfulqa.py")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="exit non-zero when the generated comparison gate fails")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
