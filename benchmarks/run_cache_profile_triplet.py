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
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "benchmarks" / "eval_truthfulqa.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.compare_profiles import build_profile_comparison  # noqa: E402


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
    max_length: int = 64
    hidden_state_capture: str = "outputs"
    eval_reps_cache_shard_size: int = 4
    cached_max_total_ratio: float = 1.10
    cache_only_max_total_ratio: float = 0.35
    python_executable: str = sys.executable
    progress_every: int = 0
    length_bucketed_batches: bool = True
    offline: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.limit is not None and int(self.limit) < 0:
            raise ValueError("limit must be >=0.")
        if self.manifold_questions is not None and int(self.manifold_questions) < 1:
            raise ValueError("manifold_questions must be >=1.")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be >=1.")
        if int(self.max_length) < 1:
            raise ValueError("max_length must be >=1.")
        if int(self.eval_reps_cache_shard_size) < 1:
            raise ValueError("eval_reps_cache_shard_size must be >=1.")
        if float(self.cached_max_total_ratio) < 0:
            raise ValueError("cached_max_total_ratio must be non-negative.")
        if float(self.cache_only_max_total_ratio) < 0:
            raise ValueError("cache_only_max_total_ratio must be non-negative.")
        object.__setattr__(self, "dtype", str(self.dtype))
        object.__setattr__(self, "hidden_state_capture", str(self.hidden_state_capture))

    @property
    def statement_encoding_cache(self) -> Path:
        return self.output_dir / "statement-encodings.json"

    @property
    def layer_stats_cache(self) -> Path:
        return self.output_dir / "layer-stats.pt"

    @property
    def eval_reps_cache(self) -> Path:
        return self.output_dir / "eval-reps-cache"

    @property
    def comparison_report(self) -> Path:
        return self.output_dir / "cache-profile-comparison.json"

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

    if name == "uncached":
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
    """Return commands for all profile triplet runs in execution order."""
    return {
        "uncached": build_eval_command(config, "uncached"),
        "cached": build_eval_command(config, "cached"),
        "cache_only": build_eval_command(config, "cache_only"),
    }


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
        }
    else:
        comparison = build_profile_comparison(
            [
                ("uncached", config.profile_path("uncached")),
                ("cached", config.profile_path("cached")),
                ("cache_only", config.profile_path("cache_only")),
            ],
            baseline="uncached",
            notes=["same-machine uncached/cached/cache-only TruthfulQA profile triplet"],
            max_run_total_ratios={
                "cached": config.cached_max_total_ratio,
                "cache_only": config.cache_only_max_total_ratio,
            },
        )
        with open(config.comparison_report, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        payload = {
            "dry_run": False,
            "output_dir": str(config.output_dir),
            "commands": command_log,
            "profiles": {
                "uncached": str(config.profile_path("uncached")),
                "cached": str(config.profile_path("cached")),
                "cache_only": str(config.profile_path("cache_only")),
            },
            "results": {
                "uncached": str(config.result_path("uncached")),
                "cached": str(config.result_path("cached")),
                "cache_only": str(config.result_path("cache_only")),
            },
            "comparison_report": str(config.comparison_report),
            "regression_gate": comparison.get("regression_gate"),
        }

    command_log_path = config.output_dir / "cache-profile-triplet-commands.json"
    with open(command_log_path, "w", encoding="utf-8") as f:
        json.dump(command_log, f, indent=2)
    payload["command_log"] = str(command_log_path)
    return payload


def _config_from_args(args: argparse.Namespace) -> CacheProfileTripletConfig:
    return CacheProfileTripletConfig(
        output_dir=Path(args.output_dir),
        model=args.model,
        dtype=args.dtype,
        layer=args.layer,
        limit=args.limit,
        manifold_questions=args.manifold_questions,
        batch_size=args.batch_size,
        max_length=args.max_length,
        hidden_state_capture=args.hidden_state_capture,
        eval_reps_cache_shard_size=args.eval_reps_cache_shard_size,
        cached_max_total_ratio=args.cached_max_total_ratio,
        cache_only_max_total_ratio=args.cache_only_max_total_ratio,
        python_executable=args.python,
        progress_every=args.progress_every,
        length_bucketed_batches=not args.no_length_bucketed_batches,
        offline=not args.real_truthfulqa,
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
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--hidden-state-capture", default="outputs",
                        help="hidden state capture mode passed to eval_truthfulqa.py")
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
    parser.add_argument("--clean", action="store_true",
                        help="remove --output-dir before running")
    parser.add_argument("--dry-run", action="store_true",
                        help="only write and print the commands; do not run eval_truthfulqa.py")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="exit non-zero when the generated comparison gate fails")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
