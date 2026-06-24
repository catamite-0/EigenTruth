"""Compare cache-profile matrix wall-clock time across worker counts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_cache_profile_matrix import (  # noqa: E402
    MATRIX_MODES,
    CacheProfileMatrixConfig,
    _parse_int_list,
    _parse_max_batch_token_budgets,
    _parse_prefix_kv_cache_modes,
    _parse_str_list,
    run_matrix,
)
from eigentruth.registry import build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class CacheWorkerSweepConfig:
    """Configuration for a worker-count sweep over cache-profile matrices."""

    output_dir: Path
    worker_counts: Sequence[int] = (1, 2)
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
    eval_reps_shard_read_cache_size: int = 2
    cached_max_total_ratio: float = 1.10
    cache_only_max_total_ratio: float = 0.35
    python_executable: str = sys.executable
    progress_every: int = 0
    length_bucketed_batches: bool = True
    offline: bool = True
    shared_cache_dir: Path | None = None
    matrix_mode: str = "triplet"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.shared_cache_dir is not None:
            object.__setattr__(self, "shared_cache_dir", Path(self.shared_cache_dir))
        worker_counts = tuple(int(value) for value in self.worker_counts)
        if not worker_counts:
            raise ValueError("worker_counts must not be empty.")
        if any(value < 1 for value in worker_counts):
            raise ValueError("worker_counts values must be >=1.")
        if len(worker_counts) != len(set(worker_counts)):
            raise ValueError("worker_counts must not contain duplicate values.")
        object.__setattr__(self, "worker_counts", worker_counts)
        object.__setattr__(self, "layers", tuple(int(layer) for layer in self.layers))
        object.__setattr__(self, "batch_sizes", tuple(int(batch_size) for batch_size in self.batch_sizes))
        object.__setattr__(
            self,
            "hidden_state_captures",
            tuple(str(capture) for capture in self.hidden_state_captures),
        )
        if int(self.eval_reps_shard_read_cache_size) < 1:
            raise ValueError("eval_reps_shard_read_cache_size must be >=1.")
        object.__setattr__(
            self,
            "eval_reps_shard_read_cache_size",
            int(self.eval_reps_shard_read_cache_size),
        )

    @property
    def report_path(self) -> Path:
        return self.output_dir / "cache-worker-sweep-report.json"

    @property
    def artifact_manifest(self) -> Path:
        return self.output_dir / "artifact-manifest.json"


def run_worker_sweep(
    config: CacheWorkerSweepConfig,
    *,
    clean: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the same matrix for each worker count and summarize wall-clock time."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    worker_reports = []
    for worker_count in config.worker_counts:
        matrix_config = _matrix_config_for_worker(config, worker_count)
        matrix_report = run_matrix(matrix_config, clean=clean, dry_run=dry_run)
        worker_reports.append(_worker_summary(worker_count, matrix_report))

    leaderboard = _leaderboard(worker_reports)
    decision = _worker_sweep_decision(worker_reports, leaderboard)
    report = {
        "schema_version": 1,
        "workflow": "cache_worker_sweep",
        "dry_run": dry_run,
        "config": {
            "worker_counts": tuple(config.worker_counts),
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
            "eval_reps_shard_read_cache_size": config.eval_reps_shard_read_cache_size,
            "offline": config.offline,
            "length_bucketed_batches": config.length_bucketed_batches,
            "shared_cache_dir": None if config.shared_cache_dir is None else str(config.shared_cache_dir),
            "matrix_mode": config.matrix_mode,
        },
        "worker_reports": tuple(worker_reports),
        "leaderboard": leaderboard,
        "worker_sweep_decision": decision,
        "report_path": str(config.report_path),
        "artifact_manifest": str(config.artifact_manifest),
    }
    config.report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_artifact_manifest(config, report)
    return report


def _matrix_config_for_worker(config: CacheWorkerSweepConfig, worker_count: int) -> CacheProfileMatrixConfig:
    return CacheProfileMatrixConfig(
        output_dir=config.output_dir / f"workers_{worker_count}",
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
        eval_reps_shard_read_cache_size=config.eval_reps_shard_read_cache_size,
        cached_max_total_ratio=config.cached_max_total_ratio,
        cache_only_max_total_ratio=config.cache_only_max_total_ratio,
        python_executable=config.python_executable,
        progress_every=config.progress_every,
        length_bucketed_batches=config.length_bucketed_batches,
        offline=config.offline,
        shared_cache_dir=_shared_cache_dir_for_worker(config, worker_count),
        matrix_mode=config.matrix_mode,
        max_workers=worker_count,
    )


def _shared_cache_dir_for_worker(config: CacheWorkerSweepConfig, worker_count: int) -> Path | None:
    if config.shared_cache_dir is None:
        return None
    return config.shared_cache_dir / f"workers_{worker_count}"


def _worker_summary(worker_count: int, matrix_report: Mapping[str, Any]) -> dict[str, Any]:
    execution = dict(matrix_report.get("execution") or {})
    decision = dict(matrix_report.get("matrix_decision") or {})
    recommended = decision.get("recommended")
    if not isinstance(recommended, Mapping):
        recommended = {}
    return {
        "worker_count": int(worker_count),
        "matrix_status": decision.get("status"),
        "wall_clock_seconds": execution.get("wall_clock_seconds"),
        "cell_count": execution.get("cell_count"),
        "recommended_cell": decision.get("recommended_cell"),
        "recommended_cache_only_total_seconds": recommended.get("cache_only_total_seconds"),
        "recommended_uncached_forced_answer_forward_seconds": recommended.get(
            "uncached_forced_answer_forward_seconds"
        ),
        "recommended_truth_proj_auroc": recommended.get("truth_proj_auroc"),
        "candidate_count": decision.get("candidate_count"),
        "failed_cells": tuple(decision.get("failed_cells") or ()),
        "blocking_reasons": tuple(decision.get("blocking_reasons") or ()),
        "matrix_report": matrix_report.get("report_path"),
        "matrix_artifact_manifest": matrix_report.get("artifact_manifest"),
    }


def _leaderboard(worker_reports: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    scored = []
    for report in worker_reports:
        wall_clock = _float_or_none(report.get("wall_clock_seconds"))
        if wall_clock is None:
            continue
        row = dict(report)
        row["_sort_value"] = wall_clock
        scored.append(row)
    ordered = sorted(scored, key=lambda item: (item["_sort_value"], int(item["worker_count"])))
    return tuple({key: value for key, value in item.items() if key != "_sort_value"} for item in ordered)


def _worker_sweep_decision(
    worker_reports: Sequence[Mapping[str, Any]],
    leaderboard: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    material = [report for report in worker_reports if report.get("matrix_status") != "dry_run"]
    if not material:
        return {
            "status": "dry_run",
            "recommended_worker_count": None,
            "recommended": None,
            "blocking_reasons": ("worker sweep was run in dry_run mode; no performance profiles were executed",),
        }

    blocked = tuple(report for report in material if report.get("matrix_status") == "blocked")
    passing_by_worker = {
        int(report["worker_count"]): report
        for report in material
        if report.get("matrix_status") == "promote"
    }
    passing_leaderboard = tuple(
        report for report in leaderboard if int(report.get("worker_count", 0)) in passing_by_worker
    )
    recommended = None if not passing_leaderboard else dict(passing_leaderboard[0])
    blocking_reasons = []
    if blocked:
        blocking_reasons.append("one or more worker-count matrix runs were blocked")
    if recommended is None:
        blocking_reasons.append("no worker-count matrix run produced a promoted candidate")
    status = "blocked" if blocked else ("promote" if recommended is not None else "no_candidate")
    return {
        "status": status,
        "recommended_worker_count": None if recommended is None else recommended.get("worker_count"),
        "recommended": recommended,
        "blocking_reasons": tuple(blocking_reasons),
    }


def _float_or_none(value: Any) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return float(value)


def _write_artifact_manifest(config: CacheWorkerSweepConfig, report: Mapping[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, str | Path | None] = {"worker_sweep_report": config.report_path}
    for worker_report in report.get("worker_reports", ()):
        if not isinstance(worker_report, Mapping):
            continue
        worker_count = worker_report.get("worker_count")
        artifacts[f"workers.{worker_count}.matrix_report"] = worker_report.get("matrix_report")
        artifacts[f"workers.{worker_count}.matrix_manifest"] = worker_report.get("matrix_artifact_manifest")
    manifest = build_artifact_manifest(
        artifacts,
        root=config.output_dir,
        metadata={
            "runner": "run_cache_worker_sweep",
            "worker_counts": tuple(config.worker_counts),
            "model": config.model,
            "dtype": config.dtype,
            "layers": tuple(config.layers),
            "batch_sizes": tuple(config.batch_sizes),
            "hidden_state_captures": tuple(config.hidden_state_captures),
            "offline": config.offline,
            "matrix_mode": config.matrix_mode,
            "eval_reps_cache_shard_size": config.eval_reps_cache_shard_size,
            "eval_reps_shard_read_cache_size": config.eval_reps_shard_read_cache_size,
            "dry_run": bool(report.get("dry_run")),
            "recommended_worker_count": dict(report.get("worker_sweep_decision") or {}).get(
                "recommended_worker_count"
            ),
        },
    )
    config.artifact_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _parse_worker_counts(value: str) -> tuple[int, ...]:
    return _parse_int_list(value, name="worker_counts")


def _config_from_args(args: argparse.Namespace) -> CacheWorkerSweepConfig:
    return CacheWorkerSweepConfig(
        output_dir=Path(args.output_dir),
        worker_counts=_parse_worker_counts(args.worker_counts),
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
        eval_reps_shard_read_cache_size=args.eval_reps_shard_read_cache_size,
        cached_max_total_ratio=args.cached_max_total_ratio,
        cache_only_max_total_ratio=args.cache_only_max_total_ratio,
        python_executable=args.python,
        progress_every=args.progress_every,
        length_bucketed_batches=not args.no_length_bucketed_batches,
        offline=not args.real_truthfulqa,
        shared_cache_dir=Path(args.shared_cache_dir) if args.shared_cache_dir else None,
        matrix_mode=args.matrix_mode,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = run_worker_sweep(_config_from_args(args), clean=bool(args.clean), dry_run=bool(args.dry_run))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_blocked and report["worker_sweep_decision"]["status"] != "promote":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare cache-profile matrix wall-clock time by worker count")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker-counts", default="1,2",
                        help="comma-list of matrix max worker counts to compare, e.g. 1,2,4")
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
    parser.add_argument("--eval-reps-shard-read-cache-size", type=int, default=2,
                        help="number of eval-reps cache shards cached by cached/cache-only reader runs")
    parser.add_argument("--cached-max-total-ratio", type=float, default=1.10)
    parser.add_argument("--cache-only-max-total-ratio", type=float, default=0.35)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-length-bucketed-batches", action="store_true")
    parser.add_argument("--shared-cache-dir", default=None,
                        help="optional shared-cache root; each worker count gets an isolated subdirectory")
    parser.add_argument("--matrix-mode", default="triplet", choices=MATRIX_MODES)
    parser.add_argument("--real-truthfulqa", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless worker_sweep_decision.status is promote")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
