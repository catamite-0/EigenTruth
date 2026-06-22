"""Run a matrix of uncached/cached/cache-only profile triplets.

This is a thin orchestration layer over ``run_cache_profile_triplet.py``. It is
intended for same-machine experiments such as comparing batch sizes, target
layers, and hidden-state capture modes before committing to a larger benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_cache_profile_triplet import CacheProfileTripletConfig, run_triplet  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402

MATRIX_MODES = ("triplet", "rescore")


@dataclass(frozen=True)
class CacheProfileMatrixConfig:
    """Configuration for a profile triplet matrix."""

    output_dir: Path
    model: str = "sshleifer/tiny-gpt2"
    dtype: str = "float32"
    layers: Sequence[int] = (-1,)
    batch_sizes: Sequence[int] = (4,)
    hidden_state_captures: Sequence[str] = ("outputs",)
    limit: int | None = None
    manifold_questions: int | None = None
    max_length: int = 64
    eval_reps_cache_shard_size: int = 4
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
        layers = tuple(int(layer) for layer in self.layers)
        batch_sizes = tuple(int(batch_size) for batch_size in self.batch_sizes)
        captures = tuple(str(capture).strip() for capture in self.hidden_state_captures if str(capture).strip())
        if not layers:
            raise ValueError("layers must not be empty.")
        if not batch_sizes:
            raise ValueError("batch_sizes must not be empty.")
        if any(batch_size < 1 for batch_size in batch_sizes):
            raise ValueError("batch_sizes must be >=1.")
        if not captures:
            raise ValueError("hidden_state_captures must not be empty.")
        matrix_mode = str(self.matrix_mode)
        if matrix_mode not in MATRIX_MODES:
            raise ValueError("matrix_mode must be one of: triplet, rescore.")
        if matrix_mode == "rescore" and self.shared_cache_dir is None:
            raise ValueError("matrix_mode='rescore' requires shared_cache_dir.")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "batch_sizes", batch_sizes)
        object.__setattr__(self, "hidden_state_captures", captures)
        object.__setattr__(self, "matrix_mode", matrix_mode)

    @property
    def report_path(self) -> Path:
        return self.output_dir / "cache-profile-matrix-report.json"

    @property
    def artifact_manifest(self) -> Path:
        return self.output_dir / "artifact-manifest.json"


def matrix_cells(config: CacheProfileMatrixConfig) -> tuple[dict[str, Any], ...]:
    """Return matrix cells in deterministic execution order."""
    cells = []
    for layer, batch_size, capture in itertools.product(
        config.layers,
        config.batch_sizes,
        config.hidden_state_captures,
    ):
        cell_id = f"layer_{layer}_batch_{batch_size}_capture_{capture}".replace("-", "m")
        cells.append({
            "id": cell_id,
            "layer": layer,
            "batch_size": batch_size,
            "hidden_state_capture": capture,
        })
    return tuple(cells)


def triplet_config_for_cell(
    config: CacheProfileMatrixConfig,
    cell: dict[str, Any],
    *,
    uncached_cache_mode: str = "refresh",
    run_names: Sequence[str] = ("uncached", "cached", "cache_only"),
) -> CacheProfileTripletConfig:
    """Build a triplet config for one matrix cell."""
    shared_paths = _shared_cache_paths(config, cell)
    return CacheProfileTripletConfig(
        output_dir=config.output_dir / str(cell["id"]),
        model=config.model,
        dtype=config.dtype,
        layer=int(cell["layer"]),
        limit=config.limit,
        manifold_questions=config.manifold_questions,
        batch_size=int(cell["batch_size"]),
        max_length=config.max_length,
        hidden_state_capture=str(cell["hidden_state_capture"]),
        eval_reps_cache_shard_size=config.eval_reps_cache_shard_size,
        cached_max_total_ratio=config.cached_max_total_ratio,
        cache_only_max_total_ratio=config.cache_only_max_total_ratio,
        python_executable=config.python_executable,
        progress_every=config.progress_every,
        length_bucketed_batches=config.length_bucketed_batches,
        offline=config.offline,
        statement_encoding_cache_path=shared_paths.get("statement_encoding_cache"),
        layer_stats_cache_path=shared_paths.get("layer_stats_cache"),
        eval_reps_cache_path=shared_paths.get("eval_reps_cache"),
        uncached_cache_mode=uncached_cache_mode,
        run_names=run_names,
    )


def run_matrix(
    config: CacheProfileMatrixConfig,
    *,
    clean: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run all matrix cells and write a matrix report."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    seen_shared_cache_groups: set[str] = set()
    for cell in matrix_cells(config):
        shared_cache_group = _shared_cache_group(config, cell)
        first_shared_group_run = shared_cache_group is None or shared_cache_group not in seen_shared_cache_groups
        uncached_cache_mode, run_names = _cell_execution_plan(
            config,
            first_shared_group_run=first_shared_group_run,
        )
        triplet_config = triplet_config_for_cell(
            config,
            cell,
            uncached_cache_mode=uncached_cache_mode,
            run_names=run_names,
        )
        triplet_payload = run_triplet(triplet_config, clean=clean, dry_run=dry_run)
        if shared_cache_group is not None:
            seen_shared_cache_groups.add(shared_cache_group)
        cells.append({
            **cell,
            "output_dir": str(triplet_config.output_dir),
            "shared_cache_group": shared_cache_group,
            "uncached_cache_mode": uncached_cache_mode,
            "run_names": tuple(run_names),
            "triplet": triplet_payload,
            "summary": _cell_summary(triplet_payload),
        })

    report = {
        "dry_run": dry_run,
        "config": {
            "model": config.model,
            "dtype": config.dtype,
            "layers": tuple(config.layers),
            "batch_sizes": tuple(config.batch_sizes),
            "hidden_state_captures": tuple(config.hidden_state_captures),
            "limit": config.limit,
            "manifold_questions": config.manifold_questions,
            "max_length": config.max_length,
            "eval_reps_cache_shard_size": config.eval_reps_cache_shard_size,
            "offline": config.offline,
            "length_bucketed_batches": config.length_bucketed_batches,
            "shared_cache_dir": None if config.shared_cache_dir is None else str(config.shared_cache_dir),
            "shared_cache_root": None if config.shared_cache_dir is None else str(_shared_cache_root(config)),
            "matrix_mode": config.matrix_mode,
        },
        "cells": cells,
        "leaderboard": _leaderboard(cells),
        "report_path": str(config.report_path),
        "artifact_manifest": str(config.artifact_manifest),
    }
    with open(config.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _write_artifact_manifest(config, report)
    return report


def _cell_execution_plan(
    config: CacheProfileMatrixConfig,
    *,
    first_shared_group_run: bool,
) -> tuple[str, tuple[str, ...]]:
    if config.matrix_mode == "triplet":
        uncached_cache_mode = "refresh" if first_shared_group_run else "warm_start"
        return uncached_cache_mode, ("uncached", "cached", "cache_only")
    if first_shared_group_run:
        return "refresh", ("uncached", "cached", "cache_only")
    return "warm_start", ("cache_only",)


def _shared_cache_paths(config: CacheProfileMatrixConfig, cell: Mapping[str, Any]) -> dict[str, Path]:
    if config.shared_cache_dir is None:
        return {}
    root = _shared_cache_root(config)
    group = _shared_cache_group_name(cell)
    return {
        "statement_encoding_cache": root / "statement-encodings.json",
        "layer_stats_cache": root / group / "layer-stats.pt",
        "eval_reps_cache": root / group / "eval-reps-cache",
    }


def _write_artifact_manifest(config: CacheProfileMatrixConfig, report: Mapping[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, str | Path | None] = {"matrix_report": config.report_path}
    for cell in report.get("cells", ()):
        if not isinstance(cell, Mapping):
            continue
        triplet = cell.get("triplet", {})
        if isinstance(triplet, Mapping):
            artifacts[f"cells.{cell.get('id')}.triplet_manifest"] = triplet.get("artifact_manifest")
    manifest = build_artifact_manifest(
        artifacts,
        root=config.output_dir,
        metadata={
            "runner": "run_cache_profile_matrix",
            "model": config.model,
            "dtype": config.dtype,
            "layers": tuple(config.layers),
            "batch_sizes": tuple(config.batch_sizes),
            "hidden_state_captures": tuple(config.hidden_state_captures),
            "offline": config.offline,
            "matrix_mode": config.matrix_mode,
            "dry_run": bool(report.get("dry_run")),
            "shared_cache_dir": None if config.shared_cache_dir is None else str(config.shared_cache_dir),
        },
    )
    config.artifact_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _shared_cache_group(config: CacheProfileMatrixConfig, cell: Mapping[str, Any]) -> str | None:
    if config.shared_cache_dir is None:
        return None
    return str(_shared_cache_root(config) / _shared_cache_group_name(cell))


def _shared_cache_root(config: CacheProfileMatrixConfig) -> Path:
    if config.shared_cache_dir is None:
        raise ValueError("shared_cache_dir is not configured.")
    payload = {
        "model": config.model,
        "dtype": config.dtype,
        "offline": config.offline,
        "limit": config.limit,
        "manifold_questions": config.manifold_questions,
        "max_length": config.max_length,
        "length_bucketed_batches": config.length_bucketed_batches,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return config.shared_cache_dir / f"{_safe_path_component(config.model)}-{digest}"


def _shared_cache_group_name(cell: Mapping[str, Any]) -> str:
    return (
        f"layer_{_layer_component(int(cell['layer']))}"
        f"_capture_{_safe_path_component(str(cell['hidden_state_capture']))}"
    )


def _layer_component(layer: int) -> str:
    return str(int(layer)).replace("-", "m")


def _safe_path_component(value: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in str(value)]
    text = "".join(chars).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or "value"


def _cell_summary(triplet_payload: dict[str, Any]) -> dict[str, Any]:
    if triplet_payload.get("dry_run"):
        return {
            "dry_run": True,
            "commands": {
                name: _shell_join(command)
                for name, command in dict(triplet_payload.get("commands", {})).items()
            },
        }
    comparison_path = triplet_payload.get("comparison_report")
    if not comparison_path:
        return _cell_summary_without_comparison(triplet_payload)
    comparison = json.loads(Path(str(comparison_path)).read_text(encoding="utf-8"))
    runs = {str(run["name"]): run for run in comparison.get("runs", [])}
    auroc = _result_auroc(triplet_payload)
    return {
        "dry_run": False,
        "regression_gate": comparison.get("regression_gate"),
        "fastest": comparison.get("fastest"),
        "truth_proj_auroc": auroc.get("truth_proj"),
        "totals": {
            name: {
                "total_seconds": run.get("total_seconds"),
                "bottleneck": _run_bottleneck(run),
                "speedup_vs_baseline": run.get("total_delta", {}).get("speedup_vs_baseline"),
                "ratio_to_baseline": run.get("total_delta", {}).get("ratio_to_baseline"),
            }
            for name, run in runs.items()
        },
    }


def _cell_summary_without_comparison(triplet_payload: dict[str, Any]) -> dict[str, Any]:
    profiles = {
        str(name): str(path)
        for name, path in dict(triplet_payload.get("profiles", {})).items()
        if path
    }
    auroc = _result_auroc(triplet_payload)
    totals = {}
    for name, path in profiles.items():
        if not Path(path).exists():
            continue
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        totals[name] = {
            "total_seconds": run.get("total_seconds"),
            "bottleneck": _run_bottleneck(run),
            "speedup_vs_baseline": None,
            "ratio_to_baseline": None,
        }
    return {
        "dry_run": False,
        "regression_gate": None,
        "fastest": None,
        "truth_proj_auroc": auroc.get("truth_proj"),
        "comparison_skipped_reason": triplet_payload.get("comparison_skipped_reason"),
        "totals": totals,
    }


def _run_bottleneck(run: Mapping[str, Any]) -> Any:
    """Return a run bottleneck from current or legacy comparison payloads."""
    if run.get("bottleneck") is not None:
        return run.get("bottleneck")
    summary = run.get("summary")
    if isinstance(summary, Mapping):
        return summary.get("bottleneck")
    return None


def _result_auroc(triplet_payload: dict[str, Any]) -> dict[str, float]:
    results = dict(triplet_payload.get("results", {}))
    result_path = results.get("cache_only") or results.get("cached") or results.get("uncached")
    if not result_path:
        return {}
    payload = json.loads(Path(str(result_path)).read_text(encoding="utf-8"))
    auroc = payload.get("auroc", {})
    if not isinstance(auroc, dict):
        return {}
    return {str(name): float(value) for name, value in auroc.items() if isinstance(value, int | float)}


def _leaderboard(cells: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    scored = []
    for cell in cells:
        summary = dict(cell.get("summary", {}))
        totals = dict(summary.get("totals", {}))
        cache_only = dict(totals.get("cache_only", {}))
        total_seconds = cache_only.get("total_seconds")
        if total_seconds is None:
            continue
        scored.append({
            "id": cell["id"],
            "layer": cell["layer"],
            "batch_size": cell["batch_size"],
            "hidden_state_capture": cell["hidden_state_capture"],
            "uncached_cache_mode": cell.get("uncached_cache_mode"),
            "shared_cache_group": cell.get("shared_cache_group"),
            "cache_only_total_seconds": total_seconds,
            "cache_only_speedup_vs_baseline": cache_only.get("speedup_vs_baseline"),
            "truth_proj_auroc": summary.get("truth_proj_auroc"),
            "gate_passed": _gate_passed(summary.get("regression_gate")),
        })
    return tuple(sorted(scored, key=lambda item: (item["cache_only_total_seconds"], str(item["id"]))))


def _gate_passed(regression_gate: Any) -> bool | None:
    if regression_gate is None:
        return None
    return bool(dict(regression_gate).get("passed", False))


def _shell_join(command: Sequence[str]) -> str:
    return " ".join(_quote_shell_arg(part) for part in command)


def _quote_shell_arg(value: Any) -> str:
    text = str(value)
    if not text:
        return "''"
    if all(ch.isalnum() or ch in "-_./:=+" for ch in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _parse_int_list(value: str, *, name: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError(f"{name} must contain at least one integer.")
    return items


def _parse_str_list(value: str, *, name: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError(f"{name} must contain at least one value.")
    return items


def _config_from_args(args: argparse.Namespace) -> CacheProfileMatrixConfig:
    return CacheProfileMatrixConfig(
        output_dir=Path(args.output_dir),
        model=args.model,
        dtype=args.dtype,
        layers=_parse_int_list(args.layers, name="layers"),
        batch_sizes=_parse_int_list(args.batch_sizes, name="batch_sizes"),
        hidden_state_captures=_parse_str_list(args.hidden_state_captures, name="hidden_state_captures"),
        limit=args.limit,
        manifold_questions=args.manifold_questions,
        max_length=args.max_length,
        eval_reps_cache_shard_size=args.eval_reps_cache_shard_size,
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
    report = run_matrix(_config_from_args(args), clean=bool(args.clean), dry_run=bool(args.dry_run))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_regression:
        failures = [
            cell
            for cell in report["cells"]
            if not cell["summary"].get("dry_run")
            and cell["summary"].get("regression_gate") is not None
            and not dict(cell["summary"].get("regression_gate") or {}).get("passed", False)
        ]
        if failures:
            raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a matrix of cache profile triplets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--layers", default="-1",
                        help="comma-list of target layers, e.g. -16,-12,-10")
    parser.add_argument("--batch-sizes", default="4",
                        help="comma-list of eval batch sizes")
    parser.add_argument("--hidden-state-captures", default="outputs",
                        help="comma-list of capture modes, e.g. outputs,hooks")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--manifold-questions", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--eval-reps-cache-shard-size", type=int, default=4)
    parser.add_argument("--cached-max-total-ratio", type=float, default=1.10)
    parser.add_argument("--cache-only-max-total-ratio", type=float, default=0.35)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-length-bucketed-batches", action="store_true")
    parser.add_argument("--shared-cache-dir", default=None,
                        help="optional directory for caches shared across cells with the same layer/capture; "
                             "later batch-size cells warm-start from statement/layer caches")
    parser.add_argument("--matrix-mode", default="triplet", choices=MATRIX_MODES,
                        help="triplet runs every cell as uncached/cached/cache-only; rescore runs the first shared "
                             "cache group cell as a full triplet and repeated group cells as cache-only")
    parser.add_argument("--real-truthfulqa", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-regression", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
