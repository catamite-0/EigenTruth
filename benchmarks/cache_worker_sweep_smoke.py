"""Run a deterministic smoke check for cache worker sweep decisions.

This script does not load a model. It replaces the matrix runner with a small
synthetic report generator, then verifies that ``run_cache_worker_sweep.py``
can recommend the fastest promoted worker count and fail closed when any worker
matrix blocks.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

worker_sweep_module = importlib.import_module("benchmarks.run_cache_worker_sweep")

from eigentruth.registry import build_artifact_manifest  # noqa: E402


def build_cache_worker_sweep_smoke(output_dir: Path) -> dict[str, Any]:
    """Write deterministic worker-sweep reports and return pass/blocked payloads."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pass_report = _run_fake_worker_sweep(output_dir / "pass", blocked_worker=None)
    blocked_report = _run_fake_worker_sweep(output_dir / "blocked", blocked_worker=2)

    if pass_report["worker_sweep_decision"]["status"] != "promote":
        raise AssertionError("worker sweep smoke candidate unexpectedly failed.")
    if pass_report["worker_sweep_decision"]["recommended_worker_count"] != 2:
        raise AssertionError("worker sweep smoke did not recommend the fastest promoted worker.")
    if blocked_report["worker_sweep_decision"]["status"] != "blocked":
        raise AssertionError("worker sweep smoke blocked matrix was not detected.")

    _write_json(output_dir / "cache_worker_sweep_pass_report.json", pass_report)
    _write_json(output_dir / "cache_worker_sweep_expected_blocked_report.json", blocked_report)
    return {
        "output_dir": str(output_dir),
        "pass_report": pass_report,
        "expected_blocked_report": blocked_report,
    }


def _run_fake_worker_sweep(output_dir: Path, *, blocked_worker: int | None) -> dict[str, Any]:
    original_run_matrix = worker_sweep_module.run_matrix

    def fake_run_matrix(config: Any, *, clean: bool, dry_run: bool) -> dict[str, Any]:
        del clean, dry_run
        return _fake_matrix_report(config, blocked=config.max_workers == blocked_worker)

    worker_sweep_module.run_matrix = fake_run_matrix
    try:
        return worker_sweep_module.run_worker_sweep(
            worker_sweep_module.CacheWorkerSweepConfig(
                output_dir=output_dir,
                shared_cache_dir=output_dir / "shared-cache",
                worker_counts=(1, 2),
                model="synthetic-local",
                layers=(-2,),
                batch_sizes=(1,),
            ),
            clean=True,
            dry_run=False,
        )
    finally:
        worker_sweep_module.run_matrix = original_run_matrix


def _fake_matrix_report(config: Any, *, blocked: bool) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    wall_clock_seconds = 120.0 if config.max_workers == 1 else 80.0
    status = "blocked" if blocked else "promote"
    recommended = None if blocked else {
        "id": f"worker_{config.max_workers}_cell",
        "cache_only_total_seconds": 0.1,
        "truth_proj_auroc": 0.9,
    }
    payload = {
        "dry_run": False,
        "report_path": str(config.report_path),
        "artifact_manifest": str(config.artifact_manifest),
        "execution": {
            "wall_clock_seconds": wall_clock_seconds,
            "cell_count": 1,
            "max_workers": config.max_workers,
        },
        "matrix_decision": {
            "status": status,
            "recommended_cell": None if recommended is None else recommended["id"],
            "recommended": recommended,
            "candidate_count": 0 if blocked else 1,
            "failed_cells": ("synthetic_blocked_cell",) if blocked else (),
            "blocking_reasons": ("synthetic blocked matrix",) if blocked else (),
        },
    }
    _write_json(config.report_path, payload)
    _write_json(
        config.artifact_manifest,
        build_artifact_manifest({"matrix_report": config.report_path}, root=config.output_dir),
    )
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the cache worker sweep smoke check."""
    if args.output_dir:
        output_dir = Path(args.output_dir)
        result = build_cache_worker_sweep_smoke(output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="eigentruth-cache-worker-sweep-") as tmpdir:
            result = build_cache_worker_sweep_smoke(Path(tmpdir))
    pass_decision = result["pass_report"]["worker_sweep_decision"]
    blocked_decision = result["expected_blocked_report"]["worker_sweep_decision"]
    print(
        "cache_worker_sweep_smoke_ok "
        f"recommended_worker={pass_decision['recommended_worker_count']} "
        f"blocked_status={blocked_decision['status']}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic cache worker sweep smoke checks")
    parser.add_argument("--output-dir", default=None,
                        help="optional directory to write synthetic worker sweep reports")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
