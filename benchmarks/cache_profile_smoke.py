"""Run a deterministic smoke check for cache profile comparison gates.

This script does not load a model or measure wall-clock performance. It writes
small synthetic profile payloads for uncached, cached, and cache-only benchmark
paths, then verifies that ``compare_profiles.py`` can enforce different total
time ratios for each cached path.
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

build_profile_comparison = importlib.import_module("benchmarks.compare_profiles").build_profile_comparison

DEFAULT_CACHED_MAX_TOTAL_RATIO = 0.75
DEFAULT_CACHE_ONLY_MAX_TOTAL_RATIO = 0.20


def build_cache_profile_smoke(output_dir: Path) -> dict[str, Any]:
    """Write deterministic cache-profile fixtures and return gate reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "profile_uncached.json"
    cached_path = output_dir / "profile_cached.json"
    cache_only_path = output_dir / "profile_cache_only.json"
    cache_only_regression_path = output_dir / "profile_cache_only_regression.json"

    _write_json(baseline_path, _profile_payload(
        total_seconds=120.0,
        phases={
            "load_model": 10.0,
            "build_layer_stats": 40.0,
            "forced_answer_forward": 60.0,
            "score_postprocess": 6.0,
        },
        throughput=4.0,
    ))
    _write_json(cached_path, _profile_payload(
        total_seconds=72.0,
        phases={
            "load_model": 10.0,
            "load_layer_stats_cache": 2.0,
            "forced_answer_forward": 52.0,
            "read_eval_reps_cache_batch": 2.0,
            "score_postprocess": 4.0,
        },
        throughput=6.5,
    ))
    _write_json(cache_only_path, _profile_payload(
        total_seconds=18.0,
        phases={
            "read_cache_metadata": 0.5,
            "load_layer_stats_cache": 2.0,
            "read_eval_reps_cache_batch": 10.0,
            "score_postprocess": 4.0,
        },
        throughput=20.0,
    ))
    _write_json(cache_only_regression_path, _profile_payload(
        total_seconds=32.0,
        phases={
            "read_cache_metadata": 0.5,
            "load_layer_stats_cache": 3.0,
            "read_eval_reps_cache_batch": 22.0,
            "score_postprocess": 5.0,
        },
        throughput=11.0,
    ))

    gate_kwargs = {
        "max_run_total_ratios": {
            "cached": DEFAULT_CACHED_MAX_TOTAL_RATIO,
            "cache_only": DEFAULT_CACHE_ONLY_MAX_TOTAL_RATIO,
            "cache_only_regression": DEFAULT_CACHE_ONLY_MAX_TOTAL_RATIO,
        },
    }
    pass_report = build_profile_comparison(
        [
            ("uncached", baseline_path),
            ("cached", cached_path),
            ("cache_only", cache_only_path),
        ],
        baseline="uncached",
        notes=["deterministic cache profile gate pass smoke"],
        **gate_kwargs,
    )
    failure_report = build_profile_comparison(
        [
            ("uncached", baseline_path),
            ("cached", cached_path),
            ("cache_only_regression", cache_only_regression_path),
        ],
        baseline="uncached",
        notes=["deterministic cache profile gate expected failure smoke"],
        **gate_kwargs,
    )

    if not pass_report["regression_gate"]["passed"]:
        raise AssertionError("cache profile smoke candidate unexpectedly failed.")
    if failure_report["regression_gate"]["passed"]:
        raise AssertionError("cache profile smoke regression was not detected.")

    _write_json(output_dir / "cache_profile_gate_pass_report.json", pass_report)
    _write_json(output_dir / "cache_profile_gate_expected_failure_report.json", failure_report)
    return {
        "output_dir": str(output_dir),
        "pass_report": pass_report,
        "expected_failure_report": failure_report,
    }


def _profile_payload(
    *,
    total_seconds: float,
    phases: dict[str, float],
    throughput: float,
) -> dict[str, Any]:
    top_phases = [
        {
            "name": name,
            "seconds": seconds,
            "share": seconds / total_seconds,
        }
        for name, seconds in sorted(phases.items(), key=lambda item: item[1], reverse=True)[:5]
    ]
    return {
        "total_seconds": total_seconds,
        "phases": phases,
        "summary": {
            "bottleneck": top_phases[0]["name"] if top_phases else None,
            "top_phases": top_phases,
            "groups": {
                "model_forward": {
                    "seconds": phases.get("build_layer_stats", 0.0) + phases.get("forced_answer_forward", 0.0),
                    "share": (
                        phases.get("build_layer_stats", 0.0) + phases.get("forced_answer_forward", 0.0)
                    ) / total_seconds,
                },
                "cache_io": {
                    "seconds": (
                        phases.get("read_cache_metadata", 0.0)
                        + phases.get("load_layer_stats_cache", 0.0)
                        + phases.get("read_eval_reps_cache_batch", 0.0)
                    ),
                    "share": (
                        phases.get("read_cache_metadata", 0.0)
                        + phases.get("load_layer_stats_cache", 0.0)
                        + phases.get("read_eval_reps_cache_batch", 0.0)
                    ) / total_seconds,
                },
            },
            "throughput": {"end_to_end_eval_records_per_second": throughput},
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the cache profile smoke check."""
    if args.output_dir:
        output_dir = Path(args.output_dir)
        result = build_cache_profile_smoke(output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="eigentruth-cache-profile-") as tmpdir:
            result = build_cache_profile_smoke(Path(tmpdir))
    pass_gate = result["pass_report"]["regression_gate"]
    failure_gate = result["expected_failure_report"]["regression_gate"]
    print(
        "cache_profile_smoke_ok "
        f"pass_checked={pass_gate['checked_runs']} "
        f"expected_failures={len(failure_gate['failures'])}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic cache profile gate smoke checks")
    parser.add_argument("--output-dir", default=None,
                        help="optional directory to write synthetic profiles and gate reports")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
