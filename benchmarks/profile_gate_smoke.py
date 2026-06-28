"""Run a deterministic smoke check for profile regression gates.

This script does not load a model or measure wall-clock performance. It writes
small synthetic profile payloads and verifies that ``compare_profiles.py`` can
both pass an acceptable candidate and flag a known regression. Use it as a
stable CI/local check for the gate machinery; use real ``eval_truthfulqa.py
--profile-json`` outputs for actual performance claims.
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


DEFAULT_MAX_TOTAL_RATIO = 1.10
DEFAULT_MAX_PHASE_RATIO = 1.10
DEFAULT_MIN_THROUGHPUT_RATIO = 0.90
THROUGHPUT_METRIC = "forced_answer_records_per_second"
PHASE_NAME = "forced_answer_forward"


def build_profile_gate_smoke(output_dir: Path) -> dict[str, Any]:
    """Write deterministic profile fixtures and return pass/fail gate reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "profile_baseline.json"
    candidate_path = output_dir / "profile_candidate.json"
    regression_path = output_dir / "profile_regression.json"

    _write_json(baseline_path, _profile_payload(
        total_seconds=100.0,
        forced_answer_forward=70.0,
        score_postprocess=8.0,
        throughput=10.0,
    ))
    _write_json(candidate_path, _profile_payload(
        total_seconds=104.0,
        forced_answer_forward=73.0,
        score_postprocess=8.5,
        throughput=9.6,
    ))
    _write_json(regression_path, _profile_payload(
        total_seconds=118.0,
        forced_answer_forward=82.0,
        score_postprocess=8.5,
        throughput=8.2,
    ))

    gate_kwargs = {
        "max_total_ratio": DEFAULT_MAX_TOTAL_RATIO,
        "max_phase_ratios": {PHASE_NAME: DEFAULT_MAX_PHASE_RATIO},
        "min_throughput_ratios": {THROUGHPUT_METRIC: DEFAULT_MIN_THROUGHPUT_RATIO},
    }
    pass_report = build_profile_comparison(
        [("baseline", baseline_path), ("candidate", candidate_path)],
        baseline="baseline",
        notes=["deterministic profile gate pass smoke"],
        **gate_kwargs,
    )
    failure_report = build_profile_comparison(
        [("baseline", baseline_path), ("regression", regression_path)],
        baseline="baseline",
        notes=["deterministic profile gate expected failure smoke"],
        **gate_kwargs,
    )

    if not pass_report["regression_gate"]["passed"]:
        raise AssertionError("profile gate smoke candidate unexpectedly failed.")
    if failure_report["regression_gate"]["passed"]:
        raise AssertionError("profile gate smoke regression was not detected.")

    _write_json(output_dir / "profile_gate_pass_report.json", pass_report)
    _write_json(output_dir / "profile_gate_expected_failure_report.json", failure_report)
    return {
        "output_dir": str(output_dir),
        "pass_report": pass_report,
        "expected_failure_report": failure_report,
    }


def _profile_payload(
    *,
    total_seconds: float,
    forced_answer_forward: float,
    score_postprocess: float,
    throughput: float,
) -> dict[str, Any]:
    phases = {
        "load_model": 10.0,
        PHASE_NAME: forced_answer_forward,
        "score_postprocess": score_postprocess,
    }
    return {
        "total_seconds": total_seconds,
        "phases": phases,
        "summary": {
            "bottleneck": PHASE_NAME,
            "top_phases": [
                {
                    "name": PHASE_NAME,
                    "seconds": forced_answer_forward,
                    "share": forced_answer_forward / total_seconds,
                },
                {
                    "name": "score_postprocess",
                    "seconds": score_postprocess,
                    "share": score_postprocess / total_seconds,
                },
            ],
            "groups": {
                "model_forward": {
                    "seconds": forced_answer_forward,
                    "share": forced_answer_forward / total_seconds,
                },
                "postprocess": {
                    "seconds": score_postprocess,
                    "share": score_postprocess / total_seconds,
                },
            },
            "throughput": {THROUGHPUT_METRIC: throughput},
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the profile gate smoke check."""
    if args.output_dir:
        output_dir = Path(args.output_dir)
        result = build_profile_gate_smoke(output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="eigentruth-profile-gate-") as tmpdir:
            result = build_profile_gate_smoke(Path(tmpdir))
    pass_gate = result["pass_report"]["regression_gate"]
    failure_gate = result["expected_failure_report"]["regression_gate"]
    print(
        "profile_gate_smoke_ok "
        f"pass_checked={pass_gate['checked_runs']} "
        f"expected_failures={len(failure_gate['failures'])}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic profile regression gate smoke checks")
    parser.add_argument("--output-dir", default=None,
                        help="optional directory to write synthetic profiles and gate reports")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
