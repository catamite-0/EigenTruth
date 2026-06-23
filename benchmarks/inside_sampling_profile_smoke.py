"""Run a deterministic smoke check for INSIDE sampling profile gates.

This script does not load a model or measure wall-clock performance. It writes
small synthetic ``eval_truthfulqa.py`` result/profile payloads and verifies that
``run_inside_sampling_profile.py`` can promote a sample-efficient adaptive run
and fail closed on a known sampling-cost regression.
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

build_inside_sampling_comparison = importlib.import_module(
    "benchmarks.run_inside_sampling_profile"
).build_inside_sampling_comparison

DEFAULT_ADAPTIVE_MAX_SAMPLE_RATIO = 0.80
DEFAULT_ADAPTIVE_SELFCHECK_MAX_SAMPLE_RATIO = 0.60
DEFAULT_MAX_INSIDE_GENERATION_SECONDS_RATIO = 0.80


def build_inside_sampling_profile_smoke(output_dir: Path) -> dict[str, Any]:
    """Write deterministic INSIDE sampling fixtures and return gate reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pass_runs = _write_sampling_fixtures(
        output_dir / "pass",
        {
            "fixed": {"samples": 20, "seconds": 10.0, "stopped": 0, "reasons": {}},
            "adaptive": {"samples": 14, "seconds": 7.0, "stopped": 3, "reasons": {"stability_delta": 3}},
            "adaptive_selfcheck": {
                "samples": 10,
                "seconds": 5.5,
                "stopped": 4,
                "reasons": {"selfcheck_refute_threshold_guaranteed": 4},
            },
        },
    )
    failure_runs = _write_sampling_fixtures(
        output_dir / "failure",
        {
            "fixed": {"samples": 20, "seconds": 10.0, "stopped": 0, "reasons": {}},
            "adaptive": {"samples": 18, "seconds": 9.0, "stopped": 1, "reasons": {"stability_delta": 1}},
            "adaptive_selfcheck": {
                "samples": 15,
                "seconds": 9.0,
                "stopped": 1,
                "reasons": {"selfcheck_refute_threshold_guaranteed": 1},
            },
        },
    )

    gate_kwargs = {
        "max_sample_ratios": {
            "adaptive": DEFAULT_ADAPTIVE_MAX_SAMPLE_RATIO,
            "adaptive_selfcheck": DEFAULT_ADAPTIVE_SELFCHECK_MAX_SAMPLE_RATIO,
        },
        "max_inside_generation_seconds_ratio": DEFAULT_MAX_INSIDE_GENERATION_SECONDS_RATIO,
    }
    pass_report = build_inside_sampling_comparison(pass_runs, **gate_kwargs)
    failure_report = build_inside_sampling_comparison(failure_runs, **gate_kwargs)

    if not pass_report["sample_efficiency_gate"]["passed"]:
        raise AssertionError("inside sampling profile smoke candidate unexpectedly failed.")
    if pass_report["recommendation"]["recommended_run"] != "adaptive_selfcheck":
        raise AssertionError("inside sampling profile smoke did not recommend the lowest-sample run.")
    if failure_report["sample_efficiency_gate"]["passed"]:
        raise AssertionError("inside sampling profile smoke regression was not detected.")

    _write_json(output_dir / "inside_sampling_profile_pass_report.json", pass_report)
    _write_json(output_dir / "inside_sampling_profile_expected_failure_report.json", failure_report)
    return {
        "output_dir": str(output_dir),
        "pass_report": pass_report,
        "expected_failure_report": failure_report,
    }


def _write_sampling_fixtures(
    output_dir: Path,
    fixtures: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name, fixture in fixtures.items():
        result_path = output_dir / f"result-{name}.json"
        profile_path = output_dir / f"profile-{name}.json"
        samples = int(fixture["samples"])
        seconds = float(fixture["seconds"])
        _write_json(
            result_path,
            {
                "inside_sampling": {
                    "adaptive": name != "fixed",
                    "selfcheck_early_stop": name == "adaptive_selfcheck",
                    "sampled": 4,
                    "total_generated_samples": samples,
                    "mean_samples_per_record": samples / 4,
                    "mean_samples_per_sampled_record": samples / 4,
                    "stopped_early": int(fixture["stopped"]),
                    "stop_reason_counts": dict(fixture["reasons"]),
                }
            },
        )
        _write_json(
            profile_path,
            {
                "total_seconds": seconds + 1.0,
                "phases": {"inside_generation": seconds},
            },
        )
        runs[name] = {"result": result_path, "profile": profile_path}
    return runs


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the INSIDE sampling profile smoke check."""
    if args.output_dir:
        output_dir = Path(args.output_dir)
        result = build_inside_sampling_profile_smoke(output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="eigentruth-inside-sampling-profile-") as tmpdir:
            result = build_inside_sampling_profile_smoke(Path(tmpdir))
    pass_gate = result["pass_report"]["sample_efficiency_gate"]
    failure_gate = result["expected_failure_report"]["sample_efficiency_gate"]
    print(
        "inside_sampling_profile_smoke_ok "
        f"recommended_run={result['pass_report']['recommendation']['recommended_run']} "
        f"pass_checked={pass_gate['checked_runs']} "
        f"expected_failures={len(failure_gate['failures'])}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic INSIDE sampling profile smoke checks")
    parser.add_argument("--output-dir", default=None,
                        help="optional directory to write synthetic sampling profile reports")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
