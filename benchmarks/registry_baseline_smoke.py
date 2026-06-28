"""Run a deterministic smoke check for registry-backed profile baselines."""

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

compare_registry_baseline = importlib.import_module(
    "benchmarks.compare_registry_baseline"
).compare_registry_baseline

from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

BASELINE_NAME = "registry-smoke-baseline"
BASELINE_VERSION = "0.3"


def build_registry_baseline_smoke(output_dir: Path) -> dict[str, Any]:
    """Write deterministic registry baseline fixtures and return gate reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_profile = output_dir / "profile-uncached.json"
    candidate_profile = output_dir / "profile-candidate.json"
    regression_profile = output_dir / "profile-regression.json"
    manifest_path = output_dir / "artifact-manifest.json"
    registry_path = output_dir / "registry.json"

    _write_json(baseline_profile, _profile_payload(total_seconds=100.0, forward_seconds=80.0))
    _write_json(candidate_profile, _profile_payload(total_seconds=104.0, forward_seconds=82.0))
    _write_json(regression_profile, _profile_payload(total_seconds=124.0, forward_seconds=98.0))
    _write_json(
        manifest_path,
        build_artifact_manifest({"profiles.uncached": baseline_profile}, root=output_dir),
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name=BASELINE_NAME,
        path=manifest_path,
        version=BASELINE_VERSION,
    ).save_json()

    pass_report = compare_registry_baseline(
        registry_path=registry_path,
        baseline_name=BASELINE_NAME,
        baseline_version=BASELINE_VERSION,
        candidate_profiles=(("candidate", candidate_profile),),
        max_total_ratio=1.10,
        notes=["deterministic registry baseline pass smoke"],
    )
    failure_report = compare_registry_baseline(
        registry_path=registry_path,
        baseline_name=BASELINE_NAME,
        baseline_version=BASELINE_VERSION,
        candidate_profiles=(("regression", regression_profile),),
        max_total_ratio=1.10,
        notes=["deterministic registry baseline expected failure smoke"],
    )

    if not pass_report["comparison"]["regression_gate"]["passed"]:
        raise AssertionError("registry baseline smoke candidate unexpectedly failed.")
    if failure_report["comparison"]["regression_gate"]["passed"]:
        raise AssertionError("registry baseline smoke regression was not detected.")

    _write_json(output_dir / "registry_baseline_gate_pass_report.json", pass_report)
    _write_json(output_dir / "registry_baseline_gate_expected_failure_report.json", failure_report)
    return {
        "output_dir": str(output_dir),
        "registry": str(registry_path),
        "manifest": str(manifest_path),
        "pass_report": pass_report,
        "expected_failure_report": failure_report,
    }


def _profile_payload(*, total_seconds: float, forward_seconds: float) -> dict[str, Any]:
    return {
        "total_seconds": total_seconds,
        "phases": {
            "load_model": 10.0,
            "forced_answer_forward": forward_seconds,
            "score_postprocess": 5.0,
        },
        "summary": {
            "bottleneck": "forced_answer_forward",
            "top_phases": [{"name": "forced_answer_forward", "seconds": forward_seconds}],
            "groups": {"model_forward": {"seconds": forward_seconds}},
            "throughput": {"forced_answer_records_per_second": 10.0},
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the registry baseline smoke check."""
    if args.output_dir:
        output_dir = Path(args.output_dir)
        result = build_registry_baseline_smoke(output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="eigentruth-registry-baseline-") as tmpdir:
            result = build_registry_baseline_smoke(Path(tmpdir))
    pass_gate = result["pass_report"]["comparison"]["regression_gate"]
    failure_gate = result["expected_failure_report"]["comparison"]["regression_gate"]
    print(
        "registry_baseline_smoke_ok "
        f"pass_checked={pass_gate['checked_runs']} "
        f"expected_failures={len(failure_gate['failures'])}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic registry baseline gate smoke checks")
    parser.add_argument("--output-dir", default=None,
                        help="optional directory to write synthetic registry baseline reports")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
