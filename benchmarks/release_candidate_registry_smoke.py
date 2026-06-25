"""No-model smoke checks for the release-candidate registry workflow."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_release_candidate_registry_workflow import (  # noqa: E402
    ReleaseCandidateRegistryWorkflowConfig,
    run_release_candidate_registry_workflow,
)
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

SMOKE_NAME = "release-candidate-smoke"
SMOKE_VERSION = "0.1"
READINESS_KEY = f"benchmark_manifest:{SMOKE_NAME}-readiness:{SMOKE_VERSION}"
ROUTE_KEY = f"benchmark_manifest:{SMOKE_NAME}-route:{SMOKE_VERSION}"
PROMOTED_RECORD_KEY = f"benchmark_manifest:{SMOKE_NAME}:{SMOKE_VERSION}"
BLOCKED_RECORD_KEY = f"benchmark_manifest:{SMOKE_NAME}-blocked:{SMOKE_VERSION}"


def build_release_candidate_registry_smoke(output_dir: Path) -> dict[str, Any]:
    """Build synthetic release-candidate fixtures and return promoted/blocked reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_registry_path = output_dir / "baseline-registry.json"
    release_registry_path = output_dir / "release-registry.json"

    _write_readiness_baseline_manifest(
        output_dir / "readiness",
        registry_path=baseline_registry_path,
    )
    _write_route_baseline_manifest(
        output_dir / "route",
        registry_path=baseline_registry_path,
    )

    promoted_report = run_release_candidate_registry_workflow(
        _workflow_config(
            output_dir=output_dir,
            release_registry_path=release_registry_path,
            name=SMOKE_NAME,
            min_best_quality_auroc=0.70,
            report_prefix="promoted",
        )
    )
    blocked_report = run_release_candidate_registry_workflow(
        _workflow_config(
            output_dir=output_dir,
            release_registry_path=release_registry_path,
            name=f"{SMOKE_NAME}-blocked",
            min_best_quality_auroc=0.95,
            report_prefix="blocked",
        )
    )

    if promoted_report["decision"]["status"] != "promote":
        raise AssertionError("release candidate registry smoke candidate unexpectedly blocked.")
    if blocked_report["decision"]["status"] != "blocked":
        raise AssertionError("release candidate registry smoke regression was not blocked.")

    registry = ArtifactRegistry.load_json(release_registry_path)
    registry.get(PROMOTED_RECORD_KEY)
    try:
        registry.get(BLOCKED_RECORD_KEY)
    except KeyError:
        pass
    else:
        raise AssertionError("blocked release candidate was unexpectedly registered.")

    return {
        "output_dir": str(output_dir),
        "baseline_registry": str(baseline_registry_path),
        "release_registry": str(release_registry_path),
        "promoted_report": promoted_report,
        "blocked_report": blocked_report,
    }


def _workflow_config(
    *,
    output_dir: Path,
    release_registry_path: Path,
    name: str,
    min_best_quality_auroc: float,
    report_prefix: str,
) -> ReleaseCandidateRegistryWorkflowConfig:
    return ReleaseCandidateRegistryWorkflowConfig(
        readiness_registry_path=output_dir / "baseline-registry.json",
        route_registry_path=output_dir / "baseline-registry.json",
        release_registry_path=release_registry_path,
        readiness_baseline_keys=(READINESS_KEY,),
        route_baseline_keys=(ROUTE_KEY,),
        name=name,
        version=SMOKE_VERSION,
        min_best_quality_auroc=min_best_quality_auroc,
        max_uncached_forward_seconds=20.0,
        max_cache_only_seconds=1.0,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
        max_verified_false_alarm=0.0,
        min_verified_detection=0.99,
        max_p99_duration_seconds=0.03,
        release_report_path=output_dir / f"release_candidate_registry_{report_prefix}_comparison.json",
        artifact_manifest_path=output_dir / f"release_candidate_registry_{report_prefix}_manifest.json",
        verification_report_path=output_dir / f"release_candidate_registry_{report_prefix}_verification.json",
        workflow_report_path=output_dir / f"release_candidate_registry_{report_prefix}_workflow.json",
    )


def _write_readiness_baseline_manifest(
    output_dir: Path,
    *,
    registry_path: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "cache-only-result.json"
    matrix_path = output_dir / "performance-matrix.json"
    manifest_path = output_dir / "artifact-manifest.json"
    cell_id = "layer_m1_batch_1_capture_outputs"
    quality_signals = {
        "truth_proj": 0.72,
        "subspace_resid": 0.76,
    }

    _write_json(result_path, {"auroc": quality_signals})
    _write_json(
        matrix_path,
        {
            "config": {
                "max_workers": 1,
                "length_bucketed_batches": True,
            },
            "matrix_decision": {
                "status": "promote",
                "recommended_cell": cell_id,
                "recommendation_metric": "uncached_forced_answer_forward_seconds",
                "blocking_reasons": (),
                "recommended": {
                    "id": cell_id,
                    "layer": -1,
                    "batch_size": 1,
                    "hidden_state_capture": "outputs",
                    "max_batch_tokens": 0,
                    "prefix_kv_cache": False,
                    "uncached_total_seconds": 10.0,
                    "cache_only_total_seconds": 0.2,
                    "uncached_forced_answer_forward_seconds": 10.0,
                    "truth_proj_auroc": quality_signals["truth_proj"],
                },
            },
            "cells": [
                {
                    "id": cell_id,
                    "layer": -1,
                    "batch_size": 1,
                    "hidden_state_capture": "outputs",
                    "summary": {
                        "quality_signals": quality_signals,
                        "truth_proj_auroc": quality_signals["truth_proj"],
                        "totals": {
                            "uncached": {"total_seconds": 10.0},
                            "cache_only": {"total_seconds": 0.2},
                        },
                    },
                    "triplet": {"results": {"cache_only": str(result_path)}},
                }
            ],
        },
    )
    metadata = {
        "runner": "run_adapter_readiness_workflow",
        "model": "synthetic-smoke",
        "dtype": "auto",
        "readiness_status": "promote",
        "adapter_family_status": "promote",
        "performance_status": "promote",
        "runtime_recommendation_status": "promote",
        "recommended_route": "structured_qa",
        "recommended_performance_cell": cell_id,
        "inside_sampling_report": None,
        "inside_trigger_budget_sweep_report": None,
    }
    _write_json(
        manifest_path,
        build_artifact_manifest(
            {"performance_matrix_report": matrix_path},
            root=output_dir,
            metadata=metadata,
        ),
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name=f"{SMOKE_NAME}-readiness",
        version=SMOKE_VERSION,
        path=manifest_path,
        metadata={
            "workflow": "run_adapter_readiness_registry_workflow",
            "readiness_status": "promote",
            "runtime_recommendation_status": "promote",
            "manifest_metadata": metadata,
        },
    ).save_json()
    return manifest_path


def _write_route_baseline_manifest(
    output_dir: Path,
    *,
    registry_path: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    route_report_path = output_dir / "route-comparison.json"
    manifest_path = output_dir / "artifact-manifest.json"
    route = "structured_qa"
    _write_json(
        route_report_path,
        {
            "schema_version": 1,
            "promotion_decision": {
                "status": "promote",
                "recommended_route": route,
            },
            "by_route": {
                route: {
                    "selected": 8,
                    "decision_accuracy": 1.0,
                    "false_supported_rate": 0.0,
                    "false_refuted_rate": 1.0,
                    "verified_false_alarm": 0.0,
                    "verified_detection": 1.0,
                    "mean_duration_seconds": 0.01,
                    "p95_duration_seconds": 0.02,
                    "p99_duration_seconds": 0.02,
                    "max_duration_seconds": 0.02,
                    "mean_attempted_route_count": 1.0,
                    "retrieval_use_rate": 0.0,
                    "invalid_metric_counts": {},
                }
            },
        },
    )
    metadata = {
        "runner": "run_adapter_promotion_workflow",
        "workflow": "adapter_promotion_workflow",
        "promotion_status": "promote",
        "route_promotion_status": "promote",
        "recommended_route": route,
    }
    _write_json(
        manifest_path,
        build_artifact_manifest(
            {"route_comparison_report": route_report_path},
            root=output_dir,
            metadata=metadata,
        ),
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name=f"{SMOKE_NAME}-route",
        version=SMOKE_VERSION,
        path=manifest_path,
        metadata={
            "workflow": "run_adapter_promotion_workflow",
            "route_promotion_status": "promote",
            "manifest_metadata": metadata,
        },
    ).save_json()
    return manifest_path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the no-model release-candidate registry smoke check")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="optional output directory; defaults to a temporary directory",
    )
    args = parser.parse_args(argv)
    if args.output_dir is not None:
        report = build_release_candidate_registry_smoke(Path(args.output_dir))
        _print_report(report)
        return
    with tempfile.TemporaryDirectory(prefix="eigentruth-release-candidate-registry-smoke-") as tmpdir:
        report = build_release_candidate_registry_smoke(Path(tmpdir))
        _print_report(report)


def _print_report(report: Mapping[str, Any]) -> None:
    promoted = dict(report["promoted_report"]["decision"])
    blocked = dict(report["blocked_report"]["decision"])
    print(
        "release_candidate_registry_smoke_ok "
        f"promoted={promoted['registry_record']} "
        f"blocked_status={blocked['status']}"
    )


if __name__ == "__main__":
    main()
