"""No-model smoke checks for the performance baseline workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_performance_baseline_workflow import (  # noqa: E402
    PerformanceBaselineWorkflowConfig,
    run_performance_baseline_workflow,
)


def build_performance_baseline_smoke(output_dir: Path) -> dict[str, Any]:
    """Build a synthetic promoted performance baseline bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    result_path = source_dir / "cache-only-result.json"
    matrix_manifest_path = source_dir / "matrix-artifact-manifest.json"
    matrix_report_path = source_dir / "cache-profile-matrix-report.json"
    registry_path = output_dir / "registry.json"
    result_path.write_text(
        json.dumps({
            "auroc": {
                "truth_proj": 0.84,
                "subspace_resid": 0.91,
            },
        }),
        encoding="utf-8",
    )
    matrix_manifest_path.write_text("{}\n", encoding="utf-8")
    matrix_report_path.write_text(
        json.dumps({
            "artifact_manifest": str(matrix_manifest_path),
            "report_path": str(matrix_report_path),
            "config": {"max_workers": 1, "length_bucketed_batches": True},
            "matrix_decision": {
                "status": "promote",
                "recommended_cell": "layer_m1_batch_1_capture_outputs",
                "recommendation_metric": "cache_only_total_seconds",
                "blocking_reasons": (),
                "recommended": {
                    "id": "layer_m1_batch_1_capture_outputs",
                    "layer": -1,
                    "batch_size": 1,
                    "hidden_state_capture": "outputs",
                    "max_batch_tokens": 0,
                    "prefix_kv_cache": False,
                    "cache_only_total_seconds": 0.05,
                    "truth_proj_auroc": 0.84,
                },
            },
            "cells": [
                {
                    "id": "layer_m1_batch_1_capture_outputs",
                    "layer": -1,
                    "batch_size": 1,
                    "hidden_state_capture": "outputs",
                    "triplet": {"results": {"cache_only": str(result_path)}},
                }
            ],
        }),
        encoding="utf-8",
    )
    return run_performance_baseline_workflow(
        PerformanceBaselineWorkflowConfig(
            output_dir=output_dir / "workflow",
            registry_path=registry_path,
            name="performance-baseline-smoke",
            version="0.1",
            matrix_report_path=matrix_report_path,
        )
    )


def main() -> None:
    output_dir = REPO_ROOT / "artifacts" / "performance_baseline_smoke"
    report = build_performance_baseline_smoke(output_dir)
    print(
        "performance_baseline_smoke_ok "
        f"status={report['status']} "
        f"record={report.get('registry_record')}"
    )


if __name__ == "__main__":
    main()
