"""Tests for release-evidence gap planning."""

import json

from benchmarks.plan_release_evidence_gaps import build_release_evidence_gap_plan
from eigentruth.control import (
    EvidenceGapPlan,
    plan_evidence_gaps_from_release_candidate,
)
from eigentruth.json_utils import strict_json_dumps
from eigentruth.registry import ArtifactRegistry


def test_evidence_gap_plan_maps_release_blockers_to_frontier_actions():
    plan = plan_evidence_gaps_from_release_candidate(
        _blocked_registry_workflow_payload(),
        source_path="artifacts/frontier-audit-release-candidate-v4/frontier-audit-comparison.json",
    )

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}
    gaps = {gap["gap_id"]: gap for gap in payload["gaps"]}

    assert payload["status"] == "needs_evidence"
    assert payload["source_workflow"] == "release_candidate_registry_workflow"
    assert payload["source_status"] == "blocked"
    assert payload["summary"]["gap_count"] == 4
    assert payload["summary"]["missing_metric_count"] == 14
    assert payload["summary"]["gates"] == {
        "performance_baseline": 1,
        "product_runtime_drift": 2,
        "readiness_baseline": 1,
    }
    assert payload["summary"]["top_action_ids"][0] == "refresh_readiness_baseline"
    assert "run_pre_generation_probe_comparison" in actions
    assert "rerun_product_trace_action_gates" in actions
    assert actions["run_pre_generation_probe_comparison"]["evidence_routes"] == (
        "pre_generation_probe_comparison",
        "product_runtime_drift",
    )
    pre_generation_gap = next(
        gap
        for gap in gaps.values()
        if gap["recommended_action_ids"] == ("run_pre_generation_probe_comparison",)
    )
    assert pre_generation_gap["root_cause"] == "model"
    assert pre_generation_gap["missing_metrics"] == (
        "promotion_contract.pre_generation_probe_comparison.coverage_rate",
        "promotion_contract.pre_generation_probe_comparison.manifest_verified_rate",
        "promotion_contract.pre_generation_probe_comparison.model_count.mean",
        "promotion_contract.pre_generation_probe_comparison.run_count.mean",
        "promotion_contract.pre_generation_probe_comparison.redline_pass_rate",
        "promotion_contract.pre_generation_probe_comparison.best_test_label_auroc.mean",
        "promotion_contract.pre_generation_probe_comparison.best_redline_auroc.mean",
        "promotion_contract.pre_generation_probe_comparison.best_redline_margin.mean",
    )
    strict_json_dumps(payload, sort_keys=True)
    assert EvidenceGapPlan.from_dict(payload).to_dict() == payload


def test_evidence_gap_plan_reports_ready_when_no_blockers():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {"status": "promote", "blocking_reasons": []},
    })

    assert plan.status == "ready"
    assert plan.summary["gap_count"] == 0
    assert plan.actions == ()


def test_evidence_gap_plan_maps_multiple_testing_frontier_blocker():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {
            "status": "blocked",
            "blocking_reasons": [
                {
                    "gate": "frontier_release_evidence",
                    "status": "blocked",
                    "reasons": (
                        "frontier release evidence multiple_testing_track_status is "
                        "'blocked', expected 'promote' or 'not_required'",
                    ),
                }
            ],
        },
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}
    gaps = payload["gaps"]

    assert payload["status"] == "needs_evidence"
    assert payload["summary"]["gap_count"] == 1
    assert payload["summary"]["research_axes"] == {"multi_signal_calibration": 1}
    assert payload["summary"]["top_action_ids"] == ("rerun_frontier_multiple_testing_gate",)
    assert gaps[0]["root_cause"] == "model"
    assert gaps[0]["metadata"]["evidence_kind"] == "frontier_multiple_testing"
    assert gaps[0]["recommended_action_ids"] == ("rerun_frontier_multiple_testing_gate",)
    assert actions["rerun_frontier_multiple_testing_gate"]["evidence_routes"] == (
        "truthfulqa_frontier_workflow",
        "multiple_testing_gate",
        "frontier_release_evidence",
    )


def test_evidence_gap_plan_maps_frontier_release_evidence_report_tracks():
    plan = plan_evidence_gaps_from_release_candidate({
        "schema_version": 1,
        "workflow": "frontier_release_evidence_comparison",
        "status": "complete",
        "decision": {
            "status": "blocked",
            "verifier_track_status": "promote",
            "abstention_track_status": "blocked",
            "detectability_track_status": "blocked",
            "multiple_testing_track_status": "blocked",
            "citation_batch_track_status": "blocked",
            "blocking_reasons": (
                "abstention_stability.synthetic.conditional_correctness_lower_bound_mean "
                "0.5 is below required minimum 0.8",
                "detectability_taxonomy.synthetic.entrenched_false_rate 0.4 exceeds maximum 0.25",
                "truthfulqa_frontier_workflow.synthetic.multiple_testing_gate.all_pass is not true",
                "citation_batch_rollup.citation-rollup.summary.missing_expected_batch_count "
                "1 is non-zero",
            ),
        },
        "evidence_summary": {
            "citation_batch_rollup_names": ("citation-rollup",),
            "citation_batch_missing_expected_batch_count": 1,
            "citation_batch_missing_expected_batches": (
                {
                    "rollup": "citation-rollup",
                    "batch_id": "unresolved-evidence-batch-0002",
                },
            ),
            "citation_batch_expected_batch_ids": (
                "unresolved-evidence-batch-0001",
                "unresolved-evidence-batch-0002",
            ),
            "citation_batch_observed_batch_ids": ("unresolved-evidence-batch-0001",),
        },
        "multiple_testing_decisions": (
            {
                "name": "synthetic",
                "metrics": {
                    "failed_cells": (
                        {
                            "cell": "synthetic-l2",
                            "status": "failed",
                            "false_alarm": 0.04,
                            "detection": 0.62,
                            "report": "synthetic-l2/multiple-testing-report.json",
                            "calibration": "synthetic-l2/multiple-testing-calibration.json",
                        },
                    ),
                    "unknown_cells": (),
                },
            },
        ),
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}
    gaps = {gap["gate"]: gap for gap in payload["gaps"]}

    assert payload["status"] == "needs_evidence"
    assert payload["source_workflow"] == "frontier_release_evidence_comparison"
    assert payload["summary"]["gates"] == {
        "abstention_stability": 1,
        "citation_batch_evidence": 1,
        "detectability_taxonomy": 1,
        "frontier_multiple_testing": 1,
    }
    assert payload["summary"]["research_axes"] == {
        "blind_spot_taxonomy": 1,
        "external_citation": 1,
        "multi_signal_calibration": 1,
        "participation_calibration": 1,
    }
    assert gaps["abstention_stability"]["recommended_action_ids"] == (
        "improve_abstention_participation_gate",
    )
    assert gaps["detectability_taxonomy"]["recommended_action_ids"] == (
        "audit_detectability_blind_spots",
    )
    assert gaps["frontier_multiple_testing"]["recommended_action_ids"] == (
        "rerun_frontier_multiple_testing_gate",
    )
    assert gaps["citation_batch_evidence"]["recommended_action_ids"] == (
        "complete_citation_batch_evidence_rollup",
    )
    assert gaps["citation_batch_evidence"]["metadata"]["citation_batch_missing_expected_batches"] == (
        {
            "rollup": "citation-rollup",
            "batch_id": "unresolved-evidence-batch-0002",
        },
    )
    assert gaps["frontier_multiple_testing"]["metadata"]["multiple_testing_failed_cells"] == (
        {
            "run": "synthetic",
            "cell": "synthetic-l2",
            "status": "failed",
            "false_alarm": 0.04,
            "detection": 0.62,
            "report": "synthetic-l2/multiple-testing-report.json",
            "calibration": "synthetic-l2/multiple-testing-calibration.json",
        },
    )
    assert actions["improve_abstention_participation_gate"]["evidence_routes"] == (
        "abstention_stability",
        "participation_gate",
        "frontier_release_evidence",
    )
    assert actions["complete_citation_batch_evidence_rollup"]["evidence_routes"] == (
        "unresolved_evidence_queue",
        "citation_search_evidence",
        "source_family_citation",
        "frontier_release_evidence",
    )


def test_evidence_gap_plan_maps_product_runtime_world_model_blockers():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {
            "status": "blocked",
            "blocking_reasons": [
                {
                    "gate": "product_runtime_drift",
                    "status": "blocked",
                    "reasons": (
                        "product runtime drift world-model evidence metrics are incomplete: "
                        "world_model.participating_trace_rate, world_model.trace_gap_rate",
                        "product runtime drift world-model evidence blocked 1 metric(s)",
                    ),
                }
            ],
        },
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}

    assert payload["status"] == "needs_evidence"
    assert payload["summary"]["gap_count"] == 2
    assert payload["summary"]["action_count"] == 1
    assert payload["summary"]["missing_metric_count"] == 2
    assert payload["summary"]["root_causes"] == {"world_model": 2}
    assert payload["summary"]["research_axes"] == {"runtime_drift": 2}
    assert payload["summary"]["top_action_ids"] == (
        "rerun_product_trace_world_model_evidence",
    )
    assert actions["rerun_product_trace_world_model_evidence"]["evidence_routes"] == (
        "product_trace_replay",
        "product_runtime_drift",
        "world_model_evidence",
    )
    for gap in payload["gaps"]:
        assert gap["metadata"]["evidence_kind"] == "product_runtime_world_model_evidence"
        assert gap["recommended_action_ids"] == (
            "rerun_product_trace_world_model_evidence",
        )
    assert payload["gaps"][0]["missing_metrics"] == (
        "world_model.participating_trace_rate",
        "world_model.trace_gap_rate",
    )
    assert payload["gaps"][1]["missing_metrics"] == ()


def test_plan_release_evidence_gaps_cli_helper_writes_and_registers(tmp_path):
    source = tmp_path / "release-workflow.json"
    output = tmp_path / "evidence-gap-plan.json"
    registry_path = tmp_path / "registry.json"
    source.write_text(json.dumps(_blocked_registry_workflow_payload()), encoding="utf-8")

    payload = build_release_evidence_gap_plan(
        source=source,
        json_path=output,
        registry_path=registry_path,
        name="frontier-audit-gap-plan",
        version="0.1",
        metadata={"scope": "unit-test"},
    )

    assert output.exists()
    assert payload["metadata"] == {"scope": "unit-test"}
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["summary"]["action_count"] == payload["summary"]["action_count"]
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get("evidence_gap_plan:frontier-audit-gap-plan:0.1")
    assert record.path == str(output)
    assert record.metadata["status"] == "needs_evidence"
    assert record.metadata["gap_count"] == 4


def _blocked_registry_workflow_payload():
    return {
        "workflow": "release_candidate_registry_workflow",
        "release_candidate_comparison": {
            "workflow": "release_candidate_comparison",
            "decision": {
                "status": "blocked",
                "blocking_reasons": [
                    {
                        "gate": "readiness_baseline",
                        "status": "blocked",
                        "reasons": (
                            "benchmark_manifest:smollm2:0.8: best quality AUROC below 0.7",
                        ),
                    },
                    {
                        "gate": "performance_baseline",
                        "status": "blocked",
                        "reasons": (
                            "release candidate is unavailable for performance baseline comparison",
                        ),
                    },
                    {
                        "gate": "product_runtime_drift",
                        "status": "blocked",
                        "reasons": (
                            "product runtime drift pre-generation evidence metrics are incomplete: "
                            "promotion_contract.pre_generation_probe_comparison.coverage_rate, "
                            "promotion_contract.pre_generation_probe_comparison.manifest_verified_rate, "
                            "promotion_contract.pre_generation_probe_comparison.model_count.mean, "
                            "promotion_contract.pre_generation_probe_comparison.run_count.mean, "
                            "promotion_contract.pre_generation_probe_comparison.redline_pass_rate, "
                            "promotion_contract.pre_generation_probe_comparison.best_test_label_auroc.mean, "
                            "promotion_contract.pre_generation_probe_comparison.best_redline_auroc.mean, "
                            "promotion_contract.pre_generation_probe_comparison.best_redline_margin.mean",
                            "product runtime drift action-gate evidence metrics are incomplete: "
                            "promotion_contract.product_trace_replay.action_audit_gate.error_rate.mean, "
                            "promotion_contract.product_trace_replay.action_audit_gate."
                            "missing_retrieval_action_rate.mean, "
                            "promotion_contract.product_trace_replay.action_execution_gate."
                            "alignment_failed_trace_rate.mean, "
                            "promotion_contract.product_trace_replay.action_execution_gate.missing_result_rate.mean, "
                            "promotion_contract.product_trace_replay.action_execution_gate."
                            "unexpected_result_rate.mean, "
                            "promotion_contract.product_trace_replay.action_execution_gate.request_id_mismatch_rate.mean",
                        ),
                    },
                ],
            },
        },
    }
