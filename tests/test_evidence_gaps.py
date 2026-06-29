"""Tests for release-evidence gap planning."""

import json

from benchmarks.plan_citation_batch_evidence_reruns import build_citation_batch_evidence_rerun_queue
from benchmarks.plan_frontier_stability_evidence_reruns import build_frontier_stability_evidence_rerun_queue
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


def test_plan_release_evidence_gaps_can_emit_multiple_testing_rerun_queue(tmp_path):
    source = tmp_path / "frontier-release-evidence.json"
    output = tmp_path / "evidence-gap-plan.json"
    queue_path = tmp_path / "multiple-testing-rerun-queue.json"
    manifest_path = tmp_path / "multiple-testing-rerun-queue-manifest.json"
    registry_path = tmp_path / "registry.json"
    workflow_path = tmp_path / "frontier" / "truthfulqa-frontier-workflow.json"
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(
        json.dumps(_frontier_workflow_payload_for_multiple_testing_queue()),
        encoding="utf-8",
    )
    source.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "frontier_release_evidence_comparison",
            "status": "complete",
            "inputs": {
                "frontier_workflow_reports": (
                    {
                        "path": str(workflow_path),
                        "workflow": "truthfulqa_frontier_workflow",
                        "status": "complete",
                    },
                ),
            },
            "decision": {
                "status": "blocked",
                "multiple_testing_track_status": "blocked",
                "blocking_reasons": (
                    "truthfulqa_frontier_workflow.synthetic.multiple_testing_gate.all_pass is not true",
                ),
            },
            "evidence_summary": {
                "multiple_testing_failed_cells": (
                    {
                        "run": "truthfulqa-frontier-workflow",
                        "cell": "a-l2",
                        "status": "failed",
                        "false_alarm": 0.03,
                        "detection": 0.7,
                        "report": "frontier/a-l2/multiple-testing-report.json",
                        "calibration": "frontier/a-l2/multiple-testing-calibration.json",
                    },
                ),
            },
        }),
        encoding="utf-8",
    )

    payload = build_release_evidence_gap_plan(
        source=source,
        json_path=output,
        registry_path=registry_path,
        name="frontier-gap-plan",
        version="0.1",
        multiple_testing_rerun_json_path=queue_path,
        multiple_testing_rerun_artifact_manifest_path=manifest_path,
        multiple_testing_rerun_output_dir=tmp_path / "reruns",
        multiple_testing_rerun_name="frontier-multiple-testing-reruns",
        multiple_testing_rerun_version="0.1",
        python_executable="python",
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    gap_record = registry.get("evidence_gap_plan:frontier-gap-plan:0.1")
    queue_record = registry.get("report:frontier-multiple-testing-reruns:0.1")

    derived = payload["derived_artifacts"]["frontier_multiple_testing_rerun_queue"]
    assert saved["derived_artifacts"] == payload["derived_artifacts"]
    assert derived["path"] == str(queue_path)
    assert derived["artifact_manifest"] == str(manifest_path)
    assert derived["status"] == "ready"
    assert derived["blocked_cell_count"] == 1
    assert derived["command_count"] == 1
    assert queue["entries"][0]["command_status"] == "ready"
    assert queue["entries"][0]["command"][:3] == [
        "python",
        "benchmarks/run_truthfulqa_frontier_workflow.py",
        "--output-dir",
    ]
    assert queue["entries"][0]["dry_run_command"][-1] == "--dry-run"
    assert manifest["artifacts"]["frontier_multiple_testing_rerun_queue"]["exists"] is True
    assert gap_record.metadata["gap_count"] == 1
    assert queue_record.metadata["blocked_cell_count"] == 1
    assert queue_record.metadata["command_count"] == 1


def test_citation_batch_evidence_rerun_queue_builds_source_family_commands(tmp_path):
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "citation-batch-rerun-queue.json"
    manifest_path = tmp_path / "citation-batch-rerun-queue-manifest.json"
    registry_path = tmp_path / "registry.json"
    source.write_text(json.dumps(_frontier_release_citation_batch_payload()), encoding="utf-8")

    payload = build_citation_batch_evidence_rerun_queue(
        source=source,
        json_path=queue_path,
        artifact_manifest_path=manifest_path,
        registry_path=registry_path,
        name="citation-batch-reruns",
        version="0.1",
        output_dir=tmp_path / "reruns",
        queue_report_path=tmp_path / "unresolved-queue.json",
        scores_path=tmp_path / "scores.jsonl",
        blind_spots_path=tmp_path / "blind-spots.jsonl",
        source_catalog_paths=(tmp_path / "catalog.jsonl",),
        controlled_sweep_paths=(tmp_path / "controlled-sweep.json",),
        python_executable="python",
    )

    saved = json.loads(queue_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get("report:citation-batch-reruns:0.1")
    entries = {entry["batch_id"]: entry for entry in payload["entries"]}
    command = entries["unresolved-evidence-batch-0002"]["command"]

    assert saved["summary"] == payload["summary"]
    assert payload["workflow"] == "citation_batch_evidence_rerun_queue"
    assert payload["summary"]["blocked_batch_count"] == 2
    assert payload["summary"]["missing_expected_batch_count"] == 1
    assert payload["summary"]["duplicate_batch_count"] == 1
    assert payload["summary"]["command_count"] == 2
    assert entries["unresolved-evidence-batch-0002"]["issue_type"] == "missing_expected"
    assert entries["unresolved-evidence-batch-0002"]["command_status"] == "ready"
    assert entries["unresolved-evidence-batch-0002"]["command_kind"] == "source_family"
    assert command[:3] == (
        "python",
        "benchmarks/run_source_family_citation_search_workflow.py",
        "--queue",
    )
    assert command[command.index("--batch-id") + 1] == "unresolved-evidence-batch-0002"
    assert command[command.index("--source-catalog") + 1] == str(tmp_path / "catalog.jsonl")
    assert command[command.index("--controlled-sweep") + 1] == str(tmp_path / "controlled-sweep.json")
    assert manifest["artifacts"]["citation_batch_evidence_rerun_queue"]["exists"] is True
    assert record.metadata["workflow"] == "citation_batch_evidence_rerun_queue"
    assert record.metadata["blocked_batch_count"] == 2


def test_plan_release_evidence_gaps_can_emit_citation_batch_rerun_queue(tmp_path):
    source = tmp_path / "frontier-release-evidence.json"
    output = tmp_path / "evidence-gap-plan.json"
    queue_path = tmp_path / "citation-batch-rerun-queue.json"
    manifest_path = tmp_path / "citation-batch-rerun-queue-manifest.json"
    registry_path = tmp_path / "registry.json"
    source.write_text(json.dumps(_frontier_release_citation_batch_payload()), encoding="utf-8")

    payload = build_release_evidence_gap_plan(
        source=source,
        json_path=output,
        registry_path=registry_path,
        name="frontier-gap-plan",
        version="0.1",
        citation_batch_rerun_json_path=queue_path,
        citation_batch_rerun_artifact_manifest_path=manifest_path,
        citation_batch_rerun_output_dir=tmp_path / "citation-reruns",
        citation_batch_rerun_name="citation-batch-reruns",
        citation_batch_rerun_version="0.1",
        citation_batch_queue_report_path=tmp_path / "unresolved-queue.json",
        citation_batch_scores_path=tmp_path / "scores.jsonl",
        citation_batch_blind_spots_path=tmp_path / "blind-spots.jsonl",
        citation_batch_search_command="python adapter.py --input {input} --output {output}",
        python_executable="python",
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    gap_record = registry.get("evidence_gap_plan:frontier-gap-plan:0.1")
    queue_record = registry.get("report:citation-batch-reruns:0.1")
    derived = payload["derived_artifacts"]["citation_batch_evidence_rerun_queue"]
    entry = queue["entries"][0]

    assert saved["derived_artifacts"] == payload["derived_artifacts"]
    assert derived["path"] == str(queue_path)
    assert derived["artifact_manifest"] == str(manifest_path)
    assert derived["status"] == "ready"
    assert derived["blocked_batch_count"] == 2
    assert derived["command_count"] == 2
    assert entry["command_status"] == "ready"
    assert entry["command_kind"] == "external"
    assert entry["command"][1] == "benchmarks/run_external_citation_search_adapter_workflow.py"
    assert entry["command"][entry["command"].index("--search-command") + 1] == (
        "python adapter.py --input {input} --output {output}"
    )
    assert gap_record.metadata["gap_count"] == 1
    assert queue_record.metadata["command_count"] == 2


def test_frontier_stability_evidence_rerun_queue_builds_commands(tmp_path):
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "stability-rerun-queue.json"
    manifest_path = tmp_path / "stability-rerun-queue-manifest.json"
    registry_path = tmp_path / "registry.json"
    source.write_text(json.dumps(_frontier_release_stability_payload()), encoding="utf-8")

    payload = build_frontier_stability_evidence_rerun_queue(
        source=source,
        json_path=queue_path,
        artifact_manifest_path=manifest_path,
        registry_path=registry_path,
        name="frontier-stability-reruns",
        version="0.1",
        output_dir=tmp_path / "stability-reruns",
        score_paths=(
            f"qwen={tmp_path / 'qwen-scores.manifest.json'}",
            f"smol={tmp_path / 'smol-scores.manifest.json'}",
        ),
        seeds="0,1",
        verifier_qa_corpus_path=tmp_path / "qa-corpus.json",
        python_executable="python",
    )

    saved = json.loads(queue_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get("report:frontier-stability-reruns:0.1")
    entries = {entry["track"]: entry for entry in payload["entries"]}
    verifier_command = entries["verifier_stability"]["command"]
    abstention_command = entries["abstention_stability"]["command"]

    assert saved["summary"] == payload["summary"]
    assert payload["workflow"] == "frontier_stability_evidence_rerun_queue"
    assert payload["summary"]["blocked_track_count"] == 2
    assert payload["summary"]["command_count"] == 2
    assert entries["verifier_stability"]["command_status"] == "ready"
    assert entries["abstention_stability"]["command_status"] == "ready"
    assert verifier_command[:2] == ("python", "benchmarks/eval_verifier_stability.py")
    assert verifier_command[verifier_command.index("--signal") + 1] == "truth_proj"
    assert verifier_command[verifier_command.index("--qa-corpus") + 1] == str(tmp_path / "qa-corpus.json")
    assert "--staged-verification" in verifier_command
    assert abstention_command[:2] == ("python", "benchmarks/eval_abstention_stability.py")
    assert abstention_command[abstention_command.index("--signals") + 1] == "maha_last,subspace_resid"
    assert abstention_command[abstention_command.index("--seeds") + 1] == "0,1"
    assert manifest["artifacts"]["frontier_stability_evidence_rerun_queue"]["exists"] is True
    assert record.metadata["workflow"] == "frontier_stability_evidence_rerun_queue"
    assert record.metadata["blocked_track_count"] == 2


def test_plan_release_evidence_gaps_can_emit_stability_rerun_queue(tmp_path):
    source = tmp_path / "frontier-release-evidence.json"
    output = tmp_path / "evidence-gap-plan.json"
    queue_path = tmp_path / "stability-rerun-queue.json"
    manifest_path = tmp_path / "stability-rerun-queue-manifest.json"
    registry_path = tmp_path / "registry.json"
    source.write_text(json.dumps(_frontier_release_stability_payload()), encoding="utf-8")

    payload = build_release_evidence_gap_plan(
        source=source,
        json_path=output,
        registry_path=registry_path,
        name="frontier-gap-plan",
        version="0.1",
        stability_rerun_json_path=queue_path,
        stability_rerun_artifact_manifest_path=manifest_path,
        stability_rerun_output_dir=tmp_path / "stability-reruns",
        stability_rerun_name="frontier-stability-reruns",
        stability_rerun_version="0.1",
        stability_score_paths=(f"qwen={tmp_path / 'qwen-scores.manifest.json'}",),
        stability_seeds="0,1",
        python_executable="python",
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    gap_record = registry.get("evidence_gap_plan:frontier-gap-plan:0.1")
    queue_record = registry.get("report:frontier-stability-reruns:0.1")
    derived = payload["derived_artifacts"]["frontier_stability_evidence_rerun_queue"]

    assert saved["derived_artifacts"] == payload["derived_artifacts"]
    assert derived["path"] == str(queue_path)
    assert derived["artifact_manifest"] == str(manifest_path)
    assert derived["status"] == "ready"
    assert derived["blocked_track_count"] == 2
    assert derived["command_count"] == 2
    assert queue["entries"][0]["command_status"] == "ready"
    assert gap_record.metadata["gap_count"] == 2
    assert queue_record.metadata["command_count"] == 2


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


def _frontier_workflow_payload_for_multiple_testing_queue():
    return {
        "schema_version": 1,
        "workflow": "truthfulqa_frontier_workflow",
        "status": "complete",
        "config": {
            "models": ({"name": "a", "model_id": "synthetic-a"},),
            "scales": (
                {
                    "name": "l2",
                    "limit": 2,
                    "manifold_questions": 2,
                    "layer": -1,
                    "sweep_layers": (-1, -2),
                },
            ),
            "dtype": "float32",
            "batch_size": 2,
            "max_batch_tokens": 0,
            "max_length": 64,
            "hidden_state_capture": "hooks",
            "covariance_mode": "diag",
            "covariance_low_rank": 4,
            "progress_every": 0,
            "offline": True,
            "signals": ("truth_proj", "subspace_resid"),
            "conformal_signal": "truth_proj",
            "conformal_repeats": 1,
            "ensemble_repeats": 1,
            "artifact_alpha": 0.2,
            "multiple_testing_signals": ("truth_proj", "subspace_resid"),
            "multiple_testing_alpha": 0.2,
            "multiple_testing_method": "bh",
            "best_alpha": 0.2,
            "best_by": "auroc",
            "ensemble_methods": ("max_rank",),
            "alphas": (0.2,),
        },
        "multiple_testing_gate": {
            "enabled": True,
            "all_pass": False,
            "cells": (
                {
                    "cell": "a-l2",
                    "pass": False,
                    "false_alarm": 0.03,
                    "detection": 0.7,
                    "report": "frontier/a-l2/multiple-testing-report.json",
                    "calibration": "frontier/a-l2/multiple-testing-calibration.json",
                },
            ),
        },
    }


def _frontier_release_citation_batch_payload():
    return {
        "schema_version": 1,
        "workflow": "frontier_release_evidence_comparison",
        "status": "complete",
        "decision": {
            "status": "blocked",
            "citation_batch_track_status": "blocked",
            "blocking_reasons": (
                "citation_batch_rollup.citation-rollup.summary.missing_expected_batch_count 1 is non-zero",
            ),
        },
        "evidence_summary": {
            "citation_batch_rollup_names": ("citation-rollup",),
            "citation_batch_expected_batch_ids": (
                "unresolved-evidence-batch-0001",
                "unresolved-evidence-batch-0002",
            ),
            "citation_batch_observed_batch_ids": (
                "unresolved-evidence-batch-0001",
                "unresolved-evidence-batch-0001",
            ),
            "citation_batch_missing_expected_batches": (
                {
                    "rollup": "citation-rollup",
                    "batch_id": "unresolved-evidence-batch-0002",
                },
            ),
            "citation_batch_duplicate_batches": (
                {
                    "rollup": "citation-rollup",
                    "batch_id": "unresolved-evidence-batch-0001",
                },
            ),
        },
    }


def _frontier_release_stability_payload():
    return {
        "schema_version": 1,
        "workflow": "frontier_release_evidence_comparison",
        "status": "complete",
        "decision": {
            "status": "blocked",
            "verifier_track_status": "blocked",
            "abstention_track_status": "blocked",
            "blocking_reasons": (
                "verifier_stability.qwen.verified_detection_mean 0.1 is below required minimum 0.2",
                "abstention_stability.qwen.conditional_correctness_lower_bound_mean 0.5 is below "
                "required minimum 0.8",
            ),
        },
        "evidence_summary": {
            "run_names": ("qwen",),
            "verifier_signal": "truth_proj",
            "abstention_signals": ("maha_last", "subspace_resid"),
        },
    }
