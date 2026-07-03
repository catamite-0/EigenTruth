"""Summarize unresolved frontier evidence lanes into one read-only report.

This report is a coordination artifact. It aggregates unresolved blind-spot
queues, citation/source-family workflow outcomes, source-family coverage audits,
world-model rule-input plans, rule-candidate promotion gates, and optional
mechanism handoff bundles. It does not promote verifier evidence or close a
release gate by itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import (  # noqa: E402
    ArtifactRegistry,
    build_artifact_manifest,
    load_and_verify_artifact_manifest,
)

WORKFLOW = "unresolved_frontier_evidence_summary"
RULE_FAMILY_INPUT_FIELDS = {
    "causal_or_procedural": frozenset({
        "mechanism",
        "mechanism_status",
        "precondition",
        "source_citation",
    }),
    "entity_disambiguation": frozenset({
        "answer_entity",
        "expected_entity",
        "requested_role",
        "source_citation",
        "subject_entity",
    }),
    "quantity_or_arithmetic": frozenset({
        "calculation.expected",
        "calculation.expression",
        "numeric_value",
        "reference_time",
        "source_citation",
        "unit",
    }),
    "temporal_consistency": frozenset({
        "claim_time",
        "retrieved_at",
        "source_citation",
        "source_time",
    }),
}


def summarize_unresolved_frontier_evidence(
    *,
    unresolved_queue: Mapping[str, Any] | None = None,
    citation_workflows: Sequence[Mapping[str, Any]] = (),
    source_family_coverage_audits: Sequence[Mapping[str, Any]] = (),
    semantic_gap_review_workflows: Sequence[Mapping[str, Any]] = (),
    frontier_command_binding_reviews: Sequence[Mapping[str, Any]] = (),
    frontier_bound_command_runs: Sequence[Mapping[str, Any]] = (),
    frontier_queue_execution_smokes: Sequence[Mapping[str, Any]] = (),
    rule_input_plan: Mapping[str, Any] | None = None,
    rule_input_audit_report: Mapping[str, Any] | None = None,
    rule_stub_requeue_report: Mapping[str, Any] | None = None,
    requeued_rule_input_plan: Mapping[str, Any] | None = None,
    input_fill_result_rollup: Mapping[str, Any] | None = None,
    rule_promotion_reports: Sequence[Mapping[str, Any]] = (),
    mechanism_handoff_bundle: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready read-only summary of unresolved frontier evidence."""
    queue_lane = _queue_lane(unresolved_queue)
    source_lane = _source_family_coverage_lane(source_family_coverage_audits)
    citation_lane = _citation_lane(citation_workflows)
    semantic_lane = _semantic_gap_review_lane(semantic_gap_review_workflows)
    frontier_queue_lane = _frontier_queue_execution_lane(
        frontier_command_binding_reviews,
        frontier_bound_command_runs,
        frontier_queue_execution_smokes,
    )
    rule_lane = _world_model_rule_lane(
        rule_input_plan=rule_input_plan,
        rule_input_audit_report=rule_input_audit_report,
        rule_stub_requeue_report=rule_stub_requeue_report,
        requeued_rule_input_plan=requeued_rule_input_plan,
        input_fill_result_rollup=input_fill_result_rollup,
        rule_promotion_reports=rule_promotion_reports,
        mechanism_handoff_bundle=mechanism_handoff_bundle,
    )
    lanes = {
        "unresolved_queue": queue_lane,
        "source_family_acquisition": source_lane,
        "citation_evidence": citation_lane,
        "semantic_gap_review": semantic_lane,
        "frontier_queue_execution": frontier_queue_lane,
        "world_model_rules": rule_lane,
    }
    next_actions = _next_actions(lanes)
    summary = _summary(lanes)
    status = "promote" if not next_actions else "needs_evidence"
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Read-only closure summary for unresolved frontier blind-spot "
            "evidence lanes. Source discovery, rule-input rows, and adapter "
            "matches remain non-evidence until their downstream gates promote."
        ),
        "summary": summary,
        "lanes": lanes,
        "next_actions": tuple(next_actions),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    unresolved_queue_path: str | Path | None = None,
    citation_workflow_paths: Sequence[str | Path] = (),
    source_family_coverage_audit_paths: Sequence[str | Path] = (),
    semantic_gap_review_workflow_paths: Sequence[str | Path] = (),
    frontier_command_binding_review_paths: Sequence[str | Path] = (),
    frontier_bound_command_run_paths: Sequence[str | Path] = (),
    frontier_queue_execution_smoke_paths: Sequence[str | Path] = (),
    rule_input_plan_path: str | Path | None = None,
    rule_input_audit_report_path: str | Path | None = None,
    rule_stub_requeue_report_path: str | Path | None = None,
    requeued_rule_input_plan_path: str | Path | None = None,
    input_fill_result_rollup_path: str | Path | None = None,
    rule_promotion_report_paths: Sequence[str | Path] = (),
    mechanism_handoff_bundle_path: str | Path | None = None,
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register the summary report."""
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if registry_path is not None and json_path is None:
        raise ValueError("registry_path requires json_path.")

    unresolved_queue = _load_optional_mapping(unresolved_queue_path)
    citation_workflows = tuple(_load_mapping(path) for path in citation_workflow_paths)
    coverage_audits = tuple(_load_mapping(path) for path in source_family_coverage_audit_paths)
    semantic_gap_reviews = tuple(_load_mapping(path) for path in semantic_gap_review_workflow_paths)
    frontier_command_reviews = tuple(
        _load_mapping(path) for path in frontier_command_binding_review_paths
    )
    frontier_command_runs = tuple(_load_mapping(path) for path in frontier_bound_command_run_paths)
    frontier_queue_smokes = tuple(
        _load_frontier_queue_execution_smoke(path) for path in frontier_queue_execution_smoke_paths
    )
    frontier_queue_smoke_manifests = tuple(
        path
        for path in (
            _resolve_report_path(
                _nested(report, "paths", "artifact_manifest"),
                base_path=Path(source_path),
            )
            for source_path, report in zip(
                frontier_queue_execution_smoke_paths,
                frontier_queue_smokes,
                strict=False,
            )
        )
        if path is not None
    )
    rule_input_plan = _load_optional_mapping(rule_input_plan_path)
    rule_input_audit_report = _load_optional_mapping(rule_input_audit_report_path)
    rule_stub_requeue_report = _load_optional_mapping(rule_stub_requeue_report_path)
    requeued_rule_input_plan = _load_optional_mapping(requeued_rule_input_plan_path)
    input_fill_result_rollup = _load_optional_mapping(input_fill_result_rollup_path)
    rule_promotion_reports = tuple(_load_mapping(path) for path in rule_promotion_report_paths)
    mechanism_bundle = _load_optional_mapping(mechanism_handoff_bundle_path)

    payload = summarize_unresolved_frontier_evidence(
        unresolved_queue=unresolved_queue,
        citation_workflows=citation_workflows,
        source_family_coverage_audits=coverage_audits,
        semantic_gap_review_workflows=semantic_gap_reviews,
        frontier_command_binding_reviews=frontier_command_reviews,
        frontier_bound_command_runs=frontier_command_runs,
        frontier_queue_execution_smokes=frontier_queue_smokes,
        rule_input_plan=rule_input_plan,
        rule_input_audit_report=rule_input_audit_report,
        rule_stub_requeue_report=rule_stub_requeue_report,
        requeued_rule_input_plan=requeued_rule_input_plan,
        input_fill_result_rollup=input_fill_result_rollup,
        rule_promotion_reports=rule_promotion_reports,
        mechanism_handoff_bundle=mechanism_bundle,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["paths"] = {
        "summary_report": None if json_path is None else str(json_path),
        "artifact_manifest": None
        if artifact_manifest_path is None
        else str(artifact_manifest_path),
        "unresolved_queue": None if unresolved_queue_path is None else str(unresolved_queue_path),
        "citation_workflows": tuple(str(path) for path in citation_workflow_paths),
        "source_family_coverage_audits": tuple(
            str(path) for path in source_family_coverage_audit_paths
        ),
        "semantic_gap_review_workflows": tuple(
            str(path) for path in semantic_gap_review_workflow_paths
        ),
        "frontier_command_binding_reviews": tuple(
            str(path) for path in frontier_command_binding_review_paths
        ),
        "frontier_bound_command_runs": tuple(
            str(path) for path in frontier_bound_command_run_paths
        ),
        "frontier_queue_execution_smokes": tuple(
            str(path) for path in frontier_queue_execution_smoke_paths
        ),
        "frontier_queue_execution_smoke_manifests": tuple(
            str(path) for path in frontier_queue_smoke_manifests
        ),
        "rule_input_plan": None if rule_input_plan_path is None else str(rule_input_plan_path),
        "rule_input_audit_report": None
        if rule_input_audit_report_path is None
        else str(rule_input_audit_report_path),
        "rule_stub_requeue_report": None
        if rule_stub_requeue_report_path is None
        else str(rule_stub_requeue_report_path),
        "requeued_rule_input_plan": None
        if requeued_rule_input_plan_path is None
        else str(requeued_rule_input_plan_path),
        "input_fill_result_rollup": None
        if input_fill_result_rollup_path is None
        else str(input_fill_result_rollup_path),
        "rule_promotion_reports": tuple(str(path) for path in rule_promotion_report_paths),
        "mechanism_handoff_bundle": None
        if mechanism_handoff_bundle_path is None
        else str(mechanism_handoff_bundle_path),
    }

    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    if output_path is not None:
        _write_json(output_path, payload, compact=compact_json)
    manifest = None
    if manifest_path is not None:
        manifest = _write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            unresolved_queue_path=unresolved_queue_path,
            citation_workflow_paths=citation_workflow_paths,
            source_family_coverage_audit_paths=source_family_coverage_audit_paths,
            semantic_gap_review_workflow_paths=semantic_gap_review_workflow_paths,
            frontier_command_binding_review_paths=frontier_command_binding_review_paths,
            frontier_bound_command_run_paths=frontier_bound_command_run_paths,
            frontier_queue_execution_smoke_paths=frontier_queue_execution_smoke_paths,
            frontier_queue_execution_smoke_manifest_paths=frontier_queue_smoke_manifests,
            rule_input_plan_path=rule_input_plan_path,
            rule_input_audit_report_path=rule_input_audit_report_path,
            rule_stub_requeue_report_path=rule_stub_requeue_report_path,
            requeued_rule_input_plan_path=requeued_rule_input_plan_path,
            input_fill_result_rollup_path=input_fill_result_rollup_path,
            rule_promotion_report_paths=rule_promotion_report_paths,
            mechanism_handoff_bundle_path=mechanism_handoff_bundle_path,
            payload=payload,
            metadata=metadata or {},
            compact=compact_json,
        )
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output_path,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "unresolved_target_count": payload["summary"]["unresolved_target_count"],
                "citation_status": payload["lanes"]["citation_evidence"]["status"],
                "citation_query_sweep_failure_reason_counts": payload["lanes"][
                    "citation_evidence"
                ]["query_sweep_failure_reason_counts"],
                "citation_query_sweep_recommended_next_action_counts": payload["lanes"][
                    "citation_evidence"
                ]["query_sweep_recommended_next_action_counts"],
                "citation_query_sweep_no_hit_strategy_count": payload["lanes"][
                    "citation_evidence"
                ]["query_sweep_no_hit_strategy_count"],
                "citation_query_sweep_target_route_not_selected_strategy_count": payload[
                    "lanes"
                ]["citation_evidence"]["query_sweep_target_route_not_selected_strategy_count"],
                "citation_query_sweep_blind_refuted_rate_below_min_strategy_count": (
                    payload["lanes"]["citation_evidence"][
                        "query_sweep_blind_refuted_rate_below_min_strategy_count"
                    ]
                ),
                "citation_query_sweep_verified_false_alarm_above_max_strategy_count": (
                    payload["lanes"]["citation_evidence"][
                        "query_sweep_verified_false_alarm_above_max_strategy_count"
                    ]
                ),
                "source_family_acquisition_status": payload["lanes"][
                    "source_family_acquisition"
                ]["status"],
                "world_model_rule_status": payload["lanes"]["world_model_rules"]["status"],
                "semantic_gap_review_status": payload["lanes"]["semantic_gap_review"]["status"],
                "semantic_gap_review_workflow_count": payload["lanes"][
                    "semantic_gap_review"
                ]["workflow_count"],
                "semantic_gap_review_promoted_workflow_count": payload["lanes"][
                    "semantic_gap_review"
                ]["promoted_workflow_count"],
                "semantic_gap_review_approved_source_document_count": payload["lanes"][
                    "semantic_gap_review"
                ]["approved_source_document_count"],
                "frontier_queue_execution_status": payload["lanes"][
                    "frontier_queue_execution"
                ]["status"],
                "frontier_command_binding_review_count": payload["lanes"][
                    "frontier_queue_execution"
                ]["command_binding_review_count"],
                "frontier_bound_command_run_count": payload["lanes"][
                    "frontier_queue_execution"
                ]["bound_command_run_count"],
                "frontier_bound_command_succeeded_count": payload["lanes"][
                    "frontier_queue_execution"
                ]["succeeded_count"],
                "frontier_queue_execution_smoke_status": payload["lanes"][
                    "frontier_queue_execution"
                ]["control_plane_smoke_status"],
                "frontier_queue_execution_smoke_count": payload["lanes"][
                    "frontier_queue_execution"
                ]["frontier_queue_execution_smoke_count"],
                "frontier_queue_execution_smoke_manifest_verified_count": payload[
                    "lanes"
                ]["frontier_queue_execution"][
                    "frontier_queue_execution_smoke_manifest_verified_count"
                ],
                "world_model_rule_remaining_task_count": payload["lanes"][
                    "world_model_rules"
                ]["remaining_task_count"],
                "world_model_rule_audit_adjusted_remaining_task_count": payload["lanes"][
                    "world_model_rules"
                ]["audit_adjusted_remaining_task_count"],
                "world_model_rule_audit_requeue_suggestion_count": payload["lanes"][
                    "world_model_rules"
                ]["rule_input_audit_requeue_suggestion_count"],
                "world_model_rule_requeue_outstanding_count": payload["lanes"][
                    "world_model_rules"
                ]["rule_input_audit_requeue_outstanding_count"],
                "world_model_rule_input_fill_rollup_status": payload["lanes"][
                    "world_model_rules"
                ]["input_fill_result_rollup_status"],
                "world_model_rule_input_fill_adapter_ready": payload["lanes"][
                    "world_model_rules"
                ]["input_fill_adapter_ready"],
                "world_model_rule_combined_rule_input_count": payload["lanes"][
                    "world_model_rules"
                ]["combined_rule_input_count"],
                "world_model_rule_combined_unfilled_task_count": payload["lanes"][
                    "world_model_rules"
                ]["combined_unfilled_task_count"],
                "next_action_count": len(payload["next_actions"]),
                "manifest_summary": {} if manifest is None else manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _queue_lane(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "status": "missing",
            "workflow": None,
            "target_count": 0,
            "adapter_request_count": 0,
            "request_type_counts": {},
            "adapter_family_counts": {},
            "batch_count": 0,
            "evidence_status_counts": {},
            "label_usage": {"requests_are_verifier_evidence": False},
        }
    summary = _mapping(payload.get("summary"))
    return {
        "status": str(payload.get("status") or "unknown"),
        "workflow": payload.get("workflow"),
        "target_count": _int(summary.get("target_count")),
        "adapter_request_count": _int(summary.get("adapter_request_count")),
        "request_type_counts": _int_mapping(summary.get("request_type_counts")),
        "adapter_family_counts": _int_mapping(summary.get("adapter_family_counts")),
        "batch_count": _int(summary.get("batch_count")),
        "evidence_status_counts": _int_mapping(summary.get("evidence_status_counts")),
        "label_usage": dict(_mapping(payload.get("label_usage"))),
    }


def _source_family_coverage_lane(audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    covered_count = 0
    best_missing: int | None = None
    covered_families: Counter[str] = Counter()
    missing_families: Counter[str] = Counter()
    for index, audit in enumerate(audits, start=1):
        summary = _mapping(audit.get("summary"))
        status = str(audit.get("status") or "unknown")
        missing_count = _int(summary.get("request_missing_target_family_count"))
        if status == "covered":
            covered_count += 1
        best_missing = missing_count if best_missing is None else min(best_missing, missing_count)
        covered_families.update(_int_mapping(summary.get("covered_target_source_family_counts")))
        missing_families.update(_int_mapping(summary.get("missing_target_source_family_counts")))
        rows.append({
            "index": index,
            "workflow": audit.get("workflow"),
            "status": status,
            "request_count": _int(summary.get("request_count")),
            "request_with_results_count": _int(summary.get("request_with_results_count")),
            "request_with_target_family_count": _int(
                summary.get("request_with_target_family_count")
            ),
            "request_missing_target_family_count": missing_count,
            "acquisition_plan_count": _int(summary.get("acquisition_plan_count")),
            "covered_target_source_family_counts": _int_mapping(
                summary.get("covered_target_source_family_counts")
            ),
            "missing_target_source_family_counts": _int_mapping(
                summary.get("missing_target_source_family_counts")
            ),
        })
    if not rows:
        status = "missing"
    elif covered_count:
        status = "covered"
    else:
        status = "needs_catalog_expansion"
    return {
        "status": status,
        "audit_count": len(rows),
        "covered_audit_count": covered_count,
        "best_missing_target_family_count": best_missing,
        "covered_target_source_family_counts": dict(sorted(covered_families.items())),
        "missing_target_source_family_counts": dict(sorted(missing_families.items())),
        "audits": tuple(rows),
    }


def _citation_lane(workflows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    status_counts: Counter[str] = Counter()
    provenance_pass_count = 0
    provenance_failed_count = 0
    provenance_status_counts: Counter[str] = Counter()
    comparison_pass_count = 0
    comparison_failed_count = 0
    comparison_status_counts: Counter[str] = Counter()
    query_sweep_no_passing_strategy_count = 0
    query_sweep_failure_reason_counts: Counter[str] = Counter()
    query_sweep_recommended_next_action_counts: Counter[str] = Counter()
    query_sweep_best_observed_strategy_counts: Counter[str] = Counter()
    query_sweep_no_hit_strategy_count = 0
    query_sweep_target_route_not_selected_strategy_count = 0
    query_sweep_blind_refuted_rate_below_min_strategy_count = 0
    query_sweep_verified_false_alarm_above_max_strategy_count = 0
    best_observed_records_with_hits_sum = 0
    best_observed_records_with_hits_max: int | None = None
    best_observed_total_hits_sum = 0
    best_observed_total_hits_max: int | None = None
    max_source_docs = 0
    blocking_reasons: Counter[str] = Counter()
    for index, workflow in enumerate(workflows, start=1):
        status = str(workflow.get("status") or "unknown")
        status_counts[status] += 1
        evidence = _mapping(workflow.get("evidence_summary")) or _mapping(workflow.get("summary"))
        gate = _mapping(workflow.get("gate"))
        provenance_passed_rows = _int(evidence.get("provenance_passed_count"))
        provenance_failed_rows = _int(evidence.get("provenance_failed_count"))
        if provenance_passed_rows:
            provenance_pass_count += provenance_passed_rows
        elif evidence.get("provenance_passed") is True:
            provenance_pass_count += 1
        if provenance_failed_rows:
            provenance_failed_count += provenance_failed_rows
        elif evidence.get("provenance_passed") is False:
            provenance_failed_count += 1
        provenance_status_counts.update(_int_mapping(evidence.get("provenance_status_counts")))
        if not evidence.get("provenance_status_counts") and evidence.get("provenance_status"):
            provenance_status_counts[str(evidence.get("provenance_status"))] += 1
        comparison_passed_rows = _int(evidence.get("comparison_passed_count"))
        comparison_failed_rows = _int(evidence.get("comparison_failed_count"))
        if comparison_passed_rows:
            comparison_pass_count += comparison_passed_rows
        elif evidence.get("comparison_passed") is True:
            comparison_pass_count += 1
        if comparison_failed_rows:
            comparison_failed_count += comparison_failed_rows
        elif evidence.get("comparison_passed") is False:
            comparison_failed_count += 1
        comparison_status_counts.update(_int_mapping(evidence.get("comparison_status_counts")))
        if not evidence.get("comparison_status_counts") and evidence.get("comparison_status"):
            comparison_status_counts[str(evidence.get("comparison_status"))] += 1
        no_passing_value = evidence.get("query_sweep_no_passing_strategy_count")
        no_passing_rows = _int(no_passing_value)
        query_sweep_no_passing_strategy_count += no_passing_rows
        if no_passing_value is None and evidence.get("query_sweep_best_strategy") and not evidence.get(
            "query_sweep_best_passing_strategy"
        ):
            query_sweep_no_passing_strategy_count += 1
        query_sweep_failure_reason_counts.update(
            _int_mapping(evidence.get("query_sweep_failure_reason_counts"))
        )
        query_sweep_recommended_next_action_counts.update(
            _int_mapping(evidence.get("query_sweep_recommended_next_action_counts"))
        )
        if not evidence.get("query_sweep_recommended_next_action_counts"):
            query_sweep_recommended_next_action_counts.update(
                str(action)
                for action in _string_tuple(evidence.get("query_sweep_recommended_next_actions"))
            )
        query_sweep_no_hit_strategy_count += _int(
            evidence.get("query_sweep_no_hit_strategy_count")
        )
        query_sweep_target_route_not_selected_strategy_count += _int(
            evidence.get("query_sweep_target_route_not_selected_strategy_count")
        )
        query_sweep_blind_refuted_rate_below_min_strategy_count += _int(
            evidence.get("query_sweep_blind_refuted_rate_below_min_strategy_count")
        )
        query_sweep_verified_false_alarm_above_max_strategy_count += _int(
            evidence.get("query_sweep_verified_false_alarm_above_max_strategy_count")
        )
        aggregate_best_strategy_counts = _int_mapping(
            evidence.get("query_sweep_best_observed_strategy_counts")
        )
        if aggregate_best_strategy_counts:
            query_sweep_best_observed_strategy_counts.update(aggregate_best_strategy_counts)
        elif evidence.get("query_sweep_best_observed_strategy"):
            query_sweep_best_observed_strategy_counts[
                str(evidence.get("query_sweep_best_observed_strategy"))
            ] += 1
        aggregate_records_with_hits_sum = _optional_int(
            evidence.get("query_sweep_best_observed_records_with_hits_sum")
        )
        aggregate_records_with_hits_max = _optional_int(
            evidence.get("query_sweep_best_observed_records_with_hits_max")
        )
        if aggregate_records_with_hits_sum is not None:
            best_observed_records_with_hits_sum += aggregate_records_with_hits_sum
            if aggregate_records_with_hits_max is not None:
                best_observed_records_with_hits_max = _max_optional_int(
                    best_observed_records_with_hits_max,
                    aggregate_records_with_hits_max,
                )
        else:
            records_with_hits = _optional_int(
                evidence.get("query_sweep_best_observed_records_with_hits")
            )
            if records_with_hits is not None:
                best_observed_records_with_hits_sum += records_with_hits
                best_observed_records_with_hits_max = _max_optional_int(
                    best_observed_records_with_hits_max,
                    records_with_hits,
                )
        aggregate_total_hits_sum = _optional_int(
            evidence.get("query_sweep_best_observed_total_hits_sum")
        )
        aggregate_total_hits_max = _optional_int(
            evidence.get("query_sweep_best_observed_total_hits_max")
        )
        if aggregate_total_hits_sum is not None:
            best_observed_total_hits_sum += aggregate_total_hits_sum
            if aggregate_total_hits_max is not None:
                best_observed_total_hits_max = _max_optional_int(
                    best_observed_total_hits_max,
                    aggregate_total_hits_max,
                )
        else:
            total_hits = _optional_int(
                evidence.get("query_sweep_best_observed_total_hits")
            )
            if total_hits is not None:
                best_observed_total_hits_sum += total_hits
                best_observed_total_hits_max = _max_optional_int(
                    best_observed_total_hits_max,
                    total_hits,
                )
        max_source_docs = max(max_source_docs, _int(evidence.get("source_document_count")))
        for item in _mapping_sequence(gate.get("blocking_reasons")):
            reason = str(item.get("gate") or item.get("reason") or "unknown")
            blocking_reasons[reason] += 1
        rows.append({
            "index": index,
            "workflow": workflow.get("workflow"),
            "status": status,
            "gate_passed": gate.get("passed"),
            "promotion_ready": gate.get("promotion_ready"),
            "adapter_request_count": _int(evidence.get("adapter_request_count")),
            "source_document_count": _int(evidence.get("source_document_count")),
            "provenance_passed_count": provenance_passed_rows,
            "provenance_failed_count": provenance_failed_rows,
            "provenance_status": evidence.get("provenance_status"),
            "comparison_passed_count": comparison_passed_rows,
            "comparison_failed_count": comparison_failed_rows,
            "comparison_status": evidence.get("comparison_status"),
            "query_sweep_no_passing_strategy_count": no_passing_rows,
            "query_sweep_best_strategy": evidence.get("query_sweep_best_strategy"),
            "query_sweep_best_passing_strategy": evidence.get(
                "query_sweep_best_passing_strategy"
            ),
            "query_sweep_best_passing_blind_refuted_count": evidence.get(
                "query_sweep_best_passing_blind_refuted_count"
            ),
            "query_sweep_failure_reason_counts": _int_mapping(
                evidence.get("query_sweep_failure_reason_counts")
            ),
            "query_sweep_recommended_next_actions": _string_tuple(
                evidence.get("query_sweep_recommended_next_actions")
            ),
            "query_sweep_no_hit_strategy_count": _int(
                evidence.get("query_sweep_no_hit_strategy_count")
            ),
            "query_sweep_target_route_not_selected_strategy_count": _int(
                evidence.get("query_sweep_target_route_not_selected_strategy_count")
            ),
            "query_sweep_blind_refuted_rate_below_min_strategy_count": _int(
                evidence.get("query_sweep_blind_refuted_rate_below_min_strategy_count")
            ),
            "query_sweep_verified_false_alarm_above_max_strategy_count": _int(
                evidence.get("query_sweep_verified_false_alarm_above_max_strategy_count")
            ),
            "query_sweep_best_observed_strategy": evidence.get(
                "query_sweep_best_observed_strategy"
            ),
            "query_sweep_best_observed_failure_reasons": _string_tuple(
                evidence.get("query_sweep_best_observed_failure_reasons")
            ),
        })
    if not rows:
        status = "missing"
    elif any(row["status"] == "promote" or row["gate_passed"] is True for row in rows):
        status = "promote"
    else:
        status = "blocked"
    return {
        "status": status,
        "workflow_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "provenance_pass_count": provenance_pass_count,
        "provenance_failed_count": provenance_failed_count,
        "provenance_status_counts": dict(sorted(provenance_status_counts.items())),
        "comparison_pass_count": comparison_pass_count,
        "comparison_failed_count": comparison_failed_count,
        "comparison_status_counts": dict(sorted(comparison_status_counts.items())),
        "query_sweep_no_passing_strategy_count": query_sweep_no_passing_strategy_count,
        "query_sweep_failure_reason_counts": dict(
            sorted(query_sweep_failure_reason_counts.items())
        ),
        "query_sweep_recommended_next_action_counts": dict(
            sorted(query_sweep_recommended_next_action_counts.items())
        ),
        "query_sweep_no_hit_strategy_count": query_sweep_no_hit_strategy_count,
        "query_sweep_target_route_not_selected_strategy_count": (
            query_sweep_target_route_not_selected_strategy_count
        ),
        "query_sweep_blind_refuted_rate_below_min_strategy_count": (
            query_sweep_blind_refuted_rate_below_min_strategy_count
        ),
        "query_sweep_verified_false_alarm_above_max_strategy_count": (
            query_sweep_verified_false_alarm_above_max_strategy_count
        ),
        "query_sweep_best_observed_strategy_counts": dict(
            sorted(query_sweep_best_observed_strategy_counts.items())
        ),
        "query_sweep_best_observed_records_with_hits_sum": (
            best_observed_records_with_hits_sum
        ),
        "query_sweep_best_observed_records_with_hits_max": (
            best_observed_records_with_hits_max
        ),
        "query_sweep_best_observed_total_hits_sum": best_observed_total_hits_sum,
        "query_sweep_best_observed_total_hits_max": best_observed_total_hits_max,
        "max_source_document_count": max_source_docs,
        "blocking_reason_counts": dict(sorted(blocking_reasons.items())),
        "workflows": tuple(rows),
    }


def _semantic_gap_review_lane(workflows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    status_counts: Counter[str] = Counter()
    candidate_count = 0
    fact_candidate_count = 0
    fact_review_document_count = 0
    approved_source_document_count = 0
    source_family_qa_document_count = 0
    covered_route_records = 0
    promoted_count = 0
    blocked_count = 0
    skipped_route_count = 0
    best_decision_accuracy: float | None = None
    best_false_refuted_rate: float | None = None
    for index, workflow in enumerate(workflows, start=1):
        status = str(workflow.get("status") or "unknown")
        summary = _mapping(workflow.get("summary"))
        stage_status = _mapping(workflow.get("stage_status"))
        route_status = str(
            summary.get("covered_fact_route_status")
            or stage_status.get("covered_fact_route")
            or ""
        )
        status_counts[status] += 1
        candidate_count += _int(summary.get("semantic_gap_candidate_count"))
        fact_candidate_count += _int(summary.get("semantic_gap_fact_candidate_count"))
        fact_review_document_count += _int(summary.get("fact_review_document_count"))
        approved_source_document_count += _int(summary.get("approved_source_document_count"))
        source_family_qa_document_count += _int(summary.get("source_family_qa_document_count"))
        covered_route_records += _int(summary.get("covered_fact_route_n_records"))
        decision_accuracy = _optional_float(summary.get("covered_fact_route_decision_accuracy"))
        false_refuted_rate = _optional_float(summary.get("covered_fact_route_false_refuted_rate"))
        if decision_accuracy is not None:
            best_decision_accuracy = (
                decision_accuracy
                if best_decision_accuracy is None
                else max(best_decision_accuracy, decision_accuracy)
            )
        if false_refuted_rate is not None:
            best_false_refuted_rate = (
                false_refuted_rate
                if best_false_refuted_rate is None
                else max(best_false_refuted_rate, false_refuted_rate)
            )
        promoted = status == "covered_fact_route_promote" or route_status == "promote"
        if promoted:
            promoted_count += 1
        elif status in {"covered_fact_route_blocked", "blocked"}:
            blocked_count += 1
        elif (
            status == "covered_fact_route_skipped"
            or route_status == "insufficient_qa_documents"
        ):
            skipped_route_count += 1
        rows.append({
            "index": index,
            "workflow": workflow.get("workflow"),
            "status": status,
            "covered_fact_route_status": route_status or None,
            "semantic_gap_candidate_count": _int(summary.get("semantic_gap_candidate_count")),
            "semantic_gap_fact_candidate_count": _int(
                summary.get("semantic_gap_fact_candidate_count")
            ),
            "fact_review_document_count": _int(summary.get("fact_review_document_count")),
            "approved_source_document_count": _int(summary.get("approved_source_document_count")),
            "source_family_qa_document_count": _int(summary.get("source_family_qa_document_count")),
            "covered_fact_route_n_records": _int(summary.get("covered_fact_route_n_records")),
            "covered_fact_route_decision_accuracy": decision_accuracy,
            "covered_fact_route_false_refuted_rate": false_refuted_rate,
        })
    if not rows:
        status = "not_configured"
    elif promoted_count:
        status = "promote"
    elif approved_source_document_count and not covered_route_records:
        status = "ready_for_covered_fact_route"
    elif fact_candidate_count or fact_review_document_count or blocked_count or skipped_route_count:
        status = "needs_evidence"
    else:
        status = "missing"
    return {
        "status": status,
        "workflow_count": len(rows),
        "promoted_workflow_count": promoted_count,
        "blocked_workflow_count": blocked_count,
        "covered_fact_route_skipped_count": skipped_route_count,
        "status_counts": dict(sorted(status_counts.items())),
        "semantic_gap_candidate_count": candidate_count,
        "semantic_gap_fact_candidate_count": fact_candidate_count,
        "fact_review_document_count": fact_review_document_count,
        "approved_source_document_count": approved_source_document_count,
        "source_family_qa_document_count": source_family_qa_document_count,
        "covered_fact_route_n_records": covered_route_records,
        "best_covered_fact_route_decision_accuracy": best_decision_accuracy,
        "best_covered_fact_route_false_refuted_rate": best_false_refuted_rate,
        "workflows": tuple(rows),
    }


def _frontier_queue_execution_lane(
    command_binding_reviews: Sequence[Mapping[str, Any]],
    bound_command_runs: Sequence[Mapping[str, Any]],
    queue_execution_smokes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    smoke_rows = []
    smoke_status_counts: Counter[str] = Counter()
    smoke_passed_count = 0
    smoke_failed_count = 0
    smoke_unverified_count = 0
    smoke_manifest_verified_count = 0
    smoke_manifest_failed_count = 0
    for index, report in enumerate(queue_execution_smokes, start=1):
        row = _frontier_queue_execution_smoke_row(report, index=index)
        smoke_rows.append(row)
        status = str(row["status"] or "unknown")
        smoke_status_counts[status] += 1
        if row["manifest_verified"] is True:
            smoke_manifest_verified_count += 1
        elif row["manifest_verified"] is False:
            smoke_manifest_failed_count += 1
        if row["passed"] is True:
            smoke_passed_count += 1
        elif row["unverified"] is True:
            smoke_unverified_count += 1
        else:
            smoke_failed_count += 1

    review_rows = []
    review_status_counts: Counter[str] = Counter()
    review_entry_count = 0
    approved_entry_count = 0
    blocked_entry_count = 0
    pending_review_count = 0
    approved_binding_count = 0
    missing_binding_count = 0
    review_failure_counts: Counter[str] = Counter()
    for index, report in enumerate(command_binding_reviews, start=1):
        status = str(report.get("status") or "unknown")
        summary = _mapping(report.get("summary"))
        review_status_counts[status] += 1
        review_entry_count += _int(summary.get("entry_count"))
        approved_entry_count += _int(summary.get("approved_entry_count"))
        blocked_entry_count += _int(summary.get("blocked_entry_count"))
        pending_review_count += _int(summary.get("pending_review_count"))
        approved_binding_count += _int(summary.get("approved_binding_count"))
        missing_binding_count += _int(summary.get("missing_binding_count"))
        review_failure_counts.update(_int_mapping(summary.get("failure_counts")))
        review_rows.append({
            "index": index,
            "workflow": report.get("workflow"),
            "status": status,
            "entry_count": _int(summary.get("entry_count")),
            "approved_entry_count": _int(summary.get("approved_entry_count")),
            "blocked_entry_count": _int(summary.get("blocked_entry_count")),
            "pending_review_count": _int(summary.get("pending_review_count")),
            "approved_binding_count": _int(summary.get("approved_binding_count")),
            "missing_binding_count": _int(summary.get("missing_binding_count")),
            "failure_counts": _int_mapping(summary.get("failure_counts")),
        })

    run_rows = []
    run_status_counts: Counter[str] = Counter()
    dry_run_report_count = 0
    executed_report_count = 0
    succeeded_run_report_count = 0
    unreviewed_execution_override_count = 0
    command_count = 0
    dry_run_count = 0
    executed_count = 0
    succeeded_count = 0
    failed_count = 0
    timed_out_count = 0
    skipped_count = 0
    invalid_command_count = 0
    binding_not_reviewed_count = 0
    materialized_output_count = 0
    missing_output_count = 0
    planned_output_count = 0
    unchecked_output_count = 0
    for index, report in enumerate(bound_command_runs, start=1):
        status = str(report.get("status") or "unknown")
        summary = _mapping(report.get("summary"))
        config = _mapping(report.get("config"))
        dry_run = config.get("dry_run") is True
        executes_commands = config.get("executes_commands") is True
        require_reviewed = config.get("require_reviewed_bindings")
        run_status_counts[status] += 1
        if dry_run:
            dry_run_report_count += 1
        if executes_commands:
            executed_report_count += 1
            if require_reviewed is False:
                unreviewed_execution_override_count += 1
        if status == "succeeded":
            succeeded_run_report_count += 1
        command_count += _int(summary.get("command_count"))
        dry_run_count += _int(summary.get("dry_run_count"))
        executed_count += _int(summary.get("executed_count"))
        succeeded_count += _int(summary.get("succeeded_count"))
        failed_count += _int(summary.get("failed_count"))
        timed_out_count += _int(summary.get("timed_out_count"))
        skipped_count += _int(summary.get("skipped_count"))
        invalid_command_count += _int(summary.get("invalid_command_count"))
        binding_not_reviewed_count += _int(summary.get("binding_not_reviewed_count"))
        materialized_output_count += _int(summary.get("materialized_output_count"))
        missing_output_count += _int(summary.get("missing_output_count"))
        planned_output_count += _int(summary.get("planned_output_count"))
        unchecked_output_count += _int(summary.get("unchecked_output_count"))
        run_rows.append({
            "index": index,
            "workflow": report.get("workflow"),
            "status": status,
            "dry_run": dry_run,
            "executes_commands": executes_commands,
            "require_reviewed_bindings": require_reviewed,
            "entry_count": _int(summary.get("entry_count")),
            "command_count": _int(summary.get("command_count")),
            "dry_run_count": _int(summary.get("dry_run_count")),
            "executed_count": _int(summary.get("executed_count")),
            "succeeded_count": _int(summary.get("succeeded_count")),
            "failed_count": _int(summary.get("failed_count")),
            "timed_out_count": _int(summary.get("timed_out_count")),
            "skipped_count": _int(summary.get("skipped_count")),
            "invalid_command_count": _int(summary.get("invalid_command_count")),
            "binding_not_reviewed_count": _int(summary.get("binding_not_reviewed_count")),
            "materialized_output_count": _int(summary.get("materialized_output_count")),
            "missing_output_count": _int(summary.get("missing_output_count")),
            "planned_output_count": _int(summary.get("planned_output_count")),
            "unchecked_output_count": _int(summary.get("unchecked_output_count")),
        })

    if not review_rows and not run_rows:
        status = "not_configured"
    elif (
        blocked_entry_count
        or pending_review_count
        or missing_binding_count
        or binding_not_reviewed_count
        or unreviewed_execution_override_count
        or any(row["status"] not in {"ready_for_execution"} for row in review_rows)
    ):
        status = "needs_review"
    elif invalid_command_count or failed_count or timed_out_count or missing_output_count:
        status = "blocked"
    elif skipped_count:
        status = "needs_inputs"
    elif succeeded_run_report_count and executed_count and succeeded_count == executed_count:
        status = "promote"
    elif dry_run_report_count or any(row["status"] == "ready_for_execution" for row in review_rows):
        status = "needs_execution"
    else:
        status = "needs_execution"

    smoke_health_status = "not_configured"
    if smoke_rows:
        if smoke_failed_count or smoke_manifest_failed_count:
            smoke_health_status = "failed"
        elif smoke_unverified_count:
            smoke_health_status = "unverified"
        elif smoke_passed_count == len(smoke_rows):
            smoke_health_status = "pass"
        else:
            smoke_health_status = "unknown"

    return {
        "status": status,
        "control_plane_smoke_status": smoke_health_status,
        "frontier_queue_execution_smoke_count": len(smoke_rows),
        "frontier_queue_execution_smoke_passed_count": smoke_passed_count,
        "frontier_queue_execution_smoke_failed_count": smoke_failed_count,
        "frontier_queue_execution_smoke_unverified_count": smoke_unverified_count,
        "frontier_queue_execution_smoke_manifest_verified_count": (
            smoke_manifest_verified_count
        ),
        "frontier_queue_execution_smoke_manifest_failed_count": (
            smoke_manifest_failed_count
        ),
        "frontier_queue_execution_smoke_status_counts": dict(
            sorted(smoke_status_counts.items())
        ),
        "command_binding_review_count": len(review_rows),
        "ready_review_count": review_status_counts.get("ready_for_execution", 0),
        "review_status_counts": dict(sorted(review_status_counts.items())),
        "review_entry_count": review_entry_count,
        "approved_entry_count": approved_entry_count,
        "blocked_entry_count": blocked_entry_count,
        "pending_review_count": pending_review_count,
        "approved_binding_count": approved_binding_count,
        "missing_binding_count": missing_binding_count,
        "review_failure_counts": dict(sorted(review_failure_counts.items())),
        "bound_command_run_count": len(run_rows),
        "run_status_counts": dict(sorted(run_status_counts.items())),
        "dry_run_report_count": dry_run_report_count,
        "executed_report_count": executed_report_count,
        "succeeded_run_report_count": succeeded_run_report_count,
        "unreviewed_execution_override_count": unreviewed_execution_override_count,
        "command_count": command_count,
        "dry_run_count": dry_run_count,
        "executed_count": executed_count,
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "timed_out_count": timed_out_count,
        "skipped_count": skipped_count,
        "invalid_command_count": invalid_command_count,
        "binding_not_reviewed_count": binding_not_reviewed_count,
        "materialized_output_count": materialized_output_count,
        "missing_output_count": missing_output_count,
        "planned_output_count": planned_output_count,
        "unchecked_output_count": unchecked_output_count,
        "command_binding_reviews": tuple(review_rows),
        "bound_command_runs": tuple(run_rows),
        "frontier_queue_execution_smokes": tuple(smoke_rows),
    }


def _frontier_queue_execution_smoke_row(
    report: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    label_usage = _mapping(report.get("label_usage"))
    verification = _mapping(report.get("manifest_verification"))
    paths = _mapping(report.get("paths"))
    manifest_verified: bool | None = None
    if verification:
        manifest_verified = verification.get("passed") is True

    failure_reasons: list[str] = []
    if report.get("workflow") != "frontier_queue_execution_smoke":
        failure_reasons.append("workflow is not frontier_queue_execution_smoke")
    if report.get("status") != "pass":
        failure_reasons.append("smoke status is not pass")
    if _int(summary.get("staged_upstream_output_count")) < 1:
        failure_reasons.append("no upstream output was staged")
    if _int(summary.get("remaining_placeholder_count")):
        failure_reasons.append("command placeholders remain unbound")
    if _int(summary.get("dry_run_count")) < 1:
        failure_reasons.append("dry-run did not cover any child command")
    if _int(summary.get("binding_not_reviewed_count")):
        failure_reasons.append("dry-run found unreviewed bindings")
    if label_usage.get("artifacts_are_verifier_evidence") is not False:
        failure_reasons.append("smoke artifacts are not explicitly marked non-evidence")
    if label_usage.get("executes_child_commands") is not False:
        failure_reasons.append("smoke did not explicitly avoid child command execution")
    if manifest_verified is False:
        failure_reasons.append("artifact manifest verification failed")
    if manifest_verified is None:
        failure_reasons.append("artifact manifest verification is missing")

    unverified = failure_reasons == ["artifact manifest verification is missing"]
    passed = not failure_reasons
    return {
        "index": index,
        "workflow": report.get("workflow"),
        "status": str(report.get("status") or "unknown"),
        "passed": passed,
        "unverified": unverified,
        "manifest_verified": manifest_verified,
        "failure_reasons": tuple(failure_reasons),
        "staged_upstream_output_count": _int(summary.get("staged_upstream_output_count")),
        "remaining_placeholder_count": _int(summary.get("remaining_placeholder_count")),
        "review_approved_entry_count": _int(summary.get("review_approved_entry_count")),
        "dry_run_count": _int(summary.get("dry_run_count")),
        "binding_not_reviewed_count": _int(summary.get("binding_not_reviewed_count")),
        "executes_child_commands": label_usage.get("executes_child_commands"),
        "artifacts_are_verifier_evidence": label_usage.get(
            "artifacts_are_verifier_evidence"
        ),
        "artifact_manifest": paths.get("artifact_manifest"),
        "dry_run_report": paths.get("dry_run_report"),
        "review_report": paths.get("review_report"),
    }


def _world_model_rule_lane(
    *,
    rule_input_plan: Mapping[str, Any] | None,
    rule_input_audit_report: Mapping[str, Any] | None,
    rule_stub_requeue_report: Mapping[str, Any] | None,
    requeued_rule_input_plan: Mapping[str, Any] | None,
    input_fill_result_rollup: Mapping[str, Any] | None,
    rule_promotion_reports: Sequence[Mapping[str, Any]],
    mechanism_handoff_bundle: Mapping[str, Any] | None,
) -> dict[str, Any]:
    plan_summary = _mapping(None if rule_input_plan is None else rule_input_plan.get("summary"))
    audit_summary = _mapping(
        None if rule_input_audit_report is None else rule_input_audit_report.get("summary")
    )
    requeue_summary = _mapping(
        None if rule_stub_requeue_report is None else rule_stub_requeue_report.get("summary")
    )
    requeued_plan_summary = _mapping(
        None if requeued_rule_input_plan is None else requeued_rule_input_plan.get("summary")
    )
    fill_rollup_summary = _mapping(
        None if input_fill_result_rollup is None else input_fill_result_rollup.get("summary")
    )
    fill_rollup_status = (
        None if input_fill_result_rollup is None else str(input_fill_result_rollup.get("status") or "")
    )
    combined_rule_input_count = _int(fill_rollup_summary.get("combined_rule_input_count"))
    combined_unfilled_task_count = _int(fill_rollup_summary.get("combined_unfilled_task_count"))
    fill_rollup_blocked_report_count = _int(fill_rollup_summary.get("blocked_fill_report_count"))
    fill_rollup_duplicate_request_id_count = _int(
        fill_rollup_summary.get("duplicate_request_id_count")
    )
    fill_rollup_rule_family_counts = _int_mapping(fill_rollup_summary.get("rule_family_counts"))
    input_fill_adapter_ready = (
        fill_rollup_status in {"ready_for_adapter", "partial"}
        and combined_rule_input_count > 0
        and fill_rollup_blocked_report_count == 0
        and fill_rollup_duplicate_request_id_count == 0
    )
    task_count = _int(plan_summary.get("task_count"))
    rule_family_counts = _int_mapping(plan_summary.get("rule_family_counts"))
    missing_input_counts = _int_mapping(plan_summary.get("missing_input_counts"))
    requeue_suggestions = _mapping_sequence(
        None if rule_input_audit_report is None else rule_input_audit_report.get("requeue_suggestions")
    )
    audit_requeue_count = _int(audit_summary.get("requeue_suggestion_count"))
    if requeue_suggestions:
        audit_requeue_count = len(requeue_suggestions)
    requeued_stub_count = _int(requeue_summary.get("requeued_stub_count"))
    skipped_requeue_count = _int(requeue_summary.get("skipped_suggestion_count"))
    requeue_outstanding_count = (
        max(audit_requeue_count - requeued_stub_count, 0) + skipped_requeue_count
    )
    promotion_rows = []
    promoted_count = 0
    blocked_count = 0
    pending_count = 0
    promoted_request_ids: list[str] = []
    promoted_rule_families: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for index, report in enumerate(rule_promotion_reports, start=1):
        summary = _mapping(report.get("summary"))
        status = str(report.get("status") or "unknown")
        status_counts[status] += 1
        promoted = _int(summary.get("promoted_count"))
        blocked = _int(summary.get("blocked_count"))
        pending = _int(summary.get("pending_count"))
        promoted_count += promoted
        blocked_count += blocked
        pending_count += pending
        promoted_rule_families.update(_int_mapping(summary.get("promoted_rule_family_counts")))
        promoted_request_ids.extend(_string_tuple(summary.get("promoted_request_ids", ())))
        promotion_rows.append({
            "index": index,
            "workflow": report.get("workflow"),
            "status": status,
            "promoted_count": promoted,
            "blocked_count": blocked,
            "pending_count": pending,
            "executed_count": _int(summary.get("executed_count")),
            "promoted_rule_family_counts": _int_mapping(
                summary.get("promoted_rule_family_counts")
            ),
            "promoted_request_ids": _string_tuple(summary.get("promoted_request_ids", ())),
            "status_counts": _int_mapping(summary.get("status_counts")),
        })
    bundle_summary = _mapping(
        None if mechanism_handoff_bundle is None else mechanism_handoff_bundle.get("summary")
    )
    bundle_status = None if mechanism_handoff_bundle is None else mechanism_handoff_bundle.get("status")
    mechanism_trace_count = _int(bundle_summary.get("trace_count"))
    mechanism_target_count = _int(bundle_summary.get("target_count"))
    closed_rule_family_counts, remaining_rule_family_counts = _rule_family_closure_counts(
        rule_family_counts,
        promoted_rule_families,
    )
    audit_adjusted_remaining_rule_family_counts = _audit_adjusted_remaining_rule_family_counts(
        remaining_rule_family_counts,
        requeue_suggestions=requeue_suggestions,
        rule_family_counts=rule_family_counts,
        promoted_rule_families=promoted_rule_families,
    )
    audit_adjusted_remaining_task_count = sum(audit_adjusted_remaining_rule_family_counts.values())
    audit_adjusted_required_input_counts = _required_input_counts_by_rule_family(
        audit_adjusted_remaining_rule_family_counts
    )
    remaining_task_count = sum(remaining_rule_family_counts.values())
    remaining_missing_input_counts = _remaining_missing_input_counts(
        missing_input_counts,
        remaining_rule_family_counts,
    )
    if (
        rule_input_plan is None
        and input_fill_result_rollup is None
        and not promotion_rows
        and mechanism_handoff_bundle is None
    ):
        status = "missing"
    elif requeue_outstanding_count:
        status = "needs_requeue"
    elif fill_rollup_status == "blocked":
        status = "blocked"
    elif input_fill_adapter_ready and not promotion_rows and not audit_adjusted_remaining_task_count:
        status = "ready_for_adapter"
    elif audit_adjusted_remaining_task_count:
        status = "partial"
    elif input_fill_result_rollup is not None and not input_fill_adapter_ready and not promotion_rows:
        status = "needs_inputs"
    elif input_fill_result_rollup is not None and combined_unfilled_task_count:
        status = "needs_inputs"
    elif blocked_count:
        status = "blocked"
    elif pending_count and audit_adjusted_remaining_task_count:
        status = "needs_inputs"
    elif bundle_status not in {None, "promote"}:
        status = "blocked"
    else:
        status = "promote"
    return {
        "status": status,
        "rule_input_plan_status": None if rule_input_plan is None else rule_input_plan.get("status"),
        "rule_input_audit_status": None
        if rule_input_audit_report is None
        else rule_input_audit_report.get("status"),
        "rule_input_audit_task_count": _int(audit_summary.get("task_count")),
        "rule_input_audit_finding_count": _int(audit_summary.get("finding_count")),
        "rule_input_audit_requeue_suggestion_count": audit_requeue_count,
        "rule_input_audit_finding_counts": _int_mapping(audit_summary.get("finding_counts")),
        "rule_input_audit_recommended_rule_family_counts": _int_mapping(
            audit_summary.get("recommended_rule_family_counts")
        ),
        "rule_input_audit_recommended_action_counts": _int_mapping(
            audit_summary.get("recommended_action_counts")
        ),
        "rule_stub_requeue_status": None
        if rule_stub_requeue_report is None
        else rule_stub_requeue_report.get("status"),
        "rule_stub_requeue_requeued_stub_count": requeued_stub_count,
        "rule_stub_requeue_skipped_suggestion_count": skipped_requeue_count,
        "rule_stub_requeue_rule_family_counts": _int_mapping(
            requeue_summary.get("rule_family_counts")
        ),
        "rule_input_audit_requeue_outstanding_count": requeue_outstanding_count,
        "requeued_rule_input_plan_status": None
        if requeued_rule_input_plan is None
        else requeued_rule_input_plan.get("status"),
        "requeued_rule_input_task_count": _int(requeued_plan_summary.get("task_count")),
        "requeued_rule_family_counts": _int_mapping(
            requeued_plan_summary.get("rule_family_counts")
        ),
        "requeued_rule_input_execution_input_counts": _int_mapping(
            requeued_plan_summary.get("execution_input_counts")
        ),
        "requeued_rule_input_missing_input_counts": _int_mapping(
            requeued_plan_summary.get("missing_input_counts")
        ),
        "input_fill_result_rollup_status": fill_rollup_status,
        "input_fill_adapter_ready": input_fill_adapter_ready,
        "input_fill_rollup_blocked_fill_report_count": fill_rollup_blocked_report_count,
        "input_fill_rollup_duplicate_request_id_count": fill_rollup_duplicate_request_id_count,
        "combined_rule_input_count": combined_rule_input_count,
        "combined_unfilled_task_count": combined_unfilled_task_count,
        "input_fill_rule_family_counts": fill_rollup_rule_family_counts,
        "input_fill_downstream_adapter_command": _mapping(
            None
            if input_fill_result_rollup is None
            else input_fill_result_rollup.get("downstream_adapter_command")
        ),
        "task_count": task_count,
        "rule_family_counts": rule_family_counts,
        "missing_input_counts": missing_input_counts,
        "closed_rule_family_counts": dict(sorted(closed_rule_family_counts.items())),
        "remaining_rule_family_counts": dict(sorted(remaining_rule_family_counts.items())),
        "audit_adjusted_remaining_rule_family_counts": dict(
            sorted(audit_adjusted_remaining_rule_family_counts.items())
        ),
        "audit_adjusted_required_input_counts": dict(
            sorted(audit_adjusted_required_input_counts.items())
        ),
        "remaining_task_count": remaining_task_count,
        "audit_adjusted_remaining_task_count": audit_adjusted_remaining_task_count,
        "remaining_missing_input_counts": dict(sorted(remaining_missing_input_counts.items())),
        "promotion_report_count": len(promotion_rows),
        "promotion_status_counts": dict(sorted(status_counts.items())),
        "promoted_count": promoted_count,
        "promoted_rule_request_ids": tuple(dict.fromkeys(promoted_request_ids)),
        "blocked_count": blocked_count,
        "pending_count": pending_count,
        "promoted_rule_family_counts": dict(sorted(promoted_rule_families.items())),
        "mechanism_handoff_bundle_status": bundle_status,
        "mechanism_handoff_trace_count": mechanism_trace_count,
        "mechanism_handoff_target_count": mechanism_target_count,
        "promotion_reports": tuple(promotion_rows),
    }


def _next_actions(lanes: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    source = lanes["source_family_acquisition"]
    citation = lanes["citation_evidence"]
    semantic = lanes["semantic_gap_review"]
    frontier_queue = lanes["frontier_queue_execution"]
    rules = lanes["world_model_rules"]
    queue = lanes["unresolved_queue"]
    if source.get("status") != "covered":
        actions.append({
            "action_id": "complete_source_family_catalog_acquisition",
            "priority": 90,
            "lane": "source_family_acquisition",
            "reason": "source-family coverage is not complete for unresolved citation requests",
        })
    if (
        source.get("status") == "covered"
        and citation.get("status") != "promote"
        and not _semantic_gap_review_covers_queue(semantic, queue)
    ):
        actions.append({
            "action_id": "improve_unresolved_citation_alignment",
            "priority": 88,
            "lane": "citation_evidence",
            "reason": (
                "source acquisition is covered but citation/search evidence gates "
                "still do not promote; inspect query alignment, claim mapping, or "
                "route thresholds before release use"
            ),
            "query_sweep_failure_reason_counts": citation.get(
                "query_sweep_failure_reason_counts", {}
            ),
            "query_sweep_recommended_next_action_counts": citation.get(
                "query_sweep_recommended_next_action_counts", {}
            ),
            "query_sweep_no_hit_strategy_count": citation.get(
                "query_sweep_no_hit_strategy_count", 0
            ),
            "query_sweep_target_route_not_selected_strategy_count": citation.get(
                "query_sweep_target_route_not_selected_strategy_count", 0
            ),
            "query_sweep_blind_refuted_rate_below_min_strategy_count": citation.get(
                "query_sweep_blind_refuted_rate_below_min_strategy_count", 0
            ),
            "query_sweep_verified_false_alarm_above_max_strategy_count": citation.get(
                "query_sweep_verified_false_alarm_above_max_strategy_count", 0
            ),
            "query_sweep_best_observed_strategy_counts": citation.get(
                "query_sweep_best_observed_strategy_counts", {}
            ),
            "semantic_gap_review_status": semantic.get("status"),
            "semantic_gap_covered_fact_route_n_records": semantic.get(
                "covered_fact_route_n_records", 0
            ),
            "unresolved_target_count": queue.get("target_count", 0),
        })
    if semantic.get("status") in {"ready_for_covered_fact_route", "needs_evidence", "missing"}:
        actions.append({
            "action_id": "complete_retrieval_semantic_gap_review",
            "priority": 87,
            "lane": "semantic_gap_review",
            "reason": (
                "retrieval semantic gaps have not yet produced promoted covered-fact "
                "route evidence for reviewed source-backed facts"
            ),
            "semantic_gap_candidate_count": semantic.get("semantic_gap_candidate_count", 0),
            "semantic_gap_fact_candidate_count": semantic.get("semantic_gap_fact_candidate_count", 0),
            "approved_source_document_count": semantic.get("approved_source_document_count", 0),
            "source_family_qa_document_count": semantic.get("source_family_qa_document_count", 0),
        })
    if _int(rules.get("rule_input_audit_requeue_outstanding_count")):
        actions.append({
            "action_id": "requeue_misaligned_world_model_rule_inputs",
            "priority": 86,
            "lane": "world_model_rules",
            "reason": (
                "rule-input audit found tasks whose rule family should be rebuilt "
                "before value collection or promotion"
            ),
            "requeue_suggestion_count": rules.get("rule_input_audit_requeue_suggestion_count", 0),
            "requeued_stub_count": rules.get("rule_stub_requeue_requeued_stub_count", 0),
            "requeue_outstanding_count": rules.get(
                "rule_input_audit_requeue_outstanding_count", 0
            ),
            "recommended_rule_family_counts": rules.get(
                "rule_input_audit_recommended_rule_family_counts", {}
            ),
            "finding_counts": rules.get("rule_input_audit_finding_counts", {}),
        })
    if rules.get("input_fill_adapter_ready") is True and not _int(
        rules.get("promotion_report_count")
    ):
        actions.append({
            "action_id": "run_world_model_rule_adapter_promotion_workflow",
            "priority": 85,
            "lane": "world_model_rules",
            "reason": (
                "reviewed rule-input fills have been rolled up and are ready "
                "for deterministic adapter replay plus promotion gating"
            ),
            "input_fill_result_rollup_status": rules.get("input_fill_result_rollup_status"),
            "combined_rule_input_count": rules.get("combined_rule_input_count", 0),
            "combined_unfilled_task_count": rules.get("combined_unfilled_task_count", 0),
            "input_fill_rule_family_counts": rules.get("input_fill_rule_family_counts", {}),
            "downstream_adapter_command": rules.get("input_fill_downstream_adapter_command", {}),
        })
    if rules.get("status") in {"needs_requeue", "partial", "needs_inputs", "blocked", "missing"}:
        actions.append({
            "action_id": "fill_and_promote_remaining_world_model_rules",
            "priority": 84,
            "lane": "world_model_rules",
            "reason": "world-model/calculator rule inputs are not fully promoted",
            "remaining_rule_family_counts": rules.get(
                "audit_adjusted_remaining_rule_family_counts",
                rules.get("remaining_rule_family_counts", {}),
            ),
            "raw_remaining_rule_family_counts": rules.get("remaining_rule_family_counts", {}),
            "missing_input_counts": rules.get("remaining_missing_input_counts", {}),
            "promoted_rule_request_ids": rules.get("promoted_rule_request_ids", ()),
            "audit_adjusted_required_input_counts": rules.get(
                "audit_adjusted_required_input_counts", {}
            ),
        })
    if frontier_queue.get("status") == "needs_review":
        actions.append({
            "action_id": "review_frontier_queue_command_bindings",
            "priority": 84,
            "lane": "frontier_queue_execution",
            "reason": (
                "frontier queue command bindings or execution records need an "
                "explicit non-evidence review before real execution can count "
                "as closure"
            ),
            "command_binding_review_count": frontier_queue.get(
                "command_binding_review_count", 0
            ),
            "blocked_entry_count": frontier_queue.get("blocked_entry_count", 0),
            "pending_review_count": frontier_queue.get("pending_review_count", 0),
            "binding_not_reviewed_count": frontier_queue.get(
                "binding_not_reviewed_count", 0
            ),
            "unreviewed_execution_override_count": frontier_queue.get(
                "unreviewed_execution_override_count", 0
            ),
        })
    if frontier_queue.get("status") in {"needs_execution"}:
        actions.append({
            "action_id": "execute_reviewed_frontier_queue_command_plan",
            "priority": 83,
            "lane": "frontier_queue_execution",
            "reason": (
                "frontier queue command bindings are reviewed or dry-run only; "
                "run the approved bound command plan to materialize child "
                "control artifacts"
            ),
            "ready_review_count": frontier_queue.get("ready_review_count", 0),
            "dry_run_report_count": frontier_queue.get("dry_run_report_count", 0),
            "bound_command_run_count": frontier_queue.get("bound_command_run_count", 0),
            "command_count": frontier_queue.get("command_count", 0),
        })
    if frontier_queue.get("status") in {"blocked", "needs_inputs"}:
        actions.append({
            "action_id": "repair_frontier_queue_command_execution",
            "priority": 82,
            "lane": "frontier_queue_execution",
            "reason": (
                "frontier queue command execution failed, skipped commands, "
                "or missed planned outputs"
            ),
            "failed_count": frontier_queue.get("failed_count", 0),
            "timed_out_count": frontier_queue.get("timed_out_count", 0),
            "skipped_count": frontier_queue.get("skipped_count", 0),
            "invalid_command_count": frontier_queue.get("invalid_command_count", 0),
            "missing_output_count": frontier_queue.get("missing_output_count", 0),
        })
    if frontier_queue.get("control_plane_smoke_status") in {"failed", "unverified"}:
        actions.append({
            "action_id": "repair_frontier_queue_execution_smoke",
            "priority": 81,
            "lane": "frontier_queue_execution",
            "reason": (
                "frontier queue execution smoke was supplied but did not pass "
                "control-plane health checks or manifest verification"
            ),
            "control_plane_smoke_status": frontier_queue.get(
                "control_plane_smoke_status"
            ),
            "frontier_queue_execution_smoke_count": frontier_queue.get(
                "frontier_queue_execution_smoke_count", 0
            ),
            "frontier_queue_execution_smoke_failed_count": frontier_queue.get(
                "frontier_queue_execution_smoke_failed_count", 0
            ),
            "frontier_queue_execution_smoke_unverified_count": frontier_queue.get(
                "frontier_queue_execution_smoke_unverified_count", 0
            ),
            "frontier_queue_execution_smoke_manifest_failed_count": (
                frontier_queue.get(
                    "frontier_queue_execution_smoke_manifest_failed_count", 0
                )
            ),
        })
    if _int(queue.get("target_count")) and not actions:
        actions.append({
            "action_id": "verify_unresolved_targets_are_closed",
            "priority": 70,
            "lane": "unresolved_queue",
            "reason": "unresolved targets remain in the source queue even though lane gates passed",
        })
    return actions


def _semantic_gap_review_covers_queue(
    semantic: Mapping[str, Any],
    queue: Mapping[str, Any],
) -> bool:
    if semantic.get("status") != "promote":
        return False
    target_count = _int(queue.get("target_count"))
    if target_count <= 0:
        return False
    return _int(semantic.get("covered_fact_route_n_records")) >= target_count


def _summary(lanes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    lane_statuses = {name: str(lane.get("status") or "unknown") for name, lane in lanes.items()}
    return {
        "unresolved_target_count": _int(lanes["unresolved_queue"].get("target_count")),
        "adapter_request_count": _int(lanes["unresolved_queue"].get("adapter_request_count")),
        "citation_workflow_count": _int(lanes["citation_evidence"].get("workflow_count")),
        "citation_provenance_pass_count": _int(
            lanes["citation_evidence"].get("provenance_pass_count")
        ),
        "citation_query_sweep_failure_reason_counts": dict(
            _mapping(lanes["citation_evidence"].get("query_sweep_failure_reason_counts"))
        ),
        "citation_query_sweep_recommended_next_action_counts": dict(
            _mapping(
                lanes["citation_evidence"].get(
                    "query_sweep_recommended_next_action_counts"
                )
            )
        ),
        "citation_query_sweep_no_hit_strategy_count": _int(
            lanes["citation_evidence"].get("query_sweep_no_hit_strategy_count")
        ),
        "citation_query_sweep_target_route_not_selected_strategy_count": _int(
            lanes["citation_evidence"].get(
                "query_sweep_target_route_not_selected_strategy_count"
            )
        ),
        "citation_query_sweep_blind_refuted_rate_below_min_strategy_count": _int(
            lanes["citation_evidence"].get(
                "query_sweep_blind_refuted_rate_below_min_strategy_count"
            )
        ),
        "citation_query_sweep_verified_false_alarm_above_max_strategy_count": _int(
            lanes["citation_evidence"].get(
                "query_sweep_verified_false_alarm_above_max_strategy_count"
            )
        ),
        "source_family_coverage_audit_count": _int(
            lanes["source_family_acquisition"].get("audit_count")
        ),
        "world_model_rule_task_count": _int(lanes["world_model_rules"].get("task_count")),
        "world_model_rule_remaining_task_count": _int(
            lanes["world_model_rules"].get("remaining_task_count")
        ),
        "world_model_rule_audit_adjusted_remaining_task_count": _int(
            lanes["world_model_rules"].get("audit_adjusted_remaining_task_count")
        ),
        "world_model_rule_audit_requeue_suggestion_count": _int(
            lanes["world_model_rules"].get("rule_input_audit_requeue_suggestion_count")
        ),
        "world_model_rule_requeue_outstanding_count": _int(
            lanes["world_model_rules"].get("rule_input_audit_requeue_outstanding_count")
        ),
        "world_model_rule_promoted_count": _int(
            lanes["world_model_rules"].get("promoted_count")
        ),
        "world_model_rule_input_fill_rollup_status": str(
            lanes["world_model_rules"].get("input_fill_result_rollup_status") or ""
        ),
        "world_model_rule_input_fill_adapter_ready": bool(
            lanes["world_model_rules"].get("input_fill_adapter_ready") is True
        ),
        "world_model_rule_combined_rule_input_count": _int(
            lanes["world_model_rules"].get("combined_rule_input_count")
        ),
        "world_model_rule_combined_unfilled_task_count": _int(
            lanes["world_model_rules"].get("combined_unfilled_task_count")
        ),
        "semantic_gap_review_workflow_count": _int(
            lanes["semantic_gap_review"].get("workflow_count")
        ),
        "semantic_gap_review_promoted_workflow_count": _int(
            lanes["semantic_gap_review"].get("promoted_workflow_count")
        ),
        "semantic_gap_review_approved_source_document_count": _int(
            lanes["semantic_gap_review"].get("approved_source_document_count")
        ),
        "semantic_gap_review_covered_fact_route_n_records": _int(
            lanes["semantic_gap_review"].get("covered_fact_route_n_records")
        ),
        "frontier_queue_execution_status": str(
            lanes["frontier_queue_execution"].get("status") or "unknown"
        ),
        "frontier_queue_execution_smoke_status": str(
            lanes["frontier_queue_execution"].get("control_plane_smoke_status")
            or "unknown"
        ),
        "frontier_queue_execution_smoke_count": _int(
            lanes["frontier_queue_execution"].get("frontier_queue_execution_smoke_count")
        ),
        "frontier_queue_execution_smoke_passed_count": _int(
            lanes["frontier_queue_execution"].get(
                "frontier_queue_execution_smoke_passed_count"
            )
        ),
        "frontier_queue_execution_smoke_manifest_verified_count": _int(
            lanes["frontier_queue_execution"].get(
                "frontier_queue_execution_smoke_manifest_verified_count"
            )
        ),
        "frontier_command_binding_review_count": _int(
            lanes["frontier_queue_execution"].get("command_binding_review_count")
        ),
        "frontier_bound_command_run_count": _int(
            lanes["frontier_queue_execution"].get("bound_command_run_count")
        ),
        "frontier_bound_command_executed_count": _int(
            lanes["frontier_queue_execution"].get("executed_count")
        ),
        "frontier_bound_command_succeeded_count": _int(
            lanes["frontier_queue_execution"].get("succeeded_count")
        ),
        "frontier_bound_command_dry_run_count": _int(
            lanes["frontier_queue_execution"].get("dry_run_count")
        ),
        "frontier_bound_command_missing_output_count": _int(
            lanes["frontier_queue_execution"].get("missing_output_count")
        ),
        "mechanism_handoff_trace_count": _int(
            lanes["world_model_rules"].get("mechanism_handoff_trace_count")
        ),
        "lane_statuses": lane_statuses,
        "blocked_lane_count": sum(
            1
            for status in lane_statuses.values()
            if status in {
                "blocked",
                "needs_execution",
                "needs_inputs",
                "needs_requeue",
                "needs_review",
                "partial",
                "ready_for_adapter",
            }
        ),
        "missing_lane_count": sum(1 for status in lane_statuses.values() if status == "missing"),
        "covered_or_promoted_lane_count": sum(
            1 for status in lane_statuses.values() if status in {"covered", "promote"}
        ),
    }


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path | None,
    unresolved_queue_path: str | Path | None,
    citation_workflow_paths: Sequence[str | Path],
    source_family_coverage_audit_paths: Sequence[str | Path],
    semantic_gap_review_workflow_paths: Sequence[str | Path],
    frontier_command_binding_review_paths: Sequence[str | Path],
    frontier_bound_command_run_paths: Sequence[str | Path],
    frontier_queue_execution_smoke_paths: Sequence[str | Path],
    frontier_queue_execution_smoke_manifest_paths: Sequence[str | Path],
    rule_input_plan_path: str | Path | None,
    rule_input_audit_report_path: str | Path | None,
    rule_stub_requeue_report_path: str | Path | None,
    requeued_rule_input_plan_path: str | Path | None,
    input_fill_result_rollup_path: str | Path | None,
    rule_promotion_report_paths: Sequence[str | Path],
    mechanism_handoff_bundle_path: str | Path | None,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> Mapping[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "unresolved_frontier_evidence_summary": output_path,
        "unresolved_queue": unresolved_queue_path,
        "rule_input_plan": rule_input_plan_path,
        "rule_input_audit_report": rule_input_audit_report_path,
        "rule_stub_requeue_report": rule_stub_requeue_report_path,
        "requeued_rule_input_plan": requeued_rule_input_plan_path,
        "input_fill_result_rollup": input_fill_result_rollup_path,
        "mechanism_handoff_bundle": mechanism_handoff_bundle_path,
    }
    artifacts.update(
        {f"citation_workflow_{idx}": path for idx, path in enumerate(citation_workflow_paths, start=1)}
    )
    artifacts.update({
        f"source_family_coverage_audit_{idx}": path
        for idx, path in enumerate(source_family_coverage_audit_paths, start=1)
    })
    artifacts.update({
        f"semantic_gap_review_workflow_{idx}": path
        for idx, path in enumerate(semantic_gap_review_workflow_paths, start=1)
    })
    artifacts.update({
        f"frontier_command_binding_review_{idx}": path
        for idx, path in enumerate(frontier_command_binding_review_paths, start=1)
    })
    artifacts.update({
        f"frontier_bound_command_run_{idx}": path
        for idx, path in enumerate(frontier_bound_command_run_paths, start=1)
    })
    artifacts.update({
        f"frontier_queue_execution_smoke_{idx}": path
        for idx, path in enumerate(frontier_queue_execution_smoke_paths, start=1)
    })
    artifacts.update({
        f"frontier_queue_execution_smoke_manifest_{idx}": path
        for idx, path in enumerate(
            frontier_queue_execution_smoke_manifest_paths,
            start=1,
        )
    })
    artifacts.update({
        f"rule_promotion_report_{idx}": path
        for idx, path in enumerate(rule_promotion_report_paths, start=1)
    })
    manifest = build_artifact_manifest(
        {name: Path(path) for name, path in artifacts.items() if path is not None},
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "unresolved_target_count": _nested(payload, "summary", "unresolved_target_count"),
            "citation_status": _nested(payload, "lanes", "citation_evidence", "status"),
            "citation_query_sweep_failure_reason_counts": _nested(
                payload,
                "lanes",
                "citation_evidence",
                "query_sweep_failure_reason_counts",
            ),
            "citation_query_sweep_recommended_next_action_counts": _nested(
                payload,
                "lanes",
                "citation_evidence",
                "query_sweep_recommended_next_action_counts",
            ),
            "citation_query_sweep_no_hit_strategy_count": _nested(
                payload,
                "lanes",
                "citation_evidence",
                "query_sweep_no_hit_strategy_count",
            ),
            "citation_query_sweep_target_route_not_selected_strategy_count": _nested(
                payload,
                "lanes",
                "citation_evidence",
                "query_sweep_target_route_not_selected_strategy_count",
            ),
            "citation_query_sweep_blind_refuted_rate_below_min_strategy_count": _nested(
                payload,
                "lanes",
                "citation_evidence",
                "query_sweep_blind_refuted_rate_below_min_strategy_count",
            ),
            "citation_query_sweep_verified_false_alarm_above_max_strategy_count": _nested(
                payload,
                "lanes",
                "citation_evidence",
                "query_sweep_verified_false_alarm_above_max_strategy_count",
            ),
            "source_family_acquisition_status": _nested(
                payload, "lanes", "source_family_acquisition", "status"
            ),
            "world_model_rule_status": _nested(payload, "lanes", "world_model_rules", "status"),
            "semantic_gap_review_status": _nested(
                payload, "lanes", "semantic_gap_review", "status"
            ),
            "semantic_gap_review_workflow_count": _nested(
                payload, "lanes", "semantic_gap_review", "workflow_count"
            ),
            "semantic_gap_review_promoted_workflow_count": _nested(
                payload, "lanes", "semantic_gap_review", "promoted_workflow_count"
            ),
            "semantic_gap_review_approved_source_document_count": _nested(
                payload, "lanes", "semantic_gap_review", "approved_source_document_count"
            ),
            "frontier_queue_execution_status": _nested(
                payload, "lanes", "frontier_queue_execution", "status"
            ),
            "frontier_command_binding_review_count": _nested(
                payload,
                "lanes",
                "frontier_queue_execution",
                "command_binding_review_count",
            ),
            "frontier_bound_command_run_count": _nested(
                payload,
                "lanes",
                "frontier_queue_execution",
                "bound_command_run_count",
            ),
            "frontier_bound_command_succeeded_count": _nested(
                payload,
                "lanes",
                "frontier_queue_execution",
                "succeeded_count",
            ),
            "frontier_queue_execution_smoke_status": _nested(
                payload,
                "lanes",
                "frontier_queue_execution",
                "control_plane_smoke_status",
            ),
            "frontier_queue_execution_smoke_count": _nested(
                payload,
                "lanes",
                "frontier_queue_execution",
                "frontier_queue_execution_smoke_count",
            ),
            "frontier_queue_execution_smoke_manifest_verified_count": _nested(
                payload,
                "lanes",
                "frontier_queue_execution",
                "frontier_queue_execution_smoke_manifest_verified_count",
            ),
            "world_model_rule_remaining_task_count": _nested(
                payload,
                "lanes",
                "world_model_rules",
                "remaining_task_count",
            ),
            "world_model_rule_audit_adjusted_remaining_task_count": _nested(
                payload,
                "lanes",
                "world_model_rules",
                "audit_adjusted_remaining_task_count",
            ),
            "world_model_rule_audit_requeue_suggestion_count": _nested(
                payload,
                "lanes",
                "world_model_rules",
                "rule_input_audit_requeue_suggestion_count",
            ),
            "world_model_rule_requeue_outstanding_count": _nested(
                payload,
                "lanes",
                "world_model_rules",
                "rule_input_audit_requeue_outstanding_count",
            ),
            "world_model_rule_input_fill_rollup_status": _nested(
                payload,
                "lanes",
                "world_model_rules",
                "input_fill_result_rollup_status",
            ),
            "world_model_rule_input_fill_adapter_ready": _nested(
                payload,
                "lanes",
                "world_model_rules",
                "input_fill_adapter_ready",
            ),
            "world_model_rule_combined_rule_input_count": _nested(
                payload,
                "lanes",
                "world_model_rules",
                "combined_rule_input_count",
            ),
            "world_model_rule_combined_unfilled_task_count": _nested(
                payload,
                "lanes",
                "world_model_rules",
                "combined_unfilled_task_count",
            ),
            "next_action_count": len(_sequence(payload.get("next_actions"))),
            **dict(metadata),
        },
    )
    _write_json(manifest_path, manifest, compact=compact)
    return manifest


def _load_optional_mapping(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _load_mapping(path)


def _load_mapping(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _load_frontier_queue_execution_smoke(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    payload = _load_mapping(report_path)
    manifest_path = _resolve_report_path(
        _nested(payload, "paths", "artifact_manifest"),
        base_path=report_path,
    )
    if manifest_path is None:
        payload["manifest_verification"] = {
            "manifest_path": None,
            "passed": False,
            "checked": 0,
            "failures": [{
                "name": "artifact_manifest",
                "path": "",
                "field": "path",
                "expected": "present",
                "actual": "missing",
            }],
            "nested": [],
        }
        return payload
    try:
        verification = load_and_verify_artifact_manifest(manifest_path, recursive=True)
    except (OSError, ValueError) as exc:
        payload["manifest_verification"] = {
            "manifest_path": str(manifest_path),
            "passed": False,
            "checked": 0,
            "failures": [{
                "name": "artifact_manifest",
                "path": str(manifest_path),
                "field": "load_error",
                "expected": "loadable manifest",
                "actual": str(exc),
            }],
            "nested": [],
        }
        return payload
    payload["manifest_verification"] = verification.to_dict()
    return payload


def _resolve_report_path(value: Any, *, base_path: Path) -> Path | None:
    if value is None:
        return None
    candidate = Path(str(value))
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return base_path.parent / candidate


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return (str(value),) if str(value) else ()
    return tuple(str(item) for item in value if str(item))


def _int_mapping(value: Any) -> dict[str, int]:
    return {
        str(key): _int(item)
        for key, item in _mapping(value).items()
        if str(key)
    }


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _max_optional_int(current: int | None, candidate: int) -> int:
    return candidate if current is None else max(current, candidate)


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _rule_family_closure_counts(
    rule_family_counts: Mapping[str, int],
    promoted_rule_families: Mapping[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    closed: dict[str, int] = {}
    remaining: dict[str, int] = {}
    for family, count in rule_family_counts.items():
        closed_count = min(_int(count), _int(promoted_rule_families.get(family)))
        remaining_count = max(_int(count) - closed_count, 0)
        if closed_count:
            closed[family] = closed_count
        if remaining_count:
            remaining[family] = remaining_count
    for family, count in promoted_rule_families.items():
        if family not in rule_family_counts and _int(count):
            closed[family] = _int(count)
    return closed, remaining


def _remaining_missing_input_counts(
    missing_input_counts: Mapping[str, int],
    remaining_rule_family_counts: Mapping[str, int],
) -> dict[str, int]:
    if not remaining_rule_family_counts:
        return {}
    remaining_fields = set()
    for family, count in remaining_rule_family_counts.items():
        if _int(count):
            remaining_fields.update(RULE_FAMILY_INPUT_FIELDS.get(family, ()))
    if not remaining_fields:
        return dict(missing_input_counts)
    return {
        field: count
        for field, count in missing_input_counts.items()
        if field in remaining_fields
    }


def _audit_adjusted_remaining_rule_family_counts(
    remaining_rule_family_counts: Mapping[str, int],
    *,
    requeue_suggestions: Sequence[Mapping[str, Any]],
    rule_family_counts: Mapping[str, int],
    promoted_rule_families: Mapping[str, int],
) -> dict[str, int]:
    adjusted = Counter({
        family: _int(count)
        for family, count in remaining_rule_family_counts.items()
        if _int(count) > 0
    })
    for suggestion in requeue_suggestions:
        current = str(suggestion.get("current_rule_family") or "")
        recommended = str(suggestion.get("recommended_rule_family") or "")
        if not current or not recommended:
            continue
        if adjusted.get(current, 0) > 0:
            adjusted[current] -= 1
        adjusted[recommended] += 1
    for family, promoted_count in promoted_rule_families.items():
        extra_promoted = max(_int(promoted_count) - _int(rule_family_counts.get(family)), 0)
        if extra_promoted and adjusted.get(family, 0) > 0:
            adjusted[family] = max(adjusted[family] - extra_promoted, 0)
    return {
        family: count
        for family, count in sorted(adjusted.items())
        if count > 0
    }


def _required_input_counts_by_rule_family(
    rule_family_counts: Mapping[str, int],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for family, count in rule_family_counts.items():
        for field in RULE_FAMILY_INPUT_FIELDS.get(family, ()):
            counts[field] += _int(count)
    return dict(sorted((field, count) for field, count in counts.items() if count > 0))


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata must be KEY=VALUE, got {value!r}.")
        key, item = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("metadata key cannot be empty.")
        metadata[key] = item
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unresolved-queue", default=None)
    parser.add_argument("--citation-workflow", action="append", default=[])
    parser.add_argument("--source-family-coverage-audit", action="append", default=[])
    parser.add_argument("--semantic-gap-review-workflow", action="append", default=[])
    parser.add_argument("--frontier-command-binding-review", action="append", default=[])
    parser.add_argument("--frontier-bound-command-run", action="append", default=[])
    parser.add_argument("--frontier-queue-execution-smoke", action="append", default=[])
    parser.add_argument("--rule-input-plan", default=None)
    parser.add_argument("--rule-input-audit-report", default=None)
    parser.add_argument("--rule-stub-requeue-report", default=None)
    parser.add_argument("--requeued-rule-input-plan", default=None)
    parser.add_argument("--input-fill-result-rollup", default=None)
    parser.add_argument("--rule-promotion-report", action="append", default=[])
    parser.add_argument("--mechanism-handoff-bundle", default=None)
    parser.add_argument("--json", default=None, help="optional output JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional local artifact registry JSON")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--metadata", action="append", default=[], help="extra metadata as KEY=VALUE")
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run(
        unresolved_queue_path=args.unresolved_queue,
        citation_workflow_paths=tuple(args.citation_workflow or ()),
        source_family_coverage_audit_paths=tuple(args.source_family_coverage_audit or ()),
        semantic_gap_review_workflow_paths=tuple(args.semantic_gap_review_workflow or ()),
        frontier_command_binding_review_paths=tuple(
            args.frontier_command_binding_review or ()
        ),
        frontier_bound_command_run_paths=tuple(args.frontier_bound_command_run or ()),
        frontier_queue_execution_smoke_paths=tuple(
            args.frontier_queue_execution_smoke or ()
        ),
        rule_input_plan_path=args.rule_input_plan,
        rule_input_audit_report_path=args.rule_input_audit_report,
        rule_stub_requeue_report_path=args.rule_stub_requeue_report,
        requeued_rule_input_plan_path=args.requeued_rule_input_plan,
        input_fill_result_rollup_path=args.input_fill_result_rollup,
        rule_promotion_report_paths=tuple(args.rule_promotion_report or ()),
        mechanism_handoff_bundle_path=args.mechanism_handoff_bundle,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    if args.json is None:
        print(strict_json_dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
