"""Shared CLI requirement checks for frontier research command handoffs."""

from __future__ import annotations

import shlex
from typing import Any, Mapping, Sequence

REQUIRED_COMMAND_FLAGS = {
    "benchmarks/compare_frontier_release_evidence.py": (
        "--verifier-stability-report",
        "--abstention-stability-report",
        "--json",
    ),
    "benchmarks/compare_product_runtime_baselines.py": ("--current", "--json"),
    "benchmarks/eval_abstention_stability.py": ("--scores", "--signals", "--json"),
    "benchmarks/export_product_promotion_contract_evidence_handoff.py": (
        "--contract",
        "--json",
        "--audit-json",
    ),
    "benchmarks/bind_frontier_research_queue_artifact_inputs.py": (
        "--input-collection-plan",
        "--base-bindings",
        "--output-dir",
        "--json",
        "--bindings-json",
        "--artifact-manifest",
    ),
    "benchmarks/bind_frontier_research_queue_command_plan.py": (
        "--command-plan",
        "--bindings",
        "--json",
        "--artifact-manifest",
    ),
    "benchmarks/plan_frontier_abstention_evidence_reruns.py": ("--source",),
    "benchmarks/run_product_runtime_baseline.py": ("--trace", "--json"),
    "benchmarks/review_frontier_research_queue_command_bindings.py": (
        "--bound-command-plan",
        "--base-bindings",
        "--output-dir",
        "--json",
        "--approved-bindings",
        "--artifact-manifest",
    ),
    "benchmarks/run_frontier_research_queue_bound_command_plan.py": (
        "--bound-command-plan",
        "--json",
        "--artifact-manifest",
    ),
    "benchmarks/run_source_family_citation_search_workflow.py": (
        "--queue",
        "--source-catalog",
        "--scores",
        "--blind-spots",
        "--output-dir",
        "--workflow-report",
        "--artifact-manifest",
    ),
    "benchmarks/audit_source_family_coverage.py": (
        "--requests",
        "--adapter-results",
        "--json",
        "--acquisition-plan-jsonl",
        "--artifact-manifest",
    ),
    "benchmarks/plan_source_family_catalog_collection.py": (
        "--acquisition-plan",
        "--tasks-jsonl",
        "--report-json",
        "--artifact-manifest",
    ),
    "benchmarks/build_citation_binding_source_family_tasks.py": (
        "--collection-plan",
        "--tasks-jsonl",
        "--report-json",
        "--artifact-manifest",
    ),
    "benchmarks/build_source_family_structured_qa_lane_execution_queue.py": (
        "--triage",
        "--collection-corpus",
        "--output-dir",
        "--report-json",
        "--request-jsonl",
        "--batch-jsonl",
        "--artifact-manifest",
    ),
    "benchmarks/plan_source_family_structured_qa_lane_reruns.py": (
        "--lane-queue",
        "--collection-corpus",
        "--json",
        "--artifact-manifest",
    ),
    "benchmarks/run_source_family_structured_qa_lane_batch_workflow.py": (
        "--lane-queue",
        "--collection-corpus",
        "--batch-id",
        "--output-dir",
        "--json",
        "--artifact-manifest",
    ),
    "benchmarks/run_crossref_source_family_catalog_adapter.py": (
        "--tasks",
        "--output",
        "--report-json",
        "--artifact-manifest",
    ),
    "benchmarks/run_openalex_source_family_catalog_adapter.py": (
        "--tasks",
        "--output",
        "--report-json",
        "--artifact-manifest",
    ),
    "benchmarks/run_worldbank_source_family_catalog_adapter.py": (
        "--tasks",
        "--output",
        "--report-json",
        "--artifact-manifest",
    ),
    "benchmarks/run_gdelt_source_family_catalog_adapter.py": (
        "--tasks",
        "--output",
        "--report-json",
        "--artifact-manifest",
    ),
    "benchmarks/run_seeded_url_source_family_catalog_adapter.py": (
        "--tasks",
        "--seeds",
        "--output",
        "--report-json",
        "--artifact-manifest",
    ),
    "benchmarks/run_official_site_source_family_catalog_adapter.py": (
        "--tasks",
        "--seeds",
        "--output",
        "--report-json",
        "--artifact-manifest",
    ),
    "benchmarks/run_retrieval_semantic_gap_review_workflow.py": (
        "--verified-records-jsonl",
        "--output-dir",
        "--workflow-report",
        "--artifact-manifest",
    ),
    "benchmarks/fill_world_model_rule_inputs_from_numeric_bindings.py": (
        "--input-tasks",
        "--numeric-bindings",
        "--output-dir",
        "--json",
    ),
    "benchmarks/fill_world_model_rule_inputs_from_temporal_bindings.py": (
        "--input-tasks",
        "--temporal-bindings",
        "--output-dir",
        "--json",
    ),
    "benchmarks/plan_world_model_rule_entity_bindings.py": (
        "--entity-bindings",
        "--output-dir",
    ),
    "benchmarks/collect_world_model_rule_entity_bindings_from_citation_corpus.py": (
        "--entity-binding-plan",
        "--citation-corpus",
        "--output-dir",
    ),
    "benchmarks/build_world_model_rule_entity_binding_citation_handoff.py": (
        "--entity-binding-plan",
        "--output-dir",
    ),
    "benchmarks/review_world_model_rule_entity_binding_candidates.py": (
        "--entity-binding-plan",
        "--output-dir",
    ),
    "benchmarks/promote_world_model_rule_entity_binding_candidates.py": (
        "--entity-binding-plan",
        "--output-dir",
    ),
    "benchmarks/fill_world_model_rule_inputs_from_entity_bindings.py": (
        "--input-tasks",
        "--entity-bindings",
        "--output-dir",
        "--json",
    ),
    "benchmarks/run_world_model_rule_authoring_adapter.py": (
        "--rule-stubs",
        "--rule-inputs",
        "--output-dir",
        "--json",
    ),
    "benchmarks/run_frontier_research_queue_rule_adapter_promotion_workflow.py": (
        "--input-fill-result-rollup",
        "--output-dir",
        "--json",
        "--artifact-manifest",
    ),
    "benchmarks/promote_world_model_rule_candidates.py": (
        "--rule-results",
        "--rule-inputs",
        "--adapter-report",
        "--output-dir",
        "--json",
    ),
    "benchmarks/rollup_frontier_abstention_evidence_reruns.py": ("--queue", "--json"),
}

REQUIRED_INPUT_FLAGS = {
    "benchmarks/compare_frontier_release_evidence.py": {
        "verifier_stability_report": "--verifier-stability-report",
        "abstention_stability_report": "--abstention-stability-report",
    },
    "benchmarks/compare_product_runtime_baselines.py": {
        "baseline_product_runtime_report": "--baseline",
    },
    "benchmarks/export_product_promotion_contract_evidence_handoff.py": {
        "product_promotion_contract_source": "--contract",
    },
    "benchmarks/plan_frontier_abstention_evidence_reruns.py": {
        "frontier_release_report_or_evidence_gap_plan": "--source",
        "abstention_score_dump_paths": "--scores",
        "abstention_signal_groups": "--signal-groups",
    },
    "benchmarks/run_product_runtime_baseline.py": {
        "product_trace_corpus": "--trace",
        "product_promotion_contract_source": "--promotion-contract",
    },
    "benchmarks/bind_frontier_research_queue_command_plan.py": {
        "frontier_command_plan": "--command-plan",
        "approved_frontier_command_bindings": "--bindings",
    },
    "benchmarks/review_frontier_research_queue_command_bindings.py": {
        "frontier_bound_command_plan": "--bound-command-plan",
        "frontier_command_bindings": "--base-bindings",
        "frontier_command_review_decisions": "--review-decisions",
    },
    "benchmarks/run_frontier_research_queue_bound_command_plan.py": {
        "reviewed_frontier_bound_command_plan": "--bound-command-plan",
    },
    "benchmarks/run_retrieval_semantic_gap_review_workflow.py": {
        "source_bound_verified_records_jsonl": "--verified-records-jsonl",
        "detectability_blind_spot_record_indices_json": "--record-indices-json",
    },
    "benchmarks/audit_source_family_coverage.py": {
        "source_family_citation_search_requests": "--requests",
        "source_family_citation_search_adapter_results": "--adapter-results",
    },
    "benchmarks/plan_source_family_catalog_collection.py": {
        "source_family_acquisition_plan": "--acquisition-plan",
    },
    "benchmarks/build_citation_binding_source_family_tasks.py": {
        "citation_binding_evidence_collection_plan": "--collection-plan",
    },
    "benchmarks/build_source_family_structured_qa_lane_execution_queue.py": {
        "source_family_structured_qa_gap_triage": "--triage",
        "source_family_structured_qa_fact_collection_corpus": "--collection-corpus",
    },
    "benchmarks/plan_source_family_structured_qa_lane_reruns.py": {
        "source_family_structured_qa_lane_execution_queue": "--lane-queue",
        "source_family_structured_qa_fact_collection_corpus": "--collection-corpus",
        "source_family_source_catalog": "--source-catalog",
    },
    "benchmarks/run_source_family_structured_qa_lane_batch_workflow.py": {
        "source_family_structured_qa_lane_execution_queue": "--lane-queue",
        "source_family_structured_qa_fact_collection_corpus": "--collection-corpus",
        "source_family_source_catalog": "--source-catalog",
    },
    "benchmarks/run_crossref_source_family_catalog_adapter.py": {
        "source_family_collection_tasks": "--tasks",
    },
    "benchmarks/run_openalex_source_family_catalog_adapter.py": {
        "source_family_collection_tasks": "--tasks",
    },
    "benchmarks/run_worldbank_source_family_catalog_adapter.py": {
        "source_family_collection_tasks": "--tasks",
    },
    "benchmarks/run_gdelt_source_family_catalog_adapter.py": {
        "source_family_collection_tasks": "--tasks",
    },
    "benchmarks/run_seeded_url_source_family_catalog_adapter.py": {
        "source_family_collection_tasks": "--tasks",
        "source_family_url_seeds": "--seeds",
    },
    "benchmarks/run_official_site_source_family_catalog_adapter.py": {
        "source_family_collection_tasks": "--tasks",
        "source_family_url_seeds": "--seeds",
    },
    "benchmarks/fill_world_model_rule_inputs_from_numeric_bindings.py": {
        "source_backed_numeric_bindings": "--numeric-bindings",
        "source_backed_subject_bindings": "--subject-bindings",
    },
    "benchmarks/fill_world_model_rule_inputs_from_temporal_bindings.py": {
        "source_backed_temporal_bindings": "--temporal-bindings",
    },
    "benchmarks/plan_world_model_rule_entity_bindings.py": {
        "source_backed_entity_bindings": "--entity-bindings",
    },
    "benchmarks/fill_world_model_rule_inputs_from_entity_bindings.py": {
        "source_backed_entity_bindings": "--entity-bindings",
    },
    "benchmarks/run_frontier_research_queue_rule_adapter_promotion_workflow.py": {
        "world_model_rule_input_fill_result_rollup": "--input-fill-result-rollup",
    },
}


def frontier_command_requirement_summary(
    command: str,
    *,
    index: int,
    required_inputs: Sequence[str] = (),
) -> dict[str, Any]:
    """Return known CLI requirements for a frontier command template or command."""
    try:
        argv = tuple(shlex.split(command))
    except ValueError as exc:
        return {
            "command_index": index,
            "script": None,
            "known_command": False,
            "parse_error": str(exc),
            "required_flags": (),
            "missing_required_flags": (),
            "required_input_flags": (),
            "missing_required_input_flags": (),
            "status": "invalid_command",
        }
    script = frontier_command_script(argv)
    required_flags = REQUIRED_COMMAND_FLAGS.get(script or "", ())
    required_input_flags = _required_input_flags(
        script,
        required_inputs=required_inputs,
    )
    missing_required_flags = tuple(flag for flag in required_flags if flag not in argv)
    missing_required_input_flags = tuple(
        item for item in required_input_flags if item["flag"] not in argv
    )
    known_command = bool(required_flags or required_input_flags)
    if missing_required_flags or missing_required_input_flags:
        status = "needs_review"
    elif known_command:
        status = "ready"
    else:
        status = "unknown"
    return {
        "command_index": index,
        "script": script,
        "known_command": known_command,
        "required_flags": required_flags,
        "missing_required_flags": missing_required_flags,
        "required_input_flags": required_input_flags,
        "missing_required_input_flags": missing_required_input_flags,
        "status": status,
    }


def frontier_command_validation_issues(
    command: str,
    *,
    index: int,
    required_inputs: Sequence[str] = (),
    ignore_placeholders: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Return fail-closed validation issues for a concrete bound command."""
    if ignore_placeholders and "..." in command:
        return ()
    summary = frontier_command_requirement_summary(
        command,
        index=index,
        required_inputs=required_inputs,
    )
    script = summary.get("script")
    if summary.get("parse_error") or script is None:
        return ()
    issues = []
    missing_required_flags = _string_tuple(summary.get("missing_required_flags"))
    if missing_required_flags:
        issues.append({
            "command_index": index,
            "script": script,
            "issue": "missing_required_cli_flags",
            "missing_flags": missing_required_flags,
        })
    missing_required_input_flags = tuple(
        item
        for item in _mapping_sequence(summary.get("missing_required_input_flags"))
        if item.get("flag")
    )
    if missing_required_input_flags:
        issues.append({
            "command_index": index,
            "script": script,
            "issue": "required_input_not_bound_to_command_flags",
            "required_inputs": tuple(str(item["input"]) for item in missing_required_input_flags),
            "missing_flags": tuple(str(item["flag"]) for item in missing_required_input_flags),
        })
    return tuple(issues)


def validate_frontier_bound_commands(
    commands: Sequence[str],
    *,
    required_inputs: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate concrete bound frontier commands against known script requirements."""
    issues = []
    for index, command in enumerate(commands, start=1):
        issues.extend(
            frontier_command_validation_issues(
                command,
                index=index,
                required_inputs=required_inputs,
            )
        )
    return {
        "issue_count": len(issues),
        "issues": tuple(issues),
    }


def frontier_command_script(argv: Sequence[str]) -> str | None:
    """Return a normalized benchmark script path from a command argv."""
    for item in argv:
        text = str(item)
        if text.endswith(".py"):
            return _normalize_script_path(text)
    return None


def _required_input_flags(
    script: str | None,
    *,
    required_inputs: Sequence[str],
) -> tuple[Mapping[str, str], ...]:
    if not script or not required_inputs:
        return ()
    configured = REQUIRED_INPUT_FLAGS.get(script, {})
    required = {str(item) for item in required_inputs if str(item)}
    return tuple(
        {"input": input_name, "flag": flag}
        for input_name, flag in configured.items()
        if input_name in required
    )


def _normalize_script_path(path: str) -> str:
    text = str(path).replace("\\", "/")
    if "/benchmarks/" in text:
        return "benchmarks/" + text.rsplit("/benchmarks/", 1)[1]
    while text.startswith("./"):
        text = text[2:]
    return text


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return (str(value),) if str(value) else ()
    return tuple(str(item) for item in value if str(item))
