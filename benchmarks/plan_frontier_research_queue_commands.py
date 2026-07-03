"""Build a command plan from a frontier status research queue."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "frontier_research_queue_command_plan"
SUPPORTED_SOURCE_WORKFLOWS = frozenset({
    "citation_binding_evidence_collection_plan",
    "frontier_status_report",
    "evidence_gap_plan",
    "source_family_catalog_collection_plan",
    "source_family_structured_qa_lane_rerun_queue",
    "unresolved_frontier_evidence_summary",
})
UNRESOLVED_SUMMARY_WORKFLOW = "unresolved_frontier_evidence_summary"
CITATION_BINDING_EVIDENCE_COLLECTION_WORKFLOW = "citation_binding_evidence_collection_plan"
SOURCE_FAMILY_CATALOG_COLLECTION_WORKFLOW = "source_family_catalog_collection_plan"
SOURCE_FAMILY_STRUCTURED_QA_LANE_RERUN_WORKFLOW = (
    "source_family_structured_qa_lane_rerun_queue"
)


def build_frontier_research_queue_command_plan(
    *,
    source: str | Path | Mapping[str, Any],
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    output_dir: str | Path | None = None,
    include_action_ids: Sequence[str] = (),
    exclude_action_ids: Sequence[str] = (),
    only_active_research_queue: bool = False,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn research-queue actions into a reviewable command plan.

    This is a planning artifact only. It does not execute commands, bind
    placeholders, or create verifier/release evidence.
    """
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    source_path, source_payload = _load_mapping_source(source)
    workflow = source_payload.get("workflow")
    if workflow not in SUPPORTED_SOURCE_WORKFLOWS:
        raise ValueError(
            "source must have workflow 'frontier_status_report', 'evidence_gap_plan', "
            "'citation_binding_evidence_collection_plan', "
            "'source_family_catalog_collection_plan', "
            "'source_family_structured_qa_lane_rerun_queue', or "
            "'unresolved_frontier_evidence_summary'."
        )

    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    plan_root = _plan_root(source_path=source_path, output_path=output_path, output_dir=output_dir)
    source_lifecycle_status = _nested(source_payload, "research_queue", "lifecycle_status")
    source_alignment_status = _nested(source_payload, "research_queue", "source_alignment", "status")
    actions = ()
    if not only_active_research_queue or _is_active_research_queue(source_payload):
        actions = _filtered_actions(
            _source_actions(source_payload, source_path=source_path),
            include_action_ids=include_action_ids,
            exclude_action_ids=exclude_action_ids,
        )
    entries = tuple(
        _command_entry(action, index=index, plan_root=plan_root)
        for index, action in enumerate(actions, start=1)
    )
    summary = _summary(entries)
    status = _status(summary)
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "source": {
            "path": None if source_path is None else str(source_path),
            "workflow": workflow,
            "status": source_payload.get("status"),
            "research_refresh_status": _nested(
                source_payload, "research_queue", "refresh_status"
            ),
            "research_lifecycle_status": source_lifecycle_status,
            "research_source_alignment_status": source_alignment_status,
        },
        "summary": summary,
        "paths": {
            "command_plan": None if output_path is None else str(output_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "output_dir": str(plan_root),
        },
        "config": {
            "executes_commands": False,
            "include_action_ids": tuple(str(item) for item in include_action_ids if str(item)),
            "exclude_action_ids": tuple(str(item) for item in exclude_action_ids if str(item)),
            "only_active_research_queue": bool(only_active_research_queue),
        },
        "entries": entries,
        "metadata": dict(metadata or {}),
    }
    if output_path is not None:
        _write_json(output_path, payload, compact=compact_json)
    manifest = None
    if manifest_path is not None:
        manifest = _write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            source_path=source_path,
            payload=payload,
            metadata=metadata or {},
            compact=compact_json,
        )
    if registry_path is not None and output_path is None and source_path is None:
        raise ValueError("registry_path requires json_path when source is an in-memory payload.")
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output_path if output_path is not None else source_path,
            metadata={
                "workflow": WORKFLOW,
                "status": status,
                "source_workflow": workflow,
                "source_path": None if source_path is None else str(source_path),
                "source_research_lifecycle_status": source_lifecycle_status,
                "source_research_alignment_status": source_alignment_status,
                "only_active_research_queue": bool(only_active_research_queue),
                "entry_count": summary["entry_count"],
                "ready_entry_count": summary["ready_entry_count"],
                "needs_input_entry_count": summary["needs_input_entry_count"],
                "missing_command_template_count": summary["missing_command_template_count"],
                "command_count": summary["command_count"],
                "placeholder_count": summary["placeholder_count"],
                "missing_input_count": summary["missing_input_count"],
                "manifest_summary": {} if manifest is None else manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _source_actions(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None,
) -> tuple[Mapping[str, Any], ...]:
    if payload.get("workflow") == "frontier_status_report":
        return tuple(_mapping_sequence(_nested(payload, "research_queue", "actions")))
    if payload.get("workflow") == CITATION_BINDING_EVIDENCE_COLLECTION_WORKFLOW:
        return (_citation_binding_collection_action(payload, source_path=source_path),)
    if payload.get("workflow") == SOURCE_FAMILY_CATALOG_COLLECTION_WORKFLOW:
        return (_source_family_catalog_adapter_action(payload, source_path=source_path),)
    if payload.get("workflow") == SOURCE_FAMILY_STRUCTURED_QA_LANE_RERUN_WORKFLOW:
        return (_source_family_structured_qa_lane_rerun_action(payload),)
    if payload.get("workflow") == UNRESOLVED_SUMMARY_WORKFLOW:
        return _unresolved_summary_actions(payload, source_path=source_path)
    return tuple(_mapping_sequence(payload.get("actions", ())))


def _citation_binding_collection_action(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None,
) -> Mapping[str, Any]:
    family_counts = _citation_binding_supported_family_counts(
        payload,
        source_path=source_path,
    )
    command_templates = [
        _citation_binding_source_family_task_command(source_path),
        *_source_family_catalog_adapter_commands(
            "...",
            family_counts=family_counts,
            metadata_source=CITATION_BINDING_EVIDENCE_COLLECTION_WORKFLOW,
        ),
    ]
    required_inputs = (
        ("citation_binding_evidence_collection_plan",)
        if source_path is None
        else ()
    )
    summary = _mapping(payload.get("summary"))
    return {
        "action_id": "run_citation_binding_source_family_collection",
        "title": "Build citation-binding source-family collection tasks",
        "action_type": "workflow_plan",
        "priority": 82,
        "evidence_routes": ("source_family_acquisition", "citation_evidence"),
        "suggested_commands": tuple(command_templates),
        "metadata": {
            "required_inputs": required_inputs,
            "closure_outputs": (
                "citation_binding_source_family_collection_plan",
                *tuple(
                    f"{adapter}_source_family_catalog_report"
                    for adapter in _source_family_adapter_names(family_counts)
                ),
            ),
            "source_collection_workflow": CITATION_BINDING_EVIDENCE_COLLECTION_WORKFLOW,
            "collection_request_count": _int_or_zero(summary.get("collection_request_count")),
            "collection_task_count": sum(family_counts.values()),
            "task_source_family_counts": family_counts,
            "lane_counts": dict(_mapping(summary.get("lane_counts"))),
            "priority_counts": dict(_mapping(summary.get("priority_counts"))),
            "preferred_source_family_counts": dict(
                _mapping(summary.get("preferred_source_family_counts"))
            ),
            "reason": (
                "citation binding collection plan is ready; bridge source-backed "
                "lanes into source-family catalog tasks before rerunning citation "
                "binding gates"
            ),
        },
    }


def _citation_binding_source_family_task_command(source_path: Path | None) -> str:
    return _shell_join((
        "python",
        "benchmarks/build_citation_binding_source_family_tasks.py",
        "--collection-plan",
        "..." if source_path is None else str(source_path),
        "--report-json",
        "...",
        "--tasks-jsonl",
        "...",
        "--artifact-manifest",
        "...",
        "--registry",
        "...",
        "--name",
        "...",
        "--version",
        "...",
        "--metadata",
        f"source={CITATION_BINDING_EVIDENCE_COLLECTION_WORKFLOW}",
    ))


def _citation_binding_supported_family_counts(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None,
) -> dict[str, int]:
    requests = _citation_binding_collection_requests(payload, source_path=source_path)
    grouped: set[tuple[str, str, bool, bool]] = set()
    counts: dict[str, int] = {}
    for request in requests:
        for family in _source_family_values(request.get("preferred_source_families", ()), supported_only=True):
            key = (
                family,
                _citation_binding_query_key(request),
                _citation_binding_freshness_required(request),
                _citation_binding_official_source_preferred(request, family=family),
            )
            if key in grouped:
                continue
            grouped.add(key)
            counts[family] = counts.get(family, 0) + 1
    if grouped:
        return dict(sorted(counts.items()))
    summary_counts = {
        key: _int_or_zero(value)
        for key, value in _mapping(
            _nested(payload, "summary", "preferred_source_family_counts")
        ).items()
        if key in _supported_source_family_names() and _int_or_zero(value) > 0
    }
    return dict(sorted(summary_counts.items()))


def _citation_binding_query_key(request: Mapping[str, Any]) -> str:
    query = _clean_text(request.get("query"))
    if not query:
        for seed in _string_tuple(request.get("query_seeds", ())):
            query = _clean_text(seed)
            if query:
                break
    return " ".join(query.casefold().split())


def _citation_binding_freshness_required(request: Mapping[str, Any]) -> bool:
    return bool(request.get("requires_timestamp")) or _clean_text(request.get("lane")) == "temporal_evidence"


def _citation_binding_official_source_preferred(
    request: Mapping[str, Any],
    *,
    family: str,
) -> bool:
    if family in {"official", "official_statistics"}:
        return True
    lane = _clean_text(request.get("lane"))
    return lane in {"numeric_statistical_evidence", "temporal_evidence"}


def _citation_binding_collection_requests(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None,
) -> tuple[Mapping[str, Any], ...]:
    requests_path = _resolve_path(
        _nested(payload, "paths", "collection_requests"),
        base=source_path,
    )
    if requests_path is not None and requests_path.exists():
        rows: list[Mapping[str, Any]] = []
        with requests_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, Mapping):
                    rows.append(dict(row))
        return tuple(rows)
    return tuple(_mapping_sequence(payload.get("collection_requests", ())))


def _source_family_values(value: Any, *, supported_only: bool = False) -> tuple[str, ...]:
    families = tuple(
        item.casefold().replace("-", "_").replace(" ", "_")
        for item in _string_tuple(value)
        if str(item)
    )
    if supported_only:
        supported = _supported_source_family_names()
        families = tuple(item for item in families if item in supported)
    return tuple(dict.fromkeys(families))


def _supported_source_family_names() -> frozenset[str]:
    return frozenset({
        "domain_specific",
        "news",
        "official",
        "official_statistics",
        "scholarly",
    })


def _source_family_catalog_adapter_action(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None,
) -> Mapping[str, Any]:
    tasks_path = _resolve_path(_nested(payload, "paths", "collection_tasks"), base=source_path)
    family_counts = _source_family_task_counts(payload, tasks_path=tasks_path)
    command_templates = _source_family_catalog_adapter_commands(
        tasks_path,
        family_counts=family_counts,
    )
    return {
        "action_id": "run_source_family_catalog_adapters",
        "title": "Run source-family catalog adapters",
        "action_type": "workflow_plan",
        "priority": 80,
        "evidence_routes": ("source_family_acquisition", "citation_evidence"),
        "suggested_commands": command_templates,
        "metadata": {
            "required_inputs": (),
            "closure_outputs": tuple(
                f"{adapter}_source_family_catalog_report"
                for adapter in _source_family_adapter_names(family_counts)
            ),
            "source_collection_workflow": SOURCE_FAMILY_CATALOG_COLLECTION_WORKFLOW,
            "collection_task_count": _int_or_zero(
                _nested(payload, "summary", "collection_task_count")
            ),
            "task_source_family_counts": family_counts,
            "reason": (
                "source-family acquisition plan is ready; run provider adapters "
                "before rerunning citation/source-family coverage"
            ),
        },
    }


def _source_family_task_counts(
    payload: Mapping[str, Any],
    *,
    tasks_path: Path | None,
) -> dict[str, int]:
    summary_counts = {
        str(key): _int_or_zero(value)
        for key, value in _mapping(
            _nested(payload, "summary", "task_source_family_counts")
        ).items()
        if str(key) and _int_or_zero(value) > 0
    }
    if summary_counts:
        return dict(sorted(summary_counts.items()))
    if tasks_path is None or not tasks_path.exists():
        return {}
    counts: dict[str, int] = {}
    with tasks_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            family = str(row.get("source_family") or "").strip()
            if family:
                counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def _source_family_adapter_names(family_counts: Mapping[str, int]) -> tuple[str, ...]:
    adapters: list[str] = []
    if _int_or_zero(family_counts.get("scholarly")):
        adapters.extend(("crossref", "openalex"))
    if _int_or_zero(family_counts.get("official_statistics")):
        adapters.append("worldbank")
    if _int_or_zero(family_counts.get("news")):
        adapters.extend(("gdelt", "seeded_news"))
    if _int_or_zero(family_counts.get("official")):
        adapters.append("official_site")
    if _int_or_zero(family_counts.get("domain_specific")):
        adapters.append("seeded_domain_specific")
    return tuple(dict.fromkeys(adapters))


def _source_family_catalog_adapter_commands(
    tasks_path: str | Path | None,
    *,
    family_counts: Mapping[str, int],
    metadata_source: str = SOURCE_FAMILY_CATALOG_COLLECTION_WORKFLOW,
) -> tuple[str, ...]:
    if tasks_path is None or not family_counts:
        return ()
    task_path = str(tasks_path)
    commands: list[str] = []
    for adapter in _source_family_adapter_names(family_counts):
        if adapter == "crossref":
            commands.append(_source_family_adapter_command(
                "benchmarks/run_crossref_source_family_catalog_adapter.py",
                tasks_path=task_path,
                source_family="scholarly",
                metadata_source=metadata_source,
                extra_parts=("--max-query-variants", "2", "--rows-per-query", "2"),
            ))
        elif adapter == "openalex":
            commands.append(_source_family_adapter_command(
                "benchmarks/run_openalex_source_family_catalog_adapter.py",
                tasks_path=task_path,
                source_family="scholarly",
                metadata_source=metadata_source,
                extra_parts=(
                    "--max-query-variants",
                    "4",
                    "--rows-per-query",
                    "2",
                    "--include-abstracts",
                ),
            ))
        elif adapter == "worldbank":
            commands.append(_source_family_adapter_command(
                "benchmarks/run_worldbank_source_family_catalog_adapter.py",
                tasks_path=task_path,
                source_family="official_statistics",
                metadata_source=metadata_source,
                extra_parts=("--indicator", "SP.POP.TOTL", "--mrnev", "1"),
            ))
        elif adapter == "gdelt":
            commands.append(_source_family_adapter_command(
                "benchmarks/run_gdelt_source_family_catalog_adapter.py",
                tasks_path=task_path,
                source_family="news",
                metadata_source=metadata_source,
                extra_parts=("--max-query-variants", "2", "--max-records", "5"),
            ))
        elif adapter == "seeded_news":
            commands.append(_source_family_adapter_command(
                "benchmarks/run_seeded_url_source_family_catalog_adapter.py",
                tasks_path=task_path,
                source_family="news",
                metadata_source=metadata_source,
                extra_parts=("--seeds", "...", "--provider", "seeded_news", "--no-fetch"),
            ))
        elif adapter == "official_site":
            commands.append(_source_family_adapter_command(
                "benchmarks/run_official_site_source_family_catalog_adapter.py",
                tasks_path=task_path,
                source_family="official",
                metadata_source=metadata_source,
                extra_parts=("--seeds", "..."),
            ))
        elif adapter == "seeded_domain_specific":
            commands.append(_source_family_adapter_command(
                "benchmarks/run_seeded_url_source_family_catalog_adapter.py",
                tasks_path=task_path,
                source_family="domain_specific",
                metadata_source=metadata_source,
                extra_parts=(
                    "--seeds",
                    "...",
                    "--provider",
                    "seeded_domain_specific",
                    "--no-fetch",
                ),
            ))
    return tuple(commands)


def _source_family_adapter_command(
    script: str,
    *,
    tasks_path: str,
    source_family: str,
    metadata_source: str,
    extra_parts: Sequence[str] = (),
) -> str:
    parts = [
        "python",
        script,
        "--tasks",
        tasks_path,
        "--output",
        "...",
        "--report-json",
        "...",
        "--artifact-manifest",
        "...",
        "--registry",
        "...",
        "--name",
        "...",
        "--version",
        "...",
        "--source-family",
        source_family,
    ]
    parts.extend(str(item) for item in extra_parts)
    parts.extend((
        "--metadata",
        f"source={metadata_source}",
        "--metadata",
        f"source_family={source_family}",
    ))
    return _shell_join(parts)


def _source_family_structured_qa_lane_rerun_action(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    entries = _mapping_sequence(payload.get("entries", ()))
    command_templates = tuple(
        command
        for command in (_lane_rerun_command(entry.get("command")) for entry in entries)
        if command
    )
    missing_roles = {
        str(item.get("role") or "")
        for entry in entries
        for item in _mapping_sequence(entry.get("missing_inputs", ()))
        if str(item.get("role") or "")
    }
    required_inputs = tuple(
        dict.fromkeys(
            input_name
            for role in sorted(missing_roles)
            for input_name in _lane_rerun_required_inputs(role)
            if input_name
        )
    )
    summary = _mapping(payload.get("summary"))
    return {
        "action_id": "run_source_family_structured_qa_lane_batches",
        "title": "Run source-family structured QA lane batches",
        "action_type": "workflow_plan",
        "priority": 78,
        "evidence_routes": ("source_family_structured_qa_lane_batches",),
        "suggested_commands": command_templates,
        "metadata": {
            "required_inputs": required_inputs,
            "closure_outputs": ("source_family_structured_qa_lane_batch_reports",),
            "source_rerun_workflow": SOURCE_FAMILY_STRUCTURED_QA_LANE_RERUN_WORKFLOW,
            "batch_count": _int_or_zero(summary.get("batch_count")),
            "ready_command_count": _int_or_zero(summary.get("ready_command_count")),
            "missing_command_count": _int_or_zero(summary.get("missing_command_count")),
            "source_backed_batch_count": _int_or_zero(
                summary.get("source_backed_batch_count")
            ),
            "rule_only_batch_count": _int_or_zero(summary.get("rule_only_batch_count")),
            "command_status_counts": dict(_mapping(summary.get("command_status_counts"))),
            "request_type_counts": dict(_mapping(summary.get("request_type_counts"))),
            "lane_counts": dict(_mapping(summary.get("lane_counts"))),
            "reason": (
                "structured-QA lane rerun queue is ready; execute reviewed "
                "candidate collection and rule-stub batches before remapping claims"
            ),
        },
    }


def _lane_rerun_command(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = tuple(str(item) for item in value if str(item))
        return _shell_join(parts) if parts else ""
    return ""


def _lane_rerun_required_inputs(role: str) -> tuple[str, ...]:
    return {
        "collection_corpus": ("source_family_structured_qa_fact_collection_corpus",),
        "lane_queue": ("source_family_structured_qa_lane_execution_queue",),
        "source_catalog": ("source_family_source_catalog",),
    }.get(str(role), ())


def _unresolved_summary_actions(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None,
) -> tuple[Mapping[str, Any], ...]:
    actions = []
    for action in _mapping_sequence(payload.get("next_actions", ())):
        action_id = str(action.get("action_id") or "")
        if action_id == "improve_unresolved_citation_alignment":
            actions.append(_unresolved_citation_alignment_action(payload, action, source_path=source_path))
        elif action_id in {
            "complete_retrieval_semantic_gap_review",
            "expand_scoped_covered_fact_alignment",
        }:
            actions.append(_unresolved_semantic_gap_review_action(payload, action, source_path=source_path))
        elif action_id == "review_frontier_queue_command_bindings":
            actions.append(_unresolved_frontier_queue_review_action(payload, action, source_path=source_path))
        elif action_id == "execute_reviewed_frontier_queue_command_plan":
            actions.append(
                _unresolved_frontier_queue_execute_action(payload, action, source_path=source_path)
            )
        elif action_id == "repair_frontier_queue_command_execution":
            actions.append(_unresolved_frontier_queue_repair_action(payload, action, source_path=source_path))
        elif action_id == "stage_frontier_queue_seed_inputs":
            actions.append(
                _unresolved_frontier_queue_seed_input_action(
                    payload,
                    action,
                    source_path=source_path,
                )
            )
        elif action_id == "run_world_model_rule_adapter_promotion_workflow":
            actions.append(
                _unresolved_world_model_rule_adapter_promotion_action(
                    payload,
                    action,
                    source_path=source_path,
                )
            )
        elif action_id == "fill_and_promote_remaining_world_model_rules":
            actions.append(_unresolved_world_model_rules_action(payload, action, source_path=source_path))
        else:
            actions.append(_generic_unresolved_summary_action(action))
    return tuple(actions)


def _unresolved_citation_alignment_action(
    payload: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    source_path: Path | None,
) -> Mapping[str, Any]:
    command_templates: list[str] = []
    workflow_paths = _string_tuple(_nested(payload, "paths", "citation_workflows"))
    diagnostic_next_actions = _mapping(action.get("query_sweep_recommended_next_action_counts"))
    diagnostic_failure_counts = _mapping(action.get("query_sweep_failure_reason_counts"))
    diagnostics = _citation_diagnostic_actions(action)
    for workflow_path in workflow_paths:
        command_templates.extend(
            _citation_alignment_commands(
                workflow_path,
                payload=payload,
                action=action,
                source_path=source_path,
            )
        )
    return {
        **dict(action),
        "title": "Rerun unresolved citation alignment candidates",
        "action_type": "workflow_plan",
        "evidence_routes": ("citation_evidence", "source_family_acquisition"),
        "suggested_commands": tuple(command_templates),
        "metadata": {
            "required_inputs": (),
            "closure_outputs": _citation_closure_outputs(diagnostics),
            "source_summary_workflow": UNRESOLVED_SUMMARY_WORKFLOW,
            "source_citation_workflow_count": len(workflow_paths),
            "reason": str(action.get("reason") or ""),
            "query_sweep_failure_reason_counts": dict(diagnostic_failure_counts),
            "query_sweep_recommended_next_action_counts": dict(diagnostic_next_actions),
            "query_sweep_no_hit_strategy_count": _int_or_zero(
                action.get("query_sweep_no_hit_strategy_count")
            ),
            "query_sweep_target_route_not_selected_strategy_count": _int_or_zero(
                action.get("query_sweep_target_route_not_selected_strategy_count")
            ),
            "query_sweep_blind_refuted_rate_below_min_strategy_count": _int_or_zero(
                action.get("query_sweep_blind_refuted_rate_below_min_strategy_count")
            ),
            "query_sweep_verified_false_alarm_above_max_strategy_count": _int_or_zero(
                action.get("query_sweep_verified_false_alarm_above_max_strategy_count")
            ),
        },
    }


def _citation_closure_outputs(diagnostics: Sequence[str]) -> tuple[str, ...]:
    outputs = [
        "unresolved_citation_alignment_workflow_report",
        "unresolved_citation_alignment_artifact_manifest",
    ]
    if "expand_or_retarget_source_corpus" in diagnostics:
        outputs.extend((
            "source_family_coverage_audit_report",
            "source_family_catalog_collection_plan",
        ))
    return tuple(outputs)


def _citation_alignment_commands(
    workflow_path_value: str,
    *,
    payload: Mapping[str, Any],
    action: Mapping[str, Any],
    source_path: Path | None,
) -> tuple[str, ...]:
    workflow_path = _resolve_path(workflow_path_value, base=source_path)
    workflow = _load_optional_json(workflow_path)
    workflow_paths = _mapping(workflow.get("paths"))
    manifest_path = _resolve_path(workflow_paths.get("artifact_manifest"), base=workflow_path)
    manifest = _load_optional_json(manifest_path)
    artifacts = _mapping(manifest.get("artifacts"))
    queue = _artifact_path(artifacts, "queue_report", base=manifest_path) or _resolve_path(
        _nested(payload, "paths", "unresolved_queue"),
        base=source_path,
    )
    scores = _artifact_path(artifacts, "scores", base=manifest_path)
    blind_spots = _artifact_path(artifacts, "blind_spots", base=manifest_path)
    source_catalogs = tuple(
        path
        for _, path in sorted(
            (
                (key, _artifact_path(artifacts, key, base=manifest_path))
                for key in artifacts
                if str(key).startswith("source_catalog_")
            ),
            key=lambda item: item[0],
        )
        if path is not None
    )
    controlled_sweeps = tuple(
        path
        for _, path in sorted(
            (
                (key, _artifact_path(artifacts, key, base=manifest_path))
                for key in artifacts
                if str(key).startswith("controlled_sweep_")
            ),
            key=lambda item: item[0],
        )
        if path is not None
    )
    requests = _resolve_path(workflow_paths.get("requests"), base=workflow_path)
    adapter_results = _resolve_path(workflow_paths.get("adapter_results"), base=workflow_path)
    if not (queue and scores and blind_spots and source_catalogs):
        return ()
    config = _mapping(workflow.get("config"))
    diagnostics = _citation_diagnostic_actions(action)
    current_mode = str(config.get("query_mode") or "claim_entity")
    candidate_modes = _citation_candidate_query_modes(
        current_mode,
        diagnostics=diagnostics,
    )
    if not candidate_modes:
        candidate_modes = (current_mode,)
    commands = []
    target_routes = _citation_candidate_target_routes(
        str(config.get("target_route") or "retrieval_groundedness"),
        diagnostics=diagnostics,
    )
    query_fields = _citation_query_fields(config, diagnostics=diagnostics)
    source_family_filters = _citation_source_family_filters(config, diagnostics=diagnostics)
    for mode in candidate_modes:
        target_route = target_routes[0]
        parts = [
            "python",
            "benchmarks/run_source_family_citation_search_workflow.py",
            "--queue",
            str(queue),
        ]
        for catalog in source_catalogs:
            parts.extend(("--source-catalog", str(catalog)))
        parts.extend(("--scores", str(scores), "--blind-spots", str(blind_spots)))
        for controlled_sweep in controlled_sweeps:
            parts.extend(("--controlled-sweep", str(controlled_sweep)))
        parts.extend((
            "--output-dir",
            "...",
            "--workflow-report",
            "...",
            "--artifact-manifest",
            "...",
            "--registry",
            "...",
            "--name",
            "...",
            "--version",
            "...",
            "--query-mode",
            mode,
            "--target-route",
            target_route,
            "--query-fields",
            query_fields,
            "--retriever-min-overlaps",
            _citation_retriever_min_overlaps(config, diagnostics=diagnostics),
            "--source-family-filters",
            source_family_filters,
            "--retrieval-limit",
            str(_citation_retrieval_limit(config, diagnostics=diagnostics)),
            "--verifier-min-overlap",
            _citation_verifier_min_overlap(config, diagnostics=diagnostics),
            "--metadata",
            "closure_action=improve_unresolved_citation_alignment",
            "--metadata",
            f"diagnostic_query_mode={mode}",
        ))
        if config.get("adapter_diversify_source_families") is True:
            parts.append("--adapter-diversify-source-families")
        commands.append(_shell_join(parts))
    if "enable_or_repair_retrieval_route_selection" in diagnostics:
        for route in target_routes[1:]:
            parts = [
                "python",
                "benchmarks/run_source_family_citation_search_workflow.py",
                "--queue",
                str(queue),
            ]
            for catalog in source_catalogs:
                parts.extend(("--source-catalog", str(catalog)))
            parts.extend(("--scores", str(scores), "--blind-spots", str(blind_spots)))
            for controlled_sweep in controlled_sweeps:
                parts.extend(("--controlled-sweep", str(controlled_sweep)))
            parts.extend((
                "--output-dir",
                "...",
                "--workflow-report",
                "...",
                "--artifact-manifest",
                "...",
                "--registry",
                "...",
                "--name",
                "...",
                "--version",
                "...",
                "--query-mode",
                current_mode,
                "--target-route",
                route,
                "--query-fields",
                query_fields,
                "--retriever-min-overlaps",
                _citation_retriever_min_overlaps(config, diagnostics=diagnostics),
                "--source-family-filters",
                source_family_filters,
                "--retrieval-limit",
                str(_citation_retrieval_limit(config, diagnostics=diagnostics)),
                "--metadata",
                "closure_action=improve_unresolved_citation_alignment",
                "--metadata",
                f"diagnostic_target_route={route}",
            ))
            commands.append(_shell_join(parts))
    if "expand_or_retarget_source_corpus" in diagnostics and requests and adapter_results:
        commands.extend(_citation_source_family_gap_commands(requests, adapter_results))
    return tuple(commands)


def _citation_diagnostic_actions(action: Mapping[str, Any]) -> tuple[str, ...]:
    counts = _mapping(action.get("query_sweep_recommended_next_action_counts"))
    actions = [str(key) for key, value in counts.items() if _int_or_zero(value) > 0]
    if actions:
        return tuple(dict.fromkeys(actions))
    failure_counts = _mapping(action.get("query_sweep_failure_reason_counts"))
    fallback = []
    if _int_or_zero(failure_counts.get("no_retrieval_hits")):
        fallback.append("expand_or_retarget_source_corpus")
    if _int_or_zero(failure_counts.get("target_route_not_selected")):
        fallback.append("enable_or_repair_retrieval_route_selection")
    if _int_or_zero(failure_counts.get("blind_refuted_rate_below_min")):
        fallback.append("improve_claim_intent_alignment_or_query_construction")
    if _int_or_zero(failure_counts.get("verified_false_alarm_above_max")):
        fallback.append("tighten_false_alarm_calibration")
    return tuple(fallback)


def _citation_candidate_query_modes(
    current_mode: str,
    *,
    diagnostics: Sequence[str],
) -> tuple[str, ...]:
    if "improve_claim_intent_alignment_or_query_construction" in diagnostics:
        ordered = ("question_and_query", "queue_query", "question", "claim_entity")
    elif "expand_or_retarget_source_corpus" in diagnostics:
        ordered = ("queue_query", "question_and_query", "claim_entity", "question")
    else:
        ordered = ("question_and_query", "queue_query", "question", "claim_entity")
    return tuple(mode for mode in ordered if mode != current_mode)[:3]


def _citation_candidate_target_routes(
    current_route: str,
    *,
    diagnostics: Sequence[str],
) -> tuple[str, ...]:
    routes = [current_route]
    if "enable_or_repair_retrieval_route_selection" in diagnostics:
        routes.extend(("retrieval_groundedness", "groundedness", "retrieval_structured_qa"))
    return tuple(dict.fromkeys(route for route in routes if route))


def _citation_query_fields(
    config: Mapping[str, Any],
    *,
    diagnostics: Sequence[str],
) -> str:
    fields = list(_string_tuple(config.get("query_fields")) or ("question", "question_answer"))
    if _citation_needs_alignment_sweep(diagnostics):
        fields.extend(("citation_question", "citation_entity"))
    return ",".join(dict.fromkeys(field for field in fields if field))


def _citation_source_family_filters(
    config: Mapping[str, Any],
    *,
    diagnostics: Sequence[str],
) -> str:
    filters = list(_string_tuple(config.get("source_family_filters")) or ("off",))
    if _citation_needs_alignment_sweep(diagnostics):
        filters = ["planned_rerank", *filters]
    return ",".join(dict.fromkeys(mode for mode in filters if mode))


def _citation_needs_alignment_sweep(diagnostics: Sequence[str]) -> bool:
    return bool({
        "extract_structured_facts_from_retrieved_sources",
        "improve_claim_evidence_alignment_rules",
        "improve_claim_intent_alignment_or_query_construction",
        "improve_query_planning_or_route_selection",
    } & set(diagnostics))


def _citation_retriever_min_overlaps(
    config: Mapping[str, Any],
    *,
    diagnostics: Sequence[str],
) -> str:
    if "expand_or_retarget_source_corpus" in diagnostics:
        return "0.5,0.35,0.2"
    values = ",".join(str(item) for item in _sequence(config.get("retriever_min_overlaps")))
    return values or "0.95,0.8,0.65,0.5"


def _citation_retrieval_limit(
    config: Mapping[str, Any],
    *,
    diagnostics: Sequence[str],
) -> int:
    current = _int_or_zero(config.get("retrieval_limit")) or 3
    if "expand_or_retarget_source_corpus" in diagnostics:
        return max(current, 5)
    return current


def _citation_verifier_min_overlap(
    config: Mapping[str, Any],
    *,
    diagnostics: Sequence[str],
) -> str:
    _ = diagnostics
    current = str(config.get("verifier_min_overlap") or "0.65")
    return current


def _citation_source_family_gap_commands(
    requests: Path,
    adapter_results: Path,
) -> tuple[str, ...]:
    audit_parts = [
        "python",
        "benchmarks/audit_source_family_coverage.py",
        "--requests",
        str(requests),
        "--adapter-results",
        str(adapter_results),
        "--json",
        "...",
        "--acquisition-plan-jsonl",
        "...",
        "--artifact-manifest",
        "...",
        "--registry",
        "...",
        "--name",
        "...",
        "--version",
        "...",
        "--metadata",
        "closure_action=improve_unresolved_citation_alignment",
    ]
    collection_parts = [
        "python",
        "benchmarks/plan_source_family_catalog_collection.py",
        "--acquisition-plan",
        "...",
        "--tasks-jsonl",
        "...",
        "--report-json",
        "...",
        "--artifact-manifest",
        "...",
        "--registry",
        "...",
        "--name",
        "...",
        "--version",
        "...",
        "--metadata",
        "closure_action=improve_unresolved_citation_alignment",
    ]
    return (_shell_join(audit_parts), _shell_join(collection_parts))


def _unresolved_semantic_gap_review_action(
    payload: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    source_path: Path | None,
) -> Mapping[str, Any]:
    action_id = str(action.get("action_id") or "complete_retrieval_semantic_gap_review")
    workflow_paths = _string_tuple(_nested(payload, "paths", "semantic_gap_review_workflows"))
    commands: list[str] = []
    for workflow_path in workflow_paths:
        commands.extend(
            _semantic_gap_review_commands(
                workflow_path,
                source_path=source_path,
                closure_action=action_id,
            )
        )
    required_inputs: tuple[str, ...] = ()
    if not commands:
        commands.append(
            _semantic_gap_review_command(
                verified_records_jsonl=None,
                record_indices_json=None,
                config={},
                include_record_indices_placeholder=True,
                closure_action=action_id,
            )
        )
        required_inputs = (
            "source_bound_verified_records_jsonl",
            "detectability_blind_spot_record_indices_json",
        )
    return {
        **dict(action),
        "title": (
            "Expand scoped covered-fact alignment coverage"
            if action_id == "expand_scoped_covered_fact_alignment"
            else "Complete retrieval semantic-gap covered-fact review"
        ),
        "action_type": "workflow_plan",
        "evidence_routes": (
            "semantic_gap_review",
            "retrieval_structured_qa",
            "source_family_acquisition",
        ),
        "suggested_commands": tuple(commands),
        "metadata": {
            "required_inputs": required_inputs,
            "closure_outputs": (
                "semantic_gap_review_workflow_report",
                "semantic_gap_review_artifact_manifest",
                "semantic_gap_covered_fact_route_summary",
                "semantic_gap_covered_fact_route_manifest",
            ),
            "source_summary_workflow": UNRESOLVED_SUMMARY_WORKFLOW,
            "source_semantic_gap_workflow_count": len(workflow_paths),
            "reason": str(action.get("reason") or ""),
            "semantic_gap_candidate_count": _int_or_zero(
                action.get("semantic_gap_candidate_count")
            ),
            "semantic_gap_fact_candidate_count": _int_or_zero(
                action.get("semantic_gap_fact_candidate_count")
            ),
            "approved_source_document_count": _int_or_zero(
                action.get("approved_source_document_count")
            ),
            "source_family_qa_document_count": _int_or_zero(
                action.get("source_family_qa_document_count")
            ),
            "semantic_gap_covered_fact_route_n_records": _int_or_zero(
                action.get("semantic_gap_covered_fact_route_n_records")
            ),
            "semantic_gap_coverage_gap_count": _int_or_zero(
                action.get("semantic_gap_coverage_gap_count")
            ),
            "semantic_gap_coverage_rate": action.get("semantic_gap_coverage_rate"),
            "unresolved_target_count": _int_or_zero(action.get("unresolved_target_count")),
        },
    }


def _semantic_gap_review_commands(
    workflow_path_value: str,
    *,
    source_path: Path | None,
    closure_action: str = "complete_retrieval_semantic_gap_review",
) -> tuple[str, ...]:
    workflow_path = _resolve_path(workflow_path_value, base=source_path)
    workflow = _load_optional_json(workflow_path)
    workflow_paths = _mapping(workflow.get("paths"))
    manifest_path = _resolve_path(workflow_paths.get("artifact_manifest"), base=workflow_path)
    manifest = _load_optional_json(manifest_path)
    artifacts = _mapping(manifest.get("artifacts"))
    verified_records = _resolve_path(
        _nested(workflow, "source", "verified_records_jsonl"),
        base=workflow_path,
    ) or _artifact_path(artifacts, "verified_records_jsonl", base=manifest_path)
    record_indices = _resolve_path(
        _nested(workflow, "source", "record_indices_json"),
        base=workflow_path,
    ) or _artifact_path(artifacts, "record_indices_json", base=manifest_path)
    return (
        _semantic_gap_review_command(
            verified_records_jsonl=verified_records,
            record_indices_json=record_indices,
            config=_mapping(workflow.get("config")),
            include_record_indices_placeholder=False,
            closure_action=closure_action,
        ),
    )


def _semantic_gap_review_command(
    *,
    verified_records_jsonl: Path | None,
    record_indices_json: Path | None,
    config: Mapping[str, Any],
    include_record_indices_placeholder: bool,
    closure_action: str = "complete_retrieval_semantic_gap_review",
) -> str:
    parts: list[Any] = [
        "python",
        "benchmarks/run_retrieval_semantic_gap_review_workflow.py",
        "--verified-records-jsonl",
        "..." if verified_records_jsonl is None else str(verified_records_jsonl),
        "--output-dir",
        "...",
        "--workflow-report",
        "...",
        "--artifact-manifest",
        "...",
        "--registry",
        "...",
        "--name",
        "...",
        "--version",
        "...",
    ]
    if record_indices_json is not None:
        parts.extend(("--record-indices-json", str(record_indices_json)))
    elif include_record_indices_placeholder:
        parts.extend(("--record-indices-json", "..."))
    _append_config_value(parts, "--mode", config.get("mode"))
    _append_config_value(parts, "--min-hits", config.get("min_hits"))
    _append_config_value(parts, "--max-targets", config.get("max_targets"))
    _append_config_value(parts, "--max-hits-per-target", config.get("max_hits_per_target"))
    _append_config_value(parts, "--min-confidence", config.get("min_confidence"))
    _append_config_value(parts, "--reviewer", config.get("reviewer"))
    _append_config_value(parts, "--reviewed-at", config.get("reviewed_at"))
    parts.append("--run-covered-fact-route")
    _append_config_value(parts, "--covered-fact-limit", config.get("covered_fact_limit"))
    _append_config_value(parts, "--covered-fact-score-name", config.get("covered_fact_score_name"))
    _append_config_value(parts, "--covered-fact-signal", config.get("covered_fact_signal"))
    _append_config_value(parts, "--covered-fact-alpha", config.get("covered_fact_alpha"))
    _append_config_value(parts, "--covered-fact-seed", config.get("covered_fact_seed"))
    if config.get("skip_qid_values") is False:
        parts.append("--keep-qid-values")
    parts.extend((
        "--metadata",
        f"closure_action={closure_action}",
    ))
    return _shell_join(parts)


def _unresolved_frontier_queue_review_action(
    payload: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    source_path: Path | None,
) -> Mapping[str, Any]:
    review_paths = _string_tuple(_nested(payload, "paths", "frontier_command_binding_reviews"))
    commands: list[str] = []
    required_inputs: list[str] = []
    for review_path_value in review_paths:
        command, missing = _frontier_command_binding_review_command_from_report(
            review_path_value,
            source_path=source_path,
        )
        commands.append(command)
        required_inputs.extend(missing)
    if not commands:
        command, missing = _frontier_command_binding_review_command(
            bound_command_plan=None,
            base_bindings=None,
            review_decisions=None,
        )
        commands.append(command)
        required_inputs.extend(missing)
    return {
        **dict(action),
        "title": "Review frontier queue command bindings",
        "action_type": "workflow_plan",
        "evidence_routes": ("frontier_queue_execution",),
        "suggested_commands": tuple(commands),
        "metadata": {
            "required_inputs": tuple(dict.fromkeys(required_inputs)),
            "closure_outputs": (
                "frontier_command_binding_review_report",
                "frontier_command_binding_review_approved_bindings",
                "frontier_command_binding_review_artifact_manifest",
            ),
            "source_summary_workflow": UNRESOLVED_SUMMARY_WORKFLOW,
            "source_command_binding_review_count": len(review_paths),
            "reason": str(action.get("reason") or ""),
            "blocked_entry_count": _int_or_zero(action.get("blocked_entry_count")),
            "pending_review_count": _int_or_zero(action.get("pending_review_count")),
            "binding_not_reviewed_count": _int_or_zero(
                action.get("binding_not_reviewed_count")
            ),
        },
    }


def _unresolved_frontier_queue_execute_action(
    payload: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    source_path: Path | None,
) -> Mapping[str, Any]:
    run_paths = _string_tuple(_nested(payload, "paths", "frontier_bound_command_runs"))
    review_paths = _string_tuple(_nested(payload, "paths", "frontier_command_binding_reviews"))
    commands: list[str] = []
    required_inputs: list[str] = []
    for run_path_value in run_paths:
        command, missing = _frontier_bound_command_run_command_from_report(
            run_path_value,
            source_path=source_path,
        )
        commands.append(command)
        required_inputs.extend(missing)
    if not commands:
        for review_path_value in review_paths:
            review_commands, missing = _frontier_queue_execute_commands_from_review(
                review_path_value,
                source_path=source_path,
            )
            commands.extend(review_commands)
            required_inputs.extend(missing)
    if not commands:
        review_commands, missing = _frontier_queue_execute_commands(
            command_plan=None,
            approved_bindings=None,
            bound_command_plan=None,
        )
        commands.extend(review_commands)
        required_inputs.extend(missing)
    return {
        **dict(action),
        "title": "Execute reviewed frontier queue command plan",
        "action_type": "workflow_plan",
        "evidence_routes": ("frontier_queue_execution",),
        "suggested_commands": tuple(commands),
        "metadata": {
            "required_inputs": tuple(dict.fromkeys(required_inputs)),
            "closure_outputs": (
                "frontier_approved_bound_command_plan",
                "frontier_approved_bound_command_manifest",
                "frontier_bound_command_run_report",
                "frontier_bound_command_run_manifest",
            ),
            "source_summary_workflow": UNRESOLVED_SUMMARY_WORKFLOW,
            "source_bound_command_run_count": len(run_paths),
            "source_command_binding_review_count": len(review_paths),
            "reason": str(action.get("reason") or ""),
            "ready_review_count": _int_or_zero(action.get("ready_review_count")),
            "dry_run_report_count": _int_or_zero(action.get("dry_run_report_count")),
            "command_count": _int_or_zero(action.get("command_count")),
        },
    }


def _unresolved_frontier_queue_repair_action(
    payload: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    source_path: Path | None,
) -> Mapping[str, Any]:
    run_paths = _string_tuple(_nested(payload, "paths", "frontier_bound_command_runs"))
    commands: list[str] = []
    required_inputs: list[str] = []
    for run_path_value in run_paths:
        command, missing = _frontier_bound_command_run_command_from_report(
            run_path_value,
            source_path=source_path,
        )
        commands.append(command)
        required_inputs.extend(missing)
    if not commands:
        command, missing = _frontier_bound_command_run_command(bound_command_plan=None)
        commands.append(command)
        required_inputs.extend(missing)
    return {
        **dict(action),
        "title": "Repair frontier queue command execution",
        "action_type": "workflow_plan",
        "evidence_routes": ("frontier_queue_execution",),
        "suggested_commands": tuple(commands),
        "metadata": {
            "required_inputs": tuple(dict.fromkeys(required_inputs)),
            "closure_outputs": (
                "frontier_bound_command_repair_run_report",
                "frontier_bound_command_repair_manifest",
            ),
            "source_summary_workflow": UNRESOLVED_SUMMARY_WORKFLOW,
            "source_bound_command_run_count": len(run_paths),
            "reason": str(action.get("reason") or ""),
            "failed_count": _int_or_zero(action.get("failed_count")),
            "timed_out_count": _int_or_zero(action.get("timed_out_count")),
            "skipped_count": _int_or_zero(action.get("skipped_count")),
            "invalid_command_count": _int_or_zero(action.get("invalid_command_count")),
            "missing_output_count": _int_or_zero(action.get("missing_output_count")),
        },
    }


def _unresolved_frontier_queue_seed_input_action(
    payload: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    source_path: Path | None,
) -> Mapping[str, Any]:
    audit_paths = _summary_path_tuple(
        payload,
        "input_binding_audits",
        "input_binding_audit",
        source_path=source_path,
    )
    binding_paths = _summary_path_tuple(
        payload,
        "frontier_command_bindings",
        "frontier_command_binding",
        source_path=source_path,
    )
    commands: list[str] = []
    required_inputs: list[str] = []
    if not audit_paths:
        required_inputs.append("frontier_input_binding_audit")
    if not binding_paths:
        required_inputs.append("frontier_command_bindings")
    for audit_path, binding_path in _paired_paths(audit_paths, binding_paths):
        commands.append(
            _frontier_seed_input_binding_command(
                input_binding_audit=audit_path,
                base_bindings=binding_path,
            )
        )
    return {
        **dict(action),
        "title": "Stage audited source-family seed inputs",
        "action_type": "workflow_plan",
        "evidence_routes": ("frontier_queue_execution", "source_family_acquisition"),
        "suggested_commands": tuple(commands),
        "metadata": {
            "required_inputs": tuple(required_inputs),
            "closure_outputs": (
                "frontier_seed_input_binding_report",
                "frontier_seed_input_bindings",
                "frontier_seed_input_binding_artifact_manifest",
            ),
            "source_summary_workflow": UNRESOLVED_SUMMARY_WORKFLOW,
            "source_input_binding_audit_count": len(audit_paths),
            "source_frontier_command_binding_count": len(binding_paths),
            "reason": str(action.get("reason") or ""),
            "blocked_seed_count": _int_or_zero(action.get("blocked_seed_count")),
            "pending_review_count": _int_or_zero(action.get("pending_review_count")),
        },
    }


def _frontier_seed_input_binding_command(
    *,
    input_binding_audit: Path | None,
    base_bindings: Path | None,
) -> str:
    parts: list[Any] = [
        "python",
        "benchmarks/bind_frontier_research_queue_seed_inputs.py",
        "--input-binding-audit",
        "..." if input_binding_audit is None else str(input_binding_audit),
        "--base-bindings",
        "..." if base_bindings is None else str(base_bindings),
        "--output-dir",
        "...",
        "--json",
        "...",
        "--bindings-json",
        "...",
        "--artifact-manifest",
        "...",
        "--registry",
        "...",
        "--name",
        "...",
        "--version",
        "...",
    ]
    return _shell_join(parts)


def _frontier_command_binding_review_command_from_report(
    review_path_value: str,
    *,
    source_path: Path | None,
) -> tuple[str, tuple[str, ...]]:
    review_path = _resolve_path(review_path_value, base=source_path)
    review = _load_optional_json(review_path)
    source = _mapping(review.get("source"))
    return _frontier_command_binding_review_command(
        bound_command_plan=_resolve_path(source.get("bound_command_plan"), base=review_path),
        base_bindings=_resolve_path(source.get("base_bindings"), base=review_path),
        review_decisions=_resolve_path(source.get("review_decisions"), base=review_path),
    )


def _frontier_command_binding_review_command(
    *,
    bound_command_plan: Path | None,
    base_bindings: Path | None,
    review_decisions: Path | None,
) -> tuple[str, tuple[str, ...]]:
    missing: list[str] = []
    if bound_command_plan is None:
        missing.append("frontier_bound_command_plan")
    if base_bindings is None:
        missing.append("frontier_command_bindings")
    if review_decisions is None:
        missing.append("frontier_command_review_decisions")
    parts: list[Any] = [
        "python",
        "benchmarks/review_frontier_research_queue_command_bindings.py",
        "--bound-command-plan",
        "..." if bound_command_plan is None else str(bound_command_plan),
        "--base-bindings",
        "..." if base_bindings is None else str(base_bindings),
        "--review-decisions",
        "..." if review_decisions is None else str(review_decisions),
        "--output-dir",
        "...",
        "--json",
        "...",
        "--approved-bindings",
        "...",
        "--artifact-manifest",
        "...",
        "--registry",
        "...",
        "--name",
        "...",
        "--version",
        "...",
    ]
    return _shell_join(parts), tuple(missing)


def _frontier_queue_execute_commands_from_review(
    review_path_value: str,
    *,
    source_path: Path | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    review_path = _resolve_path(review_path_value, base=source_path)
    review = _load_optional_json(review_path)
    review_source = _mapping(review.get("source"))
    bound_plan_path = _resolve_path(review_source.get("bound_command_plan"), base=review_path)
    bound_plan = _load_optional_json(bound_plan_path)
    command_plan = _resolve_path(_nested(bound_plan, "source", "command_plan"), base=bound_plan_path)
    approved_bindings = _resolve_path(_nested(review, "paths", "approved_bindings"), base=review_path)
    return _frontier_queue_execute_commands(
        command_plan=command_plan,
        approved_bindings=approved_bindings,
        bound_command_plan=None,
    )


def _frontier_queue_execute_commands(
    *,
    command_plan: Path | None,
    approved_bindings: Path | None,
    bound_command_plan: Path | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if bound_command_plan is not None:
        command, missing = _frontier_bound_command_run_command(
            bound_command_plan=bound_command_plan,
        )
        return (command,), missing
    missing: list[str] = []
    if command_plan is None:
        missing.append("frontier_command_plan")
    if approved_bindings is None:
        missing.append("approved_frontier_command_bindings")
    bind_command = _shell_join((
        "python",
        "benchmarks/bind_frontier_research_queue_command_plan.py",
        "--command-plan",
        "..." if command_plan is None else str(command_plan),
        "--bindings",
        "..." if approved_bindings is None else str(approved_bindings),
        "--json",
        "...",
        "--artifact-manifest",
        "...",
        "--registry",
        "...",
        "--name",
        "...",
        "--version",
        "...",
    ))
    run_command, _ = _frontier_bound_command_run_command(bound_command_plan=None)
    return (bind_command, run_command), tuple(missing)


def _frontier_bound_command_run_command_from_report(
    run_path_value: str,
    *,
    source_path: Path | None,
) -> tuple[str, tuple[str, ...]]:
    run_path = _resolve_path(run_path_value, base=source_path)
    report = _load_optional_json(run_path)
    source = _mapping(report.get("source"))
    return _frontier_bound_command_run_command(
        bound_command_plan=_resolve_path(source.get("bound_command_plan"), base=run_path),
    )


def _frontier_bound_command_run_command(
    *,
    bound_command_plan: Path | None,
) -> tuple[str, tuple[str, ...]]:
    missing = () if bound_command_plan is not None else ("reviewed_frontier_bound_command_plan",)
    parts: list[Any] = [
        "python",
        "benchmarks/run_frontier_research_queue_bound_command_plan.py",
        "--bound-command-plan",
        "..." if bound_command_plan is None else str(bound_command_plan),
        "--json",
        "...",
        "--artifact-manifest",
        "...",
        "--registry",
        "...",
        "--name",
        "...",
        "--version",
        "...",
        "--execute",
    ]
    return _shell_join(parts), missing


def _unresolved_world_model_rules_action(
    payload: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    source_path: Path | None,
) -> Mapping[str, Any]:
    rule_plan_path = _resolve_path(_nested(payload, "paths", "rule_input_plan"), base=source_path)
    rule_plan = _load_optional_json(rule_plan_path)
    rule_paths = _mapping(rule_plan.get("paths"))
    input_tasks = _resolve_path(rule_paths.get("input_tasks"), base=rule_plan_path)
    input_requests = _resolve_path(rule_paths.get("input_requests"), base=rule_plan_path)
    requeued_rule_plan_path = _resolve_path(
        _nested(payload, "paths", "requeued_rule_input_plan"),
        base=source_path,
    )
    requeued_rule_plan = _load_optional_json(requeued_rule_plan_path)
    requeued_rule_paths = _mapping(requeued_rule_plan.get("paths"))
    requeued_input_tasks = _resolve_path(
        requeued_rule_paths.get("input_tasks"),
        base=requeued_rule_plan_path,
    )
    requeued_input_requests = _resolve_path(
        requeued_rule_paths.get("input_requests"),
        base=requeued_rule_plan_path,
    )
    remaining_value = action.get("remaining_rule_family_counts")
    has_explicit_remaining = isinstance(remaining_value, Mapping)
    remaining_families = _mapping(remaining_value)
    include_primary_rules = (
        not has_explicit_remaining
        or _int_or_zero(remaining_families.get("quantity_or_arithmetic")) > 0
        or _int_or_zero(remaining_families.get("temporal_consistency")) > 0
    )
    include_entity_rules = (
        not has_explicit_remaining
        or _int_or_zero(remaining_families.get("entity_disambiguation")) > 0
    )
    primary_commands = (
        _world_model_rule_commands(input_tasks=input_tasks, input_requests=input_requests)
        if include_primary_rules
        else ()
    )
    entity_commands = (
        _entity_world_model_rule_commands(
            input_tasks=requeued_input_tasks,
            input_requests=requeued_input_requests,
        )
        if include_entity_rules
        else ()
    )
    required_inputs: list[str] = []
    closure_outputs: list[str] = []
    if primary_commands:
        required_inputs.extend((
            "source_backed_numeric_bindings",
            "source_backed_temporal_bindings",
        ))
        closure_outputs.extend((
            "numeric_rule_fill_report",
            "numeric_rule_adapter_report",
            "numeric_rule_promotion_report",
            "temporal_rule_fill_report",
            "temporal_rule_adapter_report",
            "temporal_rule_promotion_report",
        ))
    if entity_commands:
        required_inputs.append("source_backed_entity_bindings")
        closure_outputs.extend((
            "entity_rule_fill_report",
            "entity_rule_adapter_report",
            "entity_rule_promotion_report",
        ))
    commands = (*primary_commands, *entity_commands)
    return {
        **dict(action),
        "title": "Fill and promote remaining deterministic world-model rules",
        "action_type": "workflow_plan",
        "evidence_routes": ("world_model_rules",),
        "suggested_commands": commands,
        "metadata": {
            "required_inputs": tuple(required_inputs),
            "closure_outputs": tuple(closure_outputs),
            "source_summary_workflow": UNRESOLVED_SUMMARY_WORKFLOW,
            "rule_input_plan": None if rule_plan_path is None else str(rule_plan_path),
            "requeued_rule_input_plan": (
                None if requeued_rule_plan_path is None else str(requeued_rule_plan_path)
            ),
            "reason": str(action.get("reason") or ""),
            "missing_input_counts": dict(_mapping(action.get("missing_input_counts"))),
            "remaining_rule_family_counts": dict(remaining_families),
            "promoted_rule_request_ids": _string_tuple(
                action.get("promoted_rule_request_ids", ())
            ),
        },
    }


def _unresolved_world_model_rule_adapter_promotion_action(
    payload: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    source_path: Path | None,
) -> Mapping[str, Any]:
    rollup_path = _resolve_path(_nested(payload, "paths", "input_fill_result_rollup"), base=source_path)
    required_inputs = () if rollup_path is not None else ("world_model_rule_input_fill_result_rollup",)
    command = _shell_join((
        "python",
        "benchmarks/run_frontier_research_queue_rule_adapter_promotion_workflow.py",
        "--input-fill-result-rollup",
        str(rollup_path) if rollup_path is not None else "...",
        "--output-dir",
        "...",
        "--json",
        "...",
        "--artifact-manifest",
        "...",
        "--registry",
        "...",
        "--name",
        "...",
        "--version",
        "...",
        "--build-handoff",
        "--build-evidence-bundle",
        "--metadata",
        "closure_action=run_world_model_rule_adapter_promotion_workflow",
    ))
    return {
        **dict(action),
        "title": "Run deterministic world-model rule adapter and promotion gate",
        "action_type": "workflow_plan",
        "evidence_routes": ("world_model_rules",),
        "suggested_commands": (command,),
        "metadata": {
            "required_inputs": required_inputs,
            "closure_outputs": (
                "frontier_rule_adapter_promotion_workflow_report",
                "world_model_rule_authoring_adapter_report",
                "world_model_rule_candidate_promotion_report",
                "world_model_rule_candidate_handoff_report",
                "mechanism_handoff_evidence_bundle_report",
            ),
            "source_summary_workflow": UNRESOLVED_SUMMARY_WORKFLOW,
            "input_fill_result_rollup": None if rollup_path is None else str(rollup_path),
            "reason": str(action.get("reason") or ""),
            "combined_rule_input_count": _int_or_zero(action.get("combined_rule_input_count")),
            "combined_unfilled_task_count": _int_or_zero(
                action.get("combined_unfilled_task_count")
            ),
            "input_fill_rule_family_counts": dict(
                _mapping(action.get("input_fill_rule_family_counts"))
            ),
        },
    }


def _world_model_rule_commands(
    *,
    input_tasks: Path | None,
    input_requests: Path | None,
) -> tuple[str, ...]:
    if input_tasks is None or input_requests is None:
        return ()
    return (
        _shell_join((
            "python",
            "benchmarks/fill_world_model_rule_inputs_from_numeric_bindings.py",
            "--input-tasks",
            str(input_tasks),
            "--numeric-bindings",
            "...",
            "--output-dir",
            "...",
            "--json",
            "...",
            "--rule-inputs-jsonl",
            "...",
            "--artifact-manifest",
            "...",
            "--registry",
            "...",
            "--name",
            "...",
            "--version",
            "...",
            "--metadata",
            "closure_action=fill_and_promote_remaining_world_model_rules",
        )),
        _shell_join((
            "python",
            "benchmarks/run_world_model_rule_authoring_adapter.py",
            "--rule-stubs",
            str(input_requests),
            "--rule-inputs",
            "...",
            "--output-dir",
            "...",
            "--json",
            "...",
            "--rule-results-jsonl",
            "...",
            "--artifact-manifest",
            "...",
            "--registry",
            "...",
            "--name",
            "...",
            "--version",
            "...",
            "--metadata",
            "closure_lane=numeric_rules",
        )),
        _shell_join((
            "python",
            "benchmarks/promote_world_model_rule_candidates.py",
            "--rule-results",
            "...",
            "--rule-inputs",
            "...",
            "--adapter-report",
            "...",
            "--output-dir",
            "...",
            "--json",
            "...",
            "--artifact-manifest",
            "...",
            "--registry",
            "...",
            "--name",
            "...",
            "--version",
            "...",
            "--metadata",
            "closure_lane=numeric_rules",
        )),
        _shell_join((
            "python",
            "benchmarks/fill_world_model_rule_inputs_from_temporal_bindings.py",
            "--input-tasks",
            str(input_tasks),
            "--temporal-bindings",
            "...",
            "--output-dir",
            "...",
            "--json",
            "...",
            "--rule-inputs-jsonl",
            "...",
            "--artifact-manifest",
            "...",
            "--registry",
            "...",
            "--name",
            "...",
            "--version",
            "...",
            "--metadata",
            "closure_action=fill_and_promote_remaining_world_model_rules",
        )),
        _shell_join((
            "python",
            "benchmarks/run_world_model_rule_authoring_adapter.py",
            "--rule-stubs",
            str(input_requests),
            "--rule-inputs",
            "...",
            "--output-dir",
            "...",
            "--json",
            "...",
            "--rule-results-jsonl",
            "...",
            "--artifact-manifest",
            "...",
            "--registry",
            "...",
            "--name",
            "...",
            "--version",
            "...",
            "--metadata",
            "closure_lane=temporal_rules",
        )),
        _shell_join((
            "python",
            "benchmarks/promote_world_model_rule_candidates.py",
            "--rule-results",
            "...",
            "--rule-inputs",
            "...",
            "--adapter-report",
            "...",
            "--output-dir",
            "...",
            "--json",
            "...",
            "--artifact-manifest",
            "...",
            "--registry",
            "...",
            "--name",
            "...",
            "--version",
            "...",
            "--metadata",
            "closure_lane=temporal_rules",
        )),
    )


def _entity_world_model_rule_commands(
    *,
    input_tasks: Path | None,
    input_requests: Path | None,
) -> tuple[str, ...]:
    if input_tasks is None or input_requests is None:
        return ()
    return (
        _shell_join((
            "python",
            "benchmarks/fill_world_model_rule_inputs_from_entity_bindings.py",
            "--input-tasks",
            str(input_tasks),
            "--entity-bindings",
            "...",
            "--output-dir",
            "...",
            "--json",
            "...",
            "--rule-inputs-jsonl",
            "...",
            "--artifact-manifest",
            "...",
            "--registry",
            "...",
            "--name",
            "...",
            "--version",
            "...",
            "--metadata",
            "closure_action=fill_and_promote_remaining_world_model_rules",
        )),
        _shell_join((
            "python",
            "benchmarks/run_world_model_rule_authoring_adapter.py",
            "--rule-stubs",
            str(input_requests),
            "--rule-inputs",
            "...",
            "--output-dir",
            "...",
            "--json",
            "...",
            "--rule-results-jsonl",
            "...",
            "--artifact-manifest",
            "...",
            "--registry",
            "...",
            "--name",
            "...",
            "--version",
            "...",
            "--metadata",
            "closure_lane=entity_role_rules",
        )),
        _shell_join((
            "python",
            "benchmarks/promote_world_model_rule_candidates.py",
            "--rule-results",
            "...",
            "--rule-inputs",
            "...",
            "--adapter-report",
            "...",
            "--output-dir",
            "...",
            "--json",
            "...",
            "--artifact-manifest",
            "...",
            "--registry",
            "...",
            "--name",
            "...",
            "--version",
            "...",
            "--metadata",
            "closure_lane=entity_role_rules",
        )),
    )


def _generic_unresolved_summary_action(action: Mapping[str, Any]) -> Mapping[str, Any]:
    action_id = str(action.get("action_id") or "unresolved_frontier_action")
    return {
        **dict(action),
        "title": action_id,
        "action_type": "workflow_plan",
        "evidence_routes": (str(action.get("lane") or "unresolved_frontier"),),
        "suggested_commands": (),
        "metadata": {
            "required_inputs": (),
            "closure_outputs": (),
            "source_summary_workflow": UNRESOLVED_SUMMARY_WORKFLOW,
            "reason": str(action.get("reason") or ""),
        },
    }


def _is_active_research_queue(payload: Mapping[str, Any]) -> bool:
    if payload.get("workflow") != "frontier_status_report":
        return True
    research_queue = _mapping(payload.get("research_queue"))
    if research_queue.get("active") is True:
        return True
    return str(research_queue.get("lifecycle_status") or "") in {"active", "current_blocker"}


def _filtered_actions(
    actions: Sequence[Mapping[str, Any]],
    *,
    include_action_ids: Sequence[str],
    exclude_action_ids: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    include = {str(item) for item in include_action_ids if str(item)}
    exclude = {str(item) for item in exclude_action_ids if str(item)}
    filtered = []
    for action in actions:
        action_id = str(action.get("action_id") or "")
        if include and action_id not in include:
            continue
        if action_id in exclude:
            continue
        filtered.append(action)
    return tuple(
        sorted(
            filtered,
            key=lambda item: (-_int_or_zero(item.get("priority")), str(item.get("action_id") or "")),
        )
    )


def _command_entry(action: Mapping[str, Any], *, index: int, plan_root: Path) -> dict[str, Any]:
    action_id = str(action.get("action_id") or f"frontier-research-action-{index:04d}")
    metadata = _mapping(action.get("metadata"))
    command_templates = _string_tuple(action.get("suggested_commands", ()))
    placeholder_count = sum(command.count("...") for command in command_templates)
    required_inputs = _string_tuple(metadata.get("required_inputs", ()))
    closure_outputs = _string_tuple(metadata.get("closure_outputs", ()))
    missing_inputs = _missing_inputs(
        required_inputs=required_inputs,
        placeholder_count=placeholder_count,
        command_templates=command_templates,
    )
    if not command_templates:
        command_status = "missing_command_templates"
    elif missing_inputs:
        command_status = "needs_inputs"
    else:
        command_status = "ready"
    bound_output_dir = plan_root / _slug(action_id)
    planned_outputs = tuple(
        {
            "name": output,
            "path": str(bound_output_dir / f"{_slug(output)}.json"),
            "status": "planned",
        }
        for output in closure_outputs
    )
    return {
        "entry_id": f"frontier-research-{index:04d}",
        "action_id": action_id,
        "title": str(action.get("title") or action_id),
        "action_type": str(action.get("action_type") or "workflow"),
        "priority": _int_or_zero(action.get("priority")),
        "command_status": command_status,
        "evidence_routes": _string_tuple(action.get("evidence_routes", ())),
        "source_gap_ids": _string_tuple(action.get("source_gap_ids", ())),
        "command_templates": command_templates,
        "required_inputs": required_inputs,
        "missing_inputs": missing_inputs,
        "planned_outputs": planned_outputs,
        "binding_hints": {
            "action_id": action_id,
            "bound_output_dir": str(bound_output_dir),
            "command_templates_need_binding": placeholder_count > 0,
            "input_bindings": tuple(
                {
                    "name": name,
                    "placeholder": "..." if name == "bound_command_template_values" else f"<{name}>",
                    "required": True,
                    "status": "unbound",
                }
                for name in tuple(dict.fromkeys((*required_inputs, *missing_inputs)))
            ),
            "output_bindings": planned_outputs,
        },
        "command_summary": {
            "command_template_count": len(command_templates),
            "placeholder_count": placeholder_count,
            "missing_input_count": len(missing_inputs),
            "planned_output_count": len(planned_outputs),
        },
        "metadata": {
            "workflow_keys": _workflow_keys(metadata),
            "required_input_count": len(required_inputs),
            "closure_output_count": len(closure_outputs),
            "semantic_gap_candidate_count": _int_or_zero(
                metadata.get("semantic_gap_candidate_count")
            ),
            "semantic_gap_fact_candidate_count": _int_or_zero(
                metadata.get("semantic_gap_fact_candidate_count")
            ),
            "approved_source_document_count": _int_or_zero(
                metadata.get("approved_source_document_count")
            ),
            "source_family_qa_document_count": _int_or_zero(
                metadata.get("source_family_qa_document_count")
            ),
            "semantic_gap_covered_fact_route_n_records": _int_or_zero(
                metadata.get("semantic_gap_covered_fact_route_n_records")
            ),
            "semantic_gap_coverage_gap_count": _int_or_zero(
                metadata.get("semantic_gap_coverage_gap_count")
            ),
            "semantic_gap_coverage_rate": metadata.get("semantic_gap_coverage_rate"),
            "unresolved_target_count": _int_or_zero(metadata.get("unresolved_target_count")),
            "collection_request_count": _int_or_zero(
                metadata.get("collection_request_count")
            ),
            "collection_task_count": _int_or_zero(
                metadata.get("collection_task_count")
            ),
            "task_source_family_counts": dict(
                _mapping(metadata.get("task_source_family_counts"))
            ),
            "preferred_source_family_counts": dict(
                _mapping(metadata.get("preferred_source_family_counts"))
            ),
            "batch_count": _int_or_zero(metadata.get("batch_count")),
            "ready_command_count": _int_or_zero(metadata.get("ready_command_count")),
            "missing_command_count": _int_or_zero(metadata.get("missing_command_count")),
            "source_backed_batch_count": _int_or_zero(
                metadata.get("source_backed_batch_count")
            ),
            "rule_only_batch_count": _int_or_zero(metadata.get("rule_only_batch_count")),
            "command_status_counts": dict(_mapping(metadata.get("command_status_counts"))),
            "request_type_counts": dict(_mapping(metadata.get("request_type_counts"))),
            "lane_counts": dict(_mapping(metadata.get("lane_counts"))),
            "query_sweep_failure_reason_counts": dict(
                _mapping(metadata.get("query_sweep_failure_reason_counts"))
            ),
            "query_sweep_recommended_next_action_counts": dict(
                _mapping(metadata.get("query_sweep_recommended_next_action_counts"))
            ),
            "query_sweep_no_hit_strategy_count": _int_or_zero(
                metadata.get("query_sweep_no_hit_strategy_count")
            ),
            "query_sweep_target_route_not_selected_strategy_count": _int_or_zero(
                metadata.get("query_sweep_target_route_not_selected_strategy_count")
            ),
            "query_sweep_blind_refuted_rate_below_min_strategy_count": _int_or_zero(
                metadata.get("query_sweep_blind_refuted_rate_below_min_strategy_count")
            ),
            "query_sweep_verified_false_alarm_above_max_strategy_count": _int_or_zero(
                metadata.get("query_sweep_verified_false_alarm_above_max_strategy_count")
            ),
            "remaining_rule_family_counts": dict(
                _mapping(metadata.get("remaining_rule_family_counts"))
            ),
            "promoted_rule_request_ids": _string_tuple(
                metadata.get("promoted_rule_request_ids", ())
            ),
        },
    }


def _missing_inputs(
    *,
    required_inputs: Sequence[str],
    placeholder_count: int,
    command_templates: Sequence[str],
) -> tuple[str, ...]:
    missing = list(required_inputs)
    if placeholder_count > 0:
        missing.append("bound_command_template_values")
    if not command_templates:
        return ()
    return tuple(dict.fromkeys(str(item) for item in missing if str(item)))


def _summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("command_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    action_ids = tuple(str(entry.get("action_id") or "") for entry in entries)
    route_counts: dict[str, int] = {}
    for entry in entries:
        for route in _string_tuple(entry.get("evidence_routes", ())):
            route_counts[route] = route_counts.get(route, 0) + 1
    return {
        "entry_count": len(entries),
        "ready_entry_count": status_counts.get("ready", 0),
        "needs_input_entry_count": status_counts.get("needs_inputs", 0),
        "missing_command_template_count": status_counts.get("missing_command_templates", 0),
        "command_count": sum(
            len(_string_tuple(entry.get("command_templates", ()))) for entry in entries
        ),
        "placeholder_count": sum(
            int(_mapping(entry.get("command_summary")).get("placeholder_count", 0))
            for entry in entries
        ),
        "missing_input_count": sum(
            len(_string_tuple(entry.get("missing_inputs", ()))) for entry in entries
        ),
        "planned_output_count": sum(
            len(_mapping_sequence(entry.get("planned_outputs", ()))) for entry in entries
        ),
        "action_ids": action_ids,
        "evidence_route_counts": dict(sorted(route_counts.items())),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if _int_or_zero(summary.get("entry_count")) == 0:
        return "empty"
    if _int_or_zero(summary.get("missing_command_template_count")) > 0:
        return "needs_commands"
    if _int_or_zero(summary.get("needs_input_entry_count")) > 0:
        return "needs_inputs"
    return "ready"


def _plan_root(
    *,
    source_path: Path | None,
    output_path: Path | None,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if output_path is not None:
        return output_path.parent / "frontier-research-queue-commands"
    if source_path is not None:
        return source_path.parent / "frontier-research-queue-commands"
    return ROOT / "artifacts" / "frontier-research-queue-commands"


def _workflow_keys(metadata: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in metadata.items()
        if key.endswith("_workflow") and isinstance(value, str) and value
    }


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path | None,
    source_path: Path | None,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> Mapping[str, Any]:
    artifacts = {
        name: path
        for name, path in {
            "frontier_research_queue_command_plan": output_path,
            "source": source_path,
        }.items()
        if path is not None
    }
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "plan_frontier_research_queue_commands",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "entry_count": _nested(payload, "summary", "entry_count"),
            "command_count": _nested(payload, "summary", "command_count"),
            "missing_input_count": _nested(payload, "summary", "missing_input_count"),
            **dict(metadata),
        },
    )
    _write_json(manifest_path, manifest, compact=compact)
    return manifest


def _load_mapping_source(source: str | Path | Mapping[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    if isinstance(source, Mapping):
        return None, dict(source)
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return path, dict(payload)


def _load_optional_json(path: Path | None) -> Mapping[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, Mapping) else {}


def _summary_path_tuple(
    payload: Mapping[str, Any],
    plural_key: str,
    singular_key: str,
    *,
    source_path: Path | None,
) -> tuple[Path, ...]:
    values = _string_tuple(_nested(payload, "paths", plural_key))
    if not values:
        values = _string_tuple(_nested(payload, "paths", singular_key))
    return tuple(
        path
        for path in (
            _resolve_path(value, base=source_path)
            for value in values
        )
        if path is not None
    )


def _paired_paths(
    left: Sequence[Path],
    right: Sequence[Path],
) -> tuple[tuple[Path | None, Path | None], ...]:
    if not left and not right:
        return ((None, None),)
    if not left:
        return tuple((None, value) for value in right)
    if not right:
        return tuple((value, None) for value in left)
    if len(left) == len(right):
        return tuple(zip(left, right, strict=True))
    if len(right) == 1:
        return tuple((value, right[0]) for value in left)
    if len(left) == 1:
        return tuple((left[0], value) for value in right)
    return tuple((left_value, right_value) for left_value, right_value in zip(left, right, strict=False))


def _artifact_path(
    artifacts: Mapping[str, Any],
    key: str,
    *,
    base: Path | None,
) -> Path | None:
    entry = _mapping(artifacts.get(key))
    return _resolve_path(entry.get("path"), base=base)


def _resolve_path(value: Any, *, base: Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    if base is None:
        return candidate
    root = base.parent if base.suffix else base
    return root / candidate


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence):
        return (str(value),) if str(value) else ()
    return tuple(str(item) for item in value if str(item))


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value)).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "item"


def _shell_join(parts: Sequence[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part))


def _append_config_value(parts: list[Any], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        return
    text = str(value)
    if not text:
        return
    parts.extend((flag, text))


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="frontier status report or evidence-gap plan")
    parser.add_argument("--json", default=None, help="optional output JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional local artifact registry JSON")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--output-dir", default=None, help="planned output directory root")
    parser.add_argument("--include-action-id", action="append", default=[])
    parser.add_argument("--exclude-action-id", action="append", default=[])
    parser.add_argument(
        "--only-active-research-queue",
        action="store_true",
        help="when source is a frontier status report, skip closed/superseded research queues",
    )
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_frontier_research_queue_command_plan(
        source=args.source,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        output_dir=args.output_dir,
        include_action_ids=tuple(args.include_action_id or ()),
        exclude_action_ids=tuple(args.exclude_action_id or ()),
        only_active_research_queue=bool(args.only_active_research_queue),
        compact_json=bool(args.compact_json),
    )
    if args.json is None:
        print(strict_json_dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
