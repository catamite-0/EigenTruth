"""Plan collection work for unbound frontier research-queue inputs.

This workflow consumes a ``frontier_research_queue_bound_command_plan`` and
turns remaining unbound source-backed inputs into explicit collection
requests. It does not execute commands, approve bindings, fetch evidence, or
turn any request into verifier evidence.
"""

from __future__ import annotations

import argparse
import json
import shlex
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

from benchmarks.frontier_research_command_requirements import (  # noqa: E402
    REQUIRED_INPUT_FLAGS,
    frontier_command_script,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

BOUND_PLAN_WORKFLOW = "frontier_research_queue_bound_command_plan"
WORKFLOW = "frontier_research_queue_input_collection_plan"

SOURCE_BACKED_CONTRACTS: dict[str, dict[str, Any]] = {
    "source_backed_numeric_bindings": {
        "lane": "world_model_rules",
        "collection_family": "numeric_rule_input_binding_collection",
        "rule_family": "quantity_or_arithmetic",
        "recommended_next_tools": (
            "benchmarks/fill_world_model_rule_inputs_from_numeric_bindings.py",
            "benchmarks/plan_world_model_rule_numeric_subject_bindings.py",
        ),
        "target_flag": "--numeric-bindings",
        "required_binding_fields": (
            "request_id",
            "target_id",
            "subject_entity",
            "candidate_numeric_value",
            "source_numeric_value",
            "unit",
            "reference_time",
            "source_citation",
            "review_status",
            "not_verifier_evidence",
        ),
        "binding_skeleton": {
            "request_id": "",
            "target_id": "",
            "subject_entity": "",
            "candidate_numeric_value": "",
            "source_numeric_value": "",
            "unit": "",
            "reference_time": "",
            "source_citation": "",
            "review_status": "approved",
            "not_verifier_evidence": True,
        },
    },
    "source_backed_subject_bindings": {
        "lane": "world_model_rules",
        "collection_family": "numeric_subject_binding_collection",
        "rule_family": "quantity_or_arithmetic",
        "recommended_next_tools": (
            "benchmarks/plan_world_model_rule_numeric_subject_bindings.py",
            "benchmarks/fill_world_model_rule_inputs_from_numeric_bindings.py",
        ),
        "target_flag": "--subject-bindings",
        "required_binding_fields": (
            "request_id",
            "target_id",
            "subject_entity",
            "source_citation",
            "review_status",
            "not_verifier_evidence",
        ),
        "binding_skeleton": {
            "request_id": "",
            "target_id": "",
            "subject_entity": "",
            "source_citation": "",
            "review_status": "approved",
            "not_verifier_evidence": True,
        },
    },
    "source_backed_temporal_bindings": {
        "lane": "world_model_rules",
        "collection_family": "temporal_binding_collection",
        "rule_family": "temporal_consistency",
        "recommended_next_tools": (
            "benchmarks/plan_world_model_rule_temporal_bindings.py",
            "benchmarks/fill_world_model_rule_inputs_from_temporal_bindings.py",
        ),
        "target_flag": "--temporal-bindings",
        "required_binding_fields": (
            "request_id",
            "target_id",
            "claim_time",
            "source_time",
            "retrieved_at",
            "source_citation",
            "review_status",
            "not_verifier_evidence",
        ),
        "binding_skeleton": {
            "request_id": "",
            "target_id": "",
            "claim_time": "",
            "source_time": "",
            "retrieved_at": "",
            "source_citation": "",
            "review_status": "approved",
            "not_verifier_evidence": True,
        },
    },
    "source_backed_mechanism_bindings": {
        "lane": "world_model_rules",
        "collection_family": "mechanism_rule_input_collection",
        "rule_family": "causal_or_procedural",
        "recommended_next_tools": (
            "benchmarks/fill_world_model_rule_inputs_from_mechanism_bindings.py",
        ),
        "target_flag": "--mechanism-bindings",
        "required_binding_fields": (
            "request_id",
            "target_id",
            "mechanism",
            "precondition",
            "mechanism_status",
            "source_citation",
            "review_status",
            "not_verifier_evidence",
        ),
        "binding_skeleton": {
            "request_id": "",
            "target_id": "",
            "mechanism": "",
            "precondition": "",
            "mechanism_status": "",
            "source_citation": "",
            "review_status": "approved",
            "not_verifier_evidence": True,
        },
    },
    "source_backed_entity_bindings": {
        "lane": "world_model_rules",
        "collection_family": "entity_role_rule_input_binding_collection",
        "rule_family": "entity_disambiguation",
        "recommended_next_tools": (
            "benchmarks/plan_world_model_rule_entity_bindings.py",
            "benchmarks/collect_world_model_rule_entity_bindings_from_citation_corpus.py",
            "benchmarks/review_world_model_rule_entity_binding_candidates.py",
            "benchmarks/promote_world_model_rule_entity_binding_candidates.py",
            "benchmarks/fill_world_model_rule_inputs_from_entity_bindings.py",
        ),
        "target_flag": "--entity-bindings",
        "required_binding_fields": (
            "request_id",
            "target_id",
            "subject_entity",
            "answer_entity",
            "expected_entity",
            "requested_role",
            "source_citation",
            "review_status",
            "not_verifier_evidence",
        ),
        "binding_skeleton": {
            "request_id": "",
            "target_id": "",
            "subject_entity": "",
            "answer_entity": "",
            "expected_entity": "",
            "requested_role": "",
            "source_citation": "",
            "review_status": "approved",
            "not_verifier_evidence": True,
        },
    },
}

REVIEW_CONTRACTS: dict[str, dict[str, Any]] = {
    "bound_command_template_values": {
        "review_family": "command_template_binding_review",
        "reason": "unmapped_command_template_placeholders",
        "recommended_next_steps": (
            "Review remaining command placeholders and either bind them explicitly "
            "or rerun binding staging with additional reviewed inputs.",
        ),
    },
    "valid_bound_commands": {
        "review_family": "command_validation_review",
        "reason": "known_command_requirements_not_satisfied",
        "recommended_next_steps": (
            "Inspect command_validation.issues and add the missing CLI flags or "
            "reviewed input bindings before execution.",
        ),
    },
    "command_templates": {
        "review_family": "missing_command_template_review",
        "reason": "frontier_action_has_no_command_templates",
        "recommended_next_steps": (
            "Add reviewed command templates for this frontier action before binding or execution.",
        ),
    },
}


def plan_frontier_research_queue_input_collection(
    *,
    bound_command_plan: str | Path | Mapping[str, Any],
    json_path: str | Path | None = None,
    collection_requests_path: str | Path | None = None,
    review_requests_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready plan for collecting unbound frontier inputs."""
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    source_path, plan = _load_mapping_source(bound_command_plan)
    if plan.get("workflow") != BOUND_PLAN_WORKFLOW:
        raise ValueError(f"bound_command_plan must have workflow={BOUND_PLAN_WORKFLOW!r}.")
    output_path = None if json_path is None else Path(json_path)
    collection_path = None if collection_requests_path is None else Path(collection_requests_path)
    review_path = None if review_requests_path is None else Path(review_requests_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    if registry_path is not None and output_path is None and source_path is None:
        raise ValueError("registry_path requires json_path when bound_command_plan is in-memory.")

    entries = tuple(_mapping_sequence(plan.get("entries", ())))
    collection_requests: list[dict[str, Any]] = []
    review_requests: list[dict[str, Any]] = []
    covered_inputs: list[dict[str, Any]] = []
    for entry in entries:
        entry_collection, entry_review, entry_covered = _entry_requests(entry)
        collection_requests.extend(entry_collection)
        review_requests.extend(entry_review)
        covered_inputs.extend(entry_covered)

    summary = _summary(
        entries=entries,
        collection_requests=collection_requests,
        review_requests=review_requests,
        covered_inputs=covered_inputs,
    )
    status = _status(summary)
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Preflight collection plan for unbound frontier research-queue "
            "inputs. Requests are not verifier evidence and do not approve or "
            "execute the bound command plan."
        ),
        "source": {
            "bound_command_plan": None if source_path is None else str(source_path),
            "bound_plan_workflow": plan.get("workflow"),
            "bound_plan_status": plan.get("status"),
            "bound_plan_summary": _mapping(plan.get("summary")),
        },
        "label_usage": {
            "labels_used_for_collection_planning": False,
            "labels_copied_to_collection_requests": False,
            "model_answers_copied_to_collection_requests": False,
            "collection_requests_are_verifier_evidence": False,
            "planner_approves_bindings": False,
            "planner_executes_commands": False,
        },
        "config": {
            "source_backed_contracts": tuple(SOURCE_BACKED_CONTRACTS),
            "review_contracts": tuple(REVIEW_CONTRACTS),
        },
        "summary": summary,
        "collection_requests": tuple(collection_requests),
        "review_requests": tuple(review_requests),
        "covered_inputs": tuple(covered_inputs),
        "paths": {
            "input_collection_plan": None if output_path is None else str(output_path),
            "collection_requests": None if collection_path is None else str(collection_path),
            "review_requests": None if review_path is None else str(review_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
        },
        "metadata": dict(metadata or {}),
    }
    if output_path is not None:
        _write_json(output_path, payload, compact=compact_json)
    if collection_path is not None:
        _write_jsonl(collection_path, collection_requests, compact=compact_json)
    if review_path is not None:
        _write_jsonl(review_path, review_requests, compact=compact_json)
    manifest = None
    if manifest_path is not None:
        manifest = _write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            collection_path=collection_path,
            review_path=review_path,
            source_path=source_path,
            payload=payload,
            metadata=metadata or {},
            compact=compact_json,
        )
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output_path if output_path is not None else source_path,
            metadata={
                "workflow": WORKFLOW,
                "status": status,
                "bound_command_plan": None if source_path is None else str(source_path),
                "artifact_manifest": None if manifest_path is None else str(manifest_path),
                "collection_request_count": summary["collection_request_count"],
                "review_request_count": summary["review_request_count"],
                "source_backed_request_count": summary["source_backed_request_count"],
                "covered_input_count": summary["covered_input_count"],
                "actionable_entry_count": summary["actionable_entry_count"],
                "manifest_summary": {} if manifest is None else manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _entry_requests(
    entry: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    action_id = str(entry.get("action_id") or entry.get("entry_id") or "frontier-action")
    unbound_inputs = _string_tuple(entry.get("unbound_inputs", ()))
    placeholders = _blocking_placeholders(entry)
    placeholders_by_input, assigned_placeholder_ids = _placeholders_by_input(
        entry,
        placeholders=placeholders,
    )
    collection_requests: list[dict[str, Any]] = []
    review_requests: list[dict[str, Any]] = []
    covered_inputs: list[dict[str, Any]] = []
    for input_name in unbound_inputs:
        if input_name in SOURCE_BACKED_CONTRACTS or input_name.startswith("source_backed_"):
            collection_requests.append(
                _collection_request(
                    entry,
                    input_name=input_name,
                    action_id=action_id,
                    blocking_placeholders=placeholders_by_input.get(input_name, ()),
                )
            )
            continue
        if input_name == "bound_command_template_values":
            unmapped = tuple(
                placeholder
                for placeholder in placeholders
                if _placeholder_id(placeholder) not in assigned_placeholder_ids
            )
            if unmapped:
                review_requests.append(
                    _review_request(
                        entry,
                        input_name=input_name,
                        action_id=action_id,
                        blocking_placeholders=unmapped,
                    )
                )
            else:
                covered_inputs.append(
                    _covered_input(
                        entry,
                        input_name=input_name,
                        action_id=action_id,
                        blocking_placeholders=placeholders,
                    )
                )
            continue
        review_requests.append(
            _review_request(
                entry,
                input_name=input_name,
                action_id=action_id,
                blocking_placeholders=placeholders_by_input.get(input_name, ()),
            )
        )
    if (
        not _string_tuple(entry.get("bound_commands", ()))
        and str(entry.get("command_status") or "") == "missing_command_templates"
    ):
        review_requests.append(
            _review_request(
                entry,
                input_name="command_templates",
                action_id=action_id,
                blocking_placeholders=(),
            )
        )
    return tuple(collection_requests), tuple(review_requests), tuple(covered_inputs)


def _collection_request(
    entry: Mapping[str, Any],
    *,
    input_name: str,
    action_id: str,
    blocking_placeholders: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = SOURCE_BACKED_CONTRACTS.get(input_name, _generic_source_backed_contract(input_name))
    request_id = f"{_slug(action_id)}:{_slug(input_name)}"
    request_metadata = _collection_request_metadata(entry, contract)
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "request_id": request_id,
        "action_id": action_id,
        "entry_id": str(entry.get("entry_id") or action_id),
        "title": str(entry.get("title") or action_id),
        "input_name": input_name,
        "input_category": "source_backed",
        "lane": str(contract.get("lane") or "frontier_research_queue"),
        "collection_family": str(contract.get("collection_family") or "source_backed_collection"),
        "target_flag": str(contract.get("target_flag") or ""),
        "recommended_next_tools": _string_tuple(contract.get("recommended_next_tools", ())),
        "required_binding_fields": _string_tuple(contract.get("required_binding_fields", ())),
        "recommended_binding_skeleton": _mapping(contract.get("binding_skeleton")),
        "blocking_placeholders": tuple(blocking_placeholders),
        "binding_review_status": str(entry.get("binding_review_status") or "untracked"),
        "command_status": str(entry.get("command_status") or "unknown"),
        "evidence_routes": _string_tuple(entry.get("evidence_routes", ())),
        "source_gap_ids": _string_tuple(entry.get("source_gap_ids", ())),
        "metadata": request_metadata,
        "review_required": True,
        "not_verifier_evidence": True,
        "blocks_bound_command_execution": True,
    }


def _collection_request_metadata(
    entry: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    entry_metadata = _mapping(entry.get("metadata"))
    remaining_counts = _int_mapping(entry_metadata.get("remaining_rule_family_counts"))
    target_rule_family = str(contract.get("rule_family") or "")
    metadata: dict[str, Any] = {
        "target_rule_family": target_rule_family,
        "remaining_rule_family_counts": remaining_counts,
        "promoted_rule_request_ids": _string_tuple(
            entry_metadata.get("promoted_rule_request_ids", ())
        ),
    }
    if target_rule_family and target_rule_family in remaining_counts:
        metadata["target_remaining_rule_count"] = int(remaining_counts[target_rule_family])
    return metadata


def _review_request(
    entry: Mapping[str, Any],
    *,
    input_name: str,
    action_id: str,
    blocking_placeholders: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = REVIEW_CONTRACTS.get(input_name, _generic_review_contract(input_name))
    request_id = f"{_slug(action_id)}:{_slug(input_name)}"
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "request_id": request_id,
        "action_id": action_id,
        "entry_id": str(entry.get("entry_id") or action_id),
        "title": str(entry.get("title") or action_id),
        "input_name": input_name,
        "input_category": "review",
        "review_family": str(contract.get("review_family") or "manual_input_review"),
        "reason": str(contract.get("reason") or "unbound_input_requires_review"),
        "recommended_next_steps": _string_tuple(contract.get("recommended_next_steps", ())),
        "blocking_placeholders": tuple(blocking_placeholders),
        "binding_review_status": str(entry.get("binding_review_status") or "untracked"),
        "command_status": str(entry.get("command_status") or "unknown"),
        "command_validation": _mapping(entry.get("command_validation")),
        "evidence_routes": _string_tuple(entry.get("evidence_routes", ())),
        "source_gap_ids": _string_tuple(entry.get("source_gap_ids", ())),
        "review_required": True,
        "not_verifier_evidence": True,
        "blocks_bound_command_execution": True,
    }


def _covered_input(
    entry: Mapping[str, Any],
    *,
    input_name: str,
    action_id: str,
    blocking_placeholders: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "action_id": action_id,
        "entry_id": str(entry.get("entry_id") or action_id),
        "input_name": input_name,
        "coverage_reason": "remaining placeholders are represented by specific input requests",
        "blocking_placeholder_count": len(blocking_placeholders),
    }


def _blocking_placeholders(entry: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for command_index, command in enumerate(_string_tuple(entry.get("bound_commands", ())), start=1):
        tokens = _command_tokens(command)
        script = frontier_command_script(tokens)
        placeholder_index = 0
        for token_index, token in enumerate(tokens):
            if token != "...":
                continue
            placeholder_index += 1
            flag = _previous_flag(tokens, token_index)
            records.append({
                "command_index": command_index,
                "placeholder_index": placeholder_index,
                "token_index": token_index,
                "script": script,
                "flag": flag,
                "context": _context(tokens, token_index),
            })
    return tuple(records)


def _placeholders_by_input(
    entry: Mapping[str, Any],
    *,
    placeholders: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, tuple[Mapping[str, Any], ...]], set[tuple[int, int]]]:
    required = set(_string_tuple(entry.get("required_inputs", ())))
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    assigned_ids: set[tuple[int, int]] = set()
    for placeholder in placeholders:
        input_name = _input_name_for_placeholder(placeholder, required_inputs=required)
        if not input_name:
            continue
        grouped.setdefault(input_name, []).append(placeholder)
        assigned_ids.add(_placeholder_id(placeholder))
    return {key: tuple(value) for key, value in grouped.items()}, assigned_ids


def _input_name_for_placeholder(
    placeholder: Mapping[str, Any],
    *,
    required_inputs: set[str],
) -> str | None:
    script = str(placeholder.get("script") or "")
    flag = str(placeholder.get("flag") or "")
    if not script or not flag:
        return None
    for input_name, configured_flag in REQUIRED_INPUT_FLAGS.get(script, {}).items():
        if configured_flag == flag and (not required_inputs or input_name in required_inputs):
            return input_name
    return None


def _placeholder_id(placeholder: Mapping[str, Any]) -> tuple[int, int]:
    return (
        _int_or_zero(placeholder.get("command_index")),
        _int_or_zero(placeholder.get("placeholder_index")),
    )


def _summary(
    *,
    entries: Sequence[Mapping[str, Any]],
    collection_requests: Sequence[Mapping[str, Any]],
    review_requests: Sequence[Mapping[str, Any]],
    covered_inputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    input_counts = Counter()
    binding_review_status_counts = Counter()
    for entry in entries:
        for input_name in _string_tuple(entry.get("unbound_inputs", ())):
            input_counts[input_name] += 1
        binding_review_status_counts[str(entry.get("binding_review_status") or "untracked")] += 1
    collection_family_counts = Counter(
        str(item.get("collection_family") or "") for item in collection_requests
    )
    review_family_counts = Counter(str(item.get("review_family") or "") for item in review_requests)
    source_backed_requests = tuple(
        item for item in collection_requests if item.get("input_category") == "source_backed"
    )
    action_ids = tuple(
        dict.fromkeys(
            str(item.get("action_id") or "")
            for item in (*collection_requests, *review_requests, *covered_inputs)
            if str(item.get("action_id") or "")
        )
    )
    return {
        "entry_count": len(entries),
        "actionable_entry_count": len(action_ids),
        "unbound_input_count": sum(input_counts.values()),
        "collection_request_count": len(collection_requests),
        "review_request_count": len(review_requests),
        "source_backed_request_count": len(source_backed_requests),
        "covered_input_count": len(covered_inputs),
        "blocking_placeholder_count": sum(
            len(_mapping_sequence(item.get("blocking_placeholders", ())))
            for item in (*collection_requests, *review_requests)
        ),
        "input_counts": _sorted_counter(input_counts),
        "collection_family_counts": _sorted_counter(collection_family_counts),
        "review_family_counts": _sorted_counter(review_family_counts),
        "binding_review_status_counts": _sorted_counter(binding_review_status_counts),
        "action_ids": action_ids,
        "collection_request_ids": tuple(str(item.get("request_id") or "") for item in collection_requests),
        "review_request_ids": tuple(str(item.get("request_id") or "") for item in review_requests),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if _int_or_zero(summary.get("collection_request_count")) and _int_or_zero(
        summary.get("review_request_count")
    ):
        return "needs_collection_and_review"
    if _int_or_zero(summary.get("collection_request_count")):
        return "ready_for_collection"
    if _int_or_zero(summary.get("review_request_count")):
        return "needs_review"
    return "empty"


def _generic_source_backed_contract(input_name: str) -> dict[str, Any]:
    return {
        "lane": "frontier_research_queue",
        "collection_family": "source_backed_input_collection",
        "recommended_next_tools": (),
        "target_flag": "",
        "required_binding_fields": (
            "request_id",
            "target_id",
            "source_citation",
            "review_status",
            "not_verifier_evidence",
        ),
        "binding_skeleton": {
            "request_id": "",
            "target_id": "",
            "source_citation": "",
            "review_status": "approved",
            "not_verifier_evidence": True,
            "input_name": input_name,
        },
    }


def _generic_review_contract(input_name: str) -> dict[str, Any]:
    return {
        "review_family": "manual_input_review",
        "reason": "unbound_input_requires_review",
        "recommended_next_steps": (
            f"Provide a reviewed binding for {input_name!r} or remove it from the command plan.",
        ),
    }


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path | None,
    collection_path: Path | None,
    review_path: Path | None,
    source_path: Path | None,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> Mapping[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "frontier_research_queue_input_collection_plan": output_path,
        "bound_command_plan": source_path,
    }
    if collection_path is not None:
        artifacts["collection_requests"] = collection_path
    if review_path is not None:
        artifacts["review_requests"] = review_path
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "plan_frontier_research_queue_input_collection",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "collection_request_count": _nested(payload, "summary", "collection_request_count"),
            "review_request_count": _nested(payload, "summary", "review_request_count"),
            "source_backed_request_count": _nested(
                payload,
                "summary",
                "source_backed_request_count",
            ),
            "covered_input_count": _nested(payload, "summary", "covered_input_count"),
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


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    indent = None if compact else 2
    Path(path).write_text(
        strict_json_dumps(payload, indent=indent, sort_keys=True),
        encoding="utf-8",
    )


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [strict_json_dumps(row, indent=None if compact else None, sort_keys=True) for row in rows]
    Path(path).write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")


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
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return (str(value),) if str(value) else ()
    return tuple(str(item) for item in value if str(item))


def _command_tokens(command: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(command))
    except ValueError:
        return tuple(str(command).split())


def _previous_flag(tokens: Sequence[str], index: int) -> str | None:
    if index <= 0:
        return None
    previous = str(tokens[index - 1])
    return previous if previous.startswith("--") else None


def _context(tokens: Sequence[str], index: int) -> dict[str, Any]:
    start = max(0, index - 2)
    end = min(len(tokens), index + 3)
    return {
        "before": tuple(tokens[start:index]),
        "after": tuple(tokens[index + 1 : end]),
    }


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


def _int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): int(number)
        for key, number in value.items()
        if str(key) and not isinstance(number, bool) and _is_int_like(number)
    }


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((str(key), int(value)) for key, value in counter.items() if str(key)))


def _slug(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value))
    return "-".join(part for part in text.split("-") if part) or "item"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound-command-plan", required=True, help="bound frontier command plan JSON")
    parser.add_argument("--json", default=None, help="optional output JSON path")
    parser.add_argument("--collection-requests-jsonl", default=None)
    parser.add_argument("--review-requests-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional local artifact registry JSON")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = plan_frontier_research_queue_input_collection(
        bound_command_plan=args.bound_command_plan,
        json_path=args.json,
        collection_requests_path=args.collection_requests_jsonl,
        review_requests_path=args.review_requests_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        compact_json=bool(args.compact_json),
    )
    if args.json is None:
        print(strict_json_dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
