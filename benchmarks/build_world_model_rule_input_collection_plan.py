"""Build typed input-collection tasks for world-model rule stubs.

This workflow consumes the non-evidence input requests emitted by
``run_world_model_rule_authoring_adapter.py``. It does not execute rules and
does not copy answers into rule inputs; it only lowers each missing-input row
into an auditable task contract that later collection adapters or a human
review step can fill explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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

WORKFLOW = "world_model_rule_input_collection_plan"
SOURCE_WORKFLOW = "world_model_rule_authoring_adapter"
RESERVED_ADAPTER_FIELDS = {"answer", "model_answer", "label", "labels", "is_false", "score_label"}
PRIORITY_BASE = {"high": 100, "medium": 50, "low": 10}
RULE_FAMILY_RANK = {
    "entity_disambiguation": 0,
    "quantity_or_arithmetic": 1,
    "temporal_consistency": 2,
    "causal_or_procedural": 3,
}
COLLECTION_FAMILY_BY_RULE_FAMILY = {
    "quantity_or_arithmetic": "numeric_rule_input_collection",
    "entity_disambiguation": "entity_role_rule_input_collection",
    "causal_or_procedural": "mechanism_rule_input_collection",
    "temporal_consistency": "temporal_snapshot_rule_input_collection",
}
EXECUTION_FIELDS_BY_RULE_FAMILY = {
    "quantity_or_arithmetic": (
        "numeric_value",
        "unit",
        "reference_time",
        "calculation.expression",
        "calculation.expected",
        "source_citation",
    ),
    "entity_disambiguation": (
        "subject_entity",
        "answer_entity",
        "expected_entity",
        "requested_role",
        "source_citation",
    ),
    "causal_or_procedural": ("mechanism", "precondition", "source_citation"),
    "temporal_consistency": ("claim_time", "source_time", "retrieved_at", "source_citation"),
}
FIELD_HINTS = {
    "numeric_value": (
        "Collect the grounded numeric value needed by the rule; include units and source provenance.",
        ("source_family_structured_fact", "external_citation_search"),
    ),
    "unit": (
        "Normalize the measurement unit or denominator used by the numeric value.",
        ("source_family_structured_fact", "external_citation_search"),
    ),
    "reference_time": (
        "Capture the time period for which the numeric value is valid.",
        ("source_family_structured_fact", "external_citation_search"),
    ),
    "calculation.expression": (
        "Author a deterministic arithmetic expression executable by CalculatorVerifier.",
        ("calculator_rule_authoring",),
    ),
    "calculation.expected": (
        "Author the expected result used to compare the deterministic arithmetic expression.",
        ("calculator_rule_authoring",),
    ),
    "subject_entity": (
        "Identify the entity in the question whose role or property is being queried.",
        ("entity_resolution", "source_family_fact_disambiguation"),
    ),
    "answer_entity": (
        "Bind the candidate answer entity from the claim trace; do not infer it from labels.",
        ("claim_trace_binding", "entity_resolution"),
    ),
    "expected_entity": (
        "Collect the provenance-backed entity that should satisfy the requested role.",
        ("source_family_fact_disambiguation", "external_citation_search"),
    ),
    "requested_role": (
        "Normalize the role, relation, or property requested by the question.",
        ("entity_resolution", "source_family_fact_disambiguation"),
    ),
    "mechanism": (
        "Collect a source-backed mechanism or procedure statement for the causal/procedural claim.",
        ("external_citation_search", "world_model_rule_authoring"),
    ),
    "precondition": (
        "Collect the condition under which the mechanism or procedure should apply.",
        ("external_citation_search", "world_model_rule_authoring"),
    ),
    "claim_time": (
        "Capture the claim's asserted time or validity window.",
        ("claim_trace_binding", "temporal_snapshot"),
    ),
    "source_time": (
        "Capture the evidence source's publication or observation time.",
        ("external_citation_search", "temporal_snapshot"),
    ),
    "retrieved_at": (
        "Capture the retrieval timestamp for time-sensitive evidence.",
        ("external_citation_search", "temporal_snapshot"),
    ),
    "source_citation": (
        "Attach a provenance-bearing citation before any rule result can be promoted.",
        ("external_citation_search",),
    ),
}


def build_world_model_rule_input_collection_plan(
    *,
    input_requests: Sequence[Mapping[str, Any]],
    rule_families: Sequence[str] = (),
    priorities: Sequence[str] = (),
    max_tasks: int | None = None,
    max_tasks_per_batch: int = 50,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready rule-input collection plan."""
    selected_families = _normalize_optional_filter(rule_families)
    selected_priorities = _normalize_optional_filter(priorities)
    if max_tasks is not None and int(max_tasks) <= 0:
        raise ValueError("max_tasks must be positive when provided.")
    if int(max_tasks_per_batch) <= 0:
        raise ValueError("max_tasks_per_batch must be positive.")

    tasks: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for ordinal, request in enumerate(input_requests, start=1):
        family = str(request.get("rule_family") or "")
        priority = str(request.get("priority") or "")
        request_id = str(request.get("request_id") or f"input-request-{ordinal}")
        if selected_families and family not in selected_families:
            skipped.append(_skip(request, request_id=request_id, reason="rule_family_filtered"))
            continue
        if selected_priorities and priority not in selected_priorities:
            skipped.append(_skip(request, request_id=request_id, reason="priority_filtered"))
            continue
        tasks.append(_input_task(request, ordinal=ordinal))

    tasks.sort(key=_task_sort_key)
    if max_tasks is not None:
        keep_ids = {item["task_id"] for item in tasks[: int(max_tasks)]}
        skipped.extend(
            _skip(item, request_id=str(item["source_request_id"]), reason="outside_max_tasks")
            for item in tasks
            if item["task_id"] not in keep_ids
        )
        tasks = [item for item in tasks if item["task_id"] in keep_ids]

    batches = _batches(tasks, max_tasks_per_batch=int(max_tasks_per_batch))
    summary = _summary(tasks=tasks, batches=batches, skipped=skipped, source_count=len(input_requests))
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready_for_input_collection" if tasks else "empty",
        "scope": (
            "Typed collection plan for deterministic rule inputs. Tasks are "
            "not verifier evidence and do not execute rule checks; they define "
            "the explicit fields required before the rule-authoring adapter can "
            "produce candidate results."
        ),
        "source": {
            "input_request_workflow": SOURCE_WORKFLOW,
            "input_request_count": len(input_requests),
        },
        "label_usage": {
            "labels_used_for_input_planning": False,
            "answers_copied_to_rule_inputs": False,
            "model_answers_copied_to_rule_inputs": False,
            "input_tasks_are_verifier_evidence": False,
            "generated_rule_inputs_execute_adapter": False,
        },
        "config": {
            "rule_families": selected_families,
            "priorities": selected_priorities,
            "max_tasks": max_tasks,
            "max_tasks_per_batch": int(max_tasks_per_batch),
        },
        "summary": summary,
        "input_tasks": tuple(tasks),
        "execution_batches": tuple(batches),
        "skipped_requests": tuple(skipped),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    input_requests_path: str | Path,
    output_dir: str | Path,
    report_json_path: str | Path | None = None,
    input_tasks_path: str | Path | None = None,
    batches_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    rule_families: Sequence[str] = (),
    priorities: Sequence[str] = (),
    max_tasks: int | None = None,
    max_tasks_per_batch: int = 50,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register a rule-input plan."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "rule-input-collection-plan.json")
    tasks_path = Path(input_tasks_path or output / "rule-input-tasks.jsonl")
    batch_path = Path(batches_path or output / "rule-input-execution-batches.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    input_requests = _load_jsonl_mappings(input_requests_path)
    payload = build_world_model_rule_input_collection_plan(
        input_requests=input_requests,
        rule_families=rule_families,
        priorities=priorities,
        max_tasks=max_tasks,
        max_tasks_per_batch=max_tasks_per_batch,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["paths"] = {
        "input_requests": str(input_requests_path),
        "report": str(report_path),
        "input_tasks": str(tasks_path),
        "execution_batches": str(batch_path),
        "artifact_manifest": str(manifest_path),
    }

    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(tasks_path, payload["input_tasks"], compact=compact_json)
    _write_jsonl(batch_path, payload["execution_batches"], compact=compact_json)
    manifest = build_artifact_manifest(
        {
            "rule_input_collection_plan": report_path,
            "rule_input_tasks": tasks_path,
            "rule_input_execution_batches": batch_path,
            "rule_input_requests": Path(input_requests_path),
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "task_count": payload["summary"]["task_count"],
            "batch_count": payload["summary"]["batch_count"],
            **dict(metadata or {}),
        },
    )
    _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_path,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "task_count": payload["summary"]["task_count"],
                "batch_count": payload["summary"]["batch_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _input_task(request: Mapping[str, Any], *, ordinal: int) -> dict[str, Any]:
    family = str(request.get("rule_family") or "")
    requested_inputs = _requested_inputs(request)
    execution_fields = _execution_fields(family=family, requested_inputs=requested_inputs)
    collection_family = COLLECTION_FAMILY_BY_RULE_FAMILY.get(family, "world_model_rule_input_collection")
    request_id = str(request.get("request_id") or f"input-request-{ordinal}")
    output = {
        "task_id": f"rule-input-task-{ordinal:04d}",
        "source_request_id": request_id,
        "target_id": str(request.get("target_id") or ""),
        "rule_family": family,
        "adapter": str(request.get("adapter") or collection_family),
        "collection_family": collection_family,
        "priority": str(request.get("priority") or ""),
        "priority_score": _priority_score(request),
        "question": str(request.get("question") or ""),
        "question_type": str(request.get("question_type") or ""),
        "gap_type": str(request.get("gap_type") or ""),
        "required_inputs": tuple(str(item) for item in _sequence(request.get("required_inputs"))),
        "missing_inputs": tuple(str(item) for item in _sequence(request.get("missing_inputs"))),
        "execution_inputs": execution_fields,
        "field_tasks": tuple(_field_task(field) for field in execution_fields),
        "output_contract": _output_contract(family=family, request_id=request_id, execution_fields=execution_fields),
        "source_policy": {
            "not_verifier_evidence": True,
            "requires_explicit_rule_input_values": True,
            "requires_promotion_gate_after_adapter_execution": True,
            "do_not_copy_labels_or_model_answers": True,
        },
        "not_verifier_evidence": True,
        "metadata": {
            "source_workflow": SOURCE_WORKFLOW,
            "queue_workflow": WORKFLOW,
            "source_request_id": request_id,
        },
    }
    return {key: value for key, value in output.items() if key not in RESERVED_ADAPTER_FIELDS}


def _field_task(field: str) -> dict[str, Any]:
    hint, providers = FIELD_HINTS.get(
        field,
        ("Collect the explicit rule input value with provenance.", ("manual_rule_input_collection",)),
    )
    return {
        "field": field,
        "collection_hint": hint,
        "recommended_adapters": tuple(providers),
        "requires_provenance": True,
    }


def _output_contract(
    *,
    family: str,
    request_id: str,
    execution_fields: Sequence[str],
) -> dict[str, Any]:
    return {
        "rule_input_request_id": request_id,
        "rule_inputs_format": "JSON object keyed by request_id or JSONL rows with request_id",
        "accepted_execution_inputs": tuple(execution_fields),
        "adapter_execution": "candidate_only_requires_promotion_gate",
        "family_hint": family,
    }


def _requested_inputs(request: Mapping[str, Any]) -> tuple[str, ...]:
    missing = tuple(str(item) for item in _sequence(request.get("missing_inputs")) if str(item))
    if missing:
        return missing
    return tuple(str(item) for item in _sequence(request.get("required_inputs")) if str(item))


def _execution_fields(*, family: str, requested_inputs: Sequence[str]) -> tuple[str, ...]:
    configured = EXECUTION_FIELDS_BY_RULE_FAMILY.get(family, ())
    if configured:
        return tuple(dict.fromkeys((*requested_inputs, *configured)))
    return tuple(dict.fromkeys(requested_inputs))


def _batches(
    tasks: Sequence[Mapping[str, Any]],
    *,
    max_tasks_per_batch: int,
) -> tuple[dict[str, Any], ...]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[str(task.get("collection_family") or "world_model_rule_input_collection")].append(task)
    batches: list[dict[str, Any]] = []
    ordinal = 1
    for collection_family, rows in sorted(grouped.items(), key=_batch_group_sort_key):
        sorted_rows = sorted(rows, key=_task_sort_key)
        for offset in range(0, len(sorted_rows), max_tasks_per_batch):
            chunk = sorted_rows[offset : offset + max_tasks_per_batch]
            fields = tuple(dict.fromkeys(field for row in chunk for field in _sequence(row.get("execution_inputs"))))
            adapters = tuple(
                dict.fromkeys(
                    provider
                    for row in chunk
                    for task in _mapping_sequence(row.get("field_tasks"))
                    for provider in _sequence(task.get("recommended_adapters"))
                )
            )
            batches.append({
                "batch_id": f"rule-input-batch-{ordinal:04d}",
                "collection_family": collection_family,
                "rule_families": tuple(dict.fromkeys(str(item.get("rule_family")) for item in chunk)),
                "task_count": len(chunk),
                "target_count": len({str(item.get("target_id")) for item in chunk if str(item.get("target_id"))}),
                "source_request_ids": tuple(str(item.get("source_request_id")) for item in chunk),
                "execution_inputs": fields,
                "recommended_adapters": adapters,
                "not_verifier_evidence": True,
            })
            ordinal += 1
    return tuple(batches)


def _summary(
    *,
    tasks: Sequence[Mapping[str, Any]],
    batches: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
    source_count: int,
) -> dict[str, Any]:
    family_counts = Counter(str(item.get("rule_family") or "") for item in tasks)
    collection_counts = Counter(str(item.get("collection_family") or "") for item in tasks)
    priority_counts = Counter(str(item.get("priority") or "") for item in tasks)
    adapter_counts = Counter(str(item.get("adapter") or "") for item in tasks)
    requested_counts: Counter[str] = Counter()
    execution_counts: Counter[str] = Counter()
    for task in tasks:
        for item in _sequence(task.get("missing_inputs")):
            requested_counts[str(item)] += 1
        for item in _sequence(task.get("execution_inputs")):
            execution_counts[str(item)] += 1
    return {
        "source_input_request_count": int(source_count),
        "task_count": len(tasks),
        "batch_count": len(batches),
        "skipped_request_count": len(skipped),
        "rule_family_counts": _sorted_counter(family_counts),
        "collection_family_counts": _sorted_counter(collection_counts),
        "priority_counts": _sorted_counter(priority_counts),
        "adapter_counts": _sorted_counter(adapter_counts),
        "missing_input_counts": _sorted_counter(requested_counts),
        "execution_input_counts": _sorted_counter(execution_counts),
        "skipped_request_counts": _sorted_counter(Counter(str(item.get("reason") or "") for item in skipped)),
        "top_task": None
        if not tasks
        else {
            "task_id": tasks[0]["task_id"],
            "source_request_id": tasks[0]["source_request_id"],
            "collection_family": tasks[0]["collection_family"],
            "priority_score": tasks[0]["priority_score"],
        },
    }


def _skip(request: Mapping[str, Any], *, request_id: str, reason: str) -> dict[str, Any]:
    return {
        "source_request_id": request_id,
        "target_id": str(request.get("target_id") or ""),
        "rule_family": str(request.get("rule_family") or ""),
        "priority": str(request.get("priority") or ""),
        "reason": reason,
        "not_verifier_evidence": True,
    }


def _task_sort_key(item: Mapping[str, Any]) -> tuple[float, int, str, str]:
    return (
        -float(item.get("priority_score") or 0.0),
        RULE_FAMILY_RANK.get(str(item.get("rule_family") or ""), 100),
        str(item.get("target_id") or ""),
        str(item.get("source_request_id") or item.get("task_id") or ""),
    )


def _batch_group_sort_key(item: tuple[str, Sequence[Mapping[str, Any]]]) -> tuple[int, str]:
    family = str(item[1][0].get("rule_family") or "") if item[1] else ""
    return (RULE_FAMILY_RANK.get(family, 100), item[0])


def _priority_score(request: Mapping[str, Any]) -> float:
    priority = str(request.get("priority") or "").casefold()
    family = str(request.get("rule_family") or "")
    return float(PRIORITY_BASE.get(priority, 1) + (100 - RULE_FAMILY_RANK.get(family, 50)) / 100.0)


def _load_jsonl_mappings(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append({key: value for key, value in dict(row).items() if key not in RESERVED_ADAPTER_FIELDS})
    return tuple(rows)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = "".join(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    else:
        text = "".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(value)
    return ()


def _normalize_optional_filter(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(((key, value) for key, value in counter.items() if key), key=lambda item: (-item[1], item[0])))


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-requests", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--input-tasks-jsonl", default=None)
    parser.add_argument("--batches-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--rule-family", action="append", default=[])
    parser.add_argument("--priority", action="append", default=[])
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-tasks-per-batch", type=int, default=50)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        input_requests_path=args.input_requests,
        output_dir=args.output_dir,
        report_json_path=args.json,
        input_tasks_path=args.input_tasks_jsonl,
        batches_path=args.batches_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        rule_families=args.rule_family or (),
        priorities=args.priority or (),
        max_tasks=args.max_tasks,
        max_tasks_per_batch=args.max_tasks_per_batch,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "world_model_rule_input_collection_plan_ok "
        f"status={payload['status']} "
        f"tasks={summary['task_count']} "
        f"batches={summary['batch_count']} "
        f"source_requests={summary['source_input_request_count']}"
    )


if __name__ == "__main__":
    main()
