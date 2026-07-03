"""Run one or more source-family structured QA lane batches locally.

The lane execution queue is a scheduling artifact. This workflow materializes a
selected batch as a compatible fact-collection corpus, then reuses the existing
local source-family fact-collection workflow when source-backed requests are
present. Rule-only batches are emitted as rule-authoring stubs. It preserves the
boundary that adapter matches and rule stubs are candidate inputs only, not
verifier evidence.
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

from benchmarks.run_source_family_structured_qa_fact_collection_workflow import (  # noqa: E402
    RULE_REQUEST_TYPE,
    SOURCE_BACKED_REQUEST_TYPES,
    run_source_family_structured_qa_fact_collection_workflow,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "source_family_structured_qa_lane_batch_workflow"
LANE_QUEUE_WORKFLOW = "source_family_structured_qa_lane_execution_queue"
COLLECTION_WORKFLOW = "source_family_structured_qa_fact_collection_corpus"
RESERVED_REQUEST_FIELDS = {"answer", "model_answer", "label", "labels", "is_false", "score_label"}
CLOSURE_ROUTE_BY_REQUEST_TYPE = {
    "source_family_fact_disambiguation": "entity_disambiguation",
    "world_model_or_calculator_rule": "world_model_rule_authoring",
    "source_family_structured_fact": "property_or_indicator_collection",
    "entity_resolution": "entity_resolution",
    "external_citation": "citation_evidence_collection",
}


def run_source_family_structured_qa_lane_batch_workflow(
    *,
    lane_queue_path: str | Path,
    collection_corpus_path: str | Path,
    source_catalog_paths: Sequence[str | Path],
    output_dir: str | Path,
    batch_ids: Sequence[str],
    report_json_path: str | Path | None = None,
    batch_collection_corpus_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    adapter_max_results: int = 3,
    adapter_max_query_variants: int = 3,
    adapter_min_text_overlap: float = 0.05,
    adapter_diversify_source_families: bool = True,
    min_request_result_coverage: float = 1.0,
    default_source_family: str = "reference",
    keep_qid_values: bool = False,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
    fail_on_blocked: bool = False,
) -> dict[str, Any]:
    """Run selected lane batches through the local fact-collection workflow."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    selected_batch_ids = tuple(dict.fromkeys(str(item).strip() for item in batch_ids if str(item).strip()))
    if not selected_batch_ids:
        raise ValueError("at least one batch id is required.")
    if not (0.0 <= min_request_result_coverage <= 1.0):
        raise ValueError("min_request_result_coverage must be in [0, 1].")

    lane_queue = _load_lane_queue(lane_queue_path)
    collection = _load_collection(collection_corpus_path)
    selected_batches = _select_batches(lane_queue, selected_batch_ids)
    batch_collection = _batch_collection(
        lane_queue=lane_queue,
        collection=collection,
        selected_batches=selected_batches,
        batch_ids=selected_batch_ids,
        metadata=metadata,
    )
    source_backed_request_types = tuple(
        request_type
        for request_type in SOURCE_BACKED_REQUEST_TYPES
        if batch_collection["requests"].get(request_type)
    )
    if source_backed_request_types and not source_catalog_paths:
        raise ValueError("source_catalog_paths must contain at least one path for source-backed requests.")
    rule_stubs = _rule_stubs(batch_collection)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "lane-batch-workflow.json")
    collection_path = Path(batch_collection_corpus_path or output / "lane-batch-collection-corpus.json")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")
    child_dir = output / "fact-collection-workflow"
    rules_path = output / "world-model-rule-stubs.jsonl"
    _write_json(collection_path, batch_collection, compact=compact_json)
    if rule_stubs:
        _write_jsonl(rules_path, rule_stubs, compact=compact_json)

    child_payload: dict[str, Any] | None = None
    if source_backed_request_types:
        child_payload = run_source_family_structured_qa_fact_collection_workflow(
            collection_corpus_path=collection_path,
            source_catalog_paths=source_catalog_paths,
            output_dir=child_dir,
            request_types=source_backed_request_types,
            adapter_max_results=adapter_max_results,
            adapter_max_query_variants=adapter_max_query_variants,
            adapter_min_text_overlap=adapter_min_text_overlap,
            adapter_diversify_source_families=adapter_diversify_source_families,
            min_request_result_coverage=min_request_result_coverage,
            default_source_family=default_source_family,
            keep_qid_values=keep_qid_values,
            metadata={
                **dict(metadata or {}),
                "source_workflow": WORKFLOW,
                "lane_queue": str(lane_queue_path),
                "batch_ids": ",".join(selected_batch_ids),
            },
            compact_json=compact_json,
            fail_on_blocked=False,
        )
    summary = _summary(
        selected_batches=selected_batches,
        batch_collection=batch_collection,
        child_payload=child_payload,
        rule_stubs=rule_stubs,
    )
    paths: dict[str, Any] = {
        "workflow_report": str(report_path),
        "batch_collection_corpus": str(collection_path),
        "child_workflow_report": None,
        "child_artifact_manifest": None,
        "world_model_rule_stubs": str(rules_path) if rule_stubs else None,
        "artifact_manifest": str(manifest_path),
    }
    if child_payload is not None:
        paths["child_workflow_report"] = child_payload["paths"]["workflow_report"]
        paths["child_artifact_manifest"] = child_payload["paths"]["artifact_manifest"]
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": _status(child_payload, rule_stubs=rule_stubs),
        "scope": (
            "Executes selected lane queue batches by materializing a compatible "
            "source-family fact-collection corpus. Results and rule stubs remain "
            "candidate inputs and require downstream route/mapping or rule "
            "authoring gates."
        ),
        "source": {
            "lane_queue": str(lane_queue_path),
            "lane_queue_workflow": lane_queue.get("workflow"),
            "collection_corpus": str(collection_corpus_path),
            "collection_workflow": collection.get("workflow"),
            "source_catalogs": tuple(str(path) for path in source_catalog_paths),
        },
        "config": {
            "batch_ids": selected_batch_ids,
            "source_backed_request_types": source_backed_request_types,
            "rule_request_type": RULE_REQUEST_TYPE if rule_stubs else None,
            "adapter_max_results": int(adapter_max_results),
            "adapter_max_query_variants": int(adapter_max_query_variants),
            "adapter_min_text_overlap": float(adapter_min_text_overlap),
            "adapter_diversify_source_families": bool(adapter_diversify_source_families),
            "min_request_result_coverage": float(min_request_result_coverage),
            "default_source_family": str(default_source_family),
            "keep_qid_values": bool(keep_qid_values),
        },
        "label_usage": {
            "labels_used_for_batch_selection": False,
            "answers_copied_to_batch_collection": False,
            "model_answers_copied_to_batch_collection": False,
            "adapter_results_are_verifier_evidence": False,
            "rule_stubs_are_verifier_evidence": False,
        },
        "paths": paths,
        "summary": summary,
        "child_workflow_summary": {} if child_payload is None else child_payload["summary"],
        "selected_batches": tuple(dict(batch) for batch in selected_batches),
        "rule_stubs": rule_stubs,
        "metadata": dict(metadata or {}),
    }
    _write_json(report_path, payload, compact=compact_json)
    manifest_artifacts: dict[str, Path] = {
        "lane_batch_workflow": report_path,
        "lane_batch_collection_corpus": collection_path,
        "lane_queue": Path(lane_queue_path),
        "source_collection_corpus": Path(collection_corpus_path),
        **{f"source_catalog_{idx}": Path(path) for idx, path in enumerate(source_catalog_paths, start=1)},
    }
    if child_payload is not None:
        manifest_artifacts["child_fact_collection_workflow"] = Path(child_payload["paths"]["workflow_report"])
        manifest_artifacts["child_fact_collection_manifest"] = Path(child_payload["paths"]["artifact_manifest"])
    if rule_stubs:
        manifest_artifacts["world_model_rule_stubs"] = rules_path
    manifest = build_artifact_manifest(
        manifest_artifacts,
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "batch_count": summary["batch_count"],
            "target_count": summary["target_count"],
            "source_backed_request_count": summary["source_backed_request_count"],
            "request_with_results_count": summary["request_with_results_count"],
            "request_without_results_count": summary["request_without_results_count"],
            "request_result_coverage": summary["request_result_coverage"],
            "min_request_result_coverage": float(min_request_result_coverage),
            "adapter_result_count": summary["adapter_result_count"],
            "structured_qa_document_count": summary["structured_qa_document_count"],
            "rule_stub_count": summary["rule_stub_count"],
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
                "batch_count": summary["batch_count"],
                "target_count": summary["target_count"],
                "source_backed_request_count": summary["source_backed_request_count"],
                "request_with_results_count": summary["request_with_results_count"],
                "request_without_results_count": summary["request_without_results_count"],
                "request_result_coverage": summary["request_result_coverage"],
                "min_request_result_coverage": float(min_request_result_coverage),
                "adapter_result_count": summary["adapter_result_count"],
                "structured_qa_document_count": summary["structured_qa_document_count"],
                "rule_stub_count": summary["rule_stub_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    if fail_on_blocked and payload["status"] == "blocked":
        raise SystemExit(1)
    return payload


def _batch_collection(
    *,
    lane_queue: Mapping[str, Any],
    collection: Mapping[str, Any],
    selected_batches: Sequence[Mapping[str, Any]],
    batch_ids: Sequence[str],
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_request_ids = tuple(
        dict.fromkeys(
            request_id
            for batch in selected_batches
            for request_id in _string_sequence(batch.get("source_request_ids"))
        )
    )
    requests_by_id = _requests_by_id(collection.get("requests", {}))
    queue_requests_by_id = _queue_requests_by_source_id(lane_queue.get("adapter_requests", ()))
    batch_by_request_id = _batch_by_source_request_id(selected_batches)
    missing = tuple(request_id for request_id in source_request_ids if request_id not in requests_by_id)
    if missing:
        raise ValueError(f"lane batch references missing collection requests: {', '.join(missing)}")

    targets_by_id = {
        str(item.get("target_id")): dict(item)
        for item in _mapping_sequence(collection.get("targets", ()))
    }
    request_buckets: dict[str, list[dict[str, Any]]] = {
        request_type: [] for request_type in (*SOURCE_BACKED_REQUEST_TYPES, RULE_REQUEST_TYPE)
    }
    selected_target_ids: list[str] = []
    stripped_reserved = Counter()
    for request_id in source_request_ids:
        request = dict(requests_by_id[request_id])
        request_type = str(request.get("request_type") or "")
        if request_type not in request_buckets:
            continue
        sanitized, stripped = _strip_reserved(request)
        sanitized = _with_batch_route_metadata(
            sanitized,
            queue_request=queue_requests_by_id.get(request_id),
            batch=batch_by_request_id.get(request_id),
        )
        stripped_reserved.update(stripped)
        request_buckets[request_type].append(sanitized)
        target_id = str(sanitized.get("target_id") or "")
        if target_id:
            selected_target_ids.append(target_id)

    selected_targets: list[dict[str, Any]] = []
    for target_id in dict.fromkeys(selected_target_ids):
        target = targets_by_id.get(target_id, {"target_id": target_id})
        sanitized, stripped = _strip_reserved(target)
        stripped_reserved.update({f"target.{key}": value for key, value in stripped.items()})
        selected_targets.append(sanitized)

    request_counts = Counter({key: len(value) for key, value in request_buckets.items() if value})
    selected_request_rows = tuple(row for rows in request_buckets.values() for row in rows)
    lane_counts = Counter(str(batch.get("next_lane") or "") for batch in selected_batches)
    lane_status_counts = Counter(str(batch.get("lane_status") or "") for batch in selected_batches)
    primary_closure_route_counts = Counter(
        str(row.get("primary_closure_route") or "")
        for row in selected_request_rows
        if str(row.get("primary_closure_route") or "")
    )
    closure_route_counts = Counter(
        str(row.get("closure_route") or "")
        for row in selected_request_rows
        if str(row.get("closure_route") or "")
    )
    source_gap_type_counts = Counter(
        str(row.get("source_gap_type") or "")
        for row in selected_request_rows
        if str(row.get("source_gap_type") or "")
    )
    return {
        "schema_version": 1,
        "workflow": COLLECTION_WORKFLOW,
        "status": "ready_for_collection" if source_request_ids else "empty",
        "scope": (
            "Batch subset of a source-family structured QA fact-collection "
            "corpus, selected from lane execution batches. Requests are still "
            "source-discovery or rule-authoring inputs, not verifier evidence."
        ),
        "source": {
            "source_collection_workflow": collection.get("workflow"),
            "source_collection_status": collection.get("status"),
            "lane_queue_workflow": lane_queue.get("workflow"),
            "lane_queue_status": lane_queue.get("status"),
        },
        "label_usage": {
            "labels_used_for_collection_requests": False,
            "labels_copied_to_collection_requests": False,
            "model_answers_copied_to_collection_requests": False,
            "requests_are_verifier_evidence": False,
        },
        "config": {
            "batch_ids": tuple(batch_ids),
        },
        "summary": {
            "target_count": len(selected_targets),
            "total_request_count": sum(request_counts.values()),
            "request_counts": _sorted_counter(request_counts),
            "request_type_counts": _sorted_counter(request_counts),
            "lane_counts": _sorted_counter(lane_counts),
            "lane_status_counts": _sorted_counter(lane_status_counts),
            "primary_closure_route_counts": _sorted_counter(primary_closure_route_counts),
            "closure_route_counts": _sorted_counter(closure_route_counts),
            "source_gap_type_counts": _sorted_counter(source_gap_type_counts),
            "stripped_reserved_field_counts": _sorted_counter(stripped_reserved),
        },
        "targets": tuple(selected_targets),
        "requests": {key: tuple(value) for key, value in request_buckets.items()},
        "source_discovery_documents": (),
        "metadata": dict(metadata or {}),
    }


def _summary(
    *,
    selected_batches: Sequence[Mapping[str, Any]],
    batch_collection: Mapping[str, Any],
    child_payload: Mapping[str, Any] | None,
    rule_stubs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    child_summary = {} if child_payload is None else child_payload.get("summary", {})
    request_counts = Counter()
    requests_payload = batch_collection.get("requests", {})
    if isinstance(requests_payload, Mapping):
        for request_type, rows in requests_payload.items():
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
                count = len(rows)
                if count:
                    request_counts[str(request_type)] = count
    lane_counts = Counter(str(batch.get("next_lane") or "") for batch in selected_batches)
    lane_status_counts = Counter(str(batch.get("lane_status") or "") for batch in selected_batches)
    selected_request_rows = tuple(
        row
        for rows in requests_payload.values()
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray))
        for row in rows
        if isinstance(row, Mapping)
    ) if isinstance(requests_payload, Mapping) else ()
    primary_closure_route_counts = Counter(
        str(row.get("primary_closure_route") or "")
        for row in selected_request_rows
        if str(row.get("primary_closure_route") or "")
    )
    closure_route_counts = Counter(
        str(row.get("closure_route") or "")
        for row in selected_request_rows
        if str(row.get("closure_route") or "")
    )
    source_gap_type_counts = Counter(
        str(row.get("source_gap_type") or "")
        for row in selected_request_rows
        if str(row.get("source_gap_type") or "")
    )
    return {
        "batch_count": len(selected_batches),
        "batch_ids": tuple(str(batch.get("batch_id") or "") for batch in selected_batches),
        "target_count": _nested_int(batch_collection, "summary", "target_count") or 0,
        "total_request_count": _nested_int(batch_collection, "summary", "total_request_count") or 0,
        "source_backed_request_count": int(child_summary.get("source_backed_request_count", 0)),
        "request_with_results_count": int(child_summary.get("request_with_results_count", 0)),
        "request_without_results_count": int(child_summary.get("request_without_results_count", 0)),
        "request_without_results_ids": _string_sequence(child_summary.get("request_without_results_ids", ())),
        "request_result_coverage": float(child_summary.get("request_result_coverage", 1.0)),
        "adapter_result_count": int(child_summary.get("adapter_result_count", 0)),
        "structured_qa_document_count": int(child_summary.get("structured_qa_document_count", 0)),
        "rule_stub_count": max(int(child_summary.get("rule_stub_count", 0)), len(rule_stubs)),
        "request_type_counts": _sorted_counter(request_counts),
        "lane_counts": _sorted_counter(lane_counts),
        "lane_status_counts": _sorted_counter(lane_status_counts),
        "primary_closure_route_counts": _sorted_counter(primary_closure_route_counts),
        "closure_route_counts": _sorted_counter(closure_route_counts),
        "source_gap_type_counts": _sorted_counter(source_gap_type_counts),
        "child_status": None if child_payload is None else child_payload.get("status"),
        "reserved_source_document_field_hits": dict(
            child_summary.get("reserved_source_document_field_hits") or {}
        ),
    }


def _status(child_payload: Mapping[str, Any] | None, *, rule_stubs: Sequence[Mapping[str, Any]]) -> str:
    if child_payload is None:
        return "ready_for_rule_authoring" if rule_stubs else "blocked"
    child_status = str(child_payload.get("status") or "")
    if child_status == "ready_for_fact_mapping":
        return "ready_for_fact_mapping"
    if child_status == "observed":
        return "observed"
    return "blocked"


def _load_lane_queue(path: str | Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    if payload.get("workflow") != LANE_QUEUE_WORKFLOW:
        raise ValueError(f"{path} is not a {LANE_QUEUE_WORKFLOW} report.")
    return payload


def _load_collection(path: str | Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    if payload.get("workflow") != COLLECTION_WORKFLOW:
        raise ValueError(f"{path} is not a {COLLECTION_WORKFLOW} report.")
    return payload


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _select_batches(lane_queue: Mapping[str, Any], batch_ids: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
    by_id = {
        str(batch.get("batch_id") or ""): batch
        for batch in _mapping_sequence(lane_queue.get("execution_batches", ()))
    }
    missing = tuple(batch_id for batch_id in batch_ids if batch_id not in by_id)
    if missing:
        raise ValueError(f"unknown lane batch ids: {', '.join(missing)}")
    return tuple(by_id[batch_id] for batch_id in batch_ids)


def _requests_by_id(requests_payload: Any) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    if not isinstance(requests_payload, Mapping):
        return output
    for rows in requests_payload.values():
        for request in _mapping_sequence(rows):
            request_id = str(request.get("request_id") or "")
            if request_id:
                output[request_id] = request
    return output


def _queue_requests_by_source_id(rows: Any) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("source_request_id")): row
        for row in _mapping_sequence(rows)
        if str(row.get("source_request_id") or "")
    }


def _batch_by_source_request_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for batch in rows:
        for request_id in _string_sequence(batch.get("source_request_ids")):
            output[request_id] = batch
    return output


def _with_batch_route_metadata(
    row: Mapping[str, Any],
    *,
    queue_request: Mapping[str, Any] | None,
    batch: Mapping[str, Any] | None,
) -> dict[str, Any]:
    output = dict(row)
    request_type = str(output.get("request_type") or "")
    closure_route = (
        _first_string(queue_request, "closure_route")
        or _first_string(batch, "closure_route")
        or CLOSURE_ROUTE_BY_REQUEST_TYPE.get(request_type, "")
    )
    primary_closure_route = (
        _first_string(queue_request, "primary_closure_route")
        or _first_string(batch, "primary_closure_route")
        or closure_route
    )
    for key in ("next_lane", "lane_status"):
        value = _first_string(queue_request, key) or _first_string(batch, key)
        if value and not output.get(key):
            output[key] = value
    source_gap_type = (
        _first_string(queue_request, "source_gap_type")
        or _first_string(batch, "source_gap_type")
        or str(output.get("gap_type") or "")
    )
    if primary_closure_route and not output.get("primary_closure_route"):
        output["primary_closure_route"] = primary_closure_route
    if closure_route and not output.get("closure_route"):
        output["closure_route"] = closure_route
    if source_gap_type and not output.get("source_gap_type"):
        output["source_gap_type"] = source_gap_type
    if source_gap_type and not output.get("evidence_gap_type"):
        output["evidence_gap_type"] = source_gap_type
    return output


def _first_string(row: Mapping[str, Any] | None, key: str) -> str:
    if not isinstance(row, Mapping):
        return ""
    return str(row.get(key) or "")


def _strip_reserved(row: Mapping[str, Any]) -> tuple[dict[str, Any], Counter[str]]:
    stripped: Counter[str] = Counter()
    output: dict[str, Any] = {}
    for key, value in row.items():
        if key in RESERVED_REQUEST_FIELDS:
            stripped[key] += 1
            continue
        output[key] = value
    return output, stripped


def _rule_stubs(batch_collection: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    requests_payload = batch_collection.get("requests", {})
    if not isinstance(requests_payload, Mapping):
        return ()
    stubs = []
    for request in _mapping_sequence(requests_payload.get(RULE_REQUEST_TYPE, ())):
        stubs.append({
            "schema_version": 1,
            "workflow": WORKFLOW,
            "request_id": str(request.get("request_id") or ""),
            "target_id": str(request.get("target_id") or ""),
            "request_type": RULE_REQUEST_TYPE,
            "status": "requires_deterministic_rule_adapter",
            "rule_family": str(request.get("rule_family") or "world_model_consistency"),
            "rule_reason": str(request.get("rule_reason") or ""),
            "rule_seed": str(request.get("rule_seed") or ""),
            "required_inputs": _string_sequence(request.get("required_inputs", ())),
            "question": str(request.get("question") or ""),
            "question_type": str(request.get("question_type") or ""),
            "gap_type": str(request.get("gap_type") or request.get("source_gap_type") or ""),
            "source_gap_type": str(request.get("source_gap_type") or request.get("gap_type") or ""),
            "evidence_gap_type": str(
                request.get("evidence_gap_type")
                or request.get("source_gap_type")
                or request.get("gap_type")
                or ""
            ),
            "primary_closure_route": str(request.get("primary_closure_route") or ""),
            "closure_route": str(request.get("closure_route") or ""),
            "priority": str(request.get("priority") or ""),
            "not_verifier_evidence": True,
        })
    return tuple(stubs)


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value if str(item))
    return ()


def _nested_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    try:
        return None if current is None else int(current)
    except (TypeError, ValueError):
        return None


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(((key, value) for key, value in counter.items() if key), key=lambda item: (-item[1], item[0])))


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
    parser.add_argument("--lane-queue", required=True)
    parser.add_argument("--collection-corpus", required=True)
    parser.add_argument("--source-catalog", action="append", default=[])
    parser.add_argument("--batch-id", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--batch-collection-corpus", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--adapter-max-results", type=int, default=3)
    parser.add_argument("--adapter-max-query-variants", type=int, default=3)
    parser.add_argument("--adapter-min-text-overlap", type=float, default=0.05)
    parser.add_argument("--no-adapter-diversify-source-families", action="store_true")
    parser.add_argument("--min-request-result-coverage", type=float, default=1.0)
    parser.add_argument("--default-source-family", default="reference")
    parser.add_argument("--keep-qid-values", action="store_true")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)
    payload = run_source_family_structured_qa_lane_batch_workflow(
        lane_queue_path=args.lane_queue,
        collection_corpus_path=args.collection_corpus,
        source_catalog_paths=tuple(args.source_catalog or ()),
        output_dir=args.output_dir,
        batch_ids=tuple(args.batch_id or ()),
        report_json_path=args.json,
        batch_collection_corpus_path=args.batch_collection_corpus,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        adapter_max_results=args.adapter_max_results,
        adapter_max_query_variants=args.adapter_max_query_variants,
        adapter_min_text_overlap=args.adapter_min_text_overlap,
        adapter_diversify_source_families=not bool(args.no_adapter_diversify_source_families),
        min_request_result_coverage=args.min_request_result_coverage,
        default_source_family=args.default_source_family,
        keep_qid_values=bool(args.keep_qid_values),
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
        fail_on_blocked=bool(args.fail_on_blocked),
    )
    summary = payload["summary"]
    print(
        "source_family_structured_qa_lane_batch_workflow_ok "
        f"status={payload['status']} "
        f"batches={summary['batch_count']} "
        f"targets={summary['target_count']} "
        f"requests={summary['source_backed_request_count']} "
        f"coverage={summary['request_result_coverage']:.3f} "
        f"results={summary['adapter_result_count']} "
        f"qa_docs={summary['structured_qa_document_count']} "
        f"rules={summary['rule_stub_count']}"
    )


if __name__ == "__main__":
    main()
