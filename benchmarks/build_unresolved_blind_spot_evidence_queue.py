"""Build an adapter-ready queue for unresolved blind-spot evidence collection.

This workflow sits after the question/property correction handoff. It consumes
the existing evidence-expansion plan, source-discovery collection corpus, and
question-property mapping report, then filters out blind spots already handled
by explicit property correction gates. The output is not verifier evidence: it
is a prioritized queue for citation/search adapters and deterministic
world-model or calculator rule authoring.
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

DEFAULT_REQUEST_TYPES = ("external_citation", "world_model_or_calculator_rule")
REQUEST_TYPE_TO_ADAPTER = {
    "external_citation": "external_citation_search",
    "world_model_or_calculator_rule": "world_model_rule_authoring",
    "counterfactual_probe": "counterfactual_probe_generator",
}
STATUS_PRIORITY = {
    "no_joined_facts": 60,
    "generic_fact_only": 55,
    "unmapped_low_relevance": 45,
    "subject_only_or_unsupported_property": 40,
    "answer_entity_collision": 35,
    "answer_value_supported": 15,
}
PRIORITY_BASE = {"high": 100, "medium": 50, "low": 10}


def build_unresolved_blind_spot_evidence_queue(
    *,
    plan: Mapping[str, Any],
    collection_corpus: Mapping[str, Any],
    question_property_mapping: Mapping[str, Any],
    covered_fact_mapping: Mapping[str, Any] | None = None,
    request_types: Sequence[str] = DEFAULT_REQUEST_TYPES,
    priorities: Sequence[str] = (),
    max_targets: int | None = None,
    max_requests_per_target: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready unresolved adapter queue."""
    _validate_inputs(plan=plan, collection=collection_corpus, mapping=question_property_mapping)
    selected_request_types = _normalize_request_types(request_types)
    selected_priorities = tuple(dict.fromkeys(str(item).strip() for item in priorities if str(item).strip()))
    if max_targets is not None and int(max_targets) <= 0:
        raise ValueError("max_targets must be positive when provided.")
    if max_requests_per_target is not None and int(max_requests_per_target) <= 0:
        raise ValueError("max_requests_per_target must be positive when provided.")

    plan_by_record = _records_by_index(plan.get("targets", ()))
    mapping_by_record = _records_by_index(question_property_mapping.get("records", ()))
    covered_by_record = _records_by_index(
        () if covered_fact_mapping is None else covered_fact_mapping.get("records", ())
    )
    collection_targets = tuple(_mapping_sequence(collection_corpus.get("targets", ())))
    collection_requests = _collection_requests_by_target(
        collection_corpus.get("requests", {}),
        request_types=selected_request_types,
    )

    resolved_record_indices = {
        record_index
        for record_index, record in mapping_by_record.items()
        if bool(record.get("correction_candidate"))
    }

    target_queue: list[dict[str, Any]] = []
    request_queue: list[dict[str, Any]] = []
    skipped_targets: list[dict[str, Any]] = []
    for ordinal, target in enumerate(collection_targets, start=1):
        record_index = _optional_int(target.get("record_index"))
        target_id = str(target.get("target_id") or _target_id(record_index=record_index, ordinal=ordinal))
        priority = str(target.get("priority", ""))
        if selected_priorities and priority not in selected_priorities:
            continue
        mapping_record = mapping_by_record.get(record_index) if record_index is not None else None
        if record_index in resolved_record_indices:
            skipped_targets.append({
                "target_id": target_id,
                "record_index": record_index,
                "reason": "resolved_by_question_property_correction",
                "mapping_decision": _mapping_decision(mapping_record),
            })
            continue
        target_requests = collection_requests.get(target_id, ())
        if not target_requests:
            skipped_targets.append({
                "target_id": target_id,
                "record_index": record_index,
                "reason": "no_selected_request_types",
                "mapping_decision": _mapping_decision(mapping_record),
            })
            continue
        target_entry = _target_entry(
            target,
            plan_record=plan_by_record.get(record_index) if record_index is not None else None,
            mapping_record=mapping_record,
            covered_record=covered_by_record.get(record_index) if record_index is not None else None,
            target_id=target_id,
            request_count=len(target_requests),
        )
        target_queue.append(target_entry)

    target_queue.sort(
        key=lambda item: (
            -float(item["priority_score"]),
            int(item["record_index"] or 10**12),
            item["target_id"],
        )
    )
    if max_targets is not None:
        keep_ids = {item["target_id"] for item in target_queue[: int(max_targets)]}
        target_queue = [item for item in target_queue if item["target_id"] in keep_ids]
    target_rank = {item["target_id"]: idx for idx, item in enumerate(target_queue, start=1)}

    for target in target_queue:
        target_id = str(target["target_id"])
        raw_requests = list(collection_requests.get(target_id, ()))
        raw_requests.sort(
            key=lambda item: (
                _request_type_rank(str(item.get("request_type"))),
                str(item.get("request_id")),
            )
        )
        if max_requests_per_target is not None:
            raw_requests = raw_requests[: int(max_requests_per_target)]
        for request_ordinal, request in enumerate(raw_requests, start=1):
            request_queue.append(_queue_request(
                request,
                target=target,
                target_rank=target_rank[target_id],
                request_ordinal=request_ordinal,
            ))

    summary = _summary(
        target_queue=target_queue,
        request_queue=request_queue,
        skipped_targets=skipped_targets,
        resolved_record_indices=resolved_record_indices,
        source_collection_targets=collection_targets,
    )
    return {
        "schema_version": 1,
        "workflow": "unresolved_blind_spot_evidence_queue",
        "status": "ready_for_adapter_execution" if request_queue else "empty",
        "scope": (
            "Adapter execution queue for blind spots not resolved by explicit "
            "question/property correction gates. Requests are source-discovery "
            "or rule-authoring tasks, not verifier evidence."
        ),
        "source": {
            "plan_workflow": plan.get("workflow"),
            "plan_status": plan.get("status"),
            "plan_target_count": _nested_int(plan, "summary", "target_count"),
            "collection_workflow": collection_corpus.get("workflow"),
            "collection_status": collection_corpus.get("status"),
            "collection_target_count": _nested_int(collection_corpus, "summary", "target_count"),
            "question_property_mapping_workflow": question_property_mapping.get("workflow"),
            "question_property_mapping_status": question_property_mapping.get("status"),
            "covered_fact_mapping_workflow": (
                None if covered_fact_mapping is None else covered_fact_mapping.get("workflow")
            ),
            "covered_fact_mapping_status": None if covered_fact_mapping is None else covered_fact_mapping.get("status"),
        },
        "label_usage": {
            "labels_used_for_queue_selection": False,
            "labels_copied_to_queue": False,
            "requests_are_verifier_evidence": False,
        },
        "config": {
            "request_types": selected_request_types,
            "priorities": selected_priorities,
            "max_targets": max_targets,
            "max_requests_per_target": max_requests_per_target,
        },
        "summary": summary,
        "targets": tuple(target_queue),
        "adapter_requests": tuple(request_queue),
        "skipped_targets": tuple(skipped_targets),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    plan_path: str | Path,
    collection_corpus_path: str | Path,
    question_property_mapping_path: str | Path,
    output_dir: str | Path,
    covered_fact_mapping_path: str | Path | None = None,
    report_json_path: str | Path | None = None,
    target_jsonl_path: str | Path | None = None,
    request_jsonl_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    request_types: Sequence[str] = DEFAULT_REQUEST_TYPES,
    priorities: Sequence[str] = (),
    max_targets: int | None = None,
    max_requests_per_target: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register an unresolved queue."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    report_path = Path(report_json_path) if report_json_path is not None else output / "unresolved-evidence-queue.json"
    target_path = Path(target_jsonl_path) if target_jsonl_path is not None else output / "unresolved-targets.jsonl"
    request_path = Path(request_jsonl_path) if request_jsonl_path is not None else output / "adapter-requests.jsonl"
    manifest_path = (
        Path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else output / "artifact-manifest.json"
    )
    plan = _load_json_object(plan_path)
    collection = _load_json_object(collection_corpus_path)
    mapping = _load_json_object(question_property_mapping_path)
    covered = None if covered_fact_mapping_path is None else _load_json_object(covered_fact_mapping_path)
    payload = build_unresolved_blind_spot_evidence_queue(
        plan=plan,
        collection_corpus=collection,
        question_property_mapping=mapping,
        covered_fact_mapping=covered,
        request_types=request_types,
        priorities=priorities,
        max_targets=max_targets,
        max_requests_per_target=max_requests_per_target,
        metadata=metadata,
    )
    report = dict(payload)
    report["paths"] = {
        "plan": str(plan_path),
        "collection_corpus": str(collection_corpus_path),
        "question_property_mapping": str(question_property_mapping_path),
        "covered_fact_mapping": None if covered_fact_mapping_path is None else str(covered_fact_mapping_path),
        "targets_jsonl": str(target_path),
        "adapter_requests_jsonl": str(request_path),
        "artifact_manifest": str(manifest_path),
    }
    payload = dict(payload)
    payload["paths"] = report["paths"]

    _write_json(report_path, report, compact=compact_json)
    _write_jsonl(target_path, payload["targets"])
    _write_jsonl(request_path, payload["adapter_requests"])
    manifest = build_artifact_manifest(
        {
            "unresolved_blind_spot_evidence_queue": report_path,
            "unresolved_targets": target_path,
            "adapter_requests": request_path,
            "blind_spot_evidence_expansion_plan": Path(plan_path),
            "blind_spot_evidence_collection_corpus": Path(collection_corpus_path),
            "question_property_mapping": Path(question_property_mapping_path),
            "covered_fact_mapping": None if covered_fact_mapping_path is None else Path(covered_fact_mapping_path),
        },
        root=manifest_path.parent,
        metadata={
            "workflow": report["workflow"],
            "status": report["status"],
            "target_count": report["summary"]["target_count"],
            "adapter_request_count": report["summary"]["adapter_request_count"],
            "resolved_by_question_property_count": report["summary"]["resolved_by_question_property_count"],
            "external_citation_count": report["summary"]["request_type_counts"].get("external_citation", 0),
            "world_model_rule_count": report["summary"]["request_type_counts"].get(
                "world_model_or_calculator_rule",
                0,
            ),
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
                "workflow": report["workflow"],
                "status": report["status"],
                "target_count": report["summary"]["target_count"],
                "adapter_request_count": report["summary"]["adapter_request_count"],
                "resolved_by_question_property_count": report["summary"]["resolved_by_question_property_count"],
                "external_citation_count": report["summary"]["request_type_counts"].get("external_citation", 0),
                "world_model_rule_count": report["summary"]["request_type_counts"].get(
                    "world_model_or_calculator_rule",
                    0,
                ),
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _target_entry(
    target: Mapping[str, Any],
    *,
    plan_record: Mapping[str, Any] | None,
    mapping_record: Mapping[str, Any] | None,
    covered_record: Mapping[str, Any] | None,
    target_id: str,
    request_count: int,
) -> dict[str, Any]:
    record_index = _optional_int(target.get("record_index"))
    mapping_decision = _mapping_decision(mapping_record)
    source_mapping_status = str(
        (mapping_record or {}).get("source_mapping_status")
        or (covered_record or {}).get("mapping_status")
        or ""
    )
    evidence_status = _evidence_status(mapping_decision=mapping_decision, source_mapping_status=source_mapping_status)
    recommended_routes = tuple(str(item) for item in _sequence(target.get("recommended_routes")))
    requestable_routes = _requestable_routes(recommended_routes)
    priority_score = _priority_score(
        priority=str(target.get("priority", "")),
        evidence_status=evidence_status,
        question_type=str(target.get("question_type", "")),
        requestable_routes=requestable_routes,
        request_count=request_count,
    )
    return {
        "target_id": target_id,
        "record_index": record_index,
        "priority": str(target.get("priority", "")),
        "priority_score": priority_score,
        "evidence_status": evidence_status,
        "mapping_decision": mapping_decision,
        "source_mapping_status": source_mapping_status,
        "question_type": str(target.get("question_type", "")),
        "question": str(target.get("question", "")),
        "model_answer": str(target.get("model_answer", target.get("answer", ""))),
        "recommended_routes": recommended_routes,
        "requestable_routes": requestable_routes,
        "entity_candidates": tuple(str(item) for item in _sequence(target.get("entity_candidates"))),
        "query_seeds": tuple(str(item) for item in _sequence(target.get("query_seeds"))),
        "wikidata_property_hints": tuple(str(item) for item in _sequence(target.get("wikidata_property_hints"))),
        "joined_fact_count": _optional_int((covered_record or {}).get("joined_fact_count")),
        "joined_property_counts": dict((covered_record or {}).get("joined_property_counts") or {}),
        "best_mapping_score": _optional_float((mapping_record or {}).get("best_mapping_score")),
        "plan_priority": None if plan_record is None else str(plan_record.get("priority", "")),
    }


def _queue_request(
    request: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    target_rank: int,
    request_ordinal: int,
) -> dict[str, Any]:
    request_type = str(request.get("request_type", ""))
    request_id = str(request.get("request_id", ""))
    query = str(request.get("query") or request.get("rule_seed") or request.get("probe_instruction") or "")
    return {
        "queue_id": f"queue:{target['target_id']}:{request_type}:{request_ordinal}",
        "source_request_id": request_id,
        "target_rank": int(target_rank),
        "target_id": target["target_id"],
        "record_index": target["record_index"],
        "adapter_family": REQUEST_TYPE_TO_ADAPTER.get(request_type, request_type),
        "request_type": request_type,
        "evidence_status": target["evidence_status"],
        "mapping_decision": target["mapping_decision"],
        "priority": target["priority"],
        "priority_score": target["priority_score"],
        "question_type": target["question_type"],
        "question": target["question"],
        "model_answer": target["model_answer"],
        "query": query,
        "requires_timestamp": bool(request.get("requires_timestamp")),
        "rule_family": request.get("rule_family"),
        "probe_type": request.get("probe_type"),
        "usage": request.get("usage", "source_discovery_only"),
        "not_verifier_evidence": True,
        "metadata": {
            "source_workflow": "blind_spot_evidence_collection_corpus",
            "queue_workflow": "unresolved_blind_spot_evidence_queue",
            "request_id": request_id,
            "target_id": target["target_id"],
            "selected_adapter_family": REQUEST_TYPE_TO_ADAPTER.get(request_type, request_type),
        },
    }


def _summary(
    *,
    target_queue: Sequence[Mapping[str, Any]],
    request_queue: Sequence[Mapping[str, Any]],
    skipped_targets: Sequence[Mapping[str, Any]],
    resolved_record_indices: set[int],
    source_collection_targets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    request_type_counts = Counter(str(item.get("request_type")) for item in request_queue)
    adapter_counts = Counter(str(item.get("adapter_family")) for item in request_queue)
    evidence_status_counts = Counter(str(item.get("evidence_status")) for item in target_queue)
    mapping_counts = Counter(str(item.get("mapping_decision")) for item in target_queue)
    question_type_counts = Counter(str(item.get("question_type")) for item in target_queue)
    skipped_counts = Counter(str(item.get("reason")) for item in skipped_targets)
    targets_with_citation = {
        str(item.get("target_id"))
        for item in request_queue
        if item.get("request_type") == "external_citation"
    }
    targets_with_rules = {
        str(item.get("target_id"))
        for item in request_queue
        if item.get("request_type") == "world_model_or_calculator_rule"
    }
    return {
        "source_collection_target_count": len(source_collection_targets),
        "resolved_by_question_property_count": len(resolved_record_indices),
        "target_count": len(target_queue),
        "adapter_request_count": len(request_queue),
        "request_type_counts": _sorted_counter(request_type_counts),
        "adapter_family_counts": _sorted_counter(adapter_counts),
        "evidence_status_counts": _sorted_counter(evidence_status_counts),
        "mapping_decision_counts": _sorted_counter(mapping_counts),
        "question_type_counts": _sorted_counter(question_type_counts),
        "skipped_target_counts": _sorted_counter(skipped_counts),
        "targets_with_external_citation": len(targets_with_citation),
        "targets_with_world_model_or_calculator_rule": len(targets_with_rules),
        "top_target": None if not target_queue else {
            "target_id": target_queue[0]["target_id"],
            "record_index": target_queue[0]["record_index"],
            "evidence_status": target_queue[0]["evidence_status"],
            "priority_score": target_queue[0]["priority_score"],
            "question_type": target_queue[0]["question_type"],
        },
    }


def _validate_inputs(*, plan: Mapping[str, Any], collection: Mapping[str, Any], mapping: Mapping[str, Any]) -> None:
    if plan.get("workflow") != "blind_spot_evidence_expansion_plan":
        raise ValueError("plan must be a blind_spot_evidence_expansion_plan report.")
    if collection.get("workflow") != "blind_spot_evidence_collection_corpus":
        raise ValueError("collection_corpus must be a blind_spot_evidence_collection_corpus report.")
    if mapping.get("workflow") != "blind_spot_question_property_mapping":
        raise ValueError("question_property_mapping must be a blind_spot_question_property_mapping report.")


def _normalize_request_types(values: Sequence[str]) -> tuple[str, ...]:
    request_types = tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if not request_types:
        raise ValueError("at least one request type is required.")
    supported = set(REQUEST_TYPE_TO_ADAPTER)
    invalid = sorted(set(request_types) - supported)
    if invalid:
        raise ValueError(f"unsupported request types: {', '.join(invalid)}")
    return request_types


def _collection_requests_by_target(
    requests_payload: Any,
    *,
    request_types: Sequence[str],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    if not isinstance(requests_payload, Mapping):
        return {}
    request_type_set = set(request_types)
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for request_type in request_types:
        bucket = requests_payload.get(request_type, ())
        if request_type not in request_type_set:
            continue
        for request in _mapping_sequence(bucket):
            target_id = str(request.get("target_id", ""))
            if target_id:
                grouped[target_id].append(request)
    return {key: tuple(value) for key, value in grouped.items()}


def _records_by_index(records: Any) -> dict[int, Mapping[str, Any]]:
    output: dict[int, Mapping[str, Any]] = {}
    for record in _mapping_sequence(records):
        record_index = _optional_int(record.get("record_index"))
        if record_index is not None:
            output[record_index] = record
    return output


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _mapping_decision(record: Mapping[str, Any] | None) -> str:
    if not record:
        return "missing_question_property_mapping"
    return str(record.get("mapping_decision") or record.get("source_mapping_status") or "unknown")


def _evidence_status(*, mapping_decision: str, source_mapping_status: str) -> str:
    if mapping_decision in STATUS_PRIORITY:
        return mapping_decision
    if mapping_decision == "mapped_correction_candidate":
        return "resolved_by_question_property_correction"
    if source_mapping_status in STATUS_PRIORITY:
        return source_mapping_status
    if source_mapping_status:
        return source_mapping_status
    return mapping_decision or "unknown"


def _requestable_routes(routes: Sequence[str]) -> tuple[str, ...]:
    requestable: list[str] = []
    for route in routes:
        if route in {"retrieval_citation", "time_sensitive_retrieval"}:
            requestable.append("external_citation_search")
        elif route in {"world_model_rule", "calculator"}:
            requestable.append("world_model_rule_authoring")
        elif route.startswith("counterfactual_"):
            requestable.append("counterfactual_probe_generator")
    return tuple(dict.fromkeys(requestable))


def _priority_score(
    *,
    priority: str,
    evidence_status: str,
    question_type: str,
    requestable_routes: Sequence[str],
    request_count: int,
) -> float:
    score = float(PRIORITY_BASE.get(priority, 0))
    score += float(STATUS_PRIORITY.get(evidence_status, 20))
    if "external_citation_search" in requestable_routes:
        score += 12.0
    if "world_model_rule_authoring" in requestable_routes:
        score += 10.0
    if question_type in {"causal", "method", "quantity", "temporal"}:
        score += 8.0
    score += min(int(request_count), 5)
    return score


def _request_type_rank(request_type: str) -> int:
    order = {name: idx for idx, name in enumerate(DEFAULT_REQUEST_TYPES + ("counterfactual_probe",))}
    return order.get(request_type, 100)


def _target_id(*, record_index: int | None, ordinal: int) -> str:
    if record_index is None or record_index < 0:
        return f"target-{ordinal}"
    return f"record-{record_index}"


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _nested_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return _optional_int(current)


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


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
    parser.add_argument("--plan", required=True)
    parser.add_argument("--collection-corpus", required=True)
    parser.add_argument("--question-property-mapping", required=True)
    parser.add_argument("--covered-fact-mapping", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--target-jsonl", default=None)
    parser.add_argument("--request-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--request-type", action="append", default=None)
    parser.add_argument("--priority", action="append", default=[])
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--max-requests-per-target", type=int, default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        plan_path=args.plan,
        collection_corpus_path=args.collection_corpus,
        question_property_mapping_path=args.question_property_mapping,
        covered_fact_mapping_path=args.covered_fact_mapping,
        output_dir=args.output_dir,
        report_json_path=args.report_json,
        target_jsonl_path=args.target_jsonl,
        request_jsonl_path=args.request_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        request_types=tuple(args.request_type or DEFAULT_REQUEST_TYPES),
        priorities=tuple(args.priority or ()),
        max_targets=args.max_targets,
        max_requests_per_target=args.max_requests_per_target,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "unresolved_blind_spot_evidence_queue_ok "
        f"status={payload['status']} "
        f"targets={summary['target_count']} "
        f"requests={summary['adapter_request_count']} "
        f"resolved={summary['resolved_by_question_property_count']} "
        f"citations={summary['request_type_counts'].get('external_citation', 0)} "
        f"rules={summary['request_type_counts'].get('world_model_or_calculator_rule', 0)}"
    )


if __name__ == "__main__":
    main()
