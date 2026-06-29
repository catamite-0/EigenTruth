"""Build lane-aware adapter batches from source-family structured QA triage.

``triage_source_family_structured_qa_gaps.py`` explains why remaining mapping
rows are not correction handoff candidates. This workflow lowers that triage
plus the fact-collection corpus into prioritized execution batches. It does not
collect evidence and does not copy labels or model answers into adapter rows.
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

WORKFLOW = "source_family_structured_qa_lane_execution_queue"
TRIAGE_WORKFLOW = "source_family_structured_qa_gap_triage"
COLLECTION_WORKFLOW = "source_family_structured_qa_fact_collection_corpus"

DEFAULT_LANE_STATUSES = (
    "blocked_needs_disambiguation",
    "needs_property_collection",
    "needs_entity_resolution",
    "needs_citation",
    "needs_source_family_coverage",
)
DEFAULT_REQUEST_TYPES = (
    "source_family_fact_disambiguation",
    "world_model_or_calculator_rule",
    "source_family_structured_fact",
    "entity_resolution",
    "external_citation",
)
REQUEST_TYPE_TO_ADAPTER = {
    "source_family_fact_disambiguation": "source_family_fact_disambiguation",
    "world_model_or_calculator_rule": "world_model_rule_authoring",
    "source_family_structured_fact": "source_family_structured_fact",
    "entity_resolution": "entity_resolution",
    "external_citation": "external_citation_search",
}
LANE_RANK = {
    "answer_collision_audit": 0,
    "richer_property_or_indicator_collection": 1,
    "citation_retrieval_before_handoff": 2,
    "source_family_coverage_expansion": 3,
    "entity_resolution_or_subject_collection": 4,
    "covered_fact_manual_audit": 5,
    "unclassified_gap": 6,
}
RESERVED_ADAPTER_FIELDS = {"answer", "model_answer", "label", "labels", "is_false", "score_label"}


def build_source_family_structured_qa_lane_execution_queue(
    *,
    triage: Mapping[str, Any],
    collection_corpus: Mapping[str, Any],
    lanes: Sequence[str] = (),
    lane_statuses: Sequence[str] = DEFAULT_LANE_STATUSES,
    request_types: Sequence[str] = DEFAULT_REQUEST_TYPES,
    max_targets: int | None = None,
    max_requests_per_target: int | None = None,
    max_requests_per_batch: int = 50,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready lane-aware execution queue."""
    _validate_inputs(triage=triage, collection_corpus=collection_corpus)
    selected_lanes = _normalize_optional_filter(lanes)
    selected_statuses = _normalize_optional_filter(lane_statuses)
    selected_request_types = _normalize_request_types(request_types)
    if max_targets is not None and int(max_targets) <= 0:
        raise ValueError("max_targets must be positive when provided.")
    if max_requests_per_target is not None and int(max_requests_per_target) <= 0:
        raise ValueError("max_requests_per_target must be positive when provided.")
    if int(max_requests_per_batch) <= 0:
        raise ValueError("max_requests_per_batch must be positive.")

    requests_by_target = _collection_requests_by_target(
        collection_corpus.get("requests", {}),
        request_types=selected_request_types,
    )
    candidates: list[dict[str, Any]] = []
    skipped_targets: list[dict[str, Any]] = []
    for ordinal, target in enumerate(_mapping_sequence(triage.get("triage_targets", ())), start=1):
        target_id = str(target.get("target_id") or _target_id(target, ordinal))
        lane = str(target.get("next_lane") or "")
        lane_status = str(target.get("lane_status") or "")
        if selected_lanes and lane not in selected_lanes:
            skipped_targets.append(_skip(target, target_id=target_id, reason="lane_filtered"))
            continue
        if selected_statuses and lane_status not in selected_statuses:
            skipped_targets.append(_skip(target, target_id=target_id, reason="lane_status_filtered"))
            continue
        target_requests = list(requests_by_target.get(target_id, ()))
        if not target_requests:
            skipped_targets.append(_skip(target, target_id=target_id, reason="no_selected_request_types"))
            continue
        candidates.append(_target_entry(target, target_id=target_id, request_count=len(target_requests)))

    candidates.sort(key=_target_sort_key)
    if max_targets is not None:
        keep_ids = {item["target_id"] for item in candidates[: int(max_targets)]}
        skipped_targets.extend(
            _skip(item, target_id=str(item["target_id"]), reason="outside_max_targets")
            for item in candidates
            if item["target_id"] not in keep_ids
        )
        candidates = [item for item in candidates if item["target_id"] in keep_ids]

    target_rank = {item["target_id"]: idx for idx, item in enumerate(candidates, start=1)}
    adapter_requests: list[dict[str, Any]] = []
    for target in candidates:
        raw_requests = list(requests_by_target.get(str(target["target_id"]), ()))
        raw_requests.sort(key=_request_sort_key)
        if max_requests_per_target is not None:
            raw_requests = raw_requests[: int(max_requests_per_target)]
        for request_ordinal, request in enumerate(raw_requests, start=1):
            adapter_requests.append(
                _adapter_request(
                    request,
                    target=target,
                    target_rank=target_rank[str(target["target_id"])],
                    request_ordinal=request_ordinal,
                )
            )

    batches = _batches(adapter_requests, max_requests_per_batch=int(max_requests_per_batch))
    summary = _summary(
        targets=candidates,
        adapter_requests=adapter_requests,
        batches=batches,
        skipped_targets=skipped_targets,
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready_for_adapter_execution" if adapter_requests else "empty",
        "scope": (
            "Lane-aware adapter and rule-authoring execution queue for "
            "source-family structured QA gaps. Rows are collection requests, "
            "not verifier evidence, and weak matches remain blocked from "
            "correction handoff."
        ),
        "source": {
            "triage_workflow": triage.get("workflow"),
            "triage_status": triage.get("status"),
            "triage_target_count": _nested_int(triage, "summary", "target_count"),
            "collection_workflow": collection_corpus.get("workflow"),
            "collection_status": collection_corpus.get("status"),
            "collection_target_count": _nested_int(collection_corpus, "summary", "target_count"),
            "collection_total_request_count": _nested_int(
                collection_corpus,
                "summary",
                "total_request_count",
            ),
        },
        "label_usage": {
            "labels_used_for_queue_selection": False,
            "answers_copied_to_adapter_requests": False,
            "model_answers_copied_to_adapter_requests": False,
            "adapter_requests_are_verifier_evidence": False,
        },
        "config": {
            "lanes": selected_lanes,
            "lane_statuses": selected_statuses,
            "request_types": selected_request_types,
            "max_targets": max_targets,
            "max_requests_per_target": max_requests_per_target,
            "max_requests_per_batch": int(max_requests_per_batch),
        },
        "summary": summary,
        "targets": tuple(candidates),
        "adapter_requests": tuple(adapter_requests),
        "execution_batches": tuple(batches),
        "skipped_targets": tuple(skipped_targets),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    triage_path: str | Path,
    collection_corpus_path: str | Path,
    output_dir: str | Path,
    report_json_path: str | Path | None = None,
    target_jsonl_path: str | Path | None = None,
    request_jsonl_path: str | Path | None = None,
    batch_jsonl_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    lanes: Sequence[str] = (),
    lane_statuses: Sequence[str] = DEFAULT_LANE_STATUSES,
    request_types: Sequence[str] = DEFAULT_REQUEST_TYPES,
    max_targets: int | None = None,
    max_requests_per_target: int | None = None,
    max_requests_per_batch: int = 50,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register a lane execution queue."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    report_path = Path(report_json_path) if report_json_path is not None else output / "lane-execution-queue.json"
    target_path = Path(target_jsonl_path) if target_jsonl_path is not None else output / "lane-targets.jsonl"
    request_path = Path(request_jsonl_path) if request_jsonl_path is not None else output / "adapter-requests.jsonl"
    batch_path = Path(batch_jsonl_path) if batch_jsonl_path is not None else output / "execution-batches.jsonl"
    manifest_path = (
        Path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else output / "artifact-manifest.json"
    )
    triage = _load_json_object(triage_path)
    collection = _load_json_object(collection_corpus_path)
    payload = build_source_family_structured_qa_lane_execution_queue(
        triage=triage,
        collection_corpus=collection,
        lanes=lanes,
        lane_statuses=lane_statuses,
        request_types=request_types,
        max_targets=max_targets,
        max_requests_per_target=max_requests_per_target,
        max_requests_per_batch=max_requests_per_batch,
        metadata=metadata,
    )
    report = dict(payload)
    report["paths"] = {
        "triage": str(triage_path),
        "collection_corpus": str(collection_corpus_path),
        "targets_jsonl": str(target_path),
        "adapter_requests_jsonl": str(request_path),
        "execution_batches_jsonl": str(batch_path),
        "artifact_manifest": str(manifest_path),
    }
    payload = dict(payload)
    payload["paths"] = report["paths"]

    _write_json(report_path, report, compact=compact_json)
    _write_jsonl(target_path, payload["targets"])
    _write_jsonl(request_path, payload["adapter_requests"])
    _write_jsonl(batch_path, payload["execution_batches"])
    manifest = build_artifact_manifest(
        {
            "source_family_structured_qa_lane_execution_queue": report_path,
            "lane_targets": target_path,
            "adapter_requests": request_path,
            "execution_batches": batch_path,
            "source_family_structured_qa_gap_triage": Path(triage_path),
            "source_family_structured_qa_fact_collection_corpus": Path(collection_corpus_path),
        },
        root=manifest_path.parent,
        metadata={
            "workflow": report["workflow"],
            "status": report["status"],
            "target_count": report["summary"]["target_count"],
            "adapter_request_count": report["summary"]["adapter_request_count"],
            "batch_count": report["summary"]["batch_count"],
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
                "batch_count": report["summary"]["batch_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _target_entry(target: Mapping[str, Any], *, target_id: str, request_count: int) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "record_index": _optional_int(target.get("record_index")),
        "next_lane": str(target.get("next_lane") or ""),
        "lane_status": str(target.get("lane_status") or ""),
        "priority_score": _optional_float(target.get("priority_score")) or 0.0,
        "blocked_from_handoff": bool(target.get("blocked_from_handoff", True)),
        "mapping_decision": str(target.get("mapping_decision") or ""),
        "gate_recommendation": str(target.get("gate_recommendation") or ""),
        "question_type": str(target.get("question_type") or ""),
        "question": str(target.get("question") or ""),
        "source_gap_type": str(target.get("source_gap_type") or ""),
        "source_priority": str(target.get("source_priority") or ""),
        "available_request_counts": dict(target.get("available_request_counts") or {}),
        "selected_request_count": int(request_count),
        "collection_task_types": tuple(str(item) for item in _sequence(target.get("collection_task_types"))),
        "world_model_rule_families": tuple(
            str(item) for item in _sequence(target.get("world_model_rule_families"))
        ),
        "top_fact_sources": tuple(str(item) for item in _sequence(target.get("top_fact_sources")))[:5],
        "source_family_targets": tuple(
            _safe_source_family_target(item)
            for item in _sequence(target.get("source_family_targets"))
        ),
    }


def _adapter_request(
    request: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    target_rank: int,
    request_ordinal: int,
) -> dict[str, Any]:
    request_type = str(request.get("request_type") or "")
    request_id = str(request.get("request_id") or "")
    query = str(request.get("query") or request.get("rule_seed") or "")
    output = {
        "queue_id": f"sfqa-lane:{target['target_id']}:{request_type}:{request_ordinal}",
        "source_request_id": request_id,
        "target_rank": int(target_rank),
        "target_id": target["target_id"],
        "record_index": target["record_index"],
        "next_lane": target["next_lane"],
        "lane_status": target["lane_status"],
        "adapter_family": REQUEST_TYPE_TO_ADAPTER.get(request_type, request_type),
        "request_type": request_type,
        "priority_score": target["priority_score"],
        "question_type": target["question_type"],
        "question": target["question"],
        "query": query,
        "source_gap_type": target["source_gap_type"],
        "mapping_decision": target["mapping_decision"],
        "requires_timestamp": bool(request.get("requires_timestamp")),
        "rule_family": request.get("rule_family"),
        "required_inputs": tuple(str(item) for item in _sequence(request.get("required_inputs"))),
        "source_family": request.get("source_family"),
        "provider_hint": request.get("provider_hint"),
        "property_hints": tuple(str(item) for item in _sequence(request.get("property_hints"))),
        "property_ids": tuple(str(item) for item in _sequence(request.get("property_ids"))),
        "source_family_hints": tuple(str(item) for item in _sequence(request.get("source_family_hints"))),
        "usage": request.get("usage", "source_discovery_only"),
        "not_verifier_evidence": True,
        "metadata": {
            "source_workflow": COLLECTION_WORKFLOW,
            "queue_workflow": WORKFLOW,
            "source_request_id": request_id,
            "target_id": target["target_id"],
            "selected_adapter_family": REQUEST_TYPE_TO_ADAPTER.get(request_type, request_type),
        },
    }
    return {key: value for key, value in output.items() if key not in RESERVED_ADAPTER_FIELDS}


def _batches(
    adapter_requests: Sequence[Mapping[str, Any]],
    *,
    max_requests_per_batch: int,
) -> tuple[dict[str, Any], ...]:
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for request in adapter_requests:
        grouped[(str(request.get("next_lane")), str(request.get("request_type")))].append(request)

    batches: list[dict[str, Any]] = []
    ordinal = 1
    for (lane, request_type), rows in sorted(grouped.items(), key=_batch_group_sort_key):
        rows = sorted(rows, key=_queued_request_sort_key)
        for offset in range(0, len(rows), max_requests_per_batch):
            chunk = rows[offset : offset + max_requests_per_batch]
            target_ids = tuple(dict.fromkeys(str(item.get("target_id")) for item in chunk))
            batches.append({
                "batch_id": f"sfqa-lane-batch-{ordinal:04d}",
                "next_lane": lane,
                "lane_status": str(chunk[0].get("lane_status") or "") if chunk else "",
                "request_type": request_type,
                "adapter_family": str(chunk[0].get("adapter_family") or request_type) if chunk else request_type,
                "request_count": len(chunk),
                "target_count": len(target_ids),
                "target_ids": target_ids,
                "source_request_ids": tuple(str(item.get("source_request_id")) for item in chunk),
                "min_priority_score": min(float(item.get("priority_score") or 0.0) for item in chunk),
                "max_priority_score": max(float(item.get("priority_score") or 0.0) for item in chunk),
                "not_verifier_evidence": True,
            })
            ordinal += 1
    return tuple(batches)


def _summary(
    *,
    targets: Sequence[Mapping[str, Any]],
    adapter_requests: Sequence[Mapping[str, Any]],
    batches: Sequence[Mapping[str, Any]],
    skipped_targets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lane_counts = Counter(str(item.get("next_lane")) for item in targets)
    lane_status_counts = Counter(str(item.get("lane_status")) for item in targets)
    request_type_counts = Counter(str(item.get("request_type")) for item in adapter_requests)
    adapter_counts = Counter(str(item.get("adapter_family")) for item in adapter_requests)
    question_counts = Counter(str(item.get("question_type")) for item in targets)
    skipped_counts = Counter(str(item.get("reason")) for item in skipped_targets)
    return {
        "target_count": len(targets),
        "adapter_request_count": len(adapter_requests),
        "batch_count": len(batches),
        "lane_counts": _sorted_counter(lane_counts),
        "lane_status_counts": _sorted_counter(lane_status_counts),
        "request_type_counts": _sorted_counter(request_type_counts),
        "adapter_family_counts": _sorted_counter(adapter_counts),
        "question_type_counts": _sorted_counter(question_counts),
        "skipped_target_counts": _sorted_counter(skipped_counts),
        "targets_with_world_model_or_calculator_rule": _target_count_for_request_type(
            adapter_requests,
            "world_model_or_calculator_rule",
        ),
        "targets_with_external_citation": _target_count_for_request_type(adapter_requests, "external_citation"),
        "top_target": None
        if not targets
        else {
            "target_id": targets[0]["target_id"],
            "record_index": targets[0]["record_index"],
            "next_lane": targets[0]["next_lane"],
            "lane_status": targets[0]["lane_status"],
            "priority_score": targets[0]["priority_score"],
        },
        "top_batch": None
        if not batches
        else {
            "batch_id": batches[0]["batch_id"],
            "next_lane": batches[0]["next_lane"],
            "request_type": batches[0]["request_type"],
            "request_count": batches[0]["request_count"],
        },
    }


def _validate_inputs(*, triage: Mapping[str, Any], collection_corpus: Mapping[str, Any]) -> None:
    if triage.get("workflow") != TRIAGE_WORKFLOW:
        raise ValueError(f"triage must be a {TRIAGE_WORKFLOW} report.")
    if collection_corpus.get("workflow") != COLLECTION_WORKFLOW:
        raise ValueError(f"collection_corpus must be a {COLLECTION_WORKFLOW} report.")


def _normalize_request_types(values: Sequence[str]) -> tuple[str, ...]:
    request_types = tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if not request_types:
        raise ValueError("at least one request type is required.")
    invalid = sorted(set(request_types) - set(REQUEST_TYPE_TO_ADAPTER))
    if invalid:
        raise ValueError(f"unsupported request types: {', '.join(invalid)}")
    return request_types


def _normalize_optional_filter(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _collection_requests_by_target(
    requests_payload: Any,
    *,
    request_types: Sequence[str],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    if not isinstance(requests_payload, Mapping):
        return {}
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for request_type in request_types:
        for request in _mapping_sequence(requests_payload.get(request_type, ())):
            target_id = str(request.get("target_id") or "")
            if target_id:
                grouped[target_id].append(request)
    return {key: tuple(value) for key, value in grouped.items()}


def _target_sort_key(item: Mapping[str, Any]) -> tuple[float, int, int, str]:
    return (
        -float(item.get("priority_score") or 0.0),
        LANE_RANK.get(str(item.get("next_lane")), 100),
        int(item.get("record_index") or 10**12),
        str(item.get("target_id")),
    )


def _request_sort_key(item: Mapping[str, Any]) -> tuple[int, str]:
    request_type_rank = {name: idx for idx, name in enumerate(DEFAULT_REQUEST_TYPES)}
    return (request_type_rank.get(str(item.get("request_type")), 100), str(item.get("request_id") or ""))


def _queued_request_sort_key(item: Mapping[str, Any]) -> tuple[float, int, str]:
    return (
        -float(item.get("priority_score") or 0.0),
        int(item.get("target_rank") or 10**12),
        str(item.get("source_request_id")),
    )


def _batch_group_sort_key(item: tuple[tuple[str, str], Sequence[Mapping[str, Any]]]) -> tuple[int, int, str, str]:
    lane, request_type = item[0]
    request_type_rank = {name: idx for idx, name in enumerate(DEFAULT_REQUEST_TYPES)}
    return (
        LANE_RANK.get(lane, 100),
        request_type_rank.get(request_type, 100),
        lane,
        request_type,
    )


def _target_count_for_request_type(rows: Sequence[Mapping[str, Any]], request_type: str) -> int:
    return len({str(item.get("target_id")) for item in rows if item.get("request_type") == request_type})


def _skip(target: Mapping[str, Any], *, target_id: str, reason: str) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "record_index": _optional_int(target.get("record_index")),
        "next_lane": str(target.get("next_lane") or ""),
        "lane_status": str(target.get("lane_status") or ""),
        "reason": reason,
    }


def _safe_source_family_target(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "provider": str(value.get("provider") or ""),
        "source_family": str(value.get("source_family") or ""),
        "reason": str(value.get("reason") or ""),
    }


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _target_id(target: Mapping[str, Any], ordinal: int) -> str:
    record_index = _optional_int(target.get("record_index"))
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
    parser.add_argument("--triage", required=True)
    parser.add_argument("--collection-corpus", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--target-jsonl", default=None)
    parser.add_argument("--request-jsonl", default=None)
    parser.add_argument("--batch-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--lane", action="append", default=[])
    parser.add_argument("--lane-status", action="append", default=None)
    parser.add_argument("--request-type", action="append", default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--max-requests-per-target", type=int, default=None)
    parser.add_argument("--max-requests-per-batch", type=int, default=50)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        triage_path=args.triage,
        collection_corpus_path=args.collection_corpus,
        output_dir=args.output_dir,
        report_json_path=args.report_json,
        target_jsonl_path=args.target_jsonl,
        request_jsonl_path=args.request_jsonl,
        batch_jsonl_path=args.batch_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        lanes=tuple(args.lane or ()),
        lane_statuses=tuple(args.lane_status or DEFAULT_LANE_STATUSES),
        request_types=tuple(args.request_type or DEFAULT_REQUEST_TYPES),
        max_targets=args.max_targets,
        max_requests_per_target=args.max_requests_per_target,
        max_requests_per_batch=args.max_requests_per_batch,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "source_family_structured_qa_lane_execution_queue_ok "
        f"status={payload['status']} "
        f"targets={summary['target_count']} "
        f"requests={summary['adapter_request_count']} "
        f"batches={summary['batch_count']} "
        f"top_lane={next(iter(summary['lane_counts']), 'none')}"
    )


if __name__ == "__main__":
    main()
