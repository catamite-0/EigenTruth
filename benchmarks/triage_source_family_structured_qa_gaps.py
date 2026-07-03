"""Triage source-family structured QA mapping gaps into next-action lanes.

This workflow sits after ``audit_source_family_structured_qa_claim_mapping.py``.
It does not collect new evidence and does not relax the mapping gate. Instead,
it turns each conservative mapping decision into an explicit next lane:
correction handoff, answer-support audit, answer-collision audit, richer
property/indicator collection, citation retrieval, entity resolution, or
source-family coverage expansion.
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

WORKFLOW = "source_family_structured_qa_gap_triage"

LANE_BY_DECISION = {
    "mapped_qa_fact_candidate": "structured_qa_correction_handoff",
    "answer_value_supported_by_covered_fact": "answer_support_audit",
    "answer_entity_collision": "answer_collision_audit",
    "subject_only_or_missing_intent": "richer_property_or_indicator_collection",
    "intent_only_or_missing_subject": "entity_resolution_or_subject_collection",
    "weak_textual_overlap": "citation_retrieval_before_handoff",
    "no_candidate_fact": "source_family_coverage_expansion",
    "covered_fact_match_without_correction": "covered_fact_manual_audit",
}

LANE_STATUS = {
    "structured_qa_correction_handoff": "handoff_ready",
    "answer_support_audit": "audit_only",
    "answer_collision_audit": "blocked_needs_disambiguation",
    "richer_property_or_indicator_collection": "needs_property_collection",
    "entity_resolution_or_subject_collection": "needs_entity_resolution",
    "citation_retrieval_before_handoff": "needs_citation",
    "source_family_coverage_expansion": "needs_source_family_coverage",
    "covered_fact_manual_audit": "needs_manual_audit",
    "unclassified_gap": "needs_triage",
}

LANE_PRIORITY = {
    "structured_qa_correction_handoff": 120.0,
    "answer_collision_audit": 90.0,
    "richer_property_or_indicator_collection": 82.0,
    "citation_retrieval_before_handoff": 76.0,
    "source_family_coverage_expansion": 70.0,
    "entity_resolution_or_subject_collection": 66.0,
    "covered_fact_manual_audit": 50.0,
    "answer_support_audit": 25.0,
    "unclassified_gap": 40.0,
}

CLOSURE_ROUTE_BY_LANE = {
    "structured_qa_correction_handoff": "structured_qa_correction_handoff",
    "answer_support_audit": "answer_support_audit",
    "answer_collision_audit": "entity_disambiguation",
    "richer_property_or_indicator_collection": "property_or_indicator_collection",
    "entity_resolution_or_subject_collection": "entity_resolution",
    "citation_retrieval_before_handoff": "citation_evidence_collection",
    "source_family_coverage_expansion": "source_family_coverage_expansion",
    "covered_fact_manual_audit": "manual_audit",
    "unclassified_gap": "triage",
}

CLOSURE_ROUTE_BY_REQUEST_TYPE = {
    "source_family_fact_disambiguation": "entity_disambiguation",
    "world_model_or_calculator_rule": "world_model_rule_authoring",
    "source_family_structured_fact": "property_or_indicator_collection",
    "entity_resolution": "entity_resolution",
    "external_citation": "citation_evidence_collection",
}

QUESTION_TYPE_PRIORITY = {
    "quantity": 12.0,
    "temporal": 10.0,
    "causal": 10.0,
    "method": 8.0,
    "person": 5.0,
}


def triage_source_family_structured_qa_gaps(
    *,
    claim_mapping: Mapping[str, Any],
    fact_expansion_plan: Mapping[str, Any] | None = None,
    fact_collection_corpus: Mapping[str, Any] | None = None,
    fact_collection_workflow: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready source-family mapping-gap triage report."""
    _validate_claim_mapping(claim_mapping)
    _validate_optional_inputs(
        fact_expansion_plan=fact_expansion_plan,
        fact_collection_corpus=fact_collection_corpus,
        fact_collection_workflow=fact_collection_workflow,
    )
    plan_by_record = _records_by_index(() if fact_expansion_plan is None else fact_expansion_plan.get("targets", ()))
    corpus_by_record = _records_by_index(
        () if fact_collection_corpus is None else fact_collection_corpus.get("targets", ())
    )
    request_counts = _request_counts_by_target(
        None if fact_collection_corpus is None else fact_collection_corpus.get("requests")
    )
    targets = tuple(
        _triage_record(
            record,
            plan_target=plan_by_record.get(_optional_int(record.get("record_index"))),
            corpus_target=corpus_by_record.get(_optional_int(record.get("record_index"))),
            request_counts_by_target=request_counts,
        )
        for record in _mapping_sequence(claim_mapping.get("records", ()))
    )
    summary = _summary(
        targets=targets,
        claim_mapping=claim_mapping,
        fact_expansion_plan=fact_expansion_plan,
        fact_collection_corpus=fact_collection_corpus,
        fact_collection_workflow=fact_collection_workflow,
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": _status(summary),
        "scope": (
            "Classifies conservative source-family structured QA mapping "
            "decisions into explicit next-action lanes. This report is "
            "routing/triage evidence only; it is not verifier evidence and "
            "does not promote weak matches."
        ),
        "source": {
            "claim_mapping_workflow": claim_mapping.get("workflow"),
            "claim_mapping_status": claim_mapping.get("status"),
            "claim_mapping_target_count": _nested_int(claim_mapping, "summary", "target_count"),
            "mapped_qa_fact_candidate_count": _nested_int(
                claim_mapping,
                "summary",
                "mapped_qa_fact_candidate_count",
            ),
            "fact_expansion_plan_workflow": None
            if fact_expansion_plan is None
            else fact_expansion_plan.get("workflow"),
            "fact_expansion_plan_status": None
            if fact_expansion_plan is None
            else fact_expansion_plan.get("status"),
            "fact_collection_corpus_workflow": None
            if fact_collection_corpus is None
            else fact_collection_corpus.get("workflow"),
            "fact_collection_corpus_status": None
            if fact_collection_corpus is None
            else fact_collection_corpus.get("status"),
            "fact_collection_workflow": None
            if fact_collection_workflow is None
            else fact_collection_workflow.get("workflow"),
            "fact_collection_workflow_status": None
            if fact_collection_workflow is None
            else fact_collection_workflow.get("status"),
        },
        "label_usage": {
            "labels_used_for_triage": False,
            "labels_copied_to_targets": False,
            "triage_targets_are_verifier_evidence": False,
        },
        "summary": summary,
        "triage_targets": targets,
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    claim_mapping_path: str | Path,
    output_dir: str | Path,
    report_json_path: str | Path | None = None,
    target_jsonl_path: str | Path | None = None,
    fact_expansion_plan_path: str | Path | None = None,
    fact_collection_corpus_path: str | Path | None = None,
    fact_collection_workflow_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register a gap triage report."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    report_path = Path(report_json_path) if report_json_path is not None else output / "gap-triage.json"
    target_path = Path(target_jsonl_path) if target_jsonl_path is not None else output / "triage-targets.jsonl"
    manifest_path = (
        Path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else output / "artifact-manifest.json"
    )
    claim_mapping = _load_json_object(claim_mapping_path)
    fact_expansion_plan = (
        None if fact_expansion_plan_path is None else _load_json_object(fact_expansion_plan_path)
    )
    fact_collection_corpus = (
        None if fact_collection_corpus_path is None else _load_json_object(fact_collection_corpus_path)
    )
    fact_collection_workflow = (
        None if fact_collection_workflow_path is None else _load_json_object(fact_collection_workflow_path)
    )
    payload = triage_source_family_structured_qa_gaps(
        claim_mapping=claim_mapping,
        fact_expansion_plan=fact_expansion_plan,
        fact_collection_corpus=fact_collection_corpus,
        fact_collection_workflow=fact_collection_workflow,
        metadata=metadata,
    )
    report = dict(payload)
    report["paths"] = {
        "claim_mapping": str(claim_mapping_path),
        "fact_expansion_plan": None
        if fact_expansion_plan_path is None
        else str(fact_expansion_plan_path),
        "fact_collection_corpus": None
        if fact_collection_corpus_path is None
        else str(fact_collection_corpus_path),
        "fact_collection_workflow": None
        if fact_collection_workflow_path is None
        else str(fact_collection_workflow_path),
        "triage_targets_jsonl": str(target_path),
        "artifact_manifest": str(manifest_path),
    }
    payload = dict(payload)
    payload["paths"] = report["paths"]

    _write_json(report_path, report, compact=compact_json)
    _write_jsonl(target_path, payload["triage_targets"])
    manifest = build_artifact_manifest(
        {
            "source_family_structured_qa_gap_triage": report_path,
            "triage_targets": target_path,
            "source_family_structured_qa_claim_mapping": Path(claim_mapping_path),
            "source_family_structured_qa_fact_expansion_plan": None
            if fact_expansion_plan_path is None
            else Path(fact_expansion_plan_path),
            "source_family_structured_qa_fact_collection_corpus": None
            if fact_collection_corpus_path is None
            else Path(fact_collection_corpus_path),
            "source_family_structured_qa_fact_collection_workflow": None
            if fact_collection_workflow_path is None
            else Path(fact_collection_workflow_path),
        },
        root=manifest_path.parent,
        metadata={
            "workflow": report["workflow"],
            "status": report["status"],
            "target_count": report["summary"]["target_count"],
            "handoff_ready_count": report["summary"]["lane_status_counts"].get("handoff_ready", 0),
            "audit_only_count": report["summary"]["lane_status_counts"].get("audit_only", 0),
            "blocked_count": report["summary"]["blocked_target_count"],
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
                "handoff_ready_count": report["summary"]["lane_status_counts"].get("handoff_ready", 0),
                "audit_only_count": report["summary"]["lane_status_counts"].get("audit_only", 0),
                "blocked_count": report["summary"]["blocked_target_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _triage_record(
    record: Mapping[str, Any],
    *,
    plan_target: Mapping[str, Any] | None,
    corpus_target: Mapping[str, Any] | None,
    request_counts_by_target: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    decision = str(record.get("mapping_decision") or "unknown")
    lane = LANE_BY_DECISION.get(decision, "unclassified_gap")
    lane_status = LANE_STATUS[lane]
    record_index = _optional_int(record.get("record_index"))
    record_id = str(record.get("record_id") or _target_id(record_index))
    target_id = str(
        (corpus_target or {}).get("target_id")
        or (plan_target or {}).get("target_id")
        or record_id
    )
    request_counts = dict(request_counts_by_target.get(target_id, {}))
    collection_task_types = tuple(
        str(item.get("task_type"))
        for item in _mapping_sequence((plan_target or {}).get("collection_tasks", ()))
    )
    world_model_rule_families = tuple(
        str(item.get("rule_family"))
        for item in _mapping_sequence((plan_target or {}).get("world_model_rule_targets", ()))
        if item.get("rule_family")
    )
    question_type = str(record.get("question_type") or (plan_target or {}).get("question_type") or "unknown")
    priority_score = _priority_score(
        lane=lane,
        question_type=question_type,
        request_counts=request_counts,
        world_model_rule_families=world_model_rule_families,
    )
    primary_closure_route = CLOSURE_ROUTE_BY_LANE.get(lane, CLOSURE_ROUTE_BY_LANE["unclassified_gap"])
    closure_routes = _closure_routes(
        primary_closure_route=primary_closure_route,
        request_counts=request_counts,
        world_model_rule_families=world_model_rule_families,
    )
    return {
        "target_id": target_id,
        "record_id": record_id,
        "record_index": record_index,
        "claim_id": record.get("claim_id"),
        "question": str(record.get("question") or ""),
        "answer": str(record.get("answer") or ""),
        "question_type": question_type,
        "mapping_decision": decision,
        "gate_recommendation": record.get("gate_recommendation"),
        "next_lane": lane,
        "lane_status": lane_status,
        "primary_closure_route": primary_closure_route,
        "closure_routes": closure_routes,
        "priority_score": priority_score,
        "blocked_from_handoff": lane_status != "handoff_ready",
        "covered_fact_match": bool(record.get("covered_fact_match")),
        "mapped_qa_fact_candidate": bool(record.get("mapped_qa_fact_candidate")),
        "answer_value_supported": bool(record.get("answer_value_supported")),
        "answer_entity_collision": bool(record.get("answer_entity_collision")),
        "best_mapping_score": _optional_float(record.get("best_mapping_score")),
        "best_subject_coverage": _optional_float(record.get("best_subject_coverage")),
        "best_intent_score": _optional_float(record.get("best_intent_score")),
        "available_request_counts": request_counts,
        "collection_task_types": collection_task_types,
        "world_model_rule_families": world_model_rule_families,
        "source_gap_type": None if plan_target is None else plan_target.get("gap_type"),
        "source_priority": None
        if corpus_target is None
        else str(corpus_target.get("priority") or ""),
        "source_family_targets": tuple(
            _compact_source_family_target(item)
            for item in _mapping_sequence((corpus_target or {}).get("source_family_targets", ()))
        ),
        "top_fact_sources": tuple(_top_fact_sources(record)),
    }


def _summary(
    *,
    targets: Sequence[Mapping[str, Any]],
    claim_mapping: Mapping[str, Any],
    fact_expansion_plan: Mapping[str, Any] | None,
    fact_collection_corpus: Mapping[str, Any] | None,
    fact_collection_workflow: Mapping[str, Any] | None,
) -> dict[str, Any]:
    lane_counts = Counter(str(item.get("next_lane")) for item in targets)
    lane_status_counts = Counter(str(item.get("lane_status")) for item in targets)
    primary_closure_route_counts = Counter(str(item.get("primary_closure_route")) for item in targets)
    closure_route_counts: Counter[str] = Counter()
    decision_counts = Counter(str(item.get("mapping_decision")) for item in targets)
    question_type_counts = Counter(str(item.get("question_type")) for item in targets)
    source_gap_type_counts = Counter(
        str(item.get("source_gap_type") or "")
        for item in targets
        if str(item.get("source_gap_type") or "")
    )
    request_type_counts: Counter[str] = Counter()
    task_type_counts: Counter[str] = Counter()
    world_model_rule_counts: Counter[str] = Counter()
    targets_with_citation = 0
    targets_with_world_model_rules = 0
    for target in targets:
        request_counts = target.get("available_request_counts")
        if isinstance(request_counts, Mapping):
            request_type_counts.update({str(key): int(value) for key, value in request_counts.items()})
            if int(request_counts.get("external_citation", 0)) > 0:
                targets_with_citation += 1
            if int(request_counts.get("world_model_or_calculator_rule", 0)) > 0:
                targets_with_world_model_rules += 1
        task_type_counts.update(str(item) for item in _sequence(target.get("collection_task_types")))
        world_model_rule_counts.update(str(item) for item in _sequence(target.get("world_model_rule_families")))
        closure_route_counts.update(str(item) for item in _sequence(target.get("closure_routes")))
    blocked_count = sum(1 for item in targets if bool(item.get("blocked_from_handoff")))
    top_targets = tuple(
        {
            "target_id": item.get("target_id"),
            "record_index": item.get("record_index"),
            "next_lane": item.get("next_lane"),
            "lane_status": item.get("lane_status"),
            "primary_closure_route": item.get("primary_closure_route"),
            "source_gap_type": item.get("source_gap_type"),
            "priority_score": item.get("priority_score"),
            "question_type": item.get("question_type"),
        }
        for item in sorted(
            targets,
            key=lambda value: (
                -float(value.get("priority_score") or 0.0),
                int(value.get("record_index") or 10**12),
                str(value.get("target_id")),
            ),
        )[:10]
    )
    return {
        "target_count": len(targets),
        "blocked_target_count": blocked_count,
        "handoff_ready_count": lane_status_counts.get("handoff_ready", 0),
        "audit_only_count": lane_status_counts.get("audit_only", 0),
        "needs_collection_count": sum(
            count
            for status, count in lane_status_counts.items()
            if str(status).startswith("needs_") or str(status).startswith("blocked_")
        ),
        "lane_counts": _sorted_counter(lane_counts),
        "lane_status_counts": _sorted_counter(lane_status_counts),
        "primary_closure_route_counts": _sorted_counter(primary_closure_route_counts),
        "closure_route_counts": _sorted_counter(closure_route_counts),
        "mapping_decision_counts": _sorted_counter(decision_counts),
        "question_type_counts": _sorted_counter(question_type_counts),
        "source_gap_type_counts": _sorted_counter(source_gap_type_counts),
        "available_request_type_counts": _sorted_counter(request_type_counts),
        "collection_task_type_counts": _sorted_counter(task_type_counts),
        "world_model_rule_family_counts": _sorted_counter(world_model_rule_counts),
        "targets_with_external_citation": targets_with_citation,
        "targets_with_world_model_or_calculator_rule": targets_with_world_model_rules,
        "claim_mapping_summary": dict(claim_mapping.get("summary") or {}),
        "fact_expansion_plan_summary": None
        if fact_expansion_plan is None
        else dict(fact_expansion_plan.get("summary") or {}),
        "fact_collection_corpus_summary": None
        if fact_collection_corpus is None
        else dict(fact_collection_corpus.get("summary") or {}),
        "fact_collection_workflow_summary": None
        if fact_collection_workflow is None
        else dict(fact_collection_workflow.get("summary") or {}),
        "top_targets": top_targets,
    }


def _status(summary: Mapping[str, Any]) -> str:
    if int(summary.get("handoff_ready_count") or 0) > 0:
        return "handoff_ready"
    if int(summary.get("needs_collection_count") or 0) > 0:
        return "needs_collection"
    if int(summary.get("audit_only_count") or 0) > 0:
        return "audit_only"
    return "empty"


def _priority_score(
    *,
    lane: str,
    question_type: str,
    request_counts: Mapping[str, int],
    world_model_rule_families: Sequence[str],
) -> float:
    score = float(LANE_PRIORITY.get(lane, LANE_PRIORITY["unclassified_gap"]))
    score += float(QUESTION_TYPE_PRIORITY.get(question_type, 0.0))
    score += min(sum(int(value) for value in request_counts.values()), 8)
    if request_counts.get("external_citation", 0):
        score += 6.0
    if request_counts.get("source_family_structured_fact", 0):
        score += 5.0
    if request_counts.get("source_family_fact_disambiguation", 0):
        score += 4.0
    if request_counts.get("world_model_or_calculator_rule", 0) or world_model_rule_families:
        score += 7.0
    return score


def _closure_routes(
    *,
    primary_closure_route: str,
    request_counts: Mapping[str, int],
    world_model_rule_families: Sequence[str],
) -> tuple[str, ...]:
    routes = [primary_closure_route]
    for request_type in CLOSURE_ROUTE_BY_REQUEST_TYPE:
        if (_optional_int(request_counts.get(request_type)) or 0) > 0:
            routes.append(CLOSURE_ROUTE_BY_REQUEST_TYPE[request_type])
    if world_model_rule_families:
        routes.append(CLOSURE_ROUTE_BY_REQUEST_TYPE["world_model_or_calculator_rule"])
    return tuple(dict.fromkeys(route for route in routes if route))


def _request_counts_by_target(requests_payload: Any) -> dict[str, dict[str, int]]:
    if not isinstance(requests_payload, Mapping):
        return {}
    grouped: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for request_type, rows in requests_payload.items():
        for request in _mapping_sequence(rows):
            target_id = str(request.get("target_id") or "")
            if target_id:
                grouped[target_id][str(request.get("request_type") or request_type)] += 1
    return {target_id: dict(counter) for target_id, counter in grouped.items()}


def _top_fact_sources(record: Mapping[str, Any]) -> tuple[str, ...]:
    sources: list[str] = []
    for key in ("mapped_facts", "supported_facts", "collision_facts", "top_fact_candidates"):
        for item in _mapping_sequence(record.get(key, ())):
            source = str(item.get("source") or "")
            if source:
                sources.append(source)
    return tuple(dict.fromkeys(sources))[:5]


def _compact_source_family_target(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "provider": item.get("provider"),
        "source_family": item.get("source_family"),
        "reason": item.get("reason"),
    }


def _validate_claim_mapping(payload: Mapping[str, Any]) -> None:
    if payload.get("workflow") != "source_family_structured_qa_claim_mapping_audit":
        raise ValueError("claim_mapping must be a source_family_structured_qa_claim_mapping_audit report.")
    if not isinstance(payload.get("records"), Sequence):
        raise ValueError("claim_mapping must contain records.")


def _validate_optional_inputs(
    *,
    fact_expansion_plan: Mapping[str, Any] | None,
    fact_collection_corpus: Mapping[str, Any] | None,
    fact_collection_workflow: Mapping[str, Any] | None,
) -> None:
    if (
        fact_expansion_plan is not None
        and fact_expansion_plan.get("workflow") != "source_family_structured_qa_fact_expansion_plan"
    ):
        raise ValueError("fact_expansion_plan must be a source_family_structured_qa_fact_expansion_plan report.")
    if (
        fact_collection_corpus is not None
        and fact_collection_corpus.get("workflow") != "source_family_structured_qa_fact_collection_corpus"
    ):
        raise ValueError("fact_collection_corpus must be a source_family_structured_qa_fact_collection_corpus report.")
    if (
        fact_collection_workflow is not None
        and fact_collection_workflow.get("workflow") != "source_family_structured_qa_fact_collection_workflow"
    ):
        raise ValueError(
            "fact_collection_workflow must be a source_family_structured_qa_fact_collection_workflow report."
        )


def _records_by_index(records: Any) -> dict[int | None, Mapping[str, Any]]:
    output: dict[int | None, Mapping[str, Any]] = {}
    for record in _mapping_sequence(records):
        record_index = _optional_int(record.get("record_index"))
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


def _target_id(record_index: int | None) -> str:
    return "target-unknown" if record_index is None else f"record-{record_index}"


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
    parser.add_argument("--claim-mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--target-jsonl", default=None)
    parser.add_argument("--fact-expansion-plan", default=None)
    parser.add_argument("--fact-collection-corpus", default=None)
    parser.add_argument("--fact-collection-workflow", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        claim_mapping_path=args.claim_mapping,
        output_dir=args.output_dir,
        report_json_path=args.json,
        target_jsonl_path=args.target_jsonl,
        fact_expansion_plan_path=args.fact_expansion_plan,
        fact_collection_corpus_path=args.fact_collection_corpus,
        fact_collection_workflow_path=args.fact_collection_workflow,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "source_family_structured_qa_gap_triage_ok "
        f"status={payload['status']} "
        f"targets={summary['target_count']} "
        f"handoff_ready={summary['handoff_ready_count']} "
        f"audit_only={summary['audit_only_count']} "
        f"blocked={summary['blocked_target_count']} "
        f"top_lane={next(iter(summary['lane_counts']), 'none')}"
    )


if __name__ == "__main__":
    main()
