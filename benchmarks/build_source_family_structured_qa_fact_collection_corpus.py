"""Compile source-family structured QA fact-expansion plans into request queues.

``plan_source_family_structured_qa_fact_expansion.py`` identifies which claims
need new structured facts, citations, entity resolution, disambiguation, or
world-model rules. This compiler lowers that plan into adapter-ready JSONL
request buckets. The requests are source-discovery and rule-authoring inputs
only; they are not verifier evidence, and they intentionally do not copy labels
or model answers into external collection requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "source_family_structured_qa_fact_collection_corpus"
PLAN_WORKFLOW = "source_family_structured_qa_fact_expansion_plan"
DEFAULT_PRIORITIES = ("high", "medium", "low")
DEFAULT_MAX_FACT_REQUESTS_PER_TARGET = 4
DEFAULT_MAX_ENTITY_REQUESTS_PER_TARGET = 3
DEFAULT_MAX_CITATION_REQUESTS_PER_TARGET = 3
DEFAULT_MAX_RULE_REQUESTS_PER_TARGET = 3
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.'-]*|\d+(?:\.\d+)?")
PROPERTY_ID_RE = re.compile(r"\b(P[1-9][0-9]*)\b")
REQUEST_BUCKETS = {
    "source_family_structured_fact": "structured-fact-requests.jsonl",
    "entity_resolution": "entity-resolution-requests.jsonl",
    "external_citation": "citation-requests.jsonl",
    "source_family_fact_disambiguation": "fact-disambiguation-requests.jsonl",
    "world_model_or_calculator_rule": "world-model-rule-requests.jsonl",
}
RESERVED_REQUEST_FIELDS = {
    "answer",
    "is_false",
    "label",
    "labels",
    "model_answer",
    "row_index",
    "score_label",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def build_source_family_structured_qa_fact_collection_corpus(
    *,
    plan_path: str | Path,
    output_dir: str | Path,
    priorities: Sequence[str] = DEFAULT_PRIORITIES,
    task_types: Sequence[str] = (),
    max_targets: int | None = None,
    max_fact_requests_per_target: int = DEFAULT_MAX_FACT_REQUESTS_PER_TARGET,
    max_entity_requests_per_target: int = DEFAULT_MAX_ENTITY_REQUESTS_PER_TARGET,
    max_citation_requests_per_target: int = DEFAULT_MAX_CITATION_REQUESTS_PER_TARGET,
    max_rule_requests_per_target: int = DEFAULT_MAX_RULE_REQUESTS_PER_TARGET,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready request corpus and write JSONL sidecars."""
    plan = _load_plan(plan_path)
    selected_priorities = _normalize_priorities(priorities)
    selected_task_types = tuple(dict.fromkeys(str(item).strip() for item in task_types if str(item).strip()))
    _validate_positive("max_fact_requests_per_target", max_fact_requests_per_target)
    _validate_positive("max_entity_requests_per_target", max_entity_requests_per_target)
    _validate_positive("max_citation_requests_per_target", max_citation_requests_per_target)
    _validate_positive("max_rule_requests_per_target", max_rule_requests_per_target)
    if max_targets is not None and int(max_targets) <= 0:
        raise ValueError("max_targets must be positive when provided.")

    selected_targets = tuple(
        _select_targets(
            plan.get("targets", ()),
            priorities=selected_priorities,
            task_types=selected_task_types,
            max_targets=max_targets,
        )
    )
    targets: list[dict[str, Any]] = []
    requests: dict[str, list[dict[str, Any]]] = {name: [] for name in REQUEST_BUCKETS}
    source_discovery_documents: list[dict[str, Any]] = []
    for ordinal, target in enumerate(selected_targets, start=1):
        target_id = str(target.get("target_id") or _target_id(target, ordinal))
        snapshot = _target_snapshot(target, target_id=target_id, ordinal=ordinal)
        targets.append(snapshot)
        fact_requests = _structured_fact_requests(
            target,
            target_id=target_id,
            max_items=int(max_fact_requests_per_target),
        )
        entity_requests = _entity_resolution_requests(
            target,
            target_id=target_id,
            max_items=int(max_entity_requests_per_target),
        )
        citation_requests = _citation_requests(
            target,
            target_id=target_id,
            max_items=int(max_citation_requests_per_target),
        )
        disambiguation_requests = _disambiguation_requests(target, target_id=target_id)
        rule_requests = _rule_requests(
            target,
            target_id=target_id,
            max_items=int(max_rule_requests_per_target),
        )
        requests["source_family_structured_fact"].extend(fact_requests)
        requests["entity_resolution"].extend(entity_requests)
        requests["external_citation"].extend(citation_requests)
        requests["source_family_fact_disambiguation"].extend(disambiguation_requests)
        requests["world_model_or_calculator_rule"].extend(rule_requests)
        source_discovery_documents.extend(
            _source_discovery_documents((*fact_requests, *entity_requests, *citation_requests))
        )

    output = Path(output_dir)
    paths = _write_sidecars(output, requests=requests, source_discovery_documents=source_discovery_documents)
    summary = _summary(
        targets=targets,
        requests=requests,
        source_discovery_documents=source_discovery_documents,
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready_for_collection" if summary["total_request_count"] else "empty",
        "scope": (
            "Adapter-ready source-discovery and rule-authoring request corpus. "
            "Requests are not verifier evidence and external request rows do not "
            "copy labels or model answers."
        ),
        "source": {
            "plan_path": str(plan_path),
            "plan_workflow": plan.get("workflow"),
            "plan_status": plan.get("status"),
            "plan_target_count": _nested_int(plan, "summary", "target_count"),
        },
        "label_usage": {
            "labels_used_for_collection_requests": False,
            "labels_copied_to_collection_requests": False,
            "model_answers_copied_to_collection_requests": False,
            "requests_are_verifier_evidence": False,
        },
        "config": {
            "priorities": selected_priorities,
            "task_types": selected_task_types,
            "max_targets": max_targets,
            "max_fact_requests_per_target": int(max_fact_requests_per_target),
            "max_entity_requests_per_target": int(max_entity_requests_per_target),
            "max_citation_requests_per_target": int(max_citation_requests_per_target),
            "max_rule_requests_per_target": int(max_rule_requests_per_target),
        },
        "paths": paths,
        "summary": summary,
        "targets": tuple(targets),
        "requests": {key: tuple(value) for key, value in requests.items()},
        "source_discovery_documents": tuple(source_discovery_documents),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    plan_path: str | Path,
    output_dir: str | Path,
    report_json_path: str | Path | None = None,
    priorities: Sequence[str] = DEFAULT_PRIORITIES,
    task_types: Sequence[str] = (),
    max_targets: int | None = None,
    max_fact_requests_per_target: int = DEFAULT_MAX_FACT_REQUESTS_PER_TARGET,
    max_entity_requests_per_target: int = DEFAULT_MAX_ENTITY_REQUESTS_PER_TARGET,
    max_citation_requests_per_target: int = DEFAULT_MAX_CITATION_REQUESTS_PER_TARGET,
    max_rule_requests_per_target: int = DEFAULT_MAX_RULE_REQUESTS_PER_TARGET,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, optionally manifest, and optionally register a corpus."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path) if report_json_path is not None else output / "fact-collection-corpus.json"
    payload = build_source_family_structured_qa_fact_collection_corpus(
        plan_path=plan_path,
        output_dir=output,
        priorities=priorities,
        task_types=task_types,
        max_targets=max_targets,
        max_fact_requests_per_target=max_fact_requests_per_target,
        max_entity_requests_per_target=max_entity_requests_per_target,
        max_citation_requests_per_target=max_citation_requests_per_target,
        max_rule_requests_per_target=max_rule_requests_per_target,
        metadata=metadata,
    )
    _write_json(report_path, payload, compact=compact_json)
    payload["paths"]["report"] = str(report_path)
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            {
                "source_family_structured_qa_fact_collection_corpus": report_path,
                "source_family_structured_qa_fact_expansion_plan": plan_path,
                **{
                    key: path
                    for key, path in payload["paths"].items()
                    if key != "report" and str(path).endswith(".jsonl")
                },
            },
            root=manifest_path.parent,
            metadata={
                "workflow": payload["workflow"],
                "status": payload["status"],
                "target_count": payload["summary"]["target_count"],
                "total_request_count": payload["summary"]["total_request_count"],
                "structured_fact_request_count": payload["summary"]["request_counts"][
                    "source_family_structured_fact"
                ],
                "citation_request_count": payload["summary"]["request_counts"]["external_citation"],
                "rule_request_count": payload["summary"]["request_counts"]["world_model_or_calculator_rule"],
                **dict(metadata or {}),
            },
        )
        _write_json(manifest_path, manifest, compact=compact_json)
        payload["paths"]["artifact_manifest"] = str(manifest_path)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_path,
            metadata={
                "workflow": payload["workflow"],
                "status": payload["status"],
                "target_count": payload["summary"]["target_count"],
                "total_request_count": payload["summary"]["total_request_count"],
                "structured_fact_request_count": payload["summary"]["request_counts"][
                    "source_family_structured_fact"
                ],
                "citation_request_count": payload["summary"]["request_counts"]["external_citation"],
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _select_targets(
    raw_targets: Any,
    *,
    priorities: Sequence[str],
    task_types: Sequence[str],
    max_targets: int | None,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes, bytearray)):
        raise ValueError("fact expansion plan must include a targets array.")
    task_type_set = set(task_types)
    selected: list[Mapping[str, Any]] = []
    for item in raw_targets:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("priority", "")).strip() not in priorities:
            continue
        if task_type_set and not task_type_set.intersection(_target_task_types(item)):
            continue
        selected.append(item)
        if max_targets is not None and len(selected) >= int(max_targets):
            break
    return tuple(selected)


def _target_snapshot(target: Mapping[str, Any], *, target_id: str, ordinal: int) -> dict[str, Any]:
    answer = str(target.get("answer") or "")
    return {
        "target_id": target_id,
        "ordinal": int(ordinal),
        "record_index": _optional_int(target.get("record_index")),
        "priority": str(target.get("priority", "")),
        "question_type": str(target.get("question_type", "")),
        "mapping_decision": str(target.get("mapping_decision", "")),
        "gap_type": str(target.get("gap_type", "")),
        "question": str(target.get("question", "")),
        "answer_sha256": _sha256(answer) if answer else None,
        "entity_candidates": tuple(str(item) for item in _sequence(target.get("entity_candidates"))),
        "wikidata_property_hints": tuple(str(item) for item in _sequence(target.get("wikidata_property_hints"))),
        "source_family_targets": tuple(
            _source_family_target(item)
            for item in _sequence(target.get("source_family_targets"))
        ),
        "collection_task_types": _target_task_types(target),
    }


def _structured_fact_requests(
    target: Mapping[str, Any],
    *,
    target_id: str,
    max_items: int,
) -> tuple[dict[str, Any], ...]:
    if not _has_task(target, "source_family_structured_fact_request"):
        return ()
    requests: list[dict[str, Any]] = []
    source_families = tuple(_source_family_target(item) for item in _sequence(target.get("source_family_targets")))
    if not source_families:
        source_families = ({"provider": "source_family_adapter", "source_family": "reference", "reason": "fallback"},)
    for source in source_families:
        requests.append(_base_request(
            target,
            request_id=f"sfact:{target_id}:{len(requests) + 1}",
            target_id=target_id,
            request_type="source_family_structured_fact",
            extra={
                "provider_hint": source["provider"],
                "source_family": source["source_family"],
                "source_family_reason": source["reason"],
                "entity_candidates": _bounded_strings(target.get("entity_candidates"), 8),
                "property_hints": _bounded_strings(target.get("wikidata_property_hints"), 12),
                "property_ids": _property_ids(target.get("wikidata_property_hints")),
                "query": _structured_fact_query(target, source_family=str(source["source_family"])),
            },
        ))
        if len(requests) >= max_items:
            break
    return tuple(requests)


def _entity_resolution_requests(
    target: Mapping[str, Any],
    *,
    target_id: str,
    max_items: int,
) -> tuple[dict[str, Any], ...]:
    if not _has_task(target, "entity_resolution_request"):
        return ()
    requests = []
    for entity in _bounded_strings(target.get("entity_candidates"), max_items):
        requests.append(_base_request(
            target,
            request_id=f"entity:{target_id}:{len(requests) + 1}",
            target_id=target_id,
            request_type="entity_resolution",
            extra={
                "entity": entity,
                "query": _join_nonempty((entity, str(target.get("question", "")))),
                "property_hints": _bounded_strings(target.get("wikidata_property_hints"), 8),
                "provider_hints": ("wikidata", "source_family_adapter"),
            },
        ))
    return tuple(requests)


def _citation_requests(
    target: Mapping[str, Any],
    *,
    target_id: str,
    max_items: int,
) -> tuple[dict[str, Any], ...]:
    if not _has_task(target, "external_citation_request"):
        return ()
    requests = []
    for query in _safe_queries(target, max_items=max_items):
        requests.append(_base_request(
            target,
            request_id=f"cite:{target_id}:{len(requests) + 1}",
            target_id=target_id,
            request_type="external_citation",
            extra={
                "query": query,
                "source_family_hints": tuple(
                    item["source_family"]
                    for item in (_source_family_target(raw) for raw in _sequence(target.get("source_family_targets")))
                ),
                "requires_timestamp": str(target.get("question_type")) == "temporal",
            },
        ))
    return tuple(requests)


def _disambiguation_requests(target: Mapping[str, Any], *, target_id: str) -> tuple[dict[str, Any], ...]:
    if not _has_task(target, "source_family_fact_disambiguation"):
        return ()
    return (_base_request(
        target,
        request_id=f"disambig:{target_id}:1",
        target_id=target_id,
        request_type="source_family_fact_disambiguation",
        extra={
            "entities": _bounded_strings(target.get("entity_candidates"), 8),
            "property_hints": _bounded_strings(target.get("wikidata_property_hints"), 12),
            "nearest_fact_candidates": tuple(
                _candidate_summary(item)
                for item in _sequence(target.get("nearest_fact_candidates"))
                if isinstance(item, Mapping)
            )[:5],
            "query": _join_nonempty((str(target.get("question", "")), str(target.get("gap_type", "")))),
        },
    ),)


def _rule_requests(
    target: Mapping[str, Any],
    *,
    target_id: str,
    max_items: int,
) -> tuple[dict[str, Any], ...]:
    if not _has_task(target, "world_model_or_calculator_rule_request"):
        return ()
    requests = []
    for rule in _sequence(target.get("world_model_rule_targets"))[:max_items]:
        if not isinstance(rule, Mapping):
            continue
        family = str(rule.get("rule_family") or "world_model_consistency")
        requests.append(_base_request(
            target,
            request_id=f"rule:{target_id}:{len(requests) + 1}",
            target_id=target_id,
            request_type="world_model_or_calculator_rule",
            extra={
                "rule_family": family,
                "rule_reason": str(rule.get("reason") or ""),
                "rule_seed": _rule_seed(target, family=family),
                "required_inputs": _rule_inputs(family),
            },
        ))
    return tuple(requests)


def _base_request(
    target: Mapping[str, Any],
    *,
    request_id: str,
    target_id: str,
    request_type: str,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    request = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "request_id": request_id,
        "target_id": target_id,
        "request_type": request_type,
        "priority": str(target.get("priority", "")),
        "question_type": str(target.get("question_type", "")),
        "mapping_decision": str(target.get("mapping_decision", "")),
        "gap_type": str(target.get("gap_type", "")),
        "question": str(target.get("question", "")),
        "usage": "source_discovery_only",
        "not_verifier_evidence": True,
        **dict(extra),
    }
    forbidden = RESERVED_REQUEST_FIELDS.intersection(request)
    if forbidden:
        raise ValueError(f"request {request_id} copied reserved fields: {sorted(forbidden)}")
    return request


def _source_discovery_documents(requests: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    documents = []
    for request in requests:
        query = str(request.get("query") or request.get("entity") or request.get("question") or "")
        if not query:
            continue
        documents.append({
            "text": query,
            "source": f"collection-request:{request['request_id']}",
            "metadata": {
                "collection_request": True,
                "usage": request.get("usage"),
                "request_id": request["request_id"],
                "target_id": request["target_id"],
                "request_type": request["request_type"],
                "source_family": request.get("source_family"),
                "provider_hint": request.get("provider_hint"),
            },
        })
    return tuple(documents)


def _write_sidecars(
    output_dir: Path,
    *,
    requests: Mapping[str, Sequence[Mapping[str, Any]]],
    source_discovery_documents: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for bucket, filename in REQUEST_BUCKETS.items():
        path = output_dir / filename
        _write_jsonl(path, requests.get(bucket, ()))
        paths[bucket] = str(path)
    source_path = output_dir / "source-discovery-documents.jsonl"
    _write_jsonl(source_path, source_discovery_documents)
    paths["source_discovery_documents"] = str(source_path)
    return paths


def _summary(
    *,
    targets: Sequence[Mapping[str, Any]],
    requests: Mapping[str, Sequence[Mapping[str, Any]]],
    source_discovery_documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    request_counts = {bucket: len(tuple(items)) for bucket, items in requests.items()}
    priority_counts = Counter(str(target.get("priority") or "unknown") for target in targets)
    question_type_counts = Counter(str(target.get("question_type") or "unknown") for target in targets)
    gap_type_counts = Counter(str(target.get("gap_type") or "unknown") for target in targets)
    request_priority_counts = Counter(
        str(request.get("priority") or "unknown")
        for bucket in requests.values()
        for request in bucket
    )
    request_type_counts = Counter(
        str(request.get("request_type") or "unknown")
        for bucket in requests.values()
        for request in bucket
    )
    source_family_counts = Counter(
        str(request.get("source_family"))
        for request in requests.get("source_family_structured_fact", ())
        if request.get("source_family")
    )
    provider_counts = Counter(
        str(request.get("provider_hint"))
        for request in requests.get("source_family_structured_fact", ())
        if request.get("provider_hint")
    )
    return {
        "target_count": len(targets),
        "request_counts": request_counts,
        "total_request_count": sum(request_counts.values()),
        "source_discovery_document_count": len(source_discovery_documents),
        "priority_counts": _sorted_counter(priority_counts),
        "question_type_counts": _sorted_counter(question_type_counts),
        "gap_type_counts": _sorted_counter(gap_type_counts),
        "request_priority_counts": _sorted_counter(request_priority_counts),
        "request_type_counts": _sorted_counter(request_type_counts),
        "source_family_request_counts": _sorted_counter(source_family_counts),
        "provider_request_counts": _sorted_counter(provider_counts),
        "targets_with_rule_requests": len({
            str(request.get("target_id"))
            for request in requests.get("world_model_or_calculator_rule", ())
        }),
        "targets_with_citation_requests": len({
            str(request.get("target_id"))
            for request in requests.get("external_citation", ())
        }),
    }


def _load_plan(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("fact expansion plan must be a JSON object.")
    if payload.get("workflow") != PLAN_WORKFLOW:
        raise ValueError(f"{path} is not a {PLAN_WORKFLOW} report.")
    return dict(payload)


def _normalize_priorities(values: Sequence[str]) -> tuple[str, ...]:
    priorities = tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if not priorities:
        raise ValueError("at least one priority is required.")
    invalid = sorted(set(priorities) - set(DEFAULT_PRIORITIES))
    if invalid:
        raise ValueError(f"unsupported priorities: {', '.join(invalid)}")
    return priorities


def _validate_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive.")


def _target_task_types(target: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(task.get("task_type"))
        for task in _sequence(target.get("collection_tasks"))
        if isinstance(task, Mapping) and str(task.get("task_type"))
    )


def _has_task(target: Mapping[str, Any], task_type: str) -> bool:
    return task_type in set(_target_task_types(target))


def _source_family_target(value: Any) -> dict[str, str]:
    mapping = value if isinstance(value, Mapping) else {}
    return {
        "provider": str(mapping.get("provider") or "source_family_adapter"),
        "source_family": str(mapping.get("source_family") or "reference"),
        "reason": str(mapping.get("reason") or "unspecified"),
    }


def _structured_fact_query(target: Mapping[str, Any], *, source_family: str) -> str:
    entities = _bounded_strings(target.get("entity_candidates"), 3)
    hints = tuple(_property_label(hint) for hint in _bounded_strings(target.get("wikidata_property_hints"), 4))
    return _join_nonempty((
        str(target.get("question", "")),
        " ".join(entities),
        " ".join(hints),
        source_family,
    ))


def _safe_queries(target: Mapping[str, Any], *, max_items: int) -> tuple[str, ...]:
    answer = str(target.get("answer") or "")
    forbidden = {answer.casefold()} if answer else set()
    forbidden.update(token.casefold() for token in TOKEN_RE.findall(answer) if token.casefold() not in STOPWORDS)
    seeds = [
        str(target.get("question") or ""),
        _join_nonempty(_bounded_strings(target.get("entity_candidates"), 3)),
        _join_nonempty((
            str(target.get("question") or ""),
            _join_nonempty(
                _property_label(hint)
                for hint in _bounded_strings(target.get("wikidata_property_hints"), 3)
            ),
        )),
    ]
    for seed in _sequence(target.get("query_seeds")):
        text = str(seed)
        if not _contains_forbidden_answer(text, forbidden):
            seeds.append(text)
    clean = tuple(item for item in dict.fromkeys(_clean_query(seed) for seed in seeds) if item)
    return clean[:max_items]


def _contains_forbidden_answer(query: str, forbidden: set[str]) -> bool:
    lowered = query.casefold()
    return any(term and term in lowered for term in forbidden)


def _property_ids(values: Any) -> tuple[str, ...]:
    ids = []
    for value in _sequence(values):
        match = PROPERTY_ID_RE.search(str(value))
        if match is not None:
            ids.append(match.group(1))
    return tuple(dict.fromkeys(ids))


def _property_label(value: Any) -> str:
    text = str(value).split(":", 1)[0]
    return text.replace("_", " ").strip()


def _candidate_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "provider": value.get("provider"),
        "source_family": value.get("source_family"),
        "fact_type": value.get("fact_type"),
        "subject": value.get("subject"),
        "mapping_score": value.get("mapping_score"),
        "subject_coverage": value.get("subject_coverage"),
        "intent_score": value.get("intent_score"),
    }


def _rule_seed(target: Mapping[str, Any], *, family: str) -> str:
    question = str(target.get("question", ""))
    if family == "quantity_or_arithmetic":
        return f"Author a deterministic numeric check for: {question}"
    if family == "temporal_consistency":
        return f"Author a point-in-time consistency rule for: {question}"
    if family == "causal_or_procedural":
        return f"Author a causal/procedural source-backed rule for: {question}"
    if family == "entity_disambiguation":
        return f"Author an entity-role disambiguation rule for: {question}"
    return f"Author a deterministic world-model rule for: {question}"


def _rule_inputs(family: str) -> tuple[str, ...]:
    if family == "quantity_or_arithmetic":
        return ("numeric_value", "unit", "reference_time")
    if family == "temporal_consistency":
        return ("claim_time", "source_time", "retrieved_at")
    if family == "causal_or_procedural":
        return ("mechanism", "precondition", "source_citation")
    if family == "entity_disambiguation":
        return ("subject_entity", "answer_entity", "requested_role")
    return ("structured_state", "source_citation")


def _bounded_strings(value: Any, max_items: int) -> tuple[str, ...]:
    items = tuple(str(item).strip() for item in _sequence(value) if str(item).strip())
    return tuple(dict.fromkeys(items))[:max_items]


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


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _join_nonempty(values: Sequence[str]) -> str:
    return " ".join(str(value).strip() for value in values if str(value).strip())


def _clean_query(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).strip(" ?")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _target_id(target: Mapping[str, Any], ordinal: int) -> str:
    record_index = _optional_int(target.get("record_index"))
    if record_index is None or record_index < 0:
        return f"target-{ordinal}"
    return f"record-{record_index}"


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


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
    output.write_text(
        "".join(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parse_metadata(values: Sequence[str] | None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not values:
        return metadata
    for value in values:
        for part in str(value).split(","):
            item = part.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"metadata item {item!r} must use key=value format.")
            key, raw = item.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError("metadata key must be non-empty.")
            metadata[key] = raw.strip()
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--priority", action="append", choices=DEFAULT_PRIORITIES, default=None)
    parser.add_argument("--task-type", action="append", default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--max-fact-requests-per-target", type=int, default=DEFAULT_MAX_FACT_REQUESTS_PER_TARGET)
    parser.add_argument("--max-entity-requests-per-target", type=int, default=DEFAULT_MAX_ENTITY_REQUESTS_PER_TARGET)
    parser.add_argument("--max-citation-requests-per-target", type=int,
                        default=DEFAULT_MAX_CITATION_REQUESTS_PER_TARGET)
    parser.add_argument("--max-rule-requests-per-target", type=int, default=DEFAULT_MAX_RULE_REQUESTS_PER_TARGET)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=None)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        plan_path=args.plan,
        output_dir=args.output_dir,
        report_json_path=args.json,
        priorities=tuple(args.priority or DEFAULT_PRIORITIES),
        task_types=tuple(args.task_type or ()),
        max_targets=args.max_targets,
        max_fact_requests_per_target=args.max_fact_requests_per_target,
        max_entity_requests_per_target=args.max_entity_requests_per_target,
        max_citation_requests_per_target=args.max_citation_requests_per_target,
        max_rule_requests_per_target=args.max_rule_requests_per_target,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "source_family_structured_qa_fact_collection_corpus_ok "
        f"status={payload['status']} "
        f"targets={summary['target_count']} "
        f"requests={summary['total_request_count']} "
        f"facts={summary['request_counts']['source_family_structured_fact']} "
        f"citations={summary['request_counts']['external_citation']} "
        f"rules={summary['request_counts']['world_model_or_calculator_rule']}"
    )


if __name__ == "__main__":
    main()
