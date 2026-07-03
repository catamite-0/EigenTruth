"""Plan source-family structured QA fact expansion from claim-mapping gaps.

This workflow starts where ``audit_source_family_structured_qa_claim_mapping.py``
stops. A promoted source-family QA route can be useful only for claims that map
into covered facts; this planner converts unmapped or ambiguous claim rows into
concrete source-family fact, citation, entity-resolution, and world-model rule
collection tasks. It does not fetch sources, judge truth, or turn weak overlap
into verifier evidence.
"""

from __future__ import annotations

import argparse
import json
import re
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

WORKFLOW = "source_family_structured_qa_fact_expansion_plan"
DEFAULT_MAX_ENTITY_CANDIDATES = 6
DEFAULT_MAX_QUERY_SEEDS = 5
RESOLVED_DECISIONS = {
    "mapped_qa_fact_candidate",
    "answer_value_supported_by_covered_fact",
}
GAP_TYPES = {
    "no_candidate_fact": "missing_subject_and_intent",
    "subject_only_or_missing_intent": "missing_property_or_indicator",
    "intent_only_or_missing_subject": "missing_subject_entity_resolution",
    "weak_textual_overlap": "needs_citation_before_fact_promotion",
    "answer_entity_collision": "answer_entity_collision",
    "covered_fact_match_without_correction": "covered_fact_without_safe_correction",
    "mapped_qa_fact_candidate": "mapped_covered_fact_candidate",
    "answer_value_supported_by_covered_fact": "answer_already_supported",
}
PROPERTY_HINTS = {
    "definition": ("description", "instance_of:P31", "subclass_of:P279", "official_website:P856"),
    "person": ("founded_by:P112", "creator:P170", "author:P50", "occupation:P106", "country:P27"),
    "choice": ("disambiguation", "instance_of:P31", "subclass_of:P279", "country:P17"),
    "location": ("country:P17", "located_in:P131", "headquarters:P159", "location:P276"),
    "quantity": ("population:P1082", "area:P2046", "height:P2048", "mass:P2067", "point_in_time:P585"),
    "temporal": ("inception:P571", "publication_date:P577", "point_in_time:P585", "start_time:P580"),
    "method": ("procedure_citation", "has_part:P527", "use:P366"),
    "causal": ("causal_citation", "main_subject:P921", "significant_event:P793"),
    "other": ("description", "instance_of:P31", "external_citation"),
}
SOURCE_FAMILY_HINTS = {
    "definition": ("reference", "official_site", "scholarly"),
    "person": ("reference", "official_site", "scholarly"),
    "choice": ("reference", "official_site", "scholarly"),
    "location": ("reference", "official_statistics", "official_site"),
    "quantity": ("official_statistics", "reference", "official_site"),
    "temporal": ("reference", "official_site", "news_or_archive"),
    "method": ("scholarly", "official_site", "reference"),
    "causal": ("scholarly", "official_site", "reference"),
    "other": ("reference", "official_site", "scholarly"),
}
STOPWORDS = {
    "a",
    "about",
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
    "list",
    "listed",
    "lists",
    "many",
    "much",
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
    "will",
    "with",
}
CAPITALIZED_SPAN_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.'-]*|\d+(?:\.\d+)?")


def build_source_family_structured_qa_fact_expansion_plan(
    *,
    claim_mapping_path: str | Path,
    max_entity_candidates: int = DEFAULT_MAX_ENTITY_CANDIDATES,
    max_query_seeds: int = DEFAULT_MAX_QUERY_SEEDS,
    include_resolved: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready fact/rule expansion plan for claim-mapping gaps."""
    if int(max_entity_candidates) <= 0:
        raise ValueError("max_entity_candidates must be positive.")
    if int(max_query_seeds) <= 0:
        raise ValueError("max_query_seeds must be positive.")
    path = Path(claim_mapping_path)
    mapping = _load_json_object(path)
    records = _records(mapping)
    unresolved_records = tuple(
        record
        for record in records
        if bool(include_resolved) or str(record.get("mapping_decision")) not in RESOLVED_DECISIONS
    )
    targets = tuple(
        _target(
            record,
            ordinal=ordinal,
            max_entity_candidates=int(max_entity_candidates),
            max_query_seeds=int(max_query_seeds),
        )
        for ordinal, record in enumerate(unresolved_records, start=1)
    )
    summary = _summary(targets=targets, input_records=records)
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready_for_collection" if summary["target_count"] else "empty",
        "scope": (
            "Plans source-family structured fact, citation, entity-resolution, "
            "and world-model rule collection for claim-mapping gaps. The plan is "
            "not verifier evidence and does not use labels to decide what to collect."
        ),
        "source": {
            "claim_mapping_path": str(path),
            "claim_mapping_workflow": mapping.get("workflow"),
            "claim_mapping_status": mapping.get("status"),
            "claim_mapping_target_count": _nested_int(mapping, "summary", "target_count"),
            "claim_mapping_covered_fact_match_count": _nested_int(
                mapping,
                "summary",
                "covered_fact_match_count",
            ),
            "route_summary_status": _nested(mapping, "source", "route_summary_status"),
            "route_summary_promoted": _nested(mapping, "source", "route_summary_promoted"),
            "qa_document_count": _nested_int(mapping, "source", "qa_document_count"),
        },
        "label_usage": {
            "labels_used_for_collection_planning": False,
            "labels_copied_to_collection_tasks": False,
            "tasks_are_verifier_evidence": False,
        },
        "config": {
            "max_entity_candidates": int(max_entity_candidates),
            "max_query_seeds": int(max_query_seeds),
            "include_resolved": bool(include_resolved),
        },
        "summary": summary,
        "collection_plan": _collection_plan(targets),
        "targets": targets,
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    claim_mapping_path: str | Path,
    output_path: str | Path,
    max_entity_candidates: int = DEFAULT_MAX_ENTITY_CANDIDATES,
    max_query_seeds: int = DEFAULT_MAX_QUERY_SEEDS,
    include_resolved: bool = False,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, optionally manifest, and optionally register a plan."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    payload = build_source_family_structured_qa_fact_expansion_plan(
        claim_mapping_path=claim_mapping_path,
        max_entity_candidates=max_entity_candidates,
        max_query_seeds=max_query_seeds,
        include_resolved=include_resolved,
        metadata=metadata,
    )
    output = Path(output_path)
    if artifact_manifest_path is not None:
        payload["paths"] = {"artifact_manifest": str(artifact_manifest_path)}
    _write_json(output, payload, compact=compact_json)
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            {
                "source_family_structured_qa_fact_expansion_plan": output,
                "source_family_structured_qa_claim_mapping": claim_mapping_path,
            },
            root=manifest_path.parent,
            metadata={
                "workflow": payload["workflow"],
                "status": payload["status"],
                "target_count": payload["summary"]["target_count"],
                "high_priority_count": payload["summary"]["priority_counts"].get("high", 0),
                "top_gap_type": _first_key(payload["summary"]["gap_type_counts"]),
                "top_task_type": _first_key(payload["summary"]["task_type_counts"]),
                **dict(metadata or {}),
            },
        )
        _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output,
            metadata={
                "workflow": payload["workflow"],
                "status": payload["status"],
                "target_count": payload["summary"]["target_count"],
                "high_priority_count": payload["summary"]["priority_counts"].get("high", 0),
                "top_gap_type": _first_key(payload["summary"]["gap_type_counts"]),
                "top_task_type": _first_key(payload["summary"]["task_type_counts"]),
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _target(
    record: Mapping[str, Any],
    *,
    ordinal: int,
    max_entity_candidates: int,
    max_query_seeds: int,
) -> dict[str, Any]:
    decision = str(record.get("mapping_decision") or "unknown")
    question = str(record.get("question") or "")
    answer = str(record.get("answer") or "")
    text = str(record.get("text") or f"{question} {answer}").strip()
    question_type = str(record.get("question_type") or _question_type(question))
    candidates = _nearest_fact_candidates(record)
    entities = _entity_candidates(
        question=question,
        answer=answer,
        text=text,
        candidates=candidates,
        max_items=max_entity_candidates,
    )
    property_hints = _property_hints(question_type=question_type, candidates=candidates)
    source_family_targets = _source_family_targets(question_type=question_type, candidates=candidates)
    query_seeds = _query_seeds(
        question=question,
        answer=answer,
        text=text,
        entities=entities,
        question_type=question_type,
        max_items=max_query_seeds,
    )
    gap_type = GAP_TYPES.get(decision, "unclassified_mapping_gap")
    priority = _priority(decision=decision, question_type=question_type)
    rule_targets = _world_model_rule_targets(
        decision=decision,
        question_type=question_type,
        question=question,
        answer=answer,
    )
    tasks = _collection_tasks(
        decision=decision,
        gap_type=gap_type,
        priority=priority,
        question_type=question_type,
        entities=entities,
        property_hints=property_hints,
        source_family_targets=source_family_targets,
        query_seeds=query_seeds,
        rule_targets=rule_targets,
    )
    return {
        "target_id": _target_id(record, ordinal),
        "ordinal": int(ordinal),
        "record_id": str(record.get("record_id") or _target_id(record, ordinal)),
        "record_index": _optional_int(record.get("record_index")),
        "claim_id": record.get("claim_id"),
        "question": question,
        "answer": answer,
        "text": text,
        "question_type": question_type,
        "mapping_decision": decision,
        "gap_type": gap_type,
        "priority": priority,
        "gate_recommendation": record.get("gate_recommendation"),
        "best_mapping_score": _optional_float(record.get("best_mapping_score")),
        "best_subject_coverage": _optional_float(record.get("best_subject_coverage")),
        "best_intent_score": _optional_float(record.get("best_intent_score")),
        "source_family_targets": source_family_targets,
        "wikidata_property_hints": property_hints,
        "entity_candidates": entities,
        "query_seeds": query_seeds,
        "nearest_fact_candidates": candidates[:5],
        "world_model_rule_targets": rule_targets,
        "collection_tasks": tasks,
    }


def _collection_tasks(
    *,
    decision: str,
    gap_type: str,
    priority: str,
    question_type: str,
    entities: Sequence[str],
    property_hints: Sequence[str],
    source_family_targets: Sequence[Mapping[str, Any]],
    query_seeds: Sequence[str],
    rule_targets: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    tasks: list[dict[str, Any]] = []
    if decision in {"no_candidate_fact", "intent_only_or_missing_subject", "answer_entity_collision"}:
        tasks.append({
            "task_type": "entity_resolution_request",
            "priority": priority,
            "entities": tuple(entities),
            "reason": "resolve the claim subject before creating a covered structured QA fact",
        })
    if decision in {
        "no_candidate_fact",
        "subject_only_or_missing_intent",
        "intent_only_or_missing_subject",
        "weak_textual_overlap",
        "answer_entity_collision",
        "covered_fact_match_without_correction",
    }:
        tasks.append({
            "task_type": "source_family_structured_fact_request",
            "priority": priority,
            "gap_type": gap_type,
            "question_type": question_type,
            "source_family_targets": tuple(source_family_targets),
            "property_hints": tuple(property_hints),
            "reason": "collect a provenance-backed fact that can become a structured QA covered fact",
        })
    if decision in {
        "no_candidate_fact",
        "weak_textual_overlap",
        "answer_entity_collision",
        "covered_fact_match_without_correction",
    } or question_type in {"method", "causal"}:
        tasks.append({
            "task_type": "external_citation_request",
            "priority": "high" if decision == "weak_textual_overlap" else "medium",
            "query_seeds": tuple(query_seeds),
            "reason": "require source text before promoting weak or missing coverage into a correction route",
        })
    if decision in {"subject_only_or_missing_intent", "answer_entity_collision"}:
        tasks.append({
            "task_type": "source_family_fact_disambiguation",
            "priority": priority,
            "entities": tuple(entities),
            "property_hints": tuple(property_hints),
            "reason": "separate subject overlap, answer-entity collisions, and missing property intent",
        })
    if rule_targets:
        tasks.append({
            "task_type": "world_model_or_calculator_rule_request",
            "priority": "medium" if question_type in {"quantity", "temporal"} else "low",
            "rule_targets": tuple(rule_targets),
            "reason": "route non-lookup parts through explicit rules before verifier handoff",
        })
    return tuple(tasks)


def _summary(*, targets: Sequence[Mapping[str, Any]], input_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    input_decisions = Counter(str(record.get("mapping_decision") or "unknown") for record in input_records)
    decisions: Counter[str] = Counter()
    gaps: Counter[str] = Counter()
    priorities: Counter[str] = Counter()
    question_types: Counter[str] = Counter()
    task_types: Counter[str] = Counter()
    source_families: Counter[str] = Counter()
    providers: Counter[str] = Counter()
    no_entity_count = 0
    rule_target_count = 0
    for target in targets:
        decisions[str(target.get("mapping_decision"))] += 1
        gaps[str(target.get("gap_type"))] += 1
        priorities[str(target.get("priority"))] += 1
        question_types[str(target.get("question_type"))] += 1
        entities = _sequence(target.get("entity_candidates"))
        if not entities:
            no_entity_count += 1
        rule_target_count += len(_sequence(target.get("world_model_rule_targets")))
        for family in _sequence(target.get("source_family_targets")):
            if not isinstance(family, Mapping):
                continue
            source_families[str(family.get("source_family") or "unknown")] += 1
            providers[str(family.get("provider") or "source_family_adapter")] += 1
        for task in _sequence(target.get("collection_tasks")):
            if isinstance(task, Mapping):
                task_types[str(task.get("task_type"))] += 1
    return {
        "input_record_count": len(input_records),
        "target_count": len(targets),
        "skipped_resolved_count": sum(input_decisions.get(decision, 0) for decision in RESOLVED_DECISIONS),
        "input_mapping_decision_counts": _sorted_counter(input_decisions),
        "mapping_decision_counts": _sorted_counter(decisions),
        "gap_type_counts": _sorted_counter(gaps),
        "priority_counts": _sorted_counter(priorities),
        "question_type_counts": _sorted_counter(question_types),
        "task_type_counts": _sorted_counter(task_types),
        "source_family_counts": _sorted_counter(source_families),
        "provider_counts": _sorted_counter(providers),
        "targets_without_entity_candidates": no_entity_count,
        "world_model_rule_target_count": rule_target_count,
    }


def _collection_plan(targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        for task in _sequence(target.get("collection_tasks")):
            if not isinstance(task, Mapping):
                continue
            task_type = str(task.get("task_type"))
            by_task[task_type].append({
                "target_id": target.get("target_id"),
                "record_index": target.get("record_index"),
                "mapping_decision": target.get("mapping_decision"),
                "gap_type": target.get("gap_type"),
                "question_type": target.get("question_type"),
                "priority": task.get("priority"),
                "entities": tuple(target.get("entity_candidates", ())),
                "property_hints": tuple(target.get("wikidata_property_hints", ())),
                "query_seeds": tuple(target.get("query_seeds", ())),
            })
    return {
        name: {
            "count": len(items),
            "high_priority_count": sum(1 for item in items if item.get("priority") == "high"),
            "items": tuple(items),
        }
        for name, items in sorted(by_task.items())
    }


def _nearest_fact_candidates(record: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    candidates: list[dict[str, Any]] = []
    for key in ("mapped_facts", "collision_facts", "supported_facts", "top_fact_candidates"):
        for item in _sequence(record.get(key)):
            if not isinstance(item, Mapping):
                continue
            candidates.append({
                "source": item.get("source"),
                "provider": item.get("provider"),
                "source_family": item.get("source_family"),
                "fact_type": item.get("fact_type"),
                "subject": item.get("subject"),
                "intent_terms": tuple(str(term) for term in _sequence(item.get("intent_terms"))),
                "mapping_score": _optional_float(item.get("mapping_score")),
                "subject_coverage": _optional_float(item.get("subject_coverage")),
                "intent_score": _optional_float(item.get("intent_score")),
                "weak_textual_overlap": _optional_float(item.get("weak_textual_overlap")),
                "answer_subject_overlap": _optional_float(item.get("answer_subject_overlap")),
                "answer_value_overlap": _optional_float(item.get("answer_value_overlap")),
            })
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = "|".join(
            str(candidate.get(part) or "")
            for part in ("source", "provider", "source_family", "fact_type", "subject")
        )
        unique.setdefault(key, candidate)
    return tuple(unique.values())


def _property_hints(*, question_type: str, candidates: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    hints = list(PROPERTY_HINTS.get(question_type, PROPERTY_HINTS["other"]))
    for candidate in candidates:
        fact_type = str(candidate.get("fact_type") or "").strip()
        if fact_type and fact_type.casefold() != "unknown":
            hints.insert(0, fact_type)
        for term in _sequence(candidate.get("intent_terms")):
            value = str(term).strip()
            if value:
                hints.append(value)
    return tuple(dict.fromkeys(_clean_hint(hint) for hint in hints if str(hint).strip()))


def _source_family_targets(
    *,
    question_type: str,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    targets: list[dict[str, Any]] = []
    for candidate in candidates:
        provider = str(candidate.get("provider") or "").strip()
        family = str(candidate.get("source_family") or "").strip()
        if provider or family:
            targets.append({
                "provider": provider or "source_family_adapter",
                "source_family": family or "reference",
                "reason": "nearest_existing_fact_candidate",
            })
    for family in SOURCE_FAMILY_HINTS.get(question_type, SOURCE_FAMILY_HINTS["other"]):
        provider = "worldbank" if family == "official_statistics" else "source_family_adapter"
        targets.append({
            "provider": provider,
            "source_family": family,
            "reason": f"default_{question_type}_coverage",
        })
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for target in targets:
        key = (str(target["provider"]), str(target["source_family"]))
        unique.setdefault(key, target)
    return tuple(unique.values())


def _world_model_rule_targets(
    *,
    decision: str,
    question_type: str,
    question: str,
    answer: str,
) -> tuple[dict[str, Any], ...]:
    targets: list[dict[str, Any]] = []
    if question_type == "quantity" or _has_number(question) or _has_number(answer):
        targets.append({
            "rule_family": "quantity_or_arithmetic",
            "reason": "numeric claim may need unit, denominator, or arithmetic checks",
        })
    if question_type == "temporal" or _is_time_sensitive(question):
        targets.append({
            "rule_family": "temporal_consistency",
            "reason": "time-sensitive claim needs point-in-time grounding",
        })
    if question_type in {"method", "causal"}:
        targets.append({
            "rule_family": "causal_or_procedural",
            "reason": "claim needs mechanism/procedure checks beyond entity lookup",
        })
    if decision == "answer_entity_collision":
        targets.append({
            "rule_family": "entity_disambiguation",
            "reason": "model answer overlaps a subject entity rather than the requested value",
        })
    return tuple(targets)


def _entity_candidates(
    *,
    question: str,
    answer: str,
    text: str,
    candidates: Sequence[Mapping[str, Any]],
    max_items: int,
) -> tuple[str, ...]:
    values: list[str] = []
    for source in (question, answer, text):
        for match in CAPITALIZED_SPAN_RE.finditer(source):
            values.append(_clean_candidate(match.group(0)))
    answer_clean = _clean_candidate(answer)
    if _valid_candidate(answer_clean):
        values.append(answer_clean)
    for candidate in candidates:
        values.append(_clean_candidate(str(candidate.get("subject") or "")))
    keyword_phrase = _keyword_phrase(question)
    if _valid_candidate(keyword_phrase):
        values.append(keyword_phrase)
    return tuple(dict.fromkeys(value for value in values if _valid_candidate(value)))[:max_items]


def _query_seeds(
    *,
    question: str,
    answer: str,
    text: str,
    entities: Sequence[str],
    question_type: str,
    max_items: int,
) -> tuple[str, ...]:
    seeds = [
        f"{question} {answer}".strip(),
        question,
        text,
        f"{_keyword_phrase(question)} {answer}".strip(),
    ]
    for entity in entities:
        seeds.append(f"{entity} {answer}".strip())
        seeds.append(f"{question_type} {entity}".strip())
    return tuple(item for item in dict.fromkeys(_clean_query(seed) for seed in seeds) if item)[:max_items]


def _priority(*, decision: str, question_type: str) -> str:
    if decision in {"no_candidate_fact", "answer_entity_collision"}:
        return "high"
    if decision in {"subject_only_or_missing_intent", "intent_only_or_missing_subject"}:
        return "high"
    if decision == "weak_textual_overlap":
        return "medium"
    if question_type in {"quantity", "temporal", "method", "causal"}:
        return "medium"
    return "low"


def _target_id(record: Mapping[str, Any], ordinal: int) -> str:
    raw = record.get("record_index")
    if raw is not None:
        return f"record-{raw}"
    raw = record.get("record_id") or record.get("claim_id")
    if raw:
        return _slug(str(raw))
    return f"target-{ordinal}"


def _records(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    records = payload.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        raise ValueError("claim mapping report must include a records array.")
    result = tuple(record for record in records if isinstance(record, Mapping))
    if not result:
        raise ValueError("claim mapping report has no object records.")
    return result


def _question_type(question: str) -> str:
    lowered = question.casefold()
    if re.search(r"\b(?:when|what time|what year|what date|now|current|today|latest)\b", lowered):
        return "temporal"
    if re.search(r"\bwho\b", lowered):
        return "person"
    if re.search(r"\bwhere\b", lowered):
        return "location"
    if re.search(r"\b(?:how many|how much|percentage|percent|number|amount|count)\b", lowered):
        return "quantity"
    if re.search(r"\bwhy\b", lowered):
        return "causal"
    if re.search(r"\bhow\b", lowered):
        return "method"
    if re.search(r"\bwhich\b", lowered):
        return "choice"
    if re.search(r"\bwhat\b", lowered):
        return "definition"
    return "other"


def _keyword_phrase(question: str) -> str:
    tokens = [
        token
        for token in (match.group(0) for match in TOKEN_RE.finditer(question))
        if token.casefold() not in STOPWORDS
    ]
    return " ".join(tokens[:8])


def _has_number(value: str) -> bool:
    return bool(re.search(r"\d", value))


def _is_time_sensitive(value: str) -> bool:
    return bool(re.search(r"\b(?:now|current|today|latest|this year|as of|in \d{4})\b", value.casefold()))


def _clean_hint(value: Any) -> str:
    return re.sub(r"\s+", "_", str(value).strip()).casefold()


def _clean_candidate(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" \t\r\n?.!,;:\"'()[]{}")).strip()


def _clean_query(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).strip(" ?")


def _valid_candidate(value: str) -> bool:
    if not value:
        return False
    if value.casefold() in STOPWORDS:
        return False
    return bool(re.search(r"[A-Za-z0-9]", value))


def _load_json_object(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(data)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _nested_int(mapping: Mapping[str, Any], *keys: str) -> int | None:
    value = _nested(mapping, *keys)
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


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


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _first_key(mapping: Mapping[str, Any]) -> str | None:
    return next(iter(mapping), None)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-").casefold()
    return slug or "target"


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
    parser.add_argument("--claim-mapping", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--max-entity-candidates", type=int, default=DEFAULT_MAX_ENTITY_CANDIDATES)
    parser.add_argument("--max-query-seeds", type=int, default=DEFAULT_MAX_QUERY_SEEDS)
    parser.add_argument("--include-resolved", action="store_true")
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=None)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        claim_mapping_path=args.claim_mapping,
        output_path=args.json,
        max_entity_candidates=args.max_entity_candidates,
        max_query_seeds=args.max_query_seeds,
        include_resolved=bool(args.include_resolved),
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "source_family_structured_qa_fact_expansion_plan_ok "
        f"status={payload['status']} "
        f"targets={summary['target_count']} "
        f"top_gap={_first_key(summary['gap_type_counts'])} "
        f"top_task={_first_key(summary['task_type_counts'])}"
    )


if __name__ == "__main__":
    main()
