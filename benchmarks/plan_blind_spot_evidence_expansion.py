"""Plan external evidence expansion for detectability blind spots.

This no-model workflow consumes row-level blind-spot records and turns a blocked
coverage result into concrete collection targets. It does not fetch data or
promote a verifier route; it proposes which evidence families, query seeds, and
structured-fact properties should be collected next for each blind spot.
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

DEFAULT_MAX_ENTITY_CANDIDATES = 5
DEFAULT_MAX_QUERY_SEEDS = 4
STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "ever",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "his",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "many",
    "much",
    "no",
    "not",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "true",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "what's",
    "you",
    "your",
}
GENERIC_ENTITY_CANDIDATES = {
    "america",
    "american",
    "answer",
    "anything",
    "country",
    "everyone",
    "everything",
    "he",
    "her",
    "him",
    "his",
    "human",
    "it",
    "name",
    "nobody",
    "no one",
    "nothing",
    "one",
    "people",
    "person",
    "she",
    "someone",
    "something",
    "son",
    "that",
    "the answer",
    "the country",
    "the person",
    "the team",
    "these",
    "they",
    "this",
    "we",
    "you",
}
PROPERTY_HINTS = {
    "definition": ("description", "instance_of:P31", "subclass_of:P279", "official_website:P856"),
    "person": ("creator:P170", "author:P50", "founded_by:P112", "occupation:P106", "country:P27"),
    "choice": ("disambiguation", "instance_of:P31", "subclass_of:P279", "country:P17"),
    "location": ("country:P17", "located_in:P131", "headquarters:P159", "location:P276"),
    "quantity": ("population:P1082", "area:P2046", "height:P2048", "mass:P2067"),
    "temporal": ("inception:P571", "publication_date:P577", "point_in_time:P585", "start_time:P580"),
    "method": ("procedure_citation", "has_part:P527", "use:P366"),
    "causal": ("causal_citation", "main_subject:P921", "significant_event:P793"),
    "other": ("description", "instance_of:P31", "external_citation"),
}
ROUTE_HINTS = {
    "definition": ("structured_fact", "structured_qa", "retrieval_citation"),
    "person": ("structured_fact", "structured_qa", "counterfactual_entity_swap"),
    "choice": ("structured_qa", "structured_fact", "counterfactual_entity_swap"),
    "location": ("structured_fact", "structured_qa", "retrieval_citation"),
    "quantity": ("structured_fact", "calculator", "counterfactual_quantity"),
    "temporal": ("structured_fact", "time_sensitive_retrieval", "counterfactual_temporal"),
    "method": ("retrieval_citation", "world_model_rule"),
    "causal": ("retrieval_citation", "world_model_rule", "counterfactual_causal"),
    "other": ("retrieval_citation", "structured_fact"),
}
CAPITALIZED_SPAN_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.'-]*|\d+(?:\.\d+)?")


def build_blind_spot_evidence_expansion_plan(
    *,
    blind_spots_path: str | Path,
    provenance_comparison_path: str | Path | None = None,
    query_sweep_path: str | Path | None = None,
    max_entity_candidates: int = DEFAULT_MAX_ENTITY_CANDIDATES,
    max_query_seeds: int = DEFAULT_MAX_QUERY_SEEDS,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready evidence expansion plan for blind spots."""
    blind_path = Path(blind_spots_path)
    records = _load_blind_spot_records(blind_path)
    if int(max_entity_candidates) <= 0:
        raise ValueError("max_entity_candidates must be positive.")
    if int(max_query_seeds) <= 0:
        raise ValueError("max_query_seeds must be positive.")
    comparison = None if provenance_comparison_path is None else _load_json_object(provenance_comparison_path)
    query_sweep = None if query_sweep_path is None else _load_json_object(query_sweep_path)
    gap_guidance = None if query_sweep is None else _query_sweep_gap_guidance(query_sweep)
    targets = tuple(
        _record_target(
            record,
            max_entity_candidates=int(max_entity_candidates),
            max_query_seeds=int(max_query_seeds),
            query_sweep_gap_guidance=gap_guidance,
        )
        for record in records
    )
    summary = _summary(targets)
    if gap_guidance is not None:
        summary["query_sweep_guidance"] = gap_guidance
    status = _status(summary=summary, comparison=comparison)
    return {
        "schema_version": 1,
        "workflow": "blind_spot_evidence_expansion_plan",
        "status": status,
        "source": {
            "blind_spots_path": str(blind_path),
            "blind_spot_workflow": _load_json_object(blind_path).get("workflow"),
            "provenance_comparison_path": (
                None if provenance_comparison_path is None else str(provenance_comparison_path)
            ),
            "provenance_comparison_status": None if comparison is None else comparison.get("status"),
            "query_sweep_path": None if query_sweep_path is None else str(query_sweep_path),
            "query_sweep_workflow": None if query_sweep is None else query_sweep.get("workflow"),
            "query_sweep_status": None if query_sweep is None else query_sweep.get("status"),
        },
        "config": {
            "max_entity_candidates": int(max_entity_candidates),
            "max_query_seeds": int(max_query_seeds),
        },
        "summary": summary,
        "collection_plan": _collection_plan(targets),
        "targets": targets,
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    blind_spots_path: str | Path,
    output_path: str | Path,
    provenance_comparison_path: str | Path | None = None,
    query_sweep_path: str | Path | None = None,
    max_entity_candidates: int = DEFAULT_MAX_ENTITY_CANDIDATES,
    max_query_seeds: int = DEFAULT_MAX_QUERY_SEEDS,
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
    report = build_blind_spot_evidence_expansion_plan(
        blind_spots_path=blind_spots_path,
        provenance_comparison_path=provenance_comparison_path,
        query_sweep_path=query_sweep_path,
        max_entity_candidates=max_entity_candidates,
        max_query_seeds=max_query_seeds,
        metadata=metadata,
    )
    output = Path(output_path)
    if artifact_manifest_path is not None:
        report["paths"] = {"artifact_manifest": str(artifact_manifest_path)}
    _write_json(output, report, compact=compact_json)
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        artifacts: dict[str, str | Path | None] = {
            "blind_spot_evidence_expansion_plan": output,
            "blind_spots": blind_spots_path,
            "provenance_comparison": provenance_comparison_path,
            "query_sweep": query_sweep_path,
        }
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "runner": "plan_blind_spot_evidence_expansion",
                "status": report["status"],
                "target_count": report["summary"]["target_count"],
                "high_priority_count": report["summary"]["priority_counts"].get("high", 0),
                "top_question_type": _first_key(report["summary"]["question_type_counts"]),
                "top_route": _first_key(report["summary"]["recommended_route_counts"]),
                "query_sweep_best_strategy": _nested_value(
                    report,
                    "summary",
                    "query_sweep_guidance",
                    "best_strategy",
                ),
                "query_sweep_dominant_gap_bucket": _nested_value(
                    report,
                    "summary",
                    "query_sweep_guidance",
                    "dominant_gap_bucket",
                ),
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
                "workflow": report["workflow"],
                "status": report["status"],
                "target_count": report["summary"]["target_count"],
                "high_priority_count": report["summary"]["priority_counts"].get("high", 0),
                "top_question_type": _first_key(report["summary"]["question_type_counts"]),
                "top_route": _first_key(report["summary"]["recommended_route_counts"]),
                "query_sweep_best_strategy": _nested_value(
                    report,
                    "summary",
                    "query_sweep_guidance",
                    "best_strategy",
                ),
                "query_sweep_dominant_gap_bucket": _nested_value(
                    report,
                    "summary",
                    "query_sweep_guidance",
                    "dominant_gap_bucket",
                ),
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return report


def _record_target(
    record: Mapping[str, Any],
    *,
    max_entity_candidates: int,
    max_query_seeds: int,
    query_sweep_gap_guidance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    text = str(record.get("text") or f"{question} {answer}").strip()
    question_type = str(record.get("question_type") or _question_type(question))
    features = _mapping(record.get("features"))
    answer_features = _mapping(record.get("answer_features"))
    routes = _routes(question_type, features=features, answer_features=answer_features)
    routes = _routes_with_gap_guidance(routes, query_sweep_gap_guidance=query_sweep_gap_guidance)
    property_hints = _property_hints(question_type, features=features, answer_features=answer_features)
    entities = _entity_candidates(question, answer, max_items=max_entity_candidates)
    query_seeds = _query_seeds(
        question=question,
        answer=answer,
        entities=entities,
        question_type=question_type,
        max_items=max_query_seeds,
    )
    return {
        "record_index": int(record.get("record_index", -1)),
        "label": int(record.get("label", 1)),
        "question_type": question_type,
        "priority": _priority(question_type, features=features, answer_features=answer_features),
        "recommended_routes": routes,
        "wikidata_property_hints": property_hints,
        "entity_candidates": entities,
        "query_seeds": query_seeds,
        "collection_tasks": _collection_tasks(
            question_type=question_type,
            routes=routes,
            property_hints=property_hints,
            entities=entities,
            query_seeds=query_seeds,
            query_sweep_gap_guidance=query_sweep_gap_guidance,
        ),
        "features": {
            "claim": {key: bool(value) for key, value in features.items()},
            "answer": {key: bool(value) for key, value in answer_features.items()},
        },
        "question": question,
        "answer": answer,
        "text": text,
        "query_sweep_gap_guidance": query_sweep_gap_guidance,
    }


def _routes(
    question_type: str,
    *,
    features: Mapping[str, Any],
    answer_features: Mapping[str, Any],
) -> tuple[str, ...]:
    routes = list(ROUTE_HINTS.get(question_type, ROUTE_HINTS["other"]))
    if features.get("has_negation") or answer_features.get("has_negation"):
        routes.append("counterfactual_negation")
    if features.get("has_number") or answer_features.get("has_number"):
        routes.append("calculator")
        routes.append("counterfactual_quantity")
    if features.get("is_time_sensitive") or answer_features.get("is_time_sensitive"):
        routes.append("time_sensitive_retrieval")
    return tuple(dict.fromkeys(routes))


def _property_hints(
    question_type: str,
    *,
    features: Mapping[str, Any],
    answer_features: Mapping[str, Any],
) -> tuple[str, ...]:
    hints = list(PROPERTY_HINTS.get(question_type, PROPERTY_HINTS["other"]))
    if features.get("has_number") or answer_features.get("has_number"):
        hints.extend(("quantity_property", "point_in_time:P585"))
    if features.get("is_time_sensitive") or answer_features.get("is_time_sensitive"):
        hints.extend(("retrieved_at_required", "point_in_time:P585"))
    return tuple(dict.fromkeys(hints))


def _entity_candidates(question: str, answer: str, *, max_items: int) -> tuple[str, ...]:
    candidates: list[str] = []
    for source in (question, answer):
        for match in CAPITALIZED_SPAN_RE.finditer(source):
            span = _clean_candidate(match.group(0))
            if _valid_candidate(span):
                candidates.append(span)
    answer_clean = _clean_candidate(answer)
    if _valid_candidate(answer_clean):
        candidates.append(answer_clean)
    keyword_phrase = _keyword_phrase(question)
    if _valid_candidate(keyword_phrase):
        candidates.append(keyword_phrase)
    return tuple(dict.fromkeys(candidates))[:max_items]


def _query_seeds(
    *,
    question: str,
    answer: str,
    entities: Sequence[str],
    question_type: str,
    max_items: int,
) -> tuple[str, ...]:
    seeds = [
        f"{question} {answer}".strip(),
        question,
        f"{_keyword_phrase(question)} {answer}".strip(),
    ]
    answer_candidate = _clean_candidate(answer).casefold()
    for entity in entities:
        if entity.casefold() != answer_candidate:
            seeds.append(f"{entity} {answer}".strip())
        seeds.append(f"{question_type} {entity}".strip())
    return tuple(item for item in dict.fromkeys(_clean_query(seed) for seed in seeds) if item)[:max_items]


def _collection_tasks(
    *,
    question_type: str,
    routes: Sequence[str],
    property_hints: Sequence[str],
    entities: Sequence[str],
    query_seeds: Sequence[str],
    query_sweep_gap_guidance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    tasks = [
        {
            "task": "wikidata_entity_resolution",
            "priority": "high" if question_type in {"definition", "person", "choice", "location"} else "medium",
            "entities": tuple(entities),
            "property_hints": tuple(property_hints),
            "reason": "map blind-spot question terms to structured facts with source provenance",
        },
        {
            "task": "external_citation_retrieval",
            "priority": "medium",
            "query_seeds": tuple(query_seeds),
            "reason": "collect non-oracle evidence documents for retrieval/citation routes",
        },
    ]
    if any(route.startswith("counterfactual") for route in routes):
        tasks.append({
            "task": "counterfactual_probe_generation",
            "priority": "medium",
            "probe_types": tuple(
                route.replace("counterfactual_", "")
                for route in routes
                if route.startswith("counterfactual")
            ),
            "reason": "verify the correction route changes status under targeted perturbations",
        })
    if "world_model_rule" in routes or "calculator" in routes:
        tasks.append({
            "task": "world_model_or_calculator_rule",
            "priority": "low" if "calculator" not in routes else "medium",
            "reason": "route non-lookup claims through explicit rule/calculation checks",
        })
    if query_sweep_gap_guidance:
        actions = set(str(item) for item in _sequence(query_sweep_gap_guidance.get("recommended_alignment_actions")))
        if "claim_evidence_alignment" in actions:
            tasks.append({
                "task": "claim_evidence_alignment_audit",
                "priority": "high",
                "alignment_actions": tuple(sorted(actions)),
                "dominant_gap_bucket": query_sweep_gap_guidance.get("dominant_gap_bucket"),
                "reason": "align subject, property, value, and evidence spans before verifier rerun",
            })
        if "source_document_fact_extraction" in actions:
            tasks.append({
                "task": "source_document_fact_extraction",
                "priority": "medium",
                "top_hit_sources": tuple(query_sweep_gap_guidance.get("top_hit_sources", ())),
                "reason": "convert broad retrieved documents into source-backed structured facts",
            })
        if "query_refinement" in actions:
            tasks.append({
                "task": "query_refinement",
                "priority": "medium",
                "query_seeds": tuple(query_seeds),
                "reason": "increase claim-specific retrieval before another route-quality sweep",
            })
        if "negative_control_alignment_audit" in actions:
            tasks.append({
                "task": "negative_control_alignment_audit",
                "priority": "high",
                "reason": "inspect true-label support and false-positive behavior before promotion",
            })
    return tuple(tasks)


def _summary(targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    question_types: Counter[str] = Counter()
    priorities: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    properties: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    no_entity_count = 0
    for target in targets:
        question_types[str(target["question_type"])] += 1
        priorities[str(target["priority"])] += 1
        routes.update(str(item) for item in target.get("recommended_routes", ()))
        properties.update(str(item) for item in target.get("wikidata_property_hints", ()))
        tasks.update(
            str(item.get("task"))
            for item in _sequence(target.get("collection_tasks"))
            if isinstance(item, Mapping)
        )
        entities = tuple(str(item) for item in target.get("entity_candidates", ()))
        if not entities:
            no_entity_count += 1
        entity_counts.update(entities)
    return {
        "target_count": len(targets),
        "question_type_counts": _sorted_counter(question_types),
        "priority_counts": _sorted_counter(priorities),
        "recommended_route_counts": _sorted_counter(routes),
        "wikidata_property_hint_counts": _sorted_counter(properties),
        "collection_task_counts": _sorted_counter(tasks),
        "targets_without_entity_candidates": no_entity_count,
        "top_entity_candidates": dict(entity_counts.most_common(20)),
    }


def _collection_plan(targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        for task in _sequence(target.get("collection_tasks")):
            if not isinstance(task, Mapping):
                continue
            task_name = str(task.get("task"))
            by_task[task_name].append({
                "record_index": target.get("record_index"),
                "question_type": target.get("question_type"),
                "priority": task.get("priority"),
                "dominant_gap_bucket": task.get("dominant_gap_bucket"),
                "alignment_actions": tuple(task.get("alignment_actions", ())),
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


def _status(*, summary: Mapping[str, Any], comparison: Mapping[str, Any] | None) -> str:
    if comparison is not None and comparison.get("status") == "promote":
        return "ready_to_validate"
    if int(summary.get("target_count", 0)) > 0:
        return "needs_evidence_collection"
    return "complete"


def _priority(
    question_type: str,
    *,
    features: Mapping[str, Any],
    answer_features: Mapping[str, Any],
) -> str:
    if question_type in {"definition", "person", "choice", "location"}:
        return "high"
    if features.get("has_number") or answer_features.get("has_number") or question_type in {"quantity", "temporal"}:
        return "medium"
    return "medium" if question_type in {"method", "causal"} else "low"


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


def _clean_candidate(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" \t\r\n?.!,;:\"'()[]{}")).strip()


def _clean_query(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).strip(" ?")


def _valid_candidate(value: str) -> bool:
    if not value:
        return False
    normalized = value.casefold()
    if normalized in STOPWORDS or normalized in GENERIC_ENTITY_CANDIDATES:
        return False
    tokens = tuple(TOKEN_RE.findall(value))
    if len(tokens) == 1 and tokens[0].casefold() in GENERIC_ENTITY_CANDIDATES:
        return False
    return bool(re.search(r"[A-Za-z0-9]", value))


def _load_blind_spot_records(path: Path) -> tuple[dict[str, Any], ...]:
    payload = _load_json_object(path)
    records = payload.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        raise ValueError("blind spot report must include a records array.")
    result = tuple(dict(record) for record in records if isinstance(record, Mapping))
    if not result:
        raise ValueError("blind spot report has no object records.")
    return result


def _load_json_object(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(data)


def _query_sweep_gap_guidance(query_sweep: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(query_sweep.get("summary"))
    strategies = tuple(
        dict(item)
        for item in _sequence(query_sweep.get("strategies"))
        if isinstance(item, Mapping)
    )
    best_strategy = _optional_str(summary.get("best_passing_strategy")) or _optional_str(summary.get("best_strategy"))
    selected = _strategy_by_key(str(best_strategy or ""), strategies) if best_strategy else {}
    gap_analysis = _mapping(selected.get("gap_analysis"))
    gap_summary = _mapping(gap_analysis.get("summary"))
    gap_bucket_counts = _gap_bucket_counts(gap_analysis)
    dominant_gap_bucket = _first_key(gap_bucket_counts)
    false_negative_rate = _optional_float(gap_summary.get("false_negative_rate"))
    false_positive_count = _optional_int(gap_summary.get("false_positive_count")) or 0
    retrieval_hit_rate = _optional_float(gap_summary.get("records_with_retrieval_hit_rate"))
    top_hit_sources = tuple(
        str(item.get("value"))
        for item in _sequence(gap_analysis.get("top_hit_sources"))
        if isinstance(item, Mapping) and item.get("value")
    )[:5]
    actions = _alignment_actions(
        false_negative_rate=false_negative_rate,
        false_positive_count=false_positive_count,
        retrieval_hit_rate=retrieval_hit_rate,
        top_hit_sources=top_hit_sources,
    )
    return {
        "best_strategy": best_strategy,
        "best_passing_strategy": _optional_str(summary.get("best_passing_strategy")),
        "best_blind_refuted_count": _optional_int(summary.get("best_blind_refuted_count")),
        "best_passing_blind_refuted_count": _optional_int(summary.get("best_passing_blind_refuted_count")),
        "dominant_gap_bucket": dominant_gap_bucket,
        "gap_bucket_counts": gap_bucket_counts,
        "false_negative_count": _optional_int(gap_summary.get("false_negative_count")),
        "false_negative_rate": false_negative_rate,
        "false_positive_count": false_positive_count,
        "false_positive_rate": _optional_float(gap_summary.get("false_positive_rate")),
        "records_with_retrieval_hits": _optional_int(gap_summary.get("records_with_retrieval_hits")),
        "records_with_retrieval_hit_rate": retrieval_hit_rate,
        "records_using_retrieval": _optional_int(gap_summary.get("records_using_retrieval")),
        "records_using_retrieval_rate": _optional_float(gap_summary.get("records_using_retrieval_rate")),
        "top_hit_sources": top_hit_sources,
        "recommended_alignment_actions": actions,
    }


def _strategy_by_key(key: str, strategies: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for strategy in strategies:
        if str(strategy.get("key")) == key:
            return strategy
    return {}


def _gap_bucket_counts(gap_analysis: Mapping[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    buckets = _mapping(gap_analysis.get("gap_buckets"))
    for name, bucket in buckets.items():
        if isinstance(bucket, Mapping):
            counts[str(name)] = _optional_int(bucket.get("count")) or 0
    return _sorted_counter(counts)


def _alignment_actions(
    *,
    false_negative_rate: float | None,
    false_positive_count: int,
    retrieval_hit_rate: float | None,
    top_hit_sources: Sequence[str],
) -> tuple[str, ...]:
    actions: list[str] = []
    if false_negative_rate is not None and false_negative_rate >= 0.5:
        actions.append("claim_evidence_alignment")
    if retrieval_hit_rate is not None and retrieval_hit_rate < 0.25:
        actions.append("query_refinement")
    if top_hit_sources:
        actions.append("source_document_fact_extraction")
    if false_positive_count > 0:
        actions.append("negative_control_alignment_audit")
    if not actions:
        actions.append("route_quality_monitoring")
    return tuple(dict.fromkeys(actions))


def _routes_with_gap_guidance(
    routes: Sequence[str],
    *,
    query_sweep_gap_guidance: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    result = list(routes)
    if query_sweep_gap_guidance:
        result.extend(
            str(item)
            for item in _sequence(query_sweep_gap_guidance.get("recommended_alignment_actions"))
            if str(item) != "route_quality_monitoring"
        )
    return tuple(dict.fromkeys(result))


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _first_key(mapping: Mapping[str, Any]) -> str | None:
    return next(iter(mapping), None)


def _nested_value(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _optional_str(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


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
    parser.add_argument("--blind-spots", required=True)
    parser.add_argument("--provenance-comparison", default=None)
    parser.add_argument("--query-sweep", default=None)
    parser.add_argument("--max-entity-candidates", type=int, default=DEFAULT_MAX_ENTITY_CANDIDATES)
    parser.add_argument("--max-query-seeds", type=int, default=DEFAULT_MAX_QUERY_SEEDS)
    parser.add_argument("--json", required=True)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        blind_spots_path=args.blind_spots,
        provenance_comparison_path=args.provenance_comparison,
        query_sweep_path=args.query_sweep,
        output_path=args.json,
        max_entity_candidates=args.max_entity_candidates,
        max_query_seeds=args.max_query_seeds,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "blind_spot_evidence_expansion_plan_ok "
        f"status={payload['status']} "
        f"targets={summary['target_count']} "
        f"top_type={_first_key(summary['question_type_counts'])} "
        f"top_route={_first_key(summary['recommended_route_counts'])}"
    )


if __name__ == "__main__":
    main()
