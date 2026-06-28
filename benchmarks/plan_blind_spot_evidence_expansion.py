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
    targets = tuple(
        _record_target(
            record,
            max_entity_candidates=int(max_entity_candidates),
            max_query_seeds=int(max_query_seeds),
        )
        for record in records
    )
    summary = _summary(targets)
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
) -> dict[str, Any]:
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    text = str(record.get("text") or f"{question} {answer}").strip()
    question_type = str(record.get("question_type") or _question_type(question))
    features = _mapping(record.get("features"))
    answer_features = _mapping(record.get("answer_features"))
    routes = _routes(question_type, features=features, answer_features=answer_features)
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
        ),
        "features": {
            "claim": {key: bool(value) for key, value in features.items()},
            "answer": {key: bool(value) for key, value in answer_features.items()},
        },
        "question": question,
        "answer": answer,
        "text": text,
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
    for entity in entities:
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
    if value.casefold() in STOPWORDS:
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
