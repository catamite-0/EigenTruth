"""Build executable source-discovery batches from a blind-spot evidence plan.

This workflow compiles ``plan_blind_spot_evidence_expansion.py`` output into a
machine-readable collection corpus. The result is not verifier evidence: it is
a source-discovery queue that can drive Wikidata/entity resolution, citation
retrieval, counterfactual probe generation, and world-model/calculator rule
authoring before a provenance-gated verifier rerun.
"""

from __future__ import annotations

import argparse
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

DEFAULT_PRIORITIES = ("high", "medium", "low")
DEFAULT_MAX_WIKIDATA_REQUESTS_PER_TARGET = 12
DEFAULT_MAX_CITATION_REQUESTS_PER_TARGET = 4
PROPERTY_ID_RE = re.compile(r"\b(P[1-9][0-9]*)\b")
NON_PROPERTY_HINTS = {
    "description",
    "disambiguation",
    "external_citation",
    "procedure_citation",
    "causal_citation",
    "quantity_property",
    "retrieved_at_required",
}
COUNTERFACTUAL_ROUTE_PREFIX = "counterfactual_"
ALIGNMENT_ROUTES = {
    "claim_evidence_alignment",
    "source_document_fact_extraction",
    "query_refinement",
    "negative_control_alignment_audit",
}


def build_blind_spot_evidence_collection_corpus(
    *,
    plan_path: str | Path,
    priorities: Sequence[str] = DEFAULT_PRIORITIES,
    routes: Sequence[str] = (),
    max_targets: int | None = None,
    max_wikidata_requests_per_target: int = DEFAULT_MAX_WIKIDATA_REQUESTS_PER_TARGET,
    max_citation_requests_per_target: int = DEFAULT_MAX_CITATION_REQUESTS_PER_TARGET,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready collection corpus compiled from an expansion plan."""
    plan = _load_plan(plan_path)
    selected_priorities = _normalize_priorities(priorities)
    selected_routes = tuple(str(item).strip() for item in routes if str(item).strip())
    if max_targets is not None and int(max_targets) <= 0:
        raise ValueError("max_targets must be positive when provided.")
    if int(max_wikidata_requests_per_target) <= 0:
        raise ValueError("max_wikidata_requests_per_target must be positive.")
    if int(max_citation_requests_per_target) <= 0:
        raise ValueError("max_citation_requests_per_target must be positive.")

    targets = tuple(_select_targets(
        plan.get("targets", ()),
        priorities=selected_priorities,
        routes=selected_routes,
        max_targets=max_targets,
    ))
    target_snapshots: list[dict[str, Any]] = []
    wikidata_requests: list[dict[str, Any]] = []
    citation_requests: list[dict[str, Any]] = []
    counterfactual_requests: list[dict[str, Any]] = []
    rule_requests: list[dict[str, Any]] = []
    alignment_requests: list[dict[str, Any]] = []
    source_discovery_documents: list[dict[str, Any]] = []

    for ordinal, target in enumerate(targets, start=1):
        target_id = _target_id(target, ordinal)
        snapshot = _target_snapshot(target, target_id=target_id, ordinal=ordinal)
        target_snapshots.append(snapshot)
        wikidata = _wikidata_requests(
            target,
            target_id=target_id,
            max_items=int(max_wikidata_requests_per_target),
        )
        citations = _citation_requests(
            target,
            target_id=target_id,
            max_items=int(max_citation_requests_per_target),
        )
        counterfactuals = _counterfactual_requests(target, target_id=target_id)
        rules = _rule_requests(target, target_id=target_id)
        alignment = _alignment_requests(target, target_id=target_id)
        wikidata_requests.extend(wikidata)
        citation_requests.extend(citations)
        counterfactual_requests.extend(counterfactuals)
        rule_requests.extend(rules)
        alignment_requests.extend(alignment)
        source_discovery_documents.extend(_source_discovery_documents(wikidata, citations))

    summary = _summary(
        target_snapshots=target_snapshots,
        wikidata_requests=wikidata_requests,
        citation_requests=citation_requests,
        counterfactual_requests=counterfactual_requests,
        rule_requests=rule_requests,
        alignment_requests=alignment_requests,
        source_discovery_documents=source_discovery_documents,
    )
    return {
        "schema_version": 1,
        "workflow": "blind_spot_evidence_collection_corpus",
        "status": "ready_for_collection" if summary["total_request_count"] else "empty",
        "source": {
            "plan_path": str(plan_path),
            "plan_workflow": plan.get("workflow"),
            "plan_status": plan.get("status"),
            "plan_target_count": _nested_int(plan, "summary", "target_count"),
        },
        "label_usage": {
            "labels_used_for_collection_requests": False,
            "labels_copied_to_collection_requests": False,
            "requests_are_verifier_evidence": False,
        },
        "config": {
            "priorities": selected_priorities,
            "routes": selected_routes,
            "max_targets": max_targets,
            "max_wikidata_requests_per_target": int(max_wikidata_requests_per_target),
            "max_citation_requests_per_target": int(max_citation_requests_per_target),
        },
        "summary": summary,
        "targets": target_snapshots,
        "requests": {
            "wikidata_entity_property": wikidata_requests,
            "external_citation": citation_requests,
            "counterfactual_probe": counterfactual_requests,
            "world_model_or_calculator_rule": rule_requests,
            "alignment_audit": alignment_requests,
        },
        "source_discovery_documents": source_discovery_documents,
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    plan_path: str | Path,
    output_path: str | Path,
    priorities: Sequence[str] = DEFAULT_PRIORITIES,
    routes: Sequence[str] = (),
    max_targets: int | None = None,
    max_wikidata_requests_per_target: int = DEFAULT_MAX_WIKIDATA_REQUESTS_PER_TARGET,
    max_citation_requests_per_target: int = DEFAULT_MAX_CITATION_REQUESTS_PER_TARGET,
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
    payload = build_blind_spot_evidence_collection_corpus(
        plan_path=plan_path,
        priorities=priorities,
        routes=routes,
        max_targets=max_targets,
        max_wikidata_requests_per_target=max_wikidata_requests_per_target,
        max_citation_requests_per_target=max_citation_requests_per_target,
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
                "blind_spot_evidence_collection_corpus": output,
                "blind_spot_evidence_expansion_plan": plan_path,
            },
            root=manifest_path.parent,
            metadata={
                "runner": "build_blind_spot_evidence_collection_corpus",
                "status": payload["status"],
                "target_count": payload["summary"]["target_count"],
                "total_request_count": payload["summary"]["total_request_count"],
                "wikidata_request_count": payload["summary"]["request_counts"]["wikidata_entity_property"],
                "citation_request_count": payload["summary"]["request_counts"]["external_citation"],
                "alignment_request_count": payload["summary"]["request_counts"]["alignment_audit"],
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
                "total_request_count": payload["summary"]["total_request_count"],
                "wikidata_request_count": payload["summary"]["request_counts"]["wikidata_entity_property"],
                "citation_request_count": payload["summary"]["request_counts"]["external_citation"],
                "alignment_request_count": payload["summary"]["request_counts"]["alignment_audit"],
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _select_targets(
    raw_targets: Any,
    *,
    priorities: Sequence[str],
    routes: Sequence[str],
    max_targets: int | None,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes, bytearray)):
        raise ValueError("evidence expansion plan must include a targets array.")
    selected: list[Mapping[str, Any]] = []
    route_set = set(routes)
    for item in raw_targets:
        if not isinstance(item, Mapping):
            continue
        priority = str(item.get("priority", "")).strip()
        if priority not in priorities:
            continue
        if route_set and not route_set.intersection(str(route) for route in _sequence(item.get("recommended_routes"))):
            continue
        selected.append(item)
        if max_targets is not None and len(selected) >= int(max_targets):
            break
    return tuple(selected)


def _target_snapshot(target: Mapping[str, Any], *, target_id: str, ordinal: int) -> dict[str, Any]:
    snapshot = {
        "target_id": target_id,
        "ordinal": int(ordinal),
        "record_index": _optional_int(target.get("record_index")),
        "priority": str(target.get("priority", "")),
        "question_type": str(target.get("question_type", "")),
        "question": str(target.get("question", "")),
        "model_answer": str(target.get("answer", "")),
        "entity_candidates": tuple(str(item) for item in _sequence(target.get("entity_candidates"))),
        "recommended_routes": tuple(str(item) for item in _sequence(target.get("recommended_routes"))),
        "wikidata_property_hints": tuple(str(item) for item in _sequence(target.get("wikidata_property_hints"))),
        "query_seeds": tuple(str(item) for item in _sequence(target.get("query_seeds"))),
    }
    if isinstance(target.get("query_sweep_gap_guidance"), Mapping):
        snapshot["query_sweep_gap_guidance"] = dict(target["query_sweep_gap_guidance"])
    return snapshot


def _wikidata_requests(
    target: Mapping[str, Any],
    *,
    target_id: str,
    max_items: int,
) -> tuple[dict[str, Any], ...]:
    entities = tuple(str(item).strip() for item in _sequence(target.get("entity_candidates")) if str(item).strip())
    hints = tuple(str(item).strip() for item in _sequence(target.get("wikidata_property_hints")) if str(item).strip())
    if not entities or not hints:
        return ()
    requests: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entity in entities:
        for hint in hints:
            key = (entity.casefold(), hint.casefold())
            if key in seen:
                continue
            seen.add(key)
            property_id = _property_id(hint)
            requests.append({
                "request_id": f"wd:{target_id}:{len(requests) + 1}",
                "target_id": target_id,
                "request_type": "wikidata_entity_property",
                "priority": str(target.get("priority", "")),
                "question_type": str(target.get("question_type", "")),
                "entity": entity,
                "property_hint": hint,
                "property_id": property_id,
                "property_hint_kind": "wikidata_property" if property_id else _non_property_hint_kind(hint),
                "question": str(target.get("question", "")),
                "model_answer": str(target.get("answer", "")),
                "search_text": _join_nonempty((
                    entity,
                    _property_search_label(hint),
                    str(target.get("question", "")),
                )),
                "usage": "source_discovery_only",
            })
            if len(requests) >= max_items:
                return tuple(requests)
    return tuple(requests)


def _citation_requests(
    target: Mapping[str, Any],
    *,
    target_id: str,
    max_items: int,
) -> tuple[dict[str, Any], ...]:
    routes = set(str(route) for route in _sequence(target.get("recommended_routes")))
    if "retrieval_citation" not in routes and "time_sensitive_retrieval" not in routes:
        return ()
    seeds = tuple(str(item).strip() for item in _sequence(target.get("query_seeds")) if str(item).strip())
    requests = []
    for idx, seed in enumerate(seeds[:max_items], start=1):
        requests.append({
            "request_id": f"cite:{target_id}:{idx}",
            "target_id": target_id,
            "request_type": "external_citation",
            "priority": str(target.get("priority", "")),
            "question_type": str(target.get("question_type", "")),
            "query": seed,
            "requires_timestamp": _requires_timestamp(target),
            "question": str(target.get("question", "")),
            "model_answer": str(target.get("answer", "")),
            "usage": "source_discovery_only",
        })
    return tuple(requests)


def _counterfactual_requests(target: Mapping[str, Any], *, target_id: str) -> tuple[dict[str, Any], ...]:
    routes = tuple(str(route) for route in _sequence(target.get("recommended_routes")))
    probe_types = tuple(
        route[len(COUNTERFACTUAL_ROUTE_PREFIX):]
        for route in routes
        if route.startswith(COUNTERFACTUAL_ROUTE_PREFIX)
    )
    if not probe_types:
        return ()
    requests = []
    for idx, probe_type in enumerate(dict.fromkeys(probe_types), start=1):
        requests.append({
            "request_id": f"cf:{target_id}:{idx}",
            "target_id": target_id,
            "request_type": "counterfactual_probe",
            "priority": str(target.get("priority", "")),
            "question_type": str(target.get("question_type", "")),
            "probe_type": probe_type,
            "question": str(target.get("question", "")),
            "model_answer": str(target.get("answer", "")),
            "entity_candidates": tuple(str(item) for item in _sequence(target.get("entity_candidates"))),
            "probe_instruction": _counterfactual_instruction(target, probe_type=probe_type),
            "usage": "route_robustness_probe",
        })
    return tuple(requests)


def _rule_requests(target: Mapping[str, Any], *, target_id: str) -> tuple[dict[str, Any], ...]:
    routes = set(str(route) for route in _sequence(target.get("recommended_routes")))
    if not routes.intersection({"world_model_rule", "calculator", "time_sensitive_retrieval"}):
        return ()
    families = _rule_families(target, routes=routes)
    return tuple({
        "request_id": f"rule:{target_id}:{idx}",
        "target_id": target_id,
        "request_type": "world_model_or_calculator_rule",
        "priority": str(target.get("priority", "")),
        "question_type": str(target.get("question_type", "")),
        "rule_family": family,
        "question": str(target.get("question", "")),
        "model_answer": str(target.get("answer", "")),
        "rule_seed": _rule_seed(target, family=family),
        "usage": "deterministic_check_authoring",
    } for idx, family in enumerate(families, start=1))


def _alignment_requests(target: Mapping[str, Any], *, target_id: str) -> tuple[dict[str, Any], ...]:
    routes = set(str(route) for route in _sequence(target.get("recommended_routes")))
    alignment_routes = tuple(route for route in ALIGNMENT_ROUTES if route in routes)
    guidance = target.get("query_sweep_gap_guidance")
    guidance_actions = (
        tuple(
            str(item)
            for item in _sequence(guidance.get("recommended_alignment_actions"))
            if str(item) in ALIGNMENT_ROUTES
        )
        if isinstance(guidance, Mapping)
        else ()
    )
    actions = tuple(dict.fromkeys((*guidance_actions, *alignment_routes)))
    if not actions:
        return ()
    return ({
        "request_id": f"align:{target_id}:1",
        "target_id": target_id,
        "request_type": "claim_evidence_alignment",
        "priority": str(target.get("priority", "")),
        "question_type": str(target.get("question_type", "")),
        "alignment_actions": actions,
        "dominant_gap_bucket": None if not isinstance(guidance, Mapping) else guidance.get("dominant_gap_bucket"),
        "query_sweep_best_strategy": None if not isinstance(guidance, Mapping) else guidance.get("best_strategy"),
        "top_hit_sources": (
            () if not isinstance(guidance, Mapping)
            else tuple(str(item) for item in _sequence(guidance.get("top_hit_sources")))
        ),
        "question": str(target.get("question", "")),
        "model_answer": str(target.get("answer", "")),
        "entity_candidates": tuple(str(item) for item in _sequence(target.get("entity_candidates"))),
        "wikidata_property_hints": tuple(str(item) for item in _sequence(target.get("wikidata_property_hints"))),
        "query_seeds": tuple(str(item) for item in _sequence(target.get("query_seeds"))),
        "alignment_instruction": _alignment_instruction(target, actions=actions),
        "usage": "alignment_audit_only",
    },)


def _source_discovery_documents(
    wikidata_requests: Sequence[Mapping[str, Any]],
    citation_requests: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    documents = []
    for request in wikidata_requests:
        documents.append({
            "text": str(request.get("search_text", "")),
            "source": f"collection-request:{request['request_id']}",
            "metadata": {
                "collection_request": True,
                "usage": "source_discovery_only",
                "request_id": request["request_id"],
                "target_id": request["target_id"],
                "request_type": request["request_type"],
                "provider_hint": "wikidata",
                "property_hint": request.get("property_hint"),
                "property_id": request.get("property_id"),
            },
        })
    for request in citation_requests:
        documents.append({
            "text": str(request.get("query", "")),
            "source": f"collection-request:{request['request_id']}",
            "metadata": {
                "collection_request": True,
                "usage": "source_discovery_only",
                "request_id": request["request_id"],
                "target_id": request["target_id"],
                "request_type": request["request_type"],
                "provider_hint": "external_citation",
                "requires_timestamp": bool(request.get("requires_timestamp")),
            },
        })
    return tuple(documents)


def _summary(
    *,
    target_snapshots: Sequence[Mapping[str, Any]],
    wikidata_requests: Sequence[Mapping[str, Any]],
    citation_requests: Sequence[Mapping[str, Any]],
    counterfactual_requests: Sequence[Mapping[str, Any]],
    rule_requests: Sequence[Mapping[str, Any]],
    alignment_requests: Sequence[Mapping[str, Any]],
    source_discovery_documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    priority_counts = Counter(str(item.get("priority")) for item in target_snapshots)
    question_type_counts = Counter(str(item.get("question_type")) for item in target_snapshots)
    property_ids = Counter(
        str(item.get("property_id"))
        for item in wikidata_requests
        if item.get("property_id")
    )
    request_counts = {
        "wikidata_entity_property": len(wikidata_requests),
        "external_citation": len(citation_requests),
        "counterfactual_probe": len(counterfactual_requests),
        "world_model_or_calculator_rule": len(rule_requests),
        "alignment_audit": len(alignment_requests),
    }
    return {
        "target_count": len(target_snapshots),
        "priority_counts": _sorted_counter(priority_counts),
        "question_type_counts": _sorted_counter(question_type_counts),
        "request_counts": request_counts,
        "total_request_count": sum(request_counts.values()),
        "source_discovery_document_count": len(source_discovery_documents),
        "wikidata_property_request_count": sum(
            1 for item in wikidata_requests if item.get("property_id")
        ),
        "wikidata_non_property_request_count": sum(
            1 for item in wikidata_requests if not item.get("property_id")
        ),
        "top_wikidata_property_ids": _sorted_counter(property_ids),
        "targets_with_counterfactual_requests": len({
            str(item.get("target_id")) for item in counterfactual_requests
        }),
        "targets_with_rule_requests": len({
            str(item.get("target_id")) for item in rule_requests
        }),
        "targets_with_alignment_requests": len({
            str(item.get("target_id")) for item in alignment_requests
        }),
    }


def _load_plan(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("evidence expansion plan must be a JSON object.")
    if payload.get("workflow") != "blind_spot_evidence_expansion_plan":
        raise ValueError(f"{path} is not a blind_spot_evidence_expansion_plan report.")
    return dict(payload)


def _normalize_priorities(values: Sequence[str]) -> tuple[str, ...]:
    priorities = tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if not priorities:
        raise ValueError("at least one priority is required.")
    valid = set(DEFAULT_PRIORITIES)
    invalid = sorted(set(priorities) - valid)
    if invalid:
        raise ValueError(f"unsupported priorities: {', '.join(invalid)}")
    return priorities


def _property_id(hint: str) -> str | None:
    match = PROPERTY_ID_RE.search(hint)
    return None if match is None else match.group(1)


def _non_property_hint_kind(hint: str) -> str:
    normalized = hint.strip().casefold()
    return "non_property_hint" if normalized not in NON_PROPERTY_HINTS else normalized


def _property_search_label(hint: str) -> str:
    text = hint.split(":", 1)[0]
    return text.replace("_", " ").strip() or hint


def _requires_timestamp(target: Mapping[str, Any]) -> bool:
    hints = set(str(item) for item in _sequence(target.get("wikidata_property_hints")))
    routes = set(str(item) for item in _sequence(target.get("recommended_routes")))
    return "retrieved_at_required" in hints or "time_sensitive_retrieval" in routes


def _counterfactual_instruction(target: Mapping[str, Any], *, probe_type: str) -> str:
    question = str(target.get("question", ""))
    answer = str(target.get("answer", ""))
    if probe_type == "entity_swap":
        entity = next(iter(_sequence(target.get("entity_candidates"))), "")
        return (
            "Generate a paired probe by replacing the primary entity"
            f"{f' ({entity})' if entity else ''} while preserving the question form. "
            f"Original question: {question} Model answer: {answer}"
        )
    if probe_type == "negation":
        return f"Generate a negated and non-negated pair for: {question} Model answer: {answer}"
    if probe_type == "quantity":
        return f"Generate nearby numeric alternatives and exact-calculation checks for: {question}"
    if probe_type == "temporal":
        return f"Generate time-stamped variants and require retrieval timestamp for: {question}"
    return f"Generate a targeted {probe_type} counterfactual probe for: {question}"


def _rule_families(target: Mapping[str, Any], *, routes: set[str]) -> tuple[str, ...]:
    families: list[str] = []
    question_type = str(target.get("question_type", ""))
    if "calculator" in routes or question_type == "quantity":
        families.append("quantity_or_arithmetic")
    if "time_sensitive_retrieval" in routes or question_type == "temporal":
        families.append("temporal_freshness")
    if "world_model_rule" in routes or question_type in {"method", "causal"}:
        families.append("causal_or_procedural_consistency")
    if not families:
        families.append("world_model_consistency")
    return tuple(dict.fromkeys(families))


def _rule_seed(target: Mapping[str, Any], *, family: str) -> str:
    question = str(target.get("question", ""))
    answer = str(target.get("answer", ""))
    if family == "quantity_or_arithmetic":
        return f"Check whether the numeric claim in '{question}' is entailed by external values; model answer: {answer}"
    if family == "temporal_freshness":
        return f"Check the answer with an explicit retrieval timestamp: {question} model answer: {answer}"
    if family == "causal_or_procedural_consistency":
        return f"Check causal/procedural steps against an external source: {question} model answer: {answer}"
    return f"Check the answer against a deterministic world-model rule: {question} model answer: {answer}"


def _alignment_instruction(target: Mapping[str, Any], *, actions: Sequence[str]) -> str:
    question = str(target.get("question", ""))
    answer = str(target.get("answer", ""))
    action_text = ", ".join(actions)
    return (
        "Audit claim-evidence alignment before route promotion: extract the subject, "
        "property, proposed value, contradictory value when present, and exact evidence spans "
        f"for actions [{action_text}]. Question: {question} Model answer: {answer}"
    )


def _target_id(target: Mapping[str, Any], ordinal: int) -> str:
    record_index = _optional_int(target.get("record_index"))
    if record_index is None or record_index < 0:
        return f"target-{ordinal}"
    return f"record-{record_index}"


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
    parser.add_argument("--json", required=True)
    parser.add_argument("--priority", action="append", choices=DEFAULT_PRIORITIES, default=None)
    parser.add_argument("--route", action="append", default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--max-wikidata-requests-per-target", type=int,
                        default=DEFAULT_MAX_WIKIDATA_REQUESTS_PER_TARGET)
    parser.add_argument("--max-citation-requests-per-target", type=int,
                        default=DEFAULT_MAX_CITATION_REQUESTS_PER_TARGET)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        plan_path=args.plan,
        output_path=args.json,
        priorities=tuple(args.priority or DEFAULT_PRIORITIES),
        routes=tuple(args.route or ()),
        max_targets=args.max_targets,
        max_wikidata_requests_per_target=args.max_wikidata_requests_per_target,
        max_citation_requests_per_target=args.max_citation_requests_per_target,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "blind_spot_evidence_collection_corpus_ok "
        f"status={payload['status']} "
        f"targets={summary['target_count']} "
        f"requests={summary['total_request_count']} "
        f"wikidata={summary['request_counts']['wikidata_entity_property']} "
        f"citations={summary['request_counts']['external_citation']}"
    )


if __name__ == "__main__":
    main()
