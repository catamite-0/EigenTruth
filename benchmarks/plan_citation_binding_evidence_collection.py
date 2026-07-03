"""Plan citation binding evidence collection from rejected source bindings.

The citation binding audit says which returned source documents are not safe to
use as retrieval evidence. This planner turns those rejected rows into
lane-specific, non-evidence collection requests for the next adapter or manual
review pass. It never promotes evidence and intentionally copies only sanitized
request fields from the audit records.
"""

from __future__ import annotations

import argparse
import hashlib
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

WORKFLOW = "citation_binding_evidence_collection_plan"


def plan_citation_binding_evidence_collection(
    binding_audits: Sequence[Mapping[str, Any]],
    *,
    max_examples_per_request: int = 3,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready, non-evidence collection plan."""
    if not binding_audits:
        raise ValueError("binding_audits must not be empty.")
    if int(max_examples_per_request) < 0:
        raise ValueError("max_examples_per_request must be non-negative.")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    issue_counts: Counter[str] = Counter()
    question_type_counts: Counter[str] = Counter()
    rejected_count = 0
    accepted_count = 0
    source_count = 0
    for audit_index, audit in enumerate(binding_audits, start=1):
        summary = _mapping(audit.get("summary"))
        source_count += _int(summary.get("source_document_count"))
        accepted_count += _int(summary.get("accepted_source_document_count"))
        for record in _mapping_records(audit.get("records")):
            if _clean(record.get("status")) == "accepted":
                continue
            rejected_count += 1
            issue_codes = _issue_codes(record)
            issue_counts.update(issue_codes)
            question_type = _clean(record.get("question_type")) or "unknown"
            question_type_counts[question_type] += 1
            lanes = _lanes_for_record(record, issue_codes=issue_codes)
            for lane in lanes:
                key = (_request_key(record, audit_index=audit_index), lane["lane"])
                grouped[key].append(_collection_record(record, audit_index=audit_index, lane=lane))

    requests = tuple(
        _collection_request(rows, max_examples=int(max_examples_per_request))
        for _, rows in sorted(grouped.items(), key=lambda item: (item[1][0]["lane"], item[0][0]))
    )
    summary = _summary(
        requests,
        audit_count=len(binding_audits),
        source_count=source_count,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        issue_counts=issue_counts,
        question_type_counts=question_type_counts,
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready_for_collection" if requests else "monitor",
        "scope": (
            "Non-evidence collection plan derived from rejected citation binding audit rows. "
            "Requests identify source-backed evidence gaps only; they are not verifier evidence."
        ),
        "summary": summary,
        "collection_requests": requests,
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    binding_audit_paths: Sequence[str | Path],
    report_json_path: str | Path,
    collection_requests_jsonl_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    max_examples_per_request: int = 3,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, optionally manifest, and optionally register a plan."""
    if not binding_audit_paths:
        raise ValueError("binding_audit_paths must not be empty.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")

    paths = tuple(Path(path) for path in binding_audit_paths)
    payload = plan_citation_binding_evidence_collection(
        tuple(_load_mapping(path) for path in paths),
        max_examples_per_request=max_examples_per_request,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["source"] = {"binding_audits": tuple(str(path) for path in paths)}
    report_path = Path(report_json_path)
    requests_path = Path(collection_requests_jsonl_path) if collection_requests_jsonl_path is not None else None
    payload["paths"] = {
        "collection_plan": str(report_path),
        "collection_requests": None if requests_path is None else str(requests_path),
    }
    if artifact_manifest_path is not None:
        payload["paths"]["artifact_manifest"] = str(artifact_manifest_path)

    _write_json(report_path, payload, compact=compact_json)
    if requests_path is not None:
        _write_jsonl(requests_path, _mapping_records(payload.get("collection_requests")), compact=compact_json)

    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        artifacts: dict[str, Path] = {
            "citation_binding_collection_plan": report_path,
            **{f"binding_audit_{index}": path for index, path in enumerate(paths, start=1)},
        }
        if requests_path is not None:
            artifacts["citation_binding_collection_requests"] = requests_path
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "collection_request_count": payload["summary"]["collection_request_count"],
                "dominant_lane": payload["summary"]["dominant_lane"],
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
                "collection_request_count": payload["summary"]["collection_request_count"],
                "dominant_lane": payload["summary"]["dominant_lane"],
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _lanes_for_record(record: Mapping[str, Any], *, issue_codes: Sequence[str]) -> tuple[dict[str, Any], ...]:
    issues = set(issue_codes)
    question_type = _clean(record.get("question_type"))
    requires_timestamp = bool(record.get("requires_timestamp"))
    lanes: list[dict[str, Any]] = []
    if "numeric_intent_requires_numeric_evidence" in issues or question_type == "quantity":
        lanes.append(_lane(
            "numeric_statistical_evidence",
            priority="high",
            goal="Collect source-backed numbers, denominators, units, and calculation context.",
            required_fields=("source_value", "unit", "denominator", "reference_time", "source_citation"),
            source_families=("official_statistics", "official", "scholarly"),
            adapter_hints=("world_bank_or_statistics", "official_site", "structured_numeric_fact"),
        ))
    if (
        "temporal_intent_requires_temporal_evidence" in issues
        or "missing_fresh_timestamp" in issues
        or requires_timestamp
        or question_type == "temporal"
    ):
        lanes.append(_lane(
            "temporal_evidence",
            priority="high",
            goal="Collect timestamped evidence with claim/source/retrieval time metadata.",
            required_fields=("claim_time", "source_time", "retrieved_at", "source_citation"),
            source_families=("official", "news", "official_statistics"),
            adapter_hints=("timestamped_source", "official_site", "news_archive"),
        ))
    if "causal_intent_requires_causal_evidence" in issues or question_type in {"causal", "method"}:
        lanes.append(_lane(
            "causal_procedural_evidence",
            priority="high",
            goal="Collect explicit mechanism, precondition, procedure, and mechanism-status evidence.",
            required_fields=("mechanism", "precondition", "mechanism_status", "source_citation"),
            source_families=("official", "scholarly", "domain_specific"),
            adapter_hints=("mechanism_rule_input", "scholarly_review", "procedure_documentation"),
        ))
    if "person_intent_requires_relation_evidence" in issues or question_type == "person":
        lanes.append(_lane(
            "role_specific_entity_evidence",
            priority="high",
            goal="Collect explicit subject-role-object evidence such as founder, author, inventor, or leader.",
            required_fields=("subject", "relation", "object", "source_citation"),
            source_families=("official", "reference", "scholarly"),
            adapter_hints=("entity_role_binding", "official_profile", "structured_fact"),
        ))
    if "location_intent_requires_location_evidence" in issues or question_type == "location":
        lanes.append(_lane(
            "location_specific_evidence",
            priority="medium",
            goal="Collect explicit place/location/country/capital relation evidence.",
            required_fields=("subject", "location_relation", "location", "source_citation"),
            source_families=("official", "reference", "domain_specific"),
            adapter_hints=("location_binding", "official_profile", "structured_fact"),
        ))
    if any(issue.startswith("source_family") for issue in issues):
        lanes.append(_lane(
            "source_family_catalog_expansion",
            priority="medium",
            goal="Expand or rerank source-family catalogs before rerunning citation binding.",
            required_fields=("source_family", "query", "provider", "source_citation"),
            source_families=("official", "official_statistics", "scholarly", "news", "domain_specific"),
            adapter_hints=("source_family_catalog_collection", "provider_specific_catalog"),
        ))
    if "missing_source_binding" in issues or "unknown_source_binding" in issues:
        lanes.append(_lane(
            "source_binding_provenance_repair",
            priority="high",
            goal="Repair adapter-result to sanitized-request binding fingerprints before evidence use.",
            required_fields=("adapter_request_sha256", "source_queue_request_sha256", "source_citation"),
            source_families=(),
            adapter_hints=("adapter_binding_repair",),
        ))
    if any(issue.startswith("evidence_alignment_insufficient") for issue in issues):
        lanes.append(_lane(
            "claim_specific_evidence_span",
            priority="medium",
            goal="Collect or extract a sentence-level span that directly supports or refutes the claim.",
            required_fields=("evidence_span", "span_source", "source_citation"),
            source_families=("official", "reference", "scholarly", "news", "domain_specific"),
            adapter_hints=("span_extraction", "claim_evidence_alignment"),
        ))
    if any(issue.startswith("evidence_alignment_misaligned") for issue in issues):
        lanes.append(_lane(
            "claim_alignment_review",
            priority="medium",
            goal="Review subject, property, value, and evidence-span alignment before reranking.",
            required_fields=("claim_subject", "claim_relation", "candidate_subject", "candidate_relation"),
            source_families=(),
            adapter_hints=("alignment_review", "query_refinement"),
        ))
    if not lanes:
        lanes.append(_lane(
            "binding_issue_review",
            priority="low",
            goal="Inspect the rejected binding issue and choose a narrower collection lane.",
            required_fields=("issue_code", "review_decision"),
            source_families=(),
            adapter_hints=("manual_review",),
        ))
    return tuple(_dedupe_lanes(lanes))


def _lane(
    lane: str,
    *,
    priority: str,
    goal: str,
    required_fields: Sequence[str],
    source_families: Sequence[str],
    adapter_hints: Sequence[str],
) -> dict[str, Any]:
    return {
        "lane": lane,
        "priority": priority,
        "collection_goal": goal,
        "required_fields": tuple(required_fields),
        "preferred_source_families": tuple(source_families),
        "adapter_hints": tuple(adapter_hints),
    }


def _collection_record(
    record: Mapping[str, Any],
    *,
    audit_index: int,
    lane: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "audit_index": audit_index,
        "lane": str(lane["lane"]),
        "lane_spec": dict(lane),
        "request_key": _request_key(record, audit_index=audit_index),
        "request_id": _clean(record.get("request_id")),
        "query": _clean(record.get("query")),
        "question_type": _clean(record.get("question_type")) or "unknown",
        "requires_timestamp": bool(record.get("requires_timestamp")),
        "source_document_index": _int(record.get("source_document_index")),
        "source": _clean(record.get("source")),
        "issue_codes": _issue_codes(record),
        "alignment": _safe_alignment_summary(_mapping(record.get("alignment"))),
        "intent": _safe_reason(record.get("intent")),
        "source_family": _safe_reason(record.get("source_family")),
    }


def _collection_request(rows: Sequence[Mapping[str, Any]], *, max_examples: int) -> dict[str, Any]:
    first = rows[0]
    lane_spec = _mapping(first.get("lane_spec"))
    issue_codes = _dedupe(
        issue
        for row in rows
        for issue in _string_sequence(row.get("issue_codes"))
    )
    request_id = _clean(first.get("request_id"))
    query = _clean(first.get("query"))
    digest = hashlib.sha256(
        "|".join((str(first.get("request_key")), str(first.get("lane")), query)).encode("utf-8")
    ).hexdigest()[:12]
    source_indices = _dedupe(
        str(row.get("source_document_index"))
        for row in rows
        if row.get("source_document_index") is not None
    )
    examples = tuple(_example(row) for row in rows[:max_examples])
    return {
        "collection_request_id": f"citation-binding:{first['lane']}:{digest}",
        "lane": first["lane"],
        "priority": lane_spec.get("priority", "medium"),
        "request_id": request_id,
        "query": query,
        "question_type": first.get("question_type"),
        "requires_timestamp": bool(first.get("requires_timestamp")),
        "source_document_count": len(rows),
        "source_document_indices": source_indices,
        "issue_codes": issue_codes,
        "collection_goal": lane_spec.get("collection_goal"),
        "required_fields": tuple(lane_spec.get("required_fields", ())),
        "preferred_source_families": tuple(lane_spec.get("preferred_source_families", ())),
        "adapter_hints": tuple(lane_spec.get("adapter_hints", ())),
        "query_seeds": _query_seeds(query=query, question_type=str(first.get("question_type") or "")),
        "examples": examples,
        "provenance": {
            "not_verifier_evidence": True,
            "source_workflow": WORKFLOW,
            "audit_indices": _dedupe(str(row.get("audit_index")) for row in rows),
        },
    }


def _query_seeds(*, query: str, question_type: str) -> tuple[str, ...]:
    seeds = [query]
    if question_type:
        seeds.append(f"{question_type} {query}".strip())
    return tuple(item for item in _dedupe(_clean(seed) for seed in seeds) if item)


def _safe_alignment_summary(alignment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": alignment.get("status"),
        "keyword_overlap": alignment.get("keyword_overlap"),
        "number_recall": alignment.get("number_recall"),
        "entity_recall": alignment.get("entity_recall"),
        "issue_codes": tuple(_string_sequence(alignment.get("issue_codes"))),
    }


def _safe_reason(value: Any) -> dict[str, Any]:
    mapping = _mapping(value)
    return {
        key: mapping.get(key)
        for key in ("status", "reason", "expected_family", "actual_family")
        if key in mapping
    }


def _example(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_document_index": row.get("source_document_index"),
        "source": row.get("source"),
        "alignment": row.get("alignment"),
        "intent": row.get("intent"),
        "source_family": row.get("source_family"),
    }


def _summary(
    requests: Sequence[Mapping[str, Any]],
    *,
    audit_count: int,
    source_count: int,
    accepted_count: int,
    rejected_count: int,
    issue_counts: Counter[str],
    question_type_counts: Counter[str],
) -> dict[str, Any]:
    lane_counts = Counter(str(request.get("lane")) for request in requests)
    priority_counts = Counter(str(request.get("priority")) for request in requests)
    source_family_counts: Counter[str] = Counter()
    adapter_hint_counts: Counter[str] = Counter()
    for request in requests:
        source_family_counts.update(_string_sequence(request.get("preferred_source_families")))
        adapter_hint_counts.update(_string_sequence(request.get("adapter_hints")))
    return {
        "audit_count": audit_count,
        "source_document_count": source_count,
        "accepted_source_document_count": accepted_count,
        "rejected_source_document_count": rejected_count,
        "collection_request_count": len(requests),
        "lane_counts": _sorted_counter(lane_counts),
        "priority_counts": _sorted_counter(priority_counts),
        "issue_counts": _sorted_counter(issue_counts),
        "question_type_counts": _sorted_counter(question_type_counts),
        "preferred_source_family_counts": _sorted_counter(source_family_counts),
        "adapter_hint_counts": _sorted_counter(adapter_hint_counts),
        "dominant_lane": _counter_first(lane_counts),
    }


def _mapping_records(value: Any) -> tuple[Mapping[str, Any], ...]:
    return tuple(item for item in _sequence(value) if isinstance(item, Mapping))


def _dedupe_lanes(lanes: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for lane in lanes:
        key = str(lane.get("lane"))
        if key in seen:
            continue
        seen.add(key)
        result.append(lane)
    return tuple(result)


def _request_key(record: Mapping[str, Any], *, audit_index: int) -> str:
    request_id = _clean(record.get("request_id"))
    if request_id:
        return request_id
    return f"audit-{audit_index}:source-{_int(record.get('source_document_index'))}"


def _issue_codes(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in _sequence(record.get("issue_codes")) if str(item))


def _load_mapping(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--metadata entries must be key=value.")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--metadata keys must be non-empty.")
        metadata[key] = raw
    return metadata


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    indent = None if compact else 2
    output.write_text(strict_json_dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    separators = (",", ":") if compact else None
    output.write_text(
        "".join(strict_json_dumps(row, sort_keys=True, separators=separators) + "\n" for row in rows),
        encoding="utf-8",
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _string_sequence(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in _sequence(value) if str(item))


def _dedupe(values: Sequence[str] | Any) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(((key, value) for key, value in counter.items() if key), key=lambda item: (-item[1], item[0])))


def _counter_first(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan collection work from citation binding audit rejects.")
    parser.add_argument("--binding-audit", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collection-requests-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--max-examples-per-request", type=int, default=3)
    parser.add_argument("--metadata", action="append", default=None)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args()
    payload = run(
        binding_audit_paths=tuple(args.binding_audit),
        report_json_path=args.output,
        collection_requests_jsonl_path=args.collection_requests_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        max_examples_per_request=args.max_examples_per_request,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    print(
        f"{WORKFLOW}_ok status={payload['status']} "
        f"requests={payload['summary']['collection_request_count']} output={args.output}"
    )


if __name__ == "__main__":
    main()
