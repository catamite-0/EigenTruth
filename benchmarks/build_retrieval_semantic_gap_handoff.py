"""Build handoff queues for source-backed retrieval semantic gaps.

This workflow consumes ``eval_verifier_ensemble.py --verified-records-jsonl``
sidecars after retrieval sweeps. It identifies rows where retrieval has already
found local evidence but the verifier still cannot decide, then writes review
and authoring requests for structured facts, world-model rules, and alignment
audits. The output is not verifier evidence; it is an auditable next-step queue.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402
from eigentruth.verify import claim_entity_candidates, claim_features  # noqa: E402
from eigentruth.verify.features import flag_value_enabled  # noqa: E402

WORKFLOW = "retrieval_semantic_gap_handoff"
MODES = ("false_negative_with_hits", "unresolved_with_hits", "all_with_hits")
UNRESOLVED_STATUSES = {"insufficient_evidence", "not_applicable", "error"}
REFUTED_STATUS = "refuted"
DEFAULT_MAX_HITS_PER_TARGET = 3
STRUCTURED_SOURCE_PREFIXES = ("wikidata:", "worldbank:", "official:", "qa:", "structured:")
STRUCTURED_METADATA_KEYS = (
    "statement_property",
    "statement_property_label",
    "property_id",
    "property_label",
    "country_name",
    "indicator_name",
    "subject_label",
    "object_label",
)
DESCRIBED_AS_RE = re.compile(r"\bis\s+described\s+as\s+(?P<value>[^.]+)", re.IGNORECASE)
DESCRIBED_SUBJECT_RE = re.compile(
    r"^\s*(?:according\s+to\s+[^,]+,\s*)?"
    r"(?P<subject>.+?)\s+is\s+described\s+as\s+(?P<value>[^.]+)",
    re.IGNORECASE,
)
WIKIDATA_STRUCTURED_RE = re.compile(
    r"^\s*According\s+to\s+Wikidata\s+structured\s+data,\s*"
    r"(?P<subject>.+?)\s+has\s+(?P<relation>[^.]+)",
    re.IGNORECASE,
)
WORLDBANK_STAT_RE = re.compile(
    r"World\s+Bank\s+official\s+statistics\s+data[^:]*:\s*"
    r"(?P<subject>.+?)\s+had\s+(?P<property>.+?)\s+of\s+"
    r"(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)\s+in\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
GENERIC_ENTITY_CANDIDATES = {
    "according",
    "did",
    "do",
    "does",
    "more",
    "no",
    "not",
    "the",
    "truth",
    "yes",
}
WIKIDATA_PROPERTY_LABEL_BY_ID = {
    "P17": "country",
    "P27": "country of citizenship",
    "P31": "instance of",
    "P36": "capital",
    "P106": "occupation",
    "P112": "founder",
    "P361": "part of",
    "P495": "country of origin",
    "P527": "has part(s)",
    "P856": "official website",
}


def build_retrieval_semantic_gap_handoff(
    verified_records: Sequence[Mapping[str, Any]],
    *,
    mode: str = "false_negative_with_hits",
    min_hits: int = 1,
    max_targets: int | None = None,
    max_hits_per_target: int = DEFAULT_MAX_HITS_PER_TARGET,
    record_indices: Sequence[int] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready semantic-gap handoff queue."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}.")
    if int(min_hits) < 0:
        raise ValueError("min_hits must be non-negative.")
    if max_targets is not None and int(max_targets) <= 0:
        raise ValueError("max_targets must be positive when provided.")
    if int(max_hits_per_target) <= 0:
        raise ValueError("max_hits_per_target must be positive.")
    if not verified_records:
        raise ValueError("verified_records must not be empty.")
    record_index_filter = None if record_indices is None else frozenset(int(item) for item in record_indices)
    if record_index_filter is not None and not record_index_filter:
        raise ValueError("record_indices must not be empty when provided.")

    targets: list[dict[str, Any]] = []
    requests: dict[str, list[dict[str, Any]]] = {
        "claim_evidence_alignment_review": [],
        "structured_fact_candidate": [],
        "world_model_rule_candidate": [],
        "retrieval_query_refinement": [],
    }

    source_route_counts: Counter[str] = Counter()
    source_status_counts: Counter[str] = Counter()
    source_decision_rule_counts: Counter[str] = Counter()
    skipped_reason_counts: Counter[str] = Counter()

    for source_ordinal, item in enumerate(verified_records, start=1):
        record = _record_payload(item)
        if record_index_filter is not None and _optional_int(item.get("record_index")) not in record_index_filter:
            skipped_reason_counts["outside_record_index_filter"] += 1
            continue
        label = _label_key(item, record)
        final = _mapping(record.get("final"))
        status = str(final.get("status", "unknown"))
        decision_rule = _decision_rule(final)
        selected_route = _selected_route(record)
        hits = _retrieval_hits(record)
        source_route_counts[selected_route] += 1
        source_status_counts[status] += 1
        source_decision_rule_counts[decision_rule] += 1
        matches, skipped_reason = _matches_mode(
            mode=mode,
            label=label,
            status=status,
            hit_count=len(hits),
            min_hits=int(min_hits),
        )
        if not matches:
            skipped_reason_counts[skipped_reason] += 1
            continue
        target = _target_payload(
            item,
            record=record,
            source_ordinal=source_ordinal,
            max_hits_per_target=int(max_hits_per_target),
            label=label,
            final_status=status,
            decision_rule=decision_rule,
            selected_route=selected_route,
            hits=hits,
            mode=mode,
        )
        targets.append(target)
        for request_type, request in _requests_for_target(target).items():
            if request is not None:
                requests[request_type].append(request)
        if max_targets is not None and len(targets) >= int(max_targets):
            break

    fact_candidates = tuple(
        candidate
        for target in targets
        for candidate in _fact_candidates_for_target(target)
    )
    summary = _summary(
        verified_records=verified_records,
        targets=targets,
        requests=requests,
        fact_candidates=fact_candidates,
        source_route_counts=source_route_counts,
        source_status_counts=source_status_counts,
        source_decision_rule_counts=source_decision_rule_counts,
        skipped_reason_counts=skipped_reason_counts,
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready_for_handoff" if targets else "empty",
        "label_usage": {
            "labels_used_for_gap_selection": mode == "false_negative_with_hits",
            "labels_copied_to_targets": True,
            "handoff_is_verifier_evidence": False,
            "requests_are_verifier_evidence": False,
        },
        "config": {
            "mode": mode,
            "min_hits": int(min_hits),
            "max_targets": max_targets,
            "max_hits_per_target": int(max_hits_per_target),
            "record_index_filter_count": None if record_index_filter is None else len(record_index_filter),
        },
        "summary": summary,
        "targets": targets,
        "requests": {key: tuple(value) for key, value in requests.items()},
        "fact_candidates": fact_candidates,
        "metadata": dict(metadata or {}),
    }


def load_verified_records_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load compact verified-record sidecar rows."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"verified-records line {line_no} is not a JSON object.")
            records.append(dict(payload))
    if not records:
        raise ValueError("verified-records JSONL did not contain any records.")
    return tuple(records)


def _load_record_indices_json(path: str | Path) -> tuple[int, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        for key in ("records", "targets", "blind_spots", "items"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                payload = value
                break
        else:
            payload = (payload,)
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise ValueError("record index filter JSON must be an object or array.")
    indices = []
    for ordinal, item in enumerate(payload, start=1):
        value = item.get("record_index") if isinstance(item, Mapping) else item
        index = _optional_int(value)
        if index is None:
            raise ValueError(f"record index filter item {ordinal} is missing record_index.")
        indices.append(index)
    if not indices:
        raise ValueError("record index filter did not contain any indices.")
    return tuple(dict.fromkeys(indices))


def run(
    *,
    verified_records_jsonl: str | Path,
    output_path: str | Path,
    mode: str = "false_negative_with_hits",
    min_hits: int = 1,
    max_targets: int | None = None,
    max_hits_per_target: int = DEFAULT_MAX_HITS_PER_TARGET,
    record_indices_json: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, optionally manifest, and optionally register a handoff."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    source_path = Path(verified_records_jsonl)
    record_indices_path = None if record_indices_json is None else Path(record_indices_json)
    output = Path(output_path)
    payload = build_retrieval_semantic_gap_handoff(
        load_verified_records_jsonl(source_path),
        mode=mode,
        min_hits=min_hits,
        max_targets=max_targets,
        max_hits_per_target=max_hits_per_target,
        record_indices=None if record_indices_path is None else _load_record_indices_json(record_indices_path),
        metadata=metadata,
    )
    payload["source"] = {
        "verified_records_jsonl": str(source_path),
        "record_indices_json": None if record_indices_path is None else str(record_indices_path),
    }
    if artifact_manifest_path is not None:
        payload["paths"] = {"artifact_manifest": str(artifact_manifest_path)}
    _write_json(output, payload, compact=compact_json)
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            {
                "retrieval_semantic_gap_handoff": output,
                "verified_records_jsonl": source_path,
            },
            root=manifest_path.parent,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "candidate_count": payload["summary"]["candidate_count"],
                "fact_candidate_count": payload["summary"]["fact_candidate_count"],
                "total_request_count": payload["summary"]["total_request_count"],
                "mode": mode,
                "record_indices_json": None if record_indices_path is None else str(record_indices_path),
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
                "workflow": WORKFLOW,
                "status": payload["status"],
                "candidate_count": payload["summary"]["candidate_count"],
                "fact_candidate_count": payload["summary"]["fact_candidate_count"],
                "total_request_count": payload["summary"]["total_request_count"],
                "mode": mode,
                "record_indices_json": None if record_indices_path is None else str(record_indices_path),
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    print(
        f"{WORKFLOW}_ok candidates={payload['summary']['candidate_count']} "
        f"requests={payload['summary']['total_request_count']} output={output}"
    )
    return payload


def _target_payload(
    item: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    source_ordinal: int,
    max_hits_per_target: int,
    label: str,
    final_status: str,
    decision_rule: str,
    selected_route: str,
    hits: Sequence[Mapping[str, Any]],
    mode: str,
) -> dict[str, Any]:
    claim = _mapping(record.get("claim"))
    claim_text = str(claim.get("text", ""))
    claim_metadata = _mapping(claim.get("metadata"))
    record_metadata = _mapping(record.get("metadata"))
    statement = _mapping(record_metadata.get("statement"))
    alignment_record = _alignment_record(_mapping(record.get("final")))
    source_binding = _source_binding(record)
    features = _merged_features(claim_text, claim_metadata)
    issue_codes = tuple(str(item) for item in _sequence(alignment_record.get("issue_codes")))
    lanes = _recommended_lanes(
        claim_text=claim_text,
        features=features,
        issue_codes=issue_codes,
        question_type=str(statement.get("question_type", claim_metadata.get("question_type", ""))),
        hits=hits,
        source_binding=source_binding,
        decision_rule=decision_rule,
    )
    target_id = _target_id(item, source_ordinal)
    return {
        "target_id": target_id,
        "record_index": item.get("record_index"),
        "run": item.get("run"),
        "score_path": item.get("score_path"),
        "signal": item.get("signal"),
        "score": item.get("score"),
        "label": label,
        "mode": mode,
        "gap_reason": _gap_reason(label=label, final_status=final_status, decision_rule=decision_rule),
        "claim": {
            "text": claim_text,
            "claim_id": claim.get("claim_id"),
            "features": features,
            "entity_candidates": tuple(claim_entity_candidates(claim_text, max_items=6)),
        },
        "statement": {
            "question": statement.get("question"),
            "model_answer": statement.get("answer"),
            "question_type": statement.get("question_type"),
        },
        "route": {
            "selected_route": selected_route,
            "selected_verifier": _mapping(record.get("route")).get("selected_verifier"),
            "final_status": final_status,
            "decision_rule": decision_rule,
            "attempted_routes": tuple(
                str(item)
                for item in _sequence(_mapping(record.get("route")).get("attempted_routes"))
            ),
        },
        "alignment": {
            "issue_codes": issue_codes,
            "keyword_overlap": alignment_record.get("keyword_overlap"),
            "number_recall": alignment_record.get("number_recall"),
            "entity_recall": alignment_record.get("entity_recall"),
            "missing_numbers": tuple(str(item) for item in _sequence(alignment_record.get("missing_numbers"))),
            "missing_entities": tuple(str(item) for item in _sequence(alignment_record.get("missing_entities"))),
            "claim_numbers": tuple(str(item) for item in _sequence(alignment_record.get("claim_numbers"))),
            "evidence_numbers": tuple(str(item) for item in _sequence(alignment_record.get("evidence_numbers"))),
            "claim_entities": tuple(str(item) for item in _sequence(alignment_record.get("claim_entities"))),
            "evidence_entities": tuple(str(item) for item in _sequence(alignment_record.get("evidence_entities"))),
        },
        "retrieval": {
            "hit_count": len(hits),
            "source_binding": source_binding,
            "top_hits": tuple(_hit_summary(hit) for hit in hits[:max_hits_per_target]),
        },
        "recommended_lanes": lanes,
        "routing_notes": _routing_notes(lanes, issue_codes=issue_codes, source_binding=source_binding),
    }


def _requests_for_target(target: Mapping[str, Any]) -> dict[str, dict[str, Any] | None]:
    lanes = set(str(item) for item in _sequence(target.get("recommended_lanes")))
    target_id = str(target["target_id"])
    base = {
        "target_id": target_id,
        "record_index": target.get("record_index"),
        "priority": "high" if target.get("label") == "false" else "medium",
        "claim": _mapping(target.get("claim")).get("text"),
        "claim_id": _mapping(target.get("claim")).get("claim_id"),
        "question": _mapping(target.get("statement")).get("question"),
        "model_answer": _mapping(target.get("statement")).get("model_answer"),
        "final_status": _mapping(target.get("route")).get("final_status"),
        "decision_rule": _mapping(target.get("route")).get("decision_rule"),
        "gap_reason": target.get("gap_reason"),
        "top_hits": tuple(_sequence(_mapping(target.get("retrieval")).get("top_hits"))),
        "usage": "handoff_only_not_verifier_evidence",
    }
    return {
        "claim_evidence_alignment_review": {
            **base,
            "request_id": f"align:{target_id}:1",
            "request_type": "claim_evidence_alignment_review",
            "instruction": (
                "Extract the subject, predicate, proposed answer, contradictory evidence span, "
                "and unresolved slot from the top hits before changing verifier thresholds."
            ),
        },
        "structured_fact_candidate": (
            None
            if "structured_fact_candidate" not in lanes
            else {
                **base,
                "request_id": f"fact:{target_id}:1",
                "request_type": "structured_fact_candidate",
                "claim_numbers": tuple(_sequence(_mapping(target.get("alignment")).get("claim_numbers"))),
                "evidence_numbers": tuple(_sequence(_mapping(target.get("alignment")).get("evidence_numbers"))),
                "claim_entities": tuple(_sequence(_mapping(target.get("alignment")).get("claim_entities"))),
                "evidence_entities": tuple(_sequence(_mapping(target.get("alignment")).get("evidence_entities"))),
                "instruction": (
                    "Normalize source-backed evidence into a structured fact candidate with subject, "
                    "property, true value, false answer value, source id, and timestamp when available."
                ),
            }
        ),
        "world_model_rule_candidate": (
            None
            if "world_model_rule_candidate" not in lanes
            else {
                **base,
                "request_id": f"rule:{target_id}:1",
                "request_type": "world_model_rule_candidate",
                "features": _mapping(_mapping(target.get("claim")).get("features")),
                "instruction": (
                    "Author or route to a deterministic world-model/calculator rule only if the "
                    "evidence expresses a stable transition, quantity, temporal constraint, or procedure."
                ),
            }
        ),
        "retrieval_query_refinement": (
            None
            if "retrieval_query_refinement" not in lanes
            else {
                **base,
                "request_id": f"query:{target_id}:1",
                "request_type": "retrieval_query_refinement",
                "source_binding": _mapping(_mapping(target.get("retrieval")).get("source_binding")),
                "instruction": (
                    "Revise the retrieval query/source binding because available hits were fallback, "
                    "unbound, or lexically weak for the unresolved claim."
                ),
            }
        ),
    }


def _fact_candidates_for_target(target: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    lanes = set(str(item) for item in _sequence(target.get("recommended_lanes")))
    if "structured_fact_candidate" not in lanes:
        return ()
    target_id = str(target.get("target_id", "target"))
    retrieval = _mapping(target.get("retrieval"))
    candidates = []
    seen: set[tuple[str, str, str, str]] = set()
    for index, hit in enumerate(_sequence(retrieval.get("top_hits")), start=1):
        if not isinstance(hit, Mapping):
            continue
        subject = _fact_subject(target, hit)
        property_hint = _fact_property_hint(target, hit)
        value = _fact_value(target, hit)
        evidence_span = str(hit.get("text", "")).strip()
        evidence_source = str(hit.get("source", "")).strip()
        if not all((subject, property_hint, value, evidence_span, evidence_source)):
            continue
        key = (
            subject.casefold(),
            property_hint.casefold(),
            value.casefold(),
            evidence_source.casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        statement = _mapping(target.get("statement"))
        candidates.append({
            "candidate_id": f"fact:{target_id}:{index}",
            "request_id": f"fact:{target_id}:1",
            "target_id": target_id,
            "subject": subject,
            "requested_subject": _first_nonempty(_sequence(_mapping(target.get("claim")).get("entity_candidates"))),
            "matched_entity": subject,
            "property_hint": property_hint,
            "value": value,
            "model_answer": statement.get("model_answer"),
            "question": statement.get("question") or _mapping(target.get("claim")).get("text"),
            "evidence_span": evidence_span,
            "evidence_source": evidence_source,
            "source_family": _source_family_for_hit(hit),
            "provider": _provider_for_hit(hit),
            "confidence": _hit_confidence(hit),
            "usage": "structured_fact_review_only",
            "metadata": {
                "builder": WORKFLOW,
                "gap_reason": target.get("gap_reason"),
                "decision_rule": _mapping(target.get("route")).get("decision_rule"),
                **_structured_fact_metadata(hit),
            },
        })
        if len(candidates) >= 3:
            break
    return tuple(candidates)


def _recommended_lanes(
    *,
    claim_text: str,
    features: Mapping[str, Any],
    issue_codes: Sequence[str],
    question_type: str,
    hits: Sequence[Mapping[str, Any]],
    source_binding: Mapping[str, Any],
    decision_rule: str,
) -> tuple[str, ...]:
    lanes = ["claim_evidence_alignment_review"]
    codes = set(issue_codes)
    qtype = str(question_type).strip().lower()
    if _has_structured_hit(hits) or codes.intersection({"missing_claim_number", "missing_claim_entity"}):
        lanes.append("structured_fact_candidate")
    if any(
        flag_value_enabled(features.get(key))
        for key in ("has_number", "has_calculation", "is_time_sensitive")
    ) or qtype in {"quantity", "temporal", "method", "causal"}:
        lanes.append("world_model_rule_candidate")
    if (
        str(source_binding.get("mode", "")).startswith("fallback")
        or source_binding.get("requested") is False
        or decision_rule in {"low_overlap", "no_evidence"}
        or "low_keyword_overlap" in codes
    ):
        lanes.append("retrieval_query_refinement")
    if not lanes:
        lanes.append("claim_evidence_alignment_review")
    return tuple(dict.fromkeys(lanes))


def _summary(
    *,
    verified_records: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    requests: Mapping[str, Sequence[Mapping[str, Any]]],
    fact_candidates: Sequence[Mapping[str, Any]],
    source_route_counts: Counter[str],
    source_status_counts: Counter[str],
    source_decision_rule_counts: Counter[str],
    skipped_reason_counts: Counter[str],
) -> dict[str, Any]:
    evaluated_source_count = sum(source_status_counts.values())
    lane_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    target_route_counts: Counter[str] = Counter()
    target_decision_rule_counts: Counter[str] = Counter()
    source_binding_mode_counts: Counter[str] = Counter()
    hit_source_counts: Counter[str] = Counter()
    fact_candidate_property_counts: Counter[str] = Counter(
        str(item.get("property_hint"))
        for item in fact_candidates
        if item.get("property_hint")
    )
    for target in targets:
        label_counts[str(target.get("label", "unknown"))] += 1
        route = _mapping(target.get("route"))
        target_route_counts[str(route.get("selected_route", "unknown"))] += 1
        target_decision_rule_counts[str(route.get("decision_rule", "unknown"))] += 1
        retrieval = _mapping(target.get("retrieval"))
        source_binding = _mapping(retrieval.get("source_binding"))
        source_binding_mode_counts[str(source_binding.get("mode", "none"))] += 1
        for lane in _sequence(target.get("recommended_lanes")):
            lane_counts[str(lane)] += 1
        for hit in _sequence(retrieval.get("top_hits")):
            source = _mapping(hit).get("source")
            if source is not None:
                hit_source_counts[str(source)] += 1
    request_counts = {key: len(tuple(value)) for key, value in requests.items()}
    return {
        "source_record_count": len(verified_records),
        "evaluated_source_record_count": evaluated_source_count,
        "candidate_count": len(targets),
        "candidate_rate": _rate(len(targets), evaluated_source_count),
        "label_counts": _sorted_counter(label_counts),
        "target_selected_route_counts": _sorted_counter(target_route_counts),
        "target_decision_rule_counts": _sorted_counter(target_decision_rule_counts),
        "source_selected_route_counts": _sorted_counter(source_route_counts),
        "source_final_status_counts": _sorted_counter(source_status_counts),
        "source_decision_rule_counts": _sorted_counter(source_decision_rule_counts),
        "skipped_reason_counts": _sorted_counter(skipped_reason_counts),
        "recommended_lane_counts": _sorted_counter(lane_counts),
        "source_binding_mode_counts": _sorted_counter(source_binding_mode_counts),
        "top_hit_sources": _counter_top(hit_source_counts, limit=20),
        "fact_candidate_count": len(fact_candidates),
        "fact_candidate_property_counts": _sorted_counter(fact_candidate_property_counts),
        "request_counts": request_counts,
        "total_request_count": sum(request_counts.values()),
    }


def _matches_mode(
    *,
    mode: str,
    label: str,
    status: str,
    hit_count: int,
    min_hits: int,
) -> tuple[bool, str]:
    if hit_count < min_hits:
        return False, "too_few_hits"
    if mode == "all_with_hits":
        return True, ""
    if mode == "unresolved_with_hits":
        if status in UNRESOLVED_STATUSES:
            return True, ""
        return False, "resolved_status"
    if mode == "false_negative_with_hits":
        if label != "false":
            return False, "not_false_label"
        if status == REFUTED_STATUS:
            return False, "already_refuted"
        return True, ""
    raise ValueError(f"unsupported mode: {mode}")


def _record_payload(item: Mapping[str, Any]) -> Mapping[str, Any]:
    record = item.get("record", item)
    return record if isinstance(record, Mapping) else {}


def _label_key(item: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    raw = item.get("label")
    if raw is None:
        metadata = _mapping(record.get("metadata"))
        statement = _mapping(metadata.get("statement"))
        raw = statement.get("is_false", statement.get("label"))
    try:
        return "false" if int(raw) == 1 else "true"
    except (TypeError, ValueError):
        return "unknown"


def _selected_route(record: Mapping[str, Any]) -> str:
    return str(_mapping(record.get("route")).get("selected_route", "unknown"))


def _decision_rule(final: Mapping[str, Any]) -> str:
    return str(_mapping(final.get("metadata")).get("decision_rule", "unknown"))


def _retrieval_hits(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(_mapping(hit) for hit in _sequence(record.get("retrieval_hits")) if isinstance(hit, Mapping))


def _alignment_record(final: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _mapping(final.get("metadata"))
    alignment = _mapping(metadata.get("evidence_alignment"))
    records = _sequence(alignment.get("records"))
    first = records[0] if records else {}
    return first if isinstance(first, Mapping) else {}


def _source_binding(record: Mapping[str, Any]) -> dict[str, Any]:
    retrieval = _mapping(_mapping(record.get("metadata")).get("retrieval"))
    source_binding = _mapping(retrieval.get("source_binding"))
    if source_binding:
        return dict(source_binding)
    hit_bindings = []
    for hit in _retrieval_hits(record):
        metadata = _mapping(hit.get("metadata"))
        for key in ("source_queue_request_sha256", "source_request_sha256", "collection_request_sha256"):
            if metadata.get(key):
                hit_bindings.append(key)
    return {
        "requested": bool(hit_bindings),
        "mode": "hit_metadata" if hit_bindings else "none",
        "fallback": False,
        "hit_binding_key_fields": tuple(dict.fromkeys(hit_bindings)),
    }


def _merged_features(claim_text: str, claim_metadata: Mapping[str, Any]) -> dict[str, bool]:
    features = claim_features(claim_text)
    raw_features = _mapping(claim_metadata.get("features"))
    for key, value in raw_features.items():
        features[str(key)] = flag_value_enabled(value)
    return features


def _has_structured_hit(hits: Sequence[Mapping[str, Any]]) -> bool:
    for hit in hits:
        source = str(hit.get("source", "")).lower()
        if source.startswith(STRUCTURED_SOURCE_PREFIXES):
            return True
        metadata = _mapping(hit.get("metadata"))
        if any(metadata.get(key) not in (None, "") for key in STRUCTURED_METADATA_KEYS):
            return True
    return False


def _fact_subject(target: Mapping[str, Any], hit: Mapping[str, Any]) -> str:
    metadata = _mapping(hit.get("metadata"))
    metadata_subject = _clean_text(_first_nonempty((
        metadata.get("country_name"),
        metadata.get("subject_label"),
        metadata.get("subject"),
        metadata.get("entity_name"),
        metadata.get("entity"),
        metadata.get("organization_name"),
        metadata.get("location_name"),
    )))
    if metadata_subject:
        return metadata_subject
    parsed_description_subject = _extract_description_subject(str(hit.get("text", "")))
    if parsed_description_subject and _matches_target_entity(parsed_description_subject, target):
        return parsed_description_subject
    parsed_structured_subject = _structured_fact_part(hit, "subject")
    if parsed_structured_subject and _matches_target_entity(parsed_structured_subject, target):
        return parsed_structured_subject
    return _clean_text(_first_nonempty(_target_entity_candidates(target)))


def _fact_property_hint(target: Mapping[str, Any], hit: Mapping[str, Any]) -> str:
    metadata = _mapping(hit.get("metadata"))
    property_id = _clean_text(_first_nonempty((
        metadata.get("statement_property"),
        metadata.get("property_id"),
        metadata.get("property"),
        metadata.get("indicator"),
    )))
    property_label = _clean_text(_first_nonempty((
        metadata.get("indicator_name"),
        metadata.get("statement_property_label"),
        metadata.get("property_label"),
    )))
    if property_label and property_id:
        return f"{property_label}:{property_id}"
    if property_label:
        return property_label
    if property_id:
        return property_id
    source_property = _property_hint_from_source(hit)
    if source_property:
        return source_property
    structured_property = _structured_property_hint(hit)
    if structured_property:
        return structured_property
    if DESCRIBED_AS_RE.search(str(hit.get("text", ""))):
        return "description"
    question_type = _clean_text(_mapping(target.get("statement")).get("question_type"))
    if question_type in {"quantity", "temporal", "location", "person"}:
        return question_type
    return ""


def _fact_value(target: Mapping[str, Any], hit: Mapping[str, Any]) -> str:
    metadata = _mapping(hit.get("metadata"))
    alignment = _mapping(target.get("alignment"))
    claim_numbers = {_normalize_slot_value(item) for item in _sequence(alignment.get("claim_numbers"))}
    model_answer = _normalize_slot_value(_mapping(target.get("statement")).get("model_answer"))
    structured_value = _structured_fact_part(hit, "value")
    if structured_value:
        return structured_value
    for value in _sequence(alignment.get("evidence_numbers")):
        text = _clean_text(value)
        normalized = _normalize_slot_value(text)
        if text and normalized not in claim_numbers and normalized != model_answer:
            return text
    for value in _sequence(alignment.get("evidence_numbers")):
        text = _clean_text(value)
        if text:
            return text
    described_value = _extract_description_value(str(hit.get("text", "")))
    if described_value:
        return described_value
    return _clean_text(_first_nonempty((
        metadata.get("object_label"),
        metadata.get("value"),
        metadata.get("answer"),
        metadata.get("indicator_value"),
    )))


def _source_family_for_hit(hit: Mapping[str, Any]) -> str:
    metadata = _mapping(hit.get("metadata"))
    source_family = _clean_text(metadata.get("source_family"))
    if source_family:
        return source_family
    source = str(hit.get("source", "")).casefold()
    if source.startswith("worldbank:"):
        return "official_statistics"
    if source.startswith("wikidata:") or source.startswith("wikipedia:"):
        return "reference"
    if source.startswith("openalex:") or source.startswith("crossref:"):
        return "scholarly"
    if source.startswith("official:"):
        return "official"
    if source.startswith("news:") or source.startswith("gdelt:"):
        return "news"
    return "unknown"


def _provider_for_hit(hit: Mapping[str, Any]) -> str:
    metadata = _mapping(hit.get("metadata"))
    provider = _clean_text(metadata.get("provider"))
    if provider:
        return provider
    source = str(hit.get("source", "")).split(":", 1)[0].strip()
    return source or "unknown"


def _property_hint_from_source(hit: Mapping[str, Any]) -> str:
    source = str(hit.get("source", "")).strip()
    tail = source.rsplit(":", 1)[-1].strip()
    if tail == "description":
        return "description"
    return ""


def _structured_property_hint(hit: Mapping[str, Any]) -> str:
    property_label = _structured_fact_part(hit, "property")
    property_id = _source_property_id(hit)
    if property_label and property_id:
        return f"{property_label}:{property_id}"
    if property_label:
        return property_label
    return property_id


def _structured_fact_metadata(hit: Mapping[str, Any]) -> dict[str, str]:
    reference_time = _structured_fact_part(hit, "year")
    if reference_time:
        return {"reference_time": reference_time}
    return {}


def _structured_fact_part(hit: Mapping[str, Any], key: str) -> str:
    return _clean_text(_structured_fact_parts(hit).get(key))


def _structured_fact_parts(hit: Mapping[str, Any]) -> dict[str, str]:
    source = str(hit.get("source", "")).strip()
    source_key = source.casefold()
    text = str(hit.get("text", ""))
    metadata = _mapping(hit.get("metadata"))
    if source_key.startswith("worldbank:") or WORLDBANK_STAT_RE.search(text):
        match = WORLDBANK_STAT_RE.search(text)
        if match is not None:
            return {
                "subject": _clean_structured_part(match.group("subject")),
                "property": _clean_structured_part(match.group("property")),
                "value": _clean_structured_part(match.group("value")),
                "year": _clean_structured_part(match.group("year")),
            }
        return {
            "subject": _clean_structured_part(metadata.get("country_name")),
            "property": _clean_structured_part(metadata.get("indicator_name")),
            "value": _clean_structured_part(metadata.get("indicator_value")),
            "year": _clean_structured_part(metadata.get("year") or metadata.get("date")),
        }
    if source_key.startswith("wikidata:") or WIKIDATA_STRUCTURED_RE.search(text):
        subject = _clean_structured_part(_first_nonempty((
            metadata.get("subject_label"),
            metadata.get("subject"),
            metadata.get("entity_name"),
            metadata.get("entity"),
        )))
        property_label = _clean_structured_part(_first_nonempty((
            metadata.get("statement_property_label"),
            metadata.get("property_label"),
        )))
        value = _clean_structured_part(_first_nonempty((
            metadata.get("object_label"),
            metadata.get("value"),
            metadata.get("answer"),
        )))
        if subject and property_label and value:
            return {"subject": subject, "property": property_label, "value": value}
        match = WIKIDATA_STRUCTURED_RE.search(text)
        if match is None:
            return {}
        parsed_property, parsed_value = _split_wikidata_relation(
            _clean_structured_part(match.group("relation")),
            _source_property_id(hit),
        )
        return {
            "subject": subject or _clean_structured_part(match.group("subject")),
            "property": property_label or parsed_property,
            "value": value or parsed_value,
        }
    return {}


def _split_wikidata_relation(relation: str, property_id: str) -> tuple[str, str]:
    relation = _clean_structured_part(relation)
    property_label = WIKIDATA_PROPERTY_LABEL_BY_ID.get(property_id, "")
    if property_label:
        prefix = property_label.casefold()
        relation_key = relation.casefold()
        if relation_key == prefix:
            return property_label, ""
        if relation_key.startswith(prefix + " "):
            return property_label, _clean_structured_part(relation[len(property_label):])
    parts = relation.split()
    if len(parts) < 2:
        return relation, ""
    return parts[0], " ".join(parts[1:])


def _source_property_id(hit: Mapping[str, Any]) -> str:
    source = str(hit.get("source", "")).strip()
    parts = source.split(":")
    if len(parts) >= 2 and parts[0].casefold() == "worldbank":
        return parts[1].strip()
    if len(parts) >= 3 and parts[0].casefold() == "wikidata":
        property_id = parts[2].strip()
        if re.fullmatch(r"P\d+", property_id):
            return property_id
    return ""


def _clean_structured_part(value: Any) -> str:
    return re.sub(r"\s+", " ", _clean_text(value)).strip(" \t\r\n.,;:!?\"'")


def _extract_description_value(text: str) -> str:
    match = DESCRIBED_AS_RE.search(text)
    if match is None:
        return ""
    return match.group("value").strip()


def _extract_description_subject(text: str) -> str:
    match = DESCRIBED_SUBJECT_RE.search(text)
    if match is None:
        return ""
    return _clean_text(match.group("subject")).strip(" \t\r\n.,;:!?\"'")


def _matches_target_entity(subject: str, target: Mapping[str, Any]) -> bool:
    normalized_subject = _entity_match_key(subject)
    if not normalized_subject:
        return False
    subject_acronym = _subject_acronym(subject)
    for candidate in _target_entity_candidates(target):
        normalized_candidate = _entity_match_key(candidate)
        if not normalized_candidate:
            continue
        if normalized_subject == normalized_candidate:
            return True
        if subject_acronym and normalized_candidate.removesuffix("s") == subject_acronym:
            return True
    return False


def _target_entity_candidates(target: Mapping[str, Any]) -> tuple[str, ...]:
    values = (
        *_sequence(_mapping(target.get("alignment")).get("claim_entities")),
        *_sequence(_mapping(target.get("alignment")).get("evidence_entities")),
        *_sequence(_mapping(target.get("claim")).get("entity_candidates")),
    )
    return tuple(
        dict.fromkeys(
            text
            for value in values
            if (text := _clean_text(value)) and _entity_match_key(text)
        )
    )


def _entity_match_key(value: str) -> str:
    tokens = tuple(match.group(0).casefold() for match in re.finditer(r"[A-Za-z0-9]+", value))
    filtered = tuple(token for token in tokens if token not in GENERIC_ENTITY_CANDIDATES)
    return " ".join(filtered)


def _subject_acronym(value: str) -> str:
    tokens = tuple(match.group(0) for match in re.finditer(r"[A-Za-z]+", value))
    if len(tokens) < 2:
        return ""
    return "".join(token[0].casefold() for token in tokens)


def _hit_confidence(hit: Mapping[str, Any]) -> float:
    try:
        value = float(hit.get("score", 0.5))
    except (TypeError, ValueError):
        return 0.5
    if value != value:
        return 0.5
    return max(0.0, min(1.0, value))


def _hit_summary(hit: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _mapping(hit.get("metadata"))
    copied_metadata = {
        key: metadata[key]
        for key in (
            "source_family",
            "statement_property",
            "statement_property_label",
            "property_id",
            "property_label",
            "country_name",
            "indicator_name",
            "subject_label",
            "subject",
            "entity_name",
            "entity",
            "organization_name",
            "location_name",
            "object_label",
            "value",
            "answer",
            "indicator_value",
            "provider",
            "source_queue_request_sha256",
            "source_request_sha256",
            "collection_request_sha256",
        )
        if key in metadata
    }
    return {
        "text": _truncate(str(hit.get("text", "")), 400),
        "source": hit.get("source"),
        "score": hit.get("score"),
        "metadata": copied_metadata,
    }


def _first_nonempty(values: Sequence[Any]) -> Any:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return value
    return None


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalize_slot_value(value: Any) -> str:
    return _clean_text(value).replace(",", "").casefold()


def _gap_reason(
    *,
    label: str,
    final_status: str,
    decision_rule: str,
) -> str:
    if label == "false" and final_status != REFUTED_STATUS:
        if decision_rule == "low_overlap":
            return "semantic_false_negative_low_overlap"
        return "semantic_false_negative_with_hits"
    if final_status in UNRESOLVED_STATUSES:
        return "unresolved_with_hits"
    return "retrieval_hit_review"


def _routing_notes(
    lanes: Sequence[str],
    *,
    issue_codes: Sequence[str],
    source_binding: Mapping[str, Any],
) -> tuple[str, ...]:
    notes = []
    if "structured_fact_candidate" in lanes:
        notes.append("source-backed slots look structured enough for fact normalization")
    if "world_model_rule_candidate" in lanes:
        notes.append("claim features suggest deterministic rule or world-model checking may be useful")
    if "retrieval_query_refinement" in lanes:
        notes.append("retrieval/source binding is weak, fallback, or lexically low-overlap")
    if issue_codes:
        notes.append("alignment issue codes: " + ", ".join(issue_codes))
    if source_binding:
        notes.append("source binding mode: " + str(source_binding.get("mode", "none")))
    return tuple(notes)


def _target_id(item: Mapping[str, Any], source_ordinal: int) -> str:
    record_index = item.get("record_index")
    try:
        return f"record-{int(record_index)}"
    except (TypeError, ValueError):
        return f"record-{source_ordinal - 1}"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _counter_top(counter: Counter[str], *, limit: int) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _truncate(text: str, limit: int) -> str:
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata = {}
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source-backed retrieval semantic-gap handoff queues.")
    parser.add_argument("--verified-records-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=MODES, default="false_negative_with_hits")
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--max-hits-per-target", type=int, default=DEFAULT_MAX_HITS_PER_TARGET)
    parser.add_argument("--record-indices-json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=None)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args()
    run(
        verified_records_jsonl=args.verified_records_jsonl,
        output_path=args.output,
        mode=args.mode,
        min_hits=args.min_hits,
        max_targets=args.max_targets,
        max_hits_per_target=args.max_hits_per_target,
        record_indices_json=args.record_indices_json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )


if __name__ == "__main__":
    main()
