"""Plan review-gated entity-role bindings for blocked rule inputs.

This planner consumes editable ``source_backed_entity_bindings`` sidecars and
optional source-alignment records. It can draft candidate entity binding rows
from already-collected alignment evidence, but it never approves them, never
executes fill commands, and never treats candidates as verifier evidence.
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

WORKFLOW = "world_model_rule_entity_binding_plan"
COLLECTION_FAMILY = "entity_role_rule_input_collection"
INPUT_BINDING_COLLECTION_FAMILY = "entity_role_rule_input_binding_collection"
RESERVED_FIELDS = {"answer", "answers", "is_false", "label", "labels", "model_answer", "score_label"}
REQUIRED_BINDING_FIELDS = (
    "request_id",
    "target_id",
    "subject_entity",
    "answer_entity",
    "expected_entity",
    "requested_role",
    "source_citation",
    "review_status",
    "not_verifier_evidence",
)
CAPITALIZED_STOPWORDS = {
    "CBS Sports",
    "How",
    "National Football League",
    "Other",
    "Source",
    "World Bank",
    "OpenAlex",
    "Wikidata",
}


def build_world_model_rule_entity_binding_plan(
    *,
    entity_bindings: Sequence[Mapping[str, Any]],
    alignment_records: Sequence[Mapping[str, Any]] = (),
    min_alignment_score: float = 0.45,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready entity-role binding collection plan."""
    if min_alignment_score < 0.0:
        raise ValueError("min_alignment_score must be non-negative.")
    alignments_by_target = _alignment_by_target(alignment_records)
    requests: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_request_ids: set[str] = set()
    duplicate_request_ids: set[str] = set()
    for binding in entity_bindings:
        row = _sanitize(binding)
        if not _is_entity_binding(row):
            skipped.append(_skipped_binding(row, reason="not_entity_binding"))
            continue
        request_id = _request_id(row)
        if request_id in seen_request_ids:
            duplicate_request_ids.add(request_id)
        seen_request_ids.add(request_id)

    emitted_request_ids: set[str] = set()
    for binding in entity_bindings:
        row = _sanitize(binding)
        if not _is_entity_binding(row):
            continue
        request_id = _request_id(row)
        if request_id in emitted_request_ids:
            skipped.append(_skipped_binding(row, reason="duplicate_entity_binding"))
            continue
        emitted_request_ids.add(request_id)
        alignment = alignments_by_target.get(str(row.get("target_id") or ""))
        plan_request, candidate = _entity_binding_request(
            row,
            alignment=alignment,
            duplicate=request_id in duplicate_request_ids,
            min_alignment_score=min_alignment_score,
        )
        requests.append(plan_request)
        if candidate is not None:
            candidates.append(candidate)

    summary = _summary(
        entity_bindings=entity_bindings,
        alignment_records=alignment_records,
        requests=requests,
        candidates=candidates,
        skipped=skipped,
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": _status(summary),
        "scope": (
            "Plans review-gated entity-role sidecar bindings for world-model "
            "rule inputs. Candidate bindings are draft collection artifacts, "
            "not verifier evidence, and remain blocked until separately reviewed."
        ),
        "label_usage": {
            "labels_used_for_collection_planning": False,
            "labels_copied_to_entity_binding_candidates": False,
            "model_answers_used_as_candidate_answer_entities": True,
            "candidate_bindings_are_verifier_evidence": False,
            "planner_approves_entity_bindings": False,
            "planner_executes_fill_commands": False,
        },
        "config": {
            "collection_family": COLLECTION_FAMILY,
            "required_binding_fields": REQUIRED_BINDING_FIELDS,
            "min_alignment_score": float(min_alignment_score),
            "candidate_review_status": "needs_review",
        },
        "summary": summary,
        "entity_binding_requests": tuple(requests),
        "candidate_entity_bindings": tuple(candidates),
        "skipped_bindings": tuple(skipped),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    entity_bindings_path: str | Path,
    output_dir: str | Path,
    alignment_records_paths: Sequence[str | Path] = (),
    report_json_path: str | Path | None = None,
    requests_path: str | Path | None = None,
    candidate_bindings_path: str | Path | None = None,
    skipped_bindings_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    min_alignment_score: float = 0.45,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register an entity binding plan."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "entity-binding-plan.json")
    requests_rows_path = Path(requests_path or output / "entity-binding-requests.jsonl")
    candidate_rows_path = Path(candidate_bindings_path or output / "candidate-entity-bindings.jsonl")
    skipped_rows_path = Path(skipped_bindings_path or output / "skipped-entity-bindings.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")
    entity_bindings = _load_jsonl_mappings(entity_bindings_path)
    alignment_records = tuple(
        row
        for path in alignment_records_paths
        for row in _load_jsonl_mappings(path, sanitize=False)
    )
    payload = build_world_model_rule_entity_binding_plan(
        entity_bindings=entity_bindings,
        alignment_records=alignment_records,
        min_alignment_score=min_alignment_score,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["paths"] = {
        "entity_bindings": str(entity_bindings_path),
        "alignment_records": tuple(str(path) for path in alignment_records_paths),
        "report": str(report_path),
        "entity_binding_requests": str(requests_rows_path),
        "candidate_entity_bindings": str(candidate_rows_path),
        "skipped_bindings": str(skipped_rows_path),
        "artifact_manifest": str(manifest_path),
    }
    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(requests_rows_path, payload["entity_binding_requests"], compact=compact_json)
    _write_jsonl(candidate_rows_path, payload["candidate_entity_bindings"], compact=compact_json)
    _write_jsonl(skipped_rows_path, payload["skipped_bindings"], compact=compact_json)
    manifest_artifacts: dict[str, str | Path | None] = {
        "entity_binding_plan": report_path,
        "entity_binding_requests": requests_rows_path,
        "candidate_entity_bindings": candidate_rows_path,
        "skipped_entity_bindings": skipped_rows_path,
        "entity_bindings": Path(entity_bindings_path),
    }
    for index, path in enumerate(alignment_records_paths, start=1):
        manifest_artifacts[f"alignment_records_{index}"] = Path(path)
    manifest = build_artifact_manifest(
        manifest_artifacts,
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "request_count": payload["summary"]["request_count"],
            "candidate_count": payload["summary"]["candidate_count"],
            "ready_for_review_candidate_count": payload["summary"][
                "ready_for_review_candidate_count"
            ],
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
                "request_count": payload["summary"]["request_count"],
                "candidate_count": payload["summary"]["candidate_count"],
                "ready_for_review_candidate_count": payload["summary"][
                    "ready_for_review_candidate_count"
                ],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _entity_binding_request(
    row: Mapping[str, Any],
    *,
    alignment: Mapping[str, Any] | None,
    duplicate: bool,
    min_alignment_score: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    failures = list(_binding_failures(row, duplicate=duplicate))
    candidate = _candidate_binding(
        row,
        alignment=alignment,
        min_alignment_score=min_alignment_score,
    )
    candidate_missing_fields = (
        tuple(REQUIRED_BINDING_FIELDS)
        if candidate is None
        else _missing_required_candidate_fields(candidate)
    )
    request = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "request_id": f"entity-binding:{_request_id(row) or _clean(row.get('binding_id'))}",
        "source_request_id": _request_id(row),
        "target_id": _clean(row.get("target_id")),
        "task_id": _clean(row.get("task_id")),
        "collection_family": COLLECTION_FAMILY,
        "question": _clean(row.get("question")),
        "entity_binding_id": _clean(row.get("binding_id")),
        "entity_binding_review_status": _clean(row.get("review_status")).lower(),
        "required_entity_binding_fields": REQUIRED_BINDING_FIELDS,
        "blocking_failures": tuple(failures),
        "alignment_context": _alignment_context(alignment),
        "candidate_status": "" if candidate is None else str(candidate["candidate_status"]),
        "candidate_missing_fields": candidate_missing_fields,
        "candidate_binding_id": "" if candidate is None else str(candidate["binding_id"]),
        "not_verifier_evidence": True,
    }
    return request, candidate


def _candidate_binding(
    row: Mapping[str, Any],
    *,
    alignment: Mapping[str, Any] | None,
    min_alignment_score: float,
) -> dict[str, Any] | None:
    if alignment is None:
        return _candidate_from_row(row, status="needs_alignment_evidence")
    best_hit = _best_alignment_hit(alignment, min_alignment_score=min_alignment_score)
    answer_entity = _extract_answer_entity(alignment.get("model_answer") or row.get("answer_entity"))
    subject_entity = _subject_entity_candidate(
        row,
        alignment=alignment,
        answer_entity=answer_entity,
        hit=best_hit,
    )
    requested_role = _requested_role_candidate(row, alignment=alignment)
    source_citation = "" if best_hit is None else _clean(best_hit.get("source"))
    expected_entity = ""
    expected_source = ""
    if best_hit is not None and best_hit.get("model_answer_value_matched") is True and answer_entity:
        expected_entity = answer_entity
        expected_source = "alignment_model_answer_value_matched"
    candidate = {
        **_candidate_from_row(row, status=""),
        "subject_entity": subject_entity,
        "answer_entity": answer_entity,
        "expected_entity": expected_entity,
        "requested_role": requested_role,
        "source_citation": source_citation,
        "source_url": "" if best_hit is None else _clean(best_hit.get("url")),
        "source_title": "" if best_hit is None else _clean(best_hit.get("title")),
        "source_family": "" if best_hit is None else _clean(best_hit.get("source_family")),
        "provider": "" if best_hit is None else _clean(best_hit.get("provider")),
        "candidate_answer_source": "alignment_model_answer" if answer_entity else "",
        "expected_entity_source": expected_source,
        "candidate_expected_entities": _candidate_expected_entities(
            alignment=alignment,
            hit=best_hit,
            answer_entity=answer_entity,
            subject_entity=subject_entity,
        ),
        "candidate_alignment": _candidate_alignment_summary(alignment, best_hit),
        "source_note": _source_note(alignment, best_hit),
    }
    missing = _missing_required_candidate_fields(candidate)
    if not missing:
        candidate["candidate_status"] = "ready_for_review"
    elif source_citation:
        candidate["candidate_status"] = "needs_entity_value_review"
    else:
        candidate["candidate_status"] = "needs_source_evidence"
    return candidate


def _candidate_from_row(row: Mapping[str, Any], *, status: str) -> dict[str, Any]:
    request_id = _request_id(row)
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "binding_id": f"candidate:{_clean(row.get('binding_id')) or request_id}",
        "request_id": request_id,
        "source_request_id": request_id,
        "target_id": _clean(row.get("target_id")),
        "task_id": _clean(row.get("task_id")),
        "collection_family": COLLECTION_FAMILY,
        "question": _clean(row.get("question")),
        "rule_family": _clean(row.get("rule_family")) or "entity_disambiguation",
        "subject_entity": _clean(row.get("subject_entity")),
        "answer_entity": _clean(row.get("answer_entity")),
        "expected_entity": _clean(row.get("expected_entity")),
        "requested_role": _clean(row.get("requested_role")),
        "source_citation": _clean(row.get("source_citation")),
        "review_status": "needs_review",
        "not_verifier_evidence": True,
        "candidate_results_require_review": True,
        "candidate_status": status,
    }


def _binding_failures(row: Mapping[str, Any], *, duplicate: bool) -> tuple[str, ...]:
    failures: list[str] = []
    if duplicate:
        failures.append("duplicate_entity_binding")
    if not _request_id(row):
        failures.append("missing_request_id")
    if not _clean(row.get("target_id")):
        failures.append("missing_target_id")
    if row.get("not_verifier_evidence") is not True:
        failures.append("binding_not_marked_non_evidence")
    if _clean(row.get("review_status")).lower() not in {"", "ready", "approved"}:
        failures.append("binding_requires_review")
    reserved = tuple(sorted(key for key in row if str(key) in RESERVED_FIELDS))
    if reserved:
        failures.append("reserved_fields_present")
    for key in ("subject_entity", "answer_entity", "expected_entity", "requested_role", "source_citation"):
        if not _clean(row.get(key)):
            failures.append(f"missing_{key}")
    return tuple(dict.fromkeys(failures))


def _missing_required_candidate_fields(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        key
        for key in (
            "request_id",
            "target_id",
            "subject_entity",
            "answer_entity",
            "expected_entity",
            "requested_role",
            "source_citation",
        )
        if not _clean(candidate.get(key))
    )


def _alignment_by_target(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    selected: dict[str, Mapping[str, Any]] = {}
    for record in records:
        target_id = _clean(record.get("target_id"))
        if not target_id:
            continue
        existing = selected.get(target_id)
        if existing is None or _best_hit_score(record) > _best_hit_score(existing):
            selected[target_id] = record
    return selected


def _best_alignment_hit(
    alignment: Mapping[str, Any],
    *,
    min_alignment_score: float,
) -> Mapping[str, Any] | None:
    hits = tuple(_mapping_sequence(alignment.get("top_evidence_hits", ())))
    usable = tuple(
        hit
        for hit in hits
        if _float_or_zero(hit.get("alignment_score")) >= min_alignment_score
        and _clean(hit.get("source"))
    )
    if not usable:
        return None
    return max(
        usable,
        key=lambda hit: (
            hit.get("model_answer_value_matched") is True,
            _float_or_zero(hit.get("alignment_score")),
            _float_or_zero(hit.get("query_overlap")),
        ),
    )


def _best_hit_score(alignment: Mapping[str, Any]) -> float:
    hits = tuple(_mapping_sequence(alignment.get("top_evidence_hits", ())))
    return max((_float_or_zero(hit.get("alignment_score")) for hit in hits), default=0.0)


def _subject_entity_candidate(
    row: Mapping[str, Any],
    *,
    alignment: Mapping[str, Any],
    answer_entity: str,
    hit: Mapping[str, Any] | None,
) -> str:
    existing = _clean(row.get("subject_entity"))
    if existing:
        return existing
    matched = "" if hit is None else _clean(hit.get("matched_entity"))
    if matched:
        return matched
    answer_key = _entity_key(answer_entity)
    for candidate in _string_tuple(alignment.get("entity_candidates", ())):
        clean = _extract_answer_entity(candidate)
        if not clean:
            continue
        if _entity_key(clean) == answer_key:
            continue
        if clean.lower().startswith(("you will", "his name", "this american")):
            continue
        return clean
    return ""


def _requested_role_candidate(row: Mapping[str, Any], *, alignment: Mapping[str, Any]) -> str:
    existing = _clean(row.get("requested_role"))
    if existing:
        return existing
    question = _clean(row.get("question") or alignment.get("question")).lower()
    if "name the team" in question or "name of the team" in question:
        return "team_name"
    if "physically travel" in question or "will you see" in question:
        return "physical_location_expected_entity"
    if "his name is" in question or "name is" in question:
        return "name_completion"
    if question.startswith("who "):
        return "person_identity"
    return "entity_role"


def _extract_answer_entity(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    text = re.sub(r"^(you will see|his name is|the answer is)\s+", "", text, flags=re.IGNORECASE)
    text = text.strip(" .!?\"'")
    return text


def _candidate_expected_entities(
    *,
    alignment: Mapping[str, Any],
    hit: Mapping[str, Any] | None,
    answer_entity: str,
    subject_entity: str,
) -> tuple[str, ...]:
    candidates: list[str] = []
    hit_text = "" if hit is None else " ".join((
        _clean(hit.get("title")),
        _clean(hit.get("evidence_span")),
        _clean(hit.get("text")),
        _clean(hit.get("snippet")),
    ))
    if answer_entity and _entity_key(answer_entity) in _entity_key(hit_text):
        candidates.append(answer_entity)
    for candidate in _string_tuple(alignment.get("entity_candidates", ())):
        clean = _extract_answer_entity(candidate)
        if clean and _entity_key(clean) in _entity_key(hit_text):
            candidates.append(clean)
    for clean in _capitalized_phrases(hit_text):
        if _entity_key(clean) == _entity_key(subject_entity):
            continue
        candidates.append(clean)
    return tuple(dict.fromkeys(item for item in candidates if item))


def _capitalized_phrases(text: str) -> tuple[str, ...]:
    phrases = []
    for match in re.finditer(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,})){0,3}\b", text):
        phrase = match.group(0).strip()
        if phrase in CAPITALIZED_STOPWORDS:
            continue
        if len(phrase) < 3:
            continue
        phrases.append(phrase)
    return tuple(dict.fromkeys(phrases))


def _candidate_alignment_summary(
    alignment: Mapping[str, Any],
    hit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "alignment_status": _clean(alignment.get("alignment_status")),
        "gap_reason": _clean(alignment.get("gap_reason")),
        "query_sweep_best_strategy": _clean(alignment.get("query_sweep_best_strategy")),
        "model_answer": _clean(alignment.get("model_answer")),
        "hit_selected": hit is not None,
        "hit_alignment_score": None if hit is None else _float_or_zero(hit.get("alignment_score")),
        "hit_model_answer_value_matched": (
            False if hit is None else hit.get("model_answer_value_matched") is True
        ),
        "hit_matched_entity": "" if hit is None else _clean(hit.get("matched_entity")),
    }


def _alignment_context(alignment: Mapping[str, Any] | None) -> dict[str, Any]:
    if alignment is None:
        return {
            "alignment_available": False,
            "top_evidence_hits": (),
            "entity_candidates": (),
            "query_refinement_suggestions": (),
        }
    return {
        "alignment_available": True,
        "alignment_status": _clean(alignment.get("alignment_status")),
        "gap_reason": _clean(alignment.get("gap_reason")),
        "model_answer": _clean(alignment.get("model_answer")),
        "entity_candidates": _string_tuple(alignment.get("entity_candidates", ())),
        "query_refinement_suggestions": _string_tuple(alignment.get("query_refinement_suggestions", ())),
        "top_evidence_hits": tuple(
            {
                "source": _clean(hit.get("source")),
                "source_family": _clean(hit.get("source_family")),
                "provider": _clean(hit.get("provider")),
                "title": _clean(hit.get("title")),
                "url": _clean(hit.get("url")),
                "alignment_score": _float_or_zero(hit.get("alignment_score")),
                "query_overlap": _float_or_zero(hit.get("query_overlap")),
                "matched_entity": _clean(hit.get("matched_entity")),
                "model_answer_value_matched": hit.get("model_answer_value_matched") is True,
            }
            for hit in _mapping_sequence(alignment.get("top_evidence_hits", ()))[:3]
        ),
    }


def _source_note(alignment: Mapping[str, Any], hit: Mapping[str, Any] | None) -> str:
    if hit is None:
        return "No alignment hit met the source-backed candidate threshold."
    return (
        f"Drafted from {WORKFLOW}; alignment_status={_clean(alignment.get('alignment_status'))}; "
        f"source={_clean(hit.get('source'))}; review before fill."
    )


def _summary(
    *,
    entity_bindings: Sequence[Mapping[str, Any]],
    alignment_records: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_status_counts = Counter(str(item.get("candidate_status") or "") for item in candidates)
    failure_counts = Counter(
        failure
        for request in requests
        for failure in _string_tuple(request.get("blocking_failures", ()))
    )
    missing_candidate_field_counts = Counter(
        field
        for request in requests
        for field in _string_tuple(request.get("candidate_missing_fields", ()))
    )
    return {
        "entity_binding_count": len(entity_bindings),
        "alignment_record_count": len(alignment_records),
        "request_count": len(requests),
        "candidate_count": len(candidates),
        "ready_for_review_candidate_count": candidate_status_counts["ready_for_review"],
        "needs_source_evidence_candidate_count": candidate_status_counts["needs_source_evidence"],
        "needs_entity_value_review_candidate_count": candidate_status_counts["needs_entity_value_review"],
        "skipped_binding_count": len(skipped),
        "candidate_status_counts": _sorted_counter(candidate_status_counts),
        "blocking_failure_counts": _sorted_counter(failure_counts),
        "missing_candidate_field_counts": _sorted_counter(missing_candidate_field_counts),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if int(summary.get("ready_for_review_candidate_count", 0)) > 0:
        return "ready_for_review"
    if int(summary.get("candidate_count", 0)) > 0:
        return "needs_collection"
    return "empty"


def _is_entity_binding(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("sidecar_key") or "") == "entity_bindings"
        or str(row.get("input_name") or "") == "source_backed_entity_bindings"
        or str(row.get("collection_family") or "")
        in {COLLECTION_FAMILY, INPUT_BINDING_COLLECTION_FAMILY}
    )


def _skipped_binding(row: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "binding_id": _clean(row.get("binding_id")),
        "request_id": _request_id(row),
        "target_id": _clean(row.get("target_id")),
        "reason": reason,
        "not_verifier_evidence": True,
    }


def _request_id(row: Mapping[str, Any]) -> str:
    return _clean(row.get("request_id") or row.get("source_request_id"))


def _entity_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _sanitize(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in row.items() if str(key) not in RESERVED_FIELDS}


def _load_jsonl_mappings(
    path: str | Path,
    *,
    sanitize: bool = True,
) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append(_sanitize(row) if sanitize else {str(key): value for key, value in row.items()})
    return tuple(rows)


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return (str(value),) if str(value) else ()
    return tuple(str(item) for item in value if str(item))


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _float_or_zero(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter) if key}


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = (
        strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        if compact
        else strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = "".join(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    else:
        text = "".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata value must be KEY=VALUE, got {value!r}.")
        key, item = value.split("=", 1)
        metadata[key] = item
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity-bindings", required=True)
    parser.add_argument("--alignment-records", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--entity-binding-requests-jsonl", default=None)
    parser.add_argument("--candidate-entity-bindings-jsonl", default=None)
    parser.add_argument("--skipped-bindings-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--min-alignment-score", type=float, default=0.45)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(
        entity_bindings_path=args.entity_bindings,
        alignment_records_paths=tuple(args.alignment_records or ()),
        output_dir=args.output_dir,
        report_json_path=args.json,
        requests_path=args.entity_binding_requests_jsonl,
        candidate_bindings_path=args.candidate_entity_bindings_jsonl,
        skipped_bindings_path=args.skipped_bindings_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        min_alignment_score=float(args.min_alignment_score),
        metadata=_parse_metadata(args.metadata),
        compact_json=bool(args.compact_json),
    )
    print(strict_json_dumps({"status": payload["status"], "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
