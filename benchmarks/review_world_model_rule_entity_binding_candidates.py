"""Rule-review entity-role binding candidates before promotion.

This reviewer consumes the candidate rows emitted by
``plan_world_model_rule_entity_bindings.py`` and emits explicit review
decisions compatible with
``promote_world_model_rule_entity_binding_candidates.py``. It is deliberately
conservative: approval requires a complete ready-for-review candidate, an
answer entity, a source-backed expected entity, source metadata, and source
text/candidate evidence that mentions both the expected entity and queried
subject. The answer and expected entity may differ; those reviewed rows become
refutation-capable deterministic rule inputs downstream.

The reviewer does not use labels and does not make approved rows verifier
evidence. Approved decisions only allow the downstream promotion gate to
materialize a fill sidecar.
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

WORKFLOW = "world_model_rule_entity_binding_rule_review"
SOURCE_WORKFLOW = "world_model_rule_entity_binding_plan"
REVIEWER = "rule_based_entity_binding_reviewer_v1"
SUPPORTED_SOURCE_FAMILIES = {"news", "official", "official_statistics", "reference"}
REQUIRED_BINDING_FIELDS = (
    "request_id",
    "target_id",
    "subject_entity",
    "answer_entity",
    "expected_entity",
    "requested_role",
    "source_citation",
)


def review_world_model_rule_entity_binding_candidates(
    entity_binding_plan: Mapping[str, Any],
    *,
    reviewer: str = REVIEWER,
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    """Return rule-review decisions and diagnostics for entity candidates."""
    candidates = _candidate_bindings(entity_binding_plan)
    decisions: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    source_family_counts: Counter[str] = Counter()
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = _candidate_id(candidate, index=index)
        decision, reasons, checks = _review_candidate(candidate)
        decision_counts[decision] += 1
        for reason in reasons:
            reason_counts[reason] += 1
        if decision == "approved":
            source_family_counts[_clean(candidate.get("source_family")) or "unknown"] += 1
        review_id = _stable_review_id(candidate_id, decision=decision, reviewer=reviewer)
        notes = "; ".join(reasons) if reasons else "source_closed_entity_binding_candidate"
        decisions.append({
            "candidate_binding_id": candidate_id,
            "decision": decision,
            "review_id": review_id,
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "notes": notes,
        })
        records.append({
            "record_index": index,
            "candidate_binding_id": candidate_id,
            "decision": decision,
            "review_id": review_id,
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "notes": notes,
            "reasons": tuple(reasons),
            "checks": checks,
            "request_id": _clean(candidate.get("request_id")),
            "target_id": _clean(candidate.get("target_id")),
            "candidate_status": _clean(candidate.get("candidate_status")),
            "subject_entity": _clean(candidate.get("subject_entity")),
            "answer_entity": _clean(candidate.get("answer_entity")),
            "expected_entity": _clean(candidate.get("expected_entity")),
            "requested_role": _clean(candidate.get("requested_role")),
            "source_citation": _clean(candidate.get("source_citation")),
            "source_family": _clean(candidate.get("source_family")),
            "provider": _clean(candidate.get("provider")),
        })

    status = "ready_for_promotion_gate" if decision_counts["approved"] else "needs_more_evidence"
    if not candidates:
        status = "blocked"
    summary = {
        "candidate_count": len(candidates),
        "approved_count": decision_counts["approved"],
        "needs_more_evidence_count": decision_counts["needs_more_evidence"],
        "decision_counts": dict(sorted(decision_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "approved_source_family_counts": dict(sorted(source_family_counts.items())),
    }
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Rule-based review for entity-role binding candidates. Approval "
            "means the candidate has complete source-backed fields and a "
            "conservative source text/entity closure; it does not create "
            "verifier evidence."
        ),
        "label_usage": {
            "labels_used_for_review": False,
            "labels_copied_to_decisions": False,
            "model_answers_copied_to_decisions": False,
            "decisions_are_verifier_evidence": False,
        },
        "source": {
            "entity_binding_plan_workflow": entity_binding_plan.get("workflow"),
            "entity_binding_plan_status": entity_binding_plan.get("status"),
        },
        "config": {
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "supported_source_families": tuple(sorted(SUPPORTED_SOURCE_FAMILIES)),
            "decision_output_schema": "entity_binding_review_decisions",
        },
        "summary": summary,
        "decisions": tuple(decisions),
        "records": tuple(records),
    }


def run(
    *,
    entity_binding_plan_path: str | Path,
    output_dir: str | Path,
    decisions_jsonl_path: str | Path | None = None,
    report_json_path: str | Path | None = None,
    records_jsonl_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    reviewer: str = REVIEWER,
    reviewed_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register a rule-review report."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    decisions_path = Path(decisions_jsonl_path or output / "review-decisions.jsonl")
    report_path = Path(report_json_path or output / "entity-binding-review-report.json")
    records_path = Path(records_jsonl_path or output / "review-records.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    entity_binding_plan = _load_json_object(entity_binding_plan_path)
    payload = review_world_model_rule_entity_binding_candidates(
        entity_binding_plan,
        reviewer=reviewer,
        reviewed_at=reviewed_at,
    )
    payload = dict(payload)
    payload["paths"] = {
        "entity_binding_plan": str(entity_binding_plan_path),
        "review_decisions": str(decisions_path),
        "review_report": str(report_path),
        "review_records": str(records_path),
        "artifact_manifest": str(manifest_path),
    }
    payload["metadata"] = dict(metadata or {})

    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(decisions_path, payload["decisions"], compact=compact_json)
    _write_jsonl(records_path, payload["records"], compact=compact_json)
    manifest = build_artifact_manifest(
        {
            "entity_binding_rule_review_report": report_path,
            "review_decisions": decisions_path,
            "review_records": records_path,
            "entity_binding_plan": entity_binding_plan_path,
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "candidate_count": payload["summary"]["candidate_count"],
            "approved_count": payload["summary"]["approved_count"],
            "needs_more_evidence_count": payload["summary"]["needs_more_evidence_count"],
            "reviewer": reviewer,
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
                "artifact_manifest": str(manifest_path),
                "candidate_count": payload["summary"]["candidate_count"],
                "approved_count": payload["summary"]["approved_count"],
                "needs_more_evidence_count": payload["summary"]["needs_more_evidence_count"],
                "reviewer": reviewer,
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _review_candidate(candidate: Mapping[str, Any]) -> tuple[str, tuple[str, ...], dict[str, bool]]:
    source_text = _source_text(candidate)
    expected = _clean(candidate.get("expected_entity"))
    answer = _clean(candidate.get("answer_entity"))
    subject = _clean(candidate.get("subject_entity"))
    source_family = _clean(candidate.get("source_family")).casefold()
    checks = {
        "candidate_ready_for_review": _clean(candidate.get("candidate_status")) == "ready_for_review",
        "not_verifier_evidence": candidate.get("not_verifier_evidence") is True,
        "complete_required_fields": not _missing_required_fields(candidate),
        "answer_entity_present": bool(answer),
        "expected_entity_present": bool(expected),
        "has_source_citation": bool(_clean(candidate.get("source_citation"))),
        "source_family_supported": source_family in SUPPORTED_SOURCE_FAMILIES,
        "source_mentions_expected_entity": _contains_entity(source_text, expected),
        "source_or_candidates_mentions_subject": _contains_entity(source_text, subject),
    }
    reasons = tuple(key for key, passed in checks.items() if not passed)
    decision = "approved" if not reasons else "needs_more_evidence"
    return decision, reasons, checks


def _candidate_bindings(entity_binding_plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if entity_binding_plan.get("workflow") != SOURCE_WORKFLOW:
        raise ValueError(f"entity_binding_plan must have workflow={SOURCE_WORKFLOW!r}.")
    raw_candidates = entity_binding_plan.get("candidate_entity_bindings")
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes, bytearray)):
        raise ValueError("entity_binding_plan must contain candidate_entity_bindings.")
    return tuple(dict(item) for item in raw_candidates if isinstance(item, Mapping))


def _source_text(candidate: Mapping[str, Any]) -> str:
    parts = [
        _clean(candidate.get("source_title")),
        _clean(candidate.get("source_note")),
        _clean(candidate.get("source_url")),
        " ".join(_string_sequence(candidate.get("candidate_expected_entities", ()))),
    ]
    alignment = candidate.get("candidate_alignment")
    if isinstance(alignment, Mapping):
        parts.extend((
            _clean(alignment.get("hit_matched_entity")),
            _clean(alignment.get("gap_reason")),
            _clean(alignment.get("alignment_status")),
        ))
    return " ".join(part for part in parts if part)


def _missing_required_fields(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(key for key in REQUIRED_BINDING_FIELDS if not _clean(candidate.get(key)))


def _contains_entity(text: str, entity: str) -> bool:
    normalized_text = _entity_key(text)
    normalized_entity = _entity_key(entity)
    if not normalized_entity:
        return False
    return normalized_entity in normalized_text


def _candidate_id(candidate: Mapping[str, Any], *, index: int) -> str:
    return _clean(candidate.get("binding_id")) or f"entity-binding-candidate:{index}"


def _stable_review_id(candidate_id: str, *, decision: str, reviewer: str) -> str:
    payload = strict_json_dumps(
        {"candidate_id": candidate_id, "decision": decision, "reviewer": reviewer},
        sort_keys=True,
    )
    return "entity-binding-rule-review:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _string_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return (str(value),) if str(value) else ()
    return tuple(str(item) for item in value if str(item))


def _entity_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


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


def _write_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    compact: bool = False,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = "".join(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    else:
        text = "".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity-binding-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decisions-jsonl", default=None)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--records-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--reviewer", default=REVIEWER)
    parser.add_argument("--reviewed-at", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(
        entity_binding_plan_path=args.entity_binding_plan,
        output_dir=args.output_dir,
        decisions_jsonl_path=args.decisions_jsonl,
        report_json_path=args.report_json,
        records_jsonl_path=args.records_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        reviewer=args.reviewer,
        reviewed_at=args.reviewed_at,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    print(
        "world_model_rule_entity_binding_rule_review_ok "
        f"status={payload['status']} "
        f"candidates={payload['summary']['candidate_count']} "
        f"approved={payload['summary']['approved_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
