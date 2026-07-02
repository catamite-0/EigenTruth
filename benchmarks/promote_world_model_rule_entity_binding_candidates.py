"""Gate entity-role binding candidates before rule-input fill.

This workflow consumes the review-gated candidate rows emitted by
``plan_world_model_rule_entity_bindings.py`` and optional explicit review
decisions. Without approvals it writes a decision template and a blocked
``needs_review`` report. With approved decisions it materializes an
``approved-entity-bindings.jsonl`` sidecar that can be passed to
``fill_world_model_rule_inputs_from_entity_bindings.py``.

Candidates are collection artifacts, not verifier evidence. This gate never
uses labels, never approves a candidate without a reviewer decision, and only
materializes candidates that are already complete source-backed rows.
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

WORKFLOW = "world_model_rule_entity_binding_promotion_gate"
SOURCE_WORKFLOW = "world_model_rule_entity_binding_plan"
ALLOWED_DECISIONS = {"approved", "rejected", "needs_more_evidence"}
APPROVED_ALIASES = {"accept", "accepted", "approve", "approved", "pass", "promote"}
REJECTED_ALIASES = {"block", "blocked", "reject", "rejected", "fail"}
NEEDS_MORE_EVIDENCE_ALIASES = {
    "defer",
    "needs_evidence",
    "needs-more-evidence",
    "needs_more_evidence",
    "needs-review",
    "needs_review",
    "pending",
}
RESERVED_REVIEW_KEYS = {
    "answer",
    "answers",
    "is_false",
    "label",
    "labels",
    "model_answer",
    "request_id",
    "score_label",
    "source_request_id",
    "target_id",
}
REQUIRED_BINDING_FIELDS = (
    "request_id",
    "target_id",
    "subject_entity",
    "answer_entity",
    "expected_entity",
    "requested_role",
    "source_citation",
)


def promote_world_model_rule_entity_binding_candidates(
    entity_binding_plan: Mapping[str, Any],
    *,
    review_decisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return a JSON-ready candidate promotion-gate payload."""
    candidates = _candidate_bindings(entity_binding_plan)
    parsed_decisions = _parse_review_decisions(review_decisions)
    decisions_by_candidate = parsed_decisions["decisions_by_candidate"]
    records: list[dict[str, Any]] = list(parsed_decisions["records"])
    templates: list[dict[str, Any]] = []
    approved_bindings: list[dict[str, Any]] = []
    counters: Counter[str] = Counter({
        "approved": 0,
        "candidate_not_ready": 0,
        "duplicate_review_decision": parsed_decisions["duplicate_review_decision"],
        "invalid_review_decision": parsed_decisions["invalid_review_decision"],
        "missing_candidate_id": parsed_decisions["missing_candidate_id"],
        "missing_reviewer": 0,
        "needs_more_evidence": 0,
        "pending_review": 0,
        "rejected": 0,
        "reserved_review_metadata": parsed_decisions["reserved_review_metadata"],
        "unknown_candidate_id": 0,
    })
    seen_candidate_ids: set[str] = set()

    for index, candidate in enumerate(candidates, start=1):
        candidate_id = _candidate_id(candidate, index=index)
        seen_candidate_ids.add(candidate_id)
        template = _review_template(candidate, candidate_id=candidate_id, index=index)
        templates.append(template)
        decision = decisions_by_candidate.get(candidate_id)
        if decision is None:
            counters["pending_review"] += 1
            records.append(_record_from_template(template, decision="pending", skip_reason="pending_review"))
            continue
        decision_value = str(decision["decision"])
        if decision_value == "rejected":
            counters["rejected"] += 1
            records.append(_record_from_template(template, decision="rejected", review_decision=decision))
            continue
        if decision_value == "needs_more_evidence":
            counters["needs_more_evidence"] += 1
            records.append(
                _record_from_template(
                    template,
                    decision="needs_more_evidence",
                    skip_reason="needs_more_evidence",
                    review_decision=decision,
                )
            )
            continue
        reviewer = _reviewer(decision)
        if not reviewer:
            counters["missing_reviewer"] += 1
            records.append(
                _record_from_template(
                    template,
                    decision="skipped",
                    skip_reason="missing_reviewer",
                    review_decision=decision,
                )
            )
            continue
        if not _eligible_for_approval(candidate):
            counters["candidate_not_ready"] += 1
            records.append(
                _record_from_template(
                    template,
                    decision="skipped",
                    skip_reason="candidate_not_ready",
                    review_decision=decision,
                )
            )
            continue
        binding = _approved_binding(
            candidate,
            decision,
            candidate_id=candidate_id,
            binding_index=len(approved_bindings) + 1,
        )
        approved_bindings.append(binding)
        counters["approved"] += 1
        records.append(
            _record_from_template(
                template,
                decision="approved",
                review_decision=decision,
                approved_binding_id=str(binding["binding_id"]),
            )
        )

    for candidate_id, decision in decisions_by_candidate.items():
        if candidate_id in seen_candidate_ids:
            continue
        counters["unknown_candidate_id"] += 1
        records.append({
            "record_type": "review_decision",
            "candidate_binding_id": candidate_id,
            "decision": decision["decision"],
            "skip_reason": "unknown_candidate_id",
            "review_id": decision.get("review_id"),
        })

    candidate_status_counts = Counter(str(item.get("candidate_status") or "") for item in candidates)
    provider_counts = Counter(str(item.get("provider") or "") for item in approved_bindings)
    status = _status(
        candidate_count=len(candidates),
        approved_count=counters["approved"],
        pending_count=counters["pending_review"],
        blocking_issue_count=(
            counters["duplicate_review_decision"]
            + counters["invalid_review_decision"]
            + counters["reserved_review_metadata"]
        ),
    )
    summary = {
        "candidate_count": len(candidates),
        "review_decision_count": len(review_decisions),
        "review_template_count": len(templates),
        "approved_binding_count": counters["approved"],
        "pending_review_count": counters["pending_review"],
        "rejected_count": counters["rejected"],
        "needs_more_evidence_count": counters["needs_more_evidence"],
        "skip_counts": dict(sorted(counters.items())),
        "candidate_status_counts": dict(sorted(candidate_status_counts.items())),
        "provider_counts": _sorted_counter(provider_counts),
    }
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Requires explicit review approval before entity-role binding "
            "candidates can become fill sidecar rows. Approved bindings remain "
            "adapter inputs, not verifier evidence, and still require typed "
            "fill, adapter execution, and promotion."
        ),
        "source": {
            "entity_binding_plan_workflow": entity_binding_plan.get("workflow"),
            "entity_binding_plan_status": entity_binding_plan.get("status"),
        },
        "label_usage": {
            "labels_used_for_gate": False,
            "labels_copied_to_outputs": False,
            "model_answers_copied_to_outputs": False,
            "candidate_bindings_are_verifier_evidence": False,
            "approved_bindings_are_verifier_evidence": False,
            "requires_explicit_review_decision": True,
        },
        "summary": summary,
        "review_template": tuple(templates),
        "records": tuple(records),
        "approved_entity_bindings": tuple(approved_bindings),
    }


def run(
    *,
    entity_binding_plan_path: str | Path,
    output_dir: str | Path,
    review_decisions_path: str | Path | None = None,
    report_json_path: str | Path | None = None,
    approved_bindings_path: str | Path | None = None,
    template_jsonl_path: str | Path | None = None,
    records_jsonl_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register a promotion-gate report."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "entity-binding-promotion-gate.json")
    approved_path = Path(approved_bindings_path or output / "approved-entity-bindings.jsonl")
    template_path = Path(template_jsonl_path or output / "review-decision-template.jsonl")
    records_path = Path(records_jsonl_path or output / "promotion-gate-records.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    entity_binding_plan = _load_json_object(entity_binding_plan_path)
    review_decisions = load_review_decisions(review_decisions_path) if review_decisions_path else ()
    payload = promote_world_model_rule_entity_binding_candidates(
        entity_binding_plan,
        review_decisions=review_decisions,
    )
    payload = dict(payload)
    payload["paths"] = {
        "entity_binding_plan": str(entity_binding_plan_path),
        "review_decisions": None if review_decisions_path is None else str(review_decisions_path),
        "report": str(report_path),
        "approved_entity_bindings": str(approved_path),
        "review_decision_template": str(template_path),
        "records_jsonl": str(records_path),
        "artifact_manifest": str(manifest_path),
    }
    payload["metadata"] = dict(metadata or {})

    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(approved_path, payload["approved_entity_bindings"], compact=compact_json)
    _write_jsonl(template_path, payload["review_template"], compact=compact_json)
    _write_jsonl(records_path, payload["records"], compact=compact_json)
    manifest_sources: dict[str, str | Path | None] = {
        "entity_binding_promotion_gate_report": report_path,
        "approved_entity_bindings": approved_path,
        "review_decision_template": template_path,
        "promotion_gate_records": records_path,
        "entity_binding_plan": entity_binding_plan_path,
    }
    if review_decisions_path is not None:
        manifest_sources["review_decisions"] = review_decisions_path
    manifest = build_artifact_manifest(
        manifest_sources,
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "candidate_count": payload["summary"]["candidate_count"],
            "approved_binding_count": payload["summary"]["approved_binding_count"],
            "pending_review_count": payload["summary"]["pending_review_count"],
            "promotes_verifier_evidence": False,
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
                "approved_binding_count": payload["summary"]["approved_binding_count"],
                "pending_review_count": payload["summary"]["pending_review_count"],
                "promotes_verifier_evidence": False,
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def load_review_decisions(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load review decisions from JSONL, JSON list, or object wrappers."""
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        return tuple(dict(item) for item in _load_jsonl(source) if isinstance(item, Mapping))
    payload = _load_json(source)
    if isinstance(payload, Mapping):
        for key in ("review_decisions", "decisions", "records"):
            values = _non_string_sequence(payload.get(key))
            if values is not None:
                return tuple(dict(item) for item in values if isinstance(item, Mapping))
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return tuple(dict(item) for item in payload if isinstance(item, Mapping))
    raise ValueError("review decisions must be JSONL, JSON list, or JSON object with decisions.")


def _candidate_bindings(entity_binding_plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if entity_binding_plan.get("workflow") != SOURCE_WORKFLOW:
        raise ValueError(f"entity_binding_plan must have workflow={SOURCE_WORKFLOW!r}.")
    raw_candidates = entity_binding_plan.get("candidate_entity_bindings")
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes, bytearray)):
        raise ValueError("entity_binding_plan must contain candidate_entity_bindings.")
    return tuple(dict(item) for item in raw_candidates if isinstance(item, Mapping))


def _parse_review_decisions(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    decisions_by_candidate: dict[str, dict[str, Any]] = {}
    counters: Counter[str] = Counter({
        "duplicate_review_decision": 0,
        "invalid_review_decision": 0,
        "missing_candidate_id": 0,
        "reserved_review_metadata": 0,
    })
    for index, raw_decision in enumerate(decisions, start=1):
        decision = dict(raw_decision)
        candidate_id = _candidate_id_from_decision(decision)
        review_id = _clean(decision.get("review_id")) or _stable_review_id(decision, index=index)
        base_record = {
            "record_type": "review_decision",
            "decision_index": index,
            "candidate_binding_id": candidate_id,
            "review_id": review_id,
        }
        reserved_keys = tuple(sorted(key for key in decision if str(key) in RESERVED_REVIEW_KEYS))
        if reserved_keys:
            counters["reserved_review_metadata"] += 1
            records.append({
                **base_record,
                "decision": _clean(decision.get("decision")),
                "skip_reason": "reserved_review_metadata",
                "reserved_keys": reserved_keys,
            })
            continue
        if not candidate_id:
            counters["missing_candidate_id"] += 1
            records.append({
                **base_record,
                "decision": _clean(decision.get("decision")),
                "skip_reason": "missing_candidate_id",
            })
            continue
        normalized_decision = _normalize_decision(decision.get("decision"))
        if normalized_decision is None:
            counters["invalid_review_decision"] += 1
            records.append({
                **base_record,
                "decision": _clean(decision.get("decision")),
                "skip_reason": "invalid_review_decision",
            })
            continue
        if candidate_id in decisions_by_candidate:
            counters["duplicate_review_decision"] += 1
            records.append({
                **base_record,
                "decision": normalized_decision,
                "skip_reason": "duplicate_review_decision",
            })
            continue
        decisions_by_candidate[candidate_id] = {
            "candidate_binding_id": candidate_id,
            "decision": normalized_decision,
            "review_id": review_id,
            "reviewer": _reviewer(decision),
            "reviewed_at": _clean(decision.get("reviewed_at")),
            "notes": _clean(decision.get("notes")),
        }
    return {
        "records": tuple(records),
        "decisions_by_candidate": decisions_by_candidate,
        **counters,
    }


def _review_template(candidate: Mapping[str, Any], *, candidate_id: str, index: int) -> dict[str, Any]:
    return {
        "template_version": 1,
        "template_usage": "entity_binding_review_decision",
        "candidate_index": index,
        "candidate_binding_id": candidate_id,
        "decision": "pending",
        "allowed_decisions": tuple(sorted(ALLOWED_DECISIONS)),
        "reviewer": "",
        "reviewed_at": "",
        "notes": "",
        "eligible_for_approval": _eligible_for_approval(candidate),
        "candidate_status": _clean(candidate.get("candidate_status")),
        "request_id": _clean(candidate.get("request_id")),
        "target_id": _clean(candidate.get("target_id")),
        "subject_entity": _clean(candidate.get("subject_entity")),
        "answer_entity": _clean(candidate.get("answer_entity")),
        "expected_entity": _clean(candidate.get("expected_entity")),
        "requested_role": _clean(candidate.get("requested_role")),
        "source_citation": _clean(candidate.get("source_citation")),
        "source_title": _clean(candidate.get("source_title")),
        "source_url": _clean(candidate.get("source_url")),
        "source_family": _clean(candidate.get("source_family")),
        "provider": _clean(candidate.get("provider")),
        "not_verifier_evidence": candidate.get("not_verifier_evidence") is True,
    }


def _record_from_template(
    template: Mapping[str, Any],
    *,
    decision: str,
    skip_reason: str | None = None,
    review_decision: Mapping[str, Any] | None = None,
    approved_binding_id: str | None = None,
) -> dict[str, Any]:
    review = review_decision or {}
    return {
        "record_type": "candidate_review",
        "candidate_binding_id": template.get("candidate_binding_id"),
        "decision": decision,
        "skip_reason": skip_reason,
        "approved_binding_id": approved_binding_id,
        "review_id": review.get("review_id"),
        "reviewer": review.get("reviewer"),
        "reviewed_at": review.get("reviewed_at"),
        "candidate_status": template.get("candidate_status"),
        "request_id": template.get("request_id"),
        "target_id": template.get("target_id"),
        "source_citation": template.get("source_citation"),
    }


def _approved_binding(
    candidate: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    candidate_id: str,
    binding_index: int,
) -> dict[str, Any]:
    review_id = _clean(decision.get("review_id"))
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "binding_id": f"approved-entity-binding:{binding_index:04d}:{_short_hash(candidate_id)}",
        "candidate_binding_id": candidate_id,
        "request_id": _clean(candidate.get("request_id")),
        "source_request_id": _clean(candidate.get("source_request_id") or candidate.get("request_id")),
        "target_id": _clean(candidate.get("target_id")),
        "task_id": _clean(candidate.get("task_id")),
        "collection_family": _clean(candidate.get("collection_family")),
        "question": _clean(candidate.get("question")),
        "rule_family": _clean(candidate.get("rule_family")) or "entity_disambiguation",
        "subject_entity": _clean(candidate.get("subject_entity")),
        "answer_entity": _clean(candidate.get("answer_entity")),
        "expected_entity": _clean(candidate.get("expected_entity")),
        "requested_role": _clean(candidate.get("requested_role")),
        "source_citation": _clean(candidate.get("source_citation")),
        "source_url": _clean(candidate.get("source_url")),
        "source_title": _clean(candidate.get("source_title")),
        "source_family": _clean(candidate.get("source_family")),
        "provider": _clean(candidate.get("provider")),
        "candidate_answer_source": _clean(candidate.get("candidate_answer_source")),
        "expected_entity_source": _clean(candidate.get("expected_entity_source")),
        "source_note": (
            f"Approved by {WORKFLOW}; candidate={candidate_id}; "
            f"review_id={review_id}; review before product promotion."
        ),
        "review_status": "approved",
        "review_id": review_id,
        "reviewer": _reviewer(decision),
        "reviewed_at": _clean(decision.get("reviewed_at")),
        "review_notes": _clean(decision.get("notes")),
        "not_verifier_evidence": True,
        "candidate_results_require_promotion_gate": True,
    }


def _eligible_for_approval(candidate: Mapping[str, Any]) -> bool:
    if _clean(candidate.get("candidate_status")) != "ready_for_review":
        return False
    if candidate.get("not_verifier_evidence") is not True:
        return False
    if _clean(candidate.get("review_status")).lower() not in {"needs_review", "pending"}:
        return False
    return not _missing_required_candidate_fields(candidate)


def _missing_required_candidate_fields(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(key for key in REQUIRED_BINDING_FIELDS if not _clean(candidate.get(key)))


def _status(
    *,
    candidate_count: int,
    approved_count: int,
    pending_count: int,
    blocking_issue_count: int,
) -> str:
    if candidate_count == 0 or blocking_issue_count > 0:
        return "blocked" if approved_count == 0 else "partial"
    if approved_count > 0 and pending_count == 0:
        return "ready_for_fill"
    if approved_count > 0:
        return "partial"
    return "needs_review"


def _candidate_id(candidate: Mapping[str, Any], *, index: int) -> str:
    return _clean(candidate.get("binding_id")) or f"entity-binding-candidate:{index}"


def _candidate_id_from_decision(decision: Mapping[str, Any]) -> str:
    return _clean(
        decision.get("candidate_binding_id")
        or decision.get("entity_binding_candidate_id")
        or decision.get("candidate_id")
        or decision.get("binding_id")
    )


def _normalize_decision(value: Any) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", _clean(value).lower()).strip("_")
    if normalized in APPROVED_ALIASES:
        return "approved"
    if normalized in REJECTED_ALIASES:
        return "rejected"
    if normalized in NEEDS_MORE_EVIDENCE_ALIASES:
        return "needs_more_evidence"
    return None


def _reviewer(decision: Mapping[str, Any]) -> str:
    return _clean(decision.get("reviewer") or decision.get("reviewer_id") or decision.get("review_source"))


def _stable_review_id(decision: Mapping[str, Any], *, index: int) -> str:
    payload = strict_json_dumps(
        {
            "candidate_id": _candidate_id_from_decision(decision),
            "decision": _clean(decision.get("decision")),
            "index": index,
            "reviewer": _reviewer(decision),
        },
        sort_keys=True,
    )
    return "entity-binding-review:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter) if key}


def _non_string_sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return None


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = _load_json(Path(path))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> tuple[Any, ...]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
    return tuple(rows)


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
    parser.add_argument("--review-decisions", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--approved-entity-bindings-jsonl", default=None)
    parser.add_argument("--template-jsonl", default=None)
    parser.add_argument("--records-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(
        entity_binding_plan_path=args.entity_binding_plan,
        review_decisions_path=args.review_decisions,
        output_dir=args.output_dir,
        report_json_path=args.report_json,
        approved_bindings_path=args.approved_entity_bindings_jsonl,
        template_jsonl_path=args.template_jsonl,
        records_jsonl_path=args.records_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    print(
        "world_model_rule_entity_binding_promotion_gate_ok "
        f"status={payload['status']} "
        f"candidates={payload['summary']['candidate_count']} "
        f"approved_bindings={payload['summary']['approved_binding_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
