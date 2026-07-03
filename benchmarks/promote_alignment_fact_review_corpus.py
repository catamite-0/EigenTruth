"""Gate alignment fact-review rows before structured-fact promotion.

This workflow consumes the review-only corpus produced by
``build_alignment_fact_review_corpus.py`` and optional explicit review
decisions. Without approvals it writes a decision template and a blocked
``needs_review`` report. With approved decisions it materializes source-family
style structured source documents that can be passed to
``build_source_family_qa_corpus.py`` and then to the covered-facts route audit.

The gate is deliberately conservative: review-corpus rows are not verifier
evidence just because they passed the upstream extraction checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from eigentruth.verify import normalize_claim_text  # noqa: E402

WORKFLOW = "alignment_fact_review_promotion_gate"
SOURCE_CORPUS_TYPE = "alignment_review_approved_source_documents"
REVIEW_CORPUS_TYPE = "alignment_structured_fact_review_corpus"
ALLOWED_DECISIONS = {"approved", "rejected", "needs_more_evidence"}
APPROVED_ALIASES = {"accept", "accepted", "approve", "approved", "pass", "promote"}
REJECTED_ALIASES = {"block", "blocked", "reject", "rejected", "fail"}
NEEDS_MORE_EVIDENCE_ALIASES = {
    "defer",
    "needs_evidence",
    "needs_more_evidence",
    "needs-review",
    "needs_review",
    "pending",
}
RESERVED_REVIEW_KEYS = {
    "claim_id",
    "is_false",
    "label",
    "model_answer",
    "queue_id",
    "record_index",
    "request_id",
    "row_index",
    "score_dump_row",
    "score_label",
    "source_request_id",
    "target_id",
}


def promote_alignment_fact_review_corpus(
    review_corpus: Mapping[str, Any],
    *,
    review_decisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return a JSON-ready promotion-gate payload."""
    documents = _review_documents(review_corpus)
    parsed_decisions = _parse_review_decisions(review_decisions)
    decisions_by_candidate = parsed_decisions["decisions_by_candidate"]
    records: list[dict[str, Any]] = list(parsed_decisions["records"])
    templates: list[dict[str, Any]] = []
    approved_source_documents: list[dict[str, Any]] = []
    counters = Counter({
        "approved": 0,
        "duplicate_review_decision": parsed_decisions["duplicate_review_decision"],
        "invalid_review_decision": parsed_decisions["invalid_review_decision"],
        "missing_candidate_id": parsed_decisions["missing_candidate_id"],
        "missing_reviewer": 0,
        "needs_more_evidence": 0,
        "pending_review": 0,
        "rejected": 0,
        "reserved_review_metadata": parsed_decisions["reserved_review_metadata"],
        "unknown_candidate_id": 0,
        "unsupported_evidence_source": 0,
    })
    seen_candidate_ids: set[str] = set()

    for index, document in enumerate(documents, start=1):
        metadata = _metadata(document)
        candidate_id = _clean_text(metadata.get("alignment_candidate_id")) or _stable_document_id(document)
        seen_candidate_ids.add(candidate_id)
        template = _review_template_row(document, candidate_id=candidate_id, index=index)
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
        source_document = _approved_source_document(
            document,
            decision,
            candidate_id=candidate_id,
            source_index=len(approved_source_documents) + 1,
        )
        if source_document is None:
            counters["unsupported_evidence_source"] += 1
            records.append(
                _record_from_template(
                    template,
                    decision="skipped",
                    skip_reason="unsupported_evidence_source",
                    review_decision=decision,
                )
            )
            continue
        approved_source_documents.append(source_document)
        counters["approved"] += 1
        records.append(
            _record_from_template(
                template,
                decision="approved",
                review_decision=decision,
                source_document_id=str(source_document["metadata"]["alignment_source_document_id"]),
            )
        )

    for candidate_id, decision in decisions_by_candidate.items():
        if candidate_id in seen_candidate_ids:
            continue
        counters["unknown_candidate_id"] += 1
        records.append({
            "record_type": "review_decision",
            "candidate_id": candidate_id,
            "decision": decision["decision"],
            "skip_reason": "unknown_candidate_id",
            "review_id": decision.get("review_id"),
        })

    by_property = Counter(
        str(item["metadata"].get("statement_property", "unknown")) for item in approved_source_documents
    )
    by_provider = Counter(str(item["metadata"].get("provider", "unknown")) for item in approved_source_documents)
    by_source_family = Counter(
        str(item["metadata"].get("source_family", "unknown")) for item in approved_source_documents
    )
    status = "ready_for_structured_qa" if approved_source_documents else "needs_review"
    if not documents or counters["invalid_review_decision"] or counters["reserved_review_metadata"]:
        status = "blocked" if not approved_source_documents else "partial"
    summary = {
        "review_document_count": len(documents),
        "review_decision_count": len(review_decisions),
        "approved_source_document_count": len(approved_source_documents),
        "pending_review_count": counters["pending_review"],
        "rejected_count": counters["rejected"],
        "needs_more_evidence_count": counters["needs_more_evidence"],
        "skip_counts": dict(sorted(counters.items())),
        "by_property": dict(sorted(by_property.items())),
        "by_provider": dict(sorted(by_provider.items())),
        "by_source_family": dict(sorted(by_source_family.items())),
    }
    source_documents = {
        "schema_version": 1,
        "corpus_type": SOURCE_CORPUS_TYPE,
        "status": "ready" if approved_source_documents else "empty",
        "description": (
            "Approved alignment fact-review source documents. These are "
            "inputs for source-family structured QA corpus construction, not "
            "a broad route promotion by themselves."
        ),
        "label_usage": {
            "labels_used_for_documents": False,
            "labels_copied_to_document_metadata": False,
            "request_ids_copied_to_document_metadata": False,
            "target_ids_copied_to_document_metadata": False,
            "model_answers_copied_to_document_metadata": False,
        },
        "source": {
            "builder": WORKFLOW,
            "review_corpus_type": review_corpus.get("corpus_type"),
        },
        "summary": summary,
        "documents": approved_source_documents,
    }
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Requires explicit review approval before alignment fact-review "
            "rows can become structured source documents. This gate does not "
            "promote verifier evidence or open-domain coverage."
        ),
        "label_usage": {
            "labels_used_for_gate": False,
            "labels_copied_to_outputs": False,
            "request_ids_copied_to_source_documents": False,
            "target_ids_copied_to_source_documents": False,
            "model_answers_copied_to_source_documents": False,
        },
        "source": {
            "review_corpus_type": review_corpus.get("corpus_type"),
            "review_corpus_status": review_corpus.get("status"),
        },
        "summary": summary,
        "review_template": tuple(templates),
        "records": tuple(records),
        "approved_source_documents": source_documents,
    }


def run(
    *,
    review_corpus_path: str | Path,
    output_dir: str | Path,
    review_decisions_path: str | Path | None = None,
    report_json_path: str | Path | None = None,
    source_documents_json_path: str | Path | None = None,
    template_jsonl_path: str | Path | None = None,
    records_jsonl_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register a promotion gate."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    report_path = Path(report_json_path) if report_json_path is not None else output / "promotion-gate-report.json"
    source_path = (
        Path(source_documents_json_path)
        if source_documents_json_path is not None
        else output / "approved-source-documents.json"
    )
    template_path = (
        Path(template_jsonl_path)
        if template_jsonl_path is not None
        else output / "review-decision-template.jsonl"
    )
    records_path = (
        Path(records_jsonl_path)
        if records_jsonl_path is not None
        else output / "promotion-gate-records.jsonl"
    )
    manifest_path = (
        Path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else output / "artifact-manifest.json"
    )
    review_corpus = _load_json_object(review_corpus_path)
    review_decisions = load_review_decisions(review_decisions_path) if review_decisions_path is not None else ()
    payload = promote_alignment_fact_review_corpus(
        review_corpus,
        review_decisions=review_decisions,
    )
    payload = dict(payload)
    payload["paths"] = {
        "review_corpus": str(review_corpus_path),
        "review_decisions": None if review_decisions_path is None else str(review_decisions_path),
        "report": str(report_path),
        "approved_source_documents": str(source_path),
        "review_decision_template": str(template_path),
        "records_jsonl": str(records_path),
        "artifact_manifest": str(manifest_path),
    }
    payload["metadata"] = dict(metadata or {})
    _write_json(report_path, payload, compact=compact_json)
    _write_json(source_path, payload["approved_source_documents"], compact=compact_json)
    _write_jsonl(template_path, payload["review_template"])
    _write_jsonl(records_path, payload["records"])

    manifest_sources = {
        "alignment_fact_review_promotion_gate_report": report_path,
        "approved_source_documents": source_path,
        "review_decision_template": template_path,
        "promotion_gate_records": records_path,
        "alignment_fact_review_corpus": review_corpus_path,
    }
    if review_decisions_path is not None:
        manifest_sources["review_decisions"] = review_decisions_path
    manifest = build_artifact_manifest(
        manifest_sources,
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "review_document_count": payload["summary"]["review_document_count"],
            "approved_source_document_count": payload["summary"]["approved_source_document_count"],
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
                "review_document_count": payload["summary"]["review_document_count"],
                "approved_source_document_count": payload["summary"]["approved_source_document_count"],
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


def _parse_review_decisions(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    decisions_by_candidate: dict[str, dict[str, Any]] = {}
    counters = Counter({
        "duplicate_review_decision": 0,
        "invalid_review_decision": 0,
        "missing_candidate_id": 0,
        "reserved_review_metadata": 0,
    })
    for index, raw_decision in enumerate(decisions, start=1):
        decision = dict(raw_decision)
        candidate_id = _clean_text(decision.get("alignment_candidate_id") or decision.get("candidate_id"))
        review_id = _clean_text(decision.get("review_id")) or _stable_review_id(decision, index=index)
        base_record = {
            "record_type": "review_decision",
            "decision_index": index,
            "candidate_id": candidate_id,
            "review_id": review_id,
        }
        reserved_keys = sorted(key for key in decision if key in RESERVED_REVIEW_KEYS)
        if reserved_keys:
            counters["reserved_review_metadata"] += 1
            records.append({
                **base_record,
                "decision": _clean_text(decision.get("decision")),
                "skip_reason": "reserved_review_metadata",
                "reserved_keys": tuple(reserved_keys),
            })
            continue
        if not candidate_id:
            counters["missing_candidate_id"] += 1
            records.append({
                **base_record,
                "decision": _clean_text(decision.get("decision")),
                "skip_reason": "missing_candidate_id",
            })
            continue
        normalized_decision = _normalize_decision(decision.get("decision"))
        if normalized_decision is None:
            counters["invalid_review_decision"] += 1
            records.append({
                **base_record,
                "decision": _clean_text(decision.get("decision")),
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
            "alignment_candidate_id": candidate_id,
            "decision": normalized_decision,
            "review_id": review_id,
            "reviewer": _reviewer(decision),
            "reviewed_at": _clean_text(decision.get("reviewed_at")),
            "notes": _clean_text(decision.get("notes")),
        }
    return {
        "records": tuple(records),
        "decisions_by_candidate": decisions_by_candidate,
        **counters,
    }


def _review_documents(review_corpus: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if review_corpus.get("corpus_type") not in {REVIEW_CORPUS_TYPE, None}:
        raise ValueError(f"review_corpus must have corpus_type={REVIEW_CORPUS_TYPE!r}.")
    raw_documents = review_corpus.get("documents")
    if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, (str, bytes, bytearray)):
        raise ValueError("review_corpus must contain a documents list.")
    return tuple(dict(item) for item in raw_documents if isinstance(item, Mapping))


def _review_template_row(document: Mapping[str, Any], *, candidate_id: str, index: int) -> dict[str, Any]:
    metadata = _metadata(document)
    property_hint = _clean_text(metadata.get("property_hint")) or ""
    property_label = _clean_text(metadata.get("statement_property_label")) or _property_label(property_hint)
    return {
        "template_version": 1,
        "template_usage": "alignment_fact_review_decision",
        "review_document_index": index,
        "alignment_candidate_id": candidate_id,
        "decision": "pending",
        "allowed_decisions": tuple(sorted(ALLOWED_DECISIONS)),
        "reviewer": "",
        "reviewed_at": "",
        "notes": "",
        "subject": _clean_text(metadata.get("subject")),
        "property_hint": property_hint,
        "property_label": property_label,
        "value": _clean_text(document.get("answer")),
        "evidence_source": _clean_text(metadata.get("evidence_source") or document.get("source")),
        "evidence_span": _clean_text(metadata.get("evidence_span")),
        "confidence": _json_safe(metadata.get("confidence")),
        "route_hints": tuple(_sequence(metadata.get("route_hints"))),
    }


def _record_from_template(
    template: Mapping[str, Any],
    *,
    decision: str,
    skip_reason: str | None = None,
    review_decision: Mapping[str, Any] | None = None,
    source_document_id: str | None = None,
) -> dict[str, Any]:
    review = review_decision or {}
    return {
        "record_type": "review_document",
        "alignment_candidate_id": template.get("alignment_candidate_id"),
        "decision": decision,
        "skip_reason": skip_reason,
        "source_document_id": source_document_id,
        "review_id": review.get("review_id"),
        "reviewer": review.get("reviewer"),
        "reviewed_at": review.get("reviewed_at"),
        "subject": template.get("subject"),
        "property_hint": template.get("property_hint"),
        "value": template.get("value"),
        "evidence_source": template.get("evidence_source"),
    }


def _approved_source_document(
    document: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    candidate_id: str,
    source_index: int,
) -> dict[str, Any] | None:
    metadata = _metadata(document)
    evidence_source = _clean_text(metadata.get("evidence_source") or document.get("source"))
    provider = _approved_provider(evidence_source=evidence_source, metadata=metadata)
    if provider is None:
        return None
    subject = _clean_text(metadata.get("subject"))
    value = _clean_text(document.get("answer"))
    property_hint = _clean_text(metadata.get("property_hint")) or ""
    property_label = _clean_text(metadata.get("statement_property_label")) or _property_label(property_hint)
    statement_property = _clean_text(metadata.get("statement_property")) or _property_id_or_hint(property_hint)
    if not subject or not value or not property_label or not statement_property:
        return None
    source_document_id = f"alignment-approved:{source_index}"
    review_id = _clean_text(decision.get("review_id"))
    source_metadata = {
        "provider": provider,
        "source_family": _source_family_for_provider(provider, metadata),
        "source": evidence_source,
        "statement_property": statement_property,
        "statement_property_label": property_label,
        "subject": subject,
        "value": value,
        "alignment_candidate_id": candidate_id,
        "alignment_source_document_id": source_document_id,
        "review_id": review_id,
        "reviewer": _reviewer(decision),
        "reviewed_at": _clean_text(decision.get("reviewed_at")),
        "review_status": "approved",
        "evidence_span": _clean_text(metadata.get("evidence_span")),
        "confidence": _json_safe(metadata.get("confidence")),
    }
    structured_slots = _json_safe(metadata.get("structured_evidence_slots"))
    if isinstance(structured_slots, Mapping) and structured_slots:
        source_metadata["structured_evidence_slots"] = structured_slots
    return {
        "source": evidence_source,
        "provider": provider,
        "source_family": _source_family_for_provider(provider, metadata),
        "title": f"Reviewed alignment fact: {subject} {property_label}",
        "text": _clean_text(metadata.get("evidence_span")) or _clean_text(document.get("text")),
        "metadata": source_metadata,
    }


def _approved_provider(*, evidence_source: str | None, metadata: Mapping[str, Any]) -> str | None:
    if evidence_source and evidence_source.casefold().startswith("wikidata:"):
        return "wikidata"
    provider = _clean_text(metadata.get("provider"))
    if provider and provider.casefold() in {"wikidata", "worldbank"}:
        return provider.casefold()
    return None


def _source_family_for_provider(provider: str, metadata: Mapping[str, Any]) -> str:
    if provider == "wikidata":
        return "reference"
    if provider == "worldbank":
        return "official_statistics"
    return _clean_text(metadata.get("source_family")) or "unknown"


def _metadata(document: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = document.get("metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _normalize_decision(value: Any) -> str | None:
    normalized = normalize_claim_text(str(value or ""))
    if normalized in APPROVED_ALIASES:
        return "approved"
    if normalized in REJECTED_ALIASES:
        return "rejected"
    if normalized in NEEDS_MORE_EVIDENCE_ALIASES:
        return "needs_more_evidence"
    return None


def _reviewer(decision: Mapping[str, Any]) -> str | None:
    return _clean_text(
        decision.get("reviewer")
        or decision.get("reviewer_id")
        or decision.get("review_source")
    )


def _stable_review_id(decision: Mapping[str, Any], *, index: int) -> str:
    candidate_id = _clean_text(decision.get("alignment_candidate_id") or decision.get("candidate_id")) or ""
    payload = strict_json_dumps(
        {
            "candidate_id": candidate_id,
            "decision": _clean_text(decision.get("decision")),
            "index": index,
            "reviewer": _reviewer(decision),
        },
        sort_keys=True,
    )
    return "review:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _stable_document_id(document: Mapping[str, Any]) -> str:
    metadata = _metadata(document)
    payload = strict_json_dumps(
        {
            "question": document.get("question"),
            "answer": document.get("answer"),
            "source": document.get("source"),
            "subject": metadata.get("subject"),
            "property_hint": metadata.get("property_hint"),
        },
        sort_keys=True,
    )
    return "alignment-candidate:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _property_label(property_hint: str) -> str:
    label, _, _ = property_hint.partition(":")
    return label.replace("_", " ").strip() or property_hint


def _property_id_or_hint(property_hint: str) -> str:
    _, sep, property_id = property_hint.partition(":")
    return property_id.strip() if sep and property_id.strip() else property_hint


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _non_string_sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): item
            for key, item in ((key, _json_safe(item)) for key, item in value.items())
            if item is not None
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in (_json_safe(item) for item in value) if item is not None)
    return str(value)


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


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(strict_json_dumps(row, sort_keys=True) + "\n")


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
    parser.add_argument("--review-corpus", required=True)
    parser.add_argument("--review-decisions", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--source-documents-json", default=None)
    parser.add_argument("--template-jsonl", default=None)
    parser.add_argument("--records-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        review_corpus_path=args.review_corpus,
        review_decisions_path=args.review_decisions,
        output_dir=args.output_dir,
        report_json_path=args.report_json,
        source_documents_json_path=args.source_documents_json,
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
        "alignment_fact_review_promotion_gate_ok "
        f"status={payload['status']} "
        f"review_documents={payload['summary']['review_document_count']} "
        f"approved_sources={payload['summary']['approved_source_document_count']}"
    )


if __name__ == "__main__":
    main()
