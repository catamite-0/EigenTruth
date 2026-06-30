"""Rule-review alignment fact-review rows before promotion.

This is a route-specific reviewer for the review-only corpus emitted by
``build_alignment_fact_review_corpus.py``. It does not use model labels or
model answers. It only approves rows whose Wikidata evidence source and
evidence span close a conservative subject/property/value loop. All other rows
remain ``needs_more_evidence``.

The output decisions JSONL is intentionally compatible with
``promote_alignment_fact_review_corpus.py``.
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
from eigentruth.verify import normalize_claim_text  # noqa: E402

WORKFLOW = "alignment_fact_rule_review"
REVIEWER = "rule_based_alignment_fact_reviewer_v1"
REVIEW_CORPUS_TYPE = "alignment_structured_fact_review_corpus"
URL_RE = re.compile(r"https?://[^\s)]+", re.I)
QID_RE = re.compile(r"^Q\d+$")
GENERIC_VALUES = {
    "a",
    "an",
    "none",
    "source",
    "the",
    "unknown",
}
RESERVED_METADATA_KEYS = {
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


def review_alignment_fact_review_corpus(
    review_corpus: Mapping[str, Any],
    *,
    reviewer: str = REVIEWER,
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    """Return rule-review decisions and diagnostics for a review corpus."""
    documents = _review_documents(review_corpus)
    decisions: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    property_counts: Counter[str] = Counter()

    for index, document in enumerate(documents, start=1):
        candidate_id = _candidate_id(document, index=index)
        decision, reasons, checks = _review_document(document)
        decision_counts[decision] += 1
        for reason in reasons:
            reason_counts[reason] += 1
        if decision == "approved":
            property_counts[_statement_property(_metadata(document))] += 1
        review_id = _stable_review_id(candidate_id, decision=decision, reviewer=reviewer)
        notes = "; ".join(reasons) if reasons else "wikidata_subject_property_value_closed"
        decisions.append({
            "alignment_candidate_id": candidate_id,
            "decision": decision,
            "review_id": review_id,
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "notes": notes,
        })
        records.append({
            "record_index": index,
            "alignment_candidate_id": candidate_id,
            "decision": decision,
            "review_id": review_id,
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "notes": notes,
            "reasons": tuple(reasons),
            "checks": checks,
            "subject": _clean_text(_metadata(document).get("subject")),
            "property_hint": _clean_text(_metadata(document).get("property_hint")),
            "value": _clean_text(document.get("answer")),
            "evidence_source": _clean_text(_metadata(document).get("evidence_source") or document.get("source")),
        })

    status = "ready_for_promotion_gate" if decision_counts["approved"] else "needs_more_evidence"
    if not documents:
        status = "blocked"
    summary = {
        "review_document_count": len(documents),
        "approved_count": decision_counts["approved"],
        "needs_more_evidence_count": decision_counts["needs_more_evidence"],
        "decision_counts": dict(sorted(decision_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "approved_property_counts": dict(sorted(property_counts.items())),
    }
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Rule-based review of alignment fact-review rows. Approval means "
            "the row has a closed Wikidata subject/property/value evidence "
            "span, not broad open-domain route coverage."
        ),
        "label_usage": {
            "labels_used_for_review": False,
            "labels_copied_to_decisions": False,
            "request_ids_copied_to_decisions": False,
            "target_ids_copied_to_decisions": False,
            "model_answers_copied_to_decisions": False,
        },
        "source": {
            "review_corpus_type": review_corpus.get("corpus_type"),
            "review_corpus_status": review_corpus.get("status"),
        },
        "config": {
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "decision_output_schema": "promotion_gate_review_decisions",
        },
        "summary": summary,
        "decisions": tuple(decisions),
        "records": tuple(records),
    }


def run(
    *,
    review_corpus_path: str | Path,
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
    decisions_path = (
        Path(decisions_jsonl_path)
        if decisions_jsonl_path is not None
        else output / "review-decisions.jsonl"
    )
    report_path = Path(report_json_path) if report_json_path is not None else output / "review-report.json"
    records_path = Path(records_jsonl_path) if records_jsonl_path is not None else output / "review-records.jsonl"
    manifest_path = (
        Path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else output / "artifact-manifest.json"
    )

    review_corpus = _load_json_object(review_corpus_path)
    payload = review_alignment_fact_review_corpus(
        review_corpus,
        reviewer=reviewer,
        reviewed_at=reviewed_at,
    )
    payload = dict(payload)
    payload["paths"] = {
        "review_corpus": str(review_corpus_path),
        "review_decisions": str(decisions_path),
        "review_report": str(report_path),
        "review_records": str(records_path),
        "artifact_manifest": str(manifest_path),
    }
    payload["metadata"] = dict(metadata or {})

    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(decisions_path, payload["decisions"])
    _write_jsonl(records_path, payload["records"])
    manifest = build_artifact_manifest(
        {
            "alignment_fact_rule_review_report": report_path,
            "review_decisions": decisions_path,
            "review_records": records_path,
            "alignment_fact_review_corpus": review_corpus_path,
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "review_document_count": payload["summary"]["review_document_count"],
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
                "review_document_count": payload["summary"]["review_document_count"],
                "approved_count": payload["summary"]["approved_count"],
                "needs_more_evidence_count": payload["summary"]["needs_more_evidence_count"],
                "reviewer": reviewer,
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _review_document(document: Mapping[str, Any]) -> tuple[str, tuple[str, ...], dict[str, bool]]:
    metadata = _metadata(document)
    subject = _clean_text(metadata.get("subject"))
    value = _clean_text(document.get("answer"))
    evidence_span = _clean_text(metadata.get("evidence_span")) or _clean_text(document.get("text"))
    evidence_source = _clean_text(metadata.get("evidence_source") or document.get("source"))
    property_hint = _clean_text(metadata.get("property_hint")) or ""
    property_label = _clean_text(metadata.get("statement_property_label")) or _property_label(property_hint)
    property_id = _clean_text(metadata.get("statement_property")) or _property_id_or_hint(property_hint)
    normalized_span = normalize_claim_text(evidence_span or "")
    normalized_subject = normalize_claim_text(subject or "")
    normalized_value = normalize_claim_text(value or "")
    reasons: list[str] = []
    checks = {
        "has_candidate_id": bool(_clean_text(metadata.get("alignment_candidate_id"))),
        "no_reserved_metadata": not any(key in metadata for key in RESERVED_METADATA_KEYS),
        "wikidata_source": bool(evidence_source and evidence_source.casefold().startswith("wikidata:")),
        "has_subject": bool(subject),
        "has_value": bool(value),
        "subject_in_evidence": bool(normalized_subject and normalized_subject in normalized_span),
        "value_in_evidence": bool(normalized_value and normalized_value in normalized_span),
        "property_matches_source": _property_matches_source(property_id, evidence_source),
        "property_in_evidence": _property_in_evidence(property_label, property_id, evidence_span),
        "value_is_specific": _value_is_specific(value),
    }
    for name, passed in checks.items():
        if not passed:
            reasons.append(name)
    if reasons:
        return "needs_more_evidence", tuple(reasons), checks
    return "approved", (), checks


def _property_matches_source(property_id: str | None, evidence_source: str | None) -> bool:
    if not evidence_source:
        return False
    source = evidence_source.casefold()
    if property_id and property_id.casefold() == "description":
        return source.endswith(":description") or ":description" in source
    if property_id and property_id.casefold().startswith("p"):
        return f":{property_id.casefold()}:" in source or source.endswith(f":{property_id.casefold()}")
    return bool(property_id)


def _property_in_evidence(property_label: str | None, property_id: str | None, evidence_span: str | None) -> bool:
    if not evidence_span:
        return False
    span = evidence_span.casefold()
    label = (property_label or "").replace("_", " ").strip().casefold()
    prop = (property_id or "").casefold()
    if prop == "description" or label == "description":
        return "described as" in span
    if prop == "p856" or label == "official website":
        return bool(URL_RE.search(evidence_span))
    if label == "has part":
        return "has part" in span
    return bool(label and label in span)


def _value_is_specific(value: str | None) -> bool:
    if not value:
        return False
    normalized = normalize_claim_text(value)
    if normalized in GENERIC_VALUES:
        return False
    if QID_RE.fullmatch(value.strip()):
        return False
    return True


def _review_documents(review_corpus: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if review_corpus.get("corpus_type") not in {REVIEW_CORPUS_TYPE, None}:
        raise ValueError(f"review_corpus must have corpus_type={REVIEW_CORPUS_TYPE!r}.")
    raw_documents = review_corpus.get("documents")
    if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, (str, bytes, bytearray)):
        raise ValueError("review_corpus must contain a documents list.")
    return tuple(dict(item) for item in raw_documents if isinstance(item, Mapping))


def _candidate_id(document: Mapping[str, Any], *, index: int) -> str:
    metadata = _metadata(document)
    return _clean_text(metadata.get("alignment_candidate_id")) or _stable_candidate_id(document, index=index)


def _stable_candidate_id(document: Mapping[str, Any], *, index: int) -> str:
    metadata = _metadata(document)
    payload = strict_json_dumps(
        {
            "answer": document.get("answer"),
            "index": index,
            "property_hint": metadata.get("property_hint"),
            "source": document.get("source"),
            "subject": metadata.get("subject"),
        },
        sort_keys=True,
    )
    return "alignment-candidate:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _stable_review_id(candidate_id: str, *, decision: str, reviewer: str) -> str:
    payload = strict_json_dumps(
        {
            "candidate_id": candidate_id,
            "decision": decision,
            "reviewer": reviewer,
            "workflow": WORKFLOW,
        },
        sort_keys=True,
    )
    return "review:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _metadata(document: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = document.get("metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _statement_property(metadata: Mapping[str, Any]) -> str:
    return _clean_text(metadata.get("statement_property")) or _property_id_or_hint(
        _clean_text(metadata.get("property_hint")) or ""
    )


def _property_label(property_hint: str) -> str:
    label, _, _ = property_hint.partition(":")
    return label.replace("_", " ").strip() or property_hint


def _property_id_or_hint(property_hint: str) -> str:
    _, sep, property_id = property_hint.partition(":")
    return property_id.strip() if sep and property_id.strip() else property_hint


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = _load_json(Path(path))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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
    args = parser.parse_args(argv)
    payload = run(
        review_corpus_path=args.review_corpus,
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
        "alignment_fact_rule_review_ok "
        f"status={payload['status']} "
        f"review_documents={payload['summary']['review_document_count']} "
        f"approved={payload['summary']['approved_count']} "
        f"needs_more_evidence={payload['summary']['needs_more_evidence_count']}"
    )


if __name__ == "__main__":
    main()
