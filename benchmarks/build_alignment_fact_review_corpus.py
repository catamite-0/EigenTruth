"""Build a review-gated structured QA corpus from alignment fact candidates.

This is the conservative bridge after ``audit_blind_spot_alignment_requests.py``.
It deduplicates ``structured_fact_review_only`` candidates, applies small
quality gates, and materializes review-only QA rows. The emitted corpus is not a
verifier-evidence promotion; it is an input for human or later route-specific
fact review.
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
from eigentruth.verify import normalize_claim_text  # noqa: E402

WORKFLOW = "alignment_fact_review_corpus_builder"
CORPUS_TYPE = "alignment_structured_fact_review_corpus"
URL_RE = re.compile(r"https?://[^\s)]+")

REQUIRED_FIELDS = (
    "subject",
    "property_hint",
    "value",
    "evidence_span",
    "evidence_source",
)

RESERVED_DOCUMENT_METADATA_KEYS = {
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

GENERIC_VALUES = {
    "a",
    "an",
    "it",
    "no",
    "none",
    "according",
    "official usda ers",
    "population",
    "source",
    "the",
    "this",
    "unknown",
    "yes",
}


def build_alignment_fact_review_corpus(
    candidates: Sequence[Mapping[str, Any]],
    *,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    """Return a review-only QA corpus and per-candidate decision records."""
    if not 0.0 <= float(min_confidence) <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1.")
    documents: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    seen_fact_keys: set[tuple[str, str, str, str]] = set()
    skipped = Counter({
        "answer_value_matches_model_answer": 0,
        "duplicate_candidate": 0,
        "generic_value": 0,
        "invalid_row": 0,
        "low_confidence": 0,
        "missing_required_fields": 0,
        "non_review_usage": 0,
        "property_value_invalid": 0,
        "subject_not_in_evidence_span": 0,
        "value_equals_subject": 0,
        "value_not_in_evidence_span": 0,
    })

    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, Mapping):
            skipped["invalid_row"] += 1
            records.append(_decision_record(index, {}, decision="skipped", skip_reason="invalid_row"))
            continue
        item = _candidate_with_review_value(candidate)
        skip_reason = _skip_reason(
            item,
            seen_candidate_ids=seen_candidate_ids,
            seen_fact_keys=seen_fact_keys,
            min_confidence=float(min_confidence),
        )
        if skip_reason is not None:
            skipped[skip_reason] += 1
            records.append(_decision_record(index, item, decision="skipped", skip_reason=skip_reason))
            continue
        seen_candidate_ids.add(_clean_text(item.get("candidate_id")) or _stable_fact_key(item))
        fact_key = _fact_key(item)
        seen_fact_keys.add(fact_key)
        document = _document_from_candidate(item, document_index=len(documents) + 1)
        documents.append(document)
        records.append(
            _decision_record(
                index,
                item,
                decision="accepted",
                document_id=str(document["metadata"]["alignment_review_document_id"]),
            )
        )

    by_property = Counter(str(doc["metadata"].get("property_hint", "unknown")) for doc in documents)
    by_provider = Counter(str(doc["metadata"].get("provider", "unknown")) for doc in documents)
    by_source_family = Counter(str(doc["metadata"].get("source_family", "unknown")) for doc in documents)
    unique_candidate_count = len({
        _clean_text(item.get("candidate_id")) or _stable_fact_key(item)
        for item in candidates
        if isinstance(item, Mapping)
    })
    status = "ready_for_review" if documents else "blocked"
    summary = {
        "input_candidate_count": len(candidates),
        "unique_candidate_count": unique_candidate_count,
        "accepted_document_count": len(documents),
        "skipped_count": sum(skipped.values()),
        "skipped": dict(sorted(skipped.items())),
        "by_property_hint": dict(sorted(by_property.items())),
        "by_provider": dict(sorted(by_provider.items())),
        "by_source_family": dict(sorted(by_source_family.items())),
    }
    corpus = {
        "schema_version": 1,
        "corpus_type": CORPUS_TYPE,
        "status": status,
        "description": (
            "Review-gated structured QA rows derived from alignment-audit "
            "fact candidates. These rows are candidate fact-review inputs only; "
            "they are not verifier evidence or broad route promotion."
        ),
        "label_usage": {
            "labels_used_for_documents": False,
            "labels_copied_to_document_metadata": False,
            "request_ids_copied_to_document_metadata": False,
            "target_ids_copied_to_document_metadata": False,
            "model_answers_copied_to_document_metadata": False,
            "candidate_facts_are_verifier_evidence": False,
        },
        "source": {
            "builder": WORKFLOW,
            "accepted_usage": "structured_fact_review_only",
            "min_confidence": float(min_confidence),
        },
        "summary": summary,
        "documents": documents,
    }
    return {"corpus": corpus, "records": tuple(records)}


def run(
    *,
    candidates_path: str | Path,
    output_path: str | Path,
    report_json_path: str | Path | None = None,
    records_jsonl_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    min_confidence: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register a review corpus."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_path)
    report_path = Path(report_json_path) if report_json_path is not None else output.with_suffix(".report.json")
    records_path = Path(records_jsonl_path) if records_jsonl_path is not None else output.with_suffix(".records.jsonl")
    manifest_path = (
        Path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else output.parent / "artifact-manifest.json"
    )
    candidates = load_candidates(candidates_path)
    payload = build_alignment_fact_review_corpus(candidates, min_confidence=min_confidence)
    corpus = payload["corpus"]
    records = payload["records"]
    status = str(corpus["status"])
    report = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Converts alignment-audit structured_fact_review_only candidates "
            "into review-gated QA rows. It does not promote verifier evidence."
        ),
        "summary": dict(corpus["summary"]),
        "source": {
            "structured_fact_candidates": str(candidates_path),
        },
        "config": {
            "min_confidence": float(min_confidence),
        },
        "paths": {
            "review_corpus": str(output),
            "report": str(report_path),
            "records_jsonl": str(records_path),
            "artifact_manifest": str(manifest_path),
        },
        "metadata": dict(metadata or {}),
    }
    _write_json(output, corpus, compact=compact_json)
    _write_json(report_path, report, compact=compact_json)
    _write_jsonl(records_path, records)
    manifest = build_artifact_manifest(
        {
            "alignment_fact_review_corpus": output,
            "alignment_fact_review_report": report_path,
            "alignment_fact_review_records": records_path,
            "structured_fact_candidates": candidates_path,
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": status,
            "corpus_type": CORPUS_TYPE,
            "document_count": corpus["summary"]["accepted_document_count"],
            "input_candidate_count": corpus["summary"]["input_candidate_count"],
            "unique_candidate_count": corpus["summary"]["unique_candidate_count"],
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
                "status": status,
                "corpus_type": CORPUS_TYPE,
                "document_count": corpus["summary"]["accepted_document_count"],
                "input_candidate_count": corpus["summary"]["input_candidate_count"],
                "unique_candidate_count": corpus["summary"]["unique_candidate_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return {"corpus": corpus, "report": report, "records": records}


def load_candidates(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load alignment fact candidates from JSONL or an audit JSON report."""
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        return tuple(dict(item) for item in _load_jsonl(source) if isinstance(item, Mapping))
    payload = _load_json(source)
    if isinstance(payload, Mapping):
        for key in ("fact_candidates", "candidates", "records"):
            values = _non_string_sequence(payload.get(key))
            if values is not None:
                return tuple(dict(item) for item in values if isinstance(item, Mapping))
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return tuple(dict(item) for item in payload if isinstance(item, Mapping))
    raise ValueError("candidates must be JSONL, JSON list, or JSON object with fact_candidates.")


def _skip_reason(
    candidate: Mapping[str, Any],
    *,
    seen_candidate_ids: set[str],
    seen_fact_keys: set[tuple[str, str, str, str]],
    min_confidence: float,
) -> str | None:
    if str(candidate.get("usage") or "") != "structured_fact_review_only":
        return "non_review_usage"
    if any(not _clean_text(candidate.get(field)) for field in REQUIRED_FIELDS):
        return "missing_required_fields"
    candidate_id = _clean_text(candidate.get("candidate_id")) or _stable_fact_key(candidate)
    fact_key = _fact_key(candidate)
    if candidate_id in seen_candidate_ids or fact_key in seen_fact_keys:
        return "duplicate_candidate"
    if _float(candidate.get("confidence"), default=1.0) < min_confidence:
        return "low_confidence"
    subject = _clean_text(candidate.get("subject")) or ""
    value = _clean_text(candidate.get("value")) or ""
    raw_value = _clean_text(candidate.get("raw_value")) or value
    model_answer = _clean_text(candidate.get("model_answer")) or ""
    evidence_span = _clean_text(candidate.get("evidence_span")) or ""
    if _norm(value) in GENERIC_VALUES:
        return "generic_value"
    if _norm(value) == _norm(subject):
        return "value_equals_subject"
    if model_answer and (_norm(value) == _norm(model_answer) or _norm(raw_value) == _norm(model_answer)):
        return "answer_value_matches_model_answer"
    if _norm(subject) not in _norm(evidence_span):
        return "subject_not_in_evidence_span"
    if _norm(value) not in _norm(evidence_span):
        return "value_not_in_evidence_span"
    _, property_id = _property_parts(_clean_text(candidate.get("property_hint")) or "")
    if property_id == "P856" and not _is_url(value):
        return "property_value_invalid"
    return None


def _candidate_with_review_value(candidate: Mapping[str, Any]) -> dict[str, Any]:
    item = dict(candidate)
    raw_value = _clean_text(item.get("value"))
    evidence_span = _clean_text(item.get("evidence_span")) or ""
    property_hint = _clean_text(item.get("property_hint")) or ""
    property_label, property_id = _property_parts(property_hint)
    resolved_value = None
    if property_id == "P856":
        resolved_value = _extract_url(evidence_span)
    if resolved_value is None and property_label:
        resolved_value = _extract_has_property_value(evidence_span, property_label)
    if resolved_value is None and property_label == "description":
        resolved_value = _extract_description_value(evidence_span)
    if resolved_value:
        item["raw_value"] = raw_value
        item["value"] = resolved_value
    return item


def _document_from_candidate(candidate: Mapping[str, Any], *, document_index: int) -> dict[str, Any]:
    subject = _clean_text(candidate.get("subject")) or ""
    answer = _clean_text(candidate.get("value")) or ""
    property_hint = _clean_text(candidate.get("property_hint")) or ""
    property_label, property_id = _property_parts(property_hint)
    question = f"What does the aligned evidence say is the {property_label} for {subject}?"
    metadata = {
        "alignment_review_document_id": f"alignment-review:{document_index}",
        "alignment_candidate_id": _clean_text(candidate.get("candidate_id")),
        "provider": _clean_text(candidate.get("provider")) or "unknown",
        "source_family": _clean_text(candidate.get("source_family")) or "unknown",
        "evidence_source": _clean_text(candidate.get("evidence_source")),
        "evidence_span": _clean_text(candidate.get("evidence_span")),
        "property_hint": property_hint,
        "statement_property": property_id or property_hint,
        "statement_property_label": property_label,
        "subject": subject,
        "confidence": _float(candidate.get("confidence"), default=0.0),
        "review_required": True,
        "usage": "alignment_fact_review_only",
        "route_hints": ("structured_qa", "alignment_fact_review"),
        "fact_status": "candidate_review_required",
    }
    document_metadata = {
        key: _json_safe(value)
        for key, value in metadata.items()
        if key not in RESERVED_DOCUMENT_METADATA_KEYS and _json_safe(value) is not None
    }
    return {
        "question": question,
        "answer": answer,
        "text": f"{question} {answer}",
        "source": _clean_text(candidate.get("evidence_source")),
        "metadata": document_metadata,
    }


def _decision_record(
    index: int,
    candidate: Mapping[str, Any],
    *,
    decision: str,
    skip_reason: str | None = None,
    document_id: str | None = None,
) -> dict[str, Any]:
    return {
        "record_index": index,
        "decision": decision,
        "skip_reason": skip_reason,
        "document_id": document_id,
        "candidate_id": _clean_text(candidate.get("candidate_id")),
        "request_id": _clean_text(candidate.get("request_id")),
        "target_id": _clean_text(candidate.get("target_id")),
        "subject": _clean_text(candidate.get("subject")),
        "property_hint": _clean_text(candidate.get("property_hint")),
        "raw_value": _clean_text(candidate.get("raw_value")),
        "value": _clean_text(candidate.get("value")),
        "model_answer": _clean_text(candidate.get("model_answer")),
        "question": _clean_text(candidate.get("question")),
        "evidence_source": _clean_text(candidate.get("evidence_source")),
        "provider": _clean_text(candidate.get("provider")),
        "source_family": _clean_text(candidate.get("source_family")),
        "confidence": _json_safe(candidate.get("confidence")),
    }


def _fact_key(candidate: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        normalize_claim_text(str(candidate.get("subject", ""))),
        normalize_claim_text(str(candidate.get("property_hint", ""))),
        normalize_claim_text(str(candidate.get("value", ""))),
        normalize_claim_text(str(candidate.get("evidence_source", ""))),
    )


def _stable_fact_key(candidate: Mapping[str, Any]) -> str:
    return "|".join(_fact_key(candidate))


def _property_parts(property_hint: str) -> tuple[str, str | None]:
    label, sep, property_id = property_hint.partition(":")
    label = label.replace("_", " ").strip() or property_hint.strip()
    if sep and property_id.strip():
        return label, property_id.strip()
    return label, None


def _extract_url(text: str) -> str | None:
    match = URL_RE.search(text)
    if match is None:
        return None
    return match.group(0).rstrip(".,;")


def _extract_has_property_value(text: str, property_label: str) -> str | None:
    label_pattern = re.escape(property_label.replace("_", " ").strip())
    if not label_pattern:
        return None
    pattern = re.compile(rf"\bhas\s+{label_pattern}\s+(?P<value>[^.]+)", re.I)
    match = pattern.search(text)
    if match is None:
        return None
    return _clean_text(match.group("value"))


def _extract_description_value(text: str) -> str | None:
    match = re.search(r"\bis\s+described\s+as\s+(?P<value>[^.]+)", text, re.I)
    if match is None:
        return None
    return _clean_text(match.group("value"))


def _is_url(value: str) -> bool:
    return bool(URL_RE.fullmatch(value.strip().rstrip(".,;")))


def _norm(value: str) -> str:
    return normalize_claim_text(value)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _non_string_sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return None


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
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--records-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        candidates_path=args.candidates,
        output_path=args.output,
        report_json_path=args.report_json,
        records_jsonl_path=args.records_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        min_confidence=args.min_confidence,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    print(
        "alignment_fact_review_corpus_ok "
        f"status={payload['report']['status']} "
        f"documents={payload['corpus']['summary']['accepted_document_count']} "
        f"skipped={payload['corpus']['summary']['skipped_count']}"
    )


if __name__ == "__main__":
    main()
