"""Audit claim-specific bindings for citation/search source documents.

This workflow consumes the sanitized request JSONL and source-document JSONL
emitted by ``build_citation_search_adapter_handoff.py``. It does not use labels,
score rows, target ids, or model answers. The audit keeps only source documents
that can be bound back to the sanitized request and whose text/metadata aligns
with the request intent before those documents are treated as retrieval-corpus
candidates.
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

from benchmarks.build_external_retrieval_corpus import build_external_retrieval_corpus  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402
from eigentruth.verify import Claim, EvidenceAlignmentPolicy, audit_evidence_alignment  # noqa: E402

WORKFLOW = "citation_search_result_binding_audit"
DEFAULT_CORPUS_NAME = "claim_bound_citation_search"
DEFAULT_SOURCE_KIND = "claim_bound_citation_search_result"

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|thousand|%|percent))?", re.I)
NUMERIC_TERMS = {
    "amount",
    "count",
    "fewer",
    "greater",
    "how many",
    "less",
    "more",
    "number",
    "percent",
    "percentage",
    "population",
    "rate",
    "share",
    "total",
}
PERSON_RELATION_TERMS = {
    "author",
    "created",
    "creator",
    "founded",
    "founder",
    "invented",
    "inventor",
    "started",
}
LOCATION_TERMS = {"capital", "country", "located", "location", "place", "where"}
TEMPORAL_TERMS = {"date", "time", "when", "year"}
OFFICIAL_FAMILIES = {"official", "official_statistics", "domain_specific"}
SOURCE_FAMILY_COMPATIBILITY = {
    "official": {"official_statistics", "domain_specific"},
    "official_statistics": {"official", "domain_specific"},
    "reference": {"encyclopedic"},
    "encyclopedic": {"reference"},
}
RESERVED_METADATA_KEYS = {
    "answer",
    "claim_id",
    "is_false",
    "label",
    "labels",
    "model_answer",
    "record_index",
    "row_index",
    "score_label",
    "source_index",
    "target_id",
}


def audit_citation_search_result_bindings(
    *,
    requests: Sequence[Mapping[str, Any]],
    source_documents: Sequence[Mapping[str, Any]],
    min_keyword_overlap: float = 0.2,
    min_support_keyword_overlap: float = 0.65,
    min_entity_recall: float = 0.5,
    require_source_family_match: bool = False,
    require_freshness: bool = True,
    max_examples: int = 20,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a citation binding audit and accepted source documents."""
    if not 0.0 <= float(min_keyword_overlap) <= 1.0:
        raise ValueError("min_keyword_overlap must be in [0, 1].")
    if not 0.0 <= float(min_support_keyword_overlap) <= 1.0:
        raise ValueError("min_support_keyword_overlap must be in [0, 1].")
    if not 0.0 <= float(min_entity_recall) <= 1.0:
        raise ValueError("min_entity_recall must be in [0, 1].")
    if max_examples < 0:
        raise ValueError("max_examples cannot be negative.")

    request_index, duplicate_request_hashes = _request_index(requests)
    policy = EvidenceAlignmentPolicy(
        min_keyword_overlap=float(min_keyword_overlap),
        min_support_keyword_overlap=float(min_support_keyword_overlap),
        min_refute_keyword_overlap=0.5,
        min_number_recall=1.0,
        min_entity_recall=float(min_entity_recall),
    )
    records: list[dict[str, Any]] = []
    accepted_documents: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    request_counts: Counter[str] = Counter()

    for index, document in enumerate(source_documents, start=1):
        _reject_reserved_metadata(_mapping(document.get("metadata")), source=f"source_document:{index}")
        request, binding_failures = _request_for_document(document, request_index=request_index)
        if request is None:
            record = _binding_record(
                document=document,
                source_document_index=index,
                request=None,
                status="rejected",
                issue_codes=binding_failures,
                alignment_record={},
                intent={},
                source_family={},
            )
        else:
            record = _audit_document_binding(
                request=request,
                document=document,
                source_document_index=index,
                policy=policy,
                require_source_family_match=bool(require_source_family_match),
                require_freshness=bool(require_freshness),
                base_failures=binding_failures,
            )
        issue_counts.update(record["issue_codes"])
        if record["request_id"]:
            request_counts[record["request_id"]] += 1
        records.append(record)
        if record["status"] == "accepted":
            accepted_documents.append(_accepted_document(document, record=record))

    accepted_request_ids = {record["request_id"] for record in records if record["status"] == "accepted"}
    rejected_count = len(records) - len(accepted_documents)
    summary = {
        "request_count": len(requests),
        "source_document_count": len(source_documents),
        "accepted_source_document_count": len(accepted_documents),
        "rejected_source_document_count": rejected_count,
        "accepted_request_count": len(accepted_request_ids),
        "duplicate_request_hash_count": len(duplicate_request_hashes),
        "acceptance_rate": _safe_div(len(accepted_documents), len(source_documents)) or 0.0,
        "accepted_request_coverage": _safe_div(len(accepted_request_ids), len(requests)) or 0.0,
        "issue_counts": _sorted_counter(issue_counts),
        "documents_by_request": _sorted_counter(request_counts),
    }
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready" if accepted_documents else "blocked",
        "passed": bool(accepted_documents),
        "scope": (
            "Claim-specific citation/search binding audit. Accepted documents are "
            "still external-candidate evidence and must pass provenance and route gates."
        ),
        "label_usage": {
            "labels_used_for_binding": False,
            "labels_copied_to_binding_metadata": False,
            "model_answers_used_for_binding": False,
        },
        "config": {
            "min_keyword_overlap": float(min_keyword_overlap),
            "min_support_keyword_overlap": float(min_support_keyword_overlap),
            "min_entity_recall": float(min_entity_recall),
            "require_source_family_match": bool(require_source_family_match),
            "require_freshness": bool(require_freshness),
            "max_examples": int(max_examples),
        },
        "summary": summary,
        "records": tuple(records),
        "examples": tuple(records[: int(max_examples)]),
        "bound_source_documents": tuple(accepted_documents),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    requests_path: str | Path,
    source_documents_path: str | Path,
    report_json_path: str | Path,
    bound_source_documents_path: str | Path | None = None,
    bound_corpus_json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    corpus_name: str = DEFAULT_CORPUS_NAME,
    source_kind: str = DEFAULT_SOURCE_KIND,
    min_keyword_overlap: float = 0.2,
    min_support_keyword_overlap: float = 0.65,
    min_entity_recall: float = 0.5,
    require_source_family_match: bool = False,
    require_freshness: bool = True,
    max_examples: int = 20,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Run, write, optionally manifest, and optionally register the audit."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    requests = _load_jsonl(requests_path)
    source_documents = _load_jsonl(source_documents_path)
    payload = audit_citation_search_result_bindings(
        requests=requests,
        source_documents=source_documents,
        min_keyword_overlap=min_keyword_overlap,
        min_support_keyword_overlap=min_support_keyword_overlap,
        min_entity_recall=min_entity_recall,
        require_source_family_match=require_source_family_match,
        require_freshness=require_freshness,
        max_examples=max_examples,
        metadata=metadata,
    )
    report = dict(payload)
    bound_corpus = None
    bound_source_path = None if bound_source_documents_path is None else Path(bound_source_documents_path)
    if bound_source_path is not None:
        _write_jsonl(bound_source_path, payload["bound_source_documents"], compact=compact_json)
    if bound_corpus_json_path is not None and payload["bound_source_documents"]:
        if bound_source_path is None:
            raise ValueError("bound_corpus_json_path requires bound_source_documents_path.")
        bound_corpus = build_external_retrieval_corpus(
            (bound_source_path,),
            corpus_name=corpus_name,
            source_kind=source_kind,
            require_source=True,
        )
        _write_json(bound_corpus_json_path, bound_corpus, compact=compact_json)
        report["bound_corpus"] = bound_corpus
    report["paths"] = {
        "requests": str(requests_path),
        "source_documents": str(source_documents_path),
        "bound_source_documents": None if bound_source_path is None else str(bound_source_path),
        "bound_corpus": None if bound_corpus_json_path is None or bound_corpus is None else str(bound_corpus_json_path),
        "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
    }
    _write_json(report_json_path, report, compact=compact_json)

    if artifact_manifest_path is not None:
        artifacts: dict[str, str | Path | None] = {
            "citation_binding_audit": Path(report_json_path),
            "citation_requests": Path(requests_path),
            "citation_source_documents": Path(source_documents_path),
            "bound_source_documents": bound_source_path,
            "bound_corpus": None if bound_corpus is None else Path(bound_corpus_json_path),
        }
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "passed": payload["passed"],
                "request_count": payload["summary"]["request_count"],
                "source_document_count": payload["summary"]["source_document_count"],
                "accepted_source_document_count": payload["summary"]["accepted_source_document_count"],
                "accepted_request_count": payload["summary"]["accepted_request_count"],
                **dict(metadata or {}),
            },
        )
        _write_json(manifest_path, manifest, compact=compact_json)
        report["artifact_manifest"] = str(manifest_path)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_json_path,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "passed": payload["passed"],
                "request_count": payload["summary"]["request_count"],
                "source_document_count": payload["summary"]["source_document_count"],
                "accepted_source_document_count": payload["summary"]["accepted_source_document_count"],
                "accepted_request_count": payload["summary"]["accepted_request_count"],
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return report


def _audit_document_binding(
    *,
    request: Mapping[str, Any],
    document: Mapping[str, Any],
    source_document_index: int,
    policy: EvidenceAlignmentPolicy,
    require_source_family_match: bool,
    require_freshness: bool,
    base_failures: Sequence[str],
) -> dict[str, Any]:
    claim = Claim(
        text=_binding_claim_text(request),
        claim_id=str(request.get("request_id") or f"request-{source_document_index}"),
        metadata={"source": "citation_search_request"},
    )
    alignment = audit_evidence_alignment(claim, evidence=(document,), policy=policy)
    alignment_record = alignment.records[0].to_dict() if alignment.records else {}
    intent = _request_intent_match(request, document)
    source_family = _source_family_match(request, document)
    issue_codes = list(base_failures)
    alignment_status = str(alignment_record.get("status") or "")
    if alignment_status != "aligned":
        issue_codes.append(f"evidence_alignment_{alignment_status or 'not_applicable'}")
    if not bool(intent.get("match")):
        issue_codes.append(str(intent.get("reason") or "request_intent_mismatch"))
    if require_source_family_match and not bool(source_family.get("match")):
        issue_codes.append(str(source_family.get("reason") or "source_family_mismatch"))
    if require_freshness and bool(request.get("requires_timestamp")) and not _document_has_freshness(document):
        issue_codes.append("missing_fresh_timestamp")
    status = "accepted" if not issue_codes else "rejected"
    return _binding_record(
        document=document,
        source_document_index=source_document_index,
        request=request,
        status=status,
        issue_codes=tuple(dict.fromkeys(issue_codes)),
        alignment_record=alignment_record,
        intent=intent,
        source_family=source_family,
    )


def _binding_record(
    *,
    document: Mapping[str, Any],
    source_document_index: int,
    request: Mapping[str, Any] | None,
    status: str,
    issue_codes: Sequence[str],
    alignment_record: Mapping[str, Any],
    intent: Mapping[str, Any],
    source_family: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = _mapping(document.get("metadata"))
    request_metadata = _mapping(request.get("metadata")) if request is not None else {}
    return {
        "source_document_index": int(source_document_index),
        "status": status,
        "issue_codes": tuple(str(item) for item in issue_codes),
        "request_id": "" if request is None else str(request.get("request_id") or ""),
        "source": str(document.get("source") or ""),
        "source_queue_request_sha256": str(metadata.get("source_queue_request_sha256") or ""),
        "request_source_queue_request_sha256": str(request_metadata.get("source_queue_request_sha256") or ""),
        "query": "" if request is None else str(request.get("query") or ""),
        "question_type": "" if request is None else str(request.get("question_type") or ""),
        "requires_timestamp": bool(request.get("requires_timestamp")) if request is not None else False,
        "alignment": {
            "status": alignment_record.get("status"),
            "keyword_overlap": alignment_record.get("keyword_overlap"),
            "number_recall": alignment_record.get("number_recall"),
            "entity_recall": alignment_record.get("entity_recall"),
            "issue_codes": tuple(_sequence(alignment_record.get("issue_codes", ()))),
        },
        "intent": dict(intent),
        "source_family": dict(source_family),
    }


def _accepted_document(document: Mapping[str, Any], *, record: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(_mapping(document.get("metadata")))
    metadata.update({
        "citation_binding_audit": WORKFLOW,
        "citation_binding_status": "accepted",
        "citation_binding_request_id": record.get("request_id"),
        "citation_binding_intent_reason": _mapping(record.get("intent")).get("reason"),
        "citation_binding_alignment_status": _mapping(record.get("alignment")).get("status"),
        "citation_binding_source_family_status": _mapping(record.get("source_family")).get("reason"),
    })
    _reject_reserved_metadata(metadata, source=str(document.get("source") or "accepted_document"))
    return {
        **{key: value for key, value in document.items() if key != "metadata"},
        "metadata": metadata,
    }


def _request_for_document(
    document: Mapping[str, Any],
    *,
    request_index: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    metadata = _mapping(document.get("metadata"))
    source_hash = str(metadata.get("source_queue_request_sha256") or "").strip()
    if not source_hash:
        return None, ("missing_source_binding",)
    request = request_index.get(source_hash)
    if request is None:
        return None, ("unknown_source_binding",)
    return request, ()


def _request_index(requests: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]], tuple[str, ...]]:
    index: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for request in requests:
        _reject_reserved_metadata(_mapping(request.get("metadata")), source=str(request.get("request_id") or "request"))
        source_hash = str(_mapping(request.get("metadata")).get("source_queue_request_sha256") or "").strip()
        if not source_hash:
            continue
        if source_hash in index:
            duplicates.append(source_hash)
            continue
        index[source_hash] = request
    return index, tuple(duplicates)


def _binding_claim_text(request: Mapping[str, Any]) -> str:
    parts = [
        str(request.get("query") or "").strip(),
        *tuple(_string_sequence(request.get("alternate_queries", ()))),
    ]
    return " ".join(part for part in parts if part).strip()


def _request_intent_match(request: Mapping[str, Any], document: Mapping[str, Any]) -> dict[str, Any]:
    query = _binding_claim_text(request).casefold()
    question_type = str(request.get("question_type") or "").strip().casefold()
    evidence = _document_feature_text(document).casefold()
    property_text = _document_property_text(document).casefold()
    combined = f"{property_text} {evidence}".strip()
    query_tokens = set(_tokens(query))

    if question_type == "quantity" or _contains_any(query, NUMERIC_TERMS) or NUMBER_RE.search(query):
        if NUMBER_RE.search(combined) or _contains_any(property_text, NUMERIC_TERMS):
            return _intent(True, "numeric_intent_matched", ("numeric",))
        return _intent(False, "numeric_intent_requires_numeric_evidence", ("numeric",))

    if "why" in query_tokens or question_type == "causal":
        if _contains_any(combined, {"cause", "caused", "reason", "because", "etiology"}):
            return _intent(True, "causal_intent_matched", ("cause",))
        return _intent(False, "causal_intent_requires_causal_evidence", ("cause",))

    if "who" in query_tokens or question_type == "person":
        relation_terms = PERSON_RELATION_TERMS & query_tokens
        if relation_terms and _contains_any(combined, PERSON_RELATION_TERMS):
            return _intent(True, "person_relation_intent_matched", tuple(sorted(relation_terms)))
        if relation_terms:
            return _intent(False, "person_intent_requires_relation_evidence", tuple(sorted(relation_terms)))
        return _intent(True, "person_intent_no_relation_constraint", ("person",))

    if question_type == "location" or _contains_any(query, LOCATION_TERMS):
        if _contains_any(combined, LOCATION_TERMS):
            return _intent(True, "location_intent_matched", ("location",))
        return _intent(False, "location_intent_requires_location_evidence", ("location",))

    if question_type == "temporal" or _contains_any(query, TEMPORAL_TERMS):
        if NUMBER_RE.search(combined) or _contains_any(combined, TEMPORAL_TERMS):
            return _intent(True, "temporal_intent_matched", ("temporal",))
        return _intent(False, "temporal_intent_requires_temporal_evidence", ("temporal",))

    return _intent(True, "lexical_alignment_only", ())


def _source_family_match(request: Mapping[str, Any], document: Mapping[str, Any]) -> dict[str, Any]:
    planned = _planned_source_families(request)
    observed = _normalize_family(_mapping(document.get("metadata")).get("source_family"))
    if not planned:
        return {"match": True, "reason": "no_planned_source_family", "planned": (), "observed": observed}
    if not observed:
        return {"match": False, "reason": "missing_source_family", "planned": planned, "observed": ""}
    if observed in planned:
        return {"match": True, "reason": "source_family_exact_match", "planned": planned, "observed": observed}
    compatible = set(SOURCE_FAMILY_COMPATIBILITY.get(observed, set()))
    for family in planned:
        compatible |= SOURCE_FAMILY_COMPATIBILITY.get(family, set())
    if observed in compatible or any(family in compatible for family in planned):
        return {"match": True, "reason": "source_family_compatible_match", "planned": planned, "observed": observed}
    if _mapping(request.get("source_family_plan")).get("official_source_preferred") and observed in OFFICIAL_FAMILIES:
        return {
            "match": True,
            "reason": "source_family_official_preferred_match",
            "planned": planned,
            "observed": observed,
        }
    return {"match": False, "reason": "source_family_mismatch", "planned": planned, "observed": observed}


def _planned_source_families(request: Mapping[str, Any]) -> tuple[str, ...]:
    plan = _mapping(request.get("source_family_plan"))
    metadata = _mapping(request.get("metadata"))
    families = tuple(_string_sequence(plan.get("families", ()))) or tuple(
        _string_sequence(metadata.get("preferred_source_families", ()))
    )
    return tuple(dict.fromkeys(_normalize_family(item) for item in families if _normalize_family(item)))


def _document_has_freshness(document: Mapping[str, Any]) -> bool:
    metadata = _mapping(document.get("metadata"))
    return any(
        str(value or "").strip()
        for value in (
            metadata.get("published_at"),
            metadata.get("timestamp"),
            metadata.get("retrieved_at"),
            document.get("published_at"),
            document.get("timestamp"),
        )
    )


def _document_feature_text(document: Mapping[str, Any]) -> str:
    metadata = _mapping(document.get("metadata"))
    parts = [
        document.get("text"),
        document.get("content"),
        metadata.get("title"),
        metadata.get("snippet"),
        metadata.get("value"),
        metadata.get("fact_value"),
        metadata.get("object_text"),
    ]
    return " ".join(str(part).strip() for part in parts if str(part or "").strip())


def _document_property_text(document: Mapping[str, Any]) -> str:
    metadata = _mapping(document.get("metadata"))
    keys = (
        "property",
        "property_label",
        "predicate",
        "statement_property",
        "statement_property_label",
        "indicator",
        "indicator_name",
        "title",
    )
    return " ".join(str(metadata.get(key) or "").strip() for key in keys if str(metadata.get(key) or "").strip())


def _intent(match: bool, reason: str, terms: Sequence[str]) -> dict[str, Any]:
    return {"match": bool(match), "reason": reason, "terms": tuple(str(item) for item in terms)}


def _contains_any(text: str, terms: set[str]) -> bool:
    padded = f" {text.casefold()} "
    words = set(_tokens(text))
    for term in terms:
        normalized = term.casefold()
        if " " in normalized:
            if f" {normalized} " in padded:
                return True
        elif normalized in words:
            return True
    return False


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in TOKEN_RE.finditer(value))


def _normalize_family(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _reject_reserved_metadata(metadata: Mapping[str, Any], *, source: str) -> None:
    reserved = sorted(set(str(key) for key in metadata) & RESERVED_METADATA_KEYS)
    if reserved:
        raise ValueError(f"{source!r} contains reserved metadata keys: {', '.join(reserved)}")


def _load_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return tuple(rows)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) if compact else strict_json_dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
    output.write_text(text + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            if compact:
                handle.write(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            else:
                handle.write(strict_json_dumps(row, sort_keys=True) + "\n")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _string_sequence(value: Any) -> tuple[str, ...]:
    return tuple(str(item).strip() for item in _sequence(value) if str(item).strip())


def _safe_div(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(
        sorted(
            ((key, value) for key, value in counter.items() if key),
            key=lambda item: (-item[1], item[0]),
        )
    )


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
    parser.add_argument("--requests", required=True)
    parser.add_argument("--source-documents", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--bound-source-documents-jsonl", default=None)
    parser.add_argument("--bound-corpus-json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--corpus-name", default=DEFAULT_CORPUS_NAME)
    parser.add_argument("--source-kind", default=DEFAULT_SOURCE_KIND)
    parser.add_argument("--min-keyword-overlap", type=float, default=0.2)
    parser.add_argument("--min-support-keyword-overlap", type=float, default=0.65)
    parser.add_argument("--min-entity-recall", type=float, default=0.5)
    parser.add_argument("--require-source-family-match", action="store_true")
    parser.add_argument("--allow-missing-freshness", action="store_true")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        requests_path=args.requests,
        source_documents_path=args.source_documents,
        report_json_path=args.json,
        bound_source_documents_path=args.bound_source_documents_jsonl,
        bound_corpus_json_path=args.bound_corpus_json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        corpus_name=args.corpus_name,
        source_kind=args.source_kind,
        min_keyword_overlap=args.min_keyword_overlap,
        min_support_keyword_overlap=args.min_support_keyword_overlap,
        min_entity_recall=args.min_entity_recall,
        require_source_family_match=bool(args.require_source_family_match),
        require_freshness=not bool(args.allow_missing_freshness),
        max_examples=args.max_examples,
        metadata=_parse_metadata(args.metadata),
        compact_json=bool(args.compact_json),
    )
    print(
        "citation_search_result_binding_audit_ok "
        f"status={payload['status']} "
        f"accepted={payload['summary']['accepted_source_document_count']} "
        f"documents={payload['summary']['source_document_count']}"
    )


if __name__ == "__main__":
    main()
