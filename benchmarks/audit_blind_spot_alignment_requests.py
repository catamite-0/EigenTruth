"""Audit blind-spot alignment requests against external evidence documents.

This workflow consumes ``alignment_audit`` requests from
``build_blind_spot_evidence_collection_corpus.py`` and an external retrieval
corpus. It does not promote verifier evidence. It surfaces whether each request
has enough subject/property/value alignment to become a structured-fact review
candidate, or whether the next step is query refinement, source extraction, or
manual disambiguation.
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

WORKFLOW = "blind_spot_alignment_audit"
DEFAULT_MAX_DOCS_PER_REQUEST = 5
DEFAULT_MIN_ALIGNMENT_SCORE = 0.12
TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|thousand|%|percent))?", re.I)
URL_RE = re.compile(r"https?://[^\s)]+")
CAPITALIZED_SPAN_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
DESCRIBED_AS_RE = re.compile(r"\b(?:is|are|was|were)\s+described\s+as\s+(?P<value>[^.]+)", re.I)
IS_A_RE = re.compile(r"\b(?:is|are|was|were)\s+(?:an?|the)\s+(?P<value>[^.]+)", re.I)
BY_VALUE_RE = re.compile(r"\b(?:founded|created|written|authored|produced)\s+by\s+(?P<value>[^.]+)", re.I)
GENERIC_VALUE_CANDIDATES = {
    "A",
    "An",
    "The",
    "It",
    "This",
    "That",
    "Source",
    "Indicator",
    "According",
    "OpenAlex",
    "Official USDA ERS",
    "Population",
    "World Bank",
}
GENERIC_VALUE_CANDIDATES_CASEFOLD = {item.casefold() for item in GENERIC_VALUE_CANDIDATES}
QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def audit_blind_spot_alignment_requests(
    *,
    collection_corpus: Mapping[str, Any],
    evidence_documents: Sequence[Mapping[str, Any]],
    max_docs_per_request: int = DEFAULT_MAX_DOCS_PER_REQUEST,
    min_alignment_score: float = DEFAULT_MIN_ALIGNMENT_SCORE,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready alignment audit report."""
    if int(max_docs_per_request) <= 0:
        raise ValueError("max_docs_per_request must be positive.")
    if not 0.0 <= float(min_alignment_score) <= 1.0:
        raise ValueError("min_alignment_score must be between 0 and 1.")
    requests = _alignment_requests(collection_corpus)
    docs = tuple(_normalize_document(item, ordinal=idx) for idx, item in enumerate(evidence_documents, start=1))
    if not docs:
        raise ValueError("evidence_documents must not be empty.")

    audit_records = tuple(
        _audit_request(
            request,
            evidence_documents=docs,
            max_docs=int(max_docs_per_request),
            min_alignment_score=float(min_alignment_score),
        )
        for request in requests
    )
    fact_candidates = tuple(
        candidate
        for record in audit_records
        for candidate in _sequence(record.get("fact_candidates"))
    )
    summary = _summary(
        audit_records=audit_records,
        fact_candidates=fact_candidates,
        evidence_documents=docs,
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready_for_fact_review" if fact_candidates else "needs_alignment_evidence",
        "scope": (
            "Audits alignment-audit requests against external source documents. "
            "Candidate facts are review inputs only and are not verifier evidence."
        ),
        "source": {
            "collection_corpus_workflow": collection_corpus.get("workflow"),
            "collection_corpus_status": collection_corpus.get("status"),
            "collection_target_count": _nested_int(collection_corpus, "summary", "target_count"),
            "alignment_request_count": len(requests),
            "evidence_document_count": len(docs),
        },
        "label_usage": {
            "labels_used_for_alignment": False,
            "labels_copied_to_alignment_records": False,
            "candidate_facts_are_verifier_evidence": False,
        },
        "config": {
            "max_docs_per_request": int(max_docs_per_request),
            "min_alignment_score": float(min_alignment_score),
        },
        "summary": summary,
        "audit_records": audit_records,
        "fact_candidates": fact_candidates,
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    collection_corpus_path: str | Path,
    evidence_corpus_path: str | Path,
    output_dir: str | Path,
    report_json_path: str | Path | None = None,
    records_jsonl_path: str | Path | None = None,
    fact_candidates_jsonl_path: str | Path | None = None,
    max_docs_per_request: int = DEFAULT_MAX_DOCS_PER_REQUEST,
    min_alignment_score: float = DEFAULT_MIN_ALIGNMENT_SCORE,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, optionally manifest, and optionally register an audit."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    report_path = Path(report_json_path) if report_json_path is not None else output / "alignment-audit.json"
    records_path = Path(records_jsonl_path) if records_jsonl_path is not None else output / "alignment-records.jsonl"
    candidates_path = (
        Path(fact_candidates_jsonl_path)
        if fact_candidates_jsonl_path is not None
        else output / "structured-fact-candidates.jsonl"
    )
    collection_corpus = _load_json_object(collection_corpus_path)
    evidence_documents = _load_evidence_documents(evidence_corpus_path)
    payload = audit_blind_spot_alignment_requests(
        collection_corpus=collection_corpus,
        evidence_documents=evidence_documents,
        max_docs_per_request=max_docs_per_request,
        min_alignment_score=min_alignment_score,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["paths"] = {
        "collection_corpus": str(collection_corpus_path),
        "evidence_corpus": str(evidence_corpus_path),
        "alignment_records_jsonl": str(records_path),
        "structured_fact_candidates_jsonl": str(candidates_path),
        "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
    }
    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(records_path, payload["audit_records"])
    _write_jsonl(candidates_path, payload["fact_candidates"])

    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            {
                "blind_spot_alignment_audit": report_path,
                "alignment_records": records_path,
                "structured_fact_candidates": candidates_path,
                "blind_spot_evidence_collection_corpus": collection_corpus_path,
                "evidence_corpus": evidence_corpus_path,
            },
            root=manifest_path.parent,
            metadata={
                "workflow": payload["workflow"],
                "status": payload["status"],
                "alignment_request_count": payload["summary"]["alignment_request_count"],
                "fact_candidate_count": payload["summary"]["fact_candidate_count"],
                "top_gap_reason": _first_key(payload["summary"]["gap_reason_counts"]),
                "top_alignment_status": _first_key(payload["summary"]["alignment_status_counts"]),
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
                "workflow": payload["workflow"],
                "status": payload["status"],
                "alignment_request_count": payload["summary"]["alignment_request_count"],
                "fact_candidate_count": payload["summary"]["fact_candidate_count"],
                "top_gap_reason": _first_key(payload["summary"]["gap_reason_counts"]),
                "top_alignment_status": _first_key(payload["summary"]["alignment_status_counts"]),
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _audit_request(
    request: Mapping[str, Any],
    *,
    evidence_documents: Sequence[Mapping[str, Any]],
    max_docs: int,
    min_alignment_score: float,
) -> dict[str, Any]:
    scored = tuple(
        hit
        for hit in (
            _score_document(request, document, min_alignment_score=min_alignment_score)
            for document in evidence_documents
        )
        if hit is not None
    )
    ranked = tuple(
        sorted(scored, key=lambda item: (-float(item["alignment_score"]), str(item["source"])))[:max_docs]
    )
    best = ranked[0] if ranked else None
    gap_reason = _gap_reason(best, request=request)
    status = _alignment_status(gap_reason)
    candidates = tuple(_fact_candidates(request, ranked, status=status))
    return {
        "request_id": str(request.get("request_id", "")),
        "target_id": str(request.get("target_id", "")),
        "question": str(request.get("question", "")),
        "model_answer": str(request.get("model_answer", "")),
        "question_type": str(request.get("question_type", "")),
        "alignment_status": status,
        "gap_reason": gap_reason,
        "alignment_actions": tuple(str(item) for item in _sequence(request.get("alignment_actions"))),
        "dominant_gap_bucket": request.get("dominant_gap_bucket"),
        "query_sweep_best_strategy": request.get("query_sweep_best_strategy"),
        "entity_candidates": tuple(str(item) for item in _sequence(request.get("entity_candidates"))),
        "wikidata_property_hints": tuple(str(item) for item in _sequence(request.get("wikidata_property_hints"))),
        "top_evidence_hits": ranked,
        "fact_candidates": candidates,
        "query_refinement_suggestions": _query_refinement_suggestions(request, gap_reason=gap_reason),
    }


def _score_document(
    request: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    min_alignment_score: float,
) -> dict[str, Any] | None:
    doc_text = str(document.get("text", ""))
    metadata = _mapping(document.get("metadata"))
    doc_blob = " ".join((
        doc_text,
        str(metadata.get("title", "")),
        str(document.get("source", "")),
    ))
    doc_tokens = _tokens(doc_blob)
    if not doc_tokens:
        return None
    query_tokens = _request_tokens(request)
    if not query_tokens:
        return None
    query_overlap = _overlap(query_tokens, doc_tokens)
    entity = _matched_phrase(doc_blob, _sequence(request.get("entity_candidates")))
    property_hint = _matched_property(doc_blob, _sequence(request.get("wikidata_property_hints")))
    answer_value = _matched_answer_value(doc_blob, str(request.get("model_answer", "")))
    score = min(1.0, query_overlap + (0.18 if entity else 0.0) + (0.16 if property_hint else 0.0))
    if score < min_alignment_score:
        return None
    span = _best_span(request, doc_text)
    return {
        "source": str(document.get("source", "")),
        "title": str(metadata.get("title", "")),
        "url": metadata.get("url"),
        "provider": metadata.get("provider"),
        "source_family": metadata.get("source_family"),
        "alignment_score": round(score, 6),
        "query_overlap": round(query_overlap, 6),
        "matched_entity": entity,
        "matched_property_hint": property_hint,
        "model_answer_value_matched": answer_value,
        "evidence_span": span,
    }


def _fact_candidates(
    request: Mapping[str, Any],
    hits: Sequence[Mapping[str, Any]],
    *,
    status: str,
) -> tuple[dict[str, Any], ...]:
    if status != "candidate_fact_ready":
        return ()
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    request_id = str(request.get("request_id", ""))
    subject = _first_nonempty(tuple(str(item) for item in _sequence(request.get("entity_candidates"))))
    for hit in hits:
        property_hint = str(hit.get("matched_property_hint") or "")
        if not subject or not property_hint:
            continue
        value = _candidate_value(str(hit.get("evidence_span", "")), request=request, property_hint=property_hint)
        if not value:
            continue
        fact_key = (
            subject.casefold(),
            property_hint.casefold(),
            value.casefold(),
            str(hit.get("source", "")).casefold(),
        )
        if fact_key in seen:
            continue
        seen.add(fact_key)
        candidate_id = _stable_id(request_id, subject, property_hint, value, str(hit.get("source", "")))
        candidates.append({
            "candidate_id": f"fact:{candidate_id}",
            "request_id": request_id,
            "target_id": str(request.get("target_id", "")),
            "subject": subject,
            "property_hint": property_hint,
            "value": value,
            "model_answer": str(request.get("model_answer", "")),
            "question": str(request.get("question", "")),
            "evidence_span": str(hit.get("evidence_span", "")),
            "evidence_source": str(hit.get("source", "")),
            "source_family": hit.get("source_family"),
            "provider": hit.get("provider"),
            "confidence": _candidate_confidence(hit),
            "usage": "structured_fact_review_only",
        })
        if len(candidates) >= 3:
            break
    return tuple(candidates)


def _gap_reason(best_hit: Mapping[str, Any] | None, *, request: Mapping[str, Any]) -> str:
    if best_hit is None:
        return "no_candidate_evidence"
    entity = bool(best_hit.get("matched_entity"))
    prop = bool(best_hit.get("matched_property_hint"))
    value = bool(
        _candidate_value(
            str(best_hit.get("evidence_span", "")),
            request=request,
            property_hint=str(best_hit.get("matched_property_hint") or ""),
        )
    )
    if entity and prop and value:
        return "subject_property_value_aligned"
    if entity and prop:
        return "subject_property_aligned_no_value"
    if entity:
        return "subject_only_alignment"
    if prop:
        return "property_only_alignment"
    return "broad_source_no_subject_property_alignment"


def _alignment_status(gap_reason: str) -> str:
    if gap_reason == "subject_property_value_aligned":
        return "candidate_fact_ready"
    if gap_reason == "subject_property_aligned_no_value":
        return "needs_value_extraction"
    if gap_reason in {"subject_only_alignment", "property_only_alignment"}:
        return "needs_property_or_subject_alignment"
    if gap_reason == "no_candidate_evidence":
        return "needs_query_refinement"
    return "needs_source_document_fact_extraction"


def _summary(
    *,
    audit_records: Sequence[Mapping[str, Any]],
    fact_candidates: Sequence[Mapping[str, Any]],
    evidence_documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(str(item.get("alignment_status")) for item in audit_records)
    gap_counts = Counter(str(item.get("gap_reason")) for item in audit_records)
    family_counts = Counter()
    provider_counts = Counter()
    property_counts = Counter(str(item.get("property_hint")) for item in fact_candidates if item.get("property_hint"))
    requests_with_fact_candidates = len({
        str(item.get("request_id")) for item in fact_candidates if item.get("request_id")
    })
    aligned_with_hits = 0
    for record in audit_records:
        if _sequence(record.get("top_evidence_hits")):
            aligned_with_hits += 1
        for hit in _sequence(record.get("top_evidence_hits"))[:1]:
            if isinstance(hit, Mapping):
                if hit.get("source_family"):
                    family_counts[str(hit["source_family"])] += 1
                if hit.get("provider"):
                    provider_counts[str(hit["provider"])] += 1
    return {
        "alignment_request_count": len(audit_records),
        "evidence_document_count": len(evidence_documents),
        "requests_with_candidate_hits": aligned_with_hits,
        "requests_with_candidate_hit_rate": _rate(aligned_with_hits, len(audit_records)),
        "fact_candidate_count": len(fact_candidates),
        "fact_candidates_per_request": _rate(len(fact_candidates), len(audit_records)),
        "requests_with_fact_candidate_count": requests_with_fact_candidates,
        "requests_with_fact_candidate_rate": _rate(requests_with_fact_candidates, len(audit_records)),
        "alignment_status_counts": _sorted_counter(status_counts),
        "gap_reason_counts": _sorted_counter(gap_counts),
        "top_source_family_counts": _sorted_counter(family_counts),
        "top_provider_counts": _sorted_counter(provider_counts),
        "fact_candidate_property_counts": _sorted_counter(property_counts),
    }


def _alignment_requests(collection_corpus: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if collection_corpus.get("workflow") != "blind_spot_evidence_collection_corpus":
        raise ValueError("collection corpus must have workflow blind_spot_evidence_collection_corpus.")
    requests = _mapping(collection_corpus.get("requests"))
    alignment = tuple(
        dict(item)
        for item in _sequence(requests.get("alignment_audit"))
        if isinstance(item, Mapping)
    )
    if not alignment:
        raise ValueError("collection corpus has no alignment_audit requests.")
    return alignment


def _load_evidence_documents(path: str | Path) -> tuple[dict[str, Any], ...]:
    source = Path(path)
    if source.suffix == ".jsonl":
        rows = []
        with source.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"{source} line {line_no} is not a JSON object.")
                rows.append(dict(payload))
        return tuple(rows)
    payload = _load_json_object(source)
    docs = payload.get("documents")
    if isinstance(docs, Sequence) and not isinstance(docs, (str, bytes, bytearray)):
        return tuple(dict(item) for item in docs if isinstance(item, Mapping))
    raise ValueError("evidence corpus must be JSON with documents or JSONL rows.")


def _normalize_document(document: Mapping[str, Any], *, ordinal: int) -> dict[str, Any]:
    return {
        "source": str(document.get("source") or f"document-{ordinal}"),
        "text": str(document.get("text", "")),
        "metadata": dict(_mapping(document.get("metadata"))),
    }


def _request_tokens(request: Mapping[str, Any]) -> tuple[str, ...]:
    pieces = [
        str(request.get("question", "")),
        " ".join(_sequence(request.get("entity_candidates"))),
        " ".join(_property_label(str(item)) for item in _sequence(request.get("wikidata_property_hints"))),
    ]
    tokens = [
        token
        for token in _tokens(" ".join(pieces))
        if token not in QUESTION_STOPWORDS
    ]
    return tuple(dict.fromkeys(tokens))


def _best_span(request: Mapping[str, Any], text: str) -> str:
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if not sentences:
        return text[:500]
    query_tokens = _request_tokens(request)
    best = max(sentences, key=lambda item: _overlap(query_tokens, _tokens(item)))
    return best[:700]


def _matched_phrase(text: str, candidates: Sequence[Any]) -> str | None:
    lowered = text.casefold()
    for item in candidates:
        phrase = str(item).strip()
        if phrase and phrase.casefold() in lowered:
            return phrase
    return None


def _matched_property(text: str, hints: Sequence[Any]) -> str | None:
    text_tokens = set(_tokens(text))
    for hint in hints:
        raw = str(hint).strip()
        if not raw:
            continue
        property_tokens = tuple(token for token in _tokens(_property_label(raw)) if token not in QUESTION_STOPWORDS)
        if property_tokens and set(property_tokens).intersection(text_tokens):
            return raw
    return None


def _matched_answer_value(text: str, answer: str) -> bool:
    answer = answer.strip(" \t\r\n.,;:!?")
    return bool(answer) and answer.casefold() in text.casefold()


def _candidate_value(
    span: str,
    *,
    request: Mapping[str, Any],
    property_hint: str | None = None,
) -> str | None:
    qtype = str(request.get("question_type", "")).casefold()
    property_text = (
        property_hint
        if property_hint
        else " ".join(str(item) for item in _sequence(request.get("wikidata_property_hints")))
    ).casefold()
    property_label, property_id = _property_parts(property_hint or "")
    if property_id == "P856" or property_label == "official website":
        url = _extract_url(span)
        if url:
            return url
    if property_label:
        value = _extract_has_property_value(span, property_label)
        if value:
            return value
    if property_label == "description":
        value = _extract_description_value(span)
        if value:
            return value
    quantity_tokens = ("population", "area", "height", "mass", "point_in_time")
    if qtype == "quantity" or any(token in property_text for token in quantity_tokens):
        match = NUMBER_RE.search(span)
        return None if match is None else match.group(0).strip()
    answer = str(request.get("model_answer", "")).strip(" \t\r\n.,;:!?")
    if answer and answer.casefold() in span.casefold():
        return answer
    for pattern in (DESCRIBED_AS_RE, IS_A_RE, BY_VALUE_RE):
        match = pattern.search(span)
        if match is not None:
            value = _clean_candidate_value(match.group("value"))
            if value:
                return value
    entity_values = {str(item).casefold() for item in _sequence(request.get("entity_candidates"))}
    for match in CAPITALIZED_SPAN_RE.finditer(span):
        candidate = match.group(0).strip()
        if _is_generic_value(candidate):
            continue
        if candidate.casefold() in entity_values:
            continue
        return candidate
    return None


def _clean_candidate_value(value: str) -> str:
    value = re.split(r"\b(?:according to|source:|publisher:|doi:|retrieved at)\b", value, maxsplit=1, flags=re.I)[0]
    value = value.strip(" \t\r\n.,;:!?\"'")
    value = re.sub(r"\s+", " ", value)
    words = value.split()
    if len(words) > 12:
        value = " ".join(words[:12])
    if not value or _is_generic_value(value):
        return ""
    return value


def _property_parts(property_hint: str) -> tuple[str, str | None]:
    label, sep, property_id = str(property_hint).partition(":")
    label = label.replace("_", " ").strip().casefold()
    if sep and property_id.strip():
        return label, property_id.strip()
    return label, None


def _extract_url(text: str) -> str | None:
    match = URL_RE.search(text)
    if match is None:
        return None
    return match.group(0).rstrip(".,;")


def _extract_has_property_value(text: str, property_label: str) -> str | None:
    label = property_label.replace("_", " ").strip()
    if not label:
        return None
    pattern = re.compile(rf"\bhas\s+{re.escape(label)}\s+(?P<value>[^.]+)", re.I)
    match = pattern.search(text)
    if match is None:
        return None
    return _clean_candidate_value(match.group("value"))


def _extract_description_value(text: str) -> str | None:
    match = DESCRIBED_AS_RE.search(text)
    if match is None:
        return None
    return _clean_candidate_value(match.group("value"))


def _is_generic_value(value: str) -> bool:
    return value.strip().casefold() in GENERIC_VALUE_CANDIDATES_CASEFOLD


def _query_refinement_suggestions(request: Mapping[str, Any], *, gap_reason: str) -> tuple[str, ...]:
    if gap_reason not in {
        "no_candidate_evidence",
        "broad_source_no_subject_property_alignment",
        "subject_only_alignment",
        "property_only_alignment",
    }:
        return ()
    entities = tuple(str(item) for item in _sequence(request.get("entity_candidates")) if str(item).strip())
    properties = tuple(_property_label(str(item)) for item in _sequence(request.get("wikidata_property_hints")))
    question = str(request.get("question", "")).strip()
    suggestions = []
    for entity in entities[:2]:
        for prop in properties[:2]:
            suggestions.append(" ".join(part for part in (entity, prop, question) if part).strip())
    return tuple(dict.fromkeys(suggestions))[:4]


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in TOKEN_RE.finditer(str(text)))


def _overlap(query_tokens: Sequence[str], evidence_tokens: Sequence[str]) -> float:
    if not query_tokens:
        return 0.0
    evidence = set(evidence_tokens)
    return sum(1 for token in dict.fromkeys(query_tokens) if token in evidence) / len(set(query_tokens))


def _property_label(hint: str) -> str:
    return hint.split(":", 1)[0].replace("_", " ").strip()


def _candidate_confidence(hit: Mapping[str, Any]) -> float:
    score = float(hit.get("alignment_score", 0.0))
    value_boost = 0.05 if hit.get("model_answer_value_matched") else 0.0
    return round(min(0.99, 0.35 + score + value_boost), 6)


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha256("\u241f".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def _first_nonempty(values: Sequence[str]) -> str:
    return next((value for value in values if value.strip()), "")


def _load_json_object(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(data)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(strict_json_dumps(record, sort_keys=True) + "\n")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


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


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _first_key(mapping: Mapping[str, Any]) -> str | None:
    return next(iter(mapping), None)


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
    parser.add_argument("--collection-corpus", required=True)
    parser.add_argument("--evidence-corpus", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--records-jsonl", default=None)
    parser.add_argument("--fact-candidates-jsonl", default=None)
    parser.add_argument("--max-docs-per-request", type=int, default=DEFAULT_MAX_DOCS_PER_REQUEST)
    parser.add_argument("--min-alignment-score", type=float, default=DEFAULT_MIN_ALIGNMENT_SCORE)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        collection_corpus_path=args.collection_corpus,
        evidence_corpus_path=args.evidence_corpus,
        output_dir=args.output_dir,
        report_json_path=args.report_json,
        records_jsonl_path=args.records_jsonl,
        fact_candidates_jsonl_path=args.fact_candidates_jsonl,
        max_docs_per_request=args.max_docs_per_request,
        min_alignment_score=args.min_alignment_score,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "blind_spot_alignment_audit_ok "
        f"status={payload['status']} "
        f"requests={summary['alignment_request_count']} "
        f"candidates={summary['fact_candidate_count']} "
        f"top_gap={_first_key(summary['gap_reason_counts'])}"
    )


if __name__ == "__main__":
    main()
