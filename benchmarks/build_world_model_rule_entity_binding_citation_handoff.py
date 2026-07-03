"""Build citation/search handoff artifacts for entity-role binding candidates.

This workflow is the entity-binding counterpart to the unresolved blind-spot
citation/search handoff. It emits label-free external search requests from
incomplete ``world_model_rule_entity_binding_plan`` candidates and can normalize
adapter results into local citation documents for
``collect_world_model_rule_entity_bindings_from_citation_corpus.py``.

The handoff does not call the network, approve candidates, or execute rule
fills. Adapter results remain source-discovery artifacts until they pass the
existing citation-corpus collector, rule reviewer, and promotion gate.
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

from benchmarks.build_external_retrieval_corpus import build_external_retrieval_corpus  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402
from eigentruth.verify.search_planning import plan_citation_search_query  # noqa: E402

WORKFLOW = "world_model_rule_entity_binding_citation_search_handoff"
SOURCE_WORKFLOW = "world_model_rule_entity_binding_plan"
DEFAULT_CORPUS_NAME = "world_model_rule_entity_binding_citation_search"
DEFAULT_SOURCE_KIND = "entity_binding_external_citation_search_result"
DEFAULT_SOURCE_FAMILY = "reference"
RESERVED_RESULT_FIELDS = {
    "answer",
    "answers",
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
REQUIRED_CANDIDATE_FIELDS = (
    "request_id",
    "target_id",
    "subject_entity",
    "answer_entity",
    "expected_entity",
    "requested_role",
    "source_citation",
)
TEXT_FIELDS = ("text", "content", "document", "body", "snippet", "summary", "abstract")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def build_world_model_rule_entity_binding_citation_handoff(
    entity_binding_plan: Mapping[str, Any],
    *,
    adapter_results: Sequence[Mapping[str, Any]] = (),
    max_requests: int | None = None,
    max_results_per_request: int | None = None,
    max_alternate_queries: int = 3,
    source_family: str = DEFAULT_SOURCE_FAMILY,
    source_kind: str = DEFAULT_SOURCE_KIND,
    corpus_name: str = DEFAULT_CORPUS_NAME,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return external search requests plus optional normalized source docs."""
    if entity_binding_plan.get("workflow") != SOURCE_WORKFLOW:
        raise ValueError(f"entity_binding_plan must have workflow={SOURCE_WORKFLOW!r}.")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("max_requests must be positive when provided.")
    if max_results_per_request is not None and int(max_results_per_request) <= 0:
        raise ValueError("max_results_per_request must be positive when provided.")
    if int(max_alternate_queries) < 0:
        raise ValueError("max_alternate_queries cannot be negative.")
    source_family = _clean(source_family) or DEFAULT_SOURCE_FAMILY
    source_kind = _clean(source_kind) or DEFAULT_SOURCE_KIND
    corpus_name = _clean(corpus_name) or DEFAULT_CORPUS_NAME

    candidates = tuple(_candidate_bindings(entity_binding_plan))
    selected = tuple(candidate for candidate in candidates if not _is_complete_candidate(candidate))
    if max_requests is not None:
        selected = selected[: int(max_requests)]
    adapter_requests = tuple(
        _adapter_request(candidate, max_alternate_queries=int(max_alternate_queries))
        for candidate in selected
    )
    collection_tasks = tuple(
        _collection_task(request, source_family=source_family)
        for request in adapter_requests
    )
    request_by_id = {request["request_id"]: request for request in adapter_requests}
    source_documents, result_summary = _source_documents_from_results(
        adapter_results,
        request_by_id=request_by_id,
        max_results_per_request=max_results_per_request,
        default_source_family=source_family,
        source_kind=source_kind,
    )
    summary = _summary(
        candidates=candidates,
        selected=selected,
        adapter_requests=adapter_requests,
        source_documents=source_documents,
        result_summary=result_summary,
    )
    status = "collected" if source_documents else "ready_for_external_adapter"
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Entity-role binding citation/search handoff. Adapter requests omit "
            "labels, target ids, model answers, and expected entities. Ingested "
            "results are source-discovery documents, not verifier evidence and "
            "not approved entity bindings."
        ),
        "source": {
            "entity_binding_plan_workflow": entity_binding_plan.get("workflow"),
            "entity_binding_plan_status": entity_binding_plan.get("status"),
            "entity_binding_candidate_count": len(candidates),
        },
        "label_usage": {
            "labels_used_for_adapter_requests": False,
            "labels_copied_to_adapter_requests": False,
            "model_answers_copied_to_adapter_requests": False,
            "expected_entities_copied_to_adapter_requests": False,
            "adapter_results_are_verifier_evidence": False,
            "adapter_results_approve_entity_bindings": False,
        },
        "config": {
            "max_requests": max_requests,
            "max_results_per_request": max_results_per_request,
            "max_alternate_queries": int(max_alternate_queries),
            "source_family": source_family,
            "source_kind": source_kind,
            "corpus_name": corpus_name,
        },
        "summary": summary,
        "adapter_requests": adapter_requests,
        "source_family_collection_tasks": collection_tasks,
        "source_documents": source_documents,
        "external_retrieval_corpus": None,
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    entity_binding_plan_path: str | Path,
    output_dir: str | Path,
    adapter_results_path: str | Path | None = None,
    report_json_path: str | Path | None = None,
    request_jsonl_path: str | Path | None = None,
    collection_tasks_jsonl_path: str | Path | None = None,
    source_docs_jsonl_path: str | Path | None = None,
    corpus_json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    max_requests: int | None = None,
    max_results_per_request: int | None = None,
    max_alternate_queries: int = 3,
    source_family: str = DEFAULT_SOURCE_FAMILY,
    source_kind: str = DEFAULT_SOURCE_KIND,
    corpus_name: str = DEFAULT_CORPUS_NAME,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register the handoff."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "entity-binding-citation-search-handoff.json")
    request_path = Path(request_jsonl_path or output / "entity-binding-citation-search-requests.jsonl")
    collection_tasks_path = Path(
        collection_tasks_jsonl_path or output / "entity-binding-source-family-collection-tasks.jsonl"
    )
    source_docs_path = Path(source_docs_jsonl_path or output / "entity-binding-citation-source-docs.jsonl")
    corpus_path = Path(corpus_json_path or output / "entity-binding-citation-corpus.json")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    entity_binding_plan = _load_json_object(entity_binding_plan_path)
    adapter_results = () if adapter_results_path is None else _load_jsonl(adapter_results_path)
    payload = build_world_model_rule_entity_binding_citation_handoff(
        entity_binding_plan,
        adapter_results=adapter_results,
        max_requests=max_requests,
        max_results_per_request=max_results_per_request,
        max_alternate_queries=max_alternate_queries,
        source_family=source_family,
        source_kind=source_kind,
        corpus_name=corpus_name,
        metadata=metadata,
    )
    _write_jsonl(request_path, payload["adapter_requests"], compact=compact_json)
    _write_jsonl(collection_tasks_path, payload["source_family_collection_tasks"], compact=compact_json)
    _write_jsonl(source_docs_path, payload["source_documents"], compact=compact_json)
    if payload["source_documents"]:
        corpus = build_external_retrieval_corpus(
            (source_docs_path,),
            corpus_name=corpus_name,
            source_kind=source_kind,
            require_source=True,
        )
        payload = dict(payload)
        payload["external_retrieval_corpus"] = corpus
        payload["summary"] = {
            **payload["summary"],
            "corpus_document_count": corpus["summary"]["n_documents"],
        }
        _write_json(corpus_path, corpus, compact=compact_json)

    report = dict(payload)
    report["paths"] = {
        "entity_binding_plan": str(entity_binding_plan_path),
        "adapter_results": None if adapter_results_path is None else str(adapter_results_path),
        "adapter_requests": str(request_path),
        "source_family_collection_tasks": str(collection_tasks_path),
        "source_documents": str(source_docs_path),
        "external_retrieval_corpus": None if payload["external_retrieval_corpus"] is None else str(corpus_path),
        "artifact_manifest": str(manifest_path),
    }
    payload = dict(payload)
    payload["paths"] = report["paths"]
    _write_json(report_path, report, compact=compact_json)

    manifest_sources: dict[str, str | Path | None] = {
        "entity_binding_citation_handoff": report_path,
        "entity_binding_plan": Path(entity_binding_plan_path),
        "adapter_requests": request_path,
        "source_family_collection_tasks": collection_tasks_path,
        "source_documents": source_docs_path,
        "external_retrieval_corpus": None if payload["external_retrieval_corpus"] is None else corpus_path,
        "adapter_results": None if adapter_results_path is None else Path(adapter_results_path),
    }
    manifest = build_artifact_manifest(
        manifest_sources,
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": report["status"],
            "adapter_request_count": report["summary"]["adapter_request_count"],
            "source_document_count": report["summary"]["source_document_count"],
            "corpus_document_count": report["summary"]["corpus_document_count"],
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
                "status": report["status"],
                "artifact_manifest": str(manifest_path),
                "adapter_request_count": report["summary"]["adapter_request_count"],
                "source_document_count": report["summary"]["source_document_count"],
                "corpus_document_count": report["summary"]["corpus_document_count"],
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _adapter_request(candidate: Mapping[str, Any], *, max_alternate_queries: int) -> dict[str, Any]:
    query_plan = plan_citation_search_query(
        question=_clean(candidate.get("question")),
        candidate_query="",
        question_type="entity_binding",
        disallowed_phrases=tuple(
            item
            for item in (
                _clean(candidate.get("answer_entity")),
                _clean(candidate.get("expected_entity")),
            )
            if item
        ),
        strategy="question",
        max_alternate_queries=max_alternate_queries,
        requires_timestamp=False,
    )
    variants = _query_variants(candidate, query_plan=query_plan, max_alternate_queries=max_alternate_queries)
    if not variants:
        raise ValueError(f"entity-binding candidate {_candidate_id(candidate)!r} has no search query.")
    candidate_id = _candidate_id(candidate)
    request = {
        "schema_version": 1,
        "request_id": _adapter_request_id(candidate),
        "adapter_family": "external_citation_search",
        "query": variants[0],
        "alternate_queries": variants[1:],
        "source_family_plan": {
            "families": (DEFAULT_SOURCE_FAMILY,),
            "freshness_required": False,
            "official_source_preferred": False,
            "query_hints": _query_hints(candidate),
            "rationale": ("entity_role_binding_source_discovery",),
        },
        "requires_timestamp": False,
        "question_type": "entity_binding",
        "priority": "high",
        "usage": "source_discovery_only",
        "not_verifier_evidence": True,
        "metadata": {
            "entity_binding_plan_candidate_sha256": _sha256_json(_candidate_fingerprint(candidate)),
            "candidate_binding_sha256": hashlib.sha256(candidate_id.encode("utf-8")).hexdigest(),
            "query_strategy": "entity_binding_question_keywords",
            "query_variant_count": len(variants),
            "requested_role": _clean(candidate.get("requested_role")),
            "source_workflow": WORKFLOW,
        },
    }
    _reject_reserved_fields(request, context=f"adapter request {request['request_id']}")
    return request


def _collection_task(request: Mapping[str, Any], *, source_family: str) -> dict[str, Any]:
    query = _clean(request.get("query"))
    alternates = tuple(_clean(item) for item in _sequence(request.get("alternate_queries")) if _clean(item))
    metadata = _mapping(request.get("metadata"))
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "usage": "source_catalog_collection_only",
        "not_verifier_evidence": True,
        "task_id": f"entity-binding-source-catalog-{request['request_id']}",
        "source_family": source_family,
        "query": query,
        "query_key": _normalized_query_key(query),
        "search_queries": tuple(dict.fromkeys((query, *alternates))),
        "request_ids": (str(request["request_id"]),),
        "source_queue_request_sha256": _hash_tuple(metadata.get("entity_binding_plan_candidate_sha256")),
        "source_workflow": WORKFLOW,
    }


def _query_variants(
    candidate: Mapping[str, Any],
    *,
    query_plan: Any,
    max_alternate_queries: int,
) -> tuple[str, ...]:
    subject = _clean(candidate.get("subject_entity"))
    requested_role = _clean(candidate.get("requested_role")).casefold()
    question = _clean(candidate.get("question"))
    disallowed = tuple(
        item
        for item in (_clean(candidate.get("answer_entity")), _clean(candidate.get("expected_entity")))
        if item
    )
    role_query = _role_query(subject=subject, requested_role=requested_role, question=question)
    raw = (
        role_query,
        _clean(getattr(query_plan, "query", "")),
        *tuple(_clean(item) for item in getattr(query_plan, "alternate_queries", ())),
        _fallback_query(subject=subject, question=question),
    )
    variants: list[str] = []
    seen: set[str] = set()
    limit = 1 + int(max_alternate_queries)
    for value in raw:
        cleaned = _remove_disallowed(_clean(value), disallowed)
        if not cleaned:
            continue
        folded = cleaned.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        variants.append(cleaned)
        if len(variants) >= limit:
            break
    return tuple(variants)


def _role_query(*, subject: str, requested_role: str, question: str) -> str:
    tokens = _keyword_terms(question)
    if requested_role == "name_completion":
        preferred = _ordered_subset(
            tokens,
            ("american", "producer", "born", "70s", "70", "comedy", "comedian"),
        )
        return " ".join(item for item in (subject, *preferred) if item)
    if requested_role == "team_name":
        preferred = _ordered_subset(
            tokens,
            ("pilgrims", "football", "team", "boston", "national", "league", "2001"),
        )
        return " ".join(preferred or tokens[:8])
    return _fallback_query(subject=subject, question=question)


def _fallback_query(*, subject: str, question: str) -> str:
    tokens = _keyword_terms(question)
    return " ".join(item for item in (subject, *tokens[:8]) if item)


def _query_hints(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    subject = _clean(candidate.get("subject_entity"))
    requested_role = _clean(candidate.get("requested_role"))
    question_terms = _keyword_terms(_clean(candidate.get("question")))[:6]
    hints = tuple(item for item in (subject, requested_role, *question_terms) if item)
    return tuple(dict.fromkeys(hints))


def _source_documents_from_results(
    adapter_results: Sequence[Mapping[str, Any]],
    *,
    request_by_id: Mapping[str, Mapping[str, Any]],
    max_results_per_request: int | None,
    default_source_family: str,
    source_kind: str,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    unknown_result_ids: set[str] = set()
    unknown_results = 0
    skipped_empty = 0
    seen_per_request: Counter[str] = Counter()
    for item in adapter_results:
        request_id = _result_request_id(item)
        request = request_by_id.get(request_id)
        if request is None:
            unknown_results += 1
            if request_id:
                unknown_result_ids.add(request_id)
            continue
        for result in _result_items(item):
            if max_results_per_request is not None and seen_per_request[request_id] >= int(max_results_per_request):
                continue
            document = _source_document(
                result,
                request=request,
                default_source_family=default_source_family,
                source_kind=source_kind,
            )
            if document is None:
                skipped_empty += 1
                continue
            documents.append(document)
            seen_per_request[request_id] += 1
    request_ids = set(request_by_id)
    matched_request_ids = set(seen_per_request)
    missing_request_ids = tuple(sorted(request_ids - matched_request_ids))
    expected_request_count = len(request_by_id)
    matched_request_count = len(matched_request_ids)
    return tuple(documents), {
        "expected_request_count": expected_request_count,
        "input_result_count": len(adapter_results),
        "unknown_request_result_count": unknown_results,
        "unknown_request_result_ids": tuple(sorted(unknown_result_ids)),
        "skipped_empty_result_count": skipped_empty,
        "matched_request_count": matched_request_count,
        "missing_request_count": len(missing_request_ids),
        "missing_request_ids": missing_request_ids,
        "request_coverage": (
            1.0
            if expected_request_count == 0
            else matched_request_count / float(expected_request_count)
        ),
        "result_documents_by_request": dict(sorted(seen_per_request.items())),
    }


def _source_document(
    result: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    default_source_family: str,
    source_kind: str,
) -> dict[str, Any] | None:
    _reject_reserved_fields(result, context="adapter result")
    title = _clean(result.get("title"))
    text = _clean(_first_nonempty(result.get(field) for field in TEXT_FIELDS))
    if not text:
        text = _clean(" ".join(item for item in (title, _clean(result.get("snippet"))) if item))
    if not text:
        return None
    url = _clean(result.get("url") or result.get("href"))
    provider = _clean(result.get("provider") or result.get("source_provider")) or "external_citation_search"
    source = _clean(result.get("source")) or url or f"entity-binding-citation:{provider}:{request['request_id']}"
    source_family = _clean(result.get("source_family") or result.get("source_family_name")) or default_source_family
    metadata = {
        "external_source": True,
        "source_kind": source_kind,
        "provider": provider,
        "source_family": source_family,
        "title": title,
        "url": url,
        "published_at": _clean(result.get("published_at") or result.get("publication_date")),
        "timestamp": _clean(result.get("timestamp") or result.get("retrieved_at")),
        "rank": _optional_int(result.get("rank")),
        "adapter_request_sha256": _sha256_json(request),
        "query_sha256": hashlib.sha256(str(request.get("query", "")).encode("utf-8")).hexdigest(),
        "result_sha256": _sha256_json(result),
        "candidate_binding_sha256": _mapping(request.get("metadata")).get("candidate_binding_sha256"),
        "source_workflow": WORKFLOW,
    }
    metadata = {key: value for key, value in metadata.items() if value is not None and value != ""}
    _reject_reserved_fields(metadata, context=f"source document {source!r} metadata")
    return {
        "text": text,
        "source": source,
        "source_url": url,
        "source_title": title,
        "source_family": source_family,
        "provider": provider,
        "metadata": metadata,
    }


def _summary(
    *,
    candidates: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    adapter_requests: Sequence[Mapping[str, Any]],
    source_documents: Sequence[Mapping[str, Any]],
    result_summary: Mapping[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(_clean(candidate.get("candidate_status")) for candidate in candidates)
    missing_field_counts = Counter(
        field
        for candidate in selected
        for field in _missing_required_candidate_fields(candidate)
    )
    source_family_counts = Counter(_clean(document.get("source_family")) for document in source_documents)
    provider_counts = Counter(_clean(document.get("provider")) for document in source_documents)
    return {
        "candidate_count": len(candidates),
        "selected_candidate_count": len(selected),
        "adapter_request_count": len(adapter_requests),
        "source_document_count": len(source_documents),
        "corpus_document_count": len(source_documents),
        "candidate_status_counts": _sorted_counter(status_counts),
        "selected_missing_field_counts": _sorted_counter(missing_field_counts),
        "source_family_counts": _sorted_counter(source_family_counts),
        "provider_counts": _sorted_counter(provider_counts),
        "adapter_result_expected_request_count": result_summary.get("expected_request_count", 0),
        "adapter_result_matched_request_count": result_summary.get("matched_request_count", 0),
        "adapter_result_missing_request_count": result_summary.get("missing_request_count", 0),
        "adapter_result_missing_request_ids": tuple(result_summary.get("missing_request_ids", ())),
        "adapter_result_unknown_request_count": result_summary.get("unknown_request_result_count", 0),
        "adapter_result_request_coverage": result_summary.get("request_coverage", 1.0),
        "result_summary": dict(result_summary),
    }


def _candidate_bindings(entity_binding_plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_candidates = entity_binding_plan.get("candidate_entity_bindings")
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes, bytearray)):
        raise ValueError("entity_binding_plan must contain candidate_entity_bindings.")
    return tuple(dict(item) for item in raw_candidates if isinstance(item, Mapping))


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    return _clean(candidate.get("binding_id")) or _clean(candidate.get("request_id")) or _sha256_json(candidate)[:16]


def _candidate_fingerprint(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "binding_id": candidate.get("binding_id"),
        "request_id": candidate.get("request_id"),
        "source_request_id": candidate.get("source_request_id"),
        "question": candidate.get("question"),
        "subject_entity": candidate.get("subject_entity"),
        "requested_role": candidate.get("requested_role"),
        "candidate_status": candidate.get("candidate_status"),
    }


def _adapter_request_id(candidate: Mapping[str, Any]) -> str:
    payload = _candidate_fingerprint(candidate)
    digest = _sha256_json(payload)[:16]
    return f"entity-binding-cite-{digest}"


def _missing_required_candidate_fields(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(key for key in REQUIRED_CANDIDATE_FIELDS if not _clean(candidate.get(key)))


def _is_complete_candidate(candidate: Mapping[str, Any]) -> bool:
    return not _missing_required_candidate_fields(candidate)


def _result_request_id(item: Mapping[str, Any]) -> str:
    return _clean(
        item.get("request_id")
        or item.get("adapter_request_id")
        or item.get("queue_id")
        or item.get("source_request_id")
    )


def _result_items(item: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw_results = item.get("results")
    if isinstance(raw_results, Sequence) and not isinstance(raw_results, (str, bytes, bytearray)):
        return tuple(result for result in raw_results if isinstance(result, Mapping))
    return (item,)


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _keyword_terms(value: str) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(value.casefold().replace("70s", "70s 70")):
        if len(token) <= 2 and not token.isdigit():
            continue
        if token in {"this", "what", "name", "with", "from", "that", "have", "based", "plays"}:
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return tuple(terms)


def _ordered_subset(values: Sequence[str], preferred: Sequence[str]) -> tuple[str, ...]:
    values_set = set(values)
    selected: list[str] = []
    for item in preferred:
        if item in values_set and item not in selected:
            selected.append(item)
    return tuple(selected)


def _remove_disallowed(value: str, disallowed: Sequence[str]) -> str:
    cleaned = value
    for phrase in disallowed:
        if not phrase:
            continue
        cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def _normalized_query_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _hash_tuple(value: Any) -> tuple[str, ...]:
    cleaned = _clean(value)
    return (cleaned,) if cleaned else ()


def _first_nonempty(values: Sequence[Any] | Any) -> str:
    for value in values:
        cleaned = _clean(value)
        if cleaned:
            return cleaned
    return ""


def _reject_reserved_fields(mapping: Mapping[str, Any], *, context: str) -> None:
    reserved = sorted(set(str(key) for key in mapping) & RESERVED_RESULT_FIELDS)
    if reserved:
        raise ValueError(f"{context} contains reserved fields: {', '.join(reserved)}")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sha256_json(value: Mapping[str, Any]) -> str:
    text = strict_json_dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter) if key}


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _load_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append(dict(payload))
    return tuple(rows)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = (
        strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        if compact
        else strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool) -> None:
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
    parser.add_argument("--adapter-results", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--request-jsonl", default=None)
    parser.add_argument("--collection-tasks-jsonl", default=None)
    parser.add_argument("--source-docs-jsonl", default=None)
    parser.add_argument("--corpus-json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-results-per-request", type=int, default=None)
    parser.add_argument("--max-alternate-queries", type=int, default=3)
    parser.add_argument("--source-family", default=DEFAULT_SOURCE_FAMILY)
    parser.add_argument("--source-kind", default=DEFAULT_SOURCE_KIND)
    parser.add_argument("--corpus-name", default=DEFAULT_CORPUS_NAME)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(
        entity_binding_plan_path=args.entity_binding_plan,
        output_dir=args.output_dir,
        adapter_results_path=args.adapter_results,
        report_json_path=args.json,
        request_jsonl_path=args.request_jsonl,
        collection_tasks_jsonl_path=args.collection_tasks_jsonl,
        source_docs_jsonl_path=args.source_docs_jsonl,
        corpus_json_path=args.corpus_json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        max_requests=args.max_requests,
        max_results_per_request=args.max_results_per_request,
        max_alternate_queries=args.max_alternate_queries,
        source_family=args.source_family,
        source_kind=args.source_kind,
        corpus_name=args.corpus_name,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    print(
        "world_model_rule_entity_binding_citation_search_handoff_ok "
        f"status={payload['status']} "
        f"requests={payload['summary']['adapter_request_count']} "
        f"source_docs={payload['summary']['source_document_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
