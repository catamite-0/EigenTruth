"""Prepare and ingest citation/search adapter work for unresolved blind spots.

The unresolved evidence queue is product-internal: it carries record ids,
model answers, and mapping diagnostics. This workflow creates the narrower
external-adapter boundary. It emits label-free citation/search requests that
can be handed to a local search tool, and can optionally ingest that tool's
JSONL results into an external retrieval corpus.

No network calls are made here. External adapters should materialize their
results locally, then rerun this workflow with ``--adapter-results`` so the
returned documents are normalized and provenance-gated before verifier use.
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

from benchmarks.build_external_retrieval_corpus import build_external_retrieval_corpus  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402
from eigentruth.verify.search_planning import (  # noqa: E402
    QUERY_PLAN_STRATEGIES,
    CitationSearchQueryPlan,
    plan_citation_search_query,
)

CITATION_REQUEST_TYPE = "external_citation"
WORKFLOW = "citation_search_adapter_handoff"
DEFAULT_CORPUS_NAME = "unresolved_blind_spot_citation_search"
DEFAULT_SOURCE_KIND = "external_citation_search_result"
QUERY_MODES = QUERY_PLAN_STRATEGIES
DEFAULT_MAX_ALTERNATE_QUERIES = 3
RESERVED_EXTERNAL_FIELDS = {
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


def build_citation_search_adapter_handoff(
    queue_report: Mapping[str, Any],
    *,
    adapter_results: Sequence[Mapping[str, Any]] = (),
    batch_ids: Sequence[str] = (),
    query_mode: str = "question",
    max_requests: int | None = None,
    max_results_per_request: int | None = None,
    max_alternate_queries: int = DEFAULT_MAX_ALTERNATE_QUERIES,
    corpus_name: str = DEFAULT_CORPUS_NAME,
    source_kind: str = DEFAULT_SOURCE_KIND,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return adapter requests, optional source docs, and optional corpus."""
    _validate_queue(queue_report)
    if query_mode not in QUERY_MODES:
        raise ValueError(f"query_mode must be one of: {', '.join(QUERY_MODES)}.")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("max_requests must be positive when provided.")
    if max_results_per_request is not None and int(max_results_per_request) <= 0:
        raise ValueError("max_results_per_request must be positive when provided.")
    if int(max_alternate_queries) < 0:
        raise ValueError("max_alternate_queries cannot be negative.")
    selected_batch_ids = _batch_id_tuple(batch_ids)
    selected_request_ids, selected_batches = _selected_batch_request_ids(queue_report, selected_batch_ids)
    selected_request_id_set = set(selected_request_ids)
    source_requests = tuple(
        request
        for request in _mapping_sequence(queue_report.get("adapter_requests", ()))
        if request.get("request_type") == CITATION_REQUEST_TYPE
    )
    if selected_batch_ids:
        source_requests = tuple(
            request
            for request in source_requests
            if _queue_request_identifier(request) in selected_request_id_set
        )
    if max_requests is not None:
        source_requests = source_requests[: int(max_requests)]
    adapter_requests = tuple(
        _adapter_request(
            request,
            query_mode=query_mode,
            max_alternate_queries=int(max_alternate_queries),
        )
        for request in source_requests
    )
    request_by_id = {str(request["request_id"]): request for request in adapter_requests}
    source_documents, result_summary = _source_documents_from_results(
        adapter_results,
        request_by_id=request_by_id,
        max_results_per_request=max_results_per_request,
        source_kind=source_kind,
    )
    summary = _summary(
        queue_report=queue_report,
        adapter_requests=adapter_requests,
        source_documents=source_documents,
        result_summary=result_summary,
        selected_batches=selected_batches,
    )
    status = "collected" if source_documents else "ready_for_external_adapter"
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Citation/search adapter handoff for unresolved blind spots. "
            "Adapter requests omit labels, record ids, target ids, and model "
            "answers. Ingested results are external-candidate documents, not "
            "verifier evidence until provenance-audited."
        ),
        "source": {
            "queue_workflow": queue_report.get("workflow"),
            "queue_status": queue_report.get("status"),
            "queue_target_count": _nested_int(queue_report, "summary", "target_count"),
            "queue_adapter_request_count": _nested_int(queue_report, "summary", "adapter_request_count"),
            "queue_batch_count": _nested_int(queue_report, "summary", "batch_count"),
        },
        "label_usage": {
            "labels_used_for_adapter_requests": False,
            "labels_copied_to_adapter_requests": False,
            "model_answers_copied_to_adapter_requests": False,
            "adapter_results_are_verifier_evidence": False,
        },
        "config": {
            "batch_ids": selected_batch_ids,
            "query_mode": query_mode,
            "max_requests": max_requests,
            "max_results_per_request": max_results_per_request,
            "max_alternate_queries": int(max_alternate_queries),
            "corpus_name": corpus_name,
            "source_kind": source_kind,
        },
        "summary": summary,
        "adapter_requests": adapter_requests,
        "source_documents": source_documents,
        "external_retrieval_corpus": None,
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    queue_report_path: str | Path,
    output_dir: str | Path,
    adapter_results_path: str | Path | None = None,
    report_json_path: str | Path | None = None,
    request_jsonl_path: str | Path | None = None,
    source_jsonl_path: str | Path | None = None,
    corpus_json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    batch_ids: Sequence[str] = (),
    query_mode: str = "question",
    max_requests: int | None = None,
    max_results_per_request: int | None = None,
    max_alternate_queries: int = DEFAULT_MAX_ALTERNATE_QUERIES,
    corpus_name: str = DEFAULT_CORPUS_NAME,
    source_kind: str = DEFAULT_SOURCE_KIND,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register the handoff."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    report_path = Path(report_json_path) if report_json_path is not None else output / "citation-search-handoff.json"
    request_path = (
        Path(request_jsonl_path)
        if request_jsonl_path is not None
        else output / "citation-search-adapter-requests.jsonl"
    )
    source_path = (
        Path(source_jsonl_path)
        if source_jsonl_path is not None
        else output / "citation-search-source-docs.jsonl"
    )
    corpus_path = Path(corpus_json_path) if corpus_json_path is not None else output / "citation-search-corpus.json"
    manifest_path = (
        Path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else output / "artifact-manifest.json"
    )
    queue_report = _load_json_object(queue_report_path)
    adapter_results = () if adapter_results_path is None else _load_jsonl(adapter_results_path)
    payload = build_citation_search_adapter_handoff(
        queue_report,
        adapter_results=adapter_results,
        batch_ids=batch_ids,
        query_mode=query_mode,
        max_requests=max_requests,
        max_results_per_request=max_results_per_request,
        max_alternate_queries=max_alternate_queries,
        corpus_name=corpus_name,
        source_kind=source_kind,
        metadata=metadata,
    )
    _write_jsonl(request_path, payload["adapter_requests"])
    _write_jsonl(source_path, payload["source_documents"])
    if payload["source_documents"]:
        corpus = build_external_retrieval_corpus(
            (source_path,),
            corpus_name=corpus_name,
            source_kind=source_kind,
            require_source=True,
        )
        payload["external_retrieval_corpus"] = corpus
        payload["summary"] = {
            **payload["summary"],
            "corpus_document_count": corpus["summary"]["n_documents"],
        }

    report = dict(payload)
    report["paths"] = {
        "unresolved_queue": str(queue_report_path),
        "adapter_requests": str(request_path),
        "adapter_results": None if adapter_results_path is None else str(adapter_results_path),
        "source_documents": str(source_path),
        "external_retrieval_corpus": None if payload["external_retrieval_corpus"] is None else str(corpus_path),
        "artifact_manifest": str(manifest_path),
    }
    payload = dict(payload)
    payload["paths"] = report["paths"]

    _write_json(report_path, report, compact=compact_json)
    if payload["external_retrieval_corpus"] is not None:
        _write_json(corpus_path, payload["external_retrieval_corpus"], compact=compact_json)

    artifacts: dict[str, str | Path | None] = {
        "citation_search_handoff": report_path,
        "citation_search_adapter_requests": request_path,
        "citation_search_source_documents": source_path,
        "citation_search_corpus": None if payload["external_retrieval_corpus"] is None else corpus_path,
        "unresolved_blind_spot_evidence_queue": Path(queue_report_path),
        "adapter_results": None if adapter_results_path is None else Path(adapter_results_path),
    }
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "workflow": report["workflow"],
            "status": report["status"],
            "adapter_request_count": report["summary"]["adapter_request_count"],
            "selected_batch_count": report["summary"]["selected_batch_count"],
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
                "workflow": report["workflow"],
                "status": report["status"],
                "adapter_request_count": report["summary"]["adapter_request_count"],
                "selected_batch_count": report["summary"]["selected_batch_count"],
                "source_document_count": report["summary"]["source_document_count"],
                "corpus_document_count": report["summary"]["corpus_document_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _adapter_request(
    request: Mapping[str, Any],
    *,
    query_mode: str,
    max_alternate_queries: int = DEFAULT_MAX_ALTERNATE_QUERIES,
) -> dict[str, Any]:
    query_plan = _query_plan_for_request(
        request,
        query_mode=query_mode,
        max_alternate_queries=max_alternate_queries,
    )
    if not query_plan.query:
        request_id = request.get("queue_id") or request.get("source_request_id")
        raise ValueError(f"citation request {request_id} has no query.")
    request_id = _adapter_request_id(request)
    source_family_plan = query_plan.source_family_plan
    source_family_payload = None if source_family_plan is None else source_family_plan.to_dict()
    return {
        "schema_version": 1,
        "request_id": request_id,
        "adapter_family": "external_citation_search",
        "query": query_plan.query,
        "alternate_queries": tuple(query_plan.alternate_queries),
        "source_family_plan": source_family_payload,
        "requires_timestamp": bool(request.get("requires_timestamp")),
        "question_type": str(request.get("question_type", "")),
        "priority": str(request.get("priority", "")),
        "usage": "source_discovery_only",
        "not_verifier_evidence": True,
        "metadata": {
            "source_queue_request_sha256": _sha256_json(_minimal_request_fingerprint(request)),
            "query_mode": query_mode,
            "query_strategy": query_plan.strategy,
            "query_variant_count": len(query_plan.variants),
            "entity_candidates": tuple(query_plan.entity_candidates),
            "keyword_terms": tuple(query_plan.keyword_terms),
            "removed_disallowed_phrase_count": len(query_plan.removed_phrase_hashes),
            "preferred_source_families": (
                ()
                if source_family_plan is None
                else tuple(source_family_plan.families)
            ),
            "freshness_required": (
                False
                if source_family_plan is None
                else source_family_plan.freshness_required
            ),
            "official_source_preferred": (
                False
                if source_family_plan is None
                else source_family_plan.official_source_preferred
            ),
            "queue_workflow": "unresolved_blind_spot_evidence_queue",
        },
    }


def _query_for_request(request: Mapping[str, Any], *, query_mode: str) -> str:
    return _query_plan_for_request(request, query_mode=query_mode).query


def _query_plan_for_request(
    request: Mapping[str, Any],
    *,
    query_mode: str,
    max_alternate_queries: int = DEFAULT_MAX_ALTERNATE_QUERIES,
) -> CitationSearchQueryPlan:
    return plan_citation_search_query(
        question=str(request.get("question", "")),
        candidate_query=str(request.get("query", "")),
        question_type=str(request.get("question_type", "")),
        disallowed_phrases=_disallowed_query_phrases(request),
        strategy=query_mode,
        max_alternate_queries=max_alternate_queries,
        requires_timestamp=bool(request.get("requires_timestamp")),
    )


def _disallowed_query_phrases(request: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(value)
        for value in (request.get("model_answer"), request.get("answer"))
        if value is not None and str(value).strip()
    )


def _adapter_request_id(request: Mapping[str, Any]) -> str:
    raw = str(request.get("queue_id") or request.get("source_request_id") or request.get("request_id") or "")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16] if raw else _sha256_json(request)[:16]
    return f"cite-search-{digest}"


def _batch_id_tuple(batch_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_string_sequence(batch_ids)))


def _selected_batch_request_ids(
    queue_report: Mapping[str, Any],
    batch_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    selected_batch_ids = _batch_id_tuple(batch_ids)
    if not selected_batch_ids:
        return (), ()
    batches_by_id = {
        str(batch.get("batch_id")): batch
        for batch in _mapping_sequence(queue_report.get("execution_batches", ()))
        if str(batch.get("batch_id", "")).strip()
    }
    missing = tuple(batch_id for batch_id in selected_batch_ids if batch_id not in batches_by_id)
    if missing:
        raise ValueError(f"unknown execution batch ids: {', '.join(missing)}")
    selected_batches = tuple(batches_by_id[batch_id] for batch_id in selected_batch_ids)
    request_ids: list[str] = []
    for batch in selected_batches:
        request_ids.extend(_string_sequence(batch.get("source_request_ids", ())))
    return tuple(dict.fromkeys(request_ids)), selected_batches


def _queue_request_identifier(request: Mapping[str, Any]) -> str:
    return str(request.get("source_request_id") or request.get("request_id") or request.get("queue_id") or "")


def _source_documents_from_results(
    adapter_results: Sequence[Mapping[str, Any]],
    *,
    request_by_id: Mapping[str, Mapping[str, Any]],
    max_results_per_request: int | None,
    source_kind: str,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    unknown_results = 0
    skipped_empty = 0
    seen_per_request: Counter[str] = Counter()
    for item in adapter_results:
        request_id = _result_request_id(item)
        if request_id not in request_by_id:
            unknown_results += 1
            continue
        results = _result_items(item)
        for result in results:
            if max_results_per_request is not None and seen_per_request[request_id] >= int(max_results_per_request):
                continue
            document = _source_document(result, request=request_by_id[request_id], source_kind=source_kind)
            if document is None:
                skipped_empty += 1
                continue
            documents.append(document)
            seen_per_request[request_id] += 1
    return tuple(documents), {
        "input_result_count": len(adapter_results),
        "unknown_request_result_count": unknown_results,
        "skipped_empty_result_count": skipped_empty,
        "matched_request_count": len(seen_per_request),
        "result_documents_by_request": dict(sorted(seen_per_request.items())),
    }


def _result_request_id(item: Mapping[str, Any]) -> str:
    return str(
        item.get("request_id")
        or item.get("adapter_request_id")
        or item.get("queue_id")
        or item.get("source_request_id")
        or ""
    )


def _result_items(item: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw_results = item.get("results")
    if isinstance(raw_results, Sequence) and not isinstance(raw_results, (str, bytes, bytearray)):
        return tuple(result for result in raw_results if isinstance(result, Mapping))
    return (item,)


def _source_document(
    result: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    source_kind: str,
) -> dict[str, Any] | None:
    title = _clean(result.get("title"))
    snippet = _clean(result.get("snippet") or result.get("summary") or result.get("abstract"))
    content = _clean(result.get("text") or result.get("content") or result.get("document") or result.get("body"))
    text = content or _join_nonempty((title, snippet))
    if not text:
        return None
    url = _clean(result.get("url") or result.get("href"))
    provider = _clean(result.get("provider") or result.get("source_provider")) or "external_citation_search"
    source = _clean(result.get("source")) or url or f"citation-search:{provider}:{request['request_id']}"
    metadata = {
        "external_source": True,
        "source_kind": source_kind,
        "provider": provider,
        "source_family": _clean(result.get("source_family") or result.get("source_family_name")),
        "source_family_confidence": _optional_float(result.get("source_family_confidence")),
        "title": title,
        "url": url,
        "published_at": _clean(result.get("published_at") or result.get("publication_date")),
        "timestamp": _clean(result.get("timestamp") or result.get("retrieved_at")),
        "rank": _optional_int(result.get("rank")),
        "adapter_request_sha256": _sha256_json(request),
        "query_sha256": hashlib.sha256(str(request.get("query", "")).encode("utf-8")).hexdigest(),
        "result_sha256": _sha256_json(result),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None and value != ""}
    _reject_reserved_metadata(metadata, source=source)
    return {
        "text": text,
        "source": source,
        "metadata": metadata,
    }


def _summary(
    *,
    queue_report: Mapping[str, Any],
    adapter_requests: Sequence[Mapping[str, Any]],
    source_documents: Sequence[Mapping[str, Any]],
    result_summary: Mapping[str, Any],
    selected_batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    priority_counts = Counter(str(item.get("priority")) for item in adapter_requests)
    question_type_counts = Counter(str(item.get("question_type")) for item in adapter_requests)
    source_family_counts: Counter[str] = Counter()
    freshness_required_count = 0
    official_source_preferred_count = 0
    for item in adapter_requests:
        plan = _mapping(item.get("source_family_plan"))
        if plan.get("freshness_required"):
            freshness_required_count += 1
        if plan.get("official_source_preferred"):
            official_source_preferred_count += 1
        for family in _string_sequence(plan.get("families", ())):
            source_family_counts[family] += 1
    providers = Counter(str(_mapping(item.get("metadata")).get("provider")) for item in source_documents)
    selected_batch_ids = tuple(str(batch.get("batch_id")) for batch in selected_batches)
    selected_batch_request_count = sum(
        _optional_int(batch.get("request_count")) or len(_string_sequence(batch.get("source_request_ids", ())))
        for batch in selected_batches
    )
    return {
        "source_queue_target_count": _nested_int(queue_report, "summary", "target_count"),
        "source_queue_adapter_request_count": _nested_int(queue_report, "summary", "adapter_request_count"),
        "source_queue_batch_count": _nested_int(queue_report, "summary", "batch_count"),
        "selected_batch_count": len(selected_batches),
        "selected_batch_ids": selected_batch_ids,
        "selected_batch_source_request_count": selected_batch_request_count,
        "adapter_request_count": len(adapter_requests),
        "source_document_count": len(source_documents),
        "corpus_document_count": len(source_documents),
        "priority_counts": _sorted_counter(priority_counts),
        "question_type_counts": _sorted_counter(question_type_counts),
        "source_family_counts": _sorted_counter(source_family_counts),
        "freshness_required_count": freshness_required_count,
        "official_source_preferred_count": official_source_preferred_count,
        "source_provider_counts": _sorted_counter(providers),
        "result_summary": dict(result_summary),
    }


def _validate_queue(queue_report: Mapping[str, Any]) -> None:
    if queue_report.get("workflow") != "unresolved_blind_spot_evidence_queue":
        raise ValueError("queue report must be an unresolved_blind_spot_evidence_queue report.")


def _minimal_request_fingerprint(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "queue_id": request.get("queue_id"),
        "source_request_id": request.get("source_request_id"),
        "adapter_family": request.get("adapter_family"),
        "request_type": request.get("request_type"),
        "question": request.get("question"),
        "query": request.get("query"),
    }


def _reject_reserved_metadata(metadata: Mapping[str, Any], *, source: str) -> None:
    reserved = sorted(set(str(key) for key in metadata) & RESERVED_EXTERNAL_FIELDS)
    if reserved:
        raise ValueError(f"source document {source!r} contains reserved metadata keys: {', '.join(reserved)}")


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _load_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no} must contain a JSON object.")
            rows.append(dict(payload))
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
    output.write_text("".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return _optional_int(current)


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    return ()


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _join_nonempty(values: Sequence[str]) -> str:
    return " ".join(str(value).strip() for value in values if str(value).strip())


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(strict_json_dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(((key, value) for key, value in counter.items() if key), key=lambda item: (-item[1], item[0])))


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
    parser.add_argument("--queue", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-results", default=None)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--request-jsonl", default=None)
    parser.add_argument("--source-jsonl", default=None)
    parser.add_argument("--corpus-json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--batch-id",
        action="append",
        default=[],
        help="Execution batch id from the unresolved queue to hand off. May be repeated.",
    )
    parser.add_argument("--query-mode", choices=QUERY_MODES, default="question")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-results-per-request", type=int, default=None)
    parser.add_argument("--max-alternate-queries", type=int, default=DEFAULT_MAX_ALTERNATE_QUERIES)
    parser.add_argument("--corpus-name", default=DEFAULT_CORPUS_NAME)
    parser.add_argument("--source-kind", default=DEFAULT_SOURCE_KIND)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        queue_report_path=args.queue,
        output_dir=args.output_dir,
        adapter_results_path=args.adapter_results,
        report_json_path=args.report_json,
        request_jsonl_path=args.request_jsonl,
        source_jsonl_path=args.source_jsonl,
        corpus_json_path=args.corpus_json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        batch_ids=tuple(args.batch_id or ()),
        query_mode=args.query_mode,
        max_requests=args.max_requests,
        max_results_per_request=args.max_results_per_request,
        max_alternate_queries=args.max_alternate_queries,
        corpus_name=args.corpus_name,
        source_kind=args.source_kind,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "citation_search_adapter_handoff_ok "
        f"status={payload['status']} "
        f"requests={summary['adapter_request_count']} "
        f"source_docs={summary['source_document_count']} "
        f"corpus_docs={summary['corpus_document_count']}"
    )


if __name__ == "__main__":
    main()
