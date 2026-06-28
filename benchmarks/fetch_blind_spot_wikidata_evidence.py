"""Fetch Wikidata source documents for blind-spot evidence requests.

This workflow consumes ``build_blind_spot_evidence_collection_corpus.py`` output
and materializes CC0 Wikidata source documents for the queued entity/property
requests. It keeps score-row context out of source-document metadata: request
and target identifiers stay in the collection report, while the JSONL evidence
documents contain only external Wikidata provenance and a request fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

DEFAULT_API_ENDPOINT = "https://www.wikidata.org/w/api.php"
DEFAULT_USER_AGENT = (
    "EigenTruth/0.1 blind-spot-wikidata-evidence "
    "(https://github.com/catamitez0-maker/EigenTruth)"
)
WIKIDATA_LICENSE_URL = "https://www.wikidata.org/wiki/Wikidata:Licensing"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/"
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CANDIDATE_RANK = 0
WIKIDATA_ID_RE = re.compile(r"^[QP][1-9][0-9]*$")
RESERVED_SOURCE_METADATA_KEYS = {
    "claim_id",
    "collection_request_id",
    "collection_target_id",
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

SearchFn = Callable[[str, int], Sequence[Mapping[str, Any]]]
EntityLoader = Callable[[Sequence[str]], Mapping[str, Mapping[str, Any]]]


class WikidataAPIClient:
    """Tiny stdlib Wikidata API client with process-local caching."""

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_API_ENDPOINT,
        timeout: float = 30.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self.user_agent = user_agent
        self._search_cache: dict[tuple[str, int], tuple[dict[str, Any], ...]] = {}
        self._entity_cache: dict[str, dict[str, Any]] = {}

    def search_entities(self, query: str, limit: int) -> tuple[dict[str, Any], ...]:
        """Search Wikidata items for an English label query."""
        key = (query, int(limit))
        if key in self._search_cache:
            return self._search_cache[key]
        payload = self._request({
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "limit": str(int(limit)),
            "search": query,
            "type": "item",
            "uselang": "en",
        })
        results = tuple(
            dict(item)
            for item in payload.get("search", ())
            if isinstance(item, Mapping) and _is_wikidata_id(str(item.get("id", "")))
        )
        self._search_cache[key] = results
        return results

    def load_entities(self, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Load Wikidata entity payloads for ids, batching API calls."""
        normalized = tuple(dict.fromkeys(_normalize_wikidata_id(item) for item in ids if str(item).strip()))
        missing = [item for item in normalized if item not in self._entity_cache]
        for chunk in _chunks(missing, 50):
            payload = self._request({
                "action": "wbgetentities",
                "format": "json",
                "ids": "|".join(chunk),
                "languages": "en",
                "languagefallback": "1",
                "props": "labels|descriptions|claims",
            })
            entities = payload.get("entities", {})
            if not isinstance(entities, Mapping):
                continue
            for entity_id, entity in entities.items():
                if isinstance(entity, Mapping):
                    self._entity_cache[str(entity_id)] = dict(entity)
        return {
            item: self._entity_cache[item]
            for item in normalized
            if item in self._entity_cache and not self._entity_cache[item].get("missing")
        }

    def _request(self, params: Mapping[str, str]) -> dict[str, Any]:
        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(
            f"{self.endpoint}?{query}",
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Wikidata API returned a non-object JSON payload.")
        if payload.get("error"):
            raise ValueError(f"Wikidata API error: {payload['error']!r}")
        return dict(payload)


def build_wikidata_evidence_from_collection(
    collection: Mapping[str, Any],
    *,
    search_entities: SearchFn,
    load_entities: EntityLoader,
    fetched_at: str,
    endpoint: str = DEFAULT_API_ENDPOINT,
    search_limit: int = DEFAULT_SEARCH_LIMIT,
    candidate_rank: int = DEFAULT_CANDIDATE_RANK,
    max_requests: int | None = None,
    property_only: bool = False,
) -> dict[str, Any]:
    """Build report and source docs from a collection corpus using supplied lookups."""
    requests = _select_requests(collection, max_requests=max_requests, property_only=property_only)
    if int(search_limit) <= 0:
        raise ValueError("search_limit must be positive.")
    if int(candidate_rank) < 0:
        raise ValueError("candidate_rank must be non-negative.")

    resolution_by_entity = _resolve_entities(
        requests,
        search_entities=search_entities,
        search_limit=int(search_limit),
        candidate_rank=int(candidate_rank),
    )
    subject_ids = tuple(
        sorted({
            str(resolution["qid"])
            for resolution in resolution_by_entity.values()
            if resolution.get("qid")
        })
    )
    subject_entities = dict(load_entities(subject_ids)) if subject_ids else {}
    preliminary = _preliminary_request_documents(
        requests,
        resolution_by_entity=resolution_by_entity,
        subject_entities=subject_entities,
        fetched_at=fetched_at,
        endpoint=endpoint,
    )
    label_ids = sorted(preliminary["label_ids"])
    label_entities = dict(load_entities(label_ids)) if label_ids else {}
    labels = _label_lookup({**subject_entities, **label_entities})
    source_documents = _finalize_documents(preliminary["documents"], labels=labels)
    request_results = _finalize_request_results(preliminary["request_results"], labels=labels)
    summary = _summary(
        requests=requests,
        request_results=request_results,
        source_documents=source_documents,
        resolved_entities=resolution_by_entity,
    )
    return {
        "schema_version": 1,
        "workflow": "blind_spot_wikidata_evidence_fetch",
        "status": "collected" if source_documents else "no_documents",
        "source": {
            "collection_workflow": collection.get("workflow"),
            "collection_status": collection.get("status"),
            "collection_target_count": _nested_int(collection, "summary", "target_count"),
        },
        "config": {
            "endpoint": endpoint,
            "search_limit": int(search_limit),
            "candidate_rank": int(candidate_rank),
            "max_requests": max_requests,
            "property_only": bool(property_only),
            "fetched_at": fetched_at,
        },
        "label_usage": {
            "score_labels_used_for_fetch": False,
            "score_labels_copied_to_source_docs": False,
            "model_answers_copied_to_source_docs": False,
            "score_row_links_copied_to_source_docs": False,
        },
        "summary": summary,
        "request_results": request_results,
        "source_documents": source_documents,
    }


def run(
    *,
    collection_corpus_path: str | Path,
    source_jsonl_path: str | Path,
    report_json_path: str | Path,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    max_requests: int | None = None,
    property_only: bool = False,
    search_limit: int = DEFAULT_SEARCH_LIMIT,
    candidate_rank: int = DEFAULT_CANDIDATE_RANK,
    endpoint: str = DEFAULT_API_ENDPOINT,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_USER_AGENT,
    fetched_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Fetch, write, manifest, and optionally register Wikidata source docs."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    collection = _load_collection_corpus(collection_corpus_path)
    fetched = fetched_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    client = WikidataAPIClient(endpoint=endpoint, timeout=timeout, user_agent=user_agent)
    payload = build_wikidata_evidence_from_collection(
        collection,
        search_entities=client.search_entities,
        load_entities=client.load_entities,
        fetched_at=fetched,
        endpoint=endpoint,
        search_limit=search_limit,
        candidate_rank=candidate_rank,
        max_requests=max_requests,
        property_only=property_only,
    )
    payload["source"]["collection_corpus_path"] = str(collection_corpus_path)
    payload["metadata"] = dict(metadata or {})
    source_path = Path(source_jsonl_path)
    report_path = Path(report_json_path)
    _write_jsonl(source_path, payload["source_documents"])
    report_payload = dict(payload)
    report_payload["source_documents_path"] = str(source_path)
    report_payload["source_documents"] = {
        "count": len(payload["source_documents"]),
        "format": "jsonl",
        "path": str(source_path),
    }
    if artifact_manifest_path is not None:
        report_payload["paths"] = {"artifact_manifest": str(artifact_manifest_path)}
    _write_json(report_path, report_payload, compact=compact_json)
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            {
                "blind_spot_wikidata_evidence_report": report_path,
                "wikidata_source_docs": source_path,
                "blind_spot_evidence_collection_corpus": collection_corpus_path,
            },
            root=manifest_path.parent,
            metadata={
                "runner": "fetch_blind_spot_wikidata_evidence",
                "status": payload["status"],
                "request_count": payload["summary"]["request_count"],
                "document_count": payload["summary"]["document_count"],
                "resolved_entity_count": payload["summary"]["resolved_entity_count"],
                "endpoint": endpoint,
                "fetched_at": fetched,
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
                "request_count": payload["summary"]["request_count"],
                "document_count": payload["summary"]["document_count"],
                "resolved_entity_count": payload["summary"]["resolved_entity_count"],
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return report_payload


def _select_requests(
    collection: Mapping[str, Any],
    *,
    max_requests: int | None,
    property_only: bool,
) -> tuple[dict[str, Any], ...]:
    if collection.get("workflow") != "blind_spot_evidence_collection_corpus":
        raise ValueError("input must be a blind_spot_evidence_collection_corpus report.")
    raw_requests = collection.get("requests", {}).get("wikidata_entity_property")
    if not isinstance(raw_requests, Sequence) or isinstance(raw_requests, (str, bytes, bytearray)):
        raise ValueError("collection corpus is missing requests.wikidata_entity_property.")
    selected = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_requests:
        if not isinstance(raw, Mapping):
            continue
        request = dict(raw)
        if property_only and not request.get("property_id"):
            continue
        key = (
            str(request.get("entity", "")).casefold(),
            str(request.get("property_id") or request.get("property_hint", "")).casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(request)
        if max_requests is not None and len(selected) >= int(max_requests):
            break
    if not selected:
        raise ValueError("no Wikidata requests selected from collection corpus.")
    return tuple(selected)


def _resolve_entities(
    requests: Sequence[Mapping[str, Any]],
    *,
    search_entities: SearchFn,
    search_limit: int,
    candidate_rank: int,
) -> dict[str, dict[str, Any]]:
    resolution: dict[str, dict[str, Any]] = {}
    for entity in sorted({str(request.get("entity", "")).strip() for request in requests if request.get("entity")}):
        results = tuple(search_entities(entity, search_limit))
        if len(results) <= candidate_rank:
            resolution[entity] = {
                "entity": entity,
                "status": "unresolved",
                "candidate_count": len(results),
                "qid": None,
                "label": None,
                "description": None,
            }
            continue
        selected = dict(results[candidate_rank])
        resolution[entity] = {
            "entity": entity,
            "status": "resolved",
            "candidate_count": len(results),
            "candidate_rank": int(candidate_rank),
            "qid": str(selected.get("id")),
            "label": _search_result_label(selected),
            "description": _search_result_description(selected),
        }
    return resolution


def _preliminary_request_documents(
    requests: Sequence[Mapping[str, Any]],
    *,
    resolution_by_entity: Mapping[str, Mapping[str, Any]],
    subject_entities: Mapping[str, Mapping[str, Any]],
    fetched_at: str,
    endpoint: str,
) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    request_results: list[dict[str, Any]] = []
    label_ids: set[str] = set()
    seen_sources: set[str] = set()
    for request in requests:
        entity = str(request.get("entity", "")).strip()
        resolution = resolution_by_entity.get(entity, {})
        qid = resolution.get("qid")
        request_result = _request_result_base(request, resolution=resolution)
        if not qid:
            request_result["status"] = "unresolved_entity"
            request_results.append(request_result)
            continue
        subject_entity = subject_entities.get(str(qid), {})
        if not subject_entity:
            request_result["status"] = "missing_entity_payload"
            request_results.append(request_result)
            continue
        property_id = request.get("property_id")
        if property_id:
            claims = _claim_values(subject_entity, str(property_id))
            label_ids.add(str(property_id))
            if not claims:
                request_result["status"] = "resolved_no_claim"
                request_results.append(request_result)
                continue
            request_docs = []
            for claim in claims:
                value_id = claim.get("value_id")
                if value_id:
                    label_ids.add(str(value_id))
                source = _source_id(subject_qid=str(qid), property_id=str(property_id), claim=claim)
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                doc = _property_source_document(
                    request,
                    subject_entity=subject_entity,
                    subject_qid=str(qid),
                    property_id=str(property_id),
                    claim=claim,
                    source=source,
                    fetched_at=fetched_at,
                    endpoint=endpoint,
                )
                _assert_source_metadata_clean(doc["metadata"])
                request_docs.append(doc)
            documents.extend(request_docs)
            request_result["status"] = "documented" if request_docs else "duplicate_claims"
            request_result["document_count"] = len(request_docs)
            request_results.append(request_result)
            continue
        doc = _description_source_document(
            request,
            subject_entity=subject_entity,
            subject_qid=str(qid),
            resolution=resolution,
            fetched_at=fetched_at,
            endpoint=endpoint,
        )
        if doc is None:
            request_result["status"] = "resolved_no_description"
            request_results.append(request_result)
            continue
        if doc["source"] in seen_sources:
            request_result["status"] = "duplicate_description"
            request_results.append(request_result)
            continue
        seen_sources.add(doc["source"])
        _assert_source_metadata_clean(doc["metadata"])
        documents.append(doc)
        request_result["status"] = "documented"
        request_result["document_count"] = 1
        request_results.append(request_result)
    return {
        "documents": tuple(documents),
        "label_ids": label_ids,
        "request_results": tuple(request_results),
    }


def _property_source_document(
    request: Mapping[str, Any],
    *,
    subject_entity: Mapping[str, Any],
    subject_qid: str,
    property_id: str,
    claim: Mapping[str, Any],
    source: str,
    fetched_at: str,
    endpoint: str,
) -> dict[str, Any]:
    subject_label = _entity_label(subject_entity, fallback=subject_qid)
    value_label = str(claim.get("value_label") or claim.get("value") or claim.get("value_id") or "")
    property_label = str(claim.get("property_label") or property_id)
    metadata = {
        "provider": "wikidata",
        "license": "CC0-1.0",
        "license_url": WIKIDATA_LICENSE_URL,
        "endpoint": endpoint,
        "retrieved_at": fetched_at,
        "timestamp": fetched_at,
        "query_preset": "blind_spot_collection_request",
        "statement_property": property_id,
        "statement_property_label": property_label,
        "subject": subject_label,
        "subject_qid": subject_qid,
        "value": value_label,
        "value_qid": claim.get("value_id"),
        "value_datatype": claim.get("datatype"),
        "url": f"{WIKIDATA_ENTITY_URL}{subject_qid}",
        "collection_request_sha256": _request_fingerprint(request),
    }
    return {
        "text": _wikidata_fact_sentence(
            subject_label=subject_label,
            property_label=property_label,
            value_label=value_label,
        ),
        "source": source,
        "metadata": metadata,
        "_property_id": property_id,
        "_value_id": claim.get("value_id"),
    }


def _description_source_document(
    request: Mapping[str, Any],
    *,
    subject_entity: Mapping[str, Any],
    subject_qid: str,
    resolution: Mapping[str, Any],
    fetched_at: str,
    endpoint: str,
) -> dict[str, Any] | None:
    subject_label = _entity_label(subject_entity, fallback=str(resolution.get("label") or subject_qid))
    description = _entity_description(subject_entity) or str(resolution.get("description") or "").strip()
    if not description:
        return None
    metadata = {
        "provider": "wikidata",
        "license": "CC0-1.0",
        "license_url": WIKIDATA_LICENSE_URL,
        "endpoint": endpoint,
        "retrieved_at": fetched_at,
        "timestamp": fetched_at,
        "query_preset": "blind_spot_collection_request",
        "statement_property": "description",
        "statement_property_label": "description",
        "subject": subject_label,
        "subject_qid": subject_qid,
        "value": description,
        "value_qid": None,
        "value_datatype": "wikibase-description",
        "url": f"{WIKIDATA_ENTITY_URL}{subject_qid}",
        "collection_request_sha256": _request_fingerprint(request),
    }
    return {
        "text": f"According to Wikidata entity metadata, {subject_label} is described as {description}.",
        "source": f"wikidata:{subject_qid}:description",
        "metadata": metadata,
        "_property_id": "description",
        "_value_id": None,
    }


def _finalize_documents(
    documents: Sequence[Mapping[str, Any]],
    *,
    labels: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    finalized = []
    for document in documents:
        payload = dict(document)
        property_id = str(payload.pop("_property_id", ""))
        value_id = payload.pop("_value_id", None)
        metadata = dict(payload["metadata"])
        if property_id and property_id != "description":
            metadata["statement_property_label"] = labels.get(property_id, metadata["statement_property_label"])
        if value_id:
            metadata["value"] = labels.get(str(value_id), metadata["value"])
        payload["metadata"] = metadata
        if property_id and property_id != "description":
            payload["text"] = _wikidata_fact_sentence(
                subject_label=str(metadata["subject"]),
                property_label=str(metadata["statement_property_label"]),
                value_label=str(metadata["value"]),
            )
        _assert_source_metadata_clean(metadata)
        finalized.append(payload)
    return tuple(finalized)


def _finalize_request_results(
    request_results: Sequence[Mapping[str, Any]],
    *,
    labels: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    finalized = []
    for result in request_results:
        payload = dict(result)
        qid = payload.get("resolved_qid")
        if qid:
            payload["resolved_label"] = labels.get(str(qid), payload.get("resolved_label"))
        property_id = payload.get("property_id")
        if property_id:
            payload["property_label"] = labels.get(str(property_id), property_id)
        finalized.append(payload)
    return tuple(finalized)


def _claim_values(entity: Mapping[str, Any], property_id: str) -> tuple[dict[str, Any], ...]:
    claims = entity.get("claims", {}).get(property_id, ())
    if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes, bytearray)):
        return ()
    values = []
    seen: set[tuple[str | None, str]] = set()
    for claim in claims:
        if not isinstance(claim, Mapping):
            continue
        mainsnak = claim.get("mainsnak", {})
        if not isinstance(mainsnak, Mapping):
            continue
        parsed = _parse_snak_value(mainsnak)
        if parsed is None:
            continue
        key = (parsed.get("value_id"), str(parsed.get("value")))
        if key in seen:
            continue
        seen.add(key)
        values.append(parsed)
    return tuple(values)


def _parse_snak_value(snak: Mapping[str, Any]) -> dict[str, Any] | None:
    datavalue = snak.get("datavalue")
    if not isinstance(datavalue, Mapping):
        return None
    datatype = str(snak.get("datatype") or datavalue.get("type") or "")
    value = datavalue.get("value")
    if isinstance(value, Mapping) and "id" in value:
        value_id = str(value["id"])
        return {
            "datatype": datatype,
            "value": value_id,
            "value_id": value_id,
            "value_label": value_id,
        }
    if isinstance(value, Mapping) and "text" in value:
        text = str(value["text"]).strip()
        return {"datatype": datatype, "value": text, "value_id": None, "value_label": text}
    if isinstance(value, Mapping) and "time" in value:
        text = str(value["time"]).strip()
        return {"datatype": datatype, "value": text, "value_id": None, "value_label": _clean_time_value(text)}
    if isinstance(value, Mapping) and "amount" in value:
        text = _quantity_value(value)
        return {"datatype": datatype, "value": text, "value_id": None, "value_label": text}
    if isinstance(value, Mapping) and {"latitude", "longitude"}.issubset(value):
        text = f"{value['latitude']},{value['longitude']}"
        return {"datatype": datatype, "value": text, "value_id": None, "value_label": text}
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return {"datatype": datatype, "value": text, "value_id": None, "value_label": text}


def _summary(
    *,
    requests: Sequence[Mapping[str, Any]],
    request_results: Sequence[Mapping[str, Any]],
    source_documents: Sequence[Mapping[str, Any]],
    resolved_entities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(str(result.get("status")) for result in request_results)
    property_counts = Counter(
        str(document.get("metadata", {}).get("statement_property"))
        for document in source_documents
    )
    return {
        "request_count": len(requests),
        "resolved_entity_count": sum(1 for item in resolved_entities.values() if item.get("status") == "resolved"),
        "unresolved_entity_count": sum(1 for item in resolved_entities.values() if item.get("status") != "resolved"),
        "document_count": len(source_documents),
        "documented_request_count": status_counts.get("documented", 0),
        "request_status_counts": _sorted_counter(status_counts),
        "document_property_counts": _sorted_counter(property_counts),
        "source_count": len({str(item.get("source")) for item in source_documents}),
    }


def _request_result_base(request: Mapping[str, Any], *, resolution: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "request_id": str(request.get("request_id", "")),
        "target_id": str(request.get("target_id", "")),
        "entity": str(request.get("entity", "")),
        "property_id": request.get("property_id"),
        "property_hint": request.get("property_hint"),
        "resolved_qid": resolution.get("qid"),
        "resolved_label": resolution.get("label"),
        "resolution_status": resolution.get("status"),
        "document_count": 0,
    }


def _load_collection_corpus(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("collection corpus must be a JSON object.")
    if payload.get("workflow") != "blind_spot_evidence_collection_corpus":
        raise ValueError(f"{path} is not a blind_spot_evidence_collection_corpus report.")
    return dict(payload)


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


def _label_lookup(entities: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    labels = {}
    for entity_id, entity in entities.items():
        label = _entity_label(entity, fallback=str(entity_id))
        labels[str(entity_id)] = label
    return labels


def _entity_label(entity: Mapping[str, Any], *, fallback: str) -> str:
    labels = entity.get("labels", {})
    if isinstance(labels, Mapping):
        en = labels.get("en")
        if isinstance(en, Mapping) and en.get("value"):
            return str(en["value"])
    return fallback


def _entity_description(entity: Mapping[str, Any]) -> str | None:
    descriptions = entity.get("descriptions", {})
    if not isinstance(descriptions, Mapping):
        return None
    en = descriptions.get("en")
    if not isinstance(en, Mapping) or en.get("value") is None:
        return None
    text = str(en["value"]).strip()
    return text or None


def _search_result_label(result: Mapping[str, Any]) -> str | None:
    label = result.get("label")
    return None if label is None else str(label)


def _search_result_description(result: Mapping[str, Any]) -> str | None:
    description = result.get("description")
    return None if description is None else str(description)


def _request_fingerprint(request: Mapping[str, Any]) -> str:
    stable = {
        "entity": request.get("entity"),
        "property_id": request.get("property_id"),
        "property_hint": request.get("property_hint"),
        "request_type": request.get("request_type"),
    }
    encoded = strict_json_dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_id(*, subject_qid: str, property_id: str, claim: Mapping[str, Any]) -> str:
    value = str(claim.get("value_id") or claim.get("value") or "")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    suffix = value if _is_wikidata_id(value) else digest
    return f"wikidata:{subject_qid}:{property_id}:{suffix}"


def _wikidata_fact_sentence(*, subject_label: str, property_label: str, value_label: str) -> str:
    text = f"According to Wikidata structured data, {subject_label} has {property_label} {value_label}"
    if text.endswith((".", "?", "!")):
        return text
    return f"{text}."


def _quantity_value(value: Mapping[str, Any]) -> str:
    amount = str(value.get("amount", "")).lstrip("+")
    unit = str(value.get("unit", ""))
    if unit and unit != "1":
        return f"{amount} {unit}"
    return amount


def _clean_time_value(value: str) -> str:
    return value.lstrip("+")


def _normalize_wikidata_id(value: str) -> str:
    text = str(value).strip()
    if not _is_wikidata_id(text):
        raise ValueError(f"invalid Wikidata id: {value!r}")
    return text


def _is_wikidata_id(value: str) -> bool:
    return bool(WIKIDATA_ID_RE.fullmatch(str(value).strip()))


def _assert_source_metadata_clean(metadata: Mapping[str, Any]) -> None:
    reserved = sorted(set(str(key) for key in metadata) & RESERVED_SOURCE_METADATA_KEYS)
    if reserved:
        raise ValueError(
            "Wikidata source document metadata contains score-row or request-link keys: "
            f"{', '.join(reserved)}"
        )


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


def _chunks(values: Sequence[str], size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(values[index:index + size]) for index in range(0, len(values), size))


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


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
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--property-only", action="store_true")
    parser.add_argument("--search-limit", type=int, default=DEFAULT_SEARCH_LIMIT)
    parser.add_argument("--candidate-rank", type=int, default=DEFAULT_CANDIDATE_RANK)
    parser.add_argument("--endpoint", default=DEFAULT_API_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--fetched-at", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        collection_corpus_path=args.collection_corpus,
        source_jsonl_path=args.source_jsonl,
        report_json_path=args.report_json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        max_requests=args.max_requests,
        property_only=bool(args.property_only),
        search_limit=args.search_limit,
        candidate_rank=args.candidate_rank,
        endpoint=args.endpoint,
        timeout=args.timeout,
        user_agent=args.user_agent,
        fetched_at=args.fetched_at,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "blind_spot_wikidata_evidence_fetch_ok "
        f"status={payload['status']} "
        f"requests={summary['request_count']} "
        f"resolved_entities={summary['resolved_entity_count']} "
        f"documents={summary['document_count']}"
    )


if __name__ == "__main__":
    main()
