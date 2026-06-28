"""Execute source-family structured QA fact-collection requests locally.

The fact collection corpus is an adapter-ready request plan, not verifier
evidence. This workflow consumes those request sidecars through the existing
dependency-free source-family catalog ranker, emits candidate source matches,
and builds a conservative structured QA candidate corpus from matched
structured metadata. World-model/calculator requests are preserved as rule
authoring stubs for a later deterministic adapter.
"""

from __future__ import annotations

import argparse
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

from benchmarks.build_source_family_qa_corpus import run as build_source_family_qa_corpus  # noqa: E402
from benchmarks.run_source_family_citation_search_adapter import (  # noqa: E402
    DEFAULT_SOURCE_FAMILY,
    run_source_family_citation_search_adapter,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "source_family_structured_qa_fact_collection_workflow"
COLLECTION_WORKFLOW = "source_family_structured_qa_fact_collection_corpus"
SOURCE_BACKED_REQUEST_TYPES = (
    "source_family_structured_fact",
    "entity_resolution",
    "external_citation",
    "source_family_fact_disambiguation",
)
RULE_REQUEST_TYPE = "world_model_or_calculator_rule"
SOURCE_FAMILY_OFFICIAL = {"official", "official_site", "official_statistics", "domain_specific"}
RESERVED_SOURCE_DOC_FIELDS = {
    "answer",
    "claim_id",
    "is_false",
    "label",
    "labels",
    "model_answer",
    "record_index",
    "row_index",
    "score_label",
    "target_id",
}


def run_source_family_structured_qa_fact_collection_workflow(
    *,
    collection_corpus_path: str | Path,
    source_catalog_paths: Sequence[str | Path],
    output_dir: str | Path,
    workflow_report_path: str | Path | None = None,
    combined_results_path: str | Path | None = None,
    qa_corpus_path: str | Path | None = None,
    qa_report_path: str | Path | None = None,
    rule_stubs_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    request_types: Sequence[str] = SOURCE_BACKED_REQUEST_TYPES,
    max_requests_per_type: int | None = None,
    adapter_max_results: int = 3,
    adapter_max_query_variants: int = 3,
    adapter_min_text_overlap: float = 0.05,
    adapter_diversify_source_families: bool = True,
    default_source_family: str = DEFAULT_SOURCE_FAMILY,
    keep_qid_values: bool = False,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
    fail_on_blocked: bool = False,
) -> dict[str, Any]:
    """Run local catalog matching over a fact-collection request corpus."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if not source_catalog_paths:
        raise ValueError("source_catalog_paths must contain at least one path.")
    if max_requests_per_type is not None and int(max_requests_per_type) <= 0:
        raise ValueError("max_requests_per_type must be positive when provided.")
    if adapter_max_results <= 0:
        raise ValueError("adapter_max_results must be positive.")
    if adapter_max_query_variants <= 0:
        raise ValueError("adapter_max_query_variants must be positive.")
    if not (0.0 <= adapter_min_text_overlap <= 1.0):
        raise ValueError("adapter_min_text_overlap must be in [0, 1].")

    collection = _load_collection(collection_corpus_path)
    selected_request_types = _normalize_request_types(request_types)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    report_path = Path(workflow_report_path or output / "fact-collection-workflow.json")
    results_path = Path(combined_results_path or output / "fact-collection-adapter-results.jsonl")
    qa_path = Path(qa_corpus_path or output / "source-family-structured-qa-corpus.json")
    qa_report = Path(qa_report_path or output / "source-family-structured-qa-corpus-report.json")
    qa_manifest = output / "source-family-structured-qa-corpus-manifest.json"
    rules_path = Path(rule_stubs_path or output / "world-model-rule-stubs.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    workflow_metadata = {
        **dict(metadata or {}),
        "source_workflow": WORKFLOW,
        "collection_workflow": COLLECTION_WORKFLOW,
    }
    request_index: dict[str, dict[str, Any]] = {}
    combined_rows: list[dict[str, Any]] = []
    request_paths: dict[str, str] = {}
    result_paths: dict[str, str] = {}
    adapter_reports: dict[str, dict[str, Any]] = {}

    for request_type in selected_request_types:
        raw_requests = _collection_requests(collection, request_type)
        if max_requests_per_type is not None:
            raw_requests = raw_requests[: int(max_requests_per_type)]
        normalized = tuple(_adapter_request(request) for request in raw_requests)
        for request in raw_requests:
            request_id = _clean(request.get("request_id"))
            if request_id:
                request_index[request_id] = dict(request)

        stem = request_type.replace("_", "-")
        request_path = output / f"{stem}-adapter-requests.jsonl"
        result_path = output / f"{stem}-adapter-results.jsonl"
        adapter_report_path = output / f"{stem}-adapter-report.json"
        adapter_manifest_path = output / f"{stem}-adapter-manifest.json"
        _write_jsonl(request_path, normalized, compact=compact_json)
        adapter_payload = run_source_family_citation_search_adapter(
            input_path=request_path,
            output_path=result_path,
            source_catalog_paths=tuple(source_catalog_paths),
            report_json_path=adapter_report_path,
            artifact_manifest_path=adapter_manifest_path,
            max_results=adapter_max_results,
            max_query_variants=adapter_max_query_variants,
            min_text_overlap=adapter_min_text_overlap,
            diversify_source_families=adapter_diversify_source_families,
            default_source_family=default_source_family,
            compact_json=compact_json,
            metadata=workflow_metadata,
        )
        request_paths[request_type] = str(request_path)
        result_paths[request_type] = str(result_path)
        adapter_reports[request_type] = dict(adapter_payload)
        for row in _load_jsonl(result_path):
            combined_rows.append(_enrich_adapter_row(row, request_index=request_index))

    _write_jsonl(results_path, combined_rows, compact=compact_json)
    rule_stubs = _rule_stubs(collection, max_items=max_requests_per_type)
    _write_jsonl(rules_path, rule_stubs, compact=compact_json)

    qa_payload = build_source_family_qa_corpus(
        source_paths=(results_path,),
        output_path=qa_path,
        report_json_path=qa_report,
        artifact_manifest_path=qa_manifest,
        skip_qid_values=not bool(keep_qid_values),
        metadata=workflow_metadata,
        compact_json=compact_json,
    )
    summary = _summary(
        collection=collection,
        adapter_reports=adapter_reports,
        combined_rows=combined_rows,
        rule_stubs=rule_stubs,
        qa_payload=qa_payload,
        selected_request_types=selected_request_types,
    )
    status = _status(summary)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Local catalog execution over source-discovery and rule-authoring "
            "requests. Adapter results and rule stubs are candidate inputs only; "
            "route promotion still requires structured QA or external evidence gates."
        ),
        "source": {
            "collection_corpus": str(collection_corpus_path),
            "collection_workflow": collection.get("workflow"),
            "source_catalogs": tuple(str(path) for path in source_catalog_paths),
        },
        "config": {
            "request_types": selected_request_types,
            "max_requests_per_type": max_requests_per_type,
            "adapter_max_results": int(adapter_max_results),
            "adapter_max_query_variants": int(adapter_max_query_variants),
            "adapter_min_text_overlap": float(adapter_min_text_overlap),
            "adapter_diversify_source_families": bool(adapter_diversify_source_families),
            "default_source_family": str(default_source_family),
            "keep_qid_values": bool(keep_qid_values),
        },
        "label_usage": {
            "labels_used_for_adapter_execution": False,
            "labels_copied_to_adapter_requests": False,
            "model_answers_copied_to_adapter_requests": False,
            "adapter_results_are_verifier_evidence": False,
            "rule_stubs_are_verifier_evidence": False,
        },
        "paths": {
            "workflow_report": str(report_path),
            "combined_adapter_results": str(results_path),
            "structured_qa_corpus": str(qa_path),
            "structured_qa_report": str(qa_report),
            "structured_qa_manifest": str(qa_manifest),
            "world_model_rule_stubs": str(rules_path),
            "artifact_manifest": str(manifest_path),
            "adapter_requests": request_paths,
            "adapter_results_by_type": result_paths,
        },
        "summary": summary,
        "metadata": dict(metadata or {}),
    }
    _write_json(report_path, payload, compact=compact_json)
    manifest = build_artifact_manifest(
        {
            "collection_corpus": Path(collection_corpus_path),
            "workflow_report": report_path,
            "combined_adapter_results": results_path,
            "structured_qa_corpus": qa_path,
            "structured_qa_report": qa_report,
            "structured_qa_manifest": qa_manifest,
            "world_model_rule_stubs": rules_path,
            **{f"source_catalog_{idx}": Path(path) for idx, path in enumerate(source_catalog_paths, start=1)},
            **{f"{key}_adapter_requests": Path(path) for key, path in request_paths.items()},
            **{f"{key}_adapter_results": Path(path) for key, path in result_paths.items()},
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": status,
            "source_backed_request_count": summary["source_backed_request_count"],
            "request_with_results_count": summary["request_with_results_count"],
            "adapter_result_count": summary["adapter_result_count"],
            "structured_qa_document_count": summary["structured_qa_document_count"],
            "rule_stub_count": summary["rule_stub_count"],
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
                "source_backed_request_count": summary["source_backed_request_count"],
                "request_with_results_count": summary["request_with_results_count"],
                "adapter_result_count": summary["adapter_result_count"],
                "structured_qa_document_count": summary["structured_qa_document_count"],
                "rule_stub_count": summary["rule_stub_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    if fail_on_blocked and status == "blocked":
        raise SystemExit(1)
    return payload


def _load_collection(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("collection corpus must be a JSON object.")
    if payload.get("workflow") != COLLECTION_WORKFLOW:
        raise ValueError(f"{path} is not a {COLLECTION_WORKFLOW} report.")
    return dict(payload)


def _normalize_request_types(values: Sequence[str]) -> tuple[str, ...]:
    request_types = tuple(dict.fromkeys(_clean(item) for item in values if _clean(item)))
    if not request_types:
        raise ValueError("at least one request type is required.")
    invalid = sorted(set(request_types) - set(SOURCE_BACKED_REQUEST_TYPES))
    if invalid:
        raise ValueError(f"unsupported source-backed request types: {', '.join(invalid)}")
    return request_types


def _collection_requests(collection: Mapping[str, Any], request_type: str) -> tuple[dict[str, Any], ...]:
    raw_requests = collection.get("requests", {})
    if not isinstance(raw_requests, Mapping):
        raise ValueError("collection corpus requests field must be an object.")
    values = raw_requests.get(request_type, ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"collection request bucket {request_type!r} must be a list.")
    return tuple(dict(item) for item in values if isinstance(item, Mapping))


def _adapter_request(request: Mapping[str, Any]) -> dict[str, Any]:
    request_id = _clean(request.get("request_id"))
    query = _clean(request.get("query") or request.get("entity") or request.get("question"))
    if not request_id:
        raise ValueError("collection request is missing request_id.")
    if not query:
        raise ValueError(f"collection request {request_id} is missing query text.")
    return {
        "request_id": request_id,
        "query": query,
        "alternate_queries": _alternate_queries(request, query=query),
        "source_family_plan": _source_family_plan(request),
        "metadata": {
            "collection_workflow": COLLECTION_WORKFLOW,
            "collection_request_type": _clean(request.get("request_type")),
            "priority": _clean(request.get("priority")),
            "question_type": _clean(request.get("question_type")),
            "gap_type": _clean(request.get("gap_type")),
            "not_verifier_evidence": True,
        },
    }


def _alternate_queries(request: Mapping[str, Any], *, query: str) -> tuple[str, ...]:
    values = [
        _clean(request.get("question")),
        _clean(request.get("entity")),
        " ".join(_string_sequence(request.get("entity_candidates", ()))),
        " ".join(_string_sequence(request.get("entities", ()))),
    ]
    variants: list[str] = []
    seen = {query.casefold()}
    for value in values:
        if not value:
            continue
        folded = value.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        variants.append(value)
    return tuple(variants)


def _source_family_plan(request: Mapping[str, Any]) -> dict[str, Any]:
    families: list[str] = []
    source_family = _clean(request.get("source_family"))
    if source_family:
        families.append(source_family)
    families.extend(_string_sequence(request.get("source_family_hints", ())))
    if not families and "official" in _clean(request.get("provider_hint")).casefold():
        families.append("official_site")
    normalized = tuple(dict.fromkeys(_normalize_family(item) for item in families if _normalize_family(item)))
    hints = (
        *_string_sequence(request.get("property_hints", ())),
        *_string_sequence(request.get("property_ids", ())),
        *_string_sequence(request.get("entity_candidates", ())),
        *_string_sequence(request.get("entities", ())),
    )
    official_preferred = any(item in SOURCE_FAMILY_OFFICIAL for item in normalized)
    return {
        "families": normalized or (DEFAULT_SOURCE_FAMILY,),
        "freshness_required": bool(request.get("requires_timestamp")) or _clean(
            request.get("question_type")
        ).casefold() == "temporal",
        "official_source_preferred": official_preferred,
        "query_hints": tuple(dict.fromkeys(_clean(item) for item in hints if _clean(item))),
    }


def _enrich_adapter_row(
    row: Mapping[str, Any],
    *,
    request_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    request_id = _clean(row.get("request_id"))
    request = request_index.get(request_id, {})
    metadata = dict(_mapping(row.get("metadata")))
    metadata.update({
        "collection_workflow": COLLECTION_WORKFLOW,
        "collection_request_type": _clean(request.get("request_type")),
        "collection_priority": _clean(request.get("priority")),
        "collection_question_type": _clean(request.get("question_type")),
        "collection_gap_type": _clean(request.get("gap_type")),
        "not_verifier_evidence": True,
    })
    return {
        **dict(row),
        "target_id": _clean(request.get("target_id")),
        "request_type": _clean(request.get("request_type")),
        "priority": _clean(request.get("priority")),
        "usage": "candidate_source_match",
        "not_verifier_evidence": True,
        "metadata": metadata,
    }


def _rule_stubs(collection: Mapping[str, Any], *, max_items: int | None) -> tuple[dict[str, Any], ...]:
    requests = _collection_requests(collection, RULE_REQUEST_TYPE)
    if max_items is not None:
        requests = requests[: int(max_items)]
    stubs = []
    for request in requests:
        stubs.append({
            "schema_version": 1,
            "workflow": WORKFLOW,
            "request_id": _clean(request.get("request_id")),
            "target_id": _clean(request.get("target_id")),
            "request_type": RULE_REQUEST_TYPE,
            "status": "requires_deterministic_rule_adapter",
            "rule_family": _clean(request.get("rule_family")) or "world_model_consistency",
            "rule_reason": _clean(request.get("rule_reason")),
            "rule_seed": _clean(request.get("rule_seed")),
            "required_inputs": _string_sequence(request.get("required_inputs", ())),
            "not_verifier_evidence": True,
        })
    return tuple(stubs)


def _summary(
    *,
    collection: Mapping[str, Any],
    adapter_reports: Mapping[str, Mapping[str, Any]],
    combined_rows: Sequence[Mapping[str, Any]],
    rule_stubs: Sequence[Mapping[str, Any]],
    qa_payload: Mapping[str, Any],
    selected_request_types: Sequence[str],
) -> dict[str, Any]:
    adapter_summaries = {
        request_type: dict(report.get("summary", {}))
        for request_type, report in adapter_reports.items()
    }
    row_counts = [len(_result_items(row.get("results"))) for row in combined_rows]
    result_docs = tuple(result for row in combined_rows for result in _result_items(row.get("results")))
    qa_report = _mapping(qa_payload.get("report"))
    qa_summary = dict(_mapping(qa_report.get("summary")))
    request_counts = {
        request_type: len(_collection_requests(collection, request_type))
        for request_type in (*SOURCE_BACKED_REQUEST_TYPES, RULE_REQUEST_TYPE)
    }
    selected_counts = {
        request_type: int(adapter_summaries.get(request_type, {}).get("request_count", 0))
        for request_type in selected_request_types
    }
    return {
        "collection_target_count": _nested_int(collection, "summary", "target_count"),
        "collection_total_request_count": _nested_int(collection, "summary", "total_request_count"),
        "collection_request_counts": request_counts,
        "selected_request_types": tuple(selected_request_types),
        "selected_request_counts": selected_counts,
        "source_catalog_document_count": max(
            (int(summary.get("source_document_count", 0)) for summary in adapter_summaries.values()),
            default=0,
        ),
        "source_backed_request_count": sum(selected_counts.values()),
        "adapter_row_count": len(combined_rows),
        "request_with_results_count": sum(1 for count in row_counts if count > 0),
        "adapter_result_count": sum(row_counts),
        "adapter_error_count": sum(1 for row in combined_rows if row.get("error")),
        "result_provider_counts": _sorted_counter(Counter(_clean(result.get("provider")) for result in result_docs)),
        "result_source_family_counts": _sorted_counter(
            Counter(_clean(result.get("source_family")) for result in result_docs)
        ),
        "adapter_summaries_by_type": adapter_summaries,
        "structured_qa_status": qa_report.get("status"),
        "structured_qa_document_count": int(qa_summary.get("n_documents", 0)),
        "structured_qa_candidate_document_count": int(qa_summary.get("n_candidate_documents", 0)),
        "structured_qa_skipped": dict(_mapping(qa_summary.get("skipped"))),
        "rule_stub_count": len(rule_stubs),
        "reserved_source_document_field_hits": _reserved_source_document_field_hits(result_docs),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if int(summary.get("structured_qa_document_count", 0)) > 0:
        return "ready_for_fact_mapping"
    if int(summary.get("adapter_result_count", 0)) > 0 or int(summary.get("rule_stub_count", 0)) > 0:
        return "observed"
    return "blocked"


def _reserved_source_document_field_hits(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for result in results:
        for key in RESERVED_SOURCE_DOC_FIELDS:
            if key in result:
                counter[key] += 1
        metadata = _mapping(result.get("metadata"))
        for key in RESERVED_SOURCE_DOC_FIELDS:
            if key in metadata:
                counter[f"metadata.{key}"] += 1
    return _sorted_counter(counter)


def _nested_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _result_items(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(_clean(item) for item in value if _clean(item))
    return ()


def _normalize_family(value: Any) -> str:
    return _clean(value).casefold().replace("-", "_").replace(" ", "_")


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(((key, value) for key, value in counter.items() if key), key=lambda item: (-item[1], item[0])))


def _load_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no} must contain a JSON object.")
            rows.append(dict(payload))
    return tuple(rows)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = "".join(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    else:
        text = "".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


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
    parser.add_argument("--source-catalog", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--combined-results-jsonl", default=None)
    parser.add_argument("--qa-corpus", default=None)
    parser.add_argument("--qa-report", default=None)
    parser.add_argument("--rule-stubs-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--request-type", action="append", default=None)
    parser.add_argument("--max-requests-per-type", type=int, default=None)
    parser.add_argument("--adapter-max-results", type=int, default=3)
    parser.add_argument("--adapter-max-query-variants", type=int, default=3)
    parser.add_argument("--adapter-min-text-overlap", type=float, default=0.05)
    parser.add_argument("--no-adapter-diversify-source-families", action="store_true")
    parser.add_argument("--default-source-family", default=DEFAULT_SOURCE_FAMILY)
    parser.add_argument("--keep-qid-values", action="store_true")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)
    payload = run_source_family_structured_qa_fact_collection_workflow(
        collection_corpus_path=args.collection_corpus,
        source_catalog_paths=tuple(args.source_catalog or ()),
        output_dir=args.output_dir,
        workflow_report_path=args.json,
        combined_results_path=args.combined_results_jsonl,
        qa_corpus_path=args.qa_corpus,
        qa_report_path=args.qa_report,
        rule_stubs_path=args.rule_stubs_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        request_types=tuple(args.request_type or SOURCE_BACKED_REQUEST_TYPES),
        max_requests_per_type=args.max_requests_per_type,
        adapter_max_results=args.adapter_max_results,
        adapter_max_query_variants=args.adapter_max_query_variants,
        adapter_min_text_overlap=args.adapter_min_text_overlap,
        adapter_diversify_source_families=not bool(args.no_adapter_diversify_source_families),
        default_source_family=args.default_source_family,
        keep_qid_values=bool(args.keep_qid_values),
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
        fail_on_blocked=bool(args.fail_on_blocked),
    )
    summary = payload["summary"]
    print(
        "source_family_structured_qa_fact_collection_workflow_ok "
        f"status={payload['status']} "
        f"requests={summary['source_backed_request_count']} "
        f"results={summary['adapter_result_count']} "
        f"qa_docs={summary['structured_qa_document_count']} "
        f"rules={summary['rule_stub_count']}"
    )


if __name__ == "__main__":
    main()
