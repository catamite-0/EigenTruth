"""Fetch OpenAlex works for scholarly source-family catalog tasks.

This adapter consumes the non-evidence collection-task JSONL produced by
``plan_source_family_catalog_collection.py`` and writes adapter-ready
``source_family=scholarly`` catalog documents for
``run_source_family_citation_search_workflow.py``. It uses only the Python
standard library. The emitted catalog rows deliberately omit labels, target ids,
record ids, model answers, and request ids; request coverage remains in the
report, while source documents only carry safe collection-task provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
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

WORKFLOW = "openalex_source_family_catalog_adapter"
PROVIDER = "openalex"
SOURCE_FAMILY = "scholarly"
API_BASE_URL = "https://api.openalex.org/works"
DEFAULT_USER_AGENT = "EigenTruth/0.2 (https://github.com/catamitez0-maker/EigenTruth)"
DEFAULT_SELECT_FIELDS = (
    "id",
    "doi",
    "display_name",
    "publication_year",
    "publication_date",
    "type",
    "cited_by_count",
    "authorships",
    "primary_location",
    "abstract_inverted_index",
    "open_access",
    "concepts",
    "keywords",
    "relevance_score",
)
RESERVED_FIELDS = {
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


def run_openalex_source_family_catalog_adapter(
    *,
    tasks_path: str | Path,
    output_path: str | Path,
    report_json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    source_family: str = SOURCE_FAMILY,
    max_tasks: int | None = None,
    max_query_variants: int = 2,
    rows_per_query: int = 2,
    min_delay_seconds: float = 0.0,
    timeout_seconds: float = 20.0,
    mailto: str | None = None,
    api_key: str | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    include_abstracts: bool = False,
    compact_json: bool = False,
    fail_on_empty: bool = False,
    metadata: Mapping[str, Any] | None = None,
    fetch_json: Callable[[str, Mapping[str, str]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fetch OpenAlex metadata and write source-family catalog rows."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be positive when provided.")
    if max_query_variants <= 0:
        raise ValueError("max_query_variants must be positive.")
    if rows_per_query <= 0:
        raise ValueError("rows_per_query must be positive.")
    if min_delay_seconds < 0:
        raise ValueError("min_delay_seconds must be non-negative.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")

    tasks = tuple(
        task
        for task in _load_tasks(tasks_path)
        if _clean(task.get("source_family")) == source_family
    )
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    if not tasks:
        raise ValueError(f"no collection tasks found for source_family={source_family!r}.")

    fetch = fetch_json or _default_fetch_json(timeout_seconds=timeout_seconds)
    fetched_at = datetime.now(timezone.utc).isoformat()
    docs_by_key: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    query_count = 0
    skipped_duplicate_count = 0
    for task in tasks:
        task_id = _clean(task.get("task_id"))
        query_variants = _string_sequence(task.get("search_queries", ()))[:max_query_variants]
        if not task_id or not query_variants:
            errors.append({"task_id": task_id, "error": "missing_task_id_or_search_queries"})
            continue
        for query in query_variants:
            query_count += 1
            try:
                payload = fetch(
                    _openalex_url(
                        query=query,
                        rows=rows_per_query,
                        mailto=mailto,
                        api_key=api_key,
                    ),
                    {"User-Agent": user_agent},
                )
            except Exception as exc:  # noqa: BLE001 - adapter report records fetch failure details.
                errors.append({"task_id": task_id, "query": query, "error": type(exc).__name__, "message": str(exc)})
                continue
            for item in _openalex_items(payload):
                row = _catalog_row_from_openalex_item(
                    item,
                    task=task,
                    query=query,
                    source_family=source_family,
                    fetched_at=fetched_at,
                    include_abstracts=include_abstracts,
                )
                if row is None:
                    continue
                key = _document_key(row)
                existing = docs_by_key.get(key)
                if existing is not None:
                    skipped_duplicate_count += 1
                    _merge_document_metadata(existing, row)
                    continue
                docs_by_key[key] = row
            if min_delay_seconds:
                time.sleep(float(min_delay_seconds))
    rows = tuple(docs_by_key.values())
    _write_jsonl(output_path, rows, compact=compact_json)
    summary = _summary(
        rows,
        tasks=tasks,
        query_count=query_count,
        errors=errors,
        skipped_duplicate_count=skipped_duplicate_count,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready" if rows else "empty",
        "source": {
            "collection_tasks": str(tasks_path),
        },
        "output_path": str(output_path),
        "config": {
            "source_family": source_family,
            "max_tasks": max_tasks,
            "max_query_variants": int(max_query_variants),
            "rows_per_query": int(rows_per_query),
            "min_delay_seconds": float(min_delay_seconds),
            "timeout_seconds": float(timeout_seconds),
            "include_abstracts": bool(include_abstracts),
            "api_base_url": API_BASE_URL,
            "select_fields": DEFAULT_SELECT_FIELDS,
            "uses_api_key": bool(api_key),
        },
        "summary": summary,
        "errors": tuple(errors[:20]),
        "metadata": dict(metadata or {}),
    }
    if report_json_path is not None:
        _write_json(report_json_path, payload, compact=compact_json)
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        artifacts: dict[str, str | Path] = {
            "collection_tasks": Path(tasks_path),
            "source_family_catalog": Path(output_path),
        }
        if report_json_path is not None:
            artifacts["openalex_catalog_report"] = Path(report_json_path)
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "task_count": summary["task_count"],
                "query_count": summary["query_count"],
                "source_document_count": summary["source_document_count"],
                "error_count": summary["error_count"],
                **dict(metadata or {}),
            },
        )
        _write_json(manifest_path, manifest, compact=compact_json)
        payload["artifact_manifest"] = str(manifest_path)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_json_path or output_path,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "task_count": summary["task_count"],
                "query_count": summary["query_count"],
                "source_document_count": summary["source_document_count"],
                "error_count": summary["error_count"],
                "artifact_manifest": payload.get("artifact_manifest"),
                **dict(metadata or {}),
            },
        ).save_json()
    if fail_on_empty and not rows:
        raise SystemExit(1)
    return payload


def _openalex_url(*, query: str, rows: int, mailto: str | None, api_key: str | None) -> str:
    params = {
        "search": _sanitize_openalex_search_query(query),
        "per-page": str(rows),
        "select": ",".join(DEFAULT_SELECT_FIELDS),
    }
    if mailto:
        params["mailto"] = mailto
    if api_key:
        params["api_key"] = api_key
    return f"{API_BASE_URL}?{urllib.parse.urlencode(params)}"


def _sanitize_openalex_search_query(query: str) -> str:
    return _clean(str(query).replace("?", " ").replace("*", " "))


def _catalog_row_from_openalex_item(
    item: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    query: str,
    source_family: str,
    fetched_at: str,
    include_abstracts: bool,
) -> dict[str, Any] | None:
    openalex_id = _clean(item.get("id"))
    doi = _clean(item.get("doi"))
    title = _clean(item.get("display_name"))
    if not title:
        return None
    primary_location = _mapping(item.get("primary_location"))
    source_meta = _mapping(primary_location.get("source"))
    journal = _clean(source_meta.get("display_name") or primary_location.get("raw_source_name"))
    publisher = _clean(source_meta.get("host_organization_name"))
    url = _clean(primary_location.get("landing_page_url") or item.get("landing_page_url") or doi or openalex_id)
    published_at = _clean(item.get("publication_date"))
    text_parts = [f"OpenAlex scholarly metadata title: {title}."]
    if journal:
        text_parts.append(f"Published in {journal}.")
    if publisher:
        text_parts.append(f"Publisher: {publisher}.")
    if doi:
        text_parts.append(f"DOI: {doi}.")
    if include_abstracts:
        abstract = _abstract_from_inverted_index(item.get("abstract_inverted_index"))
        if abstract:
            text_parts.append(abstract)
    task_id = _clean(task.get("task_id"))
    source_fingerprints = _string_sequence(task.get("source_queue_request_sha256", ()))
    open_access = _mapping(item.get("open_access"))
    metadata = {
        "collection_task_ids": (task_id,) if task_id else (),
        "collection_task_source_family": _clean(task.get("source_family")),
        "source_queue_request_sha256": source_fingerprints,
        "matched_query": query,
        "openalex_id": openalex_id,
        "doi": doi,
        "openalex_relevance_score": _float_or_none(item.get("relevance_score")),
        "openalex_type": _clean(item.get("type")),
        "publication_year": _int_or_none(item.get("publication_year")),
        "cited_by_count": _int_or_none(item.get("cited_by_count")),
        "source_display_name": journal,
        "source_type": _clean(source_meta.get("type")),
        "publisher": publisher,
        "is_open_access": _bool_or_none(open_access.get("is_oa")),
        "open_access_status": _clean(open_access.get("oa_status")),
        "concepts": _names_from_openalex_entities(item.get("concepts")),
        "keywords": _names_from_openalex_entities(item.get("keywords")),
        "retrieved_at": fetched_at,
        "provider": PROVIDER,
    }
    return {
        "text": " ".join(part for part in text_parts if part),
        "title": title,
        "source": f"openalex:{openalex_id}" if openalex_id else f"openalex:{doi or task_id}:{_stable_suffix(title)}",
        "url": url,
        "provider": PROVIDER,
        "source_family": source_family,
        "published_at": published_at,
        "timestamp": fetched_at,
        "metadata": _drop_empty(metadata),
    }


def _merge_document_metadata(existing: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    existing_metadata = dict(_mapping(existing.get("metadata")))
    incoming_metadata = _mapping(incoming.get("metadata"))
    for key in ("collection_task_ids", "source_queue_request_sha256"):
        existing_metadata[key] = _dedupe((
            *_string_sequence(existing_metadata.get(key, ())),
            *_string_sequence(incoming_metadata.get(key, ())),
        ))
    existing_queries = _string_sequence(existing_metadata.get("matched_queries", ()))
    first_query = _clean(existing_metadata.get("matched_query"))
    matched_query = _clean(incoming_metadata.get("matched_query"))
    existing_metadata["matched_queries"] = _dedupe((*existing_queries, first_query, matched_query))
    existing["metadata"] = _drop_empty(existing_metadata)


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[Mapping[str, Any]],
    query_count: int,
    errors: Sequence[Mapping[str, Any]],
    skipped_duplicate_count: int,
) -> dict[str, Any]:
    task_families = Counter(str(task.get("source_family")) for task in tasks)
    providers = Counter(str(row.get("provider")) for row in rows if row.get("provider"))
    families = Counter(str(row.get("source_family")) for row in rows if row.get("source_family"))
    source_types = Counter(
        str(_mapping(row.get("metadata")).get("openalex_type"))
        for row in rows
        if _mapping(row.get("metadata")).get("openalex_type")
    )
    return {
        "task_count": len(tasks),
        "query_count": int(query_count),
        "source_document_count": len(rows),
        "error_count": len(errors),
        "skipped_duplicate_count": int(skipped_duplicate_count),
        "task_source_family_counts": _sorted_counter(task_families),
        "provider_counts": _sorted_counter(providers),
        "source_family_counts": _sorted_counter(families),
        "openalex_type_counts": _sorted_counter(source_types),
    }


def _load_tasks(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no} must contain a JSON object.")
            _reject_reserved_fields(payload, source=f"{path}:{line_no}")
            _reject_reserved_fields(_mapping(payload.get("metadata")), source=f"{path}:{line_no}:metadata")
            if _clean(payload.get("usage")) != "source_catalog_collection_only":
                raise ValueError(f"{path}:{line_no} is not a source catalog collection task.")
            if not bool(payload.get("not_verifier_evidence")):
                raise ValueError(f"{path}:{line_no} must be marked not_verifier_evidence.")
            rows.append(dict(payload))
    if not rows:
        raise ValueError("collection task file is empty.")
    return tuple(rows)


def _default_fetch_json(*, timeout_seconds: float) -> Callable[[str, Mapping[str, str]], Mapping[str, Any]]:
    def fetch(url: str, headers: Mapping[str, str]) -> Mapping[str, Any]:
        request = urllib.request.Request(url, headers=dict(headers))
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAlex HTTP {exc.code}: {body[:300]}") from exc

    return fetch


def _openalex_items(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    items = payload.get("results", ())
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
        return tuple(item for item in items if isinstance(item, Mapping))
    return ()


def _abstract_from_inverted_index(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    tokens: dict[int, str] = {}
    for word, positions in value.items():
        if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes, bytearray)):
            continue
        for position in positions:
            try:
                tokens[int(position)] = str(word)
            except (TypeError, ValueError):
                continue
    return " ".join(token for _position, token in sorted(tokens.items()))


def _names_from_openalex_entities(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    names: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = _clean(item.get("display_name") or item.get("name"))
        if name:
            names.append(name)
    return _dedupe(names)


def _document_key(row: Mapping[str, Any]) -> str:
    source = _clean(row.get("source"))
    if source:
        return source.casefold()
    return _clean(row.get("url") or row.get("title")).casefold()


def _stable_suffix(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _reject_reserved_fields(payload: Mapping[str, Any], *, source: str) -> None:
    reserved = sorted(set(str(key) for key in payload) & RESERVED_FIELDS)
    if reserved:
        raise ValueError(f"{source} contains reserved fields: {', '.join(reserved)}")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _drop_empty(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if value not in (None, "", (), [], {})
    }


def _dedupe(values: Sequence[str] | Any) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        item = _clean(value)
        if not item:
            continue
        folded = item.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        output.append(item)
    return tuple(output)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(((key, value) for key, value in counter.items() if key), key=lambda item: (-item[1], item[0])))


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
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--source-family", default=SOURCE_FAMILY)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-query-variants", type=int, default=2)
    parser.add_argument("--rows-per-query", type=int, default=2)
    parser.add_argument("--min-delay-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--mailto", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--include-abstracts", action="store_true")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-empty", action="store_true")
    args = parser.parse_args(argv)
    payload = run_openalex_source_family_catalog_adapter(
        tasks_path=args.tasks,
        output_path=args.output,
        report_json_path=args.report_json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        source_family=args.source_family,
        max_tasks=args.max_tasks,
        max_query_variants=args.max_query_variants,
        rows_per_query=args.rows_per_query,
        min_delay_seconds=args.min_delay_seconds,
        timeout_seconds=args.timeout_seconds,
        mailto=args.mailto,
        api_key=args.api_key,
        user_agent=args.user_agent,
        include_abstracts=bool(args.include_abstracts),
        compact_json=bool(args.compact_json),
        fail_on_empty=bool(args.fail_on_empty),
        metadata=_parse_metadata(args.metadata or ()),
    )
    summary = payload["summary"]
    print(
        "openalex_source_family_catalog_adapter_ok "
        f"status={payload['status']} "
        f"tasks={summary['task_count']} "
        f"queries={summary['query_count']} "
        f"docs={summary['source_document_count']} "
        f"errors={summary['error_count']}"
    )


if __name__ == "__main__":
    main()
