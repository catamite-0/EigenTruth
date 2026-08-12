"""Fetch Crossref works for scholarly source-family catalog tasks.

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

WORKFLOW = "crossref_source_family_catalog_adapter"
PROVIDER = "crossref"
SOURCE_FAMILY = "scholarly"
API_BASE_URL = "https://api.crossref.org/works"
DEFAULT_USER_AGENT = "EigenTruth/0.2 (https://github.com/catamite-0/EigenTruth)"
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


def run_crossref_source_family_catalog_adapter(
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
    user_agent: str = DEFAULT_USER_AGENT,
    include_abstracts: bool = False,
    compact_json: bool = False,
    fail_on_empty: bool = False,
    metadata: Mapping[str, Any] | None = None,
    fetch_json: Callable[[str, Mapping[str, str]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fetch Crossref metadata and write source-family catalog rows."""
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
                payload = fetch(_crossref_url(
                    query=query,
                    rows=rows_per_query,
                    mailto=mailto,
                ), {"User-Agent": user_agent})
            except Exception as exc:  # noqa: BLE001 - adapter report records fetch failure details.
                errors.append({"task_id": task_id, "query": query, "error": type(exc).__name__, "message": str(exc)})
                continue
            items = _crossref_items(payload)
            for item in items:
                row = _catalog_row_from_crossref_item(
                    item,
                    task=task,
                    query=query,
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
            artifacts["crossref_catalog_report"] = Path(report_json_path)
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


def _crossref_url(*, query: str, rows: int, mailto: str | None) -> str:
    params = {
        "query.bibliographic": query,
        "rows": str(rows),
        "select": ",".join((
            "DOI",
            "URL",
            "title",
            "container-title",
            "publisher",
            "type",
            "issued",
            "published-print",
            "published-online",
            "is-referenced-by-count",
            "subject",
            "score",
        )),
    }
    if mailto:
        params["mailto"] = mailto
    return f"{API_BASE_URL}?{urllib.parse.urlencode(params)}"


def _catalog_row_from_crossref_item(
    item: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    query: str,
    fetched_at: str,
    include_abstracts: bool,
) -> dict[str, Any] | None:
    doi = _clean(item.get("DOI"))
    title = _first(item.get("title"))
    if not title:
        return None
    container_title = _first(item.get("container-title"))
    publisher = _clean(item.get("publisher"))
    url = _clean(item.get("URL")) or (f"https://doi.org/{doi}" if doi else "")
    published_at = _published_date(item)
    text_parts = [f"Crossref scholarly metadata title: {title}."]
    if container_title:
        text_parts.append(f"Published in {container_title}.")
    if publisher:
        text_parts.append(f"Publisher: {publisher}.")
    if doi:
        text_parts.append(f"DOI: {doi}.")
    if include_abstracts:
        abstract = _clean_htmlish(item.get("abstract"))
        if abstract:
            text_parts.append(abstract)
    task_id = _clean(task.get("task_id"))
    source_fingerprints = _string_sequence(task.get("source_queue_request_sha256", ()))
    metadata = {
        "collection_task_ids": (task_id,) if task_id else (),
        "collection_task_source_family": _clean(task.get("source_family")),
        "source_queue_request_sha256": source_fingerprints,
        "matched_query": query,
        "crossref_score": _float_or_none(item.get("score")),
        "crossref_type": _clean(item.get("type")),
        "container_title": container_title,
        "publisher": publisher,
        "is_referenced_by_count": _int_or_none(item.get("is-referenced-by-count")),
        "subject": _string_sequence(item.get("subject", ())),
        "retrieved_at": fetched_at,
        "provider": PROVIDER,
    }
    return {
        "text": " ".join(part for part in text_parts if part),
        "title": title,
        "source": f"crossref:{doi}" if doi else f"crossref:{task_id}:{_stable_suffix(title)}",
        "url": url,
        "provider": PROVIDER,
        "source_family": SOURCE_FAMILY,
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
    matched_query = _clean(incoming_metadata.get("matched_query"))
    existing_metadata["matched_queries"] = _dedupe((*existing_queries, matched_query))
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
        str(_mapping(row.get("metadata")).get("crossref_type"))
        for row in rows
        if _mapping(row.get("metadata")).get("crossref_type")
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
        "crossref_type_counts": _sorted_counter(source_types),
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
            raise RuntimeError(f"Crossref HTTP {exc.code}: {body[:300]}") from exc

    return fetch


def _crossref_items(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    message = _mapping(payload.get("message"))
    items = message.get("items", ())
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
        return tuple(item for item in items if isinstance(item, Mapping))
    return ()


def _published_date(item: Mapping[str, Any]) -> str:
    for key in ("published-online", "published-print", "issued"):
        date = _crossref_date(item.get(key))
        if date:
            return date
    return ""


def _crossref_date(value: Any) -> str:
    parts = _mapping(value).get("date-parts")
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes, bytearray)) or not parts:
        return ""
    first = parts[0]
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes, bytearray)) or not first:
        return ""
    numbers = [int(part) for part in first[:3] if isinstance(part, (int, float)) or str(part).isdigit()]
    if not numbers:
        return ""
    year = numbers[0]
    month = numbers[1] if len(numbers) > 1 else 1
    day = numbers[2] if len(numbers) > 2 else 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def _document_key(row: Mapping[str, Any]) -> str:
    source = _clean(row.get("source"))
    if source:
        return source.casefold()
    return _clean(row.get("url") or row.get("title")).casefold()


def _stable_suffix(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _first(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _clean(value[0] if value else "")
    return _clean(value)


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


def _clean_htmlish(value: Any) -> str:
    text = _clean(value)
    return text.replace("<jats:p>", "").replace("</jats:p>", "").replace("<p>", "").replace("</p>", "")


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
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--include-abstracts", action="store_true")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-empty", action="store_true")
    args = parser.parse_args(argv)
    payload = run_crossref_source_family_catalog_adapter(
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
        user_agent=args.user_agent,
        include_abstracts=bool(args.include_abstracts),
        compact_json=bool(args.compact_json),
        fail_on_empty=bool(args.fail_on_empty),
        metadata=_parse_metadata(args.metadata or ()),
    )
    summary = payload["summary"]
    print(
        "crossref_source_family_catalog_adapter_ok "
        f"status={payload['status']} "
        f"tasks={summary['task_count']} "
        f"queries={summary['query_count']} "
        f"docs={summary['source_document_count']} "
        f"errors={summary['error_count']}"
    )


if __name__ == "__main__":
    main()
