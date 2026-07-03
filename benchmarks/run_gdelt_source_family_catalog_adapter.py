"""Fetch GDELT DOC news articles for news source-family catalog tasks.

This adapter consumes the non-evidence collection-task JSONL produced by
``plan_source_family_catalog_collection.py`` and writes adapter-ready
``source_family=news`` catalog documents for
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

WORKFLOW = "gdelt_source_family_catalog_adapter"
PROVIDER = "gdelt"
SOURCE_FAMILY = "news"
API_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_USER_AGENT = "EigenTruth/0.2 (https://github.com/catamitez0-maker/EigenTruth)"
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


def run_gdelt_source_family_catalog_adapter(
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
    max_query_variants: int = 3,
    max_records: int = 10,
    sort: str = "HybridRel",
    mode: str = "ArtList",
    min_delay_seconds: float = 6.0,
    timeout_seconds: float = 30.0,
    user_agent: str = DEFAULT_USER_AGENT,
    compact_json: bool = False,
    fail_on_empty: bool = False,
    metadata: Mapping[str, Any] | None = None,
    fetch_json: Callable[[str, Mapping[str, str]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fetch GDELT article metadata and write source-family catalog rows."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be positive when provided.")
    if max_query_variants <= 0:
        raise ValueError("max_query_variants must be positive.")
    if max_records <= 0:
        raise ValueError("max_records must be positive.")
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
        for query_index, query in enumerate(query_variants):
            query_count += 1
            if min_delay_seconds and (query_count > 1 or query_index > 0):
                time.sleep(float(min_delay_seconds))
            try:
                payload = fetch(
                    _gdelt_url(query=query, max_records=max_records, mode=mode, sort=sort),
                    {"User-Agent": user_agent},
                )
            except Exception as exc:  # noqa: BLE001 - adapter report records fetch failure details.
                errors.append({"task_id": task_id, "query": query, "error": type(exc).__name__, "message": str(exc)})
                continue
            for article in _gdelt_articles(payload):
                row = _catalog_row_from_gdelt_article(
                    article,
                    task=task,
                    query=query,
                    fetched_at=fetched_at,
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
            "max_records": int(max_records),
            "sort": sort,
            "mode": mode,
            "min_delay_seconds": float(min_delay_seconds),
            "timeout_seconds": float(timeout_seconds),
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
            artifacts["gdelt_catalog_report"] = Path(report_json_path)
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


def _gdelt_url(*, query: str, max_records: int, mode: str, sort: str) -> str:
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "maxrecords": str(max_records),
        "sort": sort,
    }
    return f"{API_BASE_URL}?{urllib.parse.urlencode(params)}"


def _catalog_row_from_gdelt_article(
    article: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    query: str,
    fetched_at: str,
) -> dict[str, Any] | None:
    url = _clean(article.get("url"))
    title = _clean(article.get("title"))
    if not url or not title:
        return None
    domain = _clean(article.get("domain"))
    seendate = _clean(article.get("seendate"))
    language = _clean(article.get("language"))
    source_country = _clean(article.get("sourcecountry"))
    text_parts = [f"GDELT news article title: {title}."]
    if domain:
        text_parts.append(f"Publisher domain: {domain}.")
    if seendate:
        text_parts.append(f"Seen by GDELT on {seendate}.")
    task_id = _clean(task.get("task_id"))
    source_fingerprints = _string_sequence(task.get("source_queue_request_sha256", ()))
    metadata = {
        "collection_task_ids": (task_id,) if task_id else (),
        "collection_task_source_family": _clean(task.get("source_family")),
        "source_queue_request_sha256": source_fingerprints,
        "matched_query": query,
        "domain": domain,
        "language": language,
        "source_country": source_country,
        "seendate": seendate,
        "socialimage": _clean(article.get("socialimage")),
        "retrieved_at": fetched_at,
        "provider": PROVIDER,
    }
    return {
        "text": " ".join(part for part in text_parts if part),
        "title": title,
        "source": f"gdelt:{_stable_suffix(url)}",
        "url": url,
        "provider": PROVIDER,
        "source_family": SOURCE_FAMILY,
        "published_at": _gdelt_date(seendate),
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
    existing_queries = _dedupe((
        *_string_sequence(existing_metadata.get("matched_queries", ())),
        _clean(existing_metadata.get("matched_query")),
    ))
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
    domains = Counter(
        str(_mapping(row.get("metadata")).get("domain"))
        for row in rows
        if _mapping(row.get("metadata")).get("domain")
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
        "domain_counts": _sorted_counter(domains),
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
            raise RuntimeError(f"GDELT HTTP {exc.code}: {body[:300]}") from exc

    return fetch


def _gdelt_articles(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    articles = payload.get("articles", ())
    if isinstance(articles, Sequence) and not isinstance(articles, (str, bytes, bytearray)):
        return tuple(article for article in articles if isinstance(article, Mapping))
    return ()


def _gdelt_date(value: str) -> str:
    text = _clean(value)
    if len(text) >= 8 and text[:8].isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


def _document_key(row: Mapping[str, Any]) -> str:
    url = _clean(row.get("url"))
    if url:
        return url.casefold()
    return _clean(row.get("source") or row.get("title")).casefold()


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
    parser.add_argument("--max-query-variants", type=int, default=3)
    parser.add_argument("--max-records", type=int, default=10)
    parser.add_argument("--sort", default="HybridRel")
    parser.add_argument("--mode", default="ArtList")
    parser.add_argument("--min-delay-seconds", type=float, default=6.0)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-empty", action="store_true")
    args = parser.parse_args(argv)
    payload = run_gdelt_source_family_catalog_adapter(
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
        max_records=args.max_records,
        sort=args.sort,
        mode=args.mode,
        min_delay_seconds=args.min_delay_seconds,
        timeout_seconds=args.timeout_seconds,
        user_agent=args.user_agent,
        compact_json=bool(args.compact_json),
        fail_on_empty=bool(args.fail_on_empty),
        metadata=_parse_metadata(args.metadata or ()),
    )
    summary = payload["summary"]
    print(
        "gdelt_source_family_catalog_adapter_ok "
        f"status={payload['status']} "
        f"tasks={summary['task_count']} "
        f"queries={summary['query_count']} "
        f"docs={summary['source_document_count']} "
        f"errors={summary['error_count']}"
    )


if __name__ == "__main__":
    main()
