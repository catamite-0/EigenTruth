"""Fetch URL-seeded pages for source-family catalog tasks.

This adapter consumes the non-evidence collection-task JSONL produced by
``plan_source_family_catalog_collection.py`` plus a label-free URL seed JSONL.
It writes adapter-ready source-family catalog documents for
``run_source_family_citation_search_workflow.py``. It uses only the Python
standard library. The emitted catalog rows deliberately omit labels, target ids,
record ids, model answers, request ids, and source row ids; request coverage
remains in the report, while source documents only carry safe collection-task
provenance and seeded page metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from html.parser import HTMLParser
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

WORKFLOW = "seeded_url_source_family_catalog_adapter"
PROVIDER = "seeded_url"
SOURCE_FAMILY = "news"
DEFAULT_USER_AGENT = "EigenTruth/0.2 (https://github.com/catamitez0-maker/EigenTruth)"
RESERVED_TASK_FIELDS = {
    "answer",
    "claim_id",
    "is_false",
    "label",
    "labels",
    "model_answer",
    "record_index",
    "row_id",
    "row_index",
    "score_label",
    "source_index",
    "target_id",
}
RESERVED_SEED_FIELDS = {
    *RESERVED_TASK_FIELDS,
    "request_id",
    "request_ids",
}


def run_seeded_url_source_family_catalog_adapter(
    *,
    tasks_path: str | Path,
    seeds_path: str | Path,
    output_path: str | Path,
    report_json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    source_family: str = SOURCE_FAMILY,
    provider: str = PROVIDER,
    max_tasks: int | None = None,
    max_seed_urls_per_task: int | None = None,
    min_delay_seconds: float = 0.0,
    timeout_seconds: float = 20.0,
    user_agent: str = DEFAULT_USER_AGENT,
    max_text_chars: int = 6000,
    fetch_pages: bool = True,
    compact_json: bool = False,
    fail_on_empty: bool = False,
    metadata: Mapping[str, Any] | None = None,
    fetch_text: Callable[[str, Mapping[str, str]], str] | None = None,
) -> dict[str, Any]:
    """Fetch seeded pages and write source-family catalog rows."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be positive when provided.")
    if max_seed_urls_per_task is not None and max_seed_urls_per_task <= 0:
        raise ValueError("max_seed_urls_per_task must be positive when provided.")
    if min_delay_seconds < 0:
        raise ValueError("min_delay_seconds must be non-negative.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    if max_text_chars <= 0:
        raise ValueError("max_text_chars must be positive.")
    source_family = _clean(source_family)
    provider = _clean(provider) or PROVIDER
    if not source_family:
        raise ValueError("source_family cannot be empty.")

    tasks = tuple(
        task
        for task in _load_tasks(tasks_path)
        if _clean(task.get("source_family")) == source_family
    )
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    if not tasks:
        raise ValueError(f"no collection tasks found for source_family={source_family!r}.")

    tasks_by_id = {_clean(task.get("task_id")): task for task in tasks if _clean(task.get("task_id"))}
    seed_rows = _load_seed_rows(seeds_path)
    matched_seeds = tuple(seed for seed in seed_rows if _seed_task_id(seed) in tasks_by_id)
    if not matched_seeds:
        raise ValueError("no URL seeds matched the selected collection tasks.")

    fetch = fetch_text or _default_fetch_text(timeout_seconds=timeout_seconds)
    fetched_at = datetime.now(UTC).isoformat()
    headers = {"User-Agent": user_agent}
    rows_by_key: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    skipped_duplicate_count = 0
    fetched_page_count = 0
    per_task_counts: Counter[str] = Counter()
    for seed in matched_seeds:
        task_id = _seed_task_id(seed)
        task = tasks_by_id[task_id]
        if max_seed_urls_per_task is not None and per_task_counts[task_id] >= max_seed_urls_per_task:
            continue
        per_task_counts[task_id] += 1
        url = _clean(seed.get("url") or seed.get("href"))
        if not url:
            errors.append({"task_id": task_id, "error": "missing_seed_url"})
            continue
        html_text = ""
        if fetch_pages:
            if min_delay_seconds and fetched_page_count:
                time.sleep(float(min_delay_seconds))
            try:
                html_text = fetch(url, headers)
                fetched_page_count += 1
            except Exception as exc:  # noqa: BLE001 - adapter report records fetch failure details.
                errors.append({"task_id": task_id, "url": url, "error": type(exc).__name__, "message": str(exc)})
        row = _catalog_row_from_seed_and_page(
            seed,
            task=task,
            html_text=html_text,
            fetched_at=fetched_at,
            max_text_chars=max_text_chars,
            source_family=source_family,
            default_provider=provider,
        )
        if row is None:
            continue
        key = _document_key(row)
        existing = rows_by_key.get(key)
        if existing is not None:
            skipped_duplicate_count += 1
            _merge_document_metadata(existing, row)
            continue
        rows_by_key[key] = row
    rows = tuple(rows_by_key.values())
    _write_jsonl(output_path, rows, compact=compact_json)
    summary = _summary(
        rows,
        tasks=tasks,
        seed_rows=seed_rows,
        matched_seeds=matched_seeds,
        fetched_page_count=fetched_page_count,
        errors=errors,
        skipped_duplicate_count=skipped_duplicate_count,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready" if rows else "empty",
        "source": {
            "collection_tasks": str(tasks_path),
            "url_seeds": str(seeds_path),
        },
        "output_path": str(output_path),
        "config": {
            "source_family": source_family,
            "provider": provider,
            "max_tasks": max_tasks,
            "max_seed_urls_per_task": max_seed_urls_per_task,
            "min_delay_seconds": float(min_delay_seconds),
            "timeout_seconds": float(timeout_seconds),
            "max_text_chars": int(max_text_chars),
            "fetch_pages": bool(fetch_pages),
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
            "url_seeds": Path(seeds_path),
            "source_family_catalog": Path(output_path),
        }
        if report_json_path is not None:
            artifacts["seeded_url_catalog_report"] = Path(report_json_path)
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "task_count": summary["task_count"],
                "matched_seed_count": summary["matched_seed_count"],
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
                "matched_seed_count": summary["matched_seed_count"],
                "source_document_count": summary["source_document_count"],
                "error_count": summary["error_count"],
                "artifact_manifest": payload.get("artifact_manifest"),
                **dict(metadata or {}),
            },
        ).save_json()
    if fail_on_empty and not rows:
        raise SystemExit(1)
    return payload


def _catalog_row_from_seed_and_page(
    seed: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    html_text: str,
    fetched_at: str,
    max_text_chars: int,
    source_family: str,
    default_provider: str,
) -> dict[str, Any] | None:
    url = _clean(seed.get("url") or seed.get("href"))
    if not url:
        return None
    extracted = _extract_page_text(html_text) if html_text else {}
    title = _clean(seed.get("title") or extracted.get("title")) or url
    seed_text = _clean(seed.get("text") or seed.get("summary") or seed.get("snippet"))
    extracted_text = _clean(extracted.get("text"))
    text = _truncate_text(extracted_text or seed_text or title, max_text_chars)
    if not text:
        return None
    domain = _clean(seed.get("domain")) or _url_domain(url)
    provider = _clean(seed.get("provider") or seed.get("source_provider")) or domain or default_provider
    task_id = _clean(task.get("task_id"))
    matched_query = _clean(seed.get("matched_query") or task.get("query"))
    metadata = {
        "collection_task_ids": (task_id,) if task_id else (),
        "collection_task_source_family": _clean(task.get("source_family")),
        "source_queue_request_sha256": _string_sequence(task.get("source_queue_request_sha256", ())),
        "query_key": _clean(task.get("query_key")),
        "matched_query": matched_query,
        "seed_key": _clean(seed.get("seed_key")),
        "domain": domain,
        "provider": provider,
        "fetch_status": "fetched" if html_text else "seed_fallback",
        "retrieved_at": fetched_at,
        "description": _clean(seed.get("description") or extracted.get("description")),
        "source_family_seed": source_family,
    }
    return {
        "text": text,
        "title": title,
        "source": _clean(seed.get("source")) or f"seeded_url:{source_family}:{_stable_suffix(url)}",
        "url": url,
        "provider": provider,
        "source_family": source_family,
        "published_at": _clean(seed.get("published_at") or seed.get("publication_date")),
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
    seed_rows: Sequence[Mapping[str, Any]],
    matched_seeds: Sequence[Mapping[str, Any]],
    fetched_page_count: int,
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
    fetch_statuses = Counter(
        str(_mapping(row.get("metadata")).get("fetch_status"))
        for row in rows
        if _mapping(row.get("metadata")).get("fetch_status")
    )
    return {
        "task_count": len(tasks),
        "seed_count": len(seed_rows),
        "matched_seed_count": len(matched_seeds),
        "fetched_page_count": int(fetched_page_count),
        "source_document_count": len(rows),
        "error_count": len(errors),
        "skipped_duplicate_count": int(skipped_duplicate_count),
        "task_source_family_counts": _sorted_counter(task_families),
        "provider_counts": _sorted_counter(providers),
        "source_family_counts": _sorted_counter(families),
        "domain_counts": _sorted_counter(domains),
        "fetch_status_counts": _sorted_counter(fetch_statuses),
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
            _reject_reserved_fields(payload, source=f"{path}:{line_no}", reserved=RESERVED_TASK_FIELDS)
            _reject_reserved_fields(
                _mapping(payload.get("metadata")),
                source=f"{path}:{line_no}:metadata",
                reserved=RESERVED_TASK_FIELDS,
            )
            if _clean(payload.get("usage")) != "source_catalog_collection_only":
                raise ValueError(f"{path}:{line_no} is not a source catalog collection task.")
            if not bool(payload.get("not_verifier_evidence")):
                raise ValueError(f"{path}:{line_no} must be marked not_verifier_evidence.")
            rows.append(dict(payload))
    if not rows:
        raise ValueError("collection task file is empty.")
    return tuple(rows)


def _load_seed_rows(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no} must contain a JSON object.")
            _reject_reserved_fields(payload, source=f"{path}:{line_no}", reserved=RESERVED_SEED_FIELDS)
            _reject_reserved_fields(
                _mapping(payload.get("metadata")),
                source=f"{path}:{line_no}:metadata",
                reserved=RESERVED_SEED_FIELDS,
            )
            if not _seed_task_id(payload):
                raise ValueError(f"{path}:{line_no} must include task_id or collection_task_id.")
            if not _clean(payload.get("url") or payload.get("href")):
                raise ValueError(f"{path}:{line_no} must include url.")
            rows.append(dict(payload))
    if not rows:
        raise ValueError("URL seed file is empty.")
    return tuple(rows)


def _seed_task_id(seed: Mapping[str, Any]) -> str:
    return _clean(seed.get("task_id") or seed.get("collection_task_id"))


def _default_fetch_text(*, timeout_seconds: float) -> Callable[[str, Mapping[str, str]], str]:
    def fetch(url: str, headers: Mapping[str, str]) -> str:
        request = urllib.request.Request(url, headers=dict(headers))
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"seeded URL HTTP {exc.code}: {body[:300]}") from exc

    return fetch


class _PageTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.description = ""
        self.text_parts: list[str] = []
        self._tag_stack: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self._tag_stack.append(tag)
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag == "meta" and not self.description:
            attr_map = {name.lower(): value or "" for name, value in attrs}
            is_description = (
                attr_map.get("name", "").lower() == "description"
                or attr_map.get("property", "").lower() == "og:description"
            )
            if is_description:
                self.description = _clean(html.unescape(attr_map.get("content", "")))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = _clean(html.unescape(data))
        if not text:
            return
        tag = self._tag_stack[-1] if self._tag_stack else ""
        if tag == "title":
            self.title_parts.append(text)
        elif tag in {"h1", "h2", "h3", "p", "li", "td", "th"}:
            self.text_parts.append(text)


def _extract_page_text(html_text: str) -> dict[str, str]:
    parser = _PageTextExtractor()
    parser.feed(html_text)
    title = _clean(" ".join(parser.title_parts))
    description = _clean(parser.description)
    text = _clean(" ".join((*parser.title_parts, description, *parser.text_parts)))
    return {"title": title, "description": description, "text": text}


def _document_key(row: Mapping[str, Any]) -> str:
    url = _clean(row.get("url"))
    if url:
        return url.casefold()
    return _clean(row.get("source") or row.get("title")).casefold()


def _stable_suffix(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _url_domain(value: str) -> str:
    try:
        return urllib.parse.urlparse(value).netloc.lower()
    except ValueError:
        return ""


def _truncate_text(value: str, max_chars: int) -> str:
    text = _clean(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def _reject_reserved_fields(payload: Mapping[str, Any], *, source: str, reserved: set[str]) -> None:
    reserved_names = sorted(set(str(key) for key in payload) & reserved)
    if reserved_names:
        raise ValueError(f"{source} contains reserved fields: {', '.join(reserved_names)}")


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
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--source-family", default=SOURCE_FAMILY)
    parser.add_argument("--provider", default=PROVIDER)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-seed-urls-per-task", type=int, default=None)
    parser.add_argument("--min-delay-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--max-text-chars", type=int, default=6000)
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-empty", action="store_true")
    args = parser.parse_args(argv)
    payload = run_seeded_url_source_family_catalog_adapter(
        tasks_path=args.tasks,
        seeds_path=args.seeds,
        output_path=args.output,
        report_json_path=args.report_json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        source_family=args.source_family,
        provider=args.provider,
        max_tasks=args.max_tasks,
        max_seed_urls_per_task=args.max_seed_urls_per_task,
        min_delay_seconds=args.min_delay_seconds,
        timeout_seconds=args.timeout_seconds,
        user_agent=args.user_agent,
        max_text_chars=args.max_text_chars,
        fetch_pages=not bool(args.no_fetch),
        compact_json=bool(args.compact_json),
        fail_on_empty=bool(args.fail_on_empty),
        metadata=_parse_metadata(args.metadata or ()),
    )
    summary = payload["summary"]
    print(
        "seeded_url_source_family_catalog_adapter_ok "
        f"status={payload['status']} "
        f"tasks={summary['task_count']} "
        f"matched_seeds={summary['matched_seed_count']} "
        f"docs={summary['source_document_count']} "
        f"errors={summary['error_count']}"
    )


if __name__ == "__main__":
    main()
