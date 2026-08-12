"""Fetch World Bank indicators for official-statistics source-family tasks.

This adapter consumes the non-evidence collection-task JSONL produced by
``plan_source_family_catalog_collection.py`` and writes adapter-ready
``source_family=official_statistics`` catalog documents for
``run_source_family_citation_search_workflow.py``. It uses only the Python
standard library. The emitted catalog rows deliberately omit labels, target ids,
record ids, model answers, and request ids; request coverage remains in the
report, while source documents only carry safe collection-task provenance.
"""

from __future__ import annotations

import argparse
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

WORKFLOW = "worldbank_source_family_catalog_adapter"
PROVIDER = "worldbank"
SOURCE_FAMILY = "official_statistics"
API_BASE_URL = "https://api.worldbank.org/v2"
DEFAULT_INDICATOR = "SP.POP.TOTL"
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


def run_worldbank_source_family_catalog_adapter(
    *,
    tasks_path: str | Path,
    output_path: str | Path,
    report_json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    source_family: str = SOURCE_FAMILY,
    indicator: str = DEFAULT_INDICATOR,
    max_tasks: int | None = None,
    per_page: int = 300,
    mrnev: int = 1,
    max_pages: int | None = None,
    max_countries: int | None = None,
    include_aggregates: bool = False,
    min_delay_seconds: float = 0.0,
    timeout_seconds: float = 20.0,
    user_agent: str = DEFAULT_USER_AGENT,
    compact_json: bool = False,
    fail_on_empty: bool = False,
    metadata: Mapping[str, Any] | None = None,
    fetch_json: Callable[[str, Mapping[str, str]], Any] | None = None,
) -> dict[str, Any]:
    """Fetch World Bank indicator data and write source-family catalog rows."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be positive when provided.")
    if per_page <= 0:
        raise ValueError("per_page must be positive.")
    if mrnev < 0:
        raise ValueError("mrnev must be non-negative.")
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive when provided.")
    if max_countries is not None and max_countries <= 0:
        raise ValueError("max_countries must be positive when provided.")
    if min_delay_seconds < 0:
        raise ValueError("min_delay_seconds must be non-negative.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    indicator = _clean(indicator)
    if not indicator:
        raise ValueError("indicator cannot be empty.")

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
    headers = {"User-Agent": user_agent}
    fetched_at = datetime.now(timezone.utc).isoformat()
    errors: list[dict[str, Any]] = []
    country_meta_by_iso3: dict[str, Mapping[str, Any]] = {}
    country_page_count = 0
    if not include_aggregates:
        try:
            country_meta_by_iso3, country_page_count = _fetch_country_metadata(
                fetch,
                headers=headers,
                per_page=per_page,
                max_pages=max_pages,
                min_delay_seconds=min_delay_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - adapter report records fetch failure details.
            errors.append({"stage": "country_metadata", "error": type(exc).__name__, "message": str(exc)})

    page_count = 0
    skipped_aggregate_count = 0
    skipped_empty_value_count = 0
    skipped_unknown_country_count = 0
    docs: list[dict[str, Any]] = []
    for page_payload in _fetch_indicator_pages(
        fetch,
        headers=headers,
        indicator=indicator,
        per_page=per_page,
        mrnev=mrnev,
        max_pages=max_pages,
        min_delay_seconds=min_delay_seconds,
        errors=errors,
    ):
        page_count += 1
        page_meta, items = _worldbank_page(page_payload)
        last_updated = _clean(page_meta.get("lastupdated"))
        for item in items:
            if item.get("value") is None:
                skipped_empty_value_count += 1
                continue
            iso3 = _clean(item.get("countryiso3code"))
            country_meta = country_meta_by_iso3.get(iso3)
            if not include_aggregates:
                if not country_meta:
                    skipped_unknown_country_count += 1
                    continue
                if _is_aggregate_country(country_meta):
                    skipped_aggregate_count += 1
                    continue
            row = _catalog_row_from_worldbank_item(
                item,
                tasks=tasks,
                indicator=indicator,
                country_meta=country_meta,
                fetched_at=fetched_at,
                last_updated=last_updated,
            )
            if row is not None:
                docs.append(row)
                if max_countries is not None and len(docs) >= max_countries:
                    break
        if max_countries is not None and len(docs) >= max_countries:
            break
    rows = tuple(docs)
    _write_jsonl(output_path, rows, compact=compact_json)
    summary = _summary(
        rows,
        tasks=tasks,
        page_count=page_count,
        country_page_count=country_page_count,
        errors=errors,
        skipped_aggregate_count=skipped_aggregate_count,
        skipped_empty_value_count=skipped_empty_value_count,
        skipped_unknown_country_count=skipped_unknown_country_count,
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
            "indicator": indicator,
            "max_tasks": max_tasks,
            "per_page": int(per_page),
            "mrnev": int(mrnev),
            "max_pages": max_pages,
            "max_countries": max_countries,
            "include_aggregates": bool(include_aggregates),
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
            artifacts["worldbank_catalog_report"] = Path(report_json_path)
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "task_count": summary["task_count"],
                "page_count": summary["page_count"],
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
                "page_count": summary["page_count"],
                "source_document_count": summary["source_document_count"],
                "error_count": summary["error_count"],
                "artifact_manifest": payload.get("artifact_manifest"),
                **dict(metadata or {}),
            },
        ).save_json()
    if fail_on_empty and not rows:
        raise SystemExit(1)
    return payload


def _fetch_country_metadata(
    fetch: Callable[[str, Mapping[str, str]], Any],
    *,
    headers: Mapping[str, str],
    per_page: int,
    max_pages: int | None,
    min_delay_seconds: float,
) -> tuple[dict[str, Mapping[str, Any]], int]:
    rows: dict[str, Mapping[str, Any]] = {}
    page = 1
    page_count = 0
    total_pages = 1
    while page <= total_pages:
        payload = fetch(_worldbank_country_url(page=page, per_page=per_page), headers)
        page_meta, items = _worldbank_page(payload)
        page_count += 1
        total_pages = _page_count(page_meta)
        for item in items:
            iso3 = _clean(item.get("id"))
            if iso3:
                rows[iso3] = dict(item)
        if max_pages is not None and page_count >= max_pages:
            break
        page += 1
        if min_delay_seconds:
            time.sleep(float(min_delay_seconds))
    return rows, page_count


def _fetch_indicator_pages(
    fetch: Callable[[str, Mapping[str, str]], Any],
    *,
    headers: Mapping[str, str],
    indicator: str,
    per_page: int,
    mrnev: int,
    max_pages: int | None,
    min_delay_seconds: float,
    errors: list[dict[str, Any]],
) -> tuple[Any, ...]:
    pages: list[Any] = []
    page = 1
    page_count = 0
    total_pages = 1
    while page <= total_pages:
        try:
            payload = fetch(
                _worldbank_indicator_url(indicator=indicator, page=page, per_page=per_page, mrnev=mrnev),
                headers,
            )
        except Exception as exc:  # noqa: BLE001 - adapter report records fetch failure details.
            errors.append({"stage": "indicator_data", "page": page, "error": type(exc).__name__, "message": str(exc)})
            break
        page_meta, _items = _worldbank_page(payload)
        pages.append(payload)
        page_count += 1
        total_pages = _page_count(page_meta)
        if max_pages is not None and page_count >= max_pages:
            break
        page += 1
        if min_delay_seconds:
            time.sleep(float(min_delay_seconds))
    return tuple(pages)


def _worldbank_country_url(*, page: int, per_page: int) -> str:
    params = {
        "format": "json",
        "page": str(page),
        "per_page": str(per_page),
    }
    return f"{API_BASE_URL}/country?{urllib.parse.urlencode(params)}"


def _worldbank_indicator_url(*, indicator: str, page: int, per_page: int, mrnev: int) -> str:
    params = {
        "format": "json",
        "per_page": str(per_page),
    }
    if mrnev > 0:
        params["mrnev"] = str(mrnev)
    else:
        params["page"] = str(page)
    safe_indicator = urllib.parse.quote(indicator, safe="")
    return f"{API_BASE_URL}/country/all/indicator/{safe_indicator}?{urllib.parse.urlencode(params)}"


def _catalog_row_from_worldbank_item(
    item: Mapping[str, Any],
    *,
    tasks: Sequence[Mapping[str, Any]],
    indicator: str,
    country_meta: Mapping[str, Any] | None,
    fetched_at: str,
    last_updated: str,
) -> dict[str, Any] | None:
    country = _mapping(item.get("country"))
    country_name = _clean(country.get("value"))
    if not country_name:
        return None
    iso3 = _clean(item.get("countryiso3code")) or _clean(country_meta.get("id") if country_meta else "")
    iso2 = _clean(_mapping(country_meta).get("iso2Code")) or _clean(country.get("id"))
    year = _clean(item.get("date"))
    value = item.get("value")
    if value is None:
        return None
    indicator_meta = _mapping(item.get("indicator"))
    indicator_name = _clean(indicator_meta.get("value")) or indicator
    value_text = _format_number(value)
    region = _clean(_mapping(_mapping(country_meta).get("region")).get("value")) if country_meta else ""
    income_level = _clean(_mapping(_mapping(country_meta).get("incomeLevel")).get("value")) if country_meta else ""
    task_ids = tuple(_clean(task.get("task_id")) for task in tasks if _clean(task.get("task_id")))
    source_fingerprints = _dedupe(
        fingerprint
        for task in tasks
        for fingerprint in _string_sequence(task.get("source_queue_request_sha256", ()))
    )
    matched_queries = _dedupe(query for task in tasks for query in _string_sequence(task.get("search_queries", ())))
    text_parts = [
        (
            "World Bank official statistics data for population country queries: "
            f"{country_name} had {indicator_name} of {value_text} in {year}."
        ),
        f"Indicator code {indicator}.",
        "Source: World Development Indicators.",
    ]
    if region:
        text_parts.append(f"Region: {region}.")
    metadata = {
        "collection_task_ids": task_ids,
        "collection_task_source_family": SOURCE_FAMILY,
        "source_queue_request_sha256": source_fingerprints,
        "matched_queries": matched_queries,
        "indicator": indicator,
        "indicator_name": indicator_name,
        "country_name": country_name,
        "country_code_iso3": iso3,
        "country_code_iso2": iso2,
        "region": region,
        "income_level": income_level,
        "reference_year": year,
        "value": _numeric_value(value),
        "unit": _clean(item.get("unit")),
        "decimal": _int_or_none(item.get("decimal")),
        "last_updated": last_updated,
        "retrieved_at": fetched_at,
        "provider": PROVIDER,
    }
    url = f"https://data.worldbank.org/indicator/{urllib.parse.quote(indicator, safe='')}"
    if iso2:
        url = f"{url}?locations={urllib.parse.quote(iso2, safe='')}"
    return {
        "text": " ".join(part for part in text_parts if part),
        "title": f"World Bank official statistics: {indicator_name} for {country_name} ({year})",
        "source": f"worldbank:{indicator}:{iso3 or _stable_country_key(country_name)}:{year}",
        "url": url,
        "provider": PROVIDER,
        "source_family": SOURCE_FAMILY,
        "published_at": last_updated or (f"{year}-01-01" if year.isdigit() else ""),
        "timestamp": fetched_at,
        "metadata": _drop_empty(metadata),
    }


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[Mapping[str, Any]],
    page_count: int,
    country_page_count: int,
    errors: Sequence[Mapping[str, Any]],
    skipped_aggregate_count: int,
    skipped_empty_value_count: int,
    skipped_unknown_country_count: int,
) -> dict[str, Any]:
    task_families = Counter(str(task.get("source_family")) for task in tasks)
    providers = Counter(str(row.get("provider")) for row in rows if row.get("provider"))
    families = Counter(str(row.get("source_family")) for row in rows if row.get("source_family"))
    regions = Counter(
        str(_mapping(row.get("metadata")).get("region"))
        for row in rows
        if _mapping(row.get("metadata")).get("region")
    )
    return {
        "task_count": len(tasks),
        "page_count": int(page_count),
        "country_metadata_page_count": int(country_page_count),
        "source_document_count": len(rows),
        "error_count": len(errors),
        "skipped_aggregate_count": int(skipped_aggregate_count),
        "skipped_empty_value_count": int(skipped_empty_value_count),
        "skipped_unknown_country_count": int(skipped_unknown_country_count),
        "task_source_family_counts": _sorted_counter(task_families),
        "provider_counts": _sorted_counter(providers),
        "source_family_counts": _sorted_counter(families),
        "region_counts": _sorted_counter(regions),
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


def _default_fetch_json(*, timeout_seconds: float) -> Callable[[str, Mapping[str, str]], Any]:
    def fetch(url: str, headers: Mapping[str, str]) -> Any:
        request = urllib.request.Request(url, headers=dict(headers))
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"World Bank HTTP {exc.code}: {body[:300]}") from exc

    return fetch


def _worldbank_page(payload: Any) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)) or len(payload) < 2:
        raise ValueError("World Bank response must be a JSON array with metadata and rows.")
    meta = payload[0] if isinstance(payload[0], Mapping) else {}
    rows = payload[1]
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        return meta, tuple(item for item in rows if isinstance(item, Mapping))
    return meta, ()


def _page_count(meta: Mapping[str, Any]) -> int:
    try:
        return max(1, int(meta.get("pages", 1)))
    except (TypeError, ValueError):
        return 1


def _is_aggregate_country(country_meta: Mapping[str, Any]) -> bool:
    region = _mapping(country_meta.get("region"))
    income = _mapping(country_meta.get("incomeLevel"))
    return (
        _clean(region.get("id")).casefold() == "na"
        or _clean(region.get("value")).casefold() == "aggregates"
        or _clean(income.get("value")).casefold() == "aggregates"
    )


def _stable_country_key(value: str) -> str:
    return "_".join(_clean(value).casefold().split())[:60] or "country"


def _format_number(value: Any) -> str:
    number = _numeric_value(value)
    if isinstance(number, int):
        return f"{number:,}"
    if isinstance(number, float):
        return f"{number:,.3f}".rstrip("0").rstrip(".")
    return _clean(value)


def _numeric_value(value: Any) -> int | float | str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return _clean(value)
    if parsed.is_integer():
        return int(parsed)
    return parsed


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
    parser.add_argument("--indicator", default=DEFAULT_INDICATOR)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--per-page", type=int, default=300)
    parser.add_argument("--mrnev", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-countries", type=int, default=None)
    parser.add_argument("--include-aggregates", action="store_true")
    parser.add_argument("--min-delay-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-empty", action="store_true")
    args = parser.parse_args(argv)
    payload = run_worldbank_source_family_catalog_adapter(
        tasks_path=args.tasks,
        output_path=args.output,
        report_json_path=args.report_json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        source_family=args.source_family,
        indicator=args.indicator,
        max_tasks=args.max_tasks,
        per_page=args.per_page,
        mrnev=args.mrnev,
        max_pages=args.max_pages,
        max_countries=args.max_countries,
        include_aggregates=bool(args.include_aggregates),
        min_delay_seconds=args.min_delay_seconds,
        timeout_seconds=args.timeout_seconds,
        user_agent=args.user_agent,
        compact_json=bool(args.compact_json),
        fail_on_empty=bool(args.fail_on_empty),
        metadata=_parse_metadata(args.metadata or ()),
    )
    summary = payload["summary"]
    print(
        "worldbank_source_family_catalog_adapter_ok "
        f"status={payload['status']} "
        f"tasks={summary['task_count']} "
        f"pages={summary['page_count']} "
        f"docs={summary['source_document_count']} "
        f"errors={summary['error_count']}"
    )


if __name__ == "__main__":
    main()
