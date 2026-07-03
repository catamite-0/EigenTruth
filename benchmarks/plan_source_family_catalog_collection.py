"""Plan provider-specific catalog collection from source-family gaps.

The input is the non-evidence acquisition JSONL emitted by
``audit_source_family_coverage.py``. This planner deduplicates repeated
source-family gaps into provider-oriented collection tasks, preserving request
ids and safe queue fingerprints while keeping labels, target ids, and model
answers out of the boundary. The output tasks are collection work items only;
they are not verifier evidence or source documents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
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

WORKFLOW = "source_family_catalog_collection_plan"
TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
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
PROVIDER_HINTS = {
    "official_statistics": (
        "official_statistics_api",
        "world_bank_or_un_data",
        "national_statistics_office",
        "data_catalog_search",
    ),
    "official": (
        "official_site_search",
        "government_or_institution_site_search",
        "official_profile_or_faq",
    ),
    "scholarly": (
        "openalex_works",
        "crossref_works",
        "scholarly_review_search",
    ),
    "news": (
        "news_search_adapter",
        "gdelt_or_news_archive",
    ),
    "domain_specific": (
        "domain_database_adapter",
        "structured_domain_api",
    ),
}
QUERY_SUFFIXES = {
    "official_statistics": ("official statistics", "data", "statistics", "official data"),
    "official": ("official", "official source", "site:.gov", "site:.edu"),
    "scholarly": ("study", "scholarly", "review", "research"),
    "news": ("news", "latest", "report"),
    "domain_specific": ("database", "registry", "source"),
}


def plan_source_family_catalog_collection(
    *,
    acquisition_plan_path: str | Path,
    tasks_jsonl_path: str | Path,
    report_json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    max_query_variants: int = 8,
    max_examples: int = 20,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deduplicated source-catalog collection tasks from gap rows."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if max_query_variants <= 0:
        raise ValueError("max_query_variants must be positive.")
    rows = _load_acquisition_rows(acquisition_plan_path)
    grouped: dict[tuple[str, str, bool, bool], list[Mapping[str, Any]]] = defaultdict(list)
    family_gap_count = 0
    for row in rows:
        missing_families = _families(row.get("missing_source_families", ()))
        family_gap_count += len(missing_families)
        for family in missing_families:
            grouped[(
                family,
                _normalized_query_key(_clean(row.get("query"))),
                bool(row.get("freshness_required") or row.get("requires_timestamp")),
                bool(row.get("official_source_preferred")),
            )].append(row)
    tasks = tuple(
        _collection_task(
            family=family,
            query_key=query_key,
            freshness_required=freshness_required,
            official_source_preferred=official_source_preferred,
            rows=group_rows,
            max_query_variants=max_query_variants,
        )
        for (family, query_key, freshness_required, official_source_preferred), group_rows in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])
        )
    )
    _write_jsonl(tasks_jsonl_path, tasks, compact=compact_json)
    summary = _summary(
        rows,
        tasks,
        family_gap_count=family_gap_count,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready_for_source_collection" if tasks else "empty",
        "source": {
            "acquisition_plan": str(acquisition_plan_path),
        },
        "paths": {
            "collection_tasks": str(tasks_jsonl_path),
        },
        "config": {
            "max_query_variants": int(max_query_variants),
            "max_examples": int(max_examples),
        },
        "summary": summary,
        "examples": tuple(tasks[:max_examples]),
        "metadata": dict(metadata or {}),
    }
    if report_json_path is not None:
        _write_json(report_json_path, payload, compact=compact_json)
    if artifact_manifest_path is not None:
        artifacts: dict[str, str | Path] = {
            "source_family_acquisition_plan": Path(acquisition_plan_path),
            "source_family_collection_tasks": Path(tasks_jsonl_path),
        }
        if report_json_path is not None:
            artifacts["collection_plan_report"] = Path(report_json_path)
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "acquisition_row_count": summary["acquisition_row_count"],
                "family_gap_count": summary["family_gap_count"],
                "collection_task_count": summary["collection_task_count"],
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
            path=report_json_path or tasks_jsonl_path,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "acquisition_row_count": summary["acquisition_row_count"],
                "family_gap_count": summary["family_gap_count"],
                "collection_task_count": summary["collection_task_count"],
                "artifact_manifest": payload.get("artifact_manifest"),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _collection_task(
    *,
    family: str,
    query_key: str,
    freshness_required: bool,
    official_source_preferred: bool,
    rows: Sequence[Mapping[str, Any]],
    max_query_variants: int,
) -> dict[str, Any]:
    primary_query = _choose_primary_query(rows)
    request_ids = _dedupe(_clean(row.get("request_id")) for row in rows)
    queue_fingerprints = _dedupe(
        _clean(_mapping(row.get("metadata")).get("source_queue_request_sha256"))
        for row in rows
    )
    alternate_queries = _dedupe(
        query
        for row in rows
        for query in _string_sequence(row.get("alternate_queries", ()))
    )
    query_hints = _dedupe(
        hint
        for row in rows
        for hint in _string_sequence(row.get("query_hints", ()))
    )
    rationales = _dedupe(
        rationale
        for row in rows
        for rationale in _string_sequence(row.get("rationale", ()))
    )
    search_queries = _search_queries(
        primary_query,
        alternate_queries=alternate_queries,
        family=family,
        max_items=max_query_variants,
    )
    priority_counts = Counter(_clean(row.get("priority")) or "unspecified" for row in rows)
    question_type_counts = Counter(_clean(row.get("question_type")) or "unspecified" for row in rows)
    task_key = strict_json_dumps({
        "family": family,
        "query_key": query_key,
        "freshness_required": freshness_required,
        "official_source_preferred": official_source_preferred,
        "request_ids": request_ids,
    }, sort_keys=True)
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "usage": "source_catalog_collection_only",
        "not_verifier_evidence": True,
        "task_id": f"catalog-{family}-{hashlib.sha256(task_key.encode('utf-8')).hexdigest()[:12]}",
        "source_family": family,
        "query": primary_query,
        "query_key": query_key,
        "search_queries": search_queries,
        "provider_hints": PROVIDER_HINTS.get(family, ("source_catalog_adapter",)),
        "freshness_required": freshness_required,
        "official_source_preferred": official_source_preferred,
        "request_count": len(request_ids),
        "request_ids": request_ids,
        "source_queue_request_sha256": queue_fingerprints,
        "priority_counts": _sorted_counter(priority_counts),
        "question_type_counts": _sorted_counter(question_type_counts),
        "query_hints": query_hints,
        "rationale": rationales,
        "metadata": {
            "input_row_count": len(rows),
            "collection_boundary": "catalog_task_not_evidence",
        },
    }


def _search_queries(
    query: str,
    *,
    alternate_queries: Sequence[str],
    family: str,
    max_items: int,
) -> tuple[str, ...]:
    bases = _dedupe((query, *alternate_queries))
    suffixes = QUERY_SUFFIXES.get(family, ())
    variants: list[str] = []
    for base in bases:
        variants.append(base)
        for suffix in suffixes:
            if suffix.casefold() in base.casefold():
                continue
            variants.append(f"{base} {suffix}")
    return _dedupe(variants)[:max_items]


def _summary(
    rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    *,
    family_gap_count: int,
) -> dict[str, Any]:
    task_families = Counter(str(task.get("source_family")) for task in tasks)
    provider_hints = Counter(
        str(provider)
        for task in tasks
        for provider in _string_sequence(task.get("provider_hints", ()))
    )
    rows_per_task = [int(task.get("metadata", {}).get("input_row_count", 0)) for task in tasks]
    source_fingerprints = {
        fingerprint
        for task in tasks
        for fingerprint in _string_sequence(task.get("source_queue_request_sha256", ()))
    }
    return {
        "acquisition_row_count": len(rows),
        "family_gap_count": int(family_gap_count),
        "collection_task_count": len(tasks),
        "source_queue_request_count": len(source_fingerprints),
        "task_source_family_counts": _sorted_counter(task_families),
        "provider_hint_counts": _sorted_counter(provider_hints),
        "freshness_required_task_count": sum(1 for task in tasks if task.get("freshness_required")),
        "official_source_preferred_task_count": sum(1 for task in tasks if task.get("official_source_preferred")),
        "max_input_rows_per_task": max(rows_per_task) if rows_per_task else 0,
        "mean_input_rows_per_task": (sum(rows_per_task) / len(rows_per_task)) if rows_per_task else 0.0,
        "deduplication_ratio": (float(family_gap_count) / len(tasks)) if tasks else 0.0,
    }


def _load_acquisition_rows(path: str | Path) -> tuple[dict[str, Any], ...]:
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
            if not _clean(payload.get("request_id")):
                raise ValueError(f"{path}:{line_no} is missing request_id.")
            if not _clean(payload.get("query")):
                raise ValueError(f"{path}:{line_no} is missing query.")
            if not _families(payload.get("missing_source_families", ())):
                raise ValueError(f"{path}:{line_no} has no missing_source_families.")
            rows.append(dict(payload))
    if not rows:
        raise ValueError("acquisition plan is empty.")
    return tuple(rows)


def _choose_primary_query(rows: Sequence[Mapping[str, Any]]) -> str:
    counter = Counter(_clean(row.get("query")) for row in rows if _clean(row.get("query")))
    if not counter:
        return ""
    return sorted(counter.items(), key=lambda item: (-item[1], len(item[0]), item[0]))[0][0]


def _families(value: Any) -> tuple[str, ...]:
    return tuple(_normalize_family(item) for item in _string_sequence(value) if _normalize_family(item))


def _normalized_query_key(value: str) -> str:
    tokens = TOKEN_RE.findall(value.casefold())
    return " ".join(tokens)


def _normalize_family(value: Any) -> str:
    return _clean(value).casefold().replace("-", "_").replace(" ", "_")


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
    parser.add_argument("--acquisition-plan", required=True)
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--max-query-variants", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = plan_source_family_catalog_collection(
        acquisition_plan_path=args.acquisition_plan,
        tasks_jsonl_path=args.tasks_jsonl,
        report_json_path=args.report_json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        max_query_variants=args.max_query_variants,
        max_examples=args.max_examples,
        compact_json=bool(args.compact_json),
        metadata=_parse_metadata(args.metadata or ()),
    )
    summary = payload["summary"]
    print(
        "source_family_catalog_collection_plan_ok "
        f"status={payload['status']} "
        f"rows={summary['acquisition_row_count']} "
        f"family_gaps={summary['family_gap_count']} "
        f"tasks={summary['collection_task_count']}"
    )


if __name__ == "__main__":
    main()
