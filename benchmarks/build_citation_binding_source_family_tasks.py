"""Build source-family collection tasks from citation-binding collection plans.

``plan_citation_binding_evidence_collection.py`` emits lane-specific
collection requests after citation source-binding audits reject candidate
documents. This bridge converts the source-collectable portion of those
requests into the same non-evidence task schema consumed by the existing
source-family catalog adapters. Review-only lanes stay counted in the report
but do not become adapter work.
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

CITATION_BINDING_WORKFLOW = "citation_binding_evidence_collection_plan"
WORKFLOW = "source_family_catalog_collection_plan"
BRIDGE_WORKFLOW = "citation_binding_source_family_task_bridge"
TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
SUPPORTED_SOURCE_FAMILIES = frozenset({
    "domain_specific",
    "news",
    "official",
    "official_statistics",
    "scholarly",
})
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


def build_citation_binding_source_family_tasks(
    *,
    collection_plan_path: str | Path,
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
    """Convert citation-binding collection requests to adapter-ready tasks."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if max_query_variants <= 0:
        raise ValueError("max_query_variants must be positive.")

    plan_path = Path(collection_plan_path)
    plan = _load_mapping(plan_path)
    if plan.get("workflow") != CITATION_BINDING_WORKFLOW:
        raise ValueError(
            f"collection_plan must have workflow={CITATION_BINDING_WORKFLOW!r}."
        )

    requests = _load_collection_requests(plan, base=plan_path)
    grouped: dict[tuple[str, str, bool, bool], list[Mapping[str, Any]]] = defaultdict(list)
    unsupported_source_families: Counter[str] = Counter()
    collectable_request_ids: set[str] = set()
    family_gap_count = 0
    for request in requests:
        supported_families = []
        for family in _families(request.get("preferred_source_families", ())):
            if family not in SUPPORTED_SOURCE_FAMILIES:
                unsupported_source_families[family] += 1
                continue
            supported_families.append(family)
        if not supported_families:
            continue
        collectable_request_ids.add(_request_identity(request))
        query_key = _normalized_query_key(_primary_query(request))
        freshness_required = _freshness_required(request)
        for family in _dedupe(supported_families):
            family_gap_count += 1
            grouped[(
                family,
                query_key,
                freshness_required,
                _official_source_preferred(request, family=family),
            )].append(request)

    tasks = tuple(
        _collection_task(
            family=family,
            query_key=query_key,
            freshness_required=freshness_required,
            official_source_preferred=official_source_preferred,
            requests=group_requests,
            max_query_variants=max_query_variants,
        )
        for (family, query_key, freshness_required, official_source_preferred), group_requests in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])
        )
    )
    _write_jsonl(tasks_jsonl_path, tasks, compact=compact_json)

    summary = _summary(
        requests,
        tasks,
        family_gap_count=family_gap_count,
        collectable_request_count=len(collectable_request_ids),
        unsupported_source_family_counts=unsupported_source_families,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "bridge_workflow": BRIDGE_WORKFLOW,
        "status": _status(requests=requests, tasks=tasks),
        "source": {
            "collection_plan": str(plan_path),
            "source_workflow": CITATION_BINDING_WORKFLOW,
        },
        "paths": {
            "collection_tasks": str(tasks_jsonl_path),
        },
        "config": {
            "max_query_variants": int(max_query_variants),
            "max_examples": int(max_examples),
            "supported_source_families": tuple(sorted(SUPPORTED_SOURCE_FAMILIES)),
        },
        "summary": summary,
        "examples": tuple(tasks[:max_examples]),
        "metadata": {
            "source_workflow": CITATION_BINDING_WORKFLOW,
            **dict(metadata or {}),
        },
    }
    if report_json_path is not None:
        _write_json(report_json_path, payload, compact=compact_json)
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        artifacts: dict[str, str | Path] = {
            "citation_binding_collection_plan": plan_path,
            "source_family_collection_tasks": Path(tasks_jsonl_path),
        }
        requests_path = _resolve_path(_nested(plan, "paths", "collection_requests"), base=plan_path)
        if requests_path is not None:
            artifacts["citation_binding_collection_requests"] = requests_path
        if report_json_path is not None:
            artifacts["collection_plan_report"] = Path(report_json_path)
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": WORKFLOW,
                "bridge_workflow": BRIDGE_WORKFLOW,
                "status": payload["status"],
                "collection_request_count": summary["collection_request_count"],
                "collectable_request_count": summary["collectable_request_count"],
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
                "bridge_workflow": BRIDGE_WORKFLOW,
                "status": payload["status"],
                "collection_request_count": summary["collection_request_count"],
                "collectable_request_count": summary["collectable_request_count"],
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
    requests: Sequence[Mapping[str, Any]],
    max_query_variants: int,
) -> dict[str, Any]:
    primary_query = _choose_primary_query(requests)
    collection_request_ids = _dedupe(
        _clean(request.get("collection_request_id")) for request in requests
    )
    request_ids = _dedupe(_clean(request.get("request_id")) for request in requests)
    alternate_queries = _dedupe(
        query
        for request in requests
        for query in _string_sequence(request.get("query_seeds", ()))
    )
    issue_codes = _dedupe(
        issue
        for request in requests
        for issue in _string_sequence(request.get("issue_codes", ()))
    )
    required_fields = _dedupe(
        field
        for request in requests
        for field in _string_sequence(request.get("required_fields", ()))
    )
    adapter_hints = _dedupe(
        hint
        for request in requests
        for hint in _string_sequence(request.get("adapter_hints", ()))
    )
    lane_counts = Counter(_clean(request.get("lane")) or "unspecified" for request in requests)
    priority_counts = Counter(_clean(request.get("priority")) or "unspecified" for request in requests)
    question_type_counts = Counter(
        _clean(request.get("question_type")) or "unspecified" for request in requests
    )
    task_key = strict_json_dumps({
        "bridge": BRIDGE_WORKFLOW,
        "family": family,
        "query_key": query_key,
        "freshness_required": freshness_required,
        "official_source_preferred": official_source_preferred,
        "collection_request_ids": collection_request_ids,
    }, sort_keys=True)
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "origin_workflow": CITATION_BINDING_WORKFLOW,
        "usage": "source_catalog_collection_only",
        "not_verifier_evidence": True,
        "task_id": f"citation-binding-catalog-{family}-{hashlib.sha256(task_key.encode('utf-8')).hexdigest()[:12]}",
        "source_family": family,
        "query": primary_query,
        "query_key": query_key,
        "search_queries": _search_queries(
            primary_query,
            alternate_queries=alternate_queries,
            family=family,
            max_items=max_query_variants,
        ),
        "provider_hints": PROVIDER_HINTS.get(family, ("source_catalog_adapter",)),
        "freshness_required": freshness_required,
        "official_source_preferred": official_source_preferred,
        "request_count": len(collection_request_ids),
        "collection_request_ids": collection_request_ids,
        "request_ids": request_ids,
        "issue_codes": issue_codes,
        "required_fields": required_fields,
        "adapter_hints": adapter_hints,
        "priority_counts": _sorted_counter(priority_counts),
        "question_type_counts": _sorted_counter(question_type_counts),
        "lane_counts": _sorted_counter(lane_counts),
        "source_queue_request_sha256": (),
        "metadata": {
            "input_request_count": len(requests),
            "source_workflow": CITATION_BINDING_WORKFLOW,
            "bridge_workflow": BRIDGE_WORKFLOW,
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
    requests: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    *,
    family_gap_count: int,
    collectable_request_count: int,
    unsupported_source_family_counts: Counter[str],
) -> dict[str, Any]:
    request_lanes = Counter(_clean(request.get("lane")) or "unspecified" for request in requests)
    request_priorities = Counter(_clean(request.get("priority")) or "unspecified" for request in requests)
    preferred_families = Counter(
        family
        for request in requests
        for family in _families(request.get("preferred_source_families", ()))
    )
    task_families = Counter(str(task.get("source_family")) for task in tasks)
    task_lanes = Counter(
        str(lane)
        for task in tasks
        for lane, count in _mapping(task.get("lane_counts")).items()
        for _ in range(_int_or_zero(count))
    )
    provider_hints = Counter(
        str(provider)
        for task in tasks
        for provider in _string_sequence(task.get("provider_hints", ()))
    )
    rows_per_task = [
        _int_or_zero(_mapping(task.get("metadata")).get("input_request_count"))
        for task in tasks
    ]
    return {
        "collection_request_count": len(requests),
        "collectable_request_count": int(collectable_request_count),
        "review_only_request_count": max(0, len(requests) - int(collectable_request_count)),
        "family_gap_count": int(family_gap_count),
        "collection_task_count": len(tasks),
        "lane_counts": _sorted_counter(request_lanes),
        "collectable_lane_counts": _sorted_counter(task_lanes),
        "priority_counts": _sorted_counter(request_priorities),
        "preferred_source_family_counts": _sorted_counter(preferred_families),
        "unsupported_source_family_counts": _sorted_counter(unsupported_source_family_counts),
        "task_source_family_counts": _sorted_counter(task_families),
        "provider_hint_counts": _sorted_counter(provider_hints),
        "freshness_required_task_count": sum(1 for task in tasks if task.get("freshness_required")),
        "official_source_preferred_task_count": sum(
            1 for task in tasks if task.get("official_source_preferred")
        ),
        "max_input_requests_per_task": max(rows_per_task) if rows_per_task else 0,
        "mean_input_requests_per_task": (sum(rows_per_task) / len(rows_per_task)) if rows_per_task else 0.0,
        "deduplication_ratio": (float(family_gap_count) / len(tasks)) if tasks else 0.0,
    }


def _load_collection_requests(
    plan: Mapping[str, Any],
    *,
    base: Path,
) -> tuple[Mapping[str, Any], ...]:
    requests_path = _resolve_path(_nested(plan, "paths", "collection_requests"), base=base)
    if requests_path is not None and requests_path.exists():
        return _load_jsonl(requests_path)
    requests = tuple(item for item in _sequence(plan.get("collection_requests")) if isinstance(item, Mapping))
    if requests:
        return requests
    raise ValueError("collection plan has no collection requests or request sidecar.")


def _load_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no} must contain a JSON object.")
            rows.append(dict(payload))
    return tuple(rows)


def _status(
    *,
    requests: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
) -> str:
    if tasks:
        return "ready_for_source_collection"
    if requests:
        return "needs_review"
    return "empty"


def _choose_primary_query(requests: Sequence[Mapping[str, Any]]) -> str:
    counter = Counter(_primary_query(request) for request in requests if _primary_query(request))
    if not counter:
        return ""
    return sorted(counter.items(), key=lambda item: (-item[1], len(item[0]), item[0]))[0][0]


def _primary_query(request: Mapping[str, Any]) -> str:
    query = _clean(request.get("query"))
    if query:
        return query
    for seed in _string_sequence(request.get("query_seeds", ())):
        if seed:
            return seed
    return ""


def _request_identity(request: Mapping[str, Any]) -> str:
    return (
        _clean(request.get("collection_request_id"))
        or _clean(request.get("request_id"))
        or hashlib.sha256(strict_json_dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
    )


def _freshness_required(request: Mapping[str, Any]) -> bool:
    lane = _clean(request.get("lane"))
    return bool(request.get("requires_timestamp")) or lane == "temporal_evidence"


def _official_source_preferred(request: Mapping[str, Any], *, family: str) -> bool:
    lane = _clean(request.get("lane"))
    if family in {"official", "official_statistics"}:
        return True
    return lane in {"numeric_statistical_evidence", "temporal_evidence"}


def _families(value: Any) -> tuple[str, ...]:
    return tuple(_normalize_family(item) for item in _string_sequence(value) if _normalize_family(item))


def _normalize_family(value: Any) -> str:
    return _clean(value).casefold().replace("-", "_").replace(" ", "_")


def _normalized_query_key(value: str) -> str:
    tokens = TOKEN_RE.findall(value.casefold())
    return " ".join(tokens)


def _load_mapping(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _resolve_path(value: Any, *, base: Path | None) -> Path | None:
    text = _clean(value)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute() or base is None:
        return path
    return base.parent / path


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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _string_sequence(value: Any) -> tuple[str, ...]:
    return tuple(str(item).strip() for item in _sequence(value) if str(item).strip())


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


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(
        sorted(((key, value) for key, value in counter.items() if key), key=lambda item: (-item[1], item[0]))
    )


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-plan", required=True)
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
    payload = build_citation_binding_source_family_tasks(
        collection_plan_path=args.collection_plan,
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
        "citation_binding_source_family_tasks_ok "
        f"status={payload['status']} "
        f"requests={summary['collection_request_count']} "
        f"collectable={summary['collectable_request_count']} "
        f"tasks={summary['collection_task_count']}"
    )


if __name__ == "__main__":
    main()
