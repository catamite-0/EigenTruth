"""Audit source-family result coverage and emit catalog acquisition targets.

This helper consumes sanitized source-family citation/search requests plus the
adapter result JSONL produced by ``run_source_family_citation_search_adapter``.
It does not create verifier evidence. Instead, it measures whether returned
documents cover the non-fallback source families requested by the plan and
writes a follow-up acquisition JSONL for missing official, scholarly,
statistical, news, or domain-specific catalog coverage.
"""

from __future__ import annotations

import argparse
import json
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

WORKFLOW = "source_family_coverage_audit"
FALLBACK_FAMILIES = ("reference", "encyclopedic")
OFFICIAL_FAMILIES = {"official", "official_statistics", "domain_specific"}
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


def audit_source_family_coverage(
    *,
    requests_path: str | Path,
    adapter_results_path: str | Path,
    report_json_path: str | Path,
    acquisition_plan_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    fallback_families: Sequence[str] = FALLBACK_FAMILIES,
    max_examples: int = 20,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit source-family coverage and write missing-family acquisition tasks."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    fallback = {_normalize_family(item) for item in fallback_families if _normalize_family(item)}
    requests = _load_jsonl(requests_path, kind="request")
    results_by_request = _load_results(adapter_results_path)
    plan_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    preferred_family_counts: Counter[str] = Counter()
    target_family_counts: Counter[str] = Counter()
    covered_target_family_counts: Counter[str] = Counter()
    missing_target_family_counts: Counter[str] = Counter()
    result_family_counts: Counter[str] = Counter()
    result_provider_counts: Counter[str] = Counter()
    request_with_results = 0
    request_with_target_family = 0
    request_with_official_result = 0
    request_with_fresh_result = 0
    official_source_preferred_with_official_result = 0
    freshness_required_with_fresh_result = 0
    official_source_preferred = 0
    freshness_required = 0

    for request in requests:
        request_id = _clean(request.get("request_id"))
        if not request_id:
            raise ValueError("request row is missing request_id.")
        _reject_reserved_fields(request, source=f"request:{request_id}")
        request_metadata = _mapping(request.get("metadata"))
        _reject_reserved_fields(request_metadata, source=f"request:{request_id}:metadata")
        plan = _source_family_plan(request)
        preferred = _preferred_families(plan, request_metadata)
        target_families = _target_families(preferred, fallback=fallback)
        results = results_by_request.get(request_id, ())
        result_families = tuple(_normalize_family(result.get("source_family")) for result in results)
        result_providers = tuple(_clean(result.get("provider")) for result in results if _clean(result.get("provider")))
        covered_targets = tuple(family for family in target_families if family in set(result_families))
        missing_targets = tuple(family for family in target_families if family not in set(result_families))
        official_preferred = bool(
            plan.get("official_source_preferred")
            or request_metadata.get("official_source_preferred")
        )
        fresh_required = bool(
            plan.get("freshness_required")
            or request.get("requires_timestamp")
            or request_metadata.get("freshness_required")
        )
        official_hits = tuple(result for result in results if _is_official_result(result))
        fresh_hits = tuple(
            result
            for result in results
            if result.get("published_at") or result.get("timestamp") or result.get("freshness_match")
        )
        status = "covered" if not missing_targets else "needs_source_family_catalog"

        preferred_family_counts.update(preferred)
        target_family_counts.update(target_families)
        covered_target_family_counts.update(covered_targets)
        missing_target_family_counts.update(missing_targets)
        result_family_counts.update(family for family in result_families if family)
        result_provider_counts.update(provider for provider in result_providers if provider)
        request_with_results += int(bool(results))
        request_with_target_family += int(bool(covered_targets))
        request_with_official_result += int(bool(official_hits))
        request_with_fresh_result += int(bool(fresh_hits))
        official_source_preferred += int(official_preferred)
        freshness_required += int(fresh_required)
        official_source_preferred_with_official_result += int(official_preferred and bool(official_hits))
        freshness_required_with_fresh_result += int(fresh_required and bool(fresh_hits))

        record = {
            "request_id": request_id,
            "query": _clean(request.get("query")),
            "priority": _clean(request.get("priority")),
            "question_type": _clean(request.get("question_type")),
            "status": status,
            "preferred_source_families": preferred,
            "target_source_families": target_families,
            "covered_target_source_families": covered_targets,
            "missing_target_source_families": missing_targets,
            "result_source_families": result_families,
            "result_providers": result_providers,
            "result_count": len(results),
            "official_source_preferred": official_preferred,
            "has_official_result": bool(official_hits),
            "freshness_required": fresh_required,
            "has_fresh_result": bool(fresh_hits),
        }
        records.append(record)
        if missing_targets:
            plan_rows.append(_acquisition_row(
                request,
                plan=plan,
                record=record,
                source_queue_request_sha256=_clean(request_metadata.get("source_queue_request_sha256")),
            ))

    summary = {
        "request_count": len(requests),
        "result_request_count": len(results_by_request),
        "request_with_results_count": request_with_results,
        "request_with_target_family_count": request_with_target_family,
        "request_missing_target_family_count": len(plan_rows),
        "official_source_preferred_count": official_source_preferred,
        "request_with_official_result_count": request_with_official_result,
        "official_source_preferred_with_official_result_count": official_source_preferred_with_official_result,
        "freshness_required_count": freshness_required,
        "request_with_fresh_result_count": request_with_fresh_result,
        "freshness_required_with_fresh_result_count": freshness_required_with_fresh_result,
        "preferred_source_family_counts": _sorted_counter(preferred_family_counts),
        "target_source_family_counts": _sorted_counter(target_family_counts),
        "covered_target_source_family_counts": _sorted_counter(covered_target_family_counts),
        "missing_target_source_family_counts": _sorted_counter(missing_target_family_counts),
        "result_source_family_counts": _sorted_counter(result_family_counts),
        "result_provider_counts": _sorted_counter(result_provider_counts),
        "acquisition_plan_count": len(plan_rows),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "needs_catalog_expansion" if plan_rows else "covered",
        "source": {
            "requests": str(requests_path),
            "adapter_results": str(adapter_results_path),
        },
        "config": {
            "fallback_families": tuple(sorted(fallback)),
            "max_examples": int(max_examples),
        },
        "summary": summary,
        "examples": _examples(records, max_examples=max_examples),
        "metadata": dict(metadata or {}),
    }
    _write_json(report_json_path, payload, compact=compact_json)
    if acquisition_plan_path is not None:
        _write_jsonl(acquisition_plan_path, plan_rows, compact=compact_json)
        payload["paths"] = {"acquisition_plan": str(acquisition_plan_path)}
    if artifact_manifest_path is not None:
        artifacts: dict[str, str | Path] = {
            "coverage_audit_report": Path(report_json_path),
            "source_family_requests": Path(requests_path),
            "source_family_adapter_results": Path(adapter_results_path),
        }
        if acquisition_plan_path is not None:
            artifacts["source_family_acquisition_plan"] = Path(acquisition_plan_path)
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "request_count": summary["request_count"],
                "acquisition_plan_count": summary["acquisition_plan_count"],
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
            path=report_json_path,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "request_count": summary["request_count"],
                "acquisition_plan_count": summary["acquisition_plan_count"],
                "request_missing_target_family_count": summary["request_missing_target_family_count"],
                "artifact_manifest": payload.get("artifact_manifest"),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _acquisition_row(
    request: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    record: Mapping[str, Any],
    source_queue_request_sha256: str,
) -> dict[str, Any]:
    metadata = _mapping(request.get("metadata"))
    row_metadata = {
        "source_queue_request_sha256": source_queue_request_sha256,
        "query_strategy": _clean(metadata.get("query_strategy") or metadata.get("query_mode")),
        "question_type": record.get("question_type"),
        "keyword_terms": _string_sequence(metadata.get("keyword_terms", ())),
    }
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "usage": "source_catalog_acquisition_only",
        "not_verifier_evidence": True,
        "request_id": record["request_id"],
        "query": record["query"],
        "alternate_queries": _string_sequence(request.get("alternate_queries", ())),
        "priority": record["priority"],
        "question_type": record["question_type"],
        "missing_source_families": record["missing_target_source_families"],
        "preferred_source_families": record["preferred_source_families"],
        "result_source_families": record["result_source_families"],
        "official_source_preferred": bool(record["official_source_preferred"]),
        "freshness_required": bool(record["freshness_required"]),
        "requires_timestamp": bool(request.get("requires_timestamp")),
        "query_hints": _string_sequence(plan.get("query_hints", ())),
        "rationale": _string_sequence(plan.get("rationale", ())),
        "metadata": _drop_empty(row_metadata),
    }


def _load_results(path: str | Path) -> dict[str, tuple[Mapping[str, Any], ...]]:
    rows = _load_jsonl(path, kind="adapter result")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        request_id = _clean(row.get("request_id"))
        if not request_id:
            raise ValueError("adapter result row is missing request_id.")
        _reject_reserved_fields(row, source=f"adapter_result:{request_id}")
        _reject_reserved_fields(_mapping(row.get("metadata")), source=f"adapter_result:{request_id}:metadata")
        for result in _mapping_sequence(row.get("results", ())):
            _reject_reserved_fields(result, source=f"adapter_result:{request_id}:result")
            _reject_reserved_fields(
                _mapping(result.get("metadata")),
                source=f"adapter_result:{request_id}:result_metadata",
            )
            grouped[request_id].append(dict(result))
    return {key: tuple(value) for key, value in grouped.items()}


def _source_family_plan(request: Mapping[str, Any]) -> Mapping[str, Any]:
    plan = request.get("source_family_plan")
    if isinstance(plan, Mapping):
        return plan
    return {}


def _preferred_families(plan: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[str, ...]:
    families = tuple(
        _normalize_family(item)
        for item in (
            _string_sequence(plan.get("families", ()))
            or _string_sequence(metadata.get("preferred_source_families", ()))
        )
        if _normalize_family(item)
    )
    return families or ("reference",)


def _target_families(preferred: Sequence[str], *, fallback: set[str]) -> tuple[str, ...]:
    nonfallback = tuple(family for family in preferred if family not in fallback)
    if nonfallback:
        return nonfallback
    return tuple(preferred[:1]) or ("reference",)


def _examples(records: Sequence[Mapping[str, Any]], *, max_examples: int) -> dict[str, tuple[Mapping[str, Any], ...]]:
    missing = tuple(record for record in records if record["status"] != "covered")[:max_examples]
    covered = tuple(record for record in records if record["status"] == "covered")[:max_examples]
    return {"missing_target_family": missing, "covered": covered}


def _is_official_result(result: Mapping[str, Any]) -> bool:
    family = _normalize_family(result.get("source_family"))
    metadata = _mapping(result.get("metadata"))
    return (
        family in OFFICIAL_FAMILIES
        or bool(result.get("official_match"))
        or _bool(metadata.get("official_source"))
        or _bool(metadata.get("trusted_source"))
    )


def _reject_reserved_fields(payload: Mapping[str, Any], *, source: str) -> None:
    reserved = sorted(set(str(key) for key in payload) & RESERVED_FIELDS)
    if reserved:
        raise ValueError(f"{source} contains reserved fields: {', '.join(reserved)}")


def _load_jsonl(path: str | Path, *, kind: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no} {kind} must be a JSON object.")
            rows.append(dict(payload))
    return tuple(rows)


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes", "on"}
    return False


def _normalize_family(value: Any) -> str:
    return _clean(value).casefold().replace("-", "_").replace(" ", "_")


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


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--adapter-results", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--acquisition-plan-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--fallback-families", default=",".join(FALLBACK_FAMILIES))
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = audit_source_family_coverage(
        requests_path=args.requests,
        adapter_results_path=args.adapter_results,
        report_json_path=args.json,
        acquisition_plan_path=args.acquisition_plan_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        fallback_families=_parse_csv(args.fallback_families),
        max_examples=args.max_examples,
        compact_json=bool(args.compact_json),
        metadata=_parse_metadata(args.metadata or ()),
    )
    summary = payload["summary"]
    print(
        "source_family_coverage_audit_ok "
        f"status={payload['status']} "
        f"requests={summary['request_count']} "
        f"missing_targets={summary['request_missing_target_family_count']} "
        f"plan={summary['acquisition_plan_count']}"
    )


if __name__ == "__main__":
    main()
