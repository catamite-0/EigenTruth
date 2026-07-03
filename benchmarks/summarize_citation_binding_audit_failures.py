"""Summarize citation/source binding audit evidence-quality gaps.

The citation binding audit filters returned source documents before they become
retrieval-corpus candidates. This report turns rejected binding records into a
read-only review artifact with issue buckets, question-type counts, examples,
and next-action recommendations. It does not promote evidence or relax gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "citation_binding_audit_failure_review"


def summarize_citation_binding_audit_failures(
    binding_audits: Sequence[Mapping[str, Any]],
    *,
    max_examples_per_issue: int = 3,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready evidence-quality review for binding audits."""
    if not binding_audits:
        raise ValueError("binding_audits must not be empty.")
    if int(max_examples_per_issue) < 0:
        raise ValueError("max_examples_per_issue must be non-negative.")

    rows = tuple(
        _review_row(audit, index=index, max_examples_per_issue=int(max_examples_per_issue))
        for index, audit in enumerate(binding_audits, start=1)
    )
    summary = _summary(rows)
    if summary["total_source_document_count"] and not summary["total_accepted_source_document_count"]:
        status = "blocked"
    elif summary["total_rejected_source_document_count"]:
        status = "needs_evidence_quality"
    else:
        status = "monitor"
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Read-only citation binding audit failure review. Rows describe why "
            "source documents failed claim-specific binding; they are not verifier evidence."
        ),
        "summary": summary,
        "audits": rows,
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    binding_audit_paths: Sequence[str | Path],
    output_path: str | Path,
    max_examples_per_issue: int = 3,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, optionally manifest, and optionally register a review."""
    if not binding_audit_paths:
        raise ValueError("binding_audit_paths must not be empty.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")

    paths = tuple(Path(path) for path in binding_audit_paths)
    payload = summarize_citation_binding_audit_failures(
        tuple(_load_mapping(path) for path in paths),
        max_examples_per_issue=max_examples_per_issue,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["source"] = {
        "binding_audits": tuple(str(path) for path in paths),
    }
    if artifact_manifest_path is not None:
        payload["paths"] = {"artifact_manifest": str(artifact_manifest_path)}
    output = Path(output_path)
    _write_json(output, payload, compact=compact_json)

    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            {
                "citation_binding_audit_failure_review": output,
                **{f"binding_audit_{index}": path for index, path in enumerate(paths, start=1)},
            },
            root=manifest_path.parent,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "audit_count": payload["summary"]["audit_count"],
                "dominant_issue": payload["summary"]["dominant_issue"],
                "dominant_recommendation": payload["summary"]["dominant_recommendation"],
                **dict(metadata or {}),
            },
        )
        _write_json(manifest_path, manifest, compact=compact_json)

    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "audit_count": payload["summary"]["audit_count"],
                "dominant_issue": payload["summary"]["dominant_issue"],
                "dominant_recommendation": payload["summary"]["dominant_recommendation"],
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _review_row(
    audit: Mapping[str, Any],
    *,
    index: int,
    max_examples_per_issue: int,
) -> dict[str, Any]:
    summary = _mapping(audit.get("summary"))
    records = tuple(item for item in _sequence(audit.get("records")) if isinstance(item, Mapping))
    rejected = tuple(record for record in records if _clean_text(record.get("status")) == "rejected")
    issue_counts: Counter[str] = Counter()
    question_type_counts: Counter[str] = Counter()
    issue_question_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    recommendation_counts: Counter[str] = Counter()
    for record in rejected:
        question_type = _clean_text(record.get("question_type")) or "unknown"
        question_type_counts[question_type] += 1
        for issue in _issue_codes(record):
            issue_counts[issue] += 1
            issue_question_type_counts[issue][question_type] += 1
            for recommendation in _recommendations_for_issue(issue):
                recommendation_counts[recommendation] += 1

    issue_reviews = []
    for issue, count in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0])):
        examples = tuple(
            _example(record)
            for record in rejected
            if issue in _issue_codes(record)
        )[:max_examples_per_issue]
        issue_reviews.append({
            "issue": issue,
            "count": count,
            "question_type_counts": dict(sorted(issue_question_type_counts[issue].items())),
            "recommendations": _recommendations_for_issue(issue),
            "examples": examples,
        })

    accepted = _int(summary.get("accepted_source_document_count"))
    source_count = _int(summary.get("source_document_count"))
    rejected_count = _int(summary.get("rejected_source_document_count"))
    if not rejected_count and source_count:
        rejected_count = max(source_count - accepted, 0)
    return {
        "index": index,
        "status": _clean_text(audit.get("status")) or "unknown",
        "source_document_count": source_count,
        "accepted_source_document_count": accepted,
        "rejected_source_document_count": rejected_count,
        "accepted_request_count": _int(summary.get("accepted_request_count")),
        "acceptance_rate": _optional_float(summary.get("acceptance_rate")),
        "accepted_request_coverage": _optional_float(summary.get("accepted_request_coverage")),
        "issue_counts": _sorted_counter(issue_counts),
        "question_type_counts": dict(sorted(question_type_counts.items())),
        "dominant_issue": _counter_first(issue_counts),
        "recommendation_counts": _sorted_counter(recommendation_counts),
        "dominant_recommendation": _counter_first(recommendation_counts),
        "issue_reviews": tuple(issue_reviews),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    issue_counts: Counter[str] = Counter()
    recommendation_counts: Counter[str] = Counter()
    question_type_counts: Counter[str] = Counter()
    source_total = 0
    accepted_total = 0
    rejected_total = 0
    accepted_request_total = 0
    for row in rows:
        source_total += _int(row.get("source_document_count"))
        accepted_total += _int(row.get("accepted_source_document_count"))
        rejected_total += _int(row.get("rejected_source_document_count"))
        accepted_request_total += _int(row.get("accepted_request_count"))
        issue_counts.update(_int_mapping(row.get("issue_counts")))
        recommendation_counts.update(_int_mapping(row.get("recommendation_counts")))
        question_type_counts.update(_int_mapping(row.get("question_type_counts")))
    return {
        "audit_count": len(rows),
        "total_source_document_count": source_total,
        "total_accepted_source_document_count": accepted_total,
        "total_rejected_source_document_count": rejected_total,
        "total_accepted_request_count": accepted_request_total,
        "overall_acceptance_rate": _safe_div(accepted_total, source_total),
        "issue_counts": _sorted_counter(issue_counts),
        "question_type_counts": _sorted_counter(question_type_counts),
        "recommendation_counts": _sorted_counter(recommendation_counts),
        "dominant_issue": _counter_first(issue_counts),
        "dominant_recommendation": _counter_first(recommendation_counts),
    }


def _recommendations_for_issue(issue: str) -> tuple[str, ...]:
    if issue in {"missing_source_binding", "unknown_source_binding"}:
        return ("repair_source_binding_provenance",)
    if issue == "numeric_intent_requires_numeric_evidence":
        return ("collect_numeric_or_statistical_evidence", "extract_structured_numeric_facts")
    if issue == "temporal_intent_requires_temporal_evidence" or issue == "missing_fresh_timestamp":
        return ("collect_timestamped_or_temporal_evidence",)
    if issue == "causal_intent_requires_causal_evidence":
        return ("collect_causal_or_procedural_evidence",)
    if issue == "person_intent_requires_relation_evidence":
        return ("collect_role_specific_entity_evidence",)
    if issue == "location_intent_requires_location_evidence":
        return ("collect_location_specific_evidence",)
    if issue.startswith("source_family"):
        return ("expand_or_rerank_source_family_catalog",)
    if issue.startswith("evidence_alignment_misaligned"):
        return ("tighten_claim_evidence_alignment_rules",)
    if issue.startswith("evidence_alignment_insufficient"):
        return ("collect_claim_specific_evidence_spans",)
    if issue.startswith("evidence_alignment"):
        return ("inspect_evidence_alignment_failure",)
    return ("inspect_binding_audit_issue",)


def _example(record: Mapping[str, Any]) -> dict[str, Any]:
    alignment = _mapping(record.get("alignment"))
    intent = _mapping(record.get("intent"))
    source_family = _mapping(record.get("source_family"))
    return {
        "source_document_index": _int(record.get("source_document_index")),
        "request_id": _clean_text(record.get("request_id")),
        "query": _clean_text(record.get("query")),
        "question_type": _clean_text(record.get("question_type")),
        "source": _clean_text(record.get("source")),
        "issue_codes": _issue_codes(record),
        "alignment_status": alignment.get("status"),
        "keyword_overlap": alignment.get("keyword_overlap"),
        "number_recall": alignment.get("number_recall"),
        "entity_recall": alignment.get("entity_recall"),
        "intent_reason": intent.get("reason"),
        "source_family_reason": source_family.get("reason"),
        "requires_timestamp": bool(record.get("requires_timestamp")),
    }


def _issue_codes(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in _sequence(record.get("issue_codes")) if str(item))


def _load_mapping(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--metadata entries must be key=value.")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--metadata keys must be non-empty.")
        metadata[key] = raw
    return metadata


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    indent = None if compact else 2
    output.write_text(strict_json_dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _int_mapping(value: Any) -> dict[str, int]:
    return {
        str(key): _int(item)
        for key, item in _mapping(value).items()
    }


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _safe_div(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(((key, value) for key, value in counter.items() if key), key=lambda item: (-item[1], item[0])))


def _counter_first(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize citation binding audit failure modes.")
    parser.add_argument("--binding-audit", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples-per-issue", type=int, default=3)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=None)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args()
    payload = run(
        binding_audit_paths=tuple(args.binding_audit),
        output_path=args.output,
        max_examples_per_issue=args.max_examples_per_issue,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    print(
        f"{WORKFLOW}_ok status={payload['status']} "
        f"audits={payload['summary']['audit_count']} output={args.output}"
    )


if __name__ == "__main__":
    main()
