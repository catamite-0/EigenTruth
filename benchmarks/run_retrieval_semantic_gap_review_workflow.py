"""Run retrieval semantic-gap handoff through fact review and promotion gates.

This workflow starts from ``eval_verifier_ensemble.py --verified-records-jsonl``
sidecars after source-bound retrieval sweeps. It chains the existing
semantic-gap handoff, alignment fact-review corpus builder, deterministic
route-specific reviewer, and promotion gate into one reproducible local command.

The output is still scoped evidence. Approved rows become structured source
documents for later covered-fact route audits; this workflow does not promote a
broad retrieval route or treat handoff requests as verifier evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.build_alignment_fact_review_corpus import run as run_fact_review_corpus  # noqa: E402
from benchmarks.build_retrieval_semantic_gap_handoff import (  # noqa: E402
    DEFAULT_MAX_HITS_PER_TARGET,
    MODES,
)
from benchmarks.build_retrieval_semantic_gap_handoff import run as run_semantic_gap_handoff  # noqa: E402
from benchmarks.promote_alignment_fact_review_corpus import run as run_promotion_gate  # noqa: E402
from benchmarks.review_alignment_fact_review_corpus import (  # noqa: E402
    REVIEWER as RULE_REVIEWER,
)
from benchmarks.review_alignment_fact_review_corpus import run as run_rule_review  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "retrieval_semantic_gap_review_workflow"


def run_retrieval_semantic_gap_review_workflow(
    *,
    verified_records_jsonl: str | Path,
    output_dir: str | Path,
    record_indices_json: str | Path | None = None,
    workflow_report_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    mode: str = "false_negative_with_hits",
    min_hits: int = 1,
    max_targets: int | None = None,
    max_hits_per_target: int = DEFAULT_MAX_HITS_PER_TARGET,
    min_confidence: float = 0.0,
    reviewer: str = RULE_REVIEWER,
    reviewed_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Run semantic-gap handoff, fact review, rule review, and promotion gate."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(workflow_report_path or output / "retrieval-semantic-gap-review-workflow.json")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")
    paths = _paths(output)
    workflow_metadata = {
        **dict(metadata or {}),
        "parent_workflow": WORKFLOW,
    }

    handoff = run_semantic_gap_handoff(
        verified_records_jsonl=verified_records_jsonl,
        output_path=paths["handoff"],
        mode=mode,
        min_hits=min_hits,
        max_targets=max_targets,
        max_hits_per_target=max_hits_per_target,
        record_indices_json=record_indices_json,
        artifact_manifest_path=paths["handoff_manifest"],
        metadata=workflow_metadata,
        compact_json=compact_json,
    )
    review_payload = run_fact_review_corpus(
        candidates_path=paths["handoff"],
        output_path=paths["fact_review_corpus"],
        report_json_path=paths["fact_review_report"],
        records_jsonl_path=paths["fact_review_records"],
        artifact_manifest_path=paths["fact_review_manifest"],
        min_confidence=min_confidence,
        metadata=workflow_metadata,
        compact_json=compact_json,
    )
    rule_review = run_rule_review(
        review_corpus_path=paths["fact_review_corpus"],
        output_dir=paths["rule_review_dir"],
        artifact_manifest_path=paths["rule_review_manifest"],
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        metadata=workflow_metadata,
        compact_json=compact_json,
    )
    promotion = run_promotion_gate(
        review_corpus_path=paths["fact_review_corpus"],
        review_decisions_path=paths["rule_review_decisions"],
        output_dir=paths["promotion_dir"],
        artifact_manifest_path=paths["promotion_manifest"],
        metadata=workflow_metadata,
        compact_json=compact_json,
    )

    summary = _summary(handoff=handoff, review_payload=review_payload, rule_review=rule_review, promotion=promotion)
    status = _workflow_status(
        handoff=handoff,
        review_payload=review_payload,
        rule_review=rule_review,
        promotion=promotion,
    )
    report = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Chains retrieval semantic-gap handoff into alignment fact review, "
            "deterministic rule review, and the reviewed-source promotion gate. "
            "It only emits scoped reviewed source documents for later covered-fact "
            "route audits; it does not promote broad retrieval correction."
        ),
        "label_usage": {
            "labels_used_for_gap_selection": bool(
                handoff.get("label_usage", {}).get("labels_used_for_gap_selection")
            ),
            "labels_copied_to_review_or_promotion_outputs": False,
            "handoff_requests_are_verifier_evidence": False,
            "approved_source_documents_are_scoped_reviewed_facts": True,
        },
        "source": {
            "verified_records_jsonl": str(verified_records_jsonl),
            "record_indices_json": None if record_indices_json is None else str(record_indices_json),
        },
        "config": {
            "mode": mode,
            "min_hits": int(min_hits),
            "max_targets": max_targets,
            "max_hits_per_target": int(max_hits_per_target),
            "min_confidence": float(min_confidence),
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
        },
        "summary": summary,
        "stage_status": {
            "semantic_gap_handoff": handoff.get("status"),
            "fact_review_corpus": review_payload["report"].get("status"),
            "rule_review": rule_review.get("status"),
            "promotion_gate": promotion.get("status"),
        },
        "paths": _report_paths(paths=paths, report_path=report_path, manifest_path=manifest_path),
        "metadata": dict(metadata or {}),
    }
    _write_json(report_path, report, compact=compact_json)
    manifest = build_artifact_manifest(
        {
            "retrieval_semantic_gap_review_workflow_report": report_path,
            "semantic_gap_handoff": paths["handoff"],
            "semantic_gap_handoff_manifest": paths["handoff_manifest"],
            "alignment_fact_review_corpus": paths["fact_review_corpus"],
            "alignment_fact_review_report": paths["fact_review_report"],
            "alignment_fact_review_records": paths["fact_review_records"],
            "alignment_fact_review_manifest": paths["fact_review_manifest"],
            "alignment_fact_rule_review_report": paths["rule_review_report"],
            "alignment_fact_rule_review_decisions": paths["rule_review_decisions"],
            "alignment_fact_rule_review_records": paths["rule_review_records"],
            "alignment_fact_rule_review_manifest": paths["rule_review_manifest"],
            "alignment_fact_review_promotion_gate_report": paths["promotion_report"],
            "approved_source_documents": paths["approved_source_documents"],
            "promotion_gate_records": paths["promotion_records"],
            "promotion_gate_manifest": paths["promotion_manifest"],
            "verified_records_jsonl": Path(verified_records_jsonl),
            "record_indices_json": None if record_indices_json is None else Path(record_indices_json),
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": status,
            "candidate_count": summary["semantic_gap_candidate_count"],
            "fact_candidate_count": summary["semantic_gap_fact_candidate_count"],
            "review_document_count": summary["fact_review_document_count"],
            "rule_review_approved_count": summary["rule_review_approved_count"],
            "approved_source_document_count": summary["approved_source_document_count"],
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
                "artifact_manifest": str(manifest_path),
                "candidate_count": summary["semantic_gap_candidate_count"],
                "fact_candidate_count": summary["semantic_gap_fact_candidate_count"],
                "review_document_count": summary["fact_review_document_count"],
                "rule_review_approved_count": summary["rule_review_approved_count"],
                "approved_source_document_count": summary["approved_source_document_count"],
                **dict(metadata or {}),
            },
        ).save_json()
    return report


def run(**kwargs: Any) -> dict[str, Any]:
    """Compatibility wrapper for tests and script-like imports."""
    return run_retrieval_semantic_gap_review_workflow(**kwargs)


def _paths(output: Path) -> dict[str, Path]:
    rule_review_dir = output / "alignment-rule-review"
    promotion_dir = output / "alignment-reviewed-promotion-gate"
    return {
        "handoff": output / "retrieval-semantic-gap-handoff.json",
        "handoff_manifest": output / "retrieval-semantic-gap-handoff.manifest.json",
        "fact_review_corpus": output / "alignment-fact-review-corpus.json",
        "fact_review_report": output / "alignment-fact-review-report.json",
        "fact_review_records": output / "alignment-fact-review-records.jsonl",
        "fact_review_manifest": output / "alignment-fact-review.manifest.json",
        "rule_review_dir": rule_review_dir,
        "rule_review_report": rule_review_dir / "review-report.json",
        "rule_review_decisions": rule_review_dir / "review-decisions.jsonl",
        "rule_review_records": rule_review_dir / "review-records.jsonl",
        "rule_review_manifest": rule_review_dir / "artifact-manifest.json",
        "promotion_dir": promotion_dir,
        "promotion_report": promotion_dir / "promotion-gate-report.json",
        "approved_source_documents": promotion_dir / "approved-source-documents.json",
        "promotion_template": promotion_dir / "review-decision-template.jsonl",
        "promotion_records": promotion_dir / "promotion-gate-records.jsonl",
        "promotion_manifest": promotion_dir / "artifact-manifest.json",
    }


def _summary(
    *,
    handoff: Mapping[str, Any],
    review_payload: Mapping[str, Any],
    rule_review: Mapping[str, Any],
    promotion: Mapping[str, Any],
) -> dict[str, Any]:
    review_report = _mapping(review_payload.get("report"))
    review_summary = _mapping(review_report.get("summary"))
    rule_summary = _mapping(rule_review.get("summary"))
    promotion_summary = _mapping(promotion.get("summary"))
    return {
        "source_record_count": _nested_int(handoff, "summary", "source_record_count"),
        "evaluated_source_record_count": _nested_int(handoff, "summary", "evaluated_source_record_count"),
        "semantic_gap_candidate_count": _nested_int(handoff, "summary", "candidate_count"),
        "semantic_gap_fact_candidate_count": _nested_int(handoff, "summary", "fact_candidate_count"),
        "semantic_gap_request_counts": _nested(handoff, "summary", "request_counts"),
        "semantic_gap_total_request_count": _nested_int(handoff, "summary", "total_request_count"),
        "fact_review_input_candidate_count": _optional_int(review_summary.get("input_candidate_count")),
        "fact_review_document_count": _optional_int(review_summary.get("accepted_document_count")),
        "fact_review_skipped_count": _optional_int(review_summary.get("skipped_count")),
        "fact_review_skipped": review_summary.get("skipped", {}),
        "rule_review_document_count": _optional_int(rule_summary.get("review_document_count")),
        "rule_review_approved_count": _optional_int(rule_summary.get("approved_count")),
        "rule_review_needs_more_evidence_count": _optional_int(
            rule_summary.get("needs_more_evidence_count")
        ),
        "rule_review_reason_counts": rule_summary.get("reason_counts", {}),
        "approved_source_document_count": _optional_int(
            promotion_summary.get("approved_source_document_count")
        ),
        "pending_review_count": _optional_int(promotion_summary.get("pending_review_count")),
        "rejected_count": _optional_int(promotion_summary.get("rejected_count")),
        "promotion_skip_counts": promotion_summary.get("skip_counts", {}),
    }


def _workflow_status(
    *,
    handoff: Mapping[str, Any],
    review_payload: Mapping[str, Any],
    rule_review: Mapping[str, Any],
    promotion: Mapping[str, Any],
) -> str:
    if handoff.get("status") == "empty":
        return "empty"
    review_summary = _mapping(_mapping(review_payload.get("report")).get("summary"))
    if _optional_int(review_summary.get("accepted_document_count")) <= 0:
        return "needs_more_evidence"
    rule_summary = _mapping(rule_review.get("summary"))
    if _optional_int(rule_summary.get("approved_count")) <= 0:
        return "needs_more_evidence"
    if promotion.get("status") == "ready_for_structured_qa":
        return "ready_for_structured_qa"
    if promotion.get("status") in {"blocked", "partial", "needs_review"}:
        return str(promotion.get("status"))
    return "needs_more_evidence"


def _report_paths(*, paths: Mapping[str, Path], report_path: Path, manifest_path: Path) -> dict[str, str]:
    return {
        "workflow_report": str(report_path),
        "artifact_manifest": str(manifest_path),
        "semantic_gap_handoff": str(paths["handoff"]),
        "semantic_gap_handoff_manifest": str(paths["handoff_manifest"]),
        "alignment_fact_review_corpus": str(paths["fact_review_corpus"]),
        "alignment_fact_review_report": str(paths["fact_review_report"]),
        "alignment_fact_review_records": str(paths["fact_review_records"]),
        "alignment_fact_review_manifest": str(paths["fact_review_manifest"]),
        "alignment_fact_rule_review_report": str(paths["rule_review_report"]),
        "alignment_fact_rule_review_decisions": str(paths["rule_review_decisions"]),
        "alignment_fact_rule_review_records": str(paths["rule_review_records"]),
        "alignment_fact_rule_review_manifest": str(paths["rule_review_manifest"]),
        "alignment_fact_review_promotion_gate_report": str(paths["promotion_report"]),
        "approved_source_documents": str(paths["approved_source_documents"]),
        "promotion_gate_review_template": str(paths["promotion_template"]),
        "promotion_gate_records": str(paths["promotion_records"]),
        "promotion_gate_manifest": str(paths["promotion_manifest"]),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(payload: Mapping[str, Any] | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _nested_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    return _optional_int(_nested(payload, *keys))


def _optional_int(value: Any) -> int:
    try:
        return 0 if value is None else int(value)
    except (TypeError, ValueError):
        return 0


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
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
    parser.add_argument("--verified-records-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--record-indices-json", default=None)
    parser.add_argument("--workflow-report", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--mode", choices=MODES, default="false_negative_with_hits")
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--max-hits-per-target", type=int, default=DEFAULT_MAX_HITS_PER_TARGET)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--reviewer", default=RULE_REVIEWER)
    parser.add_argument("--reviewed-at", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run_retrieval_semantic_gap_review_workflow(
        verified_records_jsonl=args.verified_records_jsonl,
        output_dir=args.output_dir,
        record_indices_json=args.record_indices_json,
        workflow_report_path=args.workflow_report,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        mode=args.mode,
        min_hits=args.min_hits,
        max_targets=args.max_targets,
        max_hits_per_target=args.max_hits_per_target,
        min_confidence=args.min_confidence,
        reviewer=args.reviewer,
        reviewed_at=args.reviewed_at,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    print(
        f"{WORKFLOW}_ok "
        f"status={payload['status']} "
        f"candidates={payload['summary']['semantic_gap_candidate_count']} "
        f"approved_sources={payload['summary']['approved_source_document_count']}"
    )


if __name__ == "__main__":
    main()
