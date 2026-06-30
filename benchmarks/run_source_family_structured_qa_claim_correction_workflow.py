"""Run the source-family structured QA claim-correction loop.

This workflow is a thin orchestrator over three existing gates:

1. map product or blind-spot claims to exact source-family structured QA facts,
2. triage mapped and unmapped rows into next-action lanes,
3. emit a ProductTrace-visible correction handoff for mapped candidates only.
4. optionally enrich those correction traces with trace-level triple audits.

It does not collect new evidence, does not lower mapping thresholds, and does
not promote weak matches. The correction handoff still fails closed unless the
upstream covered-fact route summary was promoted.
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

import benchmarks.audit_source_family_structured_qa_claim_mapping as claim_mapping_workflow  # noqa: E402
import benchmarks.build_source_family_structured_qa_correction_handoff as correction_handoff_workflow  # noqa: E402
import benchmarks.enrich_product_trace_triple_audit as triple_audit_workflow  # noqa: E402
import benchmarks.triage_source_family_structured_qa_gaps as gap_triage_workflow  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "source_family_structured_qa_claim_correction_workflow"


def run_source_family_structured_qa_claim_correction_workflow(
    *,
    claims_path: str | Path,
    qa_corpus_path: str | Path,
    route_summary_path: str | Path,
    output_dir: str | Path,
    workflow_report_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    fact_expansion_plan_path: str | Path | None = None,
    fact_collection_corpus_path: str | Path | None = None,
    fact_collection_workflow_path: str | Path | None = None,
    subject_coverage_threshold: float = claim_mapping_workflow.DEFAULT_SUBJECT_COVERAGE_THRESHOLD,
    mapping_score_threshold: float = claim_mapping_workflow.DEFAULT_MAPPING_SCORE_THRESHOLD,
    answer_overlap_threshold: float = claim_mapping_workflow.DEFAULT_ANSWER_OVERLAP_THRESHOLD,
    weak_overlap_threshold: float = claim_mapping_workflow.DEFAULT_WEAK_OVERLAP_THRESHOLD,
    route_name: str = correction_handoff_workflow.DEFAULT_ROUTE_NAME,
    verifier_name: str = correction_handoff_workflow.DEFAULT_VERIFIER_NAME,
    enable_triple_audit: bool = False,
    triple_audit_output_dir: str | Path | None = None,
    triple_audit_report_path: str | Path | None = None,
    triple_audit_artifact_manifest_path: str | Path | None = None,
    triple_audit_min_slot_coverage: float = 1.0,
    triple_audit_min_audit_claim_coverage: float = 1.0,
    triple_audit_min_audit_pass_rate: float = 1.0,
    triple_audit_min_slot_coverage_rate: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Run claim mapping, gap triage, and correction handoff in sequence."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata_payload = {
        **dict(metadata or {}),
        "source_workflow": WORKFLOW,
    }

    mapping_dir = output / "claim-mapping"
    triage_dir = output / "gap-triage"
    handoff_dir = output / "correction-handoff"
    triple_audit_dir = Path(triple_audit_output_dir or output / "triple-audit")
    report_path = Path(workflow_report_path or output / "claim-correction-workflow.json")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    mapping_path = mapping_dir / "source-family-structured-qa-claim-mapping.json"
    mapping_manifest_path = mapping_dir / "artifact-manifest.json"
    triage_path = triage_dir / "gap-triage.json"
    triage_targets_path = triage_dir / "triage-targets.jsonl"
    triage_manifest_path = triage_dir / "artifact-manifest.json"
    handoff_report_path = handoff_dir / "source-family-structured-qa-correction-handoff.json"
    handoff_qa_path = handoff_dir / "source-family-structured-qa-correction-corpus.json"
    handoff_trace_path = handoff_dir / "product-traces.jsonl"
    handoff_actions_path = handoff_dir / "action-results.jsonl"
    handoff_manifest_path = handoff_dir / "artifact-manifest.json"
    triple_audit_path = Path(
        triple_audit_report_path
        or triple_audit_dir / "product-trace-triple-audit-enrichment.json"
    )
    triple_audit_manifest_path = Path(
        triple_audit_artifact_manifest_path
        or triple_audit_dir / "product-trace-triple-audit-artifact-manifest.json"
    )

    mapping = claim_mapping_workflow.run(
        claims_path=claims_path,
        qa_corpus_path=qa_corpus_path,
        route_summary_path=route_summary_path,
        output_path=mapping_path,
        artifact_manifest_path=mapping_manifest_path,
        subject_coverage_threshold=subject_coverage_threshold,
        mapping_score_threshold=mapping_score_threshold,
        answer_overlap_threshold=answer_overlap_threshold,
        weak_overlap_threshold=weak_overlap_threshold,
        metadata=metadata_payload,
        compact_json=compact_json,
    )
    triage = gap_triage_workflow.run(
        claim_mapping_path=mapping_path,
        output_dir=triage_dir,
        report_json_path=triage_path,
        target_jsonl_path=triage_targets_path,
        fact_expansion_plan_path=fact_expansion_plan_path,
        fact_collection_corpus_path=fact_collection_corpus_path,
        fact_collection_workflow_path=fact_collection_workflow_path,
        artifact_manifest_path=triage_manifest_path,
        metadata=metadata_payload,
        compact_json=compact_json,
    )
    handoff = correction_handoff_workflow.run(
        claim_mapping_path=mapping_path,
        output_dir=handoff_dir,
        report_json_path=handoff_report_path,
        qa_corpus_json_path=handoff_qa_path,
        trace_jsonl_path=handoff_trace_path,
        action_results_jsonl_path=handoff_actions_path,
        artifact_manifest_path=handoff_manifest_path,
        route_name=route_name,
        verifier_name=verifier_name,
        metadata=metadata_payload,
        compact_json=compact_json,
    )
    handoff_report = dict(handoff["report"])
    triple_audit_report: dict[str, Any] | None = None
    if enable_triple_audit and handoff_report.get("status") == "promote":
        triple_audit_report = triple_audit_workflow.build_product_trace_triple_audit_enrichment(
            triple_audit_workflow.ProductTraceTripleAuditEnrichmentConfig(
                trace_paths=(),
                trace_jsonl_paths=(handoff_trace_path,),
                output_dir=triple_audit_dir,
                report_path=triple_audit_path,
                artifact_manifest_path=triple_audit_manifest_path,
                min_slot_coverage=triple_audit_min_slot_coverage,
                min_audit_claim_coverage=triple_audit_min_audit_claim_coverage,
                min_audit_pass_rate=triple_audit_min_audit_pass_rate,
                min_slot_coverage_rate=triple_audit_min_slot_coverage_rate,
                compact_json=compact_json,
                metadata=metadata_payload,
            )
        )

    summary = _summary(
        mapping=mapping,
        triage=triage,
        handoff_report=handoff_report,
        triple_audit_report=triple_audit_report,
        enable_triple_audit=enable_triple_audit,
    )
    status = _status(
        summary=summary,
        handoff_report=handoff_report,
        triage=triage,
        triple_audit_report=triple_audit_report,
        enable_triple_audit=enable_triple_audit,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Thin source-family structured QA claim-correction workflow. It "
            "maps claims to promoted covered facts, triages remaining gaps, and "
            "emits ProductTrace correction handoffs only for mapped candidates."
        ),
        "source": {
            "claims": str(claims_path),
            "qa_corpus": str(qa_corpus_path),
            "route_summary": str(route_summary_path),
            "fact_expansion_plan": None
            if fact_expansion_plan_path is None
            else str(fact_expansion_plan_path),
            "fact_collection_corpus": None
            if fact_collection_corpus_path is None
            else str(fact_collection_corpus_path),
            "fact_collection_workflow": None
            if fact_collection_workflow_path is None
            else str(fact_collection_workflow_path),
        },
        "config": {
            "subject_coverage_threshold": float(subject_coverage_threshold),
            "mapping_score_threshold": float(mapping_score_threshold),
            "answer_overlap_threshold": float(answer_overlap_threshold),
            "weak_overlap_threshold": float(weak_overlap_threshold),
            "route_name": route_name,
            "verifier_name": verifier_name,
            "enable_triple_audit": bool(enable_triple_audit),
            "triple_audit_min_slot_coverage": float(triple_audit_min_slot_coverage),
            "triple_audit_min_audit_claim_coverage": float(
                triple_audit_min_audit_claim_coverage
            ),
            "triple_audit_min_audit_pass_rate": float(triple_audit_min_audit_pass_rate),
            "triple_audit_min_slot_coverage_rate": float(
                triple_audit_min_slot_coverage_rate
            ),
        },
        "label_usage": {
            "labels_used_for_mapping": False,
            "labels_copied_to_correction_handoff": False,
            "weak_matches_promoted": False,
            "workflow_outputs_are_broad_retrieval_evidence": False,
        },
        "paths": {
            "workflow_report": str(report_path),
            "artifact_manifest": str(manifest_path),
            "claim_mapping": str(mapping_path),
            "claim_mapping_manifest": str(mapping_manifest_path),
            "gap_triage": str(triage_path),
            "gap_triage_targets": str(triage_targets_path),
            "gap_triage_manifest": str(triage_manifest_path),
            "correction_handoff": str(handoff_report_path),
            "correction_handoff_manifest": str(handoff_manifest_path),
            "correction_qa_corpus": str(handoff_qa_path),
            "product_traces": str(handoff_trace_path),
            "action_results": str(handoff_actions_path),
            "triple_audit": str(triple_audit_path) if enable_triple_audit else None,
            "triple_audit_manifest": str(triple_audit_manifest_path)
            if enable_triple_audit
            else None,
            "triple_audit_traces_dir": str(triple_audit_dir / "traces")
            if enable_triple_audit
            else None,
        },
        "summary": summary,
        "child_statuses": {
            "claim_mapping": mapping.get("status"),
            "gap_triage": triage.get("status"),
            "correction_handoff": handoff_report.get("status"),
            "triple_audit": summary["triple_audit_status"],
        },
        "next_step": _next_step(
            summary=summary,
            workflow_status=status,
            handoff_status=handoff_report.get("status"),
        ),
        "metadata": dict(metadata or {}),
    }
    _write_json(report_path, payload, compact=compact_json)
    manifest = _write_manifest(
        manifest_path=manifest_path,
        payload=payload,
        claims_path=claims_path,
        qa_corpus_path=qa_corpus_path,
        route_summary_path=route_summary_path,
        fact_expansion_plan_path=fact_expansion_plan_path,
        fact_collection_corpus_path=fact_collection_corpus_path,
        fact_collection_workflow_path=fact_collection_workflow_path,
    )
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_path,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "mapped_qa_fact_candidate_count": summary["mapped_qa_fact_candidate_count"],
                "handoff_ready_count": summary["handoff_ready_count"],
                "correction_candidate_count": summary["correction_candidate_count"],
                "trace_count": summary["trace_count"],
                "triple_audit_status": summary["triple_audit_status"],
                "triple_audit_audit_pass_rate": summary["triple_audit_audit_pass_rate"],
                "artifact_manifest": str(manifest_path),
                "manifest_artifact_count": len(manifest["artifacts"]),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = run_source_family_structured_qa_claim_correction_workflow(
        claims_path=args.claims,
        qa_corpus_path=args.qa_corpus,
        route_summary_path=args.route_summary,
        output_dir=args.output_dir,
        workflow_report_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        fact_expansion_plan_path=args.fact_expansion_plan,
        fact_collection_corpus_path=args.fact_collection_corpus,
        fact_collection_workflow_path=args.fact_collection_workflow,
        subject_coverage_threshold=args.subject_coverage_threshold,
        mapping_score_threshold=args.mapping_score_threshold,
        answer_overlap_threshold=args.answer_overlap_threshold,
        weak_overlap_threshold=args.weak_overlap_threshold,
        route_name=args.route_name,
        verifier_name=args.verifier_name,
        enable_triple_audit=bool(args.enable_triple_audit),
        triple_audit_output_dir=args.triple_audit_output_dir,
        triple_audit_report_path=args.triple_audit_report,
        triple_audit_artifact_manifest_path=args.triple_audit_artifact_manifest,
        triple_audit_min_slot_coverage=args.triple_audit_min_slot_coverage,
        triple_audit_min_audit_claim_coverage=args.triple_audit_min_audit_claim_coverage,
        triple_audit_min_audit_pass_rate=args.triple_audit_min_audit_pass_rate,
        triple_audit_min_slot_coverage_rate=args.triple_audit_min_slot_coverage_rate,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "source_family_structured_qa_claim_correction_workflow_ok "
        f"status={payload['status']} "
        f"claims={summary['target_count']} "
        f"mapped={summary['mapped_qa_fact_candidate_count']} "
        f"traces={summary['trace_count']} "
        f"triple_audit={summary['triple_audit_status']} "
        f"blocked={summary['blocked_target_count']}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True)
    parser.add_argument("--qa-corpus", required=True)
    parser.add_argument("--route-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--fact-expansion-plan", default=None)
    parser.add_argument("--fact-collection-corpus", default=None)
    parser.add_argument("--fact-collection-workflow", default=None)
    parser.add_argument(
        "--subject-coverage-threshold",
        type=float,
        default=claim_mapping_workflow.DEFAULT_SUBJECT_COVERAGE_THRESHOLD,
    )
    parser.add_argument(
        "--mapping-score-threshold",
        type=float,
        default=claim_mapping_workflow.DEFAULT_MAPPING_SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--answer-overlap-threshold",
        type=float,
        default=claim_mapping_workflow.DEFAULT_ANSWER_OVERLAP_THRESHOLD,
    )
    parser.add_argument(
        "--weak-overlap-threshold",
        type=float,
        default=claim_mapping_workflow.DEFAULT_WEAK_OVERLAP_THRESHOLD,
    )
    parser.add_argument("--route-name", default=correction_handoff_workflow.DEFAULT_ROUTE_NAME)
    parser.add_argument("--verifier-name", default=correction_handoff_workflow.DEFAULT_VERIFIER_NAME)
    parser.add_argument("--enable-triple-audit", action="store_true")
    parser.add_argument("--triple-audit-output-dir", default=None)
    parser.add_argument("--triple-audit-report", default=None)
    parser.add_argument("--triple-audit-artifact-manifest", default=None)
    parser.add_argument("--triple-audit-min-slot-coverage", type=float, default=1.0)
    parser.add_argument("--triple-audit-min-audit-claim-coverage", type=float, default=1.0)
    parser.add_argument("--triple-audit-min-audit-pass-rate", type=float, default=1.0)
    parser.add_argument("--triple-audit-min-slot-coverage-rate", type=float, default=1.0)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    run(parser.parse_args(argv))


def _summary(
    *,
    mapping: Mapping[str, Any],
    triage: Mapping[str, Any],
    handoff_report: Mapping[str, Any],
    triple_audit_report: Mapping[str, Any] | None,
    enable_triple_audit: bool,
) -> dict[str, Any]:
    mapping_summary = _mapping(mapping.get("summary"))
    triage_summary = _mapping(triage.get("summary"))
    handoff_summary = _mapping(handoff_report.get("summary"))
    triple_summary = _mapping(_mapping(triple_audit_report).get("summary"))
    return {
        "target_count": _int(mapping_summary.get("target_count")),
        "covered_fact_match_count": _int(mapping_summary.get("covered_fact_match_count")),
        "mapped_qa_fact_candidate_count": _int(mapping_summary.get("mapped_qa_fact_candidate_count")),
        "answer_value_supported_count": _int(mapping_summary.get("answer_value_supported_count")),
        "no_candidate_fact_count": _int(mapping_summary.get("no_candidate_fact_count")),
        "handoff_ready_count": _int(triage_summary.get("handoff_ready_count")),
        "audit_only_count": _int(triage_summary.get("audit_only_count")),
        "needs_collection_count": _int(triage_summary.get("needs_collection_count")),
        "blocked_target_count": _int(triage_summary.get("blocked_target_count")),
        "correction_candidate_count": _int(handoff_summary.get("correction_candidate_count")),
        "corpus_document_count": _int(handoff_summary.get("corpus_document_count")),
        "trace_count": _int(handoff_summary.get("trace_count")),
        "action_result_count": _int(handoff_summary.get("action_result_count")),
        "mapping_status": mapping.get("status"),
        "triage_status": triage.get("status"),
        "correction_handoff_status": handoff_report.get("status"),
        "triple_audit_status": _triple_audit_status(
            enable_triple_audit=enable_triple_audit,
            handoff_status=handoff_report.get("status"),
            triple_audit_report=triple_audit_report,
        ),
        "triple_audit_trace_count": _int(triple_summary.get("trace_count")),
        "triple_audit_claim_triple_coverage_rate": _float_or_none(
            triple_summary.get("claim_triple_coverage_rate")
        ),
        "triple_audit_audit_claim_coverage_rate": _float_or_none(
            triple_summary.get("audit_claim_coverage_rate")
        ),
        "triple_audit_audit_pass_rate": _float_or_none(triple_summary.get("audit_pass_rate")),
        "triple_audit_slot_coverage_rate": _float_or_none(
            triple_summary.get("slot_coverage_rate")
        ),
    }


def _status(
    *,
    summary: Mapping[str, Any],
    handoff_report: Mapping[str, Any],
    triage: Mapping[str, Any],
    triple_audit_report: Mapping[str, Any] | None,
    enable_triple_audit: bool,
) -> str:
    if enable_triple_audit and handoff_report.get("status") == "promote":
        if _mapping(triple_audit_report).get("status") != "promote":
            return "blocked"
    if handoff_report.get("status") == "promote":
        return "promote"
    if _mapping(handoff_report.get("source")).get("route_summary_promoted") is not True:
        return "blocked"
    if _int(summary.get("handoff_ready_count")) > 0:
        return "blocked"
    triage_status = str(triage.get("status") or "")
    if triage_status in {"needs_collection", "audit_only", "empty"}:
        return triage_status
    return "blocked"


def _next_step(
    *,
    summary: Mapping[str, Any],
    workflow_status: Any,
    handoff_status: Any,
) -> str:
    if workflow_status == "promote":
        return (
            "Use the correction handoff traces as target-specific ProductTrace "
            "evidence; do not treat the QA corpus as broad retrieval coverage."
        )
    if handoff_status == "promote" and summary.get("triple_audit_status") == "blocked":
        return (
            "Inspect the triple-audit report before using correction handoff traces; "
            "the optional audit gate did not promote."
        )
    if _int(summary.get("needs_collection_count")) > 0:
        return (
            "Run fact-expansion, source-family collection, citation, or "
            "world-model rule lanes for the blocked rows before retrying "
            "correction handoff."
        )
    if _int(summary.get("audit_only_count")) > 0:
        return "Review answer-support audit rows; they are not correction handoff candidates."
    return "Inspect route promotion and mapping thresholds; no safe correction handoff was produced."


def _write_manifest(
    *,
    manifest_path: Path,
    payload: Mapping[str, Any],
    claims_path: str | Path,
    qa_corpus_path: str | Path,
    route_summary_path: str | Path,
    fact_expansion_plan_path: str | Path | None,
    fact_collection_corpus_path: str | Path | None,
    fact_collection_workflow_path: str | Path | None,
) -> dict[str, Any]:
    paths = _mapping(payload.get("paths"))
    artifacts: dict[str, Path] = {
        "source_family_structured_qa_claim_correction_workflow": Path(paths["workflow_report"]),
        "claim_mapping": Path(paths["claim_mapping"]),
        "claim_mapping_manifest": Path(paths["claim_mapping_manifest"]),
        "gap_triage": Path(paths["gap_triage"]),
        "gap_triage_targets": Path(paths["gap_triage_targets"]),
        "gap_triage_manifest": Path(paths["gap_triage_manifest"]),
        "correction_handoff": Path(paths["correction_handoff"]),
        "correction_handoff_manifest": Path(paths["correction_handoff_manifest"]),
        "correction_qa_corpus": Path(paths["correction_qa_corpus"]),
        "product_traces": Path(paths["product_traces"]),
        "action_results": Path(paths["action_results"]),
        "claims": Path(claims_path),
        "qa_corpus": Path(qa_corpus_path),
        "route_summary": Path(route_summary_path),
    }
    if fact_expansion_plan_path is not None:
        artifacts["fact_expansion_plan"] = Path(fact_expansion_plan_path)
    if fact_collection_corpus_path is not None:
        artifacts["fact_collection_corpus"] = Path(fact_collection_corpus_path)
    if fact_collection_workflow_path is not None:
        artifacts["fact_collection_workflow"] = Path(fact_collection_workflow_path)
    for artifact_name, path_key in (
        ("triple_audit", "triple_audit"),
        ("triple_audit_manifest", "triple_audit_manifest"),
        ("triple_audit_traces_dir", "triple_audit_traces_dir"),
    ):
        raw_path = paths.get(path_key)
        if raw_path:
            path = Path(str(raw_path))
            if path.exists():
                artifacts[artifact_name] = path
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "mapped_qa_fact_candidate_count": _int(
                _mapping(payload.get("summary")).get("mapped_qa_fact_candidate_count")
            ),
            "correction_candidate_count": _int(
                _mapping(payload.get("summary")).get("correction_candidate_count")
            ),
            "trace_count": _int(_mapping(payload.get("summary")).get("trace_count")),
            "triple_audit_status": _mapping(payload.get("summary")).get(
                "triple_audit_status"
            ),
            "triple_audit_audit_pass_rate": _mapping(payload.get("summary")).get(
                "triple_audit_audit_pass_rate"
            ),
            **dict(_mapping(payload.get("metadata"))),
        },
    )
    _write_json(manifest_path, manifest, compact=False)
    return manifest


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":"))
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True)
    output.write_text(text + "\n", encoding="utf-8")


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata entries must be key=value, got {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"metadata key cannot be empty in {value!r}")
        metadata[key] = raw.strip()
    return metadata


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _triple_audit_status(
    *,
    enable_triple_audit: bool,
    handoff_status: Any,
    triple_audit_report: Mapping[str, Any] | None,
) -> str:
    if not enable_triple_audit:
        return "not_run"
    if handoff_status != "promote":
        return "skipped"
    return str(_mapping(triple_audit_report).get("status") or "blocked")


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
