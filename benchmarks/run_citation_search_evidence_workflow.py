"""Run citation/search adapter results through evidence gates.

This workflow starts where ``build_citation_search_adapter_handoff.py`` stops.
It consumes local JSONL results from an external citation/search adapter,
normalizes them into an external retrieval corpus, audits corpus provenance,
runs the blind-spot retrieval query sweep, and can compare that sweep against
controlled baselines before any release policy treats the result as evidence.

The command does not call the network. Search systems must materialize their
results first, keyed by the sanitized request ids emitted by the handoff.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.audit_citation_search_result_bindings import (  # noqa: E402
    DEFAULT_CORPUS_NAME as DEFAULT_BOUND_CORPUS_NAME,
)
from benchmarks.audit_citation_search_result_bindings import (  # noqa: E402
    DEFAULT_SOURCE_KIND as DEFAULT_BOUND_SOURCE_KIND,
)
from benchmarks.audit_citation_search_result_bindings import (  # noqa: E402
    run as run_citation_binding_audit,
)
from benchmarks.audit_retrieval_corpus_provenance import (  # noqa: E402
    audit_retrieval_corpus_provenance,
)
from benchmarks.build_citation_search_adapter_handoff import (  # noqa: E402
    DEFAULT_CORPUS_NAME,
    DEFAULT_MAX_ALTERNATE_QUERIES,
    DEFAULT_SOURCE_KIND,
    QUERY_MODES,
)
from benchmarks.build_citation_search_adapter_handoff import (  # noqa: E402
    run as run_citation_search_handoff,
)
from benchmarks.compare_blind_spot_query_sweeps import run as run_query_sweep_comparison  # noqa: E402
from benchmarks.summarize_citation_binding_audit_failures import (  # noqa: E402
    run as run_binding_failure_review,
)
from benchmarks.sweep_blind_spot_retrieval_queries import (  # noqa: E402
    DEFAULT_MIN_OVERLAPS,
    DEFAULT_SOURCE_FAMILY_FILTERS,
    DEFAULT_TARGET_ROUTE,
    QUERY_FIELDS,
)
from benchmarks.sweep_blind_spot_retrieval_queries import (  # noqa: E402
    run as run_query_sweep,
)
from eigentruth.eval.score_dump import load_score_dump  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "citation_search_evidence_workflow"


def run(
    *,
    queue_report_path: str | Path,
    adapter_results_path: str | Path,
    scores_path: str | Path,
    blind_spots_path: str | Path,
    output_dir: str | Path,
    controlled_sweep_paths: Sequence[str | Path] = (),
    workflow_report_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    batch_ids: Sequence[str] = (),
    query_mode: str = "question",
    max_requests: int | None = None,
    max_results_per_request: int | None = None,
    max_alternate_queries: int = DEFAULT_MAX_ALTERNATE_QUERIES,
    corpus_name: str = DEFAULT_CORPUS_NAME,
    source_kind: str = DEFAULT_SOURCE_KIND,
    query_fields: Sequence[str] = ("question", "question_answer"),
    retriever_min_overlaps: Sequence[float] = DEFAULT_MIN_OVERLAPS,
    source_family_filters: Sequence[str] = DEFAULT_SOURCE_FAMILY_FILTERS,
    query_sweep_verified_records_dir: str | Path | None = None,
    retrieval_limit: int = 3,
    signal: str = "truth_proj",
    alpha: float = 0.10,
    repeats: int = 1,
    seed: int = 0,
    verifier_min_overlap: float = 0.65,
    target_route: str = DEFAULT_TARGET_ROUTE,
    max_verified_false_alarm: float = 0.05,
    min_blind_refuted_rate: float = 0.50,
    min_controlled_blind_refuted_rate: float = 0.50,
    min_external_blind_refuted_rate: float = 0.50,
    max_controlled_verified_false_alarm: float = 0.05,
    max_external_verified_false_alarm: float = 0.05,
    min_adapter_request_coverage: float = 1.0,
    max_exact_answer_copy_rate: float = 0.80,
    max_claim_id_link_rate: float = 0.0,
    max_label_metadata_rate: float = 0.0,
    audit_source_bindings: bool = False,
    require_binding_source_family_match: bool = False,
    clip_binding_evidence_spans: bool = False,
    binding_min_keyword_overlap: float = 0.2,
    binding_min_support_keyword_overlap: float = 0.65,
    binding_min_entity_recall: float = 0.5,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Run the handoff-to-evidence gate workflow."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if not (0.0 <= float(min_adapter_request_coverage) <= 1.0):
        raise ValueError("min_adapter_request_coverage must be between 0 and 1.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = _paths(output)
    report_path = Path(workflow_report_path) if workflow_report_path is not None else paths["workflow_report"]
    manifest_path = Path(artifact_manifest_path) if artifact_manifest_path is not None else paths["artifact_manifest"]

    handoff = run_citation_search_handoff(
        queue_report_path=queue_report_path,
        adapter_results_path=adapter_results_path,
        output_dir=output,
        report_json_path=paths["handoff_report"],
        request_jsonl_path=paths["adapter_requests"],
        source_jsonl_path=paths["source_documents"],
        corpus_json_path=paths["corpus"],
        artifact_manifest_path=paths["handoff_manifest"],
        batch_ids=batch_ids,
        query_mode=query_mode,
        max_requests=max_requests,
        max_results_per_request=max_results_per_request,
        max_alternate_queries=max_alternate_queries,
        corpus_name=corpus_name,
        source_kind=source_kind,
        metadata=metadata,
        compact_json=compact_json,
    )

    provenance_report: dict[str, Any] | None = None
    query_sweep_report: dict[str, Any] | None = None
    comparison_report: dict[str, Any] | None = None
    binding_audit_report: dict[str, Any] | None = None
    binding_failure_review: dict[str, Any] | None = None
    corpus_path = paths["corpus"] if handoff["external_retrieval_corpus"] is not None else None

    if corpus_path is not None and audit_source_bindings:
        binding_audit_report = run_citation_binding_audit(
            requests_path=paths["adapter_requests"],
            source_documents_path=paths["source_documents"],
            report_json_path=paths["binding_audit"],
            bound_source_documents_path=paths["bound_source_documents"],
            bound_corpus_json_path=paths["bound_corpus"],
            artifact_manifest_path=paths["binding_manifest"],
            corpus_name=DEFAULT_BOUND_CORPUS_NAME,
            source_kind=DEFAULT_BOUND_SOURCE_KIND,
            min_keyword_overlap=float(binding_min_keyword_overlap),
            min_support_keyword_overlap=float(binding_min_support_keyword_overlap),
            min_entity_recall=float(binding_min_entity_recall),
            require_source_family_match=bool(require_binding_source_family_match),
            clip_accepted_evidence_spans=bool(clip_binding_evidence_spans),
            metadata={**dict(metadata or {}), "source_workflow": WORKFLOW},
            compact_json=compact_json,
        )
        binding_failure_review = run_binding_failure_review(
            binding_audit_paths=(paths["binding_audit"],),
            output_path=paths["binding_failure_review"],
            max_examples_per_issue=3,
            metadata={**dict(metadata or {}), "source_workflow": WORKFLOW},
            compact_json=compact_json,
        )
        if _nested_int(binding_audit_report, "summary", "accepted_source_document_count"):
            corpus_path = paths["bound_corpus"]
        else:
            corpus_path = None

    if corpus_path is not None:
        score_dump = load_score_dump(scores_path, allow_missing_scores=True, require_statements=True)
        provenance_report = audit_retrieval_corpus_provenance(
            score_dump,
            scores_path=scores_path,
            corpus_paths=(corpus_path,),
            audit_role="grounding",
            max_exact_answer_copy_rate=max_exact_answer_copy_rate,
            max_claim_id_link_rate=max_claim_id_link_rate,
            max_label_metadata_rate=max_label_metadata_rate,
        )
        _write_json(paths["provenance_audit"], provenance_report, compact=compact_json)

        query_sweep_report = run_query_sweep(
            scores_path=scores_path,
            corpus_paths=(corpus_path,),
            blind_spots_path=blind_spots_path,
            output_path=paths["query_sweep"],
            source_binding_queue_path=queue_report_path,
            query_fields=query_fields,
            retriever_min_overlaps=retriever_min_overlaps,
            source_family_filters=source_family_filters,
            verified_records_dir=query_sweep_verified_records_dir,
            retrieval_limit=retrieval_limit,
            signal=signal,
            alpha=alpha,
            repeats=repeats,
            seed=seed,
            verifier_min_overlap=verifier_min_overlap,
            target_route=target_route,
            max_verified_false_alarm=max_verified_false_alarm,
            min_blind_refuted_rate=min_blind_refuted_rate,
            artifact_manifest_path=paths["query_sweep_manifest"],
            metadata={**dict(metadata or {}), "source_workflow": WORKFLOW},
            compact_json=compact_json,
        )

        if controlled_sweep_paths:
            comparison_report = run_query_sweep_comparison(
                controlled_sweep_paths=controlled_sweep_paths,
                external_sweep_paths=(paths["query_sweep"],),
                output_path=paths["query_sweep_comparison"],
                min_controlled_blind_refuted_rate=min_controlled_blind_refuted_rate,
                min_external_blind_refuted_rate=min_external_blind_refuted_rate,
                max_controlled_verified_false_alarm=max_controlled_verified_false_alarm,
                max_external_verified_false_alarm=max_external_verified_false_alarm,
                artifact_manifest_path=paths["query_sweep_comparison_manifest"],
                metadata={**dict(metadata or {}), "source_workflow": WORKFLOW},
                compact_json=compact_json,
            )

    gate = _gate(
        handoff=handoff,
        binding_audit_report=binding_audit_report,
        provenance_report=provenance_report,
        query_sweep_report=query_sweep_report,
        comparison_report=comparison_report,
        controlled_sweep_paths=controlled_sweep_paths,
        min_adapter_request_coverage=float(min_adapter_request_coverage),
    )
    report = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": gate["status"],
        "gate": gate,
        "source": {
            "queue_report": str(queue_report_path),
            "adapter_results": str(adapter_results_path),
            "scores": str(scores_path),
            "blind_spots": str(blind_spots_path),
            "source_binding_queue": str(queue_report_path),
            "controlled_sweeps": tuple(str(path) for path in controlled_sweep_paths),
        },
        "config": {
            "batch_ids": tuple(str(item) for item in batch_ids),
            "query_mode": query_mode,
            "max_requests": max_requests,
            "max_results_per_request": max_results_per_request,
            "max_alternate_queries": int(max_alternate_queries),
            "corpus_name": corpus_name,
            "source_kind": source_kind,
            "query_fields": tuple(query_fields),
            "retriever_min_overlaps": tuple(float(value) for value in retriever_min_overlaps),
            "source_family_filters": tuple(str(value) for value in source_family_filters),
            "query_sweep_verified_records_dir": (
                None if query_sweep_verified_records_dir is None else str(query_sweep_verified_records_dir)
            ),
            "retrieval_limit": int(retrieval_limit),
            "signal": signal,
            "alpha": float(alpha),
            "repeats": int(repeats),
            "seed": int(seed),
            "verifier_min_overlap": float(verifier_min_overlap),
            "target_route": target_route,
            "max_verified_false_alarm": float(max_verified_false_alarm),
            "min_blind_refuted_rate": float(min_blind_refuted_rate),
            "min_controlled_blind_refuted_rate": float(min_controlled_blind_refuted_rate),
            "min_external_blind_refuted_rate": float(min_external_blind_refuted_rate),
            "max_controlled_verified_false_alarm": float(max_controlled_verified_false_alarm),
            "max_external_verified_false_alarm": float(max_external_verified_false_alarm),
            "min_adapter_request_coverage": float(min_adapter_request_coverage),
            "max_exact_answer_copy_rate": float(max_exact_answer_copy_rate),
            "max_claim_id_link_rate": float(max_claim_id_link_rate),
            "max_label_metadata_rate": float(max_label_metadata_rate),
            "audit_source_bindings": bool(audit_source_bindings),
            "require_binding_source_family_match": bool(require_binding_source_family_match),
            "clip_binding_evidence_spans": bool(clip_binding_evidence_spans),
            "binding_min_keyword_overlap": float(binding_min_keyword_overlap),
            "binding_min_support_keyword_overlap": float(binding_min_support_keyword_overlap),
            "binding_min_entity_recall": float(binding_min_entity_recall),
        },
        "summary": _summary(
            handoff=handoff,
            binding_audit_report=binding_audit_report,
            binding_failure_review=binding_failure_review,
            provenance_report=provenance_report,
            query_sweep_report=query_sweep_report,
            comparison_report=comparison_report,
        ),
        "paths": _report_paths(
            paths=paths,
            report_path=report_path,
            manifest_path=manifest_path,
            query_sweep_verified_records_dir=query_sweep_verified_records_dir,
            active_corpus_path=corpus_path,
            has_corpus=corpus_path is not None,
            has_binding_audit=binding_audit_report is not None,
            has_binding_failure_review=binding_failure_review is not None,
            has_provenance=provenance_report is not None,
            has_query_sweep=query_sweep_report is not None,
            has_comparison=comparison_report is not None,
        ),
        "metadata": dict(metadata or {}),
    }
    _write_json(report_path, report, compact=compact_json)

    manifest = build_artifact_manifest(
        _manifest_artifacts(
            paths=paths,
            report_path=report_path,
            queue_report_path=queue_report_path,
            adapter_results_path=adapter_results_path,
            scores_path=scores_path,
            blind_spots_path=blind_spots_path,
            controlled_sweep_paths=controlled_sweep_paths,
            query_sweep_verified_records_dir=query_sweep_verified_records_dir,
            active_corpus_path=corpus_path,
            has_corpus=corpus_path is not None,
            has_binding_audit=binding_audit_report is not None,
            has_binding_failure_review=binding_failure_review is not None,
            has_provenance=provenance_report is not None,
            has_query_sweep=query_sweep_report is not None,
            has_comparison=comparison_report is not None,
        ),
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": report["status"],
            "passed": gate["passed"],
            "promotion_ready": gate["promotion_ready"],
            "target_route": target_route,
            "blocking_reason_count": len(gate["blocking_reasons"]),
            "selected_batch_count": _nested_int(handoff, "summary", "selected_batch_count"),
            "selected_batch_ids": _nested(handoff, "summary", "selected_batch_ids"),
            "adapter_request_count": _nested_int(handoff, "summary", "adapter_request_count"),
            "adapter_result_request_coverage": _nested(
                handoff,
                "summary",
                "adapter_result_request_coverage",
            ),
            "adapter_result_missing_request_count": _nested_int(
                handoff,
                "summary",
                "adapter_result_missing_request_count",
            ),
            "source_document_count": _nested_int(handoff, "summary", "source_document_count"),
            "corpus_document_count": _nested_int(handoff, "summary", "corpus_document_count"),
            "binding_audit_status": None if binding_audit_report is None else binding_audit_report.get("status"),
            "bound_source_document_count": _nested_int(
                binding_audit_report,
                "summary",
                "accepted_source_document_count",
            ),
            "bound_request_count": _nested_int(
                binding_audit_report,
                "summary",
                "accepted_request_count",
            ),
            "query_sweep_failure_reason_counts": _nested(
                report,
                "summary",
                "query_sweep_failure_reason_counts",
            ),
            "binding_failure_dominant_issue": _nested(
                report,
                "summary",
                "binding_failure_dominant_issue",
            ),
            "binding_failure_dominant_recommendation": _nested(
                report,
                "summary",
                "binding_failure_dominant_recommendation",
            ),
            "query_sweep_no_hit_strategy_count": _nested_int(
                report,
                "summary",
                "query_sweep_no_hit_strategy_count",
            ),
            "query_sweep_target_route_not_selected_strategy_count": _nested_int(
                report,
                "summary",
                "query_sweep_target_route_not_selected_strategy_count",
            ),
            "query_sweep_best_observed_strategy": _nested(
                report,
                "summary",
                "query_sweep_best_observed_strategy",
            ),
            "query_sweep_best_observed_failure_reasons": _nested(
                report,
                "summary",
                "query_sweep_best_observed_failure_reasons",
            ),
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
                "status": report["status"],
                "passed": gate["passed"],
                "promotion_ready": gate["promotion_ready"],
                "target_route": target_route,
                "selected_batch_count": _nested_int(handoff, "summary", "selected_batch_count"),
                "selected_batch_ids": _nested(handoff, "summary", "selected_batch_ids"),
                "adapter_request_count": _nested_int(handoff, "summary", "adapter_request_count"),
                "adapter_result_request_coverage": _nested(
                    handoff,
                    "summary",
                    "adapter_result_request_coverage",
                ),
                "adapter_result_missing_request_count": _nested_int(
                    handoff,
                    "summary",
                    "adapter_result_missing_request_count",
                ),
                "source_document_count": _nested_int(handoff, "summary", "source_document_count"),
                "corpus_document_count": _nested_int(handoff, "summary", "corpus_document_count"),
                "binding_audit_status": None if binding_audit_report is None else binding_audit_report.get("status"),
                "bound_source_document_count": _nested_int(
                    binding_audit_report,
                    "summary",
                    "accepted_source_document_count",
                ),
                "bound_request_count": _nested_int(
                    binding_audit_report,
                    "summary",
                    "accepted_request_count",
                ),
                "query_sweep_failure_reason_counts": _nested(
                    report,
                    "summary",
                    "query_sweep_failure_reason_counts",
                ),
                "binding_failure_dominant_issue": _nested(
                    report,
                    "summary",
                    "binding_failure_dominant_issue",
                ),
                "binding_failure_dominant_recommendation": _nested(
                    report,
                    "summary",
                    "binding_failure_dominant_recommendation",
                ),
                "query_sweep_no_hit_strategy_count": _nested_int(
                    report,
                    "summary",
                    "query_sweep_no_hit_strategy_count",
                ),
                "query_sweep_target_route_not_selected_strategy_count": _nested_int(
                    report,
                    "summary",
                    "query_sweep_target_route_not_selected_strategy_count",
                ),
                "query_sweep_best_observed_strategy": _nested(
                    report,
                    "summary",
                    "query_sweep_best_observed_strategy",
                ),
                "query_sweep_best_observed_failure_reasons": _nested(
                    report,
                    "summary",
                    "query_sweep_best_observed_failure_reasons",
                ),
                "provenance_status": None if provenance_report is None else provenance_report.get("status"),
                "best_passing_strategy": _nested(query_sweep_report, "summary", "best_passing_strategy"),
                "comparison_status": None if comparison_report is None else comparison_report.get("status"),
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return report


def _paths(output: Path) -> dict[str, Path]:
    return {
        "adapter_requests": output / "citation-search-adapter-requests.jsonl",
        "artifact_manifest": output / "artifact-manifest.json",
        "binding_audit": output / "citation-search-binding-audit.json",
        "binding_failure_review": output / "citation-search-binding-failure-review.json",
        "binding_manifest": output / "citation-search-binding-audit-manifest.json",
        "bound_corpus": output / "citation-search-bound-corpus.json",
        "bound_source_documents": output / "citation-search-bound-source-docs.jsonl",
        "corpus": output / "citation-search-corpus.json",
        "handoff_manifest": output / "citation-search-handoff-manifest.json",
        "handoff_report": output / "citation-search-handoff.json",
        "provenance_audit": output / "citation-search-provenance-audit.json",
        "query_sweep": output / "citation-search-query-sweep.json",
        "query_sweep_comparison": output / "citation-search-query-sweep-comparison.json",
        "query_sweep_comparison_manifest": output / "citation-search-query-sweep-comparison-manifest.json",
        "query_sweep_manifest": output / "citation-search-query-sweep-manifest.json",
        "query_sweep_verified_records": output / "citation-search-query-sweep-verified-records",
        "source_documents": output / "citation-search-source-docs.jsonl",
        "workflow_report": output / "citation-search-evidence-workflow.json",
    }


def _gate(
    *,
    handoff: Mapping[str, Any],
    binding_audit_report: Mapping[str, Any] | None,
    provenance_report: Mapping[str, Any] | None,
    query_sweep_report: Mapping[str, Any] | None,
    comparison_report: Mapping[str, Any] | None,
    controlled_sweep_paths: Sequence[str | Path],
    min_adapter_request_coverage: float,
) -> dict[str, Any]:
    blocking: list[dict[str, Any]] = []
    request_coverage = _optional_float(_nested(handoff, "summary", "adapter_result_request_coverage"))
    if request_coverage is None:
        request_coverage = 0.0
    if request_coverage < min_adapter_request_coverage:
        blocking.append({
            "gate": "adapter_request_coverage",
            "reason": (
                "Adapter results covered "
                f"{request_coverage:.3f} of selected citation/search requests, "
                f"below required {min_adapter_request_coverage:.3f}."
            ),
            "missing_request_count": _nested_int(
                handoff,
                "summary",
                "adapter_result_missing_request_count",
            ),
            "missing_request_ids": _nested(
                handoff,
                "summary",
                "adapter_result_missing_request_ids",
            ),
        })
    if _nested_int(handoff, "summary", "source_document_count") == 0:
        blocking.append({
            "gate": "adapter_results",
            "reason": "No source documents were produced from adapter results.",
        })
    if binding_audit_report is not None and _nested_int(
        binding_audit_report,
        "summary",
        "accepted_source_document_count",
    ) == 0:
        blocking.append({
            "gate": "source_binding_audit",
            "reason": "No citation/search source documents passed claim-specific binding audit.",
            "issue_counts": _nested(binding_audit_report, "summary", "issue_counts"),
        })
    if provenance_report is not None and not bool(provenance_report.get("passed")):
        blocking.append({
            "gate": "provenance",
            "reason": f"Provenance audit status is {provenance_report.get('status')}.",
        })
    if query_sweep_report is not None and not _nested(query_sweep_report, "summary", "best_passing_strategy"):
        blocking.append({
            "gate": "query_sweep",
            "reason": "No blind-spot query strategy passed the configured gates.",
        })
    if comparison_report is not None and not bool(_nested(comparison_report, "decision", "passed")):
        blocking.append({
            "gate": "query_sweep_comparison",
            "reason": f"Controlled-vs-external comparison status is {comparison_report.get('status')}.",
        })
    comparison_required = bool(controlled_sweep_paths)
    passed = not blocking and provenance_report is not None and query_sweep_report is not None
    promotion_ready = bool(
        passed
        and comparison_required
        and comparison_report
        and comparison_report["decision"]["passed"]
    )
    if blocking:
        status = "blocked"
    elif promotion_ready:
        status = "promote"
    elif passed:
        status = "complete"
    else:
        status = "blocked"
    return {
        "status": status,
        "passed": bool(passed),
        "promotion_ready": promotion_ready,
        "comparison_required": comparison_required,
        "blocking_reasons": blocking,
    }


def _summary(
    *,
    handoff: Mapping[str, Any],
    binding_audit_report: Mapping[str, Any] | None,
    binding_failure_review: Mapping[str, Any] | None,
    provenance_report: Mapping[str, Any] | None,
    query_sweep_report: Mapping[str, Any] | None,
    comparison_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    query_sweep_diagnostics = _query_sweep_diagnostics(query_sweep_report)
    return {
        "selected_batch_count": _nested_int(handoff, "summary", "selected_batch_count"),
        "selected_batch_ids": _nested(handoff, "summary", "selected_batch_ids"),
        "selected_batch_source_request_count": _nested_int(
            handoff,
            "summary",
            "selected_batch_source_request_count",
        ),
        "adapter_request_count": _nested_int(handoff, "summary", "adapter_request_count"),
        "adapter_result_expected_request_count": _nested_int(
            handoff,
            "summary",
            "adapter_result_expected_request_count",
        ),
        "adapter_result_matched_request_count": _nested_int(
            handoff,
            "summary",
            "adapter_result_matched_request_count",
        ),
        "adapter_result_missing_request_count": _nested_int(
            handoff,
            "summary",
            "adapter_result_missing_request_count",
        ),
        "adapter_result_missing_request_ids": _nested(
            handoff,
            "summary",
            "adapter_result_missing_request_ids",
        ),
        "adapter_result_request_coverage": _nested(
            handoff,
            "summary",
            "adapter_result_request_coverage",
        ),
        "adapter_result_unknown_request_count": _nested_int(
            handoff,
            "summary",
            "adapter_result_unknown_request_count",
        ),
        "source_document_count": _nested_int(handoff, "summary", "source_document_count"),
        "corpus_document_count": _nested_int(handoff, "summary", "corpus_document_count"),
        "binding_audit_status": None if binding_audit_report is None else binding_audit_report.get("status"),
        "binding_audit_passed": None if binding_audit_report is None else bool(binding_audit_report.get("passed")),
        "bound_source_document_count": _nested_int(
            binding_audit_report,
            "summary",
            "accepted_source_document_count",
        ),
        "bound_request_count": _nested_int(binding_audit_report, "summary", "accepted_request_count"),
        "binding_issue_counts": _nested(binding_audit_report, "summary", "issue_counts"),
        "binding_failure_review_status": (
            None if binding_failure_review is None else binding_failure_review.get("status")
        ),
        "binding_failure_dominant_issue": _nested(binding_failure_review, "summary", "dominant_issue"),
        "binding_failure_dominant_recommendation": _nested(
            binding_failure_review,
            "summary",
            "dominant_recommendation",
        ),
        "binding_failure_recommendation_counts": _nested(
            binding_failure_review,
            "summary",
            "recommendation_counts",
        ),
        "provenance_status": None if provenance_report is None else provenance_report.get("status"),
        "provenance_passed": None if provenance_report is None else bool(provenance_report.get("passed")),
        "evidence_class": None if provenance_report is None else provenance_report.get("evidence_class"),
        "query_sweep_best_strategy": _nested(query_sweep_report, "summary", "best_strategy"),
        "query_sweep_best_passing_strategy": _nested(query_sweep_report, "summary", "best_passing_strategy"),
        "query_sweep_best_passing_blind_refuted_count": _nested(
            query_sweep_report,
            "summary",
            "best_passing_blind_refuted_count",
        ),
        "query_sweep_failure_reason_counts": query_sweep_diagnostics[
            "failure_reason_counts"
        ],
        "query_sweep_no_hit_strategy_count": query_sweep_diagnostics[
            "no_hit_strategy_count"
        ],
        "query_sweep_target_route_not_selected_strategy_count": (
            query_sweep_diagnostics["target_route_not_selected_strategy_count"]
        ),
        "query_sweep_blind_refuted_rate_below_min_strategy_count": (
            query_sweep_diagnostics["blind_refuted_rate_below_min_strategy_count"]
        ),
        "query_sweep_verified_false_alarm_above_max_strategy_count": (
            query_sweep_diagnostics["verified_false_alarm_above_max_strategy_count"]
        ),
        "query_sweep_best_observed_strategy": query_sweep_diagnostics[
            "best_observed_strategy"
        ],
        "query_sweep_best_observed_blind_refuted_rate": query_sweep_diagnostics[
            "best_observed_blind_refuted_rate"
        ],
        "query_sweep_best_observed_verified_false_alarm": query_sweep_diagnostics[
            "best_observed_verified_false_alarm"
        ],
        "query_sweep_best_observed_records_with_hits": query_sweep_diagnostics[
            "best_observed_records_with_hits"
        ],
        "query_sweep_best_observed_total_hits": query_sweep_diagnostics[
            "best_observed_total_hits"
        ],
        "query_sweep_best_observed_failure_reasons": query_sweep_diagnostics[
            "best_observed_failure_reasons"
        ],
        "query_sweep_recommended_next_actions": query_sweep_diagnostics[
            "recommended_next_actions"
        ],
        "comparison_status": None if comparison_report is None else comparison_report.get("status"),
        "comparison_passed": (
            None
            if comparison_report is None
            else bool(_nested(comparison_report, "decision", "passed"))
        ),
        "recommended_external_strategy": _nested(comparison_report, "decision", "recommended_external_strategy"),
    }


def _query_sweep_diagnostics(query_sweep_report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Summarize why citation/search query strategies did not pass gates."""
    strategies = _mapping_sequence(_nested(query_sweep_report, "strategies"))
    failure_counts: Counter[str] = Counter()
    no_hit_strategy_count = 0
    target_route_not_selected_strategy_count = 0
    blind_refuted_rate_below_min_strategy_count = 0
    verified_false_alarm_above_max_strategy_count = 0
    passing_strategy_count = 0
    best_observed: Mapping[str, Any] | None = None
    for strategy in strategies:
        reasons = _query_strategy_failure_reasons(strategy)
        if not reasons:
            passing_strategy_count += 1
        for reason in reasons:
            failure_counts[reason] += 1
        if "no_retrieval_hits" in reasons:
            no_hit_strategy_count += 1
        if "target_route_not_selected" in reasons:
            target_route_not_selected_strategy_count += 1
        if "blind_refuted_rate_below_min" in reasons:
            blind_refuted_rate_below_min_strategy_count += 1
        if "verified_false_alarm_above_max" in reasons:
            verified_false_alarm_above_max_strategy_count += 1
        if best_observed is None or _query_strategy_rank(strategy) > _query_strategy_rank(best_observed):
            best_observed = strategy

    best_reasons = (
        ()
        if best_observed is None
        else _query_strategy_failure_reasons(best_observed)
    )
    return {
        "strategy_count": len(strategies),
        "passing_strategy_count": passing_strategy_count,
        "failure_reason_counts": _sorted_counter(failure_counts),
        "no_hit_strategy_count": no_hit_strategy_count,
        "target_route_not_selected_strategy_count": target_route_not_selected_strategy_count,
        "blind_refuted_rate_below_min_strategy_count": (
            blind_refuted_rate_below_min_strategy_count
        ),
        "verified_false_alarm_above_max_strategy_count": (
            verified_false_alarm_above_max_strategy_count
        ),
        "best_observed_strategy": None if best_observed is None else best_observed.get("key"),
        "best_observed_blind_refuted_rate": (
            None
            if best_observed is None
            else _optional_float(_nested(best_observed, "blind_spot", "target_route_refuted_rate"))
        ),
        "best_observed_verified_false_alarm": (
            None
            if best_observed is None
            else _optional_float(_nested(best_observed, "gate", "verified_false_alarm"))
        ),
        "best_observed_records_with_hits": (
            None
            if best_observed is None
            else _nested_int(best_observed, "retrieval", "records_with_hits")
        ),
        "best_observed_total_hits": (
            None
            if best_observed is None
            else _nested_int(best_observed, "retrieval", "total_hits")
        ),
        "best_observed_failure_reasons": best_reasons,
        "recommended_next_actions": _query_sweep_next_actions(failure_counts),
    }


def _query_strategy_failure_reasons(strategy: Mapping[str, Any]) -> tuple[str, ...]:
    if bool(_nested(strategy, "gate", "pass")):
        return ()
    reasons: list[str] = []
    records_with_hits = _nested_int(strategy, "retrieval", "records_with_hits")
    if records_with_hits is not None and records_with_hits <= 0:
        reasons.append("no_retrieval_hits")
    selected_count = _nested_int(strategy, "blind_spot", "target_route_selected_count")
    if selected_count is not None and selected_count <= 0:
        reasons.append("target_route_not_selected")
    blind_refuted_rate = _optional_float(
        _nested(strategy, "blind_spot", "target_route_refuted_rate")
    )
    min_blind_refuted_rate = _optional_float(_nested(strategy, "gate", "min_blind_refuted_rate"))
    if (
        blind_refuted_rate is not None
        and min_blind_refuted_rate is not None
        and blind_refuted_rate < min_blind_refuted_rate
    ):
        reasons.append("blind_refuted_rate_below_min")
    verified_false_alarm = _optional_float(_nested(strategy, "gate", "verified_false_alarm"))
    max_verified_false_alarm = _optional_float(
        _nested(strategy, "gate", "max_verified_false_alarm")
    )
    if (
        verified_false_alarm is not None
        and max_verified_false_alarm is not None
        and verified_false_alarm > max_verified_false_alarm
    ):
        reasons.append("verified_false_alarm_above_max")
    if not reasons:
        reasons.append("unknown_gate_failure")
    return tuple(reasons)


def _query_strategy_rank(strategy: Mapping[str, Any]) -> tuple[float, float, float, int, int]:
    return (
        _optional_float(_nested(strategy, "blind_spot", "target_route_refuted_rate")) or 0.0,
        float(_nested_int(strategy, "blind_spot", "target_route_refuted_count") or 0),
        -(_optional_float(_nested(strategy, "gate", "verified_false_alarm")) or 1.0),
        _nested_int(strategy, "retrieval", "records_with_hits") or 0,
        _nested_int(strategy, "retrieval", "total_hits") or 0,
    )


def _query_sweep_next_actions(failure_counts: Counter[str]) -> tuple[str, ...]:
    actions: list[str] = []
    if failure_counts.get("no_retrieval_hits"):
        actions.append("expand_or_retarget_source_corpus")
    if failure_counts.get("target_route_not_selected"):
        actions.append("enable_or_repair_retrieval_route_selection")
    if failure_counts.get("blind_refuted_rate_below_min"):
        actions.append("improve_claim_intent_alignment_or_query_construction")
    if failure_counts.get("verified_false_alarm_above_max"):
        actions.append("tighten_false_alarm_calibration")
    if failure_counts.get("unknown_gate_failure"):
        actions.append("inspect_query_sweep_strategy_payload")
    return tuple(actions)


def _report_paths(
    *,
    paths: Mapping[str, Path],
    report_path: Path,
    manifest_path: Path,
    query_sweep_verified_records_dir: str | Path | None,
    active_corpus_path: Path | None,
    has_corpus: bool,
    has_binding_audit: bool,
    has_binding_failure_review: bool,
    has_provenance: bool,
    has_query_sweep: bool,
    has_comparison: bool,
) -> dict[str, str | None]:
    verified_records_dir = (
        None if query_sweep_verified_records_dir is None else Path(query_sweep_verified_records_dir)
    )
    return {
        "workflow_report": str(report_path),
        "artifact_manifest": str(manifest_path),
        "handoff_report": str(paths["handoff_report"]),
        "handoff_manifest": str(paths["handoff_manifest"]),
        "adapter_requests": str(paths["adapter_requests"]),
        "source_documents": str(paths["source_documents"]),
        "external_retrieval_corpus": str(active_corpus_path) if has_corpus and active_corpus_path is not None else None,
        "raw_external_retrieval_corpus": str(paths["corpus"]) if has_corpus else None,
        "binding_audit": str(paths["binding_audit"]) if has_binding_audit else None,
        "binding_failure_review": (
            str(paths["binding_failure_review"]) if has_binding_failure_review else None
        ),
        "bound_source_documents": str(paths["bound_source_documents"]) if has_binding_audit else None,
        "bound_retrieval_corpus": str(paths["bound_corpus"]) if has_binding_audit and has_corpus else None,
        "binding_manifest": str(paths["binding_manifest"]) if has_binding_audit else None,
        "provenance_audit": str(paths["provenance_audit"]) if has_provenance else None,
        "query_sweep": str(paths["query_sweep"]) if has_query_sweep else None,
        "query_sweep_manifest": str(paths["query_sweep_manifest"]) if has_query_sweep else None,
        "query_sweep_verified_records": (
            str(verified_records_dir)
            if has_query_sweep and verified_records_dir is not None and verified_records_dir.exists()
            else None
        ),
        "query_sweep_comparison": str(paths["query_sweep_comparison"]) if has_comparison else None,
        "query_sweep_comparison_manifest": (
            str(paths["query_sweep_comparison_manifest"]) if has_comparison else None
        ),
    }


def _manifest_artifacts(
    *,
    paths: Mapping[str, Path],
    report_path: Path,
    queue_report_path: str | Path,
    adapter_results_path: str | Path,
    scores_path: str | Path,
    blind_spots_path: str | Path,
    controlled_sweep_paths: Sequence[str | Path],
    query_sweep_verified_records_dir: str | Path | None,
    active_corpus_path: Path | None,
    has_corpus: bool,
    has_binding_audit: bool,
    has_binding_failure_review: bool,
    has_provenance: bool,
    has_query_sweep: bool,
    has_comparison: bool,
) -> dict[str, str | Path | None]:
    verified_records_dir = (
        None if query_sweep_verified_records_dir is None else Path(query_sweep_verified_records_dir)
    )
    artifacts: dict[str, str | Path | None] = {
        "workflow_report": report_path,
        "queue_report": Path(queue_report_path),
        "adapter_results": Path(adapter_results_path),
        "scores": Path(scores_path),
        "blind_spots": Path(blind_spots_path),
        "handoff_report": paths["handoff_report"],
        "handoff_manifest": paths["handoff_manifest"],
        "adapter_requests": paths["adapter_requests"],
        "source_documents": paths["source_documents"],
        "external_retrieval_corpus": active_corpus_path if has_corpus else None,
        "raw_external_retrieval_corpus": paths["corpus"] if has_corpus else None,
        "binding_audit": paths["binding_audit"] if has_binding_audit else None,
        "binding_failure_review": paths["binding_failure_review"] if has_binding_failure_review else None,
        "bound_source_documents": paths["bound_source_documents"] if has_binding_audit else None,
        "bound_retrieval_corpus": paths["bound_corpus"] if has_binding_audit and has_corpus else None,
        "binding_manifest": paths["binding_manifest"] if has_binding_audit else None,
        "provenance_audit": paths["provenance_audit"] if has_provenance else None,
        "query_sweep": paths["query_sweep"] if has_query_sweep else None,
        "query_sweep_manifest": paths["query_sweep_manifest"] if has_query_sweep else None,
        "query_sweep_verified_records": (
            verified_records_dir
            if has_query_sweep and verified_records_dir is not None and verified_records_dir.exists()
            else None
        ),
        "query_sweep_comparison": paths["query_sweep_comparison"] if has_comparison else None,
        "query_sweep_comparison_manifest": paths["query_sweep_comparison_manifest"] if has_comparison else None,
    }
    for index, path in enumerate(controlled_sweep_paths, start=1):
        artifacts[f"controlled_sweep_{index}"] = Path(path)
    return artifacts


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _nested(payload: Mapping[str, Any] | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _nested_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    value = _nested(payload, *keys)
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(
        sorted(
            ((key, value) for key, value in counter.items() if key),
            key=lambda item: (-item[1], item[0]),
        )
    )


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


def _parse_csv_strings(value: str, *, choices: Sequence[str], name: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError(f"{name} must contain at least one value.")
    invalid = sorted(set(items) - set(choices))
    if invalid:
        raise ValueError(f"{name} contains invalid values: {', '.join(invalid)}.")
    return items


def _parse_csv_floats(value: str, *, name: str) -> tuple[float, ...]:
    items = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError(f"{name} must contain at least one value.")
    return items


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--adapter-results", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--blind-spots", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--controlled-sweep", action="append", default=[])
    parser.add_argument("--workflow-report", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--batch-id",
        action="append",
        default=[],
        help="Execution batch id from the unresolved queue to pass into the citation/search handoff.",
    )
    parser.add_argument("--query-mode", choices=QUERY_MODES, default="question")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-results-per-request", type=int, default=None)
    parser.add_argument("--max-alternate-queries", type=int, default=DEFAULT_MAX_ALTERNATE_QUERIES)
    parser.add_argument("--corpus-name", default=DEFAULT_CORPUS_NAME)
    parser.add_argument("--source-kind", default=DEFAULT_SOURCE_KIND)
    parser.add_argument("--query-fields", default="question,question_answer")
    parser.add_argument("--retriever-min-overlaps", default=",".join(str(value) for value in DEFAULT_MIN_OVERLAPS))
    parser.add_argument(
        "--source-family-filters",
        default=",".join(DEFAULT_SOURCE_FAMILY_FILTERS),
        help="comma-separated source-family evidence filters to sweep: off,planned,planned_rerank",
    )
    parser.add_argument(
        "--query-sweep-verified-records-dir",
        default=None,
        help="optional directory to save per-strategy query sweep verified-records JSONL sidecars",
    )
    parser.add_argument("--retrieval-limit", type=int, default=3)
    parser.add_argument("--signal", default="truth_proj")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verifier-min-overlap", type=float, default=0.65)
    parser.add_argument("--target-route", default=DEFAULT_TARGET_ROUTE)
    parser.add_argument("--max-verified-false-alarm", type=float, default=0.05)
    parser.add_argument("--min-blind-refuted-rate", type=float, default=0.50)
    parser.add_argument("--min-controlled-blind-refuted-rate", type=float, default=0.50)
    parser.add_argument("--min-external-blind-refuted-rate", type=float, default=0.50)
    parser.add_argument("--max-controlled-verified-false-alarm", type=float, default=0.05)
    parser.add_argument("--max-external-verified-false-alarm", type=float, default=0.05)
    parser.add_argument("--min-adapter-request-coverage", type=float, default=1.0)
    parser.add_argument("--max-exact-answer-copy-rate", type=float, default=0.80)
    parser.add_argument("--max-claim-id-link-rate", type=float, default=0.0)
    parser.add_argument("--max-label-metadata-rate", type=float, default=0.0)
    parser.add_argument(
        "--audit-source-bindings",
        action="store_true",
        help="Run claim-specific citation/source binding before provenance and query-sweep gates.",
    )
    parser.add_argument("--require-binding-source-family-match", action="store_true")
    parser.add_argument(
        "--clip-binding-evidence-spans",
        action="store_true",
        help="Use accepted claim-specific evidence spans as bound-corpus text; default only records span metadata.",
    )
    parser.add_argument("--binding-min-keyword-overlap", type=float, default=0.2)
    parser.add_argument("--binding-min-support-keyword-overlap", type=float, default=0.65)
    parser.add_argument("--binding-min-entity-recall", type=float, default=0.5)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    report = run(
        queue_report_path=args.queue,
        adapter_results_path=args.adapter_results,
        scores_path=args.scores,
        blind_spots_path=args.blind_spots,
        output_dir=args.output_dir,
        controlled_sweep_paths=tuple(args.controlled_sweep or ()),
        workflow_report_path=args.workflow_report,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        batch_ids=tuple(args.batch_id or ()),
        query_mode=args.query_mode,
        max_requests=args.max_requests,
        max_results_per_request=args.max_results_per_request,
        max_alternate_queries=args.max_alternate_queries,
        corpus_name=args.corpus_name,
        source_kind=args.source_kind,
        query_fields=_parse_csv_strings(args.query_fields, choices=QUERY_FIELDS, name="query_fields"),
        retriever_min_overlaps=_parse_csv_floats(args.retriever_min_overlaps, name="retriever_min_overlaps"),
        source_family_filters=_parse_csv_strings(
            args.source_family_filters,
            choices=("off", "planned", "planned_rerank"),
            name="source_family_filters",
        ),
        query_sweep_verified_records_dir=args.query_sweep_verified_records_dir,
        retrieval_limit=args.retrieval_limit,
        signal=args.signal,
        alpha=args.alpha,
        repeats=args.repeats,
        seed=args.seed,
        verifier_min_overlap=args.verifier_min_overlap,
        target_route=args.target_route,
        max_verified_false_alarm=args.max_verified_false_alarm,
        min_blind_refuted_rate=args.min_blind_refuted_rate,
        min_controlled_blind_refuted_rate=args.min_controlled_blind_refuted_rate,
        min_external_blind_refuted_rate=args.min_external_blind_refuted_rate,
        max_controlled_verified_false_alarm=args.max_controlled_verified_false_alarm,
        max_external_verified_false_alarm=args.max_external_verified_false_alarm,
        min_adapter_request_coverage=args.min_adapter_request_coverage,
        max_exact_answer_copy_rate=args.max_exact_answer_copy_rate,
        max_claim_id_link_rate=args.max_claim_id_link_rate,
        max_label_metadata_rate=args.max_label_metadata_rate,
        audit_source_bindings=bool(args.audit_source_bindings),
        require_binding_source_family_match=bool(args.require_binding_source_family_match),
        clip_binding_evidence_spans=bool(args.clip_binding_evidence_spans),
        binding_min_keyword_overlap=float(args.binding_min_keyword_overlap),
        binding_min_support_keyword_overlap=float(args.binding_min_support_keyword_overlap),
        binding_min_entity_recall=float(args.binding_min_entity_recall),
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    print(
        "citation_search_evidence_workflow_ok "
        f"status={report['status']} "
        f"passed={report['gate']['passed']} "
        f"promotion_ready={report['gate']['promotion_ready']}"
    )


if __name__ == "__main__":
    main()
