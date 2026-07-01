"""Run local source-family citation/search through release evidence gates.

This workflow is the no-network, source-family counterpart to
``run_external_citation_search_adapter_workflow.py``. It builds sanitized
citation/search requests from an unresolved blind-spot queue, ranks caller-
supplied local source catalogs with
``run_source_family_citation_search_adapter.py``, then runs the normal
citation/search evidence workflow before any result can become release evidence.
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

from benchmarks.build_citation_search_adapter_handoff import (  # noqa: E402
    DEFAULT_CORPUS_NAME,
    DEFAULT_MAX_ALTERNATE_QUERIES,
    DEFAULT_SOURCE_KIND,
    QUERY_MODES,
)
from benchmarks.build_citation_search_adapter_handoff import (  # noqa: E402
    run as run_citation_search_handoff,
)
from benchmarks.run_citation_search_evidence_workflow import run as run_citation_search_evidence  # noqa: E402
from benchmarks.run_source_family_citation_search_adapter import (  # noqa: E402
    DEFAULT_SOURCE_FAMILY,
    run_source_family_citation_search_adapter,
)
from benchmarks.sweep_blind_spot_retrieval_queries import (  # noqa: E402
    DEFAULT_MIN_OVERLAPS,
    DEFAULT_TARGET_ROUTE,
    QUERY_FIELDS,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "source_family_citation_search_workflow"


def run_source_family_citation_search_workflow(
    *,
    queue_report_path: str | Path,
    source_catalog_paths: Sequence[str | Path],
    scores_path: str | Path,
    blind_spots_path: str | Path,
    output_dir: str | Path,
    controlled_sweep_paths: Sequence[str | Path] = (),
    request_jsonl_path: str | Path | None = None,
    adapter_results_path: str | Path | None = None,
    adapter_report_path: str | Path | None = None,
    workflow_report_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    batch_ids: Sequence[str] = (),
    query_mode: str = "claim_entity",
    max_requests: int | None = None,
    max_results_per_request: int | None = None,
    max_alternate_queries: int = DEFAULT_MAX_ALTERNATE_QUERIES,
    adapter_max_results: int = 3,
    adapter_max_query_variants: int = 3,
    adapter_min_text_overlap: float = 0.05,
    min_adapter_request_coverage: float = 1.0,
    adapter_diversify_source_families: bool = False,
    default_source_family: str = DEFAULT_SOURCE_FAMILY,
    corpus_name: str = DEFAULT_CORPUS_NAME,
    source_kind: str = DEFAULT_SOURCE_KIND,
    query_fields: Sequence[str] = ("question", "question_answer"),
    retriever_min_overlaps: Sequence[float] = DEFAULT_MIN_OVERLAPS,
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
    max_exact_answer_copy_rate: float = 0.80,
    max_claim_id_link_rate: float = 0.0,
    max_label_metadata_rate: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
    fail_on_blocked: bool = False,
) -> dict[str, Any]:
    """Run source-family catalog ranking and gate the returned evidence."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if not source_catalog_paths:
        raise ValueError("source_catalog_paths must contain at least one path.")
    if query_mode not in QUERY_MODES:
        raise ValueError(f"query_mode must be one of: {', '.join(QUERY_MODES)}.")
    if not (0.0 <= float(min_adapter_request_coverage) <= 1.0):
        raise ValueError("min_adapter_request_coverage must be between 0 and 1.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    request_jsonl = Path(request_jsonl_path or output / "source-family-citation-search-requests.jsonl")
    adapter_results = Path(adapter_results_path or output / "source-family-citation-search-results.jsonl")
    adapter_report = Path(adapter_report_path or output / "source-family-citation-search-adapter-report.json")
    adapter_manifest = output / "source-family-citation-search-adapter-manifest.json"
    workflow_report = Path(workflow_report_path or output / "source-family-citation-search-workflow.json")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")
    preflight_dir = output / "request-handoff"
    evidence_dir = output / "evidence-gate"
    workflow_metadata = {**dict(metadata or {}), "source_workflow": WORKFLOW}

    preflight = run_citation_search_handoff(
        queue_report_path=queue_report_path,
        output_dir=preflight_dir,
        report_json_path=preflight_dir / "citation-search-request-handoff.json",
        request_jsonl_path=request_jsonl,
        source_jsonl_path=preflight_dir / "citation-search-source-docs.jsonl",
        artifact_manifest_path=preflight_dir / "artifact-manifest.json",
        batch_ids=batch_ids,
        query_mode=query_mode,
        max_requests=max_requests,
        max_results_per_request=max_results_per_request,
        max_alternate_queries=max_alternate_queries,
        corpus_name=corpus_name,
        source_kind=source_kind,
        metadata=workflow_metadata,
        compact_json=compact_json,
    )
    adapter = run_source_family_citation_search_adapter(
        input_path=request_jsonl,
        output_path=adapter_results,
        source_catalog_paths=tuple(source_catalog_paths),
        report_json_path=adapter_report,
        artifact_manifest_path=adapter_manifest,
        max_results=adapter_max_results,
        max_query_variants=adapter_max_query_variants,
        min_text_overlap=adapter_min_text_overlap,
        min_request_coverage=min_adapter_request_coverage,
        diversify_source_families=adapter_diversify_source_families,
        default_source_family=default_source_family,
        metadata=workflow_metadata,
        compact_json=compact_json,
    )
    evidence = run_citation_search_evidence(
        queue_report_path=queue_report_path,
        adapter_results_path=adapter_results,
        scores_path=scores_path,
        blind_spots_path=blind_spots_path,
        controlled_sweep_paths=controlled_sweep_paths,
        output_dir=evidence_dir,
        batch_ids=batch_ids,
        query_mode=query_mode,
        max_requests=max_requests,
        max_results_per_request=max_results_per_request,
        max_alternate_queries=max_alternate_queries,
        corpus_name=corpus_name,
        source_kind=source_kind,
        query_fields=query_fields,
        retriever_min_overlaps=retriever_min_overlaps,
        retrieval_limit=retrieval_limit,
        signal=signal,
        alpha=alpha,
        repeats=repeats,
        seed=seed,
        verifier_min_overlap=verifier_min_overlap,
        target_route=target_route,
        min_adapter_request_coverage=min_adapter_request_coverage,
        max_verified_false_alarm=max_verified_false_alarm,
        min_blind_refuted_rate=min_blind_refuted_rate,
        min_controlled_blind_refuted_rate=min_controlled_blind_refuted_rate,
        min_external_blind_refuted_rate=min_external_blind_refuted_rate,
        max_controlled_verified_false_alarm=max_controlled_verified_false_alarm,
        max_external_verified_false_alarm=max_external_verified_false_alarm,
        max_exact_answer_copy_rate=max_exact_answer_copy_rate,
        max_claim_id_link_rate=max_claim_id_link_rate,
        max_label_metadata_rate=max_label_metadata_rate,
        metadata=workflow_metadata,
        compact_json=compact_json,
    )
    adapter_gate = _adapter_gate(adapter, min_adapter_request_coverage=float(min_adapter_request_coverage))
    evidence_gate = {
        "passed": bool(evidence.get("gate", {}).get("passed")),
        "promotion_ready": bool(evidence.get("gate", {}).get("promotion_ready")),
        "blocking_reasons": tuple(evidence.get("gate", {}).get("blocking_reasons", ())),
    }
    gate = _combined_gate(adapter_gate=adapter_gate, evidence_gate=evidence_gate)
    status = "promote" if gate["promotion_ready"] else ("complete" if gate["passed"] else "blocked")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "source": {
            "queue_report": str(queue_report_path),
            "scores": str(scores_path),
            "blind_spots": str(blind_spots_path),
            "controlled_sweeps": tuple(str(path) for path in controlled_sweep_paths),
            "source_catalogs": tuple(str(path) for path in source_catalog_paths),
        },
        "config": {
            "batch_ids": tuple(str(item) for item in batch_ids),
            "query_mode": query_mode,
            "max_requests": max_requests,
            "max_results_per_request": max_results_per_request,
            "max_alternate_queries": int(max_alternate_queries),
            "adapter_max_results": int(adapter_max_results),
            "adapter_max_query_variants": int(adapter_max_query_variants),
            "adapter_min_text_overlap": float(adapter_min_text_overlap),
            "min_adapter_request_coverage": float(min_adapter_request_coverage),
            "adapter_diversify_source_families": bool(adapter_diversify_source_families),
            "default_source_family": default_source_family,
            "corpus_name": corpus_name,
            "source_kind": source_kind,
            "query_fields": tuple(query_fields),
            "retriever_min_overlaps": tuple(float(value) for value in retriever_min_overlaps),
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
            "max_exact_answer_copy_rate": float(max_exact_answer_copy_rate),
            "max_claim_id_link_rate": float(max_claim_id_link_rate),
            "max_label_metadata_rate": float(max_label_metadata_rate),
        },
        "paths": {
            "requests": str(request_jsonl),
            "adapter_results": str(adapter_results),
            "adapter_report": str(adapter_report),
            "adapter_manifest": str(adapter_manifest),
            "request_handoff": str(preflight_dir / "citation-search-request-handoff.json"),
            "request_handoff_manifest": str(preflight_dir / "artifact-manifest.json"),
            "evidence_workflow": str(evidence_dir / "citation-search-evidence-workflow.json"),
            "evidence_manifest": str(evidence_dir / "artifact-manifest.json"),
            "workflow_report": str(workflow_report),
            "artifact_manifest": str(manifest_path),
        },
        "request_summary": dict(preflight.get("summary", {})),
        "adapter_summary": dict(adapter.get("summary", {})),
        "evidence_summary": dict(evidence.get("summary", {})),
        "adapter_gate": adapter_gate,
        "evidence_gate": evidence_gate,
        "gate": gate,
        "metadata": dict(metadata or {}),
    }
    _write_json(workflow_report, payload, compact=compact_json)
    manifest = build_artifact_manifest(
        _manifest_artifacts(
            queue_report_path=queue_report_path,
            scores_path=scores_path,
            blind_spots_path=blind_spots_path,
            controlled_sweep_paths=controlled_sweep_paths,
            source_catalog_paths=source_catalog_paths,
            request_jsonl_path=request_jsonl,
            adapter_results_path=adapter_results,
            adapter_report_path=adapter_report,
            adapter_manifest_path=adapter_manifest,
            preflight_dir=preflight_dir,
            evidence_dir=evidence_dir,
            workflow_report_path=workflow_report,
        ),
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "gate_passed": gate["passed"],
            "promotion_ready": gate["promotion_ready"],
            "adapter_gate_passed": adapter_gate["passed"],
            "adapter_gate_status": adapter_gate["status"],
            "adapter_request_coverage": adapter_gate["request_coverage"],
            "min_adapter_request_coverage": adapter_gate["min_request_coverage"],
            "evidence_gate_passed": evidence_gate["passed"],
            "target_route": target_route,
            "selected_batch_count": payload["request_summary"].get("selected_batch_count"),
            "selected_batch_ids": payload["request_summary"].get("selected_batch_ids"),
            "adapter_request_count": payload["request_summary"].get("adapter_request_count"),
            "catalog_source_document_count": payload["adapter_summary"].get("source_document_count"),
            "adapter_result_count": payload["adapter_summary"].get("result_count"),
            "evidence_source_document_count": payload["evidence_summary"].get("source_document_count"),
            "corpus_document_count": payload["evidence_summary"].get("corpus_document_count"),
            **dict(metadata or {}),
        },
    )
    _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=workflow_report,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "gate_passed": gate["passed"],
                "promotion_ready": gate["promotion_ready"],
                "adapter_gate_passed": adapter_gate["passed"],
                "adapter_gate_status": adapter_gate["status"],
                "adapter_request_coverage": adapter_gate["request_coverage"],
                "min_adapter_request_coverage": adapter_gate["min_request_coverage"],
                "evidence_gate_passed": evidence_gate["passed"],
                "target_route": target_route,
                "selected_batch_count": payload["request_summary"].get("selected_batch_count"),
                "selected_batch_ids": payload["request_summary"].get("selected_batch_ids"),
                "adapter_request_count": payload["request_summary"].get("adapter_request_count"),
                "catalog_source_document_count": payload["adapter_summary"].get("source_document_count"),
                "adapter_result_count": payload["adapter_summary"].get("result_count"),
                "evidence_source_document_count": payload["evidence_summary"].get("source_document_count"),
                "corpus_document_count": payload["evidence_summary"].get("corpus_document_count"),
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    if fail_on_blocked and payload["status"] != "promote":
        raise SystemExit(1)
    return payload


def _manifest_artifacts(
    *,
    queue_report_path: str | Path,
    scores_path: str | Path,
    blind_spots_path: str | Path,
    controlled_sweep_paths: Sequence[str | Path],
    source_catalog_paths: Sequence[str | Path],
    request_jsonl_path: Path,
    adapter_results_path: Path,
    adapter_report_path: Path,
    adapter_manifest_path: Path,
    preflight_dir: Path,
    evidence_dir: Path,
    workflow_report_path: Path,
) -> dict[str, str | Path]:
    artifacts: dict[str, str | Path] = {
        "workflow_report": workflow_report_path,
        "queue_report": Path(queue_report_path),
        "scores": Path(scores_path),
        "blind_spots": Path(blind_spots_path),
        "source_family_citation_search_requests": request_jsonl_path,
        "source_family_citation_search_results": adapter_results_path,
        "source_family_adapter_report": adapter_report_path,
        "source_family_adapter_manifest": adapter_manifest_path,
        "request_handoff": preflight_dir / "citation-search-request-handoff.json",
        "request_handoff_manifest": preflight_dir / "artifact-manifest.json",
        "evidence_workflow": evidence_dir / "citation-search-evidence-workflow.json",
        "evidence_manifest": evidence_dir / "artifact-manifest.json",
    }
    for index, path in enumerate(source_catalog_paths, start=1):
        artifacts[f"source_catalog_{index}"] = Path(path)
    for index, path in enumerate(controlled_sweep_paths, start=1):
        artifacts[f"controlled_sweep_{index}"] = Path(path)
    return artifacts


def _adapter_gate(
    adapter_report: Mapping[str, Any],
    *,
    min_adapter_request_coverage: float,
) -> dict[str, Any]:
    raw_gate = adapter_report.get("gate")
    if isinstance(raw_gate, Mapping):
        gate = dict(raw_gate)
    else:
        summary = adapter_report.get("summary") if isinstance(adapter_report.get("summary"), Mapping) else {}
        request_coverage = float(summary.get("request_coverage") or 0.0)
        blocking: list[dict[str, Any]] = []
        if request_coverage < min_adapter_request_coverage:
            blocking.append({
                "gate": "adapter_request_coverage",
                "reason": (
                    "Source-family citation/search adapter covered "
                    f"{request_coverage:.3f} of selected requests, "
                    f"below required {min_adapter_request_coverage:.3f}."
                ),
            })
        gate = {
            "status": "complete" if not blocking else "partial",
            "passed": not blocking,
            "request_coverage": request_coverage,
            "min_request_coverage": float(min_adapter_request_coverage),
            "blocking_reasons": tuple(blocking),
        }
    blocking_reasons = tuple(gate.get("blocking_reasons", ()))
    fallback_status = "complete" if gate.get("passed") else "blocked"
    return {
        "status": str(gate.get("status") or adapter_report.get("status") or fallback_status),
        "passed": bool(gate.get("passed")) and not blocking_reasons,
        "request_coverage": float(gate.get("request_coverage") or 0.0),
        "min_request_coverage": float(gate.get("min_request_coverage") or min_adapter_request_coverage),
        "blocking_reasons": blocking_reasons,
    }


def _combined_gate(
    *,
    adapter_gate: Mapping[str, Any],
    evidence_gate: Mapping[str, Any],
) -> dict[str, Any]:
    blocking = (
        *tuple(adapter_gate.get("blocking_reasons", ())),
        *tuple(evidence_gate.get("blocking_reasons", ())),
    )
    adapter_passed = bool(adapter_gate.get("passed"))
    evidence_passed = bool(evidence_gate.get("passed"))
    evidence_promotion_ready = bool(evidence_gate.get("promotion_ready"))
    return {
        "passed": adapter_passed and evidence_passed and not blocking,
        "promotion_ready": adapter_passed and evidence_promotion_ready and not blocking,
        "adapter_gate_passed": adapter_passed,
        "evidence_gate_passed": evidence_passed,
        "blocking_reasons": blocking,
    }


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
    parser.add_argument("--source-catalog", action="append", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--blind-spots", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--controlled-sweep", action="append", default=[])
    parser.add_argument("--request-jsonl", default=None)
    parser.add_argument("--adapter-results", default=None)
    parser.add_argument("--adapter-report", default=None)
    parser.add_argument("--workflow-report", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--batch-id",
        action="append",
        default=[],
        help="Execution batch id from the unresolved queue to pass into request handoff and evidence gating.",
    )
    parser.add_argument("--query-mode", choices=QUERY_MODES, default="claim_entity")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-results-per-request", type=int, default=None)
    parser.add_argument("--max-alternate-queries", type=int, default=DEFAULT_MAX_ALTERNATE_QUERIES)
    parser.add_argument("--adapter-max-results", type=int, default=3)
    parser.add_argument("--adapter-max-query-variants", type=int, default=3)
    parser.add_argument("--adapter-min-text-overlap", type=float, default=0.05)
    parser.add_argument("--min-adapter-request-coverage", type=float, default=1.0)
    parser.add_argument("--adapter-diversify-source-families", action="store_true")
    parser.add_argument("--default-source-family", default=DEFAULT_SOURCE_FAMILY)
    parser.add_argument("--corpus-name", default=DEFAULT_CORPUS_NAME)
    parser.add_argument("--source-kind", default=DEFAULT_SOURCE_KIND)
    parser.add_argument("--query-fields", default="question,question_answer")
    parser.add_argument("--retriever-min-overlaps", default=",".join(str(value) for value in DEFAULT_MIN_OVERLAPS))
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
    parser.add_argument("--max-exact-answer-copy-rate", type=float, default=0.80)
    parser.add_argument("--max-claim-id-link-rate", type=float, default=0.0)
    parser.add_argument("--max-label-metadata-rate", type=float, default=0.0)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)
    report = run_source_family_citation_search_workflow(
        queue_report_path=args.queue,
        source_catalog_paths=tuple(args.source_catalog or ()),
        scores_path=args.scores,
        blind_spots_path=args.blind_spots,
        output_dir=args.output_dir,
        controlled_sweep_paths=tuple(args.controlled_sweep or ()),
        request_jsonl_path=args.request_jsonl,
        adapter_results_path=args.adapter_results,
        adapter_report_path=args.adapter_report,
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
        adapter_max_results=args.adapter_max_results,
        adapter_max_query_variants=args.adapter_max_query_variants,
        adapter_min_text_overlap=args.adapter_min_text_overlap,
        min_adapter_request_coverage=args.min_adapter_request_coverage,
        adapter_diversify_source_families=bool(args.adapter_diversify_source_families),
        default_source_family=args.default_source_family,
        corpus_name=args.corpus_name,
        source_kind=args.source_kind,
        query_fields=_parse_csv_strings(args.query_fields, choices=QUERY_FIELDS, name="query_fields"),
        retriever_min_overlaps=_parse_csv_floats(args.retriever_min_overlaps, name="retriever_min_overlaps"),
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
        max_exact_answer_copy_rate=args.max_exact_answer_copy_rate,
        max_claim_id_link_rate=args.max_claim_id_link_rate,
        max_label_metadata_rate=args.max_label_metadata_rate,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
        fail_on_blocked=bool(args.fail_on_blocked),
    )
    print(
        "source_family_citation_search_workflow_ok "
        f"status={report['status']} "
        f"passed={report['gate']['passed']} "
        f"promotion_ready={report['gate']['promotion_ready']} "
        f"adapter_results={report['adapter_summary']['result_count']}"
    )


if __name__ == "__main__":
    main()
