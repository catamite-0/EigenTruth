"""Build a next-evidence plan from blocked release-candidate reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.plan_citation_batch_evidence_reruns import (  # noqa: E402
    build_citation_batch_evidence_rerun_queue,
)
from benchmarks.plan_frontier_abstention_evidence_reruns import (  # noqa: E402
    build_frontier_abstention_evidence_rerun_queue,
)
from benchmarks.plan_frontier_detectability_evidence_reruns import (  # noqa: E402
    build_frontier_detectability_evidence_rerun_queue,
)
from benchmarks.plan_frontier_multiple_testing_reruns import (  # noqa: E402
    build_frontier_multiple_testing_rerun_queue,
)
from benchmarks.plan_frontier_stability_evidence_reruns import (  # noqa: E402
    build_frontier_stability_evidence_rerun_queue,
)
from eigentruth.control import plan_evidence_gaps_from_release_candidate  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

FRONTIER_RERUN_ROLLUP_COMPLETION_WORKFLOW = "frontier_rerun_rollup_completion_plan"
RUNTIME_DRIFT_COMPLETION_WORKFLOW = "runtime_drift_evidence_completion_plan"
FRONTIER_RERUN_ROLLUP_WORKFLOW_TO_TRACK = {
    "frontier_stability_evidence_rerun_rollup": "stability",
    "frontier_abstention_evidence_rerun_rollup": "abstention",
    "frontier_detectability_evidence_rerun_rollup": "detectability",
    "frontier_multiple_testing_rerun_rollup": "multiple_testing",
}
FRONTIER_RERUN_ROLLUP_TRACKS = {
    "stability": {
        "workflow": "frontier_stability_evidence_rerun_rollup",
        "script": "benchmarks/rollup_frontier_stability_evidence_reruns.py",
    },
    "abstention": {
        "workflow": "frontier_abstention_evidence_rerun_rollup",
        "script": "benchmarks/rollup_frontier_abstention_evidence_reruns.py",
    },
    "detectability": {
        "workflow": "frontier_detectability_evidence_rerun_rollup",
        "script": "benchmarks/rollup_frontier_detectability_evidence_reruns.py",
    },
    "multiple_testing": {
        "workflow": "frontier_multiple_testing_rerun_rollup",
        "script": "benchmarks/rollup_frontier_multiple_testing_reruns.py",
    },
}
RUNTIME_DRIFT_COMPLETION_ROUTES = {
    "product_runtime_baseline",
    "product_runtime_drift",
    "product_trace_replay",
    "product_trace_runtime_evidence",
    "product_promotion_evidence_handoff",
    "evidence_handoff_audit",
    "evidence_handoff_evidence",
    "world_model_evidence",
    "context_sensitivity_evidence",
    "counterfactual_robustness_evidence",
    "provenance_evidence",
    "citation_integrity_evidence",
    "trajectory_audit_evidence",
    "frontier_release_evidence_promotion_metrics",
    "triple_audit_evidence",
    "covered_fact_property",
}
RUNTIME_DRIFT_COMPLETION_METADATA_KEYS = {
    "runtime_baseline_script",
    "runtime_drift_script",
    "trace_replay_script",
    "trace_enrichment_script",
    "evidence_handoff_script",
    "signal_workflow_script",
    "context_workflow_script",
    "counterfactual_eval_script",
    "frontier_release_evidence_script",
    "triple_extraction_matrix_script",
    "structured_route_script",
}


def build_release_evidence_gap_plan(
    *,
    source: str | Path,
    json_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    multiple_testing_rerun_json_path: str | Path | None = None,
    multiple_testing_rerun_artifact_manifest_path: str | Path | None = None,
    multiple_testing_rerun_output_dir: str | Path | None = None,
    multiple_testing_rerun_name: str | None = None,
    multiple_testing_rerun_version: str | None = None,
    citation_batch_rerun_json_path: str | Path | None = None,
    citation_batch_rerun_artifact_manifest_path: str | Path | None = None,
    citation_batch_rerun_output_dir: str | Path | None = None,
    citation_batch_rerun_name: str | None = None,
    citation_batch_rerun_version: str | None = None,
    citation_batch_queue_report_path: str | Path | None = None,
    citation_batch_scores_path: str | Path | None = None,
    citation_batch_blind_spots_path: str | Path | None = None,
    citation_batch_source_catalog_paths: Sequence[str | Path] = (),
    citation_batch_search_command: str | None = None,
    citation_batch_controlled_sweep_paths: Sequence[str | Path] = (),
    citation_batch_query_mode: str = "claim_entity",
    stability_rerun_json_path: str | Path | None = None,
    stability_rerun_artifact_manifest_path: str | Path | None = None,
    stability_rerun_output_dir: str | Path | None = None,
    stability_rerun_name: str | None = None,
    stability_rerun_version: str | None = None,
    stability_score_paths: Sequence[str | Path] = (),
    stability_seeds: str | None = None,
    verifier_signal: str | None = None,
    verifier_claims_path: str | Path | None = None,
    verifier_qa_corpus_path: str | Path | None = None,
    verifier_state_source_path: str | Path | None = None,
    verifier_staged_verification: bool = True,
    abstention_signals: Sequence[str] = (),
    abstention_rerun_json_path: str | Path | None = None,
    abstention_rerun_artifact_manifest_path: str | Path | None = None,
    abstention_rerun_output_dir: str | Path | None = None,
    abstention_rerun_name: str | None = None,
    abstention_rerun_version: str | None = None,
    abstention_score_paths: Sequence[str | Path] = (),
    abstention_profiles: Sequence[str] = (),
    abstention_signal_groups: Sequence[str] = (),
    abstention_seeds: str | None = None,
    abstention_direction: str | None = None,
    detectability_rerun_json_path: str | Path | None = None,
    detectability_rerun_artifact_manifest_path: str | Path | None = None,
    detectability_rerun_output_dir: str | Path | None = None,
    detectability_rerun_name: str | None = None,
    detectability_rerun_version: str | None = None,
    detectability_score_paths: Sequence[str | Path] = (),
    detectability_consistency_signal: str | None = None,
    detectability_confidence_signal: str | None = None,
    detectability_consistency_direction: str = "higher",
    detectability_confidence_direction: str = "higher",
    detectability_include_taxonomy_reruns: bool = False,
    detectability_taxonomy_pairs: Sequence[str] = (),
    detectability_cell: str = "entrenched",
    detectability_max_records: int = 100,
    frontier_rerun_rollup_completion_json_path: str | Path | None = None,
    frontier_rerun_rollup_completion_artifact_manifest_path: str | Path | None = None,
    frontier_rerun_rollup_completion_output_dir: str | Path | None = None,
    frontier_rerun_rollup_completion_name: str | None = None,
    frontier_rerun_rollup_completion_version: str | None = None,
    frontier_rerun_rollup_queue_paths: Sequence[str | Path] = (),
    runtime_drift_completion_json_path: str | Path | None = None,
    runtime_drift_completion_artifact_manifest_path: str | Path | None = None,
    runtime_drift_completion_output_dir: str | Path | None = None,
    runtime_drift_completion_name: str | None = None,
    runtime_drift_completion_version: str | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Load a release report and optionally write/register its evidence-gap plan."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if multiple_testing_rerun_artifact_manifest_path is not None and multiple_testing_rerun_json_path is None:
        raise ValueError("multiple_testing_rerun_artifact_manifest_path requires multiple_testing_rerun_json_path.")
    if (multiple_testing_rerun_name or multiple_testing_rerun_version) and registry_path is None:
        raise ValueError("multiple_testing_rerun_name/version require registry_path.")
    if (multiple_testing_rerun_name is None) != (multiple_testing_rerun_version is None):
        raise ValueError("multiple_testing_rerun_name and multiple_testing_rerun_version must be provided together.")
    if citation_batch_rerun_artifact_manifest_path is not None and citation_batch_rerun_json_path is None:
        raise ValueError("citation_batch_rerun_artifact_manifest_path requires citation_batch_rerun_json_path.")
    if (citation_batch_rerun_name or citation_batch_rerun_version) and registry_path is None:
        raise ValueError("citation_batch_rerun_name/version require registry_path.")
    if (citation_batch_rerun_name is None) != (citation_batch_rerun_version is None):
        raise ValueError("citation_batch_rerun_name and citation_batch_rerun_version must be provided together.")
    if stability_rerun_artifact_manifest_path is not None and stability_rerun_json_path is None:
        raise ValueError("stability_rerun_artifact_manifest_path requires stability_rerun_json_path.")
    if (stability_rerun_name or stability_rerun_version) and registry_path is None:
        raise ValueError("stability_rerun_name/version require registry_path.")
    if (stability_rerun_name is None) != (stability_rerun_version is None):
        raise ValueError("stability_rerun_name and stability_rerun_version must be provided together.")
    if abstention_rerun_artifact_manifest_path is not None and abstention_rerun_json_path is None:
        raise ValueError("abstention_rerun_artifact_manifest_path requires abstention_rerun_json_path.")
    if (abstention_rerun_name or abstention_rerun_version) and registry_path is None:
        raise ValueError("abstention_rerun_name/version require registry_path.")
    if (abstention_rerun_name is None) != (abstention_rerun_version is None):
        raise ValueError("abstention_rerun_name and abstention_rerun_version must be provided together.")
    if detectability_rerun_artifact_manifest_path is not None and detectability_rerun_json_path is None:
        raise ValueError("detectability_rerun_artifact_manifest_path requires detectability_rerun_json_path.")
    if (detectability_rerun_name or detectability_rerun_version) and registry_path is None:
        raise ValueError("detectability_rerun_name/version require registry_path.")
    if (detectability_rerun_name is None) != (detectability_rerun_version is None):
        raise ValueError("detectability_rerun_name and detectability_rerun_version must be provided together.")
    if (
        frontier_rerun_rollup_completion_artifact_manifest_path is not None
        and frontier_rerun_rollup_completion_json_path is None
    ):
        raise ValueError(
            "frontier_rerun_rollup_completion_artifact_manifest_path requires "
            "frontier_rerun_rollup_completion_json_path."
        )
    if (
        frontier_rerun_rollup_completion_name or frontier_rerun_rollup_completion_version
    ) and registry_path is None:
        raise ValueError("frontier_rerun_rollup_completion_name/version require registry_path.")
    if (frontier_rerun_rollup_completion_name is None) != (
        frontier_rerun_rollup_completion_version is None
    ):
        raise ValueError(
            "frontier_rerun_rollup_completion_name and "
            "frontier_rerun_rollup_completion_version must be provided together."
        )
    if (
        runtime_drift_completion_artifact_manifest_path is not None
        and runtime_drift_completion_json_path is None
    ):
        raise ValueError(
            "runtime_drift_completion_artifact_manifest_path requires "
            "runtime_drift_completion_json_path."
        )
    if (runtime_drift_completion_name or runtime_drift_completion_version) and registry_path is None:
        raise ValueError("runtime_drift_completion_name/version require registry_path.")
    if (runtime_drift_completion_name is None) != (runtime_drift_completion_version is None):
        raise ValueError(
            "runtime_drift_completion_name and runtime_drift_completion_version must "
            "be provided together."
        )
    source_path = Path(source)
    payload = _load_json_object(source_path)
    plan = plan_evidence_gaps_from_release_candidate(
        payload,
        source_path=source_path,
        metadata=metadata,
    )
    output = plan.to_dict()
    derived_artifacts = _build_derived_artifacts(
        source_path=source_path,
        gap_plan=output,
        registry_path=registry_path,
        multiple_testing_rerun_json_path=multiple_testing_rerun_json_path,
        multiple_testing_rerun_artifact_manifest_path=multiple_testing_rerun_artifact_manifest_path,
        multiple_testing_rerun_output_dir=multiple_testing_rerun_output_dir,
        multiple_testing_rerun_name=multiple_testing_rerun_name,
        multiple_testing_rerun_version=multiple_testing_rerun_version,
        citation_batch_rerun_json_path=citation_batch_rerun_json_path,
        citation_batch_rerun_artifact_manifest_path=citation_batch_rerun_artifact_manifest_path,
        citation_batch_rerun_output_dir=citation_batch_rerun_output_dir,
        citation_batch_rerun_name=citation_batch_rerun_name,
        citation_batch_rerun_version=citation_batch_rerun_version,
        citation_batch_queue_report_path=citation_batch_queue_report_path,
        citation_batch_scores_path=citation_batch_scores_path,
        citation_batch_blind_spots_path=citation_batch_blind_spots_path,
        citation_batch_source_catalog_paths=citation_batch_source_catalog_paths,
        citation_batch_search_command=citation_batch_search_command,
        citation_batch_controlled_sweep_paths=citation_batch_controlled_sweep_paths,
        citation_batch_query_mode=citation_batch_query_mode,
        stability_rerun_json_path=stability_rerun_json_path,
        stability_rerun_artifact_manifest_path=stability_rerun_artifact_manifest_path,
        stability_rerun_output_dir=stability_rerun_output_dir,
        stability_rerun_name=stability_rerun_name,
        stability_rerun_version=stability_rerun_version,
        stability_score_paths=stability_score_paths,
        stability_seeds=stability_seeds,
        verifier_signal=verifier_signal,
        verifier_claims_path=verifier_claims_path,
        verifier_qa_corpus_path=verifier_qa_corpus_path,
        verifier_state_source_path=verifier_state_source_path,
        verifier_staged_verification=verifier_staged_verification,
        abstention_signals=abstention_signals,
        abstention_rerun_json_path=abstention_rerun_json_path,
        abstention_rerun_artifact_manifest_path=abstention_rerun_artifact_manifest_path,
        abstention_rerun_output_dir=abstention_rerun_output_dir,
        abstention_rerun_name=abstention_rerun_name,
        abstention_rerun_version=abstention_rerun_version,
        abstention_score_paths=abstention_score_paths,
        abstention_profiles=abstention_profiles,
        abstention_signal_groups=abstention_signal_groups,
        abstention_seeds=abstention_seeds,
        abstention_direction=abstention_direction,
        detectability_rerun_json_path=detectability_rerun_json_path,
        detectability_rerun_artifact_manifest_path=detectability_rerun_artifact_manifest_path,
        detectability_rerun_output_dir=detectability_rerun_output_dir,
        detectability_rerun_name=detectability_rerun_name,
        detectability_rerun_version=detectability_rerun_version,
        detectability_score_paths=detectability_score_paths,
        detectability_consistency_signal=detectability_consistency_signal,
        detectability_confidence_signal=detectability_confidence_signal,
        detectability_consistency_direction=detectability_consistency_direction,
        detectability_confidence_direction=detectability_confidence_direction,
        detectability_include_taxonomy_reruns=detectability_include_taxonomy_reruns,
        detectability_taxonomy_pairs=detectability_taxonomy_pairs,
        detectability_cell=detectability_cell,
        detectability_max_records=detectability_max_records,
        frontier_rerun_rollup_completion_json_path=(
            frontier_rerun_rollup_completion_json_path
        ),
        frontier_rerun_rollup_completion_artifact_manifest_path=(
            frontier_rerun_rollup_completion_artifact_manifest_path
        ),
        frontier_rerun_rollup_completion_output_dir=(
            frontier_rerun_rollup_completion_output_dir
        ),
        frontier_rerun_rollup_completion_name=frontier_rerun_rollup_completion_name,
        frontier_rerun_rollup_completion_version=(
            frontier_rerun_rollup_completion_version
        ),
        frontier_rerun_rollup_queue_paths=frontier_rerun_rollup_queue_paths,
        runtime_drift_completion_json_path=runtime_drift_completion_json_path,
        runtime_drift_completion_artifact_manifest_path=(
            runtime_drift_completion_artifact_manifest_path
        ),
        runtime_drift_completion_output_dir=runtime_drift_completion_output_dir,
        runtime_drift_completion_name=runtime_drift_completion_name,
        runtime_drift_completion_version=runtime_drift_completion_version,
        python_executable=python_executable,
    )
    if derived_artifacts:
        output = {**output, "derived_artifacts": derived_artifacts}
    if json_path is not None:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            strict_json_dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if registry_path is not None:
        ArtifactRegistry.load_json(registry_path).record_evidence_gap_plan(
            name=str(name),
            version=str(version),
            path=str(json_path) if json_path is not None else str(source_path),
            metadata={
                "source": str(source_path),
                "status": output["status"],
                "gap_count": output["summary"]["gap_count"],
                "action_count": output["summary"]["action_count"],
            },
        ).save_json()
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = build_release_evidence_gap_plan(
        source=args.source,
        json_path=args.json,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        multiple_testing_rerun_json_path=args.multiple_testing_rerun_json,
        multiple_testing_rerun_artifact_manifest_path=args.multiple_testing_rerun_artifact_manifest,
        multiple_testing_rerun_output_dir=args.multiple_testing_rerun_output_dir,
        multiple_testing_rerun_name=args.multiple_testing_rerun_name,
        multiple_testing_rerun_version=args.multiple_testing_rerun_version,
        citation_batch_rerun_json_path=args.citation_batch_rerun_json,
        citation_batch_rerun_artifact_manifest_path=args.citation_batch_rerun_artifact_manifest,
        citation_batch_rerun_output_dir=args.citation_batch_rerun_output_dir,
        citation_batch_rerun_name=args.citation_batch_rerun_name,
        citation_batch_rerun_version=args.citation_batch_rerun_version,
        citation_batch_queue_report_path=args.citation_batch_queue,
        citation_batch_scores_path=args.citation_batch_scores,
        citation_batch_blind_spots_path=args.citation_batch_blind_spots,
        citation_batch_source_catalog_paths=tuple(args.citation_batch_source_catalog or ()),
        citation_batch_search_command=args.citation_batch_search_command,
        citation_batch_controlled_sweep_paths=tuple(args.citation_batch_controlled_sweep or ()),
        citation_batch_query_mode=args.citation_batch_query_mode,
        stability_rerun_json_path=args.stability_rerun_json,
        stability_rerun_artifact_manifest_path=args.stability_rerun_artifact_manifest,
        stability_rerun_output_dir=args.stability_rerun_output_dir,
        stability_rerun_name=args.stability_rerun_name,
        stability_rerun_version=args.stability_rerun_version,
        stability_score_paths=tuple(args.stability_scores or ()),
        stability_seeds=args.stability_seeds,
        verifier_signal=args.verifier_signal,
        verifier_claims_path=args.verifier_claims,
        verifier_qa_corpus_path=args.verifier_qa_corpus,
        verifier_state_source_path=args.verifier_state_source,
        verifier_staged_verification=bool(args.verifier_staged_verification),
        abstention_signals=_parse_csv(args.abstention_signals),
        abstention_rerun_json_path=args.abstention_rerun_json,
        abstention_rerun_artifact_manifest_path=args.abstention_rerun_artifact_manifest,
        abstention_rerun_output_dir=args.abstention_rerun_output_dir,
        abstention_rerun_name=args.abstention_rerun_name,
        abstention_rerun_version=args.abstention_rerun_version,
        abstention_score_paths=tuple(args.abstention_scores or ()),
        abstention_profiles=_parse_csv(args.abstention_profiles),
        abstention_signal_groups=_parse_csv(args.abstention_signal_groups),
        abstention_seeds=args.abstention_seeds,
        abstention_direction=args.abstention_direction,
        detectability_rerun_json_path=args.detectability_rerun_json,
        detectability_rerun_artifact_manifest_path=args.detectability_rerun_artifact_manifest,
        detectability_rerun_output_dir=args.detectability_rerun_output_dir,
        detectability_rerun_name=args.detectability_rerun_name,
        detectability_rerun_version=args.detectability_rerun_version,
        detectability_score_paths=tuple(args.detectability_scores or ()),
        detectability_consistency_signal=args.detectability_consistency_signal,
        detectability_confidence_signal=args.detectability_confidence_signal,
        detectability_consistency_direction=args.detectability_consistency_direction,
        detectability_confidence_direction=args.detectability_confidence_direction,
        detectability_include_taxonomy_reruns=bool(args.detectability_include_taxonomy_reruns),
        detectability_taxonomy_pairs=tuple(args.detectability_taxonomy_pair or ()),
        detectability_cell=args.detectability_cell,
        detectability_max_records=args.detectability_max_records,
        frontier_rerun_rollup_completion_json_path=(
            args.frontier_rerun_rollup_completion_json
        ),
        frontier_rerun_rollup_completion_artifact_manifest_path=(
            args.frontier_rerun_rollup_completion_artifact_manifest
        ),
        frontier_rerun_rollup_completion_output_dir=(
            args.frontier_rerun_rollup_completion_output_dir
        ),
        frontier_rerun_rollup_completion_name=(
            args.frontier_rerun_rollup_completion_name
        ),
        frontier_rerun_rollup_completion_version=(
            args.frontier_rerun_rollup_completion_version
        ),
        frontier_rerun_rollup_queue_paths=tuple(args.frontier_rerun_rollup_queue or ()),
        runtime_drift_completion_json_path=args.runtime_drift_completion_json,
        runtime_drift_completion_artifact_manifest_path=(
            args.runtime_drift_completion_artifact_manifest
        ),
        runtime_drift_completion_output_dir=args.runtime_drift_completion_output_dir,
        runtime_drift_completion_name=args.runtime_drift_completion_name,
        runtime_drift_completion_version=args.runtime_drift_completion_version,
        python_executable=args.python,
    )
    summary = payload["summary"]
    print(
        "release_evidence_gap_plan="
        f"{payload['status']} "
        f"gaps={summary['gap_count']} "
        f"actions={summary['action_count']} "
        f"missing_metrics={summary['missing_metric_count']}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a structured next-evidence plan from a blocked release report"
    )
    parser.add_argument("--source", required=True, help="release comparison or registry workflow JSON")
    parser.add_argument("--json", default=None, help="optional output JSON path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--metadata", action="append", default=[], help="KEY=VALUE metadata; repeatable")
    parser.add_argument(
        "--multiple-testing-rerun-json",
        default=None,
        help="optional output JSON path for a derived frontier multiple-testing rerun queue",
    )
    parser.add_argument(
        "--multiple-testing-rerun-artifact-manifest",
        default=None,
        help="optional artifact manifest path for the derived multiple-testing rerun queue",
    )
    parser.add_argument(
        "--multiple-testing-rerun-output-dir",
        default=None,
        help="optional root directory for derived per-cell rerun outputs",
    )
    parser.add_argument(
        "--multiple-testing-rerun-name",
        default=None,
        help="optional registry name for the derived multiple-testing rerun queue",
    )
    parser.add_argument(
        "--multiple-testing-rerun-version",
        default=None,
        help="optional registry version for the derived multiple-testing rerun queue",
    )
    parser.add_argument(
        "--citation-batch-rerun-json",
        default=None,
        help="optional output JSON path for a derived citation batch rerun queue",
    )
    parser.add_argument(
        "--citation-batch-rerun-artifact-manifest",
        default=None,
        help="optional artifact manifest path for the derived citation batch rerun queue",
    )
    parser.add_argument(
        "--citation-batch-rerun-output-dir",
        default=None,
        help="optional root directory for derived citation batch rerun outputs",
    )
    parser.add_argument("--citation-batch-rerun-name", default=None, help="optional registry name for the queue")
    parser.add_argument("--citation-batch-rerun-version", default=None, help="optional registry version for the queue")
    parser.add_argument("--citation-batch-queue", default=None, help="unresolved evidence queue for generated commands")
    parser.add_argument("--citation-batch-scores", default=None, help="score dump for generated commands")
    parser.add_argument("--citation-batch-blind-spots", default=None, help="blind-spot rows for generated commands")
    parser.add_argument(
        "--citation-batch-source-catalog",
        action="append",
        default=[],
        help="source-family catalog for generated commands; repeatable",
    )
    parser.add_argument(
        "--citation-batch-search-command",
        default=None,
        help="external search command with {input}/{output}",
    )
    parser.add_argument(
        "--citation-batch-controlled-sweep",
        action="append",
        default=[],
        help="controlled sweep report for generated citation commands; repeatable",
    )
    parser.add_argument("--citation-batch-query-mode", default="claim_entity")
    parser.add_argument(
        "--stability-rerun-json",
        default=None,
        help="optional output JSON path for a derived verifier/abstention stability rerun queue",
    )
    parser.add_argument(
        "--stability-rerun-artifact-manifest",
        default=None,
        help="optional artifact manifest path for the derived stability rerun queue",
    )
    parser.add_argument(
        "--stability-rerun-output-dir",
        default=None,
        help="optional root directory for derived stability rerun outputs",
    )
    parser.add_argument("--stability-rerun-name", default=None, help="optional registry name for the queue")
    parser.add_argument("--stability-rerun-version", default=None, help="optional registry version for the queue")
    parser.add_argument("--stability-scores", action="append", default=[], help="name=score_dump path; repeatable")
    parser.add_argument("--stability-seeds", default=None, help="comma-separated seeds for stability reruns")
    parser.add_argument("--verifier-signal", default=None)
    parser.add_argument("--verifier-claims", default=None)
    parser.add_argument("--verifier-qa-corpus", default=None)
    parser.add_argument("--verifier-state-source", default=None)
    parser.add_argument(
        "--verifier-staged-verification",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--abstention-signals", default=None)
    parser.add_argument(
        "--abstention-rerun-json",
        default=None,
        help="optional output JSON path for a derived abstention experiment rerun queue",
    )
    parser.add_argument(
        "--abstention-rerun-artifact-manifest",
        default=None,
        help="optional artifact manifest path for the derived abstention queue",
    )
    parser.add_argument(
        "--abstention-rerun-output-dir",
        default=None,
        help="optional root directory for derived abstention experiment outputs",
    )
    parser.add_argument("--abstention-rerun-name", default=None, help="optional registry name for the queue")
    parser.add_argument("--abstention-rerun-version", default=None, help="optional registry version for the queue")
    parser.add_argument("--abstention-scores", action="append", default=[], help="name=score_dump path; repeatable")
    parser.add_argument("--abstention-profiles", default=None, help="comma-separated abstention experiment profiles")
    parser.add_argument(
        "--abstention-signal-groups",
        default=None,
        help="comma-separated signal groups or signal+signal",
    )
    parser.add_argument(
        "--abstention-seeds",
        default=None,
        help="comma-separated seeds for abstention experiment reruns",
    )
    parser.add_argument("--abstention-direction", choices=("higher", "lower"), default=None)
    parser.add_argument(
        "--detectability-rerun-json",
        default=None,
        help="optional output JSON path for a derived detectability rerun/audit queue",
    )
    parser.add_argument(
        "--detectability-rerun-artifact-manifest",
        default=None,
        help="optional artifact manifest path for the derived detectability queue",
    )
    parser.add_argument(
        "--detectability-rerun-output-dir",
        default=None,
        help="optional root directory for derived detectability outputs",
    )
    parser.add_argument("--detectability-rerun-name", default=None, help="optional registry name for the queue")
    parser.add_argument("--detectability-rerun-version", default=None, help="optional registry version for the queue")
    parser.add_argument("--detectability-scores", action="append", default=[], help="name=score_dump path; repeatable")
    parser.add_argument("--detectability-consistency-signal", default=None)
    parser.add_argument("--detectability-confidence-signal", default=None)
    parser.add_argument("--detectability-consistency-direction", choices=("higher", "lower"), default="higher")
    parser.add_argument("--detectability-confidence-direction", choices=("higher", "lower"), default="higher")
    parser.add_argument(
        "--detectability-include-taxonomy-reruns",
        action="store_true",
        help="append taxonomy rerun commands to the derived detectability queue",
    )
    parser.add_argument(
        "--detectability-taxonomy-pair",
        action="append",
        default=[],
        help=(
            "detectability taxonomy rerun pair as consistency:confidence or "
            "consistency:confidence:consistency_direction:confidence_direction; repeatable"
        ),
    )
    parser.add_argument("--detectability-cell", default="entrenched")
    parser.add_argument("--detectability-max-records", type=int, default=100)
    parser.add_argument(
        "--frontier-rerun-rollup-completion-json",
        default=None,
        help="optional output JSON path for a frontier rerun-rollup completion plan",
    )
    parser.add_argument(
        "--frontier-rerun-rollup-completion-artifact-manifest",
        default=None,
        help="optional artifact manifest path for the rerun-rollup completion plan",
    )
    parser.add_argument(
        "--frontier-rerun-rollup-completion-output-dir",
        default=None,
        help="optional root directory for derived rerun-rollup outputs",
    )
    parser.add_argument(
        "--frontier-rerun-rollup-completion-name",
        default=None,
        help="optional registry name for the rerun-rollup completion plan",
    )
    parser.add_argument(
        "--frontier-rerun-rollup-completion-version",
        default=None,
        help="optional registry version for the rerun-rollup completion plan",
    )
    parser.add_argument(
        "--frontier-rerun-rollup-queue",
        action="append",
        default=[],
        help="TRACK=PATH queue mapping for rollup commands; repeatable",
    )
    parser.add_argument(
        "--runtime-drift-completion-json",
        default=None,
        help="optional output JSON path for a runtime-drift evidence completion plan",
    )
    parser.add_argument(
        "--runtime-drift-completion-artifact-manifest",
        default=None,
        help="optional artifact manifest path for the runtime-drift completion plan",
    )
    parser.add_argument(
        "--runtime-drift-completion-output-dir",
        default=None,
        help="optional root directory for bound runtime-drift completion outputs",
    )
    parser.add_argument(
        "--runtime-drift-completion-name",
        default=None,
        help="optional registry name for the runtime-drift completion plan",
    )
    parser.add_argument(
        "--runtime-drift-completion-version",
        default=None,
        help="optional registry version for the runtime-drift completion plan",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for generated rerun commands")
    run(parser.parse_args(argv))


def _build_derived_artifacts(
    *,
    source_path: Path,
    gap_plan: Mapping[str, Any],
    registry_path: str | Path | None,
    multiple_testing_rerun_json_path: str | Path | None,
    multiple_testing_rerun_artifact_manifest_path: str | Path | None,
    multiple_testing_rerun_output_dir: str | Path | None,
    multiple_testing_rerun_name: str | None,
    multiple_testing_rerun_version: str | None,
    citation_batch_rerun_json_path: str | Path | None,
    citation_batch_rerun_artifact_manifest_path: str | Path | None,
    citation_batch_rerun_output_dir: str | Path | None,
    citation_batch_rerun_name: str | None,
    citation_batch_rerun_version: str | None,
    citation_batch_queue_report_path: str | Path | None,
    citation_batch_scores_path: str | Path | None,
    citation_batch_blind_spots_path: str | Path | None,
    citation_batch_source_catalog_paths: Sequence[str | Path],
    citation_batch_search_command: str | None,
    citation_batch_controlled_sweep_paths: Sequence[str | Path],
    citation_batch_query_mode: str,
    stability_rerun_json_path: str | Path | None,
    stability_rerun_artifact_manifest_path: str | Path | None,
    stability_rerun_output_dir: str | Path | None,
    stability_rerun_name: str | None,
    stability_rerun_version: str | None,
    stability_score_paths: Sequence[str | Path],
    stability_seeds: str | None,
    verifier_signal: str | None,
    verifier_claims_path: str | Path | None,
    verifier_qa_corpus_path: str | Path | None,
    verifier_state_source_path: str | Path | None,
    verifier_staged_verification: bool,
    abstention_signals: Sequence[str],
    abstention_rerun_json_path: str | Path | None,
    abstention_rerun_artifact_manifest_path: str | Path | None,
    abstention_rerun_output_dir: str | Path | None,
    abstention_rerun_name: str | None,
    abstention_rerun_version: str | None,
    abstention_score_paths: Sequence[str | Path],
    abstention_profiles: Sequence[str],
    abstention_signal_groups: Sequence[str],
    abstention_seeds: str | None,
    abstention_direction: str | None,
    detectability_rerun_json_path: str | Path | None,
    detectability_rerun_artifact_manifest_path: str | Path | None,
    detectability_rerun_output_dir: str | Path | None,
    detectability_rerun_name: str | None,
    detectability_rerun_version: str | None,
    detectability_score_paths: Sequence[str | Path],
    detectability_consistency_signal: str | None,
    detectability_confidence_signal: str | None,
    detectability_consistency_direction: str,
    detectability_confidence_direction: str,
    detectability_include_taxonomy_reruns: bool,
    detectability_taxonomy_pairs: Sequence[str],
    detectability_cell: str,
    detectability_max_records: int,
    frontier_rerun_rollup_completion_json_path: str | Path | None,
    frontier_rerun_rollup_completion_artifact_manifest_path: str | Path | None,
    frontier_rerun_rollup_completion_output_dir: str | Path | None,
    frontier_rerun_rollup_completion_name: str | None,
    frontier_rerun_rollup_completion_version: str | None,
    frontier_rerun_rollup_queue_paths: Sequence[str | Path],
    runtime_drift_completion_json_path: str | Path | None,
    runtime_drift_completion_artifact_manifest_path: str | Path | None,
    runtime_drift_completion_output_dir: str | Path | None,
    runtime_drift_completion_name: str | None,
    runtime_drift_completion_version: str | None,
    python_executable: str,
) -> dict[str, Any]:
    derived: dict[str, Any] = {}
    if multiple_testing_rerun_json_path is not None:
        rerun_payload = build_frontier_multiple_testing_rerun_queue(
            source=source_path,
            json_path=multiple_testing_rerun_json_path,
            artifact_manifest_path=multiple_testing_rerun_artifact_manifest_path,
            registry_path=None
            if multiple_testing_rerun_name is None or multiple_testing_rerun_version is None
            else registry_path,
            name=multiple_testing_rerun_name,
            version=multiple_testing_rerun_version,
            output_dir=multiple_testing_rerun_output_dir,
            python_executable=python_executable,
        )
        summary = rerun_payload["summary"]
        derived["frontier_multiple_testing_rerun_queue"] = {
            "path": str(multiple_testing_rerun_json_path),
            "artifact_manifest": None
            if multiple_testing_rerun_artifact_manifest_path is None
            else str(multiple_testing_rerun_artifact_manifest_path),
            "status": rerun_payload["status"],
            "blocked_cell_count": summary["blocked_cell_count"],
            "command_count": summary["command_count"],
            "missing_command_count": summary["missing_command_count"],
        }
    if citation_batch_rerun_json_path is not None:
        citation_payload = build_citation_batch_evidence_rerun_queue(
            source=source_path,
            json_path=citation_batch_rerun_json_path,
            artifact_manifest_path=citation_batch_rerun_artifact_manifest_path,
            registry_path=None
            if citation_batch_rerun_name is None or citation_batch_rerun_version is None
            else registry_path,
            name=citation_batch_rerun_name,
            version=citation_batch_rerun_version,
            output_dir=citation_batch_rerun_output_dir,
            queue_report_path=citation_batch_queue_report_path,
            scores_path=citation_batch_scores_path,
            blind_spots_path=citation_batch_blind_spots_path,
            source_catalog_paths=citation_batch_source_catalog_paths,
            search_command=citation_batch_search_command,
            controlled_sweep_paths=citation_batch_controlled_sweep_paths,
            query_mode=citation_batch_query_mode,
            python_executable=python_executable,
        )
        summary = citation_payload["summary"]
        derived["citation_batch_evidence_rerun_queue"] = {
            "path": str(citation_batch_rerun_json_path),
            "artifact_manifest": None
            if citation_batch_rerun_artifact_manifest_path is None
            else str(citation_batch_rerun_artifact_manifest_path),
            "status": citation_payload["status"],
            "blocked_batch_count": summary["blocked_batch_count"],
            "command_count": summary["command_count"],
            "missing_command_count": summary["missing_command_count"],
        }
    if stability_rerun_json_path is not None:
        stability_payload = build_frontier_stability_evidence_rerun_queue(
            source=source_path,
            json_path=stability_rerun_json_path,
            artifact_manifest_path=stability_rerun_artifact_manifest_path,
            registry_path=None
            if stability_rerun_name is None or stability_rerun_version is None
            else registry_path,
            name=stability_rerun_name,
            version=stability_rerun_version,
            output_dir=stability_rerun_output_dir,
            score_paths=stability_score_paths,
            seeds=stability_seeds,
            verifier_signal=verifier_signal,
            verifier_claims_path=verifier_claims_path,
            verifier_qa_corpus_path=verifier_qa_corpus_path,
            verifier_state_source_path=verifier_state_source_path,
            verifier_staged_verification=verifier_staged_verification,
            abstention_signals=abstention_signals,
            python_executable=python_executable,
        )
        summary = stability_payload["summary"]
        derived["frontier_stability_evidence_rerun_queue"] = {
            "path": str(stability_rerun_json_path),
            "artifact_manifest": None
            if stability_rerun_artifact_manifest_path is None
            else str(stability_rerun_artifact_manifest_path),
            "status": stability_payload["status"],
            "blocked_track_count": summary["blocked_track_count"],
            "command_count": summary["command_count"],
            "missing_command_count": summary["missing_command_count"],
        }
    if abstention_rerun_json_path is not None:
        abstention_payload = build_frontier_abstention_evidence_rerun_queue(
            source=source_path,
            json_path=abstention_rerun_json_path,
            artifact_manifest_path=abstention_rerun_artifact_manifest_path,
            registry_path=None
            if abstention_rerun_name is None or abstention_rerun_version is None
            else registry_path,
            name=abstention_rerun_name,
            version=abstention_rerun_version,
            output_dir=abstention_rerun_output_dir,
            score_paths=abstention_score_paths,
            profiles=abstention_profiles or (
                "baseline",
                "alpha_0p05",
                "alpha_0p2",
                "selective_accuracy",
                "retention",
            ),
            signal_groups=abstention_signal_groups or (
                "recommended",
                "all",
                "geometry",
                "uncertainty",
            ),
            seeds=abstention_seeds,
            direction=abstention_direction,
            python_executable=python_executable,
        )
        summary = abstention_payload["summary"]
        derived["frontier_abstention_evidence_rerun_queue"] = {
            "path": str(abstention_rerun_json_path),
            "artifact_manifest": None
            if abstention_rerun_artifact_manifest_path is None
            else str(abstention_rerun_artifact_manifest_path),
            "status": abstention_payload["status"],
            "blocked_run_count": summary["blocked_run_count"],
            "entry_count": summary["entry_count"],
            "command_count": summary["command_count"],
            "missing_command_count": summary["missing_command_count"],
        }
    if detectability_rerun_json_path is not None:
        detectability_payload = build_frontier_detectability_evidence_rerun_queue(
            source=source_path,
            json_path=detectability_rerun_json_path,
            artifact_manifest_path=detectability_rerun_artifact_manifest_path,
            registry_path=None
            if detectability_rerun_name is None or detectability_rerun_version is None
            else registry_path,
            name=detectability_rerun_name,
            version=detectability_rerun_version,
            output_dir=detectability_rerun_output_dir,
            score_paths=detectability_score_paths,
            consistency_signal=detectability_consistency_signal,
            confidence_signal=detectability_confidence_signal,
            consistency_direction=detectability_consistency_direction,
            confidence_direction=detectability_confidence_direction,
            include_taxonomy_reruns=detectability_include_taxonomy_reruns,
            taxonomy_signal_pairs=detectability_taxonomy_pairs,
            cell=detectability_cell,
            max_records=detectability_max_records,
            python_executable=python_executable,
        )
        summary = detectability_payload["summary"]
        derived["frontier_detectability_evidence_rerun_queue"] = {
            "path": str(detectability_rerun_json_path),
            "artifact_manifest": None
            if detectability_rerun_artifact_manifest_path is None
            else str(detectability_rerun_artifact_manifest_path),
            "status": detectability_payload["status"],
            "blocked_run_count": summary["blocked_run_count"],
            "entry_count": summary["entry_count"],
            "command_count": summary["command_count"],
            "missing_command_count": summary["missing_command_count"],
        }
    if frontier_rerun_rollup_completion_json_path is not None:
        completion_payload = build_frontier_rerun_rollup_completion_plan(
            source=source_path,
            json_path=frontier_rerun_rollup_completion_json_path,
            artifact_manifest_path=frontier_rerun_rollup_completion_artifact_manifest_path,
            registry_path=None
            if (
                frontier_rerun_rollup_completion_name is None
                or frontier_rerun_rollup_completion_version is None
            )
            else registry_path,
            name=frontier_rerun_rollup_completion_name,
            version=frontier_rerun_rollup_completion_version,
            output_dir=frontier_rerun_rollup_completion_output_dir,
            queue_paths=frontier_rerun_rollup_queue_paths,
            python_executable=python_executable,
        )
        summary = completion_payload["summary"]
        derived["frontier_rerun_rollup_completion_plan"] = {
            "path": str(frontier_rerun_rollup_completion_json_path),
            "artifact_manifest": None
            if frontier_rerun_rollup_completion_artifact_manifest_path is None
            else str(frontier_rerun_rollup_completion_artifact_manifest_path),
            "status": completion_payload["status"],
            "entry_count": summary["entry_count"],
            "command_count": summary["command_count"],
            "missing_queue_count": summary["missing_queue_count"],
            "unsupported_workflow_count": summary["unsupported_workflow_count"],
        }
    if runtime_drift_completion_json_path is not None:
        runtime_payload = build_runtime_drift_evidence_completion_plan(
            source=gap_plan,
            json_path=runtime_drift_completion_json_path,
            artifact_manifest_path=runtime_drift_completion_artifact_manifest_path,
            registry_path=None
            if runtime_drift_completion_name is None or runtime_drift_completion_version is None
            else registry_path,
            name=runtime_drift_completion_name,
            version=runtime_drift_completion_version,
            output_dir=runtime_drift_completion_output_dir,
            python_executable=python_executable,
            metadata={"release_evidence_gap_plan_source": str(source_path)},
        )
        summary = runtime_payload["summary"]
        derived["runtime_drift_evidence_completion_plan"] = {
            "path": str(runtime_drift_completion_json_path),
            "artifact_manifest": None
            if runtime_drift_completion_artifact_manifest_path is None
            else str(runtime_drift_completion_artifact_manifest_path),
            "status": runtime_payload["status"],
            "entry_count": summary["entry_count"],
            "command_template_count": summary["command_template_count"],
            "missing_input_count": summary["missing_input_count"],
            "expected_output_count": summary["expected_output_count"],
        }
    return derived


def build_runtime_drift_evidence_completion_plan(
    *,
    source: str | Path | Mapping[str, Any],
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    output_dir: str | Path | None = None,
    python_executable: str = sys.executable,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a command-template plan for closing runtime-drift evidence blockers."""
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    source_path: Path | None = None
    if isinstance(source, Mapping):
        payload: Mapping[str, Any] = dict(source)
    else:
        source_path = Path(source)
        payload = _load_json_object(source_path)
    if not _mapping_sequence(payload.get("actions", ())):
        plan = plan_evidence_gaps_from_release_candidate(
            payload,
            source_path=source_path,
            metadata=metadata,
        )
        payload = plan.to_dict()
    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    completion_root = _runtime_drift_completion_root(
        source_path=source_path,
        output_path=output_path,
        output_dir=output_dir,
    )
    missing_metrics_by_action = _runtime_drift_missing_metrics_by_action(payload)
    entries = tuple(
        _runtime_drift_completion_entry(
            action,
            index=index,
            completion_root=completion_root,
            missing_metrics=missing_metrics_by_action.get(
                str(action.get("action_id", "")),
                (),
            ),
        )
        for index, action in enumerate(_runtime_drift_completion_actions(payload), start=1)
    )
    summary = _runtime_drift_completion_summary(entries)
    status = _runtime_drift_completion_status(summary)
    output = {
        "schema_version": 1,
        "workflow": RUNTIME_DRIFT_COMPLETION_WORKFLOW,
        "status": status,
        "source": {
            "path": None if source_path is None else str(source_path),
            "workflow": payload.get("workflow"),
            "status": payload.get("status"),
            "summary": _mapping(payload.get("summary")),
        },
        "summary": summary,
        "paths": {
            "completion_plan": None if output_path is None else str(output_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "completion_output_dir": str(completion_root),
        },
        "config": {
            "python_executable": python_executable,
            "command_binding": "templates_require_input_binding",
        },
        "entries": entries,
        "metadata": dict(metadata or {}),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            strict_json_dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest = None
    if manifest_path is not None:
        manifest = _write_runtime_drift_completion_manifest(
            source_path=source_path,
            output_path=output_path,
            manifest_path=manifest_path,
            payload=output,
            metadata=metadata or {},
        )
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output_path if output_path is not None else source_path,
            metadata={
                "workflow": RUNTIME_DRIFT_COMPLETION_WORKFLOW,
                "status": status,
                "source": None if source_path is None else str(source_path),
                "artifact_manifest": None if manifest_path is None else str(manifest_path),
                "entry_count": summary["entry_count"],
                "command_template_count": summary["command_template_count"],
                "missing_input_count": summary["missing_input_count"],
                "expected_output_count": summary["expected_output_count"],
                "routes": summary["routes"],
                "manifest_summary": {} if manifest is None else manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return output


def _runtime_drift_completion_root(
    *,
    source_path: Path | None,
    output_path: Path | None,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if output_path is not None:
        return output_path.parent / "runtime-drift-completion"
    if source_path is not None:
        return source_path.parent / "runtime-drift-completion"
    return Path("runtime-drift-completion")


def _runtime_drift_completion_actions(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    actions = tuple(
        action
        for action in _mapping_sequence(payload.get("actions", ()))
        if _is_runtime_drift_completion_action(action)
    )
    return tuple(
        sorted(
            actions,
            key=lambda action: (
                -int(action.get("priority", 0)),
                str(action.get("action_id", "")),
            ),
        )
    )


def _is_runtime_drift_completion_action(action: Mapping[str, Any]) -> bool:
    routes = set(_string_tuple(action.get("evidence_routes", ())))
    metadata = _mapping(action.get("metadata"))
    if routes & RUNTIME_DRIFT_COMPLETION_ROUTES:
        return True
    if any(key in metadata for key in RUNTIME_DRIFT_COMPLETION_METADATA_KEYS):
        return True
    if any(str(value).startswith("product_runtime_") for value in metadata.values()):
        return True
    return any(
        "product_runtime" in command or "product_trace" in command
        for command in _string_tuple(action.get("suggested_commands", ()))
    )


def _runtime_drift_completion_entry(
    action: Mapping[str, Any],
    *,
    index: int,
    completion_root: Path,
    missing_metrics: Sequence[str],
) -> dict[str, Any]:
    metadata = _mapping(action.get("metadata"))
    action_id = str(action.get("action_id") or f"runtime-drift-action-{index:04d}")
    command_templates = _string_tuple(action.get("suggested_commands", ()))
    required_inputs = _string_tuple(metadata.get("required_inputs", ()))
    missing_inputs = _runtime_drift_missing_inputs(
        required_inputs=required_inputs,
        command_templates=command_templates,
    )
    command_status = "needs_inputs" if missing_inputs else "ready"
    if not command_templates:
        command_status = "missing_command_templates"
    required_metrics = _runtime_drift_required_metrics(metadata)
    closure_outputs = _string_tuple(metadata.get("closure_outputs", ()))
    routes = _string_tuple(action.get("evidence_routes", ()))
    bound_output_dir = completion_root / _slug(action_id)
    return {
        "entry_id": f"runtime-drift-{index:04d}",
        "action_id": action_id,
        "title": str(action.get("title") or action_id),
        "action_type": str(action.get("action_type") or "workflow"),
        "priority": int(action.get("priority", 0)),
        "command_status": command_status,
        "evidence_routes": routes,
        "source_gap_ids": _string_tuple(action.get("source_gap_ids", ())),
        "missing_metrics": tuple(dict.fromkeys(str(item) for item in missing_metrics if str(item))),
        "required_inputs": required_inputs,
        "missing_inputs": missing_inputs,
        "required_metrics": required_metrics,
        "closure_outputs": closure_outputs,
        "scripts": _runtime_drift_scripts(metadata, command_templates),
        "command_templates": command_templates,
        "bound_output_dir": str(bound_output_dir),
        "binding_hints": _runtime_drift_binding_hints(
            action_id=action_id,
            bound_output_dir=bound_output_dir,
            required_inputs=required_inputs,
            missing_inputs=missing_inputs,
            closure_outputs=closure_outputs,
            command_templates=command_templates,
        ),
        "metadata": {
            "risk_control_method": metadata.get("risk_control_method"),
            "default_gate_thresholds": metadata.get("default_gate_thresholds", {}),
            "workflow_keys": _runtime_drift_workflow_keys(metadata),
        },
    }


def _runtime_drift_binding_hints(
    *,
    action_id: str,
    bound_output_dir: Path,
    required_inputs: Sequence[str],
    missing_inputs: Sequence[str],
    closure_outputs: Sequence[str],
    command_templates: Sequence[str],
) -> dict[str, Any]:
    input_names = tuple(dict.fromkeys((*required_inputs, *missing_inputs)))
    return {
        "action_id": action_id,
        "bound_output_dir": str(bound_output_dir),
        "command_templates_need_binding": any("..." in command for command in command_templates),
        "input_bindings": tuple(
            {
                "name": name,
                "placeholder": "..." if name == "bound_command_template_values" else f"<{name}>",
                "required": True,
                "status": "unbound",
            }
            for name in input_names
        ),
        "output_bindings": tuple(
            {
                "name": output,
                "path": str(bound_output_dir / f"{_slug(output)}.json"),
                "status": "planned",
            }
            for output in closure_outputs
        ),
    }


def _runtime_drift_missing_inputs(
    *,
    required_inputs: Sequence[str],
    command_templates: Sequence[str],
) -> tuple[str, ...]:
    missing = list(required_inputs)
    if any("..." in command for command in command_templates):
        missing.append("bound_command_template_values")
    return tuple(dict.fromkeys(str(item) for item in missing if str(item)))


def _runtime_drift_required_metrics(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    metrics: list[str] = []
    for key in (
        "required_trace_metrics",
        "required_runtime_metrics",
        "required_route_metrics",
        "observed_track_metrics",
    ):
        metrics.extend(_string_tuple(metadata.get(key, ())))
    return tuple(dict.fromkeys(metrics))


def _runtime_drift_workflow_keys(metadata: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in metadata.items()
        if key.endswith("_workflow") and isinstance(value, str) and value
    }


def _runtime_drift_scripts(
    metadata: Mapping[str, Any],
    command_templates: Sequence[str],
) -> tuple[str, ...]:
    scripts: list[str] = []
    for key in sorted(metadata):
        value = metadata.get(key)
        if key.endswith("_script") and isinstance(value, str) and value:
            scripts.append(value)
    for command in command_templates:
        first = command.strip().split(" ", 1)[0]
        if first.endswith(".py"):
            scripts.append(first)
    return tuple(dict.fromkeys(scripts))


def _runtime_drift_missing_metrics_by_action(
    payload: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    metrics_by_action: dict[str, list[str]] = {}
    for gap in _mapping_sequence(payload.get("gaps", ())):
        missing_metrics = _string_tuple(gap.get("missing_metrics", ()))
        if not missing_metrics:
            continue
        for action_id in _string_tuple(gap.get("recommended_action_ids", ())):
            metrics_by_action.setdefault(action_id, []).extend(missing_metrics)
    return {
        action_id: tuple(dict.fromkeys(metrics))
        for action_id, metrics in metrics_by_action.items()
    }


def _runtime_drift_completion_summary(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("command_status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        for route in _string_tuple(entry.get("evidence_routes", ())):
            route_counts[route] = route_counts.get(route, 0) + 1
    return {
        "entry_count": len(entries),
        "command_template_count": sum(
            len(_string_tuple(entry.get("command_templates", ()))) for entry in entries
        ),
        "missing_input_count": sum(
            len(_string_tuple(entry.get("missing_inputs", ()))) for entry in entries
        ),
        "missing_command_template_count": status_counts.get("missing_command_templates", 0),
        "command_status_counts": dict(sorted(status_counts.items())),
        "routes": tuple(sorted(route_counts)),
        "route_counts": dict(sorted(route_counts.items())),
        "missing_metric_count": sum(
            len(_string_tuple(entry.get("missing_metrics", ()))) for entry in entries
        ),
        "expected_output_count": sum(
            len(_mapping_sequence(_mapping(entry.get("binding_hints")).get("output_bindings", ())))
            for entry in entries
        ),
    }


def _runtime_drift_completion_status(summary: Mapping[str, Any]) -> str:
    if summary["entry_count"] == 0:
        return "empty"
    if summary["missing_input_count"] or summary["missing_command_template_count"]:
        return "needs_inputs"
    return "ready"


def _write_runtime_drift_completion_manifest(
    *,
    source_path: Path | None,
    output_path: Path | None,
    manifest_path: Path,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "runtime_drift_evidence_completion_plan": output_path,
    }
    if source_path is not None:
        artifacts["source"] = source_path
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "plan_release_evidence_gaps",
            "workflow": RUNTIME_DRIFT_COMPLETION_WORKFLOW,
            "status": payload.get("status"),
            "entry_count": _nested_value(payload, "summary", "entry_count"),
            "command_template_count": _nested_value(
                payload,
                "summary",
                "command_template_count",
            ),
            "missing_input_count": _nested_value(payload, "summary", "missing_input_count"),
            **dict(metadata),
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-") or "item"


def build_frontier_rerun_rollup_completion_plan(
    *,
    source: str | Path,
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    output_dir: str | Path | None = None,
    queue_paths: Sequence[str | Path] = (),
    python_executable: str = sys.executable,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a command plan that completes blocked frontier rerun rollups."""
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    source_path = Path(source)
    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    rollup_root = (
        Path(output_dir)
        if output_dir is not None
        else source_path.parent / "frontier-rerun-rollup-completion"
    )
    payload = _load_json_object(source_path)
    queue_map = _frontier_rerun_rollup_queue_map(queue_paths)
    decisions = _frontier_rerun_rollup_completion_decisions(payload)
    entries = tuple(
        _frontier_rerun_rollup_completion_entry(
            decision,
            rollup_root=rollup_root,
            queue_map=queue_map,
            python_executable=python_executable,
        )
        for decision in decisions
    )
    summary = {
        "entry_count": len(entries),
        "command_count": sum(1 for entry in entries if entry["command_status"] == "ready"),
        "missing_queue_count": sum(
            1 for entry in entries if entry["command_status"] == "missing_queue"
        ),
        "unsupported_workflow_count": sum(
            1 for entry in entries if entry["command_status"] == "unsupported_workflow"
        ),
        "blocked_rollup_count": sum(
            1 for entry in entries if entry["source_status"] != "promote"
        ),
        "tracks": tuple(sorted({str(entry["track"]) for entry in entries if entry.get("track")})),
    }
    status = _frontier_rerun_rollup_completion_status(summary)
    output = {
        "schema_version": 1,
        "workflow": FRONTIER_RERUN_ROLLUP_COMPLETION_WORKFLOW,
        "status": status,
        "source": {
            "path": str(source_path),
            "workflow": payload.get("workflow"),
            "status": payload.get("status"),
            "decision_status": _nested_value(payload, "decision", "status"),
            "frontier_rerun_rollup_track_status": _nested_value(
                payload,
                "decision",
                "frontier_rerun_rollup_track_status",
            ),
        },
        "summary": summary,
        "paths": {
            "completion_plan": None if output_path is None else str(output_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "rollup_output_dir": str(rollup_root),
        },
        "config": {
            "queues": queue_map,
            "python_executable": python_executable,
        },
        "entries": entries,
        "metadata": dict(metadata or {}),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            strict_json_dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest = None
    if manifest_path is not None:
        manifest = _write_frontier_rerun_rollup_completion_manifest(
            source_path=source_path,
            output_path=output_path,
            manifest_path=manifest_path,
            queue_map=queue_map,
            payload=output,
            metadata=metadata or {},
        )
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output_path if output_path is not None else source_path,
            metadata={
                "workflow": FRONTIER_RERUN_ROLLUP_COMPLETION_WORKFLOW,
                "status": status,
                "source": str(source_path),
                "artifact_manifest": None if manifest_path is None else str(manifest_path),
                "entry_count": summary["entry_count"],
                "command_count": summary["command_count"],
                "missing_queue_count": summary["missing_queue_count"],
                "unsupported_workflow_count": summary["unsupported_workflow_count"],
                "tracks": summary["tracks"],
                "manifest_summary": {} if manifest is None else manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return output


def _frontier_rerun_rollup_completion_decisions(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    decisions = tuple(
        item
        for item in _mapping_sequence(payload.get("frontier_rerun_rollup_decisions", ()))
        if item.get("status") != "promote"
    )
    if decisions:
        return decisions
    workflow = _optional_str(payload.get("workflow"))
    if workflow in FRONTIER_RERUN_ROLLUP_WORKFLOW_TO_TRACK and payload.get("status") != "promote":
        return (
            {
                "name": _frontier_rerun_rollup_name(payload),
                "status": payload.get("status", "blocked"),
                "metrics": {
                    "workflow": workflow,
                    "track": FRONTIER_RERUN_ROLLUP_WORKFLOW_TO_TRACK[workflow],
                    **dict(_mapping(payload.get("summary"))),
                },
                "blocking_reasons": _frontier_rerun_rollup_report_blocking_reasons(payload),
            },
        )
    return ()


def _frontier_rerun_rollup_completion_entry(
    decision: Mapping[str, Any],
    *,
    rollup_root: Path,
    queue_map: Mapping[str, str],
    python_executable: str,
) -> dict[str, Any]:
    metrics = _mapping(decision.get("metrics"))
    workflow = _optional_str(metrics.get("workflow")) or _infer_rollup_workflow(decision)
    track = (
        _optional_str(metrics.get("track"))
        or FRONTIER_RERUN_ROLLUP_WORKFLOW_TO_TRACK.get(workflow or "")
        or ""
    )
    config = FRONTIER_RERUN_ROLLUP_TRACKS.get(track)
    name = _optional_str(decision.get("name")) or f"frontier-{track or 'unknown'}-rerun-rollup"
    base = {
        "name": name,
        "track": track,
        "rollup_workflow": workflow,
        "source_status": _optional_str(decision.get("status")) or "blocked",
        "metrics": _frontier_rerun_rollup_metrics(metrics),
        "blocking_reasons": _string_tuple(decision.get("blocking_reasons", ())),
    }
    if config is None or workflow not in FRONTIER_RERUN_ROLLUP_WORKFLOW_TO_TRACK:
        return {
            **base,
            "command_status": "unsupported_workflow",
            "missing_inputs": ("supported_rollup_workflow",),
            "queue": None,
            "report": None,
            "artifact_manifest": None,
            "command": (),
        }
    queue = queue_map.get(track) or queue_map.get(str(config["workflow"]))
    output_dir = rollup_root / track
    report = output_dir / "frontier-rerun-rollup.json"
    manifest = output_dir / "artifact-manifest.json"
    if queue is None:
        return {
            **base,
            "command_status": "missing_queue",
            "missing_inputs": (f"{track}_queue",),
            "queue": None,
            "report": str(report),
            "artifact_manifest": str(manifest),
            "command": (),
        }
    command = (
        python_executable,
        str(config["script"]),
        "--queue",
        queue,
        "--json",
        str(report),
        "--artifact-manifest",
        str(manifest),
        "--require-all-reports",
    )
    return {
        **base,
        "command_status": "ready",
        "missing_inputs": (),
        "queue": queue,
        "report": str(report),
        "artifact_manifest": str(manifest),
        "command": command,
    }


def _frontier_rerun_rollup_queue_map(values: Sequence[str | Path]) -> dict[str, str]:
    queue_map: dict[str, str] = {}
    for value in values:
        item = str(value)
        if "=" not in item:
            raise ValueError(f"frontier rerun-rollup queue must be TRACK=PATH, got {item!r}.")
        key, path = item.split("=", 1)
        key = key.strip()
        path = path.strip()
        if not key or not path:
            raise ValueError(f"frontier rerun-rollup queue must be TRACK=PATH, got {item!r}.")
        normalized = FRONTIER_RERUN_ROLLUP_WORKFLOW_TO_TRACK.get(key, key)
        if normalized not in FRONTIER_RERUN_ROLLUP_TRACKS:
            raise ValueError(
                "frontier rerun-rollup queue track must be one of "
                f"{tuple(FRONTIER_RERUN_ROLLUP_TRACKS)}, got {key!r}."
            )
        queue_map[normalized] = path
        queue_map[FRONTIER_RERUN_ROLLUP_TRACKS[normalized]["workflow"]] = path
    return queue_map


def _frontier_rerun_rollup_completion_status(summary: Mapping[str, Any]) -> str:
    if summary["entry_count"] == 0:
        return "empty"
    if summary["missing_queue_count"] or summary["unsupported_workflow_count"]:
        return "needs_inputs"
    return "ready"


def _frontier_rerun_rollup_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_count",
        "observed_report_count",
        "missing_report_count",
        "invalid_report_count",
        "blocked_candidate_count",
        "promotion_ready_count",
        "promotion_ready",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _infer_rollup_workflow(decision: Mapping[str, Any]) -> str | None:
    text = " ".join((
        _optional_str(decision.get("name")) or "",
        " ".join(_string_tuple(decision.get("blocking_reasons", ()))),
    ))
    for workflow in FRONTIER_RERUN_ROLLUP_WORKFLOW_TO_TRACK:
        if workflow in text:
            return workflow
    return None


def _frontier_rerun_rollup_name(payload: Mapping[str, Any]) -> str:
    metadata = _mapping(payload.get("metadata"))
    return (
        _optional_str(metadata.get("name"))
        or _optional_str(_nested_value(payload, "paths", "report"))
        or "frontier-rerun-rollup"
    )


def _frontier_rerun_rollup_report_blocking_reasons(
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    gate = _mapping(payload.get("gate"))
    reasons = []
    for reason in _mapping_sequence(gate.get("blocking_reasons", ())):
        text = _optional_str(reason.get("reason"))
        if text:
            reasons.append(text)
    reasons.extend(_string_tuple(gate.get("blocking_reasons", ())))
    return tuple(dict.fromkeys(reasons))


def _write_frontier_rerun_rollup_completion_manifest(
    *,
    source_path: Path,
    output_path: Path | None,
    manifest_path: Path,
    queue_map: Mapping[str, str],
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "frontier_rerun_rollup_completion_plan": output_path,
        "source": source_path,
    }
    for track in FRONTIER_RERUN_ROLLUP_TRACKS:
        queue = queue_map.get(track)
        if queue is not None:
            artifacts[f"{track}_rerun_queue"] = queue
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "plan_release_evidence_gaps",
            "workflow": FRONTIER_RERUN_ROLLUP_COMPLETION_WORKFLOW,
            "status": payload.get("status"),
            "entry_count": _nested_value(payload, "summary", "entry_count"),
            "command_count": _nested_value(payload, "summary", "command_count"),
            "missing_queue_count": _nested_value(payload, "summary", "missing_queue_count"),
            **dict(metadata),
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_json_object(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("source JSON must contain an object.")
    return data


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value if str(item))
    return ()


def _nested_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _parse_metadata(items: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"metadata must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"metadata key must be non-empty, got {item!r}")
        metadata[key] = value.strip()
    return metadata


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


if __name__ == "__main__":
    main()
