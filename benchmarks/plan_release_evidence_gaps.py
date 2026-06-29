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
from benchmarks.plan_frontier_multiple_testing_reruns import (  # noqa: E402
    build_frontier_multiple_testing_rerun_queue,
)
from benchmarks.plan_frontier_stability_evidence_reruns import (  # noqa: E402
    build_frontier_stability_evidence_rerun_queue,
)
from eigentruth.control import plan_evidence_gaps_from_release_candidate  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry  # noqa: E402


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
    parser.add_argument("--python", default=sys.executable, help="Python executable for generated rerun commands")
    run(parser.parse_args(argv))


def _build_derived_artifacts(
    *,
    source_path: Path,
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
    return derived


def _load_json_object(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("source JSON must contain an object.")
    return data


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
