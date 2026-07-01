"""Invoke an external citation/search adapter command and gate its results.

The external searcher stays outside EigenTruth. This workflow writes sanitized
request JSONL from an unresolved blind-spot queue, invokes a local command that
must write result JSONL, then feeds those results into
``run_citation_search_evidence_workflow.py``.

The command is executed without a shell. Use ``{input}`` and ``{output}``
placeholders in the command string to receive the request JSONL and result
output path.
"""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
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
from benchmarks.sweep_blind_spot_retrieval_queries import DEFAULT_TARGET_ROUTE  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "external_citation_search_adapter_workflow"
_OUTPUT_LIMIT = 4000


def run_external_citation_search_adapter_workflow(
    *,
    queue_report_path: str | Path,
    search_command: str | Sequence[str],
    scores_path: str | Path,
    blind_spots_path: str | Path,
    output_dir: str | Path,
    controlled_sweep_paths: Sequence[str | Path] = (),
    request_jsonl_path: str | Path | None = None,
    adapter_results_path: str | Path | None = None,
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
    command_timeout_seconds: float | None = None,
    target_route: str = DEFAULT_TARGET_ROUTE,
    min_adapter_request_coverage: float = 1.0,
    evidence_metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
    fail_on_blocked: bool = False,
) -> dict[str, Any]:
    """Run an external search command and gate the returned evidence."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if not (0.0 <= float(min_adapter_request_coverage) <= 1.0):
        raise ValueError("min_adapter_request_coverage must be between 0 and 1.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    request_jsonl = Path(request_jsonl_path or output / "external-citation-search-requests.jsonl")
    adapter_results = Path(adapter_results_path or output / "external-citation-search-results.jsonl")
    workflow_report = Path(workflow_report_path or output / "external-citation-search-adapter-workflow.json")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")
    preflight_dir = output / "request-handoff"
    evidence_dir = output / "evidence-gate"

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
        metadata={**dict(evidence_metadata or {}), "source_workflow": WORKFLOW},
        compact_json=compact_json,
    )
    command_args = _format_command(
        search_command,
        input_path=request_jsonl,
        output_path=adapter_results,
    )
    adapter_results.parent.mkdir(parents=True, exist_ok=True)
    if adapter_results.exists():
        adapter_results.unlink()
    command_result = _run_command(command_args, timeout_seconds=command_timeout_seconds)
    if not adapter_results.exists():
        raise FileNotFoundError(f"external citation/search adapter did not write results: {adapter_results}")

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
        target_route=target_route,
        min_adapter_request_coverage=min_adapter_request_coverage,
        metadata={**dict(evidence_metadata or {}), "source_workflow": WORKFLOW},
        compact_json=compact_json,
    )
    gate = {
        "passed": bool(evidence.get("gate", {}).get("passed")),
        "promotion_ready": bool(evidence.get("gate", {}).get("promotion_ready")),
        "blocking_reasons": tuple(evidence.get("gate", {}).get("blocking_reasons", ())),
    }
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
        },
        "config": {
            "batch_ids": tuple(str(item) for item in batch_ids),
            "query_mode": query_mode,
            "max_requests": max_requests,
            "max_results_per_request": max_results_per_request,
            "max_alternate_queries": int(max_alternate_queries),
            "corpus_name": corpus_name,
            "source_kind": source_kind,
            "command_timeout_seconds": command_timeout_seconds,
            "target_route": target_route,
            "min_adapter_request_coverage": float(min_adapter_request_coverage),
        },
        "paths": {
            "requests": str(request_jsonl),
            "adapter_results": str(adapter_results),
            "request_handoff": str(preflight_dir / "citation-search-request-handoff.json"),
            "request_handoff_manifest": str(preflight_dir / "artifact-manifest.json"),
            "evidence_workflow": str(evidence_dir / "citation-search-evidence-workflow.json"),
            "evidence_manifest": str(evidence_dir / "artifact-manifest.json"),
            "workflow_report": str(workflow_report),
            "artifact_manifest": str(manifest_path),
        },
        "request_summary": dict(preflight.get("summary", {})),
        "command": {
            "args": command_args,
            "returncode": command_result.returncode,
            "stdout": _bounded_text(command_result.stdout),
            "stderr": _bounded_text(command_result.stderr),
        },
        "evidence_summary": dict(evidence.get("summary", {})),
        "gate": gate,
        "metadata": dict(evidence_metadata or {}),
    }
    _write_json(workflow_report, payload, compact=compact_json)
    manifest = build_artifact_manifest(
        _manifest_artifacts(
            queue_report_path=queue_report_path,
            scores_path=scores_path,
            blind_spots_path=blind_spots_path,
            controlled_sweep_paths=controlled_sweep_paths,
            request_jsonl_path=request_jsonl,
            adapter_results_path=adapter_results,
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
            "target_route": target_route,
            "selected_batch_count": payload["request_summary"].get("selected_batch_count"),
            "selected_batch_ids": payload["request_summary"].get("selected_batch_ids"),
            "adapter_request_count": payload["request_summary"].get("adapter_request_count"),
            "adapter_result_request_coverage": payload["evidence_summary"].get(
                "adapter_result_request_coverage"
            ),
            "adapter_result_missing_request_count": payload["evidence_summary"].get(
                "adapter_result_missing_request_count"
            ),
            "source_document_count": payload["evidence_summary"].get("source_document_count"),
            "corpus_document_count": payload["evidence_summary"].get("corpus_document_count"),
            **dict(evidence_metadata or {}),
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
                "target_route": target_route,
                "selected_batch_count": payload["request_summary"].get("selected_batch_count"),
                "selected_batch_ids": payload["request_summary"].get("selected_batch_ids"),
                "adapter_request_count": payload["request_summary"].get("adapter_request_count"),
                "adapter_result_request_coverage": payload["evidence_summary"].get(
                    "adapter_result_request_coverage"
                ),
                "adapter_result_missing_request_count": payload["evidence_summary"].get(
                    "adapter_result_missing_request_count"
                ),
                "source_document_count": payload["evidence_summary"].get("source_document_count"),
                "corpus_document_count": payload["evidence_summary"].get("corpus_document_count"),
                "artifact_manifest": str(manifest_path),
                **dict(evidence_metadata or {}),
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
    request_jsonl_path: Path,
    adapter_results_path: Path,
    preflight_dir: Path,
    evidence_dir: Path,
    workflow_report_path: Path,
) -> dict[str, str | Path]:
    artifacts: dict[str, str | Path] = {
        "workflow_report": workflow_report_path,
        "queue_report": Path(queue_report_path),
        "scores": Path(scores_path),
        "blind_spots": Path(blind_spots_path),
        "external_citation_search_requests": request_jsonl_path,
        "external_citation_search_results": adapter_results_path,
        "request_handoff": preflight_dir / "citation-search-request-handoff.json",
        "request_handoff_manifest": preflight_dir / "artifact-manifest.json",
        "evidence_workflow": evidence_dir / "citation-search-evidence-workflow.json",
        "evidence_manifest": evidence_dir / "artifact-manifest.json",
    }
    for index, path in enumerate(controlled_sweep_paths, start=1):
        artifacts[f"controlled_sweep_{index}"] = Path(path)
    return artifacts


def _format_command(
    command: str | Sequence[str],
    *,
    input_path: Path,
    output_path: Path,
) -> list[str]:
    parts = shlex.split(command) if isinstance(command, str) else [str(item) for item in command]
    if not parts:
        raise ValueError("search_command must be non-empty.")
    formatted = [
        part.replace("{input}", str(input_path)).replace("{output}", str(output_path))
        for part in parts
    ]
    if not any("{input}" in part for part in parts):
        raise ValueError("search_command must include {input} placeholder.")
    if not any("{output}" in part for part in parts):
        raise ValueError("search_command must include {output} placeholder.")
    return formatted


def _run_command(
    command: Sequence[str],
    *,
    timeout_seconds: float | None,
) -> subprocess.CompletedProcess[str]:
    timeout = None if timeout_seconds is None else float(timeout_seconds)
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0.0):
        raise ValueError("command_timeout_seconds must be positive and finite when set.")
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _bounded_text(text: str, *, limit: int = _OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


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
    parser.add_argument("--queue", required=True)
    parser.add_argument("--search-command", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--blind-spots", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--controlled-sweep", action="append", default=[])
    parser.add_argument("--request-jsonl", default=None)
    parser.add_argument("--adapter-results", default=None)
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
    parser.add_argument("--query-mode", choices=QUERY_MODES, default="question")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-results-per-request", type=int, default=None)
    parser.add_argument("--max-alternate-queries", type=int, default=DEFAULT_MAX_ALTERNATE_QUERIES)
    parser.add_argument("--corpus-name", default=DEFAULT_CORPUS_NAME)
    parser.add_argument("--source-kind", default=DEFAULT_SOURCE_KIND)
    parser.add_argument("--command-timeout-seconds", type=float, default=None)
    parser.add_argument("--target-route", default=DEFAULT_TARGET_ROUTE)
    parser.add_argument("--min-adapter-request-coverage", type=float, default=1.0)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)
    payload = run_external_citation_search_adapter_workflow(
        queue_report_path=args.queue,
        search_command=args.search_command,
        scores_path=args.scores,
        blind_spots_path=args.blind_spots,
        output_dir=args.output_dir,
        controlled_sweep_paths=tuple(args.controlled_sweep or ()),
        request_jsonl_path=args.request_jsonl,
        adapter_results_path=args.adapter_results,
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
        command_timeout_seconds=args.command_timeout_seconds,
        target_route=args.target_route,
        min_adapter_request_coverage=args.min_adapter_request_coverage,
        evidence_metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
        fail_on_blocked=bool(args.fail_on_blocked),
    )
    print(
        "external_citation_search_adapter_workflow_ok "
        f"status={payload['status']} "
        f"gate_passed={payload['gate']['passed']} "
        f"promotion_ready={payload['gate']['promotion_ready']}"
    )


if __name__ == "__main__":
    main()
