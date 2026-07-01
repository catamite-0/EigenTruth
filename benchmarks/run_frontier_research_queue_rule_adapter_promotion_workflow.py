"""Dry-run or execute frontier rule adapter and promotion from fill rollups.

This workflow consumes ``frontier_research_queue_input_fill_result_rollup``
reports. It can materialize the deterministic rule-adapter step, run the
fail-closed rule-candidate promotion gate, optionally build the ProductTrace
handoff, and optionally bundle that handoff as release-gate evidence. By default
it only writes a plan; explicit ``--execute`` is required before adapter,
promotion, handoff, or evidence-bundle artifacts are produced.
"""

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

from benchmarks.build_mechanism_handoff_evidence_bundle import (  # noqa: E402
    WORKFLOW as EVIDENCE_BUNDLE_WORKFLOW,
)
from benchmarks.build_mechanism_handoff_evidence_bundle import (  # noqa: E402
    run as run_mechanism_handoff_evidence_bundle,
)
from benchmarks.build_world_model_rule_candidate_handoff import (  # noqa: E402
    DEFAULT_ROUTE_NAME as DEFAULT_HANDOFF_ROUTE_NAME,
)
from benchmarks.build_world_model_rule_candidate_handoff import (  # noqa: E402
    DEFAULT_VERIFIER_NAME as DEFAULT_HANDOFF_VERIFIER_NAME,
)
from benchmarks.build_world_model_rule_candidate_handoff import (  # noqa: E402
    WORKFLOW as HANDOFF_WORKFLOW,
)
from benchmarks.build_world_model_rule_candidate_handoff import run as run_rule_candidate_handoff  # noqa: E402
from benchmarks.promote_world_model_rule_candidates import (  # noqa: E402
    WORKFLOW as PROMOTION_WORKFLOW,
)
from benchmarks.promote_world_model_rule_candidates import run as run_rule_promotion  # noqa: E402
from benchmarks.rollup_frontier_research_queue_input_fill_results import (  # noqa: E402
    WORKFLOW as FILL_ROLLUP_WORKFLOW,
)
from benchmarks.run_world_model_rule_authoring_adapter import (  # noqa: E402
    WORKFLOW as RULE_ADAPTER_WORKFLOW,
)
from benchmarks.run_world_model_rule_authoring_adapter import (  # noqa: E402
    run_world_model_rule_authoring_adapter,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "frontier_research_queue_rule_adapter_promotion_workflow"
ADAPTER_READY_ROLLUP_STATUSES = {"ready_for_adapter", "partial"}


def run_frontier_research_queue_rule_adapter_promotion_workflow(
    *,
    input_fill_result_rollup: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    rule_stubs_path: str | Path | None = None,
    combined_rule_inputs_path: str | Path | None = None,
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    execute: bool = False,
    build_handoff: bool = False,
    build_evidence_bundle: bool = False,
    min_confidence: float = 0.90,
    handoff_route_name: str = DEFAULT_HANDOFF_ROUTE_NAME,
    handoff_verifier_name: str = DEFAULT_HANDOFF_VERIFIER_NAME,
    bundle_expected_target_count: int | None = None,
    bundle_min_trace_count: int | None = None,
    bundle_min_supported_count: int | None = None,
    bundle_min_refuted_count: int | None = None,
    bundle_min_source_citation_count: int | None = None,
    bundle_require_action_execution_alignment: bool = True,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan or execute adapter + promotion gates from a fill-result rollup."""
    if not isinstance(execute, bool):
        raise ValueError("execute must be a bool.")
    if not isinstance(build_handoff, bool):
        raise ValueError("build_handoff must be a bool.")
    if not isinstance(build_evidence_bundle, bool):
        raise ValueError("build_evidence_bundle must be a bool.")
    if not isinstance(bundle_require_action_execution_alignment, bool):
        raise ValueError("bundle_require_action_execution_alignment must be a bool.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    source_path, rollup = _load_mapping_source(input_fill_result_rollup)
    if rollup.get("workflow") != FILL_ROLLUP_WORKFLOW:
        raise ValueError(f"input_fill_result_rollup must have workflow={FILL_ROLLUP_WORKFLOW!r}.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(json_path or output / "frontier-rule-adapter-promotion-workflow.json")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")
    paths = _workflow_paths(
        output=output,
        rollup=rollup,
        source_root=None if source_path is None else source_path.parent,
        rule_stubs_path=rule_stubs_path,
        combined_rule_inputs_path=combined_rule_inputs_path,
    )
    preflight = _preflight(rollup=rollup, paths=paths)
    if build_evidence_bundle and not build_handoff:
        preflight = _preflight_with_failure(preflight, "evidence_bundle_requires_handoff")
    adapter_payload: Mapping[str, Any] | None = None
    promotion_payload: Mapping[str, Any] | None = None
    handoff_payload: Mapping[str, Any] | None = None
    evidence_bundle_payload: Mapping[str, Any] | None = None
    if execute and not preflight["failures"]:
        adapter_payload = run_world_model_rule_authoring_adapter(
            rule_stubs_path=paths["rule_stubs"],
            rule_inputs_path=paths["combined_rule_inputs"],
            output_dir=paths["adapter_output_dir"],
            report_json_path=paths["adapter_report"],
            rule_results_path=paths["adapter_rule_results"],
            input_requests_path=paths["adapter_input_requests"],
            artifact_manifest_path=paths["adapter_artifact_manifest"],
            metadata={
                "parent_workflow": WORKFLOW,
                "source_rollup": None if source_path is None else str(source_path),
                **dict(metadata or {}),
            },
            compact_json=compact_json,
        )
        promotion_payload = run_rule_promotion(
            rule_results_path=paths["adapter_rule_results"],
            rule_inputs_path=paths["combined_rule_inputs"],
            adapter_report_path=paths["adapter_report"],
            output_dir=paths["promotion_output_dir"],
            report_json_path=paths["promotion_report"],
            promoted_jsonl_path=paths["promoted_candidates"],
            blocked_jsonl_path=paths["blocked_candidates"],
            pending_jsonl_path=paths["pending_inputs"],
            artifact_manifest_path=paths["promotion_artifact_manifest"],
            min_confidence=min_confidence,
            metadata={
                "parent_workflow": WORKFLOW,
                "source_rollup": None if source_path is None else str(source_path),
                **dict(metadata or {}),
            },
            compact_json=compact_json,
        )
        if build_handoff and promotion_payload.get("status") == "promote":
            handoff_payload = run_rule_candidate_handoff(
                promotion_gate_path=paths["promotion_report"],
                promoted_candidates_path=paths["promoted_candidates"],
                output_dir=paths["handoff_output_dir"],
                report_json_path=paths["handoff_report"],
                trace_jsonl_path=paths["handoff_product_traces"],
                action_results_jsonl_path=paths["handoff_action_results"],
                artifact_manifest_path=paths["handoff_artifact_manifest"],
                route_name=handoff_route_name,
                verifier_name=handoff_verifier_name,
                metadata={
                    "parent_workflow": WORKFLOW,
                    "source_rollup": None if source_path is None else str(source_path),
                    **dict(metadata or {}),
                },
                compact_json=compact_json,
            )
            if build_evidence_bundle and _mapping(handoff_payload.get("report")).get("status") == "promote":
                evidence_bundle_payload = run_mechanism_handoff_evidence_bundle(
                    handoff_paths=(paths["handoff_report"],),
                    output_dir=paths["evidence_bundle_output_dir"],
                    report_json_path=paths["evidence_bundle_report"],
                    artifact_manifest_path=paths["evidence_bundle_artifact_manifest"],
                    expected_target_count=bundle_expected_target_count,
                    min_trace_count=bundle_min_trace_count,
                    min_supported_count=bundle_min_supported_count,
                    min_refuted_count=bundle_min_refuted_count,
                    min_source_citation_count=bundle_min_source_citation_count,
                    require_action_execution_alignment=bundle_require_action_execution_alignment,
                    metadata={
                        "parent_workflow": WORKFLOW,
                        "source_rollup": None if source_path is None else str(source_path),
                        **dict(metadata or {}),
                    },
                    compact_json=compact_json,
                )

    summary = _summary(
        rollup=rollup,
        preflight=preflight,
        execute=execute,
        build_handoff=build_handoff,
        build_evidence_bundle=build_evidence_bundle,
        adapter_payload=adapter_payload,
        promotion_payload=promotion_payload,
        handoff_payload=handoff_payload,
        evidence_bundle_payload=evidence_bundle_payload,
    )
    status = _status(
        summary=summary,
        execute=execute,
        build_handoff=build_handoff,
        build_evidence_bundle=build_evidence_bundle,
    )
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Plans or executes deterministic rule-adapter replay plus the "
            "world-model rule-candidate promotion gate, with an optional "
            "ProductTrace handoff and evidence bundle, from a fill-result "
            "rollup. This workflow is not verifier evidence."
        ),
        "source": {
            "input_fill_result_rollup": None if source_path is None else str(source_path),
            "input_fill_result_rollup_workflow": rollup.get("workflow"),
            "input_fill_result_rollup_status": rollup.get("status"),
        },
        "label_usage": {
            "labels_used_for_adapter_or_promotion": False,
            "workflow_report_is_verifier_evidence": False,
            "adapter_results_require_promotion_gate": True,
            "promotion_gate_required_before_product_handoff": True,
        },
        "config": {
            "execute": bool(execute),
            "executes_adapter": bool(execute and not preflight["failures"]),
            "executes_promotion_gate": bool(execute and not preflight["failures"]),
            "build_handoff": bool(build_handoff),
            "build_evidence_bundle": bool(build_evidence_bundle),
            "executes_handoff": bool(
                execute
                and build_handoff
                and not preflight["failures"]
                and promotion_payload is not None
                and promotion_payload.get("status") == "promote"
            ),
            "executes_evidence_bundle": bool(
                execute
                and build_evidence_bundle
                and evidence_bundle_payload is not None
            ),
            "min_confidence": float(min_confidence),
            "handoff_route_name": handoff_route_name,
            "handoff_verifier_name": handoff_verifier_name,
            "bundle_expected_target_count": bundle_expected_target_count,
            "bundle_min_trace_count": bundle_min_trace_count,
            "bundle_min_supported_count": bundle_min_supported_count,
            "bundle_min_refuted_count": bundle_min_refuted_count,
            "bundle_min_source_citation_count": bundle_min_source_citation_count,
            "bundle_require_action_execution_alignment": bool(bundle_require_action_execution_alignment),
            "allowed_rollup_statuses": tuple(sorted(ADAPTER_READY_ROLLUP_STATUSES)),
        },
        "paths": _public_paths(paths, report_path=report_path, manifest_path=manifest_path),
        "preflight": preflight,
        "planned_commands": _planned_commands(
            paths=paths,
            min_confidence=min_confidence,
            build_handoff=build_handoff,
            build_evidence_bundle=build_evidence_bundle,
            handoff_route_name=handoff_route_name,
            handoff_verifier_name=handoff_verifier_name,
            bundle_expected_target_count=bundle_expected_target_count,
            bundle_min_trace_count=bundle_min_trace_count,
            bundle_min_supported_count=bundle_min_supported_count,
            bundle_min_refuted_count=bundle_min_refuted_count,
            bundle_min_source_citation_count=bundle_min_source_citation_count,
            bundle_require_action_execution_alignment=bundle_require_action_execution_alignment,
        ),
        "summary": summary,
        "metadata": dict(metadata or {}),
    }
    _write_json(report_path, payload, compact=compact_json)
    manifest = build_artifact_manifest(
        _manifest_artifacts(
            source_path=source_path,
            paths=paths,
            report_path=report_path,
            execute=execute and not preflight["failures"],
            handoff_executed=handoff_payload is not None,
            evidence_bundle_executed=evidence_bundle_payload is not None,
        ),
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": status,
            "execute": bool(execute),
            "build_handoff": bool(build_handoff),
            "build_evidence_bundle": bool(build_evidence_bundle),
            "preflight_failure_count": summary["preflight_failure_count"],
            "combined_rule_input_count": summary["combined_rule_input_count"],
            "adapter_status": summary["adapter_status"],
            "promotion_status": summary["promotion_status"],
            "handoff_status": summary["handoff_status"],
            "evidence_bundle_status": summary["evidence_bundle_status"],
            "promoted_count": summary["promoted_count"],
            "handoff_trace_count": summary["handoff_trace_count"],
            "evidence_bundle_trace_count": summary["evidence_bundle_trace_count"],
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
                "execute": bool(execute),
                "build_handoff": bool(build_handoff),
                "build_evidence_bundle": bool(build_evidence_bundle),
                "preflight_failure_count": summary["preflight_failure_count"],
                "combined_rule_input_count": summary["combined_rule_input_count"],
                "adapter_status": summary["adapter_status"],
                "promotion_status": summary["promotion_status"],
                "handoff_status": summary["handoff_status"],
                "evidence_bundle_status": summary["evidence_bundle_status"],
                "promoted_count": summary["promoted_count"],
                "handoff_trace_count": summary["handoff_trace_count"],
                "evidence_bundle_trace_count": summary["evidence_bundle_trace_count"],
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _workflow_paths(
    *,
    output: Path,
    rollup: Mapping[str, Any],
    source_root: Path | None,
    rule_stubs_path: str | Path | None,
    combined_rule_inputs_path: str | Path | None,
) -> dict[str, Path | None]:
    rollup_paths = _mapping(rollup.get("paths"))
    rollup_source = _mapping(rollup.get("source"))
    rule_stubs = (
        rule_stubs_path
        or rollup_paths.get("rule_stubs")
        or rollup_source.get("rule_stubs")
    )
    combined_inputs = combined_rule_inputs_path or rollup_paths.get("combined_rule_inputs")
    adapter_dir = output / "rule-adapter"
    promotion_dir = output / "rule-promotion"
    handoff_dir = output / "rule-candidate-handoff"
    evidence_bundle_dir = output / "mechanism-handoff-evidence-bundle"
    return {
        "rule_stubs": None if rule_stubs in (None, "") else _resolve_path(rule_stubs, source_root=source_root),
        "combined_rule_inputs": (
            None if combined_inputs in (None, "") else _resolve_path(combined_inputs, source_root=source_root)
        ),
        "adapter_output_dir": adapter_dir,
        "adapter_report": adapter_dir / "world-model-rule-authoring-adapter.json",
        "adapter_rule_results": adapter_dir / "world-model-rule-results.jsonl",
        "adapter_input_requests": adapter_dir / "world-model-rule-input-requests.jsonl",
        "adapter_artifact_manifest": adapter_dir / "artifact-manifest.json",
        "promotion_output_dir": promotion_dir,
        "promotion_report": promotion_dir / "world-model-rule-candidate-promotion-gate.json",
        "promoted_candidates": promotion_dir / "promoted-rule-candidates.jsonl",
        "blocked_candidates": promotion_dir / "blocked-rule-candidates.jsonl",
        "pending_inputs": promotion_dir / "pending-rule-inputs.jsonl",
        "promotion_artifact_manifest": promotion_dir / "artifact-manifest.json",
        "handoff_output_dir": handoff_dir,
        "handoff_report": handoff_dir / "world-model-rule-candidate-handoff.json",
        "handoff_product_traces": handoff_dir / "product-traces.jsonl",
        "handoff_action_results": handoff_dir / "action-results.jsonl",
        "handoff_artifact_manifest": handoff_dir / "artifact-manifest.json",
        "evidence_bundle_output_dir": evidence_bundle_dir,
        "evidence_bundle_report": evidence_bundle_dir / "mechanism-handoff-evidence-bundle.json",
        "evidence_bundle_artifact_manifest": evidence_bundle_dir / "artifact-manifest.json",
    }


def _preflight(*, rollup: Mapping[str, Any], paths: Mapping[str, Path | None]) -> dict[str, Any]:
    failures: list[str] = []
    rollup_status = str(rollup.get("status") or "")
    if rollup_status not in ADAPTER_READY_ROLLUP_STATUSES:
        failures.append("rollup_status_not_adapter_ready")
    if _int(_mapping(rollup.get("summary")).get("duplicate_request_id_count")) > 0:
        failures.append("rollup_has_duplicate_request_ids")
    rule_stubs = paths.get("rule_stubs")
    combined_inputs = paths.get("combined_rule_inputs")
    if rule_stubs is None:
        failures.append("missing_rule_stubs_path")
    elif not rule_stubs.exists():
        failures.append("rule_stubs_not_materialized")
    if combined_inputs is None:
        failures.append("missing_combined_rule_inputs_path")
        combined_rows: tuple[Mapping[str, Any], ...] = ()
    elif not combined_inputs.exists():
        failures.append("combined_rule_inputs_not_materialized")
        combined_rows = ()
    else:
        combined_rows = _load_jsonl_mappings(combined_inputs)
        if not combined_rows:
            failures.append("empty_combined_rule_inputs")
    return {
        "status": "ready" if not failures else "blocked",
        "failures": tuple(dict.fromkeys(failures)),
        "rollup_status": rollup_status,
        "rule_stubs_path": None if rule_stubs is None else str(rule_stubs),
        "rule_stubs_materialized": bool(rule_stubs is not None and rule_stubs.exists()),
        "combined_rule_inputs_path": None if combined_inputs is None else str(combined_inputs),
        "combined_rule_inputs_materialized": bool(combined_inputs is not None and combined_inputs.exists()),
        "combined_rule_input_count": len(combined_rows),
    }


def _preflight_with_failure(preflight: Mapping[str, Any], failure: str) -> dict[str, Any]:
    failures = tuple(dict.fromkeys((*tuple(preflight.get("failures", ())), failure)))
    return {
        **dict(preflight),
        "status": "blocked",
        "failures": failures,
    }


def _summary(
    *,
    rollup: Mapping[str, Any],
    preflight: Mapping[str, Any],
    execute: bool,
    build_handoff: bool,
    build_evidence_bundle: bool,
    adapter_payload: Mapping[str, Any] | None,
    promotion_payload: Mapping[str, Any] | None,
    handoff_payload: Mapping[str, Any] | None,
    evidence_bundle_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    rollup_summary = _mapping(rollup.get("summary"))
    adapter_summary = _mapping(None if adapter_payload is None else adapter_payload.get("summary"))
    promotion_summary = _mapping(None if promotion_payload is None else promotion_payload.get("summary"))
    handoff_report = _mapping(None if handoff_payload is None else handoff_payload.get("report"))
    handoff_summary = _mapping(handoff_report.get("summary"))
    evidence_bundle_summary = _mapping(
        None if evidence_bundle_payload is None else evidence_bundle_payload.get("summary")
    )
    evidence_bundle_gate = _mapping(
        None if evidence_bundle_payload is None else evidence_bundle_payload.get("gate")
    )
    return {
        "rollup_status": rollup.get("status"),
        "rollup_combined_rule_input_count": _int(rollup_summary.get("combined_rule_input_count")),
        "rollup_unfilled_task_count": _int(rollup_summary.get("combined_unfilled_task_count")),
        "combined_rule_input_count": int(preflight.get("combined_rule_input_count", 0)),
        "preflight_status": preflight.get("status"),
        "preflight_failure_count": len(tuple(preflight.get("failures", ()))),
        "preflight_failures": tuple(str(item) for item in preflight.get("failures", ())),
        "dry_run": not execute,
        "executed": bool(execute and preflight.get("status") == "ready"),
        "build_handoff": bool(build_handoff),
        "build_evidence_bundle": bool(build_evidence_bundle),
        "adapter_workflow": None if adapter_payload is None else adapter_payload.get("workflow"),
        "adapter_status": None if adapter_payload is None else adapter_payload.get("status"),
        "adapter_executed_count": _int(adapter_summary.get("executed_count")),
        "adapter_needs_input_count": _int(adapter_summary.get("needs_input_count")),
        "adapter_stub_result_coverage": adapter_summary.get("stub_result_coverage"),
        "promotion_workflow": None if promotion_payload is None else promotion_payload.get("workflow"),
        "promotion_status": None if promotion_payload is None else promotion_payload.get("status"),
        "promoted_count": _int(promotion_summary.get("promoted_count")),
        "blocked_count": _int(promotion_summary.get("blocked_count")),
        "pending_count": _int(promotion_summary.get("pending_count")),
        "adapter_report_gate": promotion_summary.get("adapter_report_gate"),
        "handoff_workflow": handoff_report.get("workflow"),
        "handoff_status": handoff_report.get("status"),
        "handoff_trace_count": _int(handoff_summary.get("trace_count")),
        "handoff_action_result_count": _int(handoff_summary.get("action_result_count")),
        "handoff_blocked_candidate_count": _int(handoff_summary.get("blocked_candidate_count")),
        "evidence_bundle_workflow": (
            None if evidence_bundle_payload is None else evidence_bundle_payload.get("workflow")
        ),
        "evidence_bundle_status": None if evidence_bundle_payload is None else evidence_bundle_payload.get("status"),
        "evidence_bundle_gate_passed": evidence_bundle_gate.get("passed"),
        "evidence_bundle_trace_count": _int(evidence_bundle_summary.get("trace_count")),
        "evidence_bundle_target_count": _int(evidence_bundle_summary.get("target_count")),
        "evidence_bundle_source_citation_count": _int(evidence_bundle_summary.get("source_citation_count")),
    }


def _status(
    *,
    summary: Mapping[str, Any],
    execute: bool,
    build_handoff: bool,
    build_evidence_bundle: bool,
) -> str:
    if int(summary.get("preflight_failure_count", 0)) > 0:
        return "blocked"
    if not execute:
        return "dry_run"
    if summary.get("promotion_status") == "promote":
        if build_evidence_bundle:
            return "promote" if summary.get("evidence_bundle_status") == "promote" else "blocked"
        if build_handoff:
            return "promote" if summary.get("handoff_status") == "promote" else "blocked"
        return "promote"
    if summary.get("adapter_status") == "empty":
        return "empty"
    return "blocked"


def _planned_commands(
    *,
    paths: Mapping[str, Path | None],
    min_confidence: float,
    build_handoff: bool,
    build_evidence_bundle: bool,
    handoff_route_name: str,
    handoff_verifier_name: str,
    bundle_expected_target_count: int | None,
    bundle_min_trace_count: int | None,
    bundle_min_supported_count: int | None,
    bundle_min_refuted_count: int | None,
    bundle_min_source_citation_count: int | None,
    bundle_require_action_execution_alignment: bool,
) -> tuple[dict[str, Any], ...]:
    adapter_command = (
        "benchmarks/run_world_model_rule_authoring_adapter.py "
        f"--rule-stubs {paths['rule_stubs']} "
        f"--rule-inputs {paths['combined_rule_inputs']} "
        f"--output-dir {paths['adapter_output_dir']} "
        f"--json {paths['adapter_report']} "
        f"--rule-results-jsonl {paths['adapter_rule_results']} "
        f"--input-requests-jsonl {paths['adapter_input_requests']} "
        f"--artifact-manifest {paths['adapter_artifact_manifest']}"
    )
    promotion_command = (
        "benchmarks/promote_world_model_rule_candidates.py "
        f"--rule-results {paths['adapter_rule_results']} "
        f"--rule-inputs {paths['combined_rule_inputs']} "
        f"--adapter-report {paths['adapter_report']} "
        f"--output-dir {paths['promotion_output_dir']} "
        f"--json {paths['promotion_report']} "
        f"--promoted-jsonl {paths['promoted_candidates']} "
        f"--blocked-jsonl {paths['blocked_candidates']} "
        f"--pending-jsonl {paths['pending_inputs']} "
        f"--artifact-manifest {paths['promotion_artifact_manifest']} "
        f"--min-confidence {float(min_confidence):g}"
    )
    commands = [
        {
            "workflow": RULE_ADAPTER_WORKFLOW,
            "command": adapter_command,
            "planned_outputs": (
                str(paths["adapter_report"]),
                str(paths["adapter_rule_results"]),
                str(paths["adapter_input_requests"]),
                str(paths["adapter_artifact_manifest"]),
            ),
            "executes_by_default": False,
        },
        {
            "workflow": PROMOTION_WORKFLOW,
            "command": promotion_command,
            "planned_outputs": (
                str(paths["promotion_report"]),
                str(paths["promoted_candidates"]),
                str(paths["blocked_candidates"]),
                str(paths["pending_inputs"]),
                str(paths["promotion_artifact_manifest"]),
            ),
            "executes_by_default": False,
        },
    ]
    if build_handoff:
        handoff_command = (
            "benchmarks/build_world_model_rule_candidate_handoff.py "
            f"--promotion-gate {paths['promotion_report']} "
            f"--promoted-candidates {paths['promoted_candidates']} "
            f"--output-dir {paths['handoff_output_dir']} "
            f"--json {paths['handoff_report']} "
            f"--trace-jsonl {paths['handoff_product_traces']} "
            f"--action-results-jsonl {paths['handoff_action_results']} "
            f"--artifact-manifest {paths['handoff_artifact_manifest']} "
            f"--route-name {handoff_route_name} "
            f"--verifier-name {handoff_verifier_name}"
        )
        commands.append({
            "workflow": HANDOFF_WORKFLOW,
            "command": handoff_command,
            "planned_outputs": (
                str(paths["handoff_report"]),
                str(paths["handoff_product_traces"]),
                str(paths["handoff_action_results"]),
                str(paths["handoff_artifact_manifest"]),
            ),
            "executes_by_default": False,
        })
    if build_evidence_bundle:
        bundle_command = (
            "benchmarks/build_mechanism_handoff_evidence_bundle.py "
            f"--handoff {paths['handoff_report']} "
            f"--output-dir {paths['evidence_bundle_output_dir']} "
            f"--json {paths['evidence_bundle_report']} "
            f"--artifact-manifest {paths['evidence_bundle_artifact_manifest']}"
            f"{_optional_int_flag('--expected-target-count', bundle_expected_target_count)}"
            f"{_optional_int_flag('--min-trace-count', bundle_min_trace_count)}"
            f"{_optional_int_flag('--min-supported-count', bundle_min_supported_count)}"
            f"{_optional_int_flag('--min-refuted-count', bundle_min_refuted_count)}"
            f"{_optional_int_flag('--min-source-citation-count', bundle_min_source_citation_count)}"
            f"{'' if bundle_require_action_execution_alignment else ' --allow-action-execution-misalignment'}"
        )
        commands.append({
            "workflow": EVIDENCE_BUNDLE_WORKFLOW,
            "command": bundle_command,
            "planned_outputs": (
                str(paths["evidence_bundle_report"]),
                str(paths["evidence_bundle_artifact_manifest"]),
            ),
            "executes_by_default": False,
        })
    return tuple(commands)


def _optional_int_flag(flag: str, value: int | None) -> str:
    return "" if value is None else f" {flag} {int(value)}"


def _public_paths(
    paths: Mapping[str, Path | None],
    *,
    report_path: Path,
    manifest_path: Path,
) -> dict[str, str | None]:
    output: dict[str, str | None] = {
        "report": str(report_path),
        "artifact_manifest": str(manifest_path),
    }
    output.update({key: None if value is None else str(value) for key, value in paths.items()})
    return output


def _manifest_artifacts(
    *,
    source_path: Path | None,
    paths: Mapping[str, Path | None],
    report_path: Path,
    execute: bool,
    handoff_executed: bool,
    evidence_bundle_executed: bool,
) -> dict[str, Path | None]:
    artifacts = {
        "frontier_rule_adapter_promotion_workflow": report_path,
        "input_fill_result_rollup": source_path,
        "rule_stubs": paths.get("rule_stubs"),
        "combined_rule_inputs": paths.get("combined_rule_inputs"),
    }
    if execute:
        artifacts.update({
            "rule_adapter_report": paths.get("adapter_report"),
            "rule_adapter_results": paths.get("adapter_rule_results"),
            "rule_adapter_input_requests": paths.get("adapter_input_requests"),
            "rule_adapter_manifest": paths.get("adapter_artifact_manifest"),
            "rule_promotion_report": paths.get("promotion_report"),
            "promoted_candidates": paths.get("promoted_candidates"),
            "blocked_candidates": paths.get("blocked_candidates"),
            "pending_inputs": paths.get("pending_inputs"),
            "rule_promotion_manifest": paths.get("promotion_artifact_manifest"),
        })
    if handoff_executed:
        artifacts.update({
            "rule_candidate_handoff_report": paths.get("handoff_report"),
            "rule_candidate_handoff_product_traces": paths.get("handoff_product_traces"),
            "rule_candidate_handoff_action_results": paths.get("handoff_action_results"),
            "rule_candidate_handoff_manifest": paths.get("handoff_artifact_manifest"),
        })
    if evidence_bundle_executed:
        artifacts.update({
            "mechanism_handoff_evidence_bundle": paths.get("evidence_bundle_report"),
            "mechanism_handoff_evidence_bundle_manifest": paths.get("evidence_bundle_artifact_manifest"),
        })
    return artifacts


def _load_mapping_source(source: str | Path | Mapping[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    if isinstance(source, Mapping):
        return None, dict(source)
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return path, dict(payload)


def _load_jsonl_mappings(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append(dict(row))
    return tuple(rows)


def _resolve_path(path: str | Path, *, source_root: Path | None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    if source_root is not None and (source_root / candidate).exists():
        return source_root / candidate
    return ROOT / candidate


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = (
        strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        if compact
        else strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    output.write_text(text, encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata value must be KEY=VALUE, got {value!r}.")
        key, item = value.split("=", 1)
        metadata[key] = item
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fill-result-rollup", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rule-stubs", default=None)
    parser.add_argument("--combined-rule-inputs", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--build-handoff", action="store_true")
    parser.add_argument("--build-evidence-bundle", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--handoff-route-name", default=DEFAULT_HANDOFF_ROUTE_NAME)
    parser.add_argument("--handoff-verifier-name", default=DEFAULT_HANDOFF_VERIFIER_NAME)
    parser.add_argument("--bundle-expected-target-count", type=int, default=None)
    parser.add_argument("--bundle-min-trace-count", type=int, default=None)
    parser.add_argument("--bundle-min-supported-count", type=int, default=None)
    parser.add_argument("--bundle-min-refuted-count", type=int, default=None)
    parser.add_argument("--bundle-min-source-citation-count", type=int, default=None)
    parser.add_argument("--bundle-allow-action-execution-misalignment", action="store_true")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run_frontier_research_queue_rule_adapter_promotion_workflow(
        input_fill_result_rollup=args.input_fill_result_rollup,
        output_dir=args.output_dir,
        rule_stubs_path=args.rule_stubs,
        combined_rule_inputs_path=args.combined_rule_inputs,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        execute=bool(args.execute),
        build_handoff=bool(args.build_handoff),
        build_evidence_bundle=bool(args.build_evidence_bundle),
        min_confidence=args.min_confidence,
        handoff_route_name=args.handoff_route_name,
        handoff_verifier_name=args.handoff_verifier_name,
        bundle_expected_target_count=args.bundle_expected_target_count,
        bundle_min_trace_count=args.bundle_min_trace_count,
        bundle_min_supported_count=args.bundle_min_supported_count,
        bundle_min_refuted_count=args.bundle_min_refuted_count,
        bundle_min_source_citation_count=args.bundle_min_source_citation_count,
        bundle_require_action_execution_alignment=not args.bundle_allow_action_execution_misalignment,
        metadata=_parse_metadata(args.metadata),
        compact_json=args.compact_json,
    )
    print(strict_json_dumps({"status": payload["status"], "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
