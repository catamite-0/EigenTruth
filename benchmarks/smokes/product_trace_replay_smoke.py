"""No-model smoke checks for the ProductTrace replay workflow."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.lib.paths import ensure_repo_root_on_path

REPO_ROOT = ensure_repo_root_on_path()
from benchmarks.run_product_trace_replay_workflow import (  # noqa: E402
    ProductTraceReplayWorkflowConfig,
    run_product_trace_replay_workflow,
)
from benchmarks.run_runtime_profile_selector_tuning import RuntimeProfileSelectorCandidate  # noqa: E402
from eigentruth.registry import ArtifactRegistry, load_and_verify_artifact_manifest  # noqa: E402

SMOKE_NAME = "product-trace-replay-smoke"
SMOKE_VERSION = "0.1"


def build_product_trace_replay_smoke(output_dir: Path) -> dict[str, Any]:
    """Run a synthetic full-trace replay workflow and bounded-trace rejection check."""
    output_dir.mkdir(parents=True, exist_ok=True)
    full_trace_dir = output_dir / "full-traces"
    full_trace_dir.mkdir(parents=True, exist_ok=True)
    full_trace_paths = _write_full_traces(full_trace_dir)
    replay_policy_path = output_dir / "replay-policy.json"
    replay_policy_path.write_text(
        json.dumps({
            "max_estimated_cost_units_mean": 2.0,
            "min_observed_runtime_coverage_rate": 1.0,
            "min_selected_profile_counts": {
                "latency": 1,
                "balanced": 1,
                "audit": 1,
            },
        }),
        encoding="utf-8",
    )

    workflow_report = run_product_trace_replay_workflow(
        ProductTraceReplayWorkflowConfig(
            trace_paths=full_trace_paths,
            output_dir=output_dir / "workflow",
            candidates=_selector_candidates(),
            replay_policy_path=replay_policy_path,
            registry_path=output_dir / "registry.json",
            name=SMOKE_NAME,
            version=SMOKE_VERSION,
            require_runtime_trace=True,
            compact_json=True,
            verify_manifest=True,
            fingerprint_cache_path=output_dir / "workflow" / "fingerprints.json",
            max_action_audit_error_rate=0.0,
            max_action_audit_missing_retrieval_rate=0.0,
            max_action_audit_malformed_payload_rate=0.0,
            max_action_audit_unexpected_action_rate=0.0,
            corpus_cache_path=output_dir / "workflow" / "corpus-cache.json",
            corpus_source_cache_path=output_dir / "workflow" / "corpus" / "source-cache.json",
            runtime_trace_records_cache_path=(
                output_dir / "workflow" / "runtime-baseline" / "trace-record-cache.json"
            ),
            runtime_recommended_policy_path=(
                output_dir / "workflow" / "runtime-baseline" / "recommended-policy.json"
            ),
            selector_trace_inputs_path=output_dir / "workflow" / "selector-replay" / "trace-inputs.json",
        )
    )
    if workflow_report["status"] != "promote":
        raise AssertionError("ProductTrace replay smoke workflow did not promote.")
    if workflow_report["corpus"]["accepted_count"] != len(full_trace_paths):
        raise AssertionError("ProductTrace replay smoke did not accept all full traces.")
    if workflow_report["selector_replay"]["status"] != "promote":
        raise AssertionError("ProductTrace replay smoke selector replay did not promote.")
    if workflow_report["action_audit_gate"]["status"] != "promote":
        raise AssertionError("ProductTrace replay smoke action-audit gate did not promote.")
    if workflow_report["manifest_verification"]["verification"]["passed"] is not True:
        raise AssertionError("ProductTrace replay smoke manifest verification failed.")
    if not load_and_verify_artifact_manifest(
        workflow_report["paths"]["artifact_manifest"],
        recursive=True,
    ).passed:
        raise AssertionError("ProductTrace replay smoke artifact manifest failed recursive verification.")
    registry_record = ArtifactRegistry.load_json(output_dir / "registry.json").get(
        f"report:{SMOKE_NAME}:{SMOKE_VERSION}"
    )

    bounded_report = _run_bounded_rejection_check(output_dir)
    return {
        "output_dir": str(output_dir),
        "workflow_report": workflow_report,
        "bounded_rejection_report": bounded_report,
        "registry_record": registry_record.to_dict(),
    }


def _write_full_traces(output_dir: Path) -> tuple[Path, ...]:
    payloads = (
        {
            "request_id": "latency-low-supported",
            "risk_decision": {
                "action": "accept",
                "risk_level": "low",
                "confidence": 1.0,
                "reason": "supported",
            },
            "claims": [{"claim_id": "c1", "text": "Private low-risk fact.", "metadata": {}}],
            "actions": [
                {
                    "action": "accept",
                    "reason": "supported",
                    "payload": {"mode": "pass_through", "claim_ids": ["c1"]},
                }
            ],
            "verification_plan": {
                "run_verifier": True,
                "reason": "all claims selected",
                "verification_scope": "all",
                "claims": [{"claim_id": "c1", "text": "Private low-risk fact.", "metadata": {}}],
                "verify_claim_ids": ["c1"],
                "skipped_claim_ids": [],
                "triggered_claim_ids": [],
                "triggered_features": {},
                "triggered_metadata": {},
                "route_hints": [
                    {
                        "claim_id": "c1",
                        "routes": ["groundedness"],
                        "reasons": ["smoke"],
                        "metadata": {},
                    }
                ],
                "retrieval_queries": [],
                "calculation_checks": [],
                "state_checks": [],
                "world_model_checks": [],
                "dependencies": [],
            },
            "metadata": {"runtime_profile": "latency"},
            "runtime_trace": {"total_seconds": 0.10, "phases": []},
        },
        {
            "request_id": "balanced-medium-retrieve",
            "risk_decision": {
                "action": "retrieve",
                "risk_level": "medium",
                "confidence": 0.7,
                "reason": "unsupported",
            },
            "claims": [{"claim_id": "c1", "text": "Private unsupported fact.", "metadata": {}}],
            "verification_plan": {
                "run_verifier": True,
                "reason": "unsupported claim needs retrieval",
                "verification_scope": "all",
                "claims": [{"claim_id": "c1", "text": "Private unsupported fact.", "metadata": {}}],
                "verify_claim_ids": ["c1"],
                "skipped_claim_ids": [],
                "triggered_claim_ids": ["c1"],
                "triggered_features": {},
                "triggered_metadata": {"unsupported": 1},
                "route_hints": [
                    {
                        "claim_id": "c1",
                        "routes": ["retrieval"],
                        "reasons": ["unsupported"],
                        "metadata": {},
                    }
                ],
                "retrieval_queries": [
                    {"claim_id": "c1", "query": "Private unsupported fact"}
                ],
                "calculation_checks": [],
                "state_checks": [],
                "world_model_checks": [],
                "dependencies": [],
            },
            "actions": [
                {
                    "action": "retrieve",
                    "reason": "unsupported",
                    "payload": {
                        "retrieval_targets": [
                            {"claim_id": "c1", "query": "Private unsupported fact"}
                        ]
                    },
                }
            ],
            "metadata": {"runtime_profile": "balanced"},
            "runtime_trace": {"total_seconds": 0.20, "phases": []},
        },
        {
            "request_id": "audit-low-sensitive",
            "risk_decision": {
                "action": "accept",
                "risk_level": "low",
                "confidence": 0.9,
                "reason": "numbered claim",
            },
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "Private account balance is 42.",
                    "metadata": {"features": {"has_number": True}},
                }
            ],
            "actions": [
                {
                    "action": "accept",
                    "reason": "numbered claim verified by smoke fixture",
                    "payload": {"mode": "pass_through", "claim_ids": ["c1"]},
                }
            ],
            "metadata": {"runtime_profile": "audit"},
            "runtime_trace": {"total_seconds": 0.40, "phases": []},
        },
    )
    paths = []
    for index, payload in enumerate(payloads):
        trace_path = output_dir / f"trace-{index}.json"
        trace_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        paths.append(trace_path)
    return tuple(paths)


def _selector_candidates() -> tuple[RuntimeProfileSelectorCandidate, ...]:
    return (
        RuntimeProfileSelectorCandidate(name="default", policy={}),
        RuntimeProfileSelectorCandidate(
            name="latency-biased",
            policy={
                "sensitive_claim_feature_flags": ["has_citation", "is_time_sensitive"],
            },
        ),
    )


def _run_bounded_rejection_check(output_dir: Path) -> dict[str, Any]:
    bounded_path = output_dir / "bounded-trace.json"
    bounded_path.write_text(
        json.dumps({
            "schema_version": 1,
            "trace_format": "bounded_product_trace",
            "request_id": "bounded",
            "risk_decision": {
                "action": "accept",
                "risk_level": "low",
                "confidence": 1.0,
                "reason": "bounded telemetry only",
            },
            "summaries": {
                "verification_plan": {
                    "available": False,
                    "claim_count": 0,
                    "route_counts": {},
                    "tool_payload_counts": {},
                }
            },
        }),
        encoding="utf-8",
    )
    try:
        run_product_trace_replay_workflow(
            ProductTraceReplayWorkflowConfig(
                trace_paths=(bounded_path,),
                output_dir=output_dir / "bounded-workflow",
                candidates=(RuntimeProfileSelectorCandidate(name="default", policy={}),),
                strict=True,
            )
        )
    except ValueError as exc:
        reason = str(exc)
        if "bounded ProductTrace telemetry" not in reason:
            raise
        return {
            "status": "blocked",
            "decision": {
                "status": "blocked",
                "blocking_reasons": (reason,),
            },
        }
    raise AssertionError("bounded ProductTrace replay smoke was not blocked.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the no-model ProductTrace replay smoke check")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="optional output directory; defaults to a temporary directory",
    )
    args = parser.parse_args(argv)
    if args.output_dir is not None:
        report = build_product_trace_replay_smoke(Path(args.output_dir))
        _print_report(report)
        return
    with tempfile.TemporaryDirectory(prefix="eigentruth-product-trace-replay-smoke-") as tmpdir:
        report = build_product_trace_replay_smoke(Path(tmpdir))
        _print_report(report)


def _print_report(report: Mapping[str, Any]) -> None:
    workflow = report["workflow_report"]
    bounded = report["bounded_rejection_report"]
    print(
        "product_trace_replay_smoke_ok "
        f"status={workflow['status']} "
        f"action_audit={workflow['action_audit_gate']['status']} "
        f"selector={workflow['selector_replay']['recommended_candidate']} "
        f"bounded_status={bounded['status']}"
    )


if __name__ == "__main__":
    main()
