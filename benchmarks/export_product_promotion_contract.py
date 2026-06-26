"""Export a deployable ProductPromotionContract from release evidence."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.control import ProductPromotionContract  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402


def export_product_promotion_contract(
    *,
    source_path: str | Path,
    output_path: str | Path,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
    release_efficiency_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write a ProductPromotionContract JSON and optional manifest/registry record."""
    source = Path(source_path)
    output = Path(output_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    release_efficiency_report = (
        None
        if release_efficiency_report_path is None
        else Path(release_efficiency_report_path)
    )
    if (name or version) and (registry_path is None or name is None or version is None):
        raise ValueError("registry export requires registry_path, name, and version.")

    contract = ProductPromotionContract.from_json(source)
    contract = _contract_with_release_efficiency(
        contract,
        release_efficiency_report_path=release_efficiency_report,
    )
    payload = contract.to_dict()
    control_policy_config = dict(contract.control_policy_config)
    control_defaults = dict(contract.control_defaults)
    trace_replay_workflow = dict(contract.product_trace_replay_workflow)
    selfcheck_signal_fusion_workflow = dict(contract.selfcheck_signal_fusion_workflow)
    world_model_signal_workflow = dict(contract.world_model_signal_workflow)
    feedback_policy_workflow = dict(contract.feedback_policy_workflow)
    release_efficiency = dict(contract.release_efficiency)
    release_efficiency_metadata = _release_efficiency_flat_metadata(release_efficiency)
    export_metadata = dict(metadata or {})
    _write_json(output, payload, compact=compact_json)

    manifest = None
    if manifest_path is not None:
        artifacts: dict[str, Path] = {
            "product_promotion_contract": output,
            "source_release_candidate": source,
        }
        if release_efficiency_report is not None:
            artifacts["release_efficiency_report"] = release_efficiency_report
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "runner": "export_product_promotion_contract",
                "source": str(source),
                "compact_json": compact_json,
                **release_efficiency_metadata,
                **export_metadata,
            },
        )
        _write_json(manifest_path, manifest, compact=compact_json)

    if registry_path is not None and name is not None and version is not None:
        registry = ArtifactRegistry.load_json(registry_path)
        registry.record_product_promotion_contract(
            name=name,
            path=output,
            version=version,
            metadata={
                "workflow": "export_product_promotion_contract",
                "source": str(source),
                "artifact_manifest": None if manifest_path is None else str(manifest_path),
                "source_workflow": contract.source_workflow,
                "source_status": contract.source_status,
                "model_id": contract.model_id,
                "control_policy_config": control_policy_config,
                "control_defaults": control_defaults,
                "control_default_max_verifier_route_attempts": control_defaults.get(
                    "max_verifier_route_attempts"
                ),
                "recommended_route": contract.metadata.get("recommended_route"),
                "recommended_selector_replay_candidate": contract.metadata.get(
                    "recommended_selector_replay_candidate"
                ),
                "product_runtime_drift_status": contract.metadata.get("product_runtime_drift_status"),
                "max_covariance_maha_last_auroc_drop": contract.metadata.get(
                    "max_covariance_maha_last_auroc_drop"
                ),
                "readiness_covariance_tradeoff_gate_passed": contract.metadata.get(
                    "readiness_covariance_tradeoff_gate_passed"
                ),
                "readiness_covariance_tradeoff_status": contract.metadata.get(
                    "readiness_covariance_tradeoff_status"
                ),
                "readiness_covariance_selected_mode": contract.metadata.get(
                    "readiness_covariance_selected_mode"
                ),
                "readiness_covariance_maha_last_delta_vs_baseline": contract.metadata.get(
                    "readiness_covariance_maha_last_delta_vs_baseline"
                ),
                "performance_covariance_tradeoff_gate_passed": contract.metadata.get(
                    "performance_covariance_tradeoff_gate_passed"
                ),
                "performance_covariance_tradeoff_status": contract.metadata.get(
                    "performance_covariance_tradeoff_status"
                ),
                "performance_covariance_selected_mode": contract.metadata.get(
                    "performance_covariance_selected_mode"
                ),
                "performance_covariance_maha_last_delta_vs_baseline": contract.metadata.get(
                    "performance_covariance_maha_last_delta_vs_baseline"
                ),
                "performance_best_quality_signal": contract.metadata.get(
                    "performance_best_quality_signal"
                ),
                "performance_best_quality_auroc": contract.metadata.get(
                    "performance_best_quality_auroc"
                ),
                "performance_score_fusion_status": contract.metadata.get(
                    "performance_score_fusion_status"
                ),
                "performance_score_fusion_signal": contract.metadata.get(
                    "performance_score_fusion_signal"
                ),
                "performance_score_fusion_auroc": contract.metadata.get(
                    "performance_score_fusion_auroc"
                ),
                "performance_score_fusion_conformal_gate_passed": contract.metadata.get(
                    "performance_score_fusion_conformal_gate_passed"
                ),
                "performance_selected_fusion_status": contract.metadata.get(
                    "performance_selected_fusion_status"
                ),
                "performance_selected_fusion_run": contract.metadata.get(
                    "performance_selected_fusion_run"
                ),
                "performance_selected_fusion_candidate": contract.metadata.get(
                    "performance_selected_fusion_candidate"
                ),
                "performance_selected_fusion_signal": contract.metadata.get(
                    "performance_selected_fusion_signal"
                ),
                "performance_selected_fusion_auroc": contract.metadata.get(
                    "performance_selected_fusion_auroc"
                ),
                "performance_selected_fusion_false_alarm": contract.metadata.get(
                    "performance_selected_fusion_false_alarm"
                ),
                "performance_selected_fusion_detection": contract.metadata.get(
                    "performance_selected_fusion_detection"
                ),
                "performance_selected_fusion_artifact_report": contract.metadata.get(
                    "performance_selected_fusion_artifact_report"
                ),
                "performance_selected_fusion_artifact_path": contract.metadata.get(
                    "performance_selected_fusion_artifact_path"
                ),
                "product_trace_replay_workflow_report": trace_replay_workflow.get("report_path"),
                "product_trace_replay_workflow_manifest": trace_replay_workflow.get(
                    "manifest_path"
                ),
                "product_trace_replay_workflow_source": trace_replay_workflow.get("source"),
                "product_trace_replay_workflow_registry": trace_replay_workflow.get(
                    "registry"
                ),
                "product_trace_replay_workflow_record": trace_replay_workflow.get(
                    "record_key"
                ),
                "product_trace_replay_workflow_selector_replay_report": (
                    trace_replay_workflow.get("selector_replay_report_path")
                ),
                "product_trace_replay_workflow_runtime_drift_report": (
                    trace_replay_workflow.get("product_runtime_drift_report_path")
                ),
                "selfcheck_signal_fusion_workflow_report": (
                    selfcheck_signal_fusion_workflow.get("report_path")
                ),
                "selfcheck_signal_fusion_workflow_manifest": (
                    selfcheck_signal_fusion_workflow.get("manifest_path")
                ),
                "selfcheck_signal_fusion_workflow_source": (
                    selfcheck_signal_fusion_workflow.get("source")
                ),
                "selfcheck_signal_fusion_workflow_registry": (
                    selfcheck_signal_fusion_workflow.get("registry")
                ),
                "selfcheck_signal_fusion_workflow_record": (
                    selfcheck_signal_fusion_workflow.get("record_key")
                ),
                "selfcheck_signal_fusion_workflow_sample_quality_status": (
                    selfcheck_signal_fusion_workflow.get("sample_quality_status")
                ),
                "selfcheck_signal_fusion_workflow_sample_quality_passed": (
                    selfcheck_signal_fusion_workflow.get("sample_quality_passed")
                ),
                "selfcheck_signal_fusion_workflow_failed_runs": (
                    selfcheck_signal_fusion_workflow.get("sample_quality_failed_runs")
                ),
                "selfcheck_signal_fusion_workflow_sample_quality_run_count": (
                    selfcheck_signal_fusion_workflow.get("sample_quality_run_count")
                ),
                "selfcheck_signal_fusion_workflow_fusion_run_count": (
                    selfcheck_signal_fusion_workflow.get("fusion_run_count")
                ),
                "selfcheck_signal_fusion_workflow_geometry_artifact_count": (
                    selfcheck_signal_fusion_workflow.get("geometry_fusion_artifact_count")
                ),
                "selfcheck_signal_fusion_workflow_enhanced_score_dump_count": (
                    selfcheck_signal_fusion_workflow.get("enhanced_score_dump_count")
                ),
                "world_model_signal_workflow_report": (
                    world_model_signal_workflow.get("report_path")
                ),
                "world_model_signal_workflow_manifest": (
                    world_model_signal_workflow.get("manifest_path")
                ),
                "world_model_signal_workflow_source": (
                    world_model_signal_workflow.get("source")
                ),
                "world_model_signal_workflow_registry": (
                    world_model_signal_workflow.get("registry")
                ),
                "world_model_signal_workflow_record": (
                    world_model_signal_workflow.get("record_key")
                ),
                "world_model_signal_workflow_release_gate_status": (
                    world_model_signal_workflow.get("release_gate_status")
                ),
                "world_model_signal_workflow_trace_gap_max": (
                    world_model_signal_workflow.get("trace_gap_max")
                ),
                "world_model_signal_workflow_conflict_positive_count": (
                    world_model_signal_workflow.get("conflict_positive_count")
                ),
                "world_model_signal_workflow_calibrated_conflict_signal_count": (
                    world_model_signal_workflow.get("calibrated_conflict_signal_count")
                ),
                "feedback_policy_workflow_report": feedback_policy_workflow.get("report_path"),
                "feedback_policy_workflow_manifest": feedback_policy_workflow.get(
                    "manifest_path"
                ),
                "feedback_policy_workflow_source": feedback_policy_workflow.get("source"),
                "feedback_policy_workflow_registry": feedback_policy_workflow.get("registry"),
                "feedback_policy_workflow_record": feedback_policy_workflow.get("record_key"),
                "feedback_policy_workflow_promotion_decision": (
                    feedback_policy_workflow.get("promotion_decision")
                ),
                "feedback_policy_workflow_candidate_control_policy": (
                    feedback_policy_workflow.get("candidate_control_policy")
                ),
                "feedback_policy_workflow_candidate_control_defaults": (
                    feedback_policy_workflow.get("candidate_control_defaults")
                ),
                "feedback_policy_workflow_matched_feedback_count": (
                    feedback_policy_workflow.get("matched_feedback_count")
                ),
                "feedback_policy_workflow_final_answered_but_wrong_rate": (
                    feedback_policy_workflow.get("final_answered_but_wrong_rate")
                ),
                "feedback_policy_workflow_final_answer_false_block_rate": (
                    feedback_policy_workflow.get("final_answer_false_block_rate")
                ),
                "feedback_policy_workflow_safety_coverage_rate": (
                    feedback_policy_workflow.get("safety_coverage_rate")
                ),
                "feedback_policy_workflow_unknown_safety_issue_rate": (
                    feedback_policy_workflow.get("unknown_safety_issue_rate")
                ),
                **release_efficiency_metadata,
                "compact_json": compact_json,
                **export_metadata,
            },
        )
        registry.save_json()

    return {
        "schema_version": 1,
        "workflow": "export_product_promotion_contract",
        "status": "exported",
        "paths": {
            "source": str(source),
            "contract": str(output),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "registry": None if registry_path is None else str(registry_path),
            "release_efficiency_report": (
                None if release_efficiency_report is None else str(release_efficiency_report)
            ),
        },
        "contract": {
            "model_id": contract.model_id,
            "source_workflow": contract.source_workflow,
            "source_status": contract.source_status,
            "runtime": dict(contract.runtime),
            "verifier_route": dict(contract.verifier_route),
            "control_policy_config": control_policy_config,
            "control_defaults": control_defaults,
            "product_trace_replay_workflow": trace_replay_workflow,
            "selfcheck_signal_fusion_workflow": selfcheck_signal_fusion_workflow,
            "world_model_signal_workflow": world_model_signal_workflow,
            "feedback_policy_workflow": feedback_policy_workflow,
            "release_efficiency": release_efficiency,
            "metadata": dict(contract.metadata),
        },
        "artifact_manifest_summary": None if manifest is None else manifest.get("summary"),
    }


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _contract_with_release_efficiency(
    contract: ProductPromotionContract,
    *,
    release_efficiency_report_path: Path | None,
) -> ProductPromotionContract:
    if release_efficiency_report_path is None:
        return contract
    release_efficiency = {
        **dict(contract.release_efficiency),
        **_load_release_efficiency_summary(release_efficiency_report_path),
    }
    metadata = {
        **dict(contract.metadata),
        **_release_efficiency_flat_metadata(release_efficiency),
    }
    return replace(
        contract,
        release_efficiency=release_efficiency,
        metadata=metadata,
    )


def _load_release_efficiency_summary(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    leaderboard = payload.get("leaderboard")
    top = _mapping(leaderboard[0]) if isinstance(leaderboard, (list, tuple)) and leaderboard else {}
    return _drop_none_values({
        "report_path": str(path),
        "workflow": payload.get("workflow"),
        "status": _first_present(payload.get("status"), _nested(payload, "decision", "status")),
        "manifest_path": _nested(payload, "paths", "artifact_manifest"),
        "profile_sweep_path": _nested(payload, "paths", "profile_sweep"),
        "quality_report_paths": _nested(payload, "paths", "quality_reports"),
        "recommended_profile": _nested(payload, "decision", "recommended_profile"),
        "recommended_efficiency_score": _first_present(
            _nested(payload, "decision", "recommended_efficiency_score"),
            _nested(top, "efficiency", "score"),
        ),
        "blocking_reasons": _nested(payload, "decision", "blocking_reasons"),
        "profile_count": _nested(payload, "summary", "profile_count"),
        "blocked_profile_count": _nested(payload, "summary", "blocked_profile_count"),
        "generated_trace_count": _nested(payload, "summary", "generated_trace_count"),
        "reused_trace_count": _nested(payload, "summary", "reused_trace_count"),
        "quality_passed": _nested(payload, "summary", "quality_passed"),
        "trace_record_cache_enabled_profile_count": _nested(
            payload,
            "summary",
            "trace_record_cache_enabled_profile_count",
        ),
        "trace_record_cache_hit_profile_count": _nested(
            payload,
            "summary",
            "trace_record_cache_hit_profile_count",
        ),
        "trace_record_cache_written_profile_count": _nested(
            payload,
            "summary",
            "trace_record_cache_written_profile_count",
        ),
        "leaderboard_top_profile": top.get("profile"),
        "leaderboard_top_efficiency_score": _nested(top, "efficiency", "score"),
    })


def _release_efficiency_flat_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    return _drop_none_values({
        "release_efficiency_report": report.get("report_path"),
        "release_efficiency_manifest": report.get("manifest_path"),
        "release_efficiency_status": report.get("status"),
        "release_efficiency_recommended_profile": report.get("recommended_profile"),
        "release_efficiency_score": report.get("recommended_efficiency_score"),
        "release_efficiency_profile_count": report.get("profile_count"),
        "release_efficiency_quality_passed": report.get("quality_passed"),
        "release_efficiency_trace_record_cache_hit_profile_count": report.get(
            "trace_record_cache_hit_profile_count"
        ),
    })


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _nested(payload: Mapping[str, Any], *path: str) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _drop_none_values(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if item is not None}


def _parse_metadata(values: Sequence[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--metadata entries must be formatted as key=value.")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--metadata keys must be non-empty.")
        try:
            metadata[key] = json.loads(raw)
        except json.JSONDecodeError:
            metadata[key] = raw
    return metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = export_product_promotion_contract(
        source_path=args.source,
        output_path=args.output,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
        release_efficiency_report_path=args.release_efficiency_report,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export a lightweight ProductPromotionContract from release evidence"
    )
    parser.add_argument("--source", required=True, help="release workflow/comparison or product contract JSON")
    parser.add_argument("--output", required=True, help="output ProductPromotionContract JSON")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest output path")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="optional registry record name")
    parser.add_argument("--version", default=None, help="optional registry record version")
    parser.add_argument("--metadata", action="append", default=[], help="metadata key=value; repeatable")
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON artifacts")
    parser.add_argument(
        "--release-efficiency-report",
        default=None,
        help="optional release-efficiency report to embed as promotion evidence",
    )
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
