"""Build a read-only frontier product/research status report."""

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

from eigentruth.control import plan_evidence_gaps_from_release_candidate  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "frontier_status_report"

DEFAULT_RELEASE_CANDIDATE = (
    ROOT / "artifacts" / "frontier-audit-release-candidate-v15" / "frontier-audit-comparison.json"
)
DEFAULT_PRODUCT_CONTRACT = (
    ROOT
    / "artifacts"
    / "smollm2_product_promotion_evidence_handoff_v1_9_frontier_v7"
    / "product-promotion-contract-evidence-handoff.json"
)
DEFAULT_EVIDENCE_GAP_PLAN = (
    ROOT / "artifacts" / "frontier-audit-release-candidate-v12" / "evidence-gap-plan.json"
)


def build_frontier_status_report(
    *,
    release_candidate: str | Path | Mapping[str, Any],
    product_contract: str | Path | Mapping[str, Any],
    evidence_gap_plan: str | Path | Mapping[str, Any] | None = None,
    research_queue_source: str | Path | Mapping[str, Any] | None = None,
    refresh_research_queue: bool = False,
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize productized frontier evidence and remaining research queues."""
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    release_path, release_payload = _load_mapping_source(release_candidate)
    contract_path, contract_payload = _load_mapping_source(product_contract)
    gap_path, gap_payload = (
        (None, {})
        if evidence_gap_plan is None
        else _load_mapping_source(evidence_gap_plan)
    )
    if release_payload.get("workflow") != "release_candidate_comparison":
        raise ValueError("release_candidate must have workflow='release_candidate_comparison'.")
    if contract_payload.get("workflow") != "product_promotion_contract":
        raise ValueError("product_contract must have workflow='product_promotion_contract'.")
    if gap_payload and gap_payload.get("workflow") != "evidence_gap_plan":
        raise ValueError("evidence_gap_plan must have workflow='evidence_gap_plan'.")

    release_summary = _release_summary(release_payload)
    contract_summary = _contract_summary(contract_payload)
    research_source_path = None
    research_queue_payload = gap_payload
    research_refresh_status = "not_requested"
    if refresh_research_queue:
        research_source_path, research_source_payload = _load_research_queue_source(
            research_queue_source=research_queue_source,
            gap_payload=gap_payload,
            gap_path=gap_path,
            release_path=release_path,
            release_payload=release_payload,
        )
        refreshed_plan = plan_evidence_gaps_from_release_candidate(
            research_source_payload,
            source_path=research_source_path,
            metadata={"refreshed_from_frontier_status_report": True},
        ).to_dict()
        research_queue_payload = refreshed_plan
        research_refresh_status = "refreshed"
    blockers = _active_blockers(release_summary=release_summary, contract_summary=contract_summary)
    research_queue = _research_queue(
        research_queue_payload,
        refresh_status=research_refresh_status,
        original_payload=None if research_queue_payload is gap_payload else gap_payload,
        research_source_path=research_source_path,
        current_frontier_release_evidence_report=release_summary["frontier_release_evidence"].get(
            "report_path"
        )
        or release_summary.get("frontier_release_evidence_report"),
        productized_blocking_reasons=blockers,
    )
    status = "promote" if not blockers else "needs_evidence"
    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "productized_status": {
            "release_candidate": release_summary,
            "product_contract": contract_summary,
            "blocking_reasons": tuple(blockers),
        },
        "research_queue": research_queue,
        "paths": {
            "frontier_status_report": None if output_path is None else str(output_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "release_candidate": None if release_path is None else str(release_path),
            "product_contract": None if contract_path is None else str(contract_path),
            "evidence_gap_plan": None if gap_path is None else str(gap_path),
            "research_queue_source": None
            if research_source_path is None
            else str(research_source_path),
        },
        "metadata": dict(metadata or {}),
    }
    if output_path is not None:
        _write_json(output_path, payload, compact=compact_json)
    manifest = None
    if manifest_path is not None:
        manifest = _write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            release_path=release_path,
            contract_path=contract_path,
            gap_path=gap_path,
            research_source_path=research_source_path,
            payload=payload,
            metadata=metadata or {},
            compact=compact_json,
        )
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output_path if output_path is not None else release_path,
            metadata={
                "workflow": WORKFLOW,
                "status": status,
                "release_candidate_status": release_summary["status"],
                "product_contract_status": contract_summary["status"],
                "productized_blocking_count": len(blockers),
                "required_evidence_group_count": contract_summary["required_evidence_group_count"],
                "blocked_evidence_group_count": contract_summary["blocked_evidence_group_count"],
                "research_action_count": research_queue["action_count"],
                "research_active_action_count": research_queue["active_action_count"],
                "research_superseded_action_count": research_queue[
                    "superseded_action_count"
                ],
                "research_gap_count": research_queue["gap_count"],
                "research_lifecycle_status": research_queue["lifecycle_status"],
                "research_source_alignment_status": research_queue["source_alignment"][
                    "status"
                ],
                "research_refresh_status": research_queue["refresh_status"],
                "manifest_summary": {} if manifest is None else manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _release_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    decision = _mapping(payload.get("decision"))
    candidate = _mapping(payload.get("release_candidate"))
    frontier = _mapping(payload.get("frontier_release_evidence_gate"))
    runtime_drift = _mapping(payload.get("product_runtime_drift_gate"))
    readiness = _mapping(payload.get("readiness_baseline_comparison"))
    readiness_decision = _mapping(readiness.get("decision"))
    gate_statuses = _status_fields(decision)
    promoted_gate_count = sum(1 for status in gate_statuses.values() if status == "promote")
    blocked_gate_count = sum(1 for status in gate_statuses.values() if status == "blocked")
    return {
        "workflow": payload.get("workflow"),
        "status": str(decision.get("status") or "unknown"),
        "model": decision.get("recommended_model"),
        "route": decision.get("recommended_route"),
        "readiness_record": decision.get("recommended_readiness_record"),
        "readiness_best_quality_signal": _mapping(
            readiness_decision.get("recommended_best_quality_signal")
        ),
        "performance_record": decision.get("recommended_performance_baseline_record"),
        "runtime_drift_report": decision.get("recommended_product_runtime_drift_report"),
        "frontier_release_evidence_report": decision.get(
            "recommended_frontier_release_evidence_report"
        ),
        "required_adapter_routes": tuple(decision.get("required_adapter_routes") or ()),
        "required_route_records": tuple(decision.get("required_route_baseline_records") or ()),
        "gate_statuses": gate_statuses,
        "promoted_gate_count": promoted_gate_count,
        "blocked_gate_count": blocked_gate_count,
        "blocking_reasons": tuple(decision.get("blocking_reasons") or ()),
        "frontier_release_evidence": {
            "status": frontier.get("status"),
            "decision_status": frontier.get("decision_status"),
            "report_path": frontier.get("report_path"),
            "manifest_path": frontier.get("manifest_path"),
            "run_names": tuple(frontier.get("run_names") or ()),
            "promoted_rerun_tracks": tuple(
                frontier.get("frontier_rerun_rollup_promoted_tracks") or ()
            ),
            "input_manifest_verified_count": frontier.get("input_manifest_verified_count"),
            "input_manifest_required_count": frontier.get("input_manifest_required_count"),
        },
        "product_runtime_drift": {
            "status": runtime_drift.get("decision_status") or runtime_drift.get("status"),
            "report_path": runtime_drift.get("report_path"),
            "manifest_path": runtime_drift.get("manifest_path"),
            "metric_count": len(_sequence(runtime_drift.get("metrics"))),
        },
        "source_summary": {
            "release_candidate_name": candidate.get("name"),
            "release_policy_profile": _nested(payload, "config", "release_policy_profile"),
        },
    }


def _contract_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(payload.get("summary"))
    runtime = _mapping(payload.get("runtime"))
    metadata = _mapping(payload.get("metadata"))
    evidence_groups = _mapping(summary.get("evidence_groups"))
    required_groups = {
        name: _mapping(group)
        for name, group in evidence_groups.items()
        if _mapping(group).get("required") is True
    }
    blocked_required_groups = {
        name: group
        for name, group in required_groups.items()
        if _int_or_zero(group.get("blocked_metric_count")) > 0
    }
    gate_statuses = _mapping(summary.get("gate_statuses"))
    return {
        "workflow": payload.get("workflow"),
        "status": str(payload.get("source_status") or gate_statuses.get("source") or "unknown"),
        "model": payload.get("model_id") or summary.get("model_id"),
        "recommended_runtime_seconds": metadata.get("recommended_runtime_seconds")
        or metadata.get("cache_only_total_seconds"),
        "runtime": {
            "layer": runtime.get("layer"),
            "batch_size": runtime.get("batch_size"),
            "max_workers": runtime.get("max_workers"),
            "hidden_state_capture": runtime.get("hidden_state_capture"),
            "prefix_kv_cache": runtime.get("prefix_kv_cache"),
        },
        "gate_statuses": dict(sorted(gate_statuses.items())),
        "available_gate_count": summary.get("available_gate_count"),
        "promoted_gate_count": summary.get("promoted_gate_count"),
        "blocking_gate_count": summary.get("blocking_gate_count"),
        "required_evidence_group_count": len(required_groups),
        "blocked_evidence_group_count": len(blocked_required_groups),
        "required_evidence_groups": {
            name: {
                "metric_count": group.get("metric_count"),
                "blocked_metric_count": group.get("blocked_metric_count"),
            }
            for name, group in sorted(required_groups.items())
        },
        "blocked_evidence_groups": tuple(sorted(blocked_required_groups)),
        "frontier_release_evidence": {
            "decision_status": _nested(payload, "frontier_release_evidence", "decision_status"),
            "report_path": _nested(payload, "frontier_release_evidence", "report_path"),
            "run_names": tuple(
                _sequence(_nested(payload, "frontier_release_evidence", "run_names"))
            ),
            "promoted_rerun_tracks": tuple(
                _sequence(
                    _nested(
                        payload,
                        "frontier_release_evidence",
                        "frontier_rerun_rollup_promoted_tracks",
                    )
                )
            ),
        },
    }


def _research_queue(
    payload: Mapping[str, Any],
    *,
    refresh_status: str = "not_requested",
    original_payload: Mapping[str, Any] | None = None,
    research_source_path: Path | None = None,
    current_frontier_release_evidence_report: Any = None,
    productized_blocking_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    if not payload:
        return {
            "status": "not_provided",
            "lifecycle_status": "not_provided",
            "active": False,
            "workflow": None,
            "source_workflow": None,
            "source_path": None,
            "refresh_status": refresh_status,
            "original_action_count": None,
            "original_gap_count": None,
            "source_status": None,
            "gap_count": 0,
            "action_count": 0,
            "active_action_count": 0,
            "superseded_action_count": 0,
            "missing_metric_count": 0,
            "top_action_ids": (),
            "source_alignment": {
                "status": "not_applicable",
                "is_current": None,
                "research_source_path": None,
                "current_frontier_release_evidence_report": current_frontier_release_evidence_report,
                "reason": "no research queue was provided",
            },
            "actions": (),
            "gaps": (),
        }
    summary = _mapping(payload.get("summary"))
    actions = tuple(_action_summary(action) for action in _mapping_sequence(payload.get("actions", ())))
    gaps = tuple(_gap_summary(gap) for gap in _mapping_sequence(payload.get("gaps", ())))
    original_summary = _mapping(original_payload.get("summary")) if original_payload else {}
    source_path = payload.get("source_path")
    if source_path is None and research_source_path is not None:
        source_path = str(research_source_path)
    source_status = payload.get("source_status") or summary.get("source_decision_status")
    gap_count = summary.get("gap_count", len(gaps))
    action_count = summary.get("action_count", len(actions))
    alignment = _research_source_alignment(
        source_path=source_path,
        source_status=source_status,
        current_frontier_release_evidence_report=current_frontier_release_evidence_report,
    )
    lifecycle_status = _research_queue_lifecycle_status(
        action_count=_int_or_zero(action_count),
        gap_count=_int_or_zero(gap_count),
        source_status=source_status,
        alignment_status=alignment["status"],
        productized_blocking_reasons=productized_blocking_reasons,
    )
    active = lifecycle_status in {"active", "current_blocker"}
    superseded = lifecycle_status == "superseded"
    return {
        "status": payload.get("status"),
        "lifecycle_status": lifecycle_status,
        "active": active,
        "workflow": payload.get("workflow"),
        "source_workflow": payload.get("source_workflow"),
        "source_path": source_path,
        "refresh_status": refresh_status,
        "original_action_count": original_summary.get("action_count"),
        "original_gap_count": original_summary.get("gap_count"),
        "source_status": source_status,
        "gap_count": gap_count,
        "action_count": action_count,
        "active_action_count": action_count if active else 0,
        "superseded_action_count": action_count if superseded else 0,
        "missing_metric_count": summary.get("missing_metric_count", 0),
        "gates": dict(_mapping(summary.get("gates"))),
        "research_axes": dict(_mapping(summary.get("research_axes"))),
        "root_causes": dict(_mapping(summary.get("root_causes"))),
        "top_action_ids": tuple(summary.get("top_action_ids") or ()),
        "source_alignment": alignment,
        "actions": actions,
        "gaps": gaps,
    }


def _research_source_alignment(
    *,
    source_path: Any,
    source_status: Any,
    current_frontier_release_evidence_report: Any,
) -> dict[str, Any]:
    source_key = _artifact_path_key(source_path)
    current_key = _artifact_path_key(current_frontier_release_evidence_report)
    if source_key is None:
        if current_key is None:
            status = "unknown"
            is_current = None
            reason = "research queue has no source path and no current frontier evidence path is known"
        else:
            status = "unknown"
            is_current = None
            reason = "research queue has no source path to compare with current frontier evidence"
    elif current_key is None:
        status = "unknown"
        is_current = None
        reason = "current frontier evidence path is unavailable"
    elif source_key == current_key:
        status = "current"
        is_current = True
        reason = "research queue source matches the current frontier evidence report"
    else:
        status = "stale"
        is_current = False
        reason = "research queue source differs from the current frontier evidence report"
    return {
        "status": status,
        "is_current": is_current,
        "source_status": None if source_status is None else str(source_status),
        "research_source_path": None if source_path is None else str(source_path),
        "current_frontier_release_evidence_report": None
        if current_frontier_release_evidence_report is None
        else str(current_frontier_release_evidence_report),
        "reason": reason,
    }


def _research_queue_lifecycle_status(
    *,
    action_count: int,
    gap_count: int,
    source_status: Any,
    alignment_status: str,
    productized_blocking_reasons: Sequence[str],
) -> str:
    if action_count <= 0 and gap_count <= 0:
        return "empty"
    if productized_blocking_reasons:
        return "current_blocker" if alignment_status in {"current", "unknown"} else "stale"
    if str(source_status or "").lower() == "promote":
        return "closed"
    if alignment_status == "stale":
        return "superseded"
    if alignment_status == "unknown":
        return "superseded"
    return "active"


def _artifact_path_key(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    try:
        return path.resolve(strict=False).as_posix()
    except OSError:
        return path.as_posix()


def _action_summary(action: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action_id": action.get("action_id"),
        "title": action.get("title"),
        "action_type": action.get("action_type"),
        "priority": action.get("priority"),
        "evidence_routes": tuple(action.get("evidence_routes") or ()),
        "source_gap_ids": tuple(action.get("source_gap_ids") or ()),
        "suggested_commands": tuple(action.get("suggested_commands") or ()),
        "suggested_command_count": len(_sequence(action.get("suggested_commands"))),
        "metadata": dict(_mapping(action.get("metadata"))),
    }


def _gap_summary(gap: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _mapping(gap.get("metadata"))
    return {
        "gap_id": gap.get("gap_id"),
        "gate": gap.get("gate"),
        "status": gap.get("status"),
        "root_cause": gap.get("root_cause"),
        "evidence_kind": metadata.get("evidence_kind"),
        "research_axis": metadata.get("research_axis"),
        "missing_metric_count": len(_sequence(gap.get("missing_metrics"))),
        "recommended_action_ids": tuple(gap.get("recommended_action_ids") or ()),
    }


def _active_blockers(
    *,
    release_summary: Mapping[str, Any],
    contract_summary: Mapping[str, Any],
) -> tuple[str, ...]:
    blockers = []
    if release_summary.get("status") != "promote":
        blockers.append(f"release candidate status is {release_summary.get('status')!r}")
    if _int_or_zero(release_summary.get("blocked_gate_count")) > 0:
        blockers.append("release candidate has blocked gates")
    if contract_summary.get("status") != "promote":
        blockers.append(f"product contract status is {contract_summary.get('status')!r}")
    if _int_or_zero(contract_summary.get("blocked_evidence_group_count")) > 0:
        blockers.append("product contract has blocked required evidence groups")
    return tuple(blockers)


def _status_fields(payload: Mapping[str, Any]) -> dict[str, str | None]:
    fields = {
        key.removesuffix("_status"): value
        for key, value in payload.items()
        if key.endswith("_status")
    }
    return dict(sorted((key, value) for key, value in fields.items() if value is not None))


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path | None,
    release_path: Path | None,
    contract_path: Path | None,
    gap_path: Path | None,
    research_source_path: Path | None,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> Mapping[str, Any]:
    artifacts = {
        name: path
        for name, path in {
            "frontier_status_report": output_path,
            "release_candidate": release_path,
            "product_contract": contract_path,
            "evidence_gap_plan": gap_path,
            "research_queue_source": research_source_path,
        }.items()
        if path is not None
    }
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "build_frontier_status_report",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "release_candidate_status": _nested(
                payload, "productized_status", "release_candidate", "status"
            ),
            "product_contract_status": _nested(
                payload, "productized_status", "product_contract", "status"
            ),
            "research_action_count": _nested(payload, "research_queue", "action_count"),
            "research_active_action_count": _nested(
                payload, "research_queue", "active_action_count"
            ),
            "research_superseded_action_count": _nested(
                payload, "research_queue", "superseded_action_count"
            ),
            "research_lifecycle_status": _nested(
                payload, "research_queue", "lifecycle_status"
            ),
            "research_source_alignment_status": _nested(
                payload, "research_queue", "source_alignment", "status"
            ),
            "research_refresh_status": _nested(payload, "research_queue", "refresh_status"),
            **dict(metadata),
        },
    )
    _write_json(manifest_path, manifest, compact=compact)
    return manifest


def _load_mapping_source(source: str | Path | Mapping[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    if isinstance(source, Mapping):
        return None, dict(source)
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return path, dict(payload)


def _load_research_queue_source(
    *,
    research_queue_source: str | Path | Mapping[str, Any] | None,
    gap_payload: Mapping[str, Any],
    gap_path: Path | None,
    release_path: Path | None,
    release_payload: Mapping[str, Any],
) -> tuple[Path | None, dict[str, Any]]:
    if research_queue_source is not None:
        return _load_mapping_source(research_queue_source)
    source_hint = gap_payload.get("source_path")
    if source_hint is not None:
        source_path = _resolve_artifact_path(source_hint, relative_to=gap_path)
        return _load_mapping_source(source_path)
    if release_path is not None:
        return release_path, dict(release_payload)
    return None, dict(release_payload)


def _resolve_artifact_path(value: Any, *, relative_to: Path | None) -> Path:
    path = Path(str(value))
    candidates = [path]
    if not path.is_absolute():
        candidates = []
        if relative_to is not None:
            candidates.append(relative_to.parent / path)
        candidates.append(ROOT / path)
        candidates.append(Path.cwd() / path)
        candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"research queue source does not exist: {value}")


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return tuple(value)


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-candidate", default=str(DEFAULT_RELEASE_CANDIDATE))
    parser.add_argument("--product-contract", default=str(DEFAULT_PRODUCT_CONTRACT))
    parser.add_argument("--evidence-gap-plan", default=str(DEFAULT_EVIDENCE_GAP_PLAN))
    parser.add_argument("--no-evidence-gap-plan", action="store_true")
    parser.add_argument(
        "--refresh-research-queue",
        action="store_true",
        help="recompute the research_queue from the gap plan source path or active release candidate",
    )
    parser.add_argument(
        "--research-queue-source",
        default=None,
        help="optional release/frontier report to use when refreshing research_queue",
    )
    parser.add_argument("--json", default=None, help="optional output JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional local artifact registry JSON")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    evidence_gap_plan = None if args.no_evidence_gap_plan else args.evidence_gap_plan
    payload = build_frontier_status_report(
        release_candidate=args.release_candidate,
        product_contract=args.product_contract,
        evidence_gap_plan=evidence_gap_plan,
        research_queue_source=args.research_queue_source,
        refresh_research_queue=bool(args.refresh_research_queue),
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        compact_json=bool(args.compact_json),
    )
    if args.json is None:
        print(strict_json_dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
