"""Build a ProductTrace handoff from promoted world-model rule candidates.

This workflow starts after ``promote_world_model_rule_candidates.py``. It only
turns already-promoted deterministic rule candidates into product-control traces
and dry-run action results. It does not treat pending rule inputs as verifier
evidence and does not claim open-domain world-model coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.control import (  # noqa: E402
    ActionExecutorRegistry,
    ControlAction,
    DefaultCorrectionPolicy,
    DryRunActionExecutor,
    ProductTrace,
    RiskDecision,
    RiskLevel,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402
from eigentruth.verify import Claim, VerificationResult, VerificationStatus  # noqa: E402

WORKFLOW = "world_model_rule_candidate_handoff"
PROMOTION_WORKFLOW = "world_model_rule_candidate_promotion_gate"
DEFAULT_ROUTE_NAME = "world_model_rule_candidate"
DEFAULT_VERIFIER_NAME = "PromotedWorldModelRuleCandidate"
HANDOFF_STATUSES = {"supported", "refuted"}


def build_world_model_rule_candidate_handoff(
    promotion_gate: Mapping[str, Any],
    *,
    promoted_candidates: Sequence[Mapping[str, Any]] | None = None,
    route_name: str = DEFAULT_ROUTE_NAME,
    verifier_name: str = DEFAULT_VERIFIER_NAME,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return report, ProductTrace rows, and ActionResult rows for candidates."""
    candidates = tuple(
        dict(row)
        for row in (
            promoted_candidates
            if promoted_candidates is not None
            else _sequence_of_mappings(promotion_gate.get("promoted_candidates"))
        )
    )
    source_promoted = _source_promoted(promotion_gate)
    traces: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    if source_promoted:
        for candidate in candidates:
            failures = _candidate_failures(candidate)
            if failures:
                blocked.append(_blocked_candidate(candidate, failures=failures))
                continue
            traces.append(
                _trace_for_candidate(
                    candidate,
                    promotion_gate=promotion_gate,
                    route_name=route_name,
                    verifier_name=verifier_name,
                )
            )
    else:
        blocked.extend(
            _blocked_candidate(candidate, failures=("source_promotion_gate_not_promoted",))
            for candidate in candidates
        )
    action_results = tuple(
        result
        for trace in traces
        for result in _sequence_of_mappings(trace.get("action_results"))
    )
    report = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": _status(
            source_promoted=source_promoted,
            candidate_count=len(candidates),
            blocked_count=len(blocked),
            trace_count=len(traces),
        ),
        "scope": (
            "Target-specific product handoff for promoted deterministic "
            "world-model rule candidates. It only consumes rows that passed "
            "the promotion gate and preserves source-citation provenance."
        ),
        "source": {
            "promotion_gate_workflow": promotion_gate.get("workflow"),
            "promotion_gate_status": promotion_gate.get("status"),
            "promotion_gate_summary": promotion_gate.get("summary"),
            "source_promoted": source_promoted,
        },
        "label_usage": {
            "labels_used_for_handoff": False,
            "labels_copied_to_trace_metadata": False,
            "pending_rows_are_verifier_evidence": False,
            "requires_promoted_rule_candidate": True,
            "requires_source_citation": True,
        },
        "config": {
            "route_name": route_name,
            "verifier_name": verifier_name,
        },
        "summary": _summary(candidates=candidates, blocked=blocked, traces=traces),
        "trace_summaries": tuple(_trace_summary(trace) for trace in traces),
        "blocked_candidates": tuple(blocked),
        "metadata": dict(metadata or {}),
    }
    return {
        "report": report,
        "product_traces": tuple(traces),
        "action_results": action_results,
    }


def run(
    *,
    promotion_gate_path: str | Path,
    output_dir: str | Path,
    promoted_candidates_path: str | Path | None = None,
    report_json_path: str | Path | None = None,
    trace_jsonl_path: str | Path | None = None,
    action_results_jsonl_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    route_name: str = DEFAULT_ROUTE_NAME,
    verifier_name: str = DEFAULT_VERIFIER_NAME,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register the rule-candidate handoff."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "world-model-rule-candidate-handoff.json")
    trace_path = Path(trace_jsonl_path or output / "product-traces.jsonl")
    action_results_path = Path(action_results_jsonl_path or output / "action-results.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    promotion_gate = _load_json_object(promotion_gate_path)
    promoted_candidates = (
        _load_jsonl_mappings(promoted_candidates_path)
        if promoted_candidates_path is not None
        else None
    )
    payload = build_world_model_rule_candidate_handoff(
        promotion_gate,
        promoted_candidates=promoted_candidates,
        route_name=route_name,
        verifier_name=verifier_name,
        metadata=metadata,
    )
    report = dict(payload["report"])
    report["paths"] = {
        "promotion_gate": str(promotion_gate_path),
        "promoted_candidates": None if promoted_candidates_path is None else str(promoted_candidates_path),
        "product_traces": str(trace_path),
        "action_results": str(action_results_path),
        "artifact_manifest": str(manifest_path),
    }
    payload["report"] = report

    _write_json(report_path, report, compact=compact_json)
    _write_jsonl(trace_path, payload["product_traces"], compact=compact_json)
    _write_jsonl(action_results_path, payload["action_results"], compact=compact_json)

    artifacts: dict[str, str | Path | None] = {
        "world_model_rule_candidate_handoff": report_path,
        "product_traces": trace_path,
        "action_results": action_results_path,
        "promotion_gate": Path(promotion_gate_path),
    }
    if promoted_candidates_path is not None:
        artifacts["promoted_rule_candidates"] = Path(promoted_candidates_path)
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": report["status"],
            "trace_count": report["summary"]["trace_count"],
            "blocked_candidate_count": report["summary"]["blocked_candidate_count"],
            "action_result_count": report["summary"]["action_result_count"],
            **dict(metadata or {}),
        },
    )
    _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        registry = ArtifactRegistry.load_json(registry_path)
        registry.record_report(
            name=name,
            version=version,
            path=report_path,
            metadata={
                "workflow": WORKFLOW,
                "status": report["status"],
                "trace_count": report["summary"]["trace_count"],
                "blocked_candidate_count": report["summary"]["blocked_candidate_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        )
        registry.record_trace(
            name=name,
            version=version,
            path=trace_path,
            metadata={
                "workflow": WORKFLOW,
                "status": report["status"],
                "trace_count": report["summary"]["trace_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        )
        registry.record_action_result(
            name=name,
            version=version,
            path=action_results_path,
            metadata={
                "workflow": WORKFLOW,
                "status": report["status"],
                "action_result_count": report["summary"]["action_result_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        )
        registry.save_json()
    return payload


def _trace_for_candidate(
    candidate: Mapping[str, Any],
    *,
    promotion_gate: Mapping[str, Any],
    route_name: str,
    verifier_name: str,
) -> dict[str, Any]:
    request_id = str(candidate.get("request_id") or "")
    result = _verification_result(candidate, route_name=route_name, verifier_name=verifier_name)
    claim = _claim(candidate)
    decision = _risk_decision(result, route_name=route_name)
    actions = tuple(
        replace(action, request_id=f"{request_id}:handoff-action:{idx}")
        for idx, action in enumerate(
            DefaultCorrectionPolicy().plan(
                decision,
                claims=(claim,),
                verification_results=(result,),
                context={
                    "route_name": route_name,
                    "workflow": WORKFLOW,
                    "promotion_gate": promotion_gate.get("workflow"),
                },
            ),
            start=1,
        )
    )
    executor = ActionExecutorRegistry(
        fallback_executor=DryRunActionExecutor(executor_name=WORKFLOW)
    )
    action_results = executor.execute_many(
        actions,
        context={
            "request_id": request_id,
            "route_name": route_name,
            "workflow": WORKFLOW,
        },
    )
    trace = ProductTrace(
        request_id=f"{WORKFLOW}:{request_id}",
        diagnostics={
            "world_model_rule_candidate_handoff": 1.0,
            "candidate_confidence": _float(candidate.get("confidence")),
            "evidence_count": len(_sequence(candidate.get("evidence"))),
            "source_citation_present": bool(str(candidate.get("source_citation") or "")),
        },
        claims=(claim,),
        verification_results=(result,),
        risk_decision=decision,
        actions=actions,
        action_results=action_results,
        metadata={
            "workflow": WORKFLOW,
            "route_name": route_name,
            "verifier_name": verifier_name,
            "promotion_gate": promotion_gate.get("workflow"),
            "promotion_gate_status": promotion_gate.get("status"),
            "candidate_request_id": request_id,
            "target_id": candidate.get("target_id"),
            "rule_family": candidate.get("rule_family"),
            "candidate_status": candidate.get("status"),
            "adapter": candidate.get("adapter"),
            "source_citation": candidate.get("source_citation"),
            "source_url": candidate.get("source_url"),
            "candidate_only_requires_downstream_handoff": True,
            "not_open_domain_verifier": True,
        },
    )
    payload = trace.to_dict()
    payload["bounded"] = trace.to_bounded_dict(
        metadata_keys=(
            "workflow",
            "route_name",
            "verifier_name",
            "candidate_request_id",
            "target_id",
            "rule_family",
            "candidate_status",
            "source_citation",
        )
    )
    return payload


def _claim(candidate: Mapping[str, Any]) -> Claim:
    rule_input = _mapping(candidate.get("rule_input"))
    question = str(candidate.get("question") or "").strip()
    answer_entity = str(rule_input.get("answer_entity") or "").strip()
    mechanism = str(rule_input.get("mechanism") or "").strip()
    precondition = str(rule_input.get("precondition") or "").strip()
    mechanism_status = str(rule_input.get("mechanism_status") or "").strip()
    if mechanism:
        claim_text = _mechanism_claim_text(
            question=question,
            mechanism=mechanism,
            precondition=precondition,
            mechanism_status=mechanism_status,
        )
    else:
        claim_text = f"{question} {answer_entity}".strip() or str(candidate.get("request_id") or "")
    return Claim(
        text=claim_text,
        claim_id=f"{candidate.get('request_id')}:rule-candidate-claim",
        metadata={
            "question": question,
            "answer_entity": answer_entity,
            "expected_entity": rule_input.get("expected_entity"),
            "subject_entity": rule_input.get("subject_entity"),
            "requested_role": rule_input.get("requested_role"),
            "mechanism": mechanism,
            "precondition": precondition,
            "mechanism_status": mechanism_status,
            "rule_family": candidate.get("rule_family"),
            "source_family": rule_input.get("source_family"),
            "provider": rule_input.get("provider"),
            "source_citation": candidate.get("source_citation"),
            "source_url": candidate.get("source_url"),
            "requires_verification": True,
            "world_model_rule_candidate_handoff": True,
        },
    )


def _mechanism_claim_text(
    *,
    question: str,
    mechanism: str,
    precondition: str,
    mechanism_status: str,
) -> str:
    parts = []
    if question:
        parts.append(question)
    parts.append(f"Mechanism: {mechanism}")
    if precondition:
        parts.append(f"Precondition: {precondition}")
    if mechanism_status:
        parts.append(f"Mechanism status: {mechanism_status}")
    return " ".join(parts)


def _verification_result(
    candidate: Mapping[str, Any],
    *,
    route_name: str,
    verifier_name: str,
) -> VerificationResult:
    status = VerificationStatus(str(candidate.get("status") or ""))
    confidence = _bounded_confidence(candidate.get("confidence"))
    return VerificationResult(
        status=status,
        confidence=confidence,
        evidence=tuple(str(item) for item in _sequence(candidate.get("evidence"))),
        explanation=f"promoted deterministic rule candidate returned {status.value}",
        metadata={
            "selected_route": route_name,
            "selected_verifier": verifier_name,
            "matched_routes": _unique_strings(("world_model_rule_candidate", route_name)),
            "selected_route_duration_seconds": 0.0,
            "attempted_routes": (route_name,),
            "route_family": "world_model_rule_candidate_handoff",
            "candidate_request_id": candidate.get("request_id"),
            "target_id": candidate.get("target_id"),
            "rule_family": candidate.get("rule_family"),
            "adapter": candidate.get("adapter"),
            "source_citation": candidate.get("source_citation"),
            "source_url": candidate.get("source_url"),
            "not_open_domain_verifier": True,
        },
    )


def _risk_decision(result: VerificationResult, *, route_name: str) -> RiskDecision:
    if result.status is VerificationStatus.REFUTED:
        return RiskDecision(
            action=ControlAction.ABSTAIN,
            risk_level=RiskLevel.HIGH,
            confidence=max(result.confidence, 0.9),
            reason="promoted world-model rule candidate refuted the claim",
            diagnostics={
                "verification": {
                    "total": 1,
                    "counts": {VerificationStatus.REFUTED.value: 1},
                    "route": route_name,
                }
            },
        )
    if result.status is VerificationStatus.SUPPORTED:
        return RiskDecision(
            action=ControlAction.ACCEPT,
            risk_level=RiskLevel.LOW,
            confidence=result.confidence,
            reason="promoted world-model rule candidate supported the claim",
            diagnostics={
                "verification": {
                    "total": 1,
                    "counts": {VerificationStatus.SUPPORTED.value: 1},
                    "route": route_name,
                }
            },
        )
    return RiskDecision(
        action=ControlAction.CLARIFY,
        risk_level=RiskLevel.UNKNOWN,
        confidence=result.confidence,
        reason=f"promoted world-model rule candidate returned {result.status.value}",
        diagnostics={
            "verification": {
                "total": 1,
                "counts": {result.status.value: 1},
                "route": route_name,
            }
        },
    )


def _candidate_failures(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    failures: list[str] = []
    status = str(candidate.get("status") or "")
    if candidate.get("workflow") != PROMOTION_WORKFLOW:
        failures.append("candidate_not_from_promotion_gate")
    if status not in HANDOFF_STATUSES:
        failures.append("status_not_handoffable")
    if not str(candidate.get("request_id") or ""):
        failures.append("missing_request_id")
    if not str(candidate.get("rule_family") or ""):
        failures.append("missing_rule_family")
    if _mapping(candidate.get("promotion")).get("status") != "promote":
        failures.append("promotion_status_not_promote")
    if _mapping(candidate.get("promotion")).get("candidate_only_requires_downstream_handoff") is not True:
        failures.append("missing_downstream_handoff_marker")
    source_citation = str(candidate.get("source_citation") or "")
    if not source_citation:
        failures.append("missing_source_citation")
    elif source_citation not in " ".join(str(item) for item in _sequence(candidate.get("evidence"))):
        failures.append("source_citation_not_in_evidence")
    if _float(candidate.get("confidence")) is None:
        failures.append("missing_confidence")
    if not _sequence(candidate.get("evidence")):
        failures.append("missing_evidence")
    return tuple(failures)


def _blocked_candidate(candidate: Mapping[str, Any], *, failures: Sequence[str]) -> dict[str, Any]:
    return {
        "request_id": str(candidate.get("request_id") or ""),
        "target_id": str(candidate.get("target_id") or ""),
        "rule_family": str(candidate.get("rule_family") or ""),
        "status": str(candidate.get("status") or ""),
        "failures": tuple(str(item) for item in failures),
    }


def _status(
    *,
    source_promoted: bool,
    candidate_count: int,
    blocked_count: int,
    trace_count: int,
) -> str:
    if not source_promoted:
        return "blocked"
    if blocked_count:
        return "blocked"
    if trace_count:
        return "promote"
    return "empty" if candidate_count == 0 else "blocked"


def _summary(
    *,
    candidates: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    verification_status_counts = Counter(_trace_status(trace) for trace in traces)
    action_counts = Counter(
        str(action.get("action") or "unknown")
        for trace in traces
        for action in _sequence_of_mappings(trace.get("actions"))
    )
    rule_family_counts = Counter(str(candidate.get("rule_family") or "") for candidate in candidates)
    trace_action_summaries = tuple(
        _mapping(_mapping(trace.get("bounded")).get("summaries")).get("action_execution")
        for trace in traces
    )
    return {
        "input_candidate_count": len(candidates),
        "handoff_candidate_count": len(candidates) - len(blocked),
        "blocked_candidate_count": len(blocked),
        "trace_count": len(traces),
        "action_result_count": sum(
            1
            for trace in traces
            for _result in _sequence_of_mappings(trace.get("action_results"))
        ),
        "verification_status_counts": _sorted_counter(verification_status_counts),
        "action_counts": _sorted_counter(action_counts),
        "rule_family_counts": _sorted_counter(rule_family_counts),
        "source_citation_count": sum(1 for candidate in candidates if candidate.get("source_citation")),
        "action_execution_alignment_passed": all(
            summary.get("alignment_passed") is True
            for summary in trace_action_summaries
            if isinstance(summary, Mapping)
        ),
    }


def _trace_summary(trace: Mapping[str, Any]) -> dict[str, Any]:
    results = _sequence_of_mappings(trace.get("verification_results"))
    actions = _sequence_of_mappings(trace.get("actions"))
    result = results[0] if results else {}
    metadata = _mapping(result.get("metadata"))
    return {
        "request_id": trace.get("request_id"),
        "status": result.get("status"),
        "selected_route": metadata.get("selected_route"),
        "risk_action": _mapping(trace.get("risk_decision")).get("action"),
        "risk_level": _mapping(trace.get("risk_decision")).get("risk_level"),
        "action_count": len(actions),
        "evidence_count": len(_sequence(result.get("evidence"))),
        "source_citation": metadata.get("source_citation"),
    }


def _trace_status(trace: Mapping[str, Any]) -> str:
    results = _sequence_of_mappings(trace.get("verification_results"))
    if not results:
        return "missing"
    return str(results[0].get("status") or "unknown")


def _source_promoted(promotion_gate: Mapping[str, Any]) -> bool:
    return (
        promotion_gate.get("workflow") == PROMOTION_WORKFLOW
        and promotion_gate.get("status") == "promote"
    )


def _load_json_object(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _load_jsonl_mappings(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append(dict(row))
    return tuple(rows)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = "".join(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    else:
        text = "".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(value)
    return ()


def _sequence_of_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    return tuple(dict(item) for item in _sequence(value) if isinstance(item, Mapping))


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _bounded_confidence(value: Any) -> float:
    result = _float(value)
    if result is None:
        return 0.0
    return min(max(result, 0.0), 1.0)


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(((key, value) for key, value in counter.items() if key), key=lambda item: (-item[1], item[0])))


def _unique_strings(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        item = str(value)
        if item and item not in seen:
            unique.append(item)
            seen.add(item)
    return tuple(unique)


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
    parser.add_argument("--promotion-gate", required=True)
    parser.add_argument("--promoted-candidates", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--trace-jsonl", default=None)
    parser.add_argument("--action-results-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--route-name", default=DEFAULT_ROUTE_NAME)
    parser.add_argument("--verifier-name", default=DEFAULT_VERIFIER_NAME)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        promotion_gate_path=args.promotion_gate,
        promoted_candidates_path=args.promoted_candidates,
        output_dir=args.output_dir,
        report_json_path=args.json,
        trace_jsonl_path=args.trace_jsonl,
        action_results_jsonl_path=args.action_results_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        route_name=args.route_name,
        verifier_name=args.verifier_name,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["report"]["summary"]
    print(
        "world_model_rule_candidate_handoff_ok "
        f"status={payload['report']['status']} "
        f"traces={summary['trace_count']} "
        f"blocked={summary['blocked_candidate_count']} "
        f"actions={summary['action_result_count']}"
    )


if __name__ == "__main__":
    main()
