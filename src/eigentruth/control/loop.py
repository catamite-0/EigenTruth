"""Closed-loop verification helpers for control-plane workflows."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from eigentruth.control.actions import (
    ActionExecutionStatus,
    ActionExecutorRegistry,
    ActionRequest,
    ActionResult,
    CorrectionPolicy,
    DefaultCorrectionPolicy,
)
from eigentruth.control.controller import RiskController
from eigentruth.control.policy import RiskDecision
from eigentruth.control.staging import StagedVerificationPolicy, VerificationStageDecision
from eigentruth.control.trace import ProductTrace, RuntimePhaseTiming, RuntimeTrace, TraceEvent
from eigentruth.verify.protocols import Claim, VerificationResult, Verifier


@dataclass(frozen=True)
class EvidenceBundle:
    """Evidence collected from action execution, grouped by claim when possible."""

    evidence: Sequence[Mapping[str, Any]] = ()
    by_claim_id: Mapping[str, Sequence[Mapping[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(dict(item) for item in self.evidence))
        grouped = {
            str(claim_id): tuple(dict(item) for item in items)
            for claim_id, items in self.by_claim_id.items()
        }
        object.__setattr__(self, "by_claim_id", grouped)

    def to_context(self, claim_id: str | None = None) -> dict[str, Any]:
        """Return verifier context containing claim-specific evidence when available."""
        if claim_id is not None:
            if claim_id in self.by_claim_id:
                return {"evidence": self.by_claim_id[claim_id]}
            if self.by_claim_id:
                return {"evidence": ()}
        return {"evidence": self.evidence}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "evidence": tuple(_jsonable(item) for item in self.evidence),
            "by_claim_id": {
                claim_id: tuple(_jsonable(item) for item in items)
                for claim_id, items in self.by_claim_id.items()
            },
            "total_evidence": len(self.evidence),
            "claim_ids": tuple(self.by_claim_id),
        }

    def has_evidence(self) -> bool:
        """Return whether the bundle contains any evidence snippet."""
        return bool(self.evidence or any(self.by_claim_id.values()))


@dataclass(frozen=True)
class VerificationLoopResult:
    """Result of one verify-plan-execute-reverify control loop."""

    initial_verification_results: Sequence[VerificationResult | Mapping[str, Any]]
    initial_decision: RiskDecision
    action_requests: Sequence[ActionRequest]
    action_results: Sequence[ActionResult]
    retrieval_evidence: EvidenceBundle
    final_verification_results: Sequence[VerificationResult | Mapping[str, Any]]
    final_decision: RiskDecision
    trace: ProductTrace
    verification_stage_decision: VerificationStageDecision | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_verification_results", tuple(self.initial_verification_results))
        object.__setattr__(self, "action_requests", tuple(self.action_requests))
        object.__setattr__(self, "action_results", tuple(self.action_results))
        object.__setattr__(self, "final_verification_results", tuple(self.final_verification_results))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "initial_verification_results": tuple(
                _verification_result_to_dict(result) for result in self.initial_verification_results
            ),
            "initial_decision": self.initial_decision.to_dict(),
            "action_requests": tuple(request.to_dict() for request in self.action_requests),
            "action_results": tuple(result.to_dict() for result in self.action_results),
            "retrieval_evidence": self.retrieval_evidence.to_dict(),
            "final_verification_results": tuple(
                _verification_result_to_dict(result) for result in self.final_verification_results
            ),
            "final_decision": self.final_decision.to_dict(),
            "trace": self.trace.to_dict(),
            "verification_stage_decision": _stage_decision_to_dict(self.verification_stage_decision),
        }


def evidence_bundle_from_action_results(
    action_results: Sequence[ActionResult | Mapping[str, Any]],
) -> EvidenceBundle:
    """Collect verifier-ready evidence from retrieval action results."""
    evidence: list[dict[str, Any]] = []
    by_claim_id: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str | None, str, str | None]] = set()

    for result in action_results:
        result_payload = _action_result_to_dict(result)
        if result_payload.get("action") != "retrieve":
            continue
        if result_payload.get("status") not in {ActionExecutionStatus.SUCCEEDED.value, "succeeded"}:
            continue
        output = result_payload.get("output", {})
        if not isinstance(output, Mapping):
            continue

        hits_before_result = len(evidence)
        hits_by_query = output.get("hits_by_query", ())
        if isinstance(hits_by_query, Sequence) and not isinstance(hits_by_query, (str, bytes)):
            for item in hits_by_query:
                if not isinstance(item, Mapping):
                    continue
                query = item.get("query", {})
                claim_id = _claim_id_from_query(query)
                hits = item.get("hits", ())
                if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes)):
                    continue
                for hit in hits:
                    evidence_item = _evidence_from_hit(hit, claim_id=claim_id, query=query)
                    if evidence_item is None:
                        continue
                    _append_evidence(evidence_item, evidence, by_claim_id, seen, claim_id=claim_id)

        if len(evidence) > hits_before_result:
            continue
        hits = output.get("hits", ())
        if isinstance(hits, Sequence) and not isinstance(hits, (str, bytes)):
            for hit in hits:
                evidence_item = _evidence_from_hit(hit, claim_id=None, query=None)
                if evidence_item is not None:
                    _append_evidence(evidence_item, evidence, by_claim_id, seen, claim_id=None)

    return EvidenceBundle(
        evidence=tuple(evidence),
        by_claim_id={key: tuple(value) for key, value in by_claim_id.items()},
    )


def run_verification_loop(
    *,
    request_id: str | None = None,
    diagnostics: Mapping[str, float],
    claims: Sequence[Claim],
    verifier: Verifier,
    controller: RiskController,
    correction_policy: CorrectionPolicy | None = None,
    executor_registry: ActionExecutorRegistry | None = None,
    context: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    stage_policy: StagedVerificationPolicy | None = None,
    profile_runtime: bool = True,
) -> VerificationLoopResult:
    """Run a dependency-free verification/action/reverification loop."""
    base_context = dict(context or {})
    runtime_phases: list[RuntimePhaseTiming] = []
    loop_started_at = _start_runtime_phase(profile_runtime)
    diagnostic_decision: RiskDecision | None = None
    stage_decision: VerificationStageDecision | None = None
    initial_verification_claims: tuple[Claim, ...] = tuple(claims)
    initial_verified_claim_ids: tuple[str, ...] = _claim_ids(claims)
    initial_skipped_claim_ids: tuple[str, ...] = ()
    initial_verification_scope = "all"
    if stage_policy is None:
        phase_started_at = _start_runtime_phase(profile_runtime)
        initial_results = tuple(verifier.verify_many(initial_verification_claims, context=base_context))
        _record_runtime_phase(
            runtime_phases,
            "initial_verification",
            phase_started_at,
            metadata={
                "n_claims": len(initial_verification_claims),
                "total_claims": len(claims),
                "skipped": False,
                "verification_scope": initial_verification_scope,
            },
        )
        phase_started_at = _start_runtime_phase(profile_runtime)
        initial_decision = controller.decide(diagnostics, verification_results=initial_results)
        _record_runtime_phase(
            runtime_phases,
            "initial_risk_decision",
            phase_started_at,
            metadata={"verification_results": len(initial_results)},
        )
    else:
        phase_started_at = _start_runtime_phase(profile_runtime)
        diagnostic_decision = controller.decide(diagnostics)
        _record_runtime_phase(
            runtime_phases,
            "diagnostic_risk_decision",
            phase_started_at,
            metadata={"verification_results": 0},
        )
        phase_started_at = _start_runtime_phase(profile_runtime)
        stage_decision = stage_policy.decide(
            diagnostic_decision,
            claims=claims,
            context=base_context,
        )
        _record_runtime_phase(
            runtime_phases,
            "verification_stage_decision",
            phase_started_at,
            metadata={
                "run_verifier": stage_decision.run_verifier,
                "verification_scope": stage_decision.verification_scope,
            },
        )
        if stage_decision.run_verifier:
            initial_verification_claims = _select_stage_claims(claims, stage_decision)
            initial_verified_claim_ids = _claim_ids(initial_verification_claims)
            initial_skipped_claim_ids = tuple(stage_decision.skipped_claim_ids)
            initial_verification_scope = str(stage_decision.verification_scope)
            phase_started_at = _start_runtime_phase(profile_runtime)
            initial_results = tuple(verifier.verify_many(initial_verification_claims, context=base_context))
            _record_runtime_phase(
                runtime_phases,
                "initial_verification",
                phase_started_at,
                metadata={
                    "n_claims": len(initial_verification_claims),
                    "total_claims": len(claims),
                    "skipped": False,
                    "verification_scope": initial_verification_scope,
                    "skipped_claim_count": len(initial_skipped_claim_ids),
                },
            )
            phase_started_at = _start_runtime_phase(profile_runtime)
            initial_decision = controller.decide(diagnostics, verification_results=initial_results)
            _record_runtime_phase(
                runtime_phases,
                "initial_risk_decision",
                phase_started_at,
                metadata={"verification_results": len(initial_results)},
            )
        else:
            initial_verification_claims = ()
            initial_verified_claim_ids = ()
            initial_skipped_claim_ids = tuple(stage_decision.skipped_claim_ids) or _claim_ids(claims)
            initial_verification_scope = "none"
            initial_results = ()
            initial_decision = diagnostic_decision

    phase_started_at = _start_runtime_phase(profile_runtime)
    policy = correction_policy or DefaultCorrectionPolicy()
    action_requests = policy.plan(
        initial_decision,
        claims=initial_verification_claims,
        verification_results=initial_results,
        context=base_context,
    )
    _record_runtime_phase(
        runtime_phases,
        "action_planning",
        phase_started_at,
        metadata={"n_actions": len(action_requests)},
    )
    registry = executor_registry or ActionExecutorRegistry()
    execution_context = {**base_context, "request_id": request_id}
    phase_started_at = _start_runtime_phase(profile_runtime)
    action_results = registry.execute_many(action_requests, context=execution_context)
    _record_runtime_phase(
        runtime_phases,
        "action_execution",
        phase_started_at,
        metadata={"n_actions": len(action_requests), "n_results": len(action_results)},
    )
    phase_started_at = _start_runtime_phase(profile_runtime)
    retrieval_evidence = evidence_bundle_from_action_results(action_results)
    _record_runtime_phase(
        runtime_phases,
        "retrieval_evidence_collection",
        phase_started_at,
        metadata={"n_results": len(action_results), "has_evidence": retrieval_evidence.has_evidence()},
    )

    if retrieval_evidence.has_evidence():
        final_verification_claims = initial_verification_claims or tuple(claims)
        phase_started_at = _start_runtime_phase(profile_runtime)
        final_results = _verify_with_retrieved_evidence(
            verifier,
            final_verification_claims,
            base_context=base_context,
            evidence_bundle=retrieval_evidence,
        )
        _record_runtime_phase(
            runtime_phases,
            "final_verification",
            phase_started_at,
            metadata={
                "n_claims": len(final_verification_claims),
                "total_claims": len(claims),
                "used_retrieval_evidence": True,
                "verification_scope": initial_verification_scope,
            },
        )
        phase_started_at = _start_runtime_phase(profile_runtime)
        final_decision = controller.decide(diagnostics, verification_results=final_results)
        _record_runtime_phase(
            runtime_phases,
            "final_risk_decision",
            phase_started_at,
            metadata={"verification_results": len(final_results)},
        )
    else:
        final_results = initial_results
        final_decision = initial_decision

    runtime_trace = _build_runtime_trace(runtime_phases, loop_started_at)
    trace = ProductTrace(
        request_id=request_id,
        diagnostics=diagnostics,
        claims=claims,
        verification_results=final_results,
        risk_decision=final_decision,
        actions=action_requests,
        action_results=action_results,
        events=(
            *(
                (
                    TraceEvent("diagnostic_risk_decision", diagnostic_decision.to_dict()),
                    TraceEvent("verification_stage_decision", stage_decision.to_dict()),
                )
                if stage_decision is not None and diagnostic_decision is not None
                else ()
            ),
            *(
                (
                    TraceEvent(
                        "initial_verification_skipped",
                        {
                            "reason": stage_decision.reason,
                            "skipped_claim_ids": initial_skipped_claim_ids,
                        },
                    ),
                )
                if stage_decision is not None and not stage_decision.run_verifier
                else ()
            ),
            TraceEvent(
                "initial_verification",
                {
                    "n_claims": len(claims),
                    "verification_result_count": len(initial_results),
                    "verified_claim_ids": initial_verified_claim_ids,
                    "skipped_claim_ids": initial_skipped_claim_ids,
                    "verification_scope": initial_verification_scope,
                    "skipped": stage_decision is not None and not stage_decision.run_verifier,
                    "results": tuple(_verification_result_to_dict(result) for result in initial_results),
                },
            ),
            TraceEvent("initial_risk_decision", initial_decision.to_dict()),
            TraceEvent("actions_planned", {"n_actions": len(action_requests)}),
            TraceEvent(
                "actions_executed",
                {
                    "n_results": len(action_results),
                    "summary": ProductTrace(action_results=action_results).action_execution_summary(),
                },
            ),
            TraceEvent("retrieval_evidence_collected", retrieval_evidence.to_dict()),
            TraceEvent(
                "final_verification",
                {"n_claims": len(claims), "used_retrieval_evidence": retrieval_evidence.has_evidence()},
            ),
            TraceEvent("final_risk_decision", final_decision.to_dict()),
        ),
        metadata={
            "loop_version": "0.4",
            "source": "eigentruth.control.run_verification_loop",
            "staged_verification": None if stage_policy is None else stage_policy.to_dict(),
            **dict(metadata or {}),
        },
        runtime_trace=runtime_trace,
    )
    return VerificationLoopResult(
        initial_verification_results=initial_results,
        initial_decision=initial_decision,
        action_requests=action_requests,
        action_results=action_results,
        retrieval_evidence=retrieval_evidence,
        final_verification_results=final_results,
        final_decision=final_decision,
        trace=trace,
        verification_stage_decision=stage_decision,
    )


def _start_runtime_phase(enabled: bool) -> float | None:
    return time.perf_counter() if enabled else None


def _record_runtime_phase(
    phases: list[RuntimePhaseTiming],
    name: str,
    started_at: float | None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    if started_at is None:
        return
    phases.append(
        RuntimePhaseTiming(
            name=name,
            seconds=time.perf_counter() - started_at,
            metadata={} if metadata is None else metadata,
        )
    )


def _build_runtime_trace(
    phases: Sequence[RuntimePhaseTiming],
    started_at: float | None,
) -> RuntimeTrace | None:
    if started_at is None:
        return None
    return RuntimeTrace(
        phases=tuple(phases),
        total_seconds=time.perf_counter() - started_at,
    )


def _verify_with_retrieved_evidence(
    verifier: Verifier,
    claims: Sequence[Claim],
    *,
    base_context: Mapping[str, Any],
    evidence_bundle: EvidenceBundle,
) -> tuple[VerificationResult, ...]:
    results = []
    for index, claim in enumerate(claims):
        claim_id = claim.claim_id or f"c{index + 1}"
        claim_context = _context_with_retrieved_evidence(
            base_context,
            evidence_bundle.to_context(claim_id),
        )
        results.append(verifier.verify(claim, context=claim_context))
    return tuple(results)


def _select_stage_claims(
    claims: Sequence[Claim],
    stage_decision: VerificationStageDecision,
) -> tuple[Claim, ...]:
    """Return the claim subset selected by a staged-verification decision."""
    if stage_decision.verification_scope == "all":
        return tuple(claims)
    selected_ids = set(stage_decision.verify_claim_ids)
    return tuple(
        _claim_with_id(claim, _claim_id_for_index(claim, index))
        for index, claim in enumerate(claims)
        if _claim_id_for_index(claim, index) in selected_ids
    )


def _claim_ids(claims: Sequence[Claim]) -> tuple[str, ...]:
    return tuple(_claim_id_for_index(claim, index) for index, claim in enumerate(claims))


def _claim_id_for_index(claim: Claim, index: int) -> str:
    return claim.claim_id or f"c{index + 1}"


def _claim_with_id(claim: Claim, claim_id: str) -> Claim:
    if claim.claim_id == claim_id:
        return claim
    return Claim(
        text=claim.text,
        claim_id=claim_id,
        span=claim.span,
        metadata=claim.metadata,
    )


def _action_result_to_dict(result: ActionResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, ActionResult):
        return result.to_dict()
    return dict(_jsonable(result))


def _stage_decision_to_dict(
    decision: VerificationStageDecision | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if decision is None:
        return None
    if isinstance(decision, VerificationStageDecision):
        return decision.to_dict()
    return dict(_jsonable(decision))


def _context_with_retrieved_evidence(
    base_context: Mapping[str, Any],
    retrieval_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Return context with retrieval evidence appended, preserving existing evidence."""
    context = dict(base_context)
    if "evidence" not in retrieval_context:
        return context
    retrieved_evidence = _evidence_sequence(retrieval_context.get("evidence", ()))
    if not retrieved_evidence:
        return context
    context["evidence"] = _evidence_sequence(base_context.get("evidence", ())) + retrieved_evidence
    return context


def _evidence_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(value)
    return (value,)


def _verification_result_to_dict(result: VerificationResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, VerificationResult):
        return {
            "status": result.status.value,
            "confidence": result.confidence,
            "evidence": tuple(result.evidence),
            "explanation": result.explanation,
            "metadata": _jsonable(result.metadata),
        }
    return dict(_jsonable(result))


def _claim_id_from_query(query: Any) -> str | None:
    if not isinstance(query, Mapping):
        return None
    claim_id = query.get("claim_id")
    if claim_id is None:
        metadata = query.get("metadata", {})
        target = metadata.get("target", {}) if isinstance(metadata, Mapping) else {}
        claim_id = target.get("claim_id") if isinstance(target, Mapping) else None
    return None if claim_id is None else str(claim_id)


def _evidence_from_hit(
    hit: Any,
    *,
    claim_id: str | None,
    query: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if isinstance(hit, str):
        hit_payload: Mapping[str, Any] = {"text": hit}
    elif isinstance(hit, Mapping):
        hit_payload = hit
    else:
        return None

    raw_text = hit_payload.get("text", hit_payload.get("content"))
    if raw_text is None or not str(raw_text).strip():
        return None
    source = hit_payload.get("source")
    raw_metadata = hit_payload.get("metadata", {})
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    if "score" in hit_payload:
        metadata["retrieval_score"] = hit_payload["score"]
    if claim_id is not None:
        metadata["claim_id"] = claim_id
    if query is not None:
        metadata["retrieval_query"] = _jsonable(query)
    return {
        "text": str(raw_text),
        "source": None if source is None else str(source),
        "metadata": metadata,
    }


def _append_evidence(
    evidence_item: dict[str, Any],
    evidence: list[dict[str, Any]],
    by_claim_id: dict[str, list[dict[str, Any]]],
    seen: set[tuple[str | None, str, str | None]],
    *,
    claim_id: str | None,
) -> None:
    key = (claim_id, str(evidence_item["text"]), evidence_item.get("source"))
    if key in seen:
        return
    seen.add(key)
    evidence.append(evidence_item)
    if claim_id is not None:
        by_claim_id.setdefault(claim_id, []).append(evidence_item)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_jsonable(item) for item in value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if is_dataclass(value) and hasattr(value, "to_dict"):
        return value.to_dict()
    return value
