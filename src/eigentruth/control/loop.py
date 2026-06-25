"""Closed-loop verification helpers for control-plane workflows."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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
from eigentruth.control.policy import ControlAction, RiskDecision, RiskLevel
from eigentruth.control.staging import StagedVerificationPolicy, VerificationStageDecision
from eigentruth.control.trace import ProductTrace, RuntimePhaseTiming, RuntimeTrace, TraceEvent
from eigentruth.json_utils import to_jsonable
from eigentruth.verify.coherence import (
    ClaimCoherenceReport,
    ClaimDependency,
    apply_claim_coherence,
)
from eigentruth.verify.planning import (
    DEFAULT_VERIFY_CLAIM_FEATURE_FLAGS,
    DEFAULT_VERIFY_CLAIM_METADATA_KEYS,
    ClaimVerificationPlan,
    ClaimVerificationPlanner,
)
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus, Verifier


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
    claim_verification_plan: ClaimVerificationPlan | Mapping[str, Any] | None = None
    verification_stage_decision: VerificationStageDecision | Mapping[str, Any] | None = None
    initial_coherence_report: ClaimCoherenceReport | Mapping[str, Any] | None = None
    final_coherence_report: ClaimCoherenceReport | Mapping[str, Any] | None = None

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
            "claim_verification_plan": _claim_verification_plan_to_dict(self.claim_verification_plan),
            "verification_stage_decision": _stage_decision_to_dict(self.verification_stage_decision),
            "initial_coherence_report": _coherence_report_to_dict(self.initial_coherence_report),
            "final_coherence_report": _coherence_report_to_dict(self.final_coherence_report),
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
    claim_dependencies: Sequence[ClaimDependency | Mapping[str, Any]] | None = None,
    enforce_claim_coherence: bool = False,
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
    initial_coherence_report: ClaimCoherenceReport | None = None
    final_coherence_report: ClaimCoherenceReport | None = None
    coherence_enabled = bool(enforce_claim_coherence or claim_dependencies is not None)
    if stage_policy is None:
        phase_started_at = _start_runtime_phase(profile_runtime)
        initial_results = _verify_many_fail_closed(
            verifier,
            initial_verification_claims,
            context=base_context,
            phase="initial_verification",
        )
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
        initial_results, initial_coherence_report = _apply_claim_coherence_if_enabled(
            enabled=coherence_enabled,
            claims=initial_verification_claims,
            verification_results=initial_results,
            dependency_claims=claims,
            dependencies=claim_dependencies,
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
            initial_results = _verify_many_fail_closed(
                verifier,
                initial_verification_claims,
                context=base_context,
                phase="initial_verification",
            )
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
            initial_results, initial_coherence_report = _apply_claim_coherence_if_enabled(
                enabled=coherence_enabled,
                claims=initial_verification_claims,
                verification_results=initial_results,
                dependency_claims=claims,
                dependencies=claim_dependencies,
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
            initial_decision = (
                _fail_closed_unverified_skip_decision(
                    diagnostic_decision,
                    stage_decision=stage_decision,
                    skipped_claim_ids=initial_skipped_claim_ids,
                )
                if stage_policy.fail_closed_on_skip and claims
                else diagnostic_decision
            )

    action_planning_claims, action_planning_results, missing_dependency_claim_ids = (
        _extend_scope_with_missing_parent_claims(
            verified_claims=initial_verification_claims,
            verification_results=initial_results,
            all_claims=claims,
            coherence_report=initial_coherence_report,
        )
    )

    phase_started_at = _start_runtime_phase(profile_runtime)
    policy = correction_policy or DefaultCorrectionPolicy()
    action_requests = policy.plan(
        initial_decision,
        claims=action_planning_claims,
        verification_results=action_planning_results,
        context=base_context,
    )
    _record_runtime_phase(
        runtime_phases,
        "action_planning",
        phase_started_at,
        metadata={
            "n_actions": len(action_requests),
            "missing_dependency_claim_count": len(missing_dependency_claim_ids),
        },
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
        final_verification_claims = action_planning_claims or initial_verification_claims or tuple(claims)
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
        final_results, final_coherence_report = _apply_claim_coherence_if_enabled(
            enabled=coherence_enabled,
            claims=final_verification_claims,
            verification_results=final_results,
            dependency_claims=claims,
            dependencies=claim_dependencies,
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
        final_coherence_report = initial_coherence_report

    claim_verification_plan = _build_claim_verification_plan(
        claims=claims,
        stage_policy=stage_policy,
        stage_decision=stage_decision,
        claim_dependencies=claim_dependencies,
    )
    runtime_trace = _build_runtime_trace(runtime_phases, loop_started_at)
    trace = ProductTrace(
        request_id=request_id,
        diagnostics=diagnostics,
        claims=claims,
        verification_plan=claim_verification_plan,
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
            *(
                (TraceEvent("initial_claim_coherence", initial_coherence_report.to_dict()),)
                if initial_coherence_report is not None
                else ()
            ),
            TraceEvent("initial_risk_decision", initial_decision.to_dict()),
            TraceEvent(
                "claim_verification_plan",
                {
                    "run_verifier": claim_verification_plan.run_verifier,
                    "verification_scope": claim_verification_plan.verification_scope,
                    "verify_claim_ids": claim_verification_plan.verify_claim_ids,
                    "skipped_claim_ids": claim_verification_plan.skipped_claim_ids,
                    "triggered_claim_ids": claim_verification_plan.triggered_claim_ids,
                    "route_count": len(claim_verification_plan.route_hints),
                    "retrieval_query_count": len(claim_verification_plan.retrieval_queries),
                    "calculation_check_count": len(claim_verification_plan.calculation_checks),
                    "state_check_count": len(claim_verification_plan.state_checks),
                    "world_model_check_count": len(claim_verification_plan.world_model_checks),
                    "dependency_count": len(claim_verification_plan.dependencies),
                },
            ),
            TraceEvent(
                "actions_planned",
                {
                    "n_actions": len(action_requests),
                    "missing_dependency_claim_ids": missing_dependency_claim_ids,
                },
            ),
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
                {
                    "n_claims": len(claims),
                    "used_retrieval_evidence": retrieval_evidence.has_evidence(),
                },
            ),
            *(
                (TraceEvent("final_claim_coherence", final_coherence_report.to_dict()),)
                if final_coherence_report is not None
                and final_coherence_report is not initial_coherence_report
                else ()
            ),
            TraceEvent("final_risk_decision", final_decision.to_dict()),
        ),
        metadata={
            **dict(metadata or {}),
            "loop_version": "0.4",
            "source": "eigentruth.control.run_verification_loop",
            "claim_verification_plan": {
                "run_verifier": claim_verification_plan.run_verifier,
                "verification_scope": claim_verification_plan.verification_scope,
                "verify_claim_count": len(claim_verification_plan.verify_claim_ids),
                "skipped_claim_count": len(claim_verification_plan.skipped_claim_ids),
                "triggered_claim_count": len(claim_verification_plan.triggered_claim_ids),
                "route_count": len(claim_verification_plan.route_hints),
                "dependency_count": len(claim_verification_plan.dependencies),
            },
            "staged_verification": None if stage_policy is None else stage_policy.to_dict(),
            "claim_coherence": None if not coherence_enabled else _claim_coherence_metadata(
                initial_coherence_report,
                final_coherence_report,
                missing_dependency_claim_ids=missing_dependency_claim_ids,
            ),
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
        claim_verification_plan=claim_verification_plan,
        verification_stage_decision=stage_decision,
        initial_coherence_report=initial_coherence_report,
        final_coherence_report=final_coherence_report,
    )


def _build_claim_verification_plan(
    *,
    claims: Sequence[Claim],
    stage_policy: StagedVerificationPolicy | None,
    stage_decision: VerificationStageDecision | None,
    claim_dependencies: Sequence[ClaimDependency | Mapping[str, Any]] | None,
) -> ClaimVerificationPlan:
    planner = ClaimVerificationPlanner(
        verify_all_by_default=False,
        verify_claim_feature_flags=(
            DEFAULT_VERIFY_CLAIM_FEATURE_FLAGS
            if stage_policy is None
            else stage_policy.verify_claim_feature_flags
        ),
        verify_claim_metadata_keys=(
            DEFAULT_VERIFY_CLAIM_METADATA_KEYS
            if stage_policy is None
            else stage_policy.verify_claim_metadata_keys
        ),
        verify_triggered_claims_only=(
            False
            if stage_policy is None
            else stage_policy.verify_triggered_claims_only
        ),
        infer_dependencies=claim_dependencies is None,
    )
    route_plan = planner.plan(tuple(claims))
    claim_ids = _claim_ids(claims)
    if stage_decision is None:
        run_verifier = bool(claim_ids)
        return ClaimVerificationPlan(
            run_verifier=run_verifier,
            reason="verification is not staged; all claims are selected" if run_verifier else "no claims to verify",
            verification_scope="all" if run_verifier else "none",
            claims=claims,
            verify_claim_ids=claim_ids if run_verifier else (),
            skipped_claim_ids=(),
            triggered_claim_ids=route_plan.triggered_claim_ids,
            triggered_features=route_plan.triggered_features,
            triggered_metadata=route_plan.triggered_metadata,
            route_hints=route_plan.route_hints,
            retrieval_queries=route_plan.retrieval_queries,
            calculation_checks=route_plan.calculation_checks,
            state_checks=route_plan.state_checks,
            world_model_checks=route_plan.world_model_checks,
            dependencies=(
                route_plan.dependencies
                if claim_dependencies is None
                else tuple(claim_dependencies)
            ),
        )
    return ClaimVerificationPlan(
        run_verifier=stage_decision.run_verifier,
        reason=stage_decision.reason,
        verification_scope=stage_decision.verification_scope,
        claims=claims,
        verify_claim_ids=stage_decision.verify_claim_ids,
        skipped_claim_ids=stage_decision.skipped_claim_ids,
        triggered_claim_ids=stage_decision.triggered_claim_ids,
        triggered_features=stage_decision.triggered_features,
        triggered_metadata=stage_decision.triggered_metadata,
        route_hints=route_plan.route_hints,
        retrieval_queries=route_plan.retrieval_queries,
        calculation_checks=route_plan.calculation_checks,
        state_checks=route_plan.state_checks,
        world_model_checks=route_plan.world_model_checks,
        dependencies=(
            route_plan.dependencies
            if claim_dependencies is None
            else tuple(claim_dependencies)
        ),
    )


def _verify_many_fail_closed(
    verifier: Verifier,
    claims: Sequence[Claim],
    *,
    context: Mapping[str, Any],
    phase: str,
) -> tuple[VerificationResult, ...]:
    claims_tuple = tuple(claims)
    try:
        raw_results = tuple(verifier.verify_many(claims_tuple, context=context))
    except Exception as exc:  # pragma: no cover - exact verifier failures are adapter-defined.
        return tuple(
            _verification_error_result(
                claim,
                claim_index=index,
                phase=phase,
                error=exc,
            )
            for index, claim in enumerate(claims_tuple)
        )
    return _normalize_verification_results(
        raw_results,
        claims_tuple,
        phase=phase,
    )


def _fail_closed_unverified_skip_decision(
    diagnostic_decision: RiskDecision,
    *,
    stage_decision: VerificationStageDecision,
    skipped_claim_ids: Sequence[str],
) -> RiskDecision:
    return RiskDecision(
        action=ControlAction.CLARIFY,
        risk_level=RiskLevel.UNKNOWN,
        confidence=1.0,
        reason="claim verification skipped; unverified claims require clarification",
        diagnostics={
            **dict(diagnostic_decision.diagnostics),
            "diagnostic_decision": diagnostic_decision.to_dict(),
            "verification_skipped_fail_closed": True,
            "verification_stage_reason": stage_decision.reason,
            "skipped_claim_ids": tuple(skipped_claim_ids),
        },
    )


def _verify_one_fail_closed(
    verifier: Verifier,
    claim: Claim,
    *,
    context: Mapping[str, Any],
    phase: str,
    claim_index: int,
) -> VerificationResult:
    try:
        raw_result = verifier.verify(claim, context=context)
    except Exception as exc:  # pragma: no cover - exact verifier failures are adapter-defined.
        return _verification_error_result(
            claim,
            claim_index=claim_index,
            phase=phase,
            error=exc,
        )
    return _coerce_verification_result(raw_result, claim, claim_index=claim_index, phase=phase)


def _normalize_verification_results(
    raw_results: Sequence[VerificationResult | Mapping[str, Any]],
    claims: Sequence[Claim],
    *,
    phase: str,
) -> tuple[VerificationResult, ...]:
    expected = len(claims)
    raw_tuple = tuple(raw_results)
    normalized = [
        _coerce_verification_result(result, claim, claim_index=index, phase=phase)
        for index, (claim, result) in enumerate(zip(claims, raw_tuple))
    ]
    if len(raw_tuple) > expected and normalized:
        extra_count = len(raw_tuple) - expected
        normalized[-1] = _with_result_metadata(
            normalized[-1],
            {
                "verifier_result_mismatch": True,
                "dropped_extra_result_count": extra_count,
                "expected_result_count": expected,
                "actual_result_count": len(raw_tuple),
                "phase": phase,
            },
        )
    if len(raw_tuple) < expected:
        for index in range(len(raw_tuple), expected):
            normalized.append(
                _verification_missing_result(
                    claims[index],
                    claim_index=index,
                    phase=phase,
                    expected_result_count=expected,
                    actual_result_count=len(raw_tuple),
                )
            )
    return tuple(normalized)


def _coerce_verification_result(
    result: VerificationResult | Mapping[str, Any],
    claim: Claim,
    *,
    claim_index: int,
    phase: str,
) -> VerificationResult:
    if isinstance(result, VerificationResult):
        return result
    if not isinstance(result, Mapping):
        return _invalid_verification_result(
            claim,
            claim_index=claim_index,
            phase=phase,
            reason=f"verifier returned {type(result).__name__}",
        )
    try:
        raw_status = result.get("status", VerificationStatus.ERROR.value)
        status = raw_status if isinstance(raw_status, VerificationStatus) else VerificationStatus(str(raw_status))
        evidence = result.get("evidence", ())
        if isinstance(evidence, str):
            evidence_tuple = (evidence,)
        elif isinstance(evidence, Sequence) and not isinstance(evidence, (bytes, bytearray)):
            evidence_tuple = tuple(str(item) for item in evidence)
        else:
            evidence_tuple = ()
        metadata = result.get("metadata", {})
        return VerificationResult(
            status=status,
            confidence=float(result.get("confidence", 0.0)),
            evidence=evidence_tuple,
            explanation=str(result.get("explanation", "")),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )
    except Exception as exc:
        return _invalid_verification_result(
            claim,
            claim_index=claim_index,
            phase=phase,
            reason=f"could not parse verifier result: {type(exc).__name__}: {exc}",
        )


def _verification_error_result(
    claim: Claim,
    *,
    claim_index: int,
    phase: str,
    error: Exception,
) -> VerificationResult:
    error_type = type(error).__name__
    error_text = str(error)
    return VerificationResult(
        status=VerificationStatus.ERROR,
        confidence=1.0,
        explanation=f"verifier failed during {phase}: {error_type}: {error_text}",
        metadata={
            "verifier_error": True,
            "phase": phase,
            "claim_id": _claim_id_for_index(claim, claim_index),
            "error_type": error_type,
            "error": error_text,
        },
    )


def _verification_missing_result(
    claim: Claim,
    *,
    claim_index: int,
    phase: str,
    expected_result_count: int,
    actual_result_count: int,
) -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.ERROR,
        confidence=1.0,
        explanation=f"verifier returned fewer results than claims during {phase}",
        metadata={
            "verifier_result_missing": True,
            "verifier_result_mismatch": True,
            "phase": phase,
            "claim_id": _claim_id_for_index(claim, claim_index),
            "expected_result_count": expected_result_count,
            "actual_result_count": actual_result_count,
        },
    )


def _invalid_verification_result(
    claim: Claim,
    *,
    claim_index: int,
    phase: str,
    reason: str,
) -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.ERROR,
        confidence=1.0,
        explanation=f"invalid verifier result during {phase}: {reason}",
        metadata={
            "invalid_verification_result": True,
            "phase": phase,
            "claim_id": _claim_id_for_index(claim, claim_index),
            "reason": reason,
        },
    )


def _with_result_metadata(
    result: VerificationResult,
    metadata: Mapping[str, Any],
) -> VerificationResult:
    return VerificationResult(
        status=result.status,
        confidence=result.confidence,
        evidence=tuple(result.evidence),
        explanation=result.explanation,
        metadata={**dict(result.metadata), **dict(metadata)},
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
        results.append(
            _verify_one_fail_closed(
                verifier,
                claim,
                context=claim_context,
                phase="final_verification",
                claim_index=index,
            )
        )
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


def _claim_verification_plan_to_dict(
    plan: ClaimVerificationPlan | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if plan is None:
        return None
    if isinstance(plan, ClaimVerificationPlan):
        return plan.to_dict()
    return dict(_jsonable(plan))


def _coherence_report_to_dict(
    report: ClaimCoherenceReport | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if report is None:
        return None
    if isinstance(report, ClaimCoherenceReport):
        return report.to_dict()
    return dict(_jsonable(report))


def _apply_claim_coherence_if_enabled(
    *,
    enabled: bool,
    claims: Sequence[Claim],
    verification_results: Sequence[VerificationResult],
    dependency_claims: Sequence[Claim],
    dependencies: Sequence[ClaimDependency | Mapping[str, Any]] | None,
) -> tuple[tuple[VerificationResult, ...], ClaimCoherenceReport | None]:
    results = tuple(verification_results)
    if not enabled:
        return results, None
    adjusted, report = apply_claim_coherence(
        claims,
        results,
        dependency_claims=dependency_claims,
        dependencies=dependencies,
    )
    return adjusted, report


def _extend_scope_with_missing_parent_claims(
    *,
    verified_claims: Sequence[Claim],
    verification_results: Sequence[VerificationResult],
    all_claims: Sequence[Claim],
    coherence_report: ClaimCoherenceReport | None,
) -> tuple[tuple[Claim, ...], tuple[VerificationResult, ...], tuple[str, ...]]:
    claims = tuple(verified_claims)
    results = tuple(verification_results)
    if coherence_report is None or not coherence_report.missing_parent_ids:
        return claims, results, ()

    existing_ids = set(_claim_ids(claims))
    all_claims_by_id = {
        claim_id: _claim_with_id(claim, claim_id)
        for index, claim in enumerate(all_claims)
        for claim_id in (_claim_id_for_index(claim, index),)
    }
    appended_claims: list[Claim] = []
    appended_results: list[VerificationResult] = []
    appended_ids: list[str] = []
    for claim_id in coherence_report.missing_parent_ids:
        if claim_id in existing_ids:
            continue
        parent_claim = all_claims_by_id.get(claim_id)
        if parent_claim is None:
            continue
        appended_claims.append(parent_claim)
        appended_results.append(
            VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                explanation="parent claim was not verified before coherence check",
                metadata={
                    "claim_coherence": {
                        "missing_parent": True,
                        "source": "dependency_graph",
                    }
                },
            )
        )
        appended_ids.append(claim_id)
        existing_ids.add(claim_id)
    return (
        (*claims, *appended_claims),
        (*results, *appended_results),
        tuple(appended_ids),
    )


def _claim_coherence_metadata(
    initial_report: ClaimCoherenceReport | None,
    final_report: ClaimCoherenceReport | None,
    *,
    missing_dependency_claim_ids: Sequence[str] = (),
) -> dict[str, Any]:
    active_report = final_report or initial_report
    return {
        "enabled": True,
        "dependency_count": 0 if active_report is None else len(active_report.dependencies),
        "initial_issue_count": 0 if initial_report is None else len(initial_report.issues),
        "final_issue_count": 0 if final_report is None else len(final_report.issues),
        "blocked_claim_ids": () if active_report is None else tuple(active_report.blocked_claim_ids),
        "missing_parent_ids": () if active_report is None else tuple(active_report.missing_parent_ids),
        "action_scope_added_claim_ids": tuple(str(item) for item in missing_dependency_claim_ids),
    }


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
    return to_jsonable(value)
