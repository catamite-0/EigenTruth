"""Closed-loop verification helpers for control-plane workflows."""

from __future__ import annotations

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
from eigentruth.control.trace import ProductTrace, TraceEvent
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
) -> VerificationLoopResult:
    """Run a dependency-free verification/action/reverification loop."""
    base_context = dict(context or {})
    initial_results = tuple(verifier.verify_many(claims, context=base_context))
    initial_decision = controller.decide(diagnostics, verification_results=initial_results)

    policy = correction_policy or DefaultCorrectionPolicy()
    action_requests = policy.plan(
        initial_decision,
        claims=claims,
        verification_results=initial_results,
        context=base_context,
    )
    registry = executor_registry or ActionExecutorRegistry()
    execution_context = {**base_context, "request_id": request_id}
    action_results = registry.execute_many(action_requests, context=execution_context)
    retrieval_evidence = evidence_bundle_from_action_results(action_results)

    if retrieval_evidence.has_evidence():
        final_results = _verify_with_retrieved_evidence(
            verifier,
            claims,
            base_context=base_context,
            evidence_bundle=retrieval_evidence,
        )
        final_decision = controller.decide(diagnostics, verification_results=final_results)
    else:
        final_results = initial_results
        final_decision = initial_decision

    trace = ProductTrace(
        request_id=request_id,
        diagnostics=diagnostics,
        claims=claims,
        verification_results=final_results,
        risk_decision=final_decision,
        actions=action_requests,
        action_results=action_results,
        events=(
            TraceEvent(
                "initial_verification",
                {
                    "n_claims": len(claims),
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
            **dict(metadata or {}),
        },
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
        claim_context = {**base_context, **evidence_bundle.to_context(claim_id)}
        results.append(verifier.verify(claim, context=claim_context))
    return tuple(results)


def _action_result_to_dict(result: ActionResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, ActionResult):
        return result.to_dict()
    return dict(_jsonable(result))


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
