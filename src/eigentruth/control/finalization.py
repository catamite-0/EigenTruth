"""Final response assembly for factuality-control workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from eigentruth.control.actions import ActionRequest, ActionResult
from eigentruth.control.policy import ControlAction, RiskDecision, RiskLevel
from eigentruth.json_utils import to_jsonable
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus


class FinalAnswerStatus(str, Enum):
    """Product-facing finalization status."""

    ANSWERED = "answered"
    ABSTAINED = "abstained"
    NEEDS_CLARIFICATION = "needs_clarification"
    NEEDS_RETRIEVAL = "needs_retrieval"
    NEEDS_REWRITE = "needs_rewrite"
    NEEDS_REGENERATION = "needs_regeneration"
    NEEDS_TOOL_EXECUTION = "needs_tool_execution"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class FinalAnswer:
    """JSON-ready product output from an EigenTruth control loop.

    The finalizer is intentionally conservative: it can pass through a draft
    answer after an accept decision, or expose a structured non-answer status
    with reasons and evidence. It does not call a model or synthesize corrected
    factual content.
    """

    status: FinalAnswerStatus | str
    text: str
    answerable: bool
    action: ControlAction | str
    risk_level: RiskLevel | str
    confidence: float
    reason: str
    claim_summary: Mapping[str, Any] = field(default_factory=dict)
    evidence: Sequence[Mapping[str, Any]] = ()
    followup: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        confidence = float(self.confidence)
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1].")
        object.__setattr__(self, "status", _coerce_final_status(self.status))
        object.__setattr__(self, "action", _coerce_action(self.action))
        object.__setattr__(self, "risk_level", _coerce_risk_level(self.risk_level))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "text", str(self.text))
        object.__setattr__(self, "answerable", bool(self.answerable))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "claim_summary", dict(self.claim_summary))
        object.__setattr__(self, "evidence", tuple(dict(item) for item in self.evidence))
        object.__setattr__(self, "followup", dict(self.followup))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-ready final answer payload."""
        return {
            "status": self.status.value,
            "text": self.text,
            "answerable": self.answerable,
            "action": self.action.value,
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "claim_summary": to_jsonable(self.claim_summary),
            "evidence": tuple(to_jsonable(item) for item in self.evidence),
            "followup": to_jsonable(self.followup),
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FinalAnswer":
        """Build a final answer from a JSON-like payload."""
        return cls(
            status=FinalAnswerStatus(str(data["status"])),
            text=str(data.get("text", "")),
            answerable=bool(data.get("answerable", False)),
            action=ControlAction(str(data["action"])),
            risk_level=RiskLevel(str(data["risk_level"])),
            confidence=float(data.get("confidence", 0.0)),
            reason=str(data.get("reason", "")),
            claim_summary=dict(data.get("claim_summary", {})),
            evidence=tuple(_dict_items(data.get("evidence", ()))),
            followup=dict(data.get("followup", {})),
            metadata=dict(data.get("metadata", {})),
        )


def finalize_loop_answer(
    loop_result: Any,
    *,
    draft_answer: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    max_evidence_items: int = 8,
) -> FinalAnswer:
    """Build a product final answer from ``run_verification_loop`` output."""
    trace = getattr(loop_result, "trace", None)
    diagnostics = getattr(trace, "diagnostics", {}) if trace is not None else {}
    claims = getattr(trace, "claims", ()) if trace is not None else ()
    return finalize_answer(
        draft_answer=draft_answer,
        decision=getattr(loop_result, "final_decision"),
        claims=claims,
        verification_results=getattr(loop_result, "final_verification_results", ()),
        action_requests=getattr(loop_result, "action_requests", ()),
        action_results=getattr(loop_result, "action_results", ()),
        diagnostics=diagnostics,
        metadata={
            "source": "eigentruth.control.finalize_loop_answer",
            **dict(metadata or {}),
        },
        max_evidence_items=max_evidence_items,
    )


def finalize_answer(
    *,
    draft_answer: str | None = None,
    decision: RiskDecision | Mapping[str, Any],
    claims: Sequence[Claim | Mapping[str, Any]] = (),
    verification_results: Sequence[VerificationResult | Mapping[str, Any]] = (),
    action_requests: Sequence[ActionRequest | Mapping[str, Any]] = (),
    action_results: Sequence[ActionResult | Mapping[str, Any]] = (),
    diagnostics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    max_evidence_items: int = 8,
) -> FinalAnswer:
    """Assemble a conservative final answer payload from control outputs."""
    decision_payload = _decision_to_dict(decision)
    action = _coerce_action(decision_payload["action"])
    risk_level = _coerce_risk_level(decision_payload["risk_level"])
    confidence = float(decision_payload.get("confidence", 0.0))
    reason = str(decision_payload.get("reason", ""))
    claim_summary, evidence = _summarize_claims(
        claims,
        verification_results,
        max_evidence_items=max_evidence_items,
    )
    followup = _followup_payload(
        action,
        action_requests=action_requests,
        action_results=action_results,
        claim_summary=claim_summary,
    )
    status = _status_for_action(action)
    answerable = action is ControlAction.ACCEPT
    text = _final_text(
        action,
        draft_answer=draft_answer,
        followup=followup,
    )
    return FinalAnswer(
        status=status,
        text=text,
        answerable=answerable,
        action=action,
        risk_level=risk_level,
        confidence=confidence,
        reason=reason,
        claim_summary=claim_summary,
        evidence=evidence,
        followup=followup,
        metadata={
            "finalizer": "conservative_pass_through",
            "draft_answer_present": draft_answer is not None,
            "diagnostics": to_jsonable(dict(diagnostics or {})),
            **dict(metadata or {}),
        },
    )


def _summarize_claims(
    claims: Sequence[Claim | Mapping[str, Any]],
    verification_results: Sequence[VerificationResult | Mapping[str, Any]],
    *,
    max_evidence_items: int,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    counts = {status.value: 0 for status in VerificationStatus}
    claim_ids = {status.value: [] for status in VerificationStatus}
    blocked_claims: list[dict[str, Any]] = []
    evidence_items: list[dict[str, Any]] = []
    n_items = max(len(claims), len(verification_results))

    for index in range(n_items):
        claim = claims[index] if index < len(claims) else {}
        result = verification_results[index] if index < len(verification_results) else {}
        claim_payload = _claim_to_dict(claim, fallback_id=f"c{index + 1}")
        result_payload = _verification_result_to_dict(result)
        status = _coerce_verification_status(result_payload.get("status", VerificationStatus.NOT_APPLICABLE))
        claim_id = str(claim_payload.get("claim_id") or f"c{index + 1}")
        counts[status.value] = counts.get(status.value, 0) + 1
        claim_ids.setdefault(status.value, []).append(claim_id)
        item = {
            "claim_id": claim_id,
            "text": claim_payload.get("text", ""),
            "status": status.value,
            "confidence": float(result_payload.get("confidence", 0.0)),
            "explanation": result_payload.get("explanation", ""),
        }
        if status in {
            VerificationStatus.REFUTED,
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            VerificationStatus.ERROR,
        }:
            blocked_claims.append(item)
        for evidence_text in _evidence_strings(result_payload.get("evidence", ())):
            if len(evidence_items) >= max_evidence_items:
                break
            evidence_items.append({
                "claim_id": claim_id,
                "status": status.value,
                "text": evidence_text,
            })

    summary = {
        "total_claims": n_items,
        "status_counts": counts,
        "supported_claim_ids": tuple(claim_ids.get(VerificationStatus.SUPPORTED.value, ())),
        "refuted_claim_ids": tuple(claim_ids.get(VerificationStatus.REFUTED.value, ())),
        "unsupported_claim_ids": tuple(
            claim_ids.get(VerificationStatus.INSUFFICIENT_EVIDENCE.value, ())
        ),
        "error_claim_ids": tuple(claim_ids.get(VerificationStatus.ERROR.value, ())),
        "not_applicable_claim_ids": tuple(
            claim_ids.get(VerificationStatus.NOT_APPLICABLE.value, ())
        ),
        "blocked_claims": tuple(blocked_claims),
    }
    return summary, tuple(evidence_items)


def _followup_payload(
    action: ControlAction,
    *,
    action_requests: Sequence[ActionRequest | Mapping[str, Any]],
    action_results: Sequence[ActionResult | Mapping[str, Any]],
    claim_summary: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "action": action.value,
        "requires_followup": action is not ControlAction.ACCEPT,
        "blocked_claim_count": len(tuple(claim_summary.get("blocked_claims", ()))),
    }
    request_payload = _first_action_payload(action_requests, action)
    result_output = _first_action_output(action_results, action)
    if request_payload:
        payload["request_payload"] = request_payload
    if result_output:
        payload["result_output"] = result_output

    if action is ControlAction.ABSTAIN:
        message = result_output.get("message") or request_payload.get("message")
        if message:
            payload["message"] = str(message)
    elif action is ControlAction.CLARIFY:
        questions = result_output.get("questions") or request_payload.get("questions") or ()
        payload["questions"] = tuple(str(item) for item in _as_sequence(questions))
    elif action is ControlAction.RETRIEVE:
        payload["retrieval_targets"] = tuple(
            _dict_items(request_payload.get("retrieval_targets", ()))
        )
        payload["instruction"] = str(
            request_payload.get("instruction")
            or result_output.get("instruction")
            or "retrieve evidence before answering"
        )
    elif action is ControlAction.REWRITE:
        payload["rewrite_targets"] = tuple(
            _dict_items(request_payload.get("rewrite_targets", ()))
        )
    return payload


def _final_text(
    action: ControlAction,
    *,
    draft_answer: str | None,
    followup: Mapping[str, Any],
) -> str:
    if action is ControlAction.ACCEPT:
        return "" if draft_answer is None else str(draft_answer)
    if action is ControlAction.ABSTAIN:
        return str(followup.get("message") or "I cannot answer reliably with the available evidence.")
    if action is ControlAction.CLARIFY:
        questions = tuple(str(item) for item in _as_sequence(followup.get("questions", ())))
        return questions[0] if questions else "I need more context before answering reliably."
    if action is ControlAction.RETRIEVE:
        return "I need more evidence before answering reliably."
    if action is ControlAction.REWRITE:
        return "The draft answer needs rewriting against verified evidence before finalization."
    if action is ControlAction.STEER_REGENERATE:
        return "The answer should be regenerated under calibrated controls before finalization."
    if action is ControlAction.EXECUTE_TOOL:
        return "Tool execution is required before answering reliably."
    return "The answer is blocked by the current control policy."


def _status_for_action(action: ControlAction) -> FinalAnswerStatus:
    if action is ControlAction.ACCEPT:
        return FinalAnswerStatus.ANSWERED
    if action is ControlAction.ABSTAIN:
        return FinalAnswerStatus.ABSTAINED
    if action is ControlAction.CLARIFY:
        return FinalAnswerStatus.NEEDS_CLARIFICATION
    if action is ControlAction.RETRIEVE:
        return FinalAnswerStatus.NEEDS_RETRIEVAL
    if action is ControlAction.REWRITE:
        return FinalAnswerStatus.NEEDS_REWRITE
    if action is ControlAction.STEER_REGENERATE:
        return FinalAnswerStatus.NEEDS_REGENERATION
    if action is ControlAction.EXECUTE_TOOL:
        return FinalAnswerStatus.NEEDS_TOOL_EXECUTION
    return FinalAnswerStatus.BLOCKED


def _first_action_payload(
    actions: Sequence[ActionRequest | Mapping[str, Any]],
    action: ControlAction,
) -> dict[str, Any]:
    for item in actions:
        payload = _action_request_to_dict(item)
        if payload.get("action") == action.value:
            return dict(payload.get("payload", {}))
    return {}


def _first_action_output(
    results: Sequence[ActionResult | Mapping[str, Any]],
    action: ControlAction,
) -> dict[str, Any]:
    for item in results:
        payload = _action_result_to_dict(item)
        if payload.get("action") == action.value:
            output = payload.get("output", {})
            return dict(output) if isinstance(output, Mapping) else {}
    return {}


def _decision_to_dict(decision: RiskDecision | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(decision, RiskDecision):
        return decision.to_dict()
    payload = dict(decision)
    if "action" in payload:
        payload["action"] = _coerce_action(payload["action"]).value
    if "risk_level" in payload:
        payload["risk_level"] = _coerce_risk_level(payload["risk_level"]).value
    return payload


def _action_request_to_dict(action: ActionRequest | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(action, ActionRequest):
        return action.to_dict()
    return dict(action)


def _action_result_to_dict(result: ActionResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, ActionResult):
        return result.to_dict()
    return dict(result)


def _claim_to_dict(claim: Claim | Mapping[str, Any], *, fallback_id: str) -> dict[str, Any]:
    if isinstance(claim, Claim):
        return {
            "claim_id": claim.claim_id or fallback_id,
            "text": claim.text,
            "span": claim.span,
            "metadata": to_jsonable(claim.metadata),
        }
    payload = dict(claim)
    payload.setdefault("claim_id", fallback_id)
    payload.setdefault("text", "")
    return payload


def _verification_result_to_dict(result: VerificationResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, VerificationResult):
        return {
            "status": result.status.value,
            "confidence": result.confidence,
            "evidence": result.evidence,
            "explanation": result.explanation,
            "metadata": to_jsonable(result.metadata),
        }
    payload = dict(result)
    payload.setdefault("status", VerificationStatus.NOT_APPLICABLE.value)
    payload.setdefault("confidence", 0.0)
    payload.setdefault("evidence", ())
    payload.setdefault("explanation", "")
    return payload


def _evidence_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _dict_items(value: Any) -> tuple[dict[str, Any], ...]:
    items = _as_sequence(value)
    return tuple(dict(item) for item in items if isinstance(item, Mapping))


def _coerce_final_status(status: FinalAnswerStatus | str) -> FinalAnswerStatus:
    return status if isinstance(status, FinalAnswerStatus) else FinalAnswerStatus(str(status))


def _coerce_action(action: ControlAction | str) -> ControlAction:
    return action if isinstance(action, ControlAction) else ControlAction(str(action))


def _coerce_risk_level(risk_level: RiskLevel | str) -> RiskLevel:
    return risk_level if isinstance(risk_level, RiskLevel) else RiskLevel(str(risk_level))


def _coerce_verification_status(status: VerificationStatus | str | Any) -> VerificationStatus:
    if isinstance(status, VerificationStatus):
        return status
    return VerificationStatus(str(status))
