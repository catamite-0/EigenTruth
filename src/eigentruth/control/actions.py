"""Executable action payloads for factuality-control decisions."""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from eigentruth.control.policy import ControlAction, RiskDecision
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus


@dataclass(frozen=True)
class ActionRequest:
    """JSON-ready request produced from a risk decision."""

    action: ControlAction
    reason: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "action": self.action.value,
            "reason": self.reason,
            "payload": _jsonable(self.payload),
            "metadata": _jsonable(self.metadata),
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionRequest":
        """Build an action request from JSON-like data."""
        return cls(
            action=ControlAction(str(data["action"])),
            reason=str(data.get("reason", "")),
            payload=dict(data.get("payload", {})),
            metadata=dict(data.get("metadata", {})),
            request_id=None if data.get("request_id") is None else str(data["request_id"]),
        )


@runtime_checkable
class CorrectionPolicy(Protocol):
    """Interface for turning risk decisions into executable action payloads."""

    def plan(
        self,
        decision: RiskDecision,
        *,
        claims: Sequence[Claim | Mapping[str, Any]] = (),
        verification_results: Sequence[VerificationResult | Mapping[str, Any]] = (),
        context: Mapping[str, Any] | None = None,
    ) -> tuple[ActionRequest, ...]:
        """Build one or more action requests from a risk decision."""
        ...


@dataclass(frozen=True)
class DefaultCorrectionPolicy:
    """Dependency-free action planner for monitor-first product flows."""

    abstain_message: str = "I cannot answer reliably with the available evidence."
    clarify_question: str = "Could you provide more context or evidence for the unsupported claim?"

    def plan(
        self,
        decision: RiskDecision,
        *,
        claims: Sequence[Claim | Mapping[str, Any]] = (),
        verification_results: Sequence[VerificationResult | Mapping[str, Any]] = (),
        context: Mapping[str, Any] | None = None,
    ) -> tuple[ActionRequest, ...]:
        """Build a single action request for the chosen control action."""
        claim_groups = _group_claims_by_verification(claims, verification_results)
        base_payload: dict[str, Any] = {
            "risk_level": decision.risk_level.value,
            "decision_confidence": decision.confidence,
            "decision_reason": decision.reason,
        }
        if claim_groups["total"]:
            base_payload["claim_status_counts"] = claim_groups["counts"]

        action = decision.action
        if action is ControlAction.ACCEPT:
            payload = {**base_payload, "mode": "pass_through"}
        elif action is ControlAction.RETRIEVE:
            payload = {
                **base_payload,
                "retrieval_targets": _targets(
                    claim_groups,
                    VerificationStatus.INSUFFICIENT_EVIDENCE,
                    VerificationStatus.ERROR,
                ),
                "instruction": "retrieve evidence for unresolved claims before answering",
            }
        elif action is ControlAction.REWRITE:
            payload = {
                **base_payload,
                "rewrite_targets": _targets(
                    claim_groups,
                    VerificationStatus.REFUTED,
                    VerificationStatus.INSUFFICIENT_EVIDENCE,
                    VerificationStatus.ERROR,
                ),
                "instruction": "rewrite using supported claims and evidence only",
            }
        elif action is ControlAction.STEER_REGENERATE:
            payload = {
                **base_payload,
                "diagnostics": decision.diagnostics,
                "instruction": "regenerate with the calibrated intervention policy",
            }
        elif action is ControlAction.ABSTAIN:
            payload = {
                **base_payload,
                "blocked_claims": _targets(
                    claim_groups,
                    VerificationStatus.REFUTED,
                    VerificationStatus.INSUFFICIENT_EVIDENCE,
                    VerificationStatus.ERROR,
                ),
                "message": self.abstain_message,
            }
        elif action is ControlAction.CLARIFY:
            payload = {
                **base_payload,
                "clarification_targets": _targets(
                    claim_groups,
                    VerificationStatus.INSUFFICIENT_EVIDENCE,
                    VerificationStatus.ERROR,
                ),
                "questions": (self.clarify_question,),
            }
        else:
            payload = base_payload

        metadata = {
            "policy": type(self).__name__,
            "context": dict(context or {}),
        }
        return (
            ActionRequest(
                action=action,
                reason=decision.reason,
                payload=payload,
                metadata=metadata,
            ),
        )


def _group_claims_by_verification(
    claims: Sequence[Claim | Mapping[str, Any]],
    verification_results: Sequence[VerificationResult | Mapping[str, Any]],
) -> dict[str, Any]:
    groups = {status.value: [] for status in VerificationStatus}
    counts = {status.value: 0 for status in VerificationStatus}
    n_items = max(len(claims), len(verification_results))
    for index in range(n_items):
        claim = claims[index] if index < len(claims) else None
        result = verification_results[index] if index < len(verification_results) else None
        if result is None:
            status = VerificationStatus.NOT_APPLICABLE
            confidence = 0.0
            evidence: tuple[str, ...] = ()
        else:
            status = _verification_status(result)
            confidence = _verification_confidence(result)
            evidence = _verification_evidence(result)
        item = {
            **_claim_to_dict(claim, fallback_id=f"c{index + 1}"),
            "status": status.value,
            "confidence": confidence,
            "evidence": evidence,
        }
        groups[status.value].append(item)
        counts[status.value] = counts.get(status.value, 0) + 1
    return {"groups": groups, "counts": counts, "total": n_items}


def _targets(claim_groups: Mapping[str, Any], *statuses: VerificationStatus) -> tuple[dict[str, Any], ...]:
    groups = claim_groups.get("groups", {})
    selected = []
    if isinstance(groups, Mapping):
        for status in statuses:
            selected.extend(groups.get(status.value, ()))
    return tuple(dict(item) for item in selected)


def _claim_to_dict(claim: Claim | Mapping[str, Any] | None, *, fallback_id: str) -> dict[str, Any]:
    if isinstance(claim, Claim):
        return {
            "claim_id": claim.claim_id or fallback_id,
            "text": claim.text,
            "metadata": _jsonable(claim.metadata),
        }
    if isinstance(claim, Mapping):
        claim_id = claim.get("claim_id", fallback_id)
        return {
            "claim_id": None if claim_id is None else str(claim_id),
            "text": str(claim.get("text", "")),
            "metadata": _jsonable(dict(claim.get("metadata", {}))),
        }
    return {"claim_id": fallback_id, "text": "", "metadata": {}}


def _verification_status(result: VerificationResult | Mapping[str, Any]) -> VerificationStatus:
    if isinstance(result, VerificationResult):
        return result.status
    raw_status = result.get("status", VerificationStatus.ERROR.value)
    if isinstance(raw_status, VerificationStatus):
        return raw_status
    try:
        return VerificationStatus(str(raw_status))
    except ValueError:
        return VerificationStatus.ERROR


def _verification_confidence(result: VerificationResult | Mapping[str, Any]) -> float:
    if isinstance(result, VerificationResult):
        return result.confidence
    try:
        value = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, value))


def _verification_evidence(result: VerificationResult | Mapping[str, Any]) -> tuple[str, ...]:
    if isinstance(result, VerificationResult):
        return tuple(result.evidence)
    raw_evidence = result.get("evidence", ())
    if isinstance(raw_evidence, str):
        return (raw_evidence,)
    if isinstance(raw_evidence, Sequence):
        return tuple(str(item) for item in raw_evidence)
    return ()


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
