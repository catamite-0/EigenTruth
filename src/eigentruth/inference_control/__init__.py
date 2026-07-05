"""Inference-time correction controller for evidence-grounded revision."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from eigentruth.json_utils import to_jsonable
from eigentruth.revision import RevisionTrace


class RevisionAction(str, Enum):
    """Controller actions for inference-time correction."""

    ACCEPT = "accept"
    REVISE = "revise"
    RETRIEVE_MORE = "retrieve_more"
    REGENERATE = "regenerate"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class CorrectionPolicy:
    """Thresholds for converting runtime signals into correction actions."""

    risk_score_threshold: float = 0.75
    uncertainty_threshold: float = 0.65
    contradiction_threshold: int = 1
    unsupported_threshold: int = 1
    unresolved_action: RevisionAction | str = RevisionAction.RETRIEVE_MORE
    allow_regenerate: bool = True

    def __post_init__(self) -> None:
        for name in ("risk_score_threshold", "uncertainty_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a non-negative finite number.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "contradiction_threshold", max(1, int(self.contradiction_threshold)))
        object.__setattr__(self, "unsupported_threshold", max(1, int(self.unsupported_threshold)))
        object.__setattr__(self, "unresolved_action", _coerce_action(self.unresolved_action))
        object.__setattr__(self, "allow_regenerate", bool(self.allow_regenerate))

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score_threshold": self.risk_score_threshold,
            "uncertainty_threshold": self.uncertainty_threshold,
            "contradiction_threshold": self.contradiction_threshold,
            "unsupported_threshold": self.unsupported_threshold,
            "unresolved_action": self.unresolved_action.value,
            "allow_regenerate": self.allow_regenerate,
        }


@dataclass(frozen=True)
class InferenceCorrectionDecision:
    """A traceable inference-control decision."""

    action: RevisionAction | str
    reason: str
    triggered_signals: Mapping[str, Any] = field(default_factory=dict)
    policy: CorrectionPolicy | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _coerce_action(self.action))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "triggered_signals", dict(self.triggered_signals))

    def to_dict(self) -> dict[str, Any]:
        policy = self.policy
        if isinstance(policy, CorrectionPolicy):
            policy_payload: Mapping[str, Any] | None = policy.to_dict()
        elif isinstance(policy, Mapping):
            policy_payload = dict(policy)
        else:
            policy_payload = None
        return {
            "action": self.action.value,
            "reason": self.reason,
            "triggered_signals": to_jsonable(self.triggered_signals),
            "policy": None if policy_payload is None else to_jsonable(policy_payload),
        }


class InferenceCorrectionController:
    """Policy engine that decides whether to accept, revise, retrieve, or stop."""

    def __init__(self, policy: CorrectionPolicy | None = None) -> None:
        self.policy = policy or CorrectionPolicy()

    def decide(
        self,
        *,
        revision_trace: RevisionTrace | Mapping[str, Any] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        verification_summary: Mapping[str, Any] | None = None,
    ) -> InferenceCorrectionDecision:
        trace = _trace_payload(revision_trace)
        diagnostics_payload = dict(diagnostics or {})
        verification_payload = dict(verification_summary or {})

        if _truthy(trace.get("unsupported_persistence")):
            return self._decision(
                RevisionAction.REVISE,
                "unsupported claim persisted after revision",
                {"unsupported_persistence": True},
            )
        status_counts = _mapping(_mapping(trace.get("summary")).get("status_counts"))
        contradicted = _int(status_counts.get("contradicted"))
        unresolved = _int(status_counts.get("unresolved"))
        insufficient = _int(status_counts.get("insufficient"))
        if contradicted >= self.policy.contradiction_threshold:
            return self._decision(
                RevisionAction.REVISE,
                "contradicted claim evidence reached correction threshold",
                {"contradicted_claims": contradicted},
            )
        if unresolved > 0:
            return self._decision(
                self.policy.unresolved_action,
                "revision evidence is unresolved",
                {"unresolved_claims": unresolved},
            )
        unsupported = _int(verification_payload.get("unsupported_count"))
        if unsupported >= self.policy.unsupported_threshold:
            return self._decision(
                RevisionAction.REVISE,
                "verification found unsupported claims",
                {"unsupported_count": unsupported},
            )
        if insufficient > 0:
            return self._decision(
                RevisionAction.RETRIEVE_MORE,
                "revision evidence is insufficient",
                {"insufficient_claims": insufficient},
            )
        risk_score = _finite_float(
            diagnostics_payload.get("risk_score", diagnostics_payload.get("pre_generation_risk_score"))
        )
        if risk_score is not None and risk_score >= self.policy.risk_score_threshold:
            action = RevisionAction.REGENERATE if self.policy.allow_regenerate else RevisionAction.RETRIEVE_MORE
            return self._decision(
                action,
                "diagnostic risk score reached correction threshold",
                {"risk_score": risk_score},
            )
        uncertainty = _finite_float(diagnostics_payload.get("uncertainty", diagnostics_payload.get("entropy")))
        if uncertainty is not None and uncertainty >= self.policy.uncertainty_threshold:
            return self._decision(
                RevisionAction.RETRIEVE_MORE,
                "uncertainty reached evidence-acquisition threshold",
                {"uncertainty": uncertainty},
            )
        if str(trace.get("action") or "").strip().lower() == "abstain":
            return self._decision(RevisionAction.ABSTAIN, "revision engine abstained", {"revision_action": "abstain"})
        return self._decision(RevisionAction.ACCEPT, "no correction trigger fired", {})

    def _decision(
        self,
        action: RevisionAction | str,
        reason: str,
        signals: Mapping[str, Any],
    ) -> InferenceCorrectionDecision:
        return InferenceCorrectionDecision(
            action=action,
            reason=reason,
            triggered_signals=signals,
            policy=self.policy,
        )


def _coerce_action(value: RevisionAction | str) -> RevisionAction:
    if isinstance(value, RevisionAction):
        return value
    raw = str(value).strip().lower()
    for action in RevisionAction:
        if action.value == raw:
            return action
    allowed = ", ".join(action.value for action in RevisionAction)
    raise ValueError(f"unknown revision action {value!r}; allowed: {allowed}.")


def _trace_payload(value: RevisionTrace | Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, RevisionTrace):
        return value.to_dict()
    return dict(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = [
    "CorrectionPolicy",
    "InferenceCorrectionController",
    "InferenceCorrectionDecision",
    "RevisionAction",
]

