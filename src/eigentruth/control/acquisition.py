"""Budgeted evidence-acquisition decisions for control loops.

The helpers in this module are intentionally dependency-free. They provide a
small product-control analogue of budgeted conformal evidence acquisition:
decide whether the current evidence is enough to answer, whether one more
evidence-acquisition step is worth spending, or whether the request should
abstain/clarify under the configured budget. Statistical guarantees still need
post-acquisition calibration of the whole policy score; the decision payload
records that requirement so traces do not imply a naive pre-acquisition
conformal guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence

from eigentruth.control.policy import ControlAction, RiskDecision, RiskLevel
from eigentruth.json_utils import to_jsonable
from eigentruth.verify.planning import ClaimVerificationPlan
from eigentruth.verify.protocols import VerificationResult, VerificationStatus


class EvidenceAcquisitionAction(str, Enum):
    """Three-way evidence-acquisition action."""

    ANSWER = "answer"
    ACQUIRE = "acquire"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class EvidenceAcquisitionDecision:
    """JSON-ready decision from a budgeted evidence-acquisition policy."""

    action: EvidenceAcquisitionAction
    control_action: ControlAction
    risk_level: RiskLevel
    reason: str
    acquisition_round: int = 0
    budget_exhausted: bool = False
    selected_claim_ids: Sequence[str] = ()
    selected_retrieval_queries: Sequence[Mapping[str, Any]] = ()
    dropped_retrieval_queries: Sequence[Mapping[str, Any]] = ()
    status_counts: Mapping[str, int] = field(default_factory=dict)
    estimated_cost_units: float = 0.0
    calibration_scope: str = "post_acquisition_policy"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action = self.action if isinstance(self.action, EvidenceAcquisitionAction) else EvidenceAcquisitionAction(
            str(self.action)
        )
        control_action = (
            self.control_action
            if isinstance(self.control_action, ControlAction)
            else ControlAction(str(self.control_action))
        )
        risk_level = self.risk_level if isinstance(self.risk_level, RiskLevel) else RiskLevel(str(self.risk_level))
        acquisition_round = _non_negative_int(self.acquisition_round, name="acquisition_round")
        estimated_cost_units = _non_negative_float(
            self.estimated_cost_units,
            name="estimated_cost_units",
        )
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "control_action", control_action)
        object.__setattr__(self, "risk_level", risk_level)
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "acquisition_round", acquisition_round)
        object.__setattr__(self, "budget_exhausted", _strict_bool(self.budget_exhausted))
        object.__setattr__(self, "selected_claim_ids", tuple(_non_empty_strings(self.selected_claim_ids)))
        object.__setattr__(
            self,
            "selected_retrieval_queries",
            tuple(_jsonable_mapping(item) for item in self.selected_retrieval_queries),
        )
        object.__setattr__(
            self,
            "dropped_retrieval_queries",
            tuple(_jsonable_mapping(item) for item in self.dropped_retrieval_queries),
        )
        object.__setattr__(
            self,
            "status_counts",
            {
                str(key): _non_negative_int(value, name=f"status_counts.{key}")
                for key, value in self.status_counts.items()
            },
        )
        object.__setattr__(self, "estimated_cost_units", estimated_cost_units)
        object.__setattr__(self, "calibration_scope", str(self.calibration_scope).strip() or "post_acquisition_policy")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable decision payload."""
        return {
            "action": self.action.value,
            "control_action": self.control_action.value,
            "risk_level": self.risk_level.value,
            "reason": self.reason,
            "acquisition_round": self.acquisition_round,
            "budget_exhausted": self.budget_exhausted,
            "selected_claim_ids": tuple(self.selected_claim_ids),
            "selected_retrieval_queries": tuple(dict(item) for item in self.selected_retrieval_queries),
            "dropped_retrieval_queries": tuple(dict(item) for item in self.dropped_retrieval_queries),
            "status_counts": dict(self.status_counts),
            "estimated_cost_units": self.estimated_cost_units,
            "calibration_scope": self.calibration_scope,
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class EvidenceAcquisitionPolicy:
    """Budgeted answer/acquire/abstain policy for unresolved claims.

    The policy is deliberately conservative. Refuted claims are terminal by
    default, unsupported/error claims acquire evidence if budget remains, and
    budget exhaustion routes to a configured fail-closed control action.
    """

    max_acquisition_rounds: int = 1
    max_retrieval_queries: int | None = None
    max_estimated_cost_units: float | None = None
    acquire_statuses: Sequence[VerificationStatus | str] = (
        VerificationStatus.INSUFFICIENT_EVIDENCE,
        VerificationStatus.ERROR,
    )
    terminal_statuses: Sequence[VerificationStatus | str] = (VerificationStatus.REFUTED,)
    acquire_actions: Sequence[ControlAction | str] = (ControlAction.RETRIEVE,)
    budget_exhausted_action: ControlAction | str = ControlAction.ABSTAIN
    calibration_scope: str = "post_acquisition_policy"

    def __post_init__(self) -> None:
        max_acquisition_rounds = _non_negative_int(
            self.max_acquisition_rounds,
            name="max_acquisition_rounds",
        )
        max_retrieval_queries = _optional_non_negative_int(
            self.max_retrieval_queries,
            name="max_retrieval_queries",
        )
        max_estimated_cost_units = _optional_non_negative_float(
            self.max_estimated_cost_units,
            name="max_estimated_cost_units",
        )
        acquire_statuses = tuple(_verification_status(value).value for value in self.acquire_statuses)
        terminal_statuses = tuple(_verification_status(value).value for value in self.terminal_statuses)
        acquire_actions = tuple(_control_action(value) for value in self.acquire_actions)
        if not acquire_statuses and not acquire_actions:
            raise ValueError("EvidenceAcquisitionPolicy needs acquire_statuses or acquire_actions.")
        object.__setattr__(self, "max_acquisition_rounds", max_acquisition_rounds)
        object.__setattr__(self, "max_retrieval_queries", max_retrieval_queries)
        object.__setattr__(self, "max_estimated_cost_units", max_estimated_cost_units)
        object.__setattr__(self, "acquire_statuses", acquire_statuses)
        object.__setattr__(self, "terminal_statuses", terminal_statuses)
        object.__setattr__(self, "acquire_actions", acquire_actions)
        object.__setattr__(
            self,
            "budget_exhausted_action",
            _control_action(self.budget_exhausted_action),
        )
        object.__setattr__(self, "calibration_scope", str(self.calibration_scope).strip() or "post_acquisition_policy")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvidenceAcquisitionPolicy":
        """Build a policy from a JSON-like mapping."""
        return cls(
            max_acquisition_rounds=payload.get("max_acquisition_rounds", 1),
            max_retrieval_queries=payload.get("max_retrieval_queries"),
            max_estimated_cost_units=payload.get("max_estimated_cost_units"),
            acquire_statuses=tuple(
                _as_sequence(
                    payload.get(
                        "acquire_statuses",
                        (
                            VerificationStatus.INSUFFICIENT_EVIDENCE.value,
                            VerificationStatus.ERROR.value,
                        ),
                    )
                )
            ),
            terminal_statuses=tuple(
                _as_sequence(payload.get("terminal_statuses", (VerificationStatus.REFUTED.value,)))
            ),
            acquire_actions=tuple(_as_sequence(payload.get("acquire_actions", (ControlAction.RETRIEVE.value,)))),
            budget_exhausted_action=payload.get("budget_exhausted_action", ControlAction.ABSTAIN.value),
            calibration_scope=str(payload.get("calibration_scope", "post_acquisition_policy")),
        )

    def decide(
        self,
        decision: RiskDecision,
        *,
        verification_results: Sequence[VerificationResult | Mapping[str, Any]] = (),
        verification_plan: ClaimVerificationPlan | Mapping[str, Any] | None = None,
        acquisition_round: int = 0,
    ) -> EvidenceAcquisitionDecision:
        """Return whether to answer, acquire more evidence, or abstain."""
        acquisition_round = _non_negative_int(acquisition_round, name="acquisition_round")
        status_counts = _status_counts(verification_results)
        plan = _plan_obj(verification_plan)
        selected_queries, dropped_queries = self._split_retrieval_queries(plan)
        selected_claim_ids = _selected_claim_ids(plan, verification_results)
        estimated_cost_units = _estimated_cost_units(plan)

        terminal_hits = tuple(status for status in self.terminal_statuses if status_counts.get(status, 0))
        if terminal_hits:
            return EvidenceAcquisitionDecision(
                action=EvidenceAcquisitionAction.ABSTAIN,
                control_action=ControlAction.ABSTAIN,
                risk_level=RiskLevel.HIGH,
                reason="terminal verification status present",
                acquisition_round=acquisition_round,
                selected_claim_ids=selected_claim_ids,
                selected_retrieval_queries=selected_queries,
                dropped_retrieval_queries=dropped_queries,
                status_counts=status_counts,
                estimated_cost_units=estimated_cost_units,
                calibration_scope=self.calibration_scope,
                metadata={
                    "terminal_statuses": terminal_hits,
                    "post_acquisition_calibration_required": True,
                },
            )

        needs_acquisition = self._needs_acquisition(decision, status_counts=status_counts, plan=plan)
        if not needs_acquisition:
            return EvidenceAcquisitionDecision(
                action=EvidenceAcquisitionAction.ANSWER,
                control_action=decision.action,
                risk_level=decision.risk_level,
                reason="current evidence is sufficient for the controller decision",
                acquisition_round=acquisition_round,
                selected_claim_ids=selected_claim_ids,
                selected_retrieval_queries=selected_queries,
                dropped_retrieval_queries=dropped_queries,
                status_counts=status_counts,
                estimated_cost_units=estimated_cost_units,
                calibration_scope=self.calibration_scope,
                metadata={"post_acquisition_calibration_required": False},
            )

        budget_reasons = self._budget_exhaustion_reasons(
            acquisition_round=acquisition_round,
            estimated_cost_units=estimated_cost_units,
            selected_queries=selected_queries,
            dropped_queries=dropped_queries,
        )
        if budget_reasons:
            return EvidenceAcquisitionDecision(
                action=EvidenceAcquisitionAction.ABSTAIN,
                control_action=self.budget_exhausted_action,
                risk_level=(
                    RiskLevel.UNKNOWN
                    if self.budget_exhausted_action is ControlAction.CLARIFY
                    else RiskLevel.HIGH
                ),
                reason="evidence-acquisition budget exhausted",
                acquisition_round=acquisition_round,
                budget_exhausted=True,
                selected_claim_ids=selected_claim_ids,
                selected_retrieval_queries=selected_queries,
                dropped_retrieval_queries=dropped_queries,
                status_counts=status_counts,
                estimated_cost_units=estimated_cost_units,
                calibration_scope=self.calibration_scope,
                metadata={
                    "budget_reasons": budget_reasons,
                    "post_acquisition_calibration_required": True,
                },
            )

        return EvidenceAcquisitionDecision(
            action=EvidenceAcquisitionAction.ACQUIRE,
            control_action=ControlAction.RETRIEVE,
            risk_level=RiskLevel.MEDIUM,
            reason="current evidence is unresolved and acquisition budget remains",
            acquisition_round=acquisition_round,
            selected_claim_ids=selected_claim_ids,
            selected_retrieval_queries=selected_queries,
            dropped_retrieval_queries=dropped_queries,
            status_counts=status_counts,
            estimated_cost_units=estimated_cost_units,
            calibration_scope=self.calibration_scope,
            metadata={
                "post_acquisition_calibration_required": True,
                "acquire_statuses": tuple(self.acquire_statuses),
                "acquire_actions": tuple(action.value for action in self.acquire_actions),
            },
        )

    def apply_to_plan(
        self,
        plan: ClaimVerificationPlan | Mapping[str, Any],
        decision: EvidenceAcquisitionDecision,
    ) -> ClaimVerificationPlan:
        """Return a plan annotated with the acquisition decision and query budget."""
        plan_obj = _plan_obj(plan)
        if plan_obj is None:
            raise ValueError("verification plan is required.")
        budget = dict(plan_obj.budget)
        budget["evidence_acquisition"] = decision.to_dict()
        retrieval_queries: Sequence[Mapping[str, Any]]
        if decision.action is EvidenceAcquisitionAction.ACQUIRE and self.max_retrieval_queries is not None:
            retrieval_queries = decision.selected_retrieval_queries
        else:
            retrieval_queries = plan_obj.retrieval_queries
        return replace(
            plan_obj,
            retrieval_queries=tuple(retrieval_queries),
            budget=budget,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready policy payload."""
        return {
            "max_acquisition_rounds": self.max_acquisition_rounds,
            "max_retrieval_queries": self.max_retrieval_queries,
            "max_estimated_cost_units": self.max_estimated_cost_units,
            "acquire_statuses": tuple(self.acquire_statuses),
            "terminal_statuses": tuple(self.terminal_statuses),
            "acquire_actions": tuple(action.value for action in self.acquire_actions),
            "budget_exhausted_action": self.budget_exhausted_action.value,
            "calibration_scope": self.calibration_scope,
        }

    def _needs_acquisition(
        self,
        decision: RiskDecision,
        *,
        status_counts: Mapping[str, int],
        plan: ClaimVerificationPlan | None,
    ) -> bool:
        if decision.action in set(self.acquire_actions):
            return True
        if any(status_counts.get(status, 0) for status in self.acquire_statuses):
            return True
        if (
            plan is not None
            and plan.run_verifier
            and bool(plan.retrieval_queries)
            and (not status_counts or "uncertainty_escalation" in plan.budget)
        ):
            return True
        return False

    def _split_retrieval_queries(
        self,
        plan: ClaimVerificationPlan | None,
    ) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
        if plan is None:
            return (), ()
        queries = tuple(dict(item) for item in plan.retrieval_queries)
        if self.max_retrieval_queries is None:
            return queries, ()
        return queries[: self.max_retrieval_queries], queries[self.max_retrieval_queries :]

    def _budget_exhaustion_reasons(
        self,
        *,
        acquisition_round: int,
        estimated_cost_units: float,
        selected_queries: Sequence[Mapping[str, Any]],
        dropped_queries: Sequence[Mapping[str, Any]],
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if acquisition_round >= self.max_acquisition_rounds:
            reasons.append("max_acquisition_rounds")
        if self.max_retrieval_queries == 0:
            reasons.append("max_retrieval_queries")
        if (
            self.max_estimated_cost_units is not None
            and estimated_cost_units > self.max_estimated_cost_units
        ):
            reasons.append("max_estimated_cost_units")
        return tuple(reasons)


def _plan_obj(plan: ClaimVerificationPlan | Mapping[str, Any] | None) -> ClaimVerificationPlan | None:
    if plan is None:
        return None
    if isinstance(plan, ClaimVerificationPlan):
        return plan
    if isinstance(plan, Mapping):
        return ClaimVerificationPlan(
            run_verifier=_strict_bool(plan.get("run_verifier", False)),
            reason=str(plan.get("reason", "")),
            verification_scope=None if plan.get("verification_scope") is None else str(plan.get("verification_scope")),
            claims=tuple(_as_sequence(plan.get("claims", ()))),
            verify_claim_ids=tuple(_as_sequence(plan.get("verify_claim_ids", ()))),
            skipped_claim_ids=tuple(_as_sequence(plan.get("skipped_claim_ids", ()))),
            triggered_claim_ids=tuple(_as_sequence(plan.get("triggered_claim_ids", ()))),
            triggered_features=dict(_mapping(plan.get("triggered_features"))),
            triggered_metadata=dict(_mapping(plan.get("triggered_metadata"))),
            route_hints=tuple(_as_sequence(plan.get("route_hints", ()))),
            retrieval_queries=tuple(_as_sequence(plan.get("retrieval_queries", ()))),
            citation_checks=tuple(_as_sequence(plan.get("citation_checks", ()))),
            calculation_checks=tuple(_as_sequence(plan.get("calculation_checks", ()))),
            state_checks=tuple(_as_sequence(plan.get("state_checks", ()))),
            world_model_checks=tuple(_as_sequence(plan.get("world_model_checks", ()))),
            dependencies=tuple(_as_sequence(plan.get("dependencies", ()))),
            budget=dict(_mapping(plan.get("budget"))),
        )
    raise ValueError("verification_plan must be a ClaimVerificationPlan, mapping, or None.")


def _status_counts(
    verification_results: Sequence[VerificationResult | Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in verification_results:
        status = _result_status(result)
        counts[status] = counts.get(status, 0) + 1
    return counts


def _result_status(result: VerificationResult | Mapping[str, Any]) -> str:
    if isinstance(result, VerificationResult):
        return result.status.value
    if isinstance(result, Mapping):
        return VerificationStatus(str(result.get("status"))).value
    raise ValueError("verification results must be VerificationResult or mapping objects.")


def _selected_claim_ids(
    plan: ClaimVerificationPlan | None,
    verification_results: Sequence[VerificationResult | Mapping[str, Any]],
) -> tuple[str, ...]:
    if plan is not None and plan.verify_claim_ids:
        return tuple(plan.verify_claim_ids)
    claim_ids: list[str] = []
    for result in verification_results:
        claim_id: str | None = None
        if isinstance(result, VerificationResult):
            claim_id = result.claim_id
        elif isinstance(result, Mapping):
            raw_claim_id = result.get("claim_id")
            claim_id = None if raw_claim_id is None else str(raw_claim_id)
        if claim_id:
            claim_ids.append(claim_id)
    return tuple(dict.fromkeys(claim_ids))


def _estimated_cost_units(plan: ClaimVerificationPlan | None) -> float:
    if plan is None:
        return 0.0
    return float(plan.cost_estimate().estimated_cost_units)


def _control_action(value: ControlAction | str) -> ControlAction:
    if isinstance(value, ControlAction):
        return value
    return ControlAction(str(value))


def _verification_status(value: VerificationStatus | str) -> VerificationStatus:
    if isinstance(value, VerificationStatus):
        return value
    return VerificationStatus(str(value))


def _jsonable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected a JSON-like mapping.")
    return dict(to_jsonable(dict(value)))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _non_empty_strings(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(text for text in (str(value).strip() for value in values) if text)


def _optional_non_negative_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, name=name)


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return number


def _optional_non_negative_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _non_negative_float(value, name=name)


def _non_negative_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative finite number.")
    number = float(value)
    if number < 0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    if number in {float("inf"), float("-inf")} or number != number:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return number


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError("expected a strict boolean value.")
