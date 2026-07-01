"""Pre-action world-model simulation gates for action execution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from eigentruth.adapters.state import StateCheck, StructuredStateVerifier
from eigentruth.adapters.world_model import WorldModelAdapter, WorldModelPrediction
from eigentruth.control.actions import (
    ActionExecutionStatus,
    ActionExecutor,
    ActionRequest,
    ActionResult,
)
from eigentruth.control.policy import ControlAction
from eigentruth.json_utils import to_jsonable
from eigentruth.verify import Claim, VerificationResult, VerificationStatus, stable_cache_key


class WorldModelActionGateStatus(str, Enum):
    """Outcome of a pre-action world-model gate."""

    PASSED = "passed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    ERROR = "error"


class WorldModelActionGateSeverity(str, Enum):
    """Severity for pre-action world-model gate issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class WorldModelActionGateIssue:
    """One issue found while simulating an action request."""

    code: str
    severity: WorldModelActionGateSeverity | str
    message: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = str(self.code).strip()
        if not code:
            raise ValueError("world-model action gate issue code must be non-empty.")
        message = str(self.message).strip()
        if not message:
            raise ValueError("world-model action gate issue message must be non-empty.")
        severity = (
            self.severity
            if isinstance(self.severity, WorldModelActionGateSeverity)
            else WorldModelActionGateSeverity(str(self.severity))
        )
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldModelActionGateIssue":
        """Build an issue from JSON-like data."""
        return cls(
            code=str(data["code"]),
            severity=str(data["severity"]),
            message=str(data["message"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class WorldModelActionGatePolicy:
    """Policy for pre-action world-model simulation gates.

    The policy is monitor-first and dependency-free. It only decides whether an
    action request is safe to dispatch to a wrapped executor; it does not mutate
    the model, retriever, or world model.
    """

    min_prediction_confidence: float = 0.0
    require_transition: bool = False
    require_postconditions: bool = False
    block_on_no_rule_match: bool = True
    block_on_low_confidence: bool = True
    block_on_low_agreement: bool = True
    block_on_postcondition_refuted: bool = True
    block_on_postcondition_insufficient_evidence: bool = True
    block_on_postcondition_error: bool = True
    include_predicted_state: bool = False
    correction_action: ControlAction | str = ControlAction.CLARIFY

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_prediction_confidence",
            _unit_interval_float(
                self.min_prediction_confidence,
                name="min_prediction_confidence",
            ),
        )
        for field_name in (
            "require_transition",
            "require_postconditions",
            "block_on_no_rule_match",
            "block_on_low_confidence",
            "block_on_low_agreement",
            "block_on_postcondition_refuted",
            "block_on_postcondition_insufficient_evidence",
            "block_on_postcondition_error",
            "include_predicted_state",
        ):
            object.__setattr__(self, field_name, _strict_bool(getattr(self, field_name), name=field_name))
        object.__setattr__(self, "correction_action", _coerce_action(self.correction_action))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready policy description."""
        return {
            "min_prediction_confidence": self.min_prediction_confidence,
            "require_transition": self.require_transition,
            "require_postconditions": self.require_postconditions,
            "block_on_no_rule_match": self.block_on_no_rule_match,
            "block_on_low_confidence": self.block_on_low_confidence,
            "block_on_low_agreement": self.block_on_low_agreement,
            "block_on_postcondition_refuted": self.block_on_postcondition_refuted,
            "block_on_postcondition_insufficient_evidence": (
                self.block_on_postcondition_insufficient_evidence
            ),
            "block_on_postcondition_error": self.block_on_postcondition_error,
            "include_predicted_state": self.include_predicted_state,
            "correction_action": self.correction_action.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldModelActionGatePolicy":
        """Build a policy from JSON-like data."""
        return cls(
            min_prediction_confidence=data.get("min_prediction_confidence", 0.0),
            require_transition=data.get("require_transition", False),
            require_postconditions=data.get("require_postconditions", False),
            block_on_no_rule_match=data.get("block_on_no_rule_match", True),
            block_on_low_confidence=data.get("block_on_low_confidence", True),
            block_on_low_agreement=data.get("block_on_low_agreement", True),
            block_on_postcondition_refuted=data.get("block_on_postcondition_refuted", True),
            block_on_postcondition_insufficient_evidence=(
                data.get("block_on_postcondition_insufficient_evidence", True)
            ),
            block_on_postcondition_error=data.get("block_on_postcondition_error", True),
            include_predicted_state=data.get("include_predicted_state", False),
            correction_action=data.get("correction_action", ControlAction.CLARIFY.value),
        )


@dataclass(frozen=True)
class WorldModelActionTransition:
    """Action-conditioned transition contract for one action request."""

    action: Mapping[str, Any]
    postconditions: Sequence[StateCheck | Mapping[str, Any]] = ()
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        postconditions = tuple(
            item if isinstance(item, StateCheck) else StateCheck.from_mapping(item)
            for item in self.postconditions
        )
        object.__setattr__(self, "action", dict(self.action))
        object.__setattr__(self, "postconditions", postconditions)
        object.__setattr__(self, "source", None if self.source is None else str(self.source))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "WorldModelActionTransition":
        """Build a transition contract from JSON-like data."""
        raw_action = data.get("action", data.get("world_model_action"))
        if not isinstance(raw_action, Mapping):
            raise ValueError("world-model action transition must contain action object.")
        raw_postconditions = data.get(
            "postconditions",
            data.get("postcondition", data.get("state_check", data.get("check", ()))),
        )
        if raw_postconditions is None:
            raw_postconditions = ()
        if isinstance(raw_postconditions, (Mapping, StateCheck)):
            raw_postconditions = (raw_postconditions,)
        if not isinstance(raw_postconditions, Sequence) or isinstance(raw_postconditions, (str, bytes)):
            raise ValueError("world-model action transition postconditions must be an object or sequence.")
        return cls(
            action=dict(raw_action),
            postconditions=tuple(raw_postconditions),
            source=None if data.get("source") is None else str(data.get("source")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready transition summary."""
        return {
            "action": to_jsonable(dict(self.action)),
            "postconditions": tuple(_state_check_to_dict(item) for item in self.postconditions),
            "source": self.source,
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class WorldModelActionGateReport:
    """JSON-ready report for one pre-action world-model gate."""

    status: WorldModelActionGateStatus | str
    passed: bool
    available: bool
    action: ControlAction | str
    request_id: str | None = None
    decision_rule: str = ""
    issues: Sequence[WorldModelActionGateIssue | Mapping[str, Any]] = ()
    transition: WorldModelActionTransition | Mapping[str, Any] | None = None
    prediction_confidence: float | None = None
    min_prediction_confidence: float | None = None
    base_state_fingerprint: str | None = None
    predicted_state_fingerprint: str | None = None
    prediction_metadata: Mapping[str, Any] = field(default_factory=dict)
    postcondition_results: Sequence[VerificationResult | Mapping[str, Any]] = ()
    predicted_state: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, WorldModelActionGateStatus)
            else WorldModelActionGateStatus(str(self.status))
        )
        action = _coerce_action(self.action)
        transition = self.transition
        if transition is not None and not isinstance(transition, WorldModelActionTransition):
            transition = WorldModelActionTransition.from_mapping(transition)
        issues = tuple(_coerce_issue(issue) for issue in self.issues)
        postcondition_results = tuple(_verification_result_to_dict(result) for result in self.postcondition_results)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "passed", _strict_bool(self.passed, name="passed"))
        object.__setattr__(self, "available", _strict_bool(self.available, name="available"))
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "request_id", None if self.request_id is None else str(self.request_id))
        object.__setattr__(self, "decision_rule", str(self.decision_rule))
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "transition", transition)
        object.__setattr__(
            self,
            "prediction_confidence",
            None
            if self.prediction_confidence is None
            else _unit_interval_float(self.prediction_confidence, name="prediction_confidence"),
        )
        object.__setattr__(
            self,
            "min_prediction_confidence",
            None
            if self.min_prediction_confidence is None
            else _unit_interval_float(self.min_prediction_confidence, name="min_prediction_confidence"),
        )
        object.__setattr__(self, "prediction_metadata", dict(self.prediction_metadata))
        object.__setattr__(self, "postcondition_results", postcondition_results)
        object.__setattr__(
            self,
            "predicted_state",
            None if self.predicted_state is None else dict(self.predicted_state),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def blocked(self) -> bool:
        """Return whether this gate blocked the action request."""
        return self.status in {WorldModelActionGateStatus.BLOCKED, WorldModelActionGateStatus.ERROR}

    def summary(self) -> dict[str, Any]:
        """Return compact telemetry for traces and action-result metadata."""
        counts_by_code: dict[str, int] = {}
        counts_by_severity: dict[str, int] = {}
        for issue in self.issues:
            counts_by_code[issue.code] = counts_by_code.get(issue.code, 0) + 1
            severity = issue.severity.value
            counts_by_severity[severity] = counts_by_severity.get(severity, 0) + 1
        return {
            "available": self.available,
            "passed": self.passed,
            "blocked": self.blocked,
            "status": self.status.value,
            "decision_rule": self.decision_rule,
            "action": self.action.value,
            "request_id": self.request_id,
            "issue_count": len(self.issues),
            "error_count": counts_by_severity.get(WorldModelActionGateSeverity.ERROR.value, 0),
            "warning_count": counts_by_severity.get(WorldModelActionGateSeverity.WARNING.value, 0),
            "counts_by_code": counts_by_code,
            "prediction_confidence": self.prediction_confidence,
            "min_prediction_confidence": self.min_prediction_confidence,
            "base_state_fingerprint": self.base_state_fingerprint,
            "predicted_state_fingerprint": self.predicted_state_fingerprint,
            "postcondition_count": len(self.postcondition_results),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload: dict[str, Any] = {
            "status": self.status.value,
            "passed": self.passed,
            "available": self.available,
            "action": self.action.value,
            "request_id": self.request_id,
            "decision_rule": self.decision_rule,
            "issues": tuple(issue.to_dict() for issue in self.issues),
            "transition": None if self.transition is None else self.transition.to_dict(),
            "prediction_confidence": self.prediction_confidence,
            "min_prediction_confidence": self.min_prediction_confidence,
            "base_state_fingerprint": self.base_state_fingerprint,
            "predicted_state_fingerprint": self.predicted_state_fingerprint,
            "prediction_metadata": to_jsonable(dict(self.prediction_metadata)),
            "postcondition_results": tuple(self.postcondition_results),
            "metadata": to_jsonable(dict(self.metadata)),
            "summary": self.summary(),
        }
        if self.predicted_state is not None:
            payload["predicted_state"] = to_jsonable(dict(self.predicted_state))
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldModelActionGateReport":
        """Build a report from JSON-like data."""
        return cls(
            status=str(data["status"]),
            passed=data["passed"],
            available=data["available"],
            action=str(data["action"]),
            request_id=None if data.get("request_id") is None else str(data.get("request_id")),
            decision_rule=str(data.get("decision_rule", "")),
            issues=tuple(_as_sequence(data.get("issues", ()))),
            transition=data.get("transition"),
            prediction_confidence=data.get("prediction_confidence"),
            min_prediction_confidence=data.get("min_prediction_confidence"),
            base_state_fingerprint=data.get("base_state_fingerprint"),
            predicted_state_fingerprint=data.get("predicted_state_fingerprint"),
            prediction_metadata=dict(data.get("prediction_metadata", {})),
            postcondition_results=tuple(_as_sequence(data.get("postcondition_results", ()))),
            predicted_state=data.get("predicted_state"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class WorldModelGuardedActionExecutor:
    """Wrap an action executor with a pre-action world-model gate."""

    executor: ActionExecutor
    world_model: WorldModelAdapter
    state: Mapping[str, Any] = field(default_factory=dict)
    policy: WorldModelActionGatePolicy | Mapping[str, Any] = field(default_factory=WorldModelActionGatePolicy)

    def __post_init__(self) -> None:
        policy = self.policy
        if not isinstance(policy, WorldModelActionGatePolicy):
            policy = WorldModelActionGatePolicy.from_dict(policy)
        object.__setattr__(self, "state", dict(self.state))
        object.__setattr__(self, "policy", policy)

    def execute(
        self,
        request: ActionRequest,
        context: Mapping[str, Any] | None = None,
    ) -> ActionResult:
        """Execute one request only after the world-model gate passes."""
        report = audit_world_model_action_gate(
            request=request,
            world_model=self.world_model,
            state=self.state,
            policy=self.policy,
            context=context,
        )
        if report.blocked:
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.SKIPPED,
                output={
                    "world_model_gate": report.to_dict(),
                    "blocked_request": request.to_dict(),
                    "correction_request": _correction_request(request, report, self.policy).to_dict(),
                },
                metadata={
                    "executor": type(self).__name__,
                    "wrapped_executor": type(self.executor).__name__,
                    "world_model_gate": report.summary(),
                    "side_effects": False,
                    "side_effect_status": "blocked_before_execution",
                    "possible_side_effects": False,
                    "context": dict(context or {}),
                },
                request_id=request.request_id,
                error=f"world-model action gate blocked request: {report.decision_rule}",
            )

        try:
            result = self.executor.execute(request, context=context)
        except Exception as exc:  # pragma: no cover - defensive executor boundary
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.FAILED,
                output={"world_model_gate": report.to_dict()},
                metadata={
                    "executor": type(self).__name__,
                    "wrapped_executor": type(self.executor).__name__,
                    "world_model_gate": report.summary(),
                    "side_effects": None,
                    "side_effect_status": "unknown_after_failure",
                    "possible_side_effects": True,
                    "context": dict(context or {}),
                },
                request_id=request.request_id,
                error=f"wrapped executor failed after world-model gate passed: {exc}",
            )
        metadata = dict(result.metadata)
        metadata.update({
            "world_model_gate": report.summary(),
            "world_model_guard": type(self).__name__,
            "wrapped_executor": metadata.get("executor", type(self.executor).__name__),
        })
        return ActionResult(
            action=result.action,
            status=result.status,
            output=result.output,
            metadata=metadata,
            request_id=result.request_id,
            error=result.error,
        )

    def execute_many(
        self,
        requests: Sequence[ActionRequest],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[ActionResult, ...]:
        """Execute multiple requests with the same world-model gate."""
        return tuple(self.execute(request, context=context) for request in requests)


def audit_world_model_action_gate(
    *,
    request: ActionRequest,
    world_model: WorldModelAdapter,
    state: Mapping[str, Any] | None = None,
    policy: WorldModelActionGatePolicy | Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> WorldModelActionGateReport:
    """Simulate one action request and decide whether it should dispatch."""
    resolved_policy = _coerce_policy(policy)
    try:
        transition = _resolve_transition(request, context)
    except (TypeError, ValueError) as exc:
        return _blocked_report(
            request,
            policy=resolved_policy,
            decision_rule="invalid_transition",
            issue=WorldModelActionGateIssue(
                code="invalid_transition",
                severity=WorldModelActionGateSeverity.ERROR,
                message=str(exc),
            ),
        )
    if transition is None:
        if resolved_policy.require_transition:
            return _blocked_report(
                request,
                policy=resolved_policy,
                decision_rule="missing_transition",
                issue=WorldModelActionGateIssue(
                    code="missing_transition",
                    severity=WorldModelActionGateSeverity.ERROR,
                    message="action request did not include a world-model transition contract.",
                ),
            )
        return WorldModelActionGateReport(
            status=WorldModelActionGateStatus.SKIPPED,
            passed=True,
            available=False,
            action=request.action,
            request_id=request.request_id,
            decision_rule="no_transition_available",
            min_prediction_confidence=resolved_policy.min_prediction_confidence,
            metadata={"policy": resolved_policy.to_dict()},
        )
    if resolved_policy.require_postconditions and not transition.postconditions:
        return _blocked_report(
            request,
            policy=resolved_policy,
            decision_rule="missing_postconditions",
            issue=WorldModelActionGateIssue(
                code="missing_postconditions",
                severity=WorldModelActionGateSeverity.ERROR,
                message="world-model transition did not include required postconditions.",
            ),
            transition=transition,
        )

    base_state = _merged_state(state or {}, context)
    try:
        prediction = world_model.predict(base_state, transition.action)
    except Exception as exc:  # pragma: no cover - defensive adapter boundary
        return _blocked_report(
            request,
            policy=resolved_policy,
            decision_rule="prediction_error",
            issue=WorldModelActionGateIssue(
                code="prediction_error",
                severity=WorldModelActionGateSeverity.ERROR,
                message=f"world-model prediction failed: {exc}",
            ),
            transition=transition,
            base_state_fingerprint=stable_cache_key(base_state),
        )

    postcondition_results = _postcondition_results(request, transition, prediction)
    issues = list(_prediction_issues(prediction, resolved_policy))
    issues.extend(_postcondition_issues(postcondition_results, resolved_policy))
    failed = any(issue.severity is WorldModelActionGateSeverity.ERROR for issue in issues)
    status = WorldModelActionGateStatus.BLOCKED if failed else WorldModelActionGateStatus.PASSED
    decision_rule = _decision_rule(status, issues)
    return WorldModelActionGateReport(
        status=status,
        passed=not failed,
        available=True,
        action=request.action,
        request_id=request.request_id,
        decision_rule=decision_rule,
        issues=tuple(issues),
        transition=transition,
        prediction_confidence=prediction.confidence,
        min_prediction_confidence=resolved_policy.min_prediction_confidence,
        base_state_fingerprint=stable_cache_key(base_state),
        predicted_state_fingerprint=stable_cache_key(prediction.state),
        prediction_metadata=dict(prediction.metadata),
        postcondition_results=postcondition_results,
        predicted_state=prediction.state if resolved_policy.include_predicted_state else None,
        metadata={"policy": resolved_policy.to_dict(), "prediction_explanation": prediction.explanation},
    )


def _resolve_transition(
    request: ActionRequest,
    context: Mapping[str, Any] | None,
) -> WorldModelActionTransition | None:
    for candidate in _transition_candidates(request, context):
        if candidate is None:
            continue
        if isinstance(candidate, WorldModelActionTransition):
            return candidate
        if isinstance(candidate, Mapping):
            return WorldModelActionTransition.from_mapping(candidate)
        raise ValueError("world-model transition candidate must be a mapping.")
    return None


def _transition_candidates(
    request: ActionRequest,
    context: Mapping[str, Any] | None,
) -> tuple[Any, ...]:
    payload = dict(request.payload)
    metadata = dict(request.metadata)
    context_payload = dict(context or {})
    nested_context = context_payload.get("world_model_gate")
    nested_payload = payload.get("world_model_gate")
    nested_metadata = metadata.get("world_model_gate")
    by_request_id = ()
    if request.request_id is not None and isinstance(nested_context, Mapping):
        transitions_by_request_id = nested_context.get("transitions_by_request_id")
        if isinstance(transitions_by_request_id, Mapping):
            by_request_id = (transitions_by_request_id.get(request.request_id),)
    return (
        *by_request_id,
        _nested_transition(nested_metadata),
        metadata.get("world_model_transition"),
        metadata.get("state_transition"),
        _nested_transition(nested_payload),
        payload.get("world_model_transition"),
        payload.get("state_transition"),
        _nested_transition(nested_context),
        context_payload.get("world_model_transition"),
        context_payload.get("state_transition"),
    )


def _nested_transition(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("transition", value.get("state_transition", value.get("world_model_transition")))
    return None


def _postcondition_results(
    request: ActionRequest,
    transition: WorldModelActionTransition,
    prediction: WorldModelPrediction,
) -> tuple[VerificationResult, ...]:
    verifier = StructuredStateVerifier(state=prediction.state)
    results: list[VerificationResult] = []
    for index, postcondition in enumerate(transition.postconditions):
        result = verifier.verify(
            Claim(
                text=f"world-model pre-action postcondition {index + 1} for {request.action.value}",
                claim_id=request.request_id,
                metadata={"state_check": postcondition},
            )
        )
        results.append(result)
    return tuple(results)


def _prediction_issues(
    prediction: WorldModelPrediction,
    policy: WorldModelActionGatePolicy,
) -> tuple[WorldModelActionGateIssue, ...]:
    issues: list[WorldModelActionGateIssue] = []
    if prediction.metadata.get("no_rule_matched") is True and policy.block_on_no_rule_match:
        issues.append(WorldModelActionGateIssue(
            code="no_rule_matched",
            severity=WorldModelActionGateSeverity.ERROR,
            message="world model found no matching transition rule for the action.",
            metadata={"prediction_metadata": dict(prediction.metadata)},
        ))
    if prediction.metadata.get("below_min_agreement") is True and policy.block_on_low_agreement:
        issues.append(WorldModelActionGateIssue(
            code="low_agreement",
            severity=WorldModelActionGateSeverity.ERROR,
            message="world-model ensemble agreement is below the configured threshold.",
            metadata={"prediction_metadata": dict(prediction.metadata)},
        ))
    if prediction.confidence < policy.min_prediction_confidence and policy.block_on_low_confidence:
        issues.append(WorldModelActionGateIssue(
            code="low_prediction_confidence",
            severity=WorldModelActionGateSeverity.ERROR,
            message="world-model prediction confidence is below the configured threshold.",
            metadata={
                "prediction_confidence": prediction.confidence,
                "min_prediction_confidence": policy.min_prediction_confidence,
            },
        ))
    return tuple(issues)


def _postcondition_issues(
    postcondition_results: Sequence[VerificationResult],
    policy: WorldModelActionGatePolicy,
) -> tuple[WorldModelActionGateIssue, ...]:
    issues: list[WorldModelActionGateIssue] = []
    for index, result in enumerate(postcondition_results):
        metadata = {
            "postcondition_index": index,
            "status": result.status.value,
            "confidence": result.confidence,
            "decision_rule": result.metadata.get("decision_rule"),
            "path": result.metadata.get("path"),
            "operator": result.metadata.get("operator"),
            "expected": result.metadata.get("expected"),
            "actual": result.metadata.get("actual"),
        }
        if result.status is VerificationStatus.REFUTED and policy.block_on_postcondition_refuted:
            issues.append(WorldModelActionGateIssue(
                code="postcondition_refuted",
                severity=WorldModelActionGateSeverity.ERROR,
                message="predicted state refutes a required postcondition.",
                metadata=metadata,
            ))
        elif (
            result.status is VerificationStatus.INSUFFICIENT_EVIDENCE
            and policy.block_on_postcondition_insufficient_evidence
        ):
            issues.append(WorldModelActionGateIssue(
                code="postcondition_insufficient_evidence",
                severity=WorldModelActionGateSeverity.ERROR,
                message="predicted state lacks evidence for a required postcondition.",
                metadata=metadata,
            ))
        elif result.status is VerificationStatus.ERROR and policy.block_on_postcondition_error:
            issues.append(WorldModelActionGateIssue(
                code="postcondition_error",
                severity=WorldModelActionGateSeverity.ERROR,
                message="postcondition evaluation failed on the predicted state.",
                metadata=metadata,
            ))
    return tuple(issues)


def _decision_rule(
    status: WorldModelActionGateStatus,
    issues: Sequence[WorldModelActionGateIssue],
) -> str:
    if status is WorldModelActionGateStatus.PASSED:
        return "prediction_and_postconditions_passed"
    if issues:
        return "blocked_" + issues[0].code
    return "blocked_unknown"


def _blocked_report(
    request: ActionRequest,
    *,
    policy: WorldModelActionGatePolicy,
    decision_rule: str,
    issue: WorldModelActionGateIssue,
    transition: WorldModelActionTransition | None = None,
    base_state_fingerprint: str | None = None,
) -> WorldModelActionGateReport:
    return WorldModelActionGateReport(
        status=WorldModelActionGateStatus.BLOCKED,
        passed=False,
        available=transition is not None,
        action=request.action,
        request_id=request.request_id,
        decision_rule=decision_rule,
        issues=(issue,),
        transition=transition,
        min_prediction_confidence=policy.min_prediction_confidence,
        base_state_fingerprint=base_state_fingerprint,
        metadata={"policy": policy.to_dict()},
    )


def _correction_request(
    request: ActionRequest,
    report: WorldModelActionGateReport,
    policy: WorldModelActionGatePolicy,
) -> ActionRequest:
    return ActionRequest(
        action=policy.correction_action,
        reason=f"world-model action gate blocked {request.action.value}: {report.decision_rule}",
        payload={
            "blocked_action": request.action.value,
            "blocked_request_id": request.request_id,
            "world_model_gate": report.summary(),
            "instruction": "revise, clarify, or abstain before executing the blocked action",
        },
        metadata={
            "policy": type(policy).__name__,
            "source_executor": "WorldModelGuardedActionExecutor",
        },
        request_id=None if request.request_id is None else f"{request.request_id}:world-model-correction",
    )


def _merged_state(
    base_state: Mapping[str, Any],
    context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = _deep_copy_mapping(base_state)
    if context is None or not isinstance(context.get("state"), Mapping):
        return merged
    _deep_merge(merged, context["state"])
    return merged


def _deep_merge(target: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            nested = _deep_copy_mapping(target[key])
            _deep_merge(nested, value)
            target[str(key)] = nested
        elif isinstance(value, Mapping):
            target[str(key)] = _deep_copy_mapping(value)
        else:
            target[str(key)] = value


def _deep_copy_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            copied[str(key)] = _deep_copy_mapping(value)
        elif isinstance(value, list):
            copied[str(key)] = [
                _deep_copy_mapping(item) if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            copied[str(key)] = value
    return copied


def _state_check_to_dict(check: StateCheck) -> dict[str, Any]:
    return {
        "path": check.path,
        "operator": check.operator,
        "value": to_jsonable(check.value),
        "source": check.source,
        "metadata": to_jsonable(dict(check.metadata)),
    }


def _verification_result_to_dict(result: VerificationResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, VerificationResult):
        return {
            "status": result.status.value,
            "confidence": result.confidence,
            "evidence": tuple(result.evidence),
            "explanation": result.explanation,
            "metadata": to_jsonable(dict(result.metadata)),
        }
    return dict(to_jsonable(dict(result)))


def _coerce_policy(
    policy: WorldModelActionGatePolicy | Mapping[str, Any] | None,
) -> WorldModelActionGatePolicy:
    if policy is None:
        return WorldModelActionGatePolicy()
    if isinstance(policy, WorldModelActionGatePolicy):
        return policy
    return WorldModelActionGatePolicy.from_dict(policy)


def _coerce_issue(issue: WorldModelActionGateIssue | Mapping[str, Any]) -> WorldModelActionGateIssue:
    if isinstance(issue, WorldModelActionGateIssue):
        return issue
    return WorldModelActionGateIssue.from_dict(issue)


def _coerce_action(action: ControlAction | str) -> ControlAction:
    return action if isinstance(action, ControlAction) else ControlAction(str(action))


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return (value,)


def _unit_interval_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite numeric value.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return numeric


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a bool or strict bool string.")
