"""Audit planned control actions before executor dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from eigentruth.control.actions import ActionRequest
from eigentruth.control.policy import ControlAction, RiskDecision
from eigentruth.json_utils import to_jsonable
from eigentruth.verify.planning import ClaimVerificationPlan


class ActionAuditSeverity(str, Enum):
    """Severity levels for action-planning audit findings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ActionAuditIssue:
    """One action-planning issue detected before execution."""

    code: str
    severity: ActionAuditSeverity | str
    message: str
    action_index: int | None = None
    action: str | None = None
    claim_ids: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = str(self.code).strip()
        if not code:
            raise ValueError("action audit issue code must be non-empty.")
        message = str(self.message).strip()
        if not message:
            raise ValueError("action audit issue message must be non-empty.")
        severity = (
            self.severity
            if isinstance(self.severity, ActionAuditSeverity)
            else ActionAuditSeverity(str(self.severity))
        )
        action_index = None if self.action_index is None else int(self.action_index)
        action = None if self.action is None else str(self.action)
        claim_ids = tuple(str(item) for item in self.claim_ids if str(item).strip())
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "action_index", action_index)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "claim_ids", claim_ids)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "action_index": self.action_index,
            "action": self.action,
            "claim_ids": tuple(self.claim_ids),
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionAuditIssue":
        """Build an issue from JSON-like data."""
        return cls(
            code=str(data["code"]),
            severity=str(data["severity"]),
            message=str(data["message"]),
            action_index=None if data.get("action_index") is None else int(data["action_index"]),
            action=None if data.get("action") is None else str(data["action"]),
            claim_ids=tuple(_as_sequence(data.get("claim_ids", ()))),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ActionAuditReport:
    """JSON-ready report for planned action/tool-selection audit."""

    action_count: int
    decision_action: str | None = None
    required_actions: Sequence[str] = ()
    issues: Sequence[ActionAuditIssue | Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action_count = int(self.action_count)
        if action_count < 0:
            raise ValueError("action_count must be non-negative.")
        issues = tuple(_coerce_issue(issue) for issue in self.issues)
        required_actions = tuple(_normalize_action_name(item) for item in self.required_actions)
        decision_action = None if self.decision_action is None else _normalize_action_name(self.decision_action)
        object.__setattr__(self, "action_count", action_count)
        object.__setattr__(self, "decision_action", decision_action)
        object.__setattr__(self, "required_actions", required_actions)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def passed(self) -> bool:
        """Return whether the report has no error-level issues."""
        return not any(issue.severity is ActionAuditSeverity.ERROR for issue in self.issues)

    def summary(self) -> dict[str, Any]:
        """Return a compact telemetry summary."""
        counts_by_severity: dict[str, int] = {}
        counts_by_code: dict[str, int] = {}
        for issue in self.issues:
            severity = issue.severity.value
            counts_by_severity[severity] = counts_by_severity.get(severity, 0) + 1
            counts_by_code[issue.code] = counts_by_code.get(issue.code, 0) + 1
        return {
            "available": True,
            "passed": self.passed,
            "action_count": self.action_count,
            "decision_action": self.decision_action,
            "required_actions": tuple(self.required_actions),
            "issue_count": len(self.issues),
            "error_count": counts_by_severity.get(ActionAuditSeverity.ERROR.value, 0),
            "warning_count": counts_by_severity.get(ActionAuditSeverity.WARNING.value, 0),
            "info_count": counts_by_severity.get(ActionAuditSeverity.INFO.value, 0),
            "counts_by_severity": counts_by_severity,
            "counts_by_code": counts_by_code,
            "top_issues": tuple(issue.to_dict() for issue in self.issues[:8]),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "action_count": self.action_count,
            "decision_action": self.decision_action,
            "required_actions": tuple(self.required_actions),
            "issues": tuple(issue.to_dict() for issue in self.issues),
            "metadata": to_jsonable(dict(self.metadata)),
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionAuditReport":
        """Build a report from JSON-like data."""
        return cls(
            action_count=int(data["action_count"]),
            decision_action=None if data.get("decision_action") is None else str(data["decision_action"]),
            required_actions=tuple(_as_sequence(data.get("required_actions", ()))),
            issues=tuple(_as_sequence(data.get("issues", ()))),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ActionAuditPolicy:
    """Configurable, dependency-free action/tool-selection audit policy."""

    require_decision_action: bool = True
    require_plan_retrieval: bool = True
    validate_claim_ids: bool = True
    retrieval_required_decision_actions: Sequence[ControlAction | str] = (
        ControlAction.ACCEPT,
        ControlAction.RETRIEVE,
        ControlAction.REWRITE,
        ControlAction.CLARIFY,
        ControlAction.EXECUTE_TOOL,
    )
    allowed_actions_by_decision: Mapping[ControlAction | str, Sequence[ControlAction | str]] = field(
        default_factory=lambda: {
            ControlAction.ACCEPT: (ControlAction.ACCEPT, ControlAction.RETRIEVE),
            ControlAction.RETRIEVE: (ControlAction.RETRIEVE,),
            ControlAction.REWRITE: (ControlAction.REWRITE, ControlAction.RETRIEVE),
            ControlAction.STEER_REGENERATE: (ControlAction.STEER_REGENERATE, ControlAction.RETRIEVE),
            ControlAction.EXECUTE_TOOL: (ControlAction.EXECUTE_TOOL, ControlAction.RETRIEVE),
            ControlAction.ABSTAIN: (ControlAction.ABSTAIN,),
            ControlAction.CLARIFY: (ControlAction.CLARIFY, ControlAction.RETRIEVE),
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "require_decision_action", _strict_bool(self.require_decision_action))
        object.__setattr__(self, "require_plan_retrieval", _strict_bool(self.require_plan_retrieval))
        object.__setattr__(self, "validate_claim_ids", _strict_bool(self.validate_claim_ids))
        object.__setattr__(
            self,
            "retrieval_required_decision_actions",
            tuple(_normalize_action_name(item) for item in self.retrieval_required_decision_actions),
        )
        object.__setattr__(
            self,
            "allowed_actions_by_decision",
            {
                _normalize_action_name(action): tuple(_normalize_action_name(item) for item in actions)
                for action, actions in self.allowed_actions_by_decision.items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready policy description."""
        return {
            "require_decision_action": self.require_decision_action,
            "require_plan_retrieval": self.require_plan_retrieval,
            "validate_claim_ids": self.validate_claim_ids,
            "retrieval_required_decision_actions": tuple(self.retrieval_required_decision_actions),
            "allowed_actions_by_decision": {
                str(action): tuple(actions)
                for action, actions in self.allowed_actions_by_decision.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionAuditPolicy":
        """Build a policy from JSON-like data."""
        allowed_actions = data.get("allowed_actions_by_decision")
        if not isinstance(allowed_actions, Mapping):
            allowed_actions = ActionAuditPolicy().allowed_actions_by_decision
        return cls(
            require_decision_action=_strict_bool(data.get("require_decision_action", True)),
            require_plan_retrieval=_strict_bool(data.get("require_plan_retrieval", True)),
            validate_claim_ids=_strict_bool(data.get("validate_claim_ids", True)),
            retrieval_required_decision_actions=tuple(
                _as_sequence(data.get("retrieval_required_decision_actions", ()))
            )
            or (
                ControlAction.ACCEPT,
                ControlAction.RETRIEVE,
                ControlAction.REWRITE,
                ControlAction.CLARIFY,
                ControlAction.EXECUTE_TOOL,
            ),
            allowed_actions_by_decision=allowed_actions,
        )


def audit_action_requests(
    actions: Sequence[ActionRequest | ControlAction | str | Mapping[str, Any]],
    *,
    decision: RiskDecision | ControlAction | str | Mapping[str, Any] | None = None,
    verification_plan: ClaimVerificationPlan | Mapping[str, Any] | None = None,
    policy: ActionAuditPolicy | Mapping[str, Any] | None = None,
) -> ActionAuditReport:
    """Audit planned actions against the selected risk decision and verifier plan.

    The audit is monitor-first: it produces structured issues for trace,
    registry, and release checks, but does not execute or block actions.
    """
    resolved_policy = _coerce_policy(policy)
    issues: list[ActionAuditIssue] = []
    action_payloads = []
    for index, action in enumerate(actions):
        try:
            payload = _action_payload(action)
        except ValueError as exc:
            issues.append(ActionAuditIssue(
                code="invalid_action_request",
                severity=ActionAuditSeverity.ERROR,
                message=str(exc),
                action_index=index,
            ))
            continue
        action_payloads.append(payload)
        if not isinstance(payload["payload"], Mapping):
            issues.append(ActionAuditIssue(
                code="malformed_action_payload",
                severity=ActionAuditSeverity.ERROR,
                message="action payload must be a JSON object",
                action_index=index,
                action=payload["action"],
                metadata={"payload_type": type(payload["payload"]).__name__},
            ))

    action_names = tuple(str(item["action"]) for item in action_payloads)
    action_counts = _counts(action_names)
    decision_action = _decision_action_name(decision)
    plan = _plan_payload(verification_plan)
    required_actions: list[str] = []

    if decision_action is not None:
        allowed_actions = resolved_policy.allowed_actions_by_decision.get(decision_action)
        if allowed_actions:
            for index, payload in enumerate(action_payloads):
                action_name = str(payload["action"])
                if action_name not in allowed_actions:
                    issues.append(ActionAuditIssue(
                        code="unexpected_action_for_decision",
                        severity=ActionAuditSeverity.WARNING,
                        message=f"planned action {action_name!r} is not expected for decision {decision_action!r}",
                        action_index=index,
                        action=action_name,
                        metadata={"allowed_actions": allowed_actions},
                    ))
        if resolved_policy.require_decision_action:
            required_actions.append(decision_action)
            if action_counts.get(decision_action, 0) == 0:
                issues.append(ActionAuditIssue(
                    code="missing_decision_action",
                    severity=ActionAuditSeverity.ERROR,
                    message=f"risk decision selected {decision_action!r}, but no matching action was planned",
                    action=decision_action,
                ))

    plan_retrieval_query_count = _plan_retrieval_query_count(plan)
    if (
        resolved_policy.require_plan_retrieval
        and plan_retrieval_query_count
        and _decision_allows_plan_retrieval(decision_action, resolved_policy)
    ):
        required_actions.append(ControlAction.RETRIEVE.value)
        if action_counts.get(ControlAction.RETRIEVE.value, 0) == 0:
            issues.append(ActionAuditIssue(
                code="missing_retrieval_action",
                severity=ActionAuditSeverity.ERROR,
                message="verification plan emitted retrieval queries, but no retrieve action was planned",
                action=ControlAction.RETRIEVE.value,
                metadata={"plan_retrieval_query_count": plan_retrieval_query_count},
            ))

    known_claim_ids = _known_plan_claim_ids(plan)
    for index, payload in enumerate(action_payloads):
        action_name = str(payload["action"])
        action_body = _mapping(payload["payload"])
        if action_name == ControlAction.RETRIEVE.value:
            issues.extend(_audit_retrieval_payload(index, action_body))
        elif action_name == ControlAction.EXECUTE_TOOL.value:
            issues.extend(_audit_tool_payload(index, action_body))
        if resolved_policy.validate_claim_ids and known_claim_ids:
            action_claim_ids = _action_claim_ids(action_body)
            unknown_claim_ids = tuple(claim_id for claim_id in action_claim_ids if claim_id not in known_claim_ids)
            if unknown_claim_ids:
                issues.append(ActionAuditIssue(
                    code="unknown_claim_id",
                    severity=ActionAuditSeverity.WARNING,
                    message="action payload referenced claim ids that were not present in the verification plan",
                    action_index=index,
                    action=action_name,
                    claim_ids=unknown_claim_ids,
                    metadata={"known_claim_count": len(known_claim_ids)},
                ))

    required_actions = tuple(dict.fromkeys(required_actions))
    return ActionAuditReport(
        action_count=len(action_payloads),
        decision_action=decision_action,
        required_actions=required_actions,
        issues=tuple(issues),
        metadata={
            "action_counts": action_counts,
            "plan_retrieval_query_count": plan_retrieval_query_count,
            "known_claim_count": len(known_claim_ids),
            "policy": resolved_policy.to_dict(),
        },
    )


def _audit_retrieval_payload(
    action_index: int,
    payload: Mapping[str, Any],
) -> tuple[ActionAuditIssue, ...]:
    issues = []
    executable_query_count = _executable_retrieval_query_count(payload)
    advisory_query_count = _advisory_retrieval_query_count(payload)
    if executable_query_count == 0:
        issues.append(ActionAuditIssue(
            code="malformed_retrieval_payload",
            severity=ActionAuditSeverity.ERROR,
            message="retrieve action has no executable query or retrieval target",
            action_index=action_index,
            action=ControlAction.RETRIEVE.value,
            metadata={"advisory_query_count": advisory_query_count},
        ))
    if advisory_query_count and executable_query_count == 0:
        issues.append(ActionAuditIssue(
            code="retrieval_queries_not_executable",
            severity=ActionAuditSeverity.WARNING,
            message="retrieval_queries were present, but the local retrieval executor needs query or retrieval_targets",
            action_index=action_index,
            action=ControlAction.RETRIEVE.value,
            metadata={"advisory_query_count": advisory_query_count},
        ))
    return tuple(issues)


def _audit_tool_payload(
    action_index: int,
    payload: Mapping[str, Any],
) -> tuple[ActionAuditIssue, ...]:
    issues = []
    tool_name = _first_non_empty_string(payload.get("tool_name"), payload.get("tool"))
    if tool_name is None:
        issues.append(ActionAuditIssue(
            code="malformed_tool_payload",
            severity=ActionAuditSeverity.ERROR,
            message="execute_tool action must include a non-empty tool or tool_name",
            action_index=action_index,
            action=ControlAction.EXECUTE_TOOL.value,
        ))
    argument_keys = ("tool_args", "arguments", "input")
    present_argument_keys = tuple(key for key in argument_keys if key in payload)
    if not present_argument_keys:
        issues.append(ActionAuditIssue(
            code="missing_tool_arguments",
            severity=ActionAuditSeverity.WARNING,
            message="execute_tool action has no tool_args, arguments, or input payload",
            action_index=action_index,
            action=ControlAction.EXECUTE_TOOL.value,
            metadata={"tool": tool_name},
        ))
    for key in present_argument_keys:
        value = payload.get(key)
        if value is not None and not isinstance(value, Mapping):
            issues.append(ActionAuditIssue(
                code="malformed_tool_arguments",
                severity=ActionAuditSeverity.ERROR,
                message=f"execute_tool payload field {key!r} must be a JSON object when present",
                action_index=action_index,
                action=ControlAction.EXECUTE_TOOL.value,
                metadata={"field": key, "value_type": type(value).__name__},
            ))
    return tuple(issues)


def _coerce_policy(policy: ActionAuditPolicy | Mapping[str, Any] | None) -> ActionAuditPolicy:
    if policy is None:
        return ActionAuditPolicy()
    if isinstance(policy, ActionAuditPolicy):
        return policy
    return ActionAuditPolicy.from_dict(policy)


def _coerce_issue(issue: ActionAuditIssue | Mapping[str, Any]) -> ActionAuditIssue:
    if isinstance(issue, ActionAuditIssue):
        return issue
    return ActionAuditIssue.from_dict(issue)


def _action_payload(action: ActionRequest | ControlAction | str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(action, ActionRequest):
        return {
            "action": action.action.value,
            "reason": action.reason,
            "payload": dict(action.payload),
            "metadata": dict(action.metadata),
            "request_id": action.request_id,
        }
    if isinstance(action, ControlAction):
        return {"action": action.value, "reason": "", "payload": {}, "metadata": {}, "request_id": None}
    if isinstance(action, str):
        return {
            "action": _normalize_action_name(action),
            "reason": "",
            "payload": {},
            "metadata": {},
            "request_id": None,
        }
    if not isinstance(action, Mapping):
        raise ValueError(f"action request must be an action, string, or JSON object, got {type(action).__name__}.")
    if "action" not in action:
        raise ValueError("action request JSON object must include an action field.")
    payload = action.get("payload", {})
    metadata = action.get("metadata", {})
    return {
        "action": _normalize_action_name(action["action"]),
        "reason": str(action.get("reason", "")),
        "payload": payload if isinstance(payload, Mapping) else payload,
        "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
        "request_id": action.get("request_id"),
    }


def _decision_action_name(decision: RiskDecision | ControlAction | str | Mapping[str, Any] | None) -> str | None:
    if decision is None:
        return None
    if isinstance(decision, RiskDecision):
        return decision.action.value
    if isinstance(decision, ControlAction):
        return decision.value
    if isinstance(decision, str):
        return _normalize_action_name(decision)
    if isinstance(decision, Mapping):
        raw_action = decision.get("action")
        if raw_action is None:
            return None
        return _normalize_action_name(raw_action)
    return None


def _plan_payload(plan: ClaimVerificationPlan | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    if isinstance(plan, ClaimVerificationPlan):
        return plan.to_dict()
    if isinstance(plan, Mapping):
        payload = to_jsonable(plan)
        return dict(payload) if isinstance(payload, Mapping) else None
    return None


def _decision_allows_plan_retrieval(decision_action: str | None, policy: ActionAuditPolicy) -> bool:
    if decision_action is None:
        return True
    return decision_action in set(policy.retrieval_required_decision_actions)


def _plan_retrieval_query_count(plan: Mapping[str, Any] | None) -> int:
    if not plan:
        return 0
    return sum(1 for item in _as_sequence(plan.get("retrieval_queries", ())) if _mapping_query_text(item))


def _known_plan_claim_ids(plan: Mapping[str, Any] | None) -> set[str]:
    if not plan:
        return set()
    claim_ids: set[str] = set()
    for key in ("verify_claim_ids", "skipped_claim_ids", "triggered_claim_ids"):
        claim_ids.update(str(item) for item in _as_sequence(plan.get(key, ())) if str(item).strip())
    for index, claim in enumerate(_as_sequence(plan.get("claims", ()))):
        if isinstance(claim, Mapping):
            raw_claim_id = claim.get("claim_id", claim.get("id"))
            claim_id = None if raw_claim_id is None else str(raw_claim_id).strip()
            if claim_id:
                claim_ids.add(claim_id)
            else:
                claim_ids.add(f"c{index + 1}")
    for query in _as_sequence(plan.get("retrieval_queries", ())):
        if isinstance(query, Mapping):
            raw_claim_id = query.get("claim_id")
            claim_id = None if raw_claim_id is None else str(raw_claim_id).strip()
            if claim_id:
                claim_ids.add(claim_id)
    return claim_ids


def _action_claim_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    claim_ids: list[str] = []
    for key in ("claim_id", "claim_ids"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            claim_ids.append(value.strip())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            claim_ids.extend(str(item).strip() for item in value if str(item).strip())
    for key in (
        "retrieval_targets",
        "retrieval_queries",
        "blocked_claims",
        "rewrite_targets",
        "clarification_targets",
    ):
        for item in _as_sequence(payload.get(key, ())):
            if not isinstance(item, Mapping):
                continue
            raw_claim_id = item.get("claim_id")
            claim_id = None if raw_claim_id is None else str(raw_claim_id).strip()
            if claim_id:
                claim_ids.append(claim_id)
    return tuple(dict.fromkeys(claim_ids))


def _executable_retrieval_query_count(payload: Mapping[str, Any]) -> int:
    count = 0
    raw_query = payload.get("query")
    if isinstance(raw_query, str) and raw_query.strip():
        count += 1
    targets = payload.get("retrieval_targets", ())
    if isinstance(targets, Mapping):
        targets = (targets,)
    for target in _as_sequence(targets):
        if isinstance(target, str) and target.strip():
            count += 1
        elif isinstance(target, Mapping) and _mapping_query_text(target):
            count += 1
    return count


def _advisory_retrieval_query_count(payload: Mapping[str, Any]) -> int:
    count = 0
    for key in ("retrieval_queries", "queries"):
        for query in _as_sequence(payload.get(key, ())):
            if isinstance(query, str) and query.strip():
                count += 1
            elif isinstance(query, Mapping) and _mapping_query_text(query):
                count += 1
    claim_ids = payload.get("claim_ids", ())
    if isinstance(claim_ids, str) and claim_ids.strip():
        count += 1
    elif isinstance(claim_ids, Sequence) and not isinstance(claim_ids, (str, bytes, bytearray)):
        count += sum(1 for item in claim_ids if str(item).strip())
    return count


def _mapping_query_text(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    raw_query = value.get("query", value.get("text"))
    if raw_query is None:
        return None
    text = str(raw_query).strip()
    return text or None


def _normalize_action_name(value: Any) -> str:
    if isinstance(value, ControlAction):
        return value.value
    return str(value).strip()


def _first_non_empty_string(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError("expected a boolean value.")
