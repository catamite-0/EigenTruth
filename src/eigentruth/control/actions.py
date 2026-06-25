"""Executable action payloads for factuality-control decisions."""

from __future__ import annotations

import json
import math
import sqlite3
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from eigentruth.control.policy import ControlAction, RiskDecision
from eigentruth.json_utils import strict_json_dumps, to_jsonable
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus


class ActionExecutionStatus(str, Enum):
    """Outcome of executing or dry-running an action request."""

    DRY_RUN = "dry_run"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class ActionResult:
    """JSON-ready result produced by an action executor."""

    action: ControlAction
    status: ActionExecutionStatus
    output: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "action": self.action.value,
            "status": self.status.value,
            "output": _jsonable(self.output),
            "metadata": _jsonable(self.metadata),
            "request_id": self.request_id,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionResult":
        """Build an action result from JSON-like data."""
        return cls(
            action=ControlAction(str(data["action"])),
            status=ActionExecutionStatus(str(data["status"])),
            output=dict(data.get("output", {})),
            metadata=dict(data.get("metadata", {})),
            request_id=None if data.get("request_id") is None else str(data["request_id"]),
            error=None if data.get("error") is None else str(data["error"]),
        )


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


@dataclass(frozen=True)
class ActionExecutionPolicy:
    """Request-level execution contract for action executors.

    The policy validates side-effecting action requests and records audit
    metadata. It intentionally does not enforce runtime cancellation; adapters
    that need hard timeouts should enforce them in the wrapped executor.
    """

    side_effecting: bool = False
    require_request_id: bool = False
    require_idempotency_key: bool = False
    default_timeout_seconds: float | None = None
    max_timeout_seconds: float | None = None
    required_metadata_keys: Sequence[str] = ()

    def __post_init__(self) -> None:
        default_timeout = _coerce_optional_positive_float(
            self.default_timeout_seconds,
            name="default_timeout_seconds",
        )
        max_timeout = _coerce_optional_positive_float(
            self.max_timeout_seconds,
            name="max_timeout_seconds",
        )
        if default_timeout is not None and max_timeout is not None and default_timeout > max_timeout:
            raise ValueError("default_timeout_seconds cannot exceed max_timeout_seconds.")
        required_keys = tuple(
            key
            for key in (str(item).strip() for item in self.required_metadata_keys)
            if key
        )
        object.__setattr__(self, "default_timeout_seconds", default_timeout)
        object.__setattr__(self, "max_timeout_seconds", max_timeout)
        object.__setattr__(self, "required_metadata_keys", required_keys)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready policy description."""
        return {
            "side_effecting": self.side_effecting,
            "require_request_id": self.require_request_id,
            "require_idempotency_key": self.require_idempotency_key,
            "default_timeout_seconds": self.default_timeout_seconds,
            "max_timeout_seconds": self.max_timeout_seconds,
            "required_metadata_keys": tuple(self.required_metadata_keys),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionExecutionPolicy":
        """Build an execution policy from JSON-like data."""
        return cls(
            side_effecting=_parse_policy_bool(data.get("side_effecting", False), name="side_effecting"),
            require_request_id=_parse_policy_bool(data.get("require_request_id", False), name="require_request_id"),
            require_idempotency_key=_parse_policy_bool(
                data.get("require_idempotency_key", False),
                name="require_idempotency_key",
            ),
            default_timeout_seconds=data.get("default_timeout_seconds"),
            max_timeout_seconds=data.get("max_timeout_seconds"),
            required_metadata_keys=_as_tuple(data.get("required_metadata_keys", ())),
        )

    def validate_request(self, request: ActionRequest) -> tuple[str, ...]:
        """Return policy violation messages for a request."""
        violations: list[str] = []
        if self.require_request_id and not _non_empty_string(request.request_id):
            violations.append("request_id is required.")
        if self.require_idempotency_key and self.idempotency_key(request) is None:
            violations.append("idempotency_key is required for this action.")
        metadata = dict(request.metadata)
        for key in self.required_metadata_keys:
            if key not in metadata or metadata[key] is None:
                violations.append(f"metadata.{key} is required.")
        try:
            timeout_seconds = self.timeout_seconds(request)
        except ValueError as exc:
            violations.append(str(exc))
        else:
            if (
                timeout_seconds is not None
                and self.max_timeout_seconds is not None
                and timeout_seconds > self.max_timeout_seconds
            ):
                violations.append(
                    "timeout_seconds exceeds max_timeout_seconds "
                    f"({timeout_seconds} > {self.max_timeout_seconds})."
                )
        return tuple(violations)

    def audit_metadata(self, request: ActionRequest) -> dict[str, Any]:
        """Return standardized execution-audit metadata for a request."""
        try:
            timeout_seconds = self.timeout_seconds(request)
        except ValueError:
            timeout_seconds = None
        return {
            "execution_policy": self.to_dict(),
            "idempotency_key": self.idempotency_key(request),
            "timeout_seconds": timeout_seconds,
            "timeout_enforced": False,
            "side_effecting_executor": self.side_effecting,
        }

    def idempotency_key(self, request: ActionRequest) -> str | None:
        """Resolve the idempotency key from request metadata or payload."""
        return _non_empty_string(_request_value(request, "idempotency_key"))

    def timeout_seconds(self, request: ActionRequest) -> float | None:
        """Resolve the requested timeout, falling back to the policy default."""
        value = _request_value(request, "timeout_seconds")
        if value is None:
            return self.default_timeout_seconds
        return _coerce_optional_positive_float(value, name="timeout_seconds")


@runtime_checkable
class ActionExecutionLedger(Protocol):
    """Lookup and persist idempotent action execution results."""

    def get(self, idempotency_key: str) -> ActionResult | None:
        """Return a recorded result for an idempotency key, if present."""
        ...

    def record(self, idempotency_key: str, result: ActionResult) -> None:
        """Persist a result for an idempotency key."""
        ...


@dataclass
class InMemoryActionExecutionLedger:
    """Request-local idempotency ledger for tests and in-process demos."""

    records: Mapping[str, ActionResult | Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._records = {
            str(key): _coerce_action_result(value)
            for key, value in self.records.items()
        }

    def get(self, idempotency_key: str) -> ActionResult | None:
        """Return a recorded result for an idempotency key, if present."""
        return self._records.get(str(idempotency_key))

    def record(self, idempotency_key: str, result: ActionResult) -> None:
        """Persist a result for an idempotency key."""
        self._records.setdefault(str(idempotency_key), result)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready ledger snapshot."""
        return {
            key: result.to_dict()
            for key, result in self._records.items()
        }


@dataclass(frozen=True)
class JsonActionExecutionLedger:
    """File-backed idempotency ledger for local reproducible workflows."""

    path: str | Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def get(self, idempotency_key: str) -> ActionResult | None:
        """Return a recorded result for an idempotency key, if present."""
        data = self._load()
        payload = data.get(str(idempotency_key))
        if payload is None:
            return None
        return _coerce_action_result(payload)

    def record(self, idempotency_key: str, result: ActionResult) -> None:
        """Persist a result for an idempotency key."""
        data = self._load()
        data.setdefault(str(idempotency_key), result.to_dict())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(strict_json_dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("action execution ledger must contain a JSON object.")
        return dict(payload)


@dataclass(frozen=True)
class SQLiteActionExecutionLedger:
    """SQLite-backed idempotency ledger for durable local action replay."""

    path: str | Path
    table_name: str = "action_execution_ledger"
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        table_name = _sqlite_identifier(self.table_name, name="table_name")
        timeout_seconds = _coerce_optional_positive_float(self.timeout_seconds, name="timeout_seconds")
        if timeout_seconds is None:
            raise ValueError("timeout_seconds is required.")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "table_name", table_name)
        object.__setattr__(self, "timeout_seconds", timeout_seconds)

    def get(self, idempotency_key: str) -> ActionResult | None:
        """Return a recorded result for an idempotency key, if present."""
        with self._connect() as connection:
            self._ensure_table(connection)
            row = connection.execute(
                f"select result_json from {self.table_name} where idempotency_key = ?",
                (str(idempotency_key),),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row[0]))
        if not isinstance(payload, Mapping):
            raise ValueError("sqlite action execution ledger row must contain a JSON object.")
        return _coerce_action_result(payload)

    def record(self, idempotency_key: str, result: ActionResult) -> None:
        """Persist a result for an idempotency key with first-write-wins semantics."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._ensure_table(connection)
            connection.execute(
                f"""
                insert or ignore into {self.table_name} (idempotency_key, result_json)
                values (?, ?)
                """,
                (str(idempotency_key), strict_json_dumps(result.to_dict(), sort_keys=True)),
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path), timeout=float(self.timeout_seconds))

    def _ensure_table(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            f"""
            create table if not exists {self.table_name} (
                idempotency_key text primary key,
                result_json text not null,
                created_at text not null default current_timestamp
            )
            """
        )
        connection.commit()


@runtime_checkable
class ActionExecutor(Protocol):
    """Interface for executing planned action requests."""

    def execute(
        self,
        request: ActionRequest,
        context: Mapping[str, Any] | None = None,
    ) -> ActionResult:
        """Execute or dry-run one action request."""
        ...

    def execute_many(
        self,
        requests: Sequence[ActionRequest],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[ActionResult, ...]:
        """Execute or dry-run multiple action requests."""
        ...


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


@dataclass(frozen=True)
class DryRunActionExecutor:
    """Executor that records what would happen without calling external tools."""

    executor_name: str = "dry_run"

    def execute(
        self,
        request: ActionRequest,
        context: Mapping[str, Any] | None = None,
    ) -> ActionResult:
        """Dry-run one action request and return an execution result."""
        output = _dry_run_output(request)
        metadata = {
            "executor": type(self).__name__,
            "executor_name": self.executor_name,
            "request_metadata": dict(request.metadata),
            "context": dict(context or {}),
            "side_effects": False,
        }
        return ActionResult(
            action=request.action,
            status=ActionExecutionStatus.DRY_RUN,
            output=output,
            metadata=metadata,
            request_id=request.request_id,
        )

    def execute_many(
        self,
        requests: Sequence[ActionRequest],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[ActionResult, ...]:
        """Dry-run multiple action requests."""
        return tuple(self.execute(request, context=context) for request in requests)


@dataclass(frozen=True)
class TimeoutActionExecutor:
    """Apply a best-effort wall-clock timeout around another executor.

    The wrapper uses a thread pool and returns before the configured timeout
    when the wrapped executor does not complete. Python cannot safely terminate
    an already-running thread, so use this wrapper for non-side-effecting,
    idempotent, or adapter-level cancellable work. Side-effecting integrations
    should still enforce hard cancellation inside the concrete adapter.
    """

    executor: ActionExecutor
    default_timeout_seconds: float | None = None
    max_workers: int = 1

    def __post_init__(self) -> None:
        default_timeout = _coerce_optional_positive_float(
            self.default_timeout_seconds,
            name="default_timeout_seconds",
        )
        max_workers = _coerce_positive_int(self.max_workers, name="max_workers")
        object.__setattr__(self, "default_timeout_seconds", default_timeout)
        object.__setattr__(self, "max_workers", max_workers)
        object.__setattr__(
            self,
            "_pool",
            ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="eigentruth-action-timeout"),
        )

    def execute(
        self,
        request: ActionRequest,
        context: Mapping[str, Any] | None = None,
    ) -> ActionResult:
        """Execute one action request with a best-effort timeout."""
        try:
            timeout_seconds = self._timeout_seconds(request)
        except ValueError as exc:
            return self._failed_result(request, str(exc), timeout_seconds=None, context=context)
        if timeout_seconds is None:
            return self.executor.execute(request, context=context)

        future: Future[ActionResult] = self._pool.submit(self.executor.execute, request, context)
        try:
            result = future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            cancelled = future.cancel()
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.TIMED_OUT,
                output={},
                metadata={
                    "executor": type(self).__name__,
                    "wrapped_executor": type(self.executor).__name__,
                    "timeout_seconds": timeout_seconds,
                    "timeout_enforced": True,
                    "timeout_mechanism": "thread_future_result",
                    "executor_cancelled": cancelled,
                    "side_effects": False,
                    "context": dict(context or {}),
                },
                request_id=request.request_id,
                error=f"action execution timed out after {timeout_seconds} seconds.",
            )
        except Exception as exc:  # pragma: no cover - defensive adapter boundary
            return self._failed_result(
                request,
                f"wrapped executor failed: {exc}",
                timeout_seconds=timeout_seconds,
                context=context,
            )

        metadata = dict(result.metadata)
        metadata.update({
            "timeout_wrapper": type(self).__name__,
            "timeout_seconds": timeout_seconds,
            "timeout_enforced": True,
            "timeout_mechanism": "thread_future_result",
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
        """Execute multiple action requests with per-request timeouts."""
        return tuple(self.execute(request, context=context) for request in requests)

    def shutdown(self, *, wait: bool = False, cancel_futures: bool = True) -> None:
        """Shut down the internal thread pool."""
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)

    def _timeout_seconds(self, request: ActionRequest) -> float | None:
        value = _request_value(request, "timeout_seconds")
        if value is None:
            return self.default_timeout_seconds
        return _coerce_optional_positive_float(value, name="timeout_seconds")

    def _failed_result(
        self,
        request: ActionRequest,
        error: str,
        *,
        timeout_seconds: float | None,
        context: Mapping[str, Any] | None,
    ) -> ActionResult:
        return ActionResult(
            action=request.action,
            status=ActionExecutionStatus.FAILED,
            output={},
            metadata={
                "executor": type(self).__name__,
                "wrapped_executor": type(self.executor).__name__,
                "timeout_seconds": timeout_seconds,
                "timeout_enforced": False,
                "side_effects": False,
                "context": dict(context or {}),
            },
            request_id=request.request_id,
            error=error,
        )


@dataclass(frozen=True)
class PolicyGuardedActionExecutor:
    """Validate requests and attach audit metadata around another executor."""

    executor: ActionExecutor
    policy: ActionExecutionPolicy = field(default_factory=ActionExecutionPolicy)
    idempotency_ledger: ActionExecutionLedger | None = None
    record_statuses: Sequence[ActionExecutionStatus | str] = (ActionExecutionStatus.SUCCEEDED,)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "record_statuses",
            tuple(_coerce_execution_status(status) for status in self.record_statuses),
        )

    def execute(
        self,
        request: ActionRequest,
        context: Mapping[str, Any] | None = None,
    ) -> ActionResult:
        """Validate and execute one action request."""
        violations = self.policy.validate_request(request)
        audit_metadata = self.policy.audit_metadata(request)
        if violations:
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.FAILED,
                output={},
                metadata={
                    "executor": type(self).__name__,
                    "wrapped_executor": type(self.executor).__name__,
                    "side_effects": False,
                    **audit_metadata,
                    "violations": violations,
                },
                request_id=request.request_id,
                error="action execution policy violation: " + "; ".join(violations),
            )

        idempotency_key = audit_metadata.get("idempotency_key")
        if self.idempotency_ledger is not None and idempotency_key is not None:
            recorded = self.idempotency_ledger.get(str(idempotency_key))
            if recorded is not None:
                return self._replay_result(recorded, audit_metadata)

        result = self.executor.execute(request, context=context)
        metadata = dict(result.metadata)
        if audit_metadata.get("timeout_seconds") is None and metadata.get("timeout_seconds") is not None:
            audit_metadata["timeout_seconds"] = metadata["timeout_seconds"]
        audit_metadata["timeout_enforced"] = bool(
            audit_metadata.get("timeout_enforced", False)
            or metadata.get("timeout_enforced", False)
        )
        metadata.update({
            "policy_guard": type(self).__name__,
            **audit_metadata,
            "idempotency_replayed": False,
        })
        guarded_result = ActionResult(
            action=result.action,
            status=result.status,
            output=result.output,
            metadata=metadata,
            request_id=result.request_id,
            error=result.error,
        )
        if (
            self.idempotency_ledger is not None
            and idempotency_key is not None
            and guarded_result.status in self.record_statuses
        ):
            self.idempotency_ledger.record(str(idempotency_key), guarded_result)
        return guarded_result

    def execute_many(
        self,
        requests: Sequence[ActionRequest],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[ActionResult, ...]:
        """Validate and execute multiple action requests."""
        return tuple(self.execute(request, context=context) for request in requests)

    def _replay_result(self, recorded: ActionResult, audit_metadata: Mapping[str, Any]) -> ActionResult:
        metadata = dict(recorded.metadata)
        original_side_effects = bool(metadata.get("side_effects", False))
        metadata.update({
            "policy_guard": type(self).__name__,
            **dict(audit_metadata),
            "idempotency_replayed": True,
            "idempotency_replay_source": type(self.idempotency_ledger).__name__,
            "original_side_effects": original_side_effects,
            "side_effects": False,
        })
        return ActionResult(
            action=recorded.action,
            status=recorded.status,
            output=recorded.output,
            metadata=metadata,
            request_id=recorded.request_id,
            error=recorded.error,
        )


@dataclass
class ActionExecutorRegistry:
    """Route action requests to registered executors with a dry-run fallback."""

    executors: Mapping[ControlAction, ActionExecutor] = field(default_factory=dict)
    fallback_executor: ActionExecutor = field(default_factory=DryRunActionExecutor)

    def __post_init__(self) -> None:
        self.executors = {_coerce_action(action): executor for action, executor in self.executors.items()}

    def register(
        self,
        action: ControlAction | str,
        executor: ActionExecutor,
    ) -> "ActionExecutorRegistry":
        """Register an executor for one control action."""
        next_executors = dict(self.executors)
        next_executors[_coerce_action(action)] = executor
        self.executors = next_executors
        return self

    def get(self, action: ControlAction | str) -> ActionExecutor:
        """Return the executor registered for an action, or the fallback executor."""
        return dict(self.executors).get(_coerce_action(action), self.fallback_executor)

    def execute(
        self,
        request: ActionRequest,
        context: Mapping[str, Any] | None = None,
    ) -> ActionResult:
        """Execute one action request through the registry."""
        return self.get(request.action).execute(request, context=context)

    def execute_many(
        self,
        requests: Sequence[ActionRequest],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[ActionResult, ...]:
        """Execute multiple action requests through the registry."""
        return tuple(self.execute(request, context=context) for request in requests)


def _coerce_action(action: ControlAction | str) -> ControlAction:
    if isinstance(action, ControlAction):
        return action
    return ControlAction(str(action))


def _coerce_execution_status(status: ActionExecutionStatus | str) -> ActionExecutionStatus:
    if isinstance(status, ActionExecutionStatus):
        return status
    return ActionExecutionStatus(str(status))


def _coerce_action_result(value: ActionResult | Mapping[str, Any]) -> ActionResult:
    if isinstance(value, ActionResult):
        return value
    if isinstance(value, Mapping):
        return ActionResult.from_dict(value)
    raise ValueError("ledger result must be an ActionResult or JSON object.")


def _sqlite_identifier(value: Any, *, name: str) -> str:
    text = str(value).strip()
    if not text or not all(part.isalnum() or part == "_" for part in text):
        raise ValueError(f"{name} must contain only letters, numbers, and underscores.")
    if text[0].isdigit():
        raise ValueError(f"{name} cannot start with a number.")
    return text


def _request_value(request: ActionRequest, key: str) -> Any:
    if key in request.metadata:
        return request.metadata[key]
    if key in request.payload:
        return request.payload[key]
    return None


def _coerce_optional_positive_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return number


def _coerce_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, not bool.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        stripped = value.strip()
        signless = stripped[1:] if stripped[:1] in {"+", "-"} else stripped
        if not signless or not signless.isdecimal():
            raise ValueError(f"{name} must be a positive integer.")
        parsed = int(stripped)
    else:
        raise ValueError(f"{name} must be a positive integer.")
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _non_empty_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_policy_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean.")


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _dry_run_output(request: ActionRequest) -> dict[str, Any]:
    payload = dict(request.payload)
    if request.action is ControlAction.ACCEPT:
        return {"would_execute": "accept", "finalize": True, "payload": payload}
    if request.action is ControlAction.RETRIEVE:
        return {
            "would_execute": "retriever",
            "targets": payload.get("retrieval_targets", ()),
            "instruction": payload.get("instruction"),
        }
    if request.action is ControlAction.REWRITE:
        return {
            "would_execute": "rewriter",
            "targets": payload.get("rewrite_targets", ()),
            "instruction": payload.get("instruction"),
        }
    if request.action is ControlAction.STEER_REGENERATE:
        return {
            "would_execute": "generator",
            "diagnostics": payload.get("diagnostics", {}),
            "instruction": payload.get("instruction"),
        }
    if request.action is ControlAction.EXECUTE_TOOL:
        return {
            "would_execute": "tool",
            "tool": payload.get("tool"),
            "input": payload.get("input", {}),
            "instruction": payload.get("instruction"),
        }
    if request.action is ControlAction.ABSTAIN:
        return {
            "would_execute": "abstain",
            "message": payload.get("message"),
            "blocked_claims": payload.get("blocked_claims", ()),
        }
    if request.action is ControlAction.CLARIFY:
        return {
            "would_execute": "clarification_request",
            "questions": payload.get("questions", ()),
            "targets": payload.get("clarification_targets", ()),
        }
    return {"would_execute": request.action.value, "payload": payload}


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
    return to_jsonable(value)
