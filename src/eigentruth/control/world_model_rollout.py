"""Post-action world-model rollout drift audits."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from eigentruth.control.actions import ActionResult
from eigentruth.json_utils import to_jsonable


class WorldModelRolloutStatus(str, Enum):
    """Aggregate status for observed world-model rollout fidelity."""

    PASSED = "passed"
    DRIFTED = "drifted"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class WorldModelRolloutSeverity(str, Enum):
    """Severity for one rollout drift issue."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class WorldModelRolloutPolicy:
    """Policy for comparing predicted and observed post-action states.

    This is a monitor-first audit layer. It does not execute actions or trust the
    world model as ground truth; it only compares the world model's predicted
    state against observed state carried by action-result metadata/output.
    """

    compare_paths: Sequence[str] = ()
    numeric_tolerance: float = 0.0
    max_compared_paths: int = 128

    def __post_init__(self) -> None:
        compare_paths = tuple(str(path).strip() for path in self.compare_paths if str(path).strip())
        numeric_tolerance = _non_negative_float(self.numeric_tolerance, name="numeric_tolerance")
        max_compared_paths = int(self.max_compared_paths)
        if max_compared_paths <= 0:
            raise ValueError("max_compared_paths must be positive.")
        object.__setattr__(self, "compare_paths", compare_paths)
        object.__setattr__(self, "numeric_tolerance", numeric_tolerance)
        object.__setattr__(self, "max_compared_paths", max_compared_paths)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldModelRolloutPolicy":
        """Build a policy from JSON-like data."""
        raw_paths = data.get("compare_paths", ())
        if isinstance(raw_paths, str):
            raw_paths = (raw_paths,)
        if not isinstance(raw_paths, Sequence):
            raise ValueError("compare_paths must be a sequence or string.")
        return cls(
            compare_paths=tuple(str(path) for path in raw_paths),
            numeric_tolerance=data.get("numeric_tolerance", 0.0),
            max_compared_paths=data.get("max_compared_paths", 128),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready policy representation."""
        return {
            "compare_paths": tuple(self.compare_paths),
            "numeric_tolerance": self.numeric_tolerance,
            "max_compared_paths": self.max_compared_paths,
        }


@dataclass(frozen=True)
class WorldModelRolloutIssue:
    """One mismatch or traceability gap in a world-model rollout audit."""

    code: str
    severity: WorldModelRolloutSeverity | str
    message: str
    request_id: str | None = None
    path: str | None = None
    expected: Any = None
    actual: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = str(self.code).strip()
        if not code:
            raise ValueError("world-model rollout issue code must be non-empty.")
        message = str(self.message).strip()
        if not message:
            raise ValueError("world-model rollout issue message must be non-empty.")
        severity = (
            self.severity
            if isinstance(self.severity, WorldModelRolloutSeverity)
            else WorldModelRolloutSeverity(str(self.severity))
        )
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "request_id", None if self.request_id is None else str(self.request_id))
        object.__setattr__(self, "path", None if self.path is None else str(self.path))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable issue."""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "request_id": self.request_id,
            "path": self.path,
            "expected": to_jsonable(self.expected),
            "actual": to_jsonable(self.actual),
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldModelRolloutIssue":
        """Build an issue from JSON-like data."""
        return cls(
            code=str(data["code"]),
            severity=str(data["severity"]),
            message=str(data["message"]),
            request_id=None if data.get("request_id") is None else str(data.get("request_id")),
            path=None if data.get("path") is None else str(data.get("path")),
            expected=data.get("expected"),
            actual=data.get("actual"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class WorldModelRolloutRecord:
    """One action-result comparison between predicted and observed state."""

    request_id: str | None
    action: str | None
    prediction_available: bool
    observation_available: bool
    compared_path_count: int
    mismatch_count: int
    numeric_drift_count: int = 0
    categorical_drift_count: int = 0
    missing_predicted_path_count: int = 0
    missing_observed_path_count: int = 0
    numeric_error_sum: float = 0.0
    numeric_error_max: float = 0.0
    prediction_confidence: float | None = None
    issues: Sequence[WorldModelRolloutIssue | Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        issues = tuple(
            issue if isinstance(issue, WorldModelRolloutIssue) else WorldModelRolloutIssue.from_dict(issue)
            for issue in self.issues
        )
        object.__setattr__(self, "request_id", None if self.request_id is None else str(self.request_id))
        object.__setattr__(self, "action", None if self.action is None else str(self.action))
        object.__setattr__(self, "prediction_available", bool(self.prediction_available))
        object.__setattr__(self, "observation_available", bool(self.observation_available))
        object.__setattr__(self, "compared_path_count", _non_negative_int(self.compared_path_count))
        object.__setattr__(self, "mismatch_count", _non_negative_int(self.mismatch_count))
        object.__setattr__(self, "numeric_drift_count", _non_negative_int(self.numeric_drift_count))
        object.__setattr__(self, "categorical_drift_count", _non_negative_int(self.categorical_drift_count))
        object.__setattr__(
            self,
            "missing_predicted_path_count",
            _non_negative_int(self.missing_predicted_path_count),
        )
        object.__setattr__(
            self,
            "missing_observed_path_count",
            _non_negative_int(self.missing_observed_path_count),
        )
        object.__setattr__(
            self,
            "numeric_error_sum",
            _non_negative_float(self.numeric_error_sum, name="numeric_error_sum"),
        )
        object.__setattr__(
            self,
            "numeric_error_max",
            _non_negative_float(self.numeric_error_max, name="numeric_error_max"),
        )
        confidence = None if self.prediction_confidence is None else _unit_interval_float(
            self.prediction_confidence,
            name="prediction_confidence",
        )
        object.__setattr__(self, "prediction_confidence", confidence)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def compared(self) -> bool:
        """Return whether this record had both prediction and observation."""
        return self.prediction_available and self.observation_available

    @property
    def drifted(self) -> bool:
        """Return whether any compared path drifted."""
        return self.mismatch_count > 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""
        return {
            "request_id": self.request_id,
            "action": self.action,
            "prediction_available": self.prediction_available,
            "observation_available": self.observation_available,
            "compared": self.compared,
            "drifted": self.drifted,
            "compared_path_count": self.compared_path_count,
            "mismatch_count": self.mismatch_count,
            "numeric_drift_count": self.numeric_drift_count,
            "categorical_drift_count": self.categorical_drift_count,
            "missing_predicted_path_count": self.missing_predicted_path_count,
            "missing_observed_path_count": self.missing_observed_path_count,
            "numeric_error_sum": self.numeric_error_sum,
            "numeric_error_max": self.numeric_error_max,
            "prediction_confidence": self.prediction_confidence,
            "issues": tuple(issue.to_dict() for issue in self.issues),
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class WorldModelRolloutReport:
    """JSON-ready world-model rollout drift report."""

    status: WorldModelRolloutStatus | str
    records: Sequence[WorldModelRolloutRecord | Mapping[str, Any]]
    policy: WorldModelRolloutPolicy | Mapping[str, Any] = field(default_factory=WorldModelRolloutPolicy)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, WorldModelRolloutStatus)
            else WorldModelRolloutStatus(str(self.status))
        )
        policy = (
            self.policy
            if isinstance(self.policy, WorldModelRolloutPolicy)
            else WorldModelRolloutPolicy.from_dict(self.policy)
        )
        records = tuple(
            record if isinstance(record, WorldModelRolloutRecord) else _record_from_mapping(record)
            for record in self.records
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def available(self) -> bool:
        """Return whether at least one action result carried rollout evidence."""
        return any(record.prediction_available or record.observation_available for record in self.records)

    def summary(self) -> dict[str, Any]:
        """Return compact telemetry for traces and runtime metrics."""
        result_count = len(self.records)
        prediction_available_count = sum(1 for record in self.records if record.prediction_available)
        observation_available_count = sum(1 for record in self.records if record.observation_available)
        compared_count = sum(1 for record in self.records if record.compared)
        drifted_count = sum(1 for record in self.records if record.drifted)
        matched_count = sum(1 for record in self.records if record.compared and not record.drifted)
        trace_gap_count = sum(
            1
            for record in self.records
            if record.prediction_available != record.observation_available
        )
        compared_path_count = sum(record.compared_path_count for record in self.records)
        mismatch_count = sum(record.mismatch_count for record in self.records)
        numeric_drift_count = sum(record.numeric_drift_count for record in self.records)
        categorical_drift_count = sum(record.categorical_drift_count for record in self.records)
        missing_predicted_path_count = sum(record.missing_predicted_path_count for record in self.records)
        missing_observed_path_count = sum(record.missing_observed_path_count for record in self.records)
        numeric_error_sum = sum(record.numeric_error_sum for record in self.records)
        numeric_error_max = max((record.numeric_error_max for record in self.records), default=0.0)
        prediction_confidences = [
            float(record.prediction_confidence)
            for record in self.records
            if record.prediction_confidence is not None
        ]
        counts_by_code: dict[str, int] = {}
        counts_by_severity: dict[str, int] = {}
        counts_by_action: dict[str, int] = {}
        for record in self.records:
            _increment_count(counts_by_action, record.action)
            for issue in record.issues:
                _increment_count(counts_by_code, issue.code)
                _increment_count(counts_by_severity, issue.severity.value)
        return {
            "available": self.available,
            "status": self.status.value,
            "result_count": result_count,
            "prediction_available_count": prediction_available_count,
            "observation_available_count": observation_available_count,
            "compared_count": compared_count,
            "coverage_rate": _safe_div(compared_count, result_count),
            "prediction_coverage_rate": _safe_div(prediction_available_count, result_count),
            "observation_coverage_rate": _safe_div(observation_available_count, result_count),
            "matched_count": matched_count,
            "drifted_count": drifted_count,
            "sync_rate": _safe_div(matched_count, compared_count),
            "drift_rate": _safe_div(drifted_count, compared_count),
            "trace_gap_count": trace_gap_count,
            "trace_gap_rate": _safe_div(trace_gap_count, result_count),
            "compared_path_count": compared_path_count,
            "mismatch_count": mismatch_count,
            "path_mismatch_rate": _safe_div(mismatch_count, compared_path_count),
            "numeric_drift_count": numeric_drift_count,
            "categorical_drift_count": categorical_drift_count,
            "missing_predicted_path_count": missing_predicted_path_count,
            "missing_observed_path_count": missing_observed_path_count,
            "numeric_error_mean": _safe_div(numeric_error_sum, numeric_drift_count),
            "numeric_error_max": numeric_error_max,
            "prediction_confidence_mean": _mean_or_none(prediction_confidences),
            "prediction_confidence_min": min(prediction_confidences) if prediction_confidences else None,
            "counts_by_code": dict(sorted(counts_by_code.items())),
            "counts_by_severity": dict(sorted(counts_by_severity.items())),
            "counts_by_action": dict(sorted(counts_by_action.items())),
            "traceable": compared_count > 0 and trace_gap_count == 0,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "status": self.status.value,
            "available": self.available,
            "policy": self.policy.to_dict(),
            "records": tuple(record.to_dict() for record in self.records),
            "summary": self.summary(),
            "metadata": to_jsonable(dict(self.metadata)),
        }


def audit_world_model_rollout(
    action_results: Sequence[ActionResult | Mapping[str, Any]],
    policy: WorldModelRolloutPolicy | Mapping[str, Any] | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> WorldModelRolloutReport:
    """Audit observed post-action states against world-model predictions."""
    resolved_policy = _coerce_policy(policy)
    records = tuple(_rollout_record(_result_to_mapping(result), resolved_policy) for result in action_results)
    status = _rollout_status(records)
    return WorldModelRolloutReport(
        status=status,
        records=records,
        policy=resolved_policy,
        metadata=dict(metadata or {}),
    )


def _rollout_record(
    result: Mapping[str, Any],
    policy: WorldModelRolloutPolicy,
) -> WorldModelRolloutRecord:
    request_id = None if result.get("request_id") is None else str(result.get("request_id"))
    action = None if result.get("action") is None else str(result.get("action"))
    prediction_available, predicted_state, prediction_metadata = _predicted_state(result)
    observation_available, observed_state, observation_metadata = _observed_state(result)
    prediction_confidence = _prediction_confidence(result, prediction_metadata)
    if not prediction_available or not observation_available:
        issues = []
        if prediction_available != observation_available:
            code = "missing_observed_state" if prediction_available else "missing_predicted_state"
            issues.append(WorldModelRolloutIssue(
                code=code,
                severity=WorldModelRolloutSeverity.WARNING,
                message="world-model rollout comparison is missing one side of the state pair.",
                request_id=request_id,
            ))
        return WorldModelRolloutRecord(
            request_id=request_id,
            action=action,
            prediction_available=prediction_available,
            observation_available=observation_available,
            compared_path_count=0,
            mismatch_count=0,
            prediction_confidence=prediction_confidence,
            issues=tuple(issues),
            metadata={
                "prediction_metadata": prediction_metadata,
                "observation_metadata": observation_metadata,
            },
        )

    paths, truncated = _comparison_paths(predicted_state, observed_state, policy)
    issues: list[WorldModelRolloutIssue] = []
    mismatch_count = 0
    numeric_drift_count = 0
    categorical_drift_count = 0
    missing_predicted_path_count = 0
    missing_observed_path_count = 0
    numeric_error_sum = 0.0
    numeric_error_max = 0.0
    if truncated:
        issues.append(WorldModelRolloutIssue(
            code="comparison_truncated",
            severity=WorldModelRolloutSeverity.WARNING,
            message="world-model rollout comparison exceeded max_compared_paths.",
            request_id=request_id,
            metadata={"max_compared_paths": policy.max_compared_paths},
        ))
    for path in paths:
        predicted_found, predicted_value = _get_path(predicted_state, path)
        observed_found, observed_value = _get_path(observed_state, path)
        if not predicted_found:
            missing_predicted_path_count += 1
            mismatch_count += 1
            issues.append(WorldModelRolloutIssue(
                code="missing_predicted_path",
                severity=WorldModelRolloutSeverity.ERROR,
                message="observed state contains a path absent from the predicted state.",
                request_id=request_id,
                path=path,
                actual=observed_value,
            ))
            continue
        if not observed_found:
            missing_observed_path_count += 1
            mismatch_count += 1
            issues.append(WorldModelRolloutIssue(
                code="missing_observed_path",
                severity=WorldModelRolloutSeverity.ERROR,
                message="predicted state contains a path absent from the observed state.",
                request_id=request_id,
                path=path,
                expected=predicted_value,
            ))
            continue
        if _is_number(predicted_value) and _is_number(observed_value):
            error = abs(float(predicted_value) - float(observed_value))
            if error > policy.numeric_tolerance:
                mismatch_count += 1
                numeric_drift_count += 1
                numeric_error_sum += error
                numeric_error_max = max(numeric_error_max, error)
                issues.append(WorldModelRolloutIssue(
                    code="numeric_drift",
                    severity=WorldModelRolloutSeverity.ERROR,
                    message="observed numeric state drifted from the world-model prediction.",
                    request_id=request_id,
                    path=path,
                    expected=predicted_value,
                    actual=observed_value,
                    metadata={"absolute_error": error, "numeric_tolerance": policy.numeric_tolerance},
                ))
            continue
        if predicted_value != observed_value:
            mismatch_count += 1
            categorical_drift_count += 1
            issues.append(WorldModelRolloutIssue(
                code="value_drift",
                severity=WorldModelRolloutSeverity.ERROR,
                message="observed state value differs from the world-model prediction.",
                request_id=request_id,
                path=path,
                expected=predicted_value,
                actual=observed_value,
            ))
    return WorldModelRolloutRecord(
        request_id=request_id,
        action=action,
        prediction_available=True,
        observation_available=True,
        compared_path_count=len(paths),
        mismatch_count=mismatch_count,
        numeric_drift_count=numeric_drift_count,
        categorical_drift_count=categorical_drift_count,
        missing_predicted_path_count=missing_predicted_path_count,
        missing_observed_path_count=missing_observed_path_count,
        numeric_error_sum=numeric_error_sum,
        numeric_error_max=numeric_error_max,
        prediction_confidence=prediction_confidence,
        issues=tuple(issues),
        metadata={
            "prediction_metadata": prediction_metadata,
            "observation_metadata": observation_metadata,
        },
    )


def _rollout_status(records: Sequence[WorldModelRolloutRecord]) -> WorldModelRolloutStatus:
    if not records or not any(record.prediction_available or record.observation_available for record in records):
        return WorldModelRolloutStatus.SKIPPED
    if any(record.drifted for record in records):
        return WorldModelRolloutStatus.DRIFTED
    if any(record.prediction_available != record.observation_available for record in records):
        return WorldModelRolloutStatus.PARTIAL
    return WorldModelRolloutStatus.PASSED


def _predicted_state(result: Mapping[str, Any]) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    metadata = _mapping(result.get("metadata"))
    output = _mapping(result.get("output"))
    candidates = (
        _mapping(metadata.get("world_model_rollout")),
        _mapping(output.get("world_model_rollout")),
        _mapping(metadata.get("world_model_prediction")),
        _mapping(output.get("world_model_prediction")),
        _mapping(output.get("world_model_gate")),
        _mapping(metadata.get("world_model_gate")),
    )
    for candidate in candidates:
        if "predicted_state" in candidate and isinstance(candidate.get("predicted_state"), Mapping):
            return True, _mapping(candidate.get("predicted_state")), dict(candidate)
    return False, {}, {}


def _observed_state(result: Mapping[str, Any]) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    metadata = _mapping(result.get("metadata"))
    output = _mapping(result.get("output"))
    candidates = (
        _mapping(metadata.get("world_model_rollout")),
        _mapping(output.get("world_model_rollout")),
        _mapping(metadata.get("world_model_observation")),
        _mapping(output.get("world_model_observation")),
    )
    for candidate in candidates:
        for key in ("observed_state", "state_after", "state", "actual_state"):
            if key in candidate and isinstance(candidate.get(key), Mapping):
                return True, _mapping(candidate.get(key)), dict(candidate)
    for key in ("observed_state", "state_after", "actual_state"):
        if key in output and isinstance(output.get(key), Mapping):
            return True, _mapping(output.get(key)), {"source": f"output.{key}"}
        if key in metadata and isinstance(metadata.get(key), Mapping):
            return True, _mapping(metadata.get(key)), {"source": f"metadata.{key}"}
    return False, {}, {}


def _prediction_confidence(
    result: Mapping[str, Any],
    prediction_metadata: Mapping[str, Any],
) -> float | None:
    for value in (
        prediction_metadata.get("prediction_confidence"),
        prediction_metadata.get("confidence"),
        _mapping(_mapping(result.get("metadata")).get("world_model_gate")).get("prediction_confidence"),
    ):
        numeric = _finite_float(value)
        if numeric is not None and 0.0 <= numeric <= 1.0:
            return numeric
    return None


def _comparison_paths(
    predicted_state: Mapping[str, Any],
    observed_state: Mapping[str, Any],
    policy: WorldModelRolloutPolicy,
) -> tuple[tuple[str, ...], bool]:
    if policy.compare_paths:
        paths = tuple(policy.compare_paths)
    else:
        paths = tuple(sorted(set(_leaf_paths(predicted_state)) | set(_leaf_paths(observed_state))))
    if len(paths) <= policy.max_compared_paths:
        return paths, False
    return paths[: policy.max_compared_paths], True


def _leaf_paths(data: Mapping[str, Any], prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            paths.extend(_leaf_paths(value, path))
        else:
            paths.append(path)
    return tuple(paths)


def _get_path(data: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _record_from_mapping(data: Mapping[str, Any]) -> WorldModelRolloutRecord:
    return WorldModelRolloutRecord(
        request_id=None if data.get("request_id") is None else str(data.get("request_id")),
        action=None if data.get("action") is None else str(data.get("action")),
        prediction_available=data.get("prediction_available", False),
        observation_available=data.get("observation_available", False),
        compared_path_count=data.get("compared_path_count", 0),
        mismatch_count=data.get("mismatch_count", 0),
        numeric_drift_count=data.get("numeric_drift_count", 0),
        categorical_drift_count=data.get("categorical_drift_count", 0),
        missing_predicted_path_count=data.get("missing_predicted_path_count", 0),
        missing_observed_path_count=data.get("missing_observed_path_count", 0),
        numeric_error_sum=data.get("numeric_error_sum", 0.0),
        numeric_error_max=data.get("numeric_error_max", 0.0),
        prediction_confidence=data.get("prediction_confidence"),
        issues=tuple(_as_sequence(data.get("issues", ()))),
        metadata=dict(data.get("metadata", {})),
    )


def _result_to_mapping(result: ActionResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, ActionResult):
        return result.to_dict()
    return dict(to_jsonable(dict(result)))


def _coerce_policy(
    policy: WorldModelRolloutPolicy | Mapping[str, Any] | None,
) -> WorldModelRolloutPolicy:
    if policy is None:
        return WorldModelRolloutPolicy()
    if isinstance(policy, WorldModelRolloutPolicy):
        return policy
    return WorldModelRolloutPolicy.from_dict(policy)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return (value,)


def _increment_count(counts: dict[str, int], raw_key: Any) -> None:
    if raw_key is None:
        return
    key = str(raw_key)
    if not key:
        return
    counts[key] = counts.get(key, 0) + 1


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _unit_interval_float(value: Any, *, name: str) -> float:
    numeric = _finite_float(value)
    if numeric is None:
        raise ValueError(f"{name} must be a finite numeric value.")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return numeric


def _non_negative_float(value: Any, *, name: str) -> float:
    numeric = _finite_float(value)
    if numeric is None or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return numeric


def _non_negative_int(value: Any) -> int:
    numeric = int(value)
    if numeric < 0:
        raise ValueError("count fields must be non-negative.")
    return numeric
