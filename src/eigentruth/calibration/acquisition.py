"""Post-acquisition conformal calibration for evidence-acquisition policies."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from eigentruth import __version__
from eigentruth.calibration.artifacts import CalibrationArtifact, CalibrationScore, SteeringPolicyConfig
from eigentruth.eval.conformal import (
    ConformalAbstentionReport,
    alpha_spending_schedule,
    conformal_abstention_report,
)
from eigentruth.json_utils import strict_json_dumps, to_jsonable


@dataclass(frozen=True)
class EvidenceAcquisitionCalibrationRecord:
    """One labeled row for calibrating a post-acquisition policy score."""

    post_score: float
    correct: bool
    pre_score: float | None = None
    action: str = "answer"
    acquired: bool = False
    record_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        post_score = _finite_float(self.post_score, name="post_score")
        pre_score = None if self.pre_score is None else _finite_float(self.pre_score, name="pre_score")
        action = str(self.action).strip().lower() or ("acquire" if self.acquired else "answer")
        if action not in {"answer", "acquire", "abstain"}:
            raise ValueError("action must be one of: answer, acquire, abstain.")
        object.__setattr__(self, "post_score", post_score)
        object.__setattr__(self, "pre_score", pre_score)
        object.__setattr__(self, "correct", _strict_bool(self.correct, name="correct"))
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "acquired", _strict_bool(self.acquired, name="acquired") or action == "acquire")
        if self.record_id is not None:
            object.__setattr__(self, "record_id", str(self.record_id))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready record."""
        return {
            "record_id": self.record_id,
            "pre_score": self.pre_score,
            "post_score": self.post_score,
            "correct": self.correct,
            "action": self.action,
            "acquired": self.acquired,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceAcquisitionCalibrationRecord":
        """Build a calibration record from JSON-like data."""
        acquired = data.get("acquired")
        action = str(data.get("action", "")).strip().lower()
        if acquired is None:
            acquired = action == "acquire"
        if not action:
            action = "acquire" if acquired else "answer"
        return cls(
            record_id=None if data.get("record_id") is None else str(data["record_id"]),
            pre_score=None if data.get("pre_score") is None else data["pre_score"],
            post_score=data.get("post_score", data.get("score")),
            correct=data.get("correct", data.get("label")),
            action=action,
            acquired=acquired,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class EvidenceAcquisitionCalibrationReport:
    """Compare naive pre-thresholding with a calibrated post-acquisition policy."""

    alpha: float
    score_name: str
    direction: str
    n_records: int
    n_correct: int
    n_acquired: int
    n_answered: int
    n_abstained: int
    naive_pre_report: ConformalAbstentionReport | None
    post_acquisition_report: ConformalAbstentionReport
    acquisition_rate: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        alpha = _alpha_float(self.alpha)
        n_records = _non_negative_int(self.n_records, name="n_records")
        n_correct = _non_negative_int(self.n_correct, name="n_correct")
        n_acquired = _non_negative_int(self.n_acquired, name="n_acquired")
        n_answered = _non_negative_int(self.n_answered, name="n_answered")
        n_abstained = _non_negative_int(self.n_abstained, name="n_abstained")
        acquisition_rate = _unit_interval_float(self.acquisition_rate, name="acquisition_rate")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "score_name", str(self.score_name))
        object.__setattr__(self, "n_records", n_records)
        object.__setattr__(self, "n_correct", n_correct)
        object.__setattr__(self, "n_acquired", n_acquired)
        object.__setattr__(self, "n_answered", n_answered)
        object.__setattr__(self, "n_abstained", n_abstained)
        object.__setattr__(self, "acquisition_rate", acquisition_rate)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def post_threshold(self) -> float:
        """Threshold calibrated on the post-acquisition policy score."""
        return self.post_acquisition_report.threshold

    @property
    def naive_pre_threshold(self) -> float | None:
        """Threshold from pre-acquisition scores when those scores were supplied."""
        if self.naive_pre_report is None:
            return None
        return self.naive_pre_report.threshold

    @property
    def selective_accuracy_delta(self) -> float | None:
        """Post-policy minus naive selective accuracy, when both are defined."""
        if self.naive_pre_report is None:
            return None
        return _optional_delta(
            self.post_acquisition_report.empirical_selective_accuracy,
            self.naive_pre_report.empirical_selective_accuracy,
        )

    @property
    def participation_rate_delta(self) -> float | None:
        """Post-policy minus naive participation rate, when pre scores exist."""
        if self.naive_pre_report is None:
            return None
        return (
            self.post_acquisition_report.empirical_participation_rate
            - self.naive_pre_report.empirical_participation_rate
        )

    @property
    def conditional_correctness_lower_bound_delta(self) -> float | None:
        """Post-policy minus naive conservative conditional-correctness bound."""
        if self.naive_pre_report is None:
            return None
        return (
            self.post_acquisition_report.conditional_correctness_lower_bound
            - self.naive_pre_report.conditional_correctness_lower_bound
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report."""
        return {
            "alpha": self.alpha,
            "score_name": self.score_name,
            "direction": self.direction,
            "n_records": self.n_records,
            "n_correct": self.n_correct,
            "n_acquired": self.n_acquired,
            "n_answered": self.n_answered,
            "n_abstained": self.n_abstained,
            "acquisition_rate": self.acquisition_rate,
            "naive_pre_threshold": self.naive_pre_threshold,
            "post_threshold": self.post_threshold,
            "naive_pre_report": None if self.naive_pre_report is None else self.naive_pre_report.to_dict(),
            "post_acquisition_report": self.post_acquisition_report.to_dict(),
            "deltas": {
                "selective_accuracy": self.selective_accuracy_delta,
                "participation_rate": self.participation_rate_delta,
                "conditional_correctness_lower_bound": self.conditional_correctness_lower_bound_delta,
            },
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceAcquisitionCalibrationReport":
        """Build a report from JSON-like data."""
        naive_payload = data.get("naive_pre_report")
        return cls(
            alpha=data["alpha"],
            score_name=str(data["score_name"]),
            direction=str(data.get("direction", "higher")),
            n_records=int(data.get("n_records", 0)),
            n_correct=int(data.get("n_correct", 0)),
            n_acquired=int(data.get("n_acquired", 0)),
            n_answered=int(data.get("n_answered", 0)),
            n_abstained=int(data.get("n_abstained", 0)),
            acquisition_rate=float(data.get("acquisition_rate", 0.0)),
            naive_pre_report=(None if naive_payload is None else ConformalAbstentionReport.from_dict(naive_payload)),
            post_acquisition_report=ConformalAbstentionReport.from_dict(data["post_acquisition_report"]),
            metadata=dict(data.get("metadata", {})),
        )

    def save_json(self, path: str | Path) -> None:
        """Save the report as strict JSON."""
        Path(path).write_text(strict_json_dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class EvidenceAcquisitionCalibrationResult:
    """Calibrated report plus reusable post-acquisition artifact."""

    report: EvidenceAcquisitionCalibrationReport
    artifact: CalibrationArtifact

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result."""
        return {
            "report": self.report.to_dict(),
            "artifact": self.artifact.to_dict(),
        }


@dataclass(frozen=True)
class EvidenceAcquisitionRiskCheck:
    """One prefix check for post-acquisition calibration risk monitoring."""

    step: int
    checkpoint: int
    alpha_spent: float
    cumulative_alpha_spent: float
    target_error_rate: float
    threshold: float
    direction: str
    n_records: int
    accepted_count: int
    accepted_errors: int
    accepted_error_rate: float | None
    accepted_error_upper_bound: float | None
    exceeded: bool
    n_acquired: int = 0
    n_abstained: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step = _positive_int(self.step, name="step")
        checkpoint = _positive_int(self.checkpoint, name="checkpoint")
        if checkpoint < step:
            raise ValueError("checkpoint must be at least step.")
        alpha_spent = _unit_interval_float(self.alpha_spent, name="alpha_spent")
        cumulative_alpha_spent = _unit_interval_float(
            self.cumulative_alpha_spent,
            name="cumulative_alpha_spent",
        )
        if alpha_spent <= 0.0:
            raise ValueError("alpha_spent must be positive.")
        if cumulative_alpha_spent < alpha_spent:
            raise ValueError("cumulative_alpha_spent must be at least alpha_spent.")
        direction = str(self.direction)
        if direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        n_records = _non_negative_int(self.n_records, name="n_records")
        accepted_count = _non_negative_int(self.accepted_count, name="accepted_count")
        accepted_errors = _non_negative_int(self.accepted_errors, name="accepted_errors")
        if accepted_errors > accepted_count:
            raise ValueError("accepted_errors must not exceed accepted_count.")
        accepted_error_rate = (
            None
            if self.accepted_error_rate is None
            else _unit_interval_float(self.accepted_error_rate, name="accepted_error_rate")
        )
        accepted_error_upper_bound = (
            None
            if self.accepted_error_upper_bound is None
            else _unit_interval_float(
                self.accepted_error_upper_bound,
                name="accepted_error_upper_bound",
            )
        )
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "checkpoint", checkpoint)
        object.__setattr__(self, "alpha_spent", alpha_spent)
        object.__setattr__(self, "cumulative_alpha_spent", cumulative_alpha_spent)
        object.__setattr__(
            self,
            "target_error_rate",
            _unit_interval_float(self.target_error_rate, name="target_error_rate"),
        )
        object.__setattr__(self, "threshold", _threshold_float(self.threshold, name="threshold"))
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "n_records", n_records)
        object.__setattr__(self, "accepted_count", accepted_count)
        object.__setattr__(self, "accepted_errors", accepted_errors)
        object.__setattr__(self, "accepted_error_rate", accepted_error_rate)
        object.__setattr__(self, "accepted_error_upper_bound", accepted_error_upper_bound)
        object.__setattr__(self, "exceeded", bool(self.exceeded))
        object.__setattr__(self, "n_acquired", _non_negative_int(self.n_acquired, name="n_acquired"))
        object.__setattr__(self, "n_abstained", _non_negative_int(self.n_abstained, name="n_abstained"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready check payload."""
        return to_jsonable(
            {
                "step": self.step,
                "checkpoint": self.checkpoint,
                "alpha_spent": self.alpha_spent,
                "cumulative_alpha_spent": self.cumulative_alpha_spent,
                "target_error_rate": self.target_error_rate,
                "threshold": self.threshold,
                "direction": self.direction,
                "n_records": self.n_records,
                "accepted_count": self.accepted_count,
                "accepted_errors": self.accepted_errors,
                "accepted_error_rate": self.accepted_error_rate,
                "accepted_error_upper_bound": self.accepted_error_upper_bound,
                "exceeded": self.exceeded,
                "n_acquired": self.n_acquired,
                "n_abstained": self.n_abstained,
                "metadata": to_jsonable(dict(self.metadata)),
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceAcquisitionRiskCheck":
        """Build a risk check from JSON-like data."""
        return cls(
            step=data["step"],
            checkpoint=data["checkpoint"],
            alpha_spent=data["alpha_spent"],
            cumulative_alpha_spent=data["cumulative_alpha_spent"],
            target_error_rate=data["target_error_rate"],
            threshold=data["threshold"],
            direction=str(data.get("direction", "higher")),
            n_records=data["n_records"],
            accepted_count=data["accepted_count"],
            accepted_errors=data["accepted_errors"],
            accepted_error_rate=data.get("accepted_error_rate"),
            accepted_error_upper_bound=data.get("accepted_error_upper_bound"),
            exceeded=data["exceeded"],
            n_acquired=data.get("n_acquired", 0),
            n_abstained=data.get("n_abstained", 0),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class EvidenceAcquisitionRiskMonitorReport:
    """Alpha-spending audit of a deployed post-acquisition calibration artifact."""

    score_name: str
    threshold: float
    direction: str
    target_error_rate: float
    monitor_alpha: float
    schedule: str
    n_records: int
    passed: bool
    blocking_reasons: tuple[str, ...]
    checks: tuple[EvidenceAcquisitionRiskCheck, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        direction = str(self.direction)
        if direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        checks = tuple(
            check if isinstance(check, EvidenceAcquisitionRiskCheck) else EvidenceAcquisitionRiskCheck.from_dict(check)
            for check in self.checks
        )
        if not checks:
            raise ValueError("checks must be non-empty.")
        object.__setattr__(self, "score_name", str(self.score_name))
        object.__setattr__(self, "threshold", _threshold_float(self.threshold, name="threshold"))
        object.__setattr__(self, "direction", direction)
        object.__setattr__(
            self,
            "target_error_rate",
            _unit_interval_float(self.target_error_rate, name="target_error_rate"),
        )
        object.__setattr__(self, "monitor_alpha", _alpha_float(self.monitor_alpha))
        object.__setattr__(self, "schedule", _alpha_spending_schedule_name(self.schedule))
        object.__setattr__(self, "n_records", _non_negative_int(self.n_records, name="n_records"))
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "blocking_reasons", tuple(str(reason) for reason in self.blocking_reasons))
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def first_failed_checkpoint(self) -> int | None:
        """Return the first checkpoint that exceeded the target, if any."""
        for check in self.checks:
            if check.exceeded:
                return check.checkpoint
        return None

    @property
    def max_accepted_error_upper_bound(self) -> float | None:
        """Return the largest available accepted-error upper bound."""
        bounds = tuple(
            check.accepted_error_upper_bound for check in self.checks if check.accepted_error_upper_bound is not None
        )
        if not bounds:
            return None
        return max(bounds)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready monitor report."""
        return to_jsonable(
            {
                "score_name": self.score_name,
                "threshold": self.threshold,
                "direction": self.direction,
                "target_error_rate": self.target_error_rate,
                "monitor_alpha": self.monitor_alpha,
                "schedule": self.schedule,
                "n_records": self.n_records,
                "passed": self.passed,
                "blocking_reasons": list(self.blocking_reasons),
                "first_failed_checkpoint": self.first_failed_checkpoint,
                "max_accepted_error_upper_bound": self.max_accepted_error_upper_bound,
                "checks": [check.to_dict() for check in self.checks],
                "metadata": to_jsonable(dict(self.metadata)),
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceAcquisitionRiskMonitorReport":
        """Build a monitor report from JSON-like data."""
        return cls(
            score_name=str(data["score_name"]),
            threshold=data["threshold"],
            direction=str(data.get("direction", "higher")),
            target_error_rate=data["target_error_rate"],
            monitor_alpha=data["monitor_alpha"],
            schedule=str(data.get("schedule", "harmonic")),
            n_records=data["n_records"],
            passed=data["passed"],
            blocking_reasons=tuple(data.get("blocking_reasons", ())),
            checks=tuple(EvidenceAcquisitionRiskCheck.from_dict(item) for item in data["checks"]),
            metadata=dict(data.get("metadata", {})),
        )


def audit_evidence_acquisition_risk(
    records: Sequence[EvidenceAcquisitionCalibrationRecord | Mapping[str, Any]],
    *,
    threshold: float,
    target_error_rate: float,
    monitor_alpha: float = 0.05,
    direction: str = "higher",
    schedule: str = "harmonic",
    score_name: str = "post_acquisition_policy_score",
    checkpoints: Sequence[int] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EvidenceAcquisitionRiskMonitorReport:
    """Audit a deployed post-acquisition threshold against labeled feedback.

    The threshold is treated as fixed: this function does not recalibrate on the
    incoming feedback. Instead it evaluates accepted-error confidence bounds over
    a finite sequence of prefixes and spends ``monitor_alpha`` across those
    checks. This is a conservative monitor-first guardrail for deciding whether
    an existing acquisition calibration needs rework.
    """
    rows = _records(records, require_correct=False)
    threshold_value = _threshold_float(threshold, name="threshold")
    target = _unit_interval_float(target_error_rate, name="target_error_rate")
    monitor_alpha_value = _alpha_float(monitor_alpha)
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    checkpoint_values = _risk_checkpoints(checkpoints, n_records=len(rows))
    spends = alpha_spending_schedule(monitor_alpha_value, len(checkpoint_values), schedule=schedule)
    checks: list[EvidenceAcquisitionRiskCheck] = []
    cumulative = 0.0
    for step, (checkpoint, alpha_spent) in enumerate(zip(checkpoint_values, spends, strict=True), start=1):
        cumulative += float(alpha_spent)
        prefix = rows[:checkpoint]
        accepted = tuple(row for row in prefix if _accepted_by_threshold(row.post_score, threshold_value, direction))
        accepted_count = len(accepted)
        accepted_errors = sum(1 for row in accepted if not row.correct)
        upper_bound = (
            None
            if accepted_count == 0
            else _binomial_upper_confidence_bound(
                successes=accepted_errors,
                total=accepted_count,
                alpha=float(alpha_spent),
            )
        )
        exceeded = upper_bound is not None and upper_bound > target
        checks.append(
            EvidenceAcquisitionRiskCheck(
                step=step,
                checkpoint=checkpoint,
                alpha_spent=float(alpha_spent),
                cumulative_alpha_spent=cumulative,
                target_error_rate=target,
                threshold=threshold_value,
                direction=direction,
                n_records=len(prefix),
                accepted_count=accepted_count,
                accepted_errors=accepted_errors,
                accepted_error_rate=None if accepted_count == 0 else accepted_errors / accepted_count,
                accepted_error_upper_bound=upper_bound,
                exceeded=exceeded,
                n_acquired=sum(1 for row in prefix if row.acquired or row.action == "acquire"),
                n_abstained=sum(1 for row in prefix if row.action == "abstain"),
            )
        )
    blocking_reasons = tuple(
        f"checkpoint {check.checkpoint} accepted_error_upper_bound "
        f"{check.accepted_error_upper_bound:.6g} exceeds target_error_rate {target:.6g}"
        for check in checks
        if check.exceeded and check.accepted_error_upper_bound is not None
    )
    return EvidenceAcquisitionRiskMonitorReport(
        score_name=str(score_name),
        threshold=threshold_value,
        direction=direction,
        target_error_rate=target,
        monitor_alpha=monitor_alpha_value,
        schedule=schedule,
        n_records=len(rows),
        passed=not blocking_reasons,
        blocking_reasons=blocking_reasons,
        checks=tuple(checks),
        metadata={
            "calibration_scope": "post_acquisition_policy",
            "monitor": "alpha_spent_feedback_prefix_risk",
            **({} if metadata is None else dict(metadata)),
        },
    )


def evidence_acquisition_record_from_trace(
    trace: Any,
    *,
    correct: bool | int | None = None,
    score_name: str = "post_acquisition_policy_score",
    post_score_name: str | None = None,
    pre_score_name: str | None = None,
    record_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EvidenceAcquisitionCalibrationRecord:
    """Extract one post-acquisition calibration row from a product trace.

    The trace can be a ``ProductTrace``-like object with ``to_dict()`` or a
    JSON-like mapping. ``correct`` may be supplied directly; otherwise the
    helper looks for explicit correctness labels in trace metadata.
    """
    payload = _trace_payload(trace)
    resolved_score_name = _score_name(score_name)
    post_key = _score_name(post_score_name or resolved_score_name)
    pre_key = None if pre_score_name is None else _score_name(pre_score_name)
    post_score, post_source = _post_policy_score(payload, post_key, fallback_key=resolved_score_name)
    pre_score, pre_source = _pre_policy_score(
        payload,
        pre_key or f"pre_{resolved_score_name}",
        fallback_key=resolved_score_name,
    )
    label, label_source = _trace_correctness(payload, correct=correct)
    action, action_source = _trace_acquisition_action(payload)
    fingerprint = _trace_fingerprint(payload)
    request_id = _optional_str(payload.get("request_id"))
    row_metadata = {
        "request_id": request_id,
        "trace_fingerprint": fingerprint,
        "score_name": resolved_score_name,
        "post_score_source": post_source,
        "pre_score_source": pre_source,
        "label_source": label_source,
        "action_source": action_source,
        "risk_decision_action": _nested_str(payload, "risk_decision", "action"),
        "risk_level": _nested_str(payload, "risk_decision", "risk_level"),
        **({} if metadata is None else dict(metadata)),
    }
    return EvidenceAcquisitionCalibrationRecord(
        record_id=record_id or request_id or fingerprint,
        pre_score=pre_score,
        post_score=post_score,
        correct=label,
        action=action,
        acquired=action == "acquire" or _trace_retrieved_evidence(payload),
        metadata=row_metadata,
    )


def evidence_acquisition_records_from_traces(
    traces: Sequence[Any],
    *,
    score_name: str = "post_acquisition_policy_score",
    post_score_name: str | None = None,
    pre_score_name: str | None = None,
) -> tuple[EvidenceAcquisitionCalibrationRecord, ...]:
    """Extract labeled evidence-acquisition calibration rows from traces."""
    rows = tuple(
        evidence_acquisition_record_from_trace(
            trace,
            score_name=score_name,
            post_score_name=post_score_name,
            pre_score_name=pre_score_name,
        )
        for trace in traces
    )
    if not rows:
        raise ValueError("traces must be non-empty.")
    return rows


def evidence_acquisition_records_from_trace_feedback(
    traces: Sequence[Any],
    feedback_records: Sequence[Mapping[str, Any] | Any],
    *,
    score_name: str = "post_acquisition_policy_score",
    post_score_name: str | None = None,
    pre_score_name: str | None = None,
    allow_unmatched: bool = False,
) -> tuple[EvidenceAcquisitionCalibrationRecord, ...]:
    """Join traces with feedback outcomes and return calibration records.

    Feedback rows match by ``trace_fingerprint`` when provided, otherwise by a
    unique ``request_id``. Unknown outcomes are rejected because calibration
    labels must be explicit.
    """
    trace_index = _TraceIndex.from_traces(traces)
    rows: list[EvidenceAcquisitionCalibrationRecord] = []
    unmatched = 0
    for item in feedback_records:
        feedback = _mapping_payload(item)
        trace, reason = trace_index.match(feedback)
        if trace is None:
            if allow_unmatched:
                unmatched += 1
                continue
            raise ValueError(f"feedback row did not match a unique trace: {reason}.")
        outcome = str(feedback.get("outcome", "")).strip().lower()
        correct = _correct_from_outcome(outcome)
        request_id = _optional_str(feedback.get("request_id")) or _optional_str(trace.get("request_id"))
        claim_id = _optional_str(feedback.get("claim_id"))
        record_id = ":".join(part for part in (request_id, claim_id) if part) or None
        rows.append(
            evidence_acquisition_record_from_trace(
                trace,
                correct=correct,
                score_name=score_name,
                post_score_name=post_score_name,
                pre_score_name=pre_score_name,
                record_id=record_id,
                metadata={
                    "feedback_outcome": outcome,
                    "feedback_source": _optional_str(feedback.get("feedback_source")),
                    "feedback_claim_id": claim_id,
                    "label_source": "feedback.outcome",
                },
            )
        )
    if not rows:
        detail = " all feedback rows were unmatched" if unmatched else ""
        raise ValueError(f"no matched feedback records produced calibration rows;{detail}".rstrip(";"))
    return tuple(rows)


@dataclass(frozen=True)
class EvidenceAcquisitionConformalCalibrator:
    """Calibrate the whole post-acquisition policy score.

    Calibrating the score after evidence acquisition is the key distinction from
    a naive conformal abstention filter. If acquisition changes the score
    distribution, callers should use this calibrator on records produced by the
    complete answer/acquire/abstain policy.
    """

    alpha: float = 0.1
    score_name: str = "post_acquisition_policy_score"
    direction: str = "higher"

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _alpha_float(self.alpha))
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        score_name = str(self.score_name).strip()
        if not score_name:
            raise ValueError("score_name must be non-empty.")
        object.__setattr__(self, "score_name", score_name)

    def report(
        self,
        records: Sequence[EvidenceAcquisitionCalibrationRecord | Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> EvidenceAcquisitionCalibrationReport:
        """Build a post-acquisition calibration report from labeled records."""
        rows = _records(records)
        post_scores = tuple(row.post_score for row in rows)
        correctness = tuple(row.correct for row in rows)
        post_report = conformal_abstention_report(
            post_scores,
            correctness,
            self.alpha,
            direction=self.direction,
            score_name=self.score_name,
        )
        pre_scores = tuple(row.pre_score for row in rows)
        naive_pre_report = None
        if all(score is not None for score in pre_scores):
            naive_pre_report = conformal_abstention_report(
                tuple(float(score) for score in pre_scores if score is not None),
                correctness,
                self.alpha,
                direction=self.direction,
                score_name=f"{self.score_name}_pre_naive",
            )
        n = len(rows)
        n_acquired = sum(1 for row in rows if row.acquired or row.action == "acquire")
        n_abstained = sum(1 for row in rows if row.action == "abstain")
        n_answered = sum(1 for row in rows if row.action == "answer")
        return EvidenceAcquisitionCalibrationReport(
            alpha=self.alpha,
            score_name=self.score_name,
            direction=self.direction,
            n_records=n,
            n_correct=sum(1 for value in correctness if value),
            n_acquired=n_acquired,
            n_answered=n_answered,
            n_abstained=n_abstained,
            acquisition_rate=0.0 if n == 0 else n_acquired / n,
            naive_pre_report=naive_pre_report,
            post_acquisition_report=post_report,
            metadata={
                "post_acquisition_calibration": True,
                "calibration_scope": "post_acquisition_policy",
                **({} if metadata is None else dict(metadata)),
            },
        )

    def calibrate(
        self,
        *,
        model_id: str,
        target_layer: int,
        records: Sequence[EvidenceAcquisitionCalibrationRecord | Mapping[str, Any]],
        model_revision: str | None = None,
        steering_policy: SteeringPolicyConfig | None = None,
        warmup_dataset_metadata: Mapping[str, Any] | None = None,
        calibration_dataset_metadata: Mapping[str, Any] | None = None,
        created_at: str | None = None,
        commit_sha: str | None = None,
        eigentruth_version: str = __version__,
    ) -> EvidenceAcquisitionCalibrationResult:
        """Return a report and reusable ``CalibrationArtifact`` for post scores."""
        report = self.report(records, metadata=calibration_dataset_metadata)
        metadata = {
            **dict(calibration_dataset_metadata or {}),
            "post_acquisition_calibration": {
                "score_name": self.score_name,
                "direction": self.direction,
                "alpha": self.alpha,
                "n_records": report.n_records,
                "n_correct": report.n_correct,
                "n_acquired": report.n_acquired,
                "acquisition_rate": report.acquisition_rate,
                "naive_pre_threshold": report.naive_pre_threshold,
                "post_threshold": report.post_threshold,
                "calibration_scope": "post_acquisition_policy",
            },
        }
        artifact = CalibrationArtifact(
            model_id=model_id,
            model_revision=model_revision,
            target_layer=target_layer,
            scores=(
                CalibrationScore(
                    name=self.score_name,
                    threshold=report.post_threshold,
                    conformal_alpha=self.alpha,
                    direction=self.direction,
                ),
            ),
            eigentruth_version=eigentruth_version,
            steering_policy=steering_policy or SteeringPolicyConfig(),
            warmup_dataset_metadata=warmup_dataset_metadata or {},
            calibration_dataset_metadata=metadata,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            commit_sha=commit_sha,
        )
        return EvidenceAcquisitionCalibrationResult(report=report, artifact=artifact)


def _records(
    records: Sequence[EvidenceAcquisitionCalibrationRecord | Mapping[str, Any]],
    *,
    require_correct: bool = True,
) -> tuple[EvidenceAcquisitionCalibrationRecord, ...]:
    rows = tuple(
        record
        if isinstance(record, EvidenceAcquisitionCalibrationRecord)
        else EvidenceAcquisitionCalibrationRecord.from_dict(record)
        for record in records
    )
    if not rows:
        raise ValueError("records must be non-empty.")
    if require_correct and not any(row.correct for row in rows):
        raise ValueError("records must contain at least one correct response for calibration.")
    return rows


@dataclass(frozen=True)
class _TraceIndex:
    by_fingerprint: Mapping[str, Mapping[str, Any]]
    by_request: Mapping[str, tuple[Mapping[str, Any], ...]]

    @classmethod
    def from_traces(cls, traces: Sequence[Any]) -> "_TraceIndex":
        if not traces:
            raise ValueError("traces must be non-empty.")
        by_fingerprint: dict[str, Mapping[str, Any]] = {}
        by_request: dict[str, list[Mapping[str, Any]]] = {}
        for trace in traces:
            payload = _trace_payload(trace)
            fingerprint = _trace_fingerprint(payload)
            by_fingerprint[fingerprint] = payload
            embedded = _optional_str(payload.get("trace_fingerprint"))
            if embedded is not None:
                by_fingerprint[embedded] = payload
            request_id = _optional_str(payload.get("request_id"))
            if request_id is not None:
                by_request.setdefault(request_id, []).append(payload)
        return cls(
            by_fingerprint=by_fingerprint,
            by_request={key: tuple(value) for key, value in by_request.items()},
        )

    def match(self, feedback: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str]:
        fingerprint = _optional_str(feedback.get("trace_fingerprint"))
        if fingerprint is not None:
            trace = self.by_fingerprint.get(fingerprint)
            if trace is None:
                return None, "trace_fingerprint_not_found"
            return trace, "trace_fingerprint"
        request_id = _optional_str(feedback.get("request_id"))
        if request_id is None:
            return None, "missing_request_id"
        candidates = tuple(self.by_request.get(request_id, ()))
        if not candidates:
            return None, "request_id_not_found"
        if len(candidates) > 1:
            return None, "ambiguous_request_id"
        return candidates[0], "request_id"


def _optional_delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{name} must be a bool or 0/1 label.")


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return number


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not math.isfinite(as_float) or not as_float.is_integer():
        raise ValueError(f"{name} must be a positive integer.")
    number = int(as_float)
    if number < 1:
        raise ValueError(f"{name} must be positive.")
    return number


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if number != number or number in {float("inf"), float("-inf")}:
        raise ValueError(f"{name} must be a finite number.")
    return number


def _threshold_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric and must not be NaN.")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric and must not be NaN.") from exc
    if math.isnan(threshold):
        raise ValueError(f"{name} must be numeric and must not be NaN.")
    return threshold


def _alpha_float(value: Any) -> float:
    alpha = _finite_float(value, name="alpha")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    return alpha


def _unit_interval_float(value: Any, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if not (0.0 <= number <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")
    return number


def _risk_checkpoints(checkpoints: Sequence[int] | None, *, n_records: int) -> tuple[int, ...]:
    n = _positive_int(n_records, name="n_records")
    if checkpoints is None:
        return tuple(range(1, n + 1))
    values: list[int] = []
    for value in checkpoints:
        checkpoint = _positive_int(value, name="checkpoint")
        if checkpoint > n:
            raise ValueError("checkpoints must be within the record count.")
        values.append(checkpoint)
    deduped = tuple(sorted(set(values)))
    if not deduped:
        raise ValueError("checkpoints must be non-empty.")
    return deduped


def _alpha_spending_schedule_name(value: Any) -> str:
    schedule = str(value).strip().lower().replace("_", "-")
    aliases = {
        "linear": "linear",
        "equal": "linear",
        "harmonic": "harmonic",
        "front-loaded": "harmonic",
        "geometric": "geometric",
        "halving": "geometric",
    }
    if schedule not in aliases:
        raise ValueError("schedule must be one of: linear, harmonic, geometric.")
    return aliases[schedule]


def _accepted_by_threshold(score: float, threshold: float, direction: str) -> bool:
    if direction == "higher":
        return score <= threshold
    if direction == "lower":
        return score >= threshold
    raise ValueError("direction must be 'higher' or 'lower'.")


def _binomial_upper_confidence_bound(*, successes: int, total: int, alpha: float) -> float:
    """Return a one-sided Clopper-Pearson-style upper bound."""
    k = _non_negative_int(successes, name="successes")
    n = _positive_int(total, name="total")
    if k > n:
        raise ValueError("successes must not exceed total.")
    alpha_value = _alpha_float(alpha)
    if k == n:
        return 1.0
    low = 0.0
    high = 1.0
    for _ in range(80):
        mid = (low + high) / 2.0
        cdf = _binomial_cdf_leq(k, n, mid)
        if cdf > alpha_value:
            low = mid
        else:
            high = mid
    return high


def _binomial_cdf_leq(k: int, n: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if k >= n else 0.0
    q = 1.0 - probability
    term = q**n
    total = term
    for i in range(1, k + 1):
        term *= (n - i + 1) / i * probability / q
        total += term
    return max(0.0, min(1.0, total))


def _score_name(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("score_name must be non-empty.")
    return text


def _trace_payload(trace: Any) -> Mapping[str, Any]:
    if isinstance(trace, Mapping):
        return trace
    if hasattr(trace, "to_dict"):
        payload = trace.to_dict()
        if isinstance(payload, Mapping):
            return payload
    raise ValueError("trace must be a mapping or expose to_dict().")


def _mapping_payload(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
        if isinstance(payload, Mapping):
            return payload
    raise ValueError("feedback records must be mappings or expose to_dict().")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _post_policy_score(
    trace: Mapping[str, Any],
    key: str,
    *,
    fallback_key: str,
) -> tuple[float, str]:
    candidates = (
        ("metadata.evidence_acquisition.post_score", _nested(trace, "metadata", "evidence_acquisition", "post_score")),
        (
            "metadata.evidence_acquisition.decision.metadata",
            _nested(trace, "metadata", "evidence_acquisition", "decision", "metadata", key),
        ),
        (
            "metadata.evidence_acquisition.decision",
            _nested(trace, "metadata", "evidence_acquisition", "decision", key),
        ),
        (f"risk_decision.diagnostics.{key}", _nested(trace, "risk_decision", "diagnostics", key)),
        (f"risk_decision.diagnostics.{fallback_key}", _nested(trace, "risk_decision", "diagnostics", fallback_key)),
        (f"final_risk_decision.diagnostics.{key}", _event_nested(trace, "final_risk_decision", "diagnostics", key)),
        (
            f"final_risk_decision.diagnostics.{fallback_key}",
            _event_nested(trace, "final_risk_decision", "diagnostics", fallback_key),
        ),
        (f"diagnostics.{key}", _nested(trace, "diagnostics", key)),
        (f"metadata.{key}", _nested(trace, "metadata", key)),
    )
    return _first_finite_score(candidates, name="post_score")


def _pre_policy_score(
    trace: Mapping[str, Any],
    key: str,
    *,
    fallback_key: str,
) -> tuple[float | None, str | None]:
    candidates = (
        ("metadata.evidence_acquisition.pre_score", _nested(trace, "metadata", "evidence_acquisition", "pre_score")),
        (
            "metadata.evidence_acquisition.decision.metadata",
            _nested(trace, "metadata", "evidence_acquisition", "decision", "metadata", key),
        ),
        (
            "metadata.evidence_acquisition.decision",
            _nested(trace, "metadata", "evidence_acquisition", "decision", key),
        ),
        (f"initial_risk_decision.diagnostics.{key}", _event_nested(trace, "initial_risk_decision", "diagnostics", key)),
        (
            f"initial_risk_decision.diagnostics.{fallback_key}",
            _event_nested(trace, "initial_risk_decision", "diagnostics", fallback_key),
        ),
        (f"diagnostics.{key}", _nested(trace, "diagnostics", key)),
        (f"diagnostics.{fallback_key}", _nested(trace, "diagnostics", fallback_key)),
        (f"metadata.{key}", _nested(trace, "metadata", key)),
    )
    for source, value in candidates:
        if value is None:
            continue
        return _finite_float(value, name="pre_score"), source
    return None, None


def _first_finite_score(candidates: Sequence[tuple[str, Any]], *, name: str) -> tuple[float, str]:
    for source, value in candidates:
        if value is None:
            continue
        return _finite_float(value, name=name), source
    raise ValueError(f"{name} was not found in trace.")


def _trace_correctness(trace: Mapping[str, Any], *, correct: bool | int | None) -> tuple[bool, str]:
    if correct is not None:
        return _strict_bool(correct, name="correct"), "argument.correct"
    candidates = (
        ("metadata.correct", _nested(trace, "metadata", "correct")),
        ("metadata.label", _nested(trace, "metadata", "label")),
        ("metadata.feedback_outcome", _nested(trace, "metadata", "feedback_outcome")),
        ("metadata.outcome", _nested(trace, "metadata", "outcome")),
        ("final_answer.correct", _nested(trace, "final_answer", "correct")),
        ("final_answer.answer_correct", _nested(trace, "final_answer", "answer_correct")),
    )
    for source, value in candidates:
        if value is None:
            continue
        if isinstance(value, bool) or value in {0, 1}:
            return _strict_bool(value, name="correct"), source
        return _correct_from_outcome(str(value).strip().lower()), source
    raise ValueError("correct label was not found in trace.")


def _trace_acquisition_action(trace: Mapping[str, Any]) -> tuple[str, str]:
    decision = _mapping(_nested(trace, "metadata", "evidence_acquisition", "decision"))
    action = _optional_str(decision.get("action"))
    if action in {"answer", "acquire", "abstain"}:
        return action, "metadata.evidence_acquisition.decision.action"
    event_action = _optional_str(_event_nested(trace, "evidence_acquisition_decision", "action"))
    if event_action in {"answer", "acquire", "abstain"}:
        return event_action, "events.evidence_acquisition_decision.action"
    risk_action = _nested_str(trace, "risk_decision", "action")
    if risk_action == "accept":
        return "answer", "risk_decision.action"
    if risk_action in {"retrieve", "execute_tool", "rewrite", "steer_regenerate"}:
        return "acquire", "risk_decision.action"
    if risk_action in {"abstain", "clarify"}:
        return "abstain", "risk_decision.action"
    return "answer", "default"


def _trace_retrieved_evidence(trace: Mapping[str, Any]) -> bool:
    for action in _sequence(trace.get("actions")):
        if isinstance(action, Mapping) and str(action.get("action", "")).strip().lower() == "retrieve":
            return True
    for result in _sequence(trace.get("action_results")):
        if isinstance(result, Mapping) and str(result.get("action", "")).strip().lower() == "retrieve":
            return True
    return False


def _correct_from_outcome(outcome: str) -> bool:
    normalized = str(outcome).strip().lower()
    if normalized in {"correct", "unnecessary_block", "true", "1", "pass", "passed"}:
        return True
    if normalized in {
        "incorrect",
        "unsupported",
        "partially_correct",
        "appropriate_block",
        "false",
        "0",
        "fail",
        "failed",
    }:
        return False
    raise ValueError(f"feedback outcome is not a calibration label: {outcome!r}.")


def _event_nested(trace: Mapping[str, Any], event_type: str, *path: str) -> Any:
    for event in _sequence(trace.get("events")):
        if not isinstance(event, Mapping):
            continue
        if event.get("event_type") != event_type:
            continue
        payload = _mapping(event.get("payload"))
        if not path:
            return payload
        return _nested(payload, *path)
    return None


def _nested(payload: Mapping[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _nested_str(payload: Mapping[str, Any], *path: str) -> str | None:
    value = _nested(payload, *path)
    return _optional_str(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _trace_fingerprint(trace: Mapping[str, Any]) -> str:
    embedded = _optional_str(trace.get("trace_fingerprint"))
    if embedded is not None:
        return embedded
    encoded = strict_json_dumps(
        trace,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
