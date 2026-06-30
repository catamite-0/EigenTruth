"""Post-acquisition conformal calibration for evidence-acquisition policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from eigentruth import __version__
from eigentruth.calibration.artifacts import CalibrationArtifact, CalibrationScore, SteeringPolicyConfig
from eigentruth.eval.conformal import (
    ConformalAbstentionReport,
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
            naive_pre_report=(
                None if naive_payload is None else ConformalAbstentionReport.from_dict(naive_payload)
            ),
            post_acquisition_report=ConformalAbstentionReport.from_dict(
                data["post_acquisition_report"]
            ),
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
) -> tuple[EvidenceAcquisitionCalibrationRecord, ...]:
    rows = tuple(
        record
        if isinstance(record, EvidenceAcquisitionCalibrationRecord)
        else EvidenceAcquisitionCalibrationRecord.from_dict(record)
        for record in records
    )
    if not rows:
        raise ValueError("records must be non-empty.")
    if not any(row.correct for row in rows):
        raise ValueError("records must contain at least one correct response for calibration.")
    return rows


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
