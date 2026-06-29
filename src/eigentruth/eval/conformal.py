"""EigenTruth conformal — split-conformal 校准 / Split-conformal calibration.

把任意异常分数（如马氏距离）转换为具有有限样本保证的 p 值与报警阈值。
Turns any anomaly score (e.g. Mahalanobis distance) into p-values and alarm
thresholds with finite-sample guarantees.

前提 / Assumption: 校准分数与测试点可交换（来自同一"正常"总体）。
The calibration scores and the test point are exchangeable (drawn from the
same "normal" population).

保证 / Guarantee: 对可交换的测试点，P(p_value <= alpha) <= alpha —— 即按
`p <= alpha` 报警的误报率不超过 alpha；同理 `score > conformal_threshold(alpha)`
的误报率不超过 alpha。(Vovk et al.; Angelopoulos & Bates 2023 tutorial.)
For an exchangeable test point, P(p-value <= alpha) <= alpha, so flagging at
`p <= alpha` (equivalently `score > conformal_threshold(alpha)`) has a
false-alarm rate of at most alpha.

约定 / Convention: 分数越高越异常 / higher score = more anomalous.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Sequence, Union

import torch
from torch import Tensor

from eigentruth.json_utils import to_jsonable

ArrayLike = Union[Tensor, Sequence[float]]


@dataclass(frozen=True)
class AdaptiveScoreTransform:
    """Feature-adjust a native score into a higher-is-anomalous score.

    This is a dependency-free scoring primitive for adaptive conformal workflows:
    the base score is first converted into anomaly direction, then caller-provided
    feature values add a deterministic inflation term.
    """

    feature_weights: Mapping[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    direction: str = "higher"

    def __post_init__(self) -> None:
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        intercept = _finite_float(self.intercept, name="intercept")
        weights = {
            str(name): _finite_float(weight, name=f"feature_weights.{name}")
            for name, weight in self.feature_weights.items()
        }
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "feature_weights", weights)

    def transform(
        self,
        scores: ArrayLike,
        feature_values: Mapping[str, ArrayLike] | None = None,
    ) -> Tensor:
        """Return adjusted scores where higher means more anomalous."""
        return adaptive_anomaly_scores(
            scores,
            feature_values=feature_values,
            feature_weights=self.feature_weights,
            intercept=self.intercept,
            direction=self.direction,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable transform description."""
        return {
            "feature_weights": dict(self.feature_weights),
            "intercept": self.intercept,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AdaptiveScoreTransform":
        """Build a transform from JSON-like data."""
        raw_weights = data.get("feature_weights", {})
        if not isinstance(raw_weights, Mapping):
            raise ValueError("feature_weights must be a mapping.")
        return cls(
            feature_weights={str(name): weight for name, weight in raw_weights.items()},
            intercept=data.get("intercept", 0.0),
            direction=str(data.get("direction", "higher")),
        )


@dataclass(frozen=True)
class ConformalAbstentionDecision:
    """Runtime participation decision from a conformal abstention threshold."""

    participate: bool
    score: float
    threshold: float
    direction: str = "higher"
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        score = _finite_float(self.score, name="score")
        threshold = _threshold_float(self.threshold, name="threshold")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def action(self) -> str:
        """Return ``participate`` or ``abstain`` for compact policy logs."""
        return "participate" if self.participate else "abstain"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable decision payload."""
        return {
            "participate": self.participate,
            "action": self.action,
            "score": self.score,
            "threshold": self.threshold,
            "direction": self.direction,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConformalAbstentionDecision":
        """Build a decision from JSON-like data."""
        return cls(
            participate=bool(data["participate"]),
            score=data["score"],
            threshold=data["threshold"],
            direction=str(data.get("direction", "higher")),
            reason=str(data.get("reason", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class MultipleTestingSignalResult:
    """One signal's contribution to a conformal multiple-testing decision."""

    name: str
    score: float
    direction: str
    p_value: float
    rank: int
    threshold: float
    rejected: bool
    calibration_count: int
    method: str
    rejection_cutoff: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        score = _finite_float(self.score, name="score")
        p_value = _unit_interval_float(self.p_value, name="p_value")
        threshold = _unit_interval_float(self.threshold, name="threshold")
        rank = int(self.rank)
        if rank < 1:
            raise ValueError("rank must be positive.")
        calibration_count = int(self.calibration_count)
        if calibration_count < 1:
            raise ValueError("calibration_count must be positive.")
        cutoff = None
        if self.rejection_cutoff is not None:
            cutoff = _unit_interval_float(self.rejection_cutoff, name="rejection_cutoff")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "p_value", p_value)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "rejected", bool(self.rejected))
        object.__setattr__(self, "calibration_count", calibration_count)
        object.__setattr__(self, "method", _multiple_testing_method(self.method))
        object.__setattr__(self, "rejection_cutoff", cutoff)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable signal result."""
        return {
            "name": self.name,
            "score": self.score,
            "direction": self.direction,
            "p_value": self.p_value,
            "rank": self.rank,
            "threshold": self.threshold,
            "rejected": self.rejected,
            "calibration_count": self.calibration_count,
            "method": self.method,
            "rejection_cutoff": self.rejection_cutoff,
        }


@dataclass(frozen=True)
class MultipleTestingHallucinationReport:
    """Global hallucination decision from several conformal signal p-values.

    The report controls one false-alarm budget across several mixed-direction
    signals. ``method="by"`` is the conservative default for dependent signals;
    ``method="bh"`` is less conservative, and ``method="bonferroni"`` uses a
    simple per-signal alpha split.
    """

    alpha: float
    method: str
    correction: float
    rejected: bool
    rejected_count: int
    min_p_value: float
    signals: tuple[MultipleTestingSignalResult, ...]
    rejection_cutoff: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        alpha = _alpha_float(self.alpha)
        method = _multiple_testing_method(self.method)
        correction = _finite_float(self.correction, name="correction")
        if correction <= 0.0:
            raise ValueError("correction must be positive.")
        rejected_count = int(self.rejected_count)
        if rejected_count < 0:
            raise ValueError("rejected_count must be non-negative.")
        min_p_value = _unit_interval_float(self.min_p_value, name="min_p_value")
        signals = tuple(self.signals)
        if not signals:
            raise ValueError("signals must be non-empty.")
        cutoff = None
        if self.rejection_cutoff is not None:
            cutoff = _unit_interval_float(self.rejection_cutoff, name="rejection_cutoff")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "correction", correction)
        object.__setattr__(self, "rejected", bool(self.rejected))
        object.__setattr__(self, "rejected_count", rejected_count)
        object.__setattr__(self, "min_p_value", min_p_value)
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "rejection_cutoff", cutoff)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def rejected_signal_names(self) -> tuple[str, ...]:
        """Return signal names rejected by the global multiple-testing rule."""
        return tuple(signal.name for signal in self.signals if signal.rejected)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "alpha": self.alpha,
            "method": self.method,
            "correction": self.correction,
            "rejected": self.rejected,
            "rejected_count": self.rejected_count,
            "rejected_signal_names": list(self.rejected_signal_names),
            "min_p_value": self.min_p_value,
            "rejection_cutoff": self.rejection_cutoff,
            "signals": [signal.to_dict() for signal in self.signals],
            "metadata": to_jsonable(self.metadata),
        }


@dataclass(frozen=True)
class SequentialConformalStepResult:
    """One step in a sequential alpha-spending conformal monitor."""

    step: int
    p_value: float
    alpha_spent: float
    cumulative_alpha_spent: float
    rejected: bool
    score: float | None = None
    direction: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step = int(self.step)
        if step < 1:
            raise ValueError("step must be positive.")
        p_value = _unit_interval_float(self.p_value, name="p_value")
        alpha_spent = _unit_interval_float(self.alpha_spent, name="alpha_spent")
        cumulative_alpha_spent = _unit_interval_float(
            self.cumulative_alpha_spent,
            name="cumulative_alpha_spent",
        )
        if alpha_spent <= 0.0:
            raise ValueError("alpha_spent must be positive.")
        if cumulative_alpha_spent < alpha_spent:
            raise ValueError("cumulative_alpha_spent must be at least alpha_spent.")
        if self.score is not None:
            object.__setattr__(self, "score", _finite_float(self.score, name="score"))
        if self.direction is not None and self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "p_value", p_value)
        object.__setattr__(self, "alpha_spent", alpha_spent)
        object.__setattr__(self, "cumulative_alpha_spent", cumulative_alpha_spent)
        object.__setattr__(self, "rejected", bool(self.rejected))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable step payload."""
        return {
            "step": self.step,
            "p_value": self.p_value,
            "alpha_spent": self.alpha_spent,
            "cumulative_alpha_spent": self.cumulative_alpha_spent,
            "rejected": self.rejected,
            "score": self.score,
            "direction": self.direction,
            "metadata": to_jsonable(self.metadata),
        }


@dataclass(frozen=True)
class SequentialConformalReport:
    """Alpha-spending monitor over a sequence of conformal p-values.

    The report controls the total alarm budget over a finite sequence by
    spending per-step alpha values whose sum is at most ``alpha``. This is a
    conservative monitor-first primitive for sessions, batches, or repeated
    release checks; it makes no independence assumption beyond each p-value's
    conformal super-uniformity.
    """

    alpha: float
    schedule: str
    horizon: int
    alpha_spent_total: float
    rejected: bool
    rejected_count: int
    steps: tuple[SequentialConformalStepResult, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        alpha = _alpha_float(self.alpha)
        schedule = _alpha_spending_schedule_name(self.schedule)
        horizon = _positive_int(self.horizon, name="horizon")
        alpha_spent_total = _unit_interval_float(
            self.alpha_spent_total,
            name="alpha_spent_total",
        )
        if alpha_spent_total > alpha + 1e-12:
            raise ValueError("alpha_spent_total must not exceed alpha.")
        rejected_count = int(self.rejected_count)
        if rejected_count < 0:
            raise ValueError("rejected_count must be non-negative.")
        steps = tuple(self.steps)
        if len(steps) > horizon:
            raise ValueError("steps cannot exceed horizon.")
        expected_steps = tuple(range(1, len(steps) + 1))
        if tuple(step.step for step in steps) != expected_steps:
            raise ValueError("step numbers must be contiguous starting at 1.")
        if rejected_count != sum(1 for step in steps if step.rejected):
            raise ValueError("rejected_count must match rejected steps.")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "alpha_spent_total", alpha_spent_total)
        object.__setattr__(self, "rejected", bool(self.rejected))
        object.__setattr__(self, "rejected_count", rejected_count)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def remaining_alpha(self) -> float:
        """Return unspent alpha budget."""
        return max(0.0, self.alpha - self.alpha_spent_total)

    @property
    def rejected_steps(self) -> tuple[int, ...]:
        """Return step numbers rejected by the sequential monitor."""
        return tuple(step.step for step in self.steps if step.rejected)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable sequential monitor report."""
        return {
            "alpha": self.alpha,
            "schedule": self.schedule,
            "horizon": self.horizon,
            "alpha_spent_total": self.alpha_spent_total,
            "remaining_alpha": self.remaining_alpha,
            "rejected": self.rejected,
            "rejected_count": self.rejected_count,
            "rejected_steps": list(self.rejected_steps),
            "steps": [step.to_dict() for step in self.steps],
            "metadata": to_jsonable(self.metadata),
        }


@dataclass(frozen=True)
class ConformalAbstentionReport:
    """Finite-sample participation and selective-correctness report.

    ``direction`` describes which side of the uncertainty score means "less
    reliable." For ``higher``, scores above ``threshold`` abstain. For ``lower``,
    scores below ``threshold`` abstain.
    """

    threshold: float
    alpha: float
    direction: str = "higher"
    n_calibration: int = 0
    n_correct: int = 0
    retained_count: int = 0
    correct_retained_count: int = 0
    abstained_count: int = 0
    empirical_base_accuracy: float = 0.0
    empirical_participation_rate: float = 0.0
    empirical_abstention_rate: float = 0.0
    empirical_selective_accuracy: float | None = None
    correct_retention_rate: float = 0.0
    correct_retention_lower_bound: float = 0.0
    participation_upper_bound: float = 0.0
    conditional_correctness_lower_bound: float = 0.0
    score_name: str | None = None

    def __post_init__(self) -> None:
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        threshold = _threshold_float(self.threshold, name="threshold")
        alpha = _alpha_float(self.alpha)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "alpha", alpha)
        for name in (
            "n_calibration",
            "n_correct",
            "retained_count",
            "correct_retained_count",
            "abstained_count",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)
        for name in (
            "empirical_base_accuracy",
            "empirical_participation_rate",
            "empirical_abstention_rate",
            "correct_retention_rate",
            "correct_retention_lower_bound",
            "participation_upper_bound",
            "conditional_correctness_lower_bound",
        ):
            value = _unit_interval_float(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        if self.empirical_selective_accuracy is not None:
            value = _unit_interval_float(
                self.empirical_selective_accuracy,
                name="empirical_selective_accuracy",
            )
            object.__setattr__(self, "empirical_selective_accuracy", value)
        if self.score_name is not None:
            object.__setattr__(self, "score_name", str(self.score_name))

    def should_participate(self, score: float) -> bool:
        """Return whether a runtime score is inside the retained region."""
        value = _finite_float(score, name="score")
        if self.direction == "higher":
            return value <= self.threshold
        return value >= self.threshold

    def decide(
        self,
        score: float,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ConformalAbstentionDecision:
        """Return a structured runtime abstention decision."""
        value = _finite_float(score, name="score")
        participate = self.should_participate(value)
        operator = "<=" if self.direction == "higher" else ">="
        reason = (
            f"uncertainty score {value:.6g} {operator} conformal abstention "
            f"threshold {self.threshold:.6g}"
            if participate
            else (
                f"uncertainty score {value:.6g} outside conformal abstention "
                f"threshold {self.threshold:.6g}"
            )
        )
        return ConformalAbstentionDecision(
            participate=participate,
            score=value,
            threshold=self.threshold,
            direction=self.direction,
            reason=reason,
            metadata={
                "alpha": self.alpha,
                "score_name": self.score_name,
                "conditional_correctness_lower_bound": self.conditional_correctness_lower_bound,
                **({} if metadata is None else dict(metadata)),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable abstention report."""
        return {
            "threshold": self.threshold,
            "alpha": self.alpha,
            "direction": self.direction,
            "n_calibration": self.n_calibration,
            "n_correct": self.n_correct,
            "retained_count": self.retained_count,
            "correct_retained_count": self.correct_retained_count,
            "abstained_count": self.abstained_count,
            "empirical_base_accuracy": self.empirical_base_accuracy,
            "empirical_participation_rate": self.empirical_participation_rate,
            "empirical_abstention_rate": self.empirical_abstention_rate,
            "empirical_selective_accuracy": self.empirical_selective_accuracy,
            "correct_retention_rate": self.correct_retention_rate,
            "correct_retention_lower_bound": self.correct_retention_lower_bound,
            "participation_upper_bound": self.participation_upper_bound,
            "conditional_correctness_lower_bound": self.conditional_correctness_lower_bound,
            "score_name": self.score_name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConformalAbstentionReport":
        """Build an abstention report from JSON-like data."""
        return cls(
            threshold=data["threshold"],
            alpha=data["alpha"],
            direction=str(data.get("direction", "higher")),
            n_calibration=int(data.get("n_calibration", 0)),
            n_correct=int(data.get("n_correct", 0)),
            retained_count=int(data.get("retained_count", 0)),
            correct_retained_count=int(data.get("correct_retained_count", 0)),
            abstained_count=int(data.get("abstained_count", 0)),
            empirical_base_accuracy=float(data.get("empirical_base_accuracy", 0.0)),
            empirical_participation_rate=float(data.get("empirical_participation_rate", 0.0)),
            empirical_abstention_rate=float(data.get("empirical_abstention_rate", 0.0)),
            empirical_selective_accuracy=(
                None
                if data.get("empirical_selective_accuracy") is None
                else float(data["empirical_selective_accuracy"])
            ),
            correct_retention_rate=float(data.get("correct_retention_rate", 0.0)),
            correct_retention_lower_bound=float(data.get("correct_retention_lower_bound", 0.0)),
            participation_upper_bound=float(data.get("participation_upper_bound", 0.0)),
            conditional_correctness_lower_bound=float(
                data.get("conditional_correctness_lower_bound", 0.0)
            ),
            score_name=None if data.get("score_name") is None else str(data["score_name"]),
        )


ABSTENTION_COMPARISON_METRICS = (
    "conditional_correctness_lower_bound",
    "empirical_selective_accuracy",
    "empirical_participation_rate",
    "correct_retention_lower_bound",
    "correct_retention_rate",
)


@dataclass(frozen=True)
class ConformalAbstentionComparisonCandidate:
    """One ranked score candidate in a conformal abstention comparison."""

    rank: int
    score_name: str
    direction: str
    selection_metric: str
    selection_value: float | None
    report: ConformalAbstentionReport

    def __post_init__(self) -> None:
        rank = int(self.rank)
        if rank < 1:
            raise ValueError("rank must be >= 1.")
        score_name = str(self.score_name)
        if not score_name:
            raise ValueError("score_name must be non-empty.")
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        if self.selection_metric not in ABSTENTION_COMPARISON_METRICS:
            raise ValueError(
                "selection_metric must be one of "
                f"{ABSTENTION_COMPARISON_METRICS}."
            )
        if self.selection_value is not None:
            value = _unit_interval_float(self.selection_value, name="selection_value")
            object.__setattr__(self, "selection_value", value)
        if not isinstance(self.report, ConformalAbstentionReport):
            raise ValueError("report must be a ConformalAbstentionReport.")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "score_name", score_name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate payload."""
        return {
            "rank": self.rank,
            "score_name": self.score_name,
            "direction": self.direction,
            "selection_metric": self.selection_metric,
            "selection_value": self.selection_value,
            "report": self.report.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ConformalAbstentionComparisonCandidate":
        """Build a candidate from JSON-like data."""
        raw_report = data.get("report")
        if not isinstance(raw_report, Mapping):
            raise ValueError("candidate report must be a mapping.")
        return cls(
            rank=int(data["rank"]),
            score_name=str(data["score_name"]),
            direction=str(data.get("direction", "higher")),
            selection_metric=str(
                data.get("selection_metric", "conditional_correctness_lower_bound")
            ),
            selection_value=(
                None if data.get("selection_value") is None else float(data["selection_value"])
            ),
            report=ConformalAbstentionReport.from_dict(raw_report),
        )


@dataclass(frozen=True)
class ConformalAbstentionComparisonReport:
    """Rank several abstention signals under a shared correctness target."""

    alpha: float
    best_by: str
    candidates: tuple[ConformalAbstentionComparisonCandidate, ...]

    def __post_init__(self) -> None:
        alpha = _alpha_float(self.alpha)
        if self.best_by not in ABSTENTION_COMPARISON_METRICS:
            raise ValueError(f"best_by must be one of {ABSTENTION_COMPARISON_METRICS}.")
        candidates = tuple(self.candidates)
        ranks = tuple(candidate.rank for candidate in candidates)
        if ranks != tuple(range(1, len(candidates) + 1)):
            raise ValueError("candidate ranks must be contiguous starting at 1.")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "candidates", candidates)

    @property
    def recommended(self) -> ConformalAbstentionComparisonCandidate | None:
        """Return the top-ranked candidate, if any."""
        return None if not self.candidates else self.candidates[0]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable comparison report."""
        recommended = self.recommended
        return {
            "alpha": self.alpha,
            "best_by": self.best_by,
            "candidate_count": len(self.candidates),
            "recommended": None if recommended is None else recommended.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConformalAbstentionComparisonReport":
        """Build a comparison report from JSON-like data."""
        raw_candidates = data.get("candidates", ())
        if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, str):
            raise ValueError("candidates must be a sequence.")
        return cls(
            alpha=data["alpha"],
            best_by=str(data.get("best_by", "conditional_correctness_lower_bound")),
            candidates=tuple(
                ConformalAbstentionComparisonCandidate.from_dict(candidate)
                for candidate in raw_candidates
            ),
        )


@dataclass(frozen=True)
class ConformalAbstentionReleaseGateResult:
    """Release-gate verdict for a selected conformal abstention report."""

    passed: bool
    blocking_reasons: tuple[str, ...]
    selected_score_name: str | None
    metrics: Mapping[str, float | None]
    thresholds: Mapping[str, float]
    source: str = "conformal_abstention_report"
    candidate_count: int | None = None
    selected_report: ConformalAbstentionReport | None = None

    def __post_init__(self) -> None:
        reasons = tuple(str(reason) for reason in self.blocking_reasons)
        metrics = dict(self.metrics)
        thresholds = dict(self.thresholds)
        for name, value in metrics.items():
            if value is not None:
                metrics[name] = _unit_interval_float(value, name=f"metrics.{name}")
        for name, value in thresholds.items():
            thresholds[name] = _unit_interval_float(value, name=f"thresholds.{name}")
        candidate_count = self.candidate_count
        if candidate_count is not None:
            candidate_count = int(candidate_count)
            if candidate_count < 0:
                raise ValueError("candidate_count must be non-negative.")
        if self.selected_report is not None and not isinstance(
            self.selected_report,
            ConformalAbstentionReport,
        ):
            raise ValueError("selected_report must be a ConformalAbstentionReport.")
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "blocking_reasons", reasons)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "candidate_count", candidate_count)
        if self.selected_score_name is not None:
            object.__setattr__(self, "selected_score_name", str(self.selected_score_name))

    @property
    def status(self) -> str:
        """Return ``passed`` or ``blocked`` for release workflows."""
        return "passed" if self.passed else "blocked"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable release-gate payload."""
        return {
            "passed": self.passed,
            "status": self.status,
            "blocking_reasons": list(self.blocking_reasons),
            "selected_score_name": self.selected_score_name,
            "metrics": dict(self.metrics),
            "thresholds": dict(self.thresholds),
            "source": self.source,
            "candidate_count": self.candidate_count,
            "selected_report": (
                None if self.selected_report is None else self.selected_report.to_dict()
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ConformalAbstentionReleaseGateResult":
        """Build a release-gate result from JSON-like data."""
        raw_report = data.get("selected_report")
        return cls(
            passed=bool(data["passed"]),
            blocking_reasons=tuple(str(reason) for reason in data.get("blocking_reasons", ())),
            selected_score_name=(
                None
                if data.get("selected_score_name") is None
                else str(data["selected_score_name"])
            ),
            metrics=dict(data.get("metrics", {})),
            thresholds=dict(data.get("thresholds", {})),
            source=str(data.get("source", "conformal_abstention_report")),
            candidate_count=(
                None if data.get("candidate_count") is None else int(data["candidate_count"])
            ),
            selected_report=(
                None
                if raw_report is None
                else ConformalAbstentionReport.from_dict(raw_report)
            ),
        )


@dataclass(frozen=True)
class ConformalAbstentionReleaseGate:
    """Fail-closed promotion gate for conformal abstention candidates."""

    min_conditional_correctness_lower_bound: float = 0.8
    max_abstention_rate: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_conditional_correctness_lower_bound",
            _unit_interval_float(
                self.min_conditional_correctness_lower_bound,
                name="min_conditional_correctness_lower_bound",
            ),
        )
        object.__setattr__(
            self,
            "max_abstention_rate",
            _unit_interval_float(self.max_abstention_rate, name="max_abstention_rate"),
        )

    def evaluate(
        self,
        report: (
            ConformalAbstentionReport
            | ConformalAbstentionComparisonCandidate
            | ConformalAbstentionComparisonReport
            | Mapping[str, Any]
        ),
    ) -> ConformalAbstentionReleaseGateResult:
        """Evaluate a report, candidate, or comparison report against release thresholds."""
        selected = _select_abstention_report_for_release_gate(report)
        if selected is None:
            thresholds = {
                "min_conditional_correctness_lower_bound": (
                    self.min_conditional_correctness_lower_bound
                ),
                "max_abstention_rate": self.max_abstention_rate,
            }
            return ConformalAbstentionReleaseGateResult(
                passed=False,
                blocking_reasons=("abstention comparison report has no candidates",),
                selected_score_name=None,
                metrics={},
                thresholds=thresholds,
                source="conformal_abstention_comparison_report",
                candidate_count=0,
                selected_report=None,
            )
        selected_report, source, candidate_count = selected
        metrics = {
            "conditional_correctness_lower_bound": (
                selected_report.conditional_correctness_lower_bound
            ),
            "empirical_abstention_rate": selected_report.empirical_abstention_rate,
            "empirical_participation_rate": selected_report.empirical_participation_rate,
            "empirical_selective_accuracy": selected_report.empirical_selective_accuracy,
            "correct_retention_lower_bound": selected_report.correct_retention_lower_bound,
            "correct_retention_rate": selected_report.correct_retention_rate,
        }
        thresholds = {
            "min_conditional_correctness_lower_bound": (
                self.min_conditional_correctness_lower_bound
            ),
            "max_abstention_rate": self.max_abstention_rate,
        }
        blocking_reasons: list[str] = []
        if (
            selected_report.conditional_correctness_lower_bound
            < self.min_conditional_correctness_lower_bound
        ):
            blocking_reasons.append(
                "conditional_correctness_lower_bound "
                f"{selected_report.conditional_correctness_lower_bound:.6g} "
                "is below required minimum "
                f"{self.min_conditional_correctness_lower_bound:.6g}"
            )
        if selected_report.empirical_abstention_rate > self.max_abstention_rate:
            blocking_reasons.append(
                "empirical_abstention_rate "
                f"{selected_report.empirical_abstention_rate:.6g} "
                "exceeds maximum "
                f"{self.max_abstention_rate:.6g}"
            )

        return ConformalAbstentionReleaseGateResult(
            passed=not blocking_reasons,
            blocking_reasons=tuple(blocking_reasons),
            selected_score_name=selected_report.score_name,
            metrics=metrics,
            thresholds=thresholds,
            source=source,
            candidate_count=candidate_count,
            selected_report=selected_report,
        )

    def to_dict(self) -> dict[str, float]:
        """Return release-gate threshold configuration."""
        return {
            "min_conditional_correctness_lower_bound": (
                self.min_conditional_correctness_lower_bound
            ),
            "max_abstention_rate": self.max_abstention_rate,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConformalAbstentionReleaseGate":
        """Build release-gate thresholds from JSON-like data."""
        return cls(
            min_conditional_correctness_lower_bound=data.get(
                "min_conditional_correctness_lower_bound",
                0.8,
            ),
            max_abstention_rate=data.get("max_abstention_rate", 0.5),
        )


def conformal_pvalues(calib_scores: ArrayLike, test_scores: ArrayLike) -> Tensor:
    """计算每个测试分数的保守共形 p 值。
    Conservative split-conformal p-value for each test score.

    p_i = (1 + #{calib >= s_i}) / (n_calib + 1)

    平局计入 >=（保守方向）。p 值落在 (0, 1]。
    Ties count toward >= (the conservative direction). P-values lie in (0, 1].

    Args:
        calib_scores: 校准分数（"正常"总体）/ calibration scores, shape [n_calib].
        test_scores: 测试分数 / test scores, shape [n_test].

    Returns:
        p 值张量 / p-value tensor (float64), shape [n_test].
    """
    calib = _finite_flat_tensor(calib_scores, name="calibration scores")
    test = _finite_flat_tensor(test_scores, name="test scores")
    if calib.numel() == 0:
        raise ValueError("calibration scores must be non-empty.")

    calib_sorted, _ = torch.sort(calib)
    # searchsorted(right=False) 给出 #{calib < s}，故 #{calib >= s} = n - idx
    idx = torch.searchsorted(calib_sorted, test, right=False)
    n_ge = calib.numel() - idx
    return (1.0 + n_ge.to(torch.float64)) / (calib.numel() + 1.0)


def directional_conformal_pvalues(
    calib_scores: ArrayLike,
    test_scores: ArrayLike,
    direction: str,
) -> Tensor:
    """Return conformal p-values for native scores with explicit anomaly direction."""
    if direction == "higher":
        return conformal_pvalues(calib_scores, test_scores)
    if direction == "lower":
        calib = -_finite_flat_tensor(calib_scores, name="calibration scores")
        test = -_finite_flat_tensor(test_scores, name="test scores")
        return conformal_pvalues(calib, test)
    raise ValueError("direction must be 'higher' or 'lower'.")


def multiple_testing_conformal_report(
    calibration_scores_by_signal: Mapping[str, ArrayLike],
    test_scores: Mapping[str, float],
    *,
    alpha: float,
    directions: Mapping[str, str] | None = None,
    method: str = "by",
    metadata: Mapping[str, Any] | None = None,
) -> MultipleTestingHallucinationReport:
    """Combine several directional conformal p-values under one false-alarm budget.

    Args:
        calibration_scores_by_signal: Per-signal calibration scores from the
            normal/correct population.
        test_scores: One native score per signal for the runtime item.
        alpha: Global false-alarm budget in ``(0, 1)``.
        directions: Optional per-signal anomaly direction. Missing signals
            default to ``"higher"``.
        method: ``"by"`` (default), ``"bh"``, or ``"bonferroni"``.
        metadata: Optional JSON-ready context copied into the report.

    Returns:
        A JSON-serializable global report with per-signal p-values and rejected
        signal names.
    """
    alpha_value = _alpha_float(alpha)
    method_name = _multiple_testing_method(method)
    calibration_items = tuple(
        (str(name), values) for name, values in calibration_scores_by_signal.items()
    )
    signal_names = tuple(name for name, _ in calibration_items)
    if not signal_names:
        raise ValueError("calibration_scores_by_signal must be non-empty.")
    if len(set(signal_names)) != len(signal_names):
        raise ValueError("signal names must be unique after string conversion.")
    calibration_by_name = dict(calibration_items)

    test_items = tuple((str(name), value) for name, value in test_scores.items())
    test_names = {name for name, _ in test_items}
    if len(test_names) != len(test_items):
        raise ValueError("test score names must be unique after string conversion.")
    test_by_name = dict(test_items)
    calibration_names = set(signal_names)
    missing = sorted(calibration_names - test_names)
    extra = sorted(test_names - calibration_names)
    if missing or extra:
        raise ValueError(
            "test_scores must contain exactly the calibration signals "
            f"(missing={missing}, extra={extra})."
        )

    raw_directions = {} if directions is None else {str(name): str(value) for name, value in directions.items()}
    extra_directions = sorted(set(raw_directions.keys()) - calibration_names)
    if extra_directions:
        raise ValueError(f"directions contains unknown signals: {extra_directions}.")

    prelim: list[dict[str, Any]] = []
    for name in signal_names:
        direction = raw_directions.get(name, "higher")
        if direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        score = _finite_float(test_by_name[name], name=f"test_scores.{name}")
        calibration = _finite_flat_tensor(
            calibration_by_name[name],
            name=f"calibration_scores_by_signal.{name}",
        )
        if calibration.numel() == 0:
            raise ValueError(f"calibration scores for signal '{name}' must be non-empty.")
        p_value = float(
            directional_conformal_pvalues(
                calibration,
                torch.tensor([score], dtype=torch.float64),
                direction,
            )[0].item()
        )
        prelim.append(
            {
                "name": name,
                "score": score,
                "direction": direction,
                "p_value": p_value,
                "calibration_count": int(calibration.numel()),
            }
        )

    ordered = sorted(prelim, key=lambda item: (float(item["p_value"]), str(item["name"])))
    m = len(ordered)
    correction = _multiple_testing_correction(method_name, m)
    thresholds: dict[str, float] = {}
    ranks: dict[str, int] = {}
    cutoff: float | None = None

    for rank, item in enumerate(ordered, start=1):
        name = str(item["name"])
        threshold = _multiple_testing_rank_threshold(method_name, alpha_value, rank, m, correction)
        thresholds[name] = threshold
        ranks[name] = rank
        if float(item["p_value"]) <= threshold:
            cutoff = float(item["p_value"])

    rejected_names = {
        str(item["name"])
        for item in ordered
        if cutoff is not None and float(item["p_value"]) <= cutoff
    }

    signals = tuple(
        MultipleTestingSignalResult(
            name=str(item["name"]),
            score=float(item["score"]),
            direction=str(item["direction"]),
            p_value=float(item["p_value"]),
            rank=ranks[str(item["name"])],
            threshold=thresholds[str(item["name"])],
            rejected=str(item["name"]) in rejected_names,
            calibration_count=int(item["calibration_count"]),
            method=method_name,
            rejection_cutoff=cutoff,
        )
        for item in sorted(ordered, key=lambda item: ranks[str(item["name"])])
    )

    return MultipleTestingHallucinationReport(
        alpha=alpha_value,
        method=method_name,
        correction=correction,
        rejected=bool(rejected_names),
        rejected_count=len(rejected_names),
        min_p_value=float(ordered[0]["p_value"]),
        signals=signals,
        rejection_cutoff=cutoff,
        metadata={} if metadata is None else dict(metadata),
    )


def alpha_spending_schedule(
    alpha: float,
    horizon: int,
    *,
    schedule: str = "harmonic",
) -> tuple[float, ...]:
    """Return per-step alpha budgets whose sum is at most ``alpha``.

    Supported schedules:

    - ``linear``: equal spending across the finite horizon.
    - ``harmonic``: front-loaded ``1 / t`` spending, normalized over horizon.
    - ``geometric``: conservative ``alpha / 2**t`` spending, leaving a small
      reserve for the finite horizon and matching the infinite-tail intuition.
    """
    alpha_value = _alpha_float(alpha)
    horizon_value = _positive_int(horizon, name="horizon")
    schedule_name = _alpha_spending_schedule_name(schedule)
    if schedule_name == "linear":
        return tuple(alpha_value / horizon_value for _ in range(horizon_value))
    if schedule_name == "harmonic":
        normalizer = sum(1.0 / step for step in range(1, horizon_value + 1))
        return tuple(alpha_value * (1.0 / step) / normalizer for step in range(1, horizon_value + 1))
    return tuple(alpha_value / (2.0**step) for step in range(1, horizon_value + 1))


def sequential_pvalue_monitor(
    p_values: ArrayLike,
    *,
    alpha: float,
    schedule: str = "harmonic",
    metadata: Mapping[str, Any] | None = None,
) -> SequentialConformalReport:
    """Apply alpha spending to a sequence of already-calibrated p-values."""
    p_tensor = _pvalue_flat_tensor(p_values, name="p_values")
    if p_tensor.numel() == 0:
        raise ValueError("p_values must be non-empty.")
    spends = alpha_spending_schedule(alpha, int(p_tensor.numel()), schedule=schedule)
    steps: list[SequentialConformalStepResult] = []
    cumulative = 0.0
    for index, (p_value, alpha_spent) in enumerate(zip(p_tensor.tolist(), spends, strict=True), start=1):
        cumulative += float(alpha_spent)
        steps.append(
            SequentialConformalStepResult(
                step=index,
                p_value=float(p_value),
                alpha_spent=float(alpha_spent),
                cumulative_alpha_spent=cumulative,
                rejected=float(p_value) <= float(alpha_spent),
            )
        )
    return SequentialConformalReport(
        alpha=alpha,
        schedule=schedule,
        horizon=int(p_tensor.numel()),
        alpha_spent_total=cumulative,
        rejected=any(step.rejected for step in steps),
        rejected_count=sum(1 for step in steps if step.rejected),
        steps=tuple(steps),
        metadata={} if metadata is None else dict(metadata),
    )


def sequential_conformal_monitor(
    calib_scores: ArrayLike,
    test_scores: ArrayLike,
    *,
    alpha: float,
    direction: str = "higher",
    schedule: str = "harmonic",
    metadata: Mapping[str, Any] | None = None,
) -> SequentialConformalReport:
    """Run an alpha-spending monitor over native conformal anomaly scores."""
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    scores = _finite_flat_tensor(test_scores, name="test_scores")
    p_values = directional_conformal_pvalues(calib_scores, scores, direction)
    report = sequential_pvalue_monitor(
        p_values,
        alpha=alpha,
        schedule=schedule,
        metadata=metadata,
    )
    steps = tuple(
        SequentialConformalStepResult(
            step=step.step,
            p_value=step.p_value,
            alpha_spent=step.alpha_spent,
            cumulative_alpha_spent=step.cumulative_alpha_spent,
            rejected=step.rejected,
            score=float(score),
            direction=direction,
            metadata=step.metadata,
        )
        for step, score in zip(report.steps, scores.tolist(), strict=True)
    )
    return SequentialConformalReport(
        alpha=report.alpha,
        schedule=report.schedule,
        horizon=report.horizon,
        alpha_spent_total=report.alpha_spent_total,
        rejected=report.rejected,
        rejected_count=report.rejected_count,
        steps=steps,
        metadata=dict(report.metadata),
    )


def conformal_threshold(calib_scores: ArrayLike, alpha: float) -> float:
    """给定误报预算 alpha，返回报警阈值 t。
    Alarm threshold t for a false-alarm budget alpha.

    对可交换测试点，P(score > t) <= alpha。t 取校准分数的第
    ceil((n+1)(1-alpha)) 个次序统计量；当该阶数超过 n（校准样本太少，
    无法支撑该置信水平）时返回 +inf（永不报警）。
    For an exchangeable test point, P(score > t) <= alpha. t is the
    ceil((n+1)(1-alpha))-th order statistic of the calibration scores; if that
    rank exceeds n (too few calibration samples for this alpha), returns +inf
    (never alarm).

    Args:
        calib_scores: 校准分数 / calibration scores, shape [n_calib].
        alpha: 误报预算 / false-alarm budget, in (0, 1).

    Returns:
        阈值 (float)；样本不足时为 +inf / threshold; +inf when n is too small.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    calib = _finite_flat_tensor(calib_scores, name="calibration scores")
    n = calib.numel()
    if n == 0:
        raise ValueError("calibration scores must be non-empty.")

    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float("inf")
    calib_sorted, _ = torch.sort(calib)
    return float(calib_sorted[rank - 1].item())


def directional_conformal_threshold(calib_scores: ArrayLike, alpha: float, direction: str) -> float:
    """Return a native-unit threshold for higher- or lower-is-anomalous scores."""
    if direction == "higher":
        return conformal_threshold(calib_scores, alpha)
    if direction == "lower":
        threshold = conformal_threshold(-_finite_flat_tensor(calib_scores, name="calibration scores"), alpha)
        return -threshold
    raise ValueError("direction must be 'higher' or 'lower'.")


def directional_conformal_thresholds(
    calib_scores: ArrayLike,
    alphas: Sequence[float],
    direction: str,
) -> dict[float, float]:
    """Return native-unit thresholds for several alphas using one calibration sort."""
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    calib = _finite_flat_tensor(calib_scores, name="calibration scores")
    n = calib.numel()
    if n == 0:
        raise ValueError("calibration scores must be non-empty.")

    oriented = calib if direction == "higher" else -calib
    oriented_sorted, _ = torch.sort(oriented)
    thresholds: dict[float, float] = {}
    for alpha in alphas:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        rank = math.ceil((n + 1) * (1.0 - alpha))
        if rank > n:
            oriented_threshold = float("inf")
        else:
            oriented_threshold = float(oriented_sorted[rank - 1].item())
        thresholds[float(alpha)] = (
            oriented_threshold if direction == "higher" else -oriented_threshold
        )
    return thresholds


def directional_trigger_rate(scores: ArrayLike, threshold: float, direction: str) -> float:
    """Return the fraction of scores flagged by a directional conformal threshold."""
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    if math.isnan(float(threshold)):
        raise ValueError("threshold must not be NaN.")
    scores_t = _finite_flat_tensor(scores, name="scores")
    if scores_t.numel() == 0:
        return 0.0
    if direction == "higher":
        return float((scores_t > threshold).double().mean().item())
    return float((scores_t < threshold).double().mean().item())


def conformal_abstention_report(
    uncertainty_scores: ArrayLike,
    correctness: Sequence[bool | int | float],
    alpha: float,
    *,
    direction: str = "higher",
    score_name: str | None = None,
) -> ConformalAbstentionReport:
    """Calibrate a conformal participation threshold and selective-accuracy report.

    The threshold is calibrated on responses marked correct, then evaluated on all
    calibration responses. For ``higher`` uncertainty, scores above the threshold
    abstain; for ``lower`` uncertainty, scores below the threshold abstain.

    Args:
        uncertainty_scores: Reliability scores on calibration responses. ``direction``
            determines which side is less reliable.
        correctness: Binary correctness labels for the same calibration responses.
        alpha: Correct-response miss budget in ``(0, 1)``. With exchangeable
            correct responses, at most alpha should fall outside the retained
            participation region.
        direction: ``higher`` when larger scores are less reliable; ``lower`` when
            smaller scores are less reliable.
        score_name: Optional stable score name for trace metadata.
    """
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    alpha_value = _alpha_float(alpha)
    scores = _finite_flat_tensor(uncertainty_scores, name="uncertainty scores")
    labels = _binary_flat_tensor(correctness, name="correctness")
    if scores.numel() == 0:
        raise ValueError("uncertainty scores must be non-empty.")
    if labels.numel() != scores.numel():
        raise ValueError("correctness must have the same length as uncertainty scores.")
    correct_scores = scores[labels]
    if correct_scores.numel() == 0:
        raise ValueError("correctness must contain at least one correct response for calibration.")

    threshold = directional_conformal_threshold(correct_scores, alpha_value, direction)
    return evaluate_conformal_abstention(
        scores,
        labels.tolist(),
        threshold=threshold,
        alpha=alpha_value,
        direction=direction,
        score_name=score_name,
    )


def conformal_abstention_comparison_report(
    uncertainty_scores: Mapping[str, ArrayLike],
    correctness: Sequence[bool | int | float],
    alpha: float,
    *,
    directions: Mapping[str, str] | None = None,
    best_by: str = "conditional_correctness_lower_bound",
) -> ConformalAbstentionComparisonReport:
    """Rank several uncertainty scores as conformal abstention candidates.

    All candidates share the same correctness labels and alpha. Higher values are
    better for every supported ``best_by`` metric; ties are resolved by
    conservative correctness, empirical selective accuracy, participation,
    correct retention, then score name for deterministic reports.
    """
    if not isinstance(uncertainty_scores, Mapping):
        raise ValueError("uncertainty_scores must be a mapping of score name to values.")
    if not uncertainty_scores:
        raise ValueError("uncertainty_scores must contain at least one score.")
    if best_by not in ABSTENTION_COMPARISON_METRICS:
        raise ValueError(f"best_by must be one of {ABSTENTION_COMPARISON_METRICS}.")
    alpha_value = _alpha_float(alpha)
    direction_map = (
        {}
        if directions is None
        else {str(name): str(value) for name, value in directions.items()}
    )

    reports: list[ConformalAbstentionReport] = []
    for raw_name, scores in uncertainty_scores.items():
        name = str(raw_name)
        if not name:
            raise ValueError("score names must be non-empty.")
        direction = direction_map.get(name, "higher")
        reports.append(
            conformal_abstention_report(
                scores,
                correctness,
                alpha_value,
                direction=direction,
                score_name=name,
            )
        )

    def metric_value(report: ConformalAbstentionReport, metric: str) -> float | None:
        value = getattr(report, metric)
        if value is None:
            return None
        return _unit_interval_float(value, name=metric)

    def sortable_value(value: float | None) -> float:
        return -1.0 if value is None else value

    def sort_key(report: ConformalAbstentionReport) -> tuple[float, float, float, float, float, str]:
        score_name = "" if report.score_name is None else report.score_name
        return (
            -sortable_value(metric_value(report, best_by)),
            -sortable_value(metric_value(report, "conditional_correctness_lower_bound")),
            -sortable_value(metric_value(report, "empirical_selective_accuracy")),
            -sortable_value(metric_value(report, "empirical_participation_rate")),
            -sortable_value(metric_value(report, "correct_retention_lower_bound")),
            score_name,
        )

    ranked_reports = sorted(reports, key=sort_key)
    candidates = tuple(
        ConformalAbstentionComparisonCandidate(
            rank=rank,
            score_name=str(report.score_name),
            direction=report.direction,
            selection_metric=best_by,
            selection_value=metric_value(report, best_by),
            report=report,
        )
        for rank, report in enumerate(ranked_reports, start=1)
    )
    return ConformalAbstentionComparisonReport(
        alpha=alpha_value,
        best_by=best_by,
        candidates=candidates,
    )


def conformal_abstention_release_gate(
    report: (
        ConformalAbstentionReport
        | ConformalAbstentionComparisonCandidate
        | ConformalAbstentionComparisonReport
        | Mapping[str, Any]
    ),
    *,
    min_conditional_correctness_lower_bound: float = 0.8,
    max_abstention_rate: float = 0.5,
) -> ConformalAbstentionReleaseGateResult:
    """Evaluate a conformal abstention report as a promotion gate.

    The gate is intentionally small and fail-closed: the selected candidate must
    satisfy both a conservative conditional-correctness lower bound and an upper
    bound on empirical abstention rate.
    """
    return ConformalAbstentionReleaseGate(
        min_conditional_correctness_lower_bound=min_conditional_correctness_lower_bound,
        max_abstention_rate=max_abstention_rate,
    ).evaluate(report)


def evaluate_conformal_abstention(
    uncertainty_scores: ArrayLike,
    correctness: Sequence[bool | int | float],
    *,
    threshold: float,
    alpha: float,
    direction: str = "higher",
    score_name: str | None = None,
) -> ConformalAbstentionReport:
    """Evaluate selective participation metrics for a fixed abstention threshold."""
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    alpha_value = _alpha_float(alpha)
    threshold_value = _threshold_float(threshold, name="threshold")
    scores = _finite_flat_tensor(uncertainty_scores, name="uncertainty scores")
    labels = _binary_flat_tensor(correctness, name="correctness")
    if scores.numel() == 0:
        raise ValueError("uncertainty scores must be non-empty.")
    if labels.numel() != scores.numel():
        raise ValueError("correctness must have the same length as uncertainty scores.")

    retained = scores <= threshold_value if direction == "higher" else scores >= threshold_value
    correct = labels
    correct_retained = retained & correct
    n = int(scores.numel())
    n_correct = int(correct.sum().item())
    retained_count = int(retained.sum().item())
    correct_retained_count = int(correct_retained.sum().item())
    abstained_count = n - retained_count

    empirical_base_accuracy = n_correct / n
    empirical_participation_rate = retained_count / n
    empirical_abstention_rate = abstained_count / n
    empirical_selective_accuracy = (
        None if retained_count == 0 else correct_retained_count / retained_count
    )
    correct_retention_rate = 0.0 if n_correct == 0 else correct_retained_count / n_correct
    # Conservative finite-sample style bounds used for post-hoc policy routing:
    # correct-retention uses the correct-response calibration count, base accuracy
    # uses the all-response calibration count, and participation uses the retained
    # rank among all calibration samples.
    correct_retention_lower_bound = (
        0.0 if n_correct == 0 else correct_retained_count / (n_correct + 1.0)
    )
    base_accuracy_lower_bound = n_correct / (n + 1.0)
    participation_upper_bound = min(1.0, (retained_count + 1.0) / (n + 1.0))
    conditional_correctness_lower_bound = 0.0
    if participation_upper_bound > 0.0:
        conditional_correctness_lower_bound = (
            correct_retention_lower_bound * base_accuracy_lower_bound / participation_upper_bound
        )
    conditional_correctness_lower_bound = max(0.0, min(1.0, conditional_correctness_lower_bound))

    return ConformalAbstentionReport(
        threshold=threshold_value,
        alpha=alpha_value,
        direction=direction,
        n_calibration=n,
        n_correct=n_correct,
        retained_count=retained_count,
        correct_retained_count=correct_retained_count,
        abstained_count=abstained_count,
        empirical_base_accuracy=empirical_base_accuracy,
        empirical_participation_rate=empirical_participation_rate,
        empirical_abstention_rate=empirical_abstention_rate,
        empirical_selective_accuracy=empirical_selective_accuracy,
        correct_retention_rate=correct_retention_rate,
        correct_retention_lower_bound=correct_retention_lower_bound,
        participation_upper_bound=participation_upper_bound,
        conditional_correctness_lower_bound=conditional_correctness_lower_bound,
        score_name=score_name,
    )


def adaptive_anomaly_scores(
    scores: ArrayLike,
    *,
    feature_values: Mapping[str, ArrayLike] | None = None,
    feature_weights: Mapping[str, float] | None = None,
    intercept: float = 0.0,
    direction: str = "higher",
) -> Tensor:
    """Return feature-adjusted anomaly scores for adaptive conformal calibration.

    The returned tensor is always in higher-is-more-anomalous orientation. Feature
    weights are additive: ``adjusted = oriented_score + intercept + sum(w_i*x_i)``.
    """
    if direction == "higher":
        adjusted = _finite_flat_tensor(scores, name="scores")
    elif direction == "lower":
        adjusted = -_finite_flat_tensor(scores, name="scores")
    else:
        raise ValueError("direction must be 'higher' or 'lower'.")

    offset = torch.full_like(adjusted, _finite_float(intercept, name="intercept"))
    weights = {} if feature_weights is None else dict(feature_weights)
    values = {} if feature_values is None else feature_values
    for name, raw_weight in weights.items():
        weight = _finite_float(raw_weight, name=f"feature_weights.{name}")
        if name not in values:
            raise ValueError(f"feature_values is missing required feature '{name}'.")
        feature = _finite_flat_tensor(values[name], name=f"feature_values.{name}")
        if feature.numel() != adjusted.numel():
            raise ValueError("feature values must have the same length as scores.")
        offset = offset + weight * feature
    return adjusted + offset


def _multiple_testing_method(value: object) -> str:
    method = str(value).strip().lower().replace("_", "-")
    aliases = {
        "by": "by",
        "benjamini-yekutieli": "by",
        "dependency-safe": "by",
        "bh": "bh",
        "benjamini-hochberg": "bh",
        "bonferroni": "bonferroni",
    }
    if method not in aliases:
        raise ValueError("method must be one of: by, bh, bonferroni.")
    return aliases[method]


def _multiple_testing_correction(method: str, signal_count: int) -> float:
    if signal_count < 1:
        raise ValueError("signal_count must be positive.")
    if method == "by":
        return float(sum(1.0 / rank for rank in range(1, signal_count + 1)))
    if method == "bonferroni":
        return float(signal_count)
    if method == "bh":
        return 1.0
    raise ValueError("method must be one of: by, bh, bonferroni.")


def _multiple_testing_rank_threshold(
    method: str,
    alpha: float,
    rank: int,
    signal_count: int,
    correction: float,
) -> float:
    if method == "bonferroni":
        return alpha / signal_count
    return rank * alpha / (signal_count * correction)


def _alpha_spending_schedule_name(value: object) -> str:
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


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, not bool.")
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not math.isfinite(as_float) or not as_float.is_integer():
        raise ValueError(f"{name} must be a positive integer.")
    integer = int(as_float)
    if integer < 1:
        raise ValueError(f"{name} must be positive.")
    return integer


def _finite_flat_tensor(values: ArrayLike, *, name: str) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _pvalue_flat_tensor(values: ArrayLike, *, name: str) -> Tensor:
    tensor = _finite_flat_tensor(values, name=name)
    if not (((tensor >= 0.0) & (tensor <= 1.0)).all()):
        raise ValueError(f"{name} must contain p-values in [0, 1].")
    return tensor


def _binary_flat_tensor(values: Sequence[bool | int | float], *, name: str) -> Tensor:
    try:
        tensor = torch.as_tensor(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a binary sequence.") from exc
    tensor = tensor.flatten()
    if tensor.dtype == torch.bool:
        return tensor
    numeric = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not torch.isfinite(numeric).all():
        raise ValueError(f"{name} must contain only finite values.")
    if not (((numeric == 0.0) | (numeric == 1.0)).all()):
        raise ValueError(f"{name} must contain only 0/1 or bool labels.")
    return numeric.to(torch.bool)


def _alpha_float(value: object) -> float:
    alpha = _finite_float(value, name="alpha")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    return alpha


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _threshold_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric and must not be NaN.")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric and must not be NaN.") from exc
    if math.isnan(result):
        raise ValueError(f"{name} must be numeric and must not be NaN.")
    return result


def _unit_interval_float(value: object, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if not (0.0 <= result <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")
    return result


def _select_abstention_report_for_release_gate(
    report: (
        ConformalAbstentionReport
        | ConformalAbstentionComparisonCandidate
        | ConformalAbstentionComparisonReport
        | Mapping[str, Any]
    ),
) -> tuple[ConformalAbstentionReport, str, int | None] | None:
    if isinstance(report, ConformalAbstentionReport):
        return report, "conformal_abstention_report", None
    if isinstance(report, ConformalAbstentionComparisonCandidate):
        return report.report, "conformal_abstention_comparison_candidate", None
    if isinstance(report, ConformalAbstentionComparisonReport):
        recommended = report.recommended
        if recommended is None:
            return None
        return (
            recommended.report,
            "conformal_abstention_comparison_report",
            len(report.candidates),
        )
    if not isinstance(report, Mapping):
        raise ValueError("abstention release gate input must be a report or mapping.")
    if "recommended" in report and "candidates" in report:
        return _select_abstention_report_for_release_gate(
            ConformalAbstentionComparisonReport.from_dict(report)
        )
    if "report" in report and "score_name" in report:
        return _select_abstention_report_for_release_gate(
            ConformalAbstentionComparisonCandidate.from_dict(report)
        )
    return _select_abstention_report_for_release_gate(
        ConformalAbstentionReport.from_dict(report)
    )
