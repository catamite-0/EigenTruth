"""Risk controller built on calibration artifacts and verification results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from eigentruth.calibration import (
    CalibrationArtifact,
    CalibrationScore,
    MultipleTestingConformalArtifact,
    SequentialConformalArtifact,
)
from eigentruth.control.policy import ControlAction, RiskDecision, RiskLevel
from eigentruth.eval.conformal import ConformalAbstentionReport
from eigentruth.verify.protocols import VerificationResult, VerificationStatus


@dataclass(frozen=True)
class ParticipationGateConfig:
    """Runtime participation gate derived from a conformal abstention report."""

    score_name: str
    threshold: float
    direction: str = "higher"
    alpha: float | None = None
    conditional_correctness_lower_bound: float | None = None
    source: str = "conformal_abstention_report"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        score_name = str(self.score_name)
        if not score_name:
            raise ValueError("score_name must be non-empty.")
        threshold = _threshold_float(self.threshold, name="threshold")
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        alpha = (
            None
            if self.alpha is None
            else _unit_interval_exclusive_float(self.alpha, name="alpha")
        )
        lower_bound = (
            None
            if self.conditional_correctness_lower_bound is None
            else _unit_interval_float(
                self.conditional_correctness_lower_bound,
                name="conditional_correctness_lower_bound",
            )
        )
        object.__setattr__(self, "score_name", score_name)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "conditional_correctness_lower_bound", lower_bound)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "metadata", {} if self.metadata is None else dict(self.metadata))

    @classmethod
    def from_abstention_report(
        cls,
        report: ConformalAbstentionReport | Mapping[str, Any],
        *,
        source: str = "conformal_abstention_report",
    ) -> "ParticipationGateConfig":
        """Build a gate from a single conformal abstention report."""
        if isinstance(report, ConformalAbstentionReport):
            payload = report.to_dict()
        elif isinstance(report, Mapping):
            payload = dict(report)
        else:
            raise ValueError("abstention report must be a ConformalAbstentionReport or mapping.")
        score_name = payload.get("score_name")
        if score_name is None:
            raise ValueError("abstention report must include score_name.")
        return cls(
            score_name=str(score_name),
            threshold=payload["threshold"],
            direction=str(payload.get("direction", "higher")),
            alpha=None if payload.get("alpha") is None else float(payload["alpha"]),
            conditional_correctness_lower_bound=(
                None
                if payload.get("conditional_correctness_lower_bound") is None
                else float(payload["conditional_correctness_lower_bound"])
            ),
            source=source,
            metadata={
                "n_calibration": payload.get("n_calibration"),
                "n_correct": payload.get("n_correct"),
                "empirical_participation_rate": payload.get("empirical_participation_rate"),
                "empirical_selective_accuracy": payload.get("empirical_selective_accuracy"),
            },
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParticipationGateConfig":
        """Build a participation gate from config, candidate, or comparison-report payload."""
        if "participation_gate" in data:
            nested = data["participation_gate"]
            if not isinstance(nested, Mapping):
                raise ValueError("participation_gate must be a mapping.")
            return cls.from_dict(nested)
        if "recommended" in data and "candidates" in data:
            recommended = data.get("recommended")
            if not isinstance(recommended, Mapping):
                raise ValueError("abstention comparison report must include a recommended mapping.")
            return cls.from_dict(recommended)
        if "report" in data:
            report = data["report"]
            if not isinstance(report, Mapping):
                raise ValueError("abstention candidate report must be a mapping.")
            metadata = {
                "rank": data.get("rank"),
                "selection_metric": data.get("selection_metric"),
                "selection_value": data.get("selection_value"),
            }
            gate = cls.from_abstention_report(report, source="conformal_abstention_comparison_report")
            return cls(
                score_name=gate.score_name,
                threshold=gate.threshold,
                direction=gate.direction,
                alpha=gate.alpha,
                conditional_correctness_lower_bound=gate.conditional_correctness_lower_bound,
                source=gate.source,
                metadata={**dict(gate.metadata or {}), **metadata},
            )
        if "threshold" in data and "score_name" in data:
            return cls(
                score_name=str(data["score_name"]),
                threshold=data["threshold"],
                direction=str(data.get("direction", "higher")),
                alpha=None if data.get("alpha") is None else float(data["alpha"]),
                conditional_correctness_lower_bound=(
                    None
                    if data.get("conditional_correctness_lower_bound") is None
                    else float(data["conditional_correctness_lower_bound"])
                ),
                source=str(data.get("source", "participation_gate_config")),
                metadata=dict(data.get("metadata", {})),
            )
        raise ValueError("participation gate config must include score_name and threshold.")

    def should_participate(self, score: float) -> bool:
        """Return whether a runtime score is inside the retained participation region."""
        value = _finite_float(score, name="score")
        if self.direction == "higher":
            return value <= self.threshold
        return value >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable gate configuration."""
        return {
            "score_name": self.score_name,
            "threshold": self.threshold,
            "direction": self.direction,
            "alpha": self.alpha,
            "conditional_correctness_lower_bound": self.conditional_correctness_lower_bound,
            "source": self.source,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class MultipleTestingGateConfig:
    """Runtime hallucination gate derived from a multi-signal conformal artifact."""

    artifact: MultipleTestingConformalArtifact
    source: str = "multiple_testing_conformal_artifact"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, MultipleTestingConformalArtifact):
            raise ValueError("artifact must be a MultipleTestingConformalArtifact.")
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "metadata", {} if self.metadata is None else dict(self.metadata))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MultipleTestingGateConfig":
        """Build a gate from config or an embedded artifact payload."""
        if "multiple_testing_gate" in data:
            nested = data["multiple_testing_gate"]
            if not isinstance(nested, Mapping):
                raise ValueError("multiple_testing_gate must be a mapping.")
            return cls.from_dict(nested)
        raw_artifact: object
        if "artifact" in data:
            raw_artifact = data["artifact"]
        elif {"model_id", "target_layer", "signals", "alpha"}.issubset(data):
            raw_artifact = data
        else:
            raise ValueError("multiple-testing gate config must include an artifact.")
        if not isinstance(raw_artifact, Mapping):
            raise ValueError("multiple-testing gate artifact must be a mapping.")
        return cls(
            artifact=MultipleTestingConformalArtifact.from_dict(raw_artifact),
            source=str(data.get("source", "multiple_testing_conformal_artifact")),
            metadata=dict(data.get("metadata", {})),
        )

    def signal_names(self) -> tuple[str, ...]:
        """Return required diagnostic signal names."""
        return self.artifact.signal_names()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable gate configuration."""
        return {
            "source": self.source,
            "metadata": dict(self.metadata or {}),
            "artifact": self.artifact.to_dict(),
        }


@dataclass(frozen=True)
class SequentialGateConfig:
    """Runtime sequence gate derived from a sequential conformal artifact."""

    artifact: SequentialConformalArtifact
    source: str = "sequential_conformal_artifact"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, SequentialConformalArtifact):
            raise ValueError("artifact must be a SequentialConformalArtifact.")
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "metadata", {} if self.metadata is None else dict(self.metadata))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SequentialGateConfig":
        """Build a sequence gate from config or an embedded artifact payload."""
        if "sequential_gate" in data:
            nested = data["sequential_gate"]
            if not isinstance(nested, Mapping):
                raise ValueError("sequential_gate must be a mapping.")
            return cls.from_dict(nested)
        raw_artifact: object
        if "artifact" in data:
            raw_artifact = data["artifact"]
        elif {"model_id", "target_layer", "signal_name", "calibration_scores", "alpha"}.issubset(data):
            raw_artifact = data
        else:
            raise ValueError("sequential gate config must include an artifact.")
        if not isinstance(raw_artifact, Mapping):
            raise ValueError("sequential gate artifact must be a mapping.")
        return cls(
            artifact=SequentialConformalArtifact.from_dict(raw_artifact),
            source=str(data.get("source", "sequential_conformal_artifact")),
            metadata=dict(data.get("metadata", {})),
        )

    @property
    def signal_name(self) -> str:
        """Return the required diagnostic signal name."""
        return self.artifact.signal_name

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable gate configuration."""
        return {
            "source": self.source,
            "metadata": dict(self.metadata or {}),
            "artifact": self.artifact.to_dict(),
        }


@dataclass(frozen=True)
class ControlPolicyConfig:
    """Configurable routing policy for composing diagnostics and verification."""

    refuted_action: ControlAction = ControlAction.ABSTAIN
    unsupported_action: ControlAction = ControlAction.RETRIEVE
    verification_error_action: ControlAction = ControlAction.CLARIFY
    compound_risk_action: ControlAction = ControlAction.ABSTAIN
    compound_verification_escalates: bool = True
    participation_gate_action: ControlAction = ControlAction.ABSTAIN
    participation_gate_applies_to_actions: tuple[ControlAction, ...] = (ControlAction.ACCEPT,)
    participation_gate_risk_level: RiskLevel = RiskLevel.HIGH
    participation_gate_confidence_floor: float = 0.75
    participation_gate_supported_override: bool = False
    participation_gate_supported_override_min_confidence: float = 0.85
    multiple_testing_gate_action: ControlAction = ControlAction.ABSTAIN
    multiple_testing_gate_applies_to_actions: tuple[ControlAction, ...] = (ControlAction.ACCEPT,)
    multiple_testing_gate_risk_level: RiskLevel = RiskLevel.HIGH
    multiple_testing_gate_confidence_floor: float = 0.8
    sequential_gate_action: ControlAction = ControlAction.ABSTAIN
    sequential_gate_applies_to_actions: tuple[ControlAction, ...] = (ControlAction.ACCEPT,)
    sequential_gate_risk_level: RiskLevel = RiskLevel.HIGH
    sequential_gate_confidence_floor: float = 0.8
    refuted_risk_level: RiskLevel = RiskLevel.HIGH
    unsupported_risk_level: RiskLevel = RiskLevel.MEDIUM
    verification_error_risk_level: RiskLevel = RiskLevel.UNKNOWN
    compound_risk_level: RiskLevel = RiskLevel.HIGH

    def __post_init__(self) -> None:
        refuted_action = _control_action(self.refuted_action, name="refuted_action")
        unsupported_action = _control_action(self.unsupported_action, name="unsupported_action")
        verification_error_action = _control_action(
            self.verification_error_action,
            name="verification_error_action",
        )
        compound_risk_action = _control_action(
            self.compound_risk_action,
            name="compound_risk_action",
        )
        participation_gate_action = _control_action(
            self.participation_gate_action,
            name="participation_gate_action",
        )
        multiple_testing_gate_action = _control_action(
            self.multiple_testing_gate_action,
            name="multiple_testing_gate_action",
        )
        sequential_gate_action = _control_action(
            self.sequential_gate_action,
            name="sequential_gate_action",
        )
        refuted_risk_level = _risk_level(self.refuted_risk_level, name="refuted_risk_level")
        unsupported_risk_level = _risk_level(
            self.unsupported_risk_level,
            name="unsupported_risk_level",
        )
        verification_error_risk_level = _risk_level(
            self.verification_error_risk_level,
            name="verification_error_risk_level",
        )
        compound_risk_level = _risk_level(self.compound_risk_level, name="compound_risk_level")
        participation_gate_risk_level = _risk_level(
            self.participation_gate_risk_level,
            name="participation_gate_risk_level",
        )
        multiple_testing_gate_risk_level = _risk_level(
            self.multiple_testing_gate_risk_level,
            name="multiple_testing_gate_risk_level",
        )
        sequential_gate_risk_level = _risk_level(
            self.sequential_gate_risk_level,
            name="sequential_gate_risk_level",
        )
        actions = tuple(
            action if isinstance(action, ControlAction) else ControlAction(str(action))
            for action in self.participation_gate_applies_to_actions
        )
        if not actions:
            raise ValueError("participation_gate_applies_to_actions must be non-empty.")
        multiple_testing_actions = tuple(
            action if isinstance(action, ControlAction) else ControlAction(str(action))
            for action in self.multiple_testing_gate_applies_to_actions
        )
        if not multiple_testing_actions:
            raise ValueError("multiple_testing_gate_applies_to_actions must be non-empty.")
        sequential_actions = tuple(
            action if isinstance(action, ControlAction) else ControlAction(str(action))
            for action in self.sequential_gate_applies_to_actions
        )
        if not sequential_actions:
            raise ValueError("sequential_gate_applies_to_actions must be non-empty.")
        object.__setattr__(self, "refuted_action", refuted_action)
        object.__setattr__(self, "unsupported_action", unsupported_action)
        object.__setattr__(self, "verification_error_action", verification_error_action)
        object.__setattr__(self, "compound_risk_action", compound_risk_action)
        object.__setattr__(self, "refuted_risk_level", refuted_risk_level)
        object.__setattr__(self, "unsupported_risk_level", unsupported_risk_level)
        object.__setattr__(self, "verification_error_risk_level", verification_error_risk_level)
        object.__setattr__(self, "compound_risk_level", compound_risk_level)
        object.__setattr__(self, "participation_gate_action", participation_gate_action)
        object.__setattr__(self, "participation_gate_risk_level", participation_gate_risk_level)
        object.__setattr__(self, "participation_gate_applies_to_actions", actions)
        object.__setattr__(self, "multiple_testing_gate_action", multiple_testing_gate_action)
        object.__setattr__(self, "multiple_testing_gate_risk_level", multiple_testing_gate_risk_level)
        object.__setattr__(self, "multiple_testing_gate_applies_to_actions", multiple_testing_actions)
        object.__setattr__(self, "sequential_gate_action", sequential_gate_action)
        object.__setattr__(self, "sequential_gate_risk_level", sequential_gate_risk_level)
        object.__setattr__(self, "sequential_gate_applies_to_actions", sequential_actions)
        confidence_floor = _unit_interval_float(
            self.participation_gate_confidence_floor,
            name="participation_gate_confidence_floor",
        )
        multiple_testing_confidence_floor = _unit_interval_float(
            self.multiple_testing_gate_confidence_floor,
            name="multiple_testing_gate_confidence_floor",
        )
        sequential_confidence_floor = _unit_interval_float(
            self.sequential_gate_confidence_floor,
            name="sequential_gate_confidence_floor",
        )
        override_min_confidence = _unit_interval_float(
            self.participation_gate_supported_override_min_confidence,
            name="participation_gate_supported_override_min_confidence",
        )
        object.__setattr__(self, "participation_gate_confidence_floor", confidence_floor)
        object.__setattr__(
            self,
            "multiple_testing_gate_confidence_floor",
            multiple_testing_confidence_floor,
        )
        object.__setattr__(
            self,
            "sequential_gate_confidence_floor",
            sequential_confidence_floor,
        )
        object.__setattr__(
            self,
            "participation_gate_supported_override",
            _parse_bool(
                self.participation_gate_supported_override,
                name="participation_gate_supported_override",
            ),
        )
        object.__setattr__(
            self,
            "participation_gate_supported_override_min_confidence",
            override_min_confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "refuted_action": self.refuted_action.value,
            "unsupported_action": self.unsupported_action.value,
            "verification_error_action": self.verification_error_action.value,
            "compound_risk_action": self.compound_risk_action.value,
            "compound_verification_escalates": self.compound_verification_escalates,
            "participation_gate_action": self.participation_gate_action.value,
            "participation_gate_applies_to_actions": [
                action.value for action in self.participation_gate_applies_to_actions
            ],
            "participation_gate_risk_level": self.participation_gate_risk_level.value,
            "participation_gate_confidence_floor": self.participation_gate_confidence_floor,
            "participation_gate_supported_override": self.participation_gate_supported_override,
            "participation_gate_supported_override_min_confidence": (
                self.participation_gate_supported_override_min_confidence
            ),
            "multiple_testing_gate_action": self.multiple_testing_gate_action.value,
            "multiple_testing_gate_applies_to_actions": [
                action.value for action in self.multiple_testing_gate_applies_to_actions
            ],
            "multiple_testing_gate_risk_level": self.multiple_testing_gate_risk_level.value,
            "multiple_testing_gate_confidence_floor": self.multiple_testing_gate_confidence_floor,
            "sequential_gate_action": self.sequential_gate_action.value,
            "sequential_gate_applies_to_actions": [
                action.value for action in self.sequential_gate_applies_to_actions
            ],
            "sequential_gate_risk_level": self.sequential_gate_risk_level.value,
            "sequential_gate_confidence_floor": self.sequential_gate_confidence_floor,
            "refuted_risk_level": self.refuted_risk_level.value,
            "unsupported_risk_level": self.unsupported_risk_level.value,
            "verification_error_risk_level": self.verification_error_risk_level.value,
            "compound_risk_level": self.compound_risk_level.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ControlPolicyConfig":
        """Build a control policy config from JSON-like data."""
        return cls(
            refuted_action=ControlAction(str(data.get("refuted_action", ControlAction.ABSTAIN.value))),
            unsupported_action=ControlAction(str(data.get("unsupported_action", ControlAction.RETRIEVE.value))),
            verification_error_action=ControlAction(
                str(data.get("verification_error_action", ControlAction.CLARIFY.value))
            ),
            compound_risk_action=ControlAction(str(data.get("compound_risk_action", ControlAction.ABSTAIN.value))),
            compound_verification_escalates=_parse_bool(
                data.get("compound_verification_escalates", True),
                name="compound_verification_escalates",
            ),
            participation_gate_action=ControlAction(
                str(data.get("participation_gate_action", ControlAction.ABSTAIN.value))
            ),
            participation_gate_applies_to_actions=_parse_action_tuple(
                data.get("participation_gate_applies_to_actions", (ControlAction.ACCEPT.value,)),
                name="participation_gate_applies_to_actions",
            ),
            participation_gate_risk_level=RiskLevel(
                str(data.get("participation_gate_risk_level", RiskLevel.HIGH.value))
            ),
            participation_gate_confidence_floor=float(
                data.get("participation_gate_confidence_floor", 0.75)
            ),
            participation_gate_supported_override=_parse_bool(
                data.get("participation_gate_supported_override", False),
                name="participation_gate_supported_override",
            ),
            participation_gate_supported_override_min_confidence=float(
                data.get("participation_gate_supported_override_min_confidence", 0.85)
            ),
            multiple_testing_gate_action=ControlAction(
                str(data.get("multiple_testing_gate_action", ControlAction.ABSTAIN.value))
            ),
            multiple_testing_gate_applies_to_actions=_parse_action_tuple(
                data.get("multiple_testing_gate_applies_to_actions", (ControlAction.ACCEPT.value,)),
                name="multiple_testing_gate_applies_to_actions",
            ),
            multiple_testing_gate_risk_level=RiskLevel(
                str(data.get("multiple_testing_gate_risk_level", RiskLevel.HIGH.value))
            ),
            multiple_testing_gate_confidence_floor=float(
                data.get("multiple_testing_gate_confidence_floor", 0.8)
            ),
            sequential_gate_action=ControlAction(
                str(data.get("sequential_gate_action", ControlAction.ABSTAIN.value))
            ),
            sequential_gate_applies_to_actions=_parse_action_tuple(
                data.get("sequential_gate_applies_to_actions", (ControlAction.ACCEPT.value,)),
                name="sequential_gate_applies_to_actions",
            ),
            sequential_gate_risk_level=RiskLevel(
                str(data.get("sequential_gate_risk_level", RiskLevel.HIGH.value))
            ),
            sequential_gate_confidence_floor=float(
                data.get("sequential_gate_confidence_floor", 0.8)
            ),
            refuted_risk_level=RiskLevel(str(data.get("refuted_risk_level", RiskLevel.HIGH.value))),
            unsupported_risk_level=RiskLevel(str(data.get("unsupported_risk_level", RiskLevel.MEDIUM.value))),
            verification_error_risk_level=RiskLevel(
                str(data.get("verification_error_risk_level", RiskLevel.UNKNOWN.value))
            ),
            compound_risk_level=RiskLevel(str(data.get("compound_risk_level", RiskLevel.HIGH.value))),
        )


@dataclass(frozen=True)
class RiskController:
    """Map diagnostics, calibrated thresholds, and verification results to product actions."""

    artifact: CalibrationArtifact
    medium_action: ControlAction = ControlAction.RETRIEVE
    high_action: ControlAction = ControlAction.ABSTAIN
    high_trigger_count: int = 2
    policy_config: ControlPolicyConfig | None = None
    participation_gate: ParticipationGateConfig | ConformalAbstentionReport | Mapping[str, Any] | None = None
    multiple_testing_gate: (
        MultipleTestingGateConfig | MultipleTestingConformalArtifact | Mapping[str, Any] | None
    ) = None
    sequential_gate: SequentialGateConfig | SequentialConformalArtifact | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        participation_gate = self.participation_gate
        if participation_gate is not None and not isinstance(participation_gate, ParticipationGateConfig):
            if isinstance(participation_gate, ConformalAbstentionReport):
                gate = ParticipationGateConfig.from_abstention_report(participation_gate)
            elif isinstance(participation_gate, Mapping):
                gate = ParticipationGateConfig.from_dict(participation_gate)
            else:
                raise ValueError(
                    "participation_gate must be a ParticipationGateConfig, abstention report, or mapping."
                )
            object.__setattr__(self, "participation_gate", gate)

        multiple_testing_gate = self.multiple_testing_gate
        if multiple_testing_gate is not None and not isinstance(multiple_testing_gate, MultipleTestingGateConfig):
            if isinstance(multiple_testing_gate, MultipleTestingConformalArtifact):
                gate = MultipleTestingGateConfig(artifact=multiple_testing_gate)
            elif isinstance(multiple_testing_gate, Mapping):
                gate = MultipleTestingGateConfig.from_dict(multiple_testing_gate)
            else:
                raise ValueError(
                    "multiple_testing_gate must be a MultipleTestingGateConfig, "
                    "MultipleTestingConformalArtifact, or mapping."
                )
            object.__setattr__(self, "multiple_testing_gate", gate)

        sequential_gate = self.sequential_gate
        if sequential_gate is not None and not isinstance(sequential_gate, SequentialGateConfig):
            if isinstance(sequential_gate, SequentialConformalArtifact):
                gate = SequentialGateConfig(artifact=sequential_gate)
            elif isinstance(sequential_gate, Mapping):
                gate = SequentialGateConfig.from_dict(sequential_gate)
            else:
                raise ValueError(
                    "sequential_gate must be a SequentialGateConfig, "
                    "SequentialConformalArtifact, or mapping."
                )
            object.__setattr__(self, "sequential_gate", gate)

    def decide(
        self,
        diagnostics: Mapping[str, float],
        verification_results: Sequence[VerificationResult | Mapping[str, Any]] | None = None,
    ) -> RiskDecision:
        """Evaluate diagnostics and optional claim verification results."""
        diagnostic_decision = self._decide_diagnostics(diagnostics)
        policy = self._effective_policy()
        if verification_results is None:
            return self._apply_control_gates(diagnostic_decision, diagnostics, policy)

        verification = _summarize_verification(verification_results)
        trace = dict(diagnostic_decision.diagnostics)
        trace["verification"] = verification
        if verification["total"] == 0:
            return self._apply_control_gates(
                _with_diagnostics(diagnostic_decision, trace),
                diagnostics,
                policy,
            )

        counts = verification["counts"]
        refuted_count = int(counts.get(VerificationStatus.REFUTED.value, 0))
        unsupported_count = int(counts.get(VerificationStatus.INSUFFICIENT_EVIDENCE.value, 0))
        error_count = int(counts.get(VerificationStatus.ERROR.value, 0))

        diagnostic_risk_confidence = _diagnostic_risk_confidence(diagnostic_decision)

        if refuted_count:
            reason = f"claim verification refuted {refuted_count} claim(s)"
            if diagnostic_decision.risk_level is not RiskLevel.LOW:
                reason = f"{reason}; {diagnostic_decision.reason}"
            return RiskDecision(
                action=policy.refuted_action,
                risk_level=policy.refuted_risk_level,
                confidence=_composed_confidence(
                    diagnostic_risk_confidence,
                    verification,
                    VerificationStatus.REFUTED,
                    minimum=0.8,
                ),
                reason=reason,
                diagnostics=trace,
            )

        if diagnostic_decision.risk_level is RiskLevel.UNKNOWN:
            return self._apply_control_gates(
                _with_diagnostics(diagnostic_decision, trace),
                diagnostics,
                policy,
            )

        if (
            unsupported_count
            and diagnostic_decision.risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH}
            and policy.compound_verification_escalates
        ):
            return self._apply_control_gates(
                RiskDecision(
                    action=policy.compound_risk_action,
                    risk_level=policy.compound_risk_level,
                    confidence=_composed_confidence(
                        diagnostic_risk_confidence,
                        verification,
                        VerificationStatus.INSUFFICIENT_EVIDENCE,
                        minimum=0.75,
                    ),
                    reason=(
                        f"diagnostic risk and {unsupported_count} unsupported claim(s) "
                        "require stronger control"
                    ),
                    diagnostics=trace,
                ),
                diagnostics,
                policy,
            )

        if unsupported_count:
            return self._apply_control_gates(
                RiskDecision(
                    action=policy.unsupported_action,
                    risk_level=policy.unsupported_risk_level,
                    confidence=_composed_confidence(
                        diagnostic_risk_confidence,
                        verification,
                        VerificationStatus.INSUFFICIENT_EVIDENCE,
                        minimum=0.6,
                    ),
                    reason=f"claim verification found {unsupported_count} unsupported claim(s)",
                    diagnostics=trace,
                ),
                diagnostics,
                policy,
            )

        if (
            error_count
            and diagnostic_decision.risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH}
            and policy.compound_verification_escalates
        ):
            return self._apply_control_gates(
                RiskDecision(
                    action=policy.compound_risk_action,
                    risk_level=policy.compound_risk_level,
                    confidence=_composed_confidence(
                        diagnostic_risk_confidence,
                        verification,
                        VerificationStatus.ERROR,
                        minimum=0.75,
                    ),
                    reason=f"diagnostic risk and {error_count} verification error(s) require stronger control",
                    diagnostics=trace,
                ),
                diagnostics,
                policy,
            )

        if error_count:
            return self._apply_control_gates(
                RiskDecision(
                    action=policy.verification_error_action,
                    risk_level=policy.verification_error_risk_level,
                    confidence=_composed_confidence(
                        diagnostic_risk_confidence,
                        verification,
                        VerificationStatus.ERROR,
                        minimum=0.5,
                    ),
                    reason=f"claim verification returned {error_count} error(s)",
                    diagnostics=trace,
                ),
                diagnostics,
                policy,
            )

        return self._apply_control_gates(
            _with_diagnostics(diagnostic_decision, trace),
            diagnostics,
            policy,
        )

    def decide_sequence(
        self,
        diagnostics_sequence: Sequence[Mapping[str, float]],
        verification_results_sequence: Sequence[Sequence[VerificationResult | Mapping[str, Any]] | None] | None = None,
    ) -> tuple[RiskDecision, ...]:
        """Evaluate a request/session sequence and optionally apply sequential conformal gating.

        The single-item ``decide(...)`` path remains stateless. This method is the
        explicit state boundary for session or batch alpha spending: it first
        computes ordinary per-item decisions, then spends the sequential gate's
        alpha only across decisions whose actions are in the configured gate
        scope.
        """
        diagnostics_items = tuple(diagnostics_sequence)
        if verification_results_sequence is None:
            verification_items: tuple[Sequence[VerificationResult | Mapping[str, Any]] | None, ...] = (
                (None,) * len(diagnostics_items)
            )
        else:
            verification_items = tuple(verification_results_sequence)
            if len(verification_items) != len(diagnostics_items):
                raise ValueError("verification_results_sequence must match diagnostics_sequence length.")
        decisions = tuple(
            self.decide(diagnostics, verification)
            for diagnostics, verification in zip(diagnostics_items, verification_items, strict=True)
        )
        return self._apply_sequential_gate(decisions, diagnostics_items, self._effective_policy())

    def _effective_policy(self) -> ControlPolicyConfig:
        """Return a policy config while preserving legacy action overrides."""
        if self.policy_config is not None:
            return self.policy_config
        return ControlPolicyConfig(
            refuted_action=self.high_action,
            unsupported_action=self.medium_action,
            verification_error_action=ControlAction.CLARIFY,
            compound_risk_action=self.high_action,
        )

    def _apply_control_gates(
        self,
        decision: RiskDecision,
        diagnostics: Mapping[str, float],
        policy: ControlPolicyConfig,
    ) -> RiskDecision:
        """Apply optional runtime gates in deterministic order."""
        decision = self._apply_multiple_testing_gate(decision, diagnostics, policy)
        return self._apply_participation_gate(decision, diagnostics, policy)

    def _apply_sequential_gate(
        self,
        decisions: Sequence[RiskDecision],
        diagnostics_sequence: Sequence[Mapping[str, float]],
        policy: ControlPolicyConfig,
    ) -> tuple[RiskDecision, ...]:
        """Apply an explicit sequence-level conformal gate without hidden state."""
        gate = self.sequential_gate
        if gate is None:
            return tuple(decisions)
        if not isinstance(gate, SequentialGateConfig):
            raise ValueError("sequential_gate was not normalized.")
        if len(decisions) != len(diagnostics_sequence):
            raise ValueError("decisions and diagnostics_sequence must have the same length.")

        output: list[RiskDecision] = []
        monitored_positions: list[int] = []
        monitored_scores: list[float] = []
        for index, (decision, diagnostics) in enumerate(zip(decisions, diagnostics_sequence, strict=True)):
            trace = dict(decision.diagnostics)
            gate_trace: dict[str, Any] = {
                "enabled": True,
                "source": gate.source,
                "signal_name": gate.signal_name,
                "alpha": gate.artifact.alpha,
                "schedule": gate.artifact.schedule,
                "direction": gate.artifact.direction,
                "sequence_index": index,
                "applies_to_actions": tuple(
                    action.value for action in policy.sequential_gate_applies_to_actions
                ),
                "configured_action": policy.sequential_gate_action.value,
                "metadata": dict(gate.metadata or {}),
            }
            trace["sequential_gate"] = gate_trace
            if decision.action not in policy.sequential_gate_applies_to_actions:
                gate_trace["status"] = "skipped"
                gate_trace["reason"] = f"decision action {decision.action.value!r} is outside gate scope"
                output.append(_with_diagnostics(decision, trace))
                continue
            if gate.signal_name not in diagnostics:
                gate_trace["status"] = "missing_score"
                output.append(
                    RiskDecision(
                        action=ControlAction.CLARIFY,
                        risk_level=RiskLevel.UNKNOWN,
                        confidence=1.0,
                        reason=f"missing sequential gate score: {gate.signal_name}",
                        diagnostics=trace,
                    )
                )
                continue
            raw_value = diagnostics[gate.signal_name]
            if isinstance(raw_value, bool):
                gate_trace["status"] = "invalid_score"
                gate_trace["value"] = raw_value
                output.append(
                    RiskDecision(
                        action=ControlAction.CLARIFY,
                        risk_level=RiskLevel.UNKNOWN,
                        confidence=1.0,
                        reason=f"invalid sequential gate score: {gate.signal_name}",
                        diagnostics=trace,
                    )
                )
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                gate_trace["status"] = "invalid_score"
                gate_trace["value"] = _diagnostic_value_for_trace(raw_value)
                output.append(
                    RiskDecision(
                        action=ControlAction.CLARIFY,
                        risk_level=RiskLevel.UNKNOWN,
                        confidence=1.0,
                        reason=f"invalid sequential gate score: {gate.signal_name}",
                        diagnostics=trace,
                    )
                )
                continue
            if not math.isfinite(value):
                gate_trace["status"] = "invalid_score"
                gate_trace["value"] = _diagnostic_value_for_trace(raw_value)
                output.append(
                    RiskDecision(
                        action=ControlAction.CLARIFY,
                        risk_level=RiskLevel.UNKNOWN,
                        confidence=1.0,
                        reason=f"invalid sequential gate score: {gate.signal_name}",
                        diagnostics=trace,
                    )
                )
                continue
            gate_trace["status"] = "pending"
            gate_trace["value"] = value
            monitored_positions.append(index)
            monitored_scores.append(value)
            output.append(_with_diagnostics(decision, trace))

        if not monitored_scores:
            return tuple(output)

        report = gate.artifact.decide_sequence(
            monitored_scores,
            metadata={
                "gate_source": gate.source,
                "monitored_count": len(monitored_scores),
                **dict(gate.metadata or {}),
            },
        )
        report_summary = {
            "horizon": report.horizon,
            "alpha_spent_total": report.alpha_spent_total,
            "remaining_alpha": report.remaining_alpha,
            "rejected": report.rejected,
            "rejected_count": report.rejected_count,
            "rejected_steps": report.rejected_steps,
        }
        for position, step in zip(monitored_positions, report.steps, strict=True):
            decision = output[position]
            trace = dict(decision.diagnostics)
            gate_trace = dict(trace["sequential_gate"])
            gate_trace.update({
                "status": "rejected" if step.rejected else "passed",
                "step": step.step,
                "p_value": step.p_value,
                "alpha_spent": step.alpha_spent,
                "cumulative_alpha_spent": step.cumulative_alpha_spent,
                "rejected": step.rejected,
                "report_summary": report_summary,
            })
            trace["sequential_gate"] = gate_trace
            if not step.rejected:
                output[position] = _with_diagnostics(decision, trace)
                continue
            output[position] = RiskDecision(
                action=policy.sequential_gate_action,
                risk_level=policy.sequential_gate_risk_level,
                confidence=min(
                    1.0,
                    max(decision.confidence, policy.sequential_gate_confidence_floor),
                ),
                reason=(
                    f"sequential conformal gate rejected {gate.signal_name} "
                    f"at monitored step {step.step}"
                ),
                diagnostics=trace,
            )
        return tuple(output)

    def _apply_multiple_testing_gate(
        self,
        decision: RiskDecision,
        diagnostics: Mapping[str, float],
        policy: ControlPolicyConfig,
    ) -> RiskDecision:
        """Escalate accepted answers when the global conformal gate rejects."""
        gate = self.multiple_testing_gate
        if gate is None:
            return decision
        if not isinstance(gate, MultipleTestingGateConfig):
            raise ValueError("multiple_testing_gate was not normalized.")
        trace = dict(decision.diagnostics)
        gate_trace: dict[str, Any] = {
            "enabled": True,
            "source": gate.source,
            "signals": gate.signal_names(),
            "alpha": gate.artifact.alpha,
            "method": gate.artifact.method,
            "applies_to_actions": tuple(
                action.value for action in policy.multiple_testing_gate_applies_to_actions
            ),
            "configured_action": policy.multiple_testing_gate_action.value,
            "metadata": dict(gate.metadata or {}),
        }
        trace["multiple_testing_gate"] = gate_trace
        if decision.action not in policy.multiple_testing_gate_applies_to_actions:
            gate_trace["status"] = "skipped"
            gate_trace["reason"] = f"decision action {decision.action.value!r} is outside gate scope"
            return _with_diagnostics(decision, trace)

        runtime_scores: dict[str, float] = {}
        invalid: dict[str, Any] = {}
        missing: list[str] = []
        for signal_name in gate.signal_names():
            if signal_name not in diagnostics:
                missing.append(signal_name)
                continue
            raw_value = diagnostics[signal_name]
            if isinstance(raw_value, bool):
                invalid[signal_name] = _diagnostic_value_for_trace(raw_value)
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                invalid[signal_name] = _diagnostic_value_for_trace(raw_value)
                continue
            if not math.isfinite(value):
                invalid[signal_name] = _diagnostic_value_for_trace(raw_value)
                continue
            runtime_scores[signal_name] = value

        if missing:
            gate_trace["status"] = "missing_score"
            gate_trace["missing_scores"] = tuple(missing)
            return RiskDecision(
                action=ControlAction.CLARIFY,
                risk_level=RiskLevel.UNKNOWN,
                confidence=1.0,
                reason=f"missing multiple-testing gate score(s): {', '.join(missing)}",
                diagnostics=trace,
            )
        if invalid:
            gate_trace["status"] = "invalid_score"
            gate_trace["invalid_scores"] = tuple(invalid)
            gate_trace["invalid_values"] = invalid
            return RiskDecision(
                action=ControlAction.CLARIFY,
                risk_level=RiskLevel.UNKNOWN,
                confidence=1.0,
                reason=f"invalid multiple-testing gate score(s): {', '.join(invalid)}",
                diagnostics=trace,
            )

        report = gate.artifact.decide(
            runtime_scores,
            metadata={
                "gate_source": gate.source,
                "base_action": decision.action.value,
                **dict(gate.metadata or {}),
            },
        )
        gate_trace["status"] = "rejected" if report.rejected else "passed"
        gate_trace["rejected"] = report.rejected
        gate_trace["rejected_signal_names"] = report.rejected_signal_names
        gate_trace["rejected_count"] = report.rejected_count
        gate_trace["min_p_value"] = report.min_p_value
        gate_trace["rejection_cutoff"] = report.rejection_cutoff
        gate_trace["signals"] = tuple(signal.to_dict() for signal in report.signals)
        if not report.rejected:
            return _with_diagnostics(decision, trace)

        return RiskDecision(
            action=policy.multiple_testing_gate_action,
            risk_level=policy.multiple_testing_gate_risk_level,
            confidence=min(
                1.0,
                max(decision.confidence, policy.multiple_testing_gate_confidence_floor),
            ),
            reason=(
                "multiple-testing conformal gate rejected signal(s): "
                f"{', '.join(report.rejected_signal_names)}"
            ),
            diagnostics=trace,
        )

    def _apply_participation_gate(
        self,
        decision: RiskDecision,
        diagnostics: Mapping[str, float],
        policy: ControlPolicyConfig,
    ) -> RiskDecision:
        """Escalate accepted answers when the participation gate abstains."""
        gate = self.participation_gate
        if gate is None:
            return decision
        if not isinstance(gate, ParticipationGateConfig):
            raise ValueError("participation_gate was not normalized.")
        trace = dict(decision.diagnostics)
        gate_trace: dict[str, Any] = {
            "enabled": True,
            "score_name": gate.score_name,
            "threshold": gate.threshold,
            "direction": gate.direction,
            "applies_to_actions": tuple(
                action.value for action in policy.participation_gate_applies_to_actions
            ),
            "configured_action": policy.participation_gate_action.value,
            "source": gate.source,
            "alpha": gate.alpha,
            "conditional_correctness_lower_bound": gate.conditional_correctness_lower_bound,
            "metadata": dict(gate.metadata or {}),
        }
        trace["participation_gate"] = gate_trace
        if decision.action not in policy.participation_gate_applies_to_actions:
            gate_trace["status"] = "skipped"
            gate_trace["reason"] = f"decision action {decision.action.value!r} is outside gate scope"
            return _with_diagnostics(decision, trace)
        if gate.score_name not in diagnostics:
            gate_trace["status"] = "missing_score"
            return RiskDecision(
                action=ControlAction.CLARIFY,
                risk_level=RiskLevel.UNKNOWN,
                confidence=1.0,
                reason=f"missing participation gate score: {gate.score_name}",
                diagnostics=trace,
            )
        raw_value = diagnostics[gate.score_name]
        if isinstance(raw_value, bool):
            gate_trace["status"] = "invalid_score"
            gate_trace["value"] = raw_value
            return RiskDecision(
                action=ControlAction.CLARIFY,
                risk_level=RiskLevel.UNKNOWN,
                confidence=1.0,
                reason=f"invalid participation gate score: {gate.score_name}",
                diagnostics=trace,
            )
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            gate_trace["status"] = "invalid_score"
            gate_trace["value"] = _diagnostic_value_for_trace(raw_value)
            return RiskDecision(
                action=ControlAction.CLARIFY,
                risk_level=RiskLevel.UNKNOWN,
                confidence=1.0,
                reason=f"invalid participation gate score: {gate.score_name}",
                diagnostics=trace,
            )
        if not math.isfinite(value):
            gate_trace["status"] = "invalid_score"
            gate_trace["value"] = _diagnostic_value_for_trace(raw_value)
            return RiskDecision(
                action=ControlAction.CLARIFY,
                risk_level=RiskLevel.UNKNOWN,
                confidence=1.0,
                reason=f"invalid participation gate score: {gate.score_name}",
                diagnostics=trace,
            )
        participates = gate.should_participate(value)
        gate_trace["status"] = "participate" if participates else "abstain"
        gate_trace["value"] = value
        if participates:
            return _with_diagnostics(decision, trace)
        override = _participation_gate_supported_override(trace, policy)
        gate_trace["supported_override"] = override
        if override["applied"]:
            gate_trace["status"] = "overridden_by_verification"
            return _with_diagnostics(decision, trace)
        return RiskDecision(
            action=policy.participation_gate_action,
            risk_level=policy.participation_gate_risk_level,
            confidence=min(
                1.0,
                max(decision.confidence, policy.participation_gate_confidence_floor),
            ),
            reason=(
                f"participation gate abstained on {gate.score_name}: "
                f"score {value:.6g} outside retained region"
            ),
            diagnostics=trace,
        )

    def _decide_diagnostics(self, diagnostics: Mapping[str, float]) -> RiskDecision:
        """Evaluate raw diagnostics against calibration thresholds."""
        triggered = []
        missing = []
        invalid: dict[str, Any] = {}
        severities: dict[str, float] = {}
        for score in self.artifact.scores:
            if score.name not in diagnostics:
                missing.append(score.name)
                continue
            raw_value = diagnostics[score.name]
            if isinstance(raw_value, bool):
                invalid[score.name] = _diagnostic_value_for_trace(raw_value)
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                invalid[score.name] = _diagnostic_value_for_trace(raw_value)
                continue
            if not math.isfinite(value):
                invalid[score.name] = _diagnostic_value_for_trace(raw_value)
                continue
            is_triggered = _is_triggered(value, score)
            severity = _severity(value, score) if is_triggered else 0.0
            severities[score.name] = severity
            if is_triggered:
                triggered.append(score.name)

        trace: dict[str, Any] = {
            "triggered_scores": tuple(triggered),
            "missing_scores": tuple(missing),
            "invalid_scores": tuple(invalid),
            "invalid_values": invalid,
            "severities": severities,
            "thresholds": {score.name: score.threshold for score in self.artifact.scores},
        }

        if invalid:
            return RiskDecision(
                action=ControlAction.CLARIFY,
                risk_level=RiskLevel.UNKNOWN,
                confidence=1.0,
                reason=f"invalid diagnostic score(s): {', '.join(invalid)}",
                diagnostics=trace,
            )

        if missing:
            return RiskDecision(
                action=ControlAction.CLARIFY,
                risk_level=RiskLevel.UNKNOWN,
                confidence=1.0,
                reason=f"missing calibrated diagnostic score(s): {', '.join(missing)}",
                diagnostics=trace,
            )

        if not triggered:
            return RiskDecision(
                action=ControlAction.ACCEPT,
                risk_level=RiskLevel.LOW,
                confidence=1.0,
                reason="no calibrated diagnostic threshold was exceeded",
                diagnostics=trace,
            )

        max_severity = max(severities.values(), default=0.0)
        confidence = min(1.0, 0.5 + 0.5 * max_severity)
        if len(triggered) >= self.high_trigger_count:
            return RiskDecision(
                action=self.high_action,
                risk_level=RiskLevel.HIGH,
                confidence=confidence,
                reason="multiple calibrated diagnostic thresholds were exceeded",
                diagnostics=trace,
            )

        return RiskDecision(
            action=self.medium_action,
            risk_level=RiskLevel.MEDIUM,
            confidence=confidence,
            reason=f"calibrated diagnostic threshold exceeded: {triggered[0]}",
            diagnostics=trace,
        )


def _parse_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean or a recognized boolean string.")


def _parse_action_tuple(value: Any, *, name: str) -> tuple[ControlAction, ...]:
    if isinstance(value, str):
        raw_values = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, Sequence):
        raw_values = tuple(value)
    else:
        raise ValueError(f"{name} must be a sequence or comma-separated string.")
    if not raw_values:
        raise ValueError(f"{name} must contain at least one action.")
    try:
        return tuple(
            item if isinstance(item, ControlAction) else ControlAction(str(item))
            for item in raw_values
        )
    except ValueError as exc:
        raise ValueError(f"{name} contains an unknown control action.") from exc


def _control_action(value: Any, *, name: str) -> ControlAction:
    try:
        return value if isinstance(value, ControlAction) else ControlAction(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ControlAction.") from exc


def _risk_level(value: Any, *, name: str) -> RiskLevel:
    try:
        return value if isinstance(value, RiskLevel) else RiskLevel(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid RiskLevel.") from exc


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _threshold_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric and must not be NaN.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric and must not be NaN.") from exc
    if math.isnan(result):
        raise ValueError(f"{name} must be numeric and must not be NaN.")
    return result


def _unit_interval_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if not (0.0 <= result <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")
    return result


def _unit_interval_exclusive_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if not (0.0 < result < 1.0):
        raise ValueError(f"{name} must be in (0, 1).")
    return result


def _diagnostic_value_for_trace(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _is_triggered(value: float, score: CalibrationScore) -> bool:
    if math.isinf(score.threshold):
        return False
    if score.direction == "higher":
        return value > score.threshold
    return value < score.threshold


def _severity(value: float, score: CalibrationScore) -> float:
    scale = max(abs(score.threshold), 1e-8)
    if score.direction == "higher":
        return max(0.0, (value - score.threshold) / scale)
    return max(0.0, (score.threshold - value) / scale)


def _diagnostic_risk_confidence(decision: RiskDecision) -> float:
    if decision.risk_level is RiskLevel.LOW:
        return 0.0
    return decision.confidence


def _with_diagnostics(decision: RiskDecision, diagnostics: Mapping[str, Any]) -> RiskDecision:
    return RiskDecision(
        action=decision.action,
        risk_level=decision.risk_level,
        confidence=decision.confidence,
        reason=decision.reason,
        diagnostics=diagnostics,
    )


def _participation_gate_supported_override(
    trace: Mapping[str, Any],
    policy: ControlPolicyConfig,
) -> dict[str, Any]:
    verification = trace.get("verification")
    if not policy.participation_gate_supported_override:
        return {
            "applied": False,
            "enabled": False,
            "reason": "supported override disabled",
        }
    if not isinstance(verification, Mapping):
        return {
            "applied": False,
            "enabled": True,
            "reason": "verification summary missing",
        }
    total = int(verification.get("total", 0))
    counts = verification.get("counts")
    if not isinstance(counts, Mapping) or total <= 0:
        return {
            "applied": False,
            "enabled": True,
            "total": total,
            "reason": "no verification results",
        }
    supported_count = int(counts.get(VerificationStatus.SUPPORTED.value, 0))
    triggered_statuses = tuple(str(status) for status in verification.get("triggered_statuses", ()))
    by_status = verification.get("max_confidence_by_status")
    supported_confidence = (
        float(by_status.get(VerificationStatus.SUPPORTED.value, 0.0))
        if isinstance(by_status, Mapping)
        else 0.0
    )
    required_confidence = policy.participation_gate_supported_override_min_confidence
    applied = (
        supported_count == total
        and not triggered_statuses
        and supported_confidence >= required_confidence
    )
    reason = "all claims strongly supported" if applied else "verification support is insufficient"
    return {
        "applied": applied,
        "enabled": True,
        "total": total,
        "supported_count": supported_count,
        "triggered_statuses": triggered_statuses,
        "max_supported_confidence": supported_confidence,
        "min_supported_confidence": required_confidence,
        "reason": reason,
    }


def _summarize_verification(
    results: Sequence[VerificationResult | Mapping[str, Any]],
) -> dict[str, Any]:
    counts = {status.value: 0 for status in VerificationStatus}
    max_confidence_by_status: dict[str, float] = {}
    triggered_statuses: list[str] = []
    routing_statuses = {
        VerificationStatus.REFUTED,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
        VerificationStatus.ERROR,
    }

    for result in results:
        status = _verification_status(result)
        confidence = _verification_confidence(result)
        counts[status.value] = counts.get(status.value, 0) + 1
        max_confidence_by_status[status.value] = max(
            confidence,
            max_confidence_by_status.get(status.value, 0.0),
        )
        if status in routing_statuses and status.value not in triggered_statuses:
            triggered_statuses.append(status.value)

    return {
        "total": len(results),
        "counts": counts,
        "triggered_statuses": tuple(triggered_statuses),
        "max_confidence_by_status": max_confidence_by_status,
    }


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


def _composed_confidence(
    diagnostic_confidence: float,
    verification: Mapping[str, Any],
    status: VerificationStatus,
    *,
    minimum: float,
) -> float:
    by_status = verification.get("max_confidence_by_status", {})
    status_confidence = by_status.get(status.value, 0.0) if isinstance(by_status, Mapping) else 0.0
    return min(1.0, max(float(diagnostic_confidence), float(status_confidence), minimum))
