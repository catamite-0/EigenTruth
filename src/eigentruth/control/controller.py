"""Risk controller built on calibration artifacts and verification results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from eigentruth.calibration import CalibrationArtifact, CalibrationScore
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
    refuted_risk_level: RiskLevel = RiskLevel.HIGH
    unsupported_risk_level: RiskLevel = RiskLevel.MEDIUM
    verification_error_risk_level: RiskLevel = RiskLevel.UNKNOWN
    compound_risk_level: RiskLevel = RiskLevel.HIGH

    def __post_init__(self) -> None:
        participation_gate_action = (
            self.participation_gate_action
            if isinstance(self.participation_gate_action, ControlAction)
            else ControlAction(str(self.participation_gate_action))
        )
        participation_gate_risk_level = (
            self.participation_gate_risk_level
            if isinstance(self.participation_gate_risk_level, RiskLevel)
            else RiskLevel(str(self.participation_gate_risk_level))
        )
        actions = tuple(
            action if isinstance(action, ControlAction) else ControlAction(str(action))
            for action in self.participation_gate_applies_to_actions
        )
        if not actions:
            raise ValueError("participation_gate_applies_to_actions must be non-empty.")
        object.__setattr__(self, "participation_gate_action", participation_gate_action)
        object.__setattr__(self, "participation_gate_risk_level", participation_gate_risk_level)
        object.__setattr__(self, "participation_gate_applies_to_actions", actions)
        confidence_floor = _unit_interval_float(
            self.participation_gate_confidence_floor,
            name="participation_gate_confidence_floor",
        )
        override_min_confidence = _unit_interval_float(
            self.participation_gate_supported_override_min_confidence,
            name="participation_gate_supported_override_min_confidence",
        )
        object.__setattr__(self, "participation_gate_confidence_floor", confidence_floor)
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

    def __post_init__(self) -> None:
        if self.participation_gate is None or isinstance(self.participation_gate, ParticipationGateConfig):
            return
        if isinstance(self.participation_gate, ConformalAbstentionReport):
            gate = ParticipationGateConfig.from_abstention_report(self.participation_gate)
        elif isinstance(self.participation_gate, Mapping):
            gate = ParticipationGateConfig.from_dict(self.participation_gate)
        else:
            raise ValueError("participation_gate must be a ParticipationGateConfig, abstention report, or mapping.")
        object.__setattr__(self, "participation_gate", gate)

    def decide(
        self,
        diagnostics: Mapping[str, float],
        verification_results: Sequence[VerificationResult | Mapping[str, Any]] | None = None,
    ) -> RiskDecision:
        """Evaluate diagnostics and optional claim verification results."""
        diagnostic_decision = self._decide_diagnostics(diagnostics)
        policy = self._effective_policy()
        if verification_results is None:
            return self._apply_participation_gate(diagnostic_decision, diagnostics, policy)

        verification = _summarize_verification(verification_results)
        trace = dict(diagnostic_decision.diagnostics)
        trace["verification"] = verification
        if verification["total"] == 0:
            return self._apply_participation_gate(
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
            return self._apply_participation_gate(
                _with_diagnostics(diagnostic_decision, trace),
                diagnostics,
                policy,
            )

        if (
            unsupported_count
            and diagnostic_decision.risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH}
            and policy.compound_verification_escalates
        ):
            return self._apply_participation_gate(
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
            return self._apply_participation_gate(
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
            return self._apply_participation_gate(
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
            return self._apply_participation_gate(
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

        return self._apply_participation_gate(
            _with_diagnostics(diagnostic_decision, trace),
            diagnostics,
            policy,
        )

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
