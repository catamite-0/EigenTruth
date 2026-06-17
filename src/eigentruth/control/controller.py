"""Simple risk controller built on calibration artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control.policy import ControlAction, RiskDecision, RiskLevel


@dataclass(frozen=True)
class RiskController:
    """Map diagnostics and calibrated thresholds to product actions."""

    artifact: CalibrationArtifact
    medium_action: ControlAction = ControlAction.RETRIEVE
    high_action: ControlAction = ControlAction.ABSTAIN
    high_trigger_count: int = 2

    def decide(self, diagnostics: Mapping[str, float]) -> RiskDecision:
        """Evaluate diagnostics against calibration thresholds."""
        triggered = []
        missing = []
        severities: dict[str, float] = {}
        for score in self.artifact.scores:
            if score.name not in diagnostics:
                missing.append(score.name)
                continue
            value = float(diagnostics[score.name])
            is_triggered = _is_triggered(value, score)
            severity = _severity(value, score) if is_triggered else 0.0
            severities[score.name] = severity
            if is_triggered:
                triggered.append(score.name)

        trace: dict[str, Any] = {
            "triggered_scores": tuple(triggered),
            "missing_scores": tuple(missing),
            "severities": severities,
            "thresholds": {score.name: score.threshold for score in self.artifact.scores},
        }

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
