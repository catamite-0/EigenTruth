"""Risk controller built on calibration artifacts and verification results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control.policy import ControlAction, RiskDecision, RiskLevel
from eigentruth.verify.protocols import VerificationResult, VerificationStatus


@dataclass(frozen=True)
class RiskController:
    """Map diagnostics, calibrated thresholds, and verification results to product actions."""

    artifact: CalibrationArtifact
    medium_action: ControlAction = ControlAction.RETRIEVE
    high_action: ControlAction = ControlAction.ABSTAIN
    high_trigger_count: int = 2

    def decide(
        self,
        diagnostics: Mapping[str, float],
        verification_results: Sequence[VerificationResult | Mapping[str, Any]] | None = None,
    ) -> RiskDecision:
        """Evaluate diagnostics and optional claim verification results."""
        diagnostic_decision = self._decide_diagnostics(diagnostics)
        if verification_results is None:
            return diagnostic_decision

        verification = _summarize_verification(verification_results)
        trace = dict(diagnostic_decision.diagnostics)
        trace["verification"] = verification
        if verification["total"] == 0:
            return _with_diagnostics(diagnostic_decision, trace)

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
                action=self.high_action,
                risk_level=RiskLevel.HIGH,
                confidence=_composed_confidence(
                    diagnostic_risk_confidence,
                    verification,
                    VerificationStatus.REFUTED,
                    minimum=0.8,
                ),
                reason=reason,
                diagnostics=trace,
            )

        if unsupported_count and diagnostic_decision.risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH}:
            return RiskDecision(
                action=self.high_action,
                risk_level=RiskLevel.HIGH,
                confidence=_composed_confidence(
                    diagnostic_risk_confidence,
                    verification,
                    VerificationStatus.INSUFFICIENT_EVIDENCE,
                    minimum=0.75,
                ),
                reason=(
                    f"diagnostic risk and {unsupported_count} unsupported claim(s) require stronger control"
                ),
                diagnostics=trace,
            )

        if unsupported_count:
            return RiskDecision(
                action=self.medium_action,
                risk_level=RiskLevel.MEDIUM,
                confidence=_composed_confidence(
                    diagnostic_risk_confidence,
                    verification,
                    VerificationStatus.INSUFFICIENT_EVIDENCE,
                    minimum=0.6,
                ),
                reason=f"claim verification found {unsupported_count} unsupported claim(s)",
                diagnostics=trace,
            )

        if error_count and diagnostic_decision.risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH}:
            return RiskDecision(
                action=self.high_action,
                risk_level=RiskLevel.HIGH,
                confidence=_composed_confidence(
                    diagnostic_risk_confidence,
                    verification,
                    VerificationStatus.ERROR,
                    minimum=0.75,
                ),
                reason=f"diagnostic risk and {error_count} verification error(s) require stronger control",
                diagnostics=trace,
            )

        if error_count:
            return RiskDecision(
                action=ControlAction.CLARIFY,
                risk_level=RiskLevel.UNKNOWN,
                confidence=_composed_confidence(
                    diagnostic_risk_confidence,
                    verification,
                    VerificationStatus.ERROR,
                    minimum=0.5,
                ),
                reason=f"claim verification returned {error_count} error(s)",
                diagnostics=trace,
            )

        return _with_diagnostics(diagnostic_decision, trace)

    def _decide_diagnostics(self, diagnostics: Mapping[str, float]) -> RiskDecision:
        """Evaluate raw diagnostics against calibration thresholds."""
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
