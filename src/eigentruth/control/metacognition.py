"""Metacognitive alignment audits for product traces.

The audit is intentionally lightweight: it compares a trace-level risk proxy
against the amount of uncertainty expressed in the final answer text. It is not
a claim that lexical cues fully measure semantic uncertainty; it is a
dependency-free product signal for confident high-risk outputs.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

_UNCERTAINTY_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("dont_know", r"\b(i\s+)?do\s+not\s+know\b|\b(i\s+)?don't\s+know\b", 1.0),
    ("not_sure", r"\bnot\s+(entirely\s+)?sure\b|\bunsure\b", 0.85),
    ("uncertain", r"\buncertain(?:ty)?\b|\bunclear\b", 0.8),
    ("insufficient_evidence", r"\binsufficient\s+evidence\b|\bnot\s+enough\s+evidence\b", 0.9),
    ("cannot_verify", r"\bcannot\s+verify\b|\bcan't\s+verify\b|\bnot\s+verified\b", 0.8),
    ("maybe", r"\bmaybe\b|\bperhaps\b|\bpossibly\b", 0.55),
    ("might", r"\bmight\b|\bmay\b|\bcould\b", 0.35),
    ("appears", r"\bappears?\b|\bseems?\b|\blikely\b|\bprobably\b", 0.35),
    ("approximately", r"\bapproximately\b|\broughly\b|\babout\b", 0.25),
)

_CERTAINTY_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("definitely", r"\bdefinitely\b|\bcertainly\b|\bundoubtedly\b", 0.85),
    ("clearly", r"\bclearly\b|\bobviously\b", 0.65),
    ("guaranteed", r"\bguaranteed\b|\bwithout\s+doubt\b|\bno\s+doubt\b", 0.9),
    ("always_never", r"\balways\b|\bnever\b", 0.45),
    ("proven_fact", r"\bproven\b|\bfact(?:ually)?\b", 0.45),
    ("must_will", r"\bmust\b|\bwill\b", 0.25),
)

_RISK_LEVEL_SCORES = {
    "low": 0.15,
    "medium": 0.55,
    "high": 0.85,
    "unknown": 0.75,
}

_ACTION_RISK_SCORES = {
    "accept": 0.15,
    "retrieve": 0.65,
    "clarify": 0.65,
    "rewrite": 0.7,
    "regenerate": 0.7,
    "abstain": 0.9,
}

_VERIFICATION_STATUS_SCORES = {
    "supported": 0.1,
    "refuted": 0.9,
    "insufficient_evidence": 0.7,
    "error": 0.8,
    "not_applicable": 0.35,
}

_NON_ANSWER_STATUSES = {
    "abstained",
    "needs_clarification",
    "needs_retrieval",
    "needs_rewrite",
    "needs_regeneration",
    "needs_tool_execution",
    "blocked",
}


@dataclass(frozen=True)
class VerbalUncertaintySignal:
    """Lexical estimate of expressed uncertainty in final-answer text."""

    text_available: bool
    uncertainty_score: float | None
    expressed_confidence_score: float | None
    uncertainty_cue_count: int = 0
    certainty_cue_count: int = 0
    uncertainty_cues: Mapping[str, int] = field(default_factory=dict)
    certainty_cues: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_available": self.text_available,
            "uncertainty_score": self.uncertainty_score,
            "expressed_confidence_score": self.expressed_confidence_score,
            "uncertainty_cue_count": self.uncertainty_cue_count,
            "certainty_cue_count": self.certainty_cue_count,
            "uncertainty_cues": dict(self.uncertainty_cues),
            "certainty_cues": dict(self.certainty_cues),
        }


@dataclass(frozen=True)
class MetacognitionAuditReport:
    """Trace-level alignment between expressed uncertainty and observed risk."""

    available: bool
    status: str
    passed: bool | None
    verbal_uncertainty: VerbalUncertaintySignal
    risk_proxy: float | None
    miscalibration_score: float | None
    overconfident_risk: bool
    overcautious_uncertainty: bool
    reasons: Sequence[str] = ()
    thresholds: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-ready summary."""
        verbal = self.verbal_uncertainty.to_dict()
        return {
            "available": self.available,
            "status": self.status,
            "passed": self.passed,
            "text_available": verbal["text_available"],
            "risk_proxy": self.risk_proxy,
            "verbal_uncertainty_score": verbal["uncertainty_score"],
            "expressed_confidence_score": verbal["expressed_confidence_score"],
            "miscalibration_score": self.miscalibration_score,
            "overconfident_risk": self.overconfident_risk,
            "overcautious_uncertainty": self.overcautious_uncertainty,
            "uncertainty_cue_count": verbal["uncertainty_cue_count"],
            "certainty_cue_count": verbal["certainty_cue_count"],
            "uncertainty_cues": verbal["uncertainty_cues"],
            "certainty_cues": verbal["certainty_cues"],
            "reasons": tuple(self.reasons),
            "reason_counts": _counts(self.reasons),
            "thresholds": dict(self.thresholds),
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a full JSON-ready report."""
        return self.summary()


def verbal_uncertainty_signal(
    text: str | None,
    *,
    final_answer: Mapping[str, Any] | None = None,
) -> VerbalUncertaintySignal:
    """Estimate how much uncertainty the answer text expresses."""
    normalized = "" if text is None else str(text).strip()
    if not normalized:
        return VerbalUncertaintySignal(
            text_available=False,
            uncertainty_score=None,
            expressed_confidence_score=None,
        )
    uncertainty_cues, uncertainty_weight = _pattern_counts(
        normalized,
        _UNCERTAINTY_PATTERNS,
    )
    certainty_cues, certainty_weight = _pattern_counts(
        normalized,
        _CERTAINTY_PATTERNS,
    )
    score = 0.2 + 0.18 * uncertainty_weight - 0.12 * certainty_weight
    final = _mapping(final_answer)
    if _is_non_answer(final):
        score = max(score, 0.75)
    score = _clamp(score, 0.0, 1.0)
    return VerbalUncertaintySignal(
        text_available=True,
        uncertainty_score=score,
        expressed_confidence_score=1.0 - score,
        uncertainty_cue_count=sum(uncertainty_cues.values()),
        certainty_cue_count=sum(certainty_cues.values()),
        uncertainty_cues=uncertainty_cues,
        certainty_cues=certainty_cues,
    )


def audit_metacognitive_alignment(
    *,
    text: str | None = None,
    final_answer: Mapping[str, Any] | None = None,
    risk_decision: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    verification_results: Sequence[Mapping[str, Any]] = (),
    low_verbal_uncertainty_threshold: float = 0.35,
    high_verbal_uncertainty_threshold: float = 0.65,
    high_risk_threshold: float = 0.6,
    low_risk_threshold: float = 0.35,
) -> MetacognitionAuditReport:
    """Audit whether an answer's language matches observed risk.

    Positive ``miscalibration_score`` means the trace looks riskier than the
    answer sounds uncertain. Negative values mean the answer sounds more
    uncertain than the observed risk proxy.
    """
    thresholds = {
        "low_verbal_uncertainty_threshold": _rate(low_verbal_uncertainty_threshold),
        "high_verbal_uncertainty_threshold": _rate(high_verbal_uncertainty_threshold),
        "high_risk_threshold": _rate(high_risk_threshold),
        "low_risk_threshold": _rate(low_risk_threshold),
    }
    final = _mapping(final_answer)
    answer_text = text
    if answer_text is None and final:
        answer_text = None if final.get("text") is None else str(final.get("text"))
    verbal = verbal_uncertainty_signal(answer_text, final_answer=final)
    risk_proxy, risk_metadata = _risk_proxy(
        final_answer=final,
        risk_decision=_mapping(risk_decision),
        diagnostics=_mapping(diagnostics),
        verification_results=verification_results,
    )
    available = verbal.text_available and risk_proxy is not None
    miscalibration = (
        None
        if not available or verbal.uncertainty_score is None or risk_proxy is None
        else risk_proxy - verbal.uncertainty_score
    )
    overconfident = bool(
        available
        and risk_proxy is not None
        and verbal.uncertainty_score is not None
        and risk_proxy >= thresholds["high_risk_threshold"]
        and verbal.uncertainty_score <= thresholds["low_verbal_uncertainty_threshold"]
    )
    overcautious = bool(
        available
        and risk_proxy is not None
        and verbal.uncertainty_score is not None
        and risk_proxy <= thresholds["low_risk_threshold"]
        and verbal.uncertainty_score >= thresholds["high_verbal_uncertainty_threshold"]
        and _is_answered(final)
    )
    reasons: list[str] = []
    if not verbal.text_available:
        reasons.append("missing_final_answer_text")
    if risk_proxy is None:
        reasons.append("missing_risk_proxy")
    if overconfident:
        reasons.append("high_risk_low_verbal_uncertainty")
    if overcautious:
        reasons.append("low_risk_high_verbal_uncertainty")
    status = "insufficient_signal"
    passed: bool | None = None
    if available:
        passed = not overconfident
        status = "fail" if overconfident else "pass"
    return MetacognitionAuditReport(
        available=available,
        status=status,
        passed=passed,
        verbal_uncertainty=verbal,
        risk_proxy=risk_proxy,
        miscalibration_score=miscalibration,
        overconfident_risk=overconfident,
        overcautious_uncertainty=overcautious,
        reasons=tuple(reasons),
        thresholds=thresholds,
        metadata=risk_metadata,
    )


def _risk_proxy(
    *,
    final_answer: Mapping[str, Any],
    risk_decision: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    verification_results: Sequence[Mapping[str, Any]],
) -> tuple[float | None, dict[str, Any]]:
    values: list[tuple[str, float]] = []
    risk_level = _clean(risk_decision.get("risk_level")) or _clean(final_answer.get("risk_level"))
    if risk_level in _RISK_LEVEL_SCORES:
        values.append((f"risk_level:{risk_level}", _RISK_LEVEL_SCORES[risk_level]))
    action = _clean(risk_decision.get("action")) or _clean(final_answer.get("action"))
    if action in _ACTION_RISK_SCORES:
        values.append((f"action:{action}", _ACTION_RISK_SCORES[action]))
    if final_answer:
        if final_answer.get("answerable") is False:
            values.append(("final_answer:not_answerable", 0.75))
        confidence = _finite_float(final_answer.get("confidence"))
        if confidence is not None and final_answer.get("answerable") is False:
            values.append(("final_answer:non_answer_confidence", max(0.5, confidence)))
    verifier_values = _verification_risk_values(verification_results)
    values.extend(verifier_values)
    diagnostics_value = _diagnostic_risk(diagnostics)
    if diagnostics_value is not None:
        values.append(("diagnostics:risk_proxy", diagnostics_value))
    if not values:
        return None, {"risk_sources": ()}
    score = max(value for _source, value in values)
    return score, {
        "risk_sources": tuple(source for source, _value in values),
        "risk_source_count": len(values),
    }


def _verification_risk_values(
    verification_results: Sequence[Mapping[str, Any]],
) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for result in verification_results:
        payload = _mapping(result)
        status = _clean(payload.get("status"))
        if status in _VERIFICATION_STATUS_SCORES:
            confidence = _finite_float(payload.get("confidence"))
            base = _VERIFICATION_STATUS_SCORES[status]
            if confidence is not None and status in {"refuted", "insufficient_evidence", "error"}:
                base = max(base, 0.5 + 0.5 * confidence)
            values.append((f"verification:{status}", _clamp(base, 0.0, 1.0)))
    return values


def _diagnostic_risk(diagnostics: Mapping[str, Any]) -> float | None:
    values: list[float] = []
    for raw_key, raw_value in diagnostics.items():
        value = _finite_float(raw_value)
        if value is None:
            continue
        key = str(raw_key).lower()
        if "risk" in key or "uncertainty" in key or "entropy" in key or "hse" in key:
            values.append(_clamp(value, 0.0, 1.0))
        elif "confidence" in key or "support" in key:
            values.append(_clamp(1.0 - value, 0.0, 1.0))
    if not values:
        return None
    return max(values)


def _pattern_counts(
    text: str,
    patterns: Sequence[tuple[str, str, float]],
) -> tuple[dict[str, int], float]:
    counts: dict[str, int] = {}
    weight = 0.0
    for label, pattern, pattern_weight in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        count = len(matches)
        if count <= 0:
            continue
        counts[label] = count
        weight += count * pattern_weight
    return counts, weight


def _is_non_answer(final_answer: Mapping[str, Any]) -> bool:
    if not final_answer:
        return False
    status = _clean(final_answer.get("status"))
    action = _clean(final_answer.get("action"))
    return (
        final_answer.get("answerable") is False
        or status in _NON_ANSWER_STATUSES
        or action in {"abstain", "clarify", "retrieve"}
    )


def _is_answered(final_answer: Mapping[str, Any]) -> bool:
    if not final_answer:
        return True
    return _clean(final_answer.get("status")) == "answered" or final_answer.get("answerable") is True


def _rate(value: Any) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError("metacognition thresholds must be finite rates in [0, 1]")
    return resolved


def _finite_float(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(resolved):
        return None
    return resolved


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _clean(value: Any) -> str:
    return str(value or "").strip().lower()


def _counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts
