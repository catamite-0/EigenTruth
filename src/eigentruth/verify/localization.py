"""Claim-level risk localization helpers.

The localization layer is intentionally lightweight: it turns existing claim
spans, verification results, and verification-plan budget metadata into an
auditable span report. It does not run a learned token detector or change
verifier semantics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from eigentruth.json_utils import to_jsonable
from eigentruth.verify.claims import claim_entity_candidates
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus

RISK_LEVEL_ORDER: Mapping[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


@dataclass(frozen=True)
class ClaimRiskSpan:
    """Risk annotation for one extracted claim span."""

    claim_id: str
    text: str
    span: tuple[int, int] | None
    risk_level: str
    risk_score: float
    status: str | None = None
    confidence: float | None = None
    routes: Sequence[str] = ()
    reasons: Sequence[str] = ()
    evidence_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        claim_id = str(self.claim_id).strip()
        if not claim_id:
            raise ValueError("claim_id must be non-empty.")
        text = str(self.text)
        span = _optional_span(self.span)
        risk_level = str(self.risk_level).strip().lower()
        if risk_level not in RISK_LEVEL_ORDER:
            raise ValueError("risk_level must be one of: low, medium, high.")
        risk_score = _bounded_float(self.risk_score, name="risk_score")
        confidence = None if self.confidence is None else _bounded_float(self.confidence, name="confidence")
        evidence_count = _non_negative_int(self.evidence_count, name="evidence_count")
        object.__setattr__(self, "claim_id", claim_id)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "span", span)
        object.__setattr__(self, "risk_level", risk_level)
        object.__setattr__(self, "risk_score", risk_score)
        object.__setattr__(self, "status", None if self.status is None else str(self.status))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "routes", tuple(_non_empty_strings(self.routes)))
        object.__setattr__(self, "reasons", tuple(_non_empty_strings(self.reasons)))
        object.__setattr__(self, "evidence_count", evidence_count)
        object.__setattr__(self, "metadata", dict(to_jsonable(dict(self.metadata))))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "span": self.span,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "status": self.status,
            "confidence": self.confidence,
            "routes": tuple(self.routes),
            "reasons": tuple(self.reasons),
            "evidence_count": self.evidence_count,
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class ClaimRiskLocalizationReport:
    """JSON-ready localization report over extracted claim spans."""

    spans: Sequence[ClaimRiskSpan | Mapping[str, Any]] = ()
    source: str = "claim_verification"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spans = tuple(_span_obj(span) for span in self.spans)
        object.__setattr__(self, "spans", spans)
        object.__setattr__(self, "source", str(self.source).strip() or "claim_verification")
        object.__setattr__(self, "metadata", dict(to_jsonable(dict(self.metadata))))

    def summary(self) -> dict[str, Any]:
        """Return compact counts for traces and runtime metrics."""
        counts_by_risk_level = {"low": 0, "medium": 0, "high": 0}
        counts_by_status: dict[str, int] = {}
        counts_by_feature: dict[str, int] = {}
        localized_count = 0
        max_risk_score = 0.0
        entity_candidate_count = 0
        entity_claim_ids: list[str] = []
        high_risk_entity_claim_ids: list[str] = []
        high_risk_claim_ids: list[str] = []
        medium_or_high_claim_ids: list[str] = []
        for span in self.spans:
            counts_by_risk_level[span.risk_level] += 1
            if span.status is not None:
                counts_by_status[span.status] = counts_by_status.get(span.status, 0) + 1
            metadata = _mapping(span.metadata)
            feature_flags = _mapping(metadata.get("feature_flags"))
            for feature, enabled in feature_flags.items():
                if _truthy(enabled):
                    key = str(feature)
                    counts_by_feature[key] = counts_by_feature.get(key, 0) + 1
            entity_candidates = tuple(_non_empty_strings(_as_sequence(metadata.get("entity_candidates", ()))))
            if entity_candidates:
                entity_candidate_count += len(entity_candidates)
                entity_claim_ids.append(span.claim_id)
                if span.risk_level == "high":
                    high_risk_entity_claim_ids.append(span.claim_id)
            if span.span is not None:
                localized_count += 1
            max_risk_score = max(max_risk_score, span.risk_score)
            if span.risk_level == "high":
                high_risk_claim_ids.append(span.claim_id)
            if RISK_LEVEL_ORDER[span.risk_level] >= RISK_LEVEL_ORDER["medium"]:
                medium_or_high_claim_ids.append(span.claim_id)
        return {
            "available": bool(self.spans),
            "source": self.source,
            "span_count": len(self.spans),
            "localized_span_count": localized_count,
            "counts_by_risk_level": counts_by_risk_level,
            "counts_by_status": counts_by_status,
            "counts_by_feature": dict(sorted(counts_by_feature.items())),
            "entity_claim_count": len(entity_claim_ids),
            "entity_candidate_count": entity_candidate_count,
            "high_risk_entity_claim_count": len(high_risk_entity_claim_ids),
            "entity_claim_ids": tuple(entity_claim_ids),
            "high_risk_entity_claim_ids": tuple(high_risk_entity_claim_ids),
            "max_risk_score": max_risk_score,
            "high_risk_claim_count": len(high_risk_claim_ids),
            "medium_or_high_risk_claim_count": len(medium_or_high_claim_ids),
            "high_risk_claim_ids": tuple(high_risk_claim_ids),
            "medium_or_high_risk_claim_ids": tuple(medium_or_high_claim_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "source": self.source,
            "summary": self.summary(),
            "spans": tuple(span.to_dict() for span in self.spans),
            "metadata": to_jsonable(dict(self.metadata)),
        }


def localize_claim_risk_spans(
    claims: Sequence[Claim | Mapping[str, Any]],
    *,
    verification_results: Sequence[VerificationResult | Mapping[str, Any]] = (),
    verification_plan: Mapping[str, Any] | Any | None = None,
    source_text: str | None = None,
    min_risk_level: str = "low",
) -> ClaimRiskLocalizationReport:
    """Build a claim-span risk report from existing trace artifacts.

    Results are matched to claims by explicit ``claim_id`` metadata when
    available; otherwise same-order verifier outputs are used as a fallback.
    """
    normalized_claims = tuple(_claim_payload(claim, index) for index, claim in enumerate(claims))
    results_by_claim = _results_by_claim_id(verification_results, normalized_claims)
    plan_payload = _plan_payload(verification_plan)
    route_hints = _route_hints_by_claim(plan_payload)
    budget = _mapping(plan_payload.get("budget")) if plan_payload else {}
    dropped_claim_ids = {str(item) for item in _as_sequence(budget.get("dropped_claim_ids", ()))}
    dropped_routes = {
        str(claim_id): tuple(_non_empty_strings(_as_sequence(routes)))
        for claim_id, routes in _mapping(budget.get("dropped_routes")).items()
    }
    verify_claim_ids = {str(item) for item in _as_sequence(plan_payload.get("verify_claim_ids", ()))}
    selected_claim_ids = {str(item) for item in _as_sequence(budget.get("selected_claim_ids", ()))}
    if not selected_claim_ids:
        selected_claim_ids = verify_claim_ids
    min_level = _risk_level(min_risk_level)

    spans: list[ClaimRiskSpan] = []
    for claim in normalized_claims:
        claim_id = claim["claim_id"]
        claim_text = str(claim["text"])
        claim_metadata = _mapping(claim.get("metadata"))
        entity_candidates = _claim_entity_candidates(claim_metadata, claim_text=claim_text)
        feature_flags = _claim_feature_flags(
            claim_metadata,
            claim_text=claim_text,
            entity_candidates=entity_candidates,
        )
        claim_results = results_by_claim.get(claim_id, ())
        route_hint = route_hints.get(claim_id, {})
        routes = tuple(_non_empty_strings(_as_sequence(route_hint.get("routes", ()))))
        selected_result = _highest_risk_result(claim_results)
        score, status, confidence, evidence_count, reasons = _risk_from_result(selected_result)
        feature_reasons, feature_bonus = _claim_feature_reasons(
            claim_metadata,
            feature_flags=feature_flags,
        )
        if status in {None, "supported", "not_applicable"}:
            score += feature_bonus
        reasons.extend(feature_reasons)
        if claim_id in dropped_claim_ids:
            score += 0.15
            reasons.append("budget:dropped_claim")
        if dropped_routes.get(claim_id):
            score += 0.10
            reasons.append("budget:dropped_routes")
        if claim_id in selected_claim_ids and not claim_results:
            score += 0.10
            reasons.append("verification:planned_without_result")
        score = min(score, 1.0)
        risk_level = _risk_level_for_score(score, status=status)
        if RISK_LEVEL_ORDER[risk_level] < RISK_LEVEL_ORDER[min_level]:
            continue
        span = _claim_span(claim, source_text=source_text)
        spans.append(
            ClaimRiskSpan(
                claim_id=claim_id,
                text=claim_text,
                span=span,
                risk_level=risk_level,
                risk_score=score,
                status=status,
                confidence=confidence,
                routes=routes,
                reasons=tuple(dict.fromkeys(reasons)),
                evidence_count=evidence_count,
                metadata={
                    "budget_dropped_routes": dropped_routes.get(claim_id, ()),
                    "entity_candidate_count": len(entity_candidates),
                    "entity_candidates": entity_candidates,
                    "feature_flags": feature_flags,
                    "planned_for_verification": claim_id in verify_claim_ids,
                    "selected_under_budget": claim_id in selected_claim_ids,
                },
            )
        )
    return ClaimRiskLocalizationReport(
        spans=tuple(spans),
        metadata={
            "claim_count": len(normalized_claims),
            "verification_result_count": len(tuple(verification_results)),
            "budget_enabled": bool(budget.get("enabled")) if budget else False,
        },
    )


def _claim_payload(claim: Claim | Mapping[str, Any], index: int) -> dict[str, Any]:
    if isinstance(claim, Claim):
        return {
            "claim_id": claim.claim_id or f"c{index + 1}",
            "text": claim.text,
            "span": claim.span,
            "metadata": to_jsonable(dict(claim.metadata)),
        }
    if not isinstance(claim, Mapping):
        raise ValueError("claims must be Claim objects or mappings.")
    claim_id = claim.get("claim_id") or f"c{index + 1}"
    text = claim.get("text")
    if text is None:
        raise ValueError("claim mapping must contain text.")
    return {
        "claim_id": str(claim_id),
        "text": str(text),
        "span": claim.get("span"),
        "metadata": to_jsonable(dict(_mapping(claim.get("metadata")))),
    }


def _result_payload(result: VerificationResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, VerificationResult):
        return {
            "status": result.status.value,
            "confidence": result.confidence,
            "evidence": tuple(result.evidence),
            "explanation": result.explanation,
            "metadata": to_jsonable(dict(result.metadata)),
        }
    if not isinstance(result, Mapping):
        raise ValueError("verification results must be VerificationResult objects or mappings.")
    payload = dict(to_jsonable(dict(result)))
    if isinstance(payload.get("status"), VerificationStatus):
        payload["status"] = payload["status"].value
    return payload


def _results_by_claim_id(
    results: Sequence[VerificationResult | Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    claim_ids = tuple(str(claim["claim_id"]) for claim in claims)
    for index, result in enumerate(results):
        payload = _result_payload(result)
        claim_id = _result_claim_id(payload)
        if claim_id is None and index < len(claim_ids):
            claim_id = claim_ids[index]
        if claim_id is None:
            continue
        grouped.setdefault(claim_id, []).append(payload)
    return {claim_id: tuple(items) for claim_id, items in grouped.items()}


def _result_claim_id(result: Mapping[str, Any]) -> str | None:
    for payload in (result, _mapping(result.get("metadata"))):
        raw = payload.get("claim_id")
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    metadata = _mapping(result.get("metadata"))
    audit_report = _mapping(metadata.get("audit_report"))
    raw = audit_report.get("claim_id")
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    return None


def _highest_risk_result(results: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    return max(results, key=lambda result: _risk_from_result(result)[0])


def _risk_from_result(result: Mapping[str, Any] | None) -> tuple[float, str | None, float | None, int, list[str]]:
    if result is None:
        return 0.35, None, None, 0, ["verification:missing_result"]
    status = str(result.get("status", "")).strip().lower() or None
    confidence = _optional_bounded_float(result.get("confidence"))
    conf = 0.0 if confidence is None else confidence
    evidence_count = len(_as_sequence(result.get("evidence", ())))
    reasons = [f"verification_status:{status or 'unknown'}"]
    if status == VerificationStatus.REFUTED.value:
        return 0.75 + 0.25 * conf, status, confidence, evidence_count, reasons
    if status == VerificationStatus.ERROR.value:
        return 0.65 + 0.25 * conf, status, confidence, evidence_count, reasons
    if status == VerificationStatus.INSUFFICIENT_EVIDENCE.value:
        return 0.45 + 0.30 * conf, status, confidence, evidence_count, reasons
    if status == VerificationStatus.SUPPORTED.value:
        if confidence is not None and confidence < 0.6:
            reasons.append("verification:low_confidence_support")
        return max(0.0, 0.20 * (1.0 - conf)), status, confidence, evidence_count, reasons
    if status == VerificationStatus.NOT_APPLICABLE.value:
        return 0.15, status, confidence, evidence_count, reasons
    return 0.40 + 0.20 * conf, status, confidence, evidence_count, reasons


def _claim_feature_reasons(
    metadata: Mapping[str, Any],
    *,
    feature_flags: Mapping[str, Any] | None = None,
) -> tuple[list[str], float]:
    features = _claim_feature_flags(metadata) if feature_flags is None else feature_flags
    reasons: list[str] = []
    bonus = 0.0
    for feature, increment in (
        ("has_calculation", 0.08),
        ("has_citation", 0.06),
        ("is_time_sensitive", 0.06),
        ("has_number", 0.04),
        ("has_negation", 0.03),
        ("has_named_entity_hint", 0.03),
    ):
        if _truthy(features.get(feature)):
            bonus += increment
            reasons.append(f"claim_feature:{feature}")
    for key in (
        "requires_verification",
        "state_check",
        "world_model_check",
        "requires_triple_audit",
    ):
        if _truthy(metadata.get(key)):
            bonus += 0.05
            reasons.append(f"claim_metadata:{key}")
    return reasons, min(bonus, 0.20)


def _claim_feature_flags(
    metadata: Mapping[str, Any],
    *,
    claim_text: str | None = None,
    entity_candidates: Sequence[str] | None = None,
) -> dict[str, bool]:
    raw_features = _mapping(metadata.get("features"))
    flags = {str(feature): _truthy(value) for feature, value in raw_features.items()}
    if entity_candidates is None:
        entity_candidates = _claim_entity_candidates(metadata, claim_text=claim_text)
    if entity_candidates and "has_named_entity_hint" not in flags:
        flags["has_named_entity_hint"] = True
    return dict(sorted(flags.items()))


def _claim_entity_candidates(metadata: Mapping[str, Any], *, claim_text: str | None = None) -> tuple[str, ...]:
    for key in ("entity_candidates", "entities", "named_entities"):
        candidates = tuple(_non_empty_strings(_as_sequence(metadata.get(key, ()))))
        if candidates:
            return tuple(dict.fromkeys(candidates))
    features = _mapping(metadata.get("features"))
    candidates = tuple(_non_empty_strings(_as_sequence(features.get("entity_candidates", ()))))
    if candidates:
        return tuple(dict.fromkeys(candidates))
    if claim_text is None:
        return ()
    return claim_entity_candidates(claim_text)


def _risk_level_for_score(score: float, *, status: str | None) -> str:
    if status in {VerificationStatus.REFUTED.value, VerificationStatus.ERROR.value}:
        return "high"
    if status == VerificationStatus.INSUFFICIENT_EVIDENCE.value:
        return "medium" if score < 0.75 else "high"
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _plan_payload(plan: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    if plan is None:
        return {}
    if hasattr(plan, "to_dict") and callable(plan.to_dict):
        return dict(to_jsonable(plan.to_dict()))
    if isinstance(plan, Mapping):
        return dict(to_jsonable(dict(plan)))
    raise ValueError("verification_plan must be a mapping, to_dict object, or None.")


def _route_hints_by_claim(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    hints: dict[str, dict[str, Any]] = {}
    for item in _as_sequence(plan.get("route_hints", ())):
        if not isinstance(item, Mapping):
            continue
        raw_claim_id = item.get("claim_id")
        if raw_claim_id is None:
            continue
        hints[str(raw_claim_id)] = dict(item)
    return hints


def _claim_span(claim: Mapping[str, Any], *, source_text: str | None) -> tuple[int, int] | None:
    span = _optional_span(claim.get("span"))
    if span is not None:
        return span
    if source_text is None:
        return None
    start = source_text.find(str(claim.get("text", "")))
    if start < 0:
        return None
    return start, start + len(str(claim.get("text", "")))


def _span_obj(value: ClaimRiskSpan | Mapping[str, Any]) -> ClaimRiskSpan:
    if isinstance(value, ClaimRiskSpan):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("spans must be ClaimRiskSpan objects or mappings.")
    return ClaimRiskSpan(
        claim_id=str(value["claim_id"]),
        text=str(value.get("text", "")),
        span=value.get("span"),
        risk_level=str(value["risk_level"]),
        risk_score=float(value["risk_score"]),
        status=None if value.get("status") is None else str(value.get("status")),
        confidence=None if value.get("confidence") is None else float(value.get("confidence")),
        routes=tuple(_as_sequence(value.get("routes", ()))),
        reasons=tuple(_as_sequence(value.get("reasons", ()))),
        evidence_count=int(value.get("evidence_count", 0)),
        metadata=_mapping(value.get("metadata")),
    )


def _risk_level(value: str) -> str:
    level = str(value).strip().lower()
    if level not in RISK_LEVEL_ORDER:
        raise ValueError("min_risk_level must be one of: low, medium, high.")
    return level


def _optional_span(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("span must be a two-item sequence.")
    start = _non_negative_int(value[0], name="span.start")
    end = _non_negative_int(value[1], name="span.end")
    if end < start:
        raise ValueError("span end must be >= span start.")
    return start, end


def _bounded_float(value: Any, *, name: str) -> float:
    parsed = _required_float(value, name=name)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _optional_bounded_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return _bounded_float(value, name="confidence")
    except ValueError:
        return None


def _required_float(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed


def _non_negative_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _non_empty_strings(values: Sequence[Any]) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    return tuple(value for value in normalized if value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return value is not None
