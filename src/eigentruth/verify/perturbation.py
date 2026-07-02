"""Prompt-perturbation consistency audits for high-certainty hallucination risk."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from eigentruth.json_utils import to_jsonable
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus
from eigentruth.verify.rules import normalize_claim_text
from eigentruth.verify.triples import ClaimTriple, ClaimTripleExtractor, RuleBasedTripleExtractor


@dataclass(frozen=True)
class PerturbationVariant:
    """One response produced under an answer-preserving prompt perturbation."""

    text: str
    variant_id: str | None = None
    perturbation_type: str = "unspecified"
    confidence: float | None = None
    expected_consistent: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text = str(self.text).strip()
        if not text:
            raise ValueError("perturbation variant text must be non-empty.")
        variant_id = None if self.variant_id is None else str(self.variant_id).strip() or None
        perturbation_type = str(self.perturbation_type).strip().casefold().replace("-", "_")
        if not perturbation_type:
            perturbation_type = "unspecified"
        confidence = _optional_unit_float(self.confidence, name="confidence")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "variant_id", variant_id)
        object.__setattr__(self, "perturbation_type", perturbation_type)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self,
            "expected_consistent",
            _strict_bool(self.expected_consistent, name="expected_consistent"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready variant payload."""
        return {
            "text": self.text,
            "variant_id": self.variant_id,
            "perturbation_type": self.perturbation_type,
            "confidence": self.confidence,
            "expected_consistent": self.expected_consistent,
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class PerturbationConsistencyPolicy:
    """Thresholds for an answer-preserving perturbation consistency audit."""

    min_variants: int = 2
    high_confidence_threshold: float = 0.80
    max_conflict_rate: float = 0.0
    max_high_confidence_conflict_rate: float = 0.0
    max_missing_rate: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_variants", _positive_int(self.min_variants, name="min_variants"))
        for name in (
            "high_confidence_threshold",
            "max_conflict_rate",
            "max_high_confidence_conflict_rate",
            "max_missing_rate",
        ):
            object.__setattr__(self, name, _unit_float(getattr(self, name), name=name))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready policy payload."""
        return {
            "min_variants": self.min_variants,
            "high_confidence_threshold": self.high_confidence_threshold,
            "max_conflict_rate": self.max_conflict_rate,
            "max_high_confidence_conflict_rate": self.max_high_confidence_conflict_rate,
            "max_missing_rate": self.max_missing_rate,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PerturbationConsistencyPolicy":
        """Build a policy from a JSON-like mapping."""
        return cls(
            min_variants=data.get("min_variants", 2),
            high_confidence_threshold=data.get("high_confidence_threshold", 0.80),
            max_conflict_rate=data.get("max_conflict_rate", 0.0),
            max_high_confidence_conflict_rate=data.get("max_high_confidence_conflict_rate", 0.0),
            max_missing_rate=data.get("max_missing_rate", 1.0),
        )


@dataclass(frozen=True)
class PerturbationConsistencyRecord:
    """Fact-level comparison between one variant and the anchor claim."""

    variant: PerturbationVariant
    status: VerificationStatus
    reason: str
    supported_triples: Sequence[ClaimTriple] = ()
    conflicting_triples: Sequence[ClaimTriple] = ()
    missing_triples: Sequence[ClaimTriple] = ()
    variant_triples: Sequence[ClaimTriple] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "supported_triples", tuple(self.supported_triples))
        object.__setattr__(self, "conflicting_triples", tuple(self.conflicting_triples))
        object.__setattr__(self, "missing_triples", tuple(self.missing_triples))
        object.__setattr__(self, "variant_triples", tuple(self.variant_triples))

    @property
    def has_confidence(self) -> bool:
        """Return whether the variant supplied a confidence score."""
        return self.variant.confidence is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready record payload."""
        return {
            "variant": self.variant.to_dict(),
            "status": self.status.value,
            "reason": self.reason,
            "has_confidence": self.has_confidence,
            "supported_triples": tuple(item.to_dict() for item in self.supported_triples),
            "conflicting_triples": tuple(item.to_dict() for item in self.conflicting_triples),
            "missing_triples": tuple(item.to_dict() for item in self.missing_triples),
            "variant_triples": tuple(item.to_dict() for item in self.variant_triples),
        }


@dataclass(frozen=True)
class PerturbationConsistencyReport:
    """Aggregate answer-preserving perturbation consistency report."""

    claim: Claim
    status: str
    reason: str
    policy: PerturbationConsistencyPolicy
    anchor_triples: Sequence[ClaimTriple]
    records: Sequence[PerturbationConsistencyRecord]
    skipped_variant_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status).strip().casefold().replace("-", "_")
        if status not in {"passed", "blocked", "not_applicable"}:
            raise ValueError("status must be 'passed', 'blocked', or 'not_applicable'.")
        skipped = int(self.skipped_variant_count)
        if skipped < 0:
            raise ValueError("skipped_variant_count must be non-negative.")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "anchor_triples", tuple(self.anchor_triples))
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "skipped_variant_count", skipped)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def passed(self) -> bool:
        """Return whether the audit passed."""
        return self.status == "passed"

    def summary(self) -> dict[str, Any]:
        """Return aggregate perturbation-consistency metrics."""
        records = tuple(self.records)
        variant_count = len(records)
        supported_count = sum(1 for item in records if item.status is VerificationStatus.SUPPORTED)
        conflict_count = sum(1 for item in records if item.status is VerificationStatus.REFUTED)
        missing_count = sum(
            1 for item in records if item.status is VerificationStatus.INSUFFICIENT_EVIDENCE
        )
        high_confidence = tuple(
            item
            for item in records
            if item.variant.confidence is not None
            and item.variant.confidence >= self.policy.high_confidence_threshold
        )
        high_confidence_conflict_count = sum(
            1 for item in high_confidence if item.status is VerificationStatus.REFUTED
        )
        return {
            "status": self.status,
            "passed": self.passed,
            "reason": self.reason,
            "anchor_triple_count": len(self.anchor_triples),
            "variant_count": variant_count,
            "skipped_variant_count": self.skipped_variant_count,
            "supported_count": supported_count,
            "conflict_count": conflict_count,
            "missing_count": missing_count,
            "conflict_rate": _safe_div(conflict_count, variant_count),
            "missing_rate": _safe_div(missing_count, variant_count),
            "high_confidence_threshold": self.policy.high_confidence_threshold,
            "high_confidence_variant_count": len(high_confidence),
            "high_confidence_conflict_count": high_confidence_conflict_count,
            "high_confidence_conflict_rate": _safe_div(
                high_confidence_conflict_count,
                len(high_confidence),
            ),
            "counts_by_perturbation_type": _counts_by_perturbation_type(records),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report payload."""
        return {
            "workflow": "perturbation_consistency_audit",
            "claim": _claim_to_dict(self.claim),
            "status": self.status,
            "reason": self.reason,
            "passed": self.passed,
            "policy": self.policy.to_dict(),
            "summary": self.summary(),
            "anchor_triples": tuple(item.to_dict() for item in self.anchor_triples),
            "records": tuple(item.to_dict() for item in self.records),
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class PerturbationConsistencyVerifier:
    """Verify a claim against answer-preserving prompt-perturbation variants.

    This is a dependency-free CHOKE-style audit shell: callers supply responses
    produced under prompt variants that should preserve the underlying fact, and
    the verifier reports whether those variants conflict with the anchor claim,
    especially when the variant supplied high confidence. It does not generate
    perturbations or prove the anchor claim true.
    """

    variants: Sequence[str | Mapping[str, Any]] = ()
    extractor: ClaimTripleExtractor = RuleBasedTripleExtractor()
    policy: PerturbationConsistencyPolicy | Mapping[str, Any] = field(
        default_factory=PerturbationConsistencyPolicy
    )
    context_keys: Sequence[str] = (
        "perturbation_variants",
        "prompt_perturbation_variants",
        "choke_variants",
        "choke_samples",
    )

    def __post_init__(self) -> None:
        policy = (
            self.policy
            if isinstance(self.policy, PerturbationConsistencyPolicy)
            else PerturbationConsistencyPolicy.from_dict(self.policy)
        )
        object.__setattr__(self, "variants", tuple(_coerce_variant(item) for item in self.variants))
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "context_keys", tuple(str(key) for key in self.context_keys))

    def report(self, claim: Claim, context: Mapping[str, Any] | None = None) -> PerturbationConsistencyReport:
        """Return a perturbation-consistency report for one claim."""
        variants = self._variants_from_context(context)
        return audit_perturbation_consistency(
            claim,
            variants,
            extractor=self.extractor,
            policy=self.policy,
            metadata={"verifier": "perturbation_consistency"},
        )

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one claim through the perturbation-consistency audit."""
        report = self.report(claim, context=context)
        if report.status == "blocked":
            status = VerificationStatus.REFUTED
            confidence = min(
                0.95,
                0.5 + 0.45 * max(
                    report.summary()["conflict_rate"],
                    report.summary()["high_confidence_conflict_rate"],
                ),
            )
            explanation = "answer-preserving perturbation variants conflicted with the claim"
        elif report.status == "passed":
            status = VerificationStatus.SUPPORTED
            confidence = min(0.95, 0.5 + 0.45 * (1.0 - report.summary()["missing_rate"]))
            explanation = "answer-preserving perturbation variants stayed fact-consistent"
        else:
            status = VerificationStatus.NOT_APPLICABLE
            confidence = 0.0
            explanation = report.reason
        return VerificationResult(
            status=status,
            confidence=confidence,
            evidence=tuple(_variant_label(item) for item in report.records),
            explanation=explanation,
            metadata={
                "verifier": "perturbation_consistency",
                "decision_rule": report.reason,
                "perturbation_consistency": report.to_dict(),
            },
        )

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        """Verify multiple claims."""
        return tuple(self.verify(claim, context=context) for claim in claims)

    def _variants_from_context(self, context: Mapping[str, Any] | None) -> tuple[PerturbationVariant, ...]:
        variants = list(self.variants)
        if context is not None:
            for key in self.context_keys:
                if key in context:
                    variants.extend(_coerce_variants(context[key]))
        return tuple(variants)


def audit_perturbation_consistency(
    claim: Claim | Mapping[str, Any],
    variants: Sequence[str | Mapping[str, Any] | PerturbationVariant],
    *,
    extractor: ClaimTripleExtractor = RuleBasedTripleExtractor(),
    policy: PerturbationConsistencyPolicy | Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PerturbationConsistencyReport:
    """Audit whether answer-preserving perturbation variants stay fact-consistent."""
    anchor = _coerce_claim(claim)
    resolved_policy = (
        PerturbationConsistencyPolicy()
        if policy is None
        else policy
        if isinstance(policy, PerturbationConsistencyPolicy)
        else PerturbationConsistencyPolicy.from_dict(policy)
    )
    coerced_variants = tuple(_coerce_variant(item) for item in variants)
    expected_variants = tuple(item for item in coerced_variants if item.expected_consistent)
    skipped_variant_count = len(coerced_variants) - len(expected_variants)
    anchor_triples = tuple(extractor.extract(anchor))
    if len(expected_variants) < resolved_policy.min_variants:
        return PerturbationConsistencyReport(
            claim=anchor,
            status="not_applicable",
            reason="too_few_expected_consistent_variants",
            policy=resolved_policy,
            anchor_triples=anchor_triples,
            records=(),
            skipped_variant_count=skipped_variant_count,
            metadata={} if metadata is None else dict(metadata),
        )
    if not anchor_triples:
        return PerturbationConsistencyReport(
            claim=anchor,
            status="not_applicable",
            reason="no_anchor_triples",
            policy=resolved_policy,
            anchor_triples=(),
            records=(),
            skipped_variant_count=skipped_variant_count,
            metadata={} if metadata is None else dict(metadata),
        )

    records = tuple(_variant_record(anchor_triples, variant, extractor) for variant in expected_variants)
    summary = _summary_for_policy(records, resolved_policy)
    blocked_reasons = []
    if summary["conflict_rate"] > resolved_policy.max_conflict_rate:
        blocked_reasons.append("conflict_rate")
    if summary["high_confidence_conflict_rate"] > resolved_policy.max_high_confidence_conflict_rate:
        blocked_reasons.append("high_confidence_conflict_rate")
    if summary["missing_rate"] > resolved_policy.max_missing_rate:
        blocked_reasons.append("missing_rate")
    status = "blocked" if blocked_reasons else "passed"
    reason = ",".join(blocked_reasons) if blocked_reasons else "within_policy"
    return PerturbationConsistencyReport(
        claim=anchor,
        status=status,
        reason=reason,
        policy=resolved_policy,
        anchor_triples=anchor_triples,
        records=records,
        skipped_variant_count=skipped_variant_count,
        metadata={} if metadata is None else dict(metadata),
    )


def _variant_record(
    anchor_triples: Sequence[ClaimTriple],
    variant: PerturbationVariant,
    extractor: ClaimTripleExtractor,
) -> PerturbationConsistencyRecord:
    variant_claim = Claim(
        text=variant.text,
        claim_id=variant.variant_id,
        metadata=variant.metadata,
    )
    variant_triples = tuple(extractor.extract(variant_claim))
    supported: list[ClaimTriple] = []
    conflicting: list[ClaimTriple] = []
    missing: list[ClaimTriple] = []
    for anchor_triple in anchor_triples:
        if any(_triple_key(anchor_triple) == _triple_key(item) for item in variant_triples):
            supported.append(anchor_triple)
            continue
        if any(_triple_conflicts(anchor_triple, item) for item in variant_triples):
            conflicting.append(anchor_triple)
            continue
        missing.append(anchor_triple)
    if conflicting:
        status = VerificationStatus.REFUTED
        reason = "subject_predicate_object_conflict"
    elif len(supported) == len(anchor_triples):
        status = VerificationStatus.SUPPORTED
        reason = "all_anchor_triples_supported"
    else:
        status = VerificationStatus.INSUFFICIENT_EVIDENCE
        reason = "missing_anchor_triples" if variant_triples else "no_variant_triples"
    return PerturbationConsistencyRecord(
        variant=variant,
        status=status,
        reason=reason,
        supported_triples=tuple(supported),
        conflicting_triples=tuple(conflicting),
        missing_triples=tuple(missing),
        variant_triples=variant_triples,
    )


def _summary_for_policy(
    records: Sequence[PerturbationConsistencyRecord],
    policy: PerturbationConsistencyPolicy,
) -> dict[str, float]:
    variant_count = len(records)
    conflict_count = sum(1 for item in records if item.status is VerificationStatus.REFUTED)
    missing_count = sum(1 for item in records if item.status is VerificationStatus.INSUFFICIENT_EVIDENCE)
    high_confidence = tuple(
        item
        for item in records
        if item.variant.confidence is not None and item.variant.confidence >= policy.high_confidence_threshold
    )
    high_conflict_count = sum(1 for item in high_confidence if item.status is VerificationStatus.REFUTED)
    return {
        "conflict_rate": _safe_div(conflict_count, variant_count),
        "missing_rate": _safe_div(missing_count, variant_count),
        "high_confidence_conflict_rate": _safe_div(high_conflict_count, len(high_confidence)),
    }


def _coerce_variants(value: Any) -> tuple[PerturbationVariant, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Mapping, PerturbationVariant)):
        return (_coerce_variant(value),)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(_coerce_variant(item) for item in value)
    raise ValueError("perturbation variants must be strings, mappings, or sequences of those values.")


def _coerce_variant(value: str | Mapping[str, Any] | PerturbationVariant) -> PerturbationVariant:
    if isinstance(value, PerturbationVariant):
        return value
    if isinstance(value, str):
        return PerturbationVariant(text=value)
    if not isinstance(value, Mapping):
        raise ValueError("perturbation variant must be a string, mapping, or PerturbationVariant.")
    raw_text = value.get("text", value.get("content", value.get("response", value.get("answer"))))
    if raw_text is None:
        raise ValueError("perturbation variant mapping must contain text, content, response, or answer.")
    metadata = value.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError("perturbation variant metadata must be a mapping when provided.")
    confidence = _first_present(value, ("confidence", "certainty", "probability"))
    if confidence is None:
        confidence = _first_present(metadata, ("confidence", "certainty", "probability"))
    return PerturbationVariant(
        text=str(raw_text),
        variant_id=_optional_text(_first_present(value, ("variant_id", "id", "source"))),
        perturbation_type=str(_first_present(value, ("perturbation_type", "type")) or "unspecified"),
        confidence=confidence,
        expected_consistent=_first_present(value, ("expected_consistent", "answer_preserving"), default=True),
        metadata=dict(metadata),
    )


def _coerce_claim(value: Claim | Mapping[str, Any]) -> Claim:
    if isinstance(value, Claim):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("claim must be a Claim or mapping.")
    text = value.get("text", value.get("claim", value.get("statement")))
    if text is None or not str(text).strip():
        raise ValueError("claim mapping must contain text, claim, or statement.")
    claim_id = value.get("claim_id", value.get("id"))
    metadata = value.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError("claim metadata must be a mapping when provided.")
    return Claim(
        text=str(text),
        claim_id=None if claim_id is None else str(claim_id),
        metadata=dict(metadata),
    )


def _claim_to_dict(claim: Claim) -> dict[str, Any]:
    return {
        "text": claim.text,
        "claim_id": claim.claim_id,
        "span": claim.span,
        "metadata": to_jsonable(dict(claim.metadata)),
    }


def _variant_label(record: PerturbationConsistencyRecord) -> str:
    variant_id = record.variant.variant_id or record.variant.perturbation_type
    return f"{variant_id}:{record.status.value}"


def _triple_conflicts(anchor: ClaimTriple, variant: ClaimTriple) -> bool:
    return (
        _slot_key(anchor.subject) == _slot_key(variant.subject)
        and _slot_key(anchor.predicate) == _slot_key(variant.predicate)
        and _slot_key(anchor.object) != _slot_key(variant.object)
    )


def _triple_key(triple: ClaimTriple) -> tuple[str, str, str]:
    return (_slot_key(triple.subject), _slot_key(triple.predicate), _slot_key(triple.object))


def _slot_key(value: str) -> str:
    return normalize_claim_text(str(value))


def _counts_by_perturbation_type(records: Sequence[PerturbationConsistencyRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = counts.setdefault(
            record.variant.perturbation_type,
            {"total": 0, "supported": 0, "refuted": 0, "insufficient_evidence": 0},
        )
        bucket["total"] += 1
        bucket[record.status.value] = bucket.get(record.status.value, 0) + 1
    return counts


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_unit_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _unit_float(value, name=name)


def _unit_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number in [0, 1], not bool.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number in [0, 1].") from exc
    if not math.isfinite(parsed) or not (0.0 <= parsed <= 1.0):
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    return parsed


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, not bool.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean or boolean string.")


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
