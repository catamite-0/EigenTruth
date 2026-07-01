"""Dependency-free lexical groundedness verifier."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Mapping, NamedTuple, Sequence

from eigentruth.verify.features import normalized_feature_flags
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus
from eigentruth.verify.rules import normalize_claim_text

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
_NEGATION_TOKENS = {
    "not",
    "no",
    "never",
    "false",
    "incorrect",
    "wrong",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "cannot",
    "can't",
    "不是",
    "没有",
    "并非",
    "错误",
    "不正确",
}


@dataclass(frozen=True)
class EvidenceDocument:
    """One text evidence snippet used by a groundedness verifier."""

    text: str
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("evidence text must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "text": self.text,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceDocument":
        """Build an evidence document from JSON-like data."""
        text = data.get("text", data.get("content"))
        if text is None:
            raise ValueError("evidence mapping must contain 'text' or 'content'.")
        source = data.get("source")
        metadata = dict(data.get("metadata", {}))
        for key, value in data.items():
            metadata_key = str(key)
            if metadata_key not in {"text", "content", "source", "metadata"} and metadata_key not in metadata:
                metadata[metadata_key] = value
        return cls(
            text=str(text),
            source=None if source is None else str(source),
            metadata=metadata,
        )


@dataclass(frozen=True)
class EvidenceQualityPolicy:
    """Optional quality gate for evidence-backed groundedness decisions."""

    max_age_days: int | None = None
    reference_time: str | datetime | date | None = None
    require_source: bool = False
    trusted_sources: Sequence[str] = ()
    require_trusted_source: bool = False
    time_sensitive_only: bool = True

    def __post_init__(self) -> None:
        if self.max_age_days is not None:
            object.__setattr__(self, "max_age_days", _coerce_non_negative_int(
                self.max_age_days,
                name="max_age_days",
            ))
        object.__setattr__(self, "require_source", _coerce_bool(self.require_source, name="require_source"))
        object.__setattr__(
            self,
            "require_trusted_source",
            _coerce_bool(self.require_trusted_source, name="require_trusted_source"),
        )
        object.__setattr__(
            self,
            "time_sensitive_only",
            _coerce_bool(self.time_sensitive_only, name="time_sensitive_only"),
        )
        object.__setattr__(self, "trusted_sources", _coerce_trusted_sources(self.trusted_sources))
        object.__setattr__(self, "reference_time", _coerce_reference_time(self.reference_time))

    def reference_time_or_now(self) -> datetime:
        """Return the configured reference time or the current UTC time."""
        return self.reference_time or datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        reference_time = self.reference_time
        return {
            "max_age_days": self.max_age_days,
            "reference_time": None if reference_time is None else reference_time.isoformat(),
            "require_source": self.require_source,
            "trusted_sources": tuple(self.trusted_sources),
            "require_trusted_source": self.require_trusted_source,
            "time_sensitive_only": self.time_sensitive_only,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceQualityPolicy":
        """Build an evidence quality policy from JSON-like data."""
        return cls(
            max_age_days=data.get("max_age_days"),
            reference_time=data.get("reference_time"),
            require_source=_coerce_bool(data.get("require_source", False), name="require_source"),
            trusted_sources=_coerce_trusted_sources(data.get("trusted_sources", ())),
            require_trusted_source=_coerce_bool(
                data.get("require_trusted_source", False),
                name="require_trusted_source",
            ),
            time_sensitive_only=_coerce_bool(data.get("time_sensitive_only", True), name="time_sensitive_only"),
        )


@dataclass(frozen=True)
class EvidenceQualityAssessment:
    """Result of applying an evidence quality policy to one evidence snippet."""

    passed: bool
    applied: bool
    reasons: tuple[str, ...] = ()
    source: str | None = None
    timestamp: str | None = None
    age_days: float | None = None
    trusted_source: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "passed": self.passed,
            "applied": self.applied,
            "reasons": self.reasons,
            "source": self.source,
            "timestamp": self.timestamp,
            "age_days": self.age_days,
            "trusted_source": self.trusted_source,
        }


@dataclass(frozen=True)
class EvidenceQualitySummary:
    """Aggregate quality assessment for a set of evidence snippets."""

    document_count: int
    applied_count: int
    passed_count: int
    failed_count: int
    reason_counts: Mapping[str, int] = field(default_factory=dict)
    assessments: Sequence[EvidenceQualityAssessment] = ()

    def __post_init__(self) -> None:
        document_count = _coerce_non_negative_int(self.document_count, name="document_count")
        applied_count = _coerce_non_negative_int(self.applied_count, name="applied_count")
        passed_count = _coerce_non_negative_int(self.passed_count, name="passed_count")
        failed_count = _coerce_non_negative_int(self.failed_count, name="failed_count")
        if passed_count + failed_count != applied_count:
            raise ValueError("passed_count plus failed_count must equal applied_count.")
        if applied_count > document_count:
            raise ValueError("applied_count cannot exceed document_count.")
        reason_counts = {
            str(reason): _coerce_non_negative_int(count, name=f"reason_counts[{reason!r}]")
            for reason, count in self.reason_counts.items()
        }
        object.__setattr__(self, "document_count", document_count)
        object.__setattr__(self, "applied_count", applied_count)
        object.__setattr__(self, "passed_count", passed_count)
        object.__setattr__(self, "failed_count", failed_count)
        object.__setattr__(self, "reason_counts", reason_counts)
        object.__setattr__(self, "assessments", tuple(self.assessments))

    @classmethod
    def from_assessments(
        cls,
        assessments: Sequence[EvidenceQualityAssessment],
        *,
        document_count: int | None = None,
    ) -> "EvidenceQualitySummary":
        """Build a summary from per-document assessments."""
        assessments = tuple(assessments)
        applied = tuple(item for item in assessments if item.applied)
        reason_counts: dict[str, int] = {}
        for item in applied:
            for reason in item.reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        return cls(
            document_count=len(assessments) if document_count is None else document_count,
            applied_count=len(applied),
            passed_count=sum(1 for item in applied if item.passed),
            failed_count=sum(1 for item in applied if not item.passed),
            reason_counts=reason_counts,
            assessments=assessments,
        )

    @property
    def status(self) -> str:
        """Return a compact status for release and trace summaries."""
        if self.document_count == 0:
            return "empty"
        if self.applied_count == 0:
            return "not_applied"
        if self.failed_count:
            return "fail"
        return "pass"

    @property
    def pass_rate(self) -> float:
        """Return the pass rate across policy-applied evidence snippets."""
        if self.applied_count == 0:
            return 1.0
        return self.passed_count / self.applied_count

    @property
    def failure_rate(self) -> float:
        """Return the failure rate across policy-applied evidence snippets."""
        if self.applied_count == 0:
            return 0.0
        return self.failed_count / self.applied_count

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "status": self.status,
            "document_count": self.document_count,
            "applied_count": self.applied_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "pass_rate": self.pass_rate,
            "failure_rate": self.failure_rate,
            "reason_counts": dict(self.reason_counts),
            "assessments": tuple(item.to_dict() for item in self.assessments),
        }


class _DocumentMatch(NamedTuple):
    document: EvidenceDocument
    overlap: float
    exact: bool
    negation_mismatch: bool


class _IndexedEvidenceDocument(NamedTuple):
    document: EvidenceDocument
    tokens: tuple[str, ...]
    key: str
    negated: bool


def assess_evidence_quality(
    document: EvidenceDocument | Mapping[str, Any] | str,
    *,
    policy: EvidenceQualityPolicy | Mapping[str, Any] | None,
    claim: Claim | None = None,
    features: Mapping[str, Any] | None = None,
) -> EvidenceQualityAssessment:
    """Assess one evidence snippet against a freshness/provenance policy."""
    feature_flags = _quality_features(claim=claim, features=features)
    return _assess_evidence_quality(
        _coerce_evidence(document),
        features=feature_flags,
        policy=_coerce_evidence_quality_policy(policy),
    )


def summarize_evidence_quality(
    evidence: (
        EvidenceDocument
        | Mapping[str, Any]
        | str
        | Sequence[EvidenceDocument | Mapping[str, Any] | str]
    ),
    *,
    policy: EvidenceQualityPolicy | Mapping[str, Any] | None,
    claim: Claim | None = None,
    features: Mapping[str, Any] | None = None,
) -> EvidenceQualitySummary:
    """Assess a batch of evidence snippets and return aggregate quality metrics."""
    documents = tuple(_coerce_evidence(item) for item in _evidence_sequence(evidence))
    feature_flags = _quality_features(claim=claim, features=features)
    quality_policy = _coerce_evidence_quality_policy(policy)
    assessments = tuple(
        _assess_evidence_quality(
            document,
            features=feature_flags,
            policy=quality_policy,
        )
        for document in documents
    )
    return EvidenceQualitySummary.from_assessments(assessments, document_count=len(documents))


@dataclass(frozen=True)
class GroundednessVerifier:
    """Lexical evidence-coverage verifier.

    This verifier is intentionally modest: it checks exact containment, token
    overlap, configured refutations, and simple negation mismatch. It is a stable
    dependency-free baseline for later retrieval or semantic entailment adapters.
    """

    evidence: Sequence[EvidenceDocument | Mapping[str, Any] | str]
    refutations: Mapping[str, Sequence[str] | str] = field(default_factory=dict)
    min_overlap: float = 0.65
    evidence_quality_policy: EvidenceQualityPolicy | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_overlap <= 1.0):
            raise ValueError("min_overlap must be in [0, 1].")
        evidence = tuple(_coerce_evidence(item) for item in self.evidence)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "_indexed_evidence", tuple(_index_document(item) for item in evidence))
        object.__setattr__(self, "refutations", _normalize_refutations(self.refutations))
        object.__setattr__(self, "evidence_quality_policy", _coerce_evidence_quality_policy(
            self.evidence_quality_policy,
        ))

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one claim against lexical evidence snippets."""
        claim_key = normalize_claim_text(claim.text)
        features = _claim_features(claim)
        claim_tokens = _tokens(claim.text)
        if not claim_tokens:
            return VerificationResult(
                status=VerificationStatus.NOT_APPLICABLE,
                confidence=0.0,
                explanation="groundedness verifier found no lexical tokens in claim",
                metadata={
                    "verifier": "groundedness_lexical",
                    "claim_key": claim_key,
                    "claim_features": features,
                },
            )

        refutation = _lookup_refutation(claim_key, self.refutations, context)
        if refutation is not None:
            return VerificationResult(
                status=VerificationStatus.REFUTED,
                confidence=0.9,
                evidence=refutation,
                explanation="configured refutation matched claim",
                metadata={
                    "verifier": "groundedness_lexical",
                    "claim_key": claim_key,
                    "claim_features": features,
                    "decision_rule": "configured_refutation",
                },
            )

        documents = _documents_with_context(self._indexed_evidence, context)
        best = _best_document_match(claim.text, claim_tokens, documents)
        if best is None:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.2,
                explanation="no evidence snippets were provided",
                metadata={
                    "verifier": "groundedness_lexical",
                    "claim_key": claim_key,
                    "claim_features": features,
                    "decision_rule": "no_evidence",
                },
            )

        evidence = (_evidence_label(best.document),)
        metadata = {
            "verifier": "groundedness_lexical",
            "claim_key": claim_key,
            "claim_features": features,
            "best_overlap": best.overlap,
            "best_source": best.document.source,
            "min_overlap": self.min_overlap,
        }
        quality = _assess_evidence_quality(
            best.document,
            features=features,
            policy=self.evidence_quality_policy,
        )
        if quality.applied:
            metadata["evidence_quality"] = quality.to_dict()
        if best.negation_mismatch and best.overlap >= self.min_overlap:
            if not quality.passed:
                return _quality_failure_result(evidence=evidence, metadata=metadata, quality=quality)
            return VerificationResult(
                status=VerificationStatus.REFUTED,
                confidence=min(0.85, 0.45 + 0.4 * best.overlap),
                evidence=evidence,
                explanation="best evidence has high token overlap with opposing negation",
                metadata={**metadata, "decision_rule": "negation_mismatch"},
            )
        if best.exact:
            if not quality.passed:
                return _quality_failure_result(evidence=evidence, metadata=metadata, quality=quality)
            return VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=0.9,
                evidence=evidence,
                explanation="claim text is contained in evidence",
                metadata={**metadata, "decision_rule": "exact_containment"},
            )
        if best.overlap >= self.min_overlap:
            if not quality.passed:
                return _quality_failure_result(evidence=evidence, metadata=metadata, quality=quality)
            return VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=min(0.85, 0.35 + 0.5 * best.overlap),
                evidence=evidence,
                explanation="claim tokens are covered by evidence above threshold",
                metadata={**metadata, "decision_rule": "token_overlap"},
            )
        explanation = "best evidence did not cover enough claim tokens"
        if features.get("is_time_sensitive"):
            explanation = f"{explanation}; time-sensitive claim needs fresh evidence"
        return VerificationResult(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            confidence=max(0.2, 0.5 * best.overlap),
            evidence=evidence,
            explanation=explanation,
            metadata={**metadata, "decision_rule": "low_overlap"},
        )

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        """Verify multiple claims."""
        return tuple(self.verify(claim, context=context) for claim in claims)


def _claim_features(claim: Claim) -> dict[str, bool]:
    raw_features = claim.metadata.get("features", {}) if isinstance(claim.metadata, Mapping) else {}
    if not isinstance(raw_features, Mapping):
        return {}
    return normalized_feature_flags(raw_features)


def _quality_features(
    *,
    claim: Claim | None,
    features: Mapping[str, Any] | None,
) -> dict[str, bool]:
    if features is not None:
        return normalized_feature_flags(features)
    if claim is None:
        return {}
    return _claim_features(claim)


def _coerce_evidence(value: EvidenceDocument | Mapping[str, Any] | str) -> EvidenceDocument:
    if isinstance(value, EvidenceDocument):
        return value
    if isinstance(value, str):
        return EvidenceDocument(text=value)
    return EvidenceDocument.from_dict(value)


def _coerce_evidence_quality_policy(
    value: EvidenceQualityPolicy | Mapping[str, Any] | None,
) -> EvidenceQualityPolicy | None:
    if value is None:
        return None
    if isinstance(value, EvidenceQualityPolicy):
        return value
    return EvidenceQualityPolicy.from_dict(value)


def _index_document(document: EvidenceDocument) -> _IndexedEvidenceDocument:
    tokens = _tokens(document.text)
    return _IndexedEvidenceDocument(
        document=document,
        tokens=tokens,
        key=normalize_claim_text(document.text),
        negated=_has_negation(tokens),
    )


def _normalize_refutations(refutations: Mapping[str, Sequence[str] | str]) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    for claim_text, evidence in refutations.items():
        if isinstance(evidence, str):
            normalized_evidence = (evidence,)
        else:
            normalized_evidence = tuple(str(item) for item in evidence)
        normalized[normalize_claim_text(claim_text)] = normalized_evidence
    return normalized


def _lookup_refutation(
    claim_key: str,
    refutations: Mapping[str, tuple[str, ...]],
    context: Mapping[str, Any] | None,
) -> tuple[str, ...] | None:
    if claim_key in refutations:
        return refutations[claim_key]
    if context is None or "refutations" not in context:
        return None
    context_refutations = _normalize_refutations(_as_mapping(context["refutations"], name="context refutations"))
    return context_refutations.get(claim_key)


def _documents_with_context(
    base_documents: Sequence[_IndexedEvidenceDocument],
    context: Mapping[str, Any] | None,
) -> tuple[_IndexedEvidenceDocument, ...]:
    documents = tuple(base_documents)
    if context is None or "evidence" not in context:
        return documents
    return documents + tuple(_index_document(_coerce_evidence(item)) for item in _as_sequence(context["evidence"]))


def _best_document_match(
    claim_text: str,
    claim_tokens: tuple[str, ...],
    documents: Sequence[_IndexedEvidenceDocument],
) -> _DocumentMatch | None:
    if not documents:
        return None
    claim_key = normalize_claim_text(claim_text)
    claim_negated = _has_negation(claim_tokens)
    matches = []
    for indexed in documents:
        exact = claim_key in indexed.key
        overlap = _token_overlap(claim_tokens, indexed.tokens)
        negation_mismatch = claim_negated != indexed.negated
        matches.append(_DocumentMatch(indexed.document, overlap, exact, negation_mismatch))
    return max(matches, key=lambda match: (match.exact, match.overlap))


def _assess_evidence_quality(
    document: EvidenceDocument,
    *,
    features: Mapping[str, bool],
    policy: EvidenceQualityPolicy | None,
) -> EvidenceQualityAssessment:
    if policy is None:
        return EvidenceQualityAssessment(passed=True, applied=False)
    if policy.time_sensitive_only and not bool(features.get("is_time_sensitive", False)):
        return EvidenceQualityAssessment(passed=True, applied=False)

    reasons: list[str] = []
    source = document.source
    trusted_source = _trusted_source(source, policy.trusted_sources)
    if policy.require_source and not source:
        reasons.append("missing_source")
    if policy.require_trusted_source and trusted_source is not True:
        reasons.append("untrusted_source" if source else "missing_source")

    timestamp = _document_timestamp(document)
    age_days = None
    if policy.max_age_days is not None:
        if timestamp is None:
            reasons.append("missing_timestamp")
        else:
            age_days = (policy.reference_time_or_now() - timestamp).total_seconds() / 86400.0
            if age_days < -1.0:
                reasons.append("future_timestamp")
            elif age_days > policy.max_age_days:
                reasons.append("stale_evidence")

    return EvidenceQualityAssessment(
        passed=not reasons,
        applied=True,
        reasons=tuple(dict.fromkeys(reasons)),
        source=source,
        timestamp=None if timestamp is None else timestamp.isoformat(),
        age_days=None if age_days is None else round(float(age_days), 6),
        trusted_source=trusted_source,
    )


def _quality_failure_result(
    *,
    evidence: tuple[str, ...],
    metadata: Mapping[str, Any],
    quality: EvidenceQualityAssessment,
) -> VerificationResult:
    reasons = ", ".join(quality.reasons) if quality.reasons else "unknown"
    return VerificationResult(
        status=VerificationStatus.INSUFFICIENT_EVIDENCE,
        confidence=0.35,
        evidence=evidence,
        explanation=f"best evidence failed quality policy: {reasons}",
        metadata={**dict(metadata), "decision_rule": "evidence_quality_failed"},
    )


def _document_timestamp(document: EvidenceDocument) -> datetime | None:
    for key in ("timestamp", "published_at", "updated_at", "retrieved_at", "as_of", "date"):
        raw_value = document.metadata.get(key)
        if raw_value is None:
            continue
        parsed = _parse_datetime(raw_value)
        if parsed is None:
            return None
        return parsed
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _coerce_reference_time(value: str | datetime | date | None) -> datetime | None:
    if value is None:
        return None
    parsed = _parse_datetime(value)
    if parsed is None:
        raise ValueError("reference_time must be an ISO datetime or YYYY-MM-DD date.")
    return parsed


def _trusted_source(source: str | None, trusted_sources: Sequence[str]) -> bool | None:
    if not trusted_sources:
        return None
    if not source:
        return False
    source_key = source.casefold()
    return any(str(trusted).casefold() in source_key for trusted in trusted_sources)


def _coerce_trusted_sources(value: Any) -> tuple[str, ...]:
    if value in (None, ()):
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise ValueError("trusted_sources must be a string or sequence of strings.")
    return tuple(str(source) for source in value)


def _coerce_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean.")


def _coerce_non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped.isdecimal():
            raise ValueError(f"{name} must be a non-negative integer.")
        parsed = int(stripped)
    else:
        raise ValueError(f"{name} must be a non-negative integer.")
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(text))


def _token_overlap(claim_tokens: Sequence[str], evidence_tokens: Sequence[str]) -> float:
    if not claim_tokens:
        return 0.0
    evidence_set = set(evidence_tokens)
    if not evidence_set:
        return 0.0
    covered = sum(1 for token in claim_tokens if token in evidence_set)
    return covered / len(claim_tokens)


def _has_negation(tokens: Sequence[str]) -> bool:
    token_set = set(tokens)
    return any(token in token_set for token in _NEGATION_TOKENS)


def _evidence_label(document: EvidenceDocument, *, max_chars: int = 220) -> str:
    snippet = " ".join(document.text.split())
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3].rstrip() + "..."
    if document.source:
        return f"{document.source}: {snippet}"
    return snippet


def _as_sequence(value: Any) -> Sequence[EvidenceDocument | Mapping[str, Any] | str]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return value
    raise ValueError("context evidence must be a string or sequence.")


def _evidence_sequence(
    value: EvidenceDocument | Mapping[str, Any] | str | Sequence[EvidenceDocument | Mapping[str, Any] | str],
) -> Sequence[EvidenceDocument | Mapping[str, Any] | str]:
    if isinstance(value, (EvidenceDocument, str, Mapping)):
        return (value,)
    if isinstance(value, Sequence):
        return value
    raise ValueError("evidence must be a snippet or sequence of snippets.")


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Sequence[str] | str]:
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{name} must be a mapping.")
