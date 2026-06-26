"""Dependency-free claim triple extraction and evidence-slot audits."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from eigentruth.json_utils import to_jsonable
from eigentruth.verify.groundedness import EvidenceDocument
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
_CAPITAL_OF_RE = re.compile(
    r"^(?P<object>.+?)\s+(?:is|was)\s+(?:the\s+)?capital\s+of\s+(?P<subject>.+)$",
    re.IGNORECASE,
)
_CAPITAL_SUBJECT_RE = re.compile(
    r"^(?:the\s+)?capital\s+of\s+(?P<subject>.+?)\s+(?:is|was)\s+(?P<object>.+)$",
    re.IGNORECASE,
)
_POSSESSIVE_CAPITAL_RE = re.compile(
    r"^(?P<subject>.+?)(?:'s|’s)\s+capital\s+(?:is|was)\s+(?P<object>.+)$",
    re.IGNORECASE,
)
_OFFICIAL_LANGUAGE_OF_RE = re.compile(
    r"^(?P<object>.+?)\s+(?:is|are|was|were)\s+(?:an?\s+|the\s+)?official\s+language\s+of\s+(?P<subject>.+)$",
    re.IGNORECASE,
)
_OFFICIAL_LANGUAGE_SUBJECT_RE = re.compile(
    r"^(?:the\s+)?official\s+languages?\s+of\s+(?P<subject>.+?)\s+"
    r"(?:include|includes|are|is|were|was)\s+(?P<object>.+)$",
    re.IGNORECASE,
)
_POSSESSIVE_OFFICIAL_LANGUAGE_RE = re.compile(
    r"^(?P<subject>.+?)(?:'s|’s)\s+official\s+languages?\s+"
    r"(?:include|includes|are|is|were|was)\s+(?P<object>.+)$",
    re.IGNORECASE,
)
_CURRENCY_OF_RE = re.compile(
    r"^(?P<object>.+?)\s+(?:is|are|was|were)\s+(?:a\s+|the\s+)?currency\s+of\s+(?P<subject>.+)$",
    re.IGNORECASE,
)
_CURRENCY_SUBJECT_RE = re.compile(
    r"^(?:the\s+)?currenc(?:y|ies)\s+of\s+(?P<subject>.+?)\s+"
    r"(?:include|includes|are|is|were|was)\s+(?P<object>.+)$",
    re.IGNORECASE,
)
_POSSESSIVE_CURRENCY_RE = re.compile(
    r"^(?P<subject>.+?)(?:'s|’s)\s+currenc(?:y|ies)\s+"
    r"(?:include|includes|are|is|were|was)\s+(?P<object>.+)$",
    re.IGNORECASE,
)
_USES_CURRENCY_RE = re.compile(
    r"^(?P<subject>.+?)\s+(?:uses|use|used)\s+(?P<object>.+?)\s+as\s+"
    r"(?:its\s+|their\s+|the\s+)?currenc(?:y|ies)$",
    re.IGNORECASE,
)
_LOCATED_IN_RE = re.compile(
    r"^(?P<subject>.+?)\s+(?:is|are|was|were)\s+(?:located\s+in|based\s+in)\s+(?P<object>.+)$",
    re.IGNORECASE,
)
_HAS_RE = re.compile(
    r"^(?P<subject>.+?)\s+(?P<predicate>has|have|had|contains|contain|includes|include)\s+(?P<object>.+)$",
    re.IGNORECASE,
)
_OBSERVATION_RE = re.compile(
    r"^(?P<subject>.+?)\s+(?P<predicate>enrolled|reported|found|showed|reduced|increased)\s+(?P<object>.+)$",
    re.IGNORECASE,
)
_IS_RE = re.compile(
    r"^(?P<subject>.+?)\s+(?:is|are|was|were)\s+(?P<object>.+)$",
    re.IGNORECASE,
)
_BOUNDARY_CHARS = " \t\r\n.,;:!?()[]{}\"'`“”‘’。！？"
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "had",
    "has",
    "have",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}
_PREDICATE_ALIASES = {
    "capital_of": ("capital",),
    "official_language_of": ("official", "language"),
    "currency_of": ("currency",),
    "located_in": ("located",),
    "is": (),
}
_LINK_GROUP_METADATA_KEYS = ("evidence_group", "document_group", "record_id", "source_record_id")
_LINK_CLAIM_METADATA_KEYS = ("claim_id", "claim_ids", "supports_claim_id", "supports_claim_ids")
_LINK_ENTITY_METADATA_KEYS = ("entity", "entities", "subject", "subjects", "subject_id", "entity_id")


@runtime_checkable
class ClaimTripleExtractor(Protocol):
    """Interface for pluggable claim-to-triple extractors."""

    def extract(self, claim: Claim) -> Sequence["ClaimTriple"]:
        """Return zero or more structured triples for a claim."""
        ...


@dataclass(frozen=True)
class ClaimTriple:
    """A conservative subject-predicate-object projection of one claim."""

    subject: str
    predicate: str
    object: str
    claim_id: str | None = None
    source_text: str = ""
    confidence: float = 0.5
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        subject = _clean_slot(self.subject)
        predicate = _clean_predicate(self.predicate)
        object_value = _clean_slot(self.object)
        if not subject:
            raise ValueError("triple subject must be non-empty.")
        if not predicate:
            raise ValueError("triple predicate must be non-empty.")
        if not object_value:
            raise ValueError("triple object must be non-empty.")
        confidence = _coerce_probability(self.confidence, name="confidence")
        claim_id = None if self.claim_id is None else str(self.claim_id)
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "object", object_value)
        object.__setattr__(self, "claim_id", claim_id)
        object.__setattr__(self, "source_text", str(self.source_text))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "claim_id": self.claim_id,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClaimTriple":
        """Build a triple from JSON-like data."""
        return cls(
            subject=str(data.get("subject", "")),
            predicate=str(data.get("predicate", "")),
            object=str(data.get("object", data.get("object_text", ""))),
            claim_id=None if data.get("claim_id") is None else str(data.get("claim_id")),
            source_text=str(data.get("source_text", "")),
            confidence=float(data.get("confidence", 0.5)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class LookupTripleExtractor:
    """Replay externally predicted triples from a claim-id/text lookup table."""

    predictions: Mapping[str, Sequence[ClaimTriple | Mapping[str, Any]]]
    extractor_name: str = "lookup_triple_extractor"
    prediction_source: str = "lookup_prediction"
    missing_policy: str = "empty"

    def __post_init__(self) -> None:
        predictions: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for key, triples in self.predictions.items():
            lookup_key = str(key).strip()
            if not lookup_key:
                raise ValueError("lookup triple prediction keys must be non-empty.")
            predictions[lookup_key] = _coerce_prediction_triple_payloads(triples)
        extractor_name = str(self.extractor_name).strip() or "lookup_triple_extractor"
        prediction_source = str(self.prediction_source).strip() or "lookup_prediction"
        missing_policy = str(self.missing_policy).strip().casefold().replace("-", "_")
        if missing_policy not in {"empty", "error"}:
            raise ValueError("missing_policy must be 'empty' or 'error'.")
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "extractor_name", extractor_name)
        object.__setattr__(self, "prediction_source", prediction_source)
        object.__setattr__(self, "missing_policy", missing_policy)

    def extract(self, claim: Claim) -> tuple[ClaimTriple, ...]:
        """Return externally supplied triples for this claim when present."""
        for key in _lookup_keys_for_claim(claim):
            raw_triples = self.predictions.get(key)
            if raw_triples is not None:
                return tuple(
                    _prediction_triple_for_claim(
                        item,
                        claim,
                        extractor_name=self.extractor_name,
                        prediction_source=self.prediction_source,
                    )
                    for item in raw_triples
                )
        if self.missing_policy == "error":
            raise KeyError(f"no predicted triples found for claim {claim.claim_id!r}")
        return ()


@dataclass(frozen=True)
class RuleBasedTripleExtractor:
    """Small rule-based triple extractor for audit fixtures and local gates."""

    extractor_name: str = "rule_based_triple_extractor"

    def extract(self, claim: Claim) -> tuple[ClaimTriple, ...]:
        """Extract triples from metadata first, then from simple text patterns."""
        metadata_triples = _metadata_triples(claim)
        if metadata_triples:
            return metadata_triples

        text = _clean_sentence(claim.text)
        if not text:
            return ()
        if _contains_blocked_extraction_context(text):
            return ()

        capital = _CAPITAL_OF_RE.match(text)
        if capital is not None:
            return (_triple(
                claim,
                subject=capital.group("subject"),
                predicate="capital_of",
                object_value=capital.group("object"),
                source="capital_of_rule",
            ),)

        capital_subject = _CAPITAL_SUBJECT_RE.match(text)
        if capital_subject is not None:
            return (_triple(
                claim,
                subject=capital_subject.group("subject"),
                predicate="capital_of",
                object_value=capital_subject.group("object"),
                source="capital_subject_rule",
            ),)

        possessive_capital = _POSSESSIVE_CAPITAL_RE.match(text)
        if possessive_capital is not None:
            return (_triple(
                claim,
                subject=possessive_capital.group("subject"),
                predicate="capital_of",
                object_value=possessive_capital.group("object"),
                source="possessive_capital_rule",
            ),)

        official_language = _OFFICIAL_LANGUAGE_OF_RE.match(text)
        if official_language is not None:
            return (_triple(
                claim,
                subject=official_language.group("subject"),
                predicate="official_language_of",
                object_value=official_language.group("object"),
                source="official_language_of_rule",
            ),)

        official_language_subject = _OFFICIAL_LANGUAGE_SUBJECT_RE.match(text)
        if official_language_subject is not None:
            return (_triple(
                claim,
                subject=official_language_subject.group("subject"),
                predicate="official_language_of",
                object_value=official_language_subject.group("object"),
                source="official_language_subject_rule",
            ),)

        possessive_official_language = _POSSESSIVE_OFFICIAL_LANGUAGE_RE.match(text)
        if possessive_official_language is not None:
            return (_triple(
                claim,
                subject=possessive_official_language.group("subject"),
                predicate="official_language_of",
                object_value=possessive_official_language.group("object"),
                source="possessive_official_language_rule",
            ),)

        currency = _CURRENCY_OF_RE.match(text)
        if currency is not None:
            return (_triple(
                claim,
                subject=currency.group("subject"),
                predicate="currency_of",
                object_value=currency.group("object"),
                source="currency_of_rule",
            ),)

        currency_subject = _CURRENCY_SUBJECT_RE.match(text)
        if currency_subject is not None:
            return (_triple(
                claim,
                subject=currency_subject.group("subject"),
                predicate="currency_of",
                object_value=currency_subject.group("object"),
                source="currency_subject_rule",
            ),)

        possessive_currency = _POSSESSIVE_CURRENCY_RE.match(text)
        if possessive_currency is not None:
            return (_triple(
                claim,
                subject=possessive_currency.group("subject"),
                predicate="currency_of",
                object_value=possessive_currency.group("object"),
                source="possessive_currency_rule",
            ),)

        uses_currency = _USES_CURRENCY_RE.match(text)
        if uses_currency is not None:
            return (_triple(
                claim,
                subject=uses_currency.group("subject"),
                predicate="currency_of",
                object_value=uses_currency.group("object"),
                source="uses_currency_rule",
            ),)

        located = _LOCATED_IN_RE.match(text)
        if located is not None:
            return (_triple(
                claim,
                subject=located.group("subject"),
                predicate="located_in",
                object_value=located.group("object"),
                source="located_in_rule",
            ),)

        has_match = _HAS_RE.match(text)
        if has_match is not None:
            return (_triple(
                claim,
                subject=has_match.group("subject"),
                predicate=has_match.group("predicate"),
                object_value=has_match.group("object"),
                source="has_rule",
            ),)

        observation = _OBSERVATION_RE.match(text)
        if observation is not None:
            return (_triple(
                claim,
                subject=observation.group("subject"),
                predicate=observation.group("predicate"),
                object_value=observation.group("object"),
                source="observation_rule",
            ),)

        is_match = _IS_RE.match(text)
        if is_match is not None:
            return (_triple(
                claim,
                subject=is_match.group("subject"),
                predicate="is",
                object_value=is_match.group("object"),
                source="is_rule",
            ),)
        return ()


@dataclass(frozen=True)
class RegexTriplePattern:
    """One dependency-free regex pattern for extracting a claim triple."""

    pattern: str
    predicate: str | None = None
    subject_group: str = "subject"
    predicate_group: str = "predicate"
    object_group: str = "object"
    confidence: float = 0.65
    source: str = "regex_triple_pattern"
    ignore_case: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pattern = str(self.pattern)
        if not pattern.strip():
            raise ValueError("regex triple pattern must be non-empty.")
        ignore_case = _coerce_bool(self.ignore_case, name="ignore_case")
        try:
            re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise ValueError(f"invalid regex triple pattern: {exc}") from exc
        predicate = None if self.predicate is None else _clean_predicate(self.predicate)
        subject_group = str(self.subject_group).strip()
        predicate_group = str(self.predicate_group).strip()
        object_group = str(self.object_group).strip()
        if not subject_group:
            raise ValueError("subject_group must be non-empty.")
        if predicate is None and not predicate_group:
            raise ValueError("predicate_group must be non-empty when predicate is not set.")
        if not object_group:
            raise ValueError("object_group must be non-empty.")
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "subject_group", subject_group)
        object.__setattr__(self, "predicate_group", predicate_group)
        object.__setattr__(self, "object_group", object_group)
        object.__setattr__(self, "confidence", _coerce_probability(self.confidence, name="confidence"))
        object.__setattr__(self, "source", str(self.source).strip() or "regex_triple_pattern")
        object.__setattr__(self, "ignore_case", ignore_case)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegexTriplePattern":
        """Build a regex triple pattern from JSON-like data."""
        return cls(
            pattern=str(data.get("pattern", "")),
            predicate=None if data.get("predicate") is None else str(data.get("predicate")),
            subject_group=str(data.get("subject_group", "subject")),
            predicate_group=str(data.get("predicate_group", "predicate")),
            object_group=str(data.get("object_group", "object")),
            confidence=data.get("confidence", 0.65),
            source=str(data.get("source", "regex_triple_pattern")),
            ignore_case=data.get("ignore_case", True),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "pattern": self.pattern,
            "predicate": self.predicate,
            "subject_group": self.subject_group,
            "predicate_group": self.predicate_group,
            "object_group": self.object_group,
            "confidence": self.confidence,
            "source": self.source,
            "ignore_case": self.ignore_case,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def extract(self, claim: Claim) -> ClaimTriple | None:
        """Extract one triple from a claim if this pattern matches."""
        flags = re.IGNORECASE if self.ignore_case else 0
        match = re.match(self.pattern, _clean_sentence(claim.text), flags)
        if match is None:
            return None
        try:
            subject = match.group(self.subject_group)
            object_value = match.group(self.object_group)
            predicate = self.predicate if self.predicate is not None else match.group(self.predicate_group)
        except IndexError as exc:
            raise ValueError("regex triple pattern group was not present in match.") from exc
        metadata = {
            "extractor": "regex_triple_extractor",
            "source": self.source,
            "pattern": self.pattern,
            **dict(self.metadata),
        }
        return ClaimTriple(
            subject=subject,
            predicate=predicate,
            object=object_value,
            claim_id=claim.claim_id,
            source_text=claim.text,
            confidence=self.confidence,
            metadata=metadata,
        )


@dataclass(frozen=True)
class RegexTripleExtractor:
    """Configurable regex-based triple extractor with optional fallback."""

    patterns: Sequence[RegexTriplePattern | Mapping[str, Any]]
    fallback: ClaimTripleExtractor | None = None
    stop_on_first: bool = True
    preserve_metadata_triples: bool = True
    extractor_name: str = "regex_triple_extractor"

    def __post_init__(self) -> None:
        patterns = tuple(_coerce_regex_pattern(pattern) for pattern in self.patterns)
        if not patterns:
            raise ValueError("RegexTripleExtractor requires at least one pattern.")
        object.__setattr__(self, "patterns", patterns)
        object.__setattr__(self, "stop_on_first", _coerce_bool(self.stop_on_first, name="stop_on_first"))
        object.__setattr__(
            self,
            "preserve_metadata_triples",
            _coerce_bool(self.preserve_metadata_triples, name="preserve_metadata_triples"),
        )
        object.__setattr__(self, "extractor_name", str(self.extractor_name).strip() or "regex_triple_extractor")

    def extract(self, claim: Claim) -> tuple[ClaimTriple, ...]:
        """Extract triples from regex patterns, falling back when configured."""
        if self.preserve_metadata_triples:
            metadata_triples = _metadata_triples(claim)
            if metadata_triples:
                return metadata_triples
        text = _clean_sentence(claim.text)
        if not text or _contains_blocked_extraction_context(text):
            return ()
        triples: list[ClaimTriple] = []
        seen: set[tuple[str, str, str]] = set()
        for pattern in self.patterns:
            triple = pattern.extract(claim)
            if triple is None:
                continue
            key = _triple_key(triple)
            if key not in seen:
                triples.append(triple)
                seen.add(key)
            if self.stop_on_first:
                return tuple(triples)
        if triples:
            return tuple(triples)
        if self.fallback is None:
            return ()
        return tuple(self.fallback.extract(claim))


@dataclass(frozen=True)
class CompositeTripleExtractor:
    """Combine multiple claim triple extractors in order."""

    extractors: Sequence[ClaimTripleExtractor]
    stop_on_first_non_empty: bool = True
    dedupe: bool = True
    extractor_name: str = "composite_triple_extractor"

    def __post_init__(self) -> None:
        extractors = tuple(self.extractors)
        if not extractors:
            raise ValueError("CompositeTripleExtractor requires at least one extractor.")
        object.__setattr__(self, "extractors", extractors)
        object.__setattr__(
            self,
            "stop_on_first_non_empty",
            _coerce_bool(self.stop_on_first_non_empty, name="stop_on_first_non_empty"),
        )
        object.__setattr__(self, "dedupe", _coerce_bool(self.dedupe, name="dedupe"))
        object.__setattr__(self, "extractor_name", str(self.extractor_name).strip() or "composite_triple_extractor")

    def extract(self, claim: Claim) -> tuple[ClaimTriple, ...]:
        """Extract triples by trying each configured extractor."""
        triples: list[ClaimTriple] = []
        seen: set[tuple[str, str, str]] = set()
        for extractor in self.extractors:
            extracted = tuple(extractor.extract(claim))
            if not extracted:
                continue
            for triple in extracted:
                key = _triple_key(triple)
                if self.dedupe and key in seen:
                    continue
                triples.append(triple)
                seen.add(key)
            if self.stop_on_first_non_empty:
                break
        return tuple(triples)


@dataclass(frozen=True)
class TripleSlotEvidence:
    """Evidence detail for one subject, predicate, or object slot."""

    slot: str
    expected: str
    coverage: float
    covered: bool
    evidence: str | None = None
    source: str | None = None
    matched_tokens: Sequence[str] = ()
    missing_tokens: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _valid_slot_name(self.slot))
        object.__setattr__(self, "expected", str(self.expected))
        object.__setattr__(self, "coverage", _coerce_probability(self.coverage, name=f"{self.slot} coverage"))
        object.__setattr__(self, "covered", bool(self.covered))
        object.__setattr__(self, "evidence", None if self.evidence is None else str(self.evidence))
        object.__setattr__(self, "source", None if self.source is None else str(self.source))
        object.__setattr__(self, "matched_tokens", tuple(str(token) for token in self.matched_tokens))
        object.__setattr__(self, "missing_tokens", tuple(str(token) for token in self.missing_tokens))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "slot": self.slot,
            "expected": self.expected,
            "coverage": self.coverage,
            "covered": self.covered,
            "evidence": self.evidence,
            "source": self.source,
            "matched_tokens": tuple(self.matched_tokens),
            "missing_tokens": tuple(self.missing_tokens),
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class TripleEvidenceAudit:
    """Slot-level evidence coverage for one extracted triple."""

    triple: ClaimTriple
    passed: bool
    evidence: tuple[str, ...] = ()
    covered_slots: tuple[str, ...] = ()
    missing_slots: tuple[str, ...] = ()
    slot_coverage: Mapping[str, float] = field(default_factory=dict)
    slot_evidence: Sequence[TripleSlotEvidence | Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        triple = self.triple if isinstance(self.triple, ClaimTriple) else ClaimTriple.from_dict(self.triple)
        covered_slots = tuple(_valid_slot_name(slot) for slot in self.covered_slots)
        missing_slots = tuple(_valid_slot_name(slot) for slot in self.missing_slots)
        slot_coverage = {
            str(key): _coerce_probability(value, name=f"{key} coverage")
            for key, value in self.slot_coverage.items()
        }
        slot_evidence = tuple(_coerce_slot_evidence(item) for item in self.slot_evidence)
        object.__setattr__(self, "triple", triple)
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "evidence", tuple(str(item) for item in self.evidence))
        object.__setattr__(self, "covered_slots", covered_slots)
        object.__setattr__(self, "missing_slots", missing_slots)
        object.__setattr__(self, "slot_coverage", slot_coverage)
        object.__setattr__(self, "slot_evidence", slot_evidence)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "triple": self.triple.to_dict(),
            "passed": self.passed,
            "evidence": tuple(self.evidence),
            "covered_slots": tuple(self.covered_slots),
            "missing_slots": tuple(self.missing_slots),
            "slot_coverage": dict(self.slot_coverage),
            "slot_evidence": tuple(item.to_dict() for item in self.slot_evidence),
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class TripleEvidenceAuditReport:
    """Audit report for all triples extracted from one claim."""

    claim_id: str | None
    audits: Sequence[TripleEvidenceAudit | Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        claim_id = None if self.claim_id is None else str(self.claim_id)
        audits = tuple(_coerce_audit(item) for item in self.audits)
        object.__setattr__(self, "claim_id", claim_id)
        object.__setattr__(self, "audits", audits)

    @property
    def triple_count(self) -> int:
        """Return the number of audited triples."""
        return len(self.audits)

    @property
    def passed_count(self) -> int:
        """Return the number of triples whose slots were fully covered."""
        return sum(1 for audit in self.audits if audit.passed)

    @property
    def failed_count(self) -> int:
        """Return the number of triples with missing evidence slots."""
        return self.triple_count - self.passed_count

    @property
    def covered_slot_count(self) -> int:
        """Return the number of covered slots across all audited triples."""
        return sum(len(audit.covered_slots) for audit in self.audits)

    @property
    def missing_slot_count(self) -> int:
        """Return the number of missing slots across all audited triples."""
        return sum(len(audit.missing_slots) for audit in self.audits)

    @property
    def passed(self) -> bool:
        """Return true when at least one triple exists and all triples passed."""
        return self.triple_count > 0 and self.failed_count == 0

    @property
    def slot_summary(self) -> Mapping[str, Mapping[str, Any]]:
        """Return claim-level slot coverage statistics."""
        return _slot_summary(self.audits)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "claim_id": self.claim_id,
            "triple_count": self.triple_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "covered_slot_count": self.covered_slot_count,
            "missing_slot_count": self.missing_slot_count,
            "passed": self.passed,
            "slot_summary": to_jsonable(dict(self.slot_summary)),
            "audits": tuple(audit.to_dict() for audit in self.audits),
        }


@dataclass(frozen=True)
class TripleEvidenceVerifier:
    """Verifier that requires evidence to cover extracted triple slots."""

    evidence: Sequence[EvidenceDocument | Mapping[str, Any] | str] = ()
    extractor: ClaimTripleExtractor = field(default_factory=RuleBasedTripleExtractor)
    min_slot_coverage: float = 1.0

    def __post_init__(self) -> None:
        min_slot_coverage = _coerce_probability(self.min_slot_coverage, name="min_slot_coverage")
        object.__setattr__(self, "evidence", tuple(_coerce_evidence(item) for item in self.evidence))
        object.__setattr__(self, "min_slot_coverage", min_slot_coverage)

    def audit(self, claim: Claim, context: Mapping[str, Any] | None = None) -> TripleEvidenceAuditReport:
        """Return a slot-level evidence audit for one claim."""
        triples = tuple(self.extractor.extract(claim))
        documents = _documents_with_context(self.evidence, context)
        audits = tuple(
            _audit_triple(triple, documents, min_slot_coverage=self.min_slot_coverage)
            for triple in triples
        )
        return TripleEvidenceAuditReport(claim_id=claim.claim_id, audits=audits)

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify a claim by requiring all extracted triple slots to be covered."""
        report = self.audit(claim, context=context)
        metadata = {
            "verifier": "triple_evidence",
            "decision_rule": "triple_slot_coverage",
            "audit_report": report.to_dict(),
            "min_slot_coverage": self.min_slot_coverage,
        }
        if report.triple_count == 0:
            return VerificationResult(
                status=VerificationStatus.NOT_APPLICABLE,
                confidence=0.0,
                explanation="triple evidence verifier extracted no supported triples",
                metadata=metadata,
            )
        evidence = _unique_evidence_label(report.audits)
        if report.passed:
            return VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=0.85,
                evidence=evidence,
                explanation="all extracted claim triples have subject, predicate, and object evidence coverage",
                metadata=metadata,
            )
        explanation = "one or more extracted claim triples have missing evidence slots"
        if any(
            audit.metadata.get("evidence_link_passed") is False
            for audit in report.audits
        ):
            explanation = "one or more extracted claim triples have unlinked evidence slots"
        return VerificationResult(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            confidence=0.35,
            evidence=evidence,
            explanation=explanation,
            metadata=metadata,
        )

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        """Verify multiple claims."""
        return tuple(self.verify(claim, context=context) for claim in claims)


def extract_claim_triples(
    claim: Claim,
    *,
    extractor: ClaimTripleExtractor | None = None,
) -> tuple[ClaimTriple, ...]:
    """Extract structured triples from one claim."""
    active_extractor = RuleBasedTripleExtractor() if extractor is None else extractor
    return tuple(active_extractor.extract(claim))


def audit_claim_triples(
    claim: Claim,
    evidence: Sequence[EvidenceDocument | Mapping[str, Any] | str],
    *,
    extractor: ClaimTripleExtractor | None = None,
    min_slot_coverage: float = 1.0,
    context: Mapping[str, Any] | None = None,
) -> TripleEvidenceAuditReport:
    """Audit claim triples against evidence without constructing a verifier."""
    verifier = TripleEvidenceVerifier(
        evidence=evidence,
        extractor=RuleBasedTripleExtractor() if extractor is None else extractor,
        min_slot_coverage=min_slot_coverage,
    )
    return verifier.audit(claim, context=context)


def _metadata_triples(claim: Claim) -> tuple[ClaimTriple, ...]:
    if not isinstance(claim.metadata, Mapping):
        return ()
    raw_triples = claim.metadata.get("triples", claim.metadata.get("claim_triples"))
    if raw_triples is None:
        return ()
    if isinstance(raw_triples, Mapping):
        items = (raw_triples,)
    elif isinstance(raw_triples, Sequence) and not isinstance(raw_triples, (str, bytes, bytearray)):
        items = raw_triples
    else:
        raise ValueError("claim metadata triples must be a mapping or sequence of mappings.")
    triples = []
    for item in items:
        if isinstance(item, ClaimTriple):
            triples.append(item)
        elif isinstance(item, Mapping):
            payload = dict(item)
            payload.setdefault("claim_id", claim.claim_id)
            payload.setdefault("source_text", claim.text)
            payload.setdefault("metadata", {"source": "claim_metadata"})
            triples.append(ClaimTriple.from_dict(payload))
        else:
            raise ValueError("claim metadata triples must contain mappings.")
    return tuple(triples)


def _coerce_prediction_triple_payloads(
    triples: Sequence[ClaimTriple | Mapping[str, Any]] | Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(triples, ClaimTriple):
        return (triples.to_dict(),)
    if isinstance(triples, Mapping):
        if _looks_like_triple_payload(triples):
            return (dict(triples),)
        raw = (
            triples.get("triples")
            or triples.get("claim_triples")
            or triples.get("predicted_triples")
            or triples.get("prediction_triples")
        )
        if raw is None:
            return ()
        return _coerce_prediction_triple_payloads(raw)
    if isinstance(triples, Sequence) and not isinstance(triples, (str, bytes, bytearray)):
        payloads = []
        for item in triples:
            if isinstance(item, ClaimTriple):
                payloads.append(item.to_dict())
            elif isinstance(item, Mapping):
                if not _looks_like_triple_payload(item):
                    raise ValueError("predicted triple mappings must contain subject, predicate, and object.")
                payloads.append(dict(item))
            else:
                raise ValueError("predicted triples must contain mappings.")
        return tuple(payloads)
    raise ValueError("predicted triples must be a mapping or sequence of mappings.")


def _looks_like_triple_payload(value: Mapping[str, Any]) -> bool:
    return (
        value.get("subject") is not None
        and value.get("predicate") is not None
        and (value.get("object") is not None or value.get("object_text") is not None)
    )


def _prediction_triple_for_claim(
    payload: Mapping[str, Any],
    claim: Claim,
    *,
    extractor_name: str,
    prediction_source: str,
) -> ClaimTriple:
    data = dict(payload)
    data.setdefault("claim_id", claim.claim_id)
    data.setdefault("source_text", claim.text)
    metadata = dict(data.get("metadata", {}))
    metadata.setdefault("extractor", extractor_name)
    metadata.setdefault("source", prediction_source)
    data["metadata"] = metadata
    return ClaimTriple.from_dict(data)


def _lookup_keys_for_claim(claim: Claim) -> tuple[str, ...]:
    keys: list[str] = []
    if claim.claim_id is not None and str(claim.claim_id).strip():
        claim_id = str(claim.claim_id).strip()
        keys.extend((f"claim_id:{claim_id}", claim_id))
    text = _clean_sentence(claim.text)
    if text:
        keys.extend((f"text:{text}", f"text_norm:{_clean_slot(text).casefold()}", text))
    seen: set[str] = set()
    unique = []
    for key in keys:
        if key not in seen:
            unique.append(key)
            seen.add(key)
    return tuple(unique)


def _triple(
    claim: Claim,
    *,
    subject: str,
    predicate: str,
    object_value: str,
    source: str,
) -> ClaimTriple:
    return ClaimTriple(
        subject=subject,
        predicate=predicate,
        object=object_value,
        claim_id=claim.claim_id,
        source_text=claim.text,
        confidence=0.55,
        metadata={"extractor": "rule_based_triple_extractor", "source": source},
    )


def _audit_triple(
    triple: ClaimTriple,
    documents: Sequence[EvidenceDocument],
    *,
    min_slot_coverage: float,
) -> TripleEvidenceAudit:
    if not documents:
        return TripleEvidenceAudit(
            triple=triple,
            passed=False,
            covered_slots=(),
            missing_slots=("subject", "predicate", "object"),
            slot_coverage={"subject": 0.0, "predicate": 0.0, "object": 0.0},
            slot_evidence=_slot_evidence_items(
                triple,
                {"subject": 0.0, "predicate": 0.0, "object": 0.0},
                {},
                min_slot_coverage=min_slot_coverage,
            ),
            metadata={"decision_rule": "no_evidence"},
        )
    scored = [_score_document(triple, document, min_slot_coverage=min_slot_coverage) for document in documents]
    best = max(scored, key=lambda item: (len(item.covered_slots), sum(item.slot_coverage.values())))
    aggregate = _aggregate_scored_documents(scored, min_slot_coverage=min_slot_coverage)
    if _aggregate_is_better(aggregate, best):
        return aggregate
    return _audit_from_scored_document(triple, best, min_slot_coverage=min_slot_coverage)


def _audit_from_scored_document(
    triple: ClaimTriple,
    scored: "_ScoredDocument",
    *,
    min_slot_coverage: float,
) -> TripleEvidenceAudit:
    return TripleEvidenceAudit(
        triple=triple,
        passed=not scored.missing_slots,
        evidence=(_evidence_label(scored.document),),
        covered_slots=scored.covered_slots,
        missing_slots=scored.missing_slots,
        slot_coverage=scored.slot_coverage,
        slot_evidence=_slot_evidence_items(
            triple,
            scored.slot_coverage,
            {slot: scored.document for slot in ("subject", "predicate", "object")},
            min_slot_coverage=min_slot_coverage,
        ),
        metadata={
            "decision_rule": "single_document_slot_coverage",
            "best_source": scored.document.source,
        },
    )


@dataclass(frozen=True)
class _ScoredDocument:
    triple: ClaimTriple
    document: EvidenceDocument
    covered_slots: tuple[str, ...]
    missing_slots: tuple[str, ...]
    slot_coverage: Mapping[str, float]


def _aggregate_scored_documents(
    scored: Sequence[_ScoredDocument],
    *,
    min_slot_coverage: float,
) -> TripleEvidenceAudit:
    slot_coverage: dict[str, float] = {}
    slot_documents: dict[str, EvidenceDocument] = {}
    for slot in ("subject", "predicate", "object"):
        best_for_slot = max(scored, key=lambda item: item.slot_coverage.get(slot, 0.0))
        slot_coverage[slot] = best_for_slot.slot_coverage.get(slot, 0.0)
        slot_documents[slot] = best_for_slot.document
    covered = tuple(slot for slot, value in slot_coverage.items() if value >= min_slot_coverage)
    missing = tuple(slot for slot in ("subject", "predicate", "object") if slot not in covered)
    evidence_documents = _unique_slot_documents(
        slot_documents[slot]
        for slot, value in slot_coverage.items()
        if value > 0.0
    )
    if not evidence_documents:
        evidence_documents = (max(scored, key=lambda item: sum(item.slot_coverage.values())).document,)
    link_metadata = _multi_document_link_metadata(
        scored[0].triple,
        evidence_documents,
        min_slot_coverage=min_slot_coverage,
    )
    return TripleEvidenceAudit(
        triple=scored[0].triple,
        passed=not missing and bool(link_metadata["evidence_link_passed"]),
        evidence=tuple(_evidence_label(document) for document in evidence_documents),
        covered_slots=covered,
        missing_slots=missing,
        slot_coverage=slot_coverage,
        slot_evidence=_slot_evidence_items(
            scored[0].triple,
            slot_coverage,
            slot_documents,
            min_slot_coverage=min_slot_coverage,
        ),
        metadata={
            "decision_rule": (
                "multi_document_slot_coverage"
                if len(evidence_documents) > 1
                else "single_document_slot_coverage"
            ),
            "slot_sources": {
                slot: slot_documents[slot].source
                for slot in ("subject", "predicate", "object")
            },
            "slot_evidence": {
                slot: _evidence_label(slot_documents[slot])
                for slot in ("subject", "predicate", "object")
            },
            "evidence_document_count": len(evidence_documents),
            **link_metadata,
        },
    )


def _aggregate_is_better(
    aggregate: TripleEvidenceAudit,
    best: _ScoredDocument,
) -> bool:
    if not best.missing_slots and not aggregate.passed:
        return False
    aggregate_covered = len(aggregate.covered_slots)
    best_covered = len(best.covered_slots)
    if aggregate_covered > best_covered:
        return True
    if aggregate_covered < best_covered:
        return False
    return sum(aggregate.slot_coverage.values()) > sum(best.slot_coverage.values())


def _unique_slot_documents(documents: Sequence[EvidenceDocument]) -> tuple[EvidenceDocument, ...]:
    unique: list[EvidenceDocument] = []
    seen: set[tuple[str | None, str]] = set()
    for document in documents:
        key = (document.source, document.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(document)
    return tuple(unique)


def _multi_document_link_metadata(
    triple: ClaimTriple,
    documents: Sequence[EvidenceDocument],
    *,
    min_slot_coverage: float,
) -> dict[str, Any]:
    if len(documents) <= 1:
        return {
            "evidence_link_passed": True,
            "evidence_link_rule": "single_document",
        }
    shared_source = _shared_non_empty_value(tuple(document.source for document in documents))
    if shared_source is not None:
        return {
            "evidence_link_passed": True,
            "evidence_link_rule": "shared_source",
            "evidence_link_value": shared_source,
        }
    shared_group = _shared_metadata_value(documents, _LINK_GROUP_METADATA_KEYS)
    if shared_group is not None:
        return {
            "evidence_link_passed": True,
            "evidence_link_rule": "shared_metadata_group",
            "evidence_link_value": shared_group,
        }
    if triple.claim_id is not None and all(
        _metadata_contains(document.metadata, _LINK_CLAIM_METADATA_KEYS, triple.claim_id)
        for document in documents
    ):
        return {
            "evidence_link_passed": True,
            "evidence_link_rule": "claim_id_metadata",
            "evidence_link_value": triple.claim_id,
        }
    if all(
        _metadata_contains_slot(document.metadata, _LINK_ENTITY_METADATA_KEYS, triple.subject)
        for document in documents
    ):
        return {
            "evidence_link_passed": True,
            "evidence_link_rule": "subject_metadata",
            "evidence_link_value": triple.subject,
        }
    subject_tokens = _slot_tokens(triple.subject)
    if all(
        _slot_coverage(subject_tokens, set(_tokens(document.text))) >= min_slot_coverage
        for document in documents
    ):
        return {
            "evidence_link_passed": True,
            "evidence_link_rule": "subject_text_anchor",
            "evidence_link_value": triple.subject,
        }
    return {
        "evidence_link_passed": False,
        "evidence_link_rule": "unlinked_multi_document_evidence",
        "evidence_link_value": None,
    }


def _shared_non_empty_value(values: Sequence[str | None]) -> str | None:
    normalized = tuple(str(value).strip() for value in values if value is not None and str(value).strip())
    if not normalized:
        return None
    first = normalized[0]
    if len(normalized) == len(values) and all(value == first for value in normalized):
        return first
    return None


def _shared_metadata_value(
    documents: Sequence[EvidenceDocument],
    keys: Sequence[str],
) -> str | None:
    value_sets = [
        set(_metadata_values(document.metadata, keys))
        for document in documents
    ]
    if not value_sets or any(not values for values in value_sets):
        return None
    shared = set.intersection(*value_sets)
    if not shared:
        return None
    return sorted(shared)[0]


def _metadata_contains(metadata: Mapping[str, Any], keys: Sequence[str], expected: str) -> bool:
    expected_key = _metadata_key(expected)
    return any(_metadata_key(value) == expected_key for value in _metadata_values(metadata, keys))


def _metadata_contains_slot(metadata: Mapping[str, Any], keys: Sequence[str], expected: str) -> bool:
    expected_tokens = set(_slot_tokens(expected))
    if not expected_tokens:
        return False
    for value in _metadata_values(metadata, keys):
        value_tokens = set(_slot_tokens(value))
        if expected_tokens <= value_tokens or value_tokens <= expected_tokens:
            return True
    return False


def _metadata_values(metadata: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for key in keys:
        raw_value = metadata.get(key)
        if raw_value is None:
            continue
        for item in _metadata_sequence(raw_value):
            text = str(item).strip()
            if text:
                values.append(text)
    return tuple(values)


def _metadata_sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return (value,)
    return tuple(value)


def _metadata_key(value: Any) -> str:
    return " ".join(_tokens(str(value)))


def _coerce_regex_pattern(value: RegexTriplePattern | Mapping[str, Any]) -> RegexTriplePattern:
    if isinstance(value, RegexTriplePattern):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("regex triple patterns must be RegexTriplePattern or mapping values.")
    return RegexTriplePattern.from_dict(value)


def _triple_key(triple: ClaimTriple) -> tuple[str, str, str]:
    return (
        " ".join(_slot_tokens(triple.subject)),
        _clean_predicate(triple.predicate),
        " ".join(_slot_tokens(triple.object)),
    )


def _slot_evidence_items(
    triple: ClaimTriple,
    slot_coverage: Mapping[str, float],
    slot_documents: Mapping[str, EvidenceDocument],
    *,
    min_slot_coverage: float,
) -> tuple[TripleSlotEvidence, ...]:
    items = []
    for slot in ("subject", "predicate", "object"):
        coverage = slot_coverage.get(slot, 0.0)
        document = slot_documents.get(slot)
        evidence_tokens = set(_tokens(document.text)) if document is not None else set()
        expected_tokens = _expected_slot_tokens(triple, slot)
        matched_tokens = tuple(token for token in expected_tokens if token in evidence_tokens)
        missing_tokens = tuple(token for token in expected_tokens if token not in evidence_tokens)
        items.append(
            TripleSlotEvidence(
                slot=slot,
                expected=_expected_slot_value(triple, slot),
                coverage=coverage,
                covered=coverage >= min_slot_coverage,
                evidence=None if document is None else _evidence_label(document),
                source=None if document is None else document.source,
                matched_tokens=matched_tokens,
                missing_tokens=missing_tokens,
            )
        )
    return tuple(items)


def _slot_summary(audits: Sequence[TripleEvidenceAudit]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for slot in ("subject", "predicate", "object"):
        values = [audit.slot_coverage.get(slot, 0.0) for audit in audits]
        slot_evidence = [
            item
            for audit in audits
            for item in audit.slot_evidence
            if item.slot == slot
        ]
        sources = tuple(
            sorted({
                item.source
                for item in slot_evidence
                if item.source is not None and str(item.source).strip()
            })
        )
        summary[slot] = {
            "mean_coverage": 0.0 if not values else sum(values) / len(values),
            "min_coverage": None if not values else min(values),
            "max_coverage": None if not values else max(values),
            "covered_count": sum(1 for audit in audits if slot in audit.covered_slots),
            "missing_count": sum(1 for audit in audits if slot in audit.missing_slots),
            "sources": sources,
        }
    return summary


def _expected_slot_value(triple: ClaimTriple, slot: str) -> str:
    if slot == "subject":
        return triple.subject
    if slot == "predicate":
        return triple.predicate
    if slot == "object":
        return triple.object
    raise ValueError("slot must be subject, predicate, or object.")


def _expected_slot_tokens(triple: ClaimTriple, slot: str) -> tuple[str, ...]:
    if slot == "predicate":
        return _predicate_tokens(triple.predicate)
    return _slot_tokens(_expected_slot_value(triple, slot))


def _score_document(
    triple: ClaimTriple,
    document: EvidenceDocument,
    *,
    min_slot_coverage: float,
) -> _ScoredDocument:
    evidence_tokens = set(_tokens(document.text))
    coverage = {
        "subject": _slot_coverage(_slot_tokens(triple.subject), evidence_tokens),
        "predicate": _slot_coverage(_predicate_tokens(triple.predicate), evidence_tokens),
        "object": _slot_coverage(_slot_tokens(triple.object), evidence_tokens),
    }
    covered = tuple(slot for slot, value in coverage.items() if value >= min_slot_coverage)
    missing = tuple(slot for slot in ("subject", "predicate", "object") if slot not in covered)
    return _ScoredDocument(
        triple=triple,
        document=document,
        covered_slots=covered,
        missing_slots=missing,
        slot_coverage=coverage,
    )


def _documents_with_context(
    base_documents: Sequence[EvidenceDocument],
    context: Mapping[str, Any] | None,
) -> tuple[EvidenceDocument, ...]:
    documents = tuple(base_documents)
    if context is None or "evidence" not in context:
        return documents
    return documents + tuple(_coerce_evidence(item) for item in _as_sequence(context["evidence"]))


def _coerce_evidence(value: EvidenceDocument | Mapping[str, Any] | str) -> EvidenceDocument:
    if isinstance(value, EvidenceDocument):
        return value
    if isinstance(value, str):
        return EvidenceDocument(text=value)
    return EvidenceDocument.from_dict(value)


def _coerce_audit(value: TripleEvidenceAudit | Mapping[str, Any]) -> TripleEvidenceAudit:
    if isinstance(value, TripleEvidenceAudit):
        return value
    return TripleEvidenceAudit(
        triple=ClaimTriple.from_dict(_as_mapping(value.get("triple"), name="audit triple")),
        passed=bool(value.get("passed", False)),
        evidence=tuple(str(item) for item in _as_sequence(value.get("evidence", ()))),
        covered_slots=tuple(str(item) for item in _as_sequence(value.get("covered_slots", ()))),
        missing_slots=tuple(str(item) for item in _as_sequence(value.get("missing_slots", ()))),
        slot_coverage=dict(value.get("slot_coverage", {})),
        slot_evidence=tuple(_as_sequence(value.get("slot_evidence", ()))),
        metadata=dict(value.get("metadata", {})),
    )


def _coerce_slot_evidence(value: TripleSlotEvidence | Mapping[str, Any]) -> TripleSlotEvidence:
    if isinstance(value, TripleSlotEvidence):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("slot evidence must be a mapping.")
    return TripleSlotEvidence(
        slot=str(value.get("slot", "")),
        expected=str(value.get("expected", "")),
        coverage=value.get("coverage", 0.0),
        covered=bool(value.get("covered", False)),
        evidence=None if value.get("evidence") is None else str(value.get("evidence")),
        source=None if value.get("source") is None else str(value.get("source")),
        matched_tokens=tuple(str(item) for item in _as_sequence(value.get("matched_tokens", ()))),
        missing_tokens=tuple(str(item) for item in _as_sequence(value.get("missing_tokens", ()))),
        metadata=dict(value.get("metadata", {})),
    )


def _unique_evidence_label(audits: Sequence[TripleEvidenceAudit]) -> tuple[str, ...]:
    labels: list[str] = []
    seen = set()
    for audit in audits:
        for item in audit.evidence:
            if item not in seen:
                labels.append(item)
                seen.add(item)
    return tuple(labels)


def _evidence_label(document: EvidenceDocument) -> str:
    if document.source:
        return f"{document.source}: {document.text}"
    return document.text


def _slot_coverage(slot_tokens: Sequence[str], evidence_tokens: set[str]) -> float:
    if not slot_tokens:
        return 1.0
    if not evidence_tokens:
        return 0.0
    covered = sum(1 for token in slot_tokens if token in evidence_tokens)
    return covered / len(slot_tokens)


def _slot_tokens(value: str) -> tuple[str, ...]:
    tokens = tuple(token for token in _tokens(value) if token not in _STOPWORDS)
    return tokens or _tokens(value)


def _predicate_tokens(value: str) -> tuple[str, ...]:
    normalized = _clean_predicate(value)
    if normalized in _PREDICATE_ALIASES:
        return _PREDICATE_ALIASES[normalized]
    return tuple(token for token in _tokens(normalized.replace("_", " ")) if token not in _STOPWORDS)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(text))


def _clean_sentence(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip(_BOUNDARY_CHARS))


def _contains_explicit_negation(value: str) -> bool:
    tokens = set(_tokens(value))
    return "not" in tokens or "never" in tokens


def _contains_blocked_extraction_context(value: str) -> bool:
    return (
        _contains_explicit_negation(value)
        or _contains_non_assertive_context(value)
        or _contains_ambiguous_context(value)
        or _contains_temporal_qualifier_context(value)
        or _contains_metalinguistic_context(value)
    )


def _contains_non_assertive_context(value: str) -> bool:
    tokens = set(_tokens(value))
    return bool(
        tokens
        & {
            "asks",
            "asked",
            "claim",
            "claimed",
            "claims",
            "mentions",
            "phrase",
            "question",
            "questions",
            "quoted",
            "reviewed",
            "whether",
        }
    )


def _contains_ambiguous_context(value: str) -> bool:
    tokens = set(_tokens(value))
    return bool(
        tokens
        & {
            "alternate",
            "alternative",
            "ambiguous",
            "candidate",
            "candidates",
            "either",
            "maybe",
            "or",
            "possibly",
            "possible",
        }
    )


def _contains_temporal_qualifier_context(value: str) -> bool:
    tokens = set(_tokens(value))
    return bool(
        tokens
        & {
            "former",
            "formerly",
            "historical",
            "historically",
            "outdated",
            "past",
            "previous",
            "previously",
            "timeline",
        }
    )


def _contains_metalinguistic_context(value: str) -> bool:
    tokens = set(_tokens(value))
    return bool(
        tokens
        & {
            "compared",
            "comparison",
            "literal",
            "sentence",
            "word",
            "wording",
            "words",
        }
    )


def _clean_slot(value: Any) -> str:
    return _clean_sentence(str(value))


def _clean_predicate(value: Any) -> str:
    text = _clean_sentence(str(value)).casefold()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    return text.strip("_")


def _valid_slot_name(value: str) -> str:
    text = str(value)
    if text not in {"subject", "predicate", "object"}:
        raise ValueError("audit slots must be subject, predicate, or object.")
    return text


def _coerce_probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number in [0, 1].") from exc
    if not math.isfinite(parsed) or not (0.0 <= parsed <= 1.0):
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    return parsed


def _coerce_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean.")


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if not isinstance(value, Sequence):
        raise ValueError("value must be a sequence.")
    return tuple(value)
