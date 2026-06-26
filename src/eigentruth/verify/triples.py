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
class TripleEvidenceAudit:
    """Slot-level evidence coverage for one extracted triple."""

    triple: ClaimTriple
    passed: bool
    evidence: tuple[str, ...] = ()
    covered_slots: tuple[str, ...] = ()
    missing_slots: tuple[str, ...] = ()
    slot_coverage: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        triple = self.triple if isinstance(self.triple, ClaimTriple) else ClaimTriple.from_dict(self.triple)
        covered_slots = tuple(_valid_slot_name(slot) for slot in self.covered_slots)
        missing_slots = tuple(_valid_slot_name(slot) for slot in self.missing_slots)
        slot_coverage = {
            str(key): _coerce_probability(value, name=f"{key} coverage")
            for key, value in self.slot_coverage.items()
        }
        object.__setattr__(self, "triple", triple)
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "evidence", tuple(str(item) for item in self.evidence))
        object.__setattr__(self, "covered_slots", covered_slots)
        object.__setattr__(self, "missing_slots", missing_slots)
        object.__setattr__(self, "slot_coverage", slot_coverage)
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
    def passed(self) -> bool:
        """Return true when at least one triple exists and all triples passed."""
        return self.triple_count > 0 and self.failed_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "claim_id": self.claim_id,
            "triple_count": self.triple_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "passed": self.passed,
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
            metadata={"decision_rule": "no_evidence"},
        )
    scored = [_score_document(triple, document, min_slot_coverage=min_slot_coverage) for document in documents]
    best = max(scored, key=lambda item: (len(item.covered_slots), sum(item.slot_coverage.values())))
    aggregate = _aggregate_scored_documents(scored, min_slot_coverage=min_slot_coverage)
    if _aggregate_is_better(aggregate, best):
        return aggregate
    return _audit_from_scored_document(triple, best)


def _audit_from_scored_document(
    triple: ClaimTriple,
    scored: "_ScoredDocument",
) -> TripleEvidenceAudit:
    return TripleEvidenceAudit(
        triple=triple,
        passed=not scored.missing_slots,
        evidence=(_evidence_label(scored.document),),
        covered_slots=scored.covered_slots,
        missing_slots=scored.missing_slots,
        slot_coverage=scored.slot_coverage,
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
