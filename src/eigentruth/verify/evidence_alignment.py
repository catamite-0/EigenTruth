"""Dependency-free claim-to-evidence alignment checks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from eigentruth.json_utils import to_jsonable
from eigentruth.verify.citations import extract_citation_references
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus


@dataclass(frozen=True)
class EvidenceAlignmentEvidence:
    """One local evidence snippet for claim alignment."""

    text: str
    source: str | None = None
    score: float = 1.0
    citation_id: str | None = None
    claim_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text = str(self.text).strip()
        if not text:
            raise ValueError("evidence alignment text must be non-empty.")
        score = float(self.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("evidence alignment score must be finite and in [0, 1].")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "source", _optional_string(self.source))
        object.__setattr__(self, "citation_id", _optional_string(self.citation_id))
        object.__setattr__(self, "claim_id", _optional_string(self.claim_id))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvidenceAlignmentEvidence":
        """Build an evidence snippet from a JSON-like mapping."""
        text = _text_from_payload(data)
        if text is None:
            raise ValueError("evidence mapping must contain text/content/snippet/title/source_text.")
        metadata = dict(_mapping(data.get("metadata")))
        for key, value in data.items():
            if key not in {"text", "content", "snippet", "title", "source_text", "source", "score", "metadata"}:
                metadata.setdefault(str(key), value)
        raw_citation_id = data.get(
            "citation_id",
            data.get("citation", data.get("ref", metadata.get("citation_id", metadata.get("ref")))),
        )
        raw_claim_id = data.get("claim_id", metadata.get("claim_id"))
        return cls(
            text=text,
            source=_optional_string(data.get("source", metadata.get("source"))),
            score=float(data.get("score", metadata.get("score", 1.0))),
            citation_id=_optional_string(raw_citation_id),
            claim_id=_optional_string(raw_claim_id),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready evidence payload."""
        return {
            "text": self.text,
            "source": self.source,
            "score": self.score,
            "citation_id": self.citation_id,
            "claim_id": self.claim_id,
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class EvidenceAlignmentPolicy:
    """Lexical slot-coverage thresholds for claim/evidence alignment."""

    min_keyword_overlap: float = 0.2
    min_number_recall: float = 1.0
    min_entity_recall: float = 0.5
    require_cited_evidence: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_keyword_overlap",
            _rate_float(self.min_keyword_overlap, name="min_keyword_overlap"),
        )
        object.__setattr__(
            self,
            "min_number_recall",
            _rate_float(self.min_number_recall, name="min_number_recall"),
        )
        object.__setattr__(
            self,
            "min_entity_recall",
            _rate_float(self.min_entity_recall, name="min_entity_recall"),
        )
        object.__setattr__(self, "require_cited_evidence", _coerce_bool(
            self.require_cited_evidence,
            name="require_cited_evidence",
        ))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceAlignmentPolicy":
        """Build a policy from a JSON-like mapping."""
        return cls(
            min_keyword_overlap=float(data.get("min_keyword_overlap", 0.2)),
            min_number_recall=float(data.get("min_number_recall", 1.0)),
            min_entity_recall=float(data.get("min_entity_recall", 0.5)),
            require_cited_evidence=_coerce_bool(
                data.get("require_cited_evidence", False),
                name="require_cited_evidence",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready policy payload."""
        return {
            "min_keyword_overlap": self.min_keyword_overlap,
            "min_number_recall": self.min_number_recall,
            "min_entity_recall": self.min_entity_recall,
            "require_cited_evidence": self.require_cited_evidence,
        }


@dataclass(frozen=True)
class EvidenceAlignmentRecord:
    """One claim/evidence alignment audit row."""

    claim_id: str
    status: str
    evidence_count: int
    cited_evidence_count: int = 0
    keyword_overlap: float | None = None
    number_recall: float | None = None
    entity_recall: float | None = None
    best_source: str | None = None
    best_score: float | None = None
    negation_mismatch: bool = False
    missing_numbers: Sequence[str] = ()
    missing_entities: Sequence[str] = ()
    claim_keywords: Sequence[str] = ()
    evidence_keywords: Sequence[str] = ()
    claim_numbers: Sequence[str] = ()
    evidence_numbers: Sequence[str] = ()
    claim_entities: Sequence[str] = ()
    evidence_entities: Sequence[str] = ()
    citation_references: Sequence[Mapping[str, Any]] = ()
    matched_citation_references: Sequence[Mapping[str, Any]] = ()
    issue_codes: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        claim_id = str(self.claim_id).strip()
        status = str(self.status).strip()
        if not claim_id:
            raise ValueError("evidence alignment claim_id must be non-empty.")
        if not status:
            raise ValueError("evidence alignment status must be non-empty.")
        evidence_count = int(self.evidence_count)
        cited_evidence_count = int(self.cited_evidence_count)
        if evidence_count < 0 or cited_evidence_count < 0:
            raise ValueError("evidence counts must be non-negative.")
        if cited_evidence_count > evidence_count:
            raise ValueError("cited_evidence_count cannot exceed evidence_count.")
        object.__setattr__(self, "claim_id", claim_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence_count", evidence_count)
        object.__setattr__(self, "cited_evidence_count", cited_evidence_count)
        object.__setattr__(self, "best_source", _optional_string(self.best_source))
        object.__setattr__(self, "best_score", _optional_rate(self.best_score, name="best_score"))
        object.__setattr__(self, "negation_mismatch", _coerce_bool(
            self.negation_mismatch,
            name="negation_mismatch",
        ))
        object.__setattr__(self, "missing_numbers", _string_tuple(self.missing_numbers))
        object.__setattr__(self, "missing_entities", _string_tuple(self.missing_entities))
        object.__setattr__(self, "claim_keywords", _string_tuple(self.claim_keywords))
        object.__setattr__(self, "evidence_keywords", _string_tuple(self.evidence_keywords))
        object.__setattr__(self, "claim_numbers", _string_tuple(self.claim_numbers))
        object.__setattr__(self, "evidence_numbers", _string_tuple(self.evidence_numbers))
        object.__setattr__(self, "claim_entities", _string_tuple(self.claim_entities))
        object.__setattr__(self, "evidence_entities", _string_tuple(self.evidence_entities))
        object.__setattr__(
            self,
            "citation_references",
            tuple(to_jsonable(dict(item)) for item in self.citation_references),
        )
        object.__setattr__(
            self,
            "matched_citation_references",
            tuple(to_jsonable(dict(item)) for item in self.matched_citation_references),
        )
        object.__setattr__(self, "issue_codes", _string_tuple(self.issue_codes))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceAlignmentRecord":
        """Build a record from a JSON-like mapping."""
        return cls(
            claim_id=str(data["claim_id"]),
            status=str(data["status"]),
            evidence_count=int(data["evidence_count"]),
            cited_evidence_count=int(data.get("cited_evidence_count", 0)),
            keyword_overlap=_optional_float(data.get("keyword_overlap")),
            number_recall=_optional_float(data.get("number_recall")),
            entity_recall=_optional_float(data.get("entity_recall")),
            best_source=None if data.get("best_source") is None else str(data["best_source"]),
            best_score=_optional_float(data.get("best_score")),
            negation_mismatch=bool(data.get("negation_mismatch", False)),
            missing_numbers=tuple(_sequence(data.get("missing_numbers", ()))),
            missing_entities=tuple(_sequence(data.get("missing_entities", ()))),
            claim_keywords=tuple(_sequence(data.get("claim_keywords", ()))),
            evidence_keywords=tuple(_sequence(data.get("evidence_keywords", ()))),
            claim_numbers=tuple(_sequence(data.get("claim_numbers", ()))),
            evidence_numbers=tuple(_sequence(data.get("evidence_numbers", ()))),
            claim_entities=tuple(_sequence(data.get("claim_entities", ()))),
            evidence_entities=tuple(_sequence(data.get("evidence_entities", ()))),
            citation_references=tuple(
                item for item in _sequence(data.get("citation_references", ())) if isinstance(item, Mapping)
            ),
            matched_citation_references=tuple(
                item for item in _sequence(data.get("matched_citation_references", ())) if isinstance(item, Mapping)
            ),
            issue_codes=tuple(_sequence(data.get("issue_codes", ()))),
            metadata=dict(_mapping(data.get("metadata"))),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready alignment record."""
        return {
            "claim_id": self.claim_id,
            "status": self.status,
            "evidence_count": self.evidence_count,
            "cited_evidence_count": self.cited_evidence_count,
            "keyword_overlap": self.keyword_overlap,
            "number_recall": self.number_recall,
            "entity_recall": self.entity_recall,
            "best_source": self.best_source,
            "best_score": self.best_score,
            "negation_mismatch": self.negation_mismatch,
            "missing_numbers": tuple(self.missing_numbers),
            "missing_entities": tuple(self.missing_entities),
            "claim_keywords": tuple(self.claim_keywords),
            "evidence_keywords": tuple(self.evidence_keywords),
            "claim_numbers": tuple(self.claim_numbers),
            "evidence_numbers": tuple(self.evidence_numbers),
            "claim_entities": tuple(self.claim_entities),
            "evidence_entities": tuple(self.evidence_entities),
            "citation_references": tuple(self.citation_references),
            "matched_citation_references": tuple(self.matched_citation_references),
            "issue_codes": tuple(self.issue_codes),
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class EvidenceAlignmentReport:
    """JSON-ready claim/evidence alignment report."""

    policy: EvidenceAlignmentPolicy | Mapping[str, Any] = field(default_factory=EvidenceAlignmentPolicy)
    records: Sequence[EvidenceAlignmentRecord | Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        policy = self.policy if isinstance(self.policy, EvidenceAlignmentPolicy) else (
            EvidenceAlignmentPolicy.from_dict(self.policy)
        )
        records = tuple(
            record
            if isinstance(record, EvidenceAlignmentRecord)
            else EvidenceAlignmentRecord.from_dict(record)
            for record in self.records
        )
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def passed(self) -> bool:
        """Return whether all evaluated records are aligned."""
        return bool(self.records) and all(record.status == "aligned" for record in self.records)

    def summary(self) -> dict[str, Any]:
        """Return compact alignment metrics."""
        counts_by_status: dict[str, int] = {}
        counts_by_code: dict[str, int] = {}
        keyword_overlaps = []
        number_recalls = []
        entity_recalls = []
        reference_count = 0
        matched_reference_count = 0
        cited_evidence_count = 0
        for record in self.records:
            counts_by_status[record.status] = counts_by_status.get(record.status, 0) + 1
            for code in record.issue_codes:
                counts_by_code[code] = counts_by_code.get(code, 0) + 1
            if record.keyword_overlap is not None:
                keyword_overlaps.append(record.keyword_overlap)
            if record.number_recall is not None:
                number_recalls.append(record.number_recall)
            if record.entity_recall is not None:
                entity_recalls.append(record.entity_recall)
            reference_count += len(record.citation_references)
            matched_reference_count += len(record.matched_citation_references)
            cited_evidence_count += record.cited_evidence_count
        record_count = len(self.records)
        aligned_count = counts_by_status.get("aligned", 0)
        misaligned_count = counts_by_status.get("misaligned", 0)
        insufficient_count = counts_by_status.get("insufficient_evidence", 0)
        return {
            "available": record_count > 0,
            "passed": self.passed,
            "policy": self.policy.to_dict(),
            "record_count": record_count,
            "aligned_count": aligned_count,
            "misaligned_count": misaligned_count,
            "insufficient_evidence_count": insufficient_count,
            "alignment_rate": _safe_div(aligned_count, record_count),
            "misalignment_rate": _safe_div(misaligned_count, record_count) or 0.0,
            "insufficient_evidence_rate": _safe_div(insufficient_count, record_count) or 0.0,
            "keyword_overlap_mean": _mean(keyword_overlaps),
            "keyword_overlap_min": _minimum(keyword_overlaps),
            "number_recall_mean": _mean(number_recalls),
            "entity_recall_mean": _mean(entity_recalls),
            "citation_reference_count": reference_count,
            "matched_citation_reference_count": matched_reference_count,
            "cited_evidence_count": cited_evidence_count,
            "citation_reference_coverage_rate": _safe_div(matched_reference_count, reference_count),
            "issue_count": sum(counts_by_code.values()),
            "counts_by_status": counts_by_status,
            "counts_by_code": counts_by_code,
            "top_records": tuple(record.to_dict() for record in self.records[:8]),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceAlignmentReport":
        """Build a report from a JSON-like mapping."""
        return cls(
            policy=_mapping(data.get("policy")),
            records=tuple(_sequence(data.get("records", ()))),
            metadata=dict(_mapping(data.get("metadata"))),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report."""
        return {
            "policy": self.policy.to_dict(),
            "records": tuple(record.to_dict() for record in self.records),
            "metadata": to_jsonable(dict(self.metadata)),
            "summary": self.summary(),
        }


@dataclass(frozen=True)
class EvidenceAlignmentVerifier:
    """Verify that local evidence snippets actually align with a claim."""

    evidence: Sequence[EvidenceAlignmentEvidence | Mapping[str, Any] | str] = ()
    policy: EvidenceAlignmentPolicy | Mapping[str, Any] = field(default_factory=EvidenceAlignmentPolicy)

    def __post_init__(self) -> None:
        evidence = tuple(_coerce_evidence(item) for item in self.evidence)
        policy = self.policy if isinstance(self.policy, EvidenceAlignmentPolicy) else (
            EvidenceAlignmentPolicy.from_dict(self.policy)
        )
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "policy", policy)

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one claim against local evidence snippets."""
        report = audit_evidence_alignment(
            claim,
            evidence=self.evidence,
            policy=self.policy,
            context=context,
        )
        summary = report.summary()
        record = report.records[0] if report.records else None
        metadata = {
            "verifier": "evidence_alignment",
            "decision_rule": "slot_evidence_alignment",
            "claim_id": claim.claim_id,
            "evidence_alignment": report.to_dict(),
        }
        if record is None:
            return VerificationResult(
                status=VerificationStatus.NOT_APPLICABLE,
                confidence=0.0,
                explanation="claim has no lexical tokens to align",
                metadata=metadata,
            )
        if record.status == "aligned":
            return VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=min(0.95, 0.55 + 0.4 * _alignment_strength(record)),
                evidence=_record_evidence(record),
                explanation="claim slots align with local evidence",
                metadata=metadata,
            )
        if _strong_misalignment(record):
            return VerificationResult(
                status=VerificationStatus.REFUTED,
                confidence=0.85,
                evidence=_record_evidence(record),
                explanation="local evidence is missing or contradicting required claim slots",
                metadata=metadata,
            )
        return VerificationResult(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            confidence=max(0.2, 1.0 - float(summary.get("insufficient_evidence_rate", 1.0))),
            evidence=_record_evidence(record),
            explanation="local evidence does not cover enough claim slots",
            metadata=metadata,
        )

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        """Verify multiple claims."""
        return tuple(self.verify(claim, context=context) for claim in claims)


def audit_evidence_alignment(
    claim: Claim | str,
    *,
    evidence: Sequence[EvidenceAlignmentEvidence | Mapping[str, Any] | str] = (),
    policy: EvidenceAlignmentPolicy | Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> EvidenceAlignmentReport:
    """Audit slot-level alignment between one claim and local evidence."""
    claim_obj = claim if isinstance(claim, Claim) else Claim(text=str(claim))
    policy_obj = (
        EvidenceAlignmentPolicy()
        if policy is None
        else policy
        if isinstance(policy, EvidenceAlignmentPolicy)
        else EvidenceAlignmentPolicy.from_dict(policy)
    )
    evidence_items = _evidence_with_context(evidence, context)
    claim_features = _TextFeatures.from_text(claim_obj.text)
    if not claim_features.keywords and not claim_features.numbers and not claim_features.entities:
        return EvidenceAlignmentReport(policy=policy_obj, records=(), metadata={"claim_id": claim_obj.claim_id})
    record = _alignment_record_for_claim(
        claim_obj,
        evidence_items=evidence_items,
        policy=policy_obj,
    )
    return EvidenceAlignmentReport(policy=policy_obj, records=(record,), metadata={"claim_id": record.claim_id})


@dataclass(frozen=True)
class _TextFeatures:
    keywords: tuple[str, ...] = ()
    numbers: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    negated: bool = False

    @classmethod
    def from_text(cls, text: str) -> "_TextFeatures":
        tokens = tuple(match.group(0).lower() for match in _TOKEN_RE.finditer(text))
        return cls(
            keywords=_keyword_tokens(text),
            numbers=_number_tokens(text),
            entities=_entity_tokens(text),
            negated=any(token in _NEGATION_TOKENS for token in tokens),
        )


def _alignment_record_for_claim(
    claim: Claim,
    *,
    evidence_items: Sequence[EvidenceAlignmentEvidence],
    policy: EvidenceAlignmentPolicy,
) -> EvidenceAlignmentRecord:
    claim_id = claim.claim_id or "claim"
    claim_features = _TextFeatures.from_text(claim.text)
    references = extract_citation_references(claim)
    matched_references = _matched_references(references, evidence_items)
    cited_evidence = tuple(item for item in evidence_items if _evidence_matches_any_reference(item, references))
    selected_evidence = cited_evidence if cited_evidence else tuple(evidence_items)

    if not selected_evidence:
        issue_codes = ["no_local_evidence_text"]
        if policy.require_cited_evidence and references:
            issue_codes.append("missing_cited_evidence")
        return EvidenceAlignmentRecord(
            claim_id=claim_id,
            status="insufficient_evidence",
            evidence_count=0,
            cited_evidence_count=0,
            claim_keywords=claim_features.keywords,
            claim_numbers=claim_features.numbers,
            claim_entities=claim_features.entities,
            citation_references=references,
            matched_citation_references=matched_references,
            issue_codes=tuple(issue_codes),
        )

    best_evidence = _best_evidence(claim_features, selected_evidence)
    evidence_features = _TextFeatures.from_text("\n".join(item.text for item in selected_evidence))
    best_features = _TextFeatures.from_text(best_evidence.text)
    keyword_overlap = _recall(claim_features.keywords, evidence_features.keywords)
    number_recall = _recall(claim_features.numbers, evidence_features.numbers)
    entity_recall = _recall(claim_features.entities, evidence_features.entities)
    missing_numbers = _missing_items(claim_features.numbers, evidence_features.numbers)
    missing_entities = _missing_items(claim_features.entities, evidence_features.entities)
    best_overlap = _recall(claim_features.keywords, best_features.keywords) or 0.0
    negation_mismatch = (
        bool(claim_features.keywords)
        and best_overlap >= policy.min_keyword_overlap
        and claim_features.negated != best_features.negated
    )
    issue_codes: list[str] = []
    if keyword_overlap is not None and keyword_overlap < policy.min_keyword_overlap:
        issue_codes.append("low_keyword_overlap")
    if number_recall is not None and number_recall < policy.min_number_recall:
        issue_codes.append("missing_claim_number")
    if entity_recall is not None and entity_recall < policy.min_entity_recall:
        issue_codes.append("missing_claim_entity")
    if negation_mismatch:
        issue_codes.append("negation_mismatch")
    if policy.require_cited_evidence and references and not cited_evidence:
        issue_codes.append("missing_cited_evidence")
    status = "aligned" if not issue_codes else "misaligned"
    if issue_codes == ["low_keyword_overlap"]:
        status = "insufficient_evidence"
    return EvidenceAlignmentRecord(
        claim_id=claim_id,
        status=status,
        evidence_count=len(selected_evidence),
        cited_evidence_count=len(cited_evidence),
        keyword_overlap=keyword_overlap,
        number_recall=number_recall,
        entity_recall=entity_recall,
        best_source=best_evidence.source,
        best_score=best_evidence.score,
        negation_mismatch=negation_mismatch,
        missing_numbers=missing_numbers,
        missing_entities=missing_entities,
        claim_keywords=claim_features.keywords,
        evidence_keywords=evidence_features.keywords,
        claim_numbers=claim_features.numbers,
        evidence_numbers=evidence_features.numbers,
        claim_entities=claim_features.entities,
        evidence_entities=evidence_features.entities,
        citation_references=references,
        matched_citation_references=matched_references,
        issue_codes=tuple(issue_codes),
        metadata={"selected_evidence": tuple(_evidence_summary(item) for item in selected_evidence[:5])},
    )


def _evidence_with_context(
    evidence: Sequence[EvidenceAlignmentEvidence | Mapping[str, Any] | str],
    context: Mapping[str, Any] | None,
) -> tuple[EvidenceAlignmentEvidence, ...]:
    items = [_coerce_evidence(item) for item in evidence]
    if isinstance(context, Mapping):
        for key in (
            "evidence",
            "evidence_texts",
            "retrieval_hits",
            "hits",
            "citation_evidence",
            "citation_hits",
            "search_hits",
        ):
            value = context.get(key)
            if value is None:
                continue
            items.extend(_coerce_evidence(item) for item in _sequence(value))
        output = _mapping(context.get("output"))
        for hit in _sequence(output.get("hits", ())):
            items.append(_coerce_evidence(hit))
        for query_result in _sequence(output.get("hits_by_query", ())):
            for hit in _sequence(_mapping(query_result).get("hits", ())):
                items.append(_coerce_evidence(hit))
    return tuple(items)


def _coerce_evidence(value: EvidenceAlignmentEvidence | Mapping[str, Any] | str) -> EvidenceAlignmentEvidence:
    if isinstance(value, EvidenceAlignmentEvidence):
        return value
    if isinstance(value, str):
        return EvidenceAlignmentEvidence(text=value)
    if isinstance(value, Mapping):
        return EvidenceAlignmentEvidence.from_mapping(value)
    raise ValueError("evidence must be a string, mapping, or EvidenceAlignmentEvidence.")


def _best_evidence(
    claim_features: _TextFeatures,
    evidence_items: Sequence[EvidenceAlignmentEvidence],
) -> EvidenceAlignmentEvidence:
    return max(
        evidence_items,
        key=lambda item: (
            _recall(claim_features.numbers, _TextFeatures.from_text(item.text).numbers) or 0.0,
            _recall(claim_features.entities, _TextFeatures.from_text(item.text).entities) or 0.0,
            _recall(claim_features.keywords, _TextFeatures.from_text(item.text).keywords) or 0.0,
            item.score,
        ),
    )


def _matched_references(
    references: Sequence[Mapping[str, Any]],
    evidence_items: Sequence[EvidenceAlignmentEvidence],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(reference for reference in references if _reference_matches_any_evidence(reference, evidence_items))


def _reference_matches_any_evidence(
    reference: Mapping[str, Any],
    evidence_items: Sequence[EvidenceAlignmentEvidence],
) -> bool:
    return any(_evidence_matches_reference(item, reference) for item in evidence_items)


def _evidence_matches_any_reference(
    evidence: EvidenceAlignmentEvidence,
    references: Sequence[Mapping[str, Any]],
) -> bool:
    return bool(references) and any(_evidence_matches_reference(evidence, reference) for reference in references)


def _evidence_matches_reference(evidence: EvidenceAlignmentEvidence, reference: Mapping[str, Any]) -> bool:
    metadata = _mapping(evidence.metadata)
    reference_keys = _reference_keys(reference)
    if not reference_keys:
        return False
    evidence_values = {
        _normalize_reference_value(evidence.citation_id),
        _normalize_reference_value(evidence.source),
        _normalize_reference_value(metadata.get("citation_id")),
        _normalize_reference_value(metadata.get("ref")),
        _normalize_reference_value(metadata.get("doi")),
        _normalize_reference_value(metadata.get("arxiv_id", metadata.get("arxiv"))),
        _normalize_reference_value(metadata.get("url")),
    }
    evidence_values.discard(None)
    return bool(reference_keys & evidence_values)


def _reference_keys(reference: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for key in ("citation_id", "id", "ref", "label", "doi", "arxiv_id", "arxiv", "url"):
        value = _normalize_reference_value(reference.get(key))
        if value is not None:
            keys.add(value)
    return keys


def _normalize_reference_value(value: Any) -> str | None:
    text = _optional_string(value)
    if text is None:
        return None
    lowered = text.lower()
    for prefix in ("doi:", "arxiv:"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix):].strip()
    return lowered.rstrip(".,;)")


def _record_evidence(record: EvidenceAlignmentRecord) -> tuple[str, ...]:
    selected = _sequence(record.metadata.get("selected_evidence", ()))
    labels = []
    for item in selected:
        mapping = _mapping(item)
        text = _optional_string(mapping.get("text"))
        if text is None:
            continue
        source = _optional_string(mapping.get("source"))
        labels.append(text if source is None else f"{source}: {text}")
    return tuple(labels)


def _evidence_summary(evidence: EvidenceAlignmentEvidence) -> dict[str, Any]:
    return {
        "text": _truncate(evidence.text, 240),
        "source": evidence.source,
        "score": evidence.score,
        "citation_id": evidence.citation_id,
        "claim_id": evidence.claim_id,
    }


def _alignment_strength(record: EvidenceAlignmentRecord) -> float:
    values = [
        value
        for value in (record.keyword_overlap, record.number_recall, record.entity_recall)
        if value is not None
    ]
    if not values:
        return 0.0
    return max(0.0, min(1.0, sum(values) / len(values)))


def _strong_misalignment(record: EvidenceAlignmentRecord) -> bool:
    codes = set(record.issue_codes)
    return bool(codes & {"missing_claim_number", "negation_mismatch", "missing_cited_evidence"})


def _keyword_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        token = match.group(0).lower().strip("_-'")
        if len(token) < 3 or token in _STOPWORDS or _NUMBER_RE.fullmatch(token):
            continue
        tokens.append(token)
    return tuple(dict.fromkeys(tokens))


def _number_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0).replace(",", "").strip()
        if token:
            tokens.append(token)
    return tuple(dict.fromkeys(tokens))


def _entity_tokens(text: str) -> tuple[str, ...]:
    entities: list[str] = []
    for match in _WORD_RE.finditer(text):
        raw = match.group(0).strip("_-'")
        if len(raw) < 2:
            continue
        lowered = raw.lower()
        if lowered in _STOPWORDS or _NUMBER_RE.fullmatch(raw):
            continue
        if raw[0].isupper() or raw.isupper():
            entities.append(lowered)
    return tuple(dict.fromkeys(entities))


def _recall(reference: Sequence[str], observed: Sequence[str]) -> float | None:
    reference_set = set(reference)
    if not reference_set:
        return None
    observed_set = set(observed)
    return len(reference_set & observed_set) / len(reference_set)


def _missing_items(reference: Sequence[str], observed: Sequence[str]) -> tuple[str, ...]:
    observed_set = set(observed)
    return tuple(item for item in reference if item not in observed_set)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _minimum(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return min(values)


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    denominator = float(denominator)
    if denominator == 0.0:
        return None
    return float(numerator) / denominator


def _text_from_payload(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Mapping):
        for key in ("text", "content", "snippet", "title", "source_text", "claim_text"):
            text = _optional_string(value.get(key))
            if text is not None:
                return text
    return _optional_string(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("float values must be finite.")
    return result


def _optional_rate(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _rate_float(value, name=name)


def _rate_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return result


def _coerce_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    raise ValueError(f"{name} must be a boolean or boolean string.")


def _string_tuple(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'’-]*")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:,\d{3})*(?:\.\d+)?%?")
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
    "isnt",
    "arent",
    "wasnt",
    "werent",
}
_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "can",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "his",
    "into",
    "its",
    "more",
    "not",
    "off",
    "onto",
    "our",
    "out",
    "per",
    "she",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "was",
    "were",
    "which",
    "will",
    "with",
    "would",
    "you",
}
