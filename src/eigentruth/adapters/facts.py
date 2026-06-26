"""Dependency-free structured fact verifier adapter."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from eigentruth.verify import Claim, VerificationResult, VerificationStatus, normalize_claim_text
from eigentruth.verify.triples import (
    ClaimTriple,
    ClaimTripleExtractor,
    RuleBasedTripleExtractor,
    extract_claim_triples,
)

_BOUNDARY_CHARS = " \t\r\n.,;:!?()[]{}\"'`“”‘’。！？"
_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_OBJECT_LIST_SEPARATOR_RE = re.compile(r"\s*(?:[,;]|\band\b|\bas well as\b)\s*", re.IGNORECASE)
_PREDICATE_ALIASES = {
    "p36": "capital",
    "capital": "capital",
    "capital_of": "capital",
    "p37": "official_language",
    "official_language": "official_language",
    "official_languages": "official_language",
    "official_language_of": "official_language",
    "p38": "currency",
    "currency": "currency",
    "currency_of": "currency",
}


@dataclass(frozen=True)
class StructuredFact:
    """One subject-predicate-object fact from a structured source."""

    subject: str
    predicate: str
    object: str
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        subject = _clean_slot(self.subject)
        predicate = _clean_slot(self.predicate)
        object_value = _clean_slot(self.object)
        if not subject:
            raise ValueError("fact subject must be non-empty.")
        if not predicate:
            raise ValueError("fact predicate must be non-empty.")
        if not object_value:
            raise ValueError("fact object must be non-empty.")
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "object", object_value)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StructuredFact":
        """Build a fact from a JSON-like mapping or QA-corpus document."""
        metadata = dict(data.get("metadata", {}))
        for key in ("subject_aliases", "entity_aliases", "country_aliases", "object_aliases", "answer_aliases"):
            if data.get(key) is not None and key not in metadata:
                metadata[key] = data[key]
        subject = _first_present(
            data,
            metadata,
            ("subject", "entity", "item", "country"),
        )
        predicate = _first_present(
            data,
            metadata,
            ("predicate", "property", "statement_property", "statement_property_label"),
        )
        object_value = _first_present(
            data,
            metadata,
            ("object", "object_text", "value", "answer"),
        )
        if subject is None or predicate is None or object_value is None:
            raise ValueError("fact mapping must contain subject, predicate, and object fields.")
        source = data.get("source", metadata.get("source"))
        return cls(
            subject=str(subject),
            predicate=str(predicate),
            object=str(object_value),
            source=None if source is None else str(source),
            metadata=metadata,
        )

    def to_evidence(self) -> str:
        """Return a compact evidence string."""
        text = f"{self.subject} {self.predicate} {self.object}"
        if self.source:
            return f"{self.source}: {text}"
        return text

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StructuredFactVerifier:
    """Verifier over known subject-predicate-object facts.

    Matching subject/predicate/object triples are supported. If the source has a
    known object for the same subject and predicate but the claim supplies a
    different object, the claim is refuted. Unknown subject/predicate pairs fail
    closed as insufficient evidence.
    """

    facts: Sequence[StructuredFact | Mapping[str, Any]]
    extractor: ClaimTripleExtractor = field(default_factory=RuleBasedTripleExtractor)

    def __post_init__(self) -> None:
        facts = tuple(_coerce_fact(item) for item in self.facts)
        if not facts:
            raise ValueError("StructuredFactVerifier requires at least one fact.")
        index: dict[tuple[str, str], tuple[StructuredFact, ...]] = {}
        for fact in facts:
            predicate_key = _normalize_predicate(fact.predicate)
            for subject_key in _fact_subject_keys(fact):
                key = (subject_key, predicate_key)
                if fact not in index.get(key, ()):
                    index[key] = (*index.get(key, ()), fact)
        object.__setattr__(self, "facts", facts)
        object.__setattr__(self, "_index", index)

    @classmethod
    def from_corpus(cls, corpus: Mapping[str, Any]) -> "StructuredFactVerifier":
        """Build from a corpus JSON object with documents/records/facts."""
        raw_documents = corpus.get("facts", corpus.get("documents", corpus.get("records", ())))
        if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, (str, bytes, bytearray)):
            raise ValueError("fact corpus must contain a facts, documents, or records list.")
        facts = []
        for item in raw_documents:
            if not isinstance(item, Mapping):
                continue
            try:
                facts.append(StructuredFact.from_mapping(item))
            except ValueError:
                continue
        if not facts:
            raise ValueError("fact corpus does not contain any structured facts.")
        return cls(facts)

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one claim against structured facts."""
        triples = _claim_triples(claim, context, extractor=self.extractor)
        if not triples:
            return VerificationResult(
                status=VerificationStatus.NOT_APPLICABLE,
                confidence=0.0,
                explanation="structured fact verifier extracted no supported triples",
                metadata={"verifier": "structured_fact", "decision_rule": "no_triples"},
            )

        results = tuple(self._verify_triple(triple) for triple in triples)
        refuted = tuple(result for result in results if result.status is VerificationStatus.REFUTED)
        if refuted:
            result = refuted[0]
            return VerificationResult(
                status=VerificationStatus.REFUTED,
                confidence=result.confidence,
                evidence=result.evidence,
                explanation=(
                    "structured source has known object(s) for this subject and predicate, "
                    "but the claim differs"
                ),
                metadata={
                    **dict(result.metadata),
                    "all_triple_results": tuple(_result_summary(item) for item in results),
                },
            )

        if all(result.status is VerificationStatus.SUPPORTED for result in results):
            evidence = _unique_evidence(item for result in results for item in result.evidence)
            return VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=min(result.confidence for result in results),
                evidence=evidence,
                explanation="all extracted claim triples match structured source facts",
                metadata={
                    "verifier": "structured_fact",
                    "decision_rule": "all_triples_match",
                    "triple_count": len(results),
                    "all_triple_results": tuple(_result_summary(item) for item in results),
                },
            )

        evidence = _unique_evidence(item for result in results for item in result.evidence)
        return VerificationResult(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            confidence=0.3,
            evidence=evidence,
            explanation="one or more extracted triples were not covered by the structured source",
            metadata={
                "verifier": "structured_fact",
                "decision_rule": "missing_subject_predicate",
                "triple_count": len(results),
                "all_triple_results": tuple(_result_summary(item) for item in results),
            },
        )

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        """Verify multiple claims."""
        return tuple(self.verify(claim, context=context) for claim in claims)

    def _verify_triple(self, triple: ClaimTriple) -> VerificationResult:
        subject_key = _normalize_entity(triple.subject)
        predicate_key = _normalize_predicate(triple.predicate)
        candidates = self._index.get((subject_key, predicate_key), ())
        triple_payload = triple.to_dict()
        if not candidates:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.25,
                explanation="no structured facts found for subject and predicate",
                metadata={
                    "verifier": "structured_fact",
                    "decision_rule": "subject_predicate_not_found",
                    "subject_key": subject_key,
                    "predicate_key": predicate_key,
                    "triple": triple_payload,
                },
            )

        object_key = _normalize_entity(triple.object)
        for fact in candidates:
            if object_key in _fact_object_keys(fact):
                return VerificationResult(
                    status=VerificationStatus.SUPPORTED,
                    confidence=0.95,
                    evidence=(fact.to_evidence(),),
                    explanation="claim triple matches structured source fact",
                    metadata={
                        "verifier": "structured_fact",
                        "decision_rule": "object_match",
                        "subject_key": subject_key,
                        "predicate_key": predicate_key,
                        "n_known_objects": len(candidates),
                        "triple": triple_payload,
                    },
                )

        object_values = _split_object_values(triple.object)
        if len(object_values) > 1:
            matched_evidence = []
            matched_objects = []
            unmatched_objects = []
            for object_value in object_values:
                value_key = _normalize_entity(object_value)
                match = next(
                    (
                        fact
                        for fact in candidates
                        if value_key in _fact_object_keys(fact)
                    ),
                    None,
                )
                if match is None:
                    unmatched_objects.append(object_value)
                else:
                    matched_objects.append(object_value)
                    matched_evidence.append(match.to_evidence())
            if not unmatched_objects:
                return VerificationResult(
                    status=VerificationStatus.SUPPORTED,
                    confidence=0.9,
                    evidence=_unique_evidence(matched_evidence),
                    explanation="all listed claim objects match structured source facts",
                    metadata={
                        "verifier": "structured_fact",
                        "decision_rule": "all_list_objects_match",
                        "subject_key": subject_key,
                        "predicate_key": predicate_key,
                        "n_known_objects": len(candidates),
                        "matched_objects": tuple(matched_objects),
                        "triple": triple_payload,
                    },
                )
            evidence = tuple(fact.to_evidence() for fact in candidates)
            return VerificationResult(
                status=VerificationStatus.REFUTED,
                confidence=0.9,
                evidence=evidence,
                explanation="one or more listed claim objects do not match structured source fact(s)",
                metadata={
                    "verifier": "structured_fact",
                    "decision_rule": "object_list_mismatch",
                    "subject_key": subject_key,
                    "predicate_key": predicate_key,
                    "n_known_objects": len(candidates),
                    "matched_objects": tuple(matched_objects),
                    "unmatched_objects": tuple(unmatched_objects),
                    "triple": triple_payload,
                },
            )

        evidence = tuple(fact.to_evidence() for fact in candidates)
        return VerificationResult(
            status=VerificationStatus.REFUTED,
            confidence=0.9,
            evidence=evidence,
            explanation="subject and predicate are known, but object does not match structured source fact(s)",
            metadata={
                "verifier": "structured_fact",
                "decision_rule": "object_mismatch",
                "subject_key": subject_key,
                "predicate_key": predicate_key,
                "n_known_objects": len(candidates),
                "triple": triple_payload,
            },
        )


def _coerce_fact(value: StructuredFact | Mapping[str, Any]) -> StructuredFact:
    if isinstance(value, StructuredFact):
        return value
    return StructuredFact.from_mapping(value)


def _claim_triples(
    claim: Claim,
    context: Mapping[str, Any] | None,
    *,
    extractor: ClaimTripleExtractor,
) -> tuple[ClaimTriple, ...]:
    if isinstance(claim.metadata, Mapping) and (
        claim.metadata.get("triples") is not None
        or claim.metadata.get("claim_triples") is not None
    ):
        return extract_claim_triples(claim, extractor=extractor)
    if context is not None and isinstance(context.get("triples"), Sequence):
        payload = dict(claim.metadata) if isinstance(claim.metadata, Mapping) else {}
        payload["claim_triples"] = tuple(context["triples"])
        return extract_claim_triples(
            Claim(claim.text, claim_id=claim.claim_id, span=claim.span, metadata=payload),
            extractor=extractor,
        )
    return extract_claim_triples(claim, extractor=extractor)


def _first_present(
    data: Mapping[str, Any],
    metadata: Mapping[str, Any],
    keys: Sequence[str],
) -> Any:
    for key in keys:
        if data.get(key) is not None:
            return data[key]
        if metadata.get(key) is not None:
            return metadata[key]
    return None


def _normalize_entity(value: Any) -> str:
    text = normalize_claim_text(_clean_slot(value))
    text = _LEADING_ARTICLE_RE.sub("", text).strip()
    return text


def _normalize_predicate(value: Any) -> str:
    text = normalize_claim_text(_clean_slot(value))
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text).strip("_")
    return _PREDICATE_ALIASES.get(text, text)


def _fact_subject_keys(fact: StructuredFact) -> tuple[str, ...]:
    return _unique_keys(
        (fact.subject,)
        + _metadata_text_sequence(fact.metadata, "subject_aliases")
        + _metadata_text_sequence(fact.metadata, "entity_aliases")
        + _metadata_text_sequence(fact.metadata, "country_aliases")
    )


def _fact_object_keys(fact: StructuredFact) -> tuple[str, ...]:
    return _unique_keys(
        (fact.object,)
        + _metadata_text_sequence(fact.metadata, "object_aliases")
        + _metadata_text_sequence(fact.metadata, "answer_aliases")
    )


def _metadata_text_sequence(metadata: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value if str(item).strip())
    return (str(value),)


def _unique_keys(values: Sequence[str]) -> tuple[str, ...]:
    keys = []
    seen = set()
    for value in values:
        key = _normalize_entity(value)
        if key and key not in seen:
            keys.append(key)
            seen.add(key)
    return tuple(keys)


def _split_object_values(value: Any) -> tuple[str, ...]:
    text = _clean_slot(value)
    if not _OBJECT_LIST_SEPARATOR_RE.search(text):
        return (text,)
    values = tuple(part for part in (_clean_slot(item) for item in _OBJECT_LIST_SEPARATOR_RE.split(text)) if part)
    return values if len(values) > 1 else (text,)


def _clean_slot(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip(_BOUNDARY_CHARS))


def _result_summary(result: VerificationResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "confidence": result.confidence,
        "evidence": tuple(result.evidence),
        "metadata": dict(result.metadata),
    }


def _unique_evidence(items: Sequence[str] | Any) -> tuple[str, ...]:
    labels: list[str] = []
    seen = set()
    for item in items:
        text = str(item)
        if text not in seen:
            labels.append(text)
            seen.add(text)
    return tuple(labels)
