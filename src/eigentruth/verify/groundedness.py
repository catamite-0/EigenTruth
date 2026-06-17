"""Dependency-free lexical groundedness verifier."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, NamedTuple, Sequence

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
        return cls(
            text=str(text),
            source=None if source is None else str(source),
            metadata=dict(data.get("metadata", {})),
        )


class _DocumentMatch(NamedTuple):
    document: EvidenceDocument
    overlap: float
    exact: bool
    negation_mismatch: bool


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

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_overlap <= 1.0):
            raise ValueError("min_overlap must be in [0, 1].")
        object.__setattr__(self, "evidence", tuple(_coerce_evidence(item) for item in self.evidence))
        object.__setattr__(self, "refutations", _normalize_refutations(self.refutations))

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one claim against lexical evidence snippets."""
        claim_key = normalize_claim_text(claim.text)
        claim_tokens = _tokens(claim.text)
        if not claim_tokens:
            return VerificationResult(
                status=VerificationStatus.NOT_APPLICABLE,
                confidence=0.0,
                explanation="groundedness verifier found no lexical tokens in claim",
                metadata={"verifier": "groundedness_lexical", "claim_key": claim_key},
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
                    "decision_rule": "configured_refutation",
                },
            )

        documents = _documents_with_context(self.evidence, context)
        best = _best_document_match(claim.text, claim_tokens, documents)
        if best is None:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.2,
                explanation="no evidence snippets were provided",
                metadata={
                    "verifier": "groundedness_lexical",
                    "claim_key": claim_key,
                    "decision_rule": "no_evidence",
                },
            )

        evidence = (_evidence_label(best.document),)
        metadata = {
            "verifier": "groundedness_lexical",
            "claim_key": claim_key,
            "best_overlap": best.overlap,
            "best_source": best.document.source,
            "min_overlap": self.min_overlap,
        }
        if best.negation_mismatch and best.overlap >= self.min_overlap:
            return VerificationResult(
                status=VerificationStatus.REFUTED,
                confidence=min(0.85, 0.45 + 0.4 * best.overlap),
                evidence=evidence,
                explanation="best evidence has high token overlap with opposing negation",
                metadata={**metadata, "decision_rule": "negation_mismatch"},
            )
        if best.exact:
            return VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=0.9,
                evidence=evidence,
                explanation="claim text is contained in evidence",
                metadata={**metadata, "decision_rule": "exact_containment"},
            )
        if best.overlap >= self.min_overlap:
            return VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=min(0.85, 0.35 + 0.5 * best.overlap),
                evidence=evidence,
                explanation="claim tokens are covered by evidence above threshold",
                metadata={**metadata, "decision_rule": "token_overlap"},
            )
        return VerificationResult(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            confidence=max(0.2, 0.5 * best.overlap),
            evidence=evidence,
            explanation="best evidence did not cover enough claim tokens",
            metadata={**metadata, "decision_rule": "low_overlap"},
        )

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        """Verify multiple claims."""
        return tuple(self.verify(claim, context=context) for claim in claims)


def _coerce_evidence(value: EvidenceDocument | Mapping[str, Any] | str) -> EvidenceDocument:
    if isinstance(value, EvidenceDocument):
        return value
    if isinstance(value, str):
        return EvidenceDocument(text=value)
    return EvidenceDocument.from_dict(value)


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
    base_documents: Sequence[EvidenceDocument],
    context: Mapping[str, Any] | None,
) -> tuple[EvidenceDocument, ...]:
    documents = tuple(base_documents)
    if context is None or "evidence" not in context:
        return documents
    return documents + tuple(_coerce_evidence(item) for item in _as_sequence(context["evidence"]))


def _best_document_match(
    claim_text: str,
    claim_tokens: tuple[str, ...],
    documents: Sequence[EvidenceDocument],
) -> _DocumentMatch | None:
    if not documents:
        return None
    claim_key = normalize_claim_text(claim_text)
    claim_negated = _has_negation(claim_tokens)
    matches = []
    for document in documents:
        document_tokens = _tokens(document.text)
        document_key = normalize_claim_text(document.text)
        exact = claim_key in document_key
        overlap = _token_overlap(claim_tokens, document_tokens)
        negation_mismatch = claim_negated != _has_negation(document_tokens)
        matches.append(_DocumentMatch(document, overlap, exact, negation_mismatch))
    return max(matches, key=lambda match: (match.exact, match.overlap))


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


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Sequence[str] | str]:
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{name} must be a mapping.")
