"""Protocols and result types for claim-level verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Claim:
    """An atomic factual claim extracted from generated text."""

    text: str
    claim_id: str | None = None
    span: tuple[int, int] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class VerificationStatus(str, Enum):
    """Claim verification outcome."""

    SUPPORTED = "supported"
    REFUTED = "refuted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


@dataclass(frozen=True)
class VerificationResult:
    """Evidence-backed verification result for one claim."""

    status: VerificationStatus
    confidence: float
    evidence: tuple[str, ...] = ()
    explanation: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1].")


@runtime_checkable
class Verifier(Protocol):
    """Interface for retrieval, database, rule, or world-model verifiers."""

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one atomic claim."""
        ...

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> Sequence[VerificationResult]:
        """Verify multiple claims."""
        ...
