"""Dependency-free verifier implementations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus


def normalize_claim_text(text: str) -> str:
    """Normalize claim text for exact-match verifier lookups."""
    return re.sub(r"\s+", " ", text.strip().rstrip(".!?。！？").casefold())


@dataclass(frozen=True)
class InMemoryVerifier:
    """Exact-match verifier backed by an in-memory claim table."""

    facts: Mapping[str, VerificationStatus | str]
    evidence: Mapping[str, Sequence[str]] = field(default_factory=dict)
    default_status: VerificationStatus = VerificationStatus.INSUFFICIENT_EVIDENCE

    def verify(self, claim: Claim, context: Mapping[str, object] | None = None) -> VerificationResult:
        key = normalize_claim_text(claim.text)
        status_value = self.facts.get(key, self.default_status)
        status = status_value if isinstance(status_value, VerificationStatus) else VerificationStatus(str(status_value))
        confidence = 0.9 if status is not self.default_status else 0.3
        ev = tuple(self.evidence.get(key, ()))
        return VerificationResult(
            status=status,
            confidence=confidence,
            evidence=ev,
            explanation=f"in-memory verifier matched status={status.value}",
            metadata={"key": key},
        )

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, object] | None = None,
    ) -> tuple[VerificationResult, ...]:
        return tuple(self.verify(claim, context=context) for claim in claims)
