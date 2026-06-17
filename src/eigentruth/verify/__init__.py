"""Claim verification interfaces for EigenTruth control pipelines."""

from __future__ import annotations

from eigentruth.verify.claims import extract_claims
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus, Verifier
from eigentruth.verify.rules import InMemoryVerifier, normalize_claim_text

__all__ = [
    "Claim",
    "InMemoryVerifier",
    "VerificationResult",
    "VerificationStatus",
    "Verifier",
    "extract_claims",
    "normalize_claim_text",
]
