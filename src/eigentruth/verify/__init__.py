"""Claim verification interfaces for EigenTruth control pipelines."""

from __future__ import annotations

from eigentruth.verify.claims import ClaimExtractor, SentenceClaimExtractor, claim_features, extract_claims
from eigentruth.verify.composite import CompositeVerifier, RoutedVerifier, VerifierRoute
from eigentruth.verify.groundedness import EvidenceDocument, GroundednessVerifier
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus, Verifier
from eigentruth.verify.rules import InMemoryVerifier, normalize_claim_text

__all__ = [
    "Claim",
    "ClaimExtractor",
    "CompositeVerifier",
    "EvidenceDocument",
    "GroundednessVerifier",
    "InMemoryVerifier",
    "RoutedVerifier",
    "VerificationResult",
    "VerificationStatus",
    "Verifier",
    "VerifierRoute",
    "SentenceClaimExtractor",
    "claim_features",
    "extract_claims",
    "normalize_claim_text",
]
