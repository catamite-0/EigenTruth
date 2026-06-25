"""Claim verification interfaces for EigenTruth control pipelines."""

from __future__ import annotations

from eigentruth.verify.cache import (
    CachedVerifier,
    JsonTraceCache,
    TraceCacheRecord,
    VerifierCacheStats,
    stable_cache_key,
    verifier_cache_key,
)
from eigentruth.verify.claims import (
    ClaimExtractor,
    SentenceClaimExtractor,
    claim_features,
    extract_calculation,
    extract_claims,
)
from eigentruth.verify.coherence import (
    ClaimCoherenceIssue,
    ClaimCoherenceReport,
    ClaimDependency,
    apply_claim_coherence,
    infer_claim_dependencies,
)
from eigentruth.verify.composite import CompositeVerifier, RoutedVerifier, VerifierRoute
from eigentruth.verify.groundedness import EvidenceDocument, EvidenceQualityPolicy, GroundednessVerifier
from eigentruth.verify.planning import (
    ClaimVerificationPlan,
    ClaimVerificationPlanner,
    VerificationRouteHint,
)
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus, Verifier
from eigentruth.verify.rules import InMemoryVerifier, normalize_claim_text
from eigentruth.verify.selfcheck import SelfConsistencyVerifier

__all__ = [
    "CachedVerifier",
    "Claim",
    "ClaimCoherenceIssue",
    "ClaimCoherenceReport",
    "ClaimDependency",
    "ClaimExtractor",
    "ClaimVerificationPlan",
    "ClaimVerificationPlanner",
    "CompositeVerifier",
    "EvidenceDocument",
    "EvidenceQualityPolicy",
    "GroundednessVerifier",
    "InMemoryVerifier",
    "JsonTraceCache",
    "RoutedVerifier",
    "TraceCacheRecord",
    "VerificationResult",
    "VerificationStatus",
    "VerifierCacheStats",
    "Verifier",
    "VerifierRoute",
    "VerificationRouteHint",
    "SentenceClaimExtractor",
    "SelfConsistencyVerifier",
    "apply_claim_coherence",
    "claim_features",
    "extract_calculation",
    "extract_claims",
    "infer_claim_dependencies",
    "normalize_claim_text",
    "stable_cache_key",
    "verifier_cache_key",
]
