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
    VerificationPlanCostEstimate,
    VerificationRouteHint,
    estimate_verification_plan_cost,
)
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus, Verifier
from eigentruth.verify.rules import InMemoryVerifier, normalize_claim_text
from eigentruth.verify.selfcheck import SelfConsistencyVerifier
from eigentruth.verify.triples import (
    ClaimTriple,
    ClaimTripleExtractor,
    RuleBasedTripleExtractor,
    TripleEvidenceAudit,
    TripleEvidenceAuditReport,
    TripleEvidenceVerifier,
    audit_claim_triples,
    extract_claim_triples,
)

__all__ = [
    "CachedVerifier",
    "Claim",
    "ClaimCoherenceIssue",
    "ClaimCoherenceReport",
    "ClaimDependency",
    "ClaimExtractor",
    "ClaimTriple",
    "ClaimTripleExtractor",
    "ClaimVerificationPlan",
    "ClaimVerificationPlanner",
    "CompositeVerifier",
    "EvidenceDocument",
    "EvidenceQualityPolicy",
    "GroundednessVerifier",
    "InMemoryVerifier",
    "JsonTraceCache",
    "RuleBasedTripleExtractor",
    "RoutedVerifier",
    "TraceCacheRecord",
    "TripleEvidenceAudit",
    "TripleEvidenceAuditReport",
    "TripleEvidenceVerifier",
    "VerificationResult",
    "VerificationStatus",
    "VerifierCacheStats",
    "Verifier",
    "VerifierRoute",
    "VerificationPlanCostEstimate",
    "VerificationRouteHint",
    "SentenceClaimExtractor",
    "SelfConsistencyVerifier",
    "apply_claim_coherence",
    "audit_claim_triples",
    "claim_features",
    "extract_calculation",
    "extract_claims",
    "extract_claim_triples",
    "estimate_verification_plan_cost",
    "infer_claim_dependencies",
    "normalize_claim_text",
    "stable_cache_key",
    "verifier_cache_key",
]
