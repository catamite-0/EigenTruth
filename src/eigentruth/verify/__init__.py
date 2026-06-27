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
from eigentruth.verify.citations import (
    CitationRecord,
    CitationVerifier,
    extract_citation_references,
)
from eigentruth.verify.claims import (
    ClaimExtractor,
    SentenceClaimExtractor,
    claim_features,
    enrich_claims_with_triples,
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
from eigentruth.verify.localization import (
    ClaimRiskLocalizationReport,
    ClaimRiskSpan,
    localize_claim_risk_spans,
)
from eigentruth.verify.planning import (
    ClaimVerificationPlan,
    ClaimVerificationPlanner,
    VerificationBudgetPolicy,
    VerificationPlanCostEstimate,
    VerificationRouteHint,
    budget_verification_plan,
    estimate_verification_plan_cost,
)
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus, Verifier
from eigentruth.verify.routing import default_routed_verifier, default_verifier_routes
from eigentruth.verify.rules import InMemoryVerifier, normalize_claim_text
from eigentruth.verify.selfcheck import SelfConsistencyVerifier
from eigentruth.verify.triples import (
    ClaimTriple,
    ClaimTripleExtractor,
    CompositeTripleExtractor,
    LookupTripleExtractor,
    RegexTripleExtractor,
    RegexTriplePattern,
    RuleBasedTripleExtractor,
    TripleEvidenceAudit,
    TripleEvidenceAuditReport,
    TripleEvidenceVerifier,
    TripleSlotEvidence,
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
    "ClaimRiskLocalizationReport",
    "ClaimRiskSpan",
    "CitationRecord",
    "CitationVerifier",
    "ClaimTriple",
    "ClaimTripleExtractor",
    "ClaimVerificationPlan",
    "ClaimVerificationPlanner",
    "CompositeTripleExtractor",
    "CompositeVerifier",
    "EvidenceDocument",
    "EvidenceQualityPolicy",
    "GroundednessVerifier",
    "InMemoryVerifier",
    "JsonTraceCache",
    "LookupTripleExtractor",
    "RegexTripleExtractor",
    "RegexTriplePattern",
    "RuleBasedTripleExtractor",
    "RoutedVerifier",
    "TraceCacheRecord",
    "TripleEvidenceAudit",
    "TripleEvidenceAuditReport",
    "TripleEvidenceVerifier",
    "TripleSlotEvidence",
    "VerificationResult",
    "VerificationStatus",
    "VerificationBudgetPolicy",
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
    "default_routed_verifier",
    "default_verifier_routes",
    "enrich_claims_with_triples",
    "extract_calculation",
    "extract_citation_references",
    "extract_claims",
    "extract_claim_triples",
    "localize_claim_risk_spans",
    "budget_verification_plan",
    "estimate_verification_plan_cost",
    "infer_claim_dependencies",
    "normalize_claim_text",
    "stable_cache_key",
    "verifier_cache_key",
]
