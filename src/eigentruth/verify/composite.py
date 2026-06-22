"""Verifier composition helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus, Verifier


@dataclass(frozen=True)
class CompositeVerifier:
    """Run verifiers in order, skipping `not_applicable` results.

    This is useful for tool-first verification: deterministic tools such as a
    calculator can decide applicable claims, while lexical or retrieval-backed
    verifiers handle the remaining claims.
    """

    verifiers: Sequence[Verifier]

    def __post_init__(self) -> None:
        verifiers = tuple(self.verifiers)
        if not verifiers:
            raise ValueError("CompositeVerifier requires at least one verifier.")
        object.__setattr__(self, "verifiers", verifiers)

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one claim with the first applicable verifier result."""
        skipped = []
        for verifier in self.verifiers:
            result = verifier.verify(claim, context=context)
            verifier_name = type(verifier).__name__
            if result.status is not VerificationStatus.NOT_APPLICABLE:
                return _with_composite_metadata(result, verifier_name=verifier_name, skipped=skipped)
            skipped.append({
                "verifier": verifier_name,
                "status": result.status.value,
                "explanation": result.explanation,
            })
        return VerificationResult(
            status=VerificationStatus.NOT_APPLICABLE,
            confidence=1.0,
            explanation="no verifier was applicable to claim",
            metadata={"verifier": type(self).__name__, "skipped_verifiers": tuple(skipped)},
        )

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        """Verify multiple claims."""
        return tuple(self.verify(claim, context=context) for claim in claims)


@dataclass(frozen=True)
class VerifierRoute:
    """Metadata, context, or text-based verifier route."""

    name: str
    verifier: Verifier
    feature_flags: Sequence[str] = ()
    metadata_keys: Sequence[str] = ()
    context_keys: Sequence[str] = ()
    text_patterns: Sequence[str] = ()
    fallthrough_statuses: Sequence[VerificationStatus | str] = (VerificationStatus.NOT_APPLICABLE,)
    fallback: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("route name must be non-empty.")
        object.__setattr__(self, "feature_flags", tuple(self.feature_flags))
        object.__setattr__(self, "metadata_keys", tuple(self.metadata_keys))
        object.__setattr__(self, "context_keys", tuple(self.context_keys))
        object.__setattr__(self, "text_patterns", tuple(self.text_patterns))
        object.__setattr__(
            self,
            "fallthrough_statuses",
            tuple(_coerce_status(status) for status in self.fallthrough_statuses),
        )

    def matches(self, claim: Claim, context: Mapping[str, Any] | None = None) -> bool:
        """Return whether this route should be tried for a claim."""
        if self.fallback:
            return True
        metadata = claim.metadata if isinstance(claim.metadata, Mapping) else {}
        features = metadata.get("features", {})
        if isinstance(features, Mapping):
            for flag in self.feature_flags:
                if features.get(flag) is True:
                    return True
        for key in self.metadata_keys:
            if _has_path(metadata, key):
                return True
        context_mapping = context if isinstance(context, Mapping) else {}
        for key in self.context_keys:
            if _has_path(context_mapping, key):
                return True
        return any(re.search(pattern, claim.text, re.IGNORECASE) for pattern in self.text_patterns)


@dataclass(frozen=True)
class RoutedVerifier:
    """Route claims to matching verifiers, then return the first applicable result."""

    routes: Sequence[VerifierRoute]

    def __post_init__(self) -> None:
        routes = tuple(self.routes)
        if not routes:
            raise ValueError("RoutedVerifier requires at least one route.")
        object.__setattr__(self, "routes", routes)

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one claim through matching routes."""
        matched = [route for route in self.routes if route.matches(claim, context=context)]
        skipped = []
        for route in matched:
            result = route.verifier.verify(claim, context=context)
            if result.status not in route.fallthrough_statuses:
                return _with_route_metadata(result, route=route, matched=matched, skipped=skipped)
            skipped.append({
                "route": route.name,
                "verifier": type(route.verifier).__name__,
                "status": result.status.value,
                "explanation": result.explanation,
            })
        return VerificationResult(
            status=VerificationStatus.NOT_APPLICABLE,
            confidence=1.0,
            explanation="no matched verifier route was applicable to claim",
            metadata={
                "verifier": type(self).__name__,
                "matched_routes": tuple(route.name for route in matched),
                "skipped_routes": tuple(skipped),
            },
        )

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        """Verify multiple claims."""
        return tuple(self.verify(claim, context=context) for claim in claims)


def _with_composite_metadata(
    result: VerificationResult,
    *,
    verifier_name: str,
    skipped: Sequence[Mapping[str, Any]],
) -> VerificationResult:
    metadata = {
        **dict(result.metadata),
        "composite_verifier": "CompositeVerifier",
        "selected_verifier": verifier_name,
        "skipped_verifiers": tuple(dict(item) for item in skipped),
    }
    return VerificationResult(
        status=result.status,
        confidence=result.confidence,
        evidence=tuple(result.evidence),
        explanation=result.explanation,
        metadata=metadata,
    )


def _with_route_metadata(
    result: VerificationResult,
    *,
    route: VerifierRoute,
    matched: Sequence[VerifierRoute],
    skipped: Sequence[Mapping[str, Any]],
) -> VerificationResult:
    metadata = {
        **dict(result.metadata),
        "router_verifier": "RoutedVerifier",
        "selected_route": route.name,
        "selected_verifier": type(route.verifier).__name__,
        "matched_routes": tuple(item.name for item in matched),
        "skipped_routes": tuple(dict(item) for item in skipped),
    }
    return VerificationResult(
        status=result.status,
        confidence=result.confidence,
        evidence=tuple(result.evidence),
        explanation=result.explanation,
        metadata=metadata,
    )


def _has_path(data: Mapping[str, Any], path: str) -> bool:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return current is not None


def _coerce_status(value: VerificationStatus | str) -> VerificationStatus:
    if isinstance(value, VerificationStatus):
        return value
    return VerificationStatus(str(value))
