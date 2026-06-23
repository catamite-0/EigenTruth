"""Verifier composition helpers."""

from __future__ import annotations

import re
import time
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
        return bool(self.match_reasons(claim, context=context))

    def match_reasons(self, claim: Claim, context: Mapping[str, Any] | None = None) -> tuple[str, ...]:
        """Return auditable reasons why this route matches a claim."""
        reasons = []
        if self.fallback:
            reasons.append("fallback")
        metadata = claim.metadata if isinstance(claim.metadata, Mapping) else {}
        features = metadata.get("features", {})
        if isinstance(features, Mapping):
            for flag in self.feature_flags:
                if features.get(flag) is True:
                    reasons.append(f"feature_flag:{flag}")
        for key in self.metadata_keys:
            if _has_path(metadata, key):
                reasons.append(f"metadata:{key}")
        context_mapping = context if isinstance(context, Mapping) else {}
        for key in self.context_keys:
            if _has_path(context_mapping, key):
                reasons.append(f"context:{key}")
        for pattern in self.text_patterns:
            if re.search(pattern, claim.text, re.IGNORECASE):
                reasons.append(f"text_pattern:{pattern}")
        return tuple(reasons)


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
        matched = [
            (route, reasons)
            for route in self.routes
            if (reasons := route.match_reasons(claim, context=context))
        ]
        skipped = []
        total_duration_seconds = 0.0
        for route, reasons in matched:
            started_at = time.perf_counter()
            result = route.verifier.verify(claim, context=context)
            route_duration_seconds = time.perf_counter() - started_at
            total_duration_seconds += route_duration_seconds
            if result.status not in route.fallthrough_statuses:
                return _with_route_metadata(
                    result,
                    route=route,
                    matched=matched,
                    skipped=skipped,
                    total_duration_seconds=total_duration_seconds,
                    selected_route_duration_seconds=route_duration_seconds,
                )
            skipped.append({
                "route": route.name,
                "verifier": type(route.verifier).__name__,
                "match_reasons": tuple(reasons),
                "status": result.status.value,
                "explanation": result.explanation,
                "duration_seconds": route_duration_seconds,
            })
        metadata: dict[str, Any] = {
            "verifier": type(self).__name__,
            "matched_routes": tuple(route.name for route, _ in matched),
            "matched_route_details": _route_match_details(matched),
            "skipped_routes": tuple(skipped),
        }
        if skipped:
            metadata["total_duration_seconds"] = total_duration_seconds
            metadata["attempted_route_count"] = float(len(skipped))
        return VerificationResult(
            status=VerificationStatus.NOT_APPLICABLE,
            confidence=1.0,
            explanation="no matched verifier route was applicable to claim",
            metadata=metadata,
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
    matched: Sequence[tuple[VerifierRoute, Sequence[str]]],
    skipped: Sequence[Mapping[str, Any]],
    total_duration_seconds: float,
    selected_route_duration_seconds: float,
) -> VerificationResult:
    metadata = {
        **dict(result.metadata),
        "router_verifier": "RoutedVerifier",
        "selected_route": route.name,
        "selected_verifier": type(route.verifier).__name__,
        "matched_routes": tuple(item.name for item, _ in matched),
        "matched_route_details": _route_match_details(matched),
        "skipped_routes": tuple(dict(item) for item in skipped),
        "total_duration_seconds": total_duration_seconds,
        "selected_route_duration_seconds": selected_route_duration_seconds,
        "attempted_route_count": float(len(skipped) + 1),
    }
    return VerificationResult(
        status=result.status,
        confidence=result.confidence,
        evidence=tuple(result.evidence),
        explanation=result.explanation,
        metadata=metadata,
    )


def _route_match_details(matched: Sequence[tuple[VerifierRoute, Sequence[str]]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "route": route.name,
            "verifier": type(route.verifier).__name__,
            "match_reasons": tuple(str(reason) for reason in reasons),
        }
        for route, reasons in matched
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
