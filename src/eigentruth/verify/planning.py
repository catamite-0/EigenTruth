"""Claim verification planning helpers.

The planner turns generated text or pre-extracted claims into a dependency-free
verification plan. It does not execute retrievers, calculators, state adapters,
or world-model adapters; it only emits stable routing hints and tool-shaped
payloads that downstream control/runtime layers can consume.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from eigentruth.json_utils import to_jsonable
from eigentruth.verify.citations import extract_citation_references
from eigentruth.verify.claims import ClaimExtractor, extract_claims
from eigentruth.verify.coherence import ClaimDependency, infer_claim_dependencies
from eigentruth.verify.features import enabled_feature_names, metadata_path_enabled
from eigentruth.verify.protocols import Claim

DEFAULT_VERIFICATION_ROUTE_COST_UNITS: Mapping[str, float] = {
    "groundedness": 0.25,
    "citation": 0.30,
    "triple_evidence": 0.35,
    "calculator": 0.5,
    "state": 0.75,
    "retrieval": 1.0,
    "world_model": 1.5,
}
DEFAULT_VERIFICATION_TOOL_PAYLOAD_COST_UNITS: Mapping[str, float] = {
    "retrieval_queries": 0.5,
    "citation_checks": 0.2,
    "calculation_checks": 0.25,
    "state_checks": 0.5,
    "world_model_checks": 0.75,
}
DEFAULT_VERIFICATION_ROUTE_PRIORITY = (
    "world_model",
    "state",
    "calculator",
    "citation",
    "triple_evidence",
    "retrieval",
    "groundedness",
)
DEFAULT_VERIFY_CLAIM_FEATURE_FLAGS = (
    "has_number",
    "has_citation",
    "is_time_sensitive",
    "has_calculation",
)
DEFAULT_VERIFY_CLAIM_METADATA_KEYS = (
    "requires_verification",
    "calculation",
    "state_check",
    "state_checks",
    "world_model_check",
    "world_model_checks",
    "state_transition",
    "requires_triple_audit",
    "triples",
    "claim_triples",
    "retrieval_query",
    "retrieval_queries",
    "route_hints",
    "routes",
)
DEFAULT_RETRIEVAL_FEATURE_FLAGS = (
    "has_number",
    "has_citation",
    "has_negation",
    "is_time_sensitive",
)
DEFAULT_RETRIEVAL_METADATA_KEYS = (
    "requires_verification",
    "retrieval_query",
    "retrieval_queries",
)
DEFAULT_CITATION_FEATURE_FLAGS = ("has_citation",)
DEFAULT_CITATION_METADATA_KEYS = (
    "citation",
    "citations",
    "citation_check",
    "citation_checks",
)
DEFAULT_CALCULATOR_METADATA_KEYS = ("calculation",)
DEFAULT_STATE_METADATA_KEYS = ("state_check", "state_checks")
DEFAULT_WORLD_MODEL_METADATA_KEYS = (
    "world_model_check",
    "world_model_checks",
    "state_transition",
)
DEFAULT_EXPLICIT_ROUTE_METADATA_KEYS = ("route_hints", "routes")
DEFAULT_TRIPLE_EVIDENCE_FEATURE_FLAGS = (
    "has_number",
    "has_citation",
    "is_time_sensitive",
)
DEFAULT_TRIPLE_EVIDENCE_METADATA_KEYS = (
    "requires_triple_audit",
    "triples",
    "claim_triples",
)


@dataclass(frozen=True)
class VerificationRouteHint:
    """Auditable verifier/tool route hints for one claim."""

    claim_id: str
    routes: Sequence[str]
    reasons: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        claim_id = str(self.claim_id).strip()
        if not claim_id:
            raise ValueError("route hint claim_id must be non-empty.")
        routes = tuple(_non_empty_strings(self.routes))
        if not routes:
            raise ValueError("route hint routes must be non-empty.")
        object.__setattr__(self, "claim_id", claim_id)
        object.__setattr__(self, "routes", routes)
        object.__setattr__(self, "reasons", tuple(_non_empty_strings(self.reasons)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "claim_id": self.claim_id,
            "routes": tuple(self.routes),
            "reasons": tuple(self.reasons),
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class VerificationPlanCostEstimate:
    """Dependency-free cost summary for a claim verification plan."""

    claim_count: int
    verify_claim_count: int
    skipped_claim_count: int
    route_counts: Mapping[str, int] = field(default_factory=dict)
    tool_payload_counts: Mapping[str, int] = field(default_factory=dict)
    dependency_count: int = 0
    estimated_route_attempts: int = 0
    estimated_tool_payloads: int = 0
    estimated_route_cost_units: float = 0.0
    estimated_tool_payload_cost_units: float = 0.0

    def __post_init__(self) -> None:
        claim_count = _non_negative_int(self.claim_count, name="claim_count")
        verify_claim_count = _non_negative_int(self.verify_claim_count, name="verify_claim_count")
        skipped_claim_count = _non_negative_int(self.skipped_claim_count, name="skipped_claim_count")
        dependency_count = _non_negative_int(self.dependency_count, name="dependency_count")
        route_counts = _non_negative_int_mapping(self.route_counts, name="route_counts")
        tool_payload_counts = _non_negative_int_mapping(self.tool_payload_counts, name="tool_payload_counts")
        estimated_route_attempts = _non_negative_int(
            self.estimated_route_attempts,
            name="estimated_route_attempts",
        )
        estimated_tool_payloads = _non_negative_int(
            self.estimated_tool_payloads,
            name="estimated_tool_payloads",
        )
        estimated_route_cost_units = _non_negative_float(
            self.estimated_route_cost_units,
            name="estimated_route_cost_units",
        )
        estimated_tool_payload_cost_units = _non_negative_float(
            self.estimated_tool_payload_cost_units,
            name="estimated_tool_payload_cost_units",
        )
        object.__setattr__(self, "claim_count", claim_count)
        object.__setattr__(self, "verify_claim_count", verify_claim_count)
        object.__setattr__(self, "skipped_claim_count", skipped_claim_count)
        object.__setattr__(self, "route_counts", route_counts)
        object.__setattr__(self, "tool_payload_counts", tool_payload_counts)
        object.__setattr__(self, "dependency_count", dependency_count)
        object.__setattr__(self, "estimated_route_attempts", estimated_route_attempts)
        object.__setattr__(self, "estimated_tool_payloads", estimated_tool_payloads)
        object.__setattr__(self, "estimated_route_cost_units", estimated_route_cost_units)
        object.__setattr__(self, "estimated_tool_payload_cost_units", estimated_tool_payload_cost_units)

    @property
    def estimated_cost_units(self) -> float:
        """Return route and tool-payload cost units combined."""
        return self.estimated_route_cost_units + self.estimated_tool_payload_cost_units

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "claim_count": self.claim_count,
            "verify_claim_count": self.verify_claim_count,
            "skipped_claim_count": self.skipped_claim_count,
            "route_counts": dict(self.route_counts),
            "tool_payload_counts": dict(self.tool_payload_counts),
            "dependency_count": self.dependency_count,
            "estimated_route_attempts": self.estimated_route_attempts,
            "estimated_tool_payloads": self.estimated_tool_payloads,
            "estimated_route_cost_units": self.estimated_route_cost_units,
            "estimated_tool_payload_cost_units": self.estimated_tool_payload_cost_units,
            "estimated_cost_units": self.estimated_cost_units,
        }


@dataclass(frozen=True)
class VerificationBudgetPolicy:
    """Cost-aware policy for selecting verifier claims and routes.

    The policy is intentionally dependency-free and operates on the planned
    verifier graph before any external tool or retriever runs. It keeps high-risk
    claim metadata explicit while making route/cost truncation auditable.
    """

    max_verify_claims: int | None = None
    max_route_attempts: int | None = None
    max_tool_payloads: int | None = None
    max_estimated_cost_units: float | None = None
    route_priority: Sequence[str] = DEFAULT_VERIFICATION_ROUTE_PRIORITY
    priority_feature_flags: Sequence[str] = DEFAULT_VERIFY_CLAIM_FEATURE_FLAGS
    priority_metadata_keys: Sequence[str] = DEFAULT_VERIFY_CLAIM_METADATA_KEYS
    preserve_triggered_claims: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_verify_claims",
            _optional_non_negative_int(self.max_verify_claims, name="max_verify_claims"),
        )
        object.__setattr__(
            self,
            "max_route_attempts",
            _optional_non_negative_int(self.max_route_attempts, name="max_route_attempts"),
        )
        object.__setattr__(
            self,
            "max_tool_payloads",
            _optional_non_negative_int(self.max_tool_payloads, name="max_tool_payloads"),
        )
        object.__setattr__(
            self,
            "max_estimated_cost_units",
            _optional_non_negative_float(
                self.max_estimated_cost_units,
                name="max_estimated_cost_units",
            ),
        )
        object.__setattr__(self, "route_priority", tuple(_non_empty_strings(self.route_priority)))
        object.__setattr__(
            self,
            "priority_feature_flags",
            tuple(_non_empty_strings(self.priority_feature_flags)),
        )
        object.__setattr__(
            self,
            "priority_metadata_keys",
            tuple(_non_empty_strings(self.priority_metadata_keys)),
        )
        object.__setattr__(
            self,
            "preserve_triggered_claims",
            _strict_bool(self.preserve_triggered_claims),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "VerificationBudgetPolicy":
        """Build a policy from a JSON-like mapping."""
        return cls(
            max_verify_claims=payload.get("max_verify_claims"),
            max_route_attempts=payload.get("max_route_attempts"),
            max_tool_payloads=payload.get("max_tool_payloads"),
            max_estimated_cost_units=payload.get("max_estimated_cost_units"),
            route_priority=tuple(_as_sequence(payload.get("route_priority", DEFAULT_VERIFICATION_ROUTE_PRIORITY))),
            priority_feature_flags=tuple(
                _as_sequence(payload.get("priority_feature_flags", DEFAULT_VERIFY_CLAIM_FEATURE_FLAGS))
            ),
            priority_metadata_keys=tuple(
                _as_sequence(payload.get("priority_metadata_keys", DEFAULT_VERIFY_CLAIM_METADATA_KEYS))
            ),
            preserve_triggered_claims=payload.get("preserve_triggered_claims", True),
        )

    def enabled(self) -> bool:
        """Return whether the policy has at least one active budget."""
        return (
            self.max_verify_claims is not None
            or self.max_route_attempts is not None
            or self.max_tool_payloads is not None
            or self.max_estimated_cost_units is not None
        )

    def apply(
        self,
        plan: "ClaimVerificationPlan | Mapping[str, Any]",
        *,
        route_cost_units: Mapping[str, Any] | None = None,
        tool_payload_cost_units: Mapping[str, Any] | None = None,
    ) -> "ClaimVerificationPlan":
        """Return ``plan`` with claims and routes selected under this budget."""
        return budget_verification_plan(
            plan,
            self,
            route_cost_units=route_cost_units,
            tool_payload_cost_units=tool_payload_cost_units,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable policy."""
        return {
            "max_verify_claims": self.max_verify_claims,
            "max_route_attempts": self.max_route_attempts,
            "max_tool_payloads": self.max_tool_payloads,
            "max_estimated_cost_units": self.max_estimated_cost_units,
            "route_priority": tuple(self.route_priority),
            "priority_feature_flags": tuple(self.priority_feature_flags),
            "priority_metadata_keys": tuple(self.priority_metadata_keys),
            "preserve_triggered_claims": self.preserve_triggered_claims,
            "enabled": self.enabled(),
        }


@dataclass(frozen=True)
class ClaimVerificationPlan:
    """JSON-ready plan for claim verification and tool routing."""

    run_verifier: bool
    reason: str
    verification_scope: str | None = None
    claims: Sequence[Claim | Mapping[str, Any]] = ()
    verify_claim_ids: Sequence[str] = ()
    skipped_claim_ids: Sequence[str] = ()
    triggered_claim_ids: Sequence[str] = ()
    triggered_features: Mapping[str, Sequence[str]] = field(default_factory=dict)
    triggered_metadata: Mapping[str, Sequence[str]] = field(default_factory=dict)
    route_hints: Sequence[VerificationRouteHint | Mapping[str, Any]] = ()
    retrieval_queries: Sequence[Mapping[str, Any]] = ()
    citation_checks: Sequence[Mapping[str, Any]] = ()
    calculation_checks: Sequence[Mapping[str, Any]] = ()
    state_checks: Sequence[Mapping[str, Any]] = ()
    world_model_checks: Sequence[Mapping[str, Any]] = ()
    dependencies: Sequence[ClaimDependency | Mapping[str, Any]] = ()
    budget: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scope = _verification_scope(self.verification_scope, run_verifier=self.run_verifier)
        claims = tuple(_coerce_claim(item) for item in self.claims)
        object.__setattr__(self, "verification_scope", scope)
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "verify_claim_ids", tuple(_non_empty_strings(self.verify_claim_ids)))
        object.__setattr__(self, "skipped_claim_ids", tuple(_non_empty_strings(self.skipped_claim_ids)))
        object.__setattr__(self, "triggered_claim_ids", tuple(_non_empty_strings(self.triggered_claim_ids)))
        object.__setattr__(
            self,
            "triggered_features",
            _normalize_string_sequence_mapping(self.triggered_features),
        )
        object.__setattr__(
            self,
            "triggered_metadata",
            _normalize_string_sequence_mapping(self.triggered_metadata),
        )
        object.__setattr__(self, "route_hints", tuple(_route_hint_obj(item) for item in self.route_hints))
        object.__setattr__(self, "retrieval_queries", tuple(_jsonable_mapping(item) for item in self.retrieval_queries))
        object.__setattr__(self, "citation_checks", tuple(_jsonable_mapping(item) for item in self.citation_checks))
        object.__setattr__(
            self,
            "calculation_checks",
            tuple(_jsonable_mapping(item) for item in self.calculation_checks),
        )
        object.__setattr__(self, "state_checks", tuple(_jsonable_mapping(item) for item in self.state_checks))
        object.__setattr__(
            self,
            "world_model_checks",
            tuple(_jsonable_mapping(item) for item in self.world_model_checks),
        )
        object.__setattr__(self, "dependencies", tuple(_dependency_obj(item) for item in self.dependencies))
        object.__setattr__(self, "budget", _jsonable_mapping(self.budget) if self.budget else {})

    def selected_claims(self) -> tuple[Claim, ...]:
        """Return the claims selected for verification by this plan."""
        selected_ids = set(self.verify_claim_ids)
        if not selected_ids:
            return ()
        return tuple(
            claim
            for index, claim in enumerate(self.claims)
            if _claim_id(claim, index) in selected_ids
        )

    def cost_estimate(
        self,
        *,
        route_cost_units: Mapping[str, Any] | None = None,
        tool_payload_cost_units: Mapping[str, Any] | None = None,
    ) -> VerificationPlanCostEstimate:
        """Estimate relative verifier/tool cost from the plan shape."""
        return estimate_verification_plan_cost(
            self,
            route_cost_units=route_cost_units,
            tool_payload_cost_units=tool_payload_cost_units,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "run_verifier": bool(self.run_verifier),
            "reason": self.reason,
            "verification_scope": self.verification_scope,
            "claims": tuple(
                _claim_to_dict(claim, fallback_id=f"c{index + 1}")
                for index, claim in enumerate(self.claims)
            ),
            "verify_claim_ids": tuple(self.verify_claim_ids),
            "skipped_claim_ids": tuple(self.skipped_claim_ids),
            "triggered_claim_ids": tuple(self.triggered_claim_ids),
            "triggered_features": {
                claim_id: tuple(features)
                for claim_id, features in self.triggered_features.items()
            },
            "triggered_metadata": {
                claim_id: tuple(keys)
                for claim_id, keys in self.triggered_metadata.items()
            },
            "route_hints": tuple(item.to_dict() for item in self.route_hints),
            "retrieval_queries": tuple(dict(item) for item in self.retrieval_queries),
            "citation_checks": tuple(dict(item) for item in self.citation_checks),
            "calculation_checks": tuple(dict(item) for item in self.calculation_checks),
            "state_checks": tuple(dict(item) for item in self.state_checks),
            "world_model_checks": tuple(dict(item) for item in self.world_model_checks),
            "dependencies": tuple(item.to_dict() for item in self.dependencies),
            "budget": to_jsonable(dict(self.budget)),
            "cost_estimate": self.cost_estimate().to_dict(),
        }


@dataclass(frozen=True)
class ClaimVerificationPlanner:
    """Build dependency-free claim verification plans."""

    extractor: ClaimExtractor | None = None
    min_chars: int = 3
    verify_claim_feature_flags: Sequence[str] = DEFAULT_VERIFY_CLAIM_FEATURE_FLAGS
    verify_claim_metadata_keys: Sequence[str] = DEFAULT_VERIFY_CLAIM_METADATA_KEYS
    verify_triggered_claims_only: bool = False
    verify_all_by_default: bool = True
    retrieval_feature_flags: Sequence[str] = DEFAULT_RETRIEVAL_FEATURE_FLAGS
    retrieval_metadata_keys: Sequence[str] = DEFAULT_RETRIEVAL_METADATA_KEYS
    citation_feature_flags: Sequence[str] = DEFAULT_CITATION_FEATURE_FLAGS
    citation_metadata_keys: Sequence[str] = DEFAULT_CITATION_METADATA_KEYS
    calculator_metadata_keys: Sequence[str] = DEFAULT_CALCULATOR_METADATA_KEYS
    state_metadata_keys: Sequence[str] = DEFAULT_STATE_METADATA_KEYS
    world_model_metadata_keys: Sequence[str] = DEFAULT_WORLD_MODEL_METADATA_KEYS
    explicit_route_metadata_keys: Sequence[str] = DEFAULT_EXPLICIT_ROUTE_METADATA_KEYS
    triple_evidence_feature_flags: Sequence[str] = DEFAULT_TRIPLE_EVIDENCE_FEATURE_FLAGS
    triple_evidence_metadata_keys: Sequence[str] = DEFAULT_TRIPLE_EVIDENCE_METADATA_KEYS
    infer_dependencies: bool = True
    include_extracted_triples: bool = False

    def __post_init__(self) -> None:
        if self.min_chars < 1:
            raise ValueError("min_chars must be >= 1.")
        object.__setattr__(
            self,
            "verify_claim_feature_flags",
            tuple(_non_empty_strings(self.verify_claim_feature_flags)),
        )
        object.__setattr__(
            self,
            "verify_claim_metadata_keys",
            tuple(_non_empty_strings(self.verify_claim_metadata_keys)),
        )
        object.__setattr__(self, "verify_triggered_claims_only", _strict_bool(self.verify_triggered_claims_only))
        object.__setattr__(self, "verify_all_by_default", _strict_bool(self.verify_all_by_default))
        object.__setattr__(self, "retrieval_feature_flags", tuple(_non_empty_strings(self.retrieval_feature_flags)))
        object.__setattr__(self, "retrieval_metadata_keys", tuple(_non_empty_strings(self.retrieval_metadata_keys)))
        object.__setattr__(self, "citation_feature_flags", tuple(_non_empty_strings(self.citation_feature_flags)))
        object.__setattr__(self, "citation_metadata_keys", tuple(_non_empty_strings(self.citation_metadata_keys)))
        object.__setattr__(self, "calculator_metadata_keys", tuple(_non_empty_strings(self.calculator_metadata_keys)))
        object.__setattr__(self, "state_metadata_keys", tuple(_non_empty_strings(self.state_metadata_keys)))
        object.__setattr__(
            self,
            "world_model_metadata_keys",
            tuple(_non_empty_strings(self.world_model_metadata_keys)),
        )
        object.__setattr__(
            self,
            "explicit_route_metadata_keys",
            tuple(_non_empty_strings(self.explicit_route_metadata_keys)),
        )
        object.__setattr__(
            self,
            "triple_evidence_feature_flags",
            tuple(_non_empty_strings(self.triple_evidence_feature_flags)),
        )
        object.__setattr__(
            self,
            "triple_evidence_metadata_keys",
            tuple(_non_empty_strings(self.triple_evidence_metadata_keys)),
        )
        object.__setattr__(self, "infer_dependencies", _strict_bool(self.infer_dependencies))
        object.__setattr__(
            self,
            "include_extracted_triples",
            _strict_bool(self.include_extracted_triples),
        )

    def extract(self, text: str) -> tuple[Claim, ...]:
        """Extract claims with the configured extractor."""
        return extract_claims(
            text,
            min_chars=self.min_chars,
            extractor=self.extractor,
            include_triples=self.include_extracted_triples,
        )

    def plan(
        self,
        claims_or_text: str | Sequence[Claim | Mapping[str, Any]],
        *,
        context: Mapping[str, Any] | None = None,
        budget_policy: VerificationBudgetPolicy | Mapping[str, Any] | None = None,
    ) -> ClaimVerificationPlan:
        """Return a verification plan for generated text or existing claims."""
        claims = self.extract(claims_or_text) if isinstance(claims_or_text, str) else _coerce_claims(claims_or_text)
        if not claims:
            return ClaimVerificationPlan(
                run_verifier=False,
                reason="no claims to verify",
                verification_scope="none",
            )

        claim_ids = tuple(_claim_id(claim, index) for index, claim in enumerate(claims))
        triggered_features: dict[str, tuple[str, ...]] = {}
        triggered_metadata: dict[str, tuple[str, ...]] = {}
        route_hints: list[VerificationRouteHint] = []
        retrieval_queries: list[dict[str, Any]] = []
        citation_checks: list[dict[str, Any]] = []
        calculation_checks: list[dict[str, Any]] = []
        state_checks: list[dict[str, Any]] = []
        world_model_checks: list[dict[str, Any]] = []
        triggered_claim_ids: list[str] = []

        for index, claim in enumerate(claims):
            claim_id = claim_ids[index]
            metadata = _claim_metadata(claim)
            features = _claim_features(metadata)
            matched_features = enabled_feature_names(features, self.verify_claim_feature_flags)
            matched_metadata = tuple(
                key for key in self.verify_claim_metadata_keys if metadata_path_enabled(metadata, key)
            )
            if matched_features:
                triggered_features[claim_id] = matched_features
            if matched_metadata:
                triggered_metadata[claim_id] = matched_metadata
            if matched_features or matched_metadata:
                triggered_claim_ids.append(claim_id)

            routes, reasons = self._routes_for_claim(claim, claim_id=claim_id, features=features, metadata=metadata)
            route_hints.append(VerificationRouteHint(claim_id=claim_id, routes=routes, reasons=reasons))
            retrieval_queries.extend(
                self._retrieval_queries_for_claim(
                    claim,
                    claim_id=claim_id,
                    routes=routes,
                    metadata=metadata,
                )
            )
            citation_checks.extend(
                self._citation_checks_for_claim(
                    claim,
                    claim_id=claim_id,
                    routes=routes,
                )
            )
            calculation_checks.extend(_metadata_tool_payloads(claim_id, metadata, self.calculator_metadata_keys))
            state_checks.extend(_metadata_tool_payloads(claim_id, metadata, self.state_metadata_keys))
            world_model_checks.extend(_metadata_tool_payloads(claim_id, metadata, self.world_model_metadata_keys))

        triggered_claim_ids_tuple = tuple(dict.fromkeys(triggered_claim_ids))
        if self.verify_all_by_default:
            verify_claim_ids = claim_ids
            skipped_claim_ids: tuple[str, ...] = ()
            run_verifier = True
            scope = "all"
            reason = "verify all claims by default"
        elif triggered_claim_ids_tuple:
            triggered_set = set(triggered_claim_ids_tuple)
            verify_claim_ids = triggered_claim_ids_tuple if self.verify_triggered_claims_only else claim_ids
            skipped_claim_ids = tuple(claim_id for claim_id in claim_ids if claim_id not in triggered_set)
            run_verifier = True
            scope = "triggered" if self.verify_triggered_claims_only else "all"
            reason = "claim metadata requires verification"
        else:
            verify_claim_ids = ()
            skipped_claim_ids = claim_ids
            run_verifier = False
            scope = "none"
            reason = "claim metadata did not require verification"

        dependencies = infer_claim_dependencies(claims) if self.infer_dependencies else ()
        plan = ClaimVerificationPlan(
            run_verifier=run_verifier,
            reason=reason,
            verification_scope=scope,
            claims=claims,
            verify_claim_ids=verify_claim_ids,
            skipped_claim_ids=skipped_claim_ids,
            triggered_claim_ids=triggered_claim_ids_tuple,
            triggered_features=triggered_features,
            triggered_metadata=triggered_metadata,
            route_hints=tuple(route_hints),
            retrieval_queries=tuple(retrieval_queries),
            citation_checks=tuple(citation_checks),
            calculation_checks=tuple(calculation_checks),
            state_checks=tuple(state_checks),
            world_model_checks=tuple(world_model_checks),
            dependencies=dependencies,
        )
        if budget_policy is None:
            return plan
        return budget_verification_plan(plan, budget_policy)

    def _routes_for_claim(
        self,
        claim: Claim,
        *,
        claim_id: str,
        features: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        del claim_id
        routes: list[str] = []
        reasons: list[str] = []
        for route in _explicit_routes(metadata, self.explicit_route_metadata_keys):
            _append_unique(routes, route)
            reasons.append(f"metadata:route:{route}")
        for key in self.triple_evidence_metadata_keys:
            if metadata_path_enabled(metadata, key):
                _append_unique(routes, "triple_evidence")
                reasons.append(f"metadata:{key}")
        for key in self.calculator_metadata_keys:
            if metadata_path_enabled(metadata, key):
                _append_unique(routes, "calculator")
                reasons.append(f"metadata:{key}")
        for key in self.state_metadata_keys:
            if metadata_path_enabled(metadata, key):
                _append_unique(routes, "state")
                reasons.append(f"metadata:{key}")
        for key in self.world_model_metadata_keys:
            if metadata_path_enabled(metadata, key):
                _append_unique(routes, "world_model")
                reasons.append(f"metadata:{key}")
        for key in self.citation_metadata_keys:
            if metadata_path_enabled(metadata, key):
                _append_unique(routes, "citation")
                reasons.append(f"metadata:{key}")
        for feature in enabled_feature_names(features, self.citation_feature_flags):
            _append_unique(routes, "citation")
            reasons.append(f"feature:{feature}")
        for key in self.retrieval_metadata_keys:
            if metadata_path_enabled(metadata, key):
                _append_unique(routes, "retrieval")
                reasons.append(f"metadata:{key}")
        for feature in enabled_feature_names(features, self.retrieval_feature_flags):
            _append_unique(routes, "retrieval")
            reasons.append(f"feature:{feature}")
        for feature in enabled_feature_names(features, self.triple_evidence_feature_flags):
            _append_unique(routes, "triple_evidence")
            reasons.append(f"feature:{feature}")
        if _feature_enabled(features, "has_calculation"):
            _append_unique(routes, "calculator")
            reasons.append("feature:has_calculation")
        _append_unique(routes, "groundedness")
        if not reasons:
            reasons.append("default:groundedness")
        return tuple(routes), tuple(dict.fromkeys(reasons))

    def _retrieval_queries_for_claim(
        self,
        claim: Claim,
        *,
        claim_id: str,
        routes: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        if "retrieval" not in set(routes):
            return ()
        explicit_queries = _explicit_retrieval_queries(metadata, claim_id=claim_id)
        if explicit_queries:
            return explicit_queries
        return ({
            "query": claim.text,
            "claim_id": claim_id,
            "metadata": {
                "source": "claim_text",
                "claim_text": claim.text,
            },
        },)

    def _citation_checks_for_claim(
        self,
        claim: Claim,
        *,
        claim_id: str,
        routes: Sequence[str],
    ) -> tuple[dict[str, Any], ...]:
        if "citation" not in set(routes):
            return ()
        references = extract_citation_references(claim)
        return ({
            "claim_id": claim_id,
            "claim_text": claim.text,
            "references": tuple(references),
            "source": "claim_text" if references else "claim_metadata",
        },)


def estimate_verification_plan_cost(
    plan: ClaimVerificationPlan | Mapping[str, Any],
    *,
    route_cost_units: Mapping[str, Any] | None = None,
    tool_payload_cost_units: Mapping[str, Any] | None = None,
) -> VerificationPlanCostEstimate:
    """Estimate route and tool-payload cost from a verification plan.

    The units are relative, not wall-clock predictions. They are intended for
    release-gate comparisons and runtime-profile routing before any external
    verifier, retriever, calculator, or world-model adapter runs.
    """
    payload = _cost_estimate_payload(plan)
    run_verifier = _strict_bool(payload.get("run_verifier", False))
    claims = _as_sequence(payload.get("claims", ()))
    verify_claim_ids = tuple(_non_empty_strings(_as_sequence(payload.get("verify_claim_ids", ()))))
    skipped_claim_ids = tuple(_non_empty_strings(_as_sequence(payload.get("skipped_claim_ids", ()))))
    selected_claim_ids = set(verify_claim_ids)
    route_counts: dict[str, int] = {}
    tool_payload_counts: dict[str, int] = {}
    if run_verifier:
        for raw_hint in _as_sequence(payload.get("route_hints", ())):
            hint = raw_hint.to_dict() if isinstance(raw_hint, VerificationRouteHint) else raw_hint
            if not isinstance(hint, Mapping):
                continue
            claim_id = str(hint.get("claim_id", "")).strip()
            if selected_claim_ids and claim_id not in selected_claim_ids:
                continue
            for route in _as_sequence(hint.get("routes", ())):
                route_name = str(route).strip()
                if route_name:
                    route_counts[route_name] = route_counts.get(route_name, 0) + 1
        for key in DEFAULT_VERIFICATION_TOOL_PAYLOAD_COST_UNITS:
            tool_payload_counts[key] = _selected_payload_count(
                _as_sequence(payload.get(key, ())),
                selected_claim_ids=selected_claim_ids,
            )
    else:
        tool_payload_counts = {key: 0 for key in DEFAULT_VERIFICATION_TOOL_PAYLOAD_COST_UNITS}
    route_costs = _non_negative_float_mapping(
        DEFAULT_VERIFICATION_ROUTE_COST_UNITS if route_cost_units is None else route_cost_units,
        name="route_cost_units",
    )
    tool_payload_costs = _non_negative_float_mapping(
        DEFAULT_VERIFICATION_TOOL_PAYLOAD_COST_UNITS
        if tool_payload_cost_units is None
        else tool_payload_cost_units,
        name="tool_payload_cost_units",
    )
    route_cost = sum(
        count * route_costs.get(route_name, 1.0)
        for route_name, count in route_counts.items()
    )
    tool_payload_cost = sum(
        count * tool_payload_costs.get(name, 0.0)
        for name, count in tool_payload_counts.items()
    )
    return VerificationPlanCostEstimate(
        claim_count=len(claims),
        verify_claim_count=len(verify_claim_ids) if run_verifier else 0,
        skipped_claim_count=len(skipped_claim_ids) if run_verifier else len(claims),
        route_counts=route_counts,
        tool_payload_counts=tool_payload_counts,
        dependency_count=len(_as_sequence(payload.get("dependencies", ()))),
        estimated_route_attempts=sum(route_counts.values()),
        estimated_tool_payloads=sum(tool_payload_counts.values()),
        estimated_route_cost_units=route_cost,
        estimated_tool_payload_cost_units=tool_payload_cost,
    )


def budget_verification_plan(
    plan: ClaimVerificationPlan | Mapping[str, Any],
    policy: VerificationBudgetPolicy | Mapping[str, Any],
    *,
    route_cost_units: Mapping[str, Any] | None = None,
    tool_payload_cost_units: Mapping[str, Any] | None = None,
) -> ClaimVerificationPlan:
    """Select claims and verifier routes under a cost-aware budget policy."""
    plan_obj = _plan_obj(plan)
    policy_obj = _budget_policy_obj(policy)
    if not policy_obj.enabled() or not plan_obj.run_verifier:
        return plan_obj

    route_costs = _non_negative_float_mapping(
        DEFAULT_VERIFICATION_ROUTE_COST_UNITS if route_cost_units is None else route_cost_units,
        name="route_cost_units",
    )
    tool_payload_costs = _non_negative_float_mapping(
        DEFAULT_VERIFICATION_TOOL_PAYLOAD_COST_UNITS
        if tool_payload_cost_units is None
        else tool_payload_cost_units,
        name="tool_payload_cost_units",
    )
    original_cost = plan_obj.cost_estimate(
        route_cost_units=route_costs,
        tool_payload_cost_units=tool_payload_costs,
    )
    claim_ids = tuple(_claim_id(claim, index) for index, claim in enumerate(plan_obj.claims))
    original_verify_claim_ids = tuple(
        claim_id for claim_id in plan_obj.verify_claim_ids if claim_id in set(claim_ids)
    )
    ordered_claim_ids = _budget_ordered_claim_ids(plan_obj, policy_obj)
    if policy_obj.max_verify_claims is not None:
        ordered_claim_ids = ordered_claim_ids[: policy_obj.max_verify_claims]
    hint_by_claim = {hint.claim_id: hint for hint in plan_obj.route_hints}
    candidate_routes = {
        claim_id: _ordered_routes(hint.routes, policy_obj.route_priority)
        for claim_id, hint in hint_by_claim.items()
        if claim_id in ordered_claim_ids
    }
    selected_route_lists: dict[str, list[str]] = {claim_id: [] for claim_id in ordered_claim_ids}
    dropped_route_lists: dict[str, list[str]] = {claim_id: [] for claim_id in ordered_claim_ids}
    route_attempt_count = 0
    tool_payload_count = 0
    estimated_cost = 0.0
    route_budget_exhausted = False
    tool_payload_budget_exhausted = False
    estimated_cost_budget_exhausted = False

    route_depth = 0
    while True:
        any_remaining_route = False
        for claim_id in ordered_claim_ids:
            routes = candidate_routes.get(claim_id, ())
            if route_depth >= len(routes):
                continue
            any_remaining_route = True
            route = routes[route_depth]
            route_name = str(route).strip()
            if not route_name:
                continue
            payload_key = _payload_key_for_route(route_name)
            payload_count = (
                _payload_count_for_claim(plan_obj, claim_id=claim_id, payload_key=payload_key)
                if payload_key is not None
                else 0
            )
            additional_cost = route_costs.get(route_name, 1.0)
            if payload_key is not None:
                additional_cost += payload_count * tool_payload_costs.get(payload_key, 0.0)
            if (
                policy_obj.max_route_attempts is not None
                and route_attempt_count + 1 > policy_obj.max_route_attempts
            ):
                route_budget_exhausted = True
                dropped_route_lists[claim_id].append(route_name)
                continue
            if (
                policy_obj.max_tool_payloads is not None
                and tool_payload_count + payload_count > policy_obj.max_tool_payloads
            ):
                tool_payload_budget_exhausted = True
                dropped_route_lists[claim_id].append(route_name)
                continue
            if (
                policy_obj.max_estimated_cost_units is not None
                and estimated_cost + additional_cost > policy_obj.max_estimated_cost_units
            ):
                estimated_cost_budget_exhausted = True
                dropped_route_lists[claim_id].append(route_name)
                continue
            selected_route_lists[claim_id].append(route_name)
            route_attempt_count += 1
            tool_payload_count += payload_count
            estimated_cost += additional_cost
        if not any_remaining_route:
            break
        route_depth += 1

    selected_routes: dict[str, tuple[str, ...]] = {
        claim_id: tuple(routes)
        for claim_id, routes in selected_route_lists.items()
        if routes
    }
    dropped_routes: dict[str, tuple[str, ...]] = {
        claim_id: tuple(routes)
        for claim_id, routes in dropped_route_lists.items()
        if routes
    }

    final_verify_claim_ids = tuple(
        claim_id for claim_id in original_verify_claim_ids if claim_id in selected_routes
    )
    final_verify_claim_set = set(final_verify_claim_ids)
    final_skipped_claim_ids = tuple(claim_id for claim_id in claim_ids if claim_id not in final_verify_claim_set)
    final_route_hints = tuple(
        _budgeted_route_hint(
            hint_by_claim[claim_id],
            routes=selected_routes[claim_id],
            dropped_routes=dropped_routes.get(claim_id, ()),
        )
        for claim_id in final_verify_claim_ids
        if claim_id in hint_by_claim
    )
    filtered_payloads = _budgeted_tool_payloads(plan_obj, selected_routes=selected_routes)
    run_verifier = bool(final_verify_claim_ids and final_route_hints)
    reason = (
        "verification budget selected claims and routes"
        if run_verifier
        else "verification budget exhausted before selecting verifier routes"
    )
    budget_summary = {
        "enabled": True,
        "policy": policy_obj.to_dict(),
        "original_cost_estimate": original_cost.to_dict(),
        "selected_claim_ids": final_verify_claim_ids,
        "dropped_claim_ids": tuple(claim_id for claim_id in claim_ids if claim_id not in final_verify_claim_set),
        "selected_routes": {
            claim_id: routes
            for claim_id, routes in selected_routes.items()
            if claim_id in final_verify_claim_set
        },
        "dropped_routes": dropped_routes,
        "claim_budget_exhausted": (
            policy_obj.max_verify_claims is not None
            and len(original_verify_claim_ids) > len(ordered_claim_ids)
        ),
        "route_budget_exhausted": route_budget_exhausted,
        "tool_payload_budget_exhausted": tool_payload_budget_exhausted,
        "estimated_cost_budget_exhausted": estimated_cost_budget_exhausted,
    }
    budgeted_plan = ClaimVerificationPlan(
        run_verifier=run_verifier,
        reason=reason,
        verification_scope="budgeted" if run_verifier else "none",
        claims=plan_obj.claims,
        verify_claim_ids=final_verify_claim_ids,
        skipped_claim_ids=final_skipped_claim_ids,
        triggered_claim_ids=plan_obj.triggered_claim_ids,
        triggered_features=plan_obj.triggered_features,
        triggered_metadata=plan_obj.triggered_metadata,
        route_hints=final_route_hints,
        retrieval_queries=filtered_payloads["retrieval_queries"],
        citation_checks=filtered_payloads["citation_checks"],
        calculation_checks=filtered_payloads["calculation_checks"],
        state_checks=filtered_payloads["state_checks"],
        world_model_checks=filtered_payloads["world_model_checks"],
        dependencies=plan_obj.dependencies,
        budget=budget_summary,
    )
    selected_cost = budgeted_plan.cost_estimate(
        route_cost_units=route_costs,
        tool_payload_cost_units=tool_payload_costs,
    )
    budget = dict(budgeted_plan.budget)
    budget["selected_cost_estimate"] = selected_cost.to_dict()
    return ClaimVerificationPlan(
        run_verifier=budgeted_plan.run_verifier,
        reason=budgeted_plan.reason,
        verification_scope=budgeted_plan.verification_scope,
        claims=budgeted_plan.claims,
        verify_claim_ids=budgeted_plan.verify_claim_ids,
        skipped_claim_ids=budgeted_plan.skipped_claim_ids,
        triggered_claim_ids=budgeted_plan.triggered_claim_ids,
        triggered_features=budgeted_plan.triggered_features,
        triggered_metadata=budgeted_plan.triggered_metadata,
        route_hints=budgeted_plan.route_hints,
        retrieval_queries=budgeted_plan.retrieval_queries,
        citation_checks=budgeted_plan.citation_checks,
        calculation_checks=budgeted_plan.calculation_checks,
        state_checks=budgeted_plan.state_checks,
        world_model_checks=budgeted_plan.world_model_checks,
        dependencies=budgeted_plan.dependencies,
        budget=budget,
    )


def _plan_obj(value: ClaimVerificationPlan | Mapping[str, Any]) -> ClaimVerificationPlan:
    if isinstance(value, ClaimVerificationPlan):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("verification plan must be a ClaimVerificationPlan or mapping.")
    return ClaimVerificationPlan(
        run_verifier=_strict_bool(value.get("run_verifier", False)),
        reason=str(value.get("reason", "")),
        verification_scope=None if value.get("verification_scope") is None else str(value.get("verification_scope")),
        claims=tuple(_as_sequence(value.get("claims", ()))),
        verify_claim_ids=tuple(_as_sequence(value.get("verify_claim_ids", ()))),
        skipped_claim_ids=tuple(_as_sequence(value.get("skipped_claim_ids", ()))),
        triggered_claim_ids=tuple(_as_sequence(value.get("triggered_claim_ids", ()))),
        triggered_features=_mapping_of_sequences(value.get("triggered_features", {})),
        triggered_metadata=_mapping_of_sequences(value.get("triggered_metadata", {})),
        route_hints=tuple(_as_sequence(value.get("route_hints", ()))),
        retrieval_queries=tuple(_as_sequence(value.get("retrieval_queries", ()))),
        citation_checks=tuple(_as_sequence(value.get("citation_checks", ()))),
        calculation_checks=tuple(_as_sequence(value.get("calculation_checks", ()))),
        state_checks=tuple(_as_sequence(value.get("state_checks", ()))),
        world_model_checks=tuple(_as_sequence(value.get("world_model_checks", ()))),
        dependencies=tuple(_as_sequence(value.get("dependencies", ()))),
        budget=dict(value.get("budget", {})),
    )


def _budget_policy_obj(value: VerificationBudgetPolicy | Mapping[str, Any]) -> VerificationBudgetPolicy:
    if isinstance(value, VerificationBudgetPolicy):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("verification budget policy must be a VerificationBudgetPolicy or mapping.")
    return VerificationBudgetPolicy.from_mapping(value)


def _budget_ordered_claim_ids(
    plan: ClaimVerificationPlan,
    policy: VerificationBudgetPolicy,
) -> tuple[str, ...]:
    claim_ids = tuple(_claim_id(claim, index) for index, claim in enumerate(plan.claims))
    selected = [claim_id for claim_id in plan.verify_claim_ids if claim_id in set(claim_ids)]
    triggered = set(plan.triggered_claim_ids)
    priority_features = set(policy.priority_feature_flags)
    priority_metadata = set(policy.priority_metadata_keys)
    route_priority = {route: len(policy.route_priority) - index for index, route in enumerate(policy.route_priority)}
    hint_by_claim = {hint.claim_id: hint for hint in plan.route_hints}
    order_index = {claim_id: index for index, claim_id in enumerate(claim_ids)}

    def score(claim_id: str) -> tuple[int, int]:
        value = 0
        if policy.preserve_triggered_claims and claim_id in triggered:
            value += 1000
        value += 25 * len(set(plan.triggered_features.get(claim_id, ())) & priority_features)
        value += 25 * len(set(plan.triggered_metadata.get(claim_id, ())) & priority_metadata)
        hint = hint_by_claim.get(claim_id)
        if hint is not None:
            value += max((route_priority.get(route, 0) for route in hint.routes), default=0)
        return value, -order_index.get(claim_id, 0)

    return tuple(sorted(selected, key=score, reverse=True))


def _ordered_routes(routes: Sequence[str], priority: Sequence[str]) -> tuple[str, ...]:
    priority_index = {route: index for index, route in enumerate(priority)}
    original_index = {route: index for index, route in enumerate(routes)}
    return tuple(
        sorted(
            tuple(_non_empty_strings(routes)),
            key=lambda route: (priority_index.get(route, len(priority_index)), original_index.get(route, 0)),
        )
    )


def _budgeted_route_hint(
    hint: VerificationRouteHint,
    *,
    routes: Sequence[str],
    dropped_routes: Sequence[str],
) -> VerificationRouteHint:
    metadata = dict(hint.metadata)
    metadata["verification_budget_selected_routes"] = tuple(routes)
    if dropped_routes:
        metadata["verification_budget_dropped_routes"] = tuple(dropped_routes)
    return VerificationRouteHint(
        claim_id=hint.claim_id,
        routes=tuple(routes),
        reasons=hint.reasons,
        metadata=metadata,
    )


def _payload_key_for_route(route: str) -> str | None:
    return {
        "retrieval": "retrieval_queries",
        "citation": "citation_checks",
        "calculator": "calculation_checks",
        "state": "state_checks",
        "world_model": "world_model_checks",
    }.get(route)


def _payload_count_for_claim(
    plan: ClaimVerificationPlan,
    *,
    claim_id: str,
    payload_key: str | None,
) -> int:
    if payload_key is None:
        return 0
    return sum(
        1
        for item in _payload_sequence_for_key(plan, payload_key)
        if _payload_claim_id(item) == claim_id
    )


def _budgeted_tool_payloads(
    plan: ClaimVerificationPlan,
    *,
    selected_routes: Mapping[str, Sequence[str]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    payloads: dict[str, tuple[dict[str, Any], ...]] = {}
    for payload_key in DEFAULT_VERIFICATION_TOOL_PAYLOAD_COST_UNITS:
        route = _route_for_payload_key(payload_key)
        kept: list[dict[str, Any]] = []
        for item in _payload_sequence_for_key(plan, payload_key):
            claim_id = _payload_claim_id(item)
            if claim_id is None:
                continue
            if route in set(selected_routes.get(claim_id, ())):
                kept.append(dict(item))
        payloads[payload_key] = tuple(kept)
    return payloads


def _payload_sequence_for_key(
    plan: ClaimVerificationPlan,
    payload_key: str,
) -> tuple[Mapping[str, Any], ...]:
    value = getattr(plan, payload_key)
    return tuple(item for item in value if isinstance(item, Mapping))


def _payload_claim_id(item: Mapping[str, Any]) -> str | None:
    raw_claim_id = item.get("claim_id")
    return None if raw_claim_id is None else str(raw_claim_id)


def _route_for_payload_key(payload_key: str) -> str:
    return {
        "retrieval_queries": "retrieval",
        "citation_checks": "citation",
        "calculation_checks": "calculator",
        "state_checks": "state",
        "world_model_checks": "world_model",
    }[payload_key]


def _cost_estimate_payload(plan: ClaimVerificationPlan | Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, ClaimVerificationPlan):
        return dict(plan)
    return {
        "run_verifier": plan.run_verifier,
        "claims": plan.claims,
        "verify_claim_ids": plan.verify_claim_ids,
        "skipped_claim_ids": plan.skipped_claim_ids,
        "route_hints": plan.route_hints,
        "retrieval_queries": plan.retrieval_queries,
        "citation_checks": plan.citation_checks,
        "calculation_checks": plan.calculation_checks,
        "state_checks": plan.state_checks,
        "world_model_checks": plan.world_model_checks,
        "dependencies": plan.dependencies,
    }


def _verification_scope(value: str | None, *, run_verifier: bool) -> str:
    if value is None:
        scope = "all" if run_verifier else "none"
    else:
        scope = str(value).strip().lower()
    if scope not in {"all", "triggered", "budgeted", "none"}:
        raise ValueError("verification_scope must be one of: all, triggered, budgeted, none")
    if run_verifier and scope == "none":
        raise ValueError("verification_scope cannot be 'none' when run_verifier is true")
    if not run_verifier and scope != "none":
        raise ValueError("verification_scope must be 'none' when run_verifier is false")
    return scope


def _route_hint_obj(value: VerificationRouteHint | Mapping[str, Any]) -> VerificationRouteHint:
    if isinstance(value, VerificationRouteHint):
        return value
    return VerificationRouteHint(
        claim_id=str(value["claim_id"]),
        routes=tuple(_as_sequence(value.get("routes", ()))),
        reasons=tuple(_as_sequence(value.get("reasons", ()))),
        metadata=dict(value.get("metadata", {})),
    )


def _dependency_obj(value: ClaimDependency | Mapping[str, Any]) -> ClaimDependency:
    if isinstance(value, ClaimDependency):
        return value
    return ClaimDependency.from_mapping(value)


def _coerce_claims(values: Sequence[Claim | Mapping[str, Any]]) -> tuple[Claim, ...]:
    return tuple(_coerce_claim(value) for value in values)


def _coerce_claim(value: Claim | Mapping[str, Any]) -> Claim:
    if isinstance(value, Claim):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("claims must be Claim objects or JSON-like mappings.")
    text = value.get("text")
    if text is None:
        raise ValueError("claim mapping must contain text.")
    raw_span = value.get("span")
    span = None
    if raw_span is not None:
        if not isinstance(raw_span, Sequence) or isinstance(raw_span, (str, bytes)) or len(raw_span) != 2:
            raise ValueError("claim span must be a two-item sequence.")
        span = (int(raw_span[0]), int(raw_span[1]))
    raw_claim_id = value.get("claim_id")
    return Claim(
        text=str(text),
        claim_id=None if raw_claim_id is None else str(raw_claim_id),
        span=span,
        metadata=dict(value.get("metadata", {})),
    )


def _claim_id(claim: Claim, index: int) -> str:
    return claim.claim_id or f"c{index + 1}"


def _claim_metadata(claim: Claim) -> Mapping[str, Any]:
    return claim.metadata if isinstance(claim.metadata, Mapping) else {}


def _claim_features(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    features = metadata.get("features", {})
    return features if isinstance(features, Mapping) else {}


def _claim_to_dict(claim: Claim, *, fallback_id: str) -> dict[str, Any]:
    return {
        "text": claim.text,
        "claim_id": claim.claim_id or fallback_id,
        "span": claim.span,
        "metadata": to_jsonable(dict(_claim_metadata(claim))),
    }


def _explicit_routes(metadata: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    routes: list[str] = []
    for key in keys:
        current = _metadata_path_value(metadata, key)
        if current is None:
            continue
        for route in _route_values(current):
            _append_unique(routes, route)
    return tuple(routes)


def _route_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(_non_empty_strings(value.split(",")))
    if isinstance(value, Mapping):
        raw_route = value.get("route", value.get("name", value.get("tool")))
        return () if raw_route is None else _route_values(raw_route)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        routes: list[str] = []
        for item in value:
            for route in _route_values(item):
                _append_unique(routes, route)
        return tuple(routes)
    return ()


def _explicit_retrieval_queries(metadata: Mapping[str, Any], *, claim_id: str) -> tuple[dict[str, Any], ...]:
    raw_items: list[Any] = []
    for key in ("retrieval_query", "retrieval_queries"):
        value = _metadata_path_value(metadata, key)
        if value is not None:
            raw_items.extend(_as_sequence(value))
    queries: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, str):
            query = item.strip()
            if query:
                queries.append({
                    "query": query,
                    "claim_id": claim_id,
                    "metadata": {"source": "metadata.retrieval_query"},
                })
            continue
        if isinstance(item, Mapping):
            query_text = item.get("query", item.get("text"))
            if query_text is None or not str(query_text).strip():
                continue
            raw_claim_id = item.get("claim_id", claim_id)
            metadata_payload = dict(item.get("metadata", {}))
            for key, value in item.items():
                metadata_key = str(key)
                if (
                    metadata_key not in {"query", "text", "claim_id", "metadata"}
                    and metadata_key not in metadata_payload
                ):
                    metadata_payload[metadata_key] = value
            metadata_payload.setdefault("source", "metadata.retrieval_query")
            queries.append({
                "query": str(query_text).strip(),
                "claim_id": None if raw_claim_id is None else str(raw_claim_id),
                "metadata": to_jsonable(metadata_payload),
            })
    return tuple(queries)


def _metadata_tool_payloads(
    claim_id: str,
    metadata: Mapping[str, Any],
    keys: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    payloads: list[dict[str, Any]] = []
    for key in keys:
        value = _metadata_path_value(metadata, key)
        if value is None:
            continue
        for item in _as_sequence(value):
            if isinstance(item, Mapping):
                payload = {"claim_id": claim_id, **dict(item)}
            else:
                payload = {"claim_id": claim_id, "value": item}
            payload.setdefault("source", f"metadata.{key}")
            payloads.append(to_jsonable(payload))
    return tuple(payloads)


def _metadata_path_value(metadata: Mapping[str, Any], path: str) -> Any:
    current: Any = metadata
    for part in str(path).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _feature_enabled(features: Mapping[str, Any], name: str) -> bool:
    return name in enabled_feature_names(features, (name,))


def _normalize_string_sequence_mapping(value: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    return {
        str(key): tuple(_non_empty_strings(items))
        for key, items in value.items()
    }


def _mapping_of_sequences(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): tuple(_non_empty_strings(_as_sequence(items)))
        for key, items in value.items()
    }


def _jsonable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("plan payload entries must be mappings.")
    return dict(to_jsonable(dict(value)))


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _non_empty_strings(values: Sequence[Any]) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    return tuple(value for value in normalized if value)


def _selected_payload_count(values: Sequence[Any], *, selected_claim_ids: set[str]) -> int:
    if not selected_claim_ids:
        return len(tuple(values))
    count = 0
    for item in values:
        if not isinstance(item, Mapping):
            count += 1
            continue
        raw_claim_id = item.get("claim_id")
        if raw_claim_id is not None and str(raw_claim_id) in selected_claim_ids:
            count += 1
    return count


def _append_unique(values: list[str], value: str) -> None:
    value = str(value).strip()
    if value and value not in values:
        values.append(value)


def _non_negative_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def _optional_non_negative_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, name=name)


def _non_negative_float(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative finite number.") from exc
    if parsed < 0.0 or not (parsed == parsed) or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return parsed


def _optional_non_negative_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _non_negative_float(value, name=name)


def _non_negative_int_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return {
        str(key): _non_negative_int(item, name=f"{name}.{key}")
        for key, item in value.items()
    }


def _non_negative_float_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return {
        str(key): _non_negative_float(item, name=f"{name}.{key}")
        for key, item in value.items()
    }


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError("boolean configuration values must be bool or boolean string.")
