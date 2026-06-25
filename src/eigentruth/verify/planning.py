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
from eigentruth.verify.claims import ClaimExtractor, extract_claims
from eigentruth.verify.coherence import ClaimDependency, infer_claim_dependencies
from eigentruth.verify.features import enabled_feature_names, metadata_path_enabled
from eigentruth.verify.protocols import Claim

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
DEFAULT_CALCULATOR_METADATA_KEYS = ("calculation",)
DEFAULT_STATE_METADATA_KEYS = ("state_check", "state_checks")
DEFAULT_WORLD_MODEL_METADATA_KEYS = (
    "world_model_check",
    "world_model_checks",
    "state_transition",
)
DEFAULT_EXPLICIT_ROUTE_METADATA_KEYS = ("route_hints", "routes")


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
    calculation_checks: Sequence[Mapping[str, Any]] = ()
    state_checks: Sequence[Mapping[str, Any]] = ()
    world_model_checks: Sequence[Mapping[str, Any]] = ()
    dependencies: Sequence[ClaimDependency | Mapping[str, Any]] = ()

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
            "calculation_checks": tuple(dict(item) for item in self.calculation_checks),
            "state_checks": tuple(dict(item) for item in self.state_checks),
            "world_model_checks": tuple(dict(item) for item in self.world_model_checks),
            "dependencies": tuple(item.to_dict() for item in self.dependencies),
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
    calculator_metadata_keys: Sequence[str] = DEFAULT_CALCULATOR_METADATA_KEYS
    state_metadata_keys: Sequence[str] = DEFAULT_STATE_METADATA_KEYS
    world_model_metadata_keys: Sequence[str] = DEFAULT_WORLD_MODEL_METADATA_KEYS
    explicit_route_metadata_keys: Sequence[str] = DEFAULT_EXPLICIT_ROUTE_METADATA_KEYS
    infer_dependencies: bool = True

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
        object.__setattr__(self, "infer_dependencies", _strict_bool(self.infer_dependencies))

    def extract(self, text: str) -> tuple[Claim, ...]:
        """Extract claims with the configured extractor."""
        return extract_claims(text, min_chars=self.min_chars, extractor=self.extractor)

    def plan(
        self,
        claims_or_text: str | Sequence[Claim | Mapping[str, Any]],
        *,
        context: Mapping[str, Any] | None = None,
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
        return ClaimVerificationPlan(
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
            calculation_checks=tuple(calculation_checks),
            state_checks=tuple(state_checks),
            world_model_checks=tuple(world_model_checks),
            dependencies=dependencies,
        )

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
        for key in self.retrieval_metadata_keys:
            if metadata_path_enabled(metadata, key):
                _append_unique(routes, "retrieval")
                reasons.append(f"metadata:{key}")
        for feature in enabled_feature_names(features, self.retrieval_feature_flags):
            _append_unique(routes, "retrieval")
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


def _verification_scope(value: str | None, *, run_verifier: bool) -> str:
    if value is None:
        scope = "all" if run_verifier else "none"
    else:
        scope = str(value).strip().lower()
    if scope not in {"all", "triggered", "none"}:
        raise ValueError("verification_scope must be one of: all, triggered, none")
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


def _append_unique(values: list[str], value: str) -> None:
    value = str(value).strip()
    if value and value not in values:
        values.append(value)


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
