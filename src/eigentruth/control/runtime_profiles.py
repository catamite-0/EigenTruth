"""Named runtime profiles for release-gate defaults."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import exp
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from eigentruth.control.policy import ControlAction, RiskDecision, RiskLevel
from eigentruth.verify.features import enabled_feature_names as _enabled_claim_feature_names
from eigentruth.verify.features import metadata_path_enabled, normalized_feature_flags
from eigentruth.verify.planning import ClaimVerificationPlan, estimate_verification_plan_cost

_DEFAULT_KEYS = frozenset({
    "inside_trigger_budget_policy",
    "max_inside_sample_count_ratio",
    "max_inside_generation_seconds_ratio",
    "max_mean_attempted_route_count",
    "max_retrieval_use_rate",
})
_CONTROL_DEFAULT_KEYS = frozenset({
    "staged_verification",
    "stage_verify_risk_levels",
    "stage_verify_actions",
    "stage_verify_claim_feature_flags",
    "stage_verify_claim_metadata_keys",
    "stage_verify_triggered_claims_only",
    "stage_fail_closed_on_skip",
    "max_verifier_route_attempts",
})
_DEFAULT_SENSITIVE_CLAIM_FEATURE_FLAGS = (
    "has_number",
    "has_citation",
    "is_time_sensitive",
    "has_calculation",
)
_DEFAULT_SENSITIVE_CLAIM_METADATA_KEYS = ("requires_verification",)
_DEFAULT_HIGH_RISK_PROMPT_FEATURE_FLAGS = (
    "requires_retrieval",
    "is_time_sensitive",
    "requires_domain_state",
)
_DEFAULT_MEDIUM_RISK_PROMPT_FEATURE_FLAGS = (
    "has_number",
    "has_citation",
    "has_calculation",
)
_DEFAULT_HIGH_RISK_PROMPT_METADATA_KEYS = (
    "requires_verification",
    "requires_retrieval",
    "requires_current_facts",
    "requires_domain_state",
)
_DEFAULT_MEDIUM_RISK_PROMPT_METADATA_KEYS = (
    "sensitive",
    "requires_calculation",
)
_DEFAULT_SOFT_RISK_FEATURE_WEIGHTS: Mapping[str, float] = MappingProxyType({
    "requires_retrieval": 2.0,
    "is_time_sensitive": 1.8,
    "requires_domain_state": 1.7,
    "has_calculation": 1.1,
    "has_citation": 1.0,
    "has_number": 0.7,
    "asks_exact_fact": 0.8,
    "has_named_entity_hint": 0.6,
    "is_multi_part": 0.4,
    "long_prompt": 0.3,
    "has_question": 0.2,
})
_DEFAULT_SOFT_RISK_METADATA_WEIGHTS: Mapping[str, float] = MappingProxyType({
    "requires_verification": 1.8,
    "requires_retrieval": 2.0,
    "requires_current_facts": 2.0,
    "requires_domain_state": 1.8,
    "sensitive": 1.0,
    "requires_calculation": 1.0,
})
_TIME_SENSITIVE_RE = re.compile(
    r"\b("
    r"as of|current|currently|latest|newest|recent|today|tonight|tomorrow|"
    r"yesterday|now|news|price|prices|weather|schedule|score|scores|"
    r"president|ceo|law|regulation|deadline|release|version"
    r")\b",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(r"(https?://|www\.|\bdoi:|\[[A-Za-z0-9][A-Za-z0-9, .:-]{0,40}\])", re.IGNORECASE)
_CALCULATION_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*[-+*/=]\s*\d+)|\b("
    r"calculate|compute|sum|total|average|mean|percent|percentage|ratio|rate"
    r")\b",
    re.IGNORECASE,
)
_DOMAIN_STATE_RE = re.compile(
    r"\b("
    r"account|balance|budget|contract|customer|database|inventory|invoice|"
    r"order|permission|policy|quota|stock|transaction|workflow"
    r")\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(r"\b(who|what|when|where|which|why|how)\b|\?", re.IGNORECASE)
_EXACT_FACT_RE = re.compile(
    r"\b("
    r"exact|exactly|specific|specifically|identify|name|who|when|where|"
    r"founded|invented|published|released|reported|ranked"
    r")\b",
    re.IGNORECASE,
)
_NAMED_ENTITY_HINT_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9_-]{2,}|[A-Z]{2,})\b")


@dataclass(frozen=True)
class RuntimeProfile:
    """Named set of release-gate defaults.

    Profiles only fill unset values. Explicit CLI or API parameters remain the
    source of truth for a run.
    """

    name: str
    description: str
    defaults: Mapping[str, Any]
    control_defaults: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.strip().lower().replace("-", "_")
        if not name:
            raise ValueError("runtime profile name must not be empty")
        defaults = dict(self.defaults)
        unknown = sorted(set(defaults) - _DEFAULT_KEYS)
        if unknown:
            raise ValueError(f"unknown runtime profile defaults: {', '.join(unknown)}")
        control_defaults = dict(self.control_defaults)
        unknown_control = sorted(set(control_defaults) - _CONTROL_DEFAULT_KEYS)
        if unknown_control:
            raise ValueError(f"unknown runtime profile control defaults: {', '.join(unknown_control)}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "defaults", MappingProxyType(defaults))
        object.__setattr__(self, "control_defaults", MappingProxyType(control_defaults))

    def apply_defaults(self, values: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return values with this profile's defaults filled into unset keys."""
        merged = dict(values)
        applied = {}
        for key, value in self.defaults.items():
            if merged.get(key) is None:
                merged[key] = value
                applied[key] = value
        return merged, applied

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "description": self.description,
            "defaults": dict(self.defaults),
            "control_defaults": dict(self.control_defaults),
        }


@dataclass(frozen=True)
class RuntimeProfileSelection:
    """Auditable result of choosing a product runtime profile for one request."""

    selected_profile: str
    reason: str
    diagnostic_risk_level: str | None = None
    diagnostic_action: str | None = None
    triggered_claim_ids: Sequence[str] = ()
    triggered_features: Mapping[str, Sequence[str]] = field(default_factory=dict)
    triggered_metadata: Mapping[str, Sequence[str]] = field(default_factory=dict)
    verification_plan_cost: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        selected_profile = _normalize_profile_name(self.selected_profile)
        if selected_profile not in RUNTIME_PROFILES:
            choices = ", ".join(RUNTIME_PROFILE_NAMES)
            raise ValueError(f"selected_profile must be one of: {choices}")
        object.__setattr__(self, "selected_profile", selected_profile)
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(
            self,
            "diagnostic_risk_level",
            None if self.diagnostic_risk_level is None else str(self.diagnostic_risk_level),
        )
        object.__setattr__(
            self,
            "diagnostic_action",
            None if self.diagnostic_action is None else str(self.diagnostic_action),
        )
        object.__setattr__(
            self,
            "triggered_claim_ids",
            tuple(str(item) for item in self.triggered_claim_ids),
        )
        object.__setattr__(
            self,
            "triggered_features",
            _string_sequence_mapping(self.triggered_features),
        )
        object.__setattr__(
            self,
            "triggered_metadata",
            _string_sequence_mapping(self.triggered_metadata),
        )
        object.__setattr__(
            self,
            "verification_plan_cost",
            None if self.verification_plan_cost is None else dict(self.verification_plan_cost),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "selected_profile": self.selected_profile,
            "reason": self.reason,
            "diagnostic_risk_level": self.diagnostic_risk_level,
            "diagnostic_action": self.diagnostic_action,
            "triggered_claim_ids": tuple(self.triggered_claim_ids),
            "triggered_features": {
                claim_id: tuple(features)
                for claim_id, features in self.triggered_features.items()
            },
            "triggered_metadata": {
                claim_id: tuple(keys)
                for claim_id, keys in self.triggered_metadata.items()
            },
            "verification_plan_cost": self.verification_plan_cost,
        }


@dataclass(frozen=True)
class RuntimeProfileSelectorPolicy:
    """Configurable request-time policy for automatic runtime profile selection."""

    low_risk_profile: str = "latency"
    default_profile: str = "balanced"
    high_risk_profile: str = "audit"
    sensitive_profile: str = "audit"
    low_risk_levels: Sequence[str] = (RiskLevel.LOW.value,)
    low_risk_actions: Sequence[str] = (ControlAction.ACCEPT.value,)
    high_risk_levels: Sequence[str] = (RiskLevel.HIGH.value, RiskLevel.UNKNOWN.value)
    high_risk_actions: Sequence[str] = (
        ControlAction.ABSTAIN.value,
        ControlAction.CLARIFY.value,
        ControlAction.REWRITE.value,
        ControlAction.STEER_REGENERATE.value,
        ControlAction.EXECUTE_TOOL.value,
    )
    sensitive_claim_feature_flags: Sequence[str] = _DEFAULT_SENSITIVE_CLAIM_FEATURE_FLAGS
    sensitive_claim_metadata_keys: Sequence[str] = _DEFAULT_SENSITIVE_CLAIM_METADATA_KEYS
    plan_balanced_cost_threshold: float | None = 2.0
    plan_audit_cost_threshold: float | None = 5.0

    def __post_init__(self) -> None:
        low_risk_profile = _normalize_profile_name(self.low_risk_profile)
        default_profile = _normalize_profile_name(self.default_profile)
        high_risk_profile = _normalize_profile_name(self.high_risk_profile)
        sensitive_profile = _normalize_profile_name(self.sensitive_profile)
        _validate_profile_names(low_risk_profile, default_profile, high_risk_profile, sensitive_profile)
        object.__setattr__(self, "low_risk_profile", low_risk_profile)
        object.__setattr__(self, "default_profile", default_profile)
        object.__setattr__(self, "high_risk_profile", high_risk_profile)
        object.__setattr__(self, "sensitive_profile", sensitive_profile)
        object.__setattr__(
            self,
            "low_risk_levels",
            _risk_level_values(self.low_risk_levels, field_name="low_risk_levels"),
        )
        object.__setattr__(
            self,
            "low_risk_actions",
            _control_action_values(self.low_risk_actions, field_name="low_risk_actions"),
        )
        object.__setattr__(
            self,
            "high_risk_levels",
            _risk_level_values(self.high_risk_levels, field_name="high_risk_levels"),
        )
        object.__setattr__(
            self,
            "high_risk_actions",
            _control_action_values(self.high_risk_actions, field_name="high_risk_actions"),
        )
        object.__setattr__(
            self,
            "sensitive_claim_feature_flags",
            _non_empty_string_tuple(
                self.sensitive_claim_feature_flags,
                field_name="sensitive_claim_feature_flags",
            ),
        )
        object.__setattr__(
            self,
            "sensitive_claim_metadata_keys",
            _non_empty_string_tuple(
                self.sensitive_claim_metadata_keys,
                field_name="sensitive_claim_metadata_keys",
            ),
        )
        object.__setattr__(
            self,
            "plan_balanced_cost_threshold",
            _optional_non_negative_float(
                self.plan_balanced_cost_threshold,
                field_name="plan_balanced_cost_threshold",
            ),
        )
        object.__setattr__(
            self,
            "plan_audit_cost_threshold",
            _optional_non_negative_float(
                self.plan_audit_cost_threshold,
                field_name="plan_audit_cost_threshold",
            ),
        )
        if (
            self.plan_balanced_cost_threshold is not None
            and self.plan_audit_cost_threshold is not None
            and self.plan_balanced_cost_threshold > self.plan_audit_cost_threshold
        ):
            raise ValueError("plan_balanced_cost_threshold must be <= plan_audit_cost_threshold")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RuntimeProfileSelectorPolicy":
        """Build a selector policy from a JSON-like mapping."""
        return cls(
            low_risk_profile=payload.get("low_risk_profile", cls.low_risk_profile),
            default_profile=payload.get("default_profile", cls.default_profile),
            high_risk_profile=payload.get("high_risk_profile", cls.high_risk_profile),
            sensitive_profile=payload.get("sensitive_profile", cls.sensitive_profile),
            low_risk_levels=_as_sequence(payload.get("low_risk_levels", cls.low_risk_levels)),
            low_risk_actions=_as_sequence(payload.get("low_risk_actions", cls.low_risk_actions)),
            high_risk_levels=_as_sequence(payload.get("high_risk_levels", cls.high_risk_levels)),
            high_risk_actions=_as_sequence(payload.get("high_risk_actions", cls.high_risk_actions)),
            sensitive_claim_feature_flags=_as_sequence(
                payload.get("sensitive_claim_feature_flags", cls.sensitive_claim_feature_flags)
            ),
            sensitive_claim_metadata_keys=_as_sequence(
                payload.get("sensitive_claim_metadata_keys", cls.sensitive_claim_metadata_keys)
            ),
            plan_balanced_cost_threshold=payload.get(
                "plan_balanced_cost_threshold",
                cls.plan_balanced_cost_threshold,
            ),
            plan_audit_cost_threshold=payload.get(
                "plan_audit_cost_threshold",
                cls.plan_audit_cost_threshold,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "low_risk_profile": self.low_risk_profile,
            "default_profile": self.default_profile,
            "high_risk_profile": self.high_risk_profile,
            "sensitive_profile": self.sensitive_profile,
            "low_risk_levels": tuple(self.low_risk_levels),
            "low_risk_actions": tuple(self.low_risk_actions),
            "high_risk_levels": tuple(self.high_risk_levels),
            "high_risk_actions": tuple(self.high_risk_actions),
            "sensitive_claim_feature_flags": tuple(self.sensitive_claim_feature_flags),
            "sensitive_claim_metadata_keys": tuple(self.sensitive_claim_metadata_keys),
            "plan_balanced_cost_threshold": self.plan_balanced_cost_threshold,
            "plan_audit_cost_threshold": self.plan_audit_cost_threshold,
        }


@dataclass(frozen=True)
class SoftPreGenerationRiskEstimate:
    """Dependency-free soft risk estimate computed before generation."""

    score: float
    probability: float
    risk_level: str
    feature_contributions: Mapping[str, float] = field(default_factory=dict)
    metadata_contributions: Mapping[str, float] = field(default_factory=dict)
    medium_threshold: float = 0.45
    high_threshold: float = 0.75

    def __post_init__(self) -> None:
        score = _finite_float(self.score, field_name="score")
        probability = _probability_float(self.probability, field_name="probability")
        risk_level = RiskLevel(str(self.risk_level)).value
        medium_threshold = _probability_float(self.medium_threshold, field_name="medium_threshold")
        high_threshold = _probability_float(self.high_threshold, field_name="high_threshold")
        if medium_threshold > high_threshold:
            raise ValueError("medium_threshold must be <= high_threshold")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "risk_level", risk_level)
        object.__setattr__(
            self,
            "feature_contributions",
            MappingProxyType(_float_mapping(self.feature_contributions, field_name="feature_contributions")),
        )
        object.__setattr__(
            self,
            "metadata_contributions",
            MappingProxyType(_float_mapping(self.metadata_contributions, field_name="metadata_contributions")),
        )
        object.__setattr__(self, "medium_threshold", medium_threshold)
        object.__setattr__(self, "high_threshold", high_threshold)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "score": self.score,
            "probability": self.probability,
            "risk_level": self.risk_level,
            "feature_contributions": dict(self.feature_contributions),
            "metadata_contributions": dict(self.metadata_contributions),
            "medium_threshold": self.medium_threshold,
            "high_threshold": self.high_threshold,
        }


@dataclass(frozen=True)
class SoftPreGenerationRiskConfig:
    """Config for prompt-level soft risk scoring before generation.

    The scorer is intentionally not a trained probe. It is a transparent local
    approximation that lets product code record and optionally route on a soft
    risk probability while stronger hidden-state probes remain an external or
    future adapter.
    """

    enabled: bool = True
    route_on_soft_risk: bool = False
    bias: float = -2.0
    medium_risk_probability_threshold: float = 0.45
    high_risk_probability_threshold: float = 0.75
    feature_weights: Mapping[str, float] = field(default_factory=lambda: dict(_DEFAULT_SOFT_RISK_FEATURE_WEIGHTS))
    metadata_weights: Mapping[str, float] = field(default_factory=lambda: dict(_DEFAULT_SOFT_RISK_METADATA_WEIGHTS))

    def __post_init__(self) -> None:
        enabled = _strict_bool(self.enabled, field_name="enabled")
        route_on_soft_risk = _strict_bool(self.route_on_soft_risk, field_name="route_on_soft_risk")
        bias = _finite_float(self.bias, field_name="bias")
        medium_threshold = _probability_float(
            self.medium_risk_probability_threshold,
            field_name="medium_risk_probability_threshold",
        )
        high_threshold = _probability_float(
            self.high_risk_probability_threshold,
            field_name="high_risk_probability_threshold",
        )
        if medium_threshold > high_threshold:
            raise ValueError("medium_risk_probability_threshold must be <= high_risk_probability_threshold")
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "route_on_soft_risk", route_on_soft_risk)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "medium_risk_probability_threshold", medium_threshold)
        object.__setattr__(self, "high_risk_probability_threshold", high_threshold)
        object.__setattr__(
            self,
            "feature_weights",
            MappingProxyType(_float_mapping(self.feature_weights, field_name="feature_weights")),
        )
        object.__setattr__(
            self,
            "metadata_weights",
            MappingProxyType(_float_mapping(self.metadata_weights, field_name="metadata_weights")),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SoftPreGenerationRiskConfig":
        """Build a soft pre-generation risk config from a JSON-like mapping."""
        return cls(
            enabled=payload.get("enabled", cls.enabled),
            route_on_soft_risk=payload.get("route_on_soft_risk", cls.route_on_soft_risk),
            bias=payload.get("bias", cls.bias),
            medium_risk_probability_threshold=payload.get(
                "medium_risk_probability_threshold",
                cls.medium_risk_probability_threshold,
            ),
            high_risk_probability_threshold=payload.get(
                "high_risk_probability_threshold",
                cls.high_risk_probability_threshold,
            ),
            feature_weights=payload.get("feature_weights", dict(_DEFAULT_SOFT_RISK_FEATURE_WEIGHTS)),
            metadata_weights=payload.get("metadata_weights", dict(_DEFAULT_SOFT_RISK_METADATA_WEIGHTS)),
        )

    def estimate(
        self,
        prompt_features: Mapping[str, Any],
        metadata_flags: Mapping[str, Any],
    ) -> SoftPreGenerationRiskEstimate | None:
        """Return a soft risk estimate, or ``None`` when disabled."""
        if not self.enabled:
            return None
        feature_contributions = _enabled_weight_contributions(prompt_features, self.feature_weights)
        metadata_contributions = _enabled_weight_contributions(metadata_flags, self.metadata_weights)
        score = self.bias + sum(feature_contributions.values()) + sum(metadata_contributions.values())
        probability = _sigmoid(score)
        if probability >= self.high_risk_probability_threshold:
            risk_level = RiskLevel.HIGH.value
        elif probability >= self.medium_risk_probability_threshold:
            risk_level = RiskLevel.MEDIUM.value
        else:
            risk_level = RiskLevel.LOW.value
        return SoftPreGenerationRiskEstimate(
            score=score,
            probability=probability,
            risk_level=risk_level,
            feature_contributions=feature_contributions,
            metadata_contributions=metadata_contributions,
            medium_threshold=self.medium_risk_probability_threshold,
            high_threshold=self.high_risk_probability_threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "enabled": self.enabled,
            "route_on_soft_risk": self.route_on_soft_risk,
            "bias": self.bias,
            "medium_risk_probability_threshold": self.medium_risk_probability_threshold,
            "high_risk_probability_threshold": self.high_risk_probability_threshold,
            "feature_weights": dict(self.feature_weights),
            "metadata_weights": dict(self.metadata_weights),
        }


@dataclass(frozen=True)
class PreGenerationRiskPolicy:
    """Policy for routing a request before model generation.

    The policy is intentionally cheap and dependency-free. It does not verify
    facts; it decides which runtime profile should handle the prompt before
    generation based on deterministic prompt features and caller-provided
    metadata flags.
    """

    low_risk_profile: str = "latency"
    default_profile: str = "balanced"
    high_risk_profile: str = "audit"
    high_risk_feature_flags: Sequence[str] = _DEFAULT_HIGH_RISK_PROMPT_FEATURE_FLAGS
    medium_risk_feature_flags: Sequence[str] = _DEFAULT_MEDIUM_RISK_PROMPT_FEATURE_FLAGS
    high_risk_metadata_keys: Sequence[str] = _DEFAULT_HIGH_RISK_PROMPT_METADATA_KEYS
    medium_risk_metadata_keys: Sequence[str] = _DEFAULT_MEDIUM_RISK_PROMPT_METADATA_KEYS
    soft_risk_config: SoftPreGenerationRiskConfig | Mapping[str, Any] | None = field(
        default_factory=SoftPreGenerationRiskConfig
    )

    def __post_init__(self) -> None:
        low_risk_profile = _normalize_profile_name(self.low_risk_profile)
        default_profile = _normalize_profile_name(self.default_profile)
        high_risk_profile = _normalize_profile_name(self.high_risk_profile)
        _validate_profile_names(low_risk_profile, default_profile, high_risk_profile)
        object.__setattr__(self, "low_risk_profile", low_risk_profile)
        object.__setattr__(self, "default_profile", default_profile)
        object.__setattr__(self, "high_risk_profile", high_risk_profile)
        object.__setattr__(
            self,
            "high_risk_feature_flags",
            _non_empty_string_tuple(
                self.high_risk_feature_flags,
                field_name="high_risk_feature_flags",
            ),
        )
        object.__setattr__(
            self,
            "medium_risk_feature_flags",
            _non_empty_string_tuple(
                self.medium_risk_feature_flags,
                field_name="medium_risk_feature_flags",
            ),
        )
        object.__setattr__(
            self,
            "high_risk_metadata_keys",
            _non_empty_string_tuple(
                self.high_risk_metadata_keys,
                field_name="high_risk_metadata_keys",
            ),
        )
        object.__setattr__(
            self,
            "medium_risk_metadata_keys",
            _non_empty_string_tuple(
                self.medium_risk_metadata_keys,
                field_name="medium_risk_metadata_keys",
            ),
        )
        object.__setattr__(
            self,
            "soft_risk_config",
            _soft_risk_config(self.soft_risk_config),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PreGenerationRiskPolicy":
        """Build a pre-generation risk policy from a JSON-like mapping."""
        return cls(
            low_risk_profile=payload.get("low_risk_profile", cls.low_risk_profile),
            default_profile=payload.get("default_profile", cls.default_profile),
            high_risk_profile=payload.get("high_risk_profile", cls.high_risk_profile),
            high_risk_feature_flags=_as_sequence(
                payload.get("high_risk_feature_flags", cls.high_risk_feature_flags)
            ),
            medium_risk_feature_flags=_as_sequence(
                payload.get("medium_risk_feature_flags", cls.medium_risk_feature_flags)
            ),
            high_risk_metadata_keys=_as_sequence(
                payload.get("high_risk_metadata_keys", cls.high_risk_metadata_keys)
            ),
            medium_risk_metadata_keys=_as_sequence(
                payload.get("medium_risk_metadata_keys", cls.medium_risk_metadata_keys)
            ),
            soft_risk_config=payload.get("soft_risk_config", SoftPreGenerationRiskConfig()),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "low_risk_profile": self.low_risk_profile,
            "default_profile": self.default_profile,
            "high_risk_profile": self.high_risk_profile,
            "high_risk_feature_flags": tuple(self.high_risk_feature_flags),
            "medium_risk_feature_flags": tuple(self.medium_risk_feature_flags),
            "high_risk_metadata_keys": tuple(self.high_risk_metadata_keys),
            "medium_risk_metadata_keys": tuple(self.medium_risk_metadata_keys),
            "soft_risk_config": self.soft_risk_config.to_dict() if self.soft_risk_config is not None else None,
        }


@dataclass(frozen=True)
class PreGenerationRiskAssessment:
    """Auditable pre-generation profile routing result."""

    selected_profile: str
    risk_level: str
    reason: str
    triggered_features: Sequence[str] = ()
    triggered_metadata: Sequence[str] = ()
    prompt_features: Mapping[str, bool] = field(default_factory=dict)
    metadata_flags: Mapping[str, bool] = field(default_factory=dict)
    soft_risk: SoftPreGenerationRiskEstimate | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        selected_profile = _normalize_profile_name(self.selected_profile)
        if selected_profile not in RUNTIME_PROFILES:
            choices = ", ".join(RUNTIME_PROFILE_NAMES)
            raise ValueError(f"selected_profile must be one of: {choices}")
        risk_level = RiskLevel(str(self.risk_level)).value
        object.__setattr__(self, "selected_profile", selected_profile)
        object.__setattr__(self, "risk_level", risk_level)
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(
            self,
            "triggered_features",
            tuple(str(item) for item in self.triggered_features),
        )
        object.__setattr__(
            self,
            "triggered_metadata",
            tuple(str(item) for item in self.triggered_metadata),
        )
        object.__setattr__(
            self,
            "prompt_features",
            normalized_feature_flags(self.prompt_features),
        )
        object.__setattr__(
            self,
            "metadata_flags",
            normalized_feature_flags(self.metadata_flags),
        )
        object.__setattr__(
            self,
            "soft_risk",
            _soft_risk_payload(self.soft_risk),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "selected_profile": self.selected_profile,
            "risk_level": self.risk_level,
            "reason": self.reason,
            "triggered_features": tuple(self.triggered_features),
            "triggered_metadata": tuple(self.triggered_metadata),
            "prompt_features": dict(self.prompt_features),
            "metadata_flags": dict(self.metadata_flags),
            "soft_risk": self.soft_risk,
        }


RUNTIME_PROFILES: Mapping[str, RuntimeProfile] = MappingProxyType({
    "latency": RuntimeProfile(
        name="latency",
        description="Minimize triggered INSIDE work and prefer cheap verifier routes.",
        defaults={
            "inside_trigger_budget_policy": "cost_first",
            "max_inside_sample_count_ratio": 0.25,
            "max_inside_generation_seconds_ratio": 0.35,
            "max_mean_attempted_route_count": 1.1,
            "max_retrieval_use_rate": 0.0,
        },
        control_defaults={
            "staged_verification": True,
            "stage_verify_risk_levels": ("high", "unknown"),
            "stage_verify_actions": ("abstain", "clarify"),
            "stage_verify_claim_feature_flags": ("has_number", "has_citation", "is_time_sensitive"),
            "stage_verify_claim_metadata_keys": ("requires_verification",),
            "stage_verify_triggered_claims_only": True,
            "stage_fail_closed_on_skip": True,
            "max_verifier_route_attempts": 1,
        },
    ),
    "balanced": RuntimeProfile(
        name="balanced",
        description="Default release posture balancing INSIDE quality and runtime cost.",
        defaults={
            "inside_trigger_budget_policy": "quality_balanced",
            "max_inside_sample_count_ratio": 0.60,
            "max_inside_generation_seconds_ratio": 0.80,
            "max_mean_attempted_route_count": 1.5,
            "max_retrieval_use_rate": 0.50,
        },
        control_defaults={
            "staged_verification": True,
            "stage_verify_risk_levels": ("medium", "high", "unknown"),
            "stage_verify_actions": (
                "retrieve",
                "rewrite",
                "steer_regenerate",
                "execute_tool",
                "abstain",
                "clarify",
            ),
            "stage_verify_claim_feature_flags": ("has_number", "has_citation", "is_time_sensitive"),
            "stage_verify_claim_metadata_keys": ("requires_verification",),
            "stage_fail_closed_on_skip": True,
            "max_verifier_route_attempts": 2,
        },
    ),
    "audit": RuntimeProfile(
        name="audit",
        description="Prefer the highest measured INSIDE quality while keeping cost evidence bounded.",
        defaults={
            "inside_trigger_budget_policy": "quality_first",
            "max_inside_sample_count_ratio": 1.05,
            "max_inside_generation_seconds_ratio": 1.10,
            "max_mean_attempted_route_count": 3.0,
            "max_retrieval_use_rate": 1.0,
        },
        control_defaults={
            "staged_verification": False,
        },
    ),
})
RUNTIME_PROFILE_NAMES = tuple(RUNTIME_PROFILES)


def get_runtime_profile(name: str | None) -> RuntimeProfile | None:
    """Return a runtime profile by name, or ``None`` when no profile is requested."""
    if name is None:
        return None
    normalized = _normalize_profile_name(name)
    if normalized not in RUNTIME_PROFILES:
        choices = ", ".join(RUNTIME_PROFILE_NAMES)
        raise ValueError(f"runtime_profile must be one of: {choices}")
    return RUNTIME_PROFILES[normalized]


def select_pre_generation_profile(
    prompt: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    risk_policy: PreGenerationRiskPolicy | Mapping[str, Any] | None = None,
    low_risk_profile: str = "latency",
    default_profile: str = "balanced",
    high_risk_profile: str = "audit",
    high_risk_feature_flags: Sequence[str] = _DEFAULT_HIGH_RISK_PROMPT_FEATURE_FLAGS,
    medium_risk_feature_flags: Sequence[str] = _DEFAULT_MEDIUM_RISK_PROMPT_FEATURE_FLAGS,
    high_risk_metadata_keys: Sequence[str] = _DEFAULT_HIGH_RISK_PROMPT_METADATA_KEYS,
    medium_risk_metadata_keys: Sequence[str] = _DEFAULT_MEDIUM_RISK_PROMPT_METADATA_KEYS,
) -> PreGenerationRiskAssessment:
    """Choose a runtime profile before generation from cheap prompt signals.

    This function is a deterministic routing shell for confidence-aware
    generation. It does not call a model, retriever, verifier, or network
    service. High-risk prompts that appear to require current facts, retrieval,
    or domain state route to the audit profile; prompts with numbers, citations,
    or calculation markers route to the default profile; otherwise the latency
    profile is selected.
    """
    if risk_policy is None:
        policy = PreGenerationRiskPolicy(
            low_risk_profile=low_risk_profile,
            default_profile=default_profile,
            high_risk_profile=high_risk_profile,
            high_risk_feature_flags=high_risk_feature_flags,
            medium_risk_feature_flags=medium_risk_feature_flags,
            high_risk_metadata_keys=high_risk_metadata_keys,
            medium_risk_metadata_keys=medium_risk_metadata_keys,
        )
    else:
        policy = (
            risk_policy
            if isinstance(risk_policy, PreGenerationRiskPolicy)
            else PreGenerationRiskPolicy.from_mapping(risk_policy)
        )
    prompt_features = _prompt_feature_flags(prompt)
    metadata_payload = dict(metadata or {})
    metadata_flags = _metadata_flags(
        metadata_payload,
        keys=(*policy.high_risk_metadata_keys, *policy.medium_risk_metadata_keys),
    )
    soft_risk = (
        None
        if policy.soft_risk_config is None
        else policy.soft_risk_config.estimate(prompt_features, metadata_flags)
    )
    high_features = _enabled_feature_names(prompt_features, policy.high_risk_feature_flags)
    high_metadata = _enabled_feature_names(metadata_flags, policy.high_risk_metadata_keys)
    if high_features or high_metadata:
        return PreGenerationRiskAssessment(
            selected_profile=policy.high_risk_profile,
            risk_level=RiskLevel.HIGH.value,
            reason="pre-generation input requires current facts, retrieval, or domain state",
            triggered_features=high_features,
            triggered_metadata=high_metadata,
            prompt_features=prompt_features,
            metadata_flags=metadata_flags,
            soft_risk=soft_risk,
        )
    if (
        soft_risk is not None
        and policy.soft_risk_config is not None
        and policy.soft_risk_config.route_on_soft_risk
        and soft_risk.risk_level == RiskLevel.HIGH.value
    ):
        return PreGenerationRiskAssessment(
            selected_profile=policy.high_risk_profile,
            risk_level=RiskLevel.HIGH.value,
            reason="soft pre-generation risk estimate exceeded high threshold",
            triggered_features=tuple(soft_risk.feature_contributions),
            triggered_metadata=tuple(soft_risk.metadata_contributions),
            prompt_features=prompt_features,
            metadata_flags=metadata_flags,
            soft_risk=soft_risk,
        )
    medium_features = _enabled_feature_names(prompt_features, policy.medium_risk_feature_flags)
    medium_metadata = _enabled_feature_names(metadata_flags, policy.medium_risk_metadata_keys)
    if medium_features or medium_metadata:
        return PreGenerationRiskAssessment(
            selected_profile=policy.default_profile,
            risk_level=RiskLevel.MEDIUM.value,
            reason="pre-generation input contains sensitive factual or calculation markers",
            triggered_features=medium_features,
            triggered_metadata=medium_metadata,
            prompt_features=prompt_features,
            metadata_flags=metadata_flags,
            soft_risk=soft_risk,
        )
    if (
        soft_risk is not None
        and policy.soft_risk_config is not None
        and policy.soft_risk_config.route_on_soft_risk
        and soft_risk.risk_level == RiskLevel.MEDIUM.value
    ):
        return PreGenerationRiskAssessment(
            selected_profile=policy.default_profile,
            risk_level=RiskLevel.MEDIUM.value,
            reason="soft pre-generation risk estimate exceeded medium threshold",
            triggered_features=tuple(soft_risk.feature_contributions),
            triggered_metadata=tuple(soft_risk.metadata_contributions),
            prompt_features=prompt_features,
            metadata_flags=metadata_flags,
            soft_risk=soft_risk,
        )
    return PreGenerationRiskAssessment(
        selected_profile=policy.low_risk_profile,
        risk_level=RiskLevel.LOW.value,
        reason="pre-generation input has no configured risk triggers",
        prompt_features=prompt_features,
        metadata_flags=metadata_flags,
        soft_risk=soft_risk,
    )


def select_runtime_profile(
    diagnostic_decision: RiskDecision | Mapping[str, Any],
    *,
    claims: Sequence[Any] = (),
    verification_plan: ClaimVerificationPlan | Mapping[str, Any] | None = None,
    selector_policy: RuntimeProfileSelectorPolicy | Mapping[str, Any] | None = None,
    low_risk_profile: str = "latency",
    default_profile: str = "balanced",
    high_risk_profile: str = "audit",
    sensitive_profile: str = "audit",
    sensitive_claim_feature_flags: Sequence[str] = _DEFAULT_SENSITIVE_CLAIM_FEATURE_FLAGS,
    sensitive_claim_metadata_keys: Sequence[str] = _DEFAULT_SENSITIVE_CLAIM_METADATA_KEYS,
) -> RuntimeProfileSelection:
    """Choose a runtime profile from cheap diagnostics and claim metadata.

    This selector does not run verifier or retriever work. It is intended for
    request-time routing before expensive verification: low-risk non-sensitive
    requests can use the latency profile, sensitive claims use audit, and
    medium diagnostic risk stays on balanced defaults.
    """
    if selector_policy is None:
        policy = RuntimeProfileSelectorPolicy(
            low_risk_profile=low_risk_profile,
            default_profile=default_profile,
            high_risk_profile=high_risk_profile,
            sensitive_profile=sensitive_profile,
            sensitive_claim_feature_flags=sensitive_claim_feature_flags,
            sensitive_claim_metadata_keys=sensitive_claim_metadata_keys,
        )
    else:
        policy = (
            selector_policy
            if isinstance(selector_policy, RuntimeProfileSelectorPolicy)
            else RuntimeProfileSelectorPolicy.from_mapping(selector_policy)
        )
    risk_level, action = _decision_fields(diagnostic_decision)
    sensitive_claims = _sensitive_claim_matches(
        claims,
        feature_flags=policy.sensitive_claim_feature_flags,
        metadata_keys=policy.sensitive_claim_metadata_keys,
    )
    plan_cost = (
        None if verification_plan is None else estimate_verification_plan_cost(verification_plan).to_dict()
    )
    if risk_level.value in policy.high_risk_levels:
        return RuntimeProfileSelection(
            selected_profile=policy.high_risk_profile,
            reason=f"diagnostic risk level is {risk_level.value}",
            diagnostic_risk_level=risk_level.value,
            diagnostic_action=action.value,
            verification_plan_cost=plan_cost,
        )
    if action.value in policy.high_risk_actions:
        return RuntimeProfileSelection(
            selected_profile=policy.high_risk_profile,
            reason=f"diagnostic action is {action.value}",
            diagnostic_risk_level=risk_level.value,
            diagnostic_action=action.value,
            verification_plan_cost=plan_cost,
        )
    if sensitive_claims["triggered_claim_ids"]:
        return RuntimeProfileSelection(
            selected_profile=policy.sensitive_profile,
            reason="claim metadata requires audit profile",
            diagnostic_risk_level=risk_level.value,
            diagnostic_action=action.value,
            triggered_claim_ids=sensitive_claims["triggered_claim_ids"],
            triggered_features=sensitive_claims["triggered_features"],
            triggered_metadata=sensitive_claims["triggered_metadata"],
            verification_plan_cost=plan_cost,
        )
    if plan_cost is not None:
        estimated_cost_units = float(plan_cost["estimated_cost_units"])
        if (
            policy.plan_audit_cost_threshold is not None
            and estimated_cost_units >= policy.plan_audit_cost_threshold
        ):
            return RuntimeProfileSelection(
                selected_profile=policy.high_risk_profile,
                reason="verification plan estimated cost requires audit profile",
                diagnostic_risk_level=risk_level.value,
                diagnostic_action=action.value,
                verification_plan_cost=plan_cost,
            )
        if (
            policy.plan_balanced_cost_threshold is not None
            and estimated_cost_units >= policy.plan_balanced_cost_threshold
        ):
            return RuntimeProfileSelection(
                selected_profile=policy.default_profile,
                reason="verification plan estimated cost requires balanced profile",
                diagnostic_risk_level=risk_level.value,
                diagnostic_action=action.value,
                verification_plan_cost=plan_cost,
            )
    if risk_level.value in policy.low_risk_levels and action.value in policy.low_risk_actions:
        return RuntimeProfileSelection(
            selected_profile=policy.low_risk_profile,
            reason="low diagnostic risk and no sensitive claim metadata",
            diagnostic_risk_level=risk_level.value,
            diagnostic_action=action.value,
            verification_plan_cost=plan_cost,
        )
    return RuntimeProfileSelection(
        selected_profile=policy.default_profile,
        reason=f"default profile for diagnostic risk level {risk_level.value}",
        diagnostic_risk_level=risk_level.value,
        diagnostic_action=action.value,
        verification_plan_cost=plan_cost,
    )


def _prompt_feature_flags(prompt: str) -> dict[str, bool]:
    text = "" if prompt is None else str(prompt)
    return {
        "has_number": any(char.isdigit() for char in text),
        "has_citation": bool(_CITATION_RE.search(text)),
        "has_calculation": bool(_CALCULATION_RE.search(text)),
        "is_time_sensitive": bool(_TIME_SENSITIVE_RE.search(text)),
        "requires_retrieval": bool(_TIME_SENSITIVE_RE.search(text)),
        "requires_domain_state": bool(_DOMAIN_STATE_RE.search(text)),
        "has_question": bool(_QUESTION_RE.search(text)),
        "asks_exact_fact": bool(_EXACT_FACT_RE.search(text)),
        "has_named_entity_hint": bool(_NAMED_ENTITY_HINT_RE.search(text)),
        "is_multi_part": _is_multi_part_prompt(text),
        "long_prompt": len(text.split()) >= 40,
    }


def _metadata_flags(metadata: Mapping[str, Any], *, keys: Sequence[str]) -> dict[str, bool]:
    flags = {}
    for key in dict.fromkeys(str(item) for item in keys):
        flags[key] = _metadata_key_enabled(metadata, key)
    return flags


def _enabled_feature_names(flags: Mapping[str, Any], feature_names: Sequence[str]) -> tuple[str, ...]:
    return _enabled_claim_feature_names(flags, feature_names)


def _decision_fields(decision: RiskDecision | Mapping[str, Any]) -> tuple[RiskLevel, ControlAction]:
    if isinstance(decision, RiskDecision):
        return decision.risk_level, decision.action
    payload = dict(decision)
    return RiskLevel(str(payload["risk_level"])), ControlAction(str(payload["action"]))


def _sensitive_claim_matches(
    claims: Sequence[Any],
    *,
    feature_flags: Sequence[str],
    metadata_keys: Sequence[str],
) -> dict[str, Any]:
    triggered_claim_ids: list[str] = []
    triggered_features: dict[str, tuple[str, ...]] = {}
    triggered_metadata: dict[str, tuple[str, ...]] = {}
    for index, claim in enumerate(claims):
        claim_id = _claim_id(claim, fallback=f"c{index + 1}")
        metadata = _claim_metadata(claim)
        features = metadata.get("features", {})
        if not isinstance(features, Mapping):
            features = {}
        matched_features = _enabled_claim_feature_names(features, feature_flags)
        matched_metadata = tuple(
            str(key)
            for key in metadata_keys
            if _metadata_key_enabled(metadata, str(key))
        )
        if matched_features:
            triggered_features[claim_id] = matched_features
        if matched_metadata:
            triggered_metadata[claim_id] = matched_metadata
        if matched_features or matched_metadata:
            triggered_claim_ids.append(claim_id)
    return {
        "triggered_claim_ids": tuple(triggered_claim_ids),
        "triggered_features": triggered_features,
        "triggered_metadata": triggered_metadata,
    }


def _claim_id(claim: Any, *, fallback: str) -> str:
    if isinstance(claim, Mapping):
        raw = claim.get("claim_id")
    else:
        raw = getattr(claim, "claim_id", None)
    return fallback if raw is None or not str(raw).strip() else str(raw)


def _claim_metadata(claim: Any) -> dict[str, Any]:
    if isinstance(claim, Mapping):
        metadata = claim.get("metadata", {})
    else:
        metadata = getattr(claim, "metadata", {})
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _metadata_key_enabled(metadata: Mapping[str, Any], key: str) -> bool:
    return metadata_path_enabled(metadata, key)


def _validate_profile_names(*names: str) -> None:
    for name in names:
        normalized = _normalize_profile_name(name)
        if normalized not in RUNTIME_PROFILES:
            choices = ", ".join(RUNTIME_PROFILE_NAMES)
            raise ValueError(f"runtime profile selection must use one of: {choices}")


def _normalize_profile_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _risk_level_values(values: Sequence[Any], *, field_name: str) -> tuple[str, ...]:
    normalized = []
    for value in _as_sequence(values):
        try:
            normalized.append(RiskLevel(str(value)).value)
        except ValueError as exc:
            choices = ", ".join(level.value for level in RiskLevel)
            raise ValueError(f"{field_name} must contain only: {choices}") from exc
    return tuple(dict.fromkeys(normalized))


def _control_action_values(values: Sequence[Any], *, field_name: str) -> tuple[str, ...]:
    normalized = []
    for value in _as_sequence(values):
        try:
            normalized.append(ControlAction(str(value)).value)
        except ValueError as exc:
            choices = ", ".join(action.value for action in ControlAction)
            raise ValueError(f"{field_name} must contain only: {choices}") from exc
    return tuple(dict.fromkeys(normalized))


def _non_empty_string_tuple(values: Sequence[Any], *, field_name: str) -> tuple[str, ...]:
    normalized = []
    for value in _as_sequence(values):
        item = str(value).strip()
        if not item:
            raise ValueError(f"{field_name} entries must be non-empty strings")
        normalized.append(item)
    return tuple(dict.fromkeys(normalized))


def _optional_non_negative_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative finite number or None") from exc
    if parsed < 0.0 or not (parsed == parsed) or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} must be a non-negative finite number or None")
    return parsed


def _finite_float(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not (parsed == parsed) or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} must be a finite number")
    return parsed


def _probability_float(value: Any, *, field_name: str) -> float:
    parsed = _finite_float(value, field_name=field_name)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return parsed


def _float_mapping(value: Mapping[str, Any], *, field_name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    parsed: dict[str, float] = {}
    for key, raw_weight in value.items():
        name = str(key).strip()
        if not name:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        parsed[name] = _finite_float(raw_weight, field_name=f"{field_name}.{name}")
    return parsed


def _strict_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean or one of true/false, 1/0, yes/no, on/off")


def _soft_risk_config(
    value: SoftPreGenerationRiskConfig | Mapping[str, Any] | None,
) -> SoftPreGenerationRiskConfig | None:
    if value is None:
        return None
    if isinstance(value, SoftPreGenerationRiskConfig):
        return value
    if isinstance(value, Mapping):
        return SoftPreGenerationRiskConfig.from_mapping(value)
    raise ValueError("soft_risk_config must be a mapping, SoftPreGenerationRiskConfig, or None")


def _soft_risk_payload(value: SoftPreGenerationRiskEstimate | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, SoftPreGenerationRiskEstimate):
        return value.to_dict()
    if isinstance(value, Mapping):
        payload = dict(value)
        if "score" in payload:
            payload["score"] = _finite_float(payload["score"], field_name="soft_risk.score")
        if "probability" in payload:
            payload["probability"] = _probability_float(
                payload["probability"],
                field_name="soft_risk.probability",
            )
        if "risk_level" in payload:
            payload["risk_level"] = RiskLevel(str(payload["risk_level"])).value
        return payload
    raise ValueError("soft_risk must be a mapping, SoftPreGenerationRiskEstimate, or None")


def _enabled_weight_contributions(
    flags: Mapping[str, Any],
    weights: Mapping[str, float],
) -> dict[str, float]:
    return {
        str(name): float(weight)
        for name, weight in weights.items()
        if _metadata_key_enabled(flags, str(name)) and float(weight) != 0.0
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        denominator = 1.0 + exp(-value)
        return 1.0 / denominator
    numerator = exp(value)
    return numerator / (1.0 + numerator)


def _is_multi_part_prompt(text: str) -> bool:
    lowered = text.lower()
    separators = lowered.count("?") + lowered.count(";") + lowered.count(",")
    connective_count = len(re.findall(r"\b(and|then|also|compare|versus|vs\.?)\b", lowered))
    return separators + connective_count >= 2


def _string_sequence_mapping(value: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    return {
        str(key): tuple(str(item) for item in _as_sequence(items))
        for key, items in value.items()
    }


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)
