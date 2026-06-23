"""Named runtime profiles for release-gate defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from eigentruth.control.policy import ControlAction, RiskDecision, RiskLevel

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
})
_DEFAULT_SENSITIVE_CLAIM_FEATURE_FLAGS = (
    "has_number",
    "has_citation",
    "is_time_sensitive",
    "has_calculation",
)
_DEFAULT_SENSITIVE_CLAIM_METADATA_KEYS = ("requires_verification",)


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


def select_runtime_profile(
    diagnostic_decision: RiskDecision | Mapping[str, Any],
    *,
    claims: Sequence[Any] = (),
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
    risk_level, action = _decision_fields(diagnostic_decision)
    _validate_profile_names(low_risk_profile, default_profile, high_risk_profile, sensitive_profile)
    sensitive_claims = _sensitive_claim_matches(
        claims,
        feature_flags=sensitive_claim_feature_flags,
        metadata_keys=sensitive_claim_metadata_keys,
    )
    if risk_level in {RiskLevel.HIGH, RiskLevel.UNKNOWN}:
        return RuntimeProfileSelection(
            selected_profile=high_risk_profile,
            reason=f"diagnostic risk level is {risk_level.value}",
            diagnostic_risk_level=risk_level.value,
            diagnostic_action=action.value,
        )
    if action in {
        ControlAction.ABSTAIN,
        ControlAction.CLARIFY,
        ControlAction.REWRITE,
        ControlAction.STEER_REGENERATE,
        ControlAction.EXECUTE_TOOL,
    }:
        return RuntimeProfileSelection(
            selected_profile=high_risk_profile,
            reason=f"diagnostic action is {action.value}",
            diagnostic_risk_level=risk_level.value,
            diagnostic_action=action.value,
        )
    if sensitive_claims["triggered_claim_ids"]:
        return RuntimeProfileSelection(
            selected_profile=sensitive_profile,
            reason="claim metadata requires audit profile",
            diagnostic_risk_level=risk_level.value,
            diagnostic_action=action.value,
            triggered_claim_ids=sensitive_claims["triggered_claim_ids"],
            triggered_features=sensitive_claims["triggered_features"],
            triggered_metadata=sensitive_claims["triggered_metadata"],
        )
    if risk_level is RiskLevel.LOW and action is ControlAction.ACCEPT:
        return RuntimeProfileSelection(
            selected_profile=low_risk_profile,
            reason="low diagnostic risk and no sensitive claim metadata",
            diagnostic_risk_level=risk_level.value,
            diagnostic_action=action.value,
        )
    return RuntimeProfileSelection(
        selected_profile=default_profile,
        reason=f"default profile for diagnostic risk level {risk_level.value}",
        diagnostic_risk_level=risk_level.value,
        diagnostic_action=action.value,
    )


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
        matched_features = tuple(
            str(flag)
            for flag in feature_flags
            if features.get(str(flag)) is True
        )
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
    current: Any = metadata
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return bool(current)


def _validate_profile_names(*names: str) -> None:
    for name in names:
        normalized = _normalize_profile_name(name)
        if normalized not in RUNTIME_PROFILES:
            choices = ", ".join(RUNTIME_PROFILE_NAMES)
            raise ValueError(f"runtime profile selection must use one of: {choices}")


def _normalize_profile_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


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
