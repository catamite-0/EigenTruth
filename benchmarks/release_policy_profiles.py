"""Named release-gate policy defaults shared by release workflows."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

_CANDIDATE_RELEASE_POLICY_DEFAULTS: Mapping[str, Any] = {
    "min_best_quality_auroc": 0.70,
    "max_uncached_forward_seconds": 20.0,
    "min_selected": 4,
    "min_decision_accuracy": 0.95,
    "max_false_supported_rate": 0.05,
    "min_false_refuted_rate": 0.50,
}

_STRICT_STRUCTURED_FACT_DEFAULTS: Mapping[str, Any] = {
    **_CANDIDATE_RELEASE_POLICY_DEFAULTS,
    "require_structured_fact_robustness": True,
    "min_decision_accuracy": 0.99,
    "max_false_supported_rate": 0.0,
    "min_false_refuted_rate": 0.99,
    "required_route_min_selected": 700,
    "required_route_min_decision_accuracy": 0.99,
    "required_route_max_false_supported_rate": 0.0,
    "required_route_min_false_refuted_rate": 0.99,
}

RELEASE_POLICY_PROFILES: Mapping[str, Mapping[str, Any]] = {
    "research_smoke": {
        "min_best_quality_auroc": 0.50,
        "min_selected": 1,
    },
    "candidate_release": _CANDIDATE_RELEASE_POLICY_DEFAULTS,
    "strict_structured_fact": _STRICT_STRUCTURED_FACT_DEFAULTS,
    "frontier_audit": {
        **_STRICT_STRUCTURED_FACT_DEFAULTS,
        "adapter_family_profile": "strict_audit",
        "require_state_transition_world_model": True,
        "require_product_runtime_drift_promotion_evidence": True,
    },
}
RELEASE_POLICY_PROFILE_NAMES = tuple(sorted(RELEASE_POLICY_PROFILES))


def clean_optional_key(value: str | None) -> str | None:
    """Normalize optional registry keys, treating blank strings as missing."""
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def append_unique(existing: Sequence[str], additions: Sequence[str | None]) -> tuple[str, ...]:
    """Append non-empty values while preserving order and removing duplicates."""
    values = list(existing)
    seen = set(values)
    for raw in additions:
        value = clean_optional_key(raw)
        if value is None or value in seen:
            continue
        values.append(value)
        seen.add(value)
    return tuple(values)


def normalize_release_policy_profile(value: str | None) -> str | None:
    """Normalize and validate a release policy profile name."""
    if value is None:
        return None
    profile = str(value).strip().lower().replace("-", "_")
    if profile not in RELEASE_POLICY_PROFILES:
        choices = ", ".join(RELEASE_POLICY_PROFILE_NAMES)
        raise ValueError(f"release_policy_profile must be one of: {choices}")
    return profile


def apply_release_policy_profile_defaults(
    release_policy_profile: str | None,
    values: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
    """Apply profile defaults to unset keys without overriding explicit values."""
    profile = normalize_release_policy_profile(release_policy_profile)
    merged = dict(values)
    applied: dict[str, Any] = {}
    if profile is None:
        return None, merged, applied
    for key, default in RELEASE_POLICY_PROFILES[profile].items():
        if key not in merged:
            continue
        if not _profile_default_is_unset(merged[key], default):
            continue
        merged[key] = default
        applied[key] = default
    return profile, merged, applied


def _profile_default_is_unset(current: Any, default: Any) -> bool:
    if isinstance(default, bool):
        return current is False and default is True
    if isinstance(default, (tuple, list)):
        return not tuple(current or ())
    return current is None
