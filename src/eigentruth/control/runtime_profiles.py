"""Named runtime profiles for release-gate defaults."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

_DEFAULT_KEYS = frozenset({
    "inside_trigger_budget_policy",
    "max_inside_sample_count_ratio",
    "max_inside_generation_seconds_ratio",
    "max_mean_attempted_route_count",
    "max_retrieval_use_rate",
})


@dataclass(frozen=True)
class RuntimeProfile:
    """Named set of release-gate defaults.

    Profiles only fill unset values. Explicit CLI or API parameters remain the
    source of truth for a run.
    """

    name: str
    description: str
    defaults: Mapping[str, Any]

    def __post_init__(self) -> None:
        name = self.name.strip().lower().replace("-", "_")
        if not name:
            raise ValueError("runtime profile name must not be empty")
        defaults = dict(self.defaults)
        unknown = sorted(set(defaults) - _DEFAULT_KEYS)
        if unknown:
            raise ValueError(f"unknown runtime profile defaults: {', '.join(unknown)}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "defaults", MappingProxyType(defaults))

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
    ),
})
RUNTIME_PROFILE_NAMES = tuple(RUNTIME_PROFILES)


def get_runtime_profile(name: str | None) -> RuntimeProfile | None:
    """Return a runtime profile by name, or ``None`` when no profile is requested."""
    if name is None:
        return None
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized not in RUNTIME_PROFILES:
        choices = ", ".join(RUNTIME_PROFILE_NAMES)
        raise ValueError(f"runtime_profile must be one of: {choices}")
    return RUNTIME_PROFILES[normalized]
