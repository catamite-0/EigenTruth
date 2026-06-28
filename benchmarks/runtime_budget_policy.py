"""Small fail-closed runtime budget helpers for benchmark workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RuntimeBudgetPolicy:
    """Optional runtime/cost/cache thresholds for benchmark promotion gates."""

    max_total_seconds: float | None = None
    max_retrieval_hit_count: float | None = None
    min_cache_hit_rate: float | None = None
    min_claims_cache_hit_rate: float | None = None
    min_verifier_trace_cache_hit_rate: float | None = None

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} must be a non-negative finite number.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RuntimeBudgetPolicy":
        """Build a policy from a JSON-like mapping."""
        return cls(
            max_total_seconds=_optional_float(payload.get("max_total_seconds")),
            max_retrieval_hit_count=_optional_float(payload.get("max_retrieval_hit_count")),
            min_cache_hit_rate=_optional_float(payload.get("min_cache_hit_rate")),
            min_claims_cache_hit_rate=_optional_float(payload.get("min_claims_cache_hit_rate")),
            min_verifier_trace_cache_hit_rate=_optional_float(
                payload.get("min_verifier_trace_cache_hit_rate")
            ),
        )

    def enabled(self) -> bool:
        """Return whether the policy has any active threshold."""
        return any(value is not None for value in self.to_dict().values())

    def to_dict(self) -> dict[str, float | None]:
        """Return a JSON-serializable policy payload."""
        return {
            "max_total_seconds": self.max_total_seconds,
            "max_retrieval_hit_count": self.max_retrieval_hit_count,
            "min_cache_hit_rate": self.min_cache_hit_rate,
            "min_claims_cache_hit_rate": self.min_claims_cache_hit_rate,
            "min_verifier_trace_cache_hit_rate": self.min_verifier_trace_cache_hit_rate,
        }


def evaluate_runtime_budget(
    metrics: Mapping[str, Any],
    policy: RuntimeBudgetPolicy | Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate metrics against a runtime budget policy.

    Missing or non-finite metrics fail closed only when their corresponding
    threshold is configured.
    """
    resolved = policy if isinstance(policy, RuntimeBudgetPolicy) else RuntimeBudgetPolicy.from_mapping(policy)
    failures = []
    checks = []
    max_checks = (
        ("total_seconds", resolved.max_total_seconds),
        ("retrieval_hit_count", resolved.max_retrieval_hit_count),
    )
    min_checks = (
        ("cache_hit_rate", resolved.min_cache_hit_rate),
        ("claims_cache_hit_rate", resolved.min_claims_cache_hit_rate),
        ("verifier_trace_cache_hit_rate", resolved.min_verifier_trace_cache_hit_rate),
    )
    for metric, limit in max_checks:
        if limit is None:
            continue
        check = _metric_check(metrics, metric=metric, limit=float(limit), limit_type="max")
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))
    for metric, limit in min_checks:
        if limit is None:
            continue
        check = _metric_check(metrics, metric=metric, limit=float(limit), limit_type="min")
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))
    return {
        "enabled": resolved.enabled(),
        "passed": not failures,
        "policy": resolved.to_dict(),
        "metrics": {
            "total_seconds": _finite_float(metrics.get("total_seconds")),
            "retrieval_hit_count": _finite_float(metrics.get("retrieval_hit_count")),
            "cache_hit_rate": _finite_float(metrics.get("cache_hit_rate")),
            "claims_cache_hit_rate": _finite_float(metrics.get("claims_cache_hit_rate")),
            "verifier_trace_cache_hit_rate": _finite_float(metrics.get("verifier_trace_cache_hit_rate")),
        },
        "checks": checks,
        "failures": failures,
    }


def runtime_metrics_from_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Extract budget metrics from a local workflow runtime profile."""
    cache = _mapping(profile.get("cache"))
    claims_cache = _mapping(cache.get("claims"))
    trace_cache = _mapping(cache.get("verifier_trace"))
    scale = _mapping(profile.get("scale"))
    return {
        "total_seconds": profile.get("total_seconds"),
        "retrieval_hit_count": scale.get("n_retrieval_hits"),
        "claims_cache_hit_rate": _boolean_hit_rate(
            enabled=claims_cache.get("enabled"),
            hit=claims_cache.get("hit"),
        ),
        "verifier_trace_cache_hit_rate": _count_hit_rate(
            enabled=trace_cache.get("enabled"),
            hits=trace_cache.get("hit_count"),
            requests=trace_cache.get("run_count"),
        ),
    }


def runtime_metrics_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Extract budget metrics from artifact/registry metadata."""
    total_seconds = metadata.get("runtime_total_seconds")
    if total_seconds is None:
        total_seconds = metadata.get("wall_clock_seconds")
    return {
        "total_seconds": total_seconds,
        "retrieval_hit_count": metadata.get("runtime_n_retrieval_hits"),
        "claims_cache_hit_rate": _boolean_hit_rate(
            enabled=metadata.get("claims_cache_enabled"),
            hit=metadata.get("claims_cache_hit"),
        ),
        "verifier_trace_cache_hit_rate": _count_hit_rate(
            enabled=metadata.get("verifier_trace_cache_enabled"),
            hits=metadata.get("verifier_trace_cache_hit_count"),
            requests=metadata.get("verifier_trace_cache_run_count"),
        ),
    }


def _metric_check(
    metrics: Mapping[str, Any],
    *,
    metric: str,
    limit: float,
    limit_type: str,
) -> dict[str, Any]:
    observed = _finite_float(metrics.get(metric))
    passed = observed is not None and (
        observed <= limit if limit_type == "max" else observed >= limit
    )
    return {
        "metric": metric,
        "limit_type": limit_type,
        "limit": limit,
        "value": observed,
        "raw_value": None if metrics.get(metric) is None else repr(metrics.get(metric)),
        "passed": passed,
    }


def _failure_from_check(check: Mapping[str, Any]) -> dict[str, Any]:
    reason = "missing or non-finite"
    if check.get("value") is not None:
        reason = (
            f"above {check['limit']}"
            if check.get("limit_type") == "max"
            else f"below {check['limit']}"
        )
    return {
        "metric": check.get("metric"),
        "limit_type": check.get("limit_type"),
        "limit": check.get("limit"),
        "value": check.get("value"),
        "raw_value": check.get("raw_value"),
        "reason": reason,
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _boolean_hit_rate(*, enabled: Any, hit: Any) -> float | None:
    if not bool(enabled):
        return None
    return 1.0 if bool(hit) else 0.0


def _count_hit_rate(*, enabled: Any, hits: Any, requests: Any) -> float | None:
    if not bool(enabled):
        return None
    hit_count = _finite_float(hits)
    request_count = _finite_float(requests)
    if hit_count is None or request_count is None or request_count <= 0.0:
        return None
    return hit_count / request_count


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}
