"""Runtime budget checks for product control traces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from eigentruth.control.trace import ProductTrace, RuntimeTrace


@dataclass(frozen=True)
class ProductRuntimeBudgetPolicy:
    """Optional runtime thresholds for one product trace.

    Missing or non-finite metrics fail closed only when the corresponding
    threshold is configured.
    """

    max_total_seconds: float | None = None
    max_phase_seconds: Mapping[str, float] = field(default_factory=dict)
    require_runtime_trace: bool = True

    def __post_init__(self) -> None:
        max_total_seconds = _optional_non_negative_float(
            self.max_total_seconds,
            name="max_total_seconds",
        )
        max_phase_seconds = {}
        for raw_name, raw_value in self.max_phase_seconds.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("runtime phase budget names must be non-empty")
            max_phase_seconds[name] = _required_non_negative_float(
                raw_value,
                name=f"max_phase_seconds.{name}",
            )
        object.__setattr__(self, "max_total_seconds", max_total_seconds)
        object.__setattr__(self, "max_phase_seconds", max_phase_seconds)
        object.__setattr__(self, "require_runtime_trace", bool(self.require_runtime_trace))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProductRuntimeBudgetPolicy":
        """Build a policy from a JSON-like mapping."""
        return cls(
            max_total_seconds=payload.get("max_total_seconds"),
            max_phase_seconds=dict(_mapping(payload.get("max_phase_seconds"))),
            require_runtime_trace=_bool_value(payload.get("require_runtime_trace", True)),
        )

    def enabled(self) -> bool:
        """Return whether any runtime threshold is configured."""
        return self.max_total_seconds is not None or bool(self.max_phase_seconds)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "max_total_seconds": self.max_total_seconds,
            "max_phase_seconds": dict(self.max_phase_seconds),
            "require_runtime_trace": self.require_runtime_trace,
        }


def evaluate_product_runtime_budget(
    trace: ProductTrace | Mapping[str, Any],
    policy: ProductRuntimeBudgetPolicy | Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate a product trace against a runtime budget policy."""
    resolved = (
        policy
        if isinstance(policy, ProductRuntimeBudgetPolicy)
        else ProductRuntimeBudgetPolicy.from_mapping(policy)
    )
    metrics = product_runtime_metrics(trace)
    failures = []
    checks = []

    if resolved.enabled() and resolved.require_runtime_trace and not metrics["has_runtime_trace"]:
        check = {
            "metric": "runtime_trace",
            "limit_type": "required",
            "limit": True,
            "value": False,
            "raw_value": None,
            "passed": False,
        }
        checks.append(check)
        failures.append(_failure_from_check(check, reason="missing"))

    if resolved.max_total_seconds is not None:
        check = _max_metric_check(
            metrics,
            metric="total_seconds",
            limit=resolved.max_total_seconds,
        )
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))

    phase_seconds = _mapping(metrics.get("phase_seconds"))
    raw_phase_seconds = _mapping(metrics.get("raw_phase_seconds"))
    for phase_name, limit in resolved.max_phase_seconds.items():
        metric = f"phase_seconds.{phase_name}"
        check = _max_metric_check(
            phase_seconds,
            metric=phase_name,
            limit=limit,
            output_metric=metric,
            raw_value=raw_phase_seconds.get(phase_name),
        )
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))

    return {
        "enabled": resolved.enabled(),
        "passed": not failures,
        "policy": resolved.to_dict(),
        "metrics": {
            "has_runtime_trace": metrics["has_runtime_trace"],
            "total_seconds": metrics["total_seconds"],
            "accounted_seconds": metrics["accounted_seconds"],
            "unaccounted_seconds": metrics["unaccounted_seconds"],
            "measured_phases": metrics["measured_phases"],
            "phase_seconds": phase_seconds,
            "phase_counts": _mapping(metrics.get("phase_counts")),
            "slowest_phase": metrics.get("slowest_phase"),
        },
        "checks": checks,
        "failures": failures,
    }


def product_runtime_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    """Extract runtime-budget metrics from a ProductTrace or trace payload."""
    runtime_trace = _runtime_trace_payload(trace)
    if runtime_trace is None:
        return {
            "has_runtime_trace": False,
            "total_seconds": None,
            "accounted_seconds": None,
            "unaccounted_seconds": None,
            "measured_phases": None,
            "phase_seconds": {},
            "raw_phase_seconds": {},
            "phase_counts": {},
            "slowest_phase": None,
        }
    summary = _runtime_summary(runtime_trace)
    phase_seconds = _mapping(summary.get("phase_seconds"))
    return {
        "has_runtime_trace": True,
        "total_seconds": _finite_float(summary.get("total_seconds")),
        "accounted_seconds": _finite_float(summary.get("accounted_seconds")),
        "unaccounted_seconds": _finite_float(summary.get("unaccounted_seconds")),
        "measured_phases": _finite_float(summary.get("measured_phases")),
        "phase_seconds": {
            str(name): _finite_float(value)
            for name, value in phase_seconds.items()
        },
        "raw_phase_seconds": dict(phase_seconds),
        "phase_counts": _mapping(summary.get("phase_counts")),
        "slowest_phase": summary.get("slowest_phase"),
    }


def _runtime_trace_payload(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any] | None:
    if isinstance(trace, ProductTrace):
        payload = trace.to_dict()
    elif isinstance(trace, RuntimeTrace):
        payload = trace.to_dict()
    else:
        payload = dict(trace)
    if "runtime_trace" in payload:
        runtime_trace = payload.get("runtime_trace")
    else:
        runtime_trace = payload
    if runtime_trace is None:
        return None
    if not isinstance(runtime_trace, Mapping):
        return None
    return dict(runtime_trace)


def _runtime_summary(runtime_trace: Mapping[str, Any]) -> dict[str, Any]:
    summary = runtime_trace.get("summary")
    if isinstance(summary, Mapping):
        return dict(summary)
    return RuntimeTrace.from_dict(runtime_trace).summary()


def _max_metric_check(
    metrics: Mapping[str, Any],
    *,
    metric: str,
    limit: float,
    output_metric: str | None = None,
    raw_value: Any = None,
) -> dict[str, Any]:
    observed = _finite_float(metrics.get(metric))
    raw = metrics.get(metric) if raw_value is None else raw_value
    return {
        "metric": output_metric or metric,
        "limit_type": "max",
        "limit": limit,
        "value": observed,
        "raw_value": None if raw is None else repr(raw),
        "passed": observed is not None and observed <= limit,
    }


def _failure_from_check(check: Mapping[str, Any], *, reason: str | None = None) -> dict[str, Any]:
    if reason is None:
        reason = "missing or non-finite"
        if check.get("value") is not None:
            reason = f"above {check['limit']}"
    return {
        "metric": check.get("metric"),
        "limit_type": check.get("limit_type"),
        "limit": check.get("limit"),
        "value": check.get("value"),
        "raw_value": check.get("raw_value"),
        "reason": reason,
    }


def _optional_non_negative_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _required_non_negative_float(value, name=name)


def _required_non_negative_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative finite number.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return numeric


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("require_runtime_trace must be a boolean value.")


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}
