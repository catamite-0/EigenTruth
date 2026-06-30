"""Runtime budget checks for product control traces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from eigentruth.control import runtime_drift_keys as _runtime_drift_keys
from eigentruth.control.receipts import action_receipt_summary_from_results
from eigentruth.control.trace import ProductTrace, RuntimeTrace

_PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_ACTION_GATE_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_ACTION_GATE_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_TRAJECTORY_AUDIT_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_TRAJECTORY_AUDIT_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_EVIDENCE_HANDOFF_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_EVIDENCE_HANDOFF_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_CONTEXT_SENSITIVITY_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_CONTEXT_SENSITIVITY_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_ROBUSTNESS_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_ROBUSTNESS_EVIDENCE_KEYS
)
_PRODUCT_RUNTIME_DRIFT_FRONTIER_RELEASE_EVIDENCE_PREFIXES = (
    _runtime_drift_keys.PRODUCT_RUNTIME_DRIFT_FRONTIER_RELEASE_EVIDENCE_KEYS
)


@dataclass(frozen=True)
class ProductRuntimeBudgetPolicy:
    """Optional runtime, cache, and route-cost thresholds for one product trace.

    Missing or non-finite metrics fail closed only when the corresponding
    threshold is configured.
    """

    max_total_seconds: float | None = None
    max_phase_seconds: Mapping[str, float] = field(default_factory=dict)
    max_phase_p95_seconds: Mapping[str, float] = field(default_factory=dict)
    max_phase_p99_seconds: Mapping[str, float] = field(default_factory=dict)
    max_mean_route_duration_seconds: float | None = None
    max_p95_route_duration_seconds: float | None = None
    max_p99_route_duration_seconds: float | None = None
    max_route_duration_seconds: float | None = None
    max_mean_attempted_route_count: float | None = None
    max_route_budget_exhaustion_rate: float | None = None
    max_retrieval_use_rate: float | None = None
    max_retrieval_hit_count: float | None = None
    min_cache_hit_rate: float | None = None
    min_named_cache_hit_rate: Mapping[str, float] = field(default_factory=dict)
    min_verification_skip_rate: float | None = None
    min_selective_claim_skip_rate: float | None = None
    max_verified_claim_count: float | None = None
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
        max_phase_p95_seconds = _phase_budget_mapping(
            self.max_phase_p95_seconds,
            field_name="max_phase_p95_seconds",
        )
        max_phase_p99_seconds = _phase_budget_mapping(
            self.max_phase_p99_seconds,
            field_name="max_phase_p99_seconds",
        )
        max_mean_route_duration_seconds = _optional_non_negative_float(
            self.max_mean_route_duration_seconds,
            name="max_mean_route_duration_seconds",
        )
        max_p95_route_duration_seconds = _optional_non_negative_float(
            self.max_p95_route_duration_seconds,
            name="max_p95_route_duration_seconds",
        )
        max_p99_route_duration_seconds = _optional_non_negative_float(
            self.max_p99_route_duration_seconds,
            name="max_p99_route_duration_seconds",
        )
        max_route_duration_seconds = _optional_non_negative_float(
            self.max_route_duration_seconds,
            name="max_route_duration_seconds",
        )
        max_mean_attempted_route_count = _optional_non_negative_float(
            self.max_mean_attempted_route_count,
            name="max_mean_attempted_route_count",
        )
        max_route_budget_exhaustion_rate = _optional_rate_float(
            self.max_route_budget_exhaustion_rate,
            name="max_route_budget_exhaustion_rate",
        )
        max_retrieval_use_rate = _optional_rate_float(
            self.max_retrieval_use_rate,
            name="max_retrieval_use_rate",
        )
        max_retrieval_hit_count = _optional_non_negative_float(
            self.max_retrieval_hit_count,
            name="max_retrieval_hit_count",
        )
        min_cache_hit_rate = _optional_rate_float(
            self.min_cache_hit_rate,
            name="min_cache_hit_rate",
        )
        min_verification_skip_rate = _optional_rate_float(
            self.min_verification_skip_rate,
            name="min_verification_skip_rate",
        )
        min_selective_claim_skip_rate = _optional_rate_float(
            self.min_selective_claim_skip_rate,
            name="min_selective_claim_skip_rate",
        )
        max_verified_claim_count = _optional_non_negative_float(
            self.max_verified_claim_count,
            name="max_verified_claim_count",
        )
        min_named_cache_hit_rate = {}
        for raw_name, raw_value in self.min_named_cache_hit_rate.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("named cache budget names must be non-empty")
            min_named_cache_hit_rate[name] = _required_rate_float(
                raw_value,
                name=f"min_named_cache_hit_rate.{name}",
            )
        object.__setattr__(self, "max_total_seconds", max_total_seconds)
        object.__setattr__(self, "max_phase_seconds", max_phase_seconds)
        object.__setattr__(self, "max_phase_p95_seconds", max_phase_p95_seconds)
        object.__setattr__(self, "max_phase_p99_seconds", max_phase_p99_seconds)
        object.__setattr__(
            self,
            "max_mean_route_duration_seconds",
            max_mean_route_duration_seconds,
        )
        object.__setattr__(
            self,
            "max_p95_route_duration_seconds",
            max_p95_route_duration_seconds,
        )
        object.__setattr__(
            self,
            "max_p99_route_duration_seconds",
            max_p99_route_duration_seconds,
        )
        object.__setattr__(
            self,
            "max_route_duration_seconds",
            max_route_duration_seconds,
        )
        object.__setattr__(
            self,
            "max_mean_attempted_route_count",
            max_mean_attempted_route_count,
        )
        object.__setattr__(
            self,
            "max_route_budget_exhaustion_rate",
            max_route_budget_exhaustion_rate,
        )
        object.__setattr__(self, "max_retrieval_use_rate", max_retrieval_use_rate)
        object.__setattr__(self, "max_retrieval_hit_count", max_retrieval_hit_count)
        object.__setattr__(self, "min_cache_hit_rate", min_cache_hit_rate)
        object.__setattr__(self, "min_named_cache_hit_rate", min_named_cache_hit_rate)
        object.__setattr__(self, "min_verification_skip_rate", min_verification_skip_rate)
        object.__setattr__(self, "min_selective_claim_skip_rate", min_selective_claim_skip_rate)
        object.__setattr__(self, "max_verified_claim_count", max_verified_claim_count)
        object.__setattr__(
            self,
            "require_runtime_trace",
            _bool_value(self.require_runtime_trace),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProductRuntimeBudgetPolicy":
        """Build a policy from a JSON-like mapping."""
        return cls(
            max_total_seconds=payload.get("max_total_seconds"),
            max_phase_seconds=dict(_mapping(payload.get("max_phase_seconds"))),
            max_phase_p95_seconds=dict(_mapping(payload.get("max_phase_p95_seconds"))),
            max_phase_p99_seconds=dict(_mapping(payload.get("max_phase_p99_seconds"))),
            max_mean_route_duration_seconds=payload.get("max_mean_route_duration_seconds"),
            max_p95_route_duration_seconds=payload.get("max_p95_route_duration_seconds"),
            max_p99_route_duration_seconds=payload.get("max_p99_route_duration_seconds"),
            max_route_duration_seconds=payload.get("max_route_duration_seconds"),
            max_mean_attempted_route_count=payload.get("max_mean_attempted_route_count"),
            max_route_budget_exhaustion_rate=payload.get("max_route_budget_exhaustion_rate"),
            max_retrieval_use_rate=payload.get("max_retrieval_use_rate"),
            max_retrieval_hit_count=payload.get("max_retrieval_hit_count"),
            min_cache_hit_rate=payload.get("min_cache_hit_rate"),
            min_named_cache_hit_rate=dict(_mapping(payload.get("min_named_cache_hit_rate"))),
            min_verification_skip_rate=payload.get("min_verification_skip_rate"),
            min_selective_claim_skip_rate=payload.get("min_selective_claim_skip_rate"),
            max_verified_claim_count=payload.get("max_verified_claim_count"),
            require_runtime_trace=_bool_value(payload.get("require_runtime_trace", True)),
        )

    def enabled(self) -> bool:
        """Return whether any runtime threshold is configured."""
        return (
            self.max_total_seconds is not None
            or bool(self.max_phase_seconds)
            or bool(self.max_phase_p95_seconds)
            or bool(self.max_phase_p99_seconds)
            or self.max_mean_route_duration_seconds is not None
            or self.max_p95_route_duration_seconds is not None
            or self.max_p99_route_duration_seconds is not None
            or self.max_route_duration_seconds is not None
            or self.max_mean_attempted_route_count is not None
            or self.max_route_budget_exhaustion_rate is not None
            or self.max_retrieval_use_rate is not None
            or self.max_retrieval_hit_count is not None
            or self.min_cache_hit_rate is not None
            or bool(self.min_named_cache_hit_rate)
            or self.min_verification_skip_rate is not None
            or self.min_selective_claim_skip_rate is not None
            or self.max_verified_claim_count is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "max_total_seconds": self.max_total_seconds,
            "max_phase_seconds": dict(self.max_phase_seconds),
            "max_phase_p95_seconds": dict(self.max_phase_p95_seconds),
            "max_phase_p99_seconds": dict(self.max_phase_p99_seconds),
            "max_mean_route_duration_seconds": self.max_mean_route_duration_seconds,
            "max_p95_route_duration_seconds": self.max_p95_route_duration_seconds,
            "max_p99_route_duration_seconds": self.max_p99_route_duration_seconds,
            "max_route_duration_seconds": self.max_route_duration_seconds,
            "max_mean_attempted_route_count": self.max_mean_attempted_route_count,
            "max_route_budget_exhaustion_rate": self.max_route_budget_exhaustion_rate,
            "max_retrieval_use_rate": self.max_retrieval_use_rate,
            "max_retrieval_hit_count": self.max_retrieval_hit_count,
            "min_cache_hit_rate": self.min_cache_hit_rate,
            "min_named_cache_hit_rate": dict(self.min_named_cache_hit_rate),
            "min_verification_skip_rate": self.min_verification_skip_rate,
            "min_selective_claim_skip_rate": self.min_selective_claim_skip_rate,
            "max_verified_claim_count": self.max_verified_claim_count,
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
    requires_runtime_trace = (
        resolved.max_total_seconds is not None
        or bool(resolved.max_phase_seconds)
        or bool(resolved.max_phase_p95_seconds)
        or bool(resolved.max_phase_p99_seconds)
    )

    if requires_runtime_trace and resolved.require_runtime_trace and not metrics["has_runtime_trace"]:
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

    phase_p95_seconds = _mapping(metrics.get("phase_p95_seconds"))
    raw_phase_p95_seconds = _mapping(metrics.get("raw_phase_p95_seconds"))
    for phase_name, limit in resolved.max_phase_p95_seconds.items():
        metric = f"phase_p95_seconds.{phase_name}"
        check = _max_metric_check(
            phase_p95_seconds,
            metric=phase_name,
            limit=limit,
            output_metric=metric,
            raw_value=raw_phase_p95_seconds.get(phase_name),
        )
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))

    phase_p99_seconds = _mapping(metrics.get("phase_p99_seconds"))
    raw_phase_p99_seconds = _mapping(metrics.get("raw_phase_p99_seconds"))
    for phase_name, limit in resolved.max_phase_p99_seconds.items():
        metric = f"phase_p99_seconds.{phase_name}"
        check = _max_metric_check(
            phase_p99_seconds,
            metric=phase_name,
            limit=limit,
            output_metric=metric,
            raw_value=raw_phase_p99_seconds.get(phase_name),
        )
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))

    for metric, limit in (
        ("mean_route_duration_seconds", resolved.max_mean_route_duration_seconds),
        ("p95_route_duration_seconds", resolved.max_p95_route_duration_seconds),
        ("p99_route_duration_seconds", resolved.max_p99_route_duration_seconds),
        ("max_route_duration_seconds", resolved.max_route_duration_seconds),
        ("mean_attempted_route_count", resolved.max_mean_attempted_route_count),
        ("route_budget_exhaustion_rate", resolved.max_route_budget_exhaustion_rate),
        ("retrieval_use_rate", resolved.max_retrieval_use_rate),
        ("retrieval_hit_count", resolved.max_retrieval_hit_count),
        ("verified_claim_count", resolved.max_verified_claim_count),
    ):
        if limit is None:
            continue
        check = _max_metric_check(metrics, metric=metric, limit=limit)
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))

    if resolved.min_cache_hit_rate is not None:
        check = _min_metric_check(
            metrics,
            metric="cache_hit_rate",
            limit=resolved.min_cache_hit_rate,
        )
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))

    if resolved.min_verification_skip_rate is not None:
        check = _min_metric_check(
            metrics,
            metric="verification_skip_rate",
            limit=resolved.min_verification_skip_rate,
        )
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))

    if resolved.min_selective_claim_skip_rate is not None:
        check = _min_metric_check(
            metrics,
            metric="selective_claim_skip_rate",
            limit=resolved.min_selective_claim_skip_rate,
        )
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))

    named_cache_hit_rates = _mapping(metrics.get("named_cache_hit_rates"))
    raw_named_cache_hit_rates = _mapping(metrics.get("raw_named_cache_hit_rates"))
    for cache_name, limit in resolved.min_named_cache_hit_rate.items():
        metric = f"named_cache_hit_rate.{cache_name}"
        check = _min_metric_check(
            named_cache_hit_rates,
            metric=cache_name,
            limit=limit,
            output_metric=metric,
            raw_value=raw_named_cache_hit_rates.get(cache_name),
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
            "phase_stats": _mapping(metrics.get("phase_stats")),
            "phase_p95_seconds": phase_p95_seconds,
            "phase_p99_seconds": phase_p99_seconds,
            "slowest_phase": metrics.get("slowest_phase"),
            "cache_hit_rate": metrics.get("cache_hit_rate"),
            "cache_summary": metrics.get("cache_summary"),
            "named_cache_hit_rates": named_cache_hit_rates,
            "route_cost_summary": metrics.get("route_cost_summary"),
            "mean_route_duration_seconds": metrics.get("mean_route_duration_seconds"),
            "p95_route_duration_seconds": metrics.get("p95_route_duration_seconds"),
            "p99_route_duration_seconds": metrics.get("p99_route_duration_seconds"),
            "max_route_duration_seconds": metrics.get("max_route_duration_seconds"),
            "mean_attempted_route_count": metrics.get("mean_attempted_route_count"),
            "route_budget_exhaustion_rate": metrics.get("route_budget_exhaustion_rate"),
            "route_budget_exhausted_count": metrics.get("route_budget_exhausted_count"),
            "unattempted_route_count": metrics.get("unattempted_route_count"),
            "retrieval_use_rate": metrics.get("retrieval_use_rate"),
            "retrieval_hit_count": metrics.get("retrieval_hit_count"),
            "mean_retrieval_hits": metrics.get("mean_retrieval_hits"),
            "verification_stage_summary": metrics.get("verification_stage_summary"),
            "verification_stage_enabled": metrics.get("verification_stage_enabled"),
            "verification_stage_skipped": metrics.get("verification_stage_skipped"),
            "verification_skip_rate": metrics.get("verification_skip_rate"),
            "selective_claim_skip_rate": metrics.get("selective_claim_skip_rate"),
            "verified_claim_count": metrics.get("verified_claim_count"),
            "verifier_saved_claim_count": metrics.get("verifier_saved_claim_count"),
        },
        "checks": checks,
        "failures": failures,
    }


def product_runtime_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    """Extract runtime-budget metrics from a ProductTrace or trace payload."""
    runtime_trace = _runtime_trace_payload(trace)
    if runtime_trace is None:
        metrics = {
            "has_runtime_trace": False,
            "total_seconds": None,
            "accounted_seconds": None,
            "unaccounted_seconds": None,
            "measured_phases": None,
            "phase_seconds": {},
            "raw_phase_seconds": {},
            "phase_counts": {},
            "phase_stats": {},
            "phase_p95_seconds": {},
            "raw_phase_p95_seconds": {},
            "phase_p99_seconds": {},
            "raw_phase_p99_seconds": {},
            "slowest_phase": None,
        }
        metrics.update(_cache_metrics(trace))
        metrics.update(_action_execution_metrics(trace))
        metrics.update(_action_receipt_metrics(trace))
        metrics.update(_action_audit_metrics(trace))
        metrics.update(_trajectory_audit_metrics(trace))
        metrics.update(_route_cost_metrics(trace))
        metrics.update(_verification_stage_metrics(trace))
        metrics.update(_verification_plan_metrics(trace))
        metrics.update(_claim_risk_localization_metrics(trace))
        metrics.update(_triple_coverage_metrics(trace))
        metrics.update(_world_model_metrics(trace))
        metrics.update(_context_sensitivity_metrics(trace))
        metrics.update(_counterfactual_robustness_metrics(trace))
        metrics.update(_final_answer_metrics(trace))
        metrics.update(_promotion_contract_metrics(trace))
        return metrics
    summary = _runtime_summary(runtime_trace)
    phase_seconds = _mapping(summary.get("phase_seconds"))
    phase_p95_seconds = _mapping(summary.get("phase_p95_seconds"))
    phase_p99_seconds = _mapping(summary.get("phase_p99_seconds"))
    metrics = {
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
        "phase_stats": _mapping(summary.get("phase_stats")),
        "phase_p95_seconds": {
            str(name): _finite_float(value)
            for name, value in phase_p95_seconds.items()
        },
        "raw_phase_p95_seconds": dict(phase_p95_seconds),
        "phase_p99_seconds": {
            str(name): _finite_float(value)
            for name, value in phase_p99_seconds.items()
        },
        "raw_phase_p99_seconds": dict(phase_p99_seconds),
        "slowest_phase": summary.get("slowest_phase"),
    }
    metrics.update(_cache_metrics(trace))
    metrics.update(_action_execution_metrics(trace))
    metrics.update(_action_receipt_metrics(trace))
    metrics.update(_action_audit_metrics(trace))
    metrics.update(_trajectory_audit_metrics(trace))
    metrics.update(_route_cost_metrics(trace))
    metrics.update(_verification_stage_metrics(trace))
    metrics.update(_verification_plan_metrics(trace))
    metrics.update(_claim_risk_localization_metrics(trace))
    metrics.update(_triple_coverage_metrics(trace))
    metrics.update(_world_model_metrics(trace))
    metrics.update(_context_sensitivity_metrics(trace))
    metrics.update(_counterfactual_robustness_metrics(trace))
    metrics.update(_final_answer_metrics(trace))
    metrics.update(_promotion_contract_metrics(trace))
    return metrics


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
        merged = dict(summary)
        if (
            "phase_stats" not in merged
            or "phase_p95_seconds" not in merged
            or "phase_p99_seconds" not in merged
        ):
            try:
                derived = RuntimeTrace.from_dict(runtime_trace).summary()
            except (KeyError, TypeError, ValueError):
                return merged
            for key in ("phase_stats", "phase_p95_seconds", "phase_p99_seconds"):
                if key not in merged:
                    merged[key] = derived.get(key)
        return merged
    return RuntimeTrace.from_dict(runtime_trace).summary()


def _cache_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, ProductTrace):
        summary = trace.cache_summary()
    else:
        payload = dict(trace)
        metadata = payload.get("metadata", {})
        summary = ProductTrace(metadata=metadata if isinstance(metadata, Mapping) else {}).cache_summary()
    aggregate = _mapping(summary.get("aggregate"))
    caches = _mapping(summary.get("caches"))
    named_hit_rates = {
        str(name): _finite_float(_mapping(stats).get("hit_rate"))
        for name, stats in caches.items()
    }
    raw_named_hit_rates = {
        str(name): _mapping(stats).get("hit_rate")
        for name, stats in caches.items()
    }
    return {
        "cache_hit_rate": _finite_float(aggregate.get("hit_rate")),
        "cache_summary": summary,
        "named_cache_hit_rates": named_hit_rates,
        "raw_named_cache_hit_rates": raw_named_hit_rates,
    }


def _action_audit_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, ProductTrace):
        summary = trace.action_audit_summary()
        source = "full_trace"
    else:
        payload = dict(trace)
        summary = _mapping(_mapping(payload.get("summaries")).get("action_audit"))
        if summary:
            source = "bounded_summary"
        else:
            summary = ProductTrace(
                actions=tuple(_sequence(payload.get("actions", ()))),
                risk_decision=_mapping(payload.get("risk_decision")),
                verification_plan=_mapping(payload.get("verification_plan")),
            ).action_audit_summary()
            source = "full_trace"
    counts_by_code = _mapping(summary.get("counts_by_code"))
    malformed_payload_count = sum(
        _finite_float(counts_by_code.get(code)) or 0.0
        for code in (
            "malformed_action_payload",
            "malformed_retrieval_payload",
            "malformed_tool_payload",
            "malformed_tool_arguments",
        )
    )
    return {
        "action_audit_available": bool(summary.get("available")),
        "action_audit_source": source,
        "action_audit_summary": summary,
        "action_audit_passed": _optional_bool(summary.get("passed")),
        "action_audit_issue_count": _finite_float(summary.get("issue_count")),
        "action_audit_error_count": _finite_float(summary.get("error_count")),
        "action_audit_warning_count": _finite_float(summary.get("warning_count")),
        "action_audit_missing_decision_action_count": _finite_float(
            counts_by_code.get("missing_decision_action")
        ) or 0.0,
        "action_audit_missing_retrieval_action_count": _finite_float(
            counts_by_code.get("missing_retrieval_action")
        ) or 0.0,
        "action_audit_missing_plan_retrieval_query_count": _finite_float(
            counts_by_code.get("missing_plan_retrieval_query")
        ) or 0.0,
        "action_audit_malformed_payload_count": malformed_payload_count,
        "action_audit_unexpected_action_count": _finite_float(
            counts_by_code.get("unexpected_action_for_decision")
        ) or 0.0,
        "action_audit_unknown_claim_id_count": _finite_float(counts_by_code.get("unknown_claim_id")) or 0.0,
    }


def _trajectory_audit_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    from eigentruth.control.trajectory_audit import TrajectoryHallucinationType, audit_product_trace_trajectory

    if isinstance(trace, ProductTrace):
        summary = trace.trajectory_audit_summary()
        source = "full_trace"
    else:
        payload = dict(trace)
        summary = _mapping(_mapping(payload.get("summaries")).get("trajectory_audit"))
        if summary:
            source = "bounded_summary"
        else:
            summary = audit_product_trace_trajectory(payload).summary()
            source = "full_trace"
    counts_by_type = _mapping(summary.get("counts_by_type"))
    counts_by_code = _mapping(summary.get("counts_by_code"))
    metrics = {
        "trajectory_audit_available": bool(summary.get("available")),
        "trajectory_audit_source": source,
        "trajectory_audit_summary": summary,
        "trajectory_audit_passed": _optional_bool(summary.get("passed")),
        "trajectory_audit_issue_count": _finite_float(summary.get("issue_count")),
        "trajectory_audit_error_count": _finite_float(summary.get("error_count")),
        "trajectory_audit_warning_count": _finite_float(summary.get("warning_count")),
        "trajectory_audit_info_count": _finite_float(summary.get("info_count")),
        "trajectory_audit_types": tuple(_sequence(summary.get("hallucination_types", ()))),
        "trajectory_audit_counts_by_type": counts_by_type,
        "trajectory_audit_counts_by_code": counts_by_code,
    }
    for hallucination_type in TrajectoryHallucinationType:
        metrics[f"trajectory_audit_{hallucination_type.value}_count"] = (
            _finite_float(counts_by_type.get(hallucination_type.value)) or 0.0
        )
    return metrics


def _action_execution_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, ProductTrace):
        summary = trace.action_execution_summary()
        source = "full_trace"
    else:
        payload = dict(trace)
        summary = _mapping(_mapping(payload.get("summaries")).get("action_execution"))
        if summary:
            source = "bounded_summary"
        else:
            summary = ProductTrace(
                actions=tuple(_sequence(payload.get("actions", ()))),
                action_results=tuple(_sequence(payload.get("action_results", ()))),
            ).action_execution_summary()
            source = "full_trace"
    alignment = _mapping(summary.get("alignment"))
    return {
        "action_execution_available": bool(summary.get("total") or summary.get("planned_action_count")),
        "action_execution_source": source,
        "action_execution_summary": summary,
        "action_execution_alignment_passed": _optional_bool(summary.get("alignment_passed")),
        "action_execution_planned_action_count": _finite_float(summary.get("planned_action_count")) or 0.0,
        "action_execution_result_count": _finite_float(summary.get("result_count")) or 0.0,
        "action_execution_missing_result_count": _finite_float(
            summary.get("missing_result_count")
        ) or 0.0,
        "action_execution_unexpected_result_count": _finite_float(
            summary.get("unexpected_result_count")
        ) or 0.0,
        "action_execution_request_id_mismatch_count": _finite_float(
            summary.get("request_id_mismatch_count")
        ) or 0.0,
        "action_execution_alignment_available": bool(alignment.get("available")),
    }


def _action_receipt_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, ProductTrace):
        summary = trace.action_receipt_summary()
        source = "full_trace"
    else:
        payload = dict(trace)
        summary = _mapping(_mapping(payload.get("summaries")).get("action_receipts"))
        if summary:
            source = "bounded_summary"
        else:
            summary = action_receipt_summary_from_results(tuple(_sequence(payload.get("action_results", ()))))
            source = "full_trace"
    return {
        "action_receipts_available": bool(summary.get("available")),
        "action_receipts_source": source,
        "action_receipts_summary": summary,
        "action_receipts_passed": _optional_bool(summary.get("passed")),
        "action_receipts_result_count": _finite_float(summary.get("result_count")) or 0.0,
        "action_receipts_receipt_count": _finite_float(summary.get("receipt_count")) or 0.0,
        "action_receipts_missing_receipt_count": _finite_float(summary.get("missing_receipt_count")) or 0.0,
        "action_receipts_signed_receipt_count": _finite_float(summary.get("signed_receipt_count")) or 0.0,
        "action_receipts_unsigned_receipt_count": _finite_float(summary.get("unsigned_receipt_count")) or 0.0,
        "action_receipts_invalid_receipt_count": _finite_float(summary.get("invalid_receipt_count")) or 0.0,
        "action_receipts_fingerprint_match_count": _finite_float(
            summary.get("fingerprint_match_count")
        ) or 0.0,
        "action_receipts_fingerprint_mismatch_count": _finite_float(
            summary.get("fingerprint_mismatch_count")
        ) or 0.0,
        "action_receipts_coverage": _finite_float(summary.get("coverage")),
    }


def _route_cost_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, ProductTrace):
        summary = trace.verification_route_cost_summary()
    else:
        payload = dict(trace)
        summary = ProductTrace(
            verification_results=tuple(_sequence(payload.get("verification_results", ()))),
        ).verification_route_cost_summary()
    return {
        "route_cost_summary": summary,
        "mean_route_duration_seconds": _finite_float(summary.get("mean_duration_seconds")),
        "p95_route_duration_seconds": _finite_float(summary.get("p95_duration_seconds")),
        "p99_route_duration_seconds": _finite_float(summary.get("p99_duration_seconds")),
        "max_route_duration_seconds": _finite_float(summary.get("max_duration_seconds")),
        "mean_attempted_route_count": _finite_float(summary.get("mean_attempted_route_count")),
        "route_budget_exhaustion_rate": _finite_float(summary.get("route_budget_exhaustion_rate")),
        "route_budget_exhausted_count": _finite_float(summary.get("route_budget_exhausted_count")),
        "unattempted_route_count": _finite_float(summary.get("unattempted_route_count")),
        "retrieval_use_rate": _finite_float(summary.get("retrieval_use_rate")),
        "retrieval_hit_count": _finite_float(summary.get("retrieval_hit_count")),
        "mean_retrieval_hits": _finite_float(summary.get("mean_retrieval_hits")),
    }


def _verification_stage_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, ProductTrace):
        summary = trace.verification_stage_summary()
    else:
        payload = dict(trace)
        summary = ProductTrace(
            claims=tuple(_sequence(payload.get("claims", ()))),
            verification_results=tuple(_sequence(payload.get("verification_results", ()))),
            events=tuple(_sequence(payload.get("events", ()))),
            metadata=_mapping(payload.get("metadata", {})),
        ).verification_stage_summary()
    return {
        "verification_stage_summary": summary,
        "verification_stage_enabled": bool(summary.get("enabled")),
        "verification_stage_skipped": bool(summary.get("skipped")),
        "verification_skip_rate": _finite_float(summary.get("skip_rate")),
        "selective_claim_skip_rate": (
            _finite_float(summary.get("skip_rate"))
            if str(summary.get("verification_scope", "")).strip().lower() == "triggered"
            else None
        ),
        "verified_claim_count": _finite_float(summary.get("verified_claim_count")),
        "verifier_saved_claim_count": _finite_float(summary.get("saved_claim_count")),
    }


def _verification_plan_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, ProductTrace):
        payload = trace.to_dict()
    else:
        payload = dict(trace)

    plan = _mapping(payload.get("verification_plan"))
    source = "full_trace" if plan else None
    if not plan:
        summary_plan = _mapping(_mapping(payload.get("summaries")).get("verification_plan"))
        if summary_plan:
            plan = summary_plan
            source = "bounded_summary"

    if not plan or plan.get("available") is False:
        return {
            "verification_plan_available": False,
            "verification_plan_source": source or "missing",
            "verification_plan_summary": {
                "available": False,
                "source": source or "missing",
            },
            "verification_plan_run_verifier": None,
            "verification_plan_scope": None,
            "verification_plan_claim_count": None,
            "verification_plan_verify_claim_count": None,
            "verification_plan_skipped_claim_count": None,
            "verification_plan_triggered_claim_count": None,
            "verification_plan_route_hint_count": None,
            "verification_plan_route_counts": {},
            "verification_plan_retrieval_query_count": None,
            "verification_plan_calculation_check_count": None,
            "verification_plan_state_check_count": None,
            "verification_plan_world_model_check_count": None,
            "verification_plan_dependency_count": None,
            "verification_plan_budget_enabled": None,
            "verification_plan_budget_claim_budget_exhausted": None,
            "verification_plan_budget_route_budget_exhausted": None,
            "verification_plan_budget_tool_payload_budget_exhausted": None,
            "verification_plan_budget_estimated_cost_budget_exhausted": None,
            "verification_plan_budget_selected_claim_count": None,
            "verification_plan_budget_dropped_claim_count": None,
        }

    if source == "bounded_summary":
        route_counts = _int_mapping(plan.get("route_counts"))
        tool_counts = _mapping(plan.get("tool_payload_counts"))
        claim_count = _finite_float(plan.get("claim_count"))
        verify_claim_count = _finite_float(plan.get("verify_claim_count"))
        skipped_claim_count = _finite_float(plan.get("skipped_claim_count"))
        triggered_claim_count = _finite_float(plan.get("triggered_claim_count"))
        route_hint_count = None
        retrieval_query_count = _finite_float(tool_counts.get("retrieval_queries"))
        calculation_check_count = _finite_float(tool_counts.get("calculation_checks"))
        state_check_count = _finite_float(tool_counts.get("state_checks"))
        world_model_check_count = _finite_float(tool_counts.get("world_model_checks"))
        dependency_count = _finite_float(plan.get("dependency_count"))
    else:
        route_counts = _route_counts_from_plan(plan)
        claim_count = float(len(_sequence(plan.get("claims"))))
        verify_claim_count = float(len(_sequence(plan.get("verify_claim_ids"))))
        skipped_claim_count = float(len(_sequence(plan.get("skipped_claim_ids"))))
        triggered_claim_count = float(len(_sequence(plan.get("triggered_claim_ids"))))
        route_hint_count = float(len(_sequence(plan.get("route_hints"))))
        retrieval_query_count = float(len(_sequence(plan.get("retrieval_queries"))))
        calculation_check_count = float(len(_sequence(plan.get("calculation_checks"))))
        state_check_count = float(len(_sequence(plan.get("state_checks"))))
        world_model_check_count = float(len(_sequence(plan.get("world_model_checks"))))
        dependency_count = float(len(_sequence(plan.get("dependencies"))))
    budget = _mapping(plan.get("budget"))
    budget_enabled = _optional_bool(budget.get("enabled"))
    if budget_enabled is None and budget:
        budget_enabled = True
    selected_budget_claim_count = _finite_float(budget.get("selected_claim_count"))
    if selected_budget_claim_count is None and budget:
        selected_budget_claim_count = float(len(_sequence(budget.get("selected_claim_ids"))))
    dropped_budget_claim_count = _finite_float(budget.get("dropped_claim_count"))
    if dropped_budget_claim_count is None and budget:
        dropped_budget_claim_count = float(len(_sequence(budget.get("dropped_claim_ids"))))
    claim_budget_exhausted = _optional_bool(budget.get("claim_budget_exhausted"))
    route_budget_exhausted = _optional_bool(budget.get("route_budget_exhausted"))
    tool_payload_budget_exhausted = _optional_bool(budget.get("tool_payload_budget_exhausted"))
    estimated_cost_budget_exhausted = _optional_bool(budget.get("estimated_cost_budget_exhausted"))

    summary = {
        "available": True,
        "source": source or "full_trace",
        "run_verifier": plan.get("run_verifier"),
        "verification_scope": plan.get("verification_scope"),
        "claim_count": claim_count,
        "verify_claim_count": verify_claim_count,
        "skipped_claim_count": skipped_claim_count,
        "triggered_claim_count": triggered_claim_count,
        "route_hint_count": route_hint_count,
        "route_counts": route_counts,
        "tool_payload_counts": {
            "retrieval_queries": retrieval_query_count,
            "calculation_checks": calculation_check_count,
            "state_checks": state_check_count,
            "world_model_checks": world_model_check_count,
        },
        "dependency_count": dependency_count,
        "budget": budget,
    }
    return {
        "verification_plan_available": True,
        "verification_plan_source": source or "full_trace",
        "verification_plan_summary": summary,
        "verification_plan_run_verifier": (
            plan.get("run_verifier")
            if isinstance(plan.get("run_verifier"), bool)
            else None
        ),
        "verification_plan_scope": plan.get("verification_scope"),
        "verification_plan_claim_count": claim_count,
        "verification_plan_verify_claim_count": verify_claim_count,
        "verification_plan_skipped_claim_count": skipped_claim_count,
        "verification_plan_triggered_claim_count": triggered_claim_count,
        "verification_plan_route_hint_count": route_hint_count,
        "verification_plan_route_counts": route_counts,
        "verification_plan_retrieval_query_count": retrieval_query_count,
        "verification_plan_calculation_check_count": calculation_check_count,
        "verification_plan_state_check_count": state_check_count,
        "verification_plan_world_model_check_count": world_model_check_count,
        "verification_plan_dependency_count": dependency_count,
        "verification_plan_budget_enabled": budget_enabled,
        "verification_plan_budget_claim_budget_exhausted": claim_budget_exhausted,
        "verification_plan_budget_route_budget_exhausted": route_budget_exhausted,
        "verification_plan_budget_tool_payload_budget_exhausted": tool_payload_budget_exhausted,
        "verification_plan_budget_estimated_cost_budget_exhausted": estimated_cost_budget_exhausted,
        "verification_plan_budget_selected_claim_count": selected_budget_claim_count,
        "verification_plan_budget_dropped_claim_count": dropped_budget_claim_count,
    }


def _claim_risk_localization_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, ProductTrace):
        summary = trace.claim_risk_localization_summary()
        source = "full_trace"
    else:
        payload = dict(trace)
        summary = _mapping(_mapping(payload.get("summaries")).get("claim_risk_localization"))
        if summary:
            source = "bounded_summary"
        else:
            summary = ProductTrace(
                claims=tuple(_sequence(payload.get("claims", ()))),
                verification_results=tuple(_sequence(payload.get("verification_results", ()))),
                verification_plan=_mapping(payload.get("verification_plan")),
            ).claim_risk_localization_summary()
            source = "full_trace"
    counts_by_risk_level = _mapping(summary.get("counts_by_risk_level"))
    return {
        "claim_risk_localization_available": bool(summary.get("available")),
        "claim_risk_localization_source": source,
        "claim_risk_localization_summary": summary,
        "claim_risk_span_count": _finite_float(summary.get("span_count")),
        "claim_risk_localized_span_count": _finite_float(summary.get("localized_span_count")),
        "claim_risk_high_count": _finite_float(summary.get("high_risk_claim_count")),
        "claim_risk_medium_or_high_count": _finite_float(summary.get("medium_or_high_risk_claim_count")),
        "claim_risk_entity_claim_count": _finite_float(summary.get("entity_claim_count")),
        "claim_risk_entity_candidate_count": _finite_float(summary.get("entity_candidate_count")),
        "claim_risk_unique_entity_candidate_count": _finite_float(
            summary.get("unique_entity_candidate_count")
        ),
        "claim_risk_high_entity_claim_count": _finite_float(summary.get("high_risk_entity_claim_count")),
        "claim_risk_high_entity_candidate_count": _finite_float(
            summary.get("high_risk_entity_candidate_count")
        ),
        "claim_risk_medium_or_high_entity_candidate_count": _finite_float(
            summary.get("medium_or_high_entity_candidate_count")
        ),
        "claim_risk_counts_by_entity_candidate": _int_mapping(
            summary.get("counts_by_entity_candidate")
        ),
        "claim_risk_high_counts_by_entity_candidate": _int_mapping(
            summary.get("high_risk_counts_by_entity_candidate")
        ),
        "claim_risk_medium_or_high_counts_by_entity_candidate": _int_mapping(
            summary.get("medium_or_high_counts_by_entity_candidate")
        ),
        "claim_risk_low_count": _finite_float(counts_by_risk_level.get("low")),
        "claim_risk_medium_count": _finite_float(counts_by_risk_level.get("medium")),
        "claim_risk_max_score": _finite_float(summary.get("max_risk_score")),
    }


def _triple_coverage_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, ProductTrace):
        summary = trace.triple_coverage_summary()
        source = "full_trace"
    else:
        payload = dict(trace)
        summary = _mapping(_mapping(payload.get("summaries")).get("triple_coverage"))
        if summary:
            source = "bounded_summary"
        else:
            summary = _metadata_triple_coverage_summary(payload)
            if summary:
                source = "metadata_summary"
            else:
                summary = ProductTrace(
                    claims=tuple(_sequence(payload.get("claims", ()))),
                    verification_results=tuple(_sequence(payload.get("verification_results", ()))),
                ).triple_coverage_summary()
                source = "full_trace"
    return {
        "triple_coverage_summary": summary,
        "triple_coverage_source": source,
        "triple_claim_count": _finite_float(summary.get("claim_triple_count")),
        "triple_claim_coverage_rate": _finite_float(summary.get("claim_triple_coverage_rate")),
        "triple_audit_available": _optional_bool(summary.get("audit_available")),
        "triple_audit_report_count": _finite_float(summary.get("audit_report_count")),
        "triple_audit_claim_covered_count": _finite_float(
            summary.get("audit_claim_covered_count")
        ),
        "triple_audit_claim_coverage_rate": _finite_float(summary.get("audit_claim_coverage_rate")),
        "triple_audit_triple_count": _finite_float(summary.get("audit_triple_count")),
        "triple_audit_pass_rate": _finite_float(summary.get("audit_pass_rate")),
        "triple_slot_coverage_rate": _finite_float(summary.get("slot_coverage_rate")),
        "triple_structured_fact_result_count": _finite_float(
            summary.get("structured_fact_result_count")
        ),
        "triple_claim_predicate_counts": _int_mapping(summary.get("claim_predicate_counts")),
        "triple_audit_predicate_counts": _int_mapping(summary.get("audit_predicate_counts")),
        "triple_missing_slot_counts": _int_mapping(summary.get("missing_slot_counts")),
        "triple_covered_slot_counts": _int_mapping(summary.get("covered_slot_counts")),
        "triple_structured_fact_status_counts": _int_mapping(
            summary.get("structured_fact_status_counts")
        ),
        "triple_structured_fact_predicate_counts": _int_mapping(
            summary.get("structured_fact_predicate_counts")
        ),
    }


def _metadata_triple_coverage_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _mapping(payload.get("metadata"))
    trace_corpus = _mapping(metadata.get("trace_corpus"))
    return _mapping(trace_corpus.get("triple_coverage_summary")) or _mapping(
        metadata.get("triple_coverage_summary")
    )


def _world_model_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, ProductTrace):
        summary = trace.world_model_summary()
        source = "full_trace"
    else:
        payload = dict(trace)
        summary = _mapping(_mapping(payload.get("summaries")).get("world_model"))
        if summary:
            source = "bounded_summary"
        else:
            summary = _metadata_world_model_summary(payload)
            if summary:
                source = "metadata_summary"
            else:
                summary = ProductTrace(
                    verification_results=tuple(_sequence(payload.get("verification_results", ()))),
                ).world_model_summary()
                source = "full_trace"
    return {
        "world_model_summary": summary,
        "world_model_source": source,
        "world_model_total": _finite_float(summary.get("world_model_total")),
        "world_model_coverage_rate": _finite_float(summary.get("coverage_rate")),
        "world_model_conflict_count": _finite_float(summary.get("conflict_count")),
        "world_model_conflict_rate": _finite_float(summary.get("conflict_rate")),
        "world_model_low_agreement_count": _finite_float(summary.get("low_agreement_count")),
        "world_model_low_agreement_rate": _finite_float(summary.get("low_agreement_rate")),
        "world_model_no_rule_matched_count": _finite_float(summary.get("no_rule_matched_count")),
        "world_model_trace_gap_count": _finite_float(summary.get("trace_gap_count")),
        "world_model_trace_gap_rate": _finite_float(summary.get("trace_gap_rate")),
        "world_model_traceable": _optional_bool(summary.get("traceable")),
        "world_model_prediction_confidence_mean": _finite_float(
            summary.get("prediction_confidence_mean")
        ),
        "world_model_prediction_confidence_min": _finite_float(
            summary.get("prediction_confidence_min")
        ),
        "world_model_agreement_rate_mean": _finite_float(summary.get("agreement_rate_mean")),
        "world_model_agreement_rate_min": _finite_float(summary.get("agreement_rate_min")),
        "world_model_counts_by_adapter": _int_mapping(summary.get("counts_by_adapter")),
        "world_model_counts_by_reference_id": _int_mapping(summary.get("counts_by_reference_id")),
        "world_model_counts_by_decision_rule": _int_mapping(summary.get("counts_by_decision_rule")),
        "world_model_conflict_paths": _int_mapping(summary.get("conflict_paths")),
    }


def _metadata_world_model_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _mapping(payload.get("metadata"))
    trace_corpus = _mapping(metadata.get("trace_corpus"))
    return _mapping(trace_corpus.get("world_model_summary")) or _mapping(
        metadata.get("world_model_summary")
    )


def _context_sensitivity_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, ProductTrace):
        summary = trace.context_sensitivity_summary()
        source = "full_trace"
    else:
        payload = dict(trace)
        summary = _mapping(_mapping(payload.get("summaries")).get("context_sensitivity"))
        if summary:
            source = "bounded_summary"
        else:
            summary = _metadata_context_sensitivity_summary(payload)
            if summary:
                source = "metadata_summary"
            else:
                summary = ProductTrace(
                    verification_results=tuple(_sequence(payload.get("verification_results", ()))),
                ).context_sensitivity_summary()
                source = "full_trace"
    return {
        "context_sensitivity_summary": summary,
        "context_sensitivity_source": source,
        "context_sensitivity_total": _finite_float(summary.get("context_sensitivity_total")),
        "context_sensitivity_coverage_rate": _finite_float(summary.get("coverage_rate")),
        "context_sensitivity_flagged_result_count": _finite_float(
            summary.get("flagged_result_count")
        ),
        "context_sensitivity_flagged_result_rate": _finite_float(
            summary.get("flagged_result_rate")
        ),
        "context_sensitivity_max_flagged_rate": _finite_float(summary.get("max_flagged_rate")),
        "context_sensitivity_mean_flagged_rate": _finite_float(summary.get("mean_flagged_rate")),
        "context_sensitivity_max_unsupported_context_shift": _finite_float(
            summary.get("max_unsupported_context_shift")
        ),
        "context_sensitivity_mean_unsupported_context_shift": _finite_float(
            summary.get("mean_unsupported_context_shift")
        ),
        "context_sensitivity_max_context_sensitivity_ratio": _finite_float(
            summary.get("max_context_sensitivity_ratio")
        ),
        "context_sensitivity_trace_gap_count": _finite_float(summary.get("trace_gap_count")),
        "context_sensitivity_trace_gap_rate": _finite_float(summary.get("trace_gap_rate")),
        "context_sensitivity_traceable": _optional_bool(summary.get("traceable")),
        "context_sensitivity_counts_by_source": _int_mapping(summary.get("counts_by_source")),
        "context_sensitivity_counts_by_status": _int_mapping(summary.get("counts_by_status")),
    }


def _metadata_context_sensitivity_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _mapping(payload.get("metadata"))
    trace_corpus = _mapping(metadata.get("trace_corpus"))
    return _mapping(trace_corpus.get("context_sensitivity_summary")) or _mapping(
        metadata.get("context_sensitivity_summary")
    )


def _counterfactual_robustness_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, ProductTrace):
        summary = trace.counterfactual_robustness_summary()
        source = "full_trace"
    else:
        payload = dict(trace)
        summary = _mapping(_mapping(payload.get("summaries")).get("counterfactual_robustness"))
        if summary:
            source = "bounded_summary"
        else:
            summary = _metadata_counterfactual_robustness_summary(payload)
            if summary:
                source = "metadata_summary"
            else:
                summary = ProductTrace(
                    verification_results=tuple(_sequence(payload.get("verification_results", ())))
                ).counterfactual_robustness_summary()
                source = "full_trace"
    return {
        "counterfactual_robustness_summary": summary,
        "counterfactual_robustness_source": source,
        "counterfactual_robustness_result_total": _finite_float(
            summary.get("counterfactual_result_total")
        ),
        "counterfactual_robustness_probe_total": _finite_float(
            summary.get("counterfactual_probe_total")
        ),
        "counterfactual_robustness_entity_probe_count": _finite_float(
            summary.get("entity_probe_count")
        ),
        "counterfactual_robustness_entity_candidate_count": _finite_float(
            summary.get("entity_candidate_count")
        ),
        "counterfactual_robustness_coverage_rate": _finite_float(summary.get("coverage_rate")),
        "counterfactual_robustness_pass_rate": _finite_float(summary.get("pass_rate")),
        "counterfactual_robustness_passed_count": _finite_float(summary.get("passed_count")),
        "counterfactual_robustness_failed_count": _finite_float(summary.get("failed_count")),
        "counterfactual_robustness_expected_flip_count": _finite_float(
            summary.get("expected_flip_count")
        ),
        "counterfactual_robustness_flip_success_count": _finite_float(
            summary.get("flip_success_count")
        ),
        "counterfactual_robustness_flip_success_rate": _finite_float(
            summary.get("flip_success_rate")
        ),
        "counterfactual_robustness_false_invariance_count": _finite_float(
            summary.get("false_invariance_count")
        ),
        "counterfactual_robustness_false_invariance_rate": _finite_float(
            summary.get("false_invariance_rate")
        ),
        "counterfactual_robustness_unexpected_flip_count": _finite_float(
            summary.get("unexpected_flip_count")
        ),
        "counterfactual_robustness_unexpected_flip_rate": _finite_float(
            summary.get("unexpected_flip_rate")
        ),
        "counterfactual_robustness_trace_gap_count": _finite_float(summary.get("trace_gap_count")),
        "counterfactual_robustness_trace_gap_rate": _finite_float(summary.get("trace_gap_rate")),
        "counterfactual_robustness_traceable": _optional_bool(summary.get("traceable")),
        "counterfactual_robustness_counts_by_source": _int_mapping(
            summary.get("counts_by_source")
        ),
        "counterfactual_robustness_counts_by_status": _int_mapping(
            summary.get("counts_by_status")
        ),
        "counterfactual_robustness_counts_by_probe_type": _int_mapping(
            summary.get("counts_by_probe_type")
        ),
        "counterfactual_robustness_counts_by_failure_reason": _int_mapping(
            summary.get("counts_by_failure_reason")
        ),
        "counterfactual_robustness_counts_by_entity_candidate": _int_mapping(
            summary.get("counts_by_entity_candidate")
        ),
        "counterfactual_robustness_false_invariance_by_entity_candidate": _int_mapping(
            summary.get("false_invariance_by_entity_candidate")
        ),
        "counterfactual_robustness_counts_by_entity_source_kind": _int_mapping(
            summary.get("counts_by_entity_source_kind")
        ),
    }


def _metadata_counterfactual_robustness_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _mapping(payload.get("metadata"))
    trace_corpus = _mapping(metadata.get("trace_corpus"))
    return _mapping(trace_corpus.get("counterfactual_robustness_summary")) or _mapping(
        metadata.get("counterfactual_robustness_summary")
    )


def _final_answer_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, ProductTrace):
        payload = trace.to_dict()
    else:
        payload = dict(trace)

    summary = _mapping(_mapping(payload.get("summaries")).get("final_answer"))
    answer = _mapping(payload.get("final_answer"))
    source = "full_trace" if answer else None
    if payload.get("trace_format") == "bounded_product_trace" and summary:
        answer = summary
        source = "bounded_summary"
    elif not answer and summary:
        answer = summary
        source = "bounded_summary"

    if not answer or answer.get("available") is False:
        return {
            "final_answer_available": False,
            "final_answer_source": source or "missing",
            "final_answer_summary": {
                "available": False,
                "source": source or "missing",
            },
            "final_answer_status": None,
            "final_answer_action": None,
            "final_answer_risk_level": None,
            "final_answer_answerable": None,
            "final_answer_confidence": None,
            "final_answer_evidence_count": None,
            "final_answer_total_claims": None,
            "final_answer_blocked_claim_count": None,
            "final_answer_requires_followup": None,
        }

    claim_summary = _mapping(answer.get("claim_summary"))
    status_counts = _mapping(claim_summary.get("status_counts"))
    followup = _mapping(answer.get("followup"))
    evidence = _sequence(answer.get("evidence"))
    total_claims = _finite_float(
        answer.get("total_claims")
        if source == "bounded_summary"
        else claim_summary.get("total_claims")
    )
    blocked_claim_count = _finite_float(
        answer.get("blocked_claim_count")
        if source == "bounded_summary"
        else len(_sequence(claim_summary.get("blocked_claims")))
    )
    evidence_count = _finite_float(
        answer.get("evidence_count") if source == "bounded_summary" else len(evidence)
    )
    requires_followup = (
        answer.get("requires_followup")
        if source == "bounded_summary"
        else followup.get("requires_followup")
    )
    summary = {
        "available": True,
        "source": source or "full_trace",
        "status": answer.get("status"),
        "action": answer.get("action"),
        "risk_level": answer.get("risk_level"),
        "answerable": _optional_bool(answer.get("answerable")),
        "confidence": _finite_float(answer.get("confidence")),
        "evidence_count": evidence_count,
        "total_claims": total_claims,
        "blocked_claim_count": blocked_claim_count,
        "supported_claim_count": _finite_float(
            answer.get("supported_claim_count") if source == "bounded_summary" else status_counts.get("supported")
        ),
        "refuted_claim_count": _finite_float(
            answer.get("refuted_claim_count") if source == "bounded_summary" else status_counts.get("refuted")
        ),
        "unsupported_claim_count": _finite_float(
            answer.get("unsupported_claim_count")
            if source == "bounded_summary"
            else status_counts.get("insufficient_evidence")
        ),
        "requires_followup": _optional_bool(requires_followup),
    }
    return {
        "final_answer_available": True,
        "final_answer_source": source or "full_trace",
        "final_answer_summary": summary,
        "final_answer_status": answer.get("status"),
        "final_answer_action": answer.get("action"),
        "final_answer_risk_level": answer.get("risk_level"),
        "final_answer_answerable": _optional_bool(answer.get("answerable")),
        "final_answer_confidence": _finite_float(answer.get("confidence")),
        "final_answer_evidence_count": evidence_count,
        "final_answer_total_claims": total_claims,
        "final_answer_blocked_claim_count": blocked_claim_count,
        "final_answer_requires_followup": _optional_bool(requires_followup),
    }


def _promotion_contract_metrics(trace: ProductTrace | Mapping[str, Any]) -> dict[str, Any]:
    payload = trace.to_dict() if isinstance(trace, ProductTrace) else dict(trace)
    metadata = _mapping(payload.get("metadata"))
    contract_metadata = _mapping(metadata.get("promotion_contract_metadata"))
    verifier_route = _mapping(metadata.get("promotion_contract_verifier_route"))
    nested_external_evidence = _mapping(
        metadata.get("promotion_contract_external_evidence_baseline_comparison")
    )
    external_evidence = (
        nested_external_evidence
        or _external_evidence_baseline_comparison_from_flat_metadata(metadata)
        or _external_evidence_baseline_comparison_from_flat_metadata(contract_metadata)
    )
    nested_pre_generation = _mapping(
        metadata.get("promotion_contract_pre_generation_probe_comparison")
    )
    pre_generation = (
        nested_pre_generation
        or _pre_generation_probe_comparison_from_flat_metadata(metadata)
        or _pre_generation_probe_comparison_from_flat_metadata(contract_metadata)
    )
    nested_claim_factuality = _mapping(
        metadata.get("promotion_contract_claim_factuality_probe_comparison")
    )
    claim_factuality = (
        nested_claim_factuality
        or _claim_factuality_probe_comparison_from_flat_metadata(metadata)
        or _claim_factuality_probe_comparison_from_flat_metadata(contract_metadata)
    )
    nested_matrix = _mapping(metadata.get("promotion_contract_triple_extraction_fixture_matrix"))
    matrix = nested_matrix or _matrix_from_flat_metadata(metadata) or _matrix_from_flat_metadata(
        contract_metadata
    )
    nested_counterfactual = _mapping(
        metadata.get("promotion_contract_counterfactual_verification")
    )
    counterfactual = (
        nested_counterfactual
        or _counterfactual_verification_from_flat_metadata(metadata)
        or _counterfactual_verification_from_flat_metadata(contract_metadata)
    )
    nested_pathway = _mapping(
        metadata.get("promotion_contract_pathway_intervention_workflow")
    )
    pathway = (
        nested_pathway
        or _pathway_intervention_workflow_from_flat_metadata(metadata)
        or _pathway_intervention_workflow_from_flat_metadata(contract_metadata)
    )
    covered_fact_scope = _covered_fact_scope_from_metadata(
        metadata,
        contract_metadata=contract_metadata,
        verifier_route=verifier_route,
    )
    recommended_property_rollups = _covered_fact_property_metric_rollups(
        _mapping(covered_fact_scope.get("recommended_route_property_metrics"))
    )
    required_property_rollups = _covered_fact_property_metric_rollups(
        _mapping(covered_fact_scope.get("required_route_baseline_property_metrics"))
    )
    robustness_property_rollups = _covered_fact_property_metric_rollups(
        _mapping(covered_fact_scope.get("structured_fact_robustness_property_metrics"))
    )
    runtime_drift = _promotion_contract_runtime_drift_from_metadata(
        metadata,
        contract_metadata=contract_metadata,
    )
    product_trace_replay = _promotion_contract_product_trace_replay_from_metadata(
        metadata,
        contract_metadata=contract_metadata,
    )
    evidence_handoff = _promotion_contract_evidence_handoff_from_metadata(
        metadata,
        contract_metadata=contract_metadata,
    )
    triple_audit_evidence = _promotion_contract_triple_audit_evidence_from_metadata(
        metadata,
        contract_metadata=contract_metadata,
    )
    nested_frontier_release_evidence = _mapping(
        metadata.get("promotion_contract_frontier_release_evidence")
    )
    frontier_release_evidence = {
        **_frontier_release_evidence_from_flat_metadata(contract_metadata),
        **_frontier_release_evidence_from_flat_metadata(metadata),
        **nested_frontier_release_evidence,
    }
    promotion_summary = _mapping(metadata.get("promotion_contract_promotion_summary"))
    promotion_summary_runtime = _mapping(promotion_summary.get("runtime"))
    promotion_summary_verifier_route = _mapping(
        promotion_summary.get("verifier_route")
    )
    promotion_summary_action_gates = _mapping(promotion_summary.get("action_gates"))
    manifest_verification = _mapping(
        _first_present(
            metadata.get("triple_extraction_fixture_matrix_manifest_verification"),
            contract_metadata.get("triple_extraction_fixture_matrix_manifest_verification"),
        )
    )
    source = _optional_string(metadata.get("promotion_contract_source"))
    source_status = _optional_string(metadata.get("promotion_contract_source_status"))
    budget_enabled = _optional_bool(metadata.get("promotion_contract_budget_enabled"))
    matrix_source = _optional_string(
        _first_present(
            matrix.get("source"),
            metadata.get("triple_extraction_fixture_matrix_source"),
            contract_metadata.get("triple_extraction_fixture_matrix_source"),
        )
    )
    matrix_status = _optional_string(
        _first_present(
            matrix.get("status"),
            metadata.get("triple_extraction_fixture_matrix_status"),
            contract_metadata.get("triple_extraction_fixture_matrix_status"),
        )
    )
    external_evidence_source = _optional_string(
        _first_present(
            external_evidence.get("source"),
            metadata.get("external_evidence_baseline_comparison_source"),
            contract_metadata.get("external_evidence_baseline_comparison_source"),
        )
    )
    external_evidence_status = _optional_string(
        _first_present(
            external_evidence.get("status"),
            metadata.get("external_evidence_baseline_comparison_status"),
            contract_metadata.get("external_evidence_baseline_comparison_status"),
        )
    )
    external_evidence_decision_status = _optional_string(
        _first_present(
            external_evidence.get("decision_status"),
            metadata.get("external_evidence_baseline_comparison_decision_status"),
            contract_metadata.get("external_evidence_baseline_comparison_decision_status"),
        )
    )
    external_evidence_available = bool(external_evidence)
    pre_generation_available = bool(pre_generation)
    claim_factuality_available = bool(claim_factuality)
    matrix_available = bool(matrix)
    available = bool(
        source is not None
        or budget_enabled is not None
        or metadata.get("promotion_contract_runtime") is not None
        or contract_metadata
        or external_evidence_available
        or pre_generation_available
        or claim_factuality_available
        or matrix_available
        or bool(counterfactual)
        or bool(pathway)
        or bool(runtime_drift.get("available"))
        or bool(evidence_handoff.get("available"))
        or bool(triple_audit_evidence.get("available"))
        or bool(frontier_release_evidence)
    )
    pre_generation_manifest_verification = _mapping(
        _first_present(
            metadata.get("pre_generation_probe_comparison_manifest_verification"),
            contract_metadata.get("pre_generation_probe_comparison_manifest_verification"),
        )
    )
    claim_factuality_manifest_verification = _mapping(
        _first_present(
            metadata.get("claim_factuality_probe_comparison_manifest_verification"),
            contract_metadata.get("claim_factuality_probe_comparison_manifest_verification"),
        )
    )
    counterfactual_manifest_verification = _mapping(
        _first_present(
            metadata.get("counterfactual_verification_manifest_verification"),
            contract_metadata.get("counterfactual_verification_manifest_verification"),
        )
    )
    pathway_manifest_verification = _mapping(
        _first_present(
            metadata.get("pathway_intervention_workflow_manifest_verification"),
            contract_metadata.get("pathway_intervention_workflow_manifest_verification"),
        )
    )
    pre_generation_best_run = _mapping(pre_generation.get("best_run"))
    claim_factuality_best_run = _mapping(claim_factuality.get("best_run"))
    counterfactual_available = bool(counterfactual)
    frontier_release_evidence_available = bool(frontier_release_evidence)
    frontier_release_evidence_run_names = _sequence(
        frontier_release_evidence.get("run_names")
    )
    summary = {
        "available": available,
        "source": source,
        "source_status": source_status,
        "budget_enabled": budget_enabled,
        "covered_fact_properties": covered_fact_scope,
        "covered_fact_property_rollups": {
            "recommended_route": recommended_property_rollups,
            "required_route_baseline": required_property_rollups,
            "structured_fact_robustness": robustness_property_rollups,
        },
        "product_trace_replay": product_trace_replay,
        "product_runtime_drift": runtime_drift,
        "evidence_handoff": evidence_handoff,
        "triple_audit_evidence": triple_audit_evidence,
        "frontier_release_evidence": {
            "available": frontier_release_evidence_available,
            "source": _optional_string(frontier_release_evidence.get("source")),
            "report": _optional_string(
                _first_present(
                    frontier_release_evidence.get("report_path"),
                    frontier_release_evidence.get("report"),
                )
            ),
            "manifest": _optional_string(
                _first_present(
                    frontier_release_evidence.get("manifest_path"),
                    frontier_release_evidence.get("manifest"),
                )
            ),
            "registry": _optional_string(frontier_release_evidence.get("registry")),
            "record": _optional_string(
                _first_present(
                    frontier_release_evidence.get("record_key"),
                    frontier_release_evidence.get("record"),
                )
            ),
            "status": _optional_string(frontier_release_evidence.get("status")),
            "workflow": _optional_string(frontier_release_evidence.get("workflow")),
            "report_status": _optional_string(
                frontier_release_evidence.get("report_status")
            ),
            "decision_status": _optional_string(
                frontier_release_evidence.get("decision_status")
            ),
            "verifier_track_status": _optional_string(
                frontier_release_evidence.get("verifier_track_status")
            ),
            "abstention_track_status": _optional_string(
                frontier_release_evidence.get("abstention_track_status")
            ),
            "multiple_testing_track_status": _optional_string(
                frontier_release_evidence.get("multiple_testing_track_status")
            ),
            "citation_batch_track_status": _optional_string(
                frontier_release_evidence.get("citation_batch_track_status")
            ),
            "frontier_rerun_rollup_track_status": _optional_string(
                frontier_release_evidence.get("frontier_rerun_rollup_track_status")
            ),
            "base_verifier_track_status": _optional_string(
                frontier_release_evidence.get("base_verifier_track_status")
            ),
            "base_abstention_track_status": _optional_string(
                frontier_release_evidence.get("base_abstention_track_status")
            ),
            "base_detectability_track_status": _optional_string(
                frontier_release_evidence.get("base_detectability_track_status")
            ),
            "base_multiple_testing_track_status": _optional_string(
                frontier_release_evidence.get("base_multiple_testing_track_status")
            ),
            "frontier_rerun_rollup_promoted_tracks": (
                list(
                    _sequence(
                        frontier_release_evidence.get(
                            "frontier_rerun_rollup_promoted_tracks"
                        )
                    )
                )
                or None
            ),
            "frontier_rerun_rollup_report_count": _finite_float(
                frontier_release_evidence.get("frontier_rerun_rollup_report_count")
            ),
            "frontier_rerun_rollup_candidate_count": _finite_float(
                frontier_release_evidence.get("frontier_rerun_rollup_candidate_count")
            ),
            "frontier_rerun_rollup_missing_report_count": _finite_float(
                frontier_release_evidence.get("frontier_rerun_rollup_missing_report_count")
            ),
            "frontier_rerun_rollup_invalid_report_count": _finite_float(
                frontier_release_evidence.get("frontier_rerun_rollup_invalid_report_count")
            ),
            "frontier_rerun_rollup_blocked_candidate_count": _finite_float(
                frontier_release_evidence.get("frontier_rerun_rollup_blocked_candidate_count")
            ),
            "frontier_rerun_rollup_promotion_ready_count": _finite_float(
                frontier_release_evidence.get("frontier_rerun_rollup_promotion_ready_count")
            ),
            "citation_batch_rollup_count": _finite_float(
                frontier_release_evidence.get("citation_batch_rollup_count")
            ),
            "citation_batch_expected_batch_count": _finite_float(
                frontier_release_evidence.get("citation_batch_expected_batch_count")
            ),
            "citation_batch_observed_batch_count": _finite_float(
                frontier_release_evidence.get("citation_batch_observed_batch_count")
            ),
            "citation_batch_missing_expected_batch_count": _finite_float(
                frontier_release_evidence.get("citation_batch_missing_expected_batch_count")
            ),
            "citation_batch_duplicate_batch_count": _finite_float(
                frontier_release_evidence.get("citation_batch_duplicate_batch_count")
            ),
            "citation_batch_unexpected_batch_count": _finite_float(
                frontier_release_evidence.get("citation_batch_unexpected_batch_count")
            ),
            "run_count": float(len(frontier_release_evidence_run_names))
            if frontier_release_evidence_run_names
            else None,
            "run_names": list(frontier_release_evidence_run_names)
            if frontier_release_evidence_run_names
            else None,
        },
        "pathway_intervention_workflow": {
            "available": bool(pathway),
            "source": _optional_string(
                _first_present(
                    pathway.get("source"),
                    metadata.get("pathway_intervention_workflow_source"),
                    contract_metadata.get("pathway_intervention_workflow_source"),
                )
            ),
            "report": _optional_string(
                _first_present(
                    pathway.get("report_path"),
                    pathway.get("report"),
                    metadata.get("pathway_intervention_workflow_report"),
                    contract_metadata.get("pathway_intervention_workflow_report"),
                )
            ),
            "manifest": _optional_string(
                _first_present(
                    pathway.get("manifest_path"),
                    pathway.get("manifest"),
                    metadata.get("pathway_intervention_workflow_manifest"),
                    contract_metadata.get("pathway_intervention_workflow_manifest"),
                )
            ),
            "registry": _optional_string(
                _first_present(
                    pathway.get("registry"),
                    metadata.get("pathway_intervention_workflow_registry"),
                    contract_metadata.get("pathway_intervention_workflow_registry"),
                )
            ),
            "record": _optional_string(
                _first_present(
                    pathway.get("record_key"),
                    pathway.get("record"),
                    metadata.get("pathway_intervention_workflow_record"),
                    metadata.get("pathway_intervention_workflow_registry_key"),
                    contract_metadata.get("pathway_intervention_workflow_record"),
                    contract_metadata.get("pathway_intervention_workflow_registry_key"),
                )
            ),
            "manifest_verified": _optional_bool(
                pathway_manifest_verification.get("passed")
            ),
            "status": _optional_string(
                _first_present(
                    pathway.get("status"),
                    metadata.get("pathway_intervention_workflow_status"),
                    contract_metadata.get("pathway_intervention_workflow_status"),
                )
            ),
            "report_status": _optional_string(
                _first_present(
                    pathway.get("report_status"),
                    metadata.get("pathway_intervention_workflow_report_status"),
                    contract_metadata.get("pathway_intervention_workflow_report_status"),
                )
            ),
            "release_ready": _optional_bool(
                _first_present(
                    pathway.get("release_ready"),
                    metadata.get("pathway_intervention_workflow_release_ready"),
                    contract_metadata.get("pathway_intervention_workflow_release_ready"),
                )
            ),
            "model": _optional_string(
                _first_present(
                    pathway.get("model"),
                    metadata.get("pathway_intervention_workflow_model"),
                    contract_metadata.get("pathway_intervention_workflow_model"),
                )
            ),
            "layer": _finite_float(
                _first_present(
                    pathway.get("layer"),
                    metadata.get("pathway_intervention_workflow_layer"),
                    contract_metadata.get("pathway_intervention_workflow_layer"),
                )
            ),
            "intervention_layer": _finite_float(
                _first_present(
                    pathway.get("intervention_layer"),
                    metadata.get("pathway_intervention_workflow_intervention_layer"),
                    contract_metadata.get(
                        "pathway_intervention_workflow_intervention_layer"
                    ),
                )
            ),
            "patch_layer": _finite_float(
                _first_present(
                    pathway.get("patch_layer"),
                    metadata.get("pathway_intervention_workflow_patch_layer"),
                    contract_metadata.get("pathway_intervention_workflow_patch_layer"),
                )
            ),
            "activation_ablation_gate": _optional_string(
                _first_present(
                    pathway.get("activation_ablation_gate_status"),
                    pathway.get("activation_ablation_gate"),
                    metadata.get("pathway_intervention_workflow_activation_ablation_gate"),
                    contract_metadata.get(
                        "pathway_intervention_workflow_activation_ablation_gate"
                    ),
                )
            ),
            "source_patch_gate": _optional_string(
                _first_present(
                    pathway.get("source_patch_gate_status"),
                    pathway.get("source_patch_gate"),
                    metadata.get("pathway_intervention_workflow_source_patch_gate"),
                    contract_metadata.get("pathway_intervention_workflow_source_patch_gate"),
                )
            ),
            "signals": _first_present(
                pathway.get("signals"),
                metadata.get("pathway_intervention_workflow_signals"),
                contract_metadata.get("pathway_intervention_workflow_signals"),
            ),
            "best_signals": _mapping(
                _first_present(
                    pathway.get("best_signals"),
                    metadata.get("pathway_intervention_workflow_best_signals"),
                    contract_metadata.get("pathway_intervention_workflow_best_signals"),
                )
            ),
        },
        "external_evidence_baseline_comparison": {
            "available": external_evidence_available,
            "source": external_evidence_source,
            "report": _optional_string(
                _first_present(
                    external_evidence.get("report_path"),
                    external_evidence.get("report"),
                    metadata.get("external_evidence_baseline_comparison_report"),
                    contract_metadata.get("external_evidence_baseline_comparison_report"),
                )
            ),
            "registry": _optional_string(
                _first_present(
                    external_evidence.get("registry"),
                    metadata.get("external_evidence_baseline_comparison_registry"),
                    contract_metadata.get("external_evidence_baseline_comparison_registry"),
                )
            ),
            "record": _optional_string(
                _first_present(
                    external_evidence.get("record_key"),
                    external_evidence.get("record"),
                    metadata.get("external_evidence_baseline_comparison_record"),
                    metadata.get("external_evidence_baseline_comparison_registry_key"),
                    contract_metadata.get("external_evidence_baseline_comparison_record"),
                    contract_metadata.get("external_evidence_baseline_comparison_registry_key"),
                )
            ),
            "status": external_evidence_status,
            "decision_status": external_evidence_decision_status,
            "recommended_route": _optional_string(
                _first_present(
                    external_evidence.get("recommended_route"),
                    metadata.get("external_evidence_baseline_comparison_recommended_route"),
                    contract_metadata.get(
                        "external_evidence_baseline_comparison_recommended_route"
                    ),
                )
            ),
            "recommended_route_record": _optional_string(
                _first_present(
                    external_evidence.get("recommended_route_record"),
                    metadata.get(
                        "external_evidence_baseline_comparison_recommended_route_record"
                    ),
                    contract_metadata.get(
                        "external_evidence_baseline_comparison_recommended_route_record"
                    ),
                )
            ),
            "route_passed": _optional_bool(
                _first_present(
                    external_evidence.get("route_passed"),
                    metadata.get("external_evidence_baseline_comparison_route_passed"),
                    contract_metadata.get(
                        "external_evidence_baseline_comparison_route_passed"
                    ),
                )
            ),
            "text_redline_passed": _optional_bool(
                _first_present(
                    external_evidence.get("text_redline_passed"),
                    metadata.get(
                        "external_evidence_baseline_comparison_text_redline_passed"
                    ),
                    contract_metadata.get(
                        "external_evidence_baseline_comparison_text_redline_passed"
                    ),
                )
            ),
            "text_redline_run_count": _finite_float(
                _first_present(
                    external_evidence.get("text_redline_run_count"),
                    metadata.get(
                        "external_evidence_baseline_comparison_text_redline_run_count"
                    ),
                    contract_metadata.get(
                        "external_evidence_baseline_comparison_text_redline_run_count"
                    ),
                )
            ),
        },
        "pre_generation_probe_comparison": {
            "available": pre_generation_available,
            "source": _optional_string(
                _first_present(
                    pre_generation.get("source"),
                    metadata.get("pre_generation_probe_comparison_source"),
                    contract_metadata.get("pre_generation_probe_comparison_source"),
                )
            ),
            "report": _optional_string(
                _first_present(
                    pre_generation.get("report_path"),
                    pre_generation.get("report"),
                    metadata.get("pre_generation_probe_comparison_report"),
                    contract_metadata.get("pre_generation_probe_comparison_report"),
                )
            ),
            "manifest": _optional_string(
                _first_present(
                    pre_generation.get("manifest_path"),
                    pre_generation.get("manifest"),
                    metadata.get("pre_generation_probe_comparison_manifest"),
                    contract_metadata.get("pre_generation_probe_comparison_manifest"),
                )
            ),
            "registry": _optional_string(
                _first_present(
                    pre_generation.get("registry"),
                    metadata.get("pre_generation_probe_comparison_registry"),
                    contract_metadata.get("pre_generation_probe_comparison_registry"),
                )
            ),
            "record": _optional_string(
                _first_present(
                    pre_generation.get("record_key"),
                    pre_generation.get("record"),
                    metadata.get("pre_generation_probe_comparison_record"),
                    metadata.get("pre_generation_probe_comparison_registry_key"),
                    contract_metadata.get("pre_generation_probe_comparison_record"),
                    contract_metadata.get("pre_generation_probe_comparison_registry_key"),
                )
            ),
            "manifest_verified": _optional_bool(
                _first_present(
                    pre_generation.get("manifest_verified"),
                    metadata.get("pre_generation_probe_comparison_manifest_verified"),
                    contract_metadata.get("pre_generation_probe_comparison_manifest_verified"),
                    pre_generation_manifest_verification.get("passed"),
                )
            ),
            "status": _optional_string(
                _first_present(
                    pre_generation.get("status"),
                    metadata.get("pre_generation_probe_comparison_status"),
                    contract_metadata.get("pre_generation_probe_comparison_status"),
                )
            ),
            "model_count": _finite_float(
                _first_present(
                    pre_generation.get("model_count"),
                    metadata.get("pre_generation_probe_comparison_model_count"),
                    contract_metadata.get("pre_generation_probe_comparison_model_count"),
                )
            ),
            "run_count": _finite_float(
                _first_present(
                    pre_generation.get("run_count"),
                    metadata.get("pre_generation_probe_comparison_run_count"),
                    contract_metadata.get("pre_generation_probe_comparison_run_count"),
                )
            ),
            "redline_passed": _optional_bool(
                _first_present(
                    pre_generation.get("redline_passed"),
                    metadata.get("pre_generation_probe_comparison_redline_passed"),
                    contract_metadata.get("pre_generation_probe_comparison_redline_passed"),
                )
            ),
            "redline_run_count": _finite_float(
                _first_present(
                    pre_generation.get("redline_run_count"),
                    metadata.get("pre_generation_probe_comparison_redline_run_count"),
                    contract_metadata.get(
                        "pre_generation_probe_comparison_redline_run_count"
                    ),
                )
            ),
            "best_run": _optional_string(
                _first_present(
                    pre_generation_best_run.get("name"),
                    metadata.get("pre_generation_probe_comparison_best_run"),
                    contract_metadata.get("pre_generation_probe_comparison_best_run"),
                )
            ),
            "best_model": _optional_string(
                _first_present(
                    pre_generation_best_run.get("model"),
                    metadata.get("pre_generation_probe_comparison_best_model"),
                    contract_metadata.get("pre_generation_probe_comparison_best_model"),
                )
            ),
            "best_layer": _finite_float(
                _first_present(
                    pre_generation_best_run.get("recommended_layer"),
                    metadata.get("pre_generation_probe_comparison_best_layer"),
                    contract_metadata.get("pre_generation_probe_comparison_best_layer"),
                )
            ),
            "best_test_label_auroc": _finite_float(
                _first_present(
                    pre_generation_best_run.get("test_label_auroc"),
                    metadata.get("pre_generation_probe_comparison_best_test_label_auroc"),
                    contract_metadata.get(
                        "pre_generation_probe_comparison_best_test_label_auroc"
                    ),
                )
            ),
            "best_redline_signal": _optional_string(
                _first_present(
                    pre_generation_best_run.get("redline_best_signal"),
                    metadata.get("pre_generation_probe_comparison_best_redline_signal"),
                    contract_metadata.get(
                        "pre_generation_probe_comparison_best_redline_signal"
                    ),
                )
            ),
            "best_redline_auroc": _finite_float(
                _first_present(
                    pre_generation_best_run.get("redline_best_auroc"),
                    metadata.get("pre_generation_probe_comparison_best_redline_auroc"),
                    contract_metadata.get(
                        "pre_generation_probe_comparison_best_redline_auroc"
                    ),
                )
            ),
            "best_redline_margin": _finite_float(
                _first_present(
                    pre_generation_best_run.get("redline_margin"),
                    metadata.get("pre_generation_probe_comparison_best_redline_margin"),
                    contract_metadata.get(
                        "pre_generation_probe_comparison_best_redline_margin"
                    ),
                )
            ),
        },
        "claim_factuality_probe_comparison": {
            "available": claim_factuality_available,
            "source": _optional_string(
                _first_present(
                    claim_factuality.get("source"),
                    metadata.get("claim_factuality_probe_comparison_source"),
                    contract_metadata.get("claim_factuality_probe_comparison_source"),
                )
            ),
            "report": _optional_string(
                _first_present(
                    claim_factuality.get("report_path"),
                    claim_factuality.get("report"),
                    metadata.get("claim_factuality_probe_comparison_report"),
                    contract_metadata.get("claim_factuality_probe_comparison_report"),
                )
            ),
            "manifest": _optional_string(
                _first_present(
                    claim_factuality.get("manifest_path"),
                    claim_factuality.get("manifest"),
                    metadata.get("claim_factuality_probe_comparison_manifest"),
                    contract_metadata.get("claim_factuality_probe_comparison_manifest"),
                )
            ),
            "registry": _optional_string(
                _first_present(
                    claim_factuality.get("registry"),
                    metadata.get("claim_factuality_probe_comparison_registry"),
                    contract_metadata.get("claim_factuality_probe_comparison_registry"),
                )
            ),
            "record": _optional_string(
                _first_present(
                    claim_factuality.get("record_key"),
                    claim_factuality.get("record"),
                    metadata.get("claim_factuality_probe_comparison_record"),
                    metadata.get("claim_factuality_probe_comparison_registry_key"),
                    contract_metadata.get("claim_factuality_probe_comparison_record"),
                    contract_metadata.get(
                        "claim_factuality_probe_comparison_registry_key"
                    ),
                )
            ),
            "manifest_verified": _optional_bool(
                _first_present(
                    claim_factuality.get("manifest_verified"),
                    metadata.get("claim_factuality_probe_comparison_manifest_verified"),
                    contract_metadata.get(
                        "claim_factuality_probe_comparison_manifest_verified"
                    ),
                    claim_factuality_manifest_verification.get("passed"),
                )
            ),
            "status": _optional_string(
                _first_present(
                    metadata.get("claim_factuality_probe_comparison_status"),
                    contract_metadata.get("claim_factuality_probe_comparison_status"),
                    claim_factuality.get("status"),
                )
            ),
            "report_status": _optional_string(
                _first_present(
                    claim_factuality.get("report_status"),
                    metadata.get("claim_factuality_probe_comparison_report_status"),
                    contract_metadata.get(
                        "claim_factuality_probe_comparison_report_status"
                    ),
                )
            ),
            "model_count": _finite_float(
                _first_present(
                    claim_factuality.get("model_count"),
                    metadata.get("claim_factuality_probe_comparison_model_count"),
                    contract_metadata.get("claim_factuality_probe_comparison_model_count"),
                )
            ),
            "run_count": _finite_float(
                _first_present(
                    claim_factuality.get("run_count"),
                    metadata.get("claim_factuality_probe_comparison_run_count"),
                    contract_metadata.get("claim_factuality_probe_comparison_run_count"),
                )
            ),
            "redline_passed": _optional_bool(
                _first_present(
                    claim_factuality.get("redline_passed"),
                    metadata.get("claim_factuality_probe_comparison_redline_passed"),
                    contract_metadata.get(
                        "claim_factuality_probe_comparison_redline_passed"
                    ),
                )
            ),
            "redline_run_count": _finite_float(
                _first_present(
                    claim_factuality.get("redline_run_count"),
                    metadata.get("claim_factuality_probe_comparison_redline_run_count"),
                    contract_metadata.get(
                        "claim_factuality_probe_comparison_redline_run_count"
                    ),
                )
            ),
            "best_run": _optional_string(
                _first_present(
                    claim_factuality_best_run.get("name"),
                    metadata.get("claim_factuality_probe_comparison_best_run"),
                    contract_metadata.get("claim_factuality_probe_comparison_best_run"),
                )
            ),
            "best_model": _optional_string(
                _first_present(
                    claim_factuality_best_run.get("model"),
                    metadata.get("claim_factuality_probe_comparison_best_model"),
                    contract_metadata.get("claim_factuality_probe_comparison_best_model"),
                )
            ),
            "best_record_count": _finite_float(
                _first_present(
                    claim_factuality_best_run.get("record_count"),
                    metadata.get("claim_factuality_probe_comparison_best_record_count"),
                    contract_metadata.get(
                        "claim_factuality_probe_comparison_best_record_count"
                    ),
                )
            ),
            "best_layer": _finite_float(
                _first_present(
                    claim_factuality_best_run.get("recommended_layer"),
                    metadata.get("claim_factuality_probe_comparison_best_layer"),
                    contract_metadata.get("claim_factuality_probe_comparison_best_layer"),
                )
            ),
            "best_test_label_auroc": _finite_float(
                _first_present(
                    claim_factuality_best_run.get("test_label_auroc"),
                    metadata.get("claim_factuality_probe_comparison_best_test_label_auroc"),
                    contract_metadata.get(
                        "claim_factuality_probe_comparison_best_test_label_auroc"
                    ),
                )
            ),
            "best_test_selective_accuracy": _finite_float(
                _first_present(
                    claim_factuality_best_run.get("test_selective_accuracy"),
                    metadata.get(
                        "claim_factuality_probe_comparison_best_test_selective_accuracy"
                    ),
                    contract_metadata.get(
                        "claim_factuality_probe_comparison_best_test_selective_accuracy"
                    ),
                )
            ),
            "best_test_selective_coverage": _finite_float(
                _first_present(
                    claim_factuality_best_run.get("test_selective_coverage"),
                    metadata.get(
                        "claim_factuality_probe_comparison_best_test_selective_coverage"
                    ),
                    contract_metadata.get(
                        "claim_factuality_probe_comparison_best_test_selective_coverage"
                    ),
                )
            ),
            "best_conformal_threshold": _finite_float(
                _first_present(
                    claim_factuality_best_run.get("conformal_threshold"),
                    metadata.get(
                        "claim_factuality_probe_comparison_best_conformal_threshold"
                    ),
                    contract_metadata.get(
                        "claim_factuality_probe_comparison_best_conformal_threshold"
                    ),
                )
            ),
            "best_redline_signal": _optional_string(
                _first_present(
                    claim_factuality_best_run.get("redline_best_signal"),
                    metadata.get("claim_factuality_probe_comparison_best_redline_signal"),
                    contract_metadata.get(
                        "claim_factuality_probe_comparison_best_redline_signal"
                    ),
                )
            ),
            "best_redline_auroc": _finite_float(
                _first_present(
                    claim_factuality_best_run.get("redline_best_auroc"),
                    metadata.get("claim_factuality_probe_comparison_best_redline_auroc"),
                    contract_metadata.get(
                        "claim_factuality_probe_comparison_best_redline_auroc"
                    ),
                )
            ),
            "best_redline_margin": _finite_float(
                _first_present(
                    claim_factuality_best_run.get("redline_margin"),
                    metadata.get("claim_factuality_probe_comparison_best_redline_margin"),
                    contract_metadata.get(
                        "claim_factuality_probe_comparison_best_redline_margin"
                    ),
                )
            ),
        },
        "triple_extraction_fixture_matrix": {
            "available": matrix_available,
            "source": matrix_source,
            "status": matrix_status,
            "manifest_verified": _optional_bool(
                _first_present(
                    matrix.get("manifest_verified"),
                    metadata.get("triple_extraction_fixture_matrix_manifest_verified"),
                    contract_metadata.get("triple_extraction_fixture_matrix_manifest_verified"),
                    manifest_verification.get("passed"),
                )
            ),
            "n_corpora": _finite_float(
                _first_present(
                    matrix.get("n_corpora"),
                    metadata.get("triple_extraction_fixture_matrix_n_corpora"),
                    contract_metadata.get("triple_extraction_fixture_matrix_n_corpora"),
                )
            ),
            "promoted_corpora": _finite_float(
                _first_present(
                    matrix.get("promoted_corpora"),
                    metadata.get("triple_extraction_fixture_matrix_promoted_corpora"),
                    contract_metadata.get("triple_extraction_fixture_matrix_promoted_corpora"),
                )
            ),
            "distinct_predicate_count": _finite_float(
                _first_present(
                    matrix.get("distinct_predicate_count"),
                    metadata.get("triple_extraction_fixture_matrix_distinct_predicate_count"),
                    contract_metadata.get("triple_extraction_fixture_matrix_distinct_predicate_count"),
                )
            ),
            "mean_best_f1": _finite_float(
                _first_present(
                    matrix.get("mean_best_f1"),
                    metadata.get("triple_extraction_fixture_matrix_mean_best_f1"),
                    contract_metadata.get("triple_extraction_fixture_matrix_mean_best_f1"),
                )
            ),
            "mean_f1_lift": _finite_float(
                _first_present(
                    matrix.get("mean_f1_lift"),
                    metadata.get("triple_extraction_fixture_matrix_mean_f1_lift"),
                    contract_metadata.get("triple_extraction_fixture_matrix_mean_f1_lift"),
                )
            ),
        },
        "counterfactual_verification": {
            "available": counterfactual_available,
            "source": _optional_string(
                _first_present(
                    counterfactual.get("source"),
                    metadata.get("counterfactual_verification_source"),
                    contract_metadata.get("counterfactual_verification_source"),
                )
            ),
            "report": _optional_string(
                _first_present(
                    counterfactual.get("report_path"),
                    counterfactual.get("report"),
                    metadata.get("counterfactual_verification_report"),
                    contract_metadata.get("counterfactual_verification_report"),
                )
            ),
            "manifest": _optional_string(
                _first_present(
                    counterfactual.get("manifest_path"),
                    counterfactual.get("manifest"),
                    metadata.get("counterfactual_verification_manifest"),
                    contract_metadata.get("counterfactual_verification_manifest"),
                )
            ),
            "registry": _optional_string(
                _first_present(
                    counterfactual.get("registry"),
                    metadata.get("counterfactual_verification_registry"),
                    contract_metadata.get("counterfactual_verification_registry"),
                )
            ),
            "record": _optional_string(
                _first_present(
                    counterfactual.get("record_key"),
                    counterfactual.get("record"),
                    metadata.get("counterfactual_verification_record"),
                    metadata.get("counterfactual_verification_registry_key"),
                    contract_metadata.get("counterfactual_verification_record"),
                    contract_metadata.get("counterfactual_verification_registry_key"),
                )
            ),
            "manifest_verified": _optional_bool(
                _first_present(
                    counterfactual.get("manifest_verified"),
                    metadata.get("counterfactual_verification_manifest_verified"),
                    contract_metadata.get("counterfactual_verification_manifest_verified"),
                    counterfactual_manifest_verification.get("passed"),
                )
            ),
            "status": _optional_string(
                _first_present(
                    counterfactual.get("status"),
                    metadata.get("counterfactual_verification_status"),
                    contract_metadata.get("counterfactual_verification_status"),
                )
            ),
            "workflow": _optional_string(
                _first_present(
                    counterfactual.get("workflow"),
                    metadata.get("counterfactual_verification_workflow"),
                    contract_metadata.get("counterfactual_verification_workflow"),
                )
            ),
            "record_count": _finite_float(
                _first_present(
                    counterfactual.get("record_count"),
                    metadata.get("counterfactual_verification_record_count"),
                    contract_metadata.get("counterfactual_verification_record_count"),
                )
            ),
            "pass_rate": _finite_float(
                _first_present(
                    counterfactual.get("pass_rate"),
                    metadata.get("counterfactual_verification_pass_rate"),
                    contract_metadata.get("counterfactual_verification_pass_rate"),
                )
            ),
            "false_invariance_rate": _finite_float(
                _first_present(
                    counterfactual.get("false_invariance_rate"),
                    metadata.get("counterfactual_verification_false_invariance_rate"),
                    contract_metadata.get(
                        "counterfactual_verification_false_invariance_rate"
                    ),
                )
            ),
            "flip_success_count": _finite_float(
                _first_present(
                    counterfactual.get("flip_success_count"),
                    metadata.get("counterfactual_verification_flip_success_count"),
                    contract_metadata.get(
                        "counterfactual_verification_flip_success_count"
                    ),
                )
            ),
        },
    }
    matrix_summary = _mapping(summary["triple_extraction_fixture_matrix"])
    external_evidence_summary = _mapping(
        summary["external_evidence_baseline_comparison"]
    )
    pathway_summary = _mapping(summary["pathway_intervention_workflow"])
    pre_generation_summary = _mapping(summary["pre_generation_probe_comparison"])
    claim_factuality_summary = _mapping(summary["claim_factuality_probe_comparison"])
    counterfactual_summary = _mapping(summary["counterfactual_verification"])
    frontier_release_evidence_summary = _mapping(summary["frontier_release_evidence"])
    product_trace_replay_metrics = _promotion_contract_product_trace_replay_metric_values(
        product_trace_replay
    )
    runtime_drift_metrics = _promotion_contract_runtime_drift_metric_values(runtime_drift)
    evidence_handoff_metrics = _promotion_contract_evidence_handoff_metric_values(
        evidence_handoff
    )
    triple_audit_evidence_metrics = (
        _promotion_contract_triple_audit_evidence_metric_values(triple_audit_evidence)
    )
    return {
        "promotion_contract_available": available,
        "promotion_contract_source": source,
        "promotion_contract_source_status": source_status,
        "promotion_contract_budget_enabled": budget_enabled,
        "promotion_contract_summary": summary,
        "promotion_contract_frontier_release_evidence": (
            frontier_release_evidence or None
        ),
        "promotion_contract_frontier_release_evidence_available": (
            frontier_release_evidence_available
        ),
        "promotion_contract_frontier_release_evidence_status": (
            frontier_release_evidence_summary.get("status")
        ),
        "promotion_contract_frontier_release_evidence_report": (
            frontier_release_evidence_summary.get("report")
        ),
        "promotion_contract_frontier_release_evidence_manifest": (
            frontier_release_evidence_summary.get("manifest")
        ),
        "promotion_contract_frontier_release_evidence_source": (
            frontier_release_evidence_summary.get("source")
        ),
        "promotion_contract_frontier_release_evidence_registry": (
            frontier_release_evidence_summary.get("registry")
        ),
        "promotion_contract_frontier_release_evidence_record": (
            frontier_release_evidence_summary.get("record")
        ),
        "promotion_contract_frontier_release_evidence_workflow": (
            frontier_release_evidence_summary.get("workflow")
        ),
        "promotion_contract_frontier_release_evidence_report_status": (
            frontier_release_evidence_summary.get("report_status")
        ),
        "promotion_contract_frontier_release_evidence_decision_status": (
            frontier_release_evidence_summary.get("decision_status")
        ),
        "promotion_contract_frontier_release_evidence_verifier_track_status": (
            frontier_release_evidence_summary.get("verifier_track_status")
        ),
        "promotion_contract_frontier_release_evidence_abstention_track_status": (
            frontier_release_evidence_summary.get("abstention_track_status")
        ),
        "promotion_contract_frontier_release_evidence_multiple_testing_track_status": (
            frontier_release_evidence_summary.get("multiple_testing_track_status")
        ),
        "promotion_contract_frontier_release_evidence_citation_batch_track_status": (
            frontier_release_evidence_summary.get("citation_batch_track_status")
        ),
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_track_status": (
            frontier_release_evidence_summary.get("frontier_rerun_rollup_track_status")
        ),
        "promotion_contract_frontier_release_evidence_base_verifier_track_status": (
            frontier_release_evidence_summary.get("base_verifier_track_status")
        ),
        "promotion_contract_frontier_release_evidence_base_abstention_track_status": (
            frontier_release_evidence_summary.get("base_abstention_track_status")
        ),
        "promotion_contract_frontier_release_evidence_base_detectability_track_status": (
            frontier_release_evidence_summary.get("base_detectability_track_status")
        ),
        "promotion_contract_frontier_release_evidence_base_multiple_testing_track_status": (
            frontier_release_evidence_summary.get("base_multiple_testing_track_status")
        ),
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_promoted_tracks": (
            frontier_release_evidence_summary.get("frontier_rerun_rollup_promoted_tracks")
        ),
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_report_count": (
            frontier_release_evidence_summary.get("frontier_rerun_rollup_report_count")
        ),
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_candidate_count": (
            frontier_release_evidence_summary.get("frontier_rerun_rollup_candidate_count")
        ),
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_missing_report_count": (
            frontier_release_evidence_summary.get("frontier_rerun_rollup_missing_report_count")
        ),
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_invalid_report_count": (
            frontier_release_evidence_summary.get("frontier_rerun_rollup_invalid_report_count")
        ),
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_blocked_candidate_count": (
            frontier_release_evidence_summary.get("frontier_rerun_rollup_blocked_candidate_count")
        ),
        "promotion_contract_frontier_release_evidence_frontier_rerun_rollup_promotion_ready_count": (
            frontier_release_evidence_summary.get("frontier_rerun_rollup_promotion_ready_count")
        ),
        "promotion_contract_frontier_release_evidence_citation_batch_rollup_count": (
            frontier_release_evidence_summary.get("citation_batch_rollup_count")
        ),
        "promotion_contract_frontier_release_evidence_citation_batch_expected_batch_count": (
            frontier_release_evidence_summary.get("citation_batch_expected_batch_count")
        ),
        "promotion_contract_frontier_release_evidence_citation_batch_observed_batch_count": (
            frontier_release_evidence_summary.get("citation_batch_observed_batch_count")
        ),
        "promotion_contract_frontier_release_evidence_citation_batch_missing_expected_batch_count": (
            frontier_release_evidence_summary.get(
                "citation_batch_missing_expected_batch_count"
            )
        ),
        "promotion_contract_frontier_release_evidence_citation_batch_duplicate_batch_count": (
            frontier_release_evidence_summary.get("citation_batch_duplicate_batch_count")
        ),
        "promotion_contract_frontier_release_evidence_citation_batch_unexpected_batch_count": (
            frontier_release_evidence_summary.get("citation_batch_unexpected_batch_count")
        ),
        "promotion_contract_frontier_release_evidence_run_count": _finite_float(
            frontier_release_evidence_summary.get("run_count")
        ),
        "promotion_contract_frontier_release_evidence_run_names": (
            frontier_release_evidence_summary.get("run_names")
        ),
        "promotion_contract_promotion_summary": promotion_summary or None,
        "promotion_contract_promotion_summary_status": _optional_string(
            promotion_summary.get("status")
        ),
        "promotion_contract_promotion_summary_source_status": _optional_string(
            promotion_summary.get("source_status")
        ),
        "promotion_contract_promotion_summary_available_gate_count": _finite_float(
            promotion_summary.get("available_gate_count")
        ),
        "promotion_contract_promotion_summary_promoted_gate_count": _finite_float(
            promotion_summary.get("promoted_gate_count")
        ),
        "promotion_contract_promotion_summary_blocking_gate_count": _finite_float(
            promotion_summary.get("blocking_gate_count")
        ),
        "promotion_contract_promotion_summary_blocked_evidence_group_count": (
            _finite_float(promotion_summary.get("blocked_evidence_group_count"))
        ),
        "promotion_contract_promotion_summary_runtime_layer": _finite_float(
            promotion_summary_runtime.get("layer")
        ),
        "promotion_contract_promotion_summary_recommended_runtime_seconds": (
            _finite_float(promotion_summary_runtime.get("recommended_runtime_seconds"))
        ),
        "promotion_contract_promotion_summary_recommended_runtime_cost_source": (
            _optional_string(
                promotion_summary_runtime.get("recommended_runtime_cost_source")
            )
        ),
        "promotion_contract_promotion_summary_route": _optional_string(
            promotion_summary_verifier_route.get("route")
        ),
        "promotion_contract_promotion_summary_action_audit_status": _optional_string(
            promotion_summary_action_gates.get("action_audit_status")
        ),
        "promotion_contract_promotion_summary_action_execution_status": (
            _optional_string(
                promotion_summary_action_gates.get("action_execution_status")
            )
        ),
        "promotion_contract_recommended_route_covered_fact_property_count": (
            covered_fact_scope.get("recommended_route_count")
        ),
        "promotion_contract_recommended_route_covered_fact_properties": (
            covered_fact_scope.get("recommended_route_properties")
        ),
        "promotion_contract_recommended_route_covered_fact_property_metrics": (
            covered_fact_scope.get("recommended_route_property_metrics")
        ),
        **_prefixed_property_rollup_metrics(
            "promotion_contract_recommended_route_covered_fact",
            recommended_property_rollups,
        ),
        "promotion_contract_required_route_baseline_covered_fact_property_counts": (
            covered_fact_scope.get("required_route_baseline_counts")
        ),
        "promotion_contract_required_route_baseline_covered_fact_properties": (
            covered_fact_scope.get("required_route_baseline_properties")
        ),
        "promotion_contract_required_route_baseline_covered_fact_property_metrics": (
            covered_fact_scope.get("required_route_baseline_property_metrics")
        ),
        **_prefixed_property_rollup_metrics(
            "promotion_contract_required_route_baseline_covered_fact",
            required_property_rollups,
        ),
        "promotion_contract_structured_fact_robustness_property_counts": (
            covered_fact_scope.get("structured_fact_robustness_counts")
        ),
        "promotion_contract_structured_fact_robustness_properties": (
            covered_fact_scope.get("structured_fact_robustness_properties")
        ),
        "promotion_contract_structured_fact_robustness_property_metrics": (
            covered_fact_scope.get("structured_fact_robustness_property_metrics")
        ),
        **_prefixed_property_rollup_metrics(
            "promotion_contract_structured_fact_robustness",
            robustness_property_rollups,
        ),
        "promotion_contract_external_evidence_baseline_comparison_available": (
            external_evidence_available
        ),
        "promotion_contract_external_evidence_baseline_comparison_source": (
            external_evidence_source
        ),
        "promotion_contract_external_evidence_baseline_comparison_report": (
            external_evidence_summary.get("report")
        ),
        "promotion_contract_external_evidence_baseline_comparison_registry": (
            external_evidence_summary.get("registry")
        ),
        "promotion_contract_external_evidence_baseline_comparison_record": (
            external_evidence_summary.get("record")
        ),
        "promotion_contract_external_evidence_baseline_comparison_status": (
            external_evidence_status
        ),
        "promotion_contract_external_evidence_baseline_comparison_decision_status": (
            external_evidence_decision_status
        ),
        "promotion_contract_external_evidence_baseline_comparison_recommended_route": (
            external_evidence_summary.get("recommended_route")
        ),
        "promotion_contract_external_evidence_baseline_comparison_recommended_route_record": (
            external_evidence_summary.get("recommended_route_record")
        ),
        "promotion_contract_external_evidence_baseline_comparison_route_passed": (
            external_evidence_summary.get("route_passed")
        ),
        "promotion_contract_external_evidence_baseline_comparison_text_redline_passed": (
            external_evidence_summary.get("text_redline_passed")
        ),
        "promotion_contract_external_evidence_baseline_comparison_text_redline_run_count": (
            external_evidence_summary.get("text_redline_run_count")
        ),
        "promotion_contract_pre_generation_probe_comparison_available": (
            pre_generation_available
        ),
        "promotion_contract_pre_generation_probe_comparison_source": (
            pre_generation_summary.get("source")
        ),
        "promotion_contract_pre_generation_probe_comparison_report": (
            pre_generation_summary.get("report")
        ),
        "promotion_contract_pre_generation_probe_comparison_manifest": (
            pre_generation_summary.get("manifest")
        ),
        "promotion_contract_pre_generation_probe_comparison_registry": (
            pre_generation_summary.get("registry")
        ),
        "promotion_contract_pre_generation_probe_comparison_record": (
            pre_generation_summary.get("record")
        ),
        "promotion_contract_pre_generation_probe_comparison_manifest_verified": (
            pre_generation_summary.get("manifest_verified")
        ),
        "promotion_contract_pre_generation_probe_comparison_status": (
            pre_generation_summary.get("status")
        ),
        "promotion_contract_pre_generation_probe_comparison_model_count": (
            pre_generation_summary.get("model_count")
        ),
        "promotion_contract_pre_generation_probe_comparison_run_count": (
            pre_generation_summary.get("run_count")
        ),
        "promotion_contract_pre_generation_probe_comparison_redline_passed": (
            pre_generation_summary.get("redline_passed")
        ),
        "promotion_contract_pre_generation_probe_comparison_redline_run_count": (
            pre_generation_summary.get("redline_run_count")
        ),
        "promotion_contract_pre_generation_probe_comparison_best_run": (
            pre_generation_summary.get("best_run")
        ),
        "promotion_contract_pre_generation_probe_comparison_best_model": (
            pre_generation_summary.get("best_model")
        ),
        "promotion_contract_pre_generation_probe_comparison_best_layer": (
            pre_generation_summary.get("best_layer")
        ),
        "promotion_contract_pre_generation_probe_comparison_best_test_label_auroc": (
            pre_generation_summary.get("best_test_label_auroc")
        ),
        "promotion_contract_pre_generation_probe_comparison_best_redline_signal": (
            pre_generation_summary.get("best_redline_signal")
        ),
        "promotion_contract_pre_generation_probe_comparison_best_redline_auroc": (
            pre_generation_summary.get("best_redline_auroc")
        ),
        "promotion_contract_pre_generation_probe_comparison_best_redline_margin": (
            pre_generation_summary.get("best_redline_margin")
        ),
        "promotion_contract_claim_factuality_probe_comparison_available": (
            claim_factuality_available
        ),
        "promotion_contract_claim_factuality_probe_comparison_source": (
            claim_factuality_summary.get("source")
        ),
        "promotion_contract_claim_factuality_probe_comparison_report": (
            claim_factuality_summary.get("report")
        ),
        "promotion_contract_claim_factuality_probe_comparison_manifest": (
            claim_factuality_summary.get("manifest")
        ),
        "promotion_contract_claim_factuality_probe_comparison_registry": (
            claim_factuality_summary.get("registry")
        ),
        "promotion_contract_claim_factuality_probe_comparison_record": (
            claim_factuality_summary.get("record")
        ),
        "promotion_contract_claim_factuality_probe_comparison_manifest_verified": (
            claim_factuality_summary.get("manifest_verified")
        ),
        "promotion_contract_claim_factuality_probe_comparison_status": (
            claim_factuality_summary.get("status")
        ),
        "promotion_contract_claim_factuality_probe_comparison_report_status": (
            claim_factuality_summary.get("report_status")
        ),
        "promotion_contract_claim_factuality_probe_comparison_model_count": (
            claim_factuality_summary.get("model_count")
        ),
        "promotion_contract_claim_factuality_probe_comparison_run_count": (
            claim_factuality_summary.get("run_count")
        ),
        "promotion_contract_claim_factuality_probe_comparison_redline_passed": (
            claim_factuality_summary.get("redline_passed")
        ),
        "promotion_contract_claim_factuality_probe_comparison_redline_run_count": (
            claim_factuality_summary.get("redline_run_count")
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_run": (
            claim_factuality_summary.get("best_run")
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_model": (
            claim_factuality_summary.get("best_model")
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_record_count": (
            claim_factuality_summary.get("best_record_count")
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_layer": (
            claim_factuality_summary.get("best_layer")
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_test_label_auroc": (
            claim_factuality_summary.get("best_test_label_auroc")
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_test_selective_accuracy": (
            claim_factuality_summary.get("best_test_selective_accuracy")
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_test_selective_coverage": (
            claim_factuality_summary.get("best_test_selective_coverage")
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_conformal_threshold": (
            claim_factuality_summary.get("best_conformal_threshold")
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_redline_signal": (
            claim_factuality_summary.get("best_redline_signal")
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_redline_auroc": (
            claim_factuality_summary.get("best_redline_auroc")
        ),
        "promotion_contract_claim_factuality_probe_comparison_best_redline_margin": (
            claim_factuality_summary.get("best_redline_margin")
        ),
        "promotion_contract_triple_extraction_fixture_matrix_available": matrix_available,
        "promotion_contract_triple_extraction_fixture_matrix_source": matrix_source,
        "promotion_contract_triple_extraction_fixture_matrix_status": matrix_status,
        "promotion_contract_triple_extraction_fixture_matrix_manifest_verified": matrix_summary.get(
            "manifest_verified"
        ),
        "promotion_contract_triple_extraction_fixture_matrix_n_corpora": matrix_summary.get(
            "n_corpora"
        ),
        "promotion_contract_triple_extraction_fixture_matrix_promoted_corpora": matrix_summary.get(
            "promoted_corpora"
        ),
        "promotion_contract_triple_extraction_fixture_matrix_distinct_predicate_count": matrix_summary.get(
            "distinct_predicate_count"
        ),
        "promotion_contract_triple_extraction_fixture_matrix_mean_best_f1": matrix_summary.get(
            "mean_best_f1"
        ),
        "promotion_contract_triple_extraction_fixture_matrix_mean_f1_lift": matrix_summary.get(
            "mean_f1_lift"
        ),
        "promotion_contract_pathway_intervention_workflow_available": (
            pathway_summary.get("available")
        ),
        "promotion_contract_pathway_intervention_workflow_source": (
            pathway_summary.get("source")
        ),
        "promotion_contract_pathway_intervention_workflow_report": (
            pathway_summary.get("report")
        ),
        "promotion_contract_pathway_intervention_workflow_manifest": (
            pathway_summary.get("manifest")
        ),
        "promotion_contract_pathway_intervention_workflow_registry": (
            pathway_summary.get("registry")
        ),
        "promotion_contract_pathway_intervention_workflow_record": (
            pathway_summary.get("record")
        ),
        "promotion_contract_pathway_intervention_workflow_manifest_verified": (
            pathway_summary.get("manifest_verified")
        ),
        "promotion_contract_pathway_intervention_workflow_status": (
            pathway_summary.get("status")
        ),
        "promotion_contract_pathway_intervention_workflow_report_status": (
            pathway_summary.get("report_status")
        ),
        "promotion_contract_pathway_intervention_workflow_release_ready": (
            pathway_summary.get("release_ready")
        ),
        "promotion_contract_pathway_intervention_workflow_model": (
            pathway_summary.get("model")
        ),
        "promotion_contract_pathway_intervention_workflow_layer": (
            pathway_summary.get("layer")
        ),
        "promotion_contract_pathway_intervention_workflow_intervention_layer": (
            pathway_summary.get("intervention_layer")
        ),
        "promotion_contract_pathway_intervention_workflow_patch_layer": (
            pathway_summary.get("patch_layer")
        ),
        "promotion_contract_pathway_intervention_workflow_activation_ablation_gate": (
            pathway_summary.get("activation_ablation_gate")
        ),
        "promotion_contract_pathway_intervention_workflow_source_patch_gate": (
            pathway_summary.get("source_patch_gate")
        ),
        "promotion_contract_pathway_intervention_workflow_signals": (
            pathway_summary.get("signals")
        ),
        "promotion_contract_pathway_intervention_workflow_best_signals": (
            pathway_summary.get("best_signals")
        ),
        "promotion_contract_counterfactual_verification_available": (
            counterfactual_available
        ),
        "promotion_contract_counterfactual_verification_source": (
            counterfactual_summary.get("source")
        ),
        "promotion_contract_counterfactual_verification_report": (
            counterfactual_summary.get("report")
        ),
        "promotion_contract_counterfactual_verification_manifest": (
            counterfactual_summary.get("manifest")
        ),
        "promotion_contract_counterfactual_verification_registry": (
            counterfactual_summary.get("registry")
        ),
        "promotion_contract_counterfactual_verification_record": (
            counterfactual_summary.get("record")
        ),
        "promotion_contract_counterfactual_verification_manifest_verified": (
            counterfactual_summary.get("manifest_verified")
        ),
        "promotion_contract_counterfactual_verification_status": (
            counterfactual_summary.get("status")
        ),
        "promotion_contract_counterfactual_verification_workflow": (
            counterfactual_summary.get("workflow")
        ),
        "promotion_contract_counterfactual_verification_record_count": (
            counterfactual_summary.get("record_count")
        ),
        "promotion_contract_counterfactual_verification_pass_rate": (
            counterfactual_summary.get("pass_rate")
        ),
        "promotion_contract_counterfactual_verification_false_invariance_rate": (
            counterfactual_summary.get("false_invariance_rate")
        ),
        "promotion_contract_counterfactual_verification_flip_success_count": (
            counterfactual_summary.get("flip_success_count")
        ),
        **product_trace_replay_metrics,
        **runtime_drift_metrics,
        **evidence_handoff_metrics,
        **triple_audit_evidence_metrics,
    }


def _promotion_contract_evidence_handoff_from_metadata(
    metadata: Mapping[str, Any],
    *,
    contract_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    nested = _mapping(metadata.get("promotion_contract_evidence_handoff"))

    def value(key: str) -> Any:
        return _first_present(
            nested.get(key),
            metadata.get(f"promotion_contract_evidence_handoff_{key}"),
            contract_metadata.get(f"evidence_handoff_{key}"),
            contract_metadata.get(key),
        )

    manifest_summary = _mapping(value("manifest_summary"))
    manifest_metadata = _mapping(value("manifest_metadata"))
    manifest_verification = _mapping(value("manifest_verification"))
    group_statuses = _mapping(
        _first_present(
            value("group_statuses"),
            manifest_summary.get("groups"),
            manifest_summary.get("group_statuses"),
        )
    )
    filled_groups = list(_string_sequence(value("filled_groups")))
    expected_metric_count = _finite_float(
        _first_present(value("expected_metric_count"), manifest_summary.get("expected_metric_count"))
    )
    present_metric_count = _finite_float(
        _first_present(value("present_metric_count"), manifest_summary.get("present_metric_count"))
    )
    missing_metric_count = _finite_float(
        _first_present(value("missing_metric_count"), manifest_summary.get("missing_metric_count"))
    )
    blocked_group_count = _finite_float(
        _first_present(value("blocked_group_count"), manifest_summary.get("blocked_group_count"))
    )
    group_count = _finite_float(value("group_count"))
    if group_count is None and group_statuses:
        group_count = float(len(group_statuses))
    promoted_group_count = _finite_float(value("promoted_group_count"))
    if promoted_group_count is None and group_statuses:
        promoted_group_count = float(
            sum(
                1
                for status in group_statuses.values()
                if _optional_string(status) == "promote"
            )
        )
    handoff = {
        "available": False,
        "manifest": _optional_string(value("manifest")),
        "contract": _optional_string(value("contract")),
        "audit": _optional_string(value("audit")),
        "manifest_verified": _optional_bool(
            _first_present(value("manifest_verified"), manifest_verification.get("passed"))
        ),
        "manifest_verification": manifest_verification,
        "manifest_summary": manifest_summary,
        "manifest_metadata": manifest_metadata,
        "workflow": _optional_string(value("workflow")),
        "status": _optional_string(
            _first_present(value("status"), manifest_metadata.get("status"))
        ),
        "before_missing_metric_count": _finite_float(
            _first_present(
                value("before_missing_metric_count"),
                manifest_metadata.get("before_missing_metric_count"),
            )
        ),
        "after_missing_metric_count": _finite_float(
            _first_present(
                value("after_missing_metric_count"),
                manifest_metadata.get("after_missing_metric_count"),
            )
        ),
        "resolved_missing_metric_count": _finite_float(
            _first_present(
                value("resolved_missing_metric_count"),
                manifest_metadata.get("resolved_missing_metric_count"),
            )
        ),
        "expected_metric_count": expected_metric_count,
        "present_metric_count": present_metric_count,
        "missing_metric_count": missing_metric_count,
        "blocked_group_count": blocked_group_count,
        "filled_groups": filled_groups,
        "group_statuses": {str(key): value for key, value in group_statuses.items()},
        "group_count": group_count,
        "promoted_group_count": promoted_group_count,
        "present_metric_rate": _ratio_or_none(present_metric_count, expected_metric_count),
        "missing_metric_rate": _ratio_or_none(missing_metric_count, expected_metric_count),
        "promoted_group_rate": _ratio_or_none(promoted_group_count, group_count),
    }
    handoff["available"] = _promotion_contract_evidence_handoff_available(handoff)
    return handoff


def _promotion_contract_evidence_handoff_available(handoff: Mapping[str, Any]) -> bool:
    for key, item in handoff.items():
        if key == "available":
            continue
        if isinstance(item, Mapping):
            if bool(item):
                return True
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            if bool(item):
                return True
        elif item is not None:
            return True
    return False


def _promotion_contract_evidence_handoff_metric_values(
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "promotion_contract_evidence_handoff_available": _optional_bool(
            handoff.get("available")
        ),
        "promotion_contract_evidence_handoff_manifest": handoff.get("manifest"),
        "promotion_contract_evidence_handoff_contract": handoff.get("contract"),
        "promotion_contract_evidence_handoff_audit": handoff.get("audit"),
        "promotion_contract_evidence_handoff_manifest_verified": handoff.get(
            "manifest_verified"
        ),
        "promotion_contract_evidence_handoff_workflow": handoff.get("workflow"),
        "promotion_contract_evidence_handoff_status": handoff.get("status"),
        "promotion_contract_evidence_handoff_before_missing_metric_count": (
            handoff.get("before_missing_metric_count")
        ),
        "promotion_contract_evidence_handoff_after_missing_metric_count": (
            handoff.get("after_missing_metric_count")
        ),
        "promotion_contract_evidence_handoff_resolved_missing_metric_count": (
            handoff.get("resolved_missing_metric_count")
        ),
        "promotion_contract_evidence_handoff_expected_metric_count": handoff.get(
            "expected_metric_count"
        ),
        "promotion_contract_evidence_handoff_present_metric_count": handoff.get(
            "present_metric_count"
        ),
        "promotion_contract_evidence_handoff_missing_metric_count": handoff.get(
            "missing_metric_count"
        ),
        "promotion_contract_evidence_handoff_blocked_group_count": handoff.get(
            "blocked_group_count"
        ),
        "promotion_contract_evidence_handoff_present_metric_rate": handoff.get(
            "present_metric_rate"
        ),
        "promotion_contract_evidence_handoff_missing_metric_rate": handoff.get(
            "missing_metric_rate"
        ),
        "promotion_contract_evidence_handoff_group_count": handoff.get("group_count"),
        "promotion_contract_evidence_handoff_promoted_group_count": handoff.get(
            "promoted_group_count"
        ),
        "promotion_contract_evidence_handoff_promoted_group_rate": handoff.get(
            "promoted_group_rate"
        ),
        "promotion_contract_evidence_handoff_filled_groups": list(
            _sequence(handoff.get("filled_groups"))
        ),
        "promotion_contract_evidence_handoff_group_statuses": dict(
            _mapping(handoff.get("group_statuses"))
        ),
    }


def _promotion_contract_triple_audit_evidence_from_metadata(
    metadata: Mapping[str, Any],
    *,
    contract_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    nested = _mapping(metadata.get("promotion_contract_triple_audit_evidence"))

    def value(key: str) -> Any:
        return _first_present(
            nested.get(key),
            metadata.get(f"promotion_contract_triple_audit_evidence_{key}"),
            metadata.get(f"triple_audit_evidence_{key}"),
            contract_metadata.get(f"promotion_contract_triple_audit_evidence_{key}"),
            contract_metadata.get(f"triple_audit_evidence_{key}"),
        )

    evidence = {
        "available": False,
        "source": _optional_string(value("source")),
        "report": _optional_string(value("report")),
        "workflow": _optional_string(value("workflow")),
        "status": _optional_string(value("status")),
    }
    evidence["available"] = any(
        item is not None for key, item in evidence.items() if key != "available"
    )
    return evidence


def _promotion_contract_triple_audit_evidence_metric_values(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "promotion_contract_triple_audit_evidence_available": _optional_bool(
            evidence.get("available")
        ),
        "promotion_contract_triple_audit_evidence_source": evidence.get("source"),
        "promotion_contract_triple_audit_evidence_report": evidence.get("report"),
        "promotion_contract_triple_audit_evidence_workflow": evidence.get("workflow"),
        "promotion_contract_triple_audit_evidence_status": evidence.get("status"),
    }


def _promotion_contract_product_trace_replay_from_metadata(
    metadata: Mapping[str, Any],
    *,
    contract_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    def value(key: str) -> Any:
        return _first_present(
            metadata.get(f"promotion_contract_{key}"),
            contract_metadata.get(key),
            metadata.get(key),
        )

    replay = {
        "available": False,
        "status": _optional_string(value("product_trace_replay_workflow_status")),
        "report": _optional_string(value("product_trace_replay_workflow_report")),
        "manifest": _optional_string(value("product_trace_replay_workflow_manifest")),
        "source": _optional_string(value("product_trace_replay_workflow_source")),
        "registry": _optional_string(value("product_trace_replay_workflow_registry")),
        "record": _optional_string(value("product_trace_replay_workflow_record")),
        "report_status": _optional_string(
            value("product_trace_replay_workflow_report_status")
        ),
        "selector_replay_report": _optional_string(
            value("product_trace_replay_workflow_selector_replay_report")
        ),
        "runtime_drift_report": _optional_string(
            value("product_trace_replay_workflow_runtime_drift_report")
        ),
        "action_audit_gate": {
            "required": _optional_bool(value("product_trace_action_audit_gate_required")),
            "status": _optional_string(value("product_trace_action_audit_gate_status")),
            "enabled": _optional_bool(value("product_trace_action_audit_gate_enabled")),
            "passed": _optional_bool(value("product_trace_action_audit_gate_passed")),
            "report": _optional_string(value("product_trace_action_audit_gate_report")),
            "error_rate": _finite_float(value("product_trace_action_audit_error_rate")),
            "missing_retrieval_action_rate": _finite_float(
                value("product_trace_action_audit_missing_retrieval_action_rate")
            ),
            "missing_plan_retrieval_query_rate": _finite_float(
                value("product_trace_action_audit_missing_plan_retrieval_query_rate")
            ),
            "malformed_payload_rate": _finite_float(
                value("product_trace_action_audit_malformed_payload_rate")
            ),
            "unexpected_action_rate": _finite_float(
                value("product_trace_action_audit_unexpected_action_rate")
            ),
            "unknown_claim_id_rate": _finite_float(
                value("product_trace_action_audit_unknown_claim_id_rate")
            ),
        },
        "action_execution_gate": {
            "required": _optional_bool(
                value("product_trace_action_execution_gate_required")
            ),
            "status": _optional_string(
                value("product_trace_action_execution_gate_status")
            ),
            "enabled": _optional_bool(
                value("product_trace_action_execution_gate_enabled")
            ),
            "passed": _optional_bool(value("product_trace_action_execution_gate_passed")),
            "report": _optional_string(value("product_trace_action_execution_gate_report")),
            "alignment_failed_trace_rate": _finite_float(
                value("product_trace_action_execution_alignment_failed_trace_rate")
            ),
            "missing_result_rate": _finite_float(
                value("product_trace_action_execution_missing_result_rate")
            ),
            "unexpected_result_rate": _finite_float(
                value("product_trace_action_execution_unexpected_result_rate")
            ),
            "request_id_mismatch_rate": _finite_float(
                value("product_trace_action_execution_request_id_mismatch_rate")
            ),
        },
    }
    replay["available"] = _product_trace_replay_available(replay)
    return replay


def _product_trace_replay_available(replay: Mapping[str, Any]) -> bool:
    for key, item in replay.items():
        if key == "available":
            continue
        if isinstance(item, Mapping):
            if any(value is not None for value in item.values()):
                return True
        elif item is not None:
            return True
    return False


def _promotion_contract_product_trace_replay_metric_values(
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    action_audit = _mapping(replay.get("action_audit_gate"))
    action_execution = _mapping(replay.get("action_execution_gate"))
    return {
        "promotion_contract_product_trace_replay_available": _optional_bool(
            replay.get("available")
        ),
        "promotion_contract_product_trace_replay_workflow_status": replay.get("status"),
        "promotion_contract_product_trace_replay_workflow_report": replay.get("report"),
        "promotion_contract_product_trace_replay_workflow_manifest": replay.get(
            "manifest"
        ),
        "promotion_contract_product_trace_replay_workflow_source": replay.get("source"),
        "promotion_contract_product_trace_replay_workflow_registry": replay.get(
            "registry"
        ),
        "promotion_contract_product_trace_replay_workflow_record": replay.get("record"),
        "promotion_contract_product_trace_replay_workflow_report_status": replay.get(
            "report_status"
        ),
        "promotion_contract_product_trace_replay_workflow_selector_replay_report": (
            replay.get("selector_replay_report")
        ),
        "promotion_contract_product_trace_replay_workflow_runtime_drift_report": (
            replay.get("runtime_drift_report")
        ),
        "promotion_contract_product_trace_action_audit_gate_required": action_audit.get(
            "required"
        ),
        "promotion_contract_product_trace_action_audit_gate_status": action_audit.get(
            "status"
        ),
        "promotion_contract_product_trace_action_audit_gate_enabled": action_audit.get(
            "enabled"
        ),
        "promotion_contract_product_trace_action_audit_gate_passed": action_audit.get(
            "passed"
        ),
        "promotion_contract_product_trace_action_audit_gate_report": action_audit.get(
            "report"
        ),
        "promotion_contract_product_trace_action_audit_error_rate": action_audit.get(
            "error_rate"
        ),
        "promotion_contract_product_trace_action_audit_missing_retrieval_action_rate": (
            action_audit.get("missing_retrieval_action_rate")
        ),
        "promotion_contract_product_trace_action_audit_missing_plan_retrieval_query_rate": (
            action_audit.get("missing_plan_retrieval_query_rate")
        ),
        "promotion_contract_product_trace_action_audit_malformed_payload_rate": (
            action_audit.get("malformed_payload_rate")
        ),
        "promotion_contract_product_trace_action_audit_unexpected_action_rate": (
            action_audit.get("unexpected_action_rate")
        ),
        "promotion_contract_product_trace_action_audit_unknown_claim_id_rate": (
            action_audit.get("unknown_claim_id_rate")
        ),
        "promotion_contract_product_trace_action_execution_gate_required": (
            action_execution.get("required")
        ),
        "promotion_contract_product_trace_action_execution_gate_status": (
            action_execution.get("status")
        ),
        "promotion_contract_product_trace_action_execution_gate_enabled": (
            action_execution.get("enabled")
        ),
        "promotion_contract_product_trace_action_execution_gate_passed": (
            action_execution.get("passed")
        ),
        "promotion_contract_product_trace_action_execution_gate_report": (
            action_execution.get("report")
        ),
        "promotion_contract_product_trace_action_execution_alignment_failed_trace_rate": (
            action_execution.get("alignment_failed_trace_rate")
        ),
        "promotion_contract_product_trace_action_execution_missing_result_rate": (
            action_execution.get("missing_result_rate")
        ),
        "promotion_contract_product_trace_action_execution_unexpected_result_rate": (
            action_execution.get("unexpected_result_rate")
        ),
        "promotion_contract_product_trace_action_execution_request_id_mismatch_rate": (
            action_execution.get("request_id_mismatch_rate")
        ),
    }


def _promotion_contract_runtime_drift_from_metadata(
    metadata: Mapping[str, Any],
    *,
    contract_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    def value(key: str) -> Any:
        return _first_present(
            metadata.get(f"promotion_contract_{key}"),
            contract_metadata.get(key),
            metadata.get(key),
        )

    promotion_evidence = _promotion_contract_runtime_drift_evidence(
        metadata,
        contract_metadata=contract_metadata,
        prefixes=_PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_PREFIXES,
    )
    pre_generation_evidence = _promotion_contract_runtime_drift_evidence(
        metadata,
        contract_metadata=contract_metadata,
        prefixes=_PRODUCT_RUNTIME_DRIFT_PRE_GENERATION_EVIDENCE_PREFIXES,
    )
    counterfactual_evidence = _promotion_contract_runtime_drift_evidence(
        metadata,
        contract_metadata=contract_metadata,
        prefixes=_PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_EVIDENCE_PREFIXES,
    )
    triple_audit_evidence = _promotion_contract_runtime_drift_evidence(
        metadata,
        contract_metadata=contract_metadata,
        prefixes=_PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_PREFIXES,
    )
    covered_fact_property_evidence = _promotion_contract_runtime_drift_evidence(
        metadata,
        contract_metadata=contract_metadata,
        prefixes=_PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_PREFIXES,
    )
    action_gate_evidence = _promotion_contract_runtime_drift_evidence(
        metadata,
        contract_metadata=contract_metadata,
        prefixes=_PRODUCT_RUNTIME_DRIFT_ACTION_GATE_EVIDENCE_PREFIXES,
    )
    trajectory_audit_evidence = _promotion_contract_runtime_drift_evidence(
        metadata,
        contract_metadata=contract_metadata,
        prefixes=_PRODUCT_RUNTIME_DRIFT_TRAJECTORY_AUDIT_EVIDENCE_PREFIXES,
    )
    evidence_handoff_evidence = _promotion_contract_runtime_drift_evidence(
        metadata,
        contract_metadata=contract_metadata,
        prefixes=_PRODUCT_RUNTIME_DRIFT_EVIDENCE_HANDOFF_EVIDENCE_PREFIXES,
    )
    world_model_evidence = _promotion_contract_runtime_drift_evidence(
        metadata,
        contract_metadata=contract_metadata,
        prefixes=_PRODUCT_RUNTIME_DRIFT_WORLD_MODEL_EVIDENCE_PREFIXES,
    )
    context_sensitivity_evidence = _promotion_contract_runtime_drift_evidence(
        metadata,
        contract_metadata=contract_metadata,
        prefixes=_PRODUCT_RUNTIME_DRIFT_CONTEXT_SENSITIVITY_EVIDENCE_PREFIXES,
    )
    counterfactual_robustness_evidence = _promotion_contract_runtime_drift_evidence(
        metadata,
        contract_metadata=contract_metadata,
        prefixes=_PRODUCT_RUNTIME_DRIFT_COUNTERFACTUAL_ROBUSTNESS_EVIDENCE_PREFIXES,
    )
    frontier_release_evidence = _promotion_contract_runtime_drift_evidence(
        metadata,
        contract_metadata=contract_metadata,
        prefixes=_PRODUCT_RUNTIME_DRIFT_FRONTIER_RELEASE_EVIDENCE_PREFIXES,
    )
    drift = {
        "available": False,
        "status": _optional_string(value("product_runtime_drift_status")),
        "report": _optional_string(value("product_runtime_drift_report")),
        "manifest": _optional_string(value("product_runtime_drift_manifest")),
        "baseline_path": _optional_string(value("product_runtime_drift_baseline_path")),
        "current_path": _optional_string(value("product_runtime_drift_current_path")),
        "gate_enabled": _optional_bool(value("product_runtime_drift_gate_enabled")),
        "promotion_evidence_required": _optional_bool(
            value("product_runtime_drift_promotion_evidence_required")
        ),
        "pre_generation_evidence_required": _optional_bool(
            value("product_runtime_drift_pre_generation_evidence_required")
        ),
        "counterfactual_evidence_required": _optional_bool(
            value("product_runtime_drift_counterfactual_evidence_required")
        ),
        "triple_audit_evidence_required": _optional_bool(
            value("product_runtime_drift_triple_audit_evidence_required")
        ),
        "covered_fact_property_evidence_required": _optional_bool(
            value("product_runtime_drift_covered_fact_property_evidence_required")
        ),
        "action_gate_evidence_required": _optional_bool(
            value("product_runtime_drift_action_gate_evidence_required")
        ),
        "trajectory_audit_evidence_required": _optional_bool(
            value("product_runtime_drift_trajectory_audit_evidence_required")
        ),
        "evidence_handoff_evidence_required": _optional_bool(
            value("product_runtime_drift_evidence_handoff_evidence_required")
        ),
        "world_model_evidence_required": _optional_bool(
            value("product_runtime_drift_world_model_evidence_required")
        ),
        "context_sensitivity_evidence_required": _optional_bool(
            value("product_runtime_drift_context_sensitivity_evidence_required")
        ),
        "counterfactual_robustness_evidence_required": _optional_bool(
            value("product_runtime_drift_counterfactual_robustness_evidence_required")
        ),
        "frontier_release_evidence_required": _optional_bool(
            value("product_runtime_drift_frontier_release_evidence_required")
        ),
        "compared_metric_count": _finite_float(
            value("product_runtime_drift_compared_metric_count")
        ),
        "blocked_metric_count": _finite_float(value("product_runtime_drift_blocked_metric_count")),
        "promotion_evidence_metric_count": _finite_float(
            value("product_runtime_drift_promotion_evidence_metric_count")
        ),
        "promotion_evidence_blocked_metric_count": _finite_float(
            value("product_runtime_drift_promotion_evidence_blocked_metric_count")
        ),
        "pre_generation_evidence_metric_count": _finite_float(
            value("product_runtime_drift_pre_generation_evidence_metric_count")
        ),
        "pre_generation_evidence_blocked_metric_count": _finite_float(
            value("product_runtime_drift_pre_generation_evidence_blocked_metric_count")
        ),
        "counterfactual_evidence_metric_count": _finite_float(
            value("product_runtime_drift_counterfactual_evidence_metric_count")
        ),
        "counterfactual_evidence_blocked_metric_count": _finite_float(
            value("product_runtime_drift_counterfactual_evidence_blocked_metric_count")
        ),
        "triple_audit_evidence_metric_count": _finite_float(
            value("product_runtime_drift_triple_audit_evidence_metric_count")
        ),
        "triple_audit_evidence_blocked_metric_count": _finite_float(
            value("product_runtime_drift_triple_audit_evidence_blocked_metric_count")
        ),
        "covered_fact_property_evidence_metric_count": _finite_float(
            value("product_runtime_drift_covered_fact_property_evidence_metric_count")
        ),
        "covered_fact_property_evidence_blocked_metric_count": _finite_float(
            value("product_runtime_drift_covered_fact_property_evidence_blocked_metric_count")
        ),
        "action_gate_evidence_metric_count": _finite_float(
            value("product_runtime_drift_action_gate_evidence_metric_count")
        ),
        "action_gate_evidence_blocked_metric_count": _finite_float(
            value("product_runtime_drift_action_gate_evidence_blocked_metric_count")
        ),
        "trajectory_audit_evidence_metric_count": _finite_float(
            value("product_runtime_drift_trajectory_audit_evidence_metric_count")
        ),
        "trajectory_audit_evidence_blocked_metric_count": _finite_float(
            value("product_runtime_drift_trajectory_audit_evidence_blocked_metric_count")
        ),
        "evidence_handoff_evidence_metric_count": _finite_float(
            value("product_runtime_drift_evidence_handoff_evidence_metric_count")
        ),
        "evidence_handoff_evidence_blocked_metric_count": _finite_float(
            value("product_runtime_drift_evidence_handoff_evidence_blocked_metric_count")
        ),
        "world_model_evidence_metric_count": _finite_float(
            value("product_runtime_drift_world_model_evidence_metric_count")
        ),
        "world_model_evidence_blocked_metric_count": _finite_float(
            value("product_runtime_drift_world_model_evidence_blocked_metric_count")
        ),
        "context_sensitivity_evidence_metric_count": _finite_float(
            value("product_runtime_drift_context_sensitivity_evidence_metric_count")
        ),
        "context_sensitivity_evidence_blocked_metric_count": _finite_float(
            value(
                "product_runtime_drift_context_sensitivity_evidence_blocked_metric_count"
            )
        ),
        "counterfactual_robustness_evidence_metric_count": _finite_float(
            value("product_runtime_drift_counterfactual_robustness_evidence_metric_count")
        ),
        "counterfactual_robustness_evidence_blocked_metric_count": _finite_float(
            value(
                "product_runtime_drift_counterfactual_robustness_evidence_blocked_metric_count"
            )
        ),
        "frontier_release_evidence_metric_count": _finite_float(
            value("product_runtime_drift_frontier_release_evidence_metric_count")
        ),
        "frontier_release_evidence_blocked_metric_count": _finite_float(
            value("product_runtime_drift_frontier_release_evidence_blocked_metric_count")
        ),
        "promotion_evidence": promotion_evidence,
        "pre_generation_evidence": pre_generation_evidence,
        "counterfactual_evidence": counterfactual_evidence,
        "triple_audit_evidence": triple_audit_evidence,
        "covered_fact_property_evidence": covered_fact_property_evidence,
        "action_gate_evidence": action_gate_evidence,
        "trajectory_audit_evidence": trajectory_audit_evidence,
        "evidence_handoff_evidence": evidence_handoff_evidence,
        "world_model_evidence": world_model_evidence,
        "context_sensitivity_evidence": context_sensitivity_evidence,
        "counterfactual_robustness_evidence": counterfactual_robustness_evidence,
        "frontier_release_evidence": frontier_release_evidence,
    }
    drift["available"] = any(
        item is not None
        for key, item in drift.items()
        if key
        not in {
            "available",
            "promotion_evidence",
            "pre_generation_evidence",
            "counterfactual_evidence",
            "triple_audit_evidence",
            "covered_fact_property_evidence",
            "action_gate_evidence",
            "trajectory_audit_evidence",
            "evidence_handoff_evidence",
            "world_model_evidence",
            "context_sensitivity_evidence",
            "counterfactual_robustness_evidence",
            "frontier_release_evidence",
        }
    ) or _runtime_drift_evidence_available(
        promotion_evidence,
        pre_generation_evidence,
        counterfactual_evidence,
        triple_audit_evidence,
        covered_fact_property_evidence,
        action_gate_evidence,
        trajectory_audit_evidence,
        evidence_handoff_evidence,
        world_model_evidence,
        context_sensitivity_evidence,
        counterfactual_robustness_evidence,
        frontier_release_evidence,
    )
    return drift


def _promotion_contract_runtime_drift_evidence(
    metadata: Mapping[str, Any],
    *,
    contract_metadata: Mapping[str, Any],
    prefixes: Sequence[str],
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for prefix in prefixes:
        base_key = f"product_runtime_drift_{prefix}"
        evidence[prefix] = {
            "baseline": _finite_float(
                _first_present(
                    metadata.get(f"promotion_contract_{base_key}_baseline"),
                    contract_metadata.get(f"{base_key}_baseline"),
                    metadata.get(f"{base_key}_baseline"),
                )
            ),
            "current": _finite_float(
                _first_present(
                    metadata.get(f"promotion_contract_{base_key}_current"),
                    contract_metadata.get(f"{base_key}_current"),
                    metadata.get(f"{base_key}_current"),
                )
            ),
            "status": _optional_string(
                _first_present(
                    metadata.get(f"promotion_contract_{base_key}_status"),
                    contract_metadata.get(f"{base_key}_status"),
                    metadata.get(f"{base_key}_status"),
                )
            ),
        }
    return evidence


def _runtime_drift_evidence_available(*groups: Mapping[str, Mapping[str, Any]]) -> bool:
    for group in groups:
        for values in group.values():
            if any(item is not None for item in values.values()):
                return True
    return False


def _promotion_contract_runtime_drift_metric_values(runtime_drift: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {
        "promotion_contract_product_runtime_drift_available": _optional_bool(
            runtime_drift.get("available")
        ),
        "promotion_contract_product_runtime_drift_status": runtime_drift.get("status"),
        "promotion_contract_product_runtime_drift_report": runtime_drift.get("report"),
        "promotion_contract_product_runtime_drift_manifest": runtime_drift.get("manifest"),
        "promotion_contract_product_runtime_drift_baseline_path": runtime_drift.get(
            "baseline_path"
        ),
        "promotion_contract_product_runtime_drift_current_path": runtime_drift.get(
            "current_path"
        ),
        "promotion_contract_product_runtime_drift_gate_enabled": runtime_drift.get(
            "gate_enabled"
        ),
        "promotion_contract_product_runtime_drift_promotion_evidence_required": (
            runtime_drift.get("promotion_evidence_required")
        ),
        "promotion_contract_product_runtime_drift_pre_generation_evidence_required": (
            runtime_drift.get("pre_generation_evidence_required")
        ),
        "promotion_contract_product_runtime_drift_counterfactual_evidence_required": (
            runtime_drift.get("counterfactual_evidence_required")
        ),
        "promotion_contract_product_runtime_drift_triple_audit_evidence_required": (
            runtime_drift.get("triple_audit_evidence_required")
        ),
        "promotion_contract_product_runtime_drift_covered_fact_property_evidence_required": (
            runtime_drift.get("covered_fact_property_evidence_required")
        ),
        "promotion_contract_product_runtime_drift_action_gate_evidence_required": (
            runtime_drift.get("action_gate_evidence_required")
        ),
        "promotion_contract_product_runtime_drift_trajectory_audit_evidence_required": (
            runtime_drift.get("trajectory_audit_evidence_required")
        ),
        "promotion_contract_product_runtime_drift_evidence_handoff_evidence_required": (
            runtime_drift.get("evidence_handoff_evidence_required")
        ),
        "promotion_contract_product_runtime_drift_world_model_evidence_required": (
            runtime_drift.get("world_model_evidence_required")
        ),
        "promotion_contract_product_runtime_drift_context_sensitivity_evidence_required": (
            runtime_drift.get("context_sensitivity_evidence_required")
        ),
        "promotion_contract_product_runtime_drift_counterfactual_robustness_evidence_required": (
            runtime_drift.get("counterfactual_robustness_evidence_required")
        ),
        "promotion_contract_product_runtime_drift_frontier_release_evidence_required": (
            runtime_drift.get("frontier_release_evidence_required")
        ),
        "promotion_contract_product_runtime_drift_compared_metric_count": (
            runtime_drift.get("compared_metric_count")
        ),
        "promotion_contract_product_runtime_drift_blocked_metric_count": (
            runtime_drift.get("blocked_metric_count")
        ),
        "promotion_contract_product_runtime_drift_promotion_evidence_metric_count": (
            runtime_drift.get("promotion_evidence_metric_count")
        ),
        "promotion_contract_product_runtime_drift_promotion_evidence_blocked_metric_count": (
            runtime_drift.get("promotion_evidence_blocked_metric_count")
        ),
        "promotion_contract_product_runtime_drift_pre_generation_evidence_metric_count": (
            runtime_drift.get("pre_generation_evidence_metric_count")
        ),
        "promotion_contract_product_runtime_drift_pre_generation_evidence_blocked_metric_count": (
            runtime_drift.get("pre_generation_evidence_blocked_metric_count")
        ),
        "promotion_contract_product_runtime_drift_counterfactual_evidence_metric_count": (
            runtime_drift.get("counterfactual_evidence_metric_count")
        ),
        "promotion_contract_product_runtime_drift_counterfactual_evidence_blocked_metric_count": (
            runtime_drift.get("counterfactual_evidence_blocked_metric_count")
        ),
        "promotion_contract_product_runtime_drift_triple_audit_evidence_metric_count": (
            runtime_drift.get("triple_audit_evidence_metric_count")
        ),
        "promotion_contract_product_runtime_drift_triple_audit_evidence_blocked_metric_count": (
            runtime_drift.get("triple_audit_evidence_blocked_metric_count")
        ),
        "promotion_contract_product_runtime_drift_covered_fact_property_evidence_metric_count": (
            runtime_drift.get("covered_fact_property_evidence_metric_count")
        ),
        "promotion_contract_product_runtime_drift_covered_fact_property_evidence_blocked_metric_count": (
            runtime_drift.get("covered_fact_property_evidence_blocked_metric_count")
        ),
        "promotion_contract_product_runtime_drift_action_gate_evidence_metric_count": (
            runtime_drift.get("action_gate_evidence_metric_count")
        ),
        "promotion_contract_product_runtime_drift_action_gate_evidence_blocked_metric_count": (
            runtime_drift.get("action_gate_evidence_blocked_metric_count")
        ),
        "promotion_contract_product_runtime_drift_trajectory_audit_evidence_metric_count": (
            runtime_drift.get("trajectory_audit_evidence_metric_count")
        ),
        "promotion_contract_product_runtime_drift_trajectory_audit_evidence_blocked_metric_count": (
            runtime_drift.get("trajectory_audit_evidence_blocked_metric_count")
        ),
        "promotion_contract_product_runtime_drift_evidence_handoff_evidence_metric_count": (
            runtime_drift.get("evidence_handoff_evidence_metric_count")
        ),
        "promotion_contract_product_runtime_drift_evidence_handoff_evidence_blocked_metric_count": (
            runtime_drift.get("evidence_handoff_evidence_blocked_metric_count")
        ),
        "promotion_contract_product_runtime_drift_world_model_evidence_metric_count": (
            runtime_drift.get("world_model_evidence_metric_count")
        ),
        "promotion_contract_product_runtime_drift_world_model_evidence_blocked_metric_count": (
            runtime_drift.get("world_model_evidence_blocked_metric_count")
        ),
        "promotion_contract_product_runtime_drift_context_sensitivity_evidence_metric_count": (
            runtime_drift.get("context_sensitivity_evidence_metric_count")
        ),
        "promotion_contract_product_runtime_drift_context_sensitivity_evidence_blocked_metric_count": (
            runtime_drift.get("context_sensitivity_evidence_blocked_metric_count")
        ),
        "promotion_contract_product_runtime_drift_counterfactual_robustness_evidence_metric_count": (
            runtime_drift.get("counterfactual_robustness_evidence_metric_count")
        ),
        "promotion_contract_product_runtime_drift_counterfactual_robustness_evidence_blocked_metric_count": (
            runtime_drift.get("counterfactual_robustness_evidence_blocked_metric_count")
        ),
        "promotion_contract_product_runtime_drift_frontier_release_evidence_metric_count": (
            runtime_drift.get("frontier_release_evidence_metric_count")
        ),
        "promotion_contract_product_runtime_drift_frontier_release_evidence_blocked_metric_count": (
            runtime_drift.get("frontier_release_evidence_blocked_metric_count")
        ),
    }
    for prefix, values in _mapping(runtime_drift.get("promotion_evidence")).items():
        for suffix in ("baseline", "current", "status"):
            metrics[
                f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            ] = _mapping(values).get(suffix)
    for prefix, values in _mapping(runtime_drift.get("pre_generation_evidence")).items():
        for suffix in ("baseline", "current", "status"):
            metrics[
                f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            ] = _mapping(values).get(suffix)
    for prefix, values in _mapping(runtime_drift.get("counterfactual_evidence")).items():
        for suffix in ("baseline", "current", "status"):
            metrics[
                f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            ] = _mapping(values).get(suffix)
    for prefix, values in _mapping(runtime_drift.get("triple_audit_evidence")).items():
        for suffix in ("baseline", "current", "status"):
            metrics[
                f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            ] = _mapping(values).get(suffix)
    for prefix, values in _mapping(runtime_drift.get("covered_fact_property_evidence")).items():
        for suffix in ("baseline", "current", "status"):
            metrics[
                f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            ] = _mapping(values).get(suffix)
    for prefix, values in _mapping(runtime_drift.get("action_gate_evidence")).items():
        for suffix in ("baseline", "current", "status"):
            metrics[
                f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            ] = _mapping(values).get(suffix)
    for prefix, values in _mapping(runtime_drift.get("trajectory_audit_evidence")).items():
        for suffix in ("baseline", "current", "status"):
            metrics[
                f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            ] = _mapping(values).get(suffix)
    for prefix, values in _mapping(runtime_drift.get("evidence_handoff_evidence")).items():
        for suffix in ("baseline", "current", "status"):
            metrics[
                f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            ] = _mapping(values).get(suffix)
    for prefix, values in _mapping(runtime_drift.get("world_model_evidence")).items():
        for suffix in ("baseline", "current", "status"):
            metrics[
                f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            ] = _mapping(values).get(suffix)
    for prefix, values in _mapping(
        runtime_drift.get("context_sensitivity_evidence")
    ).items():
        for suffix in ("baseline", "current", "status"):
            metrics[
                f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            ] = _mapping(values).get(suffix)
    for prefix, values in _mapping(
        runtime_drift.get("counterfactual_robustness_evidence")
    ).items():
        for suffix in ("baseline", "current", "status"):
            metrics[
                f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            ] = _mapping(values).get(suffix)
    for prefix, values in _mapping(runtime_drift.get("frontier_release_evidence")).items():
        for suffix in ("baseline", "current", "status"):
            metrics[
                f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            ] = _mapping(values).get(suffix)
    return metrics


def _covered_fact_scope_from_metadata(
    metadata: Mapping[str, Any],
    *,
    contract_metadata: Mapping[str, Any],
    verifier_route: Mapping[str, Any],
) -> dict[str, Any]:
    recommended_properties = _string_sequence(
        _first_present(
            metadata.get("promotion_contract_recommended_route_covered_fact_properties"),
            contract_metadata.get("recommended_route_covered_fact_properties"),
            verifier_route.get("covered_fact_properties"),
        )
    )
    recommended_count = _finite_float(
        _first_present(
            metadata.get("promotion_contract_recommended_route_covered_fact_property_count"),
            contract_metadata.get("recommended_route_covered_fact_property_count"),
            verifier_route.get("covered_fact_property_count"),
        )
    )
    if recommended_count is None and recommended_properties:
        recommended_count = float(len(recommended_properties))
    return {
        "recommended_route_count": recommended_count,
        "recommended_route_properties": list(recommended_properties),
        "required_route_baseline_counts": _mapping(
            _first_present(
                metadata.get("promotion_contract_required_route_baseline_covered_fact_property_counts"),
                contract_metadata.get("required_route_baseline_covered_fact_property_counts"),
            )
        ),
        "required_route_baseline_properties": _string_sequence_mapping(
            _first_present(
                metadata.get("promotion_contract_required_route_baseline_covered_fact_properties"),
                contract_metadata.get("required_route_baseline_covered_fact_properties"),
            )
        ),
        "recommended_route_property_metrics": _mapping(
            _first_present(
                metadata.get("promotion_contract_recommended_route_covered_fact_property_metrics"),
                contract_metadata.get("recommended_route_covered_fact_property_metrics"),
                verifier_route.get("covered_fact_property_metrics"),
            )
        ),
        "required_route_baseline_property_metrics": _mapping(
            _first_present(
                metadata.get("promotion_contract_required_route_baseline_covered_fact_property_metrics"),
                contract_metadata.get("required_route_baseline_covered_fact_property_metrics"),
            )
        ),
        "structured_fact_robustness_counts": _mapping(
            _first_present(
                metadata.get("promotion_contract_structured_fact_robustness_property_counts"),
                contract_metadata.get("structured_fact_robustness_property_counts"),
            )
        ),
        "structured_fact_robustness_properties": _string_sequence_mapping(
            _first_present(
                metadata.get("promotion_contract_structured_fact_robustness_properties"),
                contract_metadata.get("structured_fact_robustness_properties"),
            )
        ),
        "structured_fact_robustness_property_metrics": _mapping(
            _first_present(
                metadata.get("promotion_contract_structured_fact_robustness_property_metrics"),
                contract_metadata.get("structured_fact_robustness_property_metrics"),
            )
        ),
    }


def _covered_fact_property_metric_rollups(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    direct_rollups = _direct_covered_fact_property_rollups(metrics)
    if direct_rollups is not None:
        return direct_rollups
    leaves = tuple(_iter_covered_fact_property_metric_leaves(metrics))
    return {
        "property_metric_count": float(len(leaves)) if leaves else None,
        "min_records": _min_finite(item.get("n_records") for item in leaves),
        "min_source_documents": _min_finite(item.get("n_source_documents") for item in leaves),
        "min_decision_accuracy": _min_finite(item.get("decision_accuracy") for item in leaves),
        "max_false_supported_rate": _max_finite(
            item.get("false_supported_rate") for item in leaves
        ),
        "min_false_refuted_rate": _min_finite(item.get("false_refuted_rate") for item in leaves),
    }


def _direct_covered_fact_property_rollups(metrics: Mapping[str, Any]) -> dict[str, float | None] | None:
    mapping = _mapping(metrics)
    if not mapping:
        return None
    rollup_fields = (
        "property_metric_count",
        "min_records",
        "min_source_documents",
        "min_decision_accuracy",
        "max_false_supported_rate",
        "min_false_refuted_rate",
    )
    if not any(field in mapping for field in rollup_fields):
        return None
    return {
        field: _finite_float(mapping.get(field))
        for field in rollup_fields
    }


def _iter_covered_fact_property_metric_leaves(value: Any) -> tuple[Mapping[str, Any], ...]:
    mapping = _mapping(value)
    if not mapping:
        return ()
    metric_keys = {
        "n_records",
        "n_source_documents",
        "decision_accuracy",
        "false_supported_rate",
        "false_refuted_rate",
    }
    if any(key in mapping for key in metric_keys):
        return (mapping,)
    leaves: list[Mapping[str, Any]] = []
    for nested in mapping.values():
        leaves.extend(_iter_covered_fact_property_metric_leaves(nested))
    return tuple(leaves)


def _prefixed_property_rollup_metrics(
    prefix: str,
    rollups: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        f"{prefix}_property_metric_count": rollups.get("property_metric_count"),
        f"{prefix}_min_records": rollups.get("min_records"),
        f"{prefix}_min_source_documents": rollups.get("min_source_documents"),
        f"{prefix}_min_decision_accuracy": rollups.get("min_decision_accuracy"),
        f"{prefix}_max_false_supported_rate": rollups.get("max_false_supported_rate"),
        f"{prefix}_min_false_refuted_rate": rollups.get("min_false_refuted_rate"),
    }


def _min_finite(values: Iterable[Any]) -> float | None:
    finite = tuple(value for raw in values if (value := _finite_float(raw)) is not None)
    return min(finite) if finite else None


def _max_finite(values: Iterable[Any]) -> float | None:
    finite = tuple(value for raw in values if (value := _finite_float(raw)) is not None)
    return max(finite) if finite else None


def _matrix_from_flat_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    matrix = {
        "source": metadata.get("triple_extraction_fixture_matrix_source"),
        "status": metadata.get("triple_extraction_fixture_matrix_status"),
        "n_corpora": metadata.get("triple_extraction_fixture_matrix_n_corpora"),
        "promoted_corpora": metadata.get("triple_extraction_fixture_matrix_promoted_corpora"),
        "distinct_predicate_count": metadata.get(
            "triple_extraction_fixture_matrix_distinct_predicate_count"
        ),
        "mean_best_f1": metadata.get("triple_extraction_fixture_matrix_mean_best_f1"),
        "mean_f1_lift": metadata.get("triple_extraction_fixture_matrix_mean_f1_lift"),
    }
    return {key: value for key, value in matrix.items() if value is not None}


def _pathway_intervention_workflow_from_flat_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    def value(key: str) -> Any:
        return _first_present(
            metadata.get(key),
            metadata.get(f"promotion_contract_{key}"),
        )

    workflow = {
        "report_path": value("pathway_intervention_workflow_report"),
        "manifest_path": value("pathway_intervention_workflow_manifest"),
        "source": value("pathway_intervention_workflow_source"),
        "registry": value("pathway_intervention_workflow_registry"),
        "record_key": _first_present(
            value("pathway_intervention_workflow_record"),
            value("pathway_intervention_workflow_registry_key"),
        ),
        "status": value("pathway_intervention_workflow_status"),
        "report_status": value("pathway_intervention_workflow_report_status"),
        "release_ready": value("pathway_intervention_workflow_release_ready"),
        "model": value("pathway_intervention_workflow_model"),
        "layer": value("pathway_intervention_workflow_layer"),
        "intervention_layer": value(
            "pathway_intervention_workflow_intervention_layer"
        ),
        "patch_layer": value("pathway_intervention_workflow_patch_layer"),
        "activation_ablation_gate_status": value(
            "pathway_intervention_workflow_activation_ablation_gate"
        ),
        "source_patch_gate_status": value(
            "pathway_intervention_workflow_source_patch_gate"
        ),
        "signals": value("pathway_intervention_workflow_signals"),
        "best_signals": value("pathway_intervention_workflow_best_signals"),
    }
    return {key: item for key, item in workflow.items() if item is not None}


def _external_evidence_baseline_comparison_from_flat_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = {
        "report_path": metadata.get("external_evidence_baseline_comparison_report"),
        "source": metadata.get("external_evidence_baseline_comparison_source"),
        "registry": metadata.get("external_evidence_baseline_comparison_registry"),
        "record_key": _first_present(
            metadata.get("external_evidence_baseline_comparison_record"),
            metadata.get("external_evidence_baseline_comparison_registry_key"),
        ),
        "status": metadata.get("external_evidence_baseline_comparison_status"),
        "decision_status": metadata.get(
            "external_evidence_baseline_comparison_decision_status"
        ),
        "recommended_route": metadata.get(
            "external_evidence_baseline_comparison_recommended_route"
        ),
        "recommended_route_record": metadata.get(
            "external_evidence_baseline_comparison_recommended_route_record"
        ),
        "route_passed": metadata.get("external_evidence_baseline_comparison_route_passed"),
        "text_redline_passed": metadata.get(
            "external_evidence_baseline_comparison_text_redline_passed"
        ),
        "text_redline_run_count": metadata.get(
            "external_evidence_baseline_comparison_text_redline_run_count"
        ),
    }
    return {key: value for key, value in comparison.items() if value is not None}


def _frontier_release_evidence_from_flat_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    def value(suffix: str) -> Any:
        return _first_present(
            metadata.get(f"frontier_release_evidence_{suffix}"),
            metadata.get(f"promotion_contract_frontier_release_evidence_{suffix}"),
        )

    evidence = {
        "report_path": value("report"),
        "manifest_path": value("manifest"),
        "source": value("source"),
        "registry": value("registry"),
        "record_key": _first_present(value("record"), value("registry_key")),
        "status": value("status"),
        "workflow": value("workflow"),
        "report_status": value("report_status"),
        "decision_status": value("decision_status"),
        "verifier_track_status": value("verifier_track_status"),
        "abstention_track_status": value("abstention_track_status"),
        "multiple_testing_track_status": value("multiple_testing_track_status"),
        "citation_batch_track_status": value("citation_batch_track_status"),
        "frontier_rerun_rollup_track_status": value("frontier_rerun_rollup_track_status"),
        "base_verifier_track_status": value("base_verifier_track_status"),
        "base_abstention_track_status": value("base_abstention_track_status"),
        "base_detectability_track_status": value("base_detectability_track_status"),
        "base_multiple_testing_track_status": value("base_multiple_testing_track_status"),
        "frontier_rerun_rollup_promoted_tracks": value("frontier_rerun_rollup_promoted_tracks"),
        "frontier_rerun_rollup_report_count": value("frontier_rerun_rollup_report_count"),
        "frontier_rerun_rollup_candidate_count": value("frontier_rerun_rollup_candidate_count"),
        "frontier_rerun_rollup_missing_report_count": value(
            "frontier_rerun_rollup_missing_report_count"
        ),
        "frontier_rerun_rollup_invalid_report_count": value(
            "frontier_rerun_rollup_invalid_report_count"
        ),
        "frontier_rerun_rollup_blocked_candidate_count": value(
            "frontier_rerun_rollup_blocked_candidate_count"
        ),
        "frontier_rerun_rollup_promotion_ready_count": value(
            "frontier_rerun_rollup_promotion_ready_count"
        ),
        "citation_batch_rollup_count": value("citation_batch_rollup_count"),
        "citation_batch_expected_batch_count": value("citation_batch_expected_batch_count"),
        "citation_batch_observed_batch_count": value("citation_batch_observed_batch_count"),
        "citation_batch_missing_expected_batch_count": value(
            "citation_batch_missing_expected_batch_count"
        ),
        "citation_batch_duplicate_batch_count": value("citation_batch_duplicate_batch_count"),
        "citation_batch_unexpected_batch_count": value("citation_batch_unexpected_batch_count"),
        "run_names": value("run_names"),
        "blocking_reasons": value("blocking_reasons"),
    }
    return {key: item for key, item in evidence.items() if item is not None}


def _pre_generation_probe_comparison_from_flat_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    best_run = {
        "name": metadata.get("pre_generation_probe_comparison_best_run"),
        "model": metadata.get("pre_generation_probe_comparison_best_model"),
        "recommended_layer": metadata.get("pre_generation_probe_comparison_best_layer"),
        "test_label_auroc": metadata.get(
            "pre_generation_probe_comparison_best_test_label_auroc"
        ),
        "redline_best_signal": metadata.get(
            "pre_generation_probe_comparison_best_redline_signal"
        ),
        "redline_best_auroc": metadata.get(
            "pre_generation_probe_comparison_best_redline_auroc"
        ),
        "redline_margin": metadata.get(
            "pre_generation_probe_comparison_best_redline_margin"
        ),
    }
    cleaned_best_run = {key: value for key, value in best_run.items() if value is not None}
    comparison = {
        "report_path": metadata.get("pre_generation_probe_comparison_report"),
        "manifest_path": metadata.get("pre_generation_probe_comparison_manifest"),
        "source": metadata.get("pre_generation_probe_comparison_source"),
        "registry": metadata.get("pre_generation_probe_comparison_registry"),
        "record_key": _first_present(
            metadata.get("pre_generation_probe_comparison_record"),
            metadata.get("pre_generation_probe_comparison_registry_key"),
        ),
        "status": metadata.get("pre_generation_probe_comparison_status"),
        "model_count": metadata.get("pre_generation_probe_comparison_model_count"),
        "run_count": metadata.get("pre_generation_probe_comparison_run_count"),
        "redline_passed": metadata.get("pre_generation_probe_comparison_redline_passed"),
        "redline_run_count": metadata.get(
            "pre_generation_probe_comparison_redline_run_count"
        ),
        "best_run": cleaned_best_run or None,
    }
    return {key: value for key, value in comparison.items() if value is not None}


def _claim_factuality_probe_comparison_from_flat_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    best_run = {
        "name": metadata.get("claim_factuality_probe_comparison_best_run"),
        "model": metadata.get("claim_factuality_probe_comparison_best_model"),
        "record_count": metadata.get(
            "claim_factuality_probe_comparison_best_record_count"
        ),
        "recommended_layer": metadata.get("claim_factuality_probe_comparison_best_layer"),
        "test_label_auroc": metadata.get(
            "claim_factuality_probe_comparison_best_test_label_auroc"
        ),
        "test_selective_accuracy": metadata.get(
            "claim_factuality_probe_comparison_best_test_selective_accuracy"
        ),
        "test_selective_coverage": metadata.get(
            "claim_factuality_probe_comparison_best_test_selective_coverage"
        ),
        "conformal_threshold": metadata.get(
            "claim_factuality_probe_comparison_best_conformal_threshold"
        ),
        "redline_best_signal": metadata.get(
            "claim_factuality_probe_comparison_best_redline_signal"
        ),
        "redline_best_auroc": metadata.get(
            "claim_factuality_probe_comparison_best_redline_auroc"
        ),
        "redline_margin": metadata.get(
            "claim_factuality_probe_comparison_best_redline_margin"
        ),
    }
    cleaned_best_run = {key: value for key, value in best_run.items() if value is not None}
    comparison = {
        "report_path": metadata.get("claim_factuality_probe_comparison_report"),
        "manifest_path": metadata.get("claim_factuality_probe_comparison_manifest"),
        "source": metadata.get("claim_factuality_probe_comparison_source"),
        "registry": metadata.get("claim_factuality_probe_comparison_registry"),
        "record_key": _first_present(
            metadata.get("claim_factuality_probe_comparison_record"),
            metadata.get("claim_factuality_probe_comparison_registry_key"),
        ),
        "status": metadata.get("claim_factuality_probe_comparison_status"),
        "report_status": metadata.get("claim_factuality_probe_comparison_report_status"),
        "model_count": metadata.get("claim_factuality_probe_comparison_model_count"),
        "run_count": metadata.get("claim_factuality_probe_comparison_run_count"),
        "redline_passed": metadata.get(
            "claim_factuality_probe_comparison_redline_passed"
        ),
        "redline_run_count": metadata.get(
            "claim_factuality_probe_comparison_redline_run_count"
        ),
        "best_run": cleaned_best_run or None,
    }
    return {key: value for key, value in comparison.items() if value is not None}


def _counterfactual_verification_from_flat_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    audit = {
        "report_path": metadata.get("counterfactual_verification_report"),
        "manifest_path": metadata.get("counterfactual_verification_manifest"),
        "source": metadata.get("counterfactual_verification_source"),
        "registry": metadata.get("counterfactual_verification_registry"),
        "record_key": _first_present(
            metadata.get("counterfactual_verification_record"),
            metadata.get("counterfactual_verification_registry_key"),
        ),
        "status": metadata.get("counterfactual_verification_status"),
        "workflow": metadata.get("counterfactual_verification_workflow"),
        "record_count": metadata.get("counterfactual_verification_record_count"),
        "pass_rate": metadata.get("counterfactual_verification_pass_rate"),
        "false_invariance_rate": metadata.get(
            "counterfactual_verification_false_invariance_rate"
        ),
        "flip_success_count": metadata.get(
            "counterfactual_verification_flip_success_count"
        ),
    }
    return {key: value for key, value in audit.items() if value is not None}


def _route_counts_from_plan(plan: Mapping[str, Any]) -> dict[str, int]:
    route_counts: dict[str, int] = {}
    for hint in _sequence(plan.get("route_hints")):
        if not isinstance(hint, Mapping):
            continue
        for route in _sequence(hint.get("routes")):
            route_name = str(route)
            route_counts[route_name] = route_counts.get(route_name, 0) + 1
    return route_counts


def _int_mapping(value: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, raw_count in _mapping(value).items():
        count = _finite_float(raw_count)
        if count is None:
            continue
        counts[str(key)] = int(count)
    return counts


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


def _min_metric_check(
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
        "limit_type": "min",
        "limit": limit,
        "value": observed,
        "raw_value": None if raw is None else repr(raw),
        "passed": observed is not None and observed >= limit,
    }


def _failure_from_check(check: Mapping[str, Any], *, reason: str | None = None) -> dict[str, Any]:
    if reason is None:
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


def _optional_non_negative_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _required_non_negative_float(value, name=name)


def _phase_budget_mapping(values: Mapping[str, Any], *, field_name: str) -> dict[str, float]:
    budgets = {}
    for raw_name, raw_value in values.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("runtime phase budget names must be non-empty")
        budgets[name] = _required_non_negative_float(
            raw_value,
            name=f"{field_name}.{name}",
        )
    return budgets


def _required_non_negative_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative finite number.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return numeric


def _optional_rate_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _required_rate_float(value, name=name)


def _required_rate_float(value: Any, *, name: str) -> float:
    numeric = _required_non_negative_float(value, name=name)
    if numeric > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
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


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _ratio_or_none(numerator: Any, denominator: Any) -> float | None:
    numerator_value = _finite_float(numerator)
    denominator_value = _finite_float(denominator)
    if numerator_value is None or denominator_value is None or denominator_value == 0.0:
        return None
    return numerator_value / denominator_value


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _string_sequence(value: Any) -> tuple[str, ...]:
    values = []
    for item in _sequence(value):
        if isinstance(item, Mapping) and item.get("_truncated") is True:
            continue
        text = _optional_string(item)
        if text is not None:
            values.append(text)
    return tuple(values)


def _string_sequence_mapping(value: Any) -> dict[str, list[str]]:
    return {
        str(key): list(_string_sequence(items))
        for key, items in _mapping(value).items()
        if str(key)
    }


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)
