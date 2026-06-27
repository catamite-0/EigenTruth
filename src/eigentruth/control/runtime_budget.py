"""Runtime budget checks for product control traces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from eigentruth.control.trace import ProductTrace, RuntimeTrace

_PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_PREFIXES: tuple[str, ...] = (
    "promotion_contract_coverage_rate",
    "triple_extraction_fixture_matrix_coverage_rate",
    "triple_extraction_fixture_matrix_mean_best_f1",
    "triple_extraction_fixture_matrix_mean_f1_lift",
)
_PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_PREFIXES: tuple[str, ...] = (
    "triple_claim_coverage_rate",
    "triple_audit_claim_coverage_rate",
    "triple_audit_pass_rate",
    "triple_slot_coverage_rate",
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
        metrics.update(_route_cost_metrics(trace))
        metrics.update(_verification_stage_metrics(trace))
        metrics.update(_verification_plan_metrics(trace))
        metrics.update(_triple_coverage_metrics(trace))
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
    metrics.update(_route_cost_metrics(trace))
    metrics.update(_verification_stage_metrics(trace))
    metrics.update(_verification_plan_metrics(trace))
    metrics.update(_triple_coverage_metrics(trace))
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
    nested_matrix = _mapping(metadata.get("promotion_contract_triple_extraction_fixture_matrix"))
    matrix = nested_matrix or _matrix_from_flat_metadata(metadata) or _matrix_from_flat_metadata(
        contract_metadata
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
    matrix_available = bool(matrix)
    available = bool(
        source is not None
        or budget_enabled is not None
        or metadata.get("promotion_contract_runtime") is not None
        or contract_metadata
        or matrix_available
        or bool(runtime_drift.get("available"))
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
        "product_runtime_drift": runtime_drift,
        "triple_extraction_fixture_matrix": {
            "available": matrix_available,
            "source": matrix_source,
            "status": matrix_status,
            "manifest_verified": _optional_bool(manifest_verification.get("passed")),
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
    }
    matrix_summary = _mapping(summary["triple_extraction_fixture_matrix"])
    runtime_drift_metrics = _promotion_contract_runtime_drift_metric_values(runtime_drift)
    return {
        "promotion_contract_available": available,
        "promotion_contract_source": source,
        "promotion_contract_source_status": source_status,
        "promotion_contract_budget_enabled": budget_enabled,
        "promotion_contract_summary": summary,
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
        **runtime_drift_metrics,
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
    triple_audit_evidence = _promotion_contract_runtime_drift_evidence(
        metadata,
        contract_metadata=contract_metadata,
        prefixes=_PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_PREFIXES,
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
        "triple_audit_evidence_required": _optional_bool(
            value("product_runtime_drift_triple_audit_evidence_required")
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
        "triple_audit_evidence_metric_count": _finite_float(
            value("product_runtime_drift_triple_audit_evidence_metric_count")
        ),
        "triple_audit_evidence_blocked_metric_count": _finite_float(
            value("product_runtime_drift_triple_audit_evidence_blocked_metric_count")
        ),
        "promotion_evidence": promotion_evidence,
        "triple_audit_evidence": triple_audit_evidence,
    }
    drift["available"] = any(
        item is not None
        for key, item in drift.items()
        if key not in {"available", "promotion_evidence", "triple_audit_evidence"}
    ) or _runtime_drift_evidence_available(promotion_evidence, triple_audit_evidence)
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
        "promotion_contract_product_runtime_drift_triple_audit_evidence_required": (
            runtime_drift.get("triple_audit_evidence_required")
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
        "promotion_contract_product_runtime_drift_triple_audit_evidence_metric_count": (
            runtime_drift.get("triple_audit_evidence_metric_count")
        ),
        "promotion_contract_product_runtime_drift_triple_audit_evidence_blocked_metric_count": (
            runtime_drift.get("triple_audit_evidence_blocked_metric_count")
        ),
    }
    for prefix, values in _mapping(runtime_drift.get("promotion_evidence")).items():
        for suffix in ("baseline", "current", "status"):
            metrics[
                f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            ] = _mapping(values).get(suffix)
    for prefix, values in _mapping(runtime_drift.get("triple_audit_evidence")).items():
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
