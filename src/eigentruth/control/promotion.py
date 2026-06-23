"""Promotion contracts that bridge offline release reports to product control."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from eigentruth.control.runtime_budget import ProductRuntimeBudgetPolicy


@dataclass(frozen=True)
class ProductPromotionContract:
    """Deployable product-control contract derived from a release candidate."""

    model_id: str | None = None
    runtime: Mapping[str, Any] = field(default_factory=dict)
    verifier_route: Mapping[str, Any] = field(default_factory=dict)
    runtime_budget_policy: ProductRuntimeBudgetPolicy | Mapping[str, Any] = field(
        default_factory=ProductRuntimeBudgetPolicy
    )
    source_workflow: str | None = None
    source_status: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        policy = (
            self.runtime_budget_policy
            if isinstance(self.runtime_budget_policy, ProductRuntimeBudgetPolicy)
            else ProductRuntimeBudgetPolicy.from_mapping(self.runtime_budget_policy)
        )
        object.__setattr__(self, "runtime", dict(self.runtime))
        object.__setattr__(self, "verifier_route", dict(self.verifier_route))
        object.__setattr__(self, "runtime_budget_policy", policy)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        require_promoted: bool = True,
    ) -> "ProductPromotionContract":
        """Build a contract from either a contract payload or release report."""
        if payload.get("workflow") == "product_promotion_contract":
            return cls(
                schema_version=int(payload.get("schema_version", 1)),
                source_workflow=_optional_str(payload.get("source_workflow")),
                source_status=_optional_str(payload.get("source_status")),
                model_id=_optional_str(payload.get("model_id")),
                runtime=_mapping(payload.get("runtime")),
                verifier_route=_mapping(payload.get("verifier_route")),
                runtime_budget_policy=ProductRuntimeBudgetPolicy.from_mapping(
                    _mapping(payload.get("runtime_budget_policy"))
                ),
                metadata=_mapping(payload.get("metadata")),
            )
        return cls.from_release_candidate_report(payload, require_promoted=require_promoted)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        require_promoted: bool = True,
    ) -> "ProductPromotionContract":
        """Load a contract or release-candidate report from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("promotion contract JSON must contain an object.")
        return cls.from_mapping(payload, require_promoted=require_promoted)

    @classmethod
    def from_release_candidate_report(
        cls,
        report: Mapping[str, Any],
        *,
        require_promoted: bool = True,
    ) -> "ProductPromotionContract":
        """Build a product contract from a release-candidate comparison report."""
        comparison = _release_candidate_comparison(report)
        decision = _mapping(comparison.get("decision"))
        status = _optional_str(decision.get("status"))
        if require_promoted and status != "promote":
            raise ValueError("release candidate report did not promote.")
        candidate = _mapping(comparison.get("release_candidate"))
        if not candidate:
            raise ValueError("release candidate report does not contain a release_candidate.")
        config = _mapping(comparison.get("config"))
        manifests = _mapping(candidate.get("manifests"))
        return cls(
            source_workflow=_optional_str(comparison.get("workflow")),
            source_status=status,
            model_id=_optional_str(candidate.get("model")),
            runtime=_mapping(candidate.get("runtime")),
            verifier_route=_mapping(candidate.get("verifier_route")),
            runtime_budget_policy=product_runtime_budget_policy_from_release_candidate(
                comparison
            ),
            metadata={
                "recommended_readiness_record": decision.get("recommended_readiness_record"),
                "recommended_route_record": decision.get("recommended_route_record"),
                "recommended_performance_baseline_record": decision.get(
                    "recommended_performance_baseline_record"
                ),
                "performance_baseline_record": candidate.get("performance_baseline_record"),
                "recommended_route": decision.get("recommended_route"),
                "runtime_profile": config.get("runtime_profile"),
                "inside_trigger_budget_policy": config.get("inside_trigger_budget_policy"),
                "runtime_profile_applied_defaults": config.get(
                    "runtime_profile_applied_defaults"
                ),
                "readiness_manifest": manifests.get("readiness_manifest"),
                "route_manifest": manifests.get("route_manifest"),
                "performance_manifest": manifests.get("performance_manifest"),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable contract payload."""
        return {
            "schema_version": self.schema_version,
            "workflow": "product_promotion_contract",
            "source_workflow": self.source_workflow,
            "source_status": self.source_status,
            "model_id": self.model_id,
            "runtime": dict(self.runtime),
            "verifier_route": dict(self.verifier_route),
            "runtime_budget_policy": self.runtime_budget_policy.to_dict(),
            "metadata": dict(self.metadata),
        }

    def save_json(self, path: str | Path) -> None:
        """Write the contract payload to JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def product_runtime_budget_policy_from_release_candidate(
    report: Mapping[str, Any],
) -> ProductRuntimeBudgetPolicy:
    """Build a product runtime budget policy from release-candidate gate config."""
    comparison = _release_candidate_comparison(report)
    config = _mapping(comparison.get("config"))
    named_cache_hit_rates: dict[str, float] = {}
    if config.get("min_claims_cache_hit_rate") is not None:
        named_cache_hit_rates["claims"] = config["min_claims_cache_hit_rate"]
    if config.get("min_verifier_trace_cache_hit_rate") is not None:
        named_cache_hit_rates["verifier_trace"] = config["min_verifier_trace_cache_hit_rate"]
    return ProductRuntimeBudgetPolicy(
        max_total_seconds=config.get("max_runtime_total_seconds"),
        max_mean_route_duration_seconds=config.get("max_mean_duration_seconds"),
        max_p99_route_duration_seconds=config.get("max_p99_duration_seconds"),
        max_route_duration_seconds=config.get("max_max_duration_seconds"),
        max_mean_attempted_route_count=config.get("max_mean_attempted_route_count"),
        max_retrieval_use_rate=config.get("max_retrieval_use_rate"),
        max_retrieval_hit_count=config.get("max_retrieval_hit_count"),
        min_cache_hit_rate=config.get("min_cache_hit_rate"),
        min_named_cache_hit_rate=named_cache_hit_rates,
    )


def _release_candidate_comparison(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("workflow") == "release_candidate_registry_workflow":
        comparison = payload.get("release_candidate_comparison")
        if not isinstance(comparison, Mapping):
            raise ValueError("registry workflow payload is missing release_candidate_comparison.")
        return dict(comparison)
    return dict(payload)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
