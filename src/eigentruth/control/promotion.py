"""Promotion contracts that bridge offline release reports to product control."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from eigentruth.control.runtime_budget import ProductRuntimeBudgetPolicy
from eigentruth.registry import (
    ArtifactManifestVerification,
    ArtifactRegistry,
    RegistryRecord,
    load_and_verify_artifact_manifest,
)


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
    product_trace_replay_workflow: Mapping[str, Any] = field(default_factory=dict)
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
        object.__setattr__(
            self,
            "product_trace_replay_workflow",
            dict(self.product_trace_replay_workflow),
        )
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
                product_trace_replay_workflow=_mapping(
                    payload.get("product_trace_replay_workflow")
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
        adapter_family = _mapping(candidate.get("adapter_family_matrix"))
        required_route_baselines = _mapping(candidate.get("required_route_baselines"))
        product_trace_replay_workflow = _mapping(
            candidate.get("product_trace_replay_workflow")
        )
        selector_replay = _mapping(candidate.get("selector_replay"))
        selector_replay_recommended = _mapping(selector_replay.get("recommended"))
        product_runtime_drift = _mapping(candidate.get("product_runtime_drift"))
        product_runtime_drift_summary = _mapping(product_runtime_drift.get("summary"))
        product_runtime_drift_baseline = _mapping(product_runtime_drift.get("baseline"))
        product_runtime_drift_current = _mapping(product_runtime_drift.get("current"))
        performance_evidence_bundle = _mapping(candidate.get("performance_evidence_bundle"))
        performance_evidence_recommendation = _mapping(
            performance_evidence_bundle.get("recommendation")
        )
        performance_evidence_cost = _mapping(performance_evidence_bundle.get("cost"))
        performance_score_dump_cache = _mapping(performance_evidence_bundle.get("score_dump_cache"))
        performance_score_dump_cache_totals = _mapping(performance_score_dump_cache.get("totals"))
        performance_jsonl_view_cache = _mapping(performance_score_dump_cache_totals.get("jsonl_view"))
        performance_gate = _mapping(comparison.get("performance_baseline_gate"))
        performance_trend_gate = _mapping(performance_gate.get("performance_trend_gate"))
        performance_trend_metrics = _mapping(performance_trend_gate.get("metrics"))
        performance_uncached_trend = _mapping(
            performance_trend_metrics.get("uncached_total_seconds")
        )
        performance_cached_trend = _mapping(
            performance_trend_metrics.get("cached_total_seconds")
        )
        performance_cache_only_trend = _mapping(
            performance_trend_metrics.get("cache_only_total_seconds")
        )
        performance_cache_hit_rate_trend = _mapping(
            performance_trend_metrics.get("score_dump_cache_jsonl_view_hit_rate")
        )
        return cls(
            source_workflow=_optional_str(comparison.get("workflow")),
            source_status=status,
            model_id=_optional_str(candidate.get("model")),
            runtime=_mapping(candidate.get("runtime")),
            verifier_route=_mapping(candidate.get("verifier_route")),
            runtime_budget_policy=product_runtime_budget_policy_from_release_candidate(
                comparison
            ),
            product_trace_replay_workflow=_product_trace_replay_workflow_metadata(
                product_trace_replay_workflow,
                manifests=manifests,
            ),
            metadata={
                "recommended_readiness_record": decision.get("recommended_readiness_record"),
                "recommended_route_record": decision.get("recommended_route_record"),
                "recommended_performance_baseline_record": decision.get(
                    "recommended_performance_baseline_record"
                ),
                "recommended_selector_replay_candidate": decision.get(
                    "recommended_selector_replay_candidate"
                ),
                "recommended_product_runtime_drift_report": decision.get(
                    "recommended_product_runtime_drift_report"
                ),
                "product_trace_replay_workflow_status": decision.get(
                    "product_trace_replay_workflow_status"
                ),
                "product_trace_replay_workflow_report": product_trace_replay_workflow.get(
                    "report_path"
                ),
                "product_trace_replay_workflow_manifest": (
                    product_trace_replay_workflow.get("manifest_path")
                    or manifests.get("product_trace_replay_workflow_manifest")
                ),
                "product_trace_replay_workflow_source": product_trace_replay_workflow.get(
                    "source"
                ),
                "product_trace_replay_workflow_registry": product_trace_replay_workflow.get(
                    "registry"
                ),
                "product_trace_replay_workflow_record": product_trace_replay_workflow.get(
                    "record_key"
                ),
                "product_trace_replay_workflow_report_status": (
                    product_trace_replay_workflow.get("report_status")
                ),
                "product_trace_replay_workflow_selector_replay_report": (
                    product_trace_replay_workflow.get("selector_replay_report_path")
                ),
                "product_trace_replay_workflow_runtime_drift_report": (
                    product_trace_replay_workflow.get("product_runtime_drift_report_path")
                ),
                "performance_baseline_record": candidate.get("performance_baseline_record"),
                "performance_evidence_bundle_status": performance_evidence_bundle.get("status"),
                "performance_evidence_bundle_release_ready": (
                    performance_evidence_bundle.get("release_ready")
                ),
                "performance_cache_tuning_status": (
                    performance_evidence_recommendation.get("cache_tuning_status")
                ),
                "performance_uncached_total_seconds": (
                    performance_evidence_cost.get("uncached_total_seconds")
                ),
                "performance_cached_total_ratio": performance_evidence_cost.get(
                    "cached_total_ratio"
                ),
                "performance_cache_only_total_ratio": performance_evidence_cost.get(
                    "cache_only_total_ratio"
                ),
                "performance_score_dump_cache_required": (
                    config.get("require_performance_score_dump_cache")
                ),
                "performance_score_dump_cache_min_jsonl_view_hit_rate": (
                    config.get("min_performance_score_dump_cache_jsonl_view_hit_rate")
                ),
                "performance_score_dump_cache_source_count": (
                    performance_score_dump_cache.get("source_count")
                ),
                "performance_score_dump_cache_jsonl_view_hit_rate": (
                    performance_jsonl_view_cache.get("hit_rate")
                ),
                "performance_drift_baseline_record": config.get(
                    "performance_drift_baseline_key"
                ),
                "performance_trend_gate_passed": performance_trend_gate.get("passed"),
                "performance_trend_reference_record": performance_trend_gate.get(
                    "reference_record_key"
                ),
                "performance_uncached_total_seconds_ratio_to_drift_baseline": (
                    performance_uncached_trend.get("observed_ratio")
                ),
                "performance_cached_total_seconds_ratio_to_drift_baseline": (
                    performance_cached_trend.get("observed_ratio")
                ),
                "performance_cache_only_total_seconds_ratio_to_drift_baseline": (
                    performance_cache_only_trend.get("observed_ratio")
                ),
                "performance_score_dump_cache_jsonl_view_hit_rate_drop_from_drift_baseline": (
                    performance_cache_hit_rate_trend.get("observed_drop")
                ),
                "recommended_route": decision.get("recommended_route"),
                "selector_replay_status": decision.get("selector_replay_status"),
                "selector_replay_report": selector_replay.get("report_path"),
                "selector_replay_manifest": (
                    selector_replay.get("manifest_path")
                    or manifests.get("selector_replay_manifest")
                ),
                "selector_replay_recommended_policy_path": selector_replay.get(
                    "recommended_policy_path"
                ),
                "selector_replay_recommended": selector_replay_recommended,
                "selector_replay_estimated_cost_units_mean": selector_replay_recommended.get(
                    "estimated_cost_units_mean"
                ),
                "selector_replay_observed_runtime_coverage_rate": selector_replay_recommended.get(
                    "observed_runtime_coverage_rate"
                ),
                "selector_replay_observed_runtime_delta_coverage_rate": (
                    selector_replay_recommended.get("observed_runtime_delta_coverage_rate")
                ),
                "selector_replay_observed_selected_total_seconds_mean": (
                    selector_replay_recommended.get("observed_selected_total_seconds_mean")
                ),
                "selector_replay_observed_selected_minus_original_seconds_mean": (
                    selector_replay_recommended.get("observed_selected_minus_original_seconds_mean")
                ),
                "selector_replay_observed_selected_to_original_ratio_mean": (
                    selector_replay_recommended.get("observed_selected_to_original_ratio_mean")
                ),
                "product_runtime_drift_status": decision.get("product_runtime_drift_status"),
                "product_runtime_drift_report": product_runtime_drift.get("report_path"),
                "product_runtime_drift_manifest": (
                    product_runtime_drift.get("manifest_path")
                    or manifests.get("product_runtime_drift_manifest")
                ),
                "product_runtime_drift_baseline_path": product_runtime_drift_baseline.get("path"),
                "product_runtime_drift_current_path": product_runtime_drift_current.get("path"),
                "product_runtime_drift_gate_enabled": product_runtime_drift_summary.get("gate_enabled"),
                "product_runtime_drift_compared_metric_count": (
                    product_runtime_drift_summary.get("compared_metric_count")
                ),
                "product_runtime_drift_blocked_metric_count": (
                    product_runtime_drift_summary.get("blocked_metric_count")
                ),
                "runtime_profile": config.get("runtime_profile"),
                "inside_trigger_budget_policy": config.get("inside_trigger_budget_policy"),
                "runtime_profile_applied_defaults": config.get(
                    "runtime_profile_applied_defaults"
                ),
                "readiness_manifest": manifests.get("readiness_manifest"),
                "route_manifest": manifests.get("route_manifest"),
                "performance_manifest": manifests.get("performance_manifest"),
                "adapter_family_matrix_report": adapter_family.get("matrix_path"),
                "adapter_family_routes": adapter_family.get("routes"),
                "adapter_family_promoted_routes": adapter_family.get("promoted_routes"),
                "adapter_family_required_routes": adapter_family.get("required_routes"),
                "adapter_family_promotion_status": adapter_family.get("promotion_status"),
                "adapter_family_matrix_manifest": manifests.get("adapter_family_matrix_report"),
                "required_route_baseline_status": decision.get("required_route_baseline_status"),
                "required_route_baseline_records": (
                    required_route_baselines.get("records")
                    or decision.get("required_route_baseline_records")
                ),
                "required_route_baseline_routes": required_route_baselines.get("routes"),
                "required_route_baseline_manifests": required_route_baselines.get("manifest_paths"),
                "required_route_baseline_registry": required_route_baselines.get("registry"),
                "required_route_budget_policy": _required_route_budget_policy(config),
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
            "product_trace_replay_workflow": dict(self.product_trace_replay_workflow),
            "metadata": dict(self.metadata),
        }

    def save_json(self, path: str | Path) -> None:
        """Write the contract payload to JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class LoadedProductPromotionContract:
    """Promotion contract plus the source used to load it."""

    contract: ProductPromotionContract
    source: str
    path: Path | None = None

    def runtime_metadata(self, *, budget_enabled: bool) -> dict[str, Any]:
        """Return ProductTrace metadata for this contract."""
        return product_promotion_contract_metadata(
            self.contract,
            source=self.source,
            budget_enabled=budget_enabled,
        )


def first_existing_product_promotion_contract_path(
    paths: Iterable[str | Path],
) -> Path | None:
    """Return the first existing promotion contract path from an ordered candidate list."""
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None


def load_product_promotion_contract(
    path: str | Path | None = None,
    *,
    default_paths: Iterable[str | Path] = (),
    require_promoted: bool = True,
) -> LoadedProductPromotionContract | None:
    """Load an explicit contract or the first existing default contract.

    Explicit paths are loaded directly and keep normal file errors. When no
    explicit path is supplied and none of the defaults exist, return ``None``.
    """
    resolved_path = (
        Path(path)
        if path is not None
        else first_existing_product_promotion_contract_path(default_paths)
    )
    if resolved_path is None:
        return None
    return LoadedProductPromotionContract(
        contract=ProductPromotionContract.from_json(
            resolved_path,
            require_promoted=require_promoted,
        ),
        source=str(resolved_path),
        path=resolved_path,
    )


@dataclass(frozen=True)
class ProductRuntimeEvidenceBundle:
    """Lazy runtime evidence bundle for a deployable promotion contract."""

    loaded_contract: LoadedProductPromotionContract
    manifest_path: Path | None = None
    registry_path: Path | None = None
    registry_key: str | None = None
    manifest_recursive: bool = True
    _manifest_verification: ArtifactManifestVerification | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )
    _registry_record: RegistryRecord | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )

    @property
    def contract(self) -> ProductPromotionContract:
        """Return the loaded promotion contract."""
        return self.loaded_contract.contract

    @property
    def source(self) -> str:
        """Return the promotion contract source path or label."""
        return self.loaded_contract.source

    @property
    def contract_path(self) -> Path | None:
        """Return the local contract path when it came from a file."""
        return self.loaded_contract.path

    def verify_manifest(self) -> ArtifactManifestVerification | None:
        """Lazily verify the optional artifact manifest."""
        if self.manifest_path is None:
            return None
        if self._manifest_verification is None:
            object.__setattr__(
                self,
                "_manifest_verification",
                load_and_verify_artifact_manifest(
                    self.manifest_path,
                    recursive=self.manifest_recursive,
                ),
            )
        return self._manifest_verification

    def registry_record(self) -> RegistryRecord | None:
        """Lazily resolve the optional registry record."""
        if self.registry_path is None:
            return None
        if self._registry_record is None:
            registry = ArtifactRegistry.load_json(self.registry_path)
            record = (
                registry.get(self.registry_key)
                if self.registry_key is not None
                else _find_product_promotion_contract_record(
                    registry,
                    contract_path=self.contract_path,
                    source=self.source,
                )
            )
            object.__setattr__(self, "_registry_record", record)
        return self._registry_record

    def evidence_metadata(
        self,
        *,
        verify_manifest: bool = False,
        include_registry_record: bool = True,
    ) -> dict[str, Any]:
        """Return JSON-ready provenance metadata for this evidence bundle."""
        manifest_verification = self.verify_manifest() if verify_manifest else None
        registry_record = self.registry_record() if include_registry_record else None
        return {
            "promotion_contract_manifest": (
                None if self.manifest_path is None else str(self.manifest_path)
            ),
            "promotion_contract_manifest_verification": (
                None if manifest_verification is None else manifest_verification.to_dict()
            ),
            "promotion_contract_registry": (
                None if self.registry_path is None else str(self.registry_path)
            ),
            "promotion_contract_registry_key": (
                None if registry_record is None else registry_record.key()
            ),
            "promotion_contract_registry_record": (
                None if registry_record is None else registry_record.to_dict()
            ),
        }

    def runtime_metadata(
        self,
        *,
        budget_enabled: bool,
        verify_manifest: bool = False,
        include_registry_record: bool = True,
    ) -> dict[str, Any]:
        """Return ProductTrace metadata for contract and provenance evidence."""
        return {
            **self.loaded_contract.runtime_metadata(budget_enabled=budget_enabled),
            **self.evidence_metadata(
                verify_manifest=verify_manifest,
                include_registry_record=include_registry_record,
            ),
        }


def load_product_runtime_evidence_bundle(
    path: str | Path | None = None,
    *,
    default_contract_paths: Iterable[str | Path] = (),
    manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    registry_key: str | None = None,
    require_promoted: bool = True,
    manifest_recursive: bool = True,
) -> ProductRuntimeEvidenceBundle | None:
    """Load a promotion contract plus optional manifest and registry provenance."""
    loaded_contract = load_product_promotion_contract(
        path,
        default_paths=default_contract_paths,
        require_promoted=require_promoted,
    )
    if loaded_contract is None:
        return None
    resolved_manifest_path = _resolve_product_promotion_contract_manifest_path(
        loaded_contract.path,
        manifest_path=manifest_path,
    )
    return ProductRuntimeEvidenceBundle(
        loaded_contract=loaded_contract,
        manifest_path=resolved_manifest_path,
        registry_path=None if registry_path is None else Path(registry_path),
        registry_key=registry_key,
        manifest_recursive=manifest_recursive,
    )


def product_promotion_contract_metadata(
    contract: ProductPromotionContract | None,
    *,
    source: str | None,
    budget_enabled: bool,
) -> dict[str, Any]:
    """Return ProductTrace metadata for a promotion contract."""
    if contract is None:
        return {
            "promotion_contract_source": None,
            "promotion_contract_budget_enabled": False,
        }
    return {
        "promotion_contract_source": source,
        "promotion_contract_budget_enabled": budget_enabled,
        "promotion_contract_model_id": contract.model_id,
        "promotion_contract_source_workflow": contract.source_workflow,
        "promotion_contract_source_status": contract.source_status,
        "promotion_contract_runtime": dict(contract.runtime),
        "promotion_contract_verifier_route": dict(contract.verifier_route),
        "promotion_contract_product_trace_replay_workflow": dict(
            contract.product_trace_replay_workflow
        ),
        "promotion_contract_metadata": dict(contract.metadata),
    }


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
        min_verification_skip_rate=config.get("min_verification_skip_rate"),
        max_verified_claim_count=config.get("max_verified_claim_count"),
    )


def _required_route_budget_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: config.get(key)
        for key in (
            "required_route_min_selected",
            "required_route_min_decision_accuracy",
            "required_route_max_false_supported_rate",
            "required_route_min_false_refuted_rate",
            "required_route_max_verified_false_alarm",
            "required_route_min_verified_detection",
            "required_route_max_mean_duration_seconds",
            "required_route_max_p99_duration_seconds",
            "required_route_max_max_duration_seconds",
            "required_route_max_mean_attempted_route_count",
            "required_route_max_retrieval_use_rate",
            "required_route_max_runtime_total_seconds",
            "required_route_max_retrieval_hit_count",
            "required_route_min_claims_cache_hit_rate",
            "required_route_min_verifier_trace_cache_hit_rate",
        )
    }


def _release_candidate_comparison(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("workflow") == "release_candidate_registry_workflow":
        comparison = payload.get("release_candidate_comparison")
        if not isinstance(comparison, Mapping):
            raise ValueError("registry workflow payload is missing release_candidate_comparison.")
        return dict(comparison)
    return dict(payload)


def _product_trace_replay_workflow_metadata(
    workflow: Mapping[str, Any],
    *,
    manifests: Mapping[str, Any],
) -> dict[str, Any]:
    if not workflow:
        return {}
    return {
        "report_path": workflow.get("report_path"),
        "manifest_path": (
            workflow.get("manifest_path")
            or manifests.get("product_trace_replay_workflow_manifest")
        ),
        "source": workflow.get("source"),
        "registry": workflow.get("registry"),
        "record_key": workflow.get("record_key"),
        "report_status": workflow.get("report_status"),
        "selector_replay_report_path": workflow.get("selector_replay_report_path"),
        "product_runtime_drift_report_path": workflow.get(
            "product_runtime_drift_report_path"
        ),
    }


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _resolve_product_promotion_contract_manifest_path(
    contract_path: Path | None,
    *,
    manifest_path: str | Path | None,
) -> Path | None:
    if manifest_path is not None:
        return Path(manifest_path)
    if contract_path is None:
        return None
    sibling_manifest = contract_path.parent / "artifact-manifest.json"
    if sibling_manifest.exists():
        return sibling_manifest
    return None


def _find_product_promotion_contract_record(
    registry: ArtifactRegistry,
    *,
    contract_path: Path | None,
    source: str,
) -> RegistryRecord | None:
    for record in registry.list_records(artifact_type="product_promotion_contract"):
        if _record_path_matches(record.path, contract_path=contract_path, source=source):
            return record
    return None


def _record_path_matches(
    record_path: str,
    *,
    contract_path: Path | None,
    source: str,
) -> bool:
    if record_path == source:
        return True
    if contract_path is None:
        return False
    raw_record_path = Path(record_path)
    try:
        return raw_record_path.resolve() == contract_path.resolve()
    except OSError:
        return False
