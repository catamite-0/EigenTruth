"""Sweep product runtime profiles with deterministic demo ProductTrace runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import planned_artifact_manifest_summary, strict_bool, strict_positive_int  # noqa: E402
from benchmarks.run_product_runtime_baseline import (  # noqa: E402
    ProductRuntimeBaselineConfig,
    build_product_runtime_baseline,
)
from eigentruth.control import (  # noqa: E402
    RUNTIME_PROFILE_NAMES,
    ProductRuntimeBudgetPolicy,
    get_runtime_profile,
)
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402
from examples import calibrated_control_demo as demo  # noqa: E402

PRODUCT_RUNTIME_PROFILE_SWEEP_MODES = (*RUNTIME_PROFILE_NAMES, "auto")


@dataclass(frozen=True)
class ProductRuntimeScenario:
    """One deterministic calibrated-control demo scenario."""

    name: str
    text: str
    diagnostics_mode: str = "low"
    facts: Mapping[str, Any] | None = None
    evidence: Sequence[Any] | None = None
    refutations: Mapping[str, Any] | None = None
    retrieval_evidence: Sequence[Any] | None = None
    enable_calculator: bool = False
    calculator_context: Mapping[str, Any] | None = None
    staged_verification: bool | None = None

    def __post_init__(self) -> None:
        name = self.name.strip().lower().replace(" ", "_")
        if not name:
            raise ValueError("scenario name must be non-empty.")
        if self.diagnostics_mode not in {"low", "trigger", "none"}:
            raise ValueError("diagnostics_mode must be one of: low, trigger, none.")
        object.__setattr__(self, "name", name)


DEFAULT_SCENARIOS: tuple[ProductRuntimeScenario, ...] = (
    ProductRuntimeScenario(
        name="low_risk_supported",
        text="Paris is the capital of France.",
        diagnostics_mode="low",
        facts={"Paris is the capital of France": "supported"},
    ),
    ProductRuntimeScenario(
        name="diagnostic_refuted",
        text=demo.DEFAULT_TEXT,
        diagnostics_mode="trigger",
    ),
    ProductRuntimeScenario(
        name="calculator_refuted",
        text="2 + 2 = 5.",
        diagnostics_mode="low",
        enable_calculator=True,
    ),
)


@dataclass(frozen=True)
class ProductRuntimeProfileSweepConfig:
    """Configuration for sweeping product runtime profiles."""

    output_dir: str | Path
    profiles: Sequence[str] = PRODUCT_RUNTIME_PROFILE_SWEEP_MODES
    scenarios: Sequence[ProductRuntimeScenario] = DEFAULT_SCENARIOS
    repeats: int = 1
    artifact_path: str | Path | None = None
    promotion_contract_path: str | Path | None = None
    runtime_profile_selector_policy_path: str | Path | None = None
    policy: ProductRuntimeBudgetPolicy | Mapping[str, Any] | None = None
    policy_path: str | Path | None = None
    slo_policy: "ProductRuntimeProfileSLOPolicy | Mapping[str, Any] | None" = None
    slo_policy_path: str | Path | None = None
    report_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    max_workers: int = 1
    compact_json: bool = False

    def __post_init__(self) -> None:
        output_dir = Path(self.output_dir)
        profiles = tuple(_normalize_profile(profile) for profile in self.profiles)
        if not profiles:
            raise ValueError("at least one runtime profile is required.")
        scenarios = tuple(self.scenarios)
        if not scenarios:
            raise ValueError("at least one runtime scenario is required.")
        repeats = strict_positive_int(self.repeats, name="repeats")
        max_workers = strict_positive_int(self.max_workers, name="max_workers")
        if self.policy is not None and (self.policy_path is not None or self.promotion_contract_path is not None):
            raise ValueError("policy object is mutually exclusive with policy_path and promotion_contract_path.")
        if self.policy_path is not None and self.promotion_contract_path is not None:
            raise ValueError("policy_path and promotion_contract_path are mutually exclusive for baseline gating.")
        if self.slo_policy is not None and self.slo_policy_path is not None:
            raise ValueError("slo_policy object is mutually exclusive with slo_policy_path.")
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "repeats", repeats)
        object.__setattr__(self, "max_workers", max_workers)
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))
        if self.artifact_path is not None:
            object.__setattr__(self, "artifact_path", Path(self.artifact_path))
        if self.promotion_contract_path is not None:
            object.__setattr__(self, "promotion_contract_path", Path(self.promotion_contract_path))
        if self.runtime_profile_selector_policy_path is not None:
            object.__setattr__(
                self,
                "runtime_profile_selector_policy_path",
                Path(self.runtime_profile_selector_policy_path),
            )
        if self.policy_path is not None:
            object.__setattr__(self, "policy_path", Path(self.policy_path))
        if self.slo_policy_path is not None:
            object.__setattr__(self, "slo_policy_path", Path(self.slo_policy_path))
        if self.report_path is not None:
            object.__setattr__(self, "report_path", Path(self.report_path))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def resolved_report_path(self) -> Path:
        """Return the top-level sweep report path."""
        if self.report_path is not None:
            return Path(self.report_path)
        return Path(self.output_dir) / "product-runtime-profile-sweep.json"

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        """Return the top-level artifact manifest path."""
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return Path(self.output_dir) / "artifact-manifest.json"


@dataclass(frozen=True)
class ProductRuntimeProfileSLOPolicy:
    """Aggregate SLO gates for one product runtime profile sweep row."""

    max_total_seconds_mean: float | None = None
    max_total_seconds_p95: float | None = None
    max_measured_phases_mean: float | None = None
    max_mean_route_duration_seconds: float | None = None
    max_mean_attempted_route_count: float | None = None
    max_retrieval_use_rate: float | None = None
    max_retrieval_hit_count: float | None = None
    min_cache_hit_rate_mean: float | None = None
    min_verification_skip_rate_mean: float | None = None
    max_verified_claim_count_mean: float | None = None
    min_auto_selected_profile_counts: Mapping[str, int] = field(default_factory=dict)
    max_auto_selected_profile_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_total_seconds_mean",
            _optional_non_negative_float(self.max_total_seconds_mean, name="max_total_seconds_mean"),
        )
        object.__setattr__(
            self,
            "max_total_seconds_p95",
            _optional_non_negative_float(self.max_total_seconds_p95, name="max_total_seconds_p95"),
        )
        object.__setattr__(
            self,
            "max_measured_phases_mean",
            _optional_non_negative_float(self.max_measured_phases_mean, name="max_measured_phases_mean"),
        )
        object.__setattr__(
            self,
            "max_mean_route_duration_seconds",
            _optional_non_negative_float(
                self.max_mean_route_duration_seconds,
                name="max_mean_route_duration_seconds",
            ),
        )
        object.__setattr__(
            self,
            "max_mean_attempted_route_count",
            _optional_non_negative_float(
                self.max_mean_attempted_route_count,
                name="max_mean_attempted_route_count",
            ),
        )
        object.__setattr__(
            self,
            "max_retrieval_use_rate",
            _optional_rate_float(self.max_retrieval_use_rate, name="max_retrieval_use_rate"),
        )
        object.__setattr__(
            self,
            "max_retrieval_hit_count",
            _optional_non_negative_float(self.max_retrieval_hit_count, name="max_retrieval_hit_count"),
        )
        object.__setattr__(
            self,
            "min_cache_hit_rate_mean",
            _optional_rate_float(self.min_cache_hit_rate_mean, name="min_cache_hit_rate_mean"),
        )
        object.__setattr__(
            self,
            "min_verification_skip_rate_mean",
            _optional_rate_float(
                self.min_verification_skip_rate_mean,
                name="min_verification_skip_rate_mean",
            ),
        )
        object.__setattr__(
            self,
            "max_verified_claim_count_mean",
            _optional_non_negative_float(
                self.max_verified_claim_count_mean,
                name="max_verified_claim_count_mean",
            ),
        )
        object.__setattr__(
            self,
            "min_auto_selected_profile_counts",
            _profile_count_mapping(
                self.min_auto_selected_profile_counts,
                field_name="min_auto_selected_profile_counts",
            ),
        )
        object.__setattr__(
            self,
            "max_auto_selected_profile_counts",
            _profile_count_mapping(
                self.max_auto_selected_profile_counts,
                field_name="max_auto_selected_profile_counts",
            ),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProductRuntimeProfileSLOPolicy":
        """Build a sweep-level SLO policy from a JSON-like mapping."""
        return cls(
            max_total_seconds_mean=payload.get("max_total_seconds_mean"),
            max_total_seconds_p95=payload.get("max_total_seconds_p95"),
            max_measured_phases_mean=payload.get("max_measured_phases_mean"),
            max_mean_route_duration_seconds=payload.get("max_mean_route_duration_seconds"),
            max_mean_attempted_route_count=payload.get("max_mean_attempted_route_count"),
            max_retrieval_use_rate=payload.get("max_retrieval_use_rate"),
            max_retrieval_hit_count=payload.get("max_retrieval_hit_count"),
            min_cache_hit_rate_mean=payload.get("min_cache_hit_rate_mean"),
            min_verification_skip_rate_mean=payload.get("min_verification_skip_rate_mean"),
            max_verified_claim_count_mean=payload.get("max_verified_claim_count_mean"),
            min_auto_selected_profile_counts=dict(_mapping(payload.get("min_auto_selected_profile_counts"))),
            max_auto_selected_profile_counts=dict(_mapping(payload.get("max_auto_selected_profile_counts"))),
        )

    def enabled(self) -> bool:
        """Return whether this policy has any active SLO threshold."""
        return (
            self.max_total_seconds_mean is not None
            or self.max_total_seconds_p95 is not None
            or self.max_measured_phases_mean is not None
            or self.max_mean_route_duration_seconds is not None
            or self.max_mean_attempted_route_count is not None
            or self.max_retrieval_use_rate is not None
            or self.max_retrieval_hit_count is not None
            or self.min_cache_hit_rate_mean is not None
            or self.min_verification_skip_rate_mean is not None
            or self.max_verified_claim_count_mean is not None
            or bool(self.min_auto_selected_profile_counts)
            or bool(self.max_auto_selected_profile_counts)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "max_total_seconds_mean": self.max_total_seconds_mean,
            "max_total_seconds_p95": self.max_total_seconds_p95,
            "max_measured_phases_mean": self.max_measured_phases_mean,
            "max_mean_route_duration_seconds": self.max_mean_route_duration_seconds,
            "max_mean_attempted_route_count": self.max_mean_attempted_route_count,
            "max_retrieval_use_rate": self.max_retrieval_use_rate,
            "max_retrieval_hit_count": self.max_retrieval_hit_count,
            "min_cache_hit_rate_mean": self.min_cache_hit_rate_mean,
            "min_verification_skip_rate_mean": self.min_verification_skip_rate_mean,
            "max_verified_claim_count_mean": self.max_verified_claim_count_mean,
            "min_auto_selected_profile_counts": dict(self.min_auto_selected_profile_counts),
            "max_auto_selected_profile_counts": dict(self.max_auto_selected_profile_counts),
        }


def run_product_runtime_profile_sweep(config: ProductRuntimeProfileSweepConfig) -> dict[str, Any]:
    """Run deterministic demo traces for each runtime profile and compare baselines."""
    started_at = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = demo.load_artifact(None if config.artifact_path is None else str(config.artifact_path))
    slo_policy, slo_policy_source = _load_slo_policy(config)
    profiles = _run_profiles(config, artifact=artifact, slo_policy=slo_policy)

    leaderboard = _leaderboard(profiles)
    recommendation = leaderboard[0] if leaderboard else None
    slo_summary = _slo_summary(profiles, policy=slo_policy, policy_source=slo_policy_source)
    status = _sweep_status(profiles, slo=slo_summary)
    report = {
        "schema_version": 1,
        "workflow": "product_runtime_profile_sweep",
        "status": status,
        "decision": {
            "status": status,
            "recommended_profile": None if recommendation is None else recommendation["profile"],
            "blocking_reasons": _blocking_reasons(profiles, slo=slo_summary),
        },
        "profiles": profiles,
        "leaderboard": leaderboard,
        "slo": slo_summary,
        "paths": {
            "report": str(config.resolved_report_path),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "output_dir": str(config.output_dir),
            "policy": None if config.policy_path is None else str(config.policy_path),
            "slo_policy": None if config.slo_policy_path is None else str(config.slo_policy_path),
            "runtime_profile_selector_policy": (
                None
                if config.runtime_profile_selector_policy_path is None
                else str(config.runtime_profile_selector_policy_path)
            ),
            "promotion_contract": (
                None if config.promotion_contract_path is None else str(config.promotion_contract_path)
            ),
        },
        "config": {
            "profiles": tuple(config.profiles),
            "profile_modes": tuple(config.profiles),
            "scenario_names": tuple(scenario.name for scenario in config.scenarios),
            "repeats": config.repeats,
            "artifact_path": None if config.artifact_path is None else str(config.artifact_path),
            "runtime_profile_selector_policy_path": (
                None
                if config.runtime_profile_selector_policy_path is None
                else str(config.runtime_profile_selector_policy_path)
            ),
            "max_workers": config.max_workers,
            "compact_json": config.compact_json,
            "slo_policy_source": slo_policy_source,
            "metadata": dict(config.metadata),
        },
        "execution": {
            "wall_clock_seconds": time.perf_counter() - started_at,
            "max_workers": config.max_workers,
        },
    }
    _write_report_and_manifest(config, report)
    _record_registry(config, report)
    return report


def _write_report_and_manifest(
    config: ProductRuntimeProfileSweepConfig,
    report: dict[str, Any],
) -> dict[str, Any]:
    artifacts = _artifact_paths(config, report)
    report["artifact_manifest_summary"] = planned_artifact_manifest_summary(
        artifacts,
        assume_file_paths=(config.resolved_report_path,),
    )
    _write_json(config.resolved_report_path, report, compact=config.compact_json)
    return _write_artifact_manifest(config, report, artifacts=artifacts)


def _run_profiles(
    config: ProductRuntimeProfileSweepConfig,
    *,
    artifact: Any,
    slo_policy: ProductRuntimeProfileSLOPolicy | None,
) -> list[dict[str, Any]]:
    if config.max_workers <= 1 or len(config.profiles) <= 1:
        return [
            _run_profile(config, profile_name, artifact=artifact, slo_policy=slo_policy)
            for profile_name in config.profiles
        ]

    records_by_profile: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(config.max_workers, len(config.profiles))) as executor:
        futures = {
            executor.submit(_run_profile, config, profile_name, artifact=artifact, slo_policy=slo_policy): profile_name
            for profile_name in config.profiles
        }
        for future in as_completed(futures):
            profile_name = futures[future]
            records_by_profile[profile_name] = future.result()
    return [records_by_profile[profile_name] for profile_name in config.profiles]


def _run_profile(
    config: ProductRuntimeProfileSweepConfig,
    profile_name: str,
    *,
    artifact: Any,
    slo_policy: ProductRuntimeProfileSLOPolicy | None,
) -> dict[str, Any]:
    traces = _run_profile_traces(config, profile_name, artifact=artifact)
    baseline = build_product_runtime_baseline(
        ProductRuntimeBaselineConfig(
            trace_paths=tuple(trace["path"] for trace in traces),
            report_path=_profile_baseline_path(config, profile_name),
            policy=config.policy,
            policy_path=config.policy_path,
            promotion_contract_path=config.promotion_contract_path,
            compact_json=config.compact_json,
            metadata={
                "source": "run_product_runtime_profile_sweep",
                "runtime_profile": profile_name,
                **dict(config.metadata),
            },
        )
    )
    return _profile_record(profile_name, traces=traces, baseline=baseline, slo_policy=slo_policy)


def _run_profile_traces(
    config: ProductRuntimeProfileSweepConfig,
    profile_name: str,
    *,
    artifact: Any,
) -> tuple[dict[str, Any], ...]:
    traces = []
    for repeat_index in range(config.repeats):
        for scenario in config.scenarios:
            request_id = f"{profile_name}-{scenario.name}-r{repeat_index}"
            output_path = _trace_path(config, profile_name, scenario.name, repeat_index)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = demo.run(
                _demo_args(
                    config,
                    profile_name=profile_name,
                    scenario=scenario,
                    request_id=request_id,
                    output_path=output_path,
                    artifact=artifact,
                )
            )
            traces.append({
                "path": str(output_path),
                "request_id": request_id,
                "scenario": scenario.name,
                "repeat": repeat_index,
                "risk_action": _nested(payload, "risk_decision", "action"),
                "risk_level": _nested(payload, "risk_decision", "risk_level"),
                "selected_runtime_profile": _nested(payload, "metadata", "runtime_profile"),
                "runtime_profile_selection": _nested(payload, "metadata", "runtime_profile_selection"),
                "staged_verification_enabled": _nested(payload, "metadata", "staged_verification_enabled"),
                "runtime_total_seconds": _nested(payload, "runtime_trace", "summary", "total_seconds"),
                "measured_phases": _nested(payload, "runtime_trace", "summary", "measured_phases"),
            })
    return tuple(traces)


def _demo_args(
    config: ProductRuntimeProfileSweepConfig,
    *,
    profile_name: str,
    scenario: ProductRuntimeScenario,
    request_id: str,
    output_path: Path,
    artifact: Any,
) -> SimpleNamespace:
    return SimpleNamespace(
        artifact=None if config.artifact_path is None else str(config.artifact_path),
        diagnostics=_diagnostics_json(scenario, artifact),
        text=scenario.text,
        facts=_json_or_none(scenario.facts),
        evidence=_json_or_none(scenario.evidence),
        refutations=_json_or_none(scenario.refutations),
        retrieval_evidence=_json_or_none(scenario.retrieval_evidence),
        enable_calculator=scenario.enable_calculator,
        calculator_context=_json_or_none(scenario.calculator_context),
        runtime_profile=profile_name,
        staged_verification=scenario.staged_verification,
        runtime_trace=True,
        promotion_contract=None if config.promotion_contract_path is None else str(config.promotion_contract_path),
        runtime_profile_selector_policy=(
            None
            if config.runtime_profile_selector_policy_path is None
            else str(config.runtime_profile_selector_policy_path)
        ),
        cache_verifier=True,
        cache_retriever=True,
        max_runtime_total_seconds=None,
        max_runtime_phase_seconds=None,
        max_runtime_phase_p95_seconds=None,
        max_runtime_phase_p99_seconds=None,
        max_mean_route_duration_seconds=None,
        max_p95_route_duration_seconds=None,
        max_p99_route_duration_seconds=None,
        max_route_duration_seconds=None,
        max_mean_attempted_route_count=None,
        max_retrieval_use_rate=None,
        max_retrieval_hit_count=None,
        min_cache_hit_rate=None,
        min_named_cache_hit_rate=None,
        min_verification_skip_rate=None,
        max_verified_claim_count=None,
        request_id=request_id,
        output=str(output_path),
        registry=None,
        compact_json=config.compact_json,
    )


def _diagnostics_json(scenario: ProductRuntimeScenario, artifact: Any) -> str | None:
    if scenario.diagnostics_mode == "none":
        return None
    if scenario.diagnostics_mode == "trigger":
        return json.dumps(demo.default_diagnostics_for_artifact(artifact), sort_keys=True)
    return json.dumps(demo.low_diagnostics_for_artifact(artifact), sort_keys=True)


def _profile_record(
    profile_name: str,
    *,
    traces: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    slo_policy: ProductRuntimeProfileSLOPolicy | None,
) -> dict[str, Any]:
    summary = _mapping(baseline.get("summary"))
    routes = _mapping(_mapping(summary.get("routes")).get("overall"))
    record = {
        "profile": profile_name,
        "status": baseline.get("status"),
        "baseline_status": baseline.get("status"),
        "trace_count": len(traces),
        "baseline_path": _nested(baseline, "paths", "report"),
        "baseline_artifact_manifest": _nested(baseline, "paths", "artifact_manifest"),
        "trace_paths": tuple(str(trace["path"]) for trace in traces),
        "traces": tuple(dict(trace) for trace in traces),
        "metrics": {
            "total_seconds_mean": _nested(summary, "total_seconds", "mean"),
            "total_seconds_p95": _nested(summary, "total_seconds", "p95"),
            "measured_phases_mean": _nested(summary, "measured_phases", "mean"),
            "mean_route_duration_seconds": routes.get("mean_duration_seconds"),
            "mean_attempted_route_count": routes.get("mean_attempted_route_count"),
            "retrieval_use_rate": routes.get("retrieval_use_rate"),
            "retrieval_hit_count": routes.get("retrieval_hit_count"),
            "cache_hit_rate_mean": _nested(summary, "cache_hit_rate", "mean"),
            "verification_skip_rate_mean": _nested(summary, "verification_skip_rate", "mean"),
            "verified_claim_count_mean": _nested(summary, "verified_claim_count", "mean"),
            "verifier_saved_claim_count_mean": _nested(summary, "verifier_saved_claim_count", "mean"),
            "verification_claim_skip_rate": _nested(summary, "verification_stage", "claim_skip_rate"),
            "verification_skip_decision_rate": _nested(summary, "verification_stage", "skip_decision_rate"),
            "verification_triggered_scope_trace_count": _nested(
                summary,
                "verification_stage",
                "triggered_scope_trace_count",
            ),
            "verification_partial_skip_trace_count": _nested(
                summary,
                "verification_stage",
                "partial_skip_trace_count",
            ),
            "verification_selective_claim_skip_rate": _nested(
                summary,
                "verification_stage",
                "selective_claim_skip_rate",
            ),
        },
        "runtime_profile_selection": _runtime_profile_selection_summary(traces),
        "budget": _mapping(baseline.get("budget")),
    }
    slo = _evaluate_profile_slo(record, slo_policy)
    record["slo"] = slo
    record["status"] = _profile_status(
        baseline_status=str(baseline.get("status")),
        slo=slo,
    )
    return record


def _profile_status(*, baseline_status: str, slo: Mapping[str, Any]) -> str:
    if baseline_status == "blocked":
        return "blocked"
    if bool(slo.get("enabled")) and slo.get("passed") is not True:
        return "blocked"
    if baseline_status == "promote" or (bool(slo.get("enabled")) and slo.get("passed") is True):
        return "promote"
    return "observed"


def _runtime_profile_selection_summary(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts_by_selected_profile: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for trace in traces:
        selected = trace.get("selected_runtime_profile")
        if selected is not None:
            profile = str(selected)
            counts_by_selected_profile[profile] = counts_by_selected_profile.get(profile, 0) + 1
        selection = _mapping(trace.get("runtime_profile_selection"))
        reason = selection.get("reason")
        if reason is not None:
            reason_key = str(reason)
            reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
    return {
        "counts_by_selected_profile": counts_by_selected_profile,
        "reason_counts": reason_counts,
    }


def _evaluate_profile_slo(
    profile: Mapping[str, Any],
    policy: ProductRuntimeProfileSLOPolicy | None,
) -> dict[str, Any]:
    if policy is None or not policy.enabled():
        return {
            "enabled": False,
            "passed": None,
            "policy": None if policy is None else policy.to_dict(),
            "checks": (),
            "failures": (),
        }
    metrics = _mapping(profile.get("metrics"))
    checks = []
    failures = []
    for metric, limit in (
        ("total_seconds_mean", policy.max_total_seconds_mean),
        ("total_seconds_p95", policy.max_total_seconds_p95),
        ("measured_phases_mean", policy.max_measured_phases_mean),
        ("mean_route_duration_seconds", policy.max_mean_route_duration_seconds),
        ("mean_attempted_route_count", policy.max_mean_attempted_route_count),
        ("retrieval_use_rate", policy.max_retrieval_use_rate),
        ("retrieval_hit_count", policy.max_retrieval_hit_count),
        ("verified_claim_count_mean", policy.max_verified_claim_count_mean),
    ):
        if limit is None:
            continue
        check = _max_slo_check(metrics, metric=metric, limit=limit)
        checks.append(check)
        if not check["passed"]:
            failures.append(_slo_failure_from_check(check))
    for metric, limit in (
        ("cache_hit_rate_mean", policy.min_cache_hit_rate_mean),
        ("verification_skip_rate_mean", policy.min_verification_skip_rate_mean),
    ):
        if limit is None:
            continue
        check = _min_slo_check(metrics, metric=metric, limit=limit)
        checks.append(check)
        if not check["passed"]:
            failures.append(_slo_failure_from_check(check))
    if profile.get("profile") == "auto":
        selection = _mapping(profile.get("runtime_profile_selection"))
        counts = _mapping(selection.get("counts_by_selected_profile"))
        for selected_profile, limit in policy.min_auto_selected_profile_counts.items():
            metric = f"auto_selected_profile_count.{selected_profile}"
            check = _min_slo_check(
                counts,
                metric=selected_profile,
                output_metric=metric,
                limit=float(limit),
            )
            checks.append(check)
            if not check["passed"]:
                failures.append(_slo_failure_from_check(check))
        for selected_profile, limit in policy.max_auto_selected_profile_counts.items():
            metric = f"auto_selected_profile_count.{selected_profile}"
            check = _max_slo_check(
                counts,
                metric=selected_profile,
                output_metric=metric,
                limit=float(limit),
            )
            checks.append(check)
            if not check["passed"]:
                failures.append(_slo_failure_from_check(check))
    return {
        "enabled": policy.enabled(),
        "passed": not failures,
        "policy": policy.to_dict(),
        "checks": tuple(checks),
        "failures": tuple(failures),
    }


def _slo_summary(
    profiles: Sequence[Mapping[str, Any]],
    *,
    policy: ProductRuntimeProfileSLOPolicy | None,
    policy_source: str | None,
) -> dict[str, Any]:
    if policy is None or not policy.enabled():
        return {
            "enabled": False,
            "passed": None,
            "policy": None if policy is None else policy.to_dict(),
            "policy_source": policy_source,
            "passed_profile_count": None,
            "failed_profile_count": None,
            "failure_counts_by_metric": {},
            "failures": (),
        }
    failed_profiles = [
        profile
        for profile in profiles
        if _mapping(profile.get("slo")).get("passed") is not True
    ]
    failure_counts: dict[str, int] = {}
    failures = []
    for profile in failed_profiles:
        for failure in _sequence(_mapping(profile.get("slo")).get("failures")):
            if not isinstance(failure, Mapping):
                continue
            metric = str(failure.get("metric", "unknown"))
            failure_counts[metric] = failure_counts.get(metric, 0) + 1
    if (
        policy.min_auto_selected_profile_counts
        or policy.max_auto_selected_profile_counts
    ) and not any(profile.get("profile") == "auto" for profile in profiles):
        failure = {
            "metric": "auto_profile",
            "limit_type": "required",
            "limit": True,
            "value": False,
            "reason": "auto profile missing for auto selector SLO",
        }
        failures.append(failure)
        failure_counts["auto_profile"] = failure_counts.get("auto_profile", 0) + 1
    return {
        "enabled": policy.enabled(),
        "passed": not failed_profiles and not failures,
        "policy": policy.to_dict(),
        "policy_source": policy_source,
        "passed_profile_count": len(profiles) - len(failed_profiles),
        "failed_profile_count": len(failed_profiles),
        "failure_counts_by_metric": failure_counts,
        "failures": tuple(failures),
    }


def _leaderboard(profiles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for profile in profiles:
        metrics = _mapping(profile.get("metrics"))
        rows.append({
            "profile": profile.get("profile"),
            "status": profile.get("status"),
            "total_seconds_mean": _float_or_none(metrics.get("total_seconds_mean")),
            "total_seconds_p95": _float_or_none(metrics.get("total_seconds_p95")),
            "measured_phases_mean": _float_or_none(metrics.get("measured_phases_mean")),
            "mean_attempted_route_count": _float_or_none(metrics.get("mean_attempted_route_count")),
            "retrieval_use_rate": _float_or_none(metrics.get("retrieval_use_rate")),
            "cache_hit_rate_mean": _float_or_none(metrics.get("cache_hit_rate_mean")),
            "verification_skip_rate_mean": _float_or_none(metrics.get("verification_skip_rate_mean")),
            "verified_claim_count_mean": _float_or_none(metrics.get("verified_claim_count_mean")),
            "verification_partial_skip_trace_count": _float_or_none(
                metrics.get("verification_partial_skip_trace_count")
            ),
            "verification_selective_claim_skip_rate": _float_or_none(
                metrics.get("verification_selective_claim_skip_rate")
            ),
            "blocked": profile.get("status") == "blocked",
        })
    return sorted(
        rows,
        key=lambda row: (
            bool(row["blocked"]),
            _sort_float(row["total_seconds_mean"]),
            _sort_float(row["measured_phases_mean"]),
            _sort_float(row["verified_claim_count_mean"]),
            _sort_float(row["mean_attempted_route_count"]),
            str(row["profile"]),
        ),
    )


def _sweep_status(profiles: Sequence[Mapping[str, Any]], *, slo: Mapping[str, Any]) -> str:
    if any(profile.get("status") == "blocked" for profile in profiles):
        return "blocked"
    if bool(slo.get("enabled")) and slo.get("passed") is not True:
        return "blocked"
    if any(profile.get("status") == "promote" for profile in profiles):
        return "promote"
    return "observed"


def _blocking_reasons(
    profiles: Sequence[Mapping[str, Any]],
    *,
    slo: Mapping[str, Any],
) -> tuple[str, ...]:
    reasons = []
    for profile in profiles:
        if profile.get("status") != "blocked":
            continue
        budget = _mapping(profile.get("budget"))
        failures = _mapping(budget.get("failure_counts_by_metric"))
        if profile.get("baseline_status") == "blocked":
            if failures:
                for metric, count in sorted(failures.items()):
                    reasons.append(f"{profile.get('profile')}.{metric}: failed {count} trace(s)")
            else:
                reasons.append(f"{profile.get('profile')}: runtime baseline blocked")
        profile_slo = _mapping(profile.get("slo"))
        for failure in _sequence(profile_slo.get("failures")):
            if not isinstance(failure, Mapping):
                continue
            metric = failure.get("metric", "unknown")
            limit_type = failure.get("limit_type", "slo")
            limit = failure.get("limit")
            reasons.append(f"{profile.get('profile')}.{metric}: SLO {limit_type} {limit} failed")
    for failure in _sequence(slo.get("failures")):
        if not isinstance(failure, Mapping):
            continue
        metric = failure.get("metric", "unknown")
        limit_type = failure.get("limit_type", "slo")
        limit = failure.get("limit")
        reasons.append(f"sweep.{metric}: SLO {limit_type} {limit} failed")
    return tuple(reasons)


def _write_artifact_manifest(
    config: ProductRuntimeProfileSweepConfig,
    report: Mapping[str, Any],
    *,
    artifacts: Mapping[str, str | Path | None] | None = None,
) -> dict[str, Any]:
    slo = _mapping(report.get("slo"))
    manifest = build_artifact_manifest(
        _artifact_paths(config, report) if artifacts is None else artifacts,
        root=config.resolved_artifact_manifest_path.parent,
        metadata={
            "runner": "run_product_runtime_profile_sweep",
            "status": report.get("status"),
            "recommended_profile": _nested(report, "decision", "recommended_profile"),
            "profile_count": len(config.profiles),
            "scenario_count": len(config.scenarios),
            "repeats": config.repeats,
            "max_workers": config.max_workers,
            "compact_json": config.compact_json,
            "slo_enabled": slo.get("enabled"),
            "slo_passed": slo.get("passed"),
            "runtime_profile_selector_policy": (
                None
                if config.runtime_profile_selector_policy_path is None
                else str(config.runtime_profile_selector_policy_path)
            ),
            **dict(config.metadata),
        },
    )
    _write_json(config.resolved_artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _artifact_paths(
    config: ProductRuntimeProfileSweepConfig,
    report: Mapping[str, Any],
) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {
        "product_runtime_profile_sweep_report": config.resolved_report_path,
        "policy": config.policy_path,
        "slo_policy": config.slo_policy_path,
        "runtime_profile_selector_policy": config.runtime_profile_selector_policy_path,
        "promotion_contract": config.promotion_contract_path,
    }
    for profile in _sequence(report.get("profiles")):
        if not isinstance(profile, Mapping):
            continue
        profile_name = _safe_artifact_name(str(profile.get("profile", "profile")))
        artifacts[f"{profile_name}_baseline"] = profile.get("baseline_path")
        artifacts[f"{profile_name}_baseline_manifest"] = profile.get("baseline_artifact_manifest")
        for index, trace_path in enumerate(_sequence(profile.get("trace_paths"))):
            artifacts[f"{profile_name}_trace_{index:04d}"] = str(trace_path)
    return artifacts


def _record_registry(config: ProductRuntimeProfileSweepConfig, report: Mapping[str, Any]) -> None:
    if config.registry_path is None:
        return
    slo = _mapping(report.get("slo"))
    ArtifactRegistry.load_json(config.registry_path).record_report(
        name=str(config.name),
        path=config.resolved_report_path,
        version=str(config.version),
        metadata={
            "workflow": "run_product_runtime_profile_sweep",
            "status": report.get("status"),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "recommended_profile": _nested(report, "decision", "recommended_profile"),
            "profile_count": len(config.profiles),
            "scenario_count": len(config.scenarios),
            "repeats": config.repeats,
            "max_workers": config.max_workers,
            "compact_json": config.compact_json,
            "slo_enabled": slo.get("enabled"),
            "slo_passed": slo.get("passed"),
            "runtime_profile_selector_policy": (
                None
                if config.runtime_profile_selector_policy_path is None
                else str(config.runtime_profile_selector_policy_path)
            ),
            **dict(config.metadata),
        },
    ).save_json()


def _load_slo_policy(
    config: ProductRuntimeProfileSweepConfig,
) -> tuple[ProductRuntimeProfileSLOPolicy | None, str | None]:
    if config.slo_policy is not None:
        return (
            config.slo_policy
            if isinstance(config.slo_policy, ProductRuntimeProfileSLOPolicy)
            else ProductRuntimeProfileSLOPolicy.from_mapping(config.slo_policy),
            "inline",
        )
    if config.slo_policy_path is not None:
        payload = json.loads(Path(config.slo_policy_path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"SLO policy JSON must be an object: {config.slo_policy_path}")
        return ProductRuntimeProfileSLOPolicy.from_mapping(payload), str(config.slo_policy_path)
    return None, None


def _trace_path(
    config: ProductRuntimeProfileSweepConfig,
    profile_name: str,
    scenario_name: str,
    repeat_index: int,
) -> Path:
    return config.output_dir / "traces" / profile_name / f"{scenario_name}-r{repeat_index}.json"


def _profile_baseline_path(config: ProductRuntimeProfileSweepConfig, profile_name: str) -> Path:
    return config.output_dir / "baselines" / profile_name / "product-runtime-baseline.json"


def _normalize_profile(profile: str) -> str:
    normalized = str(profile).strip().lower().replace("-", "_")
    if normalized == "auto":
        return normalized
    resolved = get_runtime_profile(profile)
    if resolved is None:
        raise ValueError("runtime profile must not be None.")
    return resolved.name


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _sort_float(value: Any) -> float:
    numeric = _float_or_none(value)
    return float("inf") if numeric is None else numeric


def _max_slo_check(
    values: Mapping[str, Any],
    *,
    metric: str,
    limit: float,
    output_metric: str | None = None,
) -> dict[str, Any]:
    observed = _float_or_none(values.get(metric))
    return {
        "metric": output_metric or metric,
        "limit_type": "max",
        "limit": limit,
        "value": observed,
        "raw_value": None if values.get(metric) is None else repr(values.get(metric)),
        "passed": observed is not None and observed <= limit,
    }


def _min_slo_check(
    values: Mapping[str, Any],
    *,
    metric: str,
    limit: float,
    output_metric: str | None = None,
) -> dict[str, Any]:
    observed = _float_or_none(values.get(metric))
    return {
        "metric": output_metric or metric,
        "limit_type": "min",
        "limit": limit,
        "value": observed,
        "raw_value": None if values.get(metric) is None else repr(values.get(metric)),
        "passed": observed is not None and observed >= limit,
    }


def _slo_failure_from_check(check: Mapping[str, Any]) -> dict[str, Any]:
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


def _required_non_negative_float(value: Any, *, name: str) -> float:
    numeric = _float_or_none(value)
    if numeric is None or numeric < 0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return numeric


def _optional_rate_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    numeric = _required_non_negative_float(value, name=name)
    if numeric > 1:
        raise ValueError(f"{name} must be between 0 and 1.")
    return numeric


def _profile_count_mapping(values: Mapping[str, Any], *, field_name: str) -> dict[str, int]:
    counts = {}
    for raw_name, raw_count in values.items():
        profile_name = _normalize_profile(str(raw_name))
        counts[profile_name] = _required_non_negative_int(
            raw_count,
            name=f"{field_name}.{profile_name}",
        )
    return counts


def _required_non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if numeric < 0 or str(value).strip() not in {str(numeric), f"{numeric}.0"}:
        raise ValueError(f"{name} must be a non-negative integer.")
    return numeric


def _safe_artifact_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return cleaned or "artifact"


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json_text(payload, compact=compact), encoding="utf-8")


def _json_text(payload: Any, *, compact: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _parse_profiles(value: str) -> tuple[str, ...]:
    profiles = tuple(item.strip() for item in value.split(",") if item.strip())
    if not profiles:
        raise ValueError("--profiles must contain at least one profile.")
    return tuple(_normalize_profile(profile) for profile in profiles)


def _parse_policy(path: str | None) -> Path | None:
    return None if path is None else Path(path)


def _parse_metadata(values: Sequence[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--metadata entries must be formatted as key=value.")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--metadata keys must be non-empty.")
        try:
            metadata[key] = json.loads(raw)
        except json.JSONDecodeError:
            metadata[key] = raw
    return metadata


def _config_from_args(args: argparse.Namespace) -> ProductRuntimeProfileSweepConfig:
    return ProductRuntimeProfileSweepConfig(
        output_dir=Path(args.output_dir),
        profiles=_parse_profiles(args.profiles),
        repeats=args.repeats,
        artifact_path=Path(args.artifact) if args.artifact else None,
        promotion_contract_path=Path(args.promotion_contract) if args.promotion_contract else None,
        runtime_profile_selector_policy_path=(
            Path(args.runtime_profile_selector_policy)
            if args.runtime_profile_selector_policy
            else None
        ),
        policy_path=_parse_policy(args.policy),
        slo_policy_path=Path(args.slo_policy) if args.slo_policy else None,
        report_path=Path(args.json) if args.json else None,
        artifact_manifest_path=Path(args.artifact_manifest) if args.artifact_manifest else None,
        registry_path=Path(args.registry) if args.registry else None,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        max_workers=args.max_workers,
        compact_json=bool(args.compact_json),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = run_product_runtime_profile_sweep(_config_from_args(args))
    print(_json_text(report, compact=bool(args.compact_json)), end="")
    if args.fail_on_blocked and report["status"] == "blocked":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sweep calibrated-control runtime profiles")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profiles", default=",".join(PRODUCT_RUNTIME_PROFILE_SWEEP_MODES))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--promotion-contract", default=None)
    parser.add_argument("--runtime-profile-selector-policy", default=None,
                        help="RuntimeProfileSelectorPolicy JSON path for auto profile runs")
    parser.add_argument("--policy", default=None, help="ProductRuntimeBudgetPolicy JSON path for baselines")
    parser.add_argument("--slo-policy", default=None, help="ProductRuntimeProfileSLOPolicy JSON path")
    parser.add_argument("--json", default=None, help="top-level sweep report path")
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--max-workers", type=int, default=1,
                        help="run independent runtime profiles concurrently")
    parser.add_argument("--compact-json", action="store_true",
                        help="write minified trace, baseline, report, and manifest JSON")
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
