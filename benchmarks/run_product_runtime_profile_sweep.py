"""Sweep product runtime profiles with deterministic demo ProductTrace runs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    profiles: Sequence[str] = RUNTIME_PROFILE_NAMES
    scenarios: Sequence[ProductRuntimeScenario] = DEFAULT_SCENARIOS
    repeats: int = 1
    artifact_path: str | Path | None = None
    promotion_contract_path: str | Path | None = None
    policy: ProductRuntimeBudgetPolicy | Mapping[str, Any] | None = None
    policy_path: str | Path | None = None
    report_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        output_dir = Path(self.output_dir)
        profiles = tuple(_normalize_profile(profile) for profile in self.profiles)
        if not profiles:
            raise ValueError("at least one runtime profile is required.")
        scenarios = tuple(self.scenarios)
        if not scenarios:
            raise ValueError("at least one runtime scenario is required.")
        repeats = int(self.repeats)
        if repeats <= 0:
            raise ValueError("repeats must be positive.")
        if self.policy is not None and (self.policy_path is not None or self.promotion_contract_path is not None):
            raise ValueError("policy object is mutually exclusive with policy_path and promotion_contract_path.")
        if self.policy_path is not None and self.promotion_contract_path is not None:
            raise ValueError("policy_path and promotion_contract_path are mutually exclusive for baseline gating.")
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "repeats", repeats)
        if self.artifact_path is not None:
            object.__setattr__(self, "artifact_path", Path(self.artifact_path))
        if self.promotion_contract_path is not None:
            object.__setattr__(self, "promotion_contract_path", Path(self.promotion_contract_path))
        if self.policy_path is not None:
            object.__setattr__(self, "policy_path", Path(self.policy_path))
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


def run_product_runtime_profile_sweep(config: ProductRuntimeProfileSweepConfig) -> dict[str, Any]:
    """Run deterministic demo traces for each runtime profile and compare baselines."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = demo.load_artifact(None if config.artifact_path is None else str(config.artifact_path))
    profiles = []
    for profile_name in config.profiles:
        traces = _run_profile_traces(config, profile_name, artifact=artifact)
        baseline = build_product_runtime_baseline(
            ProductRuntimeBaselineConfig(
                trace_paths=tuple(trace["path"] for trace in traces),
                report_path=_profile_baseline_path(config, profile_name),
                policy=config.policy,
                policy_path=config.policy_path,
                promotion_contract_path=config.promotion_contract_path,
                metadata={
                    "source": "run_product_runtime_profile_sweep",
                    "runtime_profile": profile_name,
                    **dict(config.metadata),
                },
            )
        )
        profiles.append(_profile_record(profile_name, traces=traces, baseline=baseline))

    leaderboard = _leaderboard(profiles)
    recommendation = leaderboard[0] if leaderboard else None
    status = _sweep_status(profiles)
    report = {
        "schema_version": 1,
        "workflow": "product_runtime_profile_sweep",
        "status": status,
        "decision": {
            "status": status,
            "recommended_profile": None if recommendation is None else recommendation["profile"],
            "blocking_reasons": _blocking_reasons(profiles),
        },
        "profiles": profiles,
        "leaderboard": leaderboard,
        "paths": {
            "report": str(config.resolved_report_path),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "output_dir": str(config.output_dir),
            "policy": None if config.policy_path is None else str(config.policy_path),
            "promotion_contract": (
                None if config.promotion_contract_path is None else str(config.promotion_contract_path)
            ),
        },
        "config": {
            "profiles": tuple(config.profiles),
            "scenario_names": tuple(scenario.name for scenario in config.scenarios),
            "repeats": config.repeats,
            "artifact_path": None if config.artifact_path is None else str(config.artifact_path),
            "metadata": dict(config.metadata),
        },
    }
    _write_json(config.resolved_report_path, report)
    manifest = _write_artifact_manifest(config, report)
    report["artifact_manifest_summary"] = manifest["summary"]
    _write_json(config.resolved_report_path, report)
    _record_registry(config, report)
    return report


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
        request_id=request_id,
        output=str(output_path),
        registry=None,
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
) -> dict[str, Any]:
    summary = _mapping(baseline.get("summary"))
    routes = _mapping(_mapping(summary.get("routes")).get("overall"))
    return {
        "profile": profile_name,
        "status": baseline.get("status"),
        "trace_count": len(traces),
        "baseline_path": _nested(baseline, "paths", "report"),
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
        },
        "budget": _mapping(baseline.get("budget")),
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
            "blocked": profile.get("status") == "blocked",
        })
    return sorted(
        rows,
        key=lambda row: (
            bool(row["blocked"]),
            _sort_float(row["total_seconds_mean"]),
            _sort_float(row["measured_phases_mean"]),
            _sort_float(row["mean_attempted_route_count"]),
            str(row["profile"]),
        ),
    )


def _sweep_status(profiles: Sequence[Mapping[str, Any]]) -> str:
    if any(profile.get("status") == "blocked" for profile in profiles):
        return "blocked"
    if any(profile.get("status") == "promote" for profile in profiles):
        return "promote"
    return "observed"


def _blocking_reasons(profiles: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    reasons = []
    for profile in profiles:
        if profile.get("status") != "blocked":
            continue
        budget = _mapping(profile.get("budget"))
        failures = _mapping(budget.get("failure_counts_by_metric"))
        if failures:
            for metric, count in sorted(failures.items()):
                reasons.append(f"{profile.get('profile')}.{metric}: failed {count} trace(s)")
        else:
            reasons.append(f"{profile.get('profile')}: runtime baseline blocked")
    return tuple(reasons)


def _write_artifact_manifest(
    config: ProductRuntimeProfileSweepConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "product_runtime_profile_sweep_report": config.resolved_report_path,
        "policy": config.policy_path,
        "promotion_contract": config.promotion_contract_path,
    }
    for profile in _sequence(report.get("profiles")):
        if not isinstance(profile, Mapping):
            continue
        profile_name = _safe_artifact_name(str(profile.get("profile", "profile")))
        artifacts[f"{profile_name}_baseline"] = profile.get("baseline_path")
        for index, trace_path in enumerate(_sequence(profile.get("trace_paths"))):
            artifacts[f"{profile_name}_trace_{index:04d}"] = str(trace_path)
    manifest = build_artifact_manifest(
        artifacts,
        root=config.resolved_artifact_manifest_path.parent,
        metadata={
            "runner": "run_product_runtime_profile_sweep",
            "status": report.get("status"),
            "recommended_profile": _nested(report, "decision", "recommended_profile"),
            "profile_count": len(config.profiles),
            "scenario_count": len(config.scenarios),
            "repeats": config.repeats,
            **dict(config.metadata),
        },
    )
    _write_json(config.resolved_artifact_manifest_path, manifest)
    return manifest


def _record_registry(config: ProductRuntimeProfileSweepConfig, report: Mapping[str, Any]) -> None:
    if config.registry_path is None:
        return
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
            **dict(config.metadata),
        },
    ).save_json()


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
        return float(value)
    except (TypeError, ValueError):
        return None


def _sort_float(value: Any) -> float:
    numeric = _float_or_none(value)
    return float("inf") if numeric is None else numeric


def _safe_artifact_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return cleaned or "artifact"


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        policy_path=_parse_policy(args.policy),
        report_path=Path(args.json) if args.json else None,
        artifact_manifest_path=Path(args.artifact_manifest) if args.artifact_manifest else None,
        registry_path=Path(args.registry) if args.registry else None,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = run_product_runtime_profile_sweep(_config_from_args(args))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_blocked and report["status"] == "blocked":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sweep calibrated-control runtime profiles")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profiles", default=",".join(RUNTIME_PROFILE_NAMES))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--promotion-contract", default=None)
    parser.add_argument("--policy", default=None, help="ProductRuntimeBudgetPolicy JSON path for baselines")
    parser.add_argument("--json", default=None, help="top-level sweep report path")
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
