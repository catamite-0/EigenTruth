"""Tune request-time runtime-profile selector policies with local trace sweeps."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import planned_artifact_manifest_summary  # noqa: E402
from benchmarks.run_product_runtime_profile_sweep import (  # noqa: E402
    DEFAULT_SCENARIOS,
    ProductRuntimeProfileSweepConfig,
    ProductRuntimeScenario,
    run_product_runtime_profile_sweep,
)
from eigentruth.control import RuntimeProfileSelectorPolicy  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class RuntimeProfileSelectorCandidate:
    """One candidate selector policy for tuning."""

    name: str
    policy: RuntimeProfileSelectorPolicy | Mapping[str, Any]
    source: str | None = None

    def __post_init__(self) -> None:
        name = _safe_artifact_name(self.name)
        if not name:
            raise ValueError("selector candidate name must be non-empty.")
        policy = (
            self.policy
            if isinstance(self.policy, RuntimeProfileSelectorPolicy)
            else RuntimeProfileSelectorPolicy.from_mapping(self.policy)
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "policy", policy)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate description."""
        policy = self.policy
        if not isinstance(policy, RuntimeProfileSelectorPolicy):
            policy = RuntimeProfileSelectorPolicy.from_mapping(policy)
        return {
            "name": self.name,
            "source": self.source,
            "policy": policy.to_dict(),
        }


@dataclass(frozen=True)
class RuntimeProfileSelectorTuningConfig:
    """Configuration for selector policy tuning."""

    output_dir: str | Path
    candidates: Sequence[RuntimeProfileSelectorCandidate | Mapping[str, Any]] = field(
        default_factory=lambda: _default_selector_candidates()
    )
    profiles: Sequence[str] = ("auto",)
    scenarios: Sequence[ProductRuntimeScenario] = DEFAULT_SCENARIOS
    repeats: int = 1
    artifact_path: str | Path | None = None
    promotion_contract_path: str | Path | None = None
    policy_path: str | Path | None = None
    slo_policy_path: str | Path | None = None
    report_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    compact_json: bool = False

    def __post_init__(self) -> None:
        output_dir = Path(self.output_dir)
        candidates = tuple(_candidate_from_value(candidate) for candidate in self.candidates)
        if not candidates:
            raise ValueError("at least one selector candidate is required.")
        names = [candidate.name for candidate in candidates]
        if len(set(names)) != len(names):
            raise ValueError("selector candidate names must be unique.")
        profiles = tuple(str(profile).strip().lower().replace("-", "_") for profile in self.profiles)
        if "auto" not in profiles:
            raise ValueError("selector tuning profiles must include auto.")
        repeats = int(self.repeats)
        if repeats <= 0:
            raise ValueError("repeats must be positive.")
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        object.__setattr__(self, "repeats", repeats)
        if self.artifact_path is not None:
            object.__setattr__(self, "artifact_path", Path(self.artifact_path))
        if self.promotion_contract_path is not None:
            object.__setattr__(self, "promotion_contract_path", Path(self.promotion_contract_path))
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
        object.__setattr__(self, "compact_json", bool(self.compact_json))

    @property
    def resolved_report_path(self) -> Path:
        """Return the tuning report path."""
        if self.report_path is not None:
            return Path(self.report_path)
        return Path(self.output_dir) / "runtime-profile-selector-tuning.json"

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        """Return the tuning artifact manifest path."""
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return Path(self.output_dir) / "artifact-manifest.json"


def run_runtime_profile_selector_tuning(
    config: RuntimeProfileSelectorTuningConfig,
) -> dict[str, Any]:
    """Run selector candidates through product runtime profile sweeps."""
    started_at = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        _run_candidate(config, candidate)
        for candidate in config.candidates
    ]
    leaderboard = _leaderboard(candidates)
    recommendation = leaderboard[0] if leaderboard else None
    status = _tuning_status(candidates)
    report = {
        "schema_version": 1,
        "workflow": "runtime_profile_selector_tuning",
        "status": status,
        "decision": {
            "status": status,
            "recommended_candidate": None if recommendation is None else recommendation["candidate"],
            "recommended_policy_path": None if recommendation is None else recommendation["policy_path"],
            "blocking_reasons": _blocking_reasons(candidates),
        },
        "candidates": candidates,
        "leaderboard": leaderboard,
        "paths": {
            "report": str(config.resolved_report_path),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "output_dir": str(config.output_dir),
            "policy": None if config.policy_path is None else str(config.policy_path),
            "slo_policy": None if config.slo_policy_path is None else str(config.slo_policy_path),
            "promotion_contract": (
                None if config.promotion_contract_path is None else str(config.promotion_contract_path)
            ),
        },
        "config": {
            "candidate_names": tuple(candidate.name for candidate in config.candidates),
            "profiles": tuple(config.profiles),
            "scenario_names": tuple(scenario.name for scenario in config.scenarios),
            "repeats": config.repeats,
            "artifact_path": None if config.artifact_path is None else str(config.artifact_path),
            "compact_json": config.compact_json,
            "metadata": dict(config.metadata),
        },
        "execution": {
            "wall_clock_seconds": time.perf_counter() - started_at,
        },
    }
    _write_report_and_manifest(config, report)
    _record_registry(config, report)
    return report


def _run_candidate(
    config: RuntimeProfileSelectorTuningConfig,
    candidate: RuntimeProfileSelectorCandidate,
) -> dict[str, Any]:
    policy_path = _write_candidate_policy(config, candidate)
    sweep_output_dir = config.output_dir / "candidates" / candidate.name
    sweep = run_product_runtime_profile_sweep(
        ProductRuntimeProfileSweepConfig(
            output_dir=sweep_output_dir,
            profiles=config.profiles,
            scenarios=config.scenarios,
            repeats=config.repeats,
            artifact_path=config.artifact_path,
            promotion_contract_path=config.promotion_contract_path,
            runtime_profile_selector_policy_path=policy_path,
            policy_path=config.policy_path,
            slo_policy_path=config.slo_policy_path,
            compact_json=config.compact_json,
            metadata={
                "source": "run_runtime_profile_selector_tuning",
                "selector_candidate": candidate.name,
                **dict(config.metadata),
            },
        )
    )
    auto_profile = _auto_profile(sweep)
    metrics = _mapping(auto_profile.get("metrics"))
    return {
        "candidate": candidate.name,
        "status": sweep.get("status"),
        "policy_path": str(policy_path),
        "policy": candidate.to_dict()["policy"],
        "source": candidate.source,
        "sweep_report_path": _nested(sweep, "paths", "report"),
        "sweep_artifact_manifest": _nested(sweep, "paths", "artifact_manifest"),
        "decision": _mapping(sweep.get("decision")),
        "slo": _mapping(sweep.get("slo")),
        "runtime_profile_selection": _mapping(auto_profile.get("runtime_profile_selection")),
        "metrics": {
            "total_seconds_mean": metrics.get("total_seconds_mean"),
            "total_seconds_p95": metrics.get("total_seconds_p95"),
            "measured_phases_mean": metrics.get("measured_phases_mean"),
            "mean_attempted_route_count": metrics.get("mean_attempted_route_count"),
            "retrieval_use_rate": metrics.get("retrieval_use_rate"),
            "verification_skip_rate_mean": metrics.get("verification_skip_rate_mean"),
            "verified_claim_count_mean": metrics.get("verified_claim_count_mean"),
        },
    }


def _write_candidate_policy(
    config: RuntimeProfileSelectorTuningConfig,
    candidate: RuntimeProfileSelectorCandidate,
) -> Path:
    policy_dir = config.output_dir / "policies"
    policy_path = policy_dir / f"{candidate.name}.json"
    _write_json(policy_path, candidate.to_dict()["policy"], compact=config.compact_json)
    return policy_path


def _auto_profile(sweep: Mapping[str, Any]) -> dict[str, Any]:
    for profile in _sequence(sweep.get("profiles")):
        if isinstance(profile, Mapping) and profile.get("profile") == "auto":
            return dict(profile)
    raise ValueError("selector tuning sweep did not produce an auto profile row.")


def _leaderboard(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        metrics = _mapping(candidate.get("metrics"))
        rows.append({
            "candidate": candidate.get("candidate"),
            "status": candidate.get("status"),
            "policy_path": candidate.get("policy_path"),
            "total_seconds_mean": _float_or_none(metrics.get("total_seconds_mean")),
            "total_seconds_p95": _float_or_none(metrics.get("total_seconds_p95")),
            "measured_phases_mean": _float_or_none(metrics.get("measured_phases_mean")),
            "verified_claim_count_mean": _float_or_none(metrics.get("verified_claim_count_mean")),
            "mean_attempted_route_count": _float_or_none(metrics.get("mean_attempted_route_count")),
            "retrieval_use_rate": _float_or_none(metrics.get("retrieval_use_rate")),
            "blocked": candidate.get("status") == "blocked",
        })
    return sorted(
        rows,
        key=lambda row: (
            bool(row["blocked"]),
            _sort_float(row["total_seconds_mean"]),
            _sort_float(row["measured_phases_mean"]),
            _sort_float(row["verified_claim_count_mean"]),
            _sort_float(row["mean_attempted_route_count"]),
            str(row["candidate"]),
        ),
    )


def _tuning_status(candidates: Sequence[Mapping[str, Any]]) -> str:
    if any(candidate.get("status") == "promote" for candidate in candidates):
        return "promote"
    if all(candidate.get("status") == "blocked" for candidate in candidates):
        return "blocked"
    return "observed"


def _blocking_reasons(candidates: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    reasons = []
    for candidate in candidates:
        if candidate.get("status") != "blocked":
            continue
        decision = _mapping(candidate.get("decision"))
        for reason in _sequence(decision.get("blocking_reasons")):
            reasons.append(f"{candidate.get('candidate')}: {reason}")
    return tuple(reasons)


def _write_report_and_manifest(
    config: RuntimeProfileSelectorTuningConfig,
    report: dict[str, Any],
) -> dict[str, Any]:
    artifacts = _artifact_paths(config, report)
    report["artifact_manifest_summary"] = planned_artifact_manifest_summary(
        artifacts,
        assume_file_paths=(config.resolved_report_path,),
    )
    _write_json(config.resolved_report_path, report, compact=config.compact_json)
    return _write_artifact_manifest(config, report, artifacts=artifacts)


def _artifact_paths(
    config: RuntimeProfileSelectorTuningConfig,
    report: Mapping[str, Any],
) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {
        "runtime_profile_selector_tuning_report": config.resolved_report_path,
        "policy": config.policy_path,
        "slo_policy": config.slo_policy_path,
        "promotion_contract": config.promotion_contract_path,
    }
    for candidate in _sequence(report.get("candidates")):
        if not isinstance(candidate, Mapping):
            continue
        name = _safe_artifact_name(str(candidate.get("candidate", "candidate")))
        artifacts[f"{name}_selector_policy"] = candidate.get("policy_path")
        artifacts[f"{name}_sweep_report"] = candidate.get("sweep_report_path")
        artifacts[f"{name}_sweep_manifest"] = candidate.get("sweep_artifact_manifest")
    return artifacts


def _write_artifact_manifest(
    config: RuntimeProfileSelectorTuningConfig,
    report: Mapping[str, Any],
    *,
    artifacts: Mapping[str, str | Path | None] | None = None,
) -> dict[str, Any]:
    manifest = build_artifact_manifest(
        _artifact_paths(config, report) if artifacts is None else artifacts,
        root=config.resolved_artifact_manifest_path.parent,
        metadata={
            "runner": "run_runtime_profile_selector_tuning",
            "status": report.get("status"),
            "recommended_candidate": _nested(report, "decision", "recommended_candidate"),
            "recommended_policy_path": _nested(report, "decision", "recommended_policy_path"),
            "candidate_count": len(config.candidates),
            "scenario_count": len(config.scenarios),
            "repeats": config.repeats,
            "compact_json": config.compact_json,
            **dict(config.metadata),
        },
    )
    _write_json(config.resolved_artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _record_registry(config: RuntimeProfileSelectorTuningConfig, report: Mapping[str, Any]) -> None:
    if config.registry_path is None:
        return
    ArtifactRegistry.load_json(config.registry_path).record_report(
        name=str(config.name),
        path=config.resolved_report_path,
        version=str(config.version),
        metadata={
            "workflow": "run_runtime_profile_selector_tuning",
            "status": report.get("status"),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "recommended_candidate": _nested(report, "decision", "recommended_candidate"),
            "recommended_policy_path": _nested(report, "decision", "recommended_policy_path"),
            "candidate_count": len(config.candidates),
            "scenario_count": len(config.scenarios),
            "repeats": config.repeats,
            "compact_json": config.compact_json,
            **dict(config.metadata),
        },
    ).save_json()


def _candidate_from_value(
    value: RuntimeProfileSelectorCandidate | Mapping[str, Any],
) -> RuntimeProfileSelectorCandidate:
    if isinstance(value, RuntimeProfileSelectorCandidate):
        return value
    payload = dict(value)
    return RuntimeProfileSelectorCandidate(
        name=str(payload["name"]),
        policy=_mapping(payload.get("policy")),
        source=None if payload.get("source") is None else str(payload.get("source")),
    )


def _load_candidate(value: str) -> RuntimeProfileSelectorCandidate:
    if "=" in value:
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
    else:
        path = Path(value)
        name = path.stem
    if not str(name).strip():
        raise ValueError("--candidate name must be non-empty.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"selector candidate JSON must be an object: {path}")
    return RuntimeProfileSelectorCandidate(name=name, policy=payload, source=str(path))


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


def _default_selector_candidates() -> tuple[RuntimeProfileSelectorCandidate, ...]:
    return (
        RuntimeProfileSelectorCandidate(
            name="default",
            policy=RuntimeProfileSelectorPolicy(),
        ),
        RuntimeProfileSelectorCandidate(
            name="latency_biased",
            policy=RuntimeProfileSelectorPolicy(
                sensitive_claim_feature_flags=("has_citation", "is_time_sensitive"),
                sensitive_claim_metadata_keys=("requires_verification",),
            ),
        ),
        RuntimeProfileSelectorCandidate(
            name="audit_biased",
            policy=RuntimeProfileSelectorPolicy(
                high_risk_actions=(
                    "retrieve",
                    "abstain",
                    "clarify",
                    "rewrite",
                    "steer_regenerate",
                    "execute_tool",
                ),
            ),
        ),
    )


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
    return profiles


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


def _config_from_args(args: argparse.Namespace) -> RuntimeProfileSelectorTuningConfig:
    candidates = (
        _default_selector_candidates()
        if not args.candidate
        else tuple(_load_candidate(value) for value in args.candidate)
    )
    return RuntimeProfileSelectorTuningConfig(
        output_dir=Path(args.output_dir),
        candidates=candidates,
        profiles=_parse_profiles(args.profiles),
        repeats=args.repeats,
        artifact_path=Path(args.artifact) if args.artifact else None,
        promotion_contract_path=Path(args.promotion_contract) if args.promotion_contract else None,
        policy_path=Path(args.policy) if args.policy else None,
        slo_policy_path=Path(args.slo_policy) if args.slo_policy else None,
        report_path=Path(args.json) if args.json else None,
        artifact_manifest_path=Path(args.artifact_manifest) if args.artifact_manifest else None,
        registry_path=Path(args.registry) if args.registry else None,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = run_runtime_profile_selector_tuning(_config_from_args(args))
    print(_json_text(report, compact=bool(args.compact_json)), end="")
    if args.fail_on_blocked and report["status"] == "blocked":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tune auto runtime-profile selector policies")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", action="append", default=[],
                        help="candidate selector policy JSON path, or name=path; repeatable")
    parser.add_argument("--profiles", default="auto")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--promotion-contract", default=None)
    parser.add_argument("--policy", default=None, help="ProductRuntimeBudgetPolicy JSON path for child sweeps")
    parser.add_argument("--slo-policy", default=None, help="ProductRuntimeProfileSLOPolicy JSON path")
    parser.add_argument("--json", default=None, help="top-level tuning report path")
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
