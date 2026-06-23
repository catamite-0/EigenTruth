"""Replay runtime-profile selector policies over saved ProductTrace JSON."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import strict_bool  # noqa: E402
from benchmarks.run_runtime_profile_selector_tuning import (  # noqa: E402
    RuntimeProfileSelectorCandidate,
)
from eigentruth.control import RUNTIME_PROFILE_NAMES, select_runtime_profile  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

DEFAULT_PROFILE_COST_UNITS: Mapping[str, float] = {
    "latency": 1.0,
    "balanced": 2.0,
    "audit": 3.0,
}


@dataclass(frozen=True)
class RuntimeProfileSelectorReplayPolicy:
    """Optional gates for selector replay reports."""

    max_estimated_cost_units_mean: float | None = None
    max_observed_selected_total_seconds_mean: float | None = None
    max_observed_selected_total_seconds_p95: float | None = None
    min_observed_runtime_coverage_rate: float | None = None
    min_selected_profile_counts: Mapping[str, int] = field(default_factory=dict)
    max_selected_profile_rates: Mapping[str, float] = field(default_factory=dict)
    min_selected_profile_rates: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_estimated_cost_units_mean",
            _optional_non_negative_float(
                self.max_estimated_cost_units_mean,
                name="max_estimated_cost_units_mean",
            ),
        )
        object.__setattr__(
            self,
            "max_observed_selected_total_seconds_mean",
            _optional_non_negative_float(
                self.max_observed_selected_total_seconds_mean,
                name="max_observed_selected_total_seconds_mean",
            ),
        )
        object.__setattr__(
            self,
            "max_observed_selected_total_seconds_p95",
            _optional_non_negative_float(
                self.max_observed_selected_total_seconds_p95,
                name="max_observed_selected_total_seconds_p95",
            ),
        )
        object.__setattr__(
            self,
            "min_observed_runtime_coverage_rate",
            _optional_rate_float(
                self.min_observed_runtime_coverage_rate,
                name="min_observed_runtime_coverage_rate",
            ),
        )
        object.__setattr__(
            self,
            "min_selected_profile_counts",
            _profile_count_mapping(
                self.min_selected_profile_counts,
                field_name="min_selected_profile_counts",
            ),
        )
        object.__setattr__(
            self,
            "max_selected_profile_rates",
            _profile_rate_mapping(
                self.max_selected_profile_rates,
                field_name="max_selected_profile_rates",
            ),
        )
        object.__setattr__(
            self,
            "min_selected_profile_rates",
            _profile_rate_mapping(
                self.min_selected_profile_rates,
                field_name="min_selected_profile_rates",
            ),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RuntimeProfileSelectorReplayPolicy":
        """Build a replay policy from a JSON-like mapping."""
        return cls(
            max_estimated_cost_units_mean=payload.get("max_estimated_cost_units_mean"),
            max_observed_selected_total_seconds_mean=payload.get(
                "max_observed_selected_total_seconds_mean"
            ),
            max_observed_selected_total_seconds_p95=payload.get(
                "max_observed_selected_total_seconds_p95"
            ),
            min_observed_runtime_coverage_rate=payload.get("min_observed_runtime_coverage_rate"),
            min_selected_profile_counts=dict(_mapping(payload.get("min_selected_profile_counts"))),
            max_selected_profile_rates=dict(_mapping(payload.get("max_selected_profile_rates"))),
            min_selected_profile_rates=dict(_mapping(payload.get("min_selected_profile_rates"))),
        )

    def enabled(self) -> bool:
        """Return whether any replay gate is active."""
        return (
            self.max_estimated_cost_units_mean is not None
            or self.max_observed_selected_total_seconds_mean is not None
            or self.max_observed_selected_total_seconds_p95 is not None
            or self.min_observed_runtime_coverage_rate is not None
            or bool(self.min_selected_profile_counts)
            or bool(self.max_selected_profile_rates)
            or bool(self.min_selected_profile_rates)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "max_estimated_cost_units_mean": self.max_estimated_cost_units_mean,
            "max_observed_selected_total_seconds_mean": self.max_observed_selected_total_seconds_mean,
            "max_observed_selected_total_seconds_p95": self.max_observed_selected_total_seconds_p95,
            "min_observed_runtime_coverage_rate": self.min_observed_runtime_coverage_rate,
            "min_selected_profile_counts": dict(self.min_selected_profile_counts),
            "max_selected_profile_rates": dict(self.max_selected_profile_rates),
            "min_selected_profile_rates": dict(self.min_selected_profile_rates),
        }


@dataclass(frozen=True)
class RuntimeProfileSelectorReplayConfig:
    """Configuration for replaying selector policies over saved traces."""

    trace_paths: Sequence[str | Path]
    output_dir: str | Path
    candidates: Sequence[RuntimeProfileSelectorCandidate | Mapping[str, Any]]
    replay_policy: RuntimeProfileSelectorReplayPolicy | Mapping[str, Any] | None = None
    replay_policy_path: str | Path | None = None
    profile_cost_units: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_PROFILE_COST_UNITS))
    report_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    compact_json: bool = False

    def __post_init__(self) -> None:
        trace_paths = tuple(Path(path) for path in self.trace_paths)
        if not trace_paths:
            raise ValueError("at least one ProductTrace path is required.")
        candidates = tuple(_candidate_from_value(candidate) for candidate in self.candidates)
        if not candidates:
            raise ValueError("at least one selector candidate is required.")
        names = [candidate.name for candidate in candidates]
        if len(set(names)) != len(names):
            raise ValueError("selector candidate names must be unique.")
        if self.replay_policy is not None and self.replay_policy_path is not None:
            raise ValueError("replay_policy object is mutually exclusive with replay_policy_path.")
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        object.__setattr__(self, "trace_paths", trace_paths)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "profile_cost_units", _profile_cost_mapping(self.profile_cost_units))
        if self.replay_policy_path is not None:
            object.__setattr__(self, "replay_policy_path", Path(self.replay_policy_path))
        if self.report_path is not None:
            object.__setattr__(self, "report_path", Path(self.report_path))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))

    @property
    def resolved_report_path(self) -> Path:
        """Return the replay report path."""
        if self.report_path is not None:
            return Path(self.report_path)
        return Path(self.output_dir) / "runtime-profile-selector-replay.json"

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        """Return the replay artifact manifest path."""
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return Path(self.output_dir) / "artifact-manifest.json"


def run_runtime_profile_selector_replay(
    config: RuntimeProfileSelectorReplayConfig,
) -> dict[str, Any]:
    """Replay selector candidates over saved ProductTrace JSON payloads."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    replay_policy, replay_policy_source = _load_replay_policy(config)
    traces = tuple((path, _load_trace(path)) for path in config.trace_paths)
    runtime_pair_index = _runtime_pair_index(traces)
    candidates = [
        _candidate_record(
            config,
            candidate,
            traces=traces,
            replay_policy=replay_policy,
            runtime_pair_index=runtime_pair_index,
        )
        for candidate in config.candidates
    ]
    leaderboard = _leaderboard(candidates)
    recommendation = leaderboard[0] if leaderboard else None
    status = _replay_status(candidates)
    report = {
        "schema_version": 1,
        "workflow": "runtime_profile_selector_replay",
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
            "replay_policy": None if config.replay_policy_path is None else str(config.replay_policy_path),
            "traces": [str(path) for path, _trace in traces],
        },
        "config": {
            "candidate_names": tuple(candidate.name for candidate in config.candidates),
            "trace_count": len(traces),
            "replay_policy_source": replay_policy_source,
            "profile_cost_units": dict(config.profile_cost_units),
            "runtime_pairing": {
                "enabled": True,
                "indexed_pairs": len(runtime_pair_index),
            },
            "compact_json": config.compact_json,
            "metadata": dict(config.metadata),
        },
    }
    _write_report_and_manifest(config, report)
    _record_registry(config, report)
    return report


def _candidate_record(
    config: RuntimeProfileSelectorReplayConfig,
    candidate: RuntimeProfileSelectorCandidate,
    *,
    traces: Sequence[tuple[Path, Mapping[str, Any]]],
    replay_policy: RuntimeProfileSelectorReplayPolicy | None,
    runtime_pair_index: Mapping[tuple[str, str], Sequence[tuple[Path, Mapping[str, Any]]]],
) -> dict[str, Any]:
    policy_path = _write_candidate_policy(config, candidate)
    trace_records = [
        _trace_selection_record(
            path,
            trace,
            candidate=candidate,
            cost_units=config.profile_cost_units,
            runtime_pair_index=runtime_pair_index,
        )
        for path, trace in traces
    ]
    summary = _selection_summary(trace_records, cost_units=config.profile_cost_units)
    gate = _evaluate_replay_policy(summary, replay_policy)
    status = _candidate_status(gate)
    return {
        "candidate": candidate.name,
        "status": status,
        "policy_path": str(policy_path),
        "policy": candidate.to_dict()["policy"],
        "source": candidate.source,
        "summary": summary,
        "gate": gate,
        "traces": trace_records,
    }


def _trace_selection_record(
    path: Path,
    trace: Mapping[str, Any],
    *,
    candidate: RuntimeProfileSelectorCandidate,
    cost_units: Mapping[str, float],
    runtime_pair_index: Mapping[tuple[str, str], Sequence[tuple[Path, Mapping[str, Any]]]],
) -> dict[str, Any]:
    risk_decision = trace.get("risk_decision")
    if not isinstance(risk_decision, Mapping):
        raise ValueError(f"ProductTrace is missing risk_decision object: {path}")
    claims = tuple(_sequence(trace.get("claims")))
    selection = select_runtime_profile(
        risk_decision,
        claims=claims,
        selector_policy=candidate.policy,
    )
    original_profile = _nested(trace, "metadata", "runtime_profile")
    selected = selection.selected_profile
    request_key = _trace_request_key(path, trace)
    original_total_seconds = _runtime_total_seconds(trace)
    paired_traces = tuple(runtime_pair_index.get((request_key, selected), ()))
    paired_totals = [
        seconds
        for _paired_path, paired_payload in paired_traces
        if (seconds := _runtime_total_seconds(paired_payload)) is not None
    ]
    paired_stats = _runtime_seconds_stats(paired_totals)
    observed_selected_total_seconds = paired_stats["mean_seconds"]
    return {
        "path": str(path),
        "request_id": trace.get("request_id"),
        "request_key": request_key,
        "original_runtime_profile": None if original_profile is None else str(original_profile),
        "selected_runtime_profile": selected,
        "changed": original_profile is not None and str(original_profile) != selected,
        "estimated_cost_units": cost_units.get(selected),
        "observed_original_total_seconds": original_total_seconds,
        "observed_selected_total_seconds": observed_selected_total_seconds,
        "observed_selected_trace_path": None if not paired_traces else str(paired_traces[0][0]),
        "observed_selected_trace_paths": tuple(str(paired_path) for paired_path, _payload in paired_traces),
        "observed_selected_pair_count": len(paired_totals),
        "observed_selected_pair_stats": paired_stats,
        "observed_runtime_paired": observed_selected_total_seconds is not None,
        "risk_level": risk_decision.get("risk_level"),
        "action": risk_decision.get("action"),
        "claim_count": len(claims),
        "selection": selection.to_dict(),
    }


def _selection_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    cost_units: Mapping[str, float],
) -> dict[str, Any]:
    selected_counts: dict[str, int] = {}
    original_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    switch_counts: dict[str, int] = {}
    costs = []
    observed_original_seconds = []
    observed_selected_seconds = []
    observed_selected_by_profile: dict[str, list[float]] = {}
    changed = 0
    paired = 0
    for record in records:
        selected = str(record.get("selected_runtime_profile"))
        selected_counts[selected] = selected_counts.get(selected, 0) + 1
        original = record.get("original_runtime_profile")
        if original is not None:
            original_key = str(original)
            original_counts[original_key] = original_counts.get(original_key, 0) + 1
            switch_key = f"{original_key}->{selected}"
            switch_counts[switch_key] = switch_counts.get(switch_key, 0) + 1
        selection = _mapping(record.get("selection"))
        reason = selection.get("reason")
        if reason is not None:
            reason_key = str(reason)
            reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
        if bool(record.get("changed")):
            changed += 1
        cost = _float_or_none(record.get("estimated_cost_units"))
        if cost is not None:
            costs.append(cost)
        original_seconds = _float_or_none(record.get("observed_original_total_seconds"))
        if original_seconds is not None:
            observed_original_seconds.append(original_seconds)
        selected_seconds = _float_or_none(record.get("observed_selected_total_seconds"))
        if selected_seconds is not None:
            paired += 1
            observed_selected_seconds.append(selected_seconds)
            observed_selected_by_profile.setdefault(selected, []).append(selected_seconds)
    total = len(records)
    selected_runtime_stats = _runtime_seconds_stats(observed_selected_seconds)
    original_runtime_stats = _runtime_seconds_stats(observed_original_seconds)
    return {
        "trace_count": total,
        "selected_counts": selected_counts,
        "selected_rates": {
            profile: _safe_div(count, total)
            for profile, count in selected_counts.items()
        },
        "original_counts": original_counts,
        "switch_counts": switch_counts,
        "changed_count": changed,
        "changed_rate": _safe_div(changed, total),
        "reason_counts": reason_counts,
        "estimated_cost_units_total": None if not costs else sum(costs),
        "estimated_cost_units_mean": None if not costs else sum(costs) / len(costs),
        "profile_cost_units": dict(cost_units),
        "observed_runtime": {
            "paired_count": paired,
            "coverage_rate": _safe_div(paired, total),
            "selected_total_seconds": selected_runtime_stats,
            "original_total_seconds": original_runtime_stats,
            "selected_total_seconds_by_profile": {
                profile: _runtime_seconds_stats(values)
                for profile, values in sorted(observed_selected_by_profile.items())
            },
        },
        "observed_runtime_paired_count": paired,
        "observed_runtime_coverage_rate": _safe_div(paired, total),
        "observed_selected_total_seconds_mean": selected_runtime_stats["mean_seconds"],
        "observed_selected_total_seconds_p95": selected_runtime_stats["p95_seconds"],
        "observed_selected_total_seconds_p99": selected_runtime_stats["p99_seconds"],
        "observed_selected_total_seconds_max": selected_runtime_stats["max_seconds"],
        "observed_original_total_seconds_mean": original_runtime_stats["mean_seconds"],
    }


def _evaluate_replay_policy(
    summary: Mapping[str, Any],
    policy: RuntimeProfileSelectorReplayPolicy | None,
) -> dict[str, Any]:
    if policy is None or not policy.enabled():
        return {
            "enabled": False,
            "passed": None,
            "policy": None if policy is None else policy.to_dict(),
            "checks": (),
            "failures": (),
        }
    checks = []
    failures = []
    if policy.max_estimated_cost_units_mean is not None:
        check = _max_check(
            summary,
            metric="estimated_cost_units_mean",
            limit=policy.max_estimated_cost_units_mean,
        )
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))
    if policy.max_observed_selected_total_seconds_mean is not None:
        check = _max_check(
            summary,
            metric="observed_selected_total_seconds_mean",
            limit=policy.max_observed_selected_total_seconds_mean,
        )
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))
    if policy.max_observed_selected_total_seconds_p95 is not None:
        check = _max_check(
            summary,
            metric="observed_selected_total_seconds_p95",
            limit=policy.max_observed_selected_total_seconds_p95,
        )
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))
    if policy.min_observed_runtime_coverage_rate is not None:
        check = _min_check(
            summary,
            metric="observed_runtime_coverage_rate",
            limit=policy.min_observed_runtime_coverage_rate,
        )
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))
    selected_counts = _mapping(summary.get("selected_counts"))
    selected_rates = _mapping(summary.get("selected_rates"))
    for profile, limit in policy.min_selected_profile_counts.items():
        check = _min_check(
            selected_counts,
            metric=profile,
            output_metric=f"selected_count.{profile}",
            limit=float(limit),
        )
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))
    for profile, limit in policy.max_selected_profile_rates.items():
        check = _max_check(
            selected_rates,
            metric=profile,
            output_metric=f"selected_rate.{profile}",
            limit=limit,
        )
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))
    for profile, limit in policy.min_selected_profile_rates.items():
        check = _min_check(
            selected_rates,
            metric=profile,
            output_metric=f"selected_rate.{profile}",
            limit=limit,
        )
        checks.append(check)
        if not check["passed"]:
            failures.append(_failure_from_check(check))
    return {
        "enabled": True,
        "passed": not failures,
        "policy": policy.to_dict(),
        "checks": tuple(checks),
        "failures": tuple(failures),
    }


def _candidate_status(gate: Mapping[str, Any]) -> str:
    if bool(gate.get("enabled")):
        return "promote" if gate.get("passed") is True else "blocked"
    return "observed"


def _leaderboard(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        summary = _mapping(candidate.get("summary"))
        selected_rates = _mapping(summary.get("selected_rates"))
        rows.append({
            "candidate": candidate.get("candidate"),
            "status": candidate.get("status"),
            "policy_path": candidate.get("policy_path"),
            "estimated_cost_units_mean": _float_or_none(summary.get("estimated_cost_units_mean")),
            "observed_runtime_coverage_rate": _float_or_none(
                summary.get("observed_runtime_coverage_rate")
            ),
            "observed_selected_total_seconds_mean": _float_or_none(
                summary.get("observed_selected_total_seconds_mean")
            ),
            "observed_selected_total_seconds_p95": _float_or_none(
                summary.get("observed_selected_total_seconds_p95")
            ),
            "audit_rate": _float_or_none(selected_rates.get("audit")),
            "balanced_rate": _float_or_none(selected_rates.get("balanced")),
            "latency_rate": _float_or_none(selected_rates.get("latency")),
            "changed_rate": _float_or_none(summary.get("changed_rate")),
            "blocked": candidate.get("status") == "blocked",
        })
    return sorted(
        rows,
        key=lambda row: (
            bool(row["blocked"]),
            _sort_float(row["observed_selected_total_seconds_mean"]),
            _sort_float(row["estimated_cost_units_mean"]),
            _sort_float(row["audit_rate"]),
            str(row["candidate"]),
        ),
    )


def _replay_status(candidates: Sequence[Mapping[str, Any]]) -> str:
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
        gate = _mapping(candidate.get("gate"))
        for failure in _sequence(gate.get("failures")):
            if not isinstance(failure, Mapping):
                continue
            reasons.append(
                f"{candidate.get('candidate')}.{failure.get('metric')}: "
                f"{failure.get('reason')}"
            )
    return tuple(reasons)


def _recommended_leaderboard_row(report: Mapping[str, Any]) -> dict[str, Any]:
    recommended_candidate = _nested(report, "decision", "recommended_candidate")
    for row in _sequence(report.get("leaderboard")):
        if isinstance(row, Mapping) and row.get("candidate") == recommended_candidate:
            return dict(row)
    return {}


def _load_replay_policy(
    config: RuntimeProfileSelectorReplayConfig,
) -> tuple[RuntimeProfileSelectorReplayPolicy | None, str | None]:
    if config.replay_policy is not None:
        return (
            config.replay_policy
            if isinstance(config.replay_policy, RuntimeProfileSelectorReplayPolicy)
            else RuntimeProfileSelectorReplayPolicy.from_mapping(config.replay_policy),
            "inline",
        )
    if config.replay_policy_path is not None:
        payload = json.loads(Path(config.replay_policy_path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"replay policy JSON must be an object: {config.replay_policy_path}")
        return RuntimeProfileSelectorReplayPolicy.from_mapping(payload), str(config.replay_policy_path)
    return None, None


def _write_candidate_policy(
    config: RuntimeProfileSelectorReplayConfig,
    candidate: RuntimeProfileSelectorCandidate,
) -> Path:
    policy_dir = Path(config.output_dir) / "policies"
    policy_path = policy_dir / f"{candidate.name}.json"
    _write_json(policy_path, candidate.to_dict()["policy"], compact=config.compact_json)
    return policy_path


def _write_report_and_manifest(
    config: RuntimeProfileSelectorReplayConfig,
    report: dict[str, Any],
) -> dict[str, Any]:
    _write_json(config.resolved_report_path, report, compact=config.compact_json)
    manifest = _write_artifact_manifest(config, report)
    report["artifact_manifest_summary"] = manifest["summary"]
    _write_json(config.resolved_report_path, report, compact=config.compact_json)
    return _write_artifact_manifest(config, report)


def _write_artifact_manifest(
    config: RuntimeProfileSelectorReplayConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    recommended = _recommended_leaderboard_row(report)
    artifacts: dict[str, str | Path | None] = {
        "runtime_profile_selector_replay_report": config.resolved_report_path,
        "replay_policy": config.replay_policy_path,
    }
    for index, trace_path in enumerate(config.trace_paths):
        artifacts[f"trace_{index:04d}_{_safe_artifact_name(trace_path.stem)}"] = trace_path
    for candidate in _sequence(report.get("candidates")):
        if not isinstance(candidate, Mapping):
            continue
        name = _safe_artifact_name(str(candidate.get("candidate", "candidate")))
        artifacts[f"{name}_selector_policy"] = candidate.get("policy_path")
    manifest = build_artifact_manifest(
        artifacts,
        root=config.resolved_artifact_manifest_path.parent,
        metadata={
            "runner": "run_runtime_profile_selector_replay",
            "status": report.get("status"),
            "recommended_candidate": _nested(report, "decision", "recommended_candidate"),
            "recommended_policy_path": _nested(report, "decision", "recommended_policy_path"),
            "recommended_observed_selected_total_seconds_mean": recommended.get(
                "observed_selected_total_seconds_mean"
            ),
            "recommended_observed_selected_total_seconds_p95": recommended.get(
                "observed_selected_total_seconds_p95"
            ),
            "recommended_observed_runtime_coverage_rate": recommended.get(
                "observed_runtime_coverage_rate"
            ),
            "candidate_count": len(config.candidates),
            "trace_count": len(config.trace_paths),
            "compact_json": config.compact_json,
            **dict(config.metadata),
        },
    )
    _write_json(config.resolved_artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _record_registry(config: RuntimeProfileSelectorReplayConfig, report: Mapping[str, Any]) -> None:
    if config.registry_path is None:
        return
    recommended = _recommended_leaderboard_row(report)
    ArtifactRegistry.load_json(config.registry_path).record_report(
        name=str(config.name),
        path=config.resolved_report_path,
        version=str(config.version),
        metadata={
            "workflow": "run_runtime_profile_selector_replay",
            "status": report.get("status"),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "recommended_candidate": _nested(report, "decision", "recommended_candidate"),
            "recommended_policy_path": _nested(report, "decision", "recommended_policy_path"),
            "recommended_observed_selected_total_seconds_mean": recommended.get(
                "observed_selected_total_seconds_mean"
            ),
            "recommended_observed_selected_total_seconds_p95": recommended.get(
                "observed_selected_total_seconds_p95"
            ),
            "recommended_observed_runtime_coverage_rate": recommended.get(
                "observed_runtime_coverage_rate"
            ),
            "candidate_count": len(config.candidates),
            "trace_count": len(config.trace_paths),
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"selector candidate JSON must be an object: {path}")
    return RuntimeProfileSelectorCandidate(name=name, policy=payload, source=str(path))


def _load_trace(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"ProductTrace JSON must be an object: {path}")
    if not isinstance(payload.get("risk_decision"), Mapping):
        raise ValueError(f"ProductTrace JSON is missing risk_decision: {path}")
    return dict(payload)


def _runtime_pair_index(
    traces: Sequence[tuple[Path, Mapping[str, Any]]],
) -> dict[tuple[str, str], tuple[tuple[Path, Mapping[str, Any]], ...]]:
    grouped: dict[tuple[str, str], list[tuple[Path, Mapping[str, Any]]]] = {}
    for path, trace in traces:
        profile = _trace_runtime_profile(path, trace)
        if profile is None:
            continue
        key = (_trace_request_key(path, trace), profile)
        grouped.setdefault(key, []).append((path, trace))
    return {
        key: tuple(sorted(values, key=lambda item: str(item[0])))
        for key, values in grouped.items()
    }


def _trace_runtime_profile(path: Path, trace: Mapping[str, Any]) -> str | None:
    metadata_profile = _nested(trace, "metadata", "runtime_profile")
    if metadata_profile is not None:
        profile = str(metadata_profile).strip().lower().replace("-", "_")
        if profile in RUNTIME_PROFILE_NAMES:
            return profile
    parent_profile = path.parent.name.strip().lower().replace("-", "_")
    return parent_profile if parent_profile in RUNTIME_PROFILE_NAMES else None


def _trace_request_key(path: Path, trace: Mapping[str, Any]) -> str:
    metadata_key = _nested(trace, "metadata", "runtime_replay_key")
    if metadata_key is not None and str(metadata_key).strip():
        return str(metadata_key).strip()
    request_id = trace.get("request_id")
    if request_id is not None and str(request_id).strip():
        normalized = str(request_id).strip()
        for prefix in (*RUNTIME_PROFILE_NAMES, "auto"):
            marker = f"{prefix}-"
            if normalized.startswith(marker):
                return normalized[len(marker):]
        return normalized
    return path.stem


def _runtime_total_seconds(trace: Mapping[str, Any]) -> float | None:
    runtime_trace = trace.get("runtime_trace")
    if not isinstance(runtime_trace, Mapping):
        return None
    total_seconds = _float_or_none(runtime_trace.get("total_seconds"))
    if total_seconds is not None:
        return total_seconds
    summary = _mapping(runtime_trace.get("summary"))
    return _float_or_none(summary.get("total_seconds"))


def _trace_paths_from_args(values: Sequence[str], globs: Sequence[str]) -> tuple[Path, ...]:
    paths = [Path(value) for value in values]
    for pattern in globs:
        paths.extend(Path(match) for match in sorted(glob.glob(pattern)))
    unique = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if not unique:
        raise ValueError("at least one --trace or --trace-glob match is required.")
    return tuple(unique)


def _profile_cost_mapping(values: Mapping[str, Any]) -> dict[str, float]:
    costs = {}
    for profile in RUNTIME_PROFILE_NAMES:
        raw = values.get(profile, DEFAULT_PROFILE_COST_UNITS[profile])
        cost = _required_non_negative_float(raw, name=f"profile_cost_units.{profile}")
        costs[profile] = cost
    return costs


def _profile_count_mapping(values: Mapping[str, Any], *, field_name: str) -> dict[str, int]:
    counts = {}
    for raw_profile, raw_count in values.items():
        profile = _normalize_profile(raw_profile)
        count = _required_non_negative_int(raw_count, name=f"{field_name}.{profile}")
        counts[profile] = count
    return counts


def _profile_rate_mapping(values: Mapping[str, Any], *, field_name: str) -> dict[str, float]:
    rates = {}
    for raw_profile, raw_rate in values.items():
        profile = _normalize_profile(raw_profile)
        rate = _required_rate_float(raw_rate, name=f"{field_name}.{profile}")
        rates[profile] = rate
    return rates


def _normalize_profile(value: Any) -> str:
    profile = str(value).strip().lower().replace("-", "_")
    if profile not in RUNTIME_PROFILE_NAMES:
        raise ValueError(f"profile must be one of: {', '.join(RUNTIME_PROFILE_NAMES)}")
    return profile


def _max_check(
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


def _min_check(
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


def _optional_non_negative_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _required_non_negative_float(value, name=name)


def _required_non_negative_float(value: Any, *, name: str) -> float:
    numeric = _float_or_none(value)
    if numeric is None or numeric < 0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return numeric


def _required_rate_float(value: Any, *, name: str) -> float:
    numeric = _required_non_negative_float(value, name=name)
    if numeric > 1:
        raise ValueError(f"{name} must be between 0 and 1.")
    return numeric


def _optional_rate_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _required_rate_float(value, name=name)


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


def _runtime_seconds_stats(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if _float_or_none(value) is not None]
    if not finite:
        return {
            "count": 0,
            "total_seconds": None,
            "mean_seconds": None,
            "min_seconds": None,
            "p95_seconds": None,
            "p99_seconds": None,
            "max_seconds": None,
        }
    total = sum(finite)
    return {
        "count": len(finite),
        "total_seconds": total,
        "mean_seconds": total / len(finite),
        "min_seconds": min(finite),
        "p95_seconds": _percentile_or_none(finite, 95.0),
        "p99_seconds": _percentile_or_none(finite, 99.0),
        "max_seconds": max(finite),
    }


def _percentile_or_none(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if not (0.0 <= percentile <= 100.0):
        raise ValueError("percentile must be between 0 and 100.")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return lower + (upper - lower) * (rank - lower_index)


def _safe_div(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


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


def _parse_mapping_json(value: str | None, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return dict(payload)


def _config_from_args(args: argparse.Namespace) -> RuntimeProfileSelectorReplayConfig:
    if not args.candidate:
        raise ValueError("--candidate is required for selector replay.")
    return RuntimeProfileSelectorReplayConfig(
        trace_paths=_trace_paths_from_args(args.trace or (), args.trace_glob or ()),
        output_dir=Path(args.output_dir),
        candidates=tuple(_load_candidate(value) for value in args.candidate),
        replay_policy_path=Path(args.replay_policy) if args.replay_policy else None,
        profile_cost_units=_parse_mapping_json(args.profile_cost_units, name="--profile-cost-units"),
        report_path=Path(args.json) if args.json else None,
        artifact_manifest_path=Path(args.artifact_manifest) if args.artifact_manifest else None,
        registry_path=Path(args.registry) if args.registry else None,
        name=args.name,
        version=args.version,
        metadata=_parse_mapping_json(args.metadata_json, name="--metadata-json"),
        compact_json=bool(args.compact_json),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = run_runtime_profile_selector_replay(_config_from_args(args))
    print(_json_text(report, compact=bool(args.compact_json)), end="")
    if args.fail_on_blocked and report["status"] == "blocked":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay runtime-profile selector policies over ProductTrace JSON")
    parser.add_argument("--trace", action="append", default=[], help="ProductTrace JSON path; repeatable")
    parser.add_argument("--trace-glob", action="append", default=[], help="glob for ProductTrace JSON files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", action="append", default=[],
                        help="candidate selector policy JSON path, or name=path; repeatable")
    parser.add_argument("--replay-policy", default=None, help="RuntimeProfileSelectorReplayPolicy JSON path")
    parser.add_argument("--profile-cost-units", default=None,
                        help="optional JSON object overriding latency/balanced/audit cost units")
    parser.add_argument("--json", default=None, help="top-level replay report path")
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata-json", default=None)
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
