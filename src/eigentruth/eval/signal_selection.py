"""Evidence-driven signal selection from fusion ablation reports."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from eigentruth.json_utils import strict_json_dumps, to_jsonable

DEFAULT_TRACKED_SIGNAL = "trajectory_convergence"


@dataclass(frozen=True)
class SignalSelectionPolicy:
    """Policy for conditionally enabling one tracked signal.

    The policy compares the best candidate that contains ``tracked_signal`` with
    the best candidate that does not. The tracked signal is selected only when it
    improves or preserves detection/AUROC within configured margins and does not
    increase false alarms beyond the allowed delta.
    """

    tracked_signal: str = DEFAULT_TRACKED_SIGNAL
    alpha: float = 0.1
    min_detection_delta: float = 0.0
    min_auroc_delta: float = 0.0
    max_false_alarm_delta: float = 0.03

    def __post_init__(self) -> None:
        tracked_signal = str(self.tracked_signal).strip()
        if not tracked_signal:
            raise ValueError("tracked_signal must be non-empty.")
        alpha = _finite_float(self.alpha, name="alpha")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        min_detection_delta = _non_negative_float(self.min_detection_delta, name="min_detection_delta")
        min_auroc_delta = _non_negative_float(self.min_auroc_delta, name="min_auroc_delta")
        max_false_alarm_delta = _non_negative_float(self.max_false_alarm_delta, name="max_false_alarm_delta")
        object.__setattr__(self, "tracked_signal", tracked_signal)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "min_detection_delta", min_detection_delta)
        object.__setattr__(self, "min_auroc_delta", min_auroc_delta)
        object.__setattr__(self, "max_false_alarm_delta", max_false_alarm_delta)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "tracked_signal": self.tracked_signal,
            "alpha": self.alpha,
            "min_detection_delta": self.min_detection_delta,
            "min_auroc_delta": self.min_auroc_delta,
            "max_false_alarm_delta": self.max_false_alarm_delta,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignalSelectionPolicy":
        """Build a policy from JSON-like data."""
        return cls(
            tracked_signal=str(data.get("tracked_signal", DEFAULT_TRACKED_SIGNAL)),
            alpha=data.get("alpha", 0.1),
            min_detection_delta=data.get("min_detection_delta", 0.0),
            min_auroc_delta=data.get("min_auroc_delta", 0.0),
            max_false_alarm_delta=data.get("max_false_alarm_delta", 0.03),
        )


@dataclass(frozen=True)
class SignalSelectionDecision:
    """Selected signal bundle for one ablation run."""

    run_name: str
    selected_candidate: str
    selected_signals: tuple[str, ...]
    selected_method: str
    selected_metrics: Mapping[str, float]
    directions: Mapping[str, str]
    tracked_signal: str = DEFAULT_TRACKED_SIGNAL
    tracked_signal_enabled: bool = False
    tracked_candidate: str | None = None
    baseline_candidate: str | None = None
    metric_deltas: Mapping[str, float] = field(default_factory=dict)
    policy_checks: Mapping[str, bool] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        run_name = str(self.run_name).strip()
        selected_candidate = str(self.selected_candidate).strip()
        selected_method = str(self.selected_method).strip()
        tracked_signal = str(self.tracked_signal).strip()
        selected_signals = tuple(str(signal).strip() for signal in self.selected_signals if str(signal).strip())
        if not run_name:
            raise ValueError("run_name must be non-empty.")
        if not selected_candidate:
            raise ValueError("selected_candidate must be non-empty.")
        if not selected_signals:
            raise ValueError("selected_signals must be non-empty.")
        if not selected_method:
            raise ValueError("selected_method must be non-empty.")
        if not tracked_signal:
            raise ValueError("tracked_signal must be non-empty.")
        metrics = _metric_mapping(self.selected_metrics, name="selected_metrics")
        directions = _direction_mapping(self.directions, selected_signals=selected_signals)
        deltas = _metric_mapping(self.metric_deltas, name="metric_deltas", allow_empty=True)
        checks = {str(key): bool(value) for key, value in self.policy_checks.items()}
        object.__setattr__(self, "run_name", run_name)
        object.__setattr__(self, "selected_candidate", selected_candidate)
        object.__setattr__(self, "selected_signals", selected_signals)
        object.__setattr__(self, "selected_method", selected_method)
        object.__setattr__(self, "selected_metrics", metrics)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "tracked_signal", tracked_signal)
        object.__setattr__(self, "tracked_signal_enabled", bool(self.tracked_signal_enabled))
        object.__setattr__(
            self,
            "tracked_candidate",
            None if self.tracked_candidate is None else str(self.tracked_candidate),
        )
        object.__setattr__(
            self,
            "baseline_candidate",
            None if self.baseline_candidate is None else str(self.baseline_candidate),
        )
        object.__setattr__(self, "metric_deltas", deltas)
        object.__setattr__(self, "policy_checks", checks)
        object.__setattr__(self, "reason", str(self.reason))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "run_name": self.run_name,
            "selected_candidate": self.selected_candidate,
            "selected_signals": list(self.selected_signals),
            "selected_method": self.selected_method,
            "selected_metrics": dict(self.selected_metrics),
            "directions": dict(self.directions),
            "tracked_signal": self.tracked_signal,
            "tracked_signal_enabled": self.tracked_signal_enabled,
            "tracked_candidate": self.tracked_candidate,
            "baseline_candidate": self.baseline_candidate,
            "metric_deltas": dict(self.metric_deltas),
            "policy_checks": dict(self.policy_checks),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignalSelectionDecision":
        """Build a decision from JSON-like data."""
        return cls(
            run_name=str(data["run_name"]),
            selected_candidate=str(data["selected_candidate"]),
            selected_signals=tuple(data["selected_signals"]),
            selected_method=str(data["selected_method"]),
            selected_metrics=dict(data["selected_metrics"]),
            directions=dict(data.get("directions", {})),
            tracked_signal=str(data.get("tracked_signal", DEFAULT_TRACKED_SIGNAL)),
            tracked_signal_enabled=bool(data.get("tracked_signal_enabled", False)),
            tracked_candidate=None if data.get("tracked_candidate") is None else str(data["tracked_candidate"]),
            baseline_candidate=None if data.get("baseline_candidate") is None else str(data["baseline_candidate"]),
            metric_deltas=dict(data.get("metric_deltas", {})),
            policy_checks=dict(data.get("policy_checks", {})),
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=True)
class SignalSelectionReport:
    """Signal selection report derived from an ablation matrix."""

    decisions: tuple[SignalSelectionDecision, ...]
    policy: SignalSelectionPolicy = field(default_factory=SignalSelectionPolicy)
    source_workflow: str | None = None
    source_status: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    workflow: str = "fusion_signal_selection"
    status: str = "complete"

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        if not decisions:
            raise ValueError("decisions must be non-empty.")
        object.__setattr__(self, "decisions", decisions)
        if not isinstance(self.policy, SignalSelectionPolicy):
            object.__setattr__(self, "policy", SignalSelectionPolicy.from_dict(self.policy))  # type: ignore[arg-type]
        object.__setattr__(self, "source_workflow", None if self.source_workflow is None else str(self.source_workflow))
        object.__setattr__(self, "source_status", None if self.source_status is None else str(self.source_status))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "workflow", str(self.workflow))
        object.__setattr__(self, "status", str(self.status))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "schema_version": self.schema_version,
            "workflow": self.workflow,
            "status": self.status,
            "policy": self.policy.to_dict(),
            "source_workflow": self.source_workflow,
            "source_status": self.source_status,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignalSelectionReport":
        """Build a report from JSON-like data."""
        return cls(
            decisions=tuple(SignalSelectionDecision.from_dict(item) for item in data["decisions"]),
            policy=SignalSelectionPolicy.from_dict(data.get("policy", {})),
            source_workflow=None if data.get("source_workflow") is None else str(data["source_workflow"]),
            source_status=None if data.get("source_status") is None else str(data["source_status"]),
            metadata=dict(data.get("metadata", {})),
            schema_version=int(data.get("schema_version", 1)),
            workflow=str(data.get("workflow", "fusion_signal_selection")),
            status=str(data.get("status", "complete")),
        )

    def selected_by_run(self) -> dict[str, SignalSelectionDecision]:
        """Return decisions keyed by run name."""
        return {decision.run_name: decision for decision in self.decisions}

    def save_json(self, path: str | Path) -> None:
        """Save report as UTF-8 JSON."""
        Path(path).write_text(strict_json_dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "SignalSelectionReport":
        """Load report from UTF-8 JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def select_signals_from_fusion_ablation_matrix(
    matrix: Mapping[str, Any],
    *,
    policy: SignalSelectionPolicy | Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SignalSelectionReport:
    """Select signal bundles from a ``run_fusion_ablation_matrix`` payload."""
    resolved_policy = _policy(policy)
    runs = matrix.get("runs")
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes, bytearray)) or not runs:
        raise ValueError("matrix must contain a non-empty runs sequence.")
    decisions = tuple(_decision_for_run(run, resolved_policy) for run in runs)
    return SignalSelectionReport(
        decisions=decisions,
        policy=resolved_policy,
        source_workflow=None if matrix.get("workflow") is None else str(matrix.get("workflow")),
        source_status=None if matrix.get("status") is None else str(matrix.get("status")),
        metadata={} if metadata is None else dict(metadata),
    )


def _decision_for_run(run: Any, policy: SignalSelectionPolicy) -> SignalSelectionDecision:
    if not isinstance(run, Mapping):
        raise ValueError("each matrix run must be a mapping.")
    run_name = str(run.get("name") or "").strip()
    if not run_name:
        raise ValueError("matrix run is missing a non-empty name.")
    results = run.get("candidate_results")
    if not isinstance(results, Mapping) or not results:
        raise ValueError(f"run {run_name!r} must contain candidate_results.")
    candidate_rows = tuple(
        _candidate_row(str(name), result, alpha=policy.alpha)
        for name, result in results.items()
    )
    tracked_rows = tuple(row for row in candidate_rows if policy.tracked_signal in row["signals"])
    baseline_rows = tuple(row for row in candidate_rows if policy.tracked_signal not in row["signals"])
    best_tracked = _best_candidate(tracked_rows)
    best_baseline = _best_candidate(baseline_rows)

    if best_tracked is None and best_baseline is None:
        raise ValueError(f"run {run_name!r} has no eligible candidates.")
    if best_tracked is None:
        assert best_baseline is not None
        selected = best_baseline
        deltas: dict[str, float] = {}
        checks: dict[str, bool] = {"tracked_candidate_available": False}
        enabled = False
        reason = f"no candidate contains tracked signal {policy.tracked_signal!r}"
    elif best_baseline is None:
        selected = best_tracked
        deltas = {}
        checks = {"baseline_candidate_available": False, "tracked_candidate_available": True}
        enabled = True
        reason = f"tracked signal {policy.tracked_signal!r} is selected because no baseline candidate is available"
    else:
        deltas = _metric_deltas(best_tracked, best_baseline)
        checks = {
            "tracked_candidate_available": True,
            "baseline_candidate_available": True,
            "detection_delta_pass": deltas["detection"] >= policy.min_detection_delta,
            "auroc_delta_pass": deltas["auroc"] >= policy.min_auroc_delta,
            "false_alarm_delta_pass": deltas["false_alarm"] <= policy.max_false_alarm_delta,
        }
        enabled = all(checks.values())
        selected = best_tracked if enabled else best_baseline
        reason = _reason(policy, enabled=enabled, deltas=deltas, tracked=best_tracked, baseline=best_baseline)

    directions = _directions_for_selected(run.get("directions"), selected["signals"])
    return SignalSelectionDecision(
        run_name=run_name,
        selected_candidate=str(selected["candidate_name"]),
        selected_signals=tuple(selected["signals"]),
        selected_method=str(selected["method"]),
        selected_metrics=_selected_metrics(selected),
        directions=directions,
        tracked_signal=policy.tracked_signal,
        tracked_signal_enabled=enabled,
        tracked_candidate=None if best_tracked is None else str(best_tracked["candidate_name"]),
        baseline_candidate=None if best_baseline is None else str(best_baseline["candidate_name"]),
        metric_deltas=deltas,
        policy_checks=checks,
        reason=reason,
    )


def _candidate_row(name: str, result: Any, *, alpha: float) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError(f"candidate {name!r} must be a mapping.")
    signals = tuple(str(signal).strip() for signal in result.get("signals", ()) if str(signal).strip())
    if not signals:
        raise ValueError(f"candidate {name!r} is missing signals.")
    alphas = result.get("alphas")
    if not isinstance(alphas, Mapping) or not alphas:
        raise ValueError(f"candidate {name!r} is missing alpha results.")
    alpha_payload = _alpha_payload(alphas, alpha=alpha, candidate_name=name)
    return {
        "candidate_name": name,
        "signals": signals,
        "method": str(result.get("method") or ""),
        "auroc": _finite_float(result.get("auroc"), name=f"{name}.auroc"),
        "alpha": alpha,
        "false_alarm": _finite_float(alpha_payload.get("false_alarm"), name=f"{name}.false_alarm"),
        "detection": _finite_float(alpha_payload.get("detection"), name=f"{name}.detection"),
        "coverage": _finite_float(alpha_payload.get("coverage"), name=f"{name}.coverage"),
    }


def _alpha_payload(alphas: Mapping[Any, Any], *, alpha: float, candidate_name: str) -> Mapping[str, Any]:
    for key, payload in alphas.items():
        try:
            if math.isclose(float(key), alpha, rel_tol=0.0, abs_tol=1e-12):
                if not isinstance(payload, Mapping):
                    raise ValueError(f"candidate {candidate_name!r} alpha payload must be a mapping.")
                return payload
        except (TypeError, ValueError):
            continue
    raise ValueError(f"candidate {candidate_name!r} does not contain alpha {alpha}.")


def _best_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item["detection"]),
            float(item["auroc"]),
            -float(item["false_alarm"]),
        ),
    )


def _metric_deltas(tracked: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    return {
        "detection": float(tracked["detection"]) - float(baseline["detection"]),
        "auroc": float(tracked["auroc"]) - float(baseline["auroc"]),
        "false_alarm": float(tracked["false_alarm"]) - float(baseline["false_alarm"]),
        "coverage": float(tracked["coverage"]) - float(baseline["coverage"]),
    }


def _selected_metrics(candidate: Mapping[str, Any]) -> dict[str, float]:
    return {
        "alpha": float(candidate["alpha"]),
        "auroc": float(candidate["auroc"]),
        "false_alarm": float(candidate["false_alarm"]),
        "detection": float(candidate["detection"]),
        "coverage": float(candidate["coverage"]),
    }


def _directions_for_selected(raw_directions: Any, signals: Sequence[str]) -> dict[str, str]:
    directions = raw_directions if isinstance(raw_directions, Mapping) else {}
    return _direction_mapping(
        {signal: str(directions.get(signal, "higher")) for signal in signals},
        selected_signals=signals,
    )


def _reason(
    policy: SignalSelectionPolicy,
    *,
    enabled: bool,
    deltas: Mapping[str, float],
    tracked: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> str:
    prefix = "enable" if enabled else "disable"
    return (
        f"{prefix} {policy.tracked_signal}: "
        f"tracked={tracked['candidate_name']} baseline={baseline['candidate_name']} "
        f"detection_delta={deltas['detection']:.6g} "
        f"auroc_delta={deltas['auroc']:.6g} "
        f"false_alarm_delta={deltas['false_alarm']:.6g}"
    )


def _policy(policy: SignalSelectionPolicy | Mapping[str, Any] | None) -> SignalSelectionPolicy:
    if policy is None:
        return SignalSelectionPolicy()
    if isinstance(policy, SignalSelectionPolicy):
        return policy
    return SignalSelectionPolicy.from_dict(policy)


def _metric_mapping(values: Mapping[str, Any], *, name: str, allow_empty: bool = False) -> dict[str, float]:
    if not values and allow_empty:
        return {}
    result = {str(key): _finite_float(value, name=f"{name}.{key}") for key, value in values.items()}
    if not result and not allow_empty:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _direction_mapping(values: Mapping[str, Any], *, selected_signals: Sequence[str]) -> dict[str, str]:
    result = {str(key): str(value) for key, value in values.items()}
    for signal in selected_signals:
        direction = result.get(signal)
        if direction not in {"higher", "lower"}:
            raise ValueError(f"direction for selected signal {signal!r} must be 'higher' or 'lower'.")
    return {signal: result[signal] for signal in selected_signals}


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def _non_negative_float(value: Any, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric
