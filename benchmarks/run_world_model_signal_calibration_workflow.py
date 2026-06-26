"""Run a deterministic world-model signal calibration workflow.

This no-model workflow builds a local state-transition fixture, verifies claims
through the world-model transition route, converts verifier outputs into
standard score-dump columns, then evaluates those columns with the existing
score-fusion calibration path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.build_transition_fixture import build_order_transition_fixture  # noqa: E402
from benchmarks.build_verifier_signal_score_dump import DEFAULT_VERIFIER_SIGNALS  # noqa: E402
from benchmarks.eval_score_ensemble import GEOMETRY_FUSION_METHODS, METHODS  # noqa: E402
from benchmarks.run_verifier_signal_fusion_workflow import (  # noqa: E402
    VerifierSignalFusionWorkflowConfig,
    run_verifier_signal_fusion_workflow,
)
from eigentruth.eval.score_dump import load_score_dump  # noqa: E402
from eigentruth.registry import ArtifactRegistry, ArtifactVerificationContext  # noqa: E402

DEFAULT_WORLD_MODEL_FUSION_SIGNALS = (
    "truth_proj",
    "verifier_refuted",
    "verifier_uncertainty",
    "world_model_disagreement",
    "world_model_agreement_gap",
    "world_model_low_agreement",
    "world_model_conflict",
    "world_model_conflict_delta",
)
DEFAULT_WORLD_MODEL_UNCERTAINTY_SIGNALS = (
    "verifier_refuted",
    "world_model_disagreement",
    "world_model_agreement_gap",
    "world_model_low_agreement",
    "world_model_conflict",
)
WORLD_MODEL_CONFLICT_SIGNALS = ("world_model_conflict", "world_model_conflict_delta")


@dataclass(frozen=True)
class WorldModelSignalCalibrationWorkflowConfig:
    """Configuration for the deterministic world-model calibration workflow."""

    output_dir: Path
    run_name: str = "synthetic-world-model"
    signal: str = "truth_proj"
    n_records: int = 24
    rule_based_world_model: bool = True
    world_model_ensemble: bool = False
    world_model_ensemble_min_agreement: float = 0.75
    world_model_ensemble_strategy: str = "label_stress"
    alphas: Sequence[float] = (0.10,)
    repeats: int = 20
    seed: int = 0
    best_alpha: float = 0.10
    verifier_signals: Sequence[str] = DEFAULT_VERIFIER_SIGNALS
    fusion_signals: Sequence[str] = DEFAULT_WORLD_MODEL_FUSION_SIGNALS
    methods: Sequence[str] = METHODS
    geometry_signals: Sequence[str] | None = None
    uncertainty_signals: Sequence[str] = DEFAULT_WORLD_MODEL_UNCERTAINTY_SIGNALS
    geometry_method: str = "mean_rank"
    uncertainty_method: str = "mean_rank"
    geometry_fusion_methods: Sequence[str] = GEOMETRY_FUSION_METHODS
    min_world_model_confidence: float = 0.0
    compact_json: bool = False
    verify_manifest: bool = True
    registry_path: Path | None = None
    registry_name: str = "world-model-signal-calibration"
    registry_version: str = "0.1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        object.__setattr__(self, "run_name", _non_empty_string(self.run_name, name="run_name"))
        object.__setattr__(self, "signal", _non_empty_string(self.signal, name="signal"))
        for attr in (
            "alphas",
            "verifier_signals",
            "fusion_signals",
            "methods",
            "uncertainty_signals",
            "geometry_fusion_methods",
        ):
            object.__setattr__(self, attr, tuple(getattr(self, attr)))
        if self.geometry_signals is None:
            object.__setattr__(self, "geometry_signals", (self.signal,))
        else:
            object.__setattr__(self, "geometry_signals", tuple(self.geometry_signals))
        if int(self.n_records) < 4:
            raise ValueError("n_records must be >= 4.")
        if not (0.0 < float(self.world_model_ensemble_min_agreement) <= 1.0):
            raise ValueError("world_model_ensemble_min_agreement must be in (0, 1].")
        object.__setattr__(
            self,
            "world_model_ensemble_strategy",
            _world_model_ensemble_strategy(self.world_model_ensemble_strategy),
        )
        if int(self.repeats) < 1:
            raise ValueError("repeats must be >= 1.")
        if any(not (0.0 < float(alpha) < 1.0) for alpha in self.alphas):
            raise ValueError("alphas must be in (0, 1).")
        if not (0.0 < float(self.best_alpha) < 1.0):
            raise ValueError("best_alpha must be in (0, 1).")
        if not (0.0 <= float(self.min_world_model_confidence) <= 1.0):
            raise ValueError("min_world_model_confidence must be in [0, 1].")
        if not self.fusion_signals:
            raise ValueError("fusion_signals must contain at least one signal.")
        if not self.uncertainty_signals:
            raise ValueError("uncertainty_signals must contain at least one signal.")

    @property
    def fixture_dir(self) -> Path:
        return self.output_dir / "fixture"

    @property
    def scores_path(self) -> Path:
        return self.fixture_dir / "scores.json"

    @property
    def claims_path(self) -> Path:
        return self.fixture_dir / "claims.json"

    @property
    def state_path(self) -> Path:
        return self.fixture_dir / "state.json"

    @property
    def verifier_signal_dir(self) -> Path:
        return self.output_dir / "verifier-signal-fusion"

    @property
    def workflow_report_path(self) -> Path:
        return self.output_dir / "world-model-signal-calibration-workflow.json"

    @property
    def artifact_manifest_path(self) -> Path:
        return self.output_dir / "artifact-manifest.json"

    @property
    def manifest_verification_path(self) -> Path:
        return self.output_dir / "manifest-verification.json"


def run_world_model_signal_calibration_workflow(
    config: WorldModelSignalCalibrationWorkflowConfig,
) -> dict[str, Any]:
    """Run fixture construction, world-model verification, and signal fusion."""
    started = time.perf_counter()
    profile: dict[str, float] = {}
    config.output_dir.mkdir(parents=True, exist_ok=True)

    with _profile_phase(profile, "build_transition_fixture"):
        fixture = build_order_transition_fixture(
            n_records=int(config.n_records),
            signal=config.signal,
            rule_based_world_model=bool(config.rule_based_world_model),
            world_model_ensemble=bool(config.world_model_ensemble),
            world_model_ensemble_min_agreement=float(config.world_model_ensemble_min_agreement),
            world_model_ensemble_strategy=config.world_model_ensemble_strategy,
        )
        _write_json(config.scores_path, fixture["scores"], compact=config.compact_json)
        _write_json(config.claims_path, fixture["claims"], compact=config.compact_json)
        _write_json(config.state_path, fixture["state"], compact=config.compact_json)

    with _profile_phase(profile, "run_verifier_signal_fusion"):
        verifier_signal_payload = run_verifier_signal_fusion_workflow(
            VerifierSignalFusionWorkflowConfig(
                score_dumps=((config.run_name, config.scores_path),),
                output_dir=config.verifier_signal_dir,
                claims_path=config.claims_path,
                state_path=config.state_path,
                signal=config.signal,
                direction="higher",
                alphas=tuple(float(alpha) for alpha in config.alphas),
                repeats=int(config.repeats),
                seed=int(config.seed),
                best_alpha=float(config.best_alpha),
                keep_signals=(config.signal,),
                verifier_signals=tuple(config.verifier_signals),
                fusion_signals=tuple(config.fusion_signals),
                methods=tuple(config.methods),
                geometry_signals=tuple(config.geometry_signals or ()),
                uncertainty_signals=tuple(config.uncertainty_signals),
                geometry_method=config.geometry_method,
                uncertainty_method=config.uncertainty_method,
                geometry_fusion_methods=tuple(config.geometry_fusion_methods),
                verifier_min_overlap=0.65,
                min_world_model_confidence=float(config.min_world_model_confidence),
                compact_json=bool(config.compact_json),
                verify_manifest=bool(config.verify_manifest),
            )
        )

    enhanced_path = config.verifier_signal_dir / f"{config.run_name}-enhanced-scores.manifest.json"
    enhanced_dump = load_score_dump(
        enhanced_path,
        required_scores=tuple(signal for signal in config.fusion_signals if signal != config.signal),
    )
    score_ensemble_report_path = Path(verifier_signal_payload["score_ensemble_report_path"])
    score_ensemble_report = _read_json_mapping(score_ensemble_report_path)
    release_gate = _world_model_release_gate(
        enhanced_scores=enhanced_dump.scores,
        score_ensemble_report=score_ensemble_report,
        best_alpha=float(config.best_alpha),
    )
    manifest_metadata = _manifest_metadata(
        config,
        fixture=fixture,
        verifier_signal_payload=verifier_signal_payload,
        enhanced_dump_summary=enhanced_dump.summary(),
        release_gate=release_gate,
        profile=profile,
        total_seconds=time.perf_counter() - started,
    )
    with _profile_phase(profile, "write_artifact_manifest"):
        manifest = _write_artifact_manifest(config, metadata=manifest_metadata)

    manifest_verification = None
    if config.verify_manifest:
        with _profile_phase(profile, "verify_artifact_manifest"):
            context = ArtifactVerificationContext()
            manifest_verification = context.load_and_verify_artifact_manifest(
                config.artifact_manifest_path,
                root=config.artifact_manifest_path.parent,
                recursive=True,
            ).to_dict()
            _write_json(config.manifest_verification_path, manifest_verification, compact=False)

    registry_record_key = None
    if config.registry_path is not None:
        with _profile_phase(profile, "record_registry"):
            registry = ArtifactRegistry.load_json(config.registry_path)
            registry.record_report(
                name=config.registry_name,
                path=config.workflow_report_path,
                version=config.registry_version,
                metadata={
                    "workflow": "world_model_signal_calibration_workflow",
                    "artifact_manifest": str(config.artifact_manifest_path),
                    "manifest_verification": None
                    if manifest_verification is None
                    else str(config.manifest_verification_path),
                    "run_name": config.run_name,
                    "n_records": int(config.n_records),
                    "rule_based_world_model": bool(config.rule_based_world_model),
                    "world_model_ensemble": bool(config.world_model_ensemble),
                    "world_model_rule_count": fixture["state"]["summary"]["n_world_model_rules"],
                    "world_model_ensemble_member_count": fixture["state"]["summary"][
                        "n_world_model_ensemble_members"
                    ],
                    "world_model_ensemble_strategy": fixture["state"]["summary"][
                        "world_model_ensemble_strategy"
                    ],
                    "world_model_ensemble_min_agreement": fixture["state"]["summary"][
                        "world_model_ensemble_min_agreement"
                    ],
                    "fusion_summary": verifier_signal_payload.get("fusion_summary"),
                    "release_gate_status": release_gate["status"],
                    "world_model_trace_gap_max": release_gate["score_summary"][
                        "world_model_trace_gap"
                    ]["max"],
                    "world_model_conflict_positive_count": release_gate["score_summary"][
                        "world_model_conflict"
                    ]["positive_count"],
                },
            )
            registry.save_json()
            registry_record_key = f"report:{config.registry_name}:{config.registry_version}"

    profile["total_seconds"] = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "workflow": "world_model_signal_calibration_workflow",
        "config": _config_payload(config),
        "fixture_paths": {
            "scores": str(config.scores_path),
            "claims": str(config.claims_path),
            "state": str(config.state_path),
        },
        "verifier_signal_fusion_workflow_path": str(
            config.verifier_signal_dir / "verifier-signal-fusion-workflow.json"
        ),
        "verifier_signal_fusion": verifier_signal_payload,
        "enhanced_score_dump": str(enhanced_path),
        "artifact_manifest_path": str(config.artifact_manifest_path),
        "manifest_verification_path": None
        if manifest_verification is None
        else str(config.manifest_verification_path),
        "manifest_summary": manifest.get("summary"),
        "manifest_verification": manifest_verification,
        "registry_record_key": registry_record_key,
        "release_gate": release_gate,
        "world_model_summary": {
            "adapter": (
                "EnsembleWorldModelAdapter"
                if config.world_model_ensemble
                else "RuleBasedWorldModelAdapter"
                if config.rule_based_world_model
                else "InMemoryWorldModelAdapter"
            ),
            "rule_count": fixture["state"]["summary"]["n_world_model_rules"],
            "member_count": fixture["state"]["summary"]["n_world_model_ensemble_members"],
            "strategy": fixture["state"]["summary"]["world_model_ensemble_strategy"],
            "min_agreement": fixture["state"]["summary"]["world_model_ensemble_min_agreement"],
            "signals": list(config.fusion_signals),
            "uncertainty_signals": list(config.uncertainty_signals),
        },
        "profile": dict(profile),
    }
    _write_json(config.workflow_report_path, payload, compact=config.compact_json)
    return payload


def _write_artifact_manifest(
    config: WorldModelSignalCalibrationWorkflowConfig,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    context = ArtifactVerificationContext()
    manifest = context.build_artifact_manifest(
        {
            "fixture.scores": config.scores_path,
            "fixture.claims": config.claims_path,
            "fixture.state": config.state_path,
            "verifier_signal_workflow": config.verifier_signal_dir / "verifier-signal-fusion-workflow.json",
            "verifier_signal_manifest": config.verifier_signal_dir / "artifact-manifest.json",
            "verifier_signal_manifest_verification": config.verifier_signal_dir / "manifest-verification.json",
        },
        root=config.artifact_manifest_path.parent,
        metadata=metadata,
    )
    _write_json(config.artifact_manifest_path, manifest, compact=False)
    return manifest


def _manifest_metadata(
    config: WorldModelSignalCalibrationWorkflowConfig,
    *,
    fixture: Mapping[str, Any],
    verifier_signal_payload: Mapping[str, Any],
    enhanced_dump_summary: Mapping[str, Any],
    release_gate: Mapping[str, Any],
    profile: Mapping[str, float],
    total_seconds: float,
) -> dict[str, Any]:
    return {
        "runner": "run_world_model_signal_calibration_workflow",
        "workflow": "world_model_signal_calibration_workflow",
        "run_name": config.run_name,
        "signal": config.signal,
        "n_records": int(config.n_records),
        "rule_based_world_model": bool(config.rule_based_world_model),
        "world_model_ensemble": bool(config.world_model_ensemble),
        "world_model_ensemble_min_agreement": float(config.world_model_ensemble_min_agreement),
        "world_model_ensemble_strategy": config.world_model_ensemble_strategy,
        "world_model_rule_count": fixture["state"]["summary"]["n_world_model_rules"],
        "world_model_ensemble_member_count": fixture["state"]["summary"]["n_world_model_ensemble_members"],
        "fixture_summary": dict(fixture["claims"].get("summary", {})),
        "enhanced_score_dump_summary": dict(enhanced_dump_summary),
        "fusion_summary": verifier_signal_payload.get("fusion_summary"),
        "verifier_summary": verifier_signal_payload.get("verifier_summary"),
        "release_gate_status": release_gate.get("status"),
        "world_model_trace_gap_max": _nested_float(
            release_gate,
            "score_summary",
            "world_model_trace_gap",
            "max",
        ),
        "world_model_conflict_positive_count": _nested_float(
            release_gate,
            "score_summary",
            "world_model_conflict",
            "positive_count",
        ),
        "profile": dict(profile),
        "total_seconds": float(total_seconds),
    }


def _world_model_release_gate(
    *,
    enhanced_scores: Mapping[str, Sequence[float]],
    score_ensemble_report: Mapping[str, Any],
    best_alpha: float,
) -> dict[str, Any]:
    score_summary = {
        "world_model_conflict": _score_signal_summary(enhanced_scores, "world_model_conflict"),
        "world_model_conflict_delta": _score_signal_summary(
            enhanced_scores,
            "world_model_conflict_delta",
        ),
        "world_model_trace_gap": _score_signal_summary(enhanced_scores, "world_model_trace_gap"),
    }
    calibrated_conflict_signals = _calibrated_conflict_signal_summaries(
        score_ensemble_report,
        best_alpha=best_alpha,
    )
    failures = []
    trace_gap = score_summary["world_model_trace_gap"]
    if not trace_gap["present"]:
        failures.append("world_model_trace_gap score is missing from enhanced score dump")
    elif trace_gap["max"] is None or trace_gap["max"] > 0.0:
        failures.append(
            "world_model_trace_gap must be zero for release, "
            f"observed max={trace_gap['max']!r}"
        )
    conflict = score_summary["world_model_conflict"]
    if not conflict["present"]:
        failures.append("world_model_conflict score is missing from enhanced score dump")
    elif int(conflict["positive_count"] or 0) < 1:
        failures.append("world_model_conflict has no positive conflict examples")
    if not calibrated_conflict_signals:
        failures.append("score ensemble report did not calibrate a world-model conflict signal")
    elif not any(item["passes_calibration_gate"] for item in calibrated_conflict_signals):
        failures.append(
            "no calibrated world-model conflict signal met the false-alarm and AUROC gate"
        )
    return {
        "schema_version": 1,
        "status": "promote" if not failures else "blocked",
        "passed": not failures,
        "blocking_reasons": failures,
        "policy": {
            "max_world_model_trace_gap": 0.0,
            "min_world_model_conflict_positive_count": 1,
            "conflict_signals": list(WORLD_MODEL_CONFLICT_SIGNALS),
            "best_alpha": float(best_alpha),
            "calibrated_signal_requires_detection": False,
            "min_calibrated_signal_auroc": 0.5,
        },
        "score_summary": score_summary,
        "calibrated_conflict_signals": calibrated_conflict_signals,
    }


def _score_signal_summary(
    scores: Mapping[str, Sequence[float]],
    name: str,
) -> dict[str, Any]:
    raw_values = scores.get(name)
    if raw_values is None:
        return {
            "present": False,
            "count": 0,
            "positive_count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
    values = [float(value) for value in raw_values]
    return {
        "present": True,
        "count": len(values),
        "positive_count": sum(1 for value in values if value > 0.0),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": (sum(values) / len(values)) if values else None,
    }


def _calibrated_conflict_signal_summaries(
    score_ensemble_report: Mapping[str, Any],
    *,
    best_alpha: float,
) -> list[dict[str, Any]]:
    alpha_key = str(float(best_alpha))
    summaries = []
    for run in score_ensemble_report.get("runs", ()):
        run_payload = _mapping(run)
        single_results = _mapping(run_payload.get("single_results"))
        for signal in WORLD_MODEL_CONFLICT_SIGNALS:
            result = _mapping(single_results.get(signal))
            alpha_payload = _mapping(_mapping(result.get("alphas")).get(alpha_key))
            if not result or not alpha_payload:
                continue
            false_alarm = _optional_float(alpha_payload.get("false_alarm"))
            detection = _optional_float(alpha_payload.get("detection"))
            auroc = _optional_float(result.get("auroc"))
            summaries.append({
                "run": run_payload.get("name"),
                "signal": signal,
                "alpha": float(best_alpha),
                "auroc": auroc,
                "false_alarm": false_alarm,
                "detection": detection,
                "passes_calibration_gate": (
                    false_alarm is not None
                    and auroc is not None
                    and false_alarm <= float(best_alpha)
                    and auroc > 0.5
                ),
            })
    return summaries


def _config_payload(config: WorldModelSignalCalibrationWorkflowConfig) -> dict[str, Any]:
    return {
        "output_dir": str(config.output_dir),
        "run_name": config.run_name,
        "signal": config.signal,
        "n_records": int(config.n_records),
        "rule_based_world_model": bool(config.rule_based_world_model),
        "world_model_ensemble": bool(config.world_model_ensemble),
        "world_model_ensemble_min_agreement": float(config.world_model_ensemble_min_agreement),
        "world_model_ensemble_strategy": config.world_model_ensemble_strategy,
        "alphas": [float(alpha) for alpha in config.alphas],
        "repeats": int(config.repeats),
        "seed": int(config.seed),
        "best_alpha": float(config.best_alpha),
        "verifier_signals": list(config.verifier_signals),
        "fusion_signals": list(config.fusion_signals),
        "methods": list(config.methods),
        "geometry_signals": list(config.geometry_signals or ()),
        "uncertainty_signals": list(config.uncertainty_signals),
        "geometry_method": config.geometry_method,
        "uncertainty_method": config.uncertainty_method,
        "geometry_fusion_methods": list(config.geometry_fusion_methods),
        "min_world_model_confidence": float(config.min_world_model_confidence),
        "verify_manifest": bool(config.verify_manifest),
        "registry_path": None if config.registry_path is None else str(config.registry_path),
        "registry_name": config.registry_name,
        "registry_version": config.registry_version,
    }


@contextmanager
def _profile_phase(profile: MutableMapping[str, float], name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        profile[name] = profile.get(name, 0.0) + (time.perf_counter() - started)


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _parse_csv(value: str | None, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError(f"{name} must contain at least one value.")
    if len(set(parts)) != len(parts):
        raise ValueError(f"{name} must contain unique values.")
    return parts


def _parse_float_csv(value: str | None, *, name: str) -> tuple[float, ...]:
    return tuple(float(item) for item in (_parse_csv(value, name=name) or ()))


def _non_empty_string(value: Any, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _nested_float(payload: Mapping[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return _optional_float(current)


def _world_model_ensemble_strategy(value: Any) -> str:
    strategy = str(value).strip()
    if strategy not in {"label_stress", "policy_replay"}:
        raise ValueError("world_model_ensemble_strategy must be 'label_stress' or 'policy_replay'.")
    return strategy


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from CLI-style parsed arguments."""
    config = WorldModelSignalCalibrationWorkflowConfig(
        output_dir=Path(args.output_dir),
        run_name=args.run_name,
        signal=args.signal,
        n_records=args.n_records,
        rule_based_world_model=not bool(args.direct_action_world_model),
        world_model_ensemble=bool(args.world_model_ensemble),
        world_model_ensemble_min_agreement=args.world_model_ensemble_min_agreement,
        world_model_ensemble_strategy=args.world_model_ensemble_strategy,
        alphas=_parse_float_csv(args.alphas, name="alphas"),
        repeats=args.repeats,
        seed=args.seed,
        best_alpha=args.best_alpha,
        verifier_signals=_parse_csv(args.verifier_signals, name="verifier_signals") or DEFAULT_VERIFIER_SIGNALS,
        fusion_signals=_parse_csv(args.fusion_signals, name="fusion_signals") or DEFAULT_WORLD_MODEL_FUSION_SIGNALS,
        methods=_parse_csv(args.methods, name="methods") or METHODS,
        geometry_signals=_parse_csv(args.geometry_signals, name="geometry_signals"),
        uncertainty_signals=(
            _parse_csv(args.uncertainty_signals, name="uncertainty_signals")
            or DEFAULT_WORLD_MODEL_UNCERTAINTY_SIGNALS
        ),
        geometry_method=args.geometry_method,
        uncertainty_method=args.uncertainty_method,
        geometry_fusion_methods=(
            _parse_csv(args.geometry_fusion_methods, name="geometry_fusion_methods")
            or GEOMETRY_FUSION_METHODS
        ),
        min_world_model_confidence=args.min_world_model_confidence,
        compact_json=bool(args.compact_json),
        verify_manifest=not bool(args.no_verify_manifest),
        registry_path=None if args.registry is None else Path(args.registry),
        registry_name=args.registry_name,
        registry_version=args.registry_version,
    )
    payload = run_world_model_signal_calibration_workflow(config)
    print(
        "world_model_signal_calibration_workflow_ok "
        f"manifest={payload['artifact_manifest_path']}"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic world-model signal calibration workflow")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="synthetic-world-model")
    parser.add_argument("--signal", default="truth_proj")
    parser.add_argument("--n-records", type=int, default=24)
    parser.add_argument(
        "--direct-action-world-model",
        action="store_true",
        help="use direct action updates instead of emitted rule-based world_model_rules",
    )
    parser.add_argument(
        "--world-model-ensemble",
        action="store_true",
        help="use a rule-based world-model ensemble fixture with controlled member disagreement",
    )
    parser.add_argument(
        "--world-model-ensemble-min-agreement",
        type=float,
        default=0.75,
        help="minimum ensemble prediction agreement required before transition verification can decide",
    )
    parser.add_argument(
        "--world-model-ensemble-strategy",
        choices=("label_stress", "policy_replay"),
        default="label_stress",
        help="controlled ensemble disagreement strategy",
    )
    parser.add_argument("--alphas", default="0.1")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--best-alpha", type=float, default=0.10)
    parser.add_argument("--verifier-signals", default=",".join(DEFAULT_VERIFIER_SIGNALS))
    parser.add_argument("--fusion-signals", default=",".join(DEFAULT_WORLD_MODEL_FUSION_SIGNALS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--geometry-signals", default=None)
    parser.add_argument("--uncertainty-signals", default=",".join(DEFAULT_WORLD_MODEL_UNCERTAINTY_SIGNALS))
    parser.add_argument("--geometry-method", default="mean_rank")
    parser.add_argument("--uncertainty-method", default="mean_rank")
    parser.add_argument("--geometry-fusion-methods", default=",".join(GEOMETRY_FUSION_METHODS))
    parser.add_argument("--min-world-model-confidence", type=float, default=0.0)
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--no-verify-manifest", action="store_true")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--registry-name", default="world-model-signal-calibration")
    parser.add_argument("--registry-version", default="0.1")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
