"""Calibrate post-acquisition evidence policies from ProductTrace payloads.

This workflow turns saved product traces, optionally joined with
ProductFeedbackRecord JSONL labels, into EvidenceAcquisitionCalibrationRecord
rows and then builds a post-acquisition conformal calibration report/artifact.
It does not run models, verifiers, retrievers, or external services.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import planned_artifact_manifest_summary, strict_bool  # noqa: E402
from eigentruth.calibration import (  # noqa: E402
    EvidenceAcquisitionConformalCalibrator,
    audit_evidence_acquisition_anytime_risk,
    audit_evidence_acquisition_risk,
    evidence_acquisition_records_from_trace_feedback,
    evidence_acquisition_records_from_traces,
)
from eigentruth.control import load_feedback_jsonl  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class EvidenceAcquisitionTraceCalibrationConfig:
    """Configuration for trace-backed post-acquisition calibration."""

    trace_paths: Sequence[str | Path] = ()
    trace_jsonl_paths: Sequence[str | Path] = ()
    feedback_paths: Sequence[str | Path] = ()
    report_path: str | Path = "artifacts/evidence-acquisition-trace-calibration/report.json"
    artifact_path: str | Path | None = None
    records_jsonl_path: str | Path | None = None
    risk_monitor_path: str | Path | None = None
    anytime_risk_monitor_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    model_id: str = "product-trace-policy"
    model_revision: str | None = None
    target_layer: int = -1
    score_name: str = "post_acquisition_policy_score"
    post_score_name: str | None = None
    pre_score_name: str | None = None
    direction: str = "higher"
    alpha: float = 0.1
    risk_target_error_rate: float | None = None
    risk_monitor_alpha: float = 0.05
    risk_monitor_mode: str = "prefix"
    risk_monitor_schedule: str = "harmonic"
    risk_monitor_checkpoints: Sequence[int] = ()
    risk_monitor_bet_fractions: Sequence[float] = ()
    allow_unmatched_feedback: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    compact_json: bool = False

    def __post_init__(self) -> None:
        trace_paths = tuple(Path(path) for path in self.trace_paths)
        trace_jsonl_paths = tuple(Path(path) for path in self.trace_jsonl_paths)
        feedback_paths = tuple(Path(path) for path in self.feedback_paths)
        if not trace_paths and not trace_jsonl_paths:
            raise ValueError("at least one ProductTrace JSON or JSONL path is required.")
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        alpha = _finite_float(self.alpha, name="alpha")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        risk_monitor_alpha = _finite_float(self.risk_monitor_alpha, name="risk_monitor_alpha")
        if not (0.0 < risk_monitor_alpha < 1.0):
            raise ValueError("risk_monitor_alpha must be in (0, 1).")
        risk_target_error_rate = (
            None
            if self.risk_target_error_rate is None
            else _unit_interval_float(self.risk_target_error_rate, name="risk_target_error_rate")
        )
        risk_monitor_schedule = str(self.risk_monitor_schedule).strip().lower().replace("_", "-")
        if risk_monitor_schedule not in {"linear", "harmonic", "geometric"}:
            raise ValueError("risk_monitor_schedule must be one of: linear, harmonic, geometric.")
        risk_monitor_mode = str(self.risk_monitor_mode).strip().lower().replace("_", "-")
        if risk_monitor_mode not in {"prefix", "anytime", "both"}:
            raise ValueError("risk_monitor_mode must be one of: prefix, anytime, both.")
        risk_monitor_checkpoints = tuple(
            _positive_int(checkpoint, name="risk_monitor_checkpoint") for checkpoint in self.risk_monitor_checkpoints
        )
        risk_monitor_bet_fractions = tuple(
            _bet_fraction(value, name="risk_monitor_bet_fraction")
            for value in self.risk_monitor_bet_fractions
        )
        direction = str(self.direction).strip().lower()
        if direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        score_name = _non_empty_string(self.score_name, name="score_name")
        model_id = _non_empty_string(self.model_id, name="model_id")
        target_layer = _int_value(self.target_layer, name="target_layer")
        object.__setattr__(self, "trace_paths", trace_paths)
        object.__setattr__(self, "trace_jsonl_paths", trace_jsonl_paths)
        object.__setattr__(self, "feedback_paths", feedback_paths)
        object.__setattr__(self, "report_path", Path(self.report_path))
        if self.artifact_path is not None:
            object.__setattr__(self, "artifact_path", Path(self.artifact_path))
        if self.records_jsonl_path is not None:
            object.__setattr__(self, "records_jsonl_path", Path(self.records_jsonl_path))
        if self.risk_monitor_path is not None:
            object.__setattr__(self, "risk_monitor_path", Path(self.risk_monitor_path))
        if self.anytime_risk_monitor_path is not None:
            object.__setattr__(self, "anytime_risk_monitor_path", Path(self.anytime_risk_monitor_path))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "target_layer", target_layer)
        object.__setattr__(self, "score_name", score_name)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "risk_target_error_rate", risk_target_error_rate)
        object.__setattr__(self, "risk_monitor_alpha", risk_monitor_alpha)
        object.__setattr__(self, "risk_monitor_mode", risk_monitor_mode)
        object.__setattr__(self, "risk_monitor_schedule", risk_monitor_schedule)
        object.__setattr__(self, "risk_monitor_checkpoints", risk_monitor_checkpoints)
        object.__setattr__(self, "risk_monitor_bet_fractions", risk_monitor_bet_fractions)
        object.__setattr__(
            self,
            "allow_unmatched_feedback",
            strict_bool(self.allow_unmatched_feedback, name="allow_unmatched_feedback"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))

    @property
    def resolved_artifact_path(self) -> Path:
        """Return the saved CalibrationArtifact path."""
        if self.artifact_path is not None:
            return Path(self.artifact_path)
        return Path(self.report_path).with_name("evidence-acquisition-calibration-artifact.json")

    @property
    def resolved_risk_monitor_path(self) -> Path | None:
        """Return the optional post-acquisition feedback risk monitor path."""
        if self.risk_target_error_rate is None or self.risk_monitor_mode not in {"prefix", "both"}:
            return None
        if self.risk_monitor_path is not None:
            return Path(self.risk_monitor_path)
        return Path(self.report_path).with_name("evidence-acquisition-risk-monitor.json")

    @property
    def resolved_anytime_risk_monitor_path(self) -> Path | None:
        """Return the optional anytime post-acquisition feedback risk monitor path."""
        if self.risk_target_error_rate is None or self.risk_monitor_mode not in {"anytime", "both"}:
            return None
        if self.anytime_risk_monitor_path is not None:
            return Path(self.anytime_risk_monitor_path)
        return Path(self.report_path).with_name("evidence-acquisition-anytime-risk-monitor.json")

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        """Return the artifact manifest path."""
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return Path(self.report_path).with_name("evidence-acquisition-calibration-manifest.json")


def build_evidence_acquisition_trace_calibration(
    config: EvidenceAcquisitionTraceCalibrationConfig,
) -> dict[str, Any]:
    """Build a post-acquisition calibration report/artifact from traces."""
    traces = tuple(_iter_traces(config))
    feedback = load_feedback_jsonl(config.feedback_paths) if config.feedback_paths else ()
    records = (
        evidence_acquisition_records_from_trace_feedback(
            traces,
            feedback,
            score_name=config.score_name,
            post_score_name=config.post_score_name,
            pre_score_name=config.pre_score_name,
            allow_unmatched=config.allow_unmatched_feedback,
        )
        if feedback
        else evidence_acquisition_records_from_traces(
            traces,
            score_name=config.score_name,
            post_score_name=config.post_score_name,
            pre_score_name=config.pre_score_name,
        )
    )
    calibrator = EvidenceAcquisitionConformalCalibrator(
        alpha=config.alpha,
        score_name=config.score_name,
        direction=config.direction,
    )
    calibration_metadata = {
        "workflow": "evidence_acquisition_trace_calibration",
        "trace_count": len(traces),
        "feedback_count": len(feedback),
        "uses_feedback": bool(feedback),
        "allow_unmatched_feedback": config.allow_unmatched_feedback,
        **dict(config.metadata),
    }
    result = calibrator.calibrate(
        model_id=config.model_id,
        model_revision=config.model_revision,
        target_layer=config.target_layer,
        records=records,
        calibration_dataset_metadata=calibration_metadata,
    )

    config.resolved_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    result.artifact.save_json(config.resolved_artifact_path)
    if config.records_jsonl_path is not None:
        _write_records_jsonl(config.records_jsonl_path, records)

    risk_monitor = None
    anytime_risk_monitor = None
    risk_monitor_path = config.resolved_risk_monitor_path
    anytime_risk_monitor_path = config.resolved_anytime_risk_monitor_path
    if risk_monitor_path is not None:
        assert config.risk_target_error_rate is not None
        risk_monitor = audit_evidence_acquisition_risk(
            records,
            threshold=result.report.post_threshold,
            target_error_rate=config.risk_target_error_rate,
            monitor_alpha=config.risk_monitor_alpha,
            direction=config.direction,
            schedule=config.risk_monitor_schedule,
            score_name=config.score_name,
            checkpoints=None if not config.risk_monitor_checkpoints else config.risk_monitor_checkpoints,
            metadata={
                "workflow": "evidence_acquisition_trace_calibration",
                "calibration_report": str(config.report_path),
                "calibration_artifact": str(config.resolved_artifact_path),
                **dict(config.metadata),
            },
        )
        _write_json(risk_monitor_path, risk_monitor.to_dict(), compact=config.compact_json)
    if anytime_risk_monitor_path is not None:
        assert config.risk_target_error_rate is not None
        anytime_risk_monitor = audit_evidence_acquisition_anytime_risk(
            records,
            threshold=result.report.post_threshold,
            target_error_rate=config.risk_target_error_rate,
            monitor_alpha=config.risk_monitor_alpha,
            direction=config.direction,
            score_name=config.score_name,
            bet_fractions=None
            if not config.risk_monitor_bet_fractions
            else config.risk_monitor_bet_fractions,
            metadata={
                "workflow": "evidence_acquisition_trace_calibration",
                "calibration_report": str(config.report_path),
                "calibration_artifact": str(config.resolved_artifact_path),
                **dict(config.metadata),
            },
        )
        _write_json(anytime_risk_monitor_path, anytime_risk_monitor.to_dict(), compact=config.compact_json)

    summary = _summary(result.report.to_dict())
    if risk_monitor is not None:
        summary.update(_risk_monitor_summary(risk_monitor.to_dict()))
    if anytime_risk_monitor is not None:
        summary.update(_anytime_risk_monitor_summary(anytime_risk_monitor.to_dict()))
    blocked = (risk_monitor is not None and not risk_monitor.passed) or (
        anytime_risk_monitor is not None and not anytime_risk_monitor.passed
    )

    report_payload = {
        "schema_version": 1,
        "workflow": "evidence_acquisition_trace_calibration",
        "status": "blocked" if blocked else "passed",
        "summary": summary,
        "config": _config_payload(config),
        "calibration_report": result.report.to_dict(),
        "risk_monitor_report": None if risk_monitor is None else risk_monitor.to_dict(),
        "anytime_risk_monitor_report": None
        if anytime_risk_monitor is None
        else anytime_risk_monitor.to_dict(),
        "paths": {
            "report": str(config.report_path),
            "calibration_artifact": str(config.resolved_artifact_path),
            "records_jsonl": None if config.records_jsonl_path is None else str(config.records_jsonl_path),
            "risk_monitor_report": None if risk_monitor_path is None else str(risk_monitor_path),
            "anytime_risk_monitor_report": None
            if anytime_risk_monitor_path is None
            else str(anytime_risk_monitor_path),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
        },
        "artifact_manifest_summary": _artifact_manifest_summary(config),
        "metadata": dict(config.metadata),
    }
    _write_json(config.report_path, report_payload, compact=config.compact_json)
    _write_artifact_manifest(config, report_payload)
    if config.registry_path is not None:
        _record_registry(config, report_payload)
    return report_payload


def _iter_traces(config: EvidenceAcquisitionTraceCalibrationConfig) -> Iterator[Mapping[str, Any]]:
    for path in config.trace_paths:
        yield _load_trace_json(path)
    for path in config.trace_jsonl_paths:
        yield from _iter_trace_jsonl(path)


def _load_trace_json(path: str | Path) -> Mapping[str, Any]:
    trace_path = Path(path)
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"ProductTrace JSON must contain an object: {trace_path}")
    return _trace_payload(payload, source=str(trace_path))


def _iter_trace_jsonl(path: str | Path) -> Iterator[Mapping[str, Any]]:
    trace_path = Path(path)
    with trace_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"ProductTrace JSONL row must be an object: {trace_path}:{line_number}")
            yield _trace_payload(payload, source=f"{trace_path}:{line_number}")


def _trace_payload(payload: Mapping[str, Any], *, source: str) -> Mapping[str, Any]:
    trace = payload
    for key in ("trace", "product_trace"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            trace = nested
            break
    if trace.get("request_id") is None and trace.get("risk_decision") is None:
        raise ValueError(f"payload does not look like a ProductTrace: {source}")
    return trace


def _write_records_jsonl(
    path: str | Path,
    records: Sequence[Any],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(strict_json_dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output_path.write_text(text, encoding="utf-8")


def _write_artifact_manifest(
    config: EvidenceAcquisitionTraceCalibrationConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = build_artifact_manifest(
        _artifact_paths(config),
        root=config.resolved_artifact_manifest_path.parent,
        metadata={
            "runner": "calibrate_evidence_acquisition_from_traces",
            "workflow": report.get("workflow"),
            "status": report.get("status"),
            "n_records": _nested(report, "summary", "n_records"),
            "n_acquired": _nested(report, "summary", "n_acquired"),
            "post_threshold": _nested(report, "summary", "post_threshold"),
            "risk_monitor_passed": _nested(report, "summary", "risk_monitor_passed"),
            "risk_target_error_rate": _nested(report, "summary", "risk_target_error_rate"),
            "risk_first_failed_checkpoint": _nested(report, "summary", "risk_first_failed_checkpoint"),
            "anytime_risk_monitor_passed": _nested(report, "summary", "anytime_risk_monitor_passed"),
            "anytime_risk_first_alarm_record_index": _nested(
                report,
                "summary",
                "anytime_risk_first_alarm_record_index",
            ),
            **dict(config.metadata),
        },
    )
    _write_json(config.resolved_artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _artifact_paths(config: EvidenceAcquisitionTraceCalibrationConfig) -> dict[str, Path]:
    artifacts: dict[str, Path] = {
        "evidence_acquisition_trace_calibration_report": Path(config.report_path),
        "evidence_acquisition_calibration_artifact": config.resolved_artifact_path,
        **{f"trace_{idx}": path for idx, path in enumerate(config.trace_paths, start=1)},
        **{f"trace_jsonl_{idx}": path for idx, path in enumerate(config.trace_jsonl_paths, start=1)},
        **{f"feedback_{idx}": path for idx, path in enumerate(config.feedback_paths, start=1)},
    }
    if config.records_jsonl_path is not None:
        artifacts["evidence_acquisition_calibration_records"] = Path(config.records_jsonl_path)
    risk_monitor_path = config.resolved_risk_monitor_path
    if risk_monitor_path is not None:
        artifacts["evidence_acquisition_risk_monitor_report"] = risk_monitor_path
    anytime_risk_monitor_path = config.resolved_anytime_risk_monitor_path
    if anytime_risk_monitor_path is not None:
        artifacts["evidence_acquisition_anytime_risk_monitor_report"] = anytime_risk_monitor_path
    return artifacts


def _artifact_manifest_summary(config: EvidenceAcquisitionTraceCalibrationConfig) -> dict[str, int]:
    return planned_artifact_manifest_summary(
        _artifact_paths(config),
        assume_file_paths=(
            config.report_path,
            config.resolved_artifact_path,
            *(() if config.records_jsonl_path is None else (config.records_jsonl_path,)),
            *(() if config.resolved_risk_monitor_path is None else (config.resolved_risk_monitor_path,)),
            *(
                ()
                if config.resolved_anytime_risk_monitor_path is None
                else (config.resolved_anytime_risk_monitor_path,)
            ),
        ),
    )


def _record_registry(
    config: EvidenceAcquisitionTraceCalibrationConfig,
    report: Mapping[str, Any],
) -> None:
    assert config.registry_path is not None
    assert config.name is not None
    assert config.version is not None
    registry = ArtifactRegistry.load_json(config.registry_path)
    metadata = {
        "workflow": "evidence_acquisition_trace_calibration",
        "status": report.get("status"),
        "n_records": _nested(report, "summary", "n_records"),
        "n_acquired": _nested(report, "summary", "n_acquired"),
        "acquisition_rate": _nested(report, "summary", "acquisition_rate"),
        "post_threshold": _nested(report, "summary", "post_threshold"),
        "naive_pre_threshold": _nested(report, "summary", "naive_pre_threshold"),
        "selective_accuracy_delta": _nested(report, "summary", "selective_accuracy_delta"),
        "risk_monitor_passed": _nested(report, "summary", "risk_monitor_passed"),
        "risk_target_error_rate": _nested(report, "summary", "risk_target_error_rate"),
        "risk_first_failed_checkpoint": _nested(report, "summary", "risk_first_failed_checkpoint"),
        "anytime_risk_monitor_passed": _nested(report, "summary", "anytime_risk_monitor_passed"),
        "anytime_risk_first_alarm_record_index": _nested(
            report,
            "summary",
            "anytime_risk_first_alarm_record_index",
        ),
        "calibration_artifact": str(config.resolved_artifact_path),
        "risk_monitor_report": None
        if config.resolved_risk_monitor_path is None
        else str(config.resolved_risk_monitor_path),
        "anytime_risk_monitor_report": None
        if config.resolved_anytime_risk_monitor_path is None
        else str(config.resolved_anytime_risk_monitor_path),
        "artifact_manifest": str(config.resolved_artifact_manifest_path),
        **dict(config.metadata),
    }
    registry.record_calibration_report(
        name=config.name,
        version=config.version,
        path=config.report_path,
        metadata=metadata,
    )
    registry.record_calibration_artifact(
        name=config.name,
        version=config.version,
        path=config.resolved_artifact_path,
        metadata=metadata,
    )
    if config.resolved_risk_monitor_path is not None:
        registry.record_report(
            name=config.name,
            version=config.version,
            path=config.resolved_risk_monitor_path,
            metadata={
                **metadata,
                "report_kind": "evidence_acquisition_risk_monitor",
            },
        )
    if config.resolved_anytime_risk_monitor_path is not None:
        registry.record_artifact(
            name=config.name,
            version=config.version,
            artifact_type="evidence_acquisition_anytime_risk_monitor_report",
            path=config.resolved_anytime_risk_monitor_path,
            metadata={
                **metadata,
                "report_kind": "evidence_acquisition_anytime_risk_monitor",
            },
        )
    registry.save_json()


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    deltas = report.get("deltas", {})
    if not isinstance(deltas, Mapping):
        deltas = {}
    post_report = report.get("post_acquisition_report", {})
    if not isinstance(post_report, Mapping):
        post_report = {}
    return {
        "n_records": report.get("n_records"),
        "n_correct": report.get("n_correct"),
        "n_acquired": report.get("n_acquired"),
        "n_answered": report.get("n_answered"),
        "n_abstained": report.get("n_abstained"),
        "acquisition_rate": report.get("acquisition_rate"),
        "post_threshold": report.get("post_threshold"),
        "naive_pre_threshold": report.get("naive_pre_threshold"),
        "post_selective_accuracy": post_report.get("empirical_selective_accuracy"),
        "post_participation_rate": post_report.get("empirical_participation_rate"),
        "selective_accuracy_delta": deltas.get("selective_accuracy"),
        "participation_rate_delta": deltas.get("participation_rate"),
        "conditional_correctness_lower_bound_delta": deltas.get("conditional_correctness_lower_bound"),
    }


def _risk_monitor_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "risk_monitor_passed": report.get("passed"),
        "risk_target_error_rate": report.get("target_error_rate"),
        "risk_monitor_alpha": report.get("monitor_alpha"),
        "risk_monitor_schedule": report.get("schedule"),
        "risk_first_failed_checkpoint": report.get("first_failed_checkpoint"),
        "risk_max_accepted_error_upper_bound": report.get("max_accepted_error_upper_bound"),
        "risk_blocking_reason_count": len(report.get("blocking_reasons", ())),
    }


def _anytime_risk_monitor_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "anytime_risk_monitor_passed": report.get("passed"),
        "anytime_risk_target_error_rate": report.get("target_error_rate"),
        "anytime_risk_monitor_alpha": report.get("monitor_alpha"),
        "anytime_risk_e_value": report.get("e_value"),
        "anytime_risk_alarm_threshold": report.get("alarm_threshold"),
        "anytime_risk_first_alarm_record_index": report.get("first_alarm_record_index"),
        "anytime_risk_first_alarm_accepted_index": report.get("first_alarm_accepted_index"),
        "anytime_risk_accepted_error_rate": report.get("accepted_error_rate"),
        "anytime_risk_blocking_reason_count": len(report.get("blocking_reasons", ())),
    }


def _config_payload(config: EvidenceAcquisitionTraceCalibrationConfig) -> dict[str, Any]:
    return {
        "trace_paths": tuple(str(path) for path in config.trace_paths),
        "trace_jsonl_paths": tuple(str(path) for path in config.trace_jsonl_paths),
        "feedback_paths": tuple(str(path) for path in config.feedback_paths),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "target_layer": config.target_layer,
        "score_name": config.score_name,
        "post_score_name": config.post_score_name,
        "pre_score_name": config.pre_score_name,
        "direction": config.direction,
        "alpha": config.alpha,
        "risk_monitor_path": None
        if config.resolved_risk_monitor_path is None
        else str(config.resolved_risk_monitor_path),
        "risk_target_error_rate": config.risk_target_error_rate,
        "risk_monitor_alpha": config.risk_monitor_alpha,
        "risk_monitor_mode": config.risk_monitor_mode,
        "risk_monitor_schedule": config.risk_monitor_schedule,
        "risk_monitor_checkpoints": config.risk_monitor_checkpoints,
        "anytime_risk_monitor_path": None
        if config.resolved_anytime_risk_monitor_path is None
        else str(config.resolved_anytime_risk_monitor_path),
        "risk_monitor_bet_fractions": config.risk_monitor_bet_fractions,
        "allow_unmatched_feedback": config.allow_unmatched_feedback,
    }


def _trace_paths_from_args(args: argparse.Namespace) -> tuple[Path, ...]:
    paths = [Path(path) for path in args.trace]
    for pattern in args.trace_glob:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
    return tuple(paths)


def _metadata_from_args(items: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--metadata entries must use key=value.")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError("--metadata keys must be non-empty.")
        metadata[key] = value
    return metadata


def _config_from_args(args: argparse.Namespace) -> EvidenceAcquisitionTraceCalibrationConfig:
    return EvidenceAcquisitionTraceCalibrationConfig(
        trace_paths=_trace_paths_from_args(args),
        trace_jsonl_paths=tuple(Path(path) for path in args.trace_jsonl),
        feedback_paths=tuple(Path(path) for path in args.feedback_jsonl),
        report_path=Path(args.json),
        artifact_path=None if args.artifact_json is None else Path(args.artifact_json),
        records_jsonl_path=None if args.records_jsonl is None else Path(args.records_jsonl),
        risk_monitor_path=None if args.risk_monitor_json is None else Path(args.risk_monitor_json),
        anytime_risk_monitor_path=None
        if args.anytime_risk_monitor_json is None
        else Path(args.anytime_risk_monitor_json),
        artifact_manifest_path=None if args.artifact_manifest is None else Path(args.artifact_manifest),
        registry_path=None if args.registry is None else Path(args.registry),
        name=args.name,
        version=args.version,
        model_id=args.model_id,
        model_revision=args.model_revision,
        target_layer=args.target_layer,
        score_name=args.score_name,
        post_score_name=args.post_score_name,
        pre_score_name=args.pre_score_name,
        direction=args.direction,
        alpha=args.alpha,
        risk_target_error_rate=args.risk_target_error_rate,
        risk_monitor_alpha=args.risk_monitor_alpha,
        risk_monitor_mode=args.risk_monitor_mode,
        risk_monitor_schedule=args.risk_monitor_schedule,
        risk_monitor_checkpoints=tuple(args.risk_monitor_checkpoint),
        risk_monitor_bet_fractions=tuple(args.risk_monitor_bet_fraction),
        allow_unmatched_feedback=bool(args.allow_unmatched_feedback),
        metadata=_metadata_from_args(args.metadata),
        compact_json=bool(args.compact_json),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI args."""
    config = _config_from_args(args)
    report = build_evidence_acquisition_trace_calibration(config)
    print(
        "evidence_acquisition_trace_calibration="
        f"{report['status']} records={report['summary']['n_records']} "
        f"acquired={report['summary']['n_acquired']} "
        f"post_threshold={report['summary']['post_threshold']} "
        f"risk_monitor_passed={report['summary'].get('risk_monitor_passed')} "
        f"anytime_risk_monitor_passed={report['summary'].get('anytime_risk_monitor_passed')}"
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a post-acquisition evidence policy from ProductTrace JSON/JSONL"
    )
    parser.add_argument("--trace", action="append", default=[], help="ProductTrace JSON path; repeatable")
    parser.add_argument("--trace-glob", action="append", default=[], help="glob for ProductTrace JSON files")
    parser.add_argument("--trace-jsonl", action="append", default=[], help="ProductTrace JSONL path; repeatable")
    parser.add_argument(
        "--feedback-jsonl", action="append", default=[], help="optional ProductFeedbackRecord JSONL path; repeatable"
    )
    parser.add_argument("--json", required=True, help="output calibration report JSON path")
    parser.add_argument("--artifact-json", default=None, help="output CalibrationArtifact JSON path")
    parser.add_argument("--records-jsonl", default=None, help="optional extracted calibration records JSONL path")
    parser.add_argument("--risk-monitor-json", default=None, help="optional feedback risk monitor report JSON path")
    parser.add_argument(
        "--anytime-risk-monitor-json",
        default=None,
        help="optional anytime feedback risk monitor report JSON path",
    )
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--model-id", default="product-trace-policy")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--target-layer", type=int, default=-1)
    parser.add_argument("--score-name", default="post_acquisition_policy_score")
    parser.add_argument("--post-score-name", default=None)
    parser.add_argument("--pre-score-name", default=None)
    parser.add_argument("--direction", choices=("higher", "lower"), default="higher")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--risk-target-error-rate",
        type=float,
        default=None,
        help="enable fixed-threshold feedback risk monitoring with this target accepted-error rate",
    )
    parser.add_argument("--risk-monitor-alpha", type=float, default=0.05)
    parser.add_argument(
        "--risk-monitor-mode",
        choices=("prefix", "anytime", "both"),
        default="prefix",
        help="risk monitor family to emit when --risk-target-error-rate is set",
    )
    parser.add_argument(
        "--risk-monitor-schedule",
        choices=("linear", "harmonic", "geometric"),
        default="harmonic",
    )
    parser.add_argument(
        "--risk-monitor-checkpoint",
        action="append",
        type=int,
        default=[],
        help="prefix checkpoint for feedback risk monitoring; repeatable; defaults to every prefix",
    )
    parser.add_argument(
        "--risk-monitor-bet-fraction",
        action="append",
        type=float,
        default=[],
        help="bet fraction for anytime risk monitoring; repeatable; defaults to a conservative grid",
    )
    parser.add_argument("--allow-unmatched-feedback", action="store_true")
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="metadata key=value pair to include in report and registry; repeatable",
    )
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    run(parser.parse_args(argv))


def _nested(payload: Mapping[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite, not bool.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _unit_interval_float(value: Any, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if not (0.0 <= number <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")
    return number


def _bet_fraction(value: Any, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if not (0.0 < number <= 1.0):
        raise ValueError(f"{name} must be in (0, 1].")
    return number


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, not bool.")
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not math.isfinite(as_float) or not as_float.is_integer():
        raise ValueError(f"{name} must be a positive integer.")
    number = int(as_float)
    if number < 1:
        raise ValueError(f"{name} must be positive.")
    return number


def _int_value(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool.")
    return int(value)


def _non_empty_string(value: Any, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


if __name__ == "__main__":
    main()
