"""Build calibrated fusion artifacts from a signal-selection report."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.calibration import RankScoreFusionCalibrator  # noqa: E402
from eigentruth.eval import (  # noqa: E402
    RANK_SCORE_FUSION_METHODS,
    SignalSelectionReport,
    load_score_dump_columns,
    score_dump_cache_summary,
    score_dump_file_metadata,
)
from eigentruth.eval.metrics import confidence_error_report, roc_auc  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_TOLERANCE = 0.03
_LOWER_IS_MORE_CONFIDENT_SIGNALS = {
    "nll_answer",
    "first_token_entropy",
    "inside_eigenscore",
    "inside_semantic_entropy",
    "inside_embedding_entropy",
    "inside_semantic_energy",
}


def build_selected_fusion_artifacts(
    selection_report: SignalSelectionReport,
    score_dumps: Mapping[str, str | Path],
    *,
    output_dir: str | Path,
    alpha: float | None = None,
    confidence_signal: str | None = None,
    confidence_direction: str | None = None,
    confidence_top_fraction: float = 0.25,
    max_high_confidence_accepted_false_rate: float = 0.0,
    created_at: str | None = None,
    commit_sha: str | None = None,
    score_dump_cache: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one ``RankScoreFusionArtifact`` per selected run."""
    if not score_dumps:
        raise ValueError("score_dumps must be non-empty.")
    confidence_signal = None if confidence_signal is None else str(confidence_signal).strip() or None
    resolved_confidence_direction = _resolve_confidence_direction(
        confidence_signal,
        confidence_direction,
    )
    confidence_top_fraction = _fraction(confidence_top_fraction, name="confidence_top_fraction")
    max_high_confidence_accepted_false_rate = _bounded_rate(
        max_high_confidence_accepted_false_rate,
        name="max_high_confidence_accepted_false_rate",
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache = {} if score_dump_cache is None else score_dump_cache
    artifacts: dict[str, str] = {}
    run_payloads = []
    for decision in selection_report.decisions:
        if decision.run_name not in score_dumps:
            raise ValueError(f"missing score dump for selected run {decision.run_name!r}.")
        score_path = Path(score_dumps[decision.run_name])
        load_signals = _dedupe_signals(
            decision.selected_signals,
            () if confidence_signal is None else (confidence_signal,),
        )
        dump = load_score_dump_columns(score_path, load_signals, cache=cache)
        artifact_alpha = float(alpha) if alpha is not None else float(decision.selected_metrics["alpha"])
        method = _artifact_method(decision.selected_method, decision.selected_signals)
        selected_scores = {signal: dump.scores[signal] for signal in decision.selected_signals}
        artifact = RankScoreFusionCalibrator(alpha=artifact_alpha, method=method).calibrate(
            labels=dump.labels,
            scores=selected_scores,
            directions=decision.directions,
            model_id=_optional_str(dump.config.get("model")),
            target_layer=_optional_int(dump.config.get("layer")),
            score_dump_metadata={
                "source_scores_path": str(score_path),
                "source_format": dump.source_format,
                "summary": dict(dump.summary),
                "file_metadata": score_dump_file_metadata(score_path, cache=cache),
                "selection_decision": decision.to_dict(),
                "selection_policy": selection_report.policy.to_dict(),
                "selection_report_metadata": dict(selection_report.metadata),
                "source_selected_method": decision.selected_method,
                "artifact_method": method,
            },
            created_at=created_at,
            commit_sha=commit_sha,
        )
        artifact_path = output_root / f"{_safe_name(decision.run_name)}-selected-fusion-artifact.json"
        artifact.save_json(artifact_path)
        artifact_metrics = _artifact_metrics(
            artifact,
            labels=dump.labels,
            scores=selected_scores,
        )
        confidence_report: dict[str, Any] | None = None
        release_gate: dict[str, Any] | None = None
        if confidence_signal is not None:
            assert resolved_confidence_direction is not None
            confidence_scores = dump.scores[confidence_signal]
            confidence_report = confidence_error_report(
                artifact.score(selected_scores),
                dump.labels,
                artifact.threshold,
                confidence_scores,
                anomaly_direction="higher",
                confidence_direction=resolved_confidence_direction,
                confidence_top_fraction=confidence_top_fraction,
            )
            release_gate = _selected_artifact_release_gate(
                artifact_metrics,
                alpha=artifact_alpha,
                confidence_report=confidence_report,
                max_high_confidence_accepted_false_rate=max_high_confidence_accepted_false_rate,
            )
        artifacts[decision.run_name] = str(artifact_path)
        run_payload = {
            "run_name": decision.run_name,
            "artifact_path": str(artifact_path),
            "score_dump_path": str(score_path),
            "selected_candidate": decision.selected_candidate,
            "selected_signals": list(decision.selected_signals),
            "selected_method": decision.selected_method,
            "artifact_method": method,
            "tracked_signal": decision.tracked_signal,
            "tracked_signal_enabled": decision.tracked_signal_enabled,
            "directions": dict(decision.directions),
            "threshold": artifact.threshold,
            "conformal_alpha": artifact.conformal_alpha,
            "calibration_size": artifact.calibration_size(),
            "selected_metrics": dict(decision.selected_metrics),
            "artifact_metrics": artifact_metrics,
        }
        if confidence_report is not None and release_gate is not None:
            run_payload["confidence_error_at_artifact_threshold"] = confidence_report
            run_payload["release_gate"] = release_gate
        run_payloads.append(run_payload)
    return {
        "schema_version": 1,
        "workflow": "selected_fusion_artifact_build",
        "status": "complete",
        "selection_workflow": selection_report.workflow,
        "selection_status": selection_report.status,
        "confidence_audit": {
            "enabled": confidence_signal is not None,
            "confidence_signal": confidence_signal,
            "confidence_direction": resolved_confidence_direction,
            "confidence_top_fraction": confidence_top_fraction,
            "max_high_confidence_accepted_false_rate": max_high_confidence_accepted_false_rate,
        },
        "policy": selection_report.policy.to_dict(),
        "output_dir": str(output_root),
        "artifacts": artifacts,
        "runs": run_payloads,
        "score_dump_cache": score_dump_cache_summary(cache),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI entry point with a testable args namespace."""
    selection_path = Path(args.selection_report)
    selection = SignalSelectionReport.load_json(selection_path)
    score_dumps = dict(_parse_named_path(value) for value in args.scores)
    metadata = _parse_metadata(getattr(args, "metadata", ()))
    manifest_path = (
        None
        if getattr(args, "artifact_manifest", None) is None
        else Path(args.artifact_manifest)
    )
    registry_path = None if getattr(args, "registry", None) is None else Path(args.registry)
    if registry_path is not None and (
        not getattr(args, "name", None) or not getattr(args, "version", None)
    ):
        raise ValueError("registry requires name and version.")
    report_path = None if args.json is None else Path(args.json)
    if report_path is None and (manifest_path is not None or registry_path is not None):
        report_path = Path(args.output_dir) / "selected-fusion-artifact-build-report.json"
    payload = build_selected_fusion_artifacts(
        selection,
        score_dumps,
        output_dir=args.output_dir,
        alpha=args.alpha,
        confidence_signal=getattr(args, "confidence_signal", None),
        confidence_direction=getattr(args, "confidence_direction", None),
        confidence_top_fraction=float(getattr(args, "confidence_top_fraction", 0.25)),
        max_high_confidence_accepted_false_rate=float(
            getattr(args, "max_high_confidence_accepted_false_rate", 0.0)
        ),
        created_at=args.created_at,
        commit_sha=args.commit_sha,
    )
    payload["selection_report_path"] = str(selection_path)
    if report_path is not None:
        payload["build_report_path"] = str(report_path)
    if manifest_path is not None:
        payload["artifact_manifest_path"] = str(manifest_path)
    if metadata:
        payload["metadata"] = metadata
    payload = json.loads(strict_json_dumps(payload))
    if report_path is not None:
        _write_json(report_path, payload)
    if manifest_path is not None:
        assert report_path is not None
        _write_json(
            manifest_path,
            build_artifact_manifest(
                _manifest_artifacts(
                    payload,
                    report_path=report_path,
                    selection_path=selection_path,
                    score_dumps=score_dumps,
                ),
                root=manifest_path.parent,
                metadata=_summary_metadata(payload, artifact_manifest_path=manifest_path, metadata=metadata),
            ),
        )
    if registry_path is not None:
        assert report_path is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=str(args.name),
            version=str(args.version),
            path=report_path,
            metadata=_summary_metadata(
                payload,
                artifact_manifest_path=manifest_path,
                metadata=metadata,
            ),
        ).save_json()
    if not args.quiet:
        for run_payload in payload["runs"]:
            print(
                f"{run_payload['run_name']}: "
                f"{run_payload['artifact_path']} "
                f"signals={','.join(run_payload['selected_signals'])} "
                f"threshold={run_payload['threshold']}"
            )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build calibrated fusion artifacts from a signal-selection report")
    parser.add_argument("--selection-report", required=True, help="signal-selection report JSON path")
    parser.add_argument("--scores", action="append", required=True,
                        help="score dump path, named as run=path; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None, help="optional compact build report path")
    parser.add_argument("--alpha", type=float, default=None,
                        help="optional artifact conformal alpha; defaults to each decision alpha")
    parser.add_argument("--confidence-signal", default=None,
                        help="optional score used to audit high-confidence accepted false errors")
    parser.add_argument("--confidence-direction", choices=("higher", "lower"), default=None,
                        help="whether higher or lower confidence-signal values mean more confidence")
    parser.add_argument("--confidence-top-fraction", type=float, default=0.25,
                        help="fraction of records treated as high confidence for the release gate")
    parser.add_argument("--max-high-confidence-accepted-false-rate", type=float, default=0.0,
                        help="maximum high-confidence accepted false rate allowed by the release gate")
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--commit-sha", default=None)
    parser.add_argument("--artifact-manifest", default=None,
                        help="optional artifact manifest path for the build report, artifacts, and inputs")
    parser.add_argument("--registry", default=None,
                        help="optional ArtifactRegistry JSON path for recording the build report")
    parser.add_argument("--name", default=None,
                        help="registry report name; required with --registry")
    parser.add_argument("--version", default=None,
                        help="registry report version; required with --registry")
    parser.add_argument("--metadata", action="append", default=[],
                        help="metadata key=value pair to include in report, manifest, and registry; repeatable")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    run(args)


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("scores must be provided as run=path.")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("scores run name cannot be empty.")
    return name, Path(path)


def _artifact_method(method: str, signals: Sequence[str]) -> str:
    if method in RANK_SCORE_FUSION_METHODS:
        return method
    if method == "native" and len(tuple(signals)) == 1:
        return "max_rank"
    raise ValueError(f"selected method {method!r} is not supported by RankScoreFusionArtifact.")


def _dedupe_signals(*groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(signal for group in groups for signal in group))


def _resolve_confidence_direction(signal: str | None, direction: str | None) -> str | None:
    if signal is None:
        return None
    if direction is not None:
        if direction not in {"higher", "lower"}:
            raise ValueError("confidence_direction must be 'higher' or 'lower'.")
        return direction
    return "lower" if signal in _LOWER_IS_MORE_CONFIDENT_SIGNALS else "higher"


def _artifact_metrics(
    artifact,
    *,
    labels: Sequence[int],
    scores: Mapping[str, Any],
) -> dict[str, Any]:
    labels_t = torch.as_tensor(labels, dtype=torch.int64).flatten()
    fused = artifact.score(scores)
    flags = artifact.flags(scores)
    normal = labels_t == 0
    anomalous = labels_t == 1
    false_alarm = 0.0 if int(normal.sum().item()) == 0 else float(flags[normal].double().mean().item())
    detection = 0.0 if int(anomalous.sum().item()) == 0 else float(flags[anomalous].double().mean().item())
    return {
        "auroc": roc_auc(fused, labels_t),
        "false_alarm": false_alarm,
        "detection": detection,
        "coverage": 1.0 - false_alarm,
    }


def _selected_artifact_release_gate(
    artifact_metrics: Mapping[str, Any],
    *,
    alpha: float,
    confidence_report: Mapping[str, Any],
    max_high_confidence_accepted_false_rate: float,
) -> dict[str, Any]:
    reasons = []
    false_alarm = _optional_float(artifact_metrics.get("false_alarm"))
    false_alarm_pass = false_alarm is not None and false_alarm <= float(alpha) + _TOLERANCE
    if not false_alarm_pass:
        reasons.append("artifact false-alarm gate failed at selected alpha")
    accepted_false_rate = _nested_rate(confidence_report, "high_confidence_accepted_false_rate")
    accepted_false_count = int(confidence_report.get("n_high_confidence_accepted_false", 0))
    if accepted_false_rate is not None and accepted_false_rate > max_high_confidence_accepted_false_rate:
        reasons.append(
            "high-confidence accepted false rate "
            f"{accepted_false_rate:.6g} exceeds max {max_high_confidence_accepted_false_rate:.6g}"
        )
    elif accepted_false_rate is None and accepted_false_count > 0:
        reasons.append("high-confidence accepted false count is nonzero with undefined denominator")
    return {
        "status": "promote" if not reasons else "blocked",
        "alpha": float(alpha),
        "false_alarm_pass": false_alarm_pass,
        "false_alarm": false_alarm,
        "detection": _optional_float(artifact_metrics.get("detection")),
        "max_high_confidence_accepted_false_rate": float(max_high_confidence_accepted_false_rate),
        "high_confidence_accepted_false_rate": accepted_false_rate,
        "high_confidence_accepted_false_count": accepted_false_count,
        "high_confidence_accepted_count": int(confidence_report.get("n_high_confidence_accepted", 0)),
        "reasons": reasons,
    }


def _nested_rate(report: Mapping[str, Any], key: str) -> float | None:
    value = report.get(key)
    if isinstance(value, Mapping):
        value = value.get("estimate")
    return _optional_float(value)


def _manifest_artifacts(
    payload: Mapping[str, Any],
    *,
    report_path: Path,
    selection_path: Path,
    score_dumps: Mapping[str, Path],
) -> dict[str, Path]:
    artifacts: dict[str, Path] = {
        "selected_fusion_artifact_build_report": report_path,
        "selection_report": selection_path,
    }
    for run_payload in payload.get("runs", ()):
        if not isinstance(run_payload, Mapping):
            continue
        run_name = _safe_name(str(run_payload.get("run_name") or "run"))
        artifacts[f"selected_fusion_artifact.{run_name}"] = Path(str(run_payload["artifact_path"]))
        score_path = score_dumps.get(str(run_payload.get("run_name")))
        if score_path is not None:
            artifacts[f"score_dump.{run_name}"] = score_path
    return artifacts


def _summary_metadata(
    payload: Mapping[str, Any],
    *,
    artifact_manifest_path: Path | None,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    runs = tuple(item for item in payload.get("runs", ()) if isinstance(item, Mapping))
    gate_runs = tuple(item for item in runs if isinstance(item.get("release_gate"), Mapping))
    promoted_gates = sum(1 for item in gate_runs if item["release_gate"].get("status") == "promote")
    blocked_gates = sum(1 for item in gate_runs if item["release_gate"].get("status") == "blocked")
    confidence_audit = payload.get("confidence_audit")
    summary = {
        "workflow": payload.get("workflow"),
        "status": payload.get("status"),
        "selection_workflow": payload.get("selection_workflow"),
        "selection_status": payload.get("selection_status"),
        "run_count": len(runs),
        "artifact_count": len(payload.get("artifacts", {}) or {}),
        "confidence_audit_enabled": (
            confidence_audit.get("enabled") if isinstance(confidence_audit, Mapping) else None
        ),
        "confidence_signal": (
            confidence_audit.get("confidence_signal") if isinstance(confidence_audit, Mapping) else None
        ),
        "confidence_direction": (
            confidence_audit.get("confidence_direction") if isinstance(confidence_audit, Mapping) else None
        ),
        "release_gate_count": len(gate_runs),
        "release_gate_promote_count": promoted_gates,
        "release_gate_blocked_count": blocked_gates,
        "all_release_gates_promoted": (
            None if not gate_runs else promoted_gates == len(gate_runs)
        ),
        **dict(metadata),
    }
    if artifact_manifest_path is not None:
        summary["artifact_manifest"] = str(artifact_manifest_path)
    return summary


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(strict_json_dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_metadata(values: Sequence[str] | None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(f"metadata must be KEY=VALUE, got {value!r}.")
        key, item = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("metadata key cannot be empty.")
        metadata[key] = item
    return metadata


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _fraction(value: Any, *, name: str) -> float:
    numeric = _required_float(value, name=name)
    if not (0.0 < numeric <= 1.0):
        raise ValueError(f"{name} must be in (0, 1].")
    return numeric


def _bounded_rate(value: Any, *, name: str) -> float:
    numeric = _required_float(value, name=name)
    if not (0.0 <= numeric <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")
    return numeric


def _required_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def _safe_name(value: str) -> str:
    safe = _SAFE_NAME_RE.sub("-", value.strip()).strip("-._")
    if not safe:
        raise ValueError("run name cannot be converted to a safe file name.")
    return safe


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    main()
