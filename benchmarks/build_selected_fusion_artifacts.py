"""Build calibrated fusion artifacts from a signal-selection report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

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
from eigentruth.json_utils import strict_json_dumps  # noqa: E402

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def build_selected_fusion_artifacts(
    selection_report: SignalSelectionReport,
    score_dumps: Mapping[str, str | Path],
    *,
    output_dir: str | Path,
    alpha: float | None = None,
    created_at: str | None = None,
    commit_sha: str | None = None,
    score_dump_cache: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one ``RankScoreFusionArtifact`` per selected run."""
    if not score_dumps:
        raise ValueError("score_dumps must be non-empty.")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache = {} if score_dump_cache is None else score_dump_cache
    artifacts: dict[str, str] = {}
    run_payloads = []
    for decision in selection_report.decisions:
        if decision.run_name not in score_dumps:
            raise ValueError(f"missing score dump for selected run {decision.run_name!r}.")
        score_path = Path(score_dumps[decision.run_name])
        dump = load_score_dump_columns(score_path, decision.selected_signals, cache=cache)
        artifact_alpha = float(alpha) if alpha is not None else float(decision.selected_metrics["alpha"])
        method = _artifact_method(decision.selected_method, decision.selected_signals)
        artifact = RankScoreFusionCalibrator(alpha=artifact_alpha, method=method).calibrate(
            labels=dump.labels,
            scores=dump.scores,
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
        artifacts[decision.run_name] = str(artifact_path)
        run_payloads.append({
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
        })
    return {
        "schema_version": 1,
        "workflow": "selected_fusion_artifact_build",
        "status": "complete",
        "selection_workflow": selection_report.workflow,
        "selection_status": selection_report.status,
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
    payload = build_selected_fusion_artifacts(
        selection,
        score_dumps,
        output_dir=args.output_dir,
        alpha=args.alpha,
        created_at=args.created_at,
        commit_sha=args.commit_sha,
    )
    payload["selection_report_path"] = str(selection_path)
    payload = json.loads(strict_json_dumps(payload))
    if args.json is not None:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(strict_json_dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--commit-sha", default=None)
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
