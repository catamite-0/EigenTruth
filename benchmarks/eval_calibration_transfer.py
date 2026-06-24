"""Evaluate whether calibration thresholds transfer across score dumps.

This is a post-processing helper: it applies one or more saved
``CalibrationArtifact`` thresholds to one or more ``eval_truthfulqa.py
--dump-scores`` files. It does not load models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.eval.metrics import selective_classification_report
from eigentruth.eval.score_dump import ScoreDump, load_score_dump, score_dump_file_metadata

DEFAULT_TOLERANCE = 0.03


def _parse_named_path(value: str, *, option: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"{option} name cannot be empty.")
    return name, Path(path)


def _labels(dump: ScoreDump) -> list[int]:
    return list(dump.labels)


def _score_values_for_artifact_score(
    dump: ScoreDump,
    artifact: CalibrationArtifact,
    score: CalibrationScore,
) -> tuple[list[float], dict[str, Any]]:
    layer_key = str(artifact.target_layer)
    layer_scores = dump.sweep_scores.get(layer_key)
    if layer_scores is not None and score.name in layer_scores:
        return [float(value) for value in layer_scores[score.name]], {
            "score_source": "sweep_scores",
            "layer": artifact.target_layer,
        }

    primary_layer = dump.config.get("layer")
    primary_scores = dump.scores
    if (
        score.name in primary_scores
        and primary_layer is not None
        and int(primary_layer) == artifact.target_layer
    ):
        return [float(value) for value in primary_scores[score.name]], {
            "score_source": "scores",
            "layer": artifact.target_layer,
        }

    available_layers = sorted(int(layer) for layer in dump.sweep_scores)
    raise ValueError(
        f"score dump does not contain score {score.name!r} at artifact target layer "
        f"{artifact.target_layer}; available sweep layers: {available_layers}"
    )


def _evaluate_score(
    *,
    artifact_name: str,
    artifact_path: Path,
    artifact: CalibrationArtifact,
    target_name: str,
    target_path: Path,
    dump: ScoreDump,
    score: CalibrationScore,
    tolerance: float,
) -> dict[str, Any]:
    labels = _labels(dump)
    scores, score_source = _score_values_for_artifact_score(dump, artifact, score)
    if len(scores) != len(labels):
        raise ValueError(
            f"score {score.name!r} length does not match labels "
            f"({len(scores)} scores vs {len(labels)} labels)."
        )
    report = selective_classification_report(
        scores,
        labels,
        score.threshold,
        direction=score.direction,
    )
    alpha = score.conformal_alpha
    false_alarm = report["false_alarm"]
    controlled = None
    false_alarm_excess = None
    if alpha is not None and false_alarm is not None:
        false_alarm_excess = float(false_alarm) - float(alpha)
        controlled = float(false_alarm) <= float(alpha) + float(tolerance)

    return {
        "source_artifact": artifact_name,
        "source_artifact_path": str(artifact_path),
        "source_model_id": artifact.model_id,
        "target_dump": target_name,
        "target_scores_path": str(target_path),
        "target_model_id": dump.config.get("model"),
        "score_name": score.name,
        "direction": score.direction,
        "threshold": score.threshold,
        "conformal_alpha": alpha,
        "target_layer": artifact.target_layer,
        **score_source,
        "false_alarm_controlled": controlled,
        "false_alarm_excess": false_alarm_excess,
        "selective_report": report,
    }


def _summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    controlled_results = [
        result for result in results
        if result.get("false_alarm_controlled") is not None
    ]
    transfer_results = [
        result for result in controlled_results
        if result.get("source_artifact") != result.get("target_dump")
    ]
    self_results = [
        result for result in controlled_results
        if result.get("source_artifact") == result.get("target_dump")
    ]
    failed_transfer = [
        result for result in transfer_results
        if result.get("false_alarm_controlled") is False
    ]
    excess_values = [
        float(result["false_alarm_excess"])
        for result in transfer_results
        if result.get("false_alarm_excess") is not None
    ]
    return {
        "n_results": len(results),
        "n_self_results": len(self_results),
        "n_transfer_results": len(transfer_results),
        "self_false_alarm_controlled": sum(
            1 for result in self_results if result.get("false_alarm_controlled") is True
        ),
        "transfer_false_alarm_controlled": sum(
            1 for result in transfer_results if result.get("false_alarm_controlled") is True
        ),
        "transfer_failures": [
            {
                "source_artifact": result["source_artifact"],
                "target_dump": result["target_dump"],
                "score_name": result["score_name"],
                "target_layer": result["target_layer"],
                "false_alarm": result["selective_report"]["false_alarm"],
                "conformal_alpha": result["conformal_alpha"],
                "false_alarm_excess": result["false_alarm_excess"],
            }
            for result in failed_transfer
        ],
        "max_transfer_false_alarm_excess": max(excess_values) if excess_values else None,
    }


def build_calibration_transfer_report(
    artifacts: Sequence[tuple[str, Path]],
    score_dumps: Sequence[tuple[str, Path]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    if not artifacts:
        raise ValueError("at least one artifact is required.")
    if not score_dumps:
        raise ValueError("at least one score dump is required.")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")

    loaded_artifacts = [
        (name, path, CalibrationArtifact.load_json(path))
        for name, path in artifacts
    ]
    loaded_dumps = []
    score_dump_metadata = {}
    score_dump_metadata_cache = {}
    for name, path in score_dumps:
        score_dump = load_score_dump(path)
        loaded_dumps.append((name, path, score_dump))
        score_dump_metadata[name] = score_dump_file_metadata(
            path,
            score_dump,
            cache=score_dump_metadata_cache,
        )

    results = []
    for artifact_name, artifact_path, artifact in loaded_artifacts:
        for target_name, target_path, dump in loaded_dumps:
            for score in artifact.scores:
                results.append(_evaluate_score(
                    artifact_name=artifact_name,
                    artifact_path=artifact_path,
                    artifact=artifact,
                    target_name=target_name,
                    target_path=target_path,
                    dump=dump,
                    score=score,
                    tolerance=tolerance,
                ))

    return {
        "schema_version": 1,
        "tolerance": tolerance,
        "score_dumps": score_dump_metadata,
        "summary": _summary(results),
        "results": results,
        "notes": list(notes),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = [_parse_named_path(value, option="artifact") for value in args.artifact]
    score_dumps = [_parse_named_path(value, option="scores") for value in args.scores]
    payload = build_calibration_transfer_report(
        artifacts,
        score_dumps,
        tolerance=args.tolerance,
        notes=args.note,
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote calibration transfer report to {output_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate calibration threshold transfer across score dumps")
    parser.add_argument("--artifact", action="append", required=True,
                        help="CalibrationArtifact path, optionally named as name=path; repeatable")
    parser.add_argument("--scores", action="append", required=True,
                        help="score dump path, optionally named as name=path; repeatable")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="allowed false-alarm slack above conformal alpha")
    parser.add_argument("--note", action="append", default=[],
                        help="optional note to include in the output report; repeatable")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    args = parser.parse_args()
    payload = run(args)
    summary = payload["summary"]
    print(
        f"transfer controlled: {summary['transfer_false_alarm_controlled']}/"
        f"{summary['n_transfer_results']}  "
        f"self controlled: {summary['self_false_alarm_controlled']}/"
        f"{summary['n_self_results']}"
    )
    for failure in summary["transfer_failures"]:
        print(
            f"FAIL {failure['source_artifact']} -> {failure['target_dump']} "
            f"{failure['score_name']}@{failure['target_layer']}: "
            f"false_alarm={failure['false_alarm']:.3f} "
            f"alpha={failure['conformal_alpha']:.3f}"
        )


if __name__ == "__main__":
    main()
