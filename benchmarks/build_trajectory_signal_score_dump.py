"""Append trajectory convergence signals to a row-aligned score dump subset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.eval.score_dump import ScoreDump, load_score_dump, write_score_dump_jsonl  # noqa: E402

DEFAULT_TRAJECTORY_SIGNAL = "trajectory_convergence"
DEFAULT_TRAJECTORY_NLL_SIGNAL = "trajectory_nll_answer"


def build_trajectory_signal_score_dump(
    score_dump: ScoreDump,
    trajectory_report: Mapping[str, Any],
    *,
    source_scores_path: str | Path,
    trajectory_report_path: str | Path,
    layer: str | int = "best",
    keep_signals: Sequence[str] | None = None,
    trajectory_signal: str = DEFAULT_TRAJECTORY_SIGNAL,
    include_nll_answer: bool = False,
    trajectory_nll_signal: str = DEFAULT_TRAJECTORY_NLL_SIGNAL,
) -> ScoreDump:
    """Return a score dump subset with trajectory convergence appended."""
    selected_keep_signals = tuple(score_dump.scores) if keep_signals is None else tuple(keep_signals)
    missing = [signal for signal in selected_keep_signals if signal not in score_dump.scores]
    if missing:
        raise ValueError(f"score dump is missing requested signal(s): {missing}.")
    added_signals = [str(trajectory_signal)]
    if include_nll_answer:
        added_signals.append(str(trajectory_nll_signal))
    overlap = set(selected_keep_signals) & set(added_signals)
    if overlap:
        raise ValueError(f"trajectory signal(s) overlap existing score signals: {sorted(overlap)}.")

    layer_key, direction, layer_metadata = _resolve_layer_metadata(trajectory_report, layer=layer)
    trajectory_rows = _trajectory_rows(trajectory_report, layer_key=layer_key, include_nll_answer=include_nll_answer)
    if not trajectory_rows:
        raise ValueError("trajectory report does not contain any evaluated records.")

    selected_indices = tuple(row["index"] for row in trajectory_rows)
    _validate_unique_indices(selected_indices)
    labels = []
    statements = []
    scores = {signal: [] for signal in selected_keep_signals}
    trajectory_scores = []
    trajectory_nll_scores = []
    for row in trajectory_rows:
        index = int(row["index"])
        if index < 0 or index >= score_dump.n_total:
            raise ValueError(f"trajectory record index {index} is outside score dump range.")
        label = int(score_dump.labels[index])
        if label != int(row["label"]):
            raise ValueError(f"trajectory record index {index} label does not match score dump label.")
        labels.append(label)
        if score_dump.statements:
            statements.append(dict(score_dump.statements[index]))
        for signal in selected_keep_signals:
            scores[signal].append(float(score_dump.scores[signal][index]))
        trajectory_scores.append(float(row["trajectory_score"]))
        if include_nll_answer:
            trajectory_nll_scores.append(float(row["nll_answer"]))

    scores[str(trajectory_signal)] = trajectory_scores
    directions = {str(trajectory_signal): direction}
    if include_nll_answer:
        scores[str(trajectory_nll_signal)] = trajectory_nll_scores
        directions[str(trajectory_nll_signal)] = "higher"

    config = dict(score_dump.config)
    config["trajectory_signal_score_dump"] = {
        "builder": "build_trajectory_signal_score_dump",
        "source_scores_path": str(source_scores_path),
        "trajectory_report_path": str(trajectory_report_path),
        "layer": layer_metadata.get("layer"),
        "layer_key": layer_key,
        "resolved_layer": layer_metadata.get("resolved_layer"),
        "signals": added_signals,
        "directions": directions,
        "source_n_total": score_dump.n_total,
        "selected_indices_count": len(selected_indices),
    }
    extras = dict(score_dump.extras)
    extras["trajectory_signal_metadata"] = {
        "source_scores_path": str(source_scores_path),
        "trajectory_report_path": str(trajectory_report_path),
        "selected_indices": list(selected_indices),
        "layer": layer_metadata.get("layer"),
        "layer_key": layer_key,
        "resolved_layer": layer_metadata.get("resolved_layer"),
        "trajectory_score_direction_for_false": direction,
        "trajectory_score_best_auroc": layer_metadata.get("trajectory_score_best_auroc"),
        "signals": added_signals,
        "directions": directions,
    }
    return ScoreDump(
        labels=tuple(labels),
        scores={name: tuple(values) for name, values in scores.items()},
        config=config,
        sweep_scores={},
        statements=tuple(statements),
        extras=extras,
    )


def write_score_dump(dump: ScoreDump, output_path: str | Path, *, output_format: str) -> None:
    """Write output score dump as JSON or JSONL manifest."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dump.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if output_format == "jsonl":
        write_score_dump_jsonl(dump, output)
        return
    raise ValueError("output_format must be 'json' or 'jsonl'.")


def build_report(
    *,
    input_scores: str | Path,
    trajectory_report: str | Path,
    output: str | Path,
    output_format: str,
    layer: str | int,
    keep_signals: Sequence[str] | None,
    trajectory_signal: str,
    include_nll_answer: bool,
    trajectory_nll_signal: str,
) -> dict[str, Any]:
    """Build trajectory-signal score dump and return a compact report."""
    score_dump = load_score_dump(input_scores, allow_missing_scores=False)
    report = _load_json_object(Path(trajectory_report))
    enhanced = build_trajectory_signal_score_dump(
        score_dump,
        report,
        source_scores_path=input_scores,
        trajectory_report_path=trajectory_report,
        layer=layer,
        keep_signals=keep_signals,
        trajectory_signal=trajectory_signal,
        include_nll_answer=include_nll_answer,
        trajectory_nll_signal=trajectory_nll_signal,
    )
    write_score_dump(enhanced, output, output_format=output_format)
    metadata = enhanced.extras["trajectory_signal_metadata"]
    return {
        "schema_version": 1,
        "workflow": "trajectory_signal_score_dump_build",
        "status": "complete",
        "input_scores": str(input_scores),
        "trajectory_report": str(trajectory_report),
        "output": str(output),
        "output_format": output_format,
        "n_total": enhanced.n_total,
        "n_true": enhanced.n_true,
        "n_false": enhanced.n_false,
        "signals": list(enhanced.scores),
        "trajectory_signals": list(metadata["signals"]),
        "directions": dict(metadata["directions"]),
        "layer": metadata["layer"],
        "layer_key": metadata["layer_key"],
        "resolved_layer": metadata["resolved_layer"],
        "trajectory_score_best_auroc": metadata["trajectory_score_best_auroc"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI entry point with a testable args namespace."""
    keep_signals = _parse_csv(args.keep_signals, name="keep_signals")
    payload = build_report(
        input_scores=args.input_scores,
        trajectory_report=args.trajectory_report,
        output=args.output,
        output_format=args.output_format,
        layer=args.layer,
        keep_signals=keep_signals,
        trajectory_signal=args.trajectory_signal,
        include_nll_answer=bool(args.include_nll_answer),
        trajectory_nll_signal=args.trajectory_nll_signal,
    )
    if args.json is not None:
        Path(args.json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.quiet:
        print(
            "trajectory_signal_score_dump_ok "
            f"n_total={payload['n_total']} "
            f"signals={','.join(payload['trajectory_signals'])} "
            f"layer={payload['layer_key']}"
        )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Append trajectory signals to a score dump subset")
    parser.add_argument("--input-scores", required=True)
    parser.add_argument("--trajectory-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-format", choices=("json", "jsonl"), default="jsonl")
    parser.add_argument("--json", default=None, help="optional compact workflow report path")
    parser.add_argument("--layer", default="best")
    parser.add_argument("--keep-signals", default=None)
    parser.add_argument("--trajectory-signal", default=DEFAULT_TRAJECTORY_SIGNAL)
    parser.add_argument("--include-nll-answer", action="store_true")
    parser.add_argument("--trajectory-nll-signal", default=DEFAULT_TRAJECTORY_NLL_SIGNAL)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    run(args)


def _resolve_layer_metadata(report: Mapping[str, Any], *, layer: str | int) -> tuple[str, str, dict[str, Any]]:
    workflow = str(report.get("workflow") or "")
    summary = _mapping(report.get("summary"))
    if workflow == "truthfulqa_forced_answer_trajectory_layer_sweep":
        layer_key = _selected_layer_key(summary, layer=layer)
        layer_summary = _layer_summary(report, layer_key=layer_key)
        direction = str(layer_summary.get("trajectory_score_direction_for_false") or summary.get(
            "trajectory_score_direction_for_false"
        ))
        if direction not in {"higher", "lower"}:
            raise ValueError("trajectory score direction must be 'higher' or 'lower'.")
        return layer_key, direction, {
            "layer": layer_summary.get("layer", summary.get("best_layer")),
            "resolved_layer": layer_summary.get("resolved_layer", summary.get("best_resolved_layer")),
            "trajectory_score_best_auroc": layer_summary.get(
                "trajectory_score_best_auroc",
                summary.get("trajectory_score_best_auroc"),
            ),
        }
    if workflow == "truthfulqa_forced_answer_trajectory":
        config = _mapping(report.get("config"))
        if layer != "best" and _layer_key(layer) != _layer_key(config.get("layer")):
            raise ValueError("explicit layer does not match single-layer trajectory report.")
        direction = str(summary.get("trajectory_score_direction_for_false"))
        if direction not in {"higher", "lower"}:
            raise ValueError("trajectory score direction must be 'higher' or 'lower'.")
        resolved_layer = None
        records = _records(report)
        if records:
            resolved_layer = _mapping(_mapping(records[0].get("trajectory")).get("metadata")).get("resolved_layer")
        return _layer_key(config.get("layer")), direction, {
            "layer": config.get("layer"),
            "resolved_layer": resolved_layer,
            "trajectory_score_best_auroc": summary.get("trajectory_score_best_auroc"),
        }
    raise ValueError("trajectory report must be a TruthfulQA trajectory report.")


def _trajectory_rows(
    report: Mapping[str, Any],
    *,
    layer_key: str,
    include_nll_answer: bool,
) -> tuple[dict[str, Any], ...]:
    workflow = str(report.get("workflow") or "")
    rows = []
    for record in _records(report):
        index = _int_value(record.get("index"), name="record.index")
        label = _int_value(record.get("label"), name="record.label")
        if workflow == "truthfulqa_forced_answer_trajectory_layer_sweep":
            trajectory = _mapping(_mapping(record.get("trajectories")).get(layer_key))
        else:
            trajectory = _mapping(record.get("trajectory"))
        if not trajectory:
            raise ValueError(f"trajectory record {index} is missing layer {layer_key!r}.")
        row = {
            "index": index,
            "label": label,
            "trajectory_score": _finite_float(trajectory.get("convergence_score"), name="convergence_score"),
        }
        if include_nll_answer:
            row["nll_answer"] = _finite_float(record.get("nll_answer"), name="nll_answer")
        rows.append(row)
    return tuple(sorted(rows, key=lambda row: int(row["index"])))


def _selected_layer_key(summary: Mapping[str, Any], *, layer: str | int) -> str:
    if str(layer) == "best":
        if summary.get("best_layer_key") is not None:
            return str(summary["best_layer_key"])
        if summary.get("best_layer") is not None:
            return _layer_key(summary["best_layer"])
        raise ValueError("layer='best' requires best_layer or best_layer_key in report summary.")
    return _layer_key(layer)


def _layer_summary(report: Mapping[str, Any], *, layer_key: str) -> dict[str, Any]:
    for row in report.get("layer_summaries") or ():
        if not isinstance(row, Mapping):
            continue
        row_key = str(row.get("layer_key")) if row.get("layer_key") is not None else _layer_key(row.get("layer"))
        if row_key == layer_key:
            return dict(row)
    raise ValueError(f"layer summary {layer_key!r} is missing.")


def _records(report: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = report.get("records")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("trajectory report must contain non-empty records.")
    records = tuple(row for row in rows if isinstance(row, Mapping))
    if len(records) != len(rows):
        raise ValueError("trajectory report records must be objects.")
    return records


def _validate_unique_indices(indices: Sequence[int]) -> None:
    if len(set(indices)) != len(indices):
        raise ValueError("trajectory records must not contain duplicate indices.")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _layer_key(layer: Any) -> str:
    if isinstance(layer, str) and layer.strip():
        return layer.strip()
    if isinstance(layer, bool) or layer is None:
        raise ValueError("layer must be an integer or 'best'.")
    return str(int(layer))


def _int_value(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def _parse_csv(value: str | None, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError(f"{name} must contain at least one value.")
    if len(set(parts)) != len(parts):
        raise ValueError(f"{name} must contain unique values.")
    return parts


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


if __name__ == "__main__":
    main()
