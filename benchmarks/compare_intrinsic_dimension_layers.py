"""Compare intrinsic-dimension peak layers with calibrated AUROC layer sweeps."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.registry import build_artifact_manifest  # noqa: E402


def _parse_named_path(spec: str) -> tuple[str, Path]:
    name, sep, path_text = spec.partition("=")
    if not sep:
        path = Path(name)
        return path.stem, path
    name = name.strip()
    if not name:
        raise ValueError("named path must use a non-empty name before '='.")
    return name, Path(path_text)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_intrinsic_reports(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, list):
        raw_reports = payload
    elif isinstance(payload, dict) and isinstance(payload.get("reports"), list):
        raw_reports = payload["reports"]
    elif isinstance(payload, dict) and "profile" in payload:
        raw_reports = [payload]
    else:
        raise ValueError("intrinsic report must contain a reports list or one profile report.")

    reports = []
    for index, raw_report in enumerate(raw_reports):
        if not isinstance(raw_report, Mapping):
            raise ValueError("intrinsic report entries must be objects.")
        if "peak_layer" not in raw_report:
            raise ValueError("intrinsic report entry is missing peak_layer.")
        name = str(raw_report.get("name") or raw_report.get("model") or f"report-{index}")
        reports.append({
            "name": name,
            "source": str(path),
            "model": raw_report.get("model"),
            "peak_layer": int(raw_report["peak_layer"]),
            "peak_intrinsic_dimension": _peak_intrinsic_dimension(raw_report),
            "shape": dict(raw_report.get("shape") or {}),
            "profile": list(raw_report.get("profile") or ()),
        })
    return reports


def _peak_intrinsic_dimension(report: Mapping[str, Any]) -> float | None:
    shape = report.get("shape")
    if isinstance(shape, Mapping) and shape.get("peak_intrinsic_dimension") is not None:
        return float(shape["peak_intrinsic_dimension"])
    peak_layer = int(report["peak_layer"])
    for entry in report.get("profile") or ():
        if int(entry["layer"]) == peak_layer:
            return float(entry["intrinsic_dimension"])
    return None


def _sweep_model_id(payload: Mapping[str, Any]) -> str | None:
    model_id = payload.get("model_id")
    if model_id is not None:
        return str(model_id)
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        config = metadata.get("config")
        if isinstance(config, Mapping) and config.get("model") is not None:
            return str(config["model"])
    return None


def _score_rows(sweep_report: Mapping[str, Any], *, score_name: str) -> list[dict[str, Any]]:
    rows = []
    for layer_payload in sweep_report.get("layers") or ():
        layer = int(layer_payload["layer"])
        scores = list(layer_payload.get("scores") or ())
        if score_name in {"best", "*", "best_per_layer"}:
            if not scores:
                continue
            selected = max(scores, key=lambda item: float(item["auroc"]))
            rows.append(_row_from_score(selected, layer=layer))
            continue
        for score in scores:
            if str(score.get("score_name")) == score_name:
                rows.append(_row_from_score(score, layer=layer))
    rows.sort(key=lambda row: (-float(row["auroc"]), int(row["layer"]), str(row["score_name"])))
    return rows


def _row_from_score(score: Mapping[str, Any], *, layer: int) -> dict[str, Any]:
    return {
        "layer": layer,
        "score_name": str(score["score_name"]),
        "auroc": float(score["auroc"]),
        "detection": None if score.get("detection") is None else float(score["detection"]),
        "false_alarm": None if score.get("false_alarm") is None else float(score["false_alarm"]),
        "direction": score.get("direction"),
    }


def _build_sweep_indexes(sweeps: Sequence[tuple[str, Path]]) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    by_name: dict[str, dict[str, Any]] = {}
    by_model: dict[str, list[str]] = {}
    for name, path in sweeps:
        payload = _load_json(path)
        if name in by_name:
            raise ValueError(f"duplicate sweep report name: {name}")
        model_id = _sweep_model_id(payload)
        by_name[name] = {
            "name": name,
            "source": str(path),
            "model": model_id,
            "payload": payload,
        }
        if model_id is not None:
            by_model.setdefault(model_id, []).append(name)
    return by_name, by_model


def _match_sweep(
    intrinsic_report: Mapping[str, Any],
    *,
    by_name: Mapping[str, dict[str, Any]],
    by_model: Mapping[str, Sequence[str]],
) -> dict[str, Any] | None:
    name = str(intrinsic_report["name"])
    if name in by_name:
        return by_name[name]
    model = intrinsic_report.get("model")
    if model is not None:
        matches = tuple(by_model.get(str(model), ()))
        if len(matches) == 1:
            return by_name[matches[0]]
    return None


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _summarize_matches(runs: Sequence[Mapping[str, Any]], *, top_k: int) -> dict[str, Any]:
    matched = [run for run in runs if run.get("matched") is True]
    top_k_hits = [run for run in matched if run.get("peak_in_top_k") is True]
    exact_hits = [run for run in matched if run.get("selected_layer_rank") == 1]
    missing = [run for run in runs if run.get("matched") is not True]
    if not matched:
        status = "no_evidence"
    elif len(top_k_hits) == len(matched) and not missing:
        status = "pass"
    elif len(top_k_hits) == len(matched):
        status = "partial_pass"
    else:
        status = "fail"
    return {
        "criterion": f"in_top_{top_k}",
        "status": status,
        "n_runs": len(runs),
        "n_matched": len(matched),
        "n_missing_sweep": len(missing),
        "n_peak_in_top_k": len(top_k_hits),
        "peak_in_top_k_rate": None if not matched else len(top_k_hits) / len(matched),
        "n_exact_best_layer": len(exact_hits),
        "exact_best_layer_rate": None if not matched else len(exact_hits) / len(matched),
        "mean_auroc_regret": _mean([float(run["auroc_regret"]) for run in matched]),
        "mean_absolute_layer_gap": _mean([float(run["absolute_layer_gap"]) for run in matched]),
    }


def compare_intrinsic_dimension_layers(
    intrinsic_reports: Sequence[tuple[str, Path]],
    sweep_reports: Sequence[tuple[str, Path]],
    *,
    score_name: str = "truth_proj",
    top_k: int = 3,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a JSON-ready comparison of ID-selected layers against AUROC rankings."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    if not intrinsic_reports:
        raise ValueError("at least one intrinsic report is required.")
    if not sweep_reports:
        raise ValueError("at least one sweep report is required.")

    by_name, by_model = _build_sweep_indexes(sweep_reports)
    runs = []
    for intrinsic_source_name, intrinsic_path in intrinsic_reports:
        for intrinsic_report in _load_intrinsic_reports(intrinsic_path):
            run_name = str(intrinsic_report["name"])
            match = _match_sweep(intrinsic_report, by_name=by_name, by_model=by_model)
            base = {
                "name": run_name,
                "intrinsic_report_name": intrinsic_source_name,
                "intrinsic_source": str(intrinsic_path),
                "model": intrinsic_report.get("model"),
                "intrinsic_peak_layer": int(intrinsic_report["peak_layer"]),
                "intrinsic_peak_dimension": intrinsic_report.get("peak_intrinsic_dimension"),
                "shape": intrinsic_report.get("shape"),
            }
            if match is None:
                runs.append({
                    **base,
                    "matched": False,
                    "missing_reason": "no sweep report matched by intrinsic report name or model id",
                })
                continue
            rows = _score_rows(match["payload"], score_name=score_name)
            peak_layer = int(intrinsic_report["peak_layer"])
            selected = next((row for row in rows if int(row["layer"]) == peak_layer), None)
            if selected is None:
                runs.append({
                    **base,
                    "matched": False,
                    "sweep_name": match["name"],
                    "sweep_source": match["source"],
                    "missing_reason": f"sweep report has no {score_name!r} score for ID peak layer",
                })
                continue

            best = rows[0]
            rank = 1 + next(index for index, row in enumerate(rows) if int(row["layer"]) == peak_layer)
            top_rows = rows[:top_k]
            runs.append({
                **base,
                "matched": True,
                "sweep_name": match["name"],
                "sweep_source": match["source"],
                "score_name": score_name,
                "selected_layer_rank": rank,
                "selected_layer_auroc": float(selected["auroc"]),
                "selected_layer_score_name": selected["score_name"],
                "best_layer": int(best["layer"]),
                "best_layer_auroc": float(best["auroc"]),
                "best_layer_score_name": best["score_name"],
                "auroc_regret": float(best["auroc"]) - float(selected["auroc"]),
                "absolute_layer_gap": abs(int(best["layer"]) - peak_layer),
                "peak_in_top_k": any(int(row["layer"]) == peak_layer for row in top_rows),
                "top_layers": top_rows,
                "n_ranked_layers": len(rows),
            })

    return {
        "workflow": "compare_intrinsic_dimension_layers",
        "score_name": score_name,
        "top_k": int(top_k),
        "summary": _summarize_matches(runs, top_k=top_k),
        "runs": runs,
        "notes": list(notes),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    intrinsic_reports = [_parse_named_path(spec) for spec in args.intrinsic_report]
    sweep_reports = [_parse_named_path(spec) for spec in args.sweep_report]
    payload = compare_intrinsic_dimension_layers(
        intrinsic_reports,
        sweep_reports,
        score_name=args.score,
        top_k=args.top_k,
        notes=args.note,
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.artifact_manifest:
        if not args.json:
            raise ValueError("--artifact-manifest requires --json.")
        manifest_path = Path(args.artifact_manifest)
        manifest = build_artifact_manifest(
            {"intrinsic_layer_prediction_report": Path(args.json)},
            root=manifest_path.parent,
            metadata={
                "workflow": "compare_intrinsic_dimension_layers",
                "score_name": args.score,
                "top_k": int(args.top_k),
            },
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare ID peak layers against sweep AUROC rankings")
    parser.add_argument(
        "--intrinsic-report",
        action="append",
        required=True,
        help="intrinsic-dimension report path, optionally NAME=PATH; repeatable",
    )
    parser.add_argument(
        "--sweep-report",
        action="append",
        required=True,
        help="layer/score sweep report path, optionally NAME=PATH; repeatable",
    )
    parser.add_argument("--score", default="truth_proj", help="score to rank; use 'best' for per-layer best score")
    parser.add_argument("--top-k", type=int, default=3, help="AUROC top-k criterion for ID peak layer")
    parser.add_argument("--note", action="append", default=[], help="optional note to include; repeatable")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest path for --json output")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
