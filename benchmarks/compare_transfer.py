"""Compare TruthfulQA layer/score sweep reports across runs.

This is a post-processing helper: it consumes JSON reports produced by
``eval_conformal.py --save-sweep-report`` and summarizes whether one score stays
useful across models, sample sizes, or layer bands. It does not load models.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


def _parse_layers(value: str | None) -> set[int] | None:
    if value is None:
        return None
    layers = set()
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            layers.add(int(text))
        except ValueError as exc:
            raise ValueError("--layers must be a comma-separated list of integer layer indexes.") from exc
    if not layers:
        raise ValueError("--layers must include at least one layer.")
    return layers


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("report name cannot be empty.")
    return name, Path(path)


def _score_rows(report: Mapping[str, Any], score_name: str, layers: set[int] | None = None) -> list[dict[str, Any]]:
    rows = []
    for layer_payload in report.get("layers", []):
        layer = int(layer_payload["layer"])
        if layers is not None and layer not in layers:
            continue
        for score in layer_payload.get("scores", []):
            if score.get("score_name") != score_name:
                continue
            rows.append({
                "layer": layer,
                "auroc": float(score["auroc"]),
                "detection": float(score.get("detection", 0.0)),
                "false_alarm": float(score.get("false_alarm", 0.0)),
                "threshold": float(score.get("threshold", 0.0)),
                "n_true": int(score.get("n_true", 0)),
                "n_false": int(score.get("n_false", 0)),
                "direction": score.get("direction"),
            })
    rows.sort(key=lambda row: row["layer"])
    return rows


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_layers": 0,
            "mean_auroc": None,
            "min_auroc": None,
            "max_auroc": None,
            "best_layer": None,
            "best_auroc": None,
            "layers_above_0_6": 0,
            "layers_above_0_7": 0,
        }
    aurocs = [float(row["auroc"]) for row in rows]
    best = max(rows, key=lambda row: float(row["auroc"]))
    return {
        "n_layers": len(rows),
        "mean_auroc": statistics.fmean(aurocs),
        "min_auroc": min(aurocs),
        "max_auroc": max(aurocs),
        "best_layer": int(best["layer"]),
        "best_auroc": float(best["auroc"]),
        "layers_above_0_6": sum(1 for value in aurocs if value >= 0.6),
        "layers_above_0_7": sum(1 for value in aurocs if value >= 0.7),
    }


def build_transfer_report(
    reports: Sequence[tuple[str, Path]],
    *,
    score_name: str,
    layers: set[int] | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    run_summaries = []
    for name, path in reports:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        rows = _score_rows(report, score_name, layers)
        summary = _summarize_rows(rows)
        run_summaries.append({
            "name": name,
            "source": str(path),
            "report_best": report.get("best"),
            "selected_layers": rows,
            "summary": summary,
        })

    available = [run for run in run_summaries if run["summary"]["n_layers"]]
    best_overall = None
    if available:
        best_run = max(available, key=lambda run: float(run["summary"]["best_auroc"]))
        best_overall = {
            "name": best_run["name"],
            "layer": best_run["summary"]["best_layer"],
            "auroc": best_run["summary"]["best_auroc"],
        }

    return {
        "score_name": score_name,
        "layer_filter": sorted(layers) if layers is not None else None,
        "n_reports": len(run_summaries),
        "n_reports_with_score": len(available),
        "best_overall": best_overall,
        "runs": run_summaries,
        "notes": list(notes),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [_parse_named_path(value) for value in args.report]
    layers = _parse_layers(args.layers)
    payload = build_transfer_report(reports, score_name=args.score, layers=layers, notes=args.note)
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote transfer report to {output_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare EigenTruth sweep reports across runs")
    parser.add_argument("--report", action="append", required=True,
                        help="sweep report path, optionally named as name=path; repeatable")
    parser.add_argument("--score", default="truth_proj", help="score name to compare")
    parser.add_argument("--layers", default=None,
                        help="optional comma-list of layer indexes to include")
    parser.add_argument("--note", action="append", default=[],
                        help="optional note to include in the output report; repeatable")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    args = parser.parse_args()
    payload = run(args)
    for item in payload["runs"]:
        summary = item["summary"]
        print(
            f"{item['name']}: n_layers={summary['n_layers']} "
            f"best={summary['best_auroc']}@{summary['best_layer']} "
            f"mean={summary['mean_auroc']}"
        )


if __name__ == "__main__":
    main()
