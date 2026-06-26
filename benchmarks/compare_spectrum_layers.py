"""Compare covariance-spectrum layer heuristics with calibrated AUROC sweeps."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.registry import build_artifact_manifest  # noqa: E402

HEURISTICS: dict[str, tuple[str, str]] = {
    "max_spike_count": ("spike_count", "max"),
    "max_spike_fraction": ("spike_fraction", "max"),
    "max_effective_rank": ("effective_rank", "max"),
    "max_effective_rank_fraction": ("effective_rank_fraction", "max"),
    "max_participation_ratio": ("participation_ratio", "max"),
    "max_stable_rank": ("stable_rank", "max"),
    "max_condition_number": ("condition_number", "max"),
    "min_condition_number": ("condition_number", "min"),
    "max_top_eigenvalue": ("top_eigenvalue", "max"),
    "max_top_eigenvalue_to_mp_upper": ("top_eigenvalue_to_mp_upper", "max"),
}

DEFAULT_HEURISTICS: tuple[str, ...] = (
    "max_spike_count",
    "max_spike_fraction",
    "max_effective_rank",
    "max_effective_rank_fraction",
    "max_participation_ratio",
    "max_stable_rank",
    "max_condition_number",
    "max_top_eigenvalue",
    "max_top_eigenvalue_to_mp_upper",
)


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


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _finite_int(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None:
        return None
    return int(number)


def _payload_model_id(payload: Mapping[str, Any]) -> str | None:
    model = payload.get("model")
    if model is not None:
        return str(model)
    model_id = payload.get("model_id")
    if model_id is not None:
        return str(model_id)
    config = payload.get("config")
    if isinstance(config, Mapping) and config.get("model") is not None:
        return str(config["model"])
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        nested_config = metadata.get("config")
        if isinstance(nested_config, Mapping) and nested_config.get("model") is not None:
            return str(nested_config["model"])
    return None


def _sweep_model_id(payload: Mapping[str, Any]) -> str | None:
    return _payload_model_id(payload)


def _extract_layer_metrics(layer_payload: Mapping[str, Any]) -> dict[str, float]:
    hidden_dim = _finite_float(layer_payload.get("hidden_dim"))
    spike_count = _finite_float(layer_payload.get("spike_count"))
    effective_rank = _finite_float(layer_payload.get("effective_rank"))
    top_eigenvalue = None
    top_eigenvalues = layer_payload.get("top_eigenvalues")
    if isinstance(top_eigenvalues, Sequence) and not isinstance(top_eigenvalues, (str, bytes)) and top_eigenvalues:
        top_eigenvalue = _finite_float(top_eigenvalues[0])
    mp_upper = _finite_float(layer_payload.get("marchenko_pastur_upper"))

    metrics = {
        "spike_count": spike_count,
        "spike_fraction": (
            None
            if spike_count is None or hidden_dim is None or hidden_dim <= 0.0
            else spike_count / hidden_dim
        ),
        "effective_rank": effective_rank,
        "effective_rank_fraction": (
            None
            if effective_rank is None or hidden_dim is None or hidden_dim <= 0.0
            else effective_rank / hidden_dim
        ),
        "participation_ratio": _finite_float(layer_payload.get("participation_ratio")),
        "stable_rank": _finite_float(layer_payload.get("stable_rank")),
        "condition_number": _finite_float(layer_payload.get("condition_number")),
        "top_eigenvalue": top_eigenvalue,
        "top_eigenvalue_to_mp_upper": (
            None
            if top_eigenvalue is None or mp_upper is None or mp_upper <= 0.0
            else top_eigenvalue / mp_upper
        ),
    }
    return {key: float(value) for key, value in metrics.items() if value is not None}


def _load_spectrum_reports(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, Mapping) and isinstance(payload.get("layer_spectra"), Mapping):
        raw_reports = [payload]
    elif isinstance(payload, Mapping) and isinstance(payload.get("reports"), list):
        raw_reports = payload["reports"]
    else:
        raise ValueError("spectrum report must contain layer_spectra or reports entries with layer_spectra.")

    reports = []
    for index, raw_report in enumerate(raw_reports):
        if not isinstance(raw_report, Mapping):
            raise ValueError("spectrum report entries must be objects.")
        spectra = raw_report.get("layer_spectra")
        if not isinstance(spectra, Mapping):
            raise ValueError("spectrum report entry is missing layer_spectra.")
        name = str(raw_report.get("name") or _payload_model_id(raw_report) or f"report-{index}")
        layers = []
        for layer_text, layer_payload in spectra.items():
            if not isinstance(layer_payload, Mapping):
                continue
            if layer_payload.get("status") != "ready":
                continue
            layer = int(layer_text)
            metrics = _extract_layer_metrics(layer_payload)
            layers.append({
                "layer": layer,
                "sample_count": _finite_int(layer_payload.get("sample_count")),
                "hidden_dim": _finite_int(layer_payload.get("hidden_dim")),
                "covariance_mode": layer_payload.get("covariance_mode"),
                "source": layer_payload.get("source"),
                "metrics": metrics,
            })
        layers.sort(key=lambda item: int(item["layer"]))
        reports.append({
            "name": name,
            "source": str(path),
            "model": _payload_model_id(raw_report),
            "layers": layers,
            "n_ready_layers": len(layers),
        })
    return reports


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
    spectrum_report: Mapping[str, Any],
    *,
    by_name: Mapping[str, dict[str, Any]],
    by_model: Mapping[str, Sequence[str]],
) -> dict[str, Any] | None:
    name = str(spectrum_report["name"])
    if name in by_name:
        return by_name[name]
    model = spectrum_report.get("model")
    if model is not None:
        matches = tuple(by_model.get(str(model), ()))
        if len(matches) == 1:
            return by_name[matches[0]]
    return None


def _select_spectrum_layer(
    layers: Sequence[Mapping[str, Any]],
    *,
    heuristic: str,
) -> dict[str, Any] | None:
    if heuristic not in HEURISTICS:
        raise ValueError(f"unknown spectrum heuristic: {heuristic}")
    metric_name, order = HEURISTICS[heuristic]
    candidates = []
    for layer in layers:
        metrics = layer.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        metric_value = _finite_float(metrics.get(metric_name))
        if metric_value is None:
            continue
        candidates.append((int(layer["layer"]), metric_value, layer))
    if not candidates:
        return None
    if order == "max":
        selected = sorted(candidates, key=lambda item: (-item[1], item[0]))[0]
    elif order == "min":
        selected = sorted(candidates, key=lambda item: (item[1], item[0]))[0]
    else:
        raise ValueError(f"invalid heuristic order: {order}")
    layer, metric_value, layer_payload = selected
    return {
        "layer": layer,
        "metric_name": metric_name,
        "metric_order": order,
        "metric_value": float(metric_value),
        "layer_metrics": dict(layer_payload.get("metrics") or {}),
        "sample_count": layer_payload.get("sample_count"),
        "hidden_dim": layer_payload.get("hidden_dim"),
        "covariance_mode": layer_payload.get("covariance_mode"),
        "source": layer_payload.get("source"),
    }


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _summary_status(*, matched: int, missing: int, hits: int) -> str:
    if matched == 0:
        return "no_evidence"
    if hits == matched and missing == 0:
        return "pass"
    if hits == matched:
        return "partial_pass"
    return "fail"


def _summarize_by_heuristic(
    runs: Sequence[Mapping[str, Any]],
    *,
    heuristics: Sequence[str],
    top_k: int,
) -> dict[str, dict[str, Any]]:
    summaries = {}
    for heuristic in heuristics:
        heuristic_runs = [run for run in runs if run.get("heuristic") == heuristic]
        matched = [run for run in heuristic_runs if run.get("matched") is True]
        top_k_hits = [run for run in matched if run.get("selected_layer_in_top_k") is True]
        exact_hits = [run for run in matched if run.get("selected_layer_rank") == 1]
        missing = [run for run in heuristic_runs if run.get("matched") is not True]
        summaries[heuristic] = {
            "criterion": f"in_top_{top_k}",
            "status": _summary_status(matched=len(matched), missing=len(missing), hits=len(top_k_hits)),
            "n_runs": len(heuristic_runs),
            "n_matched": len(matched),
            "n_missing": len(missing),
            "n_selected_layer_in_top_k": len(top_k_hits),
            "selected_layer_in_top_k_rate": None if not matched else len(top_k_hits) / len(matched),
            "n_exact_best_layer": len(exact_hits),
            "exact_best_layer_rate": None if not matched else len(exact_hits) / len(matched),
            "mean_auroc_regret": _mean([float(run["auroc_regret"]) for run in matched]),
            "mean_absolute_layer_gap": _mean([float(run["absolute_layer_gap"]) for run in matched]),
            "mean_selected_metric_value": _mean([float(run["selected_metric_value"]) for run in matched]),
        }
    return summaries


def _recommend_heuristic(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for heuristic, summary in summaries.items():
        if int(summary.get("n_matched") or 0) <= 0:
            continue
        hit_rate = _finite_float(summary.get("selected_layer_in_top_k_rate"))
        regret = _finite_float(summary.get("mean_auroc_regret"))
        layer_gap = _finite_float(summary.get("mean_absolute_layer_gap"))
        if hit_rate is None or regret is None or layer_gap is None:
            continue
        candidates.append((heuristic, hit_rate, regret, layer_gap))
    if not candidates:
        return None
    heuristic, hit_rate, regret, layer_gap = sorted(
        candidates,
        key=lambda item: (-item[1], item[2], item[3], item[0]),
    )[0]
    return {
        "heuristic": heuristic,
        "selected_layer_in_top_k_rate": float(hit_rate),
        "mean_auroc_regret": float(regret),
        "mean_absolute_layer_gap": float(layer_gap),
    }


def compare_spectrum_layers(
    spectrum_reports: Sequence[tuple[str, Path]],
    sweep_reports: Sequence[tuple[str, Path]],
    *,
    score_name: str = "truth_proj",
    top_k: int = 3,
    heuristics: Sequence[str] = DEFAULT_HEURISTICS,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a JSON-ready comparison of spectrum-selected layers against AUROC rankings."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    if not spectrum_reports:
        raise ValueError("at least one spectrum report is required.")
    if not sweep_reports:
        raise ValueError("at least one sweep report is required.")
    if not heuristics:
        raise ValueError("at least one heuristic is required.")
    unknown = [heuristic for heuristic in heuristics if heuristic not in HEURISTICS]
    if unknown:
        raise ValueError(f"unknown spectrum heuristic(s): {', '.join(unknown)}")

    by_name, by_model = _build_sweep_indexes(sweep_reports)
    runs = []
    for spectrum_source_name, spectrum_path in spectrum_reports:
        for spectrum_report in _load_spectrum_reports(spectrum_path):
            run_name = str(spectrum_report["name"])
            match = _match_sweep(spectrum_report, by_name=by_name, by_model=by_model)
            base = {
                "name": run_name,
                "spectrum_report_name": spectrum_source_name,
                "spectrum_source": str(spectrum_path),
                "model": spectrum_report.get("model"),
                "n_ready_spectrum_layers": int(spectrum_report.get("n_ready_layers") or 0),
            }
            for heuristic in heuristics:
                selected = _select_spectrum_layer(spectrum_report.get("layers") or (), heuristic=heuristic)
                heuristic_base = {
                    **base,
                    "heuristic": heuristic,
                    "metric_name": HEURISTICS[heuristic][0],
                    "metric_order": HEURISTICS[heuristic][1],
                }
                if selected is None:
                    runs.append({
                        **heuristic_base,
                        "matched": False,
                        "missing_reason": "no ready spectrum layer exposes the heuristic metric",
                    })
                    continue
                if match is None:
                    runs.append({
                        **heuristic_base,
                        "matched": False,
                        "selected_spectrum_layer": int(selected["layer"]),
                        "selected_metric_value": float(selected["metric_value"]),
                        "selected_layer_metrics": selected["layer_metrics"],
                        "missing_reason": "no sweep report matched by spectrum report name or model id",
                    })
                    continue
                rows = _score_rows(match["payload"], score_name=score_name)
                if not rows:
                    runs.append({
                        **heuristic_base,
                        "matched": False,
                        "sweep_name": match["name"],
                        "sweep_source": match["source"],
                        "selected_spectrum_layer": int(selected["layer"]),
                        "selected_metric_value": float(selected["metric_value"]),
                        "selected_layer_metrics": selected["layer_metrics"],
                        "missing_reason": f"sweep report has no {score_name!r} scores",
                    })
                    continue
                selected_layer = int(selected["layer"])
                selected_score = next((row for row in rows if int(row["layer"]) == selected_layer), None)
                if selected_score is None:
                    runs.append({
                        **heuristic_base,
                        "matched": False,
                        "sweep_name": match["name"],
                        "sweep_source": match["source"],
                        "selected_spectrum_layer": selected_layer,
                        "selected_metric_value": float(selected["metric_value"]),
                        "selected_layer_metrics": selected["layer_metrics"],
                        "missing_reason": f"sweep report has no {score_name!r} score for spectrum-selected layer",
                    })
                    continue

                best = rows[0]
                rank = 1 + next(index for index, row in enumerate(rows) if int(row["layer"]) == selected_layer)
                top_rows = rows[:top_k]
                runs.append({
                    **heuristic_base,
                    "matched": True,
                    "sweep_name": match["name"],
                    "sweep_source": match["source"],
                    "score_name": score_name,
                    "selected_spectrum_layer": selected_layer,
                    "selected_metric_value": float(selected["metric_value"]),
                    "selected_layer_metrics": selected["layer_metrics"],
                    "selected_layer_rank": rank,
                    "selected_layer_auroc": float(selected_score["auroc"]),
                    "selected_layer_score_name": selected_score["score_name"],
                    "best_layer": int(best["layer"]),
                    "best_layer_auroc": float(best["auroc"]),
                    "best_layer_score_name": best["score_name"],
                    "auroc_regret": float(best["auroc"]) - float(selected_score["auroc"]),
                    "absolute_layer_gap": abs(int(best["layer"]) - selected_layer),
                    "selected_layer_in_top_k": any(int(row["layer"]) == selected_layer for row in top_rows),
                    "top_layers": top_rows,
                    "n_ranked_layers": len(rows),
                })

    summaries = _summarize_by_heuristic(runs, heuristics=heuristics, top_k=top_k)
    return {
        "workflow": "compare_spectrum_layers",
        "score_name": score_name,
        "top_k": int(top_k),
        "heuristics": list(heuristics),
        "summary": summaries,
        "recommended_heuristic": _recommend_heuristic(summaries),
        "runs": runs,
        "notes": list(notes),
    }


def _parse_heuristics(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_HEURISTICS
    heuristics: list[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                heuristics.append(item)
    return tuple(heuristics)


def _artifact_key(prefix: str, index: int, name: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(name)).strip("_")
    return f"{prefix}_{index}_{safe or 'unnamed'}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    spectrum_reports = [_parse_named_path(spec) for spec in args.spectrum_report]
    sweep_reports = [_parse_named_path(spec) for spec in args.sweep_report]
    heuristics = _parse_heuristics(args.heuristic)
    payload = compare_spectrum_layers(
        spectrum_reports,
        sweep_reports,
        score_name=args.score,
        top_k=args.top_k,
        heuristics=heuristics,
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
        manifest_artifacts: dict[str, Path] = {
            "spectrum_layer_prediction_report": Path(args.json),
        }
        for index, (name, path) in enumerate(spectrum_reports):
            manifest_artifacts[_artifact_key("spectrum_report", index, name)] = path
        for index, (name, path) in enumerate(sweep_reports):
            manifest_artifacts[_artifact_key("sweep_report", index, name)] = path
        manifest = build_artifact_manifest(
            manifest_artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": "compare_spectrum_layers",
                "score_name": args.score,
                "top_k": int(args.top_k),
                "heuristics": list(heuristics),
            },
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare spectrum-selected layers against sweep AUROC rankings")
    parser.add_argument(
        "--spectrum-report",
        action="append",
        required=True,
        help="eval_truthfulqa JSON report with layer_spectra, optionally NAME=PATH; repeatable",
    )
    parser.add_argument(
        "--sweep-report",
        action="append",
        required=True,
        help="layer/score sweep report path, optionally NAME=PATH; repeatable",
    )
    parser.add_argument("--score", default="truth_proj", help="score to rank; use 'best' for per-layer best score")
    parser.add_argument("--top-k", type=int, default=3, help="AUROC top-k criterion for spectrum-selected layer")
    parser.add_argument(
        "--heuristic",
        action="append",
        default=None,
        help="spectrum heuristic or comma-list; repeatable. Defaults to the built-in heuristic set.",
    )
    parser.add_argument("--note", action="append", default=[], help="optional note to include; repeatable")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest path for --json output")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
