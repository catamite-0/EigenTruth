"""Compare cheap layer-band selectors against calibrated layer/score sweeps."""

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

from benchmarks.compare_intrinsic_dimension_layers import (  # noqa: E402
    _load_intrinsic_reports,
    _parse_named_path,
    _score_rows,
    _sweep_model_id,
)
from benchmarks.compare_spectrum_layers import HEURISTICS, _load_spectrum_reports, _select_spectrum_layer  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402

DEFAULT_STRATEGIES: tuple[str, ...] = (
    "intrinsic:1",
    "intrinsic:2",
    "spectrum:max_top_eigenvalue_to_mp_upper:1",
    "union:1:max_top_eigenvalue_to_mp_upper:1",
    "union:1:max_effective_rank:1",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_key(prefix: str, index: int, name: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(name)).strip("_")
    return f"{prefix}_{index}_{safe or 'unnamed'}"


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _load_named_intrinsic_reports(inputs: Sequence[tuple[str, Path]]) -> list[dict[str, Any]]:
    reports = []
    for source_name, path in inputs:
        for report in _load_intrinsic_reports(path):
            item = dict(report)
            item["source_name"] = source_name
            reports.append(item)
    return reports


def _load_named_spectrum_reports(inputs: Sequence[tuple[str, Path]]) -> list[dict[str, Any]]:
    reports = []
    for source_name, path in inputs:
        for report in _load_spectrum_reports(path):
            item = dict(report)
            item["source_name"] = source_name
            reports.append(item)
    return reports


def _index_reports(
    reports: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[Mapping[str, Any]]]]:
    by_name = {str(report["name"]): report for report in reports}
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    for report in reports:
        model = report.get("model")
        if model is not None:
            by_model.setdefault(str(model), []).append(report)
    return by_name, by_model


def _match_report(
    *,
    sweep_name: str,
    sweep_model: str | None,
    by_name: Mapping[str, Mapping[str, Any]],
    by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any] | None:
    if sweep_name in by_name:
        return by_name[sweep_name]
    if sweep_model is not None:
        matches = tuple(by_model.get(str(sweep_model), ()))
        if len(matches) == 1:
            return matches[0]
    return None


def _parse_strategies(values: Sequence[str] | None) -> tuple[dict[str, Any], ...]:
    specs = values or DEFAULT_STRATEGIES
    parsed = []
    for raw in specs:
        for text in str(raw).split(","):
            spec = text.strip()
            if not spec:
                continue
            parts = spec.split(":")
            kind = parts[0]
            if kind == "intrinsic" and len(parts) == 2:
                radius = _non_negative_int(parts[1], field="intrinsic radius")
                parsed.append({
                    "name": f"intrinsic_radius_{radius}",
                    "kind": "intrinsic",
                    "intrinsic_radius": radius,
                    "spectrum_heuristic": None,
                    "spectrum_radius": None,
                    "raw": spec,
                })
            elif kind == "spectrum" and len(parts) == 3:
                heuristic = _validate_heuristic(parts[1])
                radius = _non_negative_int(parts[2], field="spectrum radius")
                parsed.append({
                    "name": f"spectrum_{heuristic}_radius_{radius}",
                    "kind": "spectrum",
                    "intrinsic_radius": None,
                    "spectrum_heuristic": heuristic,
                    "spectrum_radius": radius,
                    "raw": spec,
                })
            elif kind == "union" and len(parts) == 4:
                intrinsic_radius = _non_negative_int(parts[1], field="intrinsic radius")
                heuristic = _validate_heuristic(parts[2])
                spectrum_radius = _non_negative_int(parts[3], field="spectrum radius")
                parsed.append({
                    "name": f"union_intrinsic_{intrinsic_radius}_{heuristic}_{spectrum_radius}",
                    "kind": "union",
                    "intrinsic_radius": intrinsic_radius,
                    "spectrum_heuristic": heuristic,
                    "spectrum_radius": spectrum_radius,
                    "raw": spec,
                })
            else:
                raise ValueError(
                    "strategy must be intrinsic:R, spectrum:HEURISTIC:R, or union:ID_R:HEURISTIC:SPEC_R"
                )
    if not parsed:
        raise ValueError("at least one strategy is required.")
    names = [item["name"] for item in parsed]
    if len(names) != len(set(names)):
        raise ValueError("strategy names must be unique after parsing.")
    return tuple(parsed)


def _non_negative_int(value: str, *, field: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{field} must be non-negative.")
    return number


def _validate_heuristic(value: str) -> str:
    if value not in HEURISTICS:
        raise ValueError(f"unknown spectrum heuristic: {value}")
    return value


def _neighbor_layers(layer: int, available_layers: Sequence[int], *, radius: int) -> tuple[int, ...]:
    if layer not in available_layers:
        return ()
    index = available_layers.index(layer)
    start = max(0, index - int(radius))
    end = min(len(available_layers), index + int(radius) + 1)
    return tuple(int(item) for item in available_layers[start:end])


def _ordered_layers(layers: set[int], available_layers: Sequence[int]) -> list[int]:
    return [int(layer) for layer in available_layers if int(layer) in layers]


def _strategy_layers(
    strategy: Mapping[str, Any],
    *,
    available_layers: Sequence[int],
    intrinsic_report: Mapping[str, Any] | None,
    spectrum_report: Mapping[str, Any] | None,
) -> tuple[set[int], list[dict[str, Any]], str | None]:
    selected_layers: set[int] = set()
    components: list[dict[str, Any]] = []

    if strategy["kind"] in {"intrinsic", "union"}:
        if intrinsic_report is None:
            return set(), components, "missing intrinsic report"
        peak_layer = int(intrinsic_report["peak_layer"])
        radius = int(strategy["intrinsic_radius"])
        layers = _neighbor_layers(peak_layer, available_layers, radius=radius)
        if not layers:
            return set(), components, "intrinsic peak layer is not present in sweep layers"
        selected_layers.update(layers)
        components.append({
            "source": "intrinsic_dimension",
            "selected_layer": peak_layer,
            "radius": radius,
            "layers": list(layers),
            "peak_intrinsic_dimension": intrinsic_report.get("peak_intrinsic_dimension"),
        })

    if strategy["kind"] in {"spectrum", "union"}:
        if spectrum_report is None:
            return set(), components, "missing spectrum report"
        heuristic = str(strategy["spectrum_heuristic"])
        selected = _select_spectrum_layer(spectrum_report.get("layers") or (), heuristic=heuristic)
        if selected is None:
            return set(), components, f"spectrum heuristic {heuristic!r} selected no layer"
        selected_layer = int(selected["layer"])
        radius = int(strategy["spectrum_radius"])
        layers = _neighbor_layers(selected_layer, available_layers, radius=radius)
        if not layers:
            return set(), components, "spectrum-selected layer is not present in sweep layers"
        selected_layers.update(layers)
        components.append({
            "source": "spectrum",
            "heuristic": heuristic,
            "selected_layer": selected_layer,
            "selected_metric_value": float(selected["metric_value"]),
            "metric_name": selected["metric_name"],
            "radius": radius,
            "layers": list(layers),
        })

    return selected_layers, components, None


def _evaluate_strategy(
    strategy: Mapping[str, Any],
    *,
    sweep_name: str,
    sweep_source: Path,
    sweep_model: str | None,
    rows: Sequence[Mapping[str, Any]],
    intrinsic_report: Mapping[str, Any] | None,
    spectrum_report: Mapping[str, Any] | None,
    coverage_top_k: int,
) -> dict[str, Any]:
    available_layers = sorted({int(row["layer"]) for row in rows})
    base = {
        "name": sweep_name,
        "model": sweep_model,
        "sweep_source": str(sweep_source),
        "strategy": strategy["name"],
        "strategy_spec": strategy["raw"],
        "strategy_kind": strategy["kind"],
        "n_ranked_layers": len(available_layers),
    }
    if not rows:
        return {**base, "matched": False, "missing_reason": "sweep report has no matching score rows"}

    selected_layers, components, missing_reason = _strategy_layers(
        strategy,
        available_layers=available_layers,
        intrinsic_report=intrinsic_report,
        spectrum_report=spectrum_report,
    )
    if missing_reason is not None:
        return {**base, "matched": False, "components": components, "missing_reason": missing_reason}
    candidate_layers = _ordered_layers(selected_layers, available_layers)
    if not candidate_layers:
        return {**base, "matched": False, "components": components, "missing_reason": "empty candidate layer band"}

    candidate_rows = [row for row in rows if int(row["layer"]) in set(candidate_layers)]
    best = rows[0]
    band_best = max(candidate_rows, key=lambda row: float(row["auroc"]))
    top_rows = list(rows[:coverage_top_k])
    top_layers = {int(row["layer"]) for row in top_rows}
    candidate_set = set(candidate_layers)
    top_k_hit_count = len(top_layers & candidate_set)
    candidate_count = len(candidate_layers)
    n_ranked = len(available_layers)
    return {
        **base,
        "matched": True,
        "components": components,
        "candidate_layers": candidate_layers,
        "candidate_layer_count": candidate_count,
        "candidate_layer_fraction": candidate_count / n_ranked if n_ranked else None,
        "avoided_layer_count": max(0, n_ranked - candidate_count),
        "best_layer": int(best["layer"]),
        "best_layer_auroc": float(best["auroc"]),
        "best_layer_in_band": int(best["layer"]) in candidate_set,
        "band_best_layer": int(band_best["layer"]),
        "band_best_auroc": float(band_best["auroc"]),
        "band_best_rank": 1
        + next(index for index, row in enumerate(rows) if int(row["layer"]) == int(band_best["layer"])),
        "auroc_regret": float(best["auroc"]) - float(band_best["auroc"]),
        "coverage_top_k": int(coverage_top_k),
        "top_k_layer_hit_count": top_k_hit_count,
        "top_k_layer_coverage": top_k_hit_count / len(top_rows) if top_rows else None,
        "top_layers": top_rows,
    }


def _summarize_strategy(
    runs: Sequence[Mapping[str, Any]],
    *,
    strategy_name: str,
    coverage_top_k: int,
) -> dict[str, Any]:
    strategy_runs = [run for run in runs if run.get("strategy") == strategy_name]
    matched = [run for run in strategy_runs if run.get("matched") is True]
    missing = [run for run in strategy_runs if run.get("matched") is not True]
    best_hits = [run for run in matched if run.get("best_layer_in_band") is True]
    if not matched:
        status = "no_evidence"
    elif len(best_hits) == len(matched) and not missing:
        status = "pass"
    elif len(best_hits) == len(matched):
        status = "partial_pass"
    else:
        status = "fail"
    return {
        "criterion": "best_layer_in_band",
        "status": status,
        "coverage_top_k": int(coverage_top_k),
        "n_runs": len(strategy_runs),
        "n_matched": len(matched),
        "n_missing": len(missing),
        "n_best_layer_in_band": len(best_hits),
        "best_layer_in_band_rate": None if not matched else len(best_hits) / len(matched),
        "mean_candidate_layer_count": _mean([float(run["candidate_layer_count"]) for run in matched]),
        "mean_candidate_layer_fraction": _mean([float(run["candidate_layer_fraction"]) for run in matched]),
        "mean_avoided_layer_count": _mean([float(run["avoided_layer_count"]) for run in matched]),
        "mean_auroc_regret": _mean([float(run["auroc_regret"]) for run in matched]),
        "mean_top_k_layer_coverage": _mean([float(run["top_k_layer_coverage"]) for run in matched]),
    }


def _recommend_strategy(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for name, summary in summaries.items():
        if int(summary.get("n_matched") or 0) <= 0:
            continue
        hit_rate = summary.get("best_layer_in_band_rate")
        candidate_fraction = summary.get("mean_candidate_layer_fraction")
        regret = summary.get("mean_auroc_regret")
        top_k_coverage = summary.get("mean_top_k_layer_coverage")
        if hit_rate is None or candidate_fraction is None or regret is None or top_k_coverage is None:
            continue
        candidates.append((
            name,
            str(summary.get("status")),
            float(hit_rate),
            float(candidate_fraction),
            float(regret),
            float(top_k_coverage),
        ))
    if not candidates:
        return None
    passing = [item for item in candidates if item[1] == "pass" and item[2] >= 1.0]
    pool = passing or candidates
    selected = sorted(pool, key=lambda item: (-item[2], item[3], item[4], -item[5], item[0]))[0]
    name, status, hit_rate, candidate_fraction, regret, top_k_coverage = selected
    return {
        "strategy": name,
        "status": status,
        "best_layer_in_band_rate": hit_rate,
        "mean_candidate_layer_fraction": candidate_fraction,
        "mean_auroc_regret": regret,
        "mean_top_k_layer_coverage": top_k_coverage,
    }


def compare_layer_band_selectors(
    intrinsic_reports: Sequence[tuple[str, Path]],
    spectrum_reports: Sequence[tuple[str, Path]],
    sweep_reports: Sequence[tuple[str, Path]],
    *,
    score_name: str = "truth_proj",
    strategies: Sequence[str] | None = None,
    coverage_top_k: int = 2,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a JSON-ready layer-band selector comparison report."""
    if coverage_top_k < 1:
        raise ValueError("coverage_top_k must be >= 1.")
    if not intrinsic_reports:
        raise ValueError("at least one intrinsic report is required.")
    if not spectrum_reports:
        raise ValueError("at least one spectrum report is required.")
    if not sweep_reports:
        raise ValueError("at least one sweep report is required.")
    parsed_strategies = _parse_strategies(strategies)

    intrinsic_by_name, intrinsic_by_model = _index_reports(_load_named_intrinsic_reports(intrinsic_reports))
    spectrum_by_name, spectrum_by_model = _index_reports(_load_named_spectrum_reports(spectrum_reports))
    runs = []
    for sweep_name, sweep_path in sweep_reports:
        sweep_payload = _load_json(sweep_path)
        sweep_model = _sweep_model_id(sweep_payload)
        rows = _score_rows(sweep_payload, score_name=score_name)
        intrinsic = _match_report(
            sweep_name=sweep_name,
            sweep_model=sweep_model,
            by_name=intrinsic_by_name,
            by_model=intrinsic_by_model,
        )
        spectrum = _match_report(
            sweep_name=sweep_name,
            sweep_model=sweep_model,
            by_name=spectrum_by_name,
            by_model=spectrum_by_model,
        )
        for strategy in parsed_strategies:
            runs.append(_evaluate_strategy(
                strategy,
                sweep_name=sweep_name,
                sweep_source=sweep_path,
                sweep_model=sweep_model,
                rows=rows,
                intrinsic_report=intrinsic,
                spectrum_report=spectrum,
                coverage_top_k=coverage_top_k,
            ))

    summaries = {
        str(strategy["name"]): _summarize_strategy(
            runs,
            strategy_name=str(strategy["name"]),
            coverage_top_k=coverage_top_k,
        )
        for strategy in parsed_strategies
    }
    return {
        "workflow": "compare_layer_band_selectors",
        "score_name": score_name,
        "coverage_top_k": int(coverage_top_k),
        "strategies": [dict(strategy) for strategy in parsed_strategies],
        "summary": summaries,
        "recommended_strategy": _recommend_strategy(summaries),
        "runs": runs,
        "notes": list(notes),
    }


def _parse_strategy_args(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    strategies: list[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                strategies.append(item)
    return tuple(strategies)


def run(args: argparse.Namespace) -> dict[str, Any]:
    intrinsic_reports = [_parse_named_path(spec) for spec in args.intrinsic_report]
    spectrum_reports = [_parse_named_path(spec) for spec in args.spectrum_report]
    sweep_reports = [_parse_named_path(spec) for spec in args.sweep_report]
    strategies = _parse_strategy_args(args.strategy)
    payload = compare_layer_band_selectors(
        intrinsic_reports,
        spectrum_reports,
        sweep_reports,
        score_name=args.score,
        strategies=strategies,
        coverage_top_k=args.coverage_top_k,
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
            "layer_band_selector_report": Path(args.json),
        }
        for index, (name, path) in enumerate(intrinsic_reports):
            manifest_artifacts[_artifact_key("intrinsic_report", index, name)] = path
        for index, (name, path) in enumerate(spectrum_reports):
            manifest_artifacts[_artifact_key("spectrum_report", index, name)] = path
        for index, (name, path) in enumerate(sweep_reports):
            manifest_artifacts[_artifact_key("sweep_report", index, name)] = path
        manifest = build_artifact_manifest(
            manifest_artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": "compare_layer_band_selectors",
                "score_name": args.score,
                "coverage_top_k": int(args.coverage_top_k),
                "strategies": payload["strategies"],
            },
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare cheap layer-band selectors against sweep rankings")
    parser.add_argument(
        "--intrinsic-report",
        action="append",
        required=True,
        help="intrinsic-dimension report path, optionally NAME=PATH; repeatable",
    )
    parser.add_argument(
        "--spectrum-report",
        action="append",
        required=True,
        help="spectrum report path, optionally NAME=PATH; repeatable",
    )
    parser.add_argument(
        "--sweep-report",
        action="append",
        required=True,
        help="layer/score sweep report path, optionally NAME=PATH; repeatable",
    )
    parser.add_argument("--score", default="truth_proj", help="score to rank; use 'best' for per-layer best score")
    parser.add_argument(
        "--strategy",
        action="append",
        default=None,
        help=(
            "strategy spec or comma-list; forms: intrinsic:R, spectrum:HEURISTIC:R, "
            "union:ID_R:HEURISTIC:SPEC_R. Defaults to built-in candidates."
        ),
    )
    parser.add_argument("--coverage-top-k", type=int, default=2, help="top-k sweep layers to measure band coverage")
    parser.add_argument("--note", action="append", default=[], help="optional note to include; repeatable")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest path for --json output")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
