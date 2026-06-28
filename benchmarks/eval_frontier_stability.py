"""Evaluate multi-seed stability for frontier score-dump comparisons.

This is a model-free post-hoc benchmark. It consumes existing
``eval_truthfulqa.py --dump-scores`` artifacts, reruns split-conformal
single/ensemble comparisons across several seeds, and records whether the
recommended signal remains stable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.eval_score_ensemble import ALPHAS, METHODS, build_ensemble_report  # noqa: E402
from eigentruth.calibration import DEFAULT_SCORE_DIRECTIONS, LayerScoreSweepCalibrator  # noqa: E402
from eigentruth.eval.score_dump import score_dump_cache_summary, score_dump_file_metadata  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("scores name cannot be empty.")
    return name, Path(path)


def _parse_csv(value: str | None, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError(f"{name} must contain at least one value.")
    return parts


def _parse_int_csv(value: str | None, *, name: str) -> tuple[int, ...]:
    if value is None or not value.strip():
        raise ValueError(f"{name} must contain at least one integer.")
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise ValueError(f"{name} must contain at least one integer.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} must not contain duplicate integers.")
    return seeds


def _float_stats(values: Sequence[float]) -> dict[str, Any]:
    finite_values = [float(value) for value in values]
    if not finite_values:
        return {"count": 0, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "count": len(finite_values),
        "mean": statistics.fmean(finite_values),
        "stdev": statistics.pstdev(finite_values) if len(finite_values) > 1 else 0.0,
        "min": min(finite_values),
        "max": max(finite_values),
    }


def _name_counts(payloads: Sequence[Mapping[str, Any] | None]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for payload in payloads:
        if payload is None:
            counter["<none>"] += 1
        else:
            counter[str(payload.get("name", "<missing>"))] += 1
    return dict(sorted(counter.items()))


def _metric_values(payloads: Sequence[Mapping[str, Any] | None], key: str) -> tuple[float, ...]:
    return tuple(
        float(payload[key])
        for payload in payloads
        if payload is not None and payload.get(key) is not None
    )


def _contains_alpha(alphas: Sequence[float], target: float) -> bool:
    return any(abs(float(alpha) - float(target)) <= 1e-12 for alpha in alphas)


def _validate_unique_score_dump_names(score_dumps: Sequence[tuple[str, Path]]) -> None:
    counts = Counter(name for name, _ in score_dumps)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"score dump names must be unique; duplicate name(s): {duplicates}.")


def _summary_int(summary: Mapping[str, Any], key: str) -> int | None:
    value = summary.get(key)
    if value is None:
        return None
    return int(value)


def _validate_score_dump_metadata(
    name: str,
    path: Path,
    metadata: Mapping[str, Any],
) -> None:
    if not metadata.get("exists"):
        raise ValueError(f"score dump {name!r} does not exist: {path}.")
    if metadata.get("kind") != "file":
        raise ValueError(f"score dump {name!r} must be a file: {path}.")
    records = metadata.get("records")
    if isinstance(records, Mapping) and not records.get("exists"):
        raise ValueError(f"score dump {name!r} records sidecar does not exist: {records.get('path')}.")
    summary = metadata.get("summary")
    if not isinstance(summary, Mapping):
        return
    n_true = _summary_int(summary, "n_true")
    n_false = _summary_int(summary, "n_false")
    if n_true is not None and n_true < 2:
        raise ValueError(
            f"score dump {name!r} must contain at least 2 true labels for split-conformal stability; "
            f"got {n_true}."
        )
    if n_false is not None and n_false < 1:
        raise ValueError(
            f"score dump {name!r} must contain at least 1 false label for stability metrics; "
            f"got {n_false}."
        )


def _summarize_seed_entries(seed_entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    single = [entry.get("best_single_at_alpha") for entry in seed_entries]
    ensemble = [entry.get("best_ensemble_at_alpha") for entry in seed_entries]
    paired = [
        (single_payload, ensemble_payload)
        for single_payload, ensemble_payload in zip(single, ensemble, strict=True)
        if isinstance(single_payload, Mapping) and isinstance(ensemble_payload, Mapping)
    ]
    single_detection = _metric_values(single, "detection")
    ensemble_detection = _metric_values(ensemble, "detection")
    detection_margin = tuple(
        float(single_payload["detection"]) - float(ensemble_payload["detection"])
        for single_payload, ensemble_payload in paired
    )
    single_beats_ensemble = sum(1 for value in detection_margin if value >= 0.0)
    return {
        "seed_count": len(seed_entries),
        "best_single_name_counts": _name_counts(single),
        "best_ensemble_name_counts": _name_counts(ensemble),
        "best_single_detection": _float_stats(single_detection),
        "best_single_false_alarm": _float_stats(_metric_values(single, "false_alarm")),
        "best_single_auroc": _float_stats(_metric_values(single, "auroc")),
        "best_ensemble_detection": _float_stats(ensemble_detection),
        "best_ensemble_false_alarm": _float_stats(_metric_values(ensemble, "false_alarm")),
        "best_ensemble_auroc": _float_stats(_metric_values(ensemble, "auroc")),
        "single_minus_ensemble_detection": _float_stats(detection_margin),
        "single_beats_ensemble_seed_count": single_beats_ensemble,
        "ensemble_beats_single_seed_count": len(detection_margin) - single_beats_ensemble,
    }


def _compact_seed_run(run_payload: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "best_single_at_alpha": run_payload.get("best_single_at_alpha"),
        "best_ensemble_at_alpha": run_payload.get("best_ensemble_at_alpha"),
    }


def _sweep_best(
    path: Path,
    *,
    signals: Sequence[str],
    alpha: float,
    best_by: str,
    cache: dict[str, Any],
) -> dict[str, Any]:
    report = LayerScoreSweepCalibrator(alpha=alpha, best_by=best_by).calibrate_from_file(
        path,
        signals=signals,
        directions={signal: DEFAULT_SCORE_DIRECTIONS.get(signal, "higher") for signal in signals},
        cache=cache,
    )
    return report.best_score().to_dict()


def build_frontier_stability_report(
    score_dumps: Sequence[tuple[str, Path]],
    *,
    signals: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str] = METHODS,
    alphas: Sequence[float] = ALPHAS,
    repeats: int = 20,
    best_alpha: float = 0.10,
    sweep_alpha: float = 0.10,
    sweep_best_by: str = "auroc",
) -> dict[str, Any]:
    """Build a compact stability report from existing score dumps."""
    if not score_dumps:
        raise ValueError("at least one score dump is required.")
    if not signals:
        raise ValueError("at least one signal is required.")
    if not seeds:
        raise ValueError("at least one seed is required.")
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")
    if any(not (0.0 < float(alpha) < 1.0) for alpha in alphas):
        raise ValueError("alphas must be in (0, 1).")
    if not _contains_alpha(alphas, best_alpha):
        raise ValueError("best_alpha must be included in alphas.")
    _validate_unique_score_dump_names(score_dumps)

    score_dump_cache: dict[str, Any] = {}
    source_metadata_by_name = {}
    for name, path in score_dumps:
        metadata = score_dump_file_metadata(path, cache=score_dump_cache)
        _validate_score_dump_metadata(name, path, metadata)
        source_metadata_by_name[name] = metadata

    seed_reports = []
    seed_run_map: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in score_dumps}
    for seed in seeds:
        seed_payload = build_ensemble_report(
            score_dumps,
            signals=signals,
            methods=methods,
            alphas=alphas,
            repeats=repeats,
            seed=int(seed),
            best_alpha=best_alpha,
            score_dump_cache=score_dump_cache,
        )
        compact_runs = []
        for run_payload in seed_payload["runs"]:
            name = str(run_payload["name"])
            entry = _compact_seed_run(run_payload, seed=int(seed))
            seed_run_map.setdefault(name, []).append(entry)
            compact_runs.append({"name": name, **entry})
        seed_reports.append({"seed": int(seed), "runs": compact_runs})

    runs = []
    for name, path in score_dumps:
        seed_entries = seed_run_map.get(name, [])
        source_metadata = source_metadata_by_name[name]
        sweep_best = _sweep_best(
            path,
            signals=signals,
            alpha=sweep_alpha,
            best_by=sweep_best_by,
            cache=score_dump_cache,
        )
        runs.append({
            "name": name,
            "scores_path": str(path),
            "score_dump": source_metadata,
            "deterministic_sweep_best": sweep_best,
            "seed_runs": seed_entries,
            "stability": _summarize_seed_entries(seed_entries),
        })

    return {
        "schema_version": 1,
        "workflow": "frontier_stability",
        "status": "complete",
        "config": {
            "signals": list(signals),
            "methods": list(methods),
            "alphas": [float(alpha) for alpha in alphas],
            "repeats": int(repeats),
            "seeds": [int(seed) for seed in seeds],
            "best_alpha": float(best_alpha),
            "sweep_alpha": float(sweep_alpha),
            "sweep_best_by": sweep_best_by,
        },
        "score_dump_cache": score_dump_cache_summary(score_dump_cache),
        "seed_reports": seed_reports,
        "runs": runs,
    }


def _artifact_paths(
    *,
    output_path: Path,
    score_dumps: Sequence[tuple[str, Path]],
    payload: Mapping[str, Any] | None = None,
) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {"stability_report": output_path}
    for name, path in score_dumps:
        artifacts[f"input_scores.{name}"] = path
    if payload is not None:
        for run in payload.get("runs", ()):
            if not isinstance(run, Mapping):
                continue
            name = str(run.get("name", "unknown"))
            score_dump = run.get("score_dump")
            if not isinstance(score_dump, Mapping):
                continue
            records = score_dump.get("records")
            if isinstance(records, Mapping) and records.get("path") is not None:
                artifacts[f"input_score_records.{name}"] = Path(str(records["path"]))
    return artifacts


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    score_dumps: Sequence[tuple[str, Path]],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = build_artifact_manifest(
        _artifact_paths(output_path=output_path, score_dumps=score_dumps, payload=payload),
        root=manifest_path.parent,
        metadata={
            "runner": "eval_frontier_stability",
            "status": payload.get("status"),
            "seeds": tuple(payload.get("config", {}).get("seeds", ())),
            "best_alpha": payload.get("config", {}).get("best_alpha"),
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _registry_run_summaries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for run in payload.get("runs", ()):
        if not isinstance(run, Mapping):
            continue
        stability = run.get("stability")
        sweep_best = run.get("deterministic_sweep_best")
        if not isinstance(stability, Mapping):
            stability = {}
        if not isinstance(sweep_best, Mapping):
            sweep_best = {}
        margin = stability.get("single_minus_ensemble_detection")
        if not isinstance(margin, Mapping):
            margin = {}
        summaries.append({
            "name": run.get("name"),
            "deterministic_best_score": sweep_best.get("score_name"),
            "deterministic_best_layer": sweep_best.get("layer"),
            "deterministic_best_auroc": sweep_best.get("auroc"),
            "deterministic_best_detection": sweep_best.get("detection"),
            "deterministic_best_false_alarm": sweep_best.get("false_alarm"),
            "best_single_name_counts": stability.get("best_single_name_counts"),
            "best_ensemble_name_counts": stability.get("best_ensemble_name_counts"),
            "single_beats_ensemble_seed_count": stability.get("single_beats_ensemble_seed_count"),
            "single_minus_ensemble_detection_mean": margin.get("mean"),
        })
    return summaries


def _record_registry(
    *,
    registry_path: Path | None,
    name: str | None,
    version: str | None,
    output_path: Path,
    manifest_path: Path | None,
    payload: Mapping[str, Any],
) -> None:
    if registry_path is None:
        return
    if not name or not version:
        raise ValueError("--registry requires --name and --version.")
    registry = ArtifactRegistry.load_json(registry_path)
    registry.record_report(
        name=name,
        path=output_path,
        version=version,
        metadata={
            "workflow": "eval_frontier_stability",
            "status": payload.get("status"),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "seeds": tuple(payload.get("config", {}).get("seeds", ())),
            "best_alpha": payload.get("config", {}).get("best_alpha"),
            "runs": tuple(run.get("name") for run in payload.get("runs", ())),
            "run_summaries": _registry_run_summaries(payload),
        },
    )
    registry.save_json()


def run(args: argparse.Namespace) -> dict[str, Any]:
    score_dumps = [_parse_named_path(value) for value in args.scores]
    signals = _parse_csv(args.signals, name="signals")
    if signals is None:
        raise ValueError("--signals is required.")
    methods = _parse_csv(args.methods, name="methods") or METHODS
    alphas = tuple(float(value) for value in (_parse_csv(args.alphas, name="alphas") or ()))
    seeds = _parse_int_csv(args.seeds, name="seeds")
    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_frontier_stability_report(
        score_dumps,
        signals=signals,
        seeds=seeds,
        methods=methods,
        alphas=alphas or ALPHAS,
        repeats=int(args.repeats),
        best_alpha=float(args.best_alpha),
        sweep_alpha=float(args.sweep_alpha),
        sweep_best_by=str(args.sweep_best_by),
    )
    payload["paths"] = {"stability_report": str(output_path)}

    manifest_path = None if args.artifact_manifest is None else Path(args.artifact_manifest)
    if manifest_path is not None:
        payload["paths"]["artifact_manifest"] = str(manifest_path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if manifest_path is not None:
        initial_manifest = _write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            score_dumps=score_dumps,
            payload=payload,
        )
        payload["artifact_manifest_summary"] = initial_manifest["summary"]
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = _write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            score_dumps=score_dumps,
            payload=payload,
        )
        payload["artifact_manifest_summary"] = manifest["summary"]

    _record_registry(
        registry_path=None if args.registry is None else Path(args.registry),
        name=args.name,
        version=args.version,
        output_path=output_path,
        manifest_path=manifest_path,
        payload=payload,
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate frontier score stability across seeds")
    parser.add_argument("--scores", action="append", required=True, help="name=score_dump path; repeatable")
    parser.add_argument("--signals", required=True, help="comma-separated diagnostic signals")
    parser.add_argument("--methods", default=",".join(METHODS), help="comma-separated ensemble methods")
    parser.add_argument("--alphas", default=",".join(str(alpha) for alpha in ALPHAS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--best-alpha", type=float, default=0.10)
    parser.add_argument("--sweep-alpha", type=float, default=0.10)
    parser.add_argument("--sweep-best-by", choices=("auroc", "detection"), default="auroc")
    parser.add_argument("--json", required=True)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    payload = run(parser.parse_args(argv))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
