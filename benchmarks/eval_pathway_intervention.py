"""Compare baseline and pathway-intervention score-dump reruns.

This benchmark is intentionally model-free. It consumes two row-aligned
``eval_truthfulqa.py --dump-scores`` artifacts: one baseline run and one run
produced after a mechanism intervention such as attention-pathway knockout or
token patching. The report records direction-aware score deltas using
``pathway_intervention_effect`` so causal claims can be tied to rerun evidence
instead of tensor-only diagnostics.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.calibration import DEFAULT_SCORE_DIRECTIONS  # noqa: E402
from eigentruth.eval.score_dump import ScoreDump, load_score_dump, score_dump_file_metadata  # noqa: E402
from eigentruth.intervention import pathway_intervention_effect  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

SCHEMA_VERSION = 1


def build_pathway_intervention_report(
    baseline: ScoreDump,
    intervened: ScoreDump,
    *,
    baseline_scores_path: str | Path,
    intervened_scores_path: str | Path,
    signals: Sequence[str] | None = None,
    directions: Mapping[str, str] | None = None,
    pathway: str = "unspecified",
    intervention_name: str = "pathway_intervention",
    min_mean_risk_reduction: float = 0.0,
    min_improved_fraction: float = 0.5,
    include_record_effects: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a direction-aware intervention effect report for two score dumps."""
    _validate_score_dumps_aligned(baseline, intervened)
    selected_signals = _resolve_signals(baseline, intervened, signals=signals)
    resolved_directions = _resolve_directions(selected_signals, directions=directions)
    min_mean_risk_reduction = _finite_float(
        min_mean_risk_reduction,
        name="min_mean_risk_reduction",
    )
    min_improved_fraction = _rate_float(min_improved_fraction, name="min_improved_fraction")

    record_effects = []
    effects_by_signal: dict[str, list[dict[str, Any]]] = {signal: [] for signal in selected_signals}
    for index, label in enumerate(baseline.labels):
        signal_payload = {}
        for signal in selected_signals:
            effect = pathway_intervention_effect(
                signal,
                baseline_score=baseline.scores[signal][index],
                intervened_score=intervened.scores[signal][index],
                direction=resolved_directions[signal],
                metadata={
                    "pathway": pathway,
                    "intervention_name": intervention_name,
                    "record_index": index,
                    "label": int(label),
                },
            ).to_dict()
            effects_by_signal[signal].append(effect)
            if include_record_effects:
                signal_payload[signal] = effect
        if include_record_effects:
            record_effects.append({
                "record_index": index,
                "label": int(label),
                "statement": _statement_payload(baseline, intervened, index),
                "signals": signal_payload,
            })

    signal_summaries = {
        signal: _summarize_signal_effects(
            signal,
            effects_by_signal[signal],
            labels=baseline.labels,
            min_mean_risk_reduction=min_mean_risk_reduction,
            min_improved_fraction=min_improved_fraction,
        )
        for signal in selected_signals
    }
    best_signal = _best_signal(signal_summaries)
    gate = _gate_payload(
        signal_summaries,
        best_signal=best_signal,
        min_mean_risk_reduction=min_mean_risk_reduction,
        min_improved_fraction=min_improved_fraction,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": "pathway_intervention_effect_eval",
        "status": "complete",
        "config": {
            "baseline_scores": str(baseline_scores_path),
            "intervened_scores": str(intervened_scores_path),
            "pathway": str(pathway),
            "intervention_name": str(intervention_name),
            "signals": list(selected_signals),
            "directions": resolved_directions,
            "min_mean_risk_reduction": min_mean_risk_reduction,
            "min_improved_fraction": min_improved_fraction,
            "include_record_effects": bool(include_record_effects),
        },
        "source_score_dumps": {
            "baseline": baseline.summary(),
            "intervened": intervened.summary(),
        },
        "summary": {
            "n_total": baseline.n_total,
            "n_true": baseline.n_true,
            "n_false": baseline.n_false,
            "signal_count": len(selected_signals),
            "best_signal": best_signal,
            "gate": gate,
        },
        "signals": signal_summaries,
        "record_effects": record_effects,
        "metadata": dict(metadata or {}),
    }


def run_pathway_intervention_eval(
    *,
    baseline_scores_path: str | Path,
    intervened_scores_path: str | Path,
    output_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    register_name: str | None = None,
    register_version: str = "0.1",
    signals: Sequence[str] | None = None,
    directions: Mapping[str, str] | None = None,
    pathway: str = "unspecified",
    intervention_name: str = "pathway_intervention",
    min_mean_risk_reduction: float = 0.0,
    min_improved_fraction: float = 0.5,
    include_record_effects: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load score dumps, write optional artifacts, and return the report payload."""
    baseline_path = Path(baseline_scores_path)
    intervened_path = Path(intervened_scores_path)
    baseline = load_score_dump(baseline_path)
    intervened = load_score_dump(intervened_path)
    report = build_pathway_intervention_report(
        baseline,
        intervened,
        baseline_scores_path=baseline_path,
        intervened_scores_path=intervened_path,
        signals=signals,
        directions=directions,
        pathway=pathway,
        intervention_name=intervention_name,
        min_mean_risk_reduction=min_mean_risk_reduction,
        min_improved_fraction=min_improved_fraction,
        include_record_effects=include_record_effects,
        metadata=metadata,
    )
    report["source_score_dump_files"] = {
        "baseline": score_dump_file_metadata(baseline_path, dump=baseline),
        "intervened": score_dump_file_metadata(intervened_path, dump=intervened),
    }

    paths: dict[str, str] = {}
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(strict_json_dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths["report"] = str(output)

    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = build_artifact_manifest(
            {
                "baseline_scores": baseline_path,
                "intervened_scores": intervened_path,
                "pathway_intervention_report": output_path,
            },
            root=manifest_path.parent,
            metadata={
                "workflow": "pathway_intervention_effect_eval",
                "pathway": pathway,
                "intervention_name": intervention_name,
                "gate_status": report["summary"]["gate"]["status"],
                "best_signal": report["summary"]["best_signal"],
            },
        )
        manifest_path.write_text(strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths["artifact_manifest"] = str(manifest_path)

    registry_record = None
    if registry_path is not None and register_name:
        if output_path is None:
            raise ValueError("output_path is required when registering the intervention report.")
        registry = ArtifactRegistry.load_json(registry_path)
        registry.record_report(
            name=register_name,
            path=output_path,
            version=register_version,
            metadata={
                "workflow": "pathway_intervention_effect_eval",
                "pathway": pathway,
                "intervention_name": intervention_name,
                "gate_status": report["summary"]["gate"]["status"],
                "best_signal": report["summary"]["best_signal"],
                "n_total": report["summary"]["n_total"],
            },
        ).save_json()
        registry_record = f"report:{register_name}:{register_version}"
        paths["registry"] = str(registry_path)

    payload = dict(report)
    payload["paths"] = paths
    if registry_record is not None:
        payload["registry_record"] = registry_record
    if output_path is not None:
        Path(output_path).write_text(strict_json_dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI entry point with a testable args namespace."""
    payload = run_pathway_intervention_eval(
        baseline_scores_path=args.baseline_scores,
        intervened_scores_path=args.intervened_scores,
        output_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        register_name=args.register_name,
        register_version=args.register_version,
        signals=_parse_csv(args.signals, name="signals"),
        directions=_parse_directions(args.direction),
        pathway=args.pathway,
        intervention_name=args.intervention_name,
        min_mean_risk_reduction=args.min_mean_risk_reduction,
        min_improved_fraction=args.min_improved_fraction,
        include_record_effects=not bool(args.summary_only),
        metadata=_parse_metadata(args.metadata),
    )
    if not args.quiet:
        gate = payload["summary"]["gate"]
        print(
            "pathway_intervention_eval_ok "
            f"status={gate['status']} "
            f"best_signal={payload['summary']['best_signal']} "
            f"n_total={payload['summary']['n_total']}"
        )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare baseline/intervened score-dump reruns")
    parser.add_argument("--baseline-scores", required=True)
    parser.add_argument("--intervened-scores", required=True)
    parser.add_argument("--json", default=None, help="path to write the intervention report")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--register-name", default=None, help="optional registry report name")
    parser.add_argument("--register-version", default="0.1")
    parser.add_argument(
        "--signals",
        default=None,
        help="comma-separated primary score signals; defaults to intersection",
    )
    parser.add_argument(
        "--direction",
        action="append",
        default=(),
        help="override signal direction as name=higher or name=lower; repeatable",
    )
    parser.add_argument("--pathway", default="unspecified")
    parser.add_argument("--intervention-name", default="pathway_intervention")
    parser.add_argument("--min-mean-risk-reduction", type=float, default=0.0)
    parser.add_argument("--min-improved-fraction", type=float, default=0.5)
    parser.add_argument("--summary-only", action="store_true", help="omit per-record effects from the report")
    parser.add_argument("--metadata", action="append", default=(), help="metadata key=value; repeatable")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    run(args)


def _validate_score_dumps_aligned(baseline: ScoreDump, intervened: ScoreDump) -> None:
    if baseline.n_total != intervened.n_total:
        raise ValueError("baseline and intervened score dumps must have the same n_total.")
    if tuple(baseline.labels) != tuple(intervened.labels):
        raise ValueError("baseline and intervened score dumps must have identical labels.")
    if baseline.n_total < 1:
        raise ValueError("score dumps must contain at least one record.")


def _resolve_signals(
    baseline: ScoreDump,
    intervened: ScoreDump,
    *,
    signals: Sequence[str] | None,
) -> tuple[str, ...]:
    if signals is None:
        resolved = tuple(sorted(set(baseline.scores) & set(intervened.scores)))
    else:
        resolved = tuple(dict.fromkeys(str(signal) for signal in signals))
    if not resolved:
        raise ValueError("at least one shared primary score signal is required.")
    missing_baseline = [signal for signal in resolved if signal not in baseline.scores]
    missing_intervened = [signal for signal in resolved if signal not in intervened.scores]
    if missing_baseline or missing_intervened:
        raise ValueError(
            "requested signals must exist in both score dumps; "
            f"missing_baseline={missing_baseline}, missing_intervened={missing_intervened}."
        )
    return resolved


def _resolve_directions(
    signals: Sequence[str],
    *,
    directions: Mapping[str, str] | None,
) -> dict[str, str]:
    overrides = {} if directions is None else {str(key): _coerce_direction(value) for key, value in directions.items()}
    unknown = sorted(set(overrides) - set(signals))
    if unknown:
        raise ValueError(f"direction overrides reference unknown signal(s): {unknown}.")
    return {
        signal: overrides.get(signal, DEFAULT_SCORE_DIRECTIONS.get(signal, "higher"))
        for signal in signals
    }


def _summarize_signal_effects(
    signal: str,
    effects: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[int],
    min_mean_risk_reduction: float,
    min_improved_fraction: float,
) -> dict[str, Any]:
    risk_reductions = tuple(float(effect["risk_reduction"]) for effect in effects)
    anomalous_deltas = tuple(float(effect["anomalous_delta"]) for effect in effects)
    improved = tuple(bool(effect["improved"]) for effect in effects)
    true_values = tuple(value for value, label in zip(risk_reductions, labels, strict=True) if int(label) == 0)
    false_values = tuple(value for value, label in zip(risk_reductions, labels, strict=True) if int(label) == 1)
    improved_fraction = sum(1 for value in improved if value) / len(improved)
    worsened_fraction = sum(1 for value in risk_reductions if value < 0.0) / len(risk_reductions)
    mean_risk_reduction = statistics.fmean(risk_reductions)
    status = (
        "promote"
        if mean_risk_reduction >= min_mean_risk_reduction and improved_fraction >= min_improved_fraction
        else "blocked"
    )
    return {
        "signal": signal,
        "direction": str(effects[0]["direction"]),
        "n_total": len(effects),
        "mean_delta": statistics.fmean(float(effect["delta"]) for effect in effects),
        "mean_anomalous_delta": statistics.fmean(anomalous_deltas),
        "mean_risk_reduction": mean_risk_reduction,
        "median_risk_reduction": statistics.median(risk_reductions),
        "stdev_risk_reduction": statistics.pstdev(risk_reductions) if len(risk_reductions) > 1 else 0.0,
        "min_risk_reduction": min(risk_reductions),
        "max_risk_reduction": max(risk_reductions),
        "improved_count": sum(1 for value in improved if value),
        "worsened_count": sum(1 for value in risk_reductions if value < 0.0),
        "unchanged_count": sum(1 for value in risk_reductions if value == 0.0),
        "improved_fraction": improved_fraction,
        "worsened_fraction": worsened_fraction,
        "true_mean_risk_reduction": None if not true_values else statistics.fmean(true_values),
        "false_mean_risk_reduction": None if not false_values else statistics.fmean(false_values),
        "status": status,
    }


def _best_signal(signal_summaries: Mapping[str, Mapping[str, Any]]) -> str | None:
    if not signal_summaries:
        return None
    return max(
        signal_summaries,
        key=lambda signal: (
            float(signal_summaries[signal]["mean_risk_reduction"]),
            float(signal_summaries[signal]["improved_fraction"]),
            signal,
        ),
    )


def _gate_payload(
    signal_summaries: Mapping[str, Mapping[str, Any]],
    *,
    best_signal: str | None,
    min_mean_risk_reduction: float,
    min_improved_fraction: float,
) -> dict[str, Any]:
    if best_signal is None:
        return {
            "status": "blocked",
            "reason": "no signal summaries were produced",
            "min_mean_risk_reduction": min_mean_risk_reduction,
            "min_improved_fraction": min_improved_fraction,
        }
    best = signal_summaries[best_signal]
    if best["status"] == "promote":
        return {
            "status": "promote",
            "reason": "best signal meets intervention evidence floor",
            "best_signal": best_signal,
            "best_mean_risk_reduction": best["mean_risk_reduction"],
            "best_improved_fraction": best["improved_fraction"],
            "min_mean_risk_reduction": min_mean_risk_reduction,
            "min_improved_fraction": min_improved_fraction,
        }
    return {
        "status": "blocked",
        "reason": "no signal met intervention evidence floor",
        "best_signal": best_signal,
        "best_mean_risk_reduction": best["mean_risk_reduction"],
        "best_improved_fraction": best["improved_fraction"],
        "min_mean_risk_reduction": min_mean_risk_reduction,
        "min_improved_fraction": min_improved_fraction,
    }


def _statement_payload(baseline: ScoreDump, intervened: ScoreDump, index: int) -> dict[str, Any] | None:
    baseline_statement = baseline.statements[index] if baseline.statements else None
    intervened_statement = intervened.statements[index] if intervened.statements else None
    if baseline_statement is None and intervened_statement is None:
        return None
    if (
        baseline_statement is not None
        and intervened_statement is not None
        and dict(baseline_statement) != dict(intervened_statement)
    ):
        return {
            "baseline": dict(baseline_statement),
            "intervened": dict(intervened_statement),
            "matched": False,
        }
    statement = baseline_statement if baseline_statement is not None else intervened_statement
    return {"value": dict(statement or {}), "matched": True}


def _parse_csv(value: str | None, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError(f"{name} must contain at least one value.")
    return parts


def _parse_directions(values: Sequence[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError("direction overrides must use name=higher or name=lower.")
        name, direction = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("direction override signal name cannot be empty.")
        parsed[name] = _coerce_direction(direction)
    return parsed


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError("metadata entries must use key=value.")
        key, metadata_value = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("metadata key cannot be empty.")
        parsed[key] = metadata_value
    return parsed


def _coerce_direction(value: Any) -> str:
    text = str(value).strip().lower()
    if text not in {"higher", "lower"}:
        raise ValueError("direction must be one of: higher, lower.")
    return text


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number.")
    return numeric


def _rate_float(value: Any, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return numeric


if __name__ == "__main__":
    main()
