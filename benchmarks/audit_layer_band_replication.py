"""Audit whether a layer-band selector has enough replication evidence.

The selector comparison itself asks whether a cheap heuristic band contains the
best calibrated sweep layer. This audit adds the release-facing evidence gate:
the same strategy must be supported across enough runs, model families, and
dense enough layer grids before it can become a default benchmark preset.

It performs no model or verifier work; it only consumes saved
``compare_layer_band_selectors.py`` reports.
"""

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

from eigentruth.registry import ArtifactRegistry, ArtifactVerificationContext  # noqa: E402

DEFAULT_MIN_RUNS = 2
DEFAULT_MIN_MODEL_FAMILIES = 2
DEFAULT_MIN_RANKED_LAYERS = 8
DEFAULT_MIN_BEST_LAYER_HIT_RATE = 1.0
DEFAULT_MAX_MEAN_AUROC_REGRET = 0.005
DEFAULT_MAX_MEAN_CANDIDATE_LAYER_FRACTION = 0.50
DEFAULT_MIN_MEAN_TOP_K_COVERAGE = 0.50


def audit_layer_band_replication(
    layer_band_reports: Sequence[tuple[str, str | Path]],
    *,
    strategy: str | None = None,
    model_families: Mapping[str, str] | None = None,
    min_runs: int = DEFAULT_MIN_RUNS,
    min_model_families: int = DEFAULT_MIN_MODEL_FAMILIES,
    min_ranked_layers: int = DEFAULT_MIN_RANKED_LAYERS,
    min_best_layer_hit_rate: float = DEFAULT_MIN_BEST_LAYER_HIT_RATE,
    max_mean_auroc_regret: float = DEFAULT_MAX_MEAN_AUROC_REGRET,
    max_mean_candidate_layer_fraction: float = DEFAULT_MAX_MEAN_CANDIDATE_LAYER_FRACTION,
    min_mean_top_k_coverage: float = DEFAULT_MIN_MEAN_TOP_K_COVERAGE,
    allow_missing: bool = False,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a fail-closed replication audit for layer-band selectors."""
    if not layer_band_reports:
        raise ValueError("at least one layer-band report is required.")
    thresholds = {
        "min_runs": _non_negative_int(min_runs, name="min_runs"),
        "min_model_families": _non_negative_int(min_model_families, name="min_model_families"),
        "min_ranked_layers": _non_negative_int(min_ranked_layers, name="min_ranked_layers"),
        "min_best_layer_hit_rate": _unit_float(min_best_layer_hit_rate, name="min_best_layer_hit_rate"),
        "max_mean_auroc_regret": _non_negative_float(max_mean_auroc_regret, name="max_mean_auroc_regret"),
        "max_mean_candidate_layer_fraction": _unit_float(
            max_mean_candidate_layer_fraction,
            name="max_mean_candidate_layer_fraction",
        ),
        "min_mean_top_k_coverage": _unit_float(min_mean_top_k_coverage, name="min_mean_top_k_coverage"),
    }
    reports = [
        {
            "name": str(name),
            "path": str(path),
            "payload": _load_json_mapping(Path(path)),
        }
        for name, path in layer_band_reports
    ]
    strategy_name, strategy_reason = _resolve_strategy(
        [report["payload"] for report in reports],
        explicit_strategy=strategy,
    )
    rows = _audit_rows(
        reports,
        strategy=strategy_name,
        model_families=dict(model_families or {}),
        min_ranked_layers=thresholds["min_ranked_layers"],
    )
    summary = _summary(
        rows,
        strategy=strategy_name,
        thresholds=thresholds,
    )
    blocking_reasons = _blocking_reasons(
        rows,
        summary=summary,
        strategy_reason=strategy_reason,
        thresholds=thresholds,
        allow_missing=allow_missing,
    )
    status = "promote" if not blocking_reasons else "blocked"
    return {
        "schema_version": 1,
        "workflow": "layer_band_replication_audit",
        "status": status,
        "strategy": strategy_name,
        "config": {
            **thresholds,
            "allow_missing": bool(allow_missing),
            "model_families": dict(model_families or {}),
            "report_count": len(reports),
        },
        "summary": summary,
        "blocking_reasons": blocking_reasons,
        "reports": [
            {
                "name": str(report["name"]),
                "path": str(report["path"]),
                "workflow": report["payload"].get("workflow"),
                "recommended_strategy": _mapping(report["payload"].get("recommended_strategy")).get("strategy"),
                "score_name": report["payload"].get("score_name"),
                "coverage_top_k": report["payload"].get("coverage_top_k"),
            }
            for report in reports
        ],
        "runs": rows,
        "notes": list(notes),
    }


def _audit_rows(
    reports: Sequence[Mapping[str, Any]],
    *,
    strategy: str | None,
    model_families: Mapping[str, str],
    min_ranked_layers: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        payload = _mapping(report.get("payload"))
        report_name = str(report.get("name"))
        report_path = str(report.get("path"))
        source_rows = tuple(
            row
            for row in _sequence(payload.get("runs"))
            if strategy is not None and _mapping(row).get("strategy") == strategy
        )
        if strategy is None or not source_rows:
            rows.append({
                "report_name": report_name,
                "report_path": report_path,
                "strategy": strategy,
                "matched": False,
                "missing_reason": "strategy is unresolved or absent from report",
                "dense_grid_passed": False,
            })
            continue
        for raw_row in source_rows:
            row = _mapping(raw_row)
            n_ranked_layers = _int_or_none(row.get("n_ranked_layers"))
            model = None if row.get("model") is None else str(row.get("model"))
            run_name = str(row.get("name"))
            family = _model_family(
                run_name=run_name,
                model=model,
                overrides=model_families,
            )
            matched = row.get("matched") is True
            dense_grid_passed = (
                bool(matched)
                and n_ranked_layers is not None
                and n_ranked_layers >= min_ranked_layers
            )
            rows.append({
                "report_name": report_name,
                "report_path": report_path,
                "run_name": run_name,
                "model": model,
                "model_family": family,
                "strategy": strategy,
                "matched": bool(matched),
                "missing_reason": row.get("missing_reason"),
                "n_ranked_layers": n_ranked_layers,
                "dense_grid_passed": dense_grid_passed,
                "candidate_layer_count": _int_or_none(row.get("candidate_layer_count")),
                "candidate_layer_fraction": _float_or_none(row.get("candidate_layer_fraction")),
                "avoided_layer_count": _int_or_none(row.get("avoided_layer_count")),
                "best_layer": _int_or_none(row.get("best_layer")),
                "best_layer_in_band": bool(row.get("best_layer_in_band")) if matched else False,
                "band_best_layer": _int_or_none(row.get("band_best_layer")),
                "band_best_rank": _int_or_none(row.get("band_best_rank")),
                "auroc_regret": _float_or_none(row.get("auroc_regret")),
                "top_k_layer_coverage": _float_or_none(row.get("top_k_layer_coverage")),
            })
    return rows


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    strategy: str | None,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    matched = [row for row in rows if row.get("matched") is True]
    dense = [row for row in matched if row.get("dense_grid_passed") is True]
    families = sorted({
        str(row.get("model_family"))
        for row in matched
        if row.get("model_family") not in (None, "")
    })
    models = sorted({
        str(row.get("model"))
        for row in matched
        if row.get("model") not in (None, "")
    })
    hit_rate = None if not matched else _mean([
        1.0 if row.get("best_layer_in_band") is True else 0.0
        for row in matched
    ])
    dense_hit_rate = None if not dense else _mean([
        1.0 if row.get("best_layer_in_band") is True else 0.0
        for row in dense
    ])
    return {
        "strategy": strategy,
        "run_count": len(rows),
        "matched_run_count": len(matched),
        "missing_run_count": len(rows) - len(matched),
        "dense_run_count": len(dense),
        "model_count": len(models),
        "model_families": families,
        "model_family_count": len(families),
        "min_ranked_layers_observed": _min_int(row.get("n_ranked_layers") for row in matched),
        "max_ranked_layers_observed": _max_int(row.get("n_ranked_layers") for row in matched),
        "best_layer_in_band_rate": hit_rate,
        "dense_best_layer_in_band_rate": dense_hit_rate,
        "mean_candidate_layer_fraction": _mean_finite(row.get("candidate_layer_fraction") for row in matched),
        "mean_auroc_regret": _mean_finite(row.get("auroc_regret") for row in matched),
        "max_auroc_regret": _max_float(row.get("auroc_regret") for row in matched),
        "mean_top_k_layer_coverage": _mean_finite(row.get("top_k_layer_coverage") for row in matched),
        "candidate_default_ready": _candidate_ready(
            matched=matched,
            dense=dense,
            families=families,
            hit_rate=hit_rate,
            thresholds=thresholds,
        ),
    }


def _candidate_ready(
    *,
    matched: Sequence[Mapping[str, Any]],
    dense: Sequence[Mapping[str, Any]],
    families: Sequence[str],
    hit_rate: float | None,
    thresholds: Mapping[str, Any],
) -> bool:
    if len(matched) < int(thresholds["min_runs"]):
        return False
    if len(families) < int(thresholds["min_model_families"]):
        return False
    if len(dense) != len(matched):
        return False
    if hit_rate is None or hit_rate < float(thresholds["min_best_layer_hit_rate"]):
        return False
    return True


def _blocking_reasons(
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    strategy_reason: str | None,
    thresholds: Mapping[str, Any],
    allow_missing: bool,
) -> list[str]:
    reasons: list[str] = []
    if strategy_reason is not None:
        reasons.append(strategy_reason)
    _check_min(reasons, "matched_run_count", summary.get("matched_run_count"), thresholds["min_runs"])
    _check_min(
        reasons,
        "model_family_count",
        summary.get("model_family_count"),
        thresholds["min_model_families"],
    )
    if not allow_missing and int(summary.get("missing_run_count") or 0) > 0:
        reasons.append(f"missing_run_count above 0: {summary.get('missing_run_count')}")
    for row in rows:
        if row.get("matched") is not True:
            if not allow_missing:
                reasons.append(
                    f"{row.get('report_name')}:{row.get('run_name') or 'unmatched'} "
                    f"missing strategy row: {row.get('missing_reason')}"
                )
            continue
        if row.get("dense_grid_passed") is not True:
            reasons.append(
                f"{row.get('report_name')}:{row.get('run_name')} ranked layers "
                f"{row.get('n_ranked_layers')} below {thresholds['min_ranked_layers']}"
            )
    _check_min(
        reasons,
        "best_layer_in_band_rate",
        summary.get("best_layer_in_band_rate"),
        thresholds["min_best_layer_hit_rate"],
    )
    _check_max(
        reasons,
        "mean_auroc_regret",
        summary.get("mean_auroc_regret"),
        thresholds["max_mean_auroc_regret"],
    )
    _check_max(
        reasons,
        "mean_candidate_layer_fraction",
        summary.get("mean_candidate_layer_fraction"),
        thresholds["max_mean_candidate_layer_fraction"],
    )
    _check_min(
        reasons,
        "mean_top_k_layer_coverage",
        summary.get("mean_top_k_layer_coverage"),
        thresholds["min_mean_top_k_coverage"],
    )
    return reasons


def _resolve_strategy(
    reports: Sequence[Mapping[str, Any]],
    *,
    explicit_strategy: str | None,
) -> tuple[str | None, str | None]:
    if explicit_strategy is not None:
        return str(explicit_strategy), None
    recommended = []
    for report in reports:
        value = _mapping(report.get("recommended_strategy")).get("strategy")
        if value is not None:
            recommended.append(str(value))
    unique = sorted(set(recommended))
    if len(unique) == 1:
        return unique[0], None
    if not unique:
        return None, "strategy was not supplied and no report has a recommended strategy"
    return None, f"strategy was not supplied and reports recommend multiple strategies: {unique}"


def _model_family(
    *,
    run_name: str,
    model: str | None,
    overrides: Mapping[str, str],
) -> str:
    if run_name in overrides:
        return str(overrides[run_name])
    if model is not None and model in overrides:
        return str(overrides[model])
    source = model or run_name
    if "/" in source:
        return source.split("/", 1)[0]
    return source.split("-", 1)[0] or source


def _parse_named_path(value: str) -> tuple[str, Path]:
    text = str(value).strip()
    if not text:
        raise ValueError("named path must be non-empty.")
    if "=" in text:
        name, raw_path = text.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("named path name must be non-empty.")
        return name, Path(raw_path)
    path = Path(text)
    return path.stem, path


def _parse_model_family(values: Sequence[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if not values:
        return result
    for value in values:
        for item in str(value).split(","):
            text = item.strip()
            if not text:
                continue
            if "=" not in text:
                raise ValueError(f"model family override {text!r} must use key=family format.")
            key, family = text.split("=", 1)
            key = key.strip()
            family = family.strip()
            if not key or not family:
                raise ValueError("model family override key and family must be non-empty.")
            result[key] = family
    return result


def _load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _non_negative_int(value: Any, *, name: str) -> int:
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric


def _non_negative_float(value: Any, *, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a finite non-negative float.")
    return numeric


def _unit_float(value: Any, *, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not (0.0 <= numeric <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")
    return numeric


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _mean_finite(values: Any) -> float | None:
    finite = [_float_or_none(value) for value in values]
    clean = [value for value in finite if value is not None]
    return _mean(clean)


def _min_int(values: Any) -> int | None:
    clean = [value for value in (_int_or_none(item) for item in values) if value is not None]
    return None if not clean else min(clean)


def _max_int(values: Any) -> int | None:
    clean = [value for value in (_int_or_none(item) for item in values) if value is not None]
    return None if not clean else max(clean)


def _max_float(values: Any) -> float | None:
    clean = [value for value in (_float_or_none(item) for item in values) if value is not None]
    return None if not clean else max(clean)


def _check_min(reasons: list[str], label: str, value: Any, minimum: Any) -> None:
    numeric = _float_or_none(value)
    threshold = _float_or_none(minimum)
    if threshold is None:
        return
    if numeric is None or numeric < threshold:
        reasons.append(f"{label} below {threshold}: {value}")


def _check_max(reasons: list[str], label: str, value: Any, maximum: Any) -> None:
    numeric = _float_or_none(value)
    threshold = _float_or_none(maximum)
    if threshold is None:
        return
    if numeric is None or numeric > threshold:
        reasons.append(f"{label} above {threshold}: {value}")


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    else:
        text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def _write_artifact_manifest(
    *,
    context: ArtifactVerificationContext,
    output_path: Path,
    audit_report_path: Path,
    source_reports: Sequence[tuple[str, Path]],
    payload: Mapping[str, Any],
    max_workers: int,
) -> dict[str, Any]:
    artifacts: dict[str, str | Path] = {"layer_band_replication_audit": audit_report_path}
    for index, (name, path) in enumerate(source_reports):
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name).strip("_")
        artifacts[f"layer_band_report_{index}_{safe or 'unnamed'}"] = path
    summary = _mapping(payload.get("summary"))
    manifest = context.build_artifact_manifest(
        artifacts,
        root=output_path.parent,
        metadata={
            "workflow": payload.get("workflow"),
            "status": payload.get("status"),
            "strategy": payload.get("strategy"),
            "matched_run_count": summary.get("matched_run_count"),
            "model_family_count": summary.get("model_family_count"),
            "min_ranked_layers_observed": summary.get("min_ranked_layers_observed"),
            "candidate_default_ready": summary.get("candidate_default_ready"),
        },
        max_workers=max_workers,
    )
    _write_json(output_path, manifest)
    return manifest


def _verify_manifest(
    *,
    context: ArtifactVerificationContext,
    manifest_path: Path,
    output_path: Path,
    max_workers: int,
) -> dict[str, Any]:
    verification = context.load_and_verify_artifact_manifest(
        manifest_path,
        recursive=True,
        max_workers=max_workers,
    ).to_dict()
    _write_json(output_path, verification)
    return verification


def _record_registry(
    *,
    registry_path: Path | None,
    name: str | None,
    version: str | None,
    report_path: Path,
    manifest_path: Path | None,
    verification_path: Path | None,
    payload: Mapping[str, Any],
    verification: Mapping[str, Any] | None,
) -> None:
    if registry_path is None:
        return
    if not name or not version:
        raise ValueError("--registry requires --name and --version.")
    summary = _mapping(payload.get("summary"))
    metadata = {
        "workflow": payload.get("workflow"),
        "status": payload.get("status"),
        "strategy": payload.get("strategy"),
        "matched_run_count": summary.get("matched_run_count"),
        "model_family_count": summary.get("model_family_count"),
        "min_ranked_layers_observed": summary.get("min_ranked_layers_observed"),
        "candidate_default_ready": summary.get("candidate_default_ready"),
        "artifact_manifest": None if manifest_path is None else str(manifest_path),
        "manifest_verification_report": None if verification_path is None else str(verification_path),
        "manifest_verified": None if verification is None else bool(verification.get("passed")),
    }
    ArtifactRegistry.load_json(registry_path).record_report(
        name=name,
        path=report_path,
        version=version,
        metadata=metadata,
    ).save_json()


def _positive_int(value: Any, *, name: str) -> int:
    numeric = int(value)
    if numeric < 1:
        raise ValueError(f"{name} must be positive.")
    return numeric


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    source_reports = tuple(_parse_named_path(value) for value in args.layer_band_report)
    payload = audit_layer_band_replication(
        source_reports,
        strategy=args.strategy,
        model_families=_parse_model_family(args.model_family),
        min_runs=args.min_runs,
        min_model_families=args.min_model_families,
        min_ranked_layers=args.min_ranked_layers,
        min_best_layer_hit_rate=args.min_best_layer_hit_rate,
        max_mean_auroc_regret=args.max_mean_auroc_regret,
        max_mean_candidate_layer_fraction=args.max_mean_candidate_layer_fraction,
        min_mean_top_k_coverage=args.min_mean_top_k_coverage,
        allow_missing=bool(args.allow_missing),
        notes=tuple(args.note or ()),
    )
    report_path = None if args.json is None else Path(args.json)
    manifest_path = None if args.artifact_manifest is None else Path(args.artifact_manifest)
    verification_path = None if args.verification_report is None else Path(args.verification_report)
    if report_path is None and (
        manifest_path is not None
        or verification_path is not None
        or args.registry is not None
    ):
        raise ValueError("--artifact-manifest, --verification-report, and --registry require --json.")
    if verification_path is not None and manifest_path is None:
        raise ValueError("--verification-report requires --artifact-manifest.")
    context = ArtifactVerificationContext()
    manifest = None
    verification = None
    if report_path is not None:
        payload["paths"] = {
            "layer_band_replication_audit": str(report_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "manifest_verification": None if verification_path is None else str(verification_path),
        }
        _write_json(report_path, payload, compact=bool(args.compact_json))
        if manifest_path is not None:
            manifest = _write_artifact_manifest(
                context=context,
                output_path=manifest_path,
                audit_report_path=report_path,
                source_reports=source_reports,
                payload=payload,
                max_workers=args.manifest_fingerprint_workers,
            )
            payload["artifact_manifest_summary"] = manifest.get("summary")
            _write_json(report_path, payload, compact=bool(args.compact_json))
            manifest = _write_artifact_manifest(
                context=context,
                output_path=manifest_path,
                audit_report_path=report_path,
                source_reports=source_reports,
                payload=payload,
                max_workers=args.manifest_fingerprint_workers,
            )
            if verification_path is not None:
                verification = _verify_manifest(
                    context=context,
                    manifest_path=manifest_path,
                    output_path=verification_path,
                    max_workers=args.manifest_fingerprint_workers,
                )
        _record_registry(
            registry_path=None if args.registry is None else Path(args.registry),
            name=args.name,
            version=args.version,
            report_path=report_path,
            manifest_path=manifest_path,
            verification_path=verification_path,
            payload=payload,
            verification=verification,
        )
    print(
        "layer_band_replication_audit="
        f"{payload['status']} strategy={payload.get('strategy')} "
        f"families={payload['summary'].get('model_family_count')}"
    )
    if args.fail_on_blocked and payload["status"] != "promote":
        raise SystemExit(1)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit layer-band selector replication evidence")
    parser.add_argument(
        "--layer-band-report",
        action="append",
        required=True,
        help="layer-band selector comparison report, optionally NAME=PATH; repeatable",
    )
    parser.add_argument("--strategy", default=None, help="strategy to audit; defaults to unanimous recommendation")
    parser.add_argument("--model-family", action="append", default=None, help="run-or-model=family override")
    parser.add_argument("--min-runs", type=int, default=DEFAULT_MIN_RUNS)
    parser.add_argument("--min-model-families", type=int, default=DEFAULT_MIN_MODEL_FAMILIES)
    parser.add_argument("--min-ranked-layers", type=int, default=DEFAULT_MIN_RANKED_LAYERS)
    parser.add_argument("--min-best-layer-hit-rate", type=float, default=DEFAULT_MIN_BEST_LAYER_HIT_RATE)
    parser.add_argument("--max-mean-auroc-regret", type=float, default=DEFAULT_MAX_MEAN_AUROC_REGRET)
    parser.add_argument(
        "--max-mean-candidate-layer-fraction",
        type=float,
        default=DEFAULT_MAX_MEAN_CANDIDATE_LAYER_FRACTION,
    )
    parser.add_argument("--min-mean-top-k-coverage", type=float, default=DEFAULT_MIN_MEAN_TOP_K_COVERAGE)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--verification-report", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--manifest-fingerprint-workers", type=lambda value: _positive_int(
        value,
        name="manifest_fingerprint_workers",
    ), default=1)
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
