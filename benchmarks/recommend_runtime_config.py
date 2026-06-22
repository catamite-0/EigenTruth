"""Build a deployment-oriented runtime recommendation from benchmark reports.

The cache-profile matrix and worker-count sweep reports are intentionally
experiment shaped. This helper turns their promotion decisions into one compact
machine-readable recommendation: layer, batch size, capture mode, token budget,
prefix-KV mode, and worker count.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_runtime_recommendation(
    matrix_report: Mapping[str, Any],
    *,
    worker_sweep_report: Mapping[str, Any] | None = None,
    matrix_report_path: str | Path | None = None,
    worker_sweep_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return one runtime recommendation from matrix and optional worker evidence."""
    matrix_decision = _mapping(matrix_report.get("matrix_decision"))
    worker_decision = _mapping(
        None if worker_sweep_report is None else worker_sweep_report.get("worker_sweep_decision")
    )
    matrix_status = str(matrix_decision.get("status") or "missing")
    worker_status = None if worker_sweep_report is None else str(worker_decision.get("status") or "missing")
    status = _combined_status(matrix_status, worker_status)
    recommended = None
    if status == "promote":
        recommended = _recommendation(
            matrix_report,
            matrix_decision=matrix_decision,
            worker_sweep_report=worker_sweep_report,
            worker_decision=worker_decision,
            matrix_report_path=matrix_report_path,
        )
        if recommended is None:
            status = "no_candidate"
    blocking_reasons = _blocking_reasons(
        matrix_decision=matrix_decision,
        worker_decision=worker_decision,
        worker_sweep_report=worker_sweep_report,
    )
    if matrix_status == "promote" and recommended is None:
        blocking_reasons.append("matrix: promoted matrix did not include a recommended runtime cell")
    evidence = _evidence(
        matrix_report,
        matrix_decision=matrix_decision,
        worker_sweep_report=worker_sweep_report,
        worker_decision=worker_decision,
        matrix_report_path=matrix_report_path,
        worker_sweep_report_path=worker_sweep_report_path,
    )
    report = {
        "schema_version": 1,
        "workflow": "runtime_config_recommendation",
        "status": status,
        "recommendation": recommended,
        "evidence": evidence,
        "blocking_reasons": blocking_reasons,
    }
    if recommended is not None:
        report["benchmark_flags"] = _benchmark_flags(recommended, matrix_report)
    return report


def _recommendation(
    matrix_report: Mapping[str, Any],
    *,
    matrix_decision: Mapping[str, Any],
    worker_sweep_report: Mapping[str, Any] | None,
    worker_decision: Mapping[str, Any],
    matrix_report_path: str | Path | None,
) -> dict[str, Any] | None:
    matrix_recommended = _recommended_runtime_row(matrix_report, matrix_decision)
    if not matrix_recommended or any(
        matrix_recommended.get(key) is None
        for key in ("layer", "batch_size", "hidden_state_capture")
    ):
        return None
    worker_count = _recommended_worker_count(
        matrix_report,
        worker_sweep_report=worker_sweep_report,
        worker_decision=worker_decision,
    )
    quality = _quality_signal_summary(
        matrix_report,
        matrix_recommended,
        matrix_report_path=matrix_report_path,
    )
    recommendation = {
        "cell_id": matrix_recommended.get("id") or matrix_decision.get("recommended_cell"),
        "layer": matrix_recommended.get("layer"),
        "batch_size": matrix_recommended.get("batch_size"),
        "hidden_state_capture": matrix_recommended.get("hidden_state_capture"),
        "max_batch_tokens": int(matrix_recommended.get("max_batch_tokens") or 0),
        "prefix_kv_cache": bool(matrix_recommended.get("prefix_kv_cache", False)),
        "max_workers": worker_count,
        "recommendation_metric": matrix_decision.get("recommendation_metric"),
        "cache_only_total_seconds": matrix_recommended.get("cache_only_total_seconds"),
        "uncached_forced_answer_forward_seconds": matrix_recommended.get(
            "uncached_forced_answer_forward_seconds"
        ),
        "truth_proj_auroc": matrix_recommended.get("truth_proj_auroc"),
        "quality_signals": quality["signals"],
        "best_quality_signal": quality["best"],
    }
    return recommendation


def _quality_signal_summary(
    matrix_report: Mapping[str, Any],
    matrix_recommended: Mapping[str, Any],
    *,
    matrix_report_path: str | Path | None = None,
) -> dict[str, Any]:
    cell_id = matrix_recommended.get("id")
    cell = _find_by_id(matrix_report.get("cells"), cell_id) if cell_id is not None else {}
    signals, source = _quality_signals_from_cell(cell, matrix_report_path=matrix_report_path)
    if not signals:
        truth_proj = _float_or_none(matrix_recommended.get("truth_proj_auroc"))
        if truth_proj is not None:
            signals = {"truth_proj": truth_proj}
            source = "leaderboard"
    signals = {name: signals[name] for name in sorted(signals)}
    best = _best_quality_signal(signals)
    return {
        "signals": signals,
        "best": best,
        "source": source,
        "count": len(signals),
    }


def _quality_signals_from_cell(
    cell: Mapping[str, Any],
    *,
    matrix_report_path: str | Path | None,
) -> tuple[dict[str, float], str | None]:
    triplet = _mapping(cell.get("triplet"))
    results = _mapping(triplet.get("results"))
    for run_name in ("cache_only", "cached", "uncached"):
        result_path = results.get(run_name)
        if not result_path:
            continue
        signals, source = _quality_signals_from_result_path(
            Path(str(result_path)),
            base_dir=None if matrix_report_path is None else Path(str(matrix_report_path)).parent,
        )
        if signals:
            return signals, source
    summary_signals = _finite_float_mapping(_mapping(_mapping(cell.get("summary")).get("quality_signals")))
    if summary_signals:
        return summary_signals, "matrix_cell_summary"
    return {}, None


def _quality_signals_from_result_path(
    path: Path,
    *,
    base_dir: Path | None,
) -> tuple[dict[str, float], str | None]:
    candidates = [path]
    if base_dir is not None and not path.is_absolute():
        candidates.append(base_dir / path)
    for candidate in candidates:
        try:
            payload = _load_json(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        signals = _finite_float_mapping(_mapping(payload.get("auroc")))
        if signals:
            return signals, str(candidate)
    return {}, None


def _finite_float_mapping(values: Mapping[str, Any]) -> dict[str, float]:
    signals = {}
    for key, value in values.items():
        numeric = _float_or_none(value)
        if numeric is not None:
            signals[str(key)] = numeric
    return signals


def _best_quality_signal(signals: Mapping[str, float]) -> dict[str, Any] | None:
    if not signals:
        return None
    name, auroc = sorted(signals.items(), key=lambda item: (-item[1], item[0]))[0]
    return {"name": name, "auroc": auroc}


def _recommended_runtime_row(
    matrix_report: Mapping[str, Any],
    matrix_decision: Mapping[str, Any],
) -> dict[str, Any]:
    recommended = _mapping(matrix_decision.get("recommended"))
    cell_id = recommended.get("id") or matrix_decision.get("recommended_cell")
    merged: dict[str, Any] = {}
    if cell_id is not None:
        merged.update(_find_by_id(matrix_report.get("cells"), cell_id))
        merged.update(_find_by_id(matrix_report.get("leaderboard"), cell_id))
    merged.update(recommended)
    if cell_id is not None and merged.get("id") is None:
        merged["id"] = cell_id
    return merged


def _find_by_id(values: Any, cell_id: Any) -> dict[str, Any]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        return {}
    for value in values:
        item = _mapping(value)
        if item.get("id") == cell_id:
            return item
    return {}


def _recommended_worker_count(
    matrix_report: Mapping[str, Any],
    *,
    worker_sweep_report: Mapping[str, Any] | None,
    worker_decision: Mapping[str, Any],
) -> int | None:
    worker_count = worker_decision.get("recommended_worker_count")
    if isinstance(worker_count, int) and not isinstance(worker_count, bool):
        return worker_count
    config = _mapping(matrix_report.get("config"))
    configured = config.get("max_workers")
    if isinstance(configured, int) and not isinstance(configured, bool):
        return configured
    if worker_sweep_report is not None:
        worker_reports = worker_sweep_report.get("worker_reports")
        if isinstance(worker_reports, Sequence) and not isinstance(worker_reports, str):
            for report in worker_reports:
                report_map = _mapping(report)
                value = report_map.get("worker_count")
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
    return None


def _benchmark_flags(
    recommendation: Mapping[str, Any],
    matrix_report: Mapping[str, Any],
) -> dict[str, list[str]]:
    layer = str(recommendation.get("layer"))
    batch_size = str(recommendation.get("batch_size"))
    capture = str(recommendation.get("hidden_state_capture"))
    max_batch_tokens = int(recommendation.get("max_batch_tokens") or 0)
    max_workers = recommendation.get("max_workers")
    prefix_kv_cache = bool(recommendation.get("prefix_kv_cache", False))
    length_bucketed = bool(_mapping(matrix_report.get("config")).get("length_bucketed_batches", False))

    eval_flags = ["--layer", layer, "--batch-size", batch_size, "--hidden-state-capture", capture]
    if max_batch_tokens > 0:
        eval_flags.extend(["--max-batch-tokens", str(max_batch_tokens)])
    if prefix_kv_cache:
        eval_flags.append("--prefix-kv-cache")
    if length_bucketed:
        eval_flags.append("--length-bucketed-batches")

    matrix_flags = ["--layers", layer, "--batch-sizes", batch_size, "--hidden-state-captures", capture]
    if max_batch_tokens > 0:
        matrix_flags.extend(["--max-batch-tokens", str(max_batch_tokens)])
    if prefix_kv_cache:
        matrix_flags.append("--prefix-kv-cache")
    if isinstance(max_workers, int) and max_workers >= 1:
        matrix_flags.extend(["--max-workers", str(max_workers)])

    readiness_flags = list(matrix_flags)
    return {
        "eval_truthfulqa": eval_flags,
        "run_cache_profile_matrix": matrix_flags,
        "run_adapter_readiness_workflow": readiness_flags,
    }


def _evidence(
    matrix_report: Mapping[str, Any],
    *,
    matrix_decision: Mapping[str, Any],
    worker_sweep_report: Mapping[str, Any] | None,
    worker_decision: Mapping[str, Any],
    matrix_report_path: str | Path | None,
    worker_sweep_report_path: str | Path | None,
) -> dict[str, Any]:
    matrix_recommended = _recommended_runtime_row(matrix_report, matrix_decision)
    worker_recommended = _mapping(worker_decision.get("recommended"))
    prefix_comparison = _prefix_comparison_for_recommendation(matrix_report, matrix_recommended)
    quality = _quality_signal_summary(
        matrix_report,
        matrix_recommended,
        matrix_report_path=matrix_report_path,
    )
    config = _mapping(matrix_report.get("config"))
    evidence = {
        "matrix_report": None if matrix_report_path is None else str(matrix_report_path),
        "matrix_status": matrix_decision.get("status"),
        "matrix_recommended_cell": matrix_decision.get("recommended_cell"),
        "matrix_recommendation_metric": matrix_decision.get("recommendation_metric"),
        "matrix_candidate_count": matrix_decision.get("candidate_count"),
        "matrix_checked_cell_count": matrix_decision.get("checked_cell_count"),
        "matrix_leaderboard_sort_metric": matrix_report.get("leaderboard_sort_metric"),
        "matrix_wall_clock_seconds": _mapping(matrix_report.get("execution")).get("wall_clock_seconds"),
        "configured_matrix_workers": config.get("max_workers"),
        "length_bucketed_batches": config.get("length_bucketed_batches"),
        "prefix_kv_comparison": prefix_comparison,
        "quality_signal_source": quality["source"],
        "quality_signal_count": quality["count"],
        "worker_sweep_report": None if worker_sweep_report_path is None else str(worker_sweep_report_path),
        "worker_sweep_status": None if worker_sweep_report is None else worker_decision.get("status"),
        "worker_recommended_worker_count": worker_decision.get("recommended_worker_count"),
        "worker_recommended_wall_clock_seconds": worker_recommended.get("wall_clock_seconds"),
        "worker_matrix_report_matches": _worker_matrix_report_matches(
            worker_recommended=worker_recommended,
            matrix_report_path=matrix_report_path,
        ),
    }
    return evidence


def _worker_matrix_report_matches(
    *,
    worker_recommended: Mapping[str, Any],
    matrix_report_path: str | Path | None,
) -> bool | None:
    worker_matrix_report = worker_recommended.get("matrix_report")
    if worker_matrix_report is None or matrix_report_path is None:
        return None
    try:
        return Path(str(worker_matrix_report)).resolve() == Path(str(matrix_report_path)).resolve()
    except OSError:
        return str(worker_matrix_report) == str(matrix_report_path)


def _prefix_comparison_for_recommendation(
    matrix_report: Mapping[str, Any],
    matrix_recommended: Mapping[str, Any],
) -> dict[str, Any] | None:
    comparisons = matrix_report.get("prefix_kv_comparisons")
    if not isinstance(comparisons, Sequence) or isinstance(comparisons, str):
        return None
    layer = matrix_recommended.get("layer")
    batch_size = matrix_recommended.get("batch_size")
    capture = matrix_recommended.get("hidden_state_capture")
    for comparison in comparisons:
        item = _mapping(comparison)
        if (
            item.get("layer") == layer
            and item.get("batch_size") == batch_size
            and item.get("hidden_state_capture") == capture
        ):
            return dict(item)
    return None


def _blocking_reasons(
    *,
    matrix_decision: Mapping[str, Any],
    worker_decision: Mapping[str, Any],
    worker_sweep_report: Mapping[str, Any] | None,
) -> list[str]:
    reasons = []
    for reason in matrix_decision.get("blocking_reasons") or ():
        reasons.append(f"matrix: {reason}")
    if worker_sweep_report is not None:
        for reason in worker_decision.get("blocking_reasons") or ():
            reasons.append(f"worker_sweep: {reason}")
    return reasons


def _combined_status(matrix_status: str, worker_status: str | None) -> str:
    statuses = [matrix_status]
    if worker_status is not None:
        statuses.append(worker_status)
    if any(status == "blocked" for status in statuses):
        return "blocked"
    if any(status == "dry_run" for status in statuses):
        return "needs_evidence"
    if any(status == "no_candidate" for status in statuses):
        return "no_candidate"
    if all(status == "promote" for status in statuses):
        return "promote"
    return "unknown"


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        numeric = float(value)
    else:
        try:
            numeric = float(str(value))
        except (TypeError, ValueError):
            return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object.")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    matrix_report_path = Path(args.matrix_report)
    worker_sweep_report_path = Path(args.worker_sweep_report) if args.worker_sweep_report else None
    report = build_runtime_recommendation(
        _load_json(matrix_report_path),
        worker_sweep_report=None if worker_sweep_report_path is None else _load_json(worker_sweep_report_path),
        matrix_report_path=matrix_report_path,
        worker_sweep_report_path=worker_sweep_report_path,
    )
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output, end="")
    if args.fail_on_blocked and report["status"] != "promote":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a runtime configuration recommendation from benchmark reports"
    )
    parser.add_argument("--matrix-report", required=True,
                        help="cache-profile-matrix-report.json produced by run_cache_profile_matrix.py")
    parser.add_argument("--worker-sweep-report", default=None,
                        help="optional cache-worker-sweep-report.json produced by run_cache_worker_sweep.py")
    parser.add_argument("--output", default=None,
                        help="optional path to write the recommendation JSON")
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless the combined recommendation status is promote")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
