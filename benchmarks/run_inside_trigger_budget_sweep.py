"""Run several triggered INSIDE sampling budgets and compare their costs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_inside_sampling_profile import (  # noqa: E402
    INSIDE_PROFILE_RUN_NAMES,
    InsideSamplingProfileConfig,
    _parse_run_names,
    run_inside_sampling_profile,
)
from eigentruth.registry import build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class TriggerBudgetSpec:
    """One trigger budget for an INSIDE sampling profile run."""

    kind: str
    value: float

    def __post_init__(self) -> None:
        kind = str(self.kind).strip()
        if kind not in {"top_fraction", "threshold"}:
            raise ValueError("trigger budget kind must be 'top_fraction' or 'threshold'.")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("trigger budget value must be finite.")
        if kind == "top_fraction" and not (0.0 < value <= 1.0):
            raise ValueError("top_fraction budgets must be in (0, 1].")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "value", value)

    @property
    def id(self) -> str:
        text = f"{self.value:g}".replace("-", "m").replace(".", "p")
        prefix = "top" if self.kind == "top_fraction" else "threshold"
        return f"{prefix}_{text}"


@dataclass(frozen=True)
class InsideTriggerBudgetSweepConfig:
    """Configuration for a triggered INSIDE budget sweep."""

    output_dir: Path
    trigger_signal: str
    budgets: Sequence[TriggerBudgetSpec]
    model: str = "sshleifer/tiny-gpt2"
    dtype: str = "float32"
    layer: int = -1
    limit: int | None = None
    manifold_questions: int | None = None
    batch_size: int = 4
    max_batch_tokens: int = 0
    max_length: int = 64
    hidden_state_capture: str = "outputs"
    progress_every: int = 0
    offline: bool = True
    length_bucketed_batches: bool = True
    python_executable: str = sys.executable
    inside_samples: int = 5
    inside_batch_size: int = 1
    inside_max_new_tokens: int = 12
    inside_temperature: float = 0.7
    inside_top_p: float = 0.9
    inside_pooling: str = "last"
    inside_embedding_threshold: float = 0.90
    inside_min_samples: int = 2
    inside_sample_step: int = 1
    inside_stability_delta: float = 0.05
    inside_selfcheck_min_overlap: float = 0.65
    inside_selfcheck_support_threshold: float = 0.60
    inside_selfcheck_refute_threshold: float = 0.50
    adaptive_max_sample_ratio: float = 1.0
    adaptive_selfcheck_max_sample_ratio: float = 1.0
    max_inside_generation_seconds_ratio: float | None = None
    run_names: Sequence[str] = INSIDE_PROFILE_RUN_NAMES
    reference_report_path: Path | None = None
    shared_cache_dir: Path | None = None
    eval_reps_cache_shard_size: int = 0
    refresh_shared_caches: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        trigger_signal = str(self.trigger_signal).strip()
        if not trigger_signal:
            raise ValueError("trigger_signal must not be empty.")
        budgets = tuple(self.budgets)
        if not budgets:
            raise ValueError("at least one trigger budget is required.")
        ids = tuple(budget.id for budget in budgets)
        if len(ids) != len(set(ids)):
            raise ValueError("trigger budgets must not contain duplicate ids.")
        if self.reference_report_path is not None:
            object.__setattr__(self, "reference_report_path", Path(self.reference_report_path))
        if self.shared_cache_dir is not None:
            object.__setattr__(self, "shared_cache_dir", Path(self.shared_cache_dir))
        if int(self.eval_reps_cache_shard_size) < 0:
            raise ValueError("eval_reps_cache_shard_size must be >=0.")
        if int(self.eval_reps_cache_shard_size) > 0 and self.shared_cache_dir is None:
            raise ValueError("eval_reps_cache_shard_size requires shared_cache_dir.")
        object.__setattr__(self, "trigger_signal", trigger_signal)
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(self, "run_names", _parse_run_names(",".join(self.run_names)))

    @property
    def report_path(self) -> Path:
        return self.output_dir / "inside-trigger-budget-sweep.json"

    @property
    def artifact_manifest(self) -> Path:
        return self.output_dir / "artifact-manifest.json"


def run_inside_trigger_budget_sweep(
    config: InsideTriggerBudgetSweepConfig,
    *,
    clean: bool = False,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Run each trigger budget and write a sweep-level comparison report."""
    if clean and config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    child_payloads = {}
    for budget in config.budgets:
        child_config = _profile_config_for_budget(config, budget)
        child_payloads[budget.id] = run_inside_sampling_profile(
            child_config,
            clean=False,
            dry_run=dry_run,
            skip_existing=skip_existing,
        )

    if dry_run:
        report = _dry_run_report(config, child_payloads)
    else:
        report = _budget_sweep_report(config, child_payloads)
    report["artifact_manifest"] = str(config.artifact_manifest)
    config.report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = _write_artifact_manifest(config, report)
    report["artifact_manifest_summary"] = manifest["summary"]
    return report


def _profile_config_for_budget(
    config: InsideTriggerBudgetSweepConfig,
    budget: TriggerBudgetSpec,
) -> InsideSamplingProfileConfig:
    return InsideSamplingProfileConfig(
        output_dir=config.output_dir / budget.id,
        model=config.model,
        dtype=config.dtype,
        layer=config.layer,
        limit=config.limit,
        manifold_questions=config.manifold_questions,
        batch_size=config.batch_size,
        max_batch_tokens=config.max_batch_tokens,
        max_length=config.max_length,
        hidden_state_capture=config.hidden_state_capture,
        progress_every=config.progress_every,
        offline=config.offline,
        length_bucketed_batches=config.length_bucketed_batches,
        python_executable=config.python_executable,
        inside_samples=config.inside_samples,
        inside_batch_size=config.inside_batch_size,
        inside_max_new_tokens=config.inside_max_new_tokens,
        inside_temperature=config.inside_temperature,
        inside_top_p=config.inside_top_p,
        inside_pooling=config.inside_pooling,
        inside_embedding_threshold=config.inside_embedding_threshold,
        inside_min_samples=config.inside_min_samples,
        inside_sample_step=config.inside_sample_step,
        inside_stability_delta=config.inside_stability_delta,
        inside_selfcheck_min_overlap=config.inside_selfcheck_min_overlap,
        inside_selfcheck_support_threshold=config.inside_selfcheck_support_threshold,
        inside_selfcheck_refute_threshold=config.inside_selfcheck_refute_threshold,
        inside_trigger_signal=config.trigger_signal,
        inside_trigger_threshold=budget.value if budget.kind == "threshold" else None,
        inside_trigger_top_fraction=budget.value if budget.kind == "top_fraction" else None,
        adaptive_max_sample_ratio=config.adaptive_max_sample_ratio,
        adaptive_selfcheck_max_sample_ratio=config.adaptive_selfcheck_max_sample_ratio,
        max_inside_generation_seconds_ratio=config.max_inside_generation_seconds_ratio,
        run_names=config.run_names,
        statement_encoding_cache_path=_shared_cache_path(config, "statement-encodings.json"),
        layer_stats_cache_path=_shared_cache_path(config, "layer-stats.pt"),
        eval_reps_cache_path=_shared_cache_path(config, "eval-reps-cache"),
        eval_reps_cache_shard_size=config.eval_reps_cache_shard_size,
        refresh_shared_caches=bool(config.refresh_shared_caches and budget.id == config.budgets[0].id),
    )


def _dry_run_report(
    config: InsideTriggerBudgetSweepConfig,
    child_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow": "inside_trigger_budget_sweep",
        "dry_run": True,
        "config": _config_payload(config),
        "budgets": {
            budget_id: {
                "profile_output_dir": payload.get("output_dir"),
                "commands": payload.get("commands"),
                "caches": payload.get("caches"),
            }
            for budget_id, payload in child_payloads.items()
        },
        "leaderboard": [],
        "recommendation": None,
        "quality_balanced_recommendation": None,
    }


def _budget_sweep_report(
    config: InsideTriggerBudgetSweepConfig,
    child_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reference = _reference_payload(config.reference_report_path)
    rows = []
    budget_payloads = {}
    for budget in config.budgets:
        payload = child_payloads[budget.id]
        comparison_path = Path(str(payload["comparison_report"]))
        comparison = _read_json(comparison_path)
        recommended_run = str(dict(comparison.get("recommendation") or {}).get("recommended_run") or "")
        runs = dict(comparison.get("runs") or {})
        recommended_row = dict(runs.get(recommended_run) or {})
        result_payload = _read_optional_json(recommended_row.get("result_path"))
        auroc = dict(result_payload.get("auroc") or {}) if isinstance(result_payload, Mapping) else {}
        row = _budget_row(
            budget,
            comparison_path=comparison_path,
            recommended_run=recommended_run,
            recommended_row=recommended_row,
            auroc=auroc,
            reference=reference,
        )
        rows.append(row)
        budget_payloads[budget.id] = {
            "budget": {"kind": budget.kind, "value": budget.value},
            "profile_output_dir": payload.get("output_dir"),
            "comparison_report": payload.get("comparison_report"),
            "artifact_manifest": payload.get("artifact_manifest"),
            "caches": payload.get("caches"),
            "sample_efficiency_gate": comparison.get("sample_efficiency_gate"),
            "recommendation": comparison.get("recommendation"),
        }
    leaderboard = sorted(rows, key=_budget_sort_key)
    recommendation = {
        "budget_id": leaderboard[0]["budget_id"],
        "recommended_run": leaderboard[0]["recommended_run"],
        "reason": "lowest_total_generated_samples_then_inside_generation_seconds",
    } if leaderboard else None
    return {
        "schema_version": 1,
        "workflow": "inside_trigger_budget_sweep",
        "dry_run": False,
        "config": _config_payload(config),
        "reference": reference,
        "budgets": budget_payloads,
        "leaderboard": leaderboard,
        "recommendation": recommendation,
        "quality_balanced_recommendation": _quality_balanced_recommendation(leaderboard),
    }


def _budget_row(
    budget: TriggerBudgetSpec,
    *,
    comparison_path: Path,
    recommended_run: str,
    recommended_row: Mapping[str, Any],
    auroc: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    total_generated_samples = _optional_int(recommended_row.get("total_generated_samples"))
    inside_generation_seconds = _optional_float(recommended_row.get("inside_generation_seconds"))
    reference_samples = None if reference is None else _optional_int(reference.get("total_generated_samples"))
    reference_seconds = None if reference is None else _optional_float(reference.get("inside_generation_seconds"))
    return {
        "budget_id": budget.id,
        "budget_kind": budget.kind,
        "budget_value": budget.value,
        "comparison_report": str(comparison_path),
        "recommended_run": recommended_run,
        "sampled": _optional_int(recommended_row.get("sampled")),
        "skipped_by_trigger": _optional_int(recommended_row.get("skipped_by_trigger")),
        "total_generated_samples": total_generated_samples,
        "mean_samples_per_record": _optional_float(recommended_row.get("mean_samples_per_record")),
        "inside_generation_seconds": inside_generation_seconds,
        "sample_count_ratio_to_budget_fixed": _optional_float(
            recommended_row.get("sample_count_ratio_to_baseline")
        ),
        "inside_generation_seconds_ratio_to_budget_fixed": _optional_float(
            recommended_row.get("inside_generation_seconds_ratio_to_baseline")
        ),
        "sample_count_ratio_to_reference": _ratio(total_generated_samples, reference_samples),
        "inside_generation_seconds_ratio_to_reference": _ratio(inside_generation_seconds, reference_seconds),
        "inside_auroc": {
            key: _optional_float(value)
            for key, value in auroc.items()
            if str(key).startswith("inside_")
        },
    }


def _reference_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    report = _read_json(path)
    runs = dict(report.get("runs") or {})
    fixed = dict(runs.get("fixed") or {})
    return {
        "report_path": str(path),
        "run": "fixed",
        "total_generated_samples": _optional_int(fixed.get("total_generated_samples")),
        "inside_generation_seconds": _optional_float(fixed.get("inside_generation_seconds")),
    }


def _budget_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    samples = row.get("total_generated_samples")
    seconds = row.get("inside_generation_seconds")
    return (
        samples is None,
        float("inf") if samples is None else float(samples),
        seconds is None,
        float("inf") if seconds is None else float(seconds),
        row.get("budget_id"),
    )


def _quality_balanced_recommendation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Recommend the cheapest budget close to the best INSIDE quality signal."""
    candidates = []
    for row in rows:
        quality_metric, quality_value = _inside_quality_signal(row.get("inside_auroc"))
        if quality_metric is None or quality_value is None:
            continue
        cost_value = _optional_float(row.get("inside_generation_seconds_ratio_to_reference"))
        cost_metric = "inside_generation_seconds_ratio_to_reference"
        if cost_value is None:
            cost_value = _optional_float(row.get("inside_generation_seconds"))
            cost_metric = "inside_generation_seconds"
        if cost_value is None:
            cost_value = _optional_float(row.get("total_generated_samples"))
            cost_metric = "total_generated_samples"
        if cost_value is None:
            continue
        candidates.append((row, quality_metric, quality_value, cost_metric, cost_value))
    if not candidates:
        return None

    quality_tolerance = 0.02
    best_quality = max(item[2] for item in candidates)
    eligible = [item for item in candidates if item[2] >= best_quality - quality_tolerance]
    selected = min(
        eligible,
        key=lambda item: (
            item[4],
            _optional_int(item[0].get("total_generated_samples")) is None,
            _optional_int(item[0].get("total_generated_samples")) or sys.maxsize,
            str(item[0].get("budget_id")),
        ),
    )
    row, quality_metric, quality_value, cost_metric, cost_value = selected
    return {
        "budget_id": row.get("budget_id"),
        "recommended_run": row.get("recommended_run"),
        "reason": "lowest_cost_within_inside_quality_tolerance",
        "quality_metric": quality_metric,
        "quality_value": quality_value,
        "best_quality_value": best_quality,
        "quality_tolerance": quality_tolerance,
        "cost_metric": cost_metric,
        "cost_value": cost_value,
    }


def _inside_quality_signal(inside_auroc: Any) -> tuple[str | None, float | None]:
    if not isinstance(inside_auroc, Mapping):
        return None, None
    preferred = (
        "inside_semantic_entropy",
        "inside_embedding_entropy",
        "inside_eigenscore",
    )
    for metric in preferred:
        value = _optional_float(inside_auroc.get(metric))
        if value is not None:
            return metric, value
    finite_values = [
        (str(metric), value)
        for metric, raw_value in inside_auroc.items()
        if str(metric).startswith("inside_")
        for value in [_optional_float(raw_value)]
        if value is not None
    ]
    if not finite_values:
        return None, None
    return max(finite_values, key=lambda item: (item[1], item[0]))


def _write_artifact_manifest(
    config: InsideTriggerBudgetSweepConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts: dict[str, str | Path | None] = {"sweep_report": config.report_path}
    for budget_id, payload in dict(report.get("budgets") or {}).items():
        if isinstance(payload, Mapping):
            artifacts[f"budgets.{budget_id}.profile_manifest"] = payload.get("artifact_manifest")
            artifacts[f"budgets.{budget_id}.comparison_report"] = payload.get("comparison_report")
    manifest = build_artifact_manifest(
        artifacts,
        root=config.output_dir,
        metadata={
            "runner": "run_inside_trigger_budget_sweep",
            "model": config.model,
            "dtype": config.dtype,
            "layer": config.layer,
            "offline": config.offline,
            "trigger_signal": config.trigger_signal,
            "budgets": tuple({"kind": budget.kind, "value": budget.value} for budget in config.budgets),
            "run_names": tuple(config.run_names),
            "reference_report": None if config.reference_report_path is None else str(config.reference_report_path),
            "shared_cache_dir": None if config.shared_cache_dir is None else str(config.shared_cache_dir),
            "eval_reps_cache_shard_size": int(config.eval_reps_cache_shard_size),
            "refresh_shared_caches": bool(config.refresh_shared_caches),
            "dry_run": bool(report.get("dry_run")),
        },
    )
    config.artifact_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _config_payload(config: InsideTriggerBudgetSweepConfig) -> dict[str, Any]:
    return {
        "output_dir": str(config.output_dir),
        "model": config.model,
        "dtype": config.dtype,
        "layer": config.layer,
        "limit": config.limit,
        "manifold_questions": config.manifold_questions,
        "batch_size": config.batch_size,
        "max_batch_tokens": config.max_batch_tokens,
        "max_length": config.max_length,
        "hidden_state_capture": config.hidden_state_capture,
        "offline": config.offline,
        "trigger_signal": config.trigger_signal,
        "budgets": tuple({"kind": budget.kind, "value": budget.value, "id": budget.id} for budget in config.budgets),
        "inside_samples": config.inside_samples,
        "inside_batch_size": config.inside_batch_size,
        "inside_max_new_tokens": config.inside_max_new_tokens,
        "run_names": tuple(config.run_names),
        "reference_report": None if config.reference_report_path is None else str(config.reference_report_path),
        "shared_cache_dir": None if config.shared_cache_dir is None else str(config.shared_cache_dir),
        "eval_reps_cache_shard_size": int(config.eval_reps_cache_shard_size),
        "refresh_shared_caches": bool(config.refresh_shared_caches),
    }


def _shared_cache_path(config: InsideTriggerBudgetSweepConfig, name: str) -> Path | None:
    if config.shared_cache_dir is None:
        return None
    return config.shared_cache_dir / name


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _read_optional_json(path: Any) -> dict[str, Any]:
    if not path:
        return {}
    try:
        return _read_json(Path(str(path)))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _ratio(value: int | float | None, baseline: int | float | None) -> float | None:
    if value is None or baseline is None or float(baseline) == 0.0:
        return None
    result = float(value) / float(baseline)
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _parse_float_list(value: str | None, *, flag: str) -> tuple[float, ...]:
    if value is None:
        return ()
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError(f"{flag} must not be empty when provided.")
    if any(not math.isfinite(item) for item in values):
        raise ValueError(f"{flag} must contain finite numbers.")
    return values


def _parse_budgets(top_fractions: str | None, thresholds: str | None) -> tuple[TriggerBudgetSpec, ...]:
    budgets = [
        TriggerBudgetSpec("top_fraction", value)
        for value in _parse_float_list(top_fractions, flag="--top-fractions")
    ]
    budgets.extend(
        TriggerBudgetSpec("threshold", value)
        for value in _parse_float_list(thresholds, flag="--thresholds")
    )
    if not budgets:
        raise ValueError("provide at least one --top-fractions or --thresholds budget.")
    return tuple(budgets)


def _config_from_args(args: argparse.Namespace) -> InsideTriggerBudgetSweepConfig:
    return InsideTriggerBudgetSweepConfig(
        output_dir=Path(args.output_dir),
        trigger_signal=args.trigger_signal,
        budgets=_parse_budgets(args.top_fractions, args.thresholds),
        model=args.model,
        dtype=args.dtype,
        layer=args.layer,
        limit=args.limit,
        manifold_questions=args.manifold_questions,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_length=args.max_length,
        hidden_state_capture=args.hidden_state_capture,
        progress_every=args.progress_every,
        offline=not args.real_truthfulqa,
        length_bucketed_batches=not args.no_length_bucketed_batches,
        python_executable=args.python,
        inside_samples=args.inside_samples,
        inside_batch_size=args.inside_batch_size,
        inside_max_new_tokens=args.inside_max_new_tokens,
        inside_temperature=args.inside_temperature,
        inside_top_p=args.inside_top_p,
        inside_pooling=args.inside_pooling,
        inside_embedding_threshold=args.inside_embedding_threshold,
        inside_min_samples=args.inside_min_samples,
        inside_sample_step=args.inside_sample_step,
        inside_stability_delta=args.inside_stability_delta,
        inside_selfcheck_min_overlap=args.inside_selfcheck_min_overlap,
        inside_selfcheck_support_threshold=args.inside_selfcheck_support_threshold,
        inside_selfcheck_refute_threshold=args.inside_selfcheck_refute_threshold,
        adaptive_max_sample_ratio=args.adaptive_max_sample_ratio,
        adaptive_selfcheck_max_sample_ratio=args.adaptive_selfcheck_max_sample_ratio,
        max_inside_generation_seconds_ratio=args.max_inside_generation_seconds_ratio,
        run_names=_parse_run_names(args.runs),
        reference_report_path=Path(args.reference_report) if args.reference_report else None,
        shared_cache_dir=Path(args.shared_cache_dir) if args.shared_cache_dir else None,
        eval_reps_cache_shard_size=args.eval_reps_cache_shard_size,
        refresh_shared_caches=args.refresh_shared_caches,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = run_inside_trigger_budget_sweep(
        _config_from_args(args),
        clean=bool(args.clean),
        dry_run=bool(args.dry_run),
        skip_existing=bool(args.skip_existing),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a triggered INSIDE budget sweep")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trigger-signal", required=True)
    parser.add_argument("--top-fractions", default=None,
                        help="comma-list of top-fraction trigger budgets, e.g. 0.1,0.2,0.3")
    parser.add_argument("--thresholds", default=None,
                        help="comma-list of trigger threshold budgets")
    parser.add_argument("--reference-report", default=None,
                        help="optional full-sample inside-sampling-profile-comparison.json for ratio-to-full metrics")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--manifold-questions", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--hidden-state-capture", default="outputs")
    parser.add_argument("--progress-every", type=int, default=0)
    truthfulqa_mode = parser.add_mutually_exclusive_group()
    truthfulqa_mode.add_argument("--offline", action="store_true",
                                 help="use the built-in offline fixture; this is the default")
    truthfulqa_mode.add_argument("--real-truthfulqa", action="store_true")
    parser.add_argument("--no-length-bucketed-batches", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--inside-samples", type=int, default=5)
    parser.add_argument("--inside-batch-size", type=int, default=1)
    parser.add_argument("--inside-max-new-tokens", type=int, default=12)
    parser.add_argument("--inside-temperature", type=float, default=0.7)
    parser.add_argument("--inside-top-p", type=float, default=0.9)
    parser.add_argument("--inside-pooling", default="last", choices=("last", "mean"))
    parser.add_argument("--inside-embedding-threshold", type=float, default=0.90)
    parser.add_argument("--inside-min-samples", type=int, default=2)
    parser.add_argument("--inside-sample-step", type=int, default=1)
    parser.add_argument("--inside-stability-delta", type=float, default=0.05)
    parser.add_argument("--inside-selfcheck-min-overlap", type=float, default=0.65)
    parser.add_argument("--inside-selfcheck-support-threshold", type=float, default=0.60)
    parser.add_argument("--inside-selfcheck-refute-threshold", type=float, default=0.50)
    parser.add_argument("--adaptive-max-sample-ratio", type=float, default=1.0)
    parser.add_argument("--adaptive-selfcheck-max-sample-ratio", type=float, default=1.0)
    parser.add_argument("--max-inside-generation-seconds-ratio", type=float, default=None)
    parser.add_argument("--shared-cache-dir", default=None,
                        help="optional shared cache directory reused across all budget/profile runs")
    parser.add_argument("--eval-reps-cache-shard-size", type=int, default=0,
                        help="write the shared eval-reps cache as shards with this many records per shard")
    parser.add_argument("--refresh-shared-caches", action="store_true",
                        help="refresh shared caches on the first run that uses them; later runs load them")
    parser.add_argument("--runs", default="fixed,adaptive,adaptive_selfcheck")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
