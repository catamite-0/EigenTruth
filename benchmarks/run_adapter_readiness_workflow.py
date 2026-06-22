"""Run a combined adapter quality and performance readiness workflow.

This workflow composes the deterministic adapter-family matrix with the
same-machine cache-profile matrix. It is intended as the last local gate before
promoting a verifier adapter route or changing benchmark defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_adapter_family_matrix import (  # noqa: E402
    AdapterFamilyMatrixConfig,
    run_adapter_family_matrix,
)
from benchmarks.run_cache_profile_matrix import MATRIX_MODES, CacheProfileMatrixConfig, run_matrix  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class AdapterReadinessWorkflowConfig:
    """Configuration for the combined adapter readiness workflow."""

    output_dir: Path
    readiness_report_path: Path | None = None
    compact_json: bool = False
    alpha: float = 0.20
    n_records: int = 8
    signal: str = "truth_proj"
    min_decision_accuracy: float = 1.0
    max_false_supported_rate: float = 0.0
    min_false_refuted_rate: float = 1.0
    max_mean_duration_seconds: float = 1.0
    max_p99_duration_seconds: float = 1.0
    max_max_duration_seconds: float = 1.0
    max_mean_attempted_route_count: float = 1.1
    max_retrieval_use_rate: float = 0.0
    model: str = "sshleifer/tiny-gpt2"
    dtype: str = "float32"
    layers: Sequence[int] = (-1,)
    batch_sizes: Sequence[int] = (4,)
    hidden_state_captures: Sequence[str] = ("outputs",)
    limit: int | None = None
    manifold_questions: int | None = None
    max_length: int = 64
    prefix_kv_cache: bool = False
    eval_reps_cache_shard_size: int = 4
    cached_max_total_ratio: float = 1.10
    cache_only_max_total_ratio: float = 0.35
    python_executable: str = sys.executable
    progress_every: int = 0
    length_bucketed_batches: bool = True
    offline: bool = True
    shared_cache_dir: Path | None = None
    matrix_mode: str = "triplet"
    performance_clean: bool = False
    performance_dry_run: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.readiness_report_path is not None:
            object.__setattr__(self, "readiness_report_path", Path(self.readiness_report_path))
        if self.shared_cache_dir is not None:
            object.__setattr__(self, "shared_cache_dir", Path(self.shared_cache_dir))
        object.__setattr__(self, "layers", tuple(int(layer) for layer in self.layers))
        object.__setattr__(self, "batch_sizes", tuple(int(batch_size) for batch_size in self.batch_sizes))
        object.__setattr__(
            self,
            "hidden_state_captures",
            tuple(str(capture) for capture in self.hidden_state_captures),
        )

    @property
    def adapter_family_dir(self) -> Path:
        return self.output_dir / "adapter-family"

    @property
    def performance_matrix_dir(self) -> Path:
        return self.output_dir / "cache-profile-matrix"

    @property
    def report_path(self) -> Path:
        return self.readiness_report_path or self.output_dir / "adapter-readiness-report.json"

    @property
    def artifact_manifest_path(self) -> Path:
        return self.output_dir / "artifact-manifest.json"


def run_adapter_readiness_workflow(config: AdapterReadinessWorkflowConfig) -> dict[str, Any]:
    """Run adapter-family and performance gates, then return readiness status."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_report_path = config.output_dir / "adapter-family-matrix.json"
    performance_report_path = config.performance_matrix_dir / "cache-profile-matrix-report.json"
    adapter_report = run_adapter_family_matrix(
        AdapterFamilyMatrixConfig(
            output_dir=config.adapter_family_dir,
            matrix_report_path=adapter_report_path,
            alpha=config.alpha,
            n_records=config.n_records,
            signal=config.signal,
            compact_json=config.compact_json,
            min_decision_accuracy=config.min_decision_accuracy,
            max_false_supported_rate=config.max_false_supported_rate,
            min_false_refuted_rate=config.min_false_refuted_rate,
            max_mean_duration_seconds=config.max_mean_duration_seconds,
            max_p99_duration_seconds=config.max_p99_duration_seconds,
            max_max_duration_seconds=config.max_max_duration_seconds,
            max_mean_attempted_route_count=config.max_mean_attempted_route_count,
            max_retrieval_use_rate=config.max_retrieval_use_rate,
        )
    )
    performance_report = run_matrix(
        CacheProfileMatrixConfig(
            output_dir=config.performance_matrix_dir,
            model=config.model,
            dtype=config.dtype,
            layers=config.layers,
            batch_sizes=config.batch_sizes,
            hidden_state_captures=config.hidden_state_captures,
            limit=config.limit,
            manifold_questions=config.manifold_questions,
            max_length=config.max_length,
            prefix_kv_cache=config.prefix_kv_cache,
            eval_reps_cache_shard_size=config.eval_reps_cache_shard_size,
            cached_max_total_ratio=config.cached_max_total_ratio,
            cache_only_max_total_ratio=config.cache_only_max_total_ratio,
            python_executable=config.python_executable,
            progress_every=config.progress_every,
            length_bucketed_batches=config.length_bucketed_batches,
            offline=config.offline,
            shared_cache_dir=config.shared_cache_dir,
            matrix_mode=config.matrix_mode,
        ),
        clean=config.performance_clean,
        dry_run=config.performance_dry_run,
    )
    decision = build_readiness_decision(adapter_report, performance_report)
    report = {
        "schema_version": 1,
        "workflow": "adapter_readiness_workflow",
        "adapter_family_matrix_path": str(adapter_report_path),
        "performance_matrix_path": str(performance_report_path),
        "artifact_manifest": str(config.artifact_manifest_path),
        "adapter_family_matrix": adapter_report,
        "performance_matrix": performance_report,
        "readiness_decision": decision,
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        _json_text(report, compact=config.compact_json, sort_keys=True),
        encoding="utf-8",
    )
    _write_artifact_manifest(config, report)
    return report


def build_readiness_decision(
    adapter_report: Mapping[str, Any],
    performance_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the final fail-closed readiness decision."""
    adapter_decision = dict(adapter_report.get("promotion_decision") or {})
    performance_decision = dict(performance_report.get("matrix_decision") or {})
    adapter_status = str(adapter_decision.get("status"))
    performance_status = str(performance_decision.get("status"))
    blocking_reasons = []
    if adapter_status != "promote":
        blocking_reasons.append("adapter-family quality gate did not promote")
    if performance_status == "dry_run":
        blocking_reasons.append("performance matrix was dry-run only; run real profiles before promotion")
    elif performance_status != "promote":
        blocking_reasons.append("performance matrix decision did not promote")

    if adapter_status == "promote" and performance_status == "promote":
        status = "promote"
    elif adapter_status == "promote" and performance_status == "dry_run":
        status = "needs_performance_evidence"
    else:
        status = "blocked"

    return {
        "status": status,
        "adapter_family_status": adapter_status,
        "performance_status": performance_status,
        "recommended_route": adapter_decision.get("recommended_route"),
        "recommended_performance_cell": performance_decision.get("recommended_cell"),
        "adapter_family_promoted": adapter_status == "promote",
        "performance_promoted": performance_status == "promote",
        "blocking_reasons": tuple(blocking_reasons),
    }


def _write_artifact_manifest(
    config: AdapterReadinessWorkflowConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    adapter_report = report.get("adapter_family_matrix", {})
    performance_report = report.get("performance_matrix", {})
    artifacts = {
        "readiness_report": config.report_path,
        "adapter_family_matrix": report.get("adapter_family_matrix_path"),
        "adapter_family_route_comparison": (
            adapter_report.get("route_comparison_path") if isinstance(adapter_report, Mapping) else None
        ),
        "performance_matrix_report": (
            performance_report.get("report_path") if isinstance(performance_report, Mapping) else None
        ),
        "performance_matrix_manifest": (
            performance_report.get("artifact_manifest") if isinstance(performance_report, Mapping) else None
        ),
    }
    decision = dict(report.get("readiness_decision") or {})
    manifest = build_artifact_manifest(
        artifacts,
        root=config.output_dir,
        metadata={
            "runner": "run_adapter_readiness_workflow",
            "model": config.model,
            "dtype": config.dtype,
            "layers": tuple(config.layers),
            "batch_sizes": tuple(config.batch_sizes),
            "hidden_state_captures": tuple(config.hidden_state_captures),
            "prefix_kv_cache": config.prefix_kv_cache,
            "offline": config.offline,
            "matrix_mode": config.matrix_mode,
            "performance_dry_run": config.performance_dry_run,
            "readiness_status": decision.get("status"),
            "adapter_family_status": decision.get("adapter_family_status"),
            "performance_status": decision.get("performance_status"),
            "recommended_route": decision.get("recommended_route"),
            "recommended_performance_cell": decision.get("recommended_performance_cell"),
        },
    )
    config.artifact_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _json_text(payload: Mapping[str, Any], *, compact: bool, sort_keys: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=sort_keys, separators=(",", ":")) + "\n"
    return json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n"


def _parse_int_list(value: str, *, flag: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError(f"{flag} must not be empty.")
    return values


def _parse_str_list(value: str, *, flag: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError(f"{flag} must not be empty.")
    return values


def _config_from_args(args: argparse.Namespace) -> AdapterReadinessWorkflowConfig:
    return AdapterReadinessWorkflowConfig(
        output_dir=Path(args.output_dir),
        readiness_report_path=Path(args.json) if args.json else None,
        compact_json=bool(args.compact_json),
        alpha=args.alpha,
        n_records=args.n_records,
        signal=args.signal,
        min_decision_accuracy=args.min_decision_accuracy,
        max_false_supported_rate=args.max_false_supported_rate,
        min_false_refuted_rate=args.min_false_refuted_rate,
        max_mean_duration_seconds=args.max_mean_duration_seconds,
        max_p99_duration_seconds=args.max_p99_duration_seconds,
        max_max_duration_seconds=args.max_max_duration_seconds,
        max_mean_attempted_route_count=args.max_mean_attempted_route_count,
        max_retrieval_use_rate=args.max_retrieval_use_rate,
        model=args.model,
        dtype=args.dtype,
        layers=_parse_int_list(args.layers, flag="--layers"),
        batch_sizes=_parse_int_list(args.batch_sizes, flag="--batch-sizes"),
        hidden_state_captures=_parse_str_list(args.hidden_state_captures, flag="--hidden-state-captures"),
        limit=args.limit,
        manifold_questions=args.manifold_questions,
        max_length=args.max_length,
        prefix_kv_cache=args.prefix_kv_cache,
        eval_reps_cache_shard_size=args.eval_reps_cache_shard_size,
        cached_max_total_ratio=args.cached_max_total_ratio,
        cache_only_max_total_ratio=args.cache_only_max_total_ratio,
        python_executable=args.python,
        progress_every=args.progress_every,
        length_bucketed_batches=not args.no_length_bucketed_batches,
        offline=not args.real_truthfulqa,
        shared_cache_dir=Path(args.shared_cache_dir) if args.shared_cache_dir else None,
        matrix_mode=args.matrix_mode,
        performance_clean=bool(args.performance_clean),
        performance_dry_run=bool(args.performance_dry_run),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = run_adapter_readiness_workflow(_config_from_args(args))
    decision = report["readiness_decision"]
    print(
        "adapter_readiness="
        f"{decision['status']} "
        f"route={decision.get('recommended_route')} "
        f"performance_cell={decision.get('recommended_performance_cell')}"
    )
    if args.fail_on_blocked and decision["status"] != "promote":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run adapter quality and performance readiness gates")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None, help="optional readiness report output path")
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--n-records", type=int, default=8)
    parser.add_argument("--signal", default="truth_proj")
    parser.add_argument("--min-decision-accuracy", type=float, default=1.0)
    parser.add_argument("--max-false-supported-rate", type=float, default=0.0)
    parser.add_argument("--min-false-refuted-rate", type=float, default=1.0)
    parser.add_argument("--max-mean-duration-seconds", type=float, default=1.0)
    parser.add_argument("--max-p99-duration-seconds", type=float, default=1.0)
    parser.add_argument("--max-max-duration-seconds", type=float, default=1.0)
    parser.add_argument("--max-mean-attempted-route-count", type=float, default=1.1)
    parser.add_argument("--max-retrieval-use-rate", type=float, default=0.0)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--layers", default="-1")
    parser.add_argument("--batch-sizes", default="4")
    parser.add_argument("--hidden-state-captures", default="outputs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--manifold-questions", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--prefix-kv-cache", action="store_true",
                        help="pass --prefix-kv-cache through to non-cache-only performance eval runs")
    parser.add_argument("--eval-reps-cache-shard-size", type=int, default=4)
    parser.add_argument("--cached-max-total-ratio", type=float, default=1.10)
    parser.add_argument("--cache-only-max-total-ratio", type=float, default=0.35)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-length-bucketed-batches", action="store_true")
    parser.add_argument("--real-truthfulqa", action="store_true")
    parser.add_argument("--shared-cache-dir", default=None)
    parser.add_argument("--matrix-mode", default="triplet", choices=MATRIX_MODES)
    parser.add_argument("--performance-clean", action="store_true")
    parser.add_argument("--performance-dry-run", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless readiness_decision.status is promote")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
