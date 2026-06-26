"""Run adapter readiness and register its verified manifest as a baseline."""

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

from benchmarks.promote_artifact_manifest import promote_artifact_manifest  # noqa: E402
from benchmarks.recommend_runtime_config import INSIDE_TRIGGER_BUDGET_POLICIES  # noqa: E402
from benchmarks.run_adapter_readiness_workflow import (  # noqa: E402
    AdapterReadinessWorkflowConfig,
    _parse_int_list,
    _parse_non_negative_float,
    _parse_str_list,
    run_adapter_readiness_workflow,
    state_transition_world_model_evidence,
)
from benchmarks.run_cache_profile_matrix import (  # noqa: E402
    MATRIX_MODES,
    _parse_max_batch_token_budgets,
    _parse_prefix_kv_cache_modes,
)


@dataclass(frozen=True)
class AdapterReadinessRegistryWorkflowConfig:
    """Configuration for registering a promoted adapter readiness manifest."""

    readiness: AdapterReadinessWorkflowConfig
    registry_path: Path
    name: str
    version: str
    workflow_report_path: Path | None = None
    verification_report_path: Path | None = None
    promotion_metadata: Mapping[str, Any] | None = None
    allow_non_promote: bool = False
    allow_promotion_failures: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "registry_path", Path(self.registry_path))
        if self.workflow_report_path is not None:
            object.__setattr__(self, "workflow_report_path", Path(self.workflow_report_path))
        if self.verification_report_path is not None:
            object.__setattr__(self, "verification_report_path", Path(self.verification_report_path))

    @property
    def report_path(self) -> Path:
        return self.workflow_report_path or self.readiness.output_dir / "adapter-readiness-registry-workflow.json"

    @property
    def verification_path(self) -> Path:
        return self.verification_report_path or self.readiness.output_dir / "manifest-verification.json"


def run_adapter_readiness_registry_workflow(
    config: AdapterReadinessRegistryWorkflowConfig,
) -> dict[str, Any]:
    """Run readiness, verify/promote its manifest when eligible, and write a workflow report."""
    readiness_report = run_adapter_readiness_workflow(config.readiness)
    readiness_decision = dict(readiness_report.get("readiness_decision") or {})
    readiness_status = str(readiness_decision.get("status"))
    promotion = None
    blocking_reasons = []
    if readiness_status != "promote":
        blocking_reasons.append("adapter readiness decision did not promote")

    if readiness_status == "promote" or config.allow_non_promote:
        promotion = promote_artifact_manifest(
            manifest_path=readiness_report["artifact_manifest"],
            registry_path=config.registry_path,
            name=config.name,
            version=config.version,
            verification_report_path=config.verification_path,
            recursive=True,
            allow_failures=config.allow_promotion_failures,
            metadata=_promotion_metadata(config, readiness_report),
        )
        if not dict(promotion.get("verification") or {}).get("passed", False):
            blocking_reasons.append("readiness manifest verification did not pass")

    decision = _registry_workflow_decision(
        readiness_status=readiness_status,
        promotion=promotion,
        blocking_reasons=blocking_reasons,
    )
    payload = {
        "schema_version": 1,
        "workflow": "adapter_readiness_registry_workflow",
        "config": {
            "output_dir": str(config.readiness.output_dir),
            "registry": str(config.registry_path),
            "name": config.name,
            "version": config.version,
            "allow_non_promote": config.allow_non_promote,
            "allow_promotion_failures": config.allow_promotion_failures,
        },
        "readiness": readiness_report,
        "promotion": promotion,
        "decision": decision,
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _registry_workflow_decision(
    *,
    readiness_status: str,
    promotion: Mapping[str, Any] | None,
    blocking_reasons: Sequence[str],
) -> dict[str, Any]:
    verification = {} if promotion is None else dict(promotion.get("verification") or {})
    verified = bool(verification.get("passed", False))
    status = "promote" if readiness_status == "promote" and verified else "blocked"
    return {
        "status": status,
        "readiness_status": readiness_status,
        "manifest_promoted": promotion is not None,
        "manifest_verified": verified,
        "registry_record": None if promotion is None else dict(promotion.get("records") or {}).get(
            "benchmark_manifest"
        ),
        "blocking_reasons": tuple(blocking_reasons),
    }


def _promotion_metadata(
    config: AdapterReadinessRegistryWorkflowConfig,
    readiness_report: Mapping[str, Any],
) -> dict[str, Any]:
    decision = dict(readiness_report.get("readiness_decision") or {})
    runtime = dict(readiness_report.get("runtime_recommendation") or {})
    runtime_config = dict(runtime.get("recommendation") or {})
    adapter_family = dict(readiness_report.get("adapter_family_matrix") or {})
    world_model_evidence = state_transition_world_model_evidence(adapter_family)
    best_quality_signal = dict(runtime_config.get("best_quality_signal") or {})
    score_fusion = dict(runtime_config.get("score_fusion") or {})
    metadata = {
        "workflow": "run_adapter_readiness_registry_workflow",
        "readiness_status": decision.get("status"),
        "adapter_family_status": decision.get("adapter_family_status"),
        "performance_status": decision.get("performance_status"),
        "runtime_recommendation_status": runtime.get("status"),
        "recommended_route": decision.get("recommended_route"),
        "recommended_performance_cell": decision.get("recommended_performance_cell"),
        "recommended_layer": runtime_config.get("layer"),
        "recommended_batch_size": runtime_config.get("batch_size"),
        "recommended_hidden_state_capture": runtime_config.get("hidden_state_capture"),
        "recommended_max_batch_tokens": runtime_config.get("max_batch_tokens"),
        "recommended_prefix_kv_cache": runtime_config.get("prefix_kv_cache"),
        "recommended_max_workers": runtime_config.get("max_workers"),
        "recommended_best_quality_signal": best_quality_signal.get("name"),
        "recommended_best_quality_auroc": best_quality_signal.get("auroc"),
        "recommended_quality_signals": runtime_config.get("quality_signals"),
        "recommended_score_fusion": score_fusion or None,
        "recommended_score_fusion_status": score_fusion.get("status"),
        "recommended_score_fusion_signal": score_fusion.get("signal_name"),
        "recommended_score_fusion_auroc": score_fusion.get("auroc"),
        "recommended_score_fusion_conformal_gate_passed": score_fusion.get("conformal_gate_passed"),
        "adapter_family_matrix_report": readiness_report.get("adapter_family_matrix_path"),
        "adapter_family_routes": tuple(adapter_family.get("routes") or ()),
        "adapter_family_retrieval_routes": tuple(adapter_family.get("retrieval_routes") or ()),
        "adapter_family_audit_routes": tuple(adapter_family.get("audit_routes") or ()),
        "adapter_include_retrieval": adapter_family.get(
            "include_retrieval",
            config.readiness.include_retrieval,
        ),
        "adapter_include_retrieval_structured_qa": adapter_family.get(
            "include_retrieval_structured_qa",
            config.readiness.include_retrieval_structured_qa,
        ),
        "adapter_include_triple_evidence": adapter_family.get(
            "include_triple_evidence",
            config.readiness.include_triple_evidence,
        ),
        "adapter_triple_min_slot_coverage": config.readiness.triple_min_slot_coverage,
        "adapter_family_state_transition_world_model_adapter": world_model_evidence.get(
            "world_model_adapter"
        ),
        "adapter_family_state_transition_world_model_rule_count": world_model_evidence.get(
            "world_model_rule_count"
        ),
        "adapter_family_state_transition_rule_based_world_model": world_model_evidence.get(
            "rule_based_world_model"
        ),
    }
    inside_sampling = dict(runtime_config.get("inside_sampling") or {})
    if inside_sampling:
        metadata.update({
            "recommended_inside_sampling": inside_sampling,
            "recommended_inside_sampling_run": inside_sampling.get("recommended_run"),
            "recommended_inside_sampling_total_generated_samples": inside_sampling.get(
                "total_generated_samples"
            ),
            "recommended_inside_sampling_sample_count_ratio_to_baseline": inside_sampling.get(
                "sample_count_ratio_to_baseline"
            ),
            "recommended_inside_generation_seconds": inside_sampling.get("inside_generation_seconds"),
            "recommended_inside_generation_seconds_ratio_to_baseline": inside_sampling.get(
                "inside_generation_seconds_ratio_to_baseline"
            ),
            "recommended_inside_sampling_stop_reason_counts": inside_sampling.get("stop_reason_counts"),
        })
    inside_trigger_budget = dict(runtime_config.get("inside_trigger_budget_sweep") or {})
    if inside_trigger_budget:
        metadata.update({
            "recommended_inside_trigger_budget_sweep": inside_trigger_budget,
            "recommended_inside_trigger_budget_id": inside_trigger_budget.get("recommended_budget_id"),
            "recommended_inside_trigger_budget_run": inside_trigger_budget.get("recommended_run"),
            "recommended_inside_trigger_budget_source": inside_trigger_budget.get("recommendation_source"),
            "recommended_inside_trigger_budget_policy": inside_trigger_budget.get("selection_policy"),
            "recommended_inside_trigger_budget_derive_from_max_budget": inside_trigger_budget.get(
                "derive_from_max_budget"
            ),
            "recommended_inside_trigger_budget_sample_count_ratio_to_reference": inside_trigger_budget.get(
                "sample_count_ratio_to_reference"
            ),
            "recommended_inside_trigger_budget_generation_seconds_ratio_to_reference": inside_trigger_budget.get(
                "inside_generation_seconds_ratio_to_reference"
            ),
        })
    if config.promotion_metadata is not None:
        metadata.update(dict(config.promotion_metadata))
    return metadata


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata entry must be key=value: {value!r}")
        key, text = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"metadata key must not be empty: {value!r}")
        metadata[key] = text
    return metadata


def _config_from_args(args: argparse.Namespace) -> AdapterReadinessRegistryWorkflowConfig:
    readiness = AdapterReadinessWorkflowConfig(
        output_dir=Path(args.output_dir),
        readiness_report_path=Path(args.readiness_json) if args.readiness_json else None,
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
        include_retrieval=bool(args.include_retrieval),
        include_retrieval_structured_qa=bool(args.include_retrieval_structured_qa),
        include_triple_evidence=bool(args.include_triple_evidence),
        verifier_min_overlap=args.verifier_min_overlap,
        retriever_min_overlap=args.retriever_min_overlap,
        retrieval_limit=args.retrieval_limit,
        triple_min_slot_coverage=args.triple_min_slot_coverage,
        model=args.model,
        dtype=args.dtype,
        layers=_parse_int_list(args.layers, flag="--layers"),
        batch_sizes=_parse_int_list(args.batch_sizes, flag="--batch-sizes"),
        hidden_state_captures=_parse_str_list(args.hidden_state_captures, flag="--hidden-state-captures"),
        limit=args.limit,
        manifold_questions=args.manifold_questions,
        max_length=args.max_length,
        max_batch_tokens=args.max_batch_tokens,
        max_batch_token_budgets=_parse_max_batch_token_budgets(args.max_batch_token_budgets),
        prefix_kv_cache=args.prefix_kv_cache,
        prefix_kv_cache_modes=_parse_prefix_kv_cache_modes(args.prefix_kv_cache_modes),
        eval_reps_cache_shard_size=args.eval_reps_cache_shard_size,
        cached_max_total_ratio=args.cached_max_total_ratio,
        cache_only_max_total_ratio=args.cache_only_max_total_ratio,
        python_executable=args.python,
        progress_every=args.progress_every,
        length_bucketed_batches=not args.no_length_bucketed_batches,
        offline=not args.real_truthfulqa,
        shared_cache_dir=Path(args.shared_cache_dir) if args.shared_cache_dir else None,
        matrix_mode=args.matrix_mode,
        performance_max_workers=args.performance_max_workers,
        performance_clean=bool(args.performance_clean),
        performance_dry_run=bool(args.performance_dry_run),
        max_runtime_total_seconds=args.max_runtime_total_seconds,
        inside_sampling_report_path=Path(args.inside_sampling_report) if args.inside_sampling_report else None,
        inside_trigger_budget_sweep_report_path=(
            Path(args.inside_trigger_budget_sweep_report)
            if args.inside_trigger_budget_sweep_report
            else None
        ),
        score_ensemble_report_path=Path(args.score_ensemble_report) if args.score_ensemble_report else None,
        inside_trigger_budget_policy=args.inside_trigger_budget_policy,
        performance_report_path=Path(args.performance_report) if args.performance_report else None,
    )
    return AdapterReadinessRegistryWorkflowConfig(
        readiness=readiness,
        registry_path=Path(args.registry),
        name=args.name,
        version=args.version,
        workflow_report_path=Path(args.json) if args.json else None,
        verification_report_path=Path(args.verification_report) if args.verification_report else None,
        promotion_metadata=_parse_metadata(args.metadata or ()),
        allow_non_promote=bool(args.allow_non_promote),
        allow_promotion_failures=bool(args.allow_promotion_failures),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = run_adapter_readiness_registry_workflow(_config_from_args(args))
    decision = payload["decision"]
    print(
        "adapter_readiness_registry="
        f"{decision['status']} "
        f"readiness={decision.get('readiness_status')} "
        f"record={decision.get('registry_record')}"
    )
    if args.fail_on_blocked and decision["status"] != "promote":
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run readiness gates and register the verified manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--json", default=None, help="optional registry workflow report path")
    parser.add_argument("--readiness-json", default=None, help="optional readiness report path")
    parser.add_argument("--verification-report", default=None)
    parser.add_argument("--metadata", action="append", default=[], help="extra promotion metadata as key=value")
    parser.add_argument("--allow-non-promote", action="store_true",
                        help="register even when readiness_decision.status is not promote")
    parser.add_argument("--allow-promotion-failures", action="store_true",
                        help="register even when manifest verification fails")
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
    parser.add_argument("--include-retrieval", action="store_true",
                        help="include the local retrieval-groundedness route in the adapter-family matrix")
    parser.add_argument("--include-retrieval-structured-qa", action="store_true",
                        help="include the local retrieval structured-QA route in the adapter-family matrix")
    parser.add_argument("--include-triple-evidence", action="store_true",
                        help="include the strict triple-evidence audit route in the adapter-family matrix")
    parser.add_argument("--verifier-min-overlap", type=float, default=0.65)
    parser.add_argument("--retriever-min-overlap", type=float, default=0.6)
    parser.add_argument("--retrieval-limit", type=int, default=1)
    parser.add_argument("--triple-min-slot-coverage", type=float, default=1.0)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--layers", default="-1")
    parser.add_argument("--batch-sizes", default="4")
    parser.add_argument("--hidden-state-captures", default="outputs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--manifold-questions", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=0)
    parser.add_argument("--max-batch-token-budgets", default=None)
    parser.add_argument("--prefix-kv-cache", action="store_true")
    parser.add_argument("--prefix-kv-cache-modes", default=None)
    parser.add_argument("--eval-reps-cache-shard-size", type=int, default=4)
    parser.add_argument("--cached-max-total-ratio", type=float, default=1.10)
    parser.add_argument("--cache-only-max-total-ratio", type=float, default=0.35)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-length-bucketed-batches", action="store_true")
    parser.add_argument("--real-truthfulqa", action="store_true")
    parser.add_argument("--shared-cache-dir", default=None)
    parser.add_argument("--matrix-mode", default="triplet", choices=MATRIX_MODES)
    parser.add_argument("--performance-max-workers", type=int, default=1)
    parser.add_argument("--performance-clean", action="store_true")
    parser.add_argument("--performance-dry-run", action="store_true")
    parser.add_argument("--max-runtime-total-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-runtime-total-seconds",
    ), default=None)
    parser.add_argument("--inside-sampling-report", default=None,
                        help="optional run_inside_sampling_profile.py comparison report for runtime recommendation")
    parser.add_argument("--inside-trigger-budget-sweep-report", default=None,
                        help="optional run_inside_trigger_budget_sweep.py report for runtime recommendation")
    parser.add_argument("--score-ensemble-report", default=None,
                        help="optional eval_score_ensemble.py report for runtime recommendation")
    parser.add_argument("--inside-trigger-budget-policy", default="quality_balanced",
                        choices=INSIDE_TRIGGER_BUDGET_POLICIES,
                        help="budget selection policy for trigger-budget sweep evidence")
    parser.add_argument("--performance-report", default=None,
                        help="reuse an existing cache-profile-matrix-report.json instead of rerunning profiles")
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless registry workflow decision is promote")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
